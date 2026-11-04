//! The UI-to-engine boundary. NodeCraft's session/engine remain the authority.

use std::time::Instant;

use nodecraft_engine::EngineCache;
use nodecraft_protocol::{CacheState, Connection, Mesh, Node, Request};
use nodecraft_session::GraphDocument;

pub struct Evaluation {
    pub target: String,
    pub mesh: Option<Mesh>,
    /// Before/after pair when the selected feature can be compared.
    pub comparison: Option<(Mesh, Mesh)>,
    /// Exact before/after difference bodies from the kernel.
    pub difference: Option<(Mesh, Mesh)>,
    pub difference_error: Option<String>,
    pub node_errors: std::collections::BTreeMap<String, String>,
    pub error: Option<String>,
    pub elapsed_ms: f64,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

pub fn target_for(document: &GraphDocument, viewer_breakpoint: Option<&str>) -> Option<String> {
    // The viewer breakpoint is explicit: selection never changes it. Interface
    // outputs are the fallback when no breakpoint is pinned.
    if let Some(selected) = viewer_breakpoint {
        if document.nodes.iter().any(|node| node.id == selected) {
            return Some(selected.to_owned());
        }
    }
    if let Some(output) = document.interface.outputs.last() {
        if document.nodes.iter().any(|node| &node.id == output) {
            return Some(output.clone());
        }
    }
    document
        .nodes
        .iter()
        .rev()
        .find(|node| {
            node.node_type != "exporter"
                && !document.connections.iter().any(|link| link.from == node.id)
        })
        .or_else(|| document.nodes.last())
        .map(|node| node.id.clone())
}

pub fn evaluate(
    document: &GraphDocument,
    cache: EngineCache,
    selected: Option<&str>,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_with_linked(document, cache, selected, &std::collections::BTreeMap::new())
}

pub fn evaluate_with_linked(
    document: &GraphDocument,
    cache: EngineCache,
    selected: Option<&str>,
    linked_documents: &std::collections::BTreeMap<std::path::PathBuf, GraphDocument>,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_with_linked_progress(document, cache, selected, linked_documents, None)
}

pub fn evaluate_with_linked_progress(
    document: &GraphDocument,
    cache: EngineCache,
    selected: Option<&str>,
    linked_documents: &std::collections::BTreeMap<std::path::PathBuf, GraphDocument>,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_with_linked_progress_comparison(document, cache, selected, linked_documents, progress, None, false)
}

pub fn evaluate_with_linked_progress_comparison(
    document: &GraphDocument,
    cache: EngineCache,
    selected: Option<&str>,
    linked_documents: &std::collections::BTreeMap<std::path::PathBuf, GraphDocument>,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
) -> (Option<Evaluation>, EngineCache) {
    let Some(target) = target_for(document, selected) else {
        return (None, cache);
    };
    let started = Instant::now();
    let expanded = match nodecraft_session::expand_groups(document)
        .and_then(|grouped| nodecraft_session::expand_linked_parts_with(&grouped, linked_documents))
    {
        Ok(expanded) => expanded,
        Err(error) => {
            return (
                Some(Evaluation {
                    target,
                    mesh: None,
                    comparison: None,
                    difference: None,
                    difference_error: None,
                    node_errors: Default::default(),
                    error: Some(error.to_string()),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    cache_hits: 0,
                    cache_misses: 0,
                }),
                cache,
            );
        }
    };
    evaluate_flat_with_progress_comparison(&expanded, target, cache, progress, comparison, run_exporters)
}

/// Evaluate an already-flattened graph. Linked-document expansion is cached by
/// [`nodecraft_session::flat::FlatGraphCache`], so this function must not expand
/// again.
pub fn evaluate_flat_with_progress_comparison(
    document: &GraphDocument,
    target: String,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
) -> (Option<Evaluation>, EngineCache) {
    let started = Instant::now();
    let hashes = document.node_hashes().unwrap_or_default();
    let request = Request {
        request_id: 1,
        tab_id: None,
        graph_revision: None,
        targets: {
            let mut targets = vec![target.clone()];
            if let Some((before, after)) = comparison {
                for id in [before, after] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.to_owned());
                    }
                }
            }
            targets
        },
        mesh_targets: Some({
            let mut targets = vec![target.clone()];
            if let Some((before, after)) = comparison {
                for id in [before, after] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.to_owned());
                    }
                }
            }
            targets
        }),
        nodes: document
            .nodes
            .iter()
            .map(|node| Node {
                id: node.id.clone(),
                node_type: node.node_type.clone(),
                values: node.values.clone(),
                suppressed: node.suppressed,
                cache: Some(CacheState {
                    key: hashes.get(&node.id).map(|hash| hash.to_hex()),
                    state: Some("cached".into()),
                }),
            })
            .collect(),
        connections: document
            .connections
            .iter()
            .map(|link| Connection {
                from: link.from.clone(),
                to: link.to.clone(),
                input: link.input.clone(),
                out: link.out.clone(),
            })
            .collect(),
        binary_mesh: false,
        project_directory: None,
        progress: progress.is_some(),
        run_exporters,
        mesh_tolerance: None,
        comparison_targets: comparison.map(|(before, after)| vec![before.to_owned(), after.to_owned()]),
    };
    let output = nodecraft_engine::evaluate(request, cache, progress);
    let cache = output.cache;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cache_stats = output.response.cache_stats;
    let error = output.response.node_errors.get(&target).cloned();
    let mesh = output.response.meshes.get(&target).map(|payload| payload.mesh.clone());
    let comparison = comparison.and_then(|(before, after)| {
        let before = output.response.meshes.get(before)?.mesh.clone();
        let after = output.response.meshes.get(after)?.mesh.clone();
        Some((before, after))
    });
    (
        Some(Evaluation {
            target,
            mesh: if error.is_some() { None } else { mesh },
            comparison,
            difference: output.response.comparison.as_ref().map(|payload| (payload.added.clone(), payload.removed.clone())),
            difference_error: output.response.comparison.as_ref().and_then(|payload| payload.error.clone()),
            node_errors: output.response.node_errors,
            error,
            elapsed_ms,
            cache_hits: cache_stats.as_ref().map_or(0, |stats| stats.hits),
            cache_misses: cache_stats.as_ref().map_or(0, |stats| stats.misses),
        }),
        cache,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str) -> nodecraft_session::GraphNode {
        nodecraft_session::GraphNode {
            id: id.into(),
            node_type: "primitive".into(),
            values: json!({}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    /// Each hash map gets its own random seed, so any hash-order dependence in
    /// meshing shows up across repeated evaluations in one process.
    #[test]
    fn repeated_evaluation_produces_an_identical_mesh() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../samples/spline-showcase.nodecraft");
        let document: GraphDocument =
            serde_json::from_str(&std::fs::read_to_string(path).expect("sample")).expect("document");
        let mesh = || {
            let (evaluation, _) = evaluate(&document, EngineCache::new(), None);
            evaluation.and_then(|evaluation| evaluation.mesh).expect("mesh")
        };
        let first = mesh();
        for _ in 0..8 {
            let next = mesh();
            assert_eq!(next.vertices, first.vertices);
            assert_eq!(next.faces, first.faces);
        }
    }

    #[test]
    fn linked_part_can_use_an_unsaved_open_document() {
        let linked_path = std::env::temp_dir().join(format!(
            "nodecraft-linked-memory-{}-{}.nodecraft",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos())
        ));
        let mut linked = GraphDocument::new("part");
        let mut box_node = node("box");
        box_node.values = json!({"primitiveType": "box", "length": 10, "width": 10, "height": 10});
        linked.nodes.push(box_node);
        let mut host = GraphDocument::new("host");
        host.nodes.push(nodecraft_session::GraphNode {
            id: "link".into(),
            node_type: "linkedpart".into(),
            values: json!({ "path": linked_path.to_string_lossy() }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        let mut open = std::collections::BTreeMap::new();
        open.insert(linked_path, linked);
        let (evaluation, _) = evaluate_with_linked(&host, EngineCache::new(), Some("link"), &open);
        let evaluation = evaluation.expect("evaluation");
        assert!(evaluation.error.is_none(), "{:?}", evaluation.error);
        assert!(evaluation.mesh.is_some_and(|mesh| !mesh.faces.is_empty()));
    }

    #[test]
    fn linked_part_expands_through_the_app_bridge() {
        let dir = std::env::temp_dir().join(format!(
            "nodecraft-linked-bridge-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("dir");
        let linked_path = dir.join("part.nodecraft");
        let mut linked = GraphDocument::new("part");
        linked.nodes.push(node("box"));
        nodecraft_session::store::save_atomic(&linked_path, &linked).expect("linked save");

        let mut host = GraphDocument::new("host");
        host.nodes.push(nodecraft_session::GraphNode {
            id: "link".into(),
            node_type: "linkedpart".into(),
            values: json!({ "path": linked_path.to_string_lossy() }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        let (evaluation, _) = evaluate(&host, EngineCache::new(), Some("link"));
        let evaluation = evaluation.expect("evaluation");
        assert!(evaluation.error.is_none(), "{:?}", evaluation.error);
        assert!(evaluation.mesh.is_some_and(|mesh| !mesh.faces.is_empty()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selected_node_by_stable_id_is_a_target() {
        let mut document = GraphDocument::new("target-test");
        document.nodes = vec![node("a"), node("b")];
        assert_eq!(target_for(&document, Some("a")).as_deref(), Some("a"));
        assert_eq!(target_for(&document, Some("missing")).as_deref(), Some("b"));
        document.interface.outputs = vec!["b".into()];
        assert_eq!(target_for(&document, Some("a")).as_deref(), Some("a"));
        assert_eq!(target_for(&document, None).as_deref(), Some("b"));

        // The last node may be an upstream branch. The unnamed target must be
        // the terminal sink, not whichever node happens to be last in the array.
        let mut branching = GraphDocument::new("branching-target");
        branching.nodes = vec![node("a"), node("b"), node("c")];
        branching.connections = vec![
            nodecraft_session::GraphConnection {
                from: "a".into(),
                to: "b".into(),
                input: "a".into(),
                out: "shape".into(),
            },
            nodecraft_session::GraphConnection {
                from: "c".into(),
                to: "b".into(),
                input: "b".into(),
                out: "shape".into(),
            },
        ];
        assert_eq!(target_for(&branching, None).as_deref(), Some("b"));
    }
}
