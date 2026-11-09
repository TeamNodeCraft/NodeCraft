//! The UI-to-engine boundary. NodeCraft's session/engine remain the authority.

use std::collections::HashSet;
use std::time::Instant;

use nodecraft_engine::EngineCache;
use nodecraft_protocol::{CacheState, Connection, Mesh, Node, Request};
use nodecraft_session::GraphDocument;

/// The two input bodies of a Boolean target, before the merge.
///
/// The merged result has no per-operand display provenance. Keeping the
/// pre-merge bodies lets a transform drag show the edited branch while the
/// other operand stays put; the Boolean itself is re-evaluated on release.
#[derive(Clone)]
#[allow(dead_code)] // test-visible pre-merge provenance; production drags use PreviewWorker
pub struct PreMerge {
    pub a: (String, Mesh),
    pub b: (String, Mesh),
}

pub struct Evaluation {
    pub target: String,
    /// False for the exact-only pre-pass; true once display meshes/timing are final.
    pub display_complete: bool,
    pub mesh: Option<Mesh>,
    /// Display mesh for the node being edited, used by branch-scoped transform
    /// previews when the breakpoint result is geometrically merged.
    #[allow(dead_code)] // engine-bridge regression coverage; the app uses PreviewWorker
    pub edit_preview: Option<Mesh>,
    /// Node whose transform input produced `edit_preview`.
    #[allow(dead_code)]
    pub edit_preview_target: Option<String>,
    /// Boolean operands for a pre-merge drag display.
    #[allow(dead_code)]
    pub pre_merge: Option<PreMerge>,
    /// Before/after pair when the selected feature can be compared.
    pub comparison: Option<(Mesh, Mesh)>,
    /// Input body ghosted under a selected feature's result.
    pub ghost: Option<Mesh>,
    /// Exact before/after difference bodies from the kernel.
    pub difference: Option<(Mesh, Mesh)>,
    pub difference_error: Option<String>,
    /// The ghost came from the display-surface fallback, not the exact Boolean.
    pub difference_approximate: bool,
    pub node_errors: std::collections::BTreeMap<String, String>,
    pub error: Option<String>,
    pub elapsed_ms: f64,
    /// B-rep evaluation time for the viewed target, excluding tessellation.
    pub kernel_ms: f64,
    /// Display tessellation time for the viewed target.
    pub mesh_ms: f64,
    pub cache_hits: u32,
    pub cache_misses: u32,
}

/// Split the evaluation total into kernel and display work.
///
/// The engine's per-node timing includes tessellation; its B-rep timing is the
/// exact pass alone. The viewed node's difference is the display pass, and the
/// rest of the total is the B-rep work for that target's upstream nodes.
fn timing_breakdown(elapsed_ms: f64, target_ms: f64, target_brep_ms: f64) -> (f64, f64) {
    let mesh_ms = (target_ms - target_brep_ms).max(0.0);
    ((elapsed_ms - mesh_ms).max(0.0), mesh_ms)
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
                    display_complete: true,
                    mesh: None,
                    edit_preview: None,
                    edit_preview_target: None,
                    pre_merge: None,
                    comparison: None,
                    ghost: None,
                    difference: None,
                    difference_error: None,
                    difference_approximate: false,
                    node_errors: Default::default(),
                    error: Some(error.to_string()),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    kernel_ms: 0.0,
                    mesh_ms: 0.0,
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
/// Immediate a/b sources when `target` is a Boolean with both operands.
fn pre_merge_sources(document: &GraphDocument, target: &str) -> Option<(String, String)> {
    let node = document.nodes.iter().find(|node| node.id == target)?;
    if !matches!(node.node_type.as_str(), "boolean" | "fuse") {
        return None;
    }
    let source = |input: &str| {
        document
            .connections
            .iter()
            .find(|link| link.to == target && link.input == input)
            .map(|link| link.from.clone())
    };
    Some((source("a")?, source("b")?))
}

pub fn evaluate_flat_with_progress_comparison(
    document: &GraphDocument,
    target: String,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_flat_with_progress_comparison_dirty(
        document,
        target,
        cache,
        progress,
        comparison,
        run_exporters,
        &HashSet::new(),
        None,
        &[],
        None,
        false,
    )
}

pub fn evaluate_flat_with_progress_comparison_dirty(
    document: &GraphDocument,
    target: String,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
    dirty_nodes: &HashSet<String>,
    ghost_input: Option<&str>,
    validation_targets: &[String],
    edit_preview: Option<&str>,
    bypass_cache: bool,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_flat_with_progress_comparison_dirty_mode(
        document,
        target,
        cache,
        progress,
        comparison,
        run_exporters,
        dirty_nodes,
        ghost_input,
        validation_targets,
        edit_preview,
        false,
        bypass_cache,
    )
}

/// Exact-only pre-pass for the display scheduler. It fills the Part cache but
/// produces no display meshes, so the app can publish errors/status before the
/// final display evaluation.
pub fn evaluate_flat_exact(
    document: &GraphDocument,
    target: String,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
    dirty_nodes: &HashSet<String>,
    ghost_input: Option<&str>,
    validation_targets: &[String],
    edit_preview: Option<&str>,
    bypass_cache: bool,
) -> (Option<Evaluation>, EngineCache) {
    evaluate_flat_with_progress_comparison_dirty_mode(
        document,
        target,
        cache,
        progress,
        comparison,
        run_exporters,
        dirty_nodes,
        ghost_input,
        validation_targets,
        edit_preview,
        true,
        bypass_cache,
    )
}

fn evaluate_flat_with_progress_comparison_dirty_mode(
    document: &GraphDocument,
    target: String,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(nodecraft_engine::ProgressEvent)>,
    comparison: Option<(&str, &str)>,
    run_exporters: bool,
    dirty_nodes: &HashSet<String>,
    ghost_input: Option<&str>,
    validation_targets: &[String],
    edit_preview: Option<&str>,
    exact_only: bool,
    bypass_cache: bool,
) -> (Option<Evaluation>, EngineCache) {
    let started = Instant::now();
    let hashes = document.node_hashes().unwrap_or_default();
    let pre_merge = pre_merge_sources(document, &target);
    let request = Request {
        request_id: 1,
        tab_id: None,
        graph_revision: None,
        targets: {
            let mut targets = vec![target.clone()];
            if let Some((a, b)) = &pre_merge {
                for id in [a, b] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.clone());
                    }
                }
            }
            if let Some((before, after)) = comparison {
                for id in [before, after] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.to_owned());
                    }
                }
            }
            if let Some(ghost) = ghost_input
                && !targets.iter().any(|existing| existing == ghost)
            {
                targets.push(ghost.to_owned());
            }
            if let Some(preview) = edit_preview
                && !targets.iter().any(|existing| existing == preview)
            {
                targets.push(preview.to_owned());
            }
            for id in validation_targets {
                if !targets.iter().any(|existing| existing == id) {
                    targets.push(id.clone());
                }
            }
            targets
        },
        mesh_targets: Some({
            let mut targets = vec![target.clone()];
            if let Some((a, b)) = &pre_merge {
                for id in [a, b] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.clone());
                    }
                }
            }
            if let Some((before, after)) = comparison {
                for id in [before, after] {
                    if !targets.iter().any(|existing| existing == id) {
                        targets.push(id.to_owned());
                    }
                }
            }
            if let Some(ghost) = ghost_input
                && !targets.iter().any(|existing| existing == ghost)
            {
                targets.push(ghost.to_owned());
            }
            if let Some(preview) = edit_preview
                && !targets.iter().any(|existing| existing == preview)
            {
                targets.push(preview.to_owned());
            }
            for id in validation_targets {
                if !targets.iter().any(|existing| existing == id) {
                    targets.push(id.clone());
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
                    state: Some(if dirty_nodes.contains(&node.id) {
                        "dirty".into()
                    } else {
                        "cached".into()
                    }),
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
        bypass_cache,
    };
    let output = if exact_only {
        nodecraft_engine::evaluate_exact(request, cache, progress)
    } else {
        nodecraft_engine::evaluate(request, cache, progress)
    };
    let cache = output.cache;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    // The engine records the viewed node's full time and its exact B-rep pass;
    // their difference is display tessellation. Everything else in the total is
    // the kernel work for the target's upstream nodes.
    let (kernel_ms, mesh_ms) = timing_breakdown(
        elapsed_ms,
        output.response.timings.get(&target).copied().unwrap_or(0.0),
        output.response.brep_timings.get(&target).copied().unwrap_or(0.0),
    );
    let cache_stats = output.response.cache_stats;
    let error = output.response.node_errors.get(&target).cloned();
    let mesh = output.response.meshes.get(&target).map(|payload| payload.mesh.clone());
    let edit_preview_target = edit_preview.map(str::to_owned);
    let edit_preview = edit_preview_target
        .as_deref()
        .and_then(|id| output.response.meshes.get(id))
        .map(|payload| payload.mesh.clone());
    let pre_merge = pre_merge.and_then(|(a, b)| {
        let a_mesh = output.response.meshes.get(&a)?.mesh.clone();
        let b_mesh = output.response.meshes.get(&b)?.mesh.clone();
        Some(PreMerge { a: (a, a_mesh), b: (b, b_mesh) })
    });
    let comparison = comparison.and_then(|(before, after)| {
        let before = output.response.meshes.get(before)?.mesh.clone();
        let after = output.response.meshes.get(after)?.mesh.clone();
        Some((before, after))
    });
    let ghost = ghost_input.and_then(|id| {
        output
            .response
            .meshes
            .get(id)
            .map(|payload| payload.mesh.clone())
    });
    (
        Some(Evaluation {
            target,
            display_complete: !exact_only,
            // An errored node bypasses to its input, so the mesh can still be
            // valid even though the node reports its own error.
            mesh,
            edit_preview,
            edit_preview_target,
            pre_merge,
            comparison,
            ghost,
            difference: output.response.comparison.as_ref().map(|payload| (payload.added.clone(), payload.removed.clone())),
            difference_error: output.response.comparison.as_ref().and_then(|payload| payload.error.clone()),
            difference_approximate: output.response.comparison.as_ref().is_some_and(|payload| payload.approximate),
            node_errors: output.response.node_errors,
            error,
            elapsed_ms,
            kernel_ms,
            mesh_ms,
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
        nodecraft_session::GraphNode { expressions: Default::default(),
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
    fn timing_breakdown_separates_kernel_and_meshing() {
        let (kernel, mesh) = timing_breakdown(180.0, 70.0, 20.0);
        assert!((kernel - 130.0).abs() < 1e-9);
        assert!((mesh - 50.0).abs() < 1e-9);
        assert!(((kernel + mesh) - 180.0).abs() < 1e-9);
        // A cached exact part has no B-rep time; the node time is all mesh.
        assert_eq!(timing_breakdown(20.0, 20.0, 0.0), (0.0, 20.0));
        assert_eq!(timing_breakdown(10.0, 0.0, 0.0), (10.0, 0.0));
    }

    #[test]
    fn ghost_input_mesh_is_returned_with_the_target() {
        let mut document = GraphDocument::new("ghost-preview");
        let mut target = node("target");
        target.values = json!({"primitiveType": "box"});
        let mut ghost = node("ghost");
        ghost.values = json!({"primitiveType": "sphere"});
        document.nodes = vec![target, ghost];
        let (evaluation, _) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "target".to_owned(),
            EngineCache::new(),
            None,
            None,
            false,
            &HashSet::new(),
            Some("ghost"),
            &[],
            None,
            false,
        );
        let evaluation = evaluation.expect("evaluation");
        assert!(evaluation.mesh.as_ref().is_some_and(|mesh| !mesh.faces.is_empty()));
        let ghost = evaluation.ghost.expect("ghost input mesh");
        assert!(!ghost.faces.is_empty(), "ghost input has no display mesh");
        assert_ne!(evaluation.mesh.unwrap().faces.len(), ghost.faces.len());
    }

    #[test]
    fn bypass_cache_forces_a_full_rebuild() {
        let mut document = GraphDocument::new("bypass-cache");
        let mut target = node("target");
        target.values = json!({"primitiveType": "box"});
        document.nodes = vec![target];
        let (first, cache) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "target".to_owned(),
            EngineCache::new(),
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &[],
            None,
            false,
        );
        assert!(first.is_some());
        let (second, _) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "target".to_owned(),
            cache,
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &[],
            None,
            true,
        );
        let second = second.expect("bypass evaluation");
        assert!(second.mesh.as_ref().is_some_and(|mesh| !mesh.faces.is_empty()));
        assert_eq!(second.cache_hits, 0, "bypass still read a cache entry");
    }

    #[test]
    fn exact_prepass_fills_the_cache_before_display() {
        let mut document = GraphDocument::new("two-phase");
        let mut target = node("target");
        target.values = json!({"primitiveType": "box"});
        document.nodes = vec![target];
        let (exact, cache) = evaluate_flat_exact(
            &document,
            "target".to_owned(),
            EngineCache::new(),
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &[],
            None,
            false,
        );
        let exact = exact.expect("exact evaluation");
        assert!(!exact.display_complete);
        assert!(exact.mesh.is_none(), "exact pre-pass produced a display mesh");
        let (display, _) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "target".to_owned(),
            cache,
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &[],
            None,
            false,
        );
        let display = display.expect("display evaluation");
        assert!(display.display_complete);
        assert!(display.mesh.as_ref().is_some_and(|mesh| !mesh.faces.is_empty()));
    }

    #[test]
    fn edit_preview_mesh_is_returned_for_a_merged_breakpoint() {
        let document: GraphDocument = serde_json::from_value(json!({
            "id": "edit-preview",
            "name": "edit-preview",
            "nodes": [
                {"id": "box", "type": "solid", "values": {"primitiveType": "box", "width": 30.0}, "suppressed": false, "x": 0.0, "y": 0.0},
                {"id": "move", "type": "translate", "values": {"x": -24.0}, "suppressed": false, "x": 0.0, "y": 0.0},
                {"id": "sphere", "type": "solid", "values": {"primitiveType": "sphere"}, "suppressed": false, "x": 0.0, "y": 0.0},
                {"id": "boolean", "type": "boolean", "values": {"mode": "union"}, "suppressed": false, "x": 0.0, "y": 0.0},
                {"id": "fillet", "type": "fillet", "values": {}, "suppressed": false, "x": 0.0, "y": 0.0},
                {"id": "look", "type": "look", "values": {}, "suppressed": false, "x": 0.0, "y": 0.0}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "boolean", "in": "a", "out": "shape"},
                {"from": "sphere", "to": "boolean", "in": "b", "out": "shape"},
                {"from": "boolean", "to": "fillet", "in": "shape", "out": "shape"},
                {"from": "fillet", "to": "look", "in": "shape", "out": "shape"}
            ]
        })).expect("document");
        let (evaluation, _) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "look".to_owned(),
            EngineCache::new(),
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &[],
            Some("box"),
            false,
        );
        let evaluation = evaluation.expect("evaluation");
        let preview = evaluation.edit_preview.expect("edit preview mesh");
        assert!(!preview.faces.is_empty(), "edit preview mesh is empty");
    }

    #[test]
    fn validation_targets_report_their_node_errors() {
        let mut document = GraphDocument::new("validation-errors");
        let mut target = node("target");
        target.values = json!({"primitiveType": "box"});
        let mut broken = node("broken");
        broken.node_type = "shell".into();
        broken.values = json!({"thickness": 2.0});
        document.nodes = vec![target, broken];
        let (evaluation, _) = evaluate_flat_with_progress_comparison_dirty(
            &document,
            "target".to_owned(),
            EngineCache::new(),
            None,
            None,
            false,
            &HashSet::new(),
            None,
            &["broken".to_owned()],
            None,
            false,
        );
        let evaluation = evaluation.expect("evaluation");
        assert!(evaluation.error.is_none(), "{:?}", evaluation.error);
        assert!(
            evaluation.node_errors.contains_key("broken"),
            "validation node error missing: {:?}",
            evaluation.node_errors
        );
    }

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
    fn boolean_evaluation_returns_the_pre_merge_bodies() {
        let mut document = GraphDocument::new("boolean-pre-merge");
        let mut a = node("a");
        a.values = json!({"primitiveType": "box"});
        let mut b = node("b");
        b.values = json!({"primitiveType": "sphere"});
        let mut boolean = node("boolean");
        boolean.node_type = "boolean".into();
        boolean.values = json!({"mode": "union"});
        document.nodes = vec![a, b, boolean];
        document.connections = vec![
            nodecraft_session::GraphConnection { from: "a".into(), to: "boolean".into(), input: "a".into(), out: "shape".into() },
            nodecraft_session::GraphConnection { from: "b".into(), to: "boolean".into(), input: "b".into(), out: "shape".into() },
        ];
        let (evaluation, _) = evaluate(&document, EngineCache::new(), None);
        let evaluation = evaluation.expect("evaluation");
        let pre_merge = evaluation.pre_merge.expect("pre-merge bodies");
        assert_eq!(pre_merge.a.0, "a");
        assert_eq!(pre_merge.b.0, "b");
        assert!(!pre_merge.a.1.faces.is_empty());
        assert!(!pre_merge.b.1.faces.is_empty());
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
        host.nodes.push(nodecraft_session::GraphNode { expressions: Default::default(),
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
        host.nodes.push(nodecraft_session::GraphNode { expressions: Default::default(),
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
