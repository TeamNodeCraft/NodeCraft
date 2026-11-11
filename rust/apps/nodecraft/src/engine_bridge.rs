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

#[derive(Clone)]
pub struct Evaluation {
    pub target: String,
    /// False for the exact-only pre-pass; true for a final exact result (for example an exporter).
    pub difference_error: Option<String>,
    /// The ghost came from the display-surface fallback, not the exact Boolean.
    pub difference_approximate: bool,
    pub node_errors: std::collections::BTreeMap<String, String>,
    pub error: Option<String>,
    pub elapsed_ms: f64,
    /// B-rep evaluation time for the viewed target, excluding tessellation.
    pub kernel_ms: f64,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub brep_hits: u32,
    pub part_hits: u32,
    /// Exact Part/BRep handles from the exact pass; empty after a display pass.
    pub exact_parts: std::collections::BTreeMap<String, nodecraft_exact::ExactPart>,
    /// Exact feature difference from the exact pass.
    pub exact_comparison: Option<nodecraft_engine::ExactComparison>,
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
    let expanded = match nodecraft_session::flatten_inline_graphs(document)
        .and_then(|grouped| nodecraft_session::expand_linked_parts_with(&grouped, linked_documents))
    {
        Ok(expanded) => expanded,
        Err(error) => {
            return (
                Some(Evaluation {
                    target,
                    difference_error: None,
                    difference_approximate: false,
                    node_errors: Default::default(),
                    error: Some(error.to_string()),
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    kernel_ms: 0.0,
                    cache_hits: 0,
                    cache_misses: 0,
                    brep_hits: 0,
                    part_hits: 0,
                    exact_parts: Default::default(),
                    exact_comparison: None,
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
        None,
    )
}

/// Display evaluation with a provisional preview detail. Used for coarse-first
/// display; the exact pass remains full quality.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_flat_with_progress_comparison_dirty_preview(
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
    preview_detail: Option<u8>,
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
        preview_detail,
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
        None,
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
    preview_detail: Option<u8>,
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
        preview_detail,
        comparison_targets: comparison.map(|(before, after)| vec![before.to_owned(), after.to_owned()]),
        bypass_cache,
    };
    let output = nodecraft_engine::evaluate_exact(request, cache, progress);
    let cache = output.cache;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let cache_stats = output.response.cache_stats;
    let exact_parts = output.exact_parts;
    let exact_comparison = output.exact_comparison;
    let (kernel_ms, _) = timing_breakdown(
        elapsed_ms,
        output.response.timings.get(&target).copied().unwrap_or(0.0),
        output.response.brep_timings.get(&target).copied().unwrap_or(0.0),
    );
    let error = output.response.node_errors.get(&target).cloned();
    let mut display_cache = nodecraft_display::PlanCache::new(1024);
    let quality = nodecraft_display::DisplayQuality::Full(0.35);
    let mut plans = Vec::new();
    let mut feature_plans = Vec::new();
    let mut tessellation_diagnostics = Vec::new();
    let mut display_ms = 0.0;
    let mesh = if exact_only {
        None
    } else {
        let started = Instant::now();
        let result = exact_parts.get(&target).and_then(|part| {
            nodecraft_display::mesh_part(part, quality, &mut display_cache).ok()
        });
        display_ms = started.elapsed().as_secs_f64() * 1000.0;
        result.map(|mut display| {
            for plan in &mut display.plans { plan.target.clone_from(&target); }
            plans = display.plans;
            feature_plans = display.feature_plans;
            tessellation_diagnostics.extend(display.stats.iter().flat_map(|stats| stats.diagnostics.iter().cloned()));
            display.mesh
        })
    };
    let edit_preview_mesh = if exact_only { None } else {
        edit_preview.and_then(|id| exact_parts.get(id)).and_then(|part| {
            nodecraft_display::mesh_part(part, quality, &mut display_cache).ok().map(|display| display.mesh)
        })
    };
    let edit_preview_target = edit_preview.map(str::to_owned);
    let (ghost_mesh, ghost_plans) = if exact_only { (None, Vec::new()) } else {
        ghost_input
            .and_then(|id| exact_parts.get(id))
            .and_then(|part| nodecraft_display::mesh_part(part, quality, &mut display_cache).ok())
            .map(|display| (Some(display.mesh), display.plans))
            .unwrap_or((None, Vec::new()))
    };
    let comparison = if exact_only { None } else {
        comparison.and_then(|(before, after)| {
            let before = exact_parts.get(before).and_then(|part| nodecraft_display::mesh_part(part, quality, &mut display_cache).ok())?.mesh;
            let after = exact_parts.get(after).and_then(|part| nodecraft_display::mesh_part(part, quality, &mut display_cache).ok())?.mesh;
            Some((before, after))
        })
    };
    let mut difference_plans = None;
    let difference = if exact_only { None } else {
        exact_comparison.as_ref().and_then(|comparison| {
            let added = comparison
                .added
                .as_ref()
                .and_then(|brep| nodecraft_display::brep_display(brep, quality, &mut display_cache).ok());
            let removed = comparison
                .removed
                .as_ref()
                .and_then(|brep| nodecraft_display::brep_display(brep, quality, &mut display_cache).ok());
            difference_plans = Some((
                added.as_ref().map(|display| display.plans.clone()).unwrap_or_default(),
                removed.as_ref().map(|display| display.plans.clone()).unwrap_or_default(),
            ));
            Some((
                added.map(|display| display.mesh).unwrap_or_default(),
                removed.map(|display| display.mesh).unwrap_or_default(),
            ))
        })
    };
    let difference_error = exact_comparison.as_ref().and_then(|comparison| comparison.error.clone());
    let pre_merge = if exact_only { None } else {
        pre_merge.and_then(|(a, b)| {
            let a_mesh = exact_parts.get(&a).and_then(|part| nodecraft_display::mesh_part(part, quality, &mut display_cache).ok())?.mesh;
            let b_mesh = exact_parts.get(&b).and_then(|part| nodecraft_display::mesh_part(part, quality, &mut display_cache).ok())?.mesh;
            Some(PreMerge { a: (a, a_mesh), b: (b, b_mesh) })
        })
    };
    let exact_parts = if exact_only { exact_parts } else { Default::default() };
    let exact_comparison = if exact_only { exact_comparison } else { None };
    (
        Some(Evaluation {
            target,
            // An errored node bypasses to its input, so the mesh can still be
            // valid even though the node reports its own error.
            difference_error,
            difference_approximate: false,
            node_errors: output.response.node_errors,
            error,
            elapsed_ms,
            kernel_ms,
            cache_hits: cache_stats.as_ref().map_or(0, |stats| stats.hits),
            cache_misses: cache_stats.as_ref().map_or(0, |stats| stats.misses),
            brep_hits: cache_stats.as_ref().map_or(0, |stats| stats.brep_hits),
            part_hits: cache_stats.as_ref().map_or(0, |stats| stats.part_hits),
            exact_parts,
            exact_comparison,
        }),
        cache,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_session::GraphDocument;

    fn node(id: &str) -> nodecraft_session::GraphNode {
        nodecraft_session::GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: "primitive".into(),
            values: serde_json::json!({"primitiveType": "box"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
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