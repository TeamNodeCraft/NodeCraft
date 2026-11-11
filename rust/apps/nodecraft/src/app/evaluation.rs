use super::*;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use nodecraft_engine::EngineCache;
use nodecraft_session::GraphDocument;

fn map_eval_errors(
    binding: Option<&EvalBinding>,
    errors: &std::collections::BTreeMap<String, String>,
) -> std::collections::BTreeMap<String, String> {
    match binding {
        Some(binding) => errors
            .iter()
            .filter_map(|(id, message)| {
                map_eval_id_to_body(binding, id).map(|id| (id, message.clone()))
            })
            .collect(),
        None => errors.clone(),
    }
}


impl NodeCraftApp {
    pub(super) fn evaluate(&mut self) {
        self.evaluate_with_cache_policy(false, false);
    }

    fn linked_documents(&self) -> std::collections::BTreeMap<PathBuf, GraphDocument> {
        let mut documents = std::collections::BTreeMap::new();
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.document().nodes.iter().any(|node| node.node_type == "linkedpart"))
        {
            for (document_id, document) in self.documents.documents() {
                if let Some(path) = self.documents.path(document_id) {
                    documents.insert(path.to_path_buf(), document.clone());
                }
            }
        }
        documents
    }

    pub(super) fn evaluate_with_cache_policy(&mut self, clear_cache: bool, run_exporters: bool) {
        self.auto_eval_due = None;
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        if session.document().nodes.is_empty() {
            self.status = "Document has no nodes".to_owned();
            self.eval_pending = None;
            self.finish_remaining_node_activity();
            return;
        }
        let binding = self.active_eval_binding();
        let eval_document = binding
            .as_ref()
            .map(|binding| &binding.document)
            .unwrap_or_else(|| session.document());
        // Edge editing keeps the displayed breakpoint as the base and picks
        // from the transform-edit ghost. Other selection modes use the
        // downstream view target when one is set.
        let selection_mode = self.viewport.selection_mode();
        let raw_evaluation_selection = if selection_mode == crate::viewport::SelectionMode::Edge {
            self.selection_view_target
                .clone()
                .or_else(|| self.viewer_breakpoint.clone())
                .or_else(|| self.eval_target.clone())
                .or_else(|| self.face_pick_target.clone())
        } else {
            self.selection_view_target
                .clone()
                .or_else(|| self.face_pick_target.clone())
                .or_else(|| self.viewer_breakpoint.clone())
        };
        let evaluation_selection = match (&binding, raw_evaluation_selection) {
            (Some(binding), Some(id)) => binding
                .body
                .nodes
                .iter()
                .find(|node| node.id == id)
                .and_then(|_| map_body_id_to_eval(binding, &id))
                .or_else(|| Some(binding.prefix.clone())),
            (Some(binding), None) => Some(binding.prefix.clone()),
            (None, raw) => raw,
        };
        let selected_eval = self.selected_nodes.iter().next().and_then(|selected| {
            match &binding {
                Some(binding) => map_body_id_to_eval(binding, selected),
                None => Some(selected.clone()),
            }
        });
        let comparison = if self.selected_nodes.len() == 1 {
            comparison_for_selection(
                eval_document,
                selected_eval.as_deref(),
                self.feature_overlay,
            )
        } else {
            None
        };
        let ghost_input = if self.face_pick_target.is_none()
            && self.feature_overlay
            && self.selected_nodes.len() == 1
        {
            selected_eval
                .as_deref()
                .and_then(|selected| feature_ghost_input(eval_document, selected))
        } else {
            None
        };
        // The main exact pass evaluates the edited node's input so the display
        // service can plan its ghost directly; there is no second exact worker.
        let edit_preview: Option<String> = self.edit_target.as_ref().and_then(|target| {
            let target = binding
                .as_ref()
                .and_then(|binding| map_body_id_to_eval(binding, target))
                .unwrap_or_else(|| target.clone());
            edit_preview_input(eval_document, &target)
        });
        let dirty_nodes: std::collections::HashSet<String> = self
            .pending_dirty
            .iter()
            .filter_map(|id| {
                binding
                    .as_ref()
                    .and_then(|binding| map_body_id_to_eval(binding, id))
                    .or_else(|| Some(id.clone()))
            })
            .collect();
        self.pending_dirty.clear();
        // A selected node outside the viewed target closure is evaluated as an
        // extra validation target. It contributes node_errors (and therefore the
        // canvas error glow) without changing the displayed mesh or breakpoint.
        let viewed_closure = engine_bridge::target_for(
            eval_document,
            evaluation_selection.as_deref(),
        )
        .map(|target| target_closure(eval_document, &target))
        .unwrap_or_default();
        let mut validation_targets: Vec<String> = self
            .selected_nodes
            .iter()
            .filter_map(|id| {
                binding
                    .as_ref()
                    .and_then(|binding| map_body_id_to_eval(binding, id))
                    .or_else(|| Some(id.clone()))
            })
            .filter(|id| !viewed_closure.contains(id))
            .collect();
        validation_targets.sort();
        validation_targets.dedup();
        // Sync helpers can run every frame. Evaluation is a pure function of the
        // graph revision, viewed target and comparison pair; skip identical work
        // before allocating a new pending id.
        let eval_document_id = binding.as_ref().map_or_else(
            || session.document().id.clone(),
            |binding| format!("{}::{}", binding.source_document_id, binding.prefix),
        );
        let key = eval_key(
            self.session_generation,
            &eval_document_id,
            session.revision(),
            &evaluation_selection,
            &comparison,
            &ghost_input,
            &edit_preview,
            &validation_targets,
            run_exporters,
            self.preview_detail,
        );
        if !clear_cache && self.last_eval_key.as_ref() == Some(&key) {
            return;
        }
        self.last_eval_key = Some(key);
        self.eval_next = self.eval_next.wrapping_add(1);
        let id = self.eval_next;
        self.eval_pending = Some(id);
        let tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let total = engine_bridge::target_for(eval_document, evaluation_selection.as_deref())
            .map(|target| target_closure(eval_document, &target).len())
            .unwrap_or(0);
        self.eval_progress = (total > 0).then_some(EvalProgress {
            total,
            exact_done: 0,
        });
        self.exact_brep_hits = 0;
        self.exact_part_hits = 0;
        self.brep_hits = 0;
        self.part_hits = 0;
        self.display_mesh_hits = 0;
        self.node_editor.clear_activity();
        self.activity_seen.clear();
        self.activity_active.clear();
        self.status = "Evaluating…".to_owned();
        let linked_documents = self.linked_documents();
        self.comparison_pair = comparison.clone();
        let document_id = self
            .tabs
            .get(self.active_tab)
            .map(|tab| {
                self.group_tabs
                    .get(&tab.id)
                    .map(|meta| meta.view.document_id.clone())
                    .unwrap_or_else(|| tab.document_id.clone())
            })
            .unwrap_or_default();
        let feature_ghost = self.selected_nodes.iter().any(|id| {
            eval_document
                .nodes
                .iter()
                .find(|node| &node.id == id)
                .is_some_and(|node| matches!(node.node_type.as_str(), "fillet" | "chamfer"))
        });
        self.last_exact = None;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {id}] app sent");
        }
        self.eval_worker.send(EvalJob {
            id,
            tab_id,
            document_id,
            linked_documents,
            document: eval_document.clone(),
            selected: evaluation_selection,
            comparison,
            ghost_input,
            edit_preview,
            validation_targets,
            clear_cache,
            run_exporters,
            preview_detail: self.preview_detail,
            feature_ghost,
            dirty_nodes,
            token: nodecraft_engine::kernel_pool::RevisionToken::new(),
        });
        self.egui_ctx.request_repaint();
    }

}

pub(super) struct EvalJob {
    id: u64,
    tab_id: u64,
    document_id: String,
    linked_documents: std::collections::BTreeMap<PathBuf, GraphDocument>,
    document: GraphDocument,
    selected: Option<String>,
    comparison: Option<(String, String)>,
    ghost_input: Option<String>,
    edit_preview: Option<String>,
    validation_targets: Vec<String>,
    clear_cache: bool,
    run_exporters: bool,
    preview_detail: Option<u8>,
    /// Build fillet/chamfer blend-face plans for the selected feature ghost.
    feature_ghost: bool,
    dirty_nodes: std::collections::HashSet<String>,
    token: nodecraft_engine::kernel_pool::RevisionToken,
}

#[derive(Clone, PartialEq)]
struct ExactKey {
    document: GraphDocument,
    linked_documents: std::collections::BTreeMap<PathBuf, GraphDocument>,
    selected: Option<String>,
    comparison: Option<(String, String)>,
    ghost_input: Option<String>,
    edit_preview: Option<String>,
    feature_ghost: bool,
    validation_targets: Vec<String>,
    dirty_nodes: std::collections::HashSet<String>,
    run_exporters: bool,
}

impl ExactKey {
    fn of(job: &EvalJob) -> Self {
        Self {
            document: job.document.clone(),
            linked_documents: job.linked_documents.clone(),
            selected: job.selected.clone(),
            comparison: job.comparison.clone(),
            ghost_input: job.ghost_input.clone(),
            edit_preview: job.edit_preview.clone(),
            feature_ghost: job.feature_ghost,
            validation_targets: job.validation_targets.clone(),
            dirty_nodes: job.dirty_nodes.clone(),
            run_exporters: job.run_exporters,
        }
    }
}

#[derive(Default)]
struct TabEvalState {
    cache: EngineCache,
    flat: nodecraft_session::flat::FlatGraphCache,
    /// Exact Parts for the last graph request. Display-quality changes (camera
    /// resize, LOD tier) reuse this instead of walking the DAG again.
    exact: Option<(ExactKey, engine_bridge::Evaluation)>,
}

/// Exact geometry for one display request, owned by the display side once it
/// leaves the exact worker.
#[derive(Clone)]
pub(super) struct DisplaySnapshot {
    pub id: u64,
    pub target: String,
    pub feature_ghost: bool,
    pub snapshot: std::sync::Arc<nodecraft_display::service::ExactSnapshot>,
}

pub(super) struct EvalWorker {
    tx: Sender<EvalJob>,
    rx: Receiver<(u64, u64, engine_bridge::Evaluation)>,
    progress_rx: Receiver<(u64, u64, nodecraft_engine::ProgressEvent)>,
    snapshot_rx: Receiver<(u64, u64, DisplaySnapshot)>,
    active_token: std::sync::Arc<std::sync::Mutex<Option<nodecraft_engine::kernel_pool::RevisionToken>>>,
}

impl EvalWorker {
    pub(super) fn new(display_worker: std::sync::Arc<nodecraft_display::service::DisplayWorker>) -> Self {
        let (job_tx, job_rx) = channel::<EvalJob>();
        let (result_tx, result_rx) = channel();
        let (progress_tx, progress_rx) = channel();
        let (snapshot_tx, snapshot_rx) = channel::<(u64, u64, DisplaySnapshot)>();
        std::thread::spawn(move || {
            // Cache ownership follows the tab/document, not the worker. This keeps
            // per-document dirty/cache lifetimes aligned and prevents one tab from
            // evicting or reusing another tab's entries.
            let mut states: std::collections::HashMap<String, TabEvalState> =
                std::collections::HashMap::new();
            let mut pending_job: Option<EvalJob> = None;
            'jobs: loop {
                let mut job = match pending_job.take() {
                    Some(job) => job,
                    None => match job_rx.recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    },
                };
                // Only the newest request matters; discard queued stale work.
                // A clear request must not be lost because a newer edit coalesced
                // into it: keep the flag and clear before the newest evaluation.
                let mut clear_cache = job.clear_cache;
                while let Ok(next) = job_rx.try_recv() {
                    clear_cache |= next.clear_cache;
                    job = next;
                }
                job.clear_cache = clear_cache;
                let mut state = states.remove(&job.document_id).unwrap_or_default();
                let timeline = std::time::Instant::now();
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[t {}] worker picked", job.id);
                }
                nodecraft_engine::kernel_pool::set_current_revision_token(job.token.clone());
                // `clear_cache` means "bypass cache lookups for this run", not
                // "free every cache". The evaluator already keys entries by
                // content hash, so changed graphs cannot hit stale results; the
                // bypass is only for the explicit fresh re-evaluate action.
                let flat = match state.flat.resolve(&job.document, &job.linked_documents) {
                    Ok(flat) => flat.clone(),
                    Err(error) => {
                        let target = engine_bridge::target_for(&job.document, job.selected.as_deref());
                        let evaluation = target.map(|target| engine_bridge::Evaluation {
                            target,
                            difference_error: None,
                            difference_approximate: false,
                            node_errors: Default::default(),
                            error: Some(error.to_string()),
                            elapsed_ms: 0.0,
                            kernel_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                            brep_hits: 0,
                            part_hits: 0,
                            exact_parts: Default::default(),
                            exact_comparison: None,
                        });
                        states.insert(job.document_id.clone(), state);
                        if let Some(evaluation) = evaluation
                            && result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                        continue;
                    }
                };
                let target = engine_bridge::target_for(flat.document(), job.selected.as_deref());
                let progress_tx = progress_tx.clone();
                let mut on_progress = |event: nodecraft_engine::ProgressEvent| {
                    let _ = progress_tx.send((job.id, job.tab_id, event));
                };
                let mut cache = state.cache;
                let mut exact_for_display: Option<engine_bridge::Evaluation> = None;
                let exact_key = ExactKey::of(&job);
                let exact_started = std::time::Instant::now();
                if exact_for_display.is_none()
                    && !job.clear_cache
                    && let Some((key, exact)) = state.exact.as_ref()
                    && key == &exact_key
                {
                    // Reused exact Parts were already measured on the evaluation
                    // that produced them. Report only this display pass.
                    let mut reused = exact.clone();
                    reused.elapsed_ms = 0.0;
                    reused.kernel_ms = 0.0;
                    reused.cache_hits = 0;
                    reused.cache_misses = 0;
                    reused.brep_hits = 0;
                    reused.part_hits = 0;
                    exact_for_display = Some(reused);
                }
                if exact_for_display.is_none() {
                    let exact_outcome = catch_unwind(AssertUnwindSafe(|| {
                        engine_bridge::evaluate_flat_exact(
                            flat.document(),
                            target.clone().unwrap_or_default(),
                            cache,
                            Some(&mut on_progress),
                            job.comparison.as_ref().map(|(before, after)| (before.as_str(), after.as_str())),
                            job.run_exporters,
                            &job.dirty_nodes,
                            job.ghost_input.as_deref(),
                            &job.validation_targets,
                            job.edit_preview.as_deref(),
                            job.clear_cache,
                        )
                    }));
                    let exact = match exact_outcome {
                        Ok((evaluation, next_cache)) => {
                            cache = next_cache;
                            evaluation
                        }
                        Err(_) => {
                            cache = EngineCache::new();
                            let evaluation = target.map(|target| engine_bridge::Evaluation {
                                target,
                                difference_error: None,
                                difference_approximate: false,
                                node_errors: Default::default(),
                                error: Some("Engine panicked during exact evaluation".into()),
                                elapsed_ms: 0.0,
                                kernel_ms: 0.0,
                                cache_hits: 0,
                                cache_misses: 0,
                                brep_hits: 0,
                                part_hits: 0,
                            exact_parts: Default::default(),
                            exact_comparison: None,
                            });
                            state.cache = cache;
                            if let Some(evaluation) = evaluation
                                && result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                            states.insert(job.document_id.clone(), state);
                            continue;
                        }
                    };
                    if let Some(exact) = exact.as_ref() {
                        state.exact = Some((exact_key.clone(), exact.clone()));
                    }
                    if std::env::var_os("NODECRAFT_TIMELINE").is_some()
                        && let Some(exact) = exact.as_ref()
                    {
                        eprintln!(
                            "[t {}] exact {:?} hits {} misses {} parts {}",
                            job.id,
                            exact_started.elapsed(),
                            exact.cache_hits,
                            exact.cache_misses,
                            exact.exact_parts.len(),
                        );
                    }
                    exact_for_display = exact.clone();
                    let evaluation_target = target.clone().unwrap_or_default();
                    let service_display = exact_for_display.as_ref().is_some_and(|exact| {
                        if job.run_exporters {
                            return false;
                        }
                        // Missing overlays must not remove the breakpoint. The
                        // snapshot builder includes each overlay only when its
                        // Part exists; only a missing target means there is
                        // nothing to display.
                        exact.exact_parts.contains_key(&evaluation_target)
                    });
                    if service_display {
                        let exact = exact_for_display.as_ref().expect("service exact");
                        let mut overlays = Vec::new();
                        if let Some(edit) = job.edit_preview.as_ref()
                            && let Some(part) = exact.exact_parts.get(edit)
                        {
                            overlays.push(nodecraft_display::service::OverlayRequest {
                                kind: nodecraft_display::service::OverlayKind::EditInput,
                                part: std::sync::Arc::new(part.clone()),
                            });
                        }
                        if let Some(ghost) = job.ghost_input.as_ref()
                            && let Some(part) = exact.exact_parts.get(ghost)
                        {
                            overlays.push(nodecraft_display::service::OverlayRequest {
                                kind: nodecraft_display::service::OverlayKind::FeatureGhost,
                                part: std::sync::Arc::new(part.clone()),
                            });
                        }
                        if let Some((before, after)) = job.comparison.as_ref() {
                            for (id, kind) in [
                                (before, nodecraft_display::service::OverlayKind::ComparisonBefore),
                                (after, nodecraft_display::service::OverlayKind::ComparisonAfter),
                            ] {
                                if let Some(part) = exact.exact_parts.get(id) {
                                    overlays.push(nodecraft_display::service::OverlayRequest {
                                        kind,
                                        part: std::sync::Arc::new(part.clone()),
                                    });
                                }
                            }
                        }
                        if let Some(difference) = exact.exact_comparison.as_ref() {
                            if let (Some(added), Some(key)) = (difference.added.as_ref(), difference.added_key) {
                                overlays.push(nodecraft_display::service::OverlayRequest {
                                    kind: nodecraft_display::service::OverlayKind::DifferenceAdded,
                                    part: std::sync::Arc::new(nodecraft_exact::ExactPart::singleton(key, std::sync::Arc::clone(added))),
                                });
                            }
                            if let (Some(removed), Some(key)) = (difference.removed.as_ref(), difference.removed_key) {
                                overlays.push(nodecraft_display::service::OverlayRequest {
                                    kind: nodecraft_display::service::OverlayKind::DifferenceRemoved,
                                    part: std::sync::Arc::new(nodecraft_exact::ExactPart::singleton(key, std::sync::Arc::clone(removed))),
                                });
                            }
                        }
                        let part = exact
                            .exact_parts
                            .get(&evaluation_target)
                            .expect("service display part")
                            .clone();
                        let snapshot = nodecraft_display::service::ExactSnapshot {
                            breakpoint: std::collections::BTreeMap::from([(
                                evaluation_target.clone(),
                                part,
                            )]),
                            overlays,
                        };
                        let snapshot = std::sync::Arc::new(snapshot);
                        let mut sent = exact.clone();
                        if result_tx.send((job.id, job.tab_id, sent)).is_err() { break; }
                        let _ = snapshot_tx.send((
                            job.id,
                            job.tab_id,
                            DisplaySnapshot {
                                id: job.id,
                                target: evaluation_target.clone(),
                                feature_ghost: job.feature_ghost,
                                snapshot: std::sync::Arc::clone(&snapshot),
                            },
                        ));
                        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                            eprintln!("[t {}] snapshot sent +{:?}", job.id, timeline.elapsed());
                        }
                        display_worker.submit(nodecraft_display::service::DisplayRequest {
                            revision: job.id,
                            snapshot_id: job.id,
                            quality: match job.preview_detail {
                                Some(detail) => nodecraft_display::DisplayQuality::Preview(detail),
                                None => nodecraft_display::DisplayQuality::Full(0.35),
                            },
                            feature_ghost: job.feature_ghost,
                            snapshot: std::sync::Arc::clone(&snapshot),
                        });
                        state.cache = cache;
                        states.insert(job.document_id.clone(), state);
                        continue 'jobs;
                    }
                    if !job.run_exporters {
                        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                            eprintln!("[t {}] empty display +{:?}", job.id, timeline.elapsed());
                        }
                        if let Some(exact) = exact.as_ref() {
                            let sent = exact.clone();
                            if result_tx.send((job.id, job.tab_id, sent)).is_err() { break; }
                            // "Nothing to display" is still a display result:
                            // publish an empty snapshot/frame so stale geometry
                            // is cleared instead of waiting forever.
                            let snapshot = std::sync::Arc::new(
                                nodecraft_display::service::ExactSnapshot::default(),
                            );
                            let _ = snapshot_tx.send((
                                job.id,
                                job.tab_id,
                                DisplaySnapshot {
                                    id: job.id,
                                    target: exact.target.clone(),
                                    feature_ghost: false,
                                    snapshot: std::sync::Arc::clone(&snapshot),
                                },
                            ));
                            display_worker.submit(nodecraft_display::service::DisplayRequest {
                                revision: job.id,
                                snapshot_id: job.id,
                                quality: match job.preview_detail {
                                    Some(detail) => nodecraft_display::DisplayQuality::Preview(detail),
                                    None => nodecraft_display::DisplayQuality::Full(0.35),
                                },
                                feature_ghost: false,
                                snapshot,
                            });
                        }
                        state.cache = cache;
                        states.insert(job.document_id.clone(), state);
                        continue 'jobs;
                    }
                    // A newer edit may have arrived while the exact pass ran. Publish
                    // this exact result, abandon the now-stale display pass, and let
                    // the newest job use the warmed cache.
                    if let Ok(mut next) = job_rx.try_recv() {
                        let mut clear_cache = next.clear_cache;
                        while let Ok(newer) = job_rx.try_recv() {
                            clear_cache |= newer.clear_cache;
                            next = newer;
                        }
                        next.clear_cache = clear_cache;
                        state.cache = cache;
                        states.insert(job.document_id.clone(), state);
                        pending_job = Some(next);
                        continue 'jobs;
                    }
                }
                let evaluation_target = target.clone().unwrap_or_default();
                if job.run_exporters {
                    if let Some(exact) = exact_for_display.as_ref() {
                        if let Some(exporter) = job
                            .document
                            .nodes
                            .iter()
                            .find(|node| node.node_type == "exporter" && node.id == evaluation_target)
                            .or_else(|| job.document.nodes.iter().rev().find(|node| node.node_type == "exporter"))
                        {
                            let input = job
                                .document
                                .connections
                                .iter()
                                .find(|link| link.to == exporter.id && link.input == "shape")
                                .map(|link| link.from.clone());
                            let input_export = input.clone();
                            let result = (|| -> Result<(), String> {
                                let input = input_export.ok_or("Export node has no shape input")?;
                                let part = exact.exact_parts.get(&input).ok_or("Export input has no exact Part")?;
                                let directory = exporter.values.get("directory").and_then(serde_json::Value::as_str).unwrap_or("");
                                let root = std::env::var("NODECRAFT_PROJECT_ROOT").unwrap_or_else(|_| ".".into());
                                let mut output = if directory.is_empty() {
                                    std::path::PathBuf::from(&root).join("output")
                                } else {
                                    std::path::PathBuf::from(directory)
                                };
                                if output.is_relative() { output = std::path::PathBuf::from(&root).join(output); }
                                let stem = exporter.values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("model").trim_end_matches(".step").trim();
                                let stem = if stem.is_empty() { "model" } else { stem };
                                std::fs::create_dir_all(&output).map_err(|error| error.to_string())?;
                                nodecraft_display::export_step(part, &output.join(format!("{stem}.step")), &exporter.values)
                            })();
                            let mut evaluation = exact.clone();
                            evaluation.target = exporter.id.clone();
                            evaluation.exact_parts.clear();
                            evaluation.error = result.err();
                            state.cache = cache;
                            states.insert(job.document_id.clone(), state);
                            if result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                            continue 'jobs;
                        }
                    }
                }
                // The display service is the only display path. A request that
                // is not displayable (exporter fallthrough or a missing exact
                // part) returns its exact result so the app can report it.
                if let Some(exact) = exact_for_display {
                    let mut evaluation = exact;
                    evaluation.exact_parts.clear();
                    if result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                }
                state.cache = cache;
                states.insert(job.document_id.clone(), state);
                if states.len() > 16
                    && let Some(oldest) = states.keys().next().cloned()
                {
                    states.remove(&oldest);
                }
            }
        });
        Self {
            tx: job_tx,
            rx: result_rx,
            progress_rx,
            snapshot_rx,
            active_token: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub(super) fn send(&self, job: EvalJob) {
        if let Ok(mut active) = self.active_token.lock() {
            if let Some(previous) = active.take() {
                previous.cancel();
            }
            *active = Some(job.token.clone());
        }
        let _ = self.tx.send(job);
    }

    pub(super) fn try_recv(&self) -> Option<(u64, u64, engine_bridge::Evaluation)> {
        self.rx.try_recv().ok()
    }

    pub(super) fn try_recv_progress(&self) -> Option<(u64, u64, nodecraft_engine::ProgressEvent)> {
        self.progress_rx.try_recv().ok()
    }

    pub(super) fn try_recv_snapshot(&self) -> Option<(u64, u64, DisplaySnapshot)> {
        self.snapshot_rx.try_recv().ok()
    }
}


impl NodeCraftApp {
    fn start_node_activity(&mut self, id: String) {
        if self.activity_seen.insert(id.clone()) {
            self.activity_active.insert(id.clone());
            self.node_editor.activity_started(&id);
        }
    }

    fn finish_node_activity(&mut self, id: &str) {
        if self.activity_active.remove(id) {
            self.node_editor.activity_finished(id);
        }
    }

    /// Finish any node still marked active after the exact result arrives.
    /// Normal operation ends each node at its engine completion event; this is
    /// only a safety net and keeps the fade-out instead of clearing all glows.
    fn finish_remaining_node_activity(&mut self) {
        let active: Vec<String> = self.activity_active.drain().collect();
        for id in active {
            self.node_editor.activity_finished(&id);
        }
    }

    /// Apply the exact worker's result to node/evaluation state. Display fields
    /// are deliberately ignored here; a DisplayFrame owns those.
    fn apply_exact_result(
        &mut self,
        binding: Option<&EvalBinding>,
        evaluation: &engine_bridge::Evaluation,
    ) {
        self.eval_pending = None;
        self.eval_progress = None;
        self.finish_remaining_node_activity();
        self.evaluation_ms = evaluation.elapsed_ms;
        self.cache_hits = evaluation.cache_hits;
        self.cache_misses = evaluation.cache_misses;
        self.exact_brep_hits = evaluation.brep_hits;
        self.exact_part_hits = evaluation.part_hits;
        self.error = evaluation.error.clone();
        let errors = map_eval_errors(binding, &evaluation.node_errors);
        self.node_errors = errors.keys().cloned().collect();
        self.node_error_messages = errors;
        self.node_editor.set_error_nodes(self.node_errors.clone());
        self.difference_error = evaluation.difference_error.clone();
        self.difference_approximate = evaluation.difference_approximate;
        self.display_error = None;
        if let Some(binding) = binding {
            let display_target = map_eval_id_to_body(binding, &evaluation.target)
                .unwrap_or_else(|| evaluation.target.clone());
            self.eval_target = Some(display_target);
        } else {
            self.eval_target = Some(evaluation.target.clone());
        }
        self.warmup_until = self
            .warmup_until
            .max(Instant::now() + std::time::Duration::from_millis(500));
        self.status = if self.error.is_none() {
            format!(
                "Evaluated {} in {:.1} ms — kernel {:.1} ms",
                evaluation.target, evaluation.elapsed_ms, evaluation.kernel_ms,
            )
        } else {
            format!("Evaluation failed in {:.1} ms", evaluation.elapsed_ms)
        };
    }

    /// Apply a display frame. Owns viewport geometry and overlays only; node and
    /// evaluation state belong to `apply_exact_result`.
    pub(super) fn apply_display_frame(
        &mut self,
        frame: &nodecraft_display::service::DisplayFrame,
        target: &str,
        _feature_ghost: bool,
        force_preserve_camera: bool,
    ) {
        let apply_started = std::time::Instant::now();
        let Some(planned) = frame.breakpoint.get(target) else {
            self.mesh = None;
            self.viewport.set_mesh_preserving_camera(None);
            self.viewport.set_plans(Vec::new());
            self.egui_ctx.request_repaint();
            return;
        };
        let binding = self.active_eval_binding();
        let display_target = binding
            .as_ref()
            .and_then(|binding| map_eval_id_to_body(binding, target))
            .unwrap_or_else(|| target.to_owned());
        let mut plans = planned.plans.clone();
        for plan in &mut plans {
            plan.target.clone_from(&target.to_owned());
        }
        self.viewport.set_plans(plans);
        // Edges, picking and plan triangles must switch in the same activation.
        // While a staged plan swap is pending, keep the old display and retry
        // this frame once the new chunks are resident.
        if !self.applying_deferred_display && self.viewport.plan_pending() {
            self.deferred_display = Some((
                std::sync::Arc::new(frame.clone()),
                target.to_owned(),
                _feature_ghost,
                std::time::Instant::now(),
            ));
            self.egui_ctx.request_repaint();
            return;
        }
        let had_mesh = self.mesh.is_some();
        let same_target = self.displayed_target.as_deref() == Some(display_target.as_str());
        self.displayed_target = Some(display_target);
        self.mesh = Some(planned.edge_mesh.clone());
        let preserve = force_preserve_camera
            || mesh_preserves_camera(
                self.camera_initialized,
                had_mesh,
                same_target,
                std::mem::take(&mut self.preserve_view_for_next_mesh),
            );
        if !self.gizmo_dragging && self.gizmo_drag.is_none() {
            self.viewport.set_preview_transform(None, crate::viewport::PreviewScope::All);
        }
        self.viewport.set_tessellation_diagnostics(
            planned.stats.iter()
                .flat_map(|stats| stats.diagnostics.iter().cloned())
                .collect(),
        );

        let mut before = None;
        let mut after = None;
        let mut ghost = None;
        let mut added = None;
        let mut removed = None;
        let mut ghost_plans = Vec::new();
        let mut added_plans = Vec::new();
        let mut edit_overlay = None;
        for overlay in &frame.overlays {
            match overlay.kind {
                Some(nodecraft_display::service::OverlayKind::ComparisonBefore) => {
                    before = Some(overlay.mesh.clone());
                }
                Some(nodecraft_display::service::OverlayKind::ComparisonAfter) => {
                    after = Some(overlay.mesh.clone());
                }
                Some(nodecraft_display::service::OverlayKind::FeatureGhost) => {
                    ghost = Some(overlay.mesh.clone());
                    ghost_plans = overlay.plans.clone();
                }
                Some(nodecraft_display::service::OverlayKind::DifferenceAdded) => {
                    added = Some(overlay.mesh.clone());
                    added_plans = overlay.plans.clone();
                }
                Some(nodecraft_display::service::OverlayKind::DifferenceRemoved) => {
                    removed = Some(overlay.mesh.clone());
                }
                Some(nodecraft_display::service::OverlayKind::EditInput) => {
                    edit_overlay = Some((overlay.mesh.clone(), overlay.plans.clone()));
                }
                None => {}
            }
        }
        self.difference_stats = added.as_ref().zip(removed.as_ref())
            .map(|(added, removed)| (added.faces.len(), removed.faces.len()));
        if let Some((before, after)) = before.as_ref().zip(after.as_ref()) {
            let selected = self.selected_nodes.iter().next().cloned().unwrap_or_default();
            let before = self.reframe_comparison(&selected, before);
            let after = self.reframe_comparison(&selected, after);
            let exact = added.as_ref().zip(removed.as_ref()).filter(|(added, removed)| {
                !added.faces.is_empty() || !removed.faces.is_empty()
            });
            if self.viewport.selection_mode() == crate::viewport::SelectionMode::Object {
                if let Some((added, removed)) = exact {
                    let added = self.reframe_comparison(&selected, added);
                    let removed = self.reframe_comparison(&selected, removed);
                    if let Some(base) = &self.mesh {
                        self.viewport.set_exact_comparison_mesh(base, &added, &removed);
                    }
                } else if let Some(ghost) = &ghost {
                    if let Some(base) = &self.mesh {
                        self.viewport.set_feature_ghost_mesh(base, ghost);
                    }
                } else {
                    self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
                }
            } else if let Some((added, removed)) = exact {
                let added = self.reframe_comparison(&selected, added);
                let removed = self.reframe_comparison(&selected, removed);
                let base = self.mesh.as_ref().unwrap_or(&before);
                self.viewport.set_selection_comparison_mesh(base, base, &added, &removed);
            } else {
                let empty = Mesh::default();
                let base = self.mesh.as_ref().unwrap_or(&before);
                self.viewport.set_selection_comparison_mesh(base, base, &after, &empty);
            }
            if !preserve {
                self.viewport.fit();
                self.camera_initialized = true;
            }
        } else if let Some(ghost) = &ghost {
            if let Some(base) = &self.mesh {
                self.viewport.set_feature_ghost_mesh(base, ghost);
            }
            if !preserve {
                self.viewport.fit();
                self.camera_initialized = true;
            }
        } else if self.mesh.is_none() {
            self.viewport.set_mesh_preserving_camera(None);
            self.preserve_view_for_next_mesh = true;
            self.camera_initialized = true;
        } else if preserve {
            self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
        } else {
            self.viewport.set_mesh(self.mesh.as_ref());
            self.camera_initialized = true;
        }

        let selected_blend = self
            .selected_nodes
            .iter()
            .next()
            .and_then(|id| self.session.as_ref()?.document().nodes.iter().find(|node| &node.id == id))
            .is_some_and(|node| matches!(node.node_type.as_str(), "fillet" | "chamfer"));
        let overlay_plans = if selected_blend && !frame.feature_plans.is_empty() {
            frame.feature_plans.clone()
        } else if selected_blend {
            added_plans
        } else if !added_plans.is_empty() {
            added_plans
        } else {
            ghost_plans
        };
        self.viewport.set_feature_ghost_plans(overlay_plans);
        self.refresh_topology_picking();
        if let Some((mesh, plans)) = edit_overlay {
            let edit_started = std::time::Instant::now();
            self.edit_preview = Some(mesh);
            self.viewport.set_edit_ghost_plans(plans);
            self.attach_pending_edit_ghost();
            self.refresh_edit_overlay();
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!("[t {}] apply edit overlay {}ms", frame.revision, edit_started.elapsed().as_millis());
            }
        }
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {}] apply frame {}ms", frame.revision, apply_started.elapsed().as_millis());
        }
        self.last_applied_display = Some((
            std::sync::Arc::new(frame.clone()),
            target.to_owned(),
            _feature_ghost,
        ));
        self.egui_ctx.request_repaint();
    }

    fn poll_display_snapshot(&mut self) {
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        while let Some((_id, tab_id, snapshot)) = self.eval_worker.try_recv_snapshot() {
            if tab_id != active_tab_id {
                continue;
            }
            self.pending_display = Some((
                snapshot.id,
                snapshot.target.clone(),
                snapshot.feature_ghost,
                Instant::now(),
            ));
            if self
                .quality_display
                .as_ref()
                .is_some_and(|(_, snapshot_id, _, _, _)| *snapshot_id != snapshot.id)
            {
                self.quality_display = None;
            }
            self.display_snapshot = Some(snapshot);
        }
    }

    /// Apply a quality-tier frame without an evaluation request. This touches
    /// display state only: plans, meshes, overlays and picking data. It must
    /// not reset evaluation status, errors, activity or glow.
    fn poll_quality_display(&mut self) {
        let Some((revision, snapshot_id, target, feature_ghost, started)) =
            self.quality_display.clone()
        else {
            return;
        };
        // A newer exact snapshot means this quality frame plans old geometry.
        // Never let it overwrite the current breakpoint mesh.
        if self
            .display_snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.id > snapshot_id)
        {
            self.quality_display = None;
            self.egui_ctx.request_repaint();
            return;
        }
        let frame = self
            .display_worker
            .latest_frame()
            .filter(|frame| frame.revision == revision);
        let Some(frame) = frame else { return };
        let _ = started;
        self.quality_display = None;
        if let Some(error) = &frame.error {
            self.display_error = Some(error.clone());
            self.status = format!("Display failed: {error}");
            self.egui_ctx.request_repaint();
            return;
        }
        let started = std::time::Instant::now();
        self.apply_display_frame(&frame, &target, feature_ghost, false);
        self.last_display_us = Some((frame.plan_us, started.elapsed().as_micros() as u64));
    }

    /// Poll the display service for the exact pre-pass's deferred frame.
    fn poll_display_frame(&mut self) {
        let Some((id, target, feature_ghost, started)) = self.pending_display.clone() else { return };
        let frame = self
            .display_worker
            .latest_frame()
            .filter(|frame| frame.revision == id);
        let Some(frame) = frame else { return };
        let _ = started;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {id}] frame polled plan={}us apply next", frame.plan_us);
        }
        self.pending_display = None;
        if let Some(error) = &frame.error {
            self.display_error = Some(error.clone());
            self.status = format!("Display failed: {error}");
            self.egui_ctx.request_repaint();
            return;
        }
        let started = std::time::Instant::now();
        self.apply_display_frame(&frame, &target, feature_ghost, false);
        self.last_display_us = Some((frame.plan_us, started.elapsed().as_micros() as u64));
    }

    pub(super) fn poll_evaluation(&mut self) {
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let binding = self.active_eval_binding();
        let latest_id = self.eval_next;
        while let Some((id, tab_id, event)) = self.eval_worker.try_recv_progress() {
            if id != latest_id || tab_id != active_tab_id {
                continue;
            }
            match event.phase {
                nodecraft_engine::ProgressPhase::Start => {
                    let id = binding
                        .as_ref()
                        .and_then(|binding| map_eval_id_to_body(binding, &event.node_id))
                        .unwrap_or_else(|| event.node_id.clone());
                    self.start_node_activity(id);
                }
                nodecraft_engine::ProgressPhase::Complete => {
                    let id = binding
                        .as_ref()
                        .and_then(|binding| map_eval_id_to_body(binding, &event.node_id))
                        .unwrap_or_else(|| event.node_id.clone());
                    self.finish_node_activity(&id);
                    if let Some(progress) = self.eval_progress.as_mut() {
                        progress.exact_done = (progress.exact_done + 1).min(progress.total);
                    }
                }
                nodecraft_engine::ProgressPhase::BrepStart => {
                    let id = binding
                        .as_ref()
                        .and_then(|binding| map_eval_id_to_body(binding, &event.node_id))
                        .unwrap_or_else(|| event.node_id.clone());
                    self.start_node_activity(id);
                }
                nodecraft_engine::ProgressPhase::BrepComplete => {
                    let id = binding
                        .as_ref()
                        .and_then(|binding| map_eval_id_to_body(binding, &event.node_id))
                        .unwrap_or_else(|| event.node_id.clone());
                    self.finish_node_activity(&id);
                    if let Some(progress) = self.eval_progress.as_mut() {
                        progress.exact_done = (progress.exact_done + 1).min(progress.total);
                    }
                }
                nodecraft_engine::ProgressPhase::ExactComplete => {}
            }
        }
        self.poll_display_snapshot();
        self.poll_quality_display();
        self.poll_display_frame();
        let Some(pending) = self.eval_pending else { return };
        while let Some((id, tab_id, evaluation)) = self.eval_worker.try_recv() {
            let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
            if id != pending || tab_id != active_tab_id {
                continue;
            }
            self.last_exact = Some((
                self.session.as_ref().map(Session::revision).unwrap_or(0),
                evaluation.clone(),
            ));
            self.apply_exact_result(binding.as_ref(), &evaluation);
            self.egui_ctx.request_repaint();
            continue;
        }
    }

}