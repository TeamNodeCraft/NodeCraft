use super::*;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use nodecraft_engine::EngineCache;
use nodecraft_session::GraphDocument;

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

    /// Request a branch-only transform preview. Runs on its own thread and cache.
    pub(super) fn request_transform_preview(&mut self, target: &str) {
        let Some(session) = &self.session else { return };
        self.preview_next = self.preview_next.wrapping_add(1);
        let request_id = self.preview_next;
        self.preview_pending = Some(request_id);
        self.preview_worker.request(PreviewRequest {
            request_id,
            tab_id: self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0),
            document: session.document().clone(),
            linked_documents: self.linked_documents(),
            target: target.to_owned(),
            token: nodecraft_engine::kernel_pool::RevisionToken::new(),
        });
    }

    pub(super) fn poll_preview(&mut self) {
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        while let Some(result) = self.preview_worker.try_recv() {
            if self.preview_pending != Some(result.request_id) || result.tab_id != active_tab_id {
                continue;
            }
            self.preview_pending = None;
            self.edit_preview = result.mesh;
            self.edit_preview_target = Some(result.target);
            self.attach_pending_edit_ghost();
            self.refresh_edit_overlay();
            self.egui_ctx.request_repaint();
        }
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
            return;
        }
        let evaluation_selection = self
            .face_pick_target
            .clone()
            .or_else(|| self.viewer_breakpoint.clone());
        let comparison = if self.selected_nodes.len() == 1 {
            let selected = self.selected_nodes.iter().next().map(String::as_str);
            comparison_for_selection(session.document(), selected, self.feature_overlay)
        } else {
            None
        };
        let ghost_input = if self.face_pick_target.is_none()
            && self.feature_overlay
            && self.selected_nodes.len() == 1
        {
            self.selected_nodes
                .iter()
                .next()
                .and_then(|selected| feature_ghost_input(session.document(), selected))
        } else {
            None
        };
        // The dedicated preview worker owns the edit ghost mesh; the main
        // evaluation must not duplicate that branch-only work.
        let edit_preview: Option<String> = None;
        let dirty_nodes = self.pending_dirty.clone();
        self.pending_dirty.clear();
        // A selected node outside the viewed target closure is evaluated as an
        // extra validation target. It contributes node_errors (and therefore the
        // canvas error glow) without changing the displayed mesh or breakpoint.
        let viewed_closure = engine_bridge::target_for(
            session.document(),
            evaluation_selection.as_deref(),
        )
        .map(|target| target_closure(session.document(), &target))
        .unwrap_or_default();
        let mut validation_targets: Vec<String> = self
            .selected_nodes
            .iter()
            .filter(|id| !viewed_closure.contains(*id))
            .cloned()
            .collect();
        validation_targets.sort();
        validation_targets.dedup();
        // Sync helpers can run every frame. Evaluation is a pure function of the
        // graph revision, viewed target and comparison pair; skip identical work
        // before allocating a new pending id.
        let key = eval_key(
            self.session_generation,
            &session.document().id,
            session.revision(),
            &evaluation_selection,
            &comparison,
            &ghost_input,
            &edit_preview,
            &validation_targets,
            run_exporters,
        );
        if !clear_cache && self.last_eval_key.as_ref() == Some(&key) {
            return;
        }
        self.last_eval_key = Some(key);
        self.eval_next = self.eval_next.wrapping_add(1);
        let id = self.eval_next;
        self.eval_pending = Some(id);
        let tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let total = engine_bridge::target_for(session.document(), evaluation_selection.as_deref())
            .map(|target| target_closure(session.document(), &target).len())
            .unwrap_or(0);
        self.eval_progress = (total > 0).then_some(EvalProgress { total, done: 0 });
        self.node_editor.clear_activity();
        self.status = "Evaluating…".to_owned();
        let linked_documents = self.linked_documents();
        self.comparison_pair = comparison.clone();
        self.eval_worker.send(EvalJob {
            id,
            tab_id,
            linked_documents,
            document: session.document().clone(),
            selected: evaluation_selection,
            comparison,
            ghost_input,
            edit_preview,
            validation_targets,
            clear_cache,
            run_exporters,
            dirty_nodes,
            token: nodecraft_engine::kernel_pool::RevisionToken::new(),
        });
        self.egui_ctx.request_repaint();
    }

}

pub(super) struct EvalJob {
    id: u64,
    tab_id: u64,
    linked_documents: std::collections::BTreeMap<PathBuf, GraphDocument>,
    document: GraphDocument,
    selected: Option<String>,
    comparison: Option<(String, String)>,
    ghost_input: Option<String>,
    edit_preview: Option<String>,
    validation_targets: Vec<String>,
    clear_cache: bool,
    run_exporters: bool,
    dirty_nodes: std::collections::HashSet<String>,
    token: nodecraft_engine::kernel_pool::RevisionToken,
}

#[derive(Default)]
struct TabEvalState {
    cache: EngineCache,
    flat: nodecraft_session::flat::FlatGraphCache,
}

pub(super) struct EvalWorker {
    tx: Sender<EvalJob>,
    rx: Receiver<(u64, u64, engine_bridge::Evaluation)>,
    progress_rx: Receiver<(u64, u64, nodecraft_engine::ProgressEvent)>,
    active_token: std::sync::Arc<std::sync::Mutex<Option<nodecraft_engine::kernel_pool::RevisionToken>>>,
}

impl EvalWorker {
    pub(super) fn new() -> Self {
        let (job_tx, job_rx) = channel::<EvalJob>();
        let (result_tx, result_rx) = channel();
        let (progress_tx, progress_rx) = channel();
        std::thread::spawn(move || {
            // Cache ownership follows the tab/document, not the worker. This keeps
            // per-document dirty/cache lifetimes aligned and prevents one tab from
            // evicting or reusing another tab's entries.
            let mut states: std::collections::HashMap<u64, TabEvalState> =
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
                while let Ok(next) = job_rx.try_recv() {
                    job = next;
                }
                let mut state = states.remove(&job.tab_id).unwrap_or_default();
                nodecraft_engine::kernel_pool::set_current_revision_token(job.token.clone());
                if job.clear_cache {
                    // Re-evaluate is an explicit request for a fresh result: do
                    // not let mesh/BRep/failure/import entries answer it.
                    state.cache = EngineCache::new();
                }
                let target = engine_bridge::target_for(&job.document, job.selected.as_deref());
                let flat = match state.flat.resolve(&job.document, &job.linked_documents) {
                    Ok(flat) => flat.clone(),
                    Err(error) => {
                        let evaluation = target.map(|target| engine_bridge::Evaluation {
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
                            elapsed_ms: 0.0,
                            kernel_ms: 0.0,
                            mesh_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                        });
                        states.insert(job.tab_id, state);
                        if let Some(evaluation) = evaluation
                            && result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                        continue;
                    }
                };
                let progress_tx = progress_tx.clone();
                let mut on_progress = |event: nodecraft_engine::ProgressEvent| {
                    let _ = progress_tx.send((job.id, job.tab_id, event));
                };
                let mut cache = state.cache;
                if !job.run_exporters {
                    let exact_outcome = catch_unwind(AssertUnwindSafe(|| {
                        engine_bridge::evaluate_flat_exact(
                            flat.document(),
                            target.clone().unwrap_or_default(),
                            cache,
                            None,
                            None,
                            job.run_exporters,
                            &job.dirty_nodes,
                            None,
                            &job.validation_targets,
                            None,
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
                                error: Some("Engine panicked during exact evaluation".into()),
                                elapsed_ms: 0.0,
                                kernel_ms: 0.0,
                                mesh_ms: 0.0,
                                cache_hits: 0,
                                cache_misses: 0,
                            });
                            state.cache = cache;
                            if let Some(evaluation) = evaluation
                                && result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                            states.insert(job.tab_id, state);
                            continue;
                        }
                    };
                    if let Some(exact) = exact
                        && result_tx.send((job.id, job.tab_id, exact)).is_err() { break; }
                    // A newer edit may have arrived while the exact pass ran. Publish
                    // this exact result, abandon the now-stale display pass, and let
                    // the newest job use the warmed cache.
                    if let Ok(mut next) = job_rx.try_recv() {
                        while let Ok(newer) = job_rx.try_recv() {
                            next = newer;
                        }
                        state.cache = cache;
                        states.insert(job.tab_id, state);
                        pending_job = Some(next);
                        continue 'jobs;
                    }
                }
                let display_outcome = catch_unwind(AssertUnwindSafe(|| {
                    engine_bridge::evaluate_flat_with_progress_comparison_dirty(
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
                    )
                }));
                let evaluation = match display_outcome {
                    Ok((evaluation, next_cache)) => {
                        cache = next_cache;
                        evaluation
                    }
                    Err(_) => {
                        cache = EngineCache::new();
                        target.map(|target| engine_bridge::Evaluation {
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
                            error: Some("Engine panicked while evaluating this target".into()),
                            elapsed_ms: 0.0,
                            kernel_ms: 0.0,
                            mesh_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                        })
                    }
                };
                state.cache = cache;
                states.insert(job.tab_id, state);
                if states.len() > 16
                    && let Some(oldest) = states.keys().next().copied()
                {
                    states.remove(&oldest);
                }
                if let Some(evaluation) = evaluation {
                    if result_tx.send((job.id, job.tab_id, evaluation)).is_err() {
                        break;
                    }
                }
            }
        });
        Self {
            tx: job_tx,
            rx: result_rx,
            progress_rx,
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
}


impl NodeCraftApp {
    pub(super) fn poll_evaluation(&mut self) {
        self.poll_preview();
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let latest_id = self.eval_next;
        while let Some((id, tab_id, event)) = self.eval_worker.try_recv_progress() {
            if id != latest_id || tab_id != active_tab_id {
                continue;
            }
            match event.phase {
                nodecraft_engine::ProgressPhase::Start => {
                    self.node_editor.activity_started(&event.node_id);
                }
                nodecraft_engine::ProgressPhase::Complete => {
                    self.node_editor.activity_finished(&event.node_id);
                    if let Some(progress) = self.eval_progress.as_mut() {
                        progress.done = (progress.done + 1).min(progress.total);
                    }
                }
                nodecraft_engine::ProgressPhase::BrepStart
                | nodecraft_engine::ProgressPhase::BrepComplete => {}
            }
        }
        let Some(pending) = self.eval_pending else { return };
        while let Some((id, tab_id, evaluation)) = self.eval_worker.try_recv() {
            let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
            if id != pending || tab_id != active_tab_id {
                continue;
            }
            if !evaluation.display_complete {
                // Exact pre-pass: publish errors/status immediately, but keep the
                // current mesh and leave the evaluation pending for the display pass.
                self.evaluation_ms = evaluation.elapsed_ms;
                self.error = evaluation.error;
                self.node_errors = evaluation.node_errors.keys().cloned().collect();
                self.node_error_messages = evaluation.node_errors.clone();
                self.node_editor.set_error_nodes(self.node_errors.clone());
                self.cache_hits = evaluation.cache_hits;
                self.cache_misses = evaluation.cache_misses;
                self.egui_ctx.request_repaint();
                continue;
            }
            self.eval_pending = None;
            self.eval_progress = None;
            self.evaluation_ms = evaluation.elapsed_ms;
            self.cache_hits = evaluation.cache_hits;
            self.cache_misses = evaluation.cache_misses;
            self.error = evaluation.error;
            self.node_errors = evaluation.node_errors.keys().cloned().collect();
            self.node_error_messages = evaluation.node_errors.clone();
            self.node_editor.set_error_nodes(self.node_errors.clone());
            let had_mesh = self.mesh.is_some();
            let previous_target = self.eval_target.clone();
            let same_target = previous_target.as_deref() == Some(evaluation.target.as_str());
            self.eval_target = Some(evaluation.target.clone());
            self.mesh = evaluation.mesh;
            // Preview meshes are owned by PreviewWorker. The main evaluation
            // deliberately carries no edit preview, so assigning its None here
            // would clear a valid ghost after every drag/re-evaluation.

            // Preserve the camera for updates of the same viewed target (slider/edit),
            // but fit when the breakpoint/target changes so a sphere added to a large
            // graph is framed at its own position instead of the old assembly view.
            let preserve = mesh_preserves_camera(
                self.camera_initialized,
                had_mesh,
                same_target,
                std::mem::take(&mut self.preserve_view_for_next_mesh),
            );
            self.difference_stats = evaluation
                .difference
                .as_ref()
                .map(|(added, removed)| (added.faces.len(), removed.faces.len()));
            self.difference_error = evaluation.difference_error.clone();
            self.difference_approximate = evaluation.difference_approximate;
            if let Some((before, after)) = &evaluation.comparison {
                let selected = self.selected_nodes.iter().next().cloned().unwrap_or_default();
                let before = self.reframe_comparison(&selected, before);
                let after = self.reframe_comparison(&selected, after);
                let exact = evaluation.difference.as_ref().filter(|(added, removed)| {
                    !added.faces.is_empty() || !removed.faces.is_empty()
                });
                if self.viewport.selection_mode() == crate::viewport::SelectionMode::Object {
                    if let Some((added, removed)) = exact {
                        let added = self.reframe_comparison(&selected, added);
                        let removed = self.reframe_comparison(&selected, removed);
                        if let Some(base) = &self.mesh {
                            self.viewport.set_exact_comparison_mesh(base, &added, &removed);
                        } else {
                            self.viewport
                                .set_mesh_preserving_camera(self.mesh.as_ref());
                        }
                    } else if let Some(ghost) = &evaluation.ghost {
                        // Exact feature difference unavailable: fall back to the
                        // feature's input mesh as a faint ghost.
                        if let Some(base) = &self.mesh {
                            self.viewport.set_feature_ghost_mesh(base, ghost);
                        } else {
                            self.viewport
                                .set_mesh_preserving_camera(self.mesh.as_ref());
                        }
                    } else {
                        self.viewport
                            .set_mesh_preserving_camera(self.mesh.as_ref());
                    }
                } else if let Some((added, removed)) = exact {
                    let added = self.reframe_comparison(&selected, added);
                    let removed = self.reframe_comparison(&selected, removed);
                    self.viewport.set_selection_comparison_mesh(&after, &before, &added, &removed);
                } else {
                    let empty = Mesh::default();
                    self.viewport.set_selection_comparison_mesh(&after, &before, &empty, &before);
                }
                if !preserve {
                    self.viewport.fit();
                    self.camera_initialized = true;
                }
            } else if let Some(ghost) = &evaluation.ghost {
                if let Some(base) = &self.mesh {
                    self.viewport.set_feature_ghost_mesh(base, ghost);
                } else {
                    self.viewport.set_mesh_preserving_camera(None);
                }
                if !preserve {
                    self.viewport.fit();
                    self.camera_initialized = true;
                }
            } else if self.mesh.is_none() {
                // An empty evaluation (for example a sketch breakpoint) must clear
                // the previous body, but clearing must not refit the camera: the
                // sketch overlay and the 3D body should stay in the same view.
                // Preserve once more so switching back to the body does not fit
                // either.
                self.viewport.set_mesh_preserving_camera(None);
                self.preserve_view_for_next_mesh = true;
                self.camera_initialized = true;
            } else if preserve {
                self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
            } else {
                self.viewport.set_mesh(self.mesh.as_ref());
                self.camera_initialized = true;
            }
            self.refresh_topology_picking();
            if self.viewport.selection_mode() == crate::viewport::SelectionMode::Edge
                && self.eval_target == self.face_pick_target
                && self.selected_nodes.len() == 1
                && let Some(session) = &self.session
                && let Some(node) = session
                    .document()
                    .nodes
                    .iter()
                    .find(|node| self.selected_nodes.contains(&node.id))
                && Self::selection_spec(&node.node_type)
                    .is_some_and(|(_, mode)| mode == crate::viewport::SelectionMode::Edge)
            {
                let centers: Vec<[f64; 3]> = node
                    .values
                    .get("edges")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|item| {
                        let center = item.get("center")?.as_array()?;
                        (center.len() == 3).then(|| [
                            center[0].as_f64().unwrap_or_default(),
                            center[1].as_f64().unwrap_or_default(),
                            center[2].as_f64().unwrap_or_default(),
                        ])
                    })
                    .collect();
                self.viewport.select_edges_by_centers(&centers);
            }
            self.node_editor.pulse_glow();
            self.warmup_until = self
                .warmup_until
                .max(Instant::now() + std::time::Duration::from_millis(500));
            self.status = if self.error.is_none() {
                format!(
                    "Evaluated {} in {:.1} ms — kernel {:.1} ms, mesh {:.1} ms ({} cache hits, {} misses)",
                    evaluation.target,
                    self.evaluation_ms,
                    evaluation.kernel_ms,
                    evaluation.mesh_ms,
                    self.cache_hits,
                    self.cache_misses
                )
            } else {
                format!("Evaluation failed in {:.1} ms", self.evaluation_ms)
            };
            if self.difference_approximate {
                self.status.push_str(" — feature ghost is approximate");
            }
            self.attach_pending_edit_ghost();
            self.refresh_edit_overlay();
            self.egui_ctx.request_repaint();
        }
    }

}
