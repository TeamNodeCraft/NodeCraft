use super::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use nodecraft_engine::EngineCache;
use nodecraft_session::GraphDocument;

/// Test-only instrumentation for the evaluation worker's job hand-off. The
/// worker pauses after picking a job so a test can deterministically queue the
/// next request while the first one is still running.
#[cfg(test)]
mod test_support {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static PAUSE_AFTER_PICK_MS: AtomicU64 = AtomicU64::new(0);
    static PICKED: AtomicU64 = AtomicU64::new(0);
    /// The order in which the worker thread took messages off its channel.
    /// A foreground evaluation must always appear before the validations it
    /// owns, because that order is the worker's priority guarantee.
    pub(super) const FOREGROUND: u8 = 0;
    pub(super) const VALIDATION: u8 = 1;
    static RECEIVED: std::sync::Mutex<Vec<u8>> = std::sync::Mutex::new(Vec::new());
    /// These tests share one global pause/counter, so they must not run
    /// concurrently with each other even when cargo uses several threads.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub(super) fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(super) fn reset() {
        PAUSE_AFTER_PICK_MS.store(0, Ordering::SeqCst);
        PICKED.store(0, Ordering::SeqCst);
        RECEIVED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    pub(super) fn record_received(kind: u8) {
        RECEIVED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(kind);
    }

    pub(super) fn received() -> Vec<u8> {
        RECEIVED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(super) fn set_pause_after_pick_ms(ms: u64) {
        PAUSE_AFTER_PICK_MS.store(ms, Ordering::SeqCst);
    }

    pub(super) fn picked() -> u64 {
        PICKED.load(Ordering::SeqCst)
    }

    pub(super) fn after_pick() {
        PICKED.fetch_add(1, Ordering::SeqCst);
        let ms = PAUSE_AFTER_PICK_MS.load(Ordering::SeqCst);
        if ms > 0 {
            std::thread::sleep(Duration::from_millis(ms));
        }
    }
}

fn map_eval_warnings(
    binding: Option<&EvalBinding>,
    warnings: &std::collections::BTreeMap<String, Vec<String>>,
) -> std::collections::BTreeMap<String, Vec<String>> {
    match binding {
        Some(binding) => warnings
            .iter()
            .filter_map(|(id, messages)| {
                map_eval_id_to_body(binding, id).map(|id| (id, messages.clone()))
            })
            .collect(),
        None => warnings.clone(),
    }
}

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


/// One exact result's display submission. The app has deliberately no display
/// identity of its own: a body-key-only key cannot see placement, so it would
/// skip a newer snapshot whose bodies kept their producing node's `BodyKey`.
/// The display service reuses plans by body key plus quality, and the GPU plan
/// layer re-uploads only when the plan or its instance transforms change, so it
/// is the one place that decides what work a newer snapshot needs.
pub(super) struct DisplaySubmission<'a> {
    pub evaluation: &'a engine_bridge::Evaluation,
    pub feature_ghost: bool,
    /// Exact snapshot built by the caller, so display-only LOD state can be
    /// reconciled against the same snapshot that is submitted.
    pub snapshot: std::sync::Arc<nodecraft_display::service::ExactSnapshot>,
    pub run_exporters: bool,
    /// Fallback quality for bodies absent from `body_quality`.
    pub quality: nodecraft_display::DisplayQuality,
    /// Per-body screen-space levels from [`LodState`].
    pub body_quality: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::DisplayQuality>,
    /// Nearest-first planning order.
    pub body_priority: Vec<nodecraft_display::BodyKey>,
}

/// Hand one exact snapshot to the display service. Every exact result for the
/// viewed target goes through here, including an identical re-view: the service
/// owns de-duplication, not the app.
#[allow(clippy::too_many_arguments)]
pub(super) fn submit_exact_snapshot(
    submission: DisplaySubmission<'_>,
    worker: &nodecraft_display::service::DisplayWorker,
    next: &mut u64,
    installed: &mut Option<DisplaySnapshot>,
    pending: &mut Option<(u64, String, bool, Instant)>,
    quality_display: &mut Option<(u64, u64, String, bool, Instant)>,
) -> bool {
    if submission.run_exporters {
        return false;
    }
    let snapshot = submission.snapshot;
    *next = next.wrapping_add(1);
    let revision = *next;
    let target = submission.evaluation.target.clone();
    *installed = Some(DisplaySnapshot {
        id: revision,
        target: target.clone(),
        feature_ghost: submission.feature_ghost,
        snapshot: std::sync::Arc::clone(&snapshot),
    });
    *pending = Some((revision, target, submission.feature_ghost, Instant::now()));
    *quality_display = None;
    worker.submit(nodecraft_display::service::DisplayRequest {
        revision,
        snapshot_id: revision,
        quality: submission.quality,
        body_quality: submission.body_quality,
        body_priority: submission.body_priority,
        feature_ghost: submission.feature_ghost,
        snapshot,
    });
    true
}

/// Runtime importer-source hashes for `document`, keyed by node id.
///
/// Shared by the evaluation key/job and by the session-setup paths in `app.rs`
/// so both sides fold exactly the same source state into a revision.
pub(super) fn source_hashes(
    changes: &crate::source_service::ChangeTracker,
    document: &GraphDocument,
) -> std::collections::BTreeMap<String, String> {
    document
        .nodes
        .iter()
        .filter(|node| nodecraft_node_contract::import_source_node(&node.node_type))
        .filter_map(|node| {
            changes
                .hash(&document.id, &node.id)
                .map(|hash| (node.id.clone(), hash.to_owned()))
        })
        .collect()
}

impl NodeCraftApp {
    pub(super) fn evaluate(&mut self) {
        self.evaluate_with_cache_policy(false, false);
    }

    /// The active document's folder and ingested sources, passed to every
    /// evaluation so relative importer paths resolve in one place.
    pub(super) fn active_document_directory(&self) -> Option<PathBuf> {
        self.tabs
            .get(self.active_tab)
            .and_then(|tab| tab.path.clone())
            .or_else(|| self.path.clone())
            .and_then(|path| path.parent().map(Path::to_path_buf))
    }

    /// Runtime source content hashes for `document`'s importers, keyed by node
    /// id. They come from the ChangeTracker/source service and never live in the
    /// document; the evaluation request folds them into cache identity.
    pub(super) fn source_hashes(
        &self,
        document: &GraphDocument,
    ) -> std::collections::BTreeMap<String, String> {
        source_hashes(&self.source_changes, document)
    }

    fn import_context(&self, document: &GraphDocument) -> nodecraft_engine::ImportContext {
        self.import_context_with(document, self.source_hashes(document))
    }

    fn import_context_with(
        &self,
        document: &GraphDocument,
        source_hashes: std::collections::BTreeMap<String, String>,
    ) -> nodecraft_engine::ImportContext {
        nodecraft_engine::ImportContext::new(
            self.active_document_directory(),
            document.ingested_sources.clone(),
        )
        .with_source_hashes(source_hashes)
    }

    fn linked_documents(&self) -> std::sync::Arc<std::collections::BTreeMap<PathBuf, GraphDocument>> {
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
        std::sync::Arc::new(documents)
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
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[pool] dirty_nodes clear_cache={clear_cache} count={} ids={}",
                dirty_nodes.len(),
                dirty_nodes.iter().cloned().collect::<Vec<_>>().join(",")
            );
        }
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
        let selected_eval_ids: Vec<String> = self
            .selected_nodes
            .iter()
            .filter_map(|id| {
                binding
                    .as_ref()
                    .and_then(|binding| map_body_id_to_eval(binding, id))
                    .or_else(|| Some(id.clone()))
            })
            .collect();
        let validation_targets = validation_targets_outside(&selected_eval_ids, &viewed_closure);
        // Keep the diagnostic selection in the same body-id space as the merged
        // results so deselecting a target drops its glow again.
        let body_validation_selection: Vec<String> = validation_targets
            .iter()
            .map(|target| {
                binding
                    .as_ref()
                    .and_then(|binding| map_eval_id_to_body(binding, target))
                    .unwrap_or_else(|| target.clone())
            })
            .collect();
        self.validation_selection = Some(body_validation_selection);
        let revision = session.revision();
        let eval_document_id = binding.as_ref().map_or_else(
            || session.document().id.clone(),
            |binding| format!("{}::{}", binding.source_document_id, binding.prefix),
        );
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
        // Clone only when this selection actually needs validation work.
        let validation_document =
            (!validation_targets.is_empty()).then(|| std::sync::Arc::new(eval_document.clone()));
        let feature_ghost = self.selected_nodes.iter().any(|id| {
            eval_document
                .nodes
                .iter()
                .find(|node| &node.id == id)
                .is_some_and(|node| matches!(node.node_type.as_str(), "fillet" | "chamfer"))
        });
        let linked_documents = self.linked_documents();
        // Source hashes are runtime state and belong to this evaluation's
        // identity, exactly like the linked-document digest below.
        let source_hashes = self.source_hashes(eval_document);
        let key = ExactKey::new(
            self.session_generation,
            eval_document_id,
            revision,
            evaluation_selection.clone(),
            comparison.clone(),
            ghost_input.clone(),
            edit_preview.clone(),
            feature_ghost,
            run_exporters,
            &source_hashes,
            &linked_documents,
        );
        let tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        // A cached exact result is applied here, on the app side: no worker
        // job, no status reset, no progress reset and no activity clear.
        // Exporter runs must execute their write, so they always miss.
        if !clear_cache
            && !run_exporters
            && let Some(cached) = self
                .exact_results
                .get(&tab_id)
                .and_then(|cache| cache.get(&key))
                .cloned()
        {
            self.eval_worker.cancel_active();
            self.eval_pending = None;
            self.pending_exact = None;
            let mut evaluation = cached;
            // Reused exact Parts were already measured when they were produced.
            // Report the cache hit as free work instead of the old cost.
            evaluation.elapsed_ms = 0.0;
            evaluation.kernel_ms = 0.0;
            evaluation.cache_hits = 0;
            evaluation.cache_misses = 0;
            evaluation.brep_hits = 0;
            evaluation.part_hits = 0;
            self.last_exact = Some((revision, evaluation.clone()));
            self.apply_exact_result(binding.as_ref(), &evaluation);
            self.schedule_validation(
                tab_id,
                &document_id,
                &linked_documents,
                validation_document.clone(),
                &validation_targets,
                &binding,
                revision,
            );
            self.submit_exact_display(
                &evaluation,
                feature_ghost,
                edit_preview.as_deref(),
                ghost_input.as_deref(),
                comparison.as_ref().map(|(before, after)| (before.as_str(), after.as_str())),
                run_exporters,
            );
            self.egui_ctx.request_repaint();
            return;
        }
        // The incoming document is needed for the worker job; clone it before
        // touching evaluation state so the session borrow ends first.
        let job_document = eval_document.clone();
        // Reuse the session's revision cache only when the worker's flat graph
        // is guaranteed to be this exact document: no definition-body binding,
        // no linked documents, and no inline graph/group/linkedpart expansion.
        // The worker re-checks the flat node/connection counts before using it.
        let session_hashes = if binding.is_none()
            && linked_documents.is_empty()
            && !job_document.nodes.iter().any(|node| {
                matches!(node.node_type.as_str(), "group" | "graph" | "linkedpart")
            })
        {
            session.hashes_for_sources(&source_hashes)
        } else {
            None
        };
        let imports = self.import_context_with(&job_document, source_hashes);
        self.refresh_diagnostics();
        self.eval_next = self.eval_next.wrapping_add(1);
        let id = self.eval_next;
        self.eval_pending = Some(id);
        self.pending_exact = Some((
            key.clone(),
            DisplayFields {
                feature_ghost,
                edit_preview: edit_preview.clone(),
                ghost_input: ghost_input.clone(),
                comparison: comparison.clone(),
                run_exporters,
            },
        ));
        let total = engine_bridge::target_for(&job_document, evaluation_selection.as_deref())
            .map(|target| target_closure(&job_document, &target).len())
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
        self.comparison_pair = comparison.clone();
        // Prepare the validation requests first (bookkeeping only), then hand
        // both to the worker through the one submission path. The foreground
        // job must be enqueued before the validations it owns: the worker is
        // idle on a cache miss and takes the first message immediately, so a
        // validation sent first would run before the user's evaluation.
        let validation_jobs = self.take_validation_jobs(
            tab_id,
            &document_id,
            &linked_documents,
            validation_document,
            &validation_targets,
            &binding,
            revision,
        );
        self.last_exact = None;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {id}] app sent");
        }
        let submit_started = std::time::Instant::now();
        self.eval_worker.send_evaluation(
            EvalJob {
                id,
                tab_id,
                document_id,
                linked_documents,
                document: job_document,
                imports,
                // Session revision hashes for `document` when it is the exact
                // graph the worker will request; `None` asks the bridge to hash.
                session_hashes,
                selected: evaluation_selection,
                comparison,
                ghost_input,
                edit_preview,
                clear_cache,
                run_exporters,
                dirty_nodes,
                key,
                token: nodecraft_engine::kernel_pool::RevisionToken::new(),
            },
            validation_jobs,
        );
        self.profiler_record(
            crate::gpu_profiler::Series::EvalSubmit,
            submit_started.elapsed().as_secs_f64() * 1000.0,
        );
        self.egui_ctx.request_repaint();
    }
}

pub(super) struct EvalJob {
    id: u64,
    tab_id: u64,
    document_id: String,
    linked_documents: std::sync::Arc<std::collections::BTreeMap<PathBuf, GraphDocument>>,
    document: GraphDocument,
    imports: nodecraft_engine::ImportContext,
    /// Content hashes of the session revision this job evaluates. Present only
    /// when the job document is the session document and no flattening can
    /// change the worker's graph; the bridge otherwise hashes the flat graph.
    session_hashes: Option<std::sync::Arc<std::collections::BTreeMap<String, nodecraft_session::ContentHash>>>,
    selected: Option<String>,
    comparison: Option<(String, String)>,
    ghost_input: Option<String>,
    edit_preview: Option<String>,
    clear_cache: bool,
    run_exporters: bool,
    dirty_nodes: std::collections::HashSet<String>,
    /// The same exact identity the app records for this request. The worker
    /// uses its graph fields to retire validation work for a replaced graph.
    key: ExactKey,
    token: nodecraft_engine::kernel_pool::RevisionToken,
}

/// One low-priority exact pass for a selected node outside the viewed closure.
/// It never carries display inputs: its result only feeds diagnostics.
pub(super) struct ValidationJob {
    id: u64,
    tab_id: u64,
    document_id: String,
    linked_documents: std::sync::Arc<std::collections::BTreeMap<PathBuf, GraphDocument>>,
    document: std::sync::Arc<GraphDocument>,
    imports: nodecraft_engine::ImportContext,
    target: String,
    /// The same exact identity the app records for this request. One identity
    /// decides both app bookkeeping and worker supersession.
    key: ExactKey,
    token: nodecraft_engine::kernel_pool::RevisionToken,
}

impl ValidationJob {
    /// A newer validation request replaces an older one only when it asks for
    /// the same tab, document and target. A different target stays pending, so
    /// selecting two cold nodes still validates both.
    fn same_request(&self, other: &Self) -> bool {
        self.tab_id == other.tab_id
            && self.document_id == other.document_id
            && self.target == other.target
    }
}

/// Bookkeeping for an in-flight validation request. The key is the same exact
/// identity a later view of the target would use, so a validation result also
/// warms the per-tab exact cache.
pub(super) struct ValidationPending {
    pub tab_id: u64,
    pub body_target: String,
    pub key: ExactKey,
}

/// Remove one validation request from the app's bookkeeping. A result and an
/// explicit worker retirement use the same path, so a request that will never
/// produce a result cannot leave `validation_pending` growing forever.
pub(super) fn take_validation_request(
    pending: &mut std::collections::HashMap<u64, ValidationPending>,
    requested: &mut std::collections::HashSet<ExactKey>,
    id: u64,
) -> Option<ValidationPending> {
    let request = pending.remove(&id)?;
    requested.remove(&request.key);
    Some(request)
}

/// Diagnostics from one validation result, merged into node_errors/warnings
/// without touching status, progress, activity, last_exact or display state.
#[derive(Clone, Default)]
pub(super) struct ValidationDiagnostics {
    pub errors: std::collections::BTreeMap<String, String>,
    pub warnings: std::collections::BTreeMap<String, Vec<String>>,
}

/// Selected nodes whose exact body is outside the viewed closure. They are
/// validation work, never display targets.
pub(super) fn validation_targets_outside(selected: &[String], closure: &std::collections::HashSet<String>) -> Vec<String> {
    let mut targets: Vec<String> = selected
        .iter()
        .filter(|id| !closure.contains(*id))
        .cloned()
        .collect();
    targets.sort();
    targets.dedup();
    targets
}

/// Merge validation diagnostics for the current selection into the displayed
/// exact result's diagnostics. Validation only adds node errors/warnings; it
/// cannot replace the base result or contribute to any other evaluation state.
pub(super) fn merge_diagnostics(
    base_errors: &std::collections::BTreeMap<String, String>,
    base_warnings: &std::collections::BTreeMap<String, Vec<String>>,
    selection: &[String],
    results: &std::collections::HashMap<String, ValidationDiagnostics>,
) -> (
    std::collections::BTreeMap<String, String>,
    std::collections::BTreeMap<String, Vec<String>>,
) {
    let mut errors = base_errors.clone();
    let mut warnings = base_warnings.clone();
    for target in selection {
        let Some(diagnostics) = results.get(target) else {
            continue;
        };
        for (id, message) in &diagnostics.errors {
            errors.insert(id.clone(), message.clone());
        }
        for (id, messages) in &diagnostics.warnings {
            warnings
                .entry(id.clone())
                .or_default()
                .extend(messages.iter().cloned());
        }
    }
    (errors, warnings)
}

/// Identity of one exact evaluation.
///
/// The session generation distinguishes a replacement graph from the old
/// result even when its document id, revision and target all match. The
/// revision is the session's content revision, so any edit changes the key.
/// Linked-document content is folded into [`Self::linked_digest`] because it
/// lives outside the active session's revision, and runtime importer-source
/// hashes are folded into [`Self::source_digest`] for the same reason.
///
/// One key type serves every exact producer and consumer: the per-tab result
/// cache, the display quality tier, validation requests and the worker. Do not
/// add a second identity for the same evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct ExactKey {
    session_generation: u64,
    document_id: String,
    revision: u64,
    /// The viewed target, not the selected node. Selection only changes the
    /// comparison/ghost/edit inputs derived below.
    target: Option<String>,
    comparison: Option<(String, String)>,
    ghost_input: Option<String>,
    edit_preview: Option<String>,
    feature_ghost: bool,
    run_exporters: bool,
    linked_digest: u64,
    source_digest: u64,
}

impl ExactKey {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        session_generation: u64,
        document_id: String,
        revision: u64,
        target: Option<String>,
        comparison: Option<(String, String)>,
        ghost_input: Option<String>,
        edit_preview: Option<String>,
        feature_ghost: bool,
        run_exporters: bool,
        source_hashes: &std::collections::BTreeMap<String, String>,
        linked_documents: &std::collections::BTreeMap<PathBuf, GraphDocument>,
    ) -> Self {
        Self {
            session_generation,
            document_id,
            revision,
            target,
            comparison,
            ghost_input,
            edit_preview,
            feature_ghost,
            run_exporters,
            linked_digest: linked_digest(linked_documents),
            source_digest: source_digest(source_hashes),
        }
    }

    /// Whether a low-priority result still belongs to the live graph. A
    /// validation request can finish after an edit; its diagnostics must not
    /// light up nodes of the new graph.
    pub(super) fn is_current(&self, generation: u64, revision: u64) -> bool {
        self.session_generation == generation && self.revision == revision
    }

    /// Graph identity without the request-specific fields (target, comparison,
    /// ghost, edit preview). Foreground and validation keys for the same graph
    /// agree here even though their targets differ, so the worker can retire
    /// validation work as soon as a foreground job carries a different graph.
    /// `document_id` is deliberately excluded: a validation key uses the tab's
    /// document id while a bound foreground key uses its expanded eval id, and
    /// the worker only compares jobs from the same tab.
    fn same_graph(&self, other: &Self) -> bool {
        self.session_generation == other.session_generation
            && self.revision == other.revision
            && self.linked_digest == other.linked_digest
            && self.source_digest == other.source_digest
    }
}

/// Content identity of the runtime importer-source hashes an evaluation saw.
/// They are outside the document revision, exactly like linked documents.
fn source_digest(hashes: &std::collections::BTreeMap<String, String>) -> u64 {
    if hashes.is_empty() {
        return 0;
    }
    let mut hasher = DefaultHasher::new();
    for (id, hash) in hashes {
        id.hash(&mut hasher);
        hash.hash(&mut hasher);
    }
    hasher.finish()
}

/// Content identity of the linked documents a graph expands. Linked parts are
/// stored outside the active session, so their edits do not bump its revision.
fn linked_digest(documents: &std::collections::BTreeMap<PathBuf, GraphDocument>) -> u64 {
    if documents.is_empty() {
        return 0;
    }
    let bytes = serde_json::to_vec(documents).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Bounded per-tab map of exact results. A click that returns to a target this
/// tab already computed is a hit even when other targets were computed since;
/// a single last-result slot would have evicted it.
#[derive(Default)]
pub(super) struct ExactCache {
    entries: std::collections::VecDeque<(ExactKey, engine_bridge::Evaluation)>,
}

const EXACT_CACHE_ENTRIES: usize = 8;

impl ExactCache {
    pub(super) fn get(&self, key: &ExactKey) -> Option<&engine_bridge::Evaluation> {
        self.entries
            .iter()
            .find(|(existing, _)| existing == key)
            .map(|(_, evaluation)| evaluation)
    }

    pub(super) fn insert(&mut self, key: ExactKey, evaluation: engine_bridge::Evaluation) {
        if let Some((_, existing)) = self
            .entries
            .iter_mut()
            .find(|(existing, _)| existing == &key)
        {
            *existing = evaluation;
            return;
        }
        self.entries.push_back((key, evaluation));
        while self.entries.len() > EXACT_CACHE_ENTRIES {
            self.entries.pop_front();
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Inputs that only affect display packaging, never exact identity.
#[derive(Clone)]
pub(super) struct DisplayFields {
    pub feature_ghost: bool,
    pub edit_preview: Option<String>,
    pub ghost_input: Option<String>,
    pub comparison: Option<(String, String)>,
    pub run_exporters: bool,
}

#[derive(Default)]
struct TabEvalState {
    cache: EngineCache,
    flat: nodecraft_session::flat::FlatGraphCache,
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

/// Display-space bounds for one body identity at one placement.
///
/// Geometry identity of one body for the local-bounds cache.
///
/// `BodyKey` is the producing node's content key plus the body index. The exact
/// evaluator derives it from the node's content seed, so the same key names the
/// same B-rep geometry across evaluations (a Look keeps it while changing only
/// face attributes) and a geometry edit changes it. Placement is separate
/// instance data and is deliberately not in this key; the local bound is placed
/// on every use with `BodyBounds::placed`.
fn lod_bounds_key(body: &nodecraft_exact::ExactBody) -> nodecraft_display::BodyKey {
    body.key
}

/// Per-body screen-space level state that survives exact snapshot changes.
///
/// An exact result is a new snapshot of the same scene, not a new camera. A
/// colour edit therefore keeps every body's level and placed bounds; a geometry
/// edit changes only the edited body's `BodyKey` and plans only that body. The
/// levels are updated here, before the snapshot is submitted, so the display
/// worker receives each body's final quality on its first (and only) request.
#[derive(Debug, Default)]
pub(super) struct LodState {
    pub(super) levels: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::DisplayQuality>,
    /// Local (unplaced) bounds by body geometry identity. Placement changes
    /// reuse the measurement; every use places the local bound cheaply.
    bounds: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::BodyBounds>,
    /// The finest bound for each body key: a shared body can be placed several
    /// times, and the finest placement is the one that must not undersample.
    placed: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::BodyBounds>,
    /// Camera/viewport the installed levels were computed for. Equal to the
    /// next frame's view means nothing moved and no LOD request is needed.
    pub(super) view: Option<nodecraft_display::LodView>,
    #[cfg(test)]
    pub(super) bounds_built: usize,
}

/// One display request's per-body levels, ready to submit.
#[derive(Clone, Debug)]
pub(super) struct LodRequest {
    /// Quality for bodies absent from `body_quality` (headless/first frame).
    pub quality: nodecraft_display::DisplayQuality,
    pub body_quality: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::DisplayQuality>,
    pub body_priority: Vec<nodecraft_display::BodyKey>,
}

impl Default for LodRequest {
    fn default() -> Self {
        Self {
            quality: nodecraft_display::default_quality(),
            body_quality: BTreeMap::new(),
            body_priority: Vec::new(),
        }
    }
}

impl LodState {
    /// Reconcile the installed levels with a new exact snapshot.
    ///
    /// Levels for body keys that still exist are kept untouched, vanished keys
    /// are dropped, and only a new geometry (or a body whose placed bounds
    /// moved) is given a level. Local bounds are reused across every placement,
    /// so a colour-only edit and every move do no geometry sampling and no new
    /// tessellation.
    pub(super) fn reconcile(
        &mut self,
        snapshot: &nodecraft_display::service::ExactSnapshot,
        view: Option<nodecraft_display::LodView>,
    ) -> LodRequest {
        let mut seen = std::collections::BTreeSet::new();
        let mut placed: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::BodyBounds> =
            BTreeMap::new();
        let mut bounds: BTreeMap<nodecraft_display::BodyKey, nodecraft_display::BodyBounds> =
            BTreeMap::new();
        let mut moved = std::collections::BTreeSet::new();
        let parts = snapshot
            .breakpoint
            .values()
            .chain(snapshot.overlays.iter().map(|overlay| overlay.part.as_ref()));
        for part in parts {
            for body in &part.bodies {
                seen.insert(body.key);
                let key = lod_bounds_key(body);
                let local = match bounds
                    .get(&key)
                    .or_else(|| self.bounds.get(&key))
                    .copied()
                {
                    Some(cached) => cached,
                    None => {
                        let Some(local) = nodecraft_display::body_bounds(&body.brep) else {
                            continue;
                        };
                        #[cfg(test)]
                        {
                            self.bounds_built += 1;
                        }
                        local
                    }
                };
                let value = local.placed(&body.placement);
                // The level depends on the placed bound, not on the placement
                // value itself: a move that leaves every instance's bound the
                // same needs no new choice, and a new geometry always does.
                if self.placed.get(&body.key) != Some(&value) {
                    moved.insert(body.key);
                }
                bounds.insert(key, local);
                match placed.entry(body.key) {
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        if value.radius < entry.get().radius {
                            entry.insert(value);
                        }
                    }
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(value);
                    }
                }
            }
        }
        self.levels.retain(|key, _| seen.contains(key));
        self.bounds = bounds;
        self.placed = placed;

        let mut request = LodRequest::default();
        if let Some(view) = view {
            let camera = nodecraft_display::LodCamera {
                eye: view.eye,
                fov_y: view.fov_y,
                ortho_height: view.ortho_height,
            };
            let viewport = nodecraft_display::ViewportSize {
                width: view.viewport[0],
                height: view.viewport[1],
            };
            for (key, body_bounds) in &self.placed {
                let quality = match self.levels.get(key).copied() {
                    // The camera did not move on an edit: keep the installed
                    // level. Hysteresis is for camera moves, not colour edits.
                    Some(existing) if !moved.contains(key) => existing,
                    // A new body, or one whose placement changed, needs its
                    // level chosen from its current placed bounds.
                    _ => nodecraft_display::quality_for(
                        *body_bounds,
                        camera,
                        viewport,
                        view.pixels_per_point,
                        nodecraft_display::LodParams::new(view.error_px),
                    ),
                };
                self.levels.insert(*key, quality);
                request.body_quality.insert(*key, quality);
            }
            request.body_priority = self.priority(view.eye);
            // Only claim this view is installed when every body got a level. A
            // body with no usable bounds falls back to `request.quality` and
            // the next frame retries.
            self.view = (request.body_quality.len() == seen.len()).then_some(view);
        } else {
            // No camera has been fitted yet (first frame/headless). There is no
            // screen-space level to choose; use the normal full-plan tolerance
            // instead of a preview pass. Any levels already installed are still
            // reused, and the next frame with a camera fills in the rest.
            request.body_quality = self.levels.clone();
            request.body_priority = seen.into_iter().collect();
            self.view = None;
        }
        request
    }

    /// Recompute every body's level for a moved camera/viewport.
    ///
    /// This is the hysteresis path: each level jumps directly to the rung the
    /// refine/coarsen band would settle on, so a fast zoom that stops is not
    /// left one rung too fine. Bounds come from the snapshot cache; no geometry
    /// is sampled and no plan is built here.
    pub(super) fn levels_for_view(&mut self, view: nodecraft_display::LodView) -> LodRequest {
        let camera = nodecraft_display::LodCamera {
            eye: view.eye,
            fov_y: view.fov_y,
            ortho_height: view.ortho_height,
        };
        let viewport = nodecraft_display::ViewportSize {
            width: view.viewport[0],
            height: view.viewport[1],
        };
        let mut body_quality = BTreeMap::new();
        for (key, body_bounds) in &self.placed {
            let current_tolerance = match self.levels.get(key).copied() {
                Some(nodecraft_display::DisplayQuality::Full(tolerance)) => Some(tolerance),
                _ => None,
            };
            let quality = nodecraft_display::quality_for(
                *body_bounds,
                camera,
                viewport,
                view.pixels_per_point,
                nodecraft_display::LodParams {
                    current_tolerance,
                    ..nodecraft_display::LodParams::new(view.error_px)
                },
            );
            body_quality.insert(*key, quality);
        }
        self.levels = body_quality.clone();
        self.view = Some(view);
        LodRequest {
            quality: nodecraft_display::default_quality(),
            body_quality,
            body_priority: self.priority(view.eye),
        }
    }

    /// Nearest-first planning order. The distance is centre depth minus radius,
    /// so a large body that reaches the camera is planned early too.
    fn priority(&self, eye: [f64; 3]) -> Vec<nodecraft_display::BodyKey> {
        let mut order: Vec<(f64, nodecraft_display::BodyKey)> = self
            .placed
            .iter()
            .map(|(key, bounds)| {
                let distance = ((eye[0] - bounds.center[0]).powi(2)
                    + (eye[1] - bounds.center[1]).powi(2)
                    + (eye[2] - bounds.center[2]).powi(2))
                .sqrt()
                    - bounds.radius;
                (distance, *key)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        order.into_iter().map(|(_, key)| key).collect()
    }
}

/// Position of one display frame in the order the viewer shows geometry.
///
/// `snapshot_id` is the exact snapshot the frame planned: a newer exact result
/// always wins. `revision` is the display worker's submission sequence and only
/// breaks ties between re-plans of the same snapshot (quality tiers). Exact and
/// quality requests share one monotonically increasing revision sequence, so
/// revision order is submission order; it must never be used alone because a
/// quality frame for an older snapshot still shows older geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct DisplayOrder {
    pub snapshot_id: u64,
    pub revision: u64,
}

impl DisplayOrder {
    pub(super) fn of(frame: &nodecraft_display::service::DisplayFrame) -> Self {
        Self { snapshot_id: frame.snapshot_id, revision: frame.revision }
    }

    /// The app has already installed a newer exact snapshot, so this frame's
    /// geometry is obsolete no matter when the worker happened to publish it.
    pub(super) fn is_stale(self, installed: Option<u64>) -> bool {
        installed.is_some_and(|installed| self.snapshot_id < installed)
    }

    pub(super) fn supersedes(self, other: Self) -> bool {
        self > other
    }
}

/// Build the exact snapshot for one exact result. The exact result is the
/// display service's input; the service owns plan-cache and GPU de-duplication.
pub(super) fn build_display(
    exact: &engine_bridge::Evaluation,
    edit_preview: Option<&str>,
    ghost_input: Option<&str>,
    comparison: Option<(&str, &str)>,
) -> std::sync::Arc<nodecraft_display::service::ExactSnapshot> {
    use nodecraft_display::service::{ExactSnapshot, OverlayKind, OverlayRequest};
    let Some(part) = exact.exact_parts.get(&exact.target) else {
        return std::sync::Arc::new(ExactSnapshot::default());
    };
    let mut overlays = Vec::new();
    let mut push = |kind: OverlayKind, part: &nodecraft_exact::ExactPart| {
        overlays.push(OverlayRequest {
            kind,
            part: std::sync::Arc::new(part.clone()),
        });
    };
    if let Some(edit) = edit_preview
        && let Some(part) = exact.exact_parts.get(edit)
    {
        push(OverlayKind::EditInput, part);
    }
    if let Some(ghost) = ghost_input
        && let Some(part) = exact.exact_parts.get(ghost)
    {
        push(OverlayKind::FeatureGhost, part);
    }
    if let Some((before, after)) = comparison {
        for (id, kind) in [
            (before, OverlayKind::ComparisonBefore),
            (after, OverlayKind::ComparisonAfter),
        ] {
            if let Some(part) = exact.exact_parts.get(id) {
                push(kind, part);
            }
        }
    }
    if let Some(difference) = exact.exact_comparison.as_ref() {
        if let (Some(added), Some(key)) = (difference.added.as_ref(), difference.added_key) {
            push(
                OverlayKind::DifferenceAdded,
                &nodecraft_exact::ExactPart::singleton(key, std::sync::Arc::clone(added)),
            );
        }
        if let (Some(removed), Some(key)) = (difference.removed.as_ref(), difference.removed_key) {
            push(
                OverlayKind::DifferenceRemoved,
                &nodecraft_exact::ExactPart::singleton(key, std::sync::Arc::clone(removed)),
            );
        }
    }
    let breakpoint = vec![(exact.target.clone(), part.clone())].into_iter().collect();
    std::sync::Arc::new(ExactSnapshot {
        breakpoint,
        overlays,
    })
}

enum WorkerMessage {
    Foreground(EvalJob),
    Validation(ValidationJob),
}

/// The worker's ownership of everything taken off the job channel. Messages are
/// only ever run, kept here, or retired as superseded by a newer request of the
/// same kind and identity. Every channel take goes through [`Self::drain`] or
/// [`Self::fold`]; no other site removes a message from the channel.
#[derive(Default)]
struct PendingJobs {
    foreground: Option<EvalJob>,
    validations: std::collections::VecDeque<ValidationJob>,
}

impl PendingJobs {
    /// Fold one message into pending state. A newer foreground job supersedes
    /// an older one. A newer validation request supersedes an older one only
    /// when it names the same tab, document and target; different targets stay
    /// queued. Everything superseded is explicitly retired to the app.
    fn fold(
        &mut self,
        message: WorkerMessage,
        retired_tx: &Sender<(u64, u64)>,
    ) {
        match message {
            WorkerMessage::Foreground(mut job) => {
                #[cfg(test)]
                test_support::record_received(test_support::FOREGROUND);
                // Only the newest foreground request matters, but an explicit
                // clear request must not be lost when a newer edit coalesces
                // into it.
                if let Some(previous) = self.foreground.take() {
                    job.clear_cache |= previous.clear_cache;
                }
                self.foreground = Some(job);
                self.retire_stale_validations(retired_tx);
            }
            WorkerMessage::Validation(job) => {
                #[cfg(test)]
                test_support::record_received(test_support::VALIDATION);
                if let Some(index) = self
                    .validations
                    .iter()
                    .position(|pending| pending.same_request(&job))
                {
                    let superseded = self
                        .validations
                        .remove(index)
                        .expect("the validation index came from this queue");
                    let _ = retired_tx.send((superseded.id, superseded.tab_id));
                }
                self.validations.push_back(job);
                self.retire_stale_validations(retired_tx);
            }
        }
    }

    /// Take everything queued and fold it into pending foreground and pending
    /// validation work. This is the one drain helper every take site uses.
    fn drain(&mut self, rx: &Receiver<WorkerMessage>, retired_tx: &Sender<(u64, u64)>) {
        while let Ok(message) = rx.try_recv() {
            self.fold(message, retired_tx);
        }
    }

    /// A foreground job for a tab carries the graph the user is now looking at.
    /// Validation work for the same tab and a different graph revision would
    /// run against the wrong graph; retire it instead of running it.
    fn retire_stale_validations(&mut self, retired_tx: &Sender<(u64, u64)>) {
        let Some(foreground) = self.foreground.as_ref() else {
            return;
        };
        let mut index = 0;
        while index < self.validations.len() {
            let stale = {
                let validation = &self.validations[index];
                validation.tab_id == foreground.tab_id
                    && !validation.key.same_graph(&foreground.key)
            };
            if stale {
                let validation = self
                    .validations
                    .remove(index)
                    .expect("the validation index came from this queue");
                let _ = retired_tx.send((validation.id, validation.tab_id));
            } else {
                index += 1;
            }
        }
    }

    /// The channel is gone, so nothing queued can ever run. Retire every
    /// outstanding validation request so the app's pending map cannot grow.
    fn retire_all_validations(&mut self, retired_tx: &Sender<(u64, u64)>) {
        for validation in self.validations.drain(..) {
            let _ = retired_tx.send((validation.id, validation.tab_id));
        }
    }
}

/// A result with no diagnostics, used when a validation request cannot produce
/// an exact evaluation. Every request must retire its app bookkeeping, so the
/// worker sends one of these instead of silently sending nothing.
fn empty_validation_evaluation(
    target: String,
    error: Option<String>,
) -> engine_bridge::Evaluation {
    engine_bridge::Evaluation {
        target,
        difference_error: None,
        difference_approximate: false,
        node_errors: Default::default(),
        node_warnings: Default::default(),
        error,
        elapsed_ms: 0.0,
        kernel_ms: 0.0,
        cache_hits: 0,
        cache_misses: 0,
        brep_hits: 0,
        part_hits: 0,
        exact_parts: Default::default(),
        exact_comparison: None,
        proof_generation: 0,
    }
}

pub(super) struct EvalWorker {
    tx: Sender<WorkerMessage>,
    rx: Receiver<(u64, u64, engine_bridge::Evaluation)>,
    validation_rx: Receiver<(u64, u64, engine_bridge::Evaluation)>,
    validation_retired_rx: Receiver<(u64, u64)>,
    progress_rx: Receiver<(u64, u64, nodecraft_engine::ProgressEvent)>,
    active_token: std::sync::Arc<std::sync::Mutex<Option<nodecraft_engine::kernel_pool::RevisionToken>>>,
}

impl EvalWorker {
    pub(super) fn new() -> Self {
        let (job_tx, job_rx) = channel::<WorkerMessage>();
        let (result_tx, result_rx) = channel();
        let (validation_tx, validation_rx) = channel::<(u64, u64, engine_bridge::Evaluation)>();
        let (validation_retired_tx, validation_retired_rx) = channel::<(u64, u64)>();
        let (progress_tx, progress_rx) = channel();
        std::thread::Builder::new()
            .name("nodecraft-eval".into())
            .stack_size(nodecraft_kernel_profile::KERNEL_THREAD_STACK)
            .spawn(move || {
            // Cache ownership follows the tab/document, not the worker. This keeps
            // per-document dirty/cache lifetimes aligned and prevents one tab from
            // evicting or reusing another tab's entries. Exact results are cached
            // on the app side by tab, so this thread only owns the engine caches.
            /// Evaluate one selected node outside the viewed closure. This runs
            /// only when no foreground job is queued, on the same worker and
            /// cache as evaluation, so it never delays the display exact pass.
            fn run_validation(
                job: ValidationJob,
                states: &mut std::collections::HashMap<String, TabEvalState>,
                result_tx: &Sender<(u64, u64, engine_bridge::Evaluation)>,
            ) {
                let mut state = states.remove(&job.document_id).unwrap_or_default();
                nodecraft_engine::kernel_pool::set_current_revision_token(job.token.clone());
                let flat = match state.flat.resolve(&job.document, &job.linked_documents) {
                    Ok(flat) => flat.clone(),
                    Err(error) => {
                        let evaluation = empty_validation_evaluation(
                            job.target.clone(),
                            Some(error.to_string()),
                        );
                        states.insert(job.document_id.clone(), state);
                        let _ = result_tx.send((job.id, job.tab_id, evaluation));
                        return;
                    }
                };
                let mut cache = state.cache;
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    nodecraft_engine::kernel_pool::with_current_token(|| {
                        engine_bridge::evaluate_flat_exact_with_imports(
                            flat.document(),
                            job.target.clone(),
                            cache,
                            None,
                            None,
                            false,
                            &Default::default(),
                            None,
                            &[],
                            None,
                            false,
                            &job.imports,
                        )
                    })
                }));
                match outcome {
                    Ok((evaluation, next_cache)) => {
                        state.cache = next_cache;
                        states.insert(job.document_id.clone(), state);
                        let evaluation = evaluation.unwrap_or_else(|| {
                            empty_validation_evaluation(
                                job.target.clone(),
                                Some("Validation produced no exact result".into()),
                            )
                        });
                        let _ = result_tx.send((job.id, job.tab_id, evaluation));
                    }
                    Err(_) => {
                        state.cache = EngineCache::new();
                        states.insert(job.document_id.clone(), state);
                        let evaluation = empty_validation_evaluation(
                            job.target.clone(),
                            Some("Engine panicked during validation".into()),
                        );
                        let _ = result_tx.send((job.id, job.tab_id, evaluation));
                    }
                }
            }
            let mut states: std::collections::HashMap<String, TabEvalState> =
                std::collections::HashMap::new();
            let mut pending = PendingJobs::default();
            'jobs: loop {
                // Every message taken off the channel is folded into pending
                // state first. Nothing is discarded here: superseded work is
                // retired to the app inside `fold`.
                pending.drain(&job_rx, &validation_retired_tx);
                if pending.foreground.is_none() && pending.validations.is_empty() {
                    match job_rx.recv() {
                        Ok(message) => pending.fold(message, &validation_retired_tx),
                        Err(_) => {
                            pending.retire_all_validations(&validation_retired_tx);
                            break;
                        }
                    }
                }
                // Foreground evaluation always wins. A queued validation request
                // runs only when the foreground channel has no work, mirroring
                // the kernel pool's low-priority channel.
                let Some(job) = pending.foreground.take() else {
                    // Take one more look before starting low-priority work so a
                    // foreground request that arrived after the drain runs first.
                    pending.drain(&job_rx, &validation_retired_tx);
                    if pending.foreground.is_some() {
                        continue 'jobs;
                    }
                    if let Some(validation) = pending.validations.pop_front() {
                        run_validation(validation, &mut states, &validation_tx);
                    }
                    continue 'jobs;
                };
                let mut state = states.remove(&job.document_id).unwrap_or_default();
                let timeline = std::time::Instant::now();
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[t {}] worker picked", job.id);
                }
                #[cfg(test)]
                test_support::after_pick();
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
                            node_warnings: Default::default(),
                            error: Some(error.to_string()),
                            elapsed_ms: 0.0,
                            kernel_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                            brep_hits: 0,
                            part_hits: 0,
                            exact_parts: Default::default(),
                            exact_comparison: None,
                            proof_generation: 0,
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
                // The session map is only valid for the exact document it was
                // computed from. Flattening is a clone for a graph with no
                // linked/inline expansion, so require the same node and
                // connection counts; otherwise the bridge hashes its own flat
                // document (one pass, correctness first).
                let session_hashes = job.session_hashes.as_deref().filter(|_| {
                    flat.document().nodes.len() == job.document.nodes.len()
                        && flat.document().connections.len() == job.document.connections.len()
                });
                let exact_outcome = catch_unwind(AssertUnwindSafe(|| {
                    nodecraft_engine::kernel_pool::with_current_token(|| engine_bridge::evaluate_flat_exact_with_imports_and_hashes(
                        flat.document(),
                        target.clone().unwrap_or_default(),
                        cache,
                        Some(&mut on_progress),
                        job.comparison.as_ref().map(|(before, after)| (before.as_str(), after.as_str())),
                        job.run_exporters,
                        &job.dirty_nodes,
                        job.ghost_input.as_deref(),
                        &[],
                        job.edit_preview.as_deref(),
                        job.clear_cache,
                        &job.imports,
                        session_hashes,
                    ))
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
                            node_warnings: Default::default(),
                            error: Some("Engine panicked during exact evaluation".into()),
                            elapsed_ms: 0.0,
                            kernel_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                            brep_hits: 0,
                            part_hits: 0,
                            exact_parts: Default::default(),
                            exact_comparison: None,
                            proof_generation: 0,
                        });
                        state.cache = cache;
                        states.insert(job.document_id.clone(), state);
                        if let Some(evaluation) = evaluation
                            && result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                        continue;
                    }
                };
                if std::env::var_os("NODECRAFT_TIMELINE").is_some()
                    && let Some(exact) = exact.as_ref()
                {
                    eprintln!(
                        "[t {}] exact {:?} hits {} misses {} parts {}",
                        job.id,
                        timeline.elapsed(),
                        exact.cache_hits,
                        exact.cache_misses,
                        exact.exact_parts.len(),
                    );
                }
                let mut evaluation = match exact {
                    Some(evaluation) => evaluation,
                    None => {
                        state.cache = cache;
                        states.insert(job.document_id.clone(), state);
                        continue;
                    }
                };
                if job.run_exporters {
                    // The engine runs the exporter's effect once and records
                    // any write/proof error on the exporter node. The app must
                    // not write a second copy.
                    evaluation.exact_parts.clear();
                    state.cache = cache;
                    states.insert(job.document_id.clone(), state);
                    if result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                    continue 'jobs;
                }
                // A newer edit may have arrived while the exact pass ran. Fold
                // everything queued into the same pending state, publish this
                // exact result, abandon the now-stale display pass, and let the
                // newest foreground job use the warmed cache. Validation work
                // taken here stays pending instead of being dropped.
                pending.drain(&job_rx, &validation_retired_tx);
                if pending.foreground.is_some() {
                    state.cache = cache;
                    states.insert(job.document_id.clone(), state);
                    if result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                    if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                        eprintln!("[t {}] snapshot sent +{:?}", job.id, timeline.elapsed());
                    }
                    continue 'jobs;
                }
                state.cache = cache;
                states.insert(job.document_id.clone(), state);
                if result_tx.send((job.id, job.tab_id, evaluation)).is_err() { break; }
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[t {}] snapshot sent +{:?}", job.id, timeline.elapsed());
                }
                if states.len() > 16
                    && let Some(oldest) = states.keys().next().cloned()
                {
                    states.remove(&oldest);
                }
            }
        })
        .expect("spawn the evaluation thread");
        Self {
            tx: job_tx,
            rx: result_rx,
            validation_rx,
            validation_retired_rx,
            progress_rx,
            active_token: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub(super) fn cancel_active(&self) {
        if let Ok(mut active) = self.active_token.lock()
            && let Some(token) = active.take()
        {
            token.cancel();
        }
    }

    pub(super) fn send(&self, job: EvalJob) {
        // Validation work keeps its own token. Sending a foreground job only
        // queues it ahead of that work; it must not cancel a validation request
        // that is still waiting to run, or its diagnostics would come back
        // suppressed. Superseded validation is removed by the worker before it
        // can run.
        if let Ok(mut active) = self.active_token.lock() {
            if let Some(previous) = active.take() {
                previous.cancel();
            }
            *active = Some(job.token.clone());
        }
        let _ = self.tx.send(WorkerMessage::Foreground(job));
    }

    pub(super) fn send_validation(&self, job: ValidationJob) {
        // A validation request with a different target is independent work and
        // must run with a live token. The worker retires same-target requests it
        // supersedes, so this send never preempts a request that will still run.
        let _ = self.tx.send(WorkerMessage::Validation(job));
    }

    /// Submit a foreground evaluation and the validations it owns through the
    /// one producer. The channel is FIFO, so enqueueing the foreground job
    /// first is the priority guarantee: an idle worker cannot start a
    /// low-priority validation before the user's evaluation.
    pub(super) fn send_evaluation(
        &self,
        job: EvalJob,
        validations: Vec<ValidationJob>,
    ) {
        self.send(job);
        for validation in validations {
            self.send_validation(validation);
        }
    }

    pub(super) fn try_recv(&self) -> Option<(u64, u64, engine_bridge::Evaluation)> {
        self.rx.try_recv().ok()
    }

    pub(super) fn try_recv_validation(&self) -> Option<(u64, u64, engine_bridge::Evaluation)> {
        self.validation_rx.try_recv().ok()
    }

    /// Validation requests the worker retired as superseded. The app removes
    /// their bookkeeping without waiting for a result that will never arrive.
    pub(super) fn try_recv_retired_validation(&self) -> Option<(u64, u64)> {
        self.validation_retired_rx.try_recv().ok()
    }

    pub(super) fn try_recv_progress(&self) -> Option<(u64, u64, nodecraft_engine::ProgressEvent)> {
        self.progress_rx.try_recv().ok()
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
    pub(super) fn finish_remaining_node_activity(&mut self) {
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
        self.base_node_errors = map_eval_errors(binding, &evaluation.node_errors);
        self.proof_generation = Some(evaluation.proof_generation);
        self.base_node_warnings = map_eval_warnings(binding, &evaluation.node_warnings);
        // The displayed result replaces the base diagnostics; validation
        // diagnostics for the current selection are merged on top.
        self.refresh_diagnostics();
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

    /// Hand the exact result to the display service. This is the app's only
    /// display submission path; it never predicts whether the service needs the
    /// snapshot.
    #[allow(clippy::too_many_arguments)]
    fn submit_exact_display(
        &mut self,
        evaluation: &engine_bridge::Evaluation,
        feature_ghost: bool,
        edit_preview: Option<&str>,
        ghost_input: Option<&str>,
        comparison: Option<(&str, &str)>,
        run_exporters: bool,
    ) {
        if run_exporters {
            return;
        }
        let snapshot = build_display(evaluation, edit_preview, ghost_input, comparison);
        let view = self.viewport.lod_view(self.egui_ctx.pixels_per_point());
        // Reconcile the display-only levels with this snapshot before the
        // submission. A colour edit keeps both the levels and the placed
        // bounds; a geometry edit changes only the edited body's key. The
        // display worker therefore receives every body's final level on this
        // one request instead of a coarse preview followed by a quality tier.
        let lod = self.lod.reconcile(&snapshot, view);
        let submitted = submit_exact_snapshot(
            DisplaySubmission {
                evaluation,
                feature_ghost,
                snapshot,
                run_exporters,
                quality: lod.quality,
                body_quality: lod.body_quality,
                body_priority: lod.body_priority,
            },
            &self.display_worker,
            &mut self.display_next,
            &mut self.display_snapshot,
            &mut self.pending_display,
            &mut self.quality_display,
        );
        if submitted {
            self.egui_ctx.request_repaint();
        }
    }

    /// Schedule the low-priority exact pass for selected nodes outside the
    /// viewed closure. These targets are never part of the display request, so
    /// a cold body here cannot delay the frame the user is looking at. A target
    /// already in the exact map reuses its result and costs no job.
    #[allow(clippy::too_many_arguments)]
    fn schedule_validation(
        &mut self,
        tab_id: u64,
        document_id: &str,
        linked_documents: &std::sync::Arc<
            std::collections::BTreeMap<PathBuf, GraphDocument>,
        >,
        document: Option<std::sync::Arc<GraphDocument>>,
        targets: &[String],
        binding: &Option<EvalBinding>,
        revision: u64,
    ) {
        for job in self.take_validation_jobs(
            tab_id,
            document_id,
            linked_documents,
            document,
            targets,
            binding,
            revision,
        ) {
            self.eval_worker.send_validation(job);
        }
    }

    /// Build the low-priority validation jobs for a foreground request without
    /// sending them. The caller submits the foreground job first, then this
    /// batch, so channel order is the worker's priority order.
    #[allow(clippy::too_many_arguments)]
    fn take_validation_jobs(
        &mut self,
        tab_id: u64,
        document_id: &str,
        linked_documents: &std::sync::Arc<
            std::collections::BTreeMap<PathBuf, GraphDocument>,
        >,
        document: Option<std::sync::Arc<GraphDocument>>,
        targets: &[String],
        binding: &Option<EvalBinding>,
        revision: u64,
    ) -> Vec<ValidationJob> {
        let Some(document) = document else { return Vec::new() };
        let source_hashes = self.source_hashes(&document);
        let imports = self.import_context_with(&document, source_hashes.clone());
        let mut jobs = Vec::new();
        for target in targets {
            let key = ExactKey::new(
                self.session_generation,
                document_id.to_owned(),
                revision,
                Some(target.clone()),
                None,
                None,
                None,
                false,
                false,
                &source_hashes,
                linked_documents,
            );
            let body_target = binding
                .as_ref()
                .and_then(|binding| map_eval_id_to_body(binding, target))
                .unwrap_or_else(|| target.clone());
            if let Some(cached) = self
                .exact_results
                .get(&tab_id)
                .and_then(|cache| cache.get(&key))
                .cloned()
            {
                // Cached nodes cost nothing: reuse their diagnostics even when
                // the foreground result on screen did not include this target.
                self.validation_results.insert(
                    body_target,
                    ValidationDiagnostics {
                        errors: map_eval_errors(binding.as_ref(), &cached.node_errors),
                        warnings: map_eval_warnings(binding.as_ref(), &cached.node_warnings),
                    },
                );
                continue;
            }
            if !self.validation_requested.insert(key.clone()) {
                continue;
            }
            self.validation_next = self.validation_next.wrapping_add(1);
            let id = self.validation_next;
            self.validation_pending.insert(
                id,
                ValidationPending {
                    tab_id,
                    body_target,
                    key: key.clone(),
                },
            );
            jobs.push(ValidationJob {
                id,
                tab_id,
                document_id: document_id.to_owned(),
                linked_documents: std::sync::Arc::clone(linked_documents),
                document: std::sync::Arc::clone(&document),
                imports: imports.clone(),
                target: target.clone(),
                key,
                token: nodecraft_engine::kernel_pool::RevisionToken::new(),
            });
        }
        self.refresh_diagnostics();
        jobs
    }

    /// Merge the displayed exact result's diagnostics with the currently
    /// selected validation targets. Validation may only change node_errors and
    /// node_warnings (the canvas error glow); it must not touch status,
    /// progress, activity, last_exact or display state.
    pub(super) fn refresh_diagnostics(&mut self) {
        let (errors, warnings) = merge_diagnostics(
            &self.base_node_errors,
            &self.base_node_warnings,
            self.validation_selection.as_deref().unwrap_or(&[]),
            &self.validation_results,
        );
        self.node_errors = errors.keys().cloned().collect();
        self.node_error_messages = errors;
        self.node_warnings = warnings;
        self.node_editor.set_error_nodes(self.node_errors.clone());
        let mut warning_nodes: HashSet<String> = self.node_warnings.keys().cloned().collect();
        if let Some(document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone())
            && let Some(changed) = self.source_changes.changed(&document_id)
        {
            warning_nodes.extend(changed.iter().cloned());
        }
        self.node_editor.set_warning_nodes(warning_nodes);
    }

    /// Apply a display frame. Owns viewport geometry and overlays only; node and
    /// evaluation state belong to `apply_exact_result`.
    pub(super) fn apply_display_frame(
        &mut self,
        frame: &nodecraft_display::service::DisplayFrame,
        target: &str,
        _feature_ghost: bool,
        force_preserve_camera: bool,
    ) -> bool {
        let apply_started = std::time::Instant::now();
        let applied = DisplayOrder::of(frame);
        // The viewer shows one exact snapshot. A frame planned from an older
        // snapshot is obsolete even if its staged plan set is only now
        // activating. Never let it replace the newer exact snapshot the app
        // has already installed.
        if applied.is_stale(self.display_snapshot.as_ref().map(|snapshot| snapshot.id)) {
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!(
                    "[t {}] drop stale frame snapshot={} installed={}",
                    frame.revision,
                    frame.snapshot_id,
                    self.display_snapshot.as_ref().map(|snapshot| snapshot.id).unwrap_or(0),
                );
            }
            return false;
        }
        // An explicit breakpoint activation is a request to see this target's
        // current geometry. It frames the camera even when the target was
        // already displayed, which matters after an upstream edit moved the
        // body (for example a suppression changing the placement chain).
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let explicit_view = view_request_matches(
            self.view_request.as_ref(),
            active_tab_id,
            target,
            frame.revision,
        );
        if explicit_view {
            self.view_request = None;
        }
        let binding = self.active_eval_binding();
        let display_target = binding
            .as_ref()
            .and_then(|binding| map_eval_id_to_body(binding, target))
            .unwrap_or_else(|| target.to_owned());
        // An authoritative display retires the drag preview only when it
        // activates with the new geometry. A live drag keeps its preview over
        // the newly arrived display instead.
        let retire_preview = !self.gizmo_dragging && self.gizmo_drag.is_none();
        let Some(planned) = frame.breakpoint.get(target) else {
            self.mesh = None;
            self.viewport.set_display(
                Vec::new(),
                None,
                Vec::new(),
                &display_target,
                frame.snapshot_id,
                retire_preview,
            );
            self.egui_ctx.request_repaint();
            return false;
        };
        let mut plans = planned.plans.clone();
        for plan in &mut plans {
            plan.target.clone_from(&target.to_owned());
        }
        let had_mesh = self.mesh.is_some();
        let same_target = self.viewport.displayed_target() == Some(display_target.as_str());
        self.mesh = Some(planned.bounds_mesh.clone());
        let diagnostics = planned
            .stats
            .iter()
            .flat_map(|stats| stats.diagnostics.iter().cloned())
            .collect();
        // One hand-off: plans, edge/picking display, diagnostics and target
        // switch together. A staged plan set stages this payload with it.
        self.viewport.set_display(
            plans,
            self.mesh.as_ref(),
            diagnostics,
            &display_target,
            frame.snapshot_id,
            retire_preview,
        );
        let preserve = force_preserve_camera
            || mesh_preserves_camera(
                self.camera_initialized,
                had_mesh,
                same_target,
                std::mem::take(&mut self.preserve_view_for_next_mesh),
                explicit_view,
            );
        // No immediate clear here: `set_display` carries the retirement with
        // the staged geometry so the preview survives until the new mesh and
        // edges activate together.

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
            self.preserve_view_for_next_mesh = true;
            self.camera_initialized = true;
        } else if !preserve {
            self.viewport.fit();
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
        if let Some((mesh, plans)) = edit_overlay {
            let edit_started = std::time::Instant::now();
            self.edit_preview = Some(mesh);
            self.edit_preview_target = None;
            if let Some(preview_target) = self.edit_target.as_ref().and_then(|id| {
                self.session
                    .as_ref()
                    .and_then(|session| edit_preview_target_for(session.document(), id))
            }) {
                self.edit_preview_target = Some(preview_target);
            }
            self.viewport.set_edit_ghost_plans(plans);
            self.attach_pending_edit_ghost();
            self.refresh_edit_overlay();
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!("[t {}] apply edit overlay {}ms", frame.revision, edit_started.elapsed().as_millis());
            }
        }
        // The gate compares against the preview target, so it must run after the
        // edit-input overlay has installed it.
        self.refresh_topology_picking();
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {}] apply frame {}ms", frame.revision, apply_started.elapsed().as_millis());
        }
        self.last_applied_display = Some((
            std::sync::Arc::new(frame.clone()),
            target.to_owned(),
            _feature_ghost,
        ));
        self.egui_ctx.request_repaint();
        true
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
        // A newer exact snapshot means this quality frame plans old geometry;
        // apply_display_frame owns that ordering and drops it. The waiter is
        // still retired here so a dropped frame cannot leave it pending.
        let frame = self
            .display_worker
            .latest_frame()
            .filter(|frame| frame.revision == revision && frame.snapshot_id == snapshot_id);
        let Some(frame) = frame else { return };
        let _ = started;
        self.quality_display = None;
        if let Some(error) = &frame.error {
            self.display_error = Some(error.clone());
            self.status = format!("Display failed: {error}");
            // Let the next update retry the per-body levels instead of
            // treating the failed level set as installed.
            self.lod.view = None;
            self.egui_ctx.request_repaint();
            return;
        }
        let started = std::time::Instant::now();
        if self.apply_display_frame(&frame, &target, feature_ghost, false) {
            let apply_us = started.elapsed().as_micros() as u64;
            self.last_display_us = Some((frame.plan_us, apply_us));
            self.profiler_record(
                crate::gpu_profiler::Series::DisplayApply,
                apply_us as f64 / 1000.0,
            );
            self.profiler_record(
                crate::gpu_profiler::Series::PlanBuild,
                frame.plan_us as f64 / 1000.0,
            );
        } else {
            // A stale or unapplicable frame did not install its plans. Recompute
            // the view level on the next frame so the request retries.
            self.lod.view = None;
        }
    }

    /// Poll the display service for the exact pre-pass's frame.
    fn poll_display_frame(&mut self) {
        let Some((id, target, feature_ghost, started)) = self.pending_display.clone() else { return };
        let frame = self
            .display_worker
            .latest_frame()
            .filter(|frame| frame.revision == id && frame.snapshot_id == id);
        let Some(frame) = frame else { return };
        let _ = started;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[t {id}] frame polled plan={}us apply next", frame.plan_us);
        }
        self.pending_display = None;
        if let Some(error) = &frame.error {
            self.display_error = Some(error.clone());
            self.status = format!("Display failed: {error}");
            // The installed view did not produce a frame. Let the next update
            // recompute the per-body levels and retry instead of treating the
            // failed level set as applied.
            self.lod.view = None;
            self.egui_ctx.request_repaint();
            return;
        }
        let started = std::time::Instant::now();
        self.apply_display_frame(&frame, &target, feature_ghost, false);
        let apply_us = started.elapsed().as_micros() as u64;
        self.last_display_us = Some((frame.plan_us, apply_us));
        self.profiler_record(
            crate::gpu_profiler::Series::DisplayApply,
            apply_us as f64 / 1000.0,
        );
        self.profiler_record(
            crate::gpu_profiler::Series::PlanBuild,
            frame.plan_us as f64 / 1000.0,
        );
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
        self.poll_quality_display();
        self.poll_display_frame();
        let revision = self.session.as_ref().map(Session::revision).unwrap_or(0);
        while let Some((id, tab_id, evaluation)) = self.eval_worker.try_recv() {
            if Some(id) != self.eval_pending || tab_id != active_tab_id {
                continue;
            }
            let Some((key, fields)) = self.pending_exact.take() else {
                continue;
            };
            self.exact_results
                .entry(tab_id)
                .or_default()
                .insert(key, evaluation.clone());
            self.last_exact = Some((revision, evaluation.clone()));
            self.apply_exact_result(binding.as_ref(), &evaluation);
            self.submit_exact_display(
                &evaluation,
                fields.feature_ghost,
                fields.edit_preview.as_deref(),
                fields.ghost_input.as_deref(),
                fields
                    .comparison
                    .as_ref()
                    .map(|(before, after)| (before.as_str(), after.as_str())),
                fields.run_exporters,
            );
            self.egui_ctx.request_repaint();
        }
        // A superseded validation request will never produce a result. The
        // worker retires it explicitly so its app bookkeeping cannot leak.
        while let Some((id, _tab_id)) = self.eval_worker.try_recv_retired_validation() {
            take_validation_request(
                &mut self.validation_pending,
                &mut self.validation_requested,
                id,
            );
        }
        // Validation results are later, low-priority work: merge diagnostics
        // only. They never become the displayed result, progress or activity.
        while let Some((id, tab_id, evaluation)) = self.eval_worker.try_recv_validation() {
            let Some(pending) = take_validation_request(
                &mut self.validation_pending,
                &mut self.validation_requested,
                id,
            ) else {
                continue;
            };
            if tab_id != active_tab_id || pending.tab_id != active_tab_id {
                continue;
            }
            let revision = self.session.as_ref().map(Session::revision).unwrap_or(0);
            if !pending.key.is_current(self.session_generation, revision) {
                continue;
            }
            self.validation_results.insert(
                pending.body_target,
                ValidationDiagnostics {
                    errors: map_eval_errors(binding.as_ref(), &evaluation.node_errors),
                    warnings: map_eval_warnings(binding.as_ref(), &evaluation.node_warnings),
                },
            );
            // Warm the exact cache so viewing the validated node later is a hit.
            self.exact_results
                .entry(tab_id)
                .or_default()
                .insert(pending.key, evaluation);
            self.refresh_diagnostics();
            self.egui_ctx.request_repaint();
        }
        self.apply_proof_reports();
    }

    /// Deliver the background proof queue's later results: a refused body marks its producer
    /// node with the failed test and every downstream node with a warning. The queue finishes
    /// after the exact result, so this runs from the same per-frame poll that already takes
    /// progress and display results. A report is applied only when its generation is the
    /// result on screen; an older graph revision's report is dropped.
    fn apply_proof_reports(&mut self) {
        if self.eval_pending.is_some() {
            // A new exact pass owns the display; the queue keeps its reports for this poll
            // once that result has been applied.
            return;
        }
        let Some(generation) = self.proof_generation else {
            return;
        };
        let reports = nodecraft_engine::proof_queue::take_reports();
        if reports.is_empty() {
            return;
        }
        let binding = self.active_eval_binding();
        let map = |id: &str| {
            binding
                .as_ref()
                .and_then(|binding| map_eval_id_to_body(binding, id))
                .unwrap_or_else(|| id.to_owned())
        };
        let mut changed = false;
        for report in reports {
            if report.generation != generation {
                continue;
            }
            let producer = map(&report.producer);
            let message = format!("Proof refused: {}", report.failure);
            self.base_node_errors.insert(producer.clone(), message);
            for downstream in &report.downstream {
                let id = map(downstream);
                self.base_node_warnings
                    .entry(id)
                    .or_default()
                    .push(format!("Built on an invalid body from {producer}"));
            }
            changed = true;
        }
        if changed {
            self.refresh_diagnostics();
            self.egui_ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn box_document() -> GraphDocument {
        let mut document = GraphDocument::new("worker-test");
        for id in ["p", "q"] {
            document.nodes.push(nodecraft_session::GraphNode {
                expressions: Default::default(),
                id: id.to_owned(),
                node_type: "primitive".to_owned(),
                values: serde_json::json!({
                    "primitiveType": "box",
                    "length": 10,
                    "width": 10,
                    "height": 10,
                }),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            });
        }
        document
    }

    /// A graph with a node the engine always refuses, so a validation result
    /// carries diagnostics that a cancelled token would suppress.
    fn error_document() -> GraphDocument {
        let mut document = box_document();
        document.nodes.push(nodecraft_session::GraphNode {
            expressions: Default::default(),
            id: "bad".to_owned(),
            node_type: "fillet".to_owned(),
            values: serde_json::json!({
                "length": 2.0,
                "edges": [{"ordinal": 99}],
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.connections.push(nodecraft_session::GraphConnection {
            from: "p".to_owned(),
            to: "bad".to_owned(),
            input: "shape".to_owned(),
            out: "shape".to_owned(),
        });
        document
    }

    fn key_at(target: &str, revision: u64) -> ExactKey {
        ExactKey::new(
            1,
            "worker-test".to_owned(),
            revision,
            Some(target.to_owned()),
            None,
            None,
            None,
            false,
            false,
            &Default::default(),
            &Default::default(),
        )
    }

    fn job(id: u64) -> EvalJob {
        EvalJob {
            id,
            tab_id: 1,
            document_id: "worker-test".to_owned(),
            linked_documents: Default::default(),
            document: box_document(),
            imports: Default::default(),
            session_hashes: None,
            selected: Some("p".to_owned()),
            comparison: None,
            ghost_input: None,
            edit_preview: None,
            clear_cache: false,
            run_exporters: false,
            dirty_nodes: std::collections::HashSet::new(),
            key: key_at("p", 1),
            token: nodecraft_engine::kernel_pool::RevisionToken::new(),
        }
    }

    fn validation(id: u64, target: &str) -> ValidationJob {
        validation_at(id, target, 1, box_document())
    }

    fn validation_at(
        id: u64,
        target: &str,
        revision: u64,
        document: GraphDocument,
    ) -> ValidationJob {
        ValidationJob {
            id,
            tab_id: 1,
            document_id: "worker-test".to_owned(),
            linked_documents: Default::default(),
            document: std::sync::Arc::new(document),
            imports: Default::default(),
            target: target.to_owned(),
            key: key_at(target, revision),
            token: nodecraft_engine::kernel_pool::RevisionToken::new(),
        }
    }

    fn wait_until(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if condition() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("the worker did not produce the expected work within 10s");
    }

    /// The app must enqueue an evaluate() foreground job before the validation
    /// requests for selected nodes outside the viewed closure. On a cache miss
    /// the idle worker takes the first channel message immediately, so sending
    /// validation first makes the user's evaluation wait for a cold validation
    /// pass. The worker records every take; channel FIFO is the priority order.
    #[test]
    fn foreground_evaluation_is_received_before_its_validation() {
        let _serial = test_support::serial();
        test_support::reset();
        let document = box_document();
        let viewed = target_closure(&document, "p");
        let targets = validation_targets_outside(&["q".to_owned()], &viewed);
        assert_eq!(
            targets,
            vec!["q".to_owned()],
            "q must be outside the viewed closure"
        );

        let worker = EvalWorker::new();
        // The worker is idle: this is the same first-message race evaluate()
        // hits on a cache miss.
        worker.send_evaluation(
            job(1),
            vec![validation(10, &targets[0])],
        );

        wait_until(|| test_support::received().len() >= 2);
        let received = test_support::received();
        test_support::reset();
        assert_eq!(
            received,
            vec![test_support::FOREGROUND, test_support::VALIDATION],
            "the foreground evaluation must reach the idle worker before its validation"
        );
    }

    /// A request that arrives while an earlier evaluation is running is drained
    /// from the channel to keep the cache warm, but it must still run. Losing it
    /// leaves the app waiting for an id that never produces a result, so the
    /// newest geometry is never handed to the display service.
    #[test]
    fn a_job_queued_while_one_is_running_is_not_lost() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        // The first job is paused after being picked; this request now lands
        // while it is still in flight.
        worker.send(job(2));
        test_support::set_pause_after_pick_ms(0);

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut newest = false;
        while Instant::now() < deadline {
            while let Some((id, _, _)) = worker.try_recv() {
                if id == 2 {
                    newest = true;
                }
            }
            if newest {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        test_support::reset();
        assert!(newest, "the job queued during the first evaluation must produce a result");
    }

    /// The validation request that lands first behind a running foreground job
    /// is the exact message the mid-job drain used to consume and drop.
    #[test]
    fn a_validation_queued_while_a_foreground_job_is_running_is_not_lost() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        worker.send_validation(validation(10, "q"));
        test_support::set_pause_after_pick_ms(0);

        let mut validated = false;
        wait_until(|| {
            while let Some((id, _, _)) = worker.try_recv_validation() {
                if id == 10 {
                    validated = true;
                }
            }
            validated
        });
        test_support::reset();
        assert!(
            validated,
            "the validation queued during the foreground evaluation must produce a result"
        );
    }

    /// F1 running, then V1, then F2: the foreground keeps priority, and V1
    /// still runs after F2 instead of being dropped by the drain.
    #[test]
    fn a_validation_survives_a_foreground_job_queued_behind_it() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        worker.send_validation(validation(11, "q"));
        worker.send(job(2));
        test_support::set_pause_after_pick_ms(0);

        let mut newest_foreground = false;
        let mut validated = false;
        wait_until(|| {
            while let Some((id, _, _)) = worker.try_recv() {
                if id == 2 {
                    newest_foreground = true;
                }
            }
            while let Some((id, _, _)) = worker.try_recv_validation() {
                if id == 11 {
                    validated = true;
                }
            }
            newest_foreground && validated
        });
        test_support::reset();
        assert!(newest_foreground, "the newer foreground job must run");
        assert!(validated, "the older validation request must still run");
    }

    /// Two validation requests for different targets cover different work; a
    /// newer one must not supersede the other target's request.
    #[test]
    fn validations_for_different_targets_both_run() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        worker.send_validation(validation(20, "p"));
        worker.send_validation(validation(21, "q"));
        test_support::set_pause_after_pick_ms(0);

        let mut first = false;
        let mut second = false;
        wait_until(|| {
            while let Some((id, _, _)) = worker.try_recv_validation() {
                first |= id == 20;
                second |= id == 21;
            }
            first && second
        });
        test_support::reset();
        assert!(first, "the validation for the first target must run");
        assert!(second, "the validation for the second target must run");
    }

    /// A newer request for the same target replaces the older request, and the
    /// worker retires the older id so the app can stop waiting for its result.
    #[test]
    fn a_newer_validation_for_the_same_target_retires_the_older_request() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        worker.send_validation(validation_at(30, "q", 1, box_document()));
        worker.send_validation(validation_at(31, "q", 2, box_document()));
        test_support::set_pause_after_pick_ms(0);

        let mut retired = false;
        let mut replaced = false;
        let mut old_ran = false;
        wait_until(|| {
            while let Some((id, _)) = worker.try_recv_retired_validation() {
                retired |= id == 30;
            }
            while let Some((id, _, _)) = worker.try_recv_validation() {
                replaced |= id == 31;
                old_ran |= id == 30;
            }
            retired && replaced
        });
        test_support::reset();
        assert!(retired, "the superseded request must be explicitly retired");
        assert!(replaced, "the newer request must run");
        assert!(!old_ran, "the superseded request must not run as well");
    }

    /// Validation queued before a foreground job must keep a live token: a
    /// cancelled token suppresses the node errors the glow is made of.
    #[test]
    fn a_validation_behind_a_foreground_job_keeps_its_diagnostics() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        worker.send_validation(validation_at(40, "bad", 1, error_document()));
        worker.send(job(2));
        test_support::set_pause_after_pick_ms(0);

        let mut diagnostics: Option<engine_bridge::Evaluation> = None;
        wait_until(|| {
            while let Some((id, _, evaluation)) = worker.try_recv_validation() {
                if id == 40 {
                    diagnostics = Some(evaluation);
                }
            }
            diagnostics.is_some()
        });
        test_support::reset();
        let diagnostics = diagnostics.expect("the validation result");
        assert!(
            diagnostics.node_errors.contains_key("bad"),
            "validation diagnostics must not be suppressed by a foreground job: {:?}",
            diagnostics.node_errors
        );
    }

    /// App bookkeeping: every request produces a result or an explicit
    /// retirement, so validation_pending drains to empty.
    #[test]
    fn validation_pending_is_empty_after_all_results_are_drained() {
        let _serial = test_support::serial();
        test_support::reset();
        test_support::set_pause_after_pick_ms(300);
        let worker = EvalWorker::new();
        worker.send(job(1));

        let deadline = Instant::now() + Duration::from_secs(5);
        while test_support::picked() == 0 {
            assert!(Instant::now() < deadline, "the worker did not pick the first job");
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut pending = std::collections::HashMap::new();
        let mut requested = std::collections::HashSet::new();
        for (id, target) in [(50, "p"), (51, "q")] {
            let key = key_at(target, 1);
            pending.insert(
                id,
                ValidationPending {
                    tab_id: 1,
                    body_target: target.to_owned(),
                    key: key.clone(),
                },
            );
            requested.insert(key);
            worker.send_validation(validation(id, target));
        }
        test_support::set_pause_after_pick_ms(0);

        wait_until(|| {
            while let Some((id, _, _)) = worker.try_recv_validation() {
                take_validation_request(&mut pending, &mut requested, id);
            }
            while let Some((id, _)) = worker.try_recv_retired_validation() {
                take_validation_request(&mut pending, &mut requested, id);
            }
            pending.is_empty()
        });
        test_support::reset();
        assert!(pending.is_empty(), "every validation request must retire");
        assert!(requested.is_empty(), "no superseded key may stay requested");
    }
}
