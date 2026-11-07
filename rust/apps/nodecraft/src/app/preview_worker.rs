//! Dedicated branch-only preview mesher
//!
//! The main evaluation worker owns the displayed breakpoint and the exact Part
//! cache. Waiting for it to finish a full breakpoint display pass just to get
//! the transform input mesh makes the edit ghost feel late. This worker runs
//! the preview target (the transform's input node) on its own thread and cache,
//! so the preview is never serialized behind the breakpoint evaluation.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use nodecraft_engine::EngineCache;
use nodecraft_protocol::Mesh;
use nodecraft_session::GraphDocument;

pub(super) struct PreviewRequest {
    pub request_id: u64,
    pub tab_id: u64,
    pub document: GraphDocument,
    pub linked_documents: BTreeMap<PathBuf, GraphDocument>,
    /// Transform input node whose display mesh becomes the ghost.
    pub target: String,
    pub token: nodecraft_engine::kernel_pool::RevisionToken,
}

pub(super) struct PreviewResult {
    pub request_id: u64,
    pub tab_id: u64,
    pub target: String,
    pub mesh: Option<Mesh>,
}

#[derive(Default)]
struct TabState {
    cache: EngineCache,
    flat: nodecraft_session::flat::FlatGraphCache,
}

pub(super) struct PreviewWorker {
    tx: Sender<PreviewRequest>,
    rx: Receiver<PreviewResult>,
}

impl PreviewWorker {
    pub(super) fn new() -> Self {
        let (tx, job_rx) = channel::<PreviewRequest>();
        let (result_tx, rx) = channel();
        std::thread::Builder::new()
            .name("nodecraft-preview".into())
            .spawn(move || {
                let mut states: HashMap<u64, TabState> = HashMap::new();
                while let Ok(mut job) = job_rx.recv() {
                    while let Ok(next) = job_rx.try_recv() {
                        job = next;
                    }
                    nodecraft_engine::kernel_pool::set_current_revision_token(job.token);
                    let mut state = states.remove(&job.tab_id).unwrap_or_default();
                    let target = job.target.clone();
                    let mesh = state
                        .flat
                        .resolve(&job.document, &job.linked_documents)
                        .ok()
                        .and_then(|flat| {
                            let flat = flat.clone();
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                let (evaluation, next_cache) =
                                    crate::engine_bridge::evaluate_flat_with_progress_comparison_dirty(
                                        flat.document(),
                                        target.clone(),
                                        state.cache.clone(),
                                        None,
                                        None,
                                        false,
                                        &Default::default(),
                                        None,
                                        &[],
                                        None,
                                    );
                                state.cache = next_cache;
                                evaluation.and_then(|evaluation| evaluation.mesh)
                            }))
                            .ok()
                            .flatten()
                        });
                    if states.len() > 8 {
                        if let Some(oldest) = states.keys().next().copied() {
                            states.remove(&oldest);
                        }
                    }
                    states.insert(job.tab_id, state);
                    if result_tx
                        .send(PreviewResult {
                            request_id: job.request_id,
                            tab_id: job.tab_id,
                            target,
                            mesh,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .ok();
        Self { tx, rx }
    }

    pub(super) fn request(&self, request: PreviewRequest) {
        let _ = self.tx.send(request);
    }

    pub(super) fn try_recv(&self) -> Option<PreviewResult> {
        self.rx.try_recv().ok()
    }
}
