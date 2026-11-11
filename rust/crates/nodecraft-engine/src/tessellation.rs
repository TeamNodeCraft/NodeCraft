//! Independent display-tessellation workers.
//!
//! The exact evaluator hands a [`MeshRequest`] to these workers and receives a
//! [`MeshAsset`] back. SolidCore kernels stay on the exact side; this module
//! only turns an immutable B-rep into a local display mesh plus instance
//! placement.

use crate::display::{MeshAsset, MeshInstance, MeshRequest, MeshRevision, MeshKey};
use nodecraft_solidcore::solidcore::kernel::Kernel;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;

pub struct MeshResult {
    pub key: MeshKey,
    pub revision: u64,
    pub asset: Result<MeshAsset, String>,
}

struct Work {
    request: MeshRequest,
    revision: MeshRevision,
    result: Sender<MeshResult>,
}

struct Shared {
    queue: Mutex<VecDeque<Work>>,
    ready: Condvar,
    shutdown: AtomicBool,
    /// Bumped to make every worker replace its kernel before its next job.
    generation: AtomicU64,
}

impl Shared {
    fn next(&self) -> Option<Work> {
        let mut queue = self.queue.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(work) = queue.pop_front() {
                return Some(work);
            }
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            queue = self.ready.wait(queue).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn push(&self, work: Work) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        let mut queue = self.queue.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        queue.push_back(work);
        drop(queue);
        self.ready.notify_one();
        true
    }
}

pub struct TessellationPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl TessellationPool {
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            shutdown: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        });
        let mut workers = Vec::with_capacity(threads.max(1));
        for index in 0..threads.max(1) {
            let shared = Arc::clone(&shared);
            let worker = std::thread::Builder::new()
                .name(format!("nodecraft-tessellation-{index}"))
                .spawn(move || {
                    let mut kernel: Kernel = crate::nodecraft_solidcore_kernel();
                    let mut generation = 0;
                    while let Some(work) = shared.next() {
                        let current = shared.generation.load(Ordering::Acquire);
                        if current != generation {
                            kernel = crate::nodecraft_solidcore_kernel();
                            generation = current;
                        }
                        if work.revision.token.is_cancelled() {
                            let _ = work.result.send(MeshResult {
                                key: work.request.key,
                                revision: work.request.revision,
                                asset: Err("tessellation request cancelled".into()),
                            });
                            continue;
                        }
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            let tolerance = f64::from_bits(work.request.key.tolerance_bits);
                            let mut brep = work
                                .request
                                .body
                                .brep
                                .clone_standing_for_self()
                                .map_err(|error| format!("display copy: {error}"))?;
                            let tessellated = match work.request.key.preview_detail {
                                Some(detail) => nodecraft_layers::tessellate_kernel_preview(
                                    &mut kernel,
                                    &mut brep,
                                    detail,
                                ),
                                None => nodecraft_layers::tessellate_kernel(
                                    &mut kernel,
                                    &mut brep,
                                    tolerance,
                                ),
                            };
                            tessellated
                                .map(|display| {
                                    let stats = display.stats.clone();
                                    let edge_ids = display.plan_edge_ids.clone();
                                    (display.plan, display.mesh, stats, edge_ids)
                                })
                        }));
                        let asset = match outcome {
                            Ok(Ok((plan, mesh, stats, edge_ids))) => {
                                if work.revision.token.is_cancelled() {
                                    Err("tessellation request cancelled".into())
                                } else {
                                    Ok(MeshAsset {
                                        instances: vec![MeshInstance {
                                            mesh: Arc::new(mesh),
                                            plan: Some(plan),
                                            plan_edge_ids: edge_ids,
                                            stats: Some(stats),
                                            transforms: work.request.body.transforms,
                                        }],
                                    })
                                }
                            }
                            Ok(Err(error)) => Err(error),
                            Err(_) => {
                                kernel = crate::nodecraft_solidcore_kernel();
                                Err("tessellation worker panicked".into())
                            }
                        };
                        let _ = work.result.send(MeshResult {
                            key: work.request.key,
                            revision: work.request.revision,
                            asset,
                        });
                    }
                })
                .expect("spawn tessellation worker");
            workers.push(worker);
        }
        Self { shared, workers }
    }

    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Make every worker discard its kernel and memoisation caches before its
    /// next job.
    pub fn reset_caches(&self) {
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Queue one body request. The receiver carries the revision so stale
    /// results can be discarded by the coordinator.
    #[must_use]
    pub fn submit(&self, request: MeshRequest, revision: MeshRevision) -> Receiver<MeshResult> {
        let (result, receiver) = channel();
        if !self.shared.push(Work { request, revision, result }) {
            let (tx, rx) = channel();
            let _ = tx.send(MeshResult {
                key: MeshKey::new("shutdown", "", 0.0, None),
                revision: 0,
                asset: Err("tessellation pool is shut down".into()),
            });
            return rx;
        }
        receiver
    }
}

impl Drop for TessellationPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.ready.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

static TESSELLATION: std::sync::OnceLock<TessellationPool> = std::sync::OnceLock::new();
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Worker count of the initialized display tessellation, if it has started.
#[must_use]
pub fn tessellation_worker_count() -> Option<usize> {
    TESSELLATION.get().map(TessellationPool::workers)
}

pub fn configure_tessellation_threads(threads: usize) {
    REQUESTED_THREADS.store(threads, Ordering::Relaxed);
}

#[must_use]
pub fn shared_tessellation() -> &'static TessellationPool {
    TESSELLATION.get_or_init(|| {
        let requested = REQUESTED_THREADS.load(Ordering::Relaxed);
        let threads = if requested == 0 {
            std::thread::available_parallelism().map_or(1, |count| count.get())
        } else {
            requested
        };
        TessellationPool::new(threads)
    })
}

#[must_use]
pub fn tessellation_initialized() -> bool {
    TESSELLATION.get().is_some()
}

/// Reset the shared display-tessellation kernels after an explicit rebuild.
pub fn reset_tessellation_caches() {
    if let Some(pool) = TESSELLATION.get() {
        pool.reset_caches();
    }
}
