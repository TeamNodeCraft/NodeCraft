//! Bounded SolidCore kernel worker pool.
//!
//! One persistent [`Kernel`] per worker. Jobs receive a deterministic RNG seed
//! and a cancellation token; no kernel is shared between threads.

pub use nodecraft_solidcore::solidcore::kernel::CancelToken;
use nodecraft_solidcore::solidcore::kernel::Kernel;
use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Condvar, Mutex};

pub use nodecraft_solidcore::solidcore::kernel::CancelToken as RevisionToken;

static KERNEL_RUN_NANOS: AtomicU64 = AtomicU64::new(0);

/// Total CPU time spent inside kernel jobs since process start.
pub fn kernel_run_nanos() -> u64 {
    KERNEL_RUN_NANOS.load(Ordering::Relaxed)
}

thread_local! {
    static CURRENT_REVISION_TOKEN: std::cell::RefCell<CancelToken> = std::cell::RefCell::new(CancelToken::new());
}

/// Install the cancellation token for exact tasks started on this thread.
pub fn set_current_revision_token(token: CancelToken) {
    CURRENT_REVISION_TOKEN.with(|current| *current.borrow_mut() = token);
}

pub(crate) fn current_revision_token() -> CancelToken {
    CURRENT_REVISION_TOKEN.with(|current| current.borrow().clone())
}

/// Run `f` with the current revision token over any `solidcore-ops` call it makes.
pub fn with_current_token<R>(f: impl FnOnce() -> R) -> R {
    nodecraft_ops::cancel::within(&current_revision_token(), f)
}

/// Result of one job. The pool uses this to reset a kernel after a panic
/// without losing the panic message sent to the caller.
struct JobOutcome {
    panicked: bool,
}

type Job = Box<dyn FnOnce(&mut Kernel) -> JobOutcome + Send + 'static>;

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

struct Work {
    seed: u64,
    token: CancelToken,
    job: Job,
    submitted: std::time::Instant,
}

struct Shared {
    queue: Mutex<VecDeque<Work>>,
    /// Low-priority work (the background proof queue). A worker takes one only when the
    /// foreground queue is empty, so an evaluation request never waits behind a proof that
    /// has not started yet.
    background: Mutex<VecDeque<Work>>,
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
            // Only idle capacity runs background work: a queued foreground job is always
            // taken first, and a worker waiting for one is woken by `push`.
            if let Some(work) =
                self.background.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).pop_front()
            {
                return Some(work);
            }
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            queue = self.ready.wait(queue).unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn push_background(&self, work: Work) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return false;
        }
        let mut queue = self.background.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        queue.push_back(work);
        drop(queue);
        self.ready.notify_one();
        true
    }

    fn push(&self, work: Work) -> bool {
        let depth = self.queue.lock().map(|queue| queue.len()).unwrap_or(0);
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[pool] kernel queue_depth={depth}");
        }
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

/// Pool of worker threads, each owning one persistent kernel.
pub struct KernelPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl KernelPool {
    /// Start `threads` workers. At least one worker is always started.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            background: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            shutdown: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        });
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[pool] kernel workers={threads}");
        }
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let shared = Arc::clone(&shared);
            let worker = std::thread::Builder::new()
                .name(format!("solidcore-kernel-{index}"))
                .stack_size(nodecraft_kernel_profile::KERNEL_THREAD_STACK)
                .spawn(move || {
                    let mut kernel = nodecraft_solidcore::kernel::kernel();
                    let mut generation = 0;
                    while let Some(work) = shared.next() {
                        let started = std::time::Instant::now();
                        let current = shared.generation.load(Ordering::Acquire);
                        if current != generation {
                            kernel = nodecraft_solidcore::kernel::kernel();
                            generation = current;
                        }
                        kernel.rng = nodecraft_solidcore::solidcore::kernel::Rng::new(work.seed);
                        kernel.set_cancel_token(work.token.clone());
                        let outcome = catch_unwind(AssertUnwindSafe(|| {
                            nodecraft_ops::cancel::within(&work.token, || (work.job)(&mut kernel))
                        }));
                        if nodecraft_ops::take_needs_reset() || work.token.is_cancelled() {
                            kernel = nodecraft_solidcore::kernel::kernel();
                        }
                        let reset = match outcome {
                            Ok(outcome) => outcome.panicked,
                            Err(_) => true,
                        };
                        let elapsed = started.elapsed();
                        KERNEL_RUN_NANOS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
                        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                            eprintln!(
                                "[pool] kernel wait_ms={:.2} run_ms={:.2}",
                                started.duration_since(work.submitted).as_secs_f64() * 1000.0,
                                elapsed.as_secs_f64() * 1000.0,
                            );
                        }
                        if reset {
                            // A panicking kernel is not trusted for later jobs.
                            kernel = nodecraft_solidcore::kernel::kernel();
                        }
                    }
                })
                .expect("spawn SolidCore kernel worker");
            workers.push(worker);
        }
        Self { shared, workers }
    }

    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Make every worker discard its kernel and memoisation caches before its
    /// next job. The reset is lazy so it never blocks the caller behind a busy
    /// kernel.
    pub fn reset_caches(&self) {
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Run one job synchronously on a pool worker. This is the convenience
    /// boundary for exact-node operations that need a kernel but are already
    /// scheduled by the caller.
    pub fn execute<T, F>(&self, seed: u64, job: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&mut Kernel) -> Result<T, String> + Send + 'static,
    {
        let received = self
            .submit(seed, current_revision_token(), job)
            .recv()
            .map_err(|_| "kernel worker stopped".to_owned())?;
        received?
    }

    /// Run with an explicit cancellation token (used by the app revision).
    pub fn execute_with_token<T, F>(
        &self,
        seed: u64,
        token: CancelToken,
        job: F,
    ) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&mut Kernel) -> Result<T, String> + Send + 'static,
    {
        let received =
            self.submit(seed, token, job).recv().map_err(|_| "kernel worker stopped".to_owned())?;
        received?
    }

    /// Submit one job and return its result channel. The closure must own all
    /// geometry it mutates; shared inputs should be `Arc<Brep>`.
    pub fn submit<T, F>(&self, seed: u64, token: CancelToken, job: F) -> Receiver<Result<T, String>>
    where
        T: Send + 'static,
        F: FnOnce(&mut Kernel) -> T + Send + 'static,
    {
        let (result_tx, result_rx) = channel();
        let erased: Job = Box::new(move |kernel: &mut Kernel| {
            match catch_unwind(AssertUnwindSafe(|| job(kernel))) {
                Ok(value) => {
                    let _ = result_tx.send(Ok(value));
                    JobOutcome { panicked: false }
                }
                Err(payload) => {
                    let message = panic_message(payload);
                    let _ = result_tx.send(Err(format!("kernel job panicked: {message}")));
                    JobOutcome { panicked: true }
                }
            }
        });
        if !self.shared.push(Work {
            seed,
            token,
            job: erased,
            submitted: std::time::Instant::now(),
        }) {
            let (tx, rx) = channel();
            let _ = tx.send(Err("kernel pool is shut down".to_owned()));
            return rx;
        }
        result_rx
    }

    /// Submit one low-priority job. Workers run it only when no foreground job is queued;
    /// the background proof queue uses this so evaluation never queues behind a proof.
    pub fn submit_background<T, F>(
        &self,
        seed: u64,
        token: CancelToken,
        job: F,
    ) -> Receiver<Result<T, String>>
    where
        T: Send + 'static,
        F: FnOnce(&mut Kernel) -> T + Send + 'static,
    {
        let (result_tx, result_rx) = channel();
        let erased: Job = Box::new(move |kernel: &mut Kernel| {
            match catch_unwind(AssertUnwindSafe(|| job(kernel))) {
                Ok(value) => {
                    let _ = result_tx.send(Ok(value));
                    JobOutcome { panicked: false }
                }
                Err(payload) => {
                    let message = panic_message(payload);
                    let _ = result_tx.send(Err(format!("kernel job panicked: {message}")));
                    JobOutcome { panicked: true }
                }
            }
        });
        if !self.shared.push_background(Work {
            seed,
            token,
            job: erased,
            submitted: std::time::Instant::now(),
        }) {
            let (tx, rx) = channel();
            let _ = tx.send(Err("kernel pool is shut down".to_owned()));
            return rx;
        }
        result_rx
    }
}

impl Drop for KernelPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.ready.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

static EXACT_POOL: std::sync::OnceLock<KernelPool> = std::sync::OnceLock::new();
static REQUESTED_EXACT_THREADS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Configure the exact-evaluation pool before its first use. `0` means auto.
pub fn configure_exact_threads(threads: usize) {
    REQUESTED_EXACT_THREADS.store(threads, std::sync::atomic::Ordering::Relaxed);
}

/// Backwards-compatible alias for configuring exact kernel threads.
pub fn configure_threads(threads: usize) {
    configure_exact_threads(threads);
}

fn pool_threads(requested: &std::sync::atomic::AtomicUsize) -> usize {
    let requested = requested.load(std::sync::atomic::Ordering::Relaxed);
    if requested == 0 {
        std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1)
    } else {
        requested
    }
}

/// Whether the exact-evaluation pool has already been created.
#[must_use]
pub fn exact_pool_initialized() -> bool {
    EXACT_POOL.get().is_some()
}

/// Worker count of the initialized exact pool, if it has started.
#[must_use]
pub fn exact_pool_worker_count() -> Option<usize> {
    EXACT_POOL.get().map(KernelPool::workers)
}

/// Backwards-compatible alias for exact-pool initialisation.
#[must_use]
pub fn shared_pool_initialized() -> bool {
    exact_pool_initialized()
}

/// Exact-evaluation pool. Workers own persistent kernels and RNG state.
#[must_use]
pub fn shared_exact_pool() -> &'static KernelPool {
    EXACT_POOL.get_or_init(|| KernelPool::new(pool_threads(&REQUESTED_EXACT_THREADS)))
}

/// Backwards-compatible alias for the exact-evaluation pool.
#[must_use]
pub fn shared_pool() -> &'static KernelPool {
    shared_exact_pool()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panicking_job_reports_an_error_and_pool_stays_usable() {
        let pool = KernelPool::new(1);
        let error = pool
            .execute(1, |_| -> Result<u32, String> { panic!("boom") })
            .expect_err("panic should become an error");
        assert!(error.contains("panicked"), "{error}");
        assert_eq!(pool.execute(2, |_| Ok(7u32)), Ok(7));
    }

    #[test]
    fn rng_seed_is_injected_per_job() {
        let pool = KernelPool::new(2);
        let first = pool
            .submit(17, CancelToken::new(), |kernel| kernel.rng.next_u64())
            .recv()
            .expect("result")
            .expect("job");
        let second = pool
            .submit(17, CancelToken::new(), |kernel| kernel.rng.next_u64())
            .recv()
            .expect("result")
            .expect("job");
        assert_eq!(first, second);
        let different = pool
            .submit(18, CancelToken::new(), |kernel| kernel.rng.next_u64())
            .recv()
            .expect("result")
            .expect("job");
        assert_ne!(first, different);
    }

    #[test]
    fn revision_token_is_used_by_execute() {
        let pool = KernelPool::new(1);
        let token = RevisionToken::new();
        token.cancel();
        set_current_revision_token(token);
        let cancelled = pool.execute(1, |kernel| Ok(kernel.is_cancelled())).expect("job");
        assert!(cancelled);
        set_current_revision_token(RevisionToken::new());
    }

    #[test]
    fn pre_cancelled_job_still_rebuilds_the_kernel() {
        let pool = KernelPool::new(1);
        let token = CancelToken::new();
        token.cancel();
        pool.execute_with_token(1, token, |kernel| {
            kernel.fillet_point_step.set(Some(1.0));
            Ok(())
        })
        .expect("job");
        let marker_left = pool
            .execute(2, |kernel| Ok(kernel.fillet_point_step.get().is_some()))
            .expect("second job");
        assert!(!marker_left, "a cancelled job must leave a fresh kernel");
    }

    #[test]
    fn cancelled_session_rebuilds_the_worker_kernel() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        let pool = KernelPool::new(1);
        let token = CancelToken::new();
        let (started, wait) = std::sync::mpsc::channel();
        let cancel = token.clone();
        let canceller = std::thread::spawn(move || {
            wait.recv().expect("started");
            std::thread::sleep(std::time::Duration::from_millis(5));
            cancel.cancel();
        });
        let outcome = pool.execute_with_token(1, token, move |kernel| {
            kernel.fillet_point_step.set(Some(1.0));
            let brep = create_box(kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
                .map_err(|error| error.to_string())?;
            let edges = brep.edge_list().to_vec();
            started.send(()).expect("signal");
            match nodecraft_ops::OpsSession::new(kernel).blend(
                &brep,
                &edges,
                1.0,
                nodecraft_ops::OpsBlend::Round(nodecraft_ops::CornerTurn::Mitre),
            ) {
                Err(nodecraft_ops::SessionError::Cancelled) => Ok(()),
                other => Err(format!("expected a mid-call cancel, got {other:?}")),
            }
        });
        canceller.join().expect("canceller");
        outcome.expect("cancelled job");
        let marker_left = pool
            .execute(2, |kernel| Ok(kernel.fillet_point_step.get().is_some()))
            .expect("second job");
        assert!(!marker_left, "the worker kernel was not rebuilt");
    }

    #[test]
    fn cancellation_token_is_visible_to_the_job() {
        let pool = KernelPool::new(1);
        let token = CancelToken::new();
        token.cancel();
        let cancelled = pool
            .submit(1, token, |kernel| kernel.is_cancelled())
            .recv()
            .expect("result")
            .expect("job");
        assert!(cancelled);
    }
}

/// Reset the exact kernel pool after an explicit rebuild.
pub fn reset_kernel_caches() {
    if let Some(pool) = EXACT_POOL.get() {
        pool.reset_caches();
    }
}
