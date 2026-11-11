//! Bounded SolidCore kernel worker pool.
//!
//! One persistent [`Kernel`] per worker. Jobs receive a deterministic RNG seed
//! and a cancellation token; no kernel is shared between threads.

use nodecraft_solidcore::solidcore::kernel::{CancelToken, Kernel};
use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Condvar, Mutex};

pub use nodecraft_solidcore::solidcore::kernel::CancelToken as RevisionToken;

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
            queue = self
                .ready
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
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
            ready: Condvar::new(),
            shutdown: AtomicBool::new(false),
            generation: AtomicU64::new(0),
        });
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let shared = Arc::clone(&shared);
            let worker = std::thread::Builder::new()
                .name(format!("solidcore-kernel-{index}"))
                .spawn(move || {
                    let mut kernel = nodecraft_solidcore::kernel::kernel();
                    let mut generation = 0;
                    while let Some(work) = shared.next() {
                        let current = shared.generation.load(Ordering::Acquire);
                        if current != generation {
                            kernel = nodecraft_solidcore::kernel::kernel();
                            generation = current;
                        }
                        kernel.rng = nodecraft_solidcore::solidcore::kernel::Rng::new(work.seed);
                        kernel.set_cancel_token(work.token);
                        let outcome = catch_unwind(AssertUnwindSafe(|| (work.job)(&mut kernel)));
                        let reset = match outcome {
                            Ok(outcome) => outcome.panicked,
                            Err(_) => true,
                        };
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
        let received = self
            .submit(seed, token, job)
            .recv()
            .map_err(|_| "kernel worker stopped".to_owned())?;
        received?
    }

    /// Submit one job and return its result channel. The closure must own all
    /// geometry it mutates; shared inputs should be `Arc<Brep>`.
    pub fn submit<T, F>(
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
        if !self.shared.push(Work { seed, token, job: erased }) {
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
        std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
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
        let cancelled = pool
            .execute(1, |kernel| Ok(kernel.is_cancelled()))
            .expect("job");
        assert!(cancelled);
        set_current_revision_token(RevisionToken::new());
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
