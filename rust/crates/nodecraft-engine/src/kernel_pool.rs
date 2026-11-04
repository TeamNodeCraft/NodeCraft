//! Bounded SolidCore kernel worker pool.
//!
//! One persistent [`Kernel`] per worker. Jobs receive a deterministic RNG seed
//! and a cancellation token; no kernel is shared between threads.

use nodecraft_solidcore::solidcore::kernel::{CancelToken, Kernel, KernelOptions};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};

pub use nodecraft_solidcore::solidcore::kernel::CancelToken as RevisionToken;

thread_local! {
    static CURRENT_REVISION_TOKEN: std::cell::RefCell<CancelToken> = std::cell::RefCell::new(CancelToken::new());
}

/// Install the cancellation token for exact tasks started on this thread.
pub fn set_current_revision_token(token: CancelToken) {
    CURRENT_REVISION_TOKEN.with(|current| *current.borrow_mut() = token);
}

fn current_revision_token() -> CancelToken {
    CURRENT_REVISION_TOKEN.with(|current| current.borrow().clone())
}

type Job = Box<dyn FnOnce(&mut Kernel) -> Box<dyn Any + Send> + Send + 'static>;

enum Message {
    Job { seed: u64, token: CancelToken, job: Job },
    Shutdown,
}

/// Pool of worker threads, each owning one persistent kernel.
pub struct KernelPool {
    tx: Sender<Message>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl KernelPool {
    /// Start `threads` workers. At least one worker is always started.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let (tx, rx) = channel::<Message>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let mut workers = Vec::with_capacity(threads);
        for index in 0..threads {
            let rx = rx.clone();
            let worker = std::thread::Builder::new()
                .name(format!("solidcore-kernel-{index}"))
                .spawn(move || {
                    let mut kernel = Kernel::new(&KernelOptions::default());
                    loop {
                        let message = match rx.lock() {
                            Ok(receiver) => receiver.recv(),
                            Err(_) => return,
                        };
                        let Ok(message) = message else { return };
                        match message {
                            Message::Shutdown => return,
                            Message::Job { seed, token, job } => {
                                kernel.rng = nodecraft_solidcore::solidcore::kernel::Rng::new(seed);
                                kernel.set_cancel_token(token);
                                let outcome = catch_unwind(AssertUnwindSafe(|| job(&mut kernel)));
                                if outcome.is_err() {
                                    // A panicking kernel is not trusted for later jobs.
                                    kernel = Kernel::new(&KernelOptions::default());
                                }
                            }
                        }
                    }
                })
                .expect("spawn SolidCore kernel worker");
            workers.push(worker);
        }
        Self { tx, workers }
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
            let value = job(kernel);
            let _ = result_tx.send(Ok(value));
            Box::new(())
        });
        if self.tx.send(Message::Job { seed, token, job: erased }).is_err() {
            let (tx, rx) = channel();
            let _ = tx.send(Err("kernel pool is shut down".to_owned()));
            return rx;
        }
        result_rx
    }
}

impl Drop for KernelPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            let _ = self.tx.send(Message::Shutdown);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

static SHARED_POOL: std::sync::OnceLock<KernelPool> = std::sync::OnceLock::new();
static REQUESTED_THREADS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Configure the shared pool before its first use. `0` means auto-detect.
/// The pool is created lazily, so changing this after evaluation has started
/// does not resize the existing workers.
pub fn configure_threads(threads: usize) {
    REQUESTED_THREADS.store(threads, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the shared pool has already been created.
#[must_use]
pub fn shared_pool_initialized() -> bool {
    SHARED_POOL.get().is_some()
}

/// Shared engine pool. Workers own persistent kernels across evaluations.
#[must_use]
pub fn shared_pool() -> &'static KernelPool {
    SHARED_POOL.get_or_init(|| {
        let requested = REQUESTED_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        let threads = if requested == 0 {
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        } else {
            requested
        };
        KernelPool::new(threads)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
