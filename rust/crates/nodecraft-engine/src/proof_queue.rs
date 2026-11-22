//! Background proof queue: prove the bodies an evaluation produced, off the critical path.
//!
//! Evaluation runs ops under `Validation::EachNode`, so a node's body is not proved while the
//! user waits. When the exact pass ends, every body it produced (displayed targets first, then
//! intermediates) is handed to this queue. The queue submits work to the kernel pool's
//! low-priority channel only when no foreground job is queued, proves each body at most once
//! (the shared [`ProofState`](nodecraft_ops::ProofState) on the body is the memo) and keeps
//! finished proofs for later evaluations.
//!
//! A refusal is a *later* result. It is put on a small inbox ([`take_reports`]) that the
//! application polls on its evaluation thread, exactly where it already polls progress and
//! display results; the producer node is marked with the failed test and every downstream node
//! gets a warning. The headless worker transport has no post-result channel, so it does not
//! invent one: it leaves the refusal in the body's cached state.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use nodecraft_ops::{ProofState, SolidBody};
use nodecraft_protocol::Request;

use crate::kernel_pool::{CancelToken, shared_pool};

/// What a proof attempt did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProofStats {
    /// Bodies handed to the queue (after de-duplication).
    pub queued: usize,
    /// Proofs that actually started on a kernel.
    pub proofs_run: usize,
    /// Bodies proved valid outward solids.
    pub proved: usize,
    /// Bodies the proof refused (a report was raised).
    pub refused: usize,
    /// Bodies skipped: not a closed solid (a sheet), already proved/refused, or stale.
    pub skipped: usize,
    /// Proofs that failed for a reason other than invalidity (kernel status, cancellation).
    pub failed: usize,
    /// Summed wall time of the proof calls.
    pub total_ms: f64,
    /// Wall time from the first proof start to the queue draining the last known body.
    pub wall_ms: f64,
}

/// One refused body, delivered after evaluation.
#[derive(Clone, Debug)]
pub struct ProofReport {
    pub generation: u64,
    pub request_id: u64,
    pub tab_id: Option<String>,
    pub graph_revision: Option<String>,
    /// The node whose output first held the refused body.
    pub producer: String,
    /// The failed proof's message, naming the test the body failed.
    pub failure: String,
    /// Every node downstream of the producer, transitively.
    pub downstream: Vec<String>,
}

/// Why a body is queued.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofPriority {
    /// A body of a requested/displayed target.
    Displayed,
    /// An intermediate body a target is built from.
    Intermediate,
}

struct GraphContext {
    connections: Vec<(String, String)>,
}

impl GraphContext {
    fn downstream(&self, producer: &str) -> Vec<String> {
        let mut frontier = vec![producer.to_owned()];
        let mut seen = HashSet::new();
        seen.insert(producer.to_owned());
        let mut downstream = Vec::new();
        while let Some(current) = frontier.pop() {
            for (from, to) in &self.connections {
                if from == &current && seen.insert(to.clone()) {
                    downstream.push(to.clone());
                    frontier.push(to.clone());
                }
            }
        }
        downstream.sort();
        downstream.dedup();
        downstream
    }
}

#[derive(Clone)]
struct RequestMeta {
    request_id: u64,
    tab_id: Option<String>,
    graph_revision: Option<String>,
    context: Arc<GraphContext>,
}

#[derive(Clone)]
struct JobMeta {
    generation: u64,
    request: Arc<RequestMeta>,
    producer: String,
}

struct Job {
    body: SolidBody,
    meta: JobMeta,
}

struct Inner {
    high: VecDeque<Job>,
    low: VecDeque<Job>,
    /// Proof ids registered in the current generation: the priority they were queued at and
    /// the job metadata of the node that first produced them (a promotion keeps that producer).
    known: HashMap<usize, (ProofPriority, JobMeta)>,
    /// Proof ids a worker has taken (across generations, removed when finished).
    claimed: HashSet<usize>,
    in_flight: usize,
    active_evaluations: usize,
    reports: VecDeque<ProofReport>,
    /// The request metadata per generation, for a refusal's downstream closure.
    contexts: HashMap<u64, Arc<RequestMeta>>,
    stats: ProofStats,
    first_started: Option<Instant>,
}

/// Whether a body needs a proof job: its shared [`ProofState`] is still
/// `Unknown` and this generation's `known` map has not already queued it.
/// `known` is the queue's existing bookkeeping map, not a second set.
fn needs_proof(state: &ProofState, already_queued: bool) -> bool {
    matches!(state, ProofState::Unknown) && !already_queued
}

struct Queue {
    generation: AtomicU64,
    inner: Mutex<Inner>,
    idle: Condvar,
}

const MAX_REPORTS: usize = 1024;
const MAX_CONTEXTS: usize = 16;

impl Queue {
    fn new() -> Queue {
        Queue {
            generation: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                high: VecDeque::new(),
                low: VecDeque::new(),
                known: HashMap::new(),
                claimed: HashSet::new(),
                in_flight: 0,
                active_evaluations: 0,
                reports: VecDeque::new(),
                contexts: HashMap::new(),
                stats: ProofStats::default(),
                first_started: None,
            }),
            idle: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn begin(&self, request: &Request) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let meta = Arc::new(RequestMeta {
            request_id: request.request_id,
            tab_id: request.tab_id.clone(),
            graph_revision: request.graph_revision.clone(),
            context: Arc::new(GraphContext {
                connections: request
                    .connections
                    .iter()
                    .map(|link| (link.from.clone(), link.to.clone()))
                    .collect(),
            }),
        });
        {
            let mut inner = self.lock();
            inner.active_evaluations += 1;
            // A new request deprioritises the previous generation's unstarted work: its jobs
            // stay in the queues and are skipped when a worker reaches them. Finished proofs
            // stay on their bodies.
            inner.known.clear();
            inner.contexts.insert(generation, meta);
            if inner.contexts.len() > MAX_CONTEXTS
                && let Some(oldest) = inner.contexts.keys().min().copied()
            {
                inner.contexts.remove(&oldest);
            }
        }
        generation
    }

    fn end(self: &Arc<Self>) {
        {
            let mut inner = self.lock();
            inner.active_evaluations = inner.active_evaluations.saturating_sub(1);
        }
        self.pump();
    }

    fn register_all<'a>(
        self: &Arc<Self>,
        generation: u64,
        bodies: impl IntoIterator<Item = (&'a str, ProofPriority, &'a SolidBody)>,
    ) {
        let mut inner = self.lock();
        let Some(request) = inner.contexts.get(&generation).cloned() else {
            return;
        };
        for (producer, priority, body) in bodies {
            // A proved or refused body is already decided; no job for it and no
            // clone of its shared body. The one shared ProofState is the memo.
            if !body.needs_proof() {
                continue;
            }
            let id = body.proof_id();
            let known = inner.known.get(&id).cloned();
            if needs_proof(&body.proof_state(), known.is_some()) {
                let meta = JobMeta {
                    generation,
                    request: Arc::clone(&request),
                    producer: producer.to_owned(),
                };
                inner.known.insert(id, (priority, meta.clone()));
                inner.stats.queued += 1;
                let job = Job { body: body.clone(), meta };
                if priority == ProofPriority::Displayed {
                    inner.high.push_back(job);
                } else {
                    inner.low.push_back(job);
                }
                continue;
            }
            // Promote a body first seen as an intermediate (e.g. a translated target
            // sharing its primitive's body): a displayed job is queued ahead of it,
            // still attributed to the node that produced it.
            if let Some((existing, original)) = known
                && existing != priority
                && priority == ProofPriority::Displayed
            {
                inner.known.insert(id, (ProofPriority::Displayed, original.clone()));
                inner.high.push_back(Job { body: body.clone(), meta: original });
            }
        }
        let active = inner.active_evaluations > 0;
        drop(inner);
        // Nothing can start while an evaluation is running; the guard's drop pumps.
        if !active {
            self.pump();
        }
    }

    /// Start as many proofs as the idle capacity allows. Called when the last evaluation of a
    /// batch ends and whenever a proof finishes. A started proof reports back to this queue
    /// (it holds an `Arc` to it), never to the process-wide one, so a private queue keeps its
    /// own results.
    fn pump(self: &Arc<Self>) {
        let pool = shared_pool();
        // Leave at least half the workers for evaluation: proofs are background work.
        let max_in_flight = (pool.workers() / 2).max(1);
        loop {
            let mut inner = self.lock();
            if inner.active_evaluations > 0 || inner.in_flight >= max_in_flight {
                break;
            }
            let current = self.generation.load(Ordering::Acquire);
            let Some(job) = inner.high.pop_front().or_else(|| inner.low.pop_front()) else {
                self.idle.notify_all();
                break;
            };
            let id = job.body.proof_id();
            // Unstarted work for an older request is deprioritised away; its body keeps the
            // Unknown state and is registered again if a later request still uses it.
            if job.meta.generation < current {
                inner.stats.skipped += 1;
                continue;
            }
            if job.body.is_proved() || job.body.refused().is_some() {
                inner.stats.skipped += 1;
                continue;
            }
            if inner.claimed.contains(&id) {
                continue;
            }
            // Only a closed body has a proof to take; an open sheet stays Unknown forever.
            if !is_closed_solid(&job.body) {
                inner.stats.skipped += 1;
                continue;
            }
            inner.claimed.insert(id);
            inner.in_flight += 1;
            inner.stats.proofs_run += 1;
            if inner.first_started.is_none() {
                inner.first_started = Some(Instant::now());
            }
            let meta = job.meta;
            let body = job.body.clone();
            let started = Instant::now();
            let queue = Arc::clone(self);
            let receiver =
                pool.submit_background(seed_for(id), CancelToken::new(), move |kernel| {
                    let outcome = body.prove(kernel);
                    queue.finish(meta, id, outcome, started.elapsed());
                });
            // The job's outcome is recorded by the closure itself; dropping the receiver keeps
            // the pool from storing a result nobody reads.
            drop(receiver);
        }
    }

    fn finish(
        self: &Arc<Self>,
        meta: JobMeta,
        id: usize,
        outcome: Result<(), nodecraft_ops::SessionError>,
        elapsed: Duration,
    ) {
        let mut inner = self.lock();
        inner.claimed.remove(&id);
        inner.in_flight = inner.in_flight.saturating_sub(1);
        inner.stats.total_ms += elapsed.as_secs_f64() * 1000.0;
        match outcome {
            Ok(()) => inner.stats.proved += 1,
            Err(nodecraft_ops::SessionError::Invalid(failure)) => {
                inner.stats.refused += 1;
                let report = ProofReport {
                    generation: meta.generation,
                    request_id: meta.request.request_id,
                    tab_id: meta.request.tab_id.clone(),
                    graph_revision: meta.request.graph_revision.clone(),
                    producer: meta.producer.clone(),
                    failure,
                    downstream: meta.request.context.downstream(&meta.producer),
                };
                if inner.reports.len() >= MAX_REPORTS {
                    inner.reports.pop_front();
                }
                inner.reports.push_back(report);
            }
            Err(_) => inner.stats.failed += 1,
        }
        if inner.in_flight == 0 && inner.high.is_empty() && inner.low.is_empty() {
            if let Some(first) = inner.first_started.take() {
                inner.stats.wall_ms += first.elapsed().as_secs_f64() * 1000.0;
            }
        }
        drop(inner);
        self.idle.notify_all();
        self.pump();
    }

    fn wait(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut inner = self.lock();
        loop {
            if inner.in_flight == 0 && inner.high.is_empty() && inner.low.is_empty() {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            let (guard, _) = self
                .idle
                .wait_timeout(inner, deadline.saturating_duration_since(now))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            inner = guard;
        }
    }
}

static QUEUE: OnceLock<Arc<Queue>> = OnceLock::new();

fn queue() -> &'static Arc<Queue> {
    QUEUE.get_or_init(|| Arc::new(Queue::new()))
}

fn seed_for(id: usize) -> u64 {
    (id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// Whether the body is a closed solid (closed manifold or closed solids touching). Only such
/// a body has an outward proof; an open sheet is skipped and stays Unknown.
fn is_closed_solid(body: &SolidBody) -> bool {
    use nodecraft_solidcore::solidcore::query::api_queries::brep_is_closed_touching_solids;
    use nodecraft_solidcore::solidcore::topology::brep::p2::is_manifold_solid;
    is_manifold_solid(body.brep()).unwrap_or(false)
        || brep_is_closed_touching_solids(body.brep()).unwrap_or(false)
}

/// A running evaluation's queue registration token; dropping it lets the proofs start.
pub(crate) struct EvaluationGuard {
    generation: u64,
    queue: Arc<Queue>,
}

impl EvaluationGuard {
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

impl Drop for EvaluationGuard {
    fn drop(&mut self) {
        self.queue.end();
    }
}

/// Start a generation for this request. Queued proof work for older requests is deprioritised;
/// nothing is submitted until the guard drops (the exact pass has returned).
pub(crate) fn begin(request: &Request) -> EvaluationGuard {
    let queue = Arc::clone(queue());
    let generation = queue.begin(request);
    EvaluationGuard { generation, queue }
}

/// Hand a node's output bodies to the queue. The first node to register a body is its
/// producer; a later target registration only upgrades its priority.
pub(crate) fn register<'a>(
    generation: u64,
    producer: &str,
    bodies: impl IntoIterator<Item = &'a SolidBody>,
    priority: ProofPriority,
) {
    register_all(generation, bodies.into_iter().map(|body| (producer, priority, body)));
}

/// [`register`] for a whole evaluation pass: one queue lock for the pass and
/// one `known` lookup per body, instead of one lock per Part.
pub(crate) fn register_all<'a>(
    generation: u64,
    bodies: impl IntoIterator<Item = (&'a str, ProofPriority, &'a SolidBody)>,
) {
    queue().register_all(generation, bodies);
}

/// Take the refusals that finished since the last call. The application applies them to the
/// nodes of the evaluation generation they name.
#[must_use]
pub fn take_reports() -> Vec<ProofReport> {
    queue().lock().reports.drain(..).collect()
}

/// A snapshot of the queue's counters.
#[must_use]
pub fn stats() -> ProofStats {
    queue().lock().stats
}

/// Whether no proof is queued or running.
#[must_use]
pub fn is_idle() -> bool {
    let inner = queue().lock();
    inner.in_flight == 0 && inner.high.is_empty() && inner.low.is_empty()
}

/// Wait until the queue has drained, or `timeout` passes. Returns whether it drained.
#[must_use]
pub fn wait(timeout: Duration) -> bool {
    queue().wait(timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        Request {
            request_id: 7,
            tab_id: None,
            graph_revision: None,
            targets: vec!["c".into()],
            mesh_targets: None,
            nodes: Vec::new(),
            connections: Vec::new(),
            binary_mesh: false,
            project_directory: None,
            progress: false,
            run_exporters: false,
            mesh_tolerance: None,
            preview_detail: None,
            bypass_cache: false,
        }
    }

    #[test]
    fn downstream_is_transitive_and_sorted() {
        let context = GraphContext {
            connections: vec![
                ("a".into(), "b".into()),
                ("b".into(), "c".into()),
                ("c".into(), "d".into()),
                ("x".into(), "d".into()),
            ],
        };
        assert_eq!(context.downstream("a"), vec!["b", "c", "d"]);
        assert_eq!(context.downstream("b"), vec!["c", "d"]);
        assert!(context.downstream("d").is_empty());
    }

    #[test]
    fn only_an_unknown_un_queued_body_takes_a_job() {
        assert!(needs_proof(&ProofState::Unknown, false));
        assert!(!needs_proof(&ProofState::Unknown, true), "already queued in this generation");
        assert!(!needs_proof(&ProofState::Proved, false), "a proved body is decided");
        assert!(
            !needs_proof(&ProofState::Refused("inside out".into()), false),
            "a refused body is decided",
        );
    }

    #[test]
    fn a_displayed_target_raises_an_intermediate_body_to_high() {
        let queue = Arc::new(Queue::new());
        let generation = queue.begin(&request());
        let body = SolidBody::new(nodecraft_solidcore::solidcore::topology::Brep::default());
        queue.register_all(generation, [("a", ProofPriority::Intermediate, &body)]);
        {
            let inner = queue.lock();
            assert_eq!(inner.stats.queued, 1);
            assert_eq!(
                inner.known.get(&body.proof_id()).map(|(priority, _)| *priority),
                Some(ProofPriority::Intermediate),
            );
            assert_eq!(inner.low.len(), 1);
            assert!(inner.high.is_empty());
        }
        queue.register_all(generation, [("b", ProofPriority::Displayed, &body)]);
        {
            let inner = queue.lock();
            assert_eq!(inner.stats.queued, 1, "a promotion does not queue a second body");
            assert_eq!(
                inner.known.get(&body.proof_id()).map(|(priority, _)| *priority),
                Some(ProofPriority::Displayed),
                "the displayed target raises the body's priority",
            );
            assert_eq!(inner.high.len(), 1, "the promoted job goes to the high queue");
        }
        queue.end();
    }

    #[test]
    fn a_generation_sees_its_own_request() {
        let queue = Arc::new(Queue::new());
        let first = queue.begin(&request());
        queue.end();
        let second = queue.begin(&request());
        assert_eq!(second, first + 1);
        queue.end();
        let inner = queue.lock();
        assert!(inner.contexts.get(&second).is_some());
        assert!(inner.active_evaluations == 0);
    }

    /// A refused body is attributed to the node that first produced it and names every
    /// downstream node, transitively; the body keeps the refusal so nothing proves it again.
    ///
    /// The queue is private to this test, so no other test's evaluation can start a newer
    /// generation and skip this job: one pass is enough, with no retry.
    #[test]
    fn a_refused_body_is_reported_on_its_producer_with_downstream_warnings() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        use nodecraft_solidcore::solidcore::topology::face::p1::flip_face_orientation;
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["c"],
            "meshTargets": [],
            "nodes": [],
            "connections": [
                {"from": "a", "to": "b", "in": "shape", "out": "shape"},
                {"from": "b", "to": "c", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("request");
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let mut brep = create_box(&mut kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
            .expect("box");
        for face in brep.face_list().to_vec() {
            flip_face_orientation(&mut brep, face).expect("flip");
        }
        let body = SolidBody::new(brep);

        let queue = Arc::new(Queue::new());
        let generation = queue.begin(&request);
        queue.register_all(generation, [("a", ProofPriority::Displayed, &body)]);
        queue.end();
        assert!(queue.wait(Duration::from_secs(30)), "the private queue drains");

        let reports: Vec<ProofReport> = queue.lock().reports.drain(..).collect();
        let report =
            reports.iter().find(|report| report.producer == "a").expect("the producer is reported");
        assert_eq!(report.downstream, vec!["b", "c"]);
        assert!(report.failure.contains("closed solid"), "{}", report.failure);
        assert_eq!(report.generation, generation);
        assert!(body.refused().is_some(), "the body keeps the refusal");
        assert_eq!(body.proof_attempts(), 1, "one proof, not one per reader");
        assert_eq!(queue.lock().stats.refused, 1);
    }
}
