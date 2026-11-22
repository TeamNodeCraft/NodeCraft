//! Inert, thread-local phase timing for benchmark harnesses.
//!
//! Instrumentation is off unless a caller wraps one evaluation in
//! [`capture_eval_phases`]. While off, the engine pays only a thread-local
//! `is_some()` check per phase and never allocates here.

use std::cell::RefCell;
use std::time::Instant;

/// Wall-clock milliseconds for one `evaluate_mode` call, split at the points a
/// benchmark harness cares about. Fields are zero when a phase did not run.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalPhaseTimings {
    pub prewarm_ms: f64,
    pub walk_ms: f64,
    pub order_ms: f64,
    pub loop_ms: f64,
    pub tail_ms: f64,
    pub total_ms: f64,
}

thread_local! {
    static CAPTURE: RefCell<Option<Vec<EvalPhaseTimings>>> = const { RefCell::new(None) };
}

/// Run `f` with phase capture installed on the current thread.
///
/// Returns `f`'s result and every phase record it produced. A nested capture
/// saves and restores the outer capture state.
pub fn capture_eval_phases<R>(f: impl FnOnce() -> R) -> (R, Vec<EvalPhaseTimings>) {
    let previous = CAPTURE.with(|capture| capture.borrow_mut().replace(Vec::new()));
    let result = f();
    let recorded = CAPTURE.with(|capture| capture.borrow_mut().take().unwrap_or_default());
    CAPTURE.with(|capture| *capture.borrow_mut() = previous);
    (result, recorded)
}

/// Start a phase clock only while capture is active, so an uninstrumented run
/// reads the thread-local state and does nothing else.
pub(crate) fn phase_clock() -> Option<Instant> {
    CAPTURE.with(|capture| if capture.borrow().is_some() { Some(Instant::now()) } else { None })
}

/// Milliseconds elapsed since `start`, or `0.0` when the phase was not clocked.
pub(crate) fn phase_elapsed_ms(start: Option<Instant>) -> f64 {
    start.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1000.0)
}

/// Append one evaluation's timings when capture is active.
pub(crate) fn record(t: EvalPhaseTimings) {
    CAPTURE.with(|capture| {
        if let Some(recorded) = capture.borrow_mut().as_mut() {
            recorded.push(t);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineCache, ImportContext, evaluate_exact_with_imports};
    use nodecraft_protocol::{CacheState, Connection, Node, Request};

    /// A 64-node box-plus-translate chain, cached on every node so the second
    /// pass exercises the hit path.
    fn chain_request() -> Request {
        let cached =
            |key: &str| CacheState { key: Some(key.to_owned()), state: Some("cached".to_owned()) };
        let mut nodes = vec![Node {
            id: "n0".to_owned(),
            node_type: "primitive".to_owned(),
            values: serde_json::json!({"primitiveType": "box"}),
            suppressed: false,
            cache: Some(cached("k0")),
        }];
        let mut connections = Vec::new();
        for index in 1..64 {
            let id = format!("n{index}");
            nodes.push(Node {
                id: id.clone(),
                node_type: "translate".to_owned(),
                values: serde_json::json!({"x": 1.0}),
                suppressed: false,
                cache: Some(cached(&format!("k{index}"))),
            });
            connections.push(Connection {
                from: format!("n{}", index - 1),
                to: id,
                input: "shape".to_owned(),
                out: "shape".to_owned(),
            });
        }
        Request {
            request_id: 1,
            tab_id: None,
            graph_revision: None,
            targets: vec!["n63".to_owned()],
            mesh_targets: None,
            nodes,
            connections,
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
    #[ignore = "benchmark: engine phase capture"]
    fn capture_phase_timings_for_a_64_node_translate_chain() {
        let request = chain_request();
        let out = evaluate_exact_with_imports(
            request.clone(),
            EngineCache::new(),
            None,
            ImportContext::default(),
        );
        assert!(out.response.node_errors.is_empty(), "{:?}", out.response.node_errors);

        let (out2, phases) = capture_eval_phases(|| {
            evaluate_exact_with_imports(request.clone(), out.cache, None, ImportContext::default())
        });
        assert_eq!(phases.len(), 1, "one evaluation must record once");
        let phase = phases[0];
        eprintln!("{phase:?}");
        assert!(phase.total_ms > 0.0, "total must be measured: {phase:?}");
        let sum = phase.prewarm_ms + phase.walk_ms + phase.order_ms + phase.loop_ms + phase.tail_ms;
        assert!(sum <= phase.total_ms + 0.05, "phases {phase:?} sum to {sum}ms, above total");
        assert!(out2.response.node_errors.is_empty(), "{:?}", out2.response.node_errors);
        let stats = out2.response.cache_stats.expect("cache stats");
        assert!(stats.hits > 0, "second pass must hit the node cache: {stats:?}");
    }
}
