//! The exact pass hands its bodies to the background proof queue.
//!
//! This test observes the process-wide proof queue (its counters and report inbox), and
//! every evaluation that starts bumps that queue's generation, which by design skips older
//! unstarted proofs. In the library's unit-test binary hundreds of tests evaluate in
//! parallel, so this assertion could not be relied on there. As an integration test it runs
//! in its own process, where the queue sees only this test's evaluation.

use std::time::Duration;

use nodecraft_engine::{EngineCache, evaluate_exact, proof_queue};
use nodecraft_protocol::Request;

/// The exact pass hands its produced bodies to the background queue: the boxes of a
/// Boolean are queued, and the queue proves them after evaluation returns. Without the
/// queue these bodies stay Unknown forever.
#[test]
fn an_exact_pass_hands_its_bodies_to_the_background_proof_queue() {
    let request: Request = serde_json::from_value(serde_json::json!({
        "requestId": 1,
        "targets": ["u"],
        "meshTargets": [],
        "nodes": [
            {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "a", "state": "cached"}},
            {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "b", "state": "cached"}},
            {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u", "state": "cached"}}
        ],
        "connections": [
            {"from": "a", "to": "u", "in": "a", "out": "shape"},
            {"from": "b", "to": "u", "in": "b", "out": "shape"}
        ]
    }))
    .expect("request");
    let before = proof_queue::stats().queued;
    let output = evaluate_exact(request, EngineCache::new(), None);
    assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
    let after = proof_queue::stats().queued;
    assert!(after > before, "the exact pass queued no body: {before} -> {after}");
    assert!(output.proof_generation > 0);
    // Evaluation returned before the proofs; a caller that wants the proof time waits.
    assert!(proof_queue::wait(Duration::from_secs(120)));
    assert!(proof_queue::stats().proved >= 1);
    assert!(proof_queue::take_reports().is_empty(), "a valid Boolean must not be refused");
}
