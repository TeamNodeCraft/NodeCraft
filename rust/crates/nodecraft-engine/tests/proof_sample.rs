//! Background proof timing for one sample document.
//!
//! The user-owned `sample_compare` harness is read-only, so it reports the critical path only
//! (evaluation returns before the background proofs). This sibling test reads the same
//! `NODECRAFT_SAMPLE` document, prints the critical-path `evaluationMs` and then waits for the
//! background proof queue to drain and prints `PROOFS {json}` with proofs run / refused / ms.
//!
//! Run:
//! ```text
//! NODECRAFT_SAMPLE=<abs path> <sample_compare binary> --exact sample_report --nocapture
//! NODECRAFT_SAMPLE=<abs path> <proof_sample binary> --exact sample_proofs --nocapture
//! ```

use std::collections::HashSet;
use std::time::Duration;

use nodecraft_protocol::Request;

#[test]
fn sample_proofs() {
    let Ok(path) = std::env::var("NODECRAFT_SAMPLE") else {
        eprintln!("NODECRAFT_SAMPLE not set");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read sample");
    let document: serde_json::Value = serde_json::from_str(&text).expect("sample json");
    let root = document.get("root").cloned().unwrap_or_else(|| document.clone());
    let nodes = root.get("nodes").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    let connections = root.get("connections").and_then(|value| value.as_array()).cloned().unwrap_or_default();
    let produced: HashSet<String> = connections
        .iter()
        .filter_map(|link| link.get("from").and_then(|value| value.as_str()).map(str::to_owned))
        .collect();
    let mut targets: Vec<String> = nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(|value| value.as_str()).map(str::to_owned))
        .filter(|id| !produced.contains(id))
        .collect();
    if targets.is_empty() {
        targets = nodes.iter().filter_map(|node| node.get("id").and_then(|value| value.as_str()).map(str::to_owned)).collect();
    }
    let request: Request = serde_json::from_value(serde_json::json!({
        "requestId": 1,
        "targets": targets,
        "meshTargets": [],
        "nodes": nodes,
        "connections": connections,
        "runExporters": false,
        "bypassCache": true,
    }))
    .expect("request");
    // The critical path: this returns before the background proofs.
    let output = nodecraft_engine::evaluate_exact(request, nodecraft_engine::EngineCache::new(), None);
    let evaluation_ms = output.response.evaluation_ms;
    let generation = output.proof_generation;
    let errors = output.response.node_errors.len();
    // Background time, measured separately after the caller already has its result.
    let drained = nodecraft_engine::proof_queue::wait(Duration::from_secs(600));
    let stats = nodecraft_engine::proof_queue::stats();
    let refused = nodecraft_engine::proof_queue::take_reports();
    println!(
        "PROOFS {}",
        serde_json::json!({
            "sample": path,
            "evaluationMs": evaluation_ms,
            "generation": generation,
            "nodeErrors": errors,
            "drained": drained,
            "queued": stats.queued,
            "proofsRun": stats.proofs_run,
            "proved": stats.proved,
            "refused": stats.refused,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "proofMs": stats.total_ms,
            "wallMs": stats.wall_ms,
            "reports": refused.len(),
        })
    );
}
