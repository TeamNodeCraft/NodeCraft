//! Latency probe for transform edits that must produce a mesh.
//!
//! Run manually:
//! `cargo test -p nodecraft-engine --test transform_mesh_latency -- --ignored --nocapture`

use nodecraft_engine::{EngineCache, evaluate_exact};
use nodecraft_protocol::Request;
use std::time::Instant;

const SOLIDS: usize = 4;
const MOVES: usize = 24;

fn request(x: [f64; SOLIDS]) -> Request {
    let mut nodes = Vec::new();
    let mut connections = Vec::new();
    for index in 0..SOLIDS {
        nodes.push(serde_json::json!({
            "id": format!("box{index}"),
            "type": "primitive",
            "values": {"primitiveType": "box", "length": 40.0, "width": 30.0, "height": 20.0},
            "cache": {"key": format!("box{index}")}
        }));
        nodes.push(serde_json::json!({
            "id": format!("move{index}"),
            "type": "translate",
            "values": {"x": x[index], "y": index as f64 * 50.0, "z": 0.0},
            "cache": {"key": format!("move{index}-{}", index)}
        }));
        nodes.push(serde_json::json!({
            "id": format!("mirror{index}"),
            "type": "mirror",
            "values": {"plane": "yz", "offset": 0.0, "combine": false},
            "cache": {"key": format!("mirror{index}")}
        }));
        nodes.push(serde_json::json!({
            "id": format!("group{index}"),
            "type": "combine",
            "values": {},
            "cache": {"key": format!("group{index}")}
        }));
        connections.push(serde_json::json!({"from": format!("box{index}"), "to": format!("move{index}"), "in": "shape", "out": "shape"}));
        connections.push(serde_json::json!({"from": format!("move{index}"), "to": format!("mirror{index}"), "in": "shape", "out": "shape"}));
        connections.push(serde_json::json!({"from": format!("move{index}"), "to": format!("group{index}"), "in": "shape", "out": "shape"}));
        connections.push(serde_json::json!({"from": format!("mirror{index}"), "to": format!("group{index}"), "in": "shape", "out": "shape"}));
    }
    nodes.push(serde_json::json!({
        "id": "result",
        "type": "combine",
        "values": {},
        "cache": {"key": "result"}
    }));
    for index in 0..SOLIDS {
        connections.push(serde_json::json!({"from": format!("group{index}"), "to": "result", "in": "shape", "out": "shape"}));
    }
    // Unrelated background graph: proves prewarm is scoped to the target
    // closure instead of scheduling the entire document on every drag.
    for index in 0..200 {
        nodes.push(serde_json::json!({
            "id": format!("bg-box{index}"),
            "type": "primitive",
            "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0},
            "cache": {"key": format!("bg-box{index}")}
        }));
        nodes.push(serde_json::json!({
            "id": format!("bg-move{index}"),
            "type": "translate",
            "values": {"x": index as f64 * 20.0},
            "cache": {"key": format!("bg-move{index}")}
        }));
        connections.push(serde_json::json!({"from": format!("bg-box{index}"), "to": format!("bg-move{index}"), "in": "shape", "out": "shape"}));
    }
    serde_json::from_value(serde_json::json!({
        "requestId": 1,
        "targets": ["result"],
        "meshTargets": ["result"],
        "binaryMesh": false,
        "nodes": nodes,
        "connections": connections
    }))
    .expect("latency request")
}

#[test]
#[ignore = "latency probe; run explicitly with --ignored --nocapture"]
fn transform_moves_emit_a_mesh() {
    let mut cache = EngineCache::new();
    let mut xs = [0.0; SOLIDS];
    // Warm all primitive/mirror/combine caches first.
    let warm = evaluate_exact(request(xs), cache, None);
    cache = warm.cache;
    let mut display_cache = nodecraft_display::PlanCache::new(256);
    let warm_mesh = warm.exact_parts.get("result").and_then(|part| {
        nodecraft_display::mesh_part(part, nodecraft_display::DisplayQuality::Full(0.35), &mut display_cache).ok()
    });
    assert!(
        warm_mesh.is_some_and(|display| !display.mesh.vertices.is_empty()),
        "warm evaluation produced no mesh"
    );

    let mut wall_ms = Vec::with_capacity(MOVES);
    let mut engine_ms = Vec::with_capacity(MOVES);
    for step in 0..MOVES {
        let index = step % SOLIDS;
        xs[index] += 5.0;
        let started = Instant::now();
        let output = evaluate_exact(request(xs), cache, None);
        cache = output.cache;
        let mesh = output.exact_parts.get("result").and_then(|part| {
            nodecraft_display::mesh_part(part, nodecraft_display::DisplayQuality::Full(0.35), &mut display_cache).ok()
        }).expect("mesh result");
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        assert!(!mesh.mesh.vertices.is_empty(), "step {step} produced no mesh");
        wall_ms.push(elapsed);
        engine_ms.push(output.response.evaluation_ms);
        println!(
            "transform step {step:02} x[{index}]={:>6.1} wall={elapsed:8.3} ms engine={:8.3} ms verts={}",
            xs[index],
            output.response.evaluation_ms,
            mesh.mesh.vertices.len()
        );
    }

    let average = wall_ms.iter().sum::<f64>() / wall_ms.len() as f64;
    let maximum = wall_ms.iter().copied().fold(0.0, f64::max);
    let engine_average = engine_ms.iter().sum::<f64>() / engine_ms.len() as f64;
    println!(
        "transform latency: wall avg={average:.3} ms max={maximum:.3} ms; engine avg={engine_average:.3} ms"
    );
    assert!(maximum < 5_000.0, "transform latency is unexpectedly high: {maximum:.3} ms");
}
