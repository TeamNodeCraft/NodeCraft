//! Regression probes for mismatches found by the first Lean pass over the
//! graph engine.
//!
//! `suppression_walk_cap_misses_live_input_after_64_translates` is the fixed
//! regression for Finding 1 and runs with the normal suite. Findings 2 and 3
//! live in the `nodecraft-session` probe file and also run normally now.
//!
//! Run one explicitly with, for example:
//!
//! ```text
//! cargo test -p nodecraft-engine --test lean_findings -- --nocapture
//! ```

use nodecraft_engine::{EngineCache, evaluate_exact};
use nodecraft_protocol::{CacheState, Connection, Node, Request};
use serde_json::json;

fn node(id: &str, node_type: &str, values: serde_json::Value, suppressed: bool) -> Node {
    Node {
        id: id.to_owned(),
        node_type: node_type.to_owned(),
        values,
        suppressed,
        cache: Some(CacheState {
            key: Some(id.to_owned()),
            state: Some("cached".to_owned()),
        }),
    }
}

fn edge(from: &str, to: &str, input: &str) -> Connection {
    Connection {
        from: from.to_owned(),
        to: to.to_owned(),
        input: input.to_owned(),
        out: "shape".to_owned(),
    }
}

/// A chain of `length` unsuppressed Translate nodes between `source` and the
/// returned terminal id. Every node is live and the source is a primitive.
fn translate_chain(
    prefix: &str,
    source: &str,
    length: usize,
    nodes: &mut Vec<Node>,
    connections: &mut Vec<Connection>,
) -> String {
    let mut previous = source.to_owned();
    for index in 0..length {
        let id = format!("{prefix}{index:02}");
        nodes.push(node(&id, "translate", json!({"x": 0.0}), false));
        connections.push(edge(&previous, &id, "shape"));
        previous = id;
    }
    previous
}

/// A chain of 64 unsuppressed Translate nodes must not exhaust any walk budget:
/// the walk has to follow the whole finite chain and reach the live primitive
/// behind it. If it reports the chain's output as suppressed, both Boolean
/// inputs are classified `Inactive` and the exact Boolean reports "no active
/// operand" instead of unioning the two boxes.
#[test]
fn suppression_walk_cap_misses_live_input_after_64_translates() {
    let mut nodes = vec![
        node(
            "box-a",
            "primitive",
            json!({"primitiveType": "box", "length": 10, "width": 10, "height": 10}),
            false,
        ),
        node(
            "box-b",
            "primitive",
            json!({"primitiveType": "box", "length": 10, "width": 10, "height": 10}),
            false,
        ),
    ];
    let mut connections = Vec::new();
    let a = translate_chain("a", "box-a", 64, &mut nodes, &mut connections);
    let b = translate_chain("b", "box-b", 64, &mut nodes, &mut connections);
    nodes.push(node("out", "boolean", json!({"mode": "union"}), false));
    connections.push(edge(&a, "out", "a"));
    connections.push(edge(&b, "out", "b"));

    let request = Request {
        request_id: 1,
        tab_id: None,
        graph_revision: None,
        targets: vec!["out".to_owned()],
        mesh_targets: Some(Vec::new()),
        nodes,
        connections,
        binary_mesh: false,
        project_directory: None,
        progress: false,
        run_exporters: false,
        mesh_tolerance: None,
        preview_detail: None,
        comparison_targets: None,
        bypass_cache: false,
    };

    let output = evaluate_exact(request, EngineCache::new(), None);
    assert!(
        output.response.node_errors.is_empty(),
        "both operands descend from live primitives through 64 unsuppressed \
         Translate nodes, but the Boolean was rejected: {:?}",
        output.response.node_errors
    );
    assert!(
        output.exact_parts.contains_key("out"),
        "a live Boolean must produce an exact Part"
    );
}
