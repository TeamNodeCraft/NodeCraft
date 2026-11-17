//! Regression probes for mismatches found by the first Lean pass over the
//! session's node identity and dirty-set code.
//!
//! Both findings are fixed and run in the normal suite: Finding 2 covers the
//! shared hash encoder's field boundaries, and Finding 3 covers identity-derived
//! transaction dirty sets. Nothing here is ignored.

use nodecraft_session::{
    Command, GraphConnection, GraphDefinition, GraphDocument, GraphNode, Session,
};
use serde_json::json;

fn node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
    GraphNode {
        id: id.to_owned(),
        node_type: node_type.to_owned(),
        values,
        suppressed: false,
        x: 0.0,
        y: 0.0,
        expressions: Default::default(),
    }
}

/// `node_hashes_with_sources` must bind the id/type field boundary. Before the
/// fix it wrote the id and the type back-to-back without a length prefix or
/// separator, so node `(id="ab", type="c")` and node `(id="a", type="bc")`
/// had the same byte stream when their values were equal and therefore the
/// same `ContentHash`. `Hash128::write_framed` now prefixes every
/// variable-length field with its 8-byte little-endian length, so the two
/// documents hash differently.
#[test]
fn node_hash_distinguishes_id_and_type_boundaries() {
    let mut left = GraphDocument::new("left");
    left.root.nodes.push(node("ab", "c", json!({})));
    let mut right = GraphDocument::new("right");
    right.root.nodes.push(node("a", "bc", json!({})));

    let left_hash = left.node_hashes().expect("left hashes")["ab"];
    let right_hash = right.node_hashes().expect("right hashes")["a"];
    assert_ne!(
        left_hash, right_hash,
        "the node hash must bind the id and type fields unambiguously"
    );
}

/// A graph card's `values.name` is presentation-only: `node_hashes_with_sources`
/// removes it before hashing (`rust/crates/nodecraft-session/src/lib.rs:305-311`).
/// The transaction dirty set is now derived from those hashes, not from the
/// `SetValue` seed, so renaming the card dirties nothing.
#[test]
fn renaming_a_graph_card_does_not_dirty_geometry() {
    let mut document = GraphDocument::new("rename-graph-card");
    document.definitions.push(GraphDefinition {
        id: "feature".to_owned(),
        name: "Feature".to_owned(),
        kind: "graph".to_owned(),
        interface: Default::default(),
        nodes: vec![node("box", "primitive", json!({"primitiveType": "box"}))],
        connections: Vec::new(),
    });
    document.root.nodes.push(node(
        "g",
        "graph",
        json!({"definition": "feature", "name": "Old name"}),
    ));
    document.root.nodes.push(node("t", "translate", json!({"x": 5.0})));
    document.root.connections.push(GraphConnection {
        from: "g".to_owned(),
        to: "t".to_owned(),
        input: "shape".to_owned(),
        out: "shape".to_owned(),
    });

    let mut session = Session::new(document);
    let before = session.document().node_hashes().expect("before hashes");
    let change = session
        .apply_commands(&[Command::SetValue {
            node: "g".to_owned(),
            field: "name".to_owned(),
            value: json!("New name"),
        }])
        .expect("rename");
    let after = session.document().node_hashes().expect("after hashes");

    assert!(
        change.dirty_nodes.is_empty(),
        "a presentation-only name must not dirty geometry, got {:?}",
        change.dirty_nodes
    );
    assert_eq!(before, after, "the name is excluded from node identity");
}
