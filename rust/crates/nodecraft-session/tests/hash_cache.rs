//! Revision hash-cache regression probes.
//!
//! `Session` computes a revision's node hashes once, when the revision is
//! created, and reuses the stored `Arc` for dirty diffs and for the app/bridge
//! request. These tests count full `GraphDocument::node_hashes*` passes with the
//! thread-local counter so parallel tests cannot pollute the measurement.

use std::collections::BTreeMap;
use std::sync::Arc;

use nodecraft_session::{
    Command, GraphConnection, GraphDocument, GraphNode, Session, SessionError,
    reset_thread_node_hash_pass_count, thread_node_hash_pass_count,
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

/// One source importer, one descendant, and one unrelated node.
fn document() -> GraphDocument {
    let mut document = GraphDocument::new("hash-cache");
    document.nodes = vec![
        node("imp", "stepimporter", json!({"path": "input.step"})),
        node("t", "translate", json!({"x": 5.0})),
        node("u", "primitive", json!({"primitiveType": "box", "length": 10.0})),
    ];
    document.connections = vec![GraphConnection {
        from: "imp".into(),
        to: "t".into(),
        input: "shape".into(),
        out: "shape".into(),
    }];
    document
}

fn set_length(node: &str, length: f64) -> Command {
    Command::SetValue {
        node: node.to_owned(),
        field: "length".into(),
        value: json!(length),
    }
}

#[test]
fn new_edit_hashes_once_and_reads_are_free() {
    reset_thread_node_hash_pass_count();
    let mut session = Session::new(document());
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "Session::new must hash the initial revision once"
    );
    let before = session.node_hashes().expect("initial hashes");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "reading the cached revision hashes must not hash"
    );

    reset_thread_node_hash_pass_count();
    let change = session
        .apply_commands(&[set_length("u", 20.0)])
        .expect("edit applies");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "one edit must perform exactly one full hash pass"
    );
    assert_eq!(change.dirty_nodes, vec!["u"]);

    let after = session.node_hashes().expect("current hashes");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "reading the current hashes again must be an Arc clone"
    );
    assert_ne!(before["u"], after["u"]);
    assert_eq!(before["imp"], after["imp"]);
    assert_eq!(before["t"], after["t"]);
}

#[test]
fn no_op_edit_hashes_nothing_and_keeps_the_cache() {
    let mut session = Session::new(document());
    let command = set_length("u", 20.0);
    session.apply_commands(&[command.clone()]).expect("first edit");
    let revision = session.revision();
    let hashes = session.node_hashes().expect("hashes after edit");

    reset_thread_node_hash_pass_count();
    let change = session
        .apply_commands(&[command])
        .expect("no-op command applies");
    assert!(!change.changed);
    assert_eq!(change.revision, revision);
    assert_eq!(
        thread_node_hash_pass_count(),
        0,
        "a no-op must not hash the document"
    );
    assert_eq!(session.revision(), revision);
    assert!(
        Arc::ptr_eq(&hashes, &session.node_hashes().expect("unchanged hashes")),
        "a no-op must leave the previous revision cache in place"
    );
}

#[test]
fn undo_and_redo_each_hash_once_and_replace_the_cache() {
    let mut session = Session::new(document());
    let initial = session.node_hashes().expect("initial hashes");
    session
        .apply_commands(&[set_length("u", 20.0)])
        .expect("edit applies");
    let edited = session.node_hashes().expect("edited hashes");

    reset_thread_node_hash_pass_count();
    let undone = session.undo().expect("undo runs").expect("undo entry");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "undo must recompute exactly once"
    );
    assert_eq!(undone.dirty_nodes, vec!["u"]);
    let restored = session.node_hashes().expect("undo hashes");
    assert_eq!(*restored, *initial, "undo must restore the initial keys");
    assert!(
        !Arc::ptr_eq(&restored, &initial),
        "undo must invalidate and recompute, not keep the old map"
    );

    reset_thread_node_hash_pass_count();
    let redone = session.redo().expect("redo runs").expect("redo entry");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "redo must recompute exactly once"
    );
    assert_eq!(redone.dirty_nodes, vec!["u"]);
    let redone_hashes = session.node_hashes().expect("redo hashes");
    assert_eq!(*redone_hashes, *edited);
    assert!(!Arc::ptr_eq(&redone_hashes, &edited));
}

#[test]
fn source_change_recomputes_and_dirties_importer_descendants() {
    let mut session = Session::new(document());
    let initial = session.node_hashes().expect("initial hashes");

    let sources = BTreeMap::from([("imp".to_owned(), "source-v1".to_owned())]);
    session.set_source_hashes(sources.clone());
    reset_thread_node_hash_pass_count();
    let with_source = session
        .hashes_with_sources(&sources)
        .expect("source hashes");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "a source change falls back to one full pass"
    );
    assert_ne!(with_source["imp"], initial["imp"], "importer key changes");
    assert_ne!(with_source["t"], initial["t"], "descendant key changes");
    assert_eq!(with_source["u"], initial["u"], "unrelated node is unchanged");

    // The fallback is not the revision cache: before an edit that publishes the
    // new sources, asking again still performs the correctness pass.
    reset_thread_node_hash_pass_count();
    session
        .hashes_with_sources(&sources)
        .expect("fallback hashes");
    assert_eq!(thread_node_hash_pass_count(), 1);

    // The edit publishes the new source map into the next revision, so the
    // dirty set includes the source-changed importer and its descendant even
    // though the edited node is unrelated.
    reset_thread_node_hash_pass_count();
    let change = session
        .apply_commands(&[set_length("u", 20.0)])
        .expect("edit applies");
    assert_eq!(
        thread_node_hash_pass_count(),
        1,
        "the revision after a source change must hash exactly once"
    );
    assert_eq!(change.dirty_nodes, vec!["imp", "t", "u"]);

    let cached = session.node_hashes().expect("cached hashes");
    assert_eq!(cached["imp"], with_source["imp"]);
    assert_eq!(cached["t"], with_source["t"]);
    assert_eq!(thread_node_hash_pass_count(), 1);

    // The source map now matches the revision cache: no further pass.
    reset_thread_node_hash_pass_count();
    session
        .hashes_with_sources(&sources)
        .expect("cached source hashes");
    assert_eq!(thread_node_hash_pass_count(), 0);
}

#[test]
fn ingest_record_bumps_revision_without_another_pass() {
    let mut session = Session::new(document());
    let hashes = session.node_hashes().expect("initial hashes");
    let revision = session.revision();
    reset_thread_node_hash_pass_count();
    let record = nodecraft_protocol::IngestRecord {
        schema_version: 1,
        file_name: "input.step".into(),
        byte_length: 4,
        content_hash: "record-v1".into(),
        compression: "flate2+base64".into(),
        embedded_at: None,
        data: String::new(),
    };
    assert!(session.add_ingest_record(record));
    assert_eq!(session.revision(), revision + 1);
    assert_eq!(
        thread_node_hash_pass_count(),
        0,
        "ingested assets are not node content and must not rehash"
    );
    assert!(Arc::ptr_eq(
        &hashes,
        &session.node_hashes().expect("unchanged hashes")
    ));
}

#[test]
fn failed_structural_edit_restores_the_previous_cache() {
    let mut session = Session::new(document());
    let before = session.node_hashes().expect("initial hashes");
    let revision = session.revision();
    reset_thread_node_hash_pass_count();
    let error = session
        .apply_commands(&[Command::Connect {
            from: "t".into(),
            to: "imp".into(),
            input: "shape".into(),
            out: "shape".into(),
        }])
        .expect_err("a cycle must roll the transaction back");
    assert_eq!(error, SessionError::Cycle);
    assert_eq!(
        thread_node_hash_pass_count(),
        0,
        "a failed pass is not a completed pass"
    );
    assert_eq!(session.revision(), revision, "rollback must not bump revision");
    assert!(
        Arc::ptr_eq(&before, &session.node_hashes().expect("restored hashes")),
        "rollback must restore the previous revision cache"
    );
}
