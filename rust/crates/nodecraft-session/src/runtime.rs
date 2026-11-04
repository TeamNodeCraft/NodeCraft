//! Stateful Rust evaluation runtime for a graph session.
//!
//! The runtime owns the engine cache and the last evaluated node hashes. A
//! revision is evaluated by hashing nodes, marking only changed nodes dirty, and
//! passing the revision to the engine. Unchanged nodes are served from the
//! persistent engine cache.

use crate::{Command, ContentHash, GraphDocument, GraphNode, Session, SessionError};
use nodecraft_engine::{evaluate, EngineCache, EngineOutput, ProgressEvent};
use nodecraft_protocol::{CacheState, Connection, Node, Request};
use std::collections::BTreeMap;

pub struct RuntimeEvaluation {
    pub output: EngineOutput,
    pub dirty_nodes: Vec<String>,
}

pub struct GraphRuntime {
    session: Session,
    cache: EngineCache,
    hashes: BTreeMap<String, ContentHash>,
    revision: u64,
    request_id: u64,
    view_targets: Vec<String>,
    selected: Option<String>,
    export_targets: Vec<String>,
}

impl GraphRuntime {
    #[must_use]
    pub fn new(document: GraphDocument) -> Self {
        let hashes = document.node_hashes().unwrap_or_default();
        Self { session: Session::new(document), cache: EngineCache::new(), hashes, revision: 1, request_id: 1, view_targets: Vec::new(), selected: None, export_targets: Vec::new() }
    }

    /// Viewer targets are runtime-only view state: they are not part of the graph,
    /// its content hashes, dirty set, or undo history.
    pub fn set_view_targets(&mut self, targets: Vec<String>) {
        self.view_targets = targets;
    }

    #[must_use]
    pub fn view_targets(&self) -> &[String] { &self.view_targets }

    /// Selected node is session runtime state used by view/proxy intent; it is not
    /// graph content and does not affect hashes or history.
    pub fn set_selected(&mut self, selected: Option<String>) { self.selected = selected; }

    #[must_use]
    pub fn selected(&self) -> Option<&str> { self.selected.as_deref() }

    pub fn set_export_targets(&mut self, targets: Vec<String>) { self.export_targets = targets; }

    #[must_use]
    pub fn export_targets(&self) -> &[String] { &self.export_targets }

    /// Evaluation targets derived from view intent plus export roots. Missing and
    /// suppressed nodes are dropped; the engine evaluates each target's upstream
    /// closure and its cache skips unchanged ancestors.
    #[must_use]
    pub fn evaluation_targets(&self) -> Vec<String> {
        let mut targets = Vec::new();
        for id in self.view_targets.iter().chain(self.export_targets.iter()) {
            if targets.contains(id) { continue; }
            let Some(node) = self.session.document().nodes.iter().find(|node| &node.id == id) else { continue; };
            if node.suppressed { continue; }
            targets.push(id.clone());
        }
        targets
    }

    #[must_use]
    pub fn document(&self) -> &GraphDocument { self.session.document() }
    #[must_use]
    pub fn revision(&self) -> u64 { self.revision }
    #[must_use]
    pub fn session(&self) -> &Session { &self.session }

    pub fn undo(&mut self) -> Result<Option<crate::Change>, SessionError> {
        self.session.undo()
    }

    pub fn redo(&mut self) -> Result<Option<crate::Change>, SessionError> {
        self.session.redo()
    }

    /// Apply UI commands through the session transaction boundary.
    pub fn apply_commands(&mut self, commands: &[Command]) -> Result<crate::Change, SessionError> {
        self.session.apply_commands(commands)
    }
    pub fn apply_commands_coalesced(&mut self, commands: &[Command], coalesce_key: Option<&str>, coalesce_window: std::time::Duration) -> Result<crate::Change, SessionError> {
        self.session.apply_commands_coalesced(commands, coalesce_key, coalesce_window)
    }
    pub fn apply_derived_commands(&mut self, commands: &[Command]) -> Result<crate::Change, SessionError> {
        self.session.apply_derived_commands(commands)
    }

    /// Evaluate the current revision. Unchanged, successfully cached nodes are not
    /// recomputed; changed nodes and their downstream are marked dirty.
    pub fn evaluate(
        &mut self,
        targets: &[String],
        progress: Option<&mut dyn FnMut(ProgressEvent)>,
    ) -> Result<RuntimeEvaluation, crate::SessionError> {
        self.evaluate_with_options(targets, targets, false, progress, None)
    }

    pub fn evaluate_with_options(
        &mut self,
        targets: &[String],
        mesh_targets: &[String],
        force: bool,
        progress: Option<&mut dyn FnMut(ProgressEvent)>,
        mesh_tolerance: Option<f64>,
    ) -> Result<RuntimeEvaluation, crate::SessionError> {
        let document = self.session.document().clone();
        let current_hashes = document.node_hashes()?;
        let dirty: Vec<String> = current_hashes
            .iter()
            .filter(|(id, hash)| force || self.hashes.get(*id) != Some(*hash))
            .map(|(id, _)| id.clone())
            .collect();
        let mut stack = Vec::new();
        let grouped = crate::expand_groups(&document)?;
        let expanded = expand_linked_parts(&grouped, &mut stack)?;
        let expanded_hashes = expanded.node_hashes()?;
        let nodes: Vec<Node> = expanded.nodes.iter().map(|node| {
            let original = document.nodes.iter().find(|original| original.id == node.id);
            let hash = original.and_then(|original| current_hashes.get(&original.id)).or_else(|| expanded_hashes.get(&node.id)).copied().unwrap_or(ContentHash(0));
            let linked = original.is_some_and(|original| original.node_type == "linkedpart");
            let dirty_node = linked || (original.is_some() && (force || current_hashes.get(&node.id) != self.hashes.get(&node.id)));
            let state = if original.is_none() { "cached" } else if dirty_node { "dirty" } else { "cached" };
            Node {
                id: node.id.clone(),
                node_type: node.node_type.clone(),
                values: node.values.clone(),
                suppressed: node.suppressed,
                cache: Some(CacheState { key: Some(hash.to_hex()), state: Some(state.into()) }),
            }
        }).collect();
        let connections: Vec<Connection> = expanded.connections.iter().map(|link| Connection {
            from: link.from.clone(), to: link.to.clone(), input: link.input.clone(), out: link.out.clone(),
        }).collect();
        let request = Request {
            request_id: self.request_id,
            tab_id: Some(document.id.clone()),
            graph_revision: Some(graph_revision(&current_hashes)),
            targets: targets.to_vec(),
            mesh_targets: Some(mesh_targets.to_vec()),
            nodes,
            connections,
            binary_mesh: true,
            project_directory: None,
            progress: progress.is_some(),
            run_exporters: false,
            mesh_tolerance,
            comparison_targets: None,
        };
        self.request_id += 1;
        let output = evaluate(request, std::mem::take(&mut self.cache), progress);
        self.cache = output.cache.clone();
        self.revision += 1;
        self.hashes = current_hashes;
        Ok(RuntimeEvaluation { output, dirty_nodes: dirty })
    }
}

/// Expand referenced `.nodecraft` nodes into the current evaluation graph.
///
/// V1 references are read-only: each linked document is loaded, recursively
/// expanded, prefixed with `<link>::`, and its terminal shape nodes are connected
/// to a synthetic `combine` node at the link node id.
pub(crate) fn expand_linked_parts(document: &GraphDocument, stack: &mut Vec<String>) -> Result<GraphDocument, SessionError> {
    expand_linked_parts_with(document, stack, &std::collections::BTreeMap::new())
}

pub(crate) fn expand_linked_parts_with(
    document: &GraphDocument,
    stack: &mut Vec<String>,
    linked_documents: &std::collections::BTreeMap<std::path::PathBuf, GraphDocument>,
) -> Result<GraphDocument, SessionError> {
    let mut expanded = document.clone();
    let link_nodes: Vec<GraphNode> = document.nodes.iter().filter(|node| node.node_type == "linkedpart").cloned().collect();
    if link_nodes.is_empty() { return Ok(expanded); }
    for link in link_nodes {
        let path = link.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("").to_string();
        if path.is_empty() { return Err(SessionError::InvalidInput(format!("Linked part {} has no .nodecraft path.", link.id))); }
        let canonical = std::fs::canonicalize(&path).map(|path| path.to_string_lossy().into_owned()).unwrap_or_else(|_| path.clone());
        if stack.contains(&canonical) { return Err(SessionError::InvalidInput(format!("Linked part cycle through {path}."))); }
        stack.push(canonical.clone());
        let linked = linked_documents
            .get(std::path::Path::new(&canonical))
            .cloned()
            .map(Ok)
            .unwrap_or_else(|| crate::store::load(std::path::Path::new(&path)))
            .map_err(SessionError::InvalidInput)?;
        let linked = expand_linked_parts_with(&linked, stack, linked_documents)?;
        stack.pop();
        let sources: std::collections::BTreeSet<&str> = linked.connections.iter().map(|link| link.from.as_str()).collect();
        let terminals: Vec<String> = linked.nodes.iter().filter(|node| node.node_type != "exporter" && !sources.contains(node.id.as_str())).map(|node| node.id.clone()).collect();
        for node in &linked.nodes {
            let mut node = node.clone();
            node.id = format!("{}::{}", link.id, node.id);
            expanded.nodes.push(node);
        }
        for connection in &linked.connections {
            expanded.connections.push(crate::GraphConnection {
                from: format!("{}::{}", link.id, connection.from),
                to: format!("{}::{}", link.id, connection.to),
                input: connection.input.clone(),
                out: connection.out.clone(),
            });
        }
        if let Some(node) = expanded.nodes.iter_mut().find(|node| node.id == link.id) {
            node.node_type = "combine".into();
            node.values = serde_json::json!({});
        }
        expanded.connections.retain(|connection| connection.to != link.id);
        for terminal in terminals {
            expanded.connections.push(crate::GraphConnection { from: format!("{}::{}", link.id, terminal), to: link.id.clone(), input: "shape".into(), out: "shape".into() });
        }
    }
    Ok(expanded)
}

fn graph_revision(hashes: &BTreeMap<String, ContentHash>) -> String {
    let mut h = crate::ContentHash(0);
    let mut combined: u128 = 0;
    for hash in hashes.values() { combined = combined.wrapping_add(hash.0).wrapping_mul(0x9e3779b97f4a7c15); }
    h.0 = combined;
    h.to_hex()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphConnection, GraphNode, GraphInterface};
    use serde_json::json;

    fn doc() -> GraphDocument {
        GraphDocument {
            schema_version: 1,
            id: "root".into(),
            nodes: vec![
                GraphNode { id: "a".into(), node_type: "primitive".into(), values: json!({"primitiveType":"box","length":10,"width":10,"height":10}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "b".into(), node_type: "translate".into(), values: json!({"x":5}), suppressed: false, x: 100.0, y: 0.0 },
            ],
            connections: vec![GraphConnection { from: "a".into(), to: "b".into(), input: "shape".into(), out: "shape".into() }],
            interface: GraphInterface::default(),
        }
    }

    #[test]
    fn second_evaluation_is_served_from_cache() {
        let mut runtime = GraphRuntime::new(doc());
        let first = runtime.evaluate(&["b".into()], None).expect("first");
        assert!(first.output.response.cache_stats.as_ref().expect("stats").misses >= 1);
        let second = runtime.evaluate(&["b".into()], None).expect("second");
        let stats = second.output.response.cache_stats.as_ref().expect("stats");
        assert_eq!(stats.misses, 0, "unchanged revision must not recompute");
        assert!(stats.hits >= 2);
    }

    #[test]
    fn layout_edit_reuses_the_whole_cache() {
        let mut runtime = GraphRuntime::new(doc());
        runtime.evaluate(&["b".into()], None).expect("first");
        runtime.apply_commands(&[Command::SetPosition { node: "a".into(), x: 500.0, y: 0.0 }]).expect("layout");
        let result = runtime.evaluate(&["b".into()], None).expect("layout");
        assert!(result.dirty_nodes.is_empty());
        assert_eq!(result.output.response.cache_stats.as_ref().expect("stats").misses, 0);
    }

    #[test]
    fn value_edit_only_recomputes_changed_node_and_downstream() {
        let mut runtime = GraphRuntime::new(doc());
        runtime.evaluate(&["b".into()], None).expect("first");
        runtime.apply_commands(&[Command::SetValue { node: "b".into(), field: "x".into(), value: json!(9) }]).expect("value");
        let result = runtime.evaluate(&["b".into()], None).expect("value");
        assert_eq!(result.dirty_nodes, vec!["b"]);
        let stats = result.output.response.cache_stats.as_ref().expect("stats");
        assert!(stats.hits >= 1, "upstream a should be a hit");
        assert_eq!(stats.misses, 1, "b should be recomputed");
    }
}

#[cfg(test)]
mod command_tests {
    use super::*;
    use crate::{Command, GraphConnection, GraphNode, GraphInterface};
    use serde_json::json;

    fn document() -> GraphDocument {
        GraphDocument { schema_version: 1, id: "root".into(), interface: GraphInterface::default(),
            nodes: vec![
                GraphNode { id: "a".into(), node_type: "primitive".into(), values: json!({"primitiveType":"box"}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "b".into(), node_type: "translate".into(), values: json!({"x":0}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "c".into(), node_type: "primitive".into(), values: json!({"primitiveType":"box"}), suppressed: false, x: 0.0, y: 0.0 },
            ],
            connections: vec![GraphConnection { from: "a".into(), to: "b".into(), input: "shape".into(), out: "shape".into() }],
        }
    }

    #[test]
    fn commands_edit_graph_and_report_dirty_sets() {
        let mut runtime = GraphRuntime::new(document());
        let change = runtime.apply_commands(&[Command::SetValue { node: "b".into(), field: "x".into(), value: json!(5) }]).expect("set");
        assert_eq!(change.dirty_nodes, vec!["b"]);
        let change = runtime.apply_commands(&[Command::SetPosition { node: "a".into(), x: 300.0, y: 20.0 }]).expect("move");
        assert!(change.dirty_nodes.is_empty());
    }

    #[test]
    fn connect_command_replaces_the_input_edge() {
        let mut runtime = GraphRuntime::new(document());
        let change = runtime.apply_commands(&[Command::Connect { from: "c".into(), to: "b".into(), input: "shape".into(), out: "shape".into() }]).expect("connect");
        assert!(change.dirty_nodes.contains(&"b".to_string()));
        assert_eq!(runtime.document().connections[0].from, "c");
    }
    #[test]
    fn view_targets_are_runtime_state_and_drive_evaluation_targets() {
        let mut runtime = GraphRuntime::new(document());
        runtime.apply_commands(&[crate::Command::SetSuppressed { node: "b".into(), suppressed: true }]).expect("suppress");
        runtime.set_view_targets(vec!["b".into(), "missing".into(), "a".into()]);
        assert_eq!(runtime.view_targets(), &["b".to_string(), "missing".to_string(), "a".to_string()]);
        runtime.set_export_targets(vec!["a".into()]);
        assert_eq!(runtime.evaluation_targets(), vec!["a".to_string()]);
        runtime.set_selected(Some("a".into()));
        assert_eq!(runtime.selected(), Some("a"));
    }

    #[test]
    fn linked_part_cycles_are_rejected() {
        use crate::{GraphInterface, GraphNode};
        let dir = std::env::temp_dir().join(format!(
            "sn-linked-cycle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let a_path = dir.join("a.nodecraft");
        let b_path = dir.join("b.nodecraft");
        let linked = |id: &str, path: &std::path::Path| GraphDocument {
            schema_version: 1,
            id: id.into(),
            nodes: vec![GraphNode {
                id: "link".into(),
                node_type: "linkedpart".into(),
                values: json!({"path": path.to_string_lossy()}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::new(),
            interface: GraphInterface::default(),
        };
        crate::store::save_atomic(&a_path, &linked("a", &b_path)).unwrap();
        crate::store::save_atomic(&b_path, &linked("b", &a_path)).unwrap();
        let error = crate::expand_linked_parts(&linked("a", &b_path)).expect_err("cycle");
        assert!(error.to_string().contains("cycle"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn linked_part_evaluates_referenced_document() {
        use crate::{GraphConnection, GraphInterface, GraphNode};
        let dir = std::env::temp_dir().join(format!("sn-linked-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bracket.nodecraft");
        let linked = GraphDocument {
            schema_version: 1, id: "bracket".into(),
            nodes: vec![GraphNode { id: "p".into(), node_type: "primitive".into(), values: json!({"primitiveType":"box","length":10,"width":10,"height":10}), suppressed: false, x: 0.0, y: 0.0 }],
            connections: Vec::new(), interface: GraphInterface::default(),
        };
        crate::store::save_atomic(&path, &linked).expect("save linked");
        let document = GraphDocument {
            schema_version: 1, id: "root".into(),
            nodes: vec![
                GraphNode { id: "link".into(), node_type: "linkedpart".into(), values: json!({"path": path.to_string_lossy()}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "t".into(), node_type: "translate".into(), values: json!({"x": 5.0}), suppressed: false, x: 0.0, y: 0.0 },
            ],
            connections: vec![GraphConnection { from: "link".into(), to: "t".into(), input: "shape".into(), out: "shape".into() }],
            interface: GraphInterface::default(),
        };
        let mut runtime = GraphRuntime::new(document);
        let result = runtime.evaluate(&["t".into()], None).expect("linked evaluation");
        let mesh = &result.output.response.meshes.get("t").expect("mesh").mesh;
        assert!(!mesh.faces.is_empty());
        std::fs::remove_dir_all(dir).ok();
    }

}
