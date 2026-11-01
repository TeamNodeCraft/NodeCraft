//! Authoritative graph session for NodeCraft.
//!
//! The session owns a revisioned graph, content hashes, cache invalidation and
//! undo/redo history. The evaluation engine consumes a graph revision; the UI is
//! only a command client.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

mod defaults;
mod runtime;
pub mod store;
pub use runtime::{GraphRuntime, RuntimeEvaluation};

/// Resolve every `linkedpart` node into the flattened evaluation graph.
pub fn expand_linked_parts(document: &GraphDocument) -> Result<GraphDocument, SessionError> {
    let mut stack = Vec::new();
    runtime::expand_linked_parts(document, &mut stack)
}

/// Resolve linked parts through an in-memory document provider before falling
/// back to the filesystem. The provider receives the canonical document path.
pub fn expand_linked_parts_with(
    document: &GraphDocument,
    linked_documents: &std::collections::BTreeMap<std::path::PathBuf, GraphDocument>,
) -> Result<GraphDocument, SessionError> {
    let mut stack = Vec::new();
    runtime::expand_linked_parts_with(document, &mut stack, linked_documents)
}
pub use defaults::canonicalize_node;

/// Stable content hash for a node or connection closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub u128);

impl ContentHash {
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub values: Value,
    #[serde(default)]
    pub suppressed: bool,
    #[serde(default)]
    pub x: f64,
    #[serde(default)]
    pub y: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphConnection {
    pub from: String,
    pub to: String,
    #[serde(rename = "in")]
    pub input: String,
    #[serde(default)]
    pub out: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphInterface {
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDocument {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub id: String,
    pub nodes: Vec<GraphNode>,
    pub connections: Vec<GraphConnection>,
    #[serde(default)]
    pub interface: GraphInterface,
}

fn default_schema_version() -> u32 { 1 }

impl GraphDocument {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self { schema_version: 1, id: id.into(), nodes: Vec::new(), connections: Vec::new(), interface: GraphInterface::default() }
    }

    fn node_map(&self) -> Result<BTreeMap<&str, &GraphNode>, SessionError> {
        let mut map = BTreeMap::new();
        for node in &self.nodes {
            if map.insert(node.id.as_str(), node).is_some() {
                return Err(SessionError::DuplicateNode(node.id.clone()));
            }
        }
        Ok(map)
    }

    /// Content hash of every node, including its upstream input hashes.
    ///
    /// Positions and selection state are intentionally excluded.
    pub fn node_hashes(&self) -> Result<BTreeMap<String, ContentHash>, SessionError> {
        let nodes = self.node_map()?;
        let mut inputs: BTreeMap<&str, Vec<&GraphConnection>> = BTreeMap::new();
        for connection in &self.connections {
            if !nodes.contains_key(connection.from.as_str()) {
                return Err(SessionError::MissingNode(connection.from.clone()));
            }
            if !nodes.contains_key(connection.to.as_str()) {
                return Err(SessionError::MissingNode(connection.to.clone()));
            }
            inputs.entry(connection.to.as_str()).or_default().push(connection);
        }
        let mut indegree: BTreeMap<&str, usize> = nodes.keys().map(|id| (*id, 0)).collect();
        let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for connection in &self.connections {
            *indegree.entry(connection.to.as_str()).or_default() += 1;
            children.entry(connection.from.as_str()).or_default().push(connection.to.as_str());
        }
        let mut ready: BTreeSet<&str> = indegree.iter().filter(|(_, degree)| **degree == 0).map(|(id, _)| *id).collect();
        let mut hashes: BTreeMap<String, ContentHash> = BTreeMap::new();
        while let Some(id) = ready.pop_first() {
            let Some(node) = nodes.get(id) else { continue; };
            let mut h = Hash128::new();
            h.write(b"nodecraft-node-v1");
            h.write(node.id.as_bytes());
            h.write(node.node_type.as_bytes());
            h.write(&[u8::from(node.suppressed)]);
            canonical_json(&node.values, &mut h);
            let mut links = inputs.get(id).cloned().unwrap_or_default();
            links.sort_by(|a, b| (a.input.as_str(), a.from.as_str(), a.out.as_str()).cmp(&(b.input.as_str(), b.from.as_str(), b.out.as_str())));
            for link in links {
                h.write(link.input.as_bytes());
                h.write(link.out.as_bytes());
                if let Some(upstream) = hashes.get(&link.from) { h.write(&upstream.0.to_le_bytes()); }
            }
            hashes.insert(id.to_string(), h.finish());
            for child in children.get(id).cloned().unwrap_or_default() {
                let entry = indegree.get_mut(child).expect("child indegree");
                *entry -= 1;
                if *entry == 0 { ready.insert(child); }
            }
        }
        if hashes.len() != nodes.len() {
            return Err(SessionError::Cycle);
        }
        Ok(hashes)
    }

    /// Nodes whose evaluation content changed, including all downstream nodes.
    #[must_use]
    pub fn dirty_nodes(&self, previous: &Self) -> Result<Vec<String>, SessionError> {
        let previous_hashes = previous.node_hashes()?;
        let current_hashes = self.node_hashes()?;
        let mut dirty: BTreeSet<String> = current_hashes
            .iter()
            .filter(|(id, hash)| previous_hashes.get(*id) != Some(*hash))
            .map(|(id, _)| id.clone())
            .collect();
        let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for connection in &self.connections {
            children.entry(connection.from.as_str()).or_default().push(connection.to.as_str());
        }
        let mut queue: Vec<String> = dirty.iter().cloned().collect();
        while let Some(id) = queue.pop() {
            for child in children.get(id.as_str()).cloned().unwrap_or_default() {
                if dirty.insert(child.to_string()) { queue.push(child.to_string()); }
            }
        }
        Ok(dirty.into_iter().collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError {
    DuplicateNode(String),
    MissingNode(String),
    Cycle,
    InvalidCommand(String),
    InvalidInput(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "duplicate node {id}"),
            Self::MissingNode(id) => write!(f, "connection references missing node {id}"),
            Self::Cycle => f.write_str("graph contains a cycle"),
            Self::InvalidCommand(message) => f.write_str(message),
            Self::InvalidInput(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for SessionError {}

/// UI-facing graph edit. Commands are the future transport boundary between the
/// editor and the Rust session; positions are not part of evaluation hashes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "camelCase")]
pub enum Command {
    SetValue { node: String, field: String, value: Value },
    RemoveValue { node: String, field: String },
    SetNodeType { node: String, #[serde(rename = "nodeType")] node_type: String },
    SetPosition { node: String, x: f64, y: f64 },
    SetSuppressed { node: String, suppressed: bool },
    Connect { from: String, to: String, #[serde(default)] input: String, #[serde(default)] out: String },
    Disconnect { from: String, to: String, #[serde(default)] input: String },
    AddNode { node: GraphNode },
    RemoveNode { id: String },
}

/// Result of an edit or history movement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    pub revision: u64,
    pub dirty_nodes: Vec<String>,
    pub changed: bool,
}

#[derive(Clone, Debug)]
struct HistoryEntry {
    forward: Vec<Command>,
    inverse_groups: Vec<Vec<Command>>,
    dirty_nodes: Vec<String>,
    coalesce_key: Option<String>,
    coalesced_at: Instant,
}

#[derive(Default)]
struct ApplyEffects {
    seeds: BTreeSet<String>,
    changed: bool,
}

#[derive(Clone, Debug)]
pub struct Session {
    present: GraphDocument,
    children: BTreeMap<String, Vec<String>>,
    undo: Vec<HistoryEntry>,
    redo: Vec<HistoryEntry>,
    revision: u64,
    undo_limit: usize,
}

impl Session {
    #[must_use]
    pub fn new(mut document: GraphDocument) -> Self {
        defaults::canonicalize_document(&mut document);
        let children = build_children(&document);
        Self { present: document, children, undo: Vec::new(), redo: Vec::new(), revision: 1, undo_limit: 100 }
    }

    #[must_use]
    pub fn document(&self) -> &GraphDocument { &self.present }
    #[must_use]
    pub fn revision(&self) -> u64 { self.revision }
    #[must_use]
    pub fn has_undo(&self) -> bool { !self.undo.is_empty() }
    #[must_use]
    pub fn has_redo(&self) -> bool { !self.redo.is_empty() }

    /// Apply a transaction atomically. History stores forward and inverse command
    /// lists, not graph snapshots.
    pub fn apply_commands(&mut self, commands: &[Command]) -> Result<Change, SessionError> {
        self.apply_commands_coalesced(commands, None, Duration::ZERO)
    }

    /// Apply a transaction, merging it into the previous transaction when the
    /// caller supplies the same coalesce key inside the coalesce window.
    pub fn apply_commands_coalesced(
        &mut self,
        commands: &[Command],
        coalesce_key: Option<&str>,
        coalesce_window: Duration,
    ) -> Result<Change, SessionError> {
        let now = Instant::now();
        let mut effects = ApplyEffects::default();
        let mut inverse_groups: Vec<Vec<Command>> = Vec::new();
        let structural = commands.iter().any(Command::is_structural);
        for command in commands {
            match apply_one_command(&mut self.present, command, &mut effects) {
                Ok(group) => { if !group.is_empty() { inverse_groups.push(group); } }
                Err(error) => {
                    rollback_groups(&mut self.present, &inverse_groups);
                    self.children = build_children(&self.present);
                    return Err(error);
                }
            }
        }
        if !effects.changed {
            return Ok(Change { revision: self.revision, dirty_nodes: Vec::new(), changed: false });
        }
        if structural {
            if let Err(error) = self.present.node_hashes() {
                rollback_groups(&mut self.present, &inverse_groups);
                self.children = build_children(&self.present);
                return Err(error);
            }
            self.children = build_children(&self.present);
        }
        let dirty_nodes = dirty_nodes_from_seeds(&self.children, &effects.seeds);
        let key = coalesce_key.map(str::to_string);
        let merge = key.as_ref().is_some_and(|key| {
            coalesce_window > Duration::ZERO
                && self.undo.last().is_some_and(|entry| entry.coalesce_key.as_ref() == Some(key) && now.duration_since(entry.coalesced_at) <= coalesce_window)
        });
        let mut cumulative_dirty = dirty_nodes.clone();
        if merge {
            let entry = self.undo.last_mut().expect("coalesced history entry");
            entry.forward.extend_from_slice(commands);
            entry.inverse_groups.extend(inverse_groups);
            entry.dirty_nodes.extend(dirty_nodes);
            entry.dirty_nodes.sort();
            entry.dirty_nodes.dedup();
            entry.coalesced_at = now;
            cumulative_dirty = entry.dirty_nodes.clone();
        } else {
            self.undo.push(HistoryEntry { forward: commands.to_vec(), inverse_groups, dirty_nodes: dirty_nodes.clone(), coalesce_key: key, coalesced_at: now });
            if self.undo.len() > self.undo_limit { self.undo.remove(0); }
        }
        self.redo.clear();
        self.revision += 1;
        Ok(Change { revision: self.revision, dirty_nodes: cumulative_dirty, changed: true })
    }

    /// Apply engine-derived model updates (resolved selection references) without
    /// creating an undo entry. The revision still advances so the next evaluation
    /// sees the new content hashes.
    pub fn apply_derived_commands(&mut self, commands: &[Command]) -> Result<Change, SessionError> {
        let mut effects = ApplyEffects::default();
        let mut applied_groups: Vec<Vec<Command>> = Vec::new();
        let structural = commands.iter().any(Command::is_structural);
        for command in commands {
            match apply_one_command(&mut self.present, command, &mut effects) {
                Ok(group) => { if !group.is_empty() { applied_groups.push(group); } }
                Err(error) => {
                    rollback_groups(&mut self.present, &applied_groups);
                    self.children = build_children(&self.present);
                    return Err(error);
                }
            }
        }
        if !effects.changed {
            return Ok(Change { revision: self.revision, dirty_nodes: Vec::new(), changed: false });
        }
        if structural {
            if let Err(error) = self.present.node_hashes() {
                rollback_groups(&mut self.present, &applied_groups);
                self.children = build_children(&self.present);
                return Err(error);
            }
            self.children = build_children(&self.present);
        }
        let dirty_nodes = dirty_nodes_from_seeds(&self.children, &effects.seeds);
        self.revision += 1;
        Ok(Change { revision: self.revision, dirty_nodes, changed: true })
    }

    pub fn undo(&mut self) -> Result<Option<Change>, SessionError> {
        let Some(entry) = self.undo.pop() else { return Ok(None); };
        let mut effects = ApplyEffects::default();
        let structural = entry.forward.iter().any(Command::is_structural);
        for group in entry.inverse_groups.iter().rev() {
            for command in group {
                apply_one_command(&mut self.present, command, &mut effects)?;
            }
        }
        if structural { self.children = build_children(&self.present); }
        let dirty_nodes = dirty_nodes_from_seeds(&self.children, &effects.seeds);
        self.redo.push(entry);
        self.revision += 1;
        Ok(Some(Change { revision: self.revision, dirty_nodes, changed: true }))
    }

    pub fn redo(&mut self) -> Result<Option<Change>, SessionError> {
        let Some(entry) = self.redo.pop() else { return Ok(None); };
        let mut effects = ApplyEffects::default();
        let structural = entry.forward.iter().any(Command::is_structural);
        for command in &entry.forward {
            apply_one_command(&mut self.present, command, &mut effects)?;
        }
        if structural { self.children = build_children(&self.present); }
        let dirty_nodes = dirty_nodes_from_seeds(&self.children, &effects.seeds);
        self.undo.push(entry);
        self.revision += 1;
        Ok(Some(Change { revision: self.revision, dirty_nodes, changed: true }))
    }

}

fn find_node_mut<'a>(document: &'a mut GraphDocument, id: &str) -> Result<&'a mut GraphNode, SessionError> {
    document.nodes.iter_mut().find(|node| node.id == id).ok_or_else(|| SessionError::MissingNode(id.to_string()))
}

impl Command {
    fn is_structural(&self) -> bool {
        matches!(self, Command::AddNode { .. } | Command::RemoveNode { .. } | Command::Connect { .. } | Command::Disconnect { .. })
    }
}

fn apply_one_command(document: &mut GraphDocument, command: &Command, effects: &mut ApplyEffects) -> Result<Vec<Command>, SessionError> {
    let mut inverse = Vec::new();
    match command {
        Command::SetValue { node, field, value } => {
            let target = find_node_mut(document, node)?;
            if !target.values.is_object() { target.values = serde_json::json!({}); }
            let previous = target.values.get(field).cloned();
            if previous.as_ref() == Some(value) { return Ok(inverse); }
            match previous {
                Some(previous) => inverse.push(Command::SetValue { node: node.clone(), field: field.clone(), value: previous }),
                None => inverse.push(Command::RemoveValue { node: node.clone(), field: field.clone() }),
            }
            target.values[field] = value.clone();
            effects.seeds.insert(node.clone());
            effects.changed = true;
        }
        Command::RemoveValue { node, field } => {
            let target = find_node_mut(document, node)?;
            if let Some(previous) = target.values.get(field).cloned() {
                inverse.push(Command::SetValue { node: node.clone(), field: field.clone(), value: previous });
                if let Some(map) = target.values.as_object_mut() { map.remove(field); }
                effects.seeds.insert(node.clone());
                effects.changed = true;
            }
        }
        Command::SetNodeType { node, node_type } => {
            let target = find_node_mut(document, node)?;
            if &target.node_type == node_type { return Ok(inverse); }
            inverse.push(Command::SetNodeType { node: node.clone(), node_type: target.node_type.clone() });
            if let Some(values) = target.values.as_object() {
                for (field, value) in values {
                    inverse.push(Command::SetValue { node: node.clone(), field: field.clone(), value: value.clone() });
                }
            }
            target.node_type = node_type.clone();
            target.values = serde_json::json!({});
            defaults::canonicalize_node(target);
            effects.seeds.insert(node.clone());
            effects.changed = true;
        }
        Command::SetPosition { node, x, y } => {
            let target = find_node_mut(document, node)?;
            if target.x == *x && target.y == *y { return Ok(inverse); }
            inverse.push(Command::SetPosition { node: node.clone(), x: target.x, y: target.y });
            target.x = *x; target.y = *y;
            effects.changed = true;
        }
        Command::SetSuppressed { node, suppressed } => {
            let target = find_node_mut(document, node)?;
            if target.suppressed == *suppressed { return Ok(inverse); }
            inverse.push(Command::SetSuppressed { node: node.clone(), suppressed: target.suppressed });
            target.suppressed = *suppressed;
            effects.seeds.insert(node.clone());
            effects.changed = true;
        }
        Command::Connect { from, to, input, out } => {
            if !document.nodes.iter().any(|node| &node.id == from) { return Err(SessionError::MissingNode(from.clone())); }
            if !document.nodes.iter().any(|node| &node.id == to) { return Err(SessionError::MissingNode(to.clone())); }
            let input = if input.is_empty() { "shape".to_string() } else { input.clone() };
            let out = if out.is_empty() { "shape".to_string() } else { out.clone() };
            // A Combine is a collection: its single Shape port accepts every connected
            // Part. Every other input keeps the usual one-cable-per-port behaviour.
            let collects_many = document
                .nodes
                .iter()
                .find(|node| &node.id == to)
                .is_some_and(|node| {
                    node.node_type == "combine"
                        || (node.node_type == "surfaceloft" && input == "sections")
                });
            if collects_many {
                if document.connections.iter().any(|link| &link.from == from && &link.to == to && link.input == input && link.out == out) {
                    return Ok(inverse);
                }
                inverse.push(Command::Disconnect { from: from.clone(), to: to.clone(), input: input.clone() });
                document.connections.push(GraphConnection { from: from.clone(), to: to.clone(), input, out });
            } else if let Some(existing) = document.connections.iter_mut().find(|link| &link.to == to && link.input == input) {
                if &existing.from == from && existing.out == out { return Ok(inverse); }
                inverse.push(Command::Connect { from: existing.from.clone(), to: to.clone(), input: input.clone(), out: existing.out.clone() });
                existing.from = from.clone(); existing.out = out;
            } else {
                inverse.push(Command::Disconnect { from: from.clone(), to: to.clone(), input: input.clone() });
                document.connections.push(GraphConnection { from: from.clone(), to: to.clone(), input, out });
            }
            effects.seeds.insert(to.clone());
            effects.changed = true;
        }
        Command::Disconnect { from, to, input } => {
            let input = if input.is_empty() { "shape".to_string() } else { input.clone() };
            let removed: Vec<GraphConnection> = document.connections.iter().filter(|link| &link.from == from && &link.to == to && link.input == input).cloned().collect();
            if removed.is_empty() { return Ok(inverse); }
            document.connections.retain(|link| !(&link.from == from && &link.to == to && link.input == input));
            for link in removed { inverse.push(Command::Connect { from: link.from, to: link.to, input: link.input, out: link.out }); }
            effects.seeds.insert(to.clone());
            effects.changed = true;
        }
        Command::AddNode { node } => {
            let mut node = node.clone();
            defaults::canonicalize_node(&mut node);
            if document.nodes.iter().any(|item| item.id == node.id) { return Err(SessionError::DuplicateNode(node.id.clone())); }
            let node_id = node.id.clone();
            inverse.push(Command::RemoveNode { id: node_id.clone() });
            document.nodes.push(node);
            effects.seeds.insert(node_id);
            effects.changed = true;
        }
        Command::RemoveNode { id } => {
            let Some(index) = document.nodes.iter().position(|node| &node.id == id) else { return Err(SessionError::MissingNode(id.clone())); };
            let incident: Vec<GraphConnection> = document.connections.iter().filter(|link| &link.from == id || &link.to == id).cloned().collect();
            for link in &incident {
                if &link.to != id { effects.seeds.insert(link.to.clone()); }
            }
            let node = document.nodes.remove(index);
            document.connections.retain(|link| &link.from != id && &link.to != id);
            inverse.push(Command::AddNode { node });
            for link in incident { inverse.push(Command::Connect { from: link.from, to: link.to, input: link.input, out: link.out }); }
            effects.changed = true;
        }
    }
    Ok(inverse)
}

fn rollback_groups(document: &mut GraphDocument, groups: &[Vec<Command>]) {
    for group in groups.iter().rev() {
        for command in group {
            let mut effects = ApplyEffects::default();
            let _ = apply_one_command(document, command, &mut effects);
        }
    }
}

fn build_children(document: &GraphDocument) -> BTreeMap<String, Vec<String>> {
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for link in &document.connections {
        children.entry(link.from.clone()).or_default().push(link.to.clone());
    }
    children
}

fn dirty_nodes_from_seeds(children: &BTreeMap<String, Vec<String>>, seeds: &BTreeSet<String>) -> Vec<String> {
    if seeds.is_empty() { return Vec::new(); }
    let mut dirty: BTreeSet<String> = seeds.clone();
    let mut queue: Vec<&str> = seeds.iter().map(String::as_str).collect();
    while let Some(current) = queue.pop() {
        if let Some(consumers) = children.get(current) {
            for consumer in consumers {
                if dirty.insert(consumer.clone()) { queue.push(consumer); }
            }
        }
    }
    dirty.into_iter().collect()
}


struct Hash128 { a: u64, b: u64 }
impl Hash128 {
    fn new() -> Self { Self { a: 0xcbf29ce484222325, b: 0x84222325cbf29ce4 } }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.a ^= u64::from(*byte); self.a = self.a.wrapping_mul(0x100000001b3);
            self.b ^= u64::from(*byte); self.b = self.b.wrapping_mul(0x100000001b3).rotate_left(7);
        }
    }
    fn finish(self) -> ContentHash { ContentHash((u128::from(self.a) << 64) | u128::from(self.b)) }
}

fn canonical_json(value: &Value, out: &mut Hash128) {
    match value {
        Value::Null => out.write(b"n"),
        Value::Bool(v) => out.write(if *v { b"t" } else { b"f" }),
        Value::Number(v) => { out.write(b"#"); out.write(v.to_string().as_bytes()); }
        Value::String(v) => { out.write(b"s"); out.write(v.as_bytes()); out.write(b"\0"); }
        Value::Array(values) => { out.write(b"["); for value in values { canonical_json(value, out); } out.write(b"]"); }
        Value::Object(map) => {
            out.write(b"{");
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            for key in keys { out.write(key.as_bytes()); out.write(b"\0"); canonical_json(&map[key], out); }
            out.write(b"}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, node_type: &str, values: Value, x: f64) -> GraphNode {
        GraphNode { id: id.into(), node_type: node_type.into(), values, suppressed: false, x, y: 0.0 }
    }

    fn document() -> GraphDocument {
        GraphDocument {
            schema_version: 1,
            id: "root".into(),
            nodes: vec![
                node("a", "primitive", serde_json::json!({"primitiveType": "box", "length": 10}), 0.0),
                node("b", "translate", serde_json::json!({"x": 5}), 200.0),
                node("c", "boolean", serde_json::json!({"mode": "union"}), 400.0),
            ],
            connections: vec![
                GraphConnection { from: "a".into(), to: "b".into(), input: "shape".into(), out: "shape".into() },
                GraphConnection { from: "b".into(), to: "c".into(), input: "a".into(), out: "shape".into() },
            ],
            interface: GraphInterface::default(),
        }
    }

    #[test]
    fn layout_only_edits_do_not_dirty_nodes() {
        let mut session = Session::new(document());
        let change = session.apply_commands(&[Command::SetPosition { node: "b".into(), x: 900.0, y: 0.0 }]).expect("apply");
        assert!(change.changed);
        assert!(change.dirty_nodes.is_empty());
    }

    #[test]
    fn value_edits_dirty_node_and_downstream_only() {
        let mut session = Session::new(document());
        let change = session.apply_commands(&[Command::SetValue { node: "b".into(), field: "x".into(), value: serde_json::json!(12) }]).expect("apply");
        assert_eq!(change.dirty_nodes, vec!["b", "c"]);
    }

    #[test]
    fn upstream_edit_does_not_dirty_unrelated_branch() {
        let mut graph = document();
        graph.nodes.push(node("d", "primitive", serde_json::json!({"primitiveType": "box"}), 0.0));
        let mut session = Session::new(graph);
        let change = session.apply_commands(&[Command::SetValue { node: "a".into(), field: "length".into(), value: serde_json::json!(20) }]).expect("apply");
        assert_eq!(change.dirty_nodes, vec!["a", "b", "c"]);
        assert!(!change.dirty_nodes.contains(&"d".to_string()));
    }

    #[test]
    fn connection_changes_dirty_the_reconnected_consumer() {
        let mut session = Session::new(document());
        let change = session.apply_commands(&[Command::Connect { from: "a".into(), to: "c".into(), input: "a".into(), out: "shape".into() }]).expect("apply");
        assert_eq!(change.dirty_nodes, vec!["c"]);
    }

    #[test]
    fn combine_accepts_multiple_shape_connections() {
        let mut graph = document();
        graph.nodes.push(node("part", "combine", serde_json::json!({}), 0.0));
        let mut session = Session::new(graph);
        session.apply_commands(&[Command::Connect { from: "a".into(), to: "part".into(), input: "shape".into(), out: "shape".into() }]).expect("first part");
        session.apply_commands(&[Command::Connect { from: "b".into(), to: "part".into(), input: "shape".into(), out: "shape".into() }]).expect("second part");
        let links: Vec<_> = session.document().connections.iter().filter(|link| link.to == "part").collect();
        assert_eq!(links.len(), 2, "{links:?}");
    }

    #[test]
    fn position_changes_do_not_change_node_hashes() {
        let graph = document();
        let before = graph.node_hashes().expect("hashes");
        let mut moved = graph.clone();
        moved.nodes[0].x += 1000.0;
        moved.nodes[2].y -= 250.0;
        assert_eq!(before, moved.node_hashes().expect("hashes"));
    }

    #[test]
    fn value_key_order_is_canonical() {
        let mut left = document();
        left.nodes[0].values = serde_json::json!({"primitiveType": "box", "length": 10, "width": 5});
        let mut right = document();
        right.nodes[0].values = serde_json::json!({"width": 5, "length": 10, "primitiveType": "box"});
        assert_eq!(left.node_hashes().expect("hashes"), right.node_hashes().expect("hashes"));
    }

    #[test]
    fn undo_restores_hashes_and_reports_changed_subgraph() {
        let mut session = Session::new(document());
        session.apply_commands(&[Command::SetValue { node: "b".into(), field: "x".into(), value: serde_json::json!(12) }]).expect("edit");
        let undo = session.undo().expect("undo").expect("change");
        assert_eq!(undo.dirty_nodes, vec!["b", "c"]);
        let redo = session.redo().expect("redo").expect("change");
        assert_eq!(redo.dirty_nodes, vec!["b", "c"]);
    }

    #[test]
    fn cycles_are_rejected() {
        let mut graph = document();
        graph.connections.push(GraphConnection { from: "c".into(), to: "a".into(), input: "shape".into(), out: "shape".into() });
        assert_eq!(graph.node_hashes(), Err(SessionError::Cycle));
    }
}

#[cfg(test)]
mod transaction_tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str) -> GraphNode {
        GraphNode { id: id.into(), node_type: "primitive".into(), values: json!({"primitiveType":"box"}), suppressed: false, x: 0.0, y: 0.0 }
    }

    fn document() -> GraphDocument {
        GraphDocument { schema_version: 1, id: "root".into(), nodes: vec![node("a"), node("b")], connections: vec![], interface: GraphInterface::default() }
    }

    #[test]
    fn failed_transaction_rolls_back_without_history() {
        let mut session = Session::new(document());
        let before = session.document().clone();
        let result = session.apply_commands(&[
            Command::SetValue { node: "a".into(), field: "length".into(), value: json!(20) },
            Command::Connect { from: "missing".into(), to: "c".into(), input: "b".into(), out: "shape".into() },
        ]);
        assert!(result.is_err());
        assert_eq!(session.document(), &before);
        assert!(!session.has_undo());
        assert!(!session.has_redo());
    }

    #[test]
    fn removing_a_node_dirties_its_consumers() {
        let mut graph = document();
        graph.connections.push(GraphConnection { from: "a".into(), to: "b".into(), input: "shape".into(), out: "shape".into() });
        let mut session = Session::new(graph);
        let change = session.apply_commands(&[Command::RemoveNode { id: "a".into() }]).expect("remove");
        assert_eq!(change.dirty_nodes, vec!["b"]);
        let undo = session.undo().expect("undo").expect("entry");
        assert_eq!(undo.dirty_nodes, vec!["a", "b"]);
    }

    #[test]
    fn surfaceloft_sections_accept_multiple_connections() {
        let mut graph = GraphDocument::new("root");
        for (id, node_type, values) in [
            ("a", "sketch", json!({})),
            ("b", "sketch", json!({})),
            ("c", "surfaceloft", json!({"sections": []})),
        ] {
            graph.nodes.push(GraphNode {
                id: id.into(),
                node_type: node_type.into(),
                values,
                suppressed: false,
                x: 0.0,
                y: 0.0,
            });
        }
        let mut session = Session::new(graph);
        session
            .apply_commands(&[Command::Connect {
                from: "a".into(),
                to: "c".into(),
                input: "sections".into(),
                out: "sketch".into(),
            }])
            .expect("first section");
        session
            .apply_commands(&[Command::Connect {
                from: "b".into(),
                to: "c".into(),
                input: "sections".into(),
                out: "sketch".into(),
            }])
            .expect("second section");
        assert_eq!(session.document().connections.len(), 2);
    }

    #[test]
    fn inverse_commands_restore_removed_node_and_connections() {
        let mut session = Session::new(document());
        session.apply_commands(&[
            Command::Connect { from: "a".into(), to: "b".into(), input: "shape".into(), out: "shape".into() },
        ]).expect("connect");
        session.apply_commands(&[Command::RemoveNode { id: "a".into() }]).expect("remove");
        assert_eq!(session.document().nodes.len(), 1);
        session.undo().expect("undo").expect("entry");
        assert_eq!(session.document().nodes.len(), 2);
        assert_eq!(session.document().connections.len(), 1);
        assert_eq!(session.document().connections[0].from, "a");
    }

    #[test]
    fn inverse_commands_restore_values_and_positions() {
        let mut session = Session::new(document());
        session.apply_commands(&[
            Command::SetValue { node: "a".into(), field: "length".into(), value: json!(20) },
            Command::SetPosition { node: "a".into(), x: 50.0, y: 10.0 },
        ]).expect("edit");
        session.undo().expect("undo").expect("entry");
        assert!(session.document().nodes[0].values.get("length").is_none());
        assert_eq!(session.document().nodes[0].x, 0.0);
    }

    #[test]
    fn coalesced_typing_stream_undoes_as_one_transaction() {
        let mut session = Session::new(document());
        let window = Duration::from_secs(5);
        session.apply_commands_coalesced(&[Command::SetValue { node: "a".into(), field: "length".into(), value: json!(10) }], Some("value:a:length"), window).expect("first");
        session.apply_commands_coalesced(&[Command::SetValue { node: "a".into(), field: "length".into(), value: json!(20) }], Some("value:a:length"), window).expect("second");
        session.apply_commands_coalesced(&[Command::SetValue { node: "a".into(), field: "length".into(), value: json!(30) }], Some("value:a:length"), window).expect("third");
        session.undo().expect("undo").expect("entry");
        assert!(session.document().nodes[0].values.get("length").is_none(), "one undo should restore the pre-typing value");
        assert!(session.undo().expect("undo").is_none());
    }

    #[test]
    fn redo_replays_the_forward_transaction() {
        let mut session = Session::new(document());
        session.apply_commands(&[Command::SetValue { node: "a".into(), field: "length".into(), value: json!(20) }]).expect("edit");
        session.undo().expect("undo");
        session.redo().expect("redo").expect("entry");
        assert_eq!(session.document().nodes[0].values["length"], 20);
    }
    #[test]
    fn derived_updates_do_not_add_undo_entries() {
        let mut session = Session::new(document());
        session.apply_commands(&[Command::SetValue { node: "a".into(), field: "length".into(), value: json!(20) }]).expect("user");
        assert!(session.has_undo());
        let change = session.apply_derived_commands(&[Command::SetValue { node: "a".into(), field: "faces".into(), value: json!([{"id":"face:0"}]) }]).expect("derived");
        assert!(change.changed);
        assert_eq!(change.dirty_nodes, vec!["a"]);
        assert!(session.has_undo());
        session.undo().expect("undo").expect("entry");
        assert!(session.document().nodes[0].values.get("length").is_none());
        assert!(session.document().nodes[0].values.get("faces").is_some(), "derived field survives user undo");
    }

    #[test]
    fn session_canonicalizes_structural_defaults_once() {
        let mut graph = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            nodes: vec![
                GraphNode { id: "a".into(), node_type: "primitive".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "b".into(), node_type: "translate".into(), values: json!({"x": 5.0}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "c".into(), node_type: "chamfer".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "d".into(), node_type: "exporter".into(), values: json!({"fileName": "part.step"}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { id: "e".into(), node_type: "spline".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
            ],
            connections: Vec::new(),
            interface: GraphInterface::default(),
        };
        graph = Session::new(graph).document().clone();
        assert_eq!(graph.nodes[0].values["primitiveType"], "box");
        assert_eq!(graph.nodes[1].values["y"], 0.0);
        assert_eq!(graph.nodes[1].values["space"], "world");
        assert_eq!(graph.nodes[2].values["mode"], "chamfer");
        assert_eq!(graph.nodes[3].values["fileName"], "part");
        assert_eq!(graph.nodes[4].values["degree"], 3);
        assert_eq!(graph.nodes[4].values["control"], false);
        assert_eq!(graph.nodes[4].values["points"].as_array().unwrap().len(), 4);
    }

}
