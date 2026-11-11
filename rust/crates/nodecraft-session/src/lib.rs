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
pub mod flat;
pub mod groups;
pub mod node_descriptor;
pub mod placement;
mod runtime;
pub mod store;
pub use groups::{expand_groups, flatten_inline_graphs};
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
    /// Source expressions by value field. Optional; empty maps are omitted.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub expressions: BTreeMap<String, String>,
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

/// One reusable graph inside a document. Root graphs and local subgraphs use
/// the same representation; evaluation flattens them into one DAG.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDefinition {
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// `graph`, `iterator` or `fold`; execution constructs are added later.
    #[serde(default = "default_graph_kind")]
    pub kind: String,
    #[serde(default)]
    pub interface: GraphInterface,
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub connections: Vec<GraphConnection>,
}

fn default_graph_kind() -> String {
    "graph".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDocument {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub id: String,
    /// The document's root graph. Field access is forwarded here so existing
    /// graph operations continue to operate on the root definition.
    #[serde(default = "default_root_definition")]
    pub root: GraphDefinition,
    /// Additional reusable definitions stored in this document.
    #[serde(default)]
    pub definitions: Vec<GraphDefinition>,
}

fn default_root_definition() -> GraphDefinition {
    GraphDefinition {
        id: "root".to_owned(),
        name: "Root".to_owned(),
        kind: "graph".to_owned(),
        interface: GraphInterface::default(),
        nodes: Vec::new(),
        connections: Vec::new(),
    }
}

impl std::ops::Deref for GraphDocument {
    type Target = GraphDefinition;
    fn deref(&self) -> &Self::Target {
        &self.root
    }
}

impl std::ops::DerefMut for GraphDocument {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.root
    }
}

fn default_schema_version() -> u32 { 1 }

impl GraphDocument {
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            schema_version: 1,
            id: id.into(),
            root: default_root_definition(),
            definitions: Vec::new(),
        }
    }

    #[must_use]
    pub fn definition(&self, id: &str) -> Option<&GraphDefinition> {
        if self.root.id == id {
            Some(&self.root)
        } else {
            self.definitions.iter().find(|definition| definition.id == id)
        }
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

    /// Validate definition identity, interfaces and owned graph topology.
    pub fn validate_definitions(&self) -> Result<(), SessionError> {
        let mut ids = BTreeSet::new();
        ids.insert(self.root.id.clone());
        let mut names = BTreeSet::new();
        let validate_interface = |kind: &str, ports: &[String]| -> Result<(), SessionError> {
            let mut seen = BTreeSet::new();
            for port in ports {
                if port.trim().is_empty() {
                    return Err(SessionError::InvalidInput(format!(
                        "Graph definition {kind} has an empty port name."
                    )));
                }
                if !seen.insert(port.as_str()) {
                    return Err(SessionError::InvalidInput(format!(
                        "Graph definition {kind} has duplicate port {port:?}."
                    )));
                }
            }
            Ok(())
        };
        validate_interface("root inputs", &self.root.interface.inputs)?;
        validate_interface("root outputs", &self.root.interface.outputs)?;
        for definition in &self.definitions {
            if !ids.insert(definition.id.clone()) {
                return Err(SessionError::InvalidInput(format!(
                    "Duplicate graph definition id {}.",
                    definition.id
                )));
            }
            if !names.insert(definition.name.clone()) {
                return Err(SessionError::InvalidInput(format!(
                    "Duplicate graph definition name {:?}.",
                    definition.name
                )));
            }
            validate_interface("inputs", &definition.interface.inputs)?;
            validate_interface("outputs", &definition.interface.outputs)?;
            let mut node_ids = BTreeSet::new();
            for node in &definition.nodes {
                if !node_ids.insert(node.id.as_str()) {
                    return Err(SessionError::InvalidInput(format!(
                        "Duplicate node {} in graph definition {}.",
                        node.id, definition.id
                    )));
                }
            }
            for connection in &definition.connections {
                if !node_ids.contains(connection.from.as_str())
                    || !node_ids.contains(connection.to.as_str())
                {
                    return Err(SessionError::InvalidInput(format!(
                        "Graph definition {} has a connection to a missing node.",
                        definition.id
                    )));
                }
            }
        }
        Ok(())
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
            if node.node_type == "graph" {
                // A graph card caches its definition name for display. Names are
                // presentation only, so exclude them from the geometry hash.
                let mut values = node.values.clone();
                if let Some(object) = values.as_object_mut() {
                    object.remove("name");
                }
                canonical_json(&values, &mut h);
                let mut definition_stack = Vec::new();
                self.hash_graph_reference(node, &mut h, &mut definition_stack)?;
            } else {
                canonical_json(&node.values, &mut h);
            }
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

    fn hash_graph_reference(
        &self,
        node: &GraphNode,
        out: &mut Hash128,
        stack: &mut Vec<String>,
    ) -> Result<(), SessionError> {
        let Some(reference) = groups::GraphReference::from_node(node) else {
            return Ok(());
        };
        match reference {
            groups::GraphReference::Local(id) => {
                out.write(b"local-definition");
                self.hash_definition(&id, out, stack)
            }
            groups::GraphReference::External { path, definition } => {
                out.write(b"external-definition");
                out.write(path.as_bytes());
                out.write(definition.as_bytes());
                Ok(())
            }
        }
    }

    fn hash_definition(
        &self,
        id: &str,
        out: &mut Hash128,
        stack: &mut Vec<String>,
    ) -> Result<(), SessionError> {
        if stack.iter().any(|candidate| candidate == id) {
            return Err(SessionError::InvalidInput(format!(
                "Graph definition cycle through {id}."
            )));
        }
        let definition = self.definition(id).ok_or_else(|| {
            SessionError::InvalidInput(format!("Missing graph definition {id}."))
        })?;
        stack.push(id.to_owned());
        out.write(b"definition-v1");
        out.write(definition.id.as_bytes());
        out.write(definition.kind.as_bytes());
        for port in &definition.interface.inputs {
            out.write(b"in:"); out.write(port.as_bytes());
        }
        for port in &definition.interface.outputs {
            out.write(b"out:"); out.write(port.as_bytes());
        }
        let mut nodes: Vec<&GraphNode> = definition.nodes.iter().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        for node in nodes {
            out.write(node.id.as_bytes());
            out.write(node.node_type.as_bytes());
            out.write(&[u8::from(node.suppressed)]);
            canonical_json(&node.values, out);
            if node.node_type == "graph" {
                self.hash_graph_reference(node, out, stack)?;
            }
        }
        let mut connections: Vec<&GraphConnection> = definition.connections.iter().collect();
        connections.sort_by(|a, b| {
            (a.from.as_str(), a.to.as_str(), a.input.as_str(), a.out.as_str())
                .cmp(&(b.from.as_str(), b.to.as_str(), b.input.as_str(), b.out.as_str()))
        });
        for connection in connections {
            out.write(connection.from.as_bytes());
            out.write(connection.to.as_bytes());
            out.write(connection.input.as_bytes());
            out.write(connection.out.as_bytes());
        }
        stack.pop();
        Ok(())
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
    SetExpressions { node: String, values: BTreeMap<String, String> },
    SetNodeType { node: String, #[serde(rename = "nodeType")] node_type: String },
    AddDefinition { definition: GraphDefinition },
    RemoveDefinition { id: String },
    SetDefinition { definition: GraphDefinition },
    SetRootDefinition { definition: GraphDefinition },
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
    scope: String,
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
        self.apply_commands_scoped(commands, "root", coalesce_key, coalesce_window)
    }

    /// Apply a transaction tagged with its originating view. Entries only merge
    /// and undo within the same scope.
    pub fn apply_commands_scoped(
        &mut self,
        commands: &[Command],
        scope: &str,
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
                && self.undo.last().is_some_and(|entry| {
                    entry.scope == scope
                        && entry.coalesce_key.as_ref() == Some(key)
                        && now.duration_since(entry.coalesced_at) <= coalesce_window
                })
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
            self.undo.push(HistoryEntry { scope: scope.to_owned(), forward: commands.to_vec(), inverse_groups, dirty_nodes: dirty_nodes.clone(), coalesce_key: key, coalesced_at: now });
            if self.undo.len() > self.undo_limit { self.undo.remove(0); }
        }
        self.redo.clear();
        self.revision += 1;
        Ok(Change { revision: self.revision, dirty_nodes: cumulative_dirty, changed: true })
    }

    /// Apply node commands to a definition body as one parent-document edit.
    ///
    /// This is the session-side primitive for definition-backed graph views:
    /// commands use the body's node ids, while the parent history sees one
    /// atomic `SetDefinition`.
    pub fn apply_definition_commands(
        &mut self,
        definition_id: &str,
        commands: &[Command],
    ) -> Result<Change, SessionError> {
        self.apply_definition_commands_scoped(
            definition_id,
            &format!("definition:{definition_id}"),
            commands,
        )
    }

    /// Apply definition-body commands under an explicit view/history scope.
    pub fn apply_definition_commands_scoped(
        &mut self,
        definition_id: &str,
        scope: &str,
        commands: &[Command],
    ) -> Result<Change, SessionError> {
        let definition = self
            .present
            .definition(definition_id)
            .cloned()
            .ok_or_else(|| {
                SessionError::InvalidInput(format!("Missing graph definition {definition_id}."))
            })?;
        let original_definitions = self.present.definitions.clone();
        // A view may have been built without the parent definition table (for
        // example a stale projection). If a batch tries to add a definition id
        // or name that already exists, allocate a fresh identity and rewrite the
        // graph-node references in the same batch before applying it.
        let mut commands = commands.to_vec();
        let mut used_ids: BTreeSet<String> = original_definitions
            .iter()
            .map(|definition| definition.id.clone())
            .chain(std::iter::once(self.present.root.id.clone()))
            .collect();
        let mut used_names: BTreeSet<String> = original_definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect();
        let mut remap = BTreeMap::new();
        for command in &mut commands {
            let Command::AddDefinition { definition } = command else {
                continue;
            };
            if used_ids.contains(&definition.id) {
                let old_id = definition.id.clone();
                let base_id = old_id.clone();
                let mut suffix = 2;
                let mut new_id = format!("{base_id}{suffix}");
                while used_ids.contains(&new_id) {
                    suffix += 1;
                    new_id = format!("{base_id}{suffix}");
                }
                let base_name = definition.name.clone();
                let mut name_suffix = 2;
                let mut new_name = format!("{base_name} {name_suffix}");
                while used_names.contains(&new_name) {
                    name_suffix += 1;
                    new_name = format!("{base_name} {name_suffix}");
                }
                definition.id = new_id.clone();
                definition.name = new_name;
                remap.insert(old_id, new_id);
            }
            used_ids.insert(definition.id.clone());
            used_names.insert(definition.name.clone());
        }
        if !remap.is_empty() {
            for command in &mut commands {
                match command {
                    Command::AddNode { node } if node.node_type == "graph" => {
                        if let Some(Value::String(id)) = node.values.get_mut("definition")
                            && let Some(new_id) = remap.get(id)
                        {
                            *id = new_id.clone();
                        }
                    }
                    Command::SetValue { field, value, .. } if field == "definition" => {
                        if let Value::String(id) = value
                            && let Some(new_id) = remap.get(id)
                        {
                            *id = new_id.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut view = GraphDocument::new(format!("{definition_id}::view"));
        view.root = definition.clone();
        // Definition bodies can create/import nested definitions. Keep the
        // parent's definition table visible and mirror changes back below.
        view.definitions = original_definitions.clone();
        let mut body = Session::new(view);
        let change = body.apply_commands(&commands)?;
        if !change.changed {
            return Ok(change);
        }
        let mut updated = body.document().root.clone();
        updated.id = definition.id;
        updated.name = definition.name;
        updated.kind = definition.kind;
        let mut parent_commands = vec![Command::SetDefinition { definition: updated }];
        let original_by_id: BTreeMap<&str, &GraphDefinition> = original_definitions
            .iter()
            .map(|definition| (definition.id.as_str(), definition))
            .collect();
        let body_definitions = body.document().definitions.clone();
        let body_ids: BTreeSet<&str> = body_definitions
            .iter()
            .map(|definition| definition.id.as_str())
            .collect();
        for definition in &body_definitions {
            if definition.id == definition_id {
                continue;
            }
            match original_by_id.get(definition.id.as_str()) {
                Some(original) if *original != definition => {
                    parent_commands.push(Command::SetDefinition {
                        definition: definition.clone(),
                    });
                }
                None => parent_commands.push(Command::AddDefinition {
                    definition: definition.clone(),
                }),
                _ => {}
            }
        }
        for definition in &original_definitions {
            if !body_ids.contains(definition.id.as_str()) {
                parent_commands.push(Command::RemoveDefinition {
                    id: definition.id.clone(),
                });
            }
        }
        self.apply_commands_scoped(&parent_commands, scope, None, Duration::ZERO)
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

    /// True when the next undo entry belongs to the root document rather than a
    /// definition graph view. Root-tab history must not consume graph-view edits.
    #[must_use]
    pub fn has_root_undo(&self) -> bool {
        self.undo.last().is_some_and(|entry| entry.scope == "root")
    }

    #[must_use]
    pub fn has_root_redo(&self) -> bool {
        self.redo.last().is_some_and(|entry| entry.scope == "root")
    }

    #[must_use]
    pub fn has_scope_undo(&self, scope: &str) -> bool {
        self.undo.last().is_some_and(|entry| entry.scope == scope)
    }

    #[must_use]
    pub fn has_scope_redo(&self, scope: &str) -> bool {
        self.redo.last().is_some_and(|entry| entry.scope == scope)
    }

    pub fn undo_scope(&mut self, scope: &str) -> Result<Option<Change>, SessionError> {
        if !self.has_scope_undo(scope) {
            return Ok(None);
        }
        self.undo()
    }

    pub fn redo_scope(&mut self, scope: &str) -> Result<Option<Change>, SessionError> {
        if !self.has_scope_redo(scope) {
            return Ok(None);
        }
        self.redo()
    }

    /// True when the next undo entry is an edit of this definition body.
    #[must_use]
    pub fn has_definition_undo(&self, definition_id: &str) -> bool {
        let scope = format!("definition:{definition_id}");
        self.undo.last().is_some_and(|entry| entry.scope == scope)
    }

    #[must_use]
    pub fn has_definition_redo(&self, definition_id: &str) -> bool {
        let scope = format!("definition:{definition_id}");
        self.redo.last().is_some_and(|entry| entry.scope == scope)
    }

    /// Undo only when the next entry is an edit of this definition body.
    pub fn undo_definition(
        &mut self,
        definition_id: &str,
    ) -> Result<Option<Change>, SessionError> {
        if !self.has_definition_undo(definition_id) {
            return Ok(None);
        }
        self.undo()
    }

    pub fn redo_definition(
        &mut self,
        definition_id: &str,
    ) -> Result<Option<Change>, SessionError> {
        if !self.has_definition_redo(definition_id) {
            return Ok(None);
        }
        self.redo()
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
        matches!(
            self,
            Command::AddNode { .. }
                | Command::RemoveNode { .. }
                | Command::Connect { .. }
                | Command::Disconnect { .. }
                | Command::AddDefinition { .. }
                | Command::RemoveDefinition { .. }
                | Command::SetDefinition { .. }
                | Command::SetRootDefinition { .. }
        )
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
        Command::SetExpressions { node, values } => {
            let target = find_node_mut(document, node)?;
            if &target.expressions == values { return Ok(inverse); }
            inverse.push(Command::SetExpressions { node: node.clone(), values: target.expressions.clone() });
            target.expressions = values.clone();
            // Expressions are presentation/source metadata, not geometry
            // content: do not seed a geometry invalidation for this command.
            effects.changed = true;
        }
        Command::SetNodeType { node, node_type } => {
            let target = find_node_mut(document, node)?;
            if &target.node_type == node_type { return Ok(inverse); }
            inverse.push(Command::SetNodeType { node: node.clone(), node_type: target.node_type.clone() });
            if !target.expressions.is_empty() {
                inverse.push(Command::SetExpressions { node: node.clone(), values: target.expressions.clone() });
                target.expressions.clear();
            }
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
        Command::AddDefinition { definition } => {
            if document
                .definitions
                .iter()
                .any(|candidate| candidate.id == definition.id)
            {
                return Err(SessionError::InvalidInput(format!(
                    "Duplicate graph definition {}.",
                    definition.id
                )));
            }
            inverse.push(Command::RemoveDefinition {
                id: definition.id.clone(),
            });
            document.definitions.push(definition.clone());
            effects.changed = true;
        }
        Command::RemoveDefinition { id } => {
            let Some(index) = document
                .definitions
                .iter()
                .position(|candidate| &candidate.id == id)
            else {
                return Err(SessionError::InvalidInput(format!(
                    "Missing graph definition {id}."
                )));
            };
            let definition = document.definitions.remove(index);
            inverse.push(Command::AddDefinition { definition });
            effects.changed = true;
        }
        Command::SetRootDefinition { definition } => {
            let previous = document.root.clone();
            inverse.push(Command::SetRootDefinition { definition: previous });
            document.root = definition.clone();
            effects.changed = true;
        }
        Command::SetDefinition { definition } => {
            let Some(index) = document
                .definitions
                .iter()
                .position(|candidate| candidate.id == definition.id)
            else {
                return Err(SessionError::InvalidInput(format!(
                    "Missing graph definition {}.",
                    definition.id
                )));
            };
            let previous = document.definitions[index].clone();
            inverse.push(Command::SetDefinition { definition: previous });
            document.definitions[index] = definition.clone();
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
            if groups::is_graph_interface_node(&document.nodes[index]) {
                return Err(SessionError::InvalidInput(
                    "Graph Input/Output nodes cannot be deleted.".into(),
                ));
            }
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
        GraphNode { expressions: Default::default(), id: id.into(), node_type: node_type.into(), values, suppressed: false, x, y: 0.0 }
    }

    fn document() -> GraphDocument {
        GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
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
            },
            definitions: Vec::new(),
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
        GraphNode { expressions: Default::default(), id: id.into(), node_type: "primitive".into(), values: json!({"primitiveType":"box"}), suppressed: false, x: 0.0, y: 0.0 }
    }

    fn document() -> GraphDocument {
        GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![node("a"), node("b")],
                connections: vec![],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        }
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
    fn graph_document_round_trips_root_and_definitions() {
        let mut document = GraphDocument::new("definitions");
        document.nodes.push(GraphNode { expressions: Default::default(),
            id: "root_box".into(),
            node_type: "primitive".into(),
            values: serde_json::json!({"primitiveType": "box"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.definitions.push(crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(),
                id: "feature_box".into(),
                node_type: "primitive".into(),
                values: serde_json::json!({"primitiveType": "box"}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::new(),
        });
        let json = serde_json::to_string(&document).expect("serialize");
        let reloaded: GraphDocument = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(reloaded.schema_version, 1);
        assert_eq!(reloaded.root.nodes[0].id, "root_box");
        assert_eq!(reloaded.definitions.len(), 1);
        assert_eq!(reloaded.definition("feature").expect("definition").name, "Feature");
        assert_eq!(reloaded.definition("root").expect("root").nodes[0].id, "root_box");
    }

    #[test]
    fn definition_validation_rejects_duplicate_ids_and_names() {
        let mut document = GraphDocument::new("validate");
        let definition = crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: Vec::new(),
            connections: Vec::new(),
        };
        document.definitions.push(definition.clone());
        document.definitions.push(definition);
        assert!(document.validate_definitions().is_err());
        let mut document = GraphDocument::new("validate");
        let mut first = crate::GraphDefinition {
            id: "one".into(),
            name: "Same".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: Vec::new(),
            connections: Vec::new(),
        };
        let mut second = first.clone();
        second.id = "two".into();
        first.name = "Same".into();
        document.definitions.push(first);
        document.definitions.push(second);
        assert!(document.validate_definitions().is_err());
    }

    #[test]
    fn definition_commands_remap_colliding_nested_definition_ids() {
        let mut document = GraphDocument::new("nested-collision");
        document.definitions.push(crate::GraphDefinition {
            id: "graph_definition2".into(), name: "Graph 2".into(), kind: "graph".into(),
            interface: GraphInterface::default(), nodes: Vec::new(), connections: Vec::new(),
        });
        document.definitions.push(crate::GraphDefinition {
            id: "outer".into(), name: "Outer".into(), kind: "graph".into(),
            interface: GraphInterface::default(), nodes: Vec::new(), connections: Vec::new(),
        });
        let mut session = Session::new(document);
        let colliding = crate::GraphDefinition {
            id: "graph_definition2".into(), name: "Graph 2".into(), kind: "graph".into(),
            interface: GraphInterface::default(), nodes: Vec::new(), connections: Vec::new(),
        };
        session.apply_definition_commands("outer", &[
            Command::AddDefinition { definition: colliding },
            Command::AddNode { node: GraphNode { expressions: Default::default(), id: "nested".into(), node_type: "graph".into(), values: serde_json::json!({"definition":"graph_definition2"}), suppressed: false, x: 0.0, y: 0.0 } },
        ]).expect("nested collision");
        let nested = &session.document().definition("outer").unwrap().nodes[0];
        let id = nested.values["definition"].as_str().unwrap();
        assert_ne!(id, "graph_definition2");
        assert!(session.document().definition(id).is_some());
        assert!(session.document().definition("graph_definition2").is_some());
    }

    #[test]
    fn definition_commands_can_create_nested_definitions() {
        let mut document = GraphDocument::new("nested-create");
        document.definitions.push(crate::GraphDefinition {
            id: "outer".into(), name: "Outer".into(), kind: "graph".into(),
            interface: GraphInterface::default(), nodes: Vec::new(), connections: Vec::new(),
        });
        let mut session = Session::new(document);
        let inner = crate::GraphDefinition {
            id: "inner".into(), name: "Inner".into(), kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(), id: "box".into(), node_type: "solid".into(), values: serde_json::json!({"primitiveType":"box"}), suppressed: false, x: 0.0, y: 0.0 }],
            connections: Vec::new(),
        };
        session.apply_definition_commands(
            "outer",
            &[
                Command::AddDefinition { definition: inner },
                Command::AddNode { node: GraphNode { expressions: Default::default(), id: "nested".into(), node_type: "graph".into(), values: serde_json::json!({"definition":"inner"}), suppressed: false, x: 0.0, y: 0.0 } },
            ],
        ).expect("nested create");
        assert!(session.document().definition("inner").is_some());
        assert_eq!(session.document().definition("outer").unwrap().nodes[0].node_type, "graph");
        session.undo().expect("undo").expect("change");
        assert!(session.document().definition("inner").is_none());
        assert!(session.document().definition("outer").unwrap().nodes.is_empty());
    }

    #[test]
    fn definition_scoped_commands_are_atomic_and_undoable() {
        let mut document = GraphDocument::new("definition-view");
        document.definitions.push(crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(),
                id: "box".into(),
                node_type: "solid".into(),
                values: serde_json::json!({"primitiveType":"box"}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::new(),
        });
        let mut session = Session::new(document);
        session
            .apply_definition_commands(
                "feature",
                &[Command::SetValue {
                    node: "box".into(),
                    field: "length".into(),
                    value: serde_json::json!(20.0),
                }],
            )
            .expect("definition edit");
        assert_eq!(session.document().definition("feature").unwrap().nodes[0].values["length"], 20.0);
        session.undo().expect("undo").expect("change");
        assert!(session.document().definition("feature").unwrap().nodes[0].values.get("length").is_none());
    }

    #[test]
    fn definition_undo_does_not_consume_root_history() {
        let mut document = GraphDocument::new("definition-history");
        document.root.nodes.push(GraphNode { expressions: Default::default(),
            id: "root-node".into(),
            node_type: "solid".into(),
            values: serde_json::json!({"primitiveType":"box"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.definitions.push(crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(),
                id: "box".into(),
                node_type: "solid".into(),
                values: serde_json::json!({"primitiveType":"box"}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::new(),
        });
        let mut session = Session::new(document);
        session
            .apply_commands(&[Command::SetValue {
                node: "root-node".into(),
                field: "length".into(),
                value: serde_json::json!(5.0),
            }])
            .expect("root edit");
        session
            .apply_definition_commands(
                "feature",
                &[Command::SetValue {
                    node: "box".into(),
                    field: "length".into(),
                    value: serde_json::json!(9.0),
                }],
            )
            .expect("definition edit");
        assert!(!session.has_root_undo(), "definition edit blocks root undo first");
        session.undo_definition("feature").expect("undo def").expect("change");
        assert!(session.document().definition("feature").unwrap().nodes[0].values.get("length").is_none());
        assert_eq!(session.document().nodes[0].values["length"], 5.0);
        assert!(session.has_root_undo(), "root undo becomes available after the graph edit");
        assert!(session.undo_definition("feature").expect("no-op").is_none());
        session.undo().expect("root undo").expect("change");
        assert!(session.document().nodes[0].values.get("length").is_none());
    }

    #[test]
    fn add_and_remove_definition_are_undoable() {
        let mut session = Session::new(GraphDocument::new("definitions-command"));
        let definition = crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: Vec::new(),
            connections: Vec::new(),
        };
        session
            .apply_commands(&[Command::AddDefinition { definition }])
            .expect("add definition");
        assert!(session.document().definition("feature").is_some());
        session.undo().expect("undo").expect("change");
        assert!(session.document().definition("feature").is_none());
        session.redo().expect("redo").expect("change");
        assert!(session.document().definition("feature").is_some());
        session
            .apply_commands(&[Command::RemoveDefinition { id: "feature".into() }])
            .expect("remove definition");
        assert!(session.document().definition("feature").is_none());
    }

    #[test]
    fn scoped_history_keeps_tab_edits_separate() {
        let mut document = GraphDocument::new("scoped-tabs");
        document.root.nodes.push(GraphNode { expressions: Default::default(), id: "a".into(), node_type: "solid".into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 });
        document.root.nodes.push(GraphNode { expressions: Default::default(), id: "b".into(), node_type: "solid".into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 });
        let mut session = Session::new(document);
        session.apply_commands_scoped(
            &[Command::SetValue { node: "a".into(), field: "length".into(), value: serde_json::json!(1.0) }],
            "view:1",
            None,
            Duration::ZERO,
        ).unwrap();
        session.apply_commands_scoped(
            &[Command::SetValue { node: "b".into(), field: "length".into(), value: serde_json::json!(2.0) }],
            "view:2",
            None,
            Duration::ZERO,
        ).unwrap();
        assert!(session.undo_scope("view:1").unwrap().is_none(), "another tab's edit is on top");
        assert!(session.undo_scope("view:2").unwrap().is_some());
        assert!(session.undo_scope("view:1").unwrap().is_some());
        assert!(session.document().nodes[0].values.get("length").is_none());
    }

    #[test]
    fn interleaved_root_and_definition_history_is_scoped() {
        let mut document = GraphDocument::new("interleaved");
        document.root.nodes.push(GraphNode { expressions: Default::default(), id: "root".into(), node_type: "solid".into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 });
        document.definitions.push(crate::GraphDefinition {
            id: "feature".into(), name: "Feature".into(), kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(), id: "box".into(), node_type: "solid".into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 }],
            connections: Vec::new(),
        });
        let mut session = Session::new(document);
        session.apply_commands_scoped(
            &[Command::SetValue { node: "root".into(), field: "rootEdit".into(), value: serde_json::json!(1) }],
            "view:1", None, Duration::ZERO,
        ).unwrap();
        session.apply_definition_commands_scoped(
            "feature", "definition:feature:view:2",
            &[Command::SetValue { node: "box".into(), field: "defEdit".into(), value: serde_json::json!(2) }],
        ).unwrap();
        session.apply_commands_scoped(
            &[Command::SetValue { node: "root".into(), field: "rootEdit2".into(), value: serde_json::json!(3) }],
            "view:1", None, Duration::ZERO,
        ).unwrap();
        assert!(session.undo_scope("view:1").unwrap().is_some());
        assert!(session.undo_scope("view:1").unwrap().is_none(), "definition edit is on top");
        assert!(session.undo_scope("definition:feature:view:2").unwrap().is_some());
        assert!(session.undo_scope("view:1").unwrap().is_some());
    }

    #[test]
    fn coalescing_does_not_cross_view_scopes() {
        let mut document = GraphDocument::new("scope-coalesce");
        document.root.nodes.push(GraphNode { expressions: Default::default(), id: "a".into(), node_type: "solid".into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 });
        let mut session = Session::new(document);
        let command = Command::SetValue { node: "a".into(), field: "length".into(), value: serde_json::json!(1.0) };
        session.apply_commands_scoped(&[command], "view:1", Some("edit"), Duration::from_secs(1)).unwrap();
        let command = Command::SetValue { node: "a".into(), field: "length".into(), value: serde_json::json!(2.0) };
        session.apply_commands_scoped(&[command], "view:2", Some("edit"), Duration::from_secs(1)).unwrap();
        assert!(session.undo_scope("view:2").unwrap().is_some());
        assert_eq!(session.document().nodes[0].values["length"], 1.0);
        assert!(session.undo_scope("view:1").unwrap().is_some());
    }

    #[test]
    fn set_definition_updates_hash_and_undoes() {
        let mut document = GraphDocument::new("definition-command");
        document.root.nodes.push(GraphNode { expressions: Default::default(),
            id: "g".into(),
            node_type: "graph".into(),
            values: serde_json::json!({"definition":"feature"}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.definitions.push(crate::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![GraphNode { expressions: Default::default(),
                id: "box".into(),
                node_type: "solid".into(),
                values: serde_json::json!({"primitiveType":"box"}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::new(),
        });
        let mut session = Session::new(document.clone());
        let before = session.document().node_hashes().unwrap()["g"];
        let mut changed = document.definitions[0].clone();
        changed.nodes[0].values = serde_json::json!({"primitiveType":"sphere"});
        session
            .apply_commands(&[Command::SetDefinition { definition: changed }])
            .expect("set definition");
        let after = session.document().node_hashes().unwrap()["g"];
        assert_ne!(before, after);
        session.undo().expect("undo").expect("change");
        assert_eq!(session.document().node_hashes().unwrap()["g"], before);
    }

    #[test]
    fn graph_node_hash_tracks_its_local_definition() {
        let mut document = GraphDocument::new("definitions-hash");
        document.root.nodes.push(GraphNode { expressions: Default::default(),
            id: "g".into(),
            node_type: "group".into(),
            values: serde_json::json!({
                "name": "Feature",
                "nodes": [
                    {"id":"box","type":"solid","values":{"primitiveType":"box"},"suppressed":false,"x":0.0,"y":0.0}
                ],
                "connections": [],
                "inputs": [],
                "outputs": []
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        groups::migrate_legacy_graph_nodes(&mut document).expect("migrate");
        let before = document.node_hashes().expect("hashes")["g"];
        document.definitions[0].nodes[0].x = 500.0;
        let moved = document.node_hashes().expect("moved hashes")["g"];
        assert_eq!(before, moved, "definition layout must not dirty geometry");
        document.definitions[0].name = "Renamed".into();
        let renamed = document.node_hashes().expect("renamed hashes")["g"];
        assert_eq!(moved, renamed, "definition names must not dirty geometry");
        document.definitions[0].nodes[0].values = serde_json::json!({"primitiveType": "box", "length": 20.0});
        let changed = document.node_hashes().expect("changed hashes")["g"];
        assert_ne!(renamed, changed);
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
            graph.nodes.push(GraphNode { expressions: Default::default(),
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
    fn old_chamfer_fillet_nodes_migrate_to_fillet_type() {
        let graph = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![GraphNode { expressions: Default::default(),
                id: "f".into(),
                node_type: "chamfer".into(),
                values: json!({"mode": "fillet", "length": 3}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
                connections: Vec::new(),
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let graph = Session::new(graph).document().clone();
        assert_eq!(graph.nodes[0].node_type, "fillet");
        assert!(graph.nodes[0].values.get("mode").is_none());
        assert_eq!(graph.nodes[0].values["continuity"], "");
        assert_eq!(graph.nodes[0].values["length"], 3);
    }

    #[test]
    fn session_canonicalizes_structural_defaults_once() {
        let mut graph = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![
                GraphNode { expressions: Default::default(), id: "a".into(), node_type: "primitive".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { expressions: Default::default(), id: "b".into(), node_type: "translate".into(), values: json!({"x": 5.0}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { expressions: Default::default(), id: "c".into(), node_type: "chamfer".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { expressions: Default::default(), id: "d".into(), node_type: "exporter".into(), values: json!({"fileName": "part.step"}), suppressed: false, x: 0.0, y: 0.0 },
                GraphNode { expressions: Default::default(), id: "e".into(), node_type: "spline".into(), values: json!({}), suppressed: false, x: 0.0, y: 0.0 },
            ],
                connections: Vec::new(),
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
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

#[cfg(test)]
mod reconnect_tests {
    use super::*;

    fn node(id: &str, node_type: &str) -> GraphNode {
        GraphNode { expressions: Default::default(), id: id.into(), node_type: node_type.into(), values: serde_json::json!({}), suppressed: false, x: 0.0, y: 0.0 }
    }

    fn connect(from: &str, to: &str) -> Command {
        Command::Connect { from: from.into(), to: to.into(), input: "shape".into(), out: "shape".into() }
    }

    #[test]
    fn reconnect_after_disconnect_reports_dirty_target() {
        let mut document = GraphDocument::new("reconnect");
        document.nodes = vec![node("p", "primitive"), node("t", "translate")];
        let mut session = Session::new(document);
        let first = session.apply_commands(&[connect("p", "t")]).unwrap();
        assert!(first.dirty_nodes.contains(&"t".to_string()));
        session.apply_commands(&[Command::Disconnect { from: "p".into(), to: "t".into(), input: "shape".into() }]).unwrap();
        let reconnect = session.apply_commands(&[connect("p", "t")]).unwrap();
        assert!(reconnect.changed);
        assert!(reconnect.dirty_nodes.contains(&"t".to_string()), "{:?}", reconnect.dirty_nodes);
    }

    #[test]
    fn replacing_an_existing_input_reports_dirty_target() {
        let mut document = GraphDocument::new("replace");
        document.nodes = vec![node("p1", "primitive"), node("p2", "primitive"), node("t", "translate")];
        let mut session = Session::new(document);
        session.apply_commands(&[connect("p1", "t")]).unwrap();
        let replaced = session.apply_commands(&[connect("p2", "t")]).unwrap();
        assert!(replaced.changed);
        assert!(replaced.dirty_nodes.contains(&"t".to_string()), "{:?}", replaced.dirty_nodes);
    }

    #[test]
    fn expression_source_does_not_dirty_geometry() {
        let mut document = GraphDocument::new("expressions");
        document.nodes = vec![node("p", "primitive")];
        let before = document.node_hashes().unwrap();
        document.nodes[0].expressions.insert("length".into(), "2/1".into());
        let after = document.node_hashes().unwrap();
        assert_eq!(before, after, "source text is not geometry content");

        let mut session = Session::new(document);
        let change = session
            .apply_commands(&[Command::SetExpressions {
                node: "p".into(),
                values: [("length".to_owned(), "3/1".to_owned())].into_iter().collect(),
            }])
            .unwrap();
        assert!(change.changed);
        assert!(change.dirty_nodes.is_empty(), "expressions do not invalidate geometry");
        session.undo().unwrap();
        assert_eq!(session.document().nodes[0].expressions["length"], "2/1");
    }

    #[test]
    fn duplicate_connection_is_a_noop() {
        let mut document = GraphDocument::new("duplicate");
        document.nodes = vec![node("p", "primitive"), node("t", "translate")];
        let mut session = Session::new(document);
        session.apply_commands(&[connect("p", "t")]).unwrap();
        let duplicate = session.apply_commands(&[connect("p", "t")]).unwrap();
        assert!(!duplicate.changed);
        assert!(duplicate.dirty_nodes.is_empty());
    }
}
