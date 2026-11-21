//! Durable graph documents.
//!
//! A `.nodecraft` file is the authoritative graph. Loading canonicalizes values and
//! validates topology; saving is atomic (same-directory temp file + rename) and writes
//! a stable JSON document. Runtime cache keys, progress, and selection are not part of
//! the durable document.

use crate::{ContentHash, GraphDocument, Session, SessionError, canonicalize_node};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Stable identity of a document's graph content.
#[must_use]
pub fn document_hash(document: &GraphDocument) -> Result<ContentHash, SessionError> {
    let hashes = document.node_hashes()?;
    let mut combined = 0u128;
    for hash in hashes.values() {
        combined = combined.wrapping_add(hash.0).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    Ok(ContentHash(combined))
}

/// Reads, canonicalizes and validates a `.nodecraft` document.
fn canonicalize_all(document: &mut GraphDocument) {
    for node in &mut document.root.nodes {
        canonicalize_node(node);
    }
    for definition in &mut document.definitions {
        for node in &mut definition.nodes {
            canonicalize_node(node);
        }
    }
}

pub fn load(path: &Path) -> Result<GraphDocument, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut document: GraphDocument =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    crate::groups::migrate_legacy_group_nodes(&mut document)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    canonicalize_all(&mut document);
    document
        .validate_definitions()
        .and_then(|()| document.node_hashes())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(document)
}

/// Rewrite an importer's stored path for the file it is about to be saved
/// beside, and record a warning when a relative form is impossible.
///
/// A document saved before the path-mode fields existed has no `pathMode`, and
/// is deliberately left exactly as stored: the derived mode is for display and
/// resolution, not permission to rewrite the document. New nodes write
/// `pathMode` explicitly, which opts them into normalization.
fn rewrite_import_node(node: &mut crate::GraphNode, directory: &Path) {
    // Only an explicitly stored mode opts a node into normalization. Legacy
    // documents keep their exact fields even though `path_mode` derives a mode
    // for the inspector and resolution.
    if node.values.get(nodecraft_engine::import_io::PATH_MODE_KEY).is_none() {
        return;
    }
    let mode = nodecraft_engine::import_io::path_mode(&node.values);
    let Some(stored) = node.values.get("path").and_then(serde_json::Value::as_str) else {
        return;
    };
    if stored.is_empty() {
        return;
    }
    // Resolve through the shared helper so a rooted Windows path stored in a
    // document saved elsewhere is not joined to the document folder.
    let Some(absolute) = nodecraft_engine::import_io::primary_path(&node.values, Some(directory))
    else {
        return;
    };
    let relative = nodecraft_engine::import_io::document_relative(&absolute, directory);
    if mode == "absolute" {
        match relative {
            Some(relative) => {
                node.values["relativePath"] =
                    serde_json::Value::String(relative.to_string_lossy().into_owned());
            }
            None => {
                node.values.as_object_mut().map(|values| values.remove("relativePath"));
            }
        }
        node.values.as_object_mut().map(|values| values.remove("pathWarning"));
    } else if let Some(relative) = relative {
        node.values["path"] = serde_json::Value::String(relative.to_string_lossy().into_owned());
        let object = node.values.as_object_mut().expect("importer values object");
        object.remove("relativePath");
        object.remove("pathWarning");
    } else {
        node.values["path"] = serde_json::Value::String(absolute.to_string_lossy().into_owned());
        node.values["pathWarning"] = serde_json::Value::String(format!(
            "Stored an absolute path because \u{201c}{}\u{201d} is outside the document folder, so it cannot be made relative.",
            absolute.display()
        ));
    }
}

fn prepare_nodes(nodes: &mut [crate::GraphNode], directory: &Path) {
    for node in nodes {
        if nodecraft_node_contract::import_source_node(&node.node_type) {
            rewrite_import_node(node, directory);
        }
    }
}

/// The commands that bring a live session to the same normalized importer
/// paths the save writes. Uses derived commands, so saving does not add undo
/// steps or reorder history.
pub fn import_path_commands(document: &GraphDocument, path: &Path) -> Vec<crate::Command> {
    let Some(directory) = path.parent() else {
        return Vec::new();
    };
    let mut commands = Vec::new();
    for node in &document.root.nodes {
        let mut rewritten = node.clone();
        rewrite_import_node(&mut rewritten, directory);
        if rewritten.values == node.values {
            continue;
        }
        let before = node.values.as_object();
        let after = rewritten.values.as_object();
        if let Some(before) = before {
            for key in before.keys() {
                if !after.is_some_and(|after| after.contains_key(key)) {
                    commands.push(crate::Command::RemoveValue {
                        node: node.id.clone(),
                        field: key.clone(),
                    });
                }
            }
        }
        if let Some(after) = after {
            for (key, value) in after {
                if before.and_then(|before| before.get(key)) != Some(value) {
                    commands.push(crate::Command::SetValue {
                        node: node.id.clone(),
                        field: key.clone(),
                        value: value.clone(),
                    });
                }
            }
        }
    }
    for definition in &document.definitions {
        let mut rewritten = definition.clone();
        for node in &mut rewritten.nodes {
            rewrite_import_node(node, directory);
        }
        if rewritten != *definition {
            commands.push(crate::Command::SetDefinition { definition: rewritten });
        }
    }
    commands
}

/// Apply the save-time importer path normalization and drop unreferenced
/// ingested records. Public so the app can keep its live document in sync with
/// what it is about to write.
pub fn prepare_document_for_save(document: &mut GraphDocument, path: &Path) {
    let Some(directory) = path.parent() else {
        return;
    };
    prepare_nodes(&mut document.root.nodes, directory);
    for definition in &mut document.definitions {
        prepare_nodes(&mut definition.nodes, directory);
    }
    let referenced: std::collections::BTreeSet<String> = document
        .root
        .nodes
        .iter()
        .chain(document.definitions.iter().flat_map(|definition| definition.nodes.iter()))
        .filter_map(|node| {
            node.values.get("ingestHash").and_then(serde_json::Value::as_str).map(str::to_owned)
        })
        .collect();
    document.ingested_sources.retain(|record| referenced.contains(&record.content_hash));
    let mut seen = std::collections::BTreeSet::new();
    document.ingested_sources.retain(|record| seen.insert(record.content_hash.clone()));
}

/// Writes a document atomically. The temp file lives beside the target so the final
/// rename stays on one filesystem.
pub fn save_atomic(path: &Path, document: &GraphDocument) -> Result<(), String> {
    let mut document = document.clone();
    canonicalize_all(&mut document);
    prepare_document_for_save(&mut document, path);
    document
        .validate_definitions()
        .and_then(|()| document.node_hashes())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let value = serde_json::to_value(&document).map_err(|error| error.to_string())?;
    save_value_atomic(path, &value)
}

/// Save an editor envelope with canonical graph values merged into it.
pub fn save_atomic_value(
    path: &Path,
    value: &mut serde_json::Value,
    document: &GraphDocument,
) -> Result<(), String> {
    let mut canonical = document.clone();
    for node in &mut canonical.nodes {
        canonicalize_node(node);
    }
    prepare_document_for_save(&mut canonical, path);
    canonical.node_hashes().map_err(|error| format!("{}: {error}", path.display()))?;
    value["root"] = serde_json::to_value(&canonical.root).map_err(|error| error.to_string())?;
    value["definitions"] =
        serde_json::to_value(&canonical.definitions).map_err(|error| error.to_string())?;
    value["schemaVersion"] = serde_json::json!(canonical.schema_version);
    if canonical.ingested_sources.is_empty() {
        if let Some(object) = value.as_object_mut() {
            object.remove("ingestedSources");
        }
    } else {
        value["ingestedSources"] =
            serde_json::to_value(&canonical.ingested_sources).map_err(|error| error.to_string())?;
    }
    save_value_atomic(path, value)
}

/// Atomic save for a document envelope that carries editor metadata (view state)
/// beside the graph. The caller is responsible for canonical graph validation.
pub fn save_value_atomic(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let bytes = pretty_json(value).into_bytes();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{}: invalid file name", path.display()))?;
    let temp = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = (|| -> Result<(), String> {
        let mut file =
            std::fs::File::create(&temp).map_err(|error| format!("{}: {error}", temp.display()))?;
        use std::io::Write as _;
        file.write_all(&bytes).map_err(|error| format!("{}: {error}", temp.display()))?;
        file.sync_all().map_err(|error| format!("{}: {error}", temp.display()))?;
        std::fs::rename(&temp, path).map_err(|error| format!("{}: {error}", path.display()))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// How a linked part follows its external source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LinkMode {
    FollowLatest,
    Pinned,
}

/// One project's reference to an external `.nodecraft` document.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkedPart {
    pub document_id: String,
    pub path: PathBuf,
    pub content_hash: ContentHash,
    pub mode: LinkMode,
}

/// A linked file that changed on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkChange {
    pub host_id: String,
    pub link_id: String,
    pub path: PathBuf,
    pub old_hash: ContentHash,
    pub new_hash: ContentHash,
    pub mode: LinkMode,
}

/// In-memory registry of open/linked documents, keyed by stable document id.
#[derive(Clone, Debug, Default)]
pub struct DocumentStore {
    documents: BTreeMap<String, GraphDocument>,
    sessions: BTreeMap<String, Session>,
    paths: BTreeMap<String, PathBuf>,
    links: BTreeMap<String, BTreeMap<String, LinkedPart>>,
}

impl DocumentStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        id: impl Into<String>,
        mut document: GraphDocument,
    ) -> Result<(), SessionError> {
        for node in &mut document.nodes {
            canonicalize_node(node);
        }
        document.node_hashes()?;
        let id = id.into();
        self.documents.insert(id, document);
        Ok(())
    }

    #[must_use]
    pub fn get(&self, id: &str) -> Option<&GraphDocument> {
        self.documents.get(id)
    }

    pub fn documents(&self) -> impl Iterator<Item = (&String, &GraphDocument)> {
        self.documents.iter()
    }

    /// Publish the current graph of an open document so linked roots evaluate
    /// against the editor's live revision, not the last tab-switch snapshot.
    pub fn publish_document(&mut self, id: &str, document: &GraphDocument) {
        self.documents.insert(id.to_owned(), document.clone());
    }

    #[must_use]
    pub fn session(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }

    #[must_use]
    pub fn session_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(id)
    }

    /// Update a document from its session and keep the durable graph in sync.
    pub fn set_session(
        &mut self,
        id: impl Into<String>,
        session: Session,
    ) -> Result<(), SessionError> {
        let id = id.into();
        session.document().node_hashes()?;
        self.documents.insert(id.clone(), session.document().clone());
        self.sessions.insert(id, session);
        Ok(())
    }

    pub fn set_path(&mut self, id: &str, path: impl Into<PathBuf>) {
        self.paths.insert(id.to_string(), path.into());
    }

    pub fn clear_path(&mut self, id: &str) {
        self.paths.remove(id);
    }

    #[must_use]
    pub fn id_for_path(&self, path: &Path) -> Option<&str> {
        self.paths
            .iter()
            .find(|(_, candidate)| candidate.as_path() == path)
            .map(|(id, _)| id.as_str())
    }

    pub fn remove(&mut self, id: &str) {
        self.documents.remove(id);
        self.sessions.remove(id);
        self.paths.remove(id);
        for links in self.links.values_mut() {
            links.retain(|_, link| link.document_id != id);
        }
    }

    #[must_use]
    pub fn path(&self, id: &str) -> Option<&Path> {
        self.paths.get(id).map(PathBuf::as_path)
    }

    pub fn load(
        &mut self,
        id: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Result<&GraphDocument, String> {
        let id = id.into();
        let path = path.into();
        let document = load(&path)?;
        self.sessions.insert(id.clone(), Session::new(document.clone()));
        self.paths.insert(id.clone(), path);
        self.documents.insert(id.clone(), document);
        Ok(self.documents.get(&id).expect("inserted document"))
    }

    pub fn save(&mut self, id: &str, path: impl Into<PathBuf>) -> Result<(), String> {
        let document =
            self.documents.get(id).ok_or_else(|| format!("document {id} is not open"))?;
        let path = path.into();
        save_atomic(&path, document)?;
        self.paths.insert(id.to_string(), path);
        Ok(())
    }

    /// Save only if the file on disk still has the hash this document was loaded
    /// from. A mismatch is an external edit and must be resolved by the caller.
    pub fn save_checked(
        &mut self,
        id: &str,
        path: impl Into<PathBuf>,
        expected: ContentHash,
    ) -> Result<(), String> {
        let path = path.into();
        let current = load(&path)?;
        if document_hash(&current).map_err(|error| error.to_string())? != expected {
            return Err(format!("{} changed on disk since it was loaded", path.display()));
        }
        self.save(id, path)
    }

    /// Register an external `.nodecraft` document and a reference to it.
    pub fn link(
        &mut self,
        host_id: &str,
        link_id: &str,
        document_id: &str,
        path: impl Into<PathBuf>,
        mode: LinkMode,
    ) -> Result<&LinkedPart, String> {
        let path = path.into();
        if !self.documents.contains_key(document_id) {
            self.load(document_id, &path)?;
        }
        let document = self
            .documents
            .get(document_id)
            .ok_or_else(|| format!("linked document {document_id} is not open"))?;
        let content_hash = document_hash(document).map_err(|error| error.to_string())?;
        let link = LinkedPart {
            document_id: document_id.to_string(),
            path: path.clone(),
            content_hash,
            mode,
        };
        self.links.entry(host_id.to_string()).or_default().insert(link_id.to_string(), link);
        Ok(self.links[host_id].get(link_id).expect("inserted link"))
    }

    #[must_use]
    pub fn links(&self, host_id: &str) -> Option<&BTreeMap<String, LinkedPart>> {
        self.links.get(host_id)
    }

    #[must_use]
    pub fn resolve(&self, host_id: &str, link_id: &str) -> Option<&GraphDocument> {
        let link = self.links.get(host_id)?.get(link_id)?;
        self.documents.get(&link.document_id)
    }

    /// Re-read linked files and update follow-latest documents. Pinned links are
    /// reported but not updated.
    pub fn detect_link_changes(&mut self) -> Result<Vec<LinkChange>, String> {
        let mut changes = Vec::new();
        for (host_id, links) in &self.links {
            for (link_id, link) in links {
                let Ok(document) = load(&link.path) else {
                    continue;
                };
                let new_hash = document_hash(&document).map_err(|error| error.to_string())?;
                if new_hash == link.content_hash {
                    continue;
                }
                changes.push(LinkChange {
                    host_id: host_id.clone(),
                    link_id: link_id.clone(),
                    path: link.path.clone(),
                    old_hash: link.content_hash,
                    new_hash,
                    mode: link.mode,
                });
            }
        }
        for change in &changes {
            if change.mode == LinkMode::FollowLatest {
                let document = load(&change.path)?;
                if let Some(link) = self
                    .links
                    .get_mut(&change.host_id)
                    .and_then(|links| links.get_mut(&change.link_id))
                {
                    link.content_hash = change.new_hash;
                }
                let document_id = self.links[&change.host_id][&change.link_id].document_id.clone();
                self.sessions.insert(document_id.clone(), Session::new(document.clone()));
                self.documents.insert(document_id, document);
            }
        }
        Ok(changes)
    }
}

/// Pretty JSON that keeps small values inline. `serde_json::to_vec_pretty`
/// expands every array element and object field to its own line, which makes
/// point lists, connections and interface arrays unnecessarily tall.
fn pretty_json(value: &Value) -> String {
    let mut output = String::new();
    write_pretty(value, 0, &mut output);
    output.push('\n');
    output
}

const INLINE_LIMIT: usize = 120;

fn write_pretty(value: &Value, depth: usize, output: &mut String) {
    if depth > 0
        && let Some(inline) = inline_json(value)
    {
        output.push_str(&inline);
        return;
    }
    match value {
        Value::Array(items) => {
            output.push_str("[\n");
            for (index, item) in items.iter().enumerate() {
                output.push_str(&"  ".repeat(depth + 1));
                write_pretty(item, depth + 1, output);
                if index + 1 < items.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&"  ".repeat(depth));
            output.push(']');
        }
        Value::Object(fields) => {
            output.push_str("{\n");
            for (index, (key, field)) in fields.iter().enumerate() {
                output.push_str(&"  ".repeat(depth + 1));
                output.push_str(&serde_json::to_string(key).expect("json key"));
                output.push_str(": ");
                write_pretty(field, depth + 1, output);
                if index + 1 < fields.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&"  ".repeat(depth));
            output.push('}');
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            output.push_str(&serde_json::to_string(value).expect("json scalar"));
        }
    }
}

fn inline_json(value: &Value) -> Option<String> {
    let text = serde_json::to_string(value).ok()?;
    (text.len() <= INLINE_LIMIT).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphConnection, GraphInterface, GraphNode};
    use serde_json::json;

    #[test]
    fn saved_json_keeps_small_arrays_and_objects_inline() {
        let value = serde_json::json!({
            "points": [
                {"x": 1.0, "y": 2.0},
                {"x": 3.0, "y": 4.0}
            ],
            "interface": {"inputs": [], "outputs": ["canopy"]},
            "long": vec!["x"; 200]
        });
        let text = pretty_json(&value);
        assert!(text.contains(r#"{"x":1.0,"y":2.0}"#), "{text}");
        assert!(text.contains(r#"{"inputs":[],"outputs":["canopy"]}"#), "{text}");
        assert!(text.contains("\"long\": [\n"), "large arrays still expand:\n{text}");
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), value);
    }

    fn test_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "sn-document-store-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
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
                    GraphNode {
                        expressions: Default::default(),
                        id: "a".into(),
                        node_type: "primitive".into(),
                        values: json!({}),
                        suppressed: false,
                        x: 10.0,
                        y: 20.0,
                    },
                    GraphNode {
                        expressions: Default::default(),
                        id: "b".into(),
                        node_type: "translate".into(),
                        values: json!({"x": 5.0}),
                        suppressed: false,
                        x: 0.0,
                        y: 0.0,
                    },
                ],
                connections: vec![GraphConnection {
                    from: "a".into(),
                    to: "b".into(),
                    input: "shape".into(),
                    out: "shape".into(),
                }],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
            ingested_sources: Vec::new(),
        }
    }

    #[test]
    fn atomic_round_trip_canonicalizes_and_preserves_topology() {
        let dir = test_dir();
        let path = dir.join("part.nodecraft");
        let mut original = document();
        for node in &mut original.nodes {
            canonicalize_node(node);
        }
        save_atomic(&path, &original).expect("save");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.nodes[0].values["primitiveType"], "box");
        assert_eq!(loaded.nodes[1].values["space"], "world");
        assert_eq!(loaded.connections.len(), 1);
        assert_eq!(document_hash(&loaded).unwrap(), document_hash(&original).unwrap());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn external_graph_reference_round_trips_and_flattens() {
        let dir = test_dir();
        let external_path = dir.join("external.nodecraft");
        let external = GraphDocument {
            schema_version: 1,
            id: "external".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![
                    GraphNode {
                        expressions: Default::default(),
                        id: "in".into(),
                        node_type: "groupinput".into(),
                        values: json!({"ports":["shape"]}),
                        suppressed: false,
                        x: -120.0,
                        y: 0.0,
                    },
                    GraphNode {
                        expressions: Default::default(),
                        id: "box".into(),
                        node_type: "solid".into(),
                        values: json!({"primitiveType":"box"}),
                        suppressed: false,
                        x: 0.0,
                        y: 0.0,
                    },
                    GraphNode {
                        expressions: Default::default(),
                        id: "out".into(),
                        node_type: "groupoutput".into(),
                        values: json!({"ports":["shape"]}),
                        suppressed: false,
                        x: 120.0,
                        y: 0.0,
                    },
                ],
                connections: vec![
                    GraphConnection {
                        from: "in".into(),
                        to: "box".into(),
                        input: "shape".into(),
                        out: "shape".into(),
                    },
                    GraphConnection {
                        from: "box".into(),
                        to: "out".into(),
                        input: "shape".into(),
                        out: "shape".into(),
                    },
                ],
                interface: GraphInterface {
                    inputs: vec!["shape".into()],
                    outputs: vec!["shape".into()],
                },
            },
            definitions: Vec::new(),
            ingested_sources: Vec::new(),
        };
        save_atomic(&external_path, &external).expect("save external");
        let mut root = GraphDocument::new("external-root");
        root.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "g".into(),
            node_type: "graph".into(),
            values: json!({
                "name": "External",
                "inputs": ["shape"],
                "outputs": ["shape"],
                "external": {"path": external_path.to_string_lossy(), "definition": "root"}
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        let root_path = dir.join("root.nodecraft");
        save_atomic(&root_path, &root).expect("save root");
        let loaded = load(&root_path).expect("load root");
        assert_eq!(
            crate::groups::GraphReference::from_node(&loaded.nodes[0]),
            Some(crate::groups::GraphReference::External {
                path: external_path.to_string_lossy().into_owned(),
                definition: "root".into()
            })
        );
        let flat = crate::groups::flatten_inline_graphs(&loaded).expect("flatten external");
        assert!(flat.nodes.iter().any(|node| node.id == "g::box"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn load_migrates_legacy_group_payloads() {
        let dir = test_dir();
        let path = dir.join("legacy.nodecraft");
        let legacy = json!({
            "schemaVersion": 1,
            "id": "legacy",
            "root": {
            "id": "root",
            "name": "Root",
            "kind": "graph",
            "nodes": [{
                "id": "g",
                "type": "group",
                "values": {
                    "name": "Rounded",
                    "nodes": [
                        {"id":"in","type":"groupinput","values":{"ports":["shape"]},"suppressed":false,"x":-120.0,"y":0.0},
                        {"id":"box","type":"solid","values":{"primitiveType":"box"},"suppressed":false,"x":0.0,"y":0.0},
                        {"id":"out","type":"groupoutput","values":{"ports":["shape"]},"suppressed":false,"x":120.0,"y":0.0}
                    ],
                    "connections": [
                        {"from":"in","to":"box","in":"shape","out":"shape"},
                        {"from":"box","to":"out","in":"shape","out":"shape"}
                    ],
                    "inputs": [{"port":"shape","node":"in"}],
                    "outputs": [{"port":"shape","node":"out"}]
                },
                "suppressed": false,
                "x": 0.0,
                "y": 0.0
            }],
            "connections": []
            },
            "definitions": []
        });
        std::fs::write(&path, serde_json::to_string_pretty(&legacy).unwrap()).expect("write");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.root.nodes[0].node_type, "graph");
        assert!(loaded.definition("g").is_some());
        let flat = crate::groups::flatten_inline_graphs(&loaded).expect("flatten");
        assert!(flat.nodes.iter().any(|node| node.id == "g::box"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn migrated_graph_definition_round_trips_and_flattens() {
        let dir = test_dir();
        let path = dir.join("graph.nodecraft");
        let mut doc = GraphDocument::new("migrated");
        doc.root.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "g".into(),
            node_type: "group".into(),
            values: json!({
                "name": "Rounded",
                "nodes": [
                    {"id":"in","type":"groupinput","values":{"ports":["shape"]},"suppressed":false,"x":-120.0,"y":0.0},
                    {"id":"box","type":"solid","values":{"primitiveType":"box"},"suppressed":false,"x":0.0,"y":0.0},
                    {"id":"out","type":"groupoutput","values":{"ports":["shape"]},"suppressed":false,"x":120.0,"y":0.0}
                ],
                "connections":[
                    {"from":"in","to":"box","in":"shape","out":"shape"},
                    {"from":"box","to":"out","in":"shape","out":"shape"}
                ],
                "inputs":[{"port":"shape","node":"in"}],
                "outputs":[{"port":"shape","node":"out"}]
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        crate::groups::migrate_legacy_graph_nodes(&mut doc).expect("migrate");
        save_atomic(&path, &doc).expect("save");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.root.nodes[0].node_type, "graph");
        assert!(loaded.definition("g").is_some());
        let flat = crate::groups::flatten_inline_graphs(&loaded).expect("flatten");
        assert!(flat.nodes.iter().any(|node| node.id == "g::box"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn expression_text_survives_the_document_round_trip() {
        let dir = test_dir();
        let path = dir.join("part.nodecraft");
        let mut doc = document();
        doc.nodes[0].expressions.insert("length".into(), "2/1".into());
        save_atomic(&path, &doc).expect("save");
        let loaded = load(&path).expect("load");
        assert_eq!(loaded.nodes[0].expressions["length"], "2/1");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn document_store_loads_and_saves_open_documents() {
        let dir = test_dir();
        let path = dir.join("part.nodecraft");
        save_atomic(&path, &document()).expect("seed");
        let mut store = DocumentStore::new();
        store.load("doc", &path).expect("load");
        assert!(store.get("doc").is_some());
        let second = dir.join("copy.nodecraft");
        store.save("doc", &second).expect("save");
        assert_eq!(load(&second).expect("copy").connections.len(), 1);
        std::fs::remove_dir_all(dir).ok();
    }
    #[test]
    fn follow_latest_updates_linked_document_and_pinned_does_not() {
        let dir = test_dir();
        let part = dir.join("bracket.nodecraft");
        let mut source = document();
        source.nodes[0].values["length"] = json!(10.0);
        save_atomic(&part, &source).expect("seed part");

        let mut store = DocumentStore::new();
        store
            .link("host", "follow", "part-follow", &part, LinkMode::FollowLatest)
            .expect("follow link");
        store.link("host", "pinned", "part-pinned", &part, LinkMode::Pinned).expect("pinned link");
        let old = store.links("host").unwrap()["follow"].content_hash;

        let mut changed = source.clone();
        changed.nodes[0].values["length"] = json!(20.0);
        save_atomic(&part, &changed).expect("external edit");
        let changes = store.detect_link_changes().expect("detect");
        assert_eq!(changes.len(), 2);
        assert_ne!(store.links("host").unwrap()["follow"].content_hash, old);
        assert_eq!(store.links("host").unwrap()["pinned"].content_hash, old);
        assert_eq!(store.resolve("host", "follow").unwrap().nodes[0].values["length"], 20.0);
        assert_eq!(store.resolve("host", "pinned").unwrap().nodes[0].values["length"], 10.0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn document_store_keeps_sessions_and_path_identity() {
        let dir = test_dir();
        let path = dir.join("part.nodecraft");
        save_atomic(&path, &document()).expect("seed");
        let mut store = DocumentStore::new();
        let mut session = Session::new(load(&path).expect("load"));
        session
            .apply_commands(&[crate::Command::SetPosition { node: "a".into(), x: 42.0, y: 7.0 }])
            .expect("edit");
        store.set_session("doc", session).expect("session");
        store.set_path("doc", &path);
        assert_eq!(store.session("doc").expect("session").document().nodes[0].x, 42.0);
        assert_eq!(store.id_for_path(&path), Some("doc"));
        assert_eq!(store.get("doc").expect("graph").nodes[0].x, 42.0);
        store.remove("doc");
        assert!(store.session("doc").is_none());
        assert!(store.get("doc").is_none());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn published_document_is_visible_to_linked_roots_immediately() {
        let mut store = DocumentStore::new();
        let mut document = GraphDocument::new("child");
        document.nodes.push(crate::GraphNode {
            expressions: Default::default(),
            id: "n".into(),
            node_type: "primitive".into(),
            values: serde_json::json!({}),
            suppressed: false,
            x: 1.0,
            y: 0.0,
        });
        store.set_session("child", Session::new(document.clone())).expect("set session");
        document.nodes[0].x = 42.0;
        store.publish_document("child", &document);
        assert_eq!(store.get("child").expect("document").nodes[0].x, 42.0);
    }

    #[test]
    fn old_importer_documents_keep_their_existing_fields() {
        let dir = test_dir();
        let path = dir.join("old.nodecraft");
        let old_path = dir.join("model.step").display().to_string();
        let legacy = serde_json::json!({
            "schemaVersion": 1,
            "id": "legacy",
            "root": {
                "id": "root",
                "name": "Root",
                "kind": "graph",
                "nodes": [{
                    "id": "s",
                    "type": "stepimporter",
                    "values": {"path": old_path, "heal": true},
                    "suppressed": false,
                    "x": 0.0,
                    "y": 0.0
                }],
                "connections": [],
                "interface": {"inputs": [], "outputs": []}
            },
            "definitions": []
        });
        std::fs::write(&path, serde_json::to_string_pretty(&legacy).unwrap()).expect("write");
        let loaded = load(&path).expect("load old document");
        let node = &loaded.nodes[0];
        assert_eq!(node.values["path"], old_path);
        assert!(node.values.get("pathMode").is_none(), "load must not add pathMode");
        assert!(node.values.get("ingest").is_none(), "load must not add ingest");
        assert!(node.values.get("watch").is_none(), "load must not add watch");
        assert!(loaded.ingested_sources.is_empty());

        let saved = dir.join("saved-old.nodecraft");
        save_atomic(&saved, &loaded).expect("save old document");
        let text = std::fs::read_to_string(&saved).expect("read saved");
        assert!(!text.contains("pathMode"), "old document grew a pathMode field:\n{text}");
        assert!(!text.contains("ingestedSources"), "old document grew an ingest section:\n{text}");
        // Compare the saved value, not the raw text: JSON escapes the
        // backslashes of a Windows path, so a substring search never matches.
        let reread: serde_json::Value = serde_json::from_str(&text).expect("saved JSON");
        let saved_path = reread["root"]["nodes"][0]["values"]["path"].as_str();
        assert_eq!(saved_path, Some(old_path.as_str()), "old absolute path must be preserved");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn saving_rewrites_relative_importers_and_preserves_absolute_mode() {
        let dir = test_dir();
        let source = dir.join("model.step");
        std::fs::write(&source, b"not parsed by this test").unwrap();
        let path = dir.join("part.nodecraft");

        let mut relative = GraphDocument::new("relative");
        relative.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "s".into(),
            node_type: "stepimporter".into(),
            values: json!({
                "path": source.display().to_string(),
                "pathMode": "relative",
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        save_atomic(&path, &relative).expect("save relative");
        let loaded = load(&path).expect("load relative");
        assert_eq!(loaded.nodes[0].values["path"], "model.step");

        let mut absolute = GraphDocument::new("absolute");
        absolute.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "s".into(),
            node_type: "stepimporter".into(),
            values: json!({
                "path": source.display().to_string(),
                "pathMode": "absolute",
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        let absolute_path = dir.join("absolute.nodecraft");
        save_atomic(&absolute_path, &absolute).expect("save absolute");
        let loaded = load(&absolute_path).expect("load absolute");
        assert_eq!(loaded.nodes[0].values["path"], source.display().to_string());
        assert_eq!(loaded.nodes[0].values["relativePath"], "model.step");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_normalization_commands_update_the_live_session() {
        let dir = std::env::temp_dir().join(format!("sn-save-commands-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("part.nodecraft");
        let source = dir.join("model.step");
        let mut document = GraphDocument::new("normalize");
        document.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "s".into(),
            node_type: "stepimporter".into(),
            values: json!({
                "path": source.display().to_string(),
                "pathMode": "relative",
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        let commands = import_path_commands(&document, &path);
        assert!(
            commands.iter().any(|command| matches!(
                command,
                crate::Command::SetValue { field, value, .. }
                    if field == "path" && value == "model.step"
            )),
            "{commands:?}"
        );
        let mut session = Session::new(document);
        session.apply_derived_commands(&commands).expect("normalize");
        assert_eq!(session.document().nodes[0].values["path"], "model.step");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn ingested_sources_round_trip_deduplicate_and_drop_unreferenced() {
        let dir = test_dir();
        let path = dir.join("ingest.nodecraft");
        let used =
            nodecraft_engine::import_io::ingest_record("used.step", b"used bytes").expect("record");
        let unused = nodecraft_engine::import_io::ingest_record("unused.step", b"unused bytes")
            .expect("record");
        let mut document = GraphDocument::new("ingest");
        document.ingested_sources = vec![used.clone(), used.clone(), unused];
        document.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "s".into(),
            node_type: "stepimporter".into(),
            values: json!({
                "path": "used.step",
                "pathMode": "relative",
                "ingest": true,
                "ingestHash": used.content_hash,
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        save_atomic(&path, &document).expect("save ingest");
        let loaded = load(&path).expect("load ingest");
        assert_eq!(
            loaded.ingested_sources,
            vec![used.clone()],
            "only one referenced record survives"
        );
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(
            text.contains("\"ingestedSources\""),
            "records live in one top-level section:\n{text}"
        );
        assert_eq!(
            nodecraft_engine::import_io::decode_ingest_record(&loaded.ingested_sources[0])
                .expect("decode"),
            b"used bytes"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn checked_save_refuses_when_disk_changed() {
        let dir = test_dir();
        let path = dir.join("part.nodecraft");
        save_atomic(&path, &document()).expect("seed");
        let loaded = load(&path).expect("load");
        let expected = document_hash(&loaded).unwrap();
        let mut store = DocumentStore::new();
        store.load("doc", &path).expect("open");

        let mut changed = loaded;
        changed.nodes[0].values["length"] = json!(99.0);
        save_atomic(&path, &changed).expect("external edit");
        let error = store.save_checked("doc", &path, expected).expect_err("conflict");
        assert!(error.contains("changed on disk"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }
}
