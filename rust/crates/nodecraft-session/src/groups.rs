//! Inline subpatches (`group` nodes).
//!
//! A group is an editor-only container with an inline child graph. Evaluation
//! expands it into namespaced flat nodes; the engine never sees a group,
//! a nested runtime, or a nested cache.

use crate::{GraphConnection, GraphDefinition, GraphDocument, GraphInterface, GraphNode, SessionError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupPort {
    pub port: String,
    pub node: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupPayload {
    #[serde(default)]
    pub collapsed: bool,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub connections: Vec<GraphConnection>,
    #[serde(default)]
    pub inputs: Vec<GroupPort>,
    #[serde(default)]
    pub outputs: Vec<GroupPort>,
}


/// A graph node's definition reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphReference {
    /// A definition stored in the same document.
    Local(String),
    /// A definition stored in another `.nodecraft` file.
    External { path: String, definition: String },
}

impl GraphReference {
    /// Read the reference from a graph node's values.
    #[must_use]
    pub fn from_node(node: &GraphNode) -> Option<Self> {
        if let Some(definition) = node.values.get("definition").and_then(serde_json::Value::as_str) {
            return Some(Self::Local(definition.to_owned()));
        }
        let external = node.values.get("external")?;
        let path = external.get("path").and_then(serde_json::Value::as_str)?.to_owned();
        let definition = external
            .get("definition")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("root")
            .to_owned();
        Some(Self::External { path, definition })
    }
}

/// Maximum interface ports a graph card can expose. The editor keeps the card
/// narrow; beyond this, split the graph into another reusable definition.
pub const MAX_GRAPH_PORTS: usize = 8;

/// One atomic edit to an inline graph's body interface.
///
/// The caller writes these fields back together so the marker nodes, port
/// arrays and child connections cannot get out of step.
#[derive(Clone, Debug)]
pub struct GraphInterfaceEdit {
    pub nodes: Vec<GraphNode>,
    pub connections: Vec<GraphConnection>,
    pub inputs: Vec<GroupPort>,
    pub outputs: Vec<GroupPort>,
    pub removed_inputs: Vec<String>,
    pub removed_outputs: Vec<String>,
}

/// Resize the declared body input/output ports of an inline graph.
///
/// Existing ports keep their names; a single `groupinput` and `groupoutput`
/// marker expose every port of their direction. New ports get unique `shape`,
/// `shape2`, ... names. Removing a port drops that marker port and any child
/// connection using it, while the interface marker itself remains. The caller
/// is responsible for disconnecting parent connections that used removed port
/// names.
pub fn resize_graph_interface(
    node: &GraphNode,
    input_count: usize,
    output_count: usize,
) -> Result<GraphInterfaceEdit, SessionError> {
    if input_count > MAX_GRAPH_PORTS || output_count > MAX_GRAPH_PORTS {
        return Err(SessionError::InvalidInput(format!(
            "A graph may expose at most {MAX_GRAPH_PORTS} ports in each direction."
        )));
    }
    let mut payload = payload(node)?;
    let mut used_nodes: BTreeSet<String> = payload.nodes.iter().map(|node| node.id.clone()).collect();
    let mut removed_inputs = Vec::new();
    let mut removed_outputs = Vec::new();
    resize_port_list(
        &mut payload.inputs,
        &mut payload.nodes,
        &mut payload.connections,
        &mut used_nodes,
        "groupinput",
        "in",
        input_count,
        -120.0,
        &mut removed_inputs,
    );
    resize_port_list(
        &mut payload.outputs,
        &mut payload.nodes,
        &mut payload.connections,
        &mut used_nodes,
        "groupoutput",
        "out",
        output_count,
        120.0,
        &mut removed_outputs,
    );
    Ok(GraphInterfaceEdit {
        nodes: payload.nodes,
        connections: payload.connections,
        inputs: payload.inputs,
        outputs: payload.outputs,
        removed_inputs,
        removed_outputs,
    })
}

fn resize_port_list(
    ports: &mut Vec<GroupPort>,
    nodes: &mut Vec<GraphNode>,
    connections: &mut Vec<GraphConnection>,
    used_nodes: &mut BTreeSet<String>,
    node_type: &str,
    id_prefix: &str,
    count: usize,
    x: f64,
    removed: &mut Vec<String>,
) {
    let existing = ports.clone();
    let marker_id = existing
        .first()
        .map(|port| port.node.clone())
        .or_else(|| {
            nodes
                .iter()
                .find(|node| node.node_type == node_type)
                .map(|node| node.id.clone())
        })
        .unwrap_or_else(|| next_marker_id(id_prefix, used_nodes));
    used_nodes.insert(marker_id.clone());

    let mut used_ports: BTreeSet<String> = existing.iter().map(|port| port.port.clone()).collect();
    let mut target_names: Vec<String> = existing
        .iter()
        .take(count)
        .map(|port| port.port.clone())
        .collect();
    while target_names.len() < count {
        let port_name = next_port_name(&used_ports);
        used_ports.insert(port_name.clone());
        target_names.push(port_name);
    }
    let target_set: BTreeSet<&str> = target_names.iter().map(String::as_str).collect();
    removed.extend(
        existing
            .iter()
            .filter(|port| !target_set.contains(port.port.as_str()))
            .map(|port| port.port.clone()),
    );

    // Collapse legacy one-marker-per-port nodes onto the single interface node.
    for port in &existing {
        if port.node == marker_id {
            continue;
        }
        for connection in connections.iter_mut() {
            if connection.from == port.node {
                connection.from = marker_id.clone();
                connection.out = port.port.clone();
            }
            if connection.to == port.node {
                connection.to = marker_id.clone();
                connection.input = port.port.clone();
            }
        }
    }
    // Removed port names have no marker port any more.
    connections.retain(|connection| {
        !(connection.from == marker_id && !target_set.contains(connection.out.as_str()))
            && !(connection.to == marker_id && !target_set.contains(connection.input.as_str()))
    });
    // Only one interface node remains per direction.
    nodes.retain(|node| node.id == marker_id || node.node_type != node_type);

    let marker = nodes.iter_mut().find(|node| node.id == marker_id);
    if let Some(marker) = marker {
        marker.node_type = node_type.to_owned();
        marker.values = serde_json::json!({ "ports": &target_names });
    } else {
        nodes.push(GraphNode {
            expressions: Default::default(),
            id: marker_id.clone(),
            node_type: node_type.to_owned(),
            values: serde_json::json!({ "ports": &target_names }),
            suppressed: false,
            x,
            y: 0.0,
        });
    }
    *ports = target_names
        .into_iter()
        .map(|port| GroupPort {
            port,
            node: marker_id.clone(),
        })
        .collect();
}

fn next_port_name(used: &BTreeSet<String>) -> String {
    if !used.contains("shape") {
        return "shape".to_owned();
    }
    let mut index = 2usize;
    loop {
        let candidate = format!("shape{index}");
        if !used.contains(&candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn next_marker_id(prefix: &str, used: &BTreeSet<String>) -> String {
    if !used.contains(prefix) {
        return prefix.to_owned();
    }
    let mut index = 2usize;
    loop {
        let candidate = format!("{prefix}{index}");
        if !used.contains(&candidate) {
            return candidate;
        }
        index += 1;
    }
}

/// Graph interface markers are structural and cannot be deleted or duplicated.
#[must_use]
pub fn is_graph_interface_node(node: &GraphNode) -> bool {
    matches!(node.node_type.as_str(), "groupinput" | "groupoutput")
}

/// Convert legacy inline `group` payloads and `linkedpart` nodes into the
/// definition/instance model. Existing files are converted in place during the
/// Phase 0 transition; no schema-version compatibility layer is kept.
pub fn migrate_legacy_graph_nodes(document: &mut GraphDocument) -> Result<(), SessionError> {
    migrate_legacy_graph_nodes_inner(document, true)
}

/// Convert only inline group payloads. Linked parts keep their existing
/// resolver until external graph references are executable.
pub fn migrate_legacy_group_nodes(document: &mut GraphDocument) -> Result<(), SessionError> {
    migrate_legacy_graph_nodes_inner(document, false)
}

fn migrate_legacy_graph_nodes_inner(
    document: &mut GraphDocument,
    migrate_linked: bool,
) -> Result<(), SessionError> {
    let mut definitions = Vec::new();
    let mut used_ids = std::collections::BTreeSet::new();
    used_ids.insert(document.root.id.clone());
    let mut used_names = std::collections::BTreeSet::new();
    let mut root_nodes = std::mem::take(&mut document.root.nodes);
    migrate_graph_nodes(
        &mut root_nodes,
        "root",
        migrate_linked,
        &mut definitions,
        &mut used_ids,
        &mut used_names,
    )?;
    document.root.nodes = root_nodes;

    let existing = std::mem::take(&mut document.definitions);
    for definition in &existing {
        used_ids.insert(definition.id.clone());
        used_names.insert(definition.name.clone());
    }
    for mut definition in existing {
        migrate_graph_nodes(
            &mut definition.nodes,
            &definition.id,
            migrate_linked,
            &mut definitions,
            &mut used_ids,
            &mut used_names,
        )?;
        document.definitions.push(definition);
    }
    document.definitions.extend(definitions);
    Ok(())
}

fn migrate_graph_nodes(
    nodes: &mut [GraphNode],
    parent_definition: &str,
    migrate_linked: bool,
    definitions: &mut Vec<GraphDefinition>,
    used_ids: &mut std::collections::BTreeSet<String>,
    used_names: &mut std::collections::BTreeSet<String>,
) -> Result<(), SessionError> {
    for node in nodes {
        if node.node_type == "linkedpart" && migrate_linked {
            let path = node
                .values
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            node.node_type = "graph".to_owned();
            node.values = serde_json::json!({
                "name": std::path::Path::new(&path)
                    .file_stem()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Graph"),
                "inputs": ["shape"],
                "outputs": ["shape"],
                "external": { "path": path, "definition": "root" }
            });
            continue;
        }
        if node.node_type != "group" {
            continue;
        }
        let payload = read_group(node)?;
        let base_id = if parent_definition == "root" {
            node.id.clone()
        } else {
            format!("{parent_definition}/{}", node.id)
        };
        let definition_id = unique_definition_id(&base_id, used_ids);
        let requested_name = if payload.name.trim().is_empty() {
            node.id.clone()
        } else {
            payload.name.clone()
        };
        let name = unique_definition_name(&requested_name, used_names);
        let mut child_nodes = payload.nodes;
        migrate_graph_nodes(
            &mut child_nodes,
            &definition_id,
            migrate_linked,
            definitions,
            used_ids,
            used_names,
        )?;
        definitions.push(GraphDefinition {
            id: definition_id.clone(),
            name: name.clone(),
            kind: "graph".to_owned(),
            interface: GraphInterface {
                inputs: payload.inputs.iter().map(|port| port.port.clone()).collect(),
                outputs: payload.outputs.iter().map(|port| port.port.clone()).collect(),
            },
            nodes: child_nodes,
            connections: payload.connections,
        });
        node.node_type = "graph".to_owned();
        let interface_inputs: Vec<String> = payload.inputs.iter().map(|port| port.port.clone()).collect();
        let interface_outputs: Vec<String> = payload.outputs.iter().map(|port| port.port.clone()).collect();
        node.values = serde_json::json!({
            "definition": definition_id,
            "name": name,
            "inputs": interface_inputs,
            "outputs": interface_outputs,
        });
    }
    Ok(())
}

fn unique_definition_id(base: &str, used: &mut std::collections::BTreeSet<String>) -> String {
    if used.insert(base.to_owned()) {
        return base.to_owned();
    }
    let mut index = 2;
    loop {
        let candidate = format!("{base}_{index}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

fn unique_definition_name(base: &str, used: &mut std::collections::BTreeSet<String>) -> String {
    if used.insert(base.to_owned()) {
        return base.to_owned();
    }
    let mut index = 2;
    loop {
        let candidate = format!("{base} {index}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        index += 1;
    }
}

/// Read a group node's inline editor document.
pub fn read_group(node: &GraphNode) -> Result<GroupPayload, SessionError> {
    payload(node)
}

/// Serialize an editor document back into group node values.
pub fn group_values(payload: &GroupPayload) -> Result<serde_json::Value, SessionError> {
    serde_json::to_value(payload)
        .map_err(|error| SessionError::InvalidInput(format!("Graph values: {error}")))
}

fn payload(node: &GraphNode) -> Result<GroupPayload, SessionError> {
    serde_json::from_value(node.values.clone()).map_err(|error| {
        SessionError::InvalidInput(format!("Graph {} has an invalid body: {error}", node.id))
    })
}

fn flat_id(group: &str, child: &str) -> String {
    format!("{group}::{child}")
}

/// One atomic resize of a graph definition's public interface.
#[derive(Clone, Debug)]
pub struct DefinitionInterfaceEdit {
    pub definition: GraphDefinition,
    pub removed_inputs: Vec<String>,
    pub removed_outputs: Vec<String>,
}

/// Resize a definition's public inputs/outputs, updating its marker nodes and
/// owned connections through the same logic as an inline graph body.
pub fn resize_definition_interface(
    definition: &GraphDefinition,
    input_count: usize,
    output_count: usize,
) -> Result<DefinitionInterfaceEdit, SessionError> {
    let inputs = interface_ports(definition, "groupinput", &definition.interface.inputs)?;
    let outputs = interface_ports(definition, "groupoutput", &definition.interface.outputs)?;
    let payload = GroupPayload {
        collapsed: false,
        name: definition.name.clone(),
        nodes: definition.nodes.clone(),
        connections: definition.connections.clone(),
        inputs,
        outputs,
    };
    let values = group_values(&payload)?;
    let node = GraphNode {
        expressions: Default::default(),
        id: definition.id.clone(),
        node_type: "group".into(),
        values,
        suppressed: false,
        x: 0.0,
        y: 0.0,
    };
    let edit = resize_graph_interface(&node, input_count, output_count)?;
    Ok(DefinitionInterfaceEdit {
        definition: GraphDefinition {
            id: definition.id.clone(),
            name: definition.name.clone(),
            kind: definition.kind.clone(),
            interface: GraphInterface {
                inputs: edit.inputs.iter().map(|port| port.port.clone()).collect(),
                outputs: edit.outputs.iter().map(|port| port.port.clone()).collect(),
            },
            nodes: edit.nodes,
            connections: edit.connections,
        },
        removed_inputs: edit.removed_inputs,
        removed_outputs: edit.removed_outputs,
    })
}

/// Materialize graph-instance nodes as inline group payloads and flatten the
/// result. Local definitions resolve in the current document; external
/// definitions are loaded through the canonical store. This remains the bridge
/// until direct definition flattening lands.
pub fn flatten_inline_graphs(document: &GraphDocument) -> Result<GraphDocument, SessionError> {
    expand_groups(document)
}

fn marker_ports(node: &GraphNode) -> Vec<&str> {
    if let Some(ports) = node.values.get("ports").and_then(|value| value.as_array()) {
        return ports.iter().filter_map(|port| port.as_str()).collect();
    }
    node.values
        .get("port")
        .and_then(|value| value.as_str())
        .into_iter()
        .collect()
}

fn interface_ports(
    definition: &crate::GraphDefinition,
    marker_type: &str,
    names: &[String],
) -> Result<Vec<GroupPort>, SessionError> {
    let markers: Vec<&GraphNode> = definition
        .nodes
        .iter()
        .filter(|node| node.node_type == marker_type)
        .collect();
    let mut ports = Vec::with_capacity(names.len());
    for name in names {
        let marker = markers
            .iter()
            .find(|node| marker_ports(node).contains(&name.as_str()))
            .ok_or_else(|| {
                SessionError::InvalidInput(format!(
                    "Graph definition {} has no {marker_type} port {name:?}.",
                    definition.id
                ))
            })?;
        ports.push(GroupPort {
            port: name.clone(),
            node: marker.id.clone(),
        });
    }
    Ok(ports)
}

/// Expand every group into its namespaced child graph and rewire its interface.
pub fn expand_groups(document: &GraphDocument) -> Result<GraphDocument, SessionError> {
    let mut stack = Vec::new();
    expand_groups_at(document, 0, &mut stack)
}

fn expand_groups_at(
    document: &GraphDocument,
    depth: usize,
    stack: &mut Vec<String>,
) -> Result<GraphDocument, SessionError> {
    if depth > 64 {
        return Err(SessionError::InvalidInput("Graph nesting exceeds 64 levels.".into()));
    }
    let references: Vec<GraphNode> = document
        .nodes
        .iter()
        .filter(|node| node.node_type == "group" || node.node_type == "graph")
        .cloned()
        .collect();
    if references.is_empty() {
        return Ok(document.clone());
    }

    let mut out = GraphDocument {
        schema_version: document.schema_version,
        id: document.id.clone(),
        root: crate::GraphDefinition {
            id: "root".into(),
            name: "Root".into(),
            kind: "graph".into(),
            nodes: document
                .nodes
                .iter()
                .filter(|node| node.node_type != "group" && node.node_type != "graph")
                .cloned()
                .collect(),
            connections: Vec::new(),
            interface: GraphInterface::default(),
        },
        definitions: Vec::new(),
    };
    let mut known: BTreeSet<String> = out.nodes.iter().map(|node| node.id.clone()).collect();

    // Connections that do not touch a group or graph stay as they are.
    for connection in &document.connections {
        let touches_reference = references
            .iter()
            .any(|node| node.id == connection.to || node.id == connection.from);
        if !touches_reference {
            out.connections.push(connection.clone());
        }
    }

    for reference in &references {
        let (child_nodes, child_connections, inputs, outputs, child_definitions, reference_key) =
            match reference.node_type.as_str() {
                "group" => {
                    let payload = payload(reference)?;
                    (
                        payload.nodes,
                        payload.connections,
                        payload.inputs,
                        payload.outputs,
                        document.definitions.clone(),
                        None,
                    )
                }
                "graph" => {
                    let reference_value = GraphReference::from_node(reference).ok_or_else(|| {
                        SessionError::InvalidInput(format!(
                            "Graph {} has no definition.",
                            reference.id
                        ))
                    })?;
                    let (definition, definitions, key) = match reference_value {
                        GraphReference::Local(id) => {
                            let definition = document.definition(&id).cloned().ok_or_else(|| {
                                SessionError::InvalidInput(format!(
                                    "Missing graph definition {id}."
                                ))
                            })?;
                            (definition, document.definitions.clone(), format!("local:{id}"))
                        }
                        GraphReference::External { path, definition } => {
                            let requested = definition;
                            let external = crate::store::load(std::path::Path::new(&path))
                                .map_err(SessionError::InvalidInput)?;
                            let definition = external.definition(&requested).cloned().ok_or_else(|| {
                                SessionError::InvalidInput(format!(
                                    "Missing external graph definition {requested} in {path}."
                                ))
                            })?;
                            (
                                definition,
                                external.definitions.clone(),
                                format!("external:{path}:{requested}"),
                            )
                        }
                    };
                    let inputs = interface_ports(&definition, "groupinput", &definition.interface.inputs)?;
                    let outputs = interface_ports(&definition, "groupoutput", &definition.interface.outputs)?;
                    (
                        definition.nodes,
                        definition.connections,
                        inputs,
                        outputs,
                        definitions,
                        Some(key),
                    )
                }
                _ => unreachable!(),
            };
        if let Some(key) = &reference_key {
            if stack.iter().any(|candidate| candidate == key) {
                return Err(SessionError::InvalidInput(format!(
                    "Graph definition cycle through {key}."
                )));
            }
            stack.push(key.clone());
        }
        let child = expand_groups_at(
            &GraphDocument {
                schema_version: document.schema_version,
                id: reference.id.clone(),
                root: crate::GraphDefinition {
                    id: "root".into(),
                    name: "Root".into(),
                    kind: "graph".into(),
                    nodes: child_nodes,
                    connections: child_connections,
                    interface: GraphInterface::default(),
                },
                definitions: child_definitions,
            },
            depth + 1,
            stack,
        )?;
        if reference_key.is_some() {
            stack.pop();
        }

        // External sources entering the reference by port.
        let mut input_sources: BTreeMap<String, Vec<GraphConnection>> = BTreeMap::new();
        for connection in document
            .connections
            .iter()
            .filter(|connection| connection.to == reference.id)
        {
            let Some(port) = inputs.iter().find(|port| port.port == connection.input) else {
                return Err(SessionError::InvalidInput(format!(
                    "Graph {} has no input port {:?}.",
                    reference.id, connection.input
                )));
            };
            input_sources
                .entry(port.port.clone())
                .or_default()
                .push(connection.clone());
        }
        // External targets leaving the reference by port.
        let mut output_targets: BTreeMap<String, Vec<GraphConnection>> = BTreeMap::new();
        for connection in document
            .connections
            .iter()
            .filter(|connection| connection.from == reference.id)
        {
            let Some(port) = outputs.iter().find(|port| port.port == connection.out) else {
                return Err(SessionError::InvalidInput(format!(
                    "Graph {} has no output port {:?}.",
                    reference.id, connection.out
                )));
            };
            output_targets
                .entry(port.port.clone())
                .or_default()
                .push(connection.clone());
        }

        let input_markers: BTreeSet<&str> = inputs.iter().map(|port| port.node.as_str()).collect();
        let output_markers: BTreeSet<&str> = outputs.iter().map(|port| port.node.as_str()).collect();

        // Keep a preview target for the graph card itself. The interface
        // markers are removed from the flat graph, so without this synthetic
        // combine the graph id would vanish and the viewer could not inspect it.
        let mut preview_sources = BTreeSet::new();
        for connection in &child.connections {
            if !output_markers.contains(connection.to.as_str()) {
                continue;
            }
            if input_markers.contains(connection.from.as_str()) {
                // A direct input->output passthrough has no child node to show.
                // Use the external sources entering that marker so the graph
                // card still gets a breakpoint target.
                let ports: Vec<&GroupPort> = inputs
                    .iter()
                    .filter(|port| port.node == connection.from)
                    .collect();
                let from_port = if ports.len() == 1 {
                    ports[0].port.clone()
                } else {
                    connection.out.clone()
                };
                if let Some(sources) = input_sources.get(&from_port) {
                    for source in sources {
                        preview_sources.insert(source.from.clone());
                    }
                }
            } else {
                preview_sources.insert(flat_id(&reference.id, &connection.from));
            }
        }

        for child_node in child.root.nodes {
            if input_markers.contains(child_node.id.as_str())
                || output_markers.contains(child_node.id.as_str())
            {
                continue;
            }
            let mut child_node = child_node;
            child_node.id = flat_id(&reference.id, &child_node.id);
            if !known.insert(child_node.id.clone()) {
                return Err(SessionError::DuplicateNode(child_node.id));
            }
            out.nodes.push(child_node);
        }

        if !preview_sources.is_empty() {
            out.nodes.push(GraphNode {
                expressions: Default::default(),
                id: reference.id.clone(),
                node_type: "combine".into(),
                values: serde_json::json!({}),
                suppressed: false,
                x: reference.x,
                y: reference.y,
            });
            for source in preview_sources {
                out.connections.push(GraphConnection {
                    from: source,
                    to: reference.id.clone(),
                    input: "shape".into(),
                    out: "shape".into(),
                });
            }
        }

        let input_port_for = |marker: &str, child_port: &str| -> String {
            let ports: Vec<&GroupPort> = inputs
                .iter()
                .filter(|port| port.node == marker)
                .collect();
            if ports.len() == 1 {
                ports[0].port.clone()
            } else {
                child_port.to_owned()
            }
        };
        let output_port_for = |marker: &str, child_port: &str| -> String {
            let ports: Vec<&GroupPort> = outputs
                .iter()
                .filter(|port| port.node == marker)
                .collect();
            if ports.len() == 1 {
                ports[0].port.clone()
            } else {
                child_port.to_owned()
            }
        };
        for connection in child.root.connections {
            let from_is_input = input_markers.contains(connection.from.as_str());
            let to_is_output = output_markers.contains(connection.to.as_str());
            match (from_is_input, to_is_output) {
                (true, true) => {
                    // A fresh graph starts as an input-to-output passthrough.
                    let from_port = input_port_for(&connection.from, &connection.out);
                    let to_port = output_port_for(&connection.to, &connection.input);
                    if let (Some(sources), Some(targets)) = (
                        input_sources.get(&from_port),
                        output_targets.get(&to_port),
                    ) {
                        for source in sources {
                            for target in targets {
                                out.connections.push(GraphConnection {
                                    from: source.from.clone(),
                                    to: target.to.clone(),
                                    input: target.input.clone(),
                                    out: source.out.clone(),
                                });
                            }
                        }
                    }
                }
                (true, false) => {
                    let from_port = input_port_for(&connection.from, &connection.out);
                    if let Some(sources) = input_sources.get(&from_port) {
                        for source in sources {
                            out.connections.push(GraphConnection {
                                from: source.from.clone(),
                                to: flat_id(&reference.id, &connection.to),
                                input: connection.input.clone(),
                                out: source.out.clone(),
                            });
                        }
                    }
                }
                (false, true) => {
                    let to_port = output_port_for(&connection.to, &connection.input);
                    if let Some(targets) = output_targets.get(&to_port) {
                        for target in targets {
                            out.connections.push(GraphConnection {
                                from: flat_id(&reference.id, &connection.from),
                                to: target.to.clone(),
                                input: target.input.clone(),
                                out: connection.out.clone(),
                            });
                        }
                    }
                }
                (false, false) => {
                    out.connections.push(GraphConnection {
                        from: flat_id(&reference.id, &connection.from),
                        to: flat_id(&reference.id, &connection.to),
                        input: connection.input.clone(),
                        out: connection.out.clone(),
                    });
                }
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphInterface;
    use serde_json::json;

    fn node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
        GraphNode { expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    fn connection(from: &str, to: &str, input: &str, out: &str) -> GraphConnection {
        GraphConnection {
            from: from.into(),
            to: to.into(),
            input: input.into(),
            out: out.into(),
        }
    }

    #[test]
    fn group_expands_to_flat_rewired_nodes() {
        let group_values = json!({
            "nodes": [
                node("in", "groupinput", json!({"port":"shape"})),
                node("move", "translate", json!({"x":5.0})),
                node("out", "groupoutput", json!({"port":"shape"}))
            ],
            "connections": [
                connection("in", "move", "shape", "shape"),
                connection("move", "out", "shape", "shape")
            ],
            "inputs": [{"port":"shape","node":"in"}],
            "outputs": [{"port":"shape","node":"out"}]
        });
        let document = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![
                node("box", "primitive", json!({"primitiveType":"box"})),
                node("g", "group", group_values),
                node("after", "translate", json!({"y":2.0})),
            ],
                connections: vec![
                connection("box", "g", "shape", "shape"),
                connection("g", "after", "shape", "shape"),
            ],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let flat = expand_groups(&document).expect("expand");
        assert!(flat.nodes.iter().all(|node| node.node_type != "group"));
        assert!(flat.nodes.iter().all(|node| node.id != "in" && node.id != "out"));
        assert!(flat.nodes.iter().any(|node| node.id == "g::move"));
        assert!(flat.connections.iter().any(|link| {
            link.from == "box" && link.to == "g::move" && link.input == "shape"
        }));
        assert!(flat.connections.iter().any(|link| {
            link.from == "g::move" && link.to == "after" && link.input == "shape"
        }));
    }

    #[test]
    fn fresh_group_is_a_passthrough_between_its_ports() {
        let document = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![
                node("box", "primitive", json!({})),
                node("g", "group", json!({
                    "nodes": [
                        node("in", "groupinput", json!({"port":"shape"})),
                        node("out", "groupoutput", json!({"port":"shape"}))
                    ],
                    "connections": [connection("in", "out", "shape", "shape")],
                    "inputs": [{"port":"shape","node":"in"}],
                    "outputs": [{"port":"shape","node":"out"}]
                })),
                node("after", "translate", json!({})),
            ],
                connections: vec![
                connection("box", "g", "shape", "shape"),
                connection("g", "after", "shape", "shape"),
            ],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let flat = expand_groups(&document).expect("expand");
        assert!(flat.connections.iter().any(|link| {
            link.from == "box" && link.to == "after" && link.input == "shape"
        }));
        assert!(flat.nodes.iter().all(|node| node.node_type != "group"));
    }

    #[test]
    fn nested_groups_are_namespaced_recursively() {
        let inner = json!({
            "nodes": [
                node("in", "groupinput", json!({"port":"shape"})),
                node("move", "translate", json!({"x":1.0})),
                node("out", "groupoutput", json!({"port":"shape"}))
            ],
            "connections": [connection("in", "move", "shape", "shape"), connection("move", "out", "shape", "shape")],
            "inputs": [{"port":"shape","node":"in"}],
            "outputs": [{"port":"shape","node":"out"}]
        });
        let outer = json!({
            "nodes": [
                node("in", "groupinput", json!({"port":"shape"})),
                node("inner", "group", inner),
                node("out", "groupoutput", json!({"port":"shape"}))
            ],
            "connections": [
                connection("in", "inner", "shape", "shape"),
                connection("inner", "out", "shape", "shape")
            ],
            "inputs": [{"port":"shape","node":"in"}],
            "outputs": [{"port":"shape","node":"out"}]
        });
        let document = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![node("box", "primitive", json!({})), node("g", "group", outer)],
                connections: vec![connection("box", "g", "shape", "shape")],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let flat = expand_groups(&document).expect("expand");
        assert!(flat.nodes.iter().any(|node| node.id == "g::inner::move"));
        assert!(flat.nodes.iter().all(|node| node.node_type != "group"));
    }
}

#[cfg(test)]
mod interface_tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
        GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    fn connection(from: &str, to: &str, input: &str, out: &str) -> GraphConnection {
        GraphConnection {
            from: from.into(),
            to: to.into(),
            input: input.into(),
            out: out.into(),
        }
    }

    fn graph() -> GraphNode {
        GraphNode {
            expressions: Default::default(),
            id: "graph".into(),
            node_type: "group".into(),
            values: json!({
                "collapsed": false,
                "nodes": [
                    {"id":"in","type":"groupinput","values":{"port":"shape"},"suppressed":false,"x":-120.0,"y":0.0},
                    {"id":"child","type":"primitive","values":{},"suppressed":false,"x":0.0,"y":0.0},
                    {"id":"out","type":"groupoutput","values":{"port":"shape"},"suppressed":false,"x":120.0,"y":0.0}
                ],
                "connections": [
                    {"from":"in","to":"child","in":"shape","out":"shape"},
                    {"from":"child","to":"out","in":"shape","out":"shape"}
                ],
                "inputs": [{"port":"shape","node":"in"}],
                "outputs": [{"port":"shape","node":"out"}]
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    #[test]
    fn resizing_interface_keeps_markers_and_ports_in_step() {
        let graph = graph();
        let edit = resize_graph_interface(&graph, 3, 2).expect("resize");
        assert_eq!(edit.inputs.len(), 3);
        assert_eq!(edit.outputs.len(), 2);
        assert_eq!(edit.inputs[0].port, "shape");
        assert_eq!(edit.inputs[1].port, "shape2");
        assert_eq!(edit.inputs[2].port, "shape3");
        assert_eq!(edit.outputs[0].port, "shape");
        assert_eq!(edit.outputs[1].port, "shape2");
        let inputs: Vec<_> = edit.nodes.iter().filter(|node| node.node_type == "groupinput").collect();
        let outputs: Vec<_> = edit.nodes.iter().filter(|node| node.node_type == "groupoutput").collect();
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);
        assert_eq!(inputs[0].values["ports"].as_array().unwrap().len(), 3);
        assert_eq!(outputs[0].values["ports"].as_array().unwrap().len(), 2);
        // The existing input-to-child connection is preserved; new ports are
        // unconnected until the user wires them inside the graph.
        assert!(edit.connections.iter().any(|link| link.from == "in" && link.to == "child"));
    }

    #[test]
    fn single_interface_markers_feed_all_declared_ports() {
        let group_values = json!({
            "nodes": [
                {"id":"in","type":"groupinput","values":{"ports":["shape","shape2"]},"suppressed":false,"x":-120.0,"y":0.0},
                {"id":"out","type":"groupoutput","values":{"ports":["shape"]},"suppressed":false,"x":120.0,"y":0.0}
            ],
            "connections": [
                {"from":"in","to":"out","in":"shape","out":"shape"},
                {"from":"in","to":"out","in":"shape","out":"shape2"}
            ],
            "inputs": [{"port":"shape","node":"in"},{"port":"shape2","node":"in"}],
            "outputs": [{"port":"shape","node":"out"}]
        });
        let document = GraphDocument {
            schema_version: 1,
            id: "root".into(),
            root: crate::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes: vec![
                node("a", "primitive", json!({"primitiveType":"box"})),
                node("b", "primitive", json!({"primitiveType":"box"})),
                node("g", "group", group_values),
                node("after", "translate", json!({})),
            ],
                connections: vec![
                connection("a", "g", "shape", "shape"),
                connection("b", "g", "shape2", "shape"),
                connection("g", "after", "shape", "shape"),
            ],
                interface: GraphInterface::default(),
            },
            definitions: Vec::new(),
        };
        let flat = expand_groups(&document).expect("expand");
        assert!(flat.connections.iter().any(|link| link.from == "a" && link.to == "after"));
        assert!(flat.connections.iter().any(|link| link.from == "b" && link.to == "after"));
    }

    #[test]
    fn graph_interface_nodes_cannot_be_removed() {
        let mut document = GraphDocument::new("interface");
        document.nodes.push(node("in", "groupinput", json!({"ports":["shape"]})));
        let mut session = crate::Session::new(document);
        let error = session
            .apply_commands(&[crate::Command::RemoveNode { id: "in".into() }])
            .expect_err("interface removal must fail");
        assert!(error.to_string().contains("cannot be deleted"), "{error}");
    }

    #[test]
    fn definition_interface_resize_updates_markers_and_ports() {
        let definition = GraphDefinition {
            id: "rounded".into(),
            name: "Rounded".into(),
            kind: "graph".into(),
            interface: GraphInterface {
                inputs: vec!["shape".into()],
                outputs: vec!["shape".into()],
            },
            nodes: vec![
                node("in", "groupinput", json!({"ports":["shape"]})),
                node("out", "groupoutput", json!({"ports":["shape"]})),
            ],
            connections: vec![connection("in", "out", "shape", "shape")],
        };
        let edit = resize_definition_interface(&definition, 3, 2).expect("resize");
        assert_eq!(edit.definition.interface.inputs, vec!["shape", "shape2", "shape3"]);
        assert_eq!(edit.definition.interface.outputs, vec!["shape", "shape2"]);
        assert_eq!(edit.definition.nodes.iter().filter(|node| node.node_type == "groupinput").count(), 1);
        assert_eq!(edit.definition.nodes.iter().filter(|node| node.node_type == "groupoutput").count(), 1);
        assert_eq!(edit.removed_inputs, Vec::<String>::new());
    }

    #[test]
    fn resized_interface_round_trips_through_the_document_json() {
        let graph = graph();
        let edit = resize_graph_interface(&graph, 3, 2).expect("resize");
        let mut values = graph.values.clone();
        let object = values.as_object_mut().expect("object");
        object.insert("nodes".into(), serde_json::to_value(&edit.nodes).unwrap());
        object.insert(
            "connections".into(),
            serde_json::to_value(&edit.connections).unwrap(),
        );
        object.insert("inputs".into(), serde_json::to_value(&edit.inputs).unwrap());
        object.insert(
            "outputs".into(),
            serde_json::to_value(&edit.outputs).unwrap(),
        );
        let reloaded = GraphNode { values, ..graph };
        let payload = read_group(&reloaded).expect("read back");
        assert_eq!(payload.inputs.len(), 3);
        assert_eq!(payload.outputs.len(), 2);
        assert_eq!(payload.inputs[2].port, "shape3");
        assert_eq!(payload.outputs[1].port, "shape2");
        let input_marker = payload
            .nodes
            .iter()
            .find(|node| node.node_type == "groupinput")
            .expect("input marker");
        assert_eq!(input_marker.values["ports"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn shrinking_interface_removes_markers_and_their_child_wires() {
        let graph = graph();
        let edit = resize_graph_interface(&graph, 0, 0).expect("resize");
        assert!(edit.inputs.is_empty());
        assert!(edit.outputs.is_empty());
        assert_eq!(edit.removed_inputs, vec!["shape"]);
        assert_eq!(edit.removed_outputs, vec!["shape"]);
        let inputs: Vec<_> = edit.nodes.iter().filter(|node| node.node_type == "groupinput").collect();
        let outputs: Vec<_> = edit.nodes.iter().filter(|node| node.node_type == "groupoutput").collect();
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);
        assert!(inputs[0].values["ports"].as_array().unwrap().is_empty());
        assert!(outputs[0].values["ports"].as_array().unwrap().is_empty());
        assert!(edit.connections.is_empty());
    }
}

#[cfg(test)]
mod migration_tests {
    use super::*;
    use serde_json::json;

    fn graph_node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
        GraphNode { expressions: Default::default(), id: id.into(), node_type: node_type.into(), values, suppressed: false, x: 0.0, y: 0.0 }
    }

    #[test]
    fn group_payload_becomes_a_local_definition() {
        let group = graph_node("g", "group", json!({
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
        }));
        let mut document = GraphDocument::new("migration");
        document.root.nodes.push(group);
        migrate_legacy_graph_nodes(&mut document).expect("migrate");
        assert_eq!(document.root.nodes.len(), 1);
        assert_eq!(document.root.nodes[0].node_type, "graph");
        assert_eq!(
            GraphReference::from_node(&document.root.nodes[0]),
            Some(GraphReference::Local("g".into()))
        );
        assert_eq!(document.root.nodes[0].values["inputs"], serde_json::json!(["shape"]));
        let definition = document.definition("g").expect("local definition");
        assert_eq!(definition.name, "Rounded");
        assert_eq!(definition.interface.inputs, vec!["shape"]);
        assert_eq!(definition.nodes.iter().find(|node| node.id == "box").unwrap().node_type, "solid");
    }

    #[test]
    fn linked_part_becomes_an_external_graph_reference() {
        let mut document = GraphDocument::new("linked");
        document.root.nodes.push(graph_node("base", "linkedpart", json!({"path":"base.nodecraft"})));
        migrate_legacy_graph_nodes(&mut document).expect("migrate");
        assert_eq!(
            GraphReference::from_node(&document.root.nodes[0]),
            Some(GraphReference::External { path: "base.nodecraft".into(), definition: "root".into() })
        );
    }

    #[test]
    fn nested_graph_definitions_flatten_directly() {
        let inner = GraphDefinition {
            id: "inner".into(),
            name: "Inner".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![graph_node("box", "solid", json!({"primitiveType":"box"}))],
            connections: Vec::new(),
        };
        let outer = GraphDefinition {
            id: "outer".into(),
            name: "Outer".into(),
            kind: "graph".into(),
            interface: GraphInterface::default(),
            nodes: vec![graph_node("nested", "graph", json!({"definition":"inner"}))],
            connections: Vec::new(),
        };
        let mut document = GraphDocument::new("nested-definitions");
        document.definitions = vec![inner, outer];
        document.root.nodes.push(graph_node("g", "graph", json!({"definition":"outer"})));
        let flat = flatten_inline_graphs(&document).expect("flatten");
        assert!(flat.nodes.iter().any(|node| node.id == "g::nested::box"));
    }

    #[test]
    fn graph_passthrough_keeps_a_breakpoint_target() {
        let definition = GraphDefinition {
            id: "pass".into(),
            name: "Pass".into(),
            kind: "graph".into(),
            interface: GraphInterface {
                inputs: vec!["shape".into()],
                outputs: vec!["shape".into()],
            },
            nodes: vec![
                graph_node("in", "groupinput", json!({"ports":["shape"]})),
                graph_node("out", "groupoutput", json!({"ports":["shape"]})),
            ],
            connections: vec![GraphConnection {
                from: "in".into(),
                to: "out".into(),
                input: "shape".into(),
                out: "shape".into(),
            }],
        };
        let mut document = GraphDocument::new("pass-root");
        document.definitions.push(definition);
        document.root.nodes.push(graph_node("solid", "solid", json!({"primitiveType":"box"})));
        document.root.nodes.push(graph_node("g", "graph", json!({"definition":"pass"})));
        document.root.connections.push(GraphConnection {
            from: "solid".into(),
            to: "g".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        let flat = flatten_inline_graphs(&document).expect("flatten");
        assert!(flat.nodes.iter().any(|node| node.id == "g" && node.node_type == "combine"));
        assert!(flat.connections.iter().any(|connection| {
            connection.from == "solid" && connection.to == "g"
        }));
    }

    #[test]
    fn external_definition_flattens_through_the_store() {
        let dir = std::env::temp_dir().join(format!(
            "sn-external-graph-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("external.nodecraft");
        let mut external = GraphDocument::new("external");
        external.root.interface = crate::GraphInterface {
            inputs: vec!["shape".into()],
            outputs: vec!["shape".into()],
        };
        external.root.nodes = vec![
            graph_node("in", "groupinput", json!({"ports":["shape"]})),
            graph_node("box", "solid", json!({"primitiveType":"box"})),
            graph_node("out", "groupoutput", json!({"ports":["shape"]})),
        ];
        external.root.connections = vec![
            crate::GraphConnection { from: "in".into(), to: "box".into(), input: "shape".into(), out: "shape".into() },
            crate::GraphConnection { from: "box".into(), to: "out".into(), input: "shape".into(), out: "shape".into() },
        ];
        crate::store::save_atomic(&path, &external).expect("save external");

        let mut document = GraphDocument::new("external-root");
        let mut graph = graph_node("g", "graph", json!({"inputs":["shape"], "outputs":["shape"]}));
        graph.values["external"] = json!({"path": path.to_string_lossy(), "definition": "root"});
        document.root.nodes.push(graph);
        let flat = flatten_inline_graphs(&document).expect("flatten external");
        assert!(flat.nodes.iter().any(|node| node.id == "g::box"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn migrated_local_definition_flattens_to_owned_nodes() {
        let group = graph_node("g", "group", json!({
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
        }));
        let mut document = GraphDocument::new("flatten");
        document.root.nodes.push(group);
        migrate_legacy_graph_nodes(&mut document).expect("migrate");
        let flat = flatten_inline_graphs(&document).expect("flatten");
        assert!(flat.nodes.iter().all(|node| node.node_type != "group" && node.node_type != "graph"));
        assert!(flat.nodes.iter().any(|node| node.id == "g::box"));
    }

    #[test]
    fn nested_groups_get_scoped_definition_ids() {
        let inner = json!({
            "nodes":[{"id":"box","type":"solid","values":{},"suppressed":false,"x":0.0,"y":0.0}],
            "connections":[]
        });
        let outer = json!({
            "nodes":[
                {"id":"inner","type":"group","values":inner,"suppressed":false,"x":0.0,"y":0.0}
            ],
            "connections":[]
        });
        let mut document = GraphDocument::new("nested");
        document.root.nodes.push(graph_node("outer", "group", outer));
        migrate_legacy_graph_nodes(&mut document).expect("migrate");
        assert!(document.definition("outer").is_some());
        assert!(document.definition("outer/inner").is_some());
    }
}
