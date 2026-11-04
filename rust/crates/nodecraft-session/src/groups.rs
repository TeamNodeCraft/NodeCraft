//! Inline subpatches (`group` nodes).
//!
//! A group is an editor-only container with an inline child graph. Evaluation
//! expands it into namespaced flat nodes; the engine never sees a group,
//! a nested runtime, or a nested cache.

use crate::{GraphConnection, GraphDocument, GraphInterface, GraphNode, SessionError};
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
    pub nodes: Vec<GraphNode>,
    #[serde(default)]
    pub connections: Vec<GraphConnection>,
    #[serde(default)]
    pub inputs: Vec<GroupPort>,
    #[serde(default)]
    pub outputs: Vec<GroupPort>,
}

/// Read a group node's inline editor document.
pub fn read_group(node: &GraphNode) -> Result<GroupPayload, SessionError> {
    payload(node)
}

/// Serialize an editor document back into group node values.
pub fn group_values(payload: &GroupPayload) -> Result<serde_json::Value, SessionError> {
    serde_json::to_value(payload)
        .map_err(|error| SessionError::InvalidInput(format!("Group values: {error}")))
}

fn payload(node: &GraphNode) -> Result<GroupPayload, SessionError> {
    serde_json::from_value(node.values.clone()).map_err(|error| {
        SessionError::InvalidInput(format!("Group {} has an invalid subpatch: {error}", node.id))
    })
}

fn flat_id(group: &str, child: &str) -> String {
    format!("{group}::{child}")
}

/// Expand every group into its namespaced child graph and rewire its interface.
pub fn expand_groups(document: &GraphDocument) -> Result<GraphDocument, SessionError> {
    expand_groups_at(document, 0)
}

fn expand_groups_at(document: &GraphDocument, depth: usize) -> Result<GraphDocument, SessionError> {
    if depth > 64 {
        return Err(SessionError::InvalidInput("Group nesting exceeds 64 levels.".into()));
    }
    let groups: Vec<GraphNode> = document
        .nodes
        .iter()
        .filter(|node| node.node_type == "group")
        .cloned()
        .collect();
    if groups.is_empty() {
        return Ok(document.clone());
    }

    let mut out = GraphDocument {
        schema_version: document.schema_version,
        id: document.id.clone(),
        nodes: document
            .nodes
            .iter()
            .filter(|node| node.node_type != "group")
            .cloned()
            .collect(),
        connections: Vec::new(),
        interface: GraphInterface::default(),
    };
    let mut known: BTreeSet<String> = out.nodes.iter().map(|node| node.id.clone()).collect();

    // Connections that do not touch a group stay as they are.
    for connection in &document.connections {
        let touches_group = groups.iter().any(|group| group.id == connection.to || group.id == connection.from);
        if !touches_group {
            out.connections.push(connection.clone());
        }
    }

    for group in &groups {
        let payload = payload(group)?;
        let child = expand_groups_at(
            &GraphDocument {
                schema_version: document.schema_version,
                id: group.id.clone(),
                nodes: payload.nodes,
                connections: payload.connections,
                interface: GraphInterface::default(),
            },
            depth + 1,
        )?;

        // External sources entering the group by port.
        let mut input_sources: BTreeMap<String, Vec<GraphConnection>> = BTreeMap::new();
        for connection in document
            .connections
            .iter()
            .filter(|connection| connection.to == group.id)
        {
            let Some(port) = payload.inputs.iter().find(|port| port.port == connection.input) else {
                return Err(SessionError::InvalidInput(format!(
                    "Group {} has no input port {:?}.",
                    group.id, connection.input
                )));
            };
            input_sources
                .entry(port.node.clone())
                .or_default()
                .push(connection.clone());
        }
        // External targets leaving the group by port.
        let mut output_targets: BTreeMap<String, Vec<GraphConnection>> = BTreeMap::new();
        for connection in document
            .connections
            .iter()
            .filter(|connection| connection.from == group.id)
        {
            let Some(port) = payload.outputs.iter().find(|port| port.port == connection.out) else {
                return Err(SessionError::InvalidInput(format!(
                    "Group {} has no output port {:?}.",
                    group.id, connection.out
                )));
            };
            output_targets
                .entry(port.node.clone())
                .or_default()
                .push(connection.clone());
        }

        let input_markers: BTreeSet<&str> = payload.inputs.iter().map(|port| port.node.as_str()).collect();
        let output_markers: BTreeSet<&str> = payload.outputs.iter().map(|port| port.node.as_str()).collect();

        for node in child.nodes {
            if input_markers.contains(node.id.as_str()) || output_markers.contains(node.id.as_str()) {
                continue;
            }
            let mut node = node;
            node.id = flat_id(&group.id, &node.id);
            if !known.insert(node.id.clone()) {
                return Err(SessionError::DuplicateNode(node.id));
            }
            out.nodes.push(node);
        }

        for connection in child.connections {
            let from_is_input = input_markers.contains(connection.from.as_str());
            let to_is_output = output_markers.contains(connection.to.as_str());
            match (from_is_input, to_is_output) {
                (true, true) => {
                    // A fresh Group starts as an input-to-output passthrough.
                    if let (Some(sources), Some(targets)) = (
                        input_sources.get(&connection.from),
                        output_targets.get(&connection.to),
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
                    if let Some(sources) = input_sources.get(&connection.from) {
                        for source in sources {
                            out.connections.push(GraphConnection {
                                from: source.from.clone(),
                                to: flat_id(&group.id, &connection.to),
                                input: connection.input.clone(),
                                out: source.out.clone(),
                            });
                        }
                    }
                }
                (false, true) => {
                    if let Some(targets) = output_targets.get(&connection.to) {
                        for target in targets {
                            out.connections.push(GraphConnection {
                                from: flat_id(&group.id, &connection.from),
                                to: target.to.clone(),
                                input: target.input.clone(),
                                out: connection.out.clone(),
                            });
                        }
                    }
                }
                (false, false) => {
                    out.connections.push(GraphConnection {
                        from: flat_id(&group.id, &connection.from),
                        to: flat_id(&group.id, &connection.to),
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
        GraphNode {
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
            nodes: vec![node("box", "primitive", json!({})), node("g", "group", outer)],
            connections: vec![connection("box", "g", "shape", "shape")],
            interface: GraphInterface::default(),
        };
        let flat = expand_groups(&document).expect("expand");
        assert!(flat.nodes.iter().any(|node| node.id == "g::inner::move"));
        assert!(flat.nodes.iter().all(|node| node.node_type != "group"));
    }
}
