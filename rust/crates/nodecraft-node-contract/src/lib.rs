//! Node-descriptor contract.
//!
//! One static description per node type: its ports, value kinds, output and
//! evaluation capabilities. Evaluation can migrate to descriptor-driven
//! dispatch without changing the graph or serialized documents.

use nodecraft_protocol::{Connection, Node};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    Shape,
    Part,
    /// A solid/Part value, but not a sketch.
    Geometry,
    Sketch,
    /// Accepts any geometry kind; used by ports that support both sketch and
    /// solid inputs (Extrude is the current example).
    Any,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSpec {
    pub name: &'static str,
    pub kind: ValueKind,
    /// Whether the evaluator expects this port to be connected for a meaningful
    /// result. Documents may still contain an unconnected port temporarily.
    pub required: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeDescriptor {
    pub node_type: &'static str,
    pub inputs: &'static [PortSpec],
    pub output: ValueKind,
    pub exact: bool,
    pub mesh: bool,
    /// Input port whose result is shown as a ghost under this node's result
    /// while the node is selected. This is a display capability, independent of
    /// topology selection.
    pub preview_input: Option<&'static str>,
    /// Value field that must be non-empty for the preview to apply. Fillet and
    /// Chamfer use `edges`: an empty edge selection has no feature to ghost.
    pub preview_requires_value: Option<&'static str>,
}

const GEOMETRY: &[PortSpec] = &[PortSpec { name: "shape", kind: ValueKind::Geometry, required: true }];
const NO_PORTS: &[PortSpec] = &[];

macro_rules! descriptor {
    ($name:literal, $inputs:expr, $output:expr, $exact:literal, $mesh:literal) => {
        NodeDescriptor {
            node_type: $name,
            inputs: $inputs,
            output: $output,
            exact: $exact,
            mesh: $mesh,
            preview_input: None,
            preview_requires_value: None,
        }
    };
}

macro_rules! descriptor_with_preview {
    (
        $name:literal,
        $inputs:expr,
        $output:expr,
        $exact:literal,
        $mesh:literal,
        $preview:literal,
        $requires:literal
    ) => {
        NodeDescriptor {
            node_type: $name,
            inputs: $inputs,
            output: $output,
            exact: $exact,
            mesh: $mesh,
            preview_input: Some($preview),
            preview_requires_value: Some($requires),
        }
    };
}

static DESCRIPTORS: &[NodeDescriptor] = &[
    descriptor!("primitive", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("solid", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("stepimporter", NO_PORTS, ValueKind::Shape, true, true),
    descriptor!("translate", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("mirror", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("combine", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!(
        "boolean",
        &[
            PortSpec { name: "a", kind: ValueKind::Geometry, required: true },
            PortSpec { name: "b", kind: ValueKind::Geometry, required: true },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "fuse",
        &[
            PortSpec { name: "a", kind: ValueKind::Geometry, required: true },
            PortSpec { name: "b", kind: ValueKind::Geometry, required: true },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor_with_preview!("chamfer", GEOMETRY, ValueKind::Geometry, true, true, "shape", "edges"),
    descriptor_with_preview!("fillet", GEOMETRY, ValueKind::Geometry, true, true, "shape", "edges"),
    descriptor!("look", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("shell", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("offsetfaces", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!(
        "extrude",
        &[
            PortSpec { name: "profile", kind: ValueKind::Any, required: true },
            // Older documents connected Extrude through a generic shape port.
            PortSpec { name: "shape", kind: ValueKind::Any, required: false },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "loft",
        &[
            PortSpec { name: "bottom", kind: ValueKind::Sketch, required: true },
            PortSpec { name: "top", kind: ValueKind::Sketch, required: false },
        ],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!(
        "surfaceloft",
        &[PortSpec { name: "sections", kind: ValueKind::Sketch, required: true }],
        ValueKind::Shape,
        true,
        true
    ),
    descriptor!("scatter", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("sketch", NO_PORTS, ValueKind::Sketch, true, true),
    descriptor!("spline", NO_PORTS, ValueKind::Sketch, true, true),
    descriptor!("exporter", GEOMETRY, ValueKind::Empty, true, false),
    descriptor!("linkedpart", NO_PORTS, ValueKind::Empty, false, false),
    descriptor!("group", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("groupinput", NO_PORTS, ValueKind::Geometry, true, true),
    descriptor!("groupoutput", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("time", GEOMETRY, ValueKind::Geometry, true, true),
    descriptor!("sceneTransform", GEOMETRY, ValueKind::Geometry, true, true),
];

#[must_use]
pub fn descriptor(node_type: &str) -> Option<&'static NodeDescriptor> {
    DESCRIPTORS.iter().find(|descriptor| descriptor.node_type == node_type)
}

/// The input port whose mesh is ghosted under a selected node's result.
#[must_use]
pub fn preview_input(node_type: &str) -> Option<&'static str> {
    descriptor(node_type)?.preview_input
}

/// The preview input port, if the node currently has the values that make its
/// preview meaningful. An empty edge selection means there is no feature to
/// ghost, regardless of the descriptor capability.
#[must_use]
pub fn preview_input_if_active(node_type: &str, values: &serde_json::Value) -> Option<&'static str> {
    let descriptor = descriptor(node_type)?;
    let input = descriptor.preview_input?;
    if let Some(field) = descriptor.preview_requires_value {
        let active = values
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty());
        if !active {
            return None;
        }
    }
    Some(input)
}

pub fn descriptors() -> impl Iterator<Item = &'static NodeDescriptor> {
    DESCRIPTORS.iter()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractError {
    UnknownNodeType { node: String, node_type: String },
    UnknownInput { node: String, node_type: String, input: String },
    MissingSource { node: String, source: String },
    MissingOutput { node: String, source: String },
    WrongKind {
        node: String,
        input: String,
        expected: ValueKind,
        actual: ValueKind,
    },
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNodeType { node, node_type } => {
                write!(f, "node {node:?} has unknown type {node_type:?}")
            }
            Self::UnknownInput { node, node_type, input } => {
                write!(f, "node {node:?} ({node_type}) has no input port {input:?}")
            }
            Self::MissingSource { node, source } => {
                write!(f, "connection to {node:?} references missing source {source:?}")
            }
            Self::MissingOutput { node, source } => {
                write!(f, "connection source {source:?} has no geometry output")
            }
            Self::WrongKind { node, input, expected, actual } => {
                write!(f, "connection to {node:?}.{input} expects {expected:?}, source is {actual:?}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

/// Validate port names and value kinds. Connectivity requirements are checked by
/// the evaluator, not here, because partial documents are valid while editing.
pub fn validate_connections(
    document_nodes: &[Node],
    document_connections: &[Connection],
) -> Result<(), ContractError> {
    let mut nodes = std::collections::BTreeMap::new();
    for node in document_nodes {
        if descriptor(&node.node_type).is_none() {
            return Err(ContractError::UnknownNodeType {
                node: node.id.clone(),
                node_type: node.node_type.clone(),
            });
        }
        nodes.insert(node.id.as_str(), node);
    }
    for connection in document_connections {
        validate_connection(connection, &nodes)?;
    }
    Ok(())
}

fn accepts(port: ValueKind, output: ValueKind) -> bool {
    match port {
        ValueKind::Any => output != ValueKind::Empty,
        ValueKind::Geometry => matches!(output, ValueKind::Shape | ValueKind::Part | ValueKind::Geometry),
        _ => port == output,
    }
}

type NodeMap<'a> = std::collections::BTreeMap<&'a str, &'a Node>;

fn validate_connection(
    connection: &Connection,
    nodes: &NodeMap<'_>,
) -> Result<(), ContractError> {
    let Some(target) = nodes.get(connection.to.as_str()) else {
        return Ok(());
    };
    let target_descriptor = descriptor(&target.node_type).expect("target validated");
    let Some(port) = target_descriptor
        .inputs
        .iter()
        .find(|port| port.name == connection.input)
        .copied()
    else {
        return Err(ContractError::UnknownInput {
            node: target.id.clone(),
            node_type: target.node_type.clone(),
            input: connection.input.clone(),
        });
    };
    let Some(source) = nodes.get(connection.from.as_str()) else {
        return Err(ContractError::MissingSource {
            node: target.id.clone(),
            source: connection.from.clone(),
        });
    };
    let source_descriptor = descriptor(&source.node_type).expect("source validated");
    if source_descriptor.output == ValueKind::Empty {
        return Err(ContractError::MissingOutput {
            node: target.id.clone(),
            source: source.id.clone(),
        });
    }
    if !accepts(port.kind, source_descriptor.output) {
        return Err(ContractError::WrongKind {
            node: target.id.clone(),
            input: connection.input.clone(),
            expected: port.kind,
            actual: source_descriptor.output,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_table_is_unique_and_covers_supported_nodes() {
        let mut seen = std::collections::BTreeSet::new();
        for descriptor in descriptors() {
            assert!(seen.insert(descriptor.node_type), "duplicate {}", descriptor.node_type);
            assert!(!descriptor.inputs.iter().any(|port| port.name.is_empty()));
        }
        for node_type in [
            "primitive", "solid", "stepimporter", "translate", "mirror", "combine", "boolean",
            "fuse", "chamfer", "fillet", "look", "shell", "extrude", "loft", "surfaceloft",
            "scatter", "sketch", "spline", "exporter", "linkedpart", "time",
            "sceneTransform",
        ] {
            assert!(descriptor(node_type).is_some(), "missing descriptor for {node_type}");
        }
    }

    fn node(id: &str, node_type: &str) -> Node {
        Node {
            id: id.into(),
            node_type: node_type.into(),
            values: serde_json::json!({}),
            suppressed: false,
            cache: None,
        }
    }

    fn link(from: &str, to: &str, input: &str) -> Connection {
        Connection { from: from.into(), to: to.into(), input: input.into(), out: "shape".into() }
    }

    #[test]
    fn preview_inputs_are_declared_by_feature_descriptors() {
        assert_eq!(preview_input("fillet"), Some("shape"));
        assert_eq!(preview_input("chamfer"), Some("shape"));
        assert_eq!(preview_input("translate"), None);
        assert_eq!(
            preview_input_if_active("fillet", &serde_json::json!({"edges": []})),
            None
        );
        assert_eq!(
            preview_input_if_active("fillet", &serde_json::json!({"edges": [{"id": "edge:0"}]})),
            Some("shape")
        );
        // Every declared preview port must exist on the same descriptor.
        for descriptor in descriptors() {
            let Some(port) = descriptor.preview_input else { continue };
            assert!(
                descriptor.inputs.iter().any(|input| input.name == port),
                "{} previews missing input {port}",
                descriptor.node_type
            );
        }
    }

    #[test]
    fn connection_kinds_and_ports_validate() {
        let nodes = vec![
            node("sketch", "sketch"),
            node("extrude", "extrude"),
            node("box", "primitive"),
            node("boolean", "boolean"),
        ];
        let connections = vec![
            link("sketch", "extrude", "profile"),
            link("box", "boolean", "a"),
            link("extrude", "boolean", "b"),
        ];
        assert_eq!(validate_connections(&nodes, &connections), Ok(()));
    }

    #[test]
    fn geometry_ports_accept_shape_and_part_but_not_sketch() {
        assert!(accepts(ValueKind::Geometry, ValueKind::Shape));
        assert!(accepts(ValueKind::Geometry, ValueKind::Part));
        assert!(!accepts(ValueKind::Geometry, ValueKind::Sketch));
        assert!(accepts(ValueKind::Any, ValueKind::Sketch));
        assert!(!accepts(ValueKind::Any, ValueKind::Empty));
    }

    #[test]
    fn sketch_cannot_connect_to_shape_port() {
        let nodes = vec![node("s", "sketch"), node("t", "translate")];
        let connections = vec![link("s", "t", "shape")];
        assert!(matches!(
            validate_connections(&nodes, &connections),
            Err(ContractError::WrongKind { .. })
        ));
    }
}
