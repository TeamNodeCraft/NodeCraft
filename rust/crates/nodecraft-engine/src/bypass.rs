//! Suppression and error bypass semantics.
//!
//! A suppressed or failed feature must pass through the nearest meaningful
//! input instead of blacking out the graph. Generators remain terminal.

use nodecraft_protocol::{Node, Request};

/// The input a suppressed node passes through, if suppression means
/// passthrough for this node type. Generators return `None` and are turned off.
pub(crate) fn suppression_source(node: &Node, connections: &[nodecraft_protocol::Connection]) -> Option<String> {
    match node.node_type.as_str() {
        "solid" | "primitive" | "sketch" | "spline" | "stepimporter" | "scatter" | "loft"
        | "surfaceloft" | "combine" | "exporter" | "linkedpart" | "time" => None,
        "boolean" | "fuse" => connections
            .iter()
            .find(|link| link.to == node.id && link.input == "a")
            .or_else(|| connections.iter().find(|link| link.to == node.id && link.input == "b"))
            .map(|link| link.from.clone()),
        "translate" | "mirror" | "chamfer" | "fillet" | "shell" | "extrude" | "look" => connections
            .iter()
            .find(|link| link.to == node.id && (link.input == "shape" || link.input == "profile"))
            .map(|link| link.from.clone()),
        _ => None,
    }
}

/// The input an errored node passes through.
///
/// A failing feature must not black out the whole graph: its output becomes the
/// nearest meaningful input, the same way a suppressed node behaves. Generators
/// without an input stay terminal and report only their own error.
pub(crate) fn bypass_source(node: &Node, connections: &[nodecraft_protocol::Connection]) -> Option<String> {
    suppression_source(node, connections).or_else(|| {
        connections
            .iter()
            .find(|link| link.to == node.id && (link.input == "shape" || link.input == "profile"))
            .map(|link| link.from.clone())
    })
}

pub(crate) fn effective_output_suppressed(request: &Request, start: &str) -> bool {
    let mut current = start.to_string();
    for _ in 0..64 {
        let Some(node) = request.nodes.iter().find(|item| item.id == current) else { return true };
        if node.suppressed {
            if let Some(source) = suppression_source(node, &request.connections) {
                current = source;
                continue;
            }
            return true;
        }
        if node.node_type == "translate" {
            let Some(link) = request.connections.iter().find(|link| link.to == current && link.input == "shape") else { return true };
            current = link.from.clone();
            continue;
        }
        return false;
    }
    true
}

