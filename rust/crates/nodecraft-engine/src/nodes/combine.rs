//! Combine: concatenate every connected Part into one Part.
//!
//! Combine is a Part collection, not a Boolean: bodies and sketches keep their
//! order, placement chains, names and colours. A failed operand does not hide
//! the ones that evaluated, but the node still reports the failure.

use super::{BuiltinNode, NodeView, PartEval};
use crate::{Part, RequestIndex};

/// Registered for the `combine` contract id.
pub(crate) struct Combine;

impl BuiltinNode for Combine {
    fn node_types(&self) -> &'static [&'static str] {
        &["combine"]
    }

    fn evaluate(&self, _node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let mut bodies = Vec::new();
        let mut sketches = Vec::new();
        let mut first_error: Option<String> = None;
        for operand in ctx.input_parts("shape") {
            match operand {
                Ok(part) => {
                    bodies.extend(part.bodies.clone());
                    sketches.extend(part.sketches.clone());
                }
                // A failed import inside a Combine must not poison the other,
                // successfully imported operands.
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = &first_error {
            ctx.record_error(format!("Combine has failed operands: {error}"));
        }
        if bodies.is_empty() && sketches.is_empty() {
            return Err(first_error
                .map(|error| format!("Combine has failed operands: {error}"))
                .unwrap_or_else(|| "Combine Part has no inputs".into()));
        }
        Ok(Part { bodies, sketches })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // Combine is a generator: suppressing it turns it off.
        None
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // The Part is the flattened concatenation of every `shape` input in
        // connection order, so the display body order is the same concatenation.
        let mut layout = Vec::new();
        for (port, input) in inputs {
            if *port == "shape" {
                layout.extend(input.iter().cloned());
            }
        }
        Some(layout)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn two_operands() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["part"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "cache": {"key": "a"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "b", "type": "primitive", "cache": {"key": "b"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 20.0}},
                {"id": "part", "type": "combine", "cache": {"key": "part"}, "values": {}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "part", "in": "shape"},
                {"from": "b", "out": "shape", "to": "move", "in": "shape"},
                {"from": "move", "out": "shape", "to": "part", "in": "shape"}
            ]
        }))
        .expect("combine request")
    }

    /// Removing the registry entry makes the engine reject `combine`, so this
    /// test is the entry's fail proof.
    #[test]
    fn registry_routes_combine() {
        let builtin = crate::nodes::find("combine").expect("combine registered");
        assert_eq!(builtin.node_types(), &["combine"]);
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("part", "combine", &values, false);
        assert_eq!(
            builtin.body_layout(
                view,
                &[
                    ("shape", vec!["a".to_owned()]),
                    ("shape", vec!["b".to_owned(), "c".to_owned()]),
                ],
            ),
            Some(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]),
        );
        assert_eq!(builtin.body_layout(view, &[]), Some(Vec::new()));
    }

    /// Multi-input body layout is the flattened Part order, including each
    /// operand's own placement and body count.
    #[test]
    fn multi_input_layout_flattens_every_port_in_connection_order() {
        let output = crate::evaluate_exact(two_operands(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("part").expect("combine exact part");
        assert_eq!(part.bodies.len(), 2);
        let builtin = crate::nodes::find("combine").expect("combine registered");
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("part", "combine", &values, false);
        assert_eq!(
            builtin.body_layout(
                view,
                &[("shape", vec!["a".to_owned()]), ("shape", vec!["b".to_owned()]),],
            ),
            Some(vec!["a".to_owned(), "b".to_owned()]),
        );
    }

    /// One failed operand is reported and the successful bodies survive, which
    /// is the Combine error tolerance that predates the registry port.
    #[test]
    fn a_failed_operand_does_not_hide_the_successful_ones() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["part"], "meshTargets": [],
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "missing", "type": "stepimporter", "values": {"path": "/nonexistent/does-not-exist.step"}},
                {"id": "part", "type": "combine", "values": {}}
            ],
            "connections": [
                {"from": "box", "out": "shape", "to": "part", "in": "shape"},
                {"from": "missing", "out": "shape", "to": "part", "in": "shape"}
            ]
        }))
        .expect("combine request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let part = output.exact_parts.get("part").expect("partial combine part");
        assert_eq!(part.bodies.len(), 1, "the valid box must survive the failed import");
        assert!(
            output
                .response
                .node_errors
                .get("part")
                .is_some_and(|error| error.contains("Combine has failed operands")),
            "{:?}",
            output.response.node_errors,
        );
        assert!(output.response.node_errors.contains_key("missing"));
    }
}
