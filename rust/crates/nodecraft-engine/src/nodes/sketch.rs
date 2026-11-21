//! Sketch: a 2D sketch document solved into a typed [`SketchOutput`].
//!
//! The sketch is a generator: it publishes its solved output for the overlay
//! and downstream features, and its Part carries the same sketch value. It has
//! no 3D display body (the 2D overlay owns it), so its body layout is empty.

use super::{BuiltinNode, NodeView, OutputKind, PartEval};
use crate::{Part, RequestIndex, SketchOutput};

/// Registered for the `sketch` contract id.
pub(crate) struct SketchNode;

impl BuiltinNode for SketchNode {
    fn node_types(&self) -> &'static [&'static str] {
        &["sketch"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let sketch: crate::sketch::Sketch = serde_json::from_value(
            node.values.get("sketch").cloned().unwrap_or(serde_json::Value::Null),
        )
        .map_err(|error| format!("Sketch values: {error}"))?;
        let output = std::sync::Arc::new(SketchOutput::new(sketch));
        ctx.publish_sketch(node, output.clone());
        Ok(Part::from_sketch(output))
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn output_kind(&self) -> OutputKind {
        OutputKind::Sketch
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // A sketch is a generator: suppressing it turns it off.
        None
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        _inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // Sketches contribute no display bodies; the overlay draws the curves.
        Some(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::Request;

    fn sketch_request() -> Request {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).expect("point a");
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).expect("point b");
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).expect("point c");
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).expect("point d");
        sketch.add_line_pts(a, b, None).expect("line ab");
        sketch.add_line_pts(b, c, None).expect("line bc");
        sketch.add_line_pts(c, d, None).expect("line cd");
        sketch.add_line_pts(d, a, None).expect("line da");
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["move"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "move", "type": "translate", "values": {"z": 7.0}, "cache": {"key": "move"}}
            ],
            "connections": [{"from": "sketch", "to": "move", "in": "shape", "out": "sketch"}]
        }))
        .expect("sketch request")
    }

    /// Removing the registry entry makes this fail: the engine no longer knows
    /// the `sketch` type. The test is the entry's fail proof.
    #[test]
    fn registry_routes_sketch() {
        let builtin = crate::nodes::find("sketch").expect("sketch registered");
        assert_eq!(builtin.node_types(), &["sketch"]);
        assert_eq!(builtin.output_kind(), OutputKind::Sketch);
        let values = serde_json::json!({});
        let view = NodeView::new("s", "sketch", &values, false);
        assert_eq!(crate::nodes::output_kind(view), OutputKind::Sketch);
        assert_eq!(
            builtin.body_layout(view, &[]),
            Some(Vec::new()),
            "a sketch contributes no display body",
        );
    }

    /// Evaluating the registered node publishes the typed sketch for the
    /// overlay, and a downstream placement sees the same realised output.
    #[test]
    fn sketch_publishes_and_a_downstream_overlay_resolves_it() {
        let request = sketch_request();
        let mut eval = crate::Eval::new(&request, None);
        let part = eval.evaluate_part("sketch").expect("sketch part");
        assert_eq!(part.sketches.len(), 1);
        assert!(eval.sketches.contains_key("sketch"), "sketch was not published");

        let moved = eval.evaluate_part("move").expect("moved sketch part");
        assert_eq!(moved.sketches.len(), 1);
        let realised = moved.sketches[0].realized();
        let origin = realised.point(0.0, 0.0);
        assert!((origin[2] - 7.0).abs() < 1.0e-9, "overlay lost the translation: {origin:?}");
    }
}
