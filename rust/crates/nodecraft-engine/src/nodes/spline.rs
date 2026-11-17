//! Spline: a control/fit-point curve generator.
//!
//! The point list is data; `degree`, `control` and `periodic` come only from the
//! node contract. The node publishes the solved sketch for the overlay and
//! downstream features, and its Part carries the same sketch value.

use super::{BuiltinNode, NodeView, OutputKind, PartEval};
use crate::{Part, RequestIndex, SketchOutput};

/// Registered for the `spline` contract id.
pub(crate) struct Spline;

impl BuiltinNode for Spline {
    fn node_types(&self) -> &'static [&'static str] {
        &["spline"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let sketch = crate::spline_sketch_from_values(node.values)?;
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

    fn suppression_source(
        &self,
        _node: NodeView<'_>,
        _index: &RequestIndex<'_>,
    ) -> Option<String> {
        // A spline is a generator: suppressing it turns it off.
        None
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        _inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // A spline contributes no display body; the curve is an overlay/mesh
        // instance appended after the bodies.
        Some(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::Request;

    fn points() -> serde_json::Value {
        serde_json::json!([
            {"x": -40.0, "y": 0.0},
            {"x": -15.0, "y": 25.0},
            {"x": 15.0, "y": -25.0},
            {"x": 40.0, "y": 0.0}
        ])
    }

    fn request(control: Option<bool>) -> Request {
        let mut values = serde_json::json!({"points": points(), "degree": 3, "periodic": false});
        if let Some(control) = control {
            values["control"] = serde_json::json!(control);
        }
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["spline"], "meshTargets": [],
            "nodes": [
                {"id": "spline", "type": "spline", "values": values, "cache": {"key": "spline"}}
            ],
            "connections": []
        }))
        .expect("spline request")
    }

    fn spline_curve(values: &serde_json::Value) -> (bool, u8, bool) {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["spline"], "meshTargets": [],
            "nodes": [
                {"id": "spline", "type": "spline", "values": values, "cache": {"key": "spline"}}
            ],
            "connections": []
        }))
        .expect("spline request");
        let mut eval = crate::Eval::new(&request, None);
        let part = eval.evaluate_part("spline").expect("spline part");
        assert_eq!(part.sketches.len(), 1);
        assert!(eval.sketches.contains_key("spline"), "spline was not published");
        let curve = &part.sketches[0].output.sketch.curves[0].kind;
        match curve {
            crate::sketch::CurveKind::Spline { control, degree, periodic, .. } => {
                (*control, *degree, *periodic)
            }
            other => panic!("spline node made {other:?}"),
        }
    }

    /// Removing the registry entry makes this fail: the engine no longer knows
    /// the `spline` type. The test is the entry's fail proof.
    #[test]
    fn registry_routes_spline() {
        let builtin = crate::nodes::find("spline").expect("spline registered");
        assert_eq!(builtin.node_types(), &["spline"]);
        assert_eq!(builtin.output_kind(), OutputKind::Sketch);
        let values = serde_json::json!({});
        let view = NodeView::new("s", "spline", &values, false);
        assert_eq!(crate::nodes::output_kind(view), OutputKind::Sketch);
        assert_eq!(builtin.body_layout(view, &[]), Some(Vec::new()));
    }

    /// `control` and `degree` are read through the contract: an explicit switch
    /// makes a control-point B-spline, and an absent switch uses the contract
    /// default (fit points).
    #[test]
    fn spline_control_points_come_from_the_contract() {
        assert_eq!(
            spline_curve(&serde_json::json!({
                "points": points(), "degree": 3, "control": true, "periodic": false
            })),
            (true, 3, false),
        );
        assert_eq!(
            spline_curve(&serde_json::json!({"points": points()})),
            (false, 3, false),
            "the contract defaults are fit points and degree 3",
        );
    }

    /// The registered node still evaluates a real curve and publishes it for a
    /// downstream section consumer.
    #[test]
    fn spline_evaluates_through_the_registry() {
        let request = request(Some(false));
        let mut eval = crate::Eval::new(&request, None);
        let part = eval.evaluate_part("spline").expect("spline part");
        assert_eq!(part.sketches.len(), 1);
        assert!(!part.sketches[0].output.sketch.curves.is_empty());
    }
}
