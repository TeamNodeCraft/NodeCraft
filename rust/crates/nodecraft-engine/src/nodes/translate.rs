//! Translate: place an exact Part with one XYZ translation and Euler XYZ
//! rotation.
//!
//! The placement maths stays shared: `crate::translate_placement` reads the six
//! contract fields and `Part::translated` records the placement on each body or
//! sketch instance. This module owns only the node-shaped parts: the shape
//! input, cost, bypass, layout and the transform role the session's preview
//! chain reads.

use super::{BuiltinNode, NodeView, PartEval, TransformRole};
use crate::{Part, RequestIndex};

/// Registered for the `translate` contract id.
pub(crate) struct Translate;

impl BuiltinNode for Translate {
    fn node_types(&self) -> &'static [&'static str] {
        &["translate"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Part translate has no input".into());
        };
        let (offset, rotation) = crate::translate_placement(node.values);
        Ok(source.as_ref().translated(offset, rotation))
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        index
            .first_incoming_any(&node.id, &["shape", "profile"])
            .map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        super::input_layout(inputs, "shape").map(<[String]>::to_vec)
    }

    fn transform_role(&self) -> Option<TransformRole> {
        Some(TransformRole::Translate)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["move"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 40, "width": 10, "height": 20}},
                {"id": "turn", "type": "translate", "cache": {"key": "turn"},
                 "values": {"x": 1.0, "rx": 90.0, "rz": 90.0}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 5.0, "ry": 90.0}}
            ],
            "connections": [
                {"from": "box", "to": "turn", "in": "shape", "out": "shape"},
                {"from": "turn", "to": "move", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("translate request")
    }

    fn bounds(part: &nodecraft_exact::ExactPart) -> ([f64; 3], [f64; 3]) {
        let brep = part.realized_breps().expect("realized body").pop().expect("one body");
        let record = nodecraft_step::record_of_brep(&brep).expect("body record");
        let (min, max) = nodecraft_step::record_bounds(&record).expect("body bounds");
        ([min[0], min[1], min[2]], [max[0], max[1], max[2]])
    }

    /// Removing the registry entry makes the engine reject `translate`, so this
    /// test is the entry's fail proof. It also pins the preview-chain role.
    #[test]
    fn registry_routes_translate_and_its_transform_role() {
        let builtin = crate::nodes::find("translate").expect("translate registered");
        assert_eq!(builtin.node_types(), &["translate"]);
        assert_eq!(builtin.transform_role(), Some(crate::nodes::TransformRole::Translate));
        let values = serde_json::json!({});
        assert!(!builtin.repeats_input_bodies(crate::nodes::NodeView::new(
            "move", "translate", &values, false,
        )));
    }

    /// Two chained placements compose in Part order. The first rotates a
    /// 40x10x20 box by Euler XYZ 90/0/90 (Rz * Ry * Rx) and moves it to x=1;
    /// the second rotates that result 90 about Y and moves it to x=5.
    #[test]
    fn chained_placements_compose_in_euler_xyz_degrees() {
        let output = crate::evaluate_exact(request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("move").expect("translated exact part");
        assert_eq!(part.bodies.len(), 1);
        let (min, max) = bounds(part);
        let expected_min = [0.0, -20.0, -11.0];
        let expected_max = [10.0, 20.0, 9.0];
        for axis in 0..3 {
            assert!(
                (min[axis] - expected_min[axis]).abs() < 1.0e-6,
                "min {min:?} != {expected_min:?}",
            );
            assert!(
                (max[axis] - expected_max[axis]).abs() < 1.0e-6,
                "max {max:?} != {expected_max:?}",
            );
        }
    }

    /// The node keeps the source body layout; it places bodies, it does not
    /// repeat or replace them.
    #[test]
    fn body_layout_is_the_source_layout() {
        let builtin = crate::nodes::find("translate").expect("translate registered");
        let values = serde_json::json!({"x": 4.0});
        let view = crate::nodes::NodeView::new("move", "translate", &values, false);
        assert_eq!(
            builtin.body_layout(view, &[("shape", vec!["box".to_owned(), "cap".to_owned()])]),
            Some(vec!["box".to_owned(), "cap".to_owned()]),
        );
    }
}
