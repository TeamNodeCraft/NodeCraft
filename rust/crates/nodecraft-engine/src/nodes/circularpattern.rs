//! Circular Pattern: repeat an exact Part around an axis through a centre.
//!
//! Every copy shares the source body's plan (`source`/`source_body` are
//! preserved because the instance clones the source Part), so copies of one
//! B-rep are one display plan. Copy 0 is the untouched source body; the drag
//! preview therefore anchors there (a repeater).

use super::{BuiltinNode, NodeView, PartEval};
use crate::{BodyTransform, Part, RequestIndex};

/// Registered for the `circularpattern` contract id.
pub(crate) struct CircularPattern;

impl BuiltinNode for CircularPattern {
    fn node_types(&self) -> &'static [&'static str] {
        &["circularpattern"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Circular Pattern has no source".into());
        };
        if source.bodies.is_empty() && source.sketches.is_empty() {
            return Err("Circular Pattern source Part is empty".into());
        }
        let count = count(node);
        let angle = crate::contract_number("circularpattern", &node.values, "angle");
        let axis = crate::contract_text("circularpattern", &node.values, "axis");
        let center = [
            crate::contract_number("circularpattern", &node.values, "centerX"),
            crate::contract_number("circularpattern", &node.values, "centerY"),
            crate::contract_number("circularpattern", &node.values, "centerZ"),
        ];
        let negative_center = [-center[0], -center[1], -center[2]];
        let step = circular_pattern_step(count, angle);
        let mut bodies = Vec::new();
        let mut sketches = Vec::new();
        for index in 0..count {
            let mut instance_bodies = source.bodies.clone();
            let mut instance_sketches = source.sketches.clone();
            if index > 0 {
                let rotation = circular_pattern_rotation(&axis, step * index as f64);
                // Rotate about the line through `center`, using the one
                // placement vocabulary: translate the centre to the origin,
                // rotate, then translate it back. Copy 0 above stays untouched,
                // so its instance is the identity.
                for body in &mut instance_bodies {
                    body.push_transform(BodyTransform::Place {
                        offset: negative_center,
                        rotation: [0.0; 3],
                    });
                    body.push_transform(BodyTransform::Place { offset: [0.0; 3], rotation });
                    body.push_transform(BodyTransform::Place {
                        offset: center,
                        rotation: [0.0; 3],
                    });
                }
                for sketch in &mut instance_sketches {
                    sketch.placement = sketch
                        .placement
                        .then(BodyTransform::Place { offset: negative_center, rotation: [0.0; 3] });
                    sketch.placement =
                        sketch.placement.then(BodyTransform::Place { offset: [0.0; 3], rotation });
                    sketch.placement = sketch
                        .placement
                        .then(BodyTransform::Place { offset: center, rotation: [0.0; 3] });
                }
            }
            bodies.extend(instance_bodies);
            sketches.extend(instance_sketches);
        }
        Ok(Part { bodies, sketches })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        8
    }

    fn mesh_bodies(&self, node: NodeView<'_>) -> usize {
        count(node)
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        index.first_incoming_any(&node.id, &["shape", "profile"]).map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        let input = super::input_layout(inputs, "shape")?;
        let count = count_values(node.values);
        let mut layout = Vec::with_capacity(input.len() * count);
        for _ in 0..count {
            layout.extend(input.iter().cloned());
        }
        Some(layout)
    }

    fn repeats_input_bodies(&self, _node: NodeView<'_>) -> bool {
        true
    }
}

fn count(node: NodeView<'_>) -> usize {
    count_values(node.values)
}

fn count_values(values: &serde_json::Value) -> usize {
    crate::contract_number("circularpattern", values, "count").round().clamp(1.0, 5000.0) as usize
}

/// Angle between adjacent copies. A full turn divides by `count` (so the last
/// copy does not coincide with copy 0); a partial sweep divides by the gaps.
fn circular_pattern_step(count: usize, angle: f64) -> f64 {
    if angle.abs() >= 360.0 {
        angle / count as f64
    } else if count > 1 {
        angle / (count - 1) as f64
    } else {
        0.0
    }
}

/// Euler XYZ degrees for a rotation about `axis` by `degrees`.
fn circular_pattern_rotation(axis: &str, degrees: f64) -> [f64; 3] {
    match axis {
        "x" => [degrees, 0.0, 0.0],
        "y" => [0.0, degrees, 0.0],
        _ => [0.0, 0.0, degrees],
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn pattern_request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"}, "values": {"x": 20}},
                {"id": "pattern", "type": "circularpattern", "cache": {"key": "pattern"},
                 "values": {"count": 4, "axis": "z"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("circular pattern request")
    }

    fn centres(part: &nodecraft_exact::ExactPart) -> Vec<[f64; 3]> {
        part.realized_breps()
            .expect("realized copies")
            .iter()
            .map(|brep| {
                let record = nodecraft_step::record_of_brep(brep).expect("copy record");
                let (min, max) = nodecraft_step::record_bounds(&record).expect("copy bounds");
                [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0, (min[2] + max[2]) / 2.0]
            })
            .collect()
    }

    #[test]
    fn registry_routes_circular_pattern() {
        let builtin = crate::nodes::find("circularpattern").expect("circularpattern registered");
        assert_eq!(builtin.node_types(), &["circularpattern"]);
        assert!(builtin.repeats_input_bodies(crate::nodes::NodeView::new(
            "pattern",
            "circularpattern",
            &serde_json::json!({"count": 4}),
            false,
        )));
    }

    /// Four copies of a box moved 20 mm along X sit on the Z axis through the
    /// origin. This pins the exact instance transforms through the registry.
    #[test]
    fn four_copies_sit_on_the_circle_through_the_registry() {
        let output = crate::evaluate_exact(pattern_request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("pattern exact part");
        assert_eq!(part.bodies.len(), 4, "one body per copy");
        let keys: std::collections::HashSet<_> = part.bodies.iter().map(|body| body.key).collect();
        assert_eq!(keys.len(), 1, "every copy shares its source body plan");
        for (centre, expected) in centres(part).iter().zip([
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-20.0, 0.0, 0.0],
            [0.0, -20.0, 0.0],
        ]) {
            for axis in 0..3 {
                assert!(
                    (centre[axis] - expected[axis]).abs() < 1.0e-5,
                    "copy centre {centre:?} should be {expected:?}",
                );
            }
        }
    }
}
