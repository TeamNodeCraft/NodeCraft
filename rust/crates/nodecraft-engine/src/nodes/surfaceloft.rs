//! Surface Loft: one surface through two or more sketch/spline sections.
//!
//! Sections are consumed in connection order with their source ids, which is
//! also what a per-section error names. A body section is refused: a surface
//! loft needs profiles/curves, not solids.

use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex, contract_number};

use super::{BuiltinNode, NodeView, PartEval, section_sketch};

/// Registered for the `surfaceloft` contract id.
pub(crate) struct SurfaceLoft;

impl BuiltinNode for SurfaceLoft {
    fn node_types(&self) -> &'static [&'static str] {
        &["surfaceloft"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let sections = ctx.input_sections("sections");
        if sections.len() < 2 {
            return Err("Surface Loft needs at least two section connections".into());
        }
        let mut outputs = Vec::with_capacity(sections.len());
        for section in sections {
            let part = section.part?;
            let sketch = section_sketch(&part).ok_or_else(|| {
                format!("Surface Loft section {} must be a sketch or spline curve", section.from)
            })?;
            outputs.push(sketch.as_ref().clone());
        }
        let ruled = contract_number("surfaceloft", &node.values, "ruled") >= 0.5;
        let brep = ctx
            .execute_warned(ExactNodeTask::SurfaceLoft { sections: outputs, ruled })?
            .output
            .into_brep()
            .ok_or_else(|| "Surface Loft task returned no shape".to_owned())?;
        Ok(Part::from_brep(brep))
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        12
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // Surface Loft is a generator over its sections: suppressing it turns
        // it off.
        None
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        _inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        Some(vec![node.id.to_owned()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::Request;

    fn square_sketch() -> crate::sketch::Sketch {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).expect("a");
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).expect("b");
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).expect("c");
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).expect("d");
        sketch.add_line_pts(a, b, None).expect("ab");
        sketch.add_line_pts(b, c, None).expect("bc");
        sketch.add_line_pts(c, d, None).expect("cd");
        sketch.add_line_pts(d, a, None).expect("da");
        sketch
    }

    fn request_with(nodes: serde_json::Value, connections: serde_json::Value) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["loft"], "meshTargets": [],
            "nodes": nodes,
            "connections": connections
        }))
        .expect("surface loft request")
    }

    /// Removing the registry entry makes this fail: the engine no longer knows
    /// the `surfaceloft` type. The test is the entry's fail proof.
    #[test]
    fn registry_routes_surface_loft() {
        let builtin = crate::nodes::find("surfaceloft").expect("surfaceloft registered");
        assert_eq!(builtin.node_types(), &["surfaceloft"]);
        let values = serde_json::json!({});
        let view = NodeView::new("l", "surfaceloft", &values, false);
        assert_eq!(builtin.body_layout(view, &[]), Some(vec!["l".to_owned()]));
    }

    /// Two or more sections are required; the existing error names the rule.
    #[test]
    fn surface_loft_requires_at_least_two_sections() {
        let request = request_with(
            serde_json::json!([
                {"id": "bottom", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "bottom"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1.0}, "cache": {"key": "loft"}}
            ]),
            serde_json::json!([
                {"from": "bottom", "to": "loft", "in": "sections", "out": "sketch"}
            ]),
        );
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let error = output.response.node_errors.get("loft").expect("section count error");
        assert!(error.contains("at least two section connections"), "{error}");
    }

    /// Sections are consumed in connection order, and a bad section names its
    /// own upstream id rather than the target's. Two body sections make the
    /// order observable: the first one must be the one named.
    #[test]
    fn surface_loft_sections_keep_order_and_labels() {
        let request = request_with(
            serde_json::json!([
                {"id": "body", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "body"}},
                {"id": "body2", "type": "primitive", "values": {"primitiveType": "box", "length": 12.0, "width": 12.0, "height": 12.0}, "cache": {"key": "body2"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1.0}, "cache": {"key": "loft"}}
            ]),
            serde_json::json!([
                {"from": "body", "to": "loft", "in": "sections", "out": "shape"},
                {"from": "body2", "to": "loft", "in": "sections", "out": "shape"}
            ]),
        );
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let error = output.response.node_errors.get("loft").expect("body section error");
        assert!(
            error.contains("Surface Loft section body must be a sketch or spline curve"),
            "the first connected section keeps its source label: {error}",
        );
    }

    /// A successful multi-section surface contains faces and spans the placed
    /// sections.
    #[test]
    fn surface_loft_builds_a_surface_in_section_order() {
        let request = request_with(
            serde_json::json!([
                {"id": "bottom", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "bottom"}},
                {"id": "top-at", "type": "translate", "values": {"z": 10.0}, "cache": {"key": "top-at"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1.0}, "cache": {"key": "loft"}}
            ]),
            serde_json::json!([
                {"from": "bottom", "to": "top-at", "in": "shape", "out": "sketch"},
                {"from": "bottom", "to": "loft", "in": "sections", "out": "sketch"},
                {"from": "top-at", "to": "loft", "in": "sections", "out": "shape"}
            ]),
        );
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("loft").expect("surface loft part");
        assert_eq!(part.bodies.len(), 1);
        assert!(!part.bodies[0].brep.face_list().is_empty(), "surface has no faces");
        let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
        for vertex in part.bodies[0].brep.vertex_list() {
            let z = part.bodies[0].brep.vertex(*vertex).expect("vertex").point.z;
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        assert!((min_z - 0.0).abs() < 1.0e-6, "first section z {min_z}");
        assert!((max_z - 10.0).abs() < 1.0e-6, "second section z {max_z}");
    }
}
