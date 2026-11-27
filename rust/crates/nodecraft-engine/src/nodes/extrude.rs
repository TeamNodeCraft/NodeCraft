//! Extrude: a sketch profile (one or more) or one selected body face.
//!
//! The profile Part is the source of truth. A Part carrying sketches extrudes
//! every sketch with its placement chain baked in; the body/face path runs only
//! for a Part that has a body and a selected face. Mixed and empty inputs are
//! errors from the shared section validation, never a silent pass-through.

use crate::face_ref::{parse_face_references, resolve_face_references};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex, contract_number, contract_text};

use super::{BuiltinNode, NodeView, PartEval, ProfileInput};

/// Registered for the `extrude` contract id.
pub(crate) struct Extrude;

impl BuiltinNode for Extrude {
    fn node_types(&self) -> &'static [&'static str] {
        &["extrude"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let profile = ctx.input_profile(&["profile", "shape"])?;
        let amount = contract_number("extrude", &node.values, "amount");
        match profile {
            ProfileInput::Sketches(sketches) => {
                let mut bodies = Vec::with_capacity(sketches.len());
                for output in sketches {
                    let brep = ctx
                        .execute_warned(ExactNodeTask::ExtrudeSketch { output, amount })?
                        .output
                        .into_brep()
                        .ok_or_else(|| "sketch extrude task returned no shape".to_owned())?;
                    bodies.push(nodecraft_ops::SolidBody::new(brep));
                }
                Ok(Part::from_bodies(bodies))
            }
            ProfileInput::Body(source) => {
                // The same face-reference resolver Shell and Offset Faces use:
                // a malformed or unknown reference is an error, not a skipped
                // entry. Extrude uses the first resolved face.
                let references = parse_face_references(node.values)?;
                if references.is_empty() {
                    return Err(format!(
                        "Extrude {}: profile input is a body; select a face to extrude or connect a sketch profile.",
                        node.id
                    ));
                }
                let mut solids = source.realized_bodies()?;
                if solids.len() != 1 {
                    return Err(format!(
                        "Extrude requires exactly one solid body, got {}",
                        solids.len()
                    ));
                }
                let profile = solids.pop().ok_or("Extrude source Part is empty")?;
                let face_count = profile.brep().faces().ids().count();
                let openings = resolve_face_references(&references, &[face_count])?;
                let index = *openings[0]
                    .first()
                    .ok_or("Extrude needs a selected face reference (face:N).")?;
                let tool = ctx
                    .execute_warned(ExactNodeTask::Extrude {
                        body: profile.clone(),
                        face_index: index,
                        amount,
                    })?
                    .output
                    .into_brep()
                    .ok_or_else(|| "extrude task returned no shape".to_owned())?;
                let operation = contract_text("extrude", &node.values, "operation");
                if operation == "new" {
                    return Ok(Part::from_brep(tool));
                }
                let result = ctx
                    .execute_warned(ExactNodeTask::ExtrudeFeature {
                        body: profile,
                        face_index: index,
                        amount,
                        operation: operation.into_owned(),
                    })?
                    .output
                    .into_brep()
                    .ok_or_else(|| "Extrude feature task returned no shape".to_owned())?;
                Ok(Part::from_brep(result))
            }
        }
    }

    fn export_terminal(&self) -> bool {
        true
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        6
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        // Extrude passes the first connected shape/profile input through when
        // suppressed, in connection order, exactly as the dispatch arm did.
        index.first_incoming_any(&node.id, &["shape", "profile"]).map(|link| link.from.clone())
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

    fn sketch_extrude_request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [{"from": "sketch", "to": "extrude", "in": "profile", "out": "sketch"}]
        }))
        .expect("sketch extrude request")
    }

    fn body_extrude_request(values: serde_json::Value) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "box"}},
                {"id": "extrude", "type": "extrude", "values": values, "cache": {"key": "extrude"}}
            ],
            "connections": [{"from": "box", "to": "extrude", "in": "profile", "out": "shape"}]
        }))
        .expect("body extrude request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("record");
        nodecraft_step::record_volume(&record).expect("volume")
    }

    /// Removing the registry entry makes this fail: the engine no longer knows
    /// the `extrude` type. The test is the entry's fail proof.
    #[test]
    fn registry_routes_extrude() {
        let builtin = crate::nodes::find("extrude").expect("extrude registered");
        assert_eq!(builtin.node_types(), &["extrude"]);
        let values = serde_json::json!({});
        let view = NodeView::new("x", "extrude", &values, false);
        assert_eq!(builtin.body_layout(view, &[]), Some(vec!["x".to_owned()]),);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "nodes": [{"id": "x", "type": "extrude", "values": {}}],
            "connections": [
                {"from": "box", "to": "x", "in": "shape", "out": "shape"},
                {"from": "sketch", "to": "x", "in": "profile", "out": "sketch"}
            ]
        }))
        .expect("extrude registry request");
        let index = RequestIndex::new(&request);
        assert_eq!(
            builtin.suppression_source(view, &index),
            Some("box".into()),
            "suppression uses the first connected shape/profile input",
        );
    }

    /// A sketch profile extrudes every sketch through the Part path.
    #[test]
    fn extrude_from_sketch_profiles() {
        let output =
            crate::evaluate_exact(sketch_extrude_request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("extrude exact part");
        assert_eq!(part.bodies.len(), 1);
        assert!((volume(&part.bodies[0].brep) - 4_000.0).abs() < 1.0, "20 x 20 x 10");
        exact_bounds(part, [-10.0, -10.0, 0.0], [10.0, 10.0, 10.0]);
    }

    /// A body with a selected face extrudes along that face.
    #[test]
    fn extrude_from_a_body_face() {
        let request = body_extrude_request(serde_json::json!({
            "amount": 5.0, "operation": "new", "faces": [{"id": "face:0"}]
        }));
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("extrude exact part");
        assert_eq!(part.bodies.len(), 1);
        assert!((volume(&part.bodies[0].brep) - 500.0).abs() < 1.0, "10 x 10 x 5");
    }

    /// A body without a face selection is refused with the existing hint.
    #[test]
    fn extrude_from_a_body_without_a_face_is_refused() {
        let output = crate::evaluate_exact(
            body_extrude_request(serde_json::json!({"amount": 5.0})),
            crate::EngineCache::new(),
            None,
        );
        let error = output.response.node_errors.get("extrude").expect("body face error");
        assert!(error.contains("select a face"), "{error}");
    }

    /// A profile Part carrying both a sketch and a body is the shared mixed
    /// input error, not a first-item guess.
    #[test]
    fn extrude_refuses_a_profile_with_both_inputs() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "box"}},
                {"id": "combine", "type": "combine", "values": {}, "cache": {"key": "combine"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "sketch", "to": "combine", "in": "shape", "out": "sketch"},
                {"from": "box", "to": "combine", "in": "shape", "out": "shape"},
                {"from": "combine", "to": "extrude", "in": "profile", "out": "shape"}
            ]
        }))
        .expect("mixed extrude request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let error = output.response.node_errors.get("extrude").expect("mixed input error");
        assert!(error.contains("both bodies and sketches"), "{error}");
    }

    /// Extrude shares the Shell face-reference resolver: a malformed body
    /// prefix is an error, not a downgrade to face 1 of the only body.
    #[test]
    fn malformed_face_reference_is_an_error() {
        let output = crate::evaluate_exact(
            body_extrude_request(serde_json::json!({
                "amount": 5.0,
                "operation": "new",
                "faces": [{"id": "body:x:face:1"}]
            })),
            crate::EngineCache::new(),
            None,
        );
        let error = output.response.node_errors.get("extrude").cloned().unwrap_or_default();
        assert!(!error.is_empty(), "malformed face reference was accepted");
        assert!(error.contains("body:x:face:1"), "{error}");
    }

    fn exact_bounds(
        part: &nodecraft_exact::ExactPart,
        expected_min: [f64; 3],
        expected_max: [f64; 3],
    ) {
        let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for body in &part.bodies {
            for vertex in body.brep.vertex_list() {
                let point = body.brep.vertex(*vertex).expect("vertex").point;
                for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                    min[axis] = min[axis].min(value);
                    max[axis] = max[axis].max(value);
                }
            }
        }
        for axis in 0..3 {
            assert!((min[axis] - expected_min[axis]).abs() < 1.0e-6, "min[{axis}] = {}", min[axis]);
            assert!((max[axis] - expected_max[axis]).abs() < 1.0e-6, "max[{axis}] = {}", max[axis]);
        }
    }

    /// The z range of an exact Part's vertices.
    fn z_range(part: &nodecraft_exact::ExactPart) -> (f64, f64) {
        let mut range = (f64::INFINITY, f64::NEG_INFINITY);
        for body in &part.bodies {
            for vertex in body.brep.vertex_list() {
                let z = body.brep.vertex(*vertex).expect("vertex").point.z;
                range = (range.0.min(z), range.1.max(z));
            }
        }
        range
    }

    /// A face extrude joined onto a rounded body keeps the round exact: the feature's Boolean
    /// copies its operands with their blend definitions, so the round reaches the result as a
    /// blend, not a NURBS fit. A cylinder r 20, z -20..20, its bottom rim rounded r 10.3, gets a
    /// boss extruded 5 up from its top face.
    #[test]
    fn a_face_extrude_keeps_the_body_s_round_exact() {
        let request = |values: serde_json::Value| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1, "targets": ["extrude"], "meshTargets": [],
                "nodes": [
                    {"id": "cyl", "type": "solid", "cache": {"key": "cyl"},
                     "values": {"primitiveType": "cylinder"}},
                    {"id": "round", "type": "fillet", "cache": {"key": "round"},
                     "values": {"length": 10.3, "edges": [{"center": [-20.0, 0.0, -20.0]}]}},
                    {"id": "extrude", "type": "extrude", "values": values, "cache": {"key": "extrude"}}
                ],
                "connections": [
                    {"from": "cyl", "to": "round", "in": "shape", "out": "shape"},
                    {"from": "round", "to": "extrude", "in": "profile", "out": "shape"}
                ]
            }))
            .expect("rounded body extrude request")
        };
        // The top face: the one whose own extrusion spans z 20..25.
        let top = (0..8)
            .find(|face| {
                let output = crate::evaluate_exact(
                    request(serde_json::json!({
                        "amount": 5.0, "operation": "new", "faces": [{"id": format!("face:{face}")}]
                    })),
                    crate::EngineCache::new(),
                    None,
                );
                output.exact_parts.get("extrude").is_some_and(|part| {
                    let (low, high) = z_range(part);
                    (low - 20.0).abs() < 1.0e-6 && (high - 25.0).abs() < 1.0e-6
                })
            })
            .expect("the cylinder's top face");
        let output = crate::evaluate_exact(
            request(serde_json::json!({
                "amount": 5.0, "operation": "add", "faces": [{"id": format!("face:{top}")}]
            })),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("extrude exact part");
        assert_eq!(part.bodies.len(), 1);
        let (low, high) = z_range(part);
        assert!((low + 20.0).abs() < 1.0e-6 && (high - 25.0).abs() < 1.0e-6, "z {low}..{high}");
        let mut kernel = nodecraft_kernel_profile::kernel();
        let copy = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(
            &mut kernel,
            &part.bodies[0].brep,
        )
        .expect("copy");
        let (record, _) = nodecraft_ops::SolidBody::new(copy)
            .ops_copy(&mut kernel)
            .expect("operand")
            .to_record()
            .expect("record");
        let kinds: Vec<String> = record
            .faces
            .iter()
            .map(|face| {
                format!("{:?}", face.surface).split([' ', '{', '(']).next().unwrap_or("").to_owned()
            })
            .collect();
        let blends = record
            .faces
            .iter()
            .filter(|face| matches!(face.surface, solidcore_record::brep::Surface::Blend { .. }))
            .count();
        assert_eq!(blends, 1, "the round stays an exact blend: {kinds:?}");
    }
}
