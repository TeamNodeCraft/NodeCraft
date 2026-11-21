//! Loft: one bottom and one top section, either both sketches or both bodies.
//!
//! Each section is validated as exactly one sketch or a body. Two sketches use
//! `LoftSketch`; two bodies use `FaceLoft`; a mixed pair is an error rather
//! than a first-item guess.

use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex, contract_number};

use super::{BuiltinNode, InputSection, NodeView, PartEval};

/// Registered for the `loft` contract id.
pub(crate) struct Loft;

fn single_section(mut sections: Vec<InputSection>, missing: &str) -> Result<InputSection, String> {
    if sections.is_empty() {
        return Err(missing.to_owned());
    }
    Ok(sections.remove(0))
}

impl BuiltinNode for Loft {
    fn node_types(&self) -> &'static [&'static str] {
        &["loft"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let bottom = single_section(ctx.input_sections("bottom"), "Loft has no bottom section")?;
        let top = single_section(ctx.input_sections("top"), "Loft has no top section")?;
        let bottom_part = bottom.part?;
        let top_part = top.part?;
        let bottom_sketch = ctx.input_sketch("bottom")?;
        let top_sketch = ctx.input_sketch("top")?;
        let ruled = contract_number("loft", &node.values, "ruled") >= 0.5;
        let brep = match (bottom_sketch, top_sketch) {
            (Some(bottom), Some(top)) => ctx
                .execute_warned(ExactNodeTask::LoftSketch {
                    bottom: bottom.as_ref().clone(),
                    top: top.as_ref().clone(),
                    ruled,
                })?
                .output
                .into_brep()
                .ok_or_else(|| "Loft task returned no shape".to_owned())?,
            (None, None) => {
                let mut bottom_solids = bottom_part.realized_solids()?;
                let mut top_solids = top_part.realized_solids()?;
                if bottom_solids.len() != 1 || top_solids.len() != 1 {
                    return Err(format!(
                        "Loft needs exactly one body per section, got bottom {} and top {}",
                        bottom_solids.len(),
                        top_solids.len()
                    ));
                }
                ctx.execute_warned(ExactNodeTask::FaceLoft {
                    bottom: bottom_solids.pop().expect("one bottom body"),
                    top: top_solids.pop().expect("one top body"),
                    ruled,
                    cap_ends: true,
                })?
                .output
                .into_brep()
                .ok_or_else(|| "Face loft task returned no shape".to_owned())?
            }
            _ => return Err("Loft sections must both be sketches or both be bodies".into()),
        };
        Ok(Part::from_brep(brep))
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        10
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // Loft is a generator over its sections: suppressing it turns it off.
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

    fn sketch_loft_request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["loft"], "meshTargets": [],
            "nodes": [
                {"id": "bottom", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "bottom"}},
                {"id": "top-at", "type": "translate", "values": {"z": 10.0}, "cache": {"key": "top-at"}},
                {"id": "loft", "type": "loft", "values": {"ruled": 1.0}, "cache": {"key": "loft"}}
            ],
            "connections": [
                {"from": "bottom", "to": "top-at", "in": "shape", "out": "sketch"},
                {"from": "bottom", "to": "loft", "in": "bottom", "out": "sketch"},
                {"from": "top-at", "to": "loft", "in": "top", "out": "shape"}
            ]
        }))
        .expect("sketch loft request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("record");
        nodecraft_step::record_volume(&record).expect("volume")
    }

    /// Removing the registry entry makes this fail: the engine no longer knows
    /// the `loft` type. The test is the entry's fail proof.
    #[test]
    fn registry_routes_loft() {
        let builtin = crate::nodes::find("loft").expect("loft registered");
        assert_eq!(builtin.node_types(), &["loft"]);
        let values = serde_json::json!({});
        let view = NodeView::new("l", "loft", &values, false);
        assert_eq!(builtin.body_layout(view, &[]), Some(vec!["l".to_owned()]));
    }

    /// Two sketches loft through `LoftSketch` with the contract `ruled` value.
    #[test]
    fn loft_two_sketches_with_ruled() {
        let output = crate::evaluate_exact(sketch_loft_request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("loft").expect("loft exact part");
        assert_eq!(part.bodies.len(), 1);
        assert!(volume(&part.bodies[0].brep) > 0.0);
        let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
        for vertex in part.bodies[0].brep.vertex_list() {
            let z = part.bodies[0].brep.vertex(*vertex).expect("vertex").point.z;
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        assert!((min_z - 0.0).abs() < 1.0e-6, "bottom z {min_z}");
        assert!((max_z - 10.0).abs() < 1.0e-6, "top z {max_z}");
    }

    /// A missing section keeps its existing error.
    #[test]
    fn loft_requires_both_sections() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["loft"], "meshTargets": [],
            "nodes": [
                {"id": "bottom", "type": "sketch", "values": {"sketch": serde_json::to_value(square_sketch()).unwrap()}, "cache": {"key": "bottom"}},
                {"id": "loft", "type": "loft", "values": {}, "cache": {"key": "loft"}}
            ],
            "connections": [
                {"from": "bottom", "to": "loft", "in": "bottom", "out": "sketch"}
            ]
        }))
        .expect("missing section request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let error = output.response.node_errors.get("loft").expect("missing section error");
        assert!(error.contains("Loft has no top section"), "{error}");
    }
}
