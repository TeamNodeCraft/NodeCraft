//! Fillet: round selected edges of one exact body through `OpsSession::blend`.
//!
//! The blend itself stays `node_task::run_edge_session`: it proves the input
//! body, validates the `EdgeTreatment` from the contract fields, resolves the
//! tangent chain and calls `OpsSession`. This module owns the node-shaped
//! parts: the shape input, the no-edges pass-through, the one-body requirement,
//! warning collection and the placement/display behaviour.

use super::{BuiltinNode, NodeView, PartEval};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `fillet` contract id.
pub(crate) struct Fillet;

impl BuiltinNode for Fillet {
    fn node_types(&self) -> &'static [&'static str] {
        &["fillet"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source_part) = ctx.input_part(&["shape"])? else {
            return Err("Fillet has no shape input".into());
        };
        // A fresh edge node has no selected edges yet: pass the input through
        // until the user picks edges.
        if !super::has_edges(node.values) {
            return Ok((*source_part).clone());
        }
        let mut solids = source_part.realized_bodies()?;
        if solids.len() != 1 {
            return Err(format!("Fillet requires exactly one solid body, got {}", solids.len()));
        }
        let source = solids.pop().expect("one solid body");
        let mut values = node.values.clone();
        values["mode"] = serde_json::json!("fillet");
        let result = ctx.execute_warned(ExactNodeTask::Fillet { body: source, values })?;
        let crate::node_task::TaskResult { output, warnings } = result;
        // The warned task is the same one the display/operand path uses; keep the
        // blend's user-facing warnings on this node instead of dropping them.
        ctx.warn(warnings);
        // Keep the body as the task made it: its kernel body is the next node's operand, with
        // this round's exact blend definition.
        match output {
            crate::node_task::NodeOutput::Shape(body) => Ok(Part::from_bodies([body])),
            _ => Err("fillet task returned no shape".into()),
        }
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        12
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
        if super::has_edges(node.values) {
            Some(vec![node.id.to_owned()])
        } else {
            Some(input.to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::{Node, Request};

    fn request(edges: serde_json::Value) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["box", "fillet"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                {"id": "fillet", "type": "fillet", "cache": {"key": "fillet"},
                 "values": {"continuity": "", "length": 2, "edges": edges}}
            ],
            "connections": [{"from": "box", "to": "fillet", "in": "shape", "out": "shape"}]
        }))
        .expect("fillet request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("brep record");
        nodecraft_step::record_volume(&record).expect("brep volume")
    }

    #[test]
    fn registry_routes_fillet() {
        let builtin = crate::nodes::find("fillet").expect("fillet registered");
        assert_eq!(builtin.node_types(), &["fillet"]);
        assert_eq!(
            crate::nodes::find("fillet").map(|node| node.exact_weight(
                crate::nodes::NodeView::new("f", "fillet", &serde_json::Value::Null, false,)
            )),
            Some(12)
        );
    }

    /// Rounding a box edge removes material: the fillet's exact volume is
    /// strictly below the box's, and both are built through the registry.
    #[test]
    fn fillet_volume_is_below_the_source_box_volume() {
        let output = crate::evaluate_exact(
            request(serde_json::json!([{"ordinal": 0}])),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let fillet_part = output.exact_parts.get("fillet").expect("fillet exact part");
        assert_eq!(box_part.bodies.len(), 1);
        assert_eq!(fillet_part.bodies.len(), 1, "one blended body");
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&fillet_part.bodies[0].brep);
        assert!((before - 24_000.0).abs() < 1.0, "box volume {before}");
        assert!(after > 0.0, "fillet volume {after}");
        assert!(after < before - 1.0, "a fillet removes material: {after} vs {before}");
    }

    /// No selected edges is a pass-through, not an error or a zero-radius blend.
    #[test]
    fn no_edges_passes_the_source_part_through() {
        let output =
            crate::evaluate_exact(request(serde_json::json!([])), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let fillet_part = output.exact_parts.get("fillet").expect("fillet exact part");
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&fillet_part.bodies[0].brep);
        assert!((after - before).abs() < 1.0e-6, "pass-through volume {after} vs {before}");
    }

    /// The trait's warning channel writes to the node's `node_warnings` entry;
    /// the blend's warnings come back from `execute_exact_warned` this way.
    #[test]
    fn warnings_are_recorded_on_the_node() {
        use super::super::PartEval as _;
        let request = request(serde_json::json!([{"ordinal": 0}]));
        let node =
            request.nodes.iter().find(|node| node.id == "fillet").expect("fillet node").clone();
        let mut eval = crate::Eval::new(&request, None);
        {
            let mut ctx = crate::nodes::EnginePartEval::new(&mut eval, &node);
            ctx.warn(vec!["rounded past the adjacent face".to_owned()]);
        }
        assert_eq!(
            eval.warnings.get("fillet").map(Vec::as_slice),
            Some(["rounded past the adjacent face".to_owned()].as_slice()),
        );
    }
}
