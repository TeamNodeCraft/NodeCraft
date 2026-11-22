//! Chamfer: bevel selected edges of one exact body through `OpsSession::blend`.
//!
//! The blend itself stays `node_task::run_edge_session`, shared with Fillet: it
//! proves the input body, validates the `EdgeTreatment` from the contract
//! fields, resolves the tangent chain and calls `OpsSession`. This module owns
//! only the node-shaped parts: the shape input, the no-edges pass-through, the
//! one-body requirement, warning collection and the placement/display
//! behaviour.

use super::{BuiltinNode, NodeView, PartEval};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `chamfer` contract id.
pub(crate) struct Chamfer;

impl BuiltinNode for Chamfer {
    fn node_types(&self) -> &'static [&'static str] {
        &["chamfer"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source_part) = ctx.input_part(&["shape"])? else {
            return Err("Chamfer has no shape input".into());
        };
        // A fresh edge node has no selected edges yet: pass the input through
        // until the user picks edges.
        if !super::has_edges(node.values) {
            return Ok((*source_part).clone());
        }
        let mut solids = source_part.realized_bodies()?;
        if solids.len() != 1 {
            return Err(format!("Chamfer requires exactly one solid body, got {}", solids.len()));
        }
        let source = solids.pop().expect("one solid body");
        let mut values = node.values.clone();
        values["mode"] = serde_json::json!("chamfer");
        let result = ctx.execute_warned(ExactNodeTask::Chamfer { body: source, values })?;
        let crate::node_task::TaskResult { output, warnings } = result;
        // Keep the blend's user-facing warnings on this node instead of
        // dropping them with the task output.
        ctx.warn(warnings);
        super::edge_blend_part(output, "chamfer")
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        8
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
            "requestId": 1, "targets": ["box", "chamfer"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                {"id": "chamfer", "type": "chamfer", "cache": {"key": "chamfer"},
                 "values": {"length": 2, "edges": edges}}
            ],
            "connections": [{"from": "box", "to": "chamfer", "in": "shape", "out": "shape"}]
        }))
        .expect("chamfer request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("brep record");
        nodecraft_step::record_volume(&record).expect("brep volume")
    }

    /// Removing the registry entry makes the engine reject `chamfer`, so this
    /// test is the entry's fail proof. It also pins the moved cost estimate.
    #[test]
    fn registry_routes_chamfer() {
        let builtin = crate::nodes::find("chamfer").expect("chamfer registered");
        assert_eq!(builtin.node_types(), &["chamfer"]);
        let values = serde_json::Value::Null;
        assert_eq!(
            builtin.exact_weight(crate::nodes::NodeView::new("c", "chamfer", &values, false)),
            8
        );
    }

    /// A fresh chamfer has no selected edges: it passes the source Part through
    /// rather than running a zero-size blend.
    #[test]
    fn no_edges_passes_the_source_part_through() {
        let output =
            crate::evaluate_exact(request(serde_json::json!([])), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let chamfer_part = output.exact_parts.get("chamfer").expect("chamfer exact part");
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&chamfer_part.bodies[0].brep);
        assert!((after - before).abs() < 1.0e-6, "pass-through volume {after} vs {before}");
    }

    /// A chamfer edits exactly one body; a multi-body Part is an error rather
    /// than silently beveling the first body.
    #[test]
    fn multiple_bodies_are_an_error() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["chamfer"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "cache": {"key": "a"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "b", "type": "primitive", "cache": {"key": "b"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "part", "type": "combine", "cache": {"key": "part"}, "values": {}},
                {"id": "chamfer", "type": "chamfer", "cache": {"key": "chamfer"},
                 "values": {"length": 2, "edges": [{"ordinal": 0}]}}
            ],
            "connections": [
                {"from": "a", "to": "part", "in": "shape", "out": "shape"},
                {"from": "b", "to": "part", "in": "shape", "out": "shape"},
                {"from": "part", "to": "chamfer", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("multi-body chamfer request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        let error = output.response.node_errors.get("chamfer").expect("multi-body error");
        assert!(error.contains("exactly one solid body"), "{error}");
    }

    /// Beveling a box edge removes material: the chamfer's exact volume is
    /// strictly below the source box's, and both are built through the registry.
    #[test]
    fn chamfer_volume_is_below_the_source_box_volume() {
        let output = crate::evaluate_exact(
            request(serde_json::json!([{"ordinal": 0}])),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let chamfer_part = output.exact_parts.get("chamfer").expect("chamfer exact part");
        assert_eq!(box_part.bodies.len(), 1);
        assert_eq!(chamfer_part.bodies.len(), 1, "one beveled body");
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&chamfer_part.bodies[0].brep);
        assert!((before - 24_000.0).abs() < 1.0, "box volume {before}");
        assert!(after > 0.0, "chamfer volume {after}");
        assert!(after < before - 1.0, "a chamfer removes material: {after} vs {before}");
    }

    /// The trait's warning channel writes to the node's `node_warnings` entry;
    /// the blend's warnings come back from `execute_exact_warned` this way.
    #[test]
    fn warnings_are_recorded_on_the_node() {
        use super::super::PartEval as _;
        let request = request(serde_json::json!([{"ordinal": 0}]));
        let node =
            request.nodes.iter().find(|node| node.id == "chamfer").expect("chamfer node").clone();
        let mut eval = crate::Eval::new(&request, None);
        {
            let mut ctx = crate::nodes::EnginePartEval::new(&mut eval, &node);
            ctx.warn(vec!["bevel crossed the adjacent face".to_owned()]);
        }
        assert_eq!(
            eval.warnings.get("chamfer").map(Vec::as_slice),
            Some(["bevel crossed the adjacent face".to_owned()].as_slice()),
        );
    }

    /// The layout follows the source until edges are selected, then the bevel
    /// is one new display body.
    #[test]
    fn body_layout_follows_the_source_until_edges_are_selected() {
        let builtin = crate::nodes::find("chamfer").expect("chamfer registered");
        let no_edges = serde_json::json!({});
        let source = vec!["box".to_owned()];
        assert_eq!(
            builtin.body_layout(
                crate::nodes::NodeView::new("chamfer", "chamfer", &no_edges, false),
                &[("shape", source.clone())],
            ),
            Some(source),
        );
        let edges = serde_json::json!({"edges": [{"ordinal": 0}]});
        assert_eq!(
            builtin.body_layout(
                crate::nodes::NodeView::new("chamfer", "chamfer", &edges, false),
                &[("shape", vec!["box".to_owned()])],
            ),
            Some(vec!["chamfer".to_owned()]),
        );
    }
}
