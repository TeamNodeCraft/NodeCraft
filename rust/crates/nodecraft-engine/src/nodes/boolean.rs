//! Boolean and Fuse: exact n-ary boolean over two Part operands.
//!
//! `boolean` and `fuse` are two contract ids for one implementation. Both read
//! the same contract `mode`/`op` selector, evaluate operands through the Part
//! path, and run the shared `ExactNodeTask::BooleanLists` task. A foldable run
//! of Union nodes reaches this module as one `UnionRun` plan: the engine skips
//! the intermediates and the node runs one `union_all` over the leaf Parts.

use std::collections::HashSet;

use nodecraft_protocol::Connection;

use super::{BuiltinNode, InputState, MergeProvenance, NodeView, PartEval};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `boolean` and `fuse` contract ids.
pub(crate) struct Boolean;

/// One connected operand, in the order the node reports it.
fn operand<'a>(node: NodeView<'_>, connections: &'a [Connection], port: &str) -> Option<&'a str> {
    connections
        .iter()
        .find(|link| link.to == node.id && link.input == port)
        .map(|link| link.from.as_str())
}

impl BuiltinNode for Boolean {
    fn node_types(&self) -> &'static [&'static str] {
        &["boolean", "fuse"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let a_state = ctx.input_state("a");
        let b_state = ctx.input_state("b");
        // Resolve the shared fusion plan before reading A/B: evaluating an
        // intermediate Union here would compute the very node the engine
        // skipped on purpose. The plan carries the leaf Parts only.
        if let Some(run) = ctx.union_run(node) {
            let mut operands = Vec::new();
            for operand in run.into_operands() {
                let part = operand?;
                operands.extend(part.realized_bodies()?);
            }
            let result = ctx.execute_warned(ExactNodeTask::BooleanLists {
                op: nodecraft_bool::BooleanOp::Union,
                a: operands,
                b: Vec::new(),
            })?;
            ctx.warn(result.warnings);
            let shapes = result
                .output
                .into_shapes()
                .ok_or_else(|| "fused Union task returned no shapes".to_owned())?;
            return Ok(Part::from_bodies(shapes));
        }

        // Read only the live operands. An inactive operand contributes nothing,
        // and evaluating it would be wasted work; the state itself already
        // tells the two cases apart.
        let a = match a_state {
            InputState::Active => ctx.input_part(&["a"])?,
            InputState::Unconnected | InputState::Inactive => None,
        };
        let b = match b_state {
            InputState::Active => ctx.input_part(&["b"])?,
            InputState::Unconnected | InputState::Inactive => None,
        };
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => return Ok((*a).clone()),
            (None, Some(b)) => return Ok((*b).clone()),
            (None, None)
                if a_state == InputState::Unconnected && b_state == InputState::Unconnected =>
            {
                return Err("Part Boolean has no connected operand".into());
            }
            (None, None) if a_state == InputState::Inactive && b_state == InputState::Inactive => {
                return Err("Part Boolean has no active operand".into());
            }
            // One port is connected but inactive (it evaluated to an empty Part
            // before) and the other is unconnected.
            (None, None) => return Ok(Part::empty()),
        };
        let (op, reverse) = crate::boolean_mode(node.values)?;
        let (a, b) = if reverse { (b, a) } else { (a, b) };
        for (operand, part) in [("A", a.as_ref()), ("B", b.as_ref())] {
            if part.bodies.len() > crate::MAX_BOOLEAN_SOLIDS {
                return Err(format!(
                    "Boolean operand {operand} contains {} solids (for example a Scatter with Combine on). Combine groups the instances into one Part; a Boolean still needs an exact union of every solid, and supports at most {}. Reduce the scatter count or boolean a smaller Part.",
                    part.bodies.len(),
                    crate::MAX_BOOLEAN_SOLIDS
                ));
            }
        }
        let result = ctx.execute_warned(ExactNodeTask::BooleanLists {
            op,
            a: a.realized_bodies()?,
            b: b.realized_bodies()?,
        })?;
        ctx.warn(result.warnings);
        let shapes = result
            .output
            .into_shapes()
            .ok_or_else(|| "Boolean task returned no shapes".to_owned())?;
        Ok(Part::from_bodies(shapes))
    }

    fn export_terminal(&self) -> bool {
        true
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        8
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        // A suppressed Boolean passes operand A through, or B when A is absent.
        index
            .first_incoming(&node.id, "a")
            .or_else(|| index.first_incoming(&node.id, "b"))
            .map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        let a = super::input_layout(inputs, "a");
        let b = super::input_layout(inputs, "b");
        match (a, b) {
            // Both operands are merged into one result body; provenance is lost
            // inside it, so the body is the node's own.
            (Some(_), Some(_)) => Some(vec![node.id.to_owned()]),
            // A missing operand passes the other through, including its layout.
            (Some(a), None) => Some(a.to_vec()),
            (None, Some(b)) => Some(b.to_vec()),
            (None, None) => None,
        }
    }

    fn merge_provenance(
        &self,
        node: NodeView<'_>,
        connections: &[Connection],
        affected: &HashSet<String>,
    ) -> MergeProvenance {
        // A merge only loses provenance when both operands are present and at
        // least one is outside the edited branch. One-operand Boolean is a
        // pass-through, so it preserves the source layout.
        let (Some(a), Some(b)) = (operand(node, connections, "a"), operand(node, connections, "b"))
        else {
            return MergeProvenance::Preserved;
        };
        if !affected.contains(a) || !affected.contains(b) {
            MergeProvenance::LosesProvenance
        } else {
            MergeProvenance::Preserved
        }
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    use super::*;

    fn part(id: &str, node_type: &str, values: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "type": node_type,
            "values": values,
            "cache": {"key": id}
        })
    }

    fn request(node_type: &str, mode: &str) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": [],
            "nodes": [
                part("a", "primitive", serde_json::json!({"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0})),
                part("b", "primitive", serde_json::json!({"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0})),
                part("move", "translate", serde_json::json!({"x": 10.0})),
                part("u", node_type, serde_json::json!({"mode": mode}))
            ],
            "connections": [
                {"from": "a", "to": "u", "in": "a", "out": "shape"},
                {"from": "b", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "u", "in": "b", "out": "shape"}
            ]
        }))
        .expect("boolean request")
    }

    fn volume(part: &nodecraft_exact::ExactPart) -> f64 {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        part.bodies
            .iter()
            .map(|body| nodecraft_layers::kernel_volume(&mut kernel, &body.brep).expect("volume"))
            .sum()
    }

    fn evaluate(request: Request) -> crate::EngineOutput {
        crate::evaluate_exact(request, crate::EngineCache::new(), None)
    }

    #[test]
    fn registry_routes_boolean_and_fuse() {
        let boolean = crate::nodes::find("boolean").expect("boolean registered");
        let fuse = crate::nodes::find("fuse").expect("fuse registered");
        assert_eq!(boolean.node_types(), &["boolean", "fuse"]);
        assert!(std::ptr::eq(boolean, fuse));
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("u", "boolean", &values, false);
        assert_eq!(boolean.exact_weight(view), 8);
    }

    /// Union adds the overlap once, difference removes B, intersection keeps it.
    /// All three run through the registry entry.
    #[test]
    fn union_subtract_and_intersect_have_their_exact_volumes() {
        for (mode, expected) in [("union", 12_000.0), ("subtract", 4_000.0), ("intersect", 4_000.0)]
        {
            let output = evaluate(request("boolean", mode));
            assert!(
                output.response.node_errors.is_empty(),
                "{mode}: {:?}",
                output.response.node_errors
            );
            let part = output.exact_parts.get("u").expect("boolean exact part");
            let volume = volume(part);
            assert!((volume - expected).abs() < 1.0, "{mode} volume {volume} vs {expected}");
        }
    }

    /// `fuse` is the same implementation and reads the same contract mode.
    #[test]
    fn fuse_uses_the_shared_boolean_path() {
        let output = evaluate(request("fuse", "intersect"));
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("u").expect("fuse exact part");
        assert!((volume(part) - 4_000.0).abs() < 1.0);
    }

    /// A suppressed operand drops out; the active one is the result, not a
    /// merged body and not an error.
    #[test]
    fn a_suppressed_operand_passes_the_other_through() {
        let mut request = request("boolean", "union");
        request.nodes[0].suppressed = true;
        let output = evaluate(request);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("u").expect("boolean exact part");
        assert_eq!(part.bodies.len(), 1);
        assert!((volume(part) - 8_000.0).abs() < 1.0, "only B survives");
    }

    /// Both operands suppressed is the one Boolean error state.
    #[test]
    fn two_suppressed_operands_report_no_active_operand() {
        let mut request = request("boolean", "union");
        request.nodes[0].suppressed = true;
        request.nodes[1].suppressed = true;
        let output = evaluate(request);
        assert_eq!(
            output.response.node_errors.get("u").map(String::as_str),
            Some("Part Boolean has no active operand"),
        );
    }

    /// The fused plan makes one `union_all` over the leaves and leaves the
    /// intermediate Parts uncomputed, while the final geometry is the chain's.
    #[test]
    fn a_union_chain_fuses_through_the_registry_entry() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["u3"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "b"}},
                {"id": "c", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "c"}},
                {"id": "d", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "d"}},
                {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}},
                {"id": "u1", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u1"}},
                {"id": "u2", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u2"}},
                {"id": "u3", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u3"}}
            ],
            "connections": [
                {"from": "a", "to": "u1", "in": "a", "out": "shape"},
                {"from": "b", "to": "u1", "in": "b", "out": "shape"},
                {"from": "u1", "to": "u2", "in": "a", "out": "shape"},
                {"from": "c", "to": "u2", "in": "b", "out": "shape"},
                {"from": "u2", "to": "u3", "in": "a", "out": "shape"},
                {"from": "t", "to": "u3", "in": "b", "out": "shape"},
                {"from": "d", "to": "t", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("fused union request");
        let output = evaluate(request);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(output.exact_parts.get("u1").is_none(), "u1 was folded away");
        assert!(output.exact_parts.get("u2").is_none(), "u2 was folded away");
        let part = output.exact_parts.get("u3").expect("fused target");
        assert_eq!(part.bodies.len(), 1);
        let volume = volume(part);
        assert!((volume - 2_000.0).abs() < 1.0, "volume {volume}");
    }

    /// The three-state input declaration is what makes this possible: an
    /// inactive operand is skipped instead of being evaluated only to discover
    /// that it contributes an empty Part.
    #[test]
    fn an_inactive_operand_is_not_evaluated() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": [],
            "nodes": [
                part("dead", "primitive", serde_json::json!({"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0})),
                part("live", "primitive", serde_json::json!({"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0})),
                part("u", "boolean", serde_json::json!({"mode": "union"}))
            ],
            "connections": [
                {"from": "dead", "to": "u", "in": "a", "out": "shape"},
                {"from": "live", "to": "u", "in": "b", "out": "shape"}
            ]
        }))
        .expect("inactive operand request");
        let mut dead = request.nodes.iter().find(|node| node.id == "dead").expect("dead").clone();
        dead.suppressed = true;
        let mut request = request;
        request.nodes[0] = dead;
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "u").expect("target").clone();
        let boolean = crate::nodes::find("boolean").expect("boolean registered");
        let values = serde_json::json!({"mode": "union"});
        let view = crate::nodes::NodeView::new("u", "boolean", &values, false);
        let part = {
            let mut ctx = crate::nodes::EnginePartEval::new(&mut eval, &target);
            boolean.evaluate(view, &mut ctx).expect("B is the result")
        };
        assert_eq!(part.bodies.len(), 1);
        assert!(
            !eval.parts.contains_key("dead"),
            "the inactive operand was evaluated to an empty Part",
        );
        assert!(eval.parts.contains_key("live"), "the live operand was evaluated");
    }

    /// Body layout: both operands merge into the node's body; one operand keeps
    /// its upstream layout; none is unknown.
    #[test]
    fn body_layout_follows_the_operand_count() {
        let builtin = crate::nodes::find("boolean").expect("boolean registered");
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("u", "boolean", &values, false);
        let a = vec!["a".to_owned()];
        let b = vec!["b0".to_owned(), "b1".to_owned()];
        assert_eq!(
            builtin.body_layout(view, &[("a", a.clone()), ("b", b.clone())]),
            Some(vec!["u".to_owned()]),
        );
        assert_eq!(builtin.body_layout(view, &[("a", a.clone())]), Some(a));
        assert_eq!(builtin.body_layout(view, &[("b", b.clone())]), Some(b));
        assert_eq!(builtin.body_layout(view, &[]), None);
    }

    /// Both operands outside the edit lose provenance; both inside preserve it;
    /// a one-operand pass-through preserves it.
    #[test]
    fn merge_provenance_tracks_both_operands() {
        let builtin = crate::nodes::find("boolean").expect("boolean registered");
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("u", "boolean", &values, false);
        let connections = [
            Connection { from: "a".into(), to: "u".into(), input: "a".into(), out: "shape".into() },
            Connection { from: "b".into(), to: "u".into(), input: "b".into(), out: "shape".into() },
        ];
        let selected: HashSet<String> = ["selected".to_owned()].into_iter().collect();
        assert_eq!(
            builtin.merge_provenance(view, &connections, &selected),
            MergeProvenance::LosesProvenance,
        );
        let branch: HashSet<String> = ["a".to_owned(), "b".to_owned()].into_iter().collect();
        assert_eq!(
            builtin.merge_provenance(view, &connections, &branch),
            MergeProvenance::Preserved,
        );
        assert_eq!(
            builtin.merge_provenance(view, &connections[..1], &selected),
            MergeProvenance::Preserved,
        );
    }
}
