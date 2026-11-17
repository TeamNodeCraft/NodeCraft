//! Shell: hollow every body of an exact Part by a signed wall thickness.
//!
//! The kernel operation stays `node_task::ExactNodeTask::Shell`
//! (`OpsSession::shell_body`). This module owns the node-shaped parts: the
//! shape input, the multi-body order, positive-thickness validation, the
//! `face:N` / `body:B:face:N` opening references, per-body opening filtering
//! and the display body layout.

use super::{BuiltinNode, NodeView, PartEval};
use crate::face_ref::face_reference_body_and_index;
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `shell` contract id.
pub(crate) struct Shell;

impl BuiltinNode for Shell {
    fn node_types(&self) -> &'static [&'static str] {
        &["shell"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Shell has no shape input".into());
        };
        let solids = source.realized_bodies()?;
        if solids.is_empty() {
            return Err("Shell source Part is empty".into());
        }
        let thickness = crate::contract_number("shell", node.values, "thickness");
        if !(thickness.is_finite() && thickness > 0.0) {
            return Err("Shell thickness must be positive".into());
        }
        let inward = crate::contract_text("shell", node.values, "direction") == "inward";
        let signed = if inward { -thickness } else { thickness };
        let references = super::face_references(node.values);
        // Validate references before touching geometry. Silently ignoring a
        // malformed or out-of-range opening turns a requested shell into a
        // closed shell of the wrong shape.
        let mut parsed = Vec::with_capacity(references.len());
        for reference in &references {
            let Some((body, face_index)) = face_reference_body_and_index(reference) else {
                return Err(format!(
                    "Shell opening face reference {reference:?} is invalid; expected face:N or body:B:face:N"
                ));
            };
            if let Some(body) = body
                && body >= solids.len()
            {
                return Err(format!(
                    "Shell opening {reference:?} names body {body}, but the source has {} bodies",
                    solids.len()
                ));
            }
            parsed.push((body, face_index));
        }
        let mut bodies = Vec::with_capacity(solids.len());
        for (body_index, solid) in solids.into_iter().enumerate() {
            let mut openings = Vec::new();
            for (body_ref, face_index) in &parsed {
                // An unqualified `face:N` applies to every body; `body:B:face:N`
                // only to body B. A body-scoped reference to a missing face is
                // an error; an unqualified one that this body does not have is
                // skipped.
                if body_ref.is_some_and(|body| body != body_index) {
                    continue;
                }
                if solid.brep().faces().ids().nth(*face_index).is_none() {
                    if body_ref.is_some() {
                        return Err(format!(
                            "Shell opening body:{body_index}:face:{face_index} does not exist on this body"
                        ));
                    }
                    continue;
                }
                openings.push(*face_index);
            }
            let result = ctx.execute_warned(ExactNodeTask::Shell {
                body: solid,
                signed_thickness: signed,
                openings,
            })?;
            let crate::node_task::TaskResult { output, warnings } = result;
            ctx.warn(warnings);
            let shelled = output
                .into_shape()
                .ok_or_else(|| "Shell task returned no shape".to_owned())?;
            bodies.push(crate::BodyInstance {
                body: shelled,
                placement: nodecraft_exact::Placement::identity(),
                realized: Default::default(),
                source: None,
                source_body: 0,
                name: None,
                color: None,
            });
        }
        Ok(Part { bodies, sketches: Vec::new() })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        12
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
        // One shelled body per input body, in the same order.
        super::input_layout(inputs, "shape").map(<[String]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn link(from: &str, to: &str) -> serde_json::Value {
        serde_json::json!({"from": from, "to": to, "in": "shape", "out": "shape"})
    }

    fn two_box_request(faces: serde_json::Value, thickness: f64) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "big", "type": "primitive", "cache": {"key": "big"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "small", "type": "primitive", "cache": {"key": "small"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 30.0}},
                {"id": "part", "type": "combine", "cache": {"key": "part"}, "values": {}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": "inward", "faces": faces}}
            ],
            "connections": [
                link("big", "part"),
                link("small", "move"),
                link("move", "part"),
                link("part", "shell")
            ]
        }))
        .expect("shell request")
    }

    fn single_box_request(faces: serde_json::Value, thickness: f64) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": "inward", "faces": faces}}
            ],
            "connections": [link("box", "shell")]
        }))
        .expect("shell request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("brep record");
        nodecraft_step::record_volume(&record).expect("brep volume")
    }

    fn bounds(part: &nodecraft_exact::ExactPart, index: usize) -> ([f64; 3], [f64; 3]) {
        let brep = part.realized_breps().expect("realized body");
        let record = nodecraft_step::record_of_brep(&brep[index]).expect("body record");
        let (min, max) = nodecraft_step::record_bounds(&record).expect("body bounds");
        ([min[0], min[1], min[2]], [max[0], max[1], max[2]])
    }

    /// Removing the registry entry makes the engine reject `shell`, so this
    /// test is the entry's fail proof. It also pins the moved cost estimate.
    #[test]
    fn registry_routes_shell() {
        let builtin = crate::nodes::find("shell").expect("shell registered");
        assert_eq!(builtin.node_types(), &["shell"]);
        let values = serde_json::Value::Null;
        assert_eq!(
            builtin.exact_weight(crate::nodes::NodeView::new("s", "shell", &values, false)),
            12
        );
    }

    /// Every input body is shelled, in Part body order. The first box stays
    /// around the origin and the second stays at x=30, so both the bounds and
    /// the larger first volume prove the order.
    #[test]
    fn multi_body_shell_keeps_the_source_body_order() {
        let output = crate::evaluate_exact(
            two_box_request(serde_json::json!([]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("shell").expect("shell exact part");
        assert_eq!(part.bodies.len(), 2, "one shelled body per input body");
        let (first_min, first_max) = bounds(part, 0);
        let (second_min, second_max) = bounds(part, 1);
        assert!(first_min[0] < second_min[0], "{first_min:?} vs {second_min:?}");
        assert!((first_min[0] + 10.0).abs() < 1.0e-6, "{first_min:?}");
        assert!((first_max[0] - 10.0).abs() < 1.0e-6, "{first_max:?}");
        assert!((second_min[0] - 25.0).abs() < 1.0e-6, "{second_min:?}");
        assert!((second_max[0] - 35.0).abs() < 1.0e-6, "{second_max:?}");
        let first_volume = volume(&part.bodies[0].brep);
        let second_volume = volume(&part.bodies[1].brep);
        assert!(first_volume > second_volume, "{first_volume} vs {second_volume}");
        assert!(first_volume < 8_000.0, "the big box was shelled: {first_volume}");
        assert!(second_volume < 1_000.0, "the small box was shelled: {second_volume}");
    }

    /// A shell needs a positive, finite wall thickness; zero and negative
    /// values are errors instead of a closed or inverted body.
    #[test]
    fn shell_rejects_non_positive_thickness() {
        for thickness in [0.0, -2.0] {
            let output = crate::evaluate_exact(
                single_box_request(serde_json::json!([]), thickness),
                crate::EngineCache::new(),
                None,
            );
            let error = output.response.node_errors.get("shell").expect("thickness error");
            assert!(error.contains("thickness must be positive"), "{thickness}: {error}");
        }
    }

    /// Both opening reference forms resolve: `face:N` and `body:B:face:N`.
    #[test]
    fn shell_accepts_face_and_body_scoped_opening_references() {
        for reference in ["face:1", "body:0:face:1"] {
            let output = crate::evaluate_exact(
                single_box_request(serde_json::json!([{"id": reference}]), 1.0),
                crate::EngineCache::new(),
                None,
            );
            assert!(
                output.response.node_errors.is_empty(),
                "{reference}: {:?}",
                output.response.node_errors
            );
            let part = output.exact_parts.get("shell").expect("shell exact part");
            assert_eq!(part.bodies.len(), 1);
            assert!(volume(&part.bodies[0].brep) > 0.0);
        }
    }

    /// A body-scoped opening only opens that body. Body 0 must match a closed
    /// shell of the same source; body 1 must lose the open face's wall.
    #[test]
    fn body_scoped_openings_are_filtered_per_body() {
        let closed = crate::evaluate_exact(
            two_box_request(serde_json::json!([]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        let opened = crate::evaluate_exact(
            two_box_request(serde_json::json!([{"id": "body:1:face:1"}]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(closed.response.node_errors.is_empty(), "{:?}", closed.response.node_errors);
        assert!(opened.response.node_errors.is_empty(), "{:?}", opened.response.node_errors);
        let closed = closed.exact_parts.get("shell").expect("closed shell part");
        let opened = opened.exact_parts.get("shell").expect("opened shell part");
        let closed_first = volume(&closed.bodies[0].brep);
        let opened_first = volume(&opened.bodies[0].brep);
        assert!(
            (closed_first - opened_first).abs() < 1.0e-6,
            "body 0 must stay closed: {closed_first} vs {opened_first}"
        );
        let closed_second = volume(&closed.bodies[1].brep);
        let opened_second = volume(&opened.bodies[1].brep);
        assert!(
            closed_second - opened_second > 1.0,
            "body 1 must lose the open face's wall: {closed_second} vs {opened_second}"
        );
    }

    /// A shell keeps the source display layout: one body in, one body out, in
    /// the same order.
    #[test]
    fn body_layout_is_the_source_layout() {
        let builtin = crate::nodes::find("shell").expect("shell registered");
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("shell", "shell", &values, false);
        assert_eq!(
            builtin.body_layout(view, &[("shape", vec!["big".to_owned(), "small".to_owned()])]),
            Some(vec!["big".to_owned(), "small".to_owned()]),
        );
    }
}
