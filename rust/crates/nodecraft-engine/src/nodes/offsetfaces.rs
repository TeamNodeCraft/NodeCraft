//! Offset Faces: move selected faces of an exact Part along their normals.
//!
//! The kernel operation stays `node_task::ExactNodeTask::OffsetFaces`
//! (`OpsSession::offset_faces_body`). This module owns the node-shaped parts:
//! the shape input, the no-faces pass-through, the `face:N` /
//! `body:B:face:N` per-body filtering, the suppressed/source pass-through and
//! the display body layout.

use super::{BuiltinNode, NodeView, PartEval};
use crate::face_ref::{parse_face_references, resolve_face_references};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `offsetfaces` contract id.
pub(crate) struct OffsetFaces;

impl BuiltinNode for OffsetFaces {
    fn node_types(&self) -> &'static [&'static str] {
        &["offsetfaces"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Offset Faces has no shape input".into());
        };
        let references = parse_face_references(node.values)?;
        // No selected faces is a pass-through, not a zero-distance offset.
        if references.is_empty() {
            return Ok((*source).clone());
        }
        let distance = crate::contract_number("offsetfaces", node.values, "distance");
        let solids = source.realized_bodies()?;
        let body_faces: Vec<usize> = solids
            .iter()
            .map(|solid| solid.brep().faces().ids().count())
            .collect();
        let openings = resolve_face_references(&references, &body_faces)?;
        let mut bodies = Vec::with_capacity(solids.len());
        for (body, faces) in solids.into_iter().zip(openings) {
            if faces.is_empty() {
                bodies.push(crate::BodyInstance {
                    body,
                    placement: nodecraft_exact::Placement::identity(),
                    realized: Default::default(),
                    source: None,
                    source_body: 0,
                    name: None,
                    color: None,
                });
                continue;
            }
            let result = ctx.execute_warned(ExactNodeTask::OffsetFaces { body, faces, distance })?;
            let crate::node_task::TaskResult { output, warnings } = result;
            ctx.warn(warnings);
            let offset = output
                .into_shape()
                .ok_or_else(|| "Offset Faces task returned no shape".to_owned())?;
            bodies.push(crate::BodyInstance {
                body: offset,
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
        // Offset Faces had no workload arm before the port; the engine's
        // default feature weight is 2.
        2
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        // Offset Faces is a shape modifier: suppressing it passes the shape
        // through instead of turning it into a generator.
        index
            .first_incoming_any(&node.id, &["shape", "profile"])
            .map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // Every source body survives in order: edited bodies are replaced in
        // place, and a body with no matching face reference is unchanged.
        super::input_layout(inputs, "shape").map(<[String]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn link(from: &str, to: &str) -> serde_json::Value {
        serde_json::json!({"from": from, "to": to, "in": "shape", "out": "shape"})
    }

    fn single_box_request(faces: serde_json::Value, suppressed: bool) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["box", "offset"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "offset", "type": "offsetfaces", "cache": {"key": "offset"},
                 "suppressed": suppressed,
                 "values": {"distance": 2.0, "faces": faces}}
            ],
            "connections": [link("box", "offset")]
        }))
        .expect("offset faces request")
    }

    fn two_box_request(faces: serde_json::Value) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["part", "offset"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "big", "type": "primitive", "cache": {"key": "big"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "small", "type": "primitive", "cache": {"key": "small"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 30.0}},
                {"id": "part", "type": "combine", "cache": {"key": "part"}, "values": {}},
                {"id": "offset", "type": "offsetfaces", "cache": {"key": "offset"},
                 "values": {"distance": 2.0, "faces": faces}}
            ],
            "connections": [
                link("big", "part"),
                link("small", "move"),
                link("move", "part"),
                link("part", "offset")
            ]
        }))
        .expect("offset faces request")
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("brep record");
        nodecraft_step::record_volume(&record).expect("brep volume")
    }

    /// Removing the registry entry makes the engine reject `offsetfaces`, so
    /// this test is the entry's fail proof. It also pins the moved cost.
    #[test]
    fn registry_routes_offset_faces() {
        let builtin = crate::nodes::find("offsetfaces").expect("offset faces registered");
        assert_eq!(builtin.node_types(), &["offsetfaces"]);
        let values = serde_json::Value::Null;
        assert_eq!(
            builtin.exact_weight(crate::nodes::NodeView::new("o", "offsetfaces", &values, false)),
            2
        );
    }

    /// No selected faces passes the source Part through unchanged.
    #[test]
    fn no_faces_passes_the_source_part_through() {
        let output = crate::evaluate_exact(
            single_box_request(serde_json::json!([]), false),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let offset_part = output.exact_parts.get("offset").expect("offset exact part");
        assert_eq!(offset_part.bodies.len(), 1);
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&offset_part.bodies[0].brep);
        assert!((after - before).abs() < 1.0e-6, "pass-through volume {after} vs {before}");
    }

    /// Regression for the reviewer's decision: a suppressed Offset Faces is a
    /// shape modifier, so it passes its input Part through. Before the port it
    /// had no `suppression_source` arm and became an empty Part.
    #[test]
    fn suppressed_offset_faces_passes_the_source_part_through() {
        let output = crate::evaluate_exact(
            single_box_request(serde_json::json!([{"id": "body:0:face:1"}]), true),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let offset_part = output.exact_parts.get("offset").expect("offset exact part");
        assert_eq!(
            offset_part.bodies.len(),
            1,
            "a suppressed shape modifier must not become an empty Part",
        );
        let before = volume(&box_part.bodies[0].brep);
        let after = volume(&offset_part.bodies[0].brep);
        assert!((after - before).abs() < 1.0e-6, "suppressed volume {after} vs {before}");
    }

    /// A body-scoped reference edits only that body; the other bodies pass
    /// through in order.
    #[test]
    fn body_scoped_face_references_are_filtered_per_body() {
        let closed = crate::evaluate_exact(
            two_box_request(serde_json::json!([])),
            crate::EngineCache::new(),
            None,
        );
        let opened = crate::evaluate_exact(
            two_box_request(serde_json::json!([{"id": "body:1:face:1"}])),
            crate::EngineCache::new(),
            None,
        );
        assert!(closed.response.node_errors.is_empty(), "{:?}", closed.response.node_errors);
        assert!(opened.response.node_errors.is_empty(), "{:?}", opened.response.node_errors);
        let closed = closed.exact_parts.get("offset").expect("closed part");
        let opened = opened.exact_parts.get("offset").expect("opened part");
        assert_eq!(opened.bodies.len(), 2, "both source bodies survive");
        let closed_first = volume(&closed.bodies[0].brep);
        let opened_first = volume(&opened.bodies[0].brep);
        assert!(
            (closed_first - opened_first).abs() < 1.0e-6,
            "body 0 must not be offset: {closed_first} vs {opened_first}"
        );
        let closed_second = volume(&closed.bodies[1].brep);
        let opened_second = volume(&opened.bodies[1].brep);
        assert!(
            (closed_second - opened_second).abs() > 1.0,
            "body 1 must be offset: {closed_second} vs {opened_second}"
        );
    }

    /// The node keeps the source display layout: it replaces bodies in place,
    /// it does not add or remove them. Before the port it had no `body_layout`
    /// arm and the session could not reproduce the layout.
    #[test]
    fn body_layout_is_the_source_layout() {
        let builtin = crate::nodes::find("offsetfaces").expect("offset faces registered");
        let values = serde_json::json!({"faces": [{"id": "face:1"}]});
        let view = crate::nodes::NodeView::new("offset", "offsetfaces", &values, false);
        assert_eq!(
            builtin.body_layout(view, &[("shape", vec!["big".to_owned(), "small".to_owned()])]),
            Some(vec!["big".to_owned(), "small".to_owned()]),
        );
    }

    /// Offset Faces shares the Shell face-reference resolver: a malformed body
    /// prefix or a non-list `faces` value is an error, not a skipped entry or a
    /// silent pass-through.
    #[test]
    fn malformed_face_references_are_errors_not_silent_passes() {
        for (case, faces, expected) in [
            (
                "malformed body",
                serde_json::json!([{"id": "body:x:face:1"}]),
                "body:x:face:1",
            ),
            (
                "faces is not a list",
                serde_json::json!({"id": "face:0"}),
                "list",
            ),
        ] {
            let output = crate::evaluate_exact(
                single_box_request(faces, false),
                crate::EngineCache::new(),
                None,
            );
            let error = output
                .response
                .node_errors
                .get("offset")
                .cloned()
                .unwrap_or_default();
            assert!(!error.is_empty(), "{case}: no node error");
            assert!(error.contains(expected), "{case}: {error}");
        }
    }
}
