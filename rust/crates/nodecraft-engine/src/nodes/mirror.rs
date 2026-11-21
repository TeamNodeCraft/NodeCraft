//! Mirror: reflect an exact Part through an axis-aligned plane.
//!
//! The reflection maths stays shared: `Part::mirrored` records a
//! `BodyTransform::Mirror` and `mirror_brep_with_kernel` is the one
//! implementation both the Part realisation and the `Mirror` task use. This
//! module owns the shape input, cost, bypass, body layout, repeater role and
//! preview-chain role.

use super::{BuiltinNode, NodeView, PartEval, TransformRole};
use crate::{Part, RequestIndex};

/// Registered for the `mirror` contract id.
pub(crate) struct Mirror;

impl BuiltinNode for Mirror {
    fn node_types(&self) -> &'static [&'static str] {
        &["mirror"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Part mirror has no input".into());
        };
        Ok(source.mirrored(node.values))
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        index.first_incoming_any(&node.id, &["shape", "profile"]).map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        let source = super::input_layout(inputs, "shape")?;
        if crate::mirror_combine(node.values) {
            // The source bodies stay and the reflected copies follow them, in
            // the same order `Part::mirrored` produces.
            let mut combined = source.to_vec();
            combined.extend(source.iter().cloned());
            Some(combined)
        } else {
            Some(source.to_vec())
        }
    }

    fn repeats_input_bodies(&self, node: NodeView<'_>) -> bool {
        // A combining Mirror displays the source and the copy; a non-combining
        // one displays only the reflected body, which is not a repeater.
        crate::mirror_combine(node.values)
    }

    fn transform_role(&self) -> Option<TransformRole> {
        Some(TransformRole::Mirror)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn request(combine: bool) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["mirror"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 20.0}},
                {"id": "mirror", "type": "mirror", "cache": {"key": "mirror"},
                 "values": {"plane": "yz", "offset": 0.0, "combine": combine}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "mirror", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("mirror request")
    }

    fn centres(part: &nodecraft_exact::ExactPart) -> Vec<f64> {
        part.realized_breps()
            .expect("realized copies")
            .iter()
            .map(|brep| {
                let record = nodecraft_step::record_of_brep(brep).expect("record");
                let (min, max) = nodecraft_step::record_bounds(&record).expect("bounds");
                (min[0] + max[0]) / 2.0
            })
            .collect()
    }

    /// Removing the registry entry makes the engine reject `mirror`, so this
    /// test is the entry's fail proof.
    #[test]
    fn registry_routes_mirror_and_its_transform_role() {
        let builtin = crate::nodes::find("mirror").expect("mirror registered");
        assert_eq!(builtin.node_types(), &["mirror"]);
        assert_eq!(builtin.transform_role(), Some(crate::nodes::TransformRole::Mirror));
    }

    /// Combining keeps the source body and appends the reflected copy; without
    /// combining only the reflected copy remains, and only the combining form
    /// is a drag-preview repeater.
    #[test]
    fn combining_and_non_combining_mirror_differ_in_bodies_and_repeater_role() {
        let builtin = crate::nodes::find("mirror").expect("mirror registered");
        let combining_source = vec!["box".to_owned()];
        let combining_values = serde_json::json!({"plane": "yz", "offset": 0.0, "combine": true});
        let combining_view =
            crate::nodes::NodeView::new("mirror", "mirror", &combining_values, false);
        assert!(builtin.repeats_input_bodies(combining_view));
        assert_eq!(
            builtin.body_layout(combining_view, &[("shape", combining_source.clone())]),
            Some(vec!["box".to_owned(), "box".to_owned()]),
        );

        let plain_values = serde_json::json!({"plane": "yz", "offset": 0.0, "combine": false});
        let plain_view = crate::nodes::NodeView::new("mirror", "mirror", &plain_values, false);
        assert!(!builtin.repeats_input_bodies(plain_view));
        assert_eq!(
            builtin.body_layout(plain_view, &[("shape", combining_source)]),
            Some(vec!["box".to_owned()]),
        );

        let combined = crate::evaluate_exact(request(true), crate::EngineCache::new(), None);
        assert!(combined.response.node_errors.is_empty(), "{:?}", combined.response.node_errors);
        let part = combined.exact_parts.get("mirror").expect("combined mirror part");
        assert_eq!(part.bodies.len(), 2, "combine keeps source and copy");
        assert_eq!(centres(part), vec![20.0, -20.0]);

        let plain = crate::evaluate_exact(request(false), crate::EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let part = plain.exact_parts.get("mirror").expect("plain mirror part");
        assert_eq!(part.bodies.len(), 1, "non-combining mirror replaces the source");
        assert_eq!(centres(part), vec![-20.0]);
    }
}
