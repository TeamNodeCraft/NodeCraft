//! Look: apply one colour to every face of an exact Part.
//!
//! The colour attribute path stays shared in `crate::look`: the same attribute
//! id that display reads and that a Boolean copies through regions. The node
//! owns only the shape input, the `LookMany` task and the instance bookkeeping
//! that keeps the source body plan.

use super::{BuiltinNode, NodeView, PartEval};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `look` contract id.
pub(crate) struct Look;

impl BuiltinNode for Look {
    fn node_types(&self) -> &'static [&'static str] {
        &["look"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape", "a"])? else {
            return Err("Look has no shape input".into());
        };
        let color = crate::look::parse_look_color(node.values)?;
        let breps: Vec<std::sync::Arc<_>> =
            source.bodies.iter().map(|body| body.body.shared()).collect();
        let looked = crate::node_task::execute_exact(
            ExactNodeTask::LookMany { breps, colour: color },
            node.id,
            crate::fnv_seed(node.id),
        )?
        .into_shapes()
        .ok_or_else(|| "Look task returned no shapes".to_owned())?;
        if looked.len() != source.bodies.len() {
            return Err("Look task returned the wrong body count".to_owned());
        }
        let bodies = source
            .bodies
            .iter()
            .zip(looked)
            .map(|(body, brep)| crate::BodyInstance {
                body: brep,
                placement: body.placement.clone(),
                // A Look copies the B-rep, so a realised body of the source is
                // not this body's; the new instance realises its own.
                realized: Default::default(),
                source: body.source,
                source_body: body.source_body,
                name: body.name.clone(),
                color: Some(color),
            })
            .collect();
        Ok(Part { bodies, sketches: source.sketches.clone() })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        2
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        index.first_incoming_any(&node.id, &["shape", "profile"]).map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        super::input_layout(inputs, "shape").map(<[String]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::{Mesh, Request};

    fn request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["look"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 20.0}},
                {"id": "look", "type": "look", "cache": {"key": "look"},
                 "values": {"color": "#ff0000"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "look", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("look request")
    }

    /// Removing the registry entry makes the engine reject `look`, so this test
    /// is the entry's fail proof. It reads the colour back through the shared
    /// face-attribute path and pins source-plan reuse.
    #[test]
    fn registry_applies_colour_through_the_shared_attribute_path() {
        let builtin = crate::nodes::find("look").expect("look registered");
        assert_eq!(builtin.node_types(), &["look"]);

        let output = crate::evaluate_exact(request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let box_part = output.exact_parts.get("box").expect("box exact part");
        let look_part = output.exact_parts.get("look").expect("look exact part");
        assert_eq!(look_part.bodies.len(), 1);

        // A Look copies the B-rep but keeps the producing body key, so every
        // instance of one source still shares one display plan.
        assert_eq!(
            look_part.bodies[0].key, box_part.bodies[0].key,
            "Look dropped the source body plan",
        );

        // The placement stays on the instance.
        let brep = &look_part.bodies[0].brep;
        let realized = look_part.realized_breps().expect("realized look").pop().expect("one body");
        let record = nodecraft_step::record_of_brep(&realized).expect("look record");
        let (min, max) = nodecraft_step::record_bounds(&record).expect("look bounds");
        assert!((min[0] - 15.0).abs() < 1.0e-6, "{min:?}");
        assert!((max[0] - 25.0).abs() < 1.0e-6, "{max:?}");

        // Read the colour through the one shared attribute reader: every face
        // group named `face:N` picks up the colour the Look wrote.
        let faces = brep.face_list().len();
        let mut mesh = Mesh::default();
        mesh.face_groups = (0..faces).map(|index| format!("body:0:face:{index}")).collect();
        mesh.face_colors = vec![[0.32, 0.34, 0.37]; faces];
        crate::look::apply_brep_face_colors(brep, &mut mesh);
        assert_eq!(mesh.face_colors.len(), faces);
        for color in &mesh.face_colors {
            for (channel, expected) in color.iter().zip([1.0_f32, 0.0, 0.0]) {
                assert!((channel - expected).abs() < 1.0e-6, "{color:?}");
            }
        }
    }
}
