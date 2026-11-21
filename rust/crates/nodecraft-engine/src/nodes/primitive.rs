//! The primitive family: one exact closed B-rep per `primitiveType`.
//!
//! `solid` and `primitive` are two contract ids for the same family; the
//! primitive kind is a selector inside the node, not a node type. The whole
//! family is one engine code path, so it is one module. Box is the pinned test.

use nodecraft_protocol::Node;
use nodecraft_solidcore::solidcore::topology::Brep;

use super::{BuiltinNode, NodeView, PartEval};
use crate::{Part, RequestIndex};

/// Registered for the two contract ids of the primitive family.
pub(crate) struct Primitive;

impl BuiltinNode for Primitive {
    fn node_types(&self) -> &'static [&'static str] {
        &["solid", "primitive"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        Ok(Part::from_arc(ctx.cached_brep(node)?))
    }

    fn export_terminal(&self) -> bool {
        true
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn mesh_bodies(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // A primitive is a generator: suppressing it turns it off rather than
        // passing another node's geometry through.
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

pub(crate) fn primitive_brep(node: &Node) -> Result<Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    primitive_brep_with_kernel(&mut kernel, node)
}

/// Primitive construction through an engine-owned kernel.
pub(crate) fn primitive_brep_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    node: &Node,
) -> Result<Brep, String> {
    use nodecraft_solidcore::solidcore::math::Point3;
    use nodecraft_solidcore::solidcore::primitive::api_primitives as api;
    let kind = node.values.get("primitiveType").and_then(serde_json::Value::as_str).unwrap_or("");
    // `primitiveType` selects the dynamic field set; every dimension below comes
    // from the contract. `sphereNoPole` is a legacy engine-only kind with no
    // contract field set, so it keeps its historical radius fallback.
    let number = |key: &str| {
        nodecraft_node_contract::resolve_number("primitive", &node.values, key)
            .ok_or_else(|| format!("primitive {kind:?} has no contract field {key:?}"))
    };
    match kind {
        "box" => {
            let (length, width, height) = (number("length")?, number("width")?, number("height")?);
            api::create_box(
                kernel,
                Point3::new(-length / 2.0, -width / 2.0, -height / 2.0),
                length,
                width,
                height,
            )
            .map_err(|error| error.to_string())
        }
        "cylinder" => {
            let (radius, height) = (number("radius")?, number("height")?);
            api::create_cylinder(kernel, Point3::new(0.0, 0.0, -height / 2.0), radius, height)
                .map_err(|error| error.to_string())
        }
        "cone" => {
            let (base, top, height) =
                (number("bottomRadius")?, number("topRadius")?, number("height")?);
            api::create_cone(kernel, Point3::new(0.0, 0.0, -height / 2.0), base, top, height)
                .map_err(|error| error.to_string())
        }
        "sphere" => api::create_sphere(kernel, Point3::new(0.0, 0.0, 0.0), number("radius")?)
            .map_err(|error| error.to_string()),
        "sphereNoPole" => api::create_sphere_no_pole(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("radius").unwrap_or(20.0),
        )
        .map_err(|error| error.to_string()),
        "partialSphere" => api::create_partial_sphere(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("radius")?,
            number("startAngle")?,
            number("endAngle")?,
        )
        .map_err(|error| error.to_string()),
        "wedge" => {
            let body = nodecraft_wedge::wedge_body(
                number("length")?,
                number("width")?,
                number("height")?,
                number("topLength")?,
            )?;
            nodecraft_layers::body_to_kernel(kernel, &body)
                .map_err(|error| format!("Wedge: {error}"))
        }
        "partialTorus" => api::create_partial_torus(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("majorRadius")?,
            number("minorRadius")?,
            number("startAngle")?,
            number("endAngle")?,
        )
        .map_err(|error| error.to_string()),
        "torus" => api::create_torus(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("majorRadius")?,
            number("minorRadius")?,
        )
        .map_err(|error| error.to_string()),
        "pyramid" => {
            let (length, height) = (number("length")?, number("height")?);
            api::create_pyramid(kernel, Point3::new(0.0, 0.0, -height / 2.0), length, height)
                .map_err(|error| error.to_string())
        }
        other => Err(format!("primitive {other:?} has no exact closed-solid BRep")),
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn request() -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["box"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}}
            ],
            "connections": []
        }))
        .expect("box request")
    }

    /// The registry entry is the only path to the primitive family: removing it
    /// makes the engine reject this request before any geometry is built.
    #[test]
    fn registry_routes_the_primitive_family() {
        assert!(crate::nodes::handles("primitive"));
        assert!(crate::nodes::handles("solid"));
        let builtin = crate::nodes::find("primitive").expect("primitive registered");
        assert_eq!(builtin.node_types(), &["solid", "primitive"]);
    }

    /// Box volume and bounds through the registry: 40 x 30 x 20 mm centred on
    /// the origin.
    #[test]
    fn box_through_the_registry_has_known_volume_and_bounds() {
        let output = crate::evaluate_exact(request(), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("box").expect("box exact part");
        assert_eq!(part.bodies.len(), 1, "one box body");
        let brep = &part.bodies[0].brep;
        let record = nodecraft_step::record_of_brep(brep).expect("box record");
        let volume = nodecraft_step::record_volume(&record).expect("box volume");
        assert!((volume - 24_000.0).abs() < 1.0, "40*30*20 mm^3, got {volume}");
        let (min, max) = nodecraft_step::record_bounds(&record).expect("box bounds");
        for (axis, expected) in [(0, 40.0), (1, 30.0), (2, 20.0)] {
            let half = expected / 2.0;
            assert!((min[axis] + half).abs() < 1.0e-6, "min[{axis}] = {}", min[axis]);
            assert!((max[axis] - half).abs() < 1.0e-6, "max[{axis}] = {}", max[axis]);
        }
    }
}
