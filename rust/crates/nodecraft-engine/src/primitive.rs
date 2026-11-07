//! Primitive B-rep construction and placement helpers.

use nodecraft_protocol::Node;

pub(crate) fn copy_brep(brep: &nodecraft_solidcore::solidcore::Brep) -> Result<nodecraft_solidcore::solidcore::Brep, String> {
    brep.clone_standing_for_self()
        .map_err(|error| format!("B-rep copy: {error}"))
}

pub(crate) fn apply_brep_placement(
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    offset: [f64; 3],
    rotation: [f64; 3],
) -> Result<(), String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    apply_brep_placement_with_kernel(&mut kernel, brep, offset, rotation)
}

pub(crate) fn apply_brep_placement_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    offset: [f64; 3],
    rotation: [f64; 3],
) -> Result<(), String> {
    use nodecraft_solidcore::solidcore::geom::axis2_placement::Axis2Placement;
    use nodecraft_solidcore::solidcore::math::{Point3, Vec3};
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    let x_axis = Vec3::new(cy * cz, cy * sz, -sy);
    let y_axis = Vec3::new(sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy);
    let placement = Axis2Placement::new(Point3::new(offset[0], offset[1], offset[2]), x_axis, y_axis);
    nodecraft_solidcore::solidcore::topology::brep::p5::transform(kernel, brep, &placement, None, true)
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(crate) fn primitive_brep(node: &Node) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    primitive_brep_with_kernel(&mut kernel, node)
}

/// Primitive construction through an engine-owned kernel.
pub(crate) fn primitive_brep_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    node: &Node,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    use nodecraft_solidcore::solidcore::math::Point3;
    use nodecraft_solidcore::solidcore::primitive::api_primitives as api;
    let kind = node.values.get("primitiveType").and_then(serde_json::Value::as_str).unwrap_or("");
    let number = |key: &str, default: f64| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(default);
    match kind {
        "box" => {
            let (length, width, height) = (number("length", 40.0), number("width", 40.0), number("height", 40.0));
            api::create_box(kernel, Point3::new(-length / 2.0, -width / 2.0, -height / 2.0), length, width, height).map_err(|error| error.to_string())
        }
        "cylinder" => {
            let (radius, height) = (number("radius", 20.0), number("height", 40.0));
            api::create_cylinder(kernel, Point3::new(0.0, 0.0, -height / 2.0), radius, height).map_err(|error| error.to_string())
        }
        "cone" => {
            let (base, top, height) = (number("bottomRadius", 20.0), number("topRadius", 0.0), number("height", 40.0));
            api::create_cone(kernel, Point3::new(0.0, 0.0, -height / 2.0), base, top, height).map_err(|error| error.to_string())
        }
        "sphere" => api::create_sphere(kernel, Point3::new(0.0, 0.0, 0.0), number("radius", 20.0)).map_err(|error| error.to_string()),
        "sphereNoPole" => api::create_sphere_no_pole(kernel, Point3::new(0.0, 0.0, 0.0), number("radius", 20.0)).map_err(|error| error.to_string()),
        "partialSphere" => api::create_partial_sphere(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("radius", 20.0),
            number("startAngle", 0.0),
            number("endAngle", 90.0),
        )
        .map_err(|error| error.to_string()),
        "wedge" => {
            let body = nodecraft_wedge::wedge_body(
                number("length", 40.0),
                number("width", 30.0),
                number("height", 25.0),
                number("topLength", 0.0),
            )?;
            nodecraft_layers::body_to_kernel(kernel, &body)
                .map_err(|error| format!("Wedge: {error}"))
        }
        "partialTorus" => api::create_partial_torus(
            kernel,
            Point3::new(0.0, 0.0, 0.0),
            number("majorRadius", 15.0),
            number("minorRadius", 5.0),
            number("startAngle", 0.0),
            number("endAngle", 90.0),
        )
        .map_err(|error| error.to_string()),
        "torus" => api::create_torus(kernel, Point3::new(0.0, 0.0, 0.0), number("majorRadius", 15.0), number("minorRadius", 5.0)).map_err(|error| error.to_string()),
        "pyramid" => {
            let (length, height) = (number("length", 40.0), number("height", 30.0));
            api::create_pyramid(kernel, Point3::new(0.0, 0.0, -height / 2.0), length, height).map_err(|error| error.to_string())
        }
        other => Err(format!("primitive {other:?} has no exact closed-solid BRep")),
    }
}

pub(crate) fn snapshot_brep(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(&mut kernel, brep)
        .map_err(|error| error.to_string())
}

