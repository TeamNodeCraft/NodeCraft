//! Analytic curved-primitive STEP writers (cylinder, cone, sphere, torus).
//!
//! Every solid is centered on the origin in millimeters with `+Z` as its axis, matching
//! smlibpy's `Cylinder`/`Cone`/`Sphere`/`Torus` primitives and the NodeCraft primitive
//! convention. Geometry and topology mirror what smlibpy's `step.py` writes for the
//! same kernel solids, so `smlibpy.import_step(heal=False)` rebuilds the identical
//! B-rep:
//!
//! | Primitive | Faces | Kernel edges | Kernel vertices | STEP curves |
//! |---|---|---|---|---|
//! | cylinder | 3 (lateral + 2 caps) | 2 | 2 | 3 (`CIRCLE` ×2, seam `LINE`) |
//! | cone (frustum) | 3 (lateral + 2 caps) | 2 | 2 | 2 `CIRCLE` + seam `LINE` |
//! | cone (apex) | 2 (lateral + base) | 1 | 2 | 1 `CIRCLE` + seam `LINE` |
//! | sphere | 1 (`SPHERICAL_SURFACE`) | 0 | 2 | 1 polar `CIRCLE` seam |
//! | torus | 1 (`TOROIDAL_SURFACE`) | 0 | 1 | 2 seam `CIRCLE`s |
//!
//! The kernel edge count is smaller than the STEP `EDGE_CURVE` count because seam and
//! degenerate edges are not separate kernel edges; smlibpy reports the same counts for
//! its own exports (validated in `tests/analytic_primitives.rs`).
//!
//! Seam/vertex-loop conventions are the ones smlibpy's writer uses:
//! * the lateral cylinder/cone face loop runs down the seam, around the bottom cap,
//!   back up the seam and (for a top cap) backwards around the top circle;
//! * a sphere is one face whose loop uses the polar seam edge twice (down and up);
//! * a torus is one face whose loop uses the tube seam and the outer-equator seam once
//!   each in both directions.

use std::fs;

use crate::Writer;

const AXIS: [f64; 3] = [0.0, 0.0, 1.0];
const REFERENCE: [f64; 3] = [1.0, 0.0, 0.0];
/// The `-Y` axis used for the sphere/torus tube seam circles (as in smlibpy's writer).
const SEAM_AXIS: [f64; 3] = [0.0, -1.0, 0.0];

/// A cylinder of `radius` and `height`, centered on the origin, axis `+Z`, in mm.
///
/// Faces: bottom disc (plane, normal `-Z`), top disc (plane, normal `+Z`), and one
/// `CYLINDRICAL_SURFACE` whose loop includes the seam line and both cap circles.
pub fn write_cylinder(path: &str, radius: f64, height: f64) -> Result<(), String> {
    positive("cylinder radius", radius)?;
    positive("cylinder height", height)?;
    let half = height / 2.0;

    let mut writer = Writer::default();
    let context = writer.header();
    let lateral = writer.cylindrical_surface([0.0, 0.0, -half], AXIS, REFERENCE, radius)?;
    let seam_line = writer.line([radius, 0.0, -half], [0.0, 0.0, height])?;
    let bottom_circle = writer.circle([0.0, 0.0, -half], AXIS, REFERENCE, radius)?;
    let top_circle = writer.circle([0.0, 0.0, half], AXIS, REFERENCE, radius)?;
    let bottom_plane = writer.plane([0.0, 0.0, -half], [0.0, 0.0, -1.0], REFERENCE)?;
    let top_plane = writer.plane([0.0, 0.0, half], AXIS, REFERENCE)?;

    let bottom_vertex = vertex(&mut writer, [radius, 0.0, -half]);
    let top_vertex = vertex(&mut writer, [radius, 0.0, half]);
    let seam = edge_curve(&mut writer, bottom_vertex, top_vertex, seam_line, true);
    let bottom = edge_curve(&mut writer, bottom_vertex, bottom_vertex, bottom_circle, true);
    let top = edge_curve(&mut writer, top_vertex, top_vertex, top_circle, true);

    let seam_down = oriented_edge(&mut writer, seam, false);
    let bottom_forward = oriented_edge(&mut writer, bottom, true);
    let seam_up = oriented_edge(&mut writer, seam, true);
    let top_backward = oriented_edge(&mut writer, top, false);
    let lateral_loop = edge_loop(&mut writer, &[seam_down, bottom_forward, seam_up, top_backward]);
    let bottom_loop = {
        let edge = oriented_edge(&mut writer, bottom, false);
        edge_loop(&mut writer, &[edge])
    };
    let top_loop = {
        let edge = oriented_edge(&mut writer, top, true);
        edge_loop(&mut writer, &[edge])
    };

    let faces = [
        advanced_face(&mut writer, &[(lateral_loop, true)], lateral, true),
        advanced_face(&mut writer, &[(bottom_loop, true)], bottom_plane, true),
        advanced_face(&mut writer, &[(top_loop, true)], top_plane, true),
    ];
    let shell = closed_shell(&mut writer, &faces);
    let solid = manifold_solid(&mut writer, shell);
    let text = writer.finish_solid(context, "solid", solid);
    fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

/// A cone (or frustum) of `radius_base` at `z = -height/2` and `radius_top` at
/// `z = +height/2`, centered on the origin, axis `+Z`, in mm.
///
/// Either radius may be zero (apex); both zero is an error. Taper direction does not
/// matter: the `CONICAL_SURFACE` is placed at the wide end with its axis pointing away
/// from the narrow end, and the semi-angle is always positive.
pub fn write_cone(
    path: &str,
    radius_base: f64,
    radius_top: f64,
    height: f64,
) -> Result<(), String> {
    non_negative("cone base radius", radius_base)?;
    non_negative("cone top radius", radius_top)?;
    positive("cone height", height)?;
    if radius_base == 0.0 && radius_top == 0.0 {
        return Err("cone needs a non-zero base or top radius".into());
    }
    let half = height / 2.0;
    // Place the CONICAL_SURFACE at the wide end with the axis pointing out of the
    // solid at that end; radius then decreases towards the narrow end.
    let (placement_origin, axis, placement_radius, semi_angle) = if radius_base >= radius_top {
        (
            [0.0, 0.0, -half],
            [0.0, 0.0, -1.0],
            radius_base,
            ((radius_base - radius_top) / height).atan(),
        )
    } else {
        (
            [0.0, 0.0, half],
            AXIS,
            radius_top,
            ((radius_top - radius_base) / height).atan(),
        )
    };

    let mut writer = Writer::default();
    let context = writer.header();
    let lateral = writer.conical_surface(
        placement_origin,
        axis,
        REFERENCE,
        placement_radius,
        semi_angle,
    )?;
    let seam_line = writer.line(
        [radius_base, 0.0, -half],
        [radius_top - radius_base, 0.0, height],
    )?;

    let bottom_vertex = vertex(&mut writer, [radius_base, 0.0, -half]);
    let top_vertex = vertex(&mut writer, [radius_top, 0.0, half]);
    let seam = edge_curve(&mut writer, bottom_vertex, top_vertex, seam_line, true);

    let mut lateral_edges = vec![oriented_edge(&mut writer, seam, false)];
    let mut faces: Vec<u32> = Vec::new();
    if radius_base > 0.0 {
        let bottom_circle = writer.circle([0.0, 0.0, -half], AXIS, REFERENCE, radius_base)?;
        let bottom_plane = writer.plane([0.0, 0.0, -half], [0.0, 0.0, -1.0], REFERENCE)?;
        let bottom = edge_curve(&mut writer, bottom_vertex, bottom_vertex, bottom_circle, true);
        lateral_edges.push(oriented_edge(&mut writer, bottom, true));
        let bottom_loop = {
            let edge = oriented_edge(&mut writer, bottom, false);
            edge_loop(&mut writer, &[edge])
        };
        faces.push(advanced_face(
            &mut writer,
            &[(bottom_loop, true)],
            bottom_plane,
            true,
        ));
    }
    lateral_edges.push(oriented_edge(&mut writer, seam, true));
    if radius_top > 0.0 {
        let top_circle = writer.circle([0.0, 0.0, half], AXIS, REFERENCE, radius_top)?;
        let top_plane = writer.plane([0.0, 0.0, half], AXIS, REFERENCE)?;
        let top = edge_curve(&mut writer, top_vertex, top_vertex, top_circle, true);
        lateral_edges.push(oriented_edge(&mut writer, top, false));
        let top_loop = {
            let edge = oriented_edge(&mut writer, top, true);
            edge_loop(&mut writer, &[edge])
        };
        faces.push(advanced_face(
            &mut writer,
            &[(top_loop, true)],
            top_plane,
            true,
        ));
    }
    let lateral_loop = edge_loop(&mut writer, &lateral_edges);
    faces.insert(
        0,
        advanced_face(&mut writer, &[(lateral_loop, true)], lateral, true),
    );

    let shell = closed_shell(&mut writer, &faces);
    let solid = manifold_solid(&mut writer, shell);
    let text = writer.finish_solid(context, "solid", solid);
    fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

/// A sphere of `radius`, centered on the origin, in mm.
///
/// One `SPHERICAL_SURFACE` face with a polar seam circle (axis `-Y`, as in smlibpy's
/// export): the seam edge runs from the south pole to the north pole and the face loop
/// uses it forwards and backwards.
pub fn write_sphere(path: &str, radius: f64) -> Result<(), String> {
    positive("sphere radius", radius)?;
    let mut writer = Writer::default();
    let context = writer.header();
    let surface = writer.spherical_surface([0.0, 0.0, 0.0], AXIS, REFERENCE, radius)?;
    let seam_circle = writer.circle([0.0, 0.0, 0.0], SEAM_AXIS, REFERENCE, radius)?;

    let south = vertex(&mut writer, [0.0, 0.0, -radius]);
    let north = vertex(&mut writer, [0.0, 0.0, radius]);
    let seam = edge_curve(&mut writer, south, north, seam_circle, true);
    let seam_down = oriented_edge(&mut writer, seam, false);
    let seam_up = oriented_edge(&mut writer, seam, true);
    let loop_id = edge_loop(&mut writer, &[seam_down, seam_up]);
    let face = advanced_face(&mut writer, &[(loop_id, true)], surface, true);
    let shell = closed_shell(&mut writer, &[face]);
    let solid = manifold_solid(&mut writer, shell);
    let text = writer.finish_solid(context, "solid", solid);
    fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

/// A torus of `major_radius` and `minor_radius`, centered on the origin, axis `+Z`, in
/// mm. Requires `major_radius > minor_radius > 0` (ring torus).
///
/// One `TOROIDAL_SURFACE` face with the two natural seam circles (tube circle at the
/// `+X` meridian and the outer-equator circle); the single seam vertex is at
/// `(major_radius + minor_radius, 0, 0)`.
pub fn write_torus(path: &str, major_radius: f64, minor_radius: f64) -> Result<(), String> {
    positive("torus major radius", major_radius)?;
    positive("torus minor radius", minor_radius)?;
    if minor_radius >= major_radius {
        return Err(format!(
            "torus needs major_radius > minor_radius > 0 (got {major_radius} and {minor_radius})"
        ));
    }
    let mut writer = Writer::default();
    let context = writer.header();
    let surface =
        writer.toroidal_surface([0.0, 0.0, 0.0], AXIS, REFERENCE, major_radius, minor_radius)?;
    let tube_circle = writer.circle(
        [major_radius, 0.0, 0.0],
        SEAM_AXIS,
        REFERENCE,
        minor_radius,
    )?;
    let equator_circle = writer.circle(
        [0.0, 0.0, 0.0],
        AXIS,
        REFERENCE,
        major_radius + minor_radius,
    )?;

    let seam_vertex = vertex(&mut writer, [major_radius + minor_radius, 0.0, 0.0]);
    let tube = edge_curve(&mut writer, seam_vertex, seam_vertex, tube_circle, true);
    let equator = edge_curve(&mut writer, seam_vertex, seam_vertex, equator_circle, true);
    let tube_down = oriented_edge(&mut writer, tube, false);
    let equator_forward = oriented_edge(&mut writer, equator, true);
    let tube_up = oriented_edge(&mut writer, tube, true);
    let equator_backward = oriented_edge(&mut writer, equator, false);
    let loop_id = edge_loop(
        &mut writer,
        &[tube_down, equator_forward, tube_up, equator_backward],
    );
    let face = advanced_face(&mut writer, &[(loop_id, true)], surface, true);
    let shell = closed_shell(&mut writer, &[face]);
    let solid = manifold_solid(&mut writer, shell);
    let text = writer.finish_solid(context, "solid", solid);
    fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

// ---------------------------------------------------------------------------
// Topology helpers
// ---------------------------------------------------------------------------

fn vertex(writer: &mut Writer, point: [f64; 3]) -> u32 {
    let point = writer.point(point);
    writer.add(format!("VERTEX_POINT('',#{point})"))
}

fn edge_curve(writer: &mut Writer, start: u32, end: u32, curve: u32, same_sense: bool) -> u32 {
    let sense = if same_sense { ".T." } else { ".F." };
    writer.add(format!(
        "EDGE_CURVE('',#{start},#{end},#{curve},{sense})"
    ))
}

fn oriented_edge(writer: &mut Writer, edge: u32, forward: bool) -> u32 {
    let orientation = if forward { ".T." } else { ".F." };
    writer.add(format!("ORIENTED_EDGE('',*,*,#{edge},{orientation})"))
}

fn edge_loop(writer: &mut Writer, edges: &[u32]) -> u32 {
    let edges = edges.iter().map(|edge| format!("#{edge}")).collect::<Vec<_>>().join(",");
    writer.add(format!("EDGE_LOOP('',({edges}))"))
}

fn advanced_face(
    writer: &mut Writer,
    bounds: &[(u32, bool)],
    surface: u32,
    same_sense: bool,
) -> u32 {
    let mut references = Vec::with_capacity(bounds.len());
    for (loop_id, outer) in bounds {
        let bound = if *outer {
            writer.add(format!("FACE_OUTER_BOUND('',#{loop_id},.T.)"))
        } else {
            writer.add(format!("FACE_BOUND('',#{loop_id},.T.)"))
        };
        references.push(format!("#{bound}"));
    }
    let sense = if same_sense { ".T." } else { ".F." };
    writer.add(format!(
        "ADVANCED_FACE('',({}),#{surface},{sense})",
        references.join(",")
    ))
}

fn closed_shell(writer: &mut Writer, faces: &[u32]) -> u32 {
    let faces = faces.iter().map(|face| format!("#{face}")).collect::<Vec<_>>().join(",");
    writer.add(format!("CLOSED_SHELL('',({faces}))"))
}

fn manifold_solid(writer: &mut Writer, shell: u32) -> u32 {
    writer.add(format!("MANIFOLD_SOLID_BREP('solid',#{shell})"))
}

// ---------------------------------------------------------------------------
// Dimension validation
// ---------------------------------------------------------------------------

fn positive(what: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{what} must be a positive finite number, got {value}"));
    }
    Ok(())
}

fn non_negative(what: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{what} must be a non-negative finite number, got {value}"));
    }
    Ok(())
}
