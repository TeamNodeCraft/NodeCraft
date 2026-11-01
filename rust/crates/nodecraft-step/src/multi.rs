//! Per-solid STEP import: mixed files containing several planar or analytic bodies.

use std::collections::BTreeMap;

use crate::topology::{read_topology, StepBrep};
use crate::{AnalyticSolid, PlanarBody, StepFile};

#[derive(Debug, Clone, PartialEq)]
pub enum StepBody {
    Planar(PlanarBody),
    Analytic(AnalyticSolid),
}

fn numeric_args(entity: &crate::Entity) -> Vec<f64> {
    entity.args.iter().filter_map(|value| match value { crate::Value::Number(number) => Some(*number), _ => None }).collect()
}

/// The direction (third argument) of the `AXIS2_PLACEMENT_3D` referenced by the
/// surface's `placement_index` argument.
///
/// `CYLINDRICAL_SURFACE(name, placement, radius)` and `PLANE(name, placement)` both
/// carry their placement at index 1; the placement's axis direction is its third
/// argument and the direction entity's coordinates are its second argument.
fn surface_axis(file: &StepFile, surface: &crate::Entity, placement_index: usize) -> Option<[f64; 3]> {
    let placement_id = match surface.args.get(placement_index)? {
        crate::Value::Ref(id) => *id,
        _ => return None,
    };
    let placement = file.entities.get(&placement_id)?;
    let axis_id = match placement.args.get(2)? {
        crate::Value::Ref(id) => *id,
        _ => return None,
    };
    let axis = file.entities.get(&axis_id)?;
    let crate::Value::List(values) = axis.args.get(1)? else { return None };
    if values.len() < 3 { return None; }
    let mut direction = [0.0; 3];
    for (slot, value) in direction.iter_mut().zip(values.iter()) {
        let crate::Value::Number(number) = value else { return None };
        *slot = *number;
    }
    Some(direction)
}

/// True when a direction is parallel to the global Z axis (either sign).
fn is_axis_parallel_to_z(direction: [f64; 3]) -> bool {
    let length = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
    length.is_finite() && length > 1e-12 && (direction[2] / length).abs() >= 1.0 - 1e-9
}

/// Radius of a `CYLINDRICAL_SURFACE(name, placement, radius)`.
fn cylinder_radius(surface: &crate::Entity) -> Option<f64> {
    match surface.args.get(2) {
        Some(crate::Value::Number(radius)) => Some(*radius),
        _ => numeric_args(surface).last().copied(),
    }
}

/// Coordinates of a `CARTESIAN_POINT(name, (x, y, z))`.
fn point_coordinates(entity: &crate::Entity) -> Option<[f64; 3]> {
    let crate::Value::List(values) = entity.args.get(1)? else { return None };
    if values.len() < 3 { return None; }
    let mut point = [0.0; 3];
    for (slot, value) in point.iter_mut().zip(values.iter()) {
        let crate::Value::Number(number) = value else { return None };
        *slot = *number;
    }
    Some(point)
}

/// The axis point of the `AXIS2_PLACEMENT_3D` carried by a `PLANE` or
/// `CYLINDRICAL_SURFACE` (its first argument after the name).
///
/// The placement location is a point *on* the cylinder axis, so its X/Y are exact even
/// when the shell only has a seam edge's two vertices (a three-face cylinder) or when
/// the seam splits the lateral surface into half-cylinders.
fn surface_placement_point(file: &StepFile, surface: &crate::Entity) -> Option<[f64; 3]> {
    let placement_id = match surface.args.get(1)? {
        crate::Value::Ref(id) => *id,
        _ => return None,
    };
    let placement = file.entities.get(&placement_id)?;
    let location_id = match placement.args.get(1)? {
        crate::Value::Ref(id) => *id,
        _ => return None,
    };
    point_coordinates(file.entities.get(&location_id)?)
}

/// Recognises a full circular cylinder exported as two (or more) half-cylindrical
/// faces plus two planar end caps, all around the global Z axis.
///
/// CATIA/ST-DEVELOPER writes a full cylinder this way when the seam splits the lateral
/// surface (e.g. `cubsomcy.stp`: two `CYLINDRICAL_SURFACE` faces + two `PLANE` caps).
/// The simple three-face form is handled by the caller first; this path only claims
/// shells whose every face is either a Z-axis cylindrical face or a Z-normal planar
/// cap, with one consistent positive radius.
fn split_full_cylinder(
    file: &StepFile,
    faces: &[&crate::topology::StepFace],
    min: [f64; 3],
    max: [f64; 3],
    scale: f64,
) -> Option<AnalyticSolid> {
    let height = (max[2] - min[2]) * scale;
    if !(height > 0.0 && height.is_finite()) { return None; }
    let mut radius: Option<f64> = None;
    let mut axis_surface: Option<&crate::Entity> = None;
    let mut cylinder_faces = 0usize;
    let mut plane_faces = 0usize;
    for face in faces {
        let surface = file.entities.get(&face.surface)?;
        if surface.has_member("CYLINDRICAL_SURFACE") {
            cylinder_faces += 1;
            if !is_axis_parallel_to_z(surface_axis(file, surface, 1)?) { return None; }
            let face_radius = cylinder_radius(surface)?;
            if !(face_radius > 0.0 && face_radius.is_finite()) { return None; }
            match radius {
                None => {
                    radius = Some(face_radius);
                    axis_surface = Some(surface);
                }
                Some(existing) if (existing - face_radius).abs() <= 1e-9 * existing.abs().max(1.0) => {}
                Some(_) => return None,
            }
        } else if surface.has_member("PLANE") {
            plane_faces += 1;
            if !is_axis_parallel_to_z(surface_axis(file, surface, 1)?) { return None; }
        } else {
            return None;
        }
    }
    // A single lateral face with two caps is the ordinary 3-face cylinder (handled by
    // the caller first); require the split form so this branch cannot shadow other
    // solids.
    if cylinder_faces < 2 || plane_faces != 2 { return None; }
    let radius = radius?;
    let location = surface_placement_point(file, axis_surface?)?;
    Some(AnalyticSolid::Cylinder {
        radius: radius * scale,
        height,
        center: [
            location[0] * scale,
            location[1] * scale,
            (min[2] + max[2]) / 2.0 * scale,
        ],
    })
}

fn shell_vertices(brep: &StepBrep, shell_id: u32) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    let Some(shell) = brep.shells.get(&shell_id) else { return points };
    for face_id in &shell.faces {
        let Some(face) = brep.faces.get(face_id) else { continue };
        for (loop_id, _) in &face.bounds {
            let Some(loop_) = brep.loops.get(loop_id) else { continue };
            for use_ in &loop_.oriented_edges {
                let Some(edge) = brep.edges.get(&use_.edge) else { continue };
                if let Some(vertex) = brep.vertices.get(&edge.start) { points.push(vertex.point); }
                if let Some(vertex) = brep.vertices.get(&edge.end) { points.push(vertex.point); }
            }
        }
    }
    points
}

fn solid_bounds(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    if points.is_empty() { return None; }
    let mut min = [f64::INFINITY; 3]; let mut max = [f64::NEG_INFINITY; 3];
    for point in points { for axis in 0..3 { min[axis] = min[axis].min(point[axis]); max[axis] = max[axis].max(point[axis]); } }
    Some((min, max))
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Unique `CIRCLE` curves used by one shell's edge curves, as `(center, radius)` in
/// model units. Circles are deduped by curve entity id: one circle appears in both the
/// lateral loop and the cap loop of a primitive.
fn shell_circles(file: &StepFile, brep: &StepBrep, shell: &crate::topology::StepShell, scale: f64) -> Vec<([f64; 3], f64)> {
    let mut unique: BTreeMap<u32, ([f64; 3], f64)> = BTreeMap::new();
    for face_id in &shell.faces {
        let Some(face) = brep.faces.get(face_id) else { continue };
        for (loop_id, _) in &face.bounds {
            let Some(loop_) = brep.loops.get(loop_id) else { continue };
            for use_ in &loop_.oriented_edges {
                let Some(edge) = brep.edges.get(&use_.edge) else { continue };
                let Some(curve) = file.entities.get(&edge.curve) else { continue };
                if !curve.has_member("CIRCLE") { continue; }
                let Some(radius) = numeric_args(curve).first().copied() else { continue };
                let Some(center) = surface_placement_point(file, curve) else { continue };
                unique.insert(edge.curve, (center.map(|value| value * scale), radius * scale));
            }
        }
    }
    unique.into_values().collect()
}

fn analytic_body(file: &StepFile, brep: &StepBrep, solid_shell: u32, scale: f64) -> Option<AnalyticSolid> {
    let shell = brep.shells.get(&solid_shell)?;
    let faces: Vec<&crate::topology::StepFace> = shell.faces.iter().filter_map(|id| brep.faces.get(id)).collect();
    let mut names: BTreeMap<&str, usize> = BTreeMap::new();
    for face in &faces { if let Some(surface) = file.entities.get(&face.surface) { *names.entry(surface.name.as_str()).or_default() += 1; } }
    let points = shell_vertices(brep, solid_shell);
    let (min, max) = solid_bounds(&points)?;
    let count = |name: &str| names.get(name).copied().unwrap_or(0);
    // Only a surface belonging to *this* shell may classify this solid. Searching the
    // whole file picked up another part's surface/radius on multi-solid files.
    let surface = |name: &str| faces.iter().filter_map(|face| file.entities.get(&face.surface)).find(|entity| entity.has_member(name));

    if count("CYLINDRICAL_SURFACE") == 1 && count("PLANE") == 2 && faces.len() == 3 {
        let surface = surface("CYLINDRICAL_SURFACE")?;
        let radius = cylinder_radius(surface)? * scale;
        let axis = surface_axis(file, surface, 1).unwrap_or([0.0, 0.0, 1.0]);
        if !is_axis_parallel_to_z(axis) || !finite_positive(radius) { return None; }
        // A genuine three-face primitive cylinder has two cap circles, both with the
        // cylinder radius. Bound them to this shell and derive the height from their
        // centres (the shell bbox can be as flat as 0 for a rotated/misplaced part).
        let circles = shell_circles(file, brep, shell, scale);
        if circles.len() != 2 { return None; }
        let tolerance = radius.abs() * 1.0e-6 + 1.0e-9;
        if circles.iter().any(|(_, circle_radius)| !finite_positive(*circle_radius) || (*circle_radius - radius).abs() > tolerance) {
            return None;
        }
        let zmin = circles.iter().map(|(circle_center, _)| circle_center[2]).fold(f64::INFINITY, f64::min);
        let zmax = circles.iter().map(|(circle_center, _)| circle_center[2]).fold(f64::NEG_INFINITY, f64::max);
        let height = zmax - zmin;
        if !finite_positive(height) { return None; }
        let axis_point = surface_placement_point(file, surface)?;
        let center = [axis_point[0] * scale, axis_point[1] * scale, (zmin + zmax) / 2.0];
        if !center.iter().all(|value| value.is_finite()) { return None; }
        return Some(AnalyticSolid::Cylinder { radius, height, center });
    }
    // Split full-cylinder form: several half-cylindrical lateral faces + two planar
    // caps (CATIA/ST-DEVELOPER seam splitting, e.g. cubsomcy).
    let cylinder_faces = faces.iter().filter(|face| file.entities.get(&face.surface).is_some_and(|surface| surface.has_member("CYLINDRICAL_SURFACE"))).count();
    let plane_faces = faces.iter().filter(|face| file.entities.get(&face.surface).is_some_and(|surface| surface.has_member("PLANE"))).count();
    if cylinder_faces >= 2 && plane_faces == 2 && cylinder_faces + plane_faces == faces.len() {
        if let Some(cylinder) = split_full_cylinder(file, &faces, min, max, scale) { return Some(cylinder); }
    }
    if count("CONICAL_SURFACE") == 1 && count("PLANE") == 2 && faces.len() == 3 {
        let surface = surface("CONICAL_SURFACE")?;
        let axis = surface_axis(file, surface, 1).unwrap_or([0.0, 0.0, 1.0]);
        if !is_axis_parallel_to_z(axis) { return None; }
        let mut circles = shell_circles(file, brep, shell, scale);
        if circles.len() != 2 { return None; }
        // Base/top follow the cap circle Z order, not the radius order (an inverted
        // frustum has its smaller radius at the bottom).
        circles.sort_by(|a, b| a.0[2].partial_cmp(&b.0[2]).unwrap_or(std::cmp::Ordering::Equal));
        let radius_base = circles[0].1;
        let radius_top = circles[1].1;
        let height = circles[1].0[2] - circles[0].0[2];
        if !(radius_base.is_finite() && radius_base >= 0.0)
            || !(radius_top.is_finite() && radius_top >= 0.0)
            || (radius_base <= 0.0 && radius_top <= 0.0)
            || !finite_positive(height)
        {
            return None;
        }
        let axis_point = surface_placement_point(file, surface)?;
        let center = [axis_point[0] * scale, axis_point[1] * scale, (circles[0].0[2] + circles[1].0[2]) / 2.0];
        if !center.iter().all(|value| value.is_finite()) { return None; }
        return Some(AnalyticSolid::Cone { radius_base, radius_top, height, center });
    }
    if count("SPHERICAL_SURFACE") == 1 && faces.len() == 1 {
        let surface = surface("SPHERICAL_SURFACE")?;
        let radius = numeric_args(surface).first().copied()? * scale;
        if !finite_positive(radius) { return None; }
        let center = surface_placement_point(file, surface)?.map(|value| value * scale);
        if !center.iter().all(|value| value.is_finite()) { return None; }
        return Some(AnalyticSolid::Sphere { radius, center });
    }
    if count("TOROIDAL_SURFACE") == 1 && faces.len() == 1 {
        let surface = surface("TOROIDAL_SURFACE")?;
        let values = numeric_args(surface);
        if values.len() < 2 { return None; }
        let major = values[0] * scale;
        let minor = values[1] * scale;
        if !finite_positive(major) || !finite_positive(minor) { return None; }
        let center = surface_placement_point(file, surface)?.map(|value| value * scale);
        if !center.iter().all(|value| value.is_finite()) { return None; }
        return Some(AnalyticSolid::Torus { major, minor, center });
    }
    None
}

fn planar_body(file: &StepFile, brep: &StepBrep, shell_id: u32, scale: f64, tolerance: f64) -> Result<PlanarBody, String> {
    let shell = brep.shells.get(&shell_id).ok_or("missing shell")?;
    let mut body = PlanarBody::default();
    let mut lookup: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    for face_id in &shell.faces {
        let face = brep.faces.get(face_id).ok_or("missing face")?;
        let surface = file.entities.get(&face.surface).ok_or("missing surface")?;
        if surface.name != "PLANE" { return Err(format!("curved surface {}", surface.name)); }
        let mut loops: Vec<(Vec<usize>, bool)> = Vec::new();
        for (loop_id, orientation) in &face.bounds {
            let loop_ = brep.loops.get(loop_id).ok_or("missing loop")?;
            let mut points = Vec::new();
            for use_ in &loop_.oriented_edges {
                let edge = brep.edges.get(&use_.edge).ok_or("missing edge")?;
                let vertex = if use_.orientation { edge.start } else { edge.end };
                let vertex = brep.vertices.get(&vertex).ok_or("missing vertex")?;
                let point = [vertex.point[0]*scale, vertex.point[1]*scale, vertex.point[2]*scale];
                let key = [(point[0]*1e7).round() as i64, (point[1]*1e7).round() as i64, (point[2]*1e7).round() as i64];
                let index = *lookup.entry(key).or_insert_with(|| { body.points.push(point); body.points.len()-1 });
                if points.last().copied() != Some(index) { points.push(index); }
            }
            if points.first() == points.last() { points.pop(); }
            if !*orientation { points.reverse(); }
            if points.len() < 3 { return Err("short loop".into()); }
            loops.push((points, face.outer_bounds.contains(loop_id)));
        }
        if loops.is_empty() { continue; }
        let outer_index = loops.iter().position(|(_, outer)| *outer).unwrap_or(0);
        let outer = loops.remove(outer_index).0;
        body.push_face(outer, loops.into_iter().map(|(points, _)| points).collect());
    }
    body.weld_points(tolerance);
    if body.volume().is_finite() && body.volume() < 0.0 { body.repair_orientation()?; }
    Ok(body)
}

/// Imports each `MANIFOLD_SOLID_BREP` as a planar body or analytic primitive.
pub fn import_solids(path: &str) -> Result<Vec<StepBody>, String> {
    let (mut bodies, skipped) = import_solids_partial(path)?;
    if !skipped.is_empty() { return Err(format!("solid(s) {skipped:?} are not planar or analytic primitives")); }
    bodies.shrink_to_fit();
    Ok(bodies)
}

/// Imports the planar/analytic solids and reports solids that need a special reader.
pub fn import_solids_partial(path: &str) -> Result<(Vec<StepBody>, Vec<u32>), String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = crate::unit_scale(&text);
    let tolerance = crate::read_tolerance(path).ok().flatten().unwrap_or(1e-7).clamp(1e-9, 1e-3);
    let file = StepFile::parse(&text)?;
    let brep = read_topology(&file);
    if brep.solids.is_empty() { return Err("STEP file has no MANIFOLD_SOLID_BREP".into()); }
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for solid in brep.solids.values() {
        if let Some(analytic) = analytic_body(&file, &brep, solid.shell, scale) { out.push(StepBody::Analytic(analytic)); continue; }
        match planar_body(&file, &brep, solid.shell, scale, tolerance) {
            Ok(body) => out.push(StepBody::Planar(body)),
            Err(_) => skipped.push(solid.id),
        }
    }
    Ok((out, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Option<String> {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for directory in ["Sample AP203 Data Files"] {
            let path = manifest.join("../../../tests").join(directory).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
        }
        None
    }

    fn analytic_cylinder(body: &StepBody) -> (f64, f64, [f64; 3]) {
        match body {
            StepBody::Analytic(AnalyticSolid::Cylinder { radius, height, center }) => (*radius, *height, *center),
            other => panic!("expected an analytic cylinder, got {other:?}"),
        }
    }

    /// Regression for Step-101: `cubsomcy.stp` stores its cylinder as two half-cylindrical
    /// faces plus two planar caps. The three-face detector used to skip that solid, so the
    /// Stepio faceted path silently imported only the faceted cube (bbox X max 50).
    #[test]
    fn cubsomcy_split_cylinder_is_detected() {
        let Some(path) = fixture("cubsomcy.stp") else { return };
        let (bodies, skipped) = import_solids_partial(&path).unwrap();
        assert!(skipped.is_empty(), "split cylinder was not imported, skipped {skipped:?}");
        assert_eq!(bodies.len(), 1, "expected only the MANIFOLD_SOLID_BREP lump");
        let (radius, height, center) = analytic_cylinder(&bodies[0]);
        assert!((radius - 25.0).abs() < 1e-9, "radius {radius}");
        assert!((height - 35.0).abs() < 1e-9, "height {height}");
        assert!((center[0] - 200.0).abs() < 1e-9, "center {center:?}");
        assert!(center[1].abs() < 1e-9, "center {center:?}");
        assert!((center[2] - 17.5).abs() < 1e-9, "center {center:?}");
        // The tessellated analytic cylinder spans x 175..225, y -25..25, z 0..35.
        assert!((center[0] - radius - 175.0).abs() < 1e-9);
        assert!((center[0] + radius - 225.0).abs() < 1e-9);
        assert!((center[1] - radius + 25.0).abs() < 1e-9);
        assert!((center[1] + radius - 25.0).abs() < 1e-9);
        assert!((center[2] - height / 2.0).abs() < 1e-9);
        assert!((center[2] + height / 2.0 - 35.0).abs() < 1e-9);
    }

    /// A plain three-face cylinder must keep going through the original, simpler branch.
    #[test]
    fn simple_three_face_cylinder_still_detected() {
        let path = std::env::temp_dir().join(format!("sn-multi-cylinder-{}.step", std::process::id()));
        crate::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).unwrap();
        let (bodies, skipped) = import_solids_partial(path.to_str().unwrap()).unwrap();
        assert!(skipped.is_empty(), "skipped {skipped:?}");
        assert_eq!(bodies.len(), 1);
        let (radius, height, center) = analytic_cylinder(&bodies[0]);
        assert!((radius - 5.0).abs() < 1e-9, "radius {radius}");
        assert!((height - 10.0).abs() < 1e-9, "height {height}");
        assert!(center.iter().all(|value| value.abs() < 1e-9), "center {center:?}");
        let _ = std::fs::remove_file(path);
    }

    /// Mixed planar/cylindrical solids (doghouse style) have many planar faces, so the
    /// split-cylinder branch must not claim them.
    #[test]
    fn mixed_planar_cylinder_solid_is_not_misdetected_as_cylinder() {
        let Some(path) = fixture("doghouse.stp") else { return };
        let (bodies, skipped) = import_solids_partial(&path).unwrap();
        assert!(
            bodies.iter().all(|body| !matches!(body, StepBody::Analytic(AnalyticSolid::Cylinder { .. }))),
            "doghouse unexpectedly became a bare cylinder: {bodies:?}"
        );
        assert!(!skipped.is_empty(), "mixed doghouse solid should stay skipped, bodies {bodies:?}");
    }
}
