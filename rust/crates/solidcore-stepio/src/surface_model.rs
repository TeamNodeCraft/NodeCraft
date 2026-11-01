//! Display-mesh import for STEP surface models.
//!
//! `gear.stp` is a valid STEP model made of `CURVE_BOUNDED_SURFACE` faces over
//! planes and cylinders, with no solid shell (`MANIFOLD_SOLID_BREP`). The solid
//! readers cannot represent it as a Brep, so this module samples each bounded
//! surface's 3D boundary curves, projects them into the surface's parameter space,
//! and emits a conservative display mesh clipped to those boundaries (including
//! inner boundary loops).
//!
//! This is explicitly a display import: it returns no exact Brep and no mass
//! properties. Triangulation is a parameter-grid clip, so the boundary is stair-stepped
//! at the grid resolution rather than an exact trim. Callers should report it as a
//! surface-model fallback.

use nodecraft_protocol::Mesh;
use nodecraft_step::{StepFile, Value};

type P3 = [f64; 3];

fn entity<'a>(file: &'a StepFile, id: u32) -> Result<&'a nodecraft_step::Entity, String> {
    file.entities.get(&id).ok_or_else(|| format!("missing entity #{id}"))
}

fn reference(value: &Value) -> Result<u32, String> {
    match value { Value::Ref(id) => Ok(*id), _ => Err("expected a reference".into()) }
}

fn point(file: &StepFile, id: u32) -> Result<P3, String> {
    let entity = entity(file, id)?;
    let Some(Value::List(values)) = entity.args.get(1) else { return Err(format!("#{id} is not a point")) };
    let mut result = [0.0; 3];
    for (slot, value) in result.iter_mut().zip(values) {
        let Value::Number(number) = value else { return Err(format!("#{id} has a non-numeric coordinate")) };
        *slot = *number;
    }
    Ok(result)
}

fn direction(file: &StepFile, id: u32) -> Result<P3, String> {
    let entity = entity(file, id)?;
    let Some(Value::List(values)) = entity.args.get(1) else { return Err(format!("#{id} is not a direction")) };
    let mut result = [0.0; 3];
    for (slot, value) in result.iter_mut().zip(values) {
        let Value::Number(number) = value else { return Err(format!("#{id} has a non-numeric direction")) };
        *slot = *number;
    }
    Ok(result)
}

fn add(a: P3, b: P3) -> P3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: P3, b: P3) -> P3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn mul(a: P3, k: f64) -> P3 { [a[0] * k, a[1] * k, a[2] * k] }
fn dot(a: P3, b: P3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: P3, b: P3) -> P3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn unit(a: P3) -> Result<P3, String> {
    let length = dot(a, a).sqrt();
    if length <= 1e-15 { return Err("zero-length direction".into()); }
    Ok(mul(a, 1.0 / length))
}

fn placement(file: &StepFile, id: u32) -> Result<(P3, P3, P3), String> {
    let entity = entity(file, id)?;
    let refs: Vec<u32> = entity.args.iter().filter_map(|value| if let Value::Ref(id) = value { Some(*id) } else { None }).collect();
    let origin = point(file, *refs.first().ok_or("placement has no origin")?)?;
    let axis = unit(direction(file, *refs.get(1).ok_or("placement has no axis")?)?)?;
    let x_ref = if let Some(id) = refs.get(2) { direction(file, *id)? } else { [1.0, 0.0, 0.0] };
    let x = unit(sub(x_ref, mul(axis, dot(x_ref, axis))))?;
    Ok((origin, axis, x))
}

#[derive(Clone, Copy)]
enum Basis {
    Plane { origin: P3, x: P3, y: P3, normal: P3 },
    Cylinder { origin: P3, x: P3, y: P3, axis: P3, radius: f64 },
    Sphere { origin: P3, x: P3, y: P3, axis: P3, radius: f64 },
    Torus { origin: P3, x: P3, y: P3, axis: P3, major: f64, minor: f64 },
}

impl Basis {
    fn project(self, point: P3) -> [f64; 2] {
        match self {
            Self::Plane { origin, x, y, .. } => { let delta = sub(point, origin); [dot(delta, x), dot(delta, y)] }
            Self::Cylinder { origin, x, y, axis, .. } => { let delta = sub(point, origin); [dot(delta, y).atan2(dot(delta, x)).to_degrees(), dot(delta, axis)] }
            Self::Sphere { origin, x, y, axis, radius } => {
                let delta = sub(point, origin);
                let z = (dot(delta, axis) / radius).clamp(-1.0, 1.0);
                [dot(delta, y).atan2(dot(delta, x)).to_degrees(), z.asin().to_degrees()]
            }
            Self::Torus { origin, x, y, axis, major, .. } => {
                let delta = sub(point, origin);
                let radial = dot(delta, x).hypot(dot(delta, y));
                [dot(delta, y).atan2(dot(delta, x)).to_degrees(), dot(delta, axis).atan2(radial - major).to_degrees()]
            }
        }
    }
    fn position(self, uv: [f64; 2]) -> P3 {
        match self {
            Self::Plane { origin, x, y, .. } => add(origin, add(mul(x, uv[0]), mul(y, uv[1]))),
            Self::Cylinder { origin, x, y, axis, radius } => {
                let angle = uv[0].to_radians();
                add(origin, add(add(mul(x, radius * angle.cos()), mul(y, radius * angle.sin())), mul(axis, uv[1])))
            }
            Self::Sphere { origin, x, y, axis, radius } => {
                let (u, v) = (uv[0].to_radians(), uv[1].to_radians());
                add(origin, mul(add(add(mul(x, v.cos() * u.cos()), mul(y, v.cos() * u.sin())), mul(axis, v.sin())), radius))
            }
            Self::Torus { origin, x, y, axis, major, minor } => {
                let (u, v) = (uv[0].to_radians(), uv[1].to_radians());
                let radial = add(mul(x, u.cos()), mul(y, u.sin()));
                add(origin, add(mul(radial, major + minor * v.cos()), mul(axis, minor * v.sin())))
            }
        }
    }
    fn normal(self, uv: [f64; 2]) -> P3 {
        match self {
            Self::Plane { normal, .. } => normal,
            Self::Cylinder { x, y, .. } => { let angle = uv[0].to_radians(); add(mul(x, angle.cos()), mul(y, angle.sin())) }
            Self::Sphere { x, y, axis, .. } => {
                let (u, v) = (uv[0].to_radians(), uv[1].to_radians());
                add(add(mul(x, v.cos() * u.cos()), mul(y, v.cos() * u.sin())), mul(axis, v.sin()))
            }
            Self::Torus { x, y, axis, .. } => {
                let (u, v) = (uv[0].to_radians(), uv[1].to_radians());
                let radial = add(mul(x, u.cos()), mul(y, u.sin()));
                add(mul(radial, v.cos()), mul(axis, v.sin()))
            }
        }
    }
}

fn basis(file: &StepFile, surface_id: u32) -> Result<Basis, String> {
    let surface = entity(file, surface_id)?;
    match surface.name.as_str() {
        "PLANE" => {
            let (origin, normal, x_ref) = placement(file, reference(surface.args.get(1).ok_or("plane has no placement")?)?)?;
            let x = x_ref;
            let y = unit(cross(normal, x))?;
            Ok(Basis::Plane { origin, x, y, normal })
        }
        "CYLINDRICAL_SURFACE" => {
            let (origin, axis, x_ref) = placement(file, reference(surface.args.get(1).ok_or("cylinder has no placement")?)?)?;
            let Some(Value::Number(radius)) = surface.args.get(2) else { return Err("cylinder has no radius".into()) };
            let y = unit(cross(axis, x_ref))?;
            Ok(Basis::Cylinder { origin, x: x_ref, y, axis, radius: *radius })
        }
        "SPHERICAL_SURFACE" => {
            let (origin, axis, x_ref) = placement(file, reference(surface.args.get(1).ok_or("sphere has no placement")?)?)?;
            let Some(Value::Number(radius)) = surface.args.get(2) else { return Err("sphere has no radius".into()) };
            let y = unit(cross(axis, x_ref))?;
            Ok(Basis::Sphere { origin, x: x_ref, y, axis, radius: *radius })
        }
        "TOROIDAL_SURFACE" => {
            let (origin, axis, x_ref) = placement(file, reference(surface.args.get(1).ok_or("torus has no placement")?)?)?;
            let Some(Value::Number(major)) = surface.args.get(2) else { return Err("torus has no major radius".into()) };
            let Some(Value::Number(minor)) = surface.args.get(3) else { return Err("torus has no minor radius".into()) };
            let y = unit(cross(axis, x_ref))?;
            Ok(Basis::Torus { origin, x: x_ref, y, axis, major: *major, minor: *minor })
        }
        other => Err(format!("unsupported surface-model face type {other}")),
    }
}

fn flag(entity: &nodecraft_step::Entity, index: usize) -> bool {
    matches!(entity.args.get(index), Some(Value::Enum(value)) if value.eq_ignore_ascii_case("T"))
}

fn sample_line(file: &StepFile, base_id: u32, start: f64, end: f64) -> Result<Vec<P3>, String> {
    let base = entity(file, base_id)?;
    match base.name.as_str() {
        "LINE" => {
            let origin = point(file, reference(base.args.get(1).ok_or("line has no point")?)?)?;
            let vector_id = reference(base.args.get(2).ok_or("line has no vector")?)?;
            let vector = entity(file, vector_id)?;
            let dir = unit(direction(file, reference(vector.args.get(1).ok_or("vector has no direction")?)?)?)?;
            let Some(Value::Number(magnitude)) = vector.args.get(2) else { return Err("vector has no magnitude".into()) };
            let mut points = Vec::with_capacity(25);
            for index in 0..=24 { let t = start + (end - start) * index as f64 / 24.0; points.push(add(origin, mul(dir, *magnitude * t))); }
            Ok(points)
        }
        "CIRCLE" => {
            let (origin, axis, x_ref) = placement(file, reference(base.args.get(1).ok_or("circle has no placement")?)?)?;
            let Some(Value::Number(radius)) = base.args.get(2) else { return Err("circle has no radius".into()) };
            let y = unit(cross(axis, x_ref))?;
            let mut points = Vec::with_capacity(25);
            for index in 0..=24 {
                let angle = (start + (end - start) * index as f64 / 24.0).to_radians();
                points.push(add(origin, add(mul(x_ref, *radius * angle.cos()), mul(y, *radius * angle.sin()))));
            }
            Ok(points)
        }
        other => Err(format!("unsupported surface-model boundary curve {other}")),
    }
}

fn trimmed_parameter(file: &StepFile, value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(number) => Some(*number),
        Value::List(values) => values.first().and_then(|value| if let Value::Number(number) = value { Some(*number) } else { None }),
        Value::Ref(id) => file.entities.get(id).and_then(|entity| entity.args.first()).and_then(|value| if let Value::Number(number) = value { Some(*number) } else { None }),
        _ => None,
    }
}

fn sample_curve_range(file: &StepFile, id: u32, start: f64, end: f64) -> Result<Vec<P3>, String> {
    let curve_entity = entity(file, id)?;
    match curve_entity.name.as_str() {
        "LINE" | "CIRCLE" => sample_line(file, id, start, end),
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" | "" => {
            let curve = step_nurbs::parse_curve(file, id)?;
            let (low, high) = curve.domain();
            let (first, last) = if (start, end) == (0.0, 1.0) { (low, high) } else { (start, end) };
            let mut points = Vec::with_capacity(25);
            for index in 0..=24 {
                let parameter = first + (last - first) * index as f64 / 24.0;
                let point = step_nurbs::curve_point(&curve, parameter)?;
                points.push(point);
            }
            Ok(points)
        }
        other => Err(format!("unsupported surface-model boundary curve {other}")),
    }
}

fn sample_curve(file: &StepFile, id: u32) -> Result<Vec<P3>, String> {
    let curve_entity = entity(file, id)?;
    match curve_entity.name.as_str() {
        "TRIMMED_CURVE" => {
            let base = reference(curve_entity.args.get(1).ok_or("trimmed curve has no basis")?)?;
            let start = trimmed_parameter(file, curve_entity.args.get(2)).unwrap_or(0.0);
            let end = trimmed_parameter(file, curve_entity.args.get(3)).unwrap_or(1.0);
            let sense = flag(curve_entity, 4);
            let mut points = sample_curve_range(file, base, start, end)?;
            if !sense { points.reverse(); }
            Ok(points)
        }
        "LINE" | "CIRCLE" => {
            let end = if curve_entity.name == "CIRCLE" { 360.0 } else { 1.0 };
            sample_curve_range(file, id, 0.0, end)
        }
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" | "" => {
            let curve = step_nurbs::parse_curve(file, id)?;
            let (low, high) = curve.domain();
            sample_curve_range(file, id, low, high)
        }
        other => Err(format!("unsupported surface-model boundary curve {other}")),
    }
}

fn boundary_loops(file: &StepFile, face_id: u32) -> Result<Vec<Vec<P3>>, String> {
    let face = entity(file, face_id)?;
    let Some(Value::List(boundaries)) = face.args.get(2) else { return Err(format!("face #{face_id} has no boundary set")) };
    let mut loops = Vec::new();
    for boundary in boundaries {
        let boundary_id = reference(boundary)?;
        let boundary = entity(file, boundary_id)?;
        let Some(Value::List(segments)) = boundary.args.get(1) else { return Err(format!("boundary #{boundary_id} has no segments")) };
        let mut points: Vec<P3> = Vec::new();
        for segment in segments {
            let segment = entity(file, reference(segment)?)?;
            let curve = reference(segment.args.get(2).ok_or("composite segment has no curve")?)?;
            let mut sampled = sample_curve(file, curve)?;
            if !flag(segment, 1) { sampled.reverse(); }
            for point in sampled {
                if points.last().is_none_or(|last| dot(sub(*last, point), sub(*last, point)) > 1e-12) { points.push(point); }
            }
        }
        if points.len() >= 3 { loops.push(points); }
    }
    Ok(loops)
}

fn signed_area(polygon: &[[f64; 2]]) -> f64 {
    let mut area = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area / 2.0
}

fn point_in_polygon(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[previous];
        if (a[1] > point[1]) != (b[1] > point[1]) && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0] { inside = !inside; }
        previous = index;
    }
    inside
}

fn unwrap_loop(points: &mut [[f64; 2]]) {
    if points.is_empty() { return; }
    let mut previous = points[0][0];
    for point in points.iter_mut().skip(1) {
        while point[0] - previous > 180.0 { point[0] -= 360.0; }
        while point[0] - previous < -180.0 { point[0] += 360.0; }
        previous = point[0];
    }
}

fn face_mesh(file: &StepFile, face_id: u32) -> Result<Mesh, String> {
    let face = entity(file, face_id)?;
    let surface = basis(file, reference(face.args.get(1).ok_or("face has no basis surface")?)?)?;
    let mut loops = boundary_loops(file, face_id)?;
    if loops.is_empty() { return Err(format!("face #{face_id} has no boundary loops")); }
    let mut projected: Vec<Vec<[f64; 2]>> = loops.drain(..).map(|points| {
        let mut polygon: Vec<[f64; 2]> = points.into_iter().map(|point| surface.project(point)).collect();
        if matches!(surface, Basis::Cylinder { .. }) { unwrap_loop(&mut polygon); }
        polygon
    }).collect();
    // Largest absolute area is the outer loop; the rest are holes.
    projected.sort_by(|a, b| signed_area(b).abs().partial_cmp(&signed_area(a).abs()).unwrap_or(std::cmp::Ordering::Equal));
    let outer = projected.remove(0);
    let holes = projected;
    let (mut min_u, mut max_u, mut min_v, mut max_v) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for point in &outer {
        min_u = min_u.min(point[0]); max_u = max_u.max(point[0]);
        min_v = min_v.min(point[1]); max_v = max_v.max(point[1]);
    }
    if !(max_u > min_u && max_v > min_v) { return Err(format!("face #{face_id} has a degenerate boundary")); }
    let resolution = 24usize;
    let du = (max_u - min_u) / resolution as f64;
    let dv = (max_v - min_v) / resolution as f64;
    let mut mesh = Mesh::default();
    let group = format!("surface:{face_id}");
    for y in 0..resolution {
        for x in 0..resolution {
            let center = [min_u + (x as f64 + 0.5) * du, min_v + (y as f64 + 0.5) * dv];
            if !point_in_polygon(center, &outer) { continue; }
            if holes.iter().any(|hole| point_in_polygon(center, hole)) { continue; }
            let corners = [[min_u + x as f64 * du, min_v + y as f64 * dv], [min_u + (x + 1) as f64 * du, min_v + y as f64 * dv], [min_u + (x + 1) as f64 * du, min_v + (y + 1) as f64 * dv], [min_u + x as f64 * du, min_v + (y + 1) as f64 * dv]];
            let base = mesh.vertices.len() as u32;
            for corner in corners {
                let position = surface.position(corner);
                let normal = surface.normal(corner);
                mesh.vertices.push([position[0] as f32, position[1] as f32, position[2] as f32]);
                mesh.normals.push([normal[0] as f32, normal[1] as f32, normal[2] as f32]);
            }
            let normal = surface.normal(center);
            let a = mesh.vertices[base as usize];
            let b = mesh.vertices[base as usize + 1];
            let c = mesh.vertices[(base + 2) as usize];
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let face_normal = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
            let flip = (face_normal[0] * normal[0] as f32 + face_normal[1] * normal[1] as f32 + face_normal[2] * normal[2] as f32) < 0.0;
            if flip {
                mesh.faces.push([base, base + 2, base + 1]);
                mesh.faces.push([base, base + 3, base + 2]);
            } else {
                mesh.faces.push([base, base + 1, base + 2]);
                mesh.faces.push([base, base + 2, base + 3]);
            }
            mesh.face_groups.push(group.clone());
            mesh.face_groups.push(group.clone());
        }
    }
    mesh.lightweight = true;
    Ok(mesh)
}

/// Builds a display mesh for a STEP surface model made of plane/cylinder
/// `CURVE_BOUNDED_SURFACE` faces. Returns `Ok(None)` when the file has no such faces.
pub fn surface_model_mesh(path: &str) -> Result<Option<Mesh>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let faces: Vec<u32> = file.entities.iter().filter_map(|(id, entity)| if entity.name == "CURVE_BOUNDED_SURFACE" { Some(*id) } else { None }).collect();
    if faces.is_empty() { return Ok(None); }
    let mut meshes = Vec::new();
    let mut failures = Vec::new();
    for face in faces {
        match face_mesh(&file, face) {
            Ok(mesh) if !mesh.faces.is_empty() => meshes.push(mesh),
            Ok(_) => {}
            Err(error) => failures.push(format!("face #{face}: {error}")),
        }
    }
    if meshes.is_empty() { return Err(format!("no surface-model face could be tessellated: {}", failures.join("; "))); }
    if !failures.is_empty() { return Err(format!("some surface-model faces failed: {}", failures.join("; "))); }
    let mut mesh = Mesh::default();
    for part in meshes {
        let offset = mesh.vertices.len() as u32;
        mesh.vertices.extend(part.vertices);
        mesh.normals.extend(part.normals);
        mesh.faces.extend(part.faces.into_iter().map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]));
        mesh.face_groups.extend(part.face_groups);
    }
    mesh.lightweight = true;
    Ok(Some(mesh))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture(name: &str) -> Option<String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?.to_path_buf();
        let path: PathBuf = root.join("tests/Sample AP203 Data Files").join(name);
        path.exists().then(|| path.to_string_lossy().to_string())
    }

    #[test]
    fn gear_surface_model_produces_a_display_mesh() {
        let Some(path) = fixture("gear.stp") else { return; };
        let mesh = surface_model_mesh(&path).expect("surface mesh").expect("curve-bounded faces");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.faces.is_empty());
        assert!(mesh.face_groups.len() >= mesh.faces.len());
        assert!(mesh.vertices.iter().all(|vertex| vertex.iter().all(|value| value.is_finite())));
    }
}
