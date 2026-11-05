//! Mesh-level (preview) chamfer and circular fillet for a B-rep edge.
//!
//! # Why this exists
//!
//! solidcore has no runnable edge-treatment entry point yet: `blend.fillet_executive_p1`
//! (which owns `SmApiFilletEdges`, `SmApiChamferFillet` and
//! `SmFilletExecutive::CreateFilletCorners`) is not ported at the locked revision, and
//! the executive cannot be driven without an `ExecutiveAccess` adapter. See
//! [`crate::exact`] and `NOTES.md`.
//!
//! NodeCraft, however, previews shapes as display meshes. This module produces the
//! *shortest working path* to a usable Edge node today: the selected edge is located on
//! the exact B-rep (see [`crate::edge`]), its adjacent faces are read from
//! `brep_display_mesh`, and the display mesh is cut with the treatment's planes —
//!
//! * chamfer (equal / two-distance / distance-angle) is **one** plane through the edge,
//! * circular (G1) fillet is a sequence of tangent planes along the fillet arc,
//!   tessellated by [`FILLET_SEGMENTS`] facets (tangent planes never gouge the part,
//!   which a chord sequence would).
//!
//! Each cut is a closed mesh clip plus a triangulated cap, so the result stays
//! watertight. The output mesh uses NodeCraft's centered primitive convention.
//!
//! # Supported geometry (and the honest limits)
//!
//! Supported: a straight edge between two **planar** faces whose dihedral angle is
//! 90° (every box edge), on a convex body. Everything else returns an explicit
//! [`BlendError::Unsupported`] instead of producing wrong geometry:
//! curved or non-manifold edges, non-90° dihedrals, non-convex cut cross-sections,
//! and treatments on edges that meet at a shared corner (adjacent selections).
//!
//! G2/G3 fillets and the exact B-rep path are rejected by
//! [`crate::EdgeTreatment::validate`] / [`crate::exact`] with the missing solidcore
//! calls named.

use std::collections::HashMap;

use solidcore::kernel::Kernel;
use solidcore::math::Point2;
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::nurbs::knots::Side;
use solidcore::query::api_queries as queries;
use solidcore::query::face::p1::evaluate_curve;
use nodecraft_protocol::{BrepEdge, Mesh};
use nodecraft_step::PlanarBody;

use crate::edge::{EdgeSelection, brep_center};
use crate::error::BlendError;
use crate::treatment::{EdgeMode, EdgeTreatment};

/// Facets used to tessellate a circular fillet. Facet `i` cuts with the plane tangent
/// to the fillet arc at `i * 90° / FILLET_SEGMENTS`; the first and last tangents are the
/// adjacent faces themselves, so they are skipped.
pub const FILLET_SEGMENTS: usize = 8;

/// Relative tolerance for inside/outside classification of a clip plane.
/// Relative tolerance for welding cap-boundary points.
const WELD_EPSILON: f64 = 1.0e-7;

/// The two planes around one edge and the in-face "away from the edge" directions.
#[derive(Debug, Clone, Copy)]
struct EdgeFrame {
    /// Unit edge direction.
    direction: [f64; 3],
    /// In face A, perpendicular to the edge, pointing away from face B.
    away_a: [f64; 3],
    /// In face B, perpendicular to the edge, pointing away from face A.
    away_b: [f64; 3],
}

/// A cutting plane. The kept side is `dot(x - point, normal) <= 0`.
#[derive(Debug, Clone, Copy)]
struct Plane {
    point: [f64; 3],
    normal: [f64; 3],
}

impl Plane {
    fn distance(self, point: [f64; 3]) -> f64 {
        dot(sub(point, self.point), self.normal)
    }

    /// Flip the normal so the body (given by `interior`) stays on the negative side.
    fn orient_towards(self, interior: [f64; 3]) -> Self {
        if self.distance(interior) > 0.0 {
            Self {
                point: self.point,
                normal: scale(self.normal, -1.0),
            }
        } else {
            self
        }
    }
}

/// One triangle of the working mesh.
#[derive(Debug, Clone)]
struct Tri {
    points: [[f64; 3]; 3],
    group: String,
}

/// Chamfer or circular-fillet the selected B-rep edges and return a preview [`Mesh`]
/// in NodeCraft's centered coordinates.
///
/// `selections` must be non-empty (use [`crate::select_edges_from_values`] or
/// [`crate::select_edges_by_midpoints`]). The treatment is validated first, so G2/G3
/// fillets fail with the missing-call list rather than geometry.
pub fn apply_edge_treatment(
    kernel: &Kernel,
    brep: &solidcore::topology::Brep,
    selections: &[EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Mesh, BlendError> {
    treatment.validate()?;
    if selections.is_empty() {
        return Err(BlendError::InvalidInput(
            "an edge treatment needs at least one selected edge".into(),
        ));
    }
    let interior = brep_center(kernel, brep)?;
    let display = planar_solid_mesh(kernel, brep, interior)?;
    let mut triangles = mesh_triangles(&display)?;
    let diagonal = mesh_diagonal(&display)?;
    let scale = diagonal.max(1.0);
    let weld_epsilon = scale * WELD_EPSILON;

    for selection in selections {
        let frame = edge_frame(kernel, brep, selection, interior)?;
        let planes = treatment_planes(&frame, selection, treatment)?;
        for (index, plane) in planes.iter().enumerate() {
            let plane = plane.orient_towards(interior);
            triangles = clip_triangles(
                &triangles,
                plane,
                weld_epsilon,
                &format!("blend:{}:{}:{}", treatment.mode.as_str(), selection.ordinal, index),
            )?;
        }
    }

    if triangles.is_empty() {
        return Err(BlendError::InvalidInput(
            "the edge treatment removed the whole mesh; reduce length/radius".into(),
        ));
    }
    let center = brep_center(kernel, brep)?;
    triangles_to_mesh(&triangles, center)
}

/// Parse the node values, select the node's edges and preview the treatment.
///
/// This is the one-call entry point for a worker: `values` is the Edge node's
/// `values` JSON object.
pub fn edge_treatment_preview(
    kernel: &Kernel,
    brep: &solidcore::topology::Brep,
    values: &serde_json::Value,
) -> Result<Mesh, BlendError> {
    let treatment = EdgeTreatment::from_node_values(values)?;
    treatment.validate()?;
    let selections = crate::edge::select_edges_from_values(kernel, brep, values)?;
    apply_edge_treatment(kernel, brep, &selections, &treatment)
}

/// Convert the display mesh into working triangles.
fn mesh_triangles(mesh: &Mesh) -> Result<Vec<Tri>, BlendError> {
    if mesh.faces.is_empty() {
        return Err(BlendError::Kernel("display mesh has no triangles".into()));
    }
    let mut triangles = Vec::with_capacity(mesh.faces.len());
    for (index, face) in mesh.faces.iter().enumerate() {
        let mut points = [[0.0f64; 3]; 3];
        for (corner, vertex) in face.iter().enumerate() {
            let source = mesh.vertices.get(*vertex as usize).ok_or_else(|| {
                BlendError::Kernel(format!(
                    "display mesh triangle {index} references missing vertex {vertex}"
                ))
            })?;
            points[corner] = [
                f64::from(source[0]),
                f64::from(source[1]),
                f64::from(source[2]),
            ];
        }
        let group = mesh
            .face_groups
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("face:?{index}"));
        triangles.push(Tri { points, group });
    }
    Ok(triangles)
}

/// Diagonal of the display mesh's bounding box.
fn mesh_diagonal(mesh: &Mesh) -> Result<f64, BlendError> {
    if mesh.vertices.is_empty() {
        return Err(BlendError::Kernel("display mesh has no vertices".into()));
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            let value = f64::from(vertex[axis]);
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    Ok(length(sub(max, min)))
}

/// Build a corner-only display mesh of a planar-faced solid.
///
/// `brep_display_mesh` samples each face boundary independently, which leaves
/// T-junctions between adjacent faces and makes a watertight clip result impossible to
/// verify (and awkward to cap). The preview only supports planar-faced, convex bodies,
/// so this builder uses one point per B-rep edge (the corner), exactly like
/// `nodecraft_solidcore::box_mesh`, and rejects curved faces up front.
fn planar_solid_mesh(
    kernel: &Kernel,
    brep: &solidcore::topology::Brep,
    interior: [f64; 3],
) -> Result<Mesh, BlendError> {
    let mut mesh = Mesh {
        lightweight: true,
        ..Mesh::default()
    };
    for (face_index, (face_id, _face)) in brep.faces().iter().enumerate() {
        let context = format!("face {face_index}");
        let normal = planar_face_normal(kernel, brep, face_id, &context, interior)?;
        let loops = queries::face_get_boundary_loops(brep, face_id)
            .map_err(|error| BlendError::Kernel(format!("{context} loops: {error}")))?;
        let mut offset = 0usize;
        let mut first_loop = true;
        for loop_size in &loops.edge_counts {
            let mut polygon: Vec<[f64; 3]> = Vec::with_capacity(*loop_size);
            for use_index in offset..offset + loop_size {
                let edge_id = loops.edges[use_index];
                let edge = brep
                    .edge(edge_id)
                    .map_err(|error| BlendError::Kernel(format!("{context} edge: {error}")))?;
                let forward = loops.orientations[use_index] == OrientType::Same;
                let parameter = if forward {
                    edge.interval.evaluate(0.0)
                } else {
                    edge.interval.evaluate(1.0)
                };
                let evaluated = evaluate_curve(kernel, edge.curve.as_ref(), parameter)
                    .map_err(|error| BlendError::Kernel(format!("{context} curve: {error}")))?;
                let point = [evaluated[0].x, evaluated[0].y, evaluated[0].z];
                if polygon.last() != Some(&point) {
                    polygon.push(point);
                }
            }
            offset += loop_size;
            if polygon.len() >= 2 && near(polygon[0], polygon[polygon.len() - 1], 1.0e-9) {
                polygon.pop();
            }
            if polygon.len() < 3 {
                if first_loop {
                    return Err(BlendError::unsupported(
                        "edge treatment",
                        format!("{context} has fewer than three boundary vertices"),
                    ));
                }
                continue;
            }
            first_loop = false;
            let centroid = average(&polygon);
            let group = format!("face:{face_index}");
            let base = mesh.vertices.len() as u32;
            push_mesh_vertex(&mut mesh, centroid, normal);
            for point in &polygon {
                push_mesh_vertex(&mut mesh, *point, normal);
            }
            for corner in 0..polygon.len() {
                let a = base;
                let b = base + 1 + corner as u32;
                let c = base + 1 + ((corner + 1) % polygon.len()) as u32;
                push_mesh_triangle(&mut mesh, a, b, c, normal, &group);
            }
        }
    }
    if mesh.faces.is_empty() {
        return Err(BlendError::Kernel(
            "the B-rep produced no planar display triangles".into(),
        ));
    }
    Ok(mesh)
}

fn push_mesh_vertex(mesh: &mut Mesh, point: [f64; 3], normal: [f64; 3]) {
    mesh.vertices
        .push([point[0] as f32, point[1] as f32, point[2] as f32]);
    mesh.normals
        .push([normal[0] as f32, normal[1] as f32, normal[2] as f32]);
}

fn push_mesh_triangle(
    mesh: &mut Mesh,
    a: u32,
    b: u32,
    c: u32,
    normal: [f64; 3],
    group: &str,
) {
    let pa = mesh.vertices[a as usize].map(f64::from);
    let pb = mesh.vertices[b as usize].map(f64::from);
    let pc = mesh.vertices[c as usize].map(f64::from);
    let area = cross(sub(pb, pa), sub(pc, pa));
    if length(area) <= 0.0 {
        return;
    }
    if dot(area, normal) >= 0.0 {
        mesh.faces.push([a, b, c]);
    } else {
        mesh.faces.push([a, c, b]);
    }
    mesh.face_groups.push(group.to_string());
}

/// Find the two planar faces around a straight edge and build its cross-section frame.
///
/// The faces and their normals come from the exact B-rep (not from triangles), so the
/// preview is independent of how finely `brep_display_mesh` samples a face boundary.
fn face_normal_order(a: [f64; 3], b: [f64; 3]) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let key = |normal: [f64; 3]| {
        [
            normal[2].abs(),
            normal[0].abs(),
            normal[1].abs(),
            normal[2],
            normal[0],
            normal[1],
        ]
    };
    let (a, b) = (key(a), key(b));
    for (left, right) in a.iter().zip(b.iter()) {
        match left.partial_cmp(right).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {}
            order => return order.reverse(),
        }
    }
    Ordering::Equal
}

fn edge_frame(
    kernel: &Kernel,
    brep: &solidcore::topology::Brep,
    selection: &EdgeSelection,
    interior: [f64; 3],
) -> Result<EdgeFrame, BlendError> {
    let faces = solidcore::topology::edge::faces(brep, selection.edge, None).map_err(|error| {
        BlendError::Kernel(format!("edge {} faces: {error}", selection.ordinal))
    })?;
    let mut distinct = Vec::with_capacity(2);
    for face in faces {
        if !distinct.contains(&face) {
            distinct.push(face);
        }
    }
    if distinct.len() != 2 {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!(
                "edge {} borders {} distinct face(s) in the B-rep; only a straight edge                  between two planar faces is supported (curved faces, seams and                  non-manifold edges are not)",
                selection.ordinal,
                distinct.len()
            ),
        ));
    }
    let mut faces_with_normals = Vec::with_capacity(2);
    for face in &distinct {
        let context = format!("edge {} face", selection.ordinal);
        faces_with_normals.push((*face, planar_face_normal(kernel, brep, *face, &context, interior)?));
    }
    // A two-distance or distance-angle chamfer needs a consistent reference face.
    // B-rep face enumeration is arbitrary, so order the adjacent faces by their
    // dominant outward axis: horizontal faces first (top/bottom), then the
    // X/Y-facing sides. This keeps distances on a loop of edges from alternating
    // between the two faces, which made some chamfers face up and others down.
    faces_with_normals.sort_by(|a, b| face_normal_order(a.1, b.1));
    let normal_a = faces_with_normals[0].1;
    let normal_b = faces_with_normals[1].1;
    let direction = selection.direction().map_err(|_| {
        BlendError::unsupported(
            "edge treatment",
            format!(
                "edge {} is closed (start == end); closed edges need a different rail                  construction and are not supported by this preview",
                selection.ordinal
            ),
        )
    })?;
    if dot(normal_a, normal_b).abs() > 1.0e-3 {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!(
                "edge {} lies between faces whose normals are not perpendicular                  (dot = {:.6}); only 90-degree dihedrals are supported",
                selection.ordinal,
                dot(normal_a, normal_b)
            ),
        ));
    }
    if dot(direction, normal_a).abs() > 1.0e-3 || dot(direction, normal_b).abs() > 1.0e-3 {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!(
                "edge {} is not straight across both faces (direction/normal dot products                  {:.6}, {:.6}); only straight edges are supported",
                selection.ordinal,
                dot(direction, normal_a),
                dot(direction, normal_b)
            ),
        ));
    }
    let away_a = in_face_direction(normal_a, direction, normal_b, selection.ordinal)?;
    let away_b = in_face_direction(normal_b, direction, normal_a, selection.ordinal)?;
    if dot(away_a, away_b).abs() > 1.0e-3 {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!(
                "edge {} cross-section is not a right angle (in-face directions dot = {:.6})",
                selection.ordinal,
                dot(away_a, away_b)
            ),
        ));
    }
    Ok(EdgeFrame {
        direction,
        away_a,
        away_b,
    })
}

/// Outward unit normal of a planar face, with planarity verified on the exact surface.
fn planar_face_normal(
    kernel: &Kernel,
    brep: &solidcore::topology::Brep,
    face_id: solidcore::topology::FaceId,
    context: &str,
    interior: [f64; 3],
) -> Result<[f64; 3], BlendError> {
    let surface = solidcore::topology::face::p2::surface(brep, face_id)
        .map_err(|error| BlendError::Kernel(format!("{context} surface: {error}")))?
        .ok_or_else(|| {
            BlendError::Kernel(format!("{context} has no surface"))
        })?;
    let domain = solidcore::topology::face::p2::uv_domain(brep, face_id)
        .map_err(|error| BlendError::Kernel(format!("{context} domain: {error}")))?;
    let u0 = domain.u_interval().min();
    let u1 = domain.u_interval().max();
    let v0 = domain.v_interval().min();
    let v1 = domain.v_interval().max();
    if !(u0.is_finite() && u1.is_finite() && v0.is_finite() && v1.is_finite())
        || u1 <= u0
        || v1 <= v0
    {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!("{context} has a non-finite or empty UV domain"),
        ));
    }
    let face = brep
        .face(face_id)
        .map_err(|error| BlendError::Kernel(format!("{context}: {error}")))?;
    let flip = match face.faceuse {
        Some(faceuse) => brep
            .faceuse(faceuse)
            .map_err(|error| BlendError::Kernel(format!("{context} faceuse: {error}")))?
            .orientation
            == OrientType::Opposite,
        None => false,
    };
    let fractions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)];
    let mut normals = Vec::with_capacity(fractions.len());
    let mut points = Vec::with_capacity(fractions.len());
    for (fu, fv) in fractions {
        let uv = Point2::new(u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv);
        let point = surface
            .evaluate_point(kernel, uv)
            .map_err(|error| BlendError::Kernel(format!("{context} point: {error}")))?;
        let mut normal = surface
            .evaluate_normal(kernel, uv, Side::Left, Side::Left)
            .map_err(|error| BlendError::Kernel(format!("{context} normal: {error}")))?;
        normal = normal
            .unitize()
            .map_err(|_| BlendError::Kernel(format!("{context} normal is degenerate")))?;
        if flip {
            normal = solidcore::math::Vec3::new(-normal.x, -normal.y, -normal.z);
        }
        points.push([point.x, point.y, point.z]);
        normals.push([normal.x, normal.y, normal.z]);
    }
    let reference = normals[0];
    for normal in &normals {
        if dot(*normal, reference).abs() < 1.0 - 1.0e-6 {
            return Err(BlendError::unsupported(
                "edge treatment",
                format!("{context} is not planar; only planar faces are supported by this preview"),
            ));
        }
    }
    let point = [points[4][0], points[4][1], points[4][2]];
    let mut normal = reference;
    if dot(sub(interior, point), normal) > 0.0 {
        normal = scale(normal, -1.0);
    }
    Ok(normal)
}

/// `normalize(cross(normal, direction))` oriented away from the other face.
fn in_face_direction(
    normal: [f64; 3],
    direction: [f64; 3],
    other_normal: [f64; 3],
    ordinal: usize,
) -> Result<[f64; 3], BlendError> {
    let candidate = normalize(cross(normal, direction)).ok_or_else(|| {
        BlendError::unsupported(
            "edge treatment",
            format!("edge {ordinal} has a degenerate cross-section"),
        )
    })?;
    let sign = dot(candidate, other_normal);
    if sign.abs() < 0.5 {
        return Err(BlendError::unsupported(
            "edge treatment",
            format!("edge {ordinal} cross-section is not a right angle (dot = {sign:.6})"),
        ));
    }
    Ok(if sign < 0.0 { candidate } else { scale(candidate, -1.0) })
}

/// The cutting planes for one treatment.
fn treatment_planes(
    frame: &EdgeFrame,
    selection: &EdgeSelection,
    treatment: &EdgeTreatment,
) -> Result<Vec<Plane>, BlendError> {
    match treatment.mode {
        EdgeMode::Chamfer => {
            let (distance_a, distance_b) = treatment.chamfer_distances()?;
            let origin = selection.start;
            let point_a = add(origin, scale(frame.away_a, distance_a));
            let point_b = add(origin, scale(frame.away_b, distance_b));
            let normal = normalize(cross(frame.direction, sub(point_b, point_a))).ok_or_else(|| {
                BlendError::unsupported(
                    "chamfer",
                    format!(
                        "edge {} produces a degenerate chamfer plane (the two distances may be \
                         zero or equal to the edge length)",
                        selection.ordinal
                    ),
                )
            })?;
            Ok(vec![Plane {
                point: point_a,
                normal,
            }])
        }
        EdgeMode::Fillet => {
            let radius = treatment.length;
            let centre = add(
                selection.start,
                scale(add(frame.away_a, frame.away_b), radius),
            );
            // The fillet arc runs from the tangent point on face A to the tangent point
            // on face B; both are tangent to the adjacent faces, so skip the endpoints.
            let basis_a = scale(frame.away_b, -1.0);
            let basis_b = scale(frame.away_a, -1.0);
            let mut planes = Vec::with_capacity(FILLET_SEGMENTS.saturating_sub(1));
            for facet in 1..FILLET_SEGMENTS {
                let theta = std::f64::consts::FRAC_PI_2 * (facet as f64) / (FILLET_SEGMENTS as f64);
                let normal = add(scale(basis_a, theta.cos()), scale(basis_b, theta.sin()));
                let point = add(centre, scale(normal, radius));
                planes.push(Plane { point, normal });
            }
            if planes.is_empty() {
                return Err(BlendError::InvalidInput(
                    "circular fillet needs at least two tessellation facets".into(),
                ));
            }
            Ok(planes)
        }
    }
}

/// Clip the triangle soup against one plane and cap the opening.
fn clip_triangles(
    triangles: &[Tri],
    plane: Plane,
    weld_epsilon: f64,
    cap_group: &str,
) -> Result<Vec<Tri>, BlendError> {
    let mut kept: Vec<Tri> = Vec::with_capacity(triangles.len() + 8);
    let mut segments: Vec<([f64; 3], [f64; 3])> = Vec::new();
    let mut removed_any = false;
    for triangle in triangles {
        let distances = [
            plane.distance(triangle.points[0]),
            plane.distance(triangle.points[1]),
            plane.distance(triangle.points[2]),
        ];
        let outside = [
            distances[0] > 0.0,
            distances[1] > 0.0,
            distances[2] > 0.0,
        ];
        if !outside[0] && !outside[1] && !outside[2] {
            kept.push(triangle.clone());
            continue;
        }
        if outside[0] && outside[1] && outside[2] {
            removed_any = true;
            continue;
        }
        removed_any = true;
        let mut polygon: Vec<[f64; 3]> = Vec::with_capacity(4);
        let mut crossings: Vec<[f64; 3]> = Vec::with_capacity(2);
        for corner in 0..3 {
            let next = (corner + 1) % 3;
            if !outside[corner] {
                push_unique(&mut polygon, triangle.points[corner], weld_epsilon);
            }
            let current_distance = distances[corner];
            let next_distance = distances[next];
            let crosses = (current_distance <= 0.0 && next_distance > 0.0)
                || (current_distance > 0.0 && next_distance <= 0.0);
            if crosses {
                let denominator = current_distance - next_distance;
                if denominator != 0.0 {
                    let t = current_distance / denominator;
                    let point = lerp(triangle.points[corner], triangle.points[next], t);
                    push_unique(&mut polygon, point, weld_epsilon);
                    if !crossings
                        .iter()
                        .any(|existing| near(*existing, point, weld_epsilon))
                    {
                        crossings.push(point);
                    }
                }
            }
        }
        if polygon.len() >= 2 && near(polygon[0], polygon[polygon.len() - 1], weld_epsilon) {
            polygon.pop();
        }
        if polygon.len() >= 3 {
            fan_triangulate(&polygon, &triangle.group, &mut kept);
        }
        if crossings.len() == 2 {
            segments.push((crossings[0], crossings[1]));
        }
    }
    if segments.is_empty() {
        if removed_any {
            return Err(BlendError::unsupported(
                "edge treatment",
                "the cut removed triangles but produced no closed cross-section",
            ));
        }
        return Ok(kept);
    }
    let caps = cap_opening(&segments, plane, weld_epsilon, cap_group)?;
    kept.extend(caps);
    Ok(kept)
}

/// Triangulate the planar opening left by a clip.
fn cap_opening(
    segments: &[([f64; 3], [f64; 3])],
    plane: Plane,
    weld_epsilon: f64,
    group: &str,
) -> Result<Vec<Tri>, BlendError> {
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut index: HashMap<[i64; 3], usize> = HashMap::new();
    let mut adjacency: Vec<Vec<usize>> = Vec::new();
    for (from, to) in segments {
        let from = intern(*from, weld_epsilon, &mut points, &mut index, &mut adjacency);
        let to = intern(*to, weld_epsilon, &mut points, &mut index, &mut adjacency);
        if from == to {
            continue;
        }
        if adjacency[from].contains(&to) || adjacency[to].contains(&from) {
            return Err(BlendError::unsupported(
                "edge treatment",
                "the cut cross-section is not a simple loop (duplicate boundary segment)",
            ));
        }
        adjacency[from].push(to);
        adjacency[to].push(from);
    }
    let mut loop_points: Vec<[f64; 3]> = Vec::new();
    if points.is_empty() {
        return Err(BlendError::unsupported(
            "edge treatment",
            "the cut cross-section has no boundary points",
        ));
    }
    if points.is_empty() {
        return Err(BlendError::unsupported("edge treatment", "the cut cross-section has no boundary points"));
    }
    let mut visited = vec![false; points.len()];
    let mut loops: Vec<Vec<usize>> = Vec::new();
    for start in 0..points.len() {
        if visited[start] { continue; }
        let mut walk = Vec::new();
        let mut previous = usize::MAX;
        let mut current = start;
        loop {
            if walk.len() > points.len() {
                return Err(BlendError::unsupported("edge treatment", "the cut cross-section does not close"));
            }
            walk.push(current);
            visited[current] = true;
            let neighbours = &adjacency[current];
            if neighbours.len() != 2 {
                return Err(BlendError::unsupported("edge treatment", format!("the cut cross-section is not a manifold loop (vertex degree {})", neighbours.len())));
            }
            let next = if neighbours[0] == previous { neighbours[1] } else { neighbours[0] };
            previous = current;
            current = next;
            if current == start { break; }
            if visited[current] {
                return Err(BlendError::unsupported("edge treatment", "the cut cross-section has intersecting loops"));
            }
        }
        if walk.len() < 3 {
            return Err(BlendError::unsupported("edge treatment", "a cut cross-section loop has fewer than three points"));
        }
        loops.push(walk);
    }
    if loops.len() == 1 {
        let mut loop_points: Vec<[f64; 3]> = loops[0].iter().map(|index| points[*index]).collect();
        let newell = newell_normal(&loop_points);
        if dot(newell, plane.normal) < 0.0 { loop_points.reverse(); }
        let centroid = average(&loop_points);
        let mut caps = Vec::with_capacity(loop_points.len());
        for corner in 0..loop_points.len() {
            let next = (corner + 1) % loop_points.len();
            let facet = [centroid, loop_points[corner], loop_points[next]];
            let normal = cross(sub(facet[1], facet[0]), sub(facet[2], facet[0]));
            let normal = normalize(normal).ok_or_else(|| BlendError::unsupported("edge treatment", "the cut cross-section triangulation produced a degenerate facet"))?;
            if dot(normal, plane.normal) <= 0.0 {
                return Err(BlendError::unsupported("edge treatment", "the cut cross-section is not convex; only convex bodies are supported"));
            }
            caps.push(Tri { points: facet, group: group.to_string() });
        }
        return Ok(caps);
    }
    cap_planar_loops(&loops, &points, plane, group)
}

/// Cap a planar cut whose boundary is several loops (outer walls plus cavity loops).
fn cap_planar_loops(
    loops: &[Vec<usize>],
    points: &[[f64; 3]],
    plane: Plane,
    group: &str,
) -> Result<Vec<Tri>, BlendError> {
    let normal = plane.normal;
    let helper = if normal[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = normalize(cross(normal, helper)).ok_or_else(|| BlendError::unsupported("edge treatment", "degenerate cut plane"))?;
    let v = cross(normal, u);
    let project = |point: [f64; 3]| [dot(point, u), dot(point, v)];
    let loop_2d: Vec<Vec<[f64; 2]>> = loops.iter().map(|indices| indices.iter().map(|index| project(points[*index])).collect()).collect();
    let area = |loop2d: &[[f64; 2]]| {
        let mut sum = 0.0;
        for corner in 0..loop2d.len() {
            let next = (corner + 1) % loop2d.len();
            sum += loop2d[corner][0] * loop2d[next][1] - loop2d[next][0] * loop2d[corner][1];
        }
        sum.abs() * 0.5
    };
    let inside = |point: [f64; 2], loop2d: &[[f64; 2]]| {
        let mut hit = false;
        for corner in 0..loop2d.len() {
            let next = (corner + 1) % loop2d.len();
            let (a, b) = (loop2d[corner], loop2d[next]);
            if (a[1] > point[1]) != (b[1] > point[1])
                && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
            { hit = !hit; }
        }
        hit
    };
    let areas: Vec<f64> = loop_2d.iter().map(|loop2d| area(loop2d)).collect();
    let mut parent: Vec<Option<usize>> = vec![None; loops.len()];
    for index in 0..loops.len() {
        let point = loop_2d[index][0];
        parent[index] = (0..loops.len())
            .filter(|candidate| *candidate != index && areas[*candidate] > areas[index] && inside(point, &loop_2d[*candidate]))
            .min_by(|left, right| areas[*left].total_cmp(&areas[*right]));
    }
    let mut planar = PlanarBody { points: points.to_vec(), faces: vec![], holes: vec![] };
    for root in 0..loops.len() {
        if parent[root].is_some() { continue; }
        let mut outer = loops[root].clone();
        if dot(newell_normal(&outer.iter().map(|index| points[*index]).collect::<Vec<_>>()), normal) < 0.0 { outer.reverse(); }
        let holes: Vec<Vec<usize>> = loops
            .iter()
            .enumerate()
            .filter(|(child, _)| parent[*child] == Some(root))
            .map(|(_, hole)| hole.clone())
            .collect();
        planar.push_face(outer, holes);
    }
    let indices = planar
        .triangulate_earcut()
        .map_err(|error| BlendError::unsupported("edge treatment", error))?;
    let mut caps = Vec::with_capacity(indices.len());
    for [a, b, c] in indices {
        let mut facet = [planar.points[a], planar.points[b], planar.points[c]];
        let facet_normal = cross(sub(facet[1], facet[0]), sub(facet[2], facet[0]));
        if dot(facet_normal, normal) < 0.0 { facet.swap(1, 2); }
        caps.push(Tri { points: facet, group: group.to_string() });
    }
    Ok(caps)
}

/// Weld one cap-boundary point and return its index./// Weld one cap-boundary point and return its index.
fn intern(
    point: [f64; 3],
    weld_epsilon: f64,
    points: &mut Vec<[f64; 3]>,
    index: &mut HashMap<[i64; 3], usize>,
    adjacency: &mut Vec<Vec<usize>>,
) -> usize {
    let key = [
        (point[0] / weld_epsilon).round() as i64,
        (point[1] / weld_epsilon).round() as i64,
        (point[2] / weld_epsilon).round() as i64,
    ];
    if let Some(existing) = index.get(&key) {
        return *existing;
    }
    let position = points.len();
    points.push(point);
    index.insert(key, position);
    adjacency.push(Vec::new());
    position
}

/// Fan-triangulate an ordered polygon.
fn fan_triangulate(polygon: &[[f64; 3]], group: &str, out: &mut Vec<Tri>) {
    if polygon.len() < 3 {
        return;
    }
    for corner in 1..polygon.len() - 1 {
        let points = [polygon[0], polygon[corner], polygon[corner + 1]];
        let area = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        if length(area) <= 0.0 || !length(area).is_finite() {
            continue;
        }
        out.push(Tri {
            points,
            group: group.to_string(),
        });
    }
}

/// Build the protocol mesh from working triangles, centred on `center`.
/// Exact planar chamfer for the preview's supported straight 90-degree edges.
///
/// Each chamfer plane is realized as a large oriented box on the removed side
/// of the plane and subtracted from the input B-rep. This is not the full
/// SolidCore blend, but it produces an exact closed B-rep for the cases this
/// preview already accepts and lets chained/exported chamfers stay exact.
pub fn chamfer_brep_exact(
    brep: &solidcore::topology::Brep,
    values: &serde_json::Value,
) -> Result<solidcore::topology::Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    chamfer_brep_exact_with_kernel(&mut kernel, brep, values)
}

pub fn chamfer_brep_exact_with_kernel(
    kernel: &mut Kernel,
    brep: &solidcore::topology::Brep,
    values: &serde_json::Value,
) -> Result<solidcore::topology::Brep, String> {
    use solidcore::geom::axis2_placement::Axis2Placement;
    use solidcore::math::{Point3, Vec3};
    use solidcore::primitive::api_primitives as api;
    let treatment = EdgeTreatment::from_node_values(values).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    treatment.validate().map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if treatment.mode != EdgeMode::Chamfer {
        return Err("exact chamfer requires chamfer mode".into());
    }
    let selections = crate::edge::select_edges_from_values(kernel, brep, values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if selections.is_empty() {
        return Err("an edge treatment needs at least one selected edge".into());
    }
    let interior = brep_center(&*kernel, brep).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    let (min, max) = brep_vertex_bounds(brep)?;
    let diagonal = length(sub(max, min)).max(1.0);
    // Conservative body-scale bound. This catches nonsensical distances before the
    // kernel is asked to cut with a tool larger than the whole body.
    let (distance_a, distance_b) = treatment
        .chamfer_distances()
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if distance_a > diagonal || distance_b > diagonal {
        return Err(format!(
            "chamfer distance {:.4}/{:.4} exceeds the body diagonal {:.4}; reduce the distance",
            distance_a, distance_b, diagonal
        ));
    }
    let extent = diagonal * 4.0 + 100.0;
    let mut result = brep
        .clone_standing_for_self()
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    for selection in &selections {
        let frame = edge_frame(&*kernel, brep, selection, interior).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
        let planes = treatment_planes(&frame, selection, &treatment).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
        for plane in planes {
            let plane = plane.orient_towards(interior);
            let normal = normalize(plane.normal).ok_or("chamfer plane has no normal")?;
            let helper = if normal[2].abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
            let normal = Vec3::new(normal[0], normal[1], normal[2]);
            let x_axis = normal.cross(helper).unitize().map_err(|error| nodecraft_solidcore::error::user_message(error))?;
            let y_axis = normal.cross(x_axis).unitize().map_err(|error| nodecraft_solidcore::error::user_message(error))?;
            let mut tool = api::create_box(kernel, Point3::new(-extent, -extent, 0.0), 2.0 * extent, 2.0 * extent, extent)
                .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
            let placement = Axis2Placement::new(
                Point3::new(plane.point[0], plane.point[1], plane.point[2]),
                x_axis,
                y_axis,
            );
            solidcore::topology::brep::p5::transform(
                kernel,
                &mut tool,
                &placement,
                None,
                true,
            ).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
            result = nodecraft_bool::boolean_breps_with_kernel(
                kernel,
                nodecraft_bool::BooleanOp::Difference,
                result,
                tool,
            )?;
        }
    }
    if result.face_list().is_empty() || result.vertex_list().is_empty() {
        return Err("chamfer removed the whole body; reduce the chamfer distance".into());
    }
    Ok(result)
}

fn brep_vertex_bounds(brep: &solidcore::topology::Brep) -> Result<([f64; 3], [f64; 3]), String> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for id in brep.vertex_list() {
        let point = brep.vertex(*id).map_err(|error| nodecraft_solidcore::error::user_message(error))?.point;
        min = [min[0].min(point.x), min[1].min(point.y), min[2].min(point.z)];
        max = [max[0].max(point.x), max[1].max(point.y), max[2].max(point.z)];
    }
    Ok((min, max))
}

fn triangles_to_mesh(triangles: &[Tri], center: [f64; 3]) -> Result<Mesh, BlendError> {
    let mut mesh = Mesh {
        lightweight: true,
        ..Mesh::default()
    };
    for triangle in triangles {
        let normal = cross(
            sub(triangle.points[1], triangle.points[0]),
            sub(triangle.points[2], triangle.points[0]),
        );
        let normal = normalize(normal).ok_or_else(|| {
            BlendError::Kernel("edge treatment produced a degenerate triangle".into())
        })?;
        let base = mesh.vertices.len() as u32;
        for point in &triangle.points {
            mesh.vertices.push([
                (point[0] - center[0]) as f32,
                (point[1] - center[1]) as f32,
                (point[2] - center[2]) as f32,
            ]);
            mesh.normals
                .push([normal[0] as f32, normal[1] as f32, normal[2] as f32]);
        }
        mesh.faces.push([base, base + 1, base + 2]);
        mesh.face_groups.push(triangle.group.clone());
    }
    if mesh.faces.is_empty() {
        return Err(BlendError::Kernel(
            "edge treatment produced no triangles".into(),
        ));
    }
    add_preview_feature_edges(&mut mesh, triangles, center);
    Ok(mesh)
}

/// Add the treatment's new boundaries as preview-only display edges.
///
/// There is no exact B-rep for the clipped preview, so derive one segment per
/// boundary between different triangle groups (and per open mesh boundary). The
/// ids are deliberately namespaced as `preview-edge:...`; they are display data,
/// not addresses for a later exact feature.
fn add_preview_feature_edges(mesh: &mut Mesh, triangles: &[Tri], center: [f64; 3]) {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Segment {
        points: [[f64; 3]; 2],
        groups: Vec<String>,
    }

    let quantized = |point: [f64; 3]| {
        [
            ((point[0] - center[0]) * 1.0e7).round() as i64,
            ((point[1] - center[1]) * 1.0e7).round() as i64,
            ((point[2] - center[2]) * 1.0e7).round() as i64,
        ]
    };
    let mut segments: BTreeMap<[i64; 6], Segment> = BTreeMap::new();
    for triangle in triangles {
        for corner in 0..3 {
            let a = triangle.points[corner];
            let b = triangle.points[(corner + 1) % 3];
            let (ka, kb) = (quantized(a), quantized(b));
            let (key, first, second) = if ka < kb {
                ([ka[0], ka[1], ka[2], kb[0], kb[1], kb[2]], a, b)
            } else {
                ([kb[0], kb[1], kb[2], ka[0], ka[1], ka[2]], b, a)
            };
            let segment = segments.entry(key).or_insert_with(|| Segment {
                points: [first, second],
                groups: Vec::new(),
            });
            if !segment.groups.contains(&triangle.group) {
                segment.groups.push(triangle.group.clone());
            }
        }
    }
    let mut index = 0usize;
    for segment in segments.into_values() {
        // Only boundaries between two distinct face groups are treatment edges.
        // Edges inside one tessellated face/cap group are triangulation, not
        // topology, and must not be drawn.
        if segment.groups.len() != 2 || segment.groups[0] == segment.groups[1] {
            continue;
        }
        let base = mesh.vertices.len() as u32;
        for point in segment.points {
            mesh.vertices.push([
                (point[0] - center[0]) as f32,
                (point[1] - center[1]) as f32,
                (point[2] - center[2]) as f32,
            ]);
            mesh.normals.push([0.0, 0.0, 1.0]);
        }
        mesh.brep_edges.push(BrepEdge {
            id: format!("preview-edge:{index}"),
            kind: "preview".into(),
            segments: vec![[base, base + 1]],
        });
        index += 1;
    }
}

/// Append `point` unless it duplicates the previous polygon corner.
fn push_unique(polygon: &mut Vec<[f64; 3]>, point: [f64; 3], tolerance: f64) {
    if let Some(last) = polygon.last() {
        if near(*last, point, tolerance) {
            return;
        }
    }
    polygon.push(point);
}

fn near(a: [f64; 3], b: [f64; 3], tolerance: f64) -> bool {
    length(sub(a, b)) <= tolerance
}

fn newell_normal(points: &[[f64; 3]]) -> [f64; 3] {
    let mut normal = [0.0f64; 3];
    for corner in 0..points.len() {
        let next = (corner + 1) % points.len();
        normal = add(normal, cross(points[corner], points[next]));
    }
    normal
}

fn average(points: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    for point in points {
        sum = add(sum, *point);
    }
    scale(sum, 1.0 / points.len() as f64)
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let norm = length(a);
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    Some(scale(a, 1.0 / norm))
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    add(a, scale(sub(b, a), t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::kernel::Kernel;
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    #[test]
    fn exact_chamfer_cuts_a_box_edge() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = api::create_box(&mut kernel, Point3::new(-5.0, -5.0, -5.0), 10.0, 10.0, 10.0)
            .expect("box");
        let values = serde_json::json!({
            "mode": "chamfer",
            "chamferType": "equal",
            "length": 1.0,
            "edges": [{"ordinal": 0}]
        });
        let result = chamfer_brep_exact(&brep, &values).expect("exact chamfer");
        assert!(result.faces().len() > brep.faces().len());
        assert!(result.vertices().len() > brep.vertices().len());
    }

    #[test]
    fn top_edge_frames_use_the_horizontal_face_as_reference() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = api::create_box(&mut kernel, Point3::new(-5.0, -5.0, -5.0), 10.0, 10.0, 10.0)
            .expect("box");
        let interior = brep_center(&kernel, &brep).expect("center");
        let selections = crate::edge::edge_selections(&kernel, &brep).expect("edges");
        let top: Vec<_> = selections
            .into_iter()
            .filter(|selection| selection.midpoint[2] > 4.9)
            .collect();
        assert!(!top.is_empty());
        for selection in top {
            let frame = edge_frame(&kernel, &brep, &selection, interior).expect("frame");
            assert!(
                frame.away_a[2].abs() < 1.0e-9,
                "top edge {} reference face is not horizontal: away_a={:?}",
                selection.ordinal,
                frame.away_a
            );
            assert!(
                frame.away_b[2].abs() > 0.9,
                "top edge {} second face is not vertical: away_b={:?}",
                selection.ordinal,
                frame.away_b
            );
        }
    }
}
