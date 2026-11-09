//! Shared helpers for the nodecraft-blend tests.

#![allow(dead_code)]

use std::collections::HashMap;

use nodecraft_protocol::Mesh;
use solidcore::kernel::Kernel;
use solidcore::topology::Brep;

/// The test body: a 40 x 30 x 20 box at the origin (B-rep coordinates).
pub fn box_40_30_20() -> (Kernel, Brep) {
    nodecraft_blend::box_brep(40.0, 30.0, 20.0).expect("create box")
}

/// All edge midpoints of a B-rep (B-rep coordinates).
pub fn edge_midpoints(kernel: &Kernel, brep: &Brep) -> Vec<[f64; 3]> {
    nodecraft_blend::edge_selections(kernel, brep)
        .expect("edge selections")
        .into_iter()
        .map(|selection| selection.midpoint)
        .collect()
}

/// Quantize a point so f32/f64 round-off welds.
fn key(point: [f32; 3]) -> [i64; 3] {
    [
        (f64::from(point[0]) * 10_000.0).round() as i64,
        (f64::from(point[1]) * 10_000.0).round() as i64,
        (f64::from(point[2]) * 10_000.0).round() as i64,
    ]
}

/// Every geometric edge of a welded mesh appears exactly twice (closed orientable 2-manifold).
pub fn closure_report(mesh: &Mesh) -> (usize, usize) {
    let mut edges: HashMap<([i64; 3], [i64; 3]), usize> = HashMap::new();
    for face in &mesh.faces {
        let keys = [
            key(mesh.vertices[face[0] as usize]),
            key(mesh.vertices[face[1] as usize]),
            key(mesh.vertices[face[2] as usize]),
        ];
        for corner in 0..3 {
            let a = keys[corner];
            let b = keys[(corner + 1) % 3];
            let pair = if a <= b { (a, b) } else { (b, a) };
            *edges.entry(pair).or_insert(0) += 1;
        }
    }
    let bad = edges.values().filter(|count| **count != 2).count();
    (edges.len(), bad)
}

/// Signed volume of a closed mesh, from its triangle fans.
pub fn volume(mesh: &Mesh) -> f64 {
    let mut total = 0.0;
    for face in &mesh.faces {
        let a = mesh.vertices[face[0] as usize];
        let b = mesh.vertices[face[1] as usize];
        let c = mesh.vertices[face[2] as usize];
        let cross = [
            f64::from(b[1]) * f64::from(c[2]) - f64::from(b[2]) * f64::from(c[1]),
            f64::from(b[2]) * f64::from(c[0]) - f64::from(b[0]) * f64::from(c[2]),
            f64::from(b[0]) * f64::from(c[1]) - f64::from(b[1]) * f64::from(c[0]),
        ];
        total +=
            f64::from(a[0]) * cross[0] + f64::from(a[1]) * cross[1] + f64::from(a[2]) * cross[2];
    }
    total / 6.0
}

/// Smallest distance from any mesh vertex to the infinite line `point + t * direction`.
pub fn min_distance_to_line(mesh: &Mesh, point: [f64; 3], direction: [f64; 3]) -> f64 {
    let norm =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    let unit = [
        direction[0] / norm,
        direction[1] / norm,
        direction[2] / norm,
    ];
    let mut best = f64::INFINITY;
    for vertex in &mesh.vertices {
        let relative = [
            f64::from(vertex[0]) - point[0],
            f64::from(vertex[1]) - point[1],
            f64::from(vertex[2]) - point[2],
        ];
        let along = relative[0] * unit[0] + relative[1] * unit[1] + relative[2] * unit[2];
        let perpendicular = [
            relative[0] - along * unit[0],
            relative[1] - along * unit[1],
            relative[2] - along * unit[2],
        ];
        let distance = (perpendicular[0] * perpendicular[0]
            + perpendicular[1] * perpendicular[1]
            + perpendicular[2] * perpendicular[2])
            .sqrt();
        best = best.min(distance);
    }
    best
}

/// True when some mesh vertex is within `tol` of `point`.
pub fn has_vertex_near(mesh: &Mesh, point: [f64; 3], tol: f64) -> bool {
    mesh.vertices.iter().any(|vertex| {
        let dx = f64::from(vertex[0]) - point[0];
        let dy = f64::from(vertex[1]) - point[1];
        let dz = f64::from(vertex[2]) - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt() <= tol
    })
}

/// The plane of the first triangle whose face group starts with `prefix`:
/// `(point on the plane, unit normal)`.
pub fn blend_plane(mesh: &Mesh, prefix: &str) -> Option<([f64; 3], [f64; 3])> {
    for (index, face) in mesh.faces.iter().enumerate() {
        let group = mesh.face_groups.get(index)?;
        if !group.starts_with(prefix) {
            continue;
        }
        let a = mesh.vertices[face[0] as usize].map(f64::from);
        let b = mesh.vertices[face[1] as usize].map(f64::from);
        let c = mesh.vertices[face[2] as usize].map(f64::from);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if length > 0.0 {
            return Some((a, [n[0] / length, n[1] / length, n[2] / length]));
        }
    }
    None
}

/// Distance from `point` to the plane `(plane_point, unit normal)`.
pub fn plane_distance(point: [f64; 3], plane: ([f64; 3], [f64; 3])) -> f64 {
    let (origin, normal) = plane;
    ((point[0] - origin[0]) * normal[0]
        + (point[1] - origin[1]) * normal[1]
        + (point[2] - origin[2]) * normal[2])
        .abs()
}

/// Number of triangles whose face group starts with `prefix`.
pub fn count_group(mesh: &Mesh, prefix: &str) -> usize {
    mesh.face_groups
        .iter()
        .filter(|group| group.starts_with(prefix))
        .count()
}
