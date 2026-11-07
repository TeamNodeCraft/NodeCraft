//! Exact before/after feature comparison.
//!
//! This is intentionally independent of any node type: callers supply the
//! evaluated before and after Parts, and receive the two material-difference
//! meshes. Feature nodes opt in by supplying those Parts; the comparison layer
//! never inspects node values.

use nodecraft_protocol::Mesh;

use crate::Part;

/// Material removed (before minus after) and material added (after minus before).
#[derive(Clone, Debug, Default)]
pub(crate) struct DifferenceMeshes {
    pub removed: Mesh,
    pub added: Mesh,
}

/// Approximate display-surface difference used when the exact Boolean
/// difference is unavailable (for example a G2 fillet whose surface
/// intersection is not ported).
///
/// Faces are compared, not individual triangles: a face is kept when a vote of
/// points on it is far enough from the other display surface. This gives the
/// overlay the face boundary of the real body instead of a triangle-soup
/// outline, and a shared surface stays out of the ghost even when the two
/// tessellations differ at the chord error.
pub fn surface_difference(before: &Mesh, after: &Mesh) -> DifferenceMeshes {
    let tolerance = mesh_diagonal(before, after) * 2.0e-3;
    DifferenceMeshes {
        removed: changed_faces(before, after, tolerance),
        added: changed_faces(after, before, tolerance),
    }
}

/// More than this fraction of a face's samples must be off the other surface
/// before the face is treated as changed. Tangent rows of facets lie within the
/// tolerance, so a small blend is still found while a kept curved face is not
/// drawn as removed material.
const FACE_SAMPLE_LIMIT: usize = 24;

fn changed_faces(source: &Mesh, target: &Mesh, tolerance: f64) -> Mesh {
    if source.faces.is_empty() || target.faces.is_empty() {
        return Mesh::default();
    }
    let Some(index) = SurfaceIndex::new(target, tolerance) else {
        return Mesh::default();
    };
    let face_of = face_indices(source);
    let face_count = face_of.iter().copied().max().map_or(0, |face| face as usize + 1);
    let mut centroids: Vec<Vec<[f64; 3]>> = vec![Vec::new(); face_count];
    for (triangle_index, triangle) in source.faces.iter().enumerate() {
        let face = face_of[triangle_index] as usize;
        let Some([a, b, c]) = triangle_points(source, triangle) else { continue };
        centroids[face].push([
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ]);
    }
    let mut changed: Vec<bool> = vec![false; face_count];
    for (face, points) in centroids.iter().enumerate() {
        if points.is_empty() {
            continue;
        }
        // Sample every centroid of a small face; spread a large face's samples
        // by the golden ratio so a row of facets cannot dominate the vote.
        let samples: Vec<&[f64; 3]> = if points.len() <= FACE_SAMPLE_LIMIT {
            points.iter().collect()
        } else {
            let len = points.len() as f64;
            (0..FACE_SAMPLE_LIMIT)
                .filter_map(|i| points.get(((i as f64 * 0.618_033_988_75).fract() * len) as usize))
                .collect()
        };
        let off = samples.iter().filter(|point| !index.contains(***point)).count();
        changed[face] = off * 8 > samples.len();
    }
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for (triangle_index, triangle) in source.faces.iter().enumerate() {
        let face = face_of[triangle_index] as usize;
        if !changed.get(face).copied().unwrap_or(false) {
            continue;
        }
        let Some([a, b, c]) = triangle_points(source, triangle) else { continue };
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&[a.map(|v| v as f32), b.map(|v| v as f32), c.map(|v| v as f32)]);
        faces.push([base, base + 1, base + 2]);
    }
    Mesh { vertices, faces, ..Mesh::default() }
}

fn face_indices(mesh: &Mesh) -> Vec<u32> {
    if mesh.face_groups.len() != mesh.faces.len() {
        return (0..mesh.faces.len() as u32).collect();
    }
    let mut ids = std::collections::HashMap::new();
    mesh.face_groups
        .iter()
        .map(|group| {
            let next = ids.len() as u32;
            *ids.entry(group.as_str()).or_insert(next)
        })
        .collect()
}

fn triangle_points(mesh: &Mesh, triangle: &[u32; 3]) -> Option<[[f64; 3]; 3]> {
    let point = |index: u32| {
        mesh.vertices.get(index as usize).map(|p| {
            [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
        })
    };
    Some([point(triangle[0])?, point(triangle[1])?, point(triangle[2])?])
}

/// A spatial index of a mesh's triangles, used to ask whether a point lies on its
/// surface within the tessellation tolerance.
struct SurfaceIndex {
    tris: Vec<[[f64; 3]; 3]>,
    tol: f64,
    lo: [f64; 3],
    cell: [f64; 3],
    res: usize,
    cells: Vec<Vec<u32>>,
}

impl SurfaceIndex {
    const BUDGET: usize = 4_000_000;

    fn new(mesh: &Mesh, tolerance: f64) -> Option<Self> {
        let tris: Vec<[[f64; 3]; 3]> = mesh
            .faces
            .iter()
            .filter_map(|triangle| triangle_points(mesh, triangle))
            .collect();
        if tris.is_empty() {
            return None;
        }
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for triangle in &tris {
            for point in triangle {
                for axis in 0..3 {
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                }
            }
        }
        let tol = tolerance.max(1.0e-9);
        let lo = min.map(|value| value - tol);
        let ext = [
            max[0] - min[0] + 2.0 * tol,
            max[1] - min[1] + 2.0 * tol,
            max[2] - min[2] + 2.0 * tol,
        ];
        let mut res = ((tris.len() as f64).cbrt().round() as usize).clamp(1, 48);
        loop {
            let cell = [ext[0] / res as f64, ext[1] / res as f64, ext[2] / res as f64];
            let mut index = SurfaceIndex {
                tris: Vec::new(),
                tol,
                lo,
                cell,
                res,
                cells: vec![Vec::new(); res * res * res],
            };
            let mut entries = 0usize;
            let mut overflow = false;
            'fill: for (triangle_index, triangle) in tris.iter().enumerate() {
                let (a, b) = index.range(triangle);
                for x in a[0]..=b[0] {
                    for y in a[1]..=b[1] {
                        for z in a[2]..=b[2] {
                            entries += 1;
                            if entries > Self::BUDGET && res > 1 {
                                overflow = true;
                                break 'fill;
                            }
                            if let Some(cell) = index.cells.get_mut((x * res + y) * res + z) {
                                cell.push(triangle_index as u32);
                            }
                        }
                    }
                }
            }
            if !overflow || res == 1 {
                index.tris = tris;
                return Some(index);
            }
            res /= 2;
        }
    }

    fn cell_of(&self, value: f64, axis: usize) -> usize {
        let lo = self.lo.get(axis).copied().unwrap_or(0.0);
        let cell = self.cell.get(axis).copied().unwrap_or(1.0);
        let index = ((value - lo) / cell).floor();
        if index.is_finite() {
            (index.max(0.0) as usize).min(self.res - 1)
        } else {
            0
        }
    }

    fn range(&self, triangle: &[[f64; 3]; 3]) -> ([usize; 3], [usize; 3]) {
        let mut min = [0usize; 3];
        let mut max = [0usize; 3];
        for axis in 0..3 {
            let values = triangle.map(|point| point[axis]);
            let lo = values[0].min(values[1]).min(values[2]) - self.tol;
            let hi = values[0].max(values[1]).max(values[2]) + self.tol;
            min[axis] = self.cell_of(lo, axis);
            max[axis] = self.cell_of(hi, axis);
        }
        (min, max)
    }

    fn contains(&self, point: [f64; 3]) -> bool {
        let x = self.cell_of(point[0], 0);
        let y = self.cell_of(point[1], 1);
        let z = self.cell_of(point[2], 2);
        let Some(cell) = self.cells.get((x * self.res + y) * self.res + z) else {
            return false;
        };
        cell.iter().filter_map(|index| self.tris.get(*index as usize)).any(|triangle| {
            point_triangle_distance(point, &triangle[0].map(|v| v as f32), &triangle[1].map(|v| v as f32), &triangle[2].map(|v| v as f32)) < self.tol
        })
    }
}

fn mesh_diagonal(a: &Mesh, b: &Mesh) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for mesh in [a, b] {
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(f64::from(vertex[axis]));
                max[axis] = max[axis].max(f64::from(vertex[axis]));
            }
        }
    }
    if !min[0].is_finite() {
        return 1.0;
    }
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-9)
}

fn point_triangle_distance(
    p: [f64; 3],
    a: &[f32; 3],
    b: &[f32; 3],
    c: &[f32; 3],
) -> f64 {
    let a = [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])];
    let b = [f64::from(b[0]), f64::from(b[1]), f64::from(b[2])];
    let c = [f64::from(c[0]), f64::from(c[1]), f64::from(c[2])];
    // Closest point on triangle, Real-Time Collision Detection (Ericson).
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return len(ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return len(bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return len(sub(p, add(a, scale(ab, v))));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return len(cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return len(sub(p, add(a, scale(ac, w))));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return len(sub(p, add(b, scale(sub(c, b), w))));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    len(sub(p, add(a, add(scale(ab, v), scale(ac, w)))))
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn scale(a: [f64; 3], s: f64) -> [f64; 3] { [a[0] * s, a[1] * s, a[2] * s] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn len(a: [f64; 3]) -> f64 { dot(a, a).sqrt() }

/// Compute exact difference meshes for one feature.
///
/// Parts are flattened to their realized solids. A multi-solid Part is unioned
/// into one operand first, so the result is independent of input body count.
pub(crate) fn compare_parts(before: &Part, after: &Part) -> Result<DifferenceMeshes, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let before_solids = before.realized_solids()?;
    let after_solids = after.realized_solids()?;
    let Some(before) = nodecraft_layers::union_breps_ops(&mut kernel, &before_solids)? else {
        return Err("comparison before Part has no solids".into());
    };
    let Some(after) = nodecraft_layers::union_breps_ops(&mut kernel, &after_solids)? else {
        return Err("comparison after Part has no solids".into());
    };
    // Exact B-rep volume, not display-mesh volume. Mesh density must not hide a
    // real feature change or make coincident bodies look different.
    let before_volume = nodecraft_layers::kernel_volume(&mut kernel, &before)?;
    let after_volume = nodecraft_layers::kernel_volume(&mut kernel, &after)?;
    if (before_volume - after_volume).abs() <= before_volume.abs().max(1.0) * 1.0e-6 {
        return Ok(DifferenceMeshes { removed: Mesh::default(), added: Mesh::default() });
    }
    let removed = nodecraft_layers::boolean_breps_ops(
        &mut kernel,
        &before,
        &after,
        nodecraft_layers::BooleanOp::Subtract,
    )?;
    let added = nodecraft_layers::boolean_breps_ops(
        &mut kernel,
        &after,
        &before,
        nodecraft_layers::BooleanOp::Subtract,
    )?;
    let tolerance = 0.05;
    let mut mesh_of = |brep: Option<nodecraft_solidcore::solidcore::topology::Brep>| {
        let Some(mut brep) = brep else { return Mesh::default() };
        nodecraft_layers::tessellate_kernel(&mut kernel, &mut brep, tolerance)
            .map(|display| display.mesh)
            .unwrap_or_default()
    };
    Ok(DifferenceMeshes {
        removed: mesh_of(removed),
        added: mesh_of(added),
    })
}

pub(crate) fn mesh_volume(mesh: &Mesh) -> f64 {
    mesh.faces.iter().map(|triangle| {
        let a = mesh.vertices[triangle[0] as usize];
        let b = mesh.vertices[triangle[1] as usize];
        let c = mesh.vertices[triangle[2] as usize];
        let cross = [
            f64::from(b[1]) * f64::from(c[2]) - f64::from(b[2]) * f64::from(c[1]),
            f64::from(b[2]) * f64::from(c[0]) - f64::from(b[0]) * f64::from(c[2]),
            f64::from(b[0]) * f64::from(c[1]) - f64::from(b[1]) * f64::from(c[0]),
        ];
        (f64::from(a[0]) * cross[0] + f64::from(a[1]) * cross[1] + f64::from(a[2]) * cross[2]) / 6.0
    }).sum()
}


#[cfg(test)]
mod tests {
    use super::*;

    fn triangle(offset: f32) -> Mesh {
        Mesh {
            vertices: vec![
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        }
    }

    fn quad(offset: f32, group: &str) -> Mesh {
        Mesh {
            vertices: vec![
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset + 1.0, 1.0, 0.0],
                [offset, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            face_groups: vec![group.to_owned(); 2],
            ..Mesh::default()
        }
    }

    #[test]
    fn surface_difference_keeps_whole_faces_not_triangle_soup() {
        let kept = quad(0.0, "kept");
        let moved = quad(5.0, "moved");
        let mut before = kept.clone();
        before.vertices.extend_from_slice(&moved.vertices);
        before.faces.extend(moved.faces.iter().map(|face| face.map(|index| index + 4)));
        before.face_groups.extend(moved.face_groups.clone());
        let after = kept.clone();
        let difference = surface_difference(&before, &after);
        assert_eq!(difference.removed.faces.len(), 2, "only the moved face is removed");
        assert!(difference.removed.face_groups.is_empty(), "approximate difference has no synthetic face groups");
        assert!(difference.added.faces.is_empty(), "the kept surface is unchanged");
    }

    #[test]
    fn surface_difference_keeps_only_moved_surfaces() {
        let before = triangle(0.0);
        let after = triangle(1.0);
        let difference = surface_difference(&before, &after);
        assert!(!difference.removed.faces.is_empty());
        assert!(!difference.added.faces.is_empty());
        let same = surface_difference(&before, &before);
        assert!(same.removed.faces.is_empty());
        assert!(same.added.faces.is_empty());
    }
}
