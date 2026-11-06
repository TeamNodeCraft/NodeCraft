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
/// intersection is not ported). A triangle is kept as a ghost when its centre is
/// farther than `tolerance` from the other display surface.
pub fn surface_difference(before: &Mesh, after: &Mesh) -> DifferenceMeshes {
    let tolerance = mesh_diagonal(before, after) * 1.0e-3;
    DifferenceMeshes {
        removed: far_triangles(before, after, tolerance),
        added: far_triangles(after, before, tolerance),
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

fn far_triangles(source: &Mesh, target: &Mesh, tolerance: f64) -> Mesh {
    if source.faces.is_empty() || target.faces.is_empty() {
        return Mesh::default();
    }
    let pair_count = source.faces.len().saturating_mul(target.faces.len());
    let stride = (pair_count / 32_000_000).max(1);
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for triangle in &source.faces {
        let Some(a) = source.vertices.get(triangle[0] as usize) else { continue };
        let Some(b) = source.vertices.get(triangle[1] as usize) else { continue };
        let Some(c) = source.vertices.get(triangle[2] as usize) else { continue };
        let centroid = [
            (f64::from(a[0]) + f64::from(b[0]) + f64::from(c[0])) / 3.0,
            (f64::from(a[1]) + f64::from(b[1]) + f64::from(c[1])) / 3.0,
            (f64::from(a[2]) + f64::from(b[2]) + f64::from(c[2])) / 3.0,
        ];
        let far = target
            .faces
            .iter()
            .step_by(stride)
            .all(|other| {
                let Some(oa) = target.vertices.get(other[0] as usize) else { return true };
                let Some(ob) = target.vertices.get(other[1] as usize) else { return true };
                let Some(oc) = target.vertices.get(other[2] as usize) else { return true };
                point_triangle_distance(centroid, oa, ob, oc) > tolerance
            });
        if far {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&[*a, *b, *c]);
            faces.push([base, base + 1, base + 2]);
        }
    }
    Mesh {
        vertices,
        faces,
        ..Mesh::default()
    }
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
    let before = union_solids(before.realized_solids()?)?;
    let after = union_solids(after.realized_solids()?)?;
    let kernel = nodecraft_solidcore::kernel::kernel();
    let options = nodecraft_solidcore::curved::CurvedTessOptions::default();
    // Identical (or rigid-equivalent) bodies have no material difference. Boolean
    // difference of coincident solids is both pointless and numerically hostile.
    let before_mesh = nodecraft_solidcore::curved::brep_display_mesh(&kernel, &before, &options)
        .map_err(|error| error.to_string())?;
    let after_mesh = nodecraft_solidcore::curved::brep_display_mesh(&kernel, &after, &options)
        .map_err(|error| error.to_string())?;
    let before_volume = mesh_volume(&before_mesh);
    let after_volume = mesh_volume(&after_mesh);
    if (before_volume - after_volume).abs() <= before_volume.abs().max(1.0) * 1.0e-6 {
        return Ok(DifferenceMeshes { removed: Mesh::default(), added: Mesh::default() });
    }
    let removed = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Difference, crate::copy_brep(&before)?, crate::copy_brep(&after)?)?;
    let added = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Difference, after, before)?;
    Ok(DifferenceMeshes {
        // One side of a pure cut/extrude difference is legitimately empty. An
        // empty BRep is a valid zero-triangle result, not a comparison failure.
        removed: nodecraft_solidcore::curved::brep_display_mesh(&kernel, &removed, &options)
            .unwrap_or_default(),
        added: nodecraft_solidcore::curved::brep_display_mesh(&kernel, &added, &options)
            .unwrap_or_default(),
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

fn union_solids(mut solids: Vec<nodecraft_solidcore::solidcore::topology::Brep>) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut result = solids.drain(..).next().ok_or("comparison Part has no solids")?;
    for solid in solids {
        result = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Union, result, solid)?;
    }
    Ok(result)
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
