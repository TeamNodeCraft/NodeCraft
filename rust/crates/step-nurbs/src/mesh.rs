//! Tessellation of B-spline/NURBS surfaces into a display mesh.

use crate::BsplineSurface;

/// A sampled surface mesh with shared vertices and triangle indices.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMesh {
    pub points: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// Samples a surface on a uniform `(u, v)` grid. No trimming is applied.
pub fn tessellate_surface(surface: &BsplineSurface, u_segments: usize, v_segments: usize) -> Result<SurfaceMesh, String> {
    if u_segments == 0 || v_segments == 0 { return Err("surface tessellation needs at least one segment in each direction".into()); }
    let (u0, u1) = surface.u_domain();
    let (v0, v1) = surface.v_domain();
    if !(u1 > u0 && v1 > v0) { return Err("surface has an empty parameter domain".into()); }
    let mut points = Vec::with_capacity((u_segments + 1) * (v_segments + 1));
    for i in 0..=u_segments {
        let u = u0 + (u1 - u0) * i as f64 / u_segments as f64;
        for j in 0..=v_segments {
            let v = v0 + (v1 - v0) * j as f64 / v_segments as f64;
            points.push(surface.point_at(u, v)?);
        }
    }
    let stride = (v_segments + 1) as u32;
    let mut triangles = Vec::with_capacity(u_segments * v_segments * 2);
    for i in 0..u_segments as u32 {
        for j in 0..v_segments as u32 {
            let a = i * stride + j;
            let b = a + stride;
            triangles.push([a, b, a + 1]);
            triangles.push([b, b + 1, a + 1]);
        }
    }
    Ok(SurfaceMesh { points, triangles })
}


/// Samples a surface and keeps only grid cells whose UV center lies inside `outer`.
fn inside(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let mut j = polygon.len().saturating_sub(1);
    for i in 0..polygon.len() {
        let a = polygon[i]; let b = polygon[j];
        if (a[1] > point[1]) != (b[1] > point[1]) {
            let x = (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0];
            if point[0] < x { inside = !inside; }
        }
        j = i;
    }
    inside
}

/// Tessellates the part of a surface inside a UV trim polygon, with optional UV holes.
///
/// Cells are rejected when their center is outside `outer_uv` or inside a hole; the
/// result is a display mesh, not an exact trimmed boundary. Passing the full surface
/// domain as `outer_uv` reproduces [`tessellate_surface`] exactly.
pub fn tessellate_trimmed_surface(
    surface: &BsplineSurface,
    u_segments: usize,
    v_segments: usize,
    outer_uv: &[[f64; 2]],
    holes_uv: &[Vec<[f64; 2]>],
) -> Result<SurfaceMesh, String> {
    if outer_uv.len() < 3 { return Err("trim polygon needs at least three UV points".into()); }
    let base = tessellate_surface(surface, u_segments, v_segments)?;
    let (u0, u1) = surface.u_domain();
    let (v0, v1) = surface.v_domain();
    let stride = (v_segments + 1) as u32;
    let mut triangles = Vec::new();
    for i in 0..u_segments {
        for j in 0..v_segments {
            let u = u0 + (u1 - u0) * (i as f64 + 0.5) / u_segments as f64;
            let v = v0 + (v1 - v0) * (j as f64 + 0.5) / v_segments as f64;
            let center = [u, v];
            if !inside(center, outer_uv) || holes_uv.iter().any(|hole| inside(center, hole)) { continue; }
            let a = i as u32 * stride + j as u32;
            triangles.push([a, a + stride, a + 1]);
            triangles.push([a + stride, a + stride + 1, a + 1]);
        }
    }
    Ok(SurfaceMesh { points: base.points, triangles })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BsplineSurface, Knots};

    fn bilinear() -> BsplineSurface {
        BsplineSurface {
            u_degree: 1, v_degree: 1,
            poles: vec![vec![[0.,0.,0.],[0.,1.,0.]], vec![[1.,0.,0.],[1.,1.,0.]]],
            weights: None,
            u_knots: Knots::uniform(1, 2).unwrap(), v_knots: Knots::uniform(1, 2).unwrap(),
            u_closed: false, v_closed: false, self_intersect: false,
        }
    }

    #[test]
    fn tessellates_bilinear_unit_square() {
        let mesh = tessellate_surface(&bilinear(), 4, 3).unwrap();
        assert_eq!(mesh.points.len(), 20);
        assert_eq!(mesh.triangles.len(), 24);
        let area: f64 = mesh.triangles.iter().map(|[a,b,c]| {
            let p = mesh.points[*a as usize]; let q = mesh.points[*b as usize]; let r = mesh.points[*c as usize];
            let u = [q[0]-p[0], q[1]-p[1], q[2]-p[2]]; let v = [r[0]-p[0], r[1]-p[1], r[2]-p[2]];
            0.5 * ((u[1]*v[2]-u[2]*v[1]).powi(2) + (u[2]*v[0]-u[0]*v[2]).powi(2) + (u[0]*v[1]-u[1]*v[0]).powi(2)).sqrt()
        }).sum();
        assert!((area - 1.0).abs() < 1e-9, "{area}");
        assert_eq!(mesh.points[0], [0.,0.,0.]);
        assert!(tessellate_surface(&bilinear(), 0, 3).is_err());
    }

    #[test]
    fn trims_half_the_surface() {
        let surface = bilinear();
        let (u0, u1) = surface.u_domain();
        let (v0, v1) = surface.v_domain();
        let mid = (u0 + u1) / 2.0;
        let outer = [[u0, v0], [mid, v0], [mid, v1], [u0, v1]];
        let mesh = tessellate_trimmed_surface(&surface, 8, 4, &outer, &[]).unwrap();
        assert_eq!(mesh.triangles.len(), 4 * 4 * 2, "half the cells survive, two triangles each");
    }

    #[test]
    fn rejects_short_trim_polygon() {
        assert!(tessellate_trimmed_surface(&bilinear(), 4, 4, &[[0.0, 0.0]], &[]).is_err());
    }
}
