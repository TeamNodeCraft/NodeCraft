//! Display-mesh bridge from solidcore B-reps to the NodeCraft protocol.
//!
//! The first implementation handles planar polygonal faces (the box). Curved faces
//! will move to the ported `TessSrfCache` / poly-B-rep path.

pub use nodecraft_mesh as curved;
pub mod mesh_brep;
pub mod step_analytic;
pub mod step_nurbs;
pub mod step_mixed;
pub mod step_nurbs_faces;
pub mod step_curved;
pub mod step_import;
pub use step_curved::analytic_surface_from_step;
pub use step_mixed::{brep_from_mixed_step, brep_from_mixed_step_file};
pub use step_import::{
    StepBrepImport, StepBrepKind, StepMeshImport, brep_from_step, step_display_mesh,
    step_display_mesh_with_options,
};
pub use step_analytic::brep_from_analytic_step;
pub use step_nurbs::{bspline_curve_to_solidcore, bspline_surface_to_solidcore};
pub use step_nurbs_faces::{
    UvRectangle, brep_from_bspline_sheet, brep_from_bspline_sheet_file,
    brep_from_bspline_uv_rectangle,
};
pub use mesh_brep::{brep_from_planar_body, brep_from_planar_body_exact, brep_from_triangles, planar_body_display_mesh};
pub mod error;
pub mod kernel;
pub mod planar;
pub use solidcore;

use solidcore::kernel::Kernel;
use solidcore::math::{Point3, Vec3};
use solidcore::primitive::api_primitives as api;
use solidcore::query::api_queries::face_get_boundary_loops;
use solidcore::query::brep::{calculate_bounding_box, BoundingBoxOptions};
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::query::face::p1::evaluate_curve;
use nodecraft_protocol::{BrepEdge, Mesh};

/// A solidcore box converted to the centered NodeCraft primitive convention.
pub fn box_mesh(length: f64, width: f64, height: f64) -> Result<Mesh, String> {
    let mut kernel = crate::kernel::kernel();
    let brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), length, width, height)
        .map_err(|error| format!("solidcore create_box: {error}"))?;
    let extent = calculate_bounding_box(
        &kernel,
        &brep,
        BoundingBoxOptions {
            approx: false,
            tight: true,
            expand_by_zone_tol_3d: false,
        },
    )
    .map_err(|error| format!("solidcore bounding box: {error}"))?;
    let center = Vec3::new(
        (extent.min_at(0) + extent.max_at(0)) * 0.5,
        (extent.min_at(1) + extent.max_at(1)) * 0.5,
        (extent.min_at(2) + extent.max_at(2)) * 0.5,
    );
    let mut mesh = Mesh::default();
    let mut edge_index = 0u32;
    for (face_index, (face_id, _face)) in brep.faces().iter().enumerate() {
        let loops = face_get_boundary_loops(&brep, face_id)
            .map_err(|error| format!("solidcore face loops: {error}"))?;
        let mut offset = 0usize;
        for loop_size in &loops.edge_counts {
            let mut polygon: Vec<[f32; 3]> = Vec::new();
            for use_index in offset..offset + loop_size {
                let edge = brep
                    .edge(loops.edges[use_index])
                    .map_err(|error| format!("solidcore edge: {error}"))?;
                let forward = loops.orientations[use_index] == OrientType::Same;
                let parameter = if forward { edge.interval.evaluate(0.0) } else { edge.interval.evaluate(1.0) };
                let evaluated = evaluate_curve(&kernel, edge.curve.as_ref(), parameter)
                    .map_err(|error| format!("solidcore curve: {error}"))?;
                let point = [evaluated[0].x as f32, evaluated[0].y as f32, evaluated[0].z as f32];
                if polygon.last() != Some(&point) {
                    polygon.push(point);
                }
            }
            offset += loop_size;
            if polygon.len() < 3 {
                continue;
            }
            if polygon.first() == polygon.last() {
                polygon.pop();
            }
            if polygon.len() < 3 {
                continue;
            }
            let mut normal = triangle_normal(polygon[0], polygon[1], polygon[2]);
            let face_center = polygon
                .iter()
                .fold([0.0f32; 3], |sum, p| [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]]);
            let count = polygon.len() as f32;
            let outward = [
                face_center[0] / count - center.x as f32,
                face_center[1] / count - center.y as f32,
                face_center[2] / count - center.z as f32,
            ];
            let facing_out = normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2] >= 0.0;
            if !facing_out {
                normal = [-normal[0], -normal[1], -normal[2]];
                polygon.reverse();
            }
            let base = mesh.vertices.len() as u32;
            mesh.vertices.extend(polygon.iter().copied());
            mesh.normals.extend(std::iter::repeat_n(normal, polygon.len()));
            for index in 1..polygon.len() - 1 {
                mesh.faces.push([base, base + index as u32, base + index as u32 + 1]);
                mesh.face_groups.push(format!("face:{face_index}"));
            }
            for index in 0..polygon.len() {
                let next = (index + 1) % polygon.len();
                mesh.brep_edges.push(BrepEdge {
                    id: format!("mesh-edge:{edge_index}"),
                    kind: "mesh".into(),
                    segments: vec![[base + index as u32, base + next as u32]],
                });
                edge_index += 1;
            }
        }
    }
    for vertex in &mut mesh.vertices {
        vertex[0] -= (length * 0.5) as f32;
        vertex[1] -= (width * 0.5) as f32;
        vertex[2] -= (height * 0.5) as f32;
    }
    mesh.lightweight = true;
    if mesh.faces.is_empty() {
        return Err("solidcore box produced no triangles".into());
    }
    Ok(mesh)
}

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / length, n[1] / length, n[2] / length]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mesh_is_centered_and_closed() {
        let mesh = box_mesh(10.0, 4.0, 20.0).unwrap();
        assert_eq!(mesh.faces.len(), 12);
        assert_eq!(mesh.face_groups.len(), 12);
        assert!(!mesh.brep_edges.is_empty());
        let xs: Vec<f32> = mesh.vertices.iter().map(|p| p[0]).collect();
        let zs: Vec<f32> = mesh.vertices.iter().map(|p| p[2]).collect();
        assert!((xs.iter().cloned().fold(f32::INFINITY, f32::min) + 5.0).abs() < 1e-6);
        assert!((xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 5.0).abs() < 1e-6);
        assert!((zs.iter().cloned().fold(f32::INFINITY, f32::min) + 10.0).abs() < 1e-6);
        assert!((zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 10.0).abs() < 1e-6);
    }
}
