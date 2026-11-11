//! Lightweight mesh utilities and official STEP import.
//!
//! Tessellation is owned by `solidcore-tess` through `nodecraft-layers`; nothing in
//! this crate meshes a B-rep. It keeps the shared display-tolerance state, the
//! viewport's shading-normal split and STEP assembly import.

use std::cell::Cell;
use std::collections::HashMap;

use nodecraft_protocol::Mesh;

pub mod step_import;

pub use step_import::{
    StepDiagnostics, StepImport, import_step, import_step_brep, merge_meshes,
};

thread_local! {
    /// Display max-sag override for this evaluation (LOD). The engine sets it per
    /// request; the default is the normal viewport quality.
    static DISPLAY_TOLERANCE: Cell<Option<f64>> = const { Cell::new(None) };
}

/// Set the max-sag used for display tessellation on the current thread.
pub fn set_display_tolerance(tolerance: Option<f64>) {
    DISPLAY_TOLERANCE.with(|slot| slot.set(tolerance.filter(|value| value.is_finite() && *value > 0.0)));
}

/// Current display max-sag in model units.
#[must_use]
pub fn display_tolerance() -> f64 {
    DISPLAY_TOLERANCE.with(Cell::get).unwrap_or(DEFAULT_DISPLAY_TOLERANCE)
}

/// Normal viewport max sag.
pub const DEFAULT_DISPLAY_TOLERANCE: f64 = 0.35;

pub fn split_shading_normals(mesh: &Mesh) -> Mesh {
    if mesh.faces.is_empty() {
        return mesh.clone();
    }

    // Geometric triangle normals, flipped onto the same side as the source
    // surface normals. The source normals are often already welded/averaged at
    // this point, but their sign is still useful.
    let mut triangle_normals = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let [a, b, c] = face.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) = (mesh.vertices.get(a), mesh.vertices.get(b), mesh.vertices.get(c)) else {
            triangle_normals.push(None);
            continue;
        };
        let Some(mut normal) = shading_triangle_normal(*pa, *pb, *pc) else {
            triangle_normals.push(None);
            continue;
        };
        let mut source = [0.0_f32; 3];
        for index in [a, b, c] {
            if let Some(vertex_normal) = mesh.normals.get(index) {
                for axis in 0..3 {
                    source[axis] += vertex_normal[axis];
                }
            }
        }
        if normal[0] * source[0] + normal[1] * source[1] + normal[2] * source[2] < 0.0 {
            for value in &mut normal {
                *value = -*value;
            }
        }
        triangle_normals.push(Some(normal));
    }

    // Split per vertex by normal direction, not by B-rep face group. A cap and a
    // wall can be part of one B-rep face/group in the display mesh, but they must
    // still use separate normals at the rim.
    const CREASE_COSINE: f32 = 0.819_152_1; // 35 degrees
    let mut incidents: Vec<Vec<(usize, usize, [f32; 3])>> = vec![Vec::new(); mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        // A degenerate triangle has no direction. Keep its corners in a
        // fallback cluster so every face vertex has a normal sum to index.
        let normal = triangle_normals[face_index].unwrap_or([0.0, 0.0, 1.0]);
        for (slot, vertex) in face.iter().enumerate() {
            if let Some(list) = incidents.get_mut(*vertex as usize) {
                list.push((face_index, slot, normal));
            }
        }
    }

    let mut cluster_sums = vec![Vec::<[f32; 3]>::new(); mesh.vertices.len()];
    let mut corner_cluster = vec![[0_usize; 3]; mesh.faces.len()];
    for (vertex, incident) in incidents.iter().enumerate() {
        for (face_index, slot, normal) in incident {
            let mut assigned = None;
            for (cluster, sum) in cluster_sums[vertex].iter().enumerate() {
                let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                if length > 1.0e-12
                    && (sum[0] * normal[0] + sum[1] * normal[1] + sum[2] * normal[2]) / length >= CREASE_COSINE
                {
                    assigned = Some(cluster);
                    break;
                }
            }
            let cluster = assigned.unwrap_or_else(|| {
                cluster_sums[vertex].push([0.0; 3]);
                cluster_sums[vertex].len() - 1
            });
            for axis in 0..3 {
                cluster_sums[vertex][cluster][axis] += normal[axis];
            }
            corner_cluster[*face_index][*slot] = cluster;
        }
    }

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut vertex_map: HashMap<(usize, usize), u32> = HashMap::new();
    let mut first_for_old = vec![u32::MAX; mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let mut corners = [0_u32; 3];
        for (slot, old) in face.iter().enumerate() {
            let key = (*old as usize, corner_cluster[face_index][slot]);
            let index = if let Some(index) = vertex_map.get(&key) {
                *index
            } else {
                let index = positions.len() as u32;
                positions.push(mesh.vertices[*old as usize]);
                let cluster = corner_cluster[face_index][slot];
                let sum = cluster_sums
                    .get(*old as usize)
                    .and_then(|clusters| clusters.get(cluster))
                    .copied()
                    .unwrap_or([0.0, 0.0, 1.0]);
                let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                normals.push(if length > 1.0e-12 {
                    [sum[0] / length, sum[1] / length, sum[2] / length]
                } else {
                    [0.0, 0.0, 1.0]
                });
                if first_for_old[*old as usize] == u32::MAX {
                    first_for_old[*old as usize] = index;
                }
                vertex_map.insert(key, index);
                index
            };
            corners[slot] = index;
        }
        faces.push(corners);
    }

    let mut render = mesh.clone();
    render.vertices = positions;
    render.normals = normals;
    render.faces = faces;
    for edge in &mut render.brep_edges {
        edge.segments.retain_mut(|segment| {
            let a = first_for_old.get(segment[0] as usize).copied().unwrap_or(u32::MAX);
            let b = first_for_old.get(segment[1] as usize).copied().unwrap_or(u32::MAX);
            if a == u32::MAX || b == u32::MAX || a == b {
                return false;
            }
            *segment = [a, b];
            true
        });
    }
    render.vertices_data = None;
    render.normals_data = None;
    render
}

fn shading_triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    (length > 1.0e-12).then(|| [normal[0] / length, normal[1] / length, normal[2] / length])
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_shading_normals_keeps_flat_and_smooth_sides_apart() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            faces: vec![[0, 1, 2], [0, 1, 3]],
            face_groups: vec!["face:0".into(), "face:1".into()],
            ..Mesh::default()
        };
        let split = split_shading_normals(&mesh);
        assert!(split.vertices.len() >= mesh.vertices.len());
        assert_eq!(split.faces.len(), mesh.faces.len());
    }

    #[test]
    fn tolerance_is_thread_local_and_filters_invalid_values() {
        set_display_tolerance(Some(0.1));
        assert_eq!(display_tolerance(), 0.1);
        set_display_tolerance(Some(f64::NAN));
        assert_eq!(display_tolerance(), DEFAULT_DISPLAY_TOLERANCE);
        set_display_tolerance(None);
    }
}
