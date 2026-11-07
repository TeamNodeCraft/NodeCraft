//! Display-mesh utilities and official STEP import.
//!
//! Tessellation is owned by `solidcore-tess` through `nodecraft-layers`; the
//! compatibility wrappers here exist only for callers that still need a mesh
//! from a B-rep. No local tessellator remains.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};

use nodecraft_protocol::Mesh;
use solidcore::Brep;
use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3, Vec3};
use solidcore::nurbs::knots::Side;
use solidcore::topology::face::p2 as fp2;

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

/// Compatibility options for the display wrappers below. Only the sag is used;
/// the grid fields are retained for callers that still pass them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedTessOptions {
    pub u_segments: usize,
    pub v_segments: usize,
    pub loop_segments: usize,
    pub bspline_tolerance: f64,
    pub max_normal_angle_deg: f64,
}

impl Default for CurvedTessOptions {
    fn default() -> Self {
        Self {
            u_segments: 16,
            v_segments: 4,
            loop_segments: 8,
            bspline_tolerance: display_tolerance(),
            max_normal_angle_deg: 0.0,
        }
    }
}

/// Tessellate a B-rep through `solidcore-tess` via the layer seam.
pub fn brep_display_mesh(
    _kernel: &Kernel,
    brep: &Brep,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    if brep.faces().len() == 0 {
        return Err("display tessellation produced no triangles".into());
    }
    let mut tess_kernel = nodecraft_kernel_profile::kernel();
    let mut copy = brep
        .clone_standing_for_self()
        .map_err(|error| format!("display copy: {error}"))?;
    let sag = options.bspline_tolerance.max(0.05);
    let mut mesh = nodecraft_layers::tessellate_kernel(&mut tess_kernel, &mut copy, sag)
        .map(|display| display.mesh)?;
    if mesh.faces.is_empty() {
        return Err("display tessellation produced no triangles".into());
    }
    mesh.lightweight = true;
    Ok(mesh)
}

/// [`brep_display_mesh`], with the STEP `same_sense` flags of every face.
pub fn brep_display_mesh_with_senses(
    kernel: &Kernel,
    brep: &Brep,
    options: &CurvedTessOptions,
    senses: Option<&[bool]>,
) -> Result<Mesh, String> {
    let mut mesh = brep_display_mesh(kernel, brep, options)?;
    if let Some(senses) = senses {
        orient_mesh_to_senses(kernel, brep, &mut mesh, senses);
    }
    Ok(mesh)
}

pub(crate) fn orient_mesh_to_senses(kernel: &Kernel, brep: &Brep, mesh: &mut Mesh, senses: &[bool]) {
    let faces = brep.face_list();
    if faces.len() != senses.len() {
        return;
    }
    for (index, (&face_id, &same_sense)) in faces.iter().zip(senses).enumerate() {
        let Ok(domain) = fp2::uv_domain(brep, face_id) else {
            continue;
        };
        let Ok(Some(surface)) = fp2::surface(brep, face_id) else {
            continue;
        };
        let u = (domain.u_interval().min() + domain.u_interval().max()) * 0.5;
        let v = (domain.v_interval().min() + domain.v_interval().max()) * 0.5;
        let Ok(normal) = surface.evaluate_normal(kernel, Point2::new(u, v), Side::Left, Side::Left)
        else {
            continue;
        };
        let Ok(normal) = normal.unitize() else {
            continue;
        };
        let desired = if same_sense {
            normal
        } else {
            Vec3::new(-normal.x, -normal.y, -normal.z)
        };
        let group = format!("face:{index}");
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        for (triangle, triangle_group) in mesh.faces.iter().zip(mesh.face_groups.iter()) {
            if triangle_group != &group {
                continue;
            }
            let pa = point_from_vertex(mesh.vertices[triangle[0] as usize]);
            let pb = point_from_vertex(mesh.vertices[triangle[1] as usize]);
            let pc = point_from_vertex(mesh.vertices[triangle[2] as usize]);
            let u = Vec3::new(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
            let v = Vec3::new(pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);
            sum = sum + u.cross(v);
        }
        if sum.dot(desired) < 0.0 {
            flip_face_group(mesh, index);
        }
    }
}

/// Reverses the triangles and normals of one `face:{index}` display group.
fn flip_face_group(mesh: &mut Mesh, face_index: usize) {
    let group = format!("face:{face_index}");
    let Mesh { faces, face_groups, normals, .. } = mesh;
    let mut vertices: BTreeSet<u32> = BTreeSet::new();
    for (triangle, triangle_group) in faces.iter_mut().zip(face_groups.iter()) {
        if triangle_group != &group {
            continue;
        }
        triangle.swap(1, 2);
        vertices.extend(triangle.iter().copied());
    }
    for vertex in vertices {
        if let Some(normal) = normals.get_mut(vertex as usize) {
            *normal = [-normal[0], -normal[1], -normal[2]];
        }
    }
}


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


fn point_from_vertex(vertex: [f32; 3]) -> Point3 {
    Point3::new(vertex[0] as f64, vertex[1] as f64, vertex[2] as f64)
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
