//! Conversion from the NodeCraft protocol mesh to SolidCraft geometry types.
//!
//! This is deliberately a one-way adapter. SolidCore geometry is never modified by
//! the UI, and the SolidCraft crates stay upstream-owned.

use std::collections::HashMap;
use std::sync::Arc;

use solidcraft_geom::{Mesh as CraftMesh, Vec3};
use nodecraft_protocol::Mesh as NodeMesh;

/// A display mesh plus its edge polylines, ready for the SolidCraft CPU rasterizer.
pub struct DisplayMesh {
    pub mesh: Arc<CraftMesh>,
    /// Face-group ids indexed by `mesh.tri_face`.
    pub face_ids: Vec<String>,
    pub radius: f64,
    pub center: Vec3,
}

/// Convert a NodeCraft display mesh.
///
/// `tri_face` uses the first appearance of each NodeCraft face group as its face index;
/// this preserves per-face colouring if the UI later adds a face-colour palette.
#[must_use]
pub fn display_mesh(source: &NodeMesh) -> DisplayMesh {
    // SolidCraft does not draw tangent joins. SolidCore's B-rep edge list contains
    // every face boundary, including the tessellation strips of a loft. Build the
    // display edges from actual triangle adjacency instead of trusting that list.
    let raw = source.clone();
    // Shading needs crease-split normals; the protocol mesh stays welded for
    // topology, measurement and export.
    let split = nodecraft_mesh::split_shading_normals(&raw);
    let source = &split;
    let positions: Vec<Vec3> = source
        .vertices
        .iter()
        .map(|v| Vec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])))
        .collect();

    let mut normals = if source.normals.len() == source.vertices.len() {
        source
            .normals
            .iter()
            .map(|n| Vec3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2])))
            .collect::<Vec<_>>()
    } else {
        vec![Vec3::ZERO; positions.len()]
    };
    if source.normals.len() != source.vertices.len() {
        for [a, b, c] in &source.faces {
            let (a, b, c) = (*a as usize, *b as usize, *c as usize);
            let Some(pa) = positions.get(a).copied() else { continue };
            let Some(pb) = positions.get(b).copied() else { continue };
            let Some(pc) = positions.get(c).copied() else { continue };
            let n = (pb - pa).cross(pc - pa);
            if let Some(slot) = normals.get_mut(a) { *slot = *slot + n; }
            if let Some(slot) = normals.get_mut(b) { *slot = *slot + n; }
            if let Some(slot) = normals.get_mut(c) { *slot = *slot + n; }
        }
        for normal in &mut normals {
            *normal = normal.normalized().unwrap_or(Vec3::Z);
        }
    }

    let mut face_index = std::collections::BTreeMap::new();
    let tri_face = (0..source.faces.len())
        .map(|index| {
            let group = source.face_groups.get(index).cloned().unwrap_or_else(|| format!("face:{index}"));
            let next = face_index.len() as u32;
            *face_index.entry(group).or_insert(next)
        })
        .collect();
    let mut face_ids = vec![String::new(); face_index.len()];
    for (name, index) in face_index {
        if let Some(slot) = face_ids.get_mut(index as usize) { *slot = name; }
    }

    let edges = crease_edges(&raw);

    let mesh = CraftMesh {
        positions,
        normals,
        triangles: source.faces.clone(),
        tri_face,
        edges,
        display_only_edges: true,
        seams: Vec::new(),
        smooth_edges: Vec::new(),
        sharp_edges: Vec::new(),
        edge_faces: Vec::new(),
    };
    let bounds = mesh.bounds();
    let center = bounds.center();
    let radius = (bounds.diagonal() * 0.5).max(1.0);
    DisplayMesh { mesh: Arc::new(mesh), face_ids, radius, center }
}

/// Build display edges from triangle adjacency. A joined edge is shown only when
/// the adjacent triangle normals differ by a real crease; tangent tessellation
/// strips and smooth Boolean seams disappear.
fn crease_edges(mesh: &NodeMesh) -> Vec<Vec<Vec3>> {
    use std::collections::HashMap;
    const CREASE_COSINE: f32 = 0.866_025_4; // 30 degrees
    let mut incident: HashMap<(u32, u32), Vec<[f32; 3]>> = HashMap::new();
    for [a, b, c] in &mesh.faces {
        let (Some(pa), Some(pb), Some(pc)) = (
            mesh.vertices.get(*a as usize),
            mesh.vertices.get(*b as usize),
            mesh.vertices.get(*c as usize),
        ) else {
            continue;
        };
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length <= 1.0e-12 {
            continue;
        }
        let normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        for pair in [[*a, *b], [*b, *c], [*c, *a]] {
            let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
            incident.entry(key).or_default().push(normal);
        }
    }
    let mut edges = Vec::new();
    for (key, normals) in incident {
        if normals.is_empty() {
            continue;
        }
        let mut crease = normals.len() == 1;
        'outer: for first in 0..normals.len() {
            for second in first + 1..normals.len() {
                let a = normals[first];
                let b = normals[second];
                if a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < CREASE_COSINE {
                    crease = true;
                    break 'outer;
                }
            }
        }
        if !crease {
            continue;
        }
        let (Some(a), Some(b)) = (mesh.vertices.get(key.0 as usize), mesh.vertices.get(key.1 as usize)) else {
            continue;
        };
        let a = Vec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]));
        let b = Vec3::new(f64::from(b[0]), f64::from(b[1]), f64::from(b[2]));
        if a.dist(b) > 1.0e-9 {
            edges.push(vec![a, b]);
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::{BrepEdge, Mesh};

    #[test]
    fn converts_a_triangle_and_edge() {
        let source = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            normals: vec![],
            faces: vec![[0, 1, 2]],
            face_groups: vec!["face:0".into()],
            brep_edges: vec![BrepEdge {
                id: "edge:0".into(),
                kind: "line".into(),
                segments: vec![[0, 1]],
            }],
            ..Mesh::default()
        };
        let display = display_mesh(&source);
        assert_eq!(display.mesh.triangles.len(), 1);
        assert_eq!(display.mesh.tri_face, vec![0]);
        assert_eq!(display.face_ids, vec!["face:0".to_owned()]);
        assert_eq!(display.mesh.edges.len(), 3);
        assert!(display.radius > 0.0);
    }
}
