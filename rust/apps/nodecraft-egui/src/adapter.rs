//! Conversion from the NodeCraft protocol mesh to SolidCraft geometry types.
//!
//! This is deliberately a one-way adapter. SolidCore geometry is never modified by
//! the UI, and the SolidCraft crates stay upstream-owned.

use std::sync::Arc;

use solidcraft_geom::Mesh as CraftMesh;
use solidcore_record_new::Vec3;
use nodecraft_protocol::Mesh as NodeMesh;
use solidcraft_geom::prelude::*;

/// A display mesh plus its edge polylines, ready for the SolidCraft CPU rasterizer.
pub struct DisplayMesh {
    pub mesh: Arc<CraftMesh>,
    /// Face-group ids indexed by `mesh.tri_face`.
    pub face_ids: Vec<String>,
    /// Optional per-triangle sRGB colour, parallel to `mesh.triangles`.
    pub face_colors: Vec<[f32; 3]>,
    /// Pickable B-rep edges in world coordinates, with body-scoped ids.
    pub edges: Vec<DisplayEdge>,
    /// Feature-comparison layers, if this display is a before/after view.
    pub comparison: Option<ComparisonMesh>,
    /// Edges offered to a topology session when the visible body is the result.
    pub selection_edges: Option<Vec<DisplayEdge>>,
    pub radius: f64,
    pub center: Vec3,
}


#[derive(Clone, Debug, Default)]
pub struct ComparisonMesh {
    pub added: Vec<([Vec3; 3], [Vec3; 3])>,
    pub removed: Vec<([Vec3; 3], [Vec3; 3])>,
    /// Difference-body edges, drawn as a separate overlay edge layer.
    pub edges: Vec<DisplayEdge>,
}

#[derive(Clone, Debug)]
pub struct DisplayEdge {
    pub id: String,
    pub points: Vec<Vec3>,
    /// Midpoint of the selected edge in display coordinates, used as a robust
    /// geometric fallback when resolving the exact B-rep edge.
    pub midpoint: [f64; 3],
    /// Adjacent triangle normals for depth-lifting each polyline point.
    pub normals: Vec<[f32; 3]>,
}

/// Convert a NodeCraft display mesh.
///
/// `tri_face` uses the first appearance of each NodeCraft face group as its face index;
/// this preserves per-face colouring if the UI later adds a face-colour palette.
#[must_use]
pub fn display_mesh(source: &NodeMesh) -> DisplayMesh {
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

    // Average the normals of triangles incident on each quantized edge segment.
    // `lifted_line` uses these to pull the line toward the eye by half its width,
    // which avoids z-fighting while keeping hidden edges depth-tested.
    let quantized = |point: [f32; 3]| {
        [
            (point[0] * 1.0e5).round() as i64,
            (point[1] * 1.0e5).round() as i64,
            (point[2] * 1.0e5).round() as i64,
        ]
    };
    let mut edge_normals: std::collections::HashMap<[i64; 6], ([f32; 3], usize)> =
        std::collections::HashMap::new();
    for [a, b, c] in &raw.faces {
        let (Some(pa), Some(pb), Some(pc)) = (
            raw.vertices.get(*a as usize),
            raw.vertices.get(*b as usize),
            raw.vertices.get(*c as usize),
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
            let (Some(first), Some(second)) = (raw.vertices.get(pair[0] as usize), raw.vertices.get(pair[1] as usize)) else {
                continue;
            };
            let (ka, kb) = (quantized(*first), quantized(*second));
            let key = if ka < kb {
                [ka[0], ka[1], ka[2], kb[0], kb[1], kb[2]]
            } else {
                [kb[0], kb[1], kb[2], ka[0], ka[1], ka[2]]
            };
            let entry = edge_normals.entry(key).or_insert(([0.0; 3], 0));
            entry.0[0] += normal[0];
            entry.0[1] += normal[1];
            entry.0[2] += normal[2];
            entry.1 += 1;
        }
    }
    let edge_normal = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
        let (ka, kb) = (quantized(a), quantized(b));
        let key = if ka < kb {
            [ka[0], ka[1], ka[2], kb[0], kb[1], kb[2]]
        } else {
            [kb[0], kb[1], kb[2], ka[0], ka[1], ka[2]]
        };
        let Some((normal, count)) = edge_normals.get(&key) else { return [0.0; 3] };
        let count = (*count).max(1) as f32;
        let normal = [normal[0] / count, normal[1] / count, normal[2] / count];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length <= 1.0e-6 { [0.0; 3] } else { [normal[0] / length, normal[1] / length, normal[2] / length] }
    };

    let pick_edges: Vec<DisplayEdge> = raw
        .brep_edges
        .iter()
        // `kind == "mesh"` (and `mesh-edge:` ids) are display-only fallbacks for
        // bodies without B-rep topology. The overlay must only expose real B-rep
        // edges, which carry an exact edge id.
        .filter(|edge| edge.kind != "mesh" && !edge.id.starts_with("mesh-edge:"))
        .filter_map(|edge| {
            let mut points = Vec::new();
            let mut normals = Vec::new();
            for segment in &edge.segments {
                for index in segment {
                    let point = raw.vertices.get(*index as usize)?;
                    let point = Vec3::new(f64::from(point[0]), f64::from(point[1]), f64::from(point[2]));
                    if points.last().copied() != Some(point) {
                        points.push(point);
                        let next = raw.vertices.get(*index as usize)?;
                        let key_sorted = edge.segments.iter().find_map(|candidate| {
                            if candidate.contains(index) {
                                let other = if candidate[0] == *index { candidate[1] } else { candidate[0] };
                                raw.vertices.get(other as usize).map(|other| (*next, *other))
                            } else {
                                None
                            }
                        });
                        normals.push(key_sorted.map_or([0.0; 3], |(a, b)| edge_normal(a, b)));
                    }
                }
            }
            (points.len() >= 2).then(|| {
                // A point on the edge, not the centroid: this is the geometric
                // fallback used to resolve the exact B-rep edge. For a closed
                // circle the centroid lies at its centre and is not on the edge.
                let midpoint = {
                    let point = points[points.len() / 2];
                    [point.x, point.y, point.z]
                };
                DisplayEdge { id: edge.id.clone(), points, midpoint, normals }
            })
        })
        .collect();
    // Display the exact B-rep boundaries, not triangle adjacency. Generated
    // tessellation strips and mesh fallbacks must not appear as selectable edges.
    let edges: Vec<Vec<Vec3>> = pick_edges
        .iter()
        .map(|edge| edge.points.clone())
        .collect();

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
    DisplayMesh {
        mesh: Arc::new(mesh),
        face_ids,
        face_colors: if source.face_colors.len() == source.faces.len() {
            source.face_colors.clone()
        } else {
            Vec::new()
        },
        edges: pick_edges,
        comparison: None,
        selection_edges: None,
        radius,
        center,
    }
}

#[must_use]
pub fn display_mesh_with_difference(
    base: &NodeMesh,
    added: &NodeMesh,
    removed: &NodeMesh,
) -> DisplayMesh {
    let mut display = display_mesh(base);
    display.comparison = Some(ComparisonMesh {
        added: mesh_triangles(added),
        removed: mesh_triangles(removed),
        edges: {
            let mut edges: Vec<DisplayEdge> = comparison_edges(added)
                .into_iter()
                .chain(comparison_edges(removed))
                .collect();
            if edges.is_empty() {
                // Some difference bodies arrive with no BRep edge metadata.
                // Fall back to the boundary edges of their display triangles.
                edges.extend(boundary_edges(&mesh_triangles(added)));
                edges.extend(boundary_edges(&mesh_triangles(removed)));
            }
            edges
        },
    });
    display
}

fn comparison_edges(mesh: &NodeMesh) -> Vec<DisplayEdge> {
    mesh.brep_edges
        .iter()
        .filter_map(|edge| {
            let mut points: Vec<Vec3> = Vec::new();
            for segment in &edge.segments {
                for index in segment {
                    let point = mesh.vertices[*index as usize];
                    let point = Vec3::new(f64::from(point[0]), f64::from(point[1]), f64::from(point[2]));
                    if points.last().is_none_or(|last| *last != point) {
                        points.push(point);
                    }
                }
            }
            if points.len() < 2 { return None; }
            let point_count = points.len();
            let midpoint = points.iter().fold([0.0; 3], |mut sum, point| {
                sum[0] += point.x / points.len() as f64;
                sum[1] += point.y / points.len() as f64;
                sum[2] += point.z / points.len() as f64;
                sum
            });
            Some(DisplayEdge {
                id: format!("comparison:{}", edge.id),
                points,
                midpoint,
                normals: vec![[0.0, 1.0, 0.0]; point_count],
            })
        })
        .collect()
}

fn boundary_edges(triangles: &[([Vec3; 3], [Vec3; 3])]) -> Vec<DisplayEdge> {
    use std::collections::HashMap;
    let mut counts: HashMap<([i64; 3], [i64; 3]), (Vec3, Vec3)> = HashMap::new();
    let key = |p: Vec3| [
        (p.x * 1.0e6).round() as i64,
        (p.y * 1.0e6).round() as i64,
        (p.z * 1.0e6).round() as i64,
    ];
    for (points, _) in triangles {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (pa, pb) = (points[a], points[b]);
            let (ka, kb) = (key(pa), key(pb));
            let ordered = if ka <= kb { (ka, kb) } else { (kb, ka) };
            let entry = counts.entry(ordered).or_insert((pa, pb));
            let _ = entry;
            counts.entry(ordered).and_modify(|_| {}).or_insert((pa, pb));
        }
    }
    let mut seen = HashMap::new();
    for (points, _) in triangles {
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (pa, pb) = (points[a], points[b]);
            let (ka, kb) = (key(pa), key(pb));
            let ordered = if ka <= kb { (ka, kb) } else { (kb, ka) };
            *seen.entry(ordered).or_insert(0usize) += 1;
        }
    }
    seen.into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(k, _)| {
            let (pa, pb) = counts[&k];
            DisplayEdge {
                id: format!("comparison-boundary:{k:?}"),
                points: vec![pa, pb],
                midpoint: [
                    (pa.x + pb.x) * 0.5,
                    (pa.y + pb.y) * 0.5,
                    (pa.z + pb.z) * 0.5,
                ],
                normals: vec![[0.0, 1.0, 0.0]; 2],
            }
        })
        .collect()
}

fn mesh_triangles(mesh: &NodeMesh) -> Vec<([Vec3; 3], [Vec3; 3])> {
    mesh.faces
        .iter()
        .map(|triangle| {
            let points = triangle.map(|corner| {
                let point = mesh.vertices[corner as usize];
                Vec3::new(f64::from(point[0]), f64::from(point[1]), f64::from(point[2]))
            });
            let normal = if !mesh.normals.is_empty() {
                let normal = mesh.normals[triangle[0] as usize];
                Vec3::new(f64::from(normal[0]), f64::from(normal[1]), f64::from(normal[2]))
            } else {
                (points[1] - points[0]).cross(points[2] - points[0]).normalized().unwrap_or(Vec3::Z)
            };
            (points, [normal; 3])
        })
        .collect()
}




#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::{BrepEdge, Mesh};

    #[test]
    fn closed_edge_midpoint_lies_on_the_edge() {
        let source = Mesh {
            vertices: vec![
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, -10.0, 0.0],
            ],
            normals: vec![],
            faces: vec![],
            face_groups: vec![],
            brep_edges: vec![BrepEdge {
                id: "edge:0".into(),
                kind: "edge".into(),
                segments: vec![[0, 1], [1, 2], [2, 3], [3, 0]],
            }],
            ..Mesh::default()
        };
        let display = display_mesh(&source);
        let midpoint = display.edges[0].midpoint;
        let radius = (midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1]).sqrt();
        assert!((radius - 10.0).abs() < 1.0e-6, "midpoint {midpoint:?} is not on the circle");
    }

    #[test]
    fn converts_a_triangle_and_edge() {
        let source = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            normals: vec![],
            faces: vec![[0, 1, 2]],
            face_groups: vec!["face:0".into()],
            brep_edges: vec![
                BrepEdge {
                    id: "edge:0".into(),
                    kind: "edge".into(),
                    segments: vec![[0, 1]],
                },
                BrepEdge {
                    id: "mesh-edge:0".into(),
                    kind: "mesh".into(),
                    segments: vec![[1, 2]],
                },
            ],
            ..Mesh::default()
        };
        let display = display_mesh(&source);
        assert_eq!(display.mesh.triangles.len(), 1);
        assert_eq!(display.mesh.tri_face, vec![0]);
        assert_eq!(display.face_ids, vec!["face:0".to_owned()]);
        assert_eq!(display.edges.len(), 1, "generated mesh edges must not be pickable");
        assert_eq!(display.edges[0].id, "edge:0");
        assert_eq!(display.mesh.edges, vec![vec![
            solidcore_record_new::Vec3::new(0.0, 0.0, 0.0),
            solidcore_record_new::Vec3::new(10.0, 0.0, 0.0),
        ]]);
        assert!(display.radius > 0.0);
    }
}
