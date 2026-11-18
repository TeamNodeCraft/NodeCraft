//! Conversion from the NodeCraft protocol mesh to SolidCraft geometry types.
//!
//! This is deliberately a one-way adapter. SolidCore geometry is never modified by
//! the UI, and the SolidCraft crates stay upstream-owned.

use std::sync::Arc;

use solidcraft_geom::Mesh as CraftMesh;
use solidcore_record::Vec3;
use nodecraft_protocol::Mesh as NodeMesh;
use solidcraft_geom::prelude::*;

/// A display mesh plus its edge polylines, ready for the SolidCraft CPU rasterizer.
impl DisplayMesh {
    /// The mesh to draw for a display LOD index. Index 0 is the full mesh.
    #[must_use]
    pub fn render_mesh(&self, lod: usize) -> &CraftMesh {
        self.lods
            .get(lod.saturating_sub(1))
            .map_or(&self.mesh, |lod| &lod.mesh)
    }
}

#[derive(Clone)]
pub struct DisplayMesh {
    pub mesh: Arc<CraftMesh>,
    /// Display-only simplification levels, coarse after the full mesh.
    pub lods: Vec<DisplayLod>,
    /// Face-group ids indexed by `mesh.tri_face`.
    pub face_ids: Vec<String>,
    /// True when `mesh.normals` are the evaluated surface normals (already
    /// split per B-rep face), not normals rebuilt for display.
    pub exact_normals: bool,
    /// Optional per-triangle sRGB colour, parallel to `mesh.triangles`.
    pub face_colors: Vec<[f32; 3]>,
    /// Stable identity per body index (the `body:N` prefix in `face_ids`),
    /// supplied by the plan set. Empty falls back to a synthetic per-index id.
    pub body_ids: Vec<u64>,
    /// Pickable B-rep edges in world coordinates, with body-scoped ids.
    pub edges: Vec<DisplayEdge>,
    /// Feature-comparison layers, if this display is a before/after view.
    pub comparison: Option<ComparisonMesh>,
    /// Edges offered to a topology session when the visible body is the result.
    pub selection_edges: Option<Vec<DisplayEdge>>,
    pub radius: f64,
    pub center: Vec3,
}


#[derive(Clone, Debug)]
pub struct DisplayLod {
    pub mesh: Arc<CraftMesh>,
    /// Maximum deviation from the full mesh, retained for diagnostics.
    #[allow(dead_code)]
    pub error: f64,
    /// Triangle count of this level, retained for diagnostics.
    #[allow(dead_code)]
    pub triangles: usize,
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
    /// The one tessellator class, carried to the CPU/world edge path.
    pub class: nodecraft_layers::PlanEdgeClass,
}

#[must_use]
pub fn generate_lods(source: &NodeMesh) -> Vec<DisplayLod> {
    // Small primitives such as a sphere are still worth a coarse level: it
    // gives zooming in a visible full-detail transition instead of a fixed mesh.
    const MIN_TRIANGLES: usize = 500;
    let full = source.faces.len();
    if full < MIN_TRIANGLES || !source.face_colors.is_empty() {
        return Vec::new();
    }
    let targets: Vec<usize> = [full / 2, full / 4, full / 8]
        .into_iter()
        .filter(|target| *target >= 100)
        .collect();
    if targets.is_empty() {
        return Vec::new();
    }
    nodecraft_layers::simplify_mesh(source, &targets)
        .into_iter()
        .filter_map(|level| {
            let triangles = level.triangles.len();
            if triangles == 0 || triangles >= full {
                return None;
            }
            let mut mesh = NodeMesh {
                vertices: level.positions,
                faces: level.triangles,
                normals: Vec::new(),
                exact_normals: false,
                vertices_data: None,
                normals_data: None,
                face_groups: Vec::new(),
                face_colors: Vec::new(),
                brep_edges: Vec::new(),
                lightweight: true,
                feature_preview: None,
                feature_preview_operation: None,
            };
            mesh = nodecraft_mesh::split_shading_normals(&mesh);
            let positions: Vec<Vec3> = mesh
                .vertices
                .iter()
                .map(|p| Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2])))
                .collect();
            let normals: Vec<Vec3> = mesh
                .normals
                .iter()
                .map(|n| Vec3::new(f64::from(n[0]), f64::from(n[1]), f64::from(n[2])))
                .collect();
            let mut tri_face = vec![0; mesh.faces.len()];
            tri_face.iter_mut().for_each(|face| *face = 0);
            Some(DisplayLod {
                mesh: Arc::new(CraftMesh {
                    positions,
                    normals,
                    triangles: mesh.faces,
                    tri_face,
                    edges: Vec::new(),
                    display_only_edges: false,
                    seams: Vec::new(),
                    smooth_edges: Vec::new(),
                    sharp_edges: Vec::new(),
                    edge_faces: Vec::new(),
                }),
                error: level.error,
                triangles,
            })
        })
        .collect()
}

/// Convert a NodeCraft display mesh.
///
/// `tri_face` uses the first appearance of each NodeCraft face group as its face index;
/// this preserves per-face colouring if the UI later adds a face-colour palette.
#[must_use]
pub fn display_mesh(source: &NodeMesh) -> DisplayMesh {
    let raw = source.clone();
    // A B-rep plan's evaluated normals are already split per face and per
    // incident triangle at a pole. Rebuilding them with the crease rule would
    // average a cone apex's limit normals toward the axis. Only meshes that do
    // not carry exact normals (imported/faceted bodies, LODs) need the split.
    let use_exact = raw.exact_normals && raw.normals.len() == raw.vertices.len();
    let split = (!use_exact).then(|| nodecraft_mesh::split_shading_normals(&raw));
    let source = split.as_ref().unwrap_or(&raw);
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

    let pick_edges: Vec<DisplayEdge> = raw
        .brep_edges
        .iter()
        // `kind == "mesh"` (and `mesh-edge:` ids) are display-only fallbacks for
        // bodies without B-rep topology. The overlay must only expose real B-rep
        // edges, which carry an exact edge id.
        .filter(|edge| edge.kind != "mesh" && !edge.id.starts_with("mesh-edge:"))
        .filter_map(|edge| {
            let mut points = Vec::new();
            for segment in &edge.segments {
                for index in segment {
                    let point = raw.vertices.get(*index as usize)?;
                    let point = Vec3::new(f64::from(point[0]), f64::from(point[1]), f64::from(point[2]));
                    if points.last().copied() != Some(point) {
                        points.push(point);
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
                DisplayEdge {
                    id: edge.id.clone(),
                    points,
                    midpoint,
                    class: nodecraft_layers::PlanEdgeClass::from_bits(edge.class),
                }
            })
        })
        .collect();
    // Display the exact B-rep boundaries, not triangle adjacency. Generated
    // tessellation strips and mesh fallbacks must not appear as selectable edges.
    let edges: Vec<Vec<Vec3>> = pick_edges
        .iter()
        .map(|edge| edge.points.clone())
        .collect();
    // The CPU scene path reads the same per-edge classes as the GPU ribbon.
    let seams: Vec<bool> = pick_edges.iter().map(|edge| edge.class.is_seam()).collect();
    let smooth_edges: Vec<bool> = pick_edges.iter().map(|edge| edge.class.is_smooth()).collect();

    let mesh = CraftMesh {
        positions,
        normals,
        triangles: source.faces.clone(),
        tri_face,
        edges,
        display_only_edges: true,
        seams,
        smooth_edges,
        sharp_edges: Vec::new(),
        edge_faces: Vec::new(),
    };
    let bounds = mesh.bounds();
    let center = bounds.center();
    let radius = (bounds.diagonal() * 0.5).max(1.0);
    // Plan-backed displays replace these with the plans' real identities in
    // `Viewport::display_mesh_for_view`; imports and other plan-less meshes use
    // the synthetic per-body key from `body_instance_id`.
    let body_count = face_ids
        .iter()
        .filter_map(|name| {
            name.strip_prefix("body:")
                .and_then(|rest| rest.split_once(':'))
                .and_then(|(body, _)| body.parse::<u32>().ok())
        })
        .max()
        .map_or(1, |body| body.saturating_add(1));
    let body_ids = (0..body_count)
        .map(|body| {
            nodecraft_display::body_instance_id(nodecraft_display::BodyKey::new(0, body), 0)
        })
        .collect();
    DisplayMesh {
        mesh: Arc::new(mesh),
        lods: Vec::new(),
        face_ids,
        exact_normals: use_exact,
        face_colors: if source.face_colors.len() == source.faces.len() {
            source.face_colors.clone()
        } else {
            Vec::new()
        },
        body_ids,
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
            if edges.is_empty() && (!added.face_groups.is_empty() || !removed.face_groups.is_empty()) {
                // Some exact difference bodies arrive with no BRep edge
                // metadata. Face-classified overlays have no face groups, so
                // their triangle edges must never become a jagged outline.
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
                class: nodecraft_layers::PlanEdgeClass::from_bits(edge.class),
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
                class: nodecraft_layers::PlanEdgeClass::NORMAL,
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
                class: 0,
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
                    class: 0,
                },
                BrepEdge {
                    id: "mesh-edge:0".into(),
                    kind: "mesh".into(),
                    segments: vec![[1, 2]],
                    class: 0,
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
            solidcore_record::Vec3::new(0.0, 0.0, 0.0),
            solidcore_record::Vec3::new(10.0, 0.0, 0.0),
        ]]);
        assert!(display.radius > 0.0);
    }

    #[test]
    fn small_display_meshes_can_generate_a_coarse_lod() {
        let n = 20u32;
        let mut vertices = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        let width = n + 1;
        let mut faces = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * width + x;
                faces.push([a, a + 1, a + width + 1]);
                faces.push([a, a + width + 1, a + width]);
            }
        }
        let source = NodeMesh { vertices, faces, ..NodeMesh::default() };
        assert!(!generate_lods(&source).is_empty(), "~800 triangles should have a coarse LOD");
    }

    #[test]
    fn large_display_meshes_get_coarse_lods() {
        let n = 120u32;
        let mut vertices = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                vertices.push([x as f32, y as f32, ((x + y) % 3) as f32 * 0.1]);
            }
        }
        let width = n + 1;
        let mut faces = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * width + x;
                let b = a + 1;
                let c = a + width;
                let d = c + 1;
                faces.push([a, b, d]);
                faces.push([a, d, c]);
            }
        }
        let source = NodeMesh {
            vertices,
            faces,
            ..NodeMesh::default()
        };
        let mut display = display_mesh(&source);
        display.lods = generate_lods(&source);
        assert!(!display.lods.is_empty(), "large display should have LOD levels");
        assert!(
            display.render_mesh(1).triangles.len() < display.mesh.triangles.len(),
            "first LOD should be coarser than the full mesh"
        );
        assert!(display.lods.iter().all(|lod| lod.error.is_finite()));
    }

    fn exact_display(brep: nodecraft_solidcore::solidcore::topology::Brep) -> (DisplayMesh, NodeMesh) {
        let part = nodecraft_display::ExactPart::singleton(
            nodecraft_display::BodyKey::new(1, 0),
            Arc::new(brep),
        );
        let mut cache = nodecraft_display::PlanCache::new(4);
        let tessellated = nodecraft_display::mesh_part(
            &part,
            nodecraft_display::DisplayQuality::Full(0.1),
            &mut cache,
        )
        .expect("mesh part");
        assert!(
            tessellated.mesh.exact_normals,
            "plan evaluation must mark its evaluated normals exact"
        );
        let evaluated = nodecraft_layers::evaluate_plan(
            &tessellated.plans[0].plan,
            tessellated.plans[0].edge_ids.as_ref(),
        );
        assert!(
            evaluated.mesh.exact_normals,
            "re-evaluating the plan must keep the exact-normal flag"
        );
        (display_mesh(&tessellated.mesh), evaluated.mesh)
    }

    /// Compare every triangle corner with the untouched plan evaluation. A
    /// crease split changes normals (and can reindex vertices), so this is the
    /// direct check that the CPU display path kept the evaluated values.
    fn assert_corner_normals_match_evaluated(display: &DisplayMesh, evaluated: &NodeMesh) {
        assert_eq!(display.mesh.triangles.len(), evaluated.faces.len());
        assert_eq!(display.mesh.tri_face.len(), evaluated.face_groups.len());
        for (tri_face, group) in display.mesh.tri_face.iter().zip(&evaluated.face_groups) {
            assert_eq!(
                display.face_ids.get(*tri_face as usize).map(String::as_str),
                Some(group.as_str()),
                "kept exact normals must not change the triangle's face id"
            );
        }
        for (triangle, evaluated_face) in display.mesh.triangles.iter().zip(&evaluated.faces) {
            for corner in 0..3 {
                let actual_index = triangle[corner] as usize;
                let expected_index = evaluated_face[corner] as usize;
                let actual_position = display.mesh.positions[actual_index];
                let expected_position = evaluated.vertices[expected_index];
                for axis in 0..3 {
                    let expected = f64::from(expected_position[axis]);
                    let actual = [actual_position.x, actual_position.y, actual_position.z][axis];
                    assert!(
                        (actual - expected).abs() < 1.0e-5,
                        "triangle corner position changed: {actual} != {expected}"
                    );
                }
                let actual_normal = display.mesh.normals[actual_index];
                let expected_normal = evaluated.normals[expected_index];
                for axis in 0..3 {
                    let expected = f64::from(expected_normal[axis]);
                    let actual = [actual_normal.x, actual_normal.y, actual_normal.z][axis];
                    assert!(
                        (actual - expected).abs() < 1.0e-6,
                        "triangle corner normal changed: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn cone_apex_keeps_the_evaluated_per_triangle_normals() {
        use nodecraft_solidcore::solidcore::{math::Point3, primitive::api_primitives as api};
        const BASE: f64 = 10.0;
        const HEIGHT: f64 = 20.0;
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = api::create_cone(
            &mut kernel,
            Point3::new(0.0, 0.0, -HEIGHT / 2.0),
            BASE,
            0.0,
            HEIGHT,
        )
        .expect("cone");
        let (display, evaluated) = exact_display(brep);
        assert_corner_normals_match_evaluated(&display, &evaluated);

        // Every apex copy must keep an evaluated side normal: with a radial
        // component (not collapsed onto the axis) and perpendicular to the
        // cone's generator at that radial direction. A crease split averages
        // these copies toward the axis, which produces the bright band.
        let apex_z = HEIGHT / 2.0;
        let mut apex_copies = 0;
        for triangle in &display.mesh.triangles {
            for corner in 0..3 {
                let apex = display.mesh.positions[triangle[corner] as usize];
                let radius = (apex.x * apex.x + apex.y * apex.y).sqrt();
                if (apex.z - apex_z).abs() > 1.0e-3 || radius > 1.0e-3 {
                    continue;
                }
                let normal = display.mesh.normals[triangle[corner] as usize];
                let radial = (normal.x * normal.x + normal.y * normal.y).sqrt();
                assert!(
                    radial > 0.5 && normal.z < 0.7,
                    "apex normal {normal:?} collapsed toward the cone axis"
                );
                // The cone side runs from the apex to `BASE` at the opposite
                // end, at the normal's own azimuth. The exact limit normal is
                // perpendicular to it.
                let side = [
                    BASE * normal.x / radial,
                    BASE * normal.y / radial,
                    -HEIGHT,
                ];
                let length = (side[0] * side[0] + side[1] * side[1] + side[2] * side[2]).sqrt();
                let along_side =
                    (normal.x * side[0] + normal.y * side[1] + normal.z * side[2]) / length;
                assert!(
                    along_side.abs() < 1.0e-2,
                    "apex normal {normal:?} is tilted toward the axis (dot side {along_side})"
                );
                assert!(
                    normal.x * side[0] + normal.y * side[1] > 0.1,
                    "apex normal {normal:?} does not point outward"
                );
                apex_copies += 1;
            }
        }
        assert!(
            apex_copies >= 8,
            "expected one evaluated normal per apex triangle copy, found {apex_copies}"
        );
    }

    #[test]
    fn sphere_pole_keeps_the_evaluated_normals() {
        use nodecraft_solidcore::solidcore::{math::Point3, primitive::api_primitives as api};
        const RADIUS: f64 = 10.0;
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), RADIUS)
            .expect("sphere");
        let (display, evaluated) = exact_display(brep);
        assert_corner_normals_match_evaluated(&display, &evaluated);

        let mut pole_copies = 0;
        let pole_z = RADIUS;
        for triangle in &display.mesh.triangles {
            for corner in 0..3 {
                let point = display.mesh.positions[triangle[corner] as usize];
                if (point.z - pole_z).abs() > 1.0e-3 {
                    continue;
                }
                let normal = display.mesh.normals[triangle[corner] as usize];
                assert!(
                    normal.z > 0.999,
                    "sphere pole normal {normal:?} is not the evaluated outward normal"
                );
                pole_copies += 1;
            }
        }
        assert!(pole_copies > 0, "sphere must have pole triangles");
    }

    #[test]
    fn mesh_without_exact_normals_still_gets_split_normals() {
        // Two faces meeting at a right angle. The source normals are not the
        // face normals; without the exact flag the display path must rebuild
        // them from triangle geometry as it always has.
        let source = NodeMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            faces: vec![[0, 1, 2], [0, 1, 3]],
            face_groups: vec!["face:0".into(), "face:1".into()],
            ..NodeMesh::default()
        };
        assert!(!source.exact_normals);
        let display = display_mesh(&source);
        let normal_of = |triangle: usize| {
            display.mesh.normals[display.mesh.triangles[triangle][0] as usize]
        };
        assert!(normal_of(0).z > 0.999, "flat face keeps its +Z normal");
        assert!(
            normal_of(1).y < -0.999,
            "the crease still splits the second face onto its own normal"
        );
    }
}
