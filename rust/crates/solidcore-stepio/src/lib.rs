//! STEP import façade for the solidcore stack.
//!
//! Owns STEP dispatch, solidcore Brep construction/tessellation and the
//! display-mesh fallbacks. The NodeCraft worker consumes this crate's neutral
//! result and only handles app-specific nodes, caching and export.

use nodecraft_protocol::{BrepEdge, Mesh};

mod mixed_multi;
mod representation;
mod step_content;
mod step_sheet;
mod surface_model;

/// Fallback display tessellation density for B-spline STEP faces. 48×12 keeps
/// curved panels smooth while cutting triangle count and payload versus 64×16.
const FALLBACK_U_SEGMENTS: usize = 48;
const FALLBACK_V_SEGMENTS: usize = 12;
const FALLBACK_STRIDE: usize = FALLBACK_V_SEGMENTS + 1;

pub use representation::{RepresentationSummary, import_representation, representation_summary};

pub fn merge_meshes(mut meshes: Vec<Mesh>) -> Mesh {
    let mut merged = Mesh::default();
    for (index, mesh) in meshes.drain(..).enumerate() {
        let offset = merged.vertices.len() as u32;
        let prefix = format!("body:{index}:");
        merged.vertices.extend(mesh.vertices);
        merged.normals.extend(mesh.normals);
        merged.faces.extend(mesh.faces.into_iter().map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]));
        merged.face_groups.extend(mesh.face_groups.into_iter().map(|group| format!("{prefix}{group}")));
        merged.brep_edges.extend(mesh.brep_edges.into_iter().map(|mut edge| {
            edge.id = format!("{prefix}{}", edge.id);
            for segment in &mut edge.segments { segment[0] += offset; segment[1] += offset; }
            edge
        }));
        merged.lightweight = true;
    }
    merged
}

fn planar_body_mesh(body: &nodecraft_step::PlanarBody) -> Result<Mesh, String> {
    let mut mesh = Mesh::default();
    for face_index in 0..body.faces.len() {
        let triangles = body.face_triangles(face_index)?;
        let a = body.points[body.faces[face_index][0]];
        let b = body.points[body.faces[face_index][1]];
        let c = body.points[body.faces[face_index][2]];
        let u = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
        let v = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
        let len = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt().max(1e-12);
        let normal = [(n[0]/len) as f32, (n[1]/len) as f32, (n[2]/len) as f32];
        let loops: Vec<&Vec<usize>> = std::iter::once(&body.faces[face_index]).chain(body.face_holes(face_index).iter()).collect();
        let mut local = std::collections::BTreeMap::new();
        for loop_points in &loops {
            for index in loop_points.iter().copied() {
                local.entry(index).or_insert_with(|| {
                    let p = body.points[index];
                    mesh.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                    mesh.normals.push(normal);
                    mesh.vertices.len() as u32 - 1
                });
            }
        }
        for triangle in &triangles {
            if let (Some(a), Some(b), Some(c)) = (local.get(&triangle[0]), local.get(&triangle[1]), local.get(&triangle[2])) {
                mesh.faces.push([*a, *b, *c]);
                mesh.face_groups.push(format!("face:{face_index}"));
            }
        }
        for loop_points in &loops {
            for index in 0..loop_points.len() {
                let current = local[&loop_points[index]];
                let next = local[&loop_points[(index + 1) % loop_points.len()]];
                mesh.brep_edges.push(nodecraft_protocol::BrepEdge {
                    id: format!("mesh-edge:{}-{}", face_index, mesh.brep_edges.len()),
                    kind: "mesh".into(),
                    segments: vec![[current, next]],
                });
            }
        }
    }
    mesh.lightweight = true;
    Ok(mesh)
}

/// Diagnostics for one `stepimporter` evaluation.
///
/// They are serialized only on demand: a `stepimporter` node with
/// `values.diagnostics = true` (or `NODECRAFT_WORKER_DIAGNOSTICS=1`) adds an
/// additive top-level `diagnostics` object to the response JSON, keyed by node id.
/// Existing protocol clients ignore the extra field and the typed `Response` is
/// unchanged, so the wire protocol stays backwards compatible.
#[derive(Debug, Clone, Default)]
pub struct StepDiagnostics {
    /// "planar-brep", "planar-brep-fallback", "analytic-brep" or "analytic-mesh-fallback".
    pub kind: String,
    /// Planar bodies read from the STEP file.
    pub bodies: usize,
    /// Exact solidcore B-reps built and used as the tessellation source.
    pub breps: usize,
    pub brep_faces: usize,
    pub brep_edges: usize,
    pub brep_vertices: usize,
    pub mesh_vertices: usize,
    pub mesh_triangles: usize,
    /// Steps that fell back to a less exact path (still imported, but not exact).
    pub fallbacks: Vec<String>,
    /// Non-fatal issues encountered while building/tessellating the Brep.
    pub warnings: Vec<String>,
    /// FILE_NAME product name from the STEP header.
    pub name: Option<String>,
    /// First RGB presentation color found in the file.
    pub color: Option<[f64; 3]>,
    /// Labels of the solid B-reps in the file.
    pub body_names: Vec<String>,
    /// Declared length unit (debug name).
    pub unit: Option<String>,
    /// Declared uncertainty/tolerance in millimetres.
    pub tolerance: Option<f64>,
    /// Number of units in GLOBAL_UNIT_ASSIGNED_CONTEXT.
    pub unit_count: Option<usize>,
    /// Geometry-type classification for import dispatch/diagnostics.
    pub type_report: Option<nodecraft_step::StepTypeReport>,
    /// Number of odd-use (open-boundary) edges in the STEP topology.
    pub open_edges: usize,
    /// Per-solid geometry classification.
    pub solid_types: Vec<nodecraft_step::SolidTypeReport>,
}

impl StepDiagnostics {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind,
            "bodies": self.bodies,
            "breps": self.breps,
            "brepFaces": self.brep_faces,
            "brepEdges": self.brep_edges,
            "brepVertices": self.brep_vertices,
            "meshVertices": self.mesh_vertices,
            "meshTriangles": self.mesh_triangles,
            "bodyNames": self.body_names,
            "unit": self.unit,
            "tolerance": self.tolerance,
            "unitCount": self.unit_count,
            "openEdges": self.open_edges,
            "solids": self.solid_types.iter().map(|solid| serde_json::json!({
                "id": solid.solid, "faces": solid.faces, "planes": solid.planes,
                "cylinders": solid.cylinders, "cones": solid.cones, "spheres": solid.spheres,
                "tori": solid.tori, "bsplines": solid.bsplines, "mixed": solid.mixed(),
            })).collect::<Vec<_>>(),
            "faceTypes": self.type_report.as_ref().map(|report| serde_json::json!({
                "planes": report.planes, "cylinders": report.cylinders, "cones": report.cones,
                "spheres": report.spheres, "tori": report.tori, "bsplines": report.bspline_surfaces,
                "faceted": report.faceted_breps, "mixed": report.mixed_surfaces(),
            })),
            "fallbacks": self.fallbacks,
            "warnings": self.warnings,
            "name": self.name,
            "color": self.color,
        })
    }
}

/// Result of one `stepimporter` import.
///
/// The exact solidcore `Brep` is built first and is the source of the display mesh
/// (rather than the mesh being produced directly from the STEP text). Its topology is
/// retained in `diagnostics`; caching the `Brep` value on `Eval` for later nodes
/// would require naming `solidcore::Brep` in this crate, which needs a direct
/// `solidcore` dependency (outside the files this change owns).
#[derive(Debug, Clone)]
pub struct StepImport {
    pub mesh: Mesh,
    pub diagnostics: StepDiagnostics,
}

/// Mesh-only fallback used when the exact Brep cannot be tessellated.
fn planar_mesh_fallback(
    body: &nodecraft_step::PlanarBody,
    diagnostics: &mut StepDiagnostics,
) -> Result<Mesh, String> {
    match nodecraft_solidcore::planar_body_display_mesh(body) {
        Ok(mesh) if !mesh.faces.is_empty() => {
            diagnostics.fallbacks.push("planar_body_display_mesh".into());
            Ok(mesh)
        }
        Ok(_) => {
            diagnostics.fallbacks.push("planar_body_mesh".into());
            planar_body_mesh(body)
        }
        Err(error) => {
            diagnostics.warnings.push(format!("planar_body_display_mesh failed: {error}"));
            diagnostics.fallbacks.push("planar_body_mesh".into());
            planar_body_mesh(body)
        }
    }
}

/// Builds the exact planar B-rep and tessellates the display mesh from that Brep.
///
/// This is the internal Brep path for planar STEP: `brep_from_planar_body_exact`
/// (falling back to the triangulated Brep builder), then solidcore tessellation. The
/// mesh-only helpers are used only when tessellation fails.
fn planar_body_brep_mesh(
    body: &nodecraft_step::PlanarBody,
    diagnostics: &mut StepDiagnostics,
) -> Result<Mesh, String> {
    let brep = match nodecraft_solidcore::brep_from_planar_body_exact(body) {
        Ok(brep) => brep,
        Err(error) => {
            diagnostics.fallbacks.push(format!("exact planar Brep unavailable: {error}"));
            nodecraft_solidcore::brep_from_planar_body(body)?
        }
    };
    diagnostics.breps += 1;
    diagnostics.brep_faces += brep.faces().len();
    diagnostics.brep_edges += brep.edges().len();
    diagnostics.brep_vertices += brep.vertices().len();
    match nodecraft_solidcore::curved::brep_display_mesh(&Default::default(), &brep, &Default::default()) {
        Ok(mesh) if !mesh.faces.is_empty() => Ok(mesh),
        Ok(mesh) => {
            diagnostics.warnings.push(format!("Brep tessellation produced {} triangles", mesh.faces.len()));
            planar_mesh_fallback(body, diagnostics)
        }
        Err(error) => {
            diagnostics.warnings.push(format!("Brep tessellation failed: {error}"));
            planar_mesh_fallback(body, diagnostics)
        }
    }
}

/// Analytic primitive STEP import: exact B-rep first, legacy curved mesh as fallback.
fn import_analytic_step(
    path: &str,
    solid: nodecraft_step::AnalyticSolid,
) -> Result<StepImport, String> {
    let mut diagnostics = StepDiagnostics { kind: "analytic-brep".into(), ..StepDiagnostics::default() };
    match nodecraft_solidcore::brep_from_analytic_step(path) {
        Ok(brep) => {
            diagnostics.breps = 1;
            diagnostics.brep_faces = brep.faces().len();
            diagnostics.brep_edges = brep.edges().len();
            diagnostics.brep_vertices = brep.vertices().len();
            match nodecraft_solidcore::curved::brep_display_mesh(&Default::default(), &brep, &Default::default()) {
                Ok(mesh) if !mesh.faces.is_empty() => {
                    diagnostics.mesh_vertices = mesh.vertices.len();
                    diagnostics.mesh_triangles = mesh.faces.len();
                    return Ok(StepImport { mesh, diagnostics });
                }
                Ok(mesh) => diagnostics.warnings.push(format!(
                    "analytic Brep tessellation produced {} triangles",
                    mesh.faces.len()
                )),
                Err(error) => diagnostics.warnings.push(format!("analytic Brep tessellation failed: {error}")),
            }
        }
        Err(error) => diagnostics.warnings.push(format!("analytic Brep build failed: {error}")),
    }
    diagnostics.kind = "analytic-mesh-fallback".into();
    diagnostics.fallbacks.push("legacy curved primitive mesh".into());
    let mut mesh = match solid {
        nodecraft_step::AnalyticSolid::Cylinder { radius, height, .. } => nodecraft_solidcore::curved::cylinder_mesh(radius, height),
        nodecraft_step::AnalyticSolid::Cone { radius_base, radius_top, height, .. } => nodecraft_solidcore::curved::cone_mesh(radius_base, radius_top, height),
        nodecraft_step::AnalyticSolid::Sphere { radius, .. } => nodecraft_solidcore::curved::sphere_mesh(radius),
        nodecraft_step::AnalyticSolid::Torus { major, minor, .. } => nodecraft_solidcore::curved::torus_mesh(major, minor),
    }?;
    let center = match solid {
        nodecraft_step::AnalyticSolid::Cylinder { center, .. } | nodecraft_step::AnalyticSolid::Cone { center, .. } | nodecraft_step::AnalyticSolid::Sphere { center, .. } | nodecraft_step::AnalyticSolid::Torus { center, .. } => center,
    };
    for vertex in &mut mesh.vertices {
        vertex[0] += center[0] as f32;
        vertex[1] += center[1] as f32;
        vertex[2] += center[2] as f32;
    }
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}


/// Exact mixed planar/cylindrical STEP import: the validated solidcore mixed adapter
/// builds the B-rep and the display mesh is tessellated from it.
fn import_mixed_step(path: &str) -> Result<StepImport, String> {
    let brep = nodecraft_solidcore::brep_from_mixed_step(path)?;
    let mut diagnostics = StepDiagnostics { kind: "mixed-brep".into(), bodies: 1, breps: 1, ..StepDiagnostics::default() };
    diagnostics.brep_faces = brep.faces().len();
    diagnostics.brep_edges = brep.edges().len();
    diagnostics.brep_vertices = brep.vertices().len();
    let mesh = nodecraft_solidcore::curved::brep_display_mesh(&Default::default(), &brep, &Default::default())?;
    if mesh.faces.is_empty() {
        return Err("mixed STEP Brep tessellation produced no triangles".into());
    }
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}

fn analytic_mesh(solid: &nodecraft_step::AnalyticSolid) -> Result<Mesh, String> {
    let mut mesh = match solid {
        nodecraft_step::AnalyticSolid::Cylinder { radius, height, .. } => nodecraft_solidcore::curved::cylinder_mesh(*radius, *height),
        nodecraft_step::AnalyticSolid::Cone { radius_base, radius_top, height, .. } => nodecraft_solidcore::curved::cone_mesh(*radius_base, *radius_top, *height),
        nodecraft_step::AnalyticSolid::Sphere { radius, .. } => nodecraft_solidcore::curved::sphere_mesh(*radius),
        nodecraft_step::AnalyticSolid::Torus { major, minor, .. } => nodecraft_solidcore::curved::torus_mesh(*major, *minor),
    }?;
    let center = match solid {
        nodecraft_step::AnalyticSolid::Cylinder { center, .. } | nodecraft_step::AnalyticSolid::Cone { center, .. } | nodecraft_step::AnalyticSolid::Sphere { center, .. } | nodecraft_step::AnalyticSolid::Torus { center, .. } => center,
    };
    for vertex in &mut mesh.vertices {
        vertex[0] += center[0] as f32; vertex[1] += center[1] as f32; vertex[2] += center[2] as f32;
    }
    Ok(mesh)
}

fn import_multi_solids(path: &str) -> Result<StepImport, String> {
    import_multi_bodies(&nodecraft_step::import_solids(path)?)
}

fn import_multi_bodies(bodies: &[nodecraft_step::StepBody]) -> Result<StepImport, String> {
    let mut diagnostics = StepDiagnostics { kind: "multi-solid".into(), bodies: bodies.len(), ..StepDiagnostics::default() };
    let mut meshes = Vec::new();
    for body in bodies.iter() {
        let mesh = match body {
            nodecraft_step::StepBody::Planar(body) => planar_body_brep_mesh(body, &mut diagnostics)?,
            nodecraft_step::StepBody::Analytic(solid) => analytic_mesh(solid)?,
        };
        meshes.push(mesh);
    }
    let mesh = merge_meshes(meshes);
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}




fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f32; 3] {
    let u = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let v = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]];
    let len = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt().max(1e-12);
    [(n[0]/len) as f32, (n[1]/len) as f32, (n[2]/len) as f32]
}


/// Smooth-shaded mesh accumulator. Vertices are shared within each STEP face
/// (so curved faces shade smoothly) but never across faces (so hard edges stay).
#[derive(Default)]
struct MeshAccumulator {
    lookup: std::collections::BTreeMap<(u32, [i64; 3]), u32>,
    group_ids: std::collections::BTreeMap<String, u32>,
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
    groups: Vec<String>,
}

impl MeshAccumulator {
    fn triangle(&mut self, a: [f64; 3], b: [f64; 3], c: [f64; 3], group: &str) {
        let group_key = if let Some(id) = self.group_ids.get(group) { *id } else {
            let id = self.group_ids.len() as u32;
            self.group_ids.insert(group.to_string(), id);
            id
        };
        let normal = triangle_normal(a, b, c);
        let normal = [normal[0] as f64, normal[1] as f64, normal[2] as f64];
        let mut indices = [0u32; 3];
        for (slot, point) in indices.iter_mut().zip([a, b, c]) {
            let key = [(point[0] * 1e7).round() as i64, (point[1] * 1e7).round() as i64, (point[2] * 1e7).round() as i64];
            let index = *self.lookup.entry((group_key, key)).or_insert_with(|| {
                self.vertices.push([point[0] as f32, point[1] as f32, point[2] as f32]);
                self.normals.push([0.0; 3]);
                self.vertices.len() as u32 - 1
            });
            let accumulated = &mut self.normals[index as usize];
            accumulated[0] += normal[0]; accumulated[1] += normal[1]; accumulated[2] += normal[2];
            *slot = index;
        }
        self.faces.push(indices);
        self.groups.push(group.into());
    }

    fn finish(mut self) -> Mesh {
        for normal in &mut self.normals {
            let len = (normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]).sqrt().max(1e-12);
            *normal = [normal[0]/len, normal[1]/len, normal[2]/len];
        }
        let mut mesh = Mesh::default();
        mesh.vertices = self.vertices;
        mesh.normals = self.normals.into_iter().map(|n| [n[0] as f32, n[1] as f32, n[2] as f32]).collect();
        mesh.faces = self.faces;
        // Lightweight fallback geometry has no Brep faces to select; dropping the
        // per-triangle group strings saves several MB of JSON.
        mesh.face_groups = Vec::new();
        mesh.lightweight = true;
        mesh
    }
}





/// Append display polylines for the topological edges of the fallback solids.
///
/// B-spline edge curves are sampled between the parameters nearest the edge
/// vertices; other curve types use their endpoint chord. The points are appended
/// to the mesh so `BrepEdge::segments` can index them.
fn append_topology_edges(mesh: &mut Mesh, file: &nodecraft_step::StepFile, brep: &nodecraft_step::StepBrep, solids: &[u32], scale: f64) {
    let mut seen = std::collections::BTreeSet::new();
    let mut edge_index = 0usize;
    for solid_id in solids {
        let Some(solid) = brep.solids.get(solid_id) else { continue };
        let Some(shell) = brep.shells.get(&solid.shell) else { continue };
        for face_id in &shell.faces {
            let Some(face) = brep.faces.get(face_id) else { continue };
            for (loop_id, _) in &face.bounds {
                let Some(loop_) = brep.loops.get(loop_id) else { continue };
                for use_ in &loop_.oriented_edges {
                    if !seen.insert(use_.edge) { continue; }
                    let Some(edge) = brep.edges.get(&use_.edge) else { continue };
                    let Some(start) = brep.vertices.get(&edge.start).map(|vertex| vertex.point) else { continue };
                    let Some(end) = brep.vertices.get(&edge.end).map(|vertex| vertex.point) else { continue };
                    let mut points = Vec::new();
                    if let Some(surface_curve) = file.entities.get(&edge.curve) {
                        let is_bspline = surface_curve.name == "B_SPLINE_CURVE_WITH_KNOTS" || surface_curve.name == "B_SPLINE_CURVE" || (!surface_curve.name.is_empty() && surface_curve.has_member("B_SPLINE_CURVE"));
                        if is_bspline {
                            if let Ok(curve) = step_nurbs::parse_curve(file, edge.curve) {
                                let (low, high) = curve.domain();
                                let count = 48usize;
                                let sample = |t: f64| step_nurbs::curve_point(&curve, t).ok();
                                let candidates: Vec<(f64, [f64; 3])> = (0..=256).filter_map(|index| {
                                    let t = low + (high - low) * index as f64 / 256.0;
                                    sample(t).map(|point| (t, point))
                                }).collect();
                                if let (Some(&(start_t, _)), Some(&(end_t, _))) = (
                                    candidates.iter().min_by(|a, b| distance_squared(a.1, start).total_cmp(&distance_squared(b.1, start))),
                                    candidates.iter().min_by(|a, b| distance_squared(a.1, end).total_cmp(&distance_squared(b.1, end))),
                                ) {
                                    let mut path = Vec::new();
                                    let forward = end_t >= start_t;
                                    let steps = if forward { end_t - start_t } else { (high - start_t) + (end_t - low) };
                                    for step in 0..=count {
                                        let fraction = step as f64 / count as f64;
                                        let t = if forward {
                                            start_t + (end_t - start_t) * fraction
                                        } else {
                                            let offset = steps * fraction;
                                            if start_t + offset <= high { start_t + offset } else { low + (start_t + offset - high) }
                                        };
                                        if let Some(point) = sample(t) { path.push(point); }
                                    }
                                    if path.len() > 1 && distance_squared(path[0], start) < 1e-6 * scale * scale && distance_squared(*path.last().unwrap(), end) < 1e-6 * scale * scale {
                                        points = path;
                                    }
                                }
                            }
                        }
                    }
                    if points.len() < 2 { points = vec![start, end]; }
                    let base = mesh.vertices.len() as u32;
                    for point in &points {
                        mesh.vertices.push([(point[0]*scale) as f32, (point[1]*scale) as f32, (point[2]*scale) as f32]);
                        mesh.normals.push([0.0, 0.0, 1.0]);
                    }
                    let segments = (0..points.len().saturating_sub(1)).map(|index| [base + index as u32, base + index as u32 + 1]).collect();
                    mesh.brep_edges.push(BrepEdge { id: format!("step-edge:{edge_index}"), kind: "step".into(), segments });
                    edge_index += 1;
                }
            }
        }
    }
}

fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)
}

/// Sample one STEP edge for trim-loop reconstruction (unscaled file units).
fn trim_edge_points(file: &nodecraft_step::StepFile, brep: &nodecraft_step::StepBrep, edge: &nodecraft_step::topology::StepEdge) -> Vec<[f64; 3]> {
    let Some(start) = brep.vertices.get(&edge.start).map(|vertex| vertex.point) else { return Vec::new() };
    let Some(end) = brep.vertices.get(&edge.end).map(|vertex| vertex.point) else { return Vec::new() };
    let mut points = Vec::new();
    if let Some(curve_entity) = file.entities.get(&edge.curve) {
        let is_bspline = curve_entity.name == "B_SPLINE_CURVE_WITH_KNOTS" || curve_entity.name == "B_SPLINE_CURVE" || (!curve_entity.name.is_empty() && curve_entity.has_member("B_SPLINE_CURVE"));
        if is_bspline {
            if let Ok(curve) = step_nurbs::parse_curve(file, edge.curve) {
                let (low, high) = curve.domain();
                let sample = |t: f64| step_nurbs::curve_point(&curve, t).ok();
                let candidates: Vec<(f64, [f64; 3])> = (0..=256).filter_map(|index| {
                    let t = low + (high - low) * index as f64 / 256.0;
                    sample(t).map(|point| (t, point))
                }).collect();
                if let (Some(&(start_t, _)), Some(&(end_t, _))) = (
                    candidates.iter().min_by(|a, b| distance_squared(a.1, start).total_cmp(&distance_squared(b.1, start))),
                    candidates.iter().min_by(|a, b| distance_squared(a.1, end).total_cmp(&distance_squared(b.1, end))),
                ) {
                    let mut path = Vec::new();
                    let forward = end_t >= start_t;
                    let steps = if forward { end_t - start_t } else { (high - start_t) + (end_t - low) };
                    for step in 0..=48 {
                        let fraction = step as f64 / 48.0;
                        let t = if forward {
                            start_t + (end_t - start_t) * fraction
                        } else {
                            let offset = steps * fraction;
                            if start_t + offset <= high { start_t + offset } else { low + (start_t + offset - high) }
                        };
                        if let Some(point) = sample(t) { path.push(point); }
                    }
                    if path.len() > 1 && distance_squared(path[0], start) < 1e-6 && distance_squared(*path.last().unwrap(), end) < 1e-6 {
                        points = path;
                    }
                }
            }
        }
    }
    if points.len() < 2 { points = vec![start, end]; }
    points
}

/// Map each FACE_BOUND loop onto the surface's sampled UV grid. Returns `None`
/// when a boundary sample cannot be confidently mapped, so the caller keeps the
/// untrimmed fallback instead of clipping with a guessed polygon.
fn refine_surface_uv(surface: &step_nurbs::BsplineSurface, point: [f64; 3], u0: f64, u1: f64, v0: f64, v1: f64, start: [f64; 2]) -> Option<[f64; 2]> {
    let mut best = start;
    let mut best_distance = distance_squared(surface.point_at(start[0], start[1]).ok()?, point);
    let mut step_u = (u1 - u0) / 64.0;
    let mut step_v = (v1 - v0) / 16.0;
    for _ in 0..12 {
        let mut improved = false;
        for du in [-1.0, 0.0, 1.0] {
            for dv in [-1.0, 0.0, 1.0] {
                if du == 0.0 && dv == 0.0 { continue; }
                let candidate = [
                    (start[0] + du * step_u).clamp(u0, u1),
                    (start[1] + dv * step_v).clamp(v0, v1),
                ];
                let Ok(sample) = surface.point_at(candidate[0], candidate[1]) else { continue };
                let distance = distance_squared(sample, point);
                if distance < best_distance {
                    best_distance = distance;
                    best = candidate;
                    improved = true;
                }
            }
        }
        if !improved { step_u *= 0.5; step_v *= 0.5; }
        if step_u < 1e-12 && step_v < 1e-12 { break; }
    }
    Some(best)
}

fn face_trim_uv_loops(
    file: &nodecraft_step::StepFile,
    brep: &nodecraft_step::StepBrep,
    face: &nodecraft_step::topology::StepFace,
    surface: &step_nurbs::BsplineSurface,
    sampled: &step_nurbs::SurfaceMesh,
    u_segments: usize,
    v_segments: usize,
) -> Option<Vec<Vec<[f64; 2]>>> {
    let (u0, u1) = surface.u_domain();
    let (v0, v1) = surface.v_domain();
    let debug = std::env::var_os("NODECRAFT_TRIM_DEBUG").is_some();
    let fail = |reason: &str| -> Option<Vec<Vec<[f64; 2]>>> {
        if debug { eprintln!("trim face #{}: {reason} (u_closed={} v_closed={})", face.id, surface.u_closed, surface.v_closed); }
        None
    };
    let stride = v_segments + 1;
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in &sampled.points { for axis in 0..3 { min[axis] = min[axis].min(point[axis]); max[axis] = max[axis].max(point[axis]); } }
    let extent = (max[0]-min[0]).abs().max((max[1]-min[1]).abs()).max((max[2]-min[2]).abs()).max(1e-9);
    let tolerance = extent * 2e-2;
    // Grid samples are only seeds: allow a wider start radius, then keep the
    // strict tolerance for the refined surface point.
    let start_tolerance = extent * 6e-2;
    let mut loops = Vec::new();
    for (loop_id, _) in &face.bounds {
        let Some(loop_) = brep.loops.get(loop_id) else { return fail("missing loop"); };
        let mut points = Vec::new();
        for use_ in &loop_.oriented_edges {
            let Some(edge) = brep.edges.get(&use_.edge) else { return fail("missing edge"); };
            let mut segment = trim_edge_points(file, brep, edge);
            if !use_.orientation { segment.reverse(); }
            if points.last() == segment.first() { segment.remove(0); }
            points.extend(segment);
        }
        if points.len() < 3 { return fail("short boundary"); }
        let mut uv = Vec::with_capacity(points.len());
        for point in points {
            // Keep several nearest grid samples: a single start can sit on the
            // wrong side of a periodic seam or a trimmed corner. Refining each
            // candidate toward the actual boundary point is robust enough to
            // recover those points without a full-domain solve.
            let mut nearest: Vec<(f64, usize)> = Vec::with_capacity(4);
            for (index, candidate) in sampled.points.iter().enumerate() {
                let distance = distance_squared(*candidate, point);
                if nearest.len() < 4 {
                    nearest.push((distance, index));
                    if nearest.len() == 4 {
                        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
                    }
                } else if distance < nearest.last().map_or(f64::INFINITY, |entry| entry.0) {
                    nearest.pop();
                    nearest.push((distance, index));
                    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
                }
            }
            if nearest.first().is_none_or(|entry| entry.0.sqrt() > start_tolerance) { return fail("no grid sample near boundary"); }
            let mut best: Option<([f64; 2], f64)> = None;
            for (_, index) in &nearest {
                let i = index / stride;
                let j = index % stride;
                let start = [u0 + (u1 - u0) * i as f64 / u_segments as f64, v0 + (v1 - v0) * j as f64 / v_segments as f64];
                let Some(refined) = refine_surface_uv(surface, point, u0, u1, v0, v1, start) else { continue };
                let Ok(mapped) = surface.point_at(refined[0], refined[1]) else { continue };
                let distance = distance_squared(mapped, point).sqrt();
                if best.is_none_or(|entry| distance < entry.1) {
                    best = Some((refined, distance));
                }
                if distance <= tolerance { break; }
            }
            let Some((refined, distance)) = best else { return fail("no refined UV"); };
            if distance > tolerance { return fail("refined UV outside tolerance"); }
            let [u, v] = refined;
            if uv.last().is_none_or(|last: &[f64; 2]| (last[0]-u).abs() > 1e-10 || (last[1]-v).abs() > 1e-10) {
                uv.push([u, v]);
            }
        }
        if uv.len() >= 3 { loops.push(uv); }
    }
    if loops.is_empty() { fail("no loops") } else { Some(loops) }
}

fn point_in_uv_loops(point: [f64; 2], loops: &[Vec<[f64; 2]>]) -> bool {
    let mut inside = false;
    for loop_ in loops {
        let mut previous = loop_.len() - 1;
        for current in 0..loop_.len() {
            let [xi, yi] = loop_[current];
            let [xj, yj] = loop_[previous];
            if (yi > point[1]) != (yj > point[1]) && point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi {
                inside = !inside;
            }
            previous = current;
        }
    }
    inside
}

fn plane_heights(file: &nodecraft_step::StepFile, brep: &nodecraft_step::StepBrep, shell: &nodecraft_step::topology::StepShell) -> Vec<f64> {
    let mut out = Vec::new();
    for face_id in &shell.faces {
        let Some(face) = brep.faces.get(face_id) else { continue };
        let Some(surface) = file.entities.get(&face.surface) else { continue };
        if surface.name != "PLANE" { continue; }
        let Some(nodecraft_step::Value::Ref(placement_id)) = surface.args.get(1) else { continue };
        let Some(placement) = file.entities.get(placement_id) else { continue };
        let Some(nodecraft_step::Value::Ref(point_id)) = placement.args.get(1) else { continue };
        let Some(point) = file.entities.get(point_id) else { continue };
        let Some(nodecraft_step::Value::List(coords)) = point.args.get(1) else { continue };
        if let Some(nodecraft_step::Value::Number(z)) = coords.get(2) { out.push(*z); }
    }
    out
}

fn import_bspline_fallback(path: &str, skipped: &[u32]) -> Result<StepImport, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = nodecraft_step::StepFile::parse(&text)?;
    let brep = nodecraft_step::read_topology(&file);
    let scale = nodecraft_step::read_length_unit(path).map(|unit| unit.millimetres()).unwrap_or(1.0);
    let mut mesh = MeshAccumulator::default();
    let mut warnings = Vec::new();
    for solid_id in skipped {
        let solid = brep.solids.get(solid_id).ok_or_else(|| format!("missing solid #{solid_id}"))?;
        let shell = brep.shells.get(&solid.shell).ok_or_else(|| format!("solid #{solid_id} has no shell"))?;
        let heights = plane_heights(&file, &brep, shell);
        let z_bounds = if heights.len() >= 2 {
            Some((heights.iter().cloned().fold(f64::INFINITY, f64::min), heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)))
        } else { None };
        for face_id in &shell.faces {
            let face = brep.faces.get(face_id).ok_or_else(|| format!("missing face #{face_id}"))?;
            let surface = file.entities.get(&face.surface).ok_or_else(|| format!("missing surface #{}", face.surface))?;
            if surface.name == "PLANE" {
                let Some((loop_id, _)) = face.bounds.first() else { continue };
                let loop_ = brep.loops.get(loop_id).ok_or_else(|| format!("missing loop #{loop_id}"))?;
                let points: Vec<[f64; 3]> = loop_.oriented_edges.iter().filter_map(|use_| {
                    let edge = brep.edges.get(&use_.edge)?;
                    let vertex = if use_.orientation { edge.start } else { edge.end };
                    brep.vertices.get(&vertex).map(|vertex| vertex.point)
                }).collect();
                for index in 1..points.len().saturating_sub(1) {
                    let scaled = |point: [f64; 3]| [point[0]*scale, point[1]*scale, point[2]*scale];
                    mesh.triangle(scaled(points[0]), scaled(points[index]), scaled(points[index+1]), &format!("solid:{solid_id}:face:{face_id}"));
                }
            } else {
                let is_bspline = surface.has_member("B_SPLINE_SURFACE") || surface.has_member("RATIONAL_B_SPLINE_SURFACE") || surface.has_member("B_SPLINE_SURFACE_WITH_KNOTS");
                if !is_bspline {
                    let kind = if surface.name.is_empty() { surface.complex_members().first().copied().unwrap_or("unknown surface") } else { surface.name.as_str() };
                    return Err(format!("solid #{solid_id} is a mixed planar/analytic body: face #{face_id} uses curved surface {kind}; exact mixed STEP import is not implemented yet (Re-01/Tr-33)"));
                }
                let surface = step_nurbs::parse_surface(&file, face.surface)
                    .map_err(|error| format!("solid #{solid_id} face #{face_id}: {error}"))?;
                let sampled = step_nurbs::tessellate_surface(&surface, FALLBACK_U_SEGMENTS, FALLBACK_V_SEGMENTS)?;
                let margin = z_bounds.map(|(min, max)| (max - min).abs() * 1e-6).unwrap_or(1e-6);
                let trim_loops = face_trim_uv_loops(&file, &brep, face, &surface, &sampled, FALLBACK_U_SEGMENTS, FALLBACK_V_SEGMENTS);
                let base_vertices = mesh.vertices.len();
                let base_normals = mesh.normals.len();
                let base_faces = mesh.faces.len();
                let base_groups = mesh.groups.len();
                let stride = FALLBACK_STRIDE;
                let mut kept = 0usize;
                for triangle in &sampled.triangles {
                    let a = sampled.points[triangle[0] as usize]; let b = sampled.points[triangle[1] as usize]; let c = sampled.points[triangle[2] as usize];
                    let inside = z_bounds.is_none_or(|(min, max)| {
                        let z_ok = |p: [f64; 3]| p[2] >= min - margin && p[2] <= max + margin;
                        z_ok(a) && z_ok(b) && z_ok(c)
                    });
                    let trimmed = trim_loops.as_ref().is_none_or(|loops| {
                        // Trim loops are in the surface's parameter domain, which is
                        // not necessarily 0..1 (2π-periodic and STEP-scaled domains
                        // are common). Test triangle centroids in the same space.
                        let (u0, u1) = surface.u_domain();
                        let (v0, v1) = surface.v_domain();
                        let uv = |index: u32| {
                            let i = index as usize / stride;
                            let j = index as usize % stride;
                            [
                                u0 + (u1 - u0) * i as f64 / FALLBACK_U_SEGMENTS as f64,
                                v0 + (v1 - v0) * j as f64 / FALLBACK_V_SEGMENTS as f64,
                            ]
                        };
                        let ua = uv(triangle[0]); let ub = uv(triangle[1]); let uc = uv(triangle[2]);
                        point_in_uv_loops([(ua[0]+ub[0]+uc[0])/3.0, (ua[1]+ub[1]+uc[1])/3.0], loops)
                    });
                    if inside && trimmed { mesh.triangle([a[0]*scale,a[1]*scale,a[2]*scale], [b[0]*scale,b[1]*scale,b[2]*scale], [c[0]*scale,c[1]*scale,c[2]*scale], &format!("solid:{solid_id}:face:{face_id}")); kept += 1; }
                }
                let mut effective_trim = trim_loops;
                if effective_trim.is_some() && kept * 100 < sampled.triangles.len() {
                    if std::env::var_os("NODECRAFT_TRIM_DEBUG").is_some() {
                        if let Some(loops) = &effective_trim {
                            for (index, loop_) in loops.iter().enumerate() {
                                let min_u = loop_.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
                                let max_u = loop_.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
                                let min_v = loop_.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
                                let max_v = loop_.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
                                let area: f64 = loop_.iter().zip(loop_.iter().cycle().skip(1)).map(|(a, b)| a[0]*b[1]-b[0]*a[1]).sum::<f64>().abs() * 0.5;
                                eprintln!("safety face #{} loop {index}: kept {kept}/{} points {} u [{min_u:.4},{max_u:.4}] v [{min_v:.4},{max_v:.4}] area {area:.6}", face.id, sampled.triangles.len(), loop_.len());
                            }
                        }
                    }
                    // A trim that removes nearly the whole face is almost certainly a
                    // bad boundary mapping; keep the untrimmed fallback rather than
                    // punching a hole in the solid.
                    mesh.vertices.truncate(base_vertices);
                    mesh.normals.truncate(base_normals);
                    mesh.faces.truncate(base_faces);
                    mesh.groups.truncate(base_groups);
                    kept = 0;
                    for triangle in &sampled.triangles {
                        let a = sampled.points[triangle[0] as usize]; let b = sampled.points[triangle[1] as usize]; let c = sampled.points[triangle[2] as usize];
                        let inside = z_bounds.is_none_or(|(min, max)| {
                            let z_ok = |p: [f64; 3]| p[2] >= min - margin && p[2] <= max + margin;
                            z_ok(a) && z_ok(b) && z_ok(c)
                        });
                        if inside { mesh.triangle([a[0]*scale,a[1]*scale,a[2]*scale], [b[0]*scale,b[1]*scale,b[2]*scale], [c[0]*scale,c[1]*scale,c[2]*scale], &format!("solid:{solid_id}:face:{face_id}")); kept += 1; }
                    }
                    effective_trim = None;
                }
                warnings.push(if effective_trim.is_some() {
                    format!("solid #{solid_id} face #{face_id}: B-spline surface trimmed by mapped UV boundary loops; {kept} triangles kept")
                } else {
                    format!("solid #{solid_id} face #{face_id}: B-spline surface has no mappable UV trim loops; untrimmed full UV grid kept ({kept} triangles)")
                });
            }
        }
    }
    if mesh.faces.is_empty() { return Err("B-spline fallback produced no triangles".into()); }
    let mut mesh = mesh.finish();
    append_topology_edges(&mut mesh, &file, &brep, skipped, scale);
    let mut diagnostics = StepDiagnostics { kind: "bspline-solid-mesh".into(), bodies: skipped.len(), warnings, ..StepDiagnostics::default() };
    diagnostics.brep_edges = mesh.brep_edges.len();
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}


fn import_faceted_brep(path: &str) -> Result<StepImport, String> {
    let bodies = nodecraft_step::read_faceted_breps(path)?;
    if bodies.is_empty() { return Err("STEP file has no FACETED_BREP".into()); }
    let mut diagnostics = StepDiagnostics { kind: "faceted-brep".into(), bodies: bodies.len(), ..StepDiagnostics::default() };
    let mut meshes = Vec::new();
    for body in bodies.iter() {
        let mesh = planar_body_brep_mesh(body, &mut diagnostics)?;
        meshes.push(mesh);
    }
    let mesh = merge_meshes(meshes);
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}

/// Which STEP reader implementation the façade uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepReaderBackend {
    /// NodeCraft's existing parser/import pipeline.
    Legacy,
    /// Official `step-import` neutral reader. Requires a public STEP-to-BRep builder.
    Solidcore,
    /// Run the legacy reader and inspect the official neutral model for comparison.
    Compare,
}

/// The backend selected by `NODECRAFT_STEP_READER` (`legacy`, `solidcore`, `compare`).
pub fn selected_step_reader_backend() -> StepReaderBackend {
    match std::env::var("NODECRAFT_STEP_READER").ok().as_deref() {
        Some("legacy") => StepReaderBackend::Legacy,
        Some("compare") => StepReaderBackend::Compare,
        _ => StepReaderBackend::Solidcore,
    }
}

/// Imports a STEP file through the configured reader backend.
pub fn import_step(path: &str) -> Result<StepImport, String> {
    import_step_with_backend(path, selected_step_reader_backend())
}

/// Imports a STEP file through an explicit reader backend.
pub fn import_step_with_backend(path: &str, backend: StepReaderBackend) -> Result<StepImport, String> {
    match backend {
        StepReaderBackend::Legacy => import_step_legacy(path),
        StepReaderBackend::Solidcore => import_step_solidcore(path),
        StepReaderBackend::Compare => import_step_compare(path),
    }
}

fn import_step_legacy(path: &str) -> Result<StepImport, String> {
    let mut result = import_step_inner(path).map_err(|error| if error.starts_with(path) { error } else { format!("{path}: {error}") })?;
    if let Ok(header) = nodecraft_step::read_header(path) { result.diagnostics.name = header.name; }
    if let Ok(names) = nodecraft_step::read_solid_names(path) { result.diagnostics.body_names = names; }
    if let Ok(Some(context)) = nodecraft_step::read_unit_context(path) {
        result.diagnostics.unit = Some(format!("{:?}", context.length_unit));
        result.diagnostics.unit_count = Some(context.unit_count);
    }
    result.diagnostics.tolerance = nodecraft_step::read_tolerance(path).ok().flatten();
    result.diagnostics.type_report = nodecraft_step::classify_step(path).ok();
    result.diagnostics.solid_types = nodecraft_step::classify_solids(path).unwrap_or_default();
    if let Ok(brep) = nodecraft_step::read_topology_file(path) {
        result.diagnostics.open_edges = brep.open_edges().len();
        if result.diagnostics.open_edges > 0 {
            result.diagnostics.warnings.push(format!("STEP topology has {} open edge(s)", result.diagnostics.open_edges));
        }
    }
    if let Ok(Some((color, _))) = nodecraft_step::read_color(path) { result.diagnostics.color = Some(color); }
    validate_exact_mesh_bounds(path, &result)?;
    Ok(result)
}

fn tessellate_solidcore_brep(
    kernel: &nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &nodecraft_solidcore::solidcore::topology::Brep,
    senses: &[bool],
) -> Result<Mesh, String> {
    nodecraft_solidcore::curved::brep_display_mesh_with_senses(
        kernel,
        brep,
        &Default::default(),
        Some(senses),
    )
}

fn read_solidcore_model(path: &str) -> Result<step_import::Model, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    step_import::import(&bytes, &step_import::Limits::default())
        .map_err(|error| format!("{path}: official STEP reader: {error}"))
}

fn import_step_solidcore(path: &str) -> Result<StepImport, String> {
    let model = read_solidcore_model(path)?;
    if model.bodies.is_empty() {
        return Err(format!("{path}: official STEP reader found no B-rep bodies"));
    }
    let kernel = nodecraft_solidcore::solidcore::kernel::Kernel::default();
    let mut meshes = Vec::with_capacity(model.bodies.len());
    let mut body_names = Vec::with_capacity(model.bodies.len());
    let mut healed_bodies = 0usize;
    for body in &model.bodies {
        body_names.push(body.name.clone().unwrap_or_default());
        let mut brep = step_brep::build(body)
            .map_err(|error| format!("{path}: solid #{}: {error}", body.step_id))?;
        if !step_brep::has_uv_trims(&brep).unwrap_or(false) {
            // A face without UV trims is meshed over its whole parameter domain, which is
            // the flat-rectangle/cracked-edge failure. Heal before handing it to display.
            brep = step_brep::build_healed(body)
                .map_err(|error| format!("{path}: solid #{} healed rebuild: {error}", body.step_id))?;
            healed_bodies += 1;
        }
        step_brep::validate_for_display(&brep)
            .map_err(|error| format!("{path}: solid #{}: {error}", body.step_id))?;
        let senses: Vec<bool> = body.faces.iter().map(|face| face.same_sense).collect();
        let mesh = tessellate_solidcore_brep(&kernel, &brep, &senses)
            .map_err(|error| format!("{path}: solid #{} tessellation failed: {error}", body.step_id))?;
        if mesh.faces.is_empty() {
            return Err(format!("{path}: solid #{} tessellation produced no triangles", body.step_id));
        }
        meshes.push(mesh);
    }
    let mesh = merge_meshes(meshes);
    let unit = model.bodies.first().map(|body| body.unit_mm).unwrap_or(1.0);
    let mut warnings: Vec<String> = model.diagnostics.iter().map(|diagnostic| diagnostic.message.clone()).collect();
    if healed_bodies > 0 {
        warnings.push(format!("{healed_bodies} body/bodies required healing for display tessellation"));
    }
    let mut diagnostics = StepDiagnostics {
        kind: "solidcore-brep".into(),
        bodies: model.bodies.len(),
        breps: model.bodies.len(),
        body_names,
        unit: Some(format!("{unit} mm/unit")),
        name: Some(model.header.file_name.clone()).filter(|name| !name.is_empty()),
        mesh_vertices: mesh.vertices.len(),
        mesh_triangles: mesh.faces.len(),
        warnings,
        ..StepDiagnostics::default()
    };
    diagnostics.brep_faces = model.bodies.iter().map(|body| body.faces.len()).sum();
    diagnostics.brep_edges = model.bodies.iter().map(|body| body.edges.len()).sum();
    diagnostics.brep_vertices = model.bodies.iter().map(|body| body.vertices.len()).sum();
    Ok(StepImport { mesh, diagnostics })
}

fn import_step_compare(path: &str) -> Result<StepImport, String> {
    let mut result = import_step_legacy(path)?;
    match read_solidcore_model(path) {
        Ok(model) => result.diagnostics.warnings.push(format!(
            "reader compare: official step-import parsed {} body/{} skipped; legacy returned {} body/{} skipped",
            model.bodies.len(),
            model.skipped.len(),
            result.diagnostics.bodies,
            result.diagnostics.kind
        )),
        Err(error) => result.diagnostics.warnings.push(format!("reader compare: {error}")),
    }
    Ok(result)
}

/// Reject an exact import whose tessellated mesh does not cover the independently
/// parsed STEP bounds. Display/sheet/assembly results have their own bounds semantics.
fn validate_exact_mesh_bounds(path: &str, result: &StepImport) -> Result<(), String> {
    let kind = result.diagnostics.kind.as_str();
    // Planar/mixed/faceted B-reps have topology-referenced bounds. Analytic primitives
    // and B-spline mesh readers get their own mass/display checks.
    if !matches!(kind, "planar-brep" | "mixed-brep") {
        return Ok(());
    }
    let Ok((want_min, want_max)) = nodecraft_step::read_topology_bounds(path) else { return Ok(()); };
    if result.mesh.vertices.is_empty() { return Ok(()); }
    let mut got_min = [f64::INFINITY; 3];
    let mut got_max = [f64::NEG_INFINITY; 3];
    for vertex in &result.mesh.vertices {
        for axis in 0..3 {
            got_min[axis] = got_min[axis].min(vertex[axis] as f64);
            got_max[axis] = got_max[axis].max(vertex[axis] as f64);
        }
    }
    let extent = (0..3).map(|axis| (want_max[axis] - want_min[axis]).abs()).fold(0.0f64, f64::max).max(1.0);
    let tolerance = extent * 1e-3;
    for axis in 0..3 {
        if (got_min[axis] - want_min[axis]).abs() > tolerance || (got_max[axis] - want_max[axis]).abs() > tolerance {
            return Err(format!(
                "{path}: exact import mesh bounds [{got_min:?}]..[{got_max:?}] differ from STEP geometry [{want_min:?}]..[{want_max:?}] (axis {axis}, tolerance {tolerance:.6})"
            ));
        }
    }
    Ok(())
}

thread_local! {
    /// Assembly files currently being imported on this thread: cycle guard for external
    /// document references (`A.stp` -> `B.stp` -> `A.stp`).
    static ASSEMBLY_IMPORT_STACK: std::cell::RefCell<Vec<std::path::PathBuf>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// RAII frame for the recursive assembly import cycle guard.
struct AssemblyImportFrame {
    path: std::path::PathBuf,
}

impl AssemblyImportFrame {
    fn enter(path: &str) -> Result<Self, String> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| std::path::PathBuf::from(path));
        let cycle = ASSEMBLY_IMPORT_STACK.with(|stack| stack.borrow().contains(&canonical));
        if cycle {
            return Err(format!("assembly cycle through {path}"));
        }
        ASSEMBLY_IMPORT_STACK.with(|stack| stack.borrow_mut().push(canonical.clone()));
        Ok(Self { path: canonical })
    }
}

impl Drop for AssemblyImportFrame {
    fn drop(&mut self) {
        ASSEMBLY_IMPORT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            if let Some(index) = stack.iter().rposition(|entry| entry == &self.path) {
                stack.remove(index);
            }
        });
    }
}

/// Places a child mesh (already millimetres) into the assembly frame.
///
/// Mirrored/negative-scale occurrences (SMLib's `m_bMirrored`/signed-scale semantics) have a
/// negative determinant; their triangle winding is reversed and the normals are negated so
/// faces stay outward-facing. Normals transform by the inverse-transpose of the linear part
/// (the linear part is used when the transform is singular).
fn transform_mesh(mesh: &mut Mesh, transform: &step_assemblies::Transform) {
    let rows = transform.rows();
    let linear = [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ];
    let determinant = linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
        - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
        + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0]);
    let normal_matrix = if determinant.abs() > 1e-18 {
        let inverse = 1.0 / determinant;
        [
            [
                (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1]) * inverse,
                (linear[0][2] * linear[2][1] - linear[0][1] * linear[2][2]) * inverse,
                (linear[0][1] * linear[1][2] - linear[0][2] * linear[1][1]) * inverse,
            ],
            [
                (linear[1][2] * linear[2][0] - linear[1][0] * linear[2][2]) * inverse,
                (linear[0][0] * linear[2][2] - linear[0][2] * linear[2][0]) * inverse,
                (linear[0][2] * linear[1][0] - linear[0][0] * linear[1][2]) * inverse,
            ],
            [
                (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0]) * inverse,
                (linear[0][1] * linear[2][0] - linear[0][0] * linear[2][1]) * inverse,
                (linear[0][0] * linear[1][1] - linear[0][1] * linear[1][0]) * inverse,
            ],
        ]
    } else {
        linear
    };
    for vertex in &mut mesh.vertices {
        let x = f64::from(vertex[0]);
        let y = f64::from(vertex[1]);
        let z = f64::from(vertex[2]);
        *vertex = [
            (rows[0][0] * x + rows[0][1] * y + rows[0][2] * z + rows[0][3]) as f32,
            (rows[1][0] * x + rows[1][1] * y + rows[1][2] * z + rows[1][3]) as f32,
            (rows[2][0] * x + rows[2][1] * y + rows[2][2] * z + rows[2][3]) as f32,
        ];
    }
    let mirrored = determinant < 0.0;
    for normal in &mut mesh.normals {
        let x = f64::from(normal[0]);
        let y = f64::from(normal[1]);
        let z = f64::from(normal[2]);
        let mut transformed = [
            normal_matrix[0][0] * x + normal_matrix[0][1] * y + normal_matrix[0][2] * z,
            normal_matrix[1][0] * x + normal_matrix[1][1] * y + normal_matrix[1][2] * z,
            normal_matrix[2][0] * x + normal_matrix[2][1] * y + normal_matrix[2][2] * z,
        ];
        if mirrored {
            transformed = [-transformed[0], -transformed[1], -transformed[2]];
        }
        let length = (transformed[0] * transformed[0] + transformed[1] * transformed[1] + transformed[2] * transformed[2]).sqrt();
        if length > 1e-12 {
            transformed = [transformed[0] / length, transformed[1] / length, transformed[2] / length];
        }
        *normal = [transformed[0] as f32, transformed[1] as f32, transformed[2] as f32];
    }
    if mirrored {
        for face in &mut mesh.faces {
            face.swap(1, 2);
        }
    }
    // The compact buffers would otherwise keep the untransformed data.
    mesh.vertices_data = None;
    mesh.normals_data = None;
}

/// Same placement for a transform whose translation is in STEP file units.
fn scaled_world_transform(transform: &step_assemblies::Transform, unit_scale_mm: f64) -> step_assemblies::Transform {
    if (unit_scale_mm - 1.0).abs() <= 1e-12 {
        return *transform;
    }
    let mut rows = transform.rows();
    for row in rows.iter_mut().take(3) {
        row[3] *= unit_scale_mm;
    }
    step_assemblies::Transform::from_rows(rows).unwrap_or(*transform)
}

fn import_step_inner(path: &str) -> Result<StepImport, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    if text.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE") {
        if let Ok(components) = step_assemblies::component_instances(path) {
            if !components.instances.is_empty() {
                let root = components.instances.first().map(|instance| instance.name.as_str()).unwrap_or("assembly");
                let mut diagnostics = StepDiagnostics { kind: "assembly".into(), ..StepDiagnostics::default() };
                let mut meshes: Vec<Mesh> = Vec::new();
                let mut failures: Vec<String> = Vec::new();
                let mut mixed_representations = 0usize;
                let mut mixed_external_documents = 0usize;
                let mut meshed_occurrences = 0usize;
                let total_occurrences = components.instances.len();
                // Cycle guard: an external child may itself be an assembly.
                let _frame = AssemblyImportFrame::enter(path).map_err(|error| format!("STEP assembly '{root}': {error}"))?;
                let context = representation::RepresentationContext::load(path).ok();
                for instance in &components.instances {
                    match &instance.source {
                        step_assemblies::ComponentSource::ExternalDocument { document, path: document_path, exists } => {
                            if !exists {
                                continue; // reported through components.missing_documents
                            }
                            let child_path = document_path.to_string_lossy().into_owned();
                            match import_step_inner(&child_path) {
                                Ok(mut child) if !child.mesh.faces.is_empty() => {
                                    transform_mesh(&mut child.mesh, &instance.world_transform_mm());
                                    meshes.push(child.mesh);
                                    meshed_occurrences += 1;
                                }
                                Ok(_) => failures.push(format!("external document {} produced no triangles", document.file_name)),
                                Err(error) => {
                                    if error.contains("mixed planar/analytic") || error.contains("Mx-72") {
                                        mixed_external_documents += 1;
                                    }
                                    failures.push(format!("external document {}: {error}", document.file_name));
                                }
                            }
                        }
                        step_assemblies::ComponentSource::InFileRepresentation { representation_id } => {
                            let imported = match &context {
                                Some(context) => context.import(*representation_id),
                                None => Err(format!("representation #{representation_id}: representation context could not be built")),
                            };
                            match imported {
                                Ok(mut import) if !import.mesh.faces.is_empty() => {
                                    transform_mesh(&mut import.mesh, &instance.world_transform_mm());
                                    meshes.push(import.mesh);
                                    meshed_occurrences += 1;
                                }
                                Ok(_) => failures.push(format!("{}: representation #{representation_id} produced no triangles", instance.name)),
                                Err(error) => {
                                    if error.contains("Mx-72") || error.contains("mixed planar/analytic") {
                                        mixed_representations += 1;
                                    }
                                    failures.push(format!("{}: {error}", instance.name));
                                }
                            }
                        }
                        step_assemblies::ComponentSource::NoGeometry => {
                            failures.push(format!("{}: occurrence has no geometry source", instance.name));
                        }
                    }
                }
                // The assembly product (and nested sub-assembly products) can carry their own
                // geometry representation in addition to the placed occurrences.
                if let Some(context) = context.as_ref() {
                    if let Ok(assembly) = step_assemblies::parse_assembly_file(path) {
                        for node in assembly.walk() {
                            if node.children.is_empty() {
                                continue; // leaf occurrences were imported above
                            }
                            let Some(representation_id) = node.representation_id else { continue };
                            if !context.has_geometry(representation_id) {
                                continue;
                            }
                            match context.import(representation_id) {
                                Ok(mut import) if !import.mesh.faces.is_empty() => {
                                    transform_mesh(
                                        &mut import.mesh,
                                        &scaled_world_transform(&node.world_transform, components.unit_scale_mm),
                                    );
                                    meshes.push(import.mesh);
                                }
                                Ok(_) => failures.push(format!("{}: representation #{representation_id} produced no triangles", node.name)),
                                Err(error) => {
                                    if error.contains("Mx-72") || error.contains("mixed planar/analytic") {
                                        mixed_representations += 1;
                                    }
                                    failures.push(format!("{}: {error}", node.name));
                                }
                            }
                        }
                    }
                }
                if failures.is_empty() && !meshes.is_empty() {
                    diagnostics.bodies = meshes.len();
                    let mesh = merge_meshes(meshes);
                    diagnostics.mesh_vertices = mesh.vertices.len();
                    diagnostics.mesh_triangles = mesh.faces.len();
                    return Ok(StepImport { mesh, diagnostics });
                }
                let mut reasons: Vec<String> = Vec::new();
                if !components.missing_documents.is_empty() {
                    let searched = components
                        .unresolved
                        .iter()
                        .take(2)
                        .map(|document| {
                            let paths = document
                                .searched
                                .iter()
                                .take(4)
                                .map(|path| path.display().to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("{} (searched: {paths})", document.file_name)
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    reasons.push(format!(
                        "{} external document(s) missing: {}; set STEP_ASSEMBLY_SEARCH_PATHS or NODECRAFT_STEP_SEARCH_PATHS to a directory containing them{}",
                        components.missing_documents.len(),
                        components.missing_documents.join(", "),
                        if searched.is_empty() { String::new() } else { format!(" [{searched}]") }
                    ));
                }
                if mixed_representations > 0 {
                    reasons.push(format!(
                        "{mixed_representations} in-file representation(s) blocked by Mx-72 (mixed planar/analytic child solids)"
                    ));
                }
                if mixed_external_documents > 0 {
                    reasons.push(format!(
                        "{mixed_external_documents} external document(s) blocked by Mx-72 (mixed planar/analytic child solids)"
                    ));
                }
                for failure in failures.iter().take(4) {
                    reasons.push(failure.clone());
                }
                if failures.len() > 4 {
                    reasons.push(format!("… and {} more failure(s)", failures.len() - 4));
                }
                if reasons.is_empty() {
                    reasons.push("no occurrence produced geometry".into());
                }
                return Err(format!(
                    "STEP assembly '{root}' with {total_occurrences} occurrence(s) cannot be meshed completely (Step-106): {meshed_occurrences} meshed; {}",
                    reasons.join("; ")
                ));
            }
        }
        if let Ok(assembly) = step_assemblies::parse_assembly_file(path) {
            if assembly.node_count() > 1 {
                let root = assembly.root().map(|node| node.name.as_str()).unwrap_or("assembly");
                let mut products: Vec<&str> = Vec::new();
                for node in assembly.walk() { if !products.contains(&node.product_name.as_str()) { products.push(&node.product_name); } }
                let sample = products.iter().take(6).copied().collect::<Vec<_>>().join(", ");
                return Err(format!(
                    "STEP assembly '{root}' with {} occurrences and {} product(s) is not yet supported by the Rust worker ({sample}{}); convert it to a single solid or use the SMLib worker",
                    assembly.node_count(), products.len(), if products.len() > 6 { ", …" } else { "" }
                ));
            }
        }
    }
    if let Ok(report) = nodecraft_step::classify_step(path) {
        if report.import_path() == nodecraft_step::ImportPath::MixedUnsupported {
            if report.solids <= 1 {
                // Single-solid mixed planar/cylindrical: the validated exact mixed adapter is
                // the last precise reader for this shape, so try it before reporting unsupported.
                match import_mixed_step(path) {
                    Ok(import) => return Ok(import),
                    Err(mixed_error) => {
                        let surfaces = if report.cylinders > 0 { "CYLINDRICAL_SURFACE" } else if report.cones > 0 { "CONICAL_SURFACE" } else if report.spheres > 0 { "SPHERICAL_SURFACE" } else if report.tori > 0 { "TOROIDAL_SURFACE" } else { "curved surface" };
                        return Err(format!(
                            "STEP geometry is mixed planar/analytic ({} plane face(s), {} {surfaces} face(s)); exact mixed import failed: {mixed_error}",
                            report.planes, report.faces.saturating_sub(report.planes)
                        ));
                    }
                }
            } else {
                // Several solids in one body. First keep the established planar/analytic
                // multi-solid path when it already covers every solid (e.g. rust-export);
                // only use the exact mixed reader for the solids it could not build.
                if let Ok((bodies, skipped)) = nodecraft_step::import_solids_partial(path) {
                    if skipped.is_empty() && bodies.len() == report.solids {
                        return import_multi_bodies(&bodies);
                    }
                }
                // Several mixed solids in one body: build every solid exactly and merge the
                // display meshes, refusing partial bodies explicitly (Step-108).
                match mixed_multi::import_mixed_solids(path) {
                    Ok(import) => return Ok(import),
                    Err(mixed_error) => {
                        let surfaces = if report.cylinders > 0 { "CYLINDRICAL_SURFACE" } else if report.cones > 0 { "CONICAL_SURFACE" } else if report.spheres > 0 { "SPHERICAL_SURFACE" } else if report.tori > 0 { "TOROIDAL_SURFACE" } else { "curved surface" };
                        return Err(format!(
                            "STEP geometry is mixed planar/analytic ({} plane face(s), {} {surfaces} face(s), {} solid(s)); exact mixed multi-solid import failed: {mixed_error}",
                            report.planes, report.faces.saturating_sub(report.planes), report.solids
                        ));
                    }
                }
            }
        }
    }
    if let Err(error) = nodecraft_step::validate_header(path) {
        return Err(format!("STEP header validation failed: {error}"));
    }
    let report = step_schema::validate_file(path);
    if !report.is_valid() {
        let details: Vec<String> = report.errors().take(3).map(|issue| issue.to_string()).collect();
        return Err(format!("STEP validation failed: {}", details.join("; ")));
    }
    if let Ok(faceted) = nodecraft_step::read_faceted_breps(path) {
        if !faceted.is_empty() {
            // A STEP file can store one lump as FACETED_BREP and another as an advanced
            // MANIFOLD_SOLID_BREP (cubsomcy.stp: faceted box + split-cylinder). The
            // faceted reader only sees its own entity kind, so merge the readable
            // advanced lumps instead of silently dropping them.
            if let Ok((advanced, skipped)) = nodecraft_step::import_solids_partial(path) {
                if !advanced.is_empty() && skipped.is_empty() {
                    let mut bodies: Vec<nodecraft_step::StepBody> =
                        faceted.into_iter().map(nodecraft_step::StepBody::Planar).collect();
                    bodies.extend(advanced);
                    return import_multi_bodies(&bodies);
                }
            }
            return import_faceted_brep(path);
        }
    }
    if let Ok((bodies, skipped)) = nodecraft_step::import_solids_partial(path) {
        if bodies.len() > 1 && skipped.is_empty() { return import_multi_solids(path); }
        if !skipped.is_empty() {
            let mut result = if bodies.is_empty() { None } else { Some(import_multi_bodies(&bodies)?) };
            let fallback = import_bspline_fallback(path, &skipped)?;
            if let Some(existing) = result.take() {
                let merged = merge_meshes(vec![existing.mesh, fallback.mesh]);
                let mut diagnostics = existing.diagnostics;
                diagnostics.bodies += fallback.diagnostics.bodies;
                diagnostics.warnings.extend(fallback.diagnostics.warnings);
                diagnostics.kind = "multi-solid+bspline".into();
                diagnostics.mesh_vertices = merged.vertices.len();
                diagnostics.mesh_triangles = merged.faces.len();
                return Ok(StepImport { mesh: merged, diagnostics });
            }
            return Ok(fallback);
        }
    }
    if let Some(solid) = nodecraft_step::detect_analytic_solid(path)? {
        return import_analytic_step(path, solid);
    }
    // Planar solids keep the exact planar Brep path. Anything it cannot read (curved
    // faces, B-spline sheets/solids) falls through to the general solidcore STEP plug-in.
    match nodecraft_step::read_planar_solids(path) {
        Ok(bodies) if !bodies.is_empty() => return finish_planar_import(bodies),
        Ok(_) => {
            return import_solidcore_step_or_error(path, "STEP file has no planar solids");
        }
        Err(error) if error.contains("no MANIFOLD_SOLID_BREP") => {
            // Step-107: a shell/sheet model without solid topology (for example the
            // CATIA short-name FACETED_BREP shell in socks.stp) gets a precise planar
            // display mesh before the curved-sheet readers run.
            let sheet_error = match step_sheet::import_sheet(path) {
                Ok(Some(sheet)) => {
                    let mut diagnostics = StepDiagnostics {
                        kind: sheet.kind.into(),
                        bodies: sheet.bodies,
                        ..StepDiagnostics::default()
                    };
                    diagnostics.mesh_vertices = sheet.mesh.vertices.len();
                    diagnostics.mesh_triangles = sheet.mesh.faces.len();
                    diagnostics.warnings = sheet.warnings;
                    return Ok(StepImport { mesh: sheet.mesh, diagnostics });
                }
                Ok(None) => None,
                Err(sheet_error) => Some(sheet_error),
            };
            // Sheets: try the exact solidcore sheet reader first (B-spline surfaces).
            match import_solidcore_step(path) {
                Ok(import) => return Ok(import),
                Err(solidcore_error) => {
                    // Capture the surface-model error instead of dropping it: for a real
                    // surface model that failed to tessellate the classification must say
                    // so, not report "no geometry".
                    let surface_model_error = match surface_model::surface_model_mesh(path) {
                        Ok(Some(mesh)) => {
                            let mut diagnostics = StepDiagnostics {
                                kind: "surface-model".into(),
                                ..StepDiagnostics::default()
                            };
                            diagnostics.warnings.push("STEP surface model imported as a display mesh; exact solid Brep and mass properties are unavailable".into());
                            diagnostics.mesh_vertices = mesh.vertices.len();
                            diagnostics.mesh_triangles = mesh.faces.len();
                            return Ok(StepImport { mesh, diagnostics });
                        }
                        Ok(None) => None,
                        Err(error) => Some(error),
                    };
                    let surface_note = surface_model_error
                        .as_deref()
                        .map(|error| format!("; surface model tessellation failed: {error}"))
                        .unwrap_or_default();
                    // A curved sheet model is not representable by the planar sheet reader,
                    // so keep that precise reason in the final unsupported message instead
                    // of silently dropping it.
                    let sheet_note = sheet_error
                        .as_deref()
                        .map(|error| format!("; planar sheet import failed: {error}"))
                        .unwrap_or_default();
                    // A planar sheet is still imported exactly by the planar shell
                    // reader; approximating a *curved* sheet as a polygon would be a
                    // partial import, so only planar-only files may use that fallback.
                    let planar_only = nodecraft_step::inspect_features(path)
                        .is_ok_and(|features| {
                            features.bspline_surfaces == 0 && features.analytic_surfaces == 0
                        });
                    if !planar_only {
                        return Err(step_content::unsupported_no_solid_message(
                            path,
                            &format!("{error}; {solidcore_error}{surface_note}{sheet_note}"),
                        ));
                    }
                    return match nodecraft_step::read_planar_body(path) {
                        Ok(body) => finish_planar_import(vec![body]),
                        Err(planar_error) => Err(step_content::unsupported_no_solid_message(
                            path,
                            &format!("{error}; {solidcore_error}; {planar_error}{surface_note}{sheet_note}"),
                        )),
                    };
                }
            }
        }
        Err(error) => {
            return import_solidcore_step_or_error(path, &format!("planar import failed: {error}"));
        }
    }
}

/// Tessellate exact planar bodies into one display mesh and fill the diagnostics.
fn finish_planar_import(bodies: Vec<nodecraft_step::PlanarBody>) -> Result<StepImport, String> {
    let mut diagnostics = StepDiagnostics { kind: "planar-brep".into(), bodies: bodies.len(), ..StepDiagnostics::default() };
    let mut meshes = Vec::new();
    for body in bodies.iter() {
        let mesh = planar_body_brep_mesh(body, &mut diagnostics)?;
        meshes.push(mesh);
    }
    let mesh = merge_meshes(meshes);
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    if !diagnostics.fallbacks.is_empty() {
        diagnostics.kind = "planar-brep-fallback".into();
    }
    Ok(StepImport { mesh, diagnostics })
}

/// General STEP import through the new solidcore readers (mixed planar/cylindrical
/// solids, B-spline sheets, analytic primitives): the exact Brep is built first and the
/// display mesh is tessellated from it (never from a partial planar shell).
fn import_solidcore_step(path: &str) -> Result<StepImport, String> {
    let import = nodecraft_solidcore::step_display_mesh(path)?;
    let mesh = import.mesh;
    let mut diagnostics = StepDiagnostics {
        kind: import.kind.as_str().into(),
        breps: 1,
        ..StepDiagnostics::default()
    };
    diagnostics.brep_faces = import.brep_faces;
    diagnostics.brep_edges = import.brep_edges;
    diagnostics.brep_vertices = import.brep_vertices;
    diagnostics.warnings = import.warnings;
    diagnostics.fallbacks = import.fallbacks;
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(StepImport { mesh, diagnostics })
}

/// Reports the general solidcore import error together with the planar attempt and the
/// file's feature/topology summary, so unsupported files stay actionable.
fn import_solidcore_step_or_error(path: &str, planar_note: &str) -> Result<StepImport, String> {
    match import_solidcore_step(path) {
        Ok(import) => Ok(import),
        Err(error) => {
            let features = nodecraft_step::inspect_features(path).ok();
            let topology = nodecraft_step::read_topology_file(path).ok().map(|brep| brep.summary());
            Err(match features {
                Some(features) if features.advanced_faces > 0 || features.products > 0 => format!(
                    "{planar_note}; {error}; unsupported STEP features: {}{}",
                    features.unsupported_summary(),
                    topology.map(|summary| format!("; topology: {summary}")).unwrap_or_default()
                ),
                _ => format!("{planar_note}; {error}"),
            })
        }
    }
}




#[cfg(test)]
mod assembly_transform_tests {
    use super::transform_mesh;
    use nodecraft_protocol::Mesh;

    fn triangle() -> Mesh {
        Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        }
    }

    #[test]
    fn mirrored_occurrence_flips_winding_and_normals() {
        let mut mesh = triangle();
        let mirror = step_assemblies::Transform::from_rows([
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]).expect("mirror transform");
        transform_mesh(&mut mesh, &mirror);
        assert_eq!(mesh.faces[0], [0, 2, 1], "winding must reverse for det<0");
        assert!((mesh.normals[0][2] + 1.0).abs() < 1e-6, "normals must negate: {:?}", mesh.normals[0]);
        assert!((mesh.vertices[0][0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_scale_keeps_normals_unit_length() {
        let mut mesh = triangle();
        let scale = step_assemblies::Transform::from_rows([
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]).expect("scale transform");
        transform_mesh(&mut mesh, &scale);
        assert_eq!(mesh.faces[0], [0, 1, 2]);
        let normal = mesh.normals[0];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        assert!((length - 1.0).abs() < 1e-5, "normal length {length}");
    }
}
