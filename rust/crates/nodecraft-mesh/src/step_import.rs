//! Official STEP import: solidcore-step -> solidcore-convert -> solidcore-tess.
//!
//! `solidcore_step::p21` parses the exchange structure and
//! `solidcore_step::topology` reads every solid into the schema-4 record form; the
//! record is converted to a kernel B-rep with `solidcore-convert`, and the display
//! triangles come from the upstream layer through `nodecraft-layers` /
//! `solidcore-tess`, the same mesher every evaluated Part node uses.
//!
//! Assembly placement, body metadata and diagnostics live in this module.
//!
//! This is the worker's STEP import adapter.
//!
//! # Assemblies
//!
//! The official reader maps every B-rep in a file exactly, but it does not apply STEP
//! product/assembly instancing: its bodies stay in the coordinates of the
//! representation that defines them. Rendering those bodies as-is stacks every component
//! at the origin (or spreads unrelated, unplaced bodies through the viewport).
//!
//! When a file has placed occurrences, this module uses the product `step-assemblies`
//! parser instead:
//!
//! * [`step_assemblies::component_instances`] enumerates the leaf occurrences and their
//!   geometry source ([`step_assemblies::ComponentSource`]); the official reader still
//!   supplies the meshes.
//! * An `InFileRepresentation` names a shape representation. Its solid items are resolved
//!   through untransformed `*_REPRESENTATION_RELATIONSHIP` links (CAD files routinely put
//!   the solid in an `ADVANCED_BREP_SHAPE_REPRESENTATION` related to the placement-only
//!   `SHAPE_REPRESENTATION` a product names), then matched to the official reader's
//!   solid items.
//! * Every meshed body is placed with [`step_assemblies::ComponentInstance::world_transform_mm`]
//!   (or the accumulated node transform for the root/sub-assembly's own geometry, whose
//!   translation is in file units).
//! * `ExternalDocument` children are imported recursively and transformed; missing
//!   documents, `NoGeometry` occurrences and bodies that no occurrence places are reported
//!   as warnings and never emitted unplaced.
//!
//! The result is one merged display mesh whose bodies are all in the assembly's world
//! frame in millimetres. Bodies that no occurrence references are excluded.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;

use nodecraft_protocol::Mesh;
use nodecraft_step::{StepFile, Value};
use solidcore_step::p21;
use solidcore_step::topology::{self, Solid};
use step_assemblies::{
    Assembly, ComponentInstance, ComponentSource, Components, Representation, Transform,
};

/// Diagnostics for one `stepimporter` evaluation.
#[derive(Debug, Clone, Default)]
pub struct StepDiagnostics {
    pub kind: String,
    pub bodies: usize,
    pub breps: usize,
    pub brep_faces: usize,
    pub brep_edges: usize,
    pub brep_vertices: usize,
    pub mesh_vertices: usize,
    pub mesh_triangles: usize,
    pub fallbacks: Vec<String>,
    pub warnings: Vec<String>,
    pub name: Option<String>,
    pub color: Option<[f64; 3]>,
    pub body_names: Vec<String>,
    pub unit: Option<String>,
    pub tolerance: Option<f64>,
    pub unit_count: Option<usize>,
    pub type_report: Option<nodecraft_step::StepTypeReport>,
    pub open_edges: usize,
    pub solid_types: Vec<nodecraft_step::SolidTypeReport>,
    /// Number of placed occurrences the assembly parser found (0 for a plain solid).
    pub instances: usize,
    /// Occurrences whose geometry was meshed and placed in the world frame.
    pub placed_instances: usize,
    /// Occurrences skipped with an explicit warning (missing external document,
    /// `NoGeometry`, or an unresolved representation).
    pub unplaced_instances: usize,
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
            "instances": self.instances,
            "placedInstances": self.placed_instances,
            "unplacedInstances": self.unplaced_instances,
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
        })
    }
}

/// One imported STEP file: display mesh + diagnostics.
#[derive(Debug, Clone, Default)]
pub struct StepImport {
    pub mesh: Mesh,
    pub diagnostics: StepDiagnostics,
}

/// The solids of one STEP file, read through `solidcore-step`, plus what the
/// reader refused and noted. Coordinates are millimetres, in the frame of each
/// solid's own representation (assembly placements are not applied here: see
/// [`step_assemblies`]).
#[derive(Debug, Clone, Default)]
struct StepModel {
    file_name: String,
    /// Millimetres per file length unit (what the file's contexts state).
    unit_mm: f64,
    bodies: Vec<Solid>,
    /// Solid items the reader refused, with why.
    failures: Vec<topology::Failure>,
    /// What the reader repaired or noted on the way.
    notes: Vec<String>,
}

/// Millimetres per file length unit, from the first readable length unit the
/// file declares (millimetres when it names none).
fn file_unit_mm(exchange: &p21::Exchange) -> f64 {
    for (id, entity) in &exchange.entities {
        if !entity.has("LENGTH_UNIT") {
            continue;
        }
        if let Some((solidcore_step::units::UnitKind::Length, factor)) =
            solidcore_step::units::unit_factor(exchange, *id)
            && factor.is_finite()
            && factor > 0.0
        {
            return factor;
        }
    }
    1.0
}

/// Parses and reads every solid of a STEP file with the official reader.
fn read_step_model(path: &str) -> Result<StepModel, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    let text = String::from_utf8_lossy(&bytes);
    read_step_model_text(path, &text)
}

/// The pcurves of a face whose uses have no closed-form trim, made by the kernel
/// (`solidcore_ops::pcurve`). The reader asks for them only where the file gives none:
/// an approximated intersection curve on a cone, a file's spline edge on a spline
/// surface. This is the same trim fit every modelling operation uses, and the old STEP
/// builder's healing pass made the same pcurves.
struct KernelTrims;

impl solidcore_ops::pcurve::Trims for KernelTrims {
    fn trims(&self, face: &solidcore_ops::pcurve::TrimFace<'_>) -> Result<Vec<Vec<solidcore_record::brep::Curve2>>, String> {
        use solidcore_ops::pcurve::{EdgeUse, pcurves_within};
        let mut loops: Vec<Vec<EdgeUse<'_>>> = Vec::with_capacity(face.loops.len());
        for loop_ in &face.loops {
            let mut uses = Vec::with_capacity(loop_.len());
            for use_ in loop_ {
                let edge = face
                    .edges
                    .get(use_.edge as usize)
                    .ok_or_else(|| format!("face use names missing edge {}", use_.edge))?;
                uses.push(EdgeUse {
                    edge: use_.edge,
                    curve: edge.curve,
                    interval: edge.interval,
                    forward: use_.forward,
                    tolerance: edge.tolerance,
                    given: use_.pcurve.as_ref(),
                });
            }
            loops.push(uses);
        }
        pcurves_within(face.surface, &loops, face.tolerance, Some(face.domain)).map_err(|error| error.to_string())
    }
}

fn read_step_model_text(path: &str, text: &str) -> Result<StepModel, String> {
    let exchange = p21::parse(text).map_err(|error| format!("{path}: official STEP reader: {error}"))?;
    let solids = topology::read_solids_with(&exchange, &KernelTrims);
    let mut bodies = solids.solids;
    // Topology no product structure lists (a bare face, shell or solid in a file's DATA
    // section) is still a body: read each root the representation walk did not reach as
    // the same schema-4 record, in id order. Bare sheets are common in test files and in
    // exporters that write a face without a shell.
    let read: std::collections::BTreeSet<u64> = bodies.iter().map(|solid| solid.item).collect();
    for item in topology::roots(&exchange) {
        if read.contains(&item) || !exchange.get(item).is_some_and(|entity| topology::is_body_item(entity.name())) {
            continue;
        }
        match topology::read_solid_with(&exchange, item, solidcore_step::units::Context::default(), &KernelTrims) {
            Ok(solid) => bodies.push(solid),
            Err(message) => {
                // Keep the reason with the other refused bodies.
                return Ok(StepModel {
                    file_name: exchange.header.file_name.clone(),
                    unit_mm: file_unit_mm(&exchange),
                    bodies,
                    failures: vec![topology::Failure { item, message }],
                    notes: Vec::new(),
                });
            }
        }
    }
    let notes: Vec<String> = bodies
        .iter()
        .flat_map(|solid| solid.notes.iter().map(|note| format!("solid #{}: {note}", solid.item)))
        .collect();
    Ok(StepModel {
        file_name: exchange.header.file_name.clone(),
        unit_mm: file_unit_mm(&exchange),
        bodies,
        failures: solids.failures,
        notes,
    })
}

/// A warning line for each solid the reader refused.
fn failure_warnings(failures: &[topology::Failure]) -> Vec<String> {
    failures
        .iter()
        .map(|failure| format!("solid #{} was not read: {}", failure.item, failure.message))
        .collect()
}

/// Merges per-body meshes into one display mesh with namespaced groups/edges.
pub fn merge_meshes(mut meshes: Vec<Mesh>) -> Mesh {
    let has_colors = meshes.iter().any(|mesh| !mesh.face_colors.is_empty());
    let mut merged = Mesh::default();
    for (index, mesh) in meshes.drain(..).enumerate() {
        let offset = merged.vertices.len() as u32;
        let prefix = format!("body:{index}:");
        merged.vertices.extend(mesh.vertices);
        merged.normals.extend(mesh.normals);
        let face_count = mesh.faces.len();
        merged.faces.extend(mesh.faces.into_iter().map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]));
        merged.face_groups.extend(mesh.face_groups.into_iter().map(|group| format!("{prefix}{group}")));
        if has_colors {
            if mesh.face_colors.is_empty() {
                merged.face_colors.extend(std::iter::repeat_n([0.0; 3], face_count));
            } else {
                merged.face_colors.extend(mesh.face_colors);
            }
        }
        merged.brep_edges.extend(mesh.brep_edges.into_iter().map(|mut edge| {
            edge.id = format!("{prefix}{}", edge.id);
            for segment in &mut edge.segments { segment[0] += offset; segment[1] += offset; }
            edge
        }));
        merged.lightweight = true;
    }
    merged
}

/// Imports a STEP file through the official reader and builds display meshes.
///
/// Single solids keep the reader's own coordinates. A file with placed occurrences is
/// imported as an assembly: every placed body is meshed once and transformed into the
/// world frame (see the module docs).
/// Cheap pre-scan for large files: how many B-rep bodies does the root file hold?
///
/// Only used to choose an initial assembly display tolerance. Small files skip the
/// extra parse; large files pay ~megabytes of parsing once and can then avoid a fine
/// discovery pass followed by a coarse pass.
fn root_body_count(path: &str) -> Option<usize> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() < 400_000 {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let exchange = p21::parse(&text).ok()?;
    Some(
        exchange
            .entities
            .values()
            .filter(|entity| topology::is_solid_item(entity.name()))
            .count(),
    )
}

pub fn import_step(path: &str) -> Result<StepImport, String> {
    // Keep a single imported model within a viewport/test budget. The per-body sag is
    // extent-derived, but a large assembly can still sum to far more triangles than a
    // viewer needs; re-import at a coarser display tolerance until it fits.
    let requested = crate::display_tolerance();
    let body_count = root_body_count(path);
    let mut budget = if requested <= 0.18 {
        110_000
    } else if requested <= 0.45 {
        80_000
    } else {
        40_000
    };
    if body_count.is_some_and(|bodies| bodies >= 16) {
        // Large assemblies are dominated by many small extent-capped bodies, not one
        // huge body. Give them a realistic total triangle budget and start closer to
        // the mesh density they can actually reach.
        budget = budget.max(250_000);
    }
    let mut coarsen = body_count
        .filter(|bodies| *bodies >= 16)
        .map(|bodies| (bodies as f64 / 16.0).sqrt().clamp(1.0, 3.0))
        .unwrap_or(1.0);
    let mut attempts = 0;
    let mut previous_faces: Option<usize> = None;
    loop {
        attempts += 1;
        crate::set_display_tolerance(Some(requested * coarsen));
        let mut stack = Vec::new();
        let result = import_step_guarded(path, &mut stack);
        crate::set_display_tolerance(Some(requested));
        // Large assemblies plateau because each small body has its own extent-capped
        // tolerance. Give them a realistic display budget rather than spending the
        // remaining attempts trying to reach the single-body budget.
        if let Ok(import) = &result
            && import.diagnostics.kind == "assembly"
            && import.diagnostics.bodies >= 16
        {
            budget = budget.max(250_000);
        }
        match result {
            Ok(import) if import.mesh.faces.len() <= budget || attempts >= 5 => return Ok(import),
            Ok(import) => {
                let faces = import.mesh.faces.len();
                // Per-body display tolerance is capped by the body extent, so an
                // assembly of small parts can plateau above the triangle budget.
                // Re-parsing and re-meshing the same result four more times is pure
                // waste; accept the first plateau.
                if previous_faces.is_some_and(|previous| faces >= previous) {
                    return Ok(import);
                }
                previous_faces = Some(faces);
                coarsen *= (faces as f64 / budget as f64)
                    .powf(0.7)
                    .max(1.12);
            }
            Err(error) => return Err(error),
        }
    }
}

fn import_step_guarded(path: &str, stack: &mut Vec<PathBuf>) -> Result<StepImport, String> {
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    if stack.iter().any(|entry| entry == &canonical) {
        return Err(format!("assembly cycle through {path}"));
    }
    stack.push(canonical);
    let result = import_step_inner(path, stack);
    stack.pop();
    result
}

fn import_step_inner(path: &str, stack: &mut Vec<PathBuf>) -> Result<StepImport, String> {
    let model = read_step_model(path)
        .map_err(|error| if error.starts_with(path) { error } else { format!("{path}: {error}") })?;
    if model.bodies.is_empty() {
        let refused = model
            .failures
            .first()
            .map(|failure| format!(" (solid #{}: {})", failure.item, failure.message))
            .unwrap_or_default();
        return Err(format!("{path}: official STEP reader found no B-rep bodies{refused}"));
    }

    match step_assemblies::component_instances(path) {
        Ok(components) if !components.instances.is_empty() => {
            let assembly = step_assemblies::parse_assembly_file(path)
                .map_err(|error| format!("{path}: assembly structure: {error}"))?;
            // A single-product file also has one "leaf" (the root itself); only a real
            // product tree has placed occurrences that need assembly placement.
            if assembly.node_count() > 1 {
                import_assembly(path, &model, &assembly, &components, stack)
            } else {
                import_bodies(path, &model)
            }
        }
        Ok(_) => import_bodies(path, &model),
        Err(error) => {
            // A file that declares occurrences but whose product graph cannot be parsed
            // must not fall back to emitting unplaced bodies.
            let text = read_step_text(path)?;
            if text.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE") {
                return Err(format!("{path}: STEP assembly occurrences could not be resolved: {error}"));
            }
            import_bodies(path, &model)
        }
    }
}

fn read_step_text(path: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Builds the display mesh for the official reader's bodies without assembly placement.
fn import_bodies(path: &str, model: &StepModel) -> Result<StepImport, String> {
    let mut meshes = Vec::with_capacity(model.bodies.len());
    let mut body_names = Vec::with_capacity(model.bodies.len());
    let mut healed_bodies = 0usize;
    for body in &model.bodies {
        body_names.push(body.name.clone());
        let (mesh, healed) = body_display_mesh(body)
            .map_err(|error| format!("{path}: solid #{}: {error}", body.item))?;
        if healed {
            healed_bodies += 1;
        }
        meshes.push(mesh);
    }
    let mesh = merge_meshes(meshes);
    let mut warnings = failure_warnings(&model.failures);
    warnings.extend(model.notes.iter().cloned());
    if healed_bodies > 0 {
        warnings.push(format!("{healed_bodies} body/bodies needed STEP reader repairs (turned faces or raised tolerances)"));
    }
    let mut diagnostics = StepDiagnostics {
        kind: "solidcore-brep".into(),
        bodies: model.bodies.len(),
        breps: model.bodies.len(),
        body_names,
        unit: Some(format!("{} mm/unit", model.unit_mm)),
        name: Some(model.file_name.clone()).filter(|name| !name.is_empty()),
        mesh_vertices: mesh.vertices.len(),
        mesh_triangles: mesh.faces.len(),
        warnings,
        ..StepDiagnostics::default()
    };
    diagnostics.brep_faces = model.bodies.iter().map(|body| body.brep.faces.len()).sum();
    diagnostics.brep_edges = model.bodies.iter().map(|body| body.brep.edges.len()).sum();
    diagnostics.brep_vertices = model.bodies.iter().map(|body| body.brep.vertices.len()).sum();
    Ok(StepImport { mesh, diagnostics })
}

/// Meshes one official-reader body, rebuilding with healed UV trims when needed.
///
/// Returns the mesh and whether the healed rebuild was required.

fn mesh_has_invalid_coordinates(mesh: &Mesh) -> bool {
    mesh.vertices.iter().any(|vertex| {
        vertex
            .iter()
            .any(|coordinate| !coordinate.is_finite() || coordinate.abs() > 1_000_000.0)
    })
}

fn body_display_mesh(body: &Solid) -> Result<(Mesh, bool), String> {
    // Tessellate the reader's record directly: the record is the exact geometry the
    // reader validated, and `solidcore-tess` plans on it (no kernel round trip).
    let healed = !body.notes.is_empty() || !body.turned.is_empty();
    // ---------------------------------------------------------------------------
    // SolidCraft LOD contract (read before changing these factors)
    //
    // The importer has no camera. `requested` is the requested max sag in model units;
    // it currently doubles as a coarse LOD-tier signal. The body-extent factor is a
    // fallback so a 3 mm plug and a metre-scale part get sensible density from the same
    // request. It is not the long-term LOD policy.
    //
    // SolidCraft should pass an explicit world-space sag chosen by the worker from:
    //   desired screen sag in pixels × world units per pixel
    // and clamp it by the body extent only as a safety bound. The worker owns LOD:
    //   - exact geometry and display meshing run on independent thread pools
    //   - mesh jobs are immutable-BRep, revisioned, cancellable and coalesced
    //   - close/selected work gets a small screen sag; normal is bounded; far is coarse
    //   - cache key is geometry revision + node + quality, never just graph revision
    // The factors below are compatibility defaults until that request carries a
    // screen-space tolerance. Do not add per-model special cases here.
    // ---------------------------------------------------------------------------
    const CLOSE_FACTOR: f64 = 4.0e-5;
    const NORMAL_FACTOR: f64 = 4.0e-4;
    const FAR_FACTOR: f64 = 1.2e-3;
    let requested = crate::display_tolerance();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &body.brep.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.point[axis]);
            max[axis] = max[axis].max(vertex.point[axis]);
        }
    }
    let extent = if min[0].is_finite() {
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt()
    } else { 0.0 };
    let factor = if requested <= 0.18 {
        CLOSE_FACTOR
    } else if requested <= 0.45 {
        NORMAL_FACTOR
    } else {
        FAR_FACTOR
    };
    let effective = if extent > 0.0 { requested.min(extent * factor).max(1.0e-9) } else { requested };
    crate::set_display_tolerance(Some(effective));
    // The local LOD factor can ask for a sub-micrometre sag on a small imported body.
    // solidcore-tess's planner then spends seconds on a planar face that needs none of
    // it, while producing the same triangles. Keep the upstream tessellation at a
    // viewport-meaningful sag.
    const MIN_TESSELLATION_SAG: f64 = 0.05;
    let mesh = nodecraft_layers::tessellate_record(
        &body.brep,
        &nodecraft_layers::MeshTolerance::display(effective.max(MIN_TESSELLATION_SAG).max(1.0e-5)),
        extent.max(1.0),
    )
    .map_err(|error| format!("upstream tessellation: {error}"))?
    .mesh;

    if mesh_has_invalid_coordinates(&mesh) {
        return Err("display mesh contains infinity-marker coordinates".into());
    }
    if mesh.faces.is_empty() {
        return Err("tessellation produced no triangles".into());
    }
    Ok((mesh, healed))
}

/// Mesh independent assembly bodies on the available cores.
///
/// Bodies share only the read-only neutral model; each worker sets the thread-local
/// display tolerance, and the resulting meshes are later placed/instanced by the
/// single-threaded assembly walk.
fn premesh_bodies(
    bodies: &BTreeMap<u64, &Solid>,
) -> BTreeMap<u64, Result<(Mesh, bool), String>> {
    let requested = crate::display_tolerance();
    let entries: Vec<(u64, &Solid)> = bodies.iter().map(|(id, body)| (*id, *body)).collect();
    let mut results = BTreeMap::new();
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(entries.len().max(1));
    let entries = &entries;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                crate::set_display_tolerance(Some(requested));
                let mut local = Vec::new();
                for index in (worker..entries.len()).step_by(workers) {
                    let (id, body) = entries[index];
                    local.push((id, body_display_mesh(body)));
                }
                local
            }));
        }
        for handle in handles {
            for (id, result) in handle.join().expect("body tessellation thread panicked") {
                results.insert(id, result);
            }
        }
    });
    results
}

/// Places one body into the assembly world frame and accumulates diagnostics.
struct AssemblyCollector<'a> {
    path: &'a str,
    root: &'a str,
    bodies: BTreeMap<u64, &'a Solid>,
    premesh: BTreeMap<u64, Result<(Mesh, bool), String>>,
    cache: BTreeMap<u64, Mesh>,
    meshes: Vec<Mesh>,
    body_names: Vec<String>,
    placed_ids: BTreeSet<u64>,
    placed_bodies: usize,
    brep_faces: usize,
    brep_edges: usize,
    brep_vertices: usize,
    healed: usize,
}

impl<'a> AssemblyCollector<'a> {
    fn place_body(&mut self, body_id: u64, transform: &Transform, name: &str) -> Result<(), String> {
        let body = *self
            .bodies
            .get(&body_id)
            .ok_or_else(|| format!("{}: assembly '{}' body #{body_id} was not mapped by the official reader", self.path, self.root))?;
        if !self.cache.contains_key(&body_id) {
            let result = match self.premesh.remove(&body_id) {
                Some(result) => result,
                None => body_display_mesh(body),
            };
            let (mesh, healed) = result
                .map_err(|error| format!("{}: assembly '{}' body #{body_id}: {error}", self.path, self.root))?;
            if healed {
                self.healed += 1;
            }
            self.cache.insert(body_id, mesh);
        }
        let mut mesh = self.cache.get(&body_id).expect("body mesh was just inserted").clone();
        transform_mesh(&mut mesh, transform);
        self.meshes.push(mesh);
        self.placed_ids.insert(body_id);
        self.placed_bodies += 1;
        self.brep_faces += body.brep.faces.len();
        self.brep_edges += body.brep.edges.len();
        self.brep_vertices += body.brep.vertices.len();
        self.body_names.push(name.to_string());
        Ok(())
    }
}

fn import_assembly(
    path: &str,
    model: &StepModel,
    assembly: &Assembly,
    components: &Components,
    stack: &mut Vec<PathBuf>,
) -> Result<StepImport, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
    let resolver = BodyResolver::new(&file, &assembly.representations, model);
    let root = assembly
        .root()
        .map(|node| node.name.clone())
        .unwrap_or_else(|| "assembly".into());
    let bodies: BTreeMap<u64, &Solid> = model.bodies.iter().map(|body| (body.item, body)).collect();

    let mut warnings: Vec<String> = components.warnings.clone();
    warnings.extend(failure_warnings(&model.failures));
    warnings.extend(model.notes.iter().cloned());

    let premesh = premesh_bodies(&bodies);
    let mut collector = AssemblyCollector {
        path,
        root: &root,
        bodies,
        premesh,
        cache: BTreeMap::new(),
        meshes: Vec::new(),
        body_names: Vec::new(),
        placed_ids: BTreeSet::new(),
        placed_bodies: 0,
        brep_faces: 0,
        brep_edges: 0,
        brep_vertices: 0,
        healed: 0,
    };
    let mut placed_instances = 0usize;
    let mut unplaced_instances = 0usize;

    // The root product (and any nested sub-assembly) can carry its own geometry in
    // addition to the placed children. Its world transform is in file units, while the
    // official reader already produced millimetres, so only the translation is scaled.
    for node in assembly.walk() {
        if node.children.is_empty() {
            continue;
        }
        let Some(representation_id) = node.representation_id else { continue };
        let resolved = resolver.resolve(representation_id);
        if resolved.bodies.is_empty() {
            if resolved.mapped_items > 0 {
                warnings.push(format!(
                    "assembly node {:?}: representation #{representation_id} places its geometry through MAPPED_ITEM, which the product tree does not resolve; the node's geometry was not imported",
                    node.name
                ));
            }
            continue;
        }
        let transform = scaled_world_transform(&node.world_transform, components.unit_scale_mm);
        for body_id in resolved.bodies {
            collector.place_body(body_id, &transform, &node.name)?;
        }
    }

    for instance in &components.instances {
        match &instance.source {
            ComponentSource::InFileRepresentation { representation_id } => {
                let resolved = resolver.resolve(*representation_id);
                if resolved.bodies.is_empty() {
                    unplaced_instances += 1;
                    warnings.push(instance_without_geometry(instance, *representation_id, resolved.mapped_items));
                    continue;
                }
                let transform = instance.world_transform_mm();
                for body_id in resolved.bodies {
                    collector.place_body(body_id, &transform, &instance.name)?;
                }
                placed_instances += 1;
            }
            ComponentSource::ExternalDocument { document, path: document_path, exists } => {
                if !*exists {
                    unplaced_instances += 1;
                    warnings.push(format!(
                        "assembly instance {:?} ({}): external document {} was not found{}; the instance was not placed",
                        instance.name,
                        instance.product_name,
                        document.file_name,
                        missing_document_detail(components, &document.file_name),
                    ));
                    continue;
                }
                let child_path = document_path.to_string_lossy().into_owned();
                match import_step_guarded(&child_path, stack) {
                    Ok(child) if !child.mesh.faces.is_empty() => {
                        let transform = instance.world_transform_mm();
                        let StepImport { mut mesh, diagnostics: child_diagnostics } = child;
                        transform_mesh(&mut mesh, &transform);
                        collector.meshes.push(mesh);
                        collector.placed_bodies += child_diagnostics.bodies.max(1);
                        collector.brep_faces += child_diagnostics.brep_faces;
                        collector.brep_edges += child_diagnostics.brep_edges;
                        collector.brep_vertices += child_diagnostics.brep_vertices;
                        if child_diagnostics.body_names.is_empty() {
                            collector.body_names.push(instance.name.clone());
                        } else {
                            for name in &child_diagnostics.body_names {
                                collector.body_names.push(format!("{}:{name}", instance.name));
                            }
                        }
                        for warning in &child_diagnostics.warnings {
                            warnings.push(format!("{}: {warning}", instance.name));
                        }
                        placed_instances += 1;
                    }
                    Ok(_) => {
                        unplaced_instances += 1;
                        warnings.push(format!(
                            "assembly instance {:?} ({}): external document {} produced no triangles; the instance was not placed",
                            instance.name, instance.product_name, document.file_name
                        ));
                    }
                    Err(error) => {
                        unplaced_instances += 1;
                        warnings.push(format!(
                            "assembly instance {:?} ({}): external document {} could not be imported: {error}",
                            instance.name, instance.product_name, document.file_name
                        ));
                    }
                }
            }
            ComponentSource::NoGeometry => {
                unplaced_instances += 1;
                warnings.push(format!(
                    "assembly instance {:?} ({}): occurrence has no shape representation or external document; the instance was not placed",
                    instance.name, instance.product_name
                ));
            }
        }
    }

    // Never silently render a representation body that no occurrence places.
    let unplaced_bodies: Vec<&Solid> = model
        .bodies
        .iter()
        .filter(|body| !collector.placed_ids.contains(&body.item))
        .collect();
    if !unplaced_bodies.is_empty() {
        let sample = unplaced_bodies
            .iter()
            .take(6)
            .map(|body| {
                if body.name.is_empty() {
                    format!("#{}", body.item)
                } else {
                    format!("#{} ({})", body.item, body.name)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let more = if unplaced_bodies.len() > 6 {
            format!(", … and {} more", unplaced_bodies.len() - 6)
        } else {
            String::new()
        };
        warnings.push(format!(
            "{} STEP body/bodies are not placed by any assembly occurrence and were not imported: {sample}{more}",
            unplaced_bodies.len()
        ));
    }

    if collector.meshes.is_empty() {
        let reasons = if warnings.is_empty() {
            "no occurrence produced geometry".to_string()
        } else {
            warnings.iter().take(6).cloned().collect::<Vec<_>>().join("; ")
        };
        return Err(format!(
            "{path}: STEP assembly '{root}' with {} occurrence(s) produced no placed geometry: {reasons}",
            components.instances.len()
        ));
    }

    let mesh = merge_meshes(collector.meshes);
    let mut diagnostics = StepDiagnostics {
        kind: "assembly".into(),
        bodies: collector.placed_bodies,
        breps: collector.placed_bodies,
        body_names: collector.body_names,
        unit: Some(format!("{} mm/unit", components.unit_scale_mm)),
        name: Some(model.file_name.clone()).filter(|name| !name.is_empty()),
        mesh_vertices: mesh.vertices.len(),
        mesh_triangles: mesh.faces.len(),
        warnings,
        instances: components.instances.len(),
        placed_instances,
        unplaced_instances,
        ..StepDiagnostics::default()
    };
    diagnostics.brep_faces = collector.brep_faces;
    diagnostics.brep_edges = collector.brep_edges;
    diagnostics.brep_vertices = collector.brep_vertices;
    if collector.healed > 0 {
        diagnostics
            .warnings
            .push(format!("{} body/bodies needed STEP reader repairs (turned faces or raised tolerances)", collector.healed));
    }
    Ok(StepImport { mesh, diagnostics })
}

/// Why an in-file occurrence was skipped, naming the `MAPPED_ITEM` blocker when present.
fn instance_without_geometry(
    instance: &ComponentInstance,
    representation_id: u32,
    mapped_items: usize,
) -> String {
    if mapped_items > 0 {
        format!(
            "assembly instance {:?} ({}): representation #{representation_id} places its geometry through MAPPED_ITEM, whose transforms are not exposed by the product tree; the instance was not placed",
            instance.name, instance.product_name
        )
    } else {
        format!(
            "assembly instance {:?} ({}): representation #{representation_id} has no solid geometry; the instance was not placed",
            instance.name, instance.product_name
        )
    }
}

fn missing_document_detail(components: &Components, file_name: &str) -> String {
    let Some(unresolved) = components
        .unresolved
        .iter()
        .find(|document| document.file_name == file_name)
    else {
        return String::new();
    };
    let searched = unresolved
        .searched
        .iter()
        .take(3)
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if searched.is_empty() {
        String::new()
    } else {
        format!(" (searched: {searched})")
    }
}

/// Places a child mesh whose coordinates are already millimetres.
///
/// Mirrored/negative-scale occurrences have a negative determinant: their triangle
/// winding is reversed and the normals are negated so faces stay outward-facing. Normals
/// transform by the inverse-transpose of the linear part (the linear part is used when
/// the transform is singular).
fn transform_mesh(mesh: &mut Mesh, transform: &Transform) {
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
        let length = (transformed[0] * transformed[0]
            + transformed[1] * transformed[1]
            + transformed[2] * transformed[2])
            .sqrt();
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
fn scaled_world_transform(transform: &Transform, unit_scale_mm: f64) -> Transform {
    if (unit_scale_mm - 1.0).abs() <= 1e-12 {
        return *transform;
    }
    let mut rows = transform.rows();
    for row in rows.iter_mut().take(3) {
        row[3] *= unit_scale_mm;
    }
    Transform::from_rows(rows).unwrap_or(*transform)
}

/// Resolves a shape representation to the official reader's body `step_id`s.
///
/// Representation items are followed through untransformed representation relationships;
/// transformation-carrying relationships describe occurrence placements and are excluded.
struct BodyResolver {
    items: BTreeMap<u32, Vec<u32>>,
    relations: BTreeMap<u32, BTreeSet<u32>>,
    body_ids: BTreeSet<u64>,
    mapped_items: BTreeSet<u32>,
}

#[derive(Debug, Default)]
struct ResolvedGeometry {
    /// Body entity ids in representation order.
    bodies: Vec<u64>,
    /// Number of `MAPPED_ITEM` items seen (geometry the product tree cannot place).
    mapped_items: usize,
}

impl BodyResolver {
    fn new(file: &StepFile, representations: &[Representation], model: &StepModel) -> Self {
        let items: BTreeMap<u32, Vec<u32>> = representations
            .iter()
            .map(|representation| (representation.entity_id, representation.items.clone()))
            .collect();
        let body_ids = model.bodies.iter().map(|body| body.item).collect();
        let mapped_items = file
            .entities
            .iter()
            .filter(|(_, entity)| entity.has_member("MAPPED_ITEM"))
            .map(|(id, _)| *id)
            .collect();
        let relations = representation_relations(file, &items);
        Self { items, relations, body_ids, mapped_items }
    }

    fn resolve(&self, representation_id: u32) -> ResolvedGeometry {
        let mut geometry = ResolvedGeometry::default();
        let mut visited: BTreeSet<u32> = BTreeSet::new();
        let mut queue: VecDeque<(u32, usize)> = VecDeque::new();
        queue.push_back((representation_id, 0));
        while let Some((current, depth)) = queue.pop_front() {
            if !visited.insert(current) || depth > 4 {
                continue;
            }
            if let Some(items) = self.items.get(&current) {
                for item in items {
                    let body_id = u64::from(*item);
                    if self.body_ids.contains(&body_id) {
                        if !geometry.bodies.contains(&body_id) {
                            geometry.bodies.push(body_id);
                        }
                    } else if self.mapped_items.contains(item) {
                        geometry.mapped_items += 1;
                    }
                }
            }
            if let Some(related) = self.relations.get(&current) {
                for next in related {
                    queue.push_back((*next, depth + 1));
                }
            }
        }
        geometry
    }
}

/// Undirected untransformed representation-relationship pairs.
///
/// `REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION` and anything referenced by a
/// `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` describes an occurrence placement, not a
/// geometry link, so it is excluded.
fn representation_relations(
    file: &StepFile,
    representations: &BTreeMap<u32, Vec<u32>>,
) -> BTreeMap<u32, BTreeSet<u32>> {
    let mut occurrence_relations: BTreeSet<u32> = BTreeSet::new();
    for (id, entity) in &file.entities {
        if !entity.has_member("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION") {
            continue;
        }
        occurrence_relations.insert(*id);
        for argument in &entity.args {
            collect_refs(argument, &mut occurrence_relations);
        }
    }
    let mut relations: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for (id, entity) in &file.entities {
        if occurrence_relations.contains(id) {
            continue;
        }
        let members = entity.complex_members();
        if !members.iter().any(|name| name.contains("REPRESENTATION_RELATIONSHIP")) {
            continue;
        }
        if members.iter().any(|name| name.contains("WITH_TRANSFORMATION")) {
            continue;
        }
        let mut refs = Vec::new();
        for argument in &entity.args {
            collect_representation_refs(argument, representations, &mut refs);
        }
        if refs.len() < 2 {
            continue;
        }
        relations.entry(refs[0]).or_default().insert(refs[1]);
        relations.entry(refs[1]).or_default().insert(refs[0]);
    }
    relations
}

fn collect_refs(value: &Value, out: &mut BTreeSet<u32>) {
    match value {
        Value::Ref(id) => {
            out.insert(*id);
        }
        Value::List(values) => {
            for value in values {
                collect_refs(value, out);
            }
        }
        Value::Typed(_, values) => {
            for value in values {
                collect_refs(value, out);
            }
        }
        _ => {}
    }
}

fn collect_representation_refs(
    value: &Value,
    representations: &BTreeMap<u32, Vec<u32>>,
    out: &mut Vec<u32>,
) {
    match value {
        Value::Ref(id) => {
            if representations.contains_key(id) && !out.contains(id) {
                out.push(*id);
            }
        }
        Value::List(values) => {
            for value in values {
                collect_representation_refs(value, representations, out);
            }
        }
        Value::Typed(_, values) => {
            for value in values {
                collect_representation_refs(value, representations, out);
            }
        }
        _ => {}
    }
}

/// Imports the first B-rep body of a STEP file as a kernel B-rep.
///
/// Boolean/extrude nodes consume B-reps, not display meshes, so they must not use the
/// legacy `brep_from_step` dispatch. The official reader is authoritative here: the
/// solid is read into its schema-4 record and converted through `solidcore-convert`.
pub fn import_step_brep(path: &str) -> Result<solidcore::Brep, String> {
    let model = read_step_model(path)?;
    let body = model
        .bodies
        .first()
        .ok_or_else(|| format!("{path}: official STEP reader found no B-rep bodies"))?;
    let mut kernel = nodecraft_kernel_profile::kernel();
    solidcore_convert::brep::to_kernel_body(&mut kernel, &body.brep, true)
        .map_err(|error| format!("{path}: solid #{}: {error}", body.item))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use nodecraft_step::{PlanarBody, StepFile, Value};
    use step_assemblies::write::{PlanarPart, planar_assembly_text};

    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "nodecraft-mesh-assembly-{}-{serial}-{tag}.step",
            std::process::id()
        ))
    }

    fn write_temp(text: &str, tag: &str) -> PathBuf {
        let path = temp_path(tag);
        std::fs::write(&path, text).expect("write STEP fixture");
        path
    }

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative);
        path.is_file().then_some(path)
    }

    /// An axis-aligned box centred on `centre`, with outward-wound faces.
    fn box_body(size: [f64; 3], centre: [f64; 3]) -> PlanarBody {
        let (x, y, z) = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0);
        let (cx, cy, cz) = (centre[0], centre[1], centre[2]);
        PlanarBody {
            points: vec![
                [cx - x, cy - y, cz - z],
                [cx + x, cy - y, cz - z],
                [cx + x, cy + y, cz - z],
                [cx - x, cy + y, cz - z],
                [cx - x, cy - y, cz + z],
                [cx + x, cy - y, cz + z],
                [cx + x, cy + y, cz + z],
                [cx - x, cy + y, cz + z],
            ],
            faces: vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
            holes: Vec::new(),
        }
    }

    fn translation(offset: [f64; 3]) -> Transform {
        Transform::from_columns([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], offset)
    }

    fn mesh_bounds(mesh: &Mesh) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(f64::from(vertex[axis]));
                max[axis] = max[axis].max(f64::from(vertex[axis]));
            }
        }
        (min, max)
    }

    fn mesh_volume_area(mesh: &Mesh) -> (f64, f64) {
        let point = |index: u32| -> [f64; 3] {
            let vertex = mesh.vertices[index as usize];
            [f64::from(vertex[0]), f64::from(vertex[1]), f64::from(vertex[2])]
        };
        let mut volume = 0.0;
        let mut area = 0.0;
        for face in &mesh.faces {
            let a = point(face[0]);
            let b = point(face[1]);
            let c = point(face[2]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        (volume, area)
    }

    fn assert_bbox(
        got_min: [f64; 3],
        got_max: [f64; 3],
        want_min: [f64; 3],
        want_max: [f64; 3],
        relative: f64,
        label: &str,
    ) {
        let extent = (0..3)
            .map(|axis| (want_max[axis] - want_min[axis]).abs())
            .fold(0.0f64, f64::max)
            .max(1.0);
        let tolerance = extent * relative;
        for axis in 0..3 {
            assert!(
                (got_min[axis] - want_min[axis]).abs() <= tolerance,
                "{label}: min[{axis}] {} vs {} (tol {tolerance})",
                got_min[axis],
                want_min[axis]
            );
            assert!(
                (got_max[axis] - want_max[axis]).abs() <= tolerance,
                "{label}: max[{axis}] {} vs {} (tol {tolerance})",
                got_max[axis],
                want_max[axis]
            );
        }
    }

    fn max_entity_id(text: &str) -> u32 {
        let bytes = text.as_bytes();
        let mut position = 0usize;
        let mut max = 0u32;
        while position < bytes.len() {
            if bytes[position] == b'#' {
                let start = position + 1;
                let mut end = start;
                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                if end > start
                    && let Ok(id) = text[start..end].parse::<u32>()
                {
                    max = max.max(id);
                }
                position = end;
            } else {
                position += 1;
            }
        }
        max
    }

    /// Inserts `entities` (without `#id=`/`;`) at the end of the DATA section, numbered
    /// consecutively from one past the highest existing entity id.
    fn insert_entities(text: &str, entities: &[String]) -> String {
        assert!(!entities.is_empty());
        let base = max_entity_id(text) + 1;
        let insert = text.rfind("ENDSEC;").expect("data section end");
        let mut out = String::with_capacity(text.len() + entities.len() * 64);
        out.push_str(&text[..insert]);
        for (offset, entity) in entities.iter().enumerate() {
            out.push_str(&format!("#{}={entity};\n", base + offset as u32));
        }
        out.push_str(&text[insert..]);
        out
    }

    fn find_product_definition(file: &StepFile, name: &str) -> u32 {
        let product = file
            .entities
            .iter()
            .find_map(|(id, entity)| {
                (entity.name == "PRODUCT"
                    && entity.args.get(1) == Some(&Value::Str(name.to_string())))
                .then_some(*id)
            })
            .unwrap_or_else(|| panic!("product {name:?}"));
        let formation = file
            .entities
            .iter()
            .find_map(|(id, entity)| {
                (entity.name.starts_with("PRODUCT_DEFINITION_FORMATION")
                    && matches!(entity.args.get(2), Some(Value::Ref(value)) if *value == product))
                .then_some(*id)
            })
            .unwrap_or_else(|| panic!("formation for {name:?}"));
        file.entities
            .iter()
            .find_map(|(id, entity)| {
                (entity.name == "PRODUCT_DEFINITION"
                    && matches!(entity.args.get(2), Some(Value::Ref(value)) if *value == formation))
                .then_some(*id)
            })
            .unwrap_or_else(|| panic!("definition for {name:?}"))
    }

    #[test]
    fn assembly_import_places_components_in_the_world_frame() {
        let parts = vec![
            PlanarPart::new("A", box_body([10.0, 10.0, 10.0], [0.0, 0.0, 0.0])),
            PlanarPart::placed(
                "B",
                box_body([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]),
                translation([100.0, 0.0, 0.0]),
            ),
        ];
        let text = planar_assembly_text(&parts).expect("assembly STEP text");
        let path = write_temp(&text, "placed");
        let import = import_step(path.to_str().unwrap()).expect("assembly import");
        let _ = std::fs::remove_file(&path);

        assert_eq!(import.diagnostics.kind, "assembly");
        assert_eq!(import.diagnostics.bodies, 2, "{:?}", import.diagnostics);
        assert_eq!(import.diagnostics.instances, 2);
        assert_eq!(import.diagnostics.placed_instances, 2);
        assert_eq!(import.diagnostics.unplaced_instances, 0);
        let (min, max) = mesh_bounds(&import.mesh);
        assert_bbox(min, max, [-5.0, -5.0, -5.0], [105.0, 5.0, 5.0], 1e-5, "two boxes");
        let (volume, area) = mesh_volume_area(&import.mesh);
        assert!((volume - 2000.0).abs() < 1e-3, "volume {volume}");
        assert!((area - 1200.0).abs() < 1e-3, "area {area}");
        assert!(import.diagnostics.warnings.is_empty(), "{:?}", import.diagnostics.warnings);
    }

    #[test]
    fn assembly_import_warns_and_skips_an_occurrence_without_geometry() {
        let text = planar_assembly_text(&[PlanarPart::new(
            "A",
            box_body([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]),
        )])
        .expect("assembly STEP text");
        let file = StepFile::parse(&text).expect("parse assembly");
        let root_definition = find_product_definition(&file, "ASSEMBLY");
        let base = max_entity_id(&text) + 1;
        let text = insert_entities(
            &text,
            &[
                "PRODUCT('ghost','GHOST','',())".to_string(),
                format!("PRODUCT_DEFINITION_FORMATION('','',#{base})"),
                format!("PRODUCT_DEFINITION('design','',#{},$)", base + 1),
                format!(
                    "NEXT_ASSEMBLY_USAGE_OCCURRENCE('occ-ghost','GHOST','',#{root_definition},#{},$)",
                    base + 2
                ),
            ],
        );
        let path = write_temp(&text, "no-geometry");
        let import = import_step(path.to_str().unwrap()).expect("assembly import");
        let _ = std::fs::remove_file(&path);

        assert_eq!(import.diagnostics.kind, "assembly");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.instances, 2);
        assert_eq!(import.diagnostics.placed_instances, 1);
        assert_eq!(import.diagnostics.unplaced_instances, 1);
        assert!(
            import
                .diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("no shape representation")),
            "{:?}",
            import.diagnostics.warnings
        );
        let (min, max) = mesh_bounds(&import.mesh);
        assert_bbox(min, max, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 1e-5, "one box");
    }

    #[test]
    fn assembly_import_warns_and_skips_a_missing_external_document() {
        let parts = vec![
            PlanarPart::new("A", box_body([10.0, 10.0, 10.0], [0.0, 0.0, 0.0])),
            PlanarPart::placed(
                "B",
                box_body([10.0, 10.0, 10.0], [0.0, 0.0, 0.0]),
                translation([100.0, 0.0, 0.0]),
            ),
        ];
        let text = planar_assembly_text(&parts).expect("assembly STEP text");
        let file = StepFile::parse(&text).expect("parse assembly");
        let part_b_definition = find_product_definition(&file, "B");
        let base = max_entity_id(&text) + 1;
        let text = insert_entities(
            &text,
            &[
                "DOCUMENT_FILE('MISSING_PART.stp','','',$,())".to_string(),
                format!("APPLIED_DOCUMENT_REFERENCE(#{base},'',(#{part_b_definition}))"),
                format!(
                    "APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT('MISSING_PART.stp',$, $, (#{base}))"
                ),
            ],
        );
        let path = write_temp(&text, "missing-external");
        let import = import_step(path.to_str().unwrap()).expect("assembly import");
        let _ = std::fs::remove_file(&path);

        assert_eq!(import.diagnostics.kind, "assembly");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.instances, 2);
        assert_eq!(import.diagnostics.placed_instances, 1);
        assert_eq!(import.diagnostics.unplaced_instances, 1);
        assert!(
            import.diagnostics.warnings.iter().any(|warning| {
                warning.contains("MISSING_PART.stp") && warning.contains("was not found")
            }),
            "{:?}",
            import.diagnostics.warnings
        );
    }

    #[test]
    fn bernetl_assembly_matches_the_smlib_reference() {
        let Some(path) = repository("tests/Sample AP203 Data Files/bernetl.stp") else {
            eprintln!("skipping bernetl_assembly_matches_the_smlib_reference: fixture not present");
            return;
        };
        let import = import_step(path.to_str().unwrap()).expect("bernetl assembly import");

        assert_eq!(import.diagnostics.kind, "assembly", "{:?}", import.diagnostics.warnings);
        assert_eq!(import.diagnostics.bodies, 4, "{:?}", import.diagnostics);
        assert_eq!(import.diagnostics.instances, 3);
        assert_eq!(import.diagnostics.placed_instances, 3);
        assert_eq!(import.diagnostics.unplaced_instances, 0);
        assert!(
            import
                .diagnostics
                .warnings
                .iter()
                .any(|warning| warning.contains("3 STEP body/bodies are not placed")
                    && warning.contains("#186")),
            "{:?}",
            import.diagnostics.warnings
        );

        // Full assembly: the master plate plus the three placed DETAIL1 instances. The
        // faceted bodies that no occurrence places (#186, #1342, #1416) are excluded.
        let (min, max) = mesh_bounds(&import.mesh);
        assert_bbox(
            min,
            max,
            [-339.014958773, -150.0, -200.0],
            [883.28787976, 1523.57519443, 456.0217526524],
            1e-6,
            "bernetl assembly",
        );

        // SMLib's reference metrics describe the single MANIFOLD_SOLID_BREP (#1473):
        // SMLib reports zero bodies for faceted-only files (calo/cylcub), so its bernetl
        // result omits the DETAIL1 instances that this import places.
        let model = read_step_model(path.to_str().unwrap()).expect("official model");
        let master = model
            .bodies
            .iter()
            .find(|body| body.item == 1473)
            .expect("master MANIFOLD_SOLID_BREP");
        let (master_mesh, _) = body_display_mesh(master).expect("master display mesh");
        let (master_min, master_max) = mesh_bounds(&master_mesh);
        assert_bbox(
            master_min,
            master_max,
            [-339.014958773, 685.206366414, -200.0],
            [883.28787976, 1523.57519443, -150.0],
            1e-6,
            "bernetl master",
        );
        let (master_volume, master_area) = mesh_volume_area(&master_mesh);
        assert!(
            (master_volume - 34426866.41827984).abs() <= 34426866.41827984 * 1e-5,
            "master volume {master_volume} vs SMLib 34426866.41827984"
        );
        assert!(
            (master_area - 1536059.7315848956).abs() <= 1536059.7315848956 * 1e-5,
            "master area {master_area} vs SMLib 1536059.7315848956"
        );

        let detail = model
            .bodies
            .iter()
            .find(|body| body.item == 3202)
            .expect("DETAIL1 FACETED_BREP");
        let (detail_mesh, _) = body_display_mesh(detail).expect("detail display mesh");
        let (detail_volume, detail_area) = mesh_volume_area(&detail_mesh);
        let (total_volume, total_area) = mesh_volume_area(&import.mesh);
        assert!(
            (total_volume - (master_volume + 3.0 * detail_volume)).abs()
                <= total_volume.abs() * 1e-5,
            "assembly volume {total_volume} vs master + 3 x detail"
        );
        assert!(
            (total_area - (master_area + 3.0 * detail_area)).abs() <= total_area * 1e-5,
            "assembly area {total_area} vs master + 3 x detail"
        );
        assert_eq!(
            import.diagnostics.brep_faces,
            master.brep.faces.len() + 3 * detail.brep.faces.len()
        );
    }

    #[test]
    fn cubcylso_assembly_resolves_the_related_representation() {
        let Some(path) = repository("tests/Sample AP203 Data Files/cubcylso.stp") else {
            eprintln!("skipping cubcylso_assembly_resolves_the_related_representation: fixture not present");
            return;
        };
        let import = import_step(path.to_str().unwrap()).expect("cubcylso assembly import");

        assert_eq!(import.diagnostics.kind, "assembly", "{:?}", import.diagnostics.warnings);
        assert_eq!(import.diagnostics.bodies, 2, "{:?}", import.diagnostics);
        assert_eq!(import.diagnostics.instances, 1);
        assert_eq!(import.diagnostics.placed_instances, 1);
        assert_eq!(import.diagnostics.unplaced_instances, 0);
        // The DETAIL occurrence names a placement-only representation (#302); its solid
        // lives in the related representation (#301), so the cylinder must appear at
        // x = 200 (SMLib: [-50,-40,-50]..[225,40,50]).
        let (min, max) = mesh_bounds(&import.mesh);
        assert_bbox(min, max, [-50.0, -40.0, -50.0], [225.0, 40.0, 50.0], 1e-6, "cubcylso");
    }
}
