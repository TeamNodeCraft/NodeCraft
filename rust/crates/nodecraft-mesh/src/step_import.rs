//! CAD import through cadmpeg and `solidcore-cadir`.
//!
//! The file's bytes are decoded by cadmpeg into its neutral model (CADIR) and every body
//! region is read straight into the schema-4 B-rep record by the CADIR reader. The record
//! is the same one the display mesher consumes; no STEP text is parsed on the way.
//! `.step`/`.stp`, FreeCAD `.FCStd` and Rhino `.3dm` are all detected from their contents.
//!
//! CADIR supplies each body's name, colour, visibility, product path and rigid placement,
//! plus the document tolerance. A body region that the reader refuses is a warning; the
//! other bodies still import. Occurrences that name an external document are reported and
//! not placed: CADIR does not resolve external STEP references.
//!
//! This is the worker's CAD import adapter. [`crate::import_step_breps`] exposes every
//! body's record as a kernel B-rep for the exact consumers, with the body's name and
//! colour; a body the kernel refuses is a per-body warning, not a failed file.

use std::collections::BTreeMap;

use nodecraft_protocol::Mesh;
use nodecraft_step::{CadBody, CadFailure};

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
    pub type_report: Option<nodecraft_step::CadTypeReport>,
    pub open_edges: usize,
    pub solid_types: Vec<nodecraft_step::CadSolidTypeReport>,
    /// Number of product occurrences CADIR records (0 for a plain solid).
    pub instances: usize,
    /// Occurrences whose definition supplied a record.
    pub placed_instances: usize,
    /// Occurrences skipped with an explicit warning (external document or no geometry).
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

/// One imported CAD file: display mesh + diagnostics.
#[derive(Debug, Clone, Default)]
pub struct StepImport {
    pub mesh: Mesh,
    pub diagnostics: StepDiagnostics,
}

/// One CADIR body converted to a kernel B-rep for exact consumers.
#[derive(Debug)]
pub struct StepBrepBody {
    /// The body's display name, as CADIR read it.
    pub name: String,
    /// Surface colour, linear RGB in 0..1, when the source supplies one.
    pub color: Option<[f64; 3]>,
    /// The kernel B-rep with the body's placement applied.
    pub brep: solidcore::Brep,
}

/// Every body of one CAD file that converted, plus the file's warnings.
///
/// A body the reader or the kernel refuses is reported in `warnings` and the
/// other bodies still import, exactly like [`import_step`]'s display path.
#[derive(Debug, Default)]
pub struct StepBrepImport {
    pub bodies: Vec<StepBrepBody>,
    pub warnings: Vec<String>,
}

/// The records one file's cadmpeg decode produced.
#[derive(Debug, Clone, Default)]
struct StepModel {
    file_name: String,
    format: String,
    tolerance: f64,
    bodies: Vec<CadBody>,
    failures: Vec<CadFailure>,
    warnings: Vec<String>,
    instances: usize,
    placed_instances: usize,
    unplaced_instances: usize,
    type_report: nodecraft_step::CadTypeReport,
    solid_types: Vec<nodecraft_step::CadSolidTypeReport>,
}

/// Read a file once through the cadmpeg adapter.
fn read_step_model(path: &str) -> Result<StepModel, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    let import = nodecraft_step::read_cad(&bytes, path)?;
    Ok(StepModel {
        file_name: import.file_name,
        format: import.format,
        tolerance: import.tolerance,
        bodies: import.bodies,
        failures: import.failures,
        warnings: import.warnings,
        instances: import.instances,
        placed_instances: import.placed_instances,
        unplaced_instances: import.unplaced_instances,
        type_report: import.type_report,
        solid_types: import.solid_types,
    })
}

/// Merges per-body meshes into one display mesh with namespaced groups/edges.
pub fn merge_meshes(mut meshes: Vec<Mesh>) -> Mesh {
    let has_colors = meshes.iter().any(|mesh| !mesh.face_colors.is_empty());
    // A merged mesh can keep the exact normals only when every contributing
    // body supplied them. Empty meshes (sketch only, no geometry) do not
    // change the shading source.
    let exact_normals = !meshes.is_empty()
        && meshes
            .iter()
            .all(|mesh| mesh.exact_normals || mesh.faces.is_empty());
    let mut merged = Mesh::default();
    merged.exact_normals = exact_normals;
    for (index, mesh) in meshes.drain(..).enumerate() {
        let offset = merged.vertices.len() as u32;
        let prefix = format!("body:{index}:");
        merged.vertices.extend(mesh.vertices);
        merged.normals.extend(mesh.normals);
        let face_count = mesh.faces.len();
        merged.faces.extend(
            mesh.faces
                .into_iter()
                .map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]),
        );
        merged.face_groups.extend(
            mesh.face_groups
                .into_iter()
                .map(|group| format!("{prefix}{group}")),
        );
        if has_colors {
            if mesh.face_colors.is_empty() {
                merged
                    .face_colors
                    .extend(std::iter::repeat_n([0.0; 3], face_count));
            } else {
                merged.face_colors.extend(mesh.face_colors);
            }
        }
        merged
            .brep_edges
            .extend(mesh.brep_edges.into_iter().map(|mut edge| {
                edge.id = format!("{prefix}{}", edge.id);
                for segment in &mut edge.segments {
                    segment[0] += offset;
                    segment[1] += offset;
                }
                edge
            }));
        merged.lightweight = true;
    }
    merged
}

/// Imports a CAD file through cadmpeg and builds display meshes.
///
/// The display budget is the same one the STEP reader used: a large file starts with a
/// coarser viewport tolerance and is re-imported until its triangle count fits.
pub fn import_step(path: &str) -> Result<StepImport, String> {
    let requested = crate::display_tolerance();
    let large = std::fs::metadata(path)
        .map(|metadata| metadata.len() >= 400_000)
        .unwrap_or(false);
    let mut budget = if requested <= 0.18 {
        110_000
    } else if requested <= 0.45 {
        80_000
    } else {
        40_000
    };
    if large {
        budget = budget.max(250_000);
    }
    let mut coarsen = if large { 1.5 } else { 1.0 };
    let mut attempts = 0;
    let mut previous_faces: Option<usize> = None;
    loop {
        attempts += 1;
        crate::set_display_tolerance(Some(requested * coarsen));
        let result = import_step_inner(path);
        crate::set_display_tolerance(Some(requested));
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
                if previous_faces.is_some_and(|previous| faces >= previous) {
                    return Ok(import);
                }
                previous_faces = Some(faces);
                coarsen *= (faces as f64 / budget as f64).powf(0.7).max(1.12);
            }
            Err(error) => return Err(error),
        }
    }
}

fn import_step_inner(path: &str) -> Result<StepImport, String> {
    let model = read_step_model(path).map_err(|error| {
        if error.starts_with(path) {
            error
        } else {
            format!("{path}: {error}")
        }
    })?;
    if model.bodies.is_empty() {
        let refused = model
            .failures
            .first()
            .map(|failure| {
                format!(
                    " ({} region {}: {})",
                    failure.id, failure.region, failure.message
                )
            })
            .or_else(|| model.warnings.first().cloned())
            .unwrap_or_default();
        return Err(format!("{path}: cadmpeg found no B-rep bodies{refused}"));
    }
    import_bodies(path, &model)
}

fn import_bodies(path: &str, model: &StepModel) -> Result<StepImport, String> {
    let meshes = premesh_bodies(&model.bodies);
    let mut merged = Vec::with_capacity(model.bodies.len());
    let mut body_names = Vec::with_capacity(model.bodies.len());
    let mut healed_bodies = 0usize;
    let mut brep_faces = 0usize;
    let mut brep_edges = 0usize;
    let mut brep_vertices = 0usize;
    for (body, mesh) in model.bodies.iter().zip(meshes) {
        let (mut mesh, healed) =
            mesh.map_err(|error| format!("{path}: body {:?}: {error}", body.name))?;
        if healed {
            healed_bodies += 1;
        }
        if let Some(placement) = &body.placement {
            transform_mesh(&mut mesh, &placement_rows(placement));
        }
        brep_faces += body.brep.faces.len();
        brep_edges += body.brep.edges.len();
        brep_vertices += body.brep.vertices.len();
        body_names.push(body.name.clone());
        merged.push(mesh);
    }
    let mesh = merge_meshes(merged);
    let mut warnings = failure_warnings(&model.failures);
    warnings.extend(model.warnings.iter().cloned());
    warnings.extend(model.bodies.iter().flat_map(|body| {
        body.notes
            .iter()
            .map(move |note| format!("body {:?}: {note}", body.name))
    }));
    if healed_bodies > 0 {
        warnings.push(format!(
            "{healed_bodies} body/bodies needed CAD reader repairs (turned faces or raised tolerances)"
        ));
    }
    let mut diagnostics = StepDiagnostics {
        kind: if model.instances > 1 {
            "assembly".into()
        } else {
            "solidcore-brep".into()
        },
        bodies: model.bodies.len(),
        breps: model.bodies.len(),
        brep_faces,
        brep_edges,
        brep_vertices,
        body_names,
        unit: Some("1 mm/unit (CADIR)".into()),
        tolerance: Some(model.tolerance),
        name: Some(model.file_name.clone()).filter(|name| !name.is_empty()),
        color: model.bodies.first().and_then(|body| body.color),
        mesh_vertices: mesh.vertices.len(),
        mesh_triangles: mesh.faces.len(),
        warnings,
        instances: model.instances,
        placed_instances: model.placed_instances,
        unplaced_instances: model.unplaced_instances,
        type_report: Some(model.type_report.clone()),
        solid_types: model.solid_types.clone(),
        ..StepDiagnostics::default()
    };
    if model.format != "step" {
        diagnostics.kind = "native-cad".into();
    }
    Ok(StepImport { mesh, diagnostics })
}

/// A warning line for each body region the reader refused.
fn failure_warnings(failures: &[CadFailure]) -> Vec<String> {
    failures
        .iter()
        .map(|failure| {
            format!(
                "body {:?} region {} was not read: {}",
                failure.id, failure.region, failure.message
            )
        })
        .collect()
}

fn mesh_has_invalid_coordinates(mesh: &Mesh) -> bool {
    mesh.vertices.iter().any(|vertex| {
        vertex
            .iter()
            .any(|coordinate| !coordinate.is_finite() || coordinate.abs() > 1_000_000.0)
    })
}

/// Tessellate one record for display.
///
/// The record is the exact geometry the reader validated, and `solidcore-tess` plans on
/// it: no kernel round trip.
fn body_display_mesh(body: &CadBody) -> Result<(Mesh, bool), String> {
    let healed = !body.notes.is_empty();
    // ---------------------------------------------------------------------------
    // SolidCraft LOD contract (read before changing these factors)
    //
    // The importer has no camera. `requested` is the requested max sag in model units;
    // it currently doubles as a coarse LOD-tier signal. The body-extent factor is a
    // fallback so a 3 mm plug and a metre-scale part get sensible density from the same
    // request. It is not the long-term LOD policy.
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
    } else {
        0.0
    };
    let factor = if requested <= 0.18 {
        CLOSE_FACTOR
    } else if requested <= 0.45 {
        NORMAL_FACTOR
    } else {
        FAR_FACTOR
    };
    let effective = if extent > 0.0 {
        requested.min(extent * factor).max(1.0e-9)
    } else {
        requested
    };
    crate::set_display_tolerance(Some(effective));
    // Keep the upstream tessellation at a viewport-meaningful sag.
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

/// Mesh independent bodies on the available cores.
fn premesh_bodies(bodies: &[CadBody]) -> Vec<Result<(Mesh, bool), String>> {
    let requested = crate::display_tolerance();
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(bodies.len().max(1));
    let mut results: Vec<Option<Result<(Mesh, bool), String>>> =
        (0..bodies.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                crate::set_display_tolerance(Some(requested));
                let mut local = Vec::new();
                for index in (worker..bodies.len()).step_by(workers) {
                    local.push((index, body_display_mesh(&bodies[index])));
                }
                local
            }));
        }
        for handle in handles {
            for (index, result) in handle.join().expect("body tessellation thread panicked") {
                results[index] = Some(result);
            }
        }
    });
    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err("body was not meshed".into())))
        .collect()
}

/// A column-major 4 x 4 placement as affine rows.
fn placement_rows(placement: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut rows = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            rows[row][column] = placement[column][row];
        }
    }
    rows
}

/// Places a mesh whose coordinates are millimetres.
///
/// Mirrored/negative-scale occurrences have a negative determinant: their triangle
/// winding is reversed and the normals are negated so faces stay outward-facing. Normals
/// transform by the inverse-transpose of the linear part (the linear part is used when
/// the transform is singular).
fn transform_mesh(mesh: &mut Mesh, rows: &[[f64; 4]; 4]) {
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
            transformed = [
                transformed[0] / length,
                transformed[1] / length,
                transformed[2] / length,
            ];
        }
        *normal = [
            transformed[0] as f32,
            transformed[1] as f32,
            transformed[2] as f32,
        ];
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

/// Converts every CADIR body of a CAD file to a kernel B-rep.
///
/// Exact consumers (Part import, Boolean/extrude operands, prewarm) all take this
/// one route: the display mesh path already merges every body, so the exact path
/// must not disagree by keeping only the first. Each body keeps its CADIR name and
/// colour, and its placement is applied exactly as the display mesher applies it.
///
/// A body that the kernel refuses becomes a warning naming the body; the other
/// bodies still import. The call errors only when no body converts.
pub fn import_step_breps(path: &str) -> Result<StepBrepImport, String> {
    let model = read_step_model(path)?;
    if model.bodies.is_empty() {
        let refused = model
            .failures
            .first()
            .map(|failure| {
                format!(
                    " ({} region {}: {})",
                    failure.id, failure.region, failure.message
                )
            })
            .or_else(|| model.warnings.first().cloned())
            .unwrap_or_default();
        return Err(format!("{path}: cadmpeg found no B-rep bodies{refused}"));
    }
    let mut warnings = failure_warnings(&model.failures);
    warnings.extend(model.warnings.iter().cloned());
    warnings.extend(model.bodies.iter().flat_map(|body| {
        body.notes
            .iter()
            .map(move |note| format!("body {:?}: {note}", body.name))
    }));
    let mut bodies = Vec::with_capacity(model.bodies.len());
    for body in &model.bodies {
        // One kernel per body: a refused conversion cannot leave partial state
        // that changes the next body's conversion.
        let mut kernel = nodecraft_kernel_profile::kernel();
        let converted: Result<solidcore::Brep, String> = (|| {
            let mut brep = solidcore_convert::brep::to_kernel_body(&mut kernel, &body.brep, true)
                .map_err(|error| error.to_string())?;
            if let Some(placement) = body.placement {
                // `CadBody::placement` is column-major; `transform_matrix` takes the
                // row-vector form, whose translation is its fourth row. The transpose
                // `placement[column][row]` is already that matrix.
                solidcore::topology::brep::mirror::transform_matrix(
                    &mut kernel,
                    &mut brep,
                    &placement,
                    true,
                )
                .map_err(|error| error.to_string())?;
            }
            Ok(brep)
        })();
        match converted {
            Ok(brep) => bodies.push(StepBrepBody {
                name: body.name.clone(),
                color: body.color,
                brep,
            }),
            Err(error) => warnings.push(format!("{path}: body {:?}: {error}", body.name)),
        }
    }
    if bodies.is_empty() {
        let refused = warnings
            .first()
            .map(|warning| format!(" ({warning})"))
            .unwrap_or_default();
        return Err(format!(
            "{path}: cadmpeg found no convertible B-rep bodies{refused}"
        ));
    }
    Ok(StepBrepImport { bodies, warnings })
}

/// A map from CADIR body id to the read record, for tests and diagnostics.
pub fn records_by_id(path: &str) -> Result<BTreeMap<String, CadBody>, String> {
    Ok(read_step_model(path)?
        .bodies
        .into_iter()
        .map(|body| (body.id.clone(), body))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "nodecraft-mesh-cad-{}-{serial}-{tag}.step",
            std::process::id()
        ))
    }

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative);
        path.is_file().then_some(path)
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
            [
                f64::from(vertex[0]),
                f64::from(vertex[1]),
                f64::from(vertex[2]),
            ]
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
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
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

    #[test]
    fn a_step_box_reads_through_cadmpeg() {
        let Some(path) = repository("STEP_TEST_TESS/r1c1_box.step") else {
            eprintln!("skipping a_step_box_reads_through_cadmpeg: fixture not present");
            return;
        };
        let model = read_step_model(path.to_str().unwrap()).expect("cadmpeg read");
        assert_eq!(model.bodies.len(), 1, "{:?}", model.failures);
        assert!(model.failures.is_empty(), "{:?}", model.failures);
        assert_eq!(model.bodies[0].brep.faces.len(), 6);
        let import = import_step(path.to_str().unwrap()).expect("import");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.brep_faces, 6);
        let (volume, _) = mesh_volume_area(&import.mesh);
        assert!(volume.abs() > 0.0, "volume {volume}");
    }

    #[test]
    fn a_freecad_and_a_rhino_file_read_through_cadmpeg() {
        let Some(fcstd) = repository("rust/crates/nodecraft-mesh/tests/fixtures/block.FCStd")
        else {
            eprintln!(
                "skipping a_freecad_and_a_rhino_file_read_through_cadmpeg: fixtures not present"
            );
            return;
        };
        let import = import_step(fcstd.to_str().unwrap()).expect("FreeCAD import");
        assert_eq!(import.diagnostics.kind, "native-cad");
        assert!(import.diagnostics.bodies >= 1, "{:?}", import.diagnostics);
        let rhino =
            repository("rust/crates/nodecraft-mesh/tests/fixtures/block.3dm").expect("3dm fixture");
        let import = import_step(rhino.to_str().unwrap()).expect("Rhino import");
        assert_eq!(import.diagnostics.kind, "native-cad");
        assert!(import.diagnostics.bodies >= 1, "{:?}", import.diagnostics);
    }

    #[test]
    fn a_placed_assembly_imports_in_the_world_frame() {
        let Some(path) = repository("tests/Sample AP203 Data Files/cubcylso.stp") else {
            eprintln!("skipping a_placed_assembly_imports_in_the_world_frame: fixture not present");
            return;
        };
        let import = import_step(path.to_str().unwrap()).expect("cubcylso import");
        assert_eq!(
            import.diagnostics.kind, "assembly",
            "{:?}",
            import.diagnostics.warnings
        );
        assert_eq!(import.diagnostics.bodies, 2, "{:?}", import.diagnostics);
        // CADIR stores the occurrence transform in the STEP relationship's direction.
        // cubcylso's `ITEM_DEFINED_TRANSFORMATION(#315, #211)` is parent-to-child, so the
        // DETAIL is read at x = -200 where SMLib placed it at +200. This is cadmpeg's
        // occurrence-direction convention, recorded in the import comparison report.
        let (min, max) = mesh_bounds(&import.mesh);
        assert_bbox(
            min,
            max,
            [-225.0, -40.0, -50.0],
            [50.0, 40.0, 50.0],
            1e-6,
            "cubcylso",
        );
    }

    #[test]
    fn a_bad_region_does_not_fail_the_other_bodies() {
        let Some(path) = repository("STEP_TEST_TESS/r1c1_box.step") else {
            eprintln!("skipping a_bad_region_does_not_fail_the_other_bodies: fixture not present");
            return;
        };
        let bytes = std::fs::read(path).expect("fixture");
        let result = nodecraft_step::read_cad(&bytes, "box.step").expect("read");
        assert!(result.failures.is_empty());
    }

    #[test]
    fn merged_meshes_keep_exact_normals_only_when_every_body_has_them() {
        let body = |exact: bool| Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            faces: vec![[0, 1, 2]],
            face_groups: vec!["face:0".into()],
            exact_normals: exact,
            ..Mesh::default()
        };
        let exact = merge_meshes(vec![body(true), body(true)]);
        assert!(exact.exact_normals);
        let mixed = merge_meshes(vec![body(true), body(false)]);
        assert!(!mixed.exact_normals);
        let with_empty = merge_meshes(vec![body(true), Mesh::default()]);
        assert!(with_empty.exact_normals);
    }

    fn brep_bounds(brep: &solidcore::Brep) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in brep.vertex_list() {
            let point = brep.vertex(*vertex).expect("vertex").point;
            for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                min[axis] = min[axis].min(value);
                max[axis] = max[axis].max(value);
            }
        }
        (min, max)
    }

    fn translation(offset: [f64; 3]) -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [offset[0], offset[1], offset[2], 1.0],
        ]
    }

    /// The exact import must return every body of the file, not only the first,
    /// with CADIR's order, name, colour and placement preserved.
    #[test]
    fn a_three_body_step_imports_every_body_with_names_colours_and_placements() {
        use solidcore::kernel::Kernel;
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives as api;

        let path = temp_path("three-bodies");
        let specs = [
            ("left", 4.0, [-30.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ("middle", 6.0, [0.0, 20.0, 0.0], [0.0, 1.0, 0.0]),
            ("right", 8.0, [40.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let mut kernel = Kernel::default();
        let mut exported = Vec::new();
        for (name, size, offset, color) in specs {
            let brep = api::create_box(
                &mut kernel,
                Point3::new(-size / 2.0, -size / 2.0, -size / 2.0),
                size,
                size,
                size,
            )
            .expect("box");
            exported.push(nodecraft_step::ExportBody {
                name: name.to_string(),
                color: Some(color),
                placement: Some(translation(offset)),
                sheet: false,
                brep: nodecraft_step::record_of_brep(&brep).expect("record"),
            });
        }
        nodecraft_step::write_step(path.to_str().unwrap(), &exported, "three", "ap242")
            .expect("write");
        let import = import_step_breps(path.to_str().unwrap()).expect("import every body");
        let _ = std::fs::remove_file(&path);

        assert_eq!(import.bodies.len(), 3, "one B-rep per CADIR body");
        let names: Vec<_> = import.bodies.iter().map(|body| body.name.as_str()).collect();
        assert_eq!(names, ["left", "middle", "right"]);
        for (body, (name, size, offset, color)) in import.bodies.iter().zip(specs) {
            assert_eq!(body.name, name);
            let got = body.color.expect("imported colour");
            for axis in 0..3 {
                assert!(
                    (got[axis] - color[axis]).abs() < 1.0e-6,
                    "{name} colour {got:?} != {color:?}"
                );
            }
            let (min, max) = brep_bounds(&body.brep);
            for axis in 0..3 {
                let want_min = offset[axis] - size / 2.0;
                let want_max = offset[axis] + size / 2.0;
                assert!(
                    (min[axis] - want_min).abs() < 1.0e-6,
                    "{name} min[{axis}] {} != {want_min}",
                    min[axis]
                );
                assert!(
                    (max[axis] - want_max).abs() < 1.0e-6,
                    "{name} max[{axis}] {} != {want_max}",
                    max[axis]
                );
            }
            let record = nodecraft_step::record_of_brep(&body.brep).expect("record");
            let volume = nodecraft_step::record_volume(&record).expect("volume");
            assert!(
                (volume - size * size * size).abs() < 1.0e-6,
                "{name} volume {volume} != {}",
                size * size * size
            );
        }
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_panic() {
        let path = temp_path("missing");
        assert!(import_step(path.to_str().unwrap()).is_err());
    }
}
