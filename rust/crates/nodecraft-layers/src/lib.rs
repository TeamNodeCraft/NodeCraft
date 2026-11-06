//! Adapters from the upstream solidcore record/ops/tess layers into NodeCraft's
//! protocol types.
//!
//! This crate is the future display/geometry boundary (see
//! `docs/upstream-solidcore-layers.md`). It deliberately depends on the upstream
//! layer crates instead of the kernel: `solidcore-tess` produces the mesh,
//! `solidcore-ops` owns bodies and workflows, `solidcore-record` is the exact
//! body data.
//!
//! The kernel B-rep <-> record B-rep conversion is available at the current
//! pin through `solidcore-convert::brep`; [`tessellate_kernel`] and
//! [`kernel_body_edit`] use the upstream ops/convert path. The NodeCraft engine
//! still holds kernel B-reps, so these conversions remain the seam until the
//! record-first migration.

use nodecraft_protocol::Mesh;
use solidcore::kernel::Kernel;
use solidcore::topology::FaceId;
use solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};
use solidcore::topology::Brep as KernelBrep;
use solidcore_record::brep::Brep as RecordBrep;
use solidcore_record::Vec3;
use solidcore_tess::MeshTolerance;

pub use solidcore_ops::BooleanOp;

/// A tessellated record in the shape NodeCraft needs: protocol triangle data
/// plus the display-only edge polylines the viewport will draw once the record
/// is the rendering source of truth.
#[derive(Debug)]
pub struct TessDisplay {
    pub mesh: Mesh,
    /// Edge polylines in display space, in the record's edge order.
    pub edge_polylines: Vec<Vec<[f32; 3]>>,
    /// Per edge: the seam / tangent-join flags `solidcore-tess` reports.
    pub seam_edges: Vec<bool>,
    pub smooth_edges: Vec<bool>,
    pub stats: TessStats,
}

/// One display-only simplification level.
#[derive(Debug)]
pub struct TessLevel {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// Maximum distance from the original surface, in mm.
    pub error: f64,
}

/// Build display LOD levels from a tessellation. These carry positions, triangles
/// and an error bound only; face provenance, edges and picking stay on the full
/// mesh.
#[must_use]
pub fn simplify_display(display: &TessDisplay, targets: &[usize]) -> Vec<TessLevel> {
    let positions: Vec<Vec3> = display
        .mesh
        .vertices
        .iter()
        .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    solidcore_tess::simplify::levels(&positions, &display.mesh.faces, targets)
        .into_iter()
        .map(|level| TessLevel {
            positions: level
                .positions
                .iter()
                .map(|p| [p.x as f32, p.y as f32, p.z as f32])
                .collect(),
            triangles: level.triangles,
            error: level.error,
        })
        .collect()
}

/// The tessellation's planning report, reduced to what an application logs.
#[derive(Debug, Default)]
pub struct TessStats {
    pub faces: usize,
    pub triangles: usize,
    pub planar_faces: usize,
    pub structured_faces: usize,
    pub delaunay_faces: usize,
    /// Faces the planner or the evaluator left out of the mesh, with the reason.
    pub unmeshed_faces: Vec<(usize, String)>,
}

/// Plan and evaluate a record's display mesh.
///
/// `tolerance` is the tessellation tolerance; `size` is the body's box diagonal
/// (the planner uses it for its display/edge rules).
pub fn tessellate_record(
    record: &RecordBrep,
    tolerance: &MeshTolerance,
    size: f64,
) -> Result<TessDisplay, String> {
    let (plan, stats) = solidcore_tess::plan(record, tolerance, size)
        .map_err(|error| format!("tessellation plan: {error}"))?;
    let (tess, refused) = plan.evaluate();
    let mut unmeshed_faces: Vec<(usize, String)> = stats
        .unmeshed
        .iter()
        .map(|face| (face.face, face.reason.clone()))
        .collect();
    unmeshed_faces.extend(refused.iter().map(|face| (face.face, face.reason.clone())));
    let edge_polylines: Vec<Vec<[f32; 3]>> = tess
        .edges
        .iter()
        .map(|polyline| polyline.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect())
        .collect();
    let mesh = protocol_mesh(&tess, &edge_polylines);
    Ok(TessDisplay {
        mesh,
        edge_polylines,
        seam_edges: tess.seams.clone(),
        smooth_edges: tess.smooth_edges.clone(),
        stats: TessStats {
            faces: record.faces.len(),
            triangles: tess.triangles.len(),
            planar_faces: stats.planar,
            structured_faces: stats.structured,
            delaunay_faces: stats.delaunay,
            unmeshed_faces,
        },
    })
}

fn protocol_mesh(tess: &solidcore_tess::Mesh, edge_polylines: &[Vec<[f32; 3]>]) -> Mesh {
    let mut mesh = Mesh::default();
    mesh.vertices = tess
        .positions
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    mesh.normals = tess
        .normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32])
        .collect();
    mesh.faces = tess.triangles.clone();
    mesh.face_groups = tess
        .tri_face
        .iter()
        .map(|face| format!("face:{face}"))
        .collect();
    let vertex_key = |v: [f32; 3]| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
    let mut vertex_at = std::collections::HashMap::new();
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        vertex_at.entry(vertex_key(*vertex)).or_insert(index as u32);
    }
    for (edge_index, polyline) in edge_polylines.iter().enumerate() {
        let mut segments = Vec::with_capacity(polyline.len().saturating_sub(1));
        for pair in polyline.windows(2) {
            let (Some(a), Some(b)) = (vertex_at.get(&vertex_key(pair[0])), vertex_at.get(&vertex_key(pair[1]))) else { continue };
            segments.push([*a, *b]);
        }
        if !segments.is_empty() {
            mesh.brep_edges.push(nodecraft_protocol::BrepEdge {
                id: format!("edge:{edge_index}"),
                kind: "edge".into(),
                segments,
            });
        }
    }
    mesh
}

/// The kernel Brep <-> record bridge is now provided by `solidcore-convert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordBridge {
    Available,
}

#[must_use]
pub fn record_bridge() -> RecordBridge {
    RecordBridge::Available
}

fn record_face_indices(order: &solidcore_convert::brep::Order, faces: &[FaceId]) -> Result<Vec<usize>, String> {
    faces
        .iter()
        .map(|face| {
            order
                .faces
                .iter()
                .position(|candidate| candidate == face)
                .ok_or_else(|| format!("face {face:?} is missing from the record order"))
        })
        .collect()
}

fn kernel_body_edit(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    edit: impl FnOnce(&solidcore_ops::Body, &solidcore_convert::brep::Order) -> Result<(), String>,
) -> Result<KernelBrep, String> {
    let (record, _order) = solidcore_convert::brep::to_record(kernel, brep)
        .map_err(|error| format!("kernel to record: {error}"))?;
    let (body, order) = solidcore_ops::Body::from_record(&record, true)
        .map_err(|error| format!("record to ops body: {error}"))?;
    edit(&body, &order)?;
    let (record, _order) = body
        .to_record()
        .map_err(|error| format!("ops body to record: {error}"))?;
    solidcore_convert::brep::to_kernel(kernel, &record, true)
        .map(|(brep, _)| brep)
        .map_err(|error| format!("record to kernel: {error}"))
}

/// Shell an exact kernel B-rep through `solidcore-ops`, opening record face
/// indices directly (the same order `solidcore-tess` reports as `face:N`).
pub fn shell_record_faces(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    open: &[usize],
    thickness: f64,
) -> Result<KernelBrep, String> {
    kernel_body_edit(kernel, brep, |body, order| {
        solidcore_ops::shell(body, order, open, thickness)
            .map_err(|error| format!("solidcore-ops shell: {error}"))
    })
}

/// Offset record faces directly, by record face index.
pub fn offset_faces_record_faces(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    faces: &[usize],
    distance: f64,
) -> Result<KernelBrep, String> {
    kernel_body_edit(kernel, brep, |body, order| {
        solidcore_ops::offset_faces(body, order, faces, distance)
            .map_err(|error| format!("solidcore-ops offset_faces: {error}"))
    })
}

/// Shell an exact kernel B-rep through `solidcore-ops`, opening the given kernel
/// faces. The faces are mapped into the record order first.
pub fn shell_kernel(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    open: &[FaceId],
    thickness: f64,
) -> Result<KernelBrep, String> {
    let open = record_face_indices(&solidcore_convert::brep::to_record(kernel, brep).map_err(|error| error.to_string())?.1, open)?;
    kernel_body_edit(kernel, brep, |body, order| {
        solidcore_ops::shell(body, order, &open, thickness)
            .map_err(|error| format!("solidcore-ops shell: {error}"))
    })
}

/// Offset exact kernel faces through `solidcore-ops`, by kernel face ids.
pub fn offset_faces_kernel(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    faces: &[FaceId],
    distance: f64,
) -> Result<KernelBrep, String> {
    let faces = record_face_indices(&solidcore_convert::brep::to_record(kernel, brep).map_err(|error| error.to_string())?.1, faces)?;
    kernel_body_edit(kernel, brep, |body, order| {
        solidcore_ops::offset_faces(body, order, &faces, distance)
            .map_err(|error| format!("solidcore-ops offset_faces: {error}"))
    })
}

/// Boolean of two record bodies through the upstream ops layer.
pub fn boolean_records(
    a: &RecordBrep,
    b: &RecordBrep,
    op: solidcore_ops::BooleanOp,
) -> Result<Option<RecordBrep>, String> {
    let (a, _) = solidcore_ops::Body::from_record(a, true)
        .map_err(|error| format!("record to ops body A: {error}"))?;
    let (b, _) = solidcore_ops::Body::from_record(b, true)
        .map_err(|error| format!("record to ops body B: {error}"))?;
    solidcore_ops::boolean::boolean(a, b, op)
        .map_err(|error| format!("solidcore-ops boolean: {error}"))?
        .map(|body| {
            body.to_record()
                .map(|(record, _)| record)
                .map_err(|error| format!("ops boolean to record: {error}"))
        })
        .transpose()
}

/// Union of record bodies through the upstream ops layer.
pub fn union_records(records: &[RecordBrep]) -> Result<Option<RecordBrep>, String> {
    if records.is_empty() {
        return Ok(None);
    }
    let mut bodies = Vec::with_capacity(records.len());
    for record in records {
        let (body, _) = solidcore_ops::Body::from_record(record, true)
            .map_err(|error| format!("record to ops body: {error}"))?;
        bodies.push(body);
    }
    let body = solidcore_ops::boolean::union_all(bodies)
        .map_err(|error| format!("solidcore-ops union: {error}"))?;
    body.to_record()
        .map(|(record, _)| Some(record))
        .map_err(|error| format!("ops union to record: {error}"))
}

/// Union kernel B-reps through records + ops, returning a kernel B-rep.
pub fn union_breps_ops(
    kernel: &mut Kernel,
    breps: &[KernelBrep],
) -> Result<Option<KernelBrep>, String> {
    if breps.is_empty() {
        return Ok(None);
    }
    let mut records = Vec::with_capacity(breps.len());
    for brep in breps {
        let mut copy = brep
            .clone_standing_for_self()
            .map_err(|error| format!("union copy: {error}"))?;
        let (record, _) = solidcore_convert::brep::to_record(kernel, &mut copy)
            .map_err(|error| format!("kernel to record: {error}"))?;
        records.push(record);
    }
    let Some(record) = union_records(&records)? else {
        return Ok(None);
    };
    solidcore_convert::brep::to_kernel(kernel, &record, true)
        .map(|(brep, _)| Some(brep))
        .map_err(|error| format!("union record to kernel: {error}"))
}

/// Boolean two kernel B-reps through record + ops, returning a kernel B-rep.
pub fn boolean_breps_ops(
    kernel: &mut Kernel,
    a: &KernelBrep,
    b: &KernelBrep,
    op: solidcore_ops::BooleanOp,
) -> Result<Option<KernelBrep>, String> {
    let mut a_copy = a
        .clone_standing_for_self()
        .map_err(|error| format!("boolean copy A: {error}"))?;
    let mut b_copy = b
        .clone_standing_for_self()
        .map_err(|error| format!("boolean copy B: {error}"))?;
    let (a_record, _) = solidcore_convert::brep::to_record(kernel, &mut a_copy)
        .map_err(|error| format!("kernel A to record: {error}"))?;
    let (b_record, _) = solidcore_convert::brep::to_record(kernel, &mut b_copy)
        .map_err(|error| format!("kernel B to record: {error}"))?;
    let Some(result) = boolean_records(&a_record, &b_record, op)? else {
        return Ok(None);
    };
    solidcore_convert::brep::to_kernel(kernel, &result, true)
        .map(|(brep, _)| Some(brep))
        .map_err(|error| format!("boolean record to kernel: {error}"))
}

/// Exact volume of a kernel B-rep through the upstream ops body.
pub fn kernel_volume(kernel: &mut Kernel, brep: &KernelBrep) -> Result<f64, String> {
    let mut brep = brep
        .clone_standing_for_self()
        .map_err(|error| format!("volume copy: {error}"))?;
    let (record, _) = solidcore_convert::brep::to_record(kernel, &mut brep)
        .map_err(|error| format!("kernel to record: {error}"))?;
    let (body, _) = solidcore_ops::Body::from_record(&record, true)
        .map_err(|error| format!("record to ops body: {error}"))?;
    body.volume(1.0e-6)
        .map_err(|error| format!("ops volume: {error}"))
}

/// Convert an ops body to a kernel Brep through its record.
pub fn body_to_kernel(kernel: &mut Kernel, body: &solidcore_ops::Body) -> Result<KernelBrep, String> {
    let (record, _order) = body
        .to_record()
        .map_err(|error| format!("ops body to record: {error}"))?;
    solidcore_convert::brep::to_kernel(kernel, &record, true)
        .map(|(brep, _)| brep)
        .map_err(|error| format!("record to kernel: {error}"))
}

/// Convert a kernel B-rep to the record form, then plan and evaluate its display
/// mesh with the single upstream mesher.
pub fn tessellate_kernel(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    tolerance: f64,
) -> Result<TessDisplay, String> {
    let extent = calculate_bounding_box(
        kernel,
        &*brep,
        BoundingBoxOptions {
            approx: false,
            tight: true,
            expand_by_zone_tol_3d: false,
        },
    )
    .map_err(|error| format!("kernel bounding box: {error}"))?;
    let mut diagonal = 0.0;
    for axis in 0..3 {
        diagonal += (extent.max_at(axis) - extent.min_at(axis)).powi(2);
    }
    let size = diagonal.sqrt().max(1.0);
    let (record, _order) = solidcore_convert::brep::to_record(kernel, brep)
        .map_err(|error| format!("kernel to record: {error}"))?;
    tessellate_record(&record, &MeshTolerance::display(tolerance.max(1.0e-5)), size)
}
