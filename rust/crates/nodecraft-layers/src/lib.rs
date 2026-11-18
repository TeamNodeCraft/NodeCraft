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
use solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};
use solidcore::kernel::Kernel;
use solidcore::topology::Brep as KernelBrep;
use solidcore::topology::FaceId;
use solidcore_record::Vec3;
use solidcore_record::brep::Brep as RecordBrep;

/// Test-visible proof that LOD bounds do not tessellate.
///
/// `body_bounds` must measure the B-rep, never build a display plan. A test
/// resets this thread's counter, measures bounds, and asserts the counter is
/// still zero. Thread-local because display planning itself runs on worker
/// threads and unrelated tests may plan concurrently.
pub mod plan_stats {
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<u64> = const { Cell::new(0) };
    }

    /// Reset this thread's plan/tessellation counter.
    pub fn reset() {
        CALLS.with(|value| value.set(0));
    }

    /// Plans or tessellations started on this thread since [`reset`].
    #[must_use]
    pub fn calls() -> u64 {
        CALLS.with(Cell::get)
    }

    pub(super) fn note() {
        CALLS.with(|value| value.set(value.get() + 1));
    }
}

/// Re-export the plan types used by the app/GPU adapters so application code
/// stays behind the NodeCraft layers boundary.
pub mod tess {
    pub use solidcore_tess::{FacePlan, TessPlan};
}

pub use solidcore_ops::BooleanOp;
pub use solidcore_tess::MeshTolerance;

fn check_body(body: &solidcore_ops::Body) -> Result<(), String> {
    if !cfg!(any(test, debug_assertions))
        && std::env::var_os("NODECRAFT_VALIDATE_RESULTS").is_none()
    {
        return Ok(());
    }
    let report = body
        .check(solidcore_ops::CheckLevel::Geometry)
        .map_err(|error| format!("solidcore check: {error}"))?;
    if report.is_valid() {
        Ok(())
    } else {
        Err(format!("result failed solidcore check: {report}"))
    }
}

/// Configure the process-wide `solidcore-ops` session with the same upstream
/// fix profile the application applies to its own kernels. Ops operations
/// (Boolean, shell, offset, blend) otherwise run with the upstream default.
pub fn configure_session_fixes(fixes: &[&str]) -> Result<(), String> {
    solidcore_ops::session::set_fixes(fixes.iter().copied())
        .map_err(|error| format!("solidcore-ops fixes: {error}"))
}

/// How one plan edge is drawn and picked.
///
/// The bits are the one per-edge class shared by the GPU ribbon, the edge pick
/// pass and the CPU fallback. They are derived once from the tessellator's
/// flags (SolidCore's rule: a seam is `seams[i] && !sharp_edges[i]`) and then
/// uploaded with the edge. A setting only changes the uniform mask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PlanEdgeClass(u32);

impl PlanEdgeClass {
    /// An ordinary sharp or boundary edge.
    pub const NORMAL: Self = Self(0);
    /// A seam between two pieces of one smooth surface (hidden by default).
    pub const SEAM: Self = Self(1 << 0);
    /// A display-only tangent join. Stored now so a later setting can use the
    /// same mask without re-plumbing; it is drawn today.
    pub const SMOOTH: Self = Self(1 << 1);

    /// The class from the tessellator's per-edge flags. A certified crease
    /// (`sharp`) overrides a coarse seam fit, exactly as `Mesh::edge_visible`
    /// does upstream.
    #[must_use]
    pub const fn from_tess_flags(seam: bool, smooth: bool, sharp: bool) -> Self {
        let mut bits = 0;
        if seam && !sharp {
            bits |= Self::SEAM.0;
        }
        if smooth {
            bits |= Self::SMOOTH.0;
        }
        Self(bits)
    }

    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn is_seam(self) -> bool {
        self.0 & Self::SEAM.0 != 0
    }

    #[must_use]
    pub const fn is_smooth(self) -> bool {
        self.0 & Self::SMOOTH.0 != 0
    }

    /// Whether this edge is drawn and picked under the seam setting.
    #[must_use]
    pub const fn visible(self, show_seam_edges: bool) -> bool {
        !self.is_seam() || show_seam_edges
    }

    /// The uniform mask the shaders test against this class.
    #[must_use]
    pub const fn mask(show_seam_edges: bool) -> u32 {
        if show_seam_edges { Self::SEAM.0 } else { 0 }
    }
}

/// A tessellated record in the shape NodeCraft needs: protocol triangle data
/// plus the display-only edge polylines the viewport will draw once the record
/// is the rendering source of truth.
#[derive(Debug)]
pub struct TessDisplay {
    /// The evaluated plan's source. Kept so a GPU can evaluate the same
    /// triangulation instead of only consuming the CPU mesh.
    pub plan: std::sync::Arc<solidcore_tess::TessPlan>,
    pub mesh: Mesh,
    /// Edge polylines in display space, in the record's edge order.
    pub edge_polylines: Vec<Vec<[f32; 3]>>,
    /// Per edge: the seam / tangent-join flags `solidcore-tess` reports.
    pub seam_edges: Vec<bool>,
    pub smooth_edges: Vec<bool>,
    /// Protected creases, which override a coarse seam fit.
    pub sharp_edges: Vec<bool>,
    /// The one class per plan edge, derived from the flags above.
    pub edge_classes: Vec<PlanEdgeClass>,
    pub stats: TessStats,
    /// Display-edge id by record edge index, after the kernel-edge remap.
    pub plan_edge_ids: Vec<Option<String>>,
}

/// One display-only simplification level.
#[derive(Debug)]
pub struct TessLevel {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// Maximum distance from the original surface, in mm.
    pub error: f64,
}

/// Build display LOD levels directly from a protocol mesh.
#[must_use]
pub fn simplify_mesh(mesh: &Mesh, targets: &[usize]) -> Vec<TessLevel> {
    let positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    solidcore_tess::simplify::levels(&positions, &mesh.faces, targets)
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
#[derive(Clone, Debug, Default)]
pub struct TessStats {
    pub faces: usize,
    pub triangles: usize,
    pub planar_faces: usize,
    pub structured_faces: usize,
    pub delaunay_faces: usize,
    /// Faces the planner or the evaluator left out of the mesh, with the reason.
    pub unmeshed_faces: Vec<(usize, String)>,
    pub lenient_faces: usize,
    pub over_tolerance_faces: usize,
    pub unsampled_edges: usize,
    pub capped_edges: usize,
    pub kept_strict_edges: usize,
    pub averaged_poles: usize,
    /// Human-readable planner defects, capped for display/logging.
    pub diagnostics: Vec<String>,
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
    let plan = std::sync::Arc::new(plan);
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
        .map(|polyline| {
            polyline
                .iter()
                .map(|p| [p.x as f32, p.y as f32, p.z as f32])
                .collect()
        })
        .collect();
    let edge_classes: Vec<PlanEdgeClass> = (0..plan.edges.len())
        .map(|index| {
            PlanEdgeClass::from_tess_flags(
                tess.seams.get(index).copied().unwrap_or(false),
                tess.smooth_edges.get(index).copied().unwrap_or(false),
                tess.sharp_edges.get(index).copied().unwrap_or(false),
            )
        })
        .collect();
    let mesh = protocol_mesh(&tess, &edge_polylines, &edge_classes);
    let mut diagnostics = Vec::new();
    for face in &stats.lenient {
        diagnostics.push(format!("face {} lenient: {}", face.face, face.reason));
    }
    for face in &unmeshed_faces {
        diagnostics.push(format!("face {} unmeshed: {}", face.0, face.1));
    }
    for edge in &stats.unsampled_edges {
        diagnostics.push(format!("edge {} unsampled: {}", edge.edge, edge.reason));
    }
    for edge in &stats.edges_capped {
        diagnostics.push(format!(
            "edge {} capped {:?}: sag {:.3e}, excess {:.2}, tolerance {:.3e}{}",
            edge.edge,
            edge.cap,
            edge.sag,
            edge.excess,
            edge.tol,
            if edge.within { "" } else { " (outside tolerance)" }
        ));
    }
    for edge in &stats.kept_strict {
        diagnostics.push(format!("edge {} kept strict: {}", edge.edge, edge.reason));
    }
    let mut plan_edge_ids = vec![None; plan.edges.len()];
    for edge in &mesh.brep_edges {
        if let Some(index) = edge
            .id
            .strip_prefix("edge:")
            .and_then(|index| index.parse::<usize>().ok())
        {
            if let Some(slot) = plan_edge_ids.get_mut(index) {
                *slot = Some(edge.id.clone());
            }
        }
    }
    for face in &stats.over_tolerance {
        diagnostics.push(format!(
            "face {} over tolerance: sag {:.3e}, tolerance {:.3e}, caps {:?}{}",
            face.face,
            face.sag,
            face.tol,
            face.caps,
            if face.within { "" } else { " (outside tolerance)" }
        ));
    }
    for pole in &stats.averaged_poles {
        diagnostics.push(format!(
            "face {} averaged pole at {:.3e},{:.3e},{:.3e}: {}",
            pole.face, pole.point.x, pole.point.y, pole.point.z, pole.reason
        ));
    }
    Ok(TessDisplay {
        plan,
        mesh,
        edge_polylines,
        plan_edge_ids,
        seam_edges: tess.seams.clone(),
        smooth_edges: tess.smooth_edges.clone(),
        sharp_edges: tess.sharp_edges.clone(),
        edge_classes,
        stats: TessStats {
            faces: record.faces.len(),
            triangles: tess.triangles.len(),
            planar_faces: stats.planar,
            structured_faces: stats.structured,
            delaunay_faces: stats.delaunay,
            unmeshed_faces,
            lenient_faces: stats.lenient.len(),
            over_tolerance_faces: stats.over_tolerance.len(),
            unsampled_edges: stats.unsampled_edges.len(),
            capped_edges: stats.edges_capped.len(),
            kept_strict_edges: stats.kept_strict.len(),
            averaged_poles: stats.averaged_poles.len(),
            diagnostics,
        },
    })
}

fn protocol_mesh(
    tess: &solidcore_tess::Mesh,
    edge_polylines: &[Vec<[f32; 3]>],
    edge_classes: &[PlanEdgeClass],
) -> Mesh {
    let mut mesh = Mesh::default();
    // `solidcore-tess` evaluated these normals from the surface, split per
    // B-rep face and per incident triangle at a pole. They are the shading
    // source of truth; a later crease-split would average them back toward a
    // pole's axis.
    mesh.exact_normals = true;
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
            let (Some(a), Some(b)) = (
                vertex_at.get(&vertex_key(pair[0])),
                vertex_at.get(&vertex_key(pair[1])),
            ) else {
                continue;
            };
            segments.push([*a, *b]);
        }
        if !segments.is_empty() {
            mesh.brep_edges.push(nodecraft_protocol::BrepEdge {
                id: format!("edge:{edge_index}"),
                kind: "edge".into(),
                segments,
                class: edge_classes.get(edge_index).copied().unwrap_or_default().bits(),
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

fn record_face_indices(
    order: &solidcore_convert::brep::Order,
    faces: &[FaceId],
) -> Result<Vec<usize>, String> {
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
    check_body(&body)?;
    let (record, _order) = body
        .to_record()
        .map_err(|error| format!("ops body to record: {error}"))?;
    solidcore_convert::brep::to_kernel(kernel, &record, true)
        .map(|(brep, _)| brep)
        .map_err(|error| format!("record to kernel: {error}"))
}

/// Resolve canonical record face indices to kernel `FaceId`s.
pub fn record_face_ids(kernel: &mut Kernel, brep: &mut KernelBrep, indices: &[usize]) -> Result<Vec<FaceId>, String> {
    let order = solidcore_convert::brep::to_record(kernel, brep)
        .map_err(|error| format!("kernel to record: {error}"))?
        .1;
    indices
        .iter()
        .map(|index| order.faces.get(*index).copied().ok_or_else(|| format!("record face {index} is out of range")))
        .collect()
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
    let open = record_face_indices(
        &solidcore_convert::brep::to_record(kernel, brep)
            .map_err(|error| error.to_string())?
            .1,
        open,
    )?;
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
    let faces = record_face_indices(
        &solidcore_convert::brep::to_record(kernel, brep)
            .map_err(|error| error.to_string())?
            .1,
        faces,
    )?;
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
            check_body(&body)?;
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
    check_body(&body)?;
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
pub fn body_to_kernel(
    kernel: &mut Kernel,
    body: &solidcore_ops::Body,
) -> Result<KernelBrep, String> {
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
    let size = kernel_body_size(kernel, brep)?;
    tessellate_kernel_with(
        kernel,
        brep,
        MeshTolerance::display(tolerance.max(1.0e-5)),
        size,
    )
}

/// Tessellate a kernel B-rep at `solidcore_tess::preview` density (a provisional
/// display plan). The preview is never finer than the normal display tolerance,
/// so small bodies keep their normal quality instead of gaining triangles.
pub fn tessellate_kernel_preview(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    detail: u8,
) -> Result<TessDisplay, String> {
    let size = kernel_body_size(kernel, brep)?;
    let preview = solidcore_tess::preview::tolerance(size, usize::from(detail));
    // The preview chord is based on body size, so for small bodies it can be
    // finer than the normal display plan; never add triangles. Far previews also
    // relax the facet angle, which is the dominant lever for curved bodies.
    let angle = match detail {
        0..=2 => Some(30.0_f64.to_radians()),
        3..=4 => Some(25.0_f64.to_radians()),
        5..=6 => Some(20.0_f64.to_radians()),
        _ => Some(15.0_f64.to_radians()),
    };
    let display = MeshTolerance {
        chord: preview.chord.max(1.0e-5),
        angle,
    };
    tessellate_kernel_with(kernel, brep, display, size)
}

fn kernel_body_size(kernel: &mut Kernel, brep: &KernelBrep) -> Result<f64, String> {
    // Display only needs a conservative extent for choosing density. The tight
    // exact box evaluates every edge and face and can take tens of seconds on a
    // large imported body; the approximate path falls back to vertices above
    // the kernel's size thresholds and is never smaller than the body.
    let extent = calculate_bounding_box(
        kernel,
        brep,
        BoundingBoxOptions {
            approx: true,
            tight: false,
            expand_by_zone_tol_3d: true,
        },
    )
    .map_err(|error| format!("kernel bounding box: {error}"))?;
    let mut diagonal = 0.0;
    for axis in 0..3 {
        diagonal += (extent.max_at(axis) - extent.min_at(axis)).powi(2);
    }
    Ok(diagonal.sqrt().max(1.0))
}

/// A display plan without evaluating it into a CPU mesh. GPU callers consume the
/// plan; only edge polylines/ids are materialised for picking and overlays.
#[derive(Clone, Debug)]
pub struct TessPlanData {
    pub plan: std::sync::Arc<solidcore_tess::TessPlan>,
    pub plan_edge_ids: Vec<Option<String>>,
    pub edge_polylines: Vec<Vec<[f32; 3]>>,
    /// The one class per plan edge, from the same tessellation.
    pub edge_classes: Vec<PlanEdgeClass>,
    pub stats: TessStats,
}

/// The record edge index `id` names, if it names one.
fn edge_index(id: &str) -> Option<usize> {
    id.strip_prefix("edge:").and_then(|index| index.parse::<usize>().ok())
}

/// Derive each plan edge's class once, at the tessellation source. Seams come
/// from the kernel's own seam test (the low-level record path has no evaluated
/// `Mesh` to read `seams` from); smooth and sharp flags are carried when the
/// tessellator supplied them. The GPU never recomputes this.
fn kernel_edge_classes(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    order: &solidcore_convert::brep::Order,
    edge_count: usize,
    smooth: &[bool],
    sharp: &[bool],
) -> Vec<PlanEdgeClass> {
    (0..edge_count)
        .map(|index| {
            let seam = order.edges.get(index).is_some_and(|edge| {
                solidcore::topology::edge::is_seam(kernel, brep, *edge).unwrap_or(false)
            });
            PlanEdgeClass::from_tess_flags(
                seam,
                smooth.get(index).copied().unwrap_or(false),
                sharp.get(index).copied().unwrap_or(false),
            )
        })
        .collect()
}

/// Put the derived classes on a display: the per-edge list, the existing flag
/// vectors (one source), and every protocol B-rep edge the CPU path sees.
fn apply_edge_classes(display: &mut TessDisplay, edge_classes: Vec<PlanEdgeClass>) {
    display.seam_edges = edge_classes.iter().map(|class| class.is_seam()).collect();
    display.smooth_edges = edge_classes.iter().map(|class| class.is_smooth()).collect();
    for edge in &mut display.mesh.brep_edges {
        if let Some(index) = edge_index(&edge.id) {
            if let Some(class) = edge_classes.get(index) {
                edge.class = class.bits();
            }
        }
    }
    display.edge_classes = edge_classes;
}

/// Recover the class of each plan edge from a protocol mesh's B-rep edges.
/// Engine-side cache hits use this to keep the one source after the tessellation
/// value has travelled through the mesh.
#[must_use]
pub fn plan_edge_classes_from_brep_edges(
    plan: &solidcore_tess::TessPlan,
    edge_ids: &[Option<String>],
    edges: &[nodecraft_protocol::BrepEdge],
) -> Vec<PlanEdgeClass> {
    let mut by_id = std::collections::HashMap::new();
    for (index, id) in edge_ids.iter().enumerate() {
        if let Some(id) = id {
            by_id.entry(id.as_str()).or_insert(index);
        }
    }
    let mut classes = vec![PlanEdgeClass::NORMAL; plan.edges.len()];
    for edge in edges {
        let index = by_id
            .get(edge.id.as_str())
            .copied()
            .or_else(|| edge_index(&edge.id));
        if let Some(index) = index {
            if let Some(slot) = classes.get_mut(index) {
                *slot = PlanEdgeClass::from_bits(edge.class);
            }
        }
    }
    classes
}

/// Plan a kernel B-rep without evaluating it. `preview_detail` selects the
/// provisional display density; `None` uses the normal display tolerance.
pub fn plan_kernel(
    kernel: &mut Kernel,
    brep: &KernelBrep,
    tolerance: f64,
    preview_detail: Option<u8>,
) -> Result<TessPlanData, String> {
    plan_stats::note();
    let size = kernel_body_size(kernel, brep)?;
    let display = match preview_detail {
        None => MeshTolerance::display(tolerance.max(1.0e-5)),
        Some(detail) => {
            let preview = solidcore_tess::preview::tolerance(size, usize::from(detail));
            let angle = match detail {
                0..=2 => Some(30.0_f64.to_radians()),
                3..=4 => Some(25.0_f64.to_radians()),
                5..=6 => Some(20.0_f64.to_radians()),
                _ => Some(15.0_f64.to_radians()),
            };
            MeshTolerance { chord: preview.chord.max(1.0e-5), angle }
        }
    };
    let mut copy = brep
        .clone_standing_for_self()
        .map_err(|error| format!("display copy: {error}"))?;
    let (record, order) = solidcore_convert::brep::to_record(kernel, &mut copy)
        .map_err(|error| format!("kernel to record: {error}"))?;
    let (plan, planned) = solidcore_tess::plan(&record, &display, size)
        .map_err(|error| format!("tessellation plan: {error}"))?;
    let plan = std::sync::Arc::new(plan);
    let mut stats = TessStats {
        faces: plan.faces.len(),
        triangles: plan.faces.iter().map(|face| face.triangles.len()).sum(),
        planar_faces: planned.planar,
        structured_faces: planned.structured,
        delaunay_faces: planned.delaunay,
        lenient_faces: planned.lenient.len(),
        over_tolerance_faces: planned.over_tolerance.len(),
        unsampled_edges: planned.unsampled_edges.len(),
        capped_edges: planned.edges_capped.len(),
        kept_strict_edges: planned.kept_strict.len(),
        averaged_poles: planned.averaged_poles.len(),
        unmeshed_faces: planned.unmeshed.iter().map(|face| (face.face, face.reason.clone())).collect(),
        ..TessStats::default()
    };
    for face in &planned.lenient {
        stats.diagnostics.push(format!("face {} lenient: {}", face.face, face.reason));
    }
    for face in &planned.unmeshed {
        stats.diagnostics.push(format!("face {} unmeshed: {}", face.face, face.reason));
    }
    let edge_list = copy.edge_list();
    let edge_ordinals: std::collections::HashMap<_, _> = edge_list
        .iter()
        .enumerate()
        .map(|(ordinal, edge)| (edge.clone(), ordinal))
        .collect();
    let mut plan_edge_ids = vec![None; plan.edges.len()];
    for (index, kernel_edge) in order.edges.iter().enumerate() {
        let Some(&ordinal) = edge_ordinals.get(kernel_edge) else { continue };
        if let Some(slot) = plan_edge_ids.get_mut(index) {
            *slot = Some(format!("edge:{ordinal}"));
        }
    }
    let edge_polylines = plan
        .edges
        .iter()
        .map(|polyline| polyline.iter().map(|point| [point.x as f32, point.y as f32, point.z as f32]).collect())
        .collect();
    let edge_classes = kernel_edge_classes(kernel, &mut copy, &order, plan.edges.len(), &[], &[]);
    Ok(TessPlanData { plan, plan_edge_ids, edge_polylines, edge_classes, stats })
}

fn tessellate_kernel_with(
    kernel: &mut Kernel,
    brep: &mut KernelBrep,
    tolerance: MeshTolerance,
    size: f64,
) -> Result<TessDisplay, String> {
    plan_stats::note();
    let (record, order) = solidcore_convert::brep::to_record(kernel, brep)
        .map_err(|error| format!("kernel to record: {error}"))?;
    let mut display = tessellate_record(&record, &tolerance, size)?;
    let classes = kernel_edge_classes(
        kernel,
        brep,
        &order,
        display.plan.edges.len(),
        &display.smooth_edges,
        &display.sharp_edges,
    );
    apply_edge_classes(&mut display, classes);
    // `protocol_mesh` keeps the record edge index in each display edge's id,
    // and skips empty polylines. Index into `order.edges` instead of zipping the
    // shorter display list against it; zipping shifted every id after the first
    // skipped edge, so saved ids named the wrong BRep edge.
    let edge_list = brep.edge_list();
    let edge_ordinals: std::collections::HashMap<_, _> = edge_list
        .iter()
        .enumerate()
        .map(|(ordinal, edge)| (edge.clone(), ordinal))
        .collect();
    let mut plan_edge_ids = vec![None; display.plan.edges.len()];
    for display_edge in &mut display.mesh.brep_edges {
        let Some(record_index) = display_edge
            .id
            .strip_prefix("edge:")
            .and_then(|index| index.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(kernel_edge) = order.edges.get(record_index) else {
            continue;
        };
        if let Some(&ordinal) = edge_ordinals.get(kernel_edge) {
            display_edge.id = format!("edge:{ordinal}");
            if let Some(slot) = plan_edge_ids.get_mut(record_index) {
                *slot = Some(display_edge.id.clone());
            }
        }
    }
    display.plan_edge_ids = plan_edge_ids;
    Ok(display)
}

/// Derive the display mesh from a cached plan without re-planning. `edge_ids`
/// comes from the original [`TessDisplay::plan_edge_ids`] so edge selections
/// keep their kernel ids after a plan-cache hit.
#[must_use]
pub fn evaluate_plan(plan: &std::sync::Arc<solidcore_tess::TessPlan>, edge_ids: &[Option<String>]) -> TessDisplay {
    evaluate_plan_with_classes(plan, edge_ids, &[])
}

/// [`evaluate_plan`] carrying the per-edge classes derived at plan time. The
/// classes come from the same tessellation that produced the plan, so a cache
/// hit never recomputes them.
#[must_use]
pub fn evaluate_plan_with_classes(
    plan: &std::sync::Arc<solidcore_tess::TessPlan>,
    edge_ids: &[Option<String>],
    edge_classes: &[PlanEdgeClass],
) -> TessDisplay {
    let (tess, refused) = plan.evaluate();
    let edge_polylines: Vec<Vec<[f32; 3]>> = tess
        .edges
        .iter()
        .map(|polyline| {
            polyline
                .iter()
                .map(|p| [p.x as f32, p.y as f32, p.z as f32])
                .collect()
        })
        .collect();
    let classes: Vec<PlanEdgeClass> = (0..plan.edges.len())
        .map(|index| edge_classes.get(index).copied().unwrap_or_default())
        .collect();
    let mut mesh = protocol_mesh(&tess, &edge_polylines, &classes);
    for edge in &mut mesh.brep_edges {
        if let Some(index) = edge_index(&edge.id) {
            if let Some(Some(id)) = edge_ids.get(index) {
                edge.id.clone_from(id);
            }
        }
    }
    let mut unmeshed = Vec::new();
    unmeshed.extend(refused.into_iter().map(|face| (face.face, face.reason)));
    TessDisplay {
        plan: std::sync::Arc::clone(plan),
        mesh,
        edge_polylines,
        plan_edge_ids: edge_ids.to_vec(),
        seam_edges: classes.iter().map(|class| class.is_seam()).collect(),
        smooth_edges: classes.iter().map(|class| class.is_smooth()).collect(),
        sharp_edges: tess.sharp_edges.clone(),
        edge_classes: classes,
        stats: TessStats {
            faces: plan.faces.len(),
            triangles: tess.triangles.len(),
            unmeshed_faces: unmeshed,
            ..TessStats::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;
    use solidcore::query::api_queries;

    #[test]
    fn cached_plan_evaluation_preserves_kernel_edge_ids() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep =
            api::create_box(&mut kernel, Point3::new(-5.0, -5.0, -5.0), 10.0, 10.0, 10.0)
                .expect("box");
        let display = tessellate_kernel(&mut kernel, &mut brep, 0.1).expect("tessellate");
        let evaluated = evaluate_plan(&display.plan, &display.plan_edge_ids);
        assert!(display.mesh.exact_normals);
        assert!(evaluated.mesh.exact_normals);
        let ids = |mesh: &nodecraft_protocol::Mesh| {
            mesh.brep_edges.iter().map(|edge| edge.id.clone()).collect::<Vec<_>>()
        };
        assert_eq!(ids(&evaluated.mesh), ids(&display.mesh));
    }

    #[test]
    fn display_edge_ids_name_the_kernel_edge_they_lie_on() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep =
            api::create_box(&mut kernel, Point3::new(-5.0, -5.0, -5.0), 10.0, 10.0, 10.0)
                .expect("box");
        let display = tessellate_kernel(&mut kernel, &mut brep, 0.1).expect("tessellate");
        let edges = brep.edge_list();
        assert!(!display.mesh.brep_edges.is_empty());
        for display_edge in &display.mesh.brep_edges {
            let segment = display_edge.segments.first().expect("segment");
            let a = display.mesh.vertices[segment[0] as usize];
            let b = display.mesh.vertices[segment[1] as usize];
            let point = Point3::new(
                (f64::from(a[0]) + f64::from(b[0])) * 0.5,
                (f64::from(a[1]) + f64::from(b[1])) * 0.5,
                (f64::from(a[2]) + f64::from(b[2])) * 0.5,
            );
            let nearest = edges
                .iter()
                .min_by(|a, b| {
                    let da = api_queries::edge_closest_point(&kernel, &brep, **a, point)
                        .expect("closest")
                        .closest
                        .distance;
                    let db = api_queries::edge_closest_point(&kernel, &brep, **b, point)
                        .expect("closest")
                        .closest
                        .distance;
                    da.total_cmp(&db)
                })
                .expect("nearest");
            let ordinal = edges
                .iter()
                .position(|edge| *edge == *nearest)
                .expect("ordinal");
            assert_eq!(display_edge.id, format!("edge:{ordinal}"));
        }
    }

    #[test]
    fn a_certified_crease_overrides_a_seam_fit() {
        assert!(PlanEdgeClass::from_tess_flags(true, false, false).is_seam());
        assert!(!PlanEdgeClass::from_tess_flags(true, false, true).is_seam());
        assert!(PlanEdgeClass::from_tess_flags(false, true, false).is_smooth());
    }

    #[test]
    fn primitive_seams_are_classified_from_the_kernel_flags() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut cone = api::create_cone(&mut kernel, Point3::new(0.0, 0.0, -20.0), 20.0, 0.0, 40.0)
            .expect("cone");
        let mut cylinder = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, -20.0), 20.0, 40.0)
            .expect("cylinder");
        let mut sphere = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0)
            .expect("sphere");
        let mut boxed =
            api::create_box(&mut kernel, Point3::new(-5.0, -5.0, -5.0), 10.0, 10.0, 10.0)
                .expect("box");
        for (name, brep) in [
            ("cone", &mut cone),
            ("cylinder", &mut cylinder),
            ("sphere", &mut sphere),
            ("box", &mut boxed),
        ] {
            let display = tessellate_kernel_preview(&mut kernel, brep, 3).expect(name);
            assert_eq!(
                display.edge_classes.len(),
                display.plan.edges.len(),
                "{name}: one class per plan edge",
            );
            let seams: Vec<usize> = display
                .edge_classes
                .iter()
                .enumerate()
                .filter(|(_, class)| class.is_seam())
                .map(|(index, _)| index)
                .collect();
            if name == "box" {
                assert!(seams.is_empty(), "a box has no seam of a smooth surface: {seams:?}");
            } else {
                assert_eq!(
                    seams.len(),
                    1,
                    "{name}: exactly one wall seam, got {seams:?}",
                );
            }
            for (index, class) in display.edge_classes.iter().enumerate() {
                if class.is_seam() {
                    assert!(!class.visible(false), "{name}: seam {index} is hidden by default");
                    assert!(class.visible(true), "{name}: seam {index} shows when asked");
                } else {
                    assert!(class.visible(false), "{name}: ordinary edge {index} stays drawn");
                }
            }
            let mesh_seams = display
                .mesh
                .brep_edges
                .iter()
                .filter(|edge| edge.class & PlanEdgeClass::SEAM.bits() != 0)
                .count();
            assert_eq!(
                mesh_seams,
                seams.len(),
                "{name}: the CPU mesh carries the same seam classes",
            );
            assert_eq!(
                display.edge_polylines.len(),
                display.edge_classes.len(),
                "{name}: every plan edge has its polyline and class",
            );
        }
        // The GPU plan-only path derives the same classes without evaluating a
        // CPU mesh.
        let planned = plan_kernel(&mut kernel, &cone, 0.1, Some(3)).expect("plan-only cone");
        assert_eq!(
            planned.edge_classes.iter().filter(|class| class.is_seam()).count(),
            1,
            "the plan-only GPU path carries the cone's seam class",
        );
    }
}
