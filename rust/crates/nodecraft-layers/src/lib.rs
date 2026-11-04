//! Adapters from the upstream solidcore record/ops/tess layers into NodeCraft's
//! protocol types.
//!
//! This crate is the future display/geometry boundary (see
//! `docs/upstream-solidcore-layers.md`). It deliberately depends on the upstream
//! layer crates instead of the kernel: `solidcore-tess` produces the mesh,
//! `solidcore-ops` owns bodies and workflows, `solidcore-record` is the exact
//! body data.
//!
//! What is still missing upstream is the kernel B-rep <-> record B-rep
//! conversion. Until that lands, [`tessellate_record`] can only be driven from a
//! record that already exists (a fixture, a future record-first engine); the
//! NodeCraft engine still holds kernel B-reps.

use nodecraft_protocol::Mesh;
use solidcore_record::brep::Brep;
use solidcore_tess::MeshTolerance;

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
    record: &Brep,
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
    let mesh = protocol_mesh(&tess);
    let edge_polylines = tess
        .edges
        .iter()
        .map(|polyline| polyline.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect())
        .collect();
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

fn protocol_mesh(tess: &solidcore_tess::Mesh) -> Mesh {
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
    mesh
}

/// Why the current engine cannot drive `solidcore-tess` yet. This is the upstream
/// contribution tracked in `docs/upstream-solidcore-layers.md` §7.1: a whole-Brep
/// record -> kernel (and back) conversion, exposed through `solidcore-ops`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordBridge {
    Missing,
}

/// The engine holds kernel B-reps today; there is no conversion into the record
/// `solidcore-tess::plan` takes. Kept explicit so callers cannot accidentally
/// assume a bridge exists.
pub fn record_bridge() -> RecordBridge {
    RecordBridge::Missing
}
