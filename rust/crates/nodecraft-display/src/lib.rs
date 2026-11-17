//! Display-side plan creation and caching.
//!
//! The exact engine hands out immutable B-reps. This service owns the display
//! plan cache and quality choice; it never evaluates a graph.

pub mod lod;
pub mod service;

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

pub use lod::{
    BodyBounds, LOD_LADDER, LodCamera, LodParams, LodView, ViewportSize, body_bounds, ladder_index,
    projected_diameter_px, quality_for, world_per_pixel,
};
pub use nodecraft_exact::{Affine3, BodyKey, DisplayTransform, ExactBody, ExactPart, Placement};

/// One body's display plan and placement, kept for GPU plan evaluation.
#[derive(Clone, Debug)]
pub struct PlanInstance {
    pub target: String,
    pub plan: Arc<solidcore_tess::TessPlan>,
    /// sRGB colour by B-rep face index, from the display mesh's face groups.
    pub face_colors: Vec<[f32; 3]>,
    /// The body's one composed placement value.
    pub placement: Placement,
    /// Kernel edge id by plan/record edge index, for selection.
    pub edge_ids: std::sync::Arc<[Option<String>]>,
}

/// Map a display mesh's per-face-group colours back to B-rep face indices.
#[must_use]
pub fn face_colors_by_index(mesh: &nodecraft_protocol::Mesh) -> Vec<[f32; 3]> {
    let mut colors = Vec::new();
    for (group, color) in mesh.face_groups.iter().zip(&mesh.face_colors) {
        let Some(index) = group
            .rsplit_once("face:")
            .and_then(|(_, tail)| tail.parse::<usize>().ok())
        else {
            continue;
        };
        if colors.len() <= index {
            colors.resize(index + 1, [0.32, 0.34, 0.37]);
        }
        colors[index] = *color;
    }
    colors
}

/// Quality requested for one body's display plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DisplayQuality {
    /// Full display plan at the given tolerance.
    Full(f64),
    /// Provisional plan at `solidcore_tess::preview` detail 1–8.
    Preview(u8),
}

impl DisplayQuality {
    fn key(self) -> (u64, Option<u8>) {
        match self {
            Self::Full(tolerance) => (tolerance.to_bits(), None),
            Self::Preview(detail) => (0, Some(detail)),
        }
    }
}

/// The normal viewport tolerance, used whenever no camera/viewport quality is
/// available (headless checks, export fallbacks, the first exact frame).
pub const DEFAULT_TOLERANCE: f64 = nodecraft_mesh::DEFAULT_DISPLAY_TOLERANCE;

/// The shared fallback quality. Callers with a camera use
/// [`lod::quality_for`] per body instead.
#[must_use]
pub const fn default_quality() -> DisplayQuality {
    DisplayQuality::Full(DEFAULT_TOLERANCE)
}

/// One cached body plan plus the data needed to derive its CPU mesh without
/// re-planning.
#[derive(Clone, Debug)]
pub struct CachedBodyPlan {
    pub plan: Arc<solidcore_tess::TessPlan>,
    /// Kernel edge id by record edge index.
    pub edge_ids: Vec<Option<String>>,
    pub stats: nodecraft_layers::TessStats,
}

/// Bounded plan-only cache shared by one display worker.
#[derive(Clone, Debug)]
pub struct PlanCache {
    entries: HashMap<(BodyKey, u64, Option<u8>), CachedBodyPlan>,
    order: VecDeque<(BodyKey, u64, Option<u8>)>,
    limit: usize,
    hits: Cell<u64>,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl PlanCache {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            limit: limit.max(1),
            hits: Cell::new(0),
        }
    }

    #[must_use]
    pub fn get(&self, body: BodyKey, quality: DisplayQuality) -> Option<CachedBodyPlan> {
        let (tolerance, detail) = quality.key();
        let cached = self.entries.get(&(body, tolerance, detail)).cloned();
        if cached.is_some() {
            self.hits.set(self.hits.get().saturating_add(1));
        }
        cached
    }

    /// Number of successful plan lookups. Small camera moves that stay on the
    /// same LOD rung reuse plans and show up here; a re-tessellation does not.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.get()
    }

    pub fn reset_hits(&mut self) {
        self.hits.set(0);
    }

    pub fn insert(
        &mut self,
        body: BodyKey,
        quality: DisplayQuality,
        plan: Arc<solidcore_tess::TessPlan>,
        edge_ids: Vec<Option<String>>,
        stats: nodecraft_layers::TessStats,
    ) {
        let (tolerance, detail) = quality.key();
        let key = (body, tolerance, detail);
        if self.entries.insert(key, CachedBodyPlan { plan, edge_ids, stats }).is_none() {
            self.order.push_back(key);
        }
        while self.entries.len() > self.limit {
            let Some(oldest) = self.order.pop_front() else { break };
            self.entries.remove(&oldest);
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Plan one immutable body. The caller owns the kernel mutation required by the
/// upstream planner; this function only selects quality and stores the result.
pub fn tessellate_body(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    quality: DisplayQuality,
) -> Result<nodecraft_layers::TessDisplay, String> {
    let mut copy = brep
        .clone_standing_for_self()
        .map_err(|error| format!("display copy: {error}"))?;
    match quality {
        DisplayQuality::Full(tolerance) => {
            nodecraft_layers::tessellate_kernel(kernel, &mut copy, tolerance)
        }
        DisplayQuality::Preview(detail) => {
            nodecraft_layers::tessellate_kernel_preview(kernel, &mut copy, detail)
        }
    }
}

/// Display one standalone B-rep (for an exact difference body) through the
/// same plan service as a Part. Returns both the CPU mesh and the GPU plan.
pub fn brep_display(
    brep: &std::sync::Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> Result<PartDisplay, String> {
    let part = ExactPart {
        bodies: vec![ExactBody { key: BodyKey::new(std::sync::Arc::as_ptr(brep) as u64, 0), brep: std::sync::Arc::clone(brep), placement: Placement::identity() }],
    };
    mesh_part(&part, quality, cache)
}

/// Export-quality settings. Display quality and export quality are deliberately
/// separate: an export may request a finer plan than the viewport and must not
/// reuse a coarse display plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExportOptions {
    /// Maximum chord tolerance for the export mesh.
    pub tolerance: f64,
    /// Require every triangle edge to be shared by exactly two triangles.
    pub require_closed: bool,
}

/// A mesh prepared for an exporter. The caller writes the format (STL, etc.).
#[derive(Clone, Debug)]
pub struct ExportMesh {
    pub mesh: nodecraft_protocol::Mesh,
    pub stats: Vec<nodecraft_layers::TessStats>,
}

/// Build an export mesh at export tolerance. A cached plan is reused only when
/// its quality key matches this export's tolerance.
pub fn export_mesh(
    part: &ExactPart,
    options: ExportOptions,
    cache: &mut PlanCache,
) -> Result<ExportMesh, String> {
    let display = mesh_part(part, DisplayQuality::Full(options.tolerance), cache)?;
    if display.mesh.faces.is_empty() {
        return Err("export mesh is empty".into());
    }
    if display.mesh.vertices.iter().flatten().any(|value| !value.is_finite()) {
        return Err("export mesh has a non-finite vertex".into());
    }
    if options.require_closed {
        // Plan evaluation shares vertices within a face but not across faces;
        // weld by position before checking that every edge has two faces.
        let point = |index: u32| -> Option<[i64; 3]> {
            let p = display.mesh.vertices.get(index as usize)?;
            Some(p.map(|value| (f64::from(value) * 1.0e6).round() as i64))
        };
        let mut edges = std::collections::HashMap::<[[i64; 3]; 2], usize>::new();
        for face in &display.mesh.faces {
            let Some(points) = face.map(point).into_iter().collect::<Option<Vec<_>>>() else { continue };
            for [a, b] in [[points[0], points[1]], [points[1], points[2]], [points[2], points[0]]] {
                let key = if a <= b { [a, b] } else { [b, a] };
                *edges.entry(key).or_default() += 1;
            }
        }
        if let Some((edge, count)) = edges.into_iter().find(|(_, count)| *count != 2) {
            return Err(format!("export mesh is not closed at edge {edge:?} ({count} faces)"));
        }
    }
    Ok(ExportMesh { mesh: display.mesh, stats: display.stats })
}

/// Write a Part as STEP through CADIR. The exact format writes the realized B-reps;
/// `format = "faceted"` writes the display mesh as planar triangular faces.
pub fn export_step(
    part: &ExactPart,
    path: &std::path::Path,
    values: &serde_json::Value,
) -> Result<(), String> {
    let breps = part.realized_breps()?;
    if breps.is_empty() {
        return Err("nothing to export".into());
    }
    let path_text = path.to_str().ok_or("export path is not UTF-8")?;
    let options = export_options(values)?;
    if values.get("format").and_then(serde_json::Value::as_str) == Some("faceted") {
        let mut cache = PlanCache::new(8);
        let mesh = export_mesh(
            part,
            ExportOptions {
                tolerance: 0.05,
                require_closed: false,
            },
            &mut cache,
        )?;
        nodecraft_step::write_step_faceted(
            path_text,
            &mesh.mesh.vertices,
            &mesh.mesh.faces,
            &options,
        )?;
    } else {
        nodecraft_step::write_breps_exact_with_options(path_text, &breps, &options)?;
    }
    let bytes = std::fs::read(path).map_err(|error| format!("{path_text}: {error}"))?;
    let import = nodecraft_step::read_cad(&bytes, path_text)
        .map_err(|error| format!("exported STEP did not re-import: {error}"))?;
    if import.bodies.is_empty() {
        return Err(format!(
            "exported STEP did not re-import: {}",
            import.warnings.first().cloned().unwrap_or_else(|| "no bodies".into())
        ));
    }
    Ok(())
}

fn export_options(values: &serde_json::Value) -> Result<nodecraft_step::StepExportOptions, String> {
    let unit = values
        .get("unit")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("mm")
        .to_ascii_lowercase();
    if !matches!(unit.as_str(), "" | "mm" | "millimetre" | "millimeter") {
        return Err(format!(
            "STEP export writes millimetres; {unit:?} output is not supported"
        ));
    }
    let color = values
        .get("color")
        .and_then(serde_json::Value::as_str)
        .map(|color| {
            let hex = color.trim_start_matches('#');
            if hex.len() != 6 {
                return Err("color must be #RRGGBB".to_string());
            }
            let component = |range| {
                u8::from_str_radix(&hex[range], 16)
                    .map(|value| f64::from(value) / 255.0)
                    .map_err(|error| error.to_string())
            };
            Ok([component(0..2)?, component(2..4)?, component(4..6)?])
        })
        .transpose()?;
    Ok(nodecraft_step::StepExportOptions {
        product_name: values
            .get("productName")
            .or_else(|| values.get("fileName"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("solid")
            .to_string(),
        schema: values
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("ap242")
            .to_string(),
        author: values
            .get("author")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string(),
        organization: values
            .get("organization")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("NodeCraft")
            .to_string(),
        color,
        layer: values
            .get("layer")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
    })
}

/// Mesh one standalone B-rep through the display service.
pub fn mesh_brep(
    brep: &std::sync::Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> Result<nodecraft_protocol::Mesh, String> {
    let part = ExactPart {
        bodies: vec![ExactBody { key: BodyKey::new(std::sync::Arc::as_ptr(brep) as u64, 0), brep: std::sync::Arc::clone(brep), placement: Placement::identity() }],
    };
    mesh_part(&part, quality, cache).map(|display| display.mesh)
}

/// Build mesh payloads by target id without a protocol `Request`. Used by the
/// session runtime compatibility path.
pub fn protocol_meshes_for_targets(
    targets: &[String],
    exact_parts: &std::collections::BTreeMap<String, ExactPart>,
    quality: DisplayQuality,
    cache: &mut PlanCache,
    binary: bool,
) -> std::collections::BTreeMap<String, nodecraft_protocol::MeshPayload> {
    use base64::Engine as _;
    let mut payloads = std::collections::BTreeMap::new();
    for target in targets {
        let Some(part) = exact_parts.get(target) else { continue };
        let Ok(display) = mesh_part(part, quality, cache) else { continue };
        let mut mesh = display.mesh;
        if binary {
            let encode = |values: &[[f32; 3]]| {
                let mut bytes = Vec::with_capacity(values.len() * 12);
                for value in values {
                    for component in value {
                        bytes.extend_from_slice(&component.to_le_bytes());
                    }
                }
                base64::engine::general_purpose::STANDARD.encode(bytes)
            };
            mesh.vertices_data = Some(encode(&mesh.vertices));
            mesh.normals_data = Some(encode(&mesh.normals));
            mesh.vertices.clear();
            mesh.normals.clear();
        }
        payloads.insert(target.clone(), nodecraft_protocol::MeshPayload { key: None, mesh });
    }
    payloads
}

/// Build the protocol mesh payloads for a set of mesh targets from exact Parts.
/// This is the display-service replacement for the engine's JSON mesh response.
pub fn protocol_meshes(
    request: &nodecraft_protocol::Request,
    exact_parts: &std::collections::BTreeMap<String, ExactPart>,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> (std::collections::BTreeMap<String, nodecraft_protocol::MeshPayload>, Vec<String>) {
    use base64::Engine as _;
    let targets = request.mesh_targets.as_deref().unwrap_or(&request.targets);
    let mut payloads = std::collections::BTreeMap::new();
    let mut empty = Vec::new();
    for target in targets {
        if !request.targets.contains(target) {
            continue;
        }
        if request
            .nodes
            .iter()
            .find(|node| &node.id == target)
            .is_some_and(|node| nodecraft_node_contract::overlay_only(&node.node_type))
        {
            empty.push(target.clone());
            continue;
        }
        let Some(part) = exact_parts.get(target) else {
            empty.push(target.clone());
            continue;
        };
        let Ok(display) = mesh_part(part, quality, cache) else {
            empty.push(target.clone());
            continue;
        };
        let mut mesh = display.mesh;
        if request.binary_mesh {
            let encode = |values: &[[f32; 3]]| {
                let mut bytes = Vec::with_capacity(values.len() * 12);
                for value in values {
                    for component in value {
                        bytes.extend_from_slice(&component.to_le_bytes());
                    }
                }
                base64::engine::general_purpose::STANDARD.encode(bytes)
            };
            mesh.vertices_data = Some(encode(&mesh.vertices));
            mesh.normals_data = Some(encode(&mesh.normals));
            mesh.vertices.clear();
            mesh.normals.clear();
        }
        let key = request
            .nodes
            .iter()
            .find(|node| &node.id == target)
            .and_then(|node| node.cache.as_ref()?.key.clone());
        payloads.insert(target.clone(), nodecraft_protocol::MeshPayload { key, mesh });
    }
    (payloads, empty)
}

/// Display output for one exact Part.
#[derive(Clone, Debug, Default)]
pub struct PartDisplay {
    pub mesh: nodecraft_protocol::Mesh,
    /// Whole-body display plans.
    pub plans: Vec<PlanInstance>,
    /// Plans for the faces a fillet/chamfer operation added, when the B-rep
    /// carries blend-face marks. Used for the edit-mode feature ghost.
    pub feature_plans: Vec<PlanInstance>,
    pub stats: Vec<nodecraft_layers::TessStats>,
}

/// Plan-only display data: GPU plans plus the edge polylines needed for
/// picking/overlays. No CPU triangle mesh is evaluated.
#[derive(Clone, Debug, Default)]
pub struct PartPlans {
    pub plans: Vec<PlanInstance>,
    pub stats: Vec<nodecraft_layers::TessStats>,
    pub edge_mesh: nodecraft_protocol::Mesh,
}

/// Apply a body's one composed placement value to a display mesh.
///
/// This is the single mesh placement path: vertices and normals through the
/// cached matrix, with winding flipped once when the chain contains a mirror.
/// It replaces the separate place/mirror/scale mesh helpers.
pub fn apply_placement(mesh: &mut nodecraft_protocol::Mesh, placement: &Placement) {
    if placement.is_identity() {
        return;
    }
    let matrix = *placement.matrix();
    for vertex in &mut mesh.vertices {
        let point = matrix.apply_point([
            f64::from(vertex[0]),
            f64::from(vertex[1]),
            f64::from(vertex[2]),
        ]);
        *vertex = [point[0] as f32, point[1] as f32, point[2] as f32];
    }
    for normal in &mut mesh.normals {
        let rotated = matrix.apply_vector([
            f64::from(normal[0]),
            f64::from(normal[1]),
            f64::from(normal[2]),
        ]);
        let length = (rotated[0] * rotated[0] + rotated[1] * rotated[1] + rotated[2] * rotated[2]).sqrt();
        if length > 1.0e-12 {
            *normal = [
                (rotated[0] / length) as f32,
                (rotated[1] / length) as f32,
                (rotated[2] / length) as f32,
            ];
        }
    }
    if placement.has_mirror() {
        for face in &mut mesh.faces {
            face.swap(1, 2);
        }
    }
}

fn brep_face_colours(brep: &nodecraft_solidcore::solidcore::topology::Brep, count: usize) -> Vec<[f32; 3]> {
    use nodecraft_solidcore::solidcore::error::EntityRef;
    use nodecraft_solidcore::solidcore::geom::attribute::AttributeValue;
    const LOOK_COLOR_ATTRIBUTE: u64 = 100;
    let mut colours = vec![[0.17, 0.18, 0.195]; count];
    for (position, &face) in brep.face_list().iter().enumerate() {
        let Some(attribute) = brep.find_attribute(EntityRef::Face(face), LOOK_COLOR_ATTRIBUTE) else { continue };
        let (AttributeValue::Color(color) | AttributeValue::Vector3d(color)) = attribute.value() else { continue };
        if let Some(slot) = colours.get_mut(position) {
            *slot = [color.x as f32, color.y as f32, color.z as f32];
        }
    }
    colours
}

/// Plan each body without evaluating the plan into triangles. The returned
/// `edge_mesh` contains only plan edge polylines and is used for picking.
pub fn plan_part(part: &ExactPart, quality: DisplayQuality, cache: &mut PlanCache) -> Result<PartPlans, String> {
    plan_part_with_qualities(part, quality, &BTreeMap::new(), &[], cache)
}

/// Plan each body at its own display quality. `default` is used for bodies
/// absent from `by_body`; `priority` lists body keys nearest/on-screen first so
/// the planner starts with the bodies the user is looking at.
///
/// Edge polylines are built from the same cached plan as the surface, so both
/// use exactly the body's chosen tolerance at every level.
pub fn plan_part_with_qualities(
    part: &ExactPart,
    default: DisplayQuality,
    by_body: &BTreeMap<BodyKey, DisplayQuality>,
    priority: &[BodyKey],
    cache: &mut PlanCache,
) -> Result<PartPlans, String> {
    let started = std::time::Instant::now();
    let body_count = part.bodies.len();
    let mut planned: Vec<Option<CachedBodyPlan>> = vec![None; part.bodies.len()];
    // Keyed by body and quality: one geometry key is planned once per level and
    // every instance of that body reuses the same plan Arc. A scatter of one
    // shape therefore shares GPU meshes at each level.
    let mut pending: BTreeMap<(BodyKey, u64, Option<u8>), (DisplayQuality, Vec<usize>)> =
        BTreeMap::new();
    for (index, body) in part.bodies.iter().enumerate() {
        let quality = by_body.get(&body.key).copied().unwrap_or(default);
        if let Some(cached) = cache.get(body.key, quality) {
            planned[index] = Some(cached);
        } else {
            let (tolerance, detail) = quality.key();
            pending
                .entry((body.key, tolerance, detail))
                .or_insert_with(|| (quality, Vec::new()))
                .1
                .push(index);
        }
    }
    let rank: HashMap<BodyKey, usize> = priority
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect();
    let mut misses: Vec<(DisplayQuality, Vec<usize>)> = pending.values().cloned().collect();
    misses.sort_by_key(|(_, indices)| {
        indices
            .first()
            .and_then(|index| part.bodies.get(*index))
            .and_then(|body| rank.get(&body.key))
            .copied()
            .unwrap_or(usize::MAX)
    });
    let mut plan_threads = 0;
    if !misses.is_empty() {
        // Forest-like Parts have hundreds of independent bodies. Planning them
        // one by one is the remaining serial meshing bottleneck.
        let threads = std::thread::available_parallelism()
            .map_or(4, std::num::NonZeroUsize::get)
            .clamp(1, 16)
            .min(misses.len());
        plan_threads = threads;
        let chunk = misses.len().div_ceil(threads);
        let mut results = Vec::with_capacity(misses.len());
        std::thread::scope(|scope| -> Result<(), String> {
            let mut handles = Vec::new();
            for chunk in misses.chunks(chunk) {
                handles.push(scope.spawn(move || -> Result<Vec<(BodyKey, DisplayQuality, CachedBodyPlan)>, String> {
                    let mut kernel = nodecraft_solidcore::kernel::kernel();
                    let mut out = Vec::with_capacity(chunk.len());
                    for (quality, indices) in chunk {
                        let Some(index) = indices.first() else { continue };
                        let body = &part.bodies[*index];
                        let (tolerance, detail) = match *quality {
                            DisplayQuality::Full(tolerance) => (tolerance, None),
                            DisplayQuality::Preview(detail) => (DEFAULT_TOLERANCE, Some(detail)),
                        };
                        let data = nodecraft_layers::plan_kernel(
                            &mut kernel,
                            body.brep.as_ref(),
                            tolerance,
                            detail,
                        )?;
                        out.push((body.key, *quality, CachedBodyPlan {
                            plan: data.plan,
                            edge_ids: data.plan_edge_ids,
                            stats: data.stats,
                        }));
                    }
                    Ok(out)
                }));
            }
            for handle in handles {
                let values = handle
                    .join()
                    .map_err(|_| "display planning thread panicked".to_owned())??;
                results.extend(values);
            }
            Ok(())
        })?;
        for (key, quality, data) in results {
            cache.insert(
                key,
                quality,
                data.plan.clone(),
                data.edge_ids.clone(),
                data.stats.clone(),
            );
            let (tolerance, detail) = quality.key();
            if let Some((_, indices)) = pending.get(&(key, tolerance, detail)) {
                for &index in indices {
                    planned[index] = Some(data.clone());
                }
            }
        }
    }
    let mut plans = Vec::new();
    let mut stats = Vec::new();
    let mut edge_mesh = nodecraft_protocol::Mesh::default();
    for (body_index, body) in part.bodies.iter().enumerate() {
        let cached = planned[body_index]
            .take()
            .ok_or("display plan missing after planning")?;
        stats.push(cached.stats.clone());
        // Edge polylines in display space, for the CPU picking/overlay layer.
        for (edge_index, polyline) in cached.plan.edges.iter().enumerate() {
            if polyline.is_empty() {
                continue;
            }
            let base = edge_mesh.vertices.len() as u32;
            let matrix = *body.placement.matrix();
            for point in polyline {
                let p = matrix.apply_point([point.x, point.y, point.z]);
                edge_mesh.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
            let mut segments = Vec::with_capacity(polyline.len().saturating_sub(1));
            for index in 0..polyline.len().saturating_sub(1) {
                segments.push([base + index as u32, base + index as u32 + 1]);
            }
            if !segments.is_empty() {
                let id = cached
                    .edge_ids
                    .get(edge_index)
                    .and_then(Clone::clone)
                    .unwrap_or_else(|| format!("edge:{edge_index}"));
                let id = if part.bodies.len() > 1 {
                    format!("body:{body_index}:{id}")
                } else {
                    id
                };
                edge_mesh.brep_edges.push(nodecraft_protocol::BrepEdge { id, kind: "edge".into(), segments });
            }
        }
        let face_colors = brep_face_colours(body.brep.as_ref(), cached.plan.faces.len());
        plans.push(PlanInstance {
            target: String::new(),
            plan: cached.plan,
            face_colors,
            placement: body.placement.clone(),
            edge_ids: std::sync::Arc::from(cached.edge_ids),
        });
    }
    if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
        eprintln!(
            "[pool] plan_part bodies={body_count} misses={} threads={plan_threads} wall_ms={:.2}",
            misses.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(PartPlans { plans, stats, edge_mesh })
}

/// Build the display mesh and GPU plans for an exact Part, reusing cached plans.
pub fn mesh_part(part: &ExactPart, quality: DisplayQuality, cache: &mut PlanCache) -> Result<PartDisplay, String> {
    mesh_part_impl(part, quality, &BTreeMap::new(), cache, false)
}

/// Build a CPU display mesh and GPU plans with one quality per body.
pub fn mesh_part_with_qualities(
    part: &ExactPart,
    default: DisplayQuality,
    by_body: &BTreeMap<BodyKey, DisplayQuality>,
    _priority: &[BodyKey],
    cache: &mut PlanCache,
) -> Result<PartDisplay, String> {
    mesh_part_impl(part, default, by_body, cache, false)
}

/// Like [`mesh_part`], but also builds plans for marked fillet/chamfer blend
/// faces. That requires cloning the body plan and is only needed for the
/// selected blend edit's ghost overlay.
pub fn mesh_part_with_features(
    part: &ExactPart,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> Result<PartDisplay, String> {
    mesh_part_impl(part, quality, &BTreeMap::new(), cache, true)
}

/// [`mesh_part_with_features`] with per-body display quality.
pub fn mesh_part_with_features_and_qualities(
    part: &ExactPart,
    default: DisplayQuality,
    by_body: &BTreeMap<BodyKey, DisplayQuality>,
    _priority: &[BodyKey],
    cache: &mut PlanCache,
) -> Result<PartDisplay, String> {
    mesh_part_impl(part, default, by_body, cache, true)
}

fn mesh_part_impl(
    part: &ExactPart,
    default: DisplayQuality,
    by_body: &BTreeMap<BodyKey, DisplayQuality>,
    cache: &mut PlanCache,
    include_features: bool,
) -> Result<PartDisplay, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let mut meshes = Vec::new();
    let mut plans = Vec::new();
    let mut feature_plans = Vec::new();
    let mut stats = Vec::new();
    for body in &part.bodies {
        let quality = by_body.get(&body.key).copied().unwrap_or(default);
        let mut display = match cache.get(body.key, quality) {
            Some(cached) => nodecraft_layers::evaluate_plan(&cached.plan, &cached.edge_ids),
            None => {
                // `tessellate_body` already evaluated the plan into its display
                // mesh. Do not call `evaluate_plan` a second time on a miss.
                let display = tessellate_body(&mut kernel, &body.brep, quality)?;
                cache.insert(
                    body.key,
                    quality,
                    display.plan.clone(),
                    display.plan_edge_ids.clone(),
                    display.stats.clone(),
                );
                display
            }
        };
        apply_brep_face_colors(body.brep.as_ref(), &mut display.mesh);
        apply_placement(&mut display.mesh, &body.placement);
        if include_features
            && let Some(feature) = blend_feature_plan(
                body.brep.as_ref(),
                &display.plan,
                &body.placement,
            )
        {
            feature_plans.push(feature);
        }
        stats.push(display.stats);
        plans.push(PlanInstance {
            target: String::new(),
            plan: std::sync::Arc::clone(&display.plan),
            face_colors: face_colors_by_index(&display.mesh),
            placement: body.placement.clone(),
            edge_ids: std::sync::Arc::from(display.plan_edge_ids.clone()),
        });
        meshes.push(display.mesh);
    }
    let mesh = if meshes.len() == 1 {
        meshes.pop().unwrap_or_default()
    } else {
        nodecraft_mesh::merge_meshes(meshes)
    };
    Ok(PartDisplay { mesh, plans, feature_plans, stats })
}

fn blend_feature_plan(
    brep: &nodecraft_solidcore::solidcore::topology::Brep,
    plan: &std::sync::Arc<solidcore_tess::TessPlan>,
    placement: &Placement,
) -> Option<PlanInstance> {
    use nodecraft_solidcore::solidcore::blend::definition_check::blend_face_marks;
    let marked = blend_face_marks(brep).ok()?;
    if marked.is_empty() {
        return None;
    }
    let face_list = brep.face_list();
    let selected: std::collections::BTreeSet<usize> = marked
        .iter()
        .filter_map(|face| face_list.iter().position(|candidate| *candidate == face.face))
        .collect();
    if selected.is_empty() || selected.iter().any(|index| *index >= plan.faces.len()) {
        return None;
    }
    let mut filtered = (**plan).clone();
    filtered.faces = selected.iter().filter_map(|index| plan.faces.get(*index).cloned()).collect();
    // The ghost is a face overlay; its edges stay out of the pickable edge set.
    filtered.edges = Vec::new();
    filtered.edge_faces = Vec::new();
    Some(PlanInstance {
        target: "feature".into(),
        plan: std::sync::Arc::new(filtered),
        face_colors: Vec::new(),
        placement: placement.clone(),
        edge_ids: std::sync::Arc::from(Vec::new()),
    })
}

fn apply_brep_face_colors(brep: &nodecraft_solidcore::solidcore::topology::Brep, mesh: &mut nodecraft_protocol::Mesh) {
    use nodecraft_solidcore::solidcore::error::EntityRef;
    use nodecraft_solidcore::solidcore::geom::attribute::AttributeValue;
    const LOOK_COLOR_ATTRIBUTE: u64 = 100;
    let mut by_face = std::collections::BTreeMap::new();
    for (position, &face) in brep.face_list().iter().enumerate() {
        let Some(attribute) = brep.find_attribute(EntityRef::Face(face), LOOK_COLOR_ATTRIBUTE) else { continue };
        let (AttributeValue::Color(color) | AttributeValue::Vector3d(color)) = attribute.value() else { continue };
        by_face.insert(position, [color.x as f32, color.y as f32, color.z as f32]);
    }
    if by_face.is_empty() {
        mesh.face_colors.clear();
        return;
    }
    mesh.face_colors = mesh.face_groups.iter().map(|group| {
        group.rsplit_once("face:").and_then(|(_, tail)| tail.parse::<usize>().ok())
            .and_then(|index| by_face.get(&index).copied())
            .unwrap_or([0.32, 0.34, 0.37])
    }).collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Arc<solidcore_tess::TessPlan> {
        Arc::new(solidcore_tess::TessPlan::default())
    }

    #[test]
    fn export_step_writes_exact_breps_without_display_state() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
        let part = ExactPart {
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), placement: Placement::identity() }],
        };
        let path = std::env::temp_dir().join("nodecraft-display-export-test.step");
        export_step(&part, &path, &serde_json::json!({"fileName": "test"})).expect("export");
        assert!(path.metadata().map(|meta| meta.len()).unwrap_or(0) > 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn marked_blend_faces_produce_a_feature_only_plan() {
        use nodecraft_solidcore::solidcore::blend::definition_check::{mark_blend_faces, MadeBlendFace, BlendFaceKind};
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let mut brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
        let face = brep.face_list()[0];
        mark_blend_faces(&mut brep, &[MadeBlendFace { face, kind: BlendFaceKind::Round }])
            .expect("mark");
        let part = ExactPart {
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), placement: Placement::identity() }],
        };
        let mut cache = PlanCache::new(8);
        let display = mesh_part_with_features(&part, DisplayQuality::Full(0.35), &mut cache).expect("display");
        assert_eq!(display.feature_plans.len(), 1);
        assert_eq!(display.feature_plans[0].plan.faces.len(), 1);
    }

    #[test]
    fn export_quality_has_its_own_plan_key_and_topology_check() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
        let part = ExactPart {
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), placement: Placement::identity() }],
        };
        let mut cache = PlanCache::new(4);
        let display = mesh_part(&part, DisplayQuality::Full(0.35), &mut cache).expect("display");
        assert_eq!(cache.len(), 1);
        let export = export_mesh(&part, ExportOptions { tolerance: 0.2, require_closed: true }, &mut cache)
            .expect("export");
        assert!(!export.mesh.faces.is_empty());
        assert!(cache.len() >= 2, "export tolerance must not reuse the display plan");
        assert_eq!(display.plans.len(), 1);
    }

    #[test]
    fn mesh_part_meshes_an_exact_brep_without_a_graph() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
        let part = ExactPart {
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), placement: Placement::identity() }],
        };
        let mut cache = PlanCache::new(4);
        let display = mesh_part(&part, DisplayQuality::Full(0.35), &mut cache).expect("mesh");
        assert!(!display.mesh.faces.is_empty());
        assert_eq!(display.plans.len(), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn instances_of_one_body_share_one_plan() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
        let key = BodyKey::new(7, 0);
        let brep = std::sync::Arc::new(brep);
        let part = ExactPart {
            bodies: (0..3)
                .map(|_| ExactBody { key, brep: std::sync::Arc::clone(&brep), placement: Placement::identity() })
                .collect(),
        };
        let mut cache = PlanCache::new(16);
        let planned = plan_part(&part, DisplayQuality::Full(0.35), &mut cache).expect("plan");
        assert_eq!(planned.plans.len(), 3);
        assert!(std::sync::Arc::ptr_eq(&planned.plans[0].plan, &planned.plans[1].plan));
        assert!(std::sync::Arc::ptr_eq(&planned.plans[1].plan, &planned.plans[2].plan));
    }

    #[test]
    fn per_body_quality_drives_surface_and_edge_polylines() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = Arc::new(
            nodecraft_solidcore::solidcore::primitive::api_primitives::create_sphere(
                &mut kernel,
                nodecraft_solidcore::solidcore::math::Point3::new(0.0, 0.0, 0.0),
                10.0,
            )
            .expect("sphere"),
        );
        let fine = BodyKey::new(1, 0);
        let coarse = BodyKey::new(2, 0);
        let part = ExactPart {
            bodies: vec![
                ExactBody { key: fine, brep: Arc::clone(&brep), placement: Placement::identity() },
                ExactBody { key: coarse, brep: Arc::clone(&brep), placement: Placement::identity() },
            ],
        };
        let by_body = BTreeMap::from([
            (fine, DisplayQuality::Full(0.05)),
            (coarse, DisplayQuality::Full(1.6)),
        ]);
        let mut cache = PlanCache::new(8);
        let planned = plan_part_with_qualities(
            &part,
            default_quality(),
            &by_body,
            &[coarse, fine],
            &mut cache,
        )
        .expect("per-body plan");
        assert_eq!(planned.plans.len(), 2);
        assert_eq!(cache.len(), 2, "one plan per body and level");

        let triangles = |plans: &PartPlans| -> usize {
            plans.plans.iter().map(|instance| instance.plan.faces.iter().map(|face| face.triangles.len()).sum::<usize>()).sum()
        };
        let fine_only = plan_part(
            &ExactPart { bodies: vec![part.bodies[0].clone()] },
            DisplayQuality::Full(0.05),
            &mut PlanCache::new(2),
        )
        .expect("fine plan");
        let coarse_only = plan_part(
            &ExactPart { bodies: vec![part.bodies[1].clone()] },
            DisplayQuality::Full(1.6),
            &mut PlanCache::new(2),
        )
        .expect("coarse plan");
        assert!(
            triangles(&fine_only) > triangles(&coarse_only),
            "the fine level must carry more triangles than the coarse level",
        );
        assert_eq!(triangles(&planned), triangles(&fine_only) + triangles(&coarse_only));
        // The edge payload is built from each body's own plan, never a second
        // tolerance: the merged polylines are exactly the two per-level sets.
        assert_eq!(
            planned.edge_mesh.vertices.len(),
            fine_only.edge_mesh.vertices.len() + coarse_only.edge_mesh.vertices.len(),
        );
        assert_eq!(
            planned.edge_mesh.brep_edges.len(),
            fine_only.edge_mesh.brep_edges.len() + coarse_only.edge_mesh.brep_edges.len(),
        );
        assert_eq!(
            planned.plans[0].plan.edges.len(),
            fine_only.plans[0].plan.edges.len(),
        );
        assert_eq!(
            planned.plans[1].plan.edges.len(),
            coarse_only.plans[0].plan.edges.len(),
        );
    }

    #[test]
    fn repeated_quality_reuses_the_cached_plan_and_counts_hits() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_cylinder(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(0.0, 0.0, 0.0),
            10.0,
            20.0,
        )
        .expect("cylinder");
        let part = ExactPart {
            bodies: vec![ExactBody {
                key: BodyKey::new(3, 0),
                brep: Arc::new(brep),
                placement: Placement::identity(),
            }],
        };
        let mut cache = PlanCache::new(4);
        let first = plan_part(&part, DisplayQuality::Full(0.2), &mut cache).expect("first plan");
        let hits_after_first = cache.hits();
        assert_eq!(hits_after_first, 0, "the first plan is a miss");
        let second = plan_part(&part, DisplayQuality::Full(0.2), &mut cache).expect("second plan");
        assert!(cache.hits() > hits_after_first, "the unchanged level must reuse the plan");
        assert_eq!(cache.len(), 1, "one body and one level means one plan");
        assert!(Arc::ptr_eq(&first.plans[0].plan, &second.plans[0].plan));
    }

    #[test]
    fn quality_is_part_of_the_plan_key() {
        let mut cache = PlanCache::new(4);
        let key = BodyKey::new(0, 0);
        cache.insert(
            key,
            DisplayQuality::Full(0.35),
            plan(),
            Vec::new(),
            nodecraft_layers::TessStats::default(),
        );
        assert!(cache.get(key, DisplayQuality::Full(0.35)).is_some());
        assert!(cache.get(key, DisplayQuality::Full(0.2)).is_none());
        assert!(cache.get(key, DisplayQuality::Preview(1)).is_none());
    }
}
