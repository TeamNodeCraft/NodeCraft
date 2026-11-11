//! Display-side plan creation and caching.
//!
//! The exact engine hands out immutable B-reps. This service owns the display
//! plan cache and quality choice; it never evaluates a graph.

pub mod service;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

pub use nodecraft_exact::{BodyKey, DisplayTransform, ExactBody, ExactPart};

/// One body's display plan and placement, kept for GPU plan evaluation.
#[derive(Clone, Debug)]
pub struct PlanInstance {
    pub target: String,
    pub plan: Arc<solidcore_tess::TessPlan>,
    /// sRGB colour by B-rep face index, from the display mesh's face groups.
    pub face_colors: Vec<[f32; 3]>,
    pub transforms: Vec<DisplayTransform>,
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
        }
    }

    #[must_use]
    pub fn get(&self, body: BodyKey, quality: DisplayQuality) -> Option<CachedBodyPlan> {
        let (tolerance, detail) = quality.key();
        self.entries.get(&(body, tolerance, detail)).cloned()
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
        bodies: vec![ExactBody { key: BodyKey::new(std::sync::Arc::as_ptr(brep) as u64, 0), brep: std::sync::Arc::clone(brep), transforms: Vec::new() }],
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

/// Write a Part as exact STEP from realized B-reps. Display/mesh state is not
/// involved; faceted export is a separate future path.
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
    nodecraft_step::write_breps_exact(path_text, &breps)?;
    nodecraft_step::import_solids(path_text)
        .map_err(|error| format!("exported STEP did not re-import: {error}"))?;
    apply_export_metadata(values, path_text)
}

fn apply_export_metadata(values: &serde_json::Value, path: &str) -> Result<(), String> {
    use nodecraft_step::LengthUnit;
    let target = match values.get("unit").and_then(serde_json::Value::as_str).unwrap_or("mm").to_ascii_lowercase().as_str() {
        "cm" | "centimetre" | "centimeter" => Some(LengthUnit::Centimetre),
        "m" | "metre" | "meter" => Some(LengthUnit::Metre),
        "km" | "kilometre" | "kilometer" => Some(LengthUnit::Kilometre),
        "um" | "micrometre" | "micrometer" => Some(LengthUnit::Micrometre),
        "in" | "inch" => Some(LengthUnit::Inch),
        _ => None,
    };
    if let Some(target) = target {
        let source = nodecraft_step::read_length_unit(path).unwrap_or(LengthUnit::Millimetre);
        nodecraft_step::convert_step_units(path, source, target)?;
    }
    let name = values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("solid");
    let author = values.get("author").and_then(serde_json::Value::as_str).unwrap_or("");
    let organization = values.get("organization").and_then(serde_json::Value::as_str).unwrap_or("NodeCraft");
    nodecraft_step::set_header(path, name, author, organization)?;
    if let Some(schema) = values.get("schema").and_then(serde_json::Value::as_str) {
        nodecraft_step::set_schema(path, schema)?;
    }
    if let Some(name) = values.get("productName").and_then(serde_json::Value::as_str) {
        nodecraft_step::set_product_name(path, name)?;
    }
    if let Some(layer) = values.get("layer").and_then(serde_json::Value::as_str) {
        nodecraft_step::set_layer(path, layer)?;
    }
    if let Some(material) = values.get("material").and_then(serde_json::Value::as_str) {
        nodecraft_step::set_material(path, material)?;
    }
    if let Some(color) = values.get("color").and_then(serde_json::Value::as_str) {
        let hex = color.trim_start_matches('#');
        if hex.len() != 6 {
            return Err("color must be #RRGGBB".into());
        }
        let component = |range| u8::from_str_radix(&hex[range], 16).map(|value| f64::from(value) / 255.0).map_err(|error| error.to_string());
        nodecraft_step::set_color(path, component(0..2)?, component(2..4)?, component(4..6)?)?;
    }
    Ok(())
}

/// Mesh one standalone B-rep through the display service.
pub fn mesh_brep(
    brep: &std::sync::Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> Result<nodecraft_protocol::Mesh, String> {
    let part = ExactPart {
        bodies: vec![ExactBody { key: BodyKey::new(std::sync::Arc::as_ptr(brep) as u64, 0), brep: std::sync::Arc::clone(brep), transforms: Vec::new() }],
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
            .is_some_and(|node| node.node_type == "sketch")
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

fn transform_plan_point(mut p: [f64; 3], transform: &DisplayTransform) -> [f64; 3] {
    match transform {
        DisplayTransform::Place { offset, rotation } => {
            let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
            let (sx, cx) = rx.sin_cos();
            let (sy, cy) = ry.sin_cos();
            let (sz, cz) = rz.sin_cos();
            (p[1], p[2]) = (p[1] * cx - p[2] * sx, p[1] * sx + p[2] * cx);
            (p[0], p[2]) = (p[0] * cy + p[2] * sy, -p[0] * sy + p[2] * cy);
            (p[0], p[1]) = (p[0] * cz - p[1] * sz, p[0] * sz + p[1] * cz);
            p[0] += offset[0];
            p[1] += offset[1];
            p[2] += offset[2];
        }
        DisplayTransform::Mirror { plane, offset } => match *plane {
            "xz" => p[1] = 2.0 * offset - p[1],
            "yz" => p[0] = 2.0 * offset - p[0],
            _ => p[2] = 2.0 * offset - p[2],
        },
        DisplayTransform::Scale { factor } => {
            p[0] *= factor;
            p[1] *= factor;
            p[2] *= factor;
        }
    }
    p
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
    let (tolerance, detail) = match quality {
        DisplayQuality::Full(tolerance) => (tolerance, None),
        DisplayQuality::Preview(detail) => (0.35, Some(detail)),
    };
    let mut planned: Vec<Option<CachedBodyPlan>> = vec![None; part.bodies.len()];
    let mut misses = Vec::new();
    let mut pending: HashMap<BodyKey, Vec<usize>> = HashMap::new();
    for (index, body) in part.bodies.iter().enumerate() {
        if let Some(cached) = cache.get(body.key, quality) {
            planned[index] = Some(cached);
        } else {
            pending.entry(body.key).or_default().push(index);
        }
    }
    // One geometry key is planned once; every instance of that body reuses the
    // same plan Arc. This is what lets a scatter share GPU meshes.
    for (key, indices) in &pending {
        misses.push((*key, indices[0]));
    }
    if !misses.is_empty() {
        // Forest-like Parts have hundreds of independent bodies. Planning them
        // one by one is the remaining serial meshing bottleneck.
        let threads = std::thread::available_parallelism()
            .map_or(4, std::num::NonZeroUsize::get)
            .clamp(1, 16)
            .min(misses.len());
        let chunk = misses.len().div_ceil(threads);
        let mut results = Vec::with_capacity(misses.len());
        std::thread::scope(|scope| -> Result<(), String> {
            let mut handles = Vec::new();
            for chunk in misses.chunks(chunk) {
                handles.push(scope.spawn(move || -> Result<Vec<(BodyKey, CachedBodyPlan)>, String> {
                    let mut kernel = nodecraft_solidcore::kernel::kernel();
                    let mut out = Vec::with_capacity(chunk.len());
                    for &(key, index) in chunk {
                        let body = &part.bodies[index];
                        let data = nodecraft_layers::plan_kernel(
                            &mut kernel,
                            body.brep.as_ref(),
                            tolerance,
                            detail,
                        )?;
                        out.push((key, CachedBodyPlan {
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
        for (key, data) in results {
            cache.insert(
                key,
                quality,
                data.plan.clone(),
                data.edge_ids.clone(),
                data.stats.clone(),
            );
            if let Some(indices) = pending.get(&key) {
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
            for point in polyline {
                let mut p = [point.x, point.y, point.z];
                for transform in &body.transforms {
                    p = transform_plan_point(p, transform);
                }
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
            transforms: body.transforms.clone(),
            edge_ids: std::sync::Arc::from(cached.edge_ids),
        });
    }
    Ok(PartPlans { plans, stats, edge_mesh })
}

/// Build the display mesh and GPU plans for an exact Part, reusing cached plans.
pub fn mesh_part(part: &ExactPart, quality: DisplayQuality, cache: &mut PlanCache) -> Result<PartDisplay, String> {
    mesh_part_impl(part, quality, cache, false)
}

/// Like [`mesh_part`], but also builds plans for marked fillet/chamfer blend
/// faces. That requires cloning the body plan and is only needed for the
/// selected blend edit's ghost overlay.
pub fn mesh_part_with_features(
    part: &ExactPart,
    quality: DisplayQuality,
    cache: &mut PlanCache,
) -> Result<PartDisplay, String> {
    mesh_part_impl(part, quality, cache, true)
}

fn mesh_part_impl(
    part: &ExactPart,
    quality: DisplayQuality,
    cache: &mut PlanCache,
    include_features: bool,
) -> Result<PartDisplay, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let mut meshes = Vec::new();
    let mut plans = Vec::new();
    let mut feature_plans = Vec::new();
    let mut stats = Vec::new();
    for body in &part.bodies {
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
        for transform in &body.transforms {
            display.mesh = match transform {
                DisplayTransform::Place { offset, rotation } => mesh_placement(&display.mesh, *offset, *rotation),
                DisplayTransform::Mirror { plane, offset } => mesh_mirror(&display.mesh, plane, *offset),
                DisplayTransform::Scale { factor } => mesh_scale(&display.mesh, *factor),
            };
        }
        if include_features
            && let Some(feature) = blend_feature_plan(
                body.brep.as_ref(),
                &display.plan,
                &body.transforms,
            )
        {
            feature_plans.push(feature);
        }
        stats.push(display.stats);
        plans.push(PlanInstance {
            target: String::new(),
            plan: std::sync::Arc::clone(&display.plan),
            face_colors: face_colors_by_index(&display.mesh),
            transforms: body.transforms.clone(),
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
    transforms: &[DisplayTransform],
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
        transforms: transforms.to_vec(),
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

fn mesh_scale(mesh: &nodecraft_protocol::Mesh, factor: f64) -> nodecraft_protocol::Mesh {
    let mut scaled = mesh.clone();
    for vertex in &mut scaled.vertices {
        vertex[0] = (vertex[0] as f64 * factor) as f32;
        vertex[1] = (vertex[1] as f64 * factor) as f32;
        vertex[2] = (vertex[2] as f64 * factor) as f32;
    }
    scaled
}

fn mesh_placement(mesh: &nodecraft_protocol::Mesh, offset: [f64; 3], rotation: [f64; 3]) -> nodecraft_protocol::Mesh {
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let rotate = |point: [f64; 3]| {
        let (mut x, mut y, mut z) = (point[0], point[1], point[2]);
        let (sx, cx) = rx.sin_cos(); let (sy, cy) = ry.sin_cos(); let (sz, cz) = rz.sin_cos();
        (y, z) = (y * cx - z * sx, y * sx + z * cx);
        (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        (x, y) = (x * cz - y * sz, x * sz + y * cz);
        [x, y, z]
    };
    let mut transformed = mesh.clone();
    for vertex in &mut transformed.vertices {
        let point = rotate([vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]);
        vertex[0] = (point[0] + offset[0]) as f32;
        vertex[1] = (point[1] + offset[1]) as f32;
        vertex[2] = (point[2] + offset[2]) as f32;
    }
    for normal in &mut transformed.normals {
        let rotated = rotate([normal[0] as f64, normal[1] as f64, normal[2] as f64]);
        *normal = [rotated[0] as f32, rotated[1] as f32, rotated[2] as f32];
    }
    transformed
}

fn mesh_mirror(mesh: &nodecraft_protocol::Mesh, plane: &str, offset: f64) -> nodecraft_protocol::Mesh {
    let mut mirrored = mesh.clone();
    for vertex in &mut mirrored.vertices {
        match plane {
            "xz" => vertex[1] = (2.0 * offset - vertex[1] as f64) as f32,
            "yz" => vertex[0] = (2.0 * offset - vertex[0] as f64) as f32,
            _ => vertex[2] = (2.0 * offset - vertex[2] as f64) as f32,
        }
    }
    for normal in &mut mirrored.normals {
        match plane {
            "xz" => normal[1] = -normal[1],
            "yz" => normal[0] = -normal[0],
            _ => normal[2] = -normal[2],
        }
    }
    for face in &mut mirrored.faces { face.swap(1, 2); }
    mirrored
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
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), transforms: Vec::new() }],
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
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), transforms: Vec::new() }],
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
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), transforms: Vec::new() }],
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
            bodies: vec![ExactBody { key: BodyKey::new(0, 0), brep: Arc::new(brep), transforms: Vec::new() }],
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
                .map(|_| ExactBody { key, brep: std::sync::Arc::clone(&brep), transforms: Vec::new() })
                .collect(),
        };
        let mut cache = PlanCache::new(16);
        let planned = plan_part(&part, DisplayQuality::Full(0.35), &mut cache).expect("plan");
        assert_eq!(planned.plans.len(), 3);
        assert!(std::sync::Arc::ptr_eq(&planned.plans[0].plan, &planned.plans[1].plan));
        assert!(std::sync::Arc::ptr_eq(&planned.plans[1].plan, &planned.plans[2].plan));
    }

    #[test]
    fn quality_is_part_of_the_plan_key() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = nodecraft_solidcore::solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            nodecraft_solidcore::solidcore::math::Point3::new(-1.0, -1.0, -1.0),
            2.0,
            2.0,
            2.0,
        )
        .expect("box");
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
