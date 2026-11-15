//! Display evaluation and mesh caching.

use super::*;

/// Display-only tessellation through the upstream layer seam. Exact geometry
/// never passes through this function.
#[cfg(test)]
fn display_quality(tolerance: f64, preview_detail: Option<u8>) -> nodecraft_display::DisplayQuality {
    match preview_detail {
        Some(detail) => nodecraft_display::DisplayQuality::Preview(detail),
        None => nodecraft_display::DisplayQuality::Full(tolerance),
    }
}

#[cfg(test)]
fn display_mesh_for_brep(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> Result<Mesh, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let mut brep = copy_brep(brep)?;
    nodecraft_layers::tessellate_kernel(&mut kernel, &mut brep, 0.05).map(|display| display.mesh)
}

/// Write an evaluated exact Part as one STEP body per Part body.
///
/// Realising the Part bakes each placement into its body; the body's imported
/// name and colour travel to the written Solid. This is the export route for a
/// Part whose node exposes several bodies (for example a Circular Pattern)
/// where the single-operand exact path does not apply.
fn export_part_bodies(
    part: &Part,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let solids = part.realized_bodies()?;
    if solids.is_empty() {
        return Err("Part has no solids".into());
    }
    let stem = std::path::Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model")
        .to_string();
    let mut bodies = Vec::with_capacity(solids.len());
    for (index, solid) in solids.iter().enumerate() {
        let instance = part.bodies.get(index);
        let name = instance
            .and_then(|body| body.name.clone())
            .unwrap_or_else(|| {
                if solids.len() == 1 {
                    stem.clone()
                } else {
                    format!("{stem}_{}", index + 1)
                }
            });
        bodies.push(nodecraft_step::ExportBody {
            name,
            color: instance.and_then(|body| body.color),
            placement: None,
            sheet: false,
            brep: nodecraft_step::record_of_brep(solid.brep())?,
        });
    }
    let mut options = options.clone();
    if options.product_name.trim().is_empty() {
        options.product_name = stem;
    }
    nodecraft_step::write_step_with_options(path, &bodies, &options)
}

impl<'req, 'prog> Eval<'req, 'prog> {
    /// Cache a display mesh only when the node itself has no recorded error.
    /// Bypassed meshes from failed nodes must never become healthy cache hits.
    #[cfg(test)]
    pub(crate) fn cache_mesh_result(
        &mut self,
        node_id: &str,
        cache_key: (String, String),
        mesh: &Mesh,
    ) {
        if let Some(error) = self.errors.get(node_id) {
            self.cache.insert_failure(cache_key, error.clone());
        } else {
            self.cache.remove_failure(&cache_key);
        }
        let _ = mesh;
    }

    /// Evaluate and tessellate a Part display target without evaluating any
    /// upstream node as a mesh.
    #[cfg(test)]
    pub(crate) fn evaluate_part_display(&mut self, node_id: &str) -> Option<Mesh> {
        let node = self.request.nodes.iter().find(|node| node.id == node_id)?.clone();
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_is_fresh = node_cache_is_fresh(&node);
        let mesh_quality = nodecraft_mesh::display_tolerance();
        // Preview quality is part of the display identity: otherwise a camera
        // quality change could reuse the previous merged mesh.
        let quality_key = match self.request.preview_detail {
            Some(detail) => format!("p{detail}"),
            None => format!("q{mesh_quality:.6}"),
        };
        let cache_key = (format!("{node_id}@{quality_key}"), node_key);
        if !self.bypass_cache && cache_is_fresh && let Some(error) = self.cache.failure(&cache_key) {
            self.cache_hits += 1;
            self.cache.note_failure_hit();
            self.errors.insert(node_id.into(), error);
            if let Some(source) = bypass_source(&node, &self.request.connections)
                && let Some(mesh) = self.evaluate(&source)
            {
                self.bypassed.insert(node_id.into());
                self.objects.insert(node_id.into(), mesh.clone());
                return Some(mesh);
            }
            return None;
        }
        let started = Instant::now();
        self.cache_misses += 1;
        self.emit_progress(node_id, ProgressPhase::Start);
        let part_started = Instant::now();
        let kernel_before = crate::kernel_pool::kernel_run_nanos();
        let result = self
            .evaluate_part(node_id)
            .and_then(|part| {
                let wall = part_started.elapsed().as_secs_f64() * 1000.0;
                let kernel = (crate::kernel_pool::kernel_run_nanos() - kernel_before) as f64 / 1.0e6;
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!(
                        "[pool] part eval {node_id} wall_ms={wall:.2} kernel_ms={kernel:.2} ratio={:.2}",
                        if wall > 0.0 { kernel / wall } else { 0.0 }
                    );
                }
                self.brep_timings.insert(node_id.into(), part_started.elapsed().as_secs_f64() * 1000.0);
                self.part_display_mesh(node_id, &part)
            });
        match result {
            Ok(mesh) => {
                self.objects.insert(node_id.into(), mesh.clone());
                self.cache_mesh_result(node_id, cache_key, &mesh);
                self.order.push(node_id.into());
                self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
                self.emit_progress(node_id, ProgressPhase::Complete);
                Some(mesh)
            }
            Err(error) => {
                self.cache.insert_failure(cache_key.clone(), error.clone());
                self.record_error(node_id, error);
                if let Some(source) = bypass_source(&node, &self.request.connections)
                    && let Some(mesh) = self.evaluate(&source)
                {
                    self.bypassed.insert(node_id.into());
                    self.objects.insert(node_id.into(), mesh.clone());
                    self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
                    self.emit_progress(node_id, ProgressPhase::Complete);
                    return Some(mesh);
                }
                self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
                self.emit_progress(node_id, ProgressPhase::Complete);
                None
            }
        }
    }

    /// Record a node error unless the run has been cancelled.
    fn record_error(&mut self, node_id: &str, error: String) {
        if !crate::kernel_pool::current_revision_token().is_cancelled() {
            self.errors.insert(node_id.to_owned(), error);
        }
    }

    /// Evaluate a Part target without producing a mesh (batch/multi-target use).
    pub(crate) fn evaluate_part_only(&mut self, node_id: &str) {
        let node = self.request.nodes.iter().find(|node| node.id == node_id);
        let key = node.and_then(|node| node.cache.as_ref()?.key.clone()).unwrap_or_default();
        let cache_is_fresh = node.is_some_and(node_cache_is_fresh);
        if !self.bypass_cache
            && cache_is_fresh
            && let Some(part) = self.cache.part(&(node_id.to_owned(), key))
        {
            self.cache.note_part_hit();
            self.cache_hits += 1;
            self.part_cache_hits += 1;
            // Keep the exact result visible to the display service on a cache hit.
            self.parts.insert(node_id.to_owned(), part);
            return;
        }
        self.cache_misses += 1;
        let started = Instant::now();
        let kernel_before = crate::kernel_pool::kernel_run_nanos();
        self.emit_progress(node_id, ProgressPhase::Start);
        let part = self.evaluate_part(node_id);
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            let wall = started.elapsed().as_secs_f64() * 1000.0;
            let kernel = (crate::kernel_pool::kernel_run_nanos() - kernel_before) as f64 / 1.0e6;
            eprintln!(
                "[pool] eval {node_id} request={} wall_ms={wall:.2} kernel_ms={kernel:.2} ratio={:.2}",
                self.request.request_id,
                if wall > 0.0 { kernel / wall } else { 0.0 }
            );
        }
        match part {
            Ok(_) => {
                self.order.push(node_id.into());
                self.brep_timings.insert(
                    node_id.into(),
                    started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            Err(error) => {
                self.errors.insert(node_id.into(), error);
            }
        }
        self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
        self.emit_progress(node_id, ProgressPhase::Complete);
    }

    /// Tessellate the final exact Part once, using standard display density.
    #[cfg(test)]
    pub(crate) fn part_display_mesh(&mut self, node_id: &str, part: &Part) -> Result<Mesh, String> {
        self.part_mesh_asset(node_id, part).map(|asset| asset.merged())
    }

    #[cfg(test)]
    pub(crate) fn part_mesh_asset(&mut self, node_id: &str, part: &Part) -> Result<crate::display::MeshAsset, String> {
        if part.bodies.is_empty() && part.sketches.is_empty() {
            return Err("Part has no geometry to display".into());
        }
        let token = crate::kernel_pool::current_revision_token();
        if token.is_cancelled() {
            return Err("evaluation cancelled".into());
        }
        let mut asset = crate::display::MeshAsset::default();
        asset.instances.reserve(part.bodies.len() + part.sketches.len());
        if !part.bodies.is_empty() {
            let tolerance = nodecraft_mesh::display_tolerance();
            let content_hash = self.request.nodes.iter()
                .find(|node| node.id == node_id)
                .and_then(|node| node.cache.as_ref()?.key.clone())
                .unwrap_or_default();
            let preview_detail = self.request.preview_detail;
            let key = crate::display::MeshKey::new(node_id, content_hash.clone(), tolerance, preview_detail);
            // The plan cache key is the body's provenance (its producing node's
            // content seed), not the target's cache key: two boolean nodes that
            // differ only by mode (e.g. subtract vs subtract-reverse) have no
            // cache key but produce different geometry, and must not share a
            // display plan. Bodies with the same source still share a plan.
            let plan_key = |index: usize| {
                let body = &part.bodies[index];
                nodecraft_exact::BodyKey::new(
                    body.source.unwrap_or_else(|| fnv_seed(&content_hash)),
                    body.source.map_or(index as u32, |_| body.source_body),
                )
            };
            let mut outputs = BTreeMap::new();
            let mut pending = Vec::new();
            // Placement-only edits reuse each body's local tessellation. Only
            // the transform is reapplied; the B-rep is not tessellated again.
            for (index, body) in part.bodies.iter().enumerate() {
                let transforms = body_transforms(body.body.brep(), part, index);
                let body_key = plan_key(index);
                if !self.bypass_cache
                    && let Some(cached) = self.display_plans.get(body_key, display_quality(tolerance, preview_detail))
                {
                    let display = nodecraft_layers::evaluate_plan(&cached.plan, &cached.edge_ids);
                    let mut mesh = display.mesh;
                    apply_brep_face_colors(body.body.brep(), &mut mesh);
                    outputs.insert(index, crate::display::MeshInstance {
                        mesh: std::sync::Arc::new(mesh),
                        plan: Some(cached.plan),
                        plan_edge_ids: cached.edge_ids,
                        stats: Some(cached.stats),
                        transforms: transforms.iter().map(crate::display::DisplayTransform::from).collect(),
                    });
                } else {
                    pending.push((index, body.body.shared(), transforms));
                }
            }
            if !pending.is_empty() {
                let tessellation = crate::tessellation::shared_tessellation();
                let limit = if crate::workload_profiler_enabled() {
                    crate::workload::profile_request(self.request)
                        .mesh_bodies
                        .max(1)
                        .min(tessellation.workers().max(1) * 2)
                } else {
                    tessellation.workers().max(1) * 2
                };
                let mut pending = pending.into_iter();
                let mut running: std::collections::VecDeque<(
                    usize,
                    Arc<nodecraft_solidcore::solidcore::topology::Brep>,
                    Vec<BodyTransform>,
                    std::sync::mpsc::Receiver<crate::tessellation::MeshResult>,
                )> = std::collections::VecDeque::new();
                loop {
                    if token.is_cancelled() {
                        return Err("evaluation cancelled".into());
                    }
                    while running.len() < limit {
                        let Some((index, body, transforms)) = pending.next() else { break };
                        let revision = crate::display::MeshRevision::new(self.request.request_id, token.clone());
                        let request = crate::display::MeshRequest {
                            key: key.clone(),
                            revision: self.request.request_id,
                            body: crate::display::MeshBody {
                                brep: body.clone(),
                                transforms: transforms.iter().map(crate::display::DisplayTransform::from).collect(),
                            },
                        };
                        running.push_back((index, body, transforms, tessellation.submit(request, revision)));
                    }
                    let Some((index, body, transforms, receiver)) = running.pop_front() else { break };
                    let result = receiver.recv().map_err(|_| "tessellation worker stopped".to_owned())?;
                    if result.revision != self.request.request_id {
                        continue;
                    }
                    let mut result_asset = result.asset?;
                    let Some(instance) = result_asset.instances.pop() else {
                        return Err("tessellation returned no instance".into());
                    };
                    let plan = instance.plan.clone();
                    let stats = instance.stats.clone();
                    let plan_for_cache = plan.clone().unwrap_or_default();
                    self.display_plans.insert(
                        plan_key(index),
                        display_quality(tolerance, preview_detail),
                        plan_for_cache,
                        instance.plan_edge_ids.clone(),
                        stats.clone().unwrap_or_default(),
                    );
                    let mut mesh = (*instance.mesh).clone();
                    apply_brep_face_colors(body.as_ref(), &mut mesh);
                    outputs.insert(index, crate::display::MeshInstance {
                        mesh: std::sync::Arc::new(mesh),
                        plan,
                        plan_edge_ids: instance.plan_edge_ids.clone(),
                        stats,
                        transforms: transforms.iter().map(crate::display::DisplayTransform::from).collect(),
                    });
                }
            }
            asset.instances.extend(outputs.into_values());
        }
        for sketch in &part.sketches {
            // Sketch and spline profiles display as their boundary curves.
            let mut mesh = spline_display_mesh(&sketch.output.sketch);
            let mut transforms = Vec::with_capacity(sketch.transforms.len() + 1);
            transforms.push(crate::display::DisplayTransform::Place { offset: sketch.output.offset, rotation: [0.0; 3] });
            transforms.extend(sketch.transforms.iter().map(crate::display::DisplayTransform::from));
            mesh = crate::display::MeshAsset {
                instances: vec![crate::display::MeshInstance { mesh: std::sync::Arc::new(mesh), plan: None, plan_edge_ids: Vec::new(), stats: None, transforms }],
            }.merged();
            asset.instances.push(crate::display::MeshInstance {
                mesh: std::sync::Arc::new(mesh),
                plan: None,
                plan_edge_ids: Vec::new(),
                stats: None,
                transforms: Vec::new(),
            });
        }
        for instance in &asset.instances {
            if let Some(stats) = &instance.stats {
                for line in &stats.diagnostics {
                    self.tessellation_diagnostics.push(format!("{node_id}: {line}"));
                }
            }
        }
        Ok(asset)
    }

    #[cfg(test)]
    pub(crate) fn evaluate(&mut self, node_id: &str) -> Option<Mesh> {
        // A newer edit cancels the display pass. Check between nodes so the
        // exact worker is not held up by the rest of a stale graph.
        if crate::kernel_pool::current_revision_token().is_cancelled() {
            return None;
        }
        if let Some(mesh) = self.objects.get(node_id) {
            return Some(mesh.clone());
        }
        // Failed nodes are terminal for this pass; otherwise every branch/target
        // that reaches them re-runs the node and restarts its progress animation.
        if self.errors.contains_key(node_id) {
            return None;
        }
        if self.visiting.iter().any(|id| id == node_id) {
            self.errors.insert(node_id.into(), "Cycle detected".into());
            return None;
        }
        let node = node(self.request, node_id)?.clone();
        if node.suppressed {
            // Keep the pass-through mesh addressable by the suppressed node's
            // own id; display payloads, picking and validation all look it up
            // by that id.
            let mesh = suppression_source(&node, &self.request.connections)
                .and_then(|source| self.evaluate(&source));
            if let Some(mesh) = &mesh {
                self.objects.insert(node_id.into(), mesh.clone());
            }
            return mesh;
        }
        self.visiting.push(node_id.into());
        let started = Instant::now();
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        if node.node_type == "sketch" {
            self.ensure_sketch_output(&node);
        }
        let mesh_quality = nodecraft_mesh::display_tolerance();
        let cache_key = (format!("{node_id}@q{mesh_quality:.6}"), node_key.clone());
        // A dirty node must be recomputed even if its old cache key is still present
        // (for example after "re-evaluate graph" / render all).
        let cache_is_fresh = node_cache_is_fresh(&node);
        if !self.bypass_cache && cache_is_fresh && let Some(error) = self.cache.failure(&cache_key) {
            self.cache_hits += 1;
            self.cache.note_failure_hit();
            self.errors.insert(node_id.into(), error);
            if let Some(source) = bypass_source(&node, &self.request.connections)
                && let Some(mesh) = self.evaluate(&source)
            {
                self.bypassed.insert(node_id.into());
                self.objects.insert(node_id.into(), mesh.clone());
                self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
                self.visiting.pop();
                return Some(mesh);
            }
            self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
            self.visiting.pop();
            return None;
        }
        self.cache_misses += 1;
        // Supported modeling nodes have exactly one semantic implementation:
        // evaluate_part. The mesh path is now only an adapter for legacy/preview
        // nodes, so a supported node reached from a mesh-only consumer still uses
        // the same Geometry evaluator.
        if !self.request.mesh_targets.as_ref().is_some_and(Vec::is_empty)
            && part_supported(&node.node_type)
            && let Ok(part) = self.evaluate_part(node_id)
            && let Ok(mesh) = self.part_display_mesh(node_id, &part)
        {
            self.emit_progress(node_id, ProgressPhase::Start);
            self.objects.insert(node_id.into(), mesh.clone());
            self.cache_mesh_result(node_id, cache_key, &mesh);
            self.order.push(node_id.into());
            self.emit_progress(node_id, ProgressPhase::Complete);
            self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
            self.visiting.pop();
            return Some(mesh);
        }
        // Evaluate inputs first so progress starts in upstream-to-downstream order.
        let input_sources: Vec<String> = self.request.connections.iter().filter(|link| link.to == node_id).map(|link| link.from.clone()).collect();
        for source in input_sources { self.evaluate(&source); }
        self.emit_progress(node_id, ProgressPhase::Start);
        // Do not spend time computing a node whose input already failed.
        let failed_input = self.request.connections.iter()
            .filter(|link| link.to == node_id)
            .filter(|link| !self.bypassed.contains(&link.from))
            .find_map(|link| self.errors.get(&link.from).map(|error| (link.from.clone(), error.clone())));
        if let Some((source, error)) = failed_input {
            self.errors.insert(node_id.into(), format!("Input {source} failed: {error}"));
            self.emit_progress(node_id, ProgressPhase::Complete);
            self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
            self.visiting.pop();
            return None;
        }
        let mesh = match node.node_type.as_str() {
            "stepimporter" => {
                let path = node.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
                if path.is_empty() {
                    Err("Choose a STEP file for this Import STEP node.".into())
                } else {
                    let key = import_cache_key(path);
                    let default_quality = (nodecraft_mesh::display_tolerance()
                        - nodecraft_mesh::DEFAULT_DISPLAY_TOLERANCE).abs() < 1.0e-9;
                    if !self.bypass_cache && default_quality && node_cache_is_fresh(&node) && let Some(import) = self.cache.import(&key) {
                        self.cache.note_import_hit();
                        self.step_diagnostics.insert(node_id.into(), import.diagnostics.clone());
                        Ok(import.mesh.clone())
                    } else {
                        match import_step(path) {
                            Ok(import) => {
                                self.step_diagnostics.insert(node_id.into(), import.diagnostics.clone());
                                if default_quality {
                                    self.cache.insert_import(key, Arc::new(import.clone()));
                                }
                                Ok(import.mesh)
                            }
                            Err(error) => Err(stepio_error(path, error)),
                        }
                    }
                }
            }
            "solid" | "primitive" | "box" | "cylinder" => {
                // Mesh-only primitive graphs must not build an unused exact BRep.
                // Boolean/chamfer/exporter ancestors still need the eager build so
                // their operand cache and progress ordering stay stable.
                if self.needs_brep.contains(&node.id) {
                    let _ = self.cached_primitive_brep(&node);
                }
                let kind = node
                    .values
                    .get("primitiveType")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(&node.node_type);
                match kind {
                    // Closed primitives have exact B-reps, so their display
                    // mesh goes through the same upstream tessellator as every
                    // evaluated Part. Wedge and partial primitives still need
                    // the local mesh path until exact closed B-reps exist.
                    "box" | "cylinder" | "cone" | "sphere" | "torus" | "pyramid"
                    | "sphereNoPole" | "partialSphere" | "partialTorus" | "wedge" => self
                        .cached_primitive_brep(&node)
                        .and_then(|brep| display_mesh_for_brep(&brep)),
                    _ => Err(format!("unsupported primitive kind {kind:?}")),
                }
            }
            "exporter" => {
                let Some(link) = source(self.request, node_id) else {
                    self.errors.insert(node_id.into(), "Export node has no shape input.".into());
                    self.visiting.pop();
                    return None;
                };
                let mesh = self.evaluate(&link.from);
                if !self.request.run_exporters {
                    // Exporting is an explicit action. Ordinary preview/evaluation
                    // must not rewrite files while metadata is being typed.
                    self.visiting.pop();
                    return mesh;
                }
                let directory = node.values.get("directory").and_then(serde_json::Value::as_str).unwrap_or("");
                let root = std::env::var("NODECRAFT_PROJECT_ROOT").unwrap_or_else(|_| ".".into());
                let project_folder = self.request.project_directory.as_deref().filter(|value| !value.trim().is_empty()).map(std::path::PathBuf::from);
                let mut output = if directory.is_empty() {
                    project_folder.unwrap_or_else(|| std::path::PathBuf::from(root.clone()).join("output"))
                } else {
                    std::path::PathBuf::from(directory)
                };
                if output.is_relative() { output = std::path::PathBuf::from(root.clone()).join(output); }
                let stem = node.values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("model").trim_end_matches(".step").trim();
                let stem = if stem.is_empty() { "model" } else { stem };
                let path = output.join(format!("{stem}.step"));
                let path_text = path.to_str().unwrap_or("model.step");
                let result = if let Err(error) = std::fs::create_dir_all(&output) {
                    Err(format!("{}: {error}", output.display()))
                } else {
                    match crate::step_options(&node) {
                        Err(error) => Err(error),
                        Ok(options) => {
                            let format = node.values.get("format").and_then(serde_json::Value::as_str);
                            let source_kind = self
                                .request
                                .nodes
                                .iter()
                                .find(|item| item.id == link.from)
                                .map(|item| item.node_type.as_str());
                            let imported = crate::resolve_part_chain(self.request, &link.from)
                                .filter(|(source, _, _)| source.node_type == "stepimporter");
                            if format == Some("faceted") {
                                match &mesh {
                                    Some(mesh) => crate::write_faceted_from_mesh(path_text, mesh, &options),
                                    None => Err("Export input was not evaluated.".into()),
                                }
                            } else if source_kind == Some("combine") {
                                crate::export_combine_assembly(self, &link.from, &node, path_text)
                            } else if imported.is_some() {
                                crate::export_imported_step(self.request, &link.from, &node, path_text)
                            } else {
                                match crate::exact_operand_breps(self, &link.from) {
                                    Ok(breps) if !breps.is_empty() => {
                                        let written = nodecraft_step::write_breps_exact_with_options(path_text, &breps, &options);
                                        match written {
                                            Ok(()) => Ok(()),
                                            Err(error) => match &mesh {
                                                Some(mesh) => crate::write_faceted_from_mesh(path_text, mesh, &options),
                                                None => Err(error),
                                            },
                                        }
                                    }
                                    Ok(_) => match &mesh {
                                        Some(mesh) => crate::write_faceted_from_mesh(path_text, mesh, &options),
                                        None => Err("Export input produced no solids.".into()),
                                    },
                                    Err(error) => {
                                        // The node does not expose a single exact operand (a
                                        // Circular Pattern is the common case), but its Part is
                                        // already evaluated and carries every body with its own
                                        // placement and metadata. Export that Part instead of
                                        // collapsing the result to a faceted display mesh.
                                        let part_export = self
                                            .parts
                                            .get(&link.from)
                                            .cloned()
                                            .ok_or_else(|| error.clone())
                                            .and_then(|part| {
                                                export_part_bodies(&part, path_text, &options)
                                            });
                                        match part_export {
                                            Ok(()) => Ok(()),
                                            Err(part_error) => match &mesh {
                                                Some(mesh) => crate::write_faceted_from_mesh(path_text, mesh, &options),
                                                None => Err(part_error),
                                            },
                                        }
                                    },
                                }
                            }
                        }
                    }
                };
                if let Err(error) = result { self.errors.insert(node_id.into(), stepio_error(path_text, error)); }
                mesh.ok_or_else(|| "Export input was not evaluated.".into())
            }
            "boolean" | "fuse" => (|| -> Result<Mesh, String> {
                let a = self.request.connections.iter().find(|link| link.to == node_id && link.input == "a");
                let b = self.request.connections.iter().find(|link| link.to == node_id && link.input == "b");
                match (a, b) {
                    (Some(a), None) => return self.evaluate(&a.from).ok_or_else(|| "Boolean has no evaluated A operand".into()),
                    (None, Some(b)) => return self.evaluate(&b.from).ok_or_else(|| "Boolean has no evaluated B operand".into()),
                    (None, None) => return Err("Boolean has no connected operand".into()),
                    (Some(_), Some(_)) => {}
                }
                let a = a.expect("checked");
                let b = b.expect("checked");
                let a_suppressed = effective_output_suppressed(self.request, &a.from);
                let b_suppressed = effective_output_suppressed(self.request, &b.from);
                if a_suppressed || b_suppressed {
                    return if a_suppressed && b_suppressed {
                        Err("Boolean has no active operand".into())
                    } else if a_suppressed {
                        self.evaluate(&b.from).ok_or_else(|| "Boolean has no evaluated B operand".into())
                    } else {
                        self.evaluate(&a.from).ok_or_else(|| "Boolean has no evaluated A operand".into())
                    };
                }
                let (op, reverse) = boolean_mode(&node.values)?;
                let (a, b) = if reverse { (b.from.clone(), a.from.clone()) } else { (a.from.clone(), b.from.clone()) };
                let a_breps: Vec<nodecraft_ops::SolidBody> = self.operand_breps(&a)?.into_iter().map(nodecraft_ops::SolidBody::new).collect();
                let b_breps: Vec<nodecraft_ops::SolidBody> = self.operand_breps(&b)?.into_iter().map(nodecraft_ops::SolidBody::new).collect();
                let result = crate::node_task::execute_exact_warned(
                    crate::node_task::ExactNodeTask::Boolean {
                        op,
                        a: a_breps,
                        b: b_breps,
                    },
                    node_id,
                    fnv_seed(node_id),
                )?;
                if !result.warnings.is_empty() {
                    self.warnings.entry(node_id.to_owned()).or_default().extend(result.warnings);
                }
                let result = match result.output {
                    crate::node_task::NodeOutput::Shape(shape) => shape.shared(),
                    // Nothing left: the node keeps its amber warning and draws nothing.
                    crate::node_task::NodeOutput::Empty => return Ok(Mesh::default()),
                    _ => return Err("boolean task returned no shape".into()),
                };
                self.breps.insert(node_id.into(), result.clone());
                if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                    self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                }
                display_mesh_for_brep(&result)
            })(),
            "extrude" | "loft" | "surfaceloft" => (|| -> Result<Mesh, String> {
                // These nodes have exactly one implementation: the exact Part
                // path. Keeping a second, node-type-driven display evaluator
                // here made a translated sketch extrude disagree with the
                // exported solid. Display only tessellates the Part, then
                // publishes its body for the legacy B-rep consumers below.
                let part = self.evaluate_part(node_id)?;
                let mesh = self.part_display_mesh(node_id, &part)
                    .map_err(|error| format!("{} display mesh: {error}", node.node_type))?;
                let mut solids = part.realized_solids()?;
                if solids.len() == 1 {
                    let result = Arc::new(solids.pop().expect("one solid"));
                    self.breps.insert(node_id.into(), result.clone());
                    if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                        self.cache.insert_brep((node_id.to_string(), cache), result);
                    }
                }
                Ok(mesh)
            })(),
            "groupinput" => Ok(Mesh::default()),
            "groupoutput" => source(self.request, node_id)
                .and_then(|link| self.evaluate(&link.from))
                .ok_or_else(|| "Group output has no connected input.".into()),
            "scatter" => (|| -> Result<Mesh, String> {
                // Scatter is an exact Part generator: produce placed BRep
                // instances through the Part path, then tessellate them. The
                // display path must not maintain a separate mesh-only scatter.
                let part = self.evaluate_part(node_id)?;
                self.part_display_mesh(node_id, &part)
                    .map_err(|error| format!("Scatter display mesh: {error}"))
            })(),
            "circularpattern" => (|| -> Result<Mesh, String> {
                // Circular Pattern is the same exact Part generator shape as
                // Scatter: instances come from the Part path, never a separate
                // mesh-only placement.
                let part = self.evaluate_part(node_id)?;
                self.part_display_mesh(node_id, &part)
                    .map_err(|error| format!("Circular Pattern display mesh: {error}"))
            })(),
            "combine" => {
                let mut meshes = Vec::new();
                for link in self.request.connections.iter().filter(|link| link.to == node_id && link.input == "shape") {
                    if let Some(mesh) = self.evaluate(&link.from) {
                        if !mesh.vertices.is_empty() { meshes.push(mesh); }
                    }
                }
                Ok(merge_meshes(meshes))
            }
            "chamfer" | "fillet" => (|| -> Result<Mesh, String> {
                let link = source(self.request, node_id).ok_or("Edge node has no shape input")?;
                // A fresh edge node has no selected edges yet: pass the input through
                // until the user picks edges, just like extrude without a face.
                let has_edges = node.values.get("edges").and_then(serde_json::Value::as_array)
                    .is_some_and(|edges| !edges.is_empty());
                if !has_edges {
                    return self.evaluate(&link.from).ok_or_else(|| "Edge source did not evaluate".to_owned());
                }
                // Display the exact BRep result. The mesh preview path is
                // planar-only and cannot represent a curved chamfer or G2/G3
                // fillet, even when the exact result built successfully.
                let part = self.evaluate_part(node_id)?;
                self.part_display_mesh(node_id, &part)
                    .map_err(|error| format!("Edge display mesh: {error}"))
            })(),
            "look" => match source(self.request, node_id) {
                Some(link) => self
                    .evaluate(&link.from)
                    .ok_or_else(|| "Look input was not evaluated".to_owned()),
                None => Err("Look has no shape input".to_owned()),
            },
            "translate" => {
                let Some(link) = source(self.request, node_id) else {
                    self.errors.insert(node_id.into(), "Transform has no shape input.".into());
                    self.visiting.pop();
                    return None;
                };
                match self.evaluate(&link.from) {
                    Some(mesh) => {
                        let (offset, rotation) = crate::translate_placement(&node.values);
                        Ok(crate::mesh_placement(&mesh, offset, rotation))
                    }
                    None => Err("Transform input was not evaluated.".into()),
                }
            }
            "mirror" => (|| -> Result<Mesh, String> {
                let link = source(self.request, node_id).ok_or("Mirror has no shape input.")?;
                let mesh = self.evaluate(&link.from).ok_or("Mirror input was not evaluated.")?;
                let mirrored = mirror_mesh(&mesh, &node.values);
                Ok(if mirror_combine(&node.values) {
                    nodecraft_mesh::merge_meshes(vec![mesh, mirrored])
                } else {
                    mirrored
                })
            })(),
            "sketch" => (|| -> Result<Mesh, String> {
                // Sketches are edited in the 2D sketch overlay. Their typed output
                // is the solved profile set consumed by downstream features; the
                // mesh itself is empty.
                let sketch: crate::sketch::Sketch = serde_json::from_value(
                    node.values.get("sketch").cloned().unwrap_or(serde_json::Value::Null),
                )
                .map_err(|error| format!("Sketch values: {error}"))?;
                self.sketches.insert(node_id.into(), Arc::new(SketchOutput::new(sketch)));
                Ok(Mesh::default())
            })(),
            "spline" => (|| -> Result<Mesh, String> {
                let sketch = spline_sketch_from_values(&node.values)?;
                let display = spline_display_mesh(&sketch);
                self.sketches.insert(node_id.into(), Arc::new(SketchOutput::new(sketch)));
                Ok(display)
            })(),
            "time" => {
                // Time is a scalar source consumed by the app; it produces no geometry.
                Ok(Mesh::default())
            }
            "sceneTransform" => {
                // Scene transforms are applied by the viewer from node values. The
                // engine only supplies the unchanged source mesh so the viewer can keep
                // animating without re-tessellating.
                let Some(link) = source(self.request, node_id) else {
                    self.errors.insert(node_id.into(), "Scene Transform has no shape input.".into());
                    self.visiting.pop();
                    return None;
                };
                match self.evaluate(&link.from) {
                    Some(mut mesh) => {
                        // Scene animation is display-only; keep the render path
                        // lightweight and do not carry selectable B-rep edge data.
                        mesh.lightweight = true;
                        mesh.brep_edges.clear();
                        Ok(mesh)
                    }
                    None => Err("Scene Transform input was not evaluated.".into()),
                }
            }
            other => Err(format!("Rust experiment does not implement node type {other:?} yet")),
        };
        self.visiting.pop();
        self.emit_progress(node_id, ProgressPhase::Complete);
        let result = match mesh {
            Ok(mesh) => {
                self.order.push(node_id.into());
                self.cache_mesh_result(node_id, cache_key, &mesh);
                self.objects.insert(node_id.into(), mesh.clone());
                Some(mesh)
            }
            Err(error) => {
                if node.node_type != "exporter" {
                    self.cache.insert_failure(cache_key.clone(), error.clone());
                }
                self.errors.insert(node_id.into(), error);
                None
            }
        };
        self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
        result
    }
}
