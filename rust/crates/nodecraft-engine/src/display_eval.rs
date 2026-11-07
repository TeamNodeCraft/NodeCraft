//! Display evaluation and mesh caching.

use super::*;

/// Display-only tessellation through the upstream layer seam. Exact geometry
/// never passes through this function.
fn display_mesh_for_brep(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> Result<Mesh, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let mut brep = copy_brep(brep)?;
    nodecraft_layers::tessellate_kernel(&mut kernel, &mut brep, 0.05).map(|display| display.mesh)
}

impl<'req, 'prog> Eval<'req, 'prog> {
    /// Cache a display mesh only when the node itself has no recorded error.
    /// Bypassed meshes from failed nodes must never become healthy cache hits.
    pub(crate) fn cache_mesh_result(
        &mut self,
        node_id: &str,
        cache_key: (String, String),
        mesh: &Mesh,
    ) {
        if let Some(error) = self.errors.get(node_id) {
            self.cache.insert_failure(cache_key.clone(), error.clone());
            self.cache.remove_mesh(&cache_key);
        } else {
            self.cache.remove_failure(&cache_key);
            self.cache.insert_mesh(cache_key, mesh.clone());
        }
    }

    /// Evaluate and tessellate a Part display target without evaluating any
    /// upstream node as a mesh.
    pub(crate) fn evaluate_part_display(&mut self, node_id: &str) -> Option<Mesh> {
        let node = self.request.nodes.iter().find(|node| node.id == node_id)?.clone();
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_is_fresh = node_cache_is_fresh(&node);
        let mesh_quality = nodecraft_mesh::display_tolerance();
        let cache_key = (format!("{node_id}@q{mesh_quality:.6}"), node_key);
        if cache_is_fresh && let Some(error) = self.cache.failure(&cache_key) {
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
        if cache_is_fresh && let Some(mesh) = self.cache.mesh(&cache_key) {
            self.cache_hits += 1;
            self.objects.insert(node_id.into(), mesh.clone());
            self.order.push(node_id.into());
            return Some(mesh);
        }
        let started = Instant::now();
        self.cache_misses += 1;
        self.emit_progress(node_id, ProgressPhase::Start);
        let part_started = Instant::now();
        let result = self
            .evaluate_part(node_id)
            .and_then(|part| {
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
                self.cache.remove_mesh(&cache_key);
                self.errors.insert(node_id.into(), error);
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

    /// Evaluate a Part target without producing a mesh (batch/multi-target use).
    pub(crate) fn evaluate_part_only(&mut self, node_id: &str) {
        let node = self.request.nodes.iter().find(|node| node.id == node_id);
        let key = node.and_then(|node| node.cache.as_ref()?.key.clone()).unwrap_or_default();
        let cache_is_fresh = node.is_some_and(node_cache_is_fresh);
        if cache_is_fresh && self.cache.part(&(node_id.to_owned(), key)).is_some() {
            self.cache.note_part_hit();
            self.cache_hits += 1;
            return;
        }
        let started = Instant::now();
        self.emit_progress(node_id, ProgressPhase::Start);
        match self.evaluate_part(node_id) {
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
    pub(crate) fn part_display_mesh(&mut self, node_id: &str, part: &Part) -> Result<Mesh, String> {
        self.part_mesh_asset(node_id, part).map(|asset| asset.merged())
    }

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
            let key = crate::display::MeshKey::new(node_id, content_hash, tolerance);
            let mut outputs = BTreeMap::new();
            let mut pending = Vec::new();
            // Placement-only edits reuse each body's local tessellation. Only
            // the transform is reapplied; the B-rep is not tessellated again.
            for (index, body) in part.bodies.iter().enumerate() {
                let transforms = body_transforms(body.body.as_ref(), part, index);
                if let Some(local) = self.cache.body_mesh(&body.body, tolerance) {
                    let mut mesh = (*local).clone();
                    apply_brep_face_colors(body.body.as_ref(), &mut mesh);
                    outputs.insert(index, crate::display::MeshInstance {
                        mesh: std::sync::Arc::new(mesh),
                        transforms: transforms.iter().map(crate::display::DisplayTransform::from).collect(),
                    });
                } else {
                    pending.push((index, body.body.clone(), transforms));
                }
            }
            if !pending.is_empty() {
                let mesher = crate::mesher::shared_mesher();
                let limit = if crate::workload_profiler_enabled() {
                    crate::workload::profile_request(self.request)
                        .mesh_bodies
                        .max(1)
                        .min(mesher.workers().max(1) * 2)
                } else {
                    mesher.workers().max(1) * 2
                };
                let mut pending = pending.into_iter();
                let mut running: std::collections::VecDeque<(
                    usize,
                    Arc<nodecraft_solidcore::solidcore::topology::Brep>,
                    Vec<BodyTransform>,
                    std::sync::mpsc::Receiver<crate::mesher::MeshResult>,
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
                        running.push_back((index, body, transforms, mesher.submit(request, revision)));
                    }
                    let Some((index, body, transforms, receiver)) = running.pop_front() else { break };
                    let result = receiver.recv().map_err(|_| "mesher worker stopped".to_owned())?;
                    if result.revision != self.request.request_id {
                        continue;
                    }
                    let mut result_asset = result.asset?;
                    let Some(instance) = result_asset.instances.pop() else {
                        return Err("mesher returned no instance".into());
                    };
                    self.cache.insert_body_mesh(body.clone(), tolerance, instance.mesh.clone());
                    let mut mesh = (*instance.mesh).clone();
                    apply_brep_face_colors(body.as_ref(), &mut mesh);
                    outputs.insert(index, crate::display::MeshInstance {
                        mesh: std::sync::Arc::new(mesh),
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
                instances: vec![crate::display::MeshInstance { mesh: std::sync::Arc::new(mesh), transforms }],
            }.merged();
            asset.instances.push(crate::display::MeshInstance {
                mesh: std::sync::Arc::new(mesh),
                transforms: Vec::new(),
            });
        }
        Ok(asset)
    }

    pub(crate) fn evaluate(&mut self, node_id: &str) -> Option<Mesh> {
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
            return suppression_source(&node, &self.request.connections)
                .and_then(|source| self.evaluate(&source));
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
        if cache_is_fresh && let Some(error) = self.cache.failure(&cache_key) {
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
        if cache_is_fresh && let Some(mesh) = self.cache.mesh(&cache_key) {
            self.cache_hits += 1;
            self.visiting.pop();
            self.objects.insert(node_id.into(), mesh.clone());
            // A mesh-cache hit still has to restore the exact result used by
            // exporters and exact consumers.
            let brep_key = (node_id.to_string(), node_key.clone());
            if let Some(brep) = self.cache.brep(&brep_key) {
                self.cache.note_brep_hit();
                self.breps.insert(node_id.into(), brep);
            }
            self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
            return Some(mesh);
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
                    if default_quality && node_cache_is_fresh(&node) && let Some(import) = self.cache.import(&key) {
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
                let number = |key: &str, default: f64| {
                    node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(default)
                };
                match kind {
                    // Closed primitives have exact B-reps, so their display
                    // mesh goes through the same upstream tessellator as every
                    // evaluated Part. Wedge and partial primitives still need
                    // the local mesh path until exact closed B-reps exist.
                    "box" | "cylinder" | "cone" | "sphere" | "torus" | "pyramid"
                    | "sphereNoPole" | "partialSphere" | "partialTorus" | "wedge" => self
                        .cached_primitive_brep(&node)
                        .and_then(|brep| display_mesh_for_brep(&brep)),
                    _ => self.kernel.primitive(kind, &node.values),
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
                let mut current = link.from.clone();
                let mut offset = [0.0f64; 3];
                let mut rotation = [0.0f64; 3];
                let mut primitive = None;
                loop {
                    let Some(item) = self.request.nodes.iter().find(|item| item.id == current) else { break };
                    if item.node_type == "translate" {
                        offset[0] += item.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                        offset[1] += item.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                        offset[2] += item.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                        for (slot, key) in ["rx", "ry", "rz"].iter().enumerate() {
                            rotation[slot] = item.values.get(*key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
                        }
                        let Some(next) = self.request.connections.iter().find(|candidate| candidate.to == current && candidate.input == "shape") else { break };
                        current = next.from.clone();
                    } else {
                        primitive = Some(item);
                        break;
                    }
                }
                let number = |key: &str, default: f64| {
                    primitive.and_then(|item| item.values.get(key)).and_then(serde_json::Value::as_f64).unwrap_or(default)
                };
                let kind = primitive.and_then(|item| item.values.get("primitiveType")).and_then(serde_json::Value::as_str).unwrap_or("");
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
                    let exact_part = if node.values.get("format").and_then(serde_json::Value::as_str) != Some("faceted") {

                        self.evaluate_part(&link.from).ok().and_then(|part| part.realized_solids().ok())
                    } else {
                        None
                    };
                    if let Some(breps) = exact_part.as_ref()
                        && let Ok(()) = nodecraft_step::write_breps_exact(path_text, breps)
                        && nodecraft_step::import_solids(path_text).is_ok()
                        && let Ok(()) = apply_export_metadata(&node, path_text)
                    {
                        Ok(())
                    } else if cacheable_exact_operand(self.request, &link.from)
                        && node.values.get("format").and_then(serde_json::Value::as_str) != Some("faceted")
                        && let Some(brep) = self.cached_exact_brep(&link.from)
                    {
                        let mut brep = brep;
                        let mut heal_kernel = nodecraft_solidcore::kernel::kernel();
                        if nodecraft_solidcore::solidcore::heal::heal_data::p1::api_heal_brep(
                            &mut heal_kernel,
                            Some(&mut brep),
                        )
                        .is_err()
                        {
                            // Keep the evaluated body if the healer cannot run.
                        }
                        let exact = nodecraft_step::write_brep_exact(path_text, &brep);
                        let validated = exact.is_ok()
                            && nodecraft_mesh::import_step_brep(path_text).is_ok();
                        if validated {
                            apply_export_metadata(&node, path_text)
                        } else if let Some(mesh) = &mesh {
                            // Some curved Boolean/skin B-reps do not yet round-trip
                            // through the STEP writer. Keep the file usable by
                            // falling back to the evaluated display mesh.
                            write_faceted_from_mesh(path_text, mesh)
                                .and_then(|_| apply_export_metadata(&node, path_text))
                        } else {
                            exact.and_then(|_| apply_export_metadata(&node, path_text))
                        }
                    } else {
                        let imported_chain = resolve_part_chain(self.request, &link.from)
                        .filter(|(source, _, _)| source.node_type == "stepimporter");
                    let written = if let Some((_, offset, rotation)) = imported_chain {
                        let _ = (offset, rotation);
                        export_imported_step(self.request, &link.from, path_text).and_then(|_| {
                            if node.values.get("format").and_then(serde_json::Value::as_str) == Some("faceted") {
                                match nodecraft_step::read_planar_body(path_text) {
                                    Ok(body) => nodecraft_step::write_faceted_brep(path_text, &body)?,
                                    Err(error) => match &mesh {
                                        Some(mesh) => write_faceted_from_mesh(path_text, mesh)?,
                                        None => return Err(error),
                                    },
                                }
                            }
                            apply_export_metadata(&node, path_text)
                        })
                    } else if matches!(primitive.map(|item| item.node_type.as_str()), Some("solid" | "primitive")) {
                        match kind {
                            "box" => nodecraft_step::write_box(path_text, number("length", 40.0), number("width", 40.0), number("height", 40.0)),
                            "cylinder" => nodecraft_step::write_cylinder(path_text, number("radius", 20.0), number("height", 40.0)),
                            "cone" => nodecraft_step::write_cone(path_text, number("bottomRadius", 20.0), number("topRadius", 0.0), number("height", 40.0)),
                            "sphere" => nodecraft_step::write_sphere(path_text, number("radius", 20.0)),
                            "torus" => nodecraft_step::write_torus(path_text, number("majorRadius", 15.0), number("minorRadius", 5.0)),
                            "wedge" => nodecraft_step::write_wedge(path_text, number("length", 40.0), number("width", 30.0), number("height", 25.0), number("topLength", 0.0)),
                            "pyramid" => nodecraft_step::write_pyramid(path_text, number("length", 40.0), number("height", 30.0)),
                            other => Err(format!("Rust STEP export does not support primitive {other:?}.")),
                        }.and_then(|_| {
                            nodecraft_step::transform_step_file(path_text, rotation, offset)?;
                            let unit = node.values.get("unit").and_then(serde_json::Value::as_str).unwrap_or("mm").to_ascii_lowercase();
                            let target = match unit.as_str() {
                                "cm" | "centimetre" | "centimeter" => Some(nodecraft_step::LengthUnit::Centimetre),
                                "m" | "metre" | "meter" => Some(nodecraft_step::LengthUnit::Metre),
                                "um" | "micrometre" | "micrometer" => Some(nodecraft_step::LengthUnit::Micrometre),
                                "in" | "inch" => Some(nodecraft_step::LengthUnit::Inch),
                                _ => None,
                            };
                            if let Some(target) = target { nodecraft_step::convert_step_units(path_text, nodecraft_step::LengthUnit::Millimetre, target)?; }
                            if node.values.get("format").and_then(serde_json::Value::as_str) == Some("faceted") {
                                match nodecraft_step::read_planar_body(path_text) {
                                    Ok(body) => nodecraft_step::write_faceted_brep(path_text, &body)?,
                                    Err(error) => match &mesh {
                                        Some(mesh) => write_faceted_from_mesh(path_text, mesh)?,
                                        None => return Err(error),
                                    },
                                }
                            }
                            let name = node.values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("solid");
                            let author = node.values.get("author").and_then(serde_json::Value::as_str).unwrap_or("");
                            let organization = node.values.get("organization").and_then(serde_json::Value::as_str).unwrap_or("NodeCraft");
                            nodecraft_step::set_header(path_text, name, author, organization)?;
                            if let Some(schema) = node.values.get("schema").and_then(serde_json::Value::as_str) {
                                nodecraft_step::set_schema(path_text, schema)?;
                            }
                            if let Some(name) = node.values.get("productName").and_then(serde_json::Value::as_str) {
                                nodecraft_step::set_product_name(path_text, name)?;
                            }
                            if let Some(layer) = node.values.get("layer").and_then(serde_json::Value::as_str) {
                                nodecraft_step::set_layer(path_text, layer)?;
                            }
                            if let Some(color) = node.values.get("color").and_then(serde_json::Value::as_str) {
                                let hex = color.trim_start_matches('#');
                                if hex.len() == 6 {
                                    let r = u8::from_str_radix(&hex[0..2], 16).map_err(|error| error.to_string())? as f64 / 255.0;
                                    let g = u8::from_str_radix(&hex[2..4], 16).map_err(|error| error.to_string())? as f64 / 255.0;
                                    let b = u8::from_str_radix(&hex[4..6], 16).map_err(|error| error.to_string())? as f64 / 255.0;
                                    nodecraft_step::set_color(path_text, r, g, b)?;
                                } else { return Err("color must be #RRGGBB".into()); }
                            }
                            Ok(())
                        })
                    } else if (exports_boolean_chain(self.request, &link.from)
                        || self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("boolean")
                        || self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("fuse"))
                        && node.values.get("format").and_then(serde_json::Value::as_str) == Some("faceted")
                    {
                        match &mesh {
                            Some(mesh) => write_faceted_from_mesh(path_text, mesh).and_then(|_| apply_export_metadata(&node, path_text)),
                            None => Err("Boolean input was not evaluated.".into()),
                        }
                    } else if exports_boolean_chain(self.request, &link.from) {
                        export_boolean_chain(self.request, &link.from, path_text).and_then(|_| apply_export_metadata(&node, path_text))
                    } else if self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("boolean") || self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("fuse") {
                        export_boolean_exact(self.request, &link.from, path_text).and_then(|_| apply_export_metadata(&node, path_text))
                    } else if resolve_extrude_chain(self.request, &link.from).is_some() {
                        export_extrude_chain(self.request, &link.from, path_text).and_then(|_| apply_export_metadata(&node, path_text))
                    } else if self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("extrude") {
                        export_extrude_exact(self.request, &link.from, path_text).and_then(|_| apply_export_metadata(&node, path_text))
                    } else if self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("combine") {
                        export_combine_assembly(self.request, &link.from, path_text)
                            .or_else(|_| export_combine_exact(self.request, &link.from, path_text))
                            .and_then(|_| apply_export_metadata(&node, path_text))
                    } else if self.request.nodes.iter().find(|item| item.id == link.from).map(|item| item.node_type.as_str()) == Some("mirror") {
                        match exact_operand_brep(self.request, &link.from) {
                            Ok(brep) => nodecraft_step::write_brep_exact(path_text, &brep)
                                .and_then(|_| apply_export_metadata(&node, path_text)),
                            Err(error) => Err(error),
                        }
                    } else if let Some(source_mesh) = &mesh {
                        nodecraft_step::write_mesh_planar(path_text, &source_mesh.vertices, &source_mesh.faces)
                    } else {
                        Err("Export input was not evaluated.".into())
                    };
                        written
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
                let a_breps = self.operand_breps(&a)?;
                let b_breps = self.operand_breps(&b)?;
                let result = crate::node_task::execute_exact(
                    crate::node_task::ExactNodeTask::Boolean {
                        op: crate::ops_boolean(op),
                        a: a_breps,
                        b: b_breps,
                    },
                    node_id,
                    fnv_seed(node_id),
                )?
                .into_shape()
                .ok_or("boolean task returned no shape")?;
                self.breps.insert(node_id.into(), result.clone());
                if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                    self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                }
                display_mesh_for_brep(&result)
            })(),
            "extrude" => (|| -> Result<Mesh, String> {
                let link = self.request.connections.iter().find(|link| link.to == node_id && (link.input == "profile" || link.input == "shape")).ok_or("Extrude has no profile input.")?;
                let operation = node.values.get("operation").and_then(serde_json::Value::as_str).unwrap_or("join");
                let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
                let source_is_sketch = self
                    .request
                    .nodes
                    .iter()
                    .find(|item| item.id == link.from)
                    .is_some_and(|item| item.node_type == "sketch" || item.node_type == "spline");
                if source_is_sketch {
                    let output = self
                        .sketch_output_for(&link.from)
                        .ok_or("Extrude sketch source did not evaluate")?;
                    let result = Arc::new(sketch_extrude_brep(&output, amount)?);
                    self.breps.insert(node_id.into(), result.clone());
                    if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                        self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                    }
                    return display_mesh_for_brep(&result);
                }
                // Resolve the whole upstream chain through the B-rep cache. The source
                // may be a translate (or Boolean), not just a STEP importer.
                let brep = self.operand_brep(&link.from)?;
                let faces = node.values.get("faces").and_then(serde_json::Value::as_array);
                if faces.is_none_or(|items| items.is_empty()) {
                    // A face is optional until the user picks one. Keep the node clean:
                    // pass the upstream body through and let FACE selection populate it.
                    return self.evaluate(&link.from).ok_or_else(|| "Extrude source did not evaluate".into());
                }
                let reference = node.values.get("faces").and_then(serde_json::Value::as_array).and_then(|values| values.first()).and_then(|value| value.get("id")).and_then(serde_json::Value::as_str).unwrap_or("");
                let index: usize = face_reference_index(reference).ok_or("Extrude needs a selected face reference (face:N).")?;
                let _ = brep.faces().ids().nth(index).ok_or_else(|| format!("Extrude face {index} does not exist"))?;
                let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
                let tool = crate::node_task::execute_exact(
                    crate::node_task::ExactNodeTask::Extrude { brep: copy_brep(&brep)?, face_index: index, amount },
                    node_id,
                    fnv_seed(node_id),
                )?
                .into_brep()
                .ok_or_else(|| "extrude task returned no shape".to_owned())?;
                let preview = display_mesh_for_brep(&tool).ok();
                let result = match operation {
                    "new" => tool,
                    "remove" => crate::boolean_breps_ops(&brep, &tool, nodecraft_bool::BooleanOp::Difference)?,
                    "intersect" => crate::boolean_breps_ops(&brep, &tool, nodecraft_bool::BooleanOp::Intersection)?,
                    _ => crate::boolean_breps_ops(&brep, &tool, nodecraft_bool::BooleanOp::Union)?,
                };
                let result = Arc::new(result);
                self.breps.insert(node_id.into(), result.clone());
                if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                    self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                }
                let mut display = display_mesh_for_brep(&result)?;
                display.feature_preview = preview.filter(|mesh| !mesh.vertices.is_empty()).map(Box::new);
                display.feature_preview_operation = Some(operation.to_string());
                Ok(display)
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
                let brep = self.operand_brep(&link.from)?;
                let kernel = nodecraft_solidcore::kernel::kernel();
                let mut values = node.values.clone();
                values["mode"] = serde_json::json!(if node.node_type == "fillet" { "fillet" } else { "chamfer" });
                nodecraft_blend::edge_treatment_preview(&kernel, &brep, &values)
                    .map_err(|error| format!("Edge preview: {error}"))
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
                    Some(mesh) => self.kernel.translate(&mesh, &node.values),
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
            "loft" => (|| -> Result<Mesh, String> {
                let bottom = input_source(self.request, node_id, "bottom").ok_or("Loft has no bottom section")?.from.clone();
                let top = input_source(self.request, node_id, "top").ok_or("Loft has no top section")?.from.clone();
                let _ = self.evaluate(&bottom);
                let _ = self.evaluate(&top);
                let ruled = node.values.get("ruled").and_then(serde_json::Value::as_f64).unwrap_or(0.0) >= 0.5;
                let bottom_output = self.sketch_output_for(&bottom);
                let top_output = self.sketch_output_for(&top);
                let result = if let (Some(bottom_output), Some(top_output)) = (bottom_output, top_output) {
                    crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::LoftSketch { bottom: bottom_output, top: top_output, ruled },
                        node_id,
                        fnv_seed(node_id),
                    )?
                    .into_shape()
                    .ok_or_else(|| "Loft task returned no shape".to_owned())?
                } else {
                    let mut oracle = nodecraft_solidcore::kernel::kernel();
                    Arc::new(loft_exact(self.request, &mut oracle, &bottom, &top, ruled, true)?)
                };
                self.breps.insert(node_id.into(), result.clone());
                if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                    self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                }
                display_mesh_for_brep(&result)
                    .map_err(|error| format!("Loft display mesh: {error}"))
            })(),
            "surfaceloft" => (|| -> Result<Mesh, String> {
                let section_ids: Vec<String> = self.request.connections.iter()
                    .filter(|link| link.to == node_id && link.input == "sections")
                    .map(|link| link.from.clone())
                    .collect();
                if section_ids.len() < 2 {
                    return Err("Surface Loft needs at least two section connections".into());
                }
                let mut sections = Vec::with_capacity(section_ids.len());
                for section_id in &section_ids {
                    if !self.request.connections.iter().any(|link| &link.from == section_id && link.to == node_id) {
                        return Err(format!("Surface Loft section {section_id} is not connected"));
                    }
                    let _ = self.evaluate(section_id);
                    let output = self.sketch_output_for(section_id)
                        .ok_or_else(|| format!("Surface Loft section {section_id} must be a sketch or spline curve"))?;
                    sections.push(output);
                }
                let ruled = node.values.get("ruled").and_then(serde_json::Value::as_f64).unwrap_or(0.0) >= 0.5;
                let result = crate::node_task::execute_exact(
                    crate::node_task::ExactNodeTask::SurfaceLoft { sections, ruled },
                    node_id,
                    fnv_seed(node_id),
                )?
                .into_shape()
                .ok_or_else(|| "Surface Loft task returned no shape".to_owned())?;
                self.breps.insert(node_id.into(), result.clone());
                if let Some(cache) = node.cache.as_ref().and_then(|cache| cache.key.clone()) {
                    self.cache.insert_brep((node_id.to_string(), cache), result.clone());
                }
                display_mesh_for_brep(&result)
                    .map_err(|error| format!("Surface Loft display mesh: {error}"))
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
                self.cache.remove_mesh(&cache_key);
                self.errors.insert(node_id.into(), error);
                None
            }
        };
        self.timings.insert(node_id.into(), started.elapsed().as_secs_f64() * 1000.0);
        result
    }
}
