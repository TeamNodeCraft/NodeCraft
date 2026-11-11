//! Exact Part evaluation: B-rep construction, placement and caches.

use super::*;

impl<'req, 'prog> Eval<'req, 'prog> {
    pub(crate) fn cached_primitive_brep_arc(&mut self, node: &Node) -> Result<Arc<nodecraft_solidcore::solidcore::topology::Brep>, String> {
        if let Some(brep) = self.breps.get(&node.id) {
            return Ok(brep.clone());
        }
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_key = (node.id.clone(), node_key);
        let cache_is_fresh = node_cache_is_fresh(node);
        if !self.bypass_cache && cache_is_fresh && let Some(brep) = self.cache.brep(&cache_key) {
            self.cache.note_brep_hit();
            self.cache_hits += 1;
            self.brep_cache_hits += 1;
            self.breps.insert(node.id.clone(), brep.clone());
            return Ok(brep);
        }
        let seed = fnv_seed(&node.id);
        let brep = crate::node_task::execute_exact(
            crate::node_task::ExactNodeTask::Primitive(node.clone()),
            &node.id,
            seed,
        )?
        .into_shape()
        .ok_or("primitive task returned no shape")?;
        self.breps.insert(node.id.clone(), brep.clone());
        self.cache.insert_brep(cache_key, brep.clone());
        Ok(brep)
    }

    pub(crate) fn cached_primitive_brep(&mut self, node: &Node) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
        let brep = self.cached_primitive_brep_arc(node)?;
        snapshot_brep(&brep)
    }

    /// Latest evaluated exact B-rep for a chain, including Translate placements.
    /// This is what exporters and downstream consumers should use; the graph is
    /// not re-resolved once a result exists.
    pub(crate) fn cached_exact_brep(&self, start: &str) -> Option<nodecraft_solidcore::solidcore::topology::Brep> {
        let mut current = start.to_owned();
        let mut placements = Vec::new();
        for _ in 0..64 {
            let node = self.request.nodes.iter().find(|node| node.id == current)?;
            if node.node_type == "translate" {
                let offset = [
                    node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                    node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                    node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                ];
                let rotation = [
                    node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                    node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                    node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0),
                ];
                placements.push((offset, rotation));
                let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape")?;
                current = link.from.clone();
                continue;
            }
            let mut brep = copy_brep(self.breps.get(&current)?.as_ref()).ok()?;
            for (offset, rotation) in placements.iter().rev() {
                apply_brep_placement(&mut brep, *offset, *rotation).ok()?;
            }
            return Some(brep);
        }
        None
    }

    /// Exact B-rep for a Boolean operand, memoized per evaluation pass. Without
    /// this every fuse reconstructs its entire upstream B-rep tree.
    #[cfg(test)]
    pub(crate) fn operand_breps(&mut self, start: &str) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
        let node = self.request.nodes.iter().find(|item| item.id == start).cloned().ok_or("Boolean operand node missing")?;
        if node.node_type == "combine" {
            // Combine-to-Part is concatenation. Collect the connected operands
            // directly instead of materializing an intermediate Part and copying
            // every body through `realized_solids` before the Boolean.
            let mut inputs = Vec::new();
            let mut first_error: Option<String> = None;
            for link in self
                .request
                .connections
                .iter()
                .filter(|link| link.to == start && link.input == "shape")
            {
                match self.operand_breps(&link.from) {
                    Ok(values) => inputs.extend(values),
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            if inputs.is_empty() {
                return Err(first_error.unwrap_or_else(|| "Combine exact operand has no inputs".into()));
            }
            return Ok(inputs);
        }
        if node.node_type == "mirror" {
            let link = self
                .request
                .connections
                .iter()
                .find(|link| link.to == node.id && link.input == "shape")
                .ok_or("Mirror has no shape input")?;
            let mut brep = copy_brep(self.breps.get(&link.from).ok_or("Mirror source has no exact B-rep")?.as_ref())?;
            crate::mirror_brep(&mut brep, &node.values)?;
            Ok(vec![brep])
        } else if node.node_type == "translate" {
            let link = self.request.connections.iter().find(|link| link.to == start && link.input == "shape").ok_or("Boolean transform has no source")?;
            let mut inputs = self.operand_breps(&link.from)?;
            self.emit_progress(start, ProgressPhase::BrepStart);
            let offset = [node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
            let rotation = [node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
            for brep in &mut inputs { apply_brep_placement(brep, offset, rotation)?; }
            self.emit_progress(start, ProgressPhase::BrepComplete);
            Ok(inputs)
        } else {
            Ok(vec![self.operand_brep(start)?])
        }
    }

    #[cfg(test)]
    pub(crate) fn operand_brep(&mut self, start: &str) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
        let started = Instant::now();
        let result = self.operand_brep_inner(start);
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        *self.brep_timings.entry(start.to_string()).or_insert(0.0) += elapsed;
        result
    }

    #[cfg(test)]
    pub(crate) fn operand_brep_inner(&mut self, start: &str) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
        if let Some(brep) = self.breps.get(start).cloned() {
            self.emit_progress(start, ProgressPhase::BrepStart);
            let out = snapshot_brep(&brep)?;
            self.emit_progress(start, ProgressPhase::BrepComplete);
            return Ok(out);
        }
        let node = self.request.nodes.iter().find(|item| item.id == start).cloned().ok_or("Boolean operand node missing")?;
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_key = (start.to_string(), node_key);
        if !self.bypass_cache && node_cache_is_fresh(&node) && let Some(brep) = self.cache.brep(&cache_key) {
            self.cache.note_brep_hit();
            self.cache_hits += 1;
            self.brep_cache_hits += 1;
            self.breps.insert(start.to_string(), brep.clone());
            self.emit_progress(start, ProgressPhase::BrepStart);
            let out = snapshot_brep(&brep)?;
            self.emit_progress(start, ProgressPhase::BrepComplete);
            return Ok(out);
        }
        self.emit_progress(start, ProgressPhase::BrepStart);
        let result = if node.node_type == "translate" {
            let link = self.request.connections.iter().find(|link| link.to == start && link.input == "shape").ok_or("Boolean transform has no source")?.from.clone();
            let mut brep = self.operand_brep(&link)?;
            let offset = [node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
            let rotation = [node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
            apply_brep_placement(&mut brep, offset, rotation)?;
            Ok(brep)
        } else if node.node_type == "solid" || node.node_type == "primitive" {
            self.cached_primitive_brep(&node)
        } else if node.node_type == "boolean" || node.node_type == "fuse" {
            let a = self.request.connections.iter().find(|link| link.to == node.id && link.input == "a").ok_or("Boolean operand missing A")?.from.clone();
            let b = self.request.connections.iter().find(|link| link.to == node.id && link.input == "b").ok_or("Boolean operand missing B")?.from.clone();
            let (op, reverse) = boolean_mode(&node.values)?;
            let (a, b) = if reverse { (b.clone(), a.clone()) } else { (a.clone(), b.clone()) };
            let (mut a_breps, b_breps) = (self.operand_breps(&a)?, self.operand_breps(&b)?);
            let mut result = a_breps.drain(..).next().ok_or("Boolean operand A is empty")?;
            let mut kernel = nodecraft_solidcore::kernel::kernel();
            for brep in b_breps {
                result = nodecraft_layers::boolean_breps_ops(
                    &mut kernel,
                    &result,
                    &brep,
                    crate::ops_boolean(op.clone()),
                )?
                .unwrap_or_else(nodecraft_solidcore::solidcore::topology::Brep::new);
            }
            Ok(result)
        } else if node.node_type == "chamfer" {
            let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Chamfer has no shape input")?;
            let source = self.operand_brep(&link.from)?;
            let mut values = node.values.clone();
            values["mode"] = serde_json::json!("chamfer");
            nodecraft_blend::chamfer_brep_exact(&source, &values)
                .map_err(|error| format!("Edge chamfer: {error}"))
        } else if node.node_type == "fillet" {
            let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Fillet has no shape input")?;
            let source = self.operand_brep(&link.from)?;
            let has_edges = node.values.get("edges").and_then(serde_json::Value::as_array)
                .is_some_and(|edges| !edges.is_empty());
            if !has_edges { return Ok(source); }
            let mut values = node.values.clone();
            values["mode"] = serde_json::json!("fillet");
            crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Fillet { brep: source, values },
                &node.id,
                fnv_seed(&node.id),
            )?
            .into_brep()
            .ok_or_else(|| "Edge fillet: task returned no shape".to_owned())
            .map_err(|error| format!("Edge fillet: {error}"))
        } else if node.node_type == "shell" {
            let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Shell has no shape input")?;
            let source = self.operand_brep(&link.from)?;
            let thickness = node.values.get("thickness").and_then(serde_json::Value::as_f64).unwrap_or(2.0);
            let inward = node.values.get("direction").and_then(serde_json::Value::as_str).unwrap_or("inward") == "inward";
            let signed = if inward { -thickness } else { thickness };
            let mut openings = Vec::new();
            for face in node.values.get("faces").and_then(serde_json::Value::as_array).into_iter().flatten() {
                let Some(reference) = face.get("id").and_then(serde_json::Value::as_str) else { continue };
                let Some((body, index)) = face_reference_body_and_index(reference) else { continue };
                if body.unwrap_or(0) != 0 { continue; }
                if let Some(face) = source.faces().ids().nth(index) { openings.push(face); }
            }
            nodecraft_bool::shell_brep_solid(source, signed, &openings)
        } else {
            exact_operand_brep(self.request, start)
        };
        if let Ok(ref brep) = result {
            let brep = Arc::new(copy_brep(brep)?);
            self.breps.insert(start.to_string(), brep.clone());
            self.cache.insert_brep(cache_key, brep);
        }
        self.emit_progress(start, ProgressPhase::BrepComplete);
        result
    }

    /// Warm independent primitive/import/loft B-reps and their cheap transforms
    /// through the dependency scheduler. Heavy feature nodes (Boolean, fillet,
    /// chamfer, shell, extrude) stay on the exact Part path: its batched Boolean
    /// path is substantially faster than the generic ops path used here, and
    /// duplicating their validation was the dominant cost on blend-heavy graphs.
    pub(crate) fn prewarm_primitives(&mut self) {
        use crate::scheduler::ScheduledTask;
        // Only warm the target closure. A request carries the whole document;
        // scheduling every primitive/translate in it makes a one-node drag pay
        // for the entire graph.
        let scope: std::collections::HashSet<String> =
            evaluation_order_for_targets(self.request).into_iter().collect();
        let profile = crate::workload::profile_request(self.request);
        if !profile.exact_dispatch_is_worthwhile() {
            return;
        }
        let mut tasks = Vec::new();
        let mut keys = std::collections::BTreeMap::new();
        let by_id: std::collections::HashMap<&str, &Node> = self
            .request
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect();
        let mut inputs: std::collections::HashMap<&str, Vec<(&str, &str)>> =
            std::collections::HashMap::new();
        let mut children: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for link in &self.request.connections {
            inputs
                .entry(link.to.as_str())
                .or_default()
                .push((link.input.as_str(), link.from.as_str()));
            children
                .entry(link.from.as_str())
                .or_default()
                .push(link.to.as_str());
        }
        let mut warmable = std::collections::BTreeSet::new();
        let mut queue: std::collections::VecDeque<&str> = std::collections::VecDeque::new();
        fn ready(
            node: &Node,
            inputs: &std::collections::HashMap<&str, Vec<(&str, &str)>>,
            warmable: &std::collections::BTreeSet<String>,
        ) -> bool {
            inputs.get(node.id.as_str()).is_some_and(|inputs| {
                let connected = |input: &str| {
                    inputs.iter().any(|(candidate, from)| {
                        *candidate == input && warmable.contains(*from)
                    })
                };
                match node.node_type.as_str() {
                    "fillet" | "chamfer" | "shell" => connected("shape"),
                    "boolean" | "fuse" => connected("a") && connected("b"),
                    "extrude" => connected("profile") || connected("shape"),
                    "stepimporter" => node
                        .values
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|path| !path.is_empty()),
                    "loft" | "surfaceloft" => true,
                    _ => false,
                }
            })
        }
        for node in self.request.nodes.iter() {
            if node.suppressed || !scope.contains(&node.id) {
                continue;
            }
            let start = (node.node_type == "solid" || node.node_type == "primitive")
                || (node.node_type == "stepimporter"
                    && node
                        .values
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|path| !path.is_empty()))
                || matches!(node.node_type.as_str(), "loft" | "surfaceloft");
            if start {
                warmable.insert(node.id.clone());
                queue.push_back(node.id.as_str());
            }
        }
        while let Some(id) = queue.pop_front() {
            for child in children.get(id).cloned().unwrap_or_default() {
                if !scope.contains(child) || warmable.contains(child) {
                    continue;
                }
                let Some(node) = by_id.get(child) else { continue };
                if node.suppressed || !ready(node, &inputs, &warmable) {
                    continue;
                }
                warmable.insert(child.to_owned());
                queue.push_back(child);
            }
        }
        drop(by_id);
        let mut loft_inputs: std::collections::BTreeMap<String, (crate::SketchOutput, crate::SketchOutput, bool)> =
            std::collections::BTreeMap::new();
        let mut surface_inputs: std::collections::BTreeMap<String, (Vec<crate::SketchOutput>, bool)> =
            std::collections::BTreeMap::new();
        let section_nodes: Vec<Node> = self.request.nodes.iter()
            .filter(|node| node.node_type == "loft" || node.node_type == "surfaceloft")
            .cloned().collect();
        for node in &section_nodes {
            let ruled = node.values.get("ruled").and_then(serde_json::Value::as_f64).unwrap_or(0.0) >= 0.5;
            if node.node_type == "loft" {
                let bottom = self.request.connections.iter().find(|link| link.to == node.id && link.input == "bottom").map(|link| link.from.clone());
                let top = self.request.connections.iter().find(|link| link.to == node.id && link.input == "top").map(|link| link.from.clone());
                if let (Some(bottom), Some(top)) = (bottom, top) {
                    if let (Some(bottom), Some(top)) = (self.sketch_output_for(&bottom), self.sketch_output_for(&top)) {
                        loft_inputs.insert(node.id.clone(), (bottom, top, ruled));
                    }
                }
            } else {
                let mut sections = Vec::new();
                let mut valid = true;
                for link in self.request.connections.iter().filter(|link| link.to == node.id && link.input == "sections") {
                    match self.sketch_output_for(&link.from) {
                        Some(section) => sections.push(section),
                        None => { valid = false; break; }
                    }
                }
                if valid && sections.len() >= 2 {
                    surface_inputs.insert(node.id.clone(), (sections, ruled));
                }
            }
        }
        for node in self.request.nodes.iter() {
            if !warmable.contains(&node.id) || !node_cache_is_fresh(node) { continue; }
            // Heavy feature nodes are evaluated by the exact Part path, which
            // uses the batched Boolean implementation and avoids duplicate
            // ops-layer validation. Only warm work that is independent and cheap.
            if matches!(node.node_type.as_str(), "boolean" | "fuse" | "fillet" | "chamfer" | "shell" | "extrude") {
                continue;
            }
            let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
            let cache_key = (node.id.clone(), node_key);
            if self.breps.contains_key(&node.id)
                || (!self.bypass_cache && self.cache.brep(&cache_key).is_some())
            {
                continue;
            }
            let id = node.id.clone();
            keys.insert(id.clone(), cache_key);
            match node.node_type.as_str() {
                "solid" | "primitive" => {
                    let node_for_job = node.clone();
                    tasks.push(ScheduledTask::leaf(
                        id,
                        fnv_seed(&node.id),
                        move |kernel| {
                            primitive_brep_with_kernel(kernel, &node_for_job)
                                .map(std::sync::Arc::new)
                        },
                    ));
                }
                "loft" => {
                    let Some((bottom, top, ruled)) = loft_inputs.remove(&node.id) else { continue };
                    tasks.push(ScheduledTask::leaf(id, fnv_seed(&node.id), move |kernel| {
                        loft_sketch(kernel, &bottom, &top, ruled, true).map(std::sync::Arc::new)
                    }));
                }
                "surfaceloft" => {
                    let Some((sections, ruled)) = surface_inputs.remove(&node.id) else { continue };
                    tasks.push(ScheduledTask::leaf(id, fnv_seed(&node.id), move |kernel| {
                        multi_section_surface_loft(kernel, &sections, ruled, false).map(std::sync::Arc::new)
                    }));
                }
                "stepimporter" => {
                    let Some(path) = node.values.get("path").and_then(serde_json::Value::as_str).map(str::to_owned) else { continue };
                    if path.is_empty() { continue; }
                    tasks.push(ScheduledTask::leaf(id, fnv_seed(&node.id), move |_kernel| {
                        nodecraft_mesh::import_step_brep(&path).map(std::sync::Arc::new)
                    }));
                }
                _ => {}
            }
        }
        if tasks.is_empty() { return; }
        let pool = crate::kernel_pool::shared_pool();
        let max_in_flight = profile.recommended_exact_in_flight(pool.workers());
        let Ok(results) = crate::scheduler::run_levels_with_limit(
            pool,
            tasks,
            crate::kernel_pool::current_revision_token(),
            max_in_flight,
        ) else { return; };
        for (id, result) in results {
            let Ok(brep) = result else { continue };
            self.breps.insert(id.clone(), brep.clone());
            self.prewarmed.insert(id.clone());
            if let Some(cache_key) = keys.remove(&id) {
                self.cache.insert_brep(cache_key, brep);
            }
        }
    }

    /// Collect typed inputs by the node descriptor's flat-graph port names.
    /// Evaluation is requested first so cached mesh/Part/Sketch values exist.
    #[allow(dead_code)] // Planned typed-input path; scheduler dispatch will consume it.
    pub(crate) fn collect_node_inputs(&mut self, node: &Node) -> crate::node_task::NodeInputs {
        use crate::node_task::{NodeInputs, NodeOutput};
        let Some(descriptor) = nodecraft_node_contract::descriptor(&node.node_type) else {
            return NodeInputs::new();
        };
        let ports: Vec<&'static str> = descriptor.inputs.iter().map(|port| port.name).collect();
        let mut inputs = NodeInputs::new();
        for port in ports {
            let Some(link) = self
                .request
                .connections
                .iter()
                .find(|link| link.to == node.id && link.input == port)
            else {
                continue;
            };
            let source = link.from.clone();
            let value = if let Some(shape) = self.breps.get(&source) {
                Some(NodeOutput::Shape(shape.clone()))
            } else if let Some(part) = self.parts.get(&source) {
                Some(NodeOutput::Part(part.clone()))
            } else if let Some(sketch) = self.sketches.get(&source) {
                Some(NodeOutput::Sketch(sketch.clone()))
            } else {
                self.objects.get(&source).cloned().map(|mesh| {
                    NodeOutput::Mesh(std::sync::Arc::new(mesh))
                })
            };
            if let Some(value) = value {
                inputs.insert(port, value);
            }
        }
        inputs
    }

    pub(crate) fn emit_progress(&mut self, node_id: &str, phase: ProgressPhase) {
        if !self.request.progress { return; }
        let Some(callback) = self.progress.as_mut() else { return; };
        callback(ProgressEvent {
            request_id: self.request.request_id,
            tab_id: self.request.tab_id.clone(),
            graph_revision: self.request.graph_revision.clone(),
            node_id: node_id.to_string(),
            phase,
        });
    }

    /// Exact Part for part-shaped chains. This is the first normal-evaluation
    /// path that consumes and returns Part values instead of meshes.
    pub(crate) fn evaluate_part(&mut self, start: &str) -> Result<Arc<Part>, String> {
        if let Some(part) = self.parts.get(start) {
            return Ok(part.clone());
        }
        if self.part_visiting.iter().any(|id| id == start) {
            return Err("Part cycle detected".into());
        }
        let node = self.request.nodes.iter().find(|node| node.id == start).cloned().ok_or("Part node missing")?;
        if node.suppressed {
            return suppression_source(&node, &self.request.connections)
                .map(|source| self.evaluate_part(&source))
                .unwrap_or_else(|| Ok(std::sync::Arc::new(Part::empty())));
        }
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_key = (start.to_owned(), node_key);
        let cache_is_fresh = node_cache_is_fresh(&node);
        if !self.bypass_cache && cache_is_fresh && let Some(part) = self.cache.part(&cache_key) {
            self.cache.note_part_hit();
            self.cache_hits += 1;
            self.part_cache_hits += 1;
            self.parts.insert(start.to_owned(), part.clone());
            return Ok(part);
        }
        if self.prewarmed.contains(start)
            && let Some(brep) = self.breps.get(start).cloned()
        {
            let node_source = crate::node_content_seed(
                &self.request,
                &node.id,
                &mut std::collections::HashSet::new(),
            );
            let part = Arc::new(Part::from_arc(brep).tagged(node_source));
            self.parts.insert(start.to_owned(), part.clone());
            return Ok(part);
        }
        self.part_visiting.push(start.to_owned());
        let result = (|| -> Result<Part, String> {
            match node.node_type.as_str() {
                "groupinput" => Ok(Part::empty()),
                "groupoutput" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Group output has no connected input")?;
                    self.evaluate_part(&link.from).map(|part| part.as_ref().clone())
                }
                "solid" | "primitive" => Ok(Part::from_arc(self.cached_primitive_brep_arc(&node)?)),
                "translate" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Part translate has no input")?;
                    let source = self.evaluate_part(&link.from)?;
                    let offset = [node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
                    let rotation = [node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
                    Ok(source.as_ref().translated(offset, rotation))
                }
                "mirror" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Part mirror has no input")?;
                    let source = self.evaluate_part(&link.from)?;
                    Ok(source.mirrored(&node.values))
                }
                "combine" => {
                    let mut bodies = Vec::new();
                    let mut sketches = Vec::new();
                    let mut first_error: Option<String> = None;
                    for link in self.request.connections.iter().filter(|link| link.to == node.id && link.input == "shape") {
                        // A failed import inside a Combine must not poison the
                        // other, successfully imported operands.
                        match self.evaluate_part(&link.from) {
                            Ok(part) => {
                                bodies.extend(part.bodies.clone());
                                sketches.extend(part.sketches.clone());
                            }
                            Err(error) => {
                                first_error.get_or_insert(error);
                            }
                        }
                    }
                    if bodies.is_empty() && sketches.is_empty() {
                        return Err(first_error.unwrap_or_else(|| "Combine Part has no inputs".into()));
                    }
                    Ok(Part { bodies, sketches })
                }
                "boolean" | "fuse" => {
                    let a_link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "a");
                    let b_link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "b");
                    let a_suppressed = a_link.is_some_and(|link| effective_output_suppressed(self.request, &link.from));
                    let b_suppressed = b_link.is_some_and(|link| effective_output_suppressed(self.request, &link.from));
                    match (a_link, b_link) {
                        (Some(a), None) => return self.evaluate_part(&a.from).map(|part| part.as_ref().clone()),
                        (None, Some(b)) => return self.evaluate_part(&b.from).map(|part| part.as_ref().clone()),
                        (None, None) => return Err("Part Boolean has no connected operand".into()),
                        (Some(_), Some(_)) if a_suppressed && b_suppressed => {
                            return Err("Part Boolean has no active operand".into())
                        }
                        (Some(_), Some(b)) if a_suppressed => return self.evaluate_part(&b.from).map(|part| part.as_ref().clone()),
                        (Some(a), Some(_)) if b_suppressed => return self.evaluate_part(&a.from).map(|part| part.as_ref().clone()),
                        (Some(a), Some(b)) => {
                            let (op, reverse) = boolean_mode(&node.values)?;
                            let (a, b) = if reverse { (b.from.clone(), a.from.clone()) } else { (a.from.clone(), b.from.clone()) };
                            let a_part = self.evaluate_part(&a)?;
                            let b_part = self.evaluate_part(&b)?;
                            for (operand, part) in [("A", a_part.as_ref()), ("B", b_part.as_ref())] {
                                if part.bodies.len() > MAX_BOOLEAN_SOLIDS {
                                    return Err(format!(
                                        "Boolean operand {operand} contains {} solids (for example a Scatter with Combine on). Combine groups the instances into one Part; a Boolean still needs an exact union of every solid, and supports at most {}. Reduce the scatter count or boolean a smaller Part.",
                                        part.bodies.len(),
                                        MAX_BOOLEAN_SOLIDS
                                    ));
                                }
                            }
                            let a_parts = a_part.realized_solids()?;
                            let b_parts = b_part.realized_solids()?;
                            // N-ary list Boolean on the persistent kernel pool.
                            let breps = crate::node_task::execute_exact(
                                crate::node_task::ExactNodeTask::BooleanLists {
                                    op: op.clone(),
                                    a: a_parts,
                                    b: b_parts,
                                },
                                &node.id,
                                fnv_seed(&node.id),
                            )?
                            .into_shapes()
                            .ok_or_else(|| "Boolean task returned no shapes".to_owned())?;
                            Ok(Part {
                                bodies: breps.into_iter().map(|brep| BodyInstance {
                                    body: brep,
                                    transforms: Vec::new(),
                                    source: None,
                                    source_body: 0,
                                }).collect(),
                                sketches: Vec::new(),
                            })
                        }
                    }
                }
                "look" => {
                    let link = self.request.connections.iter()
                        .find(|link| link.to == node.id && (link.input == "shape" || link.input == "a"))
                        .ok_or("Look has no shape input")?;
                    let source = self.evaluate_part(&link.from)?;
                    let color = parse_look_color(&node.values)?;
                    let mut bodies = Vec::with_capacity(source.bodies.len());
                    for body in &source.bodies {
                        let brep = crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::Look {
                                brep: std::sync::Arc::clone(&body.body),
                                colour: color,
                            },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_shape()
                        .ok_or_else(|| "Look task returned no shape".to_owned())?;
                        bodies.push(BodyInstance {
                            body: brep,
                            transforms: body.transforms.clone(),
                            source: body.source,
                            source_body: body.source_body,
                        });
                    }
                    Ok(Part { bodies, sketches: source.sketches.clone() })
                }
                "extrude" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && (link.input == "profile" || link.input == "shape")).ok_or("Part extrude has no profile input")?;
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
                        let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
                        let brep = crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::ExtrudeSketch { output: std::sync::Arc::new(output), amount },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_brep()
                        .ok_or_else(|| "sketch extrude task returned no shape".to_owned())?;
                        return Ok(Part::from_brep(brep));
                    }
                    let source = self.evaluate_part(&link.from)?;
                    let faces = node.values.get("faces").and_then(serde_json::Value::as_array);
                    if faces.is_none_or(|items| items.is_empty()) {
                        return Ok(source.as_ref().clone());
                    }
                    let reference = faces.and_then(|values| values.first()).and_then(|value| value.get("id")).and_then(serde_json::Value::as_str).unwrap_or("");
                    let index: usize = face_reference_index(reference).ok_or("Extrude needs a selected face reference (face:N).")?;
                    let mut solids = source.realized_solids()?;
                    if solids.len() != 1 {
                        return Err(format!("Extrude requires exactly one solid body, got {}", solids.len()));
                    }
                    let profile = solids.pop().ok_or("Extrude source Part is empty")?;
                    let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
                    let tool = crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::Extrude {
                            brep: copy_brep(&profile)?,
                            face_index: index,
                            amount,
                        },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "extrude task returned no shape".to_owned())?;
                    let operation = node.values.get("operation").and_then(serde_json::Value::as_str).unwrap_or("new");
                    if operation == "new" {
                        return Ok(Part::from_brep(tool));
                    }
                    let result = crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::ExtrudeFeature {
                            brep: profile,
                            face_index: index,
                            amount,
                            operation: operation.to_owned(),
                        },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "Extrude feature task returned no shape".to_owned())?;
                    Ok(Part::from_brep(result))
                }
                "sketch" => {
                    let sketch: crate::sketch::Sketch = serde_json::from_value(
                        node.values.get("sketch").cloned().unwrap_or(serde_json::Value::Null),
                    )
                    .map_err(|error| format!("Sketch values: {error}"))?;
                    let output = Arc::new(SketchOutput::new(sketch));
                    self.sketches.insert(node.id.clone(), output.clone());
                    Ok(Part::from_sketch(output))
                }
                "spline" => {
                    let sketch = spline_sketch_from_values(&node.values)?;
                    let output = Arc::new(SketchOutput::new(sketch));
                    self.sketches.insert(node.id.clone(), output.clone());
                    Ok(Part::from_sketch(output))
                }
                "loft" => {
                    let bottom = input_source(self.request, node.id.as_str(), "bottom").ok_or("Loft has no bottom section")?.from.clone();
                    let top = input_source(self.request, node.id.as_str(), "top").ok_or("Loft has no top section")?.from.clone();
                    let ruled = node.values.get("ruled").and_then(serde_json::Value::as_f64).unwrap_or(0.0) >= 0.5;
                    let bottom_output = self.sketch_output_for(&bottom);
                    let top_output = self.sketch_output_for(&top);
                    let brep = if let (Some(bottom_output), Some(top_output)) = (bottom_output, top_output) {
                        crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::LoftSketch { bottom: bottom_output, top: top_output, ruled },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_brep()
                        .ok_or_else(|| "Loft task returned no shape".to_owned())?
                    } else {
                        let bottom_brep = crate::exact_operand_brep(self.request, &bottom)?;
                        let top_brep = crate::exact_operand_brep(self.request, &top)?;
                        crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::FaceLoft {
                                bottom: bottom_brep,
                                top: top_brep,
                                ruled,
                                cap_ends: true,
                            },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_brep()
                        .ok_or_else(|| "Face loft task returned no shape".to_owned())?
                    };
                    Ok(Part::from_brep(brep))
                }
                "surfaceloft" => {
                    let section_ids: Vec<String> = self.request.connections.iter()
                        .filter(|link| link.to == node.id && link.input == "sections")
                        .map(|link| link.from.clone())
                        .collect();
                    if section_ids.len() < 2 {
                        return Err("Surface Loft needs at least two section connections".into());
                    }
                    let mut sections = Vec::with_capacity(section_ids.len());
                    for section_id in &section_ids {
                        sections.push(self.sketch_output_for(section_id)
                            .ok_or_else(|| format!("Surface Loft section {section_id} must be a sketch or spline curve"))?);
                    }
                    let ruled = node.values.get("ruled").and_then(serde_json::Value::as_f64).unwrap_or(0.0) >= 0.5;
                    let brep = crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::SurfaceLoft { sections, ruled },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "Surface Loft task returned no shape".to_owned())?;
                    Ok(Part::from_brep(brep))
                }
                "chamfer" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Chamfer has no shape input")?;
                    let source_part = self.evaluate_part(&link.from)?;
                    let has_edges = node.values.get("edges").and_then(serde_json::Value::as_array)
                        .is_some_and(|edges| !edges.is_empty());
                    if !has_edges { return Ok((*source_part).clone()); }
                    let mut solids = source_part.realized_solids()?;
                    if solids.len() != 1 {
                        return Err(format!("Chamfer requires exactly one solid body, got {}", solids.len()));
                    }
                    let source = solids.pop().ok_or("Chamfer source Part is empty")?;
                    let mut values = node.values.clone();
                    values["mode"] = serde_json::json!("chamfer");
                    crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::Chamfer { brep: source, values },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "chamfer task returned no shape".to_owned())
                    .map(Part::from_brep)
                }
                "fillet" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Fillet has no shape input")?;
                    let source_part = self.evaluate_part(&link.from)?;
                    let has_edges = node.values.get("edges").and_then(serde_json::Value::as_array)
                        .is_some_and(|edges| !edges.is_empty());
                    if !has_edges { return Ok((*source_part).clone()); }
                    let mut solids = source_part.realized_solids()?;
                    if solids.len() != 1 {
                        return Err(format!("Fillet requires exactly one solid body, got {}", solids.len()));
                    }
                    let source = solids.pop().ok_or("Fillet source Part is empty")?;
                    let mut values = node.values.clone();
                    values["mode"] = serde_json::json!("fillet");
                    crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::Fillet { brep: source, values },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "fillet task returned no shape".to_owned())
                    .map(Part::from_brep)
                }
                "offsetfaces" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Offset Faces has no shape input")?;
                    let source = self.evaluate_part(&link.from)?;
                    let references: Vec<&str> = node.values.get("faces").and_then(serde_json::Value::as_array).map(|faces| {
                        faces.iter().filter_map(|face| face.get("id").and_then(serde_json::Value::as_str)).collect()
                    }).unwrap_or_default();
                    if references.is_empty() {
                        return Ok(source.as_ref().clone());
                    }
                    let distance = node.values.get("distance").and_then(serde_json::Value::as_f64).unwrap_or(1.0);
                    let solids = source.realized_solids()?;
                    let mut bodies = Vec::with_capacity(solids.len());
                    for (body_index, brep) in solids.into_iter().enumerate() {
                        let mut faces = Vec::new();
                        for reference in &references {
                            let Some((body, face_index)) = face_reference_body_and_index(reference) else { continue };
                            if body.is_some_and(|body| body != body_index) { continue };
                            faces.push(face_index);
                        }
                        if faces.is_empty() {
                            bodies.push(BodyInstance { body: Arc::new(brep), transforms: Vec::new(), source: None, source_body: 0 });
                            continue;
                        }
                        let offset = crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::OffsetFaces { brep, faces, distance },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_shape()
                        .ok_or_else(|| "Offset Faces task returned no shape".to_owned())?;
                        bodies.push(BodyInstance { body: offset, transforms: Vec::new(), source: None, source_body: 0 });
                    }
                    Ok(Part { bodies, ..Part::empty() })
                }
                "shell" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Shell has no shape input")?;
                    let source = self.evaluate_part(&link.from)?;
                    let solids = source.realized_solids()?;
                    if solids.is_empty() { return Err("Shell source Part is empty".into()); }
                    let thickness = node.values.get("thickness").and_then(serde_json::Value::as_f64).unwrap_or(2.0);
                    if !(thickness.is_finite() && thickness > 0.0) { return Err("Shell thickness must be positive".into()); }
                    let inward = node.values.get("direction").and_then(serde_json::Value::as_str).unwrap_or("inward") == "inward";
                    let signed = if inward { -thickness } else { thickness };
                    let references: Vec<&str> = node.values.get("faces").and_then(serde_json::Value::as_array).map(|faces| {
                        faces.iter().filter_map(|face| face.get("id").and_then(serde_json::Value::as_str)).collect()
                    }).unwrap_or_default();
                    // Validate references before touching geometry. Silently
                    // ignoring a malformed or out-of-range opening turns a
                    // requested shell into a closed shell of the wrong shape.
                    let mut parsed = Vec::with_capacity(references.len());
                    for reference in &references {
                        let Some((body, face_index)) = face_reference_body_and_index(reference) else {
                            return Err(format!("Shell opening face reference {reference:?} is invalid; expected face:N or body:B:face:N"));
                        };
                        if let Some(body) = body && body >= solids.len() {
                            return Err(format!("Shell opening {reference:?} names body {body}, but the source has {} bodies", solids.len()));
                        }
                        parsed.push((body, face_index));
                    }
                    let mut bodies = Vec::with_capacity(solids.len());
                    for (body_index, brep) in solids.into_iter().enumerate() {
                        let mut openings = Vec::new();
                        for (body, face_index) in &parsed {
                            if body.is_some_and(|body| body != body_index) { continue; }
                            if brep.faces().ids().nth(*face_index).is_none() {
                                if body.is_some() {
                                    return Err(format!("Shell opening body:{body_index}:face:{face_index} does not exist on this body"));
                                }
                                continue;
                            }
                            openings.push(*face_index);
                        }
                        let shelled = crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::Shell {
                                brep,
                                signed_thickness: signed,
                                openings,
                            },
                            &node.id,
                            fnv_seed(&node.id),
                        )?
                        .into_brep()
                        .ok_or_else(|| "Shell task returned no shape".to_owned())?;
                        bodies.push(BodyInstance { body: Arc::new(shelled), transforms: Vec::new(), source: None, source_body: 0 });
                    }
                    Ok(Part { bodies, ..Part::empty() })
                }
                "scatter" => {
                    let link = self.request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("Scatter has no source")?;
                    let source = self.evaluate_part(&link.from)?;
                    if source.bodies.is_empty() && source.sketches.is_empty() {
                        return Err("Scatter source Part is empty".into());
                    }
                    let count = node.values.get("count").and_then(serde_json::Value::as_f64).unwrap_or(100.0).round().clamp(1.0, 5000.0) as usize;
                    let span_x = node.values.get("spanX").and_then(serde_json::Value::as_f64).unwrap_or(200.0).max(1.0);
                    let span_y = node.values.get("spanY").and_then(serde_json::Value::as_f64).unwrap_or(200.0).max(1.0);
                    let seed = node.values.get("seed").and_then(serde_json::Value::as_f64).unwrap_or(1.0).to_bits();
                    let min_scale = node.values.get("minScale").and_then(serde_json::Value::as_f64).unwrap_or(0.75).max(0.05);
                    let max_scale = node.values.get("maxScale").and_then(serde_json::Value::as_f64).unwrap_or(1.25).max(min_scale);
                    let distribution = node.values.get("distribution").and_then(serde_json::Value::as_str).unwrap_or("random");
                    let columns = (count as f64).sqrt().ceil() as usize;
                    let rows = (count + columns - 1) / columns;
                    let cell_x = span_x / columns as f64;
                    let cell_y = span_y / rows as f64;
                    let mut rng = ScatterRng::new(seed);
                    let mut bodies = Vec::new();
                    let mut sketches = Vec::new();
                    for index in 0..count {
                        let col = index % columns;
                        let row = index / columns;
                        let grid_x = if columns > 1 { -span_x / 2.0 + span_x * col as f64 / (columns - 1) as f64 } else { 0.0 };
                        let grid_y = if rows > 1 { -span_y / 2.0 + span_y * row as f64 / (rows - 1) as f64 } else { 0.0 };
                        let (x, y) = match distribution {
                            "grid" => (grid_x, grid_y),
                            "stratified" => (-span_x / 2.0 + (col as f64 + rng.unit()) * cell_x, -span_y / 2.0 + (row as f64 + rng.unit()) * cell_y),
                            _ => ((rng.unit() - 0.5) * span_x, (rng.unit() - 0.5) * span_y),
                        };
                        let factor = min_scale + (max_scale - min_scale) * rng.unit();
                        let mut instance_bodies = source.bodies.clone();
                        for body in &mut instance_bodies {
                            body.transforms.push(BodyTransform::Scale { factor });
                            body.transforms.push(BodyTransform::Place { offset: [x, y, 0.0], rotation: [0.0; 3] });
                        }
                        bodies.extend(instance_bodies);
                        let mut instance_sketches = source.sketches.clone();
                        for sketch in &mut instance_sketches {
                            sketch.transforms.push(BodyTransform::Scale { factor });
                            sketch.transforms.push(BodyTransform::Place { offset: [x, y, 0.0], rotation: [0.0; 3] });
                        }
                        sketches.extend(instance_sketches);
                    }
                    Ok(Part { bodies, sketches })
                }
                "stepimporter" => {
                    let path = node.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
                    if path.is_empty() { return Err("STEP Part has no path".into()); }
                    let key = import_cache_key(path);
                    let default_quality = (nodecraft_mesh::display_tolerance()
                        - nodecraft_mesh::DEFAULT_DISPLAY_TOLERANCE).abs() < 1.0e-9;
                    if !self.bypass_cache && default_quality && node_cache_is_fresh(&node) && let Some(import) = self.cache.import(&key) {
                        self.cache.note_import_hit();
                        self.step_diagnostics.insert(node.id.clone(), import.diagnostics.clone());
                    } else {
                        let import = import_step(path)?;
                        self.step_diagnostics.insert(node.id.clone(), import.diagnostics.clone());
                        if default_quality { self.cache.insert_import(key, Arc::new(import)); }
                    }
                    let brep = crate::node_task::execute_exact(
                        crate::node_task::ExactNodeTask::ImportStep { path: path.to_owned() },
                        &node.id,
                        fnv_seed(&node.id),
                    )?
                    .into_brep()
                    .ok_or_else(|| "STEP import task returned no shape".to_owned())?;
                    Ok(Part::from_brep(brep))
                }
                other => Err(format!("exact Part evaluation does not support {other:?}")),
            }
        })();
        self.part_visiting.pop();
        if let Ok(part) = &result {
            let mut part = part.clone();
            // Newly generated bodies inherit this node's content key; bodies
            // passed through from an upstream node keep their producing key so
            // instances share plans across transforms/scatters.
            let node_source = crate::node_content_seed(
                &self.request,
                &node.id,
                &mut std::collections::HashSet::new(),
            );
            for (index, body) in part.bodies.iter_mut().enumerate() {
                if body.source.is_none() {
                    body.source = Some(node_source);
                    body.source_body = index as u32;
                }
            }
            let part = Arc::new(part);
            self.parts.insert(start.to_owned(), part.clone());
            self.cache.insert_part(cache_key, part.clone());
            result.map(|_| part)
        } else if let Err(error) = result {
            self.errors.insert(start.to_owned(), error);
            if let Some(source) = bypass_source(&node, &self.request.connections)
                && let Ok(part) = self.evaluate_part(&source)
            {
                self.bypassed.insert(start.to_owned());
                let part = Arc::new(part.as_ref().clone());
                self.parts.insert(start.to_owned(), part.clone());
                return Ok(part);
            }
            Err(self.errors.get(start).cloned().unwrap_or_else(|| "Part evaluation failed".into()))
        } else {
            unreachable!()
        }
    }

}
