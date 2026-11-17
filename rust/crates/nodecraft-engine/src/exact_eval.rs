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
        .into_shared_brep()
        .ok_or("primitive task returned no shape")?;
        self.breps.insert(node.id.clone(), brep.clone());
        self.cache.insert_brep(cache_key, brep.clone());
        Ok(brep)
    }

    pub(crate) fn cached_primitive_brep(&mut self, node: &Node) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
        let brep = self.cached_primitive_brep_arc(node)?;
        snapshot_brep(&brep)
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

                    _ => false,
                }
            })
        }
        for node in self.request.nodes.iter() {
            if node.suppressed || !scope.contains(&node.id) {
                continue;
            }
            let start = node.node_type == "solid" || node.node_type == "primitive";
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
                let Some(node) = self.index.node(child) else { continue };
                if node.suppressed || !ready(node, &inputs, &warmable) {
                    continue;
                }
                warmable.insert(child.to_owned());
                queue.push_back(child);
            }
        }
        for node in self.request.nodes.iter() {
            // Warm dirty leaves too: they are exactly the work the next serial
            // pass would otherwise do alone. Only skip nodes whose result is
            // both fresh and already cached.
            if !warmable.contains(&node.id) { continue; }
            // Heavy feature nodes are evaluated by the exact Part path, which
            // uses the batched Boolean implementation and avoids duplicate
            // ops-layer validation. Only warm work that is independent and cheap.
            if matches!(node.node_type.as_str(), "boolean" | "fuse" | "fillet" | "chamfer" | "shell" | "extrude") {
                continue;
            }
            let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
            let cache_key = (node.id.clone(), node_key);
            let fresh = node_cache_is_fresh(node);
            if self.breps.contains_key(&node.id)
                || (fresh && !self.bypass_cache && self.cache.brep(&cache_key).is_some())
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
                            crate::nodes::primitive_brep_with_kernel(kernel, &node_for_job)
                                .map(std::sync::Arc::new)
                        },
                    ));
                }
                // A STEP import is one file with any number of bodies, so it has
                // no single `Arc<Brep>` leaf value. The Part pass is the one
                // import route: it caches the whole Part, never a stale first body.
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
            let Some(link) = self.index.first_incoming(&node.id, port) else {
                continue;
            };
            let source = link.from.clone();
            let value = if let Some(shape) = self.breps.get(&source) {
                Some(NodeOutput::Shape(nodecraft_ops::SolidBody::from_arc(shape.clone())))
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
        let node = self.index.node(start).cloned().ok_or("Part node missing")?;
        if node.suppressed {
            // A suppressed node is a pass-through: publish its source's Part
            // under this node's id as well. Every consumer (display
            // breakpoint, comparison, picking and validation) looks a target
            // up by id, so returning the source Part without registering it
            // here would make the suppressed target look empty.
            let part = suppression_source(&node, &self.index)
                .map(|source| self.evaluate_part(&source))
                .unwrap_or_else(|| Ok(std::sync::Arc::new(Part::empty())))?;
            self.parts.insert(start.to_owned(), part.clone());
            return Ok(part);
        }
        let node_key = node.cache.as_ref().and_then(|cache| cache.key.clone()).unwrap_or_default();
        let cache_key = (start.to_owned(), node_key);
        let cache_is_fresh = node_cache_is_fresh(&node);
        if !self.bypass_cache && cache_is_fresh && let Some(part) = self.cache.part(&cache_key) {
            self.cache.note_part_hit();
            self.cache_hits += 1;
            self.part_cache_hits += 1;
            self.parts.insert(start.to_owned(), part.clone());
            if let Some(warnings) = self.cache.warnings(&cache_key) {
                self.warnings.insert(start.to_owned(), warnings);
            }
            return Ok(part);
        }
        if self.prewarmed.contains(start)
            && let Some(brep) = self.breps.get(start).cloned()
        {
            let node_source = crate::node_content_seed(
                &self.index,
                &node.id,
                &mut std::collections::HashSet::new(),
            );
            let part = Arc::new(Part::from_arc(brep).tagged(node_source));
            self.parts.insert(start.to_owned(), part.clone());
            self.cache.insert_part(cache_key, part.clone());
            return Ok(part);
        }
        self.part_visiting.push(start.to_owned());
        let result = (|| -> Result<Part, String> {
            if crate::nodes::find(&node.node_type).is_some() {
                let builtin = crate::nodes::find(&node.node_type).expect("registered node");
                return builtin.evaluate(
                    crate::nodes::NodeView::from(&node),
                    &mut crate::nodes::EnginePartEval::new(self, &node),
                );
            }
            match node.node_type.as_str() {
                "groupinput" => Ok(Part::empty()),
                "groupoutput" => {
                    let link = self.index.first_incoming(&node.id, "shape").ok_or("Group output has no connected input")?;
                    self.evaluate_part(&link.from).map(|part| part.as_ref().clone())
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
                &self.index,
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
            if let Some(warnings) = self.warnings.get(start) {
                self.cache.insert_warnings(cache_key.clone(), warnings.clone());
            }
            self.cache.insert_part(cache_key, part.clone());
            result.map(|_| part)
        } else if let Err(error) = result {
            if !crate::kernel_pool::current_revision_token().is_cancelled() {
                self.errors.insert(start.to_owned(), error);
            }
            if let Some(source) = bypass_source(&node, &self.index)
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

/// Whether `id` is a Union node this evaluation may fold into its parent: an unrequested
/// `boolean`/`fuse` with `union` mode whose only reader is `parent`.
fn fusable_union(index: &RequestIndex<'_>, id: &str, parent: &str) -> bool {
    if index.targets().iter().any(|target| target == id) {
        return false;
    }
    if index.mesh_targets().is_some_and(|targets| targets.iter().any(|target| target == id)) {
        return false;
    }
    let Some(node) = index.node(id) else {
        return false;
    };
    if node.suppressed || !matches!(node.node_type.as_str(), "boolean" | "fuse") {
        return false;
    }
    if index.outgoing(id).count() != 1 {
        return false;
    }
    if !index.outgoing(id).any(|link| link.to == parent) {
        return false;
    }
    matches!(crate::boolean_mode(&node.values), Ok((nodecraft_bool::BooleanOp::Union, false)))
}

/// The leaf operands of a run of Union nodes ending at `start`, in the order a node-by-node
/// chain would apply them (A's leaves then B's). `None` when `start` is a leaf.
fn union_leaves(index: &RequestIndex<'_>, start: &str, parent: &str) -> Option<Vec<String>> {
    if !fusable_union(index, start, parent) {
        return None;
    }
    let a = index.first_incoming(start, "a")?.from.clone();
    let b = index.first_incoming(start, "b")?.from.clone();
    let mut leaves = Vec::new();
    leaves.extend(union_leaves(index, &a, start).unwrap_or_else(|| vec![a]));
    leaves.extend(union_leaves(index, &b, start).unwrap_or_else(|| vec![b]));
    Some(leaves)
}

/// Every Union node folded into a Union above it, so the exact loop does not evaluate it
/// separately before the top of its run has had the chance to fuse it.
pub(crate) fn union_run_intermediates(index: &RequestIndex<'_>) -> std::collections::HashSet<String> {
    let mut folded = std::collections::HashSet::new();
    for node in index.nodes() {
        if node.suppressed
            || !matches!(node.node_type.as_str(), "boolean" | "fuse")
            || index.targets().iter().any(|target| target == &node.id)
            || index
                .mesh_targets()
                .is_some_and(|targets| targets.iter().any(|target| target == &node.id))
        {
            continue;
        }
        if !matches!(crate::boolean_mode(&node.values), Ok((nodecraft_bool::BooleanOp::Union, false))) {
            continue;
        }
        let mut readers = index.outgoing(&node.id);
        let Some(reader) = readers.next() else { continue };
        if readers.next().is_some() {
            continue;
        }
        let parent_id = reader.to.clone();
        let Some(parent) = index.node(&parent_id) else {
            continue;
        };
        if parent.suppressed
            || !matches!(parent.node_type.as_str(), "boolean" | "fuse")
            || !matches!(crate::boolean_mode(&parent.values), Ok((nodecraft_bool::BooleanOp::Union, false)))
        {
            continue;
        }
        // The parent must take this node on A or B and have both operands: a Union missing an
        // operand is a pass-through and folding changes what it means.
        let has_a = index.first_incoming(&parent_id, "a").is_some();
        let has_b = index.first_incoming(&parent_id, "b").is_some();
        let on_operand = index.outgoing(&node.id).any(|link| {
            link.to == parent_id && (link.input == "a" || link.input == "b")
        });
        if has_a && has_b && on_operand {
            folded.insert(node.id.clone());
        }
    }
    folded
}

/// The leaves of a Union run that may replace `parent`'s node-by-node chain with one
/// `union_all`, or `None` when there is nothing to fuse (fewer than three leaves, or a leaf is
/// itself the parent's only operand). Requested intermediates stay computable: they are in
/// `request.targets` and are never folded away.
pub(crate) fn fused_union_leaves(index: &RequestIndex<'_>, parent: &str, a: &str, b: &str) -> Option<Vec<String>> {
    let mut leaves = Vec::new();
    leaves.extend(union_leaves(index, a, parent).unwrap_or_else(|| vec![a.to_owned()]));
    leaves.extend(union_leaves(index, b, parent).unwrap_or_else(|| vec![b.to_owned()]));
    (leaves.len() >= 3).then_some(leaves)
}
