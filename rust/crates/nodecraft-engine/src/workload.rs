//! Cheap static workload analysis for dispatch decisions.
//!
//! The profiler never evaluates geometry and never touches a kernel. It only
//! inspects the flat request graph to estimate how much independent work exists.

use nodecraft_protocol::{Connection, Node, Request};
use std::collections::{BTreeSet, HashMap, VecDeque};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkloadProfile {
    pub node_count: usize,
    /// Longest dependency chain in the target closure.
    pub depth: usize,
    /// Largest number of nodes that can run in one dependency level.
    pub max_ready_width: usize,
    /// Relative exact-work estimate from node types.
    pub exact_units: usize,
    /// Rough display-body estimate, used by the mesh pool.
    pub mesh_bodies: usize,
    pub target_count: usize,
}

impl WorkloadProfile {
    #[must_use]
    pub fn recommended_exact_in_flight(&self, workers: usize) -> usize {
        self.max_ready_width.max(1).min(workers.max(1))
    }

    /// Serial chains do not benefit from scheduler dispatch.
    #[must_use]
    pub fn exact_dispatch_is_worthwhile(&self) -> bool {
        self.max_ready_width > 1 && self.exact_units > 1
    }
}

fn exact_weight(node_type: &str) -> usize {
    match node_type {
        "primitive" | "translate" | "mirror" | "groupinput" | "groupoutput" => 1,
        "combine" | "sketch" | "spline" | "exporter" => 1,
        "stepimporter" | "boolean" | "fuse" => 8,
        "extrude" => 6,
        "chamfer" => 8,
        "fillet" | "shell" | "surfaceloft" => 12,
        "loft" => 10,
        "scatter" => 8,
        _ => 2,
    }
}

fn mesh_bodies(node: &Node) -> usize {
    match node.node_type.as_str() {
        "scatter" => node
            .values
            .get("count")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0)
            .round()
            .clamp(1.0, 5000.0) as usize,
        "primitive" | "stepimporter" => 1,
        _ => 0,
    }
}

/// Profile the closure of `request.targets`.
#[must_use]
pub fn profile_request(request: &Request) -> WorkloadProfile {
    let by_id: HashMap<&str, &Node> = request.nodes.iter().map(|node| (node.id.as_str(), node)).collect();
    let mut dependencies: HashMap<&str, Vec<&str>> = HashMap::new();
    for connection in &request.connections {
        if by_id.contains_key(connection.from.as_str()) && by_id.contains_key(connection.to.as_str()) {
            dependencies.entry(connection.to.as_str()).or_default().push(connection.from.as_str());
        }
    }

    let mut closure = BTreeSet::new();
    let mut stack: Vec<&str> = request.targets.iter().map(String::as_str).filter(|id| by_id.contains_key(id)).collect();
    while let Some(id) = stack.pop() {
        if !closure.insert(id) {
            continue;
        }
        if let Some(inputs) = dependencies.get(id) {
            stack.extend(inputs.iter().copied());
        }
    }

    let mut exact_units = 0;
    let mut estimated_mesh_bodies = 0;
    for id in &closure {
        let node = by_id[id];
        exact_units += exact_weight(&node.node_type);
        estimated_mesh_bodies += mesh_bodies(node);
    }

    // Kahn levels over the closure. The widest level is the theoretical
    // upper bound on scheduler parallelism.
    let mut indegree: HashMap<&str, usize> = closure.iter().map(|id| (*id, 0usize)).collect();
    for id in &closure {
        if let Some(inputs) = dependencies.get(id) {
            for input in inputs {
                if closure.contains(input) {
                    *indegree.get_mut(id).expect("closure node") += 1;
                }
            }
        }
    }
    let mut ready: VecDeque<&str> = indegree.iter().filter_map(|(id, degree)| (*degree == 0).then_some(*id)).collect();
    let mut depth = 0;
    let mut max_ready_width = 0;
    let mut processed = 0;
    while !ready.is_empty() {
        let width = ready.len();
        max_ready_width = max_ready_width.max(width);
        depth += 1;
        for _ in 0..width {
            let id = ready.pop_front().expect("ready");
            processed += 1;
            for (to, inputs) in &dependencies {
                if inputs.contains(&id) && closure.contains(to) {
                    let degree = indegree.get_mut(to).expect("closure node");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push_back(to);
                    }
                }
            }
        }
    }
    if processed != closure.len() {
        // Cycles are rejected by evaluation; return the partial profile.
        max_ready_width = max_ready_width.max(1);
    }

    WorkloadProfile {
        node_count: closure.len(),
        depth,
        max_ready_width: max_ready_width.max(1),
        exact_units,
        mesh_bodies: estimated_mesh_bodies,
        target_count: request.targets.len(),
    }
}

/// Profile an already-flat node/connection slice. Useful for tests and for
/// callers that do not have a full protocol request.
#[must_use]
pub fn profile_flat(nodes: &[Node], connections: &[Connection], targets: &[String]) -> WorkloadProfile {
    profile_request(&Request {
        request_id: 0,
        tab_id: None,
        graph_revision: None,
        targets: targets.to_vec(),
        mesh_targets: None,
        nodes: nodes.to_vec(),
        connections: connections.to_vec(),
        binary_mesh: false,
        project_directory: None,
        progress: false,
        run_exporters: true,
        mesh_tolerance: None,
        comparison_targets: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str, node_type: &str) -> Node {
        Node {
            id: id.into(),
            node_type: node_type.into(),
            values: json!({}),
            suppressed: false,
            cache: None,
        }
    }

    fn link(from: &str, to: &str) -> Connection {
        Connection { from: from.into(), to: to.into(), input: "shape".into(), out: "shape".into() }
    }

    #[test]
    fn profiles_depth_and_ready_width() {
        let nodes = vec![
            node("a", "primitive"),
            node("b", "primitive"),
            node("c", "primitive"),
            node("join", "combine"),
            node("move", "translate"),
        ];
        let connections = vec![link("a", "join"), link("b", "join"), link("c", "join"), link("join", "move")];
        let profile = profile_flat(&nodes, &connections, &["move".into()]);
        assert_eq!(profile.node_count, 5);
        assert_eq!(profile.max_ready_width, 3);
        assert_eq!(profile.depth, 3);
        assert!(profile.exact_units >= 5);
        assert_eq!(profile.mesh_bodies, 3);
        assert!(profile.exact_dispatch_is_worthwhile());
    }

    #[test]
    fn serial_chain_is_not_dispatched() {
        let nodes = vec![node("a", "primitive"), node("b", "translate")];
        let connections = vec![link("a", "b")];
        let profile = profile_flat(&nodes, &connections, &["b".into()]);
        assert_eq!(profile.max_ready_width, 1);
        assert!(!profile.exact_dispatch_is_worthwhile());
    }
}
