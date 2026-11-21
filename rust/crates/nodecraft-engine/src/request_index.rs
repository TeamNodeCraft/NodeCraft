//! One owned index over a request's nodes and connections.
//!
//! Evaluation used to answer every "node by id" and "connection by target or
//! source" question with a linear scan of the request. On a chain that is one
//! scan per closure node in the exact loop, another in `evaluate_part_only`,
//! another in the `exact_parts` tail, and more inside node implementations.
//! This index is built exactly once per [`crate::Eval`] and shared by every
//! consumer, so each question is a hash lookup and the graph walk stays linear.
//!
//! Semantics deliberately match the scans it replaces:
//!
//! - `by_id` keeps the first node for a duplicate id (`or_insert`), which is
//!   what `nodes.iter().find(|node| node.id == id)` returned.
//! - `incoming`/`outgoing` push connections while walking the request in order,
//!   so callers that take the first match keep seeing request order.

use std::collections::HashMap;

use nodecraft_protocol::{Connection, Node};

#[cfg(test)]
pub(crate) mod test_counters {
    use std::cell::Cell;

    thread_local! {
        static BUILDS: Cell<u64> = const { Cell::new(0) };
        static LOOKUPS: Cell<u64> = const { Cell::new(0) };
    }

    /// Start a fresh measurement on this test thread.
    pub(crate) fn reset() {
        BUILDS.with(|value| value.set(0));
        LOOKUPS.with(|value| value.set(0));
    }

    /// Number of `RequestIndex` builds on this thread since [`reset`].
    pub(crate) fn builds() -> u64 {
        BUILDS.with(Cell::get)
    }

    /// Number of index lookups on this thread since [`reset`].
    pub(crate) fn lookups() -> u64 {
        LOOKUPS.with(Cell::get)
    }

    pub(super) fn note_build() {
        BUILDS.with(|value| value.set(value.get() + 1));
    }

    pub(super) fn note_lookup() {
        LOOKUPS.with(|value| value.set(value.get() + 1));
    }
}

pub(crate) struct RequestIndex<'a> {
    nodes: &'a [Node],
    targets: &'a [String],
    mesh_targets: Option<&'a [String]>,
    by_id: HashMap<&'a str, usize>,
    incoming: HashMap<&'a str, Vec<&'a Connection>>,
    outgoing: HashMap<&'a str, Vec<&'a Connection>>,
}

impl<'a> RequestIndex<'a> {
    pub(crate) fn new(request: &'a nodecraft_protocol::Request) -> Self {
        #[cfg(test)]
        test_counters::note_build();

        // `or_insert` is the "first match wins" rule of the linear scan this
        // index replaces. Building the map from the ordered node list keeps
        // that rule explicit instead of depending on HashMap behaviour.
        let mut by_id = HashMap::with_capacity(request.nodes.len());
        for (index, node) in request.nodes.iter().enumerate() {
            by_id.entry(node.id.as_str()).or_insert(index);
        }

        let mut incoming: HashMap<&'a str, Vec<&'a Connection>> = HashMap::new();
        let mut outgoing: HashMap<&'a str, Vec<&'a Connection>> = HashMap::new();
        for link in &request.connections {
            incoming.entry(link.to.as_str()).or_default().push(link);
            outgoing.entry(link.from.as_str()).or_default().push(link);
        }

        Self {
            nodes: &request.nodes,
            targets: &request.targets,
            mesh_targets: request.mesh_targets.as_deref(),
            by_id,
            incoming,
            outgoing,
        }
    }

    pub(crate) fn targets(&self) -> &'a [String] {
        self.targets
    }

    pub(crate) fn mesh_targets(&self) -> Option<&'a [String]> {
        self.mesh_targets
    }

    /// Every node in request order. Bulk walks (prewarm, effect collection)
    /// use this instead of reaching through the request.
    pub(crate) fn nodes(&self) -> &'a [Node] {
        #[cfg(test)]
        test_counters::note_lookup();
        self.nodes
    }

    pub(crate) fn node(&self, id: &str) -> Option<&'a Node> {
        #[cfg(test)]
        test_counters::note_lookup();
        self.by_id.get(id).map(|&index| &self.nodes[index])
    }

    /// Every connection entering `node_id`, in request order.
    pub(crate) fn incoming(&self, node_id: &str) -> impl Iterator<Item = &'a Connection> + '_ {
        #[cfg(test)]
        test_counters::note_lookup();
        self.incoming.get(node_id).into_iter().flat_map(|links| links.iter().copied())
    }

    /// The first connection entering `node_id` on `input`, in request order.
    pub(crate) fn first_incoming(&self, node_id: &str, input: &str) -> Option<&'a Connection> {
        #[cfg(test)]
        test_counters::note_lookup();
        self.incoming.get(node_id)?.iter().copied().find(|link| link.input == input)
    }

    /// The first connection entering `node_id` on any of `ports`, in request
    /// order. This is the single-port lookup callers with a port preference
    /// list used to express with a filtered linear scan.
    pub(crate) fn first_incoming_any(
        &self,
        node_id: &str,
        ports: &[&str],
    ) -> Option<&'a Connection> {
        #[cfg(test)]
        test_counters::note_lookup();
        self.incoming
            .get(node_id)?
            .iter()
            .copied()
            .find(|link| ports.contains(&link.input.as_str()))
    }

    /// Every connection leaving `node_id`, in request order.
    pub(crate) fn outgoing(&self, node_id: &str) -> impl Iterator<Item = &'a Connection> + '_ {
        #[cfg(test)]
        test_counters::note_lookup();
        self.outgoing.get(node_id).into_iter().flat_map(|links| links.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::{CacheState, Request};

    fn node(id: &str, node_type: &str, value: &str) -> Node {
        Node {
            id: id.to_owned(),
            node_type: node_type.to_owned(),
            values: serde_json::json!({ "label": value }),
            suppressed: false,
            cache: Some(CacheState { key: Some(id.to_owned()), state: None }),
        }
    }

    fn link(from: &str, to: &str, input: &str) -> Connection {
        Connection {
            from: from.to_owned(),
            to: to.to_owned(),
            input: input.to_owned(),
            out: "shape".to_owned(),
        }
    }

    fn chain_request(count: usize) -> Request {
        let nodes = (0..count)
            .map(|index| {
                if index == 0 {
                    node("n0", "primitive", "first")
                } else {
                    node(&format!("n{index}"), "translate", "move")
                }
            })
            .collect();
        let connections = (1..count)
            .map(|index| link(&format!("n{}", index - 1), &format!("n{index}"), "shape"))
            .collect();
        Request {
            request_id: 1,
            tab_id: None,
            graph_revision: None,
            targets: vec![format!("n{}", count - 1)],
            mesh_targets: Some(Vec::new()),
            nodes,
            connections,
            binary_mesh: false,
            project_directory: None,
            progress: false,
            run_exporters: false,
            mesh_tolerance: None,
            preview_detail: None,
            comparison_targets: None,
            bypass_cache: false,
        }
    }

    /// A by-id scan per closure node reappears as a missing index lookup: this
    /// test fails if the exact loop, `evaluate_part_only` or the `exact_parts`
    /// tail goes back to `request.nodes.iter().find`.
    #[test]
    fn one_index_build_and_linear_lookups_for_a_5000_node_chain() {
        const COUNT: usize = 5_000;
        let request = chain_request(COUNT);
        // Warm cache entries for every node. The chain then exercises the real
        // all-hit exact loop (node(), evaluate_part_only() and the exact_parts
        // tail) without paying the kernel's O(N^2) transform-vector copy, which
        // is what the measurement harness isolates separately.
        let mut cache = crate::EngineCache::new();
        for node in &request.nodes {
            let key = node.cache.as_ref().and_then(|cache| cache.key.clone()).expect("cache key");
            cache.insert_part((node.id.clone(), key), std::sync::Arc::new(crate::Part::empty()));
        }

        test_counters::reset();
        let output = crate::evaluate_exact(request, cache, None);

        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(test_counters::builds(), 1, "one index per evaluation");
        let lookups = test_counters::lookups();
        // The all-hit exact path asks the index once in the exact loop, once in
        // `evaluate_part_only`, and once in the `exact_parts` tail. Requiring
        // three lookups per node makes any one of those going back to a linear
        // scan fail this test instead of being hidden by the other two.
        assert!(
            lookups >= 3 * COUNT as u64,
            "expected the loop, evaluate_part_only and exact_parts tail to use the index, \
             got {lookups} lookups for {COUNT} nodes",
        );
        assert!(
            lookups <= 20 * COUNT as u64,
            "index lookups should stay linear: {lookups} for {COUNT} nodes",
        );
    }

    #[test]
    fn every_request_node_id_resolves_and_duplicate_ids_keep_the_first() {
        let request = Request {
            request_id: 2,
            tab_id: None,
            graph_revision: None,
            targets: vec!["dup".into(), "other".into()],
            mesh_targets: None,
            nodes: vec![
                node("dup", "translate", "first"),
                node("other", "translate", "other"),
                node("third", "translate", "third"),
                node("dup", "translate", "second"),
            ],
            connections: vec![link("other", "dup", "shape"), link("third", "dup", "shape")],
            binary_mesh: false,
            project_directory: None,
            progress: false,
            run_exporters: false,
            mesh_tolerance: None,
            preview_detail: None,
            comparison_targets: None,
            bypass_cache: false,
        };
        let index = RequestIndex::new(&request);
        for node in &request.nodes {
            assert!(index.node(&node.id).is_some(), "{id:?} is not resolvable", id = node.id);
        }
        assert_eq!(
            index
                .node("dup")
                .and_then(|node| node.values.get("label"))
                .and_then(serde_json::Value::as_str),
            Some("first"),
            "duplicate ids resolve to the first request node, like the scan did",
        );
        assert_eq!(
            index.first_incoming("dup", "shape").map(|link| link.from.as_str()),
            Some("other"),
        );
        assert_eq!(
            index.incoming("dup").map(|link| link.from.as_str()).collect::<Vec<_>>(),
            vec!["other", "third"],
            "connection order is request order",
        );
        assert_eq!(index.outgoing("other").count(), 1);
    }
}
