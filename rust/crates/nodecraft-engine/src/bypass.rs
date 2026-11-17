//! Suppression and error bypass semantics.
//!
//! A suppressed or failed feature must pass through the nearest meaningful
//! input instead of blacking out the graph. Generators remain terminal.

use nodecraft_protocol::Node;

use crate::RequestIndex;

/// The input a suppressed node passes through, if suppression means
/// passthrough for this node type. Generators return `None` and are turned off.
pub(crate) fn suppression_source(node: &Node, index: &RequestIndex<'_>) -> Option<String> {
    // A registered node declares its pass-through. An unported node has no
    // declared source and is treated as a generator (off when suppressed).
    crate::nodes::find(&node.node_type).and_then(|builtin| {
        builtin.suppression_source(crate::nodes::NodeView::from(node), index)
    })
}

/// The input an errored node passes through.
///
/// A failing feature must not black out the whole graph: its output becomes the
/// nearest meaningful input, the same way a suppressed node behaves. Generators
/// without an input stay terminal and report only their own error.
pub(crate) fn bypass_source(node: &Node, index: &RequestIndex<'_>) -> Option<String> {
    suppression_source(node, index).or_else(|| {
        index
            .first_incoming_any(&node.id, &["shape", "profile"])
            .map(|link| link.from.clone())
    })
}

pub(crate) fn effective_output_suppressed(index: &RequestIndex<'_>, start: &str) -> bool {
    // Walk the concrete upstream chain. The visited set is the termination
    // argument: a graph is a DAG, and a revisit can only come from malformed or
    // cyclic input, where reporting the output as suppressed is the safe
    // fallback. There is deliberately no step budget: a long but finite chain
    // of live Translate nodes must still reach the primitive behind it.
    let mut visited = std::collections::BTreeSet::new();
    let mut current = start.to_string();
    loop {
        if !visited.insert(current.clone()) {
            return true;
        }
        let Some(node) = index.node(&current) else { return true };
        if node.suppressed {
            if let Some(source) = suppression_source(node, index) {
                current = source;
                continue;
            }
            return true;
        }
        if crate::nodes::transform_role(crate::nodes::NodeView::from(node))
            == Some(crate::nodes::TransformRole::Translate)
        {
            let Some(link) = index.first_incoming(&current, "shape") else {
                return true;
            };
            current = link.from.clone();
            continue;
        }
        return false;
    }
}
