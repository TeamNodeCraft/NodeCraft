//! Pure document, clipboard and graph-navigation operations.

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct NodeClipboard {
    pub(super) nodes: Vec<GraphNode>,
    #[serde(default)]
    pub(super) connections: Vec<GraphConnection>,
}

pub(super) const PASTE_OFFSET: f64 = 32.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ClipboardAction {
    Copy,
    Cut,
    Paste(String),
}

pub(super) fn clipboard_action(events: &[egui::Event]) -> Option<ClipboardAction> {
    events.iter().fold(None, |action, event| match event {
        egui::Event::Copy => Some(ClipboardAction::Copy),
        egui::Event::Cut => Some(ClipboardAction::Cut),
        egui::Event::Paste(text) => Some(ClipboardAction::Paste(text.clone())),
        _ => action,
    })
}

pub(super) fn delete_selection_commands(
    nodes: &HashSet<String>,
    cables: &HashSet<ConnectionKey>,
) -> Vec<Command> {
    let mut commands: Vec<Command> = nodes
        .iter()
        .cloned()
        .map(|id| Command::RemoveNode { id })
        .collect();
    for key in cables {
        if nodes.contains(&key.from) || nodes.contains(&key.to) {
            continue;
        }
        commands.push(Command::Disconnect {
            from: key.from.clone(),
            to: key.to.clone(),
            input: key.input.clone(),
        });
    }
    commands
}

/// Remove cables that cross the boundary of `selected`, keeping cables whose
/// two endpoints are both selected. Used by the shake-to-detach gesture.
pub(super) fn disconnection_commands_for_selection(
    document: &GraphDocument,
    selected: &HashSet<String>,
) -> Vec<Command> {
    document
        .connections
        .iter()
        .filter(|link| selected.contains(&link.from) != selected.contains(&link.to))
        .map(|link| Command::Disconnect {
            from: link.from.clone(),
            to: link.to.clone(),
            input: link.input.clone(),
        })
        .collect()
}

pub(super) fn parse_node_clipboard(text: &str) -> Result<NodeClipboard, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
    if value.is_array() {
        return Ok(NodeClipboard {
            nodes: serde_json::from_value(value).map_err(|error| error.to_string())?,
            connections: Vec::new(),
        });
    }
    if value.get("nodes").is_some() {
        return serde_json::from_value(value).map_err(|error| error.to_string());
    }
    if value.get("id").is_some() && value.get("type").is_some() {
        return Ok(NodeClipboard {
            nodes: vec![serde_json::from_value(value).map_err(|error| error.to_string())?],
            connections: Vec::new(),
        });
    }
    Err("expected node object, node array, or { nodes: [...] }".to_owned())
}

pub(super) fn clipboard_payload(document: &GraphDocument, selected: &HashSet<&str>) -> Option<NodeClipboard> {
    let nodes: Vec<GraphNode> = document
        .nodes
        .iter()
        .filter(|node| selected.contains(node.id.as_str()))
        .cloned()
        .collect();
    if nodes.is_empty() {
        return None;
    }
    // Copy only connections whose endpoints are both part of the selection.
    // External upstream/downstream links are not owned by the copied subgraph
    // and must not be recreated on paste.
    let connections: Vec<GraphConnection> = document
        .connections
        .iter()
        .filter(|link| {
            selected.contains(link.from.as_str()) && selected.contains(link.to.as_str())
        })
        .cloned()
        .collect();
    Some(NodeClipboard { nodes, connections })
}

pub(super) fn paste_plan(
    document: &GraphDocument,
    payload: NodeClipboard,
    origin: Option<egui::Pos2>,
) -> (Vec<Command>, HashSet<String>, Option<String>) {
    let min_x = payload.nodes.iter().map(|node| node.x).fold(f64::INFINITY, f64::min);
    let min_y = payload.nodes.iter().map(|node| node.y).fold(f64::INFINITY, f64::min);
    let (origin_x, origin_y) = origin
        .map(|position| (f64::from(position.x), f64::from(position.y)))
        .unwrap_or((min_x + PASTE_OFFSET, min_y + PASTE_OFFSET));
    let mut used: HashSet<String> = document.nodes.iter().map(|node| node.id.clone()).collect();
    let mut ids = std::collections::HashMap::new();
    for node in &payload.nodes {
        ids.insert(node.id.clone(), unique_node_id(&node.id, &mut used));
    }
    let mut commands = Vec::new();
    for node in &payload.nodes {
        let id = ids.get(&node.id).cloned().unwrap_or_else(|| node.id.clone());
        commands.push(Command::AddNode {
            node: GraphNode { expressions: Default::default(),
                id,
                x: origin_x + (node.x - min_x),
                y: origin_y + (node.y - min_y),
                ..node.clone()
            },
        });
    }
    for link in payload.connections {
        let to = ids.get(&link.to).cloned().unwrap_or_else(|| link.to.clone());
        let from = ids
            .get(&link.from)
            .cloned()
            .unwrap_or_else(|| link.from.clone());
        let from_exists = ids.contains_key(&link.from)
            || document.nodes.iter().any(|node| node.id == from);
        let to_exists =
            ids.contains_key(&link.to) || document.nodes.iter().any(|node| node.id == to);
        if !from_exists || !to_exists {
            continue;
        }
        commands.push(Command::Connect {
            from,
            to,
            input: link.input,
            out: link.out,
        });
    }
    let selected: HashSet<String> = ids.values().cloned().collect();
    let pasted_last = payload.nodes.last().and_then(|node| ids.get(&node.id)).cloned();
    (commands, selected, pasted_last)
}


/// Resolve a tab drag to a final insertion index and the active index after it.
/// `to` is an insertion slot in the original ordering, so dragging right removes
/// the source before inserting.
pub(super) fn reorder_tab_index(
    len: usize,
    from: usize,
    to: usize,
    active: usize,
) -> Option<(usize, usize)> {
    if from >= len {
        return None;
    }
    let mut insert = to.min(len);
    if from < insert {
        insert -= 1;
    }
    if insert == from {
        return None;
    }
    let active = if active == from {
        insert
    } else if from < active && insert >= active {
        active - 1
    } else if from > active && insert <= active {
        active + 1
    } else {
        active
    };
    Some((insert, active))
}

pub(super) fn document_id_for_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

pub(super) fn is_nodecraft_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("nodecraft"))
}

pub(super) fn is_step_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("step") || extension.eq_ignore_ascii_case("stp"))
}

pub(super) fn unique_node_id(base: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        used.insert(base.to_owned());
        return base.to_owned();
    }
    let mut copy = 1;
    loop {
        let candidate = format!("{base}-copy{copy}");
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        copy += 1;
    }
}


pub(super) fn node_positions(document: &GraphDocument) -> HashMap<String, (f64, f64)> {
    document
        .nodes
        .iter()
        .map(|node| (node.id.clone(), (node.x, node.y)))
        .collect()
}

pub(super) fn moved_positions(
    before: &HashMap<String, (f64, f64)>,
    after: &HashMap<String, (f64, f64)>,
) -> (HashMap<String, (f64, f64)>, HashMap<String, (f64, f64)>) {
    let mut start = HashMap::new();
    let mut end = HashMap::new();
    for (id, position) in after {
        if let Some(previous) = before.get(id) {
            if previous != position {
                start.insert(id.clone(), *previous);
                end.insert(id.clone(), *position);
            }
        }
    }
    (start, end)
}

pub(super) fn downstream_nodes(document: &GraphDocument, start: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut queue = std::collections::VecDeque::from([start.to_owned()]);
    while let Some(node) = queue.pop_front() {
        if !result.insert(node.clone()) {
            continue;
        }
        for connection in document.connections.iter().filter(|link| link.from == node) {
            queue.push_back(connection.to.clone());
        }
    }
    result
}



pub(super) fn target_closure(document: &GraphDocument, target: &str) -> HashSet<String> {
    let mut closure = HashSet::new();
    let mut stack = vec![target.to_owned()];
    while let Some(id) = stack.pop() {
        if !closure.insert(id.clone()) { continue; }
        for link in document.connections.iter().filter(|link| link.to == id) {
            stack.push(link.from.clone());
        }
    }
    closure
}

/// The node the viewer should show when a document is opened. Prefer an
/// interface output, then the last terminal sink (a node with no consumers).
/// Falling back to `nodes.last()` is wrong when the final node is an upstream
/// primitive; it displays that primitive at its local origin instead of the
/// evaluated root.
pub(super) fn initial_viewer_target(document: &GraphDocument) -> Option<String> {
    if let Some(output) = document.interface.outputs.last()
        && document.nodes.iter().any(|node| &node.id == output)
    {
        return Some(output.clone());
    }
    document
        .nodes
        .iter()
        .rev()
        .find(|node| {
            node.node_type != "exporter"
                && !document.connections.iter().any(|link| link.from == node.id)
        })
        .or_else(|| document.nodes.last())
        .map(|node| node.id.clone())
}


pub(super) fn normalize_document_path(mut path: PathBuf) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension("nodecraft");
    }
    path
}
