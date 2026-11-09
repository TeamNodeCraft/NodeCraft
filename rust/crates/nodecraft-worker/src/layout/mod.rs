//! Top-down layered layout for NodeCraft's node graph.
//!
//! Pipeline: longest-path ranking, barycentre ordering, median x-placement,
//! then a small organic x-refinement. Inputs sit above consumers, so every cable
//! flows downwards; joins are centred below their inputs.
pub mod organic;

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone)]
struct Node { id: String, width: f64, height: f64, rank: usize, x: f64, y: f64 }

fn number(value: Option<&Value>, default: f64) -> f64 {
    value.and_then(Value::as_f64).filter(|value| value.is_finite() && *value > 0.0).unwrap_or(default)
}

fn packed_box(node: usize, items: &Vec<Node>, input_tree: &Vec<Vec<usize>>, gap: f64, memo: &mut Vec<Option<(f64, f64)>>) -> (f64, f64) {
    if let Some(value) = memo[node] { return value; }
    if input_tree[node].is_empty() {
        memo[node] = Some((items[node].width, items[node].height));
        return (items[node].width, items[node].height);
    }
    let child_dims: Vec<(f64, f64)> = input_tree[node].iter().map(|&child| packed_box(child, items, input_tree, gap, memo)).collect();
    let area: f64 = child_dims.iter().map(|(w, h)| (w + gap) * (h + gap)).sum();
    let target = (area.sqrt() * 1.45).max(items[node].width);
    let mut cursor = 0.0f64; let mut shelf_y = 0.0f64; let mut shelf_h = 0.0f64; let mut max_w = 0.0f64;
    for (w, h) in &child_dims {
        if cursor > 0.0 && cursor + w > target { shelf_y += shelf_h + gap; cursor = 0.0; shelf_h = 0.0; }
        cursor += w + gap;
        max_w = max_w.max(cursor - gap);
        shelf_h = shelf_h.max(*h);
    }
    let width = max_w.max(items[node].width);
    let height = shelf_y + shelf_h + gap + items[node].height;
    memo[node] = Some((width, height));
    (width, height)
}

fn place_packed(node: usize, left: f64, top: f64, items: &mut Vec<Node>, input_tree: &Vec<Vec<usize>>, dims: &Vec<(f64, f64)>, gap: f64) {
    if input_tree[node].is_empty() {
        items[node].x = left;
        items[node].y = top;
        return;
    }
    let area: f64 = input_tree[node].iter().map(|&child| (dims[child].0 + gap) * (dims[child].1 + gap)).sum();
    let target = (area.sqrt() * 1.45).max(items[node].width);
    let mut cursor = 0.0f64; let mut shelf_y = 0.0f64; let mut shelf_h = 0.0f64;
    for &child in &input_tree[node] {
        let (child_w, child_h) = dims[child];
        if cursor > 0.0 && cursor + child_w > target { shelf_y += shelf_h + gap; cursor = 0.0; shelf_h = 0.0; }
        place_packed(child, left + cursor, top + shelf_y, items, input_tree, dims, gap);
        cursor += child_w + gap;
        shelf_h = shelf_h.max(child_h);
    }
    let children_bottom = top + shelf_y + shelf_h;
    items[node].y = children_bottom + gap;
    items[node].x = left + (dims[node].0 - items[node].width) / 2.0;
}

fn layout(nodes: &[Value], connections: &[Value], spaced: bool, centered: bool) -> Result<Value, String> {
    let mut items: Vec<Node> = nodes.iter().map(|node| Node {
        id: node.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
        width: number(node.get("width"), 200.0),
        height: number(node.get("height"), 130.0),
        rank: 0, x: 0.0, y: 0.0,
    }).filter(|node| !node.id.is_empty()).collect();
    if items.is_empty() { return Ok(json!({"positions": [], "width": 0, "height": 0})); }
    let index: BTreeMap<String, usize> = items.iter().enumerate().map(|(i, n)| (n.id.clone(), i)).collect();
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for link in connections {
        let (Some(from), Some(to)) = (link.get("from").and_then(Value::as_str), link.get("to").and_then(Value::as_str)) else { continue };
        if let (Some(&from), Some(&to)) = (index.get(from), index.get(to)) {
            if from != to { edges.insert((from, to)); }
        }
    }
    let mut parents = vec![Vec::new(); items.len()];
    let mut children = vec![Vec::new(); items.len()];
    let mut indegree = vec![0usize; items.len()];
    for &(from, to) in &edges { parents[to].push(from); children[from].push(to); indegree[to] += 1; }

    // Longest-path ranking from the inputs.
    let mut queue: VecDeque<usize> = (0..items.len()).filter(|&i| indegree[i] == 0).collect();
    let mut topo = Vec::new();
    while let Some(from) = queue.pop_front() {
        topo.push(from);
        for &to in &children[from] {
            items[to].rank = items[to].rank.max(items[from].rank + 1);
            indegree[to] -= 1;
            if indegree[to] == 0 { queue.push_back(to); }
        }
    }
    // Cycle fallback: rank unresolved nodes one below their highest parent.
    for node in 0..items.len() {
        if !topo.contains(&node) {
            items[node].rank = parents[node].iter().map(|&p| items[p].rank + 1).max().unwrap_or(0);
        }
    }
    let rank_count = items.iter().map(|n| n.rank).max().unwrap_or(0) + 1;
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); rank_count];
    for (i, node) in items.iter().enumerate() { layers[node.rank].push(i); }

    // Model order first, then alternating barycentre sweeps.
    for layer in &mut layers { layer.sort_unstable(); }
    for _ in 0..7 {
        for layer in 1..rank_count {
            let rank: BTreeMap<usize, usize> = layers[layer - 1].iter().enumerate().map(|(r, &n)| (n, r)).collect();
            layers[layer].sort_by_key(|&node| {
                if parents[node].is_empty() { usize::MAX / 2 }
                else { parents[node].iter().map(|p| rank.get(p).copied().unwrap_or(0)).sum::<usize>() / parents[node].len() }
            });
        }
        for layer in (0..rank_count.saturating_sub(1)).rev() {
            let rank: BTreeMap<usize, usize> = layers[layer + 1].iter().enumerate().map(|(r, &n)| (n, r)).collect();
            layers[layer].sort_by_key(|&node| {
                if children[node].is_empty() { usize::MAX / 2 }
                else { children[node].iter().map(|c| rank.get(c).copied().unwrap_or(0)).sum::<usize>() / children[node].len() }
            });
        }
    }

    let compact = true;
    // Timeline lanes: each chain keeps one vertical lane. A split opens a new lane;
    // a join reuses one parent lane, and lanes whose branch has ended are recycled.
    let mut lane_of = vec![usize::MAX; items.len()];
    let mut inherited = vec![false; items.len()];
    let mut lane_holder: Vec<Option<usize>> = Vec::new();
    let mut free_lanes: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    let mut next_lane = 0usize;
    let mut pending_children: Vec<usize> = children.iter().map(Vec::len).collect();
    for &node in &topo {
        let node_parents = parents[node].clone();
        let lane = if node_parents.is_empty() {
            if compact { if let Some(lane) = free_lanes.iter().next().copied() { free_lanes.remove(&lane); lane } else { let lane = next_lane; next_lane += 1; lane } }
            else { let lane = next_lane; next_lane += 1; lane }
        } else if node_parents.len() == 1 {
            let parent = node_parents[0];
            if !inherited[parent] {
                inherited[parent] = true;
                lane_of[parent]
            } else if compact { if let Some(lane) = free_lanes.iter().next().copied() { free_lanes.remove(&lane); lane } else { let lane = next_lane; next_lane += 1; lane } }
            else { let lane = next_lane; next_lane += 1; lane }
        } else {
            node_parents.iter().map(|&parent| lane_of[parent]).min().unwrap_or(0)
        };
        if lane_holder.len() <= lane { lane_holder.resize(lane + 1, None); }
        lane_holder[lane] = Some(node);
        lane_of[node] = lane;
        for parent in node_parents {
            pending_children[parent] = pending_children[parent].saturating_sub(1);
            let parent_lane = lane_of[parent];
            if compact && pending_children[parent] == 0 && lane_holder.get(parent_lane) == Some(&Some(parent)) {
                lane_holder[parent_lane] = None;
                free_lanes.insert(parent_lane);
            }
        }
    }
    // Horizontal compaction: pack timeline lanes into shared columns when their
    // occupied rank ranges do not overlap. Disjoint timelines can reuse a column.
    let mut lane_column: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    if compact {
        let mut lane_first: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut lane_min_rank: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut lane_max_rank: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        for node in 0..items.len() {
            let lane = lane_of[node];
            if lane == usize::MAX { continue; }
            let rank = items[node].rank;
            lane_first.entry(lane).or_insert(rank);
            lane_min_rank.entry(lane).and_modify(|value| *value = (*value).min(rank)).or_insert(rank);
            lane_max_rank.entry(lane).and_modify(|value| *value = (*value).max(rank)).or_insert(rank);
        }
        let mut lanes: Vec<usize> = lane_first.keys().copied().collect();
        lanes.sort_by_key(|lane| lane_min_rank[lane]);
        let mut column_end: Vec<usize> = Vec::new();
        for lane in lanes {
            let start = lane_min_rank[&lane];
            let end = lane_max_rank[&lane];
            let column = column_end.iter().position(|&last| last < start).unwrap_or_else(|| {
                column_end.push(0);
                column_end.len() - 1
            });
            lane_column.insert(lane, column);
            column_end[column] = end;
        }
    } else {
        for node in 0..items.len() {
            let lane = lane_of[node];
            if lane != usize::MAX { lane_column.entry(lane).or_insert(lane); }
        }
    }
    let lane_pitch = items.iter().map(|node| node.width).fold(0.0f64, f64::max) + 8.0;
    for node in 0..items.len() {
        let column = lane_column.get(&lane_of[node]).copied().unwrap_or(0);
        items[node].x = column as f64 * lane_pitch;
    }
    // Fanout-based layer Y offsets. A layer's base offset grows with how many
    // downstream cables leave it, so dense fan-out regions get more vertical room.
    let rank_count = items.iter().map(|node| node.rank).max().unwrap_or(0) + 1;
    let mut fanout = vec![0usize; rank_count];
    for &(from, to) in &edges {
        if items[to].rank == items[from].rank + 1 { fanout[items[from].rank] += 1; }
    }
    let mut row_y = vec![0.0f64; rank_count];
    let mut y = 233.0f64;
    for rank in 0..rank_count {
        row_y[rank] = y;
        let gap = if rank + 1 >= rank_count { 0.0 }
            else if rank == 0 { 181.0 + 24.0 }
            else { 180.0 + 40.0 * (fanout[rank] as f64).sqrt() + 24.0 };
        y += gap;
        if spaced && rank + 1 < rank_count {
            y += fanout[rank] as f64 * 6.0;
        }
    }
    // Per-node fanout offset: a node with many outgoing cables lifts upwards, opening
    // vertical space below it. Single-output nodes stay on their layer row.
    let max_height_by_rank: Vec<f64> = (0..rank_count).map(|rank| {
        items.iter().filter(|node| node.rank == rank).map(|node| node.height).fold(130.0f64, f64::max)
    }).collect();
    // First lift high-fanout nodes upwards.
    for node in 0..items.len() {
        let rank = items[node].rank;
        let fanout = children[node].len();
        let mut y = row_y[rank];
        if fanout > 1 {
            let desired = ((fanout - 1) as f64 * 26.0).min(140.0);
            let available = if rank == 0 { desired }
                else { (row_y[rank] - (row_y[rank - 1] + max_height_by_rank[rank - 1] + 16.0)).max(0.0) };
            y -= desired.min(available);
        }
        items[node].y = y;
    }
    // Then move single-output nodes closer to the next row on Y.
    let row_min_y: Vec<f64> = (0..rank_count).map(|rank| {
        items.iter().filter(|node| node.rank == rank).map(|node| node.y).fold(f64::INFINITY, f64::min)
    }).collect();
    for node in 0..items.len() {
        let rank = items[node].rank;
        if rank + 1 >= rank_count || children[node].len() != 1 { continue; }
        let max_down = (row_min_y[rank + 1] - (items[node].y + items[node].height) - 16.0).max(0.0);
        items[node].y += max_down.min(90.0);
    }
    if centered {
        // Build a consumer tree from the result backwards: each node has one primary
        // consumer, and inputs are its children. Subtrees are shelf-packed above joins.
        let mut primary_consumer: Vec<Option<usize>> = vec![None; items.len()];
        for node in 0..items.len() { primary_consumer[node] = children[node].iter().copied().min(); }
        let mut input_tree: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
        for node in 0..items.len() {
            if let Some(consumer) = primary_consumer[node] { input_tree[consumer].push(node); }
        }
        for kids in &mut input_tree { kids.sort_by_key(|&child| (items[child].rank, child)); }
        let roots: Vec<usize> = (0..items.len()).filter(|&node| primary_consumer[node].is_none()).collect();
        let mut memo = vec![None; items.len()];
        let dims: Vec<(f64, f64)> = (0..items.len()).map(|node| packed_box(node, &items, &input_tree, 12.0, &mut memo)).collect();
        let mut cursor = 0.0f64;
        for &root in &roots {
            place_packed(root, cursor, 0.0, &mut items, &input_tree, &dims, 12.0);
            cursor += dims[root].0 + 12.0;
        }
        // Shelf packing can put a non-primary consumer above an input; push it down.
        for _ in 0..80 {
            let mut moved = false;
            for &(upstream, consumer) in &edges {
                let minimum = items[upstream].y + items[upstream].height + 10.0;
                if items[consumer].y < minimum {
                    items[consumer].y = minimum;
                    moved = true;
                }
            }
            if !moved { break; }
        }
        // Resolve any remaining card collisions by pushing the later node down,
        // then restore downstream ordering for its consumers.
        for _ in 0..120 {
            let mut moved = false;
            for i in 0..items.len() {
                for j in (i + 1)..items.len() {
                    let x_overlap = (items[i].x - items[j].x).abs() < (items[i].width.min(items[j].width) - 2.0);
                    let y_overlap = (items[i].y - items[j].y).abs() < (items[i].height.min(items[j].height) - 2.0);
                    if !x_overlap || !y_overlap { continue; }
                    let mover = if items[j].rank > items[i].rank { j }
                        else if items[i].rank > items[j].rank { i }
                        else if items[i].y <= items[j].y { j } else { i };
                    let other = if mover == i { j } else { i };
                    items[mover].y = items[other].y + items[other].height + 10.0;
                    moved = true;
                }
            }
            for &(upstream, consumer) in &edges {
                let minimum = items[upstream].y + items[upstream].height + 10.0;
                if items[consumer].y < minimum { items[consumer].y = minimum; moved = true; }
            }
            if !moved { break; }
        }
    }

    let min_x = items.iter().map(|n| n.x).fold(f64::INFINITY, f64::min);
    for node in &mut items { node.x -= min_x; }
    let width = items.iter().map(|n| n.x + n.width).fold(0.0f64, f64::max);
    let height = items.iter().map(|n| n.y + n.height).fold(0.0f64, f64::max);
    Ok(json!({
        "positions": items.iter().map(|n| json!({"id": n.id, "x": n.x, "y": n.y})).collect::<Vec<_>>(),
        "width": width,
        "height": height,
    }))
}

/// Compute positions for the node graph using one of the Swift app's layout modes.
///
/// `mode` is `"timeline"`, `"centered"` or `"organic"`; unknown modes fall back to
/// the timeline layout.
#[allow(dead_code)] // public worker API; the app calls it through the worker
pub fn compute(mode: &str, nodes: &[Value], connections: &[Value]) -> Result<Value, String> {
    match mode {
        "organic" => organic::layout_organic(nodes, connections),
        "centered" => layout(nodes, connections, false, true),
        _ => layout(nodes, connections, true, false),
    }
}

/// Node-graph auto-layout entry point. Kept in the worker binary so the app
/// ships a single Rust executable; Swift invokes it with the `layout` argument.
pub struct GraphLayout;

impl GraphLayout {
    pub fn run() {
    let mut input = String::new();
    if std::io::Read::read_to_string(&mut std::io::stdin(), &mut input).is_err() { std::process::exit(2); }
    let payload: Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => { eprintln!("invalid layout json: {error}"); std::process::exit(2); }
    };
    let nodes = payload.get("nodes").and_then(Value::as_array).cloned().unwrap_or_default();
    let connections = payload.get("connections").and_then(Value::as_array).cloned().unwrap_or_default();
    let mode = payload.get("mode").and_then(Value::as_str).unwrap_or("timeline-compact");
    let result = match mode {
        "organic" => organic::layout_organic(&nodes, &connections),
        "centered" => layout(&nodes, &connections, false, true),
        _ => layout(&nodes, &connections, true, false),
    };
    match result {
        Ok(result) => println!("{result}"),
        Err(error) => { eprintln!("{error}"); std::process::exit(1); }
    }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn puts_consumers_below_inputs() {
        let nodes = vec![json!({"id":"a"}), json!({"id":"b"}), json!({"id":"c"})];
        let links = vec![json!({"from":"a","to":"b"}), json!({"from":"a","to":"c"})];
        let layout = layout(&nodes, &links, false, false).unwrap();
        let positions = layout["positions"].as_array().unwrap();
        let by = |id: &str| positions.iter().find(|p| p["id"] == id).unwrap();
        assert!(by("a")["y"].as_f64().unwrap() < by("b")["y"].as_f64().unwrap());
        assert!(by("a")["y"].as_f64().unwrap() < by("c")["y"].as_f64().unwrap());
    }
}
