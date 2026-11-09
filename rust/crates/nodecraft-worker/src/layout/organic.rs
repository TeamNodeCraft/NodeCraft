//! Tidy layer layout for NodeCraft's node graph.
//!
//! Input (stdin): `{ "nodes": [{ "id", "width", "height" }], "connections": [{ "from", "to" }] }`
//! Output (stdout): `{ "positions": [{ "id", "x", "y" }], "width", "height" }`
//!
//! The algorithm is a deliberately small Sugiyama-style layout:
//! 1. layers by longest path from the roots;
//! 2. order each layer by the barycentre of its parents, then refine upward;
//! 3. place each layer left-to-right under the centres of its parents;
//! 4. run one upward centring pass so a parent sits above its children.
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};


#[derive(Clone)]
struct Node { id: String, width: f64, height: f64, layer: usize, x: f64, y: f64 }

fn number(value: Option<&Value>, default: f64) -> f64 {
    value.and_then(Value::as_f64).filter(|value| value.is_finite() && *value > 0.0).unwrap_or(default)
}

fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    fn orient(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    }
    let o1 = orient(a, b, c); let o2 = orient(a, b, d);
    let o3 = orient(c, d, a); let o4 = orient(c, d, b);
    (o1 * o2 < 0.0) && (o3 * o4 < 0.0)
}

fn tangle_score(items: &[Node], edges: &[(usize, usize)]) -> f64 {
    let mut edge_hits = 0usize;
    for (index, &(from, to)) in edges.iter().enumerate() {
        let a = [items[from].x + items[from].width / 2.0, items[from].y + items[from].height];
        let b = [items[to].x + items[to].width / 2.0, items[to].y];
        for &(other_from, other_to) in edges.iter().skip(index + 1) {
            if from == other_from || from == other_to || to == other_from || to == other_to { continue; }
            let c = [items[other_from].x + items[other_from].width / 2.0, items[other_from].y + items[other_from].height];
            let d = [items[other_to].x + items[other_to].width / 2.0, items[other_to].y];
            if segments_cross(a, b, c, d) { edge_hits += 1; }
        }
        for node in 0..items.len() {
            if node == from || node == to { continue; }
            for step in 1..20 {
                let t = step as f64 / 20.0;
                let x = a[0] + (b[0] - a[0]) * t;
                let y = a[1] + (b[1] - a[1]) * t;
                if x > items[node].x && x < items[node].x + items[node].width
                    && y > items[node].y && y < items[node].y + items[node].height {
                    edge_hits += 1;
                    break;
                }
            }
        }
    }
    edge_hits as f64
}

fn overlaps(items: &[Node], node: usize) -> bool {
    for other in 0..items.len() {
        if other == node { continue; }
        if (items[other].x - items[node].x).abs() < (items[other].width + items[node].width) / 2.0 + 6.0
            && (items[other].y - items[node].y).abs() < (items[other].height + items[node].height) / 2.0 + 6.0 {
            return true;
        }
    }
    false
}

pub fn layout_organic(nodes: &[Value], connections: &[Value]) -> Result<Value, String> {
    let mut items: Vec<Node> = nodes.iter().map(|node| Node {
        id: node.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
        width: number(node.get("width"), 216.0),
        height: number(node.get("height"), 138.0),
        layer: 0, x: 0.0, y: 0.0,
    }).filter(|node| !node.id.is_empty()).collect();
    if items.is_empty() { return Ok(json!({"positions": [], "width": 0, "height": 0})); }
    let index: BTreeMap<String, usize> = items.iter().enumerate().map(|(i, n)| (n.id.clone(), i)).collect();
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut original_edges: Vec<(usize, usize)> = Vec::new();
    for link in connections {
        let (Some(from), Some(to)) = (link.get("from").and_then(Value::as_str), link.get("to").and_then(Value::as_str)) else { continue };
        if let (Some(&from), Some(&to)) = (index.get(from), index.get(to)) {
            if from != to { original_edges.push((from, to)); edges.insert((to, from)); }
        }
    }
    let mut parents = vec![Vec::new(); items.len()];
    let mut children = vec![Vec::new(); items.len()];
    let mut indegree = vec![0usize; items.len()];
    for &(from, to) in &edges { parents[to].push(from); children[from].push(to); indegree[to] += 1; }

    // Longest-path layering. A cycle is broken by leaving the node at its current layer.
    let mut queue: VecDeque<usize> = (0..items.len()).filter(|&i| indegree[i] == 0).collect();
    let mut seen = vec![false; items.len()];
    while let Some(from) = queue.pop_front() {
        seen[from] = true;
        for &to in &children[from] {
            items[to].layer = items[to].layer.max(items[from].layer + 1);
            indegree[to] -= 1;
            if indegree[to] == 0 { queue.push_back(to); }
        }
    }
    let fallback_layers: Vec<usize> = items.iter().enumerate().map(|(i, _)| {
        parents[i].iter().map(|&p| items[p].layer + 1).max().unwrap_or(0)
    }).collect();
    for (i, node) in items.iter_mut().enumerate() {
        if !seen[i] { node.layer = node.layer.max(fallback_layers[i]); }
    }
    let layer_count = items.iter().map(|n| n.layer).max().unwrap_or(0) + 1;
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); layer_count];
    for (i, node) in items.iter().enumerate() { layers[node.layer].push(i); }

    // Barycentre ordering, down then up, a few times.
    let mut order: Vec<usize> = (0..items.len()).collect();
    for layer in 1..layer_count {
        layers[layer].sort_by_key(|&i| {
            let p = &parents[i];
            if p.is_empty() { i } else { p.iter().map(|&p| order[p]).sum::<usize>() / p.len() }
        });
    }
    for _ in 0..6 {
        for layer in 1..layer_count {
            let snapshot = layers[layer - 1].clone();
            let mut rank = vec![0usize; items.len()];
            for (r, &i) in snapshot.iter().enumerate() { rank[i] = r; }
            layers[layer].sort_by_key(|&i| {
                let p: Vec<usize> = parents[i].iter().copied().filter(|p| items[*p].layer == layer - 1).collect();
                if p.is_empty() { usize::MAX / 2 } else { p.iter().map(|p| rank[*p]).sum::<usize>() / p.len() }
            });
        }
        for layer in (0..layer_count.saturating_sub(1)).rev() {
            let snapshot = layers[layer + 1].clone();
            let mut rank = vec![0usize; items.len()];
            for (r, &i) in snapshot.iter().enumerate() { rank[i] = r; }
            layers[layer].sort_by_key(|&i| {
                let c: Vec<usize> = children[i].iter().copied().filter(|c| items[*c].layer == layer + 1).collect();
                if c.is_empty() { usize::MAX / 2 } else { c.iter().map(|c| rank[*c]).sum::<usize>() / c.len() }
            });
        }
    }
    for layer in &layers { for (r, &i) in layer.iter().enumerate() { order[i] = r; } }

    // Fan the children of a single parent around its centre so their edges do not
    // collapse into one vertical line. Multi-parent joins still use the average.
    // Bottom-up tidy tree: the result is the root at the bottom. Each node owns one
    // primary consumer, so every node appears once; its inputs are its tree children.
    // Subtrees reserve a horizontal band and branch upward from the result.
    let mut primary_consumer: Vec<Option<usize>> = vec![None; items.len()];
    for node in 0..items.len() {
        primary_consumer[node] = parents[node].iter().copied().filter(|&parent| parent != node).min();
    }
    let mut tree_inputs: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    for node in 0..items.len() {
        if let Some(consumer) = primary_consumer[node] { tree_inputs[consumer].push(node); }
    }
    for inputs in &mut tree_inputs {
        inputs.sort_by_key(|&node| (items[node].layer, node));
    }
    // Canopy density: number of leaves a branch ultimately carries.
    fn canopy_density(node: usize, tree_inputs: &Vec<Vec<usize>>, memo: &mut Vec<Option<usize>>) -> usize {
        if let Some(value) = memo[node] { return value; }
        let value = if tree_inputs[node].is_empty() { 1 } else {
            tree_inputs[node].iter().map(|&child| canopy_density(child, tree_inputs, memo)).sum()
        };
        memo[node] = Some(value);
        value
    }
    let mut density_memo = vec![None; items.len()];
    let density: Vec<usize> = (0..items.len()).map(|node| canopy_density(node, &tree_inputs, &mut density_memo)).collect();

    fn grow(node: usize, root_x: f64, root_y: f64, angle_start: f64, angle_end: f64, depth: usize,
            items: &mut Vec<Node>, tree_inputs: &Vec<Vec<usize>>, density: &Vec<usize>) {
        let middle = (angle_start + angle_end) / 2.0;
        let radius = depth as f64 * 330.0;
        let centre_x = root_x + middle.sin() * radius;
        let centre_y = root_y - middle.cos() * radius;
        items[node].x = centre_x - items[node].width / 2.0;
        items[node].y = centre_y - items[node].height / 2.0;
        let inputs = tree_inputs[node].clone();
        if inputs.is_empty() { return; }
        let total: usize = inputs.iter().map(|&child| density[child]).sum();
        let span = angle_end - angle_start;
        let mut cursor = angle_start;
        for &child in &inputs {
            let share = span * density[child] as f64 / total.max(1) as f64;
            grow(child, root_x, root_y, cursor, cursor + share, depth + 1, items, tree_inputs, density);
            cursor += share;
        }
    }

    let roots: Vec<usize> = (0..items.len()).filter(|&node| primary_consumer[node].is_none()).collect();
    let half_spread = 70.0f64.to_radians();
    for (index, &root) in roots.iter().enumerate() {
        grow(root, index as f64 * 4200.0, 0.0, -half_spread, half_spread, 0, &mut items, &tree_inputs, &density);
    }

    // Collision relaxation: separate overlapping cards and keep cables clear while
    // preserving the grown tree shape.
    let mut centers: Vec<[f64; 2]> = items.iter().map(|node| [node.x + node.width / 2.0, node.y + node.height / 2.0]).collect();
    let anchors = centers.clone();
    let edge_list: Vec<(usize, usize)> = edges.iter().copied().collect();
    for _ in 0..900 {
        // Card overlap separation.
        for left in 0..items.len() {
            for right in (left + 1)..items.len() {
                let dx = centers[right][0] - centers[left][0];
                let dy = centers[right][1] - centers[left][1];
                let min_x = (items[left].width + items[right].width) / 2.0 + 14.0;
                let min_y = (items[left].height + items[right].height) / 2.0 + 14.0;
                if dx.abs() < min_x && dy.abs() < min_y {
                    if dx.abs() / min_x < dy.abs() / min_y {
                        let push = (min_x - dx.abs()) / 2.0 * if dx >= 0.0 { 1.0 } else { -1.0 };
                        centers[left][0] -= push; centers[right][0] += push;
                    } else {
                        let push = (min_y - dy.abs()) / 2.0 * if dy >= 0.0 { 1.0 } else { -1.0 };
                        centers[left][1] -= push; centers[right][1] += push;
                    }
                }
            }
        }
        // Cohesion: keep each branch near where it grew so connected branches stay
        // grouped instead of being scattered by collision forces.
        for node in 0..items.len() {
            centers[node][0] += (anchors[node][0] - centers[node][0]) * 0.03;
            centers[node][1] += (anchors[node][1] - centers[node][1]) * 0.03;
        }
        // Clump siblings: a split branch should stay together.
        for parent in 0..items.len() {
            let inputs = &tree_inputs[parent];
            if inputs.len() < 2 { continue; }
            let mut mid_x = 0.0; let mut mid_y = 0.0;
            for &child in inputs { mid_x += centers[child][0]; mid_y += centers[child][1]; }
            mid_x /= inputs.len() as f64; mid_y /= inputs.len() as f64;
            for &child in inputs {
                centers[child][0] += (mid_x - centers[child][0]) * 0.025;
                centers[child][1] += (mid_y - centers[child][1]) * 0.025;
            }
        }
        // Edge convergence: connected nodes pull toward a consistent branch length.
        for &(from, to) in &edge_list {
            let dx = centers[to][0] - centers[from][0];
            let dy = centers[to][1] - centers[from][1];
            let distance = dx.hypot(dy).max(1.0);
            let correction = (distance - 210.0) * 0.05;
            let ux = dx / distance;
            let uy = dy / distance;
            centers[from][0] += ux * correction;
            centers[from][1] += uy * correction;
            centers[to][0] -= ux * correction;
            centers[to][1] -= uy * correction;
        }
        // Edge-to-card repulsion.
        for &(from, to) in &edge_list {
            let ax = centers[from][0]; let ay = centers[from][1];
            let bx = centers[to][0]; let by = centers[to][1];
            let dx = bx - ax; let dy = by - ay;
            let length_squared = dx * dx + dy * dy + 1e-6;
            for node in 0..items.len() {
                if node == from || node == to { continue; }
                let px = centers[node][0]; let py = centers[node][1];
                let t = (((px - ax) * dx + (py - ay) * dy) / length_squared).clamp(0.0, 1.0);
                let cx = ax + dx * t; let cy = ay + dy * t;
                let ox = px - cx; let oy = py - cy;
                let distance = ox.hypot(oy).max(0.01);
                if distance < 210.0 {
                    let push = (210.0 - distance) * 0.55;
                    centers[node][0] += ox / distance * push;
                    centers[node][1] += oy / distance * push;
                }
            }
        }
    }
    // Final card-only separation so the edge repulsion cannot leave overlaps.
    for _ in 0..300 {
        let mut moved = false;
        for left in 0..items.len() {
            for right in (left + 1)..items.len() {
                let dx = centers[right][0] - centers[left][0];
                let dy = centers[right][1] - centers[left][1];
                let min_x = (items[left].width + items[right].width) / 2.0 + 16.0;
                let min_y = (items[left].height + items[right].height) / 2.0 + 16.0;
                if dx.abs() < min_x && dy.abs() < min_y {
                    if dx.abs() / min_x < dy.abs() / min_y {
                        let push = (min_x - dx.abs()) / 2.0 * if dx >= 0.0 { 1.0 } else { -1.0 };
                        centers[left][0] -= push; centers[right][0] += push;
                    } else {
                        let push = (min_y - dy.abs()) / 2.0 * if dy >= 0.0 { 1.0 } else { -1.0 };
                        centers[left][1] -= push; centers[right][1] += push;
                    }
                    moved = true;
                }
            }
        }
        if !moved { break; }
    }
    for node in 0..items.len() {
        items[node].x = centers[node][0] - items[node].width / 2.0;
        items[node].y = centers[node][1] - items[node].height / 2.0;
    }

    // Local downstream ordering: preserve the organic growth shape and only push a
    // consumer down when one of its inputs would otherwise end up below it.
    let mut indegree = vec![0usize; items.len()];
    let mut downstream: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    for &(upstream, consumer) in &original_edges {
        downstream[upstream].push(consumer);
        indegree[consumer] += 1;
    }
    let mut queue: VecDeque<usize> = (0..items.len()).filter(|&node| indegree[node] == 0).collect();
    let mut topo = Vec::new();
    while let Some(node) = queue.pop_front() {
        topo.push(node);
        for &next in &downstream[node] {
            indegree[next] -= 1;
            if indegree[next] == 0 { queue.push_back(next); }
        }
    }
    for _ in 0..80 {
        for &node in &topo {
            for &consumer in &downstream[node] {
                let minimum = items[node].y + items[node].height + 18.0;
                if items[consumer].y < minimum { items[consumer].y = minimum; }
            }
        }
        // Resolve crowding sideways so vertical organic positions are preserved.
        for left in 0..items.len() {
            for right in (left + 1)..items.len() {
                let dx = items[right].x - items[left].x;
                let dy = items[right].y - items[left].y;
                let min_x = (items[left].width + items[right].width) / 2.0 + 18.0;
                let min_y = (items[left].height + items[right].height) / 2.0 + 18.0;
                if dx.abs() < min_x && dy.abs() < min_y {
                    let push = (min_x - dx.abs()) / 2.0 * if dx >= 0.0 { 1.0 } else { -1.0 };
                    items[left].x -= push;
                    items[right].x += push;
                }
            }
        }
    }
    for &node in &topo {
        for &consumer in &downstream[node] {
            let minimum = items[node].y + items[node].height + 18.0;
            if items[consumer].y < minimum { items[consumer].y = minimum; }
        }
    }

    // Soft same-Y grouping: inputs of one consumer drift toward a shared Y, but the
    // pull strengthens as the number of upstream inputs grows.
    let mut feed_groups: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    for &(upstream, consumer) in &original_edges { feed_groups[consumer].push(upstream); }
    for _ in 0..4 {
        for consumer in 0..items.len() {
            let inputs = feed_groups[consumer].clone();
            let count = inputs.len();
            if count < 2 { continue; }
            let target = inputs.iter().map(|&node| items[node].y).sum::<f64>() / count as f64;
            let strength = 1.0 - 1.0 / (1.0 + (count as f64 - 1.0) * 0.5);
            for &node in &inputs { items[node].y += (target - items[node].y) * strength; }
        }
        // Restore downstream ordering after the Y drift.
        for &node in &topo {
            for &consumer in &downstream[node] {
                let minimum = items[node].y + items[node].height + 10.0;
                if items[consumer].y < minimum { items[consumer].y = minimum; }
            }
        }
        // Resolve overlaps horizontally so the Y rows remain intact.
        for _ in 0..80 {
            let mut moved = false;
            for left in 0..items.len() {
                for right in (left + 1)..items.len() {
                    let dx = items[right].x - items[left].x;
                    let dy = items[right].y - items[left].y;
                    if dx.abs() < (items[left].width + items[right].width) / 2.0 + 16.0
                        && dy.abs() < (items[left].height + items[right].height) / 2.0 + 6.0 {
                        let push = ((items[left].width + items[right].width) / 2.0 + 16.0 - dx.abs()) / 2.0 * if dx >= 0.0 { 1.0 } else { -1.0 };
                        items[left].x -= push; items[right].x += push; moved = true;
                    }
                }
            }
            if !moved { break; }
        }
    }

    // Iterative tangle removal: accept only local moves that reduce edge-edge and
    // edge-card tangles, without overlaps or downstream-order violations.
    let offsets = [(-38.0, 0.0), (38.0, 0.0), (0.0, -38.0), (0.0, 38.0), (-24.0, -24.0), (24.0, -24.0), (-24.0, 24.0), (24.0, 24.0)];
    for _pass in 0..4 {
        let mut improved = false;
        let base = tangle_score(&items, &original_edges);
        for node in 0..items.len() {
            let original = [items[node].x, items[node].y];
            let mut best = base;
            let mut best_offset = (0.0, 0.0);
            for &(dx, dy) in &offsets {
                items[node].x = original[0] + dx;
                items[node].y = original[1] + dy;
                if overlaps(&items, node) { continue; }
                let mut ordered = true;
                for &(upstream, consumer) in &original_edges {
                    if upstream == node && items[consumer].y < items[node].y + items[node].height + 10.0 { ordered = false; break; }
                    if consumer == node && items[node].y < items[upstream].y + items[upstream].height + 10.0 { ordered = false; break; }
                }
                if !ordered { continue; }
                let score = tangle_score(&items, &original_edges);
                if score < best - 0.5 { best = score; best_offset = (dx, dy); }
            }
            items[node].x = original[0] + best_offset.0;
            items[node].y = original[1] + best_offset.1;
            if best < base - 0.5 { improved = true; }
        }
        if !improved { break; }
    }

    let min_x = items.iter().map(|n| n.x).fold(f64::INFINITY, f64::min);
    let min_y = items.iter().map(|n| n.y).fold(f64::INFINITY, f64::min);
    for node in &mut items { node.x -= min_x; node.y -= min_y; }
    let width = items.iter().map(|n| n.x + n.width).fold(0.0f64, f64::max);
    let height = items.iter().map(|n| n.y + n.height).fold(0.0f64, f64::max);
    Ok(json!({
        "positions": items.iter().map(|n| json!({"id": n.id, "x": n.x, "y": n.y})).collect::<Vec<_>>(),
        "width": width,
        "height": height,
    }))
}

