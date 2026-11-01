//! Close and orient a display mesh after its faces are tessellated.
//!
//! B-rep edges are sampled once in `face_loops` and shared by every adjacent face, so
//! edge boundaries already start out on exactly matching positions. This pass is the
//! residual cleanup for T-junctions, tiny corner gaps and stale face-use winding, not
//! the primary edge-alignment mechanism:
//!
//! 1. **Weld** vertices that coincide or lie within the body's stitching tolerance, so
//!    faces that meet at a B-rep edge share indices. The tolerance is relative to the
//!    body's extent; STEP files routinely represent one edge by two curves a fraction
//!    of a millimetre apart.
//! 2. **Split T-junctions**: a boundary vertex of one face that lies on another face's
//!    boundary edge is inserted into that edge, repeating until the two sides agree.
//!    Only vertices that belong to a different face are inserted, so a face's own
//!    tessellation is never fragmented.
//! 3. **Cap** the small closed boundary loops left at corners where three trimmed
//!    faces meet.
//! 4. **Orient** each shell consistently across shared boundaries and then outward.
//!
//! Normals are recomputed from the oriented triangles at the end: after welding, one
//! vertex serves several faces, so a single normal has to be the average there.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nodecraft_protocol::Mesh;

/// Weld radius as a fraction of the body's largest extent.
const WELD_RELATIVE: f64 = 2.0e-3;
/// T-junction insertion radius as a fraction of the body's largest extent.
const EDGE_RELATIVE: f64 = 1.5e-3;
/// T-junction passes; each pass may split each current triangle once.
const MAX_PASSES: usize = 8;

/// Weld, close and orient the mesh in place.
pub(crate) fn stitch_display_mesh(
    mesh: &mut Mesh,
    _kernel: &solidcore::kernel::Kernel,
    _brep: &solidcore::Brep,
) {
    if mesh.faces.is_empty() {
        return;
    }
    let extent = {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis] as f64);
                max[axis] = max[axis].max(vertex[axis] as f64);
            }
        }
        (0..3).fold(0.0f64, |acc, axis| acc.max(max[axis] - min[axis]))
    };
    if !extent.is_finite() || extent <= 0.0 {
        return;
    }
    let weld_eps = (extent * WELD_RELATIVE).max(1.0e-9);

    // 1. Weld by position. Coincident points from two faces' loops are bit-identical
    // after the f32 conversion, but two faces that do not share a B-rep edge can
    // represent the same curve with a tiny offset; merge those within `weld_eps` too.
    let mut buckets: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    let mut weld: Vec<u32> = Vec::with_capacity(mesh.vertices.len());
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for vertex in &mesh.vertices {
        let key = [
            (vertex[0] as f64 / weld_eps).floor() as i64,
            (vertex[1] as f64 / weld_eps).floor() as i64,
            (vertex[2] as f64 / weld_eps).floor() as i64,
        ];
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbour = [key[0] + dx, key[1] + dy, key[2] + dz];
                    let Some(candidates) = buckets.get(&neighbour) else {
                        continue;
                    };
                    for &candidate in candidates {
                        let p = positions[candidate as usize];
                        let distance = ((p[0] as f64 - vertex[0] as f64).powi(2)
                            + (p[1] as f64 - vertex[1] as f64).powi(2)
                            + (p[2] as f64 - vertex[2] as f64).powi(2))
                        .sqrt();
                        if distance <= weld_eps {
                            found = Some(candidate);
                            break 'search;
                        }
                    }
                }
            }
        }
        let id = match found {
            Some(id) => id,
            None => {
                let id = positions.len() as u32;
                positions.push(*vertex);
                buckets.entry(key).or_default().push(id);
                id
            }
        };
        weld.push(id);
    }

    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    let mut groups: Vec<String> = Vec::with_capacity(mesh.faces.len());
    for (index, face) in mesh.faces.iter().enumerate() {
        let mapped = [
            weld[face[0] as usize],
            weld[face[1] as usize],
            weld[face[2] as usize],
        ];
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[2] == mapped[0] {
            continue;
        }
        if triangle_normal(
            positions[mapped[0] as usize],
            positions[mapped[1] as usize],
            positions[mapped[2] as usize],
        )
        .is_none()
        {
            continue;
        }
        faces.push(mapped);
        groups.push(mesh.face_groups.get(index).cloned().unwrap_or_default());
    }
    if faces.is_empty() {
        return;
    }

    // 2. Split T-junctions until every boundary edge is shared.
    let edge_tolerance = (extent * EDGE_RELATIVE).max(weld_eps * 4.0);
    for _ in 0..MAX_PASSES {
        if !split_one_pass(&mut positions, &mut faces, &mut groups, weld_eps, edge_tolerance) {
            break;
        }
    }

    // 3. Drop the slivers the splits may have left, then cap the small boundary loops
    // left between three trimmed faces.
    drop_degenerate_faces(&mut faces, &mut groups, &positions);
    fill_small_holes(&mut positions, &mut faces, &mut groups, extent);

    // Keep cap fans even when a corner slit is collinear: dropping those triangles
    // removes the only faces that pair the open boundary edges.

    // 5. Orient every connected shell consistently, then outward.
    orient_faces(&mut faces, &groups, &positions);

    // 6. Recompute normals from the final, correctly wound triangles.
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for face in &faces {
        let [a, b, c] = *face;
        let Some(normal) = triangle_normal(
            positions[a as usize],
            positions[b as usize],
            positions[c as usize],
        ) else {
            continue;
        };
        for node in [a, b, c] {
            let slot = &mut normals[node as usize];
            slot[0] += normal[0];
            slot[1] += normal[1];
            slot[2] += normal[2];
        }
    }
    for normal in &mut normals {
        let length =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 1.0e-12 {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            *normal = [0.0, 0.0, 1.0];
        }
    }

    // 7. Remap the display edges and write the result back.
    let mut edges = Vec::with_capacity(mesh.brep_edges.len());
    for edge in &mesh.brep_edges {
        let mut remapped = Vec::with_capacity(edge.segments.len());
        for segment in &edge.segments {
            let (a, b) = (
                weld.get(segment[0] as usize).copied().unwrap_or(segment[0]),
                weld.get(segment[1] as usize).copied().unwrap_or(segment[1]),
            );
            if a != b {
                remapped.push([a, b]);
            }
        }
        if !remapped.is_empty() {
            let mut edge = edge.clone();
            edge.segments = remapped;
            edges.push(edge);
        }
    }
    mesh.vertices = positions;
    mesh.normals = normals;
    mesh.faces = faces;
    mesh.face_groups = groups;
    mesh.brep_edges = edges;
    mesh.vertices_data = None;
    mesh.normals_data = None;
}

/// One T-junction pass: insert at most one existing boundary vertex into each open
/// boundary edge. Returns whether anything was split.
fn split_one_pass(
    positions: &[[f32; 3]],
    faces: &mut Vec<[u32; 3]>,
    groups: &mut Vec<String>,
    weld_eps: f64,
    edge_tolerance: f64,
) -> bool {
    // Directed-edge census: balance is the number of forward traversals minus the
    // backward ones. A closed display edge has balance zero; a boundary edge +-1.
    let mut census: HashMap<(u32, u32), (i32, usize)> = HashMap::new();
    for (index, face) in faces.iter().enumerate() {
        for (from, to) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let (lo, hi) = (from.min(to), from.max(to));
            if lo == hi {
                continue;
            }
            let entry = census.entry((lo, hi)).or_insert((0, index));
            entry.0 += if from == lo { 1 } else { -1 };
        }
    }
    let open: Vec<((u32, u32), usize)> = census
        .iter()
        .filter(|(_, (balance, _))| (*balance).abs() == 1)
        .map(|(&key, &(_, index))| (key, index))
        .collect();
    if open.is_empty() {
        return false;
    }
    // Boundary vertices: endpoints of open edges.
    let mut boundary: BTreeSet<u32> = BTreeSet::new();
    for ((lo, hi), _) in &open {
        boundary.insert(*lo);
        boundary.insert(*hi);
    }
    if boundary.len() < 3 {
        return false;
    }

    // A coarse spatial hash of boundary vertices, cell size on the order of the
    // insertion tolerance.
    let cell = (edge_tolerance * 4.0).max(weld_eps * 8.0);
    let mut buckets: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for &index in &boundary {
        let p = positions[index as usize];
        let key = [
            (p[0] as f64 / cell).floor() as i64,
            (p[1] as f64 / cell).floor() as i64,
            (p[2] as f64 / cell).floor() as i64,
        ];
        buckets.entry(key).or_default().push(index);
    }

    // A vertex's incident display groups. A T-junction is between two faces: a vertex
    // that belongs only to the edge's own face is part of that face's tessellation and
    // must never be inserted into its own boundary (that fragments the face into slivers).
    let mut group_ids: HashMap<&str, usize> = HashMap::new();
    let mut face_group: Vec<usize> = Vec::with_capacity(groups.len());
    for group in groups.iter() {
        let next = group_ids.len();
        face_group.push(*group_ids.entry(group.as_str()).or_insert(next));
    }
    let mut vertex_groups: HashMap<u32, BTreeSet<usize>> = HashMap::new();
    for (index, face) in faces.iter().enumerate() {
        for node in face {
            vertex_groups
                .entry(*node)
                .or_default()
                .insert(face_group[index]);
        }
    }

    let mut removed = vec![false; faces.len()];
    let mut replacements: Vec<([u32; 3], String)> = Vec::new();
    let mut changed = false;
    for ((lo, hi), owner) in open {
        if removed[owner] {
            continue;
        }
        let a = positions[lo as usize];
        let b = positions[hi as usize];
        let (ax, ay, az) = (a[0] as f64, a[1] as f64, a[2] as f64);
        let (bx, by, bz) = (b[0] as f64, b[1] as f64, b[2] as f64);
        let (dx, dy, dz) = (bx - ax, by - ay, bz - az);
        let length2 = dx * dx + dy * dy + dz * dz;
        if length2 <= 0.0 {
            continue;
        }
        let length = length2.sqrt();
        // Only split a boundary edge the other face reaches within this fraction of its
        // length, and never accept a vertex nearly at one of the edge's own ends.
        let tolerance = edge_tolerance.min(length * 0.25).max(1.0e-9);
        let t_margin = (tolerance / length).min(0.25);
        let min = [
            ((ax.min(bx) - tolerance) / cell).floor() as i64,
            ((ay.min(by) - tolerance) / cell).floor() as i64,
            ((az.min(bz) - tolerance) / cell).floor() as i64,
        ];
        let max = [
            ((ax.max(bx) + tolerance) / cell).floor() as i64,
            ((ay.max(by) + tolerance) / cell).floor() as i64,
            ((az.max(bz) + tolerance) / cell).floor() as i64,
        ];
        let mut best: Option<(f64, f64, u32)> = None;
        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    let Some(candidates) = buckets.get(&[x, y, z]) else {
                        continue;
                    };
                    for &candidate in candidates {
                        if candidate == lo || candidate == hi {
                            continue;
                        }
                        let groups_of_candidate = match vertex_groups.get(&candidate) {
                            Some(groups) => groups,
                            None => continue,
                        };
                        if groups_of_candidate.len() == 1
                            && groups_of_candidate.contains(&face_group[owner])
                        {
                            continue;
                        }
                        let p = positions[candidate as usize];
                        let (px, py, pz) = (p[0] as f64, p[1] as f64, p[2] as f64);
                        let t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / length2;
                        if !(t > t_margin && t < 1.0 - t_margin) {
                            continue;
                        }
                        let qx = ax + t * dx;
                        let qy = ay + t * dy;
                        let qz = az + t * dz;
                        let distance =
                            ((px - qx).powi(2) + (py - qy).powi(2) + (pz - qz).powi(2)).sqrt();
                        if distance > tolerance {
                            continue;
                        }
                        let score = (distance, (t - 0.5).abs());
                        if best.is_none_or(|(bd, bt, _)| score < (bd, bt)) {
                            best = Some((distance, (t - 0.5).abs(), candidate));
                        }
                    }
                }
            }
        }
        let Some((_, _, split_at)) = best else {
            continue;
        };
        let face = faces[owner];
        let Some((first, second, third)) = face_with_edge(face, lo, hi) else {
            continue;
        };
        removed[owner] = true;
        let group = groups.get(owner).cloned().unwrap_or_default();
        replacements.push(([first, split_at, third], group.clone()));
        replacements.push(([split_at, second, third], group));
        changed = true;
    }
    if !changed {
        return false;
    }
    let mut next_faces = Vec::with_capacity(faces.len() + replacements.len());
    let mut next_groups = Vec::with_capacity(next_faces.capacity());
    for (index, face) in faces.iter().enumerate() {
        if removed[index] {
            continue;
        }
        next_faces.push(*face);
        next_groups.push(groups[index].clone());
    }
    for (face, group) in replacements {
        next_faces.push(face);
        next_groups.push(group);
    }
    *faces = next_faces;
    *groups = next_groups;
    true
}

/// The triangle's vertices rotated so the edge `lo -> hi` (in either direction) is the
/// first two, keeping the winding, and the third vertex.
fn face_with_edge(face: [u32; 3], lo: u32, hi: u32) -> Option<(u32, u32, u32)> {
    for rotation in 0..3 {
        let a = face[rotation % 3];
        let b = face[(rotation + 1) % 3];
        let c = face[(rotation + 2) % 3];
        // Keep the triangle's winding: rotating the cycle does, swapping two vertices
        // would not, and a reversed replacement triangle corrupts the signed volume.
        if (a == lo && b == hi) || (a == hi && b == lo) {
            return Some((a, b, c));
        }
    }
    None
}

/// Remove triangles with no measurable area; a T-junction split can leave a sliver.
fn drop_degenerate_faces(faces: &mut Vec<[u32; 3]>, groups: &mut Vec<String>, positions: &[[f32; 3]]) {
    let mut kept_faces = Vec::with_capacity(faces.len());
    let mut kept_groups = Vec::with_capacity(groups.len());
    for (index, face) in faces.iter().enumerate() {
        let [a, b, c] = *face;
        let u = [
            positions[b as usize][0] - positions[a as usize][0],
            positions[b as usize][1] - positions[a as usize][1],
            positions[b as usize][2] - positions[a as usize][2],
        ];
        let v = [
            positions[c as usize][0] - positions[a as usize][0],
            positions[c as usize][1] - positions[a as usize][1],
            positions[c as usize][2] - positions[a as usize][2],
        ];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let area2 = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if area2 <= 1.0e-9 {
            continue;
        }
        kept_faces.push(*face);
        kept_groups.push(groups[index].clone());
    }
    *faces = kept_faces;
    *groups = kept_groups;
}

/// Cap the small boundary loops a trimmed face leaves at a three-face corner.

fn fill_small_holes(
    positions: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    groups: &mut Vec<String>,
    extent: f64,
) {
    let limit = extent * 2.5e-1;
    for _ in 0..4 {
        deduplicate_faces(faces, groups);
        // Directed open edges: `balance == 1` means the lo -> hi direction is present.
        let mut census: HashMap<(u32, u32), i32> = HashMap::new();
        for face in faces.iter() {
            for (from, to) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let (lo, hi) = (from.min(to), from.max(to));
                if lo == hi {
                    continue;
                }
                *census.entry((lo, hi)).or_insert(0) += if from == lo { 1 } else { -1 };
            }
        }
        let mut outgoing: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut directed: Vec<(u32, u32)> = Vec::new();
        for (&(lo, hi), &balance) in &census {
            let (from, to) = if balance == 1 {
                (lo, hi)
            } else if balance == -1 {
                (hi, lo)
            } else {
                continue;
            };
            outgoing.entry(from).or_default().push(to);
            directed.push((from, to));
        }
        if directed.is_empty() {
            break;
        }
        let mut used: BTreeSet<(u32, u32)> = BTreeSet::new();
        let mut added = false;
        for &(start, first) in &directed {
            if used.contains(&(start, first)) {
                continue;
            }
            // Walk the existing directed open edges until the loop returns to `start`.
            // At a branch, take the unused edge that turns least, so a loop does not jump
            // into a neighbouring slit.
            let mut chain = vec![start];
            used.insert((start, first));
            let mut current = first;
            let mut closed = false;
            for _ in 0..directed.len() {
                if current == start {
                    closed = true;
                    break;
                }
                chain.push(current);
                let previous = chain
                    .get(chain.len().wrapping_sub(2))
                    .copied()
                    .unwrap_or(current);
                let Some(nexts) = outgoing.get(&current) else {
                    break;
                };
                let mut best: Option<(f64, u32)> = None;
                for &next in nexts {
                    if next == previous || used.contains(&(current, next)) {
                        continue;
                    }
                    let a = positions[previous as usize];
                    let b = positions[current as usize];
                    let c = positions[next as usize];
                    let incoming = [
                        b[0] as f64 - a[0] as f64,
                        b[1] as f64 - a[1] as f64,
                        b[2] as f64 - a[2] as f64,
                    ];
                    let outgoing = [
                        c[0] as f64 - b[0] as f64,
                        c[1] as f64 - b[1] as f64,
                        c[2] as f64 - b[2] as f64,
                    ];
                    let cross = [
                        incoming[1] * outgoing[2] - incoming[2] * outgoing[1],
                        incoming[2] * outgoing[0] - incoming[0] * outgoing[2],
                        incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                    ];
                    let angle =
                        (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                    if best.is_none_or(|(best_angle, _)| angle < best_angle) {
                        best = Some((angle, next));
                    }
                }
                let Some((_, next)) = best else {
                    break;
                };
                used.insert((current, next));
                current = next;
            }
            if !closed || chain.len() < 3 {
                continue;
            }
            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];
            for &index in &chain {
                let p = positions[index as usize];
                for axis in 0..3 {
                    min[axis] = min[axis].min(p[axis] as f64);
                    max[axis] = max[axis].max(p[axis] as f64);
                }
            }
            let diagonal = ((max[0] - min[0]).powi(2)
                + (max[1] - min[1]).powi(2)
                + (max[2] - min[2]).powi(2))
            .sqrt();
            if diagonal > limit {
                continue;
            }
            let mut center = [0.0f64; 3];
            for &index in &chain {
                let p = positions[index as usize];
                center[0] += p[0] as f64;
                center[1] += p[1] as f64;
                center[2] += p[2] as f64;
            }
            let count = chain.len() as f64;
            let mut center = [
                center[0] / count,
                center[1] / count,
                center[2] / count,
            ];
            // A three-face corner can leave a collinear slit. Its cap fan would be
            // zero-area and get dropped again, reopening the seam. Nudge the cap vertex
            // a negligible amount off the slit so the cap triangles are real.
            let mut fan_normal = [0.0f64; 3];
            for index in 0..chain.len() {
                let a = positions[chain[index] as usize];
                let b = positions[chain[(index + 1) % chain.len()] as usize];
                let c = positions[chain[(index + 2) % chain.len()] as usize];
                let u = [b[0] as f64 - a[0] as f64, b[1] as f64 - a[1] as f64, b[2] as f64 - a[2] as f64];
                let v = [c[0] as f64 - a[0] as f64, c[1] as f64 - a[1] as f64, c[2] as f64 - a[2] as f64];
                fan_normal[0] += u[1] * v[2] - u[2] * v[1];
                fan_normal[1] += u[2] * v[0] - u[0] * v[2];
                fan_normal[2] += u[0] * v[1] - u[1] * v[0];
            }
            let fan_length = (fan_normal[0].powi(2) + fan_normal[1].powi(2) + fan_normal[2].powi(2)).sqrt();
            if fan_length <= 1.0e-6 * extent * extent {
                let a = positions[chain[0] as usize];
                let b = positions[chain[1] as usize];
                let edge = [b[0] as f64 - a[0] as f64, b[1] as f64 - a[1] as f64, b[2] as f64 - a[2] as f64];
                let axis = if edge[0].abs() < edge[1].abs() && edge[0].abs() < edge[2].abs() {
                    [1.0, 0.0, 0.0]
                } else if edge[1].abs() < edge[2].abs() {
                    [0.0, 1.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                };
                let mut normal = [
                    edge[1] * axis[2] - edge[2] * axis[1],
                    edge[2] * axis[0] - edge[0] * axis[2],
                    edge[0] * axis[1] - edge[1] * axis[0],
                ];
                let normal_len = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
                if normal_len > 0.0 {
                    for axis in 0..3 { center[axis] += normal[axis] / normal_len * extent * 1.0e-4; }
                }
            }
            let center = [center[0] as f32, center[1] as f32, center[2] as f32];
            let center_index = positions.len() as u32;
            positions.push(center);
            for index in 0..chain.len() {
                let a = chain[index];
                let b = chain[(index + 1) % chain.len()];
                let Some(owner) = faces.iter().position(|face| {
                    let [x, y, z] = *face;
                    (x == a && y == b) || (y == a && z == b) || (z == a && x == b)
                }) else {
                    continue;
                };
                let group = groups.get(owner).cloned().unwrap_or_default();
                // Traverse the existing open edge backwards.
                faces.push([center_index, b, a]);
                groups.push(group);
                added = true;
            }
        }
        if !added {
            break;
        }
    }
}

/// Remove exactly repeated triangles; a cap fan over a branching loop can emit the
/// same triangle twice, which shows up as a non-manifold edge balance of +-2.
fn deduplicate_faces(faces: &mut Vec<[u32; 3]>, groups: &mut Vec<String>) {
    let mut seen: BTreeSet<[u32; 3]> = BTreeSet::new();
    let mut next_faces = Vec::with_capacity(faces.len());
    let mut next_groups = Vec::with_capacity(groups.len());
    for (index, face) in faces.iter().enumerate() {
        let mut key = *face;
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        next_faces.push(*face);
        next_groups.push(groups[index].clone());
    }
    *faces = next_faces;
    *groups = next_groups;
}

/// Make every face consistently wound across shared boundaries, then each shell outward.
///
/// The decision is taken per B-rep face, not per triangle: a face's own tessellation is
/// already internally consistent, and the display mesh may be fragmented into open
/// patches by T-junctions, so a triangle graph would split a shell into islands with
/// meaningless individual volumes. Pair votes across the boundary edges two faces share
/// say whether their windings agree; a flood fill over that face graph makes the shell
/// consistent, and its signed volume picks the outward sign.
fn orient_faces(faces: &mut [[u32; 3]], groups: &[String], positions: &[[f32; 3]]) {
    if faces.is_empty() || groups.len() != faces.len() {
        return;
    }
    let mut group_ids: HashMap<&str, usize> = HashMap::new();
    let mut face_group: Vec<usize> = Vec::with_capacity(faces.len());
    for group in groups {
        let next = group_ids.len();
        face_group.push(*group_ids.entry(group.as_str()).or_insert(next));
    }
    let group_count = group_ids.len();

    // Edge instances, then a vote per pair of faces that share an edge.
    let mut edge_faces: BTreeMap<(u32, u32), Vec<(usize, bool)>> = BTreeMap::new();
    for (index, face) in faces.iter().enumerate() {
        for (from, to) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let (lo, hi) = (from.min(to), from.max(to));
            if lo == hi {
                continue;
            }
            edge_faces.entry((lo, hi)).or_default().push((index, from == lo));
        }
    }
    let mut votes: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for instances in edge_faces.values() {
        if instances.len() != 2 {
            continue;
        }
        let (left, left_forward) = instances[0];
        let (right, right_forward) = instances[1];
        let (left_group, right_group) = (face_group[left], face_group[right]);
        if left_group == right_group {
            continue;
        }
        let key = (left_group.min(right_group), left_group.max(right_group));
        // Opposite traversal directions along the shared edge are consistent windings.
        let consistent = left_forward != right_forward;
        *votes.entry(key).or_insert(0) += if consistent { 1 } else { -1 };
    }

    let mut neighbours: Vec<Vec<(usize, bool)>> = vec![Vec::new(); group_count];
    for (&(left, right), &vote) in &votes {
        if vote == 0 {
            continue;
        }
        let consistent = vote > 0;
        neighbours[left].push((right, consistent));
        neighbours[right].push((left, consistent));
    }

    let mut flipped = vec![false; group_count];
    let mut visited = vec![false; group_count];
    let mut components: Vec<Vec<usize>> = Vec::new();
    for start in 0..group_count {
        if visited[start] || neighbours[start].is_empty() {
            continue;
        }
        visited[start] = true;
        let mut component = Vec::new();
        let mut queue = vec![start];
        while let Some(index) = queue.pop() {
            component.push(index);
            for &(neighbour, consistent) in &neighbours[index] {
                if visited[neighbour] {
                    continue;
                }
                visited[neighbour] = true;
                flipped[neighbour] = flipped[index] ^ !consistent;
                queue.push(neighbour);
            }
        }
        components.push(component);
    }

    // A shell with cavities keeps the outer shell positive and each void negative, so
    // only the sign of the whole shell is chosen here; nested-shell handling is left to
    // the caller's solid, which is one body at a time.
    for component in &components {
        let mut volume = 0.0f64;
        let mut touched = false;
        for (index, face) in faces.iter().enumerate() {
            if !component.contains(&face_group[index]) {
                continue;
            }
            touched = true;
            let mut face = *face;
            if flipped[face_group[index]] {
                face.swap(1, 2);
            }
            volume += signed_tetra(
                positions[face[0] as usize],
                positions[face[1] as usize],
                positions[face[2] as usize],
            );
        }
        if touched && volume < 0.0 {
            for &index in component {
                flipped[index] = !flipped[index];
            }
        }
    }

    for (index, face) in faces.iter_mut().enumerate() {
        if flipped[face_group[index]] {
            face.swap(1, 2);
        }
    }
}

fn signed_tetra(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f64 {
    let (ax, ay, az) = (a[0] as f64, a[1] as f64, a[2] as f64);
    let (bx, by, bz) = (b[0] as f64, b[1] as f64, b[2] as f64);
    let (cx, cy, cz) = (c[0] as f64, c[1] as f64, c[2] as f64);
    (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0
}

fn triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length <= 1.0e-12 {
        return None;
    }
    Some([normal[0] / length, normal[1] / length, normal[2] / length])
}
