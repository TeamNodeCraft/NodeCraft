//! Ear-clipping triangulation for planar faces with holes.
//!
//! A `PlanarBody` face is one outer loop plus zero or more inner loops (holes). This
//! module turns such a face into triangles by the classic bridge-and-ear-clip method:
//!
//! 1. Project the loops onto the face plane (2D coordinates, right-handed about the
//!    outer loop's Newell normal).
//! 2. Normalize winding internally: outer counter-clockwise, holes clockwise, and drop
//!    repeated/collinear vertices that only add zero-area edges.
//! 3. Bridge every hole into the outer polygon with a zero-width corridor. Holes are
//!    merged left-to-right. Each bridge starts at the hole's leftmost vertex and joins
//!    the nearest visible ring vertex; when nothing is visible from there (a hole
//!    squeezed between its neighbours), the hole's other vertices are tried. Anchors
//!    that are already pinch points are avoided so two holes do not share one vertex.
//! 4. Ear-clip the resulting single ring. Each ear must be convex, contain no reflex
//!    ring vertex, and cut along an internal diagonal. Every cut must also preserve the
//!    ring's signed area exactly: a diagonal that crosses a zero-width bridge corridor
//!    can look locally valid while inverting the remaining ring, and the area check
//!    rejects it. If a weakly simple ring (shared bridge anchors) has no single-vertex
//!    ear, it is split along a valid internal diagonal and the halves are clipped
//!    independently.
//! 5. Check that the triangles' total area equals `outer area - hole areas`; a mismatch
//!    is an error rather than a silently wrong mesh.
//!
//! Holes are always represented; there is no fan-fallback that would ignore them.

use crate::PlanarBody;
use crate::{cross3, dot3, newell_normal, unit};

/// A point projected onto the face plane.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Uv {
    x: f64,
    y: f64,
}

impl Uv {
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

fn cross2(a: Uv, b: Uv) -> f64 {
    a.x * b.y - a.y * b.x
}

fn dot2(a: Uv, b: Uv) -> f64 {
    a.x * b.x + a.y * b.y
}

fn distance2(a: Uv, b: Uv) -> f64 {
    dot2(a.sub(b), a.sub(b))
}

fn same_position(a: Uv, b: Uv, tolerance: f64) -> bool {
    distance2(a, b) <= tolerance * tolerance
}

/// Squared distance from `point` to the segment `a`–`b`.
fn point_segment_distance2(point: Uv, a: Uv, b: Uv) -> f64 {
    let ab = b.sub(a);
    let length2 = dot2(ab, ab);
    if length2 <= 0.0 {
        return distance2(point, a);
    }
    let t = (dot2(point.sub(a), ab) / length2).clamp(0.0, 1.0);
    distance2(
        point,
        Uv { x: a.x + ab.x * t, y: a.y + ab.y * t },
    )
}

fn point_on_segment(point: Uv, a: Uv, b: Uv, tolerance: f64) -> bool {
    point_segment_distance2(point, a, b) <= tolerance * tolerance
}

impl PlanarBody {
    /// Triangulates every face, returning triples of indices into [`PlanarBody::points`].
    ///
    /// Faces are emitted in order, followed by their triangles (winding follows the
    /// stored outer loop). Holes are triangulated away, never dropped.
    pub fn triangulate(&self) -> Result<Vec<[usize; 3]>, String> {
        if self.faces.is_empty() {
            return Err("planar body has no faces".into());
        }
        let mut triangles = Vec::new();
        for face in 0..self.faces.len() {
            triangles.extend(self.face_triangles(face)?);
        }
        Ok(triangles)
    }

    /// Triangulates one face, honoring its holes.
    pub fn face_triangles(&self, face: usize) -> Result<Vec<[usize; 3]>, String> {
        let outer = self
            .faces
            .get(face)
            .ok_or_else(|| format!("face {face} does not exist"))?;
        if outer.len() < 3 {
            return Err(format!(
                "face {face} outer loop has fewer than three vertices"
            ));
        }
        for (hole_index, hole) in self.face_holes(face).iter().enumerate() {
            if hole.len() < 3 {
                return Err(format!(
                    "face {face} hole {hole_index} has fewer than three vertices"
                ));
            }
        }
        // Validate indices before projecting (no panics on malformed bodies).
        for loop_points in self.loops(face) {
            for &index in loop_points {
                if index >= self.points.len() {
                    return Err(format!("face {face} references missing point {index}"));
                }
            }
        }
        let normal = newell_normal(&self.points, outer)
            .ok_or_else(|| format!("face {face} outer loop is degenerate"))?;
        let basis_a = if normal[0].abs() < 0.9 {
            unit([1.0, 0.0, 0.0], "face basis")?
        } else {
            unit([0.0, 1.0, 0.0], "face basis")?
        };
        let basis_a = unit(cross3(basis_a, normal), "face basis")?;
        let basis_b = cross3(normal, basis_a);
        let project = |index: usize| -> Result<Uv, String> {
            let point = *self
                .points
                .get(index)
                .ok_or_else(|| format!("face {face} references missing point {index}"))?;
            Ok(Uv { x: dot3(point, basis_a), y: dot3(point, basis_b) })
        };
        let mut coordinates = Vec::with_capacity(self.points.len());
        for index in 0..self.points.len() {
            coordinates.push(project(index)?);
        }
        let uv = |index: usize| coordinates[index];

        let scale = face_scale(self, face, &coordinates);
        let tolerance = scale * 1.0e-9;
        let area_epsilon = scale * scale * 1.0e-12;

        let outer = clean_loop(outer, &uv, tolerance);
        if outer.len() < 3 {
            return Err(format!("face {face} outer loop collapses to a line"));
        }
        let outer_area = signed_area(&outer, &uv);
        if !(outer_area > area_epsilon) {
            return Err(format!("face {face} outer loop is degenerate (area {outer_area})"));
        }
        let mut holes = Vec::with_capacity(self.face_holes(face).len());
        for (hole_index, hole) in self.face_holes(face).iter().enumerate() {
            let mut hole = hole.clone();
            if signed_area(&hole, &uv) > 0.0 {
                // Ear clipping needs holes opposite to the outer loop.
                hole.reverse();
            }
            let hole = clean_loop(&hole, &uv, tolerance);
            let hole_area = signed_area(&hole, &uv).abs();
            if hole.len() < 3 || !(hole_area > area_epsilon) {
                return Err(format!(
                    "face {face} hole {hole_index} is degenerate (area {hole_area})"
                ));
            }
            holes.push(hole);
        }
        // Merge holes from left to right. A hole's leftmost vertex then only has
        // already-merged loop geometry between itself and the outside, so its horizontal
        // ray cannot run through a hole that has not been bridged yet.
        holes.sort_by(|left, right| {
            let left = leftmost(left, &uv).1;
            let right = leftmost(right, &uv).1;
            left.x
                .partial_cmp(&right.x)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    left.y
                        .partial_cmp(&right.y)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        let mut ring = outer.clone();
        for hole in &holes {
            ring = bridge_hole(&ring, hole, &holes, &uv, tolerance, face)?;
        }
        let mut triangles = Vec::new();
        ear_clip(&ring, &uv, tolerance, area_epsilon, &mut triangles, face)?;

        let expected = outer_area
            - holes
                .iter()
                .map(|hole| signed_area(hole, &uv).abs())
                .sum::<f64>();
        let actual = triangles
            .iter()
            .map(|[a, b, c]| 0.5 * cross2(uv(*b).sub(uv(*a)), uv(*c).sub(uv(*a))).abs())
            .sum::<f64>();
        if (actual - expected).abs() > expected.abs() * 1.0e-9 + area_epsilon {
            return Err(format!(
                "face {face} triangulation area {actual} does not match loop area {expected}"
            ));
        }
        Ok(triangles)
    }
}

/// Removes repeated positions from a loop (including a repeated closing point).
///
/// Bridge corridors and STEP edge sampling can leave zero-length edges, which make ear
/// clipping and bridge visibility tests needlessly fragile. Collinear vertices are kept:
/// they are harmless to the ear clipper and their count is part of the exact triangle
/// counts other readers assert.
fn clean_loop(
    loop_points: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
) -> Vec<usize> {
    let mut cleaned: Vec<usize> = Vec::with_capacity(loop_points.len());
    for &index in loop_points {
        if cleaned
            .last()
            .is_some_and(|&last| same_position(uv(last), uv(index), tolerance))
        {
            continue;
        }
        cleaned.push(index);
    }
    while cleaned.len() > 3
        && same_position(uv(cleaned[0]), uv(cleaned[cleaned.len() - 1]), tolerance)
    {
        cleaned.pop();
    }
    cleaned
}

/// The loop's leftmost vertex (lowest u, then lowest v) and its position in the loop.
fn leftmost(loop_points: &[usize], uv: &impl Fn(usize) -> Uv) -> (usize, Uv) {
    let mut best_position = 0;
    let mut best = uv(loop_points[0]);
    for (position, &index) in loop_points.iter().enumerate().skip(1) {
        let point = uv(index);
        if point.x < best.x || (point.x == best.x && point.y < best.y) {
            best_position = position;
            best = point;
        }
    }
    (best_position, best)
}

/// Largest extent of the face's loops, used to scale tolerances.
fn face_scale(body: &PlanarBody, face: usize, coordinates: &[Uv]) -> f64 {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for loop_points in body.loops(face) {
        for &index in loop_points {
            let point = coordinates[index];
            min[0] = min[0].min(point.x);
            min[1] = min[1].min(point.y);
            max[0] = max[0].max(point.x);
            max[1] = max[1].max(point.y);
        }
    }
    let scale = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();
    if scale > 0.0 && scale.is_finite() { scale } else { 1.0 }
}

/// Signed 2D area of a loop (positive = counter-clockwise in the projected frame).
fn signed_area(loop_points: &[usize], uv: &impl Fn(usize) -> Uv) -> f64 {
    let mut total = 0.0;
    for position in 0..loop_points.len() {
        let a = uv(loop_points[position]);
        let b = uv(loop_points[(position + 1) % loop_points.len()]);
        total += a.x * b.y - b.x * a.y;
    }
    total * 0.5
}

/// Splicing one hole into the ring with a zero-width bridge.
fn bridge_hole(
    ring: &[usize],
    hole: &[usize],
    all_holes: &[Vec<usize>],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    face: usize,
) -> Result<Vec<usize>, String> {
    // Try the hole's vertices left to right. The leftmost vertex is the natural start
    // (its horizontal ray is guaranteed clear), but a hole squeezed between other holes
    // can have no visible ring vertex from there; another vertex of the same hole may
    // see the outer loop through free space.
    let mut starts: Vec<usize> = (0..hole.len()).collect();
    starts.sort_by(|&left, &right| {
        let left = uv(hole[left]);
        let right = uv(hole[right]);
        left.x
            .partial_cmp(&right.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.y
                    .partial_cmp(&right.y)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    let mut bridge: Option<(usize, usize)> = None;
    // Prefer ring vertices that are not already a pinch point. Bridging a second hole to
    // the same anchor makes three ring vertices coincide; such a weakly-simple ring has
    // no single-vertex ear along the pinch, so ear clipping can stall on it.
    for distinct in [true, false] {
        'search: for &start in &starts {
            let hole_vertex = hole[start];
            let hole_point = uv(hole_vertex);
            let mut candidates: Vec<usize> = (0..ring.len()).collect();
            candidates.sort_by(|left, right| {
                distance2(uv(ring[*left]), hole_point)
                    .partial_cmp(&distance2(uv(ring[*right]), hole_point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &candidate in &candidates {
                if distinct && position_count(ring, uv(ring[candidate]), uv, tolerance) > 1 {
                    continue;
                }
                if segment_visible(ring[candidate], hole_vertex, ring, all_holes, uv, tolerance) {
                    bridge = Some((start, candidate));
                    break 'search;
                }
            }
        }
        if bridge.is_some() {
            break;
        }
    }
    if let Some((start, position)) = bridge {
        return Ok(splice_hole(ring, hole, start, position));
    }

    // Last resort: cast a horizontal ray towards -u from the leftmost vertex. Because
    // holes are merged left to right, no unbridged hole can sit between that vertex and
    // the ring, so the ray reaches the ring through free space. The final area check
    // still rejects a genuinely bad cut.
    let (start, hole_point) = leftmost(hole, uv);
    let mut ray_hit: Option<(f64, usize, usize)> = None;
    for position in 0..ring.len() {
        let c = uv(ring[position]);
        let d = uv(ring[(position + 1) % ring.len()]);
        if (c.y > hole_point.y) == (d.y > hole_point.y) {
            continue;
        }
        let x = c.x + (hole_point.y - c.y) / (d.y - c.y) * (d.x - c.x);
        if x <= hole_point.x + tolerance && ray_hit.is_none_or(|(best, _, _)| x > best) {
            ray_hit = Some((x, position, (position + 1) % ring.len()));
        }
    }
    if let Some((_, left, right)) = ray_hit {
        let position = if distance2(uv(ring[left]), hole_point) <= distance2(uv(ring[right]), hole_point)
        {
            left
        } else {
            right
        };
        return Ok(splice_hole(ring, hole, start, position));
    }
    Err(format!(
        "face {face}: cannot bridge a hole into the outer loop (no visible ring vertex)"
    ))
}

/// Splices `hole` into `ring` at `position`, starting the hole at `start`, with the
/// duplicated vertices that make the zero-width corridor.
fn splice_hole(ring: &[usize], hole: &[usize], start: usize, position: usize) -> Vec<usize> {
    let mut bridged = Vec::with_capacity(ring.len() + hole.len() + 2);
    bridged.extend_from_slice(&ring[..=position]);
    bridged.extend_from_slice(&hole[start..]);
    bridged.extend_from_slice(&hole[..start]);
    bridged.push(hole[start]);
    bridged.extend_from_slice(&ring[position..]);
    if bridged.len() <= ring.len() {
        // Not reachable for a hole with at least three vertices; keep the ring valid if a
        // malformed loop is ever passed in.
        return ring.to_vec();
    }
    bridged
}

/// How many ring vertices sit at `point` (bridge anchors are duplicated by a corridor,
/// so a position used by an earlier bridge occurs more than once).
fn position_count(
    ring: &[usize],
    point: Uv,
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
) -> usize {
    ring.iter()
        .filter(|&&index| same_position(uv(index), point, tolerance))
        .count()
}

/// True when the segment `a`–`b` can become a hole bridge.
fn segment_visible(
    a: usize,
    b: usize,
    ring: &[usize],
    all_holes: &[Vec<usize>],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
) -> bool {
    let start = uv(a);
    let end = uv(b);
    if distance2(start, end) <= tolerance * tolerance {
        return false;
    }
    // The bridge must stay strictly inside the face: interior samples may not touch a
    // ring edge or a hole boundary and may not fall inside a hole. A bridge that grazes
    // another loop (for example one that runs along the boundary between two aligned
    // holes) creates a degenerate slit that cannot be ear-clipped reliably.
    for step in 1..4 {
        let fraction = step as f64 / 4.0;
        let sample = Uv {
            x: start.x + (end.x - start.x) * fraction,
            y: start.y + (end.y - start.y) * fraction,
        };
        if point_near_boundary(sample, ring, uv, tolerance) {
            return false;
        }
        if !point_in_polygon(sample, ring, uv) {
            return false;
        }
        for hole in all_holes {
            if point_near_boundary(sample, hole, uv, tolerance) {
                return false;
            }
            if point_in_polygon(sample, hole, uv) {
                return false;
            }
        }
    }
    // The bridge must not cross any ring edge or hole edge.
    let obstacles = std::iter::once(ring).chain(all_holes.iter().map(Vec::as_slice));
    for polygon in obstacles {
        for position in 0..polygon.len() {
            let c = polygon[position];
            let d = polygon[(position + 1) % polygon.len()];
            if c == a || c == b || d == a || d == b {
                continue;
            }
            let c_point = uv(c);
            let d_point = uv(d);
            if same_position(c_point, start, tolerance)
                || same_position(c_point, end, tolerance)
                || same_position(d_point, start, tolerance)
                || same_position(d_point, end, tolerance)
            {
                continue;
            }
            if distance2(c_point, d_point) <= tolerance * tolerance {
                continue;
            }
            if segments_intersect(start, end, c_point, d_point, tolerance) {
                return false;
            }
        }
    }
    true
}

/// True when `point` is on (or within tolerance of) the polygon's boundary.
fn point_near_boundary(
    point: Uv,
    polygon: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
) -> bool {
    for position in 0..polygon.len() {
        let a = uv(polygon[position]);
        let b = uv(polygon[(position + 1) % polygon.len()]);
        if point_segment_distance2(point, a, b) <= tolerance * tolerance {
            return true;
        }
    }
    false
}

/// Even-odd point-in-polygon test on projected coordinates.
fn point_in_polygon(point: Uv, polygon: &[usize], uv: &impl Fn(usize) -> Uv) -> bool {
    let mut inside = false;
    for position in 0..polygon.len() {
        let a = uv(polygon[position]);
        let b = uv(polygon[(position + 1) % polygon.len()]);
        if (a.y > point.y) != (b.y > point.y) {
            let denominator = b.y - a.y;
            if denominator.abs() <= f64::MIN_POSITIVE {
                continue;
            }
            let x = a.x + (point.y - a.y) / denominator * (b.x - a.x);
            if point.x < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// Proper segment intersection (touching counts as intersecting).
fn segments_intersect(a: Uv, b: Uv, c: Uv, d: Uv, tolerance: f64) -> bool {
    let epsilon = tolerance * tolerance * 1.0e-3;
    let o1 = cross2(b.sub(a), c.sub(a));
    let o2 = cross2(b.sub(a), d.sub(a));
    let o3 = cross2(d.sub(c), a.sub(c));
    let o4 = cross2(d.sub(c), b.sub(c));
    if ((o1 > epsilon && o2 < -epsilon) || (o1 < -epsilon && o2 > epsilon))
        && ((o3 > epsilon && o4 < -epsilon) || (o3 < -epsilon && o4 > epsilon))
    {
        return true;
    }
    // Collinear/touching cases: any endpoint on the other segment blocks the bridge.
    on_segment(a, c, d, epsilon)
        || on_segment(b, c, d, epsilon)
        || on_segment(c, a, b, epsilon)
        || on_segment(d, a, b, epsilon)
}

/// Proper crossing test: segments that only touch at an endpoint or run collinearly do
/// not cross.
fn segments_cross(a: Uv, b: Uv, c: Uv, d: Uv, epsilon: f64) -> bool {
    let o1 = cross2(b.sub(a), c.sub(a));
    let o2 = cross2(b.sub(a), d.sub(a));
    let o3 = cross2(d.sub(c), a.sub(c));
    let o4 = cross2(d.sub(c), b.sub(c));
    ((o1 > epsilon && o2 < -epsilon) || (o1 < -epsilon && o2 > epsilon))
        && ((o3 > epsilon && o4 < -epsilon) || (o3 < -epsilon && o4 > epsilon))
}

fn on_segment(point: Uv, a: Uv, b: Uv, epsilon: f64) -> bool {
    let collinear = cross2(b.sub(a), point.sub(a)).abs() <= epsilon;
    collinear
        && point.x >= a.x.min(b.x) - epsilon.max(0.0).sqrt()
        && point.x <= a.x.max(b.x) + epsilon.max(0.0).sqrt()
        && point.y >= a.y.min(b.y) - epsilon.max(0.0).sqrt()
        && point.y <= a.y.max(b.y) + epsilon.max(0.0).sqrt()
}

/// Ear clipping of one simple ring (holes already bridged in).
///
/// Besides the local ear test, every cut is checked against the ring's signed area: a
/// valid ear removes exactly its own area from the remaining polygon. A cut that crosses
/// a zero-width bridge corridor breaks that invariant even though it can look locally
/// valid, so it is rejected. This is what keeps a bridged ring from inverting.
fn ear_clip(
    ring: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    area_epsilon: f64,
    triangles: &mut Vec<[usize; 3]>,
    face: usize,
) -> Result<(), String> {
    // Splitting a pathological ring must not recurse without bound on crafted input.
    let mut split_budget = 64usize;
    clip_ring(
        ring,
        uv,
        tolerance,
        area_epsilon,
        triangles,
        face,
        0,
        &mut split_budget,
    )
}

/// Recursive ear-clipping worker. When no single-vertex ear is available (a weakly
/// simple ring whose holes were bridged to a shared pinch point), it splits the ring
/// along any valid internal diagonal and clips the two halves independently.
fn clip_ring(
    ring: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    area_epsilon: f64,
    triangles: &mut Vec<[usize; 3]>,
    face: usize,
    depth: usize,
    split_budget: &mut usize,
) -> Result<(), String> {
    let triangle_start = triangles.len();
    let mut ring = ring.to_vec();
    if ring.len() > 3 && signed_area(&ring, uv) < 0.0 {
        ring.reverse();
    }
    let mut remaining_area = signed_area(&ring, uv);
    let mut guard = ring.len() + 1;
    while ring.len() > 3 {
        // What is left can be a zero-area sliver of a bridge corridor (the polygon has
        // been fully consumed). It needs no triangles; requiring a non-degenerate ear
        // here would reject an otherwise complete triangulation.
        if remaining_area.abs() <= area_epsilon * 4.0 {
            return Ok(());
        }
        if guard == 0 {
            return Err(format!("face {face}: ear clipping did not terminate"));
        }
        guard -= 1;
        let count = ring.len();
        let mut candidates: Vec<(usize, usize)> = (0..count)
            .filter_map(|index| {
                ear_blocked(&ring, index, uv, tolerance, area_epsilon)
                    .map(|blocked| (index, blocked))
            })
            .collect();
        // Prefer true ears (no blocking vertex); bridged rings can leave only
        // overlapping candidates, which are tried afterwards by increasing overlap.
        candidates.sort_by_key(|&(_, blocked)| blocked);
        // Strict diagonals are preferred: a diagonal that only touches a zero-width
        // bridge corridor can be geometrically ambiguous, so relaxed diagonals and ears
        // with blocking vertices are only used when nothing better exists.
        let mut chosen: Option<([usize; 3], Vec<usize>, f64)> = None;
        'search: for phase in 0..4 {
            let strict = phase % 2 == 0;
            let allow_blocked = phase >= 2;
            for &(index, blocked) in &candidates {
                if !allow_blocked && blocked != 0 {
                    continue;
                }
                if let Some((triangle, reduced, after)) = try_ear(
                    &ring,
                    index,
                    uv,
                    tolerance,
                    area_epsilon,
                    remaining_area,
                    strict,
                ) {
                    chosen = Some((triangle, reduced, after));
                    break 'search;
                }
            }
        }
        let Some((triangle, reduced, after)) = chosen else {
            if depth < 32 {
                for (start, end) in split_diagonals(&ring, uv, tolerance, area_epsilon) {
                    if *split_budget == 0 {
                        break;
                    }
                    *split_budget -= 1;
                    let attempt = triangles.len();
                    let first = ring[start..=end].to_vec();
                    let mut second = ring[end..].to_vec();
                    second.extend_from_slice(&ring[..=start]);
                    let mut clipped = true;
                    for part in [&first, &second] {
                        if part.len() < 3 || signed_area(part, uv).abs() <= area_epsilon {
                            continue;
                        }
                        if clip_ring(
                            part,
                            uv,
                            tolerance,
                            area_epsilon,
                            triangles,
                            face,
                            depth + 1,
                            split_budget,
                        )
                        .is_err()
                        {
                            clipped = false;
                            break;
                        }
                    }
                    if clipped {
                        return Ok(());
                    }
                    triangles.truncate(attempt);
                }
                triangles.truncate(triangle_start);
                return Err(format!(
                    "face {face}: cannot triangulate a weakly simple ring ({} vertices)",
                    ring.len()
                ));
            }
            triangles.truncate(triangle_start);
            return Err(format!(
                "face {face}: ear clipping stalled with {} vertices left",
                ring.len()
            ));
        };
        ring = reduced;
        triangles.push(triangle);
        remaining_area = after;
    }
    if ring.len() == 3 {
        triangles.push([ring[0], ring[1], ring[2]]);
        Ok(())
    } else {
        Err(format!("face {face}: ear clipping produced no triangles"))
    }
}

/// Attempts the ear at `index`: the diagonal must be clear, the ear must have positive
/// area, and removing it must conserve the ring's signed area exactly. Returns the
/// triangle, the reduced ring and the reduced ring's area.
fn try_ear(
    ring: &[usize],
    index: usize,
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    area_epsilon: f64,
    remaining_area: f64,
    strict: bool,
) -> Option<([usize; 3], Vec<usize>, f64)> {
    let count = ring.len();
    let previous = ring[(index + count - 1) % count];
    let current = ring[index];
    let next = ring[(index + 1) % count];
    if !diagonal_clear(previous, next, ring, uv, tolerance, strict) {
        return None;
    }
    let ear_area = 0.5 * cross2(uv(current).sub(uv(previous)), uv(next).sub(uv(previous)));
    if !(ear_area > area_epsilon) {
        return None;
    }
    let mut reduced = ring.to_vec();
    reduced.remove(index);
    let after = signed_area(&reduced, uv);
    let tolerance_area = remaining_area.abs() * 1.0e-9 + area_epsilon * 4.0;
    if (after - (remaining_area - ear_area)).abs() > tolerance_area {
        return None;
    }
    // A valid cut can never consume more area than the polygon has left: that would mean
    // the ear crossed the ring boundary and inverted it.
    if after < -tolerance_area || ear_area > remaining_area + tolerance_area {
        return None;
    }
    Some(([previous, current, next], reduced, after))
}

/// Number of ring vertices that block the ear at `index`, or `None` when the vertex is
/// not a convex corner and therefore can never be an ear.
fn ear_blocked(
    ring: &[usize],
    index: usize,
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    area_epsilon: f64,
) -> Option<usize> {
    let count = ring.len();
    let previous = ring[(index + count - 1) % count];
    let current = ring[index];
    let next = ring[(index + 1) % count];
    let a = uv(previous);
    let b = uv(current);
    let c = uv(next);
    if distance2(a, b) <= tolerance * tolerance || distance2(b, c) <= tolerance * tolerance {
        return None;
    }
    if cross2(b.sub(a), c.sub(b)) <= area_epsilon {
        return None;
    }
    let mut blocked = 0usize;
    for (other_index, &other) in ring.iter().enumerate() {
        if other_index == index
            || other_index == (index + count - 1) % count
            || other_index == (index + 1) % count
        {
            continue;
        }
        let point = uv(other);
        if same_position(point, a, tolerance)
            || same_position(point, b, tolerance)
            || same_position(point, c, tolerance)
        {
            continue;
        }
        // Only reflex vertices can invalidate a convex ear. Points that lie exactly on an
        // ear edge (collinear sampled boundaries, bridge duplicates) are harmless once
        // they are not reflex, and treating them as blockers is what makes ear clipping
        // stall on dense hole patterns.
        let other_previous = ring[(other_index + count - 1) % count];
        let other_next = ring[(other_index + 1) % count];
        let other_turn = cross2(
            uv(other).sub(uv(other_previous)),
            uv(other_next).sub(uv(other)),
        );
        if other_turn > area_epsilon {
            continue;
        }
        if point_in_triangle(point, a, b, c, area_epsilon) {
            blocked += 1;
        }
    }
    Some(blocked)
}

/// Finds an internal diagonal between two non-adjacent ring vertices. Used as a
/// fallback when a weakly simple ring (bridge corridors sharing a pinch point) has no
/// single-vertex ear.
fn split_diagonals(
    ring: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    area_epsilon: f64,
) -> Vec<(usize, usize)> {
    let count = ring.len();
    let mut splits = Vec::new();
    for start in 0..count {
        for end in (start + 2)..count {
            if start == 0 && end == count - 1 {
                continue;
            }
            if !diagonal_valid(start, end, ring, uv, tolerance) {
                continue;
            }
            // Both halves must be ordinary counter-clockwise regions. A split whose
            // half winds backwards crosses the boundary rather than dividing the face,
            // and triangulating it would double-count area.
            let first_area = signed_area(&ring[start..=end], uv);
            let mut second = ring[end..].to_vec();
            second.extend_from_slice(&ring[..=start]);
            if first_area > area_epsilon && signed_area(&second, uv) > area_epsilon {
                splits.push((start, end));
                if splits.len() >= 64 {
                    return splits;
                }
            }
        }
    }
    splits
}

/// True when the diagonal between ring positions `start` and `end` lies inside the ring:
/// its midpoint is inside and it does not properly cross any boundary edge.
fn diagonal_valid(
    start: usize,
    end: usize,
    ring: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
) -> bool {
    let a = uv(ring[start]);
    let b = uv(ring[end]);
    if distance2(a, b) <= tolerance * tolerance {
        return false;
    }
    let midpoint = Uv { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
    if !point_in_polygon(midpoint, ring, uv) {
        return false;
    }
    for position in 0..ring.len() {
        let c = ring[position];
        let d = ring[(position + 1) % ring.len()];
        if c == ring[start] || c == ring[end] || d == ring[start] || d == ring[end] {
            continue;
        }
        let c_point = uv(c);
        let d_point = uv(d);
        if same_position(c_point, a, tolerance)
            || same_position(c_point, b, tolerance)
            || same_position(d_point, a, tolerance)
            || same_position(d_point, b, tolerance)
        {
            continue;
        }
        if distance2(c_point, d_point) <= tolerance * tolerance {
            continue;
        }
        if segments_cross(a, b, c_point, d_point, tolerance * tolerance * 1.0e-3) {
            return false;
        }
    }
    true
}

/// True when the segment `a`–`b` stays inside the ring: it neither crosses a ring edge
/// nor passes through another ring vertex. Such a diagonal is required for a valid ear
/// cut of a bridged polygon, whose zero-width corridors make naive ear tests cut across
/// the corridor and invert the remaining ring.
fn diagonal_clear(
    a: usize,
    b: usize,
    ring: &[usize],
    uv: &impl Fn(usize) -> Uv,
    tolerance: f64,
    strict: bool,
) -> bool {
    let start = uv(a);
    let end = uv(b);
    if distance2(start, end) <= tolerance * tolerance {
        return false;
    }
    // A valid internal diagonal lies inside the face. Cross-checking against the boundary
    // alone is not enough: a diagonal can pass through the exterior of a concave ring
    // without crossing an edge.
    if strict && !point_in_polygon(
        Uv { x: (start.x + end.x) * 0.5, y: (start.y + end.y) * 0.5 },
        ring,
        uv,
    ) {
        return false;
    }
    for position in 0..ring.len() {
        let c = ring[position];
        let d = ring[(position + 1) % ring.len()];
        if c == a || c == b || d == a || d == b {
            continue;
        }
        let c_point = uv(c);
        let d_point = uv(d);
        // Both copies of a bridge vertex sit at the same position; they are endpoints of
        // the two corridors, not an obstruction.
        if same_position(c_point, start, tolerance)
            || same_position(c_point, end, tolerance)
            || same_position(d_point, start, tolerance)
            || same_position(d_point, end, tolerance)
        {
            continue;
        }
        if distance2(c_point, d_point) <= tolerance * tolerance {
            continue;
        }
        if strict {
            if point_on_segment(c_point, start, end, tolerance) {
                return false;
            }
            if segments_intersect(start, end, c_point, d_point, tolerance) {
                return false;
            }
        } else if segments_cross(
            start,
            end,
            c_point,
            d_point,
            tolerance * tolerance * 1.0e-3,
        ) {
            // Touching a zero-width bridge corridor at a pinch point does not invalidate
            // the cut (the corridor has no area); only a proper crossing does. The area
            // check in `face_triangles` remains the safety net against overlaps.
            return false;
        }
    }
    true
}

/// Point inside or on a counter-clockwise triangle.
fn point_in_triangle(point: Uv, a: Uv, b: Uv, c: Uv, epsilon: f64) -> bool {
    cross2(b.sub(a), point.sub(a)) >= -epsilon
        && cross2(c.sub(b), point.sub(b)) >= -epsilon
        && cross2(a.sub(c), point.sub(c)) >= -epsilon
}
