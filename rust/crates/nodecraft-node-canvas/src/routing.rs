//! Continuous cable-routing.
//!
//! Cable length and bottom-to-top rise are normalised 0..1 factors rather than
//! hard direction booleans. Every routing consumer (GPU tessellation, hit
//! testing, CPU fallback) uses this one function.
//!
//! A normal cable falls from the source output port to the target input port and
//! is one cubic with vertical tangents at both ports. When the target sits above
//! the source the cable has to reverse direction: it leaves the port straight
//! down, wraps around the outside of both cards and enters the target port
//! straight down from above. Two joined cubics describe that wrap. The join is
//! kept tangent-continuous, and the wrap grows out of the straight chord as the
//! rise increases so dragging a node across the source never snaps the cable.

use egui::{Pos2, Rect, Vec2};

/// One cubic segment of a routed cable, in graph/screen space.
#[derive(Clone, Copy, Debug)]
pub struct WireCubic {
    pub p0: Pos2,
    pub p1: Pos2,
    pub p2: Pos2,
    pub p3: Pos2,
}

impl WireCubic {
    fn point(&self, t: f32) -> Pos2 {
        let u = 1.0 - t;
        let a = u * u * u;
        let b = 3.0 * u * u * t;
        let c = 3.0 * u * t * t;
        let d = t * t * t;
        Pos2::new(
            a * self.p0.x + b * self.p1.x + c * self.p2.x + d * self.p3.x,
            a * self.p0.y + b * self.p1.y + c * self.p2.y + d * self.p3.y,
        )
    }

    fn tangent(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        (self.p1 - self.p0) * (3.0 * u * u)
            + (self.p2 - self.p1) * (6.0 * u * t)
            + (self.p3 - self.p2) * (3.0 * t * t)
    }

    fn straight(a: Pos2, b: Pos2) -> Self {
        let step = (b - a) / 3.0;
        Self { p0: a, p1: a + step, p2: a + step * 2.0, p3: b }
    }

    fn blended(from: &Self, to: &Self, blend: f32) -> Self {
        let mix = |a: Pos2, b: Pos2| a + (b - a) * blend;
        Self {
            p0: mix(from.p0, to.p0),
            p1: mix(from.p1, to.p1),
            p2: mix(from.p2, to.p2),
            p3: mix(from.p3, to.p3),
        }
    }
}

/// A routed cable: one cubic for a normal cable, or eight joined cubics for the
/// upward wrap. The wrap alternates exact straight segments with cubic
/// approximations of circular quarter arcs, so every corner has constant
/// curvature like a subway-map bend.
#[derive(Clone, Copy, Debug)]
pub struct WireRouting {
    pub segments: [WireCubic; 8],
    pub count: usize,
    pub length_factor: f32,
    pub rise_factor: f32,
}

impl WireRouting {
    /// Sample the routed path at normalised parameter `t` (uniform across the
    /// segments) and return the point plus its tangent.
    pub fn sample(&self, t: f32) -> (Pos2, Vec2) {
        let t = t.clamp(0.0, 1.0);
        let count = self.count.max(1);
        let scaled = t * count as f32;
        let index = (scaled.floor() as usize).min(count - 1);
        let local = scaled - index as f32;
        let segment = &self.segments[index];
        (segment.point(local), segment.tangent(local))
    }

    /// Polyline approximation of the whole routed path.
    pub fn polyline(&self, samples: usize) -> Vec<Pos2> {
        let samples = samples.max(1);
        (0..=samples)
            .map(|index| self.sample(index as f32 / samples as f32).0)
            .collect()
    }

    /// Conservative bounding box of the routed path (the cubic control hull,
    /// which always contains the curve). Allocation-free so hit testing does not
    /// build a polyline per cable.
    pub fn control_bounds(&self) -> Rect {
        let count = self.count.max(1);
        let mut rect: Option<Rect> = None;
        for cubic in &self.segments[..count] {
            for point in [cubic.p0, cubic.p1, cubic.p2, cubic.p3] {
                let point_rect = Rect::from_min_max(point, point);
                rect = Some(match rect {
                    Some(rect) => rect.union(point_rect),
                    None => point_rect,
                });
            }
        }
        rect.unwrap_or_else(|| Rect::from_min_max(Pos2::ZERO, Pos2::ZERO))
    }

    /// Distance from `point` to the routed path without allocating. Straight
    /// segments are exact under sampling; arc segments stay within a fraction of
    /// a pixel of the chord for the few samples used per corner.
    pub fn distance_to(&self, point: Pos2, samples_per_segment: usize) -> f32 {
        let count = self.count.max(1);
        let samples = samples_per_segment.max(2);
        let mut best = f32::INFINITY;
        for cubic in &self.segments[..count] {
            let mut previous = cubic.p0;
            for step in 1..=samples {
                let current = cubic.point(step as f32 / samples as f32);
                best = best.min(distance_to_segment(point, previous, current));
                previous = current;
            }
        }
        best
    }
}

fn distance_to_segment(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let length_sq = ab.length_sq();
    if length_sq <= f32::EPSILON {
        return (point - a).length();
    }
    let t = ((point - a).dot(ab) / length_sq).clamp(0.0, 1.0);
    (point - (a + ab * t)).length()
}

pub const MIN_HANDLE: f32 = 14.0;
pub const MAX_HANDLE: f32 = 140.0;
pub const MAX_UP_HANDLE: f32 = 220.0;
pub const LENGTH_SCALE: f32 = 360.0;
pub const UP_BEND_GAIN: f32 = 1.4;
/// Minimum sideways room for the upward wrap, so the loop stays visible even
/// when the two ports are almost aligned.
pub const MIN_LOOP_WIDTH: f32 = 26.0;
pub const MIN_CORNER: f32 = 10.0;
/// Lateral bow used by the bezier-style route for near-aligned upward ports.
pub const MIN_BOW: f32 = 36.0;
pub const MAX_CORNER: f32 = 44.0;
/// Clearance kept between a wrapped cable lane and the connected cards. This
/// includes the overhanging view button on the card's right edge.
pub const NODE_CLEARANCE: f32 = 46.0;
/// Minimum straight length kept between a stair corner and the neighbouring
/// short run.
pub const STAIRCASE_MIN_RUN: f32 = 8.0;
/// Structured corners below this radius snap to a single smooth spline instead.
/// The snap is deterministic from the port geometry, so the CPU hit-test path
/// and the GPU control points always agree.
pub const SMOOTH_DOWNWARD_RADIUS: f32 = 24.0;

/// Control distance for a circular quarter-arc approximation. This is the one
/// arc constant shared by the CPU route builders; the GPU evaluates the cubics
/// it produces, so both see exactly the same corners.
pub fn arc_handle(radius: f32) -> f32 {
    radius * 0.552_284_75
}

/// Cable routing style, chosen by an app setting (not the canvas theme).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CableRouting {
    /// Smooth single-cubic bezier curves (the original cable style).
    Bezier,
    /// Horizontal/vertical runs with rounded corners.
    #[default]
    J,
}

impl CableRouting {
    pub fn setting_id(self) -> &'static str {
        match self {
            Self::Bezier => "bezier-routing",
            Self::J => "j-routing",
        }
    }

    pub fn from_setting(value: &str) -> Self {
        match value {
            "bezier-routing" => Self::Bezier,
            _ => Self::J,
        }
    }
}

/// Original smooth bezier route: one cubic with vertical port tangents.
fn bezier_route(x0: f32, y0: f32, x1: f32, y1: f32, scale: f32) -> WireRouting {
    let scale = scale.max(0.0001);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let distance = dx.hypot(dy).max(0.0001);
    let length_factor = (distance / (LENGTH_SCALE * scale)).clamp(0.0, 1.0);
    let base = MIN_HANDLE + (MAX_HANDLE - MIN_HANDLE) * length_factor;
    let a = Pos2::new(x0, y0);
    let b = Pos2::new(x1, y1);
    let (p1, p2, rise_factor) = if dy >= 0.0 {
        let handle = (base * scale)
            .clamp(MIN_HANDLE * scale, MAX_HANDLE * scale)
            .min(dy * 0.5)
            .max(0.0);
        (Pos2::new(x0, y0 + handle), Pos2::new(x1, y1 - handle), 0.0)
    } else {
        let rise = -dy;
        let rise_factor = (rise / distance).clamp(0.0, 1.0);
        let handle = ((base * (1.0 + UP_BEND_GAIN * rise_factor)) * scale)
            .clamp(MIN_HANDLE * scale, MAX_UP_HANDLE * scale)
            .min(rise * 0.75)
            .max(0.0);
        let bow_blend = (rise / (2.0 * MIN_HANDLE * scale)).clamp(0.0, 1.0);
        let bow = (MIN_BOW * scale - dx.abs()).max(0.0) * 0.5 * bow_blend;
        (Pos2::new(x0 + bow, y0 + handle), Pos2::new(x1 + bow, y1 - handle), rise_factor)
    };
    let mut segments = [WireCubic::straight(b, b); 8];
    segments[0] = WireCubic { p0: a, p1, p2, p3: b };
    WireRouting { segments, count: 1, length_factor, rise_factor }
}

/// Route a cable without card rects, using the selected style.
pub fn route_for(style: CableRouting, x0: f32, y0: f32, x1: f32, y1: f32, scale: f32) -> WireRouting {
    match style {
        CableRouting::Bezier => bezier_route(x0, y0, x1, y1, scale),
        CableRouting::J => route(x0, y0, x1, y1, scale),
    }
}

/// Route a cable without explicit card clearance. Consumers that know the card
/// rects should call [`route_with_lane`] with the right card edge so the wrap
/// clears the cards instead of crossing them.
pub fn route(x0: f32, y0: f32, x1: f32, y1: f32, scale: f32) -> WireRouting {
    route_with_lane(x0, y0, x1, y1, scale, None)
}

/// `lane_x` is the x coordinate the upward wrap must route through (normally
/// the right edge of the wider card plus a margin). It is ignored for normal
/// downward cables.
fn smooth_downward(
    a: Pos2,
    b: Pos2,
    _x0: f32,
    _y0: f32,
    _x1: f32,
    _y1: f32,
    dy: f32,
    length_factor: f32,
) -> WireRouting {
    let handle = (dy * 0.5).max(0.0);
    let mut segments = [WireCubic::straight(b, b); 8];
    segments[0] = WireCubic {
        p0: a,
        p1: Pos2::new(a.x, a.y + handle),
        p2: Pos2::new(b.x, b.y - handle),
        p3: b,
    };
    WireRouting { segments, count: 1, length_factor, rise_factor: 0.0 }
}

pub fn route_with_lane(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    scale: f32,
    lane_x: Option<f32>,
) -> WireRouting {
    let scale = scale.max(0.0001);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let distance = dx.hypot(dy).max(0.0001);
    let length_factor = (distance / (LENGTH_SCALE * scale)).clamp(0.0, 1.0);
    let a = Pos2::new(x0, y0);
    let b = Pos2::new(x1, y1);

    if dy >= 0.0 {
        // Downward cable: same subway-map vocabulary as the upward wrap. An
        // aligned drop is a straight vertical line; otherwise the cable drops,
        // rounds a corner, runs sideways, rounds back down and drops into the
        // target. Small drops blend the corners away so crossing the source
        // stays continuous.
        let rise_factor = 0.0;
        let side = if dx >= 0.0 { 1.0 } else { -1.0 };
        let horizontal = dx.abs();
        if horizontal < 0.5 * scale {
            let handle = (dy * 0.5).max(0.0);
            return WireRouting {
                segments: [
                    WireCubic {
                        p0: a,
                        p1: Pos2::new(x0, y0 + handle),
                        p2: Pos2::new(x1, y1 - handle),
                        p3: b,
                    },
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                    WireCubic::straight(b, b),
                ],
                count: 1,
                length_factor,
                rise_factor,
            };
        }

        let min_stub = 6.0 * scale;
        // Leave a short straight run between the two corners instead of
        // collapsing it to zero (a degenerate segment pinches the ribbon).
        let radius = (horizontal * 0.45)
            .min(MAX_CORNER * scale)
            .min(((dy - 2.0 * min_stub) * 0.5).max(0.0));
        if radius < SMOOTH_DOWNWARD_RADIUS * scale {
            // Too sharp for a visible subway corner; a single smooth S is
            // cleaner than a tiny jog, and it is still a deterministic cubic
            // that the CPU and GPU can both evaluate.
            return smooth_downward(a, b, x0, y0, x1, y1, dy, length_factor);
        }
        let stub = ((dy - 2.0 * radius) * 0.5).max(0.0);
        let y_mid = y0 + stub + radius;
        let k = arc_handle(radius);
        let down = Vec2::new(0.0, 1.0);
        let sideways = Vec2::new(side, 0.0);
        let arc = |start: Pos2, end: Pos2, start_tangent: Vec2, end_tangent: Vec2| {
            WireCubic {
                p0: start,
                p1: start + start_tangent * k,
                p2: end - end_tangent * k,
                p3: end,
            }
        };
        let corner1_start = Pos2::new(x0, y0 + stub);
        let corner1_end = Pos2::new(x0 + side * radius, y_mid);
        let corner2_start = Pos2::new(x1 - side * radius, y_mid);
        let corner2_end = Pos2::new(x1, y_mid + radius);
        let target = [
            WireCubic::straight(a, corner1_start),
            arc(corner1_start, corner1_end, down, sideways),
            WireCubic::straight(corner1_end, corner2_start),
            arc(corner2_start, corner2_end, sideways, down),
            WireCubic::straight(corner2_end, b),
        ];

        // The structured drop/corner/run/corner/drop path is always used for a
        // real drop. Only a near-zero drop fades into the straight chord so
        // dragging a node across the source stays continuous.
        let blend = (dy / (2.0 * MIN_HANDLE * scale)).clamp(0.0, 1.0);
        let chord = b - a;
        let mut segments = [WireCubic::straight(a, b); 8];
        for (index, slot) in segments.iter_mut().take(target.len()).enumerate() {
            let start = a + chord * (index as f32 / target.len() as f32);
            let end = a + chord * ((index + 1) as f32 / target.len() as f32);
            *slot = WireCubic::blended(
                &WireCubic::straight(start, end),
                &target[index],
                blend,
            );
        }
        for index in 1..target.len() {
            let previous_tangent = segments[index - 1].p3 - segments[index - 1].p2;
            let length = (segments[index].p1 - segments[index].p0).length();
            if previous_tangent.length() > 1.0e-6 && length > 1.0e-6 {
                segments[index].p1 =
                    segments[index].p0 + previous_tangent.normalized() * length;
            }
        }
        return WireRouting {
            segments,
            count: target.len(),
            length_factor,
            rise_factor,
        };
    }

    // Upward cable: wrap around the outside of the cards. The path alternates
    // straight runs with quarter-circle corners, so it reads like a subway-map
    // route rather than a single sweeping curve.
    let rise = y0 - y1;
    let rise_factor = (rise / distance).clamp(0.0, 1.0);
    let handle = ((MIN_HANDLE + (MAX_HANDLE - MIN_HANDLE) * length_factor)
        * (1.0 + UP_BEND_GAIN * rise_factor)
        * scale)
        .clamp(MIN_HANDLE * scale, MAX_UP_HANDLE * scale);
    let widest = x0.max(x1);
    let narrowest = x0.min(x1);
    let derived_lane = widest + (handle * 0.9).max(MIN_LOOP_WIDTH * scale);
    let lane = match lane_x {
        Some(lane) if lane < (x0 + x1) * 0.5 => {
            lane.min(narrowest - MIN_LOOP_WIDTH * scale)
        }
        Some(lane) => lane.max(widest + MIN_LOOP_WIDTH * scale),
        None => derived_lane,
    };
    let side = if lane < (x0 + x1) * 0.5 { -1.0 } else { 1.0 };
    let available = if side > 0.0 {
        (lane - widest).max(0.001)
    } else {
        (narrowest - lane).max(0.001)
    };
    let radius = (available * 0.45)
        .clamp(MIN_CORNER * scale, MAX_CORNER * scale)
        .min(available * 0.49);
    let stub = (MIN_HANDLE * scale * 0.6).max(6.0 * scale);
    let y_bottom = y0 + stub + radius;
    let y_top = y1 - radius;
    let k = arc_handle(radius);

    let corner1_start = Pos2::new(x0, y0 + stub);
    let corner1_end = Pos2::new(x0 + side * radius, y_bottom);
    let corner2_start = Pos2::new(lane - side * radius, y_bottom);
    let corner2_end = Pos2::new(lane, y_bottom - radius);
    let corner3_start = Pos2::new(lane, y_top + radius);
    let corner3_end = Pos2::new(lane - side * radius, y_top);
    let corner4_start = Pos2::new(x1 + side * radius, y_top);

    // Quarter-circle control points: a cubic with controls `k` along the end
    // tangents approximates a circular arc to within ~0.03% of the radius.
    let arc = |start: Pos2, end: Pos2, start_tangent: Vec2, end_tangent: Vec2| {
        WireCubic {
            p0: start,
            p1: start + start_tangent * k,
            p2: end - end_tangent * k,
            p3: end,
        }
    };
    let down = Vec2::new(0.0, 1.0);
    let up = Vec2::new(0.0, -1.0);
    let outward = Vec2::new(side, 0.0);
    let inward = Vec2::new(-side, 0.0);

    let loop_segments = [
        WireCubic::straight(a, corner1_start),
        arc(corner1_start, corner1_end, down, outward),
        WireCubic::straight(corner1_end, corner2_start),
        arc(corner2_start, corner2_end, outward, up),
        WireCubic::straight(corner2_end, corner3_start),
        arc(corner3_start, corner3_end, up, inward),
        WireCubic::straight(corner3_end, corner4_start),
        arc(corner4_start, b, inward, down),
    ];

    // Collapse the wrap into the straight chord while the rise is still tiny;
    // this keeps dragging a node past the source continuous.
    let blend = (rise / (2.0 * MIN_HANDLE * scale)).clamp(0.0, 1.0);
    let chord = b - a;
    let mut segments = [WireCubic::straight(a, b); 8];
    for (index, cubic) in segments.iter_mut().enumerate() {
        let start = a + chord * (index as f32 / 8.0);
        let end = a + chord * ((index + 1) as f32 / 8.0);
        *cubic = WireCubic::blended(
            &WireCubic::straight(start, end),
            &loop_segments[index],
            blend,
        );
    }
    // Keep every join tangent-continuous at all blend values while preserving
    // the control distance the corner needs.
    for index in 1..segments.len() {
        let previous_tangent = segments[index - 1].p3 - segments[index - 1].p2;
        let length = (segments[index].p1 - segments[index].p0).length();
        if previous_tangent.length() > 1.0e-6 && length > 1.0e-6 {
            segments[index].p1 =
                segments[index].p0 + previous_tangent.normalized() * length;
        }
    }

    WireRouting {
        segments,
        count: 8,
        length_factor,
        rise_factor,
    }
}

/// Build a stair in mirrored x-space where `x0 < lane < x1`. Callers reflect
/// the result when the cable crosses leftwards.
fn staircase_mirrored(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    lane: f32,
    scale: f32,
    length_factor: f32,
) -> Option<WireRouting> {
    let left_room = lane - x0;
    let right_room = x1 - lane;
    let min_run = STAIRCASE_MIN_RUN * scale;
    if left_room <= 2.0 * MIN_CORNER * scale + min_run
        || right_room <= 2.0 * MIN_CORNER * scale + min_run
    {
        return None;
    }
    let radius = (MAX_CORNER * scale)
        .min((left_room - min_run) * 0.5)
        .min((right_room - min_run) * 0.5)
        .max(0.0);
    if radius < 0.5 * scale {
        return None;
    }
    let k = arc_handle(radius);
    let stub = (MIN_HANDLE * scale * 0.5).max(4.0 * scale);
    let y_bottom = y0 + stub + radius;
    let y_top = y1 - radius;
    let a = Pos2::new(x0, y0);
    let b = Pos2::new(x1, y1);
    let down = Vec2::new(0.0, 1.0);
    let up = Vec2::new(0.0, -1.0);
    let outward = Vec2::new(1.0, 0.0);
    let arc = |start: Pos2, end: Pos2, start_tangent: Vec2, end_tangent: Vec2| {
        WireCubic {
            p0: start,
            p1: start + start_tangent * k,
            p2: end - end_tangent * k,
            p3: end,
        }
    };
    let corner1_start = Pos2::new(x0, y0 + stub);
    let corner1_end = Pos2::new(x0 + radius, y_bottom);
    let corner2_start = Pos2::new(lane - radius, y_bottom);
    let corner2_end = Pos2::new(lane, y_bottom - radius);
    let corner3_start = Pos2::new(lane, y_top + radius);
    let corner3_end = Pos2::new(lane + radius, y_top);
    let corner4_start = Pos2::new(x1 - radius, y_top);
    let segments = [
        WireCubic::straight(a, corner1_start),
        arc(corner1_start, corner1_end, down, outward),
        WireCubic::straight(corner1_end, corner2_start),
        arc(corner2_start, corner2_end, outward, up),
        WireCubic::straight(corner2_end, corner3_start),
        arc(corner3_start, corner3_end, up, outward),
        WireCubic::straight(corner3_end, corner4_start),
        arc(corner4_start, b, outward, down),
    ];
    Some(WireRouting { segments, count: 8, length_factor, rise_factor: 0.0 })
}

fn reflect_horizontally(routing: &mut WireRouting, side: f32) {
    if side >= 0.0 {
        return;
    }
    for cubic in routing.segments.iter_mut() {
        cubic.p0.x = -cubic.p0.x;
        cubic.p1.x = -cubic.p1.x;
        cubic.p2.x = -cubic.p2.x;
        cubic.p3.x = -cubic.p3.x;
    }
}

/// Node-size-aware route used by every canvas consumer.
///
/// `source` and `target` are the full card rects in the same coordinate space
/// as the ports. When the input sits higher than the output the cable wraps
/// around the card side the target is on, using the full card extents to place
/// its lane; otherwise it uses the structured drop/run/drop path between the
/// ports. The output is the same cubic list the CPU hit tests and the GPU
/// instance consume.
pub fn route_nodes(
    source: Rect,
    target: Rect,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    scale: f32,
) -> WireRouting {
    if y1 < y0 {
        // Prefer turning the input from the source-facing side: the vertical
        // bend sits just outside the target card's near edge plus padding, so
        // the cable approaches the port from the left (or right) instead of
        // winding around the whole card.
        let side = if target.center().x >= source.center().x { 1.0 } else { -1.0 };
        let lane = if side > 0.0 {
            target.min.x - NODE_CLEARANCE * scale
        } else {
            target.max.x + NODE_CLEARANCE * scale
        };
        let dx = x1 - x0;
        let dy = y1 - y0;
        let distance = dx.hypot(dy).max(1.0e-3);
        let length_factor = (distance / (LENGTH_SCALE * scale)).clamp(0.0, 1.0);
        if let Some(mut staircase) = staircase_mirrored(
            side * x0,
            y0,
            side * x1,
            y1,
            side * lane,
            scale,
            length_factor,
        ) {
            reflect_horizontally(&mut staircase, side);
            return staircase;
        }
        // The cards overlap the source-facing edge: fall back to a compact
        // loop outside the ports.
        let lane = if side > 0.0 {
            x0.max(x1) + MIN_LOOP_WIDTH * scale
        } else {
            x0.min(x1) - MIN_LOOP_WIDTH * scale
        };
        route_with_lane(x0, y0, x1, y1, scale, Some(lane))
    } else {
        route_with_lane(x0, y0, x1, y1, scale, None)
    }
}

/// Node-aware route using the selected style.
pub fn route_nodes_for(
    style: CableRouting,
    source: Rect,
    target: Rect,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    scale: f32,
) -> WireRouting {
    match style {
        CableRouting::Bezier => bezier_route(x0, y0, x1, y1, scale),
        CableRouting::J => route_nodes(source, target, x0, y0, x1, y1, scale),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speed_minimum(routing: &WireRouting) -> f32 {
        let mut minimum = f32::INFINITY;
        for step in 1..2000 {
            let t = step as f32 / 2000.0;
            minimum = minimum.min(routing.sample(t).1.length());
        }
        minimum
    }

    #[test]
    fn routing_factors_are_normalized_and_continuous() {
        let down = route(0.0, 0.0, 100.0, 300.0, 1.0);
        let up = route(0.0, 300.0, 100.0, 0.0, 1.0);
        for value in [down.length_factor, down.rise_factor, up.length_factor, up.rise_factor] {
            assert!((0.0..=1.0).contains(&value), "factor out of range: {value}");
        }
        assert_eq!(down.count, 5);
        assert_eq!(up.count, 8);
        assert!(up.rise_factor > down.rise_factor);
    }

    #[test]
    fn downward_cables_enter_and_leave_vertically_without_cusping() {
        // Aligned ports and a short drop used to retrace the same vertical line.
        for (x1, y1, expected) in [
            (0.0, 20.0, 1),
            (0.0, 300.0, 1),
            (120.0, 40.0, 1),
            (200.0, 200.0, 5),
        ] {
            let routing = route(0.0, 0.0, x1, y1, 1.0);
            assert_eq!(routing.count, expected, "segment count at {x1},{y1}");
            let first = &routing.segments[0];
            let last = &routing.segments[routing.count - 1];
            assert!((first.p1.x - first.p0.x).abs() < 1.0e-4, "start tangent must be vertical");
            assert!((last.p3.x - last.p2.x).abs() < 1.0e-4, "end tangent must be vertical");
            assert!(speed_minimum(&routing) > 0.01, "cusp at {x1},{y1}");
        }
    }

    #[test]
    fn downward_cables_round_their_corners() {
        let routing = route(0.0, 0.0, 200.0, 200.0, 1.0);
        assert_eq!(routing.count, 5);
        let mut has_horizontal = false;
        let mut has_vertical = false;
        for step in 0..=400 {
            let tangent = routing.sample(step as f32 / 400.0).1;
            if tangent.x.abs() > tangent.y.abs() * 8.0 {
                has_horizontal = true;
            }
            if tangent.y.abs() > tangent.x.abs() * 8.0 {
                has_vertical = true;
            }
        }
        assert!(has_horizontal && has_vertical);
        assert!(routing.sample(0.0).1.x.abs() < 1.0e-4);
        assert!(routing.sample(1.0).1.x.abs() < 1.0e-4);
    }

    #[test]
    fn sharp_downward_corners_snap_to_a_smooth_spline() {
        // A small offset cannot fit a visible quarter corner, so the routing
        // deterministically snaps to one smooth cubic instead of a tiny jog.
        let routing = route(0.0, 0.0, 20.0, 70.0, 1.0);
        assert_eq!(routing.count, 1);
        assert!(routing.sample(0.0).1.x.abs() < 1.0e-4);
        assert!(routing.sample(1.0).1.x.abs() < 1.0e-4);
        assert!(speed_minimum(&routing) > 0.5);
        // Roomy corners still use the structured path.
        assert_eq!(route(0.0, 0.0, 200.0, 200.0, 1.0).count, 5);
    }

    #[test]
    fn cramped_upward_routes_stay_orthogonal() {
        let routing = route_with_lane(0.0, 0.0, 0.0, -100.0, 1.0, Some(30.0));
        assert_eq!(routing.count, 8);
        assert!(speed_minimum(&routing) > 0.5);
    }

    #[test]
    fn rightward_upward_routes_extend_to_the_card_edge() {
        let source = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(152.0, 104.0));
        let target = Rect::from_min_size(Pos2::new(600.0, -100.0), Vec2::new(152.0, 104.0));
        let x0 = source.max.x * 0.5;
        let y0 = source.max.y;
        let x1 = target.min.x + 20.0;
        let y1 = target.min.y;
        let routing = route_nodes(source, target, x0, y0, x1, y1, 1.0);
        assert_eq!(routing.count, 8);
        let vertical = |cubic: &WireCubic| (cubic.p1.x - cubic.p0.x).abs() < 1.0e-3;
        let horizontal = |cubic: &WireCubic| (cubic.p1.y - cubic.p0.y).abs() < 1.0e-3;
        assert!(vertical(&routing.segments[0]));
        assert!(horizontal(&routing.segments[2]));
        assert!(vertical(&routing.segments[4]));
        assert!(horizontal(&routing.segments[6]));
        // The vertical bend sits just outside the target card's near (left)
        // edge, so the cable turns the input from the left instead of winding
        // around the card.
        let bend = routing.segments[4].p0.x;
        assert!((bend - (target.min.x - NODE_CLEARANCE)).abs() < 1.0e-3);
        assert!(bend < x1, "bend must be before the target port");
        let max_x = routing
            .polyline(64)
            .into_iter()
            .fold(f32::NEG_INFINITY, |maximum, point| maximum.max(point.x));
        assert!(max_x <= target.max.x + 1.0, "must not wind past the card");
        assert!(speed_minimum(&routing) > 0.5);
    }

    #[test]
    fn leftward_upward_routes_extend_to_the_left_card_edge() {
        let source = Rect::from_min_size(Pos2::new(600.0, 0.0), Vec2::new(152.0, 104.0));
        let target = Rect::from_min_size(Pos2::new(0.0, -100.0), Vec2::new(152.0, 104.0));
        let routing = route_nodes(
            source,
            target,
            source.min.x + source.width() * 0.5,
            source.max.y,
            target.max.x - 20.0,
            target.min.y,
            1.0,
        );
        assert_eq!(routing.count, 8);
        let bend = routing.segments[4].p0.x;
        assert!((bend - (target.max.x + NODE_CLEARANCE)).abs() < 1.0e-3);
        assert!(bend > target.max.x, "bend must be outside the target card");
    }

    #[test]
    fn aligned_upward_targets_fall_back_to_the_compact_loop() {
        let source = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(152.0, 104.0));
        let target = Rect::from_min_size(Pos2::new(0.0, -220.0), Vec2::new(152.0, 104.0));
        let routing = route_nodes(
            source,
            target,
            source.max.x * 0.5,
            source.max.y,
            target.min.x + 20.0,
            target.min.y,
            1.0,
        );
        assert_eq!(routing.count, 8);
        assert!(speed_minimum(&routing) > 0.5);
    }

    #[test]
    fn wrap_lane_clears_the_connected_card_sizes() {
        let source = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(152.0, 104.0));
        let target = Rect::from_min_size(Pos2::new(300.0, -200.0), Vec2::new(320.0, 104.0));
        let lane = source.max.x.max(target.max.x) + NODE_CLEARANCE;
        let routing = route_with_lane(
            source.max.x * 0.5,
            source.max.y,
            target.min.x + 20.0,
            target.min.y,
            1.0,
            Some(lane),
        );
        assert_eq!(routing.count, 8);
        let max_x = routing
            .polyline(64)
            .into_iter()
            .fold(f32::NEG_INFINITY, |maximum, point| maximum.max(point.x));
        assert!(max_x >= lane - 0.5);
    }

    #[test]
    fn corner_handle_matches_the_circle_approximation() {
        let handle = arc_handle(100.0);
        assert!((handle - 55.228_475).abs() < 1.0e-3);
    }

    #[test]
    fn longer_and_more_vertical_cables_get_larger_handles() {
        let short = route(0.0, 0.0, 0.0, 20.0, 1.0);
        let long = route(0.0, 0.0, 0.0, 300.0, 1.0);
        assert!(long.length_factor > short.length_factor);
        assert!(long.segments[0].p1.y > short.segments[0].p1.y);
    }
}
