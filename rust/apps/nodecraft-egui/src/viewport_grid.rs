//! Centred XY construction grid and world axes for the model viewport.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridKind {
    Minor,
    Major,
    AxisX,
    AxisY,
    AxisZ,
}

#[derive(Clone, Copy, Debug)]
pub struct GridLine {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub kind: GridKind,
    /// Opacity multiplier after the horizon fade.
    pub alpha: f32,
}


fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Minor lines every 100 units, with one major line every ten minor lines.
pub const GRID_MINOR_STEP: f64 = 100.0;
pub const GRID_MAJOR_EVERY: f64 = 10.0;
pub const GRID_MAJOR_STEP: f64 = GRID_MINOR_STEP * GRID_MAJOR_EVERY;

/// World-space spacing between minor grid lines at the fitted model scale.
pub fn grid_step(_radius: f64) -> f64 {
    GRID_MINOR_STEP
}

/// Power-of-ten minor spacing that follows the camera's zoom. The fitted camera
/// distance is three model radii, so `distance / 3` is the model scale; snapping
/// it to the nearest decade keeps roughly the same number of cells on screen and
/// subdivides the pattern as the user zooms in or out.


/// Upstream-style grid spacing for a view of the given half height.
///
/// Returns `(minor, major)`. This follows the sequence 1, 2, 5, 10, ... rather
/// than the old fixed 100-unit grid, so cells stay roughly the same size on
/// screen as the camera zooms.

#[cfg(test)]
pub fn adaptive_grid_level(camera_distance: f64) -> (f64, f64) {
    let scale = (camera_distance / 3.0).max(1.0e-6);
    let level = scale.log10();
    let exponent = level.floor();
    (10.0_f64.powf(exponent + 1.0), level - exponent)
}

#[cfg(test)]
pub fn adaptive_grid_step(camera_distance: f64) -> f64 {
    adaptive_grid_level(camera_distance).0 / 10.0
}

#[cfg(test)]
pub fn adaptive_grid(half_height: f64) -> (f64, f64) {
    let raw = (half_height / 10.0).max(1.0e-3);
    let p = 10.0_f64.powf(raw.log10().floor());
    let ratio = raw / p;
    let minor = if ratio < 2.0 {
        p
    } else if ratio < 5.0 {
        2.0 * p
    } else {
        5.0 * p
    };
    (minor, minor * 5.0)
}

/// Two grid levels for a smooth zoom crossfade.
///
/// Returns `(fine_step, coarse_step, fine_weight)`. The coarse level is always
/// 5× the fine step; the fine level fades in as the camera zooms in and out as
/// it zooms out, so newly introduced lines do not pop.
pub fn adaptive_grid_crossfade(half_height: f64) -> (f64, f64, f32) {
    let raw = (half_height / 10.0).max(1.0e-6);
    let level = (raw.log(5.0)).floor();
    let fine = 5.0_f64.powf(level);
    let t = (raw / fine).log(5.0).clamp(0.0, 1.0);
    let fine_weight = 1.0 - smoothstep64(0.35, 1.0, t) as f32;
    (fine, fine * 5.0, fine_weight)
}

/// Half-size of the XY grid for a view of the given half height and slant.
pub fn adaptive_grid_extent(half_height: f64, major: f64, slant: f64) -> f64 {
    let major = major.max(1.0e-3);
    let extent = (half_height * 3.0 / major).ceil() * major;
    (extent / slant.abs().max(0.25) / major).ceil() * major
}

/// Centre the XY grid under the camera target while keeping it near the view.
pub fn adaptive_grid_center(
    target: [f64; 3],
    back: [f64; 3],
    extent: f64,
    major: f64,
) -> [f64; 3] {
    let under = [target[0], target[1], 0.0];
    let slant = back[2];
    if slant.abs() <= 1.0e-3 {
        let major = major.max(1.0e-6);
        return [
            (under[0] / major).round() * major,
            (under[1] / major).round() * major,
            0.0,
        ];
    }
    let hit = [
        target[0] - back[0] * (target[2] / slant),
        target[1] - back[1] * (target[2] / slant),
        0.0,
    ];
    let along = [hit[0] - under[0], hit[1] - under[1], 0.0];
    let len = (along[0] * along[0] + along[1] * along[1]).sqrt();
    let center = if len > extent && len > 0.0 {
        let scale = extent / len;
        [under[0] + along[0] * scale, under[1] + along[1] * scale, 0.0]
    } else {
        hit
    };
    // Keep the grid world-aligned: if the centre followed the camera
    // continuously, the pattern would stay fixed on screen while the model
    // and axes moved.
    let major = major.max(1.0e-6);
    [
        (center[0] / major).round() * major,
        (center[1] / major).round() * major,
        0.0,
    ]
}

/// How far the XY plane is visible before the camera's far plane cuts it.
pub fn adaptive_grid_far_reach(
    eye: [f64; 3],
    back: [f64; 3],
    far: f64,
    center: [f64; 3],
) -> f64 {
    let view = [-back[0], -back[1], -back[2]];
    let forward = [view[0], view[1], 0.0];
    let len = (forward[0] * forward[0] + forward[1] * forward[1]).sqrt();
    if len < 1.0e-9 {
        return f64::INFINITY;
    }
    let depth = (center[0] - eye[0]) * view[0]
        + (center[1] - eye[1]) * view[1]
        + (center[2] - eye[2]) * view[2];
    if far <= depth {
        0.0
    } else {
        (far - depth) / len
    }
}

/// Start and end distance for the grid's distance fade.
pub fn adaptive_grid_fade(
    half_height: f64,
    slant: f64,
    extent: f64,
    far_reach: f64,
) -> (f64, f64) {
    let reach = half_height / slant.abs().max(0.25);
    let end = (2.2 * reach).min(extent).min(far_reach);
    (0.6 * reach.min(end), end)
}

fn smoothstep64(edge0: f64, edge1: f64, value: f64) -> f64 {
    let t = ((value - edge0) / (edge1 - edge0).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Coarse and fine level weights for a zoom-decade fraction. These are uploaded
/// as uniforms; the shader only evaluates the two masks.

/// World-space half-extent of the grid for a model radius.
pub fn grid_extent(radius: f64) -> f64 {
    let radius = radius.max(1.0e-3);
    let step = grid_step(radius);
    let half_cells = (radius * 80.0 / step).round().clamp(8.0, 400.0);
    half_cells * step
}

/// Build a Z=0 grid centred on the world origin, plus positive world axes.
/// The extent follows the model radius but stays bounded to a manageable line count.
pub fn grid_lines(radius: f64) -> Vec<GridLine> {
    let radius = radius.max(1.0e-3);
    let step = grid_step(radius);
    let half_cells = (radius * 80.0 / step).round().clamp(8.0, 400.0);
    let extent = grid_extent(radius) as f32;
    let step = step as f32;
    let half = half_cells as i32;
    let mut lines = Vec::with_capacity((half as usize * 4) + 3);
    let fade_segments = 8usize;
    let push_segment = |lines: &mut Vec<GridLine>, start: [f32; 3], end: [f32; 3], kind: GridKind| {
        for segment in 0..fade_segments {
            let t0 = segment as f32 / fade_segments as f32;
            let t1 = (segment + 1) as f32 / fade_segments as f32;
            let a = [
                start[0] + (end[0] - start[0]) * t0,
                start[1] + (end[1] - start[1]) * t0,
                start[2] + (end[2] - start[2]) * t0,
            ];
            let b = [
                start[0] + (end[0] - start[0]) * t1,
                start[1] + (end[1] - start[1]) * t1,
                start[2] + (end[2] - start[2]) * t1,
            ];
            let midpoint = [
                (a[0] + b[0]) * 0.5,
                (a[1] + b[1]) * 0.5,
                (a[2] + b[2]) * 0.5,
            ];
            let radius = (midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1]).sqrt();
            let t = (radius / extent).clamp(0.0, 1.0);
            let alpha = 1.0 - smoothstep(0.30, 0.92, t);
            if alpha > 0.01 {
                lines.push(GridLine { start: a, end: b, kind, alpha });
            }
        }
    };
    for index in -half..=half {
        if index == 0 {
            continue;
        }
        let coordinate = index as f32 * step;
        let major_every = (GRID_MAJOR_STEP / step as f64).round().max(1.0) as i32;
        let kind = if index % major_every == 0 { GridKind::Major } else { GridKind::Minor };
        push_segment(&mut lines, [-extent, coordinate, 0.0], [extent, coordinate, 0.0], kind);
        push_segment(&mut lines, [coordinate, -extent, 0.0], [coordinate, extent, 0.0], kind);
    }
    lines.push(GridLine { start: [0.0; 3], end: [extent, 0.0, 0.0], kind: GridKind::AxisX, alpha: 1.0 });
    lines.push(GridLine { start: [0.0; 3], end: [0.0, extent, 0.0], kind: GridKind::AxisY, alpha: 1.0 });
    lines.push(GridLine { start: [0.0; 3], end: [0.0, 0.0, extent], kind: GridKind::AxisZ, alpha: 1.0 });
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_grid_subdivides_with_camera_half_height() {
        assert_eq!(adaptive_grid(40.0), (2.0, 10.0));
        assert_eq!(adaptive_grid(300.0), (20.0, 100.0));
        assert_eq!(adaptive_grid(3_000.0), (200.0, 1_000.0));
        let (minor, major) = adaptive_grid(150.0);
        assert_eq!(major, minor * 5.0);
    }

    #[test]
    fn adaptive_grid_crossfade_keeps_fine_and_coarse_related() {
        let (fine, coarse, weight) = adaptive_grid_crossfade(40.0);
        assert_eq!(coarse, fine * 5.0);
        assert!((0.0..=1.0).contains(&weight));
        let (fine, coarse, _) = adaptive_grid_crossfade(300.0);
        assert_eq!(coarse, fine * 5.0);
    }

    #[test]
    fn grid_center_follows_camera_target_and_stays_near_view() {
        let center = adaptive_grid_center([120.0, -40.0, 0.0], [0.0, 0.0, 1.0], 1_000.0, 20.0);
        assert_eq!(center, [120.0, -40.0, 0.0]);
        let center = adaptive_grid_center([120.0, -40.0, 30.0], [0.0, 0.0, 1.0], 1_000.0, 20.0);
        assert_eq!(center, [120.0, -40.0, 0.0]);
        let snapped = adaptive_grid_center([123.0, -46.0, 0.0], [0.0, 0.0, 1.0], 1_000.0, 20.0);
        assert_eq!(snapped, [120.0, -40.0, 0.0]);
        let moved = adaptive_grid_center([180.0, -46.0, 0.0], [0.0, 0.0, 1.0], 1_000.0, 20.0);
        assert_eq!(moved, [180.0, -40.0, 0.0]);
    }

    #[test]
    fn grid_subdivides_by_decades_with_zoom() {
        assert_eq!(adaptive_grid_step(30.0), 10.0);
        assert_eq!(adaptive_grid_step(300.0), 100.0);
        assert_eq!(adaptive_grid_step(3_000.0), 1_000.0);
        assert_eq!(adaptive_grid_level(30.0).0, 100.0);
        assert!((adaptive_grid_level(150.0).1 - 0.699).abs() < 1.0e-3);
        assert_eq!(GRID_MAJOR_STEP / GRID_MINOR_STEP, 10.0);
    }

    #[test]
    fn grid_is_centred_on_the_xy_plane() {
        let lines = grid_lines(10.0);
        assert!(!lines.is_empty());
        assert!(lines
            .iter()
            .filter(|line| matches!(line.kind, GridKind::Minor | GridKind::Major))
            .all(|line| line.start[2] == 0.0 && line.end[2] == 0.0 && (0.0..=1.0).contains(&line.alpha)));
        let axis_z = lines.iter().find(|line| line.kind == GridKind::AxisZ).expect("z axis");
        assert_eq!(axis_z.start, [0.0; 3]);
        assert!(axis_z.end[2] > 0.0);
        assert_eq!(axis_z.end[0], 0.0);
        assert_eq!(axis_z.end[1], 0.0);
    }
}
