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
pub fn adaptive_grid_level(camera_distance: f64) -> (f64, f64) {
    let scale = (camera_distance / 3.0).max(1.0e-6);
    let level = scale.log10();
    let exponent = level.floor();
    // A decade is one level: at t = 0 the coarse grid is shown, and by t = 1
    // the next finer grid has fully replaced it.
    (10.0_f64.powf(exponent + 1.0), level - exponent)
}

pub fn adaptive_grid_step(camera_distance: f64) -> f64 {
    adaptive_grid_level(camera_distance).0 / 10.0
}

fn smoothstep64(edge0: f64, edge1: f64, value: f64) -> f64 {
    let t = ((value - edge0) / (edge1 - edge0).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Coarse and fine level weights for a zoom-decade fraction. These are uploaded
/// as uniforms; the shader only evaluates the two masks.
pub fn grid_level_weights(level: f64) -> (f32, f32) {
    let fine = smoothstep64(0.0, 0.65, level);
    let coarse = 1.0 - smoothstep64(0.35, 1.0, level);
    (coarse as f32, fine as f32)
}

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
