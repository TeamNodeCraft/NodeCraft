//! Screen-space level of detail for the GPU display path.
//!
//! A display plan's chord tolerance is chosen so the body's projected error
//! stays inside a fixed number of physical pixels. The same body therefore gets
//! a finer plan when the camera moves closer and a coarser one on a denser
//! screen, without any per-node tuning.

use nodecraft_solidcore::solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};
use nodecraft_solidcore::solidcore::topology::Brep;

use crate::{DisplayQuality, DisplayTransform};

/// World-space bounding sphere of one display body, after its placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyBounds {
    pub center: [f64; 3],
    pub radius: f64,
}

impl BodyBounds {
    /// Place the bounds with the body's one composed placement matrix (the same value the plan,
    /// mesh, edge and BRep paths use). Placements are rigid motions, mirrors and uniform scales, so
    /// the radius scales by cbrt(|det|).
    #[must_use]
    pub fn placed(self, placement: &nodecraft_exact::Placement) -> Self {
        let m = placement.matrix();
        let scale = m.determinant().abs().cbrt();
        Self { center: m.apply_point(self.center), radius: self.radius * scale }
    }

    /// Place the bounds by a list of operations, first to last (tests and callers that only have
    /// operations). Delegates to `placed`, so there is still one placement path.
    #[must_use]
    pub fn transformed(self, transforms: &[DisplayTransform]) -> Self {
        self.placed(&nodecraft_exact::Placement::from_ops(transforms.iter().cloned()))
    }
}

/// World-space bounding sphere of a B-rep before placement (uses the approximate
/// solidcore query box, not the tight box).
#[must_use]
pub fn body_bounds(brep: &Brep) -> Option<BodyBounds> {
    let kernel = nodecraft_solidcore::kernel::kernel();
    // Display only needs a conservative extent. The tight box evaluates every
    // edge and face; the approximate query box is cheap and never smaller than
    // the body.
    let extent = calculate_bounding_box(
        &kernel,
        brep,
        BoundingBoxOptions {
            approx: true,
            tight: false,
            expand_by_zone_tol_3d: true,
        },
    )
    .ok()?;
    let mut center = [0.0; 3];
    let mut diagonal_sq = 0.0;
    for (axis, slot) in center.iter_mut().enumerate() {
        let min = extent.min_at(axis);
        let max = extent.max_at(axis);
        if !min.is_finite() || !max.is_finite() {
            return None;
        }
        *slot = (min + max) * 0.5;
        let span = max - min;
        diagonal_sq += span * span;
    }
    let radius = 0.5 * diagonal_sq.sqrt();
    if !radius.is_finite() || center.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(BodyBounds { center, radius })
}

/// The camera values that determine how large a world unit is on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodCamera {
    pub eye: [f64; 3],
    /// Vertical field of view in radians; <= 0.0 means orthographic.
    pub fov_y: f64,
    /// Visible world height for an orthographic camera.
    pub ortho_height: f64,
}

/// Logical viewport size (points); physical height is height * pixels_per_point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportSize {
    pub width: f64,
    pub height: f64,
}

/// Camera, viewport and error-budget values the viewport publishes for LOD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodView {
    pub eye: [f64; 3],
    pub fov_y: f64,
    pub ortho_height: f64,
    pub viewport: [f64; 2],
    pub pixels_per_point: f64,
    /// Requested screen-space error budget in physical pixels. Included so a
    /// settings change re-runs the level computation without moving the camera.
    pub error_px: f64,
}

/// One screen-space tolerance choice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodParams {
    pub error_px: f64,
    pub min_tolerance: f64,
    pub max_tolerance: f64,
    pub tiny_body_px: f64,
    pub refine_error_px: f64,
    pub coarsen_error_px: f64,
    /// Previous level for hysteresis; `None` means first choice.
    pub current_tolerance: Option<f64>,
}

impl LodParams {
    pub const DEFAULT_ERROR_PX: f64 = 0.75;

    #[must_use]
    pub fn new(error_px: f64) -> Self {
        Self {
            error_px,
            min_tolerance: LOD_LADDER[0],
            max_tolerance: LOD_LADDER[LOD_LADDER.len() - 1],
            tiny_body_px: 4.0,
            refine_error_px: 1.0,
            coarsen_error_px: 0.35,
            current_tolerance: None,
        }
    }
}

impl Default for LodParams {
    fn default() -> Self {
        Self::new(Self::DEFAULT_ERROR_PX)
    }
}

/// Power-of-two ladder anchored at 0.05: each rung doubles the previous one.
pub const LOD_LADDER: [f64; 10] = [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2];

/// Physical viewport height, never zero.
fn viewport_height_px(viewport: ViewportSize, pixels_per_point: f64) -> f64 {
    viewport.height.max(1.0) * pixels_per_point.max(1.0e-9)
}

/// The tangent of half the vertical field of view, when the camera is
/// perspective. A non-finite or non-positive field of view is orthographic; a
/// tangent that is not finite and positive is treated the same way so a hostile
/// camera cannot turn the pixel size negative.
fn perspective_tangent(fov_y: f64) -> Option<f64> {
    if !fov_y.is_finite() || fov_y <= 0.0 {
        return None;
    }
    let tangent = (fov_y * 0.5).tan();
    (tangent.is_finite() && tangent > 0.0).then_some(tangent)
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// A finite, strictly positive value. `fallback` is used for NaN, zero and
/// negative results; positive infinity becomes `f64::MAX`.
fn finite_positive(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        if value > 0.0 {
            value
        } else {
            fallback.max(f64::MIN_POSITIVE)
        }
    } else if value > 0.0 {
        f64::MAX
    } else {
        fallback.max(f64::MIN_POSITIVE)
    }
}

fn nonnegative_radius(radius: f64) -> f64 {
    if radius.is_finite() {
        radius.max(0.0)
    } else {
        0.0
    }
}

/// World units covered by one physical pixel at the body's silhouette.
#[must_use]
pub fn world_per_pixel(
    bounds: BodyBounds,
    camera: LodCamera,
    viewport: ViewportSize,
    pixels_per_point: f64,
) -> f64 {
    let height_px = viewport_height_px(viewport, pixels_per_point);
    let value = if let Some(tangent) = perspective_tangent(camera.fov_y) {
        let distance = distance3(camera.eye, bounds.center);
        let depth = (distance - nonnegative_radius(bounds.radius)).max(1.0e-6);
        2.0 * depth * tangent / height_px
    } else {
        camera.ortho_height.max(1.0e-9) / height_px
    };
    finite_positive(value, 1.0)
}

/// The body's projected diameter in physical pixels. Perspective measures from
/// the centre, not the silhouette, so a body at the near plane does not divide
/// by zero.
#[must_use]
pub fn projected_diameter_px(
    bounds: BodyBounds,
    camera: LodCamera,
    viewport: ViewportSize,
    pixels_per_point: f64,
) -> f64 {
    let height_px = viewport_height_px(viewport, pixels_per_point);
    let radius = nonnegative_radius(bounds.radius);
    let value = if let Some(tangent) = perspective_tangent(camera.fov_y) {
        let distance = distance3(camera.eye, bounds.center).max(1.0e-6);
        let pixel_at_center = 2.0 * distance * tangent / height_px;
        2.0 * radius / pixel_at_center
    } else {
        let pixel = camera.ortho_height.max(1.0e-9) / height_px;
        2.0 * radius / pixel
    };
    if value.is_finite() {
        value.max(0.0)
    } else if value > 0.0 {
        f64::MAX
    } else {
        0.0
    }
}

/// The ladder rung nearest `tolerance`; ties choose the finer (lower) rung.
#[must_use]
pub fn ladder_index(tolerance: f64) -> usize {
    if tolerance.is_nan() {
        return 0;
    }
    if tolerance == f64::INFINITY {
        return LOD_LADDER.len() - 1;
    }
    if tolerance == f64::NEG_INFINITY {
        return 0;
    }
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for (index, &rung) in LOD_LADDER.iter().enumerate() {
        let distance = (rung - tolerance).abs();
        // Strictly less keeps the earlier (finer) rung on an exact tie.
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

fn sanitize_tolerance(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

/// Index range allowed by the requested min/max tolerance: the first rung at or
/// above `min` through the last rung at or below `max`.
fn allowed_range(params: &LodParams) -> (usize, usize) {
    let last = LOD_LADDER.len() - 1;
    let min = sanitize_tolerance(params.min_tolerance, LOD_LADDER[0]);
    let max = sanitize_tolerance(params.max_tolerance, LOD_LADDER[last]);
    let low = LOD_LADDER
        .iter()
        .position(|&rung| rung >= min)
        .unwrap_or(last);
    let high = LOD_LADDER
        .iter()
        .rposition(|&rung| rung <= max)
        .unwrap_or(0);
    // An inverted range has no legal rung; ignore it rather than return a
    // tolerance outside the fixed ladder.
    if low <= high { (low, high) } else { (0, last) }
}

fn clamp_index(index: usize, (low, high): (usize, usize)) -> usize {
    index.clamp(low, high)
}

fn first_choice(pixel: f64, error_px: f64, allowed: (usize, usize)) -> usize {
    let ideal = error_px * pixel;
    // The largest rung still inside the error budget. Below the first rung the
    // budget is unreachable; the ladder has no finer level, so use rung 0.
    let index = LOD_LADDER
        .iter()
        .rposition(|&rung| rung <= ideal)
        .unwrap_or(0);
    clamp_index(index, allowed)
}

fn hysteresis(current: f64, pixel: f64, params: &LodParams, allowed: (usize, usize)) -> usize {
    let index = ladder_index(current);
    let projected = current / pixel;
    let refine = sanitize_tolerance(params.refine_error_px, 1.0);
    let coarsen = sanitize_tolerance(params.coarsen_error_px, 0.35);
    let moved = if projected > refine {
        index.saturating_sub(1)
    } else if projected < coarsen {
        (index + 1).min(LOD_LADDER.len() - 1)
    } else {
        index
    };
    clamp_index(moved, allowed)
}

/// Choose the tolerance for one body from its projected size.
///
/// The first choice is the finest plan that keeps the projected error within
/// `params.error_px`. Once a level exists, hysteresis moves at most one rung per
/// call so a slow zoom crosses each boundary once. Bodies smaller than
/// `params.tiny_body_px` always take the coarsest allowed rung.
#[must_use]
pub fn quality_for(
    bounds: BodyBounds,
    camera: LodCamera,
    viewport: ViewportSize,
    pixels_per_point: f64,
    params: LodParams,
) -> DisplayQuality {
    let pixel = world_per_pixel(bounds, camera, viewport, pixels_per_point);
    let allowed = allowed_range(&params);
    let tiny_px = projected_diameter_px(bounds, camera, viewport, pixels_per_point);
    let tiny_limit = if params.tiny_body_px.is_finite() {
        params.tiny_body_px.max(0.0)
    } else {
        0.0
    };
    if tiny_px < tiny_limit {
        return DisplayQuality::Full(LOD_LADDER[allowed.1]);
    }
    let error_px = sanitize_tolerance(params.error_px, LodParams::DEFAULT_ERROR_PX);
    let index = match params.current_tolerance {
        Some(current) if current.is_finite() && current > 0.0 => {
            hysteresis(current, pixel, &params, allowed)
        }
        _ => first_choice(pixel, error_px, allowed),
    };
    DisplayQuality::Full(LOD_LADDER[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOV_90: f64 = std::f64::consts::FRAC_PI_2;

    fn tolerance(quality: DisplayQuality) -> f64 {
        match quality {
            DisplayQuality::Full(tolerance) => tolerance,
            DisplayQuality::Preview(detail) => panic!("LOD chose preview detail {detail}"),
        }
    }

    fn perspective_at(distance: f64) -> LodCamera {
        LodCamera {
            eye: [0.0, 0.0, distance],
            fov_y: FOV_90,
            ortho_height: 0.0,
        }
    }

    fn viewport_height(height: f64) -> ViewportSize {
        ViewportSize {
            width: height * 1.5,
            height,
        }
    }

    fn bounds(radius: f64) -> BodyBounds {
        BodyBounds {
            center: [0.0; 3],
            radius,
        }
    }

    /// Pick a camera distance whose ideal tolerance is `ideal` for a 90° fov,
    /// a 1000-point viewport at 1 px/point and the default 0.75 px error.
    /// The tiny factor keeps the floating-point result just above the rung.
    fn distance_for_ideal(ideal: f64, radius: f64) -> f64 {
        ideal * 1000.0 / (2.0 * LodParams::DEFAULT_ERROR_PX) * (1.0 + 1.0e-12) + radius
    }

    #[test]
    fn constant_screen_error_at_one_and_four_times_distance() {
        let radius = 10.0;
        let body = bounds(radius);
        let view = viewport_height(1000.0);
        let params = LodParams::default();

        let near = perspective_at(distance_for_ideal(0.1, radius));
        let far = perspective_at(distance_for_ideal(0.4, radius));
        let near_quality = quality_for(body, near, view, 1.0, params);
        let far_quality = quality_for(body, far, view, 1.0, params);

        assert_eq!(near_quality, DisplayQuality::Full(0.1));
        assert_eq!(far_quality, DisplayQuality::Full(0.4));
        let ratio = tolerance(far_quality) / tolerance(near_quality);
        assert!((ratio - 4.0).abs() < 1.0e-9, "tolerance ratio {ratio}");

        for (camera, quality) in [(near, near_quality), (far, far_quality)] {
            let pixel = world_per_pixel(body, camera, view, 1.0);
            let projected_error = tolerance(quality) / pixel;
            assert!(
                projected_error <= params.error_px + 1.0e-9,
                "projected error {projected_error} exceeds {}",
                params.error_px
            );
        }
    }

    #[test]
    fn slow_zoom_sweep_crosses_one_boundary_once() {
        let radius = 10.0;
        let body = bounds(radius);
        let view = viewport_height(1000.0);
        let params = LodParams::default();
        let start_depth = distance_for_ideal(0.1, radius) - radius;

        let mut current = None;
        let mut levels = Vec::new();
        for step in 0..40 {
            // Gradual zoom-out: the silhouette distance grows by 2% per step.
            let depth = start_depth * 1.02_f64.powi(step);
            let camera = perspective_at(depth + radius);
            let next = quality_for(
                body,
                camera,
                view,
                1.0,
                LodParams {
                    current_tolerance: current,
                    ..params
                },
            );
            let rung = tolerance(next);
            levels.push(rung);
            current = Some(rung);
        }

        let changes = levels.windows(2).filter(|pair| pair[0] != pair[1]).count();
        assert_eq!(changes, 1, "levels changed more than once: {levels:?}");
        assert!(
            levels.windows(2).all(|pair| pair[1] >= pair[0]),
            "zoom-out never made the level finer: {levels:?}"
        );
    }

    #[test]
    fn retina_is_one_step_finer() {
        let radius = 10.0;
        let body = bounds(radius);
        let view = viewport_height(1000.0);
        let params = LodParams::default();
        let camera = perspective_at(distance_for_ideal(0.13, radius));

        let at_one = tolerance(quality_for(body, camera, view, 1.0, params));
        let at_two = tolerance(quality_for(body, camera, view, 2.0, params));

        assert!(
            at_two < at_one,
            "retina tolerance {at_two} is not finer than {at_one}"
        );
        let one = ladder_index(at_one);
        let two = ladder_index(at_two);
        assert!(two <= one);
        assert!(one - two <= 1, "retina moved from rung {one} to {two}");
    }

    #[test]
    fn orthographic_uses_view_height() {
        let body = bounds(1.0);
        let params = LodParams::default();
        let at = |ortho_height: f64, viewport_height: f64| {
            let camera = LodCamera {
                eye: [0.0, 0.0, 10.0],
                fov_y: 0.0,
                ortho_height,
            };
            let view = ViewportSize {
                width: 1500.0,
                height: viewport_height,
            };
            tolerance(quality_for(body, camera, view, 1.0, params))
        };

        assert_eq!(at(100.0, 1000.0), LOD_LADDER[3]);
        // Doubling the visible world height doubles the pixel size, so the
        // tolerance moves exactly one rung coarser.
        assert_eq!(at(200.0, 1000.0), LOD_LADDER[4]);
        assert_eq!(at(200.0, 1000.0), at(100.0, 1000.0) * 2.0);
        // Doubling the physical viewport height halves the pixel size.
        assert_eq!(at(100.0, 2000.0), LOD_LADDER[2]);
        assert_eq!(at(100.0, 2000.0), at(100.0, 1000.0) / 2.0);

        let pixel = world_per_pixel(
            body,
            LodCamera {
                eye: [0.0, 0.0, 10.0],
                fov_y: 0.0,
                ortho_height: 100.0,
            },
            ViewportSize {
                width: 1500.0,
                height: 1000.0,
            },
            1.0,
        );
        assert!((pixel - 0.1).abs() < 1.0e-12, "orthographic pixel {pixel}");
    }

    #[test]
    fn tiny_body_uses_coarsest_level() {
        let radius = 1.0e-6;
        let body = bounds(radius);
        // Close enough that the projected error budget wants the finest rung,
        // but far enough that the body projects to less than the tiny-body
        // threshold.
        let camera = perspective_at(1.0e-3);
        let view = viewport_height(1000.0);
        let params = LodParams::default();

        assert!(projected_diameter_px(body, camera, view, 1.0) < params.tiny_body_px);
        assert_eq!(
            quality_for(body, camera, view, 1.0, params),
            DisplayQuality::Full(params.max_tolerance)
        );
    }

    #[test]
    fn ladder_is_power_of_two_and_quantized() {
        assert_eq!(LOD_LADDER.len(), 10);
        assert_eq!(LOD_LADDER[0], 0.00625);
        assert_eq!(LOD_LADDER[3], 0.05);
        for pair in LOD_LADDER.windows(2) {
            assert!(
                (pair[1] / pair[0] - 2.0).abs() < 1.0e-12,
                "{pair:?} is not a doubling"
            );
        }

        let radius = 10.0;
        let body = bounds(radius);
        let view = viewport_height(1000.0);
        let params = LodParams::default();
        for &rung in &LOD_LADDER {
            let camera = perspective_at(distance_for_ideal(rung, radius));
            let quality = quality_for(body, camera, view, 1.0, params);
            assert!(
                LOD_LADDER.contains(&tolerance(quality)),
                "{quality:?} is not a ladder rung"
            );
            assert_eq!(tolerance(quality), rung);
        }

        // Off-rung ideals still quantize to a rung.
        for step in 0..120 {
            let ideal = 0.004 * 10.0_f64.powf(f64::from(step) / 40.0);
            let camera = perspective_at(distance_for_ideal(ideal, radius));
            let quality = quality_for(body, camera, view, 1.0, params);
            assert!(
                LOD_LADDER.contains(&tolerance(quality)),
                "ideal {ideal}: {quality:?} is not a ladder rung"
            );
        }

        // Nearest-rung lookup and its endpoints.
        for (index, &rung) in LOD_LADDER.iter().enumerate() {
            assert_eq!(ladder_index(rung), index);
        }
        assert_eq!(ladder_index(f64::NEG_INFINITY), 0);
        assert_eq!(ladder_index(f64::INFINITY), LOD_LADDER.len() - 1);
        assert_eq!(ladder_index(f64::NAN), 0);
    }

    #[test]
    fn first_choice_projected_error_is_within_budget() {
        let radius = 10.0;
        let body = bounds(radius);
        let view = viewport_height(1000.0);
        let params = LodParams::default();

        // Ideal tolerances from just above the first rung through the last.
        for step in 0..60 {
            let ideal = 0.01 * 320.0_f64.powf(f64::from(step) / 59.0);
            let camera = perspective_at(distance_for_ideal(ideal, radius));
            let quality = quality_for(body, camera, view, 1.0, params);
            let pixel = world_per_pixel(body, camera, view, 1.0);
            let projected_error = tolerance(quality) / pixel;
            assert!(
                projected_error <= params.error_px + 1.0e-9,
                "ideal {ideal}: projected error {projected_error} exceeds {}",
                params.error_px
            );
        }
    }

    /// The plan path's point transform, repeated here so the bounds helper is
    /// checked against the same maths rather than against itself.
    fn reference_transform(mut p: [f64; 3], transform: &DisplayTransform) -> [f64; 3] {
        match transform {
            DisplayTransform::Place { offset, rotation } => {
                let (rx, ry, rz) = (
                    rotation[0].to_radians(),
                    rotation[1].to_radians(),
                    rotation[2].to_radians(),
                );
                let (sx, cx) = rx.sin_cos();
                let (sy, cy) = ry.sin_cos();
                let (sz, cz) = rz.sin_cos();
                (p[1], p[2]) = (p[1] * cx - p[2] * sx, p[1] * sx + p[2] * cx);
                (p[0], p[2]) = (p[0] * cy + p[2] * sy, -p[0] * sy + p[2] * cy);
                (p[0], p[1]) = (p[0] * cz - p[1] * sz, p[0] * sz + p[1] * cz);
                p[0] += offset[0];
                p[1] += offset[1];
                p[2] += offset[2];
            }
            DisplayTransform::Mirror { plane, offset } => match *plane {
                "xz" => p[1] = 2.0 * offset - p[1],
                "yz" => p[0] = 2.0 * offset - p[0],
                _ => p[2] = 2.0 * offset - p[2],
            },
            DisplayTransform::Scale { factor } => {
                p[0] *= factor;
                p[1] *= factor;
                p[2] *= factor;
            }
        }
        p
    }

    fn assert_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1.0e-12,
                "axis {axis}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn body_bounds_transformed_matches_plan_transform() {
        let original = BodyBounds {
            center: [0.3, -0.7, 1.1],
            radius: 1.5,
        };
        let place = DisplayTransform::Place {
            offset: [10.0, -2.0, 3.5],
            rotation: [20.0, -35.0, 50.0],
        };

        let placed = original.transformed(std::slice::from_ref(&place));
        assert_close(placed.center, reference_transform(original.center, &place));
        assert_eq!(placed.radius, original.radius);

        let scale = DisplayTransform::Scale { factor: -2.0 };
        let mirror = DisplayTransform::Mirror {
            plane: "xz",
            offset: 0.75,
        };
        let chain = [place, scale, mirror];
        let expected_center = chain.iter().fold(original.center, |point, transform| {
            reference_transform(point, transform)
        });
        let transformed = original.transformed(&chain);
        assert_close(transformed.center, expected_center);
        assert_eq!(transformed.radius, original.radius * 2.0);

        // Every mirror plane keeps the radius; only Scale changes it.
        for (plane, offset) in [("xz", 0.5), ("yz", -1.25), ("xy", 2.0)] {
            let mirror = DisplayTransform::Mirror { plane, offset };
            let mirrored = original.transformed(std::slice::from_ref(&mirror));
            assert_close(
                mirrored.center,
                reference_transform(original.center, &mirror),
            );
            assert_eq!(mirrored.radius, original.radius);
        }
    }
}
