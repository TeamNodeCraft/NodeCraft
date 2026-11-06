//! NodeCraft evaluation engine.
//!
//! The engine is synchronous and thread-agnostic: the worker owns the engine
//! thread, request/output queues, and JSON transport. Progress is reported through
//! a typed callback so the engine never performs I/O.

use nodecraft_kernel::{Kernel, ProceduralKernel};
use base64::Engine as _;
mod comparison;
mod cache;
mod display_eval;
mod exact_eval;

pub use cache::EngineCache;
use cache::{import_cache_key, node_cache_is_fresh, stepio_error};
mod look;
mod primitive;
use look::{apply_brep_face_colors, apply_look_color, parse_look_color};
use primitive::{apply_brep_placement, apply_brep_placement_with_kernel, copy_brep, primitive_brep, primitive_brep_with_kernel, snapshot_brep};
mod bypass;
mod face_ref;
use bypass::{bypass_source, effective_output_suppressed, suppression_source};
use face_ref::{face_reference_body_and_index, face_reference_index};

pub(crate) use comparison::compare_parts;

use nodecraft_protocol::{CacheStats, Connection, Mesh, MeshPayload, Node, Request, Response};
use nodecraft_mesh::{import_step, merge_meshes, StepDiagnostics};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Shared SolidCore kernel constructor for worker threads.
#[must_use]
pub fn nodecraft_solidcore_kernel() -> nodecraft_solidcore::solidcore::kernel::Kernel {
    nodecraft_solidcore::kernel::kernel()
}

/// Upstream SolidCraft sketch model: points, curves, constraints, solver and
/// profile detection. This replaces the retired JS/Python sketch implementation.
mod node_task;
pub mod display;
pub mod mesher;
pub mod workload;
pub mod scheduler;
pub mod kernel_pool;

static WORKLOAD_PROFILER_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Enable or disable analysis-driven dispatch. Default is enabled.
/// Configure the opt-in SolidCore kernel fixes used by new worker kernels.
pub fn configure_solidcore_fixes(enabled: bool) {
    nodecraft_solidcore::kernel::configure_experimental_fixes(enabled);
}

#[must_use]
pub fn solidcore_fixes_enabled() -> bool {
    nodecraft_solidcore::kernel::experimental_fixes_enabled()
}

pub fn configure_workload_profiler(enabled: bool) {
    WORKLOAD_PROFILER_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[must_use]
pub fn workload_profiler_enabled() -> bool {
    WORKLOAD_PROFILER_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}
pub mod node_plugin;

pub mod sketch {
    pub use solidcraft_sketch::*;
}

fn source<'a>(request: &'a Request, node_id: &str) -> Option<&'a Connection> {
    request.connections.iter().find(|link| link.to == node_id && link.input == "shape")
}

fn node<'a>(request: &'a Request, id: &str) -> Option<&'a Node> {
    request.nodes.iter().find(|node| node.id == id)
}

fn display_options() -> nodecraft_solidcore::curved::CurvedTessOptions {
    nodecraft_solidcore::curved::CurvedTessOptions {
        u_segments: 64,
        v_segments: 16,
        loop_segments: 32,
        bspline_tolerance: 0.02,
        max_normal_angle_deg: 4.0,
    }
}

#[derive(Clone)]
struct SketchTransform {
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

impl SketchTransform {
    fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    fn from_values(values: &serde_json::Value) -> Self {
        let number = |key: &str| values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let (rx, ry, rz) = (number("rx").to_radians(), number("ry").to_radians(), number("rz").to_radians());
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        // Matches ProceduralKernel::translate: rotate X, then Y, then Z.
        let rotation = [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ];
        // `space` rotates the manipulator frame only. Geometry values remain
        // in the parent frame so switching world/local cannot move the object.
        Self { rotation, translation: [number("x"), number("y"), number("z")] }
    }

    /// Return the transform that applies `self` first, then `next`.
    fn then(&self, next: &Self) -> Self {
        let mut rotation = [[0.0; 3]; 3];
        for row in 0..3 {
            for column in 0..3 {
                rotation[row][column] = (0..3)
                    .map(|index| next.rotation[row][index] * self.rotation[index][column])
                    .sum();
            }
        }
        let translation = (0..3)
            .map(|row| {
                next.translation[row]
                    + (0..3).map(|column| next.rotation[row][column] * self.translation[column]).sum::<f64>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or([0.0; 3]);
        Self { rotation, translation }
    }

    fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        (0..3)
            .map(|row| {
                self.translation[row]
                    + (0..3).map(|column| self.rotation[row][column] * point[column]).sum::<f64>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap_or([0.0; 3])
    }

    fn is_z_rotation(&self) -> bool {
        const EPSILON: f64 = 1.0e-9;
        self.rotation[0][2].abs() < EPSILON
            && self.rotation[1][2].abs() < EPSILON
            && self.rotation[2][0].abs() < EPSILON
            && self.rotation[2][1].abs() < EPSILON
            && (self.rotation[2][2] - 1.0).abs() < EPSILON
    }

    fn z_angle(&self) -> f64 {
        self.rotation[1][0].atan2(self.rotation[0][0])
    }
}

#[derive(Debug)]
struct SketchOutput {
    sketch: crate::sketch::Sketch,
    profiles: Vec<crate::sketch::Profile>,
    offset: [f64; 3],
    rotation: [[f64; 3]; 3],
}

impl SketchOutput {
    fn new(sketch: crate::sketch::Sketch) -> Self {
        let profiles = crate::sketch::find_profiles(&sketch);
        Self {
            sketch,
            profiles,
            offset: [0.0; 3],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    fn with_transform(&self, transform: &SketchTransform) -> Self {
        Self {
            sketch: self.sketch.clone(),
            profiles: self.profiles.clone(),
            offset: transform.translation,
            rotation: transform.rotation,
        }
    }

    fn transform(&self) -> SketchTransform {
        SketchTransform { rotation: self.rotation, translation: self.offset }
    }

    fn point(&self, x: f64, y: f64) -> [f64; 3] {
        self.transform().apply([x, y, 0.0])
    }

    fn is_z_rotation(&self) -> bool {
        self.transform().is_z_rotation()
    }

    fn z_angle(&self) -> f64 {
        self.transform().z_angle()
    }
}

fn input_source<'a>(request: &'a Request, node_id: &str, input: &str) -> Option<&'a Connection> {
    request.connections.iter().find(|link| link.to == node_id && link.input == input)
}

fn resample_closed(points: &[(f64, f64)], count: usize) -> Vec<(f64, f64)> {
    if points.len() < 2 {
        return Vec::new();
    }
    let mut lengths = Vec::with_capacity(points.len());
    let mut perimeter = 0.0;
    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        let length = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        lengths.push(length);
        perimeter += length;
    }
    if perimeter <= 1.0e-12 {
        return Vec::new();
    }
    let step = perimeter / count as f64;
    let mut result = Vec::with_capacity(count);
    let (mut segment, mut along) = (0usize, 0.0_f64);
    for index in 0..count {
        let target = step * index as f64;
        while segment + 1 < points.len() && along + lengths[segment] < target {
            along += lengths[segment];
            segment += 1;
        }
        let a = points[segment];
        let b = points[(segment + 1) % points.len()];
        let t = ((target - along) / lengths[segment].max(1.0e-12)).clamp(0.0, 1.0);
        result.push((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
    }
    result
}

fn sketch_profile_points(output: &SketchOutput) -> Result<Vec<(f64, f64, f64)>, String> {
    use solidcraft_geom::prelude::*;
    let mut profiles = output.profiles.clone();
    profiles.sort_by(|a, b| b.area.total_cmp(&a.area));
    let profile = profiles.into_iter().next().ok_or("Loft sketch has no closed profile")?;
    if !profile.hole_curves.is_empty() {
        return Err("Loft sketch profiles with holes are not supported yet".into());
    }
    let points = resample_closed(
        &profile
            .region
            .outer
            .polyline(0.001)
            .iter()
            .map(|point| (point.x, point.y))
            .collect::<Vec<_>>(),
        256,
    );
    if points.len() < 3 {
        return Err("Loft sketch profile is degenerate".into());
    }
    Ok(points
        .into_iter()
        .map(|(x, y)| {
            let point = output.point(x, y);
            (point[0], point[1], point[2])
        })
        .collect())
}

fn simplify_polyline(points: &[[f64; 3]], tolerance: f64) -> Vec<[f64; 3]> {
    fn distance_to_segment(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ap = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
        let length_squared = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        if length_squared <= 1.0e-18 {
            return (ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]).sqrt();
        }
        let t = ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / length_squared).clamp(0.0, 1.0);
        let dx = ap[0] - ab[0] * t;
        let dy = ap[1] - ab[1] * t;
        let dz = ap[2] - ab[2] * t;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    fn simplify(points: &[[f64; 3]], first: usize, last: usize, tolerance: f64, keep: &mut [bool]) {
        if last <= first + 1 {
            return;
        }
        let mut farthest = first;
        let mut farthest_distance = -1.0;
        for index in first + 1..last {
            let distance = distance_to_segment(points[index], points[first], points[last]);
            if distance > farthest_distance {
                farthest_distance = distance;
                farthest = index;
            }
        }
        if farthest_distance > tolerance {
            keep[farthest] = true;
            simplify(points, first, farthest, tolerance, keep);
            simplify(points, farthest, last, tolerance, keep);
        }
    }
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    simplify(points, 0, points.len() - 1, tolerance, &mut keep);
    points.iter().zip(keep).filter_map(|(point, keep)| keep.then_some(*point)).collect()
}

fn sketch_profile_curves(
    output: &SketchOutput,
) -> Result<Vec<nodecraft_solidcore::solidcore::curve::curve::Curve>, String> {
    use solidcore_record_new::Seg2;
    use solidcraft_geom::prelude::*;
    use nodecraft_solidcore::solidcore::curve::curve::Curve;
    use nodecraft_solidcore::solidcore::curve::line::Line;
    use nodecraft_solidcore::solidcore::math::Point3;
    use nodecraft_solidcore::solidcore::primitive::api_curves;

    let mut profiles = output.profiles.clone();
    profiles.sort_by(|a, b| b.area.total_cmp(&a.area));
    let profile = profiles.into_iter().next().ok_or("Loft sketch has no closed profile")?;
    if !profile.hole_curves.is_empty() {
        return Err("Loft sketch profiles with holes are not supported yet".into());
    }
    if !output.is_z_rotation() {
        return sketch_profile_curves_transformed(&profile, output);
    }
    let z_angle = output.z_angle();
    let sample_points = profile.region.outer.polyline(0.01);
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for point in &sample_points {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }
    let extent = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt().max(1.0e-9);
    let simplify_tolerance = extent * 0.001;
    let mut curves = Vec::with_capacity(profile.region.outer.segs.len());
    let mut line_run: Vec<[f64; 3]> = Vec::new();
    let flush_line_run = |curves: &mut Vec<Curve>, run: &mut Vec<[f64; 3]>| -> Result<(), String> {
        if run.len() >= 2 {
            let simplified = simplify_polyline(run, simplify_tolerance);
            for pair in simplified.windows(2) {
                let line = Line::create_line_segment(
                    3,
                    Point3::new(pair[0][0], pair[0][1], pair[0][2]),
                    Point3::new(pair[1][0], pair[1][1], pair[1][2]),
                    None,
                )
                .map_err(|error| error.to_string())?;
                curves.push(Curve::from(line));
            }
        }
        run.clear();
        Ok(())
    };
    for segment in &profile.region.outer.segs {
        match *segment {
            Seg2::Line { a, b } => {
                let a = output.point(a.x, a.y);
                let b = output.point(b.x, b.y);
                let sharp = if line_run.len() >= 2 {
                    let previous = line_run[line_run.len() - 2];
                    let last = line_run[line_run.len() - 1];
                    let incoming = [last[0] - previous[0], last[1] - previous[1], last[2] - previous[2]];
                    let outgoing = [b[0] - last[0], b[1] - last[1], b[2] - last[2]];
                    let cross = [
                        incoming[1] * outgoing[2] - incoming[2] * outgoing[1],
                        incoming[2] * outgoing[0] - incoming[0] * outgoing[2],
                        incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                    ];
                    let cross_len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                    let dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1] + incoming[2] * outgoing[2];
                    cross_len.atan2(dot).abs() > 0.14 // ~8 degrees
                } else {
                    false
                };
                if sharp {
                    flush_line_run(&mut curves, &mut line_run)?;
                }
                if line_run.is_empty() {
                    line_run.push(a);
                } else {
                    let last = *line_run.last().unwrap_or(&a);
                    if (last[0] - a[0]).abs() + (last[1] - a[1]).abs() + (last[2] - a[2]).abs() > 1.0e-9 {
                        flush_line_run(&mut curves, &mut line_run)?;
                        line_run.push(a);
                    }
                }
                line_run.push(b);
            }
            Seg2::Arc { center, radius, start, sweep } => {
                flush_line_run(&mut curves, &mut line_run)?;
                let center = output.point(center.x, center.y);
                let center = Point3::new(center[0], center[1], center[2]);
                if sweep.abs() >= std::f64::consts::TAU - 1.0e-9 {
                    let circle = api_curves::create_circle(center, radius).map_err(|error| error.to_string())?;
                    curves.push(Curve::from(circle));
                } else {
                    let arc = api_curves::create_arc(
                        center,
                        radius,
                        (start + z_angle).to_degrees(),
                        (start + sweep + z_angle).to_degrees(),
                    )
                    .map_err(|error| error.to_string())?;
                    curves.push(Curve::from(arc));
                }
            }
            _ => return Err("Loft sketch profile has an unsupported curve type".into()),
        }
    }
    flush_line_run(&mut curves, &mut line_run)?;
    // A whole circle is one closed periodic curve and a valid profile. Extrude
    // splits it into halves when solidcore's direct path is unavailable; only a
    // profile with no boundary at all is degenerate.
    if curves.is_empty() {
        return Err("Sketch profile is degenerate".into());
    }
    Ok(curves)
}

fn sketch_profile_curves_transformed(
    profile: &crate::sketch::Profile,
    output: &SketchOutput,
) -> Result<Vec<nodecraft_solidcore::solidcore::curve::curve::Curve>, String> {
    use nodecraft_solidcore::solidcore::curve::curve::Curve;
    use nodecraft_solidcore::solidcore::curve::line::Line;
    use nodecraft_solidcore::solidcore::math::Point3;
    use solidcraft_geom::prelude::*;

    let points = profile.region.outer.polyline(0.001);
    if points.len() < 3 {
        return Err("Loft sketch profile is degenerate".into());
    }
    let mut curves = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        let a = output.point(a.x, a.y);
        let b = output.point(b.x, b.y);
        let line = Line::create_line_segment(3, Point3::new(a[0], a[1], a[2]), Point3::new(b[0], b[1], b[2]), None)
            .map_err(|error| error.to_string())?;
        curves.push(Curve::from(line));
    }
    Ok(curves)
}

fn sketch_record_plane(output: &SketchOutput) -> solidcore_record::Plane {
    let x = output.rotation[0];
    let y = output.rotation[1];
    solidcore_record::Plane {
        origin: solidcore_record::Vec3::new(output.offset[0], output.offset[1], output.offset[2]),
        x: solidcore_record::Vec3::new(x[0], x[1], x[2]),
        y: solidcore_record::Vec3::new(y[0], y[1], y[2]),
    }
}

fn sketch_record_section(output: &SketchOutput) -> Option<Vec<solidcore_record::profile::Seg2>> {
    use solidcore_record_new::Seg2;
    let mut profiles = output.profiles.clone();
    profiles.sort_by(|a, b| b.area.total_cmp(&a.area));
    let profile = profiles.into_iter().next()?;
    if !profile.hole_curves.is_empty() {
        return None;
    }
    profile
        .region
        .outer
        .segs
        .iter()
        .map(|segment| match *segment {
            Seg2::Line { a, b } => Some(solidcore_record::profile::Seg2::Line {
                a: solidcore_record::Vec2::new(a.x, a.y),
                b: solidcore_record::Vec2::new(b.x, b.y),
            }),
            Seg2::Arc { center, radius, start, sweep } => Some(solidcore_record::profile::Seg2::Arc {
                center: solidcore_record::Vec2::new(center.x, center.y),
                radius,
                start,
                sweep,
            }),
            _ => None,
        })
        .collect()
}

fn sketch_extrude_brep(
    output: &SketchOutput,
    distance: f64,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let curves = sketch_profile_curves(output)?;
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    nodecraft_extrude::extrude_profile_curves(&mut kernel, &curves, distance.abs())
        .map_err(|error| error.to_string())
}

fn sketch_from_node(node: &Node) -> Result<SketchOutput, String> {
    if node.node_type == "spline" {
        return spline_sketch_from_values(&node.values).map(SketchOutput::new);
    }
    let sketch: crate::sketch::Sketch = serde_json::from_value(
        node.values.get("sketch").cloned().unwrap_or(serde_json::Value::Null),
    )
    .map_err(|error| format!("Sketch values: {error}"))?;
    Ok(SketchOutput::new(sketch))
}

fn loft_sketch(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    bottom: &SketchOutput,
    top: &SketchOutput,
    ruled: bool,
    cap_ends: bool,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    use nodecraft_solidcore::solidcore::curve::curve::Curve;
    use nodecraft_solidcore::solidcore::curve::line::Line;
    use nodecraft_solidcore::solidcore::math::Point3;
    if let (Some(bottom_section), Some(top_section)) =
        (sketch_record_section(bottom), sketch_record_section(top))
    {
        let body = nodecraft_sweep::loft_on_planes(
            &[
                (bottom_section, sketch_record_plane(bottom)),
                (top_section, sketch_record_plane(top)),
            ],
            if ruled { 1 } else { 2 },
            cap_ends,
        )
        .map_err(|error| format!("Loft: {error}"))?;
        return nodecraft_layers::body_to_kernel(kernel, &body)
            .map_err(|error| format!("Loft: {error}"));
    }
    if let (Ok(bottom_curves), Ok(top_curves)) = (
        sketch_profile_curves(bottom),
        sketch_profile_curves(top),
    ) {
        if bottom_curves.len() == top_curves.len() {
            let mut curves = bottom_curves;
            curves.extend(top_curves);
            let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
            let count = refs.len() / 2;
            let skin = nodecraft_solidcore::solidcore::sweep::api_surfaces::create_skin_primitive(
                kernel,
                &[count, count],
                &refs,
                if ruled { 1 } else { 2 },
                cap_ends,
            )
            .map_err(|error| format!("Loft: {error}"))?;
            return Ok(skin.brep);
        }
    }
    let bottom_points = sketch_profile_points(bottom)?;
    let top_points = sketch_profile_points(top)?;
    let count = bottom_points.len().min(top_points.len());
    if count < 3 {
        return Err("Loft sketch profile has fewer than three points".into());
    }
    let mut curves: Vec<Curve> = Vec::with_capacity(count * 2);
    for points in [&bottom_points, &top_points] {
        for index in 0..count {
            let a = points[index];
            let b = points[(index + 1) % count];
            let line = Line::create_line_segment(
                3,
                Point3::new(a.0, a.1, a.2),
                Point3::new(b.0, b.1, b.2),
                None,
            )
            .map_err(|error| error.to_string())?;
            curves.push(Curve::from(line));
        }
    }
    let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
    let skin = nodecraft_solidcore::solidcore::sweep::api_surfaces::create_skin_primitive(
        kernel,
        &[count, count],
        &refs,
        if ruled { 1 } else { 2 },
        cap_ends,
    )
    .map_err(|error| format!("Loft: {error}"))?;
    Ok(skin.brep)
}

fn sketch_section_curves(
    output: &SketchOutput,
) -> Result<Vec<nodecraft_solidcore::solidcore::curve::curve::Curve>, String> {
    if !output.profiles.is_empty() {
        return sketch_profile_curves(output);
    }
    use nodecraft_solidcore::solidcore::curve::curve::Curve;
    use nodecraft_solidcore::solidcore::math::Point3;
    use nodecraft_solidcore::solidcore::primitive::api_curves;
    let open_curves: Vec<usize> = output
        .sketch
        .curves
        .iter()
        .enumerate()
        .filter(|(_, curve)| !curve.construction && !curve.centerline)
        .map(|(index, _)| index)
        .collect();
    if open_curves.len() != 1 {
        return Err("open Surface Loft sections need exactly one spline curve".into());
    }
    let points: Vec<Point3> = output
        .sketch
        .polyline(open_curves[0])
        .into_iter()
        .map(|point| {
            let point = output.point(point.x, point.y);
            Point3::new(point[0], point[1], point[2])
        })
        .collect();
    if points.len() < 4 {
        return Err("open Surface Loft spline needs at least four sampled points".into());
    }
    let curve = api_curves::create_curve_approx_points(&points)
        .map_err(|error| format!("Surface Loft spline: {error}"))?;
    Ok(vec![Curve::from(curve)])
}

fn multi_section_surface_loft(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    sections: &[SketchOutput],
    ruled: bool,
    cap_ends: bool,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    use nodecraft_solidcore::solidcore::curve::curve::Curve;
    if sections.len() < 2 {
        return Err("Surface Loft needs at least two sections".into());
    }
    let record_sections: Option<Vec<_>> = sections
        .iter()
        .map(|section| sketch_record_section(section).map(|segments| (segments, sketch_record_plane(section))))
        .collect();
    if let Some(record_sections) = record_sections {
        let body = nodecraft_sweep::loft_on_planes(
            &record_sections,
            if ruled { 1 } else { 2 },
            cap_ends,
        )
        .map_err(|error| format!("Surface Loft: {error}"))?;
        return nodecraft_layers::body_to_kernel(kernel, &body)
            .map_err(|error| format!("Surface Loft: {error}"));
    }
    let mut all_curves: Vec<Vec<Curve>> = Vec::with_capacity(sections.len());
    for section in sections {
        all_curves.push(sketch_section_curves(section)?);
    }
    let count = all_curves.first().map_or(0, Vec::len);
    if count == 0 {
        return Err("Surface Loft section is degenerate".into());
    }
    if all_curves.iter().any(|curves| curves.len() != count) {
        return Err("Surface Loft sections have different boundary curve counts".into());
    }
    let mut curves: Vec<Curve> = Vec::with_capacity(count * sections.len());
    for section_curves in &all_curves {
        curves.extend(section_curves.iter().cloned());
    }
    let refs: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
    let counts = vec![count; sections.len()];
    let skin = nodecraft_solidcore::solidcore::sweep::api_surfaces::create_skin_primitive(
        kernel,
        &counts,
        &refs,
        if ruled { 1 } else { 2 },
        cap_ends,
    )
    .map_err(|error| format!("Surface Loft: {error}"))?;
    Ok(skin.brep)
}

fn loft_face(
    kernel: &nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &nodecraft_solidcore::solidcore::topology::Brep,
    top: bool,
) -> Option<nodecraft_solidcore::solidcore::topology::FaceId> {
    let mut best: Option<(f64, nodecraft_solidcore::solidcore::topology::FaceId)> = None;
    for face in brep.faces().ids() {
        let Ok(plane) = nodecraft_extrude::face_plane(kernel, brep, face) else {
            continue;
        };
        let normal_ok = if top { plane.normal.z > 0.5 } else { plane.normal.z < -0.5 };
        if !normal_ok {
            continue;
        }
        let better = best.as_ref().is_none_or(|(best_z, _)| {
            if top { plane.point.z > *best_z } else { plane.point.z < *best_z }
        });
        if better {
            best = Some((plane.point.z, face));
        }
    }
    best.map(|(_, face)| face)
}

fn loft_exact(
    request: &Request,
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    bottom_start: &str,
    top_start: &str,
    ruled: bool,
    cap_ends: bool,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let bottom = exact_operand_brep(request, bottom_start)?;
    let top = exact_operand_brep(request, top_start)?;
    let bottom_face = loft_face(kernel, &bottom, true).ok_or("Loft bottom has no upward planar face")?;
    let top_face = loft_face(kernel, &top, false).ok_or("Loft top has no downward planar face")?;
    let bottom_loop = nodecraft_extrude::face_boundary(kernel, &bottom, bottom_face)
        .map_err(|error| format!("Loft bottom boundary: {error}"))?
        .into_iter()
        .next()
        .ok_or("Loft bottom face has no boundary")?;
    let top_loop = nodecraft_extrude::face_boundary(kernel, &top, top_face)
        .map_err(|error| format!("Loft top boundary: {error}"))?
        .into_iter()
        .next()
        .ok_or("Loft top face has no boundary")?;
    let bottom_curves = bottom_loop
        .oriented_curves(kernel)
        .map_err(|error| format!("Loft bottom curves: {error}"))?;
    let top_curves = top_loop
        .oriented_curves(kernel)
        .map_err(|error| format!("Loft top curves: {error}"))?;
    if bottom_curves.len() != top_curves.len() {
        return Err("Loft sections have different boundary curve counts".into());
    }
    let mut curves: Vec<Option<&nodecraft_solidcore::solidcore::curve::curve::Curve>> = Vec::new();
    for curve in &bottom_curves {
        curves.push(Some(curve));
    }
    for curve in &top_curves {
        curves.push(Some(curve));
    }
    let degree = if ruled { 1 } else { 2 };
    let skin = nodecraft_solidcore::solidcore::sweep::api_surfaces::create_skin_primitive(
        kernel,
        &[bottom_curves.len(), top_curves.len()],
        &curves,
        degree,
        cap_ends,
    )
    .map_err(|error| format!("Loft: {error}"))?;
    Ok(skin.brep)
}

fn boolean_mode(values: &serde_json::Value) -> Result<(nodecraft_bool::BooleanOp, bool), String> {
    let mode = values.get("mode").or_else(|| values.get("operation")).or_else(|| values.get("op")).and_then(serde_json::Value::as_str).unwrap_or("union");
    let reverse = mode == "subtract-reverse";
    let op = match mode {
        "union" => nodecraft_bool::BooleanOp::Union,
        "subtract" | "difference" | "subtract-a-b" | "subtract-reverse" => nodecraft_bool::BooleanOp::Difference,
        "intersect" | "intersection" => nodecraft_bool::BooleanOp::Intersection,
        other => return Err(format!("unsupported Boolean mode {other:?}")),
    };
    Ok((op, reverse))
}

fn export_extrude_exact(request: &Request, extrude_id: &str, path: &str) -> Result<(), String> {
    let node = request.nodes.iter().find(|item| item.id == extrude_id).ok_or("extrude node missing")?;
    let link = request.connections.iter().find(|link| link.to == extrude_id && (link.input == "profile" || link.input == "shape")).ok_or("extrude has no profile")?;
    let source = request.nodes.iter().find(|item| item.id == link.from).ok_or("extrude profile source missing")?;
    let input = source.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
    if input.is_empty() { return Err("exact extrude export requires a STEP import profile".into()); }
    let imported = nodecraft_mesh::import_step_brep(input)?;
    let reference = node.values.get("faces").and_then(serde_json::Value::as_array).and_then(|values| values.first()).and_then(|value| value.get("id")).and_then(serde_json::Value::as_str).unwrap_or("");
    let index: usize = face_reference_index(reference).ok_or("extrude needs face:N selection")?;
    let face = imported.faces().ids().nth(index).ok_or_else(|| format!("extrude face {index} missing"))?;
    let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let result = nodecraft_extrude::extrude_planar_face_exact(&mut kernel, &imported, face, amount).map_err(|error| format!("extrude: {error}"))?;
    nodecraft_extrude::write_planar_brep_step(&kernel, &result, path).map_err(|error| format!("{error}"))
}


fn resolve_extrude_chain<'a>(request: &'a Request, start: &str) -> Option<(&'a Node, [f64; 3], [f64; 3])> {
    let mut current = start.to_string();
    let mut offset = [0.0; 3];
    let mut rotation = [0.0; 3];
    for _ in 0..64 {
        let node = request.nodes.iter().find(|item| item.id == current)?;
        if node.node_type == "translate" {
            offset[0] += node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[1] += node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[2] += node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            for (slot, key) in ["rx", "ry", "rz"].iter().enumerate() {
                rotation[slot] = node.values.get(*key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            }
            let next = request.connections.iter().find(|link| link.to == current && (link.input == "shape" || link.input == "profile"))?;
            current = next.from.clone();
        } else if node.node_type == "extrude" {
            return Some((node, offset, rotation));
        } else {
            return None;
        }
    }
    None
}

fn export_extrude_chain(request: &Request, start: &str, path: &str) -> Result<(), String> {
    let (extrude, offset, rotation) = resolve_extrude_chain(request, start).ok_or("no extrude in chain")?;
    export_extrude_exact(request, &extrude.id, path)?;
    nodecraft_step::transform_step_file(path, rotation, offset)
}


fn resolve_part_chain<'a>(request: &'a Request, start: &str) -> Option<(&'a Node, [f64; 3], [f64; 3])> {
    let mut current = start.to_string();
    let mut offset = [0.0; 3];
    let mut rotation = [0.0; 3];
    for _ in 0..64 {
        let node = request.nodes.iter().find(|item| item.id == current)?;
        if node.node_type == "translate" {
            offset[0] += node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[1] += node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[2] += node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            for (slot, key) in ["rx", "ry", "rz"].iter().enumerate() {
                rotation[slot] = node.values.get(*key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            }
            let next = request.connections.iter().find(|link| link.to == current && link.input == "shape")?;
            current = next.from.clone();
        } else if node.node_type == "stepimporter" || node.node_type == "extrude" || node.node_type == "boolean" || node.node_type == "fuse" || (node.node_type == "solid" || node.node_type == "primitive") {
            return Some((node, offset, rotation));
        } else {
            return None;
        }
    }
    None
}

fn exact_part_body(request: &Request, node: &Node) -> Result<nodecraft_step::PlanarBody, String> {
    if node.node_type == "solid" || node.node_type == "primitive" {
        let kind = node.values.get("primitiveType").and_then(serde_json::Value::as_str).unwrap_or("");
        let number = |key: &str, default: f64| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(default);
        return match kind {
            "pyramid" => nodecraft_step::pyramid_body(number("length", 40.0), number("height", 30.0)),
            other => Err(format!("assembly part does not support primitive {other:?}")),
        };
    }
    if node.node_type == "stepimporter" {
        let path = node.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
        let bodies = nodecraft_step::read_planar_solids(path)?;
        if bodies.len() != 1 { return Err("assembly part must contain exactly one planar solid".into()); }
        return Ok(bodies.into_iter().next().unwrap());
    }
    if node.node_type == "extrude" {
        let link = request.connections.iter().find(|link| link.to == node.id && (link.input == "profile" || link.input == "shape")).ok_or("extrude has no profile")?;
        let source = request.nodes.iter().find(|item| item.id == link.from).ok_or("extrude profile missing")?;
        let input = source.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
        if input.is_empty() { return Err("extrude part requires a STEP profile".into()); }
        let imported = nodecraft_mesh::import_step_brep(input)?;
        let reference = node.values.get("faces").and_then(serde_json::Value::as_array).and_then(|values| values.first()).and_then(|value| value.get("id")).and_then(serde_json::Value::as_str).unwrap_or("");
        let index: usize = face_reference_index(reference).ok_or("extrude needs face:N")?;
        let face = imported.faces().ids().nth(index).ok_or("extrude face missing")?;
        let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let result = nodecraft_extrude::extrude_planar_face_exact(&mut kernel, &imported, face, amount).map_err(|error| error.to_string())?;
        return nodecraft_extrude::planar_brep_to_planar_body(&kernel, &result).map_err(|error| error.to_string());
    }
    if node.node_type == "boolean" || node.node_type == "fuse" {
        let a = request.connections.iter().find(|link| link.to == node.id && link.input == "a").ok_or("Boolean part missing A")?;
        let b = request.connections.iter().find(|link| link.to == node.id && link.input == "b").ok_or("Boolean part missing B")?;
        let (op, reverse) = boolean_mode(&node.values)?;
        let (brep_a, brep_b) = if reverse { (exact_operand_brep(request, &b.from)?, exact_operand_brep(request, &a.from)?) } else { (exact_operand_brep(request, &a.from)?, exact_operand_brep(request, &b.from)?) };
        let result = nodecraft_bool::boolean_breps(op, brep_a, brep_b)?;
        let kernel = nodecraft_solidcore::kernel::kernel();
        return nodecraft_extrude::planar_brep_to_planar_body(&kernel, &result).map_err(|error| error.to_string());
    }
    Err("unsupported assembly part type".into())
}

fn export_combine_assembly(request: &Request, combine_id: &str, path: &str) -> Result<(), String> {
    let mut parts = Vec::new();
    for (index, link) in request.connections.iter().filter(|link| link.to == combine_id && link.input == "shape").enumerate() {
        let (part_node, offset, rotation) = resolve_part_chain(request, &link.from).ok_or("assembly export requires STEP import or Extrude parts")?;
        let body = exact_part_body(request, part_node)?;
        let (sx, cx) = rotation[0].to_radians().sin_cos();
        let (sy, cy) = rotation[1].to_radians().sin_cos();
        let (sz, cz) = rotation[2].to_radians().sin_cos();
        let rows = [
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz, offset[0]],
            [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz, offset[1]],
            [-sy, cy * sx, cx * cy, offset[2]],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let transform = step_assemblies::Transform::from_rows(rows).map_err(|error| error.to_string())?;
        parts.push(step_assemblies::PlanarPart::placed(format!("Part {index}"), body, transform));
    }
    if parts.len() < 2 { return Err("assembly export needs at least two parts".into()); }
    step_assemblies::write_planar_assembly(path, &parts).map_err(|error| error.to_string())
}



type WarmBrep = std::sync::Arc<nodecraft_solidcore::solidcore::topology::Brep>;
type WarmDeps = std::collections::BTreeMap<String, Result<WarmBrep, String>>;

fn fnv_seed(id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn exact_operand_breps(request: &Request, start: &str) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
    let node = request.nodes.iter().find(|item| item.id == start).ok_or("Boolean operand node missing")?;
    if node.node_type == "mirror" {
        let link = request.connections.iter().find(|link| link.to == start && link.input == "shape").ok_or("mirror has no shape input")?;
        let mut inputs = exact_operand_breps(request, &link.from)?;
        let mut outputs = Vec::new();
        for input in &inputs {
            let mirrored = crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Mirror { brep: copy_brep(input)?, values: node.values.clone() },
                &node.id,
                fnv_seed(&node.id),
            )?
            .into_brep()
            .ok_or_else(|| "mirror task returned no shape".to_owned())?;
            outputs.push(mirrored);
        }
        if mirror_combine(&node.values) {
            inputs.extend(outputs);
            Ok(inputs)
        } else {
            Ok(outputs)
        }
    } else if node.node_type == "translate" {
        let link = request.connections.iter().find(|link| link.to == start && link.input == "shape").ok_or("Boolean transform has no source")?;
        let mut inputs = exact_operand_breps(request, &link.from)?;
        let offset = [node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
        let rotation = [node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
        for brep in &mut inputs { apply_brep_placement(brep, offset, rotation)?; }
        Ok(inputs)
    } else if node.node_type == "combine" {
        // A Part is a collection of solids. Keep every connected operand rather than
        // collapsing the Part to its first shape when it reaches a Boolean.
        let mut inputs = Vec::new();
        for link in request.connections.iter().filter(|link| link.to == start && link.input == "shape") {
            inputs.extend(exact_operand_breps(request, &link.from)?);
        }
        if inputs.is_empty() { return Err("Combine exact operand has no inputs".into()); }
        Ok(inputs)
    } else {
        Ok(vec![exact_operand_brep(request, start)?])
    }
}

fn exact_operand_brep(request: &Request, start: &str) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let node = request.nodes.iter().find(|item| item.id == start).ok_or("Boolean operand node missing")?;
    if node.node_type == "translate" {
        let link = request.connections.iter().find(|link| link.to == start && link.input == "shape").ok_or("Boolean transform has no source")?;
        let mut brep = exact_operand_brep(request, &link.from)?;
        let offset = [node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
        let rotation = [node.values.get("rx").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("ry").and_then(serde_json::Value::as_f64).unwrap_or(0.0), node.values.get("rz").and_then(serde_json::Value::as_f64).unwrap_or(0.0)];
        apply_brep_placement(&mut brep, offset, rotation)?;
        return Ok(brep);
    }
    if node.node_type == "solid" || node.node_type == "primitive" {
        return primitive_brep(node);
    }
    match node.node_type.as_str() {
        "stepimporter" => {
            let path = node.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
            if path.is_empty() { return Err("Boolean STEP import has no path".into()); }
            nodecraft_mesh::import_step_brep(path)
        }
        "extrude" => {
            let link = request.connections.iter().find(|link| link.to == node.id && (link.input == "profile" || link.input == "shape")).ok_or("extrude operand has no profile")?;
            let source = request.nodes.iter().find(|item| item.id == link.from).ok_or("extrude profile source missing")?;
            let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
            if source.node_type == "sketch" || source.node_type == "spline" {
                return sketch_extrude_brep(&sketch_from_node(source)?, amount);
            }
            let path = source.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
            if path.is_empty() { return Err("extrude operand requires a STEP profile".into()); }
            let imported = nodecraft_mesh::import_step_brep(path)?;
            let reference = node.values.get("faces").and_then(serde_json::Value::as_array).and_then(|values| values.first()).and_then(|value| value.get("id")).and_then(serde_json::Value::as_str).unwrap_or("");
            let index: usize = face_reference_index(reference).ok_or("extrude operand needs face:N")?;
            let face = imported.faces().ids().nth(index).ok_or("extrude operand face missing")?;
            let amount = node.values.get("amount").and_then(serde_json::Value::as_f64).unwrap_or(10.0);
            let _ = face;
            crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Extrude {
                    brep: imported,
                    face_index: index,
                    amount,
                },
                &node.id,
                fnv_seed(&node.id),
            )?
            .into_brep()
            .ok_or_else(|| "extrude task returned no shape".to_owned())
        }
        "boolean" | "fuse" => {
            let a = request.connections.iter().find(|link| link.to == node.id && link.input == "a");
            let b = request.connections.iter().find(|link| link.to == node.id && link.input == "b");
            match (a, b) {
                (Some(a), Some(b)) => {
                    let (op, reverse) = boolean_mode(&node.values)?;
                    let (a, b) = if reverse { (b.from.clone(), a.from.clone()) } else { (a.from.clone(), b.from.clone()) };
                    let a_breps = exact_operand_breps(request, &a)?;
                    let b_breps = exact_operand_breps(request, &b)?;
                    let mut a_iter = a_breps.into_iter();
                    let mut result = a_iter.next().ok_or("Boolean operand A is empty")?;
                    // A Combine represents several solids as one Part. Union the
                    // Part's pieces first so a later Subtract/Intersect applies to
                    // the whole Part, not just its first connected shape.
                    for brep in a_iter {
                        result = nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Union, result, brep)?;
                    }
                    for brep in b_breps {
                        result = nodecraft_bool::boolean_breps(op, result, brep)?;
                    }
                    Ok(result)
                }
                (Some(a), None) => exact_operand_brep(request, &a.from),
                (None, Some(b)) => exact_operand_brep(request, &b.from),
                (None, None) => Err("Boolean has no connected operand".into()),
            }
        }
        "chamfer" => {
            let link = request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("chamfer has no shape input")?;
            let source = exact_operand_brep(request, &link.from)?;
            let mut values = node.values.clone();
            values["mode"] = serde_json::json!("chamfer");
            crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Chamfer { brep: source, values },
                &node.id,
                fnv_seed(&node.id),
            )?
            .into_brep()
            .ok_or_else(|| "Edge preview: chamfer task returned no shape".to_owned())
            .map_err(|error| format!("Edge preview: {error}"))
        }
        "shell" => {
            let link = request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("shell has no shape input")?;
            let source = exact_operand_brep(request, &link.from)?;
            let thickness = node.values.get("thickness").and_then(serde_json::Value::as_f64).unwrap_or(2.0);
            let inward = node.values.get("direction").and_then(serde_json::Value::as_str).unwrap_or("inward") == "inward";
            let signed = if inward { -thickness } else { thickness };
            let mut openings = Vec::new();
            for face in node.values.get("faces").and_then(serde_json::Value::as_array).into_iter().flatten() {
                let Some(reference) = face.get("id").and_then(serde_json::Value::as_str) else { continue };
                let Some((body, index)) = face_reference_body_and_index(reference) else { continue };
                if body.unwrap_or(0) != 0 { continue; }
                if let Some(face) = source.faces().ids().nth(index) { openings.push(face); }
            }
            crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Shell {
                    brep: source,
                    signed_thickness: signed,
                    openings,
                },
                &node.id,
                fnv_seed(&node.id),
            )?
            .into_brep()
            .ok_or_else(|| "Shell: task returned no shape".to_owned())
        }
        "mirror" => {
            let link = request.connections.iter().find(|link| link.to == node.id && link.input == "shape").ok_or("mirror has no shape input")?;
            let source_brep = exact_operand_brep(request, &link.from)?;
            let mut mirrored = copy_brep(&source_brep)?;
            mirror_brep(&mut mirrored, &node.values)?;
            if mirror_combine(&node.values) {
                nodecraft_bool::boolean_breps(nodecraft_bool::BooleanOp::Union, source_brep, mirrored)
            } else {
                Ok(mirrored)
            }
        }
        "combine" => {
            let mut inputs = Vec::new();
            for link in request.connections.iter().filter(|link| link.to == node.id && link.input == "shape") {
                inputs.push(exact_operand_brep(request, &link.from)?);
            }
            match inputs.len() {
                0 => Err("Combine exact operand has no inputs".into()),
                1 => Ok(inputs.pop().expect("one input")),
                _ => Err("Combine exact operand with multiple inputs is not supported; use a Boolean".into()),
            }
        }
        other => Err(format!("exact Boolean operand does not support {other:?}")),
    }
}


fn export_boolean_exact(request: &Request, boolean_id: &str, path: &str) -> Result<(), String> {
    let node = request.nodes.iter().find(|item| item.id == boolean_id).ok_or("boolean node missing")?;
    let link_a = request.connections.iter().find(|link| link.to == boolean_id && link.input == "a");
    let link_b = request.connections.iter().find(|link| link.to == boolean_id && link.input == "b");
    let result = match (link_a, link_b) {
        (Some(a), Some(b)) => {
            let (op, reverse) = boolean_mode(&node.values)?;
            let (a, b) = if reverse { (b.from.clone(), a.from.clone()) } else { (a.from.clone(), b.from.clone()) };
            let a_breps = exact_operand_breps(request, &a)?;
            let b_breps = exact_operand_breps(request, &b)?;
            let mut result = a_breps.into_iter().next().ok_or("Boolean operand A is empty")?;
            for brep in b_breps { result = nodecraft_bool::boolean_breps(op, result, brep)?; }
            result
        }
        (Some(a), None) => exact_operand_brep(request, &a.from)?,
        (None, Some(b)) => exact_operand_brep(request, &b.from)?,
        (None, None) => return Err("boolean has no connected operand".into()),
    };
    nodecraft_step::write_brep_exact(path, &result)
}


fn exports_boolean_chain(request: &Request, start: &str) -> bool {
    let mut current = start.to_string();
    for _ in 0..64 {
        let Some(node) = request.nodes.iter().find(|item| item.id == current) else { return false };
        if node.node_type == "translate" {
            let Some(next) = request.connections.iter().find(|link| link.to == current && link.input == "shape") else { return false };
            current = next.from.clone();
        } else { return node.node_type == "boolean" || node.node_type == "fuse"; }
    }
    false
}

/// Exact consumers that need their upstream primitives as BReps during normal
/// evaluation. Mesh-only branches must not pay for this.
fn exact_brep_demands(request: &Request) -> HashSet<String> {
    let mut demands = HashSet::new();
    let mut stack: Vec<String> = request
        .nodes
        .iter()
        .filter(|node| matches!(
            node.node_type.as_str(),
            "boolean" | "fuse" | "chamfer" | "fillet" | "shell" | "extrude" | "loft" | "surfaceloft" | "exporter"
        ))
        .map(|node| node.id.clone())
        .collect();
    while let Some(id) = stack.pop() {
        if !demands.insert(id.clone()) {
            continue;
        }
        for link in request.connections.iter().filter(|link| link.to == id) {
            stack.push(link.from.clone());
        }
    }
    demands
}

fn cacheable_exact_operand(request: &Request, start: &str) -> bool {
    let mut current = start.to_owned();
    for _ in 0..64 {
        let Some(node) = request.nodes.iter().find(|node| node.id == current) else { return false };
        match node.node_type.as_str() {
            "translate" => {
                let Some(next) = request.connections.iter().find(|link| link.to == current && link.input == "shape") else { return false };
                current = next.from.clone();
            }
            "boolean" | "fuse" | "loft" | "surfaceloft" => return true,
            _ => return false,
        }
    }
    false
}

fn export_boolean_chain(request: &Request, start: &str, path: &str) -> Result<(), String> {
    let mut current = start.to_string();
    let mut offset = [0.0; 3];
    let mut rotation = [0.0; 3];
    let mut boolean = None;
    for _ in 0..64 {
        let node = request.nodes.iter().find(|item| item.id == current).ok_or("boolean chain node missing")?;
        if node.node_type == "translate" {
            offset[0] += node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[1] += node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[2] += node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            for (slot, key) in ["rx", "ry", "rz"].iter().enumerate() { rotation[slot] = node.values.get(*key).and_then(serde_json::Value::as_f64).unwrap_or(0.0); }
            let next = request.connections.iter().find(|link| link.to == current && link.input == "shape").ok_or("boolean chain is not connected")?;
            current = next.from.clone();
        } else if node.node_type == "boolean" || node.node_type == "fuse" {
            boolean = Some(node);
            break;
        } else { return Err("no Boolean in transform chain".into()); }
    }
    let boolean = boolean.ok_or("no Boolean in transform chain")?;
    export_boolean_exact(request, &boolean.id, path)?;
    nodecraft_step::transform_step_file(path, rotation, offset)
}




fn write_faceted_from_mesh(path: &str, mesh: &Mesh) -> Result<(), String> {
    if mesh.faces.is_empty() { return Err("cannot export an empty mesh".into()); }
    let mut body = nodecraft_step::PlanarBody::default();
    let mut lookup: std::collections::BTreeMap<[i64; 3], usize> = std::collections::BTreeMap::new();
    for triangle in &mesh.faces {
        let mut loop_points = Vec::with_capacity(3);
        for index in triangle {
            let vertex = mesh.vertices.get(*index as usize).ok_or("mesh triangle index out of range")?;
            let point = [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64];
            let key = [(point[0] * 1e7).round() as i64, (point[1] * 1e7).round() as i64, (point[2] * 1e7).round() as i64];
            let target = *lookup.entry(key).or_insert_with(|| { body.points.push(point); body.points.len() - 1 });
            if loop_points.last().copied() != Some(target) { loop_points.push(target); }
        }
        if loop_points.len() == 3 { body.push_face(loop_points, Vec::new()); }
    }
    nodecraft_step::write_faceted_brep(path, &body)
}

fn apply_export_metadata(node: &Node, path: &str) -> Result<(), String> {
    let target = match node.values.get("unit").and_then(serde_json::Value::as_str).unwrap_or("mm").to_ascii_lowercase().as_str() {
        "cm" | "centimetre" | "centimeter" => Some(nodecraft_step::LengthUnit::Centimetre),
        "m" | "metre" | "meter" => Some(nodecraft_step::LengthUnit::Metre),
        "km" | "kilometre" | "kilometer" => Some(nodecraft_step::LengthUnit::Kilometre),
        "um" | "micrometre" | "micrometer" => Some(nodecraft_step::LengthUnit::Micrometre),
        "in" | "inch" => Some(nodecraft_step::LengthUnit::Inch),
        _ => None,
    };
    if let Some(target) = target {
        let source = nodecraft_step::read_length_unit(path).unwrap_or(nodecraft_step::LengthUnit::Millimetre);
        nodecraft_step::convert_step_units(path, source, target)?;
    }
    let name = node.values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("solid");
    let author = node.values.get("author").and_then(serde_json::Value::as_str).unwrap_or("");
    let organization = node.values.get("organization").and_then(serde_json::Value::as_str).unwrap_or("NodeCraft");
    nodecraft_step::set_header(path, name, author, organization)?;
    if let Some(schema) = node.values.get("schema").and_then(serde_json::Value::as_str) { nodecraft_step::set_schema(path, schema)?; }
    if let Some(name) = node.values.get("productName").and_then(serde_json::Value::as_str) { nodecraft_step::set_product_name(path, name)?; }
    if let Some(layer) = node.values.get("layer").and_then(serde_json::Value::as_str) { nodecraft_step::set_layer(path, layer)?; }
    if let Some(material) = node.values.get("material").and_then(serde_json::Value::as_str) { nodecraft_step::set_material(path, material)?; }
    if let Some(color) = node.values.get("color").and_then(serde_json::Value::as_str) {
        let hex = color.trim_start_matches('#');
        if hex.len() != 6 { return Err("color must be #RRGGBB".into()); }
        let component = |range| u8::from_str_radix(&hex[range], 16).map(|value| value as f64 / 255.0).map_err(|error| error.to_string());
        nodecraft_step::set_color(path, component(0..2)?, component(2..4)?, component(4..6)?)?;
    }
    Ok(())
}

fn export_imported_step(request: &Request, start: &str, path: &str) -> Result<(), String> {
    let (importer, offset, rotation) = resolve_part_chain(request, start).ok_or("no STEP import in chain")?;
    if importer.node_type != "stepimporter" { return Err("not an imported STEP source".into()); }
    let input = importer.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
    if input.is_empty() { return Err("STEP import has no path".into()); }
    std::fs::copy(input, path).map_err(|error| format!("{path}: {error}"))?;
    nodecraft_step::transform_step_file(path, rotation, offset)
}

fn write_primitive_step(node: &Node, path: &str) -> Result<(), String> {
    let number = |key: &str, default: f64| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(default);
    let kind = node.values.get("primitiveType").and_then(serde_json::Value::as_str).unwrap_or("");
    match kind {
        "box" => nodecraft_step::write_box(path, number("length", 40.0), number("width", 40.0), number("height", 40.0)),
        "cylinder" => nodecraft_step::write_cylinder(path, number("radius", 20.0), number("height", 40.0)),
        "cone" => nodecraft_step::write_cone(path, number("bottomRadius", 20.0), number("topRadius", 0.0), number("height", 40.0)),
        "sphere" => nodecraft_step::write_sphere(path, number("radius", 20.0)),
        "torus" => nodecraft_step::write_torus(path, number("majorRadius", 15.0), number("minorRadius", 5.0)),
        "wedge" => nodecraft_step::write_wedge(path, number("length", 40.0), number("width", 30.0), number("height", 25.0), number("topLength", 0.0)),
        "pyramid" => nodecraft_step::write_pyramid(path, number("length", 40.0), number("height", 30.0)),
        other => Err(format!("Rust STEP export does not support primitive {other:?}.")),
    }
}

fn resolve_primitive_chain<'a>(request: &'a Request, start: &str) -> Option<(&'a Node, [f64; 3], [f64; 3])> {
    let mut current = start.to_string();
    let mut offset = [0.0; 3];
    let mut rotation = [0.0; 3];
    for _ in 0..64 {
        let node = request.nodes.iter().find(|item| item.id == current)?;
        if node.node_type == "translate" {
            offset[0] += node.values.get("x").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[1] += node.values.get("y").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            offset[2] += node.values.get("z").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            for (slot, key) in ["rx", "ry", "rz"].iter().enumerate() {
                rotation[slot] = node.values.get(*key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            }
            let next = request.connections.iter().find(|link| link.to == current && link.input == "shape")?;
            current = next.from.clone();
        } else if node.node_type == "solid" || node.node_type == "primitive" {
            return Some((node, offset, rotation));
        } else {
            return None;
        }
    }
    None
}

fn export_combine_exact(request: &Request, combine_id: &str, output: &str) -> Result<(), String> {
    let mut paths = Vec::new();
    let mut temporary = Vec::new();
    for (index, link) in request.connections.iter().filter(|link| link.to == combine_id && link.input == "shape").enumerate() {
        let Some((primitive, offset, rotation)) = resolve_primitive_chain(request, &link.from) else {
            return Err("Combine STEP export supports only primitives and transformed primitives.".into());
        };
        let temp = format!("{output}.part{index}.step");
        write_primitive_step(primitive, &temp)?;
        if rotation != [0.0; 3] || offset != [0.0; 3] {
            nodecraft_step::transform_step_file(&temp, rotation, offset)?;
        }
        paths.push(temp.clone()); temporary.push(temp);
    }
    let merged = nodecraft_step::merge_step_files(&paths, output);
    for path in temporary { let _ = std::fs::remove_file(path); }
    merged
}

fn mirror_plane(values: &serde_json::Value) -> (&'static str, f64) {
    let plane = values.get("plane").and_then(serde_json::Value::as_str).unwrap_or("xy");
    let offset = values.get("offset").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    match plane {
        "xz" => ("xz", offset),
        "yz" => ("yz", offset),
        _ => ("xy", offset),
    }
}

fn mirror_mesh(mesh: &Mesh, values: &serde_json::Value) -> Mesh {
    let (plane, offset) = mirror_plane(values);
    let mut mirrored = mesh.clone();
    for vertex in &mut mirrored.vertices {
        match plane {
            "xz" => vertex[1] = (2.0 * offset - vertex[1] as f64) as f32,
            "yz" => vertex[0] = (2.0 * offset - vertex[0] as f64) as f32,
            _ => vertex[2] = (2.0 * offset - vertex[2] as f64) as f32,
        }
    }
    for normal in &mut mirrored.normals {
        match plane {
            "xz" => normal[1] = -normal[1],
            "yz" => normal[0] = -normal[0],
            _ => normal[2] = -normal[2],
        }
    }
    // A reflection reverses handedness, so reverse every triangle to keep its
    // winding outward.
    for face in &mut mirrored.faces {
        face.swap(1, 2);
    }
    mirrored
}

fn mirror_combine(values: &serde_json::Value) -> bool {
    match values.get("combine") {
        Some(serde_json::Value::Bool(value)) => *value,
        Some(serde_json::Value::String(value)) => value == "true" || value == "join",
        _ => values.get("mode").and_then(serde_json::Value::as_str) == Some("join"),
    }
}

fn mirror_brep(brep: &mut nodecraft_solidcore::solidcore::topology::Brep, values: &serde_json::Value) -> Result<(), String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    mirror_brep_with_kernel(&mut kernel, brep, values)
}

fn mirror_brep_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    values: &serde_json::Value,
) -> Result<(), String> {
    use nodecraft_solidcore::solidcore::{geom::axis2_placement::Axis2Placement, math::{Point3, Vec3}, topology::brep::mirror};
    let (plane, offset) = mirror_plane(values);
    let (origin, x_axis, y_axis) = match plane {
        "xz" => (Point3::new(0.0, offset, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        "yz" => (Point3::new(offset, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        _ => (Point3::new(0.0, 0.0, offset), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    };
    // Mirror needs an independent, owning body rather than a view of the input.
    let mut standalone = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| error.to_string())?;
    mirror::mirror(kernel, &mut standalone, &Axis2Placement::new(origin, x_axis, y_axis)).map_err(|error| error.to_string())?;
    let mut mn=[f64::INFINITY;3]; let mut mx=[f64::NEG_INFINITY;3];
    for v in standalone.vertex_list() { if let Ok(vertex)=standalone.vertex(*v) { let p=vertex.point; mn[0]=mn[0].min(p.x);mn[1]=mn[1].min(p.y);mn[2]=mn[2].min(p.z);mx[0]=mx[0].max(p.x);mx[1]=mx[1].max(p.y);mx[2]=mx[2].max(p.z); } }
    *brep = standalone;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressPhase { Start, Complete, BrepStart, BrepComplete }

impl ProgressPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Start => "node-start",
            Self::Complete => "node-complete",
            Self::BrepStart => "brep-start",
            Self::BrepComplete => "brep-complete",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProgressEvent {
    pub request_id: u64,
    pub tab_id: Option<String>,
    pub graph_revision: Option<String>,
    pub node_id: String,
    pub phase: ProgressPhase,
}

const MAX_BOOLEAN_SOLIDS: usize = 256;

pub struct EngineOutput {
    pub response: Response,
    pub cache: EngineCache,
    pub diagnostics: HashMap<String, StepDiagnostics>,
}

fn spline_display_mesh(sketch: &crate::sketch::Sketch) -> Mesh {
    use solidcraft_geom::Vec2Geom;
    let points = sketch.polyline(0);
    if points.len() < 2 {
        return Mesh::default();
    }
    let half_width = 0.9_f64;
    let mut mesh = Mesh::default();
    mesh.vertices.reserve(points.len() * 2);
    mesh.normals.reserve(points.len() * 2);
    for (index, point) in points.iter().enumerate() {
        let previous = points[index.saturating_sub(1)];
        let next = points[(index + 1).min(points.len() - 1)];
        let mut tangent = next - previous;
        if tangent.len2() <= 1.0e-18 {
            tangent = solidcore_record_new::Vec2::new(1.0, 0.0);
        }
        let normal = solidcore_record_new::Vec2::new(-tangent.y, tangent.x) * (half_width / tangent.len());
        let left = *point + normal;
        let right = *point - normal;
        mesh.vertices.push([left.x as f32, left.y as f32, 0.0]);
        mesh.vertices.push([right.x as f32, right.y as f32, 0.0]);
        mesh.normals.push([0.0, 0.0, 1.0]);
        mesh.normals.push([0.0, 0.0, 1.0]);
    }
    for index in 0..points.len().saturating_sub(1) {
        let base = (index * 2) as u32;
        mesh.faces.push([base, base + 1, base + 2]);
        mesh.faces.push([base + 1, base + 3, base + 2]);
    }
    mesh
}

fn spline_sketch_from_values(values: &serde_json::Value) -> Result<crate::sketch::Sketch, String> {
    use crate::sketch::{CurveKind, MAX_SPLINE_POINTS, Sketch};
    let items = values
        .get("points")
        .and_then(serde_json::Value::as_array)
        .ok_or("Spline needs a points array")?;
    if items.len() < 2 {
        return Err("Spline needs at least two points".into());
    }
    if items.len() > MAX_SPLINE_POINTS {
        return Err(format!("Spline supports at most {MAX_SPLINE_POINTS} points"));
    }
    let mut sketch = Sketch::new();
    let mut point_ids = Vec::with_capacity(items.len());
    for item in items {
        let x = item.get("x").or_else(|| item.get(0)).and_then(serde_json::Value::as_f64);
        let y = item.get("y").or_else(|| item.get(1)).and_then(serde_json::Value::as_f64);
        let (Some(x), Some(y)) = (x, y) else {
            return Err("Spline points need finite x and y values".into());
        };
        if !x.is_finite() || !y.is_finite() {
            return Err("Spline points need finite x and y values".into());
        }
        let id = sketch
            .add_point(solidcore_record_new::Vec2::new(x, y), None)
            .map_err(|error| error.to_string())?;
        point_ids.push(id);
    }
    let control = values.get("control").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let periodic = values.get("periodic").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let degree = values
        .get("degree")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(3)
        .clamp(1, solidcore_record_new::MAX_BEZIER_DEGREE as u64) as u8;
    sketch
        .add_curve(CurveKind::Spline { pts: point_ids, control, degree, periodic, nurbs: None }, None)
        .map_err(|error| error.to_string())?;
    Ok(sketch)
}

/// One shared body plus its placement chain.
#[derive(Clone, Debug)]
struct BodyInstance {
    body: Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    transforms: Vec<BodyTransform>,
}

#[derive(Clone, Debug)]
enum BodyTransform {
    Place { offset: [f64; 3], rotation: [f64; 3] },
    Mirror { plane: &'static str, offset: f64 },
    Scale { factor: f64 },
}

impl From<&BodyTransform> for crate::display::DisplayTransform {
    fn from(transform: &BodyTransform) -> Self {
        match transform {
            BodyTransform::Place { offset, rotation } => Self::Place { offset: *offset, rotation: *rotation },
            BodyTransform::Mirror { plane, offset } => Self::Mirror { plane, offset: *offset },
            BodyTransform::Scale { factor } => Self::Scale { factor: *factor },
        }
    }
}


#[derive(Clone, Debug)]
struct SketchInstance {
    output: Arc<SketchOutput>,
    transforms: Vec<BodyTransform>,
}

/// Immutable graph value: placed solids plus placed sketch profiles/curves.
#[derive(Clone, Debug)]
struct Part {
    bodies: Vec<BodyInstance>,
    sketches: Vec<SketchInstance>,
}

impl Part {
    fn from_brep(brep: nodecraft_solidcore::solidcore::topology::Brep) -> Self {
        Self::from_arc(Arc::new(brep))
    }

    /// Share an immutable exact body between exact evaluation and meshing.
    /// Mutating consumers must take an owning copy at their operation boundary.
    fn from_arc(brep: Arc<nodecraft_solidcore::solidcore::topology::Brep>) -> Self {
        Self {
            bodies: vec![BodyInstance { body: brep, transforms: Vec::new() }],
            sketches: Vec::new(),
        }
    }

    fn from_sketch(output: Arc<SketchOutput>) -> Self {
        Self {
            bodies: Vec::new(),
            sketches: vec![SketchInstance { output, transforms: Vec::new() }],
        }
    }

    fn empty() -> Self {
        Self { bodies: Vec::new(), sketches: Vec::new() }
    }

    /// Bake placements into standalone BReps for operations that need actual
    /// geometry (Boolean, exact export).
    fn realized_solids(&self) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
        let mut solids = Vec::with_capacity(self.bodies.len());
        let mut copy_kernel = nodecraft_solidcore::kernel::kernel();
        for body in &self.bodies {
            let mut brep = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(
                &mut copy_kernel,
                &body.body,
            )
            .map_err(|error| error.to_string())?;
            for transform in &body.transforms {
                match transform {
                    BodyTransform::Place { offset, rotation } => apply_brep_placement(&mut brep, *offset, *rotation)?,
                    BodyTransform::Mirror { plane, offset } => {
                        let values = serde_json::json!({"plane": plane, "offset": offset});
                        brep = crate::node_task::execute_exact(
                            crate::node_task::ExactNodeTask::Mirror { brep, values },
                            "part-mirror",
                            fnv_seed("part-mirror"),
                        )?
                        .into_brep()
                        .ok_or_else(|| "mirror task returned no shape".to_owned())?;
                    }
                    BodyTransform::Scale { factor } => scale_brep(&mut brep, *factor)?,
                }
            }
            solids.push(brep);
        }
        Ok(solids)
    }
}

fn scale_brep(
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    factor: f64,
) -> Result<(), String> {
    use nodecraft_solidcore::solidcore::geom::axis2_placement::Axis2Placement;
    use nodecraft_solidcore::solidcore::math::{Point3, Vec3};
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    let placement = Axis2Placement::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    nodecraft_solidcore::solidcore::topology::brep::p5::transform(
        &mut kernel,
        brep,
        &placement,
        Some(Vec3::new(factor, factor, factor)),
        true,
    )
    .map_err(|error| error.to_string())
}

fn body_transforms(
    body: &nodecraft_solidcore::solidcore::topology::Brep,
    part: &Part,
    index: usize,
) -> Vec<BodyTransform> {
    let _ = body;
    part.bodies.get(index).map(|instance| instance.transforms.clone()).unwrap_or_default()
}

fn mesh_scale(mesh: &Mesh, factor: f64) -> Mesh {
    let mut scaled = mesh.clone();
    for vertex in &mut scaled.vertices {
        vertex[0] = (vertex[0] as f64 * factor) as f32;
        vertex[1] = (vertex[1] as f64 * factor) as f32;
        vertex[2] = (vertex[2] as f64 * factor) as f32;
    }
    scaled
}

fn mesh_placement(mesh: &Mesh, offset: [f64; 3], rotation: [f64; 3]) -> Mesh {
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let rotate = |point: [f64; 3]| -> [f64; 3] {
        let (mut x, mut y, mut z) = (point[0], point[1], point[2]);
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        (y, z) = (y * cx - z * sx, y * sx + z * cx);
        (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        (x, y) = (x * cz - y * sz, x * sz + y * cz);
        [x, y, z]
    };
    let mut transformed = mesh.clone();
    for vertex in &mut transformed.vertices {
        let point = rotate([vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]);
        vertex[0] = (point[0] + offset[0]) as f32;
        vertex[1] = (point[1] + offset[1]) as f32;
        vertex[2] = (point[2] + offset[2]) as f32;
    }
    for normal in &mut transformed.normals {
        let rotated = rotate([normal[0] as f64, normal[1] as f64, normal[2] as f64]);
        *normal = [rotated[0] as f32, rotated[1] as f32, rotated[2] as f32];
    }
    transformed
}

fn mesh_mirror(mesh: &Mesh, plane: &str, offset: f64) -> Mesh {
    let mut mirrored = mesh.clone();
    for vertex in &mut mirrored.vertices {
        match plane {
            "xz" => vertex[1] = (2.0 * offset - vertex[1] as f64) as f32,
            "yz" => vertex[0] = (2.0 * offset - vertex[0] as f64) as f32,
            _ => vertex[2] = (2.0 * offset - vertex[2] as f64) as f32,
        }
    }
    for normal in &mut mirrored.normals {
        match plane {
            "xz" => normal[1] = -normal[1],
            "yz" => normal[0] = -normal[0],
            _ => normal[2] = -normal[2],
        }
    }
    for face in &mut mirrored.faces { face.swap(1, 2); }
    mirrored
}

struct Eval<'req, 'prog> {
    request: &'req Request,
    kernel: ProceduralKernel,
    objects: HashMap<String, Mesh>,
    cache: EngineCache,
    /// Exact-B-rep import diagnostics per `stepimporter` node (debug response only).
    step_diagnostics: HashMap<String, StepDiagnostics>,
    errors: BTreeMap<String, String>,
    order: Vec<String>,
    timings: BTreeMap<String, f64>,
    brep_timings: BTreeMap<String, f64>,
    visiting: Vec<String>,
    cache_hits: u32,
    cache_misses: u32,
    breps: HashMap<String, Arc<nodecraft_solidcore::solidcore::topology::Brep>>,
    /// Exact Part results for this evaluation pass.
    parts: HashMap<String, Arc<Part>>,
    part_visiting: Vec<String>,
    /// Nodes whose own evaluation failed but whose output was replaced by a
    /// bypass input. Downstream nodes must not treat these as failed inputs.
    bypassed: HashSet<String>,
    /// Nodes whose exact BRep is an input to an exact consumer. Mesh-only
    /// primitive graphs skip the eager exact-BRep build.
    needs_brep: HashSet<String>,
    sketches: HashMap<String, Arc<SketchOutput>>,
    progress: Option<&'prog mut dyn FnMut(ProgressEvent)>,
}

impl<'req, 'prog> Eval<'req, 'prog> {
    fn new(request: &'req Request, progress: Option<&'prog mut dyn FnMut(ProgressEvent)>) -> Self {
        Self {
            request,
            kernel: ProceduralKernel,
            objects: HashMap::new(),
            cache: EngineCache::new(),
            step_diagnostics: HashMap::new(),
            errors: BTreeMap::new(),
            order: Vec::new(),
            timings: BTreeMap::new(),
            brep_timings: BTreeMap::new(),
            visiting: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            breps: HashMap::new(),
            parts: HashMap::new(),
            part_visiting: Vec::new(),
            bypassed: HashSet::new(),
            needs_brep: exact_brep_demands(request),
            sketches: HashMap::new(),
            progress,
        }
    }

    fn ensure_sketch_output(&mut self, node: &Node) -> Option<Arc<SketchOutput>> {
        if let Some(output) = self.sketches.get(&node.id) {
            return Some(output.clone());
        }
        let sketch: crate::sketch::Sketch = serde_json::from_value(
            node.values.get("sketch").cloned().unwrap_or(serde_json::Value::Null),
        )
        .ok()?;
        let output = Arc::new(SketchOutput::new(sketch));
        self.sketches.insert(node.id.clone(), output.clone());
        Some(output)
    }

    fn ensure_spline_output(&mut self, node: &Node) -> Option<Arc<SketchOutput>> {
        if let Some(output) = self.sketches.get(&node.id) {
            return Some(output.clone());
        }
        let sketch = spline_sketch_from_values(&node.values).ok()?;
        let output = Arc::new(SketchOutput::new(sketch));
        self.sketches.insert(node.id.clone(), output.clone());
        Some(output)
    }

    fn sketch_output_for(&mut self, start: &str) -> Option<SketchOutput> {
        let mut current = start.to_owned();
        let mut transform = SketchTransform::identity();
        for _ in 0..64 {
            let node = self.request.nodes.iter().find(|node| node.id == current)?;
            match node.node_type.as_str() {
                "translate" => {
                    let local = SketchTransform::from_values(&node.values);
                    transform = local.then(&transform);
                    current = self
                        .request
                        .connections
                        .iter()
                        .find(|link| link.to == node.id && (link.input == "shape" || link.input == "sketch"))?
                        .from
                        .clone();
                }
                "sketch" => return self.ensure_sketch_output(node).map(|output| output.with_transform(&transform)),
                "spline" => return self.ensure_spline_output(node).map(|output| output.with_transform(&transform)),
                _ => return None,
            }
        }
        None
    }

}

fn encode_vertices(values: &[[f32; 3]]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 12);
    for value in values {
        for component in value {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Small deterministic SplitMix64 generator for scatter placement. Keeping
/// this local avoids tying geometry layout to a library's PRNG implementation.
struct ScatterRng(u64);

impl ScatterRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

fn ancestors_of(request: &Request, target: &str) -> HashSet<String> {
    let mut seen = HashSet::new();
    let mut stack = vec![target.to_owned()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) { continue; }
        for link in request.connections.iter().filter(|link| link.to == id) {
            stack.push(link.from.clone());
        }
    }
    seen
}

/// Whether display meshes for this node type come from the exact Part path.
///
/// Part-path targets flatten their solids into one merged mesh with
/// `body:{index}:` names in Part body order. Callers that need to map display
/// bodies back to graph nodes must mirror this decision.
#[must_use]
pub fn part_display_supported(node_type: &str) -> bool {
    nodecraft_node_contract::part_display_supported(node_type)
}

/// Sketches are edited in the 2D overlay and must not produce a 3D display
/// body.
///
/// Earlier the engine classified `sketch` as a Part display target, and
/// `part_mesh_asset` tessellated its profile boundary curves into a display
/// mesh. That mesh was not the sketch overlay: the viewport showed it as a
/// stray flat/edge-like body behind the sketch. Downstream nodes consume the
/// typed `SketchOutput` directly, so no display mesh is needed.
///
/// If a future feature wants a 3D sketch preview, add it deliberately as an
/// overlay/ghost with its own opt-in, rather than re-enabling sketch as a Part
/// display target.
fn sketch_is_2d_overlay_only(node_type: &str) -> bool {
    nodecraft_node_contract::overlay_only(node_type)
}

fn part_supported(node_type: &str) -> bool {
    nodecraft_node_contract::part_display_supported(node_type)
}

fn part_display_targets(request: &Request) -> HashSet<String> {
    // Explicitly empty mesh targets are a batch/exact request, not a viewport
    // request. Preserve the established progress/cache behavior for them.
    if request.mesh_targets.as_ref().is_some_and(Vec::is_empty) {
        return HashSet::new();
    }
    let requested: HashSet<String> = request
        .targets
        .iter()
        .filter(|id| {
            request
                .nodes
                .iter()
                .find(|node| node.id == **id)
                .is_some_and(|node| {
                    !sketch_is_2d_overlay_only(&node.node_type)
                        && part_supported(&node.node_type)
                })
                && ancestors_of(request, id).iter().all(|ancestor| {
                    request
                        .nodes
                        .iter()
                        .find(|node| node.id == *ancestor)
                        .is_some_and(|node| part_supported(&node.node_type))
                })
        })
        .cloned()
        .collect();
    // Include the ancestors so they still emit progress/timings even though
    // only the display target is tessellated.
    let mut targets = requested.clone();
    for target in &requested {
        targets.extend(ancestors_of(request, target));
    }
    targets
}

fn evaluation_order_for_targets(request: &Request) -> Vec<String> {
    use std::collections::HashSet;
    let ids: HashSet<&str> = request.nodes.iter().map(|node| node.id.as_str()).collect();
    let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
    for link in &request.connections {
        if ids.contains(link.from.as_str()) && ids.contains(link.to.as_str()) {
            dependencies.entry(link.to.clone()).or_default().push(link.from.clone());
        }
    }
    fn visit(
        id: &str,
        dependencies: &HashMap<String, Vec<String>>,
        seen: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if !seen.insert(id.to_string()) { return; }
        if let Some(inputs) = dependencies.get(id) {
            for input in inputs { visit(input, dependencies, seen, order); }
        }
        order.push(id.to_string());
    }
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for target in &request.targets {
        if ids.contains(target.as_str()) { visit(target, &dependencies, &mut seen, &mut order); }
    }
    order
}

#[cfg(test)]
fn diagnostics_requested(request: &Request) -> bool {
    let env_enabled = std::env::var("NODECRAFT_WORKER_DIAGNOSTICS")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    env_enabled
        || request.nodes.iter().any(|node| {
            node.node_type == "stepimporter"
                && node.values.get("diagnostics").and_then(serde_json::Value::as_bool) == Some(true)
        })
}

#[cfg(test)]
fn respond(
    request: Request,
    cache: EngineCache,
) -> (Response, EngineCache, HashMap<String, StepDiagnostics>) {
    let output = evaluate(request, cache, None);
    (output.response, output.cache, output.diagnostics)
}

fn comparison_cache_key(request: &Request, targets: &[String]) -> Option<(String, String)> {
    let key = |id: &str| {
        let node = node(request, id)?;
        if !node_cache_is_fresh(node) {
            return None;
        }
        let hash = node.cache.as_ref()?.key.as_ref()?;
        Some(format!("{id}@{hash}"))
    };
    Some((key(targets.first()?)?, key(targets.get(1)?)?))
}

pub fn evaluate(
    request: Request,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
) -> EngineOutput {
    let started = Instant::now();
    struct DisplayToleranceReset;
    impl Drop for DisplayToleranceReset {
        fn drop(&mut self) {
            nodecraft_mesh::set_display_tolerance(None);
        }
    }
    let _tolerance_reset = DisplayToleranceReset;
    nodecraft_mesh::set_display_tolerance(request.mesh_tolerance);
    let (request_id, tab_id, graph_revision) = (request.request_id, request.tab_id.clone(), request.graph_revision.clone());
    let mut eval = Eval::new(&request, progress);
    eval.cache = cache;
    eval.cache.ensure_part_capacity(request.nodes.len());
    eval.prewarm_primitives();
    // Evaluate the target closure once in topological order. Downstream nodes only
    // consume already-computed outputs; no branch/target can re-enter a node.
    let part_targets = part_display_targets(&request);
    let mesh_targets: HashSet<String> = request
        .mesh_targets
        .as_deref()
        .unwrap_or(&request.targets)
        .iter()
        .cloned()
        .collect();
    // Nodes needed by a mesh-only target must still produce a Mesh even when
    // they are also ancestors of a Part target.
    let mut mesh_required = HashSet::new();
    for target in &request.targets {
        if part_targets.contains(target) {
            continue;
        }
        mesh_required.extend(ancestors_of(&request, target));
    }
    for node_id in evaluation_order_for_targets(&request) {
        if mesh_required.contains(&node_id) {
            eval.evaluate(&node_id);
        } else if part_targets.contains(&node_id) {
            if mesh_targets.contains(&node_id) {
                eval.evaluate_part_display(&node_id);
            } else {
                eval.evaluate_part_only(&node_id);
            }
        } else {
            eval.evaluate(&node_id);
        }
    }
    let mesh_targets = request
        .mesh_targets
        .clone()
        .unwrap_or_else(|| request.targets.clone());
    let mut mesh_payloads = BTreeMap::new();
    let mut empty_targets = Vec::new();
    for id in &mesh_targets {
        if !request.targets.contains(id) {
            continue; // Python only meshes evaluated target nodes.
        }
        if node(&request, id).is_some_and(|node| sketch_is_2d_overlay_only(&node.node_type)) {
            // See `sketch_is_2d_overlay_only`: the overlay owns sketch drawing.
            // Suppress the payload even if another evaluation path produced one,
            // so a stray sketch body can never reach the 3D viewport.
            empty_targets.push(id.clone());
            continue;
        }
        match eval.objects.get(id) {
            Some(mesh) => {
                let key = node(&request, id).and_then(|node| node.cache.as_ref()?.key.clone());
                let mut mesh = mesh.clone();
                if request.binary_mesh {
                    mesh.vertices_data = Some(encode_vertices(&mesh.vertices));
                    mesh.normals_data = Some(encode_vertices(&mesh.normals));
                    mesh.vertices.clear();
                    mesh.normals.clear();
                }
                mesh_payloads.insert(id.clone(), MeshPayload { key, mesh });
            }
            None => empty_targets.push(id.clone()),
        }
    }
    let comparison = request
        .comparison_targets
        .as_ref()
        .filter(|targets| targets.len() == 2)
        .and_then(|targets| {
            // Skip the kernel difference when the display payloads already prove
            // there is no material change. This also avoids coincident-solid
            // booleans for pass-through features.
            let before_display = mesh_payloads.get(&targets[0])?.mesh.clone();
            let after_display = mesh_payloads.get(&targets[1])?.mesh.clone();
            let before_volume = comparison::mesh_volume(&before_display);
            let after_volume = comparison::mesh_volume(&after_display);
            if (before_volume - after_volume).abs() <= before_volume.abs().max(1.0) * 1.0e-6 {
                return None;
            }
            let before = eval.evaluate_part(&targets[0]).ok()?;
            let after = eval.evaluate_part(&targets[1]).ok()?;
            let cache_key = comparison_cache_key(&request, targets);
            let difference = if let Some(key) = &cache_key {
                if let Some(cached) = eval.cache.difference(key) {
                    eval.cache.note_difference_hit();
                    Ok(cached)
                } else {
                    let result = compare_parts(&before, &after);
                    if let Ok(ref difference) = result {
                        eval.cache.insert_difference(key.clone(), difference.clone());
                    }
                    result
                }
            } else {
                compare_parts(&before, &after)
            };
            Some((difference, before_display, after_display))
        })
        .map(|(difference, before_mesh, after_mesh)| match difference {
            Ok(difference) => nodecraft_protocol::ComparisonPayload {
                added: difference.added,
                removed: difference.removed,
                approximate: false,
                error: None,
            },
            Err(error) => {
                // The exact surface intersection is not ported for every
                // curved interface. Keep the feature ghost useful by falling
                // back to a display-surface difference, while reporting that it
                // is approximate.
                let approximate = comparison::surface_difference(&before_mesh, &after_mesh);
                nodecraft_protocol::ComparisonPayload {
                    added: approximate.added,
                    removed: approximate.removed,
                    approximate: true,
                    error: Some(format!("exact difference unavailable: {error}")),
                }
            }
        });
    let response = Response {
        request_id,
        tab_id,
        graph_revision,
        meshes: mesh_payloads,
        empty_targets,
        node_errors: eval.errors,
        evaluation_order: eval.order,
        timings: eval.timings,
        brep_timings: eval.brep_timings,
        evaluation_ms: started.elapsed().as_secs_f64() * 1000.0,
        cache_stats: Some(CacheStats { hits: eval.cache_hits, misses: eval.cache_misses }),
        comparison,
        engine: "solidcore-experiment".into(),
    };
    nodecraft_mesh::set_display_tolerance(None);
    EngineOutput { response, cache: eval.cache, diagnostics: eval.step_diagnostics }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn respond_with_progress(
        request: Request,
        cache: EngineCache,
        progress_tx: Option<mpsc::Sender<Vec<u8>>>,
    ) -> (Response, EngineCache, HashMap<String, StepDiagnostics>) {
        let mut callback = |event: ProgressEvent| {
            let Some(tx) = &progress_tx else { return; };
            let frame = serde_json::json!({
                "type": "evaluation-progress",
                "requestId": event.request_id,
                "tabId": event.tab_id,
                "graphRevision": event.graph_revision,
                "nodeId": event.node_id,
                "phase": event.phase.as_str(),
            });
            let _ = tx.send(format!("{frame}\n").into_bytes());
        };
        let output = evaluate(request, cache, Some(&mut callback));
        (output.response, output.cache, output.diagnostics)
    }
    use nodecraft_mesh::StepImport;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    #[test]
    fn sketch_transform_space_does_not_change_geometry() {
        let world = SketchTransform::from_values(&serde_json::json!({
            "space": "world", "x": 10.0, "y": 2.0, "z": 3.0, "rz": 90.0
        }));
        let local = SketchTransform::from_values(&serde_json::json!({
            "space": "local", "x": 10.0, "y": 2.0, "z": 3.0, "rz": 90.0
        }));
        assert_eq!(world.apply([1.0, 0.0, 0.0]), local.apply([1.0, 0.0, 0.0]));
        assert_eq!(world.apply([0.0, 0.0, 0.0]), [10.0, 2.0, 3.0]);
    }

    const BLOCK: &str = "tests/Sample AP203 Data Files/block.stp";
    const DOGHOUSE: &str = "tests/Sample AP203 Data Files/doghouse.stp";

    fn git_revision() -> String {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let hash = std::process::Command::new("git")
            .arg("-C").arg(&root).args(["rev-parse", "--short", "HEAD"])
            .output().ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|hash| !hash.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let dirty = std::process::Command::new("git")
            .arg("-C").arg(&root).args(["status", "--porcelain", "--untracked-files=no", "--", "rust/crates", "rust/Cargo.toml", "rust/Cargo.lock", ":(exclude)rust/crates/nodecraft-engine/testdata"])
            .output().ok()
            .is_some_and(|output| output.status.success() && !output.stdout.is_empty());
        if dirty { format!("{hash}+") } else { hash }
    }

    fn solidcore_revision() -> String {
        let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
        std::fs::read_to_string(lock).ok()
            .and_then(|text| text.lines().find(|line| line.contains("solidcore?branch=main#")).map(str::to_string))
            .and_then(|line| line.split_once('#').map(|(_, hash)| hash.trim_matches('"').chars().take(12).collect::<String>()))
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn record_execution_time(test: &str, evaluation_ms: f64) {
        // Debug builds are for correctness only; release builds check speed.
        if cfg!(debug_assertions) { return; }
        use std::io::Write as _;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata");
        std::fs::create_dir_all(&dir).expect("create execution log dir");
        let path = dir.join("train-execution-times.jsonl");
        let previous = std::fs::read_to_string(&path).unwrap_or_default();
        let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
        let last_passing_ms = previous
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|entry| entry["passed"].as_bool() == Some(true))
            .filter(|entry| entry["profile"].as_str() == Some(profile))
            .filter_map(|entry| entry["evaluationMs"].as_f64())
            .last();
        // Release best-of-three still has ~1% scheduler noise; allow 2%.
        const NOISE_TOLERANCE: f64 = 0.02;
        let current_ms = evaluation_ms.round() as u64;
        let slower = last_passing_ms.is_some_and(|last| evaluation_ms > last * (1.0 + NOISE_TOLERANCE));
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let entry = serde_json::json!({
            "test": test,
            "timestampUnix": seconds,
            "profile": profile,
            "gitHash": git_revision(),
            "solidcoreHash": solidcore_revision(),
            "evaluationMs": evaluation_ms,
            "previousPassingMs": last_passing_ms,
            "passed": !slower,
        });
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path).expect("open execution log");
        writeln!(file, "{}", serde_json::to_string(&entry).expect("execution log json")).expect("append execution log");
        assert!(!slower, "{test}: execution {current_ms}ms is more than 2% slower than previous {last_passing_ms:?}ms");
    }

    fn fixture(relative: &str) -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative)
            .to_string_lossy()
            .into_owned()
    }

    fn import_fixture(relative: &str) -> StepImport {
        import_step(&fixture(relative)).unwrap_or_else(|error| panic!("{relative}: {error}"))
    }

    fn unique_groups(mesh: &Mesh) -> BTreeSet<&str> {
        mesh.face_groups.iter().map(String::as_str).collect()
    }

    fn bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut any = false;
        for triangle in &mesh.faces {
            for index in triangle {
                let Some(vertex) = mesh.vertices.get(*index as usize) else { continue };
                any = true;
                for axis in 0..3 {
                    min[axis] = min[axis].min(vertex[axis]);
                    max[axis] = max[axis].max(vertex[axis]);
                }
            }
        }
        if !any {
            for vertex in &mesh.vertices {
                for axis in 0..3 {
                    min[axis] = min[axis].min(vertex[axis]);
                    max[axis] = max[axis].max(vertex[axis]);
                }
            }
        }
        (min, max)
    }

    fn assert_bounds(mesh: &Mesh, expected_min: [f32; 3], expected_max: [f32; 3], tolerance: f32) {
        let (min, max) = bounds(mesh);
        for axis in 0..3 {
            assert!(
                (min[axis] - expected_min[axis]).abs() <= tolerance,
                "min[{axis}] = {} != {}",
                min[axis],
                expected_min[axis]
            );
            assert!(
                (max[axis] - expected_max[axis]).abs() <= tolerance,
                "max[{axis}] = {} != {}",
                max[axis],
                expected_max[axis]
            );
        }
    }

    #[test]
    fn evaluates_chamfer_node_from_blend_preview() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 404,
            "targets": ["chamfer"],
            "meshTargets": ["chamfer"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                {"id": "chamfer", "type": "chamfer", "values": {"mode": "chamfer", "length": 2, "edges": [{"ordinal": 0}]}}
            ],
            "connections": [{"from": "box", "to": "chamfer", "in": "shape", "out": "shape"}]
        })).expect("chamfer request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("chamfer").expect("chamfer mesh").mesh;
        assert!(!mesh.faces.is_empty(), "chamfer preview should have faces");
        assert!(!mesh.faces.is_empty(), "exact chamfer should produce faces");
    }

    #[test]
    fn time_node_is_a_non_geometric_source() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 406,
            "targets": ["time"],
            "meshTargets": ["time"],
            "binaryMesh": false,
            "nodes": [
                {"id": "time", "type": "time", "values": {"time": 0.25, "speed": 1, "amplitude": 10}}
            ],
            "connections": []
        })).expect("time request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("time").expect("time mesh").mesh;
        assert!(mesh.faces.is_empty());
    }

    #[test]
    fn scatter_uses_float_values_and_seeded_random_positions() {
        fn request(seed: f64) -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 407,
                "targets": ["scatter"],
                "meshTargets": ["scatter"],
                "binaryMesh": false,
                "nodes": [
                    {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 2, "width": 2, "height": 2}},
                    {"id": "scatter", "type": "scatter", "values": {
                        "count": 16.0, "spanX": 200.0, "spanY": 200.0,
                        "seed": seed, "distribution": "random",
                        "minScale": 0.5, "maxScale": 1.5
                    }}
                ],
                "connections": [{"from": "box", "to": "scatter", "in": "shape", "out": "shape"}]
            })).expect("scatter request json")
        }
        let first = evaluate(request(7.0), EngineCache::new(), None);
        let second = evaluate(request(8.0), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        let first = &first.response.meshes.get("scatter").expect("scatter mesh").mesh;
        let second = &second.response.meshes.get("scatter").expect("scatter mesh").mesh;
        assert!(!first.vertices.is_empty());
        assert_ne!(first.vertices[0], second.vertices[0], "seed must change scatter positions");
    }

    #[test]
    fn edge_node_without_selection_passes_input_through() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 407,
            "targets": ["c"],
            "meshTargets": ["c"],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "b"}},
                {"id": "c", "type": "chamfer", "values": {"length": 1}, "cache": {"key": "c"}}
            ],
            "connections": [{"from": "b", "to": "c", "in": "shape", "out": "shape"}]
        })).expect("chamfer request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("c").expect("chamfer mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert!(!mesh.face_groups.iter().any(|group| group.starts_with("blend:")), "preview should wait for an edge selection");
    }

    #[test]
    fn chained_chamfers_evaluate_exactly() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 410,
            "targets": ["c2"],
            "meshTargets": ["c2"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "p"}},
                {"id": "c1", "type": "chamfer", "values": {"chamferType": "equal", "length": 2, "edges": [{"id": "edge:0"}]}, "cache": {"key": "c1"}},
                {"id": "c2", "type": "chamfer", "values": {"chamferType": "equal", "length": 2, "edges": [{"id": "edge:1"}]}, "cache": {"key": "c2"}}
            ],
            "connections": [
                {"from": "p", "to": "c1", "in": "shape", "out": "shape"},
                {"from": "c1", "to": "c2", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(!output.response.meshes["c2"].mesh.faces.is_empty());
    }

    #[test]
    fn chamfer_preview_user_distance_angle_selection() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 409,
            "targets": ["chamfer"],
            "meshTargets": ["chamfer"],
            "nodes": [
                {"id": "primitive", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "primitive"}},
                {"id": "chamfer", "type": "chamfer", "values": {"angle": 72.5, "chamferType": "distance-angle", "distance2": 3.74, "edges": [{"id": "edge:9"},{"id": "edge:3"},{"id": "edge:1"},{"id": "edge:0"}], "length": 2.61, "mode": "chamfer"}, "cache": {"key": "chamfer"}}
            ],
            "connections": [{"from": "primitive", "to": "chamfer", "in": "shape", "out": "shape"}]
        })).expect("request");
        let started = std::time::Instant::now();
        let output = evaluate(request, EngineCache::new(), None);
        let _ = started;
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes["chamfer"].mesh;
        assert!(!mesh.faces.is_empty());
        assert!(mesh.brep_edges.iter().any(|edge| edge.kind == "edge"));
    }

    #[test]
    fn chamfer_preview_accepts_body_scoped_edge_ids() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 408,
            "targets": ["c"],
            "meshTargets": ["c"],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "b"}},
                {"id": "c", "type": "chamfer", "values": {"length": 1, "edges": [{"id": "body:0:edge:0"}]}, "cache": {"key": "c"}}
            ],
            "connections": [{"from": "b", "to": "c", "in": "shape", "out": "shape"}]
        })).expect("chamfer request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("c").expect("chamfer mesh").mesh;
        assert!(!mesh.faces.is_empty(), "exact chamfer should produce faces");
    }

    #[test]
    fn chamfer_preview_accepts_a_boolean_input() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 406,
            "targets": ["c"],
            "meshTargets": ["c"],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 20}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}},
                {"id": "c", "type": "chamfer", "values": {"length": 1, "edges": [{"ordinal": 0}]}, "cache": {"key": "c"}}
            ],
            "connections": [
                {"from": "a", "to": "u", "in": "a", "out": "shape"},
                {"from": "b", "to": "u", "in": "b", "out": "shape"},
                {"from": "u", "to": "c", "in": "shape", "out": "shape"}
            ]
        })).expect("chamfer request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("c").expect("chamfer mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert!(!mesh.faces.is_empty(), "exact chamfer should produce faces");
    }

    #[test]
    fn evaluates_fillet_node_from_blend_preview() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 405,
            "targets": ["fillet"],
            "meshTargets": ["fillet"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                {"id": "fillet", "type": "fillet", "values": {"continuity": "", "length": 2, "edges": [{"ordinal": 0}]}}
            ],
            "connections": [{"from": "box", "to": "fillet", "in": "shape", "out": "shape"}]
        })).expect("fillet request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("fillet").expect("fillet mesh").mesh;
        assert!(!mesh.faces.is_empty(), "fillet preview should have faces");
        assert!(!mesh.faces.is_empty(), "exact fillet should have faces");
    }

    #[test]
    fn look_attaches_colour_to_every_face() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 407,
            "targets": ["look"],
            "meshTargets": ["look"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "look", "type": "look", "values": {"color": "#ff0000"}}
            ],
            "connections": [{"from": "box", "to": "look", "in": "shape", "out": "shape"}]
        })).expect("look request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("look").expect("look mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        for color in &mesh.face_colors {
            assert!((color[0] - 1.0).abs() < 1.0e-5, "{color:?}");
            assert!(color[1].abs() < 1.0e-5, "{color:?}");
            assert!(color[2].abs() < 1.0e-5, "{color:?}");
        }
    }

    #[test]
    fn cylinder_join_picks_resolve_near_the_saved_centres() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 412,
            "targets": ["boolean"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [
                {"id": "solid", "type": "solid", "values": {"height": 100.0, "primitiveType": "cylinder"}},
                {"id": "solid-copy1", "type": "solid", "values": {"height": 50.0, "primitiveType": "cylinder", "radius": 18.9}},
                {"id": "translate", "type": "translate", "values": {"rx": 0.0, "ry": -115.0, "rz": 0.0, "space": "world", "x": 24.0, "y": 0.0, "z": 8.0}},
                {"id": "boolean", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [
                {"from": "solid-copy1", "to": "translate", "in": "shape", "out": "shape"},
                {"from": "translate", "to": "boolean", "in": "a", "out": "shape"},
                {"from": "solid", "to": "boolean", "in": "b", "out": "shape"}
            ]
        })).expect("cylinder join json");
        let mut eval = Eval::new(&request, None);
        let part = eval.evaluate_part("boolean").expect("boolean part");
        let brep = part.bodies[0].body.as_ref();
        let kernel = nodecraft_solidcore::kernel::kernel();
        let values = serde_json::json!({"edges": [
            {"center": [14.100593090057371, -11.389926261372036, 4.566759937339359], "id": "edge:1"},
            {"center": [14.100598600175644, 11.389923453330994, 4.5666884283224745], "id": "edge:2"}
        ]});
        let selections = nodecraft_blend::select_edges_from_values_center_first(&kernel, brep, &values)
            .expect("centre-first selections");
        assert_eq!(selections.len(), 2);
        let ids: std::collections::BTreeSet<u32> = selections.iter().map(|selection| selection.edge.index()).collect();
        assert_eq!(ids, [4, 5].into_iter().collect(), "selected the wrong cylinder edges: {ids:?}");
    }

    #[test]
    fn exact_feature_comparison_is_reused_from_cache() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 997,
            "targets": ["box", "chamfer"],
            "meshTargets": ["box", "chamfer"],
            "comparisonTargets": ["box", "chamfer"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "box-v1", "state": "cached"}},
                {"id": "chamfer", "type": "chamfer", "values": {"length": 2.0, "edges": [{"ordinal": 0}]}, "cache": {"key": "chamfer-v1", "state": "cached"}}
            ],
            "connections": [{"from": "box", "to": "chamfer", "in": "shape", "out": "shape"}]
        })).expect("comparison cache json");
        let first = evaluate(request.clone(), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let payload = first.response.comparison.expect("comparison payload");
        assert!(!payload.approximate, "planar chamfer should compare exactly: {:?}", payload.error);
        assert_eq!(first.cache.difference_entries(), 1);
        let second = evaluate(request, first.cache, None);
        assert!(
            second.cache.difference_hits() > 0,
            "exact comparison was not reused from cache"
        );
        let second_payload = second.response.comparison.expect("cached comparison payload");
        assert_eq!(second_payload.added.faces.len(), payload.added.faces.len());
        assert_eq!(second_payload.removed.faces.len(), payload.removed.faces.len());
    }

    #[test]
    fn dirty_comparison_targets_bypass_the_exact_difference_cache() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 996,
            "targets": ["box", "chamfer"],
            "meshTargets": ["box", "chamfer"],
            "comparisonTargets": ["box", "chamfer"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "box-v2", "state": "dirty"}},
                {"id": "chamfer", "type": "chamfer", "values": {"length": 2.0, "edges": [{"ordinal": 0}]}, "cache": {"key": "chamfer-v2", "state": "dirty"}}
            ],
            "connections": [{"from": "box", "to": "chamfer", "in": "shape", "out": "shape"}]
        })).expect("dirty comparison json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(
            output.cache.difference_entries(),
            0,
            "dirty comparison targets must not enter the exact-comparison cache"
        );
    }

    #[test]
    fn g2_fillet_comparison_stays_on_the_fillet_area() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 998,
            "targets": ["fillet", "solid"],
            "meshTargets": ["fillet", "solid"],
            "comparisonTargets": ["solid", "fillet"],
            "binaryMesh": false,
            "nodes": [
                {"id": "solid", "type": "solid", "values": {"height": 100.0, "primitiveType": "cylinder"}},
                {"id": "fillet", "type": "fillet", "values": {"continuity": "G2", "length": 19.2, "edges": [{"center": [-20.0, 2.4492935992912173e-15, 50.0], "id": "edge:2"}]}}
            ],
            "connections": [{"from": "solid", "to": "fillet", "in": "shape", "out": "shape"}]
        })).expect("g2 fillet comparison json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(
            output.cache.difference_entries(),
            0,
            "approximate display fallbacks must not enter the exact-comparison cache"
        );
        let comparison = output.response.comparison.expect("g2 comparison payload");
        assert_eq!(
            comparison.approximate,
            comparison.error.is_some(),
            "approximate flag and exact-difference error disagree: {}",
            comparison.error.as_deref().unwrap_or("no error")
        );
        assert!(
            !comparison.removed.faces.is_empty() || !comparison.added.faces.is_empty(),
            "g2 comparison has no ghost geometry: {:?}",
            comparison.error
        );
        let min_z = comparison
            .removed
            .vertices
            .iter()
            .map(|vertex| vertex[2])
            .fold(f32::INFINITY, f32::min);
        assert!(min_z > 1.0, "ghost covers the whole cylinder: min z {min_z}");
    }

    #[test]
    fn failed_node_stays_failed_when_its_bypass_mesh_is_reused() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 993,
            "targets": ["fillet"],
            "meshTargets": ["fillet"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "fillet", "type": "fillet", "values": {"length": 2.0, "edges": [{"ordinal": 99}]}}
            ],
            "connections": [{"from": "box", "to": "fillet", "in": "shape", "out": "shape"}]
        })).expect("failed fillet json");
        let first = evaluate(request.clone(), EngineCache::new(), None);
        assert!(first.response.node_errors.contains_key("fillet"), "{:?}", first.response.node_errors);
        let second = evaluate(request, first.cache, None);
        assert!(
            second.response.node_errors.contains_key("fillet"),
            "failed node lost its error on a cached evaluation: {:?}",
            second.response.node_errors
        );
    }

    #[test]
    fn cylinders_two_edge_fillet_evaluates() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 411,
            "targets": ["fillet"],
            "meshTargets": ["fillet"],
            "binaryMesh": false,
            "nodes": [
                {"id": "solid", "type": "solid", "values": {"height": 100.0, "primitiveType": "cylinder"}},
                {"id": "solid-copy1", "type": "solid", "values": {"height": 50.0, "primitiveType": "cylinder", "radius": 18.9}},
                {"id": "translate", "type": "translate", "values": {"rx": 0.0, "ry": -115.0, "rz": 0.0, "space": "world", "x": 24.0, "y": 0.0, "z": 8.0}},
                {"id": "boolean", "type": "boolean", "values": {"mode": "union"}},
                {"id": "fillet", "type": "fillet", "values": {"continuity": "", "length": 8.3, "edges": [
                    {"center": [14.100593090057371, -11.389926261372036, 4.566759937339359], "id": "edge:1"},
                    {"center": [14.100598600175644, 11.389923453330994, 4.5666884283224745], "id": "edge:2"}
                ]}}
            ],
            "connections": [
                {"from": "solid-copy1", "to": "translate", "in": "shape", "out": "shape"},
                {"from": "translate", "to": "boolean", "in": "a", "out": "shape"},
                {"from": "solid", "to": "boolean", "in": "b", "out": "shape"},
                {"from": "boolean", "to": "fillet", "in": "shape", "out": "shape"}
            ]
        })).expect("cylinders fillet json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:#?}", output.response.node_errors);
    }

    #[test]
    fn look_after_fillet_colours_every_face() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 410,
            "targets": ["look"],
            "meshTargets": ["look"],
            "binaryMesh": false,
            "nodes": [
                {"id": "solid", "type": "solid", "values": {"primitiveType": "box"}},
                {"id": "solid-copy1", "type": "solid", "values": {"primitiveType": "cylinder"}},
                {"id": "translate", "type": "translate", "values": {"x": 24.0, "y": 0.0, "z": 10.0, "ry": -55.0}},
                {"id": "boolean", "type": "boolean", "values": {"mode": "union"}},
                {"id": "fillet", "type": "fillet", "values": {"continuity": "G2", "length": 13.0, "edges": [{"id": "edge:11"}]}},
                {"id": "look", "type": "look", "values": {"color": "#002440"}}
            ],
            "connections": [
                {"from": "solid-copy1", "to": "translate", "in": "shape", "out": "shape"},
                {"from": "translate", "to": "boolean", "in": "a", "out": "shape"},
                {"from": "solid", "to": "boolean", "in": "b", "out": "shape"},
                {"from": "boolean", "to": "fillet", "in": "shape", "out": "shape"},
                {"from": "fillet", "to": "look", "in": "shape", "out": "shape"}
            ]
        })).expect("look after fillet request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("look").expect("look mesh").mesh;
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        let unmatched = mesh
            .face_colors
            .iter()
            .filter(|color| !(color[0] < 0.01 && color[1] > 0.10 && color[2] > 0.20))
            .count();
        assert_eq!(unmatched, 0, "{unmatched} of {} faces did not receive the Look colour", mesh.faces.len());
    }

    #[test]
    fn look_colours_survive_a_fillet() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 409,
            "targets": ["fillet"],
            "meshTargets": ["fillet"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                {"id": "look", "type": "look", "values": {"color": "#ff0000"}},
                {"id": "fillet", "type": "fillet", "values": {"continuity": "", "length": 2, "edges": [{"ordinal": 0}]}}
            ],
            "connections": [
                {"from": "box", "to": "look", "in": "shape", "out": "shape"},
                {"from": "look", "to": "fillet", "in": "shape", "out": "shape"}
            ]
        })).expect("fillet look request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("fillet").expect("fillet mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        assert!(mesh.face_colors.iter().any(|color| color[0] > 0.5), "look colour was lost by the fillet");
    }

    #[test]
    fn look_colours_survive_a_boolean() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 408,
            "targets": ["boolean"],
            "meshTargets": ["boolean"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "x": -15}},
                {"id": "sphere", "type": "primitive", "values": {"primitiveType": "sphere", "x": 15}},
                {"id": "red", "type": "look", "values": {"color": "#ff0000"}},
                {"id": "blue", "type": "look", "values": {"color": "#0000ff"}},
                {"id": "boolean", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [
                {"from": "box", "to": "red", "in": "shape", "out": "shape"},
                {"from": "sphere", "to": "blue", "in": "shape", "out": "shape"},
                {"from": "red", "to": "boolean", "in": "a", "out": "shape"},
                {"from": "blue", "to": "boolean", "in": "b", "out": "shape"}
            ]
        })).expect("boolean look request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.response.meshes.get("boolean").expect("boolean mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        assert!(mesh.face_colors.iter().any(|color| color != &[0.32, 0.34, 0.37]), "no look colours survived");
    }

    #[test]
    fn evaluates_g2_and_g3_fillet_nodes() {
        for continuity in ["G2", "G3"] {
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 406,
                "targets": ["fillet"],
                "meshTargets": ["fillet"],
                "binaryMesh": false,
                "nodes": [
                    {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 30, "height": 20}},
                    {"id": "fillet", "type": "fillet", "values": {"continuity": continuity, "length": 2, "edges": [{"ordinal": 0}]}}
                ],
                "connections": [{"from": "box", "to": "fillet", "in": "shape", "out": "shape"}]
            })).expect("fillet request json");
            let output = evaluate(request, EngineCache::new(), None);
            assert!(output.response.node_errors.is_empty(), "{continuity}: {:?}", output.response.node_errors);
            let mesh = &output.response.meshes.get("fillet").expect("fillet mesh").mesh;
            assert!(!mesh.faces.is_empty(), "{continuity} fillet should have faces");
        }
    }

    #[test]
    fn train_graph_execution_regression() {
        let sample = std::fs::read_to_string(fixture("samples/train.nodecraft")).expect("train sample");
        let sample: serde_json::Value = serde_json::from_str(&sample).expect("train sample json");
        let nodes: Vec<serde_json::Value> = sample["nodes"]
            .as_array()
            .expect("train nodes")
            .iter()
            .map(|node| serde_json::json!({
                "id": node["id"],
                "type": node["type"],
                "values": node.get("values").cloned().unwrap_or(serde_json::json!({})),
                "suppressed": node.get("suppressed").and_then(serde_json::Value::as_bool).unwrap_or(false),
                "cache": node.get("cache").cloned().unwrap_or(serde_json::Value::Null),
            }))
            .collect();
        let targets: Vec<String> = nodes.iter().map(|node| node["id"].as_str().unwrap_or_default().to_string()).collect();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 303,
            "progress": true,
            "targets": targets,
            "meshTargets": ["n69"],
            "binaryMesh": false,
            "nodes": nodes,
            "connections": sample["connections"].clone(),
        })).expect("train execution json");
        let mut starts = Vec::new();
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut progress = |event: ProgressEvent| {
            if event.phase == ProgressPhase::Start { starts.push(event.node_id); }
            let _ = tx.send(());
        };
        let output = evaluate(request.clone(), EngineCache::new(), Some(&mut progress));
        let response = &output.response;
        assert!(response.evaluation_ms > 0.0);
        assert_eq!(response.timings.len(), 76, "every train node should be accounted for");
        let expected_errors = ["n76"];
        let mut actual_errors: Vec<&str> = response.node_errors.keys().map(String::as_str).collect();
        actual_errors.sort_unstable();
        assert_eq!(actual_errors, expected_errors);
        assert!(!response.node_errors.contains_key("n68"));
        assert!(!response.node_errors.contains_key("n72"));
        assert!(!response.node_errors.contains_key("n74"));
        assert!(!response.node_errors.contains_key("n75"));
        assert!(starts.len() == 76 && starts.iter().collect::<std::collections::HashSet<_>>().len() == 76);
        let mesh = &output.response.meshes.get("n69").expect("n69 mesh").mesh;
        assert!(!mesh.faces.is_empty(), "n69 should produce geometry");
        assert!(!mesh.vertices.is_empty());
        let mut execution_ms = response.evaluation_ms;
        if !cfg!(debug_assertions) {
            // Best of three fresh runs removes scheduler/thermal jitter from the
            // release-only speed gate.
            for _ in 0..2 {
                let sample = evaluate(request.clone(), EngineCache::new(), None);
                execution_ms = execution_ms.min(sample.response.evaluation_ms);
            }
        }
        record_execution_time("train_graph_execution_regression", execution_ms);
    }

    #[test]
    #[ignore = "slow full train graph integration test"]
    fn train_graph_emits_one_start_and_stop_per_processed_node() {
        let sample = std::fs::read_to_string(fixture("samples/train.nodecraft")).expect("train sample");
        let sample: serde_json::Value = serde_json::from_str(&sample).expect("train sample json");
        let nodes: Vec<serde_json::Value> = sample["nodes"]
            .as_array()
            .expect("train nodes")
            .iter()
            .map(|node| serde_json::json!({
                "id": node["id"],
                "type": node["type"],
                "values": node.get("values").cloned().unwrap_or(serde_json::json!({})),
                "suppressed": node.get("suppressed").and_then(serde_json::Value::as_bool).unwrap_or(false),
                "cache": node.get("cache").cloned().unwrap_or(serde_json::Value::Null),
            }))
            .collect();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 77,
            "progress": true,
            "targets": ["n76"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": nodes,
            "connections": sample["connections"].clone(),
        })).expect("train request json");
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) = respond_with_progress(request, EngineCache::new(), Some(progress_tx));
        let frames: Vec<serde_json::Value> = progress_rx
            .iter()
            .map(|bytes| serde_json::from_slice(&bytes).expect("progress frame json"))
            .collect();
        assert!(!frames.is_empty(), "engine emitted no progress frames");
        let mut starts = std::collections::HashMap::new();
        let mut stops = std::collections::HashMap::new();
        for (index, frame) in frames.iter().enumerate() {
            let node = frame["nodeId"].as_str().expect("progress node id").to_string();
            match frame["phase"].as_str().expect("progress phase") {
                "node-start" => {
                    assert!(starts.insert(node.clone(), index).is_none(), "duplicate start for {node}");
                }
                "node-complete" => {
                    assert!(stops.insert(node.clone(), index).is_none(), "duplicate stop for {node}");
                }
                phase => panic!("unexpected progress phase {phase:?}"),
            }
        }
        assert_eq!(starts.len(), stops.len(), "start/stop counts differ");
        assert_eq!(starts.len(), response.timings.len(), "trigger count must match processed node count");
        assert_eq!(starts.keys().collect::<std::collections::BTreeSet<_>>(), stops.keys().collect::<std::collections::BTreeSet<_>>());
        for id in starts.keys() {
            assert!(response.timings.contains_key(id), "progress for unprocessed node {id}");
            assert!(starts[id] < stops[id], "stop before start for {id}");
        }
        let sample_connections = sample["connections"].as_array().expect("connections");
        for connection in sample_connections {
            let from = connection["from"].as_str().unwrap_or_default();
            let to = connection["to"].as_str().unwrap_or_default();
            if let (Some(&from_start), Some(&to_start)) = (starts.get(from), starts.get(to)) {
                assert!(from_start < to_start, "downstream {to} started before upstream {from}");
            }
        }
    }

    #[test]
    fn dirty_part_display_bypasses_cache_even_with_the_same_key() {
        fn request(length: f64, dirty: bool) -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["a"],
                "meshTargets": ["a"],
                "binaryMesh": false,
                "nodes": [{
                    "id": "a",
                    "type": "primitive",
                    "values": {"primitiveType": "box", "length": length, "width": 10.0, "height": 10.0},
                    "cache": {"key": "same-key", "state": if dirty { "dirty" } else { "cached" }}
                }],
                "connections": []
            })).expect("part display request")
        }
        let (first, cache, _) = respond_with_progress(request(10.0, true), EngineCache::new(), None);
        let (_, first_max) = bounds(&first.meshes["a"].mesh);
        assert!((first_max[0] - 5.0).abs() < 1.0e-3, "{first_max:?}");
        let (second, cache, _) = respond_with_progress(request(20.0, true), cache, None);
        let (_, second_max) = bounds(&second.meshes["a"].mesh);
        assert!(
            (second_max[0] - 10.0).abs() < 1.0e-3,
            "dirty display target reused the old cached mesh: {second_max:?}"
        );
        let (third, _cache, _) = respond_with_progress(request(20.0, false), cache, None);
        let (_, third_max) = bounds(&third.meshes["a"].mesh);
        assert!((third_max[0] - 10.0).abs() < 1.0e-3, "{third_max:?}");
    }

    #[test]
    fn dirty_boolean_operand_bypasses_the_brep_cache() {
        fn request(length: f64, dirty: bool) -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["u"],
                "meshTargets": ["u"],
                "binaryMesh": false,
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": length, "width": 10.0, "height": 10.0}, "cache": {"key": "same-key", "state": if dirty { "dirty" } else { "cached" }}},
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b-key", "state": "cached"}},
                    {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u-key", "state": if dirty { "dirty" } else { "cached" }}}
                ],
                "connections": [
                    {"from": "a", "to": "u", "in": "a", "out": "shape"},
                    {"from": "b", "to": "u", "in": "b", "out": "shape"}
                ]
            })).expect("boolean request")
        }
        let (first, cache, _) = respond_with_progress(request(10.0, true), EngineCache::new(), None);
        let (_, first_max) = bounds(&first.meshes["u"].mesh);
        assert!((first_max[0] - 5.0).abs() < 1.0e-3, "{first_max:?}");
        let (second, _cache, _) = respond_with_progress(request(20.0, true), cache, None);
        let (_, second_max) = bounds(&second.meshes["u"].mesh);
        assert!(
            (second_max[0] - 10.0).abs() < 1.0e-3,
            "dirty Boolean operand reused the old B-rep: {second_max:?}"
        );
    }

    #[test]
    fn dirty_node_bypasses_worker_cache_and_emits_start() {
        fn request(dirty: bool, request_id: u64) -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": request_id,
                "progress": true,
                "targets": ["a"],
                "meshTargets": [],
                "binaryMesh": false,
                "nodes": [{
                    "id": "a",
                    "type": "primitive",
                    "values": {"primitiveType": "box"},
                    "cache": {"key": "same-key", "state": if dirty { "dirty" } else { "cached" }}
                }],
                "connections": []
            })).expect("dirty request json")
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, cache, _diagnostics) = respond_with_progress(request(false, 1), EngineCache::new(), Some(tx));
        let _ = rx.try_iter().count();
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, _cache, _diagnostics) = respond_with_progress(request(true, 2), cache, Some(tx));
        let phases: Vec<String> = rx.iter().map(|bytes| {
            let frame: serde_json::Value = serde_json::from_slice(&bytes).expect("progress json");
            frame["phase"].as_str().unwrap_or_default().to_string()
        }).collect();
        assert_eq!(phases, vec!["node-start", "node-complete"], "dirty node must be recomputed");
    }

    #[test]
    fn cached_node_does_not_emit_progress() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 101,
            "progress": true,
            "targets": ["a"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [{"id": "a", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "cached-key", "state": "dirty"}}],
            "connections": []
        })).expect("cached request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, cache, _diagnostics) = respond_with_progress(request, EngineCache::new(), Some(tx));
        let _ = rx.try_iter().count();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 102,
            "progress": true,
            "targets": ["a"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [{"id": "a", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "cached-key", "state": "cached"}}],
            "connections": []
        })).expect("cached request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, _cache, _diagnostics) = respond_with_progress(request, cache, Some(tx));
        assert_eq!(rx.try_iter().count(), 0, "cached node must not emit progress");
    }

    #[test]
    fn failed_node_is_not_reprocessed_from_multiple_branches() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 99,
            "progress": true,
            "targets": ["b", "c"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [
                {"id": "bad", "type": "not-a-real-node", "values": {}},
                {"id": "b", "type": "translate", "values": {"x": 1, "y": 0, "z": 0}},
                {"id": "c", "type": "translate", "values": {"x": 2, "y": 0, "z": 0}}
            ],
            "connections": [
                {"from": "bad", "to": "b", "in": "shape", "out": "shape"},
                {"from": "bad", "to": "c", "in": "shape", "out": "shape"}
            ]
        })).expect("failed branch request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) = respond_with_progress(request, EngineCache::new(), Some(tx));
        let starts: Vec<String> = rx.iter().filter_map(|bytes| {
            let frame: serde_json::Value = serde_json::from_slice(&bytes).expect("progress json");
            (frame["phase"] == "node-start").then(|| frame["nodeId"].as_str().unwrap_or_default().to_string())
        }).collect();
        assert_eq!(starts.iter().filter(|id| id.as_str() == "bad").count(), 1, "failed node restarted: {starts:?}");
        assert!(response.node_errors.contains_key("bad"));
        assert!(response.node_errors.get("b").is_some_and(|error| error.contains("Input bad failed")));
        assert!(response.node_errors.get("c").is_some_and(|error| error.contains("Input bad failed")));
    }

    #[test]
    fn errored_node_bypasses_to_its_input() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 100,
            "targets": ["move"],
            "meshTargets": ["move"],
            "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "sphere"}},
                {"id": "bad", "type": "boolean", "values": {"mode": "not-a-mode"}},
                {"id": "move", "type": "translate", "values": {"x": 3.0, "y": 0.0, "z": 0.0}}
            ],
            "connections": [
                {"from": "a", "to": "bad", "in": "a", "out": "shape"},
                {"from": "b", "to": "bad", "in": "b", "out": "shape"},
                {"from": "bad", "to": "move", "in": "shape", "out": "shape"}
            ]
        })).expect("bypass request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.contains_key("bad"));
        assert!(!output.response.node_errors.contains_key("move"));
        let mesh = output.response.meshes.get("move").expect("bypassed mesh");
        assert!(!mesh.mesh.faces.is_empty(), "bypassed output is empty");
    }

    #[test]
    fn downstream_node_runs_once_across_multiple_targets() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 55,
            "progress": true,
            "targets": ["b", "a", "b"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "b", "type": "translate", "values": {"x": 1, "y": 0, "z": 0}}
            ],
            "connections": [{"from": "a", "to": "b", "in": "shape", "out": "shape"}]
        })).expect("multi-target request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, _cache, _diagnostics) = respond_with_progress(request, EngineCache::new(), Some(tx));
        let starts: Vec<String> = rx.iter().filter_map(|bytes| {
            let frame: serde_json::Value = serde_json::from_slice(&bytes).expect("progress json");
            (frame["phase"] == "node-start").then(|| frame["nodeId"].as_str().unwrap_or_default().to_string())
        }).collect();
        assert_eq!(starts, vec!["a".to_string(), "b".to_string()], "a downstream node must run once even when listed as multiple targets");
    }

    #[test]
    fn emits_ordered_progress_signals_for_evaluated_graph() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 42,
            "progress": true,
            "targets": ["c"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "b", "type": "translate", "values": {"x": 1, "y": 0, "z": 0}},
                {"id": "c", "type": "translate", "values": {"x": 2, "y": 0, "z": 0}}
            ],
            "connections": [
                {"from": "a", "to": "b", "in": "shape", "out": "shape"},
                {"from": "b", "to": "c", "in": "shape", "out": "shape"}
            ]
        })).expect("progress request json");
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) = respond_with_progress(request, EngineCache::new(), Some(progress_tx));
        let frames: Vec<serde_json::Value> = progress_rx
            .iter()
            .map(|bytes| serde_json::from_slice(&bytes).expect("progress frame json"))
            .collect();
        let signals: Vec<(String, String)> = frames
            .iter()
            .map(|frame| {
                assert_eq!(frame["type"], "evaluation-progress");
                assert_eq!(frame["requestId"], 42);
                (
                    frame["nodeId"].as_str().unwrap_or_default().to_string(),
                    frame["phase"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let expected = vec![
            ("a".to_string(), "node-start".to_string()),
            ("a".to_string(), "node-complete".to_string()),
            ("b".to_string(), "node-start".to_string()),
            ("b".to_string(), "node-complete".to_string()),
            ("c".to_string(), "node-start".to_string()),
            ("c".to_string(), "node-complete".to_string()),
        ];
        assert_eq!(signals, expected, "progress must flow upstream to downstream");
        assert_eq!(response.evaluation_order, vec!["a", "b", "c"]);
        assert!(response.node_errors.is_empty(), "unexpected errors: {:?}", response.node_errors);
    }

    fn request_for(path: &str, diagnostics: bool) -> Request {
        let mut values = serde_json::json!({ "path": path });
        if diagnostics {
            values["diagnostics"] = serde_json::json!(true);
        }
        serde_json::from_value(serde_json::json!({
            "requestId": 7,
            "targets": ["step"],
            "meshTargets": ["step"],
            "binaryMesh": false,
            "nodes": [{
                "id": "step",
                "type": "stepimporter",
                "values": values,
            }],
            "connections": [],
        }))
        .expect("request json")
    }

    /// block.stp is the small 6-face box that used to be rejected as "no planar
    /// face loops"; it must now import through the exact planar Brep path with no
    /// partial-import fallbacks.
    #[test]
    fn imports_block_step_through_exact_planar_brep() {
        let import = import_fixture(BLOCK);
        let mesh = &import.mesh;
        assert!(!mesh.faces.is_empty(), "block.stp produced no triangles");
        assert!(!mesh.vertices.is_empty(), "block.stp produced no vertices");
        assert_eq!(unique_groups(mesh).len(), 6, "block.stp face groups");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 6);
        assert_eq!(import.diagnostics.brep_edges, 12);
        assert_eq!(import.diagnostics.brep_vertices, 8);
        assert_eq!(import.diagnostics.name.as_deref(), Some("block"));
        assert_eq!(import.diagnostics.color, None);
        assert_eq!(import.diagnostics.mesh_triangles, mesh.faces.len());
        assert!(import.diagnostics.fallbacks.is_empty(), "partial-import fallbacks: {:?}", import.diagnostics.fallbacks);
        assert!(import.diagnostics.warnings.is_empty(), "partial-import warnings: {:?}", import.diagnostics.warnings);
        assert_bounds(mesh, [-1.5, -2.0, -2.5], [1.5, 2.0, 2.5], 1e-3);
    }

    /// samples/rust-export.step contains cone, sphere, box, wedge, cylinder, torus.
    #[test]
    fn imports_multi_solid_primitive_sample() {
        let import = import_fixture("samples/rust-export.step");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.bodies, 6);
        assert!(import.diagnostics.breps >= 2);
        assert!(!import.mesh.faces.is_empty(), "multi-solid import produced no triangles");
        assert!(import.diagnostics.warnings.is_empty(), "{:?}", import.diagnostics.warnings);
        let (min, max) = bounds(&import.mesh);
        assert!((min[0] - -55.0).abs() < 1e-3 && (max[0] - 55.0).abs() < 1e-3, "{min:?} {max:?}");
    }

    #[test]
    fn imports_faceted_brep_tetrahedron() {
        let path = std::env::temp_dir().join(format!("sn-worker-faceted-{}.step", std::process::id()));
        std::fs::write(&path, "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('test'),'2;1');\nFILE_NAME('','',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,0.,0.));\n#3=CARTESIAN_POINT('',(0.,1.,0.));\n#4=CARTESIAN_POINT('',(0.,0.,1.));\n#10=POLY_LOOP('',(#1,#2,#3));\n#11=POLY_LOOP('',(#1,#3,#4));\n#12=POLY_LOOP('',(#1,#4,#2));\n#13=POLY_LOOP('',(#2,#4,#3));\n#20=FACE_OUTER_BOUND('',#10,.T.);\n#21=FACE_OUTER_BOUND('',#11,.T.);\n#22=FACE_OUTER_BOUND('',#12,.T.);\n#23=FACE_OUTER_BOUND('',#13,.T.);\n#30=FACE('',(#20));\n#31=FACE('',(#21));\n#32=FACE('',(#22));\n#33=FACE('',(#23));\n#40=CLOSED_SHELL('',(#30,#31,#32,#33));\n#50=FACETED_BREP('tetra',#40);\nENDSEC;\nEND-ISO-10303-21;\n").unwrap();
        let import = import_step(path.to_str().unwrap()).unwrap();
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.bodies, 1);
        assert!(!import.mesh.faces.is_empty());
    }

    /// doghouse.stp mixes planar and cylindrical faces. The mixed reader is not exact
    /// yet, so it must fail explicitly rather than importing a partial planar shell.
    #[test]
    fn doghouse_mixed_imports_exactly() {
        let import = match import_step(&fixture(DOGHOUSE)) {
            Ok(import) => import,
            Err(error) => panic!("doghouse must import exactly, got: {error}"),
        };
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert!(!import.mesh.faces.is_empty(), "doghouse produced no triangles");
        assert_eq!(import.diagnostics.brep_faces, 32);
    }

    /// The exporter's unit option writes conversion-based inch files.
    #[test]
    fn exports_inch_primitive() {
        let directory = std::env::temp_dir().join(format!("sn-worker-inch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 40.0, "width": 40.0, "height": 40.0}, "cache": {"key": "b"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "inch", "unit": "inch"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "b", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("inch.step");
        assert!(path.exists());
        assert_eq!(nodecraft_step::read_length_unit(path.to_str().unwrap()).unwrap(), nodecraft_step::LengthUnit::Inch);
        let (min, max) = nodecraft_step::read_step_bounds(path.to_str().unwrap()).unwrap();
        assert!((min[0] + 20.0).abs() < 1e-6 && (max[0] - 20.0).abs() < 1e-6, "{min:?} {max:?}");
    }

    #[test]
    fn exact_demands_skip_mesh_only_chains_and_follow_booleans() {
        let mesh_request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["c"], "meshTargets": ["c"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "t", "type": "translate", "values": {"x": 1.0}},
                {"id": "c", "type": "combine", "values": {}}
            ],
            "connections": [
                {"from": "p", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "c", "in": "shape"}
            ]
        })).expect("mesh request");
        assert!(exact_brep_demands(&mesh_request).is_empty());

        let boolean_request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["b"], "meshTargets": ["b"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "b", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [{"from": "p", "out": "shape", "to": "b", "in": "a"}]
        })).expect("boolean request");
        let demands = exact_brep_demands(&boolean_request);
        assert!(demands.contains("p"));
        assert!(demands.contains("b"));
    }

    #[test]
    fn part_combine_preserves_each_translated_solid() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["teddy"], "meshTargets": [],
            "nodes": [
                {"id": "body", "type": "primitive", "values": {"primitiveType": "sphere", "radius": 46}},
                {"id": "body_t", "type": "translate", "values": {"x": 0, "y": 0, "z": 55}},
                {"id": "head", "type": "primitive", "values": {"primitiveType": "sphere", "radius": 32}},
                {"id": "head_t", "type": "translate", "values": {"x": 0, "y": 0, "z": 115}},
                {"id": "teddy", "type": "combine", "values": {}}
            ],
            "connections": [
                {"from": "body", "to": "body_t", "in": "shape", "out": "shape"},
                {"from": "body_t", "to": "teddy", "in": "shape", "out": "shape"},
                {"from": "head", "to": "head_t", "in": "shape", "out": "shape"},
                {"from": "head_t", "to": "teddy", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let mut eval = Eval::new(&request, None);
        let part = eval.evaluate_part("teddy").expect("part");
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        let solids = part.realized_solids().expect("realize");
        for brep in &solids {
            for vertex in brep.vertex_list() {
                let z = brep.vertex(*vertex).expect("vertex").point.z;
                min_z = min_z.min(z);
                max_z = max_z.max(z);
            }
        }
        assert_eq!(part.bodies.len(), 2);
        assert!(max_z - min_z > 100.0, "translated solids collapsed: {min_z}..{max_z}");
    }

    #[test]
    fn space_does_not_change_translation_geometry() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["t"], "meshTargets": ["t"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 10, "height": 10}, "cache": {"key": "p"}},
                {"id": "t", "type": "translate", "values": {"space": "local", "x": 30, "y": 0, "z": 0, "rz": 90}, "cache": {"key": "t"}}
            ],
            "connections": [{"from": "p", "to": "t", "in": "shape", "out": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["t"].mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[0] - 25.0).abs() < 1e-4 && (max[0] - 35.0).abs() < 1e-4, "{min:?} {max:?}");
        assert!((min[1] + 10.0).abs() < 1e-4 && (max[1] - 10.0).abs() < 1e-4, "{min:?} {max:?}");
    }

    #[test]
    fn exact_part_cache_returns_the_same_immutable_value() {
        let mut cache = EngineCache::new();
        let part = Arc::new(Part::empty());
        cache.insert_part(("x".into(), "k".into()), part.clone());
        assert!(Arc::ptr_eq(&cache.part(&("x".into(), "k".into())).unwrap(), &part));
    }

    #[test]
    fn exports_pyramid_combine_assembly() {
        let directory = std::env::temp_dir().join(format!("sn-worker-pyramid-assembly-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "p1", "type": "primitive", "values": {"primitiveType": "pyramid", "length": 10.0, "height": 12.0}, "cache": {"key": "p1"}},
                {"id": "t1", "type": "translate", "values": {"x": -10.0, "y": 0.0, "z": 0.0}, "cache": {"key": "t1"}},
                {"id": "p2", "type": "primitive", "values": {"primitiveType": "pyramid", "length": 10.0, "height": 12.0}, "cache": {"key": "p2"}},
                {"id": "t2", "type": "translate", "values": {"x": 10.0, "y": 0.0, "z": 0.0}, "cache": {"key": "t2"}},
                {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "pyramids"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "p1", "out": "shape", "to": "t1", "in": "shape"},
                {"from": "p2", "out": "shape", "to": "t2", "in": "shape"},
                {"from": "t1", "out": "shape", "to": "c", "in": "shape"},
                {"from": "t2", "out": "shape", "to": "c", "in": "shape"},
                {"from": "c", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let bodies = nodecraft_step::read_planar_solids(directory.join("pyramids.step").to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 2);
        assert!((bodies.iter().map(|body| body.volume()).sum::<f64>() - 800.0).abs() < 1e-6);
    }

    #[test]
    fn exports_mirrored_part_as_two_exact_solids() {
        let directory = std::env::temp_dir().join(format!("sn-worker-mirror-part-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 4.0, "width": 4.0, "height": 4.0}, "cache": {"key": "p"}},
                {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}},
                {"id": "m", "type": "mirror", "values": {"plane": "yz", "offset": 0.0, "combine": true}, "cache": {"key": "m"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "mirror"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "p", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "m", "in": "shape"},
                {"from": "m", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let bodies = nodecraft_step::read_planar_solids(directory.join("mirror.step").to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 2, "mirror combine should export source and mirrored Part");
        assert!((bodies.iter().map(|body| body.volume()).sum::<f64>() - 128.0).abs() < 1e-6);
    }

    #[test]
    fn exports_exact_pyramid_step() {
        let directory = std::env::temp_dir().join(format!("sn-worker-pyramid-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "pyramid", "length": 20.0, "height": 30.0}, "cache": {"key": "p"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "pyramid"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "p", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let body = nodecraft_step::read_planar_body(directory.join("pyramid.step").to_str().unwrap()).unwrap();
        assert_eq!(body.faces.len(), 5);
        // The exact SolidCore pyramid keeps its split seam vertices; the planar
        // writer may therefore emit more than the five corner points.
        assert!(body.points.len() >= 5);
        assert!((body.volume() - 4000.0).abs() < 1e-6, "{}", body.volume());
    }

    #[test]
    fn exports_primitive_as_faceted() {
        let directory = std::env::temp_dir().join(format!("sn-worker-prim-faceted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 2.0, "width": 2.0, "height": 2.0}, "cache": {"key": "b"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "faceted", "format": "faceted"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "b", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("faceted.step");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("FACETED_BREP("));
        let bodies = nodecraft_step::read_faceted_breps(path.to_str().unwrap()).unwrap();
        assert!((bodies[0].volume() - 8.0).abs() < 1e-6, "{}", bodies[0].volume());
    }

    #[test]
    fn reexports_imported_planar_step_as_faceted() {
        let directory = std::env::temp_dir().join(format!("sn-worker-faceted-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": fixture(BLOCK)}, "cache": {"key": "i"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "faceted", "format": "faceted"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "i", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("faceted.step");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("FACETED_BREP("), "not a faceted brep");
        let bodies = nodecraft_step::read_faceted_breps(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert!((bodies[0].volume() - 60.0).abs() < 1e-6, "{}", bodies[0].volume());
    }

    #[test]
    fn exports_curved_import_as_faceted_mesh() {
        let directory = std::env::temp_dir().join(format!("sn-worker-curved-faceted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let cylinder = directory.join("cylinder.step");
        nodecraft_step::write_cylinder(cylinder.to_str().unwrap(), 5.0, 10.0).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": cylinder.to_string_lossy()}, "cache": {"key": "i"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "faceted", "format": "faceted"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "i", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("faceted.step");
        let bodies = nodecraft_step::read_faceted_breps(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].faces.is_empty());
    }

    /// Exact Boolean STEP export: disjoint imported boxes union to a valid STEP solid.
    #[test]
    fn exports_exact_boolean_step() {
        let directory = std::env::temp_dir().join(format!("sn-worker-boolexport-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        nodecraft_step::write_box(a.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        nodecraft_step::write_box(b.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        nodecraft_step::transform_step_file(b.to_str().unwrap(), [0.0; 3], [20.0, 0.0, 0.0]).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "stepimporter", "values": {"path": a.to_string_lossy()}, "cache": {"key": "a"}},
                {"id": "b", "type": "stepimporter", "values": {"path": b.to_string_lossy()}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "union", "productName": "Union", "color": "#20a060", "layer": "Parts", "schema": "ap203"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "u", "in": "b"},
                {"from": "u", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("union.step");
        assert!(path.exists(), "Boolean export was not written");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("COLOUR_RGB"), "Boolean color metadata missing");
        assert!(text.contains("PRESENTATION_LAYER_ASSIGNMENT('Parts'"), "Boolean layer metadata missing");
        assert!(text.contains("CONFIG_CONTROL_DESIGN"), "Boolean schema metadata missing");
        let body = nodecraft_step::read_planar_body(path.to_str().unwrap()).unwrap();
        assert!((body.volume() - 1600.0).abs() < 1e-6, "volume {}", body.volume());
        let (min, max) = body.bounds().unwrap();
        assert!((min[0] - -5.0).abs() < 1e-6 && (max[0] - 25.0).abs() < 1e-6, "{min:?} {max:?}");
    }

    #[test]
    fn mirror_node_mirrors_mesh_and_exports_exact_step() {
        let directory = std::env::temp_dir().join(format!("sn-mirror-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["m", "e"],
            "meshTargets": ["m"],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 40, "height": 40}, "cache": {"key": "b"}},
                {"id": "m", "type": "mirror", "values": {"plane": "yz", "offset": 10.0}, "cache": {"key": "m"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "mirrored"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "b", "out": "shape", "to": "m", "in": "shape"},
                {"from": "m", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes.get("m").unwrap_or_else(|| panic!("mirror mesh keys {:?}", response.meshes.keys().collect::<Vec<_>>())).mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[0] - 0.0).abs() < 1e-4 && (max[0] - 40.0).abs() < 1e-4, "{min:?} {max:?}");
        assert!((min[1] + 20.0).abs() < 1e-4 && (max[1] - 20.0).abs() < 1e-4, "{min:?} {max:?}");
        let body = nodecraft_step::read_planar_body(directory.join("mirrored.step").to_str().unwrap()).expect("mirrored STEP");
        assert!((body.volume().abs() - 64000.0).abs() < 1e-6, "volume {}", body.volume());
    }

    #[test]
    fn mirror_combine_keeps_the_original() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["m"],
            "meshTargets": ["m"],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 40, "height": 40}, "cache": {"key": "b"}},
                {"id": "m", "type": "mirror", "values": {"plane": "yz", "offset": 20, "combine": true}, "cache": {"key": "m"}}
            ],
            "connections": [{"from": "b", "out": "shape", "to": "m", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes.get("m").expect("mirror mesh").mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[0] + 10.0).abs() < 1e-4 && (max[0] - 50.0).abs() < 1e-4, "{min:?} {max:?}");
    }

    /// Imported STEP re-export preserves geometry and applies transforms.
    #[test]
    fn reexports_imported_step_with_transform() {
        let directory = std::env::temp_dir().join(format!("sn-worker-reexport-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let block = fixture(BLOCK);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": block}, "cache": {"key": "i"}},
                {"id": "t", "type": "translate", "values": {"x": 10}, "cache": {"key": "t"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "copy", "productName": "Bracket", "color": "#3366cc", "layer": "Parts", "schema": "ap203", "unit": "inch"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "i", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("copy.step");
        assert!(path.exists(), "re-export was not written");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("COLOUR_RGB"), "metadata color missing");
        assert!(text.contains("PRESENTATION_LAYER_ASSIGNMENT('Parts'"), "metadata layer missing");
        assert!(text.contains("CONVERSION_BASED_UNIT('INCH'"), "metadata unit missing");
        assert!(text.contains("CONFIG_CONTROL_DESIGN"), "metadata schema missing");
        assert!(text.contains("Bracket"), "metadata product name missing");
        let bodies = nodecraft_step::read_planar_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0];
        assert!((body.volume() - 60.0).abs() < 1e-6, "volume {}", body.volume());
        let (min, max) = body.bounds().unwrap();
        assert!((min[0] - 8.5).abs() < 1e-6 && (max[0] - 11.5).abs() < 1e-6, "{min:?} {max:?}");
    }

    /// Exact disjoint Boolean: two imported planar STEP solids union through solidcore.
    #[test]
    fn boolean_union_of_disjoint_imported_solids() {
        let directory = std::env::temp_dir().join(format!("sn-worker-bool-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        nodecraft_step::write_box(a.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        nodecraft_step::write_box(b.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        nodecraft_step::transform_step_file(b.to_str().unwrap(), [0.0; 3], [20.0, 0.0, 0.0]).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": ["u"],
            "nodes": [
                {"id": "a", "type": "stepimporter", "values": {"path": a.to_string_lossy()}, "cache": {"key": "a"}},
                {"id": "b", "type": "stepimporter", "values": {"path": b.to_string_lossy()}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "u", "in": "b"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["u"].mesh;
        assert!(!mesh.faces.is_empty(), "boolean produced no triangles");
        let (min, max) = bounds(mesh);
        assert!(min[0] < -4.9 && max[0] > 24.9, "{min:?}..{max:?}");
    }

    #[test]
    fn exporter_defaults_to_project_folder() {
        let directory = std::env::temp_dir().join(format!("sn-worker-project-folder-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "projectDirectory": directory.to_string_lossy(),
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "pyramid", "length": 20, "height": 30}, "cache": {"key": "p"}},
                {"id": "e", "type": "exporter", "values": {"fileName": "project-part"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "p", "out": "shape", "to": "e", "in": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(directory.join("project-part.step").exists(), "export did not use projectDirectory");
    }

    /// Overlapping imported planar STEP solids now union through solidcore.
    #[test]
    fn boolean_union_of_overlapping_imported_solids() {
        let directory = std::env::temp_dir().join(format!("sn-worker-bool-overlap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        nodecraft_step::write_box(a.to_str().unwrap(), 10.0, 10.0, 10.0).unwrap();
        nodecraft_step::write_box(b.to_str().unwrap(), 10.0, 10.0, 10.0).unwrap();
        nodecraft_step::transform_step_file(b.to_str().unwrap(), [0.0; 3], [5.0, 0.0, 0.0]).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "stepimporter", "values": {"path": a.to_string_lossy()}, "cache": {"key": "a"}},
                {"id": "b", "type": "stepimporter", "values": {"path": b.to_string_lossy()}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "overlap"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "u", "in": "b"},
                {"from": "u", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let body = nodecraft_step::read_planar_body(directory.join("overlap.step").to_str().unwrap()).unwrap();
        assert!((body.volume() - 1500.0).abs() < 1e-6, "volume {}", body.volume());
        let (min, max) = body.bounds().unwrap();
        assert!((min[0] - -5.0).abs() < 1e-6 && (max[0] - 10.0).abs() < 1e-6, "{min:?} {max:?}");
    }

    /// Boolean nodes work directly on NodeCraft primitives, including translated ones.
    #[test]
    fn boolean_union_of_overlapping_primitives() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": ["u"],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                {"id": "t", "type": "translate", "values": {"x": 5.0}, "cache": {"key": "t"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "u", "in": "b"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["u"].mesh;
        assert!(!mesh.faces.is_empty());
        let (min, max) = bounds(mesh);
        assert!(min[0] <= -4.9 && max[0] >= 9.9, "{min:?}..{max:?}");
        assert!(!response.brep_timings.is_empty(), "operand B-rep work should be timed");
    }

    #[test]
    fn operand_brep_emits_brep_progress_phases() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": [],
            "progress": true,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                {"id": "t", "type": "translate", "values": {"x": 5.0}, "cache": {"key": "t"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "u", "in": "b"}
            ]
        })).expect("request");
        let mut brep_starts = Vec::new();
        let mut progress = |event: ProgressEvent| {
            if event.phase == ProgressPhase::BrepStart { brep_starts.push(event.node_id); }
        };
        let output = evaluate(request, EngineCache::new(), Some(&mut progress));
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(brep_starts.iter().any(|id| id == "a"), "{brep_starts:?}");
        assert!(brep_starts.iter().any(|id| id == "t"), "{brep_starts:?}");
    }

    #[test]
    fn combine_node_passes_a_single_exact_operand_to_boolean() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": ["u"],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "c", "in": "shape"},
                {"from": "c", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "u", "in": "b"}
            ]
        })).expect("request");
        let response = evaluate(request, EngineCache::new(), None).response;
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["u"].mesh.faces.is_empty());
    }

    #[test]
    fn combine_node_passes_multiple_exact_operands_to_boolean() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": ["u"],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                {"id": "tb", "type": "translate", "values": {"x": 20.0}, "cache": {"key": "tb"}},
                {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "c", "in": "shape"},
                {"from": "b", "out": "shape", "to": "tb", "in": "shape"},
                {"from": "tb", "out": "shape", "to": "c", "in": "shape"},
                {"from": "c", "out": "shape", "to": "u", "in": "a"}
            ]
        })).expect("request");
        let response = evaluate(request, EngineCache::new(), None).response;
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["u"].mesh.faces.is_empty());
    }

    #[test]
    fn part_target_reuses_unchanged_operands_after_target_key_change() {
        let make = |request_id: u64, boolean_key: &str| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": request_id, "targets": ["u"], "meshTargets": ["u"],
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "a"}},
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "b"}},
                    {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}},
                    {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": boolean_key}}
                ],
                "connections": [
                    {"from": "a", "to": "u", "in": "a", "out": "shape"},
                    {"from": "b", "to": "t", "in": "shape", "out": "shape"},
                    {"from": "t", "to": "u", "in": "b", "out": "shape"}
                ]
            })).expect("request")
        };
        let first = evaluate(make(1, "u1"), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let second = evaluate(make(2, "u2"), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.part_hits() > 0, "unchanged operand Parts were rebuilt");
        assert!(second.response.cache_stats.expect("cache stats").hits > 0, "report must include Part/BRep cache hits");
    }

    #[test]
    fn part_cache_retains_a_large_evaluated_graph() {
        let mut cache = EngineCache::new();
        for index in 0..5_000 {
            cache.insert_part((format!("node{index}"), String::new()), Arc::new(Part::empty()));
        }
        assert!(cache.part(&("node0".into(), String::new())).is_some(), "early Part entry was evicted");
    }

    #[test]
    fn part_target_reuses_primitive_brep_after_part_eviction() {
        let make = || -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1, "targets": ["c"], "meshTargets": ["c"],
                "nodes": [
                    {"id": "p", "type": "primitive", "values": {"primitiveType": "sphere", "radius": 10.0}, "cache": {"key": "p"}},
                    {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}},
                    {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}}
                ],
                "connections": [
                    {"from": "p", "to": "t", "in": "shape", "out": "shape"},
                    {"from": "t", "to": "c", "in": "shape", "out": "shape"}
                ]
            })).expect("request")
        };
        let first = evaluate(make(), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let mut cache = first.cache;
        assert!(cache.brep_entries() > 0);
        // Force the small Part cache to evict its entries while retaining BReps.
        for index in 0..8_300 {
            cache.insert_part((format!("evict{index}"), String::new()), Arc::new(Part::empty()));
        }
        let second = evaluate(make(), cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.brep_hits() > 0, "primitive BRep cache did not survive Part eviction");
    }

    #[test]
    fn part_target_reuses_cached_parts_and_display_mesh() {
        let make = || -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1, "targets": ["c"], "meshTargets": ["c"],
                "nodes": [
                    {"id": "p", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "p"}},
                    {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}},
                    {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}}
                ],
                "connections": [
                    {"from": "p", "to": "t", "in": "shape", "out": "shape"},
                    {"from": "t", "to": "c", "in": "shape", "out": "shape"}
                ]
            })).expect("request")
        };
        let first = evaluate(make(), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let second = evaluate(make(), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.mesh_entries() > 0);
        assert!(second.response.cache_stats.expect("cache stats").hits > 0, "display mesh was not reused");
        assert!(second.cache.part_hits() > 0, "Part was not reused");
    }

    #[test]
    fn downstream_change_reuses_cached_operand_breps() {
        let make_request = |request_id: u64, boolean_key: &str| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": request_id,
                "targets": ["u"],
                "meshTargets": [],
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                    {"id": "t", "type": "translate", "values": {"x": 5.0}, "cache": {"key": "t"}},
                    {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": boolean_key}}
                ],
                "connections": [
                    {"from": "a", "out": "shape", "to": "u", "in": "a"},
                    {"from": "b", "out": "shape", "to": "t", "in": "shape"},
                    {"from": "t", "out": "shape", "to": "u", "in": "b"}
                ]
            })).expect("request")
        };
        let first = evaluate(make_request(1, "u1"), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        assert!(first.cache.brep_entries() > 0);
        assert_eq!(first.cache.brep_hits(), 0);
        let second = evaluate(make_request(2, "u2"), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.brep_hits() > 0, "unchanged operand B-reps were rebuilt");
    }

    /// Compound rotated curved operands still reach an upstream face-query gap.
    #[test]
    fn rotated_curved_y_axis_boolean_succeeds() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["n28"], "meshTargets": ["n28"],
            "nodes": [
                {"id": "n24", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "n24"}},
                {"id": "n25", "type": "translate", "values": {"x": -50, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0}, "cache": {"key": "n25"}},
                {"id": "n26", "type": "primitive", "values": {"radius": 3, "primitiveType": "cylinder", "height": 8}, "cache": {"key": "n26"}},
                {"id": "n27", "type": "translate", "values": {"x": -56.091, "y": 0, "z": 0, "rx": 0, "ry": 35.524, "rz": 0}, "cache": {"key": "n27"}},
                {"id": "n28", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "n28"}}
            ],
            "connections": [
                {"from": "n24", "to": "n25", "in": "shape", "out": "shape"},
                {"from": "n26", "to": "n27", "in": "shape", "out": "shape"},
                {"from": "n25", "to": "n28", "in": "a", "out": "shape"},
                {"from": "n27", "to": "n28", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["n28"].mesh.faces.is_empty());
    }

    fn rotated_curved_compound_boolean_reports_upstream_limit() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["n28"], "meshTargets": ["n28"],
            "nodes": [
                {"id": "n24", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "n24"}},
                {"id": "n25", "type": "translate", "values": {"x": -50, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0}, "cache": {"key": "n25"}},
                {"id": "n26", "type": "primitive", "values": {"radius": 5, "primitiveType": "cylinder", "height": 10}, "cache": {"key": "n26"}},
                {"id": "n27", "type": "translate", "values": {"x": -43.107, "y": 0, "z": 3.607, "rx": -14.92, "ry": 27.495, "rz": -29.992}, "cache": {"key": "n27"}},
                {"id": "n28", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "n28"}}
            ],
            "connections": [
                {"from": "n24", "to": "n25", "in": "shape", "out": "shape"},
                {"from": "n26", "to": "n27", "in": "shape", "out": "shape"},
                {"from": "n25", "to": "n28", "in": "a", "out": "shape"},
                {"from": "n27", "to": "n28", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        let error = response.node_errors.get("n28").expect("compound rotated curved operand still reports upstream");
        assert!(error.contains("SmFace::"), "{error}");
        assert!(error.contains("topology.face_p2") || error.contains("query.face_p2"), "{error}");
    }

    /// A single-axis rotated curved operand now completes exactly.
    #[test]
    fn rotated_curved_x_axis_boolean_succeeds() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["n28"], "meshTargets": ["n28"],
            "nodes": [
                {"id": "n24", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "n24"}},
                {"id": "n25", "type": "translate", "values": {"x": -50, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0}, "cache": {"key": "n25"}},
                {"id": "n26", "type": "primitive", "values": {"radius": 3, "primitiveType": "cylinder", "height": 8}, "cache": {"key": "n26"}},
                {"id": "n27", "type": "translate", "values": {"x": -56.091, "y": 0, "z": 0, "rx": 35.524, "ry": 0, "rz": 0}, "cache": {"key": "n27"}},
                {"id": "n28", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "n28"}}
            ],
            "connections": [
                {"from": "n24", "to": "n25", "in": "shape", "out": "shape"},
                {"from": "n26", "to": "n27", "in": "shape", "out": "shape"},
                {"from": "n25", "to": "n28", "in": "a", "out": "shape"},
                {"from": "n27", "to": "n28", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["n28"].mesh.faces.is_empty());
    }

    /// Boolean operands apply node rotation, not just translation.
    #[test]
    fn subtract_reverse_swaps_operands_for_solidcore_difference() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 501,
            "targets": ["normal", "reverse"],
            "meshTargets": ["normal", "reverse"],
            "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "t", "type": "translate", "values": {"x": 6, "y": 0, "z": 0}},
                {"id": "normal", "type": "boolean", "values": {"mode": "subtract"}},
                {"id": "reverse", "type": "boolean", "values": {"mode": "subtract-reverse"}}
            ],
            "connections": [
                {"from": "a", "to": "normal", "in": "a", "out": "shape"},
                {"from": "b", "to": "t", "in": "shape", "out": "shape"},
                {"from": "t", "to": "normal", "in": "b", "out": "shape"},
                {"from": "a", "to": "reverse", "in": "a", "out": "shape"},
                {"from": "t", "to": "reverse", "in": "b", "out": "shape"}
            ]
        })).expect("subtract-reverse request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let normal = bounds(&output.response.meshes["normal"].mesh);
        let reverse = bounds(&output.response.meshes["reverse"].mesh);
        assert!(!output.response.meshes["normal"].mesh.faces.is_empty());
        assert!(!output.response.meshes["reverse"].mesh.faces.is_empty());
        assert!(normal.0[0] < reverse.0[0], "reverse must swap A/B: normal={normal:?} reverse={reverse:?}");
    }

    #[test]
    fn boolean_union_of_rotated_primitives() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["u"],
            "meshTargets": ["u"],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 30.0, "width": 10.0, "height": 5.0}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 30.0, "width": 10.0, "height": 5.0}, "cache": {"key": "b"}},
                {"id": "t", "type": "translate", "values": {"rx": 0.0, "ry": 0.0, "rz": 90.0}, "cache": {"key": "t"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "u", "in": "a"},
                {"from": "b", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "u", "in": "b"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["u"].mesh;
        assert!(!mesh.faces.is_empty());
        let (min, max) = bounds(mesh);
        assert!(max[1] >= 14.9, "rotated operand was not applied: {min:?}..{max:?}");
    }

    #[test]
    fn exports_curved_boolean_as_faceted() {
        let directory = std::env::temp_dir().join(format!("sn-worker-bool-faceted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "cylinder", "radius": 3, "height": 8}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "curved-union", "format": "faceted"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "a", "to": "u", "in": "a", "out": "shape"},
                {"from": "b", "to": "u", "in": "b", "out": "shape"},
                {"from": "u", "to": "e", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let bodies = nodecraft_step::read_faceted_breps(directory.join("curved-union.step").to_str().unwrap()).unwrap();
        assert!(!bodies.is_empty());
        assert!(bodies[0].volume() > 0.0);
    }

    /// Suppressing both box and its transform leaves only the active cylinder.
    #[test]
    fn suppressed_operand_is_removed_from_boolean() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["n28"], "meshTargets": ["n28"],
            "nodes": [
                {"id": "n24", "type": "primitive", "suppressed": true, "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "n24"}},
                {"id": "n25", "type": "translate", "suppressed": true, "values": {"x": -50}, "cache": {"key": "n25"}},
                {"id": "n26", "type": "primitive", "values": {"primitiveType": "cylinder", "radius": 2, "height": 8}, "cache": {"key": "n26"}},
                {"id": "n27", "type": "translate", "values": {"x": -54.387}, "cache": {"key": "n27"}},
                {"id": "n28", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "n28"}}
            ],
            "connections": [
                {"from": "n24", "to": "n25", "in": "shape", "out": "shape"},
                {"from": "n26", "to": "n27", "in": "shape", "out": "shape"},
                {"from": "n25", "to": "n28", "in": "a", "out": "shape"},
                {"from": "n27", "to": "n28", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["n28"].mesh;
        let (min, max) = bounds(mesh);
        assert!(min[0] > -56.5 && max[0] < -52.0, "box still present: {min:?}..{max:?}");
    }

    /// Exact assembly export: two placed planar STEP parts through a Combine node.
    #[test]
    fn exports_exact_planar_assembly() {
        let directory = std::env::temp_dir().join(format!("sn-worker-assembly-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let block = fixture(BLOCK);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "stepimporter", "values": {"path": block}, "cache": {"key": "a"}},
                {"id": "ta", "type": "translate", "values": {"x": 10}, "cache": {"key": "ta"}},
                {"id": "b", "type": "stepimporter", "values": {"path": block}, "cache": {"key": "b"}},
                {"id": "tb", "type": "translate", "values": {"z": 20}, "cache": {"key": "tb"}},
                {"id": "c", "type": "combine", "values": {}, "cache": {"key": "c"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "assembly"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "ta", "in": "shape"},
                {"from": "ta", "out": "shape", "to": "c", "in": "shape"},
                {"from": "b", "out": "shape", "to": "tb", "in": "shape"},
                {"from": "tb", "out": "shape", "to": "c", "in": "shape"},
                {"from": "c", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("assembly.step");
        assert!(path.exists(), "assembly was not written");
        let instances = step_assemblies::planar_instances(path.to_str().unwrap()).unwrap();
        assert_eq!(instances.len(), 2);
        assert!(instances.iter().all(|instance| instance.body.volume() > 0.0));
        let xs: Vec<f64> = instances.iter().map(|instance| instance.body.bounds().unwrap().0[0]).collect();
        assert!(xs.iter().any(|x| (*x - 8.5).abs() < 1e-6), "{xs:?}");
        assert!(xs.iter().any(|x| (*x - -1.5).abs() < 1e-6), "{xs:?}");
    }

    /// Exact feature chain: imported planar BREP -> extrude -> export STEP.
    #[test]
    fn exports_exact_extrude_step() {
        let directory = std::env::temp_dir().join(format!("sn-worker-extrude-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": fixture(BLOCK)}, "cache": {"key": "i"}},
                {"id": "x", "type": "extrude", "values": {"faces": [{"id": "face:0"}], "amount": 5}, "cache": {"key": "x"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "feature"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "i", "out": "shape", "to": "x", "in": "profile"},
                {"from": "x", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let path = directory.join("feature.step");
        assert!(path.exists(), "export was not written");
        let body = nodecraft_step::read_planar_body(path.to_str().unwrap()).unwrap();
        assert!((body.volume() - 60.0).abs() < 1e-6, "volume {}", body.volume());
        let (min, max) = body.bounds().unwrap();
        assert!((min[2] - 2.5).abs() < 1e-6 && (max[2] - 7.5).abs() < 1e-6, "{min:?} {max:?}");
    }

    fn mesh_signed_volume(mesh: &Mesh) -> f64 {
        mesh.faces
            .iter()
            .map(|face| {
                let a = mesh.vertices[face[0] as usize];
                let b = mesh.vertices[face[1] as usize];
                let c = mesh.vertices[face[2] as usize];
                let cross = [
                    f64::from(b[1]) * f64::from(c[2]) - f64::from(b[2]) * f64::from(c[1]),
                    f64::from(b[2]) * f64::from(c[0]) - f64::from(b[0]) * f64::from(c[2]),
                    f64::from(b[0]) * f64::from(c[1]) - f64::from(b[1]) * f64::from(c[0]),
                ];
                (f64::from(a[0]) * cross[0]
                    + f64::from(a[1]) * cross[1]
                    + f64::from(a[2]) * cross[2])
                    / 6.0
            })
            .sum()
    }

    #[test]
    fn part_shell_target_uses_exact_brep() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["s"], "meshTargets": ["s"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "p"}},
                {"id": "s", "type": "shell", "values": {"thickness": 2.0, "direction": "inward"}, "cache": {"key": "s"}}
            ],
            "connections": [{"from": "p", "to": "s", "in": "shape", "out": "shape"}]
        })).expect("request");
        let (response, cache, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["s"].mesh;
        assert!(!mesh.faces.is_empty());
        let (min, max) = nodecraft_kernel::bounds(mesh).unwrap_or(([0.0; 3], [0.0; 3]));
        assert_eq!(min, [-10.0, -10.0, -10.0]);
        assert_eq!(max, [10.0, 10.0, 10.0]);
        assert!(mesh.vertices.len() >= 16, "shell has no inner wall");
        assert!(mesh.faces.len() >= 24, "shell is missing faces");
        assert!(cache.part_entries() > 0);
    }

    #[test]
    fn probe_box_brep_edge_metadata() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["p"], "meshTargets": ["p"],
            "nodes": [{"id":"p","type": "primitive","values":{"primitiveType":"box","length":20,"width":20,"height":20},"cache":{"key":"p"}}],
            "connections": []
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        let mesh = &response.meshes["p"].mesh;
        eprintln!("edges={} kinds={:?}", mesh.brep_edges.len(), mesh.brep_edges.iter().map(|e| (e.id.clone(), e.kind.clone(), e.segments.len())).collect::<Vec<_>>());
    }

    #[test]
    fn shell_rejects_invalid_openings_instead_of_shelling_a_closed_body() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["s"], "meshTargets": ["s"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "p-invalid-shell"}},
                {"id": "s", "type": "shell", "values": {"thickness": 2.0, "direction": "inward", "faces": [{"id": "body:0:face:99"}]}, "cache": {"key": "s-invalid-shell"}}
            ],
            "connections": [{"from": "p", "to": "s", "in": "shape", "out": "shape"}]
        })).expect("invalid shell request");
        let (response, _, _) = respond(request, EngineCache::new());
        let error = response.node_errors.get("s").expect("shell opening error");
        assert!(error.contains("does not exist"), "{error}");

        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 3, "targets": ["s"], "meshTargets": ["s"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "p-bad-body"}},
                {"id": "s", "type": "shell", "values": {"thickness": 2.0, "direction": "inward", "faces": [{"id": "body:4:face:0"}]}, "cache": {"key": "s-bad-body"}}
            ],
            "connections": [{"from": "p", "to": "s", "in": "shape", "out": "shape"}]
        })).expect("bad body shell request");
        let (response, _, _) = respond(request, EngineCache::new());
        let error = response.node_errors.get("s").expect("shell body error");
        assert!(error.contains("names body 4"), "{error}");
    }

    #[test]
    fn shell_resolves_body_scoped_face_references() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["s"], "meshTargets": ["s"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}, "cache": {"key": "p"}},
                {"id": "s", "type": "shell", "values": {"thickness": 2.0, "direction": "inward", "faces": [{"id": "body:0:face:1"}]}, "cache": {"key": "s"}}
            ],
            "connections": [{"from": "p", "to": "s", "in": "shape", "out": "shape"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["s"].mesh.faces.is_empty());
        assert_eq!(face_reference_body_and_index("body:3:face:7"), Some((Some(3), 7)));
        assert_eq!(face_reference_body_and_index("face:2"), Some((None, 2)));
    }

    #[test]
    fn part_scatter_target_uses_placed_geometry_instances() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["s"], "meshTargets": ["s"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 4, "width": 4, "height": 4}, "cache": {"key": "p"}},
                {"id": "s", "type": "scatter", "values": {"count": 4.0, "distribution": "grid", "spanX": 20.0, "spanY": 20.0, "minScale": 0.5, "maxScale": 1.0}, "cache": {"key": "s"}}
            ],
            "connections": [{"from": "p", "to": "s", "in": "shape", "out": "shape"}]
        })).expect("request");
        let (response, cache, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        assert!(!response.meshes["s"].mesh.faces.is_empty());
        assert!(cache.part_entries() > 0);
    }

    #[test]
    fn part_sketch_target_uses_geometry_container() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["t"], "meshTargets": ["t"],
            "nodes": [
                {"id": "s", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "s"}},
                {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t"}}
            ],
            "connections": [{"from": "s", "to": "t", "in": "shape", "out": "sketch"}]
        })).expect("request");
        let (response, cache, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["t"].mesh;
        assert!(!mesh.faces.is_empty(), "sketch geometry did not render");
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!(min[0] >= -0.1 && max[0] >= 19.9, "sketch translation missing: {min:?}..{max:?}");
        assert!(cache.part_entries() > 0, "sketch was not represented as a Geometry Part");
    }

    #[test]
    fn part_extrude_target_uses_exact_brep() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["x"],
            "meshTargets": ["x"],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": fixture(BLOCK)}, "cache": {"key": "i"}},
                {"id": "x", "type": "extrude", "values": {"faces": [{"id": "face:0"}], "amount": 5.0}, "cache": {"key": "x"}}
            ],
            "connections": [{"from": "i", "out": "shape", "to": "x", "in": "profile"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["x"].mesh;
        assert!(!mesh.faces.is_empty());
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[2] - 2.5).abs() < 1e-3 && (max[2] - 7.5).abs() < 1e-3, "{min:?} {max:?}");
    }

    #[test]
    fn part_extrude_join_uses_the_source_part() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["x"],
            "meshTargets": ["x"],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": fixture(BLOCK)}, "cache": {"key": "i"}},
                {"id": "x", "type": "extrude", "values": {"operation": "join", "faces": [{"id": "face:0"}], "amount": 5.0}, "cache": {"key": "x"}}
            ],
            "connections": [{"from": "i", "out": "shape", "to": "x", "in": "profile"}]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["x"].mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[2] + 2.5).abs() < 1e-3 && (max[2] - 7.5).abs() < 1e-3, "{min:?} {max:?}");
    }

    /// A curved (non-planar) B-spline sheet imports through the sheet reader branch of
    /// the plug-in. Using a curved patch proves the planar fallback did not swallow it.
    #[test]
    fn curved_bspline_sheet_imports_through_solidcore_sheet_reader() {
        let body = "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,0.5,0.25));\n\
             #3=CARTESIAN_POINT('',(0.,1.,1.));\n\
             #4=CARTESIAN_POINT('',(0.5,0.,0.25));\n\
             #5=CARTESIAN_POINT('',(0.5,0.5,0.5));\n\
             #6=CARTESIAN_POINT('',(0.5,1.,1.25));\n\
             #7=CARTESIAN_POINT('',(1.,0.,1.));\n\
             #8=CARTESIAN_POINT('',(1.,0.5,1.25));\n\
             #9=CARTESIAN_POINT('',(1.,1.,2.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',2,2,((#1,#2,#3),(#4,#5,#6),(#7,#8,#9)),\
                  .UNSPECIFIED.,.F.,.F.,.F.,(3,3),(3,3),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
             #20=B_SPLINE_CURVE_WITH_KNOTS('',2,(#1,#4,#7),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.);\n\
             #21=B_SPLINE_CURVE_WITH_KNOTS('',2,(#7,#8,#9),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.);\n\
             #22=B_SPLINE_CURVE_WITH_KNOTS('',2,(#3,#6,#9),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.);\n\
             #23=B_SPLINE_CURVE_WITH_KNOTS('',2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.);\n\
             #30=VERTEX_POINT('',#1);\n\
             #31=VERTEX_POINT('',#7);\n\
             #32=VERTEX_POINT('',#9);\n\
             #33=VERTEX_POINT('',#3);\n\
             #40=EDGE_CURVE('',#30,#31,#20,.T.);\n\
             #41=EDGE_CURVE('',#31,#32,#21,.T.);\n\
             #42=EDGE_CURVE('',#33,#32,#22,.T.);\n\
             #43=EDGE_CURVE('',#30,#33,#23,.T.);\n\
             #50=ORIENTED_EDGE('',*,*,#40,.T.);\n\
             #51=ORIENTED_EDGE('',*,*,#41,.T.);\n\
             #52=ORIENTED_EDGE('',*,*,#42,.F.);\n\
             #53=ORIENTED_EDGE('',*,*,#43,.F.);\n\
             #60=EDGE_LOOP('',(#50,#51,#52,#53));\n\
             #61=FACE_OUTER_BOUND('',#60,.T.);\n\
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n";
        let text = format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'nodecraft-worker-test','nodecraft-worker-test','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{body}ENDSEC;\nEND-ISO-10303-21;\n"
        );
        let path = std::env::temp_dir().join(format!(
            "nodecraft-worker-sheet-{}.step",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let result = import_step(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let import = result.expect("curved B-spline sheet import");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 1);
        assert_eq!(import.diagnostics.brep_edges, 4);
        assert_eq!(import.diagnostics.brep_vertices, 4);
        assert!(!import.mesh.faces.is_empty(), "sheet produced no triangles");
        assert!(import.diagnostics.fallbacks.is_empty(), "{:?}", import.diagnostics.fallbacks);
        let (min, max) = bounds(&import.mesh);
        assert!(max[2] - min[2] > 0.5, "sheet stayed flat: {min:?}..{max:?}");
    }

    /// B-spline solids with the `_KNOTS` knot-spec spelling import through the
    /// explicit display-mesh fallback, with warnings instead of a partial exact Brep.
    #[test]
    fn bspline_solid_with_suffixed_knot_spec_imports() {
        let path = fixture("tests/Sample AP203 Data Files/piston.stp");
        let import = import_step(&path).expect("piston.stp B-spline solid import");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert!(!import.mesh.faces.is_empty(), "piston produced no triangles");

    }

    /// Analytic primitive STEP must go through `brep_from_analytic_step` + solidcore
    /// tessellation, not the legacy centered curved mesh.
    #[test]
    fn imports_analytic_cylinder_through_exact_brep() {
        let path = std::env::temp_dir().join(format!("nodecraft-worker-cylinder-{}.step", std::process::id()));
        nodecraft_step::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).expect("write cylinder STEP");
        let import = import_step(path.to_str().unwrap()).expect("import cylinder STEP");
        let _ = std::fs::remove_file(&path);
        assert!(!import.mesh.faces.is_empty(), "cylinder produced no triangles");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 3);
        assert!(import.diagnostics.fallbacks.is_empty(), "analytic fallbacks: {:?}", import.diagnostics.fallbacks);
        assert_bounds(&import.mesh, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], 1e-2);
    }

    /// The diagnostics response field is opt-in: a normal response has no diagnostics
    /// and the typed `Response` is unchanged; a debug request gets the Brep summary.
    #[test]
    fn stepimporter_diagnostics_are_opt_in() {
        let path = fixture(BLOCK);
        let plain = request_for(&path, false);
        assert!(!diagnostics_requested(&plain));
        let (response, _cache, diagnostics) = respond(plain, EngineCache::new());
        assert!(diagnostics.contains_key("step"), "diagnostics are computed but not emitted");
        assert!(response.node_errors.is_empty(), "node errors: {:?}", response.node_errors);
        assert!(response.empty_targets.is_empty());
        assert!(response.meshes.get("step").is_some(), "no mesh payload");
        // The typed response remains diagnostics-free; only main() adds the debug object.
        let payload = serde_json::to_value(&response).expect("response json");
        assert!(payload.get("diagnostics").is_none());

        let debug = request_for(&path, true);
        assert!(diagnostics_requested(&debug));
        let (response, _cache, diagnostics) = respond(debug, EngineCache::new());
        assert!(response.node_errors.is_empty(), "node errors: {:?}", response.node_errors);
        let summarised = diagnostics.get("step").expect("diagnostics for stepimporter node");
        assert_eq!(summarised.kind, "solidcore-brep");
        assert_eq!(summarised.breps, 1);
        assert_eq!(summarised.brep_faces, 6);
        assert!(summarised.fallbacks.is_empty());
        let json = summarised.to_json();
        assert_eq!(json["kind"], "solidcore-brep");
        assert_eq!(json["brepFaces"], 6);
        // The STEP reader's diagnostics report its own import mesh; the display
        // mesh now comes from solidcore-tess, so only require the reported
        // diagnostic count to be present, not equal to the new mesher's count.
        assert!(json["meshTriangles"].as_u64().unwrap_or(0) > 0);
    }
    #[test]
    fn cache_evicts_oldest_mesh_over_budget() {
        let mut cache = EngineCache::with_limits(1, 1);
        cache.insert_mesh(("a".into(), "a".into()), Mesh::default());
        cache.insert_mesh(("b".into(), "b".into()), Mesh::default());
        assert_eq!(cache.mesh_entries(), 1);
        assert!(cache.mesh(&("a".into(), "a".into())).is_none());
        assert!(cache.mesh(&("b".into(), "b".into())).is_some());
    }

    #[test]
    fn step_import_error_names_the_file_path() {
        let path = "/tmp/nodecraft-missing-stepio-file.step";
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["s"],
            "meshTargets": ["s"],
            "nodes": [{"id": "s", "type": "stepimporter", "values": {"path": path}, "cache": {"key": "s", "state": "cached"}}],
            "connections": []
        })).expect("request");
        let response = evaluate(request, EngineCache::new(), None).response;
        let error = response.node_errors.get("s").expect("import error");
        assert!(error.contains(path), "{error}");
    }

    #[test]
    fn step_import_is_reused_by_path_and_file_identity() {
        let path = fixture("tests/Sample AP203 Data Files/block.stp");
        let make = |request_id: u64, key: &str| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": request_id,
                "targets": ["s"],
                "meshTargets": ["s"],
                "nodes": [{"id": "s", "type": "stepimporter", "values": {"path": path}, "cache": {"key": key}}],
                "connections": []
            })).expect("request")
        };
        let first = evaluate(make(1, "s1"), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let second = evaluate(make(2, "s2"), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.import_hits() > 0);
    }

    #[test]
    fn loft_uses_the_solidcore_skin_builder() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["loft"],
            "meshTargets": ["loft"],
            "nodes": [
                {"id": "bottom", "type": "primitive", "values": {"primitiveType": "box", "length": 40, "width": 40, "height": 10}, "cache": {"key": "bottom", "state": "cached"}},
                {"id": "top", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 10}, "cache": {"key": "top", "state": "cached"}},
                {"id": "bottom-at", "type": "translate", "values": {"z": 0}, "cache": {"key": "bottom-at", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 20}, "cache": {"key": "top-at", "state": "cached"}},
                {"id": "loft", "type": "loft", "values": {"ruled": 1}, "cache": {"key": "loft", "state": "cached"}}
            ],
            "connections": [
                {"from": "bottom", "to": "bottom-at", "in": "shape", "out": "shape"},
                {"from": "top", "to": "top-at", "in": "shape", "out": "shape"},
                {"from": "bottom-at", "to": "loft", "in": "bottom", "out": "shape"},
                {"from": "top-at", "to": "loft", "in": "top", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("loft").expect("loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
    }

    #[test]
    fn loft_between_sketch_profiles_uses_solidcore() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let values = serde_json::json!({ "sketch": serde_json::to_value(sketch).unwrap() });
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["loft"],
            "meshTargets": ["loft"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": values, "cache": {"key": "sketch", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 10}, "cache": {"key": "top-at", "state": "cached"}},
                {"id": "loft", "type": "loft", "values": {"ruled": 1}, "cache": {"key": "loft", "state": "cached"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"},
                {"from": "sketch", "to": "loft", "in": "bottom", "out": "sketch"},
                {"from": "top-at", "to": "loft", "in": "top", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("loft").expect("loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 10.0).abs() < 0.05, "loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn spline_generator_feeds_surface_loft() {
        let points = serde_json::json!([
            {"x": -40.0, "y": 0.0},
            {"x": -15.0, "y": 25.0},
            {"x": 15.0, "y": -25.0},
            {"x": 40.0, "y": 0.0}
        ]);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["loft", "curve"],
            "meshTargets": ["loft", "curve"],
            "nodes": [
                {"id": "curve", "type": "spline", "values": {"points": points, "degree": 3, "control": false, "periodic": false}, "cache": {"key": "curve", "state": "cached"}},
                {"id": "top", "type": "translate", "values": {"z": 10}, "cache": {"key": "top", "state": "cached"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1}, "cache": {"key": "loft", "state": "cached"}}
            ],
            "connections": [
                {"from": "curve", "to": "top", "in": "shape", "out": "sketch"},
                {"from": "curve", "to": "loft", "in": "sections", "out": "sketch"},
                {"from": "top", "to": "loft", "in": "sections", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let curve = output.response.meshes.get("curve").expect("spline display mesh");
        assert!(!curve.mesh.faces.is_empty(), "spline display curve is empty");
        let mesh = output.response.meshes.get("loft").expect("loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 10.0).abs() < 0.1, "spline loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn surface_loft_has_no_end_caps() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let values = serde_json::json!({ "sketch": serde_json::to_value(sketch).unwrap() });
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["loft"],
            "meshTargets": ["loft"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": values, "cache": {"key": "sketch", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 10}, "cache": {"key": "top-at", "state": "cached"}},
                {"id": "top2-at", "type": "translate", "values": {"z": 20}, "cache": {"key": "top2-at", "state": "cached"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1}, "cache": {"key": "loft", "state": "cached"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"},
                {"from": "sketch", "to": "top2-at", "in": "shape", "out": "sketch"},
                {"from": "sketch", "to": "loft", "in": "sections", "out": "sketch"},
                {"from": "top-at", "to": "loft", "in": "sections", "out": "shape"},
                {"from": "top2-at", "to": "loft", "in": "sections", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("loft").expect("surface loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 20.0).abs() < 0.05, "surface loft height {}..{}", min_z, max_z);
        for triangle in &mesh.mesh.faces {
            let points: Vec<[f32; 3]> = triangle
                .iter()
                .filter_map(|index| mesh.mesh.vertices.get(*index as usize).copied())
                .collect();
            let [p0, p1, p2] = [points[0], points[1], points[2]];
            let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let nz = u[0] * v[1] - u[1] * v[0];
            let length = (nz * nz
                + (u[1] * v[2] - u[2] * v[1]).powi(2)
                + (u[2] * v[0] - u[0] * v[2]).powi(2))
            .sqrt();
            assert!(
                length <= 1.0e-9 || nz.abs() / length < 0.99,
                "surface loft has a horizontal end cap"
            );
        }
    }

    #[test]
    fn surface_loft_accepts_open_spline_sections() {
        let mut sketch = crate::sketch::Sketch::new();
        for (x, y) in [(-10.0, 0.0), (-2.0, 6.0), (4.0, -3.0), (12.0, 2.0)] {
            sketch.add_point(solidcore_record_new::Vec2::new(x, y), None).unwrap();
        }
        sketch
            .add_curve(
                crate::sketch::CurveKind::Spline {
                    pts: vec![1, 2, 3, 4],
                    control: false,
                    degree: 3,
                    periodic: false,
                    nurbs: None,
                },
                None,
            )
            .unwrap();
        let values = serde_json::json!({ "sketch": serde_json::to_value(sketch).unwrap() });
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["loft"],
            "meshTargets": ["loft"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": values, "cache": {"key": "sketch", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 10}, "cache": {"key": "top-at", "state": "cached"}},
                {"id": "loft", "type": "surfaceloft", "values": {"ruled": 1}, "cache": {"key": "loft", "state": "cached"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"},
                {"from": "sketch", "to": "loft", "in": "sections", "out": "sketch"},
                {"from": "top-at", "to": "loft", "in": "sections", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("loft").expect("surface loft mesh");
        assert!(!mesh.mesh.faces.is_empty(), "open spline loft produced no faces");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 10.0).abs() < 0.1, "surface loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn sketch_output_carries_translate_rotation() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["sketch"],
            "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch).unwrap()}, "cache": {"key": "sketch", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 7.19, "rz": 90}, "cache": {"key": "top-at", "state": "cached"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"}
            ]
        })).expect("request");
        let mut eval = Eval::new(&request, None);
        assert!(eval.evaluate("sketch").is_some());
        let output = eval.sketch_output_for("top-at").expect("typed sketch output");
        let point = output.point(10.0, 0.0);
        assert!(point[0].abs() < 1.0e-9, "{point:?}");
        assert!((point[1] - 10.0).abs() < 1.0e-9, "{point:?}");
        assert!((point[2] - 7.19).abs() < 1.0e-9, "{point:?}");
    }

    #[test]
    fn loft_after_translate_edit_keeps_typed_sketch_output() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let sketch_value = serde_json::to_value(sketch).unwrap();
        let request = |z: f64, top_key: &str, loft_key: &str| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["loft"],
                "meshTargets": ["loft"],
                "nodes": [
                    {"id": "sketch", "type": "sketch", "values": {"sketch": sketch_value.clone()}, "cache": {"key": "sketch", "state": "cached"}},
                    {"id": "top-at", "type": "translate", "values": {"z": z}, "cache": {"key": top_key, "state": "cached"}},
                    {"id": "loft", "type": "loft", "values": {"ruled": 1}, "cache": {"key": loft_key, "state": "cached"}}
                ],
                "connections": [
                    {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"},
                    {"from": "sketch", "to": "loft", "in": "bottom", "out": "sketch"},
                    {"from": "top-at", "to": "loft", "in": "top", "out": "shape"}
                ]
            })).expect("request")
        };
        let first = evaluate(request(10.0, "top-at", "loft"), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let second = evaluate(request(20.0, "top-at-v2", "loft-v2"), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        let mesh = second.response.meshes.get("loft").expect("loft mesh");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 20.0).abs() < 0.05, "edited loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn zero_extrude_amount_is_rejected() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record_new::Vec2::new(-1.0, 0.0), 51.960239979110156, None, None)
            .unwrap();
        let error = sketch_extrude_brep(&SketchOutput::new(sketch), 0.0).expect_err("zero amount");
        assert!(error.contains("greater than zero"), "{error}");
    }

    #[test]
    fn extrude_consumes_a_circle_sketch_profile() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record_new::Vec2::new(1.0, 1.0), 33.18108250839226, None, None)
            .unwrap();
        let brep = sketch_extrude_brep(&SketchOutput::new(sketch), 20.0)
            .expect("circle extrude brep");
        assert!(!brep.faces().ids().collect::<Vec<_>>().is_empty(), "circle extrude has no faces");
    }

    #[test]
    fn circle_extrude_geometry_is_a_cylinder() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record_new::Vec2::new(-1.0, 0.0), 51.960239979110156, None, None)
            .unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["extrude"],
            "meshTargets": ["extrude"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 6.7, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [{"from": "sketch", "to": "extrude", "in": "profile", "out": "sketch"}]
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("extrude").expect("extrude mesh");
        let volume = mesh_signed_volume(&mesh.mesh);
        let expected = std::f64::consts::PI * 51.960239979110156_f64.powi(2) * 6.7;
        let (min, max) = nodecraft_kernel::bounds(&mesh.mesh).expect("bounds");
        let mut side_radial_error = 0.0_f64;
        for vertex in &mesh.mesh.vertices {
            if vertex[2] > 1.0e-4 && vertex[2] < 6.7 - 1.0e-4 {
                let dx = f64::from(vertex[0]) + 1.0;
                let dy = f64::from(vertex[1]);
                side_radial_error = side_radial_error.max(
                    ((dx * dx + dy * dy).sqrt() - 51.960239979110156).abs(),
                );
            }
        }
        assert!((volume - expected).abs() / expected < 0.02, "volume {volume} expected {expected}");
        assert!(side_radial_error < 1.0e-3, "side radius error {side_radial_error}");
        assert!((min[0] + 52.960239979110156).abs() < 0.1, "{min:?}");
        assert!((max[0] - 50.960239979110156).abs() < 0.1, "{max:?}");
        assert!(min[2].abs() < 0.1 && (max[2] - 6.7).abs() < 0.1, "{min:?} {max:?}");
    }

    #[test]
    fn extrude_consumes_a_closed_sketch_profile() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["extrude"],
            "meshTargets": ["extrude"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 20.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "sketch", "to": "extrude", "in": "profile", "out": "sketch"}
            ]
        })).expect("request");
        let brep = sketch_extrude_brep(&SketchOutput::new(sketch), 20.0).expect("extrude brep");
        assert_eq!(
            (brep.faces().ids().count(), brep.edges().ids().count(), brep.vertices().ids().count()),
            (6, 12, 8),
            "box topology"
        );
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = output.response.meshes.get("extrude").expect("extrude mesh");
        assert!(!mesh.mesh.faces.is_empty(), "extrude produced no faces");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 20.0).abs() < 0.05, "height {}..{}", min_z, max_z);
    }

    #[test]
    fn sketch_target_does_not_emit_a_3d_display_mesh() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["sketch"],
            "meshTargets": ["sketch"],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch).unwrap()}, "cache": {"key": "sketch"}}
            ],
            "connections": []
        })).expect("request");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        if let Some(mesh) = output.response.meshes.get("sketch") {
            assert!(mesh.mesh.faces.is_empty(), "sketch must not display as a 3D mesh");
            assert!(mesh.mesh.vertices.is_empty(), "sketch must not display as a 3D mesh");
        }
    }

    #[test]
    fn sketch_output_is_typed_cached_and_transformed() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["sketch"],
            "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch).unwrap()}, "cache": {"key": "sketch", "state": "cached"}},
                {"id": "top-at", "type": "translate", "values": {"z": 7.19}, "cache": {"key": "top-at", "state": "cached"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top-at", "in": "shape", "out": "sketch"}
            ]
        })).expect("request");
        let mut eval = Eval::new(&request, None);
        assert!(eval.evaluate("sketch").is_some());
        assert!(eval.sketches.contains_key("sketch"));
        let output = eval.sketch_output_for("top-at").expect("typed sketch output");
        assert_eq!(output.profiles.len(), 1);
        assert!((output.offset[2] - 7.19).abs() < 1.0e-9);
    }

    #[test]
    fn loft_profile_sampling_keeps_arc_curvature() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch.add_circle(solidcore_record_new::Vec2::new(0.0, 0.0), 10.0, None, None).unwrap();
        let points = sketch_profile_points(&SketchOutput::new(sketch)).expect("profile");
        assert!(points.len() > 64);
        for (x, y, _) in points {
            let radius = (x * x + y * y).sqrt();
            assert!((radius - 10.0).abs() < 0.02, "arc chord point radius {radius}");
        }
    }

    #[test]
    fn exports_boolean_of_sketch_loft_exact() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record_new::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record_new::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record_new::Vec2::new(-10.0, 10.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch.add_line_pts(b, c, None).unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        let directory = std::env::temp_dir().join(format!("sn-loft-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("loft.step");
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["exporter"],
            "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "top", "type": "translate", "values": {"z": 10}, "cache": {"key": "top"}},
                {"id": "loft", "type": "loft", "values": {"ruled": 1}, "cache": {"key": "loft"}},
                {"id": "bore", "type": "primitive", "values": {"primitiveType": "cylinder", "radius": 3, "height": 20}, "cache": {"key": "bore"}},
                {"id": "bore-at", "type": "translate", "values": {"z": 5}, "cache": {"key": "bore-at"}},
                {"id": "cut", "type": "boolean", "values": {"mode": "subtract"}, "cache": {"key": "cut"}},
                {"id": "exporter", "type": "exporter", "values": {"fileName": "loft", "directory": directory.to_string_lossy(), "overwrite": true}, "cache": {"key": "exporter"}}
            ],
            "connections": [
                {"from": "sketch", "to": "top", "in": "shape", "out": "sketch"},
                {"from": "sketch", "to": "loft", "in": "bottom", "out": "sketch"},
                {"from": "top", "to": "loft", "in": "top", "out": "shape"},
                {"from": "bore", "to": "bore-at", "in": "shape", "out": "shape"},
                {"from": "loft", "to": "cut", "in": "a", "out": "shape"},
                {"from": "bore-at", "to": "cut", "in": "b", "out": "shape"},
                {"from": "cut", "to": "exporter", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let first = evaluate(request.clone(), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        assert!(path.is_file(), "loft export was not written");
        assert!(
            nodecraft_mesh::import_step_brep(path.to_str().unwrap()).is_ok(),
            "first export did not round-trip"
        );
        let _ = std::fs::remove_file(&path);
        let mut second_request = request.clone();
        for node in &mut second_request.nodes {
            if node.id == "exporter" {
                node.cache = Some(nodecraft_protocol::CacheState { key: Some("exporter-v2".into()), state: Some("cached".into()) });
            }
        }
        let second = evaluate(second_request, first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(path.is_file(), "cached loft export was not written: {:?}", second.response.node_errors);
        assert!(
            nodecraft_mesh::import_step_brep(path.to_str().unwrap()).is_ok(),
            "cached export did not round-trip"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn deterministic_failure_is_reused_from_cache() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["bad"],
            "meshTargets": [],
            "nodes": [{"id": "bad", "type": "unsupported-test-node", "values": {}, "cache": {"key": "bad", "state": "cached"}}],
            "connections": []
        })).expect("request");
        let first = evaluate(request.clone(), EngineCache::new(), None);
        assert!(first.response.node_errors.contains_key("bad"));
        assert_eq!(first.cache.failure_entries(), 1);
        let second = evaluate(request, first.cache, None);
        assert!(second.response.node_errors.contains_key("bad"));
        assert!(second.cache.failure_hits() > 0);
    }

}
