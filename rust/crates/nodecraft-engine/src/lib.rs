//! NodeCraft evaluation engine.
//!
//! The engine is synchronous and thread-agnostic: the worker owns the engine
//! thread, request/output queues, and JSON transport. Progress is reported through
//! a typed callback so the engine never performs I/O.

use base64::Engine as _;
#[cfg(test)]
use nodecraft_kernel::Kernel;
use nodecraft_kernel::ProceduralKernel;
mod cache;
mod display_eval;
mod exact_eval;
pub mod import_io;
pub use import_io::ImportContext;

pub use cache::EngineCache;
use cache::import_cache_key;
mod brep;
mod look;
#[cfg(test)]
use brep::copy_brep;
use brep::snapshot_brep;
#[cfg(test)]
use look::apply_brep_face_colors;
mod bypass;
mod face_ref;
mod request_index;
pub(crate) use request_index::RequestIndex;
#[doc(hidden)]
pub mod bench_support;
pub mod proof_queue;
use bypass::{bypass_source, suppression_source};

#[cfg(test)]
use nodecraft_mesh::import_step;
use nodecraft_protocol::{CacheStats, Connection, Mesh, MeshPayload, Node, Request, Response};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Shared SolidCore kernel constructor for worker threads.
#[must_use]
pub fn nodecraft_solidcore_kernel() -> nodecraft_solidcore::solidcore::kernel::Kernel {
    nodecraft_solidcore::kernel::kernel()
}

pub mod display;
pub mod kernel_pool;
/// Upstream SolidCraft sketch model: points, curves, constraints, solver and
/// profile detection. This replaces the retired JS/Python sketch implementation.
mod node_task;
pub mod scheduler;
pub mod tessellation;
pub mod thread_pools;
pub mod workload;
pub use kernel_pool::CancelToken;

static WORKLOAD_PROFILER_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Enable or disable analysis-driven dispatch. Default is enabled.
/// Configure the opt-in SolidCore kernel fixes used by new worker kernels.
pub fn configure_solidcore_fixes(enabled: bool) {
    nodecraft_solidcore::kernel::configure_experimental_fixes(enabled);
    // Persistent worker kernels must rebuild with the new profile before their
    // next job; cache keys carry the profile generation separately.
    kernel_pool::shared_exact_pool().reset_caches();
    tessellation::shared_tessellation().reset_caches();
    let names = if enabled { nodecraft_solidcore::kernel::fix_names() } else { Vec::new() };
    let fixes: &[&str] = &names;
    if let Err(error) = nodecraft_layers::configure_session_fixes(fixes) {
        eprintln!("{error}");
    }
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
pub mod nodes;

pub mod sketch {
    pub use solidcraft_sketch::*;
}

fn source<'a>(index: &RequestIndex<'a>, node_id: &str) -> Option<&'a Connection> {
    index.incoming(node_id).next()
}

/// Read a contract field through the one resolver. The `expect` paths are
/// contract-programming errors: every caller below names a node type and field
/// the contract defines, so a missing field means the reader and contract tables
/// have diverged.
pub(crate) fn contract_number(node_type: &str, values: &serde_json::Value, key: &str) -> f64 {
    nodecraft_node_contract::resolve_number(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no number field {node_type}.{key}"))
}

pub(crate) fn contract_bool(node_type: &str, values: &serde_json::Value, key: &str) -> bool {
    nodecraft_node_contract::resolve_bool(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no switch field {node_type}.{key}"))
}

pub(crate) fn contract_text<'a>(
    node_type: &str,
    values: &'a serde_json::Value,
    key: &str,
) -> std::borrow::Cow<'a, str> {
    nodecraft_node_contract::resolve_text(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no text field {node_type}.{key}"))
}

/// One translate placement, read through the contract.
fn translate_placement(values: &serde_json::Value) -> ([f64; 3], [f64; 3]) {
    let offset = [
        contract_number("translate", values, "x"),
        contract_number("translate", values, "y"),
        contract_number("translate", values, "z"),
    ];
    let rotation = [
        contract_number("translate", values, "rx"),
        contract_number("translate", values, "ry"),
        contract_number("translate", values, "rz"),
    ];
    (offset, rotation)
}

fn accumulate_translate(
    values: &serde_json::Value,
    offset: &mut [f64; 3],
    rotation: &mut [f64; 3],
) {
    let (translation, placement_rotation) = translate_placement(values);
    for axis in 0..3 {
        offset[axis] += translation[axis];
        rotation[axis] = placement_rotation[axis];
    }
}

#[derive(Clone)]
struct SketchTransform {
    matrix: nodecraft_exact::Affine3,
}

impl SketchTransform {
    fn from_values(values: &serde_json::Value) -> Self {
        // `space` rotates the manipulator frame only. Geometry values remain
        // in the parent frame so switching world/local cannot move the object.
        let (translation, rotation) = translate_placement(values);
        Self { matrix: nodecraft_exact::Affine3::from_place(translation, rotation) }
    }

    fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        self.matrix.apply_point(point)
    }

    fn determinant(&self) -> f64 {
        self.matrix.determinant()
    }
}

#[derive(Clone, Debug)]
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

    /// The sketch's stored plane as the one affine placement value.
    fn matrix(&self) -> nodecraft_exact::Affine3 {
        nodecraft_exact::Affine3::from_linear_translation(self.rotation, self.offset)
    }

    /// Store a composed placement value into the plane fields.
    fn with_affine(&self, matrix: nodecraft_exact::Affine3) -> Self {
        Self {
            sketch: self.sketch.clone(),
            profiles: self.profiles.clone(),
            offset: matrix.translation(),
            rotation: matrix.linear(),
        }
    }

    /// Apply one placement operation to this sketch. Placement, mirror and
    /// scale all use the same composed value as body instances.
    fn transformed(&self, transform: &BodyTransform) -> Self {
        // This operation is applied after the sketch's existing placement.
        self.with_affine(transform.matrix().compose(self.matrix()))
    }

    fn point(&self, x: f64, y: f64) -> [f64; 3] {
        self.matrix().apply_point([x, y, 0.0])
    }

    /// The sketch's exact 2D profile and its right-handed plane.
    ///
    /// This is the one value every 3D consumer of a sketch uses. The plane's
    /// axes are orthonormal and its normal (`x × y`) is the transformed sketch
    /// normal. A uniform scale is baked into the region's positions and radii,
    /// and a reflection is baked in as the matching 2D reflection with the
    /// loops' canonical orientation restored.
    fn record_geometry(&self) -> Result<SketchRecordGeometry, String> {
        let mut profiles = self.profiles.clone();
        profiles.sort_by(|a, b| b.area.total_cmp(&a.area));
        let profile = profiles.into_iter().next().ok_or("Sketch has no closed profile")?;
        let x = matrix_column(&self.rotation, 0);
        let y = matrix_column(&self.rotation, 1);
        let z = matrix_column(&self.rotation, 2);
        let scale = dot3(x, x).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err("Sketch plane is degenerate (zero or non-finite scale)".into());
        }
        let tolerance = 1.0e-9 * scale.max(1.0);
        let same_length = |axis: [f64; 3]| (dot3(axis, axis).sqrt() - scale).abs() <= tolerance;
        if !same_length(y) || !same_length(z) {
            return Err(
                "Sketch plane does not support non-uniform scale: circles would become ellipses"
                    .into(),
            );
        }
        if dot3(x, y).abs() > tolerance * scale
            || dot3(x, z).abs() > tolerance * scale
            || dot3(y, z).abs() > tolerance * scale
        {
            return Err("Sketch plane axes are not orthogonal".into());
        }
        // A reflection makes the plane left-handed. Turn its y axis over so the
        // plane is right-handed and reverse the 2D loops, leaving the same 3D
        // point set with the transformed normal as `x × y`.
        let reflected = self.matrix().determinant() < 0.0;
        let (plane_x, plane_y) = if reflected {
            (unit3(x, scale), neg3(unit3(y, scale)))
        } else {
            (unit3(x, scale), unit3(y, scale))
        };
        let plane = solidcore_record::Plane {
            origin: solidcore_record::Vec3::new(self.offset[0], self.offset[1], self.offset[2]),
            x: solidcore_record::Vec3::new(plane_x[0], plane_x[1], plane_x[2]),
            y: solidcore_record::Vec3::new(plane_y[0], plane_y[1], plane_y[2]),
        };
        Ok(SketchRecordGeometry {
            region: map_sketch_region(&profile.region, scale, reflected),
            plane,
            scale,
        })
    }
}

/// A placed sketch's exact 2D profile and its right-handed plane.
struct SketchRecordGeometry {
    region: solidcore_record::Region2,
    plane: solidcore_record::Plane,
    /// The uniform scale the transform carries, already baked into `region`.
    scale: f64,
}

fn matrix_column(matrix: &[[f64; 3]; 3], column: usize) -> [f64; 3] {
    [matrix[0][column], matrix[1][column], matrix[2][column]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit3(axis: [f64; 3], scale: f64) -> [f64; 3] {
    [axis[0] / scale, axis[1] / scale, axis[2] / scale]
}

fn neg3(axis: [f64; 3]) -> [f64; 3] {
    [-axis[0], -axis[1], -axis[2]]
}

/// Map one exact profile segment through a uniform scale and an optional
/// reflection in the plane's y axis.
fn map_sketch_segment(
    segment: solidcore_record::Seg2,
    scale: f64,
    reflected: bool,
) -> solidcore_record::Seg2 {
    use solidcore_record::Seg2;
    let map = |point: solidcore_record::Vec2| {
        solidcore_record::Vec2::new(
            point.x * scale,
            if reflected { -point.y * scale } else { point.y * scale },
        )
    };
    match segment {
        Seg2::Line { a, b } => Seg2::Line { a: map(a), b: map(b) },
        Seg2::Arc { center, radius, start, sweep } => Seg2::Arc {
            center: map(center),
            radius: radius * scale,
            start: if reflected { -start } else { start },
            sweep: if reflected { -sweep } else { sweep },
        },
        Seg2::Cubic { p0, p1, p2, p3 } => {
            Seg2::Cubic { p0: map(p0), p1: map(p1), p2: map(p2), p3: map(p3) }
        }
        Seg2::Conic { a, apex, b, w } => Seg2::Conic { a: map(a), apex: map(apex), b: map(b), w },
        Seg2::Bezier { n, p } => Seg2::Bezier { n, p: p.map(map) },
    }
}

/// Reverse one exact profile segment.
fn reverse_sketch_segment(segment: solidcore_record::Seg2) -> solidcore_record::Seg2 {
    use solidcore_record::Seg2;
    match segment {
        Seg2::Line { a, b } => Seg2::Line { a: b, b: a },
        Seg2::Arc { center, radius, start, sweep } => {
            Seg2::Arc { center, radius, start: start + sweep, sweep: -sweep }
        }
        Seg2::Cubic { p0, p1, p2, p3 } => Seg2::Cubic { p0: p3, p1: p2, p2: p1, p3: p0 },
        Seg2::Conic { a, apex, b, w } => Seg2::Conic { a: b, apex, b: a, w },
        Seg2::Bezier { n, mut p } => {
            p[..=n as usize].reverse();
            Seg2::Bezier { n, p }
        }
    }
}

/// A reflected loop's segments are traversed the other way so the region keeps
/// the canonical orientation (outer counter-clockwise, holes clockwise).
fn map_sketch_loop(
    loop2: &solidcore_record::Loop2,
    scale: f64,
    reflected: bool,
) -> solidcore_record::Loop2 {
    let mut segs: Vec<_> = loop2
        .segs
        .iter()
        .copied()
        .map(|segment| map_sketch_segment(segment, scale, reflected))
        .collect();
    if reflected {
        segs.reverse();
        segs = segs.into_iter().map(reverse_sketch_segment).collect();
    }
    solidcore_record::Loop2 { segs }
}

fn map_sketch_region(
    region: &solidcore_record::Region2,
    scale: f64,
    reflected: bool,
) -> solidcore_record::Region2 {
    solidcore_record::Region2 {
        outer: map_sketch_loop(&region.outer, scale, reflected),
        holes: region.holes.iter().map(|hole| map_sketch_loop(hole, scale, reflected)).collect(),
    }
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

fn sketch_profile_curves(
    output: &SketchOutput,
) -> Result<Vec<nodecraft_solidcore::solidcore::curve::curve::Curve>, String> {
    let geometry = output.record_geometry()?;
    if !geometry.region.holes.is_empty() {
        return Err("Loft sketch profiles with holes are not supported yet".into());
    }
    if geometry.region.outer.segs.is_empty() {
        return Err("Sketch profile is degenerate".into());
    }
    let frame = solidcore_convert::plane(&geometry.plane);
    let kernel = nodecraft_solidcore::kernel::kernel();
    geometry
        .region
        .outer
        .segs
        .iter()
        .map(|segment| {
            let segment = solidcore_convert::seg2(segment);
            nodecraft_solidcore::solidcore::prism::profile::curve_on_frame(
                &kernel, &frame, 0.0, &segment,
            )
            .map_err(|error| error.to_string())
        })
        .collect()
}

/// One loft section as exact 2D segments and their plane, or `None` when the
/// sketch has holes or a segment without an exact sweep curve.
fn sketch_record_section(
    output: &SketchOutput,
) -> Option<(Vec<solidcore_record::Seg2>, solidcore_record::Plane)> {
    let geometry = output.record_geometry().ok()?;
    if !geometry.region.holes.is_empty() {
        return None;
    }
    nodecraft_sweep::section_from_seg2(&geometry.region.outer.segs, &geometry.plane)?;
    Some((geometry.region.outer.segs, geometry.plane))
}

fn sketch_extrude_brep(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    output: &SketchOutput,
    distance: f64,
) -> Result<nodecraft_ops::SolidBody, String> {
    if !distance.is_finite() || distance == 0.0 {
        return Err("Extrude distance must be greater than zero.".into());
    }
    let geometry = output.record_geometry()?;
    // The region has the uniform scale baked in, so the extrusion distance
    // carries it too: the solid is the true image of the untransformed one.
    // The ops session lends this worker's own kernel, so the sketch extrusion
    // runs on the same configured profile as the Boolean tasks.
    nodecraft_ops::OpsSession::new(kernel)
        .extrude_planar_region(
            &geometry.plane,
            &geometry.region,
            0.0,
            distance.abs() * geometry.scale,
        )
        .map_err(|error| format!("Extrude: {error}"))
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
            &[bottom_section, top_section],
            if ruled { 1 } else { 2 },
            cap_ends,
        )
        .map_err(|error| format!("Loft: {error}"))?;
        return nodecraft_layers::body_to_kernel(kernel, &body)
            .map_err(|error| format!("Loft: {error}"));
    }
    if let (Ok(bottom_curves), Ok(top_curves)) =
        (sketch_profile_curves(bottom), sketch_profile_curves(top))
    {
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
    let record_sections: Option<Vec<_>> = sections.iter().map(sketch_record_section).collect();
    if let Some(record_sections) = record_sections {
        let body =
            nodecraft_sweep::loft_on_planes(&record_sections, if ruled { 1 } else { 2 }, cap_ends)
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

fn loft_breps(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    bottom: nodecraft_solidcore::solidcore::topology::Brep,
    top: nodecraft_solidcore::solidcore::topology::Brep,
    ruled: bool,
    cap_ends: bool,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let bottom_face =
        loft_face(kernel, &bottom, true).ok_or("Loft bottom has no upward planar face")?;
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
    let top_curves =
        top_loop.oriented_curves(kernel).map_err(|error| format!("Loft top curves: {error}"))?;
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

/// Boolean two B-reps through the upstream ops layer, with an empty ops
/// result represented as an empty B-rep (the graph value contract).
pub(crate) fn boolean_breps_ops_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    a: &nodecraft_solidcore::solidcore::topology::Brep,
    b: &nodecraft_solidcore::solidcore::topology::Brep,
    op: nodecraft_bool::BooleanOp,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    Ok(nodecraft_layers::boolean_breps_ops(kernel, a, b, ops_boolean(op))?
        .ok_or_else(|| "Boolean produced no solids".to_owned())?)
}

fn boolean_breps_ops(
    a: &nodecraft_solidcore::solidcore::topology::Brep,
    b: &nodecraft_solidcore::solidcore::topology::Brep,
    op: nodecraft_bool::BooleanOp,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    Ok(nodecraft_layers::boolean_breps_ops(&mut kernel, a, b, ops_boolean(op))?
        .unwrap_or_else(nodecraft_solidcore::solidcore::topology::Brep::new))
}

/// Map NodeCraft's Boolean selector to the upstream ops selector.
fn ops_boolean(op: nodecraft_bool::BooleanOp) -> nodecraft_layers::BooleanOp {
    match op {
        nodecraft_bool::BooleanOp::Union => nodecraft_layers::BooleanOp::Union,
        nodecraft_bool::BooleanOp::Difference => nodecraft_layers::BooleanOp::Subtract,
        nodecraft_bool::BooleanOp::Intersection => nodecraft_layers::BooleanOp::Intersect,
    }
}

pub(crate) fn boolean_mode(
    values: &serde_json::Value,
) -> Result<(nodecraft_bool::BooleanOp, bool), String> {
    let mode = contract_text("boolean", values, "mode");
    let reverse = mode == "subtract-reverse";
    let op = match mode.as_ref() {
        "union" => nodecraft_bool::BooleanOp::Union,
        "subtract" | "difference" | "subtract-a-b" | "subtract-reverse" => {
            nodecraft_bool::BooleanOp::Difference
        }
        "intersect" | "intersection" => nodecraft_bool::BooleanOp::Intersection,
        other => return Err(format!("unsupported Boolean mode {other:?}")),
    };
    Ok((op, reverse))
}

/// Whether this node is a Combine collection (export assembly route).
///
/// Engine policy lives here rather than in a node module: the exporter node
/// asks the resolved `ExportInput`, not the graph.
pub(crate) fn is_combine_node(node: &Node) -> bool {
    node.node_type == "combine"
}

/// Whether this node is a STEP import (imported re-export route).
pub(crate) fn is_stepimporter_node(node: &Node) -> bool {
    node.node_type == "stepimporter"
}

fn resolve_part_chain<'a>(
    index: &RequestIndex<'a>,
    start: &str,
) -> Option<(&'a Node, [f64; 3], [f64; 3])> {
    let mut current = start.to_string();
    let mut offset = [0.0; 3];
    let mut rotation = [0.0; 3];
    for _ in 0..64 {
        let node = index.node(&current)?;
        if crate::nodes::transform_role(crate::nodes::NodeView::from(node))
            == Some(crate::nodes::TransformRole::Translate)
        {
            accumulate_translate(&node.values, &mut offset, &mut rotation);
            let next = index.incoming(&current).next()?;
            current = next.from.clone();
        } else if crate::nodes::export_terminal(&node.node_type) {
            return Some((node, offset, rotation));
        } else {
            return None;
        }
    }
    None
}

fn placement_of(offset: [f64; 3], rotation: [f64; 3]) -> [[f64; 4]; 4] {
    nodecraft_exact::Affine3::from_place(offset, rotation).to_column_major_f64()
}

/// Compose two column-major export placements through the one affine helper.
fn compose_placement(left: [[f64; 4]; 4], right: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    nodecraft_exact::Affine3::from_column_major(left)
        .compose(nodecraft_exact::Affine3::from_column_major(right))
        .to_column_major_f64()
}

fn fnv_seed(id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// Every exact operand is a node's evaluated [`Part`], realised.
///
/// This is the one operand representation. A separate raw-B-rep route per node
/// would be a second implementation of the same geometry; a registered node and
/// an unported one both produce their exact value through `evaluate_part`.
fn exact_operand_bodies(
    eval: &mut Eval<'_, '_>,
    start: &str,
) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
    let node_type = eval
        .index
        .node(start)
        .map(|node| node.node_type.clone())
        .ok_or("Boolean operand node missing")?;
    let part = eval.evaluate_part(start)?;
    let solids = part.realized_bodies()?;
    if solids.is_empty() {
        return Err(format!("{node_type} operand produced no solids"));
    }
    solids.into_iter().map(|body| body.into_brep().map_err(|error| error.to_string())).collect()
}

/// The single exact B-rep of a node's evaluated Part.
///
/// STEP export, assembly members and edge previews all take one operand. A node
/// whose Part has several bodies is not collapsed to its first shape: it keeps
/// the historical "an exact operand needs exactly one body" error.
pub(crate) fn exact_operand_brep(
    eval: &mut Eval<'_, '_>,
    start: &str,
) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let node_type = eval
        .index
        .node(start)
        .map(|node| node.node_type.clone())
        .ok_or("Boolean operand node missing")?;
    let mut bodies = exact_operand_bodies(eval, start)?;
    if bodies.len() != 1 {
        return Err(format!(
            "{node_type} operand produces {} bodies; an exact operand needs exactly one",
            bodies.len()
        ));
    }
    Ok(bodies.pop().expect("one body"))
}

/// Collect every solid of a node's evaluated Part, in Part body order.
fn exact_operand_breps(
    eval: &mut Eval<'_, '_>,
    start: &str,
) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
    exact_operand_bodies(eval, start)
}

/// Exact consumers that need their upstream primitives as BReps during normal
/// evaluation. Mesh-only branches must not pay for this.
fn exact_brep_demands(index: &RequestIndex<'_>) -> HashSet<String> {
    let mut demands = HashSet::new();
    let mut stack: Vec<String> = index
        .nodes()
        .iter()
        .filter(|node| {
            crate::nodes::output_kind(crate::nodes::NodeView::from(*node))
                == crate::nodes::OutputKind::Effect
                || matches!(
                    node.node_type.as_str(),
                    "boolean"
                        | "fuse"
                        | "chamfer"
                        | "fillet"
                        | "shell"
                        | "extrude"
                        | "loft"
                        | "surfaceloft"
                )
        })
        .map(|node| node.id.clone())
        .collect();
    while let Some(id) = stack.pop() {
        if !demands.insert(id.clone()) {
            continue;
        }
        for link in index.incoming(&id) {
            stack.push(link.from.clone());
        }
    }
    demands
}

#[cfg(test)]
fn write_primitive_step(node: &Node, path: &str) -> Result<(), String> {
    let brep = crate::nodes::primitive_brep(node)?;
    nodecraft_step::write_brep_exact(path, &brep)
}

fn mirror_plane(values: &serde_json::Value) -> (&'static str, f64) {
    let plane = contract_text("mirror", values, "plane");
    let offset = contract_number("mirror", values, "offset");
    match plane.as_ref() {
        "xy" => ("xy", offset),
        "yz" => ("yz", offset),
        _ => ("xz", offset),
    }
}

fn mirror_combine(values: &serde_json::Value) -> bool {
    contract_bool("mirror", values, "combine")
}

fn mirror_brep_with_kernel(
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
    brep: &mut nodecraft_solidcore::solidcore::topology::Brep,
    values: &serde_json::Value,
) -> Result<(), String> {
    use nodecraft_solidcore::solidcore::{
        geom::axis2_placement::Axis2Placement,
        math::{Point3, Vec3},
        topology::brep::mirror,
    };
    let (plane, offset) = mirror_plane(values);
    let (origin, x_axis, y_axis) = match plane {
        "xz" => (Point3::new(0.0, offset, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        "yz" => (Point3::new(offset, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        _ => (Point3::new(0.0, 0.0, offset), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    };
    // Mirror needs an independent, owning body rather than a view of the input.
    let mut standalone =
        nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(kernel, brep)
            .map_err(|error| error.to_string())?;
    mirror::mirror(kernel, &mut standalone, &Axis2Placement::new(origin, x_axis, y_axis))
        .map_err(|error| error.to_string())?;
    let mut mn = [f64::INFINITY; 3];
    let mut mx = [f64::NEG_INFINITY; 3];
    for v in standalone.vertex_list() {
        if let Ok(vertex) = standalone.vertex(*v) {
            let p = vertex.point;
            mn[0] = mn[0].min(p.x);
            mn[1] = mn[1].min(p.y);
            mn[2] = mn[2].min(p.z);
            mx[0] = mx[0].max(p.x);
            mx[1] = mx[1].max(p.y);
            mx[2] = mx[2].max(p.z);
        }
    }
    *brep = standalone;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgressPhase {
    Start,
    Complete,
    BrepStart,
    BrepComplete,
    ExactComplete,
}

impl ProgressPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Start => "node-start",
            Self::Complete => "node-complete",
            Self::BrepStart => "brep-start",
            Self::BrepComplete => "brep-complete",
            Self::ExactComplete => "exact-complete",
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

pub(crate) const MAX_BOOLEAN_SOLIDS: usize = 256;

pub struct EngineOutput {
    pub response: Response,
    pub cache: EngineCache,
    /// Test-only compatibility view of the display pass. The protocol response
    /// no longer carries meshes; production display output is built by
    /// `nodecraft-display`.
    #[cfg(test)]
    pub meshes: BTreeMap<String, MeshPayload>,
    pub diagnostics: HashMap<String, Vec<nodes::NodeDiagnostic>>,
    /// Exact Part/BRep inputs for a separate display service. The engine does
    /// not create plans or meshes for these.
    pub exact_parts: BTreeMap<String, nodecraft_exact::ExactPart>,
    /// Planner defects from the display tessellation (lenient/capped/over-tolerance
    /// faces and edges).
    pub tessellation_diagnostics: Vec<String>,
    /// The background proof queue's generation for this evaluation. A later refusal report
    /// carries the same generation and belongs to this result.
    pub proof_generation: u64,
}

#[cfg(test)]
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
            tangent = solidcore_record::Vec2::new(1.0, 0.0);
        }
        let normal =
            solidcore_record::Vec2::new(-tangent.y, tangent.x) * (half_width / tangent.len());
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
            .add_point(solidcore_record::Vec2::new(x, y), None)
            .map_err(|error| error.to_string())?;
        point_ids.push(id);
    }
    let control = contract_bool("spline", values, "control");
    let periodic = contract_bool("spline", values, "periodic");
    let degree = contract_number("spline", values, "degree")
        .round()
        .clamp(1.0, solidcore_record::MAX_BEZIER_DEGREE as f64) as u8;
    sketch
        .add_curve(
            CurveKind::Spline {
                pts: point_ids,
                control,
                degree,
                periodic,
                nurbs: None,
                handles: Vec::new(),
            },
            None,
        )
        .map_err(|error| error.to_string())?;
    Ok(sketch)
}

/// One shared body plus its one composed placement value.
#[derive(Clone, Debug)]
struct BodyInstance {
    body: nodecraft_ops::SolidBody,
    /// The body's placement: one shared chain node with the composed matrix
    /// cached. Cloning it is O(1), so a chain no longer copies 1..N history.
    placement: nodecraft_exact::Placement,
    /// The realised body for exactly this placement, made once and shared by every
    /// reader and by later evaluations of the same cached Part. A new transform resets it.
    realized: std::sync::Arc<std::sync::Mutex<Option<nodecraft_ops::SolidBody>>>,
    /// Content key of the node that produced this geometry. Placement and
    /// scatter preserve it, so every instance of one B-rep shares one display
    /// plan and one resident GPU mesh.
    source: Option<u64>,
    /// Body index in that producing node's output.
    source_body: u32,
    /// The imported body's display name, when its source supplies one.
    name: Option<String>,
    /// The imported body's surface colour (linear RGB), when its source supplies one.
    /// The colour is also written as a face attribute, the same place a Look node
    /// stores colour, so display and export read one value.
    color: Option<[f64; 3]>,
}

/// One placement operation. The engine has no second placement type: a body's
/// composed value is [`nodecraft_exact::Placement`], and this is its local
/// operation for the nodes that append to it.
pub(crate) use nodecraft_exact::DisplayTransform as BodyTransform;

/// Content seed for a node when the request omitted its cache key. It hashes
/// the node's values and upstream content, so raw worker/CLI requests still get
/// a key that changes when the geometry changes.
pub(crate) fn node_content_seed(
    index: &RequestIndex<'_>,
    id: &str,
    visiting: &mut std::collections::HashSet<String>,
) -> u64 {
    if let Some(key) =
        index.node(id).and_then(|node| node.cache.as_ref()).and_then(|cache| cache.key.as_deref())
    {
        return fnv_seed(key);
    }
    if !visiting.insert(id.to_owned()) {
        return fnv_seed(id);
    }
    let mut text = id.to_owned();
    if let Some(node) = index.node(id) {
        text.push_str(&serde_json::to_string(&node.values).unwrap_or_default());
    }
    let mut links: Vec<&nodecraft_protocol::Connection> = index.incoming(id).collect();
    links.sort_by(|a, b| (&a.input, &a.from).cmp(&(&b.input, &b.from)));
    for link in links {
        text.push('|');
        text.push_str(&link.input);
        text.push(':');
        text.push_str(&node_content_seed(index, &link.from, visiting).to_string());
    }
    visiting.remove(id);
    fnv_seed(&text)
}

fn exact_part_from(part: &Part, source: u64) -> nodecraft_exact::ExactPart {
    nodecraft_exact::ExactPart {
        bodies: part
            .bodies
            .iter()
            .enumerate()
            .map(|(index, body)| nodecraft_exact::ExactBody {
                key: nodecraft_exact::BodyKey::new(
                    body.source.unwrap_or(source),
                    body.source.map_or(index as u32, |_| body.source_body),
                ),
                brep: body.body.shared(),
                placement: body.placement.clone(),
            })
            .collect(),
    }
}

#[derive(Clone, Debug)]
struct SketchInstance {
    output: Arc<SketchOutput>,
    /// One composed value, like a body's: the output's own plane is the base
    /// matrix and each appended placement is composed once.
    placement: nodecraft_exact::Placement,
}

impl SketchInstance {
    /// The sketch with its placement baked into its plane transform.
    fn realized(&self) -> SketchOutput {
        self.output.with_affine(*self.placement.matrix())
    }
}

/// Immutable graph value: placed solids plus placed sketch profiles/curves.
#[derive(Clone, Debug)]
struct Part {
    bodies: Vec<BodyInstance>,
    sketches: Vec<SketchInstance>,
}

impl BodyInstance {
    /// Add a placement to this instance and drop any realised body: a realised body belongs
    /// to one exact transform chain.
    fn push_transform(&mut self, transform: BodyTransform) {
        self.placement = self.placement.then(transform);
        self.realized = Default::default();
    }
}

impl Part {
    fn from_brep(brep: nodecraft_solidcore::solidcore::topology::Brep) -> Self {
        Self::from_arc(Arc::new(brep))
    }

    /// Share an immutable exact body between exact evaluation and meshing. Mutating consumers
    /// must take an owning copy at their operation boundary; the [`SolidBody`] keeps the proved
    /// kernel body beside the shared B-rep.
    fn from_arc(brep: Arc<nodecraft_solidcore::solidcore::topology::Brep>) -> Self {
        Self {
            bodies: vec![BodyInstance {
                body: nodecraft_ops::SolidBody::from_arc(brep),
                placement: nodecraft_exact::Placement::identity(),
                realized: Default::default(),
                source: None,
                source_body: 0,
                name: None,
                color: None,
            }],
            sketches: Vec::new(),
        }
    }

    /// A Part built from freshly generated bodies, one instance per body in
    /// order. The constructor owns the provenance/tagging convention: every
    /// instance starts untagged with `source_body` equal to its Part index, so
    /// `evaluate_part`'s tagging pass stamps the producing node exactly as it
    /// did for hand-built `BodyInstance` literals.
    fn from_bodies(bodies: impl IntoIterator<Item = nodecraft_ops::SolidBody>) -> Self {
        Self {
            bodies: bodies
                .into_iter()
                .enumerate()
                .map(|(index, body)| BodyInstance {
                    body,
                    placement: nodecraft_exact::Placement::identity(),
                    realized: Default::default(),
                    source: None,
                    source_body: index as u32,
                    name: None,
                    color: None,
                })
                .collect(),
            sketches: Vec::new(),
        }
    }

    /// A Part with one body per converted CADIR body. Imported names and
    /// colours stay on the instance, and each colour is also written to the
    /// body's faces through the same attribute a Look uses, so later display
    /// and export read it without a second colour path.
    fn from_imported(bodies: Vec<nodecraft_mesh::StepBrepBody>) -> (Self, Vec<String>) {
        let mut warnings = Vec::new();
        let bodies = bodies
            .into_iter()
            .enumerate()
            .map(|(index, imported)| {
                let mut brep = imported.brep;
                if let Some(color) = imported.color
                    && let Err(error) = crate::look::apply_look_color(&mut brep, color)
                {
                    warnings.push(format!("body {:?}: colour: {error}", imported.name));
                }
                BodyInstance {
                    body: nodecraft_ops::SolidBody::new(brep),
                    placement: nodecraft_exact::Placement::identity(),
                    realized: Default::default(),
                    source: None,
                    source_body: index as u32,
                    name: Some(imported.name),
                    color: imported.color,
                }
            })
            .collect();
        (Part { bodies, sketches: Vec::new() }, warnings)
    }

    fn from_sketch(output: Arc<SketchOutput>) -> Self {
        Self {
            bodies: Vec::new(),
            sketches: vec![SketchInstance {
                placement: nodecraft_exact::Placement::from_matrix(
                    nodecraft_exact::Affine3::from_linear_translation(
                        output.rotation,
                        output.offset,
                    ),
                ),
                output,
            }],
        }
    }

    fn empty() -> Self {
        Self { bodies: Vec::new(), sketches: Vec::new() }
    }

    /// Give untagged bodies this node's provenance. Bodies that already carry
    /// an upstream key are left alone.
    fn tagged(mut self, source: u64) -> Self {
        for (index, body) in self.bodies.iter_mut().enumerate() {
            if body.source.is_none() {
                body.source = Some(source);
                body.source_body = index as u32;
            }
        }
        self
    }

    /// Mirror this Part without copying its bodies. The transform is applied
    /// when the Part is realized, through the same `ExactNodeTask::Mirror`
    /// implementation the operand path uses.
    fn mirrored(&self, values: &serde_json::Value) -> Part {
        let (plane, offset) = mirror_plane(values);
        let mut mirrored = self.bodies.clone();
        for body in &mut mirrored {
            body.push_transform(BodyTransform::Mirror { plane, offset });
        }
        let mut sketches = self.sketches.clone();
        for sketch in &mut sketches {
            sketch.placement = sketch.placement.then(BodyTransform::Mirror { plane, offset });
        }
        if !mirror_combine(values) {
            return Part { bodies: mirrored, sketches };
        }
        let mut bodies = self.bodies.clone();
        bodies.extend(mirrored);
        let mut all_sketches = self.sketches.clone();
        all_sketches.extend(sketches);
        Part { bodies, sketches: all_sketches }
    }

    /// Translate/rotate this Part without copying its bodies. The placement is
    /// recorded on each instance and baked only when geometry is realized.
    fn translated(&self, offset: [f64; 3], rotation: [f64; 3]) -> Part {
        let mut bodies = self.bodies.clone();
        for body in &mut bodies {
            body.push_transform(BodyTransform::Place { offset, rotation });
        }
        let mut sketches = self.sketches.clone();
        for sketch in &mut sketches {
            sketch.placement = sketch.placement.then(BodyTransform::Place { offset, rotation });
        }
        Part { bodies, sketches }
    }

    /// The Part's bodies with their placements baked in, made once per instance and shared by
    /// every reader and later evaluation of this cached Part.
    ///
    /// A body with no placement keeps its shared record and its proof. A rigid placement
    /// (translate/rotate) is the same B-rep under the same map applied to every point, curve
    /// and surface: the topology is unchanged and the map has a positive determinant, so the
    /// proof's outward sign still holds and the realised body carries the source's proof
    /// state. A mirror or a scale is a different B-rep, so those reset the state to Unknown;
    /// a mirror can flip the outward sign and a non-uniform scale changes the tolerances.
    fn realized_bodies(&self) -> Result<Vec<nodecraft_ops::SolidBody>, String> {
        let mut bodies = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            if body.placement.is_identity() {
                bodies.push(body.body.clone());
                continue;
            }
            let memoised =
                body.realized.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            if let Some(done) = memoised {
                bodies.push(done);
                continue;
            }
            let realized = crate::node_task::execute_exact(
                crate::node_task::ExactNodeTask::Realize {
                    body: body.body.clone(),
                    placement: body.placement.clone(),
                },
                "part-realize",
                fnv_seed("part-realize"),
            )?
            .into_shape()
            .ok_or_else(|| "realize task returned no shape".to_owned())?;
            if body.placement.is_rigid() {
                realized.carry_proof_from(&body.body);
            }
            *body.realized.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(realized.clone());
            bodies.push(realized);
        }
        Ok(bodies)
    }

    /// Bake placements into standalone BReps for operations that need actual geometry and must
    /// edit it (exact export, fillet/extrude/compare). The proof is dropped with the copy.
    /// Every sketch in this Part, with its placement chain applied. Extrude,
    /// loft and surface loft consume this instead of resolving a sketch by the
    /// upstream node's type.
    fn realized_sketches(&self) -> Vec<SketchOutput> {
        self.sketches.iter().map(SketchInstance::realized).collect()
    }

    fn realized_solids(
        &self,
    ) -> Result<Vec<nodecraft_solidcore::solidcore::topology::Brep>, String> {
        // Always a kernel copy, even with no placement: the mutable consumers (fillet,
        // chamfer, extrude) rely on a body with the kernel's canonical entity IDs. The
        // realisation itself is the memoised one.
        self.realized_bodies()?
            .into_iter()
            .map(|body| body.into_brep().map_err(|error| error.to_string()))
            .collect()
    }
}

struct Eval<'req, 'prog> {
    request: &'req Request,
    /// The one per-evaluation node/connection index. Every by-id and
    /// target/source lookup reads this instead of scanning the request.
    index: RequestIndex<'req>,
    /// Document folder and ingested sources for import path resolution.
    imports: crate::ImportContext,
    kernel: ProceduralKernel,
    objects: HashMap<String, Mesh>,
    cache: EngineCache,
    /// Developer rebuild: skip every cache read for this evaluation.
    bypass_cache: bool,
    /// Exact-B-rep import diagnostics per `stepimporter` node (debug response only).
    diagnostics: HashMap<String, Vec<nodes::NodeDiagnostic>>,
    errors: BTreeMap<String, String>,
    warnings: BTreeMap<String, Vec<String>>,
    order: Vec<String>,
    timings: BTreeMap<String, f64>,
    brep_timings: BTreeMap<String, f64>,
    visiting: Vec<String>,
    cache_hits: u32,
    cache_misses: u32,
    brep_cache_hits: u32,
    part_cache_hits: u32,
    display_mesh_cache_hits: u32,
    breps: HashMap<String, Arc<nodecraft_solidcore::solidcore::topology::Brep>>,
    /// Nodes whose B-rep came from the parallel prewarm scheduler.
    prewarmed: HashSet<String>,
    /// Exact Part results for this evaluation pass.
    parts: HashMap<String, Arc<Part>>,
    /// Union nodes folded into a Union above them: the exact loop must not evaluate them
    /// separately before the run's top fuses them (a requested node is never folded).
    folded_into_union: HashSet<String>,
    part_visiting: Vec<String>,
    /// Nodes whose own evaluation failed but whose output was replaced by a
    /// bypass input. Downstream nodes must not treat these as failed inputs.
    bypassed: HashSet<String>,
    /// Nodes whose exact BRep is an input to an exact consumer. Mesh-only
    /// primitive graphs skip the eager exact-BRep build.
    needs_brep: HashSet<String>,
    sketches: HashMap<String, Arc<SketchOutput>>,
    progress: Option<&'prog mut dyn FnMut(ProgressEvent)>,
    tessellation_diagnostics: Vec<String>,
    /// Display plans for the test-only display pass. Production evaluation is
    /// exact-only; display callers own their plan caches.
    #[cfg(test)]
    display_plans: nodecraft_display::PlanCache,
}

impl<'req, 'prog> Eval<'req, 'prog> {
    fn new(request: &'req Request, progress: Option<&'prog mut dyn FnMut(ProgressEvent)>) -> Self {
        let index = RequestIndex::new(request);
        let folded_into_union = exact_eval::union_run_intermediates(&index);
        let needs_brep = exact_brep_demands(&index);
        Self {
            request,
            index,
            imports: crate::ImportContext::default(),
            kernel: ProceduralKernel,
            objects: HashMap::new(),
            cache: EngineCache::new(),
            bypass_cache: request.bypass_cache,
            diagnostics: HashMap::new(),
            errors: BTreeMap::new(),
            order: Vec::new(),
            timings: BTreeMap::new(),
            brep_timings: BTreeMap::new(),
            visiting: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            brep_cache_hits: 0,
            part_cache_hits: 0,
            display_mesh_cache_hits: 0,
            breps: HashMap::new(),
            prewarmed: HashSet::new(),
            parts: HashMap::new(),
            folded_into_union,
            part_visiting: Vec::new(),
            bypassed: HashSet::new(),
            needs_brep,
            sketches: HashMap::new(),
            progress,
            warnings: BTreeMap::new(),
            tessellation_diagnostics: Vec::new(),
            #[cfg(test)]
            display_plans: nodecraft_display::PlanCache::new(16384),
        }
    }

    /// The one place a sketch output is recorded for downstream lookup.
    fn publish_sketch_output(&mut self, node_id: &str, output: Arc<SketchOutput>) {
        self.sketches.insert(node_id.to_owned(), output);
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
        self.publish_sketch_output(&node.id, output.clone());
        Some(output)
    }

    /// The one sketch a single-profile consumer (prewarm, tests) may read from
    /// an evaluated Part. Body-loft nodes use the Part's bodies instead, and a
    /// Part with several sketches is not a single section.
    #[cfg(test)]
    fn part_sketch_output(&mut self, start: &str) -> Option<SketchOutput> {
        let part = self.evaluate_part(start).ok()?;
        if !part.bodies.is_empty() || part.sketches.len() != 1 {
            return None;
        }
        Some(part.sketches[0].realized())
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
    let mut inputs: HashMap<&str, Vec<&str>> = HashMap::new();
    for link in &request.connections {
        inputs.entry(link.to.as_str()).or_default().push(link.from.as_str());
    }
    let mut seen = HashSet::new();
    let mut stack = vec![target];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.to_owned()) {
            continue;
        }
        if let Some(upstream) = inputs.get(id) {
            stack.extend(upstream.iter().copied());
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
    let node_types: HashMap<&str, &str> =
        request.nodes.iter().map(|node| (node.id.as_str(), node.node_type.as_str())).collect();
    let requested: HashSet<String> = request
        .targets
        .iter()
        .filter(|id| {
            node_types.get(id.as_str()).is_some_and(|node_type| {
                !sketch_is_2d_overlay_only(node_type) && part_supported(node_type)
            }) && ancestors_of(request, id).iter().all(|ancestor| {
                node_types.get(ancestor.as_str()).is_some_and(|node_type| part_supported(node_type))
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
        if !seen.insert(id.to_string()) {
            return;
        }
        if let Some(inputs) = dependencies.get(id) {
            for input in inputs {
                visit(input, dependencies, seen, order);
            }
        }
        order.push(id.to_string());
    }
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for target in &request.targets {
        if ids.contains(target.as_str()) {
            visit(target, &dependencies, &mut seen, &mut order);
        }
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
struct TestResponse {
    response: Response,
    meshes: BTreeMap<String, MeshPayload>,
}

#[cfg(test)]
impl std::ops::Deref for TestResponse {
    type Target = Response;
    fn deref(&self) -> &Response {
        &self.response
    }
}

#[cfg(test)]
fn respond(
    request: Request,
    cache: EngineCache,
) -> (TestResponse, EngineCache, HashMap<String, Vec<nodes::NodeDiagnostic>>) {
    let output = evaluate(request, cache, None);
    (
        TestResponse { response: output.response, meshes: output.meshes },
        output.cache,
        output.diagnostics,
    )
}

/// Evaluate the exact graph result and its display meshes.
#[cfg(test)]
pub fn evaluate(
    request: Request,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
) -> EngineOutput {
    evaluate_mode(request, cache, progress, false, crate::ImportContext::default())
}

/// Evaluate only the exact graph result, leaving display tessellation to the
/// caller. The exact Part cache is filled and can be meshed by a later request.
pub fn evaluate_exact(
    request: Request,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
) -> EngineOutput {
    evaluate_mode(request, cache, progress, true, crate::ImportContext::default())
}

/// Exact evaluation with the document folder and ingested sources needed to
/// resolve an importer's relative path or a missing file's embedded copy.
pub fn evaluate_exact_with_imports(
    request: Request,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
    imports: crate::ImportContext,
) -> EngineOutput {
    evaluate_mode(request, cache, progress, true, imports)
}

/// Prove every body of an exact-export Part (blocking). The `exporter` node and any STEP
/// path must write only bodies that are refused-never; an unknown body is proved first and a
/// refused one becomes an error naming the failed test.
///
/// # Errors
/// The realisation error, or the first refused body's test.
fn prove_part_for_export(
    part: &Part,
    kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel,
) -> Result<(), String> {
    for body in part.realized_bodies()? {
        body.prove(kernel).map_err(|error| format!("export refused: {error}"))?;
    }
    Ok(())
}

fn evaluate_mode(
    request: Request,
    cache: EngineCache,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
    exact_only: bool,
    imports: crate::ImportContext,
) -> EngineOutput {
    let started = Instant::now();
    let bench_total = bench_support::phase_clock();
    struct DisplayToleranceReset;
    impl Drop for DisplayToleranceReset {
        fn drop(&mut self) {
            nodecraft_mesh::set_display_tolerance(None);
        }
    }
    let _tolerance_reset = DisplayToleranceReset;
    nodecraft_mesh::set_display_tolerance(request.mesh_tolerance);
    let (request_id, tab_id, graph_revision) =
        (request.request_id, request.tab_id.clone(), request.graph_revision.clone());
    // Proofs are queued during this pass and start only when the guard drops at return, so
    // the critical path below never waits for one.
    let proof_guard = proof_queue::begin(&request);
    let mut eval = Eval::new(&request, progress);
    eval.imports = imports;
    eval.cache = cache;
    eval.cache.ensure_part_capacity(request.nodes.len());
    // The exact pre-pass warms the target closure before its serial exact loop.
    // For wide graphs this dispatches independent branches across the exact
    // pool; serial graphs are rejected by the workload profile inside prewarm.
    // The display pass consumes the warmed B-reps from the shared cache.
    let bench_prewarm = bench_support::phase_clock();
    if exact_only {
        eval.prewarm_primitives();
    }
    let prewarm_ms = if exact_only { bench_support::phase_elapsed_ms(bench_prewarm) } else { 0.0 };
    // Evaluate the target closure once in topological order. Downstream nodes only
    // consume already-computed outputs; no branch/target can re-enter a node.
    let bench_walk = bench_support::phase_clock();
    let part_targets = part_display_targets(&request);
    let mesh_targets: HashSet<String> =
        request.mesh_targets.as_deref().unwrap_or(&request.targets).iter().cloned().collect();
    #[cfg(not(test))]
    let _ = &mesh_targets;
    // Nodes needed by a mesh-only target must still produce a Mesh even when
    // they are also ancestors of a Part target.
    let mut mesh_required = HashSet::new();
    for target in &request.targets {
        if part_targets.contains(target) {
            continue;
        }
        mesh_required.extend(ancestors_of(&request, target));
    }
    let walk_ms = bench_support::phase_elapsed_ms(bench_walk);
    let bench_order = bench_support::phase_clock();
    let evaluation_order = evaluation_order_for_targets(&request);
    let order_ms = bench_support::phase_elapsed_ms(bench_order);
    let bench_loop = bench_support::phase_clock();
    for node_id in evaluation_order {
        if eval.folded_into_union.contains(&node_id) {
            // A Union intermediate whose run is fused into the Union above it; the top node
            // builds this body in one `union_all` when it is evaluated.
            continue;
        }
        if exact_only {
            let Some(current) = eval.index.node(&node_id) else { continue };
            if part_supported(&current.node_type) {
                eval.evaluate_part_only(&node_id);
            } else if crate::nodes::output_kind(crate::nodes::NodeView::from(current))
                == crate::nodes::OutputKind::Effect
            {
                // Export side effects run in the effect pass below. The exact
                // pass only exposes the exporter's input Part.
                let source = eval.index.incoming(&node_id).next().map(|link| link.from.clone());
                if let Some(source) = source {
                    eval.evaluate_part_only(&source);
                }
            }
            continue;
        }
        #[cfg(test)]
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
    let loop_ms = bench_support::phase_elapsed_ms(bench_loop);
    let bench_tail = bench_support::phase_clock();
    let mesh_targets = if exact_only {
        Vec::new()
    } else {
        request.mesh_targets.clone().unwrap_or_else(|| request.targets.clone())
    };
    let mut mesh_payloads = BTreeMap::new();
    #[cfg(not(test))]
    let _ = &mesh_payloads;
    let mut empty_targets = Vec::new();
    for id in &mesh_targets {
        if !request.targets.contains(id) {
            continue; // Python only meshes evaluated target nodes.
        }
        if eval.index.node(id).is_some_and(|node| sketch_is_2d_overlay_only(&node.node_type)) {
            // See `sketch_is_2d_overlay_only`: the overlay owns sketch drawing.
            // Suppress the payload even if another evaluation path produced one,
            // so a stray sketch body can never reach the 3D viewport.
            empty_targets.push(id.clone());
            continue;
        }
        match eval.objects.get(id) {
            Some(mesh) => {
                let key = eval.index.node(id).and_then(|node| node.cache.as_ref()?.key.clone());
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
    let exact_parts: BTreeMap<String, nodecraft_exact::ExactPart> = eval
        .parts
        .iter()
        .map(|(id, part)| {
            // Plan identity must follow geometry content, not the node id: a
            // transform changes the B-rep while the id stays the same. The
            // display plan cache otherwise serves stale geometry for the
            // breakpoint after an upstream transform edit.
            let source = node_content_seed(&eval.index, id, &mut std::collections::HashSet::new());
            (id.clone(), exact_part_from(part.as_ref(), source))
        })
        .collect();
    // Exact export is a blocking boundary: the exporter's realised bodies are proved before
    // the writer runs, so an unknown body is decided now and a refused one is an error.
    if request.run_exporters {
        let mut kernel = nodecraft_solidcore_kernel();
        let exporters: Vec<&Node> = eval
            .index
            .nodes()
            .iter()
            .filter(|node| {
                crate::nodes::output_kind(crate::nodes::NodeView::from(*node))
                    == crate::nodes::OutputKind::Effect
            })
            .collect();
        for exporter in exporters {
            for link in eval.index.incoming(&exporter.id) {
                let Some(part) = eval.parts.get(&link.from) else {
                    continue;
                };
                if let Err(error) = prove_part_for_export(part, &mut kernel) {
                    eval.errors.insert(exporter.id.clone(), error);
                }
            }
        }
    }
    // Node effects run once per evaluation, after the exact Part pass. A
    // validation/ghost request has `run_exporters == false`, and a cancelled
    // revision is skipped before any node can write a stale file.
    if request.run_exporters {
        let token = kernel_pool::current_revision_token();
        if !token.is_cancelled() {
            let order: HashSet<String> =
                evaluation_order_for_targets(&request).into_iter().collect();
            let effect_nodes: Vec<Node> = eval
                .index
                .nodes()
                .iter()
                .filter(|node| order.contains(&node.id))
                .cloned()
                .collect();
            for node in effect_nodes {
                let Some(builtin) = crate::nodes::find(&node.node_type) else {
                    continue;
                };
                if crate::nodes::output_kind(crate::nodes::NodeView::from(&node))
                    != crate::nodes::OutputKind::Effect
                {
                    continue;
                }
                // An errored exporter or input must not write: bypassed
                // geometry is displayable, but it is not a successful export.
                if eval.errors.contains_key(&node.id) {
                    continue;
                }
                if let Some(error) = eval
                    .index
                    .incoming(&node.id)
                    .find_map(|link| eval.errors.get(&link.from).cloned())
                {
                    eval.errors.insert(node.id.clone(), error);
                    continue;
                }
                let result = {
                    let mut part_eval = crate::nodes::EnginePartEval::new(&mut eval, &node);
                    let mut ctx = crate::nodes::EffectContext::new(
                        request.run_exporters,
                        request.project_directory.as_deref(),
                        token.clone(),
                        &mut part_eval,
                    );
                    builtin.effect(crate::nodes::NodeView::from(&node), &mut ctx)
                };
                match result {
                    Ok(_) => {}
                    Err(error) if !token.is_cancelled() => {
                        eval.errors.insert(node.id.clone(), error);
                    }
                    Err(_) => {}
                }
            }
        }
    }
    // Every body this pass produced, displayed targets first and then the intermediates, is
    // handed to the background proof queue. The first node in evaluation order to register a
    // body is its producer; a target that merely shares it only raises its priority.
    {
        fn collect_part<'a>(
            producer: &'a str,
            part: &'a Part,
            displayed: &HashSet<&str>,
            out: &mut Vec<(&'a str, proof_queue::ProofPriority, &'a nodecraft_ops::SolidBody)>,
        ) -> usize {
            if part.bodies.is_empty() {
                return 0;
            }
            let priority = if displayed.contains(producer) {
                proof_queue::ProofPriority::Displayed
            } else {
                proof_queue::ProofPriority::Intermediate
            };
            out.extend(part.bodies.iter().map(|body| (producer, priority, &body.body)));
            part.bodies.len()
        }

        let registration_started = Instant::now();
        // A fully cached pass usually has every body already decided. Read the
        // shared proof state once before building the evaluation order and the
        // registration list: when nothing is Unknown there is no job to queue,
        // so the order/priority bookkeeping and every body clone are skipped.
        let mut scanned_bodies = 0usize;
        let has_unknown = eval.parts.values().any(|part| {
            scanned_bodies += part.bodies.len();
            part.bodies.iter().any(|body| body.body.needs_proof())
        });
        if !has_unknown {
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!(
                    "[proof] order 0us register 0 parts 0 bodies (scanned {scanned_bodies}) in {}us",
                    registration_started.elapsed().as_micros(),
                );
            }
            // Nothing in this pass needs a proof; leave the already-decided bodies alone.
        } else {
            let order_started = Instant::now();
            let displayed: HashSet<&str> = request.targets.iter().map(String::as_str).collect();
            let order = evaluation_order_for_targets(&request);
            let order_us = order_started.elapsed().as_micros();
            let mut registered_parts = 0usize;
            let mut registered_bodies = 0usize;
            let ordered: HashSet<&str> = order.iter().map(String::as_str).collect();
            // One registration pass for every body this evaluation produced, in
            // evaluation order (displayed targets first). The queue reads each
            // body's shared ProofState and its own `known` map before cloning a
            // job; a fully cached pass queues nothing and clones nothing.
            let mut registrations: Vec<(
                &str,
                proof_queue::ProofPriority,
                &nodecraft_ops::SolidBody,
            )> = Vec::new();
            for id in &order {
                if let Some(part) = eval.parts.get(id) {
                    let bodies = collect_part(id, part, &displayed, &mut registrations);
                    if bodies > 0 {
                        registered_parts += 1;
                        registered_bodies += bodies;
                    }
                }
            }
            // A Part produced for an extra display input can be outside the target closure.
            let mut leftovers: Vec<(&String, &Arc<Part>)> =
                eval.parts.iter().filter(|(id, _)| !ordered.contains(id.as_str())).collect();
            leftovers.sort_by(|a, b| a.0.cmp(b.0));
            for (id, part) in leftovers {
                let bodies = collect_part(id, part, &displayed, &mut registrations);
                if bodies > 0 {
                    registered_parts += 1;
                    registered_bodies += bodies;
                }
            }
            if !registrations.is_empty() {
                proof_queue::register_all(proof_guard.generation(), registrations);
            }
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!(
                    "[proof] order {order_us}us register {registered_parts} parts {registered_bodies} bodies in {}us",
                    registration_started.elapsed().as_micros(),
                );
            }
        }
    }
    let tail_ms = bench_support::phase_elapsed_ms(bench_tail);
    // Measure the whole evaluation before recording, so `total_ms` covers every
    // phase above including the tail.
    let total_ms = bench_support::phase_elapsed_ms(bench_total);
    bench_support::record(bench_support::EvalPhaseTimings {
        prewarm_ms,
        walk_ms,
        order_ms,
        loop_ms,
        tail_ms,
        total_ms,
    });
    let response = Response {
        request_id,
        tab_id,
        graph_revision,
        empty_targets,
        node_errors: eval.errors,
        node_warnings: eval.warnings,
        evaluation_order: eval.order,
        timings: eval.timings,
        brep_timings: eval.brep_timings,
        evaluation_ms: started.elapsed().as_secs_f64() * 1000.0,
        cache_stats: Some(CacheStats {
            hits: eval.cache_hits,
            misses: eval.cache_misses,
            brep_hits: eval.brep_cache_hits,
            part_hits: eval.part_cache_hits,
            display_mesh_hits: eval.display_mesh_cache_hits,
        }),
        engine: "solidcore-experiment".into(),
    };
    nodecraft_mesh::set_display_tolerance(None);
    EngineOutput {
        response,
        cache: eval.cache,
        #[cfg(test)]
        meshes: mesh_payloads,
        diagnostics: eval.diagnostics,
        exact_parts,
        tessellation_diagnostics: eval.tessellation_diagnostics,
        proof_generation: proof_guard.generation(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn respond_with_progress(
        request: Request,
        cache: EngineCache,
        progress_tx: Option<mpsc::Sender<Vec<u8>>>,
    ) -> (TestResponse, EngineCache, HashMap<String, Vec<NodeDiagnostic>>) {
        let mut callback = |event: ProgressEvent| {
            let Some(tx) = &progress_tx else {
                return;
            };
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
        (
            TestResponse { response: output.response, meshes: output.meshes },
            output.cache,
            output.diagnostics,
        )
    }
    use crate::nodes::NodeDiagnostic;
    use nodecraft_mesh::StepImport;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn cad_import(path: &str) -> nodecraft_step::CadImport {
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{path}: {error}"));
        nodecraft_step::read_cad(&bytes, path).unwrap_or_else(|error| panic!("{path}: {error}"))
    }

    fn cad_bodies(path: &str) -> Vec<nodecraft_step::CadBody> {
        cad_import(path).bodies
    }

    fn cad_body(path: &str) -> nodecraft_step::CadBody {
        let mut bodies = cad_bodies(path);
        assert_eq!(bodies.len(), 1, "{path} has {} bodies", bodies.len());
        bodies.remove(0)
    }

    fn body_volume(body: &nodecraft_step::CadBody) -> f64 {
        nodecraft_step::record_volume(&body.brep).unwrap_or_else(|error| panic!("{error}"))
    }

    fn body_bounds(body: &nodecraft_step::CadBody) -> ([f64; 3], [f64; 3]) {
        nodecraft_step::record_bounds(&body.brep).expect("body bounds")
    }

    fn write_primitive_values(path: &str, values: serde_json::Value) {
        let node = Node {
            id: "fixture".into(),
            node_type: "primitive".into(),
            values,
            suppressed: false,
            cache: None,
        };
        write_primitive_step(&node, path).expect("write primitive STEP");
    }

    fn write_box(path: &str, length: f64, width: f64, height: f64) {
        write_primitive_values(
            path,
            serde_json::json!({
                "primitiveType": "box", "length": length, "width": width, "height": height
            }),
        );
    }

    fn write_cylinder(path: &str, radius: f64, height: f64) {
        write_primitive_values(
            path,
            serde_json::json!({
                "primitiveType": "cylinder", "radius": radius, "height": height
            }),
        );
    }

    /// Write `count` 10 mm boxes placed 30 mm apart, each with a colour, so a
    /// kernel/exact test can tell one imported body from several.
    fn write_bodies_step(path: &str, count: usize) {
        let mut bodies = Vec::with_capacity(count);
        for index in 0..count {
            let node = Node {
                id: format!("body{index}"),
                node_type: "primitive".into(),
                values: serde_json::json!({
                    "primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0
                }),
                suppressed: false,
                cache: None,
            };
            let brep = crate::nodes::primitive_brep(&node).expect("fixture box");
            let color = if index % 2 == 0 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
            bodies.push(nodecraft_step::ExportBody {
                name: format!("body{index}"),
                color: Some(color),
                placement: Some(placement_of([index as f64 * 30.0, 0.0, 0.0], [0.0; 3])),
                sheet: false,
                brep: nodecraft_step::record_of_brep(&brep).expect("fixture record"),
            });
        }
        nodecraft_step::write_step(path, &bodies, "fixture", "ap242")
            .expect("write multi-body STEP");
    }

    fn translate_step(path: &str, offset: [f64; 3]) {
        let import = cad_import(path);
        let mut bodies = Vec::with_capacity(import.bodies.len());
        for body in import.bodies {
            let local = body.placement.unwrap_or([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]);
            bodies.push(nodecraft_step::ExportBody {
                name: body.name,
                color: body.color,
                placement: Some(compose_placement(placement_of(offset, [0.0; 3]), local)),
                sheet: body.sheet,
                brep: body.brep,
            });
        }
        let options = nodecraft_step::StepExportOptions {
            product_name: std::path::Path::new(path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("fixture")
                .to_string(),
            ..nodecraft_step::StepExportOptions::default()
        };
        nodecraft_step::write_step_with_options(path, &bodies, &options).expect("translate STEP");
    }

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
            .arg("-C")
            .arg(&root)
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|hash| !hash.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let dirty = std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .args([
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "rust/crates",
                "rust/Cargo.toml",
                "rust/Cargo.lock",
                ":(exclude)rust/crates/nodecraft-engine/testdata",
            ])
            .output()
            .ok()
            .is_some_and(|output| output.status.success() && !output.stdout.is_empty());
        if dirty { format!("{hash}+") } else { hash }
    }

    fn solidcore_revision() -> String {
        let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
        std::fs::read_to_string(lock)
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.contains("solidcore?branch=main#"))
                    .map(str::to_string)
            })
            .and_then(|line| {
                line.split_once('#')
                    .map(|(_, hash)| hash.trim_matches('"').chars().take(12).collect::<String>())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn record_execution_time(test: &str, evaluation_ms: f64) {
        // Debug builds are for correctness only; release builds check speed.
        if cfg!(debug_assertions) {
            return;
        }
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
        let slower =
            last_passing_ms.is_some_and(|last| evaluation_ms > last * (1.0 + NOISE_TOLERANCE));
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
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("open execution log");
        writeln!(file, "{}", serde_json::to_string(&entry).expect("execution log json"))
            .expect("append execution log");
        assert!(
            !slower,
            "{test}: execution {current_ms}ms is more than 2% slower than previous {last_passing_ms:?}ms"
        );
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

    /// A closed profile with four lines and a filleted top-right corner: the
    /// user's sketch shape, including the arc edge a fillet can target.
    fn rounded_profile_sketch() -> crate::sketch::Sketch {
        let mut sketch = crate::sketch::Sketch::new();
        let p1 = sketch.add_point(solidcore_record::Vec2::new(-20.0, -15.0), None).unwrap();
        let p2 = sketch.add_point(solidcore_record::Vec2::new(20.0, -15.0), None).unwrap();
        let p3 = sketch.add_point(solidcore_record::Vec2::new(20.0, 9.0), None).unwrap();
        let p4 = sketch.add_point(solidcore_record::Vec2::new(14.0, 15.0), None).unwrap();
        let p5 = sketch.add_point(solidcore_record::Vec2::new(-20.0, 15.0), None).unwrap();
        let centre = sketch.add_point(solidcore_record::Vec2::new(14.0, 9.0), None).unwrap();
        sketch.add_line_pts(p1, p2, None).unwrap();
        sketch.add_line_pts(p2, p3, None).unwrap();
        sketch.add_curve(crate::sketch::CurveKind::Arc { c: centre, a: p3, b: p4 }, None).unwrap();
        sketch.add_line_pts(p4, p5, None).unwrap();
        sketch.add_line_pts(p5, p1, None).unwrap();
        sketch
    }

    /// The points of a closed periodic spline around an ellipse, the first
    /// repeated at the end as a periodic fit spline requires.
    fn periodic_spline_points() -> Vec<[f64; 2]> {
        vec![
            [-20.0, 0.0],
            [-14.142_135_623_730_95, -10.606_601_717_798_214],
            [0.0, -15.0],
            [14.142_135_623_730_95, -10.606_601_717_798_214],
            [20.0, 0.0],
            [14.142_135_623_730_95, 10.606_601_717_798_214],
            [0.0, 15.0],
            [-14.142_135_623_730_95, 10.606_601_717_798_214],
            [-20.0, 0.0],
        ]
    }

    /// The spline node's own values for a closed periodic spline profile.
    fn periodic_spline_node_values() -> serde_json::Value {
        let points: Vec<serde_json::Value> = periodic_spline_points()
            .into_iter()
            .map(|[x, y]| serde_json::json!({"x": x, "y": y}))
            .collect();
        serde_json::json!({"points": points, "degree": 3, "control": false, "periodic": true})
    }

    /// Four lines with the right side replaced by one spline segment.
    fn mixed_line_spline_sketch() -> crate::sketch::Sketch {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-20.0, -15.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(20.0, -15.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(20.0, 15.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-20.0, 15.0), None).unwrap();
        let s1 = sketch.add_point(solidcore_record::Vec2::new(27.0, -7.0), None).unwrap();
        let s2 = sketch.add_point(solidcore_record::Vec2::new(27.0, 7.0), None).unwrap();
        sketch.add_line_pts(a, b, None).unwrap();
        sketch
            .add_curve(
                crate::sketch::CurveKind::Spline {
                    pts: vec![b, s1, s2, c],
                    control: false,
                    degree: 3,
                    periodic: false,
                    nurbs: None,
                    handles: Vec::new(),
                },
                None,
            )
            .unwrap();
        sketch.add_line_pts(c, d, None).unwrap();
        sketch.add_line_pts(d, a, None).unwrap();
        sketch
    }

    fn exact_part_volume(part: &nodecraft_exact::ExactPart) -> f64 {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        part.bodies
            .iter()
            .map(|body| nodecraft_layers::kernel_volume(&mut kernel, &body.brep).expect("volume"))
            .sum()
    }

    fn exact_part_bounds(part: &nodecraft_exact::ExactPart) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for body in &part.bodies {
            for vertex in body.brep.vertex_list() {
                if let Ok(vertex) = body.brep.vertex(*vertex) {
                    let point = vertex.point;
                    for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                        min[axis] = min[axis].min(value);
                        max[axis] = max[axis].max(value);
                    }
                }
            }
        }
        (min, max)
    }

    fn assert_exact_bounds(
        part: &nodecraft_exact::ExactPart,
        expected_min: [f64; 3],
        expected_max: [f64; 3],
    ) {
        let (min, max) = exact_part_bounds(part);
        for axis in 0..3 {
            assert!(
                (min[axis] - expected_min[axis]).abs() < 1.0e-6,
                "min[{axis}] = {} != {}",
                min[axis],
                expected_min[axis]
            );
            assert!(
                (max[axis] - expected_max[axis]).abs() < 1.0e-6,
                "max[{axis}] = {} != {}",
                max[axis],
                expected_max[axis]
            );
        }
    }

    /// Bounding-box centre of every placed body, with the instance transform
    /// chain baked into each B-rep. This is the world position the viewport and
    /// the GPU instance matrices must agree on.
    fn realized_body_centres(part: &nodecraft_exact::ExactPart) -> Vec<[f64; 3]> {
        part.realized_breps()
            .expect("realize exact bodies")
            .iter()
            .map(|brep| {
                let mut min = [f64::INFINITY; 3];
                let mut max = [f64::NEG_INFINITY; 3];
                for vertex in brep.vertex_list() {
                    let point = brep.vertex(*vertex).expect("vertex").point;
                    for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                        min[axis] = min[axis].min(value);
                        max[axis] = max[axis].max(value);
                    }
                }
                [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5]
            })
            .collect()
    }

    fn assert_centres(actual: &[[f64; 3]], expected: &[[f64; 3]]) {
        assert_eq!(actual.len(), expected.len(), "copy count");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            for axis in 0..3 {
                assert!(
                    (actual[axis] - expected[axis]).abs() < 1.0e-5,
                    "copy {index} centre {actual:?} != {expected:?} (axis {axis})",
                );
            }
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
        let mesh = &output.meshes.get("chamfer").expect("chamfer mesh").mesh;
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
        let mesh = &output.meshes.get("time").expect("time mesh").mesh;
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
        let first = &first.meshes.get("scatter").expect("scatter mesh").mesh;
        let second = &second.meshes.get("scatter").expect("scatter mesh").mesh;
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
        let mesh = &output.meshes.get("c").expect("chamfer mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert!(
            !mesh.face_groups.iter().any(|group| group.starts_with("blend:")),
            "preview should wait for an edge selection"
        );
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
        assert!(!output.meshes["c2"].mesh.faces.is_empty());
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
        let mesh = &output.meshes["chamfer"].mesh;
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
        let mesh = &output.meshes.get("c").expect("chamfer mesh").mesh;
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
        let mesh = &output.meshes.get("c").expect("chamfer mesh").mesh;
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
        let mesh = &output.meshes.get("fillet").expect("fillet mesh").mesh;
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
        }))
        .expect("look request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let mesh = &output.meshes.get("look").expect("look mesh").mesh;
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
        let brep = part.bodies[0].body.brep();
        let kernel = nodecraft_solidcore::kernel::kernel();
        let values = serde_json::json!({"edges": [
            {"center": [14.100593090057371, -11.389926261372036, 4.566759937339359], "id": "edge:1"},
            {"center": [14.100598600175644, 11.389923453330994, 4.5666884283224745], "id": "edge:2"}
        ]});
        let selections = nodecraft_blend::select_edges_from_values(&kernel, brep, &values)
            .expect("centre-first selections");
        assert_eq!(selections.len(), 2);
        let ids: std::collections::BTreeSet<u32> =
            selections.iter().map(|selection| selection.edge.index()).collect();
        assert_eq!(ids, [4, 5].into_iter().collect(), "selected the wrong cylinder edges: {ids:?}");
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
        assert!(
            first.response.node_errors.contains_key("fillet"),
            "{:?}",
            first.response.node_errors
        );
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
        let mesh = &output.meshes.get("look").expect("look mesh").mesh;
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        let unmatched = mesh
            .face_colors
            .iter()
            .filter(|color| !(color[0] < 0.01 && color[1] > 0.10 && color[2] > 0.20))
            .count();
        assert_eq!(
            unmatched,
            0,
            "{unmatched} of {} faces did not receive the Look colour",
            mesh.faces.len()
        );
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
        let mesh = &output.meshes.get("fillet").expect("fillet mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        assert!(
            mesh.face_colors.iter().any(|color| color[0] > 0.5),
            "look colour was lost by the fillet"
        );
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
        let mesh = &output.meshes.get("boolean").expect("boolean mesh").mesh;
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.face_colors.len(), mesh.faces.len());
        assert!(
            mesh.face_colors.iter().any(|color| color != &[0.32, 0.34, 0.37]),
            "no look colours survived"
        );
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
            assert!(
                output.response.node_errors.is_empty(),
                "{continuity}: {:?}",
                output.response.node_errors
            );
            let mesh = &output.meshes.get("fillet").expect("fillet mesh").mesh;
            assert!(!mesh.faces.is_empty(), "{continuity} fillet should have faces");
        }
    }

    #[test]
    #[ignore = "slow: 12s"]
    fn train_graph_execution_regression() {
        let sample =
            std::fs::read_to_string(fixture("samples/train.nodecraft")).expect("train sample");
        let sample: serde_json::Value = serde_json::from_str(&sample).expect("train sample json");
        let nodes: Vec<serde_json::Value> = sample["root"]["nodes"]
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
        let targets: Vec<String> =
            nodes.iter().map(|node| node["id"].as_str().unwrap_or_default().to_string()).collect();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 303,
            "progress": true,
            "targets": targets,
            "meshTargets": ["n69"],
            "binaryMesh": false,
            "nodes": nodes,
            "connections": sample["root"]["connections"].clone(),
        }))
        .expect("train execution json");
        let mut starts = Vec::new();
        let (tx, _rx) = std::sync::mpsc::channel();
        let mut progress = |event: ProgressEvent| {
            if event.phase == ProgressPhase::Start {
                starts.push(event.node_id);
            }
            let _ = tx.send(());
        };
        let output = evaluate(request.clone(), EngineCache::new(), Some(&mut progress));
        let response = &output.response;
        assert!(response.evaluation_ms > 0.0);
        assert_eq!(response.timings.len(), 76, "every train node should be accounted for");
        // The current SolidCore blend fixes n76, which used to be the only
        // expected failure on this graph.
        let mut actual_errors: Vec<&str> =
            response.node_errors.keys().map(String::as_str).collect();
        actual_errors.sort_unstable();
        assert!(actual_errors.is_empty(), "unexpected train errors: {actual_errors:?}");
        assert!(!response.node_errors.contains_key("n68"));
        assert!(!response.node_errors.contains_key("n72"));
        assert!(!response.node_errors.contains_key("n74"));
        assert!(!response.node_errors.contains_key("n75"));
        assert!(
            starts.len() == 76
                && starts.iter().collect::<std::collections::HashSet<_>>().len() == 76
        );
        let mesh = &output.meshes.get("n69").expect("n69 mesh").mesh;
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
        let sample =
            std::fs::read_to_string(fixture("samples/train.nodecraft")).expect("train sample");
        let sample: serde_json::Value = serde_json::from_str(&sample).expect("train sample json");
        let nodes: Vec<serde_json::Value> = sample["root"]["nodes"]
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
            "connections": sample["root"]["connections"].clone(),
        }))
        .expect("train request json");
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) =
            respond_with_progress(request, EngineCache::new(), Some(progress_tx));
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
                    assert!(
                        starts.insert(node.clone(), index).is_none(),
                        "duplicate start for {node}"
                    );
                }
                "node-complete" => {
                    assert!(
                        stops.insert(node.clone(), index).is_none(),
                        "duplicate stop for {node}"
                    );
                }
                phase => panic!("unexpected progress phase {phase:?}"),
            }
        }
        assert_eq!(starts.len(), stops.len(), "start/stop counts differ");
        assert_eq!(
            starts.len(),
            response.timings.len(),
            "trigger count must match processed node count"
        );
        assert_eq!(
            starts.keys().collect::<std::collections::BTreeSet<_>>(),
            stops.keys().collect::<std::collections::BTreeSet<_>>()
        );
        for id in starts.keys() {
            assert!(response.timings.contains_key(id), "progress for unprocessed node {id}");
            assert!(starts[id] < stops[id], "stop before start for {id}");
        }
        let sample_connections = sample["root"]["connections"].as_array().expect("connections");
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
        let (first, cache, _) =
            respond_with_progress(request(10.0, true), EngineCache::new(), None);
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
        let (first, cache, _) =
            respond_with_progress(request(10.0, true), EngineCache::new(), None);
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
            }))
            .expect("dirty request json")
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, cache, _diagnostics) =
            respond_with_progress(request(false, 1), EngineCache::new(), Some(tx));
        let _ = rx.try_iter().count();
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, _cache, _diagnostics) =
            respond_with_progress(request(true, 2), cache, Some(tx));
        let phases: Vec<String> = rx
            .iter()
            .map(|bytes| {
                let frame: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("progress json");
                frame["phase"].as_str().unwrap_or_default().to_string()
            })
            .collect();
        assert_eq!(phases, vec!["node-start", "node-complete"], "dirty node must be recomputed");
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
        }))
        .expect("failed branch request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) =
            respond_with_progress(request, EngineCache::new(), Some(tx));
        let starts: Vec<String> = rx
            .iter()
            .filter_map(|bytes| {
                let frame: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("progress json");
                (frame["phase"] == "node-start")
                    .then(|| frame["nodeId"].as_str().unwrap_or_default().to_string())
            })
            .collect();
        assert_eq!(
            starts.iter().filter(|id| id.as_str() == "bad").count(),
            1,
            "failed node restarted: {starts:?}"
        );
        assert!(response.node_errors.contains_key("bad"));
        assert!(
            response.node_errors.get("b").is_some_and(|error| error.contains("Input bad failed"))
        );
        assert!(
            response.node_errors.get("c").is_some_and(|error| error.contains("Input bad failed"))
        );
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
        }))
        .expect("bypass request json");
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.contains_key("bad"));
        assert!(!output.response.node_errors.contains_key("move"));
        let mesh = output.meshes.get("move").expect("bypassed mesh");
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
        }))
        .expect("multi-target request json");
        let (tx, rx) = std::sync::mpsc::channel();
        let (_response, _cache, _diagnostics) =
            respond_with_progress(request, EngineCache::new(), Some(tx));
        let starts: Vec<String> = rx
            .iter()
            .filter_map(|bytes| {
                let frame: serde_json::Value =
                    serde_json::from_slice(&bytes).expect("progress json");
                (frame["phase"] == "node-start")
                    .then(|| frame["nodeId"].as_str().unwrap_or_default().to_string())
            })
            .collect();
        assert_eq!(
            starts,
            vec!["a".to_string(), "b".to_string()],
            "a downstream node must run once even when listed as multiple targets"
        );
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
        }))
        .expect("progress request json");
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let (response, _cache, _diagnostics) =
            respond_with_progress(request, EngineCache::new(), Some(progress_tx));
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
        assert!(
            import.diagnostics.fallbacks.is_empty(),
            "partial-import fallbacks: {:?}",
            import.diagnostics.fallbacks
        );
        assert!(
            import.diagnostics.warnings.is_empty(),
            "partial-import warnings: {:?}",
            import.diagnostics.warnings
        );
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

    /// CADIR/STEP export writes millimetres; a non-mm output unit is refused.
    #[test]
    fn rejects_non_millimetre_export_unit() {
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
        assert!(
            response.node_errors.values().any(|error| error.contains("writes millimetres")),
            "{:?}",
            response.node_errors
        );
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
        }))
        .expect("mesh request");
        let mesh_index = RequestIndex::new(&mesh_request);
        assert!(exact_brep_demands(&mesh_index).is_empty());

        let boolean_request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["b"], "meshTargets": ["b"],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box"}},
                {"id": "b", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [{"from": "p", "out": "shape", "to": "b", "in": "a"}]
        }))
        .expect("boolean request");
        let boolean_index = RequestIndex::new(&boolean_request);
        let demands = exact_brep_demands(&boolean_index);
        assert!(demands.contains("p"));
        assert!(demands.contains("b"));
    }

    #[test]
    fn exact_pass_exposes_parts_without_meshing() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 702,
            "targets": ["box"],
            "binaryMesh": false,
            "runExporters": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}}
            ],
            "connections": []
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.meshes.is_empty(), "exact pass must not mesh");
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("box").expect("box exact part");
        assert_eq!(part.bodies.len(), 1);
        assert!(part.bodies[0].placement.is_identity());
    }

    #[test]
    fn failed_combine_operand_does_not_hide_successful_ones() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["part"], "meshTargets": [],
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "missing", "type": "stepimporter", "values": {"path": "/nonexistent/does-not-exist.step"}},
                {"id": "part", "type": "combine", "values": {}}
            ],
            "connections": [
                {"from": "box", "out": "shape", "to": "part", "in": "shape"},
                {"from": "missing", "out": "shape", "to": "part", "in": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        let part = output.exact_parts.get("part").expect("partial combine part");
        assert_eq!(part.bodies.len(), 1, "valid box must survive the failed import");
        assert!(!output.response.node_errors.is_empty(), "failed import should still be reported");
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
        let directory =
            std::env::temp_dir().join(format!("sn-worker-pyramid-assembly-{}", std::process::id()));
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
        let bodies = cad_bodies(directory.join("pyramids.step").to_str().unwrap());
        assert_eq!(bodies.len(), 2);
        assert!((bodies.iter().map(|body| body_volume(&body)).sum::<f64>() - 800.0).abs() < 1e-6);
    }

    #[test]
    fn exports_mirrored_part_as_two_exact_solids() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-mirror-part-{}", std::process::id()));
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
        let bodies = cad_bodies(directory.join("mirror.step").to_str().unwrap());
        assert_eq!(bodies.len(), 2, "mirror combine should export source and mirrored Part");
        assert!((bodies.iter().map(|body| body_volume(&body)).sum::<f64>() - 128.0).abs() < 1e-6);
    }

    #[test]
    fn exports_exact_pyramid_step() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-pyramid-{}", std::process::id()));
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
        let body = cad_body(directory.join("pyramid.step").to_str().unwrap());
        assert_eq!(body.brep.faces.len(), 5);
        // The exact SolidCore pyramid keeps its split seam vertices; the planar
        // writer may therefore emit more than the five corner points.
        assert!(body.brep.vertices.len() >= 5);
        assert!((body_volume(&body) - 4000.0).abs() < 1e-6, "{}", body_volume(&body));
    }

    #[test]
    fn exports_primitive_as_faceted() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-prim-faceted-{}", std::process::id()));
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
        assert!(
            text.contains("MANIFOLD_SOLID_BREP") || text.contains("ADVANCED_FACE"),
            "not a faceted B-rep"
        );
        let bodies = cad_bodies(path.to_str().unwrap());
        assert!((body_volume(&bodies[0]) - 8.0).abs() < 1e-6, "{}", body_volume(&bodies[0]));
    }

    #[test]
    fn reexports_imported_planar_step_as_faceted() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-faceted-export-{}", std::process::id()));
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
        assert!(
            text.contains("MANIFOLD_SOLID_BREP") || text.contains("ADVANCED_FACE"),
            "not a faceted brep"
        );
        let bodies = cad_bodies(path.to_str().unwrap());
        assert_eq!(bodies.len(), 1);
        assert!((body_volume(&bodies[0]) - 60.0).abs() < 1e-6, "{}", body_volume(&bodies[0]));
    }

    #[test]
    fn exports_curved_import_as_faceted_mesh() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-curved-faceted-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let cylinder = directory.join("cylinder.step");
        write_cylinder(cylinder.to_str().unwrap(), 5.0, 10.0);
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
        let bodies = cad_bodies(path.to_str().unwrap());
        assert_eq!(bodies.len(), 1);
        assert!(!bodies[0].brep.faces.is_empty());
    }

    /// Exact Boolean STEP export: disjoint imported boxes union to a valid STEP solid.
    #[test]
    fn exports_exact_boolean_step() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-boolexport-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        write_box(a.to_str().unwrap(), 10.0, 4.0, 20.0);
        write_box(b.to_str().unwrap(), 10.0, 4.0, 20.0);
        translate_step(b.to_str().unwrap(), [20.0, 0.0, 0.0]);
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
        assert!(
            text.contains("PRESENTATION_LAYER_ASSIGNMENT('Parts'"),
            "Boolean layer metadata missing"
        );
        assert!(text.contains("CONFIG_CONTROL_DESIGN"), "Boolean schema metadata missing");
        let bodies = cad_bodies(path.to_str().unwrap());
        assert_eq!(bodies.len(), 2, "a disjoint union is two placed solids");
        let volume: f64 = bodies.iter().map(body_volume).sum();
        assert!((volume - 1600.0).abs() < 1e-6, "volume {volume}");
        let min_x = bodies.iter().map(|body| body_bounds(body).0[0]).fold(f64::INFINITY, f64::min);
        let max_x =
            bodies.iter().map(|body| body_bounds(body).1[0]).fold(f64::NEG_INFINITY, f64::max);
        assert!((min_x - -5.0).abs() < 1e-6 && (max_x - 25.0).abs() < 1e-6, "{min_x:?}..{max_x:?}");
    }

    /// A Circular Pattern's copies must survive the exporter -> importer round
    /// trip. The importer's exact Part has one body per written MANIFOLD_SOLID_BREP,
    /// and its centres are the pattern's placed centres.
    #[test]
    fn stepimporter_imports_every_body_the_exporter_wrote_for_a_pattern() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-step-all-bodies-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["exporter"],
            "meshTargets": [],
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 6, "axis": "z"}, "cache": {"key": "pattern"}},
                {"id": "exporter", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "patterned"}, "cache": {"key": "exporter"}}
            ],
            "connections": [
                {"from": "box", "out": "shape", "to": "move", "in": "shape"},
                {"from": "move", "out": "shape", "to": "pattern", "in": "shape"},
                {"from": "pattern", "out": "shape", "to": "exporter", "in": "shape"}
            ]
        })).expect("export request");
        let exported = evaluate(request, EngineCache::new(), None);
        assert!(exported.response.node_errors.is_empty(), "{:?}", exported.response.node_errors);
        let pattern = exported.exact_parts.get("pattern").expect("pattern exact part");
        let expected = realized_body_centres(pattern);
        assert_eq!(expected.len(), 6, "the pattern made six copies");

        let path = directory.join("patterned.step");
        assert!(path.is_file(), "pattern export was not written");
        let written = cad_bodies(path.to_str().unwrap());
        assert_eq!(written.len(), 6, "the exporter wrote one solid per copy");

        let import_request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 2,
            "targets": ["import"],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": [
                {"id": "import", "type": "stepimporter", "values": {"path": path.to_string_lossy()}, "cache": {"key": "import"}}
            ],
            "connections": []
        })).expect("import request");
        let imported = evaluate_exact(import_request, EngineCache::new(), None);
        assert!(imported.response.node_errors.is_empty(), "{:?}", imported.response.node_errors);
        let part = imported.exact_parts.get("import").expect("import exact part");
        assert_eq!(part.bodies.len(), 6, "the importer Part has every written body");
        assert_centres(&realized_body_centres(part), &expected);
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The single-operand exact path keeps its one-body rule. A one-body file
    /// still works; a multi-body file names the body count instead of silently
    /// dropping all but the first.
    #[test]
    fn exact_step_operand_needs_exactly_one_body() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-step-operand-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp directory");
        let one = directory.join("one.step");
        let many = directory.join("many.step");
        write_bodies_step(one.to_str().unwrap(), 1);
        write_bodies_step(many.to_str().unwrap(), 3);

        let request = |path: &std::path::Path| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["import"],
                "meshTargets": [],
                "nodes": [
                    {"id": "import", "type": "stepimporter", "values": {"path": path.to_string_lossy()}}
                ],
                "connections": []
            })).expect("operand request")
        };
        let one_request = request(&one);
        let mut eval = Eval::new(&one_request, None);
        let brep = exact_operand_brep(&mut eval, "import").expect("one body is an exact operand");
        assert!(!brep.face_list().is_empty());

        let many_request = request(&many);
        let mut eval = Eval::new(&many_request, None);
        let error = match exact_operand_brep(&mut eval, "import") {
            Ok(_) => panic!("three bodies must not be silently collapsed"),
            Err(error) => error,
        };
        assert!(error.contains("exact operand needs exactly one"), "{error}");
        assert!(error.contains('3'), "{error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// One operand representation: a registered primitive and an unported node
    /// both hand back the body of their evaluated Part.
    #[test]
    fn exact_operands_come_from_the_evaluated_part() {
        let primitive_request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["box"], "meshTargets": [],
            "nodes": [
                {"id": "box", "type": "primitive",
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}}
            ],
            "connections": []
        }))
        .expect("primitive request");
        let mut eval = Eval::new(&primitive_request, None);
        let brep = exact_operand_brep(&mut eval, "box").expect("primitive operand");
        assert!(eval.parts.contains_key("box"), "the primitive operand came from its Part");
        assert_eq!(eval.parts["box"].bodies.len(), 1);
        assert!(!brep.face_list().is_empty());

        // Extrude is not ported, so it has no node-specific raw-operand route;
        // its Part is still the one exact operand value.
        let extrude_request = extrude_request(&rounded_profile_sketch(), None);
        let mut eval = Eval::new(&extrude_request, None);
        let brep = exact_operand_brep(&mut eval, "extrude").expect("unported operand");
        assert!(eval.parts.contains_key("extrude"), "the unported operand came from its Part");
        assert!(!brep.face_list().is_empty());
    }

    /// A multi-body Part is never collapsed to its first body.
    #[test]
    fn a_two_body_part_is_not_an_exact_operand() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["part"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "part", "type": "combine", "values": {}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "part", "in": "shape"},
                {"from": "b", "out": "shape", "to": "part", "in": "shape"}
            ]
        }))
        .expect("combine request");
        let mut eval = Eval::new(&request, None);
        let error =
            exact_operand_brep(&mut eval, "part").expect_err("two bodies are not one operand");
        assert!(error.contains("exact operand needs exactly one"), "{error}");
        assert!(error.contains('2'), "{error}");
    }

    #[test]
    fn deferred_part_mirror_realizes_through_the_mirror_task() {
        let source = crate::nodes::primitive_brep(&Node {
            id: "b".into(),
            node_type: "primitive".into(),
            values: serde_json::json!({"primitiveType": "box", "length": 40.0, "width": 40.0, "height": 40.0}),
            suppressed: false,
            cache: None,
        }).expect("box");
        let values = serde_json::json!({"plane": "yz", "offset": 10.0, "combine": false});
        let part = Part::from_brep(source);
        let mirrored = part.mirrored(&values);
        let realized = mirrored.realized_solids().expect("realized mirror");
        assert_eq!(realized.len(), 1);
        let expected = crate::node_task::execute_exact(
            crate::node_task::ExactNodeTask::Mirror {
                brep: crate::nodes::primitive_brep(&Node {
                    id: "b".into(),
                    node_type: "primitive".into(),
                    values: serde_json::json!({"primitiveType": "box", "length": 40.0, "width": 40.0, "height": 40.0}),
                    suppressed: false,
                    cache: None,
                }).expect("box"),
                values,
            },
            "m",
            fnv_seed("m"),
        )
        .expect("mirror task")
        .into_brep()
        .expect("mirror shape");
        let bounds = |brep: &nodecraft_solidcore::solidcore::topology::Brep| {
            let mut min = [f64::INFINITY; 3];
            let mut max = [f64::NEG_INFINITY; 3];
            for vertex in brep.vertex_list() {
                let point = brep.vertex(*vertex).expect("vertex").point;
                for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                    min[axis] = min[axis].min(value);
                    max[axis] = max[axis].max(value);
                }
            }
            (min, max)
        };
        assert_eq!(bounds(&realized[0]), bounds(&expected));
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
                {"id": "m", "type": "mirror", "values": {"plane": "yz", "offset": 10.0, "combine": false}, "cache": {"key": "m"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "mirrored"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "b", "out": "shape", "to": "m", "in": "shape"},
                {"from": "m", "out": "shape", "to": "e", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response
            .meshes
            .get("m")
            .unwrap_or_else(|| {
                panic!("mirror mesh keys {:?}", response.meshes.keys().collect::<Vec<_>>())
            })
            .mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        assert!((min[0] - 0.0).abs() < 1e-4 && (max[0] - 40.0).abs() < 1e-4, "{min:?} {max:?}");
        assert!((min[1] + 20.0).abs() < 1e-4 && (max[1] - 20.0).abs() < 1e-4, "{min:?} {max:?}");
        let body = cad_body(directory.join("mirrored.step").to_str().unwrap());
        assert!((body_volume(&body).abs() - 64000.0).abs() < 1e-6, "volume {}", body_volume(&body));
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

    /// A mirror saved without `combine` must do what the inspector already
    /// showed (Combine on): keep the source and append the mirrored body.
    #[test]
    fn mirror_missing_combine_matches_explicit_true() {
        fn run(combine: Option<bool>) -> Vec<(u64, u32)> {
            let mut values = serde_json::json!({"plane": "yz", "offset": 10.0});
            if let Some(combine) = combine {
                values["combine"] = serde_json::json!(combine);
            }
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["m"],
                "meshTargets": [],
                "nodes": [
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "b"}},
                    {"id": "m", "type": "mirror", "values": values, "cache": {"key": "m"}}
                ],
                "connections": [{"from": "b", "out": "shape", "to": "m", "in": "shape"}]
            })).expect("request");
            let output = evaluate_exact(request, EngineCache::new(), None);
            assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
            let part = output.exact_parts.get("m").expect("mirror exact part");
            assert_eq!(part.bodies.len(), 2, "combine must keep the source and the mirror");
            part_keys(&output.exact_parts, "m")
        }

        assert_eq!(run(None), run(Some(true)));
    }

    /// A mirror saved without `plane` uses the contract default XZ, not the old
    /// reader's XY fallback.
    #[test]
    fn mirror_missing_plane_defaults_to_xz() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["m"],
            "meshTargets": ["m"],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "b"}},
                {"id": "t", "type": "translate", "values": {"y": 10.0}, "cache": {"key": "t"}},
                {"id": "m", "type": "mirror", "values": {"combine": false}, "cache": {"key": "m"}}
            ],
            "connections": [
                {"from": "b", "out": "shape", "to": "t", "in": "shape"},
                {"from": "t", "out": "shape", "to": "m", "in": "shape"}
            ]
        })).expect("request");
        let (response, _, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes.get("m").expect("mirror mesh").mesh;
        let (min, max) = nodecraft_kernel::bounds(mesh).expect("bounds");
        // XZ reflects y: 5..15 becomes -15..-5, while x and z are unchanged.
        let expected_min = [-5.0, -15.0, -5.0];
        let expected_max = [5.0, -5.0, 5.0];
        for axis in 0..3 {
            assert!((min[axis] - expected_min[axis]).abs() < 1.0e-4, "{min:?} {max:?}");
            assert!((max[axis] - expected_max[axis]).abs() < 1.0e-4, "{min:?} {max:?}");
        }
    }

    /// Missing scatter spans resolve to the contract spans, not the old
    /// 200/200 reader defaults.
    #[test]
    fn scatter_missing_spans_match_contract_defaults() {
        fn run(values: serde_json::Value) -> ([f32; 3], [f32; 3]) {
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["s"],
                "meshTargets": ["s"],
                "nodes": [
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 2, "width": 2, "height": 2}, "cache": {"key": "b"}},
                    {"id": "s", "type": "scatter", "values": values, "cache": {"key": "s"}}
                ],
                "connections": [{"from": "b", "out": "shape", "to": "s", "in": "shape"}]
            })).expect("request");
            let (response, _, _) = respond(request, EngineCache::new());
            assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
            let mesh = &response.meshes.get("s").expect("scatter mesh").mesh;
            nodecraft_kernel::bounds(mesh).expect("bounds")
        }
        let base = serde_json::json!({"count": 3.0, "distribution": "grid"});
        let mut explicit = nodecraft_node_contract::default_values("scatter", &base);
        explicit.extend(base.as_object().cloned().expect("object"));
        assert_eq!(run(base), run(serde_json::Value::Object(explicit)));
    }

    /// The primitive contract says a Pyramid defaults to height 40; the old
    /// engine reader used 30.
    #[test]
    fn primitive_missing_pyramid_height_matches_contract_default() {
        fn run(height: Option<f64>) -> ([f64; 3], [f64; 3]) {
            let mut values = serde_json::json!({"primitiveType": "pyramid", "length": 40.0});
            if let Some(height) = height {
                values["height"] = serde_json::json!(height);
            }
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["p"],
                "meshTargets": [],
                "nodes": [{"id": "p", "type": "primitive", "values": values, "cache": {"key": "p"}}],
                "connections": []
            })).expect("request");
            let output = evaluate_exact(request, EngineCache::new(), None);
            assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
            exact_part_bounds(output.exact_parts.get("p").expect("pyramid exact part"))
        }
        assert_eq!(run(None), run(Some(40.0)));
    }

    /// Table over every inspector primitive kind: a node with only its kind
    /// evaluates the same as one whose values carry every contract default.
    #[test]
    fn every_primitive_kind_matches_its_explicit_contract_defaults() {
        fn run(kind: &str, values: serde_json::Value) -> (usize, ([f64; 3], [f64; 3])) {
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["p"],
                "meshTargets": [],
                "nodes": [{"id": "p", "type": "primitive", "values": values, "cache": {"key": "p"}}],
                "connections": []
            })).expect("request");
            let output = evaluate_exact(request, EngineCache::new(), None);
            assert!(
                output.response.node_errors.is_empty(),
                "{kind}: {:?}",
                output.response.node_errors
            );
            let part = output.exact_parts.get("p").expect("primitive exact part");
            (part.bodies.len(), exact_part_bounds(part))
        }

        for (kind, _) in nodecraft_node_contract::PRIMITIVE_TYPES {
            let base = serde_json::json!({"primitiveType": kind});
            let missing = run(kind, base.clone());
            let mut explicit = nodecraft_node_contract::default_values("primitive", &base);
            explicit.extend(base.as_object().cloned().expect("object"));
            assert_eq!(
                missing,
                run(kind, serde_json::Value::Object(explicit)),
                "primitive {kind} disagrees with its contract defaults"
            );
        }
    }

    /// Imported STEP re-export preserves geometry and applies transforms.
    #[test]
    fn reexports_imported_step_with_transform() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-reexport-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let block = fixture(BLOCK);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": block}, "cache": {"key": "i"}},
                {"id": "t", "type": "translate", "values": {"x": 10}, "cache": {"key": "t"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "copy", "color": "#3366cc", "layer": "Parts", "schema": "ap203", "productName": "Bracket"}, "cache": {"key": "e"}}
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
        assert!(text.contains("CONFIG_CONTROL_DESIGN"), "metadata schema missing");
        assert!(text.contains("Bracket"), "metadata product name missing");
        let bodies = cad_bodies(path.to_str().unwrap());
        assert_eq!(bodies.len(), 1);
        let body = &bodies[0];
        assert!((body_volume(&body) - 60.0).abs() < 1e-6, "volume {}", body_volume(&body));
        let (min, max) = body_bounds(&body);
        let moved = body.placement.map_or(0.0, |placement| placement[3][0]);
        assert!(
            (moved + min[0] - 8.5).abs() < 1e-6 && (moved + max[0] - 11.5).abs() < 1e-6,
            "{min:?} {max:?} placement {moved}"
        );
    }

    /// Exact disjoint Boolean: two imported planar STEP solids union through solidcore.
    #[test]
    fn boolean_union_of_disjoint_imported_solids() {
        let directory = std::env::temp_dir().join(format!("sn-worker-bool-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        write_box(a.to_str().unwrap(), 10.0, 4.0, 20.0);
        write_box(b.to_str().unwrap(), 10.0, 4.0, 20.0);
        translate_step(b.to_str().unwrap(), [20.0, 0.0, 0.0]);
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
        let directory =
            std::env::temp_dir().join(format!("sn-worker-project-folder-{}", std::process::id()));
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
        assert!(
            directory.join("project-part.step").exists(),
            "export did not use projectDirectory"
        );
    }

    /// Overlapping imported planar STEP solids now union through solidcore.
    #[test]
    fn boolean_union_of_overlapping_imported_solids() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-bool-overlap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let a = directory.join("a.step");
        let b = directory.join("b.step");
        write_box(a.to_str().unwrap(), 10.0, 10.0, 10.0);
        write_box(b.to_str().unwrap(), 10.0, 10.0, 10.0);
        translate_step(b.to_str().unwrap(), [5.0, 0.0, 0.0]);
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
        let body = cad_body(directory.join("overlap.step").to_str().unwrap());
        assert!((body_volume(&body) - 1500.0).abs() < 1e-6, "volume {}", body_volume(&body));
        let (min, max) = body_bounds(&body);
        assert!((min[0] - -5.0).abs() < 1e-6 && (max[0] - 10.0).abs() < 1e-6, "{min:?} {max:?}");
    }

    /// Every primitive display path must produce a non-empty mesh; the closed,
    /// partial and wedge primitives now all have exact B-reps behind it.
    #[test]
    fn all_primitive_display_paths_render() {
        for (kind, values) in [
            (
                "box",
                serde_json::json!({"primitiveType":"box","length":20.0,"width":10.0,"height":8.0}),
            ),
            (
                "cylinder",
                serde_json::json!({"primitiveType":"cylinder","radius":10.0,"height":20.0}),
            ),
            (
                "cone",
                serde_json::json!({"primitiveType":"cone","bottomRadius":10.0,"topRadius":2.0,"height":20.0}),
            ),
            ("sphere", serde_json::json!({"primitiveType":"sphere","radius":10.0})),
            (
                "torus",
                serde_json::json!({"primitiveType":"torus","majorRadius":15.0,"minorRadius":4.0}),
            ),
            ("pyramid", serde_json::json!({"primitiveType":"pyramid","length":20.0,"height":15.0})),
            ("sphereNoPole", serde_json::json!({"primitiveType":"sphereNoPole","radius":10.0})),
            (
                "partialSphere",
                serde_json::json!({"primitiveType":"partialSphere","radius":10.0,"startAngle":0.0,"endAngle":90.0}),
            ),
            (
                "partialTorus",
                serde_json::json!({"primitiveType":"partialTorus","majorRadius":15.0,"minorRadius":4.0,"startAngle":0.0,"endAngle":90.0}),
            ),
            (
                "wedge",
                serde_json::json!({"primitiveType":"wedge","length":20.0,"width":10.0,"height":8.0,"topLength":5.0}),
            ),
        ] {
            let request: Request = serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": ["p"],
                "meshTargets": ["p"],
                "nodes": [{"id": "p", "type": "primitive", "values": values, "cache": {"key": "p"}}],
                "connections": []
            })).expect("primitive request");
            let (response, _, _) = respond(request, EngineCache::new());
            assert!(response.node_errors.is_empty(), "{kind}: {:?}", response.node_errors);
            let mesh = &response.meshes["p"].mesh;
            assert!(!mesh.faces.is_empty(), "{kind}: empty mesh");
            assert!(!mesh.vertices.is_empty(), "{kind}: empty vertices");
        }
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
    fn empty_boolean_result_is_a_warning_not_an_error() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["cut"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b"}},
                {"id": "far", "type": "translate", "values": {"x": 100.0}, "cache": {"key": "far"}},
                {"id": "cut", "type": "boolean", "values": {"mode": "intersect"}, "cache": {"key": "cut"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "cut", "in": "a"},
                {"from": "b", "out": "shape", "to": "far", "in": "shape"},
                {"from": "far", "out": "shape", "to": "cut", "in": "b"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(
            output.response.node_warnings.get("cut").map(Vec::as_slice),
            Some([crate::node_task::EMPTY_BOOLEAN_WARNING.to_owned()].as_slice()),
            "empty Boolean warns in amber instead of failing red"
        );
        let part = output.exact_parts.get("cut").expect("cut exact part");
        assert!(part.bodies.is_empty(), "an empty Boolean leaves no body");
    }

    #[test]
    fn intersect_a_part_with_several_solids_keeps_every_piece() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["cut"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 30.0, "width": 30.0, "height": 30.0}, "cache": {"key": "a"}},
                {"id": "b1", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b1"}},
                {"id": "t1", "type": "translate", "values": {"x": -6.0}, "cache": {"key": "t1"}},
                {"id": "b2", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "b2"}},
                {"id": "t2", "type": "translate", "values": {"x": 6.0}, "cache": {"key": "t2"}},
                {"id": "combine", "type": "combine", "values": {}, "cache": {"key": "combine"}},
                {"id": "cut", "type": "boolean", "values": {"mode": "intersect"}, "cache": {"key": "cut"}}
            ],
            "connections": [
                {"from": "a", "out": "shape", "to": "cut", "in": "a"},
                {"from": "b1", "out": "shape", "to": "t1", "in": "shape"},
                {"from": "b2", "out": "shape", "to": "t2", "in": "shape"},
                {"from": "t1", "out": "shape", "to": "combine", "in": "shape"},
                {"from": "t2", "out": "shape", "to": "combine", "in": "shape"},
                {"from": "combine", "out": "shape", "to": "cut", "in": "b"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(
            !output.response.node_warnings.contains_key("cut"),
            "{:?}",
            output.response.node_warnings
        );
        let part = output.exact_parts.get("cut").expect("cut exact part");
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let volume: f64 = part
            .bodies
            .iter()
            .map(|body| nodecraft_layers::kernel_volume(&mut kernel, &body.brep).expect("volume"))
            .sum();
        // A 30 mm cube less the rest; B is two 10 mm cubes inside it, apart: intersect
        // must give 2 x 1000, not "empty" from intersecting the pieces one at a time.
        assert!((volume - 2000.0).abs() < 1e-6, "both pieces must survive: {volume}");
    }

    #[test]
    fn boolean_operands_are_evaluated_through_the_part_path() {
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
        let mut eval = Eval::new(&request, None);
        let mesh = eval.evaluate("u").expect("union display mesh");
        assert!(!mesh.faces.is_empty());
        assert!(eval.parts.contains_key("a"), "Boolean A came through the Part path");
        assert!(eval.parts.contains_key("t"), "Boolean B's transform came through the Part path");
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
        let evaluated = evaluate(request, EngineCache::new(), None);
        let response = TestResponse { response: evaluated.response, meshes: evaluated.meshes };
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
        let evaluated = evaluate(request, EngineCache::new(), None);
        let response = TestResponse { response: evaluated.response, meshes: evaluated.meshes };
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
        assert!(
            second.response.cache_stats.expect("cache stats").hits > 0,
            "report must include Part/BRep cache hits"
        );
    }

    #[test]
    fn part_cache_retains_a_large_evaluated_graph() {
        let mut cache = EngineCache::new();
        for index in 0..5_000 {
            cache.insert_part((format!("node{index}"), String::new()), Arc::new(Part::empty()));
        }
        assert!(
            cache.part(&("node0".into(), String::new())).is_some(),
            "early Part entry was evicted"
        );
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
        assert!(second.response.cache_stats.expect("cache stats").hits > 0);
        assert!(second.cache.part_hits() > 0, "Part was not reused");
    }

    #[test]
    fn downstream_change_reuses_cached_operand_parts() {
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
        assert!(second.cache.part_hits() > 0, "unchanged operand Parts were rebuilt");
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

    #[test]
    #[ignore = "known upstream limit: a rotated curved compound's conic record is refused"]
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
        let error = response
            .node_errors
            .get("n28")
            .expect("compound rotated curved operand still reports upstream");
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
    fn ops_boolean_matches_legacy_volume_for_boxes() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives as api;

        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let a = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0)
            .expect("box A");
        let b = api::create_box(&mut kernel, Point3::new(5.0, 0.0, 0.0), 10.0, 6.0, 4.0)
            .expect("box B");
        for op in [
            nodecraft_bool::BooleanOp::Union,
            nodecraft_bool::BooleanOp::Difference,
            nodecraft_bool::BooleanOp::Intersection,
        ] {
            let legacy = nodecraft_bool::boolean_breps(
                op.clone(),
                copy_brep(&a).expect("copy A"),
                copy_brep(&b).expect("copy B"),
            )
            .expect("legacy boolean");
            let ops = nodecraft_layers::boolean_breps_ops(&mut kernel, &a, &b, ops_boolean(op))
                .expect("ops boolean")
                .unwrap_or_else(nodecraft_solidcore::solidcore::topology::Brep::new);
            let legacy_volume =
                nodecraft_layers::kernel_volume(&mut kernel, &legacy).expect("legacy volume");
            let ops_volume =
                nodecraft_layers::kernel_volume(&mut kernel, &ops).expect("ops volume");
            assert!(
                (legacy_volume - ops_volume).abs() <= legacy_volume.abs().max(1.0) * 1.0e-6,
                "Boolean volume parity: legacy={legacy_volume} ops={ops_volume}"
            );
        }
    }

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
        let normal = bounds(&output.meshes["normal"].mesh);
        let reverse = bounds(&output.meshes["reverse"].mesh);
        assert!(!output.meshes["normal"].mesh.faces.is_empty());
        assert!(!output.meshes["reverse"].mesh.faces.is_empty());
        assert!(
            normal.0[0] < reverse.0[0],
            "reverse must swap A/B: normal={normal:?} reverse={reverse:?}"
        );
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
        let directory =
            std::env::temp_dir().join(format!("sn-worker-bool-faceted-{}", std::process::id()));
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
        let bodies = cad_bodies(directory.join("curved-union.step").to_str().unwrap());
        assert!(!bodies.is_empty());
        assert!(body_volume(&bodies[0]) > 0.0);
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

    fn part_keys(
        exact_parts: &BTreeMap<String, nodecraft_exact::ExactPart>,
        id: &str,
    ) -> Vec<(u64, u32)> {
        exact_parts
            .get(id)
            .unwrap_or_else(|| panic!("no exact Part published for {id}"))
            .bodies
            .iter()
            .map(|body| (body.key.source, body.key.body))
            .collect()
    }

    /// Viewing a suppressed pass-through node must publish its source's Part
    /// under the suppressed node's own id; consumers look targets up by id.
    #[test]
    fn a_suppressed_target_publishes_its_passthrough_source_part() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["t"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "a"}},
                {"id": "t", "type": "translate", "suppressed": true, "values": {"x": 25}, "cache": {"key": "t"}}
            ],
            "connections": [
                {"from": "a", "to": "t", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(part_keys(&output.exact_parts, "t"), part_keys(&output.exact_parts, "a"));
    }

    /// A suppressed Boolean passes its A operand through, not the merged result.
    #[test]
    fn a_suppressed_boolean_target_publishes_operand_a() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["u"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "b"}},
                {"id": "u", "type": "boolean", "suppressed": true, "values": {"mode": "union"}, "cache": {"key": "u"}}
            ],
            "connections": [
                {"from": "a", "to": "u", "in": "a", "out": "shape"},
                {"from": "b", "to": "u", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(part_keys(&output.exact_parts, "u"), part_keys(&output.exact_parts, "a"));
    }

    /// A suppressed Union inside a fused run is still a target with a Part: it
    /// passes A through while the Union above it keeps fusing its leaves.
    #[test]
    fn a_suppressed_union_in_a_fused_run_publishes_operand_a() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["u2", "u3"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "a"}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "b"}},
                {"id": "c", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "c"}},
                {"id": "d", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "d"}},
                {"id": "tb", "type": "translate", "values": {"x": 20}, "cache": {"key": "tb"}},
                {"id": "tc", "type": "translate", "values": {"x": 40}, "cache": {"key": "tc"}},
                {"id": "td", "type": "translate", "values": {"x": 60}, "cache": {"key": "td"}},
                {"id": "u1", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u1"}},
                {"id": "u2", "type": "boolean", "suppressed": true, "values": {"mode": "union"}, "cache": {"key": "u2"}},
                {"id": "u3", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u3"}}
            ],
            "connections": [
                {"from": "a", "to": "u1", "in": "a", "out": "shape"},
                {"from": "b", "to": "tb", "in": "shape", "out": "shape"},
                {"from": "tb", "to": "u1", "in": "b", "out": "shape"},
                {"from": "u1", "to": "u2", "in": "a", "out": "shape"},
                {"from": "c", "to": "tc", "in": "shape", "out": "shape"},
                {"from": "tc", "to": "u2", "in": "b", "out": "shape"},
                {"from": "u2", "to": "u3", "in": "a", "out": "shape"},
                {"from": "d", "to": "td", "in": "shape", "out": "shape"},
                {"from": "td", "to": "u3", "in": "b", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(part_keys(&output.exact_parts, "u2"), part_keys(&output.exact_parts, "u1"));
        // u2 drops its B (tc), so the Union above receives u1's bodies plus td.
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let mut volume = |parts: &BTreeMap<String, nodecraft_exact::ExactPart>, id: &str| -> f64 {
            parts[id]
                .bodies
                .iter()
                .map(|body| {
                    nodecraft_layers::kernel_volume(&mut kernel, &body.brep).expect("volume")
                })
                .sum()
        };
        let (u1, td, u3) = (
            volume(&output.exact_parts, "u1"),
            volume(&output.exact_parts, "td"),
            volume(&output.exact_parts, "u3"),
        );
        assert!((u3 - (u1 + td)).abs() <= (u1 + td).abs() * 1.0e-9, "u3={u3} u1={u1} td={td}");
    }

    /// A suppressed feature (fillet) is a ghost pass-through too.
    #[test]
    fn a_suppressed_feature_target_publishes_its_input_part() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["f"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "a"}},
                {"id": "f", "type": "fillet", "suppressed": true, "values": {"length": 1}, "cache": {"key": "f"}}
            ],
            "connections": [
                {"from": "a", "to": "f", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(part_keys(&output.exact_parts, "f"), part_keys(&output.exact_parts, "a"));
    }

    /// A suppressed Circular Pattern passes its source Part through instead of
    /// leaving the copies (or an empty body) behind.
    #[test]
    fn a_suppressed_circular_pattern_publishes_its_input_part() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "height": 10, "width": 10}, "cache": {"key": "a"}},
                {"id": "pattern", "type": "circularpattern", "suppressed": true, "values": {"count": 4}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "a", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert_eq!(part_keys(&output.exact_parts, "pattern"), part_keys(&output.exact_parts, "a"));
    }

    /// Exact assembly export: two placed planar STEP parts through a Combine node.
    #[test]
    fn exports_exact_planar_assembly() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-assembly-{}", std::process::id()));
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
        let bodies = cad_bodies(path.to_str().unwrap());
        assert_eq!(bodies.len(), 2);
        assert!(bodies.iter().all(|body| body_volume(body) > 0.0));
        let xs: Vec<f64> = bodies
            .iter()
            .map(|body| {
                body.placement.map_or(0.0, |placement| placement[3][0]) + body_bounds(body).0[0]
            })
            .collect();
        assert!(xs.iter().any(|x| (*x - 8.5).abs() < 1e-6), "{xs:?}");
        assert!(xs.iter().any(|x| (*x - -1.5).abs() < 1e-6), "{xs:?}");
    }

    /// Exact feature chain: imported planar BREP -> extrude -> export STEP.
    #[test]
    fn exports_exact_extrude_step() {
        let directory =
            std::env::temp_dir().join(format!("sn-worker-extrude-{}", std::process::id()));
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
        let body = cad_body(path.to_str().unwrap());
        assert!((body_volume(&body) - 60.0).abs() < 1e-6, "volume {}", body_volume(&body));
        let (min, max) = body_bounds(&body);
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
        eprintln!(
            "edges={} kinds={:?}",
            mesh.brep_edges.len(),
            mesh.brep_edges
                .iter()
                .map(|e| (e.id.clone(), e.kind.clone(), e.segments.len()))
                .collect::<Vec<_>>()
        );
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
        assert!(error.contains("exist on body 0"), "{error}");

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
        assert_eq!(
            crate::face_ref::face_reference_body_and_index("body:3:face:7"),
            Some((Some(3), 7))
        );
        assert_eq!(crate::face_ref::face_reference_body_and_index("face:2"), Some((None, 2)));
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
    fn part_circular_pattern_target_uses_placed_geometry_instances() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": ["pattern"],
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 4, "axis": "z"}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let (response, cache, _) = respond(request, EngineCache::new());
        assert!(response.node_errors.is_empty(), "{:?}", response.node_errors);
        let mesh = &response.meshes["pattern"].mesh;
        assert!(!mesh.faces.is_empty());
        let (min, max) = bounds(mesh);
        assert!((min[0] + 25.0).abs() < 0.1 && (max[0] - 25.0).abs() < 0.1, "{min:?} {max:?}");
        assert!((min[1] + 25.0).abs() < 0.1 && (max[1] - 25.0).abs() < 0.1, "{min:?} {max:?}");
        assert!(cache.part_entries() > 0);
    }

    #[test]
    fn circular_pattern_full_circle_places_count_copies_around_the_axis() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 4, "axis": "z"}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("circular pattern request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("circular pattern exact part");
        assert_eq!(part.bodies.len(), 4);
        assert_centres(
            &realized_body_centres(part),
            &[[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [-20.0, 0.0, 0.0], [0.0, -20.0, 0.0]],
        );
        let keys: std::collections::HashSet<_> = part.bodies.iter().map(|body| body.key).collect();
        assert_eq!(keys.len(), 1, "every copy must share its source body plan");
    }

    #[test]
    fn circular_pattern_partial_sweep_ends_on_the_requested_angle() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 3, "angle": 90, "axis": "z"}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("partial sweep request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("circular pattern exact part");
        // A partial sweep divides by the gaps: 0, 45 and 90 degrees.
        let diagonal = 20.0 / 2.0_f64.sqrt();
        assert_centres(
            &realized_body_centres(part),
            &[[20.0, 0.0, 0.0], [diagonal, diagonal, 0.0], [0.0, 20.0, 0.0]],
        );
    }

    #[test]
    fn circular_pattern_rotates_around_the_center_point() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 4, "axis": "z", "centerX": 10}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("offset centre request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("circular pattern exact part");
        // The source sits 10 units from the axis through (10, 0, 0).
        assert_centres(
            &realized_body_centres(part),
            &[[20.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0], [10.0, -10.0, 0.0]],
        );
    }

    #[test]
    fn circular_pattern_negative_angle_reverses_the_sweep() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 3, "angle": -90, "axis": "z"}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("negative sweep request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("circular pattern exact part");
        let diagonal = 20.0 / 2.0_f64.sqrt();
        assert_centres(
            &realized_body_centres(part),
            &[[20.0, 0.0, 0.0], [diagonal, -diagonal, 0.0], [0.0, -20.0, 0.0]],
        );
    }

    #[test]
    fn part_sketch_target_uses_geometry_container() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
    fn exact_evaluation_can_fill_the_cache_without_meshing() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["box"],
            "meshTargets": ["box"],
            "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box"}, "cache": {"key": "box"}}
            ],
            "connections": []
        })).expect("exact request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(output.meshes.is_empty(), "exact evaluation produced display geometry");
        assert!(output.cache.part_entries() > 0, "exact evaluation did not cache the Part");
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

    /// Every level of the kernel's Check on an imported record.
    fn check_imported_at_all_levels(brep: &solidcore_record::brep::Brep) {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let imported = solidcore_convert::brep::to_kernel_body(&mut kernel, brep, true)
            .expect("imported record as a kernel body");
        let mut session = nodecraft_ops::OpsSession::new(&mut kernel);
        for level in [
            nodecraft_ops::CheckLevel::Basic,
            nodecraft_ops::CheckLevel::Geometry,
            nodecraft_ops::CheckLevel::All,
        ] {
            let report = session
                .check(&imported, level)
                .unwrap_or_else(|error| panic!("Check {level:?}: {error}"));
            assert!(report.is_valid(), "Check {level:?}: {report}");
        }
    }

    /// The record the real importer produces: its face count, volume and Check.
    fn imported_record(relative: &str, faces: usize) -> f64 {
        let path = fixture(relative);
        let bytes = std::fs::read(&path).expect("read fixture");
        let import = nodecraft_step::read_cad(&bytes, &path)
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        assert!(import.failures.is_empty(), "{relative}: {:?}", import.failures);
        let [body] = import.bodies.as_slice() else {
            panic!("{relative}: {} bodies", import.bodies.len());
        };
        assert!(!body.sheet, "{relative}: a surface body");
        assert_eq!(body.brep.faces.len(), faces, "{relative}: all faces");
        let volume = nodecraft_step::record_volume(&body.brep)
            .unwrap_or_else(|error| panic!("{relative} volume: {error}"));
        check_imported_at_all_levels(&body.brep);
        volume
    }

    /// The exact bodies the real importer hands the engine, one closed solid each.
    fn imported_kernel_body(relative: &str, faces: usize) {
        let path = fixture(relative);
        let breps = nodecraft_mesh::import_step_breps(&path)
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        let [body] = breps.bodies.as_slice() else {
            panic!("{relative}: {} bodies", breps.bodies.len());
        };
        let body = &body.brep;
        assert_eq!(body.faces().len(), faces, "{relative}: all faces");
        assert!(
            nodecraft_solidcore::solidcore::topology::brep::p2::is_closed_manifold(body)
                .expect("closed manifold"),
            "{relative}: not a closed solid"
        );
    }

    /// B-spline solids with the `_KNOTS` knot-spec spelling now import as one closed
    /// exact solid: the importer's `Trims` makes the pcurves the file does not carry.
    #[test]
    fn bspline_solid_with_suffixed_knot_spec_imports() {
        let path = fixture("tests/Sample AP203 Data Files/piston.stp");
        let import = import_step(&path).expect("piston.stp B-spline solid import");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 66);
        assert_eq!(import.diagnostics.open_edges, 0);
        assert!(
            import.diagnostics.fallbacks.is_empty(),
            "piston fell back: {:?}",
            import.diagnostics.fallbacks
        );
        assert!(!import.mesh.faces.is_empty(), "piston produced no triangles");

        let volume = imported_record("tests/Sample AP203 Data Files/piston.stp", 66);
        assert!(volume > 0.0, "piston volume {volume}");
        imported_kernel_body("tests/Sample AP203 Data Files/piston.stp", 66);
    }

    /// `moon_buggy.stp`: 35 of its 77 faces have no closed-form pcurve. The kernel-fitted
    /// trims make every face, and the body is one closed solid with the SMLib oracle's
    /// volume.
    #[test]
    fn moon_buggy_bspline_solid_imports_closed() {
        const ORACLE_VOLUME: f64 = 1_678_611_094.382_282;
        let path = fixture("tests/Sample AP203 Data Files/moon_buggy.stp");
        let import = import_step(&path).expect("moon_buggy.stp B-spline solid import");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 77);
        assert_eq!(import.diagnostics.open_edges, 0);
        assert!(
            import.diagnostics.fallbacks.is_empty(),
            "moon_buggy fell back: {:?}",
            import.diagnostics.fallbacks
        );

        let volume = imported_record("tests/Sample AP203 Data Files/moon_buggy.stp", 77);
        let relative = (volume - ORACLE_VOLUME).abs() / ORACLE_VOLUME;
        assert!(
            relative < 1e-3,
            "moon_buggy volume {volume} is {:.6}% from the oracle {ORACLE_VOLUME}",
            relative * 100.0
        );
        imported_kernel_body("tests/Sample AP203 Data Files/moon_buggy.stp", 77);
    }

    /// One import evaluation must decode the STEP file once: the display mesh
    /// and the exact bodies come from the same cadmpeg model. The old code
    /// called `import_step` and then `import_step_breps`, decoding twice.
    ///
    /// The decode counter is process-wide and the test harness runs tests in
    /// parallel, so this test re-runs itself in a child process whose filter
    /// selects only this test. That keeps the counter measurement isolated.
    #[test]
    fn step_import_decodes_the_file_once() {
        const CHILD: &str = "NODECRAFT_STEP_DECODE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let exe = std::env::current_exe().expect("test binary");
            let status = std::process::Command::new(exe)
                .arg("step_import_decodes_the_file_once")
                .env(CHILD, "1")
                .status()
                .expect("run isolated decode-count test");
            assert!(status.success(), "isolated decode-count test failed");
            return;
        }
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["i"],
            "meshTargets": ["i"],
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": fixture("STEP_TEST_TESS/r1c1_box.step")}, "cache": {"key": "i1"}}
            ],
            "connections": []
        }))
        .expect("request");

        nodecraft_mesh::reset_step_decode_count();
        let first = evaluate_exact(request.clone(), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        assert_eq!(
            nodecraft_mesh::step_decode_count(),
            1,
            "one STEP import evaluation should decode the file once"
        );
        assert_eq!(first.cache.import_hits(), 0);

        // A cache hit reuses the display import and reads exact bodies directly;
        // it still decodes the file exactly once. A new node cache key misses
        // the Part cache so the importer runs, while the content-addressed
        // display import cache still hits.
        nodecraft_mesh::reset_step_decode_count();
        let mut second_request = request;
        second_request.request_id = 2;
        if let Some(cache) = second_request.nodes[0].cache.as_mut() {
            cache.key = Some("i2".into());
        }
        let second = evaluate_exact(second_request, first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert_eq!(
            nodecraft_mesh::step_decode_count(),
            1,
            "a cache-hit evaluation should decode the file once"
        );
        assert_eq!(
            second.cache.import_hits(),
            1,
            "the second evaluation should use the import cache"
        );
    }

    /// Analytic primitive STEP must go through `brep_from_analytic_step` + solidcore
    /// tessellation, not the legacy centered curved mesh.
    #[test]
    fn imports_analytic_cylinder_through_exact_brep() {
        let path = std::env::temp_dir()
            .join(format!("nodecraft-worker-cylinder-{}.step", std::process::id()));
        write_cylinder(path.to_str().unwrap(), 5.0, 10.0);
        let import = import_step(path.to_str().unwrap()).expect("import cylinder STEP");
        let _ = std::fs::remove_file(&path);
        assert!(!import.mesh.faces.is_empty(), "cylinder produced no triangles");
        assert_eq!(import.diagnostics.kind, "solidcore-brep");
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.brep_faces, 3);
        assert!(
            import.diagnostics.fallbacks.is_empty(),
            "analytic fallbacks: {:?}",
            import.diagnostics.fallbacks
        );
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
        let payload = serde_json::to_value(&response.response).expect("response json");
        assert!(payload.get("diagnostics").is_none());

        let debug = request_for(&path, true);
        assert!(diagnostics_requested(&debug));
        let (response, _cache, diagnostics) = respond(debug, EngineCache::new());
        assert!(response.node_errors.is_empty(), "node errors: {:?}", response.node_errors);
        let summarised = diagnostics.get("step").expect("diagnostics for stepimporter node");
        let NodeDiagnostic::Step(summarised) =
            summarised.first().expect("one step diagnostic for the import node");
        assert_eq!(summarised.kind, "solidcore-brep");
        assert_eq!(summarised.breps, 1);
        assert_eq!(summarised.brep_faces, 6);
        assert!(summarised.fallbacks.is_empty());
        // The response JSON stays one object per node id.
        let json = NodeDiagnostic::Step(summarised.clone()).to_json();
        assert_eq!(json["kind"], "solidcore-brep");
        assert_eq!(json["brepFaces"], 6);
        // The STEP reader's diagnostics report its own import mesh; the display
        // mesh now comes from solidcore-tess, so only require the reported
        // diagnostic count to be present, not equal to the new mesher's count.
        assert!(json["meshTriangles"].as_u64().unwrap_or(0) > 0);
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

    fn copy_fixture(label: &str, source: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let directory =
            std::env::temp_dir().join(format!("sn-import-cache-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp directory");
        let target = directory.join("renamed.step");
        std::fs::copy(fixture(source), &target).expect("copy fixture");
        (directory, target)
    }

    fn importer_request(request_id: u64, path: &std::path::Path) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": request_id,
            "targets": ["s"],
            "meshTargets": ["s"],
            "nodes": [{
                "id": "s",
                "type": "stepimporter",
                "values": {"path": path.display().to_string()},
                "cache": {"key": format!("s{request_id}")}
            }],
            "connections": []
        }))
        .expect("request")
    }

    /// The cache identity is the content, not the path: a renamed copy of the
    /// same bytes is a hit.
    #[test]
    fn step_import_cache_hits_a_renamed_copy_of_the_same_content() {
        let (directory, renamed) =
            copy_fixture("renamed", "tests/Sample AP203 Data Files/block.stp");
        let original = std::path::PathBuf::from(fixture("tests/Sample AP203 Data Files/block.stp"));
        assert_eq!(crate::import_io::hash_file(&original), crate::import_io::hash_file(&renamed));
        let first = evaluate_exact(importer_request(1, &original), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let second = evaluate_exact(importer_request(2, &renamed), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(second.cache.import_hits() > 0, "renamed copy must hit the content-hash cache");
        std::fs::remove_dir_all(directory).ok();
    }

    /// When the file is missing, the importer reads the embedded copy and says
    /// so in its warnings; every embedded body still comes through.
    #[test]
    fn missing_source_imports_the_embedded_copy_and_warns() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;

        let directory =
            std::env::temp_dir().join(format!("sn-ingest-fallback-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp directory");
        let source = directory.join("part.step");
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let breps: Vec<_> = (0..2)
            .map(|index| {
                create_box(
                    &mut kernel,
                    Point3::new(index as f64 * 20.0, 0.0, 0.0),
                    10.0,
                    10.0,
                    10.0,
                )
                .expect("box")
            })
            .collect();
        nodecraft_step::write_breps_exact(source.to_str().expect("utf-8"), &breps).expect("write");
        let bytes = std::fs::read(&source).expect("read source");
        let record = crate::import_io::ingest_record("part.step", &bytes).expect("record");
        std::fs::remove_file(&source).expect("remove source");

        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["s"],
            "meshTargets": [],
            "nodes": [{
                "id": "s",
                "type": "stepimporter",
                "values": {
                    "path": "part.step",
                    "pathMode": "relative",
                    "ingest": true,
                    "ingestHash": record.content_hash,
                },
                "cache": {"key": "s"}
            }],
            "connections": []
        }))
        .expect("request");
        let imports = crate::ImportContext::new(Some(directory.clone()), vec![record.clone()]);
        let output = evaluate_exact_with_imports(request, EngineCache::new(), None, imports);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("s").expect("embedded part");
        assert_eq!(part.bodies.len(), 2, "every body in the embedded copy is kept");
        let warnings = output.response.node_warnings.get("s").expect("fallback warning");
        // cadmpeg 0.6.12 also reports a product definition whose shape representation
        // carries no body of its own (the assembly product above the two parts), so the
        // fallback warning is asserted by its exact wording rather than as the only line.
        let fallback = "Original not found; using the embedded copy of \u{201c}part.step\u{201d}.";
        assert!(
            warnings.iter().any(|warning| warning == fallback),
            "the warning names the embedded copy without a hash: {warnings:?}"
        );
        std::fs::remove_dir_all(directory).ok();
    }

    /// A content change is a miss even when the path, size and mtime are close
    /// enough that an old metadata key might have collided.
    #[test]
    fn step_import_cache_misses_when_the_content_changes() {
        let (directory, renamed) =
            copy_fixture("changed", "tests/Sample AP203 Data Files/block.stp");
        let first = evaluate_exact(importer_request(1, &renamed), EngineCache::new(), None);
        assert!(first.response.node_errors.is_empty(), "{:?}", first.response.node_errors);
        let mut bytes = std::fs::read(&renamed).expect("read");
        bytes.extend_from_slice(b"\n");
        std::fs::write(&renamed, bytes).expect("append");
        let second = evaluate_exact(importer_request(2, &renamed), first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert_eq!(second.cache.import_hits(), 0, "changed content must miss the import cache");
        std::fs::remove_dir_all(directory).ok();
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
        let mesh = output.meshes.get("loft").expect("loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
    }

    #[test]
    fn loft_between_sketch_profiles_uses_solidcore() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let mesh = output.meshes.get("loft").expect("loft mesh");
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
        let curve = output.meshes.get("curve").expect("spline display mesh");
        assert!(!curve.mesh.faces.is_empty(), "spline display curve is empty");
        let mesh = output.meshes.get("loft").expect("loft mesh");
        assert!(!mesh.mesh.faces.is_empty());
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 10.0).abs() < 0.1, "spline loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn surface_loft_has_no_end_caps() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let mesh = output.meshes.get("loft").expect("surface loft mesh");
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
            sketch.add_point(solidcore_record::Vec2::new(x, y), None).unwrap();
        }
        sketch
            .add_curve(
                crate::sketch::CurveKind::Spline {
                    pts: vec![1, 2, 3, 4],
                    control: false,
                    degree: 3,
                    periodic: false,
                    nurbs: None,
                    handles: Vec::new(),
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
        let mesh = output.meshes.get("loft").expect("surface loft mesh");
        assert!(!mesh.mesh.faces.is_empty(), "open spline loft produced no faces");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 10.0).abs() < 0.1, "surface loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn sketch_output_carries_translate_rotation() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let output = eval.part_sketch_output("top-at").expect("typed sketch output");
        let point = output.point(10.0, 0.0);
        assert!(point[0].abs() < 1.0e-9, "{point:?}");
        assert!((point[1] - 10.0).abs() < 1.0e-9, "{point:?}");
        assert!((point[2] - 7.19).abs() < 1.0e-9, "{point:?}");
    }

    #[test]
    fn loft_after_translate_edit_keeps_typed_sketch_output() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let mesh = second.meshes.get("loft").expect("loft mesh");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 20.0).abs() < 0.05, "edited loft height {}..{}", min_z, max_z);
    }

    #[test]
    fn zero_extrude_amount_is_rejected() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record::Vec2::new(-1.0, 0.0), 51.960239979110156, None, None)
            .unwrap();
        let error = sketch_extrude_brep(&SketchOutput::new(sketch), 0.0).expect_err("zero amount");
        assert!(error.contains("greater than zero"), "{error}");
    }

    #[test]
    fn extrude_consumes_a_circle_sketch_profile() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record::Vec2::new(1.0, 1.0), 33.18108250839226, None, None)
            .unwrap();
        let brep =
            sketch_extrude_brep(&SketchOutput::new(sketch), 20.0).expect("circle extrude brep");
        assert!(!brep.faces().ids().collect::<Vec<_>>().is_empty(), "circle extrude has no faces");
    }

    #[test]
    fn circle_extrude_geometry_is_a_cylinder() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch
            .add_circle(solidcore_record::Vec2::new(-1.0, 0.0), 51.960239979110156, None, None)
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
        let mesh = output.meshes.get("extrude").expect("extrude mesh");
        let volume = mesh_signed_volume(&mesh.mesh);
        let expected = std::f64::consts::PI * 51.960239979110156_f64.powi(2) * 6.7;
        let (min, max) = nodecraft_kernel::bounds(&mesh.mesh).expect("bounds");
        let mut side_radial_error = 0.0_f64;
        for vertex in &mesh.mesh.vertices {
            if vertex[2] > 1.0e-4 && vertex[2] < 6.7 - 1.0e-4 {
                let dx = f64::from(vertex[0]) + 1.0;
                let dy = f64::from(vertex[1]);
                side_radial_error =
                    side_radial_error.max(((dx * dx + dy * dy).sqrt() - 51.960239979110156).abs());
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
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let mesh = output.meshes.get("extrude").expect("extrude mesh");
        assert!(!mesh.mesh.faces.is_empty(), "extrude produced no faces");
        let min_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::INFINITY, f32::min);
        let max_z = mesh.mesh.vertices.iter().map(|v| v[2]).fold(f32::NEG_INFINITY, f32::max);
        assert!((max_z - min_z - 20.0).abs() < 0.05, "height {}..{}", min_z, max_z);
    }

    fn extrude_request(
        sketch: &crate::sketch::Sketch,
        transform: Option<(&str, serde_json::Value)>,
    ) -> Request {
        let sketch_value = serde_json::json!({"sketch": serde_json::to_value(sketch).unwrap()});
        let (nodes, connections) = match transform {
            Some((kind, values)) => (
                vec![
                    serde_json::json!({"id": "sketch", "type": "sketch", "values": sketch_value, "cache": {"key": "sketch"}}),
                    serde_json::json!({"id": "move", "type": kind, "values": values, "cache": {"key": "move"}}),
                    serde_json::json!({"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}),
                ],
                vec![
                    serde_json::json!({"from": "sketch", "to": "move", "in": "shape", "out": "sketch"}),
                    serde_json::json!({"from": "move", "to": "extrude", "in": "profile", "out": "shape"}),
                ],
            ),
            None => (
                vec![
                    serde_json::json!({"id": "sketch", "type": "sketch", "values": sketch_value, "cache": {"key": "sketch"}}),
                    serde_json::json!({"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}),
                ],
                vec![
                    serde_json::json!({"from": "sketch", "to": "extrude", "in": "profile", "out": "sketch"}),
                ],
            ),
        };
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["extrude"],
            "meshTargets": [],
            "nodes": nodes,
            "connections": connections,
        }))
        .expect("request")
    }

    /// As [`extrude_request`], from a spline node rather than a sketch node.
    fn spline_extrude_request(
        values: &serde_json::Value,
        transform: Option<(&str, serde_json::Value)>,
    ) -> Request {
        let (nodes, connections) = match transform {
            Some((kind, transform_values)) => (
                vec![
                    serde_json::json!({"id": "spline", "type": "spline", "values": values, "cache": {"key": "spline"}}),
                    serde_json::json!({"id": "move", "type": kind, "values": transform_values, "cache": {"key": "move"}}),
                    serde_json::json!({"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}),
                ],
                vec![
                    serde_json::json!({"from": "spline", "to": "move", "in": "shape", "out": "sketch"}),
                    serde_json::json!({"from": "move", "to": "extrude", "in": "profile", "out": "shape"}),
                ],
            ),
            None => (
                vec![
                    serde_json::json!({"id": "spline", "type": "spline", "values": values, "cache": {"key": "spline"}}),
                    serde_json::json!({"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}),
                ],
                vec![
                    serde_json::json!({"from": "spline", "to": "extrude", "in": "profile", "out": "sketch"}),
                ],
            ),
        };
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["extrude"],
            "meshTargets": [],
            "nodes": nodes,
            "connections": connections,
        }))
        .expect("request")
    }

    fn brep_counts(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> (usize, usize, usize) {
        (brep.faces().ids().count(), brep.edges().ids().count(), brep.vertices().ids().count())
    }

    /// The sketch extrude test entry point: run the production kernel-aware extrude on
    /// a local NodeCraft-profile kernel and hand the tests its B-rep. The tests never
    /// configure (or read) the process-wide ops session.
    fn sketch_extrude_brep(
        output: &SketchOutput,
        distance: f64,
    ) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        super::sketch_extrude_brep(&mut kernel, output, distance)?
            .into_brep()
            .map_err(|error| format!("Extrude: {error}"))
    }

    fn brep_volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        nodecraft_layers::kernel_volume(&mut kernel, brep).expect("brep volume")
    }

    fn brep_bounds(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> ([f64; 3], [f64; 3]) {
        let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for vertex in brep.vertex_list() {
            let point = brep.vertex(*vertex).expect("vertex").point;
            for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                min[axis] = min[axis].min(value);
                max[axis] = max[axis].max(value);
            }
        }
        (min, max)
    }

    /// Radii of the exact circular edges in `brep`.
    fn brep_circle_radii(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> Vec<f64> {
        use nodecraft_solidcore::solidcore::curve::curve::CurveKind;
        let mut radii = Vec::new();
        for edge in brep.edges().ids() {
            let Ok(edge) = brep.edge(edge) else { continue };
            let Some(curve) = edge.curve.as_ref() else { continue };
            if let CurveKind::Ellipse(ellipse) = curve.kind()
                && ellipse.is_circle()
            {
                radii.push(ellipse.x_radius());
            }
        }
        radii
    }

    /// How many edges are exact circular arcs, not a NURBS form of one.
    fn brep_circle_edges(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> usize {
        brep_circle_radii(brep).len()
    }

    /// How many edges are exact B-spline (or Bézier) curves.
    fn brep_bspline_edges(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> usize {
        use nodecraft_solidcore::solidcore::curve::curve::CurveKind;
        brep.edges()
            .ids()
            .filter(|edge| {
                brep.edge(*edge)
                    .ok()
                    .and_then(|edge| edge.curve.as_ref())
                    .is_some_and(|curve| matches!(curve.kind(), CurveKind::BSpline(_)))
            })
            .count()
    }

    /// How many faces lie on a cylinder (the kernel's analytic cone kind with a
    /// zero half-angle, not a NURBS surface of extrusion).
    fn brep_cylindrical_faces(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> usize {
        use nodecraft_solidcore::solidcore::surface::surface::SurfaceKind;
        brep.faces()
            .ids()
            .filter(|face| {
                brep.face(*face)
                    .ok()
                    .and_then(|face| face.surface.as_ref())
                    .is_some_and(|surface| {
                        matches!(surface.kind(), SurfaceKind::Cone(cone) if cone.is_cylinder())
                    })
            })
            .count()
    }

    /// The sketch extrude must run on the kernel the worker lends `OpsSession`, not on a
    /// fresh NodeCraft-profile kernel or the process-wide session. A kernel with the upstream
    /// default profile turns the profile's partial arc into a NURBS side wall; NodeCraft's
    /// profile keeps it an analytic circle/cylinder. The task is handed both kernels and must
    /// follow each: the old layers helper ignored the worker and always built its own profile
    /// kernel, so it would give a cylinder on the default kernel too.
    #[test]
    fn sketch_extrude_uses_the_lent_worker_kernel() {
        let output = std::sync::Arc::new(SketchOutput::new(rounded_profile_sketch()));
        let task = |output: std::sync::Arc<SketchOutput>| {
            crate::node_task::ExactNodeTask::ExtrudeSketch { output, amount: 10.0 }
        };

        // Upstream default fixes: the partial arc is a surface of extrusion, so its edges
        // are NURBS, not circles.
        let mut default_kernel = nodecraft_solidcore::solidcore::kernel::Kernel::default();
        assert!(
            !default_kernel.fixes.linear_sweep_arc_is_cylinder,
            "the default kernel does not have the NodeCraft fix"
        );
        let crate::node_task::NodeOutput::Shape(default_body) =
            task(output.clone()).run(&mut default_kernel).expect("default-kernel extrude")
        else {
            panic!("ExtrudeSketch has a shape output");
        };
        assert_eq!(brep_circle_edges(default_body.brep()), 0, "default kernel arc is not analytic");
        assert!(brep_bspline_edges(default_body.brep()) > 0, "default kernel arc is NURBS");
        assert_eq!(
            brep_cylindrical_faces(default_body.brep()),
            0,
            "default kernel makes no cylinder"
        );

        // NodeCraft's profile, under the evaluation path's `EachNode`: analytic and unproved.
        let mut profile_kernel = nodecraft_solidcore::kernel::kernel();
        let crate::node_task::NodeOutput::Shape(profile_body) =
            nodecraft_ops::validating(nodecraft_ops::Validation::EachNode, || {
                task(output).run(&mut profile_kernel)
            })
            .expect("profile-kernel extrude")
        else {
            panic!("ExtrudeSketch has a shape output");
        };
        assert_eq!(brep_circle_edges(profile_body.brep()), 2, "profile arc edges are circles");
        assert_eq!(brep_bspline_edges(profile_body.brep()), 0, "profile arc edges are not NURBS");
        assert_eq!(
            brep_cylindrical_faces(profile_body.brep()),
            1,
            "profile arc side is a cylinder"
        );
        assert_eq!(
            profile_body.proof_state(),
            nodecraft_ops::ProofState::Unknown,
            "no proof on the evaluation path"
        );
        assert_eq!(profile_body.proof_attempts(), 0);
    }

    #[test]
    fn rotated_sketch_extrude_keeps_the_exact_arc_profile() {
        let sketch = rounded_profile_sketch();
        let plain = evaluate_exact(extrude_request(&sketch, None), EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let plain_part = plain.exact_parts.get("extrude").expect("plain extrude part");
        let plain_body = &plain_part.bodies[0].brep;
        let plain_volume = exact_part_volume(plain_part);
        // The profile's arc is exact and analytic on any plane: its top and
        // bottom edges are circles, its side face a cylinder, never the 0.001
        // polyline's chords or a rational NURBS stand-in.
        assert_eq!(brep_circle_edges(plain_body), 2, "plain sketch arc edges are circular");
        assert_eq!(brep_bspline_edges(plain_body), 0, "plain sketch arc edges are not NURBS");
        assert_eq!(brep_cylindrical_faces(plain_body), 1, "plain sketch arc side is a cylinder");

        let rotated = evaluate_exact(
            extrude_request(&sketch, Some(("translate", serde_json::json!({"rx": 90.0})))),
            EngineCache::new(),
            None,
        );
        assert!(rotated.response.node_errors.is_empty(), "{:?}", rotated.response.node_errors);
        let rotated_part = rotated.exact_parts.get("extrude").expect("rotated extrude part");
        let rotated_body = &rotated_part.bodies[0].brep;
        assert_eq!(brep_counts(rotated_body), brep_counts(plain_body), "rotated topology");
        assert_eq!(
            brep_circle_edges(rotated_body),
            brep_circle_edges(plain_body),
            "rotated arc edges are circular"
        );
        assert_eq!(brep_bspline_edges(rotated_body), 0, "rotated arc edges are not NURBS");
        assert_eq!(brep_cylindrical_faces(rotated_body), 1, "rotated arc side is a cylinder");
        let rotated_volume = exact_part_volume(rotated_part);
        assert!(
            (rotated_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, rotated {rotated_volume}"
        );
        assert_exact_bounds(rotated_part, [-20.0, -10.0, -15.0], [20.0, 0.0, 15.0]);
    }

    #[test]
    fn uniform_scaled_sketch_extrude_scales_arcs_and_volume() {
        let plain = SketchOutput::new(rounded_profile_sketch());
        let plain_brep = sketch_extrude_brep(&plain, 10.0).expect("plain extrude");
        let scaled = plain.transformed(&BodyTransform::Scale { factor: 2.0 });
        let scaled_brep = sketch_extrude_brep(&scaled, 10.0).expect("scaled extrude");
        let plain_volume = brep_volume(&plain_brep);
        let scaled_volume = brep_volume(&scaled_brep);
        let expected = plain_volume * 8.0;
        assert!(
            (scaled_volume - expected).abs() <= expected.abs() * 1.0e-9,
            "{scaled_volume} != {expected}"
        );
        assert_eq!(brep_counts(&scaled_brep), brep_counts(&plain_brep), "scaled topology");
        assert_eq!(brep_circle_edges(&plain_brep), 2, "plain arc edges are circular");
        assert_eq!(brep_circle_edges(&scaled_brep), 2, "scaled arc edges are circular");
        assert_eq!(brep_bspline_edges(&scaled_brep), 0, "scaled arc edges are not NURBS");
        assert_eq!(brep_cylindrical_faces(&scaled_brep), 1, "scaled arc side is a cylinder");
        let (min, max) = brep_bounds(&scaled_brep);
        for (actual, expected) in
            min.into_iter().zip([-40.0, -30.0, 0.0]).chain(max.into_iter().zip([40.0, 30.0, 20.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }
        // The 2D arc keeps its shape and doubles its radius; the plane stays unit.
        let plain_geometry = plain.record_geometry().expect("plain record geometry");
        let plain_radius = plain_geometry
            .region
            .outer
            .segs
            .iter()
            .find_map(|segment| match *segment {
                solidcore_record::Seg2::Arc { radius, .. } => Some(radius),
                _ => None,
            })
            .expect("plain exact arc");
        let geometry = scaled.record_geometry().expect("scaled record geometry");
        let radii: Vec<f64> = geometry
            .region
            .outer
            .segs
            .iter()
            .filter_map(|segment| match *segment {
                solidcore_record::Seg2::Arc { radius, .. } => Some(radius),
                _ => None,
            })
            .collect();
        assert_eq!(radii.len(), 1, "one exact arc");
        assert!((radii[0] - plain_radius * 2.0).abs() < 1.0e-12, "arc radius {}", radii[0]);
    }

    #[test]
    fn rotated_holed_sketch_extrude_keeps_the_hole_exact() {
        let mut sketch = rounded_profile_sketch();
        sketch
            .add_circle(solidcore_record::Vec2::new(0.0, 0.0), 5.0, None, None)
            .expect("hole circle");
        let plain = evaluate_exact(extrude_request(&sketch, None), EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let plain_part = plain.exact_parts.get("extrude").expect("plain holed extrude");
        let plain_body = &plain_part.bodies[0].brep;
        let plain_volume = exact_part_volume(plain_part);
        let plain_counts = brep_counts(plain_body);
        assert!(
            brep_circle_radii(plain_body).iter().any(|radius| (radius - 5.0).abs() < 1.0e-12),
            "hole circle"
        );
        assert_eq!(brep_circle_edges(plain_body), 6, "outer arc and hole are circular");
        assert_eq!(brep_bspline_edges(plain_body), 0, "holed arc extrude has no NURBS edges");
        assert_eq!(
            brep_cylindrical_faces(plain_body),
            3,
            "outer and hole side faces are cylinders"
        );

        let outer = evaluate_exact(
            extrude_request(&rounded_profile_sketch(), None),
            EngineCache::new(),
            None,
        );
        let outer_part = outer.exact_parts.get("extrude").expect("outer extrude");
        let expected = exact_part_volume(outer_part) - std::f64::consts::PI * 25.0 * 10.0;
        assert!(
            (plain_volume - expected).abs() <= expected.abs() * 1.0e-9,
            "{plain_volume} != {expected}"
        );

        let rotated = evaluate_exact(
            extrude_request(&sketch, Some(("translate", serde_json::json!({"rx": 90.0})))),
            EngineCache::new(),
            None,
        );
        assert!(rotated.response.node_errors.is_empty(), "{:?}", rotated.response.node_errors);
        let rotated_part = rotated.exact_parts.get("extrude").expect("rotated holed extrude");
        let rotated_body = &rotated_part.bodies[0].brep;
        assert_eq!(brep_counts(rotated_body), plain_counts, "rotated holed topology");
        let rotated_volume = exact_part_volume(rotated_part);
        assert!(
            (rotated_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, rotated {rotated_volume}"
        );
        assert!(
            brep_circle_radii(rotated_body).iter().any(|radius| (radius - 5.0).abs() < 1.0e-12),
            "rotated hole circle"
        );
        assert_eq!(brep_circle_edges(rotated_body), 6, "rotated outer arc and hole are circular");
        assert_eq!(
            brep_bspline_edges(rotated_body),
            0,
            "rotated holed arc extrude has no NURBS edges"
        );
        assert_eq!(
            brep_cylindrical_faces(rotated_body),
            3,
            "rotated outer and hole side faces are cylinders"
        );
        assert_exact_bounds(rotated_part, [-20.0, -10.0, -15.0], [20.0, 0.0, 15.0]);
    }

    #[test]
    fn non_uniform_sketch_scale_is_refused_not_faceted() {
        let mut output = SketchOutput::new(rounded_profile_sketch());
        output.rotation = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let error = sketch_extrude_brep(&output, 10.0).expect_err("non-uniform scale");
        assert!(error.contains("non-uniform scale"), "{error}");
    }

    #[test]
    fn translated_sketch_extrude_keeps_volume_and_moves_bounds() {
        let sketch = rounded_profile_sketch();
        let plain = evaluate_exact(extrude_request(&sketch, None), EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let plain_part = plain.exact_parts.get("extrude").expect("plain extrude part");
        assert_eq!(plain_part.bodies.len(), 1);
        let plain_volume = exact_part_volume(plain_part);
        assert!(plain_volume > 0.0, "plain volume {plain_volume}");
        assert_exact_bounds(plain_part, [-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]);

        let moved = evaluate_exact(
            extrude_request(
                &sketch,
                Some(("translate", serde_json::json!({"x": 15.0, "z": 7.0, "rz": 90.0}))),
            ),
            EngineCache::new(),
            None,
        );
        assert!(moved.response.node_errors.is_empty(), "{:?}", moved.response.node_errors);
        let moved_part = moved.exact_parts.get("extrude").expect("translated extrude part");
        assert_eq!(moved_part.bodies.len(), 1);
        let moved_volume = exact_part_volume(moved_part);
        assert!(
            (moved_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, moved {moved_volume}"
        );
        assert_exact_bounds(moved_part, [0.0, -20.0, 7.0], [30.0, 20.0, 17.0]);

        let (plain_faces, plain_edges, plain_vertices) = (
            plain_part.bodies[0].brep.faces().ids().count(),
            plain_part.bodies[0].brep.edges().ids().count(),
            plain_part.bodies[0].brep.vertices().ids().count(),
        );
        let moved_body = &moved_part.bodies[0].brep;
        assert_eq!(moved_body.faces().ids().count(), plain_faces);
        assert_eq!(moved_body.edges().ids().count(), plain_edges);
        assert_eq!(moved_body.vertices().ids().count(), plain_vertices);
    }

    #[test]
    fn mirrored_sketch_extrude_is_a_valid_outward_solid() {
        let sketch = rounded_profile_sketch();
        let plain = evaluate_exact(extrude_request(&sketch, None), EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let plain_part = plain.exact_parts.get("extrude").expect("plain extrude part");
        let plain_body = &plain_part.bodies[0].brep;
        let plain_volume = exact_part_volume(plain_part);
        let plain_counts = brep_counts(plain_body);

        let mirrored = evaluate_exact(
            extrude_request(
                &sketch,
                Some((
                    "mirror",
                    serde_json::json!({"plane": "yz", "offset": 0.0, "combine": false}),
                )),
            ),
            EngineCache::new(),
            None,
        );
        assert!(mirrored.response.node_errors.is_empty(), "{:?}", mirrored.response.node_errors);
        let part = mirrored.exact_parts.get("extrude").expect("mirrored extrude part");
        assert_eq!(part.bodies.len(), 1);
        let volume = exact_part_volume(part);
        assert!(volume > 0.0, "mirrored volume {volume}");
        assert!(
            (volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "mirrored volume {volume}, plain {plain_volume}"
        );
        assert_eq!(brep_counts(&part.bodies[0].brep), plain_counts, "yz mirror topology");
        assert_eq!(
            brep_circle_edges(&part.bodies[0].brep),
            brep_circle_edges(plain_body),
            "yz mirror arc edges are circular"
        );
        assert_eq!(
            brep_bspline_edges(&part.bodies[0].brep),
            0,
            "yz mirror arc edges are not NURBS"
        );
        assert_eq!(
            brep_cylindrical_faces(&part.bodies[0].brep),
            1,
            "yz mirror arc side is a cylinder"
        );
        // A yz mirror keeps the sweep on the original +z side; the profile is
        // re-oriented so the solid is the mirror of the extrusion, not its
        // opposite-side twin.
        assert_exact_bounds(part, [-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]);

        let through_sketch_plane = evaluate_exact(
            extrude_request(
                &sketch,
                Some((
                    "mirror",
                    serde_json::json!({"plane": "xy", "offset": 0.0, "combine": false}),
                )),
            ),
            EngineCache::new(),
            None,
        );
        assert!(
            through_sketch_plane.response.node_errors.is_empty(),
            "{:?}",
            through_sketch_plane.response.node_errors
        );
        let part =
            through_sketch_plane.exact_parts.get("extrude").expect("xy-mirrored extrude part");
        assert_eq!(part.bodies.len(), 1);
        let volume = exact_part_volume(part);
        assert!(volume > 0.0);
        assert!(
            (volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "xy mirrored volume {volume}, plain {plain_volume}"
        );
        assert_eq!(brep_counts(&part.bodies[0].brep), plain_counts, "xy mirror topology");
        assert_eq!(
            brep_circle_edges(&part.bodies[0].brep),
            brep_circle_edges(plain_body),
            "xy mirror arc edges are circular"
        );
        assert_eq!(
            brep_bspline_edges(&part.bodies[0].brep),
            0,
            "xy mirror arc edges are not NURBS"
        );
        assert_eq!(
            brep_cylindrical_faces(&part.bodies[0].brep),
            1,
            "xy mirror arc side is a cylinder"
        );
        // A mirror through the sketch plane sweeps the other way: the image of
        // the 0..10 extrusion is -10..0.
        assert_exact_bounds(part, [-20.0, -15.0, -10.0], [20.0, 15.0, 0.0]);
    }

    #[test]
    fn periodic_spline_node_extrudes_exact_on_every_plane() {
        let values = periodic_spline_node_values();
        let plain = evaluate_exact(spline_extrude_request(&values, None), EngineCache::new(), None);
        assert!(plain.response.node_errors.is_empty(), "{:?}", plain.response.node_errors);
        let plain_part = plain.exact_parts.get("extrude").expect("plain spline extrude part");
        assert_eq!(plain_part.bodies.len(), 1);
        let plain_body = &plain_part.bodies[0].brep;
        let plain_volume = exact_part_volume(plain_part);
        assert!(plain_volume > 0.0, "plain spline volume {plain_volume}");
        assert_eq!(brep_bspline_edges(plain_body), 24, "closed periodic spline spans are exact");
        assert_eq!(brep_circle_edges(plain_body), 0, "a spline profile has no circular edges");
        let plain_counts = brep_counts(plain_body);
        assert_exact_bounds(plain_part, [-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]);

        let rotated = evaluate_exact(
            spline_extrude_request(&values, Some(("translate", serde_json::json!({"rx": 90.0})))),
            EngineCache::new(),
            None,
        );
        assert!(rotated.response.node_errors.is_empty(), "{:?}", rotated.response.node_errors);
        let rotated_part = rotated.exact_parts.get("extrude").expect("rotated spline extrude part");
        let rotated_body = &rotated_part.bodies[0].brep;
        let rotated_volume = exact_part_volume(rotated_part);
        assert!(
            (rotated_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, rotated {rotated_volume}"
        );
        assert_eq!(brep_counts(rotated_body), plain_counts, "rotated spline topology");
        assert_eq!(brep_bspline_edges(rotated_body), 24, "rotated spline spans are exact");
        assert_exact_bounds(rotated_part, [-20.0, -10.0, -15.0], [20.0, 0.0, 15.0]);

        let mirrored = evaluate_exact(
            spline_extrude_request(
                &values,
                Some((
                    "mirror",
                    serde_json::json!({"plane": "yz", "offset": 0.0, "combine": false}),
                )),
            ),
            EngineCache::new(),
            None,
        );
        assert!(mirrored.response.node_errors.is_empty(), "{:?}", mirrored.response.node_errors);
        let mirrored_part =
            mirrored.exact_parts.get("extrude").expect("mirrored spline extrude part");
        let mirrored_body = &mirrored_part.bodies[0].brep;
        let mirrored_volume = exact_part_volume(mirrored_part);
        assert!(
            (mirrored_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, mirrored {mirrored_volume}"
        );
        assert_eq!(brep_counts(mirrored_body), plain_counts, "mirrored spline topology");
        assert_eq!(brep_bspline_edges(mirrored_body), 24, "mirrored spline spans are exact");
        assert_exact_bounds(mirrored_part, [-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]);

        let sketch = spline_sketch_from_values(&values).expect("spline node sketch");
        let plain_output = SketchOutput::new(sketch);
        let scaled_output = plain_output.transformed(&BodyTransform::Scale { factor: 2.0 });
        let plain_brep = sketch_extrude_brep(&plain_output, 10.0).expect("plain spline extrude");
        let scaled_brep = sketch_extrude_brep(&scaled_output, 10.0).expect("scaled spline extrude");
        let scaled_volume = brep_volume(&scaled_brep);
        let expected = plain_volume * 8.0;
        assert!(
            (scaled_volume - expected).abs() <= expected.abs() * 1.0e-9,
            "{scaled_volume} != {expected}"
        );
        assert_eq!(brep_counts(&scaled_brep), plain_counts, "scaled spline topology");
        assert_eq!(brep_bspline_edges(&scaled_brep), 24, "scaled spline spans are exact");
        assert_eq!(brep_bspline_edges(&plain_brep), 24, "plain spline spans are exact");
        let (scaled_min, scaled_max) = brep_bounds(&scaled_brep);
        for (actual, expected) in scaled_min
            .into_iter()
            .zip([-40.0, -30.0, 0.0])
            .chain(scaled_max.into_iter().zip([40.0, 30.0, 20.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }
    }

    #[test]
    fn mixed_line_and_spline_sketch_extrudes_exact_on_every_plane() {
        let output = SketchOutput::new(mixed_line_spline_sketch());
        let plain_brep = sketch_extrude_brep(&output, 10.0).expect("plain mixed extrude");
        let plain_volume = brep_volume(&plain_brep);
        assert!(plain_volume > 0.0, "plain mixed volume {plain_volume}");
        assert_eq!(brep_bspline_edges(&plain_brep), 10, "mixed spline spans are exact");
        assert_eq!(
            brep_circle_edges(&plain_brep),
            0,
            "a mixed spline profile has no circular edges"
        );
        let plain_counts = brep_counts(&plain_brep);
        let (plain_min, plain_max) = brep_bounds(&plain_brep);
        for (actual, expected) in plain_min
            .into_iter()
            .zip([-20.0, -15.0, 0.0])
            .chain(plain_max.into_iter().zip([27.0, 15.0, 10.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }

        let rotated = output.transformed(&BodyTransform::Place {
            offset: [0.0, 0.0, 0.0],
            rotation: [90.0, 0.0, 0.0],
        });
        let rotated_brep = sketch_extrude_brep(&rotated, 10.0).expect("rotated mixed extrude");
        let rotated_volume = brep_volume(&rotated_brep);
        assert!(
            (rotated_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, rotated {rotated_volume}"
        );
        assert_eq!(brep_counts(&rotated_brep), plain_counts, "rotated mixed topology");
        assert_eq!(brep_bspline_edges(&rotated_brep), 10, "rotated mixed spans are exact");
        let (rotated_min, rotated_max) = brep_bounds(&rotated_brep);
        for (actual, expected) in rotated_min
            .into_iter()
            .zip([-20.0, -10.0, -15.0])
            .chain(rotated_max.into_iter().zip([27.0, 0.0, 15.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }

        let mirrored = output.transformed(&BodyTransform::Mirror { plane: "yz", offset: 0.0 });
        let mirrored_brep = sketch_extrude_brep(&mirrored, 10.0).expect("mirrored mixed extrude");
        let mirrored_volume = brep_volume(&mirrored_brep);
        assert!(
            (mirrored_volume - plain_volume).abs() <= plain_volume.abs() * 1.0e-9,
            "plain {plain_volume}, mirrored {mirrored_volume}"
        );
        assert_eq!(brep_counts(&mirrored_brep), plain_counts, "mirrored mixed topology");
        assert_eq!(brep_bspline_edges(&mirrored_brep), 10, "mirrored mixed spans are exact");
        let (mirrored_min, mirrored_max) = brep_bounds(&mirrored_brep);
        for (actual, expected) in mirrored_min
            .into_iter()
            .zip([-27.0, -15.0, 0.0])
            .chain(mirrored_max.into_iter().zip([20.0, 15.0, 10.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }

        let scaled = output.transformed(&BodyTransform::Scale { factor: 2.0 });
        let scaled_brep = sketch_extrude_brep(&scaled, 10.0).expect("scaled mixed extrude");
        let scaled_volume = brep_volume(&scaled_brep);
        let expected = plain_volume * 8.0;
        assert!(
            (scaled_volume - expected).abs() <= expected.abs() * 1.0e-9,
            "{scaled_volume} != {expected}"
        );
        assert_eq!(brep_counts(&scaled_brep), plain_counts, "scaled mixed topology");
        assert_eq!(brep_bspline_edges(&scaled_brep), 10, "scaled mixed spans are exact");
        let (scaled_min, scaled_max) = brep_bounds(&scaled_brep);
        for (actual, expected) in scaled_min
            .into_iter()
            .zip([-40.0, -30.0, 0.0])
            .chain(scaled_max.into_iter().zip([54.0, 30.0, 20.0]))
        {
            assert!((actual - expected).abs() < 1.0e-9, "{actual} != {expected}");
        }
    }

    #[test]
    fn suppressed_node_between_sketch_and_extrude_still_extrudes() {
        let sketch = rounded_profile_sketch();
        let sketch_value = serde_json::json!({"sketch": serde_json::to_value(&sketch).unwrap()});
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": sketch_value, "cache": {"key": "sketch"}},
                {"id": "move", "type": "translate", "suppressed": true, "values": {"x": 0.0, "z": 0.0}, "cache": {"key": "move"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "sketch", "to": "move", "in": "shape", "out": "sketch"},
                {"from": "move", "to": "extrude", "in": "profile", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("suppressed extrude part");
        assert_eq!(part.bodies.len(), 1);
        assert!(exact_part_volume(part) > 0.0);
        assert_exact_bounds(part, [-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]);
    }

    #[test]
    fn user_sketch_translate_extrude_fillet_graph_evaluates() {
        let sketch = rounded_profile_sketch();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["fillet"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "translate", "type": "translate", "values": {"x": 0.0, "y": 0.0, "z": 0.0}, "cache": {"key": "translate"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}},
                {"id": "fillet", "type": "fillet", "values": {"edges": [{"id": "edge:11"}]}, "cache": {"key": "fillet"}}
            ],
            "connections": [
                {"from": "sketch", "to": "translate", "in": "shape", "out": "sketch"},
                {"from": "translate", "to": "extrude", "in": "profile", "out": "shape"},
                {"from": "extrude", "to": "fillet", "in": "shape", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let extruded = output.exact_parts.get("extrude").expect("extrude part");
        let filleted = output.exact_parts.get("fillet").expect("fillet part");
        assert_eq!(filleted.bodies.len(), 1);
        let (extruded_volume, filleted_volume) =
            (exact_part_volume(extruded), exact_part_volume(filleted));
        assert!(extruded_volume > 0.0 && filleted_volume > 0.0);
        assert!(
            filleted_volume <= extruded_volume + 1.0e-9,
            "fillet {filleted_volume} > extrude {extruded_volume}"
        );
    }

    #[test]
    fn display_extrude_after_translate_uses_the_part_sketch_path() {
        let sketch = rounded_profile_sketch();
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": {"sketch": serde_json::to_value(&sketch).unwrap()}, "cache": {"key": "sketch"}},
                {"id": "translate", "type": "translate", "values": {"x": 15.0, "z": 7.0, "rz": 90.0}, "cache": {"key": "translate"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "sketch", "to": "translate", "in": "shape", "out": "sketch"},
                {"from": "translate", "to": "extrude", "in": "profile", "out": "shape"}
            ]
        })).expect("request");
        // Empty mesh targets force the legacy display evaluator (instead of
        // the Part target pre-pass) to be the path under test.
        let output = evaluate(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("display-path extrude part");
        assert_eq!(part.bodies.len(), 1);
        assert!(exact_part_volume(part) > 0.0);
        assert_exact_bounds(part, [0.0, -20.0, 7.0], [30.0, 20.0, 17.0]);
    }

    #[test]
    fn combine_sketches_extrude_as_separate_bodies_and_mixed_input_errors() {
        let sketch = rounded_profile_sketch();
        let sketch_value = serde_json::json!({"sketch": serde_json::to_value(&sketch).unwrap()});
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "sketch", "values": sketch_value, "cache": {"key": "a"}},
                {"id": "b", "type": "sketch", "values": sketch_value, "cache": {"key": "b"}},
                {"id": "combine", "type": "combine", "values": {}, "cache": {"key": "combine"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "a", "to": "combine", "in": "shape", "out": "sketch"},
                {"from": "b", "to": "combine", "in": "shape", "out": "sketch"},
                {"from": "combine", "to": "extrude", "in": "profile", "out": "shape"}
            ]
        })).expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("extrude").expect("combined extrude part");
        assert_eq!(part.bodies.len(), 2, "each sketch in a Combine extrudes to its own body");
        assert!(exact_part_volume(part) > 0.0);

        let mixed: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["extrude"], "meshTargets": [],
            "nodes": [
                {"id": "sketch", "type": "sketch", "values": sketch_value, "cache": {"key": "sketch"}},
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}, "cache": {"key": "box"}},
                {"id": "combine", "type": "combine", "values": {}, "cache": {"key": "combine"}},
                {"id": "extrude", "type": "extrude", "values": {"amount": 10.0, "operation": "new"}, "cache": {"key": "extrude"}}
            ],
            "connections": [
                {"from": "sketch", "to": "combine", "in": "shape", "out": "sketch"},
                {"from": "box", "to": "combine", "in": "shape", "out": "shape"},
                {"from": "combine", "to": "extrude", "in": "profile", "out": "shape"}
            ]
        })).expect("request");
        let mixed_output = evaluate_exact(mixed, EngineCache::new(), None);
        let error = mixed_output.response.node_errors.get("extrude").expect("mixed input error");
        assert!(error.contains("both bodies and sketches"), "{error}");
    }

    #[test]
    fn sketch_target_does_not_emit_a_3d_display_mesh() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        if let Some(mesh) = output.meshes.get("sketch") {
            assert!(mesh.mesh.faces.is_empty(), "sketch must not display as a 3D mesh");
            assert!(mesh.mesh.vertices.is_empty(), "sketch must not display as a 3D mesh");
        }
    }

    #[test]
    fn sketch_output_is_typed_cached_and_transformed() {
        let mut sketch = crate::sketch::Sketch::new();
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
        let output = eval.part_sketch_output("top-at").expect("typed sketch output");
        assert_eq!(output.profiles.len(), 1);
        assert!((output.offset[2] - 7.19).abs() < 1.0e-9);
    }

    #[test]
    fn loft_profile_sampling_keeps_arc_curvature() {
        let mut sketch = crate::sketch::Sketch::new();
        sketch.add_circle(solidcore_record::Vec2::new(0.0, 0.0), 10.0, None, None).unwrap();
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
        let a = sketch.add_point(solidcore_record::Vec2::new(-10.0, -10.0), None).unwrap();
        let b = sketch.add_point(solidcore_record::Vec2::new(10.0, -10.0), None).unwrap();
        let c = sketch.add_point(solidcore_record::Vec2::new(10.0, 10.0), None).unwrap();
        let d = sketch.add_point(solidcore_record::Vec2::new(-10.0, 10.0), None).unwrap();
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
            nodecraft_mesh::import_step_breps(path.to_str().unwrap()).is_ok(),
            "first export did not round-trip"
        );
        let _ = std::fs::remove_file(&path);
        let mut second_request = request.clone();
        for node in &mut second_request.nodes {
            if node.id == "exporter" {
                node.cache = Some(nodecraft_protocol::CacheState {
                    key: Some("exporter-v2".into()),
                    state: Some("cached".into()),
                });
            }
        }
        let second = evaluate(second_request, first.cache, None);
        assert!(second.response.node_errors.is_empty(), "{:?}", second.response.node_errors);
        assert!(
            path.is_file(),
            "cached loft export was not written: {:?}",
            second.response.node_errors
        );
        assert!(
            nodecraft_mesh::import_step_breps(path.to_str().unwrap()).is_ok(),
            "cached export did not round-trip"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// A Boolean refused because its revision was cancelled passes its first operand through, so
    /// anything built on it in that run is not the graph's result. The next, live evaluation of the
    /// same graph must see the cut through the translate after it, not a Part the cancelled run
    /// cached from the pass-through (a mug handle drawn with the half of the torus a box cut away).
    #[test]
    fn a_cancelled_run_caches_nothing_built_on_a_refused_node() {
        let request = |targets: &[&str]| -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": targets,
                "meshTargets": [],
                "nodes": [
                    {"id": "block", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "block", "state": "cached"}},
                    {"id": "pin", "type": "primitive", "values": {"primitiveType": "box", "length": 4, "width": 4, "height": 20}, "cache": {"key": "pin", "state": "cached"}},
                    {"id": "cut", "type": "boolean", "values": {"mode": "subtract"}, "cache": {"key": "cut", "state": "cached"}},
                    {"id": "moved", "type": "translate", "values": {"x": 30}, "cache": {"key": "moved", "state": "cached"}}
                ],
                "connections": [
                    {"from": "block", "to": "cut", "in": "a", "out": "shape"},
                    {"from": "pin", "to": "cut", "in": "b", "out": "shape"},
                    {"from": "cut", "to": "moved", "in": "shape", "out": "shape"}
                ]
            }))
            .expect("request")
        };
        let faces = |output: &EngineOutput, id: &str| -> usize {
            let part = output.exact_parts.get(id).unwrap_or_else(|| panic!("{id} has no Part"));
            part.bodies.iter().map(|body| body.brep.face_list().len()).sum()
        };
        // the operands, evaluated live
        crate::kernel_pool::set_current_revision_token(crate::kernel_pool::RevisionToken::new());
        let warm = evaluate_exact(request(&["block", "pin"]), EngineCache::new(), None);
        // the whole graph under a cancelled revision: the Boolean is refused and passes `block` on
        let cancelled_token = crate::kernel_pool::RevisionToken::new();
        cancelled_token.cancel();
        crate::kernel_pool::set_current_revision_token(cancelled_token);
        let cancelled = evaluate_exact(request(&["moved"]), warm.cache, None);
        // the same graph, live
        crate::kernel_pool::set_current_revision_token(crate::kernel_pool::RevisionToken::new());
        let live = evaluate_exact(request(&["moved", "cut"]), cancelled.cache, None);
        assert!(live.response.node_errors.is_empty(), "{:?}", live.response.node_errors);
        assert!(faces(&live, "cut") > 6, "the pin must cut the block");
        assert_eq!(
            faces(&live, "moved"),
            faces(&live, "cut"),
            "the translate after the Boolean came from the cancelled run's pass-through"
        );
    }

    /// Fillet and Chamfer hand on their result through one helper (`nodes::edge_blend_part`):
    /// a round made before a chamfer reaches the node after it with its exact blend definition,
    /// and the chamfer tags the face it made for the selected-feature overlay.
    #[test]
    fn a_round_stays_exact_through_a_chamfer() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["bevel"],
            "meshTargets": [],
            "nodes": [
                {"id": "cyl", "type": "solid", "values": {"primitiveType": "cylinder"}, "cache": {"key": "cyl", "state": "cached"}},
                {"id": "round", "type": "fillet", "values": {"length": 10.3, "edges": [{"center": [-20.0, 0.0, -20.0]}]}, "cache": {"key": "round", "state": "cached"}},
                {"id": "bevel", "type": "chamfer", "values": {"length": 2.0, "edges": [{"center": [-20.0, 0.0, 20.0]}]}, "cache": {"key": "bevel", "state": "cached"}}
            ],
            "connections": [
                {"from": "cyl", "to": "round", "in": "shape", "out": "shape"},
                {"from": "round", "to": "bevel", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("bevel").expect("chamfer Part");
        assert_eq!(part.bodies.len(), 1);
        let brep = &part.bodies[0].brep;
        let marks = nodecraft_solidcore::solidcore::blend::definition_check::blend_face_marks(brep)
            .expect("marks");
        assert_eq!(marks.len(), 2, "the round's face and the chamfer's face are tagged");
        let mut kernel = nodecraft_kernel_profile::kernel();
        let copy = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(&mut kernel, brep)
            .expect("copy");
        let (record, _) = nodecraft_ops::SolidBody::new(copy)
            .ops_copy(&mut kernel)
            .expect("operand")
            .to_record()
            .expect("record");
        let exact = record
            .faces
            .iter()
            .filter(|face| matches!(face.surface, solidcore_record::brep::Surface::Blend { .. }))
            .count();
        assert_eq!(exact, 1, "the round reaches the chamfer's result with its blend definition");
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

    #[test]
    fn scatter_instances_share_one_display_body_key() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives as api;

        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep =
            api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).expect("box");
        let body = BodyInstance {
            body: nodecraft_ops::SolidBody::from_arc(Arc::new(brep)),
            placement: nodecraft_exact::Placement::identity(),
            realized: Default::default(),
            source: Some(41),
            source_body: 3,
            name: None,
            color: None,
        };
        let part = Part { bodies: vec![body.clone(), body], sketches: Vec::new() };
        let exact = exact_part_from(&part, 999);
        assert_eq!(exact.bodies[0].key, exact.bodies[1].key);
        assert_eq!(exact.bodies[0].key, nodecraft_exact::BodyKey::new(41, 3));
        let translated = part.translated([10.0, 0.0, 0.0], [0.0; 3]);
        let translated = exact_part_from(&translated, 999);
        assert_eq!(translated.bodies[0].key, exact.bodies[0].key);
    }

    #[test]
    fn scatter_instances_share_exact_body_keys_end_to_end() {
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["forest"],
            "meshTargets": ["forest"],
            "binaryMesh": false,
            "nodes": [
                {"id": "trunk", "type": "primitive", "values": {"primitiveType": "cylinder", "radius": 1.0, "height": 8.0}},
                {"id": "canopy", "type": "primitive", "values": {"primitiveType": "sphere", "radius": 4.0}},
                {"id": "tree", "type": "combine", "values": {}},
                {"id": "forest", "type": "scatter", "values": {"count": 10.0, "spanX": 20.0, "spanY": 20.0, "seed": 1.0, "minScale": 0.8, "maxScale": 1.2, "distribution": "grid"}}
            ],
            "connections": [
                {"from": "trunk", "to": "tree", "in": "shape", "out": "shape"},
                {"from": "canopy", "to": "tree", "in": "shape", "out": "shape"},
                {"from": "tree", "to": "forest", "in": "shape", "out": "shape"}
            ]
        })).expect("scatter request");
        let output = evaluate_exact(request, EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let forest = output.exact_parts.get("forest").expect("forest exact part");
        assert_eq!(forest.bodies.len(), 20);
        let keys: std::collections::HashSet<_> =
            forest.bodies.iter().map(|body| body.key).collect();
        assert_eq!(keys.len(), 2, "scatter bodies must share their source plans");
    }

    /// A Union run that only feeds the next Union becomes one `union_all`; requesting an
    /// intermediate keeps it computable. The fused final body matches the node-by-node chain
    /// in bodies, volume and faces.
    #[test]
    fn a_union_chain_fuses_to_one_union_all_with_the_same_geometry() {
        fn request(targets: Vec<&str>) -> Request {
            serde_json::from_value(serde_json::json!({
                "requestId": 1,
                "targets": targets,
                "meshTargets": targets,
                "binaryMesh": false,
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "a", "state": "cached"}},
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "b", "state": "cached"}},
                    {"id": "c", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "c", "state": "cached"}},
                    {"id": "d", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0, "width": 20.0, "height": 20.0}, "cache": {"key": "d", "state": "cached"}},
                    {"id": "t", "type": "translate", "values": {"x": 10.0}, "cache": {"key": "t", "state": "cached"}},
                    {"id": "u1", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u1", "state": "cached"}},
                    {"id": "u2", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u2", "state": "cached"}},
                    {"id": "u3", "type": "boolean", "values": {"mode": "union"}, "cache": {"key": "u3", "state": "cached"}}
                ],
                "connections": [
                    {"from": "a", "to": "u1", "in": "a", "out": "shape"},
                    {"from": "b", "to": "u1", "in": "b", "out": "shape"},
                    {"from": "u1", "to": "u2", "in": "a", "out": "shape"},
                    {"from": "c", "to": "u2", "in": "b", "out": "shape"},
                    {"from": "u2", "to": "u3", "in": "a", "out": "shape"},
                    {"from": "t", "to": "u3", "in": "b", "out": "shape"},
                    {"from": "d", "to": "t", "in": "shape", "out": "shape"}
                ]
            })).expect("request")
        }
        let fused = evaluate_exact(request(vec!["u3"]), EngineCache::new(), None);
        assert!(fused.response.node_errors.is_empty(), "{:?}", fused.response.node_errors);
        let chain = evaluate_exact(request(vec!["u1", "u2", "u3"]), EngineCache::new(), None);
        assert!(chain.response.node_errors.is_empty(), "{:?}", chain.response.node_errors);
        let fused_part = fused.exact_parts.get("u3").expect("fused target");
        let chain_part = chain.exact_parts.get("u3").expect("chain target");
        assert_eq!(fused_part.bodies.len(), chain_part.bodies.len(), "body count");
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let mut volume = |part: &nodecraft_exact::ExactPart| -> f64 {
            part.bodies
                .iter()
                .map(|body| {
                    nodecraft_layers::kernel_volume(&mut kernel, &body.brep).expect("volume")
                })
                .sum()
        };
        let (fused_volume, chain_volume) = (volume(fused_part), volume(chain_part));
        assert!(
            (fused_volume - chain_volume).abs() <= fused_volume.abs().max(1.0) * 1.0e-6,
            "volume {fused_volume} vs {chain_volume}"
        );
        let faces = |part: &nodecraft_exact::ExactPart| -> usize {
            part.bodies.iter().map(|body| body.brep.face_list().len()).sum()
        };
        assert_eq!(faces(fused_part), faces(chain_part), "face count");
        // The intermediate is gone from the fused pass but still computable on request.
        assert!(fused.exact_parts.get("u1").is_none());
        assert!(chain.exact_parts.get("u1").is_some());
    }

    /// A translated body is realised once per instance and every reader gets the same
    /// allocation; the rigid map keeps the source's proof state. A mirror resets it.
    #[test]
    fn a_placed_body_is_realised_once_and_a_rigid_map_keeps_its_proof() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box");
        let part = Part::from_brep(brep).tagged(1).translated([10.0, 0.0, 0.0], [0.0; 3]);
        part.bodies[0].body.prove(&mut kernel).expect("prove the source");
        let first = part.realized_bodies().expect("realize");
        let second = part.realized_bodies().expect("realize again");
        assert!(
            Arc::ptr_eq(&first[0].shared(), &second[0].shared()),
            "the realised body is memoised per instance"
        );
        assert!(first[0].is_proved(), "a translate/rotate carries the proof state");
        assert_eq!(first[0].proof_attempts(), 0, "carrying a proof does not prove again");

        let mirrored =
            part.mirrored(&serde_json::json!({"plane": "yz", "offset": 0.0, "combine": false}));
        let mirrored = mirrored.realized_bodies().expect("mirror realize");
        assert_eq!(
            mirrored[0].proof_state(),
            nodecraft_ops::ProofState::Unknown,
            "a mirror is a different B-rep and resets the proof"
        );
    }

    /// Exact export proves the realised bodies before writing: a valid body is proved by the
    /// gate, an inside-out body is refused with the failed test.
    #[test]
    fn exact_export_proves_its_bodies_and_refuses_an_inside_out_one() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        use nodecraft_solidcore::solidcore::topology::face::p1::flip_face_orientation;
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let box_brep = create_box(&mut kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
            .expect("box");
        let valid = Part::from_brep(box_brep.clone_standing_for_self().expect("copy"));
        prove_part_for_export(&valid, &mut kernel).expect("a valid body is exportable");
        assert!(valid.bodies[0].body.is_proved(), "the export gate proved the body");

        let mut reversed = box_brep.clone_standing_for_self().expect("copy");
        for face in reversed.face_list().to_vec() {
            flip_face_orientation(&mut reversed, face).expect("flip");
        }
        let invalid = Part::from_brep(reversed);
        let error = prove_part_for_export(&invalid, &mut kernel)
            .expect_err("an inside-out body must not export");
        assert!(error.contains("closed solid"), "{error}");
        assert!(invalid.bodies[0].body.refused().is_some());
    }

    /// Build a box -> translate chain of `n` nodes for placement-work checks.
    fn translate_chain_request(n: usize) -> Request {
        let mut nodes = Vec::with_capacity(n);
        let mut connections = Vec::with_capacity(n.saturating_sub(1));
        nodes.push(serde_json::json!({
            "id": "box", "type": "primitive",
            "values": {"primitiveType": "box", "length": 2, "width": 2, "height": 2},
            "cache": {"key": "box"}
        }));
        for index in 1..n {
            nodes.push(serde_json::json!({
                "id": format!("t{index}"), "type": "translate",
                "values": {"x": 1.0}, "cache": {"key": format!("t{index}")}
            }));
            connections.push(serde_json::json!({
                "from": if index == 1 { "box".to_owned() } else { format!("t{}", index - 1) },
                "to": format!("t{index}"), "in": "shape", "out": "shape"
            }));
        }
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": [format!("t{}", n - 1)],
            "meshTargets": [],
            "nodes": nodes,
            "connections": connections
        }))
        .expect("translate chain request")
    }

    fn chain_placement_work(
        n: usize,
    ) -> (nodecraft_exact::placement_stats::PlacementCounters, usize) {
        nodecraft_exact::placement_stats::reset();
        let output = evaluate_exact(translate_chain_request(n), EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let target = format!("t{}", n - 1);
        let bodies = output.exact_parts.get(&target).expect("target part").bodies.len();
        (nodecraft_exact::placement_stats::counters(), bodies)
    }

    /// The measured regression: a 5,000-body placement chain must copy O(N)
    /// transform elements. The old per-body `Vec<BodyTransform>` history copied
    /// 1..N elements per node, so the same measurement grew quadratically.
    #[test]
    fn a_5000_node_placement_chain_copies_transform_data_linearly() {
        let (small, small_bodies) = chain_placement_work(1000);
        let (large, large_bodies) = chain_placement_work(5000);
        assert_eq!((small_bodies, large_bodies), (1, 1));
        eprintln!(
            "placement work: n=1000 copied={} composed={}; n=5000 copied={} composed={}",
            small.matrix_elements_copied,
            small.compositions,
            large.matrix_elements_copied,
            large.compositions,
        );

        // Each chain node composes one 4x4 matrix and each body placement is
        // shared; 80 elements per node leaves room for the per-part clone while
        // staying far below the old O(N^2) copy.
        assert!(
            large.matrix_elements_copied <= 80 * 5000,
            "placement copied {} elements for 5000 nodes",
            large.matrix_elements_copied,
        );
        assert!(
            large.compositions <= 2 * 5000,
            "composed {} placement matrices for 5000 nodes",
            large.compositions,
        );
        // Linear work: 5x the nodes is about 5x the copies, not 25x.
        assert!(
            large.matrix_elements_copied <= small.matrix_elements_copied * 8 + 2048,
            "placement work grew faster than linearly: {} at 1000, {} at 5000",
            small.matrix_elements_copied,
            large.matrix_elements_copied,
        );
    }

    /// The old first-to-last chain product, rebuilt from the operations for
    /// equivalence checks. It must agree with the cached placement matrix.
    fn old_chain_matrix(placement: &nodecraft_exact::Placement) -> nodecraft_exact::Affine3 {
        let mut matrix = nodecraft_exact::Affine3::IDENTITY;
        for operation in placement.ops() {
            matrix = operation.matrix().compose(matrix);
        }
        matrix
    }

    fn placement_brep_bounds(
        brep: &nodecraft_solidcore::solidcore::topology::Brep,
    ) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in brep.vertex_list() {
            let point = brep.vertex(*vertex).expect("vertex").point;
            for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                min[axis] = min[axis].min(value);
                max[axis] = max[axis].max(value);
            }
        }
        (min, max)
    }

    fn assert_close(left: f64, right: f64, context: &str) {
        assert!((left - right).abs() <= 1.0e-9, "{context}: {left} vs {right}",);
    }

    /// For translate/rotate/scale chains, patterns, mirror, combine and
    /// boolean graphs: every body's composed matrix equals the old chain
    /// product within 1e-9, and the realised B-rep bounds equal the local
    /// bounds placed by that same value.
    #[test]
    fn placement_value_matches_the_old_chain_and_realised_bounds() {
        let requests: Vec<Request> = vec![
            serde_json::from_value(serde_json::json!({
                "requestId": 1, "targets": ["scatter"], "meshTargets": [],
                "nodes": [
                    {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 3, "width": 4, "height": 5}},
                    {"id": "t1", "type": "translate", "values": {"x": 7, "y": -2, "z": 3, "rx": 20, "ry": -15, "rz": 40}},
                    {"id": "t2", "type": "translate", "values": {"x": -1, "y": 5, "z": 2, "rz": 25}},
                    {"id": "scatter", "type": "scatter", "values": {"count": 4, "distribution": "grid", "minScale": 1.5, "maxScale": 1.5}}
                ],
                "connections": [
                    {"from": "box", "to": "t1", "in": "shape", "out": "shape"},
                    {"from": "t1", "to": "t2", "in": "shape", "out": "shape"},
                    {"from": "t2", "to": "scatter", "in": "shape", "out": "shape"}
                ]
            })).expect("scale/rotate chain"),
            serde_json::from_value(serde_json::json!({
                "requestId": 2, "targets": ["pattern"], "meshTargets": [],
                "nodes": [
                    {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 4, "width": 3, "height": 2}},
                    {"id": "move", "type": "translate", "values": {"x": 9, "rz": 30}},
                    {"id": "pattern", "type": "circularpattern", "values": {"count": 5, "axis": "z"}}
                ],
                "connections": [
                    {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                    {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
                ]
            })).expect("pattern"),
            serde_json::from_value(serde_json::json!({
                "requestId": 3, "targets": ["combine"], "meshTargets": [],
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 2, "width": 2, "height": 2}},
                    {"id": "ta", "type": "translate", "values": {"x": -4, "rz": 15}},
                    {"id": "m", "type": "mirror", "values": {"plane": "yz", "offset": 3, "combine": true}},
                    {"id": "combine", "type": "combine", "values": {}}
                ],
                "connections": [
                    {"from": "a", "to": "ta", "in": "shape", "out": "shape"},
                    {"from": "ta", "to": "m", "in": "shape", "out": "shape"},
                    {"from": "m", "to": "combine", "in": "shape", "out": "shape"}
                ]
            })).expect("mirror combine"),
            serde_json::from_value(serde_json::json!({
                "requestId": 4, "targets": ["cut"], "meshTargets": [],
                "nodes": [
                    {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                    {"id": "ta", "type": "translate", "values": {"x": 1, "ry": 20}},
                    {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 4, "width": 4, "height": 20}},
                    {"id": "tb", "type": "translate", "values": {"x": 4, "rz": 10}},
                    {"id": "cut", "type": "boolean", "values": {"operation": "difference"}}
                ],
                "connections": [
                    {"from": "a", "to": "ta", "in": "shape", "out": "shape"},
                    {"from": "b", "to": "tb", "in": "shape", "out": "shape"},
                    {"from": "ta", "to": "cut", "in": "a", "out": "shape"},
                    {"from": "tb", "to": "cut", "in": "b", "out": "shape"}
                ]
            })).expect("boolean cut"),
        ];

        for request in requests {
            let output = evaluate_exact(request, EngineCache::new(), None);
            assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
            for (target, part) in &output.exact_parts {
                let realised = part.realized_breps().expect("realise part");
                assert_eq!(realised.len(), part.bodies.len());
                for (body, realised) in part.bodies.iter().zip(&realised) {
                    let expected = old_chain_matrix(&body.placement);
                    let actual = *body.placement.matrix();
                    for row in 0..4 {
                        for column in 0..4 {
                            assert_close(
                                actual.row(row)[column],
                                expected.row(row)[column],
                                &format!("{target} body matrix [{row}][{column}]"),
                            );
                        }
                    }
                    // The realised B-rep bounds must be the local bounds placed
                    // by the same composed value.
                    let (min, max) = placement_brep_bounds(body.brep.as_ref());
                    let mut expected_min = [f64::INFINITY; 3];
                    let mut expected_max = [f64::NEG_INFINITY; 3];
                    for corner in 0..8 {
                        let point = [
                            if corner & 1 == 0 { min[0] } else { max[0] },
                            if corner & 2 == 0 { min[1] } else { max[1] },
                            if corner & 4 == 0 { min[2] } else { max[2] },
                        ];
                        let placed = actual.apply_point(point);
                        for axis in 0..3 {
                            expected_min[axis] = expected_min[axis].min(placed[axis]);
                            expected_max[axis] = expected_max[axis].max(placed[axis]);
                        }
                    }
                    let (realised_min, realised_max) = placement_brep_bounds(realised);
                    for axis in 0..3 {
                        assert!(
                            (realised_min[axis] - expected_min[axis]).abs() <= 1.0e-6
                                && (realised_max[axis] - expected_max[axis]).abs() <= 1.0e-6,
                            "{target} body {axis}: realised {realised_min:?}..{realised_max:?}, expected {expected_min:?}..{expected_max:?}",
                        );
                    }
                }
            }
        }
    }
}
