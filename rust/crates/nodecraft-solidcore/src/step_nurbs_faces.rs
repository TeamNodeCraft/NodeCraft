//! STEP B-spline face (sheet) → solidcore `Brep` with UV-rectangular trims.
//!
//! This module is the topology half of the B-spline STEP import: `step_nurbs` parses a
//! STEP surface, [`crate::step_nurbs`] maps it to a solidcore `SmSurface`, and here the
//! `ADVANCED_FACE` that uses it becomes a real one-face B-rep:
//!
//! * one `ADVANCED_FACE` on a `B_SPLINE_SURFACE_WITH_KNOTS` (or its rational complex
//!   instance form),
//! * one outer bounding loop with four edges,
//! * the loop's corners map onto an axis-aligned rectangle in the surface's `(u, v)`
//!   parameter space (a "UV-rectangular" trim),
//! * the four boundary edges are straight in UV (line-like STEP curves) or are trusted
//!   iso-curves (higher-degree curves are not checked point-by-point).
//!
//! The B-rep is built with the public solidcore topology-records path
//! (`BrepData` + `make_topology_from_data`), the same one
//! [`crate::mesh_brep::brep_from_planar_body_exact`] uses for exact planar faces:
//!
//! * `vertices`: the four UV corners evaluated on the surface,
//! * `curves_3d`: the four exact iso-parametric boundary curves
//!   (`BSplineSurface::create_iso_parametric_curve`),
//! * `uv_curves`: the four 2D trim lines in `(u, v)`,
//! * `edges` / `edgeuses` / `loop` / `face` with `uv_domain` and `rectangular_trim = 1`,
//! * one open shell / region.
//!
//! Anything else is reported as an explicit `unsupported` error instead of being guessed:
//! faces without a B-spline surface, faces with holes, loops that are not a single
//! four-edge cycle, corner points that do not map onto an axis-aligned UV rectangle, or
//! straight STEP edges that leave the rectangle between their end points.
//!
//! Limitations (documented, not silently accepted):
//!
//! * Only single-sheet faces: no shells/solids, no holes, no multiple faces.
//! * The face is always built with its natural orientation (`faceuse.orientation = 1`);
//!   the STEP `same_sense` flag is not carried onto a sheet because a sheet has no volume.
//! * Curved STEP boundary curves are not verified against the surface; their exact
//!   iso-curves are regenerated from the surface rather than read from the STEP edges.
//! * Non-rectangular trims (trimmed patches, holes, seams) are unsupported.


use solidcore::curve::curve::Curve;
use solidcore::error::EntityRef;
use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::geom::extent2d::Extent2d;
use solidcore::geom::object::ObjectRef;
use solidcore::math::Point3;
use solidcore::nurbs::b_spline_curve::BSplineCurve;
use solidcore::nurbs::b_spline_surface::{IsoCurveOptions, SurfParamType};
use solidcore::surface::surface::SurfaceKind;
use solidcore::topology::brep::p10::{
    HEALER_VERSION, MakeTopologyOptions, TopologyEnv, make_topology_from_data,
};
use solidcore::topology::brep_data::{
    BrepData, EdgeData, EuData, FaceData, FaceuseData, Keyed, LoopData, RegionData, ShellData,
    VertexData,
};
use solidcore::topology::Brep;
use nodecraft_step::StepFile;
use nodecraft_step::topology::{StepBrep, StepFace, read_topology};

/// Relative tolerance used for on-surface checks (`1e-6 * model size`).
const SURFACE_TOLERANCE: f64 = 1e-6;

fn key_of(entity: EntityRef) -> ObjectRef {
    let (type_code, index) = match entity {
        EntityRef::Face(id) => (16002, u64::from(id.index())),
        EntityRef::Edge(id) => (16016, u64::from(id.index())),
        EntityRef::Edgeuse(id) => (16005, u64::from(id.index())),
        EntityRef::Vertex(id) => (16021, u64::from(id.index())),
        EntityRef::Vertexuse(id) => (16006, u64::from(id.index())),
        EntityRef::Loop(id) => (16003, u64::from(id.index())),
        EntityRef::Loopuse(id) => (16004, u64::from(id.index())),
        EntityRef::Faceuse(id) => (16022, u64::from(id.index())),
        EntityRef::Shell(id) => (16013, u64::from(id.index())),
        EntityRef::Region(id) => (16012, u64::from(id.index())),
        EntityRef::Brep(_) => (16011, 0),
    };
    ObjectRef { type_code, key: ObjectKey(1_000_000 + type_code as u64 * 100_000 + index) }
}

/// The `(u, v)` rectangle trimmed out of a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvRectangle {
    /// Lower-left corner in parameter space.
    pub min: [f64; 2],
    /// Upper-right corner in parameter space.
    pub max: [f64; 2],
}

impl UvRectangle {
    fn new(min: [f64; 2], max: [f64; 2]) -> Result<Self, String> {
        if !(min[0].is_finite() && min[1].is_finite() && max[0].is_finite() && max[1].is_finite())
        {
            return Err("UV rectangle bounds must be finite".into());
        }
        if !(min[0] < max[0] && min[1] < max[1]) {
            return Err(format!(
                "UV rectangle is empty: min {min:?} is not below max {max:?}"
            ));
        }
        Ok(Self { min, max })
    }

    fn corners(&self) -> [[f64; 2]; 4] {
        [
            [self.min[0], self.min[1]],
            [self.max[0], self.min[1]],
            [self.max[0], self.max[1]],
            [self.min[0], self.max[1]],
        ]
    }

    fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }
}

/// Builds the solidcore B-rep for the STEP `ADVANCED_FACE` `face_id`, whose surface must
/// be a B-spline surface and whose outer loop must be a four-edge UV rectangle.
///
/// # Errors
///
/// Returns a descriptive error when the entity is not an `ADVANCED_FACE`, the surface is
/// not a B-spline surface (or cannot be parsed/converted), the face has more than one
/// bound (holes), the outer loop is not a four-edge cycle, a loop corner does not map
/// back onto the surface (within tolerance), the corners do not form an axis-aligned UV
/// rectangle, or a straight STEP edge leaves that rectangle. Messages for unsupported
/// topology start with `unsupported:`.
pub fn brep_from_bspline_sheet(file: &StepFile, face_id: u32) -> Result<Brep, String> {
    let topology = read_topology(file);
    let face = topology
        .faces
        .get(&face_id)
        .ok_or_else(|| format!("STEP entity #{face_id} is not an ADVANCED_FACE"))?;
    let surface_entity = file.entities.get(&face.surface).ok_or_else(|| {
        format!(
            "face #{face_id} references missing surface entity #{}",
            face.surface
        )
    })?;
    if !is_bspline_surface_entity(surface_entity) {
        return Err(format!(
            "unsupported: face #{face_id} surface #{} is {}; only B-spline surfaces are supported",
            face.surface,
            entity_label(surface_entity)
        ));
    }
    let surface = ::step_nurbs::parse_surface(file, face.surface)
        .map_err(|error| format!("face #{face_id} surface #{}: {error}", face.surface))?;
    let rectangle = uv_rectangle_from_face_bounds(file, &topology, face, &surface)?;
    brep_from_bspline_uv_rectangle(&surface, rectangle.min, rectangle.max)
        .map_err(|error| format!("face #{face_id}: {error}"))
}

/// Reads `path` and builds the B-rep of its single B-spline sheet face.
///
/// # Errors
///
/// Those of [`brep_from_bspline_sheet`], plus a missing file, unparsable STEP data, no
/// B-spline face in the file, or more than one B-spline face (ambiguous).
pub fn brep_from_bspline_sheet_file(path: &str) -> Result<Brep, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: {error}"))?;
    brep_from_bspline_sheet_file_parsed(&file)
}

fn brep_from_bspline_sheet_file_parsed(file: &StepFile) -> Result<Brep, String> {
    let topology = read_topology(file);
    let mut bspline_faces: Vec<u32> = topology
        .faces
        .values()
        .filter(|face| {
            file.entities
                .get(&face.surface)
                .is_some_and(is_bspline_surface_entity)
        })
        .map(|face| face.id)
        .collect();
    bspline_faces.sort_unstable();
    match bspline_faces.as_slice() {
        [] => Err("unsupported: the STEP file has no B-spline surface face".into()),
        [face_id] => brep_from_bspline_sheet(file, *face_id),
        _ => Err(format!(
            "unsupported: the STEP file has {} B-spline faces; only a single sheet is supported",
            bspline_faces.len()
        )),
    }
}

/// Builds a one-face B-rep over `surface`, trimmed to the UV rectangle
/// `uv_min`..`uv_max` (which must lie inside the surface's natural domain).
///
/// The boundary is built from exact iso-parametric curves of the surface, so the result
/// is exact for any B-spline surface, not just planar patches.
///
/// # Errors
///
/// Empty or non-finite rectangles, rectangles outside the surface's natural parameter
/// domain, a surface that cannot be converted (see [`crate::step_nurbs`]), or a failure
/// while extracting the iso-parametric boundary curves.
pub fn brep_from_bspline_uv_rectangle(
    surface: &::step_nurbs::BsplineSurface,
    uv_min: [f64; 2],
    uv_max: [f64; 2],
) -> Result<Brep, String> {
    let rectangle = UvRectangle::new(uv_min, uv_max)?;
    let converted = crate::step_nurbs::bspline_surface_to_solidcore(surface)?;
    let SurfaceKind::BSpline(kernel_surface) = converted.kind() else {
        return Err("unsupported: converted surface did not come back as a B-spline surface".into());
    };
    let (u0, u1) = surface.u_domain();
    let (v0, v1) = surface.v_domain();
    let span = (u1 - u0).abs().max((v1 - v0).abs()).max(1.0);
    let parameter_tolerance = SURFACE_TOLERANCE * span;
    if rectangle.min[0] < u0 - parameter_tolerance
        || rectangle.min[1] < v0 - parameter_tolerance
        || rectangle.max[0] > u1 + parameter_tolerance
        || rectangle.max[1] > v1 + parameter_tolerance
    {
        return Err(format!(
            "UV rectangle {uv_min:?}..{uv_max:?} lies outside the surface domain {u0}..{u1} x {v0}..{v1}"
        ));
    }

    let uv_domain = Extent2d::from_bounds(
        rectangle.min[0],
        rectangle.min[1],
        rectangle.max[0],
        rectangle.max[1],
    );
    let corners_uv = rectangle.corners();
    let mut corners_3d = Vec::with_capacity(4);
    for corner in &corners_uv {
        corners_3d.push(
            surface
                .point_at(corner[0], corner[1])
                .map_err(|error| format!("surface corner {corner:?}: {error}"))?,
        );
    }

    let kernel = crate::kernel::kernel();
    let mut data = BrepData::new(1, solidcore::topology::brep_data::DataSourceType::Unknown);
    data.healer_version = HEALER_VERSION;
    data.regions.push(RegionData { start_shell: 0, num_shells: 1, is_void: 0, ..RegionData::zeroed() });
    data.shells.push(ShellData { shell_type: 0, faceuse_start: 0, num_faceuses: 1, ..ShellData::zeroed() });
    data.faceuses.push(FaceuseData { face: 0, orientation: 1, ..FaceuseData::zeroed() });
    data.faces.push(FaceData {
        surface: 0,
        start_loop: 0,
        num_loops: 1,
        uv_domain,
        rectangular_trim: 1,
        zone_tol_3d: 1e-6,
        ..FaceData::zeroed()
    });
    data.loops.push(LoopData { loop_type: 0, start_eu: 0, num_eu: 4, ..LoopData::zeroed() });
    for corner in &corners_3d {
        data.vertices.push(VertexData {
            point: Point3::new(corner[0], corner[1], corner[2]),
            zone_tol_3d: 1e-7,
            ..VertexData::zeroed()
        });
    }

    // Four boundary sides in loop (CCW) order. `start`/`end` are vertex indices; the
    // edge itself always runs in the iso-curve's natural parameter direction, and the
    // edge use orientation records whether the loop follows that direction.
    //   side 0: v = v0, u increasing        (natural direction u)
    //   side 1: u = u1, v increasing        (natural direction v)
    //   side 2: v = v1, u increasing        (loop traverses it backwards)
    //   side 3: u = u0, v increasing        (loop traverses it backwards)
    let sides = [
        (SurfParamType::V, rectangle.min[1], 0usize, 1usize, true),
        (SurfParamType::U, rectangle.max[0], 1usize, 2usize, true),
        (SurfParamType::V, rectangle.max[1], 3usize, 2usize, false),
        (SurfParamType::U, rectangle.min[0], 0usize, 3usize, false),
    ];
    for (index, (param, iso_parameter, start_vertex, end_vertex, orientation)) in
        sides.into_iter().enumerate()
    {
        let iso = kernel_surface
            .create_iso_parametric_curve(
                &kernel,
                param,
                iso_parameter,
                IsoCurveOptions { domain: Some(uv_domain), uv_iso_curve: false },
            )
            .map_err(|error| format!("iso-parametric boundary curve {index}: {error}"))?;
        let interval = iso.curve.natural_interval();
        data.curves_3d.push(Keyed {
            key: ObjectKey(900 + index as u64),
            object: Curve::from(iso.curve),
        });
        data.edges.push(EdgeData {
            curve: index as u64,
            start_vertex: start_vertex as u64,
            end_vertex: end_vertex as u64,
            prim_eu: index as i64,
            interval,
            zone_tol_3d: 1e-6,
            ..EdgeData::zeroed()
        });
        let uv_start = corners_uv[start_vertex];
        let uv_end = corners_uv[end_vertex];
        let uv_curve = Curve::from(
            BSplineCurve::create_line_segment(
                2,
                Point3::new(uv_start[0], uv_start[1], 0.0),
                Point3::new(uv_end[0], uv_end[1], 0.0),
            )
            .map_err(|error| format!("UV trim curve {index}: {error}"))?,
        );
        data.uv_curves.push(Keyed { key: ObjectKey(1200 + index as u64), object: uv_curve });
        data.edgeuses.push(EuData {
            eu_type: 0,
            edge: index as u64,
            loop_: 0,
            orientation: i32::from(orientation),
            uv_curve: index as u64,
            ..EuData::zeroed()
        });
    }

    data.surfaces.push(Keyed { key: ObjectKey(700), object: converted });
    let mut kernel = crate::kernel::kernel();
    let mut brep = Brep::new_with_infinite_region().map_err(|error| error.to_string())?;
    let mut store = AttributeStore::new();
    let mut records = solidcore::topology::topology::NotifyRecords::new();
    let mut env = TopologyEnv {
        all_attributes: &[],
        attributes: &mut store,
        key_of: &key_of,
        records: &mut records,
        next_key: 1,
    };
    make_topology_from_data(&mut kernel, &mut brep, &mut data, &mut env, MakeTopologyOptions::default())
        .map_err(|error| error.to_string())?;
    Ok(brep)
}

fn is_bspline_surface_entity(entity: &nodecraft_step::Entity) -> bool {
    match entity.name.as_str() {
        "B_SPLINE_SURFACE_WITH_KNOTS" | "B_SPLINE_SURFACE" => true,
        "" => entity.args.iter().any(|value| {
            matches!(value, nodecraft_step::Value::Typed(name, _) if name == "B_SPLINE_SURFACE")
        }),
        _ => false,
    }
}

fn entity_label(entity: &nodecraft_step::Entity) -> String {
    if entity.name.is_empty() {
        "a complex instance".into()
    } else {
        entity.name.clone()
    }
}

/// The face's outer loop must be the natural-domain rectangle, or any axis-aligned
/// rectangle whose corners invert onto the surface.
fn uv_rectangle_from_face_bounds(
    file: &StepFile,
    topology: &StepBrep,
    face: &StepFace,
    surface: &::step_nurbs::BsplineSurface,
) -> Result<UvRectangle, String> {
    if face.bounds.len() != 1 {
        return Err(format!(
            "unsupported: face has {} bounds; only a single outer bound (a sheet without holes) is supported",
            face.bounds.len()
        ));
    }
    let (loop_id, _orientation) = face.bounds[0];
    let step_loop = topology
        .loops
        .get(&loop_id)
        .ok_or_else(|| format!("face loop #{loop_id} is missing from the topology"))?;
    if step_loop.oriented_edges.len() != 4 {
        return Err(format!(
            "unsupported: face outer loop has {} edges; only a four-edge loop is supported",
            step_loop.oriented_edges.len()
        ));
    }

    let (u0, u1) = surface.u_domain();
    let (v0, v1) = surface.v_domain();
    let natural = UvRectangle::new([u0, v0], [u1, v1])
        .map_err(|error| format!("surface has an invalid natural domain: {error}"))?;
    let natural_corners = natural.corners();
    let natural_points = natural_corners
        .iter()
        .map(|corner| surface.point_at(corner[0], corner[1]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("surface corner evaluation: {error}"))?;
    let scale = model_scale(&natural_points);
    let tolerance = SURFACE_TOLERANCE * scale;

    // Ordered loop points and the STEP edge curves between them.
    let mut loop_points = Vec::with_capacity(4);
    let mut loop_edges = Vec::with_capacity(4);
    let mut loop_orientations = Vec::with_capacity(4);
    for oriented in &step_loop.oriented_edges {
        let edge = topology
            .edges
            .get(&oriented.edge)
            .ok_or_else(|| format!("oriented edge references missing edge #{}", oriented.edge))?;
        let (start, end) = if oriented.orientation {
            (edge.start, edge.end)
        } else {
            (edge.end, edge.start)
        };
        let start_point = topology
            .vertices
            .get(&start)
            .ok_or_else(|| format!("edge #{} start vertex #{start} is missing", edge.id))?
            .point;
        let end_point = topology
            .vertices
            .get(&end)
            .ok_or_else(|| format!("edge #{} end vertex #{end} is missing", edge.id))?
            .point;
        loop_points.push([start_point, end_point]);
        loop_edges.push(edge.clone());
        loop_orientations.push(oriented.orientation);
    }
    // The loop must be a closed four-point cycle.
    for index in 0..4 {
        let end = loop_points[index][1];
        let next_start = loop_points[(index + 1) % 4][0];
        if distance(end, next_start) > tolerance {
            return Err(format!(
                "unsupported: face loop is not a closed cycle (edge {index} ends at {end:?}, next starts at {next_start:?})"
            ));
        }
    }

    // Map every loop vertex into UV space: exact for natural-domain corners, otherwise by
    // numerically inverting the surface.
    let mut loop_uv = Vec::with_capacity(4);
    for [start_point, _] in &loop_points {
        let uv = invert_surface_point(surface, *start_point, &natural, tolerance)?;
        loop_uv.push(uv);
    }
    let rectangle = uv_rectangle_from_corners(&loop_uv).ok_or_else(|| {
        format!(
            "unsupported: face bounds are not an axis-aligned UV rectangle (UV corners: {loop_uv:?})"
        )
    })?;
    let span = model_scale(&natural_points).max(1.0);
    if rectangle.width() <= SURFACE_TOLERANCE * span
        || rectangle.height() <= SURFACE_TOLERANCE * span
    {
        return Err("unsupported: face trim rectangle has zero width or height".into());
    }
    if !rectangle_is_inside(&rectangle, &natural, SURFACE_TOLERANCE * span) {
        return Err(format!(
            "unsupported: face trim rectangle {:?}..{:?} lies outside the surface domain",
            rectangle.min, rectangle.max
        ));
    }

    // Straight STEP edges must actually be the rectangle side: check their geometry
    // between the corners against the corresponding iso-curves at 1/4, 1/2 and 3/4.
    for (index, edge) in loop_edges.iter().enumerate() {
        // The stored edge curve always runs in the surface's iso direction; the loop may
        // traverse it backwards (`oriented_edges[index] == false`).
        let (uv_start, uv_end) = if loop_orientations[index] {
            (loop_uv[index], loop_uv[(index + 1) % 4])
        } else {
            (loop_uv[(index + 1) % 4], loop_uv[index])
        };
        let Some(points) = line_like_curve_points(file, edge.curve, loop_points[index]) else {
            continue;
        };
        for (fraction, point) in points {
            let expected = surface
                .point_at(
                    uv_start[0] + (uv_end[0] - uv_start[0]) * fraction,
                    uv_start[1] + (uv_end[1] - uv_start[1]) * fraction,
                )
                .map_err(|error| format!("surface evaluation on edge {index}: {error}"))?;
            if distance(point, expected) > tolerance {
                return Err(format!(
                    "unsupported: straight edge #{0} leaves the UV rectangle (at {fraction} it is {point:?}, expected {expected:?})",
                    edge.id
                ));
            }
        }
    }
    Ok(rectangle)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn model_scale(points: &[[f64; 3]]) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let diagonal = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    diagonal.max(1.0)
}

/// `Some` samples of a line-like STEP edge curve (a `LINE` or a B-spline of degree <= 1)
/// at the given parameter fractions; `None` for curves that cannot be cheaply sampled.
fn line_like_curve_points(
    file: &StepFile,
    curve_id: u32,
    endpoints: [[f64; 3]; 2],
) -> Option<Vec<(f64, [f64; 3])>> {
    let entity = file.entities.get(&curve_id)?;
    if entity.name == "LINE" {
        let [a, b] = endpoints;
        return Some(
            [0.25, 0.5, 0.75]
                .into_iter()
                .map(|fraction| {
                    (
                        fraction,
                        [
                            a[0] + (b[0] - a[0]) * fraction,
                            a[1] + (b[1] - a[1]) * fraction,
                            a[2] + (b[2] - a[2]) * fraction,
                        ],
                    )
                })
                .collect(),
        );
    }
    let curve = ::step_nurbs::parse_curve(file, curve_id).ok()?;
    if curve.degree > 1 {
        return None;
    }
    let (low, high) = curve.domain();
    let mut samples = Vec::with_capacity(3);
    for fraction in [0.25, 0.5, 0.75] {
        let parameter = low + (high - low) * fraction;
        samples.push((fraction, curve.point_at(parameter).ok()?));
    }
    Some(samples)
}

/// Finds the axis-aligned rectangle described by the four ordered loop corners.
///
/// The corners must be exactly the four rectangle corners, in rectangle-cycle order
/// (adjacent corners share one coordinate), so a bow-tie/diagonal loop is rejected.
fn uv_rectangle_from_corners(corners: &[[f64; 2]]) -> Option<UvRectangle> {
    if corners.len() != 4 {
        return None;
    }
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for corner in corners {
        for axis in 0..2 {
            min[axis] = min[axis].min(corner[axis]);
            max[axis] = max[axis].max(corner[axis]);
        }
    }
    let span = ((max[0] - min[0]).abs()).max((max[1] - min[1]).abs()).max(1.0);
    let tolerance = SURFACE_TOLERANCE * span;
    // Canonical cycle: lower-left, lower-right, upper-right, upper-left.
    let expected = [[min[0], min[1]], [max[0], min[1]], [max[0], max[1]], [min[0], max[1]]];
    let mut slots = [usize::MAX; 4];
    for (index, corner) in corners.iter().enumerate() {
        let slot = expected.iter().position(|candidate| {
            (corner[0] - candidate[0]).abs() <= tolerance
                && (corner[1] - candidate[1]).abs() <= tolerance
        })?;
        if slots.contains(&slot) {
            return None;
        }
        slots[index] = slot;
    }
    let mut used = [false; 4];
    for (index, slot) in slots.iter().enumerate() {
        if *slot == usize::MAX || used[*slot] {
            return None;
        }
        used[*slot] = true;
        let next = slots[(index + 1) % 4];
        // Adjacent rectangle corners differ by exactly one step in the canonical ring.
        if (*slot + 1) % 4 != next && (next + 1) % 4 != *slot {
            return None;
        }
    }
    used.iter().all(|flag| *flag).then_some(UvRectangle { min, max })
}

fn rectangle_is_inside(rectangle: &UvRectangle, natural: &UvRectangle, tolerance: f64) -> bool {
    rectangle.min[0] >= natural.min[0] - tolerance
        && rectangle.min[1] >= natural.min[1] - tolerance
        && rectangle.max[0] <= natural.max[0] + tolerance
        && rectangle.max[1] <= natural.max[1] + tolerance
}

/// Inverts `surface` at `target`: the `(u, v)` with `S(u, v) == target`.
///
/// The natural-domain corners are tried exactly first (the common sheet case); otherwise
/// a damped Gauss-Newton iteration is started from a 4x4 seed grid over the domain.
fn invert_surface_point(
    surface: &::step_nurbs::BsplineSurface,
    target: [f64; 3],
    natural: &UvRectangle,
    tolerance: f64,
) -> Result<[f64; 2], String> {
    for corner in natural.corners() {
        let point = surface
            .point_at(corner[0], corner[1])
            .map_err(|error| format!("surface evaluation: {error}"))?;
        if distance(point, target) <= tolerance {
            return Ok(corner);
        }
    }

    let mut best: Option<([f64; 2], f64)> = None;
    for i in 0..4 {
        for j in 0..4 {
            let seed = [
                natural.min[0] + natural.width() * (i as f64 + 0.5) / 4.0,
                natural.min[1] + natural.height() * (j as f64 + 0.5) / 4.0,
            ];
            let Some((uv, residual)) = newton_invert(surface, target, seed, natural) else {
                continue;
            };
            if best.is_none_or(|(_, best_residual)| residual < best_residual) {
                best = Some((uv, residual));
            }
        }
    }
    let Some((uv, residual)) = best else {
        return Err("unsupported: loop corner does not project onto the surface".into());
    };
    if residual > tolerance {
        return Err(format!(
            "unsupported: loop vertex {target:?} is not on the surface (closest point {uv:?}, residual {residual})"
        ));
    }
    Ok(uv)
}

fn newton_invert(
    surface: &::step_nurbs::BsplineSurface,
    target: [f64; 3],
    seed: [f64; 2],
    natural: &UvRectangle,
) -> Option<([f64; 2], f64)> {
    let span = natural.width().max(natural.height()).max(1.0);
    let step = 1e-6 * span;
    let mut uv = seed;
    let mut best = (uv, f64::INFINITY);
    for _ in 0..40 {
        let point = surface.point_at(uv[0], uv[1]).ok()?;
        let residual = [point[0] - target[0], point[1] - target[1], point[2] - target[2]];
        let residual_norm = (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]).sqrt();
        if residual_norm < best.1 {
            best = (uv, residual_norm);
        }
        if residual_norm <= 1e-12 * span.max(1.0) {
            break;
        }
        let u_plus = surface.point_at((uv[0] + step).min(natural.max[0]), uv[1]).ok()?;
        let u_minus = surface.point_at((uv[0] - step).max(natural.min[0]), uv[1]).ok()?;
        let v_plus = surface.point_at(uv[0], (uv[1] + step).min(natural.max[1])).ok()?;
        let v_minus = surface.point_at(uv[0], (uv[1] - step).max(natural.min[1])).ok()?;
        let du = [
            (u_plus[0] - u_minus[0]) / (2.0 * step),
            (u_plus[1] - u_minus[1]) / (2.0 * step),
            (u_plus[2] - u_minus[2]) / (2.0 * step),
        ];
        let dv = [
            (v_plus[0] - v_minus[0]) / (2.0 * step),
            (v_plus[1] - v_minus[1]) / (2.0 * step),
            (v_plus[2] - v_minus[2]) / (2.0 * step),
        ];
        let a = du[0] * du[0] + du[1] * du[1] + du[2] * du[2];
        let b = du[0] * dv[0] + du[1] * dv[1] + du[2] * dv[2];
        let c = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
        let determinant = a * c - b * b;
        if determinant.abs() < 1e-18 {
            break;
        }
        let g0 = du[0] * residual[0] + du[1] * residual[1] + du[2] * residual[2];
        let g1 = dv[0] * residual[0] + dv[1] * residual[1] + dv[2] * residual[2];
        let mut delta_u = (c * g0 - b * g1) / determinant;
        let mut delta_v = (a * g1 - b * g0) / determinant;
        let step_norm = (delta_u * delta_u + delta_v * delta_v).sqrt();
        let max_step = span;
        if step_norm > max_step {
            delta_u *= max_step / step_norm;
            delta_v *= max_step / step_norm;
        }
        uv = [
            (uv[0] - delta_u).clamp(natural.min[0], natural.max[0]),
            (uv[1] - delta_v).clamp(natural.min[1], natural.max[1]),
        ];
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::mass::integrator::properties::brep_area;

    fn exchange(body: &str) -> String {
        format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'nodecraft-solidcore-test','nodecraft-solidcore-test','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{body}\nENDSEC;\n\
             END-ISO-10303-21;\n"
        )
    }

    /// A degree-1 boundary curve between two points, as STEP text entities.
    fn boundary_curve(points: &[u32], entity_id: u32) -> String {
        format!(
            "#{entity_id}=B_SPLINE_CURVE_WITH_KNOTS('',1,({}),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);",
            points
                .iter()
                .map(|id| format!("#{id}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Bilinear unit-square sheet whose face loop is the natural UV rectangle.
    fn planar_sheet_fixture() -> String {
        format!(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,1.,0.));\n\
             #3=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(1.,1.,0.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
             {}\n{}\n{}\n{}\n\
             #30=VERTEX_POINT('',#1);\n\
             #31=VERTEX_POINT('',#3);\n\
             #32=VERTEX_POINT('',#4);\n\
             #33=VERTEX_POINT('',#2);\n\
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
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n",
            boundary_curve(&[1, 3], 20),
            boundary_curve(&[3, 4], 21),
            boundary_curve(&[2, 4], 22),
            boundary_curve(&[1, 2], 23),
        )
    }

    fn parse_fixture(body: &str) -> StepFile {
        StepFile::parse(&exchange(body)).expect("fixture parses")
    }

    fn area(brep: &Brep) -> f64 {
        let mut copy = brep.clone_standing_for_self().expect("standing copy");
        brep_area(&crate::kernel::kernel(), &mut copy, 1e-6).unwrap()
    }

    #[test]
    fn planar_bspline_sheet_builds_one_face_brep() {
        let file = parse_fixture(&planar_sheet_fixture());
        let brep = brep_from_bspline_sheet(&file, 70).unwrap();
        assert_eq!(brep.faces().len(), 1);
        assert_eq!(brep.edges().len(), 4);
        assert_eq!(brep.vertices().len(), 4);
        assert!((area(&brep) - 1.0).abs() < 1e-6, "area {}", area(&brep));
        // The face's UV domain is the natural domain of the surface.
        let face_id = brep.faces().ids().next().unwrap();
        let face = brep.face(face_id).unwrap();
        assert_eq!(face.uv_domain.min().x, 0.0);
        assert_eq!(face.uv_domain.max().y, 1.0);
        assert_eq!(face.rectangular_trim, 1);
    }

    #[test]
    fn uv_rectangle_builder_trims_a_sub_rectangle() {
        let file = parse_fixture(&planar_sheet_fixture());
        let topology = read_topology(&file);
        assert!(topology.faces.contains_key(&70));
        let surface = ::step_nurbs::parse_surface(&file, 10).unwrap();
        let brep = brep_from_bspline_uv_rectangle(&surface, [0.25, 0.25], [0.75, 0.75]).unwrap();
        assert_eq!(brep.faces().len(), 1);
        assert_eq!(brep.edges().len(), 4);
        let trimmed = area(&brep);
        assert!((trimmed - 0.25).abs() < 1e-6, "trimmed area {trimmed}");
        let face_id = brep.faces().ids().next().unwrap();
        let face = brep.face(face_id).unwrap();
        assert_eq!(face.uv_domain.min().x, 0.25);
        assert_eq!(face.uv_domain.max().y, 0.75);
    }

    /// Curved sheet: S(u, v) = (u, v, u^2 + v^2) over [0, 1]^2 as a quadratic Bézier
    /// patch; the four boundary curves are its exact iso-curves.
    fn curved_sheet_fixture() -> String {
        // P[i][j] = (i/2, j/2, (i/2)^2 + (j/2)^2)
        let poles = [
            ["#1", "#2", "#3"],
            ["#4", "#5", "#6"],
            ["#7", "#8", "#9"],
        ];
        let grid = format!(
            "(({},{},{}),({},{},{}),({},{},{}))",
            poles[0][0], poles[0][1], poles[0][2],
            poles[1][0], poles[1][1], poles[1][2],
            poles[2][0], poles[2][1], poles[2][2]
        );
        format!(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,0.5,0.25));\n\
             #3=CARTESIAN_POINT('',(0.,1.,1.));\n\
             #4=CARTESIAN_POINT('',(0.5,0.,0.25));\n\
             #5=CARTESIAN_POINT('',(0.5,0.5,0.5));\n\
             #6=CARTESIAN_POINT('',(0.5,1.,1.25));\n\
             #7=CARTESIAN_POINT('',(1.,0.,1.));\n\
             #8=CARTESIAN_POINT('',(1.,0.5,1.25));\n\
             #9=CARTESIAN_POINT('',(1.,1.,2.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',2,2,{grid},.UNSPECIFIED.,.F.,.F.,.F.,\
                  (3,3),(3,3),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
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
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n",
            grid = grid
        )
    }

    #[test]
    fn curved_bspline_sheet_has_numerical_area() {
        let file = parse_fixture(&curved_sheet_fixture());
        let brep = brep_from_bspline_sheet(&file, 70).unwrap();
        assert_eq!(brep.faces().len(), 1);
        assert_eq!(brep.edges().len(), 4);
        // The quadratic patch has z = 0.5u + 0.5u^2 + 0.5v + 0.5v^2, so
        // |P_u x P_v| = sqrt(1 + (0.5 + u)^2 + (0.5 + v)^2); midpoint-rule reference.
        let n = 400;
        let mut expected = 0.0;
        for i in 0..n {
            for j in 0..n {
                let u = (i as f64 + 0.5) / n as f64;
                let v = (j as f64 + 0.5) / n as f64;
                expected += (1.0 + (0.5 + u).powi(2) + (0.5 + v).powi(2)).sqrt();
            }
        }
        expected /= (n * n) as f64;
        let actual = area(&brep);
        assert!(
            (actual - expected).abs() < 1e-3,
            "curved sheet area {actual}, expected {expected}"
        );
    }

    #[test]
    fn rational_bspline_sheet_builds_brep() {
        // Rational complex-instance form of the same unit-square patch (all weights 1).
        let body = planar_sheet_fixture().replace(
            "#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);",
            "#10=(BOUNDED_SURFACE() B_SPLINE_SURFACE(1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.) \
                  B_SPLINE_SURFACE_WITH_KNOTS((2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.) \
                  GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(((1.,1.),(1.,1.))) \
                  REPRESENTATION_ITEM('') SURFACE());",
        );
        let file = parse_fixture(&body);
        let surface = ::step_nurbs::parse_surface(&file, 10).unwrap();
        assert!(surface.is_rational());
        let brep = brep_from_bspline_sheet(&file, 70).unwrap();
        assert_eq!(brep.faces().len(), 1);
        assert!((area(&brep) - 1.0).abs() < 1e-6, "rational sheet area {}", area(&brep));
    }

    #[test]
    fn file_helper_finds_the_single_sheet() {
        let path = std::env::temp_dir().join(format!(
            "nodecraft-bspline-sheet-{}.step",
            std::process::id()
        ));
        std::fs::write(&path, exchange(&planar_sheet_fixture())).unwrap();
        let brep = brep_from_bspline_sheet_file(path.to_str().unwrap()).unwrap();
        assert_eq!(brep.faces().len(), 1);
        assert!((area(&brep) - 1.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sub_rectangle_face_bounds_are_inverted_to_uv() {
        // Surface: the unit-square patch #1..#4. Face loop: its sub-rectangle
        // [0.25, 0.75] x [0.25, 0.75] with separate corner points #5..#8.
        let body = format!(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,1.,0.));\n\
             #3=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(1.,1.,0.));\n\
             #5=CARTESIAN_POINT('',(0.25,0.25,0.));\n\
             #6=CARTESIAN_POINT('',(0.75,0.25,0.));\n\
             #7=CARTESIAN_POINT('',(0.75,0.75,0.));\n\
             #8=CARTESIAN_POINT('',(0.25,0.75,0.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
             {}\n{}\n{}\n{}\n\
             #30=VERTEX_POINT('',#5);\n\
             #31=VERTEX_POINT('',#6);\n\
             #32=VERTEX_POINT('',#7);\n\
             #33=VERTEX_POINT('',#8);\n\
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
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n",
            boundary_curve(&[5, 6], 20),
            boundary_curve(&[6, 7], 21),
            boundary_curve(&[8, 7], 22),
            boundary_curve(&[5, 8], 23),
        );
        let file = parse_fixture(&body);
        let brep = brep_from_bspline_sheet(&file, 70).unwrap();
        let face_id = brep.faces().ids().next().unwrap();
        let face = brep.face(face_id).unwrap();
        assert!((face.uv_domain.min().x - 0.25).abs() < 1e-6);
        assert!((face.uv_domain.max().y - 0.75).abs() < 1e-6);
        let trimmed = area(&brep);
        assert!((trimmed - 0.25).abs() < 1e-6, "sub-rectangle area {trimmed}");
    }

    /// A sheet whose face loop corners are four points on the unit patch that do not form
    /// an axis-aligned UV rectangle (a diamond).
    fn diamond_sheet_fixture() -> String {
        format!(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,1.,0.));\n\
             #3=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(1.,1.,0.));\n\
             #5=CARTESIAN_POINT('',(0.5,0.,0.));\n\
             #6=CARTESIAN_POINT('',(1.,0.5,0.));\n\
             #7=CARTESIAN_POINT('',(0.5,1.,0.));\n\
             #8=CARTESIAN_POINT('',(0.,0.5,0.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
             {}\n{}\n{}\n{}\n\
             #30=VERTEX_POINT('',#5);\n\
             #31=VERTEX_POINT('',#6);\n\
             #32=VERTEX_POINT('',#7);\n\
             #33=VERTEX_POINT('',#8);\n\
             #40=EDGE_CURVE('',#30,#31,#20,.T.);\n\
             #41=EDGE_CURVE('',#31,#32,#21,.T.);\n\
             #42=EDGE_CURVE('',#32,#33,#22,.T.);\n\
             #43=EDGE_CURVE('',#33,#30,#23,.T.);\n\
             #50=ORIENTED_EDGE('',*,*,#40,.T.);\n\
             #51=ORIENTED_EDGE('',*,*,#41,.T.);\n\
             #52=ORIENTED_EDGE('',*,*,#42,.T.);\n\
             #53=ORIENTED_EDGE('',*,*,#43,.T.);\n\
             #60=EDGE_LOOP('',(#50,#51,#52,#53));\n\
             #61=FACE_OUTER_BOUND('',#60,.T.);\n\
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n",
            boundary_curve(&[5, 6], 20),
            boundary_curve(&[6, 7], 21),
            boundary_curve(&[7, 8], 22),
            boundary_curve(&[8, 5], 23),
        )
    }

    #[test]
    fn non_rectangular_face_bounds_are_unsupported() {
        let file = parse_fixture(&diamond_sheet_fixture());
        let error = brep_from_bspline_sheet(&file, 70).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
    }

    #[test]
    fn bow_tie_face_loop_is_unsupported() {
        // All four natural corners, but the loop crosses diagonally (0 -> 2 -> 1 -> 3).
        let body = planar_sheet_fixture().replace(
            "#60=EDGE_LOOP('',(#50,#51,#52,#53));",
            "#60=EDGE_LOOP('',(#50,#52,#51,#53));",
        );
        let file = parse_fixture(&body);
        let error = brep_from_bspline_sheet(&file, 70).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
    }

    #[test]
    fn three_edge_loop_is_unsupported() {
        let body = planar_sheet_fixture()
            .replace(
                "#60=EDGE_LOOP('',(#50,#51,#52,#53));",
                "#60=EDGE_LOOP('',(#50,#51,#52));",
            )
            .replace("#61=FACE_OUTER_BOUND('',#60,.T.);", "#61=FACE_OUTER_BOUND('',#60,.T.);");
        let file = parse_fixture(&body);
        let error = brep_from_bspline_sheet(&file, 70).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
        assert!(error.contains("four-edge"), "{error}");
    }

    #[test]
    fn non_bspline_surface_is_unsupported() {
        let body = planar_sheet_fixture().replace(
            "#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);",
            "#10=PLANE('',#90);\n#90=AXIS2_PLACEMENT_3D('',#1,$,$);",
        );
        let file = parse_fixture(&body);
        let error = brep_from_bspline_sheet(&file, 70).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
        assert!(error.contains("PLANE"), "{error}");
    }

    #[test]
    fn face_with_hole_is_unsupported() {
        let sheet = planar_sheet_fixture();
        let body = sheet.replace(
            "#61=FACE_OUTER_BOUND('',#60,.T.);",
            "#61=FACE_OUTER_BOUND('',#60,.T.);\n#62=FACE_BOUND('',#60,.T.);",
        )
        .replace("#70=ADVANCED_FACE('',(#61),#10,.T.);", "#70=ADVANCED_FACE('',(#61,#62),#10,.T.);");
        let file = parse_fixture(&body);
        let error = brep_from_bspline_sheet(&file, 70).unwrap_err();
        assert!(error.contains("unsupported"), "{error}");
        assert!(error.contains("bounds"), "{error}");
    }

    #[test]
    fn rectangle_outside_the_natural_domain_is_rejected() {
        let file = parse_fixture(&planar_sheet_fixture());
        let surface = ::step_nurbs::parse_surface(&file, 10).unwrap();
        let error = brep_from_bspline_uv_rectangle(&surface, [-0.5, 0.0], [0.5, 1.0]).unwrap_err();
        assert!(error.contains("outside the surface domain"), "{error}");
        let error = brep_from_bspline_uv_rectangle(&surface, [0.5, 0.0], [0.5, 1.0]).unwrap_err();
        assert!(error.contains("empty"), "{error}");
    }

    #[test]
    fn rectangle_builder_rejects_invalid_surfaces() {
        let mut surface = ::step_nurbs::BsplineSurface {
            u_degree: 1,
            v_degree: 1,
            poles: vec![vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
            weights: None,
            u_knots: ::step_nurbs::Knots::quasi_uniform(1, 2).unwrap(),
            v_knots: ::step_nurbs::Knots::quasi_uniform(1, 2).unwrap(),
            u_closed: false,
            v_closed: false,
            self_intersect: false,
        };
        assert!(brep_from_bspline_uv_rectangle(&surface, [0.0, 0.0], [1.0, 1.0]).is_ok());
        surface.poles[0][0] = [f64::NAN, 0.0, 0.0];
        let error = brep_from_bspline_uv_rectangle(&surface, [0.0, 0.0], [1.0, 1.0]).unwrap_err();
        assert!(error.contains("not finite"), "{error}");
    }

    #[test]
    fn uv_rectangle_helpers_validate_corners() {
        let rectangle = UvRectangle::new([0.0, 0.0], [1.0, 1.0]).unwrap();
        assert_eq!(rectangle.width(), 1.0);
        assert_eq!(rectangle.height(), 1.0);
        assert!(UvRectangle::new([1.0, 0.0], [0.0, 1.0]).is_err());
        assert!(UvRectangle::new([0.0, f64::NAN], [1.0, 1.0]).is_err());
        assert_eq!(
            uv_rectangle_from_corners(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            Some(rectangle)
        );
        assert!(uv_rectangle_from_corners(&[[0.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]).is_none());
        assert!(uv_rectangle_from_corners(&[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]).is_none());
    }
}
