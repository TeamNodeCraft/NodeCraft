//! Edge selection on a solidcore B-rep.
//!
//! NodeCraft's Edge node stores references to edges of the upstream shape. The app
//! collects them in the viewer from the display mesh and passes them through as
//! `values.edges`; each reference carries an `ordinal` (index in the kernel edge list)
//! and/or a `center`/`point` in the centered display-mesh coordinates.
//!
//! This module turns either form into an [`EdgeSelection`] with exact geometry sampled
//! from the B-rep (never from the mesh), so a treatment can be placed independently of
//! tessellation density. It only uses solidcore's public query API
//! (`SmApiGetEdges`, `SmEdge::GetInterval`, `SmApiEvaluateCurve`, `SmApiBrepBoundingBox`),
//! all of which are ported at the locked revision.

use serde_json::Value;
use solidcore::kernel::Kernel;
use solidcore::query::api_queries as queries;
use solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};
use solidcore::query::face::p1::evaluate_curve;
use solidcore::topology::{Brep, EdgeId};

use crate::error::BlendError;

/// Parameter samples used to measure an edge and to find the nearest one.
const SAMPLES: usize = 16;

/// A point farther than this fraction of the B-rep diagonal from every edge
/// midpoint is not treated as an edge selection.
const SELECTION_RADIUS_FRACTION: f64 = 0.25;

/// One selected B-rep edge with exact geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSelection {
    /// Index in `SmBrep::GetEdges` order (the node's `ordinal`).
    pub ordinal: usize,
    /// solidcore edge id.
    pub edge: EdgeId,
    /// Curve start point in B-rep coordinates.
    pub start: [f64; 3],
    /// Curve end point in B-rep coordinates.
    pub end: [f64; 3],
    /// Curve point at the parameter interval's midpoint.
    pub midpoint: [f64; 3],
    /// Approximate arc length (16-sample polyline).
    pub length: f64,
}

impl EdgeSelection {
    /// Unit direction from [`Self::start`] to [`Self::end`], when the edge has length.
    pub fn direction(&self) -> Result<[f64; 3], BlendError> {
        let delta = sub(self.end, self.start);
        let norm = length3(delta);
        if !(norm > 0.0) || !norm.is_finite() {
            return Err(BlendError::InvalidInput(format!(
                "edge {} is degenerate (zero length)",
                self.ordinal
            )));
        }
        Ok([delta[0] / norm, delta[1] / norm, delta[2] / norm])
    }
}

/// The B-rep's tight bounding-box centre, i.e. the offset between B-rep coordinates
/// and the centered display-mesh coordinates NodeCraft primitives use.
pub fn brep_center(kernel: &Kernel, brep: &Brep) -> Result<[f64; 3], BlendError> {
    let extent = calculate_bounding_box(
        kernel,
        brep,
        BoundingBoxOptions {
            approx: false,
            tight: true,
            expand_by_zone_tol_3d: false,
        },
    )
    .map_err(|error| BlendError::Kernel(format!("bounding box: {error}")))?;
    Ok([
        (extent.min_at(0) + extent.max_at(0)) * 0.5,
        (extent.min_at(1) + extent.max_at(1)) * 0.5,
        (extent.min_at(2) + extent.max_at(2)) * 0.5,
    ])
}

/// All edges of the B-rep in `SmBrep::GetEdges` order, with sampled geometry.
pub fn edge_selections(kernel: &Kernel, brep: &Brep) -> Result<Vec<EdgeSelection>, BlendError> {
    let ids = queries::get_edges(brep)
        .map_err(|error| BlendError::Kernel(format!("get_edges: {error}")))?;
    let mut out = Vec::with_capacity(ids.len());
    for (ordinal, edge) in ids.into_iter().enumerate() {
        out.push(edge_selection(kernel, brep, edge, ordinal)?);
    }
    Ok(out)
}

/// Select the edge nearest to `point` (B-rep coordinates), e.g. its approximate
/// midpoint.
pub fn select_edge_by_midpoint(
    kernel: &Kernel,
    brep: &Brep,
    point: [f64; 3],
) -> Result<EdgeSelection, BlendError> {
    let selections = edge_selections(kernel, brep)?;
    if selections.is_empty() {
        return Err(BlendError::InvalidInput(
            "the B-rep has no edges to select".into(),
        ));
    }
    let center = brep_center(kernel, brep)?;
    let extent = calculate_bounding_box(
        kernel,
        brep,
        BoundingBoxOptions {
            approx: false,
            tight: true,
            expand_by_zone_tol_3d: false,
        },
    )
    .map_err(|error| BlendError::Kernel(format!("bounding box: {error}")))?;
    let diagonal = length3([
        extent.max_at(0) - extent.min_at(0),
        extent.max_at(1) - extent.min_at(1),
        extent.max_at(2) - extent.min_at(2),
    ]);
    let limit = diagonal * SELECTION_RADIUS_FRACTION;
    let _ = center;

    let mut best: Option<(f64, &EdgeSelection)> = None;
    for selection in &selections {
        let distance = point_to_polyline_distance(
            kernel,
            brep,
            selection.edge,
            point,
        )?;
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, selection));
        }
    }
    let (distance, selection) = best.ok_or_else(|| {
        BlendError::InvalidInput("no edge could be sampled for selection".into())
    })?;
    if distance > limit {
        return Err(BlendError::InvalidInput(format!(
            "point {} is not within {limit:.6} of any edge (nearest is edge {} at \
             distance {distance:.6})",
            format_point(point),
            selection.ordinal
        )));
    }
    Ok(selection.clone())
}

/// Select one edge per approximate midpoint (B-rep coordinates), in input order.
pub fn select_edges_by_midpoints(
    kernel: &Kernel,
    brep: &Brep,
    points: &[[f64; 3]],
) -> Result<Vec<EdgeSelection>, BlendError> {
    let mut out = Vec::with_capacity(points.len());
    for point in points {
        let selection = select_edge_by_midpoint(kernel, brep, *point)?;
        if !out.contains(&selection) {
            out.push(selection);
        }
    }
    if out.is_empty() {
        return Err(BlendError::InvalidInput(
            "no edge midpoints were given".into(),
        ));
    }
    Ok(out)
}

/// Select one edge per ordinal (index in `SmBrep::GetEdges` order).
pub fn select_edges_by_ordinals(
    kernel: &Kernel,
    brep: &Brep,
    ordinals: &[usize],
) -> Result<Vec<EdgeSelection>, BlendError> {
    let selections = edge_selections(kernel, brep)?;
    let mut out = Vec::with_capacity(ordinals.len());
    for ordinal in ordinals {
        let selection = selections.get(*ordinal).ok_or_else(|| {
            BlendError::InvalidInput(format!(
                "edge ordinal {ordinal} is out of range; the B-rep has {} edges",
                selections.len()
            ))
        })?;
        if !out.contains(selection) {
            out.push(selection.clone());
        }
    }
    if out.is_empty() {
        return Err(BlendError::InvalidInput("no edge ordinals were given".into()));
    }
    Ok(out)
}

/// Select the edges referenced by a NodeCraft Edge-node `values` object.
///
/// Each entry of `values.edges` is matched, in priority order, by
///  1. `center` or `point` (centered display-mesh coordinates): nearest edge,
///  2. `ordinal`: index in the solidcore edge list.
///
/// Duplicates collapse while preserving the node's order. An empty or missing
/// selection is an explicit [`BlendError::InvalidInput`].
pub fn select_edges_from_values(
    kernel: &Kernel,
    brep: &Brep,
    values: &Value,
) -> Result<Vec<EdgeSelection>, BlendError> {
    let refs = values
        .get("edges")
        .ok_or_else(|| BlendError::InvalidInput("edge node has no \"edges\" field".into()))?;
    let refs = refs
        .as_array()
        .ok_or_else(|| BlendError::InvalidInput("edge node \"edges\" must be an array".into()))?;
    if refs.is_empty() {
        return Err(BlendError::InvalidInput(
            "edge node has no edges selected; select at least one edge in the viewer".into(),
        ));
    }
    let center = brep_center(kernel, brep)?;
    let mut out: Vec<EdgeSelection> = Vec::with_capacity(refs.len());
    for (index, reference) in refs.iter().enumerate() {
        let selection = select_reference(kernel, brep, reference, center).map_err(|error| {
            BlendError::InvalidInput(format!("edges[{index}]: {error}"))
        })?;
        if !out.contains(&selection) {
            out.push(selection);
        }
    }
    Ok(out)
}

/// Resolve one node edge reference.
fn select_reference(
    kernel: &Kernel,
    brep: &Brep,
    reference: &Value,
    center: [f64; 3],
) -> Result<EdgeSelection, BlendError> {
    if let Some(point) = reference.get("center").or_else(|| reference.get("point")) {
        let centered = json_point(point)?;
        let in_brep = add(centered, center);
        return select_edge_by_midpoint(kernel, brep, in_brep);
    }
    if let Some(ordinal) = reference.get("ordinal") {
        let ordinal = json_ordinal(ordinal)?;
        return select_edges_by_ordinals(kernel, brep, &[ordinal])
            .map(|mut selections| selections.remove(0))
            .map_err(|error| BlendError::InvalidInput(error.to_string()));
    }
    Err(BlendError::InvalidInput(
        "edge reference needs a \"center\"/\"point\" or an \"ordinal\"".into(),
    ))
}

/// Sample one edge into an [`EdgeSelection`].
fn edge_selection(
    kernel: &Kernel,
    brep: &Brep,
    edge: EdgeId,
    ordinal: usize,
) -> Result<EdgeSelection, BlendError> {
    let record = brep
        .edge(edge)
        .map_err(|error| BlendError::Kernel(format!("edge {ordinal}: {error}")))?;
    let interval = record.interval;
    let start = evaluate_point(kernel, record.curve.as_ref(), interval.evaluate(0.0), ordinal)?;
    let end = evaluate_point(kernel, record.curve.as_ref(), interval.evaluate(1.0), ordinal)?;
    let midpoint = evaluate_point(kernel, record.curve.as_ref(), interval.evaluate(0.5), ordinal)?;
    let mut length = 0.0;
    let mut previous = start;
    for sample in 1..=SAMPLES {
        let parameter = interval.evaluate(sample as f64 / SAMPLES as f64);
        let point = evaluate_point(kernel, record.curve.as_ref(), parameter, ordinal)?;
        length += length3(sub(point, previous));
        previous = point;
    }
    Ok(EdgeSelection {
        ordinal,
        edge,
        start,
        end,
        midpoint,
        length,
    })
}

/// Evaluate a curve point, turning solidcore errors into blend errors.
fn evaluate_point(
    kernel: &Kernel,
    curve: Option<&solidcore::curve::curve::Curve>,
    parameter: f64,
    ordinal: usize,
) -> Result<[f64; 3], BlendError> {
    let evaluated = evaluate_curve(kernel, curve, parameter)
        .map_err(|error| BlendError::Kernel(format!("edge {ordinal} curve: {error}")))?;
    Ok([evaluated[0].x, evaluated[0].y, evaluated[0].z])
}

/// Distance from `point` to the edge's sampled polyline.
fn point_to_polyline_distance(
    kernel: &Kernel,
    brep: &Brep,
    edge: EdgeId,
    point: [f64; 3],
) -> Result<f64, BlendError> {
    let record = brep
        .edge(edge)
        .map_err(|error| BlendError::Kernel(format!("edge drop: {error}")))?;
    let interval = record.interval;
    let mut samples = Vec::with_capacity(SAMPLES + 1);
    for sample in 0..=SAMPLES {
        let parameter = interval.evaluate(sample as f64 / SAMPLES as f64);
        samples.push(evaluate_point(kernel, record.curve.as_ref(), parameter, 0)?);
    }
    let mut best = f64::INFINITY;
    for pair in samples.windows(2) {
        best = best.min(point_segment_distance(point, pair[0], pair[1]));
    }
    Ok(best)
}

/// Distance from a point to a segment.
fn point_segment_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let length_squared = dot(ab, ab);
    if !(length_squared > 0.0) {
        return length3(sub(point, a));
    }
    let t = (dot(sub(point, a), ab) / length_squared).clamp(0.0, 1.0);
    length3(sub(point, add(a, scale(ab, t))))
}

/// Parse `[x, y, z]` or `{ "x": .., "y": .., "z": .. }`.
fn json_point(value: &Value) -> Result<[f64; 3], BlendError> {
    if let Some(array) = value.as_array() {
        if array.len() != 3 {
            return Err(BlendError::InvalidInput(format!(
                "point must have three coordinates, got {}",
                array.len()
            )));
        }
        let mut out = [0.0; 3];
        for (axis, entry) in array.iter().enumerate() {
            out[axis] = entry.as_f64().ok_or_else(|| {
                BlendError::InvalidInput(format!("point coordinate {axis} is not a number: {entry}"))
            })?;
        }
        return Ok(out);
    }
    if value.is_object() {
        let mut out = [0.0; 3];
        for (axis, key) in ["x", "y", "z"].iter().enumerate() {
            let entry = value.get(key).ok_or_else(|| {
                BlendError::InvalidInput(format!("point object is missing {key:?}"))
            })?;
            out[axis] = entry.as_f64().ok_or_else(|| {
                BlendError::InvalidInput(format!("point {key:?} is not a number: {entry}"))
            })?;
        }
        return Ok(out);
    }
    Err(BlendError::InvalidInput(format!(
        "point must be an array or an object, got {value}"
    )))
}

/// Parse a non-negative integer ordinal from JSON.
fn json_ordinal(value: &Value) -> Result<usize, BlendError> {
    if let Some(number) = value.as_u64() {
        return usize::try_from(number)
            .map_err(|_| BlendError::InvalidInput(format!("ordinal {number} is too large")));
    }
    Err(BlendError::InvalidInput(format!(
        "edge ordinal must be a non-negative integer, got {value}"
    )))
}

fn format_point(point: [f64; 3]) -> String {
    format!("[{:.6}, {:.6}, {:.6}]", point[0], point[1], point[2])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length3(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
