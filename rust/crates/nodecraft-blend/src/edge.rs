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
use solidcore::math::Point3;
use solidcore::query::api_queries as queries;
use solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};
use solidcore::query::face::p1::evaluate_curve;
use solidcore::topology::{Brep, EdgeId};

use crate::error::BlendError;

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
    if !point.iter().all(|coordinate| coordinate.is_finite()) {
        return Err(BlendError::InvalidInput(
            "edge selection point must be finite".into(),
        ));
    }
    let selections = edge_selections(kernel, brep)?;
    if selections.is_empty() {
        return Err(BlendError::InvalidInput(
            "the B-rep has no edges to select".into(),
        ));
    }
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

    let mut best: Option<(f64, f64, &EdgeSelection)> = None;
    for selection in &selections {
        let distance = point_to_edge_distance(kernel, brep, selection.edge, point)?;
        // A closed circle's centre is equidistant to the circle and its mate;
        // break the tie by the exact edge midpoint so old centroid references
        // still resolve to the intended ring.
        let midpoint_distance = length3(sub(selection.midpoint, point));
        if best.is_none_or(|(best_distance, best_mid, _)| {
            distance.total_cmp(&best_distance) == std::cmp::Ordering::Less
                || (distance == best_distance && midpoint_distance < best_mid)
        }) {
            best = Some((distance, midpoint_distance, selection));
        }
    }
    let (distance, _, selection) = best
        .ok_or_else(|| BlendError::InvalidInput("no edge could be sampled for selection".into()))?;
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
        return Err(BlendError::InvalidInput(
            "no edge ordinals were given".into(),
        ));
    }
    Ok(out)
}

/// Expand a selection along SolidCore's tangent edge chain.
///
/// A Boolean join is often split into several edges at seams even though the
/// join is one continuous curve. Selecting one of those edges must round the
/// whole run, not stop at the hidden seam vertex. SolidCore's `tangent_chain`
/// is the shared definition of that run.
pub fn tangent_chain(
    kernel: &mut Kernel,
    brep: &Brep,
    selections: &[EdgeSelection],
    tangency_radians: f64,
) -> Result<Vec<EdgeSelection>, BlendError> {
    let mut working = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| BlendError::Kernel(error.to_string()))?;
    let seeds: Vec<EdgeId> = selections.iter().map(|selection| selection.edge).collect();
    let chain = solidcore::blend::blend_refusal::tangent_chain(
        kernel,
        &mut working,
        &seeds,
        tangency_radians,
    )
    .map_err(|error| BlendError::Kernel(format!("tangent chain: {error}")))?;

    let all = edge_selections(kernel, brep)?;
    let mut out = Vec::with_capacity(chain.len());
    for edge in chain {
        if let Some(selection) = all.iter().find(|selection| selection.edge == edge) {
            out.push(selection.clone());
        }
    }
    Ok(out)
}

/// Select the edges referenced by a NodeCraft Edge-node `values` object.
///
/// The saved `id` is the canonical B-rep edge identity. Its `center` is the
/// exact geometric witness: when the body has changed and the id no longer
/// names the edge under that point, the centre resolves the edge with
/// SolidCore's closest-point measure instead. This is the one resolver shared
/// by the viewer, fillet and chamfer; do not add a separate display-id or
/// midpoint-only path.
pub fn select_edges_from_values(
    kernel: &Kernel,
    brep: &Brep,
    values: &Value,
) -> Result<Vec<EdgeSelection>, BlendError> {
    let refs = values
        .get("edges")
        .and_then(Value::as_array)
        .ok_or_else(|| BlendError::InvalidInput("edge node has no \"edges\" array".into()))?;
    if refs.is_empty() {
        return Err(BlendError::InvalidInput(
            "edge node has no edges selected; select at least one edge in the viewer".into(),
        ));
    }
    let selections = edge_selections(kernel, brep)?;
    // A display pick is f32-rounded, so allow a small absolute error between
    // the saved centre and the exact edge. It only decides whether a saved id
    // is still trustworthy; it never overrides a matching id.
    const CENTER_TOLERANCE: f64 = 1.0e-4;
    let mut out: Vec<EdgeSelection> = Vec::with_capacity(refs.len());
    for (index, reference) in refs.iter().enumerate() {
        let selection =
            resolve_edge_reference(kernel, brep, &selections, reference, CENTER_TOLERANCE)
                .map_err(|error| BlendError::InvalidInput(format!("edges[{index}]: {error}")))?;
        if out.contains(&selection) {
            return Err(BlendError::InvalidInput(format!(
                "edges[{index}]: more than one reference resolves to B-rep edge {}; reselect the join",
                selection.ordinal
            )));
        }
        out.push(selection);
    }
    Ok(out)
}

/// Resolve one saved reference against the exact body.
///
/// The id wins when its edge actually lies under the saved centre; otherwise
/// the centre is the fallback for a stale id. A reference without either is
/// invalid rather than silently ignored.
fn resolve_edge_reference(
    kernel: &Kernel,
    brep: &Brep,
    selections: &[EdgeSelection],
    reference: &Value,
    center_tolerance: f64,
) -> Result<EdgeSelection, BlendError> {
    let center = reference
        .get("center")
        .or_else(|| reference.get("point"))
        .map(json_point)
        .transpose()?;
    if reference
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| id.starts_with("preview-edge:"))
    {
        return Err(BlendError::InvalidInput(
            "a preview-only edge cannot be used for an exact treatment; select a B-rep edge"
                .into(),
        ));
    }
    let ordinal = reference
        .get("ordinal")
        .map(json_ordinal)
        .transpose()?
        .or_else(|| {
            reference
                .get("id")
                .and_then(Value::as_str)
                .and_then(|id| id.rsplit("edge:").next())
                .and_then(|tail| tail.parse::<usize>().ok())
        });
    if let Some(ordinal) = ordinal {
        let selection = selections.get(ordinal).ok_or_else(|| {
            BlendError::InvalidInput(format!(
                "edge ordinal {ordinal} is out of range; the B-rep has {} edges",
                selections.len()
            ))
        })?;
        match center {
            Some(point) => {
                let distance = point_to_edge_distance(kernel, brep, selection.edge, point)?;
                if distance <= center_tolerance {
                    return Ok(selection.clone());
                }
                // The id still addresses an edge, but not the one the centre
                // points at: the body changed under the saved reference.
                select_edge_by_midpoint(kernel, brep, point)
            }
            None => Ok(selection.clone()),
        }
    } else if let Some(point) = center {
        select_edge_by_midpoint(kernel, brep, point)
    } else {
        Err(BlendError::InvalidInput(
            "edge reference needs a \"center\"/\"point\", an \"ordinal\" or an \"id\"".into(),
        ))
    }
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
    let start = evaluate_point(
        kernel,
        record.curve.as_ref(),
        interval.evaluate(0.0),
        ordinal,
    )?;
    let end = evaluate_point(
        kernel,
        record.curve.as_ref(),
        interval.evaluate(1.0),
        ordinal,
    )?;
    let midpoint = evaluate_point(
        kernel,
        record.curve.as_ref(),
        interval.evaluate(0.5),
        ordinal,
    )?;
    let length = solidcore::mass::integrator::properties::edge_length(kernel, brep, edge, 1.0e-9)
        .map_err(|error| BlendError::Kernel(format!("edge {ordinal} length: {error}")))?;
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
    let parameter = curve.map_or(parameter, |curve| {
        let natural = curve.natural_interval();
        parameter.clamp(natural.min(), natural.max())
    });
    let evaluated = evaluate_curve(kernel, curve, parameter)
        .map_err(|error| BlendError::Kernel(format!("edge {ordinal} curve: {error}")))?;
    Ok([evaluated[0].x, evaluated[0].y, evaluated[0].z])
}

/// Distance from `point` to the edge, measured by the kernel.
fn point_to_edge_distance(
    kernel: &Kernel,
    brep: &Brep,
    edge: EdgeId,
    point: [f64; 3],
) -> Result<f64, BlendError> {
    queries::edge_closest_point(
        kernel,
        brep,
        edge,
        Point3::new(point[0], point[1], point[2]),
    )
    .map(|closest| closest.closest.distance)
    .map_err(|error| BlendError::Kernel(format!("edge closest point: {error}")))
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
                BlendError::InvalidInput(format!(
                    "point coordinate {axis} is not a number: {entry}"
                ))
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

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length3(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    #[test]
    fn closed_circle_edge_resolves_from_a_point_on_it() {
        let mut kernel = Kernel::default();
        let cylinder = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 20.0)
            .expect("cylinder");
        let bottom = select_edge_by_midpoint(&kernel, &cylinder, [0.0, 10.0, 0.0])
            .expect("bottom circle by point");
        assert!((bottom.midpoint[2]).abs() < 1.0e-6, "{:?}", bottom.midpoint);
        let top = select_edge_by_midpoint(&kernel, &cylinder, [0.0, 10.0, 20.0])
            .expect("top circle by point");
        assert!(
            (top.midpoint[2] - 20.0).abs() < 1.0e-6,
            "{:?}",
            top.midpoint
        );
    }
}
