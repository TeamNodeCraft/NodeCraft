//! Extrude a planar solidcore face — prototype.
//!
//! Given a solidcore [`Brep`] and a [`FaceId`] this crate
//!
//! 1. reads the face's plane (point + outward normal + planarity deviation) with
//!    [`face_plane`],
//! 2. reads its boundary as ordered loops of [`Point3`]s and [`Curve`]s with
//!    [`face_boundary`],
//! 3. extrudes the face along its normal with [`extrude_planar_face`] (a `Brep`),
//!    or previews the same operation as a [`Mesh`] with
//!    [`extrude_planar_face_preview`].
//!
//! # Extrusion routes
//!
//! The exact route is
//! [`create_taper_extrude`](solidcore::sweep::api_brep::create_taper_extrude) with a
//! zero draft angle and both ends capped (the ported upstream `SmApiCreateTaperExtrude`
//! extrude feature), exposed as [`extrude_planar_face_exact`] and
//! [`extrude_profile_curves`]. For a planar polygon it returns a closed, oriented solid
//! B-rep with exactly the expected area and volume (a box face gives 6 faces with
//! 4-edge loops, 12 shared edges, 8 vertices). A periodic single-curve profile (full
//! circle/ellipse) hits the unported `boolean.merge_p2::SmMerge::SmMerge` inside
//! solidcore; the exact route splits such a profile into two half-arcs and retries.
//!
//! [`extrude_planar_face`] tries the exact route first, then two older curve sweeps:
//!
//! * [`extrude_planar_face_linear_sweep`] uses
//!   [`create_linear_sweep`](solidcore::sweep::api_brep::create_linear_sweep) with
//!   the boundary curves as its profile array.
//! * [`extrude_planar_face_curve_sweep`] uses
//!   [`create_curve_sweep_from_edges`](solidcore::sweep::api_brep::create_curve_sweep_from_edges)
//!   with the face's own boundary edges and a straight path along the normal.
//!
//! Those two curve sweeps measure exactly the same area/volume but leave one live face
//! record without a face use per cap in the B-rep's face arena (the cap creation path
//! of the port), so [`brep_is_connected_solid`] is false for them. Use
//! [`extrude_planar_face_exact`] for exact feature output and
//! [`extrude_planar_face_preview`] for a display mesh. See `NOTES.md` for the route
//! evidence and the precise unported symbol.
//!
//! # STEP export
//!
//! [`planar_brep_to_step_brep`] and [`planar_brep_to_planar_body`] bridge a planar,
//! polygonal B-rep (in particular an exact extrude result) to the neutral forms written
//! by `nodecraft-step`; [`write_planar_brep_step`] writes that chain as AP214 STEP and
//! [`write_exact_extrude_step`] extrudes and writes in one call. Only line-bounded
//! planar faces are representable — curved profile extrusions are rejected with
//! [`ExtrudeError::ExportUnsupported`] and need an analytic/NURBS writer. The worker
//! still needs a feature node that calls this API; see `NOTES.md`.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use solidcore::primitive::api_primitives as api;
//! use solidcore::query::api_queries::get_faces;
//! use solidcore::{Kernel, math::Point3};
//!
//! let mut kernel = Kernel::default();
//! let brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 4.0, 20.0)?;
//! let face = get_faces(&brep)?.last().copied().expect("a box face");
//! let profile = nodecraft_extrude::planar_face_profile(&kernel, &brep, face)?;
//! println!("normal = {:?}", profile.plane.normal);
//! let solid = nodecraft_extrude::extrude_planar_face(&mut kernel, &brep, face, 5.0)?;
//! println!("faces = {}", solid.faces().len());
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use std::fmt;

use solidcore::curve::curve::{Curve, CurveKind};
use solidcore::geom::extent1d::Extent1d;
use solidcore::geom::tol::{USE_DEFAULT, zone_tol3d};
use solidcore::kernel::Kernel;
use solidcore::math::{INFINITE_PARAMETER, Point3, Vec3};
use solidcore::nurbs::b_spline_curve_n_lib::TrimMode;
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::nurbs::knots::Side;
use solidcore::primitive::api_curves;
use solidcore::primitive::primitive_creation::SweepOptions;
use solidcore::query::api_queries::{brep_is_manifold_solid, face_get_boundary_loops};
use solidcore::query::face::p1::evaluate_curve;
use solidcore::sweep::api_brep::{
    create_curve_sweep_from_edges, create_linear_sweep, create_taper_extrude,
};
use solidcore::topology::face::p2 as face_p2;
use solidcore::{Brep, EdgeId, FaceId, SolidError};
use nodecraft_protocol::Mesh;

mod export;
pub use export::{
    ExportableExtrude, extrude_planar_face_exact_for_export, planar_brep_to_planar_body,
    planar_brep_to_step_brep, write_exact_extrude_step, write_planar_brep_step,
};
pub use nodecraft_step::{PlanarBody, StepBrep};

/// Distance below which two consecutive loop samples are treated as one point.
const POINT_EPSILON: f64 = 1.0e-9;

/// Angle tolerance used by the normal-sampling planarity fallback: normals are
/// parallel when their dot product is at least `1 - NORMAL_PARALLEL_TOLERANCE`.
const NORMAL_PARALLEL_TOLERANCE: f64 = 1.0e-6;

/// Default number of segments used to sample one curved boundary curve for a preview.
const DEFAULT_CURVED_SEGMENTS: usize = 8;

/// Errors of the face-plane, boundary, extrusion and preview helpers.
#[derive(Debug)]
pub enum ExtrudeError {
    /// A solidcore call failed.
    Solidcore {
        /// The failing operation, in solidcore terms.
        operation: &'static str,
        /// The underlying solidcore error.
        source: SolidError,
    },
    /// The face has no surface, so it has no normal.
    NoSurface(FaceId),
    /// The face's surface is not planar within tolerance.
    NotPlanar {
        /// The face asked about.
        face: FaceId,
        /// The planarity deviation reported by solidcore.
        deviation: f64,
    },
    /// The face has no boundary loops (or the loop data is empty).
    NoBoundaryLoops(FaceId),
    /// A loop segment's edge has no 3D curve.
    NoCurve {
        /// The face whose boundary is incomplete.
        face: FaceId,
        /// The edge without a curve.
        edge: EdgeId,
    },
    /// A loop's edge-use orientation is neither `Same` nor `Opposite`.
    InvalidOrientation {
        /// The face whose boundary is malformed.
        face: FaceId,
        /// The offending edge.
        edge: EdgeId,
    },
    /// The boundary-loop arrays returned by solidcore are inconsistent.
    MalformedBoundary(FaceId),
    /// A loop has fewer than three distinct points after sampling.
    DegenerateLoop {
        /// The loop index within the face.
        loop_index: usize,
    },
    /// The requested extrusion distance must be finite and greater than zero.
    InvalidDistance(f64),
    /// A multi-loop (holed) face cannot be extruded by the exact route: upstream's
    /// taper extrude consumes a single curve chain.
    MultipleLoopsUnsupported {
        /// The face asked about.
        face: FaceId,
        /// The number of boundary loops found.
        loops: usize,
    },
    /// The profile curve list is empty.
    EmptyProfile,
    /// The planar STEP export cannot represent this B-rep (curved geometry, unwired
    /// face records, no faces, ...).
    ExportUnsupported {
        /// What is not representable.
        why: String,
    },
    /// STEP export (neutral topology conversion or file write) failed.
    Export {
        /// The writer's message.
        message: String,
    },
    /// The preview mesh outgrew the protocol's `u32` index space.
    PreviewOverflow,
}

impl ExtrudeError {
    fn solid(operation: &'static str, source: SolidError) -> Self {
        Self::Solidcore { operation, source }
    }

    /// The underlying solidcore error when this error wraps one.
    #[must_use]
    pub const fn solidcore(&self) -> Option<&SolidError> {
        match self {
            Self::Solidcore { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl fmt::Display for ExtrudeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solidcore { operation, source } => write!(f, "{operation}: {source}"),
            Self::NoSurface(face) => write!(f, "face {face:?} has no surface"),
            Self::NotPlanar { face, deviation } => {
                write!(f, "face {face:?} is not planar (deviation {deviation})")
            }
            Self::NoBoundaryLoops(face) => write!(f, "face {face:?} has no boundary loops"),
            Self::NoCurve { face, edge } => {
                write!(f, "face {face:?} boundary edge {edge:?} has no 3D curve")
            }
            Self::InvalidOrientation { face, edge } => write!(
                f,
                "face {face:?} boundary edge {edge:?} has an invalid edge-use orientation"
            ),
            Self::MalformedBoundary(face) => {
                write!(f, "face {face:?} returned inconsistent boundary-loop arrays")
            }
            Self::DegenerateLoop { loop_index } => {
                write!(f, "boundary loop {loop_index} has fewer than three points")
            }
            Self::InvalidDistance(distance) => {
                write!(f, "extrude amount must be greater than zero (got {distance})")
            }
            Self::MultipleLoopsUnsupported { face, loops } => write!(
                f,
                "face {face:?} has {loops} boundary loops; the exact extrude takes one chain"
            ),
            Self::EmptyProfile => write!(f, "profile has no curves"),
            Self::ExportUnsupported { why } => {
                write!(f, "planar STEP export cannot represent this B-rep: {why}")
            }
            Self::Export { message } => write!(f, "STEP export failed: {message}"),
            Self::PreviewOverflow => write!(f, "preview mesh exceeds the u32 index space"),
        }
    }
}

impl std::error::Error for ExtrudeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Solidcore { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A planar face's plane: a point on it, its outward unit normal and the deviation
/// solidcore measured for the face's surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacePlane {
    /// A point on the plane (the face surface's point).
    pub point: Point3,
    /// Unit normal in the face's (upward face use) sense — outward for primitive
    /// solids.
    pub normal: Vec3,
    /// The planarity deviation reported by solidcore (`0` for an exact plane).
    pub deviation: f64,
}

impl FacePlane {
    /// The signed distance of `point` from the plane, positive along [`Self::normal`].
    #[must_use]
    pub fn signed_distance(&self, point: Point3) -> f64 {
        self.normal.dot(point - self.point)
    }
}

/// One oriented boundary segment of a face loop.
#[derive(Debug, Clone)]
pub struct LoopSegment {
    /// A copy of the edge's 3D curve.
    pub curve: Curve,
    /// The parameter interval of `curve` the edge uses.
    pub interval: Extent1d,
    /// `true` when the loop traverses the curve from [`Extent1d::min`] to
    /// [`Extent1d::max`] (`OrientType::Same`); `false` for `Opposite`.
    pub sense: bool,
    /// The edge this segment came from (for the edge-based sweep route).
    pub edge: EdgeId,
}

impl LoopSegment {
    /// The point at the start of the loop segment.
    ///
    /// # Errors
    ///
    /// Those of [`evaluate_curve`].
    pub fn start_point(&self, k: &Kernel) -> Result<Point3, ExtrudeError> {
        let parameter = if self.sense {
            self.interval.min()
        } else {
            self.interval.max()
        };
        let values = evaluate_curve(k, Some(&self.curve), parameter)
            .map_err(|error| ExtrudeError::solid("query::face::p1::evaluate_curve", error))?;
        Ok(values[0])
    }

    /// A copy of the curve running in the loop direction.
    ///
    /// # Errors
    ///
    /// The copy error of [`Curve::copy`], or the reversal error of
    /// [`Curve::reverse_parameterization`] for a curve kind that cannot be reversed.
    pub fn oriented_curve(&self, k: &Kernel) -> Result<Curve, ExtrudeError> {
        let mut curve = self
            .curve
            .copy(k)
            .map_err(|error| ExtrudeError::solid("curve::Curve::copy", error))?;
        curve = bound_infinite_line(curve, &self.interval);
        if !self.sense {
            curve
                .reverse_parameterization(k, &self.interval)
                .map_err(|error| {
                    ExtrudeError::solid("curve::Curve::reverse_parameterization", error)
                })?;
        }
        Ok(curve)
    }
}

/// The sweep needs the part of a profile curve its edge runs on. A curve read from a
/// STEP record is the kernel's infinite line (analytic domain
/// `[-INFINITE_PARAMETER, INFINITE_PARAMETER]`); `create_taper_extrude` builds its walls
/// from bounded lines, and refuses (returning the profile's cap alone) an infinite one.
/// Bounding the line to `interval` keeps exactly the geometry the edge has there.
fn bound_infinite_line(curve: Curve, interval: &Extent1d) -> Curve {
    use solidcore::curve::line::{BoundedLineOptions, Line};
    let CurveKind::Line(line) = curve.kind() else { return curve };
    let canonical = line.canonical_unit();
    let infinite = (canonical.domain.min() + INFINITE_PARAMETER).abs() < 1.0
        && (canonical.domain.max() - INFINITE_PARAMETER).abs() < 1.0;
    if !infinite || !(interval.max() > interval.min()) {
        return curve;
    }
    let bounded = Line::bounded(
        canonical.point,
        canonical.vector,
        interval,
        &BoundedLineOptions {
            scale: 1.0,
            dim: 3,
            nurb: None,
            inside_out: canonical.inside_out,
        },
    );
    Curve::from(bounded)
}

/// One closed boundary loop of a face, in loop order.
#[derive(Debug, Clone)]
pub struct FaceLoop {
    /// The loop's vertex points in traversal order (the last connects to the
    /// first; the first point is not repeated).
    pub points: Vec<Point3>,
    /// The loop's oriented edge segments in traversal order.
    pub segments: Vec<LoopSegment>,
}

impl FaceLoop {
    /// Samples the loop into an ordered, closed polyline.
    ///
    /// Line segments contribute just their end points; other curve kinds are
    /// sampled with `curved_segments` steps. The first point is not repeated.
    ///
    /// # Errors
    ///
    /// [`ExtrudeError::DegenerateLoop`] for fewer than three distinct points; those
    /// of [`evaluate_curve`].
    pub fn sample(&self, k: &Kernel, curved_segments: usize) -> Result<Vec<Point3>, ExtrudeError> {
        let mut points: Vec<Point3> = Vec::new();
        for segment in &self.segments {
            let count = sample_count(segment, curved_segments);
            let span = segment.interval.max() - segment.interval.min();
            for step in 0..=count {
                let fraction = step as f64 / count as f64;
                let parameter = if segment.sense {
                    segment.interval.min() + fraction * span
                } else {
                    segment.interval.max() - fraction * span
                };
                let values =
                    evaluate_curve(k, Some(&segment.curve), parameter).map_err(|error| {
                        ExtrudeError::solid("query::face::p1::evaluate_curve", error)
                    })?;
                push_distinct(&mut points, values[0]);
            }
        }
        if points.len() > 1 && points_coincide(points[0], points[points.len() - 1]) {
            points.pop();
        }
        if points.len() < 3 {
            return Err(ExtrudeError::DegenerateLoop { loop_index: 0 });
        }
        Ok(points)
    }

    /// Copies of the loop's curves, each reversed when needed so it runs in the
    /// loop direction.
    ///
    /// # Errors
    ///
    /// Those of [`LoopSegment::oriented_curve`].
    pub fn oriented_curves(&self, k: &Kernel) -> Result<Vec<Curve>, ExtrudeError> {
        self.segments
            .iter()
            .map(|segment| segment.oriented_curve(k))
            .collect()
    }
}

/// A planar face prepared for extrusion: its plane and its boundary loops.
#[derive(Debug, Clone)]
pub struct PlanarFaceProfile {
    /// The source face.
    pub face: FaceId,
    /// The face's plane.
    pub plane: FacePlane,
    /// The face's boundary loops, outer loop first.
    pub loops: Vec<FaceLoop>,
}

impl PlanarFaceProfile {
    /// The outer boundary loop, when the face has one.
    #[must_use]
    pub fn outer_loop(&self) -> Option<&FaceLoop> {
        self.loops.first()
    }
}

/// Reads the plane of `face` and checks that it is planar.
///
/// The normal is the surface normal in the face's upward-face-use sense: outward
/// for the primitive solids, and the direction [`extrude_planar_face`] extrudes.
///
/// # Errors
///
/// [`ExtrudeError::NoSurface`] when the face has no surface;
/// [`ExtrudeError::NotPlanar`] when the surface is not planar within the face's
/// tolerance; those of the solidcore query used.
pub fn face_plane(k: &Kernel, brep: &Brep, face: FaceId) -> Result<FacePlane, ExtrudeError> {
    let surface = face_p2::surface(brep, face)
        .map_err(|error| ExtrudeError::solid("topology::face::p2::surface", error))?
        .ok_or(ExtrudeError::NoSurface(face))?;
    let tolerance = face_tolerance(k, brep, face)?;

    match surface.is_planar(k, tolerance) {
        Ok(planar) if planar.is_planar => {
            let normal = planar
                .normal
                .unitize()
                .map_err(|error| ExtrudeError::solid("math::Vec3::unitize", error))?;
            Ok(FacePlane {
                point: planar.point,
                normal,
                deviation: planar.deviation,
            })
        }
        Ok(planar) => Err(ExtrudeError::NotPlanar {
            face,
            deviation: planar.deviation,
        }),
        // A surface whose planarity test needs an unported face tolerance still
        // gets an answer: sample the surface normal over the face's UV domain.
        Err(error) if error.is_unported() => sample_face_plane(k, brep, face, surface, tolerance),
        Err(error) => Err(ExtrudeError::solid(
            "surface::surface::p7::Surface::is_planar",
            error,
        )),
    }
}

/// Reads the boundary of `face` as ordered loops of points and curve segments.
///
/// Loops come from
/// [`face_get_boundary_loops`](solidcore::query::api_queries::face_get_boundary_loops),
/// so the outer loop is first and the loops are oriented in the face's
/// upward-face-use sense.
///
/// # Errors
///
/// [`ExtrudeError::NoBoundaryLoops`] for a face without boundary loops, plus the
/// errors of the solidcore queries and curve evaluation used.
pub fn face_boundary(
    k: &Kernel,
    brep: &Brep,
    face: FaceId,
) -> Result<Vec<FaceLoop>, ExtrudeError> {
    let loops = face_get_boundary_loops(brep, face).map_err(|error| {
        ExtrudeError::solid("query::api_queries::face_get_boundary_loops", error)
    })?;
    if loops.edge_counts.is_empty() {
        return Err(ExtrudeError::NoBoundaryLoops(face));
    }

    let mut result = Vec::with_capacity(loops.edge_counts.len());
    let mut cursor = 0usize;
    for &edge_count in &loops.edge_counts {
        if edge_count == 0 {
            // A vertex loop carries no curve to extrude.
            return Err(ExtrudeError::DegenerateLoop {
                loop_index: result.len(),
            });
        }
        let end = cursor
            .checked_add(edge_count)
            .ok_or(ExtrudeError::MalformedBoundary(face))?;
        let edge_ids = loops
            .edges
            .get(cursor..end)
            .ok_or(ExtrudeError::MalformedBoundary(face))?;
        let orientations = loops
            .orientations
            .get(cursor..end)
            .ok_or(ExtrudeError::MalformedBoundary(face))?;

        let mut segments = Vec::with_capacity(edge_count);
        let mut points = Vec::with_capacity(edge_count);
        for (&edge_id, &orientation) in edge_ids.iter().zip(orientations) {
            let edge = brep
                .edge(edge_id)
                .map_err(|error| ExtrudeError::solid("topology::Brep::edge", error))?;
            let curve = edge
                .curve
                .as_ref()
                .ok_or(ExtrudeError::NoCurve { face, edge: edge_id })?;
            let sense = match orientation {
                OrientType::Same => true,
                OrientType::Opposite => false,
                _ => {
                    return Err(ExtrudeError::InvalidOrientation {
                        face,
                        edge: edge_id,
                    });
                }
            };
            let segment = LoopSegment {
                curve: curve
                    .copy(k)
                    .map_err(|error| ExtrudeError::solid("curve::Curve::copy", error))?,
                interval: edge.interval,
                sense,
                edge: edge_id,
            };
            points.push(segment.start_point(k)?);
            segments.push(segment);
        }
        result.push(FaceLoop { points, segments });
        cursor = end;
    }
    if cursor != loops.edges.len() || cursor != loops.orientations.len() {
        return Err(ExtrudeError::MalformedBoundary(face));
    }
    Ok(result)
}

/// Reads the face's plane and boundary and verifies that every loop point lies on
/// the plane.
///
/// # Errors
///
/// Those of [`face_plane`] and [`face_boundary`]; [`ExtrudeError::NotPlanar`] when a
/// boundary point is off the face's plane by more than ten face tolerances.
pub fn planar_face_profile(
    k: &Kernel,
    brep: &Brep,
    face: FaceId,
) -> Result<PlanarFaceProfile, ExtrudeError> {
    let plane = face_plane(k, brep, face)?;
    let loops = face_boundary(k, brep, face)?;
    let tolerance = face_tolerance(k, brep, face)?;
    let allowed = tolerance.max(zone_tol3d(k, USE_DEFAULT)) * 10.0;
    let mut deviation = 0.0_f64;
    for loop_ in &loops {
        for &point in &loop_.points {
            let distance = plane.signed_distance(point).abs();
            if distance > deviation {
                deviation = distance;
            }
        }
    }
    if deviation > allowed * (1.0 + 1e-9) + 1e-12 {
        return Err(ExtrudeError::NotPlanar { face, deviation });
    }
    Ok(PlanarFaceProfile {
        face,
        plane,
        loops,
    })
}

/// Extrudes `face` into an exact solidcore `Brep` with
/// [`create_taper_extrude`](solidcore::sweep::api_brep::create_taper_extrude) at a zero
/// draft angle and with both ends capped (the ported `SmApiCreateTaperExtrude` extrude
/// feature).
///
/// The result is a closed, oriented solid: for a box face it has 6 faces with 4-edge
/// loops, 12 shared edges and 8 vertices, and exactly the expected area/volume. The
/// face's boundary must be one closed loop; faces with inner (hole) loops are rejected
/// because the upstream taper extrude consumes a single curve chain.
///
/// A profile that is one closed periodic curve (a full circle or ellipse) makes
/// solidcore take its offset/merge fallback, which is blocked by the unported
/// `boolean.merge_p2::SmMerge::SmMerge`. The wrapper then splits that curve into two
/// half-arcs and retries; the geometry stays exact, with the side wall split into two
/// faces at a seam. See `NOTES.md`.
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// [`ExtrudeError::MultipleLoopsUnsupported`] for a holed face; those of
/// [`planar_face_profile`]; the solidcore extrude errors (including
/// [`SolidError::Unported`] naming `SmMerge::SmMerge`/`boolean.merge_p2`).
pub fn extrude_planar_face_exact(
    k: &mut Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    check_distance(distance)?;
    let profile = planar_face_profile(k, brep, face)?;
    sweep_exact(k, &profile, distance)
}

/// Extrudes a closed planar profile given directly as curves into an exact solidcore
/// `Brep` (the same route as [`extrude_planar_face_exact`]).
///
/// The curves must form one closed planar chain in head-tail order; the extrusion
/// direction is the profile's computed normal and `distance` is along it. This is the
/// entry point for STEP profile loops that do not come from an existing B-rep face.
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// [`ExtrudeError::EmptyProfile`] for no curves; the solidcore extrude errors.
pub fn extrude_profile_curves(
    k: &mut Kernel,
    curves: &[Curve],
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    check_distance(distance)?;
    if curves.is_empty() {
        return Err(ExtrudeError::EmptyProfile);
    }
    taper_extrude(k, curves.to_vec(), distance)
}

fn sweep_exact(
    k: &mut Kernel,
    profile: &PlanarFaceProfile,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    if profile.loops.len() != 1 {
        return Err(ExtrudeError::MultipleLoopsUnsupported {
            face: profile.face,
            loops: profile.loops.len(),
        });
    }
    let curves = profile.loops[0].oriented_curves(k)?;
    taper_extrude(k, curves, distance)
}

/// `CreateTaperExtrude` with no draft, both caps, for one profile loop.
///
/// A single closed periodic curve is retried as two half-arcs: solidcore's direct path
/// for such a profile falls into `SmMerge::SmMerge`, which the port does not provide
/// (see `NOTES.md`).
fn taper_extrude(k: &mut Kernel, curves: Vec<Curve>, distance: f64) -> Result<Brep, ExtrudeError> {
    match taper_once(k, &curves, distance) {
        Ok(brep) => Ok(brep),
        Err(original) => {
            let blocked = original
                .solidcore()
                .is_some_and(SolidError::is_unported);
            if blocked && curves.len() == 1 {
                if let Some(halves) = split_closed_curve(k, &curves[0]) {
                    if let Ok(brep) = taper_once(k, &halves, distance) {
                        return Ok(brep);
                    }
                }
            }
            Err(original)
        }
    }
}

fn taper_once(k: &mut Kernel, curves: &[Curve], distance: f64) -> Result<Brep, ExtrudeError> {
    let curves_ref: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
    create_taper_extrude(k, &curves_ref, distance, 0.0, 3, false)
        .map_err(|error| ExtrudeError::solid("sweep::api_brep::create_taper_extrude", error))
}

/// Splits a closed periodic curve into two equal-parameter open halves.
///
/// Returns `None` for a curve that is not closed or cannot be copied/trimmed; the
/// caller then keeps the original solidcore error.
fn split_closed_curve(k: &Kernel, curve: &Curve) -> Option<Vec<Curve>> {
    let interval = curve.natural_interval();
    let closed = curve
        .is_closed(k, &interval, k.tolerances.zone_tol_3d)
        .ok()?;
    if !closed {
        return None;
    }
    let middle = interval.min() + (interval.max() - interval.min()) * 0.5;
    let mut halves = Vec::with_capacity(2);
    for (start, end) in [(interval.min(), middle), (middle, interval.max())] {
        let mut half = curve.copy(k).ok()?;
        let mut trim = Extent1d::from_min_max(start, end);
        half.trim(k, &mut trim, TrimMode::Notify).ok()?;
        halves.push(half);
    }
    Some(halves)
}

/// Extrudes `face` along its plane normal by `distance`, trying
/// [`extrude_planar_face_exact`] first and falling back to the two older curve sweeps
/// ([`extrude_planar_face_linear_sweep`], then
/// [`extrude_planar_face_curve_sweep`]).
///
/// Prefer [`extrude_planar_face_exact`] when exact feature output matters: the legacy
/// curve sweeps measure the same area/volume but leave face records without face uses
/// behind (so [`brep_is_connected_solid`] is false for them).
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// those of [`planar_face_profile`]; the exact route's error when every route fails.
pub fn extrude_planar_face(
    k: &mut Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    check_distance(distance)?;
    let profile = planar_face_profile(k, brep, face)?;
    match sweep_exact(k, &profile, distance) {
        Ok(solid) => Ok(solid),
        Err(exact_error) => match sweep_linear(k, brep, &profile, distance) {
            Ok(solid) => Ok(solid),
            Err(_) => match sweep_edges(k, brep, &profile, distance) {
                Ok(solid) => Ok(solid),
                Err(_) => Err(exact_error),
            },
        },
    }
}

/// Extrudes `face` along its plane normal by `distance` with
/// [`create_linear_sweep`](solidcore::sweep::api_brep::create_linear_sweep),
/// passing the oriented boundary curves as the profile array.
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// those of [`planar_face_profile`]; the solidcore sweep error.
pub fn extrude_planar_face_linear_sweep(
    k: &mut Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    check_distance(distance)?;
    let profile = planar_face_profile(k, brep, face)?;
    sweep_linear(k, brep, &profile, distance)
}

/// Extrudes `face` along its plane normal by `distance` with
/// [`create_curve_sweep_from_edges`](solidcore::sweep::api_brep::create_curve_sweep_from_edges):
/// the face's own boundary edges are the profile and a straight line along the
/// normal is the path.
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// those of [`planar_face_profile`]; the solidcore sweep error.
pub fn extrude_planar_face_curve_sweep(
    k: &mut Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    check_distance(distance)?;
    let profile = planar_face_profile(k, brep, face)?;
    sweep_edges(k, brep, &profile, distance)
}

fn sweep_linear(
    k: &mut Kernel,
    _brep: &Brep,
    profile: &PlanarFaceProfile,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    let mut curves = Vec::new();
    for loop_ in &profile.loops {
        curves.extend(loop_.oriented_curves(k)?);
    }
    if curves.is_empty() {
        return Err(ExtrudeError::NoBoundaryLoops(profile.face));
    }
    let curves_ref: Vec<Option<&Curve>> = curves.iter().map(Some).collect();
    create_linear_sweep(k, &curves_ref, profile.plane.normal, distance, true).map_err(|error| {
        ExtrudeError::solid("sweep::api_brep::create_linear_sweep", error)
    })
}

fn sweep_edges(
    k: &mut Kernel,
    brep: &Brep,
    profile: &PlanarFaceProfile,
    distance: f64,
) -> Result<Brep, ExtrudeError> {
    let mut edges: Vec<(&Brep, EdgeId)> = Vec::new();
    for loop_ in &profile.loops {
        for segment in &loop_.segments {
            edges.push((brep, segment.edge));
        }
    }
    let Some(start) = profile
        .outer_loop()
        .and_then(|loop_| loop_.points.first())
        .copied()
    else {
        return Err(ExtrudeError::NoBoundaryLoops(profile.face));
    };
    let path = api_curves::create_line_segment(start, start + profile.plane.normal * distance)
        .map_err(|error| ExtrudeError::solid("primitive::api_curves::create_line_segment", error))?;
    let path: Curve = path.into();
    let options = SweepOptions {
        cap_ends: true,
        translational_sweep: true,
        move_profile_center_to_path: false,
        orient_profile_perpendicular_to_path: false,
        move_to_profile: true,
        profile_at_path_start: false,
    };
    create_curve_sweep_from_edges(k, &edges, Some(&path), None, None, Some(&options))
        .map(|sweep| sweep.brep)
        .map_err(|error| {
            ExtrudeError::solid("sweep::api_brep::create_curve_sweep_from_edges", error)
        })
}

/// Options of [`extrude_planar_face_preview_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreviewOptions {
    /// Segments per curved boundary curve. Lines always contribute one segment.
    pub curved_segments: usize,
}

impl Default for PreviewOptions {
    fn default() -> Self {
        Self {
            curved_segments: DEFAULT_CURVED_SEGMENTS,
        }
    }
}

/// Extrudes `face` along its plane normal by `distance` and returns a display
/// [`Mesh`] preview with default [`PreviewOptions`].
///
/// # Errors
///
/// Those of [`extrude_planar_face_preview_with_options`].
pub fn extrude_planar_face_preview(
    k: &Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<Mesh, ExtrudeError> {
    extrude_planar_face_preview_with_options(k, brep, face, distance, &PreviewOptions::default())
}

/// Extrudes `face` along its plane normal by `distance` and returns a flat-shaded
/// display [`Mesh`] preview.
///
/// The preview is built directly from the face's boundary loops, so it does not
/// depend on the currently incomplete capped-sweep topology: side walls connect
/// every loop to its translated copy, the outer loop is capped at both ends, and
/// inner (hole) loops contribute side walls only (their caps are not triangulated).
/// Vertices are duplicated per triangle for flat shading and no B-rep edges are
/// emitted.
///
/// # Errors
///
/// [`ExtrudeError::InvalidDistance`] for a non-positive or non-finite `distance`;
/// those of [`planar_face_profile`] and [`FaceLoop::sample`];
/// [`ExtrudeError::PreviewOverflow`] when the mesh outgrows `u32` indices.
pub fn extrude_planar_face_preview_with_options(
    k: &Kernel,
    brep: &Brep,
    face: FaceId,
    distance: f64,
    options: &PreviewOptions,
) -> Result<Mesh, ExtrudeError> {
    check_distance(distance)?;
    let profile = planar_face_profile(k, brep, face)?;
    let curved_segments = options.curved_segments.max(1);

    let mut mesh = Mesh {
        lightweight: true,
        ..Mesh::default()
    };
    let normal = profile.plane.normal;

    for (loop_index, loop_) in profile.loops.iter().enumerate() {
        let points = loop_.sample(k, curved_segments)?;
        let count = points.len();
        let base_center = centroid(&points);
        let top: Vec<Point3> = points
            .iter()
            .map(|point| *point + normal * distance)
            .collect();
        let top_center = base_center + normal * distance;

        // Side walls: one quad (two triangles) per boundary segment.
        for index in 0..count {
            let next = (index + 1) % count;
            let base_a = points[index];
            let base_b = points[next];
            let top_a = top[index];
            let top_b = top[next];
            let face_normal = (base_b - base_a)
                .cross(normal)
                .unitize()
                .unwrap_or(Vec3::ZERO);
            push_triangle(
                &mut mesh,
                face_normal,
                base_a,
                base_b,
                top_b,
                &format!("extrude:{loop_index}:side"),
            )?;
            push_triangle(
                &mut mesh,
                face_normal,
                base_a,
                top_b,
                top_a,
                &format!("extrude:{loop_index}:side"),
            )?;
        }

        // Only the outer loop is capped: filling an inner loop would cover the hole.
        if loop_index != 0 {
            continue;
        }
        for index in 0..count {
            let next = (index + 1) % count;
            push_triangle(
                &mut mesh,
                -normal,
                base_center,
                points[index],
                points[next],
                "extrude:0:base",
            )?;
            push_triangle(
                &mut mesh,
                normal,
                top_center,
                top[index],
                top[next],
                "extrude:0:top",
            )?;
        }
    }

    if mesh.faces.is_empty() {
        return Err(ExtrudeError::DegenerateLoop { loop_index: 0 });
    }
    Ok(mesh)
}

/// Whether `brep` looks like a usable solid: it has faces, every face has a face
/// use and a boundary loop, and solidcore classifies the B-rep as a manifold solid.
///
/// The exact route ([`extrude_planar_face_exact`]) satisfies this. The older capped
/// curve sweeps ([`extrude_planar_face_linear_sweep`],
/// [`extrude_planar_face_curve_sweep`]) do not: the port's cap creation leaves one
/// live face record without a face use per cap. See `NOTES.md`.
#[must_use]
pub fn brep_is_connected_solid(brep: &Brep) -> bool {
    let faces = brep.faces();
    if faces.is_empty() {
        return false;
    }
    for (_face_id, face) in faces.iter() {
        if face.faceuse.is_none() || face.loops.is_empty() {
            return false;
        }
    }
    brep_is_manifold_solid(brep).unwrap_or(false)
}

fn check_distance(distance: f64) -> Result<(), ExtrudeError> {
    if distance.is_finite() && distance > 0.0 {
        Ok(())
    } else {
        Err(ExtrudeError::InvalidDistance(distance))
    }
}

fn face_tolerance(k: &Kernel, brep: &Brep, face: FaceId) -> Result<f64, ExtrudeError> {
    let tolerance = face_p2::tolerance(brep, face)
        .map_err(|error| ExtrudeError::solid("topology::face::p2::tolerance", error))?;
    if tolerance > 0.0 {
        Ok(tolerance)
    } else {
        Ok(zone_tol3d(k, USE_DEFAULT))
    }
}

/// Planarity fallback for surfaces whose `is_planar` needs an unported face
/// tolerance: sample the normal and point over the face's UV domain and check that
/// the normals are parallel and all sampled points are coplanar.
fn sample_face_plane(
    k: &Kernel,
    brep: &Brep,
    face: FaceId,
    surface: &solidcore::surface::surface::Surface,
    tolerance: f64,
) -> Result<FacePlane, ExtrudeError> {
    let domain = face_p2::uv_domain(brep, face)
        .map_err(|error| ExtrudeError::solid("topology::face::p2::uv_domain", error))?;
    let (u_min, u_max) = (domain.min().x, domain.max().x);
    let (v_min, v_max) = (domain.min().y, domain.max().y);
    let fractions = [0.0, 0.5, 1.0];

    let mut reference: Option<(Point3, Vec3)> = None;
    let mut samples: Vec<Point3> = Vec::new();
    let mut first_error: Option<SolidError> = None;
    for &u_fraction in &fractions {
        for &v_fraction in &fractions {
            let uv = solidcore::math::Point2::new(
                u_min + u_fraction * (u_max - u_min),
                v_min + v_fraction * (v_max - v_min),
            );
            match surface.evaluate_normal(k, uv, Side::Left, Side::Left) {
                Ok(normal) if normal.length_squared() > 0.0 => {
                    let point = surface.evaluate_point(k, uv).map_err(|error| {
                        ExtrudeError::solid("surface::Surface::evaluate_point", error)
                    })?;
                    samples.push(point);
                    if reference.is_none() {
                        let normal = normal.unitize().map_err(|error| {
                            ExtrudeError::solid("math::Vec3::unitize", error)
                        })?;
                        reference = Some((point, normal));
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
    }
    let Some((plane_point, plane_normal)) = reference else {
        return Err(first_error.map_or(ExtrudeError::NoSurface(face), |error| {
            ExtrudeError::solid("surface::Surface::evaluate_normal", error)
        }));
    };

    let mut deviation = 0.0_f64;
    let mut parallel = true;
    for &point in &samples {
        let distance = plane_normal.dot(point - plane_point).abs();
        deviation = deviation.max(distance);
        let distance_from_plane_point = (point - plane_point).length_squared();
        if distance_from_plane_point > 0.0
            && distance / distance_from_plane_point.sqrt() > 1.0 - NORMAL_PARALLEL_TOLERANCE
        {
            parallel = false;
        }
    }
    if !parallel || deviation > tolerance * 10.0 {
        return Err(ExtrudeError::NotPlanar { face, deviation });
    }
    Ok(FacePlane {
        point: plane_point,
        normal: plane_normal,
        deviation,
    })
}

fn sample_count(segment: &LoopSegment, curved_segments: usize) -> usize {
    match segment.curve.kind() {
        CurveKind::Line(_) => 1,
        _ => curved_segments.max(1),
    }
}

fn points_coincide(a: Point3, b: Point3) -> bool {
    (a - b).length_squared() <= POINT_EPSILON * POINT_EPSILON
}

fn push_distinct(points: &mut Vec<Point3>, point: Point3) {
    if points
        .last()
        .is_none_or(|last| !points_coincide(*last, point))
    {
        points.push(point);
    }
}

fn centroid(points: &[Point3]) -> Point3 {
    let mut sum = Vec3::ZERO;
    for point in points {
        sum += *point;
    }
    sum * (1.0 / points.len() as f64)
}

/// Emits a flat-shaded triangle with the given normal, swapping two corners when
/// that is needed for the winding to agree with `normal`.
fn push_triangle(
    mesh: &mut Mesh,
    normal: Vec3,
    a: Point3,
    b: Point3,
    c: Point3,
    group: &str,
) -> Result<(), ExtrudeError> {
    let (b, c) = if (b - a).cross(c - a).dot(normal) < 0.0 {
        (c, b)
    } else {
        (b, c)
    };
    let ia = push_vertex(mesh, a, normal)?;
    let ib = push_vertex(mesh, b, normal)?;
    let ic = push_vertex(mesh, c, normal)?;
    mesh.faces.push([ia, ib, ic]);
    mesh.face_groups.push(group.to_owned());
    Ok(())
}

fn push_vertex(mesh: &mut Mesh, point: Point3, normal: Vec3) -> Result<u32, ExtrudeError> {
    let index = u32::try_from(mesh.vertices.len()).map_err(|_| ExtrudeError::PreviewOverflow)?;
    mesh.vertices
        .push([point.x as f32, point.y as f32, point.z as f32]);
    mesh.normals
        .push([normal.x as f32, normal.y as f32, normal.z as f32]);
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::primitive::api_primitives as api;
    use solidcore::query::brep::{BoundingBoxOptions, calculate_bounding_box};

    /// The `10 x 4 x 20` box `create_box` makes at the origin.
    fn box_brep(kernel: &mut Kernel) -> Brep {
        api::create_box(kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 4.0, 20.0).expect("solidcore box")
    }

    fn face_with_normal(k: &Kernel, brep: &Brep, expected: Vec3) -> FaceId {
        for (face, _) in brep.faces().iter() {
            let Ok(plane) = face_plane(k, brep, face) else {
                continue;
            };
            if plane.normal.dot(expected) > 0.999 {
                return face;
            }
        }
        panic!("no face with normal {expected:?}");
    }

    fn bounds(k: &Kernel, brep: &Brep) -> ([f64; 3], [f64; 3]) {
        let extent = calculate_bounding_box(
            k,
            brep,
            BoundingBoxOptions {
                approx: false,
                tight: true,
                expand_by_zone_tol_3d: false,
            },
        )
        .expect("bounding box");
        (
            [extent.min_at(0), extent.min_at(1), extent.min_at(2)],
            [extent.max_at(0), extent.max_at(1), extent.max_at(2)],
        )
    }

    fn assert_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1.0e-9,
                "axis {axis}: {actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn box_top_face_plane_is_xy_plane_at_height() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));
        let plane = face_plane(&k, &brep, top).expect("planar top face");

        assert_close([plane.normal.x, plane.normal.y, plane.normal.z], [0.0, 0.0, 1.0]);
        assert!((plane.point.z - 20.0).abs() < 1.0e-9);
        assert!(plane.deviation <= 1.0e-9);
        // A point on the face is at signed distance zero; a point above it is positive.
        assert!(plane.signed_distance(Point3::new(3.0, 2.0, 20.0)).abs() < 1.0e-9);
        assert!((plane.signed_distance(Point3::new(3.0, 2.0, 25.0)) - 5.0).abs() < 1.0e-9);
    }

    #[test]
    fn box_top_face_boundary_is_rectangle_on_plane() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));
        let loops = face_boundary(&k, &brep, top).expect("top face boundary");

        assert_eq!(loops.len(), 1);
        let outer = &loops[0];
        assert_eq!(outer.points.len(), 4);
        assert_eq!(outer.segments.len(), 4);
        for point in &outer.points {
            assert!((point.z - 20.0).abs() < 1.0e-9, "off-plane point {point:?}");
        }
        let xs: Vec<f64> = outer.points.iter().map(|point| point.x).collect();
        let ys: Vec<f64> = outer.points.iter().map(|point| point.y).collect();
        let min_x = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_x = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_y = ys.iter().copied().fold(f64::INFINITY, f64::min);
        let max_y = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((min_x - 0.0).abs() < 1.0e-9 && (max_x - 10.0).abs() < 1.0e-9);
        assert!((min_y - 0.0).abs() < 1.0e-9 && (max_y - 4.0).abs() < 1.0e-9);

        let sampled = outer.sample(&k, DEFAULT_CURVED_SEGMENTS).expect("sample");
        assert_eq!(sampled.len(), 4, "lines contribute one segment each");
    }

    #[test]
    fn box_top_face_extrusion_distance_and_bounds() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

        for distance in [1.0, 5.0, 12.5] {
            let solid = extrude_planar_face(&mut k, &brep, top, distance).expect("extrude");
            let (min, max) = bounds(&k, &solid);
            assert_close(min, [0.0, 0.0, 20.0]);
            assert_close(max, [10.0, 4.0, 20.0 + distance]);
            assert!(!solid.faces().is_empty());
        }
    }

    #[test]
    fn box_side_face_extrusion_uses_outward_normal() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let side = face_with_normal(&k, &brep, Vec3::new(1.0, 0.0, 0.0));
        let solid = extrude_planar_face(&mut k, &brep, side, 2.0).expect("extrude");
        let (min, max) = bounds(&k, &solid);
        assert_close(min, [10.0, 0.0, 0.0]);
        assert_close(max, [12.0, 4.0, 20.0]);
    }

    #[test]
    fn both_sweep_routes_reach_the_same_bounds() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

        let linear =
            extrude_planar_face_linear_sweep(&mut k, &brep, top, 5.0).expect("linear sweep");
        let curve =
            extrude_planar_face_curve_sweep(&mut k, &brep, top, 5.0).expect("curve sweep");
        let expected_min = [0.0, 0.0, 20.0];
        let expected_max = [10.0, 4.0, 25.0];
        for solid in [&linear, &curve] {
            let (min, max) = bounds(&k, solid);
            assert_close(min, expected_min);
            assert_close(max, expected_max);
        }
    }

    #[test]
    fn non_planar_face_is_rejected() {
        let mut k = Kernel::default();
        let brep =
            api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0).expect("cylinder");

        let mut non_planar = 0usize;
        let mut planar = 0usize;
        for (face, _) in brep.faces().iter() {
            match face_plane(&k, &brep, face) {
                Ok(plane) => {
                    planar += 1;
                    assert!(
                        plane.normal.z.abs() > 0.999,
                        "only the caps are planar in a cylinder"
                    );
                }
                Err(ExtrudeError::NotPlanar { deviation, .. }) => {
                    non_planar += 1;
                    assert!(deviation > 0.0, "non-planar deviation must be positive");
                }
                Err(error) => panic!("unexpected face_plane error: {error}"),
            }
        }
        assert_eq!((planar, non_planar), (2, 1), "cylinder caps and side face");

        let lateral = brep
            .faces()
            .iter()
            .map(|(face, _)| face)
            .find(|&face| {
                matches!(
                    face_plane(&k, &brep, face),
                    Err(ExtrudeError::NotPlanar { .. })
                )
            })
            .expect("cylinder side face");
        let error = extrude_planar_face(&mut k, &brep, lateral, 1.0).expect_err("must reject");
        assert!(
            matches!(error, ExtrudeError::NotPlanar { deviation, .. } if deviation > 0.0),
            "expected NotPlanar, got {error}"
        );
    }

    #[test]
    fn non_positive_and_non_finite_distances_are_rejected() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

        for distance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = extrude_planar_face(&mut k, &brep, top, distance).expect_err("reject");
            assert!(
                matches!(error, ExtrudeError::InvalidDistance(value) if value.to_bits() == distance.to_bits()),
                "expected InvalidDistance({distance}), got {error}"
            );
            let error = extrude_planar_face_preview(&k, &brep, top, distance).expect_err("reject");
            assert!(matches!(error, ExtrudeError::InvalidDistance(_)));
        }
    }

    #[test]
    fn preview_mesh_of_box_top_face_is_closed_and_bounded() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));
        let mesh = extrude_planar_face_preview(&k, &brep, top, 5.0).expect("preview");

        // 4 side quads (2 triangles each) + 2 caps of 4 triangles.
        assert_eq!(mesh.faces.len(), 16);
        assert_eq!(mesh.face_groups.len(), mesh.faces.len());
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        assert!(mesh.lightweight);
        assert!(mesh.brep_edges.is_empty());

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                assert!(vertex[axis].is_finite());
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        assert_close(
            [min[0] as f64, min[1] as f64, min[2] as f64],
            [0.0, 0.0, 20.0],
        );
        assert_close(
            [max[0] as f64, max[1] as f64, max[2] as f64],
            [10.0, 4.0, 25.0],
        );

        for normal in &mesh.normals {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((length - 1.0).abs() < 1.0e-4, "non-unit normal {normal:?}");
        }
        for face in &mesh.faces {
            for index in face {
                assert!((*index as usize) < mesh.vertices.len());
            }
        }
        // Flat-shaded preview: every vertex belongs to exactly one triangle.
        assert_eq!(mesh.vertices.len(), 3 * mesh.faces.len());
    }

    #[test]
    fn connected_solid_diagnostics() {
        let mut k = Kernel::default();
        let solid = box_brep(&mut k);
        assert!(brep_is_connected_solid(&solid));

        let cylinder =
            api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0).expect("cylinder");
        assert!(brep_is_connected_solid(&cylinder));
    }

    #[test]
    fn exact_extrude_matches_solidcore_ops_prism_volume() {
        let mut kernel = Kernel::default();
        let brep = box_brep(&mut kernel);
        let top = face_with_normal(&kernel, &brep, Vec3::new(0.0, 0.0, 1.0));
        let extruded = extrude_planar_face_exact(&mut kernel, &brep, top, 5.0).expect("extrude");
        let extruded_volume = nodecraft_layers::kernel_volume(&mut kernel, &extruded)
            .expect("extruded volume");

        let region = solidcore_record::Region2 {
            outer: solidcore_record::Loop2::polygon(&[
                solidcore_record::Vec2::new(0.0, 0.0),
                solidcore_record::Vec2::new(10.0, 0.0),
                solidcore_record::Vec2::new(10.0, 4.0),
                solidcore_record::Vec2::new(0.0, 4.0),
            ]),
            holes: Vec::new(),
        };
        let prism = solidcore_ops::prism::extrude_planar_regions(
            &solidcore_record::Plane::XY,
            &[region],
            20.0,
            25.0,
        )
        .expect("ops prism");
        let prism_volume = prism.volume(1.0e-7).expect("prism volume");

        assert!((extruded_volume - 200.0).abs() < 1.0e-6, "{extruded_volume}");
        assert!((prism_volume - 200.0).abs() < 1.0e-6, "{prism_volume}");
        assert!((extruded_volume - prism_volume).abs() < 1.0e-6);
    }

    /// Documents the pinned solidcore capped-sweep defect: both sweep routes produce
    /// correct geometry but not a connected solid yet. Unignore once solidcore's cap
    /// creation/stitching produces one shell.
    #[test]
    #[ignore = "known solidcore capped-sweep topology defect; unignore when fixed"]
    fn sweep_routes_do_not_yet_produce_a_connected_solid() {
        let mut k = Kernel::default();
        let brep = box_brep(&mut k);
        let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));
        let linear = extrude_planar_face_linear_sweep(&mut k, &brep, top, 5.0).expect("linear");
        let curve = extrude_planar_face_curve_sweep(&mut k, &brep, top, 5.0).expect("curve");
        assert!(!brep_is_connected_solid(&linear));
        assert!(!brep_is_connected_solid(&curve));
    }
}
