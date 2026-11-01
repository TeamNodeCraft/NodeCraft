//! B-spline / NURBS curve and surface support for STEP (ISO 10303-21) data.
//!
//! The crate sits on top of [`nodecraft_step`]'s generic STEP entity parser. STEP
//! writes rational B-splines as *complex instances*, for example
//!
//! ```text
//! #20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
//!      B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
//!      CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE((1.,0.7,1.))
//!      REPRESENTATION_ITEM(''));
//! ```
//!
//! which [`nodecraft_step::StepFile`] parses as an entity with an empty name whose
//! arguments alternate between typed member names and their argument lists. This crate
//! reassembles those members into one [`BsplineCurve`] / [`BsplineSurface`], resolves
//! the control-point references to coordinates, derives default knot vectors when a
//! file omits them, and offers evaluation plus matching entity-string emitters.
//!
//! Supported entities:
//!
//! * `B_SPLINE_CURVE_WITH_KNOTS` (simple entity, non-rational),
//! * `B_SPLINE_CURVE` (simple entity without knot data — uniform knots are derived),
//! * complex instances with `B_SPLINE_CURVE`, `B_SPLINE_CURVE_WITH_KNOTS` and
//!   `RATIONAL_B_SPLINE_CURVE` members,
//! * `B_SPLINE_SURFACE_WITH_KNOTS`, `B_SPLINE_SURFACE`, and complex instances with the
//!   matching `B_SPLINE_SURFACE` / `RATIONAL_B_SPLINE_SURFACE` members.
//!
//! All geometry is returned in the STEP file's own coordinates (no unit conversion).
//!
//! # Example
//!
//! ```no_run
//! use nodecraft_step::StepFile;
//! use step_nurbs::{curves, curve_point};
//!
//! let text = std::fs::read_to_string("part.step").unwrap();
//! let file = StepFile::parse(&text).unwrap();
//! for curve in curves(&file).unwrap() {
//!     let (low, high) = curve.domain();
//!     let point = curve_point(&curve, 0.5 * (low + high)).unwrap();
//!     println!("{point:?}");
//! }
//! ```

pub mod emit;
pub mod eval;
pub mod knots;
pub mod parse;

pub mod mesh;
pub mod composite;
pub mod sweep;
pub use sweep::{parse_offset_surface, parse_surface_of_linear_extrusion, parse_surface_of_revolution, OffsetSurface, SurfaceOfLinearExtrusion, SurfaceOfRevolution};
pub use composite::{parse_composite_curve, CompositeCurve, CompositeSegment};
pub use mesh::{tessellate_surface, tessellate_trimmed_surface, SurfaceMesh};
pub use emit::{
    bspline_curve_entity, bspline_surface_entity, format_int_list, format_real, format_real_list,
};
pub use eval::{curve_point, surface_point};
pub use knots::{KnotSpec, Knots};
pub use parse::{curves, parse_curve, parse_surface, surfaces};

/// A polynomial or rational B-spline curve in 3D.
#[derive(Debug, Clone, PartialEq)]
pub struct BsplineCurve {
    /// Polynomial degree (1 = straight segments, 3 = cubic, ...).
    pub degree: usize,
    /// Control points in file coordinates.
    pub poles: Vec<[f64; 3]>,
    /// Weights for a rational curve; `None` for a polynomial B-spline.
    pub weights: Option<Vec<f64>>,
    /// Knot multiplicities/values plus how they were obtained.
    pub knots: Knots,
    /// `closed_curve` flag (STEP `LOGICAL`; `.U.` unknown reads as `false`).
    pub closed: bool,
    /// `self_intersect` flag (STEP `LOGICAL`; `.U.` unknown reads as `false`).
    pub self_intersect: bool,
}

impl BsplineCurve {
    /// `true` when the curve carries `RATIONAL_B_SPLINE_CURVE` weights.
    pub fn is_rational(&self) -> bool {
        self.weights.is_some()
    }

    /// Weight of pole `index` (1.0 for polynomial curves).
    pub fn weight(&self, index: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |weights| weights[index])
    }

    /// Expanded knot vector (each value repeated by its multiplicity).
    pub fn knot_vector(&self) -> Vec<f64> {
        self.knots.expanded()
    }

    /// Parameter domain `[U[degree], U[poles.len()]]` of the curve.
    pub fn domain(&self) -> (f64, f64) {
        eval::curve_domain(self)
    }


    /// Samples the curve at `segments + 1` uniformly spaced parameters over its domain.
    pub fn sample(&self, segments: usize) -> Result<Vec<[f64; 3]>, String> {
        if segments == 0 { return Err("sample count must be positive".into()); }
        let (start, end) = self.domain();
        let mut out = Vec::with_capacity(segments + 1);
        for index in 0..=segments {
            let u = start + (end - start) * index as f64 / segments as f64;
            out.push(self.point_at(u)?);
        }
        Ok(out)
    }

    /// Point on the curve at `u` (clamped to [`Self::domain`]).
    pub fn point_at(&self, u: f64) -> Result<[f64; 3], String> {
        eval::curve_point(self, u)
    }
}

/// A polynomial or rational B-spline surface in 3D.
#[derive(Debug, Clone, PartialEq)]
pub struct BsplineSurface {
    /// Polynomial degree in `u`.
    pub u_degree: usize,
    /// Polynomial degree in `v`.
    pub v_degree: usize,
    /// Control net indexed `poles[u][v]`.
    pub poles: Vec<Vec<[f64; 3]>>,
    /// Weights indexed `weights[u][v]`; `None` for a polynomial surface.
    pub weights: Option<Vec<Vec<f64>>>,
    /// Knot multiplicities/values in `u`.
    pub u_knots: Knots,
    /// Knot multiplicities/values in `v`.
    pub v_knots: Knots,
    /// `u_closed` flag (STEP `LOGICAL`; `.U.` unknown reads as `false`).
    pub u_closed: bool,
    /// `v_closed` flag (STEP `LOGICAL`; `.U.` unknown reads as `false`).
    pub v_closed: bool,
    /// `self_intersect` flag (STEP `LOGICAL`; `.U.` unknown reads as `false`).
    pub self_intersect: bool,
}

impl BsplineSurface {
    /// `true` when the surface carries `RATIONAL_B_SPLINE_SURFACE` weights.
    pub fn is_rational(&self) -> bool {
        self.weights.is_some()
    }

    /// Weight of pole `(i, j)` (1.0 for polynomial surfaces).
    pub fn weight(&self, i: usize, j: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |weights| weights[i][j])
    }

    /// Number of control points along `u`.
    pub fn u_control_count(&self) -> usize {
        self.poles.len()
    }

    /// Number of control points along `v`.
    pub fn v_control_count(&self) -> usize {
        self.poles.first().map_or(0, Vec::len)
    }

    /// Parameter domain in `u`.
    pub fn u_domain(&self) -> (f64, f64) {
        eval::knot_domain(&self.u_knots.expanded(), self.u_degree)
    }

    /// Parameter domain in `v`.
    pub fn v_domain(&self) -> (f64, f64) {
        eval::knot_domain(&self.v_knots.expanded(), self.v_degree)
    }


    /// Samples a `(u_segments+1) × (v_segments+1)` grid over the parameter domain.
    pub fn sample(&self, u_segments: usize, v_segments: usize) -> Result<Vec<Vec<[f64; 3]>>, String> {
        if u_segments == 0 || v_segments == 0 { return Err("sample counts must be positive".into()); }
        let (u0, u1) = self.u_domain();
        let (v0, v1) = self.v_domain();
        let mut rows = Vec::with_capacity(u_segments + 1);
        for i in 0..=u_segments {
            let u = u0 + (u1 - u0) * i as f64 / u_segments as f64;
            let mut row = Vec::with_capacity(v_segments + 1);
            for j in 0..=v_segments {
                let v = v0 + (v1 - v0) * j as f64 / v_segments as f64;
                row.push(self.point_at(u, v)?);
            }
            rows.push(row);
        }
        Ok(rows)
    }

    /// Point on the surface at `(u, v)` (clamped to the parameter domains).
    pub fn point_at(&self, u: f64, v: f64) -> Result<[f64; 3], String> {
        eval::surface_point(self, u, v)
    }
}
