//! Analytic conic STEP support for `ELLIPSE`, `HYPERBOLA` and `PARABOLA`.
//!
//! This crate reads and writes the analytic conic curve entities of ISO-10303-21
//! (AP214) on top of the public [`nodecraft_step`] reader:
//!
//! * [`ConicScene::parse`] / [`read_conics`] parse every `ELLIPSE`, `HYPERBOLA` and
//!   `PARABOLA` in a STEP file together with the `AXIS2_PLACEMENT_3D`, `DIRECTION` and
//!   `CARTESIAN_POINT` entities they reference.
//! * [`Conic`], [`Ellipse`], [`Hyperbola`], [`Parabola`] and [`Axis2Placement3d`] are
//!   the neutral, kernel-independent representation.
//! * [`ConicWriter`] (and [`write_conics`]) emit `ISO-10303-21` text whose conic
//!   entities round-trip through [`ConicScene::parse`]; the per-entity strings are
//!   available through [`Conic::step_entity`], [`Trim::step_trimmed_curve`] and
//!   [`ConicWriter::entities`].
//! * `TRIMMED_CURVE` wrappers around conics are read as [`TrimmedConic`] with their raw
//!   parameter trims; [`validate_trim`] turns them into a [`TrimRange`] with the
//!   explicit periodicity limits described below.
//!
//! # Explicit limits
//!
//! * Only the *3D* conic definitions are read. Conics referenced through
//!   `SURFACE_CURVE`, `PCURVE` or `SEAM_CURVE` wrappers are returned as the 3D curve
//!   they wrap; no UV/parametric representation is reconstructed.
//! * `TRIMMED_CURVE` is supported for parameter trims only. A trim whose
//!   `master_representation` is `.CARTESIAN.`, or whose trim set carries no
//!   `PARAMETER_VALUE`, is rejected with [`ConicError::UnsupportedTrim`]; a trim set
//!   with more than one parameter value is rejected too (the neutral model stores one
//!   parameter per end).
//! * Ellipses are periodic with period `2π` radians (`360°`); hyperbolas and parabolas
//!   are open, single-branch curves. [`validate_trim`] makes this explicit:
//!   `ELLIPSE` trims must be increasing and span at most one period, while open conics
//!   accept any increasing parameter interval. STEP does not record whether an ellipse
//!   trim value is in radians or degrees — ISO 10303-42 parameterizes ellipses in
//!   radians, but common AP214 exporters write degrees — so the caller selects the
//!   unit through [`AngleUnit`]; the crate never guesses.
//! * Trimmed conics are not evaluated or sampled here. [`TrimRange`] is a
//!   validated description of the trim interval, not a tessellation.
//! * STEP reals that are not finite are rejected on read and on write.
//!
//! # Example
//!
//! ```
//! use step_conics::{ConicScene, ConicDomain, AngleUnit, validate_trim};
//!
//! let text = "\
//! ISO-10303-21;
//! HEADER;
//! FILE_DESCRIPTION((''),'2;1');
//! FILE_NAME('','',(''),(''),'','','');
//! FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
//! ENDSEC;
//! DATA;
//! #1=CARTESIAN_POINT('',(0.,0.,0.));
//! #2=DIRECTION('',(0.,0.,1.));
//! #3=DIRECTION('',(1.,0.,0.));
//! #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
//! #5=ELLIPSE('',#4,10.,5.);
//! #6=TRIMMED_CURVE('',#5,(PARAMETER_VALUE(0.)),(PARAMETER_VALUE(1.5)),.T.,.PARAMETER.);
//! ENDSEC;
//! END-ISO-10303-21;
//! ";
//!
//! let scene = ConicScene::parse(text)?;
//! let conic = scene.conic(5).expect("ellipse #5");
//! assert_eq!(conic.domain(), ConicDomain::PeriodicEllipse);
//! let trimmed = scene.trim_for(5).expect("TRIMMED_CURVE #6");
//! let range = validate_trim(conic, &trimmed.trim, AngleUnit::Radians)?;
//! assert_eq!(range.span, 1.5);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]

mod limits;
mod parse;
mod write;

use std::collections::BTreeMap;
use std::fmt;

use nodecraft_step::StepFile;

pub use limits::{AngleUnit, ConicDomain, ConicLimit, TrimRange, validate_trim};
pub use parse::{parse_placement, read_conics};
pub use write::{ConicWriter, write_conics};

/// The parameter period of an ellipse in radians (`2π`).
pub const ELLIPSE_PERIOD_RADIANS: f64 = std::f64::consts::TAU;
/// The parameter period of an ellipse in degrees (`360`).
pub const ELLIPSE_PERIOD_DEGREES: f64 = 360.0;

/// The plane frame an `AXIS2_PLACEMENT_3D` resolves to: an origin and a right-handed
/// orthonormal basis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The placement location.
    pub origin: [f64; 3],
    /// Unit x direction.
    pub x: [f64; 3],
    /// Unit y direction (`z × x`).
    pub y: [f64; 3],
    /// Unit z direction.
    pub z: [f64; 3],
}

/// A STEP `AXIS2_PLACEMENT_3D`: a location with an optional axis (z direction) and
/// optional reference direction (x direction).
///
/// `axis`/`ref_direction` are `None` when the file wrote `$`; the STEP defaults are
/// applied by [`Self::resolve`], while the neutral representation preserves the file's
/// choice so a write/parse round-trip is exact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis2Placement3d {
    /// The placement location (`CARTESIAN_POINT`).
    pub location: [f64; 3],
    /// The placement axis (`DIRECTION`), or `None` for the STEP default `+Z`.
    pub axis: Option<[f64; 3]>,
    /// The reference direction (`DIRECTION`), or `None` for the STEP default.
    pub ref_direction: Option<[f64; 3]>,
}

impl Axis2Placement3d {
    /// A placement at `location` with default axis and reference direction.
    #[must_use]
    pub const fn new(location: [f64; 3]) -> Self {
        Self {
            location,
            axis: None,
            ref_direction: None,
        }
    }

    /// A placement with an explicit axis and the default reference direction.
    #[must_use]
    pub const fn with_axis(location: [f64; 3], axis: [f64; 3]) -> Self {
        Self {
            location,
            axis: Some(axis),
            ref_direction: None,
        }
    }

    /// A placement with an explicit axis and reference direction.
    #[must_use]
    pub const fn with_axes(
        location: [f64; 3],
        axis: [f64; 3],
        ref_direction: [f64; 3],
    ) -> Self {
        Self {
            location,
            axis: Some(axis),
            ref_direction: Some(ref_direction),
        }
    }

    /// Resolves the STEP placement defaults into an orthonormal [`Frame`].
    ///
    /// * The default axis is `+Z`.
    /// * The default reference direction is `+X`, or `+Y` when the axis is parallel to
    ///   `+X`; an explicit reference direction is orthogonalized against the axis.
    /// * `y = z × x`, so the frame is right handed.
    ///
    /// # Errors
    ///
    /// [`PlacementError`] for non-finite values, a zero axis or reference direction, or
    /// a reference direction parallel to the axis.
    pub fn resolve(&self) -> Result<Frame, PlacementError> {
        if !self.location.iter().all(|value| value.is_finite()) {
            return Err(PlacementError::NonFinite("location"));
        }
        let z = match self.axis {
            None => [0.0, 0.0, 1.0],
            Some(axis) => {
                if !axis.iter().all(|value| value.is_finite()) {
                    return Err(PlacementError::NonFinite("axis"));
                }
                unit(axis).ok_or(PlacementError::ZeroAxis)?
            }
        };
        let reference = match self.ref_direction {
            None => {
                if z[0].abs() >= 1.0 - 1.0e-12 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                }
            }
            Some(reference) => {
                if !reference.iter().all(|value| value.is_finite()) {
                    return Err(PlacementError::NonFinite("ref_direction"));
                }
                if unit(reference).is_none() {
                    return Err(PlacementError::ZeroRefDirection);
                }
                reference
            }
        };
        let x = unit(orthogonalize(reference, z))
            .ok_or(PlacementError::RefDirectionParallelToAxis)?;
        let y = cross(z, x);
        Ok(Frame {
            origin: self.location,
            x,
            y,
            z,
        })
    }
}

/// Why an [`Axis2Placement3d`] cannot be resolved to a [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementError {
    /// One of the placement's components is not finite.
    NonFinite(&'static str),
    /// The placement axis is the zero vector.
    ZeroAxis,
    /// The reference direction is the zero vector.
    ZeroRefDirection,
    /// The reference direction has no component outside the axis.
    RefDirectionParallelToAxis,
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "placement {field} is not finite"),
            Self::ZeroAxis => write!(f, "placement axis is the zero vector"),
            Self::ZeroRefDirection => write!(f, "placement reference direction is the zero vector"),
            Self::RefDirectionParallelToAxis => {
                write!(f, "placement reference direction is parallel to the axis")
            }
        }
    }
}

impl std::error::Error for PlacementError {}

/// A STEP `ELLIPSE`: `ELLIPSE(name, position, semi_axis_1, semi_axis_2)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// The ellipse's placement: its centre, plane axis and major-axis direction.
    pub placement: Axis2Placement3d,
    /// The major semi-axis (`semi_axis_1`), in model units.
    pub semi_axis_1: f64,
    /// The minor semi-axis (`semi_axis_2`), in model units.
    pub semi_axis_2: f64,
}

/// A STEP `HYPERBOLA`: `HYPERBOLA(name, position, semi_axis, semi_imag_axis)`.
///
/// The entity is the branch in the placement's `+X` half-plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hyperbola {
    /// The hyperbola's placement: its centre, plane axis and real-axis direction.
    pub placement: Axis2Placement3d,
    /// The real semi-axis (`semi_axis`), in model units.
    pub semi_axis: f64,
    /// The imaginary semi-axis (`semi_imag_axis`), in model units.
    pub semi_imag_axis: f64,
}

/// A STEP `PARABOLA`: `PARABOLA(name, position, focal_dist)`.
///
/// The entity is the branch opening along the placement's `+X`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parabola {
    /// The parabola's placement: its vertex, plane axis and opening direction.
    pub placement: Axis2Placement3d,
    /// The focal distance (`focal_dist`), in model units.
    pub focal_distance: f64,
}

/// A neutral analytic conic curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Conic {
    /// An `ELLIPSE`.
    Ellipse(Ellipse),
    /// A `HYPERBOLA`.
    Hyperbola(Hyperbola),
    /// A `PARABOLA`.
    Parabola(Parabola),
}

impl Conic {
    /// The STEP entity name (`"ELLIPSE"`, `"HYPERBOLA"` or `"PARABOLA"`).
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Ellipse(_) => "ELLIPSE",
            Self::Hyperbola(_) => "HYPERBOLA",
            Self::Parabola(_) => "PARABOLA",
        }
    }

    /// The conic's placement.
    #[must_use]
    pub fn placement(&self) -> &Axis2Placement3d {
        match self {
            Self::Ellipse(ellipse) => &ellipse.placement,
            Self::Hyperbola(hyperbola) => &hyperbola.placement,
            Self::Parabola(parabola) => &parabola.placement,
        }
    }

    /// The curve's parameter domain: periodic for an ellipse, an open single branch for
    /// a hyperbola or parabola.
    #[must_use]
    pub const fn domain(&self) -> ConicDomain {
        match self {
            Self::Ellipse(_) => ConicDomain::PeriodicEllipse,
            Self::Hyperbola(_) => ConicDomain::HyperbolaBranch,
            Self::Parabola(_) => ConicDomain::ParabolaBranch,
        }
    }


    /// Evaluates the conic at parameter `t` and returns a model-space point.
    ///
    /// Ellipses use `(a cos t, b sin t)`; hyperbolas use `(a cosh t, b sinh t)`;
    /// parabolas use `(t²/(4f), t)` on the placement's `+X` branch.
    pub fn point_at(&self, t: f64) -> Result<[f64; 3], ConicError> {
        if !t.is_finite() {
            return Err(ConicError::NonFinite { id: 0, entity: self.name().into(), field: "parameter".into(), value: t });
        }
        let frame = self.placement().resolve().map_err(|error| ConicError::InvalidGeometry { entity: self.name(), why: format!("placement: {error:?}") })?;
        let (x, y) = match self {
            Self::Ellipse(ellipse) => (ellipse.semi_axis_1 * t.cos(), ellipse.semi_axis_2 * t.sin()),
            Self::Hyperbola(hyperbola) => (hyperbola.semi_axis * t.cosh(), hyperbola.semi_imag_axis * t.sinh()),
            Self::Parabola(parabola) => { let focal = parabola.focal_distance; (t * t / (4.0 * focal), t) }
        };
        Ok([
            frame.origin[0] + frame.x[0]*x + frame.y[0]*y,
            frame.origin[1] + frame.x[1]*x + frame.y[1]*y,
            frame.origin[2] + frame.x[2]*x + frame.y[2]*y,
        ])
    }


    /// Samples the conic over a validated trim range (inclusive endpoints).
    pub fn sample(&self, range: &TrimRange, segments: usize) -> Result<Vec<[f64; 3]>, ConicError> {
        if segments == 0 { return Err(ConicError::InvalidGeometry { entity: self.name(), why: "sample count must be positive".into() }); }
        if !range.start.is_finite() || !range.end.is_finite() {
            return Err(ConicError::InvalidGeometry { entity: self.name(), why: "trim range is not finite".into() });
        }
        let mut points = Vec::with_capacity(segments + 1);
        for index in 0..=segments {
            let t = range.start + (range.end - range.start) * index as f64 / segments as f64;
            points.push(self.point_at(t)?);
        }
        Ok(points)
    }

    /// The STEP entity body for this conic, without the leading `#id=`, referencing
    /// `placement_id`.
    ///
    /// # Errors
    ///
    /// [`ConicError::InvalidGeometry`] for a non-finite or non-positive axis/parameter
    /// value (the same checks the writer applies).
    pub fn step_entity(&self, placement_id: u32) -> Result<String, ConicError> {
        match self {
            Self::Ellipse(ellipse) => {
                validate_positive("ELLIPSE", "semi_axis_1", ellipse.semi_axis_1)?;
                validate_positive("ELLIPSE", "semi_axis_2", ellipse.semi_axis_2)?;
                Ok(format!(
                    "ELLIPSE('',#{placement_id},{},{})",
                    write::real(ellipse.semi_axis_1),
                    write::real(ellipse.semi_axis_2)
                ))
            }
            Self::Hyperbola(hyperbola) => {
                validate_positive("HYPERBOLA", "semi_axis", hyperbola.semi_axis)?;
                validate_positive("HYPERBOLA", "semi_imag_axis", hyperbola.semi_imag_axis)?;
                Ok(format!(
                    "HYPERBOLA('',#{placement_id},{},{})",
                    write::real(hyperbola.semi_axis),
                    write::real(hyperbola.semi_imag_axis)
                ))
            }
            Self::Parabola(parabola) => {
                validate_positive("PARABOLA", "focal_dist", parabola.focal_distance)?;
                Ok(format!(
                    "PARABOLA('',#{placement_id},{})",
                    write::real(parabola.focal_distance)
                ))
            }
        }
    }
}

/// A parameter trim of a conic, as read from a `TRIMMED_CURVE` (`trim_1`, `trim_2`,
/// `sense_agreement`).
///
/// The values are the file's raw parameter values; see [`validate_trim`] for the
/// explicit periodicity/ordering limits and the unit choice for ellipses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trim {
    /// The first trim parameter (`trim_1`).
    pub trim_1: f64,
    /// The second trim parameter (`trim_2`).
    pub trim_2: f64,
    /// `sense_agreement`: the trimmed curve runs in the basis curve's direction.
    pub sense_agreement: bool,
}

impl Trim {
    /// A trim with `sense_agreement = true`.
    #[must_use]
    pub const fn new(trim_1: f64, trim_2: f64) -> Self {
        Self {
            trim_1,
            trim_2,
            sense_agreement: true,
        }
    }

    /// A trim with an explicit `sense_agreement`.
    #[must_use]
    pub const fn with_sense(trim_1: f64, trim_2: f64, sense_agreement: bool) -> Self {
        Self {
            trim_1,
            trim_2,
            sense_agreement,
        }
    }

    /// The STEP `TRIMMED_CURVE` entity body, without the leading `#id=`, referencing
    /// `basis_id`.
    ///
    /// Both trim ends are written as `(PARAMETER_VALUE(...))` sets; the master
    /// representation is always `.PARAMETER.` because the neutral model has no cartesian
    /// trim points.
    ///
    /// # Errors
    ///
    /// [`ConicError::InvalidGeometry`] for a non-finite trim parameter.
    pub fn step_trimmed_curve(&self, basis_id: u32) -> Result<String, ConicError> {
        validate_finite("TRIMMED_CURVE", "trim_1", self.trim_1)?;
        validate_finite("TRIMMED_CURVE", "trim_2", self.trim_2)?;
        Ok(format!(
            "TRIMMED_CURVE('',#{basis_id},(PARAMETER_VALUE({})),(PARAMETER_VALUE({})),{},.PARAMETER.)",
            write::real(self.trim_1),
            write::real(self.trim_2),
            if self.sense_agreement { ".T." } else { ".F." }
        ))
    }
}

/// The `master_representation` of a `TRIMMED_CURVE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterRepresentation {
    /// `.PARAMETER.`: the trims are curve parameters.
    Parameter,
    /// `.CARTESIAN.`: the trims are cartesian points (unsupported by this crate).
    Cartesian,
    /// `.UNSPECIFIED.`: the file gives both parameter and cartesian trims.
    Unspecified,
}

/// A `TRIMMED_CURVE` whose basis is a conic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimmedConic {
    /// The `TRIMMED_CURVE` entity id.
    pub id: u32,
    /// The id of the basis conic entity.
    pub basis: u32,
    /// The raw parameter trim.
    pub trim: Trim,
    /// The trim's master representation (`Parameter` or `Unspecified` in this crate).
    pub master: MasterRepresentation,
}

/// Everything this crate reads from a STEP file: the conics, the placements they
/// reference, and the conic-basis `TRIMMED_CURVE`s.
///
/// All maps are keyed by STEP entity id and ordered by id, so iteration is
/// deterministic.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConicScene {
    /// Placements referenced by the parsed conics, keyed by placement entity id.
    pub placements: BTreeMap<u32, Axis2Placement3d>,
    /// The parsed conics, keyed by entity id.
    pub conics: BTreeMap<u32, Conic>,
    /// Conic-basis `TRIMMED_CURVE`s, keyed by `TRIMMED_CURVE` entity id.
    pub trimmed_curves: BTreeMap<u32, TrimmedConic>,
}

impl ConicScene {
    /// Parses `text` with [`nodecraft_step::StepFile::parse`] and reads its conics.
    ///
    /// # Errors
    ///
    /// [`ConicError::Step`] when the STEP text does not parse; the errors of
    /// [`read_conics`] otherwise.
    pub fn parse(text: &str) -> Result<Self, ConicError> {
        let file = StepFile::parse(text).map_err(|message| ConicError::Step { message })?;
        read_conics(&file)
    }

    /// The conic with entity id `id`.
    #[must_use]
    pub fn conic(&self, id: u32) -> Option<&Conic> {
        self.conics.get(&id)
    }

    /// The first parsed `TRIMMED_CURVE` on conic `conic_id`, in entity-id order.
    #[must_use]
    pub fn trim_for(&self, conic_id: u32) -> Option<&TrimmedConic> {
        self.trimmed_curves
            .values()
            .find(|trimmed| trimmed.basis == conic_id)
    }
}

/// Errors of reading and writing conic STEP data.
#[derive(Debug, Clone, PartialEq)]
pub enum ConicError {
    /// The underlying [`nodecraft_step::StepFile::parse`] failed.
    Step {
        /// The parser's message.
        message: String,
    },
    /// A referenced entity id is missing from the file.
    MissingEntity {
        /// The missing id.
        id: u32,
    },
    /// An entity has the wrong type.
    WrongEntity {
        /// The entity id.
        id: u32,
        /// The type the caller expected.
        expected: &'static str,
        /// The type the file has.
        found: String,
    },
    /// An entity has the wrong number of arguments.
    BadArgumentCount {
        /// The entity id.
        id: u32,
        /// The entity type.
        entity: String,
        /// The expected argument count.
        expected: usize,
        /// The actual argument count.
        found: usize,
    },
    /// An entity argument has the wrong shape or value.
    BadValue {
        /// The entity id.
        id: u32,
        /// The entity type.
        entity: String,
        /// What is wrong with the argument.
        why: String,
    },
    /// An `AXIS2_PLACEMENT_3D` cannot be resolved to a frame.
    InvalidPlacement {
        /// The placement entity id.
        id: u32,
        /// The resolution error.
        why: String,
    },
    /// A conic axis or parameter value is not positive.
    NonPositive {
        /// The entity id.
        id: u32,
        /// The entity type.
        entity: String,
        /// The STEP argument name.
        field: String,
        /// The offending value.
        value: f64,
    },
    /// A conic axis, placement component or trim value is not finite.
    NonFinite {
        /// The entity id (`0` for a value supplied to the writer).
        id: u32,
        /// The entity type.
        entity: String,
        /// The STEP argument name.
        field: String,
        /// The offending value.
        value: f64,
    },
    /// A `TRIMMED_CURVE` trim parameter is not finite.
    NonFiniteTrim {
        /// The `TRIMMED_CURVE` entity id.
        id: u32,
        /// The offending parameter.
        value: f64,
    },
    /// A `TRIMMED_CURVE` uses a trim representation this crate does not model.
    UnsupportedTrim {
        /// The `TRIMMED_CURVE` entity id.
        id: u32,
        /// What is unsupported.
        why: String,
    },
    /// A value supplied to the writer is not usable geometry.
    InvalidGeometry {
        /// The STEP entity type the value belongs to.
        entity: &'static str,
        /// What is wrong with the value.
        why: String,
    },
}

impl fmt::Display for ConicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step { message } => write!(f, "STEP parse error: {message}"),
            Self::MissingEntity { id } => write!(f, "STEP entity #{id} is referenced but missing"),
            Self::WrongEntity {
                id,
                expected,
                found,
            } => write!(f, "STEP entity #{id} is {found}, expected {expected}"),
            Self::BadArgumentCount {
                id,
                entity,
                expected,
                found,
            } => write!(
                f,
                "STEP entity #{id} ({entity}) has {found} arguments, expected {expected}"
            ),
            Self::BadValue { id, entity, why } => {
                write!(f, "STEP entity #{id} ({entity}): {why}")
            }
            Self::InvalidPlacement { id, why } => {
                write!(f, "STEP entity #{id} (AXIS2_PLACEMENT_3D): {why}")
            }
            Self::NonPositive {
                id,
                entity,
                field,
                value,
            } => write!(
                f,
                "STEP entity #{id} ({entity}) has non-positive {field} {value}"
            ),
            Self::NonFinite {
                id,
                entity,
                field,
                value,
            } => write!(f, "STEP entity #{id} ({entity}) has non-finite {field} {value}"),
            Self::NonFiniteTrim { id, value } => {
                write!(f, "TRIMMED_CURVE #{id} has non-finite trim parameter {value}")
            }
            Self::UnsupportedTrim { id, why } => {
                write!(f, "TRIMMED_CURVE #{id} is not supported: {why}")
            }
            Self::InvalidGeometry { entity, why } => {
                write!(f, "{entity} cannot be written: {why}")
            }
        }
    }
}

impl std::error::Error for ConicError {}

pub(crate) fn validate_positive(
    entity: &'static str,
    field: &str,
    value: f64,
) -> Result<(), ConicError> {
    if !value.is_finite() {
        return Err(ConicError::InvalidGeometry {
            entity,
            why: format!("{field} must be finite, got {value}"),
        });
    }
    if value <= 0.0 {
        return Err(ConicError::InvalidGeometry {
            entity,
            why: format!("{field} must be positive, got {value}"),
        });
    }
    Ok(())
}

pub(crate) fn validate_finite(
    entity: &'static str,
    field: &str,
    value: f64,
) -> Result<(), ConicError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConicError::InvalidGeometry {
            entity,
            why: format!("{field} must be finite, got {value}"),
        })
    }
}

fn unit(vector: [f64; 3]) -> Option<[f64; 3]> {
    if !vector.iter().all(|value| value.is_finite()) {
        return None;
    }
    let length = dot(vector, vector).sqrt();
    if length == 0.0 {
        return None;
    }
    Some(vector.map(|value| value / length))
}

fn orthogonalize(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let projection = dot(vector, axis);
    [
        vector[0] - axis[0] * projection,
        vector[1] - axis[1] * projection,
        vector[2] - axis[2] * projection,
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]

}
