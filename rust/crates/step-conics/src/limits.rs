//! Explicit periodicity and trim limits.
//!
//! `ELLIPSE` is periodic; `HYPERBOLA` and `PARABOLA` are open single branches. A
//! `TRIMMED_CURVE` therefore has two very different meanings, and STEP records neither
//! the ellipse parameter unit nor a seam rule. This module makes the resulting limits
//! explicit instead of guessing.

use std::fmt;

use crate::{Conic, Trim, ELLIPSE_PERIOD_DEGREES, ELLIPSE_PERIOD_RADIANS};

/// The parameter domain of a conic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConicDomain {
    /// The ellipse closes after one period (`2π` radians / `360°`).
    PeriodicEllipse,
    /// One branch of a hyperbola: open at both ends, no periodic seam.
    HyperbolaBranch,
    /// One branch of a parabola: open at both ends, no periodic seam.
    ParabolaBranch,
}

impl ConicDomain {
    /// The period of a periodic domain in `units`, or `None` for an open branch.
    #[must_use]
    pub const fn period(self, units: AngleUnit) -> Option<f64> {
        match self {
            Self::PeriodicEllipse => Some(units.ellipse_period()),
            Self::HyperbolaBranch | Self::ParabolaBranch => None,
        }
    }
}

/// The unit of an ellipse trim parameter.
///
/// ISO 10303-42 parameterizes the ellipse in radians, but common AP214 exporters write
/// circle/ellipse trims in degrees (for example a full circle `0.` .. `360.`). STEP does
/// not say which convention a file uses, so the caller chooses when validating a trim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleUnit {
    /// Radians: the ellipse period is `2π`.
    Radians,
    /// Degrees: the ellipse period is `360`.
    Degrees,
}

impl AngleUnit {
    /// The ellipse period in this unit.
    #[must_use]
    pub const fn ellipse_period(self) -> f64 {
        match self {
            Self::Radians => ELLIPSE_PERIOD_RADIANS,
            Self::Degrees => ELLIPSE_PERIOD_DEGREES,
        }
    }
}

/// A validated trim interval: the raw start/end parameters, their span in the file's
/// units, and whether the interval covers the whole ellipse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimRange {
    /// The first trim parameter (`trim_1`).
    pub start: f64,
    /// The second trim parameter (`trim_2`).
    pub end: f64,
    /// `end - start`.
    pub span: f64,
    /// `true` when `span` is exactly one ellipse period (the full periodic curve).
    pub full_period: bool,
}

/// A trim the crate refuses to interpret.
#[derive(Debug, Clone, PartialEq)]
pub enum ConicLimit {
    /// A trim parameter is not finite.
    NonFiniteTrim {
        /// The offending parameter.
        value: f64,
    },
    /// `trim_1 == trim_2`: the interval has no length.
    EmptyTrim,
    /// `trim_2 < trim_1`: reversed parameter intervals are not supported; STEP's
    /// `sense_agreement` describes the trimmed curve's direction instead.
    TrimReversed {
        /// The first trim parameter.
        trim_1: f64,
        /// The second trim parameter.
        trim_2: f64,
    },
    /// A periodic ellipse trim spans more than one period.
    TrimSpanExceedsPeriod {
        /// `trim_2 - trim_1`.
        span: f64,
        /// The period chosen by the caller's [`AngleUnit`].
        period: f64,
    },
}

impl fmt::Display for ConicLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteTrim { value } => {
                write!(f, "trim parameter is not finite: {value}")
            }
            Self::EmptyTrim => write!(f, "trim interval has zero length"),
            Self::TrimReversed { trim_1, trim_2 } => write!(
                f,
                "trim interval is reversed ({trim_1} > {trim_2}); use sense_agreement for direction"
            ),
            Self::TrimSpanExceedsPeriod { span, period } => write!(
                f,
                "ellipse trim spans {span}, more than one period ({period})"
            ),
        }
    }
}

impl std::error::Error for ConicLimit {}

/// Validates a conic trim against the conic's [`ConicDomain`].
///
/// * Both parameters must be finite, increasing (`trim_2 > trim_1`) and non-empty.
/// * For a periodic `ELLIPSE`, the span may be at most one period in `units`; a span of
///   exactly one period is the full ellipse and is reported as
///   [`TrimRange::full_period`]. Trims beyond one period or across the seam are not
///   normalized: they are rejected, because STEP gives no rule for which period a seam
///   crossing belongs to.
/// * For an open `HYPERBOLA`/`PARABOLA` branch, any increasing interval is accepted; the
///   branch itself is unbounded.
///
/// # Errors
///
/// [`ConicLimit`] when the trim cannot be interpreted under these rules.
pub fn validate_trim(
    conic: &Conic,
    trim: &Trim,
    units: AngleUnit,
) -> Result<TrimRange, ConicLimit> {
    if !trim.trim_1.is_finite() {
        return Err(ConicLimit::NonFiniteTrim {
            value: trim.trim_1,
        });
    }
    if !trim.trim_2.is_finite() {
        return Err(ConicLimit::NonFiniteTrim {
            value: trim.trim_2,
        });
    }
    if trim.trim_2 < trim.trim_1 {
        return Err(ConicLimit::TrimReversed {
            trim_1: trim.trim_1,
            trim_2: trim.trim_2,
        });
    }
    if trim.trim_2 == trim.trim_1 {
        return Err(ConicLimit::EmptyTrim);
    }
    let span = trim.trim_2 - trim.trim_1;
    match conic.domain() {
        ConicDomain::PeriodicEllipse => {
            let period = units.ellipse_period();
            if span > period {
                return Err(ConicLimit::TrimSpanExceedsPeriod { span, period });
            }
            Ok(TrimRange {
                start: trim.trim_1,
                end: trim.trim_2,
                span,
                full_period: span == period,
            })
        }
        ConicDomain::HyperbolaBranch | ConicDomain::ParabolaBranch => Ok(TrimRange {
            start: trim.trim_1,
            end: trim.trim_2,
            span,
            full_period: false,
        }),
    }
}
