//! Knot vectors: explicit STEP `(multiplicities, values)` pairs, default vectors and
//! validation against a degree / control-point count.

use nodecraft_step::Value;

/// The STEP `knot_spec` enumeration of a B-spline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KnotSpec {
    /// `.UNSPECIFIED.` — the file does not say how the knots are spaced.
    #[default]
    Unspecified,
    /// `.UNIFORM.` — knots with multiplicity 1, equally spaced.
    Uniform,
    /// `.QUASI_UNIFORM.` — clamped: the ends have multiplicity `degree + 1`, the
    /// interior knots are equally spaced.
    QuasiUniform,
    /// `.PIECEWISE_BEZIER.` — interior knots have multiplicity `degree`.
    PiecewiseBezier,
}

impl KnotSpec {
    /// Parse a STEP enumeration value (`.UNSPECIFIED.`, `.T.`, ... are stored by the
    /// generic parser as [`Value::Enum`] without the dots).
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Enum(name) => {
                // Some exporters spell the knot spec with a `_KNOTS` suffix
                // (`.QUASI_UNIFORM_KNOTS.`), notably the stress fixtures.
                let normalized = name.strip_suffix("_KNOTS").unwrap_or(name);
                match normalized {
                    "UNSPECIFIED" => Some(Self::Unspecified),
                    "UNIFORM" => Some(Self::Uniform),
                    "QUASI_UNIFORM" => Some(Self::QuasiUniform),
                    "PIECEWISE_BEZIER" => Some(Self::PiecewiseBezier),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The STEP spelling including dots, e.g. `.QUASI_UNIFORM.`.
    pub fn as_step(self) -> &'static str {
        match self {
            Self::Unspecified => ".UNSPECIFIED.",
            Self::Uniform => ".UNIFORM.",
            Self::QuasiUniform => ".QUASI_UNIFORM.",
            Self::PiecewiseBezier => ".PIECEWISE_BEZIER.",
        }
    }
}

/// A STEP knot vector: distinct knot values plus their multiplicities.
///
/// STEP stores a knot vector as two parallel lists, `knot_multiplicities` and `knots`
/// (distinct values); the evaluation form ([`Self::expanded`]) repeats every value.
#[derive(Debug, Clone, PartialEq)]
pub struct Knots {
    /// Multiplicity of each distinct knot value, in ascending value order.
    pub multiplicities: Vec<usize>,
    /// Distinct, strictly increasing knot values.
    pub values: Vec<f64>,
    /// The `knot_spec` enumeration that came with the entity (or the one used when the
    /// knots were derived).
    pub spec: KnotSpec,
    /// `true` when the file had no knot data and these knots were generated from
    /// `spec`, degree and control-point count.
    pub derived: bool,
}

impl Knots {
    /// A knot vector read from a file.
    pub fn explicit(
        multiplicities: &[usize],
        values: &[f64],
        spec: KnotSpec,
    ) -> Result<Self, String> {
        if multiplicities.len() != values.len() {
            return Err(format!(
                "knot multiplicities ({}) and knot values ({}) differ in length",
                multiplicities.len(),
                values.len()
            ));
        }
        if values.is_empty() {
            return Err("knot vector is empty".into());
        }
        if multiplicities.iter().any(|&m| m == 0) {
            return Err("knot multiplicities must be positive".into());
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("knot values must be finite".into());
        }
        if values.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("knot values must be strictly increasing".into());
        }
        Ok(Self {
            multiplicities: multiplicities.to_vec(),
            values: values.to_vec(),
            spec,
            derived: false,
        })
    }

    /// Expand the knot vector used by the evaluation algorithms.
    pub fn expanded(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(
            self.multiplicities.iter().sum::<usize>(),
        );
        for (value, &multiplicity) in self.values.iter().zip(&self.multiplicities) {
            out.extend(std::iter::repeat_n(*value, multiplicity));
        }
        out
    }

    /// Number of knots in the expanded vector.
    pub fn len(&self) -> usize {
        self.multiplicities.iter().sum()
    }

    /// `true` when the vector has no knots (never the case for a valid B-spline).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clamped uniform knots (`QUASI_UNIFORM`): ends multiplicity `degree + 1`,
    /// interior multiplicities 1. The parameter domain becomes `[0, 1]`.
    pub fn quasi_uniform(degree: usize, control_count: usize) -> Result<Self, String> {
        validate_degree_and_controls(degree, control_count)?;
        let interior = control_count - degree - 1;
        let spans = interior + 1;
        let values: Vec<f64> = (0..=spans).map(|index| index as f64 / spans as f64).collect();
        let mut multiplicities = Vec::with_capacity(values.len());
        multiplicities.push(degree + 1);
        multiplicities.extend(std::iter::repeat_n(1, interior));
        multiplicities.push(degree + 1);
        Self::explicit(&multiplicities, &values, KnotSpec::QuasiUniform).map(|mut knots| {
            knots.derived = true;
            knots
        })
    }

    /// Uniform knots (`.UNIFORM.`): every multiplicity is 1 and the values are the
    /// integers `0 .. control_count + degree`, so the curve is not clamped at its ends.
    pub fn uniform(degree: usize, control_count: usize) -> Result<Self, String> {
        validate_degree_and_controls(degree, control_count)?;
        let count = control_count + degree + 1;
        let values: Vec<f64> = (0..count).map(|index| index as f64).collect();
        let multiplicities = vec![1; count];
        Self::explicit(&multiplicities, &values, KnotSpec::Uniform).map(|mut knots| {
            knots.derived = true;
            knots
        })
    }

    /// Piecewise Bézier knots (`.PIECEWISE_BEZIER.`): interior multiplicity `degree`.
    ///
    /// The control-point count must satisfy `(control_count - 1) % degree == 0`, i.e.
    /// it must describe a whole number of Bézier segments.
    pub fn piecewise_bezier(degree: usize, control_count: usize) -> Result<Self, String> {
        validate_degree_and_controls(degree, control_count)?;
        if (control_count - 1) % degree != 0 {
            return Err(format!(
                "piecewise Bézier degree {degree} needs (control_count - 1) divisible by \
                 degree, got {control_count} control points"
            ));
        }
        let segments = (control_count - 1) / degree;
        let values: Vec<f64> = (0..=segments).map(|index| index as f64).collect();
        let mut multiplicities = Vec::with_capacity(values.len());
        multiplicities.push(degree + 1);
        multiplicities.extend(std::iter::repeat_n(degree, segments - 1));
        multiplicities.push(degree + 1);
        Self::explicit(&multiplicities, &values, KnotSpec::PiecewiseBezier).map(|mut knots| {
            knots.derived = true;
            knots
        })
    }

    /// Derive the default knots for a file that omits them.
    ///
    /// `.UNSPECIFIED.` falls back to clamped (`QUASI_UNIFORM`) knots, the representation
    /// every common writer means when it leaves the knot values out.
    pub fn default_for(
        degree: usize,
        control_count: usize,
        spec: KnotSpec,
    ) -> Result<Self, String> {
        match spec {
            KnotSpec::Uniform => Self::uniform(degree, control_count),
            KnotSpec::PiecewiseBezier => Self::piecewise_bezier(degree, control_count),
            KnotSpec::QuasiUniform | KnotSpec::Unspecified => {
                Self::quasi_uniform(degree, control_count)
            }
        }
        .map(|mut knots| {
            knots.derived = true;
            knots
        })
    }

    /// Check that this knot vector is usable with `degree` and `control_count` poles.
    pub fn validate_for(&self, degree: usize, control_count: usize) -> Result<(), String> {
        validate_degree_and_controls(degree, control_count)?;
        if self.multiplicities.len() != self.values.len() {
            return Err(format!(
                "knot multiplicities ({}) and knot values ({}) differ in length",
                self.multiplicities.len(),
                self.values.len()
            ));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err("knot values must be finite".into());
        }
        if self.values.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("knot values must be strictly increasing".into());
        }
        if let Some(index) = self.multiplicities.iter().position(|&m| m == 0) {
            return Err(format!("knot {index} has multiplicity 0"));
        }
        let expected = control_count + degree + 1;
        let actual = self.len();
        if actual != expected {
            return Err(format!(
                "knot vector has {actual} entries but degree {degree} with {control_count} \
                 control points needs {expected}"
            ));
        }
        if let Some(&multiplicity) = self.multiplicities.iter().find(|&&m| m > degree + 1) {
            return Err(format!(
                "knot multiplicity {multiplicity} exceeds degree + 1 ({})",
                degree + 1
            ));
        }
        Ok(())
    }

    /// The parameter domain `[U[degree], U[last - degree]]` of the expanded vector.
    pub fn domain(&self, degree: usize) -> (f64, f64) {
        // The vector is validated at parse/emit time, so fall back to the value range
        // when it is too short to be indexed by the degree.
        let expanded = self.expanded();
        if expanded.len() > degree {
            let lo = expanded[degree];
            let hi = expanded[expanded.len() - 1 - degree];
            (lo, hi)
        } else {
            (0.0, 0.0)
        }
    }
}

fn validate_degree_and_controls(degree: usize, control_count: usize) -> Result<(), String> {
    if degree == 0 {
        return Err("B-spline degree must be at least 1".into());
    }
    if control_count < degree + 1 {
        return Err(format!(
            "degree {degree} needs at least {} control points, got {control_count}",
            degree + 1
        ));
    }
    Ok(())
}
