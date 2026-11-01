//! Emit STEP entity strings for polynomial and rational B-splines.
//!
//! The helpers only produce the entity body (no `#id=` prefix and no trailing
//! semicolon), so a caller can append it to any writer:
//!
//! ```no_run
//! use step_nurbs::{bspline_curve_entity, KnotSpec, Knots};
//!
//! let knots = Knots::explicit(&[4, 4], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
//! let body = bspline_curve_entity(3, &[1, 2, 3, 4], &knots, None, false, false).unwrap();
//! let line = format!("#100={body};");
//! ```
//!
//! Rational entities are written as STEP complex instances, matching the layout
//! smlibpy/Truck-style writers use (`BOUNDED_CURVE() B_SPLINE_CURVE(...)
//! B_SPLINE_CURVE_WITH_KNOTS(...) CURVE() ... RATIONAL_B_SPLINE_CURVE(...)`).

use crate::{Knots, eval::knot_domain};

/// A STEP REAL literal; rejects non-finite values.
///
/// The output always contains a decimal point and uses an upper-case `E` exponent,
/// e.g. `0.`, `1.0`, `0.5`, `1.E-7`.
pub fn format_real(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err(format!("cannot write non-finite STEP real {value}"));
    }
    if value == 0.0 {
        return Ok("0.".into());
    }
    let literal = format!("{value:?}");
    if let Some(position) = literal.find(['e', 'E']) {
        let (mantissa, exponent) = literal.split_at(position);
        let mantissa = if mantissa.contains('.') {
            mantissa.to_string()
        } else {
            format!("{mantissa}.")
        };
        let exponent: i32 = exponent[1..]
            .parse()
            .map_err(|_| format!("cannot format exponent in {literal}"))?;
        Ok(format!("{mantissa}E{exponent}"))
    } else if literal.contains('.') {
        Ok(literal)
    } else {
        Ok(format!("{literal}."))
    }
}

/// `(1.,0.5,0.)`
pub fn format_real_list(values: &[f64]) -> Result<String, String> {
    let mut out = String::from("(");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format_real(*value)?);
    }
    out.push(')');
    Ok(out)
}

/// `(4,1,4)`
pub fn format_int_list(values: &[usize]) -> String {
    format!(
        "({})",
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// `(#1,#2,#3)`
pub fn format_ref_list(refs: &[u32]) -> String {
    format!(
        "({})",
        refs.iter()
            .map(|id| format!("#{id}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// `((#1,#2),(#3,#4))`
pub fn format_ref_grid(rows: &[Vec<u32>]) -> String {
    format!(
        "({})",
        rows.iter()
            .map(|row| format_ref_list(row))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// `((1.,1.),(1.,1.))`
pub fn format_real_grid(rows: &[Vec<f64>]) -> Result<String, String> {
    let mut out = String::from("(");
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format_real_list(row)?);
    }
    out.push(')');
    Ok(out)
}

fn step_boolean(value: bool) -> &'static str {
    if value { ".T." } else { ".F." }
}

/// Entity body for a polynomial (`B_SPLINE_CURVE_WITH_KNOTS`) or rational (complex
/// instance with `RATIONAL_B_SPLINE_CURVE`) curve.
///
/// `pole_refs` are the `CARTESIAN_POINT` entity ids, `weights` is `Some` only for a
/// rational curve and must match `pole_refs` in length.
pub fn bspline_curve_entity(
    degree: usize,
    pole_refs: &[u32],
    knots: &Knots,
    weights: Option<&[f64]>,
    closed: bool,
    self_intersect: bool,
) -> Result<String, String> {
    if degree == 0 {
        return Err("B-spline degree must be at least 1".into());
    }
    if pole_refs.is_empty() {
        return Err("a B-spline curve needs at least one control point".into());
    }
    knots.validate_for(degree, pole_refs.len())?;
    let domain = knot_domain(&knots.expanded(), degree);
    if domain.0 >= domain.1 {
        return Err("B-spline curve has an empty parameter domain".into());
    }
    let poles = format_ref_list(pole_refs);
    let multiplicities = format_int_list(&knots.multiplicities);
    let values = format_real_list(&knots.values)?;
    let spec = knots.spec.as_step();
    match weights {
        None => Ok(format!(
            "B_SPLINE_CURVE_WITH_KNOTS('',{degree},{poles},.UNSPECIFIED.,{},{},{multiplicities},{values},{spec})",
            step_boolean(closed),
            step_boolean(self_intersect)
        )),
        Some(weights) => {
            validate_weights(weights, pole_refs.len())?;
            let weights = format_real_list(weights)?;
            Ok(format!(
                "(BOUNDED_CURVE() B_SPLINE_CURVE({degree},{poles},.UNSPECIFIED.,{},{}) \
                 B_SPLINE_CURVE_WITH_KNOTS({multiplicities},{values},{spec}) CURVE() \
                 GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE({weights}) \
                 REPRESENTATION_ITEM(''))",
                step_boolean(closed),
                step_boolean(self_intersect)
            ))
        }
    }
}

/// Entity body for a polynomial (`B_SPLINE_SURFACE_WITH_KNOTS`) or rational (complex
/// instance with `RATIONAL_B_SPLINE_SURFACE`) surface.
///
/// `pole_refs` is indexed `[u][v]`; `weights` is `Some` only for a rational surface and
/// must match the control-net shape.
#[allow(clippy::too_many_arguments)]
pub fn bspline_surface_entity(
    u_degree: usize,
    v_degree: usize,
    pole_refs: &[Vec<u32>],
    u_knots: &Knots,
    v_knots: &Knots,
    weights: Option<&[Vec<f64>]>,
    u_closed: bool,
    v_closed: bool,
    self_intersect: bool,
) -> Result<String, String> {
    if u_degree == 0 || v_degree == 0 {
        return Err("B-spline degrees must be at least 1".into());
    }
    if pole_refs.is_empty() {
        return Err("a B-spline surface needs at least one control row".into());
    }
    let v_count = pole_refs[0].len();
    if v_count == 0 {
        return Err("a B-spline surface needs at least one control column".into());
    }
    if pole_refs.iter().any(|row| row.len() != v_count) {
        return Err("control-point grid is not rectangular".into());
    }
    let u_count = pole_refs.len();
    u_knots.validate_for(u_degree, u_count)?;
    v_knots.validate_for(v_degree, v_count)?;
    if knot_domain(&u_knots.expanded(), u_degree).0
        >= knot_domain(&u_knots.expanded(), u_degree).1
    {
        return Err("B-spline surface has an empty u domain".into());
    }
    if knot_domain(&v_knots.expanded(), v_degree).0
        >= knot_domain(&v_knots.expanded(), v_degree).1
    {
        return Err("B-spline surface has an empty v domain".into());
    }
    let poles = format_ref_grid(pole_refs);
    let u_multiplicities = format_int_list(&u_knots.multiplicities);
    let v_multiplicities = format_int_list(&v_knots.multiplicities);
    let u_values = format_real_list(&u_knots.values)?;
    let v_values = format_real_list(&v_knots.values)?;
    let spec = u_knots.spec.as_step();
    match weights {
        None => Ok(format!(
            "B_SPLINE_SURFACE_WITH_KNOTS('',{u_degree},{v_degree},{poles},.UNSPECIFIED.,{},{},{},{u_multiplicities},{v_multiplicities},{u_values},{v_values},{spec})",
            step_boolean(u_closed),
            step_boolean(v_closed),
            step_boolean(self_intersect)
        )),
        Some(weights) => {
            if weights.len() != u_count || weights.iter().any(|row| row.len() != v_count) {
                return Err(format!(
                    "weight grid is {}x{}, expected {u_count}x{v_count}",
                    weights.len(),
                    weights.first().map_or(0, Vec::len)
                ));
            }
            let flat: Vec<f64> = weights.iter().flatten().copied().collect();
            validate_weights(&flat, u_count * v_count)?;
            let weights = format_real_grid(weights)?;
            Ok(format!(
                "(BOUNDED_SURFACE() B_SPLINE_SURFACE({u_degree},{v_degree},{poles},.UNSPECIFIED.,{},{},{}) \
                 B_SPLINE_SURFACE_WITH_KNOTS({u_multiplicities},{v_multiplicities},{u_values},{v_values},{spec}) \
                 GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({weights}) \
                 REPRESENTATION_ITEM('') SURFACE())",
                step_boolean(u_closed),
                step_boolean(v_closed),
                step_boolean(self_intersect)
            ))
        }
    }
}

fn validate_weights(weights: &[f64], expected: usize) -> Result<(), String> {
    if weights.len() != expected {
        return Err(format!(
            "{} weights for {expected} control points",
            weights.len()
        ));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err("weights must be finite".into());
    }
    if let Some(index) = weights.iter().position(|&weight| weight <= 0.0) {
        return Err(format!("weight {index} is not positive ({})", weights[index]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KnotSpec;

    #[test]
    fn formats_step_reals() {
        assert_eq!(format_real(0.0).unwrap(), "0.");
        assert_eq!(format_real(2.0).unwrap(), "2.0");
        assert_eq!(format_real(0.5).unwrap(), "0.5");
        assert_eq!(format_real(-0.5).unwrap(), "-0.5");
        assert_eq!(format_real(1e-7).unwrap(), "1.E-7");
        assert_eq!(format_real(1.5e-7).unwrap(), "1.5E-7");
        assert!(format_real(f64::NAN).is_err());
        assert!(format_real(f64::INFINITY).is_err());
    }

    #[test]
    fn formats_lists() {
        assert_eq!(format_int_list(&[4, 1, 4]), "(4,1,4)");
        assert_eq!(format_ref_list(&[1, 2]), "(#1,#2)");
        assert_eq!(
            format_real_list(&[0.0, 0.5, 1.0]).unwrap(),
            "(0.,0.5,1.0)"
        );
        assert_eq!(
            format_ref_grid(&[vec![1, 2], vec![3, 4]]),
            "((#1,#2),(#3,#4))"
        );
        assert_eq!(
            format_real_grid(&[vec![1.0, 1.0]]).unwrap(),
            "((1.0,1.0))"
        );
    }

    #[test]
    fn emits_polynomial_curve_body() {
        let knots = Knots::explicit(&[4, 4], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
        let body =
            bspline_curve_entity(3, &[1, 2, 3, 4], &knots, None, false, false).unwrap();
        assert_eq!(
            body,
            "B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.0),.UNSPECIFIED.)"
        );
    }

    #[test]
    fn emits_rational_curve_complex_instance() {
        let knots = Knots::explicit(&[3, 3], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
        let weights = [1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0];
        let body =
            bspline_curve_entity(2, &[1, 2, 3], &knots, Some(&weights), false, false).unwrap();
        assert!(body.starts_with("(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3)"));
        assert!(body.contains("B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.0),.UNSPECIFIED.)"));
        assert!(body.contains("RATIONAL_B_SPLINE_CURVE((1.0,0.7071067811865476,1.0))"));
        assert!(body.ends_with("REPRESENTATION_ITEM(''))"));
    }

    #[test]
    fn rejects_inconsistent_knots_and_weights() {
        let knots = Knots::explicit(&[4, 4], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
        assert!(bspline_curve_entity(3, &[1, 2], &knots, None, false, false).is_err());
        assert!(bspline_curve_entity(3, &[1, 2, 3, 4], &knots, Some(&[1.0]), false, false).is_err());
        assert!(bspline_curve_entity(3, &[1, 2, 3, 4], &knots, Some(&[1.0, 0.0, 1.0, 1.0]), false, false).is_err());
    }
}
