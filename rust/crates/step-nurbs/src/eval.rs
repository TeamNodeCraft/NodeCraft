//! De Boor evaluation of polynomial and rational B-spline curves and surfaces.
//!
//! The implementation follows *The NURBS Book* (Piegl & Tiller), algorithms A2.1
//! (`FindSpan`), A2.2 (`BasisFuns`) and A3.1 (`CurvePoint`), extended to homogeneous
//! coordinates for rational entities.

use crate::{BsplineCurve, BsplineSurface, knots::Knots};

/// Parameter domain `[U[degree], U[last - degree]]` of an expanded knot vector.
pub fn knot_domain(expanded: &[f64], degree: usize) -> (f64, f64) {
    if expanded.len() > degree {
        (expanded[degree], expanded[expanded.len() - 1 - degree])
    } else {
        (0.0, 0.0)
    }
}

/// Domain of `curve` in its own parameterisation.
pub fn curve_domain(curve: &BsplineCurve) -> (f64, f64) {
    knot_domain(&curve.knot_vector(), curve.degree)
}

/// Point on `curve` at parameter `u` (clamped to the curve domain).
pub fn curve_point(curve: &BsplineCurve, u: f64) -> Result<[f64; 3], String> {
    let degree = curve.degree;
    let control_count = curve.poles.len();
    let knots = curve.knot_vector();
    knots_for(degree, control_count, &knots)?;
    if !u.is_finite() {
        return Err(format!("curve parameter is not finite ({u})"));
    }
    let (low, high) = knot_domain(&knots, degree);
    let u = u.clamp(low, high);
    let span = find_span(control_count - 1, degree, u, &knots);
    let basis = basis_functions(span, u, degree, &knots);
    let mut numerator = [0.0; 3];
    let mut denominator = 0.0;
    for (offset, weight) in basis.iter().enumerate() {
        let index = span - degree + offset;
        let pole_weight = curve.weight(index);
        numerator[0] += weight * pole_weight * curve.poles[index][0];
        numerator[1] += weight * pole_weight * curve.poles[index][1];
        numerator[2] += weight * pole_weight * curve.poles[index][2];
        denominator += weight * pole_weight;
    }
    if denominator.abs() <= f64::EPSILON {
        return Err(format!("degenerate curve at u = {u}: weight sum is zero"));
    }
    Ok([
        numerator[0] / denominator,
        numerator[1] / denominator,
        numerator[2] / denominator,
    ])
}

/// Point on `surface` at parameters `(u, v)` (clamped to the surface domains).
pub fn surface_point(surface: &BsplineSurface, u: f64, v: f64) -> Result<[f64; 3], String> {
    let u_count = surface.u_control_count();
    let v_count = surface.v_control_count();
    if u_count == 0 || v_count == 0 {
        return Err("surface has an empty control net".into());
    }
    if surface.poles.iter().any(|row| row.len() != v_count) {
        return Err("surface control net is not rectangular".into());
    }
    let u_knots = surface.u_knots.expanded();
    let v_knots = surface.v_knots.expanded();
    knots_for(surface.u_degree, u_count, &u_knots)?;
    knots_for(surface.v_degree, v_count, &v_knots)?;
    if !(u.is_finite() && v.is_finite()) {
        return Err(format!("surface parameters are not finite ({u}, {v})"));
    }
    let (u_low, u_high) = knot_domain(&u_knots, surface.u_degree);
    let (v_low, v_high) = knot_domain(&v_knots, surface.v_degree);
    let u = u.clamp(u_low, u_high);
    let v = v.clamp(v_low, v_high);
    let u_span = find_span(u_count - 1, surface.u_degree, u, &u_knots);
    let v_span = find_span(v_count - 1, surface.v_degree, v, &v_knots);
    let u_basis = basis_functions(u_span, u, surface.u_degree, &u_knots);
    let v_basis = basis_functions(v_span, v, surface.v_degree, &v_knots);
    let mut numerator = [0.0; 3];
    let mut denominator = 0.0;
    for (u_offset, u_weight) in u_basis.iter().enumerate() {
        let i = u_span - surface.u_degree + u_offset;
        for (v_offset, v_weight) in v_basis.iter().enumerate() {
            let j = v_span - surface.v_degree + v_offset;
            let pole_weight = surface.weight(i, j);
            let weight = u_weight * v_weight * pole_weight;
            numerator[0] += weight * surface.poles[i][j][0];
            numerator[1] += weight * surface.poles[i][j][1];
            numerator[2] += weight * surface.poles[i][j][2];
            denominator += weight;
        }
    }
    if denominator.abs() <= f64::EPSILON {
        return Err(format!(
            "degenerate surface at (u, v) = ({u}, {v}): weight sum is zero"
        ));
    }
    Ok([
        numerator[0] / denominator,
        numerator[1] / denominator,
        numerator[2] / denominator,
    ])
}

/// Reject knot vectors that cannot describe the curve/surface.
fn knots_for(degree: usize, control_count: usize, expanded: &[f64]) -> Result<(), String> {
    if degree == 0 {
        return Err("B-spline degree must be at least 1".into());
    }
    if control_count < degree + 1 {
        return Err(format!(
            "degree {degree} needs at least {} control points, got {control_count}",
            degree + 1
        ));
    }
    let expected = control_count + degree + 1;
    if expanded.len() != expected {
        return Err(format!(
            "knot vector has {} entries but degree {degree} with {control_count} control \
             points needs {expected}",
            expanded.len()
        ));
    }
    if expanded.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("knot values must be non-decreasing".into());
    }
    Ok(())
}

/// `FindSpan`: index of the knot span containing `u`.
fn find_span(last_control: usize, degree: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[last_control + 1] {
        return last_control;
    }
    if u <= knots[degree] {
        return degree;
    }
    let mut low = degree;
    let mut high = last_control + 1;
    let mut mid = (low + high) / 2;
    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// `BasisFuns`: the `degree + 1` non-zero basis functions at `u` in `span`.
fn basis_functions(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut basis = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    basis[0] = 1.0;
    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let denominator = right[r + 1] + left[j - r];
            let temp = if denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                basis[r] / denominator
            };
            basis[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        basis[j] = saved;
    }
    basis
}

/// Evaluate every distinct knot value in order (handy for sampling a curve).
pub fn knot_parameters(knots: &Knots) -> impl Iterator<Item = f64> + '_ {
    knots.values.iter().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_span_clamps_to_domain() {
        // Degree 1, knots (2,2),(0.,1.) expanded: [0,0,1,1]
        let knots = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(find_span(1, 1, 0.0, &knots), 1);
        assert_eq!(find_span(1, 1, 1.0, &knots), 1);
        assert_eq!(find_span(1, 1, 0.25, &knots), 1);
    }

    #[test]
    fn basis_functions_partition_unity() {
        let knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let span = find_span(3, 2, 0.25, &knots);
        let basis = basis_functions(span, 0.25, 2, &knots);
        let sum: f64 = basis.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "basis sum {sum}");
    }
}
