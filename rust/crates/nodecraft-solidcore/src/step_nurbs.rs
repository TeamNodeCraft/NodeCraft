//! Map `step_nurbs` B-spline curves and surfaces onto solidcore NURBS geometry.
//!
//! STEP exchange data is parsed by the `step-nurbs` crate into
//! [`step_nurbs::BsplineCurve`] / [`step_nurbs::BsplineSurface`] (degree, poles,
//! optional rational weights, knot multiplicities/values). This module turns those into
//! the solidcore kernel types:
//!
//! * [`solidcore::nurbs::b_spline_curve::BSplineCurve`] wrapped in a
//!   [`solidcore::curve::curve::Curve`] (`SmBSplineCurve` / `SmCurve`), and
//! * [`solidcore::nurbs::b_spline_surface::BSplineSurface`] wrapped in a
//!   [`solidcore::surface::surface::Surface`] (`SmBSplineSurface` / `SmSurface`).
//!
//! The conversion uses the public solidcore constructors — no geometry is forged:
//!
//! * curves: `BSplineCurve::from_data(dim, n, m, p, pw, u)` copies the homogeneous
//!   control points and the knot vector, then lets solidcore fix up the knots;
//! * surfaces: `nurbs::srf_basic::Surface::new(net, p, q, u, v)` builds the NURB and
//!   `BSplineSurface::from_nurb(nurb, borrowed = false)` wraps it. The higher-level
//!   `BSplineSurface::from_data` is deliberately not used: its documented upstream
//!   quirk only copies the v knot vector when `s <= r`, leaving v knots at zero for a
//!   control net whose v knot count exceeds its u knot count (e.g. a 2x3 net). Building
//!   the NURB first always keeps both knot vectors.
//!
//! Poles become homogeneous [`CPoint`]s: non-rational STEP data maps to
//! `CPoint::from_point` (`w == NOW`), rational data to `CPoint::weighted` (`w * P`, `w`).
//! Weights must be finite and strictly positive. Knot vectors are expanded from their
//! STEP multiplicities and validated against degree and control-point count. solidcore
//! evaluates only *clamped* B-splines (end multiplicity `degree + 1`), so a non-clamped
//! STEP knot vector (`.UNIFORM.` data, or a knotless `B_SPLINE_CURVE` that `step-nurbs`
//! expands to uniform knots) is rejected with an explanatory error instead of producing
//! a curve/surface that fails at evaluation time. Supporting it would require knot
//! insertion before conversion.

use solidcore::curve::curve::Curve;
use solidcore::math::{EFF_ZERO_PARAM, Vec3};
use solidcore::nurbs::b_spline_curve::BSplineCurve as KernelBSplineCurve;
use solidcore::nurbs::b_spline_surface::BSplineSurface as KernelBSplineSurface;
use solidcore::nurbs::math::CPoint;
use solidcore::nurbs::srf_basic::Surface as KernelNurbSurface;
use solidcore::surface::surface::Surface;

/// Convert a STEP B-spline surface into a solidcore `SmSurface`
/// (`SurfaceKind::BSpline(SmBSplineSurface)`).
///
/// # Errors
///
/// Returns a message when the control net is empty or not rectangular, the weights do
/// not match the net, a pole/weight is not finite, a weight is not positive, a knot
/// vector does not match its degree and control-point count, or solidcore rejects the
/// NURB.
pub fn bspline_surface_to_solidcore(
    surface: &::step_nurbs::BsplineSurface,
) -> Result<Surface, String> {
    if surface.u_degree == 0 || surface.v_degree == 0 {
        return Err(format!(
            "B-spline surface degrees must be at least 1, got {} x {}",
            surface.u_degree, surface.v_degree
        ));
    }
    let u_count = surface.poles.len();
    let v_count = surface.poles.first().map_or(0, Vec::len);
    if u_count == 0 || v_count == 0 {
        return Err("B-spline surface has an empty control net".into());
    }
    if u_count < surface.u_degree + 1 {
        return Err(format!(
            "u degree {} needs at least {} control rows, got {u_count}",
            surface.u_degree,
            surface.u_degree + 1
        ));
    }
    if v_count < surface.v_degree + 1 {
        return Err(format!(
            "v degree {} needs at least {} control points per row, got {v_count}",
            surface.v_degree,
            surface.v_degree + 1
        ));
    }
    if let Some(row) = surface.poles.iter().find(|row| row.len() != v_count) {
        return Err(format!(
            "control net is not rectangular: expected {v_count} points per row, found {}",
            row.len()
        ));
    }
    if let Some(weights) = &surface.weights {
        if weights.len() != u_count || weights.iter().any(|row| row.len() != v_count) {
            return Err(format!(
                "weight grid is {}x{}, expected {u_count}x{v_count}",
                weights.len(),
                weights.first().map_or(0, Vec::len)
            ));
        }
    }

    let u_knots = knot_vector(&surface.u_knots, surface.u_degree, u_count, "u")?;
    let v_knots = knot_vector(&surface.v_knots, surface.v_degree, v_count, "v")?;

    let mut net: Vec<Vec<CPoint>> = Vec::with_capacity(u_count);
    for (i, row) in surface.poles.iter().enumerate() {
        let mut kernel_row = Vec::with_capacity(v_count);
        for (j, pole) in row.iter().enumerate() {
            let weight = surface.weights.as_ref().map(|weights| weights[i][j]);
            kernel_row.push(control_point(*pole, weight).map_err(|error| {
                format!("surface pole ({i}, {j}) at {pole:?}: {error}")
            })?);
        }
        net.push(kernel_row);
    }

    let nurb = KernelNurbSurface::new(net, surface.u_degree, surface.v_degree, u_knots, v_knots)
        .map_err(|error| format!("solidcore NURB surface: {error}"))?;
    Ok(Surface::from(KernelBSplineSurface::from_nurb(nurb, false)))
}

/// Convert a STEP B-spline curve into a solidcore `SmCurve`
/// (`CurveKind::BSpline(SmBSplineCurve)`).
///
/// # Errors
///
/// Returns a message when the curve is empty, its degree is zero, the control points or
/// weights do not match, a pole/weight is not finite, a weight is not positive, a knot
/// vector does not match its degree and control-point count, or solidcore rejects the
/// NURB.
pub fn bspline_curve_to_solidcore(
    curve: &::step_nurbs::BsplineCurve,
) -> Result<Curve, String> {
    if curve.degree == 0 {
        return Err("B-spline curve degree must be at least 1".into());
    }
    let n = curve
        .poles
        .len()
        .checked_sub(1)
        .ok_or_else(|| "B-spline curve has no control points".to_string())?;
    if curve.poles.len() < curve.degree + 1 {
        return Err(format!(
            "degree {} needs at least {} control points, got {}",
            curve.degree,
            curve.degree + 1,
            curve.poles.len()
        ));
    }
    if let Some(weights) = &curve.weights {
        if weights.len() != curve.poles.len() {
            return Err(format!(
                "{} weights for {} control points",
                weights.len(),
                curve.poles.len()
            ));
        }
    }

    let p = curve.degree;
    let m = n + p + 1;
    let knots = knot_vector(&curve.knots, p, n + 1, "curve")?;
    let poles: Vec<CPoint> = curve
        .poles
        .iter()
        .enumerate()
        .map(|(index, pole)| {
            let weight = curve.weights.as_ref().map(|weights| weights[index]);
            control_point(*pole, weight)
                .map_err(|error| format!("curve pole {index} at {pole:?}: {error}"))
        })
        .collect::<Result<_, _>>()?;

    let kernel = KernelBSplineCurve::from_data(3, n, m, p, &poles, &knots)
        .map_err(|error| format!("solidcore NURB curve: {error}"))?;
    Ok(Curve::from(kernel))
}

/// Expanded, validated STEP knot vector for `control_count` poles of `degree`.
fn knot_vector(
    knots: &::step_nurbs::Knots,
    degree: usize,
    control_count: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    if knots.multiplicities.len() != knots.values.len() {
        return Err(format!(
            "{label} knot multiplicities ({}) and values ({}) differ in length",
            knots.multiplicities.len(),
            knots.values.len()
        ));
    }
    if knots.values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} knot values must be finite"));
    }
    if knots.values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!("{label} knot values must be strictly increasing"));
    }
    if knots.multiplicities.iter().any(|&multiplicity| multiplicity == 0) {
        return Err(format!("{label} knot multiplicities must be positive"));
    }
    let expanded = knots.expanded();
    let expected = control_count + degree + 1;
    if expanded.len() != expected {
        return Err(format!(
            "{label} knot vector has {} entries but degree {degree} with \
             {control_count} control points needs {expected}",
            expanded.len()
        ));
    }
    if expanded.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(format!("{label} knot values must be non-decreasing"));
    }
    let first = expanded[0];
    let last = expanded[expanded.len() - 1];
    // solidcore's knot fix-up merges end knots closer than SM_EFF_ZERO_PARAM, so the
    // clamping test uses the same scaled tolerance.
    let close = |a: f64, b: f64| (a - b).abs() <= EFF_ZERO_PARAM * (1.0 + a.abs().max(b.abs()));
    let clamped_start = expanded[..=degree].iter().all(|value| close(*value, first));
    let clamped_end = expanded[expanded.len() - 1 - degree..]
        .iter()
        .all(|value| close(*value, last));
    if !(clamped_start && clamped_end) {
        return Err(format!(
            "{label} knot vector is not clamped (end multiplicity below degree + 1): \
             solidcore evaluates only clamped B-splines, so knot insertion is needed first"
        ));
    }
    if !(first < last) {
        return Err(format!("{label} knot vector has an empty parameter domain"));
    }
    Ok(expanded)
}

/// One finite control point, homogeneous when a weight is given.
fn control_point(pole: [f64; 3], weight: Option<f64>) -> Result<CPoint, String> {
    if !pole.iter().all(|value| value.is_finite()) {
        return Err("control point is not finite".into());
    }
    let point = Vec3::new(pole[0], pole[1], pole[2]);
    match weight {
        None => Ok(CPoint::from_point(point)),
        Some(weight) if !weight.is_finite() => Err(format!("weight {weight} is not finite")),
        Some(weight) if weight <= 0.0 => Err(format!("weight {weight} is not positive")),
        Some(weight) => Ok(CPoint::weighted(point, weight)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::curve::curve::CurveKind;
    use solidcore::kernel::Kernel;
    use solidcore::math::{Point2, Point3};
    use solidcore::nurbs::b_spline_surface::SurfParamType;
    use solidcore::surface::surface::SurfaceKind;

    const TOLERANCE: f64 = 1e-9;

    fn knots(multiplicities: &[usize], values: &[f64]) -> ::step_nurbs::Knots {
        ::step_nurbs::Knots::explicit(
            multiplicities,
            values,
            ::step_nurbs::KnotSpec::Unspecified,
        )
        .unwrap()
    }

    fn assert_point(actual: Point3, expected: [f64; 3], context: &str) {
        let delta = [
            (actual.x - expected[0]).abs(),
            (actual.y - expected[1]).abs(),
            (actual.z - expected[2]).abs(),
        ];
        assert!(
            delta.iter().all(|value| *value < TOLERANCE),
            "{context}: got ({}, {}, {}), expected {expected:?}",
            actual.x,
            actual.y,
            actual.z
        );
    }

    /// Bilinear patch over the unit square: `poles[u][v] = (u, v, 0)`.
    fn bilinear_surface(weights: Option<Vec<Vec<f64>>>) -> ::step_nurbs::BsplineSurface {
        ::step_nurbs::BsplineSurface {
            u_degree: 1,
            v_degree: 1,
            poles: vec![
                vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            weights,
            u_knots: knots(&[2, 2], &[0.0, 1.0]),
            v_knots: knots(&[2, 2], &[0.0, 1.0]),
            u_closed: false,
            v_closed: false,
            self_intersect: false,
        }
    }

    /// Rational quadratic quarter circle: poles (1,0) (1,1) (0,1), weights 1, 1/√2, 1.
    fn quarter_circle() -> ::step_nurbs::BsplineCurve {
        ::step_nurbs::BsplineCurve {
            degree: 2,
            poles: vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            weights: Some(vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0]),
            knots: knots(&[3, 3], &[0.0, 1.0]),
            closed: false,
            self_intersect: false,
        }
    }

    #[test]
    fn bilinear_surface_round_trips_evaluation() {
        let source = bilinear_surface(None);
        let converted = bspline_surface_to_solidcore(&source).unwrap();

        let SurfaceKind::BSpline(kernel) = converted.kind() else {
            panic!("expected a B-spline surface, got {:?}", converted.kind());
        };
        assert!(!kernel.is_rational());
        assert_eq!(kernel.degree(SurfParamType::U), 1);
        assert_eq!(kernel.degree(SurfParamType::V), 1);
        let domain = converted.natural_uv_domain_if_built();
        assert!((domain.min().x - 0.0).abs() < TOLERANCE);
        assert!((domain.min().y - 0.0).abs() < TOLERANCE);
        assert!((domain.max().x - 1.0).abs() < TOLERANCE);
        assert!((domain.max().y - 1.0).abs() < TOLERANCE);

        let kernel_ctx = Kernel::default();
        for (u, v) in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.75), (1.0, 0.0)] {
            let expected = ::step_nurbs::surface_point(&source, u, v).unwrap();
            let actual = converted
                .evaluate_point(&kernel_ctx, Point2::new(u, v))
                .unwrap();
            assert_point(actual, expected, &format!("bilinear ({u}, {v})"));
        }
    }

    #[test]
    fn quarter_circle_curve_round_trips_evaluation() {
        let source = quarter_circle();
        let converted = bspline_curve_to_solidcore(&source).unwrap();

        let CurveKind::BSpline(kernel) = converted.kind() else {
            panic!("expected a B-spline curve, got {:?}", converted.kind());
        };
        assert!(kernel.is_rational().unwrap());
        assert_eq!(kernel.degree().unwrap(), 2);

        let kernel_ctx = Kernel::default();
        for t in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0] {
            let expected = ::step_nurbs::curve_point(&source, t).unwrap();
            let actual = converted.evaluate_point(&kernel_ctx, t).unwrap();
            assert_point(actual, expected, &format!("quarter circle t={t}"));
        }
        // The classic rational midpoint of the arc.
        let mid = converted.evaluate_point(&kernel_ctx, 0.5).unwrap();
        assert_point(mid, [0.5f64.sqrt(), 0.5f64.sqrt(), 0.0], "arc midpoint");
    }

    #[test]
    fn rational_surface_weights_survive_conversion() {
        let source = bilinear_surface(Some(vec![vec![1.0, 4.0], vec![1.0, 1.0]]));
        let converted = bspline_surface_to_solidcore(&source).unwrap();

        let SurfaceKind::BSpline(kernel) = converted.kind() else {
            panic!("expected a B-spline surface");
        };
        assert!(kernel.is_rational());
        let (u_count, v_count, control_points) = kernel.control_points().unwrap();
        assert_eq!((u_count, v_count), (2, 2));
        assert_eq!(control_points.len(), 4);
        for (i, row) in source.poles.iter().enumerate() {
            for (j, pole) in row.iter().enumerate() {
                let point = control_points[i * v_count + j];
                let weight = source.weights.as_ref().unwrap()[i][j];
                assert!((point.w - weight).abs() < TOLERANCE);
                assert!((point.x - weight * pole[0]).abs() < TOLERANCE);
                assert!((point.y - weight * pole[1]).abs() < TOLERANCE);
                assert!((point.z - weight * pole[2]).abs() < TOLERANCE);
            }
        }

        let kernel_ctx = Kernel::default();
        for (u, v) in [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)] {
            let expected = ::step_nurbs::surface_point(&source, u, v).unwrap();
            let actual = converted
                .evaluate_point(&kernel_ctx, Point2::new(u, v))
                .unwrap();
            assert_point(actual, expected, &format!("rational ({u}, {v})"));
        }
        let center = converted
            .evaluate_point(&kernel_ctx, Point2::new(0.5, 0.5))
            .unwrap();
        assert_point(center, [2.0 / 7.0, 5.0 / 7.0, 0.0], "rational center");
    }

    #[test]
    fn surface_with_more_v_than_u_knots_keeps_both_knot_vectors() {
        // u: 2 control rows, v: 3 control columns with degree 1 each, so the v knot
        // count (5) exceeds the u knot count (4) — the case `from_data` would zero.
        let source = ::step_nurbs::BsplineSurface {
            u_degree: 1,
            v_degree: 1,
            poles: vec![
                vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                vec![[1.0, 0.0, 0.0], [1.0, 1.0, 5.0], [1.0, 2.0, 0.0]],
            ],
            weights: None,
            u_knots: knots(&[2, 2], &[0.0, 1.0]),
            v_knots: knots(&[2, 1, 2], &[0.0, 0.5, 1.0]),
            u_closed: false,
            v_closed: false,
            self_intersect: false,
        };
        let converted = bspline_surface_to_solidcore(&source).unwrap();
        let domain = converted.natural_uv_domain_if_built();
        assert!((domain.max().x - 1.0).abs() < TOLERANCE);
        assert!((domain.max().y - 1.0).abs() < TOLERANCE);

        let kernel_ctx = Kernel::default();
        for (u, v) in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.25, 0.75)] {
            let expected = ::step_nurbs::surface_point(&source, u, v).unwrap();
            let actual = converted
                .evaluate_point(&kernel_ctx, Point2::new(u, v))
                .unwrap();
            assert_point(actual, expected, &format!("2x3 net ({u}, {v})"));
        }
        // A degree-1 curve with an interior knot at v = 0.5 interpolates the middle
        // control column there, so the surface center averages the two middle poles:
        // ((0, 1, 0) + (1, 1, 5)) / 2.
        let mid = converted
            .evaluate_point(&kernel_ctx, Point2::new(0.5, 0.5))
            .unwrap();
        assert_point(mid, [0.5, 1.0, 2.5], "2x3 net center");
    }

    #[test]
    fn cubic_polynomial_curve_round_trips_evaluation() {
        let source = ::step_nurbs::BsplineCurve {
            degree: 3,
            poles: vec![
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            weights: None,
            knots: knots(&[4, 4], &[0.0, 1.0]),
            closed: false,
            self_intersect: false,
        };
        let converted = bspline_curve_to_solidcore(&source).unwrap();
        let kernel_ctx = Kernel::default();
        for t in [0.0, 0.5, 1.0] {
            let expected = ::step_nurbs::curve_point(&source, t).unwrap();
            let actual = converted.evaluate_point(&kernel_ctx, t).unwrap();
            assert_point(actual, expected, &format!("cubic t={t}"));
        }
        let mid = converted.evaluate_point(&kernel_ctx, 0.5).unwrap();
        assert_point(mid, [1.5, 1.5, 0.0], "cubic midpoint");
    }

    #[test]
    fn conversion_rejects_invalid_curves() {
        let mut curve = quarter_circle();
        curve.degree = 0;
        assert!(bspline_curve_to_solidcore(&curve).is_err());

        let mut curve = quarter_circle();
        curve.knots = knots(&[2, 2], &[0.0, 1.0]); // 4 entries, 6 needed
        let error = bspline_curve_to_solidcore(&curve).unwrap_err();
        assert!(error.contains("knot vector has 4 entries"), "{error}");

        let mut curve = quarter_circle();
        curve.weights = Some(vec![1.0, 0.0, 1.0]);
        let error = bspline_curve_to_solidcore(&curve).unwrap_err();
        assert!(error.contains("not positive"), "{error}");

        let mut curve = quarter_circle();
        curve.poles[1] = [f64::NAN, 0.0, 0.0];
        let error = bspline_curve_to_solidcore(&curve).unwrap_err();
        assert!(error.contains("not finite"), "{error}");

        let mut curve = quarter_circle();
        curve.weights = Some(vec![1.0, 1.0]);
        let error = bspline_curve_to_solidcore(&curve).unwrap_err();
        assert!(error.contains("2 weights for 3 control points"), "{error}");
    }

    #[test]
    fn conversion_rejects_invalid_surfaces() {
        let mut surface = bilinear_surface(None);
        surface.u_degree = 0;
        assert!(bspline_surface_to_solidcore(&surface).is_err());

        let mut surface = bilinear_surface(None);
        surface.poles[1].pop();
        let error = bspline_surface_to_solidcore(&surface).unwrap_err();
        assert!(error.contains("not rectangular"), "{error}");

        let surface = bilinear_surface(Some(vec![vec![1.0, 1.0]]));
        let error = bspline_surface_to_solidcore(&surface).unwrap_err();
        assert!(error.contains("weight grid"), "{error}");

        let mut surface = bilinear_surface(None);
        surface.u_knots = knots(&[3, 3], &[0.0, 1.0]); // 6 entries, 4 needed
        let error = bspline_surface_to_solidcore(&surface).unwrap_err();
        assert!(error.contains("u knot vector has 6 entries"), "{error}");
    }

    fn exchange(body: &str) -> String {
        format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'nodecraft-solidcore-test','nodecraft-solidcore-test','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{body}\nENDSEC;\n\
             END-ISO-10303-21;\n"
        )
    }

    #[test]
    fn step_text_curve_parses_and_converts_end_to_end() {
        // The same rational quarter circle, but written as STEP text and parsed by
        // `step-nurbs` first: StepFile -> BsplineCurve -> solidcore Curve.
        let body = "
#1=CARTESIAN_POINT('',(1.,0.,0.));
#2=CARTESIAN_POINT('',(1.,1.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
#20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
     B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
     CURVE() GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_CURVE((1.,0.7071067811865476,1.))
     REPRESENTATION_ITEM(''));
";
        let file = nodecraft_step::StepFile::parse(&exchange(body)).unwrap();
        let source = ::step_nurbs::parse_curve(&file, 20).unwrap();
        assert!(source.is_rational());

        let converted = bspline_curve_to_solidcore(&source).unwrap();
        let kernel_ctx = Kernel::default();
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let expected = ::step_nurbs::curve_point(&source, t).unwrap();
            let actual = converted.evaluate_point(&kernel_ctx, t).unwrap();
            assert_point(actual, expected, &format!("STEP curve t={t}"));
        }
    }

    #[test]
    fn step_text_surface_parses_and_converts_end_to_end() {
        let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,1.,0.));
#3=CARTESIAN_POINT('',(1.,0.,0.));
#4=CARTESIAN_POINT('',(1.,1.,0.));
#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,
     (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);
";
        let file = nodecraft_step::StepFile::parse(&exchange(body)).unwrap();
        let source = ::step_nurbs::parse_surface(&file, 10).unwrap();
        let converted = bspline_surface_to_solidcore(&source).unwrap();
        let kernel_ctx = Kernel::default();
        for (u, v) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
            let expected = ::step_nurbs::surface_point(&source, u, v).unwrap();
            let actual = converted
                .evaluate_point(&kernel_ctx, Point2::new(u, v))
                .unwrap();
            assert_point(actual, expected, &format!("STEP surface ({u}, {v})"));
        }
    }

    #[test]
    fn uniform_non_clamped_knot_vector_is_rejected() {
        // A knotless STEP `B_SPLINE_CURVE` becomes a uniform (non-clamped) knot vector,
        // which solidcore cannot evaluate: the conversion reports it up front.
        let source = ::step_nurbs::BsplineCurve {
            degree: 2,
            poles: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            weights: None,
            knots: ::step_nurbs::Knots::uniform(2, 3).unwrap(),
            closed: false,
            self_intersect: false,
        };
        let error = bspline_curve_to_solidcore(&source).unwrap_err();
        assert!(error.contains("not clamped"), "{error}");

        let mut surface = bilinear_surface(None);
        surface.u_knots = ::step_nurbs::Knots::uniform(1, 2).unwrap();
        let error = bspline_surface_to_solidcore(&surface).unwrap_err();
        assert!(error.contains("u knot vector is not clamped"), "{error}");
    }

    #[test]
    fn empty_inputs_are_rejected() {
        let curve = ::step_nurbs::BsplineCurve {
            degree: 1,
            poles: Vec::new(),
            weights: None,
            knots: knots(&[2, 2], &[0.0, 1.0]),
            closed: false,
            self_intersect: false,
        };
        assert!(bspline_curve_to_solidcore(&curve).is_err());

        let surface = ::step_nurbs::BsplineSurface {
            u_degree: 1,
            v_degree: 1,
            poles: Vec::new(),
            weights: None,
            u_knots: knots(&[2, 2], &[0.0, 1.0]),
            v_knots: knots(&[2, 2], &[0.0, 1.0]),
            u_closed: false,
            v_closed: false,
            self_intersect: false,
        };
        assert!(bspline_surface_to_solidcore(&surface).is_err());
    }
}
