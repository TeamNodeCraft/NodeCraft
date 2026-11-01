//! Integration tests: inline STEP fixtures, parsing, evaluation and emit roundtrips.
//!
//! Every fixture is a complete ISO-10303-21 exchange structure so the files exercise the
//! same path an imported STEP file takes through [`nodecraft_step::StepFile::parse`].

use nodecraft_step::StepFile;
use step_nurbs::{
    KnotSpec, Knots, bspline_curve_entity, bspline_surface_entity, curves, parse_curve,
    parse_surface, surfaces,
};

fn exchange(body: &str) -> String {
    format!(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
         FILE_NAME('','',(''),(''),'step-nurbs-test','step-nurbs-test','');\n\
         FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{body}\nENDSEC;\n\
         END-ISO-10303-21;\n"
    )
}

fn assert_point(actual: [f64; 3], expected: [f64; 3]) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 1e-9,
            "axis {axis}: got {actual:?}, expected {expected:?}"
        );
    }
}

// ---------------------------------------------------------------------------------
// polynomial curves
// ---------------------------------------------------------------------------------

/// Cubic Bézier (clamped knots) with poles (0,0,0) (1,2,0) (2,2,0) (3,0,0).
const CUBIC_BEZIER: &str = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,2.,0.));
#3=CARTESIAN_POINT('',(2.,2.,0.));
#4=CARTESIAN_POINT('',(3.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.),.UNSPECIFIED.);
";

#[test]
fn parses_cubic_bezier_curve() {
    let file = StepFile::parse(&exchange(CUBIC_BEZIER)).unwrap();
    let curve = parse_curve(&file, 10).unwrap();

    assert_eq!(curve.degree, 3);
    assert_eq!(curve.poles.len(), 4);
    assert_eq!(curve.weights, None);
    assert!(!curve.is_rational());
    assert!(!curve.closed);
    assert!(!curve.self_intersect);
    assert_eq!(curve.knots.multiplicities, vec![4, 4]);
    assert_eq!(curve.knots.values, vec![0.0, 1.0]);
    assert_eq!(curve.knots.spec, KnotSpec::Unspecified);
    assert!(!curve.knots.derived);
    assert_eq!(curve.knot_vector(), vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(curve.domain(), (0.0, 1.0));

    assert_point(curve.point_at(0.0).unwrap(), [0.0, 0.0, 0.0]);
    assert_point(curve.point_at(1.0).unwrap(), [3.0, 0.0, 0.0]);
    // Bézier midpoint: (P0 + 3 P1 + 3 P2 + P3) / 8.
    assert_point(curve.point_at(0.5).unwrap(), [1.5, 1.5, 0.0]);
    // Values outside the domain are clamped.
    assert_point(curve.point_at(-4.0).unwrap(), [0.0, 0.0, 0.0]);
    assert_point(curve.point_at(9.0).unwrap(), [3.0, 0.0, 0.0]);
}

#[test]
fn derives_uniform_knots_for_knotless_curve() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(2.,0.,0.));
#3=CARTESIAN_POINT('',(4.,0.,0.));
#10=B_SPLINE_CURVE('',2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let curve = parse_curve(&file, 10).unwrap();

    assert!(curve.knots.derived);
    assert_eq!(curve.knots.spec, KnotSpec::Uniform);
    assert_eq!(curve.knots.multiplicities, vec![1, 1, 1, 1, 1, 1]);
    assert_eq!(curve.knots.values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    // Uniform quadratic: domain is [U[2], U[3]] = [2, 3].
    assert_eq!(curve.domain(), (2.0, 3.0));
    // At u = 2.5 the quadratic basis weights are (1/8, 3/4, 1/8).
    assert_point(curve.point_at(2.5).unwrap(), [2.0, 0.0, 0.0]);
}

#[test]
fn reads_two_dimensional_control_points() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.));
#2=CARTESIAN_POINT('',(2.,2.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let curve = parse_curve(&file, 10).unwrap();
    assert_eq!(curve.poles, vec![[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]);
    assert_point(curve.point_at(0.5).unwrap(), [1.0, 1.0, 0.0]);
}

// ---------------------------------------------------------------------------------
// rational curves (complex instances)
// ---------------------------------------------------------------------------------

/// Rational quadratic quarter circle: poles (1,0) (1,1) (0,1), weights 1, 1/√2, 1.
const QUARTER_CIRCLE: &str = "
#1=CARTESIAN_POINT('',(1.,0.,0.));
#2=CARTESIAN_POINT('',(1.,1.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
#20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
     B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
     CURVE() GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_CURVE((1.,0.7071067811865476,1.))
     REPRESENTATION_ITEM(''));
";

#[test]
fn parses_rational_complex_instance() {
    let file = StepFile::parse(&exchange(QUARTER_CIRCLE)).unwrap();
    let curve = parse_curve(&file, 20).unwrap();

    assert_eq!(curve.degree, 2);
    assert!(curve.is_rational());
    assert_eq!(
        curve.weights,
        Some(vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0])
    );
    assert_eq!(curve.knots.multiplicities, vec![3, 3]);
    assert_eq!(curve.knots.values, vec![0.0, 1.0]);
    assert_eq!(curve.knots.spec, KnotSpec::Unspecified);
    assert_eq!(curve.domain(), (0.0, 1.0));

    assert_point(curve.point_at(0.0).unwrap(), [1.0, 0.0, 0.0]);
    assert_point(curve.point_at(1.0).unwrap(), [0.0, 1.0, 0.0]);
    // The rational midpoint of the quarter circle is (√2/2, √2/2).
    assert_point(curve.point_at(0.5).unwrap(), [0.5f64.sqrt(), 0.5f64.sqrt(), 0.0]);
}

#[test]
fn rational_curve_roundtrip_through_emitted_entity() {
    let file = StepFile::parse(&exchange(QUARTER_CIRCLE)).unwrap();
    let original = parse_curve(&file, 20).unwrap();

    let body = bspline_curve_entity(
        original.degree,
        &[1, 2, 3],
        &original.knots,
        original.weights.as_deref(),
        original.closed,
        original.self_intersect,
    )
    .unwrap();
    assert!(body.starts_with("(BOUNDED_CURVE() B_SPLINE_CURVE(2,"));

    let roundtrip_file = StepFile::parse(&exchange(&format!(
        "#1=CARTESIAN_POINT('',(1.,0.,0.));\n\
         #2=CARTESIAN_POINT('',(1.,1.,0.));\n\
         #3=CARTESIAN_POINT('',(0.,1.,0.));\n\
         #30={body};"
    )))
    .unwrap();
    let parsed = parse_curve(&roundtrip_file, 30).unwrap();

    assert_eq!(parsed.degree, original.degree);
    assert_eq!(parsed.poles, original.poles);
    assert_eq!(parsed.weights, original.weights);
    assert_eq!(parsed.knots.multiplicities, original.knots.multiplicities);
    assert_eq!(parsed.knots.values, original.knots.values);
    assert_eq!(parsed.knots.spec, original.knots.spec);
    assert_eq!(parsed.closed, original.closed);
    assert_eq!(parsed.self_intersect, original.self_intersect);
    assert_point(parsed.point_at(0.5).unwrap(), [0.5f64.sqrt(), 0.5f64.sqrt(), 0.0]);
}

// ---------------------------------------------------------------------------------
// surfaces
// ---------------------------------------------------------------------------------

/// Bilinear patch: u along X, v along Y.
const BILINEAR_PATCH: &str = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,1.,0.));
#3=CARTESIAN_POINT('',(1.,0.,0.));
#4=CARTESIAN_POINT('',(1.,1.,0.));
#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,
     (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);
";

#[test]
fn parses_bilinear_surface() {
    let file = StepFile::parse(&exchange(BILINEAR_PATCH)).unwrap();
    let surface = parse_surface(&file, 10).unwrap();

    assert_eq!(surface.u_degree, 1);
    assert_eq!(surface.v_degree, 1);
    assert_eq!(surface.u_control_count(), 2);
    assert_eq!(surface.v_control_count(), 2);
    assert_eq!(surface.poles[1][1], [1.0, 1.0, 0.0]);
    assert!(!surface.is_rational());
    assert_eq!(surface.u_knots.multiplicities, vec![2, 2]);
    assert_eq!(surface.v_knots.multiplicities, vec![2, 2]);
    assert_eq!(surface.u_domain(), (0.0, 1.0));
    assert_eq!(surface.v_domain(), (0.0, 1.0));

    assert_point(surface.point_at(0.0, 0.0).unwrap(), [0.0, 0.0, 0.0]);
    assert_point(surface.point_at(1.0, 1.0).unwrap(), [1.0, 1.0, 0.0]);
    assert_point(surface.point_at(0.5, 0.5).unwrap(), [0.5, 0.5, 0.0]);
    assert_point(surface.point_at(0.25, 0.75).unwrap(), [0.25, 0.75, 0.0]);
}

#[test]
fn derives_quasi_uniform_default_knots_for_surface() {
    // 3x3 degree-1 control net with `$` multiplicities and `$` knot values.
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,1.,0.));
#3=CARTESIAN_POINT('',(0.,2.,0.));
#4=CARTESIAN_POINT('',(1.,0.,0.));
#5=CARTESIAN_POINT('',(1.,1.,9.));
#6=CARTESIAN_POINT('',(1.,2.,0.));
#7=CARTESIAN_POINT('',(2.,0.,0.));
#8=CARTESIAN_POINT('',(2.,1.,0.));
#9=CARTESIAN_POINT('',(2.,2.,0.));
#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2,#3),(#4,#5,#6),(#7,#8,#9)),
     .UNSPECIFIED.,.F.,.F.,.F.,$, $, $, $,.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let surface = parse_surface(&file, 10).unwrap();

    assert!(surface.u_knots.derived);
    assert!(surface.v_knots.derived);
    assert_eq!(surface.u_knots.spec, KnotSpec::QuasiUniform);
    assert_eq!(surface.u_knots.multiplicities, vec![2, 1, 2]);
    assert_eq!(surface.u_knots.values, vec![0.0, 0.5, 1.0]);
    assert_eq!(surface.v_knots.values, vec![0.0, 0.5, 1.0]);
    assert_eq!(surface.u_domain(), (0.0, 1.0));
    // A degree-1 clamped curve interpolates the middle row at the interior knot.
    assert_point(surface.point_at(0.5, 0.5).unwrap(), [1.0, 1.0, 9.0]);
}

/// Rational bilinear patch with weights 1, 4, 1, 1.
const RATIONAL_PATCH: &str = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,1.,0.));
#3=CARTESIAN_POINT('',(1.,0.,0.));
#4=CARTESIAN_POINT('',(1.,1.,0.));
#40=(BOUNDED_SURFACE() B_SPLINE_SURFACE(1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.)
     B_SPLINE_SURFACE_WITH_KNOTS((2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.)
     GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_SURFACE(((1.,4.),(1.,1.)))
     REPRESENTATION_ITEM('') SURFACE());
";

#[test]
fn parses_rational_surface_and_weights_affect_geometry() {
    let file = StepFile::parse(&exchange(RATIONAL_PATCH)).unwrap();
    let surface = parse_surface(&file, 40).unwrap();

    assert!(surface.is_rational());
    assert_eq!(surface.weights, Some(vec![vec![1.0, 4.0], vec![1.0, 1.0]]));
    assert_eq!(surface.weight(0, 1), 4.0);

    // Bilinear basis at (0.5, 0.5) is 1/4 each; the (u0, v1) pole carries weight 4.
    // x = (0*1 + 0*4 + 1*1 + 1*1) / (1 + 4 + 1 + 1) = 2/7
    // y = (0*1 + 1*4 + 0*1 + 1*1) / 7 = 5/7
    assert_point(surface.point_at(0.5, 0.5).unwrap(), [2.0 / 7.0, 5.0 / 7.0, 0.0]);
    // Corners are interpolated regardless of the weights.
    assert_point(surface.point_at(0.0, 1.0).unwrap(), [0.0, 1.0, 0.0]);
    assert_point(surface.point_at(1.0, 0.0).unwrap(), [1.0, 0.0, 0.0]);
}

#[test]
fn rational_surface_roundtrip_through_emitted_entity() {
    let file = StepFile::parse(&exchange(RATIONAL_PATCH)).unwrap();
    let original = parse_surface(&file, 40).unwrap();

    let body = bspline_surface_entity(
        original.u_degree,
        original.v_degree,
        &[vec![1, 2], vec![3, 4]],
        &original.u_knots,
        &original.v_knots,
        original.weights.as_deref(),
        original.u_closed,
        original.v_closed,
        original.self_intersect,
    )
    .unwrap();
    assert!(body.starts_with("(BOUNDED_SURFACE() B_SPLINE_SURFACE(1,1,"));

    let roundtrip_file = StepFile::parse(&exchange(&format!(
        "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
         #2=CARTESIAN_POINT('',(0.,1.,0.));\n\
         #3=CARTESIAN_POINT('',(1.,0.,0.));\n\
         #4=CARTESIAN_POINT('',(1.,1.,0.));\n\
         #50={body};"
    )))
    .unwrap();
    let parsed = parse_surface(&roundtrip_file, 50).unwrap();

    assert_eq!(parsed.u_degree, original.u_degree);
    assert_eq!(parsed.v_degree, original.v_degree);
    assert_eq!(parsed.poles, original.poles);
    assert_eq!(parsed.weights, original.weights);
    assert_eq!(parsed.u_knots.multiplicities, original.u_knots.multiplicities);
    assert_eq!(parsed.v_knots.multiplicities, original.v_knots.multiplicities);
    assert_point(
        parsed.point_at(0.5, 0.5).unwrap(),
        [2.0 / 7.0, 5.0 / 7.0, 0.0],
    );
}

#[test]
fn polynomial_curve_roundtrip_through_emitted_entity() {
    // Clamped cubic with 5 poles and one interior knot (multiplicity 1).
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,2.,0.));
#3=CARTESIAN_POINT('',(2.,-1.,0.));
#4=CARTESIAN_POINT('',(3.,2.,0.));
#5=CARTESIAN_POINT('',(4.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4,#5),.UNSPECIFIED.,.F.,.F.,
     (4,1,4),(0.,0.5,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let original = parse_curve(&file, 10).unwrap();
    assert_eq!(original.knots.multiplicities, vec![4, 1, 4]);
    assert_eq!(original.knots.values, vec![0.0, 0.5, 1.0]);

    let emitted = bspline_curve_entity(
        3,
        &[1, 2, 3, 4, 5],
        &original.knots,
        None,
        false,
        false,
    )
    .unwrap();
    assert!(emitted.starts_with("B_SPLINE_CURVE_WITH_KNOTS('',3,"));

    let roundtrip_file = StepFile::parse(&exchange(&format!(
        "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
         #2=CARTESIAN_POINT('',(1.,2.,0.));\n\
         #3=CARTESIAN_POINT('',(2.,-1.,0.));\n\
         #4=CARTESIAN_POINT('',(3.,2.,0.));\n\
         #5=CARTESIAN_POINT('',(4.,0.,0.));\n\
         #11={emitted};"
    )))
    .unwrap();
    let parsed = parse_curve(&roundtrip_file, 11).unwrap();
    assert_eq!(parsed.poles, original.poles);
    assert_eq!(parsed.weights, None);
    assert_eq!(parsed.knots, original.knots);
    assert_point(parsed.point_at(0.25).unwrap(), original.point_at(0.25).unwrap());
}

#[test]
fn discovers_curves_and_surfaces_in_entity_order() {
    // Same geometry as the fixtures above, renumbered so entity ids stay unique.
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,2.,0.));
#3=CARTESIAN_POINT('',(2.,2.,0.));
#4=CARTESIAN_POINT('',(3.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.),.UNSPECIFIED.);
#21=CARTESIAN_POINT('',(1.,0.,0.));
#22=CARTESIAN_POINT('',(1.,1.,0.));
#23=CARTESIAN_POINT('',(0.,1.,0.));
#30=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#21,#22,#23),.UNSPECIFIED.,.F.,.F.)
     B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
     CURVE() GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_CURVE((1.,0.7071067811865476,1.))
     REPRESENTATION_ITEM(''));
#41=CARTESIAN_POINT('',(0.,0.,0.));
#42=CARTESIAN_POINT('',(0.,1.,0.));
#43=CARTESIAN_POINT('',(1.,0.,0.));
#44=CARTESIAN_POINT('',(1.,1.,0.));
#50=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#41,#42),(#43,#44)),.UNSPECIFIED.,.F.,.F.,.F.,
     (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let found_curves = curves(&file).unwrap();
    let found_surfaces = surfaces(&file).unwrap();
    assert_eq!(found_curves.len(), 2);
    assert_eq!(found_surfaces.len(), 1);
    assert_eq!(found_curves[0].degree, 3);
    assert_eq!(found_curves[1].degree, 2);
    assert!(found_curves[1].is_rational());
    assert_eq!(found_surfaces[0].u_degree, 1);
}

// ---------------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------------

#[test]
fn rejects_missing_control_point_reference() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#99),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let error = parse_curve(&file, 10).unwrap_err();
    assert!(error.contains("#99"), "{error}");
}

#[test]
fn rejects_knot_count_mismatch() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(3,3),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let error = parse_curve(&file, 10).unwrap_err();
    assert!(error.contains("knot vector has 6 entries"), "{error}");
}

#[test]
fn rejects_zero_degree_and_short_control_net() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#10=B_SPLINE_CURVE_WITH_KNOTS('',0,(#1,#2),.UNSPECIFIED.,.F.,.F.,(1,1),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    assert!(parse_curve(&file, 10).is_err());
}

#[test]
fn rejects_rational_weights_of_wrong_length() {
    let body = "
#1=CARTESIAN_POINT('',(1.,0.,0.));
#2=CARTESIAN_POINT('',(1.,1.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
#20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
     B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
     CURVE() GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_CURVE((1.,1.))
     REPRESENTATION_ITEM(''));
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let error = parse_curve(&file, 20).unwrap_err();
    assert!(error.contains("2 weights for 3 control points"), "{error}");
}

#[test]
fn rejects_non_positive_weights() {
    let body = "
#1=CARTESIAN_POINT('',(1.,0.,0.));
#2=CARTESIAN_POINT('',(1.,1.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
#20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
     B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
     CURVE() GEOMETRIC_REPRESENTATION_ITEM()
     RATIONAL_B_SPLINE_CURVE((1.,0.,1.))
     REPRESENTATION_ITEM(''));
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let error = parse_curve(&file, 20).unwrap_err();
    assert!(error.contains("not positive"), "{error}");
}

#[test]
fn rejects_non_rectangular_control_grid() {
    let body = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(0.,1.,0.));
#3=CARTESIAN_POINT('',(1.,0.,0.));
#10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3)),.UNSPECIFIED.,.F.,.F.,.F.,
     (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);
";
    let file = StepFile::parse(&exchange(body)).unwrap();
    let error = parse_surface(&file, 10).unwrap_err();
    assert!(error.contains("not rectangular"), "{error}");
}

#[test]
fn rejects_entity_of_the_wrong_type() {
    let file = StepFile::parse(&exchange("#1=CARTESIAN_POINT('',(0.,0.,0.));")).unwrap();
    assert!(parse_curve(&file, 1).is_err());
    assert!(parse_curve(&file, 999).is_err());
    assert!(parse_surface(&file, 1).is_err());
}

#[test]
fn default_knot_constructors_validate_inputs() {
    let quasi = Knots::quasi_uniform(2, 4).unwrap();
    assert_eq!(quasi.multiplicities, vec![3, 1, 3]);
    assert_eq!(quasi.values, vec![0.0, 0.5, 1.0]);
    assert!(quasi.derived);

    let uniform = Knots::uniform(1, 3).unwrap();
    assert_eq!(uniform.multiplicities, vec![1, 1, 1, 1, 1]);
    assert_eq!(uniform.values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    let bezier = Knots::piecewise_bezier(2, 5).unwrap();
    assert_eq!(bezier.multiplicities, vec![3, 2, 3]);
    assert_eq!(bezier.values, vec![0.0, 1.0, 2.0]);
    assert!(Knots::piecewise_bezier(2, 4).is_err());
    assert!(Knots::quasi_uniform(2, 2).is_err());

    assert_eq!(
        Knots::default_for(2, 4, KnotSpec::Unspecified).unwrap().multiplicities,
        vec![3, 1, 3]
    );
    assert_eq!(
        Knots::default_for(2, 4, KnotSpec::Uniform).unwrap().spec,
        KnotSpec::Uniform
    );
}

/// Repository fixture, if this checkout ships it.
fn repo_file(relative: &str) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(relative);
    path.is_file().then_some(path)
}

#[test]
fn parses_polynomial_curves_from_real_ap203_sample() {
    let Some(path) = repo_file("tests/Sample AP203 Data Files/gear.stp") else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    let file = StepFile::parse(&text).unwrap();
    let found = curves(&file).unwrap();
    assert!(found.len() >= 8, "found {} curves", found.len());
    for curve in &found {
        assert_eq!(curve.degree, 3);
        assert!(!curve.is_rational());
        let (low, high) = curve.domain();
        assert!(high > low, "empty domain {low}..{high}");
        let point = curve.point_at(0.5 * (low + high)).unwrap();
        assert!(point.iter().all(|value| value.is_finite()), "{point:?}");
    }
    // The sample has no B-spline surfaces.
    assert!(surfaces(&file).unwrap().is_empty());
}

#[test]
fn parses_rational_surfaces_from_real_ap203_sample() {
    let Some(path) = repo_file("tests/Sample AP203 Data Files/jack_in_the_box.stp") else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    let file = StepFile::parse(&text).unwrap();
    let found = surfaces(&file).unwrap();
    assert!(!found.is_empty(), "no B-spline surfaces parsed");
    let rational: Vec<_> = found.iter().filter(|surface| surface.is_rational()).collect();
    assert!(!rational.is_empty(), "no rational surfaces parsed");
    for surface in rational.iter().take(8) {
        let (u_low, u_high) = surface.u_domain();
        let (v_low, v_high) = surface.v_domain();
        assert!(u_high > u_low && v_high > v_low);
        let point = surface
            .point_at(0.5 * (u_low + u_high), 0.5 * (v_low + v_high))
            .unwrap();
        assert!(point.iter().all(|value| value.is_finite()), "{point:?}");
    }
}

#[test]
fn emit_helpers_reject_invalid_weights_and_knots() {
    let knots = Knots::explicit(&[4, 4], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
    assert!(bspline_curve_entity(3, &[1, 2, 3], &knots, None, false, false).is_err());
    assert!(
        bspline_curve_entity(3, &[1, 2, 3, 4], &knots, Some(&[1.0, 1.0]), false, false).is_err()
    );
    let surface_knots = Knots::explicit(&[2, 2], &[0.0, 1.0], KnotSpec::Unspecified).unwrap();
    assert!(
        bspline_surface_entity(
            1,
            1,
            &[vec![1, 2]],
            &surface_knots,
            &surface_knots,
            None,
            false,
            false,
            false
        )
        .is_err()
    );
}
