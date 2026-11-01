use step_conics::{Axis2Placement3d, Conic, Ellipse, Hyperbola, Parabola};

#[test]
fn evaluates_conics_at_parameter() {
    let placement = Axis2Placement3d { location: [1.0, 2.0, 3.0], axis: None, ref_direction: None };
    let ellipse = Conic::Ellipse(Ellipse { placement, semi_axis_1: 2.0, semi_axis_2: 1.0 });
    let point = ellipse.point_at(0.0).unwrap();
    assert!((point[0]-3.0).abs() < 1e-12 && (point[1]-2.0).abs() < 1e-12 && (point[2]-3.0).abs() < 1e-12, "{point:?}");
    let hyperbola = Conic::Hyperbola(Hyperbola { placement, semi_axis: 2.0, semi_imag_axis: 1.0 });
    assert!((hyperbola.point_at(0.0).unwrap()[0]-3.0).abs() < 1e-12);
    let parabola = Conic::Parabola(Parabola { placement, focal_distance: 2.0 });
    let point = parabola.point_at(2.0).unwrap();
    assert!((point[0]-1.5).abs() < 1e-12 && (point[1]-4.0).abs() < 1e-12, "{point:?}");
}

#[test]
fn samples_trimmed_conic() {
    use step_conics::TrimRange;
    let placement = Axis2Placement3d { location: [0.0, 0.0, 0.0], axis: None, ref_direction: None };
    let ellipse = Conic::Ellipse(Ellipse { placement, semi_axis_1: 2.0, semi_axis_2: 1.0 });
    let points = ellipse.sample(&TrimRange { start: 0.0, end: std::f64::consts::FRAC_PI_2, span: std::f64::consts::FRAC_PI_2, full_period: false }, 2).unwrap();
    assert_eq!(points.len(), 3);
    assert!((points[0][0]-2.0).abs() < 1e-12);
    assert!((points[2][1]-1.0).abs() < 1e-12);
}

#[test]
fn samples_hyperbola_and_parabola() {
    use step_conics::TrimRange;
    let placement = Axis2Placement3d { location: [0.0, 0.0, 0.0], axis: None, ref_direction: None };
    let hyperbola = Conic::Hyperbola(Hyperbola { placement, semi_axis: 2.0, semi_imag_axis: 1.0 });
    let points = hyperbola.sample(&TrimRange { start: -1.0, end: 1.0, span: 2.0, full_period: false }, 2).unwrap();
    assert_eq!(points.len(), 3);
    assert!((points[1][0] - 2.0).abs() < 1e-9, "{:?}", points[1]);
    assert!((points[2][0] - 2.0 * 1.0f64.cosh()).abs() < 1e-9, "{:?}", points[2]);
    let parabola = Conic::Parabola(Parabola { placement, focal_distance: 2.0 });
    let points = parabola.sample(&TrimRange { start: -2.0, end: 2.0, span: 4.0, full_period: false }, 2).unwrap();
    assert_eq!(points.len(), 3);
    assert!((points[1][0]).abs() < 1e-12 && (points[1][1]).abs() < 1e-12, "{:?}", points[1]);
}
