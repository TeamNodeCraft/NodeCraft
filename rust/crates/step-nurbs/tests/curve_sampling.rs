use step_nurbs::{BsplineCurve, Knots};

#[test]
fn samples_bspline_curve_endpoints() {
    let curve = BsplineCurve {
        degree: 1,
        poles: vec![[0.0,0.0,0.0],[2.0,0.0,0.0],[4.0,0.0,0.0]],
        weights: None,
        knots: Knots::quasi_uniform(1, 3).unwrap(),
        closed: false,
        self_intersect: false,
    };
    let points = curve.sample(4).unwrap();
    assert_eq!(points.len(), 5);
    assert_eq!(points[0], [0.0,0.0,0.0]);
    assert_eq!(points[4], [4.0,0.0,0.0]);
}
