use step_nurbs::{BsplineSurface, Knots};

#[test]
fn samples_bilinear_surface_corners() {
    let surface = BsplineSurface {
        u_degree: 1, v_degree: 1,
        poles: vec![vec![[0.0,0.0,0.0],[0.0,1.0,0.0]], vec![[1.0,0.0,0.0],[1.0,1.0,0.0]]],
        weights: None,
        u_knots: Knots::quasi_uniform(1, 2).unwrap(),
        v_knots: Knots::quasi_uniform(1, 2).unwrap(),
        u_closed: false, v_closed: false, self_intersect: false,
    };
    let grid = surface.sample(2, 2).unwrap();
    assert_eq!(grid.len(), 3);
    assert_eq!(grid[0].len(), 3);
    assert_eq!(grid[0][0], [0.0,0.0,0.0]);
    assert_eq!(grid[2][2], [1.0,1.0,0.0]);
}
