//! Mesh-preview chamfer and circular fillet on a solidcore box.

mod common;

use nodecraft_blend::{
    BlendError, ChamferType, Continuity, CornerTurn, EdgeMode, EdgeTreatment, FILLET_SEGMENTS,
    apply_edge_treatment, edge_selections, select_edges_by_midpoints,
};

/// Edge 0 of the 40 x 30 x 20 box in `SmBrep::GetEdges` order runs along +Y at
/// `x = 40, z = 20` (see the probe in NOTES.md); test points are given in B-rep
/// coordinates and the returned preview mesh is centred, so translate by the box
/// centre `(20, 15, 10)` before comparing.
const CENTER: [f64; 3] = [20.0, 15.0, 10.0];

fn centred(point: [f64; 3]) -> [f64; 3] {
    [
        point[0] - CENTER[0],
        point[1] - CENTER[1],
        point[2] - CENTER[2],
    ]
}

#[test]
fn equal_chamfer_removes_the_edge_and_stays_closed() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let edge = selections[0].clone();
    let treatment = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        chamfer_type: ChamferType::Equal,
        continuity: Continuity::Circular,
        length: 2.0,
        distance2: None,
        angle: None,
        corner_turn: CornerTurn::Mitre,
    };
    let mesh = apply_edge_treatment(&kernel, &brep, std::slice::from_ref(&edge), &treatment)
        .expect("chamfer preview");

    assert!(mesh.faces.len() > 12, "chamfer adds facets");
    assert!(
        common::count_group(&mesh, "blend:chamfer") > 0,
        "chamfer cap facets"
    );
    let (edges, bad) = common::closure_report(&mesh);
    assert!(edges > 0);
    assert_eq!(bad, 0, "chamfered preview must stay watertight");

    // The ideal 45-degree chamfer plane is length/sqrt(2) from the edge line.
    let distance = common::min_distance_to_line(&mesh, centred([40.0, 0.0, 20.0]), [0.0, 1.0, 0.0]);
    assert!(
        (distance - 2.0 / 2.0_f64.sqrt()).abs() < 0.01,
        "chamfer plane distance {distance} != 1.414"
    );

    // Cut corners on both adjacent faces.
    assert!(common::has_vertex_near(
        &mesh,
        centred([40.0, 0.0, 18.0]),
        0.05
    ));
    assert!(common::has_vertex_near(
        &mesh,
        centred([38.0, 0.0, 20.0]),
        0.05
    ));

    // Box volume minus the 0.5 * d^2 * edge_length prism.
    let expected = 40.0 * 30.0 * 20.0 - 0.5 * 2.0 * 2.0 * 30.0;
    let volume = common::volume(&mesh);
    assert!(
        (volume - expected).abs() < 1.0,
        "volume {volume} != {expected}"
    );
}

#[test]
fn two_distance_chamfer_is_asymmetric() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let treatment = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        chamfer_type: ChamferType::TwoDistance,
        continuity: Continuity::Circular,
        length: 4.0,
        distance2: Some(1.0),
        angle: None,
        corner_turn: CornerTurn::Mitre,
    };
    let mesh = apply_edge_treatment(&kernel, &brep, &selections[..1], &treatment).expect("preview");
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);

    // Distance from the edge line to the chamfer plane: d1*d2 / sqrt(d1^2 + d2^2).
    let expected_distance = 4.0 * 1.0 / (4.0f64 * 4.0 + 1.0).sqrt();
    let plane = common::blend_plane(&mesh, "blend:").expect("chamfer facets");
    let distance = common::plane_distance(centred([40.0, 0.0, 20.0]), plane);
    assert!(
        (distance - expected_distance).abs() < 0.01,
        "two-distance plane distance {distance} != {expected_distance}"
    );
    // Asymmetric cut-back: the corner distances on the two faces are 4 and 1.
    let corner_distances: Vec<f64> = mesh
        .faces
        .iter()
        .enumerate()
        .filter(|(index, _)| mesh.face_groups[*index].starts_with("blend:"))
        .flat_map(|(_, face)| face.iter())
        .map(|vertex| {
            let point = mesh.vertices[*vertex as usize].map(f64::from);
            point
        })
        .map(|point| {
            common::min_distance_to_line(
                &nodecraft_protocol::Mesh {
                    vertices: vec![[point[0] as f32, point[1] as f32, point[2] as f32]],
                    ..Default::default()
                },
                centred([40.0, 0.0, 20.0]),
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    assert!(
        corner_distances
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            < 1.05
    );
    assert!(
        corner_distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            > 3.95
    );
    // The two cut-back lines: 4 on one face, 1 on the other (order depends on face order).
    let cut_a = common::has_vertex_near(&mesh, centred([40.0, 0.0, 16.0]), 0.05)
        && common::has_vertex_near(&mesh, centred([39.0, 0.0, 20.0]), 0.05);
    let cut_b = common::has_vertex_near(&mesh, centred([36.0, 0.0, 20.0]), 0.05)
        && common::has_vertex_near(&mesh, centred([40.0, 0.0, 19.0]), 0.05);
    assert!(cut_a || cut_b, "one 4 and one 1 distance cut-back expected");

    let expected_volume = 40.0 * 30.0 * 20.0 - 0.5 * 4.0 * 1.0 * 30.0;
    assert!((common::volume(&mesh) - expected_volume).abs() < 1.0);
}

#[test]
fn distance_angle_chamfer_matches_the_angle_construction() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let treatment = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        chamfer_type: ChamferType::DistanceAngle,
        continuity: Continuity::Circular,
        length: 2.0,
        distance2: None,
        angle: Some(30.0),
        corner_turn: CornerTurn::Mitre,
    };
    let mesh = apply_edge_treatment(&kernel, &brep, &selections[..1], &treatment).expect("preview");
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);

    // A distance-angle chamfer with d1 and angle alpha has plane distance d1 * sin(alpha).
    let expected_distance = 2.0 * 30.0f64.to_radians().sin();
    let plane = common::blend_plane(&mesh, "blend:").expect("chamfer facets");
    let distance = common::plane_distance(centred([40.0, 0.0, 20.0]), plane);
    assert!(
        (distance - expected_distance).abs() < 0.01,
        "distance-angle plane distance {distance} != {expected_distance}"
    );

    // The removed prism is 0.5 * d1 * d2 * edge_length; d2 = d1 * tan(alpha).
    let d2 = 2.0 * 30.0f64.to_radians().tan();
    let expected_volume = 40.0 * 30.0 * 20.0 - 0.5 * 2.0 * d2 * 30.0;
    assert!(
        (common::volume(&mesh) - expected_volume).abs() < 1.0,
        "30-degree volume {} != {expected_volume}",
        common::volume(&mesh)
    );

    // 45 degrees is the equal-distance chamfer: plane distance = 2 / sqrt(2).
    let equal = EdgeTreatment {
        angle: Some(45.0),
        ..treatment
    };
    let equal_mesh = apply_edge_treatment(&kernel, &brep, &selections[..1], &equal).expect("45");
    let equal_plane = common::blend_plane(&equal_mesh, "blend:").expect("facets");
    let equal_distance = common::plane_distance(centred([40.0, 0.0, 20.0]), equal_plane);
    assert!((equal_distance - 2.0 / 2.0_f64.sqrt()).abs() < 0.01);
    let equal_volume = 40.0 * 30.0 * 20.0 - 0.5 * 2.0 * 2.0 * 30.0;
    assert!((common::volume(&equal_mesh) - equal_volume).abs() < 1.0);
}

#[test]
fn circular_fillet_is_faceted_tangent_and_closed() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let treatment = EdgeTreatment {
        mode: EdgeMode::Fillet,
        chamfer_type: ChamferType::Equal,
        continuity: Continuity::Circular,
        length: 2.0,
        distance2: None,
        angle: None,
        corner_turn: CornerTurn::Mitre,
    };
    let mesh = apply_edge_treatment(&kernel, &brep, &selections[..1], &treatment).expect("fillet");
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0, "filled preview must stay watertight");

    // The tangent-plane facets are built one per interior arc position.
    let cap_facets = common::count_group(&mesh, "blend:fillet");
    assert!(
        cap_facets >= FILLET_SEGMENTS - 1,
        "expected at least {} fillet facets, got {cap_facets}",
        FILLET_SEGMENTS - 1
    );

    // A faceted circular fillet keeps the arc within about r * (1 - cos(pi/2N)) of the
    // edge; it must never come closer than ~r * (sqrt(2) - 1) / 2.
    let distance = common::min_distance_to_line(&mesh, centred([40.0, 0.0, 20.0]), [0.0, 1.0, 0.0]);
    let ideal = 2.0 * (2.0f64.sqrt() - 1.0);
    assert!(
        distance > ideal * 0.6 && distance < ideal * 1.2,
        "fillet surface distance {distance} is not near the ideal {ideal}"
    );

    // Volume: box minus (r^2 - quarter circle) * edge length, with the circumscribed
    // polygon removing slightly less than the ideal arc.
    let ideal_volume =
        40.0 * 30.0 * 20.0 - 30.0 * (2.0 * 2.0 - std::f64::consts::PI * 2.0 * 2.0 / 4.0);
    let volume = common::volume(&mesh);
    assert!(
        volume > ideal_volume - 1.0 && volume < 40.0 * 30.0 * 20.0,
        "fillet volume {volume} not in [{ideal_volume}, 24000]"
    );
}

#[test]
fn two_parallel_edges_can_be_treated_together() {
    let (kernel, brep) = common::box_40_30_20();
    // The two +Y edges at z = 20: (40, 15, 20) and (0, 15, 20).
    let both = select_edges_by_midpoints(&kernel, &brep, &[[40.0, 15.0, 20.0], [0.0, 15.0, 20.0]])
        .expect("both edges");
    assert_eq!(both.len(), 2);
    let treatment = EdgeTreatment::default();
    let mesh = apply_edge_treatment(&kernel, &brep, &both, &treatment).expect("two edges");
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);
    let expected = 40.0 * 30.0 * 20.0 - 2.0 * 0.5 * 2.0 * 2.0 * 30.0;
    assert!((common::volume(&mesh) - expected).abs() < 1.0);
    assert!(common::count_group(&mesh, "blend:chamfer") > 0);
}

#[test]
fn unsupported_treatments_return_explicit_errors() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");

    let g2 = EdgeTreatment {
        mode: EdgeMode::Fillet,
        continuity: Continuity::G2,
        ..EdgeTreatment::default()
    };
    let error = apply_edge_treatment(&kernel, &brep, &selections[..1], &g2).expect_err("G2");
    match &error {
        BlendError::Unsupported {
            feature, missing, ..
        } => {
            assert_eq!(*feature, "fillet G2/G3");
            assert!(
                missing
                    .iter()
                    .any(|call| call.contains("SmBSplineFilletLaw"))
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }

    let mut wide = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        chamfer_type: ChamferType::DistanceAngle,
        angle: Some(120.0),
        ..EdgeTreatment::default()
    };
    let error = apply_edge_treatment(&kernel, &brep, &selections[..1], &wide).expect_err("120 deg");
    assert!(matches!(error, BlendError::Unsupported { .. }));

    wide.chamfer_type = ChamferType::TwoDistance;
    wide.distance2 = Some(-1.0);
    let error = apply_edge_treatment(&kernel, &brep, &selections[..1], &wide).expect_err("neg d2");
    assert!(matches!(error, BlendError::InvalidInput(_)));

    let zero = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        length: 0.0,
        ..EdgeTreatment::default()
    };
    let error = apply_edge_treatment(&kernel, &brep, &selections[..1], &zero).expect_err("zero");
    assert!(matches!(error, BlendError::InvalidInput(_)));

    let empty =
        apply_edge_treatment(&kernel, &brep, &[], &EdgeTreatment::default()).expect_err("no edges");
    assert!(matches!(empty, BlendError::InvalidInput(_)));
}

#[test]
fn curved_edges_are_rejected_without_panicking() {
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    let mut kernel = solidcore::kernel::Kernel::default();
    let cylinder = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 20.0)
        .expect("cylinder");
    let selections = edge_selections(&kernel, &cylinder).expect("cylinder edges");
    assert!(!selections.is_empty());
    let result = apply_edge_treatment(
        &kernel,
        &cylinder,
        &selections[..1],
        &EdgeTreatment::default(),
    );
    match result {
        Err(BlendError::Unsupported { .. }) | Err(BlendError::InvalidInput(_)) => {}
        other => panic!("curved edge treatment must fail explicitly, got {other:?}"),
    }
}
