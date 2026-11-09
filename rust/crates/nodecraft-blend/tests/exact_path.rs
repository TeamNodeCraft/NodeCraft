//! The exact (B-rep) path: the blockers are named, and nothing panics.

mod common;

use nodecraft_blend::{
    BlendError, ChamferType, Continuity, EdgeMode, EdgeTreatment, apply_edge_treatment_brep,
    chamfer_brep, edge_selections, fillet_brep, fillet_brep_exact, missing_calls, port_units,
};

#[test]
fn exact_chamfer_reports_the_missing_sm_api() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let treatment = EdgeTreatment::default();
    let error = chamfer_brep(&kernel, &brep, &selections[..1], &treatment).expect_err("blocked");
    match &error {
        BlendError::Unsupported {
            feature,
            reason,
            missing,
        } => {
            assert_eq!(*feature, "exact chamfer");
            assert!(
                reason.contains("blend.fillet_executive_p1"),
                "reason {reason}"
            );
            assert!(
                missing
                    .iter()
                    .any(|call| call.contains("SmApiChamferFillet")),
                "missing {missing:?}"
            );
            assert!(
                missing
                    .iter()
                    .any(|call| call.contains("SmFilletExecutive::CreateFilletCorners")),
                "missing {missing:?}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn exact_fillet_reports_the_missing_standard_solver() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let treatment = EdgeTreatment {
        mode: EdgeMode::Fillet,
        length: 2.0,
        ..EdgeTreatment::default()
    };
    let error = fillet_brep(&kernel, &brep, &selections[..1], &treatment).expect_err("blocked");
    match &error {
        BlendError::Unsupported {
            feature, missing, ..
        } => {
            assert_eq!(*feature, "exact circular fillet (G1)");
            assert!(missing.iter().any(|call| call.contains("SmApiFilletEdges")));
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn exact_g2_and_g3_name_the_law_machinery() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    for continuity in [Continuity::G2, Continuity::G3] {
        let treatment = EdgeTreatment {
            mode: EdgeMode::Fillet,
            continuity,
            length: 2.0,
            ..EdgeTreatment::default()
        };
        let error = apply_edge_treatment_brep(&kernel, &brep, &selections[..1], &treatment)
            .expect_err("blocked");
        match &error {
            BlendError::Unsupported { missing, .. } => {
                assert!(
                    missing
                        .iter()
                        .any(|call| call.contains("SmBSplineFilletLaw")),
                    "missing {missing:?}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
fn exact_path_validates_its_arguments() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");

    let fillet = EdgeTreatment {
        mode: EdgeMode::Fillet,
        ..EdgeTreatment::default()
    };
    let wrong_mode = chamfer_brep(&kernel, &brep, &selections[..1], &fillet).expect_err("mode");
    assert!(matches!(wrong_mode, BlendError::InvalidInput(_)));
    assert!(wrong_mode.to_string().contains("fillet treatment"));

    let chamfer = EdgeTreatment {
        mode: EdgeMode::Chamfer,
        chamfer_type: ChamferType::Equal,
        ..EdgeTreatment::default()
    };
    let wrong_mode = fillet_brep(&kernel, &brep, &selections[..1], &chamfer).expect_err("mode");
    assert!(matches!(wrong_mode, BlendError::InvalidInput(_)));

    let empty = apply_edge_treatment_brep(&kernel, &brep, &[], &chamfer).expect_err("empty");
    assert!(matches!(empty, BlendError::InvalidInput(_)));

    let bad_length = EdgeTreatment {
        length: f64::NAN,
        ..chamfer
    };
    let error =
        apply_edge_treatment_brep(&kernel, &brep, &selections[..1], &bad_length).expect_err("NaN");
    assert!(matches!(error, BlendError::InvalidInput(_)));
}

#[test]
fn blocker_lists_are_populated() {
    assert!(!port_units().is_empty());
    assert!(port_units().contains(&"blend.fillet_executive_p1"));
    let fillet = missing_calls(EdgeMode::Fillet, Continuity::Circular);
    assert!(fillet.iter().any(|call| call.contains("SmApiFilletEdges")));
    assert!(
        fillet
            .iter()
            .any(|call| call.contains("SetFilletParameters"))
    );
    let chamfer = missing_calls(EdgeMode::Chamfer, Continuity::Circular);
    assert!(
        chamfer
            .iter()
            .any(|call| call.contains("SmApiChamferFillet"))
    );
    let g3 = missing_calls(EdgeMode::Fillet, Continuity::G3);
    assert!(g3.iter().any(|call| call.contains("SmBSplineFilletLaw")));
    assert!(g3.len() > fillet.len());
}

#[test]
fn g2_multi_edge_corner_is_refused_not_silently_circular() {
    let (kernel, brep) = common::box_40_30_20();
    let edges = brep.edge_list();
    let vertex = brep.edge(edges[0]).expect("edge").vertices[0];
    let incident: Vec<_> = edges
        .iter()
        .copied()
        .filter(|edge| {
            brep.edge(*edge)
                .expect("edge")
                .vertices
                .contains(&vertex)
        })
        .collect();
    assert!(incident.len() >= 3, "box corner has three edges");
    let selections: Vec<_> = edge_selections(&kernel, &brep)
        .expect("selections")
        .into_iter()
        .filter(|selection| incident.contains(&selection.edge))
        .collect();
    assert_eq!(selections.len(), incident.len());
    let values = serde_json::json!({
        "mode": "fillet",
        "continuity": "G2",
        "length": 2.0,
        "tangentChain": false,
        "edges": selections
            .iter()
            .map(|selection| serde_json::json!({"center": selection.midpoint}))
            .collect::<Vec<_>>(),
    });
    let error = fillet_brep_exact(&brep, &values).expect_err("G2 corner must be refused");
    assert!(
        error.contains("matched G2 corner patch is not ported"),
        "unexpected error: {error}"
    );
}
