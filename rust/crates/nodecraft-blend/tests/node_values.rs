//! The NodeCraft Edge-node JSON surface and the end-to-end preview entry point.

mod common;

use serde_json::json;
use nodecraft_blend::{
    BlendError, ChamferType, Continuity, EdgeMode, EdgeTreatment, edge_treatment_preview,
};

#[test]
fn parses_the_node_values_the_app_writes() {
    let values = json!({
        "mode": "fillet",
        "chamferType": "equal",
        "continuity": "G3",
        "length": 3.5,
        "distance2": 1.25,
        "angle": 30.0
    });
    let treatment = EdgeTreatment::from_node_values(&values).expect("parse");
    assert_eq!(treatment.mode, EdgeMode::Fillet);
    assert_eq!(treatment.chamfer_type, ChamferType::Equal);
    assert_eq!(treatment.continuity, Continuity::G3);
    assert!((treatment.length - 3.5).abs() < 1.0e-12);
    assert_eq!(treatment.distance2, Some(1.25));
    assert_eq!(treatment.angle, Some(30.0));

    // Missing fields use the app defaults (mode chamfer, equal, circular, length 2).
    let defaults = EdgeTreatment::from_node_values(&json!({})).expect("defaults");
    assert_eq!(defaults, EdgeTreatment::default());
    assert_eq!(defaults.length, 2.0);
    assert_eq!(defaults.continuity, Continuity::Circular);
    assert_eq!(defaults.chamfer_type, ChamferType::Equal);
    assert_eq!(defaults.mode, EdgeMode::Chamfer);
}

#[test]
fn node_value_errors_are_explicit() {
    let unknown_mode = EdgeTreatment::from_node_values(&json!({"mode": "bevel"}))
        .expect_err("unknown mode");
    assert!(matches!(unknown_mode, BlendError::InvalidInput(_)));
    assert!(unknown_mode.to_string().contains("chamfer"));

    let bad_length = EdgeTreatment::from_node_values(&json!({"length": "2"}))
        .expect_err("string length");
    assert!(matches!(bad_length, BlendError::InvalidInput(_)));

    let not_an_object = EdgeTreatment::from_node_values(&json!([])).expect_err("array values");
    assert!(matches!(not_an_object, BlendError::InvalidInput(_)));

    let bad_continuity =
        EdgeTreatment::from_node_values(&json!({"continuity": "G4"})).expect_err("G4");
    assert!(matches!(bad_continuity, BlendError::InvalidInput(_)));
}

#[test]
fn node_values_drive_the_preview_end_to_end() {
    let (kernel, brep) = common::box_40_30_20();
    let values = json!({
        "mode": "chamfer",
        "chamferType": "equal",
        "continuity": "",
        "length": 2.0,
        "edges": [{ "ordinal": 0 }]
    });
    let mesh = edge_treatment_preview(&kernel, &brep, &values).expect("preview");
    assert!(common::count_group(&mesh, "blend:chamfer") > 0);
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);
    let expected = 40.0 * 30.0 * 20.0 - 0.5 * 2.0 * 2.0 * 30.0;
    assert!((common::volume(&mesh) - expected).abs() < 1.0);
}

#[test]
fn node_edges_can_be_referenced_by_centered_centre() {
    let (kernel, brep) = common::box_40_30_20();
    // Edge 0 of the box runs along +Y at x = 40, z = 20, so its midpoint is
    // (40, 15, 20) in B-rep coordinates and (20, 0, 10) in the centered mesh
    // coordinates the viewer reports.
    let values = json!({
        "mode": "chamfer",
        "length": 2.0,
        "edges": [{ "center": [20.0, 0.0, 10.0] }]
    });
    let mesh = edge_treatment_preview(&kernel, &brep, &values).expect("preview");
    assert!(common::count_group(&mesh, "blend:chamfer") > 0);
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);

    // An object-form point works too.
    let values = json!({
        "mode": "chamfer",
        "length": 2.0,
        "edges": [{ "point": { "x": 20.0, "y": 0.0, "z": 10.0 } }]
    });
    let mesh = edge_treatment_preview(&kernel, &brep, &values).expect("preview");
    let (_, bad) = common::closure_report(&mesh);
    assert_eq!(bad, 0);
}

#[test]
fn node_edge_errors_are_explicit() {
    let (kernel, brep) = common::box_40_30_20();

    let no_edges = edge_treatment_preview(
        &kernel,
        &brep,
        &json!({"mode": "chamfer", "edges": []}),
    )
    .expect_err("no edges");
    assert!(matches!(no_edges, BlendError::InvalidInput(_)));
    assert!(no_edges.to_string().contains("no edges selected"));

    let missing_edges = edge_treatment_preview(&kernel, &brep, &json!({"mode": "chamfer"}))
        .expect_err("missing edges");
    assert!(matches!(missing_edges, BlendError::InvalidInput(_)));

    let bad_ordinal = edge_treatment_preview(
        &kernel,
        &brep,
        &json!({"edges": [{"ordinal": 99}]}),
    )
    .expect_err("bad ordinal");
    assert!(matches!(bad_ordinal, BlendError::InvalidInput(_)));

    let bad_reference = edge_treatment_preview(
        &kernel,
        &brep,
        &json!({"edges": [{"name": "edge-3"}]}),
    )
    .expect_err("bad reference");
    assert!(matches!(bad_reference, BlendError::InvalidInput(_)));
}

#[test]
fn g2_and_g3_node_fillets_report_the_missing_law_api() {
    let (kernel, brep) = common::box_40_30_20();
    for continuity in ["G2", "G3"] {
        let values = json!({
            "mode": "fillet",
            "continuity": continuity,
            "length": 2.0,
            "edges": [{ "ordinal": 0 }]
        });
        let error = edge_treatment_preview(&kernel, &brep, &values).expect_err("continuity");
        match error {
            BlendError::Unsupported { feature, missing, .. } => {
                assert_eq!(feature, "fillet G2/G3");
                assert!(
                    missing.iter().any(|call| call.contains("SmFilletLaw")),
                    "missing {missing:?}"
                );
            }
            other => panic!("expected Unsupported for {continuity}, got {other:?}"),
        }
    }
}

#[test]
fn malformed_node_values_never_panic() {
    let (kernel, brep) = common::box_40_30_20();
    let cases = [
        json!({"edges": "0"}),
        json!({"edges": [{}]}),
        json!({"edges": [{"center": [1.0, 2.0]}]}),
        json!({"edges": [{"center": ["x", 0.0, 0.0]}]}),
        json!({"edges": [{"ordinal": -1}]}),
        json!({"edges": [{"ordinal": 1.5}]}),
    ];
    for case in cases {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            edge_treatment_preview(&kernel, &brep, &case)
        }));
        let result = outcome.expect("node values must not panic");
        assert!(result.is_err(), "expected an error for {case}");
    }
}
