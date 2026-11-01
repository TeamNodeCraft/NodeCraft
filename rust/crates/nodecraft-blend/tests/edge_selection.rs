//! Edge selection on a solidcore box.

mod common;

use nodecraft_blend::{
    BlendError, brep_center, edge_selections, select_edge_by_midpoint, select_edges_by_midpoints,
    select_edges_by_ordinals,
};

#[test]
fn box_has_twelve_edges_with_distinct_midpoints() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    assert_eq!(selections.len(), 12);
    let mut midpoints: Vec<[f64; 3]> = selections.iter().map(|s| s.midpoint).collect();
    midpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    midpoints.dedup_by(|a, b| a == b);
    assert_eq!(midpoints.len(), 12, "edge midpoints must be distinct");

    let mut lengths: Vec<f64> = selections.iter().map(|s| s.length).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut buckets = [0usize; 3];
    for length in &lengths {
        if (*length - 40.0).abs() < 1.0e-9 {
            buckets[0] += 1;
        } else if (*length - 30.0).abs() < 1.0e-9 {
            buckets[1] += 1;
        } else if (*length - 20.0).abs() < 1.0e-9 {
            buckets[2] += 1;
        } else {
            panic!("unexpected edge length {length}");
        }
    }
    assert_eq!(buckets, [4, 4, 4]);
    for selection in &selections {
        let direction = selection.direction().expect("edge direction");
        assert!((direction[0] * direction[0] + direction[1] * direction[1]
            + direction[2] * direction[2]
            - 1.0)
            .abs()
            < 1.0e-9);
    }
}

#[test]
fn selects_each_edge_by_its_approximate_midpoint() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    for expected in &selections {
        // "Approximate": offset the midpoint slightly, as a viewer pick would be.
        let jittered = [
            expected.midpoint[0] + 0.2,
            expected.midpoint[1] - 0.1,
            expected.midpoint[2] + 0.05,
        ];
        let selected = select_edge_by_midpoint(&kernel, &brep, jittered).expect("select");
        assert_eq!(
            selected.ordinal, expected.ordinal,
            "midpoint {:?} selected edge {} instead of {}",
            expected.midpoint, selected.ordinal, expected.ordinal
        );
        assert_eq!(selected, *expected);
    }
}

#[test]
fn selects_edges_by_ordinal_and_deduplicates() {
    let (kernel, brep) = common::box_40_30_20();
    let selections = edge_selections(&kernel, &brep).expect("selections");
    let selected =
        select_edges_by_ordinals(&kernel, &brep, &[3, 3, 7]).expect("ordinal selection");
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0], selections[3]);
    assert_eq!(selected[1], selections[7]);

    let by_midpoint =
        select_edges_by_midpoints(&kernel, &brep, &[selections[3].midpoint, selections[3].midpoint])
            .expect("midpoint selection");
    assert_eq!(by_midpoint.len(), 1);
    assert_eq!(by_midpoint[0], selections[3]);
}

#[test]
fn edge_selection_errors_are_explicit() {
    let (kernel, brep) = common::box_40_30_20();
    assert_eq!(brep_center(&kernel, &brep).expect("centre"), [20.0, 15.0, 10.0]);

    let out_of_range = select_edges_by_ordinals(&kernel, &brep, &[99]).expect_err("range");
    assert!(matches!(out_of_range, BlendError::InvalidInput(_)));
    assert!(out_of_range.to_string().contains("out of range"));

    // The middle of a face is not an edge pick.
    let far = select_edge_by_midpoint(&kernel, &brep, [20.0, 15.0, 20.0]).expect_err("far point");
    assert!(matches!(far, BlendError::InvalidInput(_)));
    assert!(far.to_string().contains("not within"));

    let empty = select_edges_by_ordinals(&kernel, &brep, &[]).expect_err("empty");
    assert!(matches!(empty, BlendError::InvalidInput(_)));
}

#[test]
fn selection_never_panics_on_bad_input() {
    let (kernel, brep) = common::box_40_30_20();
    let nan = select_edge_by_midpoint(&kernel, &brep, [f64::NAN, 0.0, 0.0]);
    assert!(nan.is_err(), "NaN pick must be an explicit error");
    let infinite = select_edge_by_midpoint(&kernel, &brep, [f64::INFINITY, 0.0, 0.0]);
    assert!(infinite.is_err());
}
