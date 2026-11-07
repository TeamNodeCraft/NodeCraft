//! Fixture tests for the upstream layer adapters.
//!
//! `box_10x6x4.json.gz` is copied from solidcore's
//! `crates/solidcore-tess/tests/data/box_10x6x4.json.gz` (Apache-2.0,
//! Tangency Labs, revision bff1569a4c3347ed9af67faab71578cadf84fde2). See
//! THIRD_PARTY_NOTICES.md.
use std::io::Read;

use flate2::read::GzDecoder;
use solidcore_record::brep::Brep;
use solidcore_tess::MeshTolerance;

fn fixture_record(name: &str) -> Brep {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(path).expect("fixture file");
    let mut json = String::new();
    GzDecoder::new(&bytes[..])
        .read_to_string(&mut json)
        .expect("gunzip fixture");
    let record: Brep = serde_json::from_str(&json).expect("fixture record");
    record.validate().expect("fixture record validates");
    record
}

#[test]
fn upstream_tess_meshes_a_record_box() {
    let record = fixture_record("box_10x6x4.json.gz");
    let display = nodecraft_layers::tessellate_record(
        &record,
        &MeshTolerance::display(0.05),
        12.33,
    )
    .expect("tessellate");

    assert_eq!(display.stats.faces, 6);
    assert!(!display.mesh.faces.is_empty());
    assert_eq!(display.mesh.vertices.len(), display.mesh.normals.len());
    assert_eq!(display.mesh.face_groups.len(), display.mesh.faces.len());
    let groups: std::collections::HashSet<_> = display.mesh.face_groups.iter().collect();
    assert_eq!(groups.len(), 6, "one face group per B-rep face");
    assert_eq!(display.edge_polylines.len(), 12, "one polyline per B-rep edge");
    for (edge, polyline) in display.edge_polylines.iter().enumerate() {
        for pair in polyline.windows(2) {
            assert!(
                pair[0] != pair[1],
                "edge {edge} has a zero-length segment at {:?}",
                pair[0]
            );
        }
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &display.mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    for (axis, expected) in [10.0f32, 6.0, 4.0].iter().enumerate() {
        assert!((min[axis] - 0.0).abs() < 1.0e-3, "min {axis}: {}", min[axis]);
        assert!((max[axis] - expected).abs() < 1.0e-3, "max {axis}: {}", max[axis]);
    }
}

#[test]
fn ops_shell_round_trips_a_kernel_box() {
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    let mut kernel = solidcore::kernel::Kernel::default();
    let mut brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0)
        .expect("box");
    let shelled = nodecraft_layers::shell_kernel(&mut kernel, &mut brep, &[], -1.0)
        .expect("ops shell");
    let mut record_kernel = solidcore::kernel::Kernel::default();
    let mut shelled = shelled;
    let (record, _order) = solidcore_convert::brep::to_record(&mut record_kernel, &mut shelled)
        .expect("shelled record");
    assert!(record.faces.len() > 6, "shell should add inner faces");
}

#[test]
fn ops_offset_faces_round_trips_a_kernel_box() {
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    let mut kernel = solidcore::kernel::Kernel::default();
    let mut brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0)
        .expect("box");
    let face = brep.faces().ids().next().expect("face");
    let offset = nodecraft_layers::offset_faces_kernel(&mut kernel, &mut brep, &[face], 1.0)
        .expect("ops offset_faces");
    assert!(!offset.faces().ids().next().is_none());
}

#[test]
fn display_lod_levels_reduce_triangles_with_a_finite_error() {
    let record = fixture_record("box_10x6x4.json.gz");
    let display = nodecraft_layers::tessellate_record(
        &record,
        &MeshTolerance::display(0.05),
        12.33,
    )
    .expect("tessellate");
    let full = display.mesh.faces.len();
    let levels = nodecraft_layers::simplify_display(&display, &[full / 2, full / 4]);
    assert!(!levels.is_empty());
    for level in &levels {
        assert!(level.error.is_finite());
        assert!(!level.triangles.is_empty());
        assert!(level.triangles.len() < full);
    }
}

#[test]
fn upstream_ops_primitives_round_trip() {
    let body = solidcore_ops::make_box([0.0, 0.0, 0.0], [10.0, 6.0, 4.0]).expect("box");
    assert_eq!(body.counts().expect("counts"), [6, 12, 8]);
    assert!((body.volume(1.0e-3).expect("volume") - 240.0).abs() < 1.0e-6);
    let bytes = body.to_smb(false).expect("smb");
    let round = solidcore_ops::Body::from_smb(bytes, false).expect("read smb");
    assert_eq!(round.counts().expect("round counts"), [6, 12, 8]);
    assert!((round.volume(1.0e-3).expect("round volume") - 240.0).abs() < 1.0e-6);
}

#[test]
fn kernel_brep_converts_to_record_and_tessellates() {
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives as api;

    assert_eq!(
        nodecraft_layers::record_bridge(),
        nodecraft_layers::RecordBridge::Available
    );
    let mut kernel = solidcore::kernel::Kernel::default();
    let mut brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0)
        .expect("box");
    let display = nodecraft_layers::tessellate_kernel(&mut kernel, &mut brep, 0.05)
        .expect("kernel tessellation");
    assert_eq!(display.stats.faces, 6);
    assert!(!display.mesh.faces.is_empty());
    assert_eq!(display.edge_polylines.len(), 12);
}

#[test]
fn ops_boolean_difference_returns_a_tessellatable_record() {
    use solidcore_ops::BooleanOp;

    let a = solidcore_ops::make_box([0.0, 0.0, 0.0], [10.0, 6.0, 4.0])
        .expect("box A")
        .to_record()
        .expect("record A")
        .0;
    let b = solidcore_ops::make_box([5.0, 0.0, 0.0], [10.0, 6.0, 4.0])
        .expect("box B")
        .to_record()
        .expect("record B")
        .0;
    let difference = nodecraft_layers::boolean_records(&a, &b, BooleanOp::Subtract)
        .expect("ops boolean")
        .expect("non-empty difference");
    let display = nodecraft_layers::tessellate_record(
        &difference,
        &MeshTolerance::display(0.05),
        10.0,
    )
    .expect("difference tessellation");
    assert_eq!(display.stats.faces, 6);
    assert!(!display.mesh.faces.is_empty());
    assert_eq!(display.mesh.face_groups.len(), display.mesh.faces.len());
}
