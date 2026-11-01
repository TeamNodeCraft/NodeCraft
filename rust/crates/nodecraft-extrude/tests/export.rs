//! Integration tests: exact extrude results exported through the planar StepBrep /
//! PlanarBody writers.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::kernel::Kernel;
use solidcore::mass::integrator::properties::{PropertyAttributes, brep_area, brep_volume};
use solidcore::math::{Point3, Vec3};
use solidcore::primitive::api_primitives as api;
use solidcore::query::api_queries::face_get_boundary_loops;
use solidcore::{Brep, FaceId};
use nodecraft_extrude::{
    ExtrudeError, extrude_planar_face_exact, extrude_planar_face_exact_for_export, face_plane,
    planar_brep_to_planar_body, planar_brep_to_step_brep, write_exact_extrude_step,
    write_planar_brep_step,
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

fn temp_step(tag: &str) -> PathBuf {
    let unique = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nodecraft-extrude-{tag}-{}-{unique}.step",
        std::process::id()
    ))
}

fn box_brep(kernel: &mut Kernel) -> Brep {
    api::create_box(kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 4.0, 20.0).expect("box")
}

fn face_with_normal(k: &Kernel, brep: &Brep, normal: Vec3) -> FaceId {
    for (face, _) in brep.faces().iter() {
        if let Ok(plane) = face_plane(k, brep, face) {
            if plane.normal.dot(normal) > 0.999 {
                return face;
            }
        }
    }
    panic!("no face with normal {normal:?}");
}

fn solidcore_measures(k: &Kernel, brep: &Brep) -> (f64, f64) {
    let store = AttributeStore::default();
    let attributes = PropertyAttributes {
        store: &store,
        key_of: &|_| ObjectKey(0),
    };
    let mut area_copy = brep.clone();
    let mut volume_copy = brep.clone();
    (
        brep_area(k, &mut area_copy, 1e-4).expect("area"),
        brep_volume(k, &mut volume_copy, 1e-4, &attributes).expect("volume"),
    )
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tolerance {tolerance})"
    );
}

fn planar_bounds(body: &nodecraft_step::PlanarBody) -> ([f64; 3], [f64; 3]) {
    body.bounds().expect("planar body bounds")
}

#[test]
fn exact_extrude_exports_as_step_brep_and_planar_body() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    let export = extrude_planar_face_exact_for_export(&mut k, &brep, top, 5.0)
        .expect("exact extrude + export forms");

    // 1. The topological STEP form is complete and valid.
    assert!(export.step_brep.validate().is_empty(), "{:?}", export.step_brep.validate());
    assert_eq!(export.step_brep.solids.len(), 1);
    assert_eq!(export.step_brep.shells.len(), 1);
    assert_eq!(export.step_brep.faces.len(), 6);
    assert_eq!(export.step_brep.edges.len(), 12);
    assert_eq!(export.step_brep.vertices.len(), 8);
    assert!(
        export.step_brep.summary().contains("1 solid(s), 1 shell(s), 6 face(s)"),
        "{}",
        export.step_brep.summary()
    );

    // 2. The PlanarBody form measures the exact box-like solid.
    assert_eq!(export.planar_body.faces.len(), 6);
    assert_eq!(export.planar_body.points.len(), 8);
    assert_eq!(export.planar_body.validate_closed(), Ok(()));
    assert_close(export.planar_body.try_volume().expect("planar volume"), 200.0, 1.0e-9);
    assert_close(export.planar_body.try_area().expect("planar area"), 220.0, 1.0e-9);
    let (min, max) = planar_bounds(&export.planar_body);
    for (actual, expected) in [
        (min[0], 0.0),
        (min[1], 0.0),
        (min[2], 20.0),
        (max[0], 10.0),
        (max[1], 4.0),
        (max[2], 25.0),
    ] {
        assert_close(actual, expected, 1.0e-9);
    }

    // 3. The source solidcore B-rep measures identically.
    let (area, volume) = solidcore_measures(&k, &export.brep);
    assert_close(area, 220.0, 1.0e-9);
    assert_close(volume, 200.0, 1.0e-9);

    // 4. `planar_brep_to_planar_body` is stable and agrees with the analogous
    //    nodecraft-solidcore bridge.
    let repeated = planar_brep_to_planar_body(&k, &export.brep).expect("planar body bridge");
    assert_eq!(repeated, export.planar_body);
    let analogous =
        nodecraft_solidcore::planar::planar_body(&k, &export.brep).expect("analogous bridge");
    assert_eq!(analogous.faces.len(), export.planar_body.faces.len());
    assert_eq!(analogous.points.len(), export.planar_body.points.len());
    assert_close(
        analogous.try_volume().expect("analogous volume"),
        export.planar_body.try_volume().expect("planar volume"),
        1.0e-9,
    );
    assert_close(
        analogous.try_area().expect("analogous area"),
        export.planar_body.try_area().expect("planar area"),
        1.0e-9,
    );
}

#[test]
fn exported_step_file_round_trips_through_the_planar_writer() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));
    let path = temp_step("roundtrip");
    let path_text = path.to_string_lossy().into_owned();

    let export = write_exact_extrude_step(&mut k, &brep, top, 5.0, &path_text)
        .expect("write exact extrude STEP");
    let text = std::fs::read_to_string(&path).expect("written STEP file");
    assert!(text.contains("MANIFOLD_SOLID_BREP"));
    assert!(text.contains("ADVANCED_FACE"));
    assert!(text.contains("PLANE"));
    assert!(text.contains("AXIS2_PLACEMENT_3D"));

    // Re-read through the PlanarBody reader: same closed solid and measures.
    let reread = nodecraft_step::read_planar_body(&path_text).expect("re-read planar body");
    assert_eq!(reread.faces.len(), 6);
    assert_eq!(reread.validate_closed(), Ok(()));
    assert_close(reread.try_volume().expect("reread volume"), 200.0, 1.0e-9);
    assert_close(reread.try_area().expect("reread area"), 220.0, 1.0e-9);
    let (min, max) = planar_bounds(&reread);
    for (actual, expected) in [
        (min[0], 0.0),
        (min[1], 0.0),
        (min[2], 20.0),
        (max[0], 10.0),
        (max[1], 4.0),
        (max[2], 25.0),
    ] {
        assert_close(actual, expected, 1.0e-9);
    }

    // Re-read through the topology reader: one solid/shell, 6 faces, 12 edges,
    // 8 vertices — the same topology as the exact Brep.
    let topology = nodecraft_step::read_topology_file(&path_text).expect("re-read topology");
    assert_eq!(topology.solids.len(), 1);
    assert_eq!(topology.shells.len(), 1);
    assert_eq!(topology.faces.len(), 6);
    assert_eq!(topology.edges.len(), 12);
    assert_eq!(topology.vertices.len(), 8);
    assert!(topology.validate().is_empty(), "{:?}", topology.validate());

    // The written file's measures match the in-memory PlanarBody form.
    assert_eq!(reread, export.planar_body);
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_planar_brep_step_writes_a_generic_planar_brep() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let side = face_with_normal(&k, &brep, Vec3::new(1.0, 0.0, 0.0));
    let extruded = extrude_planar_face_exact(&mut k, &brep, side, 2.0).expect("extrude");
    let path = temp_step("generic");
    let path_text = path.to_string_lossy().into_owned();

    write_planar_brep_step(&k, &extruded, &path_text).expect("write planar brep STEP");
    let reread = nodecraft_step::read_planar_body(&path_text).expect("re-read");
    assert_eq!(reread.faces.len(), 6);
    assert_close(reread.try_volume().expect("volume"), 160.0, 1.0e-9);
    assert_close(reread.try_area().expect("area"), 256.0, 1.0e-9);
    let (min, max) = planar_bounds(&reread);
    assert_close(min[0], 10.0, 1.0e-9);
    assert_close(max[0], 12.0, 1.0e-9);
    assert_close(min[2], 0.0, 1.0e-9);
    assert_close(max[2], 20.0, 1.0e-9);
    std::fs::remove_file(&path).ok();
}

#[test]
fn triangle_profile_exports_with_exact_measures() {
    let mut k = Kernel::default();
    let pyramid = api::create_pyramid(&mut k, Point3::new(0.0, 0.0, 0.0), 8.0, 6.0)
        .expect("pyramid");
    let mut triangle = None;
    for (face, _) in pyramid.faces().iter() {
        if let Ok(loops) = face_get_boundary_loops(&pyramid, face) {
            if loops.edge_counts == [3] {
                triangle = Some(face);
            }
        }
    }
    let triangle = triangle.expect("triangular side face");

    let profile = nodecraft_extrude::planar_face_profile(&k, &pyramid, triangle)
        .expect("triangle profile");
    let points = &profile.loops[0].points;
    let triangle_area = (points[1] - points[0])
        .cross(points[2] - points[0])
        .length()
        * 0.5;
    let perimeter = (points[1] - points[0]).length()
        + (points[2] - points[1]).length()
        + (points[0] - points[2]).length();

    let export = extrude_planar_face_exact_for_export(&mut k, &pyramid, triangle, 2.0)
        .expect("exact extrude + export");
    assert_eq!(export.step_brep.faces.len(), 5);
    assert_eq!(export.step_brep.edges.len(), 9);
    assert_eq!(export.step_brep.vertices.len(), 6);
    assert!(export.step_brep.validate().is_empty());
    assert_close(
        export.planar_body.try_volume().expect("volume"),
        triangle_area * 2.0,
        1.0e-9,
    );
    assert_close(
        export.planar_body.try_area().expect("area"),
        2.0 * triangle_area + perimeter * 2.0,
        1.0e-9,
    );
}

#[test]
fn curved_or_unwired_breps_are_rejected_for_planar_export() {
    let mut k = Kernel::default();

    // A cylinder has curved faces: not representable by the planar writer.
    let cylinder = api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0)
        .expect("cylinder");
    let error = planar_brep_to_step_brep(&k, &cylinder).expect_err("curved brep rejected");
    assert!(
        matches!(error, ExtrudeError::ExportUnsupported { .. }),
        "got {error}"
    );

    // An exact extrude of a circular profile has curved side faces: rejected too.
    let mut cap = None;
    for (face, _) in cylinder.faces().iter() {
        if let Ok(plane) = face_plane(&k, &cylinder, face) {
            if plane.normal.dot(Vec3::new(0.0, 0.0, 1.0)) > 0.999 {
                cap = Some(face);
            }
        }
    }
    let cap = cap.expect("cylinder cap");
    let curved_extrude =
        extrude_planar_face_exact(&mut k, &cylinder, cap, 3.0).expect("circular profile extrude");
    let error =
        planar_brep_to_step_brep(&k, &curved_extrude).expect_err("curved profile rejected");
    assert!(
        matches!(error, ExtrudeError::ExportUnsupported { .. }),
        "got {error}"
    );

    // The legacy capped linear sweep leaves live face records outside the face list;
    // export would drop them, so it is rejected explicitly.
    let box_brep = box_brep(&mut k);
    let top = face_with_normal(&k, &box_brep, Vec3::new(0.0, 0.0, 1.0));
    let legacy = nodecraft_extrude::extrude_planar_face_linear_sweep(
        &mut k, &box_brep, top, 5.0,
    )
    .expect("legacy sweep");
    match planar_brep_to_step_brep(&k, &legacy) {
        Err(ExtrudeError::ExportUnsupported { why }) => {
            assert!(why.contains("outside its face list"), "{why}");
        }
        Err(other) => panic!("expected ExportUnsupported, got {other}"),
        Ok(step) => {
            // Newer solidcore may retain only unused deleted face records, which the
            // planar writer can safely omit. Accept that only when the result is a
            // valid solid with measurable volume.
            let body = nodecraft_step::step_brep_to_planar_body(&step).expect("legacy export body");
            assert!(body.volume() > 0.0, "legacy export has no volume");
        }
    }
}
