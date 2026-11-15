//! Round-trip and assembly behaviour through the CADIR adapter.

use std::path::PathBuf;

use nodecraft_step::{
    CadImport, ExportBody, StepExportOptions, read_cad, record_of_brep, record_volume, write_step,
    write_step_faceted,
};
use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives as api;

fn temp_path(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nodecraft-step-{}-{serial}-{tag}.step",
        std::process::id()
    ))
}

fn read(path: &std::path::Path) -> CadImport {
    let bytes = std::fs::read(path).expect("read written STEP");
    read_cad(&bytes, path.to_str().unwrap()).expect("read written STEP back")
}

fn box_brep(size: f64) -> solidcore::topology::Brep {
    let mut kernel = Kernel::default();
    api::create_box(
        &mut kernel,
        Point3::new(-size / 2.0, -size / 2.0, -size / 2.0),
        size,
        size,
        size,
    )
    .expect("box")
}

fn translation(x: f64) -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [x, 0.0, 0.0, 1.0],
    ]
}

#[test]
fn a_box_round_trips() {
    let path = temp_path("box");
    let body = box_brep(10.0);
    nodecraft_step::write_brep_exact(path.to_str().unwrap(), &body).expect("write");
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 1, "{:?}", import.failures);
    assert_eq!(import.bodies[0].brep.faces.len(), 6);
    assert!((record_volume(&import.bodies[0].brep).unwrap() - 1000.0).abs() < 1e-6);
}

#[test]
fn a_cylinder_round_trips() {
    let path = temp_path("cylinder");
    let mut kernel = Kernel::default();
    let body = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 3.0, 8.0).unwrap();
    nodecraft_step::write_brep_exact(path.to_str().unwrap(), &body).expect("write");
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 1, "{:?}", import.failures);
    assert_eq!(import.bodies[0].brep.faces.len(), 3);
    let expected = std::f64::consts::PI * 9.0 * 8.0;
    let volume = record_volume(&import.bodies[0].brep).unwrap();
    assert!((volume - expected).abs() <= expected * 1e-6);
}

#[test]
fn a_multi_body_part_round_trips_with_names_colours_and_placements() {
    let record_a = record_of_brep(&box_brep(10.0)).unwrap();
    let record_b = record_of_brep(&box_brep(4.0)).unwrap();
    let bodies = vec![
        ExportBody {
            name: "red".into(),
            color: Some([1.0, 0.0, 0.0]),
            placement: Some(translation(100.0)),
            sheet: false,
            brep: record_a,
        },
        ExportBody {
            name: "blue".into(),
            color: Some([0.0, 0.0, 1.0]),
            placement: Some(translation(200.0)),
            sheet: false,
            brep: record_b,
        },
    ];
    let path = temp_path("multi");
    write_step(path.to_str().unwrap(), &bodies, "multi", "ap242").expect("write");
    let text = std::fs::read_to_string(&path).unwrap();
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 2, "{:?}", import.failures);
    let names: Vec<_> = import.bodies.iter().map(|body| body.name.as_str()).collect();
    assert!(names.contains(&"red") && names.contains(&"blue"), "{names:?}");
    let colors: Vec<_> = import.bodies.iter().filter_map(|body| body.color).collect();
    assert_eq!(colors.len(), 2, "{:?}", import.bodies);
    assert!(colors.iter().any(|c| c[0] > 0.9 && c[2] < 0.1), "{colors:?}");
    assert!(colors.iter().any(|c| c[2] > 0.9 && c[0] < 0.1), "{colors:?}");
    let translations: Vec<f64> = import
        .bodies
        .iter()
        .filter_map(|body| body.placement)
        .map(|placement| placement[3][0])
        .collect();
    assert!(translations.iter().any(|x| (*x - 100.0).abs() < 1e-9), "{translations:?}");
    assert!(translations.iter().any(|x| (*x - 200.0).abs() < 1e-9), "{translations:?}");
    assert!(text.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE"), "not written as an assembly");
    assert!(
        text.contains("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"),
        "no placed occurrence"
    );
    assert!(text.contains("ITEM_DEFINED_TRANSFORMATION"));
}

#[test]
fn nested_occurrences_compose_their_placements() {
    let record_a = record_of_brep(&box_brep(2.0)).unwrap();
    let record_b = record_of_brep(&box_brep(2.0)).unwrap();
    let bodies = vec![
        ExportBody {
            name: "A".into(),
            color: None,
            placement: Some(translation(10.0)),
            sheet: false,
            brep: record_a,
        },
        ExportBody {
            name: "C".into(),
            color: None,
            placement: None,
            sheet: false,
            brep: record_b,
        },
    ];
    let path = temp_path("nested");
    let mut ir = nodecraft_step::cadir_export::cadir(&bodies, "nested").expect("cadir");
    // Make C a child of A. C's local placement is 5, so its world placement is 15.
    let parent = ir.model.occurrences[0].id.clone();
    let child = &mut ir.model.occurrences[1];
    child.parent = cadmpeg_ir::products::OccurrenceParent::Occurrence {
        occurrence: parent,
    };
    child.transform = cadmpeg_ir::transform::Transform::affine([
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    .expect("translate");
    // Write the edited IR directly through cadmpeg.
    let bytes = encode(&ir);
    std::fs::write(&path, bytes).expect("write");
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 2, "{:?}", import.failures);
    let translations: Vec<f64> = import
        .bodies
        .iter()
        .filter_map(|body| body.placement)
        .map(|placement| placement[3][0])
        .collect();
    assert!(translations.iter().any(|x| (*x - 10.0).abs() < 1e-6), "{translations:?}");
    assert!(translations.iter().any(|x| (*x - 15.0).abs() < 1e-6), "{translations:?}");
}

fn encode(ir: &cadmpeg_ir::CadIr) -> Vec<u8> {
    use cadmpeg_codec_step::{StepCodec, StepWriteOptions};
    use cadmpeg_ir::codec::write::target::TargetRequest;
    use cadmpeg_ir::codec::write::{EncodeInput, Encoder};
    let plan = StepCodec {
        options: StepWriteOptions::default(),
    }
    .plan(EncodeInput::new(ir, None), TargetRequest::Explicit("ap242e1"))
    .expect("plan");
    let mut bytes = Vec::new();
    plan.write_to(&mut bytes).expect("write");
    bytes
}

#[test]
fn a_faceted_mesh_round_trips_through_cadir() {
    let vertices = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let triangles = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let path = temp_path("faceted");
    let options = StepExportOptions {
        product_name: "faceted".into(),
        ..StepExportOptions::default()
    };
    write_step_faceted(path.to_str().unwrap(), &vertices, &triangles, &options).expect("write");
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 1, "{:?}", import.failures);
    assert_eq!(import.bodies[0].brep.faces.len(), 12);
    let volume = record_volume(&import.bodies[0].brep).unwrap().abs();
    assert!((volume - 8.0).abs() < 1e-4, "volume {volume}");
}

#[test]
fn an_imported_planar_body_reexports_with_volume_and_placement() {
    let block = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/Sample AP203 Data Files/block.stp");
    let bytes = std::fs::read(&block).unwrap();
    let import = read_cad(&bytes, block.to_str().unwrap()).unwrap();
    assert_eq!(import.bodies.len(), 1, "{:?}", import.failures);
    let body = &import.bodies[0];
    assert!((record_volume(&body.brep).unwrap() - 60.0).abs() < 1e-6);
    for placement in [None, Some(translation(10.0))] {
        let bodies = vec![ExportBody {
            name: body.name.clone(),
            color: body.color,
            placement,
            sheet: body.sheet,
            brep: body.brep.clone(),
        }];
        let path = temp_path("block-reexport");
        let options = StepExportOptions {
            product_name: "block".into(),
            ..StepExportOptions::default()
        };
        nodecraft_step::write_step_with_options(path.to_str().unwrap(), &bodies, &options)
            .expect("re-export");
        let back = read(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.bodies.len(), 1, "{:?}", back.failures);
        assert_eq!(back.bodies[0].brep.faces.len(), 6);
        assert_eq!(back.bodies[0].turned, Vec::<u32>::new());
        assert!(back.bodies[0].notes.is_empty(), "{:?}", back.bodies[0].notes);
        assert!((record_volume(&back.bodies[0].brep).unwrap() - 60.0).abs() < 1e-6);
        assert_eq!(back.bodies[0].placement, placement);
    }
}


// Curved, blended, boolean and spline round trips: export -> cadmpeg import ->
// read_bodies keeps one body with the same volume (and face count where it is fixed).

fn round_trip_one(tag: &str, brep: &solidcore::topology::Brep) -> CadImport {
    let path = temp_path(tag);
    nodecraft_step::write_brep_exact(path.to_str().unwrap(), brep).expect("write");
    let import = read(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(import.bodies.len(), 1, "{tag}: {:?}", import.failures);
    import
}

fn assert_volume(tag: &str, import: &CadImport, expected: f64) {
    let volume = record_volume(&import.bodies[0].brep).unwrap();
    assert!(
        (volume - expected).abs() <= expected.abs() * 1e-6,
        "{tag}: volume {volume}, expected {expected}"
    );
}

fn bbox(brep: &solidcore_record::brep::Brep) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &brep.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.point[axis]);
            max[axis] = max[axis].max(vertex.point[axis]);
        }
    }
    (min, max)
}

#[test]
fn a_box_keeps_its_bounding_box() {
    let import = round_trip_one("box-bbox", &box_brep(10.0));
    assert_eq!(bbox(&import.bodies[0].brep), ([-5.0; 3], [5.0; 3]));
}

#[test]
fn a_sphere_round_trips() {
    let mut kernel = Kernel::default();
    let body = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), 5.0).unwrap();
    let import = round_trip_one("sphere", &body);
    assert_eq!(import.bodies[0].brep.faces.len(), 1);
    assert_volume("sphere", &import, 4.0 / 3.0 * std::f64::consts::PI * 125.0);
}

#[test]
fn a_torus_round_trips() {
    let mut kernel = Kernel::default();
    let body = api::create_torus(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 2.0).unwrap();
    let import = round_trip_one("torus", &body);
    assert_eq!(import.bodies[0].brep.faces.len(), 1);
    let pi = std::f64::consts::PI;
    assert_volume("torus", &import, 2.0 * pi * pi * 10.0 * 4.0);
}

#[test]
fn a_cone_round_trips() {
    let mut kernel = Kernel::default();
    let body = api::create_cone(&mut kernel, Point3::new(0.0, 0.0, 0.0), 5.0, 2.0, 10.0).unwrap();
    let import = round_trip_one("cone", &body);
    assert_volume("cone", &import, std::f64::consts::PI * 10.0 / 3.0 * (25.0 + 10.0 + 4.0));
}

fn blended(kind: nodecraft_ops::OpsBlend) -> solidcore::topology::Brep {
    use nodecraft_ops::OpsSession;
    let mut kernel = nodecraft_kernel_profile::kernel();
    let block = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).unwrap();
    let edge = block.edge_list()[0];
    OpsSession::new(&mut kernel)
        .blend(&block, &[edge], 1.0, kind)
        .expect("blend")
        .brep
}

#[test]
fn a_fillet_result_round_trips() {
    let result = blended(nodecraft_ops::OpsBlend::Round(nodecraft_ops::CornerTurn::Mitre));
    let expected = record_volume(&record_of_brep(&result).unwrap()).unwrap();
    let import = round_trip_one("fillet", &result);
    assert_volume("fillet", &import, expected);
}

#[test]
fn a_chamfer_result_round_trips() {
    let result = blended(nodecraft_ops::OpsBlend::Chamfer(nodecraft_ops::ChamferCorner::Mitre));
    let expected = record_volume(&record_of_brep(&result).unwrap()).unwrap();
    let import = round_trip_one("chamfer", &result);
    assert_volume("chamfer", &import, expected);
}

#[test]
fn a_boolean_result_round_trips() {
    use solidcore::boolean::api_brep::boolean_difference;
    let mut kernel = Kernel::default();
    let block = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let hole = api::create_cylinder(&mut kernel, Point3::new(5.0, 5.0, -1.0), 2.0, 12.0).unwrap();
    let result = boolean_difference(&mut kernel, block, hole).expect("boolean difference");
    let import = round_trip_one("boolean", &result);
    assert_volume("boolean", &import, 1000.0 - std::f64::consts::PI * 4.0 * 10.0);
}

#[test]
fn a_nurbs_extruded_profile_round_trips() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../samples/gg.step");
    if !path.is_file() {
        eprintln!("skipping a_nurbs_extruded_profile_round_trips: samples/gg.step not present");
        return;
    }
    let before = read(&path);
    assert_eq!(before.bodies.len(), 1, "{:?}", before.failures);
    let body = &before.bodies[0];
    assert_eq!(body.brep.faces.len(), 255);
    let expected = record_volume(&body.brep).unwrap();
    let bodies = vec![ExportBody {
        name: body.name.clone(),
        color: body.color,
        placement: body.placement,
        sheet: body.sheet,
        brep: body.brep.clone(),
    }];
    let out = temp_path("nurbs");
    write_step(out.to_str().unwrap(), &bodies, "nurbs", "ap242").expect("write");
    let after = read(&out);
    let _ = std::fs::remove_file(&out);
    assert_eq!(after.bodies.len(), 1, "{:?}", after.failures);
    assert_eq!(after.bodies[0].brep.faces.len(), 255);
    assert_volume("nurbs", &after, expected);
}
