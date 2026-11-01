//! Round-trip tests for [`step_assemblies::write`]: export planar parts as a STEP
//! assembly and read them back with the crate's own parsers.
//!
//! All test names start with `write_` so `cargo test -p step-assemblies write` runs the
//! whole assembly-export suite.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use nodecraft_step::PlanarBody;
use step_assemblies::planar_instances;
use step_assemblies::write::{
    PlanarPart, UNSUPPORTED_ERROR_PREFIX, planar_assembly_text, planar_assembly_text_named,
    write_planar_assembly,
};
use step_assemblies::{Transform, parse_assembly_file};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn temp_path(tag: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "step-write-{}-{serial}-{tag}.step",
        std::process::id()
    ))
}

/// An axis-aligned box centred on `centre`, with outward-wound faces (same table as
/// `nodecraft_step::write_box`).
fn box_body(size: [f64; 3], centre: [f64; 3]) -> PlanarBody {
    let (x, y, z) = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0);
    let (cx, cy, cz) = (centre[0], centre[1], centre[2]);
    let points = vec![
        [cx - x, cy - y, cz - z],
        [cx + x, cy - y, cz - z],
        [cx + x, cy + y, cz - z],
        [cx - x, cy + y, cz - z],
        [cx - x, cy - y, cz + z],
        [cx + x, cy - y, cz + z],
        [cx + x, cy + y, cz + z],
        [cx - x, cy + y, cz + z],
    ];
    let faces = vec![
        vec![0, 3, 2, 1],
        vec![4, 5, 6, 7],
        vec![0, 1, 5, 4],
        vec![1, 2, 6, 5],
        vec![2, 3, 7, 6],
        vec![3, 0, 4, 7],
    ];
    PlanarBody { points, faces, holes: Vec::new() }
}

/// A 4x4 sheet at `z = 0` with a 1x1 square hole (area 15), as a single planar face.
fn plate_with_hole_body() -> PlanarBody {
    let points = vec![
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    PlanarBody {
        points,
        faces: vec![vec![0, 1, 2, 3]],
        holes: vec![vec![vec![4, 5, 6, 7]]],
    }
}

fn rotation_z_90(translation: [f64; 3]) -> Transform {
    // Right-handed: X -> +Y, Y -> -X, Z stays +Z.
    Transform::from_columns([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], translation)
}

fn translation(offset: [f64; 3]) -> Transform {
    Transform::from_columns([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], offset)
}

fn by_product<'a>(instances: &'a [step_assemblies::PlanarInstance], product: &str) -> &'a step_assemblies::PlanarInstance {
    instances
        .iter()
        .find(|instance| instance.product_name == product)
        .unwrap_or_else(|| panic!("no instance named {product}: {instances:#?}"))
}

fn quantized(point: [f64; 3]) -> [i64; 3] {
    point.map(|value| (value * 1e6).round() as i64)
}

fn assert_body_geometry(original: &PlanarBody, actual: &PlanarBody, context: &str) {
    assert_eq!(actual.faces.len(), original.faces.len(), "{context}: face count");
    let original_holes: Vec<usize> = (0..original.faces.len())
        .map(|face| original.face_holes(face).len())
        .collect();
    let actual_holes: Vec<usize> = (0..actual.faces.len())
        .map(|face| actual.face_holes(face).len())
        .collect();
    assert_eq!(actual_holes, original_holes, "{context}: hole counts");
    if let (Some(original_volume), Some(actual_volume)) =
        (original.volume().is_finite().then(|| original.volume()), actual.volume().is_finite().then(|| actual.volume()))
    {
        assert!(
            (original_volume - actual_volume).abs() < 1e-6 * original_volume.abs().max(1.0),
            "{context}: volume {actual_volume} != {original_volume}"
        );
    }
    assert!(
        (original.area() - actual.area()).abs() < 1e-6 * original.area().max(1.0),
        "{context}: area {} != {}",
        actual.area(),
        original.area()
    );
}

fn assert_transformed_points(original: &PlanarBody, transform: &Transform, actual: &PlanarBody, context: &str) {
    let mut expected: Vec<[i64; 3]> = original
        .points
        .iter()
        .map(|point| quantized(transform.transform_point(*point)))
        .collect();
    let mut found: Vec<[i64; 3]> = actual.points.iter().map(|point| quantized(*point)).collect();
    expected.sort_unstable();
    expected.dedup();
    found.sort_unstable();
    found.dedup();
    assert_eq!(found, expected, "{context}: transformed point sets differ");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn write_identity_assembly_round_trips_to_two_instances() {
    let parts = [
        PlanarPart::new("BLOCK_A", box_body([10.0, 4.0, 20.0], [0.0, 0.0, 0.0])),
        PlanarPart::new("BLOCK_B", box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])),
    ];
    let text = planar_assembly_text(&parts).expect("export text");
    assert_eq!(text.matches("MANIFOLD_SOLID_BREP").count(), 2);
    assert_eq!(text.matches("NEXT_ASSEMBLY_USAGE_OCCURRENCE").count(), 2);
    assert!(text.contains("SI_UNIT(.MILLI.,.METRE.)"), "no millimetre unit");
    assert!(text.contains("FILE_SCHEMA(('AUTOMOTIVE_DESIGN'))"), "no schema header");

    let path = temp_path("identity");
    write_planar_assembly(path.to_str().unwrap(), &parts).expect("write assembly");
    let file_text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(file_text, text, "file text differs from planar_assembly_text");

    let assembly = parse_assembly_file(&path).expect("parse exported assembly");
    assert_eq!(assembly.roots.len(), 1);
    assert_eq!(assembly.roots[0].product_name, "ASSEMBLY");
    assert_eq!(assembly.roots[0].children.len(), 2);
    assert_eq!(assembly.warnings, Vec::<String>::new());

    let instances = planar_instances(path.to_str().unwrap()).expect("planar instances");
    let _ = std::fs::remove_file(&path);
    assert_eq!(instances.len(), 2);
    for (index, original) in parts.iter().enumerate() {
        let instance = by_product(&instances, &original.name);
        assert!(instance.world_transform.is_identity(1e-9), "instance {index} moved");
        assert_body_geometry(&original.body, &instance.body, &original.name);
        assert_transformed_points(&original.body, &Transform::IDENTITY, &instance.body, &original.name);
        assert_eq!(instance.name, format!("Placement {index}"));
    }
}

#[test]
fn write_placed_assembly_round_trips_transforms() {
    let parts = [
        PlanarPart::placed(
            "TRANSLATED",
            box_body([10.0, 4.0, 20.0], [0.0, 0.0, 0.0]),
            translation([10.0, 0.0, 0.0]),
        ),
        PlanarPart::placed(
            "ROTATED",
            box_body([6.0, 6.0, 6.0], [0.0, 0.0, 0.0]),
            rotation_z_90([0.0, 20.0, 0.0]),
        ),
    ];
    let path = temp_path("placed");
    write_planar_assembly(path.to_str().unwrap(), &parts).unwrap();
    let instances = planar_instances(path.to_str().unwrap()).expect("planar instances");
    let _ = std::fs::remove_file(&path);
    assert_eq!(instances.len(), 2);

    let translated = by_product(&instances, "TRANSLATED");
    assert!(translated.world_transform.approx_eq(&parts[0].transform, 1e-9));
    assert_body_geometry(&parts[0].body, &translated.body, "TRANSLATED");
    assert_transformed_points(&parts[0].body, &parts[0].transform, &translated.body, "TRANSLATED");
    let (min, max) = translated.body.bounds().unwrap();
    assert!((min[0] - 5.0).abs() < 1e-6 && (max[0] - 15.0).abs() < 1e-6, "{min:?} {max:?}");

    let rotated = by_product(&instances, "ROTATED");
    assert!(rotated.world_transform.approx_eq(&parts[1].transform, 1e-9));
    assert_body_geometry(&parts[1].body, &rotated.body, "ROTATED");
    assert_transformed_points(&parts[1].body, &parts[1].transform, &rotated.body, "ROTATED");
    let (min, max) = rotated.body.bounds().unwrap();
    assert!((min[0] + 3.0).abs() < 1e-6, "x min {min:?}");
    assert!((max[1] - 23.0).abs() < 1e-6, "y max {max:?}");

}

#[test]
fn write_assembly_tree_reports_declared_world_transforms() {
    let parts = [
        PlanarPart::placed("A", box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]), translation([5.0, 0.0, 0.0])),
        PlanarPart::placed("B", box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]), rotation_z_90([0.0, 7.0, 0.0])),
    ];
    let path = temp_path("tree");
    write_planar_assembly(path.to_str().unwrap(), &parts).unwrap();
    let assembly = parse_assembly_file(path.to_str().unwrap()).expect("parse assembly");
    let _ = std::fs::remove_file(&path);
    assert_eq!(assembly.roots.len(), 1);
    let root = &assembly.roots[0];
    assert!(root.world_transform.is_identity(1e-12));
    assert_eq!(root.children.len(), 2);
    for (index, (child, part)) in root.children.iter().zip(parts.iter()).enumerate() {
        assert_eq!(child.product_name, part.name);
        assert!(
            child.world_transform.approx_eq(&part.transform, 1e-9),
            "{}: {:?} != {:?}",
            part.name,
            child.world_transform.rows(),
            part.transform.rows()
        );
        assert_eq!(child.name, format!("Placement {index}"));
    }
}

#[test]
fn write_holes_are_preserved() {
    let plate = plate_with_hole_body();
    assert_eq!(plate.faces.len(), 1);
    assert_eq!(plate.face_holes(0).len(), 1);
    let parts = [PlanarPart::new("PLATE", plate.clone())];
    let path = temp_path("holes");
    write_planar_assembly(path.to_str().unwrap(), &parts).unwrap();
    let instances = planar_instances(path.to_str().unwrap()).expect("planar instances");
    let _ = std::fs::remove_file(&path);
    assert_eq!(instances.len(), 1);
    let instance = &instances[0];
    assert_eq!(instance.body.faces.len(), 1);
    assert_eq!(instance.body.face_holes(0).len(), 1, "hole dropped");
    assert!(
        (instance.body.area() - plate.area()).abs() < 1e-9,
        "area {} != {}",
        instance.body.area(),
        plate.area()
    );
    assert_eq!(instance.body.holes, plate.holes);
    assert_eq!(instance.body.bounds(), plate.bounds());
}

#[test]
fn write_apostrophe_names_survive_the_round_trip() {
    let parts = [PlanarPart::new("O'Brien bracket", box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]))];
    let text = planar_assembly_text(&parts).expect("export text");
    assert!(text.contains("O''Brien bracket"), "quotes not escaped: {text}");
    let path = temp_path("names");
    write_planar_assembly(path.to_str().unwrap(), &parts).unwrap();
    let assembly = parse_assembly_file(path.to_str().unwrap()).expect("parse assembly");
    let instances = planar_instances(path.to_str().unwrap()).expect("planar instances");
    let _ = std::fs::remove_file(&path);
    assert_eq!(assembly.products[0].name, "O'Brien bracket");
    assert_eq!(assembly.roots[0].children[0].product_name, "O'Brien bracket");
    assert_eq!(instances[0].product_name, "O'Brien bracket");
}

#[test]
fn write_named_root_changes_the_assembly_product_name() {
    let parts = [PlanarPart::new("PART", box_body([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]))];
    let text = planar_assembly_text_named("MY ASSEMBLY", &parts).expect("export text");
    let path = temp_path("root-name");
    std::fs::write(&path, text).unwrap();
    let assembly = parse_assembly_file(path.to_str().unwrap()).expect("parse assembly");
    let _ = std::fs::remove_file(&path);
    assert_eq!(assembly.roots[0].product_name, "MY ASSEMBLY");
}

#[test]
fn write_repeated_body_instances_round_trip() {
    let body = box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
    let parts = [
        PlanarPart::placed("BLOCK", body.clone(), translation([0.0, 0.0, 0.0])),
        PlanarPart::placed("BLOCK", body.clone(), translation([10.0, 0.0, 0.0])),
        PlanarPart::placed("BLOCK", body.clone(), translation([20.0, 0.0, 0.0])),
    ];
    let text = planar_assembly_text(&parts).expect("export text");
    assert_eq!(text.matches("MANIFOLD_SOLID_BREP").count(), 3, "geometry not duplicated per instance");
    let path = temp_path("repeated");
    write_planar_assembly(path.to_str().unwrap(), &parts).unwrap();
    let instances = planar_instances(path.to_str().unwrap()).expect("planar instances");
    let _ = std::fs::remove_file(&path);
    assert_eq!(instances.len(), 3);
    for (index, instance) in instances.iter().enumerate() {
        let (min, max) = instance.body.bounds().unwrap();
        let expected_centre = index as f64 * 10.0;
        assert!((min[0] - (expected_centre - 1.0)).abs() < 1e-6, "{min:?}");
        assert!((max[0] - (expected_centre + 1.0)).abs() < 1e-6, "{max:?}");
    }
}

#[test]
fn write_rejects_non_rigid_transforms() {
    let parts = [PlanarPart::placed(
        "SCALED",
        box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]),
        Transform::from_rows([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap(),
    )];
    let error = planar_assembly_text(&parts).expect_err("scale must be rejected");
    assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("unit vector"), "{error}");
}

#[test]
fn write_rejects_mirrored_transforms() {
    let mirrored = Transform::from_columns(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
    );
    let parts = [PlanarPart::placed(
        "MIRRORED",
        box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]),
        mirrored,
    )];
    let error = planar_assembly_text(&parts).expect_err("mirror must be rejected");
    assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
    assert!(error.contains("left-handed"), "{error}");
}

#[test]
fn write_rejects_empty_input() {
    let error = planar_assembly_text(&[]).expect_err("no parts");
    assert!(error.contains("at least one part"), "{error}");
}

#[test]
fn write_rejects_invalid_bodies() {
    let mut missing_point = box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
    missing_point.faces[0][0] = 42;
    let error = planar_assembly_text(&[PlanarPart::new("BAD", missing_point)]).expect_err("index");
    assert!(error.contains("missing point"), "{error}");

    let mut short_face = box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
    short_face.faces[0] = vec![0, 1];
    let error = planar_assembly_text(&[PlanarPart::new("BAD", short_face)]).expect_err("short loop");
    assert!(error.contains("fewer than three"), "{error}");

    let mut empty_body = box_body([2.0, 2.0, 2.0], [0.0, 0.0, 0.0]);
    empty_body.faces.clear();
    let error = planar_assembly_text(&[PlanarPart::new("BAD", empty_body)]).expect_err("no faces");
    assert!(error.contains("empty"), "{error}");
}
