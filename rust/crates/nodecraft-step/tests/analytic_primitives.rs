//! Analytic curved-primitive writers: structural checks plus a smlibpy round-trip.
//!
//! The structural half always runs. The round-trip half imports every written file
//! through `smlibpy.import_step(..., heal=False)` and compares faces/edges/vertices,
//! volume, area and bounds with the closed-form values for each primitive. It needs the
//! workspace `.venv`; when that interpreter is unavailable the test reports a skip
//! instead of a failure (the run above it still proves the file structure).

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nodecraft_step::{
    StepFile, Value, write_cone, write_cylinder, write_sphere, write_torus,
};

struct Expected {
    faces: usize,
    /// Kernel edges reported by smlibpy (seams/degenerate edges are not kernel edges).
    edges: usize,
    vertices: usize,
    /// `EDGE_CURVE` entities in the STEP file (includes seam edges).
    edge_curves: usize,
    circles: usize,
    lines: usize,
    surface: &'static str,
    /// Numeric arguments expected on the analytic surface entity.
    surface_args: Vec<f64>,
    volume: f64,
    area: f64,
    min: [f64; 3],
    max: [f64; 3],
}

struct Case {
    name: &'static str,
    path: PathBuf,
    expected: Expected,
}

fn output_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nodecraft-step-analytic-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn write_cases() -> Vec<Case> {
    let dir = output_dir();
    let mut cases = Vec::new();
    let mut push = |name: &'static str, path: PathBuf, expected: Expected| {
        // The writer itself is exercised by every call below.
        cases.push(Case { name, path, expected });
    };

    let cylinder_path = dir.join("cylinder.step");
    write_cylinder(cylinder_path.to_str().unwrap(), 5.0, 10.0).expect("cylinder");
    push(
        "cylinder",
        cylinder_path,
        Expected {
            faces: 3,
            edges: 2,
            vertices: 2,
            edge_curves: 3,
            circles: 2,
            lines: 1,
            surface: "CYLINDRICAL_SURFACE",
            surface_args: vec![5.0],
            volume: PI * 5.0 * 5.0 * 10.0,
            area: 2.0 * PI * 25.0 + 2.0 * PI * 5.0 * 10.0,
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    let cylinder_decimal = dir.join("cylinder-decimal.step");
    write_cylinder(cylinder_decimal.to_str().unwrap(), 2.5, 4.0).expect("cylinder decimal");
    push(
        "cylinder-decimal",
        cylinder_decimal,
        Expected {
            faces: 3,
            edges: 2,
            vertices: 2,
            edge_curves: 3,
            circles: 2,
            lines: 1,
            surface: "CYLINDRICAL_SURFACE",
            surface_args: vec![2.5],
            volume: PI * 2.5 * 2.5 * 4.0,
            area: 2.0 * PI * 6.25 + 2.0 * PI * 2.5 * 4.0,
            min: [-2.5, -2.5, -2.0],
            max: [2.5, 2.5, 2.0],
        },
    );

    // Frustum: base radius larger than top radius.
    let cone_frustum = dir.join("cone-frustum.step");
    write_cone(cone_frustum.to_str().unwrap(), 5.0, 2.0, 10.0).expect("cone frustum");
    push(
        "cone-frustum",
        cone_frustum,
        Expected {
            faces: 3,
            edges: 2,
            vertices: 2,
            edge_curves: 3,
            circles: 2,
            lines: 1,
            surface: "CONICAL_SURFACE",
            surface_args: vec![5.0, ((5.0f64 - 2.0) / 10.0).atan()],
            volume: PI * 10.0 / 3.0 * (25.0 + 10.0 + 4.0),
            area: PI * (25.0 + 4.0)
                + PI * 7.0 * ((5.0f64 - 2.0).powi(2) + 100.0).sqrt(),
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    // Inverted frustum: top radius larger than base radius (the surface is placed at
    // the top with its axis pointing up).
    let cone_inverted = dir.join("cone-inverted.step");
    write_cone(cone_inverted.to_str().unwrap(), 2.0, 5.0, 10.0).expect("cone inverted");
    push(
        "cone-inverted",
        cone_inverted,
        Expected {
            faces: 3,
            edges: 2,
            vertices: 2,
            edge_curves: 3,
            circles: 2,
            lines: 1,
            surface: "CONICAL_SURFACE",
            surface_args: vec![5.0, ((5.0f64 - 2.0) / 10.0).atan()],
            volume: PI * 10.0 / 3.0 * (4.0 + 10.0 + 25.0),
            area: PI * (4.0 + 25.0)
                + PI * 7.0 * ((5.0f64 - 2.0).powi(2) + 100.0).sqrt(),
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    // Apex cone (top radius zero): lateral face meets the apex vertex, no top cap.
    let cone_apex = dir.join("cone-apex.step");
    write_cone(cone_apex.to_str().unwrap(), 5.0, 0.0, 10.0).expect("cone apex");
    push(
        "cone-apex",
        cone_apex,
        Expected {
            faces: 2,
            edges: 1,
            vertices: 2,
            edge_curves: 2,
            circles: 1,
            lines: 1,
            surface: "CONICAL_SURFACE",
            surface_args: vec![5.0, (5.0f64 / 10.0).atan()],
            volume: PI * 25.0 * 10.0 / 3.0,
            area: PI * 25.0 + PI * 5.0 * (25.0f64 + 100.0).sqrt(),
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    // Apex at the bottom, base at the top: the CONICAL_SURFACE is placed at the wide
    // (top) end with its axis pointing up and the semi-angle measured from there.
    let cone_apex_bottom = dir.join("cone-apex-bottom.step");
    write_cone(cone_apex_bottom.to_str().unwrap(), 0.0, 5.0, 10.0).expect("cone apex bottom");
    push(
        "cone-apex-bottom",
        cone_apex_bottom,
        Expected {
            faces: 2,
            edges: 1,
            vertices: 2,
            edge_curves: 2,
            circles: 1,
            lines: 1,
            surface: "CONICAL_SURFACE",
            surface_args: vec![5.0, (5.0f64 / 10.0).atan()],
            volume: PI * 25.0 * 10.0 / 3.0,
            area: PI * 25.0 + PI * 5.0 * (25.0f64 + 100.0).sqrt(),
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    // Equal radii: a valid zero-taper cone (cylinder-like, zero semi-angle).
    let cone_equal = dir.join("cone-equal.step");
    write_cone(cone_equal.to_str().unwrap(), 3.0, 3.0, 6.0).expect("cone equal");
    push(
        "cone-equal",
        cone_equal,
        Expected {
            faces: 3,
            edges: 2,
            vertices: 2,
            edge_curves: 3,
            circles: 2,
            lines: 1,
            surface: "CONICAL_SURFACE",
            surface_args: vec![3.0, 0.0],
            volume: PI * 9.0 * 6.0,
            area: 2.0 * PI * 9.0 + 2.0 * PI * 3.0 * 6.0,
            min: [-3.0, -3.0, -3.0],
            max: [3.0, 3.0, 3.0],
        },
    );

    let sphere_path = dir.join("sphere.step");
    write_sphere(sphere_path.to_str().unwrap(), 5.0).expect("sphere");
    push(
        "sphere",
        sphere_path,
        Expected {
            faces: 1,
            edges: 0,
            vertices: 2,
            edge_curves: 1,
            circles: 1,
            lines: 0,
            surface: "SPHERICAL_SURFACE",
            surface_args: vec![5.0],
            volume: 4.0 / 3.0 * PI * 125.0,
            area: 4.0 * PI * 25.0,
            min: [-5.0, -5.0, -5.0],
            max: [5.0, 5.0, 5.0],
        },
    );

    let torus_path = dir.join("torus.step");
    write_torus(torus_path.to_str().unwrap(), 8.0, 2.0).expect("torus");
    push(
        "torus",
        torus_path,
        Expected {
            faces: 1,
            edges: 0,
            vertices: 1,
            edge_curves: 2,
            circles: 2,
            lines: 0,
            surface: "TOROIDAL_SURFACE",
            surface_args: vec![8.0, 2.0],
            volume: 2.0 * PI * PI * 8.0 * 4.0,
            area: 4.0 * PI * PI * 8.0 * 2.0,
            min: [-10.0, -10.0, -2.0],
            max: [10.0, 10.0, 2.0],
        },
    );

    cases
}

fn entity_names(file: &StepFile) -> std::collections::BTreeMap<&str, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for entity in file.entities.values() {
        *counts.entry(entity.name.as_str()).or_insert(0) += 1;
    }
    counts
}

fn numeric_args(entity: &nodecraft_step::Entity) -> Vec<f64> {
    entity
        .args
        .iter()
        .filter_map(|value| match value {
            Value::Number(number) => Some(*number),
            _ => None,
        })
        .collect()
}

/// Walk every value and assert each `#ref` resolves to an entity in the file.
fn assert_references_resolve(file: &StepFile) {
    for (id, entity) in &file.entities {
        fn walk(file: &StepFile, owner: u32, values: &[Value]) {
            for value in values {
                match value {
                    Value::Ref(reference) => {
                        assert!(
                            file.entities.contains_key(reference),
                            "entity #{owner} references missing #{reference}"
                        );
                    }
                    Value::List(values) | Value::Typed(_, values) => walk(file, owner, values),
                    _ => {}
                }
            }
        }
        walk(file, *id, &entity.args);
    }
}

#[test]
fn files_have_expected_topology_and_analytic_surfaces() {
    for case in write_cases() {
        let text = fs::read_to_string(&case.path).expect("read step");
        let file = StepFile::parse(&text).expect("parse step");
        let counts = entity_names(&file);
        let expected = &case.expected;
        let count = |name: &str| *counts.get(name).unwrap_or(&0);

        assert_eq!(count("ADVANCED_FACE"), expected.faces, "{} faces", case.name);
        assert_eq!(
            count("VERTEX_POINT"),
            expected.vertices,
            "{} vertices",
            case.name
        );
        assert_eq!(
            count("EDGE_CURVE"),
            expected.edge_curves,
            "{} edge curves",
            case.name
        );
        assert_eq!(count("CIRCLE"), expected.circles, "{} circles", case.name);
        assert_eq!(count("LINE"), expected.lines, "{} lines", case.name);
        assert_eq!(count("CLOSED_SHELL"), 1, "{} shell", case.name);
        assert_eq!(
            count("MANIFOLD_SOLID_BREP"),
            1,
            "{} solid",
            case.name
        );
        assert_eq!(
            count("ADVANCED_BREP_SHAPE_REPRESENTATION"),
            1,
            "{} representation",
            case.name
        );
        assert_eq!(
            count(expected.surface),
            1,
            "{} analytic surface",
            case.name
        );
        // Circle edges must be walkable from a face: every face references a bound.
        assert_eq!(count("FACE_OUTER_BOUND"), expected.faces, "{} bounds", case.name);
        assert_eq!(count("EDGE_LOOP"), expected.faces, "{} loops", case.name);

        let surface = file
            .entities
            .values()
            .find(|entity| entity.name == expected.surface)
            .expect("surface entity");
        let args = numeric_args(surface);
        assert_eq!(args.len(), expected.surface_args.len(), "{} args", case.name);
        for (actual, wanted) in args.iter().zip(&expected.surface_args) {
            assert!(
                (actual - wanted).abs() <= 1e-12 * wanted.abs().max(1.0),
                "{} surface arg {actual} != {wanted}",
                case.name
            );
        }
        assert_references_resolve(&file);
        for entity in file.entities.values() {
            for value in numeric_args(entity) {
                assert!(value.is_finite(), "{} non-finite number", case.name);
            }
        }
    }
}

/// Locate the workspace `.venv` interpreter.
fn python() -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("NODECRAFT_PYTHON") {
        let path = PathBuf::from(configured);
        return path.exists().then_some(path);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.ancestors().nth(3)?;
    let candidate = root.join(".venv").join("bin").join("python");
    candidate.exists().then_some(candidate)
}

const IMPORT_SCRIPT: &str = r"
import glob, os, sys
root = sys.argv[1]
for candidate in glob.glob(os.path.join(root, '.venv', 'lib', 'python*', 'site-packages')):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
import smlibpy
for path in sys.argv[2:]:
    part = smlibpy.import_step(path, heal=False)
    bbox = part.bbox()
    fields = [
        os.path.basename(path),
        str(len(part.faces())),
        str(len(part.edges())),
        str(len(part.vertices())),
        '1' if part.is_valid() else '0',
        repr(float(part.volume)),
        repr(float(part.area)),
        repr(float(bbox.min.x)), repr(float(bbox.min.y)), repr(float(bbox.min.z)),
        repr(float(bbox.max.x)), repr(float(bbox.max.y)), repr(float(bbox.max.z)),
    ]
    print('|'.join(fields))
";

#[test]
fn smlibpy_reimport_matches_closed_form_metrics() {
    let Some(python) = python() else {
        eprintln!("skipping smlibpy round-trip: workspace .venv/bin/python not found");
        return;
    };
    let cases = write_cases();
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.ancestors().nth(3).expect("workspace root");
    let mut command = Command::new(&python);
    command.arg("-c").arg(IMPORT_SCRIPT).arg(root);
    for case in &cases {
        command.arg(&case.path);
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(error) => panic!("cannot run {python:?}: {error}"),
    };
    if !output.status.success() {
        panic!(
            "smlibpy round-trip failed: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    for case in &cases {
        let line = lines.next().unwrap_or_else(|| {
            panic!("smlibpy produced no row for {}", case.name)
        });
        let fields: Vec<&str> = line.split('|').collect();
        assert_eq!(fields.len(), 13, "{}: malformed row {line}", case.name);
        let expected = &case.expected;
        assert_eq!(fields[0], case.path.file_name().unwrap().to_str().unwrap());
        assert_eq!(fields[1].parse::<usize>().unwrap(), expected.faces, "{} faces", case.name);
        assert_eq!(fields[2].parse::<usize>().unwrap(), expected.edges, "{} edges", case.name);
        assert_eq!(
            fields[3].parse::<usize>().unwrap(),
            expected.vertices,
            "{} vertices",
            case.name
        );
        assert_eq!(fields[4], "1", "{} is_valid", case.name);

        let volume: f64 = fields[5].parse().unwrap();
        let area: f64 = fields[6].parse().unwrap();
        let volume_error = (volume - expected.volume).abs() / expected.volume.abs().max(1.0);
        let area_error = (area - expected.area).abs() / expected.area.abs().max(1.0);
        assert!(
            volume_error < 1e-9,
            "{} volume {volume} != {} (rel {volume_error:e})",
            case.name,
            expected.volume
        );
        assert!(
            area_error < 1e-9,
            "{} area {area} != {} (rel {area_error:e})",
            case.name,
            expected.area
        );

        let bounds: Vec<f64> = fields[7..13].iter().map(|field| field.parse().unwrap()).collect();
        for (axis, actual) in bounds[..3].iter().enumerate() {
            assert!(
                (actual - expected.min[axis]).abs() < 1e-6,
                "{} min[{axis}] {actual} != {}",
                case.name,
                expected.min[axis]
            );
        }
        for (axis, actual) in bounds[3..].iter().enumerate() {
            assert!(
                (actual - expected.max[axis]).abs() < 1e-6,
                "{} max[{axis}] {actual} != {}",
                case.name,
                expected.max[axis]
            );
        }
    }
}

#[test]
fn invalid_dimensions_are_explicit_errors() {
    let dir = output_dir();
    let path = dir.join("invalid.step");
    let path = path.to_str().unwrap();

    assert!(write_cylinder(path, 0.0, 1.0).is_err());
    assert!(write_cylinder(path, 1.0, -1.0).is_err());
    assert!(write_sphere(path, f64::NAN).is_err());
    assert!(write_torus(path, 1.0, 2.0).is_err(), "minor > major");
    assert!(write_torus(path, 2.0, 2.0).is_err(), "horn torus rejected");
    assert!(write_cone(path, 0.0, 0.0, 1.0).is_err(), "no radii");
    assert!(write_cone(path, 1.0, 1.0, 0.0).is_err(), "zero height");
    assert!(write_cone(path, -1.0, 1.0, 1.0).is_err(), "negative radius");

    // The rejected calls must not leave a file behind.
    assert!(!Path::new(path).exists());
}
