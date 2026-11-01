//! STEP analytic curved surface → solidcore `Surface` converter tests.
//!
//! The converter lives in `src/step_curved.rs`; it is not yet declared as a module in
//! `lib.rs` (that file is owned elsewhere), so this test includes the source directly.
//! Wiring it in is one line: `pub mod step_curved;`.

#[path = "../src/step_curved.rs"]
mod step_curved;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use solidcore::math::Point2;
use solidcore::surface::surface::SurfaceKind;
use nodecraft_step::StepFile;

use step_curved::analytic_surface_from_step;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nodecraft-step-curved-{}-{name}", std::process::id()))
}

fn parse(path: &Path) -> StepFile {
    let text = fs::read_to_string(path).expect("read STEP file");
    StepFile::parse(&text).expect("parse STEP file")
}

fn entity_id(file: &StepFile, kind: &str) -> u32 {
    file.entities
        .iter()
        .find(|(_, entity)| entity.name == kind)
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no {kind} entity"))
}

fn assert_point(actual: solidcore::math::Point3, expected: [f64; 3], context: &str) {
    let tolerance = 1e-9;
    assert!(
        (actual.x - expected[0]).abs() < tolerance
            && (actual.y - expected[1]).abs() < tolerance
            && (actual.z - expected[2]).abs() < tolerance,
        "{context}: got ({}, {}, {}), expected {expected:?}",
        actual.x,
        actual.y,
        actual.z
    );
}

// ---------------------------------------------------------------------------
// Inline STEP text for placement cases
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Inline {
    lines: Vec<String>,
}

impl Inline {
    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, body: impl AsRef<str>) -> u32 {
        self.lines.push(body.as_ref().to_string());
        self.lines.len() as u32
    }

    fn text(&self) -> String {
        let mut out = String::from(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'step-curved-test','','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n",
        );
        for (index, line) in self.lines.iter().enumerate() {
            out.push_str(&format!("#{}={};\n", index + 1, line));
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }

    fn file(&self) -> StepFile {
        StepFile::parse(&self.text()).expect("inline STEP parses")
    }

    fn point(&mut self, point: [f64; 3]) -> u32 {
        self.add(format!(
            "CARTESIAN_POINT('',({},{},{}))",
            number(point[0]),
            number(point[1]),
            number(point[2])
        ))
    }

    fn direction(&mut self, direction: [f64; 3]) -> u32 {
        self.add(format!(
            "DIRECTION('',({},{},{}))",
            number(direction[0]),
            number(direction[1]),
            number(direction[2])
        ))
    }

    fn placement(&mut self, origin: [f64; 3], axis: [f64; 3], reference: [f64; 3]) -> u32 {
        let point = self.point(origin);
        let z = self.direction(axis);
        let x = self.direction(reference);
        self.add(format!("AXIS2_PLACEMENT_3D('',#{point},#{z},#{x})"))
    }
}

fn number(value: f64) -> String {
    if value == 0.0 {
        "0.".into()
    } else if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value:?}")
    }
}

// ---------------------------------------------------------------------------
// Primitive STEP files written by nodecraft-step
// ---------------------------------------------------------------------------

#[test]
fn converts_a_cylindrical_surface() {
    let path = temp_path("cylinder.step");
    nodecraft_step::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).expect("write cylinder");
    let file = parse(&path);
    let surface_id = entity_id(&file, "CYLINDRICAL_SURFACE");
    let surface = analytic_surface_from_step(&file, surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Cone(cylinder) => {
            assert!(cylinder.is_cylinder(), "cylinder surface must be a cylinder class");
            assert!((cylinder.bot_radius() - 5.0).abs() < 1e-12);
            assert!((cylinder.top_radius() - 5.0).abs() < 1e-12);
            // The writer places the surface origin at (0, 0, -5), axis +Z, ref +X.
            assert_point(
                cylinder.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [5.0, 0.0, -5.0],
                "cylinder (u=0, v=0)",
            );
            assert_point(
                cylinder.evaluate_step_point(Point2::new(90.0, 3.0)).unwrap(),
                [0.0, 5.0, -2.0],
                "cylinder (u=90, v=3)",
            );
            assert_point(
                cylinder.evaluate_step_point(Point2::new(0.0, 10.0)).unwrap(),
                [5.0, 0.0, 5.0],
                "cylinder top seam",
            );
        }
        other => panic!("expected a cone-kind cylinder surface, got {other:?}"),
    }
}

#[test]
fn converts_a_conical_surface() {
    let path = temp_path("cone.step");
    nodecraft_step::write_cone(path.to_str().unwrap(), 5.0, 2.0, 10.0).expect("write cone");
    let file = parse(&path);
    let surface_id = entity_id(&file, "CONICAL_SURFACE");
    let surface = analytic_surface_from_step(&file, surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Cone(cone) => {
            assert!(!cone.is_cylinder());
            // Placement origin (0, 0, -5), axis -Z, radius 5 at the origin,
            // semi-angle atan(3 / 10).
            let canonical = cone.canonical();
            assert!((canonical.radius - 5.0).abs() < 1e-12);
            let expected_semi = (0.3f64).atan().to_degrees();
            assert!((canonical.semi_angle_deg - expected_semi).abs() < 1e-12);
            let tangent = (cone.radius(1.0) - cone.radius(0.0)).abs();
            assert!((tangent - 0.3).abs() < 1e-12, "tan(semi-angle) = {tangent}");
            assert!((cone.radius(-10.0) - 2.0).abs() < 1e-12, "top radius");
            assert_point(
                cone.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [5.0, 0.0, -5.0],
                "cone base seam",
            );
            // v runs along the placement axis (-Z): v = -10 is the physical top.
            assert_point(
                cone.evaluate_step_point(Point2::new(0.0, -10.0)).unwrap(),
                [2.0, 0.0, 5.0],
                "cone top seam",
            );
        }
        other => panic!("expected a cone surface, got {other:?}"),
    }
}

#[test]
fn converts_a_spherical_surface() {
    let path = temp_path("sphere.step");
    nodecraft_step::write_sphere(path.to_str().unwrap(), 5.0).expect("write sphere");
    let file = parse(&path);
    let surface_id = entity_id(&file, "SPHERICAL_SURFACE");
    let surface = analytic_surface_from_step(&file, surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Sphere(sphere) => {
            assert!((sphere.radius() - 5.0).abs() < 1e-12);
            assert_point(
                sphere.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [5.0, 0.0, 0.0],
                "sphere (u=0, v=0)",
            );
            assert_point(
                sphere.evaluate_step_point(Point2::new(90.0, 0.0)).unwrap(),
                [0.0, 5.0, 0.0],
                "sphere (u=90, v=0)",
            );
            assert_point(
                sphere.evaluate_step_point(Point2::new(0.0, 90.0)).unwrap(),
                [0.0, 0.0, 5.0],
                "sphere north pole",
            );
        }
        other => panic!("expected a sphere surface, got {other:?}"),
    }
}

#[test]
fn converts_a_toroidal_surface() {
    let path = temp_path("torus.step");
    nodecraft_step::write_torus(path.to_str().unwrap(), 8.0, 2.0).expect("write torus");
    let file = parse(&path);
    let surface_id = entity_id(&file, "TOROIDAL_SURFACE");
    let surface = analytic_surface_from_step(&file, surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Torus(torus) => {
            assert!((torus.major_radius() - 8.0).abs() < 1e-12);
            assert!((torus.minor_radius() - 2.0).abs() < 1e-12);
            assert_point(
                torus.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [10.0, 0.0, 0.0],
                "torus outer seam",
            );
            assert_point(
                torus.evaluate_step_point(Point2::new(90.0, 0.0)).unwrap(),
                [0.0, 10.0, 0.0],
                "torus (u=90, v=0)",
            );
            assert_point(
                torus.evaluate_step_point(Point2::new(0.0, 90.0)).unwrap(),
                [8.0, 0.0, 2.0],
                "torus top of tube",
            );
        }
        other => panic!("expected a torus surface, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Placement handling
// ---------------------------------------------------------------------------

#[test]
fn preserves_a_rotated_translated_placement() {
    // Sphere at (10, 20, 30) with axis +Y and reference +Z:
    // local X = +Z, local Y = axis x X = +X.
    let mut inline = Inline::new();
    let placement = inline.placement([10.0, 20.0, 30.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
    let surface_id = inline.add(format!("SPHERICAL_SURFACE('',#{placement},2.0)"));
    let file = inline.file();
    let surface = analytic_surface_from_step(&file, surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Sphere(sphere) => {
            assert_point(
                sphere.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [10.0, 20.0, 32.0],
                "rotated sphere (u=0, v=0) = origin + r * local X",
            );
            assert_point(
                sphere.evaluate_step_point(Point2::new(90.0, 0.0)).unwrap(),
                [12.0, 20.0, 30.0],
                "rotated sphere (u=90, v=0) = origin + r * local Y",
            );
            assert_point(
                sphere.evaluate_step_point(Point2::new(0.0, 90.0)).unwrap(),
                [10.0, 22.0, 30.0],
                "rotated sphere pole = origin + r * local Z",
            );
        }
        other => panic!("expected a sphere surface, got {other:?}"),
    }
}

#[test]
fn preserves_a_placement_without_axis_or_reference_directions() {
    let mut inline = Inline::new();
    let origin = inline.point([1.0, 2.0, 3.0]);
    let placement = inline.add(format!("AXIS2_PLACEMENT_3D('',#{origin},$,$)"));
    let surface_id = inline.add(format!("CYLINDRICAL_SURFACE('',#{placement},4.0)"));
    let surface = analytic_surface_from_step(&inline.file(), surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Cone(cylinder) => {
            assert!(cylinder.is_cylinder());
            // STEP defaults: axis +Z, reference +X, origin (1, 2, 3).
            assert_point(
                cylinder.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [5.0, 2.0, 3.0],
                "default placement (u=0, v=0)",
            );
            assert_point(
                cylinder.evaluate_step_point(Point2::new(90.0, 2.0)).unwrap(),
                [1.0, 6.0, 5.0],
                "default placement (u=90, v=2)",
            );
        }
        other => panic!("expected a cylinder surface, got {other:?}"),
    }
}

#[test]
fn translates_and_rotates_a_cylinder_placement() {
    // Origin (1, 2, 3), axis +X, reference +Z: local X = +Z, local Y = X x Z = -Y.
    let mut inline = Inline::new();
    let placement = inline.placement([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let surface_id = inline.add(format!("CYLINDRICAL_SURFACE('',#{placement},4.0)"));
    let surface = analytic_surface_from_step(&inline.file(), surface_id).expect("convert");
    match surface.into_kind() {
        SurfaceKind::Cone(cylinder) => {
            assert_point(
                cylinder.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [1.0, 2.0, 7.0],
                "rotated cylinder (u=0, v=0)",
            );
            assert_point(
                cylinder.evaluate_step_point(Point2::new(90.0, 0.0)).unwrap(),
                [1.0, -2.0, 3.0],
                "rotated cylinder (u=90, v=0)",
            );
            assert_point(
                cylinder.evaluate_step_point(Point2::new(0.0, 5.0)).unwrap(),
                [6.0, 2.0, 7.0],
                "rotated cylinder (u=0, v=5)",
            );
        }
        other => panic!("expected a cylinder surface, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[test]
fn reports_missing_entities_and_non_analytic_kinds() {
    let path = temp_path("errors.step");
    nodecraft_step::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).unwrap();
    let file = parse(&path);
    let missing = analytic_surface_from_step(&file, 99_999).expect_err("missing entity");
    assert!(missing.contains("does not exist"), "{missing}");

    let plane = entity_id(&file, "PLANE");
    let wrong = analytic_surface_from_step(&file, plane).expect_err("plane is not analytic");
    assert!(wrong.contains("not an analytic surface"), "{wrong}");

    let point = entity_id(&file, "CARTESIAN_POINT");
    let wrong = analytic_surface_from_step(&file, point).expect_err("point is not a surface");
    assert!(wrong.contains("not an analytic surface"), "{wrong}");
}

#[test]
fn rejects_invalid_dimensions() {
    let placement = |inline: &mut Inline| inline.placement([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("CYLINDRICAL_SURFACE('',#{p},0.0)"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "zero cylinder radius");

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("SPHERICAL_SURFACE('',#{p},-2.0)"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "negative sphere radius");

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("CONICAL_SURFACE('',#{p},3.0,0.0)"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "zero cone semi-angle");

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("CONICAL_SURFACE('',#{p},3.0,1.5707963267948966)"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "90-degree cone semi-angle");

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("TOROIDAL_SURFACE('',#{p},8.0,0.0)"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "zero torus minor radius");

    let mut inline = Inline::new();
    let p = placement(&mut inline);
    let id = inline.add(format!("CYLINDRICAL_SURFACE('',#{p})"));
    assert!(analytic_surface_from_step(&inline.file(), id).is_err(), "missing radius");
}

#[test]
fn rejects_broken_placements() {
    // Zero axis.
    let mut inline = Inline::new();
    let placement = inline.placement([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    let id = inline.add(format!("CYLINDRICAL_SURFACE('',#{placement},2.0)"));
    let error = analytic_surface_from_step(&inline.file(), id).expect_err("zero axis");
    assert!(error.contains("axis"), "{error}");

    // Missing placement entity.
    let mut inline = Inline::new();
    let id = inline.add("CYLINDRICAL_SURFACE('',#999,2.0)");
    let error = analytic_surface_from_step(&inline.file(), id).expect_err("missing placement");
    assert!(error.contains("does not exist"), "{error}");

    // Placement that is not AXIS2_PLACEMENT_3D.
    let mut inline = Inline::new();
    let point = inline.point([0.0, 0.0, 0.0]);
    let id = inline.add(format!("CYLINDRICAL_SURFACE('',#{point},2.0)"));
    let error = analytic_surface_from_step(&inline.file(), id).expect_err("wrong placement type");
    assert!(error.contains("AXIS2_PLACEMENT_3D"), "{error}");

    // Non-numeric radius.
    let mut inline = Inline::new();
    let placement = inline.placement([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let id = inline.add(format!("CYLINDRICAL_SURFACE('',#{placement},'two')"));
    let error = analytic_surface_from_step(&inline.file(), id).expect_err("bad radius type");
    assert!(error.contains("radius"), "{error}");
}

// ---------------------------------------------------------------------------
// smlibpy-written primitive STEP files
// ---------------------------------------------------------------------------

const EXPORT_SCRIPT: &str = r"
import os, sys
import smlibpy
out = sys.argv[1]
shapes = {
    'cylinder': smlibpy.Cylinder(3.0, 7.0),
    'cone': smlibpy.Cone(4.0, 1.0, 6.0),
    'sphere': smlibpy.Sphere(2.5),
    'torus': smlibpy.Torus(6.0, 1.5),
}
for name, shape in shapes.items():
    shape.export(os.path.join(out, name + '.step'))
print('ok')
";

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

#[test]
fn converts_smlibpy_written_primitive_files() {
    let Some(python) = python() else {
        eprintln!("skipping smlibpy primitive conversion: workspace .venv/bin/python not found");
        return;
    };
    let directory = temp_path("smlibpy");
    fs::create_dir_all(&directory).unwrap();
    let output = Command::new(&python)
        .arg("-c")
        .arg(EXPORT_SCRIPT)
        .arg(&directory)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {python:?}: {error}"));
    assert!(
        output.status.success(),
        "smlibpy export failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let cylinder_file = parse(&directory.join("cylinder.step"));
    let cylinder = analytic_surface_from_step(
        &cylinder_file,
        entity_id(&cylinder_file, "CYLINDRICAL_SURFACE"),
    )
    .expect("smlibpy cylinder surface");
    match cylinder.into_kind() {
        SurfaceKind::Cone(cone) => {
            assert!(cone.is_cylinder());
            assert!((cone.bot_radius() - 3.0).abs() < 1e-9);
            assert_point(
                cone.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [3.0, 0.0, -3.5],
                "smlibpy cylinder base seam",
            );
        }
        other => panic!("expected cylinder, got {other:?}"),
    }

    let cone_file = parse(&directory.join("cone.step"));
    let cone = analytic_surface_from_step(&cone_file, entity_id(&cone_file, "CONICAL_SURFACE"))
        .expect("smlibpy cone surface");
    match cone.into_kind() {
        SurfaceKind::Cone(cone) => {
            assert!(!cone.is_cylinder());
            let canonical = cone.canonical();
            assert!((canonical.radius - 4.0).abs() < 1e-9);
            assert!((canonical.semi_angle_deg - (0.5f64).atan().to_degrees()).abs() < 1e-9);
            assert!(((cone.radius(1.0) - cone.radius(0.0)).abs() - 0.5).abs() < 1e-9);
            assert!((cone.radius(-6.0) - 1.0).abs() < 1e-9);
            assert_point(
                cone.evaluate_step_point(Point2::new(0.0, -6.0)).unwrap(),
                [1.0, 0.0, 3.0],
                "smlibpy cone top seam",
            );
        }
        other => panic!("expected cone, got {other:?}"),
    }

    let sphere_file = parse(&directory.join("sphere.step"));
    let sphere = analytic_surface_from_step(&sphere_file, entity_id(&sphere_file, "SPHERICAL_SURFACE"))
        .expect("smlibpy sphere surface");
    match sphere.into_kind() {
        SurfaceKind::Sphere(sphere) => {
            assert!((sphere.radius() - 2.5).abs() < 1e-9);
            assert_point(
                sphere.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [2.5, 0.0, 0.0],
                "smlibpy sphere seam",
            );
        }
        other => panic!("expected sphere, got {other:?}"),
    }

    let torus_file = parse(&directory.join("torus.step"));
    let torus = analytic_surface_from_step(&torus_file, entity_id(&torus_file, "TOROIDAL_SURFACE"))
        .expect("smlibpy torus surface");
    match torus.into_kind() {
        SurfaceKind::Torus(torus) => {
            assert!((torus.major_radius() - 6.0).abs() < 1e-9);
            assert!((torus.minor_radius() - 1.5).abs() < 1e-9);
            assert_point(
                torus.evaluate_step_point(Point2::new(0.0, 0.0)).unwrap(),
                [7.5, 0.0, 0.0],
                "smlibpy torus outer seam",
            );
        }
        other => panic!("expected torus, got {other:?}"),
    }
}
