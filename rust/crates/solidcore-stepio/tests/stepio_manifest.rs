//! Manifest-driven Stepio import validation.
//!
//! The point is not just “a mesh came back”: exact imports must agree with the
//! independently parsed STEP point bounds, display imports must be labelled as such,
//! and unsupported files must fail with their category and path.

use std::path::{Path, PathBuf};

use nodecraft_protocol::Mesh;
use nodecraft_step::read_step_bounds;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).expect("repo root").to_path_buf()
}

fn mesh_bounds(mesh: &Mesh) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis] as f64);
            max[axis] = max[axis].max(vertex[axis] as f64);
        }
    }
    (min, max)
}

fn exact_bounds(path: &str) -> ([f64; 3], [f64; 3]) {
    // Prefer the topology actually referenced by the solid: STEP files may contain
    // unreferenced geometry whose points would widen a raw CARTESIAN_POINT bounds.
    if let Ok(bodies) = nodecraft_step::read_planar_solids(path) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for body in &bodies {
            for point in &body.points {
                for axis in 0..3 {
                    min[axis] = min[axis].min(point[axis]);
                    max[axis] = max[axis].max(point[axis]);
                }
            }
        }
        if min[0].is_finite() { return (min, max); }
    }
    nodecraft_step::read_topology_bounds(path).expect("independent STEP bounds")
}

fn assert_bounds(got_min: [f64; 3], got_max: [f64; 3], want_min: [f64; 3], want_max: [f64; 3], relative: f64, label: &str) {
    let extent = (0..3).map(|axis| (want_max[axis] - want_min[axis]).abs()).fold(0.0f64, f64::max).max(1.0);
    let tolerance = extent * relative;
    for axis in 0..3 {
        assert!((got_min[axis] - want_min[axis]).abs() <= tolerance, "{label}: min[{axis}] {} vs {} (tol {tolerance})", got_min[axis], want_min[axis]);
        assert!((got_max[axis] - want_max[axis]).abs() <= tolerance, "{label}: max[{axis}] {} vs {} (tol {tolerance})", got_max[axis], want_max[axis]);
    }
}

#[ignore = "slow legacy STEP manifest; superseded by NIST geometry and corpus gates"]
#[test]
fn stepio_manifest_cases_hold() {
    let root = repo_root();
    let manifest_text = std::fs::read_to_string(root.join("tests/stepio/manifest.json")).expect("manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("manifest json");
    let fixtures = manifest["fixtures"].as_array().expect("fixtures");
    assert!(fixtures.len() >= 8, "manifest unexpectedly shrank");
    for case in fixtures {
        let name = case["name"].as_str().unwrap_or("unnamed");
        let relative = case["path"].as_str().expect("fixture path");
        let path = root.join(relative);
        assert!(path.exists(), "{name}: missing fixture {relative}");
        let path_text = path.to_string_lossy();
        let result = solidcore_stepio::import_step(&path_text);
        match case["expect"].as_str().expect("expect") {
            "exact" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected exact import: {error}"));
                assert_ne!(import.diagnostics.kind, "surface-model", "{name}: exact case fell back to surface-model");
                assert!(!import.mesh.faces.is_empty(), "{name}: exact import produced no faces");
                let (want_min, want_max) = if let Some(bbox) = case.get("bbox") {
                    let values = |key: &str| -> [f64; 3] {
                        let values = bbox[key].as_array().expect("bbox axis");
                        [values[0].as_f64().unwrap(), values[1].as_f64().unwrap(), values[2].as_f64().unwrap()]
                    };
                    (values("min"), values("max"))
                } else {
                    exact_bounds(&path_text)
                };
                let (got_min, got_max) = mesh_bounds(&import.mesh);
                assert_bounds(got_min, got_max, want_min, want_max, case["bboxRelativeTolerance"].as_f64().unwrap_or(1e-4), name);
            }
            "surface-model" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected surface-model import: {error}"));
                assert_eq!(import.diagnostics.kind, "surface-model", "{name}");
                assert!(!import.mesh.faces.is_empty(), "{name}: surface import produced no faces");
                assert!(import.diagnostics.warnings.iter().any(|warning| warning.contains("display mesh")), "{name}: missing display warning");
                let (want_min, want_max) = if let Some(bbox) = case.get("bbox") {
                    let values = |key: &str| -> [f64; 3] {
                        let values = bbox[key].as_array().expect("bbox axis");
                        [values[0].as_f64().unwrap(), values[1].as_f64().unwrap(), values[2].as_f64().unwrap()]
                    };
                    (values("min"), values("max"))
                } else {
                    read_step_bounds(&path_text).expect("independent STEP bounds")
                };
                let (got_min, got_max) = mesh_bounds(&import.mesh);
                assert_bounds(got_min, got_max, want_min, want_max, case["bboxRelativeTolerance"].as_f64().unwrap_or(0.05), name);
            }
            "sheet-model" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected sheet-model import: {error}"));
                assert_eq!(import.diagnostics.kind, "sheet-brep", "{name}");
                assert!(!import.mesh.faces.is_empty(), "{name}: sheet import produced no faces");
            }
            "mesh-model" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected mesh-model import: {error}"));
                assert_eq!(import.diagnostics.kind, "bspline-solid-mesh", "{name}");
                assert!(!import.mesh.faces.is_empty(), "{name}: mesh import produced no faces");
            }
            "assembly" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected assembly import: {error}"));
                assert_eq!(import.diagnostics.kind, "assembly", "{name}");
                assert!(!import.mesh.faces.is_empty(), "{name}: assembly import produced no faces");
            }
            "multi-solid" => {
                let import = result.unwrap_or_else(|error| panic!("{name}: expected multi-solid import: {error}"));
                assert_eq!(import.diagnostics.kind, "multi-solid", "{name}");
                assert!(!import.mesh.faces.is_empty(), "{name}: multi-solid import produced no faces");
            }
            "timeout" => {
                // Explicitly tracked: the import is known to exceed the corpus timeout
                // (mass integration on the current kernel path). Do not silently hide it.
                continue;
            }
            "unsupported" => {
                let error = result.expect_err(&format!("{name}: expected unsupported import"));
                let expected = case["contains"].as_str().expect("contains");
                assert!(error.contains(expected), "{name}: expected {expected:?}, got {error}");
                assert!(error.contains(path_text.as_ref()), "{name}: error must name its file: {error}");
            }
            other => panic!("{name}: unknown expectation {other:?}"),
        }
    }
}

#[test]
fn stepio_manifest_covers_corpus() {
    let root = repo_root();
    let manifest_text = std::fs::read_to_string(root.join("tests/stepio/manifest.json")).expect("manifest");
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).expect("manifest json");
    let mut listed = std::collections::BTreeSet::new();
    for case in manifest["fixtures"].as_array().expect("fixtures") {
        listed.insert(case["path"].as_str().expect("fixture path").to_string());
    }
    let directories = [
        "tests/Sample AP203 Data Files",
        "tests/step-nist",
        "tests/step-parts",
        "tests/step-stepcode",
        "tests/step-truck",
    ];
    let mut found = Vec::new();
    fn collect(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() { collect(base, &path, out); }
            else if path.extension().and_then(|value| value.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("step") || ext.eq_ignore_ascii_case("stp")) {
                out.push(path.strip_prefix(base).expect("relative").display().to_string());
            }
        }
    }
    for dir in directories { collect(&root, &root.join(dir), &mut found); }
    assert_eq!(found.len(), listed.len(), "manifest must cover every corpus fixture");
    for path in &found { assert!(listed.contains(path), "manifest missing corpus fixture {path}"); }
    for path in &listed { assert!(found.contains(path), "manifest lists a missing fixture {path}"); }
}

#[test]
fn block_exact_mass_matches_analytic_values() {
    use nodecraft_solidcore::solidcore::geom::attribute::{AttributeStore, ObjectKey};
    use nodecraft_solidcore::solidcore::kernel::Kernel;
    use nodecraft_solidcore::solidcore::mass::integrator::properties::{brep_mass_properties, PropertyAttributes};
    use nodecraft_solidcore::solidcore::math::Point3;

    let path = repo_root().join("tests/Sample AP203 Data Files/block.stp");
    let import = nodecraft_solidcore::brep_from_step(path.to_str().unwrap()).expect("block exact brep");
    let mut brep = import.brep;
    let kernel = Kernel::default();
    let store = AttributeStore::new();
    let key = |_| ObjectKey(0);
    let attributes = PropertyAttributes { store: &store, key_of: &key };
    let properties = brep_mass_properties(&kernel, &mut brep, 1e-7, 1.0, Point3::new(0.0, 0.0, 0.0), &attributes).expect("mass properties");
    assert!((properties.properties.volume - 60.0).abs() < 1e-6, "volume {}", properties.properties.volume);
    assert!((properties.properties.area - 94.0).abs() < 1e-6, "area {}", properties.properties.area);
}
