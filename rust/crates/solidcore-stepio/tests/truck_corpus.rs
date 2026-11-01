//! STEP regression corpus from Truck's test resources.
//!
//! These files currently exercise the unsupported mixed planar/analytic solid path.
//! The expectation is intentionally explicit: a corpus file may be unsupported, but it
//! must fail deterministically and name its own path.

use std::path::Path;

fn corpus(relative: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-truck").join(relative)
}

#[test]
fn truck_abc_corpus_reports_its_current_import_outcome() {
    let cases: &[(&str, &str)] = &[
        ("abc-0000.step", "mixed planar/analytic"),
        ("abc-0006.step", "mixed planar/analytic"),
        ("abc-0008.step", "mixed planar/analytic"),
        ("abc-0035.step", "mixed planar/analytic"),
    ];
    for (name, expected) in cases {
        let path = corpus(name);
        if !path.exists() {
            eprintln!("skipping missing Truck fixture {name}");
            continue;
        }
        let path_text = path.to_string_lossy();
        let error = solidcore_stepio::import_step(&path_text)
            .expect_err("fixture is currently expected to be unsupported");
        assert!(error.contains(expected), "{name}: unexpected error: {error}");
        assert!(error.contains(path_text.as_ref()), "{name}: error must name its file: {error}");
    }
}

#[test]
fn gear_surface_model_is_imported_as_a_display_mesh() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/Sample AP203 Data Files/gear.stp");
    if !path.exists() {
        eprintln!("skipping missing gear.stp");
        return;
    }
    let import = solidcore_stepio::import_step(path.to_str().unwrap()).expect("gear surface model");
    assert_eq!(import.diagnostics.kind, "surface-model");
    assert!(!import.mesh.vertices.is_empty());
    assert!(!import.mesh.faces.is_empty());
    assert!(import.diagnostics.warnings.iter().any(|warning| warning.contains("display mesh")), "{:?}", import.diagnostics.warnings);
}
