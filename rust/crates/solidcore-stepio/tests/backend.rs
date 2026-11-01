use std::path::Path;

fn block() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).expect("repo root");
    root.join("tests/Sample AP203 Data Files/block.stp").to_string_lossy().into_owned()
}

#[test]
fn legacy_backend_imports_block() {
    let import = solidcore_stepio::import_step_with_backend(&block(), solidcore_stepio::StepReaderBackend::Legacy).expect("legacy");
    assert!(!import.mesh.faces.is_empty());
}

#[test]
fn solidcore_backend_imports_block_through_the_builder() {
    let import = solidcore_stepio::import_step_with_backend(&block(), solidcore_stepio::StepReaderBackend::Solidcore)
        .expect("solidcore builder");
    assert_eq!(import.diagnostics.kind, "solidcore-brep");
    assert!(!import.mesh.faces.is_empty());
}

#[test]
fn compare_backend_returns_legacy_with_a_warning() {
    let import = solidcore_stepio::import_step_with_backend(&block(), solidcore_stepio::StepReaderBackend::Compare).expect("compare");
    assert!(!import.mesh.faces.is_empty());
    assert!(import.diagnostics.warnings.iter().any(|warning| warning.contains("reader compare")));
}
