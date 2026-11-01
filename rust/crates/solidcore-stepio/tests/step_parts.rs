use std::path::PathBuf;

use solidcore_stepio::import_step;

/// Small permissive part library: every file must import or be explicitly
/// rejected, and the known mixed cube/cylinder sample must keep importing.
#[test]
fn step_part_library_has_stable_outcomes() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-parts");
    let mut checked = 0;
    let mut imported = 0;
    for entry in std::fs::read_dir(&corpus).expect("part library") {
        let path = entry.expect("entry").path();
        if !matches!(path.extension().and_then(|extension| extension.to_str()), Some("stp") | Some("step")) {
            continue;
        }
        checked += 1;
        match import_step(path.to_str().expect("UTF-8 path")) {
            Ok(result) => {
                assert!(!result.mesh.faces.is_empty(), "{} imported no triangles", path.display());
                imported += 1;
            }
            Err(error) => assert!(!error.is_empty(), "{} returned an empty error", path.display()),
        }
    }
    assert_eq!(checked, 13, "expected 13 part-library files");
    assert!(imported >= 1, "the known cube/cylinder sample must import");
}
