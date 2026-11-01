use std::path::PathBuf;

use solidcore_stepio::import_step;

/// STEPcode's permissive fixtures must either import or fail with an explicit,
/// known category. Unknown errors or panics are regressions.
#[test]
fn stepcode_corpus_has_only_known_outcomes() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-stepcode");
    let mut checked = 0;
    let mut unexpected = Vec::new();
    for entry in std::fs::read_dir(&corpus).expect("STEPcode corpus") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("stp") {
            continue;
        }
        checked += 1;
        match import_step(path.to_str().expect("UTF-8 path")) {
            Ok(import) => assert!(!import.mesh.faces.is_empty(), "{} imported no geometry", path.display()),
            Err(error) => {
                assert!(!error.is_empty(), "{} returned empty error", path.display());
                let known = error.contains("mixed planar/analytic")
                    || error.contains("STEP validation failed")
                    || error.contains("STEP assembly")
                    || error.contains("no MANIFOLD_SOLID_BREP")
                    || error.contains("official STEP reader found no B-rep bodies");
                if !known {
                    unexpected.push(format!("{}: {error}", path.display()));
                }
            }
        }
    }
    assert_eq!(checked, 4, "expected four STEPcode fixtures");
    assert!(unexpected.is_empty(), "unexpected STEPcode failures:\n{}", unexpected.join("\n"));
}
