use std::path::PathBuf;

use solidcore_stepio::import_step;

/// The committed NIST AP203 corpus. Imports may be explicitly unsupported, but
/// every file must either import geometry or fail in one of the known importer
/// categories; unknown errors, empty errors and panics are regressions.
#[test]
fn nist_corpus_has_only_known_outcomes() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-nist");
    let mut checked = 0;
    let mut imported = 0;
    let mut mixed = 0;
    let mut schema = 0;
    let mut known = 0;
    let mut unexpected = Vec::new();
    for entry in std::fs::read_dir(&corpus).expect("NIST corpus directory") {
        let path = entry.expect("corpus entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("stp") {
            continue;
        }
        checked += 1;
        match import_step(path.to_str().expect("UTF-8 corpus path")) {
            Ok(result) => {
                assert!(!result.mesh.faces.is_empty(), "{} imported no geometry", path.display());
                imported += 1;
            }
            Err(error) => {
                assert!(!error.is_empty(), "{} returned an empty error", path.display());
                if error.contains("mixed planar/analytic") {
                    mixed += 1;
                } else if error.contains("STEP validation failed") {
                    schema += 1;
                } else if error.contains("loop gap larger than tolerance") {
                    known += 1;
                } else {
                    unexpected.push(format!("{}: {error}", path.display()));
                }
            }
        }
    }
    assert!(checked >= 11, "expected the NIST corpus, found {checked} files");
    assert!(unexpected.is_empty(), "unexpected NIST failures:\n{}", unexpected.join("\n"));
    assert_eq!(imported + mixed + schema + known, checked, "NIST outcome categories do not sum");
}
