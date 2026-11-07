//! Keep application/engine code on the layer side of the SolidCore boundary.
//!
//! NodeCraft still has a kernel-first engine during the transition, but new
//! app/engine code must cross through `nodecraft-layers` rather than importing
//! the upstream layer crates directly.

use std::path::{Path, PathBuf};

fn rust_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn app_and_engine_do_not_import_upstream_layers_directly() {
    let rust = rust_root();
    let forbidden = [
        "use solidcore::",
        "use solidcore_ops::",
        "use solidcore_tess::",
        "use solidcore_convert::",
    ];
    let mut offenders = Vec::new();
    for crate_dir in ["apps/nodecraft/src", "crates/nodecraft-engine/src"] {
        let mut files = Vec::new();
        source_files(&rust.join(crate_dir), &mut files);
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else { continue };
            for line in text.lines() {
                if forbidden.iter().any(|prefix| line.trim_start().starts_with(prefix)) {
                    offenders.push(format!("{}: {}", file.display(), line.trim()));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "app/engine code must cross the SolidCore boundary through nodecraft-layers:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn app_does_not_depend_on_upstream_layer_crates_directly() {
    let rust = rust_root();
    let manifest = std::fs::read_to_string(rust.join("apps/nodecraft/Cargo.toml"))
        .expect("app manifest");
    let forbidden = ["solidcore =", "solidcore-ops", "solidcore-tess", "solidcore-convert"];
    let mut offenders = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim_start();
        if forbidden.iter().any(|name| trimmed.starts_with(name)) {
            offenders.push(trimmed.to_owned());
        }
    }
    assert!(
        offenders.is_empty(),
        "the app must use nodecraft-layers for ops/tess; direct deps found:\n{}",
        offenders.join("\n")
    );
}
