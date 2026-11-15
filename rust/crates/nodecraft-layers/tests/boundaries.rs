//! Keep application/engine code on the layer side of the SolidCore boundary.
//!
//! NodeCraft still has a kernel-first engine during the transition, but new
//! app/engine code must cross through `nodecraft-layers` rather than importing
//! the upstream layer crates directly.
//!
//! The last test is the ops-session rule: `solidcore_ops` is named only in
//! `nodecraft-ops` (which wraps it in `OpsSession`) and in the few crates listed
//! as exceptions, each of which needs an `OpsSession` method before it can move.

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

/// Where `solidcore_ops` may still be named outside `nodecraft-ops`, and why. Each entry is
/// a promise to add an `OpsSession` method, not a permanent exemption.
const SOLIDCORE_OPS_EXCEPTIONS: &[(&str, &str)] = &[
    (
        "crates/nodecraft-layers",
        "record-first adapters: record conversion, the record-body shell/offset/Boolean \
         helpers and the ops body/volume bridge have no OpsSession form yet",
    ),
    (
        "crates/nodecraft-step",
        "the cadmpeg adapter's pcurve Trims implementation and export round-trip volume are \
         upstream record API, not operations",
    ),
    (
        "crates/nodecraft-sweep",
        "sweep loft returns an ops Body and Error; it needs an OpsSession::loft",
    ),
    (
        "crates/nodecraft-wedge",
        "the wedge extrudes through ops' prism family; it needs an OpsSession method",
    ),
    (
        "crates/nodecraft-extrude",
        "a dev-dependency: the extrusion volume is pinned against ops' prism in a test",
    ),
];

fn source_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

fn solidcore_ops_exception(path: &str) -> Option<(&'static str, &'static str)> {
    SOLIDCORE_OPS_EXCEPTIONS
        .iter()
        .copied()
        .find(|(prefix, _)| path == *prefix || path.starts_with(&format!("{prefix}/")))
}

#[test]
fn solidcore_ops_is_named_only_in_nodecraft_ops_or_a_listed_exception() {
    let rust = rust_root();
    let mut files = Vec::new();
    for dir in ["crates", "apps"] {
        source_files(&rust.join(dir), &mut files);
    }
    assert!(!files.is_empty(), "no source files found under {}", rust.display());

    let mut offenders = Vec::new();
    let mut used = std::collections::BTreeSet::new();
    for file in files {
        let path = source_relative(&rust, &file);
        if path.starts_with("crates/nodecraft-ops/") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        if !text.contains("solidcore_ops") {
            continue;
        }
        match solidcore_ops_exception(&path) {
            Some((prefix, _)) => {
                used.insert(prefix);
            }
            None => offenders.push(path),
        }
    }
    assert!(
        offenders.is_empty(),
        "solidcore_ops must be reached through nodecraft_ops::OpsSession; these files name it directly:\n  {}",
        offenders.join("\n  ")
    );

    let unused: Vec<&str> = SOLIDCORE_OPS_EXCEPTIONS
        .iter()
        .map(|(prefix, _)| *prefix)
        .filter(|prefix| !used.contains(prefix))
        .collect();
    assert!(
        unused.is_empty(),
        "these exceptions no longer match any use; remove them from the list: {unused:?}"
    );
}

/// The sketch extrude is an `OpsSession` operation. Neither the layers adapters nor the
/// engine may name its ops recipe directly again: the worker lends `OpsSession` its own
/// configured kernel and gets a `SolidBody` back.
#[test]
fn sketch_extrude_goes_through_the_ops_session() {
    let rust = rust_root();
    for file in ["crates/nodecraft-layers/src/lib.rs", "crates/nodecraft-engine/src/lib.rs"] {
        let text = std::fs::read_to_string(rust.join(file)).expect("source file");
        assert!(
            !text.contains("extrude_planar_regions"),
            "{file} names the ops extrusion recipe directly; use \
             nodecraft_ops::OpsSession::extrude_planar_region"
        );
    }
}

#[test]
fn only_nodecraft_ops_and_the_listed_exceptions_depend_on_solidcore_ops() {
    let rust = rust_root();
    let mut offenders = Vec::new();
    for dir in ["crates", "apps"] {
        let Ok(entries) = std::fs::read_dir(rust.join(dir)) else { continue };
        for entry in entries.flatten() {
            let path = entry.path().join("Cargo.toml");
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if !text.contains("solidcore-ops") {
                continue;
            }
            let crate_dir = source_relative(&rust, &entry.path());
            if crate_dir == "crates/nodecraft-ops" || solidcore_ops_exception(&crate_dir).is_some() {
                continue;
            }
            offenders.push(crate_dir);
        }
    }
    assert!(
        offenders.is_empty(),
        "these crates depend on solidcore-ops without an exception; use nodecraft-ops instead: {offenders:?}"
    );
}

/// cadmpeg's neutral model and `solidcore-cadir` are the import/export seam: exactly one
/// NodeCraft crate may name them. Everything else works in schema-4 records.
#[test]
fn cadmpeg_and_cadir_are_named_only_in_the_adapter_crate() {
    let rust = rust_root();
    let mut files = Vec::new();
    for dir in ["crates", "apps"] {
        source_files(&rust.join(dir), &mut files);
    }
    let mut offenders = Vec::new();
    for file in files {
        let path = source_relative(&rust, &file);
        // Product source only: tests may name the seam to pin it.
        if !path.contains("/src/") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        let names = text.contains("cadmpeg_") || text.contains("solidcore_cadir");
        if names && path != "crates/nodecraft-step" && !path.starts_with("crates/nodecraft-step/") {
            offenders.push(path);
        }
    }
    assert!(
        offenders.is_empty(),
        "cadmpeg and solidcore_cadir must stay behind crates/nodecraft-step; these files name them:\n  {}",
        offenders.join("\n  ")
    );
}

/// The old STEP route is gone: no NodeCraft source or manifest names `solidcore_step`,
/// `solidcore-step`, `step_export`, `step-export`, or the `p21` parser.
#[test]
fn the_old_step_route_is_not_named() {
    let rust = rust_root();
    let mut offenders = Vec::new();
    let mut files = Vec::new();
    for dir in ["crates", "apps"] {
        source_files(&rust.join(dir), &mut files);
    }
    for file in files {
        let path = source_relative(&rust, &file);
        if path == "crates/nodecraft-layers/tests/boundaries.rs" {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        for needle in ["solidcore_step", "step_export"] {
            if text.contains(needle) {
                offenders.push(format!("{path}: {needle}"));
            }
        }
        // `p21` as its own token, not the `p21` inside `ap214`.
        let bytes = text.as_bytes();
        for index in 0..bytes.len().saturating_sub(2) {
            if &bytes[index..index + 3] != b"p21" {
                continue;
            }
            let before = index.checked_sub(1).map(|at| bytes[at]);
            if before.is_none_or(|byte| !byte.is_ascii_alphanumeric() && byte != b'_') {
                offenders.push(format!("{path}: p21"));
            }
        }
    }
    for dir in ["crates", "apps"] {
        let Ok(entries) = std::fs::read_dir(rust.join(dir)) else { continue };
        for entry in entries.flatten() {
            let path = entry.path().join("Cargo.toml");
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            for needle in ["solidcore-step", "step-export"] {
                if text.contains(needle) {
                    offenders.push(format!("{}: {needle}", source_relative(&rust, &path)));
                }
            }
        }
    }
    let workspace = std::fs::read_to_string(rust.join("Cargo.toml")).expect("workspace manifest");
    for needle in ["solidcore-step", "step-export"] {
        if workspace.contains(needle) {
            offenders.push(format!("Cargo.toml: {needle}"));
        }
    }
    assert!(offenders.is_empty(), "the old STEP route is still named:\n  {}", offenders.join("\n  "));
}

/// The retired assembly crate is gone from every manifest and product source.
#[test]
fn step_assemblies_is_not_named() {
    let rust = rust_root();
    let mut offenders = Vec::new();
    let mut manifests = Vec::new();
    source_files(&rust.join("crates"), &mut manifests);
    source_files(&rust.join("apps"), &mut manifests);
    for path in &manifests {
        if path.file_name().is_some_and(|name| name == "Cargo.toml") {
            let Ok(text) = std::fs::read_to_string(path) else { continue };
            if text.contains("step-assemblies") || text.contains("step_assemblies") {
                offenders.push(source_relative(&rust, path));
            }
        }
    }
    let workspace = std::fs::read_to_string(rust.join("Cargo.toml")).expect("workspace manifest");
    if workspace.contains("step-assemblies") || workspace.contains("step_assemblies") {
        offenders.push("Cargo.toml".into());
    }
    let mut files = Vec::new();
    for dir in ["crates", "apps"] {
        source_files(&rust.join(dir), &mut files);
    }
    for file in files {
        let path = source_relative(&rust, &file);
        if !path.contains("/src/") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        if text.contains("step_assemblies") || text.contains("step-assemblies") {
            offenders.push(path);
        }
    }
    assert!(
        offenders.is_empty(),
        "the retired step-assemblies crate is still named:\n  {}",
        offenders.join("\n  ")
    );
}

/// Part 21 text is written only by cadmpeg. Product source may not contain the
/// framing strings or an entity-id formatter.
#[test]
fn product_source_has_no_hand_written_part_21() {
    let rust = rust_root();
    let mut files = Vec::new();
    for dir in ["crates", "apps"] {
        source_files(&rust.join(dir), &mut files);
    }
    let framing = [
        "ISO-10303-21",
        "END-ISO-10303-21",
        "FILE_DESCRIPTION(",
        "FILE_NAME(",
        "FILE_SCHEMA",
        "HEADER;",
        "DATA;",
        "ENDSEC;",
    ];
    let mut offenders = Vec::new();
    for file in files {
        let path = source_relative(&rust, &file);
        // Product source only; fixture readers and this boundary test may name the
        // framing while proving that nothing else does.
        if !path.contains("/src/") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        for needle in framing {
            if text.contains(needle) {
                offenders.push(format!("{path}: {needle}"));
            }
        }
        // `format!("#{id}={body};")`-style entity formatting.
        for line in text.lines() {
            let compact = line.replace(' ', "");
            if compact.contains("\"#{")
                && compact.contains("}=")
                && (compact.contains("format!(") || compact.contains("write!(") || compact.contains("writeln!("))
            {
                offenders.push(format!("{path}: entity-format line"));
                break;
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "STEP Part 21 text belongs to cadmpeg; hand-written output found:\n  {}",
        offenders.join("\n  ")
    );
}
