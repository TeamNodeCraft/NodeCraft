//! Run the project's structural lint from `cargo test`.
//!
//! `ast-grep` is a development tool, not a Rust dependency: public builds and CI
//! images may not have it. When the binary is absent this test prints a clear
//! skip message and passes; when it is present it runs the error-severity rules
//! and the rule tests from the repository root.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The ast-grep executable, or `None` when the tool is not installed.
///
/// `sg` is also checked because some ast-grep installs only provide it. Another
/// `sg` (for example the Unix `newgrp` command) is rejected by its version text.
fn ast_grep_binary() -> Option<String> {
    let candidates: Vec<String> = match std::env::var_os("AST_GREP") {
        Some(path) => vec![path.to_string_lossy().into_owned()],
        None => vec!["ast-grep".to_owned(), "sg".to_owned()],
    };
    for candidate in candidates {
        let Ok(output) = Command::new(&candidate).arg("--version").output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let version = String::from_utf8_lossy(&output.stdout).to_ascii_lowercase();
        if candidate == "sg" && !version.contains("ast-grep") {
            continue;
        }
        if version.contains("ast-grep") {
            return Some(candidate);
        }
    }
    None
}

/// Rule ids declared `severity: error` in `ast-grep/rules/*.yml`.
///
/// `ast-grep scan --error` promotes every rule it selects to error, so the scan is
/// scoped with `--filter` to the rules that already declare error severity. That
/// keeps warning-only rules visible in `ast-grep scan` without failing this test.
fn error_rule_ids(root: &Path) -> Vec<String> {
    let rules = root.join("ast-grep/rules");
    let entries = std::fs::read_dir(&rules)
        .unwrap_or_else(|error| panic!("read {}: {error}", rules.display()));
    let mut ids = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "yml") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let id = text.lines().find_map(|line| line.strip_prefix("id: ")).map(str::trim);
        let severity =
            text.lines().find_map(|line| line.strip_prefix("severity: ")).map(str::trim);
        if let (Some(id), Some("error")) = (id, severity) {
            ids.push(id.to_owned());
        }
    }
    ids.sort();
    ids
}

fn run(binary: &str, root: &Path, args: &[&str]) {
    let output = Command::new(binary)
        .args(args)
        .current_dir(root)
        .output()
        .unwrap_or_else(|error| panic!("run `{binary} {}`: {error}", args.join(" ")));
    assert!(
        output.status.success(),
        "`{binary} {}` failed in {}\nstatus: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        args.join(" "),
        root.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn ast_grep_project_rules_pass() {
    let root = repo_root();
    let Some(binary) = ast_grep_binary() else {
        eprintln!(
            "skipping ast-grep project rules: `ast-grep` was not found in PATH \
             (install ast-grep 0.51.2 or set AST_GREP to its path)"
        );
        return;
    };

    // `ast-grep/` and `sgconfig.yml` are private tooling and are not exported to
    // the public snapshot, even though this test is. Without the config there is
    // nothing to run; do not turn a public build into a failure.
    if !root.join("sgconfig.yml").is_file() || !root.join("ast-grep/rules").is_dir() {
        eprintln!(
            "skipping ast-grep project rules: private ast-grep config is absent from {}",
            root.display()
        );
        return;
    }

    let error_rules = error_rule_ids(&root);
    if error_rules.is_empty() {
        eprintln!("no ast-grep rules declare severity: error; skipping the scan");
    } else {
        let filter = format!("^({})$", error_rules.join("|"));
        run(&binary, &root, &["scan", "--error", "--filter", &filter]);
    }
    run(&binary, &root, &["test"]);
}
