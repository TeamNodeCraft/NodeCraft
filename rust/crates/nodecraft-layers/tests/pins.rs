//! Keep the SolidCore connection on one workspace pin.
//!
//! NodeCraft deliberately consumes upstream as a small set of git-pinned layer
//! crates. A member manifest that repeats a revision can silently stay on an old
//! generation while the rest of the workspace moves.

use std::path::Path;

fn revs(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with("solidcore")
        })
        .filter_map(|line| {
            let rest = line.split("rev = \"").nth(1)?;
            rest.split('"').next().map(str::to_owned)
        })
        .collect()
}

#[test]
fn workspace_uses_one_solidcore_revision() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = std::fs::read_to_string(rust.join("Cargo.toml")).expect("workspace manifest");
    let revs = revs(&workspace);
    assert!(!revs.is_empty(), "no SolidCore workspace pins found");
    let first = &revs[0];
    assert!(
        revs.iter().all(|rev| rev == first),
        "SolidCore workspace dependencies must share one revision: {revs:?}"
    );
    assert_eq!(first.len(), 40, "SolidCore pin should be a full commit hash");
}

#[test]
fn member_manifests_do_not_repeat_the_solidcore_pin() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = std::fs::read_to_string(rust.join("Cargo.toml")).expect("workspace manifest");
    let mut members = Vec::new();
    let mut in_members = false;
    for line in workspace.lines() {
        if line.trim_start().starts_with("members = [") {
            in_members = true;
            continue;
        }
        if in_members {
            let trimmed = line.trim();
            if trimmed == "]" {
                break;
            }
            if let Some(name) = trimmed.split('"').nth(1) {
                members.push(name.to_owned());
            }
        }
    }
    assert!(!members.is_empty(), "workspace has no members");
    let mut offenders = Vec::new();
    for member in members {
        let manifest = rust.join(&member).join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else { continue };
        for line in text.lines() {
            if line.contains("github.com/tangency-labs/solidcore") {
                offenders.push(format!("{}: {}", manifest.display(), line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "member manifests must use `workspace = true` for SolidCore deps:\n{}",
        offenders.join("\n")
    );
}

/// The cadmpeg crates are pinned to exactly the versions SolidCraft pins, with the
/// same open-format features. They are bumped together with the SolidCraft pin.
#[test]
fn cadmpeg_pins_match_the_solidcraft_pin() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = std::fs::read_to_string(rust.join("Cargo.toml")).expect("workspace manifest");
    for name in ["cadmpeg-core", "cadmpeg-ir", "cadmpeg-registry"] {
        let line = workspace
            .lines()
            .find(|line| line.trim_start().starts_with(&format!("{name} =")))
            .unwrap_or_else(|| panic!("{name} is missing from the workspace dependencies"));
        assert!(
            line.contains("\"=0.6.7\""),
            "{name} must be pinned exactly to 0.6.7: {line}"
        );
    }
    let registry = workspace
        .lines()
        .find(|line| line.trim_start().starts_with("cadmpeg-registry ="))
        .expect("cadmpeg-registry line");
    for feature in ["default-features = false", "\"fcstd\"", "\"rhino\"", "\"step\""] {
        assert!(registry.contains(feature), "cadmpeg-registry must state {feature}: {registry}");
    }
    let lock = std::fs::read_to_string(rust.join("Cargo.lock")).expect("lockfile");
    for name in ["cadmpeg-core", "cadmpeg-ir", "cadmpeg-registry"] {
        let mut found = false;
        let mut version = None;
        for line in lock.lines() {
            if line == format!("name = \"{name}\"") {
                found = true;
            } else if found && version.is_none() && line.starts_with("version = ") {
                version = Some(line.trim().to_owned());
            }
        }
        assert!(found, "Cargo.lock has no {name}");
        assert_eq!(
            version.as_deref(),
            Some("version = \"0.6.7\""),
            "{name} in Cargo.lock must resolve to 0.6.7"
        );
    }
}
