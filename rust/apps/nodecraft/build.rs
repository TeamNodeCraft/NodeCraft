use std::process::Command;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&manifest)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|hash| hash.trim().to_owned())
        .filter(|hash| !hash.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=NODECRAFT_GIT_HASH={hash}");
    let lock = std::fs::read_to_string(format!("{manifest}/../../Cargo.lock"))
        .unwrap_or_default();
    let mut in_solidcore = false;
    let mut solidcore = "unknown".to_owned();
    for line in lock.lines() {
        if line.trim() == "name = \"solidcore\"" {
            in_solidcore = true;
        } else if in_solidcore && line.trim_start().starts_with("source = \"git+") {
            if let Some(rev) = line.split('#').nth(1) {
                solidcore = rev.trim_end_matches('"').chars().take(7).collect();
            }
            break;
        }
    }
    println!("cargo:rustc-env=SOLIDCORE_GIT_HASH={solidcore}");
}
