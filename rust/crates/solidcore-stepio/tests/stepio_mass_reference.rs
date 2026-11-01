//! Independent mass-reference validation for STEP imports.
//!
//! Reads `tests/stepio/reference-volumes.json`, an independent SMLib C++ reference
//! table (only fixtures where SMLib returns a valid B-rep), imports each fixture with
//! `solidcore_stepio::import_step`, derives volume/area/bbox from the imported display
//! mesh (divergence theorem / triangle areas - independent of the B-rep mass code),
//! and compares against the reference at 0.5% relative tolerance.
//!
//! A mismatch on an imported exact solid is always reported with the fixture name and
//! both values. Fixtures that are deliberately accepted with a known mismatch must be
//! listed in `KNOWN_MISMATCHES`; the test then fails if that list and the observed
//! mismatches differ (so a new "relaxation" like nist_ftc_09 importing at 158,638
//! mm^3 against the SMLib 136,447 mm^3 can never pass silently).

use std::path::{Path, PathBuf};

use serde_json::Value;
use nodecraft_protocol::Mesh;

/// Fixtures currently imported but known not to match the SMLib reference. Each entry
/// must stay mismatching (the test fails if one becomes correct, asking to update the
/// table) and any mismatch not listed here fails the test.
const KNOWN_MISMATCHES: &[(&str, &str)] = &[
    ("cubsomcy.stp", "exact multi-solid mesh claims a solid; mesh volume 868,524.998535 vs SMLib 68,722.339297 (12.6x). Tracked until multi-solid construction/bbox is fixed."),
    ("doghouse.stp", "exact mixed-brep import; mesh volume 63,387,372,591.96 vs SMLib 12,629,528,647.66 and mesh area +21.5%. The current working-tree step_mixed candidate produces an invalid display mesh; B-rep mass in the separate harness is 12,629,511,672.7 (-1.3e-6). Tracked visible; must pass again once the candidate is fixed."),
];

const VOLUME_AREA_TOLERANCE: f64 = 0.005;
const BBOX_TOLERANCE: f64 = 0.005;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .to_path_buf()
}

#[derive(Debug, Clone, Copy)]
struct Mass {
    volume: f64,
    area: f64,
}

fn mesh_mass(mesh: &Mesh) -> Mass {
    let v = |i: u32| {
        let p = mesh.vertices[i as usize];
        [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
    };
    let mut volume = 0.0f64;
    let mut area = 0.0f64;
    for f in &mesh.faces {
        let (a, b, c) = (v(f[0]), v(f[1]), v(f[2]));
        volume += (a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            / 6.0;
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    }
    Mass { volume, area }
}

fn mesh_bbox(mesh: &Mesh) -> Option<([f64; 3], [f64; 3])> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in &mesh.vertices {
        for a in 0..3 {
            lo[a] = lo[a].min(f64::from(p[a]));
            hi[a] = hi[a].max(f64::from(p[a]));
        }
    }
    (lo[0].is_finite() && hi[0].is_finite()).then_some((lo, hi))
}

fn rel(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 { (a - b).abs() } else { (a - b).abs() / b.abs() }
}

fn find_by_name(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_by_name(&path, name) { return Some(found); }
        } else if path.file_name().map_or(false, |n| n.to_string_lossy() == name) {
            return Some(path);
        }
    }
    None
}

#[test]
fn smlib_mass_reference_mismatches_are_visible() {
    let root = repo_root();
    let table_path = root.join("tests/stepio/reference-volumes.json");
    let table: Value = serde_json::from_str(&std::fs::read_to_string(&table_path).expect("reference table")).expect("json");
    let fixtures = table["fixtures"].as_array().expect("fixtures");
    let tol = table["volume_area_tolerance"].as_f64().unwrap_or(VOLUME_AREA_TOLERANCE);
    let bbox_tol = table["bbox_tolerance"].as_f64().unwrap_or(BBOX_TOLERANCE);

    let mut exact_matched: Vec<String> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();
    let mut inexact: Vec<String> = Vec::new();
    let mut not_imported: Vec<String> = Vec::new();
    let mut nist_ftc_09_checked = false;

    println!(
        "{:<24} {:<10} {:>18} {:>18} {:>10} {:>10} {:>8} verdict",
        "fixture", "import", "volume", "smlib_volume", "vol_err", "area_err", "bbox_err"
    );
    for fixture in fixtures {
        let name = fixture["name"].as_str().unwrap_or("?");
        let rel_path = fixture["path"].as_str().expect("path");
        let mut path = root.join(rel_path);
        if !path.exists() {
            if let Some(found) = find_by_name(&root.join("tests"), &fixture["name"].as_str().unwrap_or("")) {
                path = found;
            }
        }
        if !path.exists() {
            println!("{name:<24} {kind:<10} reference fixture missing: {rel_path}", kind = "missing");
            not_imported.push(name.to_string());
            continue;
        }
        let ref_volume = fixture["volume"].as_f64().expect("volume");
        let ref_area = fixture["area"].as_f64().expect("area");
        let ref_bbox: Vec<f64> = fixture["bbox"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().filter_map(|v| v.as_f64()))
            .collect();
        let import = solidcore_stepio::import_step(path.to_str().unwrap());
        let mut row = |import: &str, volume: Option<f64>, vol_err: f64, area_err: f64, bbox_err: f64, verdict: &str| {
            println!(
                "{name:<24} {import:<10} {:>18} {ref_volume:>18.6} {vol_err:>10.3e} {area_err:>10.3e} {bbox_err:>8.3e} {verdict}",
                volume.map(|v| format!("{v:.6}")).unwrap_or_else(|| "-".into())
            );
        };
        match import {
            Err(error) => {
                let first = error.lines().next().unwrap_or("").chars().take(110).collect::<String>();
                println!(
                    "{name:<24} {import:<10} {volume:>18} {ref_volume:>18.6} {vol_err:>10} {area_err:>10} {bbox_err:>8} {verdict}",
                    import = "error",
                    volume = "-",
                    vol_err = "-",
                    area_err = "-",
                    bbox_err = "-",
                    verdict = format!("not-imported: {first}"),
                );
                not_imported.push(name.to_string());
            }
            Ok(import) => {
                if !import.diagnostics.fallbacks.is_empty() {
                    let verdict = format!("inexact (fallbacks: {})", import.diagnostics.fallbacks.join(","));
                    row("inexact", None, f64::NAN, f64::NAN, f64::NAN, &verdict);
                    inexact.push(format!("{name}: {verdict}"));
                    continue;
                }
                let kind = import.diagnostics.kind.clone();
                if kind.contains("assembly") || kind.contains("bspline") {
                    // Assembly occurrence transforms / B-spline mesh fallback: the mesh is a
                    // display composition, not a claimed exact solid; mass is not comparable
                    // to the SMLib single-body reference. Keep it visible in the table.
                    println!("{name:<24} {status:<10} mesh-only:{kind}", status = "mesh-only");
                    inexact.push(format!("{name}: mesh-only ({kind})"));
                    continue;
                }
                let mass = mesh_mass(&import.mesh);
                let volume_ok = rel(mass.volume.abs(), ref_volume.abs()) <= tol;
                let area_ok = rel(mass.area, ref_area) <= tol;
                let bbox_err = match (mesh_bbox(&import.mesh), ref_bbox.len()) {
                    (Some((lo, hi)), 6) => {
                        let ext = ref_bbox[3..].iter().zip(ref_bbox[..3].iter()).map(|(h, l)| (h - l).abs()).fold(0.0f64, f64::max).max(1e-9);
                        let mut worst = 0.0f64;
                        for a in 0..3 {
                            worst = worst.max((lo[a] - ref_bbox[a]).abs());
                            worst = worst.max((hi[a] - ref_bbox[3 + a]).abs());
                        }
                        worst / ext
                    }
                    _ => f64::INFINITY,
                };
                let bbox_ok = bbox_err <= bbox_tol;
                let vol_err = rel(mass.volume.abs(), ref_volume.abs());
                let area_err = rel(mass.area, ref_area);
                if name.contains("nist_ftc_09") {
                    nist_ftc_09_checked = true;
                    if !(volume_ok && area_ok) {
                        // The historical relaxation accepted this fixture at
                        // volume 158,638 mm^3 against the SMLib 136,447 mm^3 (+16.2%).
                        println!(
                            "REGRESSION: {name} imports exact but does not match SMLib: mesh volume {:.6} vs {:.6} (rel {:.3e}), area {:.6} vs {:.6} (rel {:.3e})",
                            mass.volume.abs(), ref_volume.abs(), vol_err, mass.area, ref_area, area_err
                        );
                        mismatches.push(name.to_string());
                        continue;
                    }
                }
                row(&format!("exact:{kind}"), Some(mass.volume.abs()), vol_err, area_err, bbox_err, if volume_ok && area_ok && bbox_ok { "match" } else { "MISMATCH" });
                if volume_ok && area_ok && bbox_ok {
                    exact_matched.push(name.to_string());
                } else {
                    mismatches.push(name.to_string());
                }
            }
        }
    }

    println!(
        "\nsummary: {} exact match, {} inexact imports, {} not imported, {} mismatch",
        exact_matched.len(), inexact.len(), not_imported.len(), mismatches.len()
    );
    for line in &inexact { println!("inexact: {line}"); }
    for name in &mismatches { println!("mismatch: {name}"); }

    // Guard: the test must never silently pass by importing nothing exactly.
    assert!(
        !exact_matched.is_empty(),
        "no exact STEP fixture matched its SMLib reference; import has regressed or the test cannot see exact imports"
    );
    // Known-mismatch classification must track the observed mismatches exactly.
    let known: Vec<&str> = KNOWN_MISMATCHES.iter().map(|(name, _)| *name).collect();
    let unexpected: Vec<&String> = mismatches.iter().filter(|name| !known.contains(&name.as_str())).collect();
    assert!(
        unexpected.is_empty(),
        "imported exact fixture(s) do not match the SMLib mass/bbox reference and are not classified as known failures: {unexpected:?}\nAll mismatches: {mismatches:?}"
    );
    for (name, reason) in KNOWN_MISMATCHES {
        assert!(
            mismatches.iter().any(|m| m == name),
            "known failure {name} ({reason}) now matches the SMLib reference; remove it from KNOWN_MISMATCHES"
        );
    }
    // If nist_ftc_09 is importable, it must be exact and correct (see guard above).
    if !nist_ftc_09_checked {
        println!("nist_ftc_09: not imported by the current policy (recorded above as not-imported)");
    }
}
