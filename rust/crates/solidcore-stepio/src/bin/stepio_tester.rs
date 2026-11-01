//! Stepio corpus tester.
//!
//! Walks every `.step`/`.stp` fixture under `tests/`, imports it through the real
//! Stepio façade, and reports a tally by outcome/category/root. It is intentionally a
//! reporting tool: it does not assert success.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct Outcome {
    relative: String,
    status: &'static str,
    kind: String,
    category: &'static str,
    detail: String,
    millis: f64,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3).expect("repo root").to_path_buf()
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() { collect(&path, out); }
        else if path.extension().and_then(|value| value.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("step") || ext.eq_ignore_ascii_case("stp")) { out.push(path); }
    }
}

fn category(error: &str) -> &'static str {
    if error.contains("STEP file has no geometry") { "non-geometric" }
    else if error.contains("STEP assembly") { "assembly" }
    else if error.contains("mixed planar/analytic") { "mixed-solid" }
    else if error.contains("STEP validation failed") { "validation" }
    else if error.contains("no MANIFOLD_SOLID_BREP") { "no-solid-topology" }
    else if error.contains("B-spline") { "bspline" }
    else if error.contains("surface-model") || error.contains("surface model") { "surface-model" }
    else if error.contains("solidcore") { "solidcore-error" }
    else if error.contains("STEP file has no") { "no-fixture-geometry" }
    else if error.contains("STEP header validation") { "header" }
    else { "other" }
}

/// Conservative: only explicit missing-kernel-capability errors are upstream blockers.
fn known_invalid_fixture(error: &str) -> bool {
    // The STEPcode ATS1 fixture is deliberately non-conformant: its
    // PRODUCT_RELATED_PRODUCT_CATEGORY has an empty product set (SET [1:?]).
    error.contains("empty-reference-list") && error.contains("PRODUCT_RELATED_PRODUCT_CATEGORY")
}

fn blocked_by_upstream(error: &str) -> bool {
    error.contains("not available in solidcore")
        || error.contains("solidcore does not implement")
        || error.contains("SolidError::Unported")
}

fn mesh_bounds(mesh: &nodecraft_protocol::Mesh) -> ([f64; 3], [f64; 3]) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis] as f64);
            max[axis] = max[axis].max(vertex[axis] as f64);
        }
    }
    (min, max)
}

/// Classifications whose import is a display representation, never an exact B-rep.
///
/// `bspline-solid-mesh` is a display mesh produced by the B-spline fallback: it is
/// deliberately never checked against exact STEP bounds or mass properties.
fn display_kind(kind: &str) -> bool {
    matches!(kind, "surface-model" | "sheet-brep" | "bspline-solid-mesh")
}

fn geometry_mismatch(path: &Path, mesh: &nodecraft_protocol::Mesh, kind: &str) -> Option<String> {
    if display_kind(kind) || kind == "multi-solid" || kind == "assembly" { return None; }
    let Ok((want_min, want_max)) = nodecraft_step::read_topology_bounds(&path.to_string_lossy()) else { return None; };
    let (got_min, got_max) = mesh_bounds(mesh);
    let extent = (0..3).map(|axis| (want_max[axis] - want_min[axis]).abs()).fold(0.0f64, f64::max).max(1.0);
    let tolerance = extent * 1e-3;
    for axis in 0..3 {
        if (got_min[axis] - want_min[axis]).abs() > tolerance || (got_max[axis] - want_max[axis]).abs() > tolerance {
            return Some(format!("mesh bbox {:?}..{:?} differs from STEP bounds {:?}..{:?} (tol {tolerance})", got_min, got_max, want_min, want_max));
        }
    }
    None
}

fn outcome(path: &Path) -> Outcome {
    let root = repo_root();
    let relative = path.strip_prefix(&root).unwrap_or(path).display().to_string();
    let started = Instant::now();
    let (tx, rx) = std::sync::mpsc::channel();
    let path_text = path.to_string_lossy().into_owned();
    std::thread::spawn(move || { let _ = tx.send(solidcore_stepio::import_step(&path_text)); });
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    match rx.recv_timeout(Duration::from_secs(90)) {
        Ok(Ok(import)) => {
            let kind = import.diagnostics.kind.clone();
            let mismatch = geometry_mismatch(path, &import.mesh, &kind);
            // A B-spline solid is only a display mesh: require an explicit display/trim
            // diagnostic so it can never silently masquerade as an exact import.
            let has_bspline_diagnostic = import.diagnostics.warnings.iter().any(|warning| {
                warning.contains("B-spline surface") || warning.contains("display-only") || warning.contains("display mesh")
            });
            let (status, category) = if kind == "surface-model" {
                ("surface-model", "surface-model")
            } else if kind == "sheet-brep" {
                ("sheet-model", "sheet-model")
            } else if kind == "bspline-solid-mesh" {
                if has_bspline_diagnostic { ("mesh-model", "mesh-model") } else { ("suspect", "mesh-model-no-diagnostic") }
            } else if kind == "assembly" {
                ("assembly", "assembly")
            } else if kind == "multi-solid" {
                ("multi-solid", "multi-solid")
            } else if mismatch.is_some() {
                ("suspect", "geometry-mismatch")
            } else {
                ("exact", "ok")
            };
            let detail = if kind == "bspline-solid-mesh" {
                format!(
                    "display-only mesh ({}); {} verts, {} tris, {} warning(s)",
                    if has_bspline_diagnostic { "B-spline diagnostic present" } else { "MISSING B-spline display diagnostic" },
                    import.mesh.vertices.len(), import.mesh.faces.len(), import.diagnostics.warnings.len()
                )
            } else {
                mismatch.unwrap_or_else(|| format!("{} verts, {} tris, {} warning(s)", import.mesh.vertices.len(), import.mesh.faces.len(), import.diagnostics.warnings.len()))
            };
            Outcome { relative, status, kind, category, detail, millis: elapsed }
        }
        Ok(Err(error)) => {
            let category = category(&error);
            Outcome { relative, status: "unsupported", kind: String::new(), category, detail: error, millis: elapsed }
        }
        Err(_) => Outcome { relative, status: "timeout", kind: String::new(), category: "timeout", detail: "import exceeded 90 s".into(), millis: elapsed },
    }
}

fn main() {
    let root = repo_root();
    let mut files = Vec::new();
    for dir in ["tests/Sample AP203 Data Files", "tests/step-nist", "tests/step-parts", "tests/step-stepcode", "tests/step-truck"] {
        collect(&root.join(dir), &mut files);
    }
    files.sort();
    files.dedup();
    let jobs = std::env::args()
        .position(|arg| arg == "--jobs")
        .and_then(|index| std::env::args().nth(index + 1))
        .or_else(|| std::env::var("STEPIO_JOBS").ok())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|value| value.get()).unwrap_or(8))
        .max(8)
        .clamp(1, files.len().max(1));
    let started = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<Option<Outcome>>> = std::sync::Mutex::new(vec![None; files.len()]);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if index >= files.len() { break; }
                let result = outcome(&files[index]);
                results.lock().expect("results")[index] = Some(result);
            });
        }
    });
    let outcomes: Vec<Outcome> = results.into_inner().expect("results").into_iter().map(|item| item.expect("outcome")).collect();
    let elapsed = started.elapsed().as_secs_f64();

    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    let mut categories: BTreeMap<&str, usize> = BTreeMap::new();
    let mut roots: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    for outcome in &outcomes {
        *tally.entry(outcome.status).or_default() += 1;
        *categories.entry(outcome.category).or_default() += 1;
        let root = outcome.relative.split('/').nth(1).unwrap_or("other").to_string();
        *roots.entry(root).or_default().entry(outcome.status).or_default() += 1;
    }

    if std::env::args().any(|arg| arg == "--json") {
        let rows: Vec<_> = outcomes.iter().map(|outcome| serde_json::json!({
            "path": outcome.relative, "status": outcome.status, "kind": outcome.kind,
            "category": outcome.category, "displayOnly": display_kind(&outcome.kind),
            "exactClaim": outcome.status == "exact",
            "blockedByUpstream": blocked_by_upstream(&outcome.detail),
            "detail": outcome.detail, "ms": outcome.millis
        })).collect();
        println!("{}", serde_json::to_string_pretty(&rows).expect("json"));
        return;
    }

    println!("Stepio corpus: {} fixtures in {:.2}s with {} job(s)", outcomes.len(), elapsed, jobs);
    println!("status: {}", tally.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join("  "));
    println!("categories: {}", categories.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join("  "));
    println!();
    let mut root_rows: Vec<_> = roots.iter().collect();
    root_rows.sort_by_key(|(root, _)| root.as_str());
    for (root, counts) in root_rows { println!("{root}: {}", counts.iter().map(|(key, value)| format!("{key}={value}")).collect::<Vec<_>>().join("  ")); }
    println!();
    // Display classes (mesh-model/sheet-model) are reported in the matrix but are not
    // failures: they deliberately make no exact claim.
    let display_count = outcomes.iter().filter(|outcome| matches!(outcome.status, "mesh-model" | "sheet-model")).count();
    println!("display-only (not exact): {display_count} fixture(s)");
    let mut failures: Vec<_> = outcomes.iter().filter(|outcome| !matches!(outcome.status, "exact" | "surface-model" | "mesh-model" | "sheet-model")).collect();
    failures.sort_by(|a, b| a.category.cmp(b.category).then(a.relative.cmp(&b.relative)));
    for failure in failures {
        println!("[{:>17}] {}: {}", failure.category, failure.relative, failure.detail);
        let classification = if known_invalid_fixture(&failure.detail) { "INVALID-FIXTURE" } else if blocked_by_upstream(&failure.detail) { "BLOCKED-UPSTREAM" } else { "LOCAL/DOABLE" };
        println!("                    classification: {classification}");
    }
}
