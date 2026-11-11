//! Import one STEP file through the current evaluation stack and write its display mesh.
//!
//! Usage: step_report_mesh <input.step> <output.obj> <output.json>
//! Used by the generic report tooling under reports/; it deliberately takes one file
//! per process so a bad import cannot take down a batch.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::time::Instant;

use nodecraft_engine::{evaluate_exact, EngineCache};
use nodecraft_protocol::Request;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: step_report_mesh <input.step> <output.obj> <output.json>");
        std::process::exit(2);
    }
    let input = PathBuf::from(&args[1]);
    let obj_path = PathBuf::from(&args[2]);
    let json_path = PathBuf::from(&args[3]);
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| run(&input)));
    let elapsed_ms = started.elapsed().as_millis();
    let (record, mesh) = match result {
        Ok(value) => value,
        Err(_) => (
            serde_json::json!({
                "ok": false,
                "error": "panic while importing or tessellating",
                "category": "crash",
            }),
            None,
        ),
    };
    let mut record = record;
    record["input"] = serde_json::json!(input.to_string_lossy());
    record["bytes"] = serde_json::json!(std::fs::metadata(&input).map(|m| m.len()).unwrap_or(0));
    record["elapsed_ms"] = serde_json::json!(elapsed_ms);
    if let Some((vertices, faces)) = mesh {
        let mut obj = String::with_capacity(vertices.len() * 32 + faces.len() * 16);
        for v in &vertices {
            obj.push_str(&format!("v {} {} {}\n", v[0], v[1], v[2]));
        }
        for f in &faces {
            obj.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        if let Err(error) = std::fs::write(&obj_path, obj) {
            record["ok"] = serde_json::json!(false);
            record["error"] = serde_json::json!(format!("write OBJ: {error}"));
            record["category"] = serde_json::json!("output");
        } else {
            record["faces"] = serde_json::json!(faces.len());
            record["vertices"] = serde_json::json!(vertices.len());
        }
    }
    if let Some(parent) = json_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&json_path, serde_json::to_vec_pretty(&record).unwrap_or_default());
    if record["ok"].as_bool() != Some(true) {
        std::process::exit(1);
    }
}

type MeshData = (Vec<[f64; 3]>, Vec<[u32; 3]>);

fn run(input: &PathBuf) -> (serde_json::Value, Option<MeshData>) {
    let nodes = serde_json::json!([{
        "id": "n1",
        "type": "stepimporter",
        "values": { "path": input.to_string_lossy(), "heal": true },
        "cache": { "key": "", "state": "dirty" },
        "suppressed": false,
        "x": 0,
        "y": 0
    }]);
    let request = match serde_json::from_value::<Request>(serde_json::json!({
        "requestId": 1,
        "targets": ["n1"],
        "meshTargets": ["n1"],
        "binaryMesh": false,
        "nodes": nodes,
        "connections": []
    })) {
        Ok(request) => request,
        Err(error) => {
            return (
                serde_json::json!({"ok": false, "error": error.to_string(), "category": "request"}),
                None,
            )
        }
    };
    let output = evaluate_exact(request.clone(), EngineCache::new(), None);
    let mut plans = nodecraft_display::PlanCache::new(16);
    let (meshes, _) = nodecraft_display::protocol_meshes(
        &request,
        &output.exact_parts,
        nodecraft_display::DisplayQuality::Full(0.35),
        &mut plans,
    );
    let error = output
        .response
        .node_errors
        .get("n1")
        .cloned()
        .filter(|error| !error.trim().is_empty());
    let mesh = meshes.get("n1").map(|payload| &payload.mesh);
    if let Some(error) = error {
        return (
            serde_json::json!({
                "ok": false,
                "error": error,
                "category": classify(&error),
            }),
            None,
        );
    }
    let Some(mesh) = mesh else {
        return (
            serde_json::json!({
                "ok": false,
                "error": "importer returned no mesh",
                "category": "no-output",
            }),
            None,
        );
    };
    if mesh.faces.is_empty() {
        return (
            serde_json::json!({
                "ok": false,
                "error": "importer returned an empty mesh",
                "category": "empty",
            }),
            None,
        );
    }
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect::<Vec<_>>();
    let faces = mesh
        .faces
        .iter()
        .map(|f| [f[0], f[1], f[2]])
        .collect::<Vec<_>>();
    (
        serde_json::json!({
            "ok": true,
            "category": "success",
            "faces": faces.len(),
            "vertices": vertices.len(),
            "brep_edges": mesh.brep_edges.len(),
            "face_groups": mesh.face_groups.len(),
        }),
        Some((vertices, faces)),
    )
}

fn classify(error: &str) -> &'static str {
    let text = error.to_ascii_lowercase();
    if text.contains("no solidcore step importer") {
        "unsupported"
    } else if text.contains("tessellation") || text.contains("display mesh") || text.contains("triangle") {
        "tessellation"
    } else if text.contains("timed out") {
        "timeout"
    } else if text.contains("panic") {
        "crash"
    } else if text.contains("invalid") || text.contains("parse") || text.contains("read") {
        "parse"
    } else {
        "import"
    }
}
