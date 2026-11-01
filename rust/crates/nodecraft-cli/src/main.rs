//! Thin Rust CLI over the NodeCraft Rust worker.
//!
//! It prepares a graph request, spawns `nodecraft-worker`, and prints the
//! response; evaluation remains entirely in the worker (the new solidcore stack).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use serde_json::{json, Value};

struct Args { graph: PathBuf, targets: Vec<String>, all_targets: bool, mesh: bool, export: Option<PathBuf>, format: Option<String>, json: bool, log: Option<PathBuf>, patches: Vec<String>, time: bool }

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let graph = args.next().ok_or("usage: nodecraft GRAPH.nodecraft [--target ID] [--all-targets] [--mesh] [--time] [--set NODE:FIELD=VALUE] [--log PATH] [--export PATH] [--json]")?;
    let mut out = Args { graph: PathBuf::from(graph), targets: Vec::new(), all_targets: false, mesh: false, export: None, format: None, json: false, log: None, patches: Vec::new(), time: false };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--target" => out.targets.push(args.next().ok_or("--target needs a node id")?),
            "--all-targets" | "--all" => out.all_targets = true,
            "--mesh" => out.mesh = true,
            "--json" => out.json = true,
            "--time" => out.time = true,
            "--log" => out.log = Some(PathBuf::from(args.next().ok_or("--log needs a path")?)),
            "--set" => out.patches.push(args.next().ok_or("--set needs NODE:FIELD=VALUE")?),
            "--export" => out.export = Some(PathBuf::from(args.next().ok_or("--export needs a path")?)),
            "--format" => out.format = Some(args.next().ok_or("--format needs exact|faceted")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(out)
}

fn worker_binary() -> Result<PathBuf, String> {
    if let Ok(value) = std::env::var("NODECRAFT_WORKER_BIN") { return Ok(PathBuf::from(value)); }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for name in ["nodecraft-worker", "nodecraft-worker.exe"] {
                let candidate = parent.join(name);
                if candidate.is_file() { return Ok(candidate); }
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in ["target/release/nodecraft-worker", "target/debug/nodecraft-worker"] {
        let candidate = root.join(relative);
        if candidate.is_file() { return Ok(candidate); }
    }
    Err("nodecraft-worker not found; build with ./rust/scripts/build.sh or set NODECRAFT_WORKER_BIN".into())
}

fn terminal_targets(nodes: &[Value], connections: &[Value]) -> Vec<String> {
    let sources: std::collections::BTreeSet<&str> = connections.iter().filter_map(|link| link.get("from").and_then(Value::as_str)).collect();
    nodes.iter().filter_map(|node| node.get("id").and_then(Value::as_str)).filter(|id| !sources.contains(*id)).map(str::to_string).collect()
}

fn main() {
    if let Err(error) = run() { eprintln!("error: {error}"); std::process::exit(2); }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.time && cfg!(debug_assertions) {
        eprintln!("warning: --time in a debug build; timings are not release-representative");
    }
    let text = std::fs::read_to_string(&args.graph).map_err(|error| format!("{}: {error}", args.graph.display()))?;
    let document: Value = serde_json::from_str(&text).map_err(|error| format!("{}: {error}", args.graph.display()))?;
    let mut nodes = document.get("nodes").and_then(Value::as_array).cloned().ok_or("graph needs nodes[]")?;
    let mut connections = document.get("connections").and_then(Value::as_array).cloned().ok_or("graph needs connections[]")?;
    let viewed: Vec<String> = document.get("viewedNodes").and_then(Value::as_object).map(|values| values.values().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    for patch in &args.patches {
        let (path, raw) = patch.split_once('=').ok_or_else(|| format!("--set expects NODE:FIELD=VALUE, got {patch}"))?;
        let (node_id, field) = path.split_once(':').ok_or_else(|| format!("--set expects NODE:FIELD=VALUE, got {patch}"))?;
        let value = serde_json::from_str::<Value>(raw).unwrap_or_else(|_| json!(raw));
        let node = nodes.iter_mut().find(|node| node.get("id").and_then(Value::as_str) == Some(node_id)).ok_or_else(|| format!("--set node {node_id} not found"))?;
        node["values"][field] = value;
    }
    let mut targets = if args.all_targets {
        nodes.iter().filter_map(|node| node.get("id").and_then(Value::as_str).map(str::to_string)).collect()
    } else if args.targets.is_empty() { terminal_targets(&nodes, &connections) } else { args.targets.clone() };
    for viewed in viewed { if !targets.contains(&viewed) { targets.push(viewed); } }
    if targets.is_empty() { return Err("no target node".into()); }
    if let Some(path) = &args.export {
        let absolute = path.canonicalize().unwrap_or_else(|_| std::env::current_dir().unwrap_or_default().join(path));
        let parent = absolute.parent().ok_or("export path has no parent")?;
        let stem = absolute.file_stem().and_then(|value| value.to_str()).unwrap_or("model");
        std::fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
        let mut exporter_values = json!({"directory":parent.to_string_lossy(),"fileName":stem,"overwrite":true});
        if let Some(format) = &args.format { exporter_values["format"] = json!(format); }
        nodes.push(json!({"id":"__cli_export","type":"exporter","values":exporter_values,"x":0,"y":0}));
        let selected = targets[0].clone();
        let source = if nodes.iter().find(|node| node.get("id").and_then(Value::as_str) == Some(selected.as_str())).and_then(|node| node.get("type")).and_then(Value::as_str) == Some("exporter") {
            connections.iter().find(|link| link.get("to").and_then(Value::as_str) == Some(selected.as_str())).and_then(|link| link.get("from")).and_then(Value::as_str).map(str::to_string).ok_or("selected exporter has no input")?
        } else { selected };
        connections.push(json!({"from":source,"out":"shape","to":"__cli_export","in":"shape"}));
        targets = vec!["__cli_export".into()];
    }
    let request = json!({"requestId":1,"targets":targets,"meshTargets":if args.mesh { targets.clone() } else { Vec::<String>::new() },"nodes":nodes,"connections":connections,"binaryMesh":true});
    let worker = worker_binary()?;
    let stderr = if std::env::var("NODECRAFT_CLI_DEBUG").is_ok() { Stdio::inherit() } else { Stdio::null() };
    let mut child = Command::new(&worker).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(stderr).spawn().map_err(|error| format!("{}: {error}", worker.display()))?;
    child.stdin.as_mut().ok_or("worker stdin")?.write_all(format!("{}\n", request).as_bytes()).map_err(|error| error.to_string())?;
    let mut line = String::new();
    BufReader::new(child.stdout.take().ok_or("worker stdout")?).read_line(&mut line).map_err(|error| error.to_string())?;
    let response: Value = serde_json::from_str(&line).map_err(|error| format!("bad worker response: {error}"))?;
    if let Some(path) = &args.log {
        let errors = response.get("nodeErrors").and_then(Value::as_object).cloned().unwrap_or_default();
        let entry = json!({
            "graph": args.graph.to_string_lossy(),
            "targets": targets,
            "ok": errors.is_empty(),
            "evaluationMs": response.get("evaluationMs").and_then(Value::as_f64),
            "nodeErrors": errors,
            "emptyTargets": response.get("emptyTargets").cloned().unwrap_or_else(|| json!([])),
            "meshes": response.get("meshes").and_then(Value::as_object).map(|meshes| meshes.iter().map(|(id, payload)| (id.clone(), json!({
                "triangles": payload.get("mesh").and_then(|mesh| mesh.get("faces")).and_then(Value::as_array).map_or(0, Vec::len),
                "vertices": payload.get("mesh").and_then(|mesh| mesh.get("vertices")).and_then(Value::as_array).map_or(0, Vec::len),
            }))).collect::<serde_json::Map<_, _>>()),
        });
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        writeln!(file, "{}", serde_json::to_string(&entry).unwrap()).map_err(|error| error.to_string())?;
    }
    if args.time {
        let ms = response.get("evaluationMs").and_then(Value::as_f64).unwrap_or(f64::NAN);
        println!("TIME {ms:.3} ms");
    }
    if args.json { println!("{}", serde_json::to_string_pretty(&response).unwrap()); }
    else {
        let errors = response.get("nodeErrors").and_then(Value::as_object).cloned().unwrap_or_default();
        let empty = response.get("emptyTargets").and_then(Value::as_array).cloned().unwrap_or_default();
        for (id, message) in &errors { println!("ERROR {id}: {}", message.as_str().unwrap_or("")); }
        if !empty.is_empty() { println!("EMPTY {}", empty.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")); }
        if let Some(meshes) = response.get("meshes").and_then(Value::as_object) {
            for (id, payload) in meshes {
                if let Some(mesh) = payload.get("mesh") {
                    let triangles = mesh.get("faces").and_then(Value::as_array).map_or(0, Vec::len);
                    let vertices = mesh.get("vertices").and_then(Value::as_array).map_or(0, Vec::len);
                    if triangles > 0 || vertices > 0 { println!("{id}: triangles={triangles} vertices={vertices}"); }
                }
            }
        }
        if errors.is_empty() && empty.is_empty() { println!("OK"); }
    }
    if let Some(path) = args.export {
        let produced = path.canonicalize().unwrap_or(path);
        if !produced.exists() { return Err(format!("worker did not produce {}", produced.display())); }
        println!("exported: {}", produced.display());
    }
    if !response.get("nodeErrors").and_then(Value::as_object).map_or(true, |errors| errors.is_empty()) { std::process::exit(1); }
    Ok(())
}
