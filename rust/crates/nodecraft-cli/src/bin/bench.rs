//! Bounded graph benchmark.
//!
//! Runs one persistent worker so cold, warm and edit-cache phases measure the
//! same process. Every phase has a hard timeout; on timeout the worker is
//! killed instead of blocking the harness.
//!
//! Usage:
//!   nodecraft-bench GRAPH.nodecraft [--edits N] [--timeout-ms MS] [--mesh]
//!                    [--target ID] [--json]

use serde_json::{Value, json};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

struct Args {
    graph: PathBuf,
    edits: usize,
    timeout: Duration,
    mesh: bool,
    target: Option<String>,
    json: bool,
    fillet_edits: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let graph = args.next().ok_or("usage: nodecraft-bench GRAPH.nodecraft [--edits N] [--timeout-ms MS] [--mesh] [--target ID] [--json]")?;
    let mut out = Args {
        graph: PathBuf::from(graph),
        edits: 3,
        timeout: Duration::from_secs(30),
        mesh: false,
        target: None,
        json: false,
        fillet_edits: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--edits" => {
                out.edits = args
                    .next()
                    .ok_or("--edits needs a number")?
                    .parse()
                    .map_err(|error| format!("--edits: {error}"))?
            }
            "--timeout-ms" => {
                out.timeout = Duration::from_millis(
                    args.next()
                        .ok_or("--timeout-ms needs milliseconds")?
                        .parse()
                        .map_err(|error| format!("--timeout-ms: {error}"))?,
                )
            }
            "--mesh" => out.mesh = true,
            "--target" => out.target = Some(args.next().ok_or("--target needs a node id")?),
            "--json" => out.json = true,
            "--fillet-edits" => out.fillet_edits = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(out)
}

fn worker_binary() -> Result<PathBuf, String> {
    if let Ok(value) = std::env::var("NODECRAFT_WORKER_BIN") {
        return Ok(PathBuf::from(value));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for relative in ["target/release/nodecraft-worker", "target/debug/nodecraft-worker"] {
        let candidate = root.join(relative);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err("nodecraft-worker not found; build the workspace or set NODECRAFT_WORKER_BIN".into())
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Worker {
    fn spawn() -> Result<Self, String> {
        let binary = worker_binary()?;
        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("{}: {error}", binary.display()))?;
        let stdin = child.stdin.take().ok_or("worker stdin")?;
        let stdout = child.stdout.take().ok_or("worker stdout")?;
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self { child, stdin, lines: rx })
    }

    fn request(&mut self, request: &Value, timeout: Duration) -> Result<Value, String> {
        let line = serde_json::to_string(request).map_err(|error| error.to_string())?;
        self.stdin
            .write_all(format!("{line}\n").as_bytes())
            .and_then(|()| self.stdin.flush())
            .map_err(|error| format!("worker write: {error}"))?;
        let line = match self.lines.recv_timeout(timeout) {
            Ok(line) => line,
            Err(_) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Err(format!("timeout after {} ms", timeout.as_millis()));
            }
        };
        serde_json::from_str(&line).map_err(|error| format!("bad worker response: {error}"))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() {
    if let Err(error) = nodecraft_engine::thread_pools::init_rayon_pool() {
        eprintln!("{error}");
    }
    let kernel_fixes = std::env::var("NODECRAFT_KERNEL_FIXES").map_or(true, |value| value != "0");
    nodecraft_engine::configure_solidcore_fixes(kernel_fixes);
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let load_start = Instant::now();
    let text = std::fs::read_to_string(&args.graph)
        .map_err(|error| format!("{}: {error}", args.graph.display()))?;
    let document: Value = serde_json::from_str(&text)
        .map_err(|error| format!("{}: {error}", args.graph.display()))?;
    let root = document.get("root").unwrap_or(&document);
    let mut nodes =
        root.get("nodes").and_then(Value::as_array).cloned().ok_or("graph needs nodes[]")?;
    let connections =
        root.get("connections").and_then(Value::as_array).cloned().unwrap_or_default();
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let target = match args.target {
        Some(target) => target,
        None => terminal_target(&nodes, &connections).ok_or("graph has no terminal target")?,
    };
    if args.fillet_edits {
        return run_fillet_edits(&nodes, &connections, &target, args.timeout);
    }
    let edit_index = nodes.iter().position(|node| numeric_field(node).is_some());
    let mesh_targets = if args.mesh { vec![target.clone()] } else { Vec::new() };
    let mut worker = Worker::spawn()?;
    let mut results = Vec::new();

    let mut run_phase = |name: &str,
                         nodes: &[Value],
                         worker: &mut Worker|
     -> Result<Value, String> {
        let keyed = keyed_nodes(nodes, &connections);
        let request = json!({
            "requestId": 1,
            "targets": [target],
            "meshTargets": mesh_targets,
            "binaryMesh": false,
            "nodes": keyed,
            "connections": connections,
        });
        let started = Instant::now();
        let response = worker.request(&request, args.timeout)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let errors =
            response.get("nodeErrors").and_then(Value::as_object).cloned().unwrap_or_default();
        let entry = json!({
            "phase": name,
            "wallMs": wall_ms,
            "evaluationMs": response.get("evaluationMs").cloned().unwrap_or(Value::Null),
            "responseBytes": serde_json::to_vec(&response).map(|bytes| bytes.len()).unwrap_or(0),
            "triangles": mesh_stat(&response, "faces"),
            "vertices": mesh_stat(&response, "vertices"),
            "cacheHit": response.get("cacheStats").and_then(|stats| stats.get("hits")).cloned().unwrap_or(Value::Null),
            "cacheMiss": response.get("cacheStats").and_then(|stats| stats.get("misses")).cloned().unwrap_or(Value::Null),
            "errors": errors,
        });
        results.push(entry);
        Ok(response)
    };

    let cold = run_phase("cold", &nodes, &mut worker)?;
    run_phase("warm", &nodes, &mut worker)?;
    let mut response = cold;
    for edit in 0..args.edits {
        let Some(edit_index) = edit_index else { break };
        let (field, next) = {
            let node = &mut nodes[edit_index];
            let (field, value) = numeric_field(node).ok_or("edit node lost its numeric field")?;
            let next = json!(value + 0.01 * (edit + 1) as f64);
            node["values"][&field] = next.clone();
            (field, next)
        };
        let phase = format!("edit-{}-{}={}", edit + 1, field, next);
        response = run_phase(&phase, &nodes, &mut worker)?;
    }
    let summary = json!({
        "graph": args.graph.to_string_lossy(),
        "target": target,
        "loadMs": load_ms,
        "edits": args.edits,
        "mesh": args.mesh,
        "phases": results,
        "ok": response.get("nodeErrors").and_then(Value::as_object).map_or(true, |errors| errors.is_empty()),
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary).map_err(|error| error.to_string())?);
    } else {
        println!(
            "graph: {}  target: {}  load: {load_ms:.1} ms",
            args.graph.display(),
            summary["target"].as_str().unwrap_or("")
        );
        println!(
            "{:<28} {:>10} {:>12} {:>10} {:>10}",
            "phase", "wall ms", "eval ms", "triangles", "cache"
        );
        for phase in summary["phases"].as_array().into_iter().flatten() {
            println!(
                "{:<28} {:>10.1} {:>12.1} {:>10} {:>10}",
                phase["phase"].as_str().unwrap_or(""),
                phase["wallMs"].as_f64().unwrap_or(0.0),
                phase["evaluationMs"].as_f64().unwrap_or(0.0),
                phase["triangles"].as_u64().unwrap_or(0),
                format!(
                    "{}/{}",
                    phase["cacheHit"].as_u64().unwrap_or(0),
                    phase["cacheMiss"].as_u64().unwrap_or(0)
                ),
            );
        }
        if !summary["ok"].as_bool().unwrap_or(false) {
            eprintln!("benchmark reported node errors");
        }
    }
    if !summary["ok"].as_bool().unwrap_or(false) {
        std::process::exit(1);
    }
    Ok(())
}

fn run_fillet_edits(
    nodes: &[Value],
    connections: &[Value],
    target: &str,
    timeout: Duration,
) -> Result<(), String> {
    use nodecraft_engine::CancelToken;
    use nodecraft_engine::kernel_pool::set_current_revision_token;
    let fillet = nodes
        .iter()
        .position(|node| node.get("type").and_then(Value::as_str) == Some("fillet"))
        .ok_or("graph has no fillet node")?;
    let mut nodes = nodes.to_vec();
    let edits: [(f64, &str); 5] = [(1.0, ""), (1.2, ""), (1.4, ""), (1.6, ""), (1.6, "G2")];
    let (tx, rx) = std::sync::mpsc::channel();
    let mut previous: Option<CancelToken> = None;
    for (index, (length, continuity)) in edits.iter().copied().enumerate() {
        nodes[fillet]["values"]["length"] = json!(length);
        nodes[fillet]["values"]["continuity"] = json!(continuity);
        let keyed = keyed_nodes(&nodes, connections);
        let request = json!({
            "requestId": index as u64 + 1,
            "targets": [target],
            "meshTargets": [],
            "binaryMesh": false,
            "nodes": keyed,
            "connections": connections,
        });
        let request: nodecraft_protocol::Request =
            serde_json::from_value(request).map_err(|error| format!("request: {error}"))?;
        let token = CancelToken::new();
        if let Some(previous) = previous.take() {
            previous.cancel();
        }
        previous = Some(token.clone());
        let tx = tx.clone();
        std::thread::spawn(move || {
            set_current_revision_token(token.clone());
            let started = Instant::now();
            let output = nodecraft_engine::evaluate_exact(
                request,
                nodecraft_engine::EngineCache::new(),
                None,
            );
            let _ = tx.send((
                index + 1,
                length,
                continuity,
                token.is_cancelled(),
                started.elapsed(),
                output.response.node_errors.len(),
            ));
        });
        std::thread::sleep(Duration::from_millis(30));
    }
    drop(tx);
    println!(
        "{:<5} {:>6} {:>4} {:>10} {:>9} {:>8}",
        "edit", "size", "cont", "cancelled", "wall ms", "errors"
    );
    let mut seen = 0;
    let mut last_result = None;
    while seen < edits.len() {
        match rx.recv_timeout(timeout) {
            Ok((index, length, continuity, cancelled, elapsed, errors)) => {
                seen += 1;
                if index == edits.len() {
                    last_result = Some(elapsed);
                }
                println!(
                    "{:<5} {:>6.1} {:>4} {:>10} {:>9.1} {:>8}",
                    index,
                    length,
                    if continuity.is_empty() { "G1" } else { continuity },
                    if cancelled { "yes" } else { "no" },
                    elapsed.as_secs_f64() * 1000.0,
                    errors,
                );
            }
            Err(_) => {
                println!(
                    "edit {}: TIMEOUT after {:.0} ms",
                    seen + 1,
                    timeout.as_secs_f64() * 1000.0
                );
                break;
            }
        }
    }
    match last_result {
        Some(elapsed) => println!("last edit -> result: {:.1} ms", elapsed.as_secs_f64() * 1000.0),
        None => println!("last edit -> result: TIMEOUT / still running"),
    }
    Ok(())
}

fn terminal_target(nodes: &[Value], connections: &[Value]) -> Option<String> {
    let sources: std::collections::BTreeSet<&str> =
        connections.iter().filter_map(|link| link.get("from").and_then(Value::as_str)).collect();
    nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(Value::as_str))
        .find(|id| !sources.contains(id))
        .map(str::to_string)
}

fn numeric_field(node: &Value) -> Option<(String, f64)> {
    node.get("values")?
        .as_object()?
        .iter()
        .find_map(|(field, value)| value.as_f64().map(|value| (field.clone(), value)))
}

fn mesh_stat(response: &Value, key: &str) -> usize {
    response
        .get("meshes")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(_, payload)| {
            payload.get("mesh").and_then(|mesh| mesh.get(key)).and_then(Value::as_array)
        })
        .map(Vec::len)
        .sum()
}

fn keyed_nodes(nodes: &[Value], connections: &[Value]) -> Vec<Value> {
    let mut incoming: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
    for link in connections {
        let (Some(to), Some(from), input) = (
            link.get("to").and_then(Value::as_str),
            link.get("from").and_then(Value::as_str),
            link.get("in").and_then(Value::as_str).unwrap_or(""),
        ) else {
            continue;
        };
        incoming.entry(to).or_default().push((input, from));
    }
    let by_id: HashMap<&str, &Value> = nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(Value::as_str).map(|id| (id, node)))
        .collect();
    let mut memo: HashMap<String, u64> = HashMap::new();
    let mut visiting = std::collections::HashSet::new();
    fn content_key(
        id: &str,
        incoming: &HashMap<&str, Vec<(&str, &str)>>,
        by_id: &HashMap<&str, &Value>,
        memo: &mut HashMap<String, u64>,
        visiting: &mut std::collections::HashSet<String>,
    ) -> u64 {
        if let Some(key) = memo.get(id) {
            return *key;
        }
        if !visiting.insert(id.to_string()) {
            return 0;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if let Some(node) = by_id.get(id) {
            node.get("type").and_then(Value::as_str).unwrap_or("").hash(&mut hasher);
            node.get("values")
                .map_or_else(|| "null".to_owned(), |values| values.to_string())
                .hash(&mut hasher);
        }
        let mut upstream = incoming.get(id).cloned().unwrap_or_default();
        upstream.sort();
        for (input, from) in upstream {
            input.hash(&mut hasher);
            content_key(from, incoming, by_id, memo, visiting).hash(&mut hasher);
        }
        visiting.remove(id);
        let key = hasher.finish();
        memo.insert(id.to_string(), key);
        key
    }
    nodes
        .iter()
        .map(|node| {
            let mut node = node.clone();
            let id = node.get("id").and_then(Value::as_str).unwrap_or("").to_string();
            let key = content_key(&id, &incoming, &by_id, &mut memo, &mut visiting);
            node["cache"] = json!({"key": format!("{key:016x}"), "state": "cached"});
            node
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> (Vec<Value>, Vec<Value>) {
        let nodes = vec![
            json!({"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10.0}}),
            json!({"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20.0}}),
            json!({"id": "u", "type": "boolean", "values": {"mode": "union"}}),
        ];
        let connections = vec![
            json!({"from": "a", "to": "u", "in": "a", "out": "shape"}),
            json!({"from": "b", "to": "u", "in": "b", "out": "shape"}),
        ];
        (nodes, connections)
    }

    fn keys(nodes: &[Value], connections: &[Value]) -> HashMap<String, String> {
        keyed_nodes(nodes, connections)
            .into_iter()
            .map(|node| {
                (
                    node["id"].as_str().unwrap().to_owned(),
                    node["cache"]["key"].as_str().unwrap().to_owned(),
                )
            })
            .collect()
    }

    #[test]
    fn an_edit_only_changes_the_edited_branch_and_downstream() {
        let (mut nodes, connections) = graph();
        let before = keys(&nodes, &connections);
        nodes[1]["values"]["length"] = json!(21.0);
        let after = keys(&nodes, &connections);
        assert_eq!(before["a"], after["a"]);
        assert_ne!(before["b"], after["b"]);
        assert_ne!(before["u"], after["u"]);
    }

    #[test]
    fn terminal_target_finds_the_downstream_node() {
        let (nodes, connections) = graph();
        assert_eq!(terminal_target(&nodes, &connections).as_deref(), Some("u"));
    }
}
