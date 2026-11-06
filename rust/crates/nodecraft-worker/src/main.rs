//! NodeCraft worker transport.
//!
//! The evaluation engine and authoritative session live in Rust. This binary owns
//! stdin, the engine thread, progress serialization, and the output queue.

mod layout;

use serde::Deserialize;
use nodecraft_engine::{evaluate, EngineCache, ProgressEvent};
use nodecraft_protocol::Request;
use nodecraft_session::{Command, GraphDocument, GraphRuntime};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::mpsc;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionEnvelope {
    action: String,
    tab_id: String,
    #[serde(default)]
    document: Option<GraphDocument>,
    #[serde(default)]
    command: Option<Command>,
    #[serde(default)]
    commands: Vec<Command>,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    progress: bool,
    #[serde(default)]
    mesh_tolerance: Option<f64>,
    #[serde(default)]
    request_id: Option<u64>,
    #[serde(default)]
    mesh_targets: Vec<String>,
    #[serde(default)]
    force: bool,
    #[serde(default)]
    coalesce_key: Option<String>,
    #[serde(default)]
    view_targets: Option<Vec<String>>,
    #[serde(default)]
    export_targets: Vec<String>,
    #[serde(default)]
    selected: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentRequest {
    action: String,
    path: String,
    #[serde(default)]
    tab_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    document: Option<serde_json::Value>,
}

enum Incoming { Evaluation(Request), Session(SessionEnvelope), Document(DocumentRequest) }

fn emit_progress(out: &mpsc::Sender<Vec<u8>>, event: ProgressEvent) {
    let frame = serde_json::json!({
        "type": "evaluation-progress",
        "requestId": event.request_id,
        "tabId": event.tab_id,
        "graphRevision": event.graph_revision,
        "nodeId": event.node_id,
        "phase": event.phase.as_str(),
    });
    let _ = out.send(format!("{frame}\n").into_bytes());
}

fn send(out: &mpsc::Sender<Vec<u8>>, value: serde_json::Value) {
    if let Ok(text) = serde_json::to_string(&value) {
        eprintln!("RESP {}", text.chars().take(300).collect::<String>());
        let _ = out.send(format!("{text}\n").into_bytes());
    }
}

fn session_error(out: &mpsc::Sender<Vec<u8>>, tab_id: &str, message: impl Into<String>) {
    send(out, serde_json::json!({"type":"session-error","tabId":tab_id,"error":message.into()}));
}

fn terminal_targets(document: &GraphDocument) -> Vec<String> {
    let sources: std::collections::BTreeSet<&str> = document.connections.iter().map(|link| link.from.as_str()).collect();
    document.nodes.iter().filter(|node| !sources.contains(node.id.as_str())).map(|node| node.id.clone()).collect()
}

fn handle_session(envelope: SessionEnvelope, sessions: &mut HashMap<String, GraphRuntime>, out: &mpsc::Sender<Vec<u8>>) {
    let tab = envelope.tab_id.clone();
    match envelope.action.as_str() {
        "sessionOpen" => {
            let Some(document) = envelope.document else {
                session_error(out, &tab, "sessionOpen requires a document"); return;
            };
            let runtime = GraphRuntime::new(document);
            let revision = runtime.revision();
            let document = serde_json::to_value(runtime.document()).unwrap_or(serde_json::Value::Null);
            sessions.insert(tab.clone(), runtime);
            send(out, serde_json::json!({"type":"session-opened","tabId":tab,"revision":revision,"document":document}));
        }
        "sessionCommand" => {
            let commands = if !envelope.commands.is_empty() {
                envelope.commands.clone()
            } else if let Some(command) = envelope.command.clone() {
                vec![command]
            } else {
                session_error(out, &tab, "sessionCommand requires a command"); return;
            };
            let Some(runtime) = sessions.get_mut(&tab) else {
                session_error(out, &tab, "session is not open"); return;
            };
            match runtime.apply_commands_coalesced(&commands, envelope.coalesce_key.as_deref(), std::time::Duration::from_millis(700)) {
                Ok(change) => send(out, serde_json::json!({
                    "type":"session-command","tabId":tab,"revision":change.revision,
                    "changed":change.changed,"dirtyNodes":change.dirty_nodes,
                    "canUndo":runtime.session().has_undo(),"canRedo":runtime.session().has_redo(),
                })),
                Err(error) => session_error(out, &tab, error.to_string()),
            }
        }
        "sessionDerived" => {
            if envelope.commands.is_empty() {
                session_error(out, &tab, "sessionDerived requires commands"); return;
            }
            let Some(runtime) = sessions.get_mut(&tab) else {
                session_error(out, &tab, "session is not open"); return;
            };
            match runtime.apply_derived_commands(&envelope.commands) {
                Ok(change) => send(out, serde_json::json!({
                    "type":"session-derived","tabId":tab,"revision":change.revision,
                    "changed":change.changed,"dirtyNodes":change.dirty_nodes,
                    "canUndo":runtime.session().has_undo(),"canRedo":runtime.session().has_redo(),
                })),
                Err(error) => session_error(out, &tab, error.to_string()),
            }
        }
        "sessionUndo" | "sessionRedo" => {
            let Some(runtime) = sessions.get_mut(&tab) else {
                session_error(out, &tab, "session is not open"); return;
            };
            let result = if envelope.action == "sessionUndo" { runtime.undo() } else { runtime.redo() };
            match result {
                Ok(Some(change)) => send(out, serde_json::json!({
                    "type":"session-undo","tabId":tab,"revision":change.revision,
                    "changed":change.changed,"dirtyNodes":change.dirty_nodes,
                    "canUndo":runtime.session().has_undo(),"canRedo":runtime.session().has_redo(),
                    "document":serde_json::to_value(runtime.document()).unwrap_or(serde_json::Value::Null),
                })),
                Ok(None) => send(out, serde_json::json!({"type":"session-undo","tabId":tab,"changed":false,"dirtyNodes":[],
                    "canUndo":runtime.session().has_undo(),"canRedo":runtime.session().has_redo()})),
                Err(error) => session_error(out, &tab, error.to_string()),
            }
        }
        "sessionEvaluate" => {
            let Some(runtime) = sessions.get_mut(&tab) else {
                session_error(out, &tab, "session is not open"); return;
            };
            if let Some(view_targets) = envelope.view_targets.clone() { runtime.set_view_targets(view_targets); }
            if let Some(selected) = envelope.selected.clone() { runtime.set_selected(if selected.is_empty() { None } else { Some(selected) }); }
            runtime.set_export_targets(envelope.export_targets.clone());
            let targets = if !envelope.targets.is_empty() {
                envelope.targets.clone()
            } else {
                let derived = runtime.evaluation_targets();
                if derived.is_empty() { terminal_targets(runtime.document()) } else { derived }
            };
            let mesh_targets = if envelope.mesh_targets.is_empty() { targets.clone() } else { envelope.mesh_targets.clone() };
            let request_id = envelope.request_id;
            let mut progress = |mut event: ProgressEvent| {
                event.tab_id = Some(tab.clone());
                if let Some(id) = request_id { event.request_id = id; }
                if envelope.progress { emit_progress(out, event); }
            };
            match runtime.evaluate_with_options(&targets, &mesh_targets, envelope.force, envelope.progress.then_some(&mut progress as &mut dyn FnMut(ProgressEvent)), envelope.mesh_tolerance) {
                Ok(result) => {
                    let mut value = serde_json::to_value(&result.output.response)
                        .unwrap_or_else(|error| serde_json::json!({"error":error.to_string()}));
                    value["type"] = serde_json::json!("session-result");
                    value["tabId"] = serde_json::json!(tab);
                    value["revision"] = serde_json::json!(runtime.revision());
                    value["dirtyNodes"] = serde_json::json!(result.dirty_nodes);
                    if let Some(id) = request_id { value["requestId"] = serde_json::json!(id); }
                    send(out, value);
                }
                Err(error) => session_error(out, &tab, error.to_string()),
            }
        }
        other => session_error(out, &tab, format!("unknown session action {other:?}")),
    }
}

fn handle_document(message: DocumentRequest, out: &mpsc::Sender<Vec<u8>>) {
    let path = std::path::PathBuf::from(&message.path);
    match message.action.as_str() {
        "documentSave" => {
            let Some(mut value) = message.document else {
                send(out, serde_json::json!({"type":"document-error","tabId":message.tab_id,"path":message.path,"error":"documentSave requires a document"}));
                return;
            };
            let document: GraphDocument = match serde_json::from_value(value.clone()) {
                Ok(document) => document,
                Err(error) => {
                    send(out, serde_json::json!({"type":"document-error","tabId":message.tab_id,"path":message.path,"error":error.to_string()}));
                    return;
                }
            };
            // Keep editor metadata (view, viewedNodes) while writing canonical graph values.
            match nodecraft_session::store::save_atomic_value(&path, &mut value, &document) {
                Ok(()) => {
                    let hash = nodecraft_session::store::document_hash(&document).map(|hash| hash.to_hex()).unwrap_or_default();
                    let file_name = message.file_name.clone().or_else(|| {
                        path.file_name().map(|name| name.to_string_lossy().into_owned())
                    });
                    send(out, serde_json::json!({"type":"document-saved","tabId":message.tab_id,"path":message.path,"fileName":file_name,"hash":hash}));
                }
                Err(error) => send(out, serde_json::json!({"type":"document-error","tabId":message.tab_id,"path":message.path,"error":error})),
            }
        }
        "documentLoad" => {
            match nodecraft_session::store::load(&path) {
                Ok(document) => {
                    let hash = nodecraft_session::store::document_hash(&document).map(|hash| hash.to_hex()).unwrap_or_default();
                    // Preserve editor-only metadata (view, viewedNodes, format) while
                    // returning canonical graph values.
                    let text = std::fs::read_to_string(&path).unwrap_or_default();
                    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|_| serde_json::json!({}));
                    value["nodes"] = serde_json::to_value(&document.nodes).unwrap_or(serde_json::Value::Null);
                    value["connections"] = serde_json::to_value(&document.connections).unwrap_or(serde_json::Value::Null);
                    let file_name = message.file_name.clone().or_else(|| {
                        path.file_name().map(|name| name.to_string_lossy().into_owned())
                    });
                    send(out, serde_json::json!({"type":"document-loaded","tabId":message.tab_id,"path":message.path,"fileName":file_name,"hash":hash,"document":value}));
                }
                Err(error) => send(out, serde_json::json!({"type":"document-error","tabId":message.tab_id,"path":message.path,"error":error})),
            }
        }
        other => send(out, serde_json::json!({"type":"document-error","tabId":message.tab_id,"path":message.path,"error":format!("unknown document action {other:?}")})),
    }
}

fn handle_evaluation(request: Request, cache: &mut EngineCache, out: &mpsc::Sender<Vec<u8>>) {
    let include_diagnostics = request.nodes.iter().any(|node| {
        node.node_type == "stepimporter"
            && node.values.get("diagnostics").and_then(serde_json::Value::as_bool) == Some(true)
    }) || std::env::var("NODECRAFT_WORKER_DIAGNOSTICS")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    let request_cache = if include_diagnostics { EngineCache::new() } else { std::mem::take(cache) };
    let mut progress = |event: ProgressEvent| emit_progress(out, event);
    let output = evaluate(request, request_cache, Some(&mut progress));
    *cache = output.cache;
    let mut value = serde_json::to_value(&output.response)
        .unwrap_or_else(|error| serde_json::json!({"error":error.to_string()}));
    if include_diagnostics && !output.diagnostics.is_empty() {
        let entries: serde_json::Map<String, serde_json::Value> = output
            .diagnostics
            .iter()
            .map(|(id, item)| (id.clone(), item.to_json()))
            .collect();
        value["diagnostics"] = serde_json::Value::Object(entries);
    }
    send(out, value);
}

fn main() {
    if std::env::args().skip(1).any(|argument| argument == "layout" || argument == "--layout") {
        layout::GraphLayout::run();
        return;
    }
    let (out_tx, out_rx) = mpsc::channel::<Vec<u8>>();
    let writer = std::thread::spawn(move || {
        let mut stdout = std::io::stdout().lock();
        for bytes in out_rx {
            let _ = stdout.write_all(&bytes);
            let _ = stdout.flush();
        }
    });
    let engine_out = out_tx.clone();
    let (request_tx, request_rx) = mpsc::channel::<Incoming>();
    let engine = std::thread::spawn(move || {
        let mut cache = EngineCache::new();
        let mut sessions: HashMap<String, GraphRuntime> = HashMap::new();
        while let Ok(message) = request_rx.recv() {
            match message {
                Incoming::Evaluation(request) => handle_evaluation(request, &mut cache, &engine_out),
                Incoming::Session(envelope) => handle_session(envelope, &mut sessions, &engine_out),
                Incoming::Document(document) => handle_document(document, &engine_out),
            }
        }
    });
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() { continue; }
        eprintln!("REQ {}", line.chars().take(500).collect::<String>());
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                let _ = out_tx.send(format!("{}\n", serde_json::json!({"requestId":null,"error":format!("invalid request: {error}")})).into_bytes());
                continue;
            }
        };
        let action = value.get("action").and_then(serde_json::Value::as_str).unwrap_or("");
        let message = if action.starts_with("session") {
            serde_json::from_value::<SessionEnvelope>(value).map(Incoming::Session)
        } else if action == "documentLoad" || action == "documentSave" {
            serde_json::from_value::<DocumentRequest>(value).map(Incoming::Document)
        } else {
            serde_json::from_value::<Request>(value).map(Incoming::Evaluation)
        };
        match message {
            Ok(message) => { if request_tx.send(message).is_err() { break; } }
            Err(error) => {
                let _ = out_tx.send(format!("{}\n", serde_json::json!({"requestId":null,"error":format!("invalid request: {error}")})).into_bytes());
            }
        }
    }
    drop(request_tx);
    let _ = engine.join();
    drop(out_tx);
    let _ = writer.join();
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn document_save_and_load_use_the_rust_store() {
        let (out_tx, out_rx) = mpsc::channel();
        let dir = std::env::temp_dir().join(format!("sn-worker-doc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("part.nodecraft");
        let document: GraphDocument = serde_json::from_value(json!({
            "id":"doc",
            "nodes":[{"id":"a","type":"primitive","values":{},"x":10.0,"y":20.0}],
            "connections":[]
        })).expect("document");
        handle_document(DocumentRequest { action:"documentSave".into(), path:path.to_string_lossy().into(), tab_id:"t".into(), file_name:Some("part.nodecraft".into()), document:Some(serde_json::to_value(document).unwrap()) }, &out_tx);
        let saved = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("save")).expect("json");
        assert_eq!(saved["type"], "document-saved");
        assert!(path.exists());
        handle_document(DocumentRequest { action:"documentLoad".into(), path:path.to_string_lossy().into(), tab_id:"t".into(), file_name:Some("part.nodecraft".into()), document:None }, &out_tx);
        let loaded = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("load")).expect("json");
        assert_eq!(loaded["type"], "document-loaded");
        assert_eq!(loaded["document"]["nodes"][0]["values"]["primitiveType"], "box");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn session_command_then_undo_returns_the_previous_document() {
        let (out_tx, out_rx) = mpsc::channel();
        let mut sessions = HashMap::new();
        let document: GraphDocument = serde_json::from_value(json!({
            "id":"tab-2",
            "nodes":[{"id":"a","type":"primitive","values":{"primitiveType":"box","length":10}}],
            "connections":[]
        })).expect("document");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionOpen".into(), tab_id:"tab-2".into(), document:Some(document), command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let _ = out_rx.recv().expect("open");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionCommand".into(), tab_id:"tab-2".into(), document:None, command:None, commands:vec![Command::SetValue { node:"a".into(), field:"length".into(), value:json!(20) }], targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let committed = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("command")).expect("json");
        assert_eq!(committed["canUndo"], true);
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionUndo".into(), tab_id:"tab-2".into(), document:None, command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let undone = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("undo")).expect("json");
        assert_eq!(undone["type"], "session-undo");
        assert_eq!(undone["document"]["nodes"][0]["values"]["length"], 10);
    }

    #[test]
    fn session_derived_commands_do_not_add_undo_history() {
        let (out_tx, out_rx) = mpsc::channel();
        let mut sessions = HashMap::new();
        let document: GraphDocument = serde_json::from_value(json!({
            "id":"tab-3",
            "nodes":[{"id":"a","type":"primitive","values":{"primitiveType":"box","length":10}}],
            "connections":[]
        })).expect("document");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionOpen".into(), tab_id:"tab-3".into(), document:Some(document), command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let _ = out_rx.recv().expect("open");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionCommand".into(), tab_id:"tab-3".into(), document:None, command:None, commands:vec![Command::SetValue { node:"a".into(), field:"length".into(), value:json!(20) }], targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let user = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("user command")).expect("json");
        assert_eq!(user["canUndo"], true);
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionDerived".into(), tab_id:"tab-3".into(), document:None, command:None, commands:vec![Command::SetValue { node:"a".into(), field:"faces".into(), value:json!([{"id":"face:0"}]) }], targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let derived = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("derived")).expect("json");
        assert_eq!(derived["type"], "session-derived");
        assert_eq!(derived["canUndo"], true);
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionUndo".into(), tab_id:"tab-3".into(), document:None, command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let undone = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("undo")).expect("json");
        assert_eq!(undone["document"]["nodes"][0]["values"]["length"], 10);
        assert_eq!(undone["document"]["nodes"][0]["values"]["faces"][0]["id"], "face:0");
    }

    #[test]
    fn session_evaluate_uses_selected_export_and_view_state() {
        let (out_tx, out_rx) = mpsc::channel();
        let mut sessions = HashMap::new();
        let document: GraphDocument = serde_json::from_value(json!({
            "id":"tab-4",
            "nodes":[
                {"id":"a","type":"primitive","values":{"primitiveType":"box","length":10}},
                {"id":"b","type":"primitive","values":{"primitiveType":"box","length":10}}
            ],
            "connections":[]
        })).expect("document");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionOpen".into(), tab_id:"tab-4".into(), document:Some(document), command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:None, mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:None, export_targets:Vec::new(), selected:None }, &mut sessions, &out_tx);
        let _ = out_rx.recv().expect("open");
        handle_session(SessionEnvelope { mesh_tolerance: None, action:"sessionEvaluate".into(), tab_id:"tab-4".into(), document:None, command:None, commands:Vec::new(), targets:Vec::new(), progress:false, request_id:Some(9), mesh_targets:Vec::new(), force:false, coalesce_key:None, view_targets:Some(vec!["a".into()]), export_targets:vec!["b".into()], selected:Some("a".into()) }, &mut sessions, &out_tx);
        let result = loop {
            let value = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("result")).expect("json");
            if value["type"] == "session-result" { break value; }
        };
        assert!(result["meshes"].get("a").is_some());
        assert!(result["meshes"].get("b").is_some());
    }

    #[test]
    fn session_protocol_opens_commands_and_evaluates_with_cache() {
        let (out_tx, out_rx) = mpsc::channel();
        let mut sessions = HashMap::new();
        let document: GraphDocument = serde_json::from_value(json!({
            "id": "tab-1",
            "nodes": [
                {"id":"a","type":"primitive","values":{"primitiveType":"box","length":10,"width":10,"height":10}},
                {"id":"b","type":"translate","values":{"x":5}}
            ],
            "connections": [{"from":"a","to":"b","in":"shape","out":"shape"}]
        })).expect("document");
        handle_session(SessionEnvelope {
            mesh_tolerance: None,
            action: "sessionOpen".into(), tab_id: "tab-1".into(), document: Some(document),
            command: None, commands: Vec::new(), targets: Vec::new(), progress: true,
            request_id: None, mesh_targets: Vec::new(), force: false, coalesce_key: None, view_targets: None, export_targets: Vec::new(), selected: None,
        }, &mut sessions, &out_tx);
        let opened = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("opened")).expect("json");
        assert_eq!(opened["type"], "session-opened");
        assert!(opened["document"]["nodes"].is_array());
        handle_session(SessionEnvelope {
            mesh_tolerance: None,
            action: "sessionEvaluate".into(), tab_id: "tab-1".into(), document: None,
            command: None, commands: Vec::new(), targets: Vec::new(), progress: false,
            request_id: None, mesh_targets: Vec::new(), force: false, coalesce_key: None, view_targets: Some(vec!["b".into()]), export_targets: Vec::new(), selected: None,
        }, &mut sessions, &out_tx);
        let mut result = None;
        for line in out_rx.try_iter() {
            let value: serde_json::Value = serde_json::from_slice(&line).expect("json");
            if value["type"] == "session-result" { result = Some(value); }
        }
        let result = result.expect("session result");
        // Primitive operands are prewarmed before the target closure is walked,
        // so the target itself is the only cache miss in this small graph.
        assert!(
            result["cacheStats"]["misses"].as_u64().unwrap_or(0) >= 1,
            "cache stats did not record the target evaluation: {result}"
        );
        handle_session(SessionEnvelope {
            mesh_tolerance: None,
            action: "sessionCommand".into(), tab_id: "tab-1".into(), document: None,
            command: Some(Command::SetPosition { node: "a".into(), x: 300.0, y: 0.0 }), commands: Vec::new(),
            targets: Vec::new(), progress: false,
            request_id: None, mesh_targets: Vec::new(), force: false, coalesce_key: None, view_targets: None, export_targets: Vec::new(), selected: None,
        }, &mut sessions, &out_tx);
        let change = serde_json::from_slice::<serde_json::Value>(&out_rx.recv().expect("change")).expect("json");
        assert_eq!(change["type"], "session-command");
        assert_eq!(change["dirtyNodes"].as_array().expect("dirty").len(), 0);
    }
}
