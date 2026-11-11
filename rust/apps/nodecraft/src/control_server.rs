//! Small localhost control server for the NodeCraft MCP bridge.
//!
//! It intentionally accepts only loopback HTTP and mirrors the endpoints the old
//! WebView app exposed: /project, /open-step, /fit-graph, /zoom-out,
//! /ensure-visible and /state.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

pub enum Command {
    Project(serde_json::Value),
    ReplaceProject(serde_json::Value),
    OpenStep(String),
    FitGraph,
    ZoomOut,
    EnsureVisible,
    SelectNode(String),
    EditNode(String),
    SetValue { node: String, field: String, value: serde_json::Value },
    BeginSelection { node: String, field: String },
    ViewNode(String),
    Home,
    Orbit { dx: f64, dy: f64 },
    Zoom(f64),
    Triangles(bool),
    Translucent(bool),
    Capture(String),
    ActivateTab(usize),
    State,
}

pub struct Request {
    pub command: Command,
    pub reply: Sender<String>,
}

pub fn start(ctx: egui::Context) -> Receiver<Request> {
    let (tx, rx) = channel::<Request>();
    std::thread::Builder::new()
        .name("nodecraft-mcp-control".into())
        .spawn(move || {
            let listener = match TcpListener::bind(("127.0.0.1", 47831)) {
                Ok(listener) => listener,
                Err(error) => {
                    eprintln!("MCP control server: cannot bind 127.0.0.1:47831: {error}");
                    return;
                }
            };
            eprintln!("MCP control server listening on 127.0.0.1:47831");
            for stream in listener.incoming().flatten() {
                let tx = tx.clone();
                let ctx = ctx.clone();
                std::thread::spawn(move || handle(stream, tx, ctx));
            }
        })
        .ok();
    rx
}

fn handle(mut stream: TcpStream, tx: Sender<Request>, ctx: egui::Context) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let header_end = loop {
        match stream.read(&mut chunk) {
            Ok(0) => break None,
            Ok(count) => {
                buffer.extend_from_slice(&chunk[..count]);
                if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                    break Some(index + 4);
                }
            }
            Err(_) => break None,
        }
    };
    let Some(header_end) = header_end else {
        let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"bad request\"}");
        return;
    };
    let header = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let content_length = header
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length:").map(str::trim))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    while buffer.len() < header_end + content_length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(count) => buffer.extend_from_slice(&chunk[..count]),
        }
    }
    let body = String::from_utf8_lossy(&buffer[header_end..]).to_string();
    let Some(path) = header
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
    else {
        let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"bad request\"}");
        return;
    };
    let command = match path {
        "/replace-project" => match serde_json::from_str(&body) {
            Ok(value) => Command::ReplaceProject(value),
            Err(error) => {
                let _ = respond(&mut stream, 400, &format!("{{\"ok\":false,\"error\":{}}}", serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"invalid project\"".into())));
                return;
            }
        },
        "/project" => match serde_json::from_str(&body) {
            Ok(value) => Command::Project(value),
            Err(error) => {
                let _ = respond(&mut stream, 400, &format!("{{\"ok\":false,\"error\":{}}}", serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"invalid project\"".into())));
                return;
            }
        },
        "/open-step" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("path").and_then(|path| path.as_str()).map(str::to_owned))
        {
            Some(path) => Command::OpenStep(path),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"path\\\": ...}\"}");
                return;
            }
        },
        "/select" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::SelectNode(id),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}");
                return;
            }
        },
        "/edit" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::EditNode(id),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}");
                return;
            }
        },
        "/set" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| {
            Some(Command::SetValue {
                node: value.get("node")?.as_str()?.to_owned(),
                field: value.get("field")?.as_str()?.to_owned(),
                value: value.get("value")?.clone(),
            })
        }) {
            Some(command) => command,
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {node, field, value}\"}");
                return;
            }
        },
        "/view" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::ViewNode(id),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}");
                return;
            }
        },
        "/home" => Command::Home,
        "/orbit" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| {
            Some((value.get("dx")?.as_f64()?, value.get("dy")?.as_f64()?))
        }) {
            Some((dx, dy)) => Command::Orbit { dx, dy },
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"dx\\\": ..., \\\"dy\\\": ...}\"}");
                return;
            }
        },
        "/zoom" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| value.get("factor").and_then(serde_json::Value::as_f64)) {
            Some(factor) => Command::Zoom(factor),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\"factor\": 1.0}\"}");
                return;
            }
        },
        "/triangles" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| value.get("enabled").and_then(serde_json::Value::as_bool)) {
            Some(enabled) => Command::Triangles(enabled),
            None => { let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"enabled\\\": true|false}\"}"); return; }
        },
        "/translucent" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| value.get("enabled").and_then(serde_json::Value::as_bool)) {
            Some(enabled) => Command::Translucent(enabled),
            None => { let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"enabled\\\": true|false}\"}"); return; }
        },
        "/capture" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("path").and_then(|path| path.as_str()).map(str::to_owned))
        {
            Some(path) => Command::Capture(path),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"path\\\": ...}\"}");
                return;
            }
        },
        "/begin-selection" => match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| {
            let node = value.get("node")?.as_str()?.to_owned();
            let field = value.get("field")?.as_str()?.to_owned();
            Some((node, field))
        }) {
            Some((node, field)) => Command::BeginSelection { node, field },
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"node\\\": ..., \\\"field\\\": ...}\"}");
                return;
            }
        },
        "/fit-graph" => Command::FitGraph,
        "/zoom-out" => Command::ZoomOut,
        "/ensure-visible" => Command::EnsureVisible,
        "/activate-tab" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("index").and_then(serde_json::Value::as_u64))
        {
            Some(index) => Command::ActivateTab(index as usize),
            None => {
                let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"expected {\\\"index\\\": ...}\"}");
                return;
            }
        },
        "/state" => Command::State,
        _ => {
            let _ = respond(&mut stream, 404, "{\"ok\":false,\"error\":\"unknown endpoint\"}");
            return;
        }
    };
    let (reply_tx, reply_rx) = channel::<String>();
    if tx.send(Request { command, reply: reply_tx }).is_err() {
        let _ = respond(&mut stream, 503, "{\"ok\":false,\"error\":\"app unavailable\"}");
        return;
    }
    ctx.request_repaint();
    match reply_rx.recv_timeout(Duration::from_secs(8)) {
        Ok(text) => {
            let _ = respond(&mut stream, 200, &text);
        }
        Err(_) => {
            let _ = respond(&mut stream, 504, "{\"ok\":false,\"error\":\"app did not answer\"}");
        }
    }
}

fn respond(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}
