//! Small localhost control server for the NodeCraft MCP bridge.
//!
//! It intentionally accepts only loopback HTTP and mirrors the endpoints the old
//! WebView app exposed: /project, /open-step, /fit-graph, /zoom-out,
//! /ensure-visible and /state.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
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
    SetValue {
        node: String,
        field: String,
        value: serde_json::Value,
    },
    BeginSelection {
        node: String,
        field: String,
    },
    ViewNode(String),
    /// Flip the node's suppression flag through the same edit + evaluate path
    /// as the node editor's toggle.
    ToggleSuppressed(String),
    Home,
    Orbit {
        dx: f64,
        dy: f64,
    },
    Zoom(f64),
    Triangles(bool),
    Translucent(bool),
    Capture(String),
    ActivateTab(usize),
    State,
}

impl Command {
    /// Read-only commands do not need another frame after they reply.
    pub fn changes_state(&self) -> bool {
        !matches!(self, Command::State)
    }
}

/// A command handed to the app together with the one reply slot. The HTTP
/// handler and the app race to take the slot: once either side has it, the
/// other cannot run/answer. A handler that times out withdraws the slot, so a
/// late frame must not execute a command whose client already gave up.
pub struct Request {
    pub command: Command,
    reply: Arc<Mutex<Option<Sender<String>>>>,
}

impl Request {
    /// Take the reply slot. `None` means the HTTP handler timed out first.
    pub fn claim(&self) -> Option<Sender<String>> {
        self.reply.lock().ok().and_then(|mut reply| reply.take())
    }

    /// Withdraw an unanswered request. Returns true if it was still pending.
    pub fn withdraw(&self) -> bool {
        self.reply.lock().map(|mut reply| reply.take().is_some()).unwrap_or(false)
    }
}

/// The loopback port the control server binds. Automation passes
/// `NODECRAFT_CONTROL_PORT` so a probe app never competes with an already
/// running NodeCraft for the default port.
fn control_port_from(value: Option<String>) -> u16 {
    value.and_then(|value| value.parse().ok()).unwrap_or(DEFAULT_CONTROL_PORT)
}

fn control_port() -> u16 {
    control_port_from(std::env::var("NODECRAFT_CONTROL_PORT").ok())
}

const DEFAULT_CONTROL_PORT: u16 = 47831;

pub fn start(ctx: egui::Context) -> Receiver<Request> {
    let (tx, rx) = channel::<Request>();
    std::thread::Builder::new()
        .name("nodecraft-mcp-control".into())
        .spawn(move || {
            let port = control_port();
            let listener = match TcpListener::bind(("127.0.0.1", port)) {
                Ok(listener) => listener,
                Err(error) => {
                    eprintln!("MCP control server: cannot bind 127.0.0.1:{port}: {error}");
                    return;
                }
            };
            eprintln!("MCP control server listening on 127.0.0.1:{port}");
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
    let Some(path) = header.lines().next().and_then(|line| line.split_whitespace().nth(1)) else {
        let _ = respond(&mut stream, 400, "{\"ok\":false,\"error\":\"bad request\"}");
        return;
    };
    let command = match path {
        "/replace-project" => match serde_json::from_str(&body) {
            Ok(value) => Command::ReplaceProject(value),
            Err(error) => {
                let _ = respond(
                    &mut stream,
                    400,
                    &format!(
                        "{{\"ok\":false,\"error\":{}}}",
                        serde_json::to_string(&error.to_string())
                            .unwrap_or_else(|_| "\"invalid project\"".into())
                    ),
                );
                return;
            }
        },
        "/project" => match serde_json::from_str(&body) {
            Ok(value) => Command::Project(value),
            Err(error) => {
                let _ = respond(
                    &mut stream,
                    400,
                    &format!(
                        "{{\"ok\":false,\"error\":{}}}",
                        serde_json::to_string(&error.to_string())
                            .unwrap_or_else(|_| "\"invalid project\"".into())
                    ),
                );
                return;
            }
        },
        "/open-step" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("path").and_then(|path| path.as_str()).map(str::to_owned))
        {
            Some(path) => Command::OpenStep(path),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"path\\\": ...}\"}",
                );
                return;
            }
        },
        "/select" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::SelectNode(id),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}",
                );
                return;
            }
        },
        "/edit" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::EditNode(id),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}",
                );
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
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {node, field, value}\"}",
                );
                return;
            }
        },
        "/view" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::ViewNode(id),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}",
                );
                return;
            }
        },
        "/toggle-suppress" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("id").and_then(|id| id.as_str()).map(str::to_owned))
        {
            Some(id) => Command::ToggleSuppressed(id),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"id\\\": ...}\"}",
                );
                return;
            }
        },
        "/home" => Command::Home,
        "/orbit" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| Some((value.get("dx")?.as_f64()?, value.get("dy")?.as_f64()?)))
        {
            Some((dx, dy)) => Command::Orbit { dx, dy },
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"dx\\\": ..., \\\"dy\\\": ...}\"}",
                );
                return;
            }
        },
        "/zoom" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("factor").and_then(serde_json::Value::as_f64))
        {
            Some(factor) => Command::Zoom(factor),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\"factor\": 1.0}\"}",
                );
                return;
            }
        },
        "/triangles" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("enabled").and_then(serde_json::Value::as_bool))
        {
            Some(enabled) => Command::Triangles(enabled),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"enabled\\\": true|false}\"}",
                );
                return;
            }
        },
        "/translucent" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("enabled").and_then(serde_json::Value::as_bool))
        {
            Some(enabled) => Command::Translucent(enabled),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"enabled\\\": true|false}\"}",
                );
                return;
            }
        },
        "/capture" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("path").and_then(|path| path.as_str()).map(str::to_owned))
        {
            Some(path) => Command::Capture(path),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"path\\\": ...}\"}",
                );
                return;
            }
        },
        "/begin-selection" => {
            match serde_json::from_str::<serde_json::Value>(&body).ok().and_then(|value| {
                let node = value.get("node")?.as_str()?.to_owned();
                let field = value.get("field")?.as_str()?.to_owned();
                Some((node, field))
            }) {
                Some((node, field)) => Command::BeginSelection { node, field },
                None => {
                    let _ = respond(
                        &mut stream,
                        400,
                        "{\"ok\":false,\"error\":\"expected {\\\"node\\\": ..., \\\"field\\\": ...}\"}",
                    );
                    return;
                }
            }
        }
        "/fit-graph" => Command::FitGraph,
        "/zoom-out" => Command::ZoomOut,
        "/ensure-visible" => Command::EnsureVisible,
        "/activate-tab" => match serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| value.get("index").and_then(serde_json::Value::as_u64))
        {
            Some(index) => Command::ActivateTab(index as usize),
            None => {
                let _ = respond(
                    &mut stream,
                    400,
                    "{\"ok\":false,\"error\":\"expected {\\\"index\\\": ...}\"}",
                );
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
    let reply = Arc::new(Mutex::new(Some(reply_tx)));
    let request = Request { command, reply: Arc::clone(&reply) };
    if tx.send(request).is_err() {
        let _ = respond(&mut stream, 503, "{\"ok\":false,\"error\":\"app unavailable\"}");
        return;
    }
    // Wake the app even when the window is hidden or occluded; eframe calls
    // `logic` for that case.
    ctx.request_repaint();
    match reply_rx.recv_timeout(Duration::from_secs(8)) {
        Ok(text) => {
            let _ = respond(&mut stream, 200, &text);
        }
        Err(_) => {
            // Withdraw before replying: if the app has not claimed this command
            // yet, it must never execute after the client has gone.
            let _ = reply.lock().map(|mut reply| reply.take());
            let _ = respond(&mut stream, 504, "{\"ok\":false,\"error\":\"app did not answer\"}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending() -> (Request, Receiver<String>) {
        let (tx, rx) = channel::<String>();
        (Request { command: Command::State, reply: Arc::new(Mutex::new(Some(tx))) }, rx)
    }

    #[test]
    fn timed_out_command_is_withdrawn_before_the_app_drains_it() {
        let (request, reply_rx) = pending();
        // The HTTP handler hit its timeout first.
        assert!(request.withdraw());
        // A later frame must not run it or answer it.
        assert!(request.claim().is_none());
        assert!(reply_rx.try_recv().is_err());
    }

    #[test]
    fn a_command_is_claimed_and_answered_at_most_once() {
        let (request, reply_rx) = pending();
        let reply = request.claim().expect("the app claims the pending command");
        assert!(request.claim().is_none(), "a second drain must not run it again");
        assert!(!request.withdraw(), "a claimed command cannot be withdrawn");
        reply.send("ok".to_owned()).expect("reply");
        assert_eq!(reply_rx.recv_timeout(Duration::from_secs(1)).unwrap(), "ok");
    }

    #[test]
    fn the_control_port_defaults_and_can_move_for_automation() {
        assert_eq!(control_port_from(None), DEFAULT_CONTROL_PORT);
        assert_eq!(control_port_from(Some("47832".to_owned())), 47832);
        assert_eq!(control_port_from(Some("not a port".to_owned())), DEFAULT_CONTROL_PORT);
    }

    #[test]
    fn only_read_only_commands_skip_the_repaint_request() {
        assert!(!Command::State.changes_state());
        assert!(Command::SelectNode("u10".to_owned()).changes_state());
        assert!(Command::FitGraph.changes_state());
        assert!(Command::Capture("shot.png".to_owned()).changes_state());
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
