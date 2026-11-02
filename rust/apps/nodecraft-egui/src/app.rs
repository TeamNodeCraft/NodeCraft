//! Application shell and the first Swift/WebView-to-egui port slice.

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use nodecraft_engine::EngineCache;
use nodecraft_protocol::Mesh;
use serde::{Deserialize, Serialize};
use nodecraft_session::store::DocumentStore;
use nodecraft_session::{Command, GraphConnection, GraphDocument, GraphNode, Session};

const BUILD_GIT_HASH: &str = match option_env!("NODECRAFT_GIT_HASH") {
    Some(hash) => hash,
    None => "unknown",
};

use crate::engine_bridge;
use crate::inspector;
use nodecraft_node_canvas::canvas::{
    node_width, ConnectionKey, LayoutMode, NodeEditor, NodeEditorAction, NODE_HEIGHT,
    NODE_LEFT_PADDING, NODE_RIGHT_PADDING,
};
use nodecraft_node_canvas::theme::CanvasTheme;
use crate::ui_shape;
use crate::control_server;
use crate::sketch_gui::SketchGui;
use crate::viewport::Viewport;
use crate::viewer_gizmo::Gizmo;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NodeClipboard {
    nodes: Vec<GraphNode>,
    #[serde(default)]
    connections: Vec<GraphConnection>,
}

const PASTE_OFFSET: f64 = 32.0;
fn tint_svg(source: &[u8], colour: &str) -> String {
    let mut svg = std::str::from_utf8(source)
        .unwrap_or_default()
        .replace("#212121", colour)
        .replace("#b4b4b4", colour)
        .replace("#d8d8dc", colour);
    if let Some(root_end) = svg.find('>') {
        let root = &svg[..root_end];
        if !root.contains("fill=") && !root.contains("stroke=") {
            svg.insert_str(root_end, &format!(" fill=\"{colour}\""));
        }
    }
    svg
}

const TOOL_DOCK_WIDTH: f32 = 44.0;
const TOOL_DOCK_GAP: f32 = 8.0;

fn tool_dock_rect(workspace: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::Pos2::new(workspace.min.x + 12.0, workspace.min.y + 12.0),
        egui::Vec2::new(TOOL_DOCK_WIDTH, (workspace.height() - 24.0).max(80.0)),
    )
}

#[derive(Clone, Copy)]
struct ToolItem {
    label: &'static str,
    node_type: &'static str,
    primitive: Option<&'static str>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolSection {
    Solids,
    Transform,
    Modify,
    Io,
}

impl ToolSection {
    const ALL: [Self; 4] = [Self::Solids, Self::Transform, Self::Modify, Self::Io];

    fn label(self) -> &'static str {
        match self {
            Self::Solids => "Solids",
            Self::Transform => "Transform",
            Self::Modify => "Modify",
            Self::Io => "Import / Export",
        }
    }

    fn icon(self) -> (&'static str, &'static [u8]) {
        match self {
            Self::Solids => ("cube", include_bytes!("../../../../vendor/fluent/cube.svg")),
            Self::Transform => ("move", include_bytes!("../../../../vendor/fluent/arrow_move.svg")),
            Self::Modify => ("merge", include_bytes!("../../../../vendor/fluent/merge.svg")),
            Self::Io => ("io", include_bytes!("../../../../vendor/fluent/import_step.svg")),
        }
    }

    fn tools(self) -> &'static [ToolItem] {
        const SOLIDS: &[ToolItem] = &[
            ToolItem { label: "Box", node_type: "primitive", primitive: Some("box") },
            ToolItem { label: "Sphere", node_type: "primitive", primitive: Some("sphere") },
            ToolItem { label: "Cylinder", node_type: "primitive", primitive: Some("cylinder") },
            ToolItem { label: "Cone", node_type: "primitive", primitive: Some("cone") },
            ToolItem { label: "Torus", node_type: "primitive", primitive: Some("torus") },
            ToolItem { label: "Pyramid", node_type: "primitive", primitive: Some("pyramid") },
            ToolItem { label: "Wedge", node_type: "primitive", primitive: Some("wedge") },
            ToolItem { label: "Partial Sphere", node_type: "primitive", primitive: Some("partialSphere") },
            ToolItem { label: "Partial Torus", node_type: "primitive", primitive: Some("partialTorus") },
            ToolItem { label: "Sketch", node_type: "sketch", primitive: None },
            ToolItem { label: "Spline", node_type: "spline", primitive: None },
        ];
        const TRANSFORM: &[ToolItem] = &[
            ToolItem { label: "Transform", node_type: "translate", primitive: None },
            ToolItem { label: "Mirror", node_type: "mirror", primitive: None },
        ];
        const MODIFY: &[ToolItem] = &[
            ToolItem { label: "Boolean", node_type: "boolean", primitive: None },
            ToolItem { label: "Combine to Part", node_type: "combine", primitive: None },
            ToolItem { label: "Chamfer", node_type: "chamfer", primitive: None },
            ToolItem { label: "Shell", node_type: "shell", primitive: None },
            ToolItem { label: "Extrude", node_type: "extrude", primitive: None },
            ToolItem { label: "Loft", node_type: "loft", primitive: None },
            ToolItem { label: "Surface Loft", node_type: "surfaceloft", primitive: None },
            ToolItem { label: "Scatter", node_type: "scatter", primitive: None },
        ];
        const IO: &[ToolItem] = &[
            ToolItem { label: "Import STEP", node_type: "stepimporter", primitive: None },
            ToolItem { label: "Linked Part", node_type: "linkedpart", primitive: None },
            ToolItem { label: "Export STEP", node_type: "exporter", primitive: None },
        ];
        match self {
            Self::Solids => SOLIDS,
            Self::Transform => TRANSFORM,
            Self::Modify => MODIFY,
            Self::Io => IO,
        }
    }
}



#[derive(Clone, Debug, PartialEq, Eq)]
enum ClipboardAction {
    Copy,
    Cut,
    Paste(String),
}

fn clipboard_action(events: &[egui::Event]) -> Option<ClipboardAction> {
    events.iter().fold(None, |action, event| match event {
        egui::Event::Copy => Some(ClipboardAction::Copy),
        egui::Event::Cut => Some(ClipboardAction::Cut),
        egui::Event::Paste(text) => Some(ClipboardAction::Paste(text.clone())),
        _ => action,
    })
}

fn delete_selection_commands(
    nodes: &HashSet<String>,
    cables: &HashSet<ConnectionKey>,
) -> Vec<Command> {
    let mut commands: Vec<Command> = nodes
        .iter()
        .cloned()
        .map(|id| Command::RemoveNode { id })
        .collect();
    for key in cables {
        if nodes.contains(&key.from) || nodes.contains(&key.to) {
            continue;
        }
        commands.push(Command::Disconnect {
            from: key.from.clone(),
            to: key.to.clone(),
            input: key.input.clone(),
        });
    }
    commands
}

fn parse_node_clipboard(text: &str) -> Result<NodeClipboard, String> {
    let value: serde_json::Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
    if value.is_array() {
        return Ok(NodeClipboard {
            nodes: serde_json::from_value(value).map_err(|error| error.to_string())?,
            connections: Vec::new(),
        });
    }
    if value.get("nodes").is_some() {
        return serde_json::from_value(value).map_err(|error| error.to_string());
    }
    if value.get("id").is_some() && value.get("type").is_some() {
        return Ok(NodeClipboard {
            nodes: vec![serde_json::from_value(value).map_err(|error| error.to_string())?],
            connections: Vec::new(),
        });
    }
    Err("expected node object, node array, or { nodes: [...] }".to_owned())
}

fn clipboard_payload(document: &GraphDocument, selected: &HashSet<&str>) -> Option<NodeClipboard> {
    let nodes: Vec<GraphNode> = document
        .nodes
        .iter()
        .filter(|node| selected.contains(node.id.as_str()))
        .cloned()
        .collect();
    if nodes.is_empty() {
        return None;
    }
    // Copy only connections whose endpoints are both part of the selection.
    // External upstream/downstream links are not owned by the copied subgraph
    // and must not be recreated on paste.
    let connections: Vec<GraphConnection> = document
        .connections
        .iter()
        .filter(|link| {
            selected.contains(link.from.as_str()) && selected.contains(link.to.as_str())
        })
        .cloned()
        .collect();
    Some(NodeClipboard { nodes, connections })
}

fn paste_plan(
    document: &GraphDocument,
    payload: NodeClipboard,
    origin: Option<egui::Pos2>,
) -> (Vec<Command>, HashSet<String>, Option<String>) {
    let min_x = payload.nodes.iter().map(|node| node.x).fold(f64::INFINITY, f64::min);
    let min_y = payload.nodes.iter().map(|node| node.y).fold(f64::INFINITY, f64::min);
    let (origin_x, origin_y) = origin
        .map(|position| (f64::from(position.x), f64::from(position.y)))
        .unwrap_or((min_x + PASTE_OFFSET, min_y + PASTE_OFFSET));
    let mut used: HashSet<String> = document.nodes.iter().map(|node| node.id.clone()).collect();
    let mut ids = std::collections::HashMap::new();
    for node in &payload.nodes {
        ids.insert(node.id.clone(), unique_node_id(&node.id, &mut used));
    }
    let mut commands = Vec::new();
    for node in &payload.nodes {
        let id = ids.get(&node.id).cloned().unwrap_or_else(|| node.id.clone());
        commands.push(Command::AddNode {
            node: GraphNode {
                id,
                x: origin_x + (node.x - min_x),
                y: origin_y + (node.y - min_y),
                ..node.clone()
            },
        });
    }
    for link in payload.connections {
        let to = ids.get(&link.to).cloned().unwrap_or_else(|| link.to.clone());
        let from = ids
            .get(&link.from)
            .cloned()
            .unwrap_or_else(|| link.from.clone());
        let from_exists = ids.contains_key(&link.from)
            || document.nodes.iter().any(|node| node.id == from);
        let to_exists =
            ids.contains_key(&link.to) || document.nodes.iter().any(|node| node.id == to);
        if !from_exists || !to_exists {
            continue;
        }
        commands.push(Command::Connect {
            from,
            to,
            input: link.input,
            out: link.out,
        });
    }
    let selected: HashSet<String> = ids.values().cloned().collect();
    let pasted_last = payload.nodes.last().and_then(|node| ids.get(&node.id)).cloned();
    (commands, selected, pasted_last)
}

fn document_id_for_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn is_nodecraft_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("nodecraft"))
}

fn is_step_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("step") || extension.eq_ignore_ascii_case("stp"))
}

fn unique_node_id(base: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        used.insert(base.to_owned());
        return base.to_owned();
    }
    let mut copy = 1;
    loop {
        let candidate = format!("{base}-copy{copy}");
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        copy += 1;
    }
}

const LAYOUT_ANIMATION: Duration = Duration::from_millis(650);

struct InspectorAnimation {
    started: Instant,
    from: f32,
    to: f32,
}

struct LayoutAnimation {
    start: HashMap<String, (f64, f64)>,
    end: HashMap<String, (f64, f64)>,
    started: Instant,
}

struct NodePositionAnimation {
    start: HashMap<String, (f64, f64)>,
    end: HashMap<String, (f64, f64)>,
    started: Instant,
}

impl NodePositionAnimation {
    fn document_at(&self, base: &GraphDocument, progress: f32) -> GraphDocument {
        let eased = progress * progress * (3.0 - 2.0 * progress);
        let mut document = base.clone();
        for node in &mut document.nodes {
            let (Some((x0, y0)), Some((x1, y1))) =
                (self.start.get(&node.id), self.end.get(&node.id))
            else {
                continue;
            };
            node.x = x0 + (x1 - x0) * f64::from(eased);
            node.y = y0 + (y1 - y0) * f64::from(eased);
        }
        document
    }
}

fn node_positions(document: &GraphDocument) -> HashMap<String, (f64, f64)> {
    document
        .nodes
        .iter()
        .map(|node| (node.id.clone(), (node.x, node.y)))
        .collect()
}

fn moved_positions(
    before: &HashMap<String, (f64, f64)>,
    after: &HashMap<String, (f64, f64)>,
) -> (HashMap<String, (f64, f64)>, HashMap<String, (f64, f64)>) {
    let mut start = HashMap::new();
    let mut end = HashMap::new();
    for (id, position) in after {
        if let Some(previous) = before.get(id) {
            if previous != position {
                start.insert(id.clone(), *previous);
                end.insert(id.clone(), *position);
            }
        }
    }
    (start, end)
}

fn downstream_nodes(document: &GraphDocument, start: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    let mut queue = std::collections::VecDeque::from([start.to_owned()]);
    while let Some(node) = queue.pop_front() {
        if !result.insert(node.clone()) {
            continue;
        }
        for connection in document.connections.iter().filter(|link| link.from == node) {
            queue.push_back(connection.to.clone());
        }
    }
    result
}


#[derive(Clone)]
struct DocumentTab {
    id: u64,
    document_id: String,
    path: Option<PathBuf>,
    selected_nodes: HashSet<String>,
    selected_connections: HashSet<ConnectionKey>,
    viewer_breakpoint: Option<String>,
    mesh: Option<Mesh>,
    status: String,
    error: Option<String>,
    evaluation_ms: f64,
    cache_hits: u32,
    cache_misses: u32,
    camera: Option<solidcraft_render::Camera>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Appearance,
    Viewport,
    Diagnostics,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            Self::Appearance => "Appearance",
            Self::Viewport => "Viewport",
            Self::Diagnostics => "Diagnostics",
        }
    }
}

fn target_closure(document: &GraphDocument, target: &str) -> HashSet<String> {
    let mut closure = HashSet::new();
    let mut stack = vec![target.to_owned()];
    while let Some(id) = stack.pop() {
        if !closure.insert(id.clone()) { continue; }
        for link in document.connections.iter().filter(|link| link.to == id) {
            stack.push(link.from.clone());
        }
    }
    closure
}

/// The node the viewer should show when a document is opened. Prefer an
/// interface output, then the last terminal sink (a node with no consumers).
/// Falling back to `nodes.last()` is wrong when the final node is an upstream
/// primitive; it displays that primitive at its local origin instead of the
/// evaluated root.
fn initial_viewer_target(document: &GraphDocument) -> Option<String> {
    if let Some(output) = document.interface.outputs.last()
        && document.nodes.iter().any(|node| &node.id == output)
    {
        return Some(output.clone());
    }
    document
        .nodes
        .iter()
        .rev()
        .find(|node| {
            node.node_type != "exporter"
                && !document.connections.iter().any(|link| link.from == node.id)
        })
        .or_else(|| document.nodes.last())
        .map(|node| node.id.clone())
}

pub struct NodeCraftApp {
    session: Option<Session>,
    path: Option<PathBuf>,
    documents: DocumentStore,
    next_document_id: u64,
    selected_nodes: HashSet<String>,
    selected_connections: HashSet<ConnectionKey>,
    viewer_breakpoint: Option<String>,
    eval_worker: EvalWorker,
    eval_next: u64,
    eval_pending: Option<u64>,
    egui_ctx: egui::Context,
    mesh: Option<Mesh>,
    viewport: Viewport,
    node_editor: NodeEditor,
    status: String,
    error: Option<String>,
    evaluation_ms: f64,
    cache_hits: u32,
    cache_misses: u32,
    warmup_until: Instant,
    screenshot: Option<(std::path::PathBuf, u32)>,
    render_model: bool,
    preserve_view_for_next_mesh: bool,
    camera_initialized: bool,
    last_saved_view: crate::viewport::ViewSettings,
    last_saved_theme: Option<String>,
    activity: ActivitySettings,
    recent_files: Vec<PathBuf>,
    last_saved_recent: Vec<PathBuf>,
    view_save_due: Option<Instant>,
    auto_eval_due: Option<Instant>,
    pending_dirty: HashSet<String>,
    layout_animation: Option<LayoutAnimation>,
    node_position_animation: Option<NodePositionAnimation>,
    tabs: Vec<DocumentTab>,
    active_tab: usize,
    next_tab_id: u64,
    inspector_width: f32,
    inspector_open: bool,
    inspector_reveal: f32,
    inspector_animation: Option<InspectorAnimation>,
    inspector_reflow: bool,
    settings_open: bool,
    settings_open_last: bool,
    settings_tab: SettingsTab,
    theme_dirty: bool,
    theme_save_due: Option<Instant>,
    theme_rename_id: Option<String>,
    theme_rename_name: String,
    theme_rename_focus: bool,
    tool_section: Option<ToolSection>,
    sketch_gui: SketchGui,
    control_requests: Receiver<control_server::Request>,
    last_pointer: Option<egui::Pos2>,
}

#[cfg(target_os = "macos")]
fn hide_settings_zoom_button() {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSWindowButton};

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    for window in app.windows().iter() {
        if window.title().to_string() == "NodeCraft Settings" {
            if let Some(button) = window.standardWindowButton(NSWindowButton::ZoomButton) {
                button.setHidden(true);
            }
            if let Some(button) = window.standardWindowButton(NSWindowButton::MiniaturizeButton) {
                button.setHidden(true);
            }
        }
    }
}

impl NodeCraftApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial: Option<PathBuf>,
        screenshot: Option<PathBuf>,
        render_model: bool,
    ) -> Self {
        // Installed before the first frame so node-type SVG icons can be loaded.
        egui_extras::install_image_loaders(&cc.egui_ctx);
        // Keep egui and the native macOS titlebar in dark mode even on a light system.
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::SetTheme(egui::SystemTheme::Dark));
        let settings = load_app_settings().unwrap_or_default();
        if std::env::var_os("NODECRAFT_THEME").is_none() {
            apply_saved_theme(settings.theme.as_deref());
        }
        configure_style(&cc.egui_ctx);
        let mut viewport = Viewport::default();
        let mut node_editor = NodeEditor::default();
        let initial_theme = crate::theme::get();
        node_editor.set_theme(canvas_theme(&initial_theme));
        viewport.set_theme(&initial_theme);
        node_editor.set_left_dock_width(TOOL_DOCK_WIDTH + TOOL_DOCK_GAP);
        if let Some(render_state) = &cc.wgpu_render_state {
            viewport.install_gpu(render_state);
            #[cfg(feature = "gpu-viewport")]
            node_editor.install_gpu(
                render_state,
                crate::gpu_viewport::msaa_samples(),
                solidcraft_gpu::depth_format(&render_state.device, true),
            );
        }
        if render_model {
            viewport.set_chrome_visible(false);
        }
        let persisted_view = settings.view;
        let activity = settings.activity;
        node_editor.set_activity_timings(
            u64::from(activity.fade_in_ms),
            u64::from(activity.hold_ms),
            u64::from(activity.fade_out_ms),
        );
        let initial_theme_id = Some(crate::theme::active_preset().id);
        let mut recent_files: Vec<PathBuf> = settings
            .recent_files
            .iter()
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .collect();
        recent_files.truncate(10);
        let startup_recent = recent_files.first().cloned();
        viewport.set_view(persisted_view);
        let persisted_recent = recent_files.clone();
        let mut app = Self {
            session: None,
            path: None,
            documents: DocumentStore::new(),
            next_document_id: 1,
            selected_nodes: HashSet::new(),
            selected_connections: HashSet::new(),
            viewer_breakpoint: None,
            eval_worker: EvalWorker::new(),
            eval_next: 0,
            eval_pending: None,
            egui_ctx: cc.egui_ctx.clone(),
            mesh: None,
            viewport,
            node_editor,
            status: "Ready".to_owned(),
            error: None,
            evaluation_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            // The SolidCraft callback needs a prepare frame before its first blit.
            warmup_until: Instant::now() + std::time::Duration::from_secs(2),
            screenshot: screenshot.map(|path| (path, 0)),
            render_model,
            preserve_view_for_next_mesh: false,
            camera_initialized: false,
            last_saved_view: persisted_view,
            last_saved_theme: initial_theme_id,
            activity,
            recent_files,
            last_saved_recent: persisted_recent,
            view_save_due: None,
            auto_eval_due: None,
            pending_dirty: HashSet::new(),
            layout_animation: None,
            node_position_animation: None,
            tabs: Vec::new(),
            active_tab: 0,
            next_tab_id: 1,
            inspector_width: 280.0,
            inspector_open: true,
            inspector_reveal: 1.0,
            inspector_animation: None,
            inspector_reflow: false,
            settings_open: false,
            settings_open_last: false,
            settings_tab: SettingsTab::Appearance,
            theme_dirty: false,
            theme_save_due: None,
            theme_rename_id: None,
            theme_rename_name: String::new(),
            theme_rename_focus: false,
            tool_section: None,
            sketch_gui: SketchGui::default(),
            control_requests: control_server::start(cc.egui_ctx.clone()),
            last_pointer: None,
        };
        if let Some(path) = choose_startup_document(initial, startup_recent, default_document()) {
            app.open(path);
        } else {
            app.new_blank();
        }
        app
    }

    fn open(&mut self, path: PathBuf) {
        self.layout_animation = None;
        self.node_position_animation = None;
        let started = Instant::now();
        let document: GraphDocument = match nodecraft_session::store::load(&path) {
            Ok(document) => document,
            Err(error) => {
                self.error = Some(error);
                self.status = "Open failed".to_owned();
                return;
            }
        };
        let document_id = document_id_for_path(&path);
        let canonical = PathBuf::from(&document_id);
        let document_id = self
            .documents
            .id_for_path(&canonical)
            .map(str::to_owned)
            .unwrap_or(document_id);
        if let Some(index) = self.tabs.iter().position(|tab| tab.document_id == document_id) {
            self.remember_document(&path);
            self.activate_tab(index);
            return;
        }
        let session = Session::new(document);
        self.remember_document(&path);
        self.open_session(
            session,
            Some(path),
            format!("Opened in {:.1} ms", started.elapsed().as_secs_f64() * 1000.0),
        );
    }

    fn remember_document(&mut self, path: &Path) {
        let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.recent_files.retain(|recent| recent != &absolute);
        self.recent_files.insert(0, absolute);
        self.recent_files.truncate(10);
    }

    fn clear_recent_documents(&mut self) {
        self.recent_files.clear();
    }

    fn save_active_tab(&mut self) {
        let Some(session) = self.session.clone() else { return };
        let Some(document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else { return };
        let _ = self.documents.set_session(document_id.clone(), session);
        if let Some(path) = &self.path {
            self.documents.set_path(&document_id, path);
        }
        let Some(tab) = self.tabs.get_mut(self.active_tab) else { return };
        tab.path = self.path.clone();
        tab.selected_nodes = self.selected_nodes.clone();
        tab.selected_connections = self.selected_connections.clone();
        tab.viewer_breakpoint = self.viewer_breakpoint.clone();
        tab.mesh = self.mesh.clone();
        tab.status = self.status.clone();
        tab.error = self.error.clone();
        tab.evaluation_ms = self.evaluation_ms;
        tab.cache_hits = self.cache_hits;
        tab.cache_misses = self.cache_misses;
        tab.camera = Some(self.viewport.camera);
    }

    fn save_document(&mut self, save_as: bool) {
        let Some(session) = self.session.clone() else {
            self.error = Some("No document loaded to save".to_owned());
            self.status = "Save failed".to_owned();
            return;
        };
        let path = if save_as || self.path.is_none() {
            let Some(path) = rfd::FileDialog::new()
                .add_filter("NodeCraft graph", &["nodecraft"])
                .set_file_name("untitled.nodecraft")
                .save_file()
            else {
                return;
            };
            normalize_document_path(path)
        } else {
            self.path.clone().unwrap_or_default()
        };
        if let Err(error) = nodecraft_session::store::save_atomic(&path, session.document()) {
            self.error = Some(error);
            self.status = "Save failed".to_owned();
            return;
        }
        self.path = Some(path.clone());
        self.remember_document(&path);
        if let Some(tab) = self.tabs.get(self.active_tab) {
            let document_id = tab.document_id.clone();
            let _ = self.documents.set_session(document_id.clone(), session);
            self.documents.set_path(&document_id, &path);
        }
        self.save_active_tab();
        self.status = format!("Saved {}", path.display());
        self.error = None;
    }

    fn restore_active_tab(&mut self) {
        self.pending_dirty.clear();
        let Some(tab) = self.tabs.get(self.active_tab).cloned() else { return };
        let Some(session) = self.documents.session(&tab.document_id).cloned() else {
            self.session = None;
            return;
        };
        self.session = Some(session);
        self.pending_dirty.clear();
        self.path = tab.path;
        self.selected_nodes = tab.selected_nodes;
        self.selected_connections = tab.selected_connections;
        self.viewer_breakpoint = tab.viewer_breakpoint;
        self.mesh = tab.mesh;
        self.status = tab.status;
        self.error = tab.error;
        self.evaluation_ms = tab.evaluation_ms;
        self.cache_hits = tab.cache_hits;
        self.cache_misses = tab.cache_misses;
        if let Some(camera) = tab.camera {
            self.viewport.camera = camera;
            self.camera_initialized = true;
            self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
        } else {
            self.camera_initialized = false;
            self.viewport.set_mesh(self.mesh.as_ref());
        }
        if let Some(path) = self.path.clone() {
            self.remember_document(&path);
        }
        self.node_editor.reset_view();
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        self.evaluate();
    }

    fn close_tab(&mut self, index: usize) {
        if index >= self.tabs.len() {
            return;
        }
        self.save_active_tab();
        let closing_active = index == self.active_tab;
        let removed = self.tabs.remove(index);
        if !self.tabs.iter().any(|tab| tab.document_id == removed.document_id) {
            self.documents.remove(&removed.document_id);
        }
        if self.tabs.is_empty() {
            self.session = None;
            self.path = None;
            self.selected_nodes.clear();
            self.selected_connections.clear();
            self.viewer_breakpoint = None;
            self.new_blank();
            return;
        }
        if closing_active {
            self.active_tab = index.min(self.tabs.len() - 1);
            self.restore_active_tab();
        } else if index < self.active_tab {
            self.active_tab -= 1;
        }
    }

    /// Replace the graph in the active tab without opening another tab.
    fn replace_active_session(&mut self, session: Session, status: String) {
        let selected_nodes: HashSet<String> = initial_viewer_target(session.document())
            .into_iter()
            .collect();
        let viewer_breakpoint = selected_nodes.iter().next().cloned();
        self.session = Some(session.clone());
        self.pending_dirty.clear();
        self.path = None;
        self.selected_nodes = selected_nodes.clone();
        self.selected_connections.clear();
        self.viewer_breakpoint = viewer_breakpoint.clone();
        self.mesh = None;
        self.camera_initialized = false;
        self.viewport.set_mesh(None);
        self.node_editor.reset_view();
        self.error = None;
        self.status = status;
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            let document_id = tab.document_id.clone();
            let _ = self.documents.set_session(document_id.clone(), session.clone());
            self.documents.clear_path(&document_id);
            tab.path = None;
            tab.selected_nodes = selected_nodes;
            tab.selected_connections.clear();
            tab.viewer_breakpoint = viewer_breakpoint;
            tab.mesh = None;
            tab.status = self.status.clone();
            tab.error = None;
            tab.evaluation_ms = 0.0;
            tab.cache_hits = 0;
            tab.cache_misses = 0;
        } else {
            self.open_session(self.session.clone().expect("session was just set"), None, self.status.clone());
            return;
        }
        self.evaluate();
    }

    fn open_session(&mut self, session: Session, path: Option<PathBuf>, status: String) {
        self.save_active_tab();
        let id = self.next_tab_id;
        self.next_tab_id = self.next_tab_id.wrapping_add(1);
        let document_id = match &path {
            Some(path) => document_id_for_path(path),
            None => {
                let id = self.next_document_id;
                self.next_document_id = self.next_document_id.wrapping_add(1);
                format!("untitled-{id}")
            }
        };
        let _ = self.documents.set_session(document_id.clone(), session.clone());
        if let Some(path) = &path {
            self.documents.set_path(&document_id, path);
        }
        let selected_nodes: HashSet<String> = initial_viewer_target(session.document())
            .into_iter()
            .collect();
        let viewer_breakpoint = selected_nodes.iter().next().cloned();
        let tab = DocumentTab {
            id,
            document_id,
            path: path.clone(),
            selected_nodes: selected_nodes.clone(),
            selected_connections: HashSet::new(),
            viewer_breakpoint: viewer_breakpoint.clone(),
            mesh: None,
            status: status.clone(),
            error: None,
            evaluation_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            camera: None,
        };
        self.tabs.push(tab);
        self.active_tab = self.tabs.len().saturating_sub(1);
        self.session = Some(session);
        self.pending_dirty.clear();
        self.path = path;
        self.selected_nodes = selected_nodes;
        self.selected_connections.clear();
        self.viewer_breakpoint = viewer_breakpoint;
        self.mesh = None;
        self.camera_initialized = false;
        self.viewport.set_mesh(None);
        self.node_editor.reset_view();
        self.error = None;
        self.status = status;
        self.eval_pending = None;
        self.layout_animation = None;
        self.node_position_animation = None;
        self.evaluate();
    }

    fn activate_tab(&mut self, index: usize) {
        if index >= self.tabs.len() || index == self.active_tab {
            return;
        }
        self.save_active_tab();
        self.active_tab = index;
        self.restore_active_tab();
    }

    fn remap_selection(&mut self) {
        let Some(session) = self.session.as_ref() else { return };
        let document = session.document();
        self.selected_nodes.retain(|id| document.nodes.iter().any(|node| &node.id == id));
        if self
            .viewer_breakpoint
            .as_ref()
            .is_some_and(|id| !self.selected_nodes.contains(id))
        {
            self.viewer_breakpoint = None;
        }
        self.selected_connections.retain(|key| {
            document.connections.iter().any(|connection| {
                connection.from == key.from
                    && connection.to == key.to
                    && connection.input == key.input
                    && connection.out == key.out
            })
        });
    }

    fn export_document(&mut self) {
        let Some(session) = &self.session else {
            self.error = Some("No document loaded to export".to_owned());
            self.status = "Export failed".to_owned();
            return;
        };
        let exporter = session
            .document()
            .nodes
            .iter()
            .rev()
            .find(|node| node.node_type == "exporter")
            .map(|node| node.id.clone());
        let Some(exporter) = exporter else {
            self.error = Some("No Export STEP node is connected".to_owned());
            self.status = "Export failed".to_owned();
            return;
        };
        self.viewer_breakpoint = Some(exporter.clone());
        self.status = format!("Exporting via {exporter}…");
        self.evaluate();
    }

    fn non_export_target(document: &GraphDocument, start: Option<&str>) -> Option<String> {
        let mut current = start.map(str::to_owned).or_else(|| {
            document
                .nodes
                .iter()
                .rev()
                .find(|node| node.node_type != "exporter")
                .map(|node| node.id.clone())
        })?;
        let mut seen = HashSet::new();
        while seen.insert(current.clone()) {
            let node = document.nodes.iter().find(|node| node.id == current)?;
            if node.node_type != "exporter" {
                return Some(node.id.clone());
            }
            let link = document
                .connections
                .iter()
                .find(|link| link.to == node.id && link.input == "shape")?;
            current = link.from.clone();
        }
        None
    }

    fn reevaluate_non_export(&mut self) {
        let Some(session) = &self.session else {
            self.error = Some("No document loaded".to_owned());
            self.status = "Re-evaluate failed".to_owned();
            return;
        };
        let Some(target) = Self::non_export_target(session.document(), self.viewer_breakpoint.as_deref()) else {
            self.error = Some("No non-export node to evaluate".to_owned());
            self.status = "Re-evaluate failed".to_owned();
            return;
        };
        self.viewer_breakpoint = Some(target.clone());
        self.error = None;
        self.status = format!("Re-evaluating {target}…");
        self.evaluate_with_cache_policy(true);
    }

    fn add_tool_node(&mut self, tool: ToolItem, workspace: egui::Rect) {
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        let panel = self.node_editor.panel_rect(workspace);
        let center = panel.center();
        let graph = self.node_editor.graph_pos(center).unwrap_or(egui::Pos2::ZERO);
        let mut used: HashSet<String> = session.document().nodes.iter().map(|node| node.id.clone()).collect();
        let id = unique_node_id(tool.node_type, &mut used);
        let values = match tool.primitive {
            Some(primitive) => serde_json::json!({ "primitiveType": primitive }),
            None => serde_json::json!({}),
        };
        let mut node = GraphNode {
            id: id.clone(),
            node_type: tool.node_type.to_owned(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        node.x = f64::from(graph.x) - f64::from(node_width(&node)) * 0.5;
        node.y = f64::from(graph.y) - f64::from(NODE_HEIGHT) * 0.5;
        if self.apply_commands(&[Command::AddNode { node }], Some("tool-dock-add-node")) {
            self.selected_nodes.clear();
            self.selected_nodes.insert(id.clone());
            self.selected_connections.clear();
            self.status = format!("Added {}", tool.label);
            self.evaluate();
        }
    }

    fn show_tool_dock(
        &mut self,
        ctx: &egui::Context,
        workspace: egui::Rect,
        backdrop: Option<(egui::TextureId, egui::Rect)>,
    ) {
        let theme = crate::theme::get();
        let dock = tool_dock_rect(workspace);
        let background = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("node-tool-dock-background"),
        ));
        if let Some((texture, uv)) = backdrop {
            ui_shape::image(&background, texture, dock, uv);
        } else {
            ui_shape::fill(&background, dock, theme.color(theme.blur_tint));
        }
        ui_shape::outline(
            &background,
            dock,
            egui::Stroke::new(1.0, theme.color(theme.border)),
        );

        let mut clicked_section = None;
        let mut section_rects = Vec::new();
        egui::Area::new(egui::Id::new("node-tool-dock"))
            .order(egui::Order::Tooltip)
            .fixed_pos(dock.min + egui::vec2(0.0, 8.0))
            .show(ctx, |ui| {
                ui.set_width(TOOL_DOCK_WIDTH);
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 6.0);
                for section in ToolSection::ALL {
                    let (rect, response) = ui.allocate_exact_size(
                        egui::Vec2::new(TOOL_DOCK_WIDTH, TOOL_DOCK_WIDTH),
                        egui::Sense::click(),
                    );
                    let active = self.tool_section == Some(section);
                    if active || response.hovered() {
                        let fill = if active {
                            theme.color(theme.accent)
                        } else {
                            theme.with_alpha(theme.border, 190)
                        };
                        ui_shape::fill(&ui.painter(), rect.shrink(4.0), fill);
                    }
                    let tint = crate::theme::Theme::hex(if active {
                        theme.text
                    } else if response.hovered() {
                        theme.icon_hover
                    } else {
                        theme.icon
                    });
                    let (name, source) = section.icon();
                    let svg = tint_svg(source, &tint);
                    let image = egui::Image::from_bytes(
                        format!("bytes://node-tool-section-{name}.svg"),
                        svg.into_bytes(),
                    )
                    .fit_to_exact_size(egui::Vec2::splat(19.0));
                    ui.put(rect.shrink(13.0), image);
                    let response = response.on_hover_text(section.label());
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if response.clicked_by(egui::PointerButton::Primary) {
                        clicked_section = Some(section);
                    }
                    section_rects.push((section, rect));
                }
            });

        let category_clicked = clicked_section.is_some();
        if let Some(section) = clicked_section {
            self.tool_section = if self.tool_section == Some(section) {
                None
            } else {
                Some(section)
            };
        }

        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.tool_section = None;
        }
        let Some(section) = self.tool_section else {
            return;
        };
        let tools = section.tools();
        let menu_height = 40.0 + tools.len() as f32 * 30.0;
        let y = section_rects
            .iter()
            .find(|(candidate, _)| *candidate == section)
            .map(|(_, rect)| rect.max.y + 4.0)
            .unwrap_or(dock.min.y + 8.0)
            .min((workspace.bottom() - 12.0 - menu_height).max(dock.min.y + 8.0));
        let menu_pos = egui::Pos2::new(dock.max.x + TOOL_DOCK_GAP, y);
        let mut chosen = None;
        let menu_response = egui::Area::new(egui::Id::new("node-tool-dock-menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(menu_pos)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(theme.color(theme.popup))
                    .stroke(egui::Stroke::new(1.0, theme.color(theme.border_strong)))
                    .corner_radius(egui::CornerRadius::same(12))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(182.0);
                        ui.label(
                            egui::RichText::new(section.label().to_uppercase())
                                .size(10.0)
                                .color(theme.color(theme.muted)),
                        );
                        ui.add_space(3.0);
                        for tool in tools {
                            let response = ui
                                .add_sized(
                                    [182.0, 28.0],
                                    egui::Button::new(
                                        egui::RichText::new(tool.label)
                                            .size(13.0)
                                            .color(theme.color(theme.text)),
                                    )
                                    .frame(false),
                                )
                                .on_hover_text(match tool.primitive {
                                    Some(primitive) => format!("Add {} ({primitive})", tool.label),
                                    None => format!("Add {}", tool.label),
                                });
                            if response.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            if response.clicked() {
                                chosen = Some(*tool);
                            }
                        }
                    });
            });
        if !category_clicked && menu_response.response.clicked_elsewhere() {
            self.tool_section = None;
        }
        if let Some(tool) = chosen {
            self.add_tool_node(tool, workspace);
            self.tool_section = None;
        }
    }

    fn new_blank(&mut self) {
        self.open_session(
            Session::new(GraphDocument::new("Untitled")),
            None,
            "New document".to_owned(),
        );
    }

    fn adopt_session_document(&mut self) {
        self.remap_selection();
        self.evaluate();
    }

    fn copy_selected_nodes(&mut self) -> bool {
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return false;
        };
        let selected: HashSet<&str> = self.selected_nodes.iter().map(String::as_str).collect();
        let Some(payload) = clipboard_payload(session.document(), &selected) else {
            self.status = "No nodes selected".to_owned();
            return false;
        };
        let count = payload.nodes.len();
        let text = match serde_json::to_string_pretty(&payload) {
            Ok(text) => text,
            Err(error) => {
                self.status = format!("Copy failed: {error}");
                return false;
            }
        };
        match arboard::Clipboard::new().and_then(|mut clipboard| clipboard.set_text(text)) {
            Ok(()) => {
                self.status = format!("Copied {count} node{}", if count == 1 { "" } else { "s" });
                true
            }
            Err(error) => {
                self.status = format!("Copy failed: {error}");
                false
            }
        }
    }

    fn cut_selected_nodes(&mut self) {
        if !self.copy_selected_nodes() {
            return;
        }
        let ids: Vec<String> = self.selected_nodes.iter().cloned().collect();
        let count = ids.len();
        let commands: Vec<Command> = ids
            .into_iter()
            .map(|id| Command::RemoveNode { id })
            .collect();
        if self.apply_commands(&commands, Some("cut-nodes")) {
            self.selected_nodes.clear();
            self.selected_connections.clear();
            self.viewer_breakpoint = None;
            self.evaluate();
            self.status = format!("Cut {count} node{}", if count == 1 { "" } else { "s" });
        }
    }

    fn paste_nodes(&mut self, text: String, origin: Option<egui::Pos2>) {
        let payload = match parse_node_clipboard(&text) {
            Ok(payload) => payload,
            Err(error) => {
                self.status = format!("Paste failed: clipboard is not node JSON ({error})");
                return;
            }
        };
        if payload.nodes.is_empty() {
            self.status = "Paste failed: clipboard has no nodes".to_owned();
            return;
        }
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        let (commands, selected_nodes, _) = paste_plan(session.document(), payload, origin);
        self.selected_nodes = selected_nodes;
        self.selected_connections.clear();
        if self.apply_commands(&commands, Some("paste-nodes")) {
            let count = self.selected_nodes.len();
            self.evaluate();
            self.status = format!("Pasted {count} node{}", if count == 1 { "" } else { "s" });
        } else {
            self.selected_nodes.clear();
        }
    }

    fn selected_transform_gizmo(&self) -> Option<Gizmo> {
        if self.selected_nodes.len() != 1 {
            return None;
        }
        let selected_id = self.selected_nodes.iter().next()?.clone();
        let session = self.session.as_ref()?;
        let document = session.document();
        let node = document.nodes.iter().find(|node| node.id == selected_id)?;
        if node.node_type != "translate" {
            return None;
        }
        let value = |key: &str| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let x = value("x");
        let y = value("y");
        let z = value("z");
        let chain = self
            .viewer_breakpoint
            .as_deref()
            .filter(|target| *target != selected_id)
            .and_then(|target| {
                nodecraft_session::placement::transform_chain(document, &selected_id, target)
            })
            .unwrap_or_else(nodecraft_session::placement::Affine3::identity);
        let local = node.values.get("space").and_then(serde_json::Value::as_str) == Some("local");
        // In local mode x/y/z are along the node's rotated axes. Compose that
        // rotation with the downstream chain once and use it everywhere: pivot,
        // triad orientation, and inverse drag mapping.
        let basis = if local {
            chain.compose(nodecraft_session::placement::Affine3::rotation(&node.values))
        } else {
            chain
        };
        let axes = [
            basis.apply_vector([1.0, 0.0, 0.0]),
            basis.apply_vector([0.0, 1.0, 0.0]),
            basis.apply_vector([0.0, 0.0, 1.0]),
        ];
        Some(Gizmo {
            node_id: selected_id,
            pivot: basis.apply_point([x, y, z]),
            x,
            y,
            z,
            rx: value("rx"),
            ry: value("ry"),
            rz: value("rz"),
            local,
            world_to_local: basis.inverse_linear(),
            axes,
        })
    }

    fn apply_commands(&mut self, commands: &[Command], coalesce: Option<&str>) -> bool {
        let Some(session) = self.session.as_mut() else { return false };
        match session.apply_commands_coalesced(
            commands,
            coalesce,
            std::time::Duration::from_millis(700),
        ) {
            Ok(change) => {
                if change.changed && !change.dirty_nodes.is_empty() {
                    self.pending_dirty.extend(change.dirty_nodes);
                    self.auto_eval_due = Some(Instant::now());
                }
                change.changed
            }
            Err(error) => {
                self.error = Some(format!("Graph edit failed: {error}"));
                false
            }
        }
    }

    fn undo(&mut self) {
        let before = self.session.as_ref().map(|session| node_positions(session.document()));
        let changed = self.session.as_mut().is_some_and(|session| session.undo().ok().flatten().is_some());
        if changed {
            self.layout_animation = None;
            let after = self.session.as_ref().map(|session| node_positions(session.document()));
            self.adopt_session_document();
            if let (Some(before), Some(after)) = (before, after) {
                let (start, end) = moved_positions(&before, &after);
                if !start.is_empty() {
                    self.node_position_animation = Some(NodePositionAnimation {
                        start,
                        end,
                        started: Instant::now(),
                    });
                }
            }
        }
    }

    fn redo(&mut self) {
        let before = self.session.as_ref().map(|session| node_positions(session.document()));
        let changed = self.session.as_mut().is_some_and(|session| session.redo().ok().flatten().is_some());
        if changed {
            self.layout_animation = None;
            let after = self.session.as_ref().map(|session| node_positions(session.document()));
            self.adopt_session_document();
            if let (Some(before), Some(after)) = (before, after) {
                let (start, end) = moved_positions(&before, &after);
                if !start.is_empty() {
                    self.node_position_animation = Some(NodePositionAnimation {
                        start,
                        end,
                        started: Instant::now(),
                    });
                }
            }
        }
    }

    fn evaluate(&mut self) {
        self.evaluate_with_cache_policy(false);
    }

    fn evaluate_with_cache_policy(&mut self, clear_cache: bool) {
        self.auto_eval_due = None;
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        if session.document().nodes.is_empty() {
            self.status = "Document has no nodes".to_owned();
            self.eval_pending = None;
            return;
        }
        if !self.pending_dirty.is_empty() {
            let target = engine_bridge::target_for(session.document(), self.viewer_breakpoint.as_deref());
            let affected = target.as_deref().is_some_and(|target| {
                let closure = target_closure(session.document(), target);
                self.pending_dirty.iter().any(|id| closure.contains(id))
            });
            self.pending_dirty.clear();
            if !affected {
                self.status = "Graph updated; view is unchanged".to_owned();
                return;
            }
        }
        self.eval_next = self.eval_next.wrapping_add(1);
        let id = self.eval_next;
        self.eval_pending = Some(id);
        self.node_editor.clear_activity();
        self.status = "Evaluating…".to_owned();
        let tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let mut linked_documents = std::collections::BTreeMap::new();
        for (document_id, document) in self.documents.documents() {
            if let Some(path) = self.documents.path(document_id) {
                linked_documents.insert(path.to_path_buf(), document.clone());
            }
        }
        self.eval_worker.send(EvalJob {
            id,
            tab_id,
            linked_documents,
            document: session.document().clone(),
            selected: self.viewer_breakpoint.clone(),
            clear_cache,
        });
        self.egui_ctx.request_repaint();
    }

    fn advance_inspector_animation(&mut self, ctx: &egui::Context) {
        let Some(animation) = &self.inspector_animation else { return };
        let progress = (animation.started.elapsed().as_secs_f32() / 0.24).clamp(0.0, 1.0);
        let eased = progress * progress * (3.0 - 2.0 * progress);
        self.inspector_reveal = animation.from + (animation.to - animation.from) * eased;
        if progress >= 1.0 {
            self.inspector_reveal = animation.to;
            self.inspector_animation = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn toggle_inspector(&mut self) {
        self.inspector_open = !self.inspector_open;
        self.inspector_reflow = true;
        self.inspector_animation = Some(InspectorAnimation {
            started: Instant::now(),
            from: self.inspector_reveal,
            to: if self.inspector_open { 1.0 } else { 0.0 },
        });
        self.egui_ctx.request_repaint();
    }

    fn show_app_settings(&mut self, ctx: &egui::Context) {
        let mut theme = crate::theme::get();
        let theme_before = theme.clone();
        let mut theme_selection_changed = false;
        let mut theme_error: Option<String> = None;
        let mut theme_rename_id = self.theme_rename_id.clone();
        let mut theme_rename_name = self.theme_rename_name.clone();
        let mut theme_rename_focus = self.theme_rename_focus;
        let mut tab = self.settings_tab;
        let mut view = self.viewport.view();
        let mut activity = self.activity;
        let mut close = false;
        let mut builder = egui::ViewportBuilder::default()
            .with_title("NodeCraft Settings")
            .with_inner_size([760.0, 620.0])
            .with_min_inner_size([520.0, 400.0])
            .with_resizable(true)
            // Keep settings above the main NodeCraft window even if the main
            // window is clicked or activated.
            .with_always_on_top()
            // Native OS chrome: keep close and minimize, hide maximize. The
            // platform owns the button order/side and rounded window corners.
            // macOS verified: close/minimize at left, no zoom, rounded corners.
            // TODO(future agent): verify native button placement and rounded
            // corners on Windows and Linux; keep OS chrome, not custom controls.
            .with_decorations(true)
            .with_close_button(true)
            .with_minimize_button(false)
            .with_maximize_button(false);
        let parent_outer = ctx.input(|input| input.viewport().outer_rect);
        let monitor_size = ctx.input(|input| input.viewport().monitor_size);
        let mut center_once = !self.settings_open_last;
        if !self.settings_open_last {
            self.settings_open_last = true;
            if let Some(parent) = parent_outer {
                let settings_size = egui::vec2(760.0, 620.0);
                let mut pos = parent.center() - settings_size * 0.5;
                if let Some(monitor) = monitor_size {
                    pos.x = pos.x.clamp(0.0, (monitor.x - settings_size.x).max(0.0));
                    pos.y = pos.y.clamp(0.0, (monitor.y - settings_size.y).max(0.0));
                }
                builder = builder.with_position(pos);
            }
        }
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("nodecraft-app-settings"),
            builder,
            |ui, _class| {
                if ui.ctx().input(|input| input.viewport().close_requested()) {
                    close = true;
                    return;
                }
                if center_once {
                    if let (Some(parent), Some(child)) = (
                        parent_outer,
                        ui.ctx().input(|input| input.viewport().outer_rect),
                    ) {
                        let mut pos = parent.center() - child.size() * 0.5;
                        if let Some(monitor) = monitor_size {
                            pos.x = pos.x.clamp(0.0, (monitor.x - child.width()).max(0.0));
                            pos.y = pos.y.clamp(0.0, (monitor.y - child.height()).max(0.0));
                        }
                        ui.ctx()
                            .send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                        center_once = false;
                    }
                }
                let panel = ui.max_rect();
                let left = egui::Rect::from_min_max(
                    panel.min,
                    egui::pos2(panel.min.x + 124.0, panel.max.y),
                );
                let right = egui::Rect::from_min_max(
                    egui::pos2(left.max.x, panel.min.y),
                    panel.max,
                );
                ui.painter().rect_filled(left, 0.0, theme.color(theme.panel));
                ui.painter().rect_filled(right, 0.0, theme.color(theme.window));

                let tabs = [
                    SettingsTab::Appearance,
                    SettingsTab::Viewport,
                    SettingsTab::Diagnostics,
                ];
                let mut tab_y = left.min.y + 12.0;
                for candidate in tabs {
                    let button = egui::Rect::from_min_size(
                        egui::pos2(left.min.x + 10.0, tab_y),
                        egui::vec2(104.0, 30.0),
                    );
                    let response = ui.interact(
                        button,
                        ui.make_persistent_id(("settings-tab", candidate.label())),
                        egui::Sense::click(),
                    );
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if response.clicked() {
                        tab = candidate;
                    }
                    if tab == candidate {
                        ui.painter().rect_filled(
                            button,
                            egui::CornerRadius::same(7),
                            theme.color(theme.accent),
                        );
                    }
                    ui.painter().text(
                        egui::pos2(button.min.x + 10.0, button.center().y),
                        egui::Align2::LEFT_CENTER,
                        candidate.label(),
                        egui::FontId::proportional(13.0),
                        if tab == candidate {
                            theme.color(theme.window)
                        } else {
                            theme.color(theme.muted)
                        },
                    );
                    tab_y += 36.0;
                }

                let palette_row_width = (right.width() - 64.0).max(300.0);
                let mut content = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(right.shrink2(egui::vec2(18.0, 14.0)))
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                match tab {
                    SettingsTab::Appearance => {
                        content.heading("Appearance");
                        let mut selected_theme = crate::theme::active_preset();
                        content.horizontal(|ui| {
                            ui.label("Theme");
                            let combo_width = (palette_row_width
                                - 42.0
                                - 78.0
                                - ui.spacing().item_spacing.x * 4.0)
                                .max(120.0);
                            if let Some(rename_id) = theme_rename_id.clone() {
                                let response = ui.add_sized(
                                    [combo_width, 24.0],
                                    egui::TextEdit::singleline(&mut theme_rename_name)
                                        .hint_text("Theme name"),
                                );
                                if theme_rename_focus {
                                    response.request_focus();
                                    theme_rename_focus = false;
                                }
                                if ui.ctx().input(|input| input.key_pressed(egui::Key::Escape)) {
                                    theme_rename_id = None;
                                } else {
                                    let commit = response.lost_focus()
                                        || (response.has_focus()
                                            && ui.input(|input| {
                                                input.key_pressed(egui::Key::Enter)
                                            }));
                                    if commit && !theme_rename_name.trim().is_empty() {
                                        if let Some(preset) = crate::theme::available_themes()
                                            .into_iter()
                                            .find(|preset| preset.id == rename_id)
                                        {
                                            match crate::theme::rename_theme(
                                                &preset,
                                                &theme_rename_name,
                                            ) {
                                                Ok(renamed) => {
                                                    selected_theme = renamed;
                                                    theme_rename_id = None;
                                                    theme_selection_changed = true;
                                                }
                                                Err(error) => theme_error = Some(error),
                                            }
                                        }
                                    }
                                }
                            } else {
                                egui::ComboBox::from_id_salt("theme-preset")
                                    .width(combo_width)
                                    .popup_style(egui::style::StyleModifier::from(crate::theme::menu_style))
                                    .selected_text(crate::theme::active_name())
                                    .show_ui(ui, |ui| {
                                        for preset in crate::theme::available_themes() {
                                            if ui
                                                .selectable_label(
                                                    selected_theme.id == preset.id,
                                                    &preset.name,
                                                )
                                                .clicked()
                                                && selected_theme.id != preset.id
                                            {
                                                match crate::theme::load_preset(&preset) {
                                                    Ok(loaded) => {
                                                        theme = loaded;
                                                        selected_theme = preset;
                                                        theme_selection_changed = true;
                                                    }
                                                    Err(error) => theme_error = Some(error),
                                                }
                                            }
                                        }
                                    });
                            }
                            let can_reset = theme != crate::theme::active_defaults();
                            let reset = theme_revert_button(ui, can_reset)
                                .on_hover_text(if can_reset {
                                    "Reset current theme to defaults"
                                } else {
                                    "Theme already matches defaults"
                                });
                            if reset.clicked() && can_reset {
                                match crate::theme::reset_current_theme() {
                                    Ok(loaded) => {
                                        theme = loaded;
                                        selected_theme = crate::theme::active_preset();
                                        theme_selection_changed = true;
                                    }
                                    Err(error) => theme_error = Some(error),
                                }
                            }
                            let active_preset = crate::theme::active_preset();
                            let can_edit = active_preset.path.is_some();
                            let edit = theme_edit_button(ui, can_edit)
                                .on_hover_text(if can_edit {
                                    "Edit theme name"
                                } else {
                                    "Built-in themes cannot be renamed"
                                });
                            if edit.clicked() && can_edit {
                                theme_rename_id = Some(active_preset.id);
                                theme_rename_name = active_preset.name;
                                theme_rename_focus = true;
                            }
                            let add = theme_add_button(ui).on_hover_text("New theme");
                            if add.clicked() {
                                match crate::theme::create_theme("New Theme", &theme) {
                                    Ok(preset) => {
                                        selected_theme = preset.clone();
                                        theme_rename_id = Some(preset.id);
                                        theme_rename_name = preset.name;
                                        theme_rename_focus = true;
                                        theme_selection_changed = true;
                                    }
                                    Err(error) => {
                                        eprintln!("theme: cannot create user theme: {error}");
                                        theme_error = Some(error);
                                    }
                                }
                            }
                        });
                        if let Some(error) = &theme_error {
                            content.small(egui::RichText::new(error).color(theme.color(theme.text)));
                        }
                        content.small(
                            "Click a colour dot to edit. Edits apply live and are saved as \
                             NodeCraft Dark (custom).",
                        );
                        content.add_space(6.0);
                        content.scope(|ui| {
                            ui.style_mut().spacing.scroll = crate::theme::scroll_style_general();
                            ui.visuals_mut().widgets.inactive.bg_fill =
                                theme.with_alpha(theme.muted, 70);
                            ui.visuals_mut().widgets.hovered.bg_fill =
                                theme.with_alpha(theme.muted, 155);
                            ui.visuals_mut().widgets.active.bg_fill =
                                theme.with_alpha(theme.accent, 210);
                            ui.visuals_mut().widgets.inactive.corner_radius =
                                egui::CornerRadius::same(4);
                            ui.visuals_mut().widgets.hovered.corner_radius =
                                egui::CornerRadius::same(4);
                            ui.visuals_mut().widgets.active.corner_radius =
                                egui::CornerRadius::same(4);
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    for group in crate::theme::palette_groups(&theme) {
                                        egui::Frame::new()
                                            .fill(theme.color(theme.panel))
                                            .corner_radius(egui::CornerRadius::same(7))
                                            .inner_margin(egui::Margin::same(8))
                                            .show(ui, |ui| {
                                                ui.set_min_width(palette_row_width + 16.0);
                                                ui.label(egui::RichText::new(group.title).strong());
                                            ui.small(group.description);
                                            ui.add_space(3.0);
                                            for entry in &group.entries {
                                                theme_palette_row(ui, entry, &mut theme, palette_row_width);
                                            }
                                        });
                                    ui.add_space(6.0);
                                    }
                                });
                        });
                    }
                    SettingsTab::Viewport => {
                        content.heading("Viewport");
                        content.checkbox(&mut view.cad_mouse, "CAD mouse (middle orbit, Shift+middle pan)");
                        content.add(
                            egui::Slider::new(&mut view.pan_speed, 0.25..=4.0)
                                .text("Pan speed")
                                .max_decimals(2),
                        );
                        content.add(
                            egui::Slider::new(&mut view.orbit_speed, 0.25..=4.0)
                                .text("Orbit speed")
                                .max_decimals(2),
                        );
                        content.checkbox(&mut view.translucent, "Translucent");
                        content.checkbox(&mut view.edges, "Edges");
                        content.checkbox(&mut view.triangles, "Triangles");
                        content.checkbox(&mut view.normals, "Face normals");
                        content.checkbox(&mut view.face_colours, "Face colours");
                        content.checkbox(&mut view.zebra, "Zebra analysis");
                    }
                    SettingsTab::Diagnostics => {
                        content.heading("Diagnostics");
                        content.monospace(format!("NodeCraft {}", env!("CARGO_PKG_VERSION")));
                        content.monospace(format!("GIT: {BUILD_GIT_HASH}"));
                        #[cfg(feature = "gpu-viewport")]
                        content.monospace(format!("MSAA: {}x", crate::gpu_viewport::msaa_samples()));
                        content.add_space(8.0);
                        content.heading("Node activity glow");
                        content.small(
                            "Diagnostic pulse shown while each node is processing. \n\
                             Hold is applied from node start and is at least 500 ms.",
                        );
                        content.add(
                            egui::Slider::new(&mut activity.fade_in_ms, 0..=500)
                                .text("Fade in (ms)"),
                        );
                        content.add(
                            egui::Slider::new(&mut activity.hold_ms, 500..=5000)
                                .text("Hold (ms, min 500)"),
                        );
                        content.add(
                            egui::Slider::new(&mut activity.fade_out_ms, 0..=500)
                                .text("Fade out (ms)"),
                        );
                    }
                }
                #[cfg(target_os = "macos")]
                hide_settings_zoom_button();
            },
        );
        self.theme_rename_id = theme_rename_id;
        self.theme_rename_name = theme_rename_name;
        self.theme_rename_focus = theme_rename_focus;
        if theme_selection_changed {
            self.theme_dirty = false;
            self.theme_save_due = None;
        }
        if theme != theme_before {
            crate::theme::set(theme.clone());
            configure_style(ctx);
            self.node_editor.set_theme(canvas_theme(&theme));
            self.viewport.set_theme(&theme);
            if !theme_selection_changed {
                self.theme_dirty = true;
                self.theme_save_due =
                    Some(Instant::now() + std::time::Duration::from_millis(350));
            }
            ctx.request_repaint();
        }
        if view != self.viewport.view() {
            self.viewport.set_view(view);
        }
        if activity != self.activity {
            self.activity = activity;
            self.node_editor.set_activity_timings(
                u64::from(activity.fade_in_ms),
                u64::from(activity.hold_ms),
                u64::from(activity.fade_out_ms),
            );
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if close {
            self.settings_open = false;
            self.settings_open_last = false;
        }
        self.settings_tab = tab;
    }

    fn advance_layout_animation(&mut self, ctx: &egui::Context) {
        let Some(animation) = &self.layout_animation else { return };
        let progress = (animation.started.elapsed().as_secs_f32()
            / LAYOUT_ANIMATION.as_secs_f32().max(0.001))
        .clamp(0.0, 1.0);
        let eased = progress * progress * (3.0 - 2.0 * progress);
        let start = animation.start.clone();
        let end = animation.end.clone();
        let commands: Vec<Command> = end
            .iter()
            .map(|(id, (x1, y1))| {
                let (x0, y0) = start.get(id).copied().unwrap_or((*x1, *y1));
                Command::SetPosition {
                    node: id.clone(),
                    x: x0 + (x1 - x0) * f64::from(eased),
                    y: y0 + (y1 - y0) * f64::from(eased),
                }
            })
            .collect();
        self.apply_commands(&commands, Some("cleanup-layout"));
        if progress >= 1.0 {
            self.layout_animation = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn advance_node_position_animation(&mut self, ctx: &egui::Context) {
        let Some(animation) = &self.node_position_animation else { return };
        if animation.started.elapsed() >= Duration::from_millis(450) {
            self.node_position_animation = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn poll_evaluation(&mut self) {
        let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
        let latest_id = self.eval_next;
        while let Some((id, tab_id, event)) = self.eval_worker.try_recv_progress() {
            if id != latest_id || tab_id != active_tab_id {
                continue;
            }
            match event.phase {
                nodecraft_engine::ProgressPhase::Start => {
                    self.node_editor.activity_started(&event.node_id);
                }
                nodecraft_engine::ProgressPhase::Complete => {
                    self.node_editor.activity_finished(&event.node_id);
                }
                nodecraft_engine::ProgressPhase::BrepStart
                | nodecraft_engine::ProgressPhase::BrepComplete => {}
            }
        }
        let Some(pending) = self.eval_pending else { return };
        while let Some((id, tab_id, evaluation)) = self.eval_worker.try_recv() {
            let active_tab_id = self.tabs.get(self.active_tab).map(|tab| tab.id).unwrap_or(0);
            if id != pending || tab_id != active_tab_id {
                continue;
            }
            self.eval_pending = None;
            self.evaluation_ms = evaluation.elapsed_ms;
            self.cache_hits = evaluation.cache_hits;
            self.cache_misses = evaluation.cache_misses;
            self.error = evaluation.error;
            let had_mesh = self.mesh.is_some();
            self.mesh = evaluation.mesh;
            let preserve = std::mem::take(&mut self.preserve_view_for_next_mesh)
                || (self.camera_initialized && had_mesh);
            if preserve {
                self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
            } else {
                self.viewport.set_mesh(self.mesh.as_ref());
                self.camera_initialized = true;
            }
            self.node_editor.pulse_glow();
            self.warmup_until = self
                .warmup_until
                .max(Instant::now() + std::time::Duration::from_millis(500));
            self.status = if self.error.is_none() {
                format!(
                    "Evaluated {} in {:.1} ms ({} cache hits, {} misses)",
                    evaluation.target, self.evaluation_ms, self.cache_hits, self.cache_misses
                )
            } else {
                format!("Evaluation failed in {:.1} ms", self.evaluation_ms)
            };
            self.egui_ctx.request_repaint();
        }
    }

    fn add_dropped_step(&mut self, path: PathBuf, pointer: Option<egui::Pos2>) {
        if self.session.is_none() {
            self.new_blank();
        }
        let Some(session) = &self.session else { return };
        let graph = pointer
            .and_then(|point| self.node_editor.graph_pos(point))
            .or_else(|| {
                session
                    .document()
                    .nodes
                    .last()
                    .map(|node| egui::pos2(node.x as f32 + 220.0, node.y as f32 + 80.0))
            })
            .unwrap_or(egui::Pos2::ZERO);
        let mut used: HashSet<String> = session.document().nodes.iter().map(|node| node.id.clone()).collect();
        let id = unique_node_id("step", &mut used);
        let mut node = GraphNode {
            id: id.clone(),
            node_type: "stepimporter".to_owned(),
            values: serde_json::json!({ "path": path.display().to_string(), "heal": true }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        node.x = f64::from(graph.x) - f64::from(node_width(&node)) * 0.5;
        node.y = f64::from(graph.y) - f64::from(NODE_HEIGHT) * 0.5;
        if self.apply_commands(&[Command::AddNode { node }], Some("drop-step-import")) {
            self.selected_nodes.clear();
            self.selected_nodes.insert(id.clone());
            self.selected_connections.clear();
            self.status = format!("Added STEP import: {}", path.display());
            self.evaluate();
        }
    }

    fn accept_dropped_files(&mut self, ctx: &egui::Context) {
        if let Some(pointer) = ctx.input(|input| input.pointer.latest_pos()) {
            self.last_pointer = Some(pointer);
        }
        let (document, step) = ctx.input(|input| {
            let mut document = None;
            let mut step = None;
            for file in &input.raw.dropped_files {
                let path = file.path().to_path_buf();
                if document.is_none() && is_nodecraft_path(&path) {
                    document = Some(path);
                } else if step.is_none() && is_step_path(&path) {
                    step = Some(path);
                }
            }
            (document, step)
        });
        if let Some(path) = step {
            self.add_dropped_step(path, self.last_pointer);
        } else if let Some(path) = document {
            self.status = format!("Opening {}", path.display());
            self.open(path);
        }
    }

    fn poll_control(&mut self) {
        while let Ok(request) = self.control_requests.try_recv() {
            let reply = match request.command {
                control_server::Command::ReplaceProject(value) => {
                    match serde_json::from_value::<GraphDocument>(value) {
                        Ok(document) => {
                            self.replace_active_session(Session::new(document), "MCP project".to_owned());
                            self.node_editor.request_fit();
                            self.node_editor.reset_view();
                            "{\"ok\":true}".to_owned()
                        }
                        Err(error) => format!(
                            "{{\"ok\":false,\"error\":{}}}",
                            serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"invalid project\"".to_owned())
                        ),
                    }
                }
                control_server::Command::Project(value) => {
                    match serde_json::from_value::<GraphDocument>(value) {
                        Ok(document) => {
                            self.open_session(Session::new(document), None, "MCP project".to_owned());
                            self.node_editor.request_fit();
                            self.node_editor.reset_view();
                            "{\"ok\":true}".to_owned()
                        }
                        Err(error) => format!(
                            "{{\"ok\":false,\"error\":{}}}",
                            serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"invalid project\"".to_owned())
                        ),
                    }
                }
                control_server::Command::OpenStep(path) => {
                    let mut document = GraphDocument::new("MCP");
                    document.nodes.push(GraphNode {
                        id: "step".to_owned(),
                        node_type: "stepimporter".to_owned(),
                        values: serde_json::json!({ "path": path, "heal": true }),
                        suppressed: false,
                        x: 0.0,
                        y: 0.0,
                    });
                    self.open_session(Session::new(document), None, format!("MCP STEP: {path}"));
                    self.node_editor.request_fit();
                    self.node_editor.reset_view();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::FitGraph => {
                    self.node_editor.request_fit();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::ZoomOut => {
                    self.node_editor.zoom_out();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::EnsureVisible => {
                    self.node_editor.request_fit();
                    self.viewport.fit();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::State => {
                    if let Some(session) = &self.session {
                        serde_json::json!({
                        "nodes": session.document().nodes,
                        "connections": session.document().connections,
                        "viewed": self.viewer_breakpoint,
                        "selection": self.selected_nodes,
                        "errors": self.error,
                        "status": self.status,
                        "mesh": self.mesh.as_ref().map(|mesh| serde_json::json!({
                            "vertices": mesh.vertices.len(),
                            "faces": mesh.faces.len(),
                        })),
                        }).to_string()
                    } else {
                        "{\"ok\":false,\"error\":\"no document\"}".to_owned()
                    }
                }
            };
            let _ = request.reply.send(reply);
        }
    }

}

impl eframe::App for NodeCraftApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        {
            let save_as = ctx.input_mut(|input| {
                input.consume_shortcut(&egui::KeyboardShortcut::new(
                    egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                    egui::Key::S,
                ))
            });
            let save = !save_as
                && ctx.input_mut(|input| {
                    input.consume_shortcut(&egui::KeyboardShortcut::new(
                        egui::Modifiers::COMMAND,
                        egui::Key::S,
                    ))
                });
            if save_as {
                self.save_document(true);
            } else if save {
                self.save_document(false);
            }
        }
        let view = self.viewport.view();
        let theme_id = Some(crate::theme::active_preset().id);
        if view != self.last_saved_view
            || theme_id != self.last_saved_theme
            || self.recent_files != self.last_saved_recent
        {
            self.last_saved_view = view;
            self.last_saved_theme = theme_id.clone();
            self.last_saved_recent = self.recent_files.clone();
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(500));
            ctx.request_repaint_after(std::time::Duration::from_millis(500));
        }
        if self.view_save_due.is_some_and(|due| Instant::now() >= due) {
            if save_app_settings(
                self.last_saved_view,
                self.last_saved_theme.clone(),
                self.activity,
                &self.last_saved_recent,
            )
            .is_ok()
            {
                self.view_save_due = None;
            } else {
                self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(2000));
            }
        }
        self.poll_evaluation();
        if self.eval_pending.is_none()
            && self
                .auto_eval_due
                .is_some_and(|due| due.elapsed() >= std::time::Duration::from_millis(32))
        {
            self.evaluate();
        }
        if self.auto_eval_due.is_some() || self.eval_pending.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        if self.eval_pending.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        if Instant::now() < self.warmup_until {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }

        let Some((path, frames)) = self.screenshot.as_mut() else { return };
        *frames += 1;
        // Keep the event loop awake and ask again until the compositor has presented
        // at least one complete frame. A single request can race the first show.
        ctx.request_repaint();
        if self.eval_pending.is_none() && *frames >= 10 && (*frames - 10) % 30 == 0 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        let image = ctx.input(|input| {
            input.raw.events.iter().find_map(|event| match event {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = image {
            let rgba: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
            match image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                image.size[0] as u32,
                image.size[1] as u32,
                rgba,
            ) {
                Some(buffer) => match buffer.save(&*path) {
                    Ok(()) => eprintln!("screenshot saved: {}", path.display()),
                    Err(error) => eprintln!("screenshot save failed: {error}"),
                },
                None => eprintln!("screenshot buffer dimensions were invalid"),
            }
            self.screenshot = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        } else if *frames > 3600 {
            eprintln!("screenshot timed out after 3600 frames");
            self.screenshot = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        let theme = crate::theme::get();
        theme.color(theme.window).to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.render_model {
            egui::CentralPanel::no_frame().show(ui, |ui| {
                let rect = ui.max_rect();
                self.viewport
                    .show_with_backdrop_blockers(ui, rect, &[], rect);
            });
            return;
        }
        self.accept_dropped_files(ui.ctx());
        self.poll_control();
        if let Some(direction) = window_resize_direction(ui.ctx()) {
            ui.ctx().set_cursor_icon(resize_cursor(direction));
            if ui.ctx().input(|input| input.pointer.primary_pressed()) {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
            }
        }
        self.advance_inspector_animation(ui.ctx());
        self.advance_layout_animation(ui.ctx());
        self.advance_node_position_animation(ui.ctx());
        // Standard document shortcut: close the active tab, not the window.
        // Keep this ahead of the tab bar so it also works while a node inspector
        // field has focus.
        if ui.ctx().input(|input| input.modifiers.command && input.key_pressed(egui::Key::W)) {
            let index = self.active_tab;
            self.close_tab(index);
            ui.ctx().request_repaint();
            return;
        }
        let typing = ui.ctx().egui_wants_keyboard_input();
        // Command-A selects the whole graph only while the pointer is over the
        // node panel. Text fields retain their normal select-all behavior.
        let pointer_over_node_canvas = ui
            .ctx()
            .pointer_latest_pos()
            .is_some_and(|pointer| {
                self.node_editor
                    .panel_screen_rect()
                    .is_some_and(|rect| rect.contains(pointer))
            });
        if !typing
            && pointer_over_node_canvas
            && ui.ctx().input(|input| input.modifiers.command && input.key_pressed(egui::Key::A))
            && let Some(session) = &self.session
        {
            self.selected_nodes = session
                .document()
                .nodes
                .iter()
                .map(|node| node.id.clone())
                .collect();
            self.selected_connections.clear();
            self.status = format!("Selected {} nodes", self.selected_nodes.len());
            ui.ctx().request_repaint();
        }
        let clipboard_action = ui.ctx().input(|input| clipboard_action(&input.events));
        if !typing {
            let paste_origin = ui
                .ctx()
                .pointer_latest_pos()
                .and_then(|pointer| self.node_editor.graph_pos(pointer));
            match clipboard_action {
                Some(ClipboardAction::Paste(text)) => self.paste_nodes(text, paste_origin),
                Some(ClipboardAction::Cut) => self.cut_selected_nodes(),
                Some(ClipboardAction::Copy) => {
                    self.copy_selected_nodes();
                }
                None => {}
            }
        }
        let undo_pressed = !typing && ui.input(|i| i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z));
        let redo_pressed = !typing && ui.input(|i| i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z));
        if redo_pressed {
            self.redo();
        } else if undo_pressed {
            self.undo();
        }
        if !typing
            && (!self.selected_nodes.is_empty() || !self.selected_connections.is_empty())
            && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            let commands = delete_selection_commands(&self.selected_nodes, &self.selected_connections);
            let node_count = self.selected_nodes.len();
            let cable_count = self.selected_connections.len();
            self.selected_nodes.clear();
            self.selected_connections.clear();
            self.viewer_breakpoint = None;
            if self.apply_commands(&commands, Some("delete-selection")) {
                self.evaluate();
                self.status = format!(
                    "Deleted {node_count} node{} and {cable_count} cable{}",
                    if node_count == 1 { "" } else { "s" },
                    if cable_count == 1 { "" } else { "s" }
                );
            }
        }
        let theme = crate::theme::get();
        let mut requested_tab: Option<usize> = None;
        let mut requested_close: Option<usize> = None;
        egui::Panel::top("app-bar")
            .exact_size(50.0)
            .frame(
                egui::Frame::new()
                    .fill(theme.color(theme.window))
                    .inner_margin(egui::Margin::symmetric(8, 0)),
            )
            .show(ui, |ui| {
                ui.style_mut().interaction.selectable_labels = false;
                ui.horizontal_centered(|ui| {
                    // The 40 px bar is the window title bar. Buttons/widgets added
                    // below take precedence; dragging empty bar space starts a
                    // native window drag.
                    let bar_drag = ui.interact(
                        ui.max_rect(),
                        egui::Id::new("app-bar-drag"),
                        egui::Sense::drag(),
                    );
                    // The custom title bar owns dragging on macOS. Other
                    // platforms keep their native title bar, so empty space in
                    // this toolbar must not start a second window drag.
                    if cfg!(target_os = "macos")
                        && bar_drag.drag_started()
                        && window_resize_direction(ui.ctx()).is_none()
                    {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }

                    #[cfg(target_os = "macos")]
                    {
                        let (lights, _) = ui.allocate_exact_size(
                            egui::vec2(76.0, 50.0),
                            egui::Sense::hover(),
                        );
                        let colors = [
                            egui::Color32::from_rgb(255, 95, 86),
                            egui::Color32::from_rgb(255, 189, 46),
                            egui::Color32::from_rgb(39, 201, 63),
                        ];
                        for (index, color) in colors.into_iter().enumerate() {
                            let center = egui::Pos2::new(
                                lights.min.x + 16.0 + index as f32 * 20.0,
                                lights.center().y,
                            );
                            let hit = egui::Rect::from_center_size(center, egui::Vec2::splat(18.0));
                            let response = ui.interact(
                                hit,
                                egui::Id::new(("app-bar-light", index)),
                                egui::Sense::click(),
                            );
                            let fill = if response.hovered() {
                                color.gamma_multiply(1.15)
                            } else {
                                color
                            };
                            ui.painter().circle_filled(center, 6.0, fill);
                            if response.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            if response.clicked() {
                                let command = match index {
                                    0 => egui::ViewportCommand::Close,
                                    1 => egui::ViewportCommand::Minimized(true),
                                    _ => {
                                        let fullscreen = ui.ctx().input(|input| {
                                            input.viewport().fullscreen.unwrap_or(false)
                                        });
                                        egui::ViewportCommand::Fullscreen(!fullscreen)
                                    }
                                };
                                ui.ctx().send_viewport_cmd(command);
                            }
                        }
                    }

                    ui.add_space(4.0);
                    ui.strong("NodeCraft");
                    ui.add_space(8.0);
                    let (file_button, _) = egui::containers::menu::MenuButton::from_button(
                        egui::Button::new(egui::RichText::new("File   ").size(13.0))
                            .frame(false),
                    )
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .style(crate::theme::menu_style),
                    )
                    .ui(ui, |ui| {
                        ui.set_min_width(230.0);
                        if ui.button("New").clicked() {
                            ui.close();
                            self.new_blank();
                        }
                        if ui.button("Open…").clicked() {
                            ui.close();
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("NodeCraft graph", &["nodecraft"])
                                .pick_file()
                            {
                                self.open(path);
                            }
                        }
                        ui.menu_button("Open Recent", |ui| {
                            ui.set_min_width(230.0);
                            if self.recent_files.is_empty() {
                                ui.add_enabled(false, egui::Button::new("No recent documents"));
                            } else {
                                for path in self.recent_files.clone() {
                                    let label = path
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("Recent document");
                                    if ui
                                        .button(label)
                                        .on_hover_text(path.display().to_string())
                                        .clicked()
                                    {
                                        ui.close();
                                        self.open(path);
                                    }
                                }
                            }
                            ui.separator();
                            if ui
                                .add_enabled(
                                    !self.recent_files.is_empty(),
                                    egui::Button::new("Clear Recent"),
                                )
                                .clicked()
                            {
                                ui.close();
                                self.clear_recent_documents();
                            }
                        });
                        ui.separator();
                        let has_document = self.session.is_some();
                        if ui.add_enabled(has_document, egui::Button::new("Save").shortcut_text("⌘S")).clicked() {
                            ui.close();
                            self.save_document(false);
                        }
                        if ui.add_enabled(has_document, egui::Button::new("Save As…").shortcut_text("⇧⌘S")).clicked() {
                            ui.close();
                            self.save_document(true);
                        }
                        if ui.add_enabled(has_document, egui::Button::new("Export STEP…")).clicked() {
                            ui.close();
                            self.export_document();
                        }
                        ui.separator();
                        if ui
                            .add_enabled(!self.tabs.is_empty(), egui::Button::new("Close Tab"))
                            .clicked()
                        {
                            ui.close();
                            let index = self.active_tab;
                            self.close_tab(index);
                        }
                        ui.separator();
                        if ui.button("Quit").clicked() {
                            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    let chevron = egui::Pos2::new(
                        file_button.rect.right() - 2.0,
                        file_button.rect.center().y + 4.0,
                    );
                    let chevron_colour = theme.color(if file_button.hovered() {
                        theme.icon_hover
                    } else {
                        theme.icon
                    });
                    let stroke = egui::Stroke::new(1.4, chevron_colour);
                    ui.painter().line_segment(
                        [chevron + egui::vec2(-4.0, -2.0), chevron + egui::vec2(0.0, 2.0)],
                        stroke,
                    );
                    ui.painter().line_segment(
                        [chevron + egui::vec2(0.0, 2.0), chevron + egui::vec2(4.0, -2.0)],
                        stroke,
                    );
                    let can_undo = self.session.as_ref().is_some_and(Session::has_undo);
                    let can_redo = self.session.as_ref().is_some_and(Session::has_redo);
                    let spacing = ui.spacing().item_spacing.x;
                    // The File chevron sits 2 px inside the menu's right edge;
                    // 4 px here plus the 8 px arrow spacing puts the arrow 10 px
                    // from it.
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.add_space(4.0);
                    ui.spacing_mut().item_spacing.x = 8.0;
                    let undo = ui
                        .add_enabled(
                            can_undo,
                            egui::Button::new(egui::RichText::new("←").size(16.0)).frame(false),
                        )
                        .on_hover_text("Undo (⌘Z)");
                    if undo.clicked() {
                        self.undo();
                    }
                    let redo = ui
                        .add_enabled(
                            can_redo,
                            egui::Button::new(egui::RichText::new("→").size(16.0)).frame(false),
                        )
                        .on_hover_text("Redo (⇧⌘Z)");
                    if redo.clicked() {
                        self.redo();
                    }
                    ui.spacing_mut().item_spacing.x = spacing;
                    ui.add_space(14.0);

                    // Document tabs. The active tab is the session currently
                    // displayed by the viewport, inspector and node canvas.
                    for index in 0..self.tabs.len() {
                        let title = self.tabs[index]
                            .path
                            .as_ref()
                            .and_then(|path| path.file_stem().and_then(|name| name.to_str()))
                            .unwrap_or("Untitled")
                            .to_owned();
                        let active = index == self.active_tab;
                        let galley = ui.painter().layout_no_wrap(
                            title,
                            egui::FontId::proportional(13.0),
                            theme.color(if active { theme.text } else { theme.muted }),
                        );
                        let tab_fill = theme.color(if active { theme.control } else { theme.window });
                        let (tab_rect, response) = ui.allocate_exact_size(
                            egui::vec2(galley.size().x + 42.0, 30.0),
                            egui::Sense::click(),
                        );
                        ui.painter().add(ui_shape::shape(
                            tab_rect,
                            tab_fill,
                            egui::Stroke::NONE,
                        ));
                        ui.painter().galley(
                            tab_rect.center() - galley.size() * 0.5,
                            galley,
                            theme.color(if active { theme.text } else { theme.muted }),
                        );
                        let close_rect = egui::Rect::from_center_size(
                            egui::Pos2::new(tab_rect.right() - 14.0, tab_rect.center().y),
                            egui::Vec2::splat(20.0),
                        );
                        let close = ui.interact(
                            close_rect,
                            egui::Id::new(("close-document-tab", &self.tabs[index].id)),
                            egui::Sense::click(),
                        );
                        if close.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if response.hovered() || close.hovered() {
                            let gradient = egui::Rect::from_min_max(
                                egui::Pos2::new(tab_rect.right() - 34.0, tab_rect.min.y + 2.0),
                                egui::Pos2::new(tab_rect.right() - 2.0, tab_rect.max.y - 2.0),
                            );
                            const STEPS: usize = 14;
                            for step in 0..STEPS {
                                let t = step as f32 / (STEPS - 1) as f32;
                                let x0 = gradient.left() + gradient.width() * t
                                    - gradient.width() / STEPS as f32;
                                let x1 = gradient.left() + gradient.width() * (step + 1) as f32
                                    / STEPS as f32;
                                // Reach full opacity before the close hit area.
                                let alpha = ((t / 0.55).min(1.0) * 255.0) as u8;
                                ui.painter().rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::Pos2::new(x0.max(gradient.left()), gradient.top()),
                                        egui::Pos2::new(x1, gradient.bottom()),
                                    ),
                                    egui::CornerRadius::ZERO,
                                    egui::Color32::from_rgba_unmultiplied(
                                        tab_fill.r(),
                                        tab_fill.g(),
                                        tab_fill.b(),
                                        alpha,
                                    ),
                                );
                            }
                            let cross_colour = if close.hovered() {
                                theme.color(theme.text)
                            } else {
                                theme.color(theme.muted)
                            };
                            let centre = close_rect.center();
                            let arm = 4.0;
                            let stroke = egui::Stroke::new(1.4, cross_colour);
                            ui.painter().line_segment(
                                [centre + egui::vec2(-arm, -arm), centre + egui::vec2(arm, arm)],
                                stroke,
                            );
                            ui.painter().line_segment(
                                [centre + egui::vec2(arm, -arm), centre + egui::vec2(-arm, arm)],
                                stroke,
                            );
                        }
                        if close.clicked() {
                            requested_close = Some(index);
                        } else if response.clicked() {
                            requested_tab = Some(index);
                        }
                        ui.add_space(4.0);
                    }
                    let new_tab = ui
                        .add(
                            egui::Button::new(egui::RichText::new("+").size(17.0))
                                .frame(false)
                                .min_size(egui::vec2(30.0, 30.0)),
                        )
                        .on_hover_text("New blank document");
                    if new_tab.clicked() {
                        self.new_blank();
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let export = ui_shape::button(
                            ui,
                            "Export",
                            78.0,
                            28.0,
                            theme.color(theme.control),
                            egui::Stroke::new(1.0, theme.with_alpha(theme.border_strong, 210)),
                            theme.color(theme.text),
                        )
                        .on_hover_text("Export through the connected Export STEP node");
                        if export.clicked() {
                            self.export_document();
                        }
                        ui.add_space(6.0);
                        let (runner_rect, runner) = ui.allocate_exact_size(
                            egui::Vec2::splat(28.0),
                            egui::Sense::click(),
                        );
                        let tint = crate::theme::Theme::hex(if runner.hovered() {
                            theme.icon_hover
                        } else {
                            theme.icon
                        });
                        let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/arrow_clockwise.svg"))
                            .unwrap_or_default()
                            .replace("#212121", &tint);
                        let image = egui::Image::from_bytes(
                            "bytes://nodecraft-reevaluate.svg",
                            svg.into_bytes(),
                        )
                        .fit_to_exact_size(egui::Vec2::splat(16.0));
                        ui.put(runner_rect.shrink(6.0), image);
                        if runner.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        let runner = runner.on_hover_text("Re-evaluate graph without running exporters");
                        if runner.clicked() {
                            self.reevaluate_non_export();
                        }
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(format!("GIT: {BUILD_GIT_HASH}"))
                                .size(11.0)
                                .color(theme.with_alpha(theme.muted, 220)),
                        );
                        ui.add_space(6.0);
                        let (settings_rect, settings) = ui.allocate_exact_size(
                            egui::Vec2::splat(28.0),
                            egui::Sense::click(),
                        );
                        let tint = crate::theme::Theme::hex(if settings.hovered() {
                            theme.icon_hover
                        } else {
                            theme.icon
                        });
                        let svg = std::str::from_utf8(include_bytes!(
                            "../../../../vendor/fluent/settings.svg"
                        ))
                        .unwrap_or_default()
                        .replace("#d8d8dc", &tint);
                        let image = egui::Image::from_bytes(
                            "bytes://nodecraft-topbar-settings.svg",
                            svg.into_bytes(),
                        )
                        .fit_to_exact_size(egui::Vec2::splat(16.0));
                        ui.put(settings_rect.shrink(6.0), image);
                        if settings.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        let settings = settings.on_hover_text(if self.settings_open {
                            "Close settings"
                        } else {
                            "Open settings"
                        });
                        if settings.clicked() {
                            self.settings_open = !self.settings_open;
                            if !self.settings_open {
                                self.settings_open_last = false;
                            }
                        }
                    });
                });
            });

        if let Some(index) = requested_close {
            self.close_tab(index);
        } else if let Some(index) = requested_tab {
            self.activate_tab(index);
        }

        if self.settings_open {
            self.show_app_settings(ui.ctx());
        }
        if self.theme_dirty {
            let now = Instant::now();
            if self.theme_save_due.is_none_or(|due| now >= due) {
                if let Err(error) = crate::theme::save(&crate::theme::get()) {
                    eprintln!("theme: cannot save live edit: {error}");
                }
                self.theme_dirty = false;
                self.theme_save_due = None;
            } else if let Some(due) = self.theme_save_due {
                ui.ctx()
                    .request_repaint_after(due.saturating_duration_since(now));
            }
        }

        egui::Panel::bottom("app-status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                if let Some(error) = &self.error {
                    ui.separator();
                    let error_color = ui.visuals().error_fg_color;
                    ui.colored_label(error_color, error);
                }
            });
        });

        let selected_node = if self.selected_nodes.len() == 1 {
            self.session.as_ref().and_then(|session| {
                let id = self.selected_nodes.iter().next()?.as_str();
                session.document().nodes.iter().find(|node| node.id == id).cloned()
            })
        } else {
            None
        };
        let surface_sections: Vec<String> = selected_node
            .as_ref()
            .filter(|node| node.node_type == "surfaceloft")
            .and_then(|node| {
                self.session.as_ref().map(|session| {
                    session
                        .document()
                        .connections
                        .iter()
                        .filter(|link| link.to == node.id && link.input == "sections")
                        .map(|link| link.from.clone())
                        .collect()
                })
            })
            .unwrap_or_default();
        self.viewport.set_transform_gizmo(self.selected_transform_gizmo());
        let sketch_node = selected_node
            .as_ref()
            .filter(|node| node.node_type == "sketch")
            .cloned();
        if let Some(node) = &sketch_node {
            self.sketch_gui.sync(&node.id, &node.values);
        }
        let mut inspector_output: Option<inspector::Output> = None;
        let inspector_title = selected_node
            .as_ref()
            .map(inspector::node_title)
            .unwrap_or_else(|| {
                if self.selected_nodes.len() > 1 {
                    format!("{} nodes selected", self.selected_nodes.len())
                } else if self.session.is_some() {
                    "No selection".to_owned()
                } else {
                    "No document".to_owned()
                }
            });
        let inspector_badge = selected_node
            .as_ref()
            .map(inspector::node_badge)
            .filter(|badge| {
                !inspector_title
                    .to_lowercase()
                    .contains(&badge.to_lowercase())
            })
            .unwrap_or_default();
        let workspace_rect = ui.available_rect_before_wrap();
        let inspector_margin = 12.0;
        let max_inspector_width = (workspace_rect.width() * 0.45).clamp(240.0, 480.0);
        let inspector_width = self.inspector_width.clamp(220.0, max_inspector_width);
        let inspector_open_height = (workspace_rect.height() * 0.5)
            .max(340.0)
            .min((workspace_rect.height() - 2.0 * inspector_margin).max(240.0));
        let inspector_closed_height = 68.0;
        let inspector_height = inspector_closed_height
            + (inspector_open_height - inspector_closed_height) * self.inspector_reveal;
        let inspector_rect = egui::Rect::from_min_max(
            egui::Pos2::new(
                workspace_rect.right() - inspector_width - inspector_margin,
                workspace_rect.bottom() - inspector_margin - inspector_height,
            ),
            egui::Pos2::new(
                workspace_rect.right() - inspector_margin,
                workspace_rect.bottom() - inspector_margin,
            ),
        );
        let node_panel_rect = self.node_editor.panel_rect(workspace_rect);
        let tool_dock_rect = tool_dock_rect(workspace_rect);
        let backdrop_rect = node_panel_rect
            .union(inspector_rect)
            .union(tool_dock_rect);
        let node_backdrop = self.viewport.backdrop_view_for(node_panel_rect, backdrop_rect);
        let inspector_backdrop = self
            .viewport
            .backdrop_view_for(inspector_rect, backdrop_rect);
        let tool_dock_backdrop = self
            .viewport
            .backdrop_view_for(tool_dock_rect, backdrop_rect);
        // The inspector now sits in the bottom-right corner, so the axis chrome
        // can reclaim the top-right corner of the window.
        self.viewport.chrome_right_inset = 0.0;

        let mut editor_actions = Vec::new();
        let mut viewport_rect = workspace_rect;
        // No central-panel frame/margin: the wgpu viewport paint callback must start
        // at the available workspace edge.
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let central_rect = ui.available_rect_before_wrap();
            viewport_rect = central_rect;
            let panel_rect = self.node_editor.panel_rect(central_rect);
            self.viewport.show_with_backdrop_blockers(
                ui,
                panel_rect,
                &[panel_rect, inspector_rect],
                backdrop_rect,
            );
            if let Some(session) = &self.session {
                let animated = self.node_position_animation.as_ref().map(|animation| {
                    let duration = Duration::from_millis(450).as_secs_f32();
                    let progress = (animation.started.elapsed().as_secs_f32() / duration)
                        .clamp(0.0, 1.0);
                    animation.document_at(session.document(), progress)
                });
                let document = animated.as_ref().unwrap_or_else(|| session.document());
                editor_actions = self.node_editor.show(
                    ui.ctx(),
                    central_rect,
                    document,
                    &self.selected_nodes,
                    &self.selected_connections,
                    self.viewer_breakpoint.as_deref(),
                    node_backdrop,
                );
            }
        });
        let gizmo_changes = self.viewport.take_gizmo_changes();
        if let Some(node) = &selected_node {
            if node.node_type == "translate" && !gizmo_changes.is_empty() {
                let mut changed = false;
                for (field, value) in gizmo_changes {
                    let command = Command::SetValue {
                        node: node.id.clone(),
                        field,
                        value: serde_json::Value::from(value),
                    };
                    changed |= self.apply_commands(
                        &[command],
                        Some(&format!("gizmo:{}", node.id)),
                    );
                }
                if changed {
                    self.preserve_view_for_next_mesh = true;
                    self.evaluate();
                }
            }
        }

        let theme = crate::theme::get();
        let inspector_info = {
            let mut info = format!(
                "Evaluation: {:.1} ms\nCache: {} hits / {} misses",
                self.evaluation_ms, self.cache_hits, self.cache_misses
            );
            if let Some(mesh) = &self.mesh {
                info.push_str(&format!(
                    "\nMesh: {} verts / {} tris / {} edges",
                    mesh.vertices.len(),
                    mesh.faces.len(),
                    mesh.brep_edges.len()
                ));
            }
            info
        };
        let mut toggle_inspector = false;
        egui::Area::new(egui::Id::new("inspector-panel"))
            .order(egui::Order::Foreground)
            .fixed_pos(inspector_rect.min)
            .pivot(egui::Align2::LEFT_TOP)
            .constrain(false)
            .interactable(true)
            .show(ui.ctx(), |ui| {
                let panel_rect =
                    egui::Rect::from_min_size(inspector_rect.min, inspector_rect.size());
                // An Area can retain its previous content height across animation
                // frames. Pin the interaction/content area to the animated panel so
                // reopening reflows the list instead of keeping the minimized height.
                ui.set_height(panel_rect.height());
                ui.set_width(panel_rect.width());
                if let Some((texture, uv)) = inspector_backdrop {
                    ui_shape::image(&ui.painter(), texture, panel_rect, uv);
                } else {
                    ui_shape::fill(
                        &ui.painter(),
                        panel_rect,
                        theme.color(theme.blur_tint),
                    );
                }
                ui_shape::outline(
                    &ui.painter(),
                    panel_rect,
                    egui::Stroke::new(1.0, theme.with_alpha(theme.border, 180)),
                );
                let resize_rect = egui::Rect::from_min_max(
                    egui::Pos2::new(panel_rect.left() - 4.0, panel_rect.top() + 12.0),
                    egui::Pos2::new(panel_rect.left() + 4.0, panel_rect.bottom() - 12.0),
                );
                let resize = ui.interact(
                    resize_rect,
                    egui::Id::new("inspector-resize"),
                    egui::Sense::drag(),
                );
                if resize.hovered() || resize.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
                if resize.dragged() {
                    self.inspector_width = (self.inspector_width - resize.drag_delta().x)
                        .clamp(220.0, max_inspector_width);
                }
                let grip_colour = if resize.hovered() || resize.dragged() {
                    theme.color(theme.highlight)
                } else {
                    theme.with_alpha(theme.muted, 70)
                };
                ui.painter().rect_filled(
                    egui::Rect::from_min_max(
                        egui::Pos2::new(panel_rect.left() - 1.0, panel_rect.center().y - 26.0),
                        egui::Pos2::new(panel_rect.left() + 1.0, panel_rect.center().y + 26.0),
                    ),
                    egui::CornerRadius::same(1),
                    grip_colour,
                );
                // Keep collapsing body content inside the glass bounds without
                // clipping the first glyph of labels at the left inset.
                ui.set_clip_rect(panel_rect.shrink(2.0));
                let (top_margin, bottom_margin) = if self.inspector_reveal < 0.5 {
                    // The collapsed pill carries both the INSPECTOR kicker and the
                    // node name; give the two lines breathing room at the bottom.
                    (12, 12)
                } else {
                    (16, 16)
                };
                egui::Frame::new()
                    .inner_margin(egui::Margin {
                        left: 16,
                        right: 16,
                        top: top_margin,
                        bottom: bottom_margin,
                    })
                    .show(ui, |ui| {
                        ui.set_width((panel_rect.width() - 32.0).max(100.0));
                        // The title and disclosure control stay fully opaque; only the
                        // body below fades with the collapse animation.
                        ui.set_opacity(1.0);
                        let header_top = ui.cursor().min.y;
                        ui.label(
                            egui::RichText::new("INSPECTOR")
                                .size(9.0)
                                .strong()
                                .color(theme.with_alpha(theme.muted, 220)),
                        );
                        ui.horizontal(|ui| {
                            ui.heading(inspector_title.clone());
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let (chevron_rect, chevron) = ui.allocate_exact_size(
                                        egui::Vec2::splat(26.0),
                                        egui::Sense::click(),
                                    );
                                    let chevron = chevron.on_hover_text(&inspector_info);
                                    if chevron.hovered() || chevron.dragged() {
                                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                    }
                                    if chevron.hovered() {
                                        ui.painter().circle_filled(
                                            chevron_rect.center(),
                                            13.0,
                                            theme.with_alpha(theme.highlight, 26),
                                        );
                                    }
                                    let colour = theme.color(if chevron.hovered() {
                                        theme.text
                                    } else {
                                        theme.muted
                                    });
                                    let angle = if self.inspector_open {
                                        std::f32::consts::FRAC_PI_2
                                    } else {
                                        0.0
                                    };
                                    let rotate = |v: egui::Vec2| {
                                        egui::vec2(
                                            v.x * angle.cos() - v.y * angle.sin(),
                                            v.x * angle.sin() + v.y * angle.cos(),
                                        )
                                    };
                                    let centre = chevron_rect.center();
                                    let tip = centre + rotate(egui::vec2(3.0, 0.0));
                                    let top = centre + rotate(egui::vec2(-3.0, -4.0));
                                    let bottom = centre + rotate(egui::vec2(-3.0, 4.0));
                                    let stroke = egui::Stroke::new(1.5, colour);
                                    ui.painter().line_segment([top, tip], stroke);
                                    ui.painter().line_segment([tip, bottom], stroke);
                                    if chevron.clicked() {
                                        toggle_inspector = true;
                                    }
                                    if !inspector_badge.is_empty() {
                                        ui.label(
                                            egui::RichText::new(&inspector_badge)
                                                .size(11.0)
                                                .color(theme.with_alpha(theme.muted, 220)),
                                        );
                                    }
                                },
                            );
                        });
                        // Double-clicking anywhere in the header toggles the
                        // inspector, including over the disclosure chevron.
                        let header_bottom = ui.cursor().min.y;
                        let header_rect = egui::Rect::from_min_max(
                            egui::pos2(panel_rect.left(), header_top),
                            egui::pos2(panel_rect.right(), header_bottom),
                        );
                        let header_double_clicked = ui.ctx().input(|input| {
                            if !input.pointer.button_double_clicked(egui::PointerButton::Primary) {
                                return false;
                            }
                            let Some(pointer) = input
                                .pointer
                                .interact_pos()
                                .or_else(|| input.pointer.latest_pos())
                            else {
                                return false;
                            };
                            header_rect.contains(pointer)
                        });
                        if header_double_clicked {
                            toggle_inspector = true;
                        }
                        if self.inspector_reveal > 0.001 {
                            ui.scope(|ui| {
                                ui.set_opacity(self.inspector_reveal.clamp(0.0, 1.0));
                        ui.separator();
                        if self.selected_nodes.len() > 1 {
                            ui.label(format!("{} nodes selected", self.selected_nodes.len()));
                            ui.separator();
                        }
                        if let Some(node) = &selected_node {
                            // Derive the list height from the animated panel rectangle,
                            // not the UI's cached available height. An Area being reopened
                            // can keep a stale content height for a frame and otherwise
                            // leaves the list stuck at its minimized size.
                            let list_height = (inspector_rect.height() - 112.0).max(80.0);
                            let scroll_bar_rect = egui::Rect::from_min_max(
                                egui::Pos2::new(panel_rect.right() - 11.0, ui.cursor().top()),
                                egui::Pos2::new(panel_rect.right() - 1.0, panel_rect.bottom() - 6.0),
                            );
                            ui.scope(|ui| {
                            ui.style_mut().spacing.scroll.floating = true;
                            ui.style_mut().spacing.scroll.foreground_color = false;
                            ui.style_mut().spacing.scroll.bar_width = 5.0;
                            ui.style_mut().spacing.scroll.floating_width = 3.0;
                            ui.style_mut().spacing.scroll.floating_allocated_width = 0.0;
                            ui.style_mut().spacing.scroll.bar_outer_margin = 0.5;
                            ui.style_mut().spacing.scroll.dormant_handle_opacity = 1.0;
                            ui.style_mut().spacing.scroll.active_handle_opacity = 1.0;
                            ui.style_mut().spacing.scroll.interact_handle_opacity = 1.0;
                            let bar_corner = egui::CornerRadius::same(2);
                            let bar_inactive = theme.with_alpha(theme.muted, 110);
                            let bar_hovered = theme.with_alpha(theme.muted, 205);
                            let bar_active = theme.with_alpha(theme.muted, 230);
                            ui.style_mut().visuals.widgets.inactive.bg_fill = bar_inactive;
                            ui.style_mut().visuals.widgets.hovered.bg_fill = bar_hovered;
                            ui.style_mut().visuals.widgets.active.bg_fill = bar_active;
                            ui.style_mut().visuals.widgets.inactive.corner_radius = bar_corner;
                            ui.style_mut().visuals.widgets.hovered.corner_radius = bar_corner;
                            ui.style_mut().visuals.widgets.active.corner_radius = bar_corner;
                            let reflow = self.inspector_reflow;
                            let mut scroll = egui::ScrollArea::vertical()
                                .max_height(list_height)
                                .auto_shrink([false, false])
                                .scroll_bar_visibility(
                                    egui::containers::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                                )
                                .scroll_bar_rect(scroll_bar_rect);
                            if reflow {
                                scroll = scroll.vertical_scroll_offset(0.0);
                            }
                            scroll.show(ui, |ui| {
                                    let mut output = inspector::show(ui, node, &surface_sections);
                                    inspector::note(ui, node);
                                    ui.add_space(8.0);
                                    inspector::actions(ui, node, &mut output);
                                    inspector_output = Some(output);
                                });
                            if reflow {
                                self.inspector_reflow = false;
                            }
                            });
                        } else if self.selected_nodes.is_empty() {
                            let message = if self.session.is_some() {
                                "No node selected. Select a node to inspect its values."
                            } else {
                                "No document loaded. Open a NodeCraft graph to begin."
                            };
                            ui.label(egui::RichText::new(message).color(theme.color(theme.muted)));
                        }
                            });
                        }
                        ui.set_opacity(1.0);
                    });
            });
        if toggle_inspector {
            self.toggle_inspector();
        }
        if let (Some(output), Some(node)) = (inspector_output, &selected_node) {
            if output.delete {
                let mut nodes = HashSet::new();
                nodes.insert(node.id.clone());
                let commands = delete_selection_commands(&nodes, &HashSet::new());
                self.selected_nodes.clear();
                self.selected_connections.clear();
                if self.viewer_breakpoint.as_deref() == Some(node.id.as_str()) {
                    self.viewer_breakpoint = None;
                }
                if self.apply_commands(&commands, Some("inspector-delete")) {
                    self.evaluate();
                }
            } else {
                if let Some(order) = output.section_order.clone() {
                    let mut commands = Vec::new();
                    if let Some(session) = &self.session {
                        for link in session
                            .document()
                            .connections
                            .iter()
                            .filter(|link| link.to == node.id && link.input == "sections")
                        {
                            commands.push(Command::Disconnect {
                                from: link.from.clone(),
                                to: link.to.clone(),
                                input: link.input.clone(),
                            });
                        }
                    }
                    for from in order {
                        commands.push(Command::Connect {
                            from,
                            to: node.id.clone(),
                            input: "sections".to_owned(),
                            out: "shape".to_owned(),
                        });
                    }
                    if self.apply_commands(&commands, Some("surface-loft-sections")) {
                        self.evaluate();
                    }
                }
                // Apply the whole frame's inspector edits before asking the engine
                // to evaluate the resulting document. Each field keeps its own
                // undo coalescing key, but the worker sees one consistent update.
                let mut changed = false;
                for change in output.changes {
                    let coalesce = format!("value:{}:{}", node.id, change.field);
                    let command = Command::SetValue {
                        node: node.id.clone(),
                        field: change.field,
                        value: change.value,
                    };
                    changed |= self.apply_commands(&[command], Some(&coalesce));
                }
                if output.choose_file {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("STEP model", &["step", "stp"])
                        .pick_file()
                    {
                        let command = Command::SetValue {
                            node: node.id.clone(),
                            field: "path".to_owned(),
                            value: serde_json::Value::String(path.display().to_string()),
                        };
                        changed |= self.apply_commands(&[command], Some("step-path"));
                    }
                }
                if output.choose_directory {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        let command = Command::SetValue {
                            node: node.id.clone(),
                            field: "directory".to_owned(),
                            value: serde_json::Value::String(path.display().to_string()),
                        };
                        changed |= self.apply_commands(&[command], Some("step-directory"));
                    }
                }
                if output.toggle_suppressed {
                    let command = Command::SetSuppressed {
                        node: node.id.clone(),
                        suppressed: !node.suppressed,
                    };
                    changed |= self.apply_commands(&[command], Some("suppress-node"));
                }
                if changed {
                    self.evaluate();
                }
            }
        }

        for action in editor_actions {
            match action {
                NodeEditorAction::SelectNode { id, additive } => {
                    // Selection is independent of the viewer breakpoint. Only the
                    // eye control changes what the engine/viewport evaluates.
                    if !additive {
                        self.selected_nodes.clear();
                        self.selected_nodes.insert(id);
                    } else if !self.selected_nodes.remove(&id) {
                        self.selected_nodes.insert(id);
                    }
                    self.selected_connections.clear();
                }
                NodeEditorAction::SetLassoSelection { nodes, .. } => {
                    self.selected_nodes = nodes;
                    self.selected_connections.clear();
                }
                NodeEditorAction::SelectCable { key, additive } => {
                    self.selected_nodes.clear();
                    if !additive {
                        self.selected_connections.clear();
                    }
                    if additive && self.selected_connections.contains(&key) {
                        self.selected_connections.remove(&key);
                    } else {
                        self.selected_connections.insert(key);
                    }
                }
                NodeEditorAction::Connect { from, to, input, out } => {
                    let command = Command::Connect { from, to, input, out };
                    let changed = self.apply_commands(&[command], Some("connect-nodes"));
                    if changed {
                        self.status = "Connected nodes".to_owned();
                    } else if self.error.is_none() {
                        // A duplicate connection is a session no-op, but the UI
                        // still asked to view the connected target. Refresh so a
                        // stale/empty viewport cannot survive the reconnect.
                        self.status = "Connection unchanged".to_owned();
                        self.evaluate();
                        continue;
                    }
                    self.evaluate();
                }
                NodeEditorAction::InsertNodeOnCable { node, connection, input, out } => {
                    if let Some(session) = &self.session {
                        let document = session.document();
                        let descendants = downstream_nodes(document, &connection.to);
                        let mut commands = vec![Command::Disconnect {
                            from: connection.from.clone(),
                            to: connection.to.clone(),
                            input: connection.input.clone(),
                        }];
                        commands.push(Command::Connect {
                            from: connection.from.clone(),
                            to: node.clone(),
                            input,
                            out: connection.out.clone(),
                        });
                        commands.push(Command::Connect {
                            from: node.clone(),
                            to: connection.to.clone(),
                            input: connection.input.clone(),
                            out,
                        });
                        // Move the target and its whole downstream graph down by a
                        // full card plus padding so the inserted node and its cables
                        // have room.
                        let shift = f64::from(NODE_HEIGHT) + 80.0;
                        let positions: Vec<(String, f64, f64)> = document
                            .nodes
                            .iter()
                            .filter(|candidate| candidate.id != node && descendants.contains(&candidate.id))
                            .map(|candidate| (candidate.id.clone(), candidate.x, candidate.y + shift))
                            .collect();
                        commands.extend(positions.into_iter().map(|(node, x, y)| Command::SetPosition { node, x, y }));
                        if self.apply_commands(&commands, Some("insert-node-on-cable")) {
                            self.evaluate();
                            self.status = "Inserted node into cable".to_owned();
                        }
                    }
                }
                NodeEditorAction::SetViewerBreakpoint { id } => {
                    // The eye controls the viewer only; it never mutates selection.
                    self.viewer_breakpoint = Some(id);
                    self.evaluate();
                }
                NodeEditorAction::CleanupLayout { mode } => {
                    let mut planned: Option<Vec<(String, f64, f64)>> = None;
                    if let Some(session) = &self.session {
                        let document = session.document();
                        let nodes: Vec<serde_json::Value> = document
                            .nodes
                            .iter()
                            .map(|node| {
                                serde_json::json!({
                                    "id": node.id,
                                    "width": node_width(node) + NODE_LEFT_PADDING + NODE_RIGHT_PADDING + 24.0,
                                    "height": NODE_HEIGHT + 24.0,
                                })
                            })
                            .collect();
                        let connections: Vec<serde_json::Value> = document
                            .connections
                            .iter()
                            .map(|link| serde_json::json!({ "from": link.from, "to": link.to }))
                            .collect();
                        let mode = match mode {
                            LayoutMode::Timeline => "timeline",
                            LayoutMode::Centered => "centered",
                            LayoutMode::Organic => "organic",
                        };
                        if let Ok(result) = nodecraft_worker::layout::compute(mode, &nodes, &connections) {
                            if let Some(positions) = result.get("positions").and_then(serde_json::Value::as_array) {
                                planned = Some(positions
                                    .iter()
                                    .filter_map(|position| {
                                        Some((
                                            position.get("id")?.as_str()?.to_owned(),
                                            position.get("x")?.as_f64()?,
                                            position.get("y")?.as_f64()?,
                                        ))
                                    })
                                    .collect());
                            }
                        }
                    }
                    if let (Some(positions), Some(session)) = (planned, &self.session) {
                        let document = session.document();
                        let start: HashMap<String, (f64, f64)> = document
                            .nodes
                            .iter()
                            .map(|node| (node.id.clone(), (node.x, node.y)))
                            .collect();
                        let end: HashMap<String, (f64, f64)> = positions
                            .into_iter()
                            .map(|(id, x, y)| (id, (x, y)))
                            .collect();
                        if let Some(roi) = layout_roi(document, &end) {
                            self.node_position_animation = None;
                            self.node_editor.animate_to_roi(roi, LAYOUT_ANIMATION);
                            self.layout_animation = Some(LayoutAnimation {
                                start,
                                end,
                                started: Instant::now(),
                            });
                        }
                    }
                }
                NodeEditorAction::MoveNodes { positions } => {
                    let commands: Vec<Command> = positions
                        .into_iter()
                        .map(|(node, x, y)| Command::SetPosition { node, x, y })
                        .collect();
                    self.apply_commands(&commands, Some("move:nodes"));
                }
                NodeEditorAction::Deselect => {
                    self.selected_nodes.clear();
                    self.selected_connections.clear();
                }
                NodeEditorAction::SetPositions(positions) => {
                    let commands: Vec<Command> = positions
                        .into_iter()
                        .map(|(node, x, y)| Command::SetPosition { node, x, y })
                        .collect();
                    if self.apply_commands(&commands, Some("cleanup-layout")) {
                        self.node_editor.request_fit();
                    }
                }
            }
        }
        if sketch_node.is_some() {
            let viewport = egui::Rect::from_min_max(
                egui::Pos2::new(node_panel_rect.right() + 12.0, workspace_rect.top() + 12.0),
                egui::Pos2::new(workspace_rect.right() - 12.0, workspace_rect.bottom() - 12.0),
            );
            if let Some(value) = self.sketch_gui.show(ui.ctx(), viewport_rect, viewport, &self.viewport.camera) {
                if let Some(node) = &sketch_node {
                    if self.apply_commands(
                        &[Command::SetValue {
                            node: node.id.clone(),
                            field: "sketch".to_owned(),
                            value,
                        }],
                        Some("sketch-edit"),
                    ) {
                        self.evaluate();
                    }
                }
            }
        }
        self.show_tool_dock(ui.ctx(), workspace_rect, tool_dock_backdrop);
    }
}

struct EvalJob {
    id: u64,
    tab_id: u64,
    linked_documents: std::collections::BTreeMap<PathBuf, GraphDocument>,
    document: GraphDocument,
    selected: Option<String>,
    clear_cache: bool,
}

struct EvalWorker {
    tx: Sender<EvalJob>,
    rx: Receiver<(u64, u64, engine_bridge::Evaluation)>,
    progress_rx: Receiver<(u64, u64, nodecraft_engine::ProgressEvent)>,
}

impl EvalWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = channel::<EvalJob>();
        let (result_tx, result_rx) = channel();
        let (progress_tx, progress_rx) = channel();
        std::thread::spawn(move || {
            let mut cache = EngineCache::new();
            while let Ok(mut job) = job_rx.recv() {
                // Only the newest request matters; discard queued stale work.
                while let Ok(next) = job_rx.try_recv() {
                    job = next;
                }
                if job.clear_cache {
                    // Re-evaluate is an explicit request for a fresh result: do
                    // not let mesh/BRep/failure/import entries answer it.
                    cache = EngineCache::new();
                }
                let target = engine_bridge::target_for(&job.document, job.selected.as_deref());
                let progress_tx = progress_tx.clone();
                let mut on_progress = |event: nodecraft_engine::ProgressEvent| {
                    let _ = progress_tx.send((job.id, job.tab_id, event));
                };
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    engine_bridge::evaluate_with_linked_progress(
                        &job.document,
                        cache,
                        job.selected.as_deref(),
                        &job.linked_documents,
                        Some(&mut on_progress),
                    )
                }));
                let evaluation = match outcome {
                    Ok((evaluation, next_cache)) => {
                        cache = next_cache;
                        evaluation
                    }
                    Err(_) => {
                        // A kernel panic must not silently kill the only evaluation
                        // thread. Drop the poisoned cache and let later edits retry.
                        cache = EngineCache::new();
                        target.map(|target| engine_bridge::Evaluation {
                            target,
                            mesh: None,
                            error: Some("Engine panicked while evaluating this target".into()),
                            elapsed_ms: 0.0,
                            cache_hits: 0,
                            cache_misses: 0,
                        })
                    }
                };
                if let Some(evaluation) = evaluation {
                    if result_tx.send((job.id, job.tab_id, evaluation)).is_err() {
                        break;
                    }
                }
            }
        });
        Self { tx: job_tx, rx: result_rx, progress_rx }
    }

    fn send(&self, job: EvalJob) {
        let _ = self.tx.send(job);
    }

    fn try_recv(&self) -> Option<(u64, u64, engine_bridge::Evaluation)> {
        self.rx.try_recv().ok()
    }

    fn try_recv_progress(&self) -> Option<(u64, u64, nodecraft_engine::ProgressEvent)> {
        self.progress_rx.try_recv().ok()
    }
}

fn default_document() -> Option<PathBuf> {
    let candidate = PathBuf::from("samples/step-import.nodecraft");
    candidate.is_file().then_some(candidate)
}

fn choose_startup_document(
    explicit: Option<PathBuf>,
    last: Option<PathBuf>,
    fallback: Option<PathBuf>,
) -> Option<PathBuf> {
    explicit.or(last).or(fallback)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConfigPlatform {
    MacOs,
    Windows,
    Unix,
}

/// The platform config root, as a pure function so path behavior can be tested
/// for every target without running on it.
fn config_dir_for(
    platform: ConfigPlatform,
    home: Option<&std::ffi::OsStr>,
    appdata: Option<&std::ffi::OsStr>,
    userprofile: Option<&std::ffi::OsStr>,
    xdg_config_home: Option<&std::ffi::OsStr>,
) -> Option<PathBuf> {
    match platform {
        ConfigPlatform::MacOs => home
            .map(|home| PathBuf::from(home).join("Library/Application Support/NodeCraft")),
        ConfigPlatform::Windows => appdata
            .map(|path| PathBuf::from(path).join("NodeCraft"))
            .or_else(|| {
                userprofile.map(|path| {
                    PathBuf::from(path)
                        .join("AppData")
                        .join("Roaming")
                        .join("NodeCraft")
                })
            }),
        ConfigPlatform::Unix => xdg_config_home
            .map(|path| PathBuf::from(path).join("nodecraft"))
            .or_else(|| home.map(|home| PathBuf::from(home).join(".config/nodecraft"))),
    }
}

fn current_config_platform() -> ConfigPlatform {
    if cfg!(target_os = "macos") {
        ConfigPlatform::MacOs
    } else if cfg!(windows) {
        ConfigPlatform::Windows
    } else {
        ConfigPlatform::Unix
    }
}

fn settings_file() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("NODECRAFT_CONFIG_DIR") {
        return Some(PathBuf::from(dir).join("settings.json"));
    }
    let home = std::env::var_os("HOME");
    let appdata = std::env::var_os("APPDATA");
    let userprofile = std::env::var_os("USERPROFILE");
    let xdg = std::env::var_os("XDG_CONFIG_HOME");
    Some(
        config_dir_for(
            current_config_platform(),
            home.as_deref(),
            appdata.as_deref(),
            userprofile.as_deref(),
            xdg.as_deref(),
        )?
        .join("settings.json"),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ActivitySettings {
    fade_in_ms: u32,
    hold_ms: u32,
    fade_out_ms: u32,
}

impl Default for ActivitySettings {
    fn default() -> Self {
        Self { fade_in_ms: 40, hold_ms: 500, fade_out_ms: 40 }
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct AppSettings {
    #[serde(flatten)]
    view: crate::viewport::ViewSettings,
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    activity: ActivitySettings,
    #[serde(default)]
    recent_files: Vec<String>,
}

fn load_app_settings() -> Option<AppSettings> {
    let text = std::fs::read_to_string(settings_file()?).ok()?;
    serde_json::from_str(&text).ok()
}

fn save_app_settings(
    view: crate::viewport::ViewSettings,
    theme: Option<String>,
    activity: ActivitySettings,
    recent_files: &[PathBuf],
) -> Result<(), ()> {
    let path = settings_file().ok_or(())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| ())?;
    }
    let settings = AppSettings {
        view,
        theme,
        activity,
        recent_files: recent_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };
    let text = serde_json::to_string_pretty(&settings).map_err(|_| ())?;
    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, text).map_err(|_| ())?;
    std::fs::rename(&temp, &path).map_err(|_| ())?;
    Ok(())
}

/// Apply the saved theme id. Missing, moved or invalid themes intentionally fall
/// back to the embedded default dark palette.
fn apply_saved_theme(id: Option<&str>) {
    let mut preset = id.and_then(|id| {
        crate::theme::available_themes()
            .into_iter()
            .find(|preset| preset.id == id)
    });
    if preset.is_none()
        && let Some(id) = id
        && id != "default"
        && id != "light"
        && id != "forest"
        && std::path::Path::new(id).is_file()
    {
        preset = Some(crate::theme::ThemePreset {
            name: id.to_owned(),
            id: id.to_owned(),
            path: Some(PathBuf::from(id)),
        });
    }
    let loaded = preset
        .as_ref()
        .is_some_and(|preset| crate::theme::load_preset(preset).is_ok());
    if !loaded {
        let default = crate::theme::ThemePreset {
            name: crate::theme::DEFAULT_THEME_NAME.to_owned(),
            id: "default".to_owned(),
            path: None,
        };
        let _ = crate::theme::load_preset(&default);
    }
}

fn normalize_document_path(mut path: PathBuf) -> PathBuf {
    if path.extension().is_none() {
        path.set_extension("nodecraft");
    }
    path
}

fn window_resize_direction(ctx: &egui::Context) -> Option<egui::viewport::ResizeDirection> {
    use egui::viewport::ResizeDirection;
    let rect = ctx.input(|input| input.viewport_rect());
    let pointer = ctx.pointer_latest_pos()?;
    if !rect.contains(pointer) {
        return None;
    }
    const EDGE: f32 = 5.0;
    let west = pointer.x <= rect.left() + EDGE;
    let east = pointer.x >= rect.right() - EDGE;
    let north = pointer.y <= rect.top() + EDGE;
    let south = pointer.y >= rect.bottom() - EDGE;
    match (west, east, north, south) {
        (true, _, true, _) => Some(ResizeDirection::NorthWest),
        (_, true, true, _) => Some(ResizeDirection::NorthEast),
        (true, _, _, true) => Some(ResizeDirection::SouthWest),
        (_, true, _, true) => Some(ResizeDirection::SouthEast),
        (true, _, _, _) => Some(ResizeDirection::West),
        (_, true, _, _) => Some(ResizeDirection::East),
        (_, _, true, _) => Some(ResizeDirection::North),
        (_, _, _, true) => Some(ResizeDirection::South),
        _ => None,
    }
}

fn resize_cursor(direction: egui::viewport::ResizeDirection) -> egui::CursorIcon {
    use egui::viewport::ResizeDirection;
    match direction {
        ResizeDirection::North | ResizeDirection::South => egui::CursorIcon::ResizeVertical,
        ResizeDirection::East | ResizeDirection::West => egui::CursorIcon::ResizeHorizontal,
        ResizeDirection::NorthEast | ResizeDirection::SouthWest => egui::CursorIcon::ResizeNeSw,
        ResizeDirection::NorthWest | ResizeDirection::SouthEast => egui::CursorIcon::ResizeNwSe,
    }
}

fn canvas_theme(theme: &crate::theme::Theme) -> CanvasTheme {
    CanvasTheme {
        window: theme.window,
        blur_tint: theme.blur_tint,
        control: theme.control,
        border: theme.border,
        text: theme.text,
        muted: theme.muted,
        accent: theme.accent,
        highlight: theme.highlight,
        port_input: theme.port_input,
        card: theme.card,
        card_border: theme.card_border,
        cable: theme.cable,
        cable_selected: theme.cable_selected,
        cable_outline: theme.cable_outline,
        cable_selected_outline: theme.cable_selected_outline,
        cable_width: theme.cable_width,
        cable_outline_width: theme.cable_outline_width,
        minimap_bg: theme.minimap_bg,
        minimap_border: theme.minimap_border,
        minimap_node: theme.minimap_node,
        icon: theme.icon,
        icon_hover: theme.icon_hover,
        popup: theme.popup,
    }
}

fn layout_roi(
    document: &GraphDocument,
    positions: &HashMap<String, (f64, f64)>,
) -> Option<egui::Rect> {
    let mut min = egui::Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = egui::Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for node in &document.nodes {
        let (x, y) = positions
            .get(&node.id)
            .copied()
            .unwrap_or((node.x, node.y));
        let x = x as f32;
        let y = y as f32;
        min.x = min.x.min(x - NODE_LEFT_PADDING);
        min.y = min.y.min(y);
        max.x = max.x.max(x + node_width(node) + NODE_RIGHT_PADDING);
        max.y = max.y.max(y + NODE_HEIGHT);
    }
    (min.x.is_finite() && max.x.is_finite()).then(|| egui::Rect::from_min_max(min, max))
}

fn theme_color_dot(ui: &mut egui::Ui, rgba: [u8; 4]) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
    if response.hovered() || response.clicked() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let theme = crate::theme::get();
    let center = rect.center();
    let radius = 8.5;
    ui.painter().circle_filled(center, radius, theme.color(theme.panel));
    ui.painter().circle_filled(center, radius, theme.color(rgba));
    ui.painter().circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, theme.color(theme.border_strong)),
    );
    if response.hovered() {
        ui.painter().circle_stroke(
            center,
            radius + 1.5,
            egui::Stroke::new(1.0, theme.color(theme.accent)),
        );
    }
    response
}

fn theme_revert_button(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(26.0, 26.0),
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );
    let theme = crate::theme::get();
    let hot = enabled && response.hovered();
    if hot {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let tint = if !enabled {
        theme.icon
    } else if hot {
        theme.icon_hover
    } else {
        theme.icon
    };
    let svg = std::str::from_utf8(include_bytes!(
        "../../../../vendor/fluent/arrow_clockwise.svg"
    ))
    .unwrap_or_default();
    let svg = tint_svg(svg.as_bytes(), &crate::theme::Theme::hex(tint));
    let image = egui::Image::from_bytes(
        "bytes://theme-revert-defaults.svg",
        svg.into_bytes(),
    )
    .fit_to_exact_size(egui::Vec2::splat(16.0));
    ui.put(rect.shrink(5.0), image);
    response
}

fn theme_edit_button(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(26.0, 26.0),
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );
    let theme = crate::theme::get();
    let hot = enabled && response.hovered();
    if hot {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let tint = if !enabled {
        theme.icon
    } else if hot {
        theme.icon_hover
    } else {
        theme.icon
    };
    let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/edit.svg"))
        .unwrap_or_default();
    let svg = tint_svg(svg.as_bytes(), &crate::theme::Theme::hex(tint));
    let image = egui::Image::from_bytes(
        "bytes://theme-edit-name.svg",
        svg.into_bytes(),
    )
    .fit_to_exact_size(egui::Vec2::splat(15.0));
    ui.put(rect.shrink(5.0), image);
    if !enabled {
        // Built-in themes cannot be renamed: show a crossed-out pencil rather
        // than merely dimming it.
        let slash = egui::Stroke::new(1.4, theme.color(theme.muted));
        ui.painter().line_segment(
            [
                rect.left_top() + egui::vec2(6.0, 6.0),
                rect.right_bottom() - egui::vec2(6.0, 6.0),
            ],
            slash,
        );
    }
    response
}

fn theme_add_button(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::click());
    let theme = crate::theme::get();
    if response.hovered() {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let color = if response.hovered() {
        theme.color(theme.icon_hover)
    } else {
        theme.color(theme.icon)
    };
    let center = rect.center();
    let arm = 6.0;
    let stroke = egui::Stroke::new(1.6, color);
    ui.painter().line_segment(
        [center + egui::vec2(-arm, 0.0), center + egui::vec2(arm, 0.0)],
        stroke,
    );
    ui.painter().line_segment(
        [center + egui::vec2(0.0, -arm), center + egui::vec2(0.0, arm)],
        stroke,
    );
    response
}

#[derive(Clone)]
struct ThemeHexDraft {
    text: String,
    rgba: [u8; 4],
}

/// Editable hex field for a palette colour. Accepts `#RGB`, `#RGBA`,
/// `#RRGGBB`, and `#RRGGBBAA`; six-digit input keeps the current alpha.
fn theme_hex_edit(ui: &mut egui::Ui, key: &str, rgba: [u8; 4]) -> Option<[u8; 4]> {
    let data_id = ui.make_persistent_id(("theme-hex-color", key));
    let field_id = egui::Id::new(("theme-hex-field", key));
    let mut draft = ui
        .data(|data| data.get_temp::<ThemeHexDraft>(data_id))
        .unwrap_or_else(|| ThemeHexDraft {
            text: crate::theme::Theme::hex_rgba(rgba),
            rgba,
        });

    let response = ui
        .add_sized(
            [ui.available_width(), 18.0],
            egui::TextEdit::singleline(&mut draft.text)
                .id(field_id)
                .font(egui::TextStyle::Monospace)
                .horizontal_align(egui::Align::RIGHT)
                .hint_text("#rrggbbaa"),
        )
        .on_hover_text(
            "Hex colour: #RGB, #RGBA, #RRGGBB, or #RRGGBBAA\n             Six-digit input preserves the current alpha.",
        );

    let mut changed = None;
    if response.changed() {
        if let Some(next) = crate::theme::Theme::parse_hex(&draft.text, draft.rgba[3]) {
            draft.rgba = next;
            changed = Some(next);
        }
    }
    if response.has_focus() {
        // Keep partial/invalid input while the user is typing.
    } else if draft.rgba != rgba {
        draft.rgba = rgba;
        draft.text = crate::theme::Theme::hex_rgba(rgba);
    } else if response.lost_focus() {
        draft.text = crate::theme::Theme::hex_rgba(draft.rgba);
    }

    ui.data_mut(|data| data.insert_temp(data_id, draft));
    changed
}

fn theme_palette_row(
    ui: &mut egui::Ui,
    entry: &crate::theme::ThemeEntry,
    theme: &mut crate::theme::Theme,
    row_width: f32,
) {
    const DOT_WIDTH: f32 = 26.0;
    const VALUE_WIDTH: f32 = 112.0;
    let spacing = ui.spacing().item_spacing.x;
    let label_width = (row_width - DOT_WIDTH - VALUE_WIDTH - spacing * 2.0).max(72.0);
    let value = match entry.value {
        crate::theme::ThemeValue::Color(_) => String::new(),
        crate::theme::ThemeValue::Number { value, .. } => format!("{value:.1}"),
    };

    let row_rect = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(row_width, 20.0),
    );
    if ui.rect_contains_pointer(row_rect) {
        ui.painter().rect_filled(
            row_rect.expand2(egui::vec2(4.0, 1.0)),
            egui::CornerRadius::same(4),
            theme.with_alpha(theme.highlight, 24),
        );
    }

    ui.scope_builder(
        egui::UiBuilder::new().max_rect(row_rect),
        |ui| {
            ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DOT_WIDTH, 20.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(DOT_WIDTH);
                if let crate::theme::ThemeValue::Color(rgba) = entry.value {
                    let dot = theme_color_dot(ui, rgba).on_hover_text(format!(
                        "theme.json: {}\n{}\nClick to edit",
                        entry.key, entry.description
                    ));
                    egui::Popup::menu(&dot)
                        .id(egui::Id::new(("theme-color-picker", entry.key)))
                        .style(crate::theme::menu_style)
                        .show(|ui| {
                            if let Some(color) = theme_color_wheel(ui, entry.key, rgba) {
                                let _ = theme.set_color(entry.key, color);
                            }
                        });
                } else {
                    ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
                }
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(label_width, 20.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(label_width);
                ui.add(egui::Label::new(entry.label).truncate())
                    .on_hover_text(format!(
                        "theme.json: {}\n{}",
                        entry.key, entry.description
                    ));
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(VALUE_WIDTH, 20.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.set_width(VALUE_WIDTH);
                if let crate::theme::ThemeValue::Number {
                    value: number,
                    min,
                    max,
                    step,
                } = entry.value
                {
                    let mut number = number;
                    let response = if entry.key == "backdrop_blur" {
                        ui.add(
                            egui::Slider::new(&mut number, min..=max)
                                .step_by(f64::from(step))
                                .show_value(true),
                        )
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut number)
                                .range(min..=max)
                                .speed(step)
                                .max_decimals(1),
                        )
                    };
                    if response.changed() {
                        let _ = theme.set_number(entry.key, number);
                    }
                } else if let crate::theme::ThemeValue::Color(rgba) = entry.value {
                    if let Some(color) = theme_hex_edit(ui, entry.key, rgba) {
                        let _ = theme.set_color(entry.key, color);
                    }
                } else {
                    ui.monospace(value);
                }
            },
        );
            });
        },
    );
}

fn make_color_wheel_texture(ctx: &egui::Context, radius: f32, size: f32) -> egui::TextureHandle {
    use egui::ecolor::Hsva;
    const WHEEL_SIZE: usize = 256;
    let radius_px = WHEEL_SIZE as f32 * (radius / size);
    let center_px = WHEEL_SIZE as f32 * 0.5;
    let mut pixels = Vec::with_capacity(WHEEL_SIZE * WHEEL_SIZE);
    for y in 0..WHEEL_SIZE {
        for x in 0..WHEEL_SIZE {
            let dx = x as f32 + 0.5 - center_px;
            let dy = y as f32 + 0.5 - center_px;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > radius_px {
                pixels.push(egui::Color32::TRANSPARENT);
            } else {
                let hue = dy.atan2(dx).rem_euclid(std::f32::consts::TAU)
                    / std::f32::consts::TAU;
                let saturation = (distance / radius_px).clamp(0.0, 1.0);
                // Fixed full HSV value: luminosity is a separate control and
                // must never feed back into the wheel.
                pixels.push(egui::Color32::from(Hsva::new(hue, saturation, 1.0, 1.0)));
            }
        }
    }
    ctx.load_texture(
        "theme-color-wheel-gradient",
        egui::ColorImage::new([WHEEL_SIZE, WHEEL_SIZE], pixels),
        egui::TextureOptions::LINEAR,
    )
}

/// Perceptual response for the luminosity slider: slider position is
/// gamma-decoded to HSV value, giving much finer control near black.
const LUMINOSITY_GAMMA: f32 = 2.2;

#[derive(Clone, Copy)]
struct ColorWheelState {
    hsva: egui::ecolor::Hsva,
    rgba: [u8; 4],
}

fn theme_color_wheel(ui: &mut egui::Ui, key: &str, rgba: [u8; 4]) -> Option<[u8; 4]> {
    use egui::ecolor::Hsva;

    let state_id = ui.make_persistent_id(("theme-color-wheel-state", key));
    let mut state = ui
        .data(|data| data.get_temp::<ColorWheelState>(state_id))
        .filter(|state| state.rgba == rgba)
        .unwrap_or_else(|| ColorWheelState {
            hsva: Hsva::from(crate::theme::get().color(rgba)),
            rgba,
        });

    let size = 188.0;
    let (wheel_rect, wheel_response) =
        ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
    let center = wheel_rect.center();
    let radius = size * 0.5 - 3.0;
    let mut changed = false;
    let mut hsva = state.hsva;

    // Circular HSV hue/saturation wheel at fixed full value. Hue/saturation,
    // luminosity and alpha are independent controls with no feedback path.
    let texture = {
        let ctx = ui.ctx().clone();
        let id = egui::Id::new("theme-color-wheel-gradient");
        if let Some(texture) = ctx.data(|data| data.get_temp::<egui::TextureHandle>(id)) {
            texture
        } else {
            let texture = make_color_wheel_texture(&ctx, radius, size);
            ctx.data_mut(|data| data.insert_temp(id, texture.clone()));
            texture
        }
    };
    ui.painter().image(
        texture.id(),
        wheel_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    ui.painter().circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, crate::theme::get().color(crate::theme::get().border_strong)),
    );

    if let Some(pointer) = wheel_response.interact_pointer_pos() {
        let delta = pointer - center;
        let distance = delta.length();
        if distance <= radius + 2.0 {
            hsva.h = delta.y.atan2(delta.x).rem_euclid(std::f32::consts::TAU)
                / std::f32::consts::TAU;
            hsva.s = (distance / radius).clamp(0.0, 1.0);
            changed = true;
        }
    }

    let marker = center
        + egui::vec2(
            (hsva.h * std::f32::consts::TAU).cos(),
            (hsva.h * std::f32::consts::TAU).sin(),
        ) * (hsva.s * radius);
    let marker_colour = if hsva.v > 0.55 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    };
    ui.painter().circle_stroke(
        marker,
        5.0,
        egui::Stroke::new(2.0, marker_colour),
    );
    ui.painter().circle_stroke(
        marker,
        7.0,
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(140)),
    );

    ui.add_space(6.0);
    ui.spacing_mut().slider_width = 150.0;
    let mut luminosity = hsva.v.powf(1.0 / LUMINOSITY_GAMMA);
    if ui
        .add(egui::Slider::new(&mut luminosity, 0.0..=1.0).text("Luminosity"))
        .changed()
    {
        hsva.v = luminosity.powf(LUMINOSITY_GAMMA);
        changed = true;
    }
    let mut alpha = hsva.a;
    if ui
        .add(egui::Slider::new(&mut alpha, 0.0..=1.0).text("Alpha"))
        .changed()
    {
        hsva.a = alpha;
        changed = true;
    }
    if changed {
        let color = egui::Color32::from(Hsva::new(hsva.h, hsva.s, hsva.v, hsva.a));
        let out = color.to_srgba_unmultiplied();
        state.hsva = hsva;
        state.rgba = out;
        ui.data_mut(|data| data.insert_temp(state_id, state));
        return Some(out);
    }
    None
}

fn configure_style(ctx: &egui::Context) {
    // The Swift/WebView app ships Inter (vendor/InterVariable.woff2); use the same
    // Inter family as the leading egui proportional font.
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "inter".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/InterVariable.ttf")).into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "inter".to_owned());
    ctx.set_fonts(fonts);

    let theme = crate::theme::get();
    let bg = theme.color(theme.window);
    let panel = theme.color(theme.panel);
    let control = theme.color(theme.control);
    let border = theme.color(theme.border);
    let border_hi = theme.color(theme.border_strong);
    let text = theme.color(theme.text);
    let muted = theme.color(theme.muted);
    let accent = theme.color(theme.accent);

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = panel;
    visuals.window_fill = bg;
    visuals.extreme_bg_color = bg;
    visuals.faint_bg_color = control;
    visuals.window_stroke = egui::Stroke::new(1.0, border);
    visuals.window_corner_radius = egui::CornerRadius::same(12);
    visuals.menu_corner_radius = egui::CornerRadius::same(12);
    visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 8],
        blur: 24,
        spread: 0,
        color: egui::Color32::from_black_alpha(130),
    };
    visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 8],
        blur: 24,
        spread: 0,
        color: egui::Color32::from_black_alpha(150),
    };
    visuals.override_text_color = Some(text);
    visuals.widgets.noninteractive.bg_fill = panel;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, border);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, muted);
    visuals.widgets.inactive.bg_fill = control;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, border_hi);
    visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text);
    visuals.widgets.hovered.bg_fill = border;
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, border_hi);
    visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, text);
    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, accent);
    visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, text);
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke = egui::Stroke::new(1.0, text);
    visuals.hyperlink_color = accent;
    ctx.set_visuals(visuals);
    ctx.all_styles_mut(|style| {
        // Twelve points is the smallest UI text size we use. Dense technical
        // panels still need readable labels at default display scaling.
        const MIN_UI_TEXT_SIZE: f32 = 12.0;
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        // Labels are UI chrome, not data. TextEdit/DragValue remain selectable.
        style.interaction.selectable_labels = false;
        style.text_styles.insert(
            egui::TextStyle::Body,
            egui::FontId::new(13.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Button,
            egui::FontId::new(13.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Small,
            egui::FontId::new(MIN_UI_TEXT_SIZE, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Heading,
            egui::FontId::new(16.0, egui::FontFamily::Proportional),
        );
        style.text_styles.insert(
            egui::TextStyle::Monospace,
            egui::FontId::new(MIN_UI_TEXT_SIZE, egui::FontFamily::Monospace),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn menu_rows_keep_stable_metrics_across_states() {
        let mut style = egui::Style::default();
        crate::theme::menu_style(&mut style);
        assert_eq!(style.spacing.button_padding, egui::vec2(12.0, 5.0));
        assert_eq!(style.spacing.item_spacing, egui::vec2(8.0, 2.0));
        let widgets = &style.visuals.widgets;
        for state in [
            &widgets.inactive,
            &widgets.hovered,
            &widgets.active,
            &widgets.open,
        ] {
            assert_eq!(state.expansion, 0.0);
            assert_eq!(state.bg_stroke, egui::Stroke::NONE);
            assert_eq!(state.corner_radius, egui::CornerRadius::same(5));
        }
    }

    fn history_fixture() -> GraphDocument {
        let nodes = vec![
            nodecraft_session::GraphNode {
                id: "a".into(),
                node_type: "primitive".into(),
                values: json!({}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            },
            nodecraft_session::GraphNode {
                id: "b".into(),
                node_type: "translate".into(),
                values: json!({}),
                suppressed: false,
                x: 100.0,
                y: 0.0,
            },
        ];
        let connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        GraphDocument {
            schema_version: 1,
            id: "history-fixture".into(),
            nodes,
            connections,
            interface: Default::default(),
        }
    }

    #[test]
    fn session_undo_restores_deleted_cable() {
        let document = history_fixture();
        let mut session = Session::new(document);
        session
            .apply_commands(&[Command::Disconnect {
                from: "a".into(),
                to: "b".into(),
                input: "shape".into(),
            }])
            .expect("disconnect");
        assert!(session.document().connections.is_empty());
        assert!(session.has_undo());
        session.undo().expect("undo").expect("change");
        assert_eq!(session.document().connections.len(), 1);
        session.redo().expect("redo").expect("change");
        assert!(session.document().connections.is_empty());
    }

    #[test]
    fn session_drag_coalesces_and_undoes_as_one_transaction() {
        let mut session = Session::new(history_fixture());
        for (x, y) in [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)] {
            session
                .apply_commands_coalesced(
                    &[Command::SetPosition { node: "a".into(), x, y }],
                    Some("move:a"),
                    std::time::Duration::from_secs(1),
                )
                .expect("move");
        }
        session.undo().expect("undo").expect("change");
        let node = session.document().nodes.iter().find(|node| node.id == "a").expect("a");
        assert_eq!((node.x, node.y), (0.0, 0.0));
        assert!(!session.has_undo());
    }

    #[test]
    fn stable_connection_key_ignores_document_order() {
        let document = history_fixture();
        let key = ConnectionKey::from_connection(&document.connections[0]);
        let moved = GraphDocument {
            nodes: document.nodes.iter().rev().cloned().collect(),
            ..document.clone()
        };
        assert_eq!(ConnectionKey::from_connection(&moved.connections[0]), key);
    }
    #[test]
    fn clipboard_payload_roundtrips_node_descriptions() {
        let payload = NodeClipboard {
            nodes: history_fixture().nodes,
            connections: history_fixture().connections,
        };
        let text = serde_json::to_string(&payload).expect("serialize");
        let decoded: NodeClipboard = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(decoded.nodes, payload.nodes);
        assert_eq!(decoded.connections, payload.connections);
        assert!(text.contains("\"nodes\""));
        assert!(text.contains("\"type\""));
    }

    #[test]
    fn pasted_ids_do_not_collide_with_existing_nodes() {
        let mut used: HashSet<String> = ["primitive".to_owned()].into_iter().collect();
        assert_eq!(unique_node_id("primitive", &mut used), "primitive-copy1");
        assert_eq!(unique_node_id("primitive", &mut used), "primitive-copy2");
    }

    #[test]
    fn paste_accepts_node_arrays_and_single_node_json() {
        let node = history_fixture().nodes[0].clone();
        let array = serde_json::to_string(&vec![node.clone()]).expect("array");
        assert_eq!(parse_node_clipboard(&array).expect("parse array").nodes.len(), 1);
        let single = serde_json::to_string(&node).expect("single");
        assert_eq!(parse_node_clipboard(&single).expect("parse single").nodes[0].id, "a");
    }

    #[test]
    fn clipboard_actions_come_from_egui_events() {
        assert_eq!(clipboard_action(&[egui::Event::Copy]), Some(ClipboardAction::Copy));
        assert_eq!(clipboard_action(&[egui::Event::Cut]), Some(ClipboardAction::Cut));
        assert_eq!(
            clipboard_action(&[egui::Event::Paste("nodes".to_owned())]),
            Some(ClipboardAction::Paste("nodes".to_owned()))
        );
        assert_eq!(clipboard_action(&[]), None);
    }

    #[test]
    fn dropped_files_are_classified_by_extension() {
        assert!(is_nodecraft_path(std::path::Path::new("/tmp/part.nodecraft")));
        assert!(is_nodecraft_path(std::path::Path::new("/tmp/PART.NODECRAFT")));
        assert!(!is_nodecraft_path(std::path::Path::new("/tmp/part.step")));
        assert!(!is_nodecraft_path(std::path::Path::new("/tmp/part")));
        assert!(is_step_path(std::path::Path::new("/tmp/part.step")));
        assert!(is_step_path(std::path::Path::new("/tmp/PART.STP")));
        assert!(!is_step_path(std::path::Path::new("/tmp/part.nodecraft")));
    }

    #[test]
    fn document_ids_follow_canonical_paths() {
        let dir = std::env::temp_dir().join(format!(
            "nodecraft-id-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("part.nodecraft");
        std::fs::write(&path, "{}").expect("file");
        let canonical = std::fs::canonicalize(&path).expect("canonical");
        let via_direct = document_id_for_path(&path);
        let via_dot = document_id_for_path(&dir.join(".").join("part.nodecraft"));
        assert_eq!(via_direct, via_dot);
        assert_eq!(via_direct, canonical.to_string_lossy());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn startup_prefers_explicit_then_last_then_fallback_document() {
        let explicit = Some(std::path::PathBuf::from("/tmp/explicit.nodecraft"));
        let last = Some(std::path::PathBuf::from("/tmp/last.nodecraft"));
        let fallback = Some(std::path::PathBuf::from("/tmp/fallback.nodecraft"));
        assert_eq!(choose_startup_document(explicit.clone(), last.clone(), fallback.clone()), explicit);
        assert_eq!(choose_startup_document(None, last.clone(), fallback.clone()), last);
        assert_eq!(choose_startup_document(None, None, fallback.clone()), fallback);
        assert_eq!(choose_startup_document(None, None, None), None);
    }

    #[test]
    fn app_settings_round_trip_recent_files() {
        let settings = AppSettings {
            view: crate::viewport::ViewSettings::default(),
            theme: Some("punk".to_owned()),
            activity: ActivitySettings::default(),
            recent_files: vec!["/tmp/one.nodecraft".to_owned(), "/tmp/two.nodecraft".to_owned()],
        };
        let text = serde_json::to_string(&settings).expect("serialize settings");
        let parsed: AppSettings = serde_json::from_str(&text).expect("parse settings");
        assert_eq!(parsed.theme.as_deref(), Some("punk"));
        assert_eq!(
            parsed.recent_files,
            vec!["/tmp/one.nodecraft".to_owned(), "/tmp/two.nodecraft".to_owned()]
        );
        assert_eq!(parsed.view, crate::viewport::ViewSettings::default());
        assert_eq!(parsed.activity, ActivitySettings::default());
    }

    #[test]
    fn document_save_paths_are_cross_platform() {
        let base = std::path::PathBuf::from("folder");
        assert_eq!(
            normalize_document_path(base.join("part")),
            base.join("part.nodecraft")
        );
        assert_eq!(
            normalize_document_path(base.join("part.nodecraft")),
            base.join("part.nodecraft")
        );
        assert_eq!(
            normalize_document_path(base.join("part.NODECRAFT")),
            base.join("part.NODECRAFT")
        );
    }

    #[test]
    fn saved_graph_round_trips_on_a_cross_platform_path() {
        let dir = std::env::temp_dir().join(format!(
            "nodecraft-save-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = normalize_document_path(dir.join("part"));
        let document = history_fixture();
        let text = serde_json::to_string_pretty(&document).expect("serialize");
        std::fs::write(&path, text).expect("write");
        let loaded: GraphDocument =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("parse");
        assert_eq!(loaded, document);
        assert_eq!(path.extension().and_then(|value| value.to_str()), Some("nodecraft"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_directories_are_defined_for_every_platform() {
        let home = std::ffi::OsStr::new("/home/test");
        let appdata = std::ffi::OsStr::new("C:\\Users\\test\\AppData\\Roaming");
        let userprofile = std::ffi::OsStr::new("C:\\Users\\test");
        let xdg = std::ffi::OsStr::new("/home/test/.config");
        assert_eq!(
            config_dir_for(ConfigPlatform::MacOs, Some(home), None, None, None),
            Some(PathBuf::from("/home/test/Library/Application Support/NodeCraft"))
        );
        assert_eq!(
            config_dir_for(ConfigPlatform::Windows, None, Some(appdata), None, None),
            Some(PathBuf::from("C:\\Users\\test\\AppData\\Roaming").join("NodeCraft"))
        );
        assert_eq!(
            config_dir_for(ConfigPlatform::Windows, None, None, Some(userprofile), None),
            Some(PathBuf::from("C:\\Users\\test/AppData/Roaming/NodeCraft"))
        );
        assert_eq!(
            config_dir_for(ConfigPlatform::Unix, Some(home), None, None, None),
            Some(PathBuf::from("/home/test/.config/nodecraft"))
        );
        assert_eq!(
            config_dir_for(ConfigPlatform::Unix, Some(home), None, None, Some(xdg)),
            Some(PathBuf::from("/home/test/.config/nodecraft"))
        );
    }

    fn clipboard_fixture() -> GraphDocument {
        let node = |id: &str, node_type: &str, values: Value, x: f64| nodecraft_session::GraphNode {
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x,
            y: 0.0,
        };
        let mut document = GraphDocument::new("clipboard");
        document.nodes = vec![
            node("sketch", "sketch", json!({}), 0.0),
            node("translate", "translate", json!({}), 100.0),
            node("loft", "surfaceloft", json!({}), 200.0),
        ];
        document.connections = vec![
            nodecraft_session::GraphConnection {
                from: "sketch".into(),
                to: "translate".into(),
                input: "shape".into(),
                out: "sketch".into(),
            },
            nodecraft_session::GraphConnection {
                from: "translate".into(),
                to: "loft".into(),
                input: "sections".into(),
                out: "shape".into(),
            },
        ];
        document
    }

    #[test]
    fn copying_a_single_node_does_not_copy_external_links() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["translate"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        assert_eq!(payload.nodes.len(), 1);
        assert!(payload.connections.is_empty());
    }

    #[test]
    fn copying_a_source_does_not_copy_its_outgoing_link() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["sketch"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        assert_eq!(payload.nodes.len(), 1);
        assert!(payload.connections.is_empty());
    }

    #[test]
    fn copying_a_multiselect_keeps_internal_links_only() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["translate", "loft"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        assert_eq!(payload.nodes.len(), 2);
        assert_eq!(payload.connections.len(), 1);
        assert_eq!(payload.connections[0].from, "translate");
        assert_eq!(payload.connections[0].to, "loft");
    }

    #[test]
    fn pasting_single_loft_does_not_restore_upstream_sections() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["loft"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        let (commands, _, _) = paste_plan(&document, payload, None);
        assert!(commands.iter().any(|command| matches!(command, Command::AddNode { node } if node.node_type == "surfaceloft")));
        assert!(!commands.iter().any(|command| matches!(command, Command::Connect { input, .. } if input == "sections")));
    }

    #[test]
    fn pasting_sources_and_loft_together_reconnects_new_sources() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["sketch", "translate", "loft"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        let (commands, _, _) = paste_plan(&document, payload, None);
        assert!(commands.iter().any(|command| matches!(
            command,
            Command::Connect { from, to, input, .. }
                if input == "sections" && from == "translate-copy1" && to == "loft-copy1"
        )));
    }

    #[test]
    fn pasting_into_a_document_without_the_source_drops_the_cable() {
        let source = clipboard_fixture();
        let selected: HashSet<&str> = ["loft"].into_iter().collect();
        let payload = clipboard_payload(&source, &selected).expect("payload");
        let destination = GraphDocument::new("destination");
        let (commands, _, _) = paste_plan(&destination, payload, None);
        assert!(commands.iter().any(|command| matches!(command, Command::AddNode { node } if node.node_type == "surfaceloft")));
        assert!(!commands.iter().any(|command| matches!(command, Command::Connect { .. })));
    }

    #[test]
    fn pasted_nodes_apply_to_the_session() {
        let document = clipboard_fixture();
        let selected: HashSet<&str> = ["translate", "loft"].into_iter().collect();
        let payload = clipboard_payload(&document, &selected).expect("payload");
        let (commands, _, _) = paste_plan(&document, payload, None);
        let mut session = Session::new(document);
        session.apply_commands(&commands).expect("paste");
        assert_eq!(session.document().nodes.len(), 5);
        assert!(session.document().connections.iter().any(|link| {
            link.input == "sections"
                && link.from == "translate-copy1"
                && link.to == "loft-copy1"
        }));
    }

    #[test]
    fn deleting_nodes_does_not_disconnect_cables_removed_with_them() {
        let nodes: HashSet<String> = ["a".to_owned()].into_iter().collect();
        let cables: HashSet<ConnectionKey> = [ConnectionKey {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }]
        .into_iter()
        .collect();
        let commands = delete_selection_commands(&nodes, &cables);
        assert_eq!(commands, vec![Command::RemoveNode { id: "a".into() }]);
    }

    #[test]
    fn undo_movement_animation_only_tracks_moved_nodes() {
        let before_document = history_fixture();
        let mut after_document = before_document.clone();
        after_document.nodes[1].x = 240.0;
        let before = node_positions(&before_document);
        let after = node_positions(&after_document);
        let (start, end) = moved_positions(&before, &after);
        assert_eq!(start.len(), 1);
        assert_eq!(start.get("b"), Some(&(100.0, 0.0)));
        assert_eq!(end.get("b"), Some(&(240.0, 0.0)));
    }

}
