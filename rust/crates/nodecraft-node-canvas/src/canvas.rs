//! GPU node editor overlay.
//!
//! Card ports, headers, cards and connection wires are built as a custom wgpu draw
//! list (`node_editor_gpu`). Text and panel chrome stay egui, which eframe also
//! tessellates and renders with wgpu. The editor never reduces the viewport size: it
//! is an overlay over the full 3D view.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use nodecraft_session::GraphDocument;

use crate::gpu::{
    circle_path, squircle_rect_path, CableVertex, GeometryBuilder, NodeCallback, NodeDraw,
    NodeGeometry, NodeText, ShapeInstance,
};

/// Minimum card width: room for the square operation icon between the small
/// left suppress control and the large right eye control, with edge padding for
/// the top/bottom ports. Cards only grow when their port count or title needs it.
pub const NODE_MIN_WIDTH: f32 = 152.0;
pub const NODE_HEIGHT: f32 = 104.0;
const CABLE_HIT_THRESHOLD: f32 = 18.0;
/// Keep an acquired insertion target while the pointer is still close to its
/// moved path. This absorbs one frame of node/path movement without making the
/// target flicker or jump to a different cable.
const CABLE_HIT_RELEASE: f32 = 24.0;
const BUTTON_RADIUS: f32 = 15.0;
const VIEW_BUTTON_RADIUS: f32 = 30.0;
/// The eye control is centred on the card's right edge, so the node needs this
/// much extra width in fitting, lasso bounds and layout.
pub const NODE_RIGHT_PADDING: f32 = VIEW_BUTTON_RADIUS;
/// The suppress control is centred on the card's left edge.
pub const NODE_LEFT_PADDING: f32 = BUTTON_RADIUS;
const CANVAS_TOP: f32 = 36.0;
const MIN_ZOOM: f32 = 0.02;
const MAX_ZOOM: f32 = 4.0;
const PANEL_FRACTION: f32 = 0.38;
const PANEL_MIN: f32 = 260.0;
const PANEL_DEFAULT_MAX: f32 = 520.0;
const PANEL_MAX_FRACTION: f32 = 0.72;
const PANEL_INSET: f32 = 12.0;
const PANEL_RADIUS: f32 = 14.0;


#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Timeline,
    Centered,
    Organic,
}

impl LayoutMode {
    const ALL: [LayoutMode; 3] = [LayoutMode::Timeline, LayoutMode::Centered, LayoutMode::Organic];
    fn label(self) -> &'static str {
        match self {
            LayoutMode::Timeline => "Timeline",
            LayoutMode::Centered => "Centred",
            LayoutMode::Organic => "Organic",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnectionKey {
    pub from: String,
    pub to: String,
    pub input: String,
    pub out: String,
}

impl ConnectionKey {
    pub fn from_connection(connection: &nodecraft_session::GraphConnection) -> Self {
        Self {
            from: connection.from.clone(),
            to: connection.to.clone(),
            input: connection.input.clone(),
            out: connection.out.clone(),
        }
    }
}

pub enum NodeEditorAction {
    SelectNode { id: String, additive: bool },
    SelectCable { key: ConnectionKey, additive: bool },
    Connect { from: String, to: String, input: String, out: String },
    InsertNodeOnCable { node: String, connection: ConnectionKey, input: String, out: String },
    SetViewerBreakpoint { id: String },
    ToggleSuppressed { id: String },
    SetLassoSelection { nodes: HashSet<String>, live: bool },
    MoveNodes { positions: Vec<(String, f64, f64)> },
    SetPositions(Vec<(String, f64, f64)>),
    CleanupLayout { mode: LayoutMode },
    Deselect,
}

struct ViewAnimation {
    started: Instant,
    duration: Duration,
    roi: Rect,
    start_pan: Vec2,
    start_zoom: f32,
}

struct PanAnimation {
    started: Instant,
    duration: Duration,
    start_pan: Vec2,
    target_pan: Vec2,
}

struct NodeDrag {
    primary: String,
    ids: Vec<String>,
    last_pointer: Pos2,
}

#[derive(Clone)]
struct WirePort {
    node_id: String,
    port: &'static str,
    is_input: bool,
    position: Pos2,
}

struct WireDrag {
    origin: WirePort,
    current: Pos2,
}

#[derive(Clone, PartialEq, Eq)]
struct InsertCandidate {
    node: String,
    connection: ConnectionKey,
    input: String,
    out: String,
}

struct InsertPreview {
    candidate: InsertCandidate,
    progress: f32,
    target: f32,
    shift: f32,
    shift_target: f32,
    shift_from: f32,
    shift_started: f32,
    previous: Option<InsertCandidate>,
    previous_shift: f32,
    previous_from: f32,
    previous_started: f32,
    exit_since: Option<Instant>,
}

impl InsertPreview {
    fn set_shift_target(&mut self, target: f32, now: f32) {
        if (self.shift_target - target).abs() > 0.5 {
            self.shift_from = self.shift;
            self.shift_started = now;
            self.shift_target = target;
        }
    }

    fn matches(&self, candidate: &InsertCandidate) -> bool {
        self.candidate.node == candidate.node
            && self.candidate.connection == candidate.connection
            && self.candidate.input == candidate.input
            && self.candidate.out == candidate.out
    }
}

#[derive(Clone, Copy)]
struct ActivityGlow {
    started: Instant,
    ended: Option<Instant>,
}

pub struct NodeEditor {
    pan: Vec2,
    zoom: f32,
    view_document: Option<String>,
    gpu_ready: bool,
    width: f32,
    left_dock: f32,
    resize_start: Option<(f32, f32)>,
    geometry_key: Option<u64>,
    occlusion_key: Option<u64>,
    occlusion: Option<HashMap<String, Vec<Rect>>>,
    geometry: Option<Arc<NodeGeometry>>,
    shapes: Option<Arc<Vec<ShapeInstance>>>,
    cables: Option<Arc<Vec<CableVertex>>>,
    layout_mode: LayoutMode,
    layout_menu_open: bool,
    fit_next: bool,
    minimap_drag: Option<(Pos2, Vec2)>,
    lasso: Option<(Pos2, Pos2)>,
    lasso_base: Option<HashSet<String>>,
    lasso_additive: bool,
    lasso_last: Option<HashSet<String>>,
    z_order: Vec<String>,
    node_drag: Option<NodeDrag>,
    wire_drag: Option<WireDrag>,
    insert_target: Option<InsertCandidate>,
    insert_preview: Option<InsertPreview>,
    icons: HashMap<String, egui::TextureHandle>,
    theme: crate::theme::CanvasTheme,
    last_viewport: Option<Rect>,
    view_animation: Option<ViewAnimation>,
    pan_animation: Option<PanAnimation>,
    node_fades: HashMap<String, Instant>,
    known_nodes: HashSet<String>,
    fade_document: Option<String>,
    glow_pulse: Option<Instant>,
    error_nodes: HashSet<String>,
    last_selected: HashSet<String>,
    activity: HashMap<String, ActivityGlow>,
    activity_fade_in: Duration,
    activity_hold: Duration,
    activity_fade_out: Duration,
}

impl Default for NodeEditor {
    fn default() -> Self {
        Self {
            pan: Vec2::new(18.0, 76.0),
            zoom: 1.0,
            view_document: None,
            gpu_ready: false,
            width: 0.0,
            left_dock: 0.0,
            resize_start: None,
            geometry_key: None,
            occlusion_key: None,
            occlusion: None,
            geometry: None,
            shapes: None,
            cables: None,
            layout_mode: LayoutMode::Timeline,
            layout_menu_open: false,
            fit_next: false,
            minimap_drag: None,
            lasso: None,
            lasso_base: None,
            lasso_additive: false,
            lasso_last: None,
            z_order: Vec::new(),
            node_drag: None,
            wire_drag: None,
            insert_target: None,
            insert_preview: None,
            icons: HashMap::new(),
            theme: crate::theme::CanvasTheme::default(),
            last_viewport: None,
            view_animation: None,
            pan_animation: None,
            node_fades: HashMap::new(),
            known_nodes: HashSet::new(),
            fade_document: None,
            glow_pulse: None,
            error_nodes: HashSet::new(),
            last_selected: HashSet::new(),
            activity: HashMap::new(),
            activity_fade_in: Duration::from_millis(40),
            // Activity must remain visible after a very fast node completes.
            activity_hold: Duration::from_millis(500),
            activity_fade_out: Duration::from_millis(40),
        }
    }
}

/// One place to tune new-node appearance.
const NODE_FADE_IN: Duration = Duration::from_millis(240);

/// Selection-pulse envelope: 40 ms fade up, 500 ms hold, 40 ms fade down.
fn glow_strength(elapsed: Duration) -> f32 {
    const UP: f32 = 0.040;
    const HOLD: f32 = 0.500;
    const DOWN: f32 = 0.040;
    let t = elapsed.as_secs_f32();
    if t < UP {
        t / UP
    } else if t < UP + HOLD {
        1.0
    } else if t < UP + HOLD + DOWN {
        1.0 - (t - UP - HOLD) / DOWN
    } else {
        0.0
    }
}

fn activity_strength(
    started: Instant,
    ended: Option<Instant>,
    now: Instant,
    fade_in: Duration,
    hold: Duration,
    fade_out: Duration,
) -> f32 {
    let Some(ended) = ended else {
        return if fade_in.is_zero() {
            1.0
        } else {
            (now.duration_since(started).as_secs_f32() / fade_in.as_secs_f32()).clamp(0.0, 1.0)
        };
    };
    let fade_start = ended.max(started + hold);
    if now < fade_start {
        1.0
    } else if fade_out.is_zero() {
        0.0
    } else {
        (1.0 - now.duration_since(fade_start).as_secs_f32() / fade_out.as_secs_f32()).clamp(0.0, 1.0)
    }
}

impl NodeEditor {
    pub fn set_left_dock_width(&mut self, width: f32) {
        self.left_dock = width.max(0.0);
    }

    /// Screen-space rect of the node panel from the last painted frame.
    ///
    /// Used by the host to scope canvas keyboard actions (Delete) to pointer
    /// interaction with the node editor rather than the whole window.
    pub fn panel_screen_rect(&self) -> Option<Rect> {
        self.last_viewport.map(|viewport| self.panel_rect(viewport))
    }

    pub fn panel_rect(&self, viewport: Rect) -> Rect {
        let inset_viewport = viewport.shrink(PANEL_INSET);
        let default = (viewport.width() * PANEL_FRACTION).clamp(PANEL_MIN, PANEL_DEFAULT_MAX);
        let width = if self.width > 0.0 { self.width } else { default };
        let max_width = ((inset_viewport.width() - self.left_dock).max(PANEL_MIN) * PANEL_MAX_FRACTION)
            .max(PANEL_MIN);
        let width = width.clamp(PANEL_MIN, max_width);
        // Float inside the workspace with the same margin as the inspector and
        // leave room for the left-hand tool dock.
        let left = inset_viewport.min.x + self.left_dock;
        Rect::from_min_size(Pos2::new(left, inset_viewport.min.y), Vec2::new(width, inset_viewport.height()))
    }

    /// Pan just enough to bring `graph` into the visible canvas with padding.
    /// Used after inserting a node so a card placed below the current view does
    /// not appear off-screen.
    pub fn ensure_graph_rect_visible(&mut self, viewport: Rect, graph: Rect, padding: f32) {
        let panel = self.panel_rect(viewport);
        let canvas = Rect::from_min_max(Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP), panel.max);
        let screen = Rect::from_min_size(
            canvas.min + self.pan + graph.min.to_vec2() * self.zoom,
            graph.size() * self.zoom,
        );
        let mut delta = ensure_visible_delta(canvas, screen, padding);
        // Keep the added node clear of the minimap. Move the canvas up just enough
        // if the target would sit underneath it.
        let minimap = node_minimap_rect(panel).expand(padding);
        let adjusted = screen.translate(delta);
        if adjusted.intersects(minimap) {
            delta.y -= adjusted.max.y - (minimap.min.y - padding);
        }
        let target = self.pan + delta;
        if target != self.pan {
            self.view_animation = None;
            self.pan_animation = Some(PanAnimation {
                started: Instant::now(),
                duration: Duration::from_millis(220),
                start_pan: self.pan,
                target_pan: target,
            });
        }
    }

    pub fn install_gpu(&mut self, render_state: &egui_wgpu::RenderState, msaa_samples: u32, depth: wgpu::TextureFormat) {
        // Diagnostic escape hatch for exercising the CPU fallback renderer.
        if std::env::var_os("NODECRAFT_CPU_NODES").is_some() {
            self.gpu_ready = false;
            return;
        }
        self.gpu_ready = crate::gpu::install(render_state, msaa_samples, depth);
    }

    pub fn gpu_ready(&self) -> bool {
        self.gpu_ready
    }

    #[cfg(test)]
    fn set_gpu_ready_for_test(&mut self, ready: bool) {
        self.gpu_ready = ready;
    }

    fn icon_texture(
        &mut self,
        ctx: &egui::Context,
        name: &str,
        source: &'static [u8],
        colour: [u8; 4],
    ) -> egui::TextureHandle {
        if let Some(texture) = self.icons.get(name) {
            return texture.clone();
        }
        let svg = colorized_icon_svg(source, colour);
        let image = egui_extras::image::load_svg_bytes_with_size(
            svg.as_bytes(),
            egui::SizeHint::Size {
                width: 128,
                height: 128,
                maintain_aspect_ratio: true,
            },
            &Default::default(),
        );
        let image = image.unwrap_or_else(|_| egui::ColorImage::example());
        let texture = ctx.load_texture(
            format!("node-icon-{name}"),
            image,
            egui::TextureOptions::LINEAR,
        );
        self.icons.insert(name.to_owned(), texture.clone());
        texture
    }

    pub fn set_theme(&mut self, theme: crate::theme::CanvasTheme) {
        self.theme = theme;
        self.icons.clear();
        self.geometry_key = None;
    }

    pub fn reset_view(&mut self) {
        self.view_document = None;
        self.view_animation = None;
        self.pan_animation = None;
        self.fit_next = true;
    }

    pub fn zoom_out(&mut self) {
        self.zoom = (self.zoom * 0.82).max(MIN_ZOOM);
    }

    pub fn request_fit(&mut self) {
        self.view_animation = None;
        self.pan_animation = None;
        self.fit_next = true;
    }

    /// Smoothly move the camera from the current view to a fit of `roi`
    /// (graph coordinates). Node positions are animated by the app in parallel.
    pub fn animate_to_roi(&mut self, roi: Rect, duration: Duration) {
        if roi.width() <= 1.0 && roi.height() <= 1.0 {
            return;
        }
        self.fit_next = false;
        self.pan_animation = None;
        self.view_animation = Some(ViewAnimation {
            started: Instant::now(),
            duration,
            roi,
            start_pan: self.pan,
            start_zoom: self.zoom,
        });
    }

    /// Convert a screen position to canvas graph coordinates. Returns None until
    /// the first frame has established the panel viewport.
    pub fn graph_pos(&self, screen: Pos2) -> Option<Pos2> {
        let viewport = self.last_viewport?;
        let panel = self.panel_rect(viewport);
        let canvas = Rect::from_min_max(Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP), panel.max);
        Some(((screen - canvas.min - self.pan) / self.zoom).to_pos2())
    }

    /// Routed cable paths in graph coordinates, optionally using temporary node
    /// positions (used by placement before the moves are committed).
    pub fn connection_paths(
        document: &GraphDocument,
        positions: &HashMap<String, (f64, f64)>,
    ) -> Vec<([Pos2; 4], ConnectionKey)> {
        let nodes: HashMap<&str, &nodecraft_session::GraphNode> = document
            .nodes
            .iter()
            .map(|node| (node.id.as_str(), node))
            .collect();
        let mut paths = Vec::new();
        for connection in &document.connections {
            let Some(from) = nodes.get(connection.from.as_str()).copied() else { continue };
            let Some(to) = nodes.get(connection.to.as_str()).copied() else { continue };
            let (fx, fy) = positions.get(&from.id).copied().unwrap_or((from.x, from.y));
            let (tx, ty) = positions.get(&to.id).copied().unwrap_or((to.x, to.y));
            let source = Rect::from_min_size(
                Pos2::new(fx as f32, fy as f32),
                Vec2::new(node_width(from), NODE_HEIGHT),
            );
            let target = Rect::from_min_size(
                Pos2::new(tx as f32, ty as f32),
                Vec2::new(node_width(to), NODE_HEIGHT),
            );
            let outputs = node_outputs(from);
            let inputs = node_inputs(to);
            let x0 = port_x(source, port_index(outputs, &connection.out), outputs.len(), 1.0);
            let y0 = source.max.y;
            let x1 = port_x(target, port_index(inputs, &connection.input), inputs.len(), 1.0);
            let y1 = target.min.y;
            let route = crate::routing::route(x0, y0, x1, y1, 1.0);
            paths.push((
                [
                    Pos2::new(x0, y0),
                    route.control1,
                    route.control2,
                    Pos2::new(x1, y1),
                ],
                ConnectionKey::from_connection(connection),
            ));
        }
        paths
    }

    fn insert_candidate(
        &self,
        ctx: &egui::Context,
        document: &GraphDocument,
        panel: Rect,
    ) -> Option<InsertCandidate> {
        let drag = self.node_drag.as_ref()?;
        if !ctx.input(|input| input.modifiers.shift) {
            return None;
        }
        let node = document.nodes.iter().find(|node| node.id == drag.primary)?;
        let (input, out) = insertion_ports(node)?;
        // Interactive drags probe from the pointer. Headless/MCP insertion has
        // no pointer, so use the centre of the dragged node instead.
        let probe = insertion_probe(
            ctx.pointer_latest_pos(),
            node,
            panel,
            self.pan,
            self.zoom,
        );
        let held = self.insert_target.as_ref().map(|target| target.connection.clone());
        // The preview shifts downstream cards, so the path actually painted this
        // frame comes from the preview document. Hit-test that path first; while
        // the shift animates, accept the original path as a hold so the target
        // does not drop out from under the pointer.
        let preview = self.preview_document(document);
        let hit_document = preview.as_ref().unwrap_or(document);
        let connection = cable_hit_with_hold(
            hit_document,
            panel,
            self.pan,
            self.zoom,
            probe,
            held.as_ref(),
        )
        .or_else(|| {
            let held = held.as_ref()?;
            (cable_hit_with_hold(
                document,
                panel,
                self.pan,
                self.zoom,
                probe,
                Some(held),
            ) == Some(held.clone()))
            .then_some(held.clone())
        })?;
        if connection.from == drag.primary || connection.to == drag.primary {
            return None;
        }
        Some(InsertCandidate {
            node: drag.primary.clone(),
            connection,
            input: input.to_owned(),
            out: out.to_owned(),
        })
    }

    fn update_insert_preview(
        &mut self,
        ctx: &egui::Context,
        document: &GraphDocument,
        panel: Rect,
    ) {
        let hovered = self.insert_candidate(ctx, document, panel);
        // Cable choice is immediate; the room-making shift animates in at once
        // but once shifted it is held briefly before retracting.
        self.insert_target = hovered.clone();
        let now_instant = Instant::now();
        let now = ctx.input(|input| input.time) as f32;
        if let Some(candidate) = hovered {
            if self.insert_preview.as_ref().is_none_or(|preview| !preview.matches(&candidate)) {
                let previous = self.insert_preview.as_ref().map(|preview| {
                    (preview.candidate.clone(), preview.shift)
                });
                self.insert_preview = Some(InsertPreview {
                    candidate,
                    progress: 0.0,
                    target: 1.0,
                    shift: 0.0,
                    shift_target: 0.0,
                    shift_from: 0.0,
                    shift_started: now,
                    previous: previous.as_ref().map(|(candidate, _)| candidate.clone()),
                    previous_shift: previous.as_ref().map(|(_, shift)| *shift).unwrap_or(0.0),
                    previous_from: previous.as_ref().map(|(_, shift)| *shift).unwrap_or(0.0),
                    previous_started: now,
                    exit_since: None,
                });
            }
            if let Some(preview) = &mut self.insert_preview {
                preview.target = 1.0;
                preview.set_shift_target(1.0, now);
                preview.exit_since = None;
            }
        } else if let Some(preview) = &mut self.insert_preview {
            preview.target = 0.0;
            let exit = *preview.exit_since.get_or_insert(now_instant);
            if now_instant.duration_since(exit) >= Duration::from_secs(1) {
                preview.set_shift_target(0.0, now);
            }
        }

        if let Some(preview) = &mut self.insert_preview {
            let dt = ctx.input(|input| input.stable_dt).clamp(0.0, 0.05);
            let fade = dt / 0.18;
            preview.progress = if preview.target > 0.5 {
                (preview.progress + fade).min(1.0)
            } else {
                (preview.progress - fade).max(0.0)
            };
            let t = ((now - preview.shift_started) / 0.45).clamp(0.0, 1.0);
            let eased = t * t * (3.0 - 2.0 * t);
            preview.shift = preview.shift_from + (preview.shift_target - preview.shift_from) * eased;
            let previous_t = ((now - preview.previous_started) / 0.45).clamp(0.0, 1.0);
            let previous_eased = previous_t * previous_t * (3.0 - 2.0 * previous_t);
            preview.previous_shift = preview.previous_from * (1.0 - previous_eased);
            if preview.previous_shift <= 0.001 {
                preview.previous = None;
            }
            if preview.progress <= 0.0
                && preview.shift <= 0.0
                && preview.shift_target <= 0.0
                && preview.previous.is_none()
            {
                self.insert_preview = None;
            } else {
                ctx.request_repaint();
            }
        }
    }

    fn preview_document(&self, document: &GraphDocument) -> Option<GraphDocument> {
        let preview = self.insert_preview.as_ref()?;
        if preview.shift <= 0.001 && preview.previous_shift <= 0.001 {
            return None;
        }
        let mut preview_document = document.clone();
        let descendants = downstream_nodes(document, &preview.candidate.connection.to);
        let previous_descendants = preview
            .previous
            .as_ref()
            .map(|candidate| downstream_nodes(document, &candidate.connection.to));
        let unit = f64::from(NODE_HEIGHT + 80.0);
        for node in &mut preview_document.nodes {
            let mut amount = 0.0;
            if node.id != preview.candidate.node && descendants.contains(&node.id) {
                amount += preview.shift;
            }
            if let (Some(previous), Some(previous_descendants)) =
                (preview.previous.as_ref(), previous_descendants.as_ref())
            {
                if node.id != previous.node && previous_descendants.contains(&node.id) {
                    amount += preview.previous_shift;
                }
            }
            if amount > 0.001 {
                node.y += unit * f64::from(amount);
            }
        }
        Some(preview_document)
    }

    pub fn set_error_nodes(&mut self, nodes: HashSet<String>) {
        if self.error_nodes != nodes {
            self.error_nodes = nodes;
        }
    }

    /// Restart the selected-node glow envelope. Called when evaluation
    /// completes so the pulse follows the result rather than the click.
    pub fn pulse_glow(&mut self) {
        self.glow_pulse = Some(Instant::now());
    }

    pub fn set_activity_timings(&mut self, fade_in_ms: u64, hold_ms: u64, fade_out_ms: u64) {
        self.activity_fade_in = Duration::from_millis(fade_in_ms);
        self.activity_hold = Duration::from_millis(hold_ms.max(500));
        self.activity_fade_out = Duration::from_millis(fade_out_ms);
    }

    pub fn clear_activity(&mut self) {
        self.activity.clear();
    }

    pub fn activity_started(&mut self, node_id: &str) {
        self.activity.insert(
            node_id.to_owned(),
            ActivityGlow { started: Instant::now(), ended: None },
        );
    }

    pub fn activity_finished(&mut self, node_id: &str) {
        if let Some(activity) = self.activity.get_mut(node_id)
            && activity.ended.is_none()
        {
            activity.ended = Some(Instant::now());
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        viewport: Rect,
        document: &GraphDocument,
        selected_nodes: &HashSet<String>,
        selected_connections: &HashSet<ConnectionKey>,
        viewer_breakpoint: Option<&str>,
        backdrop: Option<(egui::TextureId, egui::Rect)>,
    ) -> Vec<NodeEditorAction> {
        self.last_viewport = Some(viewport);
        if self.last_selected != *selected_nodes {
            self.last_selected = selected_nodes.clone();
            if !selected_nodes.is_empty() {
                self.glow_pulse = Some(Instant::now());
            }
        }
        let glow = self.glow_pulse.map_or(0.0, |started| glow_strength(started.elapsed()));
        if glow > 0.0 {
            ctx.request_repaint_after(Duration::from_millis(16));
        } else if self.glow_pulse.is_some() {
            self.glow_pulse = None;
        }
        let panel = self.panel_rect(viewport);
        if self.width <= 0.0 {
            self.width = panel.width();
        }
        let fit_canvas = panel_canvas(panel);
        if self.fit_next {
            self.fit_next = false;
            self.fit_graph(fit_canvas, document);
            self.view_document = Some(document.id.clone());
        }
        if self.view_document.as_deref() != Some(document.id.as_str()) {
            self.view_document = Some(document.id.clone());
            self.view_animation = None;
            self.fit_graph(fit_canvas, document);
        }
        self.tick_view_animation(fit_canvas, ctx);
        self.tick_pan_animation(ctx);
        self.update_insert_preview(ctx, document, panel);
        self.update_node_fades(document);
        let preview_document = self.preview_document(document);
        let document = preview_document.as_ref().unwrap_or(document);
        let now = Instant::now();
        let node_fades = self.node_fade_map(document, now);
        if !node_fades.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        self.activity.retain(|_, activity| {
            activity.ended.is_none_or(|ended| {
                now < ended.max(activity.started + self.activity_hold) + self.activity_fade_out
            })
        });
        if !self.activity.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        let mut activity_glows = Vec::new();
        if !self.activity.is_empty() || !selected_nodes.is_empty() {
            let nodes: HashMap<&str, &nodecraft_session::GraphNode> = document
                .nodes
                .iter()
                .map(|node| (node.id.as_str(), node))
                .collect();
            let radius = glow_radius(self.zoom);
            let accent = self.theme.f32(self.theme.node_activity);
            for (id, activity) in &self.activity {
                if self.error_nodes.contains(id) { continue; }
                let Some(node) = nodes.get(id.as_str()) else { continue };
                let strength = activity_strength(
                    activity.started,
                    activity.ended,
                    now,
                    self.activity_fade_in,
                    self.activity_hold,
                    self.activity_fade_out,
                );
                if strength > 0.0 {
                    shape_glow(&mut activity_glows, graph_rect(node), accent, 3.0, strength, radius);
                }
            }
            // Selection glow is dynamic too: zooming changes its screen-space
            // radius but must not rebuild every card's shape instances.
            for id in selected_nodes {
                if self.error_nodes.contains(id) { continue; }
                if let Some(node) = nodes.get(id.as_str()) {
                    shape_glow(&mut activity_glows, graph_rect(node), accent, 2.0, 0.0, radius);
                }
            }
        }
        if !self.error_nodes.is_empty() {
            let radius = glow_radius(self.zoom);
            for id in &self.error_nodes {
                if let Some(node) = document.nodes.iter().find(|node| &node.id == id) {
                    shape_glow(&mut activity_glows, graph_rect(node), self.theme.f32(self.theme.node_error), 4.0, 1.0, radius * 1.25);
                }
            }
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        self.sync_z_order(document);
        let mut clicked = None;
        let mut moved_nodes: Option<Vec<(String, f64, f64)>> = None;
        let mut actions = Vec::new();
        let mut chrome_clicked = false;
        let mut empty_canvas_clicked = false;
        let mut overlay_texts: Vec<(String, Pos2, f32, Rect, Vec<Rect>, f32)> = Vec::new();
        let mut key = geometry_key(document, selected_nodes, selected_connections, &self.z_order, viewer_breakpoint, &self.theme);
        if !node_fades.is_empty() {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            for (id, alpha) in &node_fades {
                id.hash(&mut hasher);
                alpha.to_bits().hash(&mut hasher);
            }
            key = hasher.finish();
        }
        if self.geometry_key != Some(key) {
            if self.gpu_ready {
                self.shapes = Some(Arc::new(build_shape_instances(
                    document,
                    selected_nodes,
                    &self.z_order,
                    viewer_breakpoint,
                    &self.theme,
                    &node_fades,
                )));
                self.geometry = None;
            } else {
                self.geometry = Some(Arc::new(build_geometry(document, selected_nodes, &self.z_order, viewer_breakpoint, &self.theme, &node_fades)));
                self.shapes = None;
            }
            self.cables = Some(Arc::new(build_cables(document, selected_nodes, selected_connections, &self.theme)));
            self.geometry_key = Some(key);
        }
        // Always carry the cached geometry into the callback. egui may prepare a
        // second pass in the same frame, and a None here would prevent the upload
        // from ever reaching the renderer even though the geometry exists.
        let geometry = self.geometry.clone();
        let shapes = self.shapes.clone();
        let cables = self.cables.clone();
        let ppp = ctx.pixels_per_point();
        let mut draw = NodeDraw {
            geometry_key: key,
            geometry,
            shapes,
            glows: activity_glows,
            cables,
            transform: [self.zoom, panel.min.x + self.pan.x, panel.min.y + CANVAS_TOP + self.pan.y, 0.0],
            clip: [
                panel.min.x + 1.0,
                panel.min.y + 1.0,
                panel.max.x - 1.0,
                panel.max.y - 1.0,
            ],
            clip_radius: PANEL_RADIUS - 1.0,
            ppp,
            // Multiplier for cable subdivision; one place to tune the visual
            // smoothness/vertex-cost tradeoff.
            curve_resolution: 1.5,
            glow: [glow, 0.0, 0.0, 0.0],
            texts: Vec::new(),
        };
        

        egui::Area::new(egui::Id::new("node-editor"))
            .order(egui::Order::Foreground)
            .fixed_pos(panel.min)
            .interactable(true)
            .show(ctx, |ui| {
                ui.set_clip_rect(panel);
                let painter = ui.painter().with_clip_rect(panel);
                if let Some((texture, uv)) = backdrop {
                    // The blurred framebuffer is rendered with a gutter around the
                    // panel crop for edge sampling; display only the inner crop.
                    crate::squircle::image(&painter, texture, panel, uv, PANEL_RADIUS);
                } else {
                    painter.add(crate::squircle::shape(
                        panel,
                        PANEL_RADIUS,
                        self.theme.color(self.theme.blur_tint),
                        Stroke::NONE,
                    ));
                }
                painter.add(crate::squircle::shape(
                    panel,
                    PANEL_RADIUS,
                    Color32::TRANSPARENT,
                    Stroke::new(1.0, self.theme.color(self.theme.border)),
                ));
                // No header band: the panel name sits directly on the canvas.
                painter.text(
                    panel.min + Vec2::new(16.0, 18.0),
                    egui::Align2::LEFT_CENTER,
                    "NODE EDITOR",
                    egui::FontId::proportional(12.0),
                    self.theme.color(self.theme.text),
                );

                let canvas = Rect::from_min_max(Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP), panel.max);
                let mini = node_minimap_rect(panel);
                let pointer = ui.ctx().pointer_latest_pos();
                if self.wire_drag.is_none()
                    && pointer.is_some_and(|pointer| panel.contains(pointer))
                    && ui.ctx().input(|input| input.pointer.primary_pressed())
                {
                    if let Some(port) = pointer.and_then(|pointer| {
                        port_at(document, canvas, self.pan, self.zoom, pointer)
                    }) {
                        self.wire_drag = Some(WireDrag { current: port.position, origin: port });
                    }
                }
                let response = ui.interact(canvas, egui::Id::new("node-editor-canvas"), Sense::click_and_drag());
                if response.dragged_by(egui::PointerButton::Middle)
                    || response.dragged_by(egui::PointerButton::Secondary)
                {
                    self.pan += response.drag_delta();
                }
                // Start the lasso on the press, not after egui's drag threshold.
                // Waiting for drag_started makes the rubber band visibly lag the
                // pointer by several pixels. The node/minimap hit checks keep a
                // normal node press or chrome click from starting a lasso.
                if self.wire_drag.is_none()
                    && ui.ctx().input(|input| input.pointer.primary_pressed())
                    && let Some(pointer) = pointer
                {
                    let over_node = document.nodes.iter().any(|node| {
                        node_bounds(node, canvas, self.pan, self.zoom).contains(pointer)
                    });
                    if canvas.contains(pointer) && !mini.contains(pointer) && !over_node {
                        self.lasso = Some((pointer, pointer));
                        self.lasso_base = Some(selected_nodes.clone());
                        self.lasso_additive =
                            ui.ctx().input(|i| i.modifiers.command || i.modifiers.shift);
                        self.lasso_last = None;
                    }
                }
                if let Some(drag) = &mut self.wire_drag {
                    if let Some(pointer) = ui.ctx().pointer_latest_pos() {
                        drag.current = pointer;
                    }
                    if !ui.ctx().input(|input| input.pointer.primary_down()) {
                        let target = ui.ctx().pointer_latest_pos().and_then(|pointer| {
                            port_at(document, canvas, self.pan, self.zoom, pointer)
                        });
                        if let Some(target) = target.filter(|target| port_compatible(&drag.origin, target)) {
                            let (from, to, input, out) = if drag.origin.is_input {
                                (target.node_id.clone(), drag.origin.node_id.clone(), drag.origin.port.to_owned(), target.port.to_owned())
                            } else {
                                (drag.origin.node_id.clone(), target.node_id.clone(), target.port.to_owned(), drag.origin.port.to_owned())
                            };
                            actions.push(NodeEditorAction::Connect { from, to, input, out });
                        }
                        self.wire_drag = None;
                    }
                }
                if let Some((start, end)) = self.lasso {
                    let end = response.interact_pointer_pos().unwrap_or(end);
                    self.lasso = Some((start, end));
                    let lasso_rect = Rect::from_two_pos(start, end);
                    let nodes = lasso_selection(
                        document,
                        canvas,
                        self.pan,
                        self.zoom,
                        lasso_rect,
                        self.lasso_base.as_ref().unwrap_or(&selected_nodes),
                        self.lasso_additive,
                    );
                    if self.lasso_last.as_ref() != Some(&nodes) {
                        self.lasso_last = Some(nodes.clone());
                        actions.push(NodeEditorAction::SetLassoSelection { nodes, live: true });
                    }
                }
                if !ui.ctx().input(|input| input.pointer.primary_down()) {
                    if let Some((start, end)) = self.lasso.take() {
                        let lasso_rect = Rect::from_two_pos(start, end);
                        let nodes = lasso_selection(
                            document,
                            canvas,
                            self.pan,
                            self.zoom,
                            lasso_rect,
                            self.lasso_base.as_ref().unwrap_or(&selected_nodes),
                            self.lasso_additive,
                        );
                        self.lasso_base = None;
                        self.lasso_last = None;
                        actions.push(NodeEditorAction::SetLassoSelection { nodes, live: false });
                    }
                }
                // Navigation gestures belong to the panel, not to whatever widget is
                // under the pointer. Keep zoom/pan working while hovering node cards.
                let pointer_over_panel = ui
                    .ctx()
                    .pointer_latest_pos()
                    .is_some_and(|pointer| panel.contains(pointer));
                if pointer_over_panel {
                    let gesture = crate::gesture::TrackpadGesture::read(ui.ctx());
                    if (gesture.zoom - 1.0).abs() > f32::EPSILON {
                        let anchor = gesture.pointer.unwrap_or(canvas.center());
                        self.zoom_about(canvas, anchor, gesture.zoom);
                    } else if gesture.wheel != egui::Vec2::ZERO {
                        let anchor = gesture.pointer.unwrap_or(canvas.center());
                        self.zoom_about(canvas, anchor, (gesture.wheel.y * 0.0015).exp());
                    } else if gesture.pan != egui::Vec2::ZERO {
                        // Two-finger trackpad movement pans the graph.
                        self.pan += gesture.pan;
                    }
                }

                // Quantize the nav offset to physical pixels so the retained card
                // geometry and glyphon labels move on the same pixel grid. Also
                // refresh the transform after this frame's gesture updates: the
                // callback otherwise uses last frame's pan while labels use the new
                // pan, which is the source of the relative jitter.
                let draw_pan = Vec2::new(
                    (self.pan.x * ppp).round() / ppp,
                    (self.pan.y * ppp).round() / ppp,
                );
                draw.transform = [
                    self.zoom,
                    panel.min.x + draw_pan.x,
                    panel.min.y + CANVAS_TOP + draw_pan.y,
                    0.0,
                ];

                // One viewport test per node per frame. Every remaining loop uses
                // this list instead of walking the whole graph repeatedly.
                let visible_nodes: Vec<&nodecraft_session::GraphNode> = document
                    .nodes
                    .iter()
                    .filter(|node| {
                        node_bounds(node, canvas, draw_pan, self.zoom).intersects(
                            canvas.expand(VIEW_BUTTON_RADIUS * self.zoom),
                        )
                    })
                    .collect();

                // Keep a node drag alive independently of what is under the
                // pointer. The egui node widget may no longer be hovered once the
                // card moves or the pointer leaves the panel; the captured drag
                // must continue until the primary button is released.
                if let Some(drag) = &mut self.node_drag {
                    let down = ui.ctx().input(|i| i.pointer.primary_down());
                    if !down {
                        self.node_drag = None;
                        let mut inserting = false;
                        if ui.ctx().input(|i| i.modifiers.shift) {
                            if let Some(candidate) = self.insert_target.take() {
                                actions.push(NodeEditorAction::InsertNodeOnCable {
                                    node: candidate.node,
                                    connection: candidate.connection,
                                    input: candidate.input,
                                    out: candidate.out,
                                });
                                inserting = true;
                            }
                        }
                        self.insert_target = None;
                        // If the drag did not join, leave the preview alive so its
                        // downstream shift can ease back out instead of snapping.
                        if inserting {
                            self.insert_preview = None;
                        }
                    } else if let Some(pointer) = ui.ctx().pointer_latest_pos() {
                        let delta = pointer - drag.last_pointer;
                        drag.last_pointer = pointer;
                        if delta != Vec2::ZERO {
                            let positions: Vec<(String, f64, f64)> = drag
                                .ids
                                .iter()
                                .filter_map(|id| document.nodes.iter().find(|node| &node.id == id))
                                .map(|node| {
                                    (
                                        node.id.clone(),
                                        node.x + f64::from(delta.x / self.zoom),
                                        node.y + f64::from(delta.y / self.zoom),
                                    )
                                })
                                .collect();
                            if !positions.is_empty() {
                                moved_nodes = Some(positions);
                            }
                        }
                    }
                }

                // Only nodes under the pointer need egui interaction regions.
                // Creating a widget for every visible node makes the graph cost
                // scale with the viewport, not with the pointer's neighbourhood.
                let pointer = ui.ctx().pointer_latest_pos();
                let interactive_nodes: Vec<&nodecraft_session::GraphNode> = match pointer {
                    Some(pointer) if panel.contains(pointer) => visible_nodes
                        .iter()
                        .copied()
                        .filter(|node| {
                            let rect = node_rect(node, canvas, self.pan, self.zoom);
                            if rect.contains(pointer) {
                                return true;
                            }
                            let (suppress_center, view_center) = button_centers(rect);
                            round_button_hit(
                                suppress_center,
                                pointer,
                                BUTTON_RADIUS * self.zoom,
                            ) || round_button_hit(
                                view_center,
                                pointer,
                                VIEW_BUTTON_RADIUS * self.zoom,
                            )
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                for node in &interactive_nodes {
                    let rect = node_rect(node, canvas, self.pan, self.zoom);
                    let (suppress_center, view_center) = button_centers(rect);
                    let power_rect = round_button_rect(
                        suppress_center,
                        BUTTON_RADIUS * self.zoom,
                    )
                    .intersect(panel);
                    if pointer.is_some_and(|pointer| {
                        round_button_hit(
                            suppress_center,
                            pointer,
                            BUTTON_RADIUS * self.zoom,
                        )
                    }) {
                        let power = ui.interact(
                            power_rect,
                            egui::Id::new(("node-power", &node.id)),
                            Sense::click(),
                        );
                        if power.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if power.clicked() {
                            actions.push(NodeEditorAction::ToggleSuppressed { id: node.id.clone() });
                        }
                        // The power control sits above the card. Do not register
                        // the node drag/click region underneath it, otherwise the
                        // card steals the button hit.
                        continue;
                    }
                    let eye_rect = Rect::from_center_size(
                        view_center,
                        Vec2::splat(VIEW_BUTTON_RADIUS * 2.0 * self.zoom),
                    );
                    let eye_hit = eye_rect.intersect(panel);
                    let over_eye = pointer.is_some_and(|pointer| eye_hit.contains(pointer));
                    if over_eye {
                        let eye = ui.interact(
                            eye_hit,
                            egui::Id::new(("node-eye", &node.id)),
                            Sense::click(),
                        );
                        if eye.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if eye.clicked() {
                            // Shift-click steps the viewer to the first consumer, which
                            // is the explicit way to inspect the graph after this
                            // breakpoint without moving it implicitly on selection.
                            let target = if ui.ctx().input(|input| input.modifiers.shift) {
                                document
                                    .connections
                                    .iter()
                                    .find(|link| link.from == node.id)
                                    .map_or_else(|| node.id.clone(), |link| link.to.clone())
                            } else {
                                node.id.clone()
                            };
                            actions.push(NodeEditorAction::SetViewerBreakpoint { id: target });
                            chrome_clicked = true;
                        }
                        eye.on_hover_text("View output · Shift-click: next node");
                        continue;
                    }
                    let response = ui.interact(rect, egui::Id::new(("node", &node.id)), Sense::click_and_drag());
                    if self.wire_drag.is_none() && response.drag_started_by(egui::PointerButton::Primary) {
                        self.bring_to_front(&node.id);
                        // A drag exceeds egui's click threshold, so `clicked()` never
                        // fires on release. Select an unselected node as the drag
                        // starts; dragging a node that is already selected keeps the
                        // existing multi-selection intact.
                        if !selected_nodes.contains(node.id.as_str()) {
                            clicked = Some(node.id.clone());
                        }
                        let ids = if selected_nodes.contains(node.id.as_str()) && selected_nodes.len() > 1 {
                            document
                                .nodes
                                .iter()
                                .filter(|candidate| selected_nodes.contains(candidate.id.as_str()))
                                .map(|candidate| candidate.id.clone())
                                .collect()
                        } else {
                            vec![node.id.clone()]
                        };
                        self.node_drag = response.interact_pointer_pos().map(|pointer| NodeDrag {
                            primary: node.id.clone(),
                            ids,
                            last_pointer: pointer,
                        });
                    }
                    if response.clicked() {
                        clicked = Some(node.id.clone());
                    }
                }

                let occlusion_key = {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    key.hash(&mut hasher);
                    canvas.min.x.to_bits().hash(&mut hasher);
                    canvas.min.y.to_bits().hash(&mut hasher);
                    canvas.width().to_bits().hash(&mut hasher);
                    canvas.height().to_bits().hash(&mut hasher);
                    draw_pan.x.to_bits().hash(&mut hasher);
                    draw_pan.y.to_bits().hash(&mut hasher);
                    self.zoom.to_bits().hash(&mut hasher);
                    hasher.finish()
                };
                let blockers_by_node = if self.occlusion_key == Some(occlusion_key) {
                    self.occlusion.clone().unwrap_or_default()
                } else {
                    let map = occlusion_map(&visible_nodes, &self.z_order, canvas, draw_pan, self.zoom);
                    self.occlusion = Some(map.clone());
                    self.occlusion_key = Some(occlusion_key);
                    map
                };
                for node in &visible_nodes {
                    let rect = node_rect(node, canvas, draw_pan, self.zoom);
                    let text_size = 11.0 * self.zoom;
                    if text_size < 7.0 {
                        break;
                    }
                    let text_width = (node.node_type.chars().count() as f32 * text_size * 0.62)
                        .max(text_size * 2.0);
                    let text_top = rect.max.y - 24.0 * self.zoom;
                    let label_rect = Rect::from_min_size(
                        Pos2::new(rect.center().x - text_width * 0.5, text_top - text_size * 0.5),
                        Vec2::new(text_width, text_size * 1.5),
                    );
                    let blockers = blockers_by_node.get(&node.id);
                    if blockers.map_or(true, Vec::is_empty) {
                        let fade = self.node_fade_alpha(&node.id, now);
                        draw.texts.push(NodeText {
                            text: node.node_type.to_uppercase(),
                            center_x: rect.center().x,
                            top: rect.max.y - 24.0 * self.zoom,
                            size: 11.0 * self.zoom,
                            color: fade_color(self.theme.text, fade),
                            bounds: glyphon::TextBounds::default(),
                        });
                    } else {
                        overlay_texts.push((
                            node.node_type.to_uppercase(),
                            Pos2::new(rect.center().x, text_top),
                            text_size,
                            label_rect,
                            blockers.cloned().unwrap_or_default(),
                            self.node_fade_alpha(&node.id, now),
                        ));
                    }
                }
                if self.gpu_ready {
                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(ctx.content_rect(), NodeCallback(draw)));
                }

                for (text, position, size, label_rect, blockers, fade) in &overlay_texts {
                    for part in subtract_rects(*label_rect, blockers) {
                        painter.with_clip_rect(part).text(
                            *position,
                            egui::Align2::CENTER_TOP,
                            text,
                            egui::FontId::proportional(*size),
                            fade_color32(self.theme.color(self.theme.text), *fade),
                        );
                    }
                }

                // CPU fallback if the wgpu node pipeline could not initialize.
                if !self.gpu_ready {
                    for index in ordered_cable_connections(document, selected_nodes, selected_connections) {
                        let connection = &document.connections[index];
                        let Some(from) = document.nodes.iter().find(|node| node.id == connection.from) else { continue };
                        let Some(to) = document.nodes.iter().find(|node| node.id == connection.to) else { continue };
                        let start = node_rect(from, canvas, draw_pan, self.zoom);
                        let end = node_rect(to, canvas, draw_pan, self.zoom);
                        let outputs = node_outputs(from);
                        let inputs = node_inputs(to);
                        let x0 = port_x(start, port_index(outputs, &connection.out), outputs.len(), self.zoom);
                        let y0 = start.max.y;
                        let x1 = port_x(end, port_index(inputs, &connection.input), inputs.len(), self.zoom);
                        let y1 = end.min.y;
                        let points = wire_points(x0, y0, x1, y1, self.zoom);
                        let key = ConnectionKey::from_connection(connection);
                        painter.add(egui::Shape::line(
                            points.to_vec(),
                            Stroke::new(18.0, self.theme.with_alpha(self.theme.window, 191)),
                        ));
                        painter.add(egui::Shape::line(
                            points.to_vec(),
                            Stroke::new(
                                7.0,
                                self.theme.color(if selected_connections.contains(&key) {
                                    self.theme.accent
                                } else {
                                    self.theme.cable
                                }),
                            ),
                        ));
                    }
                    for node in &document.nodes {
                        let rect = node_rect(node, canvas, draw_pan, self.zoom);
                        if !rect.intersects(canvas) {
                            continue;
                        }
                        let selected_here = selected_nodes.contains(node.id.as_str());
                        painter.add(crate::squircle::shape(
                            rect,
                            26.0 * self.zoom,
                            self.theme.color(self.theme.card),
                            Stroke::new(
                                if selected_here { 3.0 } else { 2.0 },
                                self.theme.color(if selected_here {
                                    self.theme.accent
                                } else {
                                    self.theme.card_border
                                }),
                            ),
                        ));
                        for (index, port) in node_inputs(node).iter().enumerate() {
                            let x = port_x(rect, index, node_inputs(node).len(), self.zoom);
                            painter.circle_filled(
                                Pos2::new(x, rect.min.y),
                                8.0 * self.zoom,
                                self.theme.color(self.theme.card),
                            );
                            painter.circle_stroke(
                                Pos2::new(x, rect.min.y),
                                8.0 * self.zoom,
                                Stroke::new(
                                    2.0,
                                    self.theme.color(if inlet_is_data(port) {
                                        self.theme.port_input
                                    } else {
                                        self.theme.accent
                                    }),
                                ),
                            );
                        }
                        for (port, _) in node_outputs(node).iter().enumerate() {
                            let x = port_x(rect, port, node_outputs(node).len(), self.zoom);
                            painter.circle_filled(
                                Pos2::new(x, rect.max.y),
                                8.0 * self.zoom,
                                self.theme.color(self.theme.card),
                            );
                            painter.circle_stroke(
                                Pos2::new(x, rect.max.y),
                                8.0 * self.zoom,
                                Stroke::new(2.0, self.theme.color(self.theme.accent)),
                            );
                        }
                        painter.text(
                            Pos2::new(rect.center().x, rect.max.y - 24.0 * self.zoom),
                            egui::Align2::CENTER_CENTER,
                            node.node_type.to_uppercase(),
                            egui::FontId::proportional(11.0 * self.zoom),
                            {
                                let fade = self.node_fade_alpha(&node.id, now);
                                fade_color32(self.theme.color(self.theme.text), fade)
                            },
                        );
                    }
                }

                // Port captions are immediate-mode so they work for both the GPU
                // card draw and the CPU fallback.
                let port_label_size = 9.0 * self.zoom;
                for node in &visible_nodes {
                    if node.node_type != "loft" && node.node_type != "surfaceloft" {
                        continue;
                    }
                    if port_label_size < 7.0 {
                        break;
                    }
                    let rect = node_rect(node, canvas, draw_pan, self.zoom);
                    for (index, port) in node_inputs(node).iter().enumerate() {
                        let x = port_x(rect, index, node_inputs(node).len(), self.zoom);
                        painter.text(
                            Pos2::new(x, rect.min.y - 8.0 * self.zoom.max(0.6)),
                            egui::Align2::CENTER_BOTTOM,
                            *port,
                            egui::FontId::proportional((9.0 * self.zoom).clamp(8.0, 12.0)),
                            self.theme.color(self.theme.muted),
                        );
                    }
                }

                // Node-type icons sit above the card, below the divider. The SVG
                // is rasterized once per type into a texture; per-frame icon work
                // is then a single textured quad, not an SVG parse for every node.
                for node in &visible_nodes {
                    let rect = node_rect(node, canvas, draw_pan, self.zoom);
                    let size = 56.0 * self.zoom;
                    if size < 8.0 {
                        break;
                    }
                    let icon_rect = Rect::from_center_size(
                        Pos2::new(rect.center().x, rect.center().y - 4.0 * self.zoom),
                        Vec2::splat(size),
                    );
                    let blockers = blockers_by_node
                        .get(&node.id)
                        .cloned()
                        .unwrap_or_default();
                    let (name, source) = icon_for(node);
                    let node_fade = self.node_fade_alpha(&node.id, now);
                    let texture = self.icon_texture(ui.ctx(), name, source, self.theme.text);
                    // Large, low-opacity silhouette, clipped against any card
                    // that is in front of this node.
                    let visible = subtract_rects(icon_rect, &blockers);
                    for part in visible {
                        painter.with_clip_rect(part).image(
                            texture.id(),
                            icon_rect,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::from_rgba_unmultiplied(255, 255, 255, (150.0 * node_fade) as u8),
                        );
                    }
                    let (suppress_center, view_center) = button_centers(rect);
                    for (name, source, center, size) in [
                        (
                            "power",
                            include_bytes!("../../../../vendor/fluent/power.svg") as &'static [u8],
                            suppress_center,
                            16.0 * self.zoom,
                        ),
                        (
                            "eye",
                            include_bytes!("../../../../vendor/fluent/eye.svg") as &'static [u8],
                            view_center,
                            34.0 * self.zoom,
                        ),
                    ] {
                        let texture = self.icon_texture(ui.ctx(), name, source, self.theme.text);
                        let button_icon = Rect::from_center_size(center, Vec2::splat(size));
                        for part in subtract_rects(button_icon, &blockers) {
                            painter.with_clip_rect(part).image(
                                texture.id(),
                                button_icon,
                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::from_rgba_unmultiplied(255, 255, 255, (255.0 * node_fade) as u8),
                            );
                        }
                    }
                }

                // Vertical grab divider. Interaction is created after the canvas so
                // dragging the strip resizes the panel rather than panning the graph.
                let divider = Rect::from_min_max(
                    Pos2::new(panel.max.x - 4.0, panel.min.y + 12.0),
                    Pos2::new(panel.max.x + 4.0, panel.max.y - 12.0),
                );
                let resize = ui.interact(divider, egui::Id::new("node-editor-resize"), Sense::drag());
                if resize.hovered() || resize.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
                if resize.drag_started() {
                    // Anchor the drag to the pointer, not to per-frame deltas. If the
                    // pointer overshoots min/max, moving back must first recover that
                    // overshoot before the divider moves again.
                    self.resize_start = resize
                        .interact_pointer_pos()
                        .map(|pointer| (panel.width(), pointer.x));
                }
                if resize.dragged() {
                    if let Some((start_width, start_pointer)) = self.resize_start {
                        if let Some(pointer) = resize.interact_pointer_pos() {
                            self.width = (start_width + (pointer.x - start_pointer)).clamp(
                                PANEL_MIN,
                                (viewport.width() * PANEL_MAX_FRACTION).max(PANEL_MIN),
                            );
                        }
                    }
                }
                if resize.drag_stopped() {
                    self.resize_start = None;
                }
                // The inspector's edge grip, mirrored onto the panel's right side.
                let grip = if resize.hovered() || resize.dragged() {
                    self.theme.color(self.theme.highlight)
                } else {
                    self.theme.with_alpha(self.theme.muted, 70)
                };
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(panel.max.x - 1.0, panel.center().y - 26.0),
                        Pos2::new(panel.max.x + 1.0, panel.center().y + 26.0),
                    ),
                    egui::CornerRadius::same(1),
                    grip,
                );

                // Swift-editor minimap: bottom-right, square, dark rounded card
                // showing the currently loaded patch and the viewport rectangle.
                painter.add(crate::squircle::shape(
                    mini,
                    5.0,
                    self.theme.color(self.theme.minimap_bg),
                    Stroke::new(1.0, self.theme.color(self.theme.minimap_border)),
                ));

                if !document.nodes.is_empty() {
                    let (mut min_x, mut min_y) = (f32::INFINITY, f32::INFINITY);
                    let (mut max_x, mut max_y) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for node in &document.nodes {
                        let r = graph_rect(node);
                        min_x = min_x.min(r.min.x);
                        min_y = min_y.min(r.min.y);
                        max_x = max_x.max(r.max.x);
                        max_y = max_y.max(r.max.y);
                    }
                    let graph_w = (max_x - min_x).max(1.0);
                    let graph_h = (max_y - min_y).max(1.0);
                    let pad = 8.0;
                    let scale = ((mini.width() - 2.0 * pad) / graph_w)
                        .min((mini.height() - 2.0 * pad) / graph_h)
                        .max(0.001);
                    let graph_center = Vec2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
                    let offset = mini.center().to_vec2() - graph_center * scale;
                    // `offset` is already an absolute screen position: do not add
                    // mini.min again or the patch lands outside the minimap.
                    let map = |p: Pos2| offset + Vec2::new(p.x, p.y) * scale;
                    let stride = (document.nodes.len() / 2048).max(1);
                    for (index, node) in document.nodes.iter().enumerate().step_by(stride) {
                        let r = graph_rect(node);
                        let a = map(r.min);
                        let b = map(r.max);
                        let rect = Rect::from_min_max(Pos2::new(a.x, a.y), Pos2::new(b.x, b.y));
                        let selected_here = selected_nodes.contains(document.nodes[index].id.as_str());
                        painter.add(crate::squircle::shape(
                            rect,
                            2.5,
                            if selected_here {
                                self.theme.color(self.theme.accent)
                            } else {
                                self.theme.color(self.theme.minimap_node)
                            },
                            Stroke::NONE,
                        ));
                    }
                    let view_min = map(Pos2::new(
                        (canvas.min.x - canvas.min.x - self.pan.x) / self.zoom,
                        (canvas.min.y - canvas.min.y - self.pan.y) / self.zoom,
                    ));
                    let view_max = map(Pos2::new(
                        (canvas.max.x - canvas.min.x - self.pan.x) / self.zoom,
                        (canvas.max.y - canvas.min.y - self.pan.y) / self.zoom,
                    ));
                    let view_rect = Rect::from_min_max(
                        Pos2::new(view_min.x.min(view_max.x), view_min.y.min(view_max.y)),
                        Pos2::new(view_min.x.max(view_max.x), view_min.y.max(view_max.y)),
                    )
                    .intersect(mini);
                    if view_rect.is_positive() {
                        painter.rect_stroke(
                            view_rect,
                            egui::CornerRadius::ZERO,
                            Stroke::new(1.0, self.theme.color(self.theme.accent)),
                            egui::StrokeKind::Inside,
                        );
                    }

                    // Canvas zoom is relative to 1 graph unit = 1 egui point.
                    // Keep the readout inside the minimap's top-right corner.
                    let percent = format!("{}%", zoom_percent(self.zoom));
                    let font = egui::FontId::proportional(14.0);
                    let galley = painter.layout_no_wrap(
                        percent,
                        font,
                        self.theme.color(self.theme.text),
                    );
                    let text_pos = egui::pos2(
                        mini.max.x - 9.0 - galley.size().x,
                        mini.max.y - 9.0 - galley.size().y,
                    );
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            text_pos - egui::vec2(5.0, 3.0),
                            galley.size() + egui::vec2(10.0, 6.0),
                        ),
                        egui::CornerRadius::same(5),
                        self.theme.with_alpha(self.theme.window, 200),
                    );
                    painter.galley(text_pos, galley, self.theme.color(self.theme.text));

                    let minimap = ui.interact(mini, egui::Id::new("node-editor-minimap"), Sense::click_and_drag());
                    if minimap.clicked() {
                        chrome_clicked = true;
                    }
                    if minimap.hovered() || minimap.dragged() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if let Some(pointer) = minimap.interact_pointer_pos() {
                        if minimap.drag_started() {
                            // Anchor a minimap drag to the viewport's current pan so
                            // grabbing the view rect never jumps it under the cursor.
                            self.minimap_drag = Some((pointer, self.pan));
                        }
                        if minimap.dragged() {
                            if let Some((start_pointer, start_pan)) = self.minimap_drag {
                                let delta = pointer - start_pointer;
                                self.pan = start_pan
                                    - Vec2::new(delta.x, delta.y) / scale * self.zoom;
                                ui.ctx().request_repaint();
                            }
                        } else if minimap.clicked() {
                            let graph = (pointer - offset).to_vec2() / scale;
                            self.pan = canvas.center() - canvas.min - graph * self.zoom;
                            ui.ctx().request_repaint();
                        }
                    }
                    if minimap.drag_stopped() {
                        self.minimap_drag = None;
                    }
                }

                // Zoom-to-fit above the minimap, matching the viewport home control:
                // icon only, circular hover card, right edge aligned to the minimap.
                let fit_icon_size = 18.0;
                let fit_icon = Rect::from_min_size(
                    Pos2::new(mini.max.x - fit_icon_size, mini.min.y - 10.0 - fit_icon_size),
                    Vec2::splat(fit_icon_size),
                );
                let fit_rect = Rect::from_center_size(fit_icon.center(), Vec2::splat(30.0));
                let fit = ui
                    .interact(fit_rect, egui::Id::new("node-editor-fit"), Sense::click())
                    .on_hover_text("Zoom to fit");
                if fit.hovered() {
                    painter.circle_filled(
                        fit_icon.center(),
                        14.0,
                        self.theme.with_alpha(self.theme.highlight, 26),
                    );
                    painter.circle_stroke(
                        fit_icon.center(),
                        14.0,
                        Stroke::new(1.0, self.theme.with_alpha(self.theme.muted, 100)),
                    );
                }
                if fit.hovered() || fit.clicked() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if fit.clicked() {
                    chrome_clicked = true;
                    self.animate_fit(document);
                    ui.ctx().request_repaint();
                }
                let fit_icon_colour = if fit.hovered() { hex_colour(self.theme.icon_hover) } else { hex_colour(self.theme.icon) };
                let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/zoom_fit.svg"))
                    .unwrap_or_default()
                    .replace("#b4b4b4", &fit_icon_colour);
                let image = egui::Image::from_bytes("bytes://nodecraft-zoom-fit.svg", svg.into_bytes())
                    .fit_to_exact_size(Vec2::splat(fit_icon_size));
                ui.put(fit_icon, image);

                // Cleanup above fit. Left click runs the current layout mode;
                // right click opens the Timeline/Centred/Organic mode menu.
                let cleanup_size = 18.0;
                let cleanup_icon = Rect::from_center_size(
                    Pos2::new(
                        fit_icon.center().x,
                        fit_icon.min.y - 18.0 - cleanup_size * 0.5,
                    ),
                    Vec2::splat(cleanup_size),
                );
                let cleanup_rect = Rect::from_center_size(cleanup_icon.center(), Vec2::splat(30.0));
                let cleanup = ui
                    .interact(cleanup_rect, egui::Id::new("node-editor-cleanup"), Sense::click())
                    .on_hover_text(format!("Clean up layout ({}) · right-click for mode", self.layout_mode.label()));
                if cleanup.hovered() || cleanup.dragged() {
                    painter.circle_filled(
                        cleanup_icon.center(),
                        14.0,
                        self.theme.with_alpha(self.theme.highlight, 26),
                    );
                    painter.circle_stroke(
                        cleanup_icon.center(),
                        14.0,
                        Stroke::new(1.0, self.theme.with_alpha(self.theme.muted, 100)),
                    );
                }
                if cleanup.hovered() || cleanup.clicked() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if cleanup.clicked() {
                    chrome_clicked = true;
                    actions.push(NodeEditorAction::CleanupLayout { mode: self.layout_mode });
                    ui.ctx().request_repaint();
                }
                let toggled_layout_menu = cleanup.secondary_clicked();
                if toggled_layout_menu {
                    self.layout_menu_open = !self.layout_menu_open;
                }
                let cleanup_icon_colour = if cleanup.hovered() { hex_colour(self.theme.icon_hover) } else { hex_colour(self.theme.icon) };
                let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/broom.svg"))
                    .unwrap_or_default()
                    .replace("#b4b4b4", &cleanup_icon_colour);
                let image = egui::Image::from_bytes("bytes://nodecraft-layout-cleanup.svg", svg.into_bytes())
                    .fit_to_exact_size(Vec2::splat(cleanup_size));
                ui.put(cleanup_icon, image);

                if self.layout_menu_open {
                    let menu_width = layout_menu_width(ui);
                    let menu_height = 3.0 * 26.0 + 2.0 * 2.0 + 16.0;
                    let menu_pos = Pos2::new(
                        cleanup_icon.min.x - menu_width - 8.0,
                        cleanup_icon.max.y + 4.0,
                    );
                    let mut chosen = None;
                    let menu_response = egui::Area::new(egui::Id::new("node-layout-mode-menu"))
                        .order(egui::Order::Foreground)
                        .fixed_pos(menu_pos)
                        .show(ui.ctx(), |ui| {
                            let frame = egui::Rect::from_min_size(
                                menu_pos,
                                egui::vec2(menu_width, menu_height),
                            );
                            ui.painter().add(crate::squircle::shape(
                                frame,
                                10.0,
                                self.theme.color(self.theme.popup),
                                egui::Stroke::new(
                                    1.0,
                                    self.theme.color(self.theme.border),
                                ),
                            ));
                            ui.scope_builder(egui::UiBuilder::new().max_rect(frame.shrink2(egui::vec2(8.0, 8.0))), |ui| {
                                ui.spacing_mut().item_spacing = egui::vec2(6.0, 2.0);
                                ui.set_min_width(menu_width - 16.0);
                                for mode in LayoutMode::ALL {
                                    let response = layout_menu_row(
                                        ui,
                                        &self.theme,
                                        self.layout_mode == mode,
                                        mode.label(),
                                    );
                                    if response.clicked() {
                                        chosen = Some(mode);
                                    }
                                }
                            });
                        });
                    if !toggled_layout_menu && menu_response.response.clicked_elsewhere() {
                        self.layout_menu_open = false;
                    }
                    if let Some(mode) = chosen {
                        self.layout_mode = mode;
                        self.layout_menu_open = false;
                        actions.push(NodeEditorAction::CleanupLayout { mode });
                        ui.ctx().request_repaint();
                    }
                }
                if let Some((start, end)) = self.lasso {
                    let rect = Rect::from_two_pos(start, end);
                    painter.add(crate::squircle::shape(
                        rect,
                        0.0,
                        Color32::from_rgba_unmultiplied(10, 132, 255, 26),
                        Stroke::new(1.0, Color32::from_rgba_unmultiplied(10, 132, 255, 190)),
                    ));
                }
                if let Some(drag) = &self.wire_drag {
                    let mut end = drag.current;
                    if let Some(target) = ui.ctx().pointer_latest_pos().and_then(|pointer| {
                        port_at(document, canvas, self.pan, self.zoom, pointer)
                    }) {
                        if port_compatible(&drag.origin, &target) {
                            end = target.position;
                        }
                    }
                    if drag.origin.is_input {
                        // Cables are directed output → input even when the drag
                        // started at an input port; keep the preview in that
                        // direction.
                        dashed_curve(&painter, end, drag.origin.position, self.zoom);
                    } else {
                        dashed_curve(&painter, drag.origin.position, end, self.zoom);
                    }
                }
                if let Some(preview) = &self.insert_preview {
                    if preview.progress > 0.01 {
                        let candidate = &preview.candidate;
                        let source = document.nodes.iter().find(|node| node.id == candidate.connection.from);
                        let target = document.nodes.iter().find(|node| node.id == candidate.connection.to);
                        let inserted = document.nodes.iter().find(|node| node.id == candidate.node);
                        if let (Some(source), Some(target), Some(inserted)) = (source, target, inserted) {
                            let source_rect = node_rect(source, canvas, draw_pan, self.zoom);
                            let target_rect = node_rect(target, canvas, draw_pan, self.zoom);
                            let inserted_rect = node_rect(inserted, canvas, draw_pan, self.zoom);
                            let source_out = Pos2::new(
                                port_x(source_rect, port_index(node_outputs(source), &candidate.connection.out), node_outputs(source).len(), self.zoom),
                                source_rect.max.y,
                            );
                            let inserted_in = Pos2::new(
                                port_x(inserted_rect, port_index(node_inputs(inserted), &candidate.input), node_inputs(inserted).len(), self.zoom),
                                inserted_rect.min.y,
                            );
                            let inserted_out = Pos2::new(
                                port_x(inserted_rect, port_index(node_outputs(inserted), &candidate.out), node_outputs(inserted).len(), self.zoom),
                                inserted_rect.max.y,
                            );
                            let target_in = Pos2::new(
                                port_x(target_rect, port_index(node_inputs(target), &candidate.connection.input), node_inputs(target).len(), self.zoom),
                                target_rect.min.y,
                            );
                            let alpha = (preview.progress * 255.0) as u8;
                            let colour = Color32::from_rgba_unmultiplied(124, 156, 255, alpha);
                            let phase = ui.ctx().input(|input| input.time) as f32 * 42.0;
                            dashed_curve_with_colour(&painter, source_out, inserted_in, self.zoom, colour, phase);
                            dashed_curve_with_colour(&painter, inserted_out, target_in, self.zoom, colour, phase);
                        }
                    }
                }
                empty_canvas_clicked = response.clicked() && !chrome_clicked && clicked.is_none();
            });

        if let Some(positions) = moved_nodes {
            actions.push(NodeEditorAction::MoveNodes { positions });
        }
        if let Some(id) = clicked {
            self.bring_to_front(&id);
            let additive = ctx.input(|i| i.modifiers.command || i.modifiers.shift);
            actions.push(NodeEditorAction::SelectNode { id, additive });
        } else if let Some(pointer) = ctx.pointer_latest_pos() {
            if panel.contains(pointer) && ctx.input(|i| i.pointer.primary_clicked()) {
                let additive = ctx.input(|i| i.modifiers.command || i.modifiers.shift);
                if let Some(key) = cable_hit(document, panel, self.pan, self.zoom, pointer) {
                    actions.push(NodeEditorAction::SelectCable { key, additive });
                } else if empty_canvas_clicked {
                    actions.push(NodeEditorAction::Deselect);
                }
            }
        } else if empty_canvas_clicked {
            actions.push(NodeEditorAction::Deselect);
        }
        actions
    }

    fn sync_z_order(&mut self, document: &GraphDocument) {
        let ids: HashSet<&str> = document.nodes.iter().map(|node| node.id.as_str()).collect();
        self.z_order.retain(|id| ids.contains(id.as_str()));
        let mut seen: HashSet<String> = self.z_order.iter().cloned().collect();
        for node in &document.nodes {
            if seen.insert(node.id.clone()) {
                self.z_order.push(node.id.clone());
            }
        }
    }

    fn bring_to_front(&mut self, id: &str) {
        if let Some(index) = self.z_order.iter().position(|candidate| candidate == id) {
            let id = self.z_order.remove(index);
            self.z_order.push(id);
        }
    }

    /// Smooth zoom-to-fit for the current document.
    pub fn animate_fit(&mut self, document: &GraphDocument) {
        if let Some(roi) = document_roi(document) {
            self.animate_to_roi(roi, Duration::from_millis(450));
        }
    }

    fn tick_pan_animation(&mut self, ctx: &egui::Context) {
        let Some(animation) = &self.pan_animation else { return };
        let progress = (animation.started.elapsed().as_secs_f32()
            / animation.duration.as_secs_f32().max(0.001))
        .clamp(0.0, 1.0);
        let eased = progress * progress * (3.0 - 2.0 * progress);
        self.pan = animation.start_pan + (animation.target_pan - animation.start_pan) * eased;
        if progress >= 1.0 {
            self.pan_animation = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn update_node_fades(&mut self, document: &GraphDocument) {
        let now = Instant::now();
        if self.fade_document.as_deref() != Some(document.id.as_str()) {
            self.fade_document = Some(document.id.clone());
            self.known_nodes = document.nodes.iter().map(|node| node.id.clone()).collect();
            self.node_fades.clear();
            return;
        }
        for node in &document.nodes {
            if self.known_nodes.insert(node.id.clone()) {
                self.node_fades.insert(node.id.clone(), now);
            }
        }
        let live: HashSet<&str> = document.nodes.iter().map(|node| node.id.as_str()).collect();
        self.node_fades.retain(|id, _| live.contains(id.as_str()));
    }

    fn node_fade_alpha(&self, id: &str, now: Instant) -> f32 {
        let Some(started) = self.node_fades.get(id) else { return 1.0 };
        let t = (now.duration_since(*started).as_secs_f32()
            / NODE_FADE_IN.as_secs_f32().max(0.001))
        .clamp(0.0, 1.0);
        // Smoothstep; change NODE_FADE_IN for timing and this for the curve.
        t * t * (3.0 - 2.0 * t)
    }

    /// Start a fade for a node the app has just added, so the fade and the
    /// view-pan animation begin on the same frame.
    pub fn begin_node_fade(&mut self, id: &str) {
        self.known_nodes.insert(id.to_owned());
        self.node_fades.insert(id.to_owned(), Instant::now());
    }

    fn node_fade_map(&self, document: &GraphDocument, now: Instant) -> HashMap<String, f32> {
        document
            .nodes
            .iter()
            .filter_map(|node| {
                let alpha = self.node_fade_alpha(&node.id, now);
                (alpha < 1.0).then(|| (node.id.clone(), alpha))
            })
            .collect()
    }

    fn tick_view_animation(&mut self, canvas: Rect, ctx: &egui::Context) {
        let Some(animation) = &self.view_animation else { return };
        let progress = (animation.started.elapsed().as_secs_f32()
            / animation.duration.as_secs_f32().max(0.001))
        .clamp(0.0, 1.0);
        let eased = progress * progress * (3.0 - 2.0 * progress);
        let (target_pan, target_zoom) = fitted_view(canvas, animation.roi);
        self.zoom = animation.start_zoom
            * (target_zoom / animation.start_zoom.max(0.001)).powf(eased);
        self.pan = animation.start_pan + (target_pan - animation.start_pan) * eased;
        if progress >= 1.0 {
            self.view_animation = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn fit_graph(&mut self, canvas: Rect, document: &GraphDocument) {
        let Some((mut min_x, mut min_y)) = document
            .nodes
            .iter()
            .map(|node| (node.x as f32 - NODE_LEFT_PADDING, node.y as f32))
            .reduce(|a, b| (a.0.min(b.0), a.1.min(b.1)))
        else {
            return;
        };
        let mut max_x = min_x + node_width(&document.nodes[0]) + NODE_RIGHT_PADDING;
        let mut max_y = min_y + NODE_HEIGHT;
        for node in &document.nodes {
            max_x = max_x.max(node.x as f32 + node_width(node) + NODE_RIGHT_PADDING);
            max_y = max_y.max(node.y as f32 + NODE_HEIGHT);
            min_x = min_x.min(node.x as f32 - NODE_LEFT_PADDING);
            min_y = min_y.min(node.y as f32);
        }
        let (pan, zoom) = fitted_view(
            canvas,
            Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y)),
        );
        self.pan = pan;
        self.zoom = zoom;
        
    }

    fn zoom_about(&mut self, canvas: Rect, anchor: Pos2, factor: f32) {
        let next = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        if (next - self.zoom).abs() < f32::EPSILON {
            return;
        }
        let local = anchor - canvas.min;
        let scale = next / self.zoom;
        // Keep the graph point under the pointer stationary while zooming.
        self.pan = local - (local - self.pan) * scale;
        self.zoom = next;
    }
}

fn occlusion_map(
    nodes: &[&nodecraft_session::GraphNode],
    z_order: &[String],
    canvas: Rect,
    pan: Vec2,
    zoom: f32,
) -> HashMap<String, Vec<Rect>> {
    if nodes.len() < 2 {
        return HashMap::new();
    }
    let ranks: HashMap<&str, usize> = z_order
        .iter()
        .enumerate()
        .map(|(rank, id)| (id.as_str(), rank))
        .collect();
    let cards: Vec<(usize, Rect)> = nodes
        .iter()
        .map(|node| {
            (
                ranks.get(node.id.as_str()).copied().unwrap_or(usize::MAX),
                node_rect(node, canvas, pan, zoom),
            )
        })
        .collect();
    // A card's left/right buttons protrude outside the card. Use those expanded
    // bounds for overlap detection; otherwise a back node's icon can draw over
    // a front node's eye/power button.
    let bounds: Vec<Rect> = nodes
        .iter()
        .map(|node| node_bounds(node, canvas, pan, zoom))
        .collect();

    // Cards only occlude nearby cards. Hash them into screen-space cells sized
    // to the card itself, then compare each card with its local candidates.
    // Dense overlapping layouts still fall back toward pairwise work, but the
    // common sparse graph no longer pays an O(N²) price every frame.
    let cell = (NODE_MIN_WIDTH * zoom).max(16.0);
    let cell_of = |p: Pos2| ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32);
    let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (index, bound) in bounds.iter().enumerate() {
        let (x0, y0) = cell_of(bound.min);
        let (x1, y1) = cell_of(bound.max);
        for y in y0..=y1 {
            for x in x0..=x1 {
                grid.entry((x, y)).or_default().push(index);
            }
        }
    }

    let mut map: HashMap<String, Vec<Rect>> = HashMap::new();
    let mut seen: HashSet<usize> = HashSet::new();
    for (index, node) in nodes.iter().enumerate() {
        let (rank, card) = cards[index];
        let bound = bounds[index];
        let mut candidates: Vec<usize> = Vec::new();
        let (x0, y0) = cell_of(bound.min);
        let (x1, y1) = cell_of(card.max);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if let Some(indices) = grid.get(&(x, y)) {
                    for &candidate in indices {
                        if seen.insert(candidate) {
                            candidates.push(candidate);
                        }
                    }
                }
            }
        }
        for candidate in candidates {
            let (front_rank, front) = cards[candidate];
            if front_rank > rank && bounds[candidate].intersects(bound) {
                let blockers = map.entry(node.id.clone()).or_default();
                blockers.push(front);
                let (power, eye) = button_centers(front);
                blockers.push(Rect::from_center_size(
                    power,
                    Vec2::splat(BUTTON_RADIUS * 2.0 * zoom),
                ));
                blockers.push(Rect::from_center_size(
                    eye,
                    Vec2::splat(VIEW_BUTTON_RADIUS * 2.0 * zoom),
                ));
            }
        }
        // The candidate set is indexed by card index, so clearing by walking the
        // touched cells is cheaper than allocating a fresh HashSet per card.
        let (x0, y0) = cell_of(bound.min);
        let (x1, y1) = cell_of(bound.max);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if let Some(indices) = grid.get(&(x, y)) {
                    for &candidate in indices {
                        seen.remove(&candidate);
                    }
                }
            }
        }
    }
    map
}

fn subtract_rects(area: Rect, blockers: &[Rect]) -> Vec<Rect> {
    let mut parts = vec![area];
    for blocker in blockers {
        let mut next = Vec::new();
        for part in parts {
            if !part.intersects(*blocker) {
                next.push(part);
                continue;
            }
            let overlap = part.intersect(*blocker);
            if overlap.min.x > part.min.x {
                next.push(Rect::from_min_max(part.min, Pos2::new(overlap.min.x, part.max.y)));
            }
            if overlap.max.x < part.max.x {
                next.push(Rect::from_min_max(Pos2::new(overlap.max.x, part.min.y), part.max));
            }
            if overlap.min.y > part.min.y {
                next.push(Rect::from_min_max(
                    Pos2::new(overlap.min.x, part.min.y),
                    Pos2::new(overlap.max.x, overlap.min.y),
                ));
            }
            if overlap.max.y < part.max.y {
                next.push(Rect::from_min_max(
                    Pos2::new(overlap.min.x, overlap.max.y),
                    Pos2::new(overlap.max.x, part.max.y),
                ));
            }
        }
        parts = next;
        if parts.is_empty() {
            break;
        }
    }
    parts
}

fn node_bounds(node: &nodecraft_session::GraphNode, canvas: Rect, pan: Vec2, zoom: f32) -> Rect {
    let rect = node_rect(node, canvas, pan, zoom);
    Rect::from_min_max(
        rect.min - Vec2::new(NODE_LEFT_PADDING * zoom, 0.0),
        rect.max + Vec2::new(NODE_RIGHT_PADDING * zoom, 0.0),
    )
}

fn lasso_selection(
    document: &GraphDocument,
    canvas: Rect,
    pan: Vec2,
    zoom: f32,
    lasso: Rect,
    base: &HashSet<String>,
    additive: bool,
) -> HashSet<String> {
    let mut selection: HashSet<String> = if additive { base.clone() } else { HashSet::new() };
    for node in &document.nodes {
        if !node_bounds(node, canvas, pan, zoom).intersects(lasso) {
            continue;
        }
        if additive {
            if !selection.remove(&node.id) {
                selection.insert(node.id.clone());
            }
        } else {
            selection.insert(node.id.clone());
        }
    }
    selection
}

fn wire_points(x0: f32, y0: f32, x1: f32, y1: f32, zoom: f32) -> [Pos2; 4] {
    let route = crate::routing::route(x0, y0, x1, y1, zoom);
    [
        Pos2::new(x0, y0),
        route.control1,
        route.control2,
        Pos2::new(x1, y1),
    ]
}

fn colorized_icon_svg(source: &[u8], theme_colour: [u8; 4]) -> String {
    let colour = hex_colour(theme_colour);
    let mut svg = std::str::from_utf8(source)
        .unwrap_or_default()
        .replace("#212121", &colour)
        .replace("#b4b4b4", &colour)
        .replace("#d8d8dc", &colour);
    // Fluent outline SVGs often rely on the SVG default black fill. Inject the
    // theme colour on the root only when there is no explicit root paint.
    if let Some(root_end) = svg.find('>') {
        let root = &svg[..root_end];
        if !root.contains("fill=") && !root.contains("stroke=") {
            svg.insert_str(root_end, &format!(" fill=\"{colour}\""));
        }
    }
    svg
}

fn primitive_icon(primitive: &str) -> &'static str {
    match primitive {
        "sphere" | "partialSphere" => "globe",
        "cylinder" => "database",
        "torus" | "partialTorus" => "circle",
        "cone" | "pyramid" | "wedge" => "triangle",
        _ => "cube",
    }
}

fn icon_for(node: &nodecraft_session::GraphNode) -> (&'static str, &'static [u8]) {
    let primitive = node
        .values
        .get("primitiveType")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let name = match node.node_type.as_str() {
        "loft" | "surfaceloft" | "extrude" => "arrow_up",
        "sketch" => "edit",
        "spline" => "spline",
        "chamfer" | "fillet" => "subtract",
        "shell" => "square_multiple",
        "primitive" => primitive_icon(primitive),
        "box" => "cube",
        "cylinder" => "database",
        "cone" | "wedge" => "triangle",
        "sphere" => "globe",
        "torus" => "circle",
        "translate" => "arrow_move",
        "mirror" => "mirror",
        "boolean" | "fuse" | "combine" => "merge",
        "exporter" => "arrow_download",
        "stepimporter" | "linkedpart" => "import_step",
        _ => "cube",
    };
    let bytes: &'static [u8] = match name {
        "arrow_up" => include_bytes!("../../../../vendor/fluent/arrow_up.svg"),
        "edit" => include_bytes!("../../../../vendor/fluent/edit.svg"),
        "spline" => include_bytes!("../../../../vendor/fluent/spline.svg"),
        "subtract" => include_bytes!("../../../../vendor/fluent/subtract.svg"),
        "square_multiple" => include_bytes!("../../../../vendor/fluent/square_multiple.svg"),
        "database" => include_bytes!("../../../../vendor/fluent/database.svg"),
        "triangle" => include_bytes!("../../../../vendor/fluent/triangle.svg"),
        "globe" => include_bytes!("../../../../vendor/fluent/globe.svg"),
        "circle" => include_bytes!("../../../../vendor/fluent/circle.svg"),
        "arrow_move" => include_bytes!("../../../../vendor/fluent/arrow_move.svg"),
        "mirror" => include_bytes!("../../../../vendor/fluent/mirror.svg"),
        "merge" => include_bytes!("../../../../vendor/fluent/merge.svg"),
        "arrow_download" => include_bytes!("../../../../vendor/fluent/arrow_download.svg"),
        "import_step" => include_bytes!("../../../../vendor/fluent/import_step.svg"),
        _ => include_bytes!("../../../../vendor/fluent/cube.svg"),
    };
    (name, bytes)
}

fn node_title(node_type: &str) -> &'static str {
    match node_type {
        "primitive" => "Solid",
        "sketch" => "Sketch",
        "spline" => "Spline",
        "extrude" => "Extrude",
        "chamfer" => "Chamfer",
        "fillet" => "Fillet",
        "shell" => "Shell",
        "box" => "Box",
        "cylinder" => "Cylinder",
        "cone" => "Cone",
        "sphere" => "Sphere",
        "torus" => "Torus",
        "wedge" => "Wedge",
        "translate" => "Transform",
        "scatter" => "Scatter",
        "combine" => "Combine to Part",
        "fuse" | "boolean" => "Boolean",
        "linkedpart" => "Linked Part",
        "stepimporter" => "Import STEP",
        "exporter" => "Export STEP",
        "loft" => "Loft",
        "surfaceloft" => "Surface Loft",
        "mirror" => "Mirror",
        _ => "Node",
    }
}

fn node_inputs(node: &nodecraft_session::GraphNode) -> &'static [&'static str] {
    match node.node_type.as_str() {
        "primitive" => match node.values.get("primitiveType").and_then(|value| value.as_str()) {
            Some("sphere") => &["radius"],
            Some("cylinder") => &["radius", "height"],
            Some("torus") => &["majorRadius", "minorRadius"],
            Some("pyramid") => &["length", "height"],
            Some("cone") => &["bottomRadius", "topRadius", "height"],
            Some("partialSphere") => &["radius", "startAngle", "endAngle"],
            Some("wedge") => &["length", "width", "height", "topLength"],
            Some("partialTorus") => &["majorRadius", "minorRadius", "startAngle", "endAngle"],
            _ => &["length", "width", "height"],
        },
        "sketch" | "spline" | "linkedpart" | "stepimporter" => &[],
        "extrude" => &["profile", "amount"],
        "chamfer" | "fillet" => &["shape", "length"],
        "shell" => &["shape", "thickness"],
        "translate" => &["shape", "x", "y", "z", "rx", "ry", "rz"],
        "scatter" => &["shape", "count", "spanX", "spanY", "seed", "distribution", "minScale", "maxScale"],
        "combine" => &["shape"],
        "boolean" | "fuse" => &["a", "b"],
        "loft" => &["bottom", "top", "ruled"],
        "surfaceloft" => &["sections", "ruled"],
        "exporter" | "mirror" => &["shape"],
        _ => &["shape"],
    }
}

/// Inlet ports that carry geometry rather than scalar data. Only data inlets
/// use the orange port colour; geometry inlets and every outlet stay blue.
fn inlet_is_data(port: &str) -> bool {
    !matches!(
        port,
        "shape" | "profile" | "sketch" | "sections" | "bottom" | "top" | "a" | "b" | "faces" | "edges"
    )
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

fn insertion_ports(node: &nodecraft_session::GraphNode) -> Option<(&'static str, &'static str)> {
    let input = node_inputs(node).iter().copied().find(|port| !inlet_is_data(port))?;
    let output = node_outputs(node).first().copied()?;
    Some((input, output))
}

fn node_outputs(node: &nodecraft_session::GraphNode) -> &'static [&'static str] {
    match node.node_type.as_str() {
        "exporter" => &[],
        "sketch" | "spline" => &["sketch"],
        _ => &["shape"],
    }
}

fn node_input_count(node: &nodecraft_session::GraphNode) -> usize {
    node_inputs(node).len()
}

fn port_x(rect: Rect, index: usize, count: usize, scale: f32) -> f32 {
    if count <= 1 {
        return rect.center().x;
    }
    let inset = 22.0 * scale;
    let left = rect.min.x + inset;
    let right = rect.max.x - inset;
    left + (right - left) * index as f32 / (count - 1) as f32
}

fn port_at(
    document: &GraphDocument,
    canvas: Rect,
    pan: Vec2,
    zoom: f32,
    pointer: Pos2,
) -> Option<WirePort> {
    let radius = (12.0 * zoom).max(8.0);
    let mut best: Option<(f32, WirePort)> = None;
    for node in &document.nodes {
        let rect = node_rect(node, canvas, pan, zoom);
        if !rect.intersects(canvas) {
            continue;
        }
        for (index, port) in node_inputs(node).iter().enumerate() {
            let position = Pos2::new(port_x(rect, index, node_inputs(node).len(), zoom), rect.min.y);
            let distance = position.distance(pointer);
            if distance <= radius && best.as_ref().is_none_or(|(best, _)| distance < *best) {
                best = Some((distance, WirePort { node_id: node.id.clone(), port, is_input: true, position }));
            }
        }
        for (index, port) in node_outputs(node).iter().enumerate() {
            let position = Pos2::new(port_x(rect, index, node_outputs(node).len(), zoom), rect.max.y);
            let distance = position.distance(pointer);
            if distance <= radius && best.as_ref().is_none_or(|(best, _)| distance < *best) {
                best = Some((distance, WirePort { node_id: node.id.clone(), port, is_input: false, position }));
            }
        }
    }
    best.map(|(_, port)| port)
}

fn port_compatible(origin: &WirePort, target: &WirePort) -> bool {
    origin.node_id != target.node_id && origin.is_input != target.is_input
}

fn cable_stroke(zoom: f32) -> Stroke {
    Stroke::new((3.2 * zoom).clamp(2.0, 4.5), Color32::from_rgba_unmultiplied(124, 156, 255, 230))
}

fn dashed_curve(painter: &egui::Painter, start: Pos2, end: Pos2, zoom: f32) {
    dashed_curve_with_colour(painter, start, end, zoom, Color32::from_rgba_unmultiplied(124, 156, 255, 230), 0.0);
}

fn dashed_curve_with_colour(
    painter: &egui::Painter,
    start: Pos2,
    end: Pos2,
    zoom: f32,
    colour: Color32,
    phase: f32,
) {
    let points = wire_points(start.x, start.y, end.x, end.y, zoom);
    let stroke = Stroke::new(cable_stroke(zoom).width, colour);
    let dash = (8.0 * zoom).clamp(4.0, 11.0);
    let gap = (6.0 * zoom).clamp(3.0, 8.0);
    let cycle = dash + gap;
    const SAMPLES: usize = 64;
    let samples: Vec<Pos2> = (0..=SAMPLES)
        .map(|sample| {
            let t = sample as f32 / SAMPLES as f32;
            let mt = 1.0 - t;
            (points[0].to_vec2() * (mt * mt * mt)
                + points[1].to_vec2() * (3.0 * mt * mt * t)
                + points[2].to_vec2() * (3.0 * mt * t * t)
                + points[3].to_vec2() * (t * t * t))
                .to_pos2()
        })
        .collect();
    let mut travelled = 0.0_f32;
    for pair in samples.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let length = from.distance(to);
        let mut local = 0.0_f32;
        while local < length - 1.0e-4 {
            let pattern = (travelled + local - phase).rem_euclid(cycle);
            let painting = pattern < dash;
            let step = (if painting { dash - pattern } else { cycle - pattern })
                .min(length - local)
                .max(0.001);
            let a = from + (to - from) * (local / length.max(1.0e-6));
            let b = from + (to - from) * ((local + step) / length.max(1.0e-6));
            if painting {
                painter.line_segment([a, b], stroke);
            }
            local += step;
        }
        travelled += length;
    }
}

fn port_index(ports: &[&str], name: &str) -> usize {
    ports.iter().position(|port| *port == name).unwrap_or(0)
}

pub fn node_width(node: &nodecraft_session::GraphNode) -> f32 {
    let title = node_title(&node.node_type).chars().count() as f32;
    let ports = node_input_count(node).max(node_outputs(node).len()) as f32;
    NODE_MIN_WIDTH
        .max(44.0 + ports * 24.0)
        .max(52.0 + title * 9.5)
}

/// Graph-space card and clearance metrics used by layout code. Keep background,
/// shadow and port overhang changes here rather than in placement call sites.
#[derive(Clone, Copy, Debug)]
pub struct NodeExtent {
    pub width: f32,
    pub height: f32,
    pub padding: f32,
}

impl NodeExtent {
    #[must_use]
    pub fn card_rect(self, x: f64, y: f64) -> Rect {
        Rect::from_min_size(
            Pos2::new(x as f32, y as f32),
            Vec2::new(self.width, self.height),
        )
    }

    #[must_use]
    pub fn padded_rect(self, x: f64, y: f64) -> Rect {
        self.card_rect(x, y).expand(self.padding)
    }
}

#[must_use]
pub fn node_extent(node: &nodecraft_session::GraphNode) -> NodeExtent {
    const CLEARANCE: f32 = 24.0;
    NodeExtent {
        width: node_width(node),
        height: NODE_HEIGHT,
        padding: CLEARANCE,
    }
}

fn graph_rect(node: &nodecraft_session::GraphNode) -> Rect {
    Rect::from_min_size(
        Pos2::new(node.x as f32, node.y as f32),
        Vec2::new(node_width(node), NODE_HEIGHT),
    )
}


fn insertion_probe(
    pointer: Option<Pos2>,
    node: &nodecraft_session::GraphNode,
    panel: Rect,
    pan: Vec2,
    zoom: f32,
) -> Pos2 {
    pointer.unwrap_or_else(|| node_screen_center(node, panel, pan, zoom))
}

fn node_screen_center(
    node: &nodecraft_session::GraphNode,
    panel: Rect,
    pan: Vec2,
    zoom: f32,
) -> Pos2 {
    let canvas_min = Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP);
    let origin = canvas_min + pan + Vec2::new(node.x as f32, node.y as f32) * zoom;
    origin + Vec2::new(node_width(node), NODE_HEIGHT) * zoom * 0.5
}

fn cable_hit(
    document: &GraphDocument,
    panel: Rect,
    pan: Vec2,
    zoom: f32,
    pointer: Pos2,
) -> Option<ConnectionKey> {
    cable_hit_with_hold(document, panel, pan, zoom, pointer, None)
}

/// Hit-test the cable path reconstructed from the current node positions.
/// `held` keeps an existing target while the pointer is still within the wider
/// release band of its new path.
fn cable_hit_with_hold(
    document: &GraphDocument,
    panel: Rect,
    pan: Vec2,
    zoom: f32,
    pointer: Pos2,
    held: Option<&ConnectionKey>,
) -> Option<ConnectionKey> {
    let canvas_min = Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP);
    let map = |x: f32, y: f32| canvas_min + pan + Vec2::new(x, y) * zoom;
    let nodes: HashMap<&str, &nodecraft_session::GraphNode> = document
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect();
    let mut best = (CABLE_HIT_THRESHOLD, None);
    let mut held_distance = f32::INFINITY;
    for connection in &document.connections {
        let Some(from) = nodes.get(connection.from.as_str()).copied() else { continue };
        let Some(to) = nodes.get(connection.to.as_str()).copied() else { continue };
        let source = map(from.x as f32, from.y as f32);
        let target = map(to.x as f32, to.y as f32);
        let source_rect = Rect::from_min_size(source, Vec2::new(node_width(from), NODE_HEIGHT) * zoom);
        let target_rect = Rect::from_min_size(target, Vec2::new(node_width(to), NODE_HEIGHT) * zoom);
        let outputs = node_outputs(from);
        let inputs = node_inputs(to);
        let x0 = port_x(source_rect, port_index(outputs, &connection.out), outputs.len(), zoom);
        let y0 = source_rect.max.y;
        let x1 = port_x(target_rect, port_index(inputs, &connection.input), inputs.len(), zoom);
        let y1 = target_rect.min.y;
        let route = crate::routing::route(x0, y0, x1, y1, zoom);
        let p0 = Pos2::new(x0, y0);
        let p1 = route.control1;
        let p2 = route.control2;
        let p3 = Pos2::new(x1, y1);
        // Broad phase: the cubic lies inside the bounding box of its four
        // control points. This is exact for the routed path, so culling can
        // never hide a cable near the pointer.
        let hull = Rect::from_min_max(
            Pos2::new(
                p0.x.min(p1.x).min(p2.x).min(p3.x),
                p0.y.min(p1.y).min(p2.y).min(p3.y),
            ),
            Pos2::new(
                p0.x.max(p1.x).max(p2.x).max(p3.x),
                p0.y.max(p1.y).max(p2.y).max(p3.y),
            ),
        )
        .expand(CABLE_HIT_RELEASE);
        if !hull.contains(pointer) {
            continue;
        }
        let mut distance = f32::INFINITY;
        let mut previous = p0;
        for step in 1..=24 {
            let t = step as f32 / 24.0;
            let u = 1.0 - t;
            let point = Pos2::new(
                u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x,
                u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y,
            );
            distance = distance.min(distance_to_segment(pointer, previous, point));
            previous = point;
        }
        let key = ConnectionKey::from_connection(connection);
        if held.is_some_and(|held| held == &key) {
            held_distance = distance;
        }
        if distance < best.0 {
            best = (distance, Some(key));
        }
    }
    if let Some(held) = held
        && held_distance <= CABLE_HIT_RELEASE
    {
        // A different cable already inside the acquire radius wins immediately;
        // otherwise the existing target keeps its small release slack.
        if let (distance, Some(candidate)) = best
            && distance <= CABLE_HIT_THRESHOLD
            && candidate != *held
        {
            return Some(candidate);
        }
        return Some(held.clone());
    }
    best.1
}

fn distance_to_segment(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let length_sq = ab.length_sq();
    if length_sq <= f32::EPSILON {
        return (point - a).length();
    }
    let t = ((point - a).dot(ab) / length_sq).clamp(0.0, 1.0);
    (point - (a + ab * t)).length()
}

fn document_roi(document: &GraphDocument) -> Option<Rect> {
    let mut min = Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for node in &document.nodes {
        let rect = graph_rect(node);
        min.x = min.x.min(rect.min.x - NODE_LEFT_PADDING);
        min.y = min.y.min(rect.min.y);
        max.x = max.x.max(rect.max.x + NODE_RIGHT_PADDING);
        max.y = max.y.max(rect.max.y);
    }
    (min.x.is_finite() && max.x.is_finite()).then(|| Rect::from_min_max(min, max))
}

fn node_minimap_rect(panel: Rect) -> Rect {
    Rect::from_min_size(
        Pos2::new(panel.max.x - 12.0 - 132.0, panel.max.y - 12.0 - 132.0),
        Vec2::new(132.0, 132.0),
    )
}

fn ensure_visible_delta(canvas: Rect, target: Rect, padding: f32) -> Vec2 {
    let min = canvas.min + Vec2::splat(padding);
    let max = canvas.max - Vec2::splat(padding);
    let mut delta = Vec2::ZERO;
    if target.max.x > max.x {
        delta.x = max.x - target.max.x;
    } else if target.min.x < min.x {
        delta.x = min.x - target.min.x;
    }
    if target.max.y > max.y {
        delta.y = max.y - target.max.y;
    } else if target.min.y < min.y {
        delta.y = min.y - target.min.y;
    }
    delta
}

fn fitted_view(canvas: Rect, roi: Rect) -> (Vec2, f32) {
    let pad = 24.0;
    let graph_w = roi.width().max(1.0);
    let graph_h = roi.height().max(1.0);
    // Fit may zoom out to make a large graph visible, but it must not magnify past
    // 100% when the graph already fits with room to spare.
    let zoom = ((canvas.width() - 2.0 * pad) / graph_w)
        .min((canvas.height() - 2.0 * pad) / graph_h)
        .clamp(MIN_ZOOM, MAX_ZOOM.min(1.0));
    let pan = canvas.center().to_vec2() - canvas.min.to_vec2() - roi.center().to_vec2() * zoom;
    (pan, zoom)
}

fn panel_canvas(panel: Rect) -> Rect {
    Rect::from_min_max(Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP), panel.max)
}

fn geometry_key(
    document: &GraphDocument,
    selected: &HashSet<String>,
    selected_connections: &HashSet<ConnectionKey>,
    z_order: &[String],
    viewer_breakpoint: Option<&str>,
    theme: &crate::theme::CanvasTheme,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for node in &document.nodes {
        node.id.hash(&mut hasher);
        node.node_type.hash(&mut hasher);
        node.x.to_bits().hash(&mut hasher);
        node.y.to_bits().hash(&mut hasher);
        // Suppression changes the shader hatch flag, so it must invalidate the
        // cached shape instances just like selection does.
        node.suppressed.hash(&mut hasher);
    }
    for connection in &document.connections {
        connection.from.hash(&mut hasher);
        connection.to.hash(&mut hasher);
        connection.input.hash(&mut hasher);
        connection.out.hash(&mut hasher);
    }
    let mut selected_nodes: Vec<&str> = selected.iter().map(String::as_str).collect();
    selected_nodes.sort_unstable();
    selected_nodes.hash(&mut hasher);
    z_order.hash(&mut hasher);
    viewer_breakpoint.hash(&mut hasher);
    let mut cables: Vec<(&str, &str, &str, &str)> = selected_connections
        .iter()
        .map(|key| (key.from.as_str(), key.to.as_str(), key.input.as_str(), key.out.as_str()))
        .collect();
    cables.sort_unstable();
    cables.hash(&mut hasher);
    theme.hash(&mut hasher);
    hasher.finish()
}

fn build_geometry(
    document: &GraphDocument,
    selected: &HashSet<String>,
    z_order: &[String],
    viewer_breakpoint: Option<&str>,
    theme: &crate::theme::CanvasTheme,
    fades: &HashMap<String, f32>,
) -> NodeGeometry {
    let mut builder = GeometryBuilder::default();
    let rects: Vec<Rect> = document.nodes.iter().map(graph_rect).collect();
    let ranks: HashMap<&str, usize> = z_order
        .iter()
        .enumerate()
        .map(|(rank, id)| (id.as_str(), rank))
        .collect();
    let mut draw_order: Vec<usize> = (0..document.nodes.len()).collect();
    draw_order.sort_by_key(|&index| {
        ranks
            .get(document.nodes[index].id.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });
    for index in draw_order {
        let rect = rects[index];
        let node = &document.nodes[index];
        let selected_here = selected.contains(node.id.as_str());
        let viewer_here = viewer_breakpoint == Some(node.id.as_str());
        let fade = fades.get(&node.id).copied().unwrap_or(1.0);
        let card = squircle_rect_path(rect.min.x, rect.min.y, rect.width(), rect.height(), 26.0);
        builder.fill_path(&card, fade_rgba(theme.f32(theme.card), fade));
        builder.stroke_path(
            &card,
            fade_rgba(if selected_here { theme.f32(theme.accent) } else { theme.f32(theme.card_border) }, fade),
            if selected_here { 3.0 } else { 2.0 },
        );
        let (suppress_center, view_center) = button_centers(rect);
        let suppress = circle_path(suppress_center.x, suppress_center.y, BUTTON_RADIUS);
        builder.fill_path(&suppress, fade_rgba(theme.f32(theme.control), fade));
        builder.stroke_path(&suppress, fade_rgba(theme.f32(theme.card_border), fade), 1.0);
        let view = circle_path(view_center.x, view_center.y, VIEW_BUTTON_RADIUS);
        builder.fill_path(&view, fade_rgba(if viewer_here { theme.f32(theme.accent) } else { theme.f32(theme.control) }, fade));
        builder.stroke_path(&view, fade_rgba(theme.f32(theme.card_border), fade), 1.0);
        // Eyelet ports: fill the outer disc, stroke the accent ring, then paint
        // an opaque card-coloured inner disc over the ring's antialiased edge.
        // One eyelet is drawn per declared inlet/outlet, spread across the edge.
        let card_colour = fade_rgba(theme.f32(theme.card), fade);
        let inputs = node_inputs(node);
        for (index, port) in inputs.iter().enumerate() {
            let x = port_x(rect, index, inputs.len(), 1.0);
            let input = circle_path(x, rect.min.y, 8.0);
            builder.fill_path(&input, card_colour);
            let colour = if inlet_is_data(port) {
                theme.port_input
            } else {
                theme.accent
            };
            builder.stroke_path(&input, fade_rgba(theme.f32(colour), fade), 2.0);
            builder.fill_path(&circle_path(x, rect.min.y, 5.8), card_colour);
        }
        let outputs = node_outputs(node);
        for (port, _) in outputs.iter().enumerate() {
            let x = port_x(rect, port, outputs.len(), 1.0);
            let output = circle_path(x, rect.max.y, 8.0);
            builder.fill_path(&output, card_colour);
            builder.stroke_path(&output, fade_rgba(theme.f32(theme.accent), fade), 2.0);
            builder.fill_path(&circle_path(x, rect.max.y, 5.8), card_colour);
        }
    }
    builder.finish()
}

fn fade_color(color: [u8; 4], alpha: f32) -> [u8; 4] {
    [color[0], color[1], color[2], (f32::from(color[3]) * alpha) as u8]
}

fn fade_color32(color: egui::Color32, alpha: f32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        color.r(),
        color.g(),
        color.b(),
        (f32::from(color.a()) * alpha) as u8,
    )
}

fn fade_rgba(color: [f32; 4], alpha: f32) -> [f32; 4] {
    [color[0], color[1], color[2], color[3] * alpha]
}

fn shape_circle(
    out: &mut Vec<ShapeInstance>,
    center: Pos2,
    diameter: f32,
    fill: [f32; 4],
    stroke: [f32; 4],
    stroke_width: f32,
) {
    out.push(ShapeInstance {
        center: [center.x, center.y],
        size: [diameter, diameter],
        fill,
        stroke,
        radius: 0.0,
        stroke_width,
        kind: 1.0,
        _pad: 0.0,
    });
}

/// Glow radius in graph units, with a screen-space floor so distant/zoomed-out
/// nodes still show a readable activity halo.
fn glow_radius(zoom: f32) -> f32 {
    (18.0 / zoom.max(0.01)).max(24.0)
}

fn shape_glow(
    out: &mut Vec<ShapeInstance>,
    rect: Rect,
    color: [f32; 4],
    kind: f32,
    intensity: f32,
    radius: f32,
) {
    let glow = radius;
    out.push(ShapeInstance {
        center: [rect.center().x, rect.center().y],
        size: [rect.width() + glow * 2.0, rect.height() + glow * 2.0],
        fill: [color[0], color[1], color[2], 0.55],
        stroke: [0.0, 0.0, 0.0, intensity],
        radius: 26.0 + glow,
        stroke_width: glow,
        kind,
        _pad: 0.0,
    });
}

fn build_shape_instances(
    document: &GraphDocument,
    selected: &HashSet<String>,
    z_order: &[String],
    viewer_breakpoint: Option<&str>,
    theme: &crate::theme::CanvasTheme,
    fades: &HashMap<String, f32>,
) -> Vec<ShapeInstance> {
    let ranks: HashMap<&str, usize> = z_order
        .iter()
        .enumerate()
        .map(|(rank, id)| (id.as_str(), rank))
        .collect();
    let mut draw_order: Vec<usize> = (0..document.nodes.len()).collect();
    draw_order.sort_by_key(|&index| {
        ranks
            .get(document.nodes[index].id.as_str())
            .copied()
            .unwrap_or(usize::MAX)
    });
    let card_fill = theme.f32(theme.card);
    let card_stroke = theme.f32(theme.card_border);
    let accent = theme.f32(theme.accent);
    let control = theme.f32(theme.control);
    let mut out = Vec::with_capacity(document.nodes.len() * 7);
    for index in draw_order {
        let node = &document.nodes[index];
        let rect = graph_rect(node);
        let selected_here = selected.contains(node.id.as_str());
        let viewer_here = viewer_breakpoint == Some(node.id.as_str());
        let fade = fades.get(&node.id).copied().unwrap_or(1.0);
        let fill = fade_rgba(card_fill, fade);
        let stroke = fade_rgba(if selected_here { accent } else { card_stroke }, fade);
        let control = fade_rgba(control, fade);
        out.push(ShapeInstance {
            center: [rect.center().x, rect.center().y],
            size: [rect.width(), rect.height()],
            fill,
            stroke,
            radius: 26.0,
            stroke_width: if selected_here { 3.0 } else { 2.0 },
            kind: 0.0,
            _pad: if node.suppressed { 1.0 } else { 0.0 },
        });

        let (suppress_center, view_center) = button_centers(rect);
        shape_circle(
            &mut out,
            suppress_center,
            BUTTON_RADIUS * 2.0,
            control,
            stroke,
            1.0,
        );
        shape_circle(
            &mut out,
            view_center,
            VIEW_BUTTON_RADIUS * 2.0,
            if viewer_here { fade_rgba(accent, fade) } else { control },
            stroke,
            1.0,
        );

        let inputs = node_inputs(node);
        for (port_index, port) in inputs.iter().enumerate() {
            let x = port_x(rect, port_index, inputs.len(), 1.0);
            let center = Pos2::new(x, rect.min.y);
            let ring = if inlet_is_data(port) {
                theme.f32(theme.port_input)
            } else {
                accent
            };
            shape_circle(&mut out, center, 16.0, fill, fade_rgba(ring, fade), 2.0);
            shape_circle(&mut out, center, 11.6, fill, fill, 0.0);
        }
        let outputs = node_outputs(node);
        for (port_index, _) in outputs.iter().enumerate() {
            let x = port_x(rect, port_index, outputs.len(), 1.0);
            let center = Pos2::new(x, rect.max.y);
            shape_circle(&mut out, center, 16.0, fill, fade_rgba(accent, fade), 2.0);
            shape_circle(&mut out, center, 11.6, fill, fill, 0.0);
        }
    }
    out
}

/// Cable instance order inside the cable layer. Later entries paint on top.
///
/// Build three buckets in one pass so unrelated cables keep graph order without
/// sorting or cloning connection strings:
///
/// ```text
/// ordinary -> directly connected to selected node -> selected cable
/// ```
type CableRef<'a> = (&'a str, &'a str, &'a str, &'a str);

fn selected_cable_lookup(
    selected_connections: &HashSet<ConnectionKey>,
) -> Option<HashSet<CableRef<'_>>> {
    (!selected_connections.is_empty()).then(|| {
        selected_connections
            .iter()
            .map(|key| {
                (
                    key.from.as_str(),
                    key.to.as_str(),
                    key.input.as_str(),
                    key.out.as_str(),
                )
            })
            .collect()
    })
}

fn ordered_cable_connections(
    document: &GraphDocument,
    selected: &HashSet<String>,
    selected_connections: &HashSet<ConnectionKey>,
) -> Vec<usize> {
    if selected.is_empty() && selected_connections.is_empty() {
        return (0..document.connections.len()).collect();
    }

    // Borrow from the selected keys rather than allocating a ConnectionKey per
    // cable just to test membership.
    let selected_cables = selected_cable_lookup(selected_connections);

    let mut ordinary = Vec::with_capacity(document.connections.len());
    let mut connected = Vec::new();
    let mut selected_out = Vec::new();
    for (index, connection) in document.connections.iter().enumerate() {
        let is_selected = selected_cables.as_ref().is_some_and(|keys| {
            keys.contains(&(
                connection.from.as_str(),
                connection.to.as_str(),
                connection.input.as_str(),
                connection.out.as_str(),
            ))
        });
        if is_selected {
            selected_out.push(index);
        } else if selected.contains(&connection.from) || selected.contains(&connection.to) {
            connected.push(index);
        } else {
            ordinary.push(index);
        }
    }
    ordinary.extend(connected);
    ordinary.extend(selected_out);
    ordinary
}

fn build_cables(
    document: &GraphDocument,
    selected: &HashSet<String>,
    selected_connections: &HashSet<ConnectionKey>,
    theme: &crate::theme::CanvasTheme,
) -> Vec<CableVertex> {
    let nodes: HashMap<&str, &nodecraft_session::GraphNode> = document
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect();
    let highlighted = highlighted_edges(document, selected);
    let selected_cables = selected_cable_lookup(selected_connections);
    let outline = theme.f32(theme.cable_outline);
    let selected_outline = theme.f32(theme.cable_selected_outline);
    let core_width = theme.cable_width.clamp(1.0, 32.0);
    let outline_width = theme.cable_outline_width.clamp(0.0, 15.0);
    let total_width = core_width + outline_width;
    let core_ratio = (core_width / total_width).clamp(0.05, 1.0);
    let mut cables = Vec::with_capacity(document.connections.len());
    for index in ordered_cable_connections(document, selected, selected_connections) {
        let connection = &document.connections[index];
        let (Some(from_node), Some(to_node)) = (
            nodes.get(connection.from.as_str()).copied(),
            nodes.get(connection.to.as_str()).copied(),
        ) else {
            continue;
        };
        let start = graph_rect(from_node);
        let end = graph_rect(to_node);
        let outputs = node_outputs(from_node);
        let inputs = node_inputs(to_node);
        let from_x = port_x(start, port_index(outputs, &connection.out), outputs.len(), 1.0);
        let to_x = port_x(end, port_index(inputs, &connection.input), inputs.len(), 1.0);
        let is_selected = selected_cables.as_ref().is_some_and(|keys| {
            keys.contains(&(
                connection.from.as_str(),
                connection.to.as_str(),
                connection.input.as_str(),
                connection.out.as_str(),
            ))
        });
        let highlighted = highlighted.contains(&index) || is_selected;
        let core = if highlighted {
            theme.f32(theme.cable_selected)
        } else {
            theme.f32(theme.cable)
        };
        let outline = if highlighted { selected_outline } else { outline };
        cables.push(CableVertex {
            a: [from_x, start.max.y],
            b: [to_x, end.min.y],
            core,
            outline,
            width: total_width,
            core_ratio,
            _pad: [0.0; 2],
        });
    }
    cables
}

fn highlighted_edges(document: &GraphDocument, selected: &HashSet<String>) -> HashSet<usize> {
    if selected.is_empty() {
        return HashSet::new();
    }

    // Build adjacency once. The old traversal rescanned every connection for
    // every visited node, which made a selected node in a long chain quadratic.
    let mut incoming: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut outgoing: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, link) in document.connections.iter().enumerate() {
        incoming.entry(link.to.as_str()).or_default().push(index);
        outgoing.entry(link.from.as_str()).or_default().push(index);
    }

    let mut highlighted = HashSet::new();
    for selected_id in selected {
        for (adjacency, forward) in [(&incoming, false), (&outgoing, true)] {
            let mut seen = HashSet::from([selected_id.as_str()]);
            let mut queue = vec![selected_id.as_str()];
            while let Some(current) = queue.pop() {
                let Some(edges) = adjacency.get(current) else { continue };
                for &index in edges {
                    highlighted.insert(index);
                    let link = &document.connections[index];
                    let next = if forward { link.to.as_str() } else { link.from.as_str() };
                    if seen.insert(next) {
                        queue.push(next);
                    }
                }
            }
        }
    }
    highlighted
}

/// Canvas zoom percentage. 100% means one graph unit per egui point.
fn zoom_percent(zoom: f32) -> i32 {
    (zoom * 100.0).round() as i32
}

fn hex_colour(rgba: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgba[0], rgba[1], rgba[2])
}

/// Minimum screen-space hit radius for the small power control. At low zoom the
/// drawn circle is only a few pixels wide; the hit target stays a full round
/// area so the button remains usable without pixel hunting.
const POWER_MIN_HIT_RADIUS: f32 = 11.0;

fn round_button_hit(center: Pos2, pointer: Pos2, radius: f32) -> bool {
    pointer.distance(center) <= radius.max(POWER_MIN_HIT_RADIUS)
}

fn round_button_rect(center: Pos2, radius: f32) -> Rect {
    Rect::from_center_size(center, Vec2::splat(radius.max(POWER_MIN_HIT_RADIUS) * 2.0))
}

fn button_centers(rect: Rect) -> (Pos2, Pos2) {
    let y = rect.center().y;
    (Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y))
}

fn node_rect(node: &nodecraft_session::GraphNode, canvas: Rect, pan: Vec2, zoom: f32) -> Rect {
    // Must match graph_rect exactly: card geometry, hit boxes, icons and labels
    // all use this transform. A legacy 24-point offset for (0, 0) nodes shifted
    // only labels/icons and made them look off-centre.
    let origin = canvas.min + pan + Vec2::new(node.x as f32, node.y as f32) * zoom;
    Rect::from_min_size(origin, Vec2::new(node_width(node), NODE_HEIGHT) * zoom)
}

fn layout_menu_width(ui: &egui::Ui) -> f32 {
    let font = egui::FontId::proportional(13.0);
    let longest = LayoutMode::ALL
        .iter()
        .map(|mode| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(mode.label().to_owned(), font.clone(), egui::Color32::WHITE)
                    .size()
                    .x
            })
        })
        .fold(0.0_f32, f32::max);
    28.0 + longest + 12.0 + 16.0
}

fn layout_menu_row(
    ui: &mut egui::Ui,
    theme: &crate::theme::CanvasTheme,
    selected: bool,
    label: &str,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 26.0),
        egui::Sense::click(),
    );
    if response.hovered() || response.clicked() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let fill = if response.hovered() {
        theme.color(theme.accent)
    } else {
        egui::Color32::TRANSPARENT
    };
    let text = if response.hovered() {
        egui::Color32::WHITE
    } else {
        theme.color(theme.text)
    };
    ui.painter().rect_filled(rect, egui::CornerRadius::same(5), fill);
    if selected {
        ui.painter().text(
            egui::pos2(rect.left() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            "✓",
            egui::FontId::proportional(13.0),
            text,
        );
    }
    ui.painter().text(
        egui::pos2(rect.left() + 28.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        text,
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node_at(x: f64, y: f64) -> nodecraft_session::GraphNode {
        nodecraft_session::GraphNode {
            id: "n".into(),
            node_type: "primitive".into(),
            values: Default::default(),
            suppressed: false,
            x,
            y,
        }
    }

    fn cable_midpoint(
        document: &GraphDocument,
        panel: Rect,
        pan: Vec2,
        zoom: f32,
    ) -> (Pos2, ConnectionKey, Vec2) {
        let connection = &document.connections[0];
        let from = document.nodes.iter().find(|node| node.id == connection.from).unwrap();
        let to = document.nodes.iter().find(|node| node.id == connection.to).unwrap();
        let from_rect = graph_rect(from);
        let to_rect = graph_rect(to);
        let outputs = node_outputs(from);
        let inputs = node_inputs(to);
        let x0 = port_x(from_rect, port_index(outputs, &connection.out), outputs.len(), 1.0);
        let x1 = port_x(to_rect, port_index(inputs, &connection.input), inputs.len(), 1.0);
        let route = crate::routing::route(x0, from_rect.max.y, x1, to_rect.min.y, 1.0);
        let map = |point: Pos2| {
            Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP) + pan + point.to_vec2() * zoom
        };
        let p0 = map(Pos2::new(x0, from_rect.max.y));
        let p1 = map(route.control1);
        let p2 = map(route.control2);
        let p3 = map(Pos2::new(x1, to_rect.min.y));
        let t = 0.5;
        let u = 1.0 - t;
        let point = Pos2::new(
            u * u * u * p0.x + 3.0 * u * u * t * p1.x + 3.0 * u * t * t * p2.x + t * t * t * p3.x,
            u * u * u * p0.y + 3.0 * u * u * t * p1.y + 3.0 * u * t * t * p2.y + t * t * t * p3.y,
        );
        let derivative = Vec2::new(
            3.0 * (u * u * (p1.x - p0.x) + 2.0 * u * t * (p2.x - p1.x) + t * t * (p3.x - p2.x)),
            3.0 * (u * u * (p1.y - p0.y) + 2.0 * u * t * (p2.y - p1.y) + t * t * (p3.y - p2.y)),
        );
        let normal = Vec2::new(-derivative.y, derivative.x).normalized();
        (point, ConnectionKey::from_connection(connection), normal)
    }

    #[test]
    fn highlighted_edges_follow_the_chain_without_rescanning() {
        let mut document = GraphDocument::new("highlight-chain");
        for (id, x) in [("a", 0.0), ("b", 200.0), ("c", 400.0), ("d", 600.0)] {
            document.nodes.push(nodecraft_session::GraphNode {
                id: id.into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x,
                y: 0.0,
            });
        }
        for (from, to) in [("a", "b"), ("b", "c"), ("c", "d")] {
            document.connections.push(nodecraft_session::GraphConnection {
                from: from.into(),
                to: to.into(),
                input: "shape".into(),
                out: "shape".into(),
            });
        }
        let selected = HashSet::from(["b".to_owned()]);
        assert_eq!(highlighted_edges(&document, &selected), HashSet::from([0, 1, 2]));
        assert!(highlighted_edges(&document, &HashSet::new()).is_empty());
    }

    #[test]
    fn selected_and_directly_connected_cables_sort_last() {
        let mut document = GraphDocument::new("cable-layering");
        for (id, x) in [("a", 0.0), ("b", 200.0), ("c", 400.0), ("d", 600.0)] {
            document.nodes.push(nodecraft_session::GraphNode {
                id: id.into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x,
                y: 0.0,
            });
        }
        for (from, to) in [("a", "b"), ("b", "c"), ("c", "d")] {
            document.connections.push(nodecraft_session::GraphConnection {
                from: from.into(),
                to: to.into(),
                input: "shape".into(),
                out: "shape".into(),
            });
        }
        let mut selected = HashSet::from(["b".to_owned()]);
        let ordered = ordered_cable_connections(&document, &selected, &HashSet::new());
        assert_eq!(
            ordered
                .iter()
                .map(|&index| document.connections[index].from.as_str())
                .collect::<Vec<_>>(),
            ["c", "a", "b"]
        );

        selected.clear();
        let selected_cable = HashSet::from([ConnectionKey::from_connection(&document.connections[2])]);
        let ordered = ordered_cable_connections(&document, &selected, &selected_cable);
        assert_eq!(
            ordered.last().map(|&index| document.connections[index].to.as_str()),
            Some("d")
        );
    }

    #[test]
    fn cable_instances_are_built_from_connection_endpoints() {
        let mut document = GraphDocument::new("cables");
        document.nodes = vec![
            nodecraft_session::GraphNode {
                id: "a".into(),
                node_type: "primitive".into(),
                values: Default::default(),
                suppressed: false,
                x: 20.0,
                y: 30.0,
            },
            nodecraft_session::GraphNode {
                id: "b".into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x: 320.0,
                y: 430.0,
            },
        ];
        document.connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        let cables = build_cables(
            &document,
            &HashSet::new(),
            &HashSet::new(),
            &crate::theme::CanvasTheme::default(),
        );
        assert_eq!(cables.len(), 1);
        assert!(cables[0].a[0] < cables[0].b[0]);
        assert!(cables[0].a[1] < cables[0].b[1]);
        assert_eq!(cables[0].width, 7.0);
    }

    #[test]
    fn cable_width_and_outline_width_come_from_theme() {
        let mut document = GraphDocument::new("cable-width");
        document.nodes = vec![
            nodecraft_session::GraphNode {
                id: "a".into(),
                node_type: "primitive".into(),
                values: Default::default(),
                suppressed: false,
                x: 20.0,
                y: 30.0,
            },
            nodecraft_session::GraphNode {
                id: "b".into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x: 320.0,
                y: 430.0,
            },
        ];
        document.connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        let mut theme = crate::theme::CanvasTheme::default();
        theme.cable_width = 24.0;
        theme.cable_outline_width = 6.0;
        let cables = build_cables(&document, &HashSet::new(), &HashSet::new(), &theme);
        assert_eq!(cables[0].width, 30.0);
        assert!((cables[0].core_ratio - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn cable_alpha_comes_from_theme() {
        let mut document = GraphDocument::new("cable-alpha");
        document.nodes = vec![
            nodecraft_session::GraphNode {
                id: "a".into(),
                node_type: "primitive".into(),
                values: Default::default(),
                suppressed: false,
                x: 20.0,
                y: 30.0,
            },
            nodecraft_session::GraphNode {
                id: "b".into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x: 320.0,
                y: 430.0,
            },
        ];
        document.connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        let mut theme = crate::theme::CanvasTheme::default();
        theme.cable = [10, 20, 30, 77];
        let cables = build_cables(&document, &HashSet::new(), &HashSet::new(), &theme);
        assert!((cables[0].core[0] - 10.0 / 255.0).abs() < f32::EPSILON);
        assert!((cables[0].core[1] - 20.0 / 255.0).abs() < f32::EPSILON);
        assert!((cables[0].core[2] - 30.0 / 255.0).abs() < f32::EPSILON);
        assert!((cables[0].core[3] - 77.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn left_tool_dock_offsets_the_floating_panel() {
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 800.0));
        let mut editor = NodeEditor::default();
        let docked = editor.panel_rect(viewport);
        editor.set_left_dock_width(52.0);
        let shifted = editor.panel_rect(viewport);
        assert_eq!(shifted.min.x, docked.min.x + 52.0);
        assert_eq!(shifted.min.y, docked.min.y);
        assert_eq!(shifted.height(), docked.height());
    }

    #[test]
    #[ignore = "performance baseline; run with --ignored --nocapture"]
    fn baseline_canvas_frame_scaling() {
        use std::time::Instant;

        fn document(n: usize) -> GraphDocument {
            let mut document = GraphDocument::new("baseline-frame");
            let columns = (n as f64).sqrt().ceil() as usize;
            for index in 0..n {
                document.nodes.push(nodecraft_session::GraphNode {
                    id: format!("n{index}"),
                    node_type: "translate".into(),
                    values: Default::default(),
                    suppressed: false,
                    x: (index % columns) as f64 * 220.0,
                    y: (index / columns) as f64 * 150.0,
                });
                if index > 0 {
                    document.connections.push(nodecraft_session::GraphConnection {
                        from: format!("n{}", index - 1),
                        to: format!("n{index}"),
                        input: "shape".into(),
                        out: "shape".into(),
                    });
                }
            }
            document
        }

        for n in [100usize, 1_000, 5_000, 10_000, 100_000] {
            let document = document(n);
            let context = egui::Context::default();
            let mut editor = NodeEditor::default();
            editor.set_gpu_ready_for_test(true);
            editor.set_theme(crate::theme::CanvasTheme::default());
            let mut input = egui::RawInput::default();
            input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0)));
            let selected = HashSet::new();
            let selected_connections = HashSet::new();
            let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0));
            // Warm up layout and the cached geometry.
            for _ in 0..2 {
                context.run_ui(input.clone(), |ui| {
                    editor.show(
                        ui.ctx(),
                        viewport,
                        &document,
                        &selected,
                        &selected_connections,
                        None,
                        None,
                    );
                });
            }
            let mut total = 0.0;
            let frames = 4;
            for _ in 0..frames {
                let start = Instant::now();
                context.run_ui(input.clone(), |ui| {
                    editor.show(
                        ui.ctx(),
                        viewport,
                        &document,
                        &selected,
                        &selected_connections,
                        None,
                        None,
                    );
                });
                total += start.elapsed().as_secs_f64() * 1000.0;
            }
            println!(
                "frame nodes={n:>6} avg_cpu_ms={:>8.2}",
                total / f64::from(frames)
            );
        }
    }

    #[test]
    #[ignore = "performance baseline; run with --ignored --nocapture"]
    fn baseline_activity_glow_scaling() {
        use std::time::Instant;
        fn document(n: usize) -> GraphDocument {
            let mut document = GraphDocument::new("activity-baseline");
            let columns = (n as f64).sqrt().ceil() as usize;
            for index in 0..n {
                document.nodes.push(nodecraft_session::GraphNode {
                    id: format!("n{index}"),
                    node_type: "translate".into(),
                    values: Default::default(),
                    suppressed: false,
                    x: (index % columns) as f64 * 220.0,
                    y: (index / columns) as f64 * 150.0,
                });
            }
            document
        }
        let document = document(5_000);
        let context = egui::Context::default();
        let mut editor = NodeEditor::default();
        editor.set_gpu_ready_for_test(true);
        editor.set_theme(crate::theme::CanvasTheme::default());
        for node in &document.nodes {
            editor.activity_started(&node.id);
        }
        let mut input = egui::RawInput::default();
        input.screen_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0)));
        let selected = HashSet::new();
        let selected_connections = HashSet::new();
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0));
        for _ in 0..2 {
            let mut output = context.run_ui(input.clone(), |ui| {
                editor.show(ui.ctx(), viewport, &document, &selected, &selected_connections, None, None);
            });
            output.textures_delta.clear();
        }
        let start = Instant::now();
        let mut output = context.run_ui(input.clone(), |ui| {
            editor.show(ui.ctx(), viewport, &document, &selected, &selected_connections, None, None);
        });
        output.textures_delta.clear();
        let with_activity = start.elapsed().as_secs_f64() * 1000.0;
        editor.clear_activity();
        for _ in 0..2 {
            let mut output = context.run_ui(input.clone(), |ui| {
                editor.show(ui.ctx(), viewport, &document, &selected, &selected_connections, None, None);
            });
            output.textures_delta.clear();
        }
        let start = Instant::now();
        let mut output = context.run_ui(input.clone(), |ui| {
            editor.show(ui.ctx(), viewport, &document, &selected, &selected_connections, None, None);
        });
        output.textures_delta.clear();
        println!(
            "activity glow nodes=5000 with={with_activity:.2}ms without={:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }

    #[test]
    fn cable_geometry_key_includes_port_names() {
        let mut document = GraphDocument::new("key-ports");
        document.nodes.push(nodecraft_session::GraphNode {
            id: "a".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.nodes.push(nodecraft_session::GraphNode {
            id: "b".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 200.0,
            y: 0.0,
        });
        let selected = HashSet::new();
        let connections = HashSet::new();
        let z_order = vec!["a".into(), "b".into()];
        document.connections.push(nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        let first = geometry_key(
            &document,
            &selected,
            &connections,
            &z_order,
            None,
            &crate::theme::CanvasTheme::default(),
        );
        document.connections[0].input = "other".into();
        let second = geometry_key(
            &document,
            &selected,
            &connections,
            &z_order,
            None,
            &crate::theme::CanvasTheme::default(),
        );
        assert_ne!(first, second);
    }

    #[test]
    fn theme_change_rekeys_canvas_geometry() {
        let document = GraphDocument::new("key-theme");
        let selected = HashSet::new();
        let connections = HashSet::new();
        let z_order = Vec::new();
        let theme = crate::theme::CanvasTheme::default();
        let first = geometry_key(
            &document,
            &selected,
            &connections,
            &z_order,
            None,
            &theme,
        );
        let mut changed = theme;
        changed.card = [1, 2, 3, 4];
        let second = geometry_key(
            &document,
            &selected,
            &connections,
            &z_order,
            None,
            &changed,
        );
        assert_ne!(first, second);
    }

    #[test]
    #[ignore = "performance baseline; run with --ignored --nocapture"]
    fn baseline_canvas_scaling() {
        use std::time::Instant;

        fn document(n: usize) -> GraphDocument {
            let mut document = GraphDocument::new("baseline");
            let columns = (n as f64).sqrt().ceil() as usize;
            for index in 0..n {
                document.nodes.push(nodecraft_session::GraphNode {
                    id: format!("n{index}"),
                    node_type: "translate".into(),
                    values: Default::default(),
                    suppressed: false,
                    x: (index % columns) as f64 * 220.0,
                    y: (index / columns) as f64 * 150.0,
                });
                if index > 0 {
                    document.connections.push(nodecraft_session::GraphConnection {
                        from: format!("n{}", index - 1),
                        to: format!("n{index}"),
                        input: "shape".into(),
                        out: "shape".into(),
                    });
                }
            }
            document
        }

        let theme = crate::theme::CanvasTheme::default();
        let selected = HashSet::new();
        let connections = HashSet::new();
        for n in [100usize, 1_000, 5_000, 10_000, 100_000] {
            let document = document(n);
            let z_order: Vec<String> = document.nodes.iter().map(|node| node.id.clone()).collect();
            let start = Instant::now();
            let key = geometry_key(
                &document,
                &selected,
                &connections,
                &z_order,
                None,
                &crate::theme::CanvasTheme::default(),
            );
            let key_ms = start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            let geometry = build_geometry(&document, &selected, &z_order, None, &theme, &HashMap::new());
            let geometry_ms = start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            let shapes = build_shape_instances(&document, &selected, &z_order, None, &theme, &HashMap::new());
            let shapes_ms = start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            let cables = build_cables(&document, &selected, &connections, &theme);
            let cables_ms = start.elapsed().as_secs_f64() * 1000.0;

            let start = Instant::now();
            let visible: Vec<&nodecraft_session::GraphNode> = document.nodes.iter().collect();
            let _occlusion = occlusion_map(
                &visible,
                &z_order,
                Rect::from_min_size(Pos2::ZERO, Vec2::new(1400.0, 900.0)),
                Vec2::ZERO,
                1.0,
            );
            let occlusion_ms = start.elapsed().as_secs_f64() * 1000.0;

            println!(
                "nodes={n:>6} key_ms={key_ms:>7.2} geometry_ms={geometry_ms:>9.2} shapes_ms={shapes_ms:>7.2} shape_count={:>8} cables_ms={cables_ms:>7.2} occlusion_ms={occlusion_ms:>9.2} verts={:>9} indices={:>9} cables={} key={key:x}",
                shapes.len(),
                geometry.vertices.len(),
                geometry.indices.len(),
                cables.len(),
            );
        }
    }

    #[test]
    fn glow_envelope_fades_up_holds_and_fades_down() {
        assert!((glow_strength(Duration::ZERO) - 0.0).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(20)) - 0.5).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(40)) - 1.0).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(300)) - 1.0).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(540)) - 1.0).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(560)) - 0.5).abs() < 1e-6);
        assert!((glow_strength(Duration::from_millis(580)) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn activity_glow_only_fades_down_after_the_end_signal() {
        let started = Instant::now();
        let fade_in = Duration::from_millis(40);
        let hold = Duration::from_millis(500);
        let fade_out = Duration::from_millis(40);
        let strength = |ended, now| activity_strength(started, ended, now, fade_in, hold, fade_out);
        assert!(strength(None, started) < 0.01);
        assert!((strength(None, started + Duration::from_millis(40)) - 1.0).abs() < 1e-6);
        assert!((strength(None, started + Duration::from_secs(5)) - 1.0).abs() < 1e-6);

        // A fast node still holds for 500 ms after start before fading.
        let ended = started + Duration::from_millis(5);
        assert!((strength(Some(ended), ended) - 1.0).abs() < 1e-6);
        assert!((strength(Some(ended), started + Duration::from_millis(499)) - 1.0).abs() < 1e-6);
        assert!((strength(Some(ended), started + Duration::from_millis(520)) - 0.5).abs() < 1e-6);
        assert!(strength(Some(ended), started + Duration::from_millis(540)) < 0.01);

        // A slow node starts fading when its end signal arrives.
        let ended = started + Duration::from_secs(2);
        assert!((strength(Some(ended), ended) - 1.0).abs() < 1e-6);
        assert!((strength(Some(ended), ended + Duration::from_millis(20)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn glow_keeps_a_minimum_screen_size_when_zoomed_out() {
        assert!((glow_radius(1.0) - 24.0).abs() < 1e-6);
        let far = glow_radius(0.02);
        assert!(far * 0.02 >= 18.0 - 1e-6, "screen radius {}", far * 0.02);
        assert!(glow_radius(0.02) > glow_radius(1.0));
    }

    #[test]
    fn glow_instances_are_larger_and_centered_on_the_card() {
        let rect = Rect::from_min_size(Pos2::new(40.0, 60.0), Vec2::new(220.0, 104.0));
        let mut glows = Vec::new();
        shape_glow(&mut glows, rect, [0.2, 0.4, 0.9, 1.0], 2.0, 0.5, glow_radius(1.0));
        assert_eq!(glows.len(), 1);
        let glow = &glows[0];
        assert_eq!(glow.kind, 2.0);
        assert!(glow.size[0] > rect.width() && glow.size[1] > rect.height());
        assert!((glow.center[0] - rect.center().x).abs() < 1e-6);
        assert!((glow.center[1] - rect.center().y).abs() < 1e-6);
        assert!((glow.stroke[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn card_labels_icons_and_hit_tests_share_one_transform() {
        let canvas = Rect::from_min_size(Pos2::new(17.0, 29.0), Vec2::new(640.0, 480.0));
        let pan = Vec2::new(3.5, -7.25);
        let zoom = 1.75;
        for (x, y) in [(0.0, 0.0), (123.5, 45.25), (-40.0, 200.0)] {
            let node = node_at(x, y);
            let graph = graph_rect(&node);
            let screen = node_rect(&node, canvas, pan, zoom);
            let expected = Rect::from_min_size(
                canvas.min + pan + graph.min.to_vec2() * zoom,
                graph.size() * zoom,
            );
            assert_eq!(screen, expected, "transform mismatch at ({x}, {y})");
        }
    }
    #[test]
    fn lasso_grows_and_additive_lasso_toggles_existing_selection() {
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 800.0));
        let mut document = GraphDocument::new("lasso");
        document.nodes = vec![
            node_at(0.0, 0.0),
            nodecraft_session::GraphNode {
                id: "b".into(),
                node_type: "translate".into(),
                values: Default::default(),
                suppressed: false,
                x: 300.0,
                y: 0.0,
            },
        ];
        let base: HashSet<String> = ["n".to_owned(), "b".to_owned()].into_iter().collect();
        let lasso = Rect::from_min_size(Pos2::ZERO, Vec2::new(250.0, 120.0));
        let plain = lasso_selection(&document, canvas, Vec2::ZERO, 1.0, lasso, &base, false);
        assert_eq!(plain, HashSet::from(["n".to_owned()]));
        let additive = lasso_selection(&document, canvas, Vec2::ZERO, 1.0, lasso, &base, true);
        assert_eq!(additive, HashSet::from(["b".to_owned()]));
        let wider = Rect::from_min_size(Pos2::ZERO, Vec2::new(540.0, 120.0));
        let growing = lasso_selection(&document, canvas, Vec2::ZERO, 1.0, wider, &base, false);
        assert_eq!(growing, HashSet::from(["n".to_owned(), "b".to_owned()]));
    }

    #[test]
    fn input_and_output_ports_are_wire_drag_targets() {
        let canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut document = GraphDocument::new("ports");
        document.nodes = vec![node_at(80.0, 80.0)];
        let node = &document.nodes[0];
        let rect = node_rect(node, canvas, Vec2::ZERO, 1.0);
        let input = Pos2::new(port_x(rect, 0, node_inputs(node).len(), 1.0), rect.min.y);
        let output = Pos2::new(port_x(rect, 0, node_outputs(node).len(), 1.0), rect.max.y);
        let input_hit = port_at(&document, canvas, Vec2::ZERO, 1.0, input).expect("input hit");
        let output_hit = port_at(&document, canvas, Vec2::ZERO, 1.0, output).expect("output hit");
        assert!(input_hit.is_input);
        assert!(!output_hit.is_input);
        assert_eq!(input_hit.node_id, node.id);
        assert_eq!(output_hit.node_id, node.id);
        assert!(!port_compatible(&input_hit, &output_hit));
    }

    #[test]
    fn cable_insertion_uses_the_first_geometry_inlet_and_outlet() {
        let primitive = node_at(0.0, 0.0);
        assert!(insertion_ports(&primitive).is_none());
        let mut translate = node_at(0.0, 0.0);
        translate.node_type = "translate".into();
        assert_eq!(insertion_ports(&translate), Some(("shape", "shape")));
        let mut boolean = node_at(0.0, 0.0);
        boolean.node_type = "boolean".into();
        assert_eq!(insertion_ports(&boolean), Some(("a", "shape")));
    }

    #[test]
    fn insertion_preview_shifts_downstream_nodes_and_clears() {
        let mut document = GraphDocument::new("preview");
        let mut source = node_at(0.0, 0.0);
        source.id = "a".into();
        let mut target = node_at(0.0, 200.0);
        target.id = "b".into();
        let mut inserted = node_at(0.0, 100.0);
        inserted.id = "c".into();
        document.nodes = vec![source, target, inserted];
        document.connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        let mut editor = NodeEditor::default();
        editor.insert_preview = Some(InsertPreview {
            candidate: InsertCandidate {
                node: "c".into(),
                connection: ConnectionKey::from_connection(&document.connections[0]),
                input: "shape".into(),
                out: "shape".into(),
            },
            progress: 1.0,
            target: 1.0,
            shift: 0.5,
            shift_target: 1.0,
            shift_from: 0.0,
            shift_started: 0.0,
            previous: None,
            previous_shift: 0.0,
            previous_from: 0.0,
            previous_started: 0.0,
            exit_since: None,
        });
        let preview = editor.preview_document(&document).expect("preview");
        let target = preview.nodes.iter().find(|node| node.id == "b").expect("target");
        assert!((target.y - (200.0 + f64::from(NODE_HEIGHT + 80.0) * 0.5)).abs() < 1.0e-6);
        assert_eq!(preview.connections.len(), 1);
        editor.insert_preview.as_mut().expect("preview").shift = 0.0;
        assert!(editor.preview_document(&document).is_none());
    }

    #[test]
    fn primitive_icons_follow_the_selected_solid_type() {
        for (primitive, expected) in [
            ("box", "cube"),
            ("sphere", "globe"),
            ("cylinder", "database"),
            ("torus", "circle"),
            ("cone", "triangle"),
            ("pyramid", "triangle"),
            ("partialSphere", "globe"),
            ("partialTorus", "circle"),
            ("wedge", "triangle"),
        ] {
            assert_eq!(primitive_icon(primitive), expected, "icon for {primitive}");
        }
    }

    #[test]
    fn only_data_inlets_use_the_orange_port_colour() {
        assert!(inlet_is_data("x"));
        assert!(inlet_is_data("radius"));
        assert!(inlet_is_data("amount"));
        assert!(!inlet_is_data("shape"));
        assert!(!inlet_is_data("profile"));
        assert!(!inlet_is_data("a"));
        assert!(!inlet_is_data("bottom"));
    }

    #[test]
    fn power_button_hit_is_a_round_minimum_area() {
        let center = Pos2::new(100.0, 100.0);
        assert!(round_button_hit(center, Pos2::new(100.0, 100.0), 0.01));
        assert!(round_button_hit(center, Pos2::new(108.0, 100.0), 0.01));
        assert!(!round_button_hit(center, Pos2::new(100.0, 113.0), 0.01));
        assert!(!round_button_hit(center, Pos2::new(113.0, 113.0), 0.01));
        assert_eq!(round_button_rect(center, 0.01).size(), Vec2::splat(22.0));
    }

    #[test]
    fn suppression_invalidates_cached_shape_instances() {
        use nodecraft_session::{GraphDocument, GraphInterface};
        let mut node = nodecraft_session::GraphNode {
            id: "n".into(),
            node_type: "primitive".into(),
            values: Default::default(),
            suppressed: false,
            x: 10.0,
            y: 20.0,
        };
        let document = GraphDocument {
            schema_version: 1,
            id: "d".into(),
            nodes: vec![node.clone()],
            connections: Vec::new(),
            interface: GraphInterface::default(),
        };
        let theme = crate::theme::CanvasTheme::default();
        let selected = HashSet::new();
        let first = geometry_key(&document, &selected, &HashSet::new(), &[], None, &theme);
        node.suppressed = true;
        let mut document = document;
        document.nodes[0] = node;
        let second = geometry_key(&document, &selected, &HashSet::new(), &[], None, &theme);
        assert_ne!(first, second);
    }

    #[test]
    fn button_centers_track_the_card_edges() {
        let card = Rect::from_min_size(Pos2::new(100.0, 200.0), Vec2::new(232.0, 104.0));
        let (left, right) = button_centers(card);
        assert_eq!(left, Pos2::new(100.0, 252.0));
        assert_eq!(right, Pos2::new(332.0, 252.0));
        assert_eq!(left.y, card.center().y);
        assert_eq!(right.y, card.center().y);
    }

    #[test]
    fn panel_screen_rect_tracks_the_last_viewport() {
        let mut editor = NodeEditor::default();
        assert!(editor.panel_screen_rect().is_none());
        editor.last_viewport = Some(Rect::from_min_size(
            Pos2::new(0.0, 50.0),
            Vec2::new(1200.0, 800.0),
        ));
        let panel = editor.panel_screen_rect().expect("panel rect");
        assert!(panel.width() > 0.0);
        assert!(panel.min.y >= 50.0);
        assert!(panel.max.x <= 1200.0);
    }

    #[test]
    fn cable_hit_reconstructs_the_retained_gpu_wire_path() {
        let from = nodecraft_session::GraphNode {
            id: "a".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 10.0,
            y: 20.0,
        };
        let to = nodecraft_session::GraphNode {
            id: "b".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 300.0,
            y: 400.0,
        };
        let mut document = GraphDocument::new("cable-hit");
        document.nodes = vec![from.clone(), to.clone()];
        let connection = nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        };
        document.connections = vec![connection.clone()];
        let panel = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(900.0, 600.0));
        let pan = Vec2::new(13.0, -7.0);
        let zoom = 1.5;
        let from_rect = graph_rect(&from);
        let to_rect = graph_rect(&to);
        let outputs = node_outputs(&from);
        let inputs = node_inputs(&to);
        let x0 = port_x(from_rect, port_index(outputs, "shape"), outputs.len(), 1.0);
        let x1 = port_x(to_rect, port_index(inputs, "shape"), inputs.len(), 1.0);
        // This is the same world-space cubic the retained GPU geometry builds.
        let world = crate::routing::route(x0, from_rect.max.y, x1, to_rect.min.y, 1.0);
        let map = |point: Pos2| {
            Pos2::new(panel.min.x, panel.min.y + CANVAS_TOP)
                + pan
                + point.to_vec2() * zoom
        };
        let start = map(Pos2::new(x0, from_rect.max.y));
        let c1 = map(world.control1);
        let c2 = map(world.control2);
        let end = map(Pos2::new(x1, to_rect.min.y));
        let t = 0.5;
        let mt = 1.0 - t;
        let cursor = Pos2::new(
            mt * mt * mt * start.x
                + 3.0 * mt * mt * t * c1.x
                + 3.0 * mt * t * t * c2.x
                + t * t * t * end.x,
            mt * mt * mt * start.y
                + 3.0 * mt * mt * t * c1.y
                + 3.0 * mt * t * t * c2.y
                + t * t * t * end.y,
        );
        assert_eq!(
            cable_hit(&document, panel, pan, zoom, cursor),
            Some(ConnectionKey::from_connection(&connection))
        );
        document.nodes.push(nodecraft_session::GraphNode {
            id: "c".into(),
            node_type: "primitive".into(),
            values: Default::default(),
            suppressed: false,
            x: 900.0,
            y: 100.0,
        });
        document.nodes.push(nodecraft_session::GraphNode {
            id: "d".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 900.0,
            y: 500.0,
        });
        document.connections.push(nodecraft_session::GraphConnection {
            from: "c".into(),
            to: "d".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        document.connections.push(nodecraft_session::GraphConnection {
            from: "missing".into(),
            to: "d".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        assert_eq!(
            cable_hit(&document, panel, pan, zoom, cursor),
            Some(ConnectionKey::from_connection(&connection)),
            "indexed lookup must keep the near cable and skip unrelated/dangling ones"
        );
    }

    #[test]
    fn insertion_target_holds_through_small_path_moves() {
        let from = nodecraft_session::GraphNode {
            id: "a".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 10.0,
            y: 20.0,
        };
        let to = nodecraft_session::GraphNode {
            id: "b".into(),
            node_type: "translate".into(),
            values: Default::default(),
            suppressed: false,
            x: 300.0,
            y: 400.0,
        };
        let mut document = GraphDocument::new("cable-hold");
        document.nodes = vec![from, to];
        document.connections = vec![nodecraft_session::GraphConnection {
            from: "a".into(),
            to: "b".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        let panel = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(900.0, 600.0));
        let pan = Vec2::new(13.0, -7.0);
        let zoom = 1.5;
        let (cursor, key, normal) = cable_midpoint(&document, panel, pan, zoom);
        assert_eq!(cable_hit(&document, panel, pan, zoom, cursor), Some(key.clone()));
        let slack = cursor + normal * 22.0;
        assert_eq!(cable_hit(&document, panel, pan, zoom, slack), None);
        assert_eq!(
            cable_hit_with_hold(&document, panel, pan, zoom, slack, Some(&key)),
            Some(key.clone())
        );
        let outside = cursor + normal * 40.0;
        assert_eq!(
            cable_hit_with_hold(&document, panel, pan, zoom, outside, Some(&key)),
            None
        );
        // Moving the target node moves the cubic. The retained target follows it.
        document.nodes[1].x += 80.0;
        let (moved, _, _) = cable_midpoint(&document, panel, pan, zoom);
        assert_eq!(
            cable_hit_with_hold(&document, panel, pan, zoom, moved, Some(&key)),
            Some(key)
        );
    }

    #[test]
    fn headless_insertion_probe_uses_the_node_centre() {
        let node = node_at(100.0, 200.0);
        let panel = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(900.0, 600.0));
        let pan = Vec2::new(7.0, 11.0);
        let zoom = 2.0;
        let centre = node_screen_center(&node, panel, pan, zoom);
        let expected = Pos2::new(20.0, 30.0 + CANVAS_TOP)
            + pan
            + Vec2::new(100.0, 200.0) * zoom
            + Vec2::new(node_width(&node), NODE_HEIGHT) * zoom * 0.5;
        assert_eq!(centre, expected);
        assert_eq!(
            insertion_probe(None, &node, panel, pan, zoom),
            expected
        );
        let outside = Pos2::new(1.0, 2.0);
        assert_eq!(
            insertion_probe(Some(outside), &node, panel, pan, zoom),
            outside
        );
    }

    #[test]
    #[test]
    fn fit_does_not_zoom_past_one_hundred_percent() {
        let canvas = Rect::from_min_max(Pos2::ZERO, Pos2::new(1000.0, 800.0));
        let small = Rect::from_min_max(Pos2::ZERO, Pos2::new(100.0, 80.0));
        let (_, zoom) = fitted_view(canvas, small);
        assert_eq!(zoom, 1.0);
    }

    #[test]
    fn newly_added_nodes_start_a_fade_in() {
        let mut editor = NodeEditor::default();
        let mut document = GraphDocument::new("fade");
        document.nodes.push(nodecraft_session::GraphNode {
            id: "a".into(),
            node_type: "primitive".into(),
            values: Default::default(),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        editor.update_node_fades(&document);
        assert!(editor.node_fades.is_empty(), "initial nodes do not animate");
        document.nodes.push(nodecraft_session::GraphNode {
            id: "b".into(),
            node_type: "chamfer".into(),
            values: Default::default(),
            suppressed: false,
            x: 0.0,
            y: 200.0,
        });
        editor.update_node_fades(&document);
        assert!(editor.node_fades.contains_key("b"));
        assert!(editor.node_fade_alpha("b", Instant::now()) < 1.0);
    }

    #[test]
    fn ensure_visible_pans_a_below_view_node_into_view() {
        let canvas = Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 600.0));
        let below = Rect::from_min_size(Pos2::new(100.0, 650.0), Vec2::new(180.0, 70.0));
        let delta = ensure_visible_delta(canvas, below, 24.0);
        assert!(delta.y < 0.0, "content should move up: {delta:?}");
        assert_eq!(delta.y, 576.0 - 720.0);
        assert_eq!(delta.x, 0.0);
    }

    fn zoom_percentage_uses_one_graph_unit_per_point_as_100() {
        assert_eq!(zoom_percent(1.0), 100);
        assert_eq!(zoom_percent(0.5), 50);
        assert_eq!(zoom_percent(2.0), 200);
        assert_eq!(zoom_percent(0.125), 13);
    }

    #[test]
    fn simple_cards_are_narrow_but_busy_cards_grow_for_ports() {
        let primitive = node_at(0.0, 0.0);
        assert_eq!(node_width(&primitive), NODE_MIN_WIDTH);
        let mut translate = node_at(0.0, 0.0);
        translate.node_type = "translate".into();
        assert!(node_width(&translate) > node_width(&primitive));
    }

    #[test]
    fn translate_exposes_every_declared_inlet() {
        let mut translate = node_at(0.0, 0.0);
        translate.node_type = "translate".into();
        let inputs = node_inputs(&translate);
        assert_eq!(inputs.len(), 7);
        let width = node_width(&translate);
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(width, NODE_HEIGHT));
        let xs: Vec<f32> = (0..inputs.len())
            .map(|index| port_x(rect, index, inputs.len(), 1.0))
            .collect();
        assert!(xs.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(xs.first().is_some_and(|x| *x > rect.min.x));
        assert!(xs.last().is_some_and(|x| *x < rect.max.x));
    }

}
