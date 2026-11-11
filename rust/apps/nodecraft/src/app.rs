//! Application shell and the first Swift/WebView-to-egui port slice.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

mod evaluation;
mod document_ops;
mod settings;
mod settings_ui;
mod tabs;
mod theme_editor;
mod titlebar;
mod tool_dock;
use tabs::{DocumentTab, TabDrag, TabViewMode};
use theme_editor::*;
use document_ops::*;
use evaluation::EvalWorker;
pub(crate) use tool_dock::tool_dock_style;
use tool_dock::{ToolDockFocus, ToolItem, tool_dock_height, tool_dock_rect};
use settings::{ActivitySettings, apply_saved_theme, choose_startup_document, default_document, load_app_settings, save_app_settings};

use nodecraft_protocol::Mesh;
use serde::{Deserialize, Serialize};
use nodecraft_session::groups::{
    group_values, is_graph_interface_node, read_group, resize_graph_interface, GraphReference,
    GroupPayload, GroupPort,
};
use nodecraft_session::store::DocumentStore;
use nodecraft_session::{Command, GraphConnection, GraphDocument, GraphNode, Session};

const SOLIDCORE_GIT_HASH: &str = match option_env!("SOLIDCORE_GIT_HASH") {
    Some(hash) => hash,
    None => "unknown",
};

const BUILD_GIT_HASH: &str = match option_env!("NODECRAFT_GIT_HASH") {
    Some(hash) => hash,
    None => "unknown",
};

use crate::engine_bridge;
/// Panel overlays that must not cover menus/popups. egui paints same-Order
/// layers by creation order, so a later-created panel overlay would otherwise
/// lose to a menu created earlier in the frame.
pub(super) const CONTENT_OVERLAY_ORDER: egui::Order = egui::Order::Middle;

/// The breadcrumb belongs to the node canvas (Foreground), but the File menu is
/// raised to the Tooltip order so it still paints above the breadcrumb.
pub(super) const BREADCRUMB_LAYER_ORDER: egui::Order = egui::Order::Foreground;

use crate::file_dialog::{FileDialogKind, FileDialogResult, PendingFileDialog};
use crate::inspector;
use nodecraft_node_canvas::canvas::{
    node_extent, node_width, ConnectionKey, LayoutMode, NodeEditor, NodeEditorAction, NODE_HEIGHT,
    NODE_LEFT_PADDING, NODE_RIGHT_PADDING,
};
use nodecraft_node_canvas::routing::CableRouting;
use nodecraft_node_canvas::theme::CanvasTheme;
use crate::ui_shape;
use crate::control_server;
use crate::sketch_gui::SketchGui;
use crate::viewport::{PreviewScope, Viewport};
use crate::viewer_gizmo::Gizmo;

pub(crate) fn tint_svg(source: &[u8], colour: &str) -> String {
    let mut svg = std::str::from_utf8(source)
        .unwrap_or_default()
        .replace("#212121", colour)
        .replace("#b4b4b4", colour)
        .replace("#d8d8dc", colour)
        .replace("rgb(33,33,33)", colour)
        .replace("rgb(33, 33, 33)", colour);
    if let Some(root_end) = svg.find('>') {
        let root = &svg[..root_end];
        if !root.contains("fill=") && !root.contains("stroke=") {
            svg.insert_str(root_end, &format!(" fill=\"{colour}\""));
        }
    }
    svg
}

const LAYOUT_ANIMATION: Duration = Duration::from_millis(650);
/// Horizontal inset of the inspector's header and body content. Left and right
/// use this one value; the scroll bar gets its own offset so moving it never
/// changes the content margins.
const INSPECTOR_INSET: f32 = 16.0;
/// How far the scroll-area outer edge extends into the right inset. egui puts a
/// floating vertical bar at that outer edge.
const INSPECTOR_SCROLL_BAR_OFFSET: f32 = 9.0;

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

#[derive(Clone, Debug)]
struct EvalProgress {
    total: usize,
    exact_done: usize,
}

impl EvalProgress {
    fn done(&self) -> usize {
        self.exact_done.min(self.total)
    }
}

/// The part of the workspace that is actually the 3D viewer: between the node
/// editor and the inspector overlay. Kept as one helper so overlay placement
/// and hit-testing consumers agree on the viewport bounds.
fn viewer_content_rect(
    viewport: egui::Rect,
    node_panel: egui::Rect,
    inspector: egui::Rect,
) -> egui::Rect {
    let left = node_panel.right().max(viewport.left());
    let right = inspector.left().min(viewport.right()).max(left);
    egui::Rect::from_min_max(
        egui::Pos2::new(left, viewport.top()),
        egui::Pos2::new(right, viewport.bottom()),
    )
}

/// Draw the selected node's evaluation error as a bottom-centered floating
/// banner over the 3D viewer. The inspector no longer owns this state.
fn show_viewer_error(
    ctx: &egui::Context,
    viewport: egui::Rect,
    node_panel: egui::Rect,
    inspector: egui::Rect,
    error: &str,
) {
    if error.trim().is_empty() {
        return;
    }
    let viewer = viewer_content_rect(viewport, node_panel, inspector);
    if viewer.width() < 80.0 || viewer.height() < 40.0 {
        return;
    }
    let max_width = (viewer.width() - 32.0).clamp(180.0, 560.0);
    let theme = crate::theme::get();
    egui::Area::new(egui::Id::new("viewer-node-error"))
        .order(egui::Order::Foreground)
        .fixed_pos(egui::pos2(viewer.center().x, viewer.bottom() - 16.0))
        .pivot(egui::Align2::CENTER_BOTTOM)
        .constrain(true)
        .show(ctx, |ui| {
            ui.set_max_width(max_width);
            egui::Frame::new()
                .fill(theme.with_alpha(theme.node_error, 34))
                .stroke(egui::Stroke::new(1.0, theme.with_alpha(theme.node_error, 190)))
                .corner_radius(8.0)
                .inner_margin(egui::Margin::symmetric(12, 8))
                .show(ui, |ui| {
                    ui.set_max_width((max_width - 24.0).max(120.0));
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("NODE ERROR")
                                .size(10.0)
                                .strong()
                                .color(theme.color(theme.node_error)),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                let svg = tint_svg(
                                    include_bytes!("../../../../vendor/iconify-design/copy.svg"),
                                    &crate::theme::Theme::hex(theme.icon),
                                );
                                let response = ui.add(
                                    egui::Image::from_bytes(
                                        "bytes://viewer-error-copy.svg",
                                        svg.into_bytes(),
                                    )
                                    .fit_to_exact_size(egui::Vec2::splat(14.0))
                                    .sense(egui::Sense::click()),
                                )
                                .on_hover_text("Copy error");
                                if response.hovered() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                                if response.clicked() {
                                    ui.ctx().copy_text(error.to_owned());
                                }
                            },
                        );
                    });
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(error)
                                .size(12.0)
                                .color(theme.color(theme.text)),
                        )
                        .wrap(),
                    );
                });
        });
}

fn progress_fraction(progress: &EvalProgress) -> f32 {
    if progress.total == 0 {
        return 0.0;
    }
    (progress.exact_done as f32 / progress.total as f32).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Appearance,
    Viewport,
    Engine,
    Diagnostics,
    Keys,
}

impl SettingsTab {
    fn label(self) -> &'static str {
        match self {
            Self::Appearance => "Appearance",
            Self::Viewport => "Viewport",
            Self::Engine => "Engine",
            Self::Diagnostics => "Diagnostics",
            Self::Keys => "Keys",
        }
    }
}

struct EvalBinding {
    source_document_id: String,
    instance_path: Vec<String>,
    prefix: String,
    document: GraphDocument,
    body: GraphDocument,
}

#[derive(Clone, Debug)]
struct BreadcrumbEntry {
    label: String,
    definition_id: Option<String>,
}

/// Identity of a graph view over a document definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ViewKey {
    document_id: String,
    definition_id: Option<String>,
}

impl ViewKey {
    fn root(document_id: String) -> Self {
        Self { document_id, definition_id: None }
    }

    fn definition(document_id: String, definition_id: String) -> Self {
        Self { document_id, definition_id: Some(definition_id) }
    }
}

#[derive(Clone)]
struct GroupTab {
    view: ViewKey,
    instance_path: Vec<String>,
    breadcrumb: Vec<BreadcrumbEntry>,
    group_id: String,
    inputs: Vec<GroupPort>,
    outputs: Vec<GroupPort>,
    collapsed: bool,
}

/// Node state captured when a transform-gizmo drag starts.
struct GizmoDragPreview {
    node_id: String,
    /// Values before the drag; the preview maps the old evaluated geometry
    /// through the placement delta onto the live values.
    start_values: serde_json::Value,
    /// Downstream placement chain from the edited node to the displayed target.
    /// Topology does not change during a drag, so it is resolved once.
    chain: nodecraft_session::placement::Affine3,
    /// Display bodies the edit feeds, also resolved once at drag start.
    scope: PreviewScope,
    /// Placement from the edited branch's local space into the viewed target's
    /// space at drag start. Branch previews combine this with the drag delta.
    base_transform: nodecraft_session::placement::Affine3,
    /// Pre-boolean bodies shown while dragging a branch merged by a Boolean.
    /// The merged result has no per-operand provenance, so the drag displays
    /// the inputs instead; the Boolean re-evaluates on release.
    preview_mesh: Option<Mesh>,
    /// The preview mesh is the edited branch input, not the merged breakpoint
    /// body. Draw it solid over the breakpoint mesh as a faint ghost, and map
    /// only the node's own delta onto it.
    branch_preview: bool,
}

/// What to do once a native file dialog closes.
enum DialogAction {
    SaveDocument,
    OpenDocument,
    /// Write the chosen path into a node value (`path` or `directory`).
    SetNodePath {
        node: String,
        field: &'static str,
        open_linked_part: bool,
    },
}

/// A native dialog running off the event loop, with the action to apply once it
/// closes (see `file_dialog` for why dialogs must not run inline).
struct PendingDialogState {
    action: DialogAction,
    dialog: PendingFileDialog,
    status_before: String,
}

/// Identity of one evaluation request. The session generation distinguishes a
/// replacement graph from the old result even when every graph-level field matches.
type EvalKey = (
    u64,
    String,
    u64,
    Option<String>,
    Option<(String, String)>,
    Option<String>,
    Option<String>,
    Vec<String>,
    bool,
    Option<u8>,
);

fn eval_key(
    session_generation: u64,
    document_id: &str,
    revision: u64,
    selection: &Option<String>,
    comparison: &Option<(String, String)>,
    ghost_input: &Option<String>,
    edit_preview: &Option<String>,
    validation_targets: &[String],
    run_exporters: bool,
    preview_detail: Option<u8>,
) -> EvalKey {
    (
        session_generation,
        document_id.to_owned(),
        revision,
        selection.clone(),
        comparison.clone(),
        ghost_input.clone(),
        edit_preview.clone(),
        validation_targets.to_vec(),
        run_exporters,
        preview_detail,
    )
}

/// The before/after pair for the selected feature. The viewport chooses whether
/// to use the selected result as the base or overlay its difference on the
/// viewed breakpoint's body.
fn comparison_for_selection(
    document: &GraphDocument,
    selected: Option<&str>,
    feature_overlay: bool,
) -> Option<(String, String)> {
    if !feature_overlay {
        return None;
    }
    NodeCraftApp::comparison_pair_for(document, selected?)
}


/// The upstream input mesh to ghost under a selected feature's result.
fn transform_edit_input(document: &GraphDocument, edit_target: &str) -> Option<String> {
    let node = document.nodes.iter().find(|node| node.id == edit_target)?;
    if node.node_type != "translate" {
        return None;
    }
    document
        .connections
        .iter()
        .find(|link| link.to == node.id && link.input == "shape")
        .map(|link| link.from.clone())
}

/// The input mesh a feature edit should preview: a transform's shape input or
/// an edge tool's source body. This is what the transform-edit ghost draws and
/// what an edge selection picks against, without replacing the breakpoint.
fn edit_preview_input(document: &GraphDocument, edit_target: &str) -> Option<String> {
    let node = document.nodes.iter().find(|node| node.id == edit_target)?;
    match node.node_type.as_str() {
        "translate" | "mirror" => transform_edit_input(document, edit_target),
        "chamfer" | "fillet" => NodeCraftApp::single_geometry_input(document, node),
        _ => None,
    }
}

/// The placement from the edited branch into the viewed breakpoint. A transform
/// edit includes its own node's placement; a feature edit uses only the
/// downstream placement.
fn edit_placement(
    document: &GraphDocument,
    edit_target: &str,
    target: &str,
) -> Option<nodecraft_session::placement::Affine3> {
    use nodecraft_session::placement::{transform_chain, Affine3};
    let node = document.nodes.iter().find(|node| node.id == edit_target)?;
    let chain = if edit_target == target {
        Affine3::identity()
    } else {
        transform_chain(document, edit_target, target)?
    };
    Some(match Affine3::from_node(node) {
        Some(local) => chain.compose(local),
        None if matches!(node.node_type.as_str(), "translate" | "mirror") => return None,
        None => chain,
    })
}

fn feature_ghost_input(document: &GraphDocument, selected: &str) -> Option<String> {
    let node = document.nodes.iter().find(|node| node.id == selected)?;
    let port = nodecraft_node_contract::preview_input_if_active(&node.node_type, &node.values)?;
    document
        .connections
        .iter()
        .find(|link| link.to == node.id && link.input == port)
        .map(|link| link.from.clone())
}

/// What a double-click activation does for a node type.
///
/// This is deliberately the single policy point for double-click behaviour. If
/// activation later grows modes (for example opening a linked-part document or
/// entering a subnet), add the mode here and handle it in `activate_node`; do not
/// spread node-type checks through the canvas or the action dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeActivation {
    /// Legacy groups switch the canvas into the group's network.
    EnterGroup,
    /// Graph instances open their referenced definition.
    EnterGraph,
    /// Everything else becomes the explicit edit target and reveals its inspector.
    EditNode,
}

fn node_activation(node_type: &str) -> NodeActivation {
    match node_type {
        "group" => NodeActivation::EnterGroup,
        "graph" => NodeActivation::EnterGraph,
        _ => NodeActivation::EditNode,
    }
}

fn selection_is_edit_target(
    selected_nodes: &HashSet<String>,
    edit_target: Option<&str>,
) -> bool {
    selected_nodes.len() == 1 && edit_target == selected_nodes.iter().next().map(String::as_str)
}

fn ghost_alpha(view: &crate::viewport::ViewSettings) -> f32 {
    view.ghost_opacity.clamp(0.0, 1.0)
}

/// Dim the breakpoint mesh for edit mode. The branch is uploaded separately as
/// the ghost layer; it must never be merged into this scene mesh.
fn dim_breakpoint_mesh(breakpoint: &Mesh) -> Mesh {
    let mut mesh = breakpoint.clone();
    if mesh.face_groups.len() != mesh.faces.len() {
        mesh.face_groups = (0..mesh.faces.len()).map(|index| format!("face:{index}")).collect();
    }
    let colors = if mesh.face_colors.is_empty() {
        vec![[0.32, 0.34, 0.37]; mesh.faces.len()]
    } else {
        mesh.face_colors.clone()
    };
    mesh.face_colors = colors
        .into_iter()
        .map(|color| color.map(|channel| channel * 0.45))
        .collect();
    mesh
}

/// Shared placement frame for a transform node relative to the displayed
/// target. The gizmo and the edit ghost must consume this one resolution;
/// computing their chains separately is how a gizmo can end up at a different
/// pivot from the geometry it edits.
struct TransformFrame {
    chain: nodecraft_session::placement::Affine3,
    placement: nodecraft_session::placement::Affine3,
    pivot: [f64; 3],
    axes_basis: nodecraft_session::placement::Affine3,
}

fn transform_frame(
    document: &GraphDocument,
    selected: &str,
    target: &str,
) -> Option<TransformFrame> {
    use nodecraft_session::placement::{transform_chain, Affine3};
    let node = document.nodes.iter().find(|node| node.id == selected)?;
    if node.node_type != "translate" || node.suppressed {
        return None;
    }
    let chain = if selected == target {
        Affine3::identity()
    } else {
        transform_chain(document, selected, target)?
    };
    let value = |key: &str| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    let local = node.values.get("space").and_then(serde_json::Value::as_str) == Some("local");
    let axes_basis = if local {
        chain.compose(Affine3::rotation(&node.values))
    } else {
        chain
    };
    Some(TransformFrame {
        chain,
        placement: chain.compose(Affine3::from_node(node)?),
        pivot: chain.apply_point([value("x"), value("y"), value("z")]),
        axes_basis,
    })
}

/// The one transform-ghost description: a two-body overlay mesh where body 0 is
/// the edited branch (ghost layer) and body 1 is the dimmed, stationary
/// breakpoint. This is the only constructor for transform ghosts.
struct TransformGhost {
    base: Mesh,
    branch: Mesh,
    base_transform: nodecraft_session::placement::Affine3,
    /// Opacity snapshot taken when the ghost is created. Live view-setting
    /// changes still update the active preview through Viewport::set_view.
    alpha: f32,
}

fn transform_ghost_for(
    document: &GraphDocument,
    edit_target: &str,
    target: &str,
    edit_preview: Option<&Mesh>,
    breakpoint: Option<&Mesh>,
    alpha: f32,
) -> Option<TransformGhost> {
    if edit_target == target {
        return None;
    }
    let node = document.nodes.iter().find(|node| node.id == edit_target)?;
    if node.suppressed {
        return None;
    }
    if !matches!(node.node_type.as_str(), "translate" | "mirror" | "chamfer" | "fillet") {
        return None;
    }
    if NodeCraftApp::transform_input_contains_sketch(document, edit_target) {
        return None;
    }
    let preview = edit_preview?;
    let breakpoint = breakpoint?;
    let placement = edit_placement(document, edit_target, target)?;
    Some(TransformGhost {
        base: dim_breakpoint_mesh(breakpoint),
        branch: preview.clone(),
        base_transform: placement,
        alpha: alpha.clamp(0.0, 1.0),
    })
}


pub struct NodeCraftApp {
    session: Option<Session>,
    active_session_revision: u64,
    path: Option<PathBuf>,
    documents: DocumentStore,
    next_document_id: u64,
    selected_nodes: HashSet<String>,
    selected_connections: HashSet<ConnectionKey>,
    /// The node explicitly entered for editing. Selection alone never opens a
    /// gizmo; double-clicking a node sets this independent edit target.
    edit_target: Option<String>,
    viewer_breakpoint: Option<String>,
    /// Selection remembered while the tool dock is open; the canvas click-away
    /// that opens the dock clears `selected_nodes` before the tool item is chosen.
    tool_add_source: Option<String>,
    /// Screen rect of the open tool menu; clicks there belong to the dock, not
    /// to the canvas underneath.
    tool_menu_rect: Option<egui::Rect>,
    face_pick_target: Option<String>,
    /// Downstream target displayed while a topology selection session is active.
    /// When set, picks are mapped back into the edited feature's frame instead
    /// of replacing the downstream view with the feature's input body.
    selection_view_target: Option<String>,
    /// The node whose inspector started the active topology session. If the
    /// selection changes or clears, the session ends rather than silently
    /// reusing the old target.
    selection_node: Option<String>,
    /// Target whose display mesh is currently shown. Picking is only valid when
    /// this is the explicit topology source for the active selection session.
    eval_target: Option<String>,
    comparison_pair: Option<(String, String)>,
    /// Display-only input ghost for the selected feature.
    preview_ghost: Option<String>,
    /// Selected nodes outside the viewed closure that are evaluated for error
    /// glow even though they do not contribute to the displayed mesh.
    validation_selection: Option<Vec<String>>,
    difference_stats: Option<(usize, usize)>,
    node_errors: HashSet<String>,
    activity_seen: HashSet<String>,
    activity_active: HashSet<String>,
    node_error_messages: std::collections::BTreeMap<String, String>,
    difference_error: Option<String>,
    difference_approximate: bool,
    /// Developer-only: panic on the first frame to exercise the local crash dialog.
    crash_test: bool,
    feature_overlay: bool,
    engine_threads: usize,
    tessellation_threads: usize,
    workload_profiler: bool,
    kernel_fixes: bool,
    cable_routing: CableRouting,
    eval_worker: EvalWorker,
    display_worker: std::sync::Arc<nodecraft_display::service::DisplayWorker>,
    pending_display: Option<(u64, String, bool, Instant)>,
    quality_display: Option<(u64, u64, String, bool, Instant)>,
    display_snapshot: Option<evaluation::DisplaySnapshot>,
    display_error: Option<String>,
    last_display_us: Option<(u64, u64)>,
    deferred_display: Option<(
        std::sync::Arc<nodecraft_display::service::DisplayFrame>,
        String,
        bool,
        Instant,
    )>,
    last_applied_display: Option<(
        std::sync::Arc<nodecraft_display::service::DisplayFrame>,
        String,
        bool,
    )>,
    applying_deferred_display: bool,
    displayed_target: Option<String>,
    display_next: u64,
    eval_next: u64,
    eval_pending: Option<u64>,
    /// Bumped whenever the active session is installed. The evaluation cache is
    /// keyed by this too, so a replacement graph that happens to share the old
    /// document id, revision and target cannot be mistaken for the old result.
    session_generation: u64,
    last_eval_key: Option<EvalKey>,
    eval_progress: Option<EvalProgress>,
    egui_ctx: egui::Context,
    mesh: Option<Mesh>,
    /// Display mesh for the node being edited, used by a branch-scoped transform
    /// preview when the breakpoint mesh is geometrically merged.
    edit_preview: Option<Mesh>,
    /// Which node's transform input produced `edit_preview`.
    edit_preview_target: Option<String>,
    /// One retry is allowed when an evaluation reports the edit target but no
    /// preview mesh, so a dropped/queued preview request is not permanent.
    viewport: Viewport,
    node_editor: NodeEditor,
    status: String,
    error: Option<String>,
    evaluation_ms: f64,
    cache_hits: u32,
    cache_misses: u32,
    exact_brep_hits: u32,
    exact_part_hits: u32,
    brep_hits: u32,
    part_hits: u32,
    display_mesh_hits: u32,
    /// Planner defects from the latest display tessellation.
    tessellation_diagnostics: Vec<String>,
    /// Provisional display quality detail; None is the full plan.
    preview_detail: Option<u8>,
    warmup_until: Instant,
    screenshot: Option<(std::path::PathBuf, u32)>,
    render_model: bool,
    preserve_view_for_next_mesh: bool,
    /// Values captured when a gizmo drag starts, driving the display-only
    /// preview transform until the kernel re-evaluates on release.
    gizmo_drag: Option<GizmoDragPreview>,
    /// A triad drag is in progress: auto-evaluation is suspended so the preview
    /// geometry is not replaced (and double-transformed) mid-drag.
    gizmo_dragging: bool,
    /// The drag wrote values that the kernel still has to evaluate.
    gizmo_edited: bool,
    /// A drag applied values while document publishing was deferred; publish
    /// once when the drag releases instead of cloning the graph every frame.
    gizmo_publish_pending: bool,
    /// A native file dialog is open off the event thread; its action runs when
    /// the result arrives (never synchronously inside the event handler).
    pending_dialog: Option<PendingDialogState>,
    camera_initialized: bool,
    last_saved_view: crate::viewport::ViewSettings,
    last_saved_theme: Option<String>,
    activity: ActivitySettings,
    recent_files: Vec<PathBuf>,
    last_saved_recent: Vec<PathBuf>,
    view_save_due: Option<Instant>,
    auto_eval_due: Option<Instant>,
    /// Last exact evaluation, used for display-only LOD swaps.
    last_exact: Option<(u64, engine_bridge::Evaluation)>,
    pending_dirty: HashSet<String>,
    layout_animation: Option<LayoutAnimation>,
    node_position_animation: Option<NodePositionAnimation>,
    tabs: Vec<DocumentTab>,
    group_tabs: HashMap<u64, GroupTab>,
    graph_tab_icon: Option<(egui::Color32, egui::TextureHandle)>,
    tab_positions: HashMap<u64, f32>,
    tab_drag: Option<TabDrag>,
    tab_strip: Option<egui::Rect>,
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
    tool_dock_focus: Option<ToolDockFocus>,
    tool_dock_keyboard_active: bool,
    tool_search_open: bool,
    tool_search_query: String,
    tool_dock_shortcut: String,
    tool_search_shortcut: String,
    tool_search_result_index: Option<usize>,
    tool_search_clear_due: Option<Instant>,
    tool_menu_index: Option<usize>,
    /// Solid split menu: open the primitive list beside the main Create menu.
    tool_solid_submenu_open: bool,
    tool_dock_offset_y: f32,
    tool_dock_measured_height: f32,
    sketch_gui: SketchGui,
    /// The sketch node whose 2D editor is open; node values stay authoritative.
    editing_sketch: Option<String>,
    control_requests: Receiver<control_server::Request>,
    last_pointer: Option<egui::Pos2>,
}

/// Whether the next mesh result should keep the current camera. A target change
/// normally refits, but entering/leaving a topology selection explicitly asks to
/// preserve the view so the inspector edit does not move the 3D viewport.
fn mesh_preserves_camera(
    camera_initialized: bool,
    had_mesh: bool,
    same_target: bool,
    requested: bool,
) -> bool {
    requested || (camera_initialized && had_mesh && same_target)
}

/// Build the atomic command set for resizing an inline graph's body ports.
fn graph_port_commands(
    node_id: &str,
    node: &GraphNode,
    connections: &[GraphConnection],
    inputs: usize,
    outputs: usize,
) -> Option<Vec<Command>> {
    let edit = resize_graph_interface(node, inputs, outputs).ok()?;
    let mut commands = Vec::new();
    for connection in connections {
        let removed_input = connection.to == node_id
            && edit.removed_inputs.iter().any(|port| *port == connection.input);
        let removed_output = connection.from == node_id
            && edit.removed_outputs.iter().any(|port| *port == connection.out);
        if removed_input || removed_output {
            commands.push(Command::Disconnect {
                from: connection.from.clone(),
                to: connection.to.clone(),
                input: connection.input.clone(),
            });
        }
    }
    for (field, value) in [
        ("nodes", serde_json::to_value(&edit.nodes).unwrap_or(serde_json::Value::Null)),
        (
            "connections",
            serde_json::to_value(&edit.connections).unwrap_or(serde_json::Value::Null),
        ),
        (
            "inputs",
            serde_json::to_value(&edit.inputs).unwrap_or(serde_json::Value::Null),
        ),
        (
            "outputs",
            serde_json::to_value(&edit.outputs).unwrap_or(serde_json::Value::Null),
        ),
    ] {
        commands.push(Command::SetValue {
            node: node_id.to_owned(),
            field: field.to_owned(),
            value,
        });
    }
    Some(commands)
}

/// Build the atomic edits for resizing the current document's root interface.
fn root_interface_commands(
    document: &GraphDocument,
    inputs: Option<usize>,
    outputs: Option<usize>,
) -> Option<Vec<Command>> {
    let root = &document.root;
    let input_count = inputs.unwrap_or(root.interface.inputs.len());
    let output_count = outputs.unwrap_or(root.interface.outputs.len());
    let edit = nodecraft_session::groups::resize_definition_interface(
        root,
        input_count,
        output_count,
    )
    .ok()?;
    let input_marker = root.nodes.iter().find(|node| node.node_type == "groupinput").map(|node| node.id.clone());
    let output_marker = root.nodes.iter().find(|node| node.node_type == "groupoutput").map(|node| node.id.clone());
    let mut commands = vec![Command::SetRootDefinition {
        definition: edit.definition,
    }];
    for connection in &document.connections {
        let removed_input = input_marker.as_deref() == Some(connection.from.as_str())
            && edit.removed_inputs.iter().any(|port| *port == connection.out);
        let removed_output = output_marker.as_deref() == Some(connection.to.as_str())
            && edit.removed_outputs.iter().any(|port| *port == connection.input);
        if removed_input || removed_output {
            commands.push(Command::Disconnect {
                from: connection.from.clone(),
                to: connection.to.clone(),
                input: connection.input.clone(),
            });
        }
    }
    Some(commands)
}

fn map_body_id_to_eval(binding: &EvalBinding, id: &str) -> Option<String> {
    let node = binding.body.nodes.iter().find(|node| node.id == id)?;
    match node.node_type.as_str() {
        "groupinput" => {
            // A first-level graph view can bind a Graph Input to the source
            // connected to its owning graph node.
            if binding.instance_path.len() != 1 {
                return None;
            }
            let graph_id = binding.instance_path.first()?;
            let mut sources = binding
                .document
                .connections
                .iter()
                .filter(|connection| &connection.to == graph_id);
            let source = sources.next()?;
            sources.next().is_none().then(|| source.from.clone())
        }
        "groupoutput" => Some(binding.prefix.clone()),
        _ => Some(format!("{}::{id}", binding.prefix)),
    }
}

fn map_eval_id_to_body(binding: &EvalBinding, id: &str) -> Option<String> {
    if id == binding.prefix {
        return binding
            .body
            .nodes
            .iter()
            .rev()
            .find(|node| node.node_type == "groupoutput")
            .map(|node| node.id.clone())
            .or_else(|| Some(binding.prefix.clone()));
    }
    if let Some(id) = id.strip_prefix(&format!("{}::", binding.prefix)) {
        return Some(id.to_owned());
    }
    // External input sources are shared with the parent graph. Reverse-map them
    // to the Graph Input marker so the body sees a body-local preview target.
    binding
        .body
        .nodes
        .iter()
        .find(|node| map_body_id_to_eval(binding, &node.id).as_deref() == Some(id))
        .map(|node| node.id.clone())
}

fn nodecraft_graph_definition(
    definition: &nodecraft_session::GraphDefinition,
) -> nodecraft_session::GraphDefinition {
    nodecraft_session::GraphDefinition {
        id: definition.id.clone(),
        name: definition.name.clone(),
        kind: definition.kind.clone(),
        interface: definition.interface.clone(),
        nodes: definition.nodes.clone(),
        connections: definition.connections.clone(),
    }
}

fn cascade_removed_definitions(
    document: &GraphDocument,
    removed_nodes: &HashSet<String>,
) -> BTreeSet<String> {
    let mut refs: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut count_node = |node: &GraphNode| {
        if node.node_type == "graph"
            && let Some(GraphReference::Local(id)) = GraphReference::from_node(node)
        {
            *refs.entry(id).or_default() += 1;
        }
    };
    for node in &document.nodes {
        if !removed_nodes.contains(&node.id) {
            count_node(node);
        }
    }
    for definition in &document.definitions {
        for node in &definition.nodes {
            count_node(node);
        }
    }
    let mut queue: Vec<String> = document
        .nodes
        .iter()
        .filter(|node| removed_nodes.contains(&node.id))
        .filter_map(|node| match GraphReference::from_node(node) {
            Some(GraphReference::Local(id)) => Some(id),
            _ => None,
        })
        .collect();
    let mut removed = BTreeSet::new();
    while let Some(id) = queue.pop() {
        if removed.contains(&id) || refs.get(&id).copied().unwrap_or(0) > 0 {
            continue;
        }
        removed.insert(id.clone());
        if let Some(definition) = document.definition(&id) {
            for node in &definition.nodes {
                if node.node_type == "graph"
                    && let Some(GraphReference::Local(child)) = GraphReference::from_node(node)
                {
                    if let Some(count) = refs.get_mut(&child) {
                        *count = count.saturating_sub(1);
                    }
                    if refs.get(&child).copied().unwrap_or(0) == 0 {
                        queue.push(child);
                    }
                }
            }
        }
    }
    removed
}

fn commands_are_layout_only(commands: &[Command]) -> bool {
    !commands.is_empty()
        && commands
            .iter()
            .all(|command| matches!(command, Command::SetPosition { .. }))
}

/// Build the atomic edits for resizing a graph definition interface.
///
/// Returns the commands plus every definition whose body changed (the target
/// first, then definitions containing nested instances).
fn definition_port_commands(
    document: &GraphDocument,
    definition_id: &str,
    inputs: usize,
    outputs: usize,
) -> Option<(Vec<Command>, Vec<nodecraft_session::GraphDefinition>)> {
    let definition = document.definition(definition_id)?;
    let edit = nodecraft_session::groups::resize_definition_interface(definition, inputs, outputs).ok()?;
    let mut commands = vec![Command::SetDefinition {
        definition: edit.definition.clone(),
    }];
    let mut refreshed = vec![edit.definition.clone()];
    let references = |node: &GraphNode| {
        node.node_type == "graph"
            && matches!(
                GraphReference::from_node(node),
                Some(GraphReference::Local(ref id)) if id == definition_id
            )
    };
    let instances: Vec<String> = document
        .nodes
        .iter()
        .filter(|node| references(node))
        .map(|node| node.id.clone())
        .collect();
    for instance in &instances {
        for connection in &document.connections {
            let removed_input = connection.to == *instance
                && edit.removed_inputs.iter().any(|port| *port == connection.input);
            let removed_output = connection.from == *instance
                && edit.removed_outputs.iter().any(|port| *port == connection.out);
            if removed_input || removed_output {
                commands.push(Command::Disconnect {
                    from: connection.from.clone(),
                    to: connection.to.clone(),
                    input: connection.input.clone(),
                });
            }
        }
        commands.push(Command::SetValue {
            node: instance.clone(),
            field: "inputs".to_owned(),
            value: serde_json::to_value(&edit.definition.interface.inputs)
                .unwrap_or(serde_json::Value::Null),
        });
        commands.push(Command::SetValue {
            node: instance.clone(),
            field: "outputs".to_owned(),
            value: serde_json::to_value(&edit.definition.interface.outputs)
                .unwrap_or(serde_json::Value::Null),
        });
    }
    for definition in document.definitions.clone() {
        if definition.id == definition_id {
            continue;
        }
        let nested_instances: Vec<String> = definition
            .nodes
            .iter()
            .filter(|node| references(node))
            .map(|node| node.id.clone())
            .collect();
        if nested_instances.is_empty() {
            continue;
        }
        let mut updated = definition;
        for node in &mut updated.nodes {
            if nested_instances.iter().any(|id| id == &node.id) {
                node.values["inputs"] = serde_json::to_value(&edit.definition.interface.inputs)
                    .unwrap_or(serde_json::Value::Null);
                node.values["outputs"] = serde_json::to_value(&edit.definition.interface.outputs)
                    .unwrap_or(serde_json::Value::Null);
            }
        }
        updated.connections.retain(|connection| {
            let removed_input = nested_instances.iter().any(|id| id == &connection.to)
                && edit.removed_inputs.iter().any(|port| *port == connection.input);
            let removed_output = nested_instances.iter().any(|id| id == &connection.from)
                && edit.removed_outputs.iter().any(|port| *port == connection.out);
            !removed_input && !removed_output
        });
        commands.push(Command::SetDefinition {
            definition: updated.clone(),
        });
        refreshed.push(updated);
    }
    Some((commands, refreshed))
}

fn unique_graph_definition_id(base: &str, ids: &BTreeSet<String>) -> String {
    if !ids.contains(base) {
        return base.to_owned();
    }
    let mut suffix = 2;
    loop {
        let candidate = format!("{base}{suffix}");
        if !ids.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn unique_graph_definition_name(names: &BTreeSet<String>) -> String {
    let base = "Graph";
    if !names.contains(base) {
        return base.to_owned();
    }
    let mut suffix = 2;
    loop {
        let candidate = format!("{base} {suffix}");
        if !names.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

impl NodeCraftApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial: Option<PathBuf>,
        screenshot: Option<PathBuf>,
        render_model: bool,
        crash_test: bool,
    ) -> Self {
        // Until the GPU tessellation pipeline finishes compiling in the
        // background, plan frames carry a CPU mesh so the first window is
        // visible and editable immediately.
        nodecraft_display::service::set_gpu_tess_ready(false);
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
        node_editor.set_left_dock_width(0.0);
        if let Some(render_state) = &cc.wgpu_render_state {
            viewport.install_gpu(render_state);
            #[cfg(feature = "gpu-viewport")]
            node_editor.install_gpu(
                render_state,
                crate::gpu_viewport::msaa_samples(),
                solidcraft_viewport_gpu::depth_format(&render_state.device, true),
            );
        }
        if render_model {
            viewport.set_chrome_visible(false);
        }
        let persisted_view = settings.view;
        let activity = settings.activity;
        nodecraft_engine::kernel_pool::configure_exact_threads(settings.engine_threads);
        nodecraft_engine::tessellation::configure_tessellation_threads(settings.tessellation_threads);
        nodecraft_engine::configure_workload_profiler(settings.workload_profiler);
        nodecraft_engine::configure_solidcore_fixes(settings.kernel_fixes);
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
        #[cfg(target_os = "macos")]
        nodecraft_macos::round_main_window();
        crate::window_chrome::prepare_main_window();
        let display_worker = std::sync::Arc::new(
            nodecraft_display::service::DisplayWorker::start(16384, {
                let ctx = cc.egui_ctx.clone();
                move || ctx.request_repaint()
            })
            .expect("display worker"),
        );
        let mut app = Self {
            session: None,
            active_session_revision: 0,
            path: None,
            documents: DocumentStore::new(),
            next_document_id: 1,
            selected_nodes: HashSet::new(),
            edit_target: None,
            selected_connections: HashSet::new(),
            viewer_breakpoint: None,
            tool_add_source: None,
            tool_menu_rect: None,
            face_pick_target: None,
            selection_view_target: None,
            selection_node: None,
            eval_target: None,
            comparison_pair: None,
            preview_ghost: None,
            validation_selection: None,
            difference_stats: None,
            node_errors: HashSet::new(),
            activity_seen: HashSet::new(),
            activity_active: HashSet::new(),
            node_error_messages: std::collections::BTreeMap::new(),
            difference_error: None,
            difference_approximate: false,
            crash_test,
            feature_overlay: settings.feature_overlay,
            engine_threads: settings.engine_threads,
            tessellation_threads: settings.tessellation_threads,
            workload_profiler: settings.workload_profiler,
            kernel_fixes: settings.kernel_fixes,
            cable_routing: CableRouting::from_setting(&settings.cable_routing),
            eval_worker: EvalWorker::new(std::sync::Arc::clone(&display_worker)),
            display_worker,
            pending_display: None,
            quality_display: None,
            display_snapshot: None,
            display_error: None,
            last_display_us: None,
            deferred_display: None,
            applying_deferred_display: false,
            last_applied_display: None,
            displayed_target: None,
            display_next: 0,
            eval_next: 0,
            eval_pending: None,
            session_generation: 0,
            last_eval_key: None,
            eval_progress: None,
            egui_ctx: cc.egui_ctx.clone(),
            mesh: None,
            edit_preview: None,
            edit_preview_target: None,
            viewport,
            node_editor,
            status: "Ready".to_owned(),
            error: None,
            evaluation_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            exact_brep_hits: 0,
            exact_part_hits: 0,
            brep_hits: 0,
            part_hits: 0,
            display_mesh_hits: 0,
            tessellation_diagnostics: Vec::new(),
            preview_detail: Some(1),
            // The SolidCraft callback needs a prepare frame before its first blit.
            warmup_until: Instant::now() + std::time::Duration::from_secs(2),
            screenshot: screenshot.map(|path| (path, 0)),
            render_model,
            preserve_view_for_next_mesh: false,
            gizmo_drag: None,
            gizmo_dragging: false,
            gizmo_edited: false,
            gizmo_publish_pending: false,
            pending_dialog: None,
            camera_initialized: false,
            last_saved_view: persisted_view,
            last_saved_theme: initial_theme_id,
            activity,
            recent_files,
            last_saved_recent: persisted_recent,
            view_save_due: None,
            auto_eval_due: None,
            last_exact: None,
            pending_dirty: HashSet::new(),
            layout_animation: None,
            node_position_animation: None,
            tabs: Vec::new(),
            group_tabs: HashMap::new(),
            graph_tab_icon: None,
            tab_positions: HashMap::new(),
            tab_drag: None,
            tab_strip: None,
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
            tool_dock_focus: None,
            tool_dock_keyboard_active: false,
            tool_search_open: false,
            tool_search_query: String::new(),
            tool_dock_shortcut: settings.tool_dock_shortcut.clone(),
            tool_search_shortcut: settings.tool_search_shortcut.clone(),
            tool_search_result_index: None,
            tool_search_clear_due: None,
            tool_menu_index: None,
            tool_solid_submenu_open: false,
            tool_dock_offset_y: settings.tool_dock_offset_y,
            tool_dock_measured_height: 0.0,
            sketch_gui: SketchGui::default(),
            editing_sketch: None,
            control_requests: control_server::start(cc.egui_ctx.clone()),
            last_pointer: None,
        };
        #[cfg(target_os = "macos")]
        nodecraft_macos::install(cc.egui_ctx.clone());
        app.node_editor.set_cable_routing(app.cable_routing);
        if let Some(path) = choose_startup_document(initial, startup_recent, default_document()) {
            app.open(path);
        } else {
            app.new_blank();
        }
        app
    }

    fn remap_selection(&mut self) {
        let Some(session) = self.session.as_ref() else { return };
        let document = session.document();
        self.selected_nodes.retain(|id| document.nodes.iter().any(|node| &node.id == id));
        let clear_edit_target = self
            .edit_target
            .as_ref()
            .is_some_and(|id| !document.nodes.iter().any(|node| &node.id == id));
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
        if clear_edit_target {
            self.clear_edit_target();
        }
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
        self.evaluate_with_cache_policy(false, true);
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

    /// Developer/debug action: force a fresh evaluation and drop every result
    /// cache. Production invalidation must come from the graph content hashes, so
    /// this button is temporary and should not grow a record-cache dependency.
    fn reevaluate_non_export(&mut self) {
        // Re-evaluate bypasses cache lookups instead of freeing every cache.
        // Content hashes already invalidate changed results; dropping large
        // B-rep/mesh maps here only adds deallocation latency and is not needed
        // for a fresh evaluation.
        self.edit_preview = None;
        self.edit_preview_target = None;
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
        self.evaluate_with_cache_policy(true, false);
    }

    /// The primary geometry input a newly-added node should use when inserted
    /// downstream of the selected node. Non-geometry nodes return `None`.
    fn primary_geometry_input(node_type: &str) -> Option<&'static str> {
        nodecraft_session::node_descriptor::descriptor(node_type)
            .and_then(|descriptor| descriptor.inputs.first())
            .map(|port| port.name)
    }

    fn primary_output(node_type: &str) -> Option<&'static str> {
        match node_type {
            "exporter" | "time" | "linkedpart" | "groupoutput" => None,
            "sketch" | "spline" => Some("sketch"),
            _ => Some("shape"),
        }
    }

    fn auto_connect_ports(previous: &str, added: &str) -> Option<(&'static str, &'static str)> {
        Some((Self::primary_geometry_input(added)?, Self::primary_output(previous)?))
    }

    fn active_eval_binding(&self) -> Option<EvalBinding> {
        let tab = self.tabs.get(self.active_tab)?;
        let meta = self.group_tabs.get(&tab.id)?;
        meta.view.definition_id.as_ref()?;
        if meta.instance_path.is_empty() {
            return None;
        }
        let source_document_id = meta.view.document_id.clone();
        let document = self
            .documents
            .session(&source_document_id)?
            .document()
            .clone();
        let body = self.session.as_ref()?.document().clone();
        Some(EvalBinding {
            source_document_id,
            instance_path: meta.instance_path.clone(),
            prefix: meta.instance_path.join("::"),
            document,
            body,
        })
    }

    fn breadcrumb_entries(&self) -> Vec<BreadcrumbEntry> {
        let Some(tab) = self.tabs.get(self.active_tab) else {
            return Vec::new();
        };
        if let Some(meta) = self.group_tabs.get(&tab.id)
            && !meta.breadcrumb.is_empty()
        {
            return meta.breadcrumb.clone();
        }
        vec![BreadcrumbEntry {
            label: self.document_title(tab),
            definition_id: None,
        }]
    }

    fn show_breadcrumb(&mut self, ctx: &egui::Context, panel: egui::Rect) {
        let entries = self.breadcrumb_entries();
        if entries.is_empty() {
            return;
        }
        let theme = crate::theme::get();
        let font = egui::FontId::proportional(12.0);
        let text = theme.color(theme.text);
        let muted = theme.color(theme.muted);
        let hover_fill = theme.with_alpha(theme.control, 225);
        let mut clicked = None;
        egui::Area::new(egui::Id::new("node-editor-breadcrumb"))
            .order(BREADCRUMB_LAYER_ORDER)
            .fixed_pos(panel.min + egui::vec2(16.0, 7.0))
            .interactable(true)
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                ui.horizontal(|ui| {
                    let title_width = ui
                        .painter()
                        .layout_no_wrap("NODE EDITOR".to_owned(), font.clone(), text)
                        .size()
                        .x;
                    let (title_rect, _) = ui.allocate_exact_size(
                        egui::vec2(title_width, 22.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().text(
                        title_rect.left_center(),
                        egui::Align2::LEFT_CENTER,
                        "NODE EDITOR",
                        font.clone(),
                        text,
                    );
                    for (index, entry) in entries.iter().enumerate() {
                        let (chevron_rect, _) = ui.allocate_exact_size(
                            egui::vec2(14.0, 22.0),
                            egui::Sense::hover(),
                        );
                        let centre = chevron_rect.center();
                        let stroke = egui::Stroke::new(1.4, muted);
                        ui.painter().line_segment(
                            [
                                egui::pos2(centre.x - 3.0, centre.y - 4.0),
                                egui::pos2(centre.x + 1.5, centre.y),
                            ],
                            stroke,
                        );
                        ui.painter().line_segment(
                            [
                                egui::pos2(centre.x + 1.5, centre.y),
                                egui::pos2(centre.x - 3.0, centre.y + 4.0),
                            ],
                            stroke,
                        );
                        let width = ui
                            .painter()
                            .layout_no_wrap(entry.label.clone(), font.clone(), text)
                            .size()
                            .x
                            + 14.0;
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(width, 22.0),
                            egui::Sense::click(),
                        );
                        if response.hovered() {
                            ui.painter().add(nodecraft_node_canvas::squircle::shape(
                                rect,
                                8.0,
                                hover_fill,
                                egui::Stroke::NONE,
                            ));
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        ui.painter().text(
                            rect.center(),
                            egui::Align2::CENTER_CENTER,
                            entry.label.clone(),
                            font.clone(),
                            text,
                        );
                        if response.clicked() {
                            clicked = Some(index);
                        }
                    }
                });
            });
        if let Some(index) = clicked {
            self.navigate_breadcrumb(index);
        }
    }

    fn navigate_breadcrumb(&mut self, index: usize) {
        let Some(tab) = self.tabs.get(self.active_tab).cloned() else {
            return;
        };
        let Some(meta) = self.group_tabs.get(&tab.id).cloned() else {
            return;
        };
        if index + 1 >= meta.breadcrumb.len() {
            return;
        }
        let Some(entry) = meta.breadcrumb.get(index).cloned() else {
            return;
        };
        let source_document_id = meta.view.document_id.clone();
        if let Some(definition_id) = entry.definition_id {
            self.navigate_definition_view(
                &source_document_id,
                &definition_id,
                meta.breadcrumb[..=index].to_vec(),
            );
        } else {
            self.navigate_root_view(&source_document_id);
        }
    }

    fn navigate_root_view(&mut self, source_document_id: &str) {
        let Some(parent) = self.documents.session(source_document_id).cloned() else {
            return;
        };
        let Some(tab) = self.tabs.get(self.active_tab).cloned() else {
            return;
        };
        let path = self.documents.path(source_document_id).map(Path::to_path_buf);
        self.group_tabs.remove(&tab.id);
        self.documents.remove(&tab.document_id);
        self.install_session(Some(parent));
        self.active_session_revision = self.session.as_ref().map_or(0, Session::revision);
        if let Some(current) = self.tabs.get_mut(self.active_tab) {
            current.document_id = source_document_id.to_owned();
            current.path = path.clone();
            current.mesh = None;
            current.selected_nodes.clear();
            current.selected_connections.clear();
            current.edit_target = None;
            current.viewer_breakpoint = None;
        }
        self.path = path;
        self.selected_nodes.clear();
        self.selected_connections.clear();
        self.edit_target = None;
        self.viewer_breakpoint = None;
        self.mesh = None;
        self.node_editor.reset_view();
        self.publish_active_document();
        self.evaluate();
    }

    fn navigate_definition_view(
        &mut self,
        source_document_id: &str,
        definition_id: &str,
        breadcrumb: Vec<BreadcrumbEntry>,
    ) {
        let Some(parent) = self.documents.session(source_document_id) else {
            return;
        };
        let Some(definition) = parent.document().definition(definition_id).cloned() else {
            return;
        };
        let definitions = parent.document().definitions.clone();
        let Some(tab) = self.tabs.get(self.active_tab).cloned() else {
            return;
        };
        let projection = GraphDocument {
            schema_version: 1,
            id: format!("{definition_id}::graph"),
            root: nodecraft_graph_definition(&definition),
            definitions,
        };
        if let Some(meta) = self.group_tabs.get_mut(&tab.id) {
            meta.view = ViewKey::definition(source_document_id.to_owned(), definition_id.to_owned());
            meta.instance_path.truncate(breadcrumb.len().saturating_sub(1));
            meta.breadcrumb = breadcrumb;
            meta.group_id.clear();
        }
        self.documents.remove(&tab.document_id);
        self.install_session(Some(Session::new(projection)));
        self.active_session_revision = self.session.as_ref().map_or(0, Session::revision);
        if let Some(current) = self.tabs.get_mut(self.active_tab) {
            current.path = None;
            current.mesh = None;
            current.selected_nodes.clear();
            current.selected_connections.clear();
            current.edit_target = None;
            current.viewer_breakpoint = None;
        }
        self.path = None;
        self.selected_nodes.clear();
        self.selected_connections.clear();
        self.edit_target = None;
        self.viewer_breakpoint = None;
        self.mesh = None;
        self.node_editor.reset_view();
        self.evaluate();
    }

    fn add_definition_removals(
        &self,
        mut commands: Vec<Command>,
        removed_nodes: &HashSet<String>,
    ) -> (Vec<Command>, BTreeSet<String>) {
        if self.active_definition_view().is_some() {
            return (commands, BTreeSet::new());
        }
        let Some(document) = self.session.as_ref().map(Session::document) else {
            return (commands, BTreeSet::new());
        };
        let removed = cascade_removed_definitions(document, removed_nodes);
        for id in &removed {
            commands.push(Command::RemoveDefinition { id: id.clone() });
        }
        (commands, removed)
    }

    fn close_tabs_for_removed_definitions(&mut self, removed: &BTreeSet<String>) {
        if removed.is_empty() {
            return;
        }
        loop {
            let index = self.tabs.iter().position(|tab| {
                self.group_tabs.get(&tab.id).is_some_and(|meta| {
                    meta.view
                        .definition_id
                        .as_deref()
                        .is_some_and(|id| removed.contains(id))
                })
            });
            let Some(index) = index else { break };
            self.close_tab(index);
        }
    }

    /// Definition ids and names visible from the active view. Definition tabs
    /// render a projection with an empty table, so uniqueness must come from the
    /// parent document.
    fn definition_registry(&self) -> (BTreeSet<String>, BTreeSet<String>) {
        let document = if let Some((source_document_id, _)) = self.active_definition_view() {
            self.documents
                .session(&source_document_id)
                .map(Session::document)
        } else {
            self.session.as_ref().map(Session::document)
        };
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        if let Some(document) = document {
            ids.insert(document.root.id.clone());
            for definition in &document.definitions {
                ids.insert(definition.id.clone());
                names.insert(definition.name.clone());
            }
        }
        (ids, names)
    }

    /// Initial vertical layout for a new graph's interface markers. The node
    /// canvas fits to content on first entry, so keeping the two markers in one
    /// column spans them across the available editor height.
    fn new_graph_port_positions(workspace: egui::Rect) -> ((f64, f64), (f64, f64)) {
        let panel_height = f64::from(workspace.height().max(480.0));
        let half_span = (panel_height * 0.22).clamp(160.0, 320.0);
        let x = 0.0;
        ((x, -half_span), (x, half_span))
    }

    fn auto_connect_position(
        source_x: f64,
        source_y: f64,
        source_width: f32,
        added_width: f32,
    ) -> (f64, f64) {
        const VERTICAL_GAP: f32 = 56.0;
        (
            source_x + f64::from(source_width - added_width) * 0.5,
            source_y + f64::from(NODE_HEIGHT + VERTICAL_GAP),
        )
    }

    fn cable_intersects_rect(path: &[egui::Pos2], rect: egui::Rect, padding: f32) -> bool {
        let rect = rect.expand(padding);
        path.windows(2)
            .any(|pair| rect.intersects(egui::Rect::from_two_pos(pair[0], pair[1])))
    }


    /// A free node goes below every existing node, in the first free slot of
    /// that bottom band, so repeated additions do not pile up.
    fn free_node_placement(document: &GraphDocument, node: &GraphNode) -> (f64, f64) {
        if document.nodes.is_empty() {
            return (0.0, 0.0);
        }
        let extent = node_extent(node);
        let bounds = document.nodes.iter().map(|existing| {
            let extent = node_extent(existing);
            extent.padded_rect(existing.x, existing.y)
        });
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for rect in bounds {
            min_x = min_x.min(f64::from(rect.min.x));
            max_x = max_x.max(f64::from(rect.max.x));
            max_y = max_y.max(f64::from(rect.max.y));
        }
        let gap = 48.0;
        let step_x = f64::from(extent.width) + gap;
        let step_y = f64::from(extent.height) + gap;
        let center_x = (min_x + max_x) * 0.5 - f64::from(extent.width) * 0.5;
        let mut y = max_y + gap;
        for _ in 0..64 {
            // Start under the middle of the existing tree and fill outward left
            // and right before moving to the next row.
            for offset in 0..64 {
                let side = if offset == 0 { 0.0 } else {
                    let magnitude = ((offset + 1) / 2) as f64;
                    if offset % 2 == 1 { magnitude } else { -magnitude }
                };
                let x = center_x + side * step_x;
                let candidate = extent.padded_rect(x, y);
                let clear = !document.nodes.iter().any(|existing| {
                    let rect = node_extent(existing).padded_rect(existing.x, existing.y);
                    candidate.intersects(rect)
                });
                if clear {
                    return (x, y);
                }
            }
            y += step_y;
        }
        (center_x, max_y + gap)
    }

    fn auto_connect_placement(
        document: &GraphDocument,
        added: &GraphNode,
        source: &str,
        initial: (f64, f64),
        cable_routing: CableRouting,
    ) -> (f64, f64, Vec<(String, f64, f64)>) {
        let added_extent = node_extent(added);
        let added_width = added_extent.width;
        let added_height = added_extent.height;
        let padded_rect = |x: f64, y: f64, extent: nodecraft_node_canvas::canvas::NodeExtent| {
            extent.padded_rect(x, y)
        };
        let node_rect = |node: &GraphNode, positions: &HashMap<String, (f64, f64)>| {
            let (x, y) = positions.get(&node.id).copied().unwrap_or((node.x, node.y));
            padded_rect(x, y, node_extent(node))
        };
        let mut positions: HashMap<String, (f64, f64)> = document
            .nodes
            .iter()
            .map(|node| (node.id.clone(), (node.x, node.y)))
            .collect();
        let mut moved: std::collections::BTreeMap<String, (f64, f64)> = std::collections::BTreeMap::new();
        let (mut x, y) = initial;
        for _ in 0..16 {
            let candidate = added_extent.padded_rect(x, y);
            let cables = NodeEditor::connection_paths(document, &positions, cable_routing);
            let node_blocker = document.nodes.iter().find(|node| {
                node.id != added.id && candidate.intersects(node_rect(node, &positions))
            });
            let cable_blocker = cables.iter().find_map(|(path, key)| {
                Self::cable_intersects_rect(path, candidate, 10.0).then(|| {
                    document
                        .nodes
                        .iter()
                        .find(|node| node.id == key.to)
                        .or_else(|| document.nodes.iter().find(|node| node.id == key.from))
                }).flatten()
            });
            let Some(blocker) = node_blocker.or(cable_blocker) else {
                return (x, y, moved.into_iter().map(|(id, (x, y))| (id, x, y)).collect());
            };
            let blocker_rect = node_rect(blocker, &positions);
            let candidates = [
                (f64::from(blocker_rect.max.x) + f64::from(added_extent.padding), blocker_rect.min.y as f64),
                (f64::from(blocker_rect.min.x) - f64::from(added_width) - f64::from(added_extent.padding), blocker_rect.min.y as f64),
                (blocker_rect.min.x as f64, f64::from(blocker_rect.max.y) + f64::from(added_extent.padding)),
                (blocker_rect.min.x as f64, f64::from(blocker_rect.min.y) - f64::from(added_extent.padding) - f64::from(added_height)),
            ];
            if let Some((side_x, side_y)) = candidates.into_iter().find(|(side_x, side_y)| {
                let side = added_extent.padded_rect(*side_x, *side_y);
                let clear_nodes = !document.nodes.iter().any(|node| {
                    node.id != added.id && side.intersects(node_rect(node, &positions))
                });
                let clear_cables = !cables.iter().any(|(path, _)| {
                    Self::cable_intersects_rect(path, side, 10.0)
                });
                clear_nodes && clear_cables
            }) {
                return (side_x, side_y, moved.into_iter().map(|(id, (x, y))| (id, x, y)).collect());
            }

            // No free side: move the blocking branch across, including children.
            let branch = {
                let mut branch = HashSet::new();
                let mut stack = vec![blocker.id.clone()];
                while let Some(current) = stack.pop() {
                    if !branch.insert(current.clone()) { continue; }
                    for link in document.connections.iter().filter(|link| link.from == current) {
                        stack.push(link.to.clone());
                    }
                }
                branch
            };
            let candidate_rect = added_extent.padded_rect(x, y);
            let delta = if candidate_rect.center().x >= blocker_rect.center().x {
                f64::from(candidate_rect.max.x - blocker_rect.min.x)
            } else {
                f64::from(candidate_rect.min.x - blocker_rect.max.x)
            };
            for id in &branch {
                if let Some((px, py)) = positions.get_mut(id) {
                    *px += delta;
                    moved.insert(id.clone(), (*px, *py));
                }
            }
            if branch.contains(source) {
                x += delta;
            }
        }
        (x, y, moved.into_iter().map(|(id, (x, y))| (id, x, y)).collect())
    }

    /// Apply edits made on a graph-owner inspector while viewing its body.
    /// Only value edits (not actions or port-count rewrites) are routed to the
    /// owning document; source ownership stays with the parent tab.
    fn apply_background_inspector_output(
        &mut self,
        document_id: &str,
        node_id: &str,
        output: inspector::Output,
    ) {
        if output.changes.is_empty() {
            return;
        }
        let Some(mut session) = self.documents.session(document_id).cloned() else {
            return;
        };
        let commands: Vec<Command> = output
            .changes
            .iter()
            .map(|change| Command::SetValue {
                node: node_id.to_owned(),
                field: change.field.clone(),
                value: change.value.clone(),
            })
            .collect();
        if session.apply_commands(&commands).is_ok() {
            let _ = self.documents.set_session(document_id.to_owned(), session);
            self.egui_ctx.request_repaint();
        }
    }

    /// Replace an open definition tab's body with the resized definition.
    fn refresh_definition_tab(&mut self, tab_id: u64, definition: &nodecraft_session::GraphDefinition) {
        let Some(tab) = self.tabs.iter().find(|tab| tab.id == tab_id).cloned() else {
            return;
        };
        let command = Command::SetRootDefinition {
            definition: definition.clone(),
        };
        if self.tabs.get(self.active_tab).is_some_and(|active| active.id == tab_id) {
            if let Some(session) = self.session.as_mut()
                && session.apply_derived_commands(&[command]).is_ok()
            {
                self.active_session_revision = session.revision();
                self.last_eval_key = None;
                self.publish_active_document();
            }
            self.mesh = None;
        } else if let Some(mut child) = self.documents.session(&tab.document_id).cloned() {
            if child.apply_derived_commands(&[command]).is_ok() {
                let _ = self.documents.set_session(tab.document_id.clone(), child);
                if let Some(tab) = self.tabs.iter_mut().find(|tab| tab.id == tab_id) {
                    tab.mesh = None;
                    tab.status = "Definition ports updated".to_owned();
                }
            }
        }
    }

    /// Refresh every open definition tab whose definition changed with the port
    /// resize, including definitions that merely contain nested instances.
    fn refresh_definition_tabs_for(
        &mut self,
        source_document_id: &str,
        definitions: &[nodecraft_session::GraphDefinition],
    ) {
        let tab_ids: Vec<(u64, nodecraft_session::GraphDefinition)> = definitions
            .iter()
            .flat_map(|definition| {
                self.tabs
                    .iter()
                    .filter(|tab| {
                        self.group_tabs.get(&tab.id).is_some_and(|meta| {
                            meta.view.document_id == source_document_id
                                && meta.view.definition_id.as_deref()
                                    == Some(definition.id.as_str())
                        })
                    })
                    .map(|tab| (tab.id, definition.clone()))
            })
            .collect();
        for (tab_id, definition) in tab_ids {
            self.refresh_definition_tab(tab_id, &definition);
        }
    }

    /// Resize the current document's root graph interface from a Graph
    /// Input/Output marker. Only shown outside definition tabs.
    fn resize_root_graph_interface(&mut self, inputs: Option<usize>, outputs: Option<usize>) {
        if let Some((source_document_id, definition_id)) = self.active_definition_view() {
            let Some(mut parent) = self.documents.session(&source_document_id).cloned() else {
                return;
            };
            let Some(definition) = parent.document().definition(&definition_id).cloned() else {
                return;
            };
            let input_count = inputs.unwrap_or(definition.interface.inputs.len());
            let output_count = outputs.unwrap_or(definition.interface.outputs.len());
            let Some((commands, refreshed)) = definition_port_commands(
                parent.document(),
                &definition_id,
                input_count,
                output_count,
            ) else {
                return;
            };
            if parent.apply_commands(&commands).is_err() {
                return;
            }
            let Some(target) = refreshed
                .iter()
                .find(|definition| definition.id == definition_id)
                .cloned()
            else {
                return;
            };
            if self
                .documents
                .set_session(source_document_id.clone(), parent)
                .is_err()
            {
                return;
            }
            if let Some(session) = self.session.as_mut()
                && session
                    .apply_derived_commands(&[Command::SetRootDefinition {
                        definition: target,
                    }])
                    .is_ok()
            {
                self.active_session_revision = session.revision();
                self.last_eval_key = None;
                self.publish_active_document();
            }
            if let Some(tab) = self
                .tabs
                .iter_mut()
                .find(|tab| tab.document_id == source_document_id)
            {
                tab.mesh = None;
            }
            self.evaluate();
            return;
        }
        let Some(session) = &self.session else { return };
        let Some(commands) = root_interface_commands(session.document(), inputs, outputs) else {
            return;
        };
        if self.apply_commands(&commands, Some("root-interface-ports")) {
            self.evaluate();
        }
    }

    /// Import an external graph definition (and its local definition closure)
    /// into the current document, then repoint this instance at the copy.
    fn make_graph_local(&mut self, graph_id: &str) {
        let Some(active_session) = &self.session else { return };
        let Some(node) = active_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == graph_id && node.node_type == "graph")
            .cloned()
        else {
            return;
        };
        let Some(GraphReference::External { path, definition }) = GraphReference::from_node(&node) else {
            return;
        };
        let Ok(external) = nodecraft_session::store::load(std::path::Path::new(&path)) else {
            self.status = format!("Could not load external graph {path}");
            return;
        };
        let Some(target) = external.definition(&definition).cloned() else {
            self.status = format!("External graph {path} has no definition {definition}");
            return;
        };
        let Some(active_document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else {
            return;
        };
        let source_document_id = self
            .tabs
            .iter()
            .find(|tab| tab.document_id == active_document_id)
            .and_then(|tab| self.group_tabs.get(&tab.id))
            .map(|meta| meta.view.document_id.clone())
            .unwrap_or_else(|| active_document_id.clone());
        let source = if source_document_id == active_document_id {
            self.session.clone()
        } else {
            self.documents.session(&source_document_id).cloned()
        };
        let Some(mut source) = source else { return };
        let mut used_ids: std::collections::BTreeSet<String> = source
            .document()
            .definitions
            .iter()
            .map(|definition| definition.id.clone())
            .chain(std::iter::once(source.document().root.id.clone()))
            .collect();
        let mut used_names: std::collections::BTreeSet<String> = source
            .document()
            .definitions
            .iter()
            .map(|definition| definition.name.clone())
            .collect();
        let mut imported = Vec::new();
        let mut mapping: std::collections::BTreeMap<String, (String, String)> = Default::default();
        let mut queue = vec![target.id.clone()];
        while let Some(old_id) = queue.pop() {
            if mapping.contains_key(&old_id) {
                continue;
            }
            let Some(original) = external.definition(&old_id) else { continue };
            let base_id = if old_id == target.id { old_id.clone() } else { format!("{old_id}_local") };
            let mut new_id = base_id.clone();
            let mut suffix = 2;
            while !used_ids.insert(new_id.clone()) {
                new_id = format!("{base_id}_{suffix}");
                suffix += 1;
            }
            let base_name = original.name.clone();
            let mut new_name = base_name.clone();
            suffix = 2;
            while !used_names.insert(new_name.clone()) {
                new_name = format!("{base_name} {suffix}");
                suffix += 1;
            }
            for child in &original.nodes {
                if child.node_type == "graph"
                    && let Some(GraphReference::Local(id)) = GraphReference::from_node(child)
                {
                    queue.push(id);
                }
            }
            mapping.insert(old_id, (new_id, new_name));
            imported.push(original.clone());
        }
        let mut definitions = Vec::new();
        for mut definition in imported {
            let Some((new_id, new_name)) = mapping
                .get(&definition.id)
                .cloned()
            else {
                continue;
            };
            definition.id = new_id.clone();
            definition.name = new_name.clone();
            for node in &mut definition.nodes {
                if node.node_type == "graph"
                    && let Some(GraphReference::Local(id)) = GraphReference::from_node(node)
                    && let Some((mapped, _)) = mapping.get(&id)
                {
                    node.values["definition"] = serde_json::Value::String(mapped.clone());
                }
            }
            definitions.push(definition);
        }
        let Some((target_id, target_name)) = mapping.get(&target.id).cloned() else {
            return;
        };
        let mut commands: Vec<Command> = definitions
            .into_iter()
            .map(|definition| Command::AddDefinition { definition })
            .collect();
        commands.push(Command::SetValue {
            node: graph_id.to_owned(),
            field: "definition".to_owned(),
            value: serde_json::Value::String(target_id),
        });
        commands.push(Command::SetValue {
            node: graph_id.to_owned(),
            field: "name".to_owned(),
            value: serde_json::Value::String(target_name),
        });
        if source_document_id == active_document_id {
            if self.apply_commands(&commands, Some(&format!("graph-local:{graph_id}"))) {
                self.evaluate();
            }
            return;
        }
        let add_count = commands.len().saturating_sub(2);
        if source.apply_commands(&commands[..add_count]).is_err() {
            return;
        }
        let _ = self
            .documents
            .set_session(source_document_id, source);
        let _ = self.apply_commands(
            &commands[add_count..],
            Some(&format!("graph-local:{graph_id}")),
        );
        self.evaluate();
    }

    /// Clone this instance's local definition and point the instance at the
    /// copy, giving the definition a fresh id and name.
    fn make_graph_unique(&mut self, graph_id: &str) {
        let Some(active_session) = &self.session else { return };
        let Some(node) = active_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == graph_id && node.node_type == "graph")
            .cloned()
        else {
            return;
        };
        let Some(GraphReference::Local(definition_id)) = GraphReference::from_node(&node) else {
            self.status = "Make unique is only available for local definitions".to_owned();
            return;
        };
        let Some(active_document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else {
            return;
        };
        let source_document_id = self
            .tabs
            .iter()
            .find(|tab| tab.document_id == active_document_id)
            .and_then(|tab| self.group_tabs.get(&tab.id))
            .map(|meta| meta.view.document_id.clone())
            .unwrap_or_else(|| active_document_id.clone());
        let source = if source_document_id == active_document_id {
            self.session.clone()
        } else {
            self.documents.session(&source_document_id).cloned()
        };
        let Some(mut source) = source else { return };
        let Some(original) = source.document().definition(&definition_id).cloned() else {
            return;
        };
        let mut id = format!("{definition_id}_copy");
        let mut suffix = 2;
        while source.document().root.id == id
            || source.document().definitions.iter().any(|definition| definition.id == id)
        {
            id = format!("{definition_id}_copy{suffix}");
            suffix += 1;
        }
        let mut name = format!("{} Copy", original.name);
        suffix = 2;
        while source.document().definitions.iter().any(|definition| definition.name == name) {
            name = format!("{} Copy {suffix}", original.name);
            suffix += 1;
        }
        let mut definition = original;
        definition.id = id.clone();
        definition.name = name.clone();
        let mut commands = vec![Command::AddDefinition {
            definition,
        }];
        if source_document_id == active_document_id {
            commands.push(Command::SetValue {
                node: graph_id.to_owned(),
                field: "definition".to_owned(),
                value: serde_json::Value::String(id),
            });
            commands.push(Command::SetValue {
                node: graph_id.to_owned(),
                field: "name".to_owned(),
                value: serde_json::Value::String(name),
            });
            if self.apply_commands(&commands, Some(&format!("graph-unique:{graph_id}"))) {
                self.evaluate();
            }
            return;
        }
        if source.apply_commands(&commands[..1]).is_err() {
            return;
        }
        let _ = self
            .documents
            .set_session(source_document_id.clone(), source);
        let _ = self.apply_commands(
            &[
                Command::SetValue {
                    node: graph_id.to_owned(),
                    field: "definition".to_owned(),
                    value: serde_json::Value::String(id),
                },
                Command::SetValue {
                    node: graph_id.to_owned(),
                    field: "name".to_owned(),
                    value: serde_json::Value::String(name),
                },
            ],
            Some(&format!("graph-unique:{graph_id}")),
        );
        self.evaluate();
    }

    /// Resize a local graph definition from its graph-instance inspector.
    ///
    /// Editing a definition while one of its body tabs is open would leave that
    /// tab's marker nodes stale, so this path asks for the tab to be closed first.
    fn resize_graph_definition_ports(&mut self, graph_id: &str, inputs: usize, outputs: usize) {
        let Some(active_session) = &self.session else { return };
        let Some(node) = active_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == graph_id && node.node_type == "graph")
            .cloned()
        else {
            return;
        };
        let Some(active_document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else {
            return;
        };
        let Some(reference) = GraphReference::from_node(&node) else {
            return;
        };
        let (source_document_id, definition_id) = match reference {
            GraphReference::Local(definition_id) => {
                let source_document_id = self
                    .tabs
                    .iter()
                    .find(|tab| tab.document_id == active_document_id)
                    .and_then(|tab| self.group_tabs.get(&tab.id))
                    .map(|meta| meta.view.document_id.clone())
                    .unwrap_or_else(|| active_document_id.clone());
                (source_document_id, definition_id)
            }
            GraphReference::External { path, definition } => {
                let Some(source_document_id) = self
                    .documents
                    .id_for_path(std::path::Path::new(&path))
                    .map(str::to_owned)
                else {
                    self.status = "Open the external graph before editing its ports".to_owned();
                    return;
                };
                (source_document_id, definition)
            }
        };
        let open_definition_tabs: Vec<u64> = self
            .tabs
            .iter()
            .filter(|tab| {
                self.group_tabs.get(&tab.id).is_some_and(|meta| {
                    meta.view.document_id == source_document_id
                        && meta.view.definition_id.as_deref() == Some(definition_id.as_str())
                })
            })
            .map(|tab| tab.id)
            .collect();
        // Flush body edits before resizing the interface, then replace each open
        // body session with the resized definition so marker nodes cannot go stale.
        for tab_id in &open_definition_tabs {
            self.sync_group_tab(*tab_id);
        }
        let source_session = if source_document_id == active_document_id {
            self.session.clone()
        } else {
            self.documents.session(&source_document_id).cloned()
        };
        let Some(mut source_session) = source_session else { return };
        let Some((commands, refreshed_definitions)) = definition_port_commands(
            source_session.document(),
            &definition_id,
            inputs,
            outputs,
        ) else {
            return;
        };
        let target = refreshed_definitions
            .iter()
            .find(|definition| definition.id == definition_id)
            .cloned()
            .unwrap_or_else(|| source_session.document().root.clone());
        if source_document_id == active_document_id {
            if self.apply_commands(&commands, Some(&format!("graph-ports:{graph_id}"))) {
                self.refresh_definition_tabs_for(&source_document_id, &refreshed_definitions);
                self.evaluate();
            }
            return;
        }
        if source_session.apply_commands(&commands).is_err() {
            return;
        }
        let _ = self
            .documents
            .set_session(source_document_id.clone(), source_session);
        if let Some(parent_tab) = self
            .tabs
            .iter_mut()
            .find(|tab| tab.document_id == source_document_id)
        {
            parent_tab.mesh = None;
            parent_tab.status = "Definition ports updated".to_owned();
        }
        self.refresh_definition_tabs_for(&source_document_id, &refreshed_definitions);
        let _ = self.apply_commands(
            &[
                Command::SetValue {
                    node: graph_id.to_owned(),
                    field: "inputs".to_owned(),
                    value: serde_json::to_value(&target.interface.inputs)
                        .unwrap_or(serde_json::Value::Null),
                },
                Command::SetValue {
                    node: graph_id.to_owned(),
                    field: "outputs".to_owned(),
                    value: serde_json::to_value(&target.interface.outputs)
                        .unwrap_or(serde_json::Value::Null),
                },
            ],
            Some(&format!("graph-ports:{graph_id}")),
        );
        self.evaluate();
    }

    /// Rename a local graph definition from its graph-instance inspector.
    fn rename_graph_definition(&mut self, graph_id: &str, name: String) {
        let Some(active_session) = &self.session else { return };
        let Some(node) = active_session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == graph_id && node.node_type == "graph")
            .cloned()
        else {
            return;
        };
        let Some(active_document_id) = self.tabs.get(self.active_tab).map(|tab| tab.document_id.clone()) else {
            return;
        };
        let Some(reference) = GraphReference::from_node(&node) else {
            return;
        };
        let (source_document_id, definition_id) = match reference {
            GraphReference::Local(definition_id) => {
                let source_document_id = self
                    .tabs
                    .iter()
                    .find(|tab| tab.document_id == active_document_id)
                    .and_then(|tab| self.group_tabs.get(&tab.id))
                    .map(|meta| meta.view.document_id.clone())
                    .unwrap_or_else(|| active_document_id.clone());
                (source_document_id, definition_id)
            }
            GraphReference::External { path, definition } => {
                let Some(source_document_id) = self
                    .documents
                    .id_for_path(std::path::Path::new(&path))
                    .map(str::to_owned)
                else {
                    self.status = "Open the external graph before renaming it".to_owned();
                    return;
                };
                (source_document_id, definition)
            }
        };
        let mut source = if source_document_id == active_document_id {
            self.session.clone()
        } else {
            self.documents.session(&source_document_id).cloned()
        };
        let Some(source_session) = source.as_mut() else {
            return;
        };
        let Some(mut definition) = source_session
            .document()
            .definition(&definition_id)
            .cloned()
        else {
            return;
        };
        definition.name = name.clone();
        let mut commands = vec![Command::SetDefinition { definition }];
        let references = |node: &GraphNode| {
            node.node_type == "graph"
                && matches!(
                    GraphReference::from_node(node),
                    Some(GraphReference::Local(ref id)) if id == &definition_id
                )
        };
        for node in source_session.document().nodes.clone() {
            if references(&node) {
                commands.push(Command::SetValue {
                    node: node.id,
                    field: "name".to_owned(),
                    value: serde_json::Value::String(name.clone()),
                });
            }
        }
        for definition in source_session.document().definitions.clone() {
            if definition.id == definition_id {
                continue;
            }
            let mut updated = definition;
            let mut changed = false;
            for node in &mut updated.nodes {
                if references(node) {
                    node.values["name"] = serde_json::Value::String(name.clone());
                    changed = true;
                }
            }
            if changed {
                commands.push(Command::SetDefinition { definition: updated });
            }
        }
        if source_document_id == active_document_id {
            let _ = self.apply_commands(&commands, Some(&format!("graph-name:{graph_id}")));
        } else {
            if source_session.apply_commands(&commands).is_err() {
                return;
            }
            let updated_source = source_session.clone();
            let _ = self
                .documents
                .set_session(source_document_id.clone(), updated_source);
            let _ = self.apply_commands(
                &[Command::SetValue {
                    node: graph_id.to_owned(),
                    field: "name".to_owned(),
                    value: serde_json::Value::String(name),
                }],
                Some(&format!("graph-name:{graph_id}")),
            );
        }
        self.egui_ctx.request_repaint();
    }

    /// Apply a body-port count change to an inline graph. Port arrays, marker
    /// nodes, child wires and parent wires must move as one undoable edit.
    fn apply_graph_ports(&mut self, node_id: &str, inputs: usize, outputs: usize) {
        let Some(session) = &self.session else { return };
        let Some(node) = session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == node_id)
            .cloned()
        else {
            return;
        };
        let Some(commands) = graph_port_commands(
            node_id,
            &node,
            &session.document().connections,
            inputs,
            outputs,
        ) else {
            return;
        };
        if self.apply_commands(&commands, Some(&format!("graph-ports:{node_id}"))) {
            self.evaluate();
        }
    }

    fn add_tool_node(
        &mut self,
        tool: ToolItem,
        workspace: egui::Rect,
        connect_source: Option<String>,
    ) {
        let (definition_ids, definition_names) = self.definition_registry();
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        let mut used: HashSet<String> = session.document().nodes.iter().map(|node| node.id.clone()).collect();
        let id = unique_node_id(tool.node_type, &mut used);
        let mut graph_definition = None;
        let values = if tool.node_type == "graph" {
            let definition_id = unique_graph_definition_id(
                &format!("{id}_definition"),
                &definition_ids,
            );
            let definition_name = unique_graph_definition_name(&definition_names);
            graph_definition = Some(nodecraft_session::GraphDefinition {
                id: definition_id.clone(),
                name: definition_name.clone(),
                kind: "graph".to_owned(),
                interface: nodecraft_session::GraphInterface {
                    inputs: vec!["shape".into()],
                    outputs: vec!["shape".into()],
                },
                nodes: {
                    let ((in_x, in_y), (out_x, out_y)) = Self::new_graph_port_positions(workspace);
                    vec![
                        GraphNode {
                            expressions: Default::default(),
                            id: "in".into(),
                            node_type: "groupinput".into(),
                            values: serde_json::json!({"ports":["shape"]}),
                            suppressed: false,
                            x: in_x,
                            y: in_y,
                        },
                        GraphNode {
                            expressions: Default::default(),
                            id: "out".into(),
                            node_type: "groupoutput".into(),
                            values: serde_json::json!({"ports":["shape"]}),
                            suppressed: false,
                            x: out_x,
                            y: out_y,
                        },
                    ]
                },
                connections: vec![GraphConnection {
                    from: "in".into(),
                    to: "out".into(),
                    input: "shape".into(),
                    out: "shape".into(),
                }],
            });
            serde_json::json!({
                "definition": definition_id,
                "name": definition_name,
                "inputs": ["shape"],
                "outputs": ["shape"],
            })
        } else {
            match tool.primitive {
            Some(primitive) => serde_json::json!({ "primitiveType": primitive }),
            None if tool.node_type == "group" => serde_json::json!({
                "collapsed": false,
                "nodes": [
                    {"id":"in","type":"groupinput","values":{"ports":["shape"]},"suppressed":false,"x":-120.0,"y":0.0},
                    {"id":"out","type":"groupoutput","values":{"ports":["shape"]},"suppressed":false,"x":120.0,"y":0.0}
                ],
                "connections": [{"from":"in","to":"out","in":"shape","out":"shape"}],
                "inputs": [{"port":"shape","node":"in"}],
                "outputs": [{"port":"shape","node":"out"}]
            }),
            None => serde_json::json!({}),
            }
        };
        let mut node = GraphNode { expressions: Default::default(),
            id: id.clone(),
            node_type: tool.node_type.to_owned(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let connect = connect_source.and_then(|selected| {
            let session = self.session.as_ref()?;
            let previous = session.document().nodes.iter().find(|node| node.id == selected)?;
            let (input, output) = Self::auto_connect_ports(&previous.node_type, &tool.node_type)?;
            Some((
                selected,
                input,
                output,
                previous.x,
                previous.y,
                node_width(previous),
            ))
        });
        let mut moved = Vec::new();
        if let Some((selected, _, _, x, y, source_width)) = connect.clone() {
            let (x, y) = Self::auto_connect_position(x, y, source_width, node_width(&node));
            let session = self.session.as_ref().expect("session checked above");
            let (x, y, branch_moves) =
                Self::auto_connect_placement(session.document(), &node, &selected, (x, y), self.cable_routing);
            node.x = x;
            node.y = y;
            moved = branch_moves;
        } else {
            let (x, y) = Self::free_node_placement(session.document(), &node);
            node.x = x;
            node.y = y;
        }
        let added_rect = egui::Rect::from_min_size(
            egui::pos2(node.x as f32, node.y as f32),
            egui::vec2(node_width(&node), NODE_HEIGHT),
        );
        let selection_before = self.selected_nodes.clone();
        let connections_before = self.selected_connections.clone();
        let mut commands = Vec::new();
        if let Some(definition) = graph_definition {
            commands.push(Command::AddDefinition { definition });
        }
        commands.push(Command::AddNode { node });
        if let Some((selected, input, output, _, _, _)) = connect {
            commands.push(Command::Connect {
                from: selected,
                to: id.clone(),
                input: input.to_owned(),
                out: output.to_owned(),
            });
        }
        for (moved_id, x, y) in moved {
            commands.push(Command::SetPosition {
                node: moved_id,
                x,
                y,
            });
        }
        let connected = commands.iter().any(|command| matches!(command, Command::Connect { .. }));
        if self.apply_commands(&commands, Some("tool-dock-add-node")) {
            self.node_editor.begin_node_fade(&id);
            self.node_editor
                .ensure_graph_rect_visible(workspace, added_rect, 24.0);
            self.egui_ctx.request_repaint();
            self.tool_add_source = None;
            if connected || selection_before.is_empty() {
                // A successful auto-connect hands selection to the new node;
                // a free node leaves the current selection alone.
                self.selected_nodes.clear();
                self.selected_nodes.insert(id.clone());
                self.selected_connections.clear();
            } else {
                self.selected_nodes = selection_before;
                self.selected_connections = connections_before;
            }
            self.status = if connected {
                format!("Added {} and connected it downstream", tool.label)
            } else {
                format!("Added {}", tool.label)
            };
            self.evaluate();
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

    fn duplicate_selected_nodes(&mut self) {
        let Some(session) = &self.session else {
            self.status = "No document".to_owned();
            return;
        };
        let selected: HashSet<&str> = self.selected_nodes.iter().map(String::as_str).collect();
        let Some(payload) = clipboard_payload(session.document(), &selected) else {
            self.status = "No nodes selected".to_owned();
            return;
        };
        let count = payload.nodes.len();
        let (commands, selected_nodes, _) = paste_plan(session.document(), payload, None);
        self.selected_nodes = selected_nodes;
        self.selected_connections.clear();
        if self.apply_commands(&commands, Some("duplicate-nodes")) {
            self.evaluate();
            self.status = format!(
                "Duplicated {count} node{}",
                if count == 1 { "" } else { "s" }
            );
        } else {
            self.selected_nodes.clear();
        }
    }

    fn cut_selected_nodes(&mut self) {
        if !self.copy_selected_nodes() {
            return;
        }
        let ids: Vec<String> = self
            .selected_nodes
            .iter()
            .filter(|id| {
                self.session
                    .as_ref()
                    .and_then(|session| session.document().nodes.iter().find(|node| &node.id == *id))
                    .is_none_or(|node| !is_graph_interface_node(node))
            })
            .cloned()
            .collect();
        let count = ids.len();
        let removed_nodes: HashSet<String> = ids.iter().cloned().collect();
        let commands: Vec<Command> = ids
            .into_iter()
            .map(|id| Command::RemoveNode { id })
            .collect();
        let (commands, removed_definitions) =
            self.add_definition_removals(commands, &removed_nodes);
        if self.apply_commands(&commands, Some("cut-nodes")) {
            self.close_tabs_for_removed_definitions(&removed_definitions);
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

    /// Enter a node through a double-click.
    ///
    /// Selection is handled separately and always runs first; activation then
    /// decides what "enter" means. Keep this as the one place that policy lives.
    fn activate_node(&mut self, id: &str) {
        let Some(node_type) = self
            .session
            .as_ref()
            .and_then(|session| session.document().nodes.iter().find(|node| node.id == id))
            .map(|node| node.node_type.clone())
        else {
            return;
        };
        match node_activation(&node_type) {
            NodeActivation::EnterGroup => self.enter_group(id),
            NodeActivation::EnterGraph => self.enter_graph(id),
            NodeActivation::EditNode => {
                self.selected_nodes.clear();
                self.selected_nodes.insert(id.to_owned());
                self.selected_connections.clear();
                let expected = self.session.as_ref().and_then(|session| {
                    edit_preview_input(session.document(), id)
                });
                if self.edit_preview_target != expected {
                    self.edit_preview = None;
                    self.edit_preview_target = None;
                    // The selection prefetch may have produced the same key but
                    // not the branch mesh; force the main pass to refresh.
                    self.last_eval_key = None;
                }
                self.edit_target = Some(id.to_owned());
                self.open_inspector();
                self.status = format!("Editing {id}");
                self.evaluate();
            }
        }
    }

    /// Branch ghost overlay shown while a transform edit target is active.
    ///
    /// Only used when the breakpoint result has consumed the edited branch
    /// (no per-body provenance). The raw input mesh is placed through the same
    /// downstream chain the gizmo uses, then drawn over a dimmed breakpoint mesh.
    fn transform_ghost(&self) -> Option<TransformGhost> {
        if self.gizmo_drag.is_some() {
            return None;
        }
        let session = self.session.as_ref()?;
        transform_ghost_for(
            session.document(),
            self.edit_target.as_deref()?,
            self.eval_target.as_deref()?,
            self.edit_preview.as_ref(),
            self.mesh.as_ref(),
            ghost_alpha(&self.viewport.view()),
        )
    }

    fn refresh_edit_overlay(&mut self) {
        let Some(ghost) = self.transform_ghost() else {
            return;
        };
        self.viewport.set_edit_ghost(
            &ghost.base,
            &ghost.branch,
            ghost.base_transform.to_column_major_f32(),
            ghost.alpha,
        );
    }

    fn selected_transform_gizmo(&self) -> Option<Gizmo> {
        if !selection_is_edit_target(&self.selected_nodes, self.edit_target.as_deref()) {
            return None;
        }
        let selected_id = self.selected_nodes.iter().next()?.clone();
        let session = self.session.as_ref()?;
        let document = session.document();
        let node = document.nodes.iter().find(|node| node.id == selected_id)?;
        if node.node_type != "translate" {
            return None;
        }
        // Use the target that the viewport is actually displaying, not the
        // separate viewer-breakpoint state.
        let target = self.eval_target.as_deref()?;
        let frame = transform_frame(document, &selected_id, target)?;
        let value = |key: &str| node.values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let axes = [
            frame.axes_basis.apply_vector([1.0, 0.0, 0.0]),
            frame.axes_basis.apply_vector([0.0, 1.0, 0.0]),
            frame.axes_basis.apply_vector([0.0, 0.0, 1.0]),
        ];
        Some(Gizmo {
            node_id: selected_id,
            pivot: frame.pivot,
            x: value("x"),
            y: value("y"),
            z: value("z"),
            rx: value("rx"),
            ry: value("ry"),
            rz: value("rz"),
            world_to_local: frame.chain.inverse_linear(),
            axes,
        })
    }

    /// Resolve the immutable parts of a drag preview — the downstream placement
    /// chain and the display bodies it feeds — once, when the drag starts.
    /// True when any upstream input of `node` resolves to a sketch/spline.
    fn transform_input_contains_sketch(document: &GraphDocument, node: &str) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![node.to_owned()];
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let Some(current) = document.nodes.iter().find(|node| node.id == id) else {
                continue;
            };
            if matches!(current.node_type.as_str(), "sketch" | "spline") {
                return true;
            }
            for link in document.connections.iter().filter(|link| link.to == id) {
                stack.push(link.from.clone());
            }
        }
        false
    }

    fn begin_gizmo_preview(&self, node: &GraphNode) -> Option<GizmoDragPreview> {
        use nodecraft_session::placement::Affine3;
        if node.node_type != "translate" {
            return None;
        }
        let session = self.session.as_ref()?;
        let document = session.document();
        let target = self.eval_target.as_deref()?;
        if target == node.id {
            // Editing the displayed transform itself: the merged breakpoint body
            // is the thing being moved, so its own placement is the whole delta.
            return Some(GizmoDragPreview {
                node_id: node.id.clone(),
                start_values: node.values.clone(),
                chain: Affine3::identity(),
                scope: PreviewScope::All,
                base_transform: Affine3::identity(),
                preview_mesh: None,
                branch_preview: false,
            });
        }
        // Any upstream edit point uses the single transform-ghost path. If the
        // branch mesh is not ready, show no preview rather than falling back to
        // the breakpoint mesh.
        let frame = transform_frame(document, &node.id, target)?;
        Some(GizmoDragPreview {
            node_id: node.id.clone(),
            start_values: node.values.clone(),
            chain: frame.chain,
            scope: PreviewScope::All,
            base_transform: frame.placement,
            // The branch mesh and its GPU plans may still be in flight. The drag
            // values can start immediately; `attach_pending_edit_ghost` adds the
            // GPU ghost when the preview evaluation returns.
            preview_mesh: None,
            branch_preview: true,
        })
    }

    /// Per-frame preview update: only the values change now. The chain, scope
    /// and (for large meshes) the affected geometry are all cached.
    fn refresh_gizmo_preview(&mut self) {
        let resolved = match self.gizmo_drag.as_ref() {
            Some(drag) => {
                let current = self.session.as_ref().and_then(|session| {
                    session
                        .document()
                        .nodes
                        .iter()
                        .find(|node| node.id == drag.node_id)
                        .map(|node| node.values.clone())
                });
                current.map(|current| {
                    let delta = nodecraft_session::placement::preview_transform(
                        drag.chain,
                        &drag.start_values,
                        &current,
                    );
                    let matrix = if drag.branch_preview {
                        delta.compose(drag.base_transform).to_column_major_f32()
                    } else {
                        delta.to_column_major_f32()
                    };
                    (matrix, drag.scope.clone())
                })
            }
            None => None,
        };
        let (matrix, scope) = resolved
            .map(|(matrix, scope)| (Some(matrix), scope))
            .unwrap_or((None, PreviewScope::All));
        let alpha = self
            .gizmo_drag
            .as_ref()
            .is_some_and(|drag| drag.branch_preview)
            .then(|| ghost_alpha(&self.viewport.view()))
            .unwrap_or(1.0);
        if self
            .gizmo_drag
            .as_ref()
            .is_some_and(|drag| drag.branch_preview)
        {
            self.viewport.update_ghost_transform(matrix, alpha);
        } else {
            self.viewport.set_preview_transform_with_alpha(matrix, scope, alpha);
        }
    }

    /// If an upstream drag started before the branch preview mesh arrived,
    /// attach the ghost as soon as the evaluation delivers it.
    fn attach_pending_edit_ghost(&mut self) {
        let Some(edit_preview) = self.edit_preview.clone() else { return };
        let Some(drag) = self.gizmo_drag.as_ref() else { return };
        if !drag.branch_preview || drag.preview_mesh.is_some() {
            return;
        }
        if !self.viewport.has_edit_ghost_plans() {
            return;
        }
        let (Some(edit_target), Some(target)) = (self.edit_target.clone(), self.eval_target.clone()) else {
            return;
        };
        let Some(session) = self.session.as_ref() else { return };
        let document = session.document();
        let ghost = transform_ghost_for(
            document,
            &edit_target,
            &target,
            Some(&edit_preview),
            self.mesh.as_ref(),
            ghost_alpha(&self.viewport.view()),
        );
        let Some(ghost) = ghost else { return };
        let current = document
            .nodes
            .iter()
            .find(|node| node.id == drag.node_id)
            .map(|node| node.values.clone());
        let Some(current) = current else { return };
        let delta = nodecraft_session::placement::preview_transform(
            drag.chain,
            &drag.start_values,
            &current,
        );
        let matrix = delta.compose(drag.base_transform).to_column_major_f32();
        let alpha = ghost_alpha(&self.viewport.view());
        let branch = ghost.branch.clone();
        self.viewport.set_edit_ghost(&ghost.base, &branch, matrix, alpha);
        if let Some(drag) = self.gizmo_drag.as_mut() {
            drag.preview_mesh = Some(branch);
        }
    }

    /// Consume one frame of transform-triad state: write the live values to the
    /// document, move the display mesh with a preview transform, and re-evaluate
    /// exactly once when the drag is released.
    fn handle_gizmo_update(&mut self, selected_node: Option<&GraphNode>) {
        let Some(update) = self.viewport.take_gizmo_update() else {
            // Without a triad frame a drag cannot continue (selection changed,
            // chrome hidden, or the document was replaced); retire its preview
            // rather than leaving the display geometry displaced.
            if self.gizmo_dragging || self.gizmo_drag.is_some() {
                self.gizmo_dragging = false;
                self.gizmo_drag = None;
                self.gizmo_edited = false;
                if self.gizmo_publish_pending {
                    self.gizmo_publish_pending = false;
                    let started = Instant::now();
                    self.publish_active_document();
                    if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                        eprintln!("[gizmo] publish {}ms", started.elapsed().as_millis());
                    }
                }
                self.viewport.set_preview_transform(None, PreviewScope::All);
            }
            return;
        };
        self.gizmo_dragging = update.dragging;
        if update.started {
            self.gizmo_drag = selected_node.and_then(|node| self.begin_gizmo_preview(node));
            self.gizmo_edited = false;
            self.gizmo_publish_pending = false;
            if let Some(mesh) = self.gizmo_drag.as_ref().and_then(|drag| drag.preview_mesh.clone())
                && !self
                    .gizmo_drag
                    .as_ref()
                    .is_some_and(|drag| drag.branch_preview)
            {
                self.viewport.set_mesh_preserving_camera(Some(&mesh));
            }
            // The service-published edit ghost may already have arrived before
            // the drag began; attach it now instead of waiting for another
            // preview result.
            self.attach_pending_edit_ghost();
        }
        if let Some(node) = selected_node
            && node.node_type == "translate"
            && !update.changes.is_empty()
        {
            // One command batch per frame: each separate apply would clone and
            // dirty-walk the document again.
            let commands: Vec<Command> = update
                .changes
                .into_iter()
                .map(|(field, value)| Command::SetValue {
                    node: node.id.clone(),
                    field,
                    value: serde_json::Value::from(value),
                })
                .collect();
            let started = Instant::now();
            let changed = self.apply_commands(&commands, Some(&format!("gizmo:{}", node.id)));
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!("[gizmo] apply_commands {}ms", started.elapsed().as_millis());
            }
            self.gizmo_edited |= changed;
        }
        if update.dragging || update.started || update.ended {
            // The session now holds this frame's values, so the delta maps the
            // last evaluated geometry onto the dragged placement.
            self.refresh_gizmo_preview();
        }
        if update.ended {
            let had_preview_mesh = self.gizmo_drag.as_ref().is_some_and(|drag| drag.preview_mesh.is_some());
            self.gizmo_dragging = false;
            self.gizmo_drag = None;
            if self.gizmo_edited {
                self.gizmo_edited = false;
                if self.gizmo_publish_pending {
                    self.gizmo_publish_pending = false;
                    self.publish_active_document();
                }
                // Keep the preview on screen while the kernel catches up; the
                // authoritative mesh retires it when it arrives.
                self.preserve_view_for_next_mesh = true;
                let started = Instant::now();
                self.evaluate();
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[gizmo] release evaluate {}ms", started.elapsed().as_millis());
                }
                if self.eval_pending.is_none() {
                    // Nothing started (the viewed target is unaffected), so no
                    // new mesh will replace the preview.
                    self.viewport.set_preview_transform(None, PreviewScope::All);
                    if had_preview_mesh {
                        self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
                    }
                }
            } else {
                self.viewport.set_preview_transform(None, PreviewScope::All);
                if had_preview_mesh {
                    self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
                }
            }
        }
    }

    /// Make the active editor revision the canonical document that linked roots
    /// flatten from. Tabs remain views; the document store is the shared source.
    fn publish_active_document(&mut self) {
        if self.active_definition_view().is_some() {
            return;
        }
        let Some(session) = &self.session else { return };
        let Some(tab) = self.tabs.get(self.active_tab) else { return };
        self.documents
            .publish_document(&tab.document_id, session.document());
    }

    fn active_definition_view(&self) -> Option<(String, String)> {
        let tab = self.tabs.get(self.active_tab)?;
        let meta = self.group_tabs.get(&tab.id)?;
        Some((meta.view.document_id.clone(), meta.view.definition_id.clone()?))
    }

    /// Definition-backed tabs edit the parent document session directly. The
    /// visible child session is a projection and is refreshed without its own
    /// history; the parent session owns the undo entry.
    fn apply_definition_view_commands(
        &mut self,
        source_document_id: &str,
        definition_id: &str,
        commands: &[Command],
    ) -> bool {
        let Some(mut parent) = self.documents.session(source_document_id).cloned() else {
            return false;
        };
        let scope = self.history_scope();
        match parent.apply_definition_commands_scoped(definition_id, &scope, commands) {
            Ok(change) if change.changed => {
                let updated = parent
                    .document()
                    .definition(definition_id)
                    .cloned();
                let projection_definitions = parent.document().definitions.clone();
                if self
                    .documents
                    .set_session(source_document_id.to_owned(), parent)
                    .is_err()
                {
                    return false;
                }
                if let Some(updated) = updated {
                    let layout_only = commands_are_layout_only(commands);
                    let structural = commands.iter().any(|command| {
                        matches!(
                            command,
                            Command::AddNode { .. }
                                | Command::RemoveNode { .. }
                                | Command::AddDefinition { .. }
                                | Command::RemoveDefinition { .. }
                        )
                    });
                    if structural {
                        // Structural graph-view edits rebuild the projection so a
                        // newly added node/definition is present immediately rather
                        // than only after re-entering the tab.
                        let projection = GraphDocument {
                            schema_version: 1,
                            id: format!("{}::graph", updated.id),
                            root: updated,
                            definitions: projection_definitions,
                        };
                        self.install_session(Some(Session::new(projection)));
                        self.active_session_revision =
                            self.session.as_ref().map_or(0, Session::revision);
                        self.last_eval_key = None;
                        self.node_editor.request_fit();
                    } else if let Some(session) = self.session.as_mut()
                        && session
                            .apply_derived_commands(&[Command::SetRootDefinition {
                                definition: updated,
                            }])
                            .is_ok()
                    {
                        self.active_session_revision = session.revision();
                        if !layout_only {
                            self.last_eval_key = None;
                        }
                    }
                    if !layout_only {
                        self.auto_eval_due = Some(Instant::now());
                    }
                }
                if let Some(tab) = self
                    .tabs
                    .iter_mut()
                    .find(|tab| tab.document_id == source_document_id)
                {
                    if !commands_are_layout_only(commands) {
                        tab.mesh = None;
                    }
                    tab.status = "Definition updated".to_owned();
                }
                true
            }
            Ok(_) => false,
            Err(error) => {
                self.error = Some(format!("Graph edit failed: {error}"));
                false
            }
        }
    }

    fn apply_commands(&mut self, commands: &[Command], coalesce: Option<&str>) -> bool {
        if let Some((source_document_id, definition_id)) = self.active_definition_view() {
            return self.apply_definition_view_commands(
                &source_document_id,
                &definition_id,
                commands,
            );
        }
        let scope = self.history_scope();
        let Some(session) = self.session.as_mut() else { return false };
        match session.apply_commands_scoped(
            commands,
            &scope,
            coalesce,
            std::time::Duration::from_millis(700),
        ) {
            Ok(change) => {
                if change.changed && !change.dirty_nodes.is_empty() {
                    self.pending_dirty.extend(change.dirty_nodes);
                    self.auto_eval_due = Some(Instant::now());
                }
                let changed = change.changed;
                if changed {
                    if self.gizmo_dragging {
                        // A drag writes values every frame; publishing clones the
                        // whole document, so defer it until the drag releases.
                        self.gizmo_publish_pending = true;
                    } else {
                        self.publish_active_document();
                    }
                }
                changed
            }
            Err(error) => {
                self.error = Some(format!("Graph edit failed: {error}"));
                false
            }
        }
    }

    fn history_scope(&self) -> String {
        let tab_id = self.tabs.get(self.active_tab).map_or(0, |tab| tab.id);
        if let Some((_, definition_id)) = self.active_definition_view() {
            return format!("definition:{definition_id}:view:{tab_id}");
        }
        format!("view:{tab_id}")
    }

    fn can_undo(&self) -> bool {
        let scope = self.history_scope();
        if let Some((source_document_id, _)) = self.active_definition_view() {
            return self
                .documents
                .session(&source_document_id)
                .is_some_and(|session| session.has_scope_undo(&scope));
        }
        self.session
            .as_ref()
            .is_some_and(|session| session.has_scope_undo(&scope))
    }

    fn can_redo(&self) -> bool {
        let scope = self.history_scope();
        if let Some((source_document_id, _)) = self.active_definition_view() {
            return self
                .documents
                .session(&source_document_id)
                .is_some_and(|session| session.has_scope_redo(&scope));
        }
        self.session
            .as_ref()
            .is_some_and(|session| session.has_scope_redo(&scope))
    }

    fn definition_view_history(&mut self, redo: bool) -> bool {
        let Some((source_document_id, definition_id)) = self.active_definition_view() else {
            return false;
        };
        let Some(mut parent) = self.documents.session(&source_document_id).cloned() else {
            return false;
        };
        let scope = self.history_scope();
        let change = if redo {
            parent.redo_scope(&scope)
        } else {
            parent.undo_scope(&scope)
        };
        let Ok(Some(_)) = change else {
            self.status = if redo {
                "Nothing to redo in this graph".to_owned()
            } else {
                "Nothing to undo in this graph".to_owned()
            };
            return false;
        };
        let Some(updated) = parent.document().definition(&definition_id).cloned() else {
            return false;
        };
        if self
            .documents
            .set_session(source_document_id.clone(), parent)
            .is_err()
        {
            return false;
        }
        if let Some(session) = self.session.as_mut()
            && session
                .apply_derived_commands(&[Command::SetRootDefinition { definition: updated }])
                .is_ok()
        {
            self.active_session_revision = session.revision();
            self.last_eval_key = None;
            self.publish_active_document();
        }
        if let Some(tab) = self
            .tabs
            .iter_mut()
            .find(|tab| tab.document_id == source_document_id)
        {
            tab.mesh = None;
            tab.status = "Definition history".to_owned();
        }
        self.evaluate();
        true
    }

    fn undo(&mut self) {
        if self.active_definition_view().is_some() {
            self.definition_view_history(false);
            return;
        }
        let scope = self.history_scope();
        if self.session.as_ref().is_some_and(|session| session.has_undo() && !session.has_scope_undo(&scope)) {
            self.status = "Next undo belongs to another view".to_owned();
            return;
        }
        let before = self.session.as_ref().map(|session| node_positions(session.document()));
        let changed = self.session.as_mut().is_some_and(|session| session.undo_scope(&scope).ok().flatten().is_some());
        if changed {
            self.layout_animation = None;
            let after = self.session.as_ref().map(|session| node_positions(session.document()));
            self.adopt_session_document();
            self.publish_active_document();
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
        if self.active_definition_view().is_some() {
            self.definition_view_history(true);
            return;
        }
        let scope = self.history_scope();
        if self.session.as_ref().is_some_and(|session| session.has_redo() && !session.has_scope_redo(&scope)) {
            self.status = "Next redo belongs to another view".to_owned();
            return;
        }
        let before = self.session.as_ref().map(|session| node_positions(session.document()));
        let changed = self.session.as_mut().is_some_and(|session| session.redo_scope(&scope).ok().flatten().is_some());
        if changed {
            self.layout_animation = None;
            let after = self.session.as_ref().map(|session| node_positions(session.document()));
            self.adopt_session_document();
            self.publish_active_document();
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

    /// While the face selector is active on a Shell node, show the shell's
    /// upstream body so picked face groups refer to the exact input topology.
    fn comparison_pair_for(
        document: &GraphDocument,
        selected: &str,
    ) -> Option<(String, String)> {
        let node = document.nodes.iter().find(|node| node.id == selected)?;
        // Comparison is an opt-in capability, not a universal selection effect.
        // Only nodes that own a topology-selection field and currently have a
        // non-empty selection can produce a meaningful before/after difference.
        let (field, _) = Self::selection_spec(&node.node_type)?;
        let has_selection = node
            .values
            .get(field)
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty());
        if !has_selection {
            return None;
        }
        let before = Self::single_geometry_input(document, node)?;
        Some((before, node.id.clone()))
    }

    fn begin_selection_session(&mut self, node_id: &str, field: &str) -> bool {
        let Some(session) = &self.session else { return false };
        let Some(node) = session.document().nodes.iter().find(|node| node.id == node_id) else { return false; };
        let Some((spec_field, mode)) = Self::selection_spec(&node.node_type) else { return false; };
        if spec_field != field { return false; }
        let active = self.viewport.selection_mode() == mode;
        if active {
            self.viewport.set_selection_mode(crate::viewport::SelectionMode::Object);
            self.face_pick_target = None;
            self.selection_view_target = None;
            self.selection_node = None;
            self.comparison_pair = None;
            self.viewport.set_edge_picking_enabled(false);
            self.sync_comparison_target();
            self.preserve_view_for_next_mesh = true;
            self.evaluate();
            return true;
        }
        let source = Self::single_geometry_input(session.document(), node)
            .or_else(|| {
                session.document().connections.iter()
                    .find(|link| link.to == node.id && (link.input == "shape" || link.input == "a"))
                    .map(|link| link.from.clone())
            });
        self.viewport.set_selection_mode(mode);
        self.viewport.set_edge_picking_enabled(false);
        self.selection_node = Some(node_id.to_owned());
        self.face_pick_target = source;
        self.selection_view_target = None;
        if mode == crate::viewport::SelectionMode::Edge {
            // Keep the breakpoint solid displayed. The feature's input body is
            // previewed as the transform-edit ghost and is the pickable layer.
            self.edit_target = Some(node_id.to_owned());
            if self.edit_preview_target != self.face_pick_target {
                self.edit_preview = None;
                self.edit_preview_target = None;
            }
        }
        self.comparison_pair = None;
        self.viewport.clear_edge_selection();
        self.refresh_topology_picking();
        // Keep the camera where the user left it instead of refitting to the
        // feature input only.
        self.preserve_view_for_next_mesh = true;
        self.evaluate();
        true
    }

    fn end_selection_session(&mut self) {
        self.viewport
            .set_selection_mode(crate::viewport::SelectionMode::Object);
        self.viewport.set_edge_picking_enabled(false);
        self.viewport.clear_edge_selection();
        self.face_pick_target = None;
        self.selection_view_target = None;
        self.selection_node = None;
        if self.edit_target.is_some() && self.edge_selection_on_preview() {
            self.clear_edit_target();
        }
        self.comparison_pair = None;
        self.sync_comparison_target();
        self.preserve_view_for_next_mesh = true;
        self.evaluate();
        self.egui_ctx.request_repaint();
    }

    /// Transform nodes between the edited feature and the viewed breakpoint. The
    /// engine graph stays canonical; this only re-frames comparison display data.
    /// Selected-feature frame → displayed downstream frame, for picks and
    /// highlights while a downstream selection target is active.
    fn selection_view_affine(
        &self,
        selected: &str,
    ) -> Option<nodecraft_session::placement::Affine3> {
        let target = self.selection_view_target.as_deref()?;
        let document = self.session.as_ref()?.document();
        nodecraft_session::placement::transform_chain(document, selected, target)
    }

    /// Frame for comparison meshes: the selected feature's result placed in
    /// the downstream view. Uses the one affine path helper, not a second
    /// graph walk.
    fn comparison_view_affine(
        &self,
        selected: &str,
    ) -> Option<nodecraft_session::placement::Affine3> {
        let document = self.session.as_ref()?.document();
        if let Some(target) = self.selection_view_target.as_deref() {
            return nodecraft_session::placement::transform_chain(document, selected, target);
        }
        let target = engine_bridge::target_for(document, self.viewer_breakpoint.as_deref())?;
        if target == selected {
            return Some(nodecraft_session::placement::Affine3::identity());
        }
        nodecraft_session::placement::transform_chain(document, selected, &target)
    }

    fn reframe_comparison(&self, selected: &str, mesh: &Mesh) -> Mesh {
        let Some(frame) = self.comparison_view_affine(selected) else {
            return mesh.clone();
        };
        let mut transformed = mesh.clone();
        for vertex in &mut transformed.vertices {
            let point = frame.apply_point([
                f64::from(vertex[0]),
                f64::from(vertex[1]),
                f64::from(vertex[2]),
            ]);
            *vertex = [point[0] as f32, point[1] as f32, point[2] as f32];
        }
        for normal in &mut transformed.normals {
            let vector = frame.apply_vector([
                f64::from(normal[0]),
                f64::from(normal[1]),
                f64::from(normal[2]),
            ]);
            let length = (vector[0] * vector[0]
                + vector[1] * vector[1]
                + vector[2] * vector[2])
                .sqrt();
            if length > 0.0 {
                *normal = [
                    (vector[0] / length) as f32,
                    (vector[1] / length) as f32,
                    (vector[2] / length) as f32,
                ];
            }
        }
        transformed
    }

    fn active_selection_field(&self) -> Option<&'static str> {
        let mode = self.viewport.selection_mode();
        if mode == crate::viewport::SelectionMode::Object || self.selected_nodes.len() != 1 {
            return None;
        }
        let selected = self.selected_nodes.iter().next()?;
        let session = self.session.as_ref()?;
        let node = session.document().nodes.iter().find(|node| &node.id == selected)?;
        Self::selection_spec(&node.node_type)
            .filter(|(_, required)| *required == mode)
            .map(|(field, _)| field)
    }

    fn selection_spec(node_type: &str) -> Option<(&'static str, crate::viewport::SelectionMode)> {
        let spec = nodecraft_node_contract::selection_spec(node_type)?;
        let mode = match spec.kind {
            nodecraft_node_contract::SelectionKind::Edge => crate::viewport::SelectionMode::Edge,
            nodecraft_node_contract::SelectionKind::Face => crate::viewport::SelectionMode::Face,
        };
        Some((spec.field, mode))
    }

    /// The body the viewport is showing while a topology session is active.
    fn topology_target(&self) -> Option<String> {
        self.selection_view_target
            .clone()
            .or_else(|| self.face_pick_target.clone())
    }

    /// True once the edit preview has delivered the feature's input body. Edge
    /// picks then come from the ghost branch while the breakpoint stays the
    /// displayed solid.
    fn edge_selection_on_preview(&self) -> bool {
        self.viewport.selection_mode() == crate::viewport::SelectionMode::Edge
            && self
                .selected_nodes
                .iter()
                .next()
                .is_some_and(|selected| self.edit_target.as_deref() == Some(selected.as_str()))
            && self.face_pick_target.is_some()
            && self.face_pick_target == self.edit_preview_target
    }

    fn refresh_topology_picking(&mut self) {
        let mode = self.viewport.selection_mode();
        let matching_target = mode != crate::viewport::SelectionMode::Object
            && self.eval_target.is_some()
            && (self.eval_target == self.topology_target() || self.edge_selection_on_preview());
        self.viewport.set_edge_picking_enabled(matching_target);
    }

    fn sync_face_pick_target(&mut self) {
        let mode = self.viewport.selection_mode();
        if mode != crate::viewport::SelectionMode::Object {
            let selected = self.selected_nodes.iter().next().cloned();
            if selected.as_deref() != self.selection_node.as_deref() {
                // A topology session belongs to one node. Clearing or changing
                // the canvas selection ends it; otherwise the next selected
                // node reuses a target from the previous session and the
                // viewport can collapse a Part back to one upstream body.
                self.end_selection_session();
                return;
            }
        }
        let desired = if self.selected_nodes.len() == 1 {
            let selected = self.selected_nodes.iter().next().cloned();
            self.session.as_ref().and_then(|session| {
                let document = session.document();
                let node = document
                    .nodes
                    .iter()
                    .find(|node| Some(&node.id) == selected.as_ref())?;
                let pickable = Self::selection_spec(&node.node_type)
                    .is_some_and(|(_, required)| required == mode);
                pickable.then(|| {
                    document
                        .connections
                        .iter()
                        .find(|link| link.to == node.id && (link.input == "shape" || link.input == "a"))
                        .map(|link| link.from.clone())
                })?
            })
        } else {
            None
        };
        if desired != self.face_pick_target {
            if desired.is_none() {
                self.selection_view_target = None;
            }
            self.face_pick_target = desired;
            self.viewport.clear_edge_selection();
            self.viewport.set_edge_picking_enabled(false);
            self.evaluate();
        }
        self.refresh_topology_picking();
    }

    fn single_geometry_input(document: &GraphDocument, node: &GraphNode) -> Option<String> {
        let geometry_inputs = ["shape", "a", "b", "profile", "bottom", "top", "sections"];
        let mut links = document.connections.iter().filter(|link| {
            link.to == node.id && geometry_inputs.contains(&link.input.as_str())
        });
        let link = links.next()?;
        links.next().is_none().then(|| link.from.clone())
    }

    fn sync_comparison_target(&mut self) {
        let mut changed = false;
        let desired = if self.selected_nodes.len() == 1 {
            let selected = self.selected_nodes.iter().next().cloned();
            self.session.as_ref().and_then(|session| {
                comparison_for_selection(
                    session.document(),
                    selected.as_deref(),
                    self.feature_overlay,
                )
            })
        } else {
            None
        };
        if desired != self.comparison_pair {
            self.comparison_pair = desired;
            changed = true;
        }
        let preview_ghost = if self.selected_nodes.len() == 1
            && self.face_pick_target.is_none()
            && self.feature_overlay
        {
            self.selected_nodes.iter().next().and_then(|selected| {
                self.session
                    .as_ref()
                    .and_then(|session| feature_ghost_input(session.document(), selected))
            })
        } else {
            None
        };
        if preview_ghost != self.preview_ghost {
            self.preview_ghost = preview_ghost;
            changed = true;
        }
        let validation_selection = self.session.as_ref().map(|session| {
            let document = session.document();
            let closure = engine_bridge::target_for(
                document,
                self.viewer_breakpoint.as_deref(),
            )
            .map(|target| target_closure(document, &target))
            .unwrap_or_default();
            let mut selected: Vec<String> = self
                .selected_nodes
                .iter()
                .filter(|id| !closure.contains(*id))
                .cloned()
                .collect();
            selected.sort();
            selected
        });
        if validation_selection != self.validation_selection {
            self.validation_selection = validation_selection;
            changed = true;
        }
        if changed {
            self.viewport
                .set_mesh_preserving_camera(self.mesh.as_ref());
            self.evaluate();
        }
    }

    fn toggle_reference_picks(
        &mut self,
        node: &nodecraft_session::GraphNode,
        field: &'static str,
        picks: Vec<String>,
        tag: &str,
    ) -> bool {
        let mut references: Vec<String> = node
            .values
            .get(field)
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        for pick in picks {
            if let Some(index) = references.iter().position(|reference| reference == &pick) {
                references.remove(index);
            } else {
                references.push(pick);
            }
        }
        let changed = self.apply_commands(
            &[Command::SetValue {
                node: node.id.clone(),
                field: field.to_owned(),
                value: serde_json::Value::Array(
                    references
                        .into_iter()
                        .map(|id| serde_json::json!({ "id": id }))
                        .collect(),
                ),
            }],
            Some(tag),
        );
        if changed {
            // Keep the topology selection session active so a user can click
            // several faces without re-entering edit mode after each one.
            self.evaluate();
        }
        changed
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

    /// Leave node editing and restore the authoritative breakpoint display.
    fn clear_edit_target(&mut self) {
        if self.edit_target.take().is_some() {
            self.edit_preview = None;
            self.edit_preview_target = None;
            self.viewport.clear_edit_ghost();
            self.viewport.set_mesh_preserving_camera(self.mesh.as_ref());
        }
    }

    fn reset_inspector_node(&mut self, node: &GraphNode) {
        let mut changed = false;
        for change in inspector::reset_values(node) {
            let command = Command::SetValue {
                node: node.id.clone(),
                field: change.field,
                value: change.value,
            };
            changed |= self.apply_commands(&[command], Some(&format!("reset:{}", node.id)));
        }
        if !node.expressions.is_empty() {
            let command = Command::SetExpressions {
                node: node.id.clone(),
                values: Default::default(),
            };
            changed |= self.apply_commands(&[command], Some(&format!("reset-expressions:{}", node.id)));
        }
        if changed {
            self.evaluate();
        }
    }

    /// Open the inspector without closing it. Double-clicking a node enters its
    /// editor, so the node panel must be visible even if the user collapsed it.
    fn open_inspector(&mut self) {
        if self.inspector_open {
            return;
        }
        self.inspector_open = true;
        self.inspector_reflow = true;
        self.inspector_animation = Some(InspectorAnimation {
            started: Instant::now(),
            from: self.inspector_reveal,
            to: 1.0,
        });
        self.egui_ctx.request_repaint();
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

    fn add_dropped_step(&mut self, path: PathBuf, pointer: Option<egui::Pos2>) {
        if self.session.is_none() {
            self.new_blank();
        }
        let Some(session) = &self.session else { return };
        let graph = pointer.and_then(|point| self.node_editor.graph_pos(point));
        let mut used: HashSet<String> = session.document().nodes.iter().map(|node| node.id.clone()).collect();
        let id = unique_node_id("step", &mut used);
        let mut node = GraphNode { expressions: Default::default(),
            id: id.clone(),
            node_type: "stepimporter".to_owned(),
            values: serde_json::json!({ "path": path.display().to_string(), "heal": true }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        if let Some(graph) = graph {
            node.x = f64::from(graph.x) - f64::from(node_width(&node)) * 0.5;
            node.y = f64::from(graph.y) - f64::from(NODE_HEIGHT) * 0.5;
        } else {
            let (x, y) = Self::free_node_placement(session.document(), &node);
            node.x = x;
            node.y = y;
        }
        let added_rect = egui::Rect::from_min_size(
            egui::pos2(node.x as f32, node.y as f32),
            egui::vec2(node_width(&node), NODE_HEIGHT),
        );
        if self.apply_commands(&[Command::AddNode { node }], Some("drop-step-import")) {
            let _ = added_rect;
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

    #[cfg(target_os = "macos")]
    fn poll_open_documents(&mut self) {
        for path in nodecraft_macos::take_pending() {
            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("nodecraft"))
            {
                self.open(path);
            }
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
                    document.nodes.push(GraphNode { expressions: Default::default(),
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
                control_server::Command::SelectNode(id) => {
                    self.selected_nodes.clear();
                    self.selected_nodes.insert(id);
                    self.selected_connections.clear();
                    self.sync_comparison_target();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::EditNode(id) => {
                    self.activate_node(&id);
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::SetValue { node, field, value } => {
                    let changed = self.apply_commands(
                        &[nodecraft_session::Command::SetValue {
                            node: node.clone(),
                            field: field.clone(),
                            value,
                        }],
                        Some("control:set"),
                    );
                    if changed {
                        self.evaluate();
                    }
                    serde_json::json!({ "ok": true, "changed": changed }).to_string()
                }
                control_server::Command::BeginSelection { node, field } => {
                    if self.begin_selection_session(&node, &field) {
                        self.egui_ctx.request_repaint();
                        "{\"ok\":true}".to_owned()
                    } else {
                        "{\"ok\":false}".to_owned()
                    }
                }
                control_server::Command::ViewNode(id) => {
                    self.viewer_breakpoint = Some(id);
                    self.evaluate();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::Home => {
                    self.viewport.home();
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::Orbit { dx, dy } => {
                    self.viewport.camera.orbit(dx, dy);
                    self.viewport.mark_dirty();
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::Zoom(factor) => {
                    self.viewport.camera.zoom_at(factor.max(0.01), None);
                    self.viewport.mark_dirty();
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::Translucent(enabled) => {
                    let mut view = self.viewport.view();
                    view.translucent = enabled;
                    self.viewport.set_view(view);
                    self.viewport.mark_dirty();
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::Triangles(enabled) => {
                    let mut view = self.viewport.view();
                    view.triangles = enabled;
                    self.viewport.set_view(view);
                    self.viewport.mark_dirty();
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::ActivateTab(index) => {
                    if index < self.tabs.len() {
                        self.activate_tab(index);
                        "{\"ok\":true}".to_owned()
                    } else {
                        "{\"ok\":false,\"error\":\"tab index out of range\"}".to_owned()
                    }
                }
                control_server::Command::Capture(path) => {
                    self.screenshot = Some((PathBuf::from(path), 0));
                    self.egui_ctx.request_repaint();
                    "{\"ok\":true}".to_owned()
                }
                control_server::Command::State => {
                    if let Some(session) = &self.session {
                        let display_plan_count = self.viewport.plans.len();
                        let unique_plan_count = self.viewport.plans.iter()
                            .map(|plan| std::sync::Arc::as_ptr(&plan.plan) as usize)
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        let body_keys = self.last_exact.as_ref().map(|(_, exact)| {
                            let keys: std::collections::HashSet<_> = exact.exact_parts.values()
                                .flat_map(|part| part.bodies.iter().map(|body| body.key))
                                .collect();
                            let first: Vec<_> = exact.exact_parts.values()
                                .flat_map(|part| part.bodies.iter().map(|body| (body.key.source, body.key.body)))
                                .take(5).collect();
                            (keys.len(), first)
                        });
                        serde_json::json!({
                        "nodes": session.document().nodes,
                        "connections": session.document().connections,
                        "viewed": self.viewer_breakpoint,
                        "selection": self.selected_nodes,
                        "evalTarget": self.eval_target,
                        "pickTarget": self.face_pick_target,
                        "selectionMode": format!("{:?}", self.viewport.selection_mode()),
                        "edgePicking": self.viewport.edge_picking_enabled(),
                        "visibleEdges": self.viewport.edge_ids().len(),
                        "displayPlans": display_plan_count,
                        "planDebug": self.viewport.plan_debug(),
                        "displayTiming": self.last_display_us,
                        "deferredDisplay": self.deferred_display.is_some(),
                        "viewPlansHash": self.viewport.view_plans_hash(),
                        "uniquePlans": unique_plan_count,
                        "bodyKeys": body_keys,
                        "evalPending": self.eval_pending,
                        "qualityDisplay": self.quality_display.as_ref().map(|(revision, _, _, _, _)| *revision),
                        "displaySnapshot": self.display_snapshot.as_ref().map(|snapshot| snapshot.id),
                        "pendingDisplay": self.pending_display.as_ref().map(|(id, _, _, _)| *id),
                        "displayPlans": self.viewport.plans.len(),
                        "nodeActivity": self.activity_active.len(),
                        "previewDetail": self.preview_detail,
                        "editTarget": self.edit_target,
                        "editPreview": self.edit_preview.is_some(),
                        "editPreviewTarget": self.edit_preview_target,
                        "gizmoDragging": self.gizmo_dragging,
                        "gizmoEdit": self.gizmo_edited,
                        "gizmoDrag": self.gizmo_drag.as_ref().map(|drag| drag.node_id.clone()),
                        "beginSelectionActive": self.active_selection_field(),
                        "comparisonActive": self.comparison_pair.is_some(),
                        "differenceFaces": self.difference_stats,
                        "differenceError": self.difference_error,
                        "differenceApproximate": self.difference_approximate,
                        "errors": self.error,
                        "nodeErrors": self.node_errors,
                        "tessellationDiagnostics": self.tessellation_diagnostics.iter().take(64).collect::<Vec<_>>(),
                        "tessellationDiagnosticCount": self.tessellation_diagnostics.len(),
                        "status": self.status,
                        "meshVertices": self.mesh.as_ref().map_or(0, |mesh| mesh.vertices.len()),
                        "meshFaces": self.mesh.as_ref().map_or(0, |mesh| mesh.faces.len()),
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
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let started = Instant::now();
        let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.logic_inner(ctx, frame);
        }));
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            let elapsed = started.elapsed();
            if elapsed.as_millis() >= 50 {
                eprintln!("[ui] logic {}ms", elapsed.as_millis());
            }
        }
        if let Err(payload) = guarded {
            crate::crash_report::handle_payload(payload);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let started = Instant::now();
        let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.ui_inner(ui, frame);
        }));
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            let elapsed = started.elapsed();
            if elapsed.as_millis() >= 50 {
                eprintln!("[ui] ui {}ms", elapsed.as_millis());
            }
        }
        if let Err(payload) = guarded {
            crate::crash_report::handle_payload(payload);
        }
    }
}

impl NodeCraftApp {
    fn logic_inner(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let tess_ready = self.viewport.plan_ready();
        let tess_was_ready = nodecraft_display::service::gpu_tess_ready();
        nodecraft_display::service::set_gpu_tess_ready(tess_ready);
        if tess_ready && !tess_was_ready {
            // The pending plan swap was waiting on the tessellation pipeline.
            ctx.request_repaint();
        }
        if self.crash_test {
            self.crash_test = false;
            panic!("NodeCraft crash-handler test");
        }
        if self
            .tool_search_clear_due
            .is_some_and(|due| Instant::now() >= due)
        {
            self.tool_search_query.clear();
            self.tool_search_result_index = None;
            self.tool_search_clear_due = None;
            ctx.request_repaint();
        }
        self.poll_dialog(ctx);
        self.node_editor.set_cable_routing(self.cable_routing);
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
                self.save_document(ctx, true);
            } else if save {
                self.save_document(ctx, false);
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
                self.feature_overlay,
                self.engine_threads,
                self.tessellation_threads,
                self.workload_profiler,
                self.kernel_fixes,
                self.cable_routing,
                self.tool_dock_offset_y,
                &self.tool_dock_shortcut,
                &self.tool_search_shortcut,
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
        if self.deferred_display.is_some() {
            let expired = self
                .deferred_display
                .as_ref()
                .is_some_and(|(_, _, _, at)| at.elapsed() >= std::time::Duration::from_millis(250));
            if self.viewport.plan_pending() && !expired {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            } else if let Some((frame, target, feature_ghost, _)) = self.deferred_display.take() {
                self.applying_deferred_display = true;
                self.apply_display_frame(&frame, &target, feature_ghost, false);
                self.applying_deferred_display = false;
            }
        }
        if self.eval_pending.is_none()
            // A drag previews the placement on the GPU; evaluating mid-drag would
            // replace that geometry and double-apply the transform.
            && self.gizmo_drag.is_none()
            && self
                .auto_eval_due
                .is_some_and(|due| due.elapsed() >= std::time::Duration::from_millis(32))
        {
            self.evaluate();
        }
        if self.eval_pending.is_none()
            && self.quality_display.is_none()
            // The display worker has a single latest-frame slot. A quality
            // request submitted while an exact frame is in flight overwrites
            // that frame before the app can poll it.
            && self.pending_display.is_none()
            && self
                .display_snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.id == self.eval_next)
            && self.gizmo_drag.is_none()
            && self.session.is_some()
        {
            let target = self.viewport.quality_detail();
            if target != self.preview_detail {
                self.preview_detail = target;
                // A quality tier is display work, not graph work. Reuse the
                // exact snapshot and submit straight to the display worker; do
                // not enter evaluate(), which would reset node activity and run
                // the legacy planner.
                if let Some(snapshot) = self.display_snapshot.clone() {
                    self.display_next = self.display_next.wrapping_add(1);
                    // Revisions share one display worker with evaluation
                    // requests; camera-only frames use a distinct range.
                    let revision = (1u64 << 63) | self.display_next;
                    self.display_worker.submit(nodecraft_display::service::DisplayRequest {
                        revision,
                        snapshot_id: snapshot.id,
                        quality: self.preview_detail.map_or(
                            nodecraft_display::DisplayQuality::Full(0.35),
                            nodecraft_display::DisplayQuality::Preview,
                        ),
                        feature_ghost: snapshot.feature_ghost,
                        snapshot: snapshot.snapshot,
                    });
                    self.quality_display = Some((
                        revision,
                        snapshot.id,
                        snapshot.target,
                        snapshot.feature_ghost,
                        Instant::now(),
                    ));
                    self.egui_ctx.request_repaint();
                }
            }
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

    fn ui_inner(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.render_model {
            egui::CentralPanel::no_frame().show(ui, |ui| {
                let rect = ui.max_rect();
                self.viewport
                    .show_with_backdrop_blockers(ui, rect, &[], rect);
            });
            return;
        }
        crate::window_chrome::prepare_main_window();
        self.accept_dropped_files(ui.ctx());
        self.poll_control();
        #[cfg(target_os = "macos")]
        self.poll_open_documents();
        // The native macOS titlebar must not start a window move when the
        // press begins on the document tab strip. The tab strip rect comes
        // from the previous frame, which is enough because the pointer must
        // already be over it before pressing.
        let pointer = ui.ctx().pointer_latest_pos();
        let over_tabs = pointer.is_some_and(|pointer| {
            self.tab_strip.is_some_and(|strip| strip.contains(pointer))
        });
        crate::window_chrome::set_main_window_draggable(!over_tabs && self.tab_drag.is_none());
        if let Some(direction) = titlebar::resize_direction(ui.ctx()) {
            ui.ctx().set_cursor_icon(titlebar::resize_cursor(direction));
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
        if !typing && ui.ctx().input(|input| input.key_pressed(egui::Key::Escape)) {
            self.clear_edit_target();
        }
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
        let duplicate_pressed = !typing
            && !self.selected_nodes.is_empty()
            && ui.input(|i| {
                i.modifiers.command
                    && !i.modifiers.shift
                    && !i.modifiers.alt
                    && i.key_pressed(egui::Key::D)
            });
        if duplicate_pressed {
            self.duplicate_selected_nodes();
        }
        if !typing
            && self.editing_sketch.is_none()
            && (!self.selected_nodes.is_empty() || !self.selected_connections.is_empty())
            && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            let removed_nodes = self.selected_nodes.clone();
            let commands = delete_selection_commands(
                self.session.as_ref().map(Session::document),
                &self.selected_nodes,
                &self.selected_connections,
            );
            let (commands, removed_definitions) =
                self.add_definition_removals(commands, &removed_nodes);
            let node_count = self.selected_nodes.len();
            let cable_count = self.selected_connections.len();
            self.selected_nodes.clear();
            self.selected_connections.clear();
            self.viewer_breakpoint = None;
            if self.apply_commands(&commands, Some("delete-selection")) {
                self.close_tabs_for_removed_definitions(&removed_definitions);
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
        let mut requested_move: Option<(usize, usize)> = None;
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
                    // native window drag. The tab strip is excluded from the
                    // drag rects entirely so a tab press can never reach this
                    // layer, regardless of egui's interaction resolution.
                    let bar_rect = ui.max_rect();
                    let reorderable_tabs = self.tabs.len() > 1;
                    let (bar_drag, bar_drag_right) = match self
                        .tab_strip
                        .filter(|strip| reorderable_tabs && strip.intersects(bar_rect))
                    {
                        Some(strip) => {
                            let left = egui::Rect::from_min_max(
                                bar_rect.min,
                                egui::pos2(strip.min.x, bar_rect.max.y),
                            );
                            let right = egui::Rect::from_min_max(
                                egui::pos2(strip.max.x, bar_rect.min.y),
                                bar_rect.max,
                            );
                            (
                                ui.interact(left, egui::Id::new("app-bar-drag-left"), egui::Sense::drag()),
                                Some(ui.interact(right, egui::Id::new("app-bar-drag-right"), egui::Sense::drag())),
                            )
                        }
                        None => (
                            ui.interact(bar_rect, egui::Id::new("app-bar-drag"), egui::Sense::drag()),
                            None,
                        ),
                    };
                    // The custom title bar owns dragging on macOS. Other
                    // platforms keep their native title bar, so empty space in
                    // this toolbar must not start a second window drag.
                    let bar_drag_started = cfg!(target_os = "macos")
                        && (bar_drag.drag_started()
                            || bar_drag_right.is_some_and(|response| response.drag_started()))
                        && titlebar::resize_direction(ui.ctx()).is_none();

                    ui.add_space(crate::window_chrome::main_bar_controls_width());
                    ui.add_space(4.0);
                    ui.strong("NodeCraft");
                    ui.add_space(8.0);
                    let file_button = ui.add(
                        egui::Button::new(egui::RichText::new("File   ").size(13.0))
                            .frame(false),
                    );
                    egui::Popup::menu(&file_button)
                        .kind(egui::PopupKind::Tooltip)
                        .style(crate::theme::menu_style)
                        .show(|ui| {
                        ui.set_min_width(230.0);
                        if ui.button("New").clicked() {
                            ui.close();
                            self.new_blank();
                        }
                        if ui.button("Open…").clicked() {
                            ui.close();
                            self.start_dialog(
                                ui.ctx(),
                                DialogAction::OpenDocument,
                                FileDialogKind::OpenDocument,
                            );
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
                            self.save_document(ui.ctx(), false);
                        }
                        if ui.add_enabled(has_document, egui::Button::new("Save As…").shortcut_text("⇧⌘S")).clicked() {
                            ui.close();
                            self.save_document(ui.ctx(), true);
                        }
                        if ui.add_enabled(has_document, egui::Button::new("Export STEP…")).clicked() {
                            ui.close();
                            self.export_document();
                        }
                        let view_mode = self
                            .tabs
                            .get(self.active_tab)
                            .map_or(TabViewMode::Split, |tab| tab.view_mode);
                        for (mode, label) in [
                            (TabViewMode::Split, "Split view"),
                            (TabViewMode::NodesOnly, "Nodes only view"),
                            (TabViewMode::ViewportOnly, "3D only view"),
                        ] {
                            if ui
                                .add_enabled(
                                    has_document,
                                    egui::Button::new(if view_mode == mode {
                                        format!("✓ {label}")
                                    } else {
                                        label.to_owned()
                                    }),
                                )
                                .clicked()
                            {
                                if let Some(tab) = self.tabs.get_mut(self.active_tab) {
                                    tab.view_mode = mode;
                                }
                                ui.close();
                            }
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
                        if ui
                            .add_enabled(has_document, egui::Button::new("Duplicate Tab"))
                            .clicked()
                        {
                            ui.close();
                            self.duplicate_active_tab();
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
                    let can_undo = self.can_undo();
                    let can_redo = self.can_redo();
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

                    // Document tabs are laid out manually so dragging can lift a
                    // tab and animate the others into their new slots.
                    let (tab_click, tab_close, tab_move) = self.document_tabs(ui, &theme);
                    if tab_click.is_some() {
                        requested_tab = tab_click;
                    }
                    if tab_close.is_some() {
                        requested_close = tab_close;
                    }
                    if tab_move.is_some() {
                        requested_move = tab_move;
                    }
                    // Tabs own drags inside their strip; only empty title-bar
                    // space starts a native window drag. Use the press origin:
                    // egui reports drag_started after the pointer has already
                    // moved, which may be outside the strip by then.
                    let press = ui.ctx().input(|input| input.pointer.press_origin());
                    let started_on_tabs = press.is_some_and(|press| {
                        self.tab_strip.is_some_and(|tab_strip| tab_strip.contains(press))
                    });
                    if bar_drag_started && !started_on_tabs && self.tab_drag.is_none() {
                        titlebar::start_drag(ui);
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
                        let (runner_rect, _) = ui.allocate_exact_size(
                            egui::Vec2::splat(28.0),
                            egui::Sense::hover(),
                        );
                        let runner = if self.eval_pending.is_some() {
                            let response = ui.interact(
                                runner_rect,
                                ui.id().with("reevaluate-progress"),
                                egui::Sense::click(),
                            );
                            let painter = ui.painter();
                            let centre = runner_rect.center();
                            let radius = 7.0;
                            painter.circle_stroke(
                                centre,
                                radius,
                                egui::Stroke::new(
                                    1.4,
                                    theme.with_alpha(theme.icon, 90),
                                ),
                            );
                            let colour = theme.color(theme.node_activity);
                            let fraction = self
                                .eval_progress
                                .as_ref()
                                .map(progress_fraction)
                                .unwrap_or(0.0);
                            let start = -std::f32::consts::FRAC_PI_2;
                            let sweep = fraction * std::f32::consts::TAU;
                            if sweep > 0.0 {
                                let steps = ((sweep.abs() / std::f32::consts::TAU) * 96.0)
                                    .ceil()
                                    .max(2.0) as usize;
                                let points: Vec<egui::Pos2> = (0..=steps)
                                    .map(|index| {
                                        let angle = start + sweep * index as f32 / steps as f32;
                                        centre + egui::vec2(angle.cos(), angle.sin()) * radius
                                    })
                                    .collect();
                                painter.add(egui::Shape::line(
                                    points,
                                    egui::Stroke::new(1.8, colour),
                                ));
                            }
                            response
                        } else {
                            let tint = crate::theme::Theme::hex(if ui.rect_contains_pointer(runner_rect) {
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
                            .fit_to_exact_size(egui::Vec2::splat(16.0))
                            .sense(egui::Sense::click());
                            ui.put(runner_rect, image)
                        };
                        let runner = runner
                            .on_hover_text("Re-evaluate graph without running exporters");
                        if runner.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
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

        if let Some((from, to)) = requested_move {
            self.move_tab(from, to);
        } else if let Some(index) = requested_close {
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
            let row_height = ui.spacing().interact_size.y.max(18.0);
            let spacing = ui.spacing().item_spacing.x;
            ui.horizontal(|ui| {
                let progress_reserved = if self.eval_progress.is_some() {
                    150.0 + 70.0 + spacing * 4.0
                } else {
                    0.0
                };
                let status_width = (ui.available_width() - progress_reserved).max(120.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(status_width, row_height),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.add(egui::Label::new(&self.status).truncate());
                    },
                );
                if let Some(progress) = &self.eval_progress {
                    ui.separator();
                    let fraction = progress_fraction(progress);
                    let percent = (fraction * 100.0).round() as u32;
                    // Fixed cells keep the bar anchored while the node name and
                    // completed count change every frame.
                    ui.add_sized(
                        [150.0, row_height],
                        egui::ProgressBar::new(fraction)
                            .desired_width(150.0)
                            .text(format!("{percent}%")),
                    );
                    ui.add_sized(
                        [70.0, row_height],
                        egui::Label::new(format!("{}/{}", progress.done(), progress.total)),
                    );
                }
                if let Some(error) = &self.error {
                    ui.separator();
                    let error_color = ui.visuals().error_fg_color;
                    let response = ui.add(
                        egui::Label::new(
                            egui::RichText::new(error).color(error_color),
                        )
                        .selectable(true),
                    );
                    response.context_menu(|ui| {
                        if ui.button("Copy error").clicked() {
                            ui.ctx().copy_text(error.clone());
                            ui.close();
                        }
                    });
                    if ui
                        .small_button("Copy")
                        .on_hover_text("Copy error message")
                        .clicked()
                    {
                        ui.ctx().copy_text(error.clone());
                    }
                }
            });
        });

        let active_document_id = self
            .tabs
            .get(self.active_tab)
            .map(|tab| tab.document_id.clone());
        let selected = if self.selected_nodes.len() == 1 {
            self.session.as_ref().and_then(|session| {
                let id = self.selected_nodes.iter().next()?.as_str();
                session.document().nodes.iter().find(|node| node.id == id).cloned()
            })
        } else {
            None
        };
        let no_selection =
            self.selected_nodes.is_empty() && self.selected_connections.is_empty();
        let fallback_owner = if selected.is_none() && no_selection {
            self.tabs
                .get(self.active_tab)
                .and_then(|tab| self.group_owner_node(tab.id))
        } else {
            None
        };
        let inspector_is_fallback = fallback_owner.is_some();
        let (selected_node, inspector_document_id) = match (selected, fallback_owner) {
            (Some(node), _) => (Some(node), active_document_id),
            (None, Some((document_id, node))) => (Some(node), Some(document_id)),
            (None, None) => (None, active_document_id),
        };
        if let Some(editing) = &self.editing_sketch
            && selected_node.as_ref().is_none_or(|node| node.id != *editing)
        {
            self.editing_sketch = None;
        }
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
        let sketch_editing = sketch_node
            .as_ref()
            .is_some_and(|node| self.editing_sketch.as_deref() == Some(node.id.as_str()));
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
        let view_mode = self
            .tabs
            .get(self.active_tab)
            .map_or(TabViewMode::Split, |tab| tab.view_mode);
        let node_only = view_mode == TabViewMode::NodesOnly;
        let viewport_only = view_mode == TabViewMode::ViewportOnly;
        self.node_editor.set_maximized(node_only);
        self.node_editor.set_right_inset(if node_only {
            inspector_width + inspector_margin + 12.0
        } else {
            0.0
        });
        self.node_editor.set_view_label(None);
        self.node_editor.set_header_visible(false);
        let node_panel_rect = self.node_editor.panel_rect(workspace_rect);
        let backdrop_rect = node_panel_rect.union(inspector_rect);
        let node_backdrop = self.viewport.backdrop_view_for(node_panel_rect, backdrop_rect);
        let inspector_backdrop = self
            .viewport
            .backdrop_view_for(inspector_rect, backdrop_rect);
        // The inspector now sits in the bottom-right corner, so the axis chrome
        // can reclaim the top-right corner of the window.
        self.viewport.chrome_right_inset = 0.0;

        if self.selected_nodes.len() == 1 {
            self.tool_add_source = self.selected_nodes.iter().next().cloned();
        } else if self.active_tool_menu().is_none() {
            self.tool_add_source = None;
        }
        let tool_connect_source = self.tool_add_source.clone();
        let mut editor_actions = Vec::new();
        let mut viewport_rect = workspace_rect;
        // No central-panel frame/margin: the wgpu viewport paint callback must start
        // at the available workspace edge.
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let central_rect = ui.available_rect_before_wrap();
            viewport_rect = central_rect;
            let panel_rect = self.node_editor.panel_rect(central_rect);
            let mut blockers = vec![tool_dock_rect(panel_rect, self.tool_dock_offset_y, self.tool_dock_measured_height.max(tool_dock_height()))];
            if let Some(menu) = self.tool_menu_rect {
                blockers.push(menu);
            }
            self.node_editor.set_pointer_blockers(blockers);
            if !viewport_only {
                self.viewport.show_with_backdrop_blockers(
                    ui,
                    panel_rect,
                    &[panel_rect, inspector_rect],
                    backdrop_rect,
                );
            }
            if !viewport_only && let Some(session) = &self.session {
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
                    if node_only { None } else { node_backdrop },
                );
            }
        });
        self.show_breadcrumb(ui.ctx(), node_panel_rect);
        self.sync_comparison_target();
        self.handle_gizmo_update(selected_node.as_ref());

        self.sync_face_pick_target();
        let face_picks = self.viewport.take_face_picks();
        if !face_picks.is_empty()
            && self.eval_target.is_some()
            && self.eval_target == self.topology_target()
            && let Some(node) = selected_node.as_ref()
            && let Some((field, mode)) = Self::selection_spec(&node.node_type)
            && mode == crate::viewport::SelectionMode::Face
        {
            self.toggle_reference_picks(node, field, face_picks, "face-pick");
        }
        if self.viewport.take_selection_miss()
            && self.viewport.selection_mode() != crate::viewport::SelectionMode::Object
        {
            self.end_selection_session();
        }
        let edge_picks = self.viewport.take_edge_picks();
        if !edge_picks.is_empty()
            && self.eval_target.is_some()
            && (self.eval_target == self.topology_target() || self.edge_selection_on_preview())
            && let Some(node) = selected_node.as_ref()
            && let Some((field, mode)) = Self::selection_spec(&node.node_type)
            && mode == crate::viewport::SelectionMode::Edge
        {
            let mut references: Vec<serde_json::Value> = node
                .values
                .get(field)
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            let pick_node = node.id.clone();
            let inverse = if self.selection_view_target.is_some() {
                self.selection_view_affine(&pick_node)
                    .map(nodecraft_session::placement::Affine3::inverse)
            } else {
                None
            };
            for pick in edge_picks {
                let center = inverse
                    .map(|frame| frame.apply_point(pick.center))
                    .unwrap_or(pick.center);
                let existing = references.iter().position(|reference| {
                    if inverse.is_none()
                        && reference.get("id").and_then(serde_json::Value::as_str)
                            == Some(&pick.id)
                    {
                        return true;
                    }
                    let Some(reference_center) =
                        reference.get("center").and_then(serde_json::Value::as_array)
                    else {
                        return false;
                    };
                    if reference_center.len() != 3 {
                        return false;
                    }
                    let dx = reference_center[0].as_f64().unwrap_or_default() - center[0];
                    let dy = reference_center[1].as_f64().unwrap_or_default() - center[1];
                    let dz = reference_center[2].as_f64().unwrap_or_default() - center[2];
                    (dx * dx + dy * dy + dz * dz).sqrt() <= 1.0e-4
                });
                if let Some(index) = existing {
                    references.remove(index);
                } else if inverse.is_some() {
                    // Downstream ids are not the feature's B-rep ids. Keep the
                    // mapped centre so the exact resolver can find the edge.
                    references.push(serde_json::json!({"center": center}));
                } else {
                    references.push(serde_json::json!({
                        "id": pick.id,
                        "center": center,
                    }));
                }
            }
            if self.apply_commands(
                &[Command::SetValue {
                    node: node.id.clone(),
                    field: field.to_owned(),
                    value: serde_json::Value::Array(references),
                }],
                Some("edge-pick"),
            ) {
                // Stay in the edge-selection session so existing selections remain
                // visible and can be toggled off.
                self.evaluate();
            }
        }

        let theme = crate::theme::get();
        let mut toggle_inspector = false;
        let mut reset_inspector = false;
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
                        left: INSPECTOR_INSET as i8,
                        right: INSPECTOR_INSET as i8,
                        top: top_margin,
                        bottom: bottom_margin,
                    })
                    .show(ui, |ui| {
                        ui.set_width(
                            (panel_rect.width() - 2.0 * INSPECTOR_INSET).max(100.0),
                        );
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
                                    // Reserve room for the absolutely centred reset
                                    // control placed against the full header below.
                                    ui.add_space(30.0);
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
                        // Double-clicking the header still toggles the inspector;
                        // the visible disclosure chevron has been replaced by the
                        // reset action.
                        let header_bottom = ui.cursor().min.y;
                        let header_rect = egui::Rect::from_min_max(
                            egui::pos2(panel_rect.left(), header_top),
                            egui::pos2(panel_rect.right(), header_bottom),
                        );
                        let reset_rect = egui::Rect::from_center_size(
                            // Use the frame's inner content edge, not the panel
                            // border, so the icon keeps the 16px header inset.
                            egui::pos2(ui.max_rect().right() - 13.0, header_rect.center().y),
                            egui::vec2(26.0, 26.0),
                        );
                        if !inspector_is_fallback
                            && inspector::reset_icon_button_at(ui, reset_rect).clicked()
                        {
                            reset_inspector = true;
                        }
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
                            // egui places a vertical bar from the ScrollArea's
                            // outer rect. Widen only that rect so the bar moves
                            // right; the content below keeps its normal width.
                            let content_width = ui.available_width();
                            ui.set_width(content_width + INSPECTOR_SCROLL_BAR_OFFSET);
                            let active_selection =
                                self.active_selection_field().map(str::to_owned);
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
                                    ui.set_width(content_width);
                                    let mut output = inspector::show(
                                        ui,
                                        node,
                                        &surface_sections,
                                        active_selection.as_deref(),
                                        self.editing_sketch.as_deref() == Some(node.id.as_str()),
                                        !inspector_is_fallback,
                                        !inspector_is_fallback
                                            && self.tabs.get(self.active_tab).is_some_and(|tab| {
                                                self.group_tabs.get(&tab.id).map_or(true, |meta| {
                                                    meta.view.definition_id.is_some()
                                                })
                                            }),
                                    );
                                    if sketch_editing {
                                        self.sketch_gui.show_palette(ui);
                                        ui.add_space(8.0);
                                    }
                                    inspector::note(ui, node);
                                    ui.add_space(8.0);
                                    if !inspector_is_fallback {
                                        inspector::actions(ui, node, &mut output);
                                    }
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
        if !inspector_is_fallback
            && let Some(node) = &selected_node
            && let Some(error) = self.node_error_messages.get(&node.id)
        {
            show_viewer_error(ui.ctx(), viewport_rect, node_panel_rect, inspector_rect, error);
        }
        if inspector_is_fallback {
            if let (Some(output), Some(node), Some(document_id)) = (
                inspector_output,
                &selected_node,
                inspector_document_id.as_deref(),
            ) {
                self.apply_background_inspector_output(document_id, &node.id, output);
            }
        } else if let (Some(output), Some(node)) = (inspector_output, &selected_node) {
            if output.delete {
                let mut nodes = HashSet::new();
                nodes.insert(node.id.clone());
                let commands = delete_selection_commands(
                    self.session.as_ref().map(Session::document),
                    &nodes,
                    &HashSet::new(),
                );
                let (commands, removed_definitions) =
                    self.add_definition_removals(commands, &nodes);
                self.selected_nodes.clear();
                self.selected_connections.clear();
                if self.viewer_breakpoint.as_deref() == Some(node.id.as_str()) {
                    self.viewer_breakpoint = None;
                }
                if self.apply_commands(&commands, Some("inspector-delete")) {
                    self.close_tabs_for_removed_definitions(&removed_definitions);
                    self.evaluate();
                }
            } else {
                if output.open_definition {
                    self.enter_graph(&node.id);
                }
                if let Some(name) = output.graph_name.clone() {
                    self.rename_graph_definition(&node.id, name);
                }
                if let Some((inputs, outputs)) = output.definition_ports {
                    self.resize_graph_definition_ports(&node.id, inputs, outputs);
                }
                if let Some((inputs, outputs)) = output.root_interface_ports {
                    self.resize_root_graph_interface(inputs, outputs);
                }
                if output.make_unique {
                    self.make_graph_unique(&node.id);
                }
                if output.make_local {
                    self.make_graph_local(&node.id);
                }
                if output.toggle_sketch_edit {
                    if self.editing_sketch.as_deref() == Some(node.id.as_str()) {
                        self.editing_sketch = None;
                    } else {
                        self.sketch_gui.sync(&node.id, &node.values);
                        self.editing_sketch = Some(node.id.clone());
                    }
                    self.egui_ctx.request_repaint();
                }
                if let Some(field) = output.begin_selection.clone()
                    && self.begin_selection_session(&node.id, &field)
                {
                    self.egui_ctx.request_repaint();
                }
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
                if let Some((inputs, outputs)) = output.graph_ports {
                    self.apply_graph_ports(&node.id, inputs, outputs);
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
                if !output.expressions.is_empty() {
                    let mut expressions = node.expressions.clone();
                    for (field, expression) in output.expressions {
                        match expression {
                            Some(expression) => {
                                expressions.insert(field, expression);
                            }
                            None => {
                                expressions.remove(&field);
                            }
                        }
                    }
                    let command = Command::SetExpressions {
                        node: node.id.clone(),
                        values: expressions,
                    };
                    changed |= self.apply_commands(&[command], Some(&format!("expressions:{}", node.id)));
                }
                if output.choose_linked_part {
                    self.start_dialog(
                        ui.ctx(),
                        DialogAction::SetNodePath {
                            node: node.id.clone(),
                            field: "path",
                            open_linked_part: true,
                        },
                        FileDialogKind::OpenDocument,
                    );
                }
                if output.open_linked_part {
                    self.open_linked_part_document(&node.id);
                }
                if output.choose_file {
                    self.start_dialog(
                        ui.ctx(),
                        DialogAction::SetNodePath {
                            node: node.id.clone(),
                            field: "path",
                            open_linked_part: false,
                        },
                        FileDialogKind::OpenStepFile,
                    );
                }
                if output.choose_directory {
                    self.start_dialog(
                        ui.ctx(),
                        DialogAction::SetNodePath {
                            node: node.id.clone(),
                            field: "directory",
                            open_linked_part: false,
                        },
                        FileDialogKind::ChooseDirectory,
                    );
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

        if reset_inspector
            && let Some(node) = selected_node.clone()
        {
            self.reset_inspector_node(&node);
        }

        for action in editor_actions {
            match action {
                NodeEditorAction::SelectNode { id, additive } => {
                    // Selection is independent of the viewer breakpoint. Only the
                    // eye control changes what the engine/viewport evaluates.
                    self.clear_edit_target();
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
                    self.clear_edit_target();
                }
                NodeEditorAction::SelectCable { key, additive } => {
                    self.selected_nodes.clear();
                    self.clear_edit_target();
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
                NodeEditorAction::ToggleSuppressed { id } => {
                    let suppressed = self
                        .session
                        .as_ref()
                        .and_then(|session| session.document().nodes.iter().find(|node| node.id == id))
                        .map(|node| node.suppressed)
                        .unwrap_or(false);
                    if self.apply_commands(
                        &[Command::SetSuppressed { node: id, suppressed: !suppressed }],
                        Some("suppress-node"),
                    ) {
                        self.evaluate();
                    }
                }
                NodeEditorAction::EnterGroup { id } => {
                    self.enter_group(&id);
                }
                NodeEditorAction::ActivateNode { id } => {
                    self.activate_node(&id);
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
                NodeEditorAction::ShakeSelection { nodes } => {
                    let Some(session) = &self.session else { continue };
                    let commands = disconnection_commands_for_selection(session.document(), &nodes);
                    let count = commands.len();
                    if count > 0
                        && self.apply_commands(&commands, Some("shake-disconnect"))
                    {
                        self.evaluate();
                        self.status = format!(
                            "Detached {} cable{}",
                            count,
                            if count == 1 { "" } else { "s" }
                        );
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
                    self.clear_edit_target();
                    // The floating tool dock sits over the canvas; clicking it is
                    // not a click-away from the current selection.
                    let pointer = ui.ctx().input(|input| input.pointer.interact_pos());
                    let node_panel = self.node_editor.panel_rect(workspace_rect);
                    let over_tool_dock = pointer.is_some_and(|pointer| {
                        tool_dock_rect(node_panel, self.tool_dock_offset_y, self.tool_dock_measured_height.max(tool_dock_height()))
                            .contains(pointer)
                            || self
                                .tool_menu_rect
                                .is_some_and(|menu| menu.contains(pointer))
                    });
                    let over_divider = pointer.is_some_and(|pointer| {
                        pointer.x >= node_panel.right() - 8.0
                            && pointer.x <= node_panel.right() + 8.0
                            && pointer.y >= node_panel.top() + 12.0
                            && pointer.y <= node_panel.bottom() - 12.0
                    });
                    if !over_tool_dock && !over_divider && !self.node_editor.is_resizing() {
                        self.selected_nodes.clear();
                        self.selected_connections.clear();
                    }
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
        let sketch_viewport = egui::Rect::from_min_max(
            egui::Pos2::new(node_panel_rect.right() + 12.0, workspace_rect.top() + 12.0),
            egui::Pos2::new(workspace_rect.right() - 12.0, workspace_rect.bottom() - 12.0),
        );
        if sketch_editing {
            let blockers = self.viewport.interaction_blockers().to_vec();
            if let Some(value) = self.sketch_gui.show(
                ui.ctx(),
                viewport_rect,
                sketch_viewport,
                &self.viewport.camera,
                self.viewport.lens_shift(),
                &blockers,
            ) {
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
        } else if sketch_node.is_some() {
            self.sketch_gui.show_selection_preview(
                ui.ctx(),
                viewport_rect,
                sketch_viewport,
                &self.viewport.camera,
                self.viewport.lens_shift(),
            );
        }
        self.show_tool_dock(ui.ctx(), workspace_rect, tool_connect_source);
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
        node_activity: theme.node_activity,
        node_error: theme.node_error,
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

fn configure_style(ctx: &egui::Context) {
    // Use the embedded Inter family as the leading egui proportional font on
    // every platform.
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
    visuals.handle_shape = egui::style::HandleShape::Circle;
    visuals.override_text_color = Some(text);
    visuals.widgets.noninteractive.bg_fill = panel;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, border);
    visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, muted);
    visuals.widgets.inactive.bg_fill = control;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, border_hi);
    // The slider rail uses this same radius. At the default 8 px rail height,
    // 4 px makes the rail's end caps circular instead of rounded rectangles.
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(4);
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
        // Round slider handles with a slightly smaller track/handle than egui's
        // default. Menus can override this locally, so the picker does too.
        style.spacing.interact_size.y = 16.0;
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
    use super::settings::{AppSettings, ConfigPlatform, config_dir_for};
    use serde_json::{json, Value};

    #[test]
    fn tab_drag_reorders_and_preserves_active_tab() {
        assert_eq!(reorder_tab_index(4, 0, 3, 0), Some((2, 2)));
        assert_eq!(reorder_tab_index(4, 3, 0, 1), Some((0, 2)));
        assert_eq!(reorder_tab_index(4, 1, 4, 1), Some((3, 3)));
        assert_eq!(reorder_tab_index(4, 1, 3, 0), Some((2, 0)));
        assert_eq!(reorder_tab_index(4, 2, 2, 2), None);
        assert_eq!(reorder_tab_index(4, 9, 0, 0), None);
    }

    #[test]
    fn removing_a_graph_instance_cascades_unreferenced_definitions() {
        let node = |id: &str, node_type: &str, definition: Option<&str>| GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values: definition.map_or_else(|| json!({}), |id| json!({"definition": id})),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let def = |id: &str, body: Vec<GraphNode>| nodecraft_session::GraphDefinition {
            id: id.into(),
            name: id.into(),
            kind: "graph".into(),
            interface: nodecraft_session::GraphInterface::default(),
            nodes: body,
            connections: Vec::new(),
        };
        let mut document = GraphDocument::new("cascade");
        document.nodes.push(node("g1", "graph", Some("g1")));
        document.definitions.push(def("g1", vec![node("g2", "graph", Some("g2"))]));
        document.definitions.push(def("g2", vec![node("g3", "graph", Some("g3"))]));
        document.definitions.push(def("g3", Vec::new()));
        let removed: HashSet<String> = ["g1".to_owned()].into_iter().collect();
        let cascade = cascade_removed_definitions(&document, &removed);
        assert_eq!(cascade.into_iter().collect::<Vec<_>>(), vec!["g1", "g2", "g3"]);

        let mut shared = document.clone();
        shared.definitions.push(def("g2_shared", vec![node("shared_g3", "graph", Some("g3"))]));
        let cascade = cascade_removed_definitions(&shared, &removed);
        assert!(cascade.contains("g1"));
        assert!(cascade.contains("g2"));
        assert!(!cascade.contains("g3"), "g3 still has another reference");
    }

    #[test]
    fn body_targets_map_to_the_instance_flat_path() {
        let node = |id: &str, node_type: &str| GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values: json!({}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let body = GraphDocument {
            schema_version: 1,
            id: "g1::graph".into(),
            root: nodecraft_session::GraphDefinition {
                id: "g1".into(),
                name: "G1".into(),
                kind: "graph".into(),
                interface: nodecraft_session::GraphInterface::default(),
                nodes: vec![node("in", "groupinput"), node("solid", "solid"), node("out", "groupoutput")],
                connections: Vec::new(),
            },
            definitions: Vec::new(),
        };
        let binding = EvalBinding {
            source_document_id: "root".into(),
            instance_path: vec!["g1".into(), "g2".into()],
            prefix: "g1::g2".into(),
            document: GraphDocument::new("root"),
            body: body.clone(),
        };
        assert_eq!(map_body_id_to_eval(&binding, "solid").as_deref(), Some("g1::g2::solid"));
        assert_eq!(map_body_id_to_eval(&binding, "out").as_deref(), Some("g1::g2"));
        assert_eq!(map_body_id_to_eval(&binding, "in"), None);
        assert_eq!(map_eval_id_to_body(&binding, "g1::g2::solid").as_deref(), Some("solid"));
        assert_eq!(map_eval_id_to_body(&binding, "g1::g2").as_deref(), Some("out"));

        let mut root = GraphDocument::new("root");
        root.nodes.push(node("solid", "solid"));
        root.nodes.push(node("graph", "graph"));
        root.connections.push(GraphConnection {
            from: "solid".into(),
            to: "graph".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        let input_binding = EvalBinding {
            source_document_id: "root".into(),
            instance_path: vec!["graph".into()],
            prefix: "graph".into(),
            document: root,
            body,
        };
        assert_eq!(map_body_id_to_eval(&input_binding, "in").as_deref(), Some("solid"));
        assert_eq!(map_eval_id_to_body(&input_binding, "solid").as_deref(), Some("in"));
    }

    #[test]
    fn target_closure_handles_large_chains_linearly() {
        let mut document = GraphDocument::new("chain");
        for index in 0..5_000 {
            document.nodes.push(GraphNode {
                expressions: Default::default(),
                id: format!("n{index}"),
                node_type: "boolean".into(),
                values: json!({}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            });
            if index > 0 {
                document.connections.push(GraphConnection {
                    from: format!("n{}", index - 1),
                    to: format!("n{index}"),
                    input: "shape".into(),
                    out: "shape".into(),
                });
            }
        }
        let closure = target_closure(&document, "n4999");
        assert_eq!(closure.len(), 5_000);
        assert!(closure.contains("n0"));
    }

    #[test]
    fn layout_only_commands_do_not_trigger_engine_updates() {
        assert!(commands_are_layout_only(&[Command::SetPosition {
            node: "a".into(), x: 1.0, y: 2.0
        }]));
        assert!(!commands_are_layout_only(&[
            Command::SetPosition { node: "a".into(), x: 1.0, y: 2.0 },
            Command::SetValue { node: "a".into(), field: "length".into(), value: serde_json::json!(2.0) },
        ]));
        assert!(!commands_are_layout_only(&[]));
    }

    #[test]
    fn graph_port_counts_commit_and_survive_the_session() {
        let graph = GraphNode {
            expressions: Default::default(),
            id: "g".into(),
            node_type: "group".into(),
            values: json!({
                "name": "Rounded Box",
                "nodes": [
                    {"id":"in","type":"groupinput","values":{"port":"shape"},"suppressed":false,"x":-120.0,"y":0.0},
                    {"id":"out","type":"groupoutput","values":{"port":"shape"},"suppressed":false,"x":120.0,"y":0.0}
                ],
                "connections": [{"from":"in","to":"out","in":"shape","out":"shape"}],
                "inputs": [{"port":"shape","node":"in"}],
                "outputs": [{"port":"shape","node":"out"}]
            }),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut document = GraphDocument::new("graph-ports");
        document.nodes.push(graph.clone());
        let mut session = Session::new(document);
        let commands = graph_port_commands("g", &graph, &session.document().connections, 3, 2)
            .expect("port commands");
        session.apply_commands(&commands).expect("apply");
        let node = session
            .document()
            .nodes
            .iter()
            .find(|node| node.id == "g")
            .expect("graph");
        let payload = read_group(node).expect("graph payload");
        assert_eq!(payload.inputs.len(), 3);
        assert_eq!(payload.outputs.len(), 2);
        assert_eq!(payload.name, "Rounded Box");
        assert_eq!(payload.inputs[2].port, "shape3");
        assert_eq!(payload.outputs[1].port, "shape2");
        assert_eq!(
            payload
                .nodes
                .iter()
                .find(|node| node.node_type == "groupinput")
                .unwrap()
                .values["ports"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn nested_graph_identity_skips_definitions_in_the_parent_document() {
        let ids: BTreeSet<String> = ["graph_definition", "graph_definition2"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(
            unique_graph_definition_id("graph_definition", &ids),
            "graph_definition3"
        );
        let names: BTreeSet<String> = ["Graph", "Graph 2"].into_iter().map(str::to_owned).collect();
        assert_eq!(unique_graph_definition_name(&names), "Graph 3");
    }

    #[test]
    fn new_graph_interface_starts_in_a_vertical_column() {
        let workspace = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 900.0));
        let ((in_x, in_y), (out_x, out_y)) = NodeCraftApp::new_graph_port_positions(workspace);
        assert_eq!(in_x, 0.0);
        assert_eq!(out_x, 0.0);
        assert_eq!(in_x, out_x, "no horizontal offset between the markers");
        assert!(in_y < 0.0);
        assert!(out_y > 0.0);
        assert!(out_y - in_y >= 380.0, "markers span the available height");
    }

    #[test]
    fn definition_port_commands_update_nested_instances_and_drop_removed_wires() {
        let node = |id: &str, node_type: &str, values: Value| GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let definition = nodecraft_session::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: nodecraft_session::GraphInterface {
                inputs: vec!["shape".into(), "shape2".into()],
                outputs: vec!["shape".into()],
            },
            nodes: vec![
                node("in", "groupinput", json!({"ports":["shape", "shape2"]})),
                node("out", "groupoutput", json!({"ports":["shape"]})),
            ],
            connections: vec![GraphConnection { from: "in".into(), to: "out".into(), input: "shape".into(), out: "shape".into() }],
        };
        let mut document = GraphDocument::new("definition-ports");
        document.definitions.push(definition.clone());
        document.nodes.push(node("g", "graph", json!({"definition":"feature","inputs":["shape","shape2"],"outputs":["shape"]})));
        document.nodes.push(node("src", "solid", json!({"primitiveType":"box"})));
        document.connections.push(GraphConnection { from: "src".into(), to: "g".into(), input: "shape2".into(), out: "shape".into() });
        document.definitions.push(nodecraft_session::GraphDefinition {
            id: "outer".into(),
            name: "Outer".into(),
            kind: "graph".into(),
            interface: nodecraft_session::GraphInterface::default(),
            nodes: vec![
                node("nested", "graph", json!({"definition":"feature","inputs":["shape","shape2"],"outputs":["shape"]})),
                node("inner_src", "solid", json!({"primitiveType":"box"})),
            ],
            connections: vec![GraphConnection { from: "inner_src".into(), to: "nested".into(), input: "shape2".into(), out: "shape".into() }],
        });
        let (commands, refreshed) = definition_port_commands(&document, "feature", 1, 1).expect("commands");
        assert_eq!(refreshed[0].interface.inputs, vec!["shape"]);
        let outer = refreshed.iter().find(|definition| definition.id == "outer").expect("outer refreshed");
        assert!(outer.connections.is_empty());
        let mut session = Session::new(document);
        session.apply_commands(&commands).expect("apply");
        assert!(session.document().connections.is_empty());
        let outer = session.document().definition("outer").expect("outer");
        assert!(outer.connections.is_empty());
        assert_eq!(outer.nodes.iter().find(|node| node.id == "nested").unwrap().values["inputs"], json!(["shape"]));
        session.undo().expect("undo").expect("change");
        assert_eq!(session.document().definition("feature").unwrap().interface.inputs.len(), 2);
        assert_eq!(session.document().connections.len(), 1);
    }

    #[test]
    fn root_interface_commands_resize_markers_and_drop_removed_wires() {
        let node = |id: &str, node_type: &str, values: Value| GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut document = GraphDocument::new("root-interface");
        document.root.interface = nodecraft_session::GraphInterface {
            inputs: vec!["shape".into(), "shape2".into()],
            outputs: vec!["shape".into()],
        };
        document.root.nodes = vec![
            node("in", "groupinput", json!({"ports":["shape", "shape2"]})),
            node("box", "solid", json!({"primitiveType":"box"})),
            node("out", "groupoutput", json!({"ports":["shape"]})),
        ];
        document.root.connections = vec![
            GraphConnection { from: "in".into(), to: "box".into(), input: "shape".into(), out: "shape2".into() },
            GraphConnection { from: "box".into(), to: "out".into(), input: "shape".into(), out: "shape".into() },
        ];
        let commands = root_interface_commands(&document, Some(1), None).expect("commands");
        let mut session = Session::new(document);
        session.apply_commands(&commands).expect("apply");
        assert_eq!(session.document().root.interface.inputs, vec!["shape"]);
        assert_eq!(session.document().connections.len(), 1);
        assert_eq!(session.document().connections[0].from, "box");
        session.undo().expect("undo").expect("change");
        assert_eq!(session.document().root.interface.inputs.len(), 2);
        assert_eq!(session.document().connections.len(), 2);
    }

    #[test]
    fn replaced_sessions_get_a_new_evaluation_identity() {
        let selection = Some("extrude".to_owned());
        let comparison = None;
        let first = eval_key(1, "untitled-1", 0, &selection, &comparison, &None, &None, &[], false, None);
        let replacement = eval_key(2, "untitled-1", 0, &selection, &comparison, &None, &None, &[], false, None);
        assert_ne!(first, replacement, "a replaced session must not reuse the old result");
    }


    fn legacy_document(value: Value) -> GraphDocument {
        if value.get("root").is_some() {
            return serde_json::from_value(value).expect("document");
        }
        serde_json::from_value(json!({
            "schemaVersion": 1,
            "id": "root",
            "root": {
                "id": "root",
                "name": "Root",
                "kind": "graph",
                "nodes": value["nodes"].clone(),
                "connections": value["connections"].clone(),
            },
            "definitions": [],
        }))
        .expect("document")
    }

    #[test]
    fn selected_feature_previews_without_changing_the_breakpoint() {
        let document = legacy_document(json!({
            "nodes": [
                {"id": "extrude", "type": "extrude", "values": {}, "x": 0.0, "y": 0.0},
                {"id": "fillet", "type": "fillet", "values": {"edges": [{"id": "edge:0"}]}, "x": 0.0, "y": 0.0},
                {"id": "fillet-copy1", "type": "fillet", "values": {"edges": [{"id": "edge:0"}]}, "x": 0.0, "y": 0.0}
            ],
            "connections": [
                {"from": "extrude", "to": "fillet", "in": "shape", "out": "shape"},
                {"from": "fillet", "to": "fillet-copy1", "in": "shape", "out": "shape"}
            ]
        }));

        assert_eq!(
            comparison_for_selection(&document, Some("fillet"), true),
            Some(("extrude".to_owned(), "fillet".to_owned()))
        );
        assert_eq!(
            feature_ghost_input(&document, "fillet"),
            Some("extrude".to_owned())
        );
        assert_eq!(
            feature_ghost_input(&document, "fillet-copy1"),
            Some("fillet".to_owned())
        );

        let inactive = legacy_document(json!({
            "nodes": [
                {"id": "extrude", "type": "extrude", "values": {}, "x": 0.0, "y": 0.0},
                {"id": "fillet", "type": "fillet", "values": {"edges": []}, "x": 0.0, "y": 0.0}
            ],
            "connections": [
                {"from": "extrude", "to": "fillet", "in": "shape", "out": "shape"}
            ]
        }));
        assert_eq!(feature_ghost_input(&inactive, "fillet"), None);
        assert_eq!(comparison_for_selection(&inactive, Some("fillet"), true), None);
    }

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
            nodecraft_session::GraphNode { expressions: Default::default(),
                id: "a".into(),
                node_type: "primitive".into(),
                values: json!({}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            },
            nodecraft_session::GraphNode { expressions: Default::default(),
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
            root: nodecraft_session::GraphDefinition {
                id: "root".into(),
                name: "Root".into(),
                kind: "graph".into(),
                nodes,
                connections,
                interface: Default::default(),
            },
            definitions: Vec::new(),
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
        let mut moved = document.clone();
        moved.root.nodes = document.nodes.iter().rev().cloned().collect();
        assert_eq!(ConnectionKey::from_connection(&moved.connections[0]), key);
    }
    #[test]
    fn clipboard_payload_roundtrips_node_descriptions() {
        let payload = NodeClipboard {
            nodes: history_fixture().root.nodes,
            connections: history_fixture().root.connections,
            definitions: Vec::new(),
        };
        let text = serde_json::to_string(&payload).expect("serialize");
        let decoded: NodeClipboard = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(decoded.nodes, payload.nodes);
        assert_eq!(decoded.connections, payload.connections);
        assert!(text.contains("\"nodes\""));
        assert!(text.contains("\"type\""));
    }

    #[test]
    fn graph_node_paste_carries_and_remaps_its_definition() {
        let definition = nodecraft_session::GraphDefinition {
            id: "feature".into(),
            name: "Feature".into(),
            kind: "graph".into(),
            interface: nodecraft_session::GraphInterface {
                inputs: vec!["shape".into()],
                outputs: vec!["shape".into()],
            },
            nodes: vec![
                GraphNode {
                    expressions: Default::default(),
                    id: "in".into(),
                    node_type: "groupinput".into(),
                    values: json!({"ports":["shape"]}),
                    suppressed: false,
                    x: 0.0,
                    y: 0.0,
                },
                GraphNode {
                    expressions: Default::default(),
                    id: "out".into(),
                    node_type: "groupoutput".into(),
                    values: json!({"ports":["shape"]}),
                    suppressed: false,
                    x: 0.0,
                    y: 0.0,
                },
            ],
            connections: Vec::new(),
        };
        let mut source = GraphDocument::new("source");
        source.definitions.push(definition);
        source.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "graph".into(),
            node_type: "graph".into(),
            values: json!({"definition":"feature","name":"Feature","inputs":["shape"],"outputs":["shape"]}),
            suppressed: false,
            x: 4.0,
            y: 8.0,
        });
        let selected: HashSet<&str> = ["graph".to_owned()].into_iter().map(|id| Box::leak(id.into_boxed_str()) as &str).collect();
        let payload = clipboard_payload(&source, &selected).expect("payload");
        assert_eq!(payload.definitions.len(), 1);
        let target = GraphDocument::new("target");
        let (commands, _, _) = paste_plan(&target, payload, None);
        let added_definition = commands.iter().find_map(|command| match command {
            Command::AddDefinition { definition } => Some(definition),
            _ => None,
        }).expect("definition must be pasted with the node");
        let added_node = commands.iter().find_map(|command| match command {
            Command::AddNode { node } => Some(node),
            _ => None,
        }).expect("node");
        assert_eq!(added_node.values["definition"], added_definition.id);
        assert_eq!(added_definition.name, "Feature");
    }

    #[test]
    fn hand_written_graph_clipboard_synthesizes_a_passthrough_definition() {
        let json = r#"{
            "nodes": [{"id":"graph","type":"graph","values":{"definition":"graph_definition","name":"Graph","inputs":["shape"],"outputs":["shape"]},"suppressed":false,"x":0,"y":0}],
            "connections": []
        }"#;
        let payload = parse_node_clipboard(json).expect("parse");
        let (commands, _, _) = paste_plan(&GraphDocument::new("target"), payload, None);
        assert!(commands.iter().any(|command| matches!(command, Command::AddDefinition { definition } if definition.id == "graph_definition")));
        assert!(commands.iter().any(|command| matches!(command, Command::AddNode { node } if node.values["definition"] == "graph_definition")));
        let mut session = Session::new(GraphDocument::new("target"));
        session.apply_commands(&commands).expect("paste applies");
    }

    #[test]
    fn duplicate_plan_copies_selection_with_fresh_ids_and_offset() {
        let document = history_fixture();
        let selected: HashSet<&str> = document.nodes.iter().map(|node| node.id.as_str()).collect();
        let payload = clipboard_payload(&document, &selected).expect("selection payload");
        let (commands, selected, _) = paste_plan(&document, payload, None);
        let added: Vec<&GraphNode> = commands
            .iter()
            .filter_map(|command| match command {
                Command::AddNode { node } => Some(node),
                _ => None,
            })
            .collect();
        assert_eq!(added.len(), 2);
        assert!(added.iter().any(|node| node.id == "a-copy1"));
        assert!(added.iter().any(|node| node.id == "b-copy1"));
        assert_eq!(selected.len(), 2);
        let a = added.iter().find(|node| node.id == "a-copy1").expect("a copy");
        assert_eq!(a.x, PASTE_OFFSET);
        assert_eq!(a.y, PASTE_OFFSET);
        assert!(
            commands.iter().any(|command| matches!(
                command,
                Command::Connect { from, to, .. } if from == "a-copy1" && to == "b-copy1"
            )),
            "internal connection copied"
        );
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
    fn evaluation_progress_fraction_is_bounded() {
        assert_eq!(progress_fraction(&EvalProgress { total: 10, exact_done: 0 }), 0.0);
        assert!((progress_fraction(&EvalProgress { total: 10, exact_done: 5 }) - 0.5).abs() < 1e-6);
        assert_eq!(progress_fraction(&EvalProgress { total: 10, exact_done: 10 }), 1.0);
        assert_eq!(progress_fraction(&EvalProgress { total: 10, exact_done: 20 }), 1.0);
        assert_eq!(progress_fraction(&EvalProgress { total: 0, exact_done: 0 }), 0.0);
    }

    #[test]
    fn transform_edit_preview_uses_the_transform_input() {
        let node = |id: &str, node_type: &str| GraphNode { expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values: Default::default(),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut document = GraphDocument::new("preview-input");
        document.nodes = vec![node("box", "primitive"), node("move", "translate")];
        document.connections = vec![GraphConnection {
            from: "box".into(),
            to: "move".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        assert_eq!(
            transform_edit_input(&document, "move").as_deref(),
            Some("box")
        );
        assert_eq!(transform_edit_input(&document, "box"), None);
    }

    #[test]
    fn edit_overlay_has_separate_branch_and_breakpoint_bodies() {
        let tri = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        };
        let overlay = dim_breakpoint_mesh(&tri);
        assert_eq!(overlay.faces.len(), 1);
        assert_eq!(overlay.face_groups.len(), 1);
        assert_eq!(overlay.face_colors.len(), 1);
        assert!(overlay.face_colors[0][0] < 0.32, "{:?}", overlay.face_colors);
    }

    #[test]
    fn ghost_alpha_follows_the_view_setting() {
        let mut view = crate::viewport::ViewSettings::default();
        assert!((ghost_alpha(&view) - 0.45).abs() < 1.0e-6);
        view.ghost_opacity = 0.8;
        assert!((ghost_alpha(&view) - 0.8).abs() < 1.0e-6);
        view.ghost_opacity = -1.0;
        assert_eq!(ghost_alpha(&view), 0.0);
        view.ghost_opacity = 2.0;
        assert_eq!(ghost_alpha(&view), 1.0);
    }

    #[test]
    fn gizmo_frame_uses_the_displayed_target_chain() {
        let node = |id: &str, node_type: &str, values: serde_json::Value| GraphNode { expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut document = GraphDocument::new("frame");
        document.nodes = vec![
            node("torus", "solid", serde_json::json!({"primitiveType": "torus"})),
            node("move", "translate", serde_json::json!({"rx": 90.0, "z": 6.0})),
            node("box", "solid", serde_json::json!({"primitiveType": "box"})),
            node("move2", "translate", serde_json::json!({"x": -24.0})),
            node("boolean", "boolean", serde_json::json!({"mode": "subtract"})),
            node("post", "translate", serde_json::json!({"x": 22.0})),
            node("look", "look", serde_json::json!({})),
        ];
        document.connections = vec![
            GraphConnection { from: "torus".into(), to: "move".into(), input: "shape".into(), out: "shape".into() },
            GraphConnection { from: "move".into(), to: "boolean".into(), input: "a".into(), out: "shape".into() },
            GraphConnection { from: "box".into(), to: "move2".into(), input: "shape".into(), out: "shape".into() },
            GraphConnection { from: "move2".into(), to: "boolean".into(), input: "b".into(), out: "shape".into() },
            GraphConnection { from: "boolean".into(), to: "post".into(), input: "shape".into(), out: "shape".into() },
            GraphConnection { from: "post".into(), to: "look".into(), input: "shape".into(), out: "shape".into() },
        ];
        let frame = transform_frame(&document, "move", "look").expect("frame");
        assert_eq!(frame.pivot, [22.0, 0.0, 6.0]);
        assert_eq!(frame.placement.apply_point([0.0, 0.0, 0.0]), [22.0, 0.0, 6.0]);
    }

    #[test]
    fn deep_transform_overlay_uses_the_input_mesh() {
        let node = |id: &str, node_type: &str, values: serde_json::Value| GraphNode { expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut document = GraphDocument::new("overlay-plan");
        document.nodes = vec![
            node("box", "solid", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": -24.0})),
            node("sphere", "solid", serde_json::json!({"primitiveType": "sphere"})),
            node("boolean", "boolean", serde_json::json!({"mode": "union"})),
            node("fillet", "fillet", serde_json::json!({})),
            node("look", "look", serde_json::json!({})),
        ];
        document.connections = vec![
            GraphConnection { from: "box".into(), to: "move".into(), input: "shape".into(), out: "shape".into() },
            GraphConnection { from: "move".into(), to: "boolean".into(), input: "a".into(), out: "shape".into() },
            GraphConnection { from: "sphere".into(), to: "boolean".into(), input: "b".into(), out: "shape".into() },
            GraphConnection { from: "boolean".into(), to: "fillet".into(), input: "shape".into(), out: "shape".into() },
            GraphConnection { from: "fillet".into(), to: "look".into(), input: "shape".into(), out: "shape".into() },
        ];
        let preview = Mesh { vertices: vec![[0.0; 3]], faces: vec![[0, 0, 0]], ..Mesh::default() };
        let breakpoint = preview.clone();
        let plan = transform_ghost_for(&document, "move", "look", Some(&preview), Some(&breakpoint), 0.45);
        assert!(plan.is_some(), "deep transform should get a branch overlay");
        assert!((plan.unwrap().alpha - 0.45).abs() < 1.0e-6);
        let plan = transform_ghost_for(&document, "move", "boolean", Some(&preview), Some(&breakpoint), 0.45);
        assert!(plan.is_some(), "a Boolean breakpoint should also get the branch overlay");
    }

    #[test]
    fn feature_edit_ghost_previews_its_input_without_replacing_the_breakpoint() {
        let node = |id: &str, node_type: &str, values: serde_json::Value| GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let link = |from: &str, to: &str, input: &str| GraphConnection {
            from: from.into(),
            to: to.into(),
            input: input.into(),
            out: "shape".into(),
        };
        let mut document = GraphDocument::new("feature-ghost");
        document.nodes = vec![
            node("box", "solid", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 12.0})),
            node("boolean", "boolean", serde_json::json!({"mode": "union"})),
            node("chamfer", "chamfer", serde_json::json!({})),
            node("out", "combine", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "move", "shape"),
            link("move", "boolean", "a"),
            link("boolean", "chamfer", "shape"),
            link("chamfer", "out", "shape"),
        ];
        assert_eq!(edit_preview_input(&document, "chamfer").as_deref(), Some("boolean"));
        let preview = Mesh { vertices: vec![[0.0; 3]], faces: vec![[0, 0, 0]], ..Mesh::default() };
        let plan = transform_ghost_for(
            &document,
            "chamfer",
            "out",
            Some(&preview),
            Some(&preview),
            0.45,
        )
        .expect("feature ghost");
        assert_eq!(plan.branch.vertices, preview.vertices);
    }

    #[test]
    fn double_click_activation_policy_is_one_place() {
        assert_eq!(node_activation("group"), NodeActivation::EnterGroup);
        assert_eq!(node_activation("graph"), NodeActivation::EnterGraph);
        assert_eq!(node_activation("translate"), NodeActivation::EditNode);
        assert_eq!(node_activation("boolean"), NodeActivation::EditNode);
    }

    #[test]
    fn selection_alone_does_not_activate_an_edit_target() {
        let selected = HashSet::from(["move".to_owned()]);
        assert!(!selection_is_edit_target(&selected, None));
        assert!(!selection_is_edit_target(&selected, Some("other")));
        assert!(selection_is_edit_target(&selected, Some("move")));
    }

    #[test]
    fn transform_preview_is_skipped_for_sketch_inputs() {
        let node = |id: &str, node_type: &str| GraphNode { expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values: Default::default(),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        };
        let mut sketch_graph = GraphDocument::new("sketch-transform");
        sketch_graph.nodes = vec![
            node("s0", "spline"),
            node("t0", "translate"),
            node("roof", "surfaceloft"),
        ];
        sketch_graph.connections = vec![
            GraphConnection {
                from: "s0".into(),
                to: "t0".into(),
                input: "shape".into(),
                out: "sketch".into(),
            },
            GraphConnection {
                from: "t0".into(),
                to: "roof".into(),
                input: "sections".into(),
                out: "shape".into(),
            },
        ];
        assert!(NodeCraftApp::transform_input_contains_sketch(
            &sketch_graph,
            "t0"
        ));

        let mut solid_graph = GraphDocument::new("solid-transform");
        solid_graph.nodes = vec![node("p", "primitive"), node("t", "translate")];
        solid_graph.connections = vec![GraphConnection {
            from: "p".into(),
            to: "t".into(),
            input: "shape".into(),
            out: "shape".into(),
        }];
        assert!(!NodeCraftApp::transform_input_contains_sketch(
            &solid_graph,
            "t"
        ));
    }

    #[test]
    fn app_settings_round_trip_recent_files() {
        let settings = AppSettings {
            view: crate::viewport::ViewSettings::default(),
            theme: Some("punk".to_owned()),
            activity: ActivitySettings::default(),
            feature_overlay: true,
            engine_threads: 0,
            tessellation_threads: 0,
            workload_profiler: true,
            kernel_fixes: true,
            cable_routing: "bezier-routing".to_owned(),
            tool_dock_offset_y: 12.0,
            tool_dock_shortcut: "Cmd+M".to_owned(),
            tool_search_shortcut: "Cmd+F".to_owned(),
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
        assert!(parsed.feature_overlay);
        assert_eq!(parsed.cable_routing, "bezier-routing");
        assert_eq!(parsed.tool_dock_offset_y, 12.0);
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
        let node = |id: &str, node_type: &str, values: Value, x: f64| nodecraft_session::GraphNode { expressions: Default::default(),
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
    fn shake_disconnects_only_cables_crossing_the_selection() {
        let document = history_fixture();
        let selected: HashSet<String> = ["a".to_owned()].into_iter().collect();
        let commands = disconnection_commands_for_selection(&document, &selected);
        assert!(matches!(
            commands.as_slice(),
            [Command::Disconnect { from, to, .. }] if from == "a" && to == "b"
        ));
        let both: HashSet<String> = ["a".to_owned(), "b".to_owned()].into_iter().collect();
        assert!(disconnection_commands_for_selection(&document, &both).is_empty());
    }

    #[test]
    fn graph_interface_cables_can_be_deleted() {
        let mut document = GraphDocument::new("interface");
        document.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "in".into(),
            node_type: "groupinput".into(),
            values: json!({"ports": ["shape"]}),
            suppressed: false,
            x: 0.0,
            y: 0.0,
        });
        document.nodes.push(GraphNode {
            expressions: Default::default(),
            id: "out".into(),
            node_type: "groupoutput".into(),
            values: json!({"ports": ["shape"]}),
            suppressed: false,
            x: 120.0,
            y: 0.0,
        });
        document.connections.push(GraphConnection {
            from: "in".into(),
            to: "out".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        let cables: HashSet<ConnectionKey> = [ConnectionKey {
            from: "in".into(),
            to: "out".into(),
            input: "shape".into(),
            out: "shape".into(),
        }]
        .into_iter()
        .collect();
        let commands = delete_selection_commands(Some(&document), &HashSet::new(), &cables);
        assert_eq!(
            commands,
            vec![Command::Disconnect {
                from: "in".into(),
                to: "out".into(),
                input: "shape".into(),
            }]
        );
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
        let commands = delete_selection_commands(None, &nodes, &cables);
        assert_eq!(commands, vec![Command::RemoveNode { id: "a".into() }]);
    }

    #[test]
    fn tool_insert_uses_the_primary_geometry_ports() {
        assert_eq!(NodeCraftApp::auto_connect_ports("primitive", "chamfer"), Some(("shape", "shape")));
        assert_eq!(NodeCraftApp::auto_connect_ports("primitive", "fillet"), Some(("shape", "shape")));
        assert_eq!(NodeCraftApp::auto_connect_ports("primitive", "look"), Some(("shape", "shape")));
        assert_eq!(NodeCraftApp::auto_connect_ports("primitive", "boolean"), Some(("a", "shape")));
        assert_eq!(NodeCraftApp::auto_connect_ports("sketch", "loft"), Some(("bottom", "sketch")));
        assert_eq!(NodeCraftApp::auto_connect_ports("spline", "surfaceloft"), Some(("sections", "sketch")));
        assert_eq!(NodeCraftApp::auto_connect_ports("exporter", "chamfer"), None);
        assert_eq!(NodeCraftApp::auto_connect_ports("groupoutput", "graph"), None);
        assert_eq!(NodeCraftApp::auto_connect_ports("primitive", "primitive"), None);
    }

    #[test]
    fn free_node_placement_puts_nodes_below_without_overlap() {
        let mut document = history_fixture();
        let max_y = document
            .nodes
            .iter()
            .map(|node| node.y + f64::from(node_extent(node).height))
            .fold(f64::NEG_INFINITY, f64::max);
        let template = document.nodes[0].clone();
        let first = GraphNode { expressions: Default::default(),
            id: "free-one".into(),
            ..template.clone()
        };
        let (x1, y1) = NodeCraftApp::free_node_placement(&document, &first);
        assert!(y1 >= max_y);
        document.nodes.push(GraphNode { expressions: Default::default(), x: x1, y: y1, ..first.clone() });
        let second = GraphNode { expressions: Default::default(),
            id: "free-two".into(),
            ..template
        };
        let (x2, y2) = NodeCraftApp::free_node_placement(&document, &second);
        let first_rect = node_extent(&first).padded_rect(x1, y1);
        let second_rect = node_extent(&second).padded_rect(x2, y2);
        assert!(!first_rect.intersects(second_rect));
    }

    #[test]
    fn auto_connect_placement_avoids_an_existing_node() {
        let mut document = history_fixture();
        let mut blocker = document.nodes[0].clone();
        blocker.id = "blocker".into();
        blocker.node_type = "chamfer".into();
        blocker.x = 0.0;
        blocker.y = f64::from(NODE_HEIGHT + 56.0);
        document.nodes.push(blocker.clone());
        let mut added = blocker.clone();
        added.id = "new".into();
        added.node_type = "fillet".into();
        let initial = NodeCraftApp::auto_connect_position(
            document.nodes[0].x,
            document.nodes[0].y,
            node_width(&document.nodes[0]),
            node_width(&added),
        );
        let (x, y, moved) =
            NodeCraftApp::auto_connect_placement(&document, &added, "a", initial, CableRouting::J);
        let placed = egui::Rect::from_min_size(
            egui::pos2(x as f32, y as f32),
            egui::vec2(node_width(&added), NODE_HEIGHT),
        );
        let (blocker_x, blocker_y) = moved
            .iter()
            .find(|(id, _, _)| id == "blocker")
            .map(|(_, x, y)| (*x, *y))
            .unwrap_or((blocker.x, blocker.y));
        let blocked = egui::Rect::from_min_size(
            egui::pos2(blocker_x as f32, blocker_y as f32),
            egui::vec2(node_width(&blocker), NODE_HEIGHT),
        );
        assert!(!placed.intersects(blocked));
    }

    #[test]
    fn auto_connect_placement_moves_a_blocked_branch_and_children() {
        let mut document = history_fixture();
        let y = f64::from(NODE_HEIGHT + 56.0);
        let mut blocker = document.nodes[0].clone();
        blocker.id = "blocker".into();
        blocker.node_type = "chamfer".into();
        blocker.x = 0.0;
        blocker.y = y;
        let mut child = blocker.clone();
        child.id = "child".into();
        child.node_type = "fillet".into();
        child.y = y + f64::from(NODE_HEIGHT) + 24.0;
        let blocker_width = f64::from(node_width(&blocker));
        let mut right = blocker.clone();
        right.id = "right".into();
        right.x = blocker_width + 28.0;
        right.y = y;
        let mut left = blocker.clone();
        left.id = "left".into();
        left.x = -blocker_width - 28.0;
        left.y = y;
        let mut below = blocker.clone();
        below.id = "below".into();
        below.x = 0.0;
        below.y = y + f64::from(NODE_HEIGHT) + 28.0;
        document.nodes.extend([blocker.clone(), child, right, left, below]);
        document.connections.push(GraphConnection {
            from: "blocker".into(),
            to: "child".into(),
            input: "shape".into(),
            out: "shape".into(),
        });
        let mut added = blocker.clone();
        added.id = "new".into();
        added.node_type = "fillet".into();
        let initial = NodeCraftApp::auto_connect_position(
            document.nodes[0].x,
            document.nodes[0].y,
            node_width(&document.nodes[0]),
            node_width(&added),
        );
        let (x, y, moved) =
            NodeCraftApp::auto_connect_placement(&document, &added, "a", initial, CableRouting::J);
        let moved_ids: HashSet<&str> = moved.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(moved_ids.contains("blocker"), "{moved:?}");
        assert!(moved_ids.contains("child"), "{moved:?}");
        let placed = egui::Rect::from_min_size(
            egui::pos2(x as f32, y as f32),
            egui::vec2(node_width(&added), NODE_HEIGHT),
        );
        for node in &document.nodes {
            let (nx, ny) = moved
                .iter()
                .find(|(id, _, _)| id == &node.id)
                .map(|(_, x, y)| (*x, *y))
                .unwrap_or((node.x, node.y));
            let rect = egui::Rect::from_min_size(
                egui::pos2(nx as f32, ny as f32),
                egui::vec2(node_width(node), NODE_HEIGHT),
            );
            assert!(!placed.intersects(rect), "{} overlaps new node", node.id);
        }
    }

    #[test]
    fn entering_a_topology_selection_keeps_the_viewport_camera() {
        // Same target: always preserve; different target: only when the selection
        // transition asked for it.
        assert!(mesh_preserves_camera(true, true, true, false));
        assert!(!mesh_preserves_camera(true, true, false, false));
        assert!(mesh_preserves_camera(true, true, false, true));
        assert!(mesh_preserves_camera(false, false, false, true));
    }

    #[test]
    fn selection_spec_defers_to_the_node_opting_in() {
        assert_eq!(
            NodeCraftApp::selection_spec("chamfer"),
            Some(("edges", crate::viewport::SelectionMode::Edge))
        );
        assert_eq!(
            NodeCraftApp::selection_spec("shell"),
            Some(("faces", crate::viewport::SelectionMode::Face))
        );
        assert_eq!(NodeCraftApp::selection_spec("primitive"), None);
    }

    #[test]
    fn auto_connected_nodes_align_below_their_source() {
        let source = history_fixture().nodes[0].clone();
        let mut added = source.clone();
        added.node_type = "chamfer".to_owned();
        let (x, y) = NodeCraftApp::auto_connect_position(
            source.x,
            source.y,
            node_width(&source),
            node_width(&added),
        );
        assert_eq!(y, source.y + f64::from(NODE_HEIGHT) + 56.0);
        assert_eq!(
            x,
            source.x + f64::from(node_width(&source) - node_width(&added)) * 0.5
        );
    }


    #[test]
    fn viewer_overlay_rect_excludes_panel_and_inspector() {
        let viewport = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1000.0, 700.0));
        let node_panel = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(300.0, 700.0));
        let inspector = egui::Rect::from_min_max(egui::pos2(700.0, 300.0), egui::pos2(1000.0, 700.0));
        let viewer = viewer_content_rect(viewport, node_panel, inspector);
        assert_eq!(viewer.left(), 300.0);
        assert_eq!(viewer.right(), 700.0);
        assert_eq!(viewer.top(), 0.0);
        assert_eq!(viewer.bottom(), 700.0);
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
