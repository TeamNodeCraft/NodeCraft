//! CPU viewport built on SolidCraft's camera, scene and rasterizer.
//!
//! The wgpu path from `solidcraft-viewport-gpu` will replace the rasterizer once the viewport
//! adapter lands. Keeping the camera/scene/raster on SolidCraft types means the
//! switch is local to this module.

use solidcraft_viewport::{Camera, Rgb, Scene, SceneLine, SceneMesh, StandardView};
use nodecraft_protocol::Mesh as NodeMesh;

use crate::adapter::{self, DisplayMesh};
use crate::hover::HoverMap;
use crate::viewer_gizmo::{self, Gizmo, GizmoState};
use crate::viewport_grid::{grid_lines, GridKind};
#[cfg(feature = "gpu-viewport")]
use crate::gpu_viewport::GpuViewport;
use solidcraft_geom::prelude::*;
use solidcore_record::Vec3;

/// The shared world origin used by both the XY construction grid and the
/// Home/ISO framing. Keeping one constant makes the view centre and the grid
/// origin the same point by construction.
const VIEW_ORIGIN: solidcore_record::Vec3 = solidcore_record::Vec3::ZERO;

fn lens_shift_for_visible(full: egui::Rect, visible: egui::Rect) -> [f32; 2] {
    [
        2.0 * (visible.center().x - full.center().x) / full.width().max(1.0),
        -2.0 * (visible.center().y - full.center().y) / full.height().max(1.0),
    ]
}

fn center_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ViewSettings {
    /// Draw the shaded solid with alpha blending.
    pub translucent: bool,
    /// Opacity of the transform edit ghost layer, 0 = fully transparent,
    /// 1 = fully opaque.
    pub ghost_opacity: f32,
    /// Ignore face colours authored by Look nodes for display only.
    pub ignore_look_colors: bool,
    pub edges: bool,
    pub triangles: bool,
    /// Draw one normal per display face.
    pub normals: bool,
    /// Tint each display face from a stable palette.
    pub face_colours: bool,
    /// Zebra-stripe reflection analysis for checking surface continuity.
    pub zebra: bool,
    /// PBR material roughness.
    pub roughness: f32,
    /// PBR material metallic fraction.
    pub metallic: f32,
    /// CAD-style mouse navigation: middle drag orbits, Shift+middle pans.
    pub cad_mouse: bool,
    /// Multiplier for pan interaction speed.
    pub pan_speed: f32,
    /// Multiplier for orbit interaction speed.
    pub orbit_speed: f32,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            translucent: false,
            ghost_opacity: 0.45,
            ignore_look_colors: false,
            edges: true,
            triangles: false,
            normals: false,
            face_colours: false,
            zebra: false,
            roughness: 0.42,
            metallic: 0.0,
            cad_mouse: true,
            pan_speed: 1.0,
            orbit_speed: 1.0,
        }
    }
}

impl ViewSettings {
    fn any_extra(self) -> bool {
        self.translucent
            || !self.edges
            || self.triangles
            || self.normals
            || self.face_colours
            || self.zebra
            || self.ignore_look_colors
            || (self.ghost_opacity - 0.45).abs() > f32::EPSILON

    }
}

/// Which display bodies a gizmo drag preview moves.
///
/// Combine/Part results namespace their inputs as `body:{index}:` (nested
/// combines nest prefixes), so an edit deep in one branch can preview without
/// dragging the unrelated bodies it is assembled with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewScope {
    /// Every body in the display mesh is downstream of the edited node.
    All,
    /// Only bodies whose namespaced ids start with one of these prefixes.
    #[allow(dead_code)] // constructed by the body-scoped preview regression tests
    Bodies(Vec<String>),
}

impl Default for PreviewScope {
    fn default() -> Self {
        Self::All
    }
}

impl PreviewScope {
    pub fn is_affected(&self, body_id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Bodies(prefixes) => prefixes.iter().any(|prefix| body_id.starts_with(prefix)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectionMode {
    #[default]
    Object,
    Face,
    Edge,
}


fn distance_to_segment(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let length_sq = ab.length_sq();
    if length_sq <= f32::EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(ab) / length_sq).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

fn raycast_face(
    mesh: &solidcraft_geom::Mesh,
    face_ids: &[String],
    origin: solidcore_record::Vec3,
    direction: solidcore_record::Vec3,
) -> Option<String> {
    use solidcore_record::Vec3;
    let mut best = f64::INFINITY;
    let mut best_face = None;
    for (index, triangle) in mesh.triangles.iter().enumerate() {
        let (Some(a), Some(b), Some(c)) = (
            mesh.positions.get(triangle[0] as usize),
            mesh.positions.get(triangle[1] as usize),
            mesh.positions.get(triangle[2] as usize),
        ) else {
            continue;
        };
        let (a, b, c) = (*a, *b, *c);
        let edge1: Vec3 = b - a;
        let edge2: Vec3 = c - a;
        let h = direction.cross(edge2);
        let det = edge1.dot(h);
        if det.abs() < 1.0e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let s = origin - a;
        let u = inv * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(edge1);
        let v = inv * direction.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = inv * edge2.dot(q);
        if t > 1.0e-9 && t < best {
            let face = mesh.tri_face.get(index).copied().unwrap_or(u32::MAX) as usize;
            if let Some(id) = face_ids.get(face) {
                best = t;
                best_face = Some(id.clone());
            }
        }
    }
    best_face
}

#[derive(Clone, Debug)]
pub struct EdgePick {
    pub id: String,
    pub center: [f64; 3],
}

const ASYNC_DISPLAY_TRIANGLES: usize = 100_000;

#[cfg(feature = "gpu-viewport")]
type PreparedObject = crate::gpu_viewport::PreparedObject;
#[cfg(not(feature = "gpu-viewport"))]
type PreparedObject = ();

struct DisplayPrep {
    tx: std::sync::mpsc::Sender<(u64, NodeMesh, ViewSettings)>,
    rx: std::sync::mpsc::Receiver<(u64, DisplayMesh, Option<PreparedObject>)>,
}

impl Default for DisplayPrep {
    fn default() -> Self {
        let (tx, requests) = std::sync::mpsc::channel::<(u64, NodeMesh, ViewSettings)>();
        let (results, rx) = std::sync::mpsc::channel::<(u64, DisplayMesh, Option<PreparedObject>)>();
        std::thread::Builder::new()
            .name("nodecraft-display-prep".into())
            .spawn(move || {
                while let Ok((generation, mesh, view)) = requests.recv() {
                    let mut display = adapter::display_mesh(&mesh);
                    if view.ignore_look_colors {
                        display.face_colors.clear();
                    }
                    #[cfg(feature = "gpu-viewport")]
                    let prepared = Some(crate::gpu_viewport::prepare_object(&display, &view));
                    #[cfg(not(feature = "gpu-viewport"))]
                    let prepared = None;
                    if results.send((generation, display, prepared)).is_err() {
                        break;
                    }
                }
            })
            .ok();
        Self { tx, rx }
    }
}

fn ghost_plan_colour() -> [f32; 3] {
    let colour = crate::theme::get().ghost;
    [
        f32::from(colour[0]) / 255.0,
        f32::from(colour[1]) / 255.0,
        f32::from(colour[2]) / 255.0,
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct ViewSnapshot {
    pub camera: Camera,
    pub lens_shift: [f32; 2],
    pub camera_fit: bool,
    pub quality_detail: Option<u8>,
    pub full: [f32; 2],
    pub visible: [f32; 2],
}

pub struct Viewport {
    pub camera: Camera,
    chrome_visible: bool,
    display: Option<DisplayMesh>,
    /// Index 0 is the full display mesh; >0 selects a coarser display LOD.
    lod_index: usize,
    /// Display plans from the current evaluation, kept for GPU tessellation.
    pub(crate) plans: Vec<nodecraft_display::PlanInstance>,
    /// Planner defects from the current display tessellation.
    tessellation_diagnostics: Vec<String>,
    /// Whether the camera has been fitted to a display mesh. Before that, the
    /// target radius is unknown and the quality tier must stay at its initial
    /// coarse value.
    camera_fit: bool,
    /// Background display-mesh preparation (normal splitting, edges).
    display_prep: DisplayPrep,
    display_generation: u64,
    pending_display: bool,
    display_prep_started: Option<std::time::Instant>,
    fit_when_display_ready: bool,
    /// Background LOD generation for the current source mesh.
    lod_rx: Option<std::sync::mpsc::Receiver<(u64, Vec<adapter::DisplayLod>)>>,
    lod_generation: u64,
    lod_attempted: bool,
    /// Original protocol mesh for the current display, used to rebuild when a
    /// display-only option such as Ignore Look colours changes.
    source_mesh: Option<NodeMesh>,
    view: ViewSettings,
    /// NDC projection offset used to place the pivot at the visible viewport
    /// centre while leaving `camera.target` at the model centre.
    lens_shift: [f32; 2],
    /// The navigation mode latched at the start of a trackpad gesture: `true`
    /// for orbit, `false` for pan. A Shift release or missed modifier event
    /// must not switch a gesture from orbiting to panning halfway through.
    trackpad_orbit: Option<bool>,
    texture: Option<egui::TextureHandle>,
    dirty: bool,
    last_size: [usize; 2],
    /// Right-hand chrome inset used when a floating panel covers the viewport.
    pub chrome_right_inset: f32,
    last_full: Option<egui::Rect>,
    last_visible: Option<egui::Rect>,
    pending_view_restore: Option<ViewSnapshot>,
    /// Development-only camera diagnostics. Deliberately not serialized.
    show_view_matrix: bool,
    /// Opt-in GPU/frame diagnostics overlay. Off by default.
    show_gpu_info: bool,
    /// Smoothed frame interval in milliseconds, for the GPU info overlay.
    frame_ms: f32,
    selection_mode: SelectionMode,
    face_picks: Vec<String>,
    edge_picks: Vec<EdgePick>,
    /// Picking is disabled while the viewport still shows a different target than
    /// the selection session's input geometry.
    edge_picking_enabled: bool,
    /// Last quality tier selected, used for threshold hysteresis.
    quality_detail: Option<u8>,
    /// A click in a topology mode landed on empty space.
    selection_miss: bool,
    /// Rectangles owned by viewport chrome (compass, home, projection and display
    /// settings) that must swallow pointer input before sketch overlays see it.
    chrome_blockers: Vec<egui::Rect>,
    hovered_edge: Option<String>,
    hovered_face: Option<String>,
    selected_edges: std::collections::BTreeSet<String>,
    dock_rect: Option<egui::Rect>,
    transform_gizmo: Option<Gizmo>,
    gizmo_state: GizmoState,
    gizmo_update: Option<viewer_gizmo::GizmoUpdate>,
    /// The comparison layer is a feature-input ghost rather than an exact
    /// material difference; draw it more faintly.
    comparison_ghost: bool,
    /// Display-only drag preview applied to the GPU instances and scene edges
    /// until the kernel result for the edited values arrives.
    preview_transform: Option<[[f32; 4]; 4]>,
    /// Display bodies the preview moves (see [`PreviewScope`]).
    preview_scope: PreviewScope,
    /// Alpha for scoped edit-preview geometry. 1.0 is the normal opaque drag
    /// preview; lower values put the moved range in the ghost layer.
    preview_alpha: f32,
    /// Current standalone edit-ghost transform, retained so display-setting
    /// changes can update the ghost renderer without re-evaluating.
    edit_ghost_transform: Option<[[f32; 4]; 4]>,
    /// Branch B-rep edges for the transparent edit ghost, painted as a 2D
    /// overlay after the ghost faces.
    edit_ghost_edges: Vec<adapter::DisplayEdge>,
    /// B-rep plans for the edit-input body, used for exact edge picking.
    edit_ghost_plans: Vec<nodecraft_display::PlanInstance>,
    /// Display plans for the feature-input ghost (comparison fallback).
    feature_ghost_plans: Vec<nodecraft_display::PlanInstance>,
    /// Theme ghost colour last uploaded to the GPU ghost renderer.
    edit_ghost_color: [u8; 4],
    edit_ghost_alpha: f32,
    #[cfg(feature = "gpu-viewport")]
    gpu: Option<GpuViewport>,
}

/// The projection toggle shows the projection in use: perspective lattice when
/// perspective is active, square lattice when orthographic.
fn projection_icon(perspective: bool) -> (&'static [u8], &'static str) {
    if perspective {
        (
            include_bytes!("../../../../vendor/iconify-design/viewport/perspective.svg"),
            "bytes://nodecraft-perspective.svg",
        )
    } else {
        (
            include_bytes!("../../../../vendor/iconify-design/viewport/orthographic.svg"),
            "bytes://nodecraft-orthographic.svg",
        )
    }
}

fn draw_debug_box(ui: &egui::Ui, pos: egui::Pos2, width: f32, lines: &[String]) -> f32 {
    // Debug is the last layer egui paints, so the diagnostics stay above the
    // node editor, inspector, tool dock and any other floating panel. They are
    // painter-only and have no widgets, so nothing in them is clickable.
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Debug,
        egui::Id::new("viewport-debug-overlay"),
    ));
    let height = 19.0 + 15.0 * lines.len() as f32;
    painter.rect_filled(
        egui::Rect::from_min_size(pos - egui::vec2(6.0, 6.0), egui::vec2(width, height)),
        5.0,
        // More transparent than the old 190-alpha diagnostic box.
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
    );
    for (index, line) in lines.iter().enumerate() {
        painter.text(
            pos + egui::vec2(0.0, 15.0 * index as f32),
            egui::Align2::LEFT_TOP,
            line,
            egui::FontId::monospace(11.0),
            egui::Color32::from_rgb(210, 225, 220),
        );
    }
    height
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            chrome_visible: true,
            display: None,
            plans: Vec::new(),
            tessellation_diagnostics: Vec::new(),
            camera_fit: false,
            display_prep: DisplayPrep::default(),
            display_generation: 0,
            pending_display: false,
            display_prep_started: None,
            fit_when_display_ready: false,
            lod_index: 0,
            lod_rx: None,
            lod_generation: 0,
            lod_attempted: false,
            source_mesh: None,
            view: ViewSettings::default(),
            lens_shift: [0.0, 0.0],
            trackpad_orbit: None,
            texture: None,
            dirty: true,
            last_size: [0, 0],
            chrome_right_inset: 0.0,
            last_full: None,
            last_visible: None,
            pending_view_restore: None,
            show_view_matrix: false,
            show_gpu_info: false,
            frame_ms: 0.0,
            selection_mode: SelectionMode::Object,
            face_picks: Vec::new(),
            edge_picks: Vec::new(),
            edge_picking_enabled: false,
            quality_detail: None,
            selection_miss: false,
            chrome_blockers: Vec::new(),
            hovered_edge: None,
            hovered_face: None,
            selected_edges: std::collections::BTreeSet::new(),
            dock_rect: None,
            transform_gizmo: None,
            gizmo_state: GizmoState::default(),
            gizmo_update: None,
            comparison_ghost: false,
            preview_transform: None,
            preview_scope: PreviewScope::All,
            preview_alpha: 1.0,
            edit_ghost_transform: None,
            edit_ghost_edges: Vec::new(),
                    edit_ghost_plans: Vec::new(),
            feature_ghost_plans: Vec::new(),
            edit_ghost_color: [0; 4],
            edit_ghost_alpha: 0.45,
            #[cfg(feature = "gpu-viewport")]
            gpu: None,
        }
    }
}

impl Viewport {
    #[cfg(feature = "gpu-viewport")]
    pub fn install_gpu(&mut self, render_state: &egui_wgpu::RenderState) {
        self.gpu = GpuViewport::install(render_state);
    }

    #[cfg(not(feature = "gpu-viewport"))]
    pub fn install_gpu(&mut self, _render_state: &egui_wgpu::RenderState) {}

    pub fn gpu_active(&self) -> bool {
        #[cfg(feature = "gpu-viewport")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            false
        }
    }

    pub fn set_theme(&mut self, theme: &crate::theme::Theme) {
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_background(theme.viewport_background);
        }
        self.dirty = true;
    }

    pub fn set_chrome_visible(&mut self, visible: bool) {
        self.chrome_visible = visible;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_chrome_visible(visible);
        }
        self.dirty = true;
    }

    /// Provisional display detail for the current camera: `None` means the full
    /// plan, `Some(detail)` asks the tessellation worker for a bounded preview
    /// plan. Small bodies clamp back to full quality in the worker.
    #[must_use]
    pub fn quality_detail(&mut self) -> Option<u8> {
        let Some(display) = self.display.as_ref() else {
            return Some(1);
        };
        if !self.camera_fit {
            return Some(1);
        }
        let ratio = self.camera.distance / display.radius.max(1.0e-9);
        let raw = if ratio < 0.75 {
            8
        } else if ratio < 1.25 {
            6
        } else if ratio < 2.5 {
            4
        } else if ratio < 4.0 {
            2
        } else {
            1
        };
        let Some(current) = self.quality_detail else {
            self.quality_detail = Some(raw);
            return Some(raw);
        };
        if raw == current {
            return Some(current);
        }
        // Hysteresis: require the camera to cross the tier boundary by 10%
        // before switching, so a threshold cannot oscillate every frame.
        const TIERS: [u8; 5] = [8, 6, 4, 2, 1];
        const BOUNDS: [f64; 4] = [0.75, 1.25, 2.5, 4.0];
        let Some(current_index) = TIERS.iter().position(|tier| *tier == current) else {
            self.quality_detail = Some(raw);
            return Some(raw);
        };
        let Some(raw_index) = TIERS.iter().position(|tier| *tier == raw) else {
            return Some(current);
        };
        let accepted = if raw_index < current_index {
            BOUNDS.get(current_index.saturating_sub(1)).is_some_and(|boundary| ratio < boundary * 0.9)
        } else {
            BOUNDS.get(current_index).is_some_and(|boundary| ratio > boundary * 1.1)
        };
        if accepted {
            self.quality_detail = Some(raw);
            Some(raw)
        } else {
            Some(current)
        }
    }

    pub fn set_tessellation_diagnostics(&mut self, diagnostics: Vec<String>) {
        self.tessellation_diagnostics = diagnostics;
    }

    pub fn plan_ready(&self) -> bool {
        #[cfg(feature = "gpu-viewport")]
        {
            self.gpu.as_ref().is_some_and(|gpu| gpu.plan_ready())
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            true
        }
    }

    pub fn plan_pending(&self) -> bool {
        #[cfg(feature = "gpu-viewport")]
        {
            self.plan_ready()
                && self
                    .gpu
                    .as_ref()
                    .and_then(|gpu| gpu.plan_debug())
                    .is_some_and(|debug| debug.pending)
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            false
        }
    }

    pub fn view_plans_hash(&self) -> u64 {
        crate::gpu_plan::plan_identity_hash(&self.plans)
    }

    pub fn plan_debug(&self) -> Option<crate::gpu_plan::PlanDebug> {
        #[cfg(feature = "gpu-viewport")]
        {
            self.gpu.as_ref().and_then(|gpu| gpu.plan_debug())
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            None
        }
    }

    pub fn set_plans(&mut self, plans: Vec<nodecraft_display::PlanInstance>) {
        self.plans = plans;
        self.refresh_plan_state();
    }

    /// Plan rendering needs a uniform body colour. Re-evaluate that after the
    /// display mesh (and therefore its face colours) becomes available.
    fn refresh_plan_state(&mut self) {
        // The plan path carries per-vertex face colours itself; only the
        // override modes need the CPU display colours.
        let colour = self
            .view
            .ignore_look_colors
            .then_some([0.17, 0.18, 0.195]);
        let enabled = !self.plans.is_empty() && !self.view.face_colours;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_plans(self.plans.clone(), enabled, colour);
        }
    }

    /// Drop every display resource owned by the current document: plans, GPU
    /// plan chunks, meshes and ghosts. Session/tab changes must go through this
    /// so a closed project cannot keep resident plan meshes.
    pub fn clear_document_display(&mut self) {
        self.set_plans(Vec::new());
        self.set_mesh(None);
        self.clear_edit_ghost();
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_plan_document();
            gpu.clear_ghost();
        }
    }

    pub fn set_mesh(&mut self, mesh: Option<&NodeMesh>) {
        self.update_mesh(mesh);
        self.fit();
    }

    /// Update the mesh without recentering the camera. Used while a transform
    /// gizmo is being dragged so the viewport follows the edit instead of
    /// snapping back to a fitted view on every evaluation result.
    pub fn set_mesh_preserving_camera(&mut self, mesh: Option<&NodeMesh>) {
        self.update_mesh(mesh);
    }



    pub fn set_exact_comparison_mesh(
        &mut self,
        base: &NodeMesh,
        added: &NodeMesh,
        removed: &NodeMesh,
    ) {
        self.comparison_ghost = false;
        self.selected_edges.clear();
        self.hovered_edge = None;
        self.reset_lods();
        self.display = Some(crate::adapter::display_mesh_with_difference(base, added, removed));
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
        }
        self.dirty = true;
    }

    /// Display `ghost` as a faint feature-input layer under `base`. This is not
    /// an exact material difference; it is the selected feature's upstream body.
    pub fn set_feature_ghost_mesh(&mut self, base: &NodeMesh, ghost: &NodeMesh) {
        self.comparison_ghost = true;
        self.selected_edges.clear();
        self.hovered_edge = None;
        let empty = NodeMesh::default();
        self.reset_lods();
        self.display = Some(crate::adapter::display_mesh_with_difference(base, &empty, ghost));
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
            gpu.set_ghost_matrix(None);
            gpu.set_ghost_plans(
                self.feature_ghost_plans.clone(),
                self.view.ghost_opacity,
                Some(ghost_plan_colour()),
                !self.feature_ghost_plans.is_empty(),
            );
        }
        self.dirty = true;
    }

    pub fn set_selection_comparison_mesh(
        &mut self,
        base: &NodeMesh,
        selection_source: &NodeMesh,
        added: &NodeMesh,
        removed: &NodeMesh,
    ) {
        self.comparison_ghost = false;
        self.selected_edges.clear();
        self.hovered_edge = None;
        let mut display = crate::adapter::display_mesh_with_difference(base, added, removed);
        // The selection display is originalIn -> difference overlay -> originalIn.edges.
        // Replace the result body's edge set so no generated edge can be drawn or picked.
        display.edges = crate::adapter::display_mesh(selection_source).edges;
        display.selection_edges = None;
        self.reset_lods();
        self.display = Some(display);
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
        }
        self.dirty = true;
    }

    fn update_mesh(&mut self, mesh: Option<&NodeMesh>) {
        self.comparison_ghost = false;
        // Edge ids belong to the previous display target. Keeping the highlight
        // across an evaluation makes unrelated edges look selected on the next
        // node/document.
        self.selected_edges.clear();
        self.hovered_edge = None;
        // A new display mesh is authoritative: it already sits at the edited
        // placement, so the drag preview has done its job. Keep an edit ghost
        // while its transform is still live; the main evaluation must not blank
        // the ghost's faces between preview frames.
        let keep_edit_ghost = self.edit_ghost_transform.is_some();
        self.preview_transform = None;
        self.preview_scope = PreviewScope::All;
        self.preview_alpha = 1.0;
        if !keep_edit_ghost {
            self.edit_ghost_transform = None;
            self.edit_ghost_edges.clear();
            self.edit_ghost_plans.clear();
            #[cfg(feature = "gpu-viewport")]
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.clear_ghost_edges();
            }
            self.feature_ghost_plans.clear();
            self.edit_ghost_color = [0; 4];
        }
        self.source_mesh = mesh.cloned();
        self.reset_lods();
        if let Some(mesh) = mesh {
            if mesh.faces.len() >= ASYNC_DISPLAY_TRIANGLES {
                // Conversion and normal splitting run on the display-prep
                // worker; the old display stays visible until the new one is
                // ready.
                self.pending_display = true;
                self.display_prep_started = Some(std::time::Instant::now());
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[viewport] async display prep faces={}", mesh.faces.len());
                }
                let generation = self.display_generation;
                let _ = self
                    .display_prep
                    .tx
                    .send((generation, mesh.clone(), self.view));
                self.dirty = true;
                #[cfg(feature = "gpu-viewport")]
                if let Some(gpu) = self.gpu.as_mut() {
                    if !keep_edit_ghost {
                        gpu.clear_ghost();
                    }
                }
                return;
            }
            self.display = Some(self.display_mesh_for_view(mesh));
        } else {
            self.display = None;
        }
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
            if !keep_edit_ghost {
                gpu.clear_ghost();
            }
        }
    }

    fn poll_display_prep(&mut self) {
        let Some((generation, display, prepared)) = self.display_prep.rx.try_recv().ok() else {
            return;
        };
        if generation != self.display_generation || !self.pending_display {
            return;
        }
        self.pending_display = false;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[viewport] display prep done {}ms",
                self.display_prep_started.take().map_or(0, |started| started.elapsed().as_millis())
            );
        }
        self.display = Some(display);
        self.refresh_plan_state();
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_prepared_mesh(self.display.as_ref(), prepared);
        }
        if self.fit_when_display_ready {
            self.fit_when_display_ready = false;
            self.fit();
        }
        self.dirty = true;
    }

    fn display_mesh_for_view(&self, mesh: &NodeMesh) -> DisplayMesh {
        let mut display = adapter::display_mesh(mesh);
        if self.view.ignore_look_colors {
            display.face_colors.clear();
        }
        display
    }

    fn reset_lods(&mut self) {
        self.lod_index = 0;
        self.lod_rx = None;
        self.lod_generation = self.lod_generation.wrapping_add(1);
        self.lod_attempted = false;
        self.display_generation = self.display_generation.wrapping_add(1);
        self.pending_display = false;
        self.fit_when_display_ready = false;
    }

    fn poll_lods(&mut self) {
        let Some(receiver) = self.lod_rx.as_ref() else {
            return;
        };
        let result = receiver.try_recv();
        let Ok((generation, lods)) = result else {
            return;
        };
        self.lod_rx = None;
        self.lod_attempted = true;
        if generation != self.lod_generation {
            return;
        }
        if let Some(display) = self.display.as_mut() {
            display.lods = lods;
            self.dirty = true;
        }
    }

    fn request_lods(&mut self) {
        if self.lod_rx.is_some() || self.lod_attempted {
            return;
        }
        let Some(source) = self.source_mesh.clone() else {
            return;
        };
        self.lod_attempted = true;
        let generation = self.lod_generation;
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("nodecraft-display-lod".into())
            .spawn(move || {
                let lods = adapter::generate_lods(&source);
                let _ = sender.send((generation, lods));
            })
            .ok();
        self.lod_rx = Some(receiver);
    }

    fn update_display_lod(&mut self) {
        self.poll_lods();
        let lod = {
            let Some(display) = self.display.as_ref() else {
                self.lod_index = 0;
                return;
            };
            if display.lods.is_empty()
                || !display.face_colors.is_empty()
                || self.view.face_colours
                || self.view.triangles
                || self.view.normals
                || self.view.zebra
            {
                0
            } else {
                let ratio = self.camera.distance / display.radius.max(1.0e-9);
                // Fit uses distance = 3 * radius, so keep the full mesh at fit
                // and simplify only when zooming out beyond it.
                match ratio {
                    ratio if ratio < 3.2 => 0,
                    ratio if ratio < 4.5 => 1,
                    ratio if ratio < 7.0 => 2,
                    _ => 3,
                }
                .min(display.lods.len())
            }
        };
        // Generate levels lazily on the first zoom-out. The full mesh is shown
        // until they arrive.
        if lod > 0 && self.display.as_ref().is_some_and(|display| display.lods.is_empty()) {
            self.request_lods();
        }
        if lod == self.lod_index {
            return;
        }
        let Some(display) = self.display.as_ref() else { return };
        self.lod_index = lod;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_lod_mesh(display.render_mesh(lod));
        }
        self.dirty = true;
    }

    pub fn set_view(&mut self, view: ViewSettings) {
        if self.view == view {
            return;
        }
        let ghost_changed = self.view.ghost_opacity != view.ghost_opacity;
        let look_bypass_changed = self.view.ignore_look_colors != view.ignore_look_colors;
        self.view = view;
        if look_bypass_changed && let Some(source) = self.source_mesh.clone() {
            self.reset_lods();
            self.display = Some(self.display_mesh_for_view(&source));
            #[cfg(feature = "gpu-viewport")]
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.set_view(self.view, self.display.as_ref());
                gpu.set_mesh(self.display.as_ref());
            }
        }
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(view, self.display.as_ref());
            if ghost_changed {
                self.edit_ghost_alpha = view.ghost_opacity;
                // The GPU plan ghost carries its alpha on the instances, not in
                // a uniform; rebuild its chunks when the slider moves.
                if self.edit_ghost_transform.is_some() && !self.edit_ghost_plans.is_empty() {
                    gpu.set_ghost_plans(
                        self.edit_ghost_plans.clone(),
                        view.ghost_opacity,
                        Some(ghost_plan_colour()),
                        true,
                    );
                } else if !self.feature_ghost_plans.is_empty() {
                    gpu.set_ghost_plans(
                        self.feature_ghost_plans.clone(),
                        view.ghost_opacity,
                        Some(ghost_plan_colour()),
                        true,
                    );
                }
            }
            if ghost_changed && self.preview_transform.is_some() {
                // A live ghost-transparency slider must update the current
                // preview instances, not wait for the next evaluation.
                gpu.set_preview_transform(
                    self.preview_transform,
                    self.display.as_ref(),
                    self.preview_scope.clone(),
                    view.ghost_opacity,
                );
            }
        }
        if ghost_changed {
            self.preview_alpha = view.ghost_opacity;
        }
        // Face-colour and translucency modes switch the plan renderer off/on.
        self.refresh_plan_state();
        self.dirty = true;
    }

    pub fn view(&self) -> ViewSettings {
        self.view
    }

    pub fn selection_mode(&self) -> SelectionMode {
        self.selection_mode
    }

    pub fn take_face_picks(&mut self) -> Vec<String> {
        std::mem::take(&mut self.face_picks)
    }

    pub fn take_edge_picks(&mut self) -> Vec<EdgePick> {
        std::mem::take(&mut self.edge_picks)
    }


    pub fn set_selection_mode(&mut self, mode: SelectionMode) {
        if mode != self.selection_mode {
            self.selected_edges.clear();
            self.hovered_edge = None;
        }
        self.selection_mode = mode;
    }

    pub fn clear_edge_selection(&mut self) {
        self.selected_edges.clear();
        self.hovered_edge = None;
    }


    pub fn edge_picking_enabled(&self) -> bool {
        self.edge_picking_enabled
    }

    pub fn take_selection_miss(&mut self) -> bool {
        std::mem::take(&mut self.selection_miss)
    }

    /// Rects of interactive viewport chrome, for overlays that read raw pointer
    /// input instead of egui widget responses.
    pub fn interaction_blockers(&self) -> &[egui::Rect] {
        &self.chrome_blockers
    }

    pub fn set_edge_picking_enabled(&mut self, enabled: bool) {
        self.edge_picking_enabled = enabled;
        if !enabled {
            self.edge_picks.clear();
        }
    }

    /// Highlight the exact display-edge ids stored on the node. These are the
    /// same ids the blend resolver validates, so the highlight and the
    /// treatment cannot disagree about which edge was picked.
    pub fn select_edges_by_ids(&mut self, ids: &[String]) {
        let Some(display) = self.display.as_ref() else {
            self.selected_edges.clear();
            return;
        };
        let available: std::collections::HashSet<&str> = if self.picking_input_ghost_only() {
            self.edit_ghost_edges.iter().map(|edge| edge.id.as_str()).collect()
        } else {
            self.topology_edges(display)
                .iter()
                .chain(self.edit_ghost_edges.iter())
                .map(|edge| edge.id.as_str())
                .collect()
        };
        self.selected_edges = ids
            .iter()
            .filter(|id| available.contains(id.as_str()))
            .cloned()
            .collect();
    }

    pub fn select_edges_by_centers(&mut self, centers: &[[f64; 3]]) {
        let Some(display) = self.display.as_ref() else {
            self.selected_edges.clear();
            return;
        };
        let source: &[crate::adapter::DisplayEdge] = if self.picking_input_ghost_only() {
            &self.edit_ghost_edges
        } else {
            self.topology_edges(display)
        };
        let selected: Vec<String> = centers
            .iter()
            .filter_map(|center| {
                source
                    .iter()
                    .min_by(|a, b| {
                        center_distance(&a.midpoint, center)
                            .total_cmp(&center_distance(&b.midpoint, center))
                    })
                    .filter(|edge| center_distance(&edge.midpoint, center) <= 1.0e-4)
                    .map(|edge| edge.id.clone())
            })
            .collect();
        self.selected_edges = selected.into_iter().collect();
    }

    /// While a feature's input body is shown as the edit ghost, topology
    /// picking belongs to that body only. The downstream breakpoint remains
    /// visible underneath, but its edges are not valid feature selections.
    fn picking_input_ghost_only(&self) -> bool {
        self.edit_ghost_transform.is_some() && !self.edit_ghost_edges.is_empty()
    }

    fn topology_edges<'a>(&'a self, display: &'a DisplayMesh) -> &'a [crate::adapter::DisplayEdge] {
        display
            .selection_edges
            .as_deref()
            .unwrap_or(display.edges.as_slice())
    }

    /// Face ids in display-triangle order, derived from the display target's
    /// `TessPlan` faces. This keeps face picking on the exact B-rep face ids
    /// instead of relying on the display mesh's group strings.
    fn plan_face_ids(&self) -> Option<Vec<String>> {
        let display = self.display.as_ref()?;
        let multiple = self.plans.len() > 1;
        let mut ids = Vec::new();
        for (body, instance) in self.plans.iter().enumerate() {
            for face in &instance.plan.faces {
                let triangles = face
                    .triangles
                    .iter()
                    .filter(|triangle| {
                        triangle
                            .iter()
                            .all(|index| (*index as usize) < face.uv.len())
                    })
                    .count();
                let id = if multiple {
                    format!("body:{body}:face:{}", face.face)
                } else {
                    format!("face:{}", face.face)
                };
                ids.extend(std::iter::repeat_n(id, triangles));
            }
        }
        (ids.len() == display.mesh.triangles.len()).then_some(ids)
    }

    pub fn edge_ids(&self) -> std::collections::HashSet<String> {
        if self.picking_input_ghost_only() {
            return self.edit_ghost_edges.iter().map(|edge| edge.id.clone()).collect();
        }
        self.display
            .as_ref()
            .map(|display| {
                self.topology_edges(display)
                    .iter()
                    .map(|edge| edge.id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn fit(&mut self) {
        if self.pending_display {
            self.fit_when_display_ready = true;
            return;
        }
        let Some(display) = &self.display else {
            self.camera = Camera::default();
            self.dirty = true;
            return;
        };
        let (yaw, pitch) = StandardView::Iso.angles();
        self.camera.target = display.center;
        // SolidCraft's yaw sign is opposite the Swift viewer's projection convention.
        // Iso is the view that exposes the mirror; front/back are unchanged by yaw 0/pi.
        self.camera.yaw = -yaw;
        self.camera.pitch = pitch;
        self.camera.distance = (display.radius * 3.0).max(1.0);
        self.lens_shift = [0.0, 0.0];
        self.camera_fit = true;
        self.dirty = true;
    }

    /// Fit distance for a bounding radius and viewport aspect ratio.
    fn fit_distance(&self, radius: f64, aspect: f64) -> f64 {
        let tan_half = if self.camera.fov > 0.0 {
            (self.camera.fov * 0.5).tan()
        } else {
            0.4
        };
        let vertical = radius / (tan_half * 0.9);
        let horizontal = radius / (tan_half * aspect.max(1.0e-3) * 0.9);
        vertical.max(horizontal).max(1.0)
    }

    /// Home/ISO framing: put the shared world origin at the centre of the
    /// unobstructed 3D viewport and frame the whole model around it. The orbit
    /// pivot stays at the origin; composition uses the projection lens shift.
    fn frame_origin(&mut self, radius: f64, aspect: f64, lens_shift: [f32; 2]) {
        let (yaw, pitch) = StandardView::Iso.angles();
        self.camera.yaw = -yaw;
        self.camera.pitch = pitch;
        self.camera.roll = 0.0;
        self.camera.target = VIEW_ORIGIN;
        self.camera.distance = self.fit_distance(radius, aspect);
        self.lens_shift = lens_shift;
        self.dirty = true;
    }

    pub fn fit_origin(&mut self, full: egui::Rect, visible: egui::Rect) {
        let Some(display) = &self.display else {
            self.camera = Camera::default();
            self.dirty = true;
            return;
        };
        let visible = visible.intersect(full);
        if visible.width() <= 1.0 || visible.height() <= 1.0 {
            let radius = (display.center.dist(VIEW_ORIGIN) + display.radius).max(1.0);
            self.frame_origin(radius, f64::from(full.width() / full.height().max(1.0)), [0.0, 0.0]);
            return;
        }
        let radius = (display.center.dist(VIEW_ORIGIN) + display.radius).max(1.0);
        let aspect = f64::from(visible.width() / visible.height().max(1.0));
        let lens = lens_shift_for_visible(full, visible);
        self.frame_origin(radius, aspect, lens);
    }

    pub fn show_view_matrix(&self) -> bool {
        self.show_view_matrix
    }

    pub fn show_gpu_info(&self) -> bool {
        self.show_gpu_info
    }

    pub fn set_show_gpu_info(&mut self, show: bool) {
        self.show_gpu_info = show;
    }

    fn tick_frame_stats(&mut self, ui: &egui::Ui) {
        let dt = ui.ctx().input(|input| input.stable_dt).clamp(1.0e-4, 1.0);
        let ms = dt * 1000.0;
        self.frame_ms = if self.frame_ms > 0.0 {
            self.frame_ms * 0.85 + ms * 0.15
        } else {
            ms
        };
    }

    /// NDC offset used to compose the camera into the visible viewport.
    pub fn lens_shift(&self) -> [f32; 2] {
        self.lens_shift
    }

    #[must_use]
    pub fn view_snapshot(&self) -> Option<ViewSnapshot> {
        // Before the first frame after a restore, keep the original framing
        // rects rather than the stale live rects.
        if let Some(pending) = self.pending_view_restore {
            return Some(pending);
        }
        let full = self.last_full?;
        let visible = self.last_visible?;
        Some(ViewSnapshot {
            camera: self.camera,
            lens_shift: self.lens_shift,
            camera_fit: self.camera_fit,
            quality_detail: self.quality_detail,
            full: [full.width(), full.height()],
            visible: [visible.width(), visible.height()],
        })
    }

    pub fn restore_view(&mut self, snapshot: ViewSnapshot) {
        self.camera = snapshot.camera;
        self.lens_shift = snapshot.lens_shift;
        self.camera_fit = snapshot.camera_fit;
        self.quality_detail = snapshot.quality_detail;
        self.pending_view_restore = Some(snapshot);
        self.dirty = true;
    }

    fn apply_pending_view_restore(&mut self, full: egui::Rect, visible: egui::Rect) {
        let Some(snapshot) = self.pending_view_restore.take() else { return };
        let old_aspect = if snapshot.visible[1] > 1.0 {
            snapshot.visible[0] / snapshot.visible[1]
        } else {
            1.0
        };
        let new_aspect = visible.width() / visible.height().max(1.0);
        if new_aspect < old_aspect && new_aspect > 1.0e-6 {
            self.camera.distance *= f64::from(old_aspect / new_aspect);
            self.dirty = true;
        }
        if snapshot.lens_shift != [0.0, 0.0] {
            self.lens_shift = lens_shift_for_visible(full, visible);
        }
    }

    pub fn set_show_view_matrix(&mut self, show: bool) {
        self.show_view_matrix = show;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn home(&mut self) {
        if let (Some(full), Some(visible)) = (self.last_full, self.last_visible) {
            self.fit_origin(full, visible);
        }
    }

    pub fn set_standard_view(&mut self, view: StandardView) {
        let (yaw, pitch) = view.angles();
        self.camera.yaw = if matches!(view, StandardView::Iso) { -yaw } else { yaw };
        self.camera.pitch = pitch;
        self.dirty = true;
    }

    /// Switch projection while keeping the pivot, orientation and apparent
    /// half-height at the target fixed. Industrial CAD toggles projection
    /// without recomposing the view.
    pub fn toggle_projection(&mut self) {
        const FOV: f64 = std::f64::consts::PI * 32.0 / 180.0;
        const ORTHO_HALF_HEIGHT: f64 = 0.4;
        if self.camera.fov > 0.0 {
            let perspective_half = self.camera.distance * (self.camera.fov * 0.5).tan();
            self.camera.distance = perspective_half / ORTHO_HALF_HEIGHT;
            self.camera.fov = 0.0;
        } else {
            let ortho_half = self.camera.distance * ORTHO_HALF_HEIGHT;
            self.camera.distance = ortho_half / (FOV * 0.5).tan();
            self.camera.fov = FOV;
        }
        self.dirty = true;
    }

    pub fn set_transform_gizmo(&mut self, gizmo: Option<Gizmo>) {
        if self
            .transform_gizmo
            .as_ref()
            .map(|current| current.node_id.as_str())
            != gizmo.as_ref().map(|next| next.node_id.as_str())
        {
            self.gizmo_state.reset();
            // A selection change mid-drag must not leave the previewed geometry
            // displaced after the triad disappears.
            self.set_preview_transform(None, PreviewScope::All);
        }
        self.transform_gizmo = gizmo;
    }

    pub fn take_gizmo_update(&mut self) -> Option<viewer_gizmo::GizmoUpdate> {
        self.gizmo_update.take()
    }

    /// Move the display geometry by `transform` without re-uploading the merged
    /// mesh: the object renderer updates the instances of the bodies in `scope`
    /// and the scene edge lines are rebuilt at the previewed placement.
    /// Show a standalone edit ghost over a dimmed breakpoint mesh. The branch is
    /// uploaded to its own object renderer, so it is never part of the scene.
    pub fn set_edit_ghost_plans(&mut self, plans: Vec<nodecraft_display::PlanInstance>) {
        self.edit_ghost_plans = plans;
        self.apply_ghost_plan_edges();
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            let enabled = self.edit_ghost_transform.is_some() && !self.edit_ghost_plans.is_empty();
            gpu.set_ghost_plans(
                self.edit_ghost_plans.clone(),
                self.edit_ghost_alpha,
                Some(ghost_plan_colour()),
                enabled,
            );
        }
    }

    /// Plan instances for the comparison fallback ghost, owned by the display
    /// service and evaluated by the GPU mesher.
    /// Whether the transform ghost has GPU plans ready to draw.
    pub fn has_edit_ghost_plans(&self) -> bool {
        !self.edit_ghost_plans.is_empty()
    }

    pub fn set_feature_ghost_plans(&mut self, plans: Vec<nodecraft_display::PlanInstance>) {
        self.feature_ghost_plans = plans;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_ghost_plans(
                self.feature_ghost_plans.clone(),
                self.view.ghost_opacity,
                Some(ghost_plan_colour()),
                !self.feature_ghost_plans.is_empty(),
            );
        }
    }

    /// Prefer the input body's B-rep plan edges over display-mesh edges for the
    /// feature's pickable ghost. Plan edges are the exact body's edge polylines
    /// and carry body-scoped ids, independent of the displayed result/LOD.
    fn polyline_midpoint(points: &[Vec3]) -> Vec3 {
        let length: f64 = points
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).len())
            .sum();
        if length <= 1.0e-18 {
            return points.first().copied().unwrap_or(Vec3::ZERO);
        }
        let mut walked = 0.0;
        for pair in points.windows(2) {
            let segment = (pair[1] - pair[0]).len();
            if walked + segment >= length * 0.5 {
                let t = ((length * 0.5) - walked) / segment.max(1.0e-18);
                return pair[0] + (pair[1] - pair[0]) * t;
            }
            walked += segment;
        }
        points.last().copied().unwrap_or(Vec3::ZERO)
    }

    fn apply_ghost_plan_edges(&mut self) {
        if self.edit_ghost_plans.is_empty() {
            return;
        }
        let multiple_bodies = self.edit_ghost_plans.len() > 1;
        let mut edges = Vec::new();
        for (body, instance) in self.edit_ghost_plans.iter().enumerate() {
            let matrix = crate::gpu_plan::instance_matrix(&instance.transforms);
            for (index, polyline) in instance.plan.edges.iter().enumerate() {
                let points: Vec<Vec3> = polyline
                    .iter()
                    .map(|point| {
                        let x = f64::from(matrix[0][0]) * point.x
                            + f64::from(matrix[1][0]) * point.y
                            + f64::from(matrix[2][0]) * point.z
                            + f64::from(matrix[3][0]);
                        let y = f64::from(matrix[0][1]) * point.x
                            + f64::from(matrix[1][1]) * point.y
                            + f64::from(matrix[2][1]) * point.z
                            + f64::from(matrix[3][1]);
                        let z = f64::from(matrix[0][2]) * point.x
                            + f64::from(matrix[1][2]) * point.y
                            + f64::from(matrix[2][2]) * point.z
                            + f64::from(matrix[3][2]);
                        Vec3::new(x, y, z)
                    })
                    .collect();
                if points.len() < 2 {
                    continue;
                }
                let midpoint = Self::polyline_midpoint(&points);
                let id = instance
                    .edge_ids
                    .get(index)
                    .and_then(Clone::clone)
                    .map(|id| {
                        if multiple_bodies {
                            format!("body:{body}:{id}")
                        } else {
                            id
                        }
                    })
                    .unwrap_or_else(|| {
                        if multiple_bodies {
                            format!("preview-edge:body:{body}:{index}")
                        } else {
                            format!("preview-edge:{index}")
                        }
                    });
                edges.push(crate::adapter::DisplayEdge {
                    id,
                    points,
                    midpoint: [midpoint.x, midpoint.y, midpoint.z],
                    normals: Vec::new(),
                });
            }
        }
        if !edges.is_empty() {
            self.edit_ghost_edges = edges;
        }
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_ghost_edges(&self.edit_ghost_edges);
            gpu.set_ghost_edge_matrix(self.edit_ghost_transform);
        }
    }

    pub fn set_edit_ghost(
        &mut self,
        base: &NodeMesh,
        branch: &NodeMesh,
        transform: [[f32; 4]; 4],
        alpha: f32,
    ) {
        let started = std::time::Instant::now();
        // With resident plans, the breakpoint is already on the GPU. Rebuilding
        // the merged 1,000-body mesh on the UI thread just to dim it costs
        // seconds; keep the plan chunks and only install the ghost overlay.
        if !self.plans.is_empty() && self.gpu_active() {
            self.comparison_ghost = false;
            self.preview_transform = None;
            self.preview_scope = PreviewScope::All;
            self.preview_alpha = 1.0;
            self.edit_ghost_transform = Some(transform);
            if self.edit_ghost_plans.is_empty() {
                self.edit_ghost_edges = self.display_mesh_for_view(branch).edges;
            } else {
                self.apply_ghost_plan_edges();
            }
            self.edit_ghost_alpha = alpha;
            #[cfg(feature = "gpu-viewport")]
            if let Some(gpu) = self.gpu.as_mut() {
                let use_plan_ghost = !self.edit_ghost_plans.is_empty();
                gpu.set_ghost_matrix(Some(transform));
                gpu.set_ghost_plans(
                    self.edit_ghost_plans.clone(),
                    alpha,
                    Some(ghost_plan_colour()),
                    use_plan_ghost,
                );
                gpu.set_ghost_edges(&self.edit_ghost_edges);
                gpu.set_ghost_edge_matrix(Some(transform));
            }
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!("[viewport] plan-base edit ghost {}ms", started.elapsed().as_millis());
            }
            self.dirty = true;
            return;
        }
        let base_source = base.clone();
        let base = self.display_mesh_for_view(base);
        self.comparison_ghost = false;
        self.preview_transform = None;
        self.preview_scope = PreviewScope::All;
        self.preview_alpha = 1.0;
        self.reset_lods();
        self.display = Some(base);
        // Keep the dimmed breakpoint source so display-only options such as
        // Ignore Look colours can rebuild the base while ghost mode is active.
        self.source_mesh = Some(base_source);
        self.edit_ghost_transform = Some(transform);
        // GPU plans carry the exact branch edges; only build a CPU branch
        // display when plans have not arrived yet.
        if self.edit_ghost_plans.is_empty() {
            self.edit_ghost_edges = self.display_mesh_for_view(branch).edges;
        } else {
            self.apply_ghost_plan_edges();
        }
        self.edit_ghost_alpha = alpha;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
            let use_plan_ghost = !self.edit_ghost_plans.is_empty();
            gpu.set_ghost_matrix(Some(transform));
            gpu.set_ghost_plans(
                self.edit_ghost_plans.clone(),
                alpha,
                Some(ghost_plan_colour()),
                use_plan_ghost,
            );
            gpu.set_ghost_edges(&self.edit_ghost_edges);
            gpu.set_ghost_edge_matrix(Some(transform));
        }
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[viewport] cpu edit ghost {}ms", started.elapsed().as_millis());
        }
        self.dirty = true;
    }

    pub fn update_ghost_transform(&mut self, transform: Option<[[f32; 4]; 4]>, alpha: f32) {
        let was_enabled = self.edit_ghost_transform.is_some();
        let alpha_changed = (self.edit_ghost_alpha - alpha).abs() > f32::EPSILON;
        self.edit_ghost_transform = transform;
        self.edit_ghost_alpha = alpha;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            // Matrix changes are a uniform; only a set/alpha/enabled change
            // requires rebuilding the ghost chunks.
            gpu.set_ghost_matrix(transform);
            gpu.set_ghost_edge_matrix(transform);
            if alpha_changed || was_enabled != transform.is_some() {
                gpu.set_ghost_plans(
                    self.edit_ghost_plans.clone(),
                    alpha,
                    Some(ghost_plan_colour()),
                    transform.is_some() && !self.edit_ghost_plans.is_empty(),
                );
            }
        }
        self.dirty = true;
    }

    pub fn clear_edit_ghost(&mut self) {
        self.edit_ghost_transform = None;
        self.edit_ghost_edges.clear();
        self.edit_ghost_plans.clear();
        self.edit_ghost_color = [0; 4];
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_ghost();
        }
    }

    pub fn set_preview_transform(&mut self, transform: Option<[[f32; 4]; 4]>, scope: PreviewScope) {
        self.set_preview_transform_with_alpha(transform, scope, 1.0);
    }

    /// Scoped preview with an explicit alpha. Values below 1.0 render the moved
    /// range in the depth-test-free ghost layer over the opaque scene.
    pub fn set_preview_transform_with_alpha(
        &mut self,
        transform: Option<[[f32; 4]; 4]>,
        scope: PreviewScope,
        alpha: f32,
    ) {
        // A scoped preview only works when the display mesh actually carries
        // those body names; otherwise fall back to moving the whole mesh rather
        // than freezing the preview.
        let scope = self.resolve_preview_scope(scope);
        if self.preview_transform == transform
            && self.preview_scope == scope
            && self.preview_alpha == alpha
        {
            return;
        }
        self.preview_transform = transform;
        self.preview_scope = scope.clone();
        self.preview_alpha = alpha;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_preview_transform(transform, self.display.as_ref(), scope, alpha);
        }
        self.dirty = true;
    }

    fn resolve_preview_scope(&self, scope: PreviewScope) -> PreviewScope {
        let PreviewScope::Bodies(_) = &scope else { return scope };
        let Some(display) = self.display.as_ref() else { return PreviewScope::All };
        // Fall back only when the display mesh carries no `body:N:` provenance
        // at all. A scope that simply matches no body legitimately means
        // "nothing moves" and must not be widened to the whole mesh.
        let namespaced = if display.mesh.triangles.is_empty() {
            display.edges.iter().any(|edge| edge.id.starts_with("body:"))
        } else {
            display.face_ids.iter().any(|id| id.starts_with("body:"))
        };
        if namespaced {
            scope
        } else {
            PreviewScope::All
        }
    }

    fn draw_transform_gizmo(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(gizmo) = self.transform_gizmo.as_mut() else { return };
        let update = viewer_gizmo::show(
            ui,
            rect,
            &self.camera,
            self.lens_shift,
            display.radius,
            gizmo,
            &mut self.gizmo_state,
        );
        match self.gizmo_update.as_mut() {
            Some(pending) => {
                pending.changes.extend(update.changes);
                pending.started |= update.started;
                pending.dragging = update.dragging;
                pending.ended |= update.ended;
            }
            None => self.gizmo_update = Some(update),
        }
    }

    /// Draw the viewport while blocking navigation gestures over every floating
    /// panel, not just the node editor.
    fn sync_edit_ghost_theme(&mut self) {
        let color = crate::theme::get().ghost;
        if self.edit_ghost_color == color {
            return;
        }
        self.edit_ghost_color = color;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut()
            && self.edit_ghost_transform.is_some()
        {
            gpu.set_ghost_plans(
                self.edit_ghost_plans.clone(),
                self.edit_ghost_alpha,
                Some(ghost_plan_colour()),
                !self.edit_ghost_plans.is_empty(),
            );
        }
        self.dirty = true;
    }

    pub fn show_with_backdrop_blockers(
        &mut self,
        ui: &mut egui::Ui,
        overlay: egui::Rect,
        blockers: &[egui::Rect],
        backdrop_rect: egui::Rect,
    ) {
        self.poll_display_prep();
        if self.show_gpu_info {
            self.tick_frame_stats(ui);
        }
        let available = ui.available_size();
        let width = available.x.max(32.0) as usize;
        let height = available.y.max(32.0) as usize;
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let visible = egui::Rect::from_min_max(
            egui::pos2(overlay.max.x.clamp(rect.min.x, rect.max.x), rect.min.y),
            rect.max,
        );
        self.last_full = Some(rect);
        self.last_visible = Some(visible);
        self.apply_pending_view_restore(rect, visible);
        self.chrome_blockers.clear();

        let modifiers = ui.ctx().input(|input| input.modifiers);
        let middle = response.dragged_by(egui::PointerButton::Middle);
        let secondary = response.dragged_by(egui::PointerButton::Secondary);
        let orbit_drag = self.view.cad_mouse && middle && !modifiers.shift;
        let pan_drag = if self.view.cad_mouse {
            (middle && modifiers.shift) || secondary
        } else {
            middle || secondary
        };
        if orbit_drag || pan_drag {
            let delta = response.drag_delta();
            if orbit_drag {
                self.camera.orbit(
                    f64::from(delta.x) * f64::from(self.view.orbit_speed),
                    f64::from(delta.y) * f64::from(self.view.orbit_speed),
                );
            } else {
                self.camera.pan(
                    f64::from(delta.x) * f64::from(self.view.pan_speed),
                    f64::from(delta.y) * f64::from(self.view.pan_speed),
                    f64::from(rect.height()),
                );
            }
            self.dirty = true;
        }
        let gesture = nodecraft_node_canvas::gesture::TrackpadGesture::read(ui.ctx());
        let pointer_over_overlay = gesture.pointer.is_some_and(|pointer| {
            let mut blocked = blockers.to_vec();
            blocked.push(overlay);
            HoverMap::new(rect, &blocked, Some(pointer)).over_blocker()
        });
        if response.hovered() && !pointer_over_overlay && gesture.has_navigation() {
            if (gesture.zoom - 1.0).abs() > f32::EPSILON {
                // egui's zoom_delta grows when pinching out; camera distance shrinks.
                self.camera.zoom_at(f64::from(1.0 / gesture.zoom), None);
            } else if gesture.wheel != egui::Vec2::ZERO {
                // Mouse wheel: preserve the existing zoom behaviour.
                self.camera
                    .zoom_at(f64::from((-gesture.wheel.y * 0.0015).exp()), None);
            } else if gesture.pan != egui::Vec2::ZERO {
                // Latch the mode for the whole gesture. macOS scroll momentum can
                // keep delivering pan deltas after the Shift key-up; using the
                // current modifier per frame would turn the last part of an orbit
                // into a pivot-moving pan.
                let orbit = *self.trackpad_orbit.get_or_insert(gesture.shift);
                if orbit {
                    self.camera.orbit(
                        f64::from(gesture.pan.x) * f64::from(self.view.orbit_speed),
                        f64::from(gesture.pan.y) * f64::from(self.view.orbit_speed),
                    );
                } else {
                    self.camera.pan(
                        f64::from(gesture.pan.x) * f64::from(self.view.pan_speed),
                        f64::from(gesture.pan.y) * f64::from(self.view.pan_speed),
                        f64::from(rect.height()),
                    );
                }
            }
            if gesture.pan == egui::Vec2::ZERO {
                self.trackpad_orbit = None;
            }
            self.dirty = true;
        }
        if !gesture.has_navigation() {
            self.trackpad_orbit = None;
        }

        self.update_display_lod();
        if self.plan_pending() {
            // A staged plan swap needs another frame to prepare the next plan.
            ui.ctx().request_repaint();
        }
        self.sync_edit_ghost_theme();
        let size = [width.min(1400), height.min(1000)];
        if !self.gpu_active()
            && (self.dirty || self.texture.is_none() || self.last_size != size)
        {
            if self.display.is_some() {
                self.render_texture(ui.ctx(), size);
            }
            self.last_size = size;
            self.dirty = false;
        }

        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_ref() {
            let pixels_per_point = ui.ctx().pixels_per_point();
            let bounds = *gpu.bounds();
            let frame = gpu.frame(
                &self.camera,
                self.lens_shift,
                rect.width(),
                rect.height(),
                pixels_per_point,
                &bounds,
                {
                    let x = (backdrop_rect.min.x - rect.min.x) / rect.width().max(1.0);
                    let y = (backdrop_rect.min.y - rect.min.y) / rect.height().max(1.0);
                    let w = backdrop_rect.width() / rect.width().max(1.0);
                    let h = backdrop_rect.height() / rect.height().max(1.0);
                    // Clip the blur source to the glass panels: no outside geometry
                    // can bleed into them before it enters the panel.
                    [x, y, w, h]
                },
                {
                    let theme = crate::theme::get();
                    let radius = (theme.backdrop_blur * 0.5).clamp(0.0, 64.0);
                    [
                        radius,
                        radius,
                        crate::gpu_backdrop::BLUR_WIDTH as f32,
                        crate::gpu_backdrop::BLUR_HEIGHT as f32,
                    ]
                },
            );
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, frame));
            if self.chrome_visible {
                self.axis_gizmo(ui, rect);
                // Top selection dock intentionally disabled for the inspector-driven
                // selection experiment. Keep `selection_dock` for a future opt-in.
                self.pick_face(ui, rect, response.clicked(), pointer_over_overlay);
                self.update_edge_selection(ui, rect, response.clicked(), pointer_over_overlay);
                self.draw_face_hover(ui, rect);
                self.draw_comparison_overlay(ui, rect);
                // The feature-input edge field is a quiet guide; selected edges
                // are painted fully opaque afterwards.
                if !self.gpu_active() {
                    self.draw_edit_ghost_edges(ui, rect);
                }
                self.draw_edge_highlights(ui, rect);
                self.draw_comparison_edges(ui, rect);
                self.draw_transform_gizmo(ui, rect);
            }
            self.draw_debug_overlays(ui, rect);
            return;
        }

        if let Some(texture) = &self.texture {
            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Open a NodeCraft document to evaluate a mesh",
                egui::FontId::proportional(16.0),
                ui.visuals().weak_text_color(),
            );
        }
        self.draw_debug_overlays(ui, rect);
        if self.chrome_visible {
            self.axis_gizmo(ui, rect);
            // Top selection dock intentionally disabled; see inspector edit action.
            self.pick_face(ui, rect, response.clicked(), pointer_over_overlay);
            self.update_edge_selection(ui, rect, response.clicked(), pointer_over_overlay);
            self.draw_face_hover(ui, rect);
            self.draw_comparison_overlay(ui, rect);
            self.draw_edge_highlights(ui, rect);
            self.draw_comparison_edges(ui, rect);
            self.draw_transform_gizmo(ui, rect);
        }
    }

    /// Update the hovered edge and toggle the selected edge on click.
    fn update_edge_selection(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        clicked: bool,
        blocked: bool,
    ) {
        let pointer = ui.ctx().pointer_hover_pos();
        if blocked
            || self.dock_rect.is_some_and(|dock| pointer.is_some_and(|pos| dock.contains(pos)))
            || !pointer.is_some_and(|pos| viewport.contains(pos))
            || self.selection_mode != SelectionMode::Edge
        {
            if self.hovered_edge.take().is_some() { ui.ctx().request_repaint(); }
            return;
        }
        let Some(pointer) = pointer else { return };
        let nearest = self.nearest_edge(pointer, viewport, 7.0);
        let nearest_id = nearest.as_ref().map(|(id, _)| id.clone());
        if nearest_id != self.hovered_edge {
            self.hovered_edge = nearest_id;
            ui.ctx().request_repaint();
        }
        if clicked {
            let Some((edge_id, center)) = nearest else {
                if !self.pointer_hits_body(pointer, viewport) {
                    self.selection_miss = true;
                    ui.ctx().request_repaint();
                }
                return;
            };
            if !self.selected_edges.remove(&edge_id) {
                self.selected_edges.insert(edge_id.clone());
            }
            self.edge_picks.push(EdgePick { id: edge_id, center });
            ui.ctx().request_repaint();
        }
    }

    fn nearest_edge(&self, pointer: egui::Pos2, viewport: egui::Rect, threshold: f32) -> Option<(String, [f64; 3])> {
        if !self.edge_picking_enabled {
            return None;
        }
        let display = self.display.as_ref()?;
        let mut best = threshold * threshold;
        let mut nearest = None;
        if !self.picking_input_ghost_only() {
            for edge in &display.edges {
                // Preview treatment edges have no exact BRep id and cannot be used as
                // feature selections.
                if edge.id.starts_with("preview-edge:") {
                    continue;
                }
                let points: Vec<egui::Pos2> = edge
                    .points
                    .iter()
                    .filter_map(|point| self.project_to_viewport_body(*point, viewport, &edge.id))
                    .collect();
                for pair in points.windows(2) {
                    let distance = distance_to_segment(pointer, pair[0], pair[1]);
                    if distance <= best {
                        best = distance;
                        nearest = Some((edge.id.clone(), edge.midpoint));
                    }
                }
            }
        }
        // While editing a feature, the pickable layer is the transparent input
        // ghost, not the downstream breakpoint base. Its ids/centres are in the
        // feature's own frame, so they can be stored without inverse mapping.
        if self.edit_ghost_transform.is_some() {
            for edge in &self.edit_ghost_edges {
                let points: Vec<egui::Pos2> = edge
                    .points
                    .iter()
                    .filter_map(|point| self.project_to_viewport_ghost(*point, viewport))
                    .collect();
                for pair in points.windows(2) {
                    let distance = distance_to_segment(pointer, pair[0], pair[1]);
                    if distance <= best {
                        best = distance;
                        nearest = Some((edge.id.clone(), edge.midpoint));
                    }
                }
            }
        }
        nearest
    }

    fn project_to_viewport(&self, point: solidcore_record::Vec3, viewport: egui::Rect) -> Option<egui::Pos2> {
        // Unclassified overlay geometry follows the preview only when the whole
        // display is downstream of the edit.
        self.project_to_viewport_body(point, viewport, "")
    }

    /// Project an overlay point belonging to `body_id`, applying the preview
    /// transform only when that body moves. Mesh-derived overlays (edge
    /// highlights, face hover) use this so they cannot drift from the shaded
    /// solid during a partial-body drag.
    fn project_to_viewport_body(
        &self,
        point: solidcore_record::Vec3,
        viewport: egui::Rect,
        body_id: &str,
    ) -> Option<egui::Pos2> {
        let display = self.display.as_ref()?;
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let transform = self
            .preview_transform
            .filter(|_| self.preview_scope.is_affected(body_id));
        let point = transform.map_or(point, |matrix| {
            let m = matrix.map(|column| column.map(f64::from));
            solidcore_record::Vec3::new(
                m[0][0] * point.x + m[1][0] * point.y + m[2][0] * point.z + m[3][0],
                m[0][1] * point.x + m[1][1] * point.y + m[2][1] * point.z + m[3][1],
                m[0][2] * point.x + m[1][2] * point.y + m[2][2] * point.z + m[3][2],
            )
        });
        let (x, y, _) = self.camera.to_screen(point, width, height, display.radius)?;
        Some(egui::pos2(
            viewport.min.x + x as f32 + self.lens_shift[0] * viewport.width() * 0.5,
            viewport.min.y + y as f32 - self.lens_shift[1] * viewport.height() * 0.5,
        ))
    }

    /// Debug overlays stack from the top-left: GPU info first, then the camera /
    /// view-matrix block underneath it. Both are paint-only, so they never take
    /// pointer input from the viewport.
    fn debug_overlays_enabled(&self) -> bool {
        self.show_gpu_info || self.show_view_matrix
    }

    fn draw_debug_overlays(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        if !self.debug_overlays_enabled() {
            return;
        }
        let mut pos = viewport.min + egui::vec2(12.0, 12.0);
        if self.show_gpu_info {
            let lines = self.gpu_info_lines(viewport);
            let height = draw_debug_box(ui, pos, 365.0, &lines);
            pos.y += height + 8.0;
        }
        if self.show_view_matrix {
            let camera = &self.camera;
            let (right, up, back) = camera.basis();
            let view = camera.view().0;
            let target = camera.target;
            let eye = camera.eye();
            let lines = [
                format!(
                    "camera yaw {:.2}  pitch {:.2}  roll {:.2}",
                    camera.yaw.to_degrees(), camera.pitch.to_degrees(), camera.roll.to_degrees()
                ),
                format!("fov {:.3}  distance {:.3}  half_h {:.3}", camera.fov, camera.distance, camera.half_height()),
                format!("target {:.3} {:.3} {:.3}", target.x, target.y, target.z),
                format!("eye    {:.3} {:.3} {:.3}", eye.x, eye.y, eye.z),
                format!("right  {:.3} {:.3} {:.3}", right.x, right.y, right.z),
                format!("up     {:.3} {:.3} {:.3}", up.x, up.y, up.z),
                format!("back   {:.3} {:.3} {:.3}", back.x, back.y, back.z),
                format!("lens   {:.4} {:.4}", self.lens_shift[0], self.lens_shift[1]),
                "view matrix".to_owned(),
            ]
            .into_iter()
            .chain(view.iter().map(|row| {
                format!("{:.3} {:.3} {:.3} {:.3}", row[0], row[1], row[2], row[3])
            }))
            .collect::<Vec<_>>();
            draw_debug_box(ui, pos, 365.0, &lines);
        }
    }

    fn gpu_info_lines(&self, viewport: egui::Rect) -> Vec<String> {
        let fps = if self.frame_ms > 0.0 { 1000.0 / self.frame_ms } else { 0.0 };
        let mut lines = vec![format!(
            "frame {:.1} ms  {:.0} fps  {}x{}",
            self.frame_ms,
            fps,
            viewport.width().round() as i32,
            viewport.height().round() as i32
        )];
        #[cfg(feature = "gpu-viewport")]
        {
            if let Some(gpu) = self.gpu.as_ref().and_then(|gpu| gpu.debug_info()) {
                lines.push(format!(
                    "gpu {} ({}, {})",
                    gpu.adapter, gpu.backend, gpu.device_type
                ));
                lines.push(format!(
                    "object {} tris  {} chunks  {:.1} MB",
                    gpu.triangles,
                    gpu.chunks,
                    gpu.object_bytes as f64 / (1024.0 * 1024.0)
                ));
                lines.push(format!("msaa {}x", crate::gpu_viewport::msaa_samples()));
            } else {
                lines.push("gpu viewport not initialized".to_owned());
            }
        }
        #[cfg(not(feature = "gpu-viewport"))]
        lines.push("gpu viewport feature off".to_owned());
        if !self.tessellation_diagnostics.is_empty() {
            lines.push(format!(
                "tessellation warnings: {}",
                self.tessellation_diagnostics.len()
            ));
            for diagnostic in self.tessellation_diagnostics.iter().take(6) {
                lines.push(format!("  {diagnostic}"));
            }
        }
        lines
    }

    fn draw_face_hover(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(hovered) = self.hovered_face.as_deref() else { return };
        let theme = crate::theme::get();
        let colour = egui::Color32::from_rgba_unmultiplied(
            theme.highlight[0], theme.highlight[1], theme.highlight[2], 90,
        );
        let painter = ui.painter().with_clip_rect(viewport);
        for (index, triangle) in display.mesh.triangles.iter().enumerate() {
            let face = display.mesh.tri_face.get(index).copied().unwrap_or(u32::MAX) as usize;
            if display.face_ids.get(face).map(String::as_str) != Some(hovered) {
                continue;
            }
            let body = display.face_ids.get(face).map(String::as_str).unwrap_or("");
            let points: Vec<egui::Pos2> = triangle
                .iter()
                .filter_map(|corner| {
                    let point = display.mesh.positions.get(*corner as usize)?;
                    self.project_to_viewport_body(*point, viewport, body)
                })
                .collect();
            if points.len() != 3 { continue; }
            let mut mesh = egui::Mesh::default();
            for point in points {
                mesh.colored_vertex(point, colour);
            }
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));
        }
    }

    fn draw_comparison_overlay(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(comparison) = &display.comparison else { return };
        let theme = crate::theme::get();
        let painter = ui.painter().with_clip_rect(viewport);
        let draw_layer = |triangles: &[([solidcore_record::Vec3; 3], [solidcore_record::Vec3; 3])], color: egui::Color32| {
            for (points, _) in triangles {
                let projected: Vec<egui::Pos2> = points
                    .iter()
                    .filter_map(|point| self.project_to_viewport(*point, viewport))
                    .collect();
                if projected.len() != 3 { continue; }
                let mut mesh = egui::Mesh::default();
                for point in projected {
                    mesh.colored_vertex(point, color);
                }
                mesh.add_triangle(0, 1, 2);
                painter.add(egui::Shape::mesh(mesh));
            }
        };
        // Display-only comparison layer. It is painted above the shaded result,
        // never enters the display mesh, and therefore never feeds face/edge/vertex
        // picking. 50% alpha keeps the result readable underneath.
        let overlay_alpha = if self.comparison_ghost { 14 } else { 38 };
        let add = theme.preview_add;
        draw_layer(&comparison.added, egui::Color32::from_rgba_unmultiplied(add[0], add[1], add[2], overlay_alpha));
        let cut = theme.preview_cut;
        draw_layer(&comparison.removed, egui::Color32::from_rgba_unmultiplied(cut[0], cut[1], cut[2], overlay_alpha));
    }

    fn draw_comparison_edges(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(comparison) = &display.comparison else { return };
        if comparison.edges.is_empty() { return; }
        let theme = crate::theme::get();
        let cut = theme.preview_cut;
        let painter = ui.painter().with_clip_rect(viewport);
        let edge_colour = egui::Color32::from_rgba_unmultiplied(
            cut[0],
            cut[1],
            cut[2],
            if self.comparison_ghost { 90 } else { 128 },
        );
        for edge in &comparison.edges {
            let points: Vec<egui::Pos2> = edge
                .points
                .iter()
                .filter_map(|point| self.project_to_viewport(*point, viewport))
                .collect();
            for pair in points.windows(2) {
                if pair[0].distance_sq(pair[1]) <= 1.0e-6 {
                    continue;
                }
                painter.line_segment([pair[0], pair[1]], egui::Stroke::new(2.0, edge_colour));
            }
        }
    }

    /// Project a point in the edit ghost's branch frame into the viewport.
    fn project_to_viewport_ghost(
        &self,
        point: solidcore_record::Vec3,
        viewport: egui::Rect,
    ) -> Option<egui::Pos2> {
        let matrix = self.edit_ghost_transform?;
        let x = f64::from(matrix[0][0]) * point.x
            + f64::from(matrix[1][0]) * point.y
            + f64::from(matrix[2][0]) * point.z
            + f64::from(matrix[3][0]);
        let y = f64::from(matrix[0][1]) * point.x
            + f64::from(matrix[1][1]) * point.y
            + f64::from(matrix[2][1]) * point.z
            + f64::from(matrix[3][1]);
        let z = f64::from(matrix[0][2]) * point.x
            + f64::from(matrix[1][2]) * point.y
            + f64::from(matrix[2][2]) * point.z
            + f64::from(matrix[3][2]);
        self.project_to_viewport(
            solidcore_record::Vec3::new(x, y, z),
            viewport,
        )
    }

    fn draw_edit_ghost_edges(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(matrix) = self.edit_ghost_transform else { return };
        if self.edit_ghost_edges.is_empty() { return; }
        let painter = ui.painter().with_clip_rect(viewport);
        let [r, g, b, a] = crate::theme::get().ghost_edge;
        // Edge selection is a quiet guide until an edge is picked; selected
        // edges are drawn opaque by `draw_edge_highlights`.
        let colour = egui::Color32::from_rgba_unmultiplied(r, g, b, a / 3);
        for edge in &self.edit_ghost_edges {
            let projected: Vec<egui::Pos2> = edge
                .points
                .iter()
                .filter_map(|point| {
                    let x = f64::from(matrix[0][0]) * point.x
                        + f64::from(matrix[1][0]) * point.y
                        + f64::from(matrix[2][0]) * point.z
                        + f64::from(matrix[3][0]);
                    let y = f64::from(matrix[0][1]) * point.x
                        + f64::from(matrix[1][1]) * point.y
                        + f64::from(matrix[2][1]) * point.z
                        + f64::from(matrix[3][1]);
                    let z = f64::from(matrix[0][2]) * point.x
                        + f64::from(matrix[1][2]) * point.y
                        + f64::from(matrix[2][2]) * point.z
                        + f64::from(matrix[3][2]);
                    self.project_to_viewport(
                        solidcore_record::Vec3::new(x, y, z),
                        viewport,
                    )
                })
                .collect();
            for pair in projected.windows(2) {
                painter.line_segment([pair[0], pair[1]], egui::Stroke::new(1.6, colour));
            }
        }
    }

    /// Draw body-scoped edge ids over the rendered viewport.
    fn draw_edge_highlights(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let theme = crate::theme::get();
        let painter = ui.painter().with_clip_rect(viewport);
        let draw = |id: &str, colour: egui::Color32, width: f32| {
            let (edge, ghost) = match display.edges.iter().find(|edge| edge.id == id) {
                Some(edge) => (edge, false),
                None => {
                    let Some(edge) = self.edit_ghost_edges.iter().find(|edge| edge.id == id) else {
                        return;
                    };
                    (edge, true)
                }
            };
            let points: Vec<egui::Pos2> = if ghost {
                edge.points
                    .iter()
                    .filter_map(|point| self.project_to_viewport_ghost(*point, viewport))
                    .collect()
            } else {
                edge.points
                    .iter()
                    .filter_map(|point| self.project_to_viewport_body(*point, viewport, id))
                    .collect()
            };
            for pair in points.windows(2) {
                if pair[0].distance_sq(pair[1]) <= 1.0e-6 {
                    continue;
                }
                painter.line_segment([pair[0], pair[1]], egui::Stroke::new(width, colour));
            }
        };
        let selected_colour =
            egui::Color32::from_rgb(theme.accent[0], theme.accent[1], theme.accent[2]);
        for id in &self.selected_edges {
            draw(id, selected_colour, 3.0);
        }
        if self.selection_mode == SelectionMode::Edge
            && let Some(id) = &self.hovered_edge
            && !self.selected_edges.contains(id)
        {
            draw(id, theme.color(theme.highlight), 2.5);
        }
    }

    /// Pick the nearest display-mesh face under the pointer in face mode.
    fn pointer_hits_body(&self, pointer: egui::Pos2, viewport: egui::Rect) -> bool {
        let Some(display) = self.display.as_ref() else { return false };
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let x = f64::from(pointer.x - viewport.min.x) - f64::from(self.lens_shift[0]) * width * 0.5;
        let y = f64::from(pointer.y - viewport.min.y) + f64::from(self.lens_shift[1]) * height * 0.5;
        let (origin, direction) = self.camera.ray(x, y, width, height);
        raycast_face(&display.mesh, display.face_ids.as_slice(), origin, direction).is_some()
    }

    fn pick_face(&mut self, ui: &mut egui::Ui, viewport: egui::Rect, clicked: bool, blocked: bool) {
        if blocked || self.selection_mode != SelectionMode::Face {
            if self.hovered_face.take().is_some() {
                ui.ctx().request_repaint();
            }
            return;
        }
        let Some(pointer) = ui.ctx().pointer_hover_pos() else {
            self.hovered_face = None;
            return;
        };
        if !viewport.contains(pointer) || self.dock_rect.is_some_and(|dock| dock.contains(pointer)) {
            self.hovered_face = None;
            return;
        }
        let Some(display) = self.display.as_ref() else { return };
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let x = f64::from(pointer.x - viewport.min.x) - f64::from(self.lens_shift[0]) * width * 0.5;
        let y = f64::from(pointer.y - viewport.min.y) + f64::from(self.lens_shift[1]) * height * 0.5;
        let (origin, direction) = self.camera.ray(x, y, width, height);
        let plan_ids = self.plan_face_ids();
        let face_ids = plan_ids.as_deref().unwrap_or(display.face_ids.as_slice());
        let hovered = raycast_face(&display.mesh, face_ids, origin, direction);
        if hovered != self.hovered_face {
            self.hovered_face = hovered.clone();
            ui.ctx().request_repaint();
        }
        if clicked {
            let Some(face) = hovered else {
                self.selection_miss = true;
                ui.ctx().request_repaint();
                return;
            };
            self.face_picks.push(face);
            ui.ctx().request_repaint();
        }
    }

    /// SolidCraft-style signed XYZ compass in the viewport's top-right corner.
    /// Clicking a positive tip chooses that standard view; dragging the compass orbits.
    fn axis_gizmo(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let theme = crate::theme::get();
        const SIZE: f32 = 94.0;
        const MARGIN: f32 = 14.0;
        const ART: f32 = 0.84;
        const AXIS_LENGTH: f32 = 38.0;

        let rect = egui::Rect::from_min_size(
            egui::Pos2::new(
                viewport.right() - SIZE - MARGIN - self.chrome_right_inset,
                viewport.top() + MARGIN,
            ),
            egui::Vec2::splat(SIZE),
        );
        // Click a tip to choose a standard view, or drag the compass to orbit.
        let response = ui.interact(
            rect,
            egui::Id::new("viewport-axis-gizmo"),
            egui::Sense::click_and_drag(),
        );
        let center = rect.center();
        let (right, up, back) = self.camera.basis();
        let axes = [
            ("x", solidcore_record::Vec3::X),
            ("-x", -solidcore_record::Vec3::X),
            ("y", solidcore_record::Vec3::Y),
            ("-y", -solidcore_record::Vec3::Y),
            ("z", solidcore_record::Vec3::Z),
            ("-z", -solidcore_record::Vec3::Z),
        ];
        let mut tips: Vec<(&'static str, egui::Pos2, f32)> = axes
            .iter()
            .map(|(name, direction)| {
                let dir = egui::vec2(
                    direction.dot(right) as f32,
                    -(direction.dot(up) as f32),
                ) * AXIS_LENGTH
                    * ART;
                (*name, center + dir, direction.dot(back) as f32)
            })
            .collect();
        tips.sort_by(|a, b| a.2.total_cmp(&b.2));

        let hover = ui.input(|input| input.pointer.hover_pos()).filter(|pointer| rect.contains(*pointer));
        let hovered = hover.and_then(|pointer| {
            tips.iter().enumerate().rev().find(|(_, (_, tip, _))| tip.distance(pointer) <= 10.0).map(|(index, _)| index)
        });
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            if delta != egui::Vec2::ZERO {
                self.camera.orbit(f64::from(delta.x), f64::from(delta.y));
                self.dirty = true;
                ui.ctx().request_repaint();
            }
        }
        if response.hovered() {
            ui.ctx().set_cursor_icon(if response.dragged() {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::PointingHand
            });
        }
        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                if let Some((name, _, _)) = tips.iter().rev().find(|(_, tip, _)| tip.distance(pointer) <= 12.0) {
                    let view = match *name {
                        "x" => StandardView::Right,
                        "-x" => StandardView::Left,
                        "y" => StandardView::Back,
                        "-y" => StandardView::Front,
                        "z" => StandardView::Top,
                        _ => StandardView::Bottom,
                    };
                    self.set_standard_view(view);
                    ui.ctx().request_repaint();
                }
            }
        }

        let painter = ui.painter().with_clip_rect(viewport);
        painter.circle_filled(center, SIZE * 0.5, theme.with_alpha(theme.window, 150));
        painter.circle_stroke(center, SIZE * 0.5, egui::Stroke::new(1.0, theme.with_alpha(theme.border, 180)));
        for (index, (name, at, _depth)) in tips.iter().enumerate() {
            let positive = !name.starts_with('-');
            let colour = match name.trim_start_matches('-') {
                "x" => theme.color(theme.axis_x),
                "y" => theme.color(theme.axis_y),
                _ => theme.color(theme.axis_z),
            };
            let colour = if positive { colour } else { colour.gamma_multiply(0.45) };
            painter.line_segment([center, *at], egui::Stroke::new(1.6, colour.gamma_multiply(0.8)));
            let radius = if hovered == Some(index) { 10.0 } else { 8.0 };
            painter.circle_filled(*at, radius, colour);
            painter.circle_stroke(*at, radius, egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180)));
            if positive {
                painter.text(*at, egui::Align2::CENTER_CENTER, name.to_uppercase(), egui::FontId::proportional(8.0), theme.color(theme.window));
            }
        }
        painter.circle_filled(center, 3.5, egui::Color32::from_rgb(200, 200, 204));

        // Home / isometric button directly under the compass. Its right edge
        // shares the gizmo's viewport padding; chrome appears only on hover.
        let icon_size = 18.0;
        let icon_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.right() - icon_size, rect.bottom() + 10.0),
            egui::Vec2::splat(icon_size),
        );
        let home_rect = egui::Rect::from_center_size(icon_rect.center(), egui::Vec2::splat(30.0));
        let home = ui
            .interact(home_rect, egui::Id::new("viewport-axis-home"), egui::Sense::click())
            .on_hover_text("Home / isometric view");
        if home.hovered() {
            painter.circle_filled(
                icon_rect.center(),
                14.0,
                theme.with_alpha(theme.highlight, 26),
            );
            painter.circle_stroke(
                icon_rect.center(),
                14.0,
                egui::Stroke::new(1.0, theme.with_alpha(theme.muted, 100)),
            );
        }
        if home.hovered() || home.clicked() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if home.clicked() {
            self.home();
            ui.ctx().request_repaint();
        }
        let icon_colour = if home.hovered() {
            crate::theme::Theme::hex(theme.icon_hover)
        } else {
            crate::theme::Theme::hex(theme.icon)
        };
        let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/home.svg"))
            .unwrap_or_default()
            .replace("#d8d8dc", &icon_colour);
        let image = egui::Image::from_bytes("bytes://nodecraft-home.svg", svg.into_bytes())
            .fit_to_exact_size(egui::Vec2::splat(icon_size));
        ui.put(icon_rect, image);

        // Projection toggle directly below the home button.
        let perspective_icon = egui::Rect::from_min_size(
            egui::Pos2::new(icon_rect.right() - icon_size, home_rect.bottom() + 10.0),
            egui::Vec2::splat(icon_size),
        );
        let perspective_rect = egui::Rect::from_center_size(
            perspective_icon.center(),
            egui::Vec2::splat(30.0),
        );
        let perspective = ui
            .interact(
                perspective_rect,
                egui::Id::new("viewport-perspective-toggle"),
                egui::Sense::click(),
            )
            .on_hover_text(if self.camera.fov > 0.0 {
                "Switch to orthographic"
            } else {
                "Switch to perspective"
            });
        if perspective.hovered() {
            painter.circle_filled(
                perspective_icon.center(),
                14.0,
                theme.with_alpha(theme.highlight, 26),
            );
            painter.circle_stroke(
                perspective_icon.center(),
                14.0,
                egui::Stroke::new(1.0, theme.with_alpha(theme.muted, 100)),
            );
        }
        if perspective.hovered() || perspective.clicked() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if perspective.clicked() {
            self.toggle_projection();
            ui.ctx().request_repaint();
        }
        let perspective_colour = if self.camera.fov > 0.0 {
            crate::theme::Theme::hex(theme.accent)
        } else if perspective.hovered() {
            crate::theme::Theme::hex(theme.icon_hover)
        } else {
            crate::theme::Theme::hex(theme.icon)
        };
        let (icon_source, icon_uri) = projection_icon(self.camera.fov > 0.0);
        let svg = std::str::from_utf8(icon_source)
            .unwrap_or_default()
            .replace("#d8d8dc", &perspective_colour);
        let image = egui::Image::from_bytes(icon_uri, svg.into_bytes())
            .fit_to_exact_size(egui::Vec2::splat(icon_size));
        ui.put(perspective_icon, image);

        // Display-mode list below the projection toggle.
        let view_icon = egui::Rect::from_min_size(
            egui::Pos2::new(perspective_icon.right() - icon_size, perspective_rect.bottom() + 10.0),
            egui::Vec2::splat(icon_size),
        );
        let view_rect = egui::Rect::from_center_size(view_icon.center(), egui::Vec2::splat(30.0));
        let view = ui
            .interact(view_rect, egui::Id::new("viewport-view-modes"), egui::Sense::click())
            .on_hover_text("Display modes");
        if view.hovered() {
            painter.circle_filled(view_icon.center(), 14.0, theme.with_alpha(theme.highlight, 26));
            painter.circle_stroke(view_icon.center(), 14.0, egui::Stroke::new(1.0, theme.with_alpha(theme.muted, 100)));
        }
        if view.hovered() || view.clicked() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        let view_colour = if view.hovered() {
            crate::theme::Theme::hex(theme.icon_hover)
        } else if self.view.any_extra() {
            crate::theme::Theme::hex(theme.accent)
        } else {
            crate::theme::Theme::hex(theme.icon)
        };
        let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/settings.svg"))
            .unwrap_or_default()
            .replace("#d8d8dc", &view_colour);
        let image = egui::Image::from_bytes("bytes://nodecraft-view-modes.svg", svg.into_bytes())
            .fit_to_exact_size(egui::Vec2::splat(icon_size));
        ui.put(view_icon, image);
        self.show_view_settings_menu(ui, &view);
        self.chrome_blockers = vec![rect, home_rect, perspective_rect, view_rect];
    }

    fn settings_popup() -> f32 {
        300.0
    }

    /// Toggle rows reserve this much for their checkmark; section headers and
    /// slider labels use the same inset so the popup has one alignment column.
    const SETTINGS_LABEL_INSET: f32 = 28.0;
    /// Slider handle radius is derived from the allocated slider thickness
    /// (`max(body text height, interact_size.y)`), so `add_sized` cannot shrink
    /// it. Give these viewport sliders a smaller body text style and interaction
    /// height while the slider is added, then restore the menu style.
    const SETTINGS_SLIDER_HEIGHT: f32 = 18.0;
    const SETTINGS_SLIDER_FONT_SIZE: f32 = 11.0;

    fn settings_section(ui: &mut egui::Ui, title: &str) {
        let theme = crate::theme::get();
        ui.add_space(Self::SETTINGS_LABEL_INSET);
        ui.label(
            egui::RichText::new(title.to_uppercase())
                .size(9.0)
                .strong()
                .color(theme.with_alpha(theme.muted, 190)),
        );
        ui.add_space(1.0);
    }

    fn settings_slider(
        ui: &mut egui::Ui,
        label: &str,
        value: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        decimals: usize,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.add_space(Self::SETTINGS_LABEL_INSET);
            ui.label(egui::RichText::new(label).size(12.0));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_sized(
                    [34.0, 18.0],
                    egui::Label::new(
                        egui::RichText::new(format!("{value:.decimals$}"))
                            .size(11.0)
                            .monospace(),
                    ),
                );
                ui.spacing_mut().slider_width = 118.0;
                let previous_style = (**ui.style()).clone();
                ui.style_mut().text_styles.insert(
                    egui::TextStyle::Body,
                    egui::FontId::proportional(Self::SETTINGS_SLIDER_FONT_SIZE),
                );
                ui.spacing_mut().interact_size.y = Self::SETTINGS_SLIDER_HEIGHT;
                changed = ui
                    .add(egui::Slider::new(value, range).show_value(false))
                    .changed();
                *ui.style_mut() = previous_style;
            });
        });
        changed
    }

    /// Shared display-settings popup, used by the viewport cog and the top bar.
    pub fn show_view_settings_menu(&mut self, _ui: &mut egui::Ui, response: &egui::Response) {
        let mut edited = self.view;
        let mut changed = false;
        egui::Popup::menu(response)
            // Anchor the right edge to the viewport settings button instead of
            // letting a wide menu grow leftward under egui's default BOTTOM_START.
            .align(egui::RectAlign::BOTTOM_END)
            // Match the viewport panel inset so the popup sits inside the 3D
            // viewport area rather than tight against the settings button.
            .gap(12.0)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .style(crate::theme::menu_style)
            .show(|ui| {
                // A fixed column avoids popup auto-resize jitter while a slider
                // drag changes its formatted value text.
                let width = Self::settings_popup();
                ui.set_min_width(width);
                ui.set_max_width(width);
                ui.set_width(width);
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);

                Self::settings_section(ui, "Display");
                changed |= crate::theme::menu_toggle(ui, &mut edited.translucent, "Translucent");
                changed |= crate::theme::menu_toggle(ui, &mut edited.edges, "Edges");
                changed |= crate::theme::menu_toggle(ui, &mut edited.triangles, "Triangles");
                changed |= crate::theme::menu_toggle(ui, &mut edited.normals, "Face normals");
                changed |= crate::theme::menu_toggle(ui, &mut edited.face_colours, "Face colours");
                changed |= crate::theme::menu_toggle(ui, &mut edited.zebra, "Zebra analysis");
                changed |= crate::theme::menu_toggle(
                    ui,
                    &mut edited.ignore_look_colors,
                    "Ignore Look colours",
                );
                changed |= Self::settings_slider(
                    ui,
                    "Ghost opacity",
                    &mut edited.ghost_opacity,
                    0.0..=1.0,
                    2,
                );

                Self::settings_section(ui, "Material");
                changed |= Self::settings_slider(
                    ui,
                    "Roughness",
                    &mut edited.roughness,
                    0.045..=1.0,
                    2,
                );
                changed |= Self::settings_slider(
                    ui,
                    "Metallic",
                    &mut edited.metallic,
                    0.0..=1.0,
                    2,
                );
                ui.add_space(6.0);
            });
        if changed {
            self.set_view(edited);
        }
    }

    pub fn backdrop_view(&self) -> Option<(egui::TextureId, egui::Rect)> {
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = &self.gpu {
            let uv = egui::Rect::from_min_max(
                egui::pos2(crate::gpu_backdrop::INNER_UV_MIN[0], crate::gpu_backdrop::INNER_UV_MIN[1]),
                egui::pos2(crate::gpu_backdrop::INNER_UV_MAX[0], crate::gpu_backdrop::INNER_UV_MAX[1]),
            );
            return Some((gpu.backdrop_texture_id(), uv));
        }
        None
    }

    pub fn backdrop_view_for(
        &self,
        rect: egui::Rect,
        backdrop: egui::Rect,
    ) -> Option<(egui::TextureId, egui::Rect)> {
        let (texture, full) = self.backdrop_view()?;
        let width = backdrop.width().max(1.0);
        let height = backdrop.height().max(1.0);
        let u0 = ((rect.min.x - backdrop.min.x) / width).clamp(0.0, 1.0);
        let v0 = ((rect.min.y - backdrop.min.y) / height).clamp(0.0, 1.0);
        let u1 = ((rect.max.x - backdrop.min.x) / width).clamp(0.0, 1.0);
        let v1 = ((rect.max.y - backdrop.min.y) / height).clamp(0.0, 1.0);
        Some((
            texture,
            egui::Rect::from_min_max(
                egui::pos2(full.min.x + full.width() * u0, full.min.y + full.height() * v0),
                egui::pos2(full.min.x + full.width() * u1, full.min.y + full.height() * v1),
            ),
        ))
    }

    fn render_texture(&mut self, ctx: &egui::Context, [width, height]: [usize; 2]) {
        let Some(display) = &self.display else {
            self.texture = None;
            return;
        };

        let scene = scene_for(display, &self.camera, self.lod_index);
        let canvas = solidcraft_viewport::render(&scene, &self.camera, width, height);
        let image = egui::ColorImage::from_rgba_unmultiplied([canvas.w, canvas.h], &canvas.rgba);
        self.texture = Some(ctx.load_texture(
            "nodecraft-viewport",
            image,
            egui::TextureOptions::LINEAR,
        ));
    }
}


fn add_grid(
    scene: &mut Scene,
    radius: f64,
    center: solidcore_record::Vec3,
    perspective: bool,
    ortho_fade: f32,
) {
    let theme = crate::theme::get();
    for line in grid_lines(radius) {
        let base_alpha = if perspective { line.alpha } else { ortho_fade };
        let (colour, alpha, width) = match line.kind {
            GridKind::Minor => (theme.grid_minor, 0.08 * base_alpha, 1.0),
            GridKind::Major => (theme.grid_major, 0.24 * base_alpha, 1.2),
            GridKind::AxisX => (theme.axis_x, 0.9 * base_alpha, 2.2),
            GridKind::AxisY => (theme.axis_y, 0.9 * base_alpha, 2.2),
            GridKind::AxisZ => (theme.axis_z, 0.9 * base_alpha, 2.2),
        };
        scene.grid.push(SceneLine {
            points: vec![
                solidcore_record::Vec3::new(
                    center.x + line.start[0] as f64,
                    center.y + line.start[1] as f64,
                    center.z + line.start[2] as f64,
                ),
                solidcore_record::Vec3::new(
                    center.x + line.end[0] as f64,
                    center.y + line.end[1] as f64,
                    center.z + line.end[2] as f64,
                ),
            ],
            color: Rgb(colour[0], colour[1], colour[2]),
            alpha,
            width,
            on_top: false,
        });
    }
}

fn scene_for(display: &DisplayMesh, camera: &Camera, lod_index: usize) -> Scene {
    let mut scene = Scene::default();
    // Keep model depth precision; do not push the far plane out for the grid.
    scene.radius = display.radius;
    // Match the sharp GPU viewport background and mesh colours so the blurred
    // backdrop is the same content, only defocused.
    let theme = crate::theme::get();
    let background = Rgb(
        theme.viewport_background[0],
        theme.viewport_background[1],
        theme.viewport_background[2],
    );
    scene.background = Some((background, background));
    let ortho_fade = if camera.fov > 0.0 {
        1.0
    } else {
        let angle = camera.back().z.abs() as f32;
        let t = ((angle - 0.04) / 0.18).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    add_grid(
        &mut scene,
        display.radius,
        VIEW_ORIGIN,
        camera.fov > 0.0,
        ortho_fade,
    );
    scene.meshes.push(SceneMesh::solid(
        std::sync::Arc::new(display.render_mesh(lod_index).clone()),
        Rgb(151, 162, 176),
    ));
    for edge in &display.mesh.edges {
        scene.lines.push(SceneLine {
            points: edge.clone(),
            color: Rgb(30, 35, 40),
            alpha: 0.95,
            width: 1.0,
            on_top: false,
        });
    }
    scene
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_info_overlay_is_opt_in() {
        let mut viewport = Viewport::default();
        assert!(!viewport.show_gpu_info());
        assert!(!viewport.show_view_matrix());
        assert!(!viewport.debug_overlays_enabled());
        viewport.set_show_gpu_info(true);
        assert!(viewport.debug_overlays_enabled());
    }

    #[test]
    fn home_frames_the_shared_world_origin() {
        let mut viewport = Viewport::default();
        viewport.camera.target = solidcore_record::Vec3::new(40.0, -25.0, 12.0);
        viewport.camera.distance = 500.0;
        viewport.lens_shift = [0.25, -0.15];
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 800.0));
        let visible = egui::Rect::from_min_max(egui::pos2(400.0, 0.0), full.max);
        let expected = lens_shift_for_visible(full, visible);
        viewport.frame_origin(30.0, f64::from(visible.width() / visible.height()), expected);
        assert_eq!(viewport.camera.target, VIEW_ORIGIN);
        assert_eq!(viewport.lens_shift, expected);
        assert!(expected[0] > 0.0, "visible origin should shift right of full centre");
        let (yaw, pitch) = StandardView::Iso.angles();
        assert_eq!(viewport.camera.yaw, -yaw);
        assert_eq!(viewport.camera.pitch, pitch);
        assert!(viewport.camera.distance > 0.0);
    }

    #[test]
    fn projection_toggle_swaps_between_lattice_icons() {
        let (perspective, perspective_uri) = projection_icon(true);
        let (orthographic, orthographic_uri) = projection_icon(false);
        assert_ne!(perspective, orthographic);
        assert_ne!(perspective_uri, orthographic_uri);
        assert!(std::str::from_utf8(perspective).unwrap().contains("M5.854 4H18.146"));
        assert!(std::str::from_utf8(orthographic).unwrap().contains("M3 3H21V21H3Z"));
    }

    #[test]
    fn projection_toggle_preserves_target_rotation_and_half_height() {
        let mut viewport = Viewport::default();
        viewport.camera.target = solidcore_record::Vec3::new(12.0, -4.0, 7.0);
        viewport.camera.yaw = 0.4;
        viewport.camera.pitch = 0.3;
        viewport.camera.roll = 0.2;
        viewport.camera.distance = 100.0;
        viewport.camera.fov = 0.0;
        let target = viewport.camera.target;
        let before = viewport.camera.half_height();
        viewport.toggle_projection();
        assert!(viewport.camera.fov > 0.0);
        assert_eq!(viewport.camera.target, target);
        assert_eq!(viewport.camera.yaw, 0.4);
        assert_eq!(viewport.camera.pitch, 0.3);
        assert_eq!(viewport.camera.roll, 0.2);
        assert!((viewport.camera.half_height() - before).abs() < 1e-9);

        let perspective_half = viewport.camera.half_height();
        viewport.toggle_projection();
        assert_eq!(viewport.camera.fov, 0.0);
        assert_eq!(viewport.camera.target, target);
        assert!((viewport.camera.half_height() - perspective_half).abs() < 1e-9);
    }

    #[test]
    fn zebra_is_an_active_display_override() {
        let mut view = ViewSettings::default();
        assert!(!view.any_extra());
        view.zebra = true;
        assert!(view.any_extra());
    }

    #[test]
    fn changing_the_display_mesh_clears_edge_highlights() {
        let mut viewport = Viewport::default();
        viewport.selected_edges.insert("edge:0".into());
        viewport.hovered_edge = Some("edge:1".into());
        viewport.set_mesh(Some(&NodeMesh::default()));
        assert!(viewport.selected_edges.is_empty());
        assert!(viewport.hovered_edge.is_none());
    }

    #[test]
    fn overlays_follow_the_preview_transform_and_retire_it_with_new_meshes() {
        let mut viewport = Viewport::default();
        viewport.set_mesh(Some(&NodeMesh::default()));
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let point = solidcore_record::Vec3::new(1.0, 0.5, 0.0);
        let shift = solidcore_record::Vec3::new(4.0, 0.0, 0.0);
        let shifted = viewport
            .project_to_viewport(point + shift, rect)
            .expect("shifted point projects");
        let mut preview = [[0.0f32; 4]; 4];
        for (axis, column) in preview.iter_mut().enumerate() {
            column[axis] = 1.0;
        }
        preview[3][0] = shift.x as f32;
        viewport.preview_transform = Some(preview);
        let previewed = viewport
            .project_to_viewport(point, rect)
            .expect("previewed point projects");
        // Drawing a previewed point must equal drawing the moved point, so mesh
        // overlays cannot drift away from the shaded solid during a drag.
        assert!((previewed - shifted).length() < 1.0e-3, "{previewed:?} {shifted:?}");
        // The kernel result is authoritative and retires the preview.
        viewport.set_mesh(Some(&NodeMesh::default()));
        assert!(viewport.preview_transform.is_none());
    }

    #[test]
    fn edge_pick_distance_is_screen_space() {
        let a = egui::pos2(0.0, 0.0);
        let b = egui::pos2(20.0, 0.0);
        assert!(distance_to_segment(egui::pos2(10.0, 3.0), a, b) - 3.0 < 1.0e-6);
        assert!(distance_to_segment(egui::pos2(-5.0, 0.0), a, b) - 5.0 < 1.0e-6);
        assert!(distance_to_segment(egui::pos2(25.0, 0.0), a, b) - 5.0 < 1.0e-6);
    }

    #[test]
    fn face_pick_returns_the_solidcore_face_group() {
        let mesh = solidcraft_geom::Mesh {
            positions: vec![
                solidcore_record::Vec3::new(-1.0, -1.0, 0.0),
                solidcore_record::Vec3::new(1.0, -1.0, 0.0),
                solidcore_record::Vec3::new(1.0, 1.0, 0.0),
            ],
            normals: vec![solidcore_record::Vec3::Z; 3],
            triangles: vec![[0, 1, 2]],
            tri_face: vec![0],
            edges: Vec::new(),
            display_only_edges: false,
            seams: Vec::new(),
            smooth_edges: Vec::new(),
            sharp_edges: Vec::new(),
            edge_faces: Vec::new(),
        };
        let picked = raycast_face(
            &mesh,
            &["body:0:face:9".to_owned()],
            solidcore_record::Vec3::new(0.0, 0.0, 5.0),
            solidcore_record::Vec3::new(0.0, 0.0, -1.0),
        );
        assert_eq!(picked.as_deref(), Some("body:0:face:9"));
    }

    #[test]
    fn mesh_update_can_preserve_the_camera() {
        let mut viewport = Viewport::default();
        viewport.camera.target = solidcore_record::Vec3::new(12.0, -4.0, 3.0);
        viewport.camera.distance = 73.0;
        viewport.camera.yaw = 0.6;
        viewport.camera.pitch = 0.3;
        let before = (
            viewport.camera.target,
            viewport.camera.distance,
            viewport.camera.yaw,
            viewport.camera.pitch,
        );
        viewport.set_mesh_preserving_camera(Some(&NodeMesh::default()));
        let after = (
            viewport.camera.target,
            viewport.camera.distance,
            viewport.camera.yaw,
            viewport.camera.pitch,
        );
        assert_eq!(before, after);
    }

    #[test]
    fn edit_overlay_scope_resolves_to_the_branch_body() {
        let mut mesh = crate::adapter::display_mesh(&nodecraft_protocol::Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 6],
            faces: vec![[0, 1, 2], [3, 4, 5]],
            face_groups: vec!["body:0:face:0".to_owned(), "body:1:face:0".to_owned()],
            ..nodecraft_protocol::Mesh::default()
        });
        mesh.mesh = std::sync::Arc::new((*mesh.mesh).clone());
        let mut viewport = Viewport::default();
        viewport.display = Some(mesh);
        assert_eq!(
            viewport.resolve_preview_scope(PreviewScope::Bodies(vec!["body:0:".to_owned()])),
            PreviewScope::Bodies(vec!["body:0:".to_owned()])
        );
    }

    #[test]
    fn clearing_a_document_evicts_plan_state() {
        let mut viewport = Viewport::default();
        viewport.plans = vec![nodecraft_display::PlanInstance {
            target: "old-project".into(),
            plan: std::sync::Arc::new(nodecraft_layers::tess::TessPlan::default()),
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(Vec::new()),
        }];
        viewport.set_feature_ghost_plans(vec![nodecraft_display::PlanInstance {
            target: "old-ghost".into(),
            plan: std::sync::Arc::new(nodecraft_layers::tess::TessPlan::default()),
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(Vec::new()),
        }]);
        viewport.clear_document_display();
        assert!(viewport.plans.is_empty(), "closed project kept its plans");
        assert!(viewport.feature_ghost_plans.is_empty(), "closed project kept ghost plans");
        assert!(viewport.display.is_none(), "closed project kept its display mesh");
    }

    #[test]
    fn face_picking_uses_plan_brep_face_ids() {
        let mut face = nodecraft_layers::tess::FacePlan::default();
        face.face = 7;
        face.uv = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        face.fixed = vec![
            solidcore_record::Vec3::ZERO,
            solidcore_record::Vec3::X,
            solidcore_record::Vec3::Y,
        ];
        face.triangles = vec![[0, 1, 2]];
        let mut plan = nodecraft_layers::tess::TessPlan::default();
        plan.faces = vec![face];
        let display = crate::adapter::DisplayMesh {
            mesh: std::sync::Arc::new(solidcraft_geom::Mesh {
                positions: vec![
                    solidcore_record::Vec3::ZERO,
                    solidcore_record::Vec3::X,
                    solidcore_record::Vec3::Y,
                ],
                normals: vec![solidcore_record::Vec3::Z; 3],
                triangles: vec![[0, 1, 2]],
                tri_face: vec![0],
                edges: Vec::new(),
                display_only_edges: false,
                seams: Vec::new(),
                smooth_edges: Vec::new(),
                sharp_edges: Vec::new(),
                edge_faces: Vec::new(),
            }),
            lods: Vec::new(),
            face_ids: vec!["display-only".into()],
            face_colors: Vec::new(),
            edges: Vec::new(),
            comparison: None,
            selection_edges: None,
            radius: 1.0,
            center: solidcore_record::Vec3::ZERO,
        };
        let mut viewport = Viewport::default();
        viewport.display = Some(display);
        viewport.plans = vec![nodecraft_display::PlanInstance {
            target: "input".into(),
            plan: std::sync::Arc::new(plan),
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(Vec::new()),
        }];
        assert_eq!(viewport.plan_face_ids().expect("ids"), vec!["face:7"]);
    }

    #[test]
    fn edit_ghost_plan_edges_are_body_scoped_brep_edges() {
        let mut first = nodecraft_layers::tess::TessPlan::default();
        first.edges = vec![vec![
            solidcore_record::Vec3::new(0.0, 0.0, 0.0),
            solidcore_record::Vec3::new(1.0, 0.0, 0.0),
        ]];
        let mut second = nodecraft_layers::tess::TessPlan::default();
        second.edges = vec![vec![
            solidcore_record::Vec3::new(0.0, 1.0, 0.0),
            solidcore_record::Vec3::new(1.0, 1.0, 0.0),
        ]];
        let plan = |plan| nodecraft_display::PlanInstance {
            target: "input".into(),
            plan: std::sync::Arc::new(plan),
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(vec![Some("edge:0".to_owned())]),
        };
        let mut viewport = Viewport::default();
        viewport.set_edit_ghost_plans(vec![plan(first), plan(second)]);
        assert_eq!(viewport.edit_ghost_edges.len(), 2);
        assert_eq!(viewport.edit_ghost_edges[0].id, "body:0:edge:0");
        assert_eq!(viewport.edit_ghost_edges[1].id, "body:1:edge:0");
    }

    #[test]
    fn feature_edge_picking_ignores_the_downstream_display_edges() {
        let display = crate::adapter::DisplayMesh {
            mesh: std::sync::Arc::new(solidcraft_geom::Mesh::default()),
            lods: Vec::new(),
            face_ids: Vec::new(),
            face_colors: Vec::new(),
            edges: vec![crate::adapter::DisplayEdge {
                id: "edge:downstream".into(),
                points: vec![
                    solidcore_record::Vec3::ZERO,
                    solidcore_record::Vec3::X,
                ],
                midpoint: [0.5, 0.0, 0.0],
                normals: Vec::new(),
            }],
            comparison: None,
            selection_edges: None,
            radius: 1.0,
            center: solidcore_record::Vec3::ZERO,
        };
        let mut viewport = Viewport::default();
        viewport.display = Some(display);
        viewport.edit_ghost_transform = Some([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        viewport.edit_ghost_edges = vec![crate::adapter::DisplayEdge {
            id: "body:0:edge:0".into(),
            points: vec![
                solidcore_record::Vec3::ZERO,
                solidcore_record::Vec3::Y,
            ],
            midpoint: [0.0, 0.5, 0.0],
            normals: Vec::new(),
        }];
        let ids = viewport.edge_ids();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains("body:0:edge:0"));
        assert!(!ids.contains("edge:downstream"));
    }

    #[test]
    fn quality_detail_uses_threshold_hysteresis() {
        let display = crate::adapter::DisplayMesh {
            mesh: std::sync::Arc::new(solidcraft_geom::Mesh::default()),
            lods: Vec::new(),
            face_ids: Vec::new(),
            face_colors: Vec::new(),
            edges: Vec::new(),
            comparison: None,
            selection_edges: None,
            radius: 10.0,
            center: solidcore_record::Vec3::ZERO,
        };
        let mut viewport = Viewport::default();
        viewport.display = Some(display);
        viewport.camera_fit = true;
        viewport.camera.distance = 45.0; // ratio 4.5 -> tier 1
        assert_eq!(viewport.quality_detail(), Some(1));
        viewport.camera.distance = 39.0; // ratio 3.9 is inside the deadband
        assert_eq!(viewport.quality_detail(), Some(1));
        viewport.camera.distance = 35.0; // ratio 3.5 crosses the 4.0 boundary
        assert_eq!(viewport.quality_detail(), Some(2));
    }

    #[test]
    fn preview_scope_only_moves_overlays_of_affected_bodies() {
        let mut viewport = Viewport::default();
        viewport.set_mesh(Some(&NodeMesh::default()));
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let point = solidcore_record::Vec3::new(1.0, 0.5, 0.0);
        let shift = solidcore_record::Vec3::new(4.0, 0.0, 0.0);
        let plain = viewport
            .project_to_viewport_body(point, rect, "body:1:face:0")
            .expect("point projects");
        let moved = viewport
            .project_to_viewport_body(point + shift, rect, "body:0:face:0")
            .expect("moved point projects");
        let mut preview = [[0.0f32; 4]; 4];
        for (axis, column) in preview.iter_mut().enumerate() {
            column[axis] = 1.0;
        }
        preview[3][0] = shift.x as f32;
        viewport.preview_transform = Some(preview);
        viewport.preview_scope = PreviewScope::Bodies(vec!["body:0:".to_owned()]);
        let affected = viewport
            .project_to_viewport_body(point, rect, "body:0:face:0")
            .expect("affected point projects");
        let stationary = viewport
            .project_to_viewport_body(point, rect, "body:1:face:0")
            .expect("stationary point projects");
        assert!((affected - moved).length() < 1.0e-3, "{affected:?} {moved:?}");
        assert!((stationary - plain).length() < 1.0e-3, "{stationary:?} {plain:?}");
    }
    fn view_snapshot(dist: f64, visible: [f32; 2], lens: [f32; 2]) -> ViewSnapshot {
        ViewSnapshot {
            camera: Camera { distance: dist, ..Camera::default() },
            lens_shift: lens,
            camera_fit: true,
            quality_detail: Some(2),
            full: visible,
            visible,
        }
    }

    #[test]
    fn view_snapshot_keeps_the_pending_restore_rects() {
        let mut viewport = Viewport::default();
        let snapshot = view_snapshot(10.0, [800.0, 400.0], [0.0, 0.0]);
        viewport.restore_view(snapshot);
        let saved = viewport.view_snapshot().expect("snapshot");
        assert_eq!(saved.visible, [800.0, 400.0]);
    }

    #[test]
    fn restoring_at_the_same_size_keeps_the_camera() {
        let mut viewport = Viewport::default();
        viewport.restore_view(view_snapshot(10.0, [800.0, 600.0], [0.0, 0.0]));
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        viewport.apply_pending_view_restore(full, full);
        assert_eq!(viewport.camera.distance, 10.0);
        assert_eq!(viewport.quality_detail, Some(2));
    }

    #[test]
    fn restoring_narrower_zooms_out_by_the_aspect_ratio() {
        let mut viewport = Viewport::default();
        viewport.restore_view(view_snapshot(10.0, [800.0, 400.0], [0.0, 0.0]));
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(600.0, 600.0));
        viewport.apply_pending_view_restore(full, full);
        assert!((viewport.camera.distance - 20.0).abs() < 1.0e-9);
    }

    #[test]
    fn restoring_wider_keeps_the_distance() {
        let mut viewport = Viewport::default();
        viewport.restore_view(view_snapshot(10.0, [600.0, 600.0], [0.0, 0.0]));
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1200.0, 600.0));
        viewport.apply_pending_view_restore(full, full);
        assert_eq!(viewport.camera.distance, 10.0);
    }

    #[test]
    fn restoring_lens_shift_recentres_on_the_new_visible_rect() {
        let mut viewport = Viewport::default();
        viewport.restore_view(view_snapshot(10.0, [800.0, 600.0], [0.25, -0.1]));
        let full = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 700.0));
        let visible = egui::Rect::from_min_max(egui::pos2(200.0, 0.0), egui::pos2(1000.0, 700.0));
        viewport.apply_pending_view_restore(full, visible);
        assert_eq!(viewport.lens_shift, lens_shift_for_visible(full, visible));
    }

}
