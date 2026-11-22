//! CPU viewport built on SolidCraft's camera, scene and rasterizer.
//!
//! The wgpu path from `solidcraft-viewport-gpu` will replace the rasterizer once the viewport
//! adapter lands. Keeping the camera/scene/raster on SolidCraft types means the
//! switch is local to this module.

use nodecraft_protocol::Mesh as NodeMesh;
use solidcraft_viewport::{Camera, Rgb, Scene, SceneLine, SceneMesh, StandardView};

use crate::adapter::{self, DisplayMesh};
#[cfg(feature = "gpu-viewport")]
use crate::gpu_viewport::GpuViewport;
use crate::hover::HoverMap;
use crate::viewer_gizmo::{self, Gizmo, GizmoState};
use crate::viewport_grid::{GridKind, grid_lines};
use solidcore_record::Vec3;
use solidcraft_geom::prelude::*;

/// The shared world origin used by both the XY construction grid and the
/// Home/ISO framing. Keeping one constant makes the view centre and the grid
/// origin the same point by construction.
const VIEW_ORIGIN: solidcore_record::Vec3 = solidcore_record::Vec3::ZERO;

/// The colour every path uses when a body has no authored Look colour, and the
/// ignore-look override. The plan table's missing-face fill uses this too.
pub(crate) const DEFAULT_FACE_COLOUR: [f32; 3] = [0.17, 0.18, 0.195];

/// The plan path's enablement and override colour. Face-colours deliberately
/// does not appear in the enable decision: it is a table fill on the same plan
/// renderer, not a reason to fall back to the CPU object mesh (which plan-only
/// evaluation leaves without triangles).
fn plan_draw_state(
    plans_empty: bool,
    _face_colours: bool,
    ignore_look: bool,
) -> (bool, Option<[f32; 3]>) {
    (!plans_empty, ignore_look.then_some(DEFAULT_FACE_COLOUR))
}

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
    /// Draw the XY ground grid, including the axis lines drawn with it.
    pub grid: bool,
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
    /// Display tessellation error budget in logical (on-screen) pixels.
    /// Per-body screen-space LOD chooses each body's tolerance from this; the
    /// display crate converts it to physical pixels once at `pixels_per_point`.
    #[serde(default = "default_display_error_px")]
    pub display_error_px: f64,
    /// Standard B-rep edge width in screen pixels. Ghost and highlight edges
    /// derive their widths from this one value. The GPU edge ribbon reads it
    /// from a uniform, so changing it never rebuilds a plan or re-uploads an
    /// edge. The viewport size is physical pixels, so this is one device pixel
    /// at any HiDPI scale, exactly as the previous fixed width was.
    #[serde(default = "default_edge_width_px")]
    pub edge_width_px: f32,
    /// Draw seam edges (a seam between two pieces of one smooth surface, such
    /// as a cone's wall line or a sphere's meridian). Off by default, matching
    /// SolidCore's shaded display. This is a uniform mask: toggling it never
    /// rebuilds a plan or re-uploads an edge.
    #[serde(default)]
    pub show_seam_edges: bool,
}

/// Default screen-space display error budget, in logical (on-screen) pixels.
#[must_use]
pub fn default_display_error_px() -> f64 {
    0.75
}

/// The standard B-rep edge width in screen pixels. 1.0 is the width the
/// viewport drew before the setting existed.
#[must_use]
pub fn default_edge_width_px() -> f32 {
    1.0
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            translucent: false,
            ghost_opacity: 0.45,
            ignore_look_colors: false,
            edges: true,
            grid: true,
            triangles: false,
            normals: false,
            face_colours: false,
            zebra: false,
            roughness: 0.42,
            metallic: 0.0,
            cad_mouse: true,
            pan_speed: 1.0,
            orbit_speed: 1.0,
            display_error_px: default_display_error_px(),
            edge_width_px: default_edge_width_px(),
            show_seam_edges: false,
        }
    }
}

/// The one grid visibility decision shared by the CPU scene and the GPU
/// grid/axis passes. Display-only: toggling it never asks the engine to
/// re-evaluate or re-tessellate.
pub(crate) fn draws_grid(view: &ViewSettings) -> bool {
    view.grid
}

impl ViewSettings {
    fn any_extra(self) -> bool {
        self.translucent
            || !self.edges
            || !self.grid
            || self.triangles
            || self.normals
            || self.face_colours
            || self.zebra
            || self.ignore_look_colors
            || self.show_seam_edges
            || (self.ghost_opacity - 0.45).abs() > f32::EPSILON
            || (self.display_error_px - default_display_error_px()).abs() > f64::EPSILON
            || (self.edge_width_px - default_edge_width_px()).abs() > f32::EPSILON
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
    tx: std::sync::mpsc::Sender<(u64, u64, NodeMesh, ViewSettings)>,
    rx: std::sync::mpsc::Receiver<(u64, u64, DisplayMesh, Option<PreparedObject>)>,
}

impl Default for DisplayPrep {
    fn default() -> Self {
        let (tx, requests) = std::sync::mpsc::channel::<(u64, u64, NodeMesh, ViewSettings)>();
        let (results, rx) =
            std::sync::mpsc::channel::<(u64, u64, DisplayMesh, Option<PreparedObject>)>();
        std::thread::Builder::new()
            .name("nodecraft-display-prep".into())
            .spawn(move || {
                while let Ok((generation, snapshot, mesh, view)) = requests.recv() {
                    let mut display = adapter::display_mesh(&mesh);
                    if view.ignore_look_colors {
                        display.face_colors.clear();
                    }
                    #[cfg(feature = "gpu-viewport")]
                    let prepared = Some(crate::gpu_viewport::prepare_object(&display, &view));
                    #[cfg(not(feature = "gpu-viewport"))]
                    let prepared = None;
                    if results.send((generation, snapshot, display, prepared)).is_err() {
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
    [f32::from(colour[0]) / 255.0, f32::from(colour[1]) / 255.0, f32::from(colour[2]) / 255.0]
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
    /// The plan set is staged in the GPU layer: display updates ride with it
    /// and become visible in the same activation.
    plan_set_staged: bool,
    /// Exact snapshot the current/staged plan set came from. Edge/object
    /// payloads are tagged with it so a pair can only activate together.
    plans_snapshot: u64,
    /// Snapshot of the display geometry currently on the GPU.
    displayed_snapshot: u64,
    /// Target whose display mesh is currently shown. Staged targets switch only
    /// when the plan set activates.
    displayed_target: Option<String>,
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
    last_gpu_pick_inputs: Option<([f32; 2], u64, Option<[[f32; 4]; 4]>, Camera)>,
    /// An edge click waiting for its own GPU pick result, with the pick scene
    /// generation it was requested against.
    pending_gpu_pick: Option<(u64, egui::Pos2, u64)>,
    /// A face pick used only to answer "did this edge click miss the body?".
    pending_face_empty: Option<(u64, egui::Pos2, u64)>,
    /// The latest hover request. A result from an older request must not move
    /// the hover back to a stale pointer.
    latest_hover_pick: Option<u64>,
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
    /// The Axes navigator's drag state across frames (including a pending
    /// cursor warp at the window edge).
    navigator_drag: crate::view_navigator::Drag,
    /// The pointer gesture currently in flight and whether viewport chrome
    /// owned it from its press. Every viewport consumer of the pointer reads
    /// this one decision instead of re-testing the hover position.
    chrome_pointer: Option<ChromePointer>,
    hovered_edge: Option<String>,
    hovered_face: Option<String>,
    selected_edges: std::collections::BTreeSet<String>,
    /// Faces committed on the active feature node. They are the same reference
    /// ids the node stores, so the highlight and the picked value cannot
    /// disagree after a re-evaluation.
    selected_faces: std::collections::BTreeSet<String>,
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
    /// Alpha for the drag-preview instances. 1.0 is the normal opaque drag
    /// preview; lower values put the preview in the ghost layer.
    preview_alpha: f32,
    /// What the display currently being staged does to the preview when its
    /// plan set and edges activate. Overlays staged in the same display update
    /// use the same decision.
    display_retires_preview: bool,
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

/// The 18-point Home icon under the compass, right-aligned with its box.
fn navigator_home_icon_rect(outer: egui::Rect) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::pos2(outer.right() - 18.0, outer.bottom() + 10.0),
        egui::vec2(18.0, 18.0),
    )
}

/// The Home button's 30-point round hit area around its icon.
fn navigator_home_rect(outer: egui::Rect) -> egui::Rect {
    egui::Rect::from_center_size(navigator_home_icon_rect(outer).center(), egui::vec2(30.0, 30.0))
}

/// The projection toggle's 18-point icon, anchored on the Home button.
fn navigator_projection_icon_rect(outer: egui::Rect) -> egui::Rect {
    let home_icon = navigator_home_icon_rect(outer);
    egui::Rect::from_min_size(
        egui::pos2(
            home_icon.right() - home_icon.width(),
            navigator_home_rect(outer).bottom() + 10.0,
        ),
        home_icon.size(),
    )
}

/// The projection toggle's 30-point hit area.
fn navigator_projection_rect(outer: egui::Rect) -> egui::Rect {
    egui::Rect::from_center_size(
        navigator_projection_icon_rect(outer).center(),
        egui::vec2(30.0, 30.0),
    )
}

/// The display-settings cog's 18-point icon, anchored on the projection button.
fn navigator_view_icon_rect(outer: egui::Rect) -> egui::Rect {
    let projection_icon = navigator_projection_icon_rect(outer);
    let projection = navigator_projection_rect(outer);
    egui::Rect::from_min_size(
        egui::pos2(projection_icon.right() - projection_icon.width(), projection.bottom() + 10.0),
        projection_icon.size(),
    )
}

/// The display-settings cog's 30-point hit area.
fn navigator_view_rect(outer: egui::Rect) -> egui::Rect {
    egui::Rect::from_center_size(navigator_view_icon_rect(outer).center(), egui::vec2(30.0, 30.0))
}

/// Every floating control in the viewport's top-right chrome. Drawing and hit
/// testing both go through this one layout: the compass box, the Home button,
/// the projection toggle and the display-settings cog.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NavigatorChrome {
    layout: crate::view_navigator::Layout,
    home: egui::Rect,
    projection: egui::Rect,
    display: egui::Rect,
}

impl NavigatorChrome {
    fn at(viewport: egui::Rect, right_inset: f32) -> Self {
        let layout = crate::view_navigator::layout(viewport, right_inset);
        Self {
            layout,
            home: navigator_home_rect(layout.outer),
            projection: navigator_projection_rect(layout.outer),
            display: navigator_view_rect(layout.outer),
        }
    }

    fn rects(self) -> [egui::Rect; 4] {
        [self.layout.outer, self.home, self.projection, self.display]
    }

    fn contains(self, pointer: egui::Pos2) -> bool {
        self.rects().iter().any(|rect| rect.contains(pointer))
    }
}

/// Which button began the current pointer gesture, where it began, and whether
/// the viewport's chrome owns it. The ownership decision is made once, from
/// the press position, and held for the rest of the press, drag and release
/// frame. Drawing, picking, selection and camera navigation all read it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ChromePointer {
    button: egui::PointerButton,
    origin: egui::Pos2,
    owned: bool,
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

/// A labelled diagnostic row for the Debug window. The viewport reads live
/// values into these; the window owns presentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugRow {
    pub label: &'static str,
    pub value: String,
}

impl DebugRow {
    fn new(label: &'static str, value: String) -> Self {
        Self { label, value }
    }
}

fn debug_vec3(value: solidcore_record::Vec3) -> String {
    format!("{:.3} {:.3} {:.3}", value.x, value.y, value.z)
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            chrome_visible: true,
            display: None,
            plans: Vec::new(),
            plan_set_staged: false,
            plans_snapshot: 0,
            displayed_snapshot: 0,
            displayed_target: None,
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
            last_gpu_pick_inputs: None,
            pending_gpu_pick: None,
            pending_face_empty: None,
            latest_hover_pick: None,
            selection_mode: SelectionMode::Object,
            face_picks: Vec::new(),
            edge_picks: Vec::new(),
            edge_picking_enabled: false,
            quality_detail: None,
            selection_miss: false,
            chrome_blockers: Vec::new(),
            navigator_drag: crate::view_navigator::Drag::default(),
            chrome_pointer: None,
            hovered_edge: None,
            hovered_face: None,
            selected_edges: std::collections::BTreeSet::new(),
            selected_faces: std::collections::BTreeSet::new(),
            dock_rect: None,
            transform_gizmo: None,
            gizmo_state: GizmoState::default(),
            gizmo_update: None,
            comparison_ghost: false,
            preview_transform: None,
            preview_alpha: 1.0,
            display_retires_preview: true,
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

    /// Tell the GPU callback whether pass scopes should be recorded. Off is a
    /// plain bool it checks before touching the profiler slot.
    pub fn set_gpu_profiler_enabled(&mut self, enabled: bool) {
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_gpu_profiler_enabled(enabled);
        }
        #[cfg(not(feature = "gpu-viewport"))]
        let _ = enabled;
    }

    /// Cumulative uploaded object bytes from the existing GPU debug snapshot.
    /// The profiler differences consecutive samples; off never calls this.
    pub fn gpu_object_bytes(&self) -> Option<u64> {
        #[cfg(feature = "gpu-viewport")]
        {
            self.gpu.as_ref()?.debug_info().map(|info| info.object_bytes)
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            None
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
    ///
    /// The GPU path now gets a per-body screen-space level from
    /// [`Self::lod_view`], so this CPU-mesh tier is retained only for the CPU
    /// LOD path and its tests. Follow-up: remove it with that path.
    #[allow(dead_code)]
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
            BOUNDS
                .get(current_index.saturating_sub(1))
                .is_some_and(|boundary| ratio < boundary * 0.9)
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
        if self.plan_set_staged {
            #[cfg(feature = "gpu-viewport")]
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.stage_diagnostics(diagnostics);
                return;
            }
        }
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
            self.plan_ready() && self.plan_debug_struct().is_some_and(|debug| debug.pending)
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            false
        }
    }

    pub fn view_plans_hash(&self) -> u64 {
        #[cfg(feature = "gpu-viewport")]
        {
            crate::plan_math::plan_identity_hash(&self.plans)
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            0
        }
    }

    #[cfg(feature = "gpu-viewport")]
    fn plan_debug_struct(&self) -> Option<crate::gpu_plan::PlanDebug> {
        self.gpu.as_ref().and_then(|gpu| gpu.plan_debug())
    }

    pub fn plan_debug(&self) -> Option<serde_json::Value> {
        #[cfg(feature = "gpu-viewport")]
        {
            self.plan_debug_struct().and_then(|debug| serde_json::to_value(debug).ok())
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            None
        }
    }

    pub fn set_plans(&mut self, plans: Vec<nodecraft_display::PlanInstance>) {
        self.plans = plans;
        self.plan_set_staged = self.refresh_plan_state();
    }

    /// Hand one display frame to the viewport. The plan set and the display it
    /// belongs to are one unit: when the plans need staging, the edge mesh,
    /// picking display, diagnostics and target are staged with them and become
    /// visible in the same GPU activation. `retire_preview` is part of that
    /// pair: an authoritative display at the edited placement retires the drag
    /// preview only when it activates, while a live drag keeps it. Returns
    /// whether the display is staged.
    pub fn set_display(
        &mut self,
        plans: Vec<nodecraft_display::PlanInstance>,
        mesh: Option<&NodeMesh>,
        diagnostics: Vec<String>,
        target: &str,
        snapshot: u64,
        retire_preview: bool,
    ) -> bool {
        self.sync_activated_display();
        self.display_retires_preview = retire_preview;
        self.plans = plans;
        self.plans_snapshot = snapshot;
        let staged = self.refresh_plan_state();
        if staged {
            #[cfg(feature = "gpu-viewport")]
            {
                let display = mesh.map(|mesh| self.display_mesh_for_view(mesh));
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.stage_display(
                        display,
                        mesh.cloned(),
                        Some(diagnostics),
                        Some(target.to_owned()),
                        snapshot,
                        retire_preview,
                    );
                }
                self.plan_set_staged = true;
                // Any older async display prep belongs to a superseded
                // snapshot; its result will be rejected by the snapshot check.
                self.pending_display = false;
                self.dirty = true;
                return true;
            }
        }
        self.plan_set_staged = false;
        self.displayed_snapshot = snapshot;
        self.displayed_target = Some(target.to_owned());
        self.tessellation_diagnostics = diagnostics;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.commit_immediate_snapshot(snapshot);
        }
        // Plan triangles and this display are one commit. Never let an async
        // display-prep worker leave the new plan chunks visible with the old
        // edge mesh: plan-path displays commit synchronously. The >=100k-face
        // async path remains for CPU fallback displays with no plan set.
        self.update_mesh_with(mesh, self.plans.is_empty(), retire_preview);
        false
    }

    pub fn displayed_target(&self) -> Option<&str> {
        self.displayed_target.as_deref()
    }

    pub fn displayed_snapshot(&self) -> u64 {
        self.displayed_snapshot
    }

    /// Diagnostic entry point: copy a display the GPU already activated into
    /// the CPU mirror before reporting state.
    pub fn refresh_display_mirror(&mut self) {
        self.sync_activated_display();
    }

    /// Copy the display payload the GPU layer activated during a paint callback
    /// into this CPU mirror. The GPU display, edges and picking already switched
    /// together; this only keeps the viewport's CPU-side copy in step.
    fn sync_activated_display(&mut self) {
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut()
            && let Some(active) = gpu.take_activated_display()
        {
            self.display = active.display.map(|display| (*display).clone());
            self.source_mesh = active.source_mesh;
            if let Some(diagnostics) = active.diagnostics {
                self.tessellation_diagnostics = diagnostics;
            }
            if let Some(target) = active.target {
                self.displayed_target = Some(target);
            }
            // A retirement is the one activation change the CPU mirror cannot
            // know on its own. A kept preview is already the app's current
            // value (or newer), so never roll a live drag back to the copy the
            // pair captured when it was staged.
            if active.preview.is_none() {
                self.preview_transform = None;
                self.preview_alpha = 1.0;
            }
            self.displayed_snapshot = active.snapshot;
            self.plan_set_staged = false;
            self.dirty = true;
        }
    }

    /// Re-evaluate the plan draw state after the display mesh (and therefore
    /// its face colours) becomes available. Returns whether the GPU layer
    /// staged the set for a later activation.
    fn refresh_plan_state(&mut self) -> bool {
        // The plan shader looks colours up in the shared table; only the
        // ignore-look override replaces every face at once. Face-colours is a
        // table fill, so it must not disable the plan path: that fallback used
        // the CPU display mesh, which plan-only evaluation has not built since
        // 7fe0c772a.
        let (enabled, colour) = plan_draw_state(
            self.plans.is_empty(),
            self.view.face_colours,
            self.view.ignore_look_colors,
        );
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            return gpu.set_plans(self.plans.clone(), enabled, colour, self.plans_snapshot)
                == crate::gpu_plan::PlanSetState::Staged;
        }
        #[cfg(not(feature = "gpu-viewport"))]
        let _ = (enabled, colour);
        false
    }

    /// Drop every display resource owned by the current document: plans, GPU
    /// plan chunks, meshes and ghosts. Session/tab changes must go through this
    /// so a closed project cannot keep resident plan meshes.
    pub fn clear_document_display(&mut self) {
        self.set_plans(Vec::new());
        self.set_mesh(None);
        self.clear_edit_ghost();
        self.plan_set_staged = false;
        self.plans_snapshot = 0;
        self.displayed_snapshot = 0;
        self.displayed_target = None;
        self.tessellation_diagnostics.clear();
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.clear_plan_document();
            gpu.clear_ghost();
        }
    }

    pub fn set_mesh(&mut self, mesh: Option<&NodeMesh>) {
        self.update_mesh(mesh, true);
        self.fit();
    }

    /// Update the mesh without recentering the camera. Used while a transform
    /// gizmo is being dragged so the viewport follows the edit instead of
    /// snapping back to a fitted view on every evaluation result.
    pub fn set_mesh_preserving_camera(&mut self, mesh: Option<&NodeMesh>) {
        // A caller that wants the preview retired does so explicitly through
        // the display unit; this path keeps a live drag preview in step.
        self.update_mesh(mesh, false);
    }

    /// Display `ghost` as a faint feature-input layer under `base`. This is not
    /// an exact material difference; it is the selected feature's upstream body.
    pub fn set_feature_ghost_mesh(&mut self, base: &NodeMesh, ghost: &NodeMesh) {
        self.sync_activated_display();
        self.comparison_ghost = true;
        self.selected_edges.clear();
        self.hovered_edge = None;
        let empty = NodeMesh::default();
        self.reset_lods();
        let display = crate::adapter::display_mesh_with_difference(base, &empty, ghost);
        #[cfg(feature = "gpu-viewport")]
        if self.plan_set_staged {
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.stage_display(
                    Some(display),
                    None,
                    None,
                    None,
                    self.plans_snapshot,
                    self.display_retires_preview,
                );
                gpu.set_ghost_matrix(None);
                gpu.set_ghost_plans(
                    self.feature_ghost_plans.clone(),
                    self.view.ghost_opacity,
                    Some(ghost_plan_colour()),
                    !self.feature_ghost_plans.is_empty(),
                );
            }
            self.dirty = true;
            return;
        }
        self.display = Some(display);
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref(), self.display_retires_preview);
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
        self.sync_activated_display();
        self.comparison_ghost = false;
        self.selected_edges.clear();
        self.hovered_edge = None;
        let mut display = crate::adapter::display_mesh_with_difference(base, added, removed);
        // The selection display is originalIn -> difference overlay -> originalIn.edges.
        // Replace the result body's edge set so no generated edge can be drawn or picked.
        display.edges = crate::adapter::display_mesh(selection_source).edges;
        display.selection_edges = None;
        self.reset_lods();
        #[cfg(feature = "gpu-viewport")]
        if self.plan_set_staged {
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.stage_display(
                    Some(display),
                    None,
                    None,
                    None,
                    self.plans_snapshot,
                    self.display_retires_preview,
                );
            }
            self.dirty = true;
            return;
        }
        self.display = Some(display);
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref(), self.display_retires_preview);
        }
        self.dirty = true;
    }

    fn update_mesh(&mut self, mesh: Option<&NodeMesh>, retire_preview: bool) {
        self.update_mesh_with(mesh, true, retire_preview);
    }

    fn update_mesh_with(
        &mut self,
        mesh: Option<&NodeMesh>,
        allow_async: bool,
        retire_preview: bool,
    ) {
        self.sync_activated_display();
        if self.plan_set_staged {
            #[cfg(feature = "gpu-viewport")]
            {
                let display = mesh.map(|mesh| self.display_mesh_for_view(mesh));
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.stage_display(
                        display,
                        mesh.cloned(),
                        None,
                        None,
                        self.plans_snapshot,
                        retire_preview,
                    );
                }
                self.dirty = true;
                return;
            }
        }
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
        if retire_preview {
            self.preview_transform = None;
            self.preview_alpha = 1.0;
        }
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
            if allow_async && mesh.faces.len() >= ASYNC_DISPLAY_TRIANGLES {
                // Conversion and normal splitting run on the display-prep
                // worker; the old display stays visible until the new one is
                // ready.
                self.pending_display = true;
                self.display_prep_started = Some(std::time::Instant::now());
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[viewport] async display prep faces={}", mesh.faces.len());
                }
                let generation = self.display_generation;
                let _ = self.display_prep.tx.send((
                    generation,
                    self.plans_snapshot,
                    mesh.clone(),
                    self.view,
                ));
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
            gpu.set_mesh(self.display.as_ref(), retire_preview);
            if !keep_edit_ghost {
                gpu.clear_ghost();
            }
        }
    }

    fn poll_display_prep(&mut self) {
        let Some((generation, snapshot, display, prepared)) = self.display_prep.rx.try_recv().ok()
        else {
            return;
        };
        if generation != self.display_generation || !self.pending_display {
            return;
        }
        // An async prep belongs to one object state. If a newer snapshot has
        // since replaced the plans, this display must not attach to it.
        if snapshot != self.plans_snapshot {
            self.pending_display = false;
            return;
        }
        self.pending_display = false;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[viewport] display prep done {}ms",
                self.display_prep_started.take().map_or(0, |started| started.elapsed().as_millis())
            );
        }
        self.sync_activated_display();
        let staged = self.refresh_plan_state();
        #[cfg(feature = "gpu-viewport")]
        if staged {
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.stage_display(
                    Some(display),
                    self.source_mesh.clone(),
                    None,
                    None,
                    self.plans_snapshot,
                    true,
                );
            }
            self.plan_set_staged = true;
            self.dirty = true;
            return;
        }
        self.plan_set_staged = false;
        self.display = Some(display);
        self.preview_transform = None;
        self.preview_alpha = 1.0;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_prepared_mesh(self.display.as_ref(), prepared, true);
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
        // The merged display mesh's `body:N` index is the plan-list index. Give
        // the CPU colour path the same stable identity the plan table uses.
        if !self.plans.is_empty() {
            display.body_ids = self.plans.iter().map(|plan| plan.body).collect();
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
                gpu.set_mesh(self.display.as_ref(), true);
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
                gpu.set_preview_transform(self.preview_transform, view.ghost_opacity);
            }
        }
        if ghost_changed {
            self.preview_alpha = view.ghost_opacity;
        }
        // Face-colour and translucency modes switch the plan renderer off/on.
        self.plan_set_staged = self.refresh_plan_state();
        self.dirty = true;
    }

    pub fn view(&self) -> ViewSettings {
        self.view
    }

    /// Camera and viewport inputs for per-body screen-space LOD.
    ///
    /// `None` until the camera has been fitted and a display with a known full
    /// viewport rect is installed. The full rect (not the visible sub-rect) is
    /// the camera's aspect.
    #[must_use]
    pub fn lod_view(&self, pixels_per_point: f32) -> Option<nodecraft_display::LodView> {
        if !self.camera_fit || self.display.is_none() {
            return None;
        }
        let full = self.last_full?;
        Some(nodecraft_display::LodView {
            eye: self.camera.eye().into(),
            fov_y: self.camera.fov,
            ortho_height: self.camera.half_height() * 2.0,
            viewport: [full.width() as f64, full.height() as f64],
            pixels_per_point: pixels_per_point as f64,
            error_px: self.view.display_error_px,
        })
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
            self.selected_faces.clear();
            self.hovered_edge = None;
            self.hovered_face = None;
            self.pending_gpu_pick = None;
            self.pending_face_empty = None;
            self.latest_hover_pick = None;
        }
        self.selection_mode = mode;
    }

    /// Highlight the feature node's committed face references. The ids come
    /// from the node's `faces` value, not from a second pick model, so the
    /// selected highlight survives re-evaluation unchanged.
    pub fn set_selected_faces(&mut self, ids: impl IntoIterator<Item = String>) {
        let ids: std::collections::BTreeSet<String> = ids.into_iter().collect();
        if ids != self.selected_faces {
            self.selected_faces = ids;
            self.dirty = true;
        }
    }

    #[cfg(test)]
    pub fn selected_faces(&self) -> &std::collections::BTreeSet<String> {
        &self.selected_faces
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
        let ghost_edges =
            self.edit_ghost_edges.iter().filter(|edge| !edge.id.starts_with("preview-edge:"));
        let available: std::collections::HashSet<&str> = if self.picking_input_ghost_only() {
            ghost_edges.map(|edge| edge.id.as_str()).collect()
        } else {
            self.topology_edges(display)
                .iter()
                .chain(ghost_edges)
                .map(|edge| edge.id.as_str())
                .collect()
        };
        self.selected_edges =
            ids.iter().filter(|id| available.contains(id.as_str())).cloned().collect();
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
        display.selection_edges.as_deref().unwrap_or(display.edges.as_slice())
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
                        triangle.iter().all(|index| (*index as usize) < face.uv.len())
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
            return self
                .edit_ghost_edges
                .iter()
                .filter(|edge| !edge.id.starts_with("preview-edge:"))
                .map(|edge| edge.id.clone())
                .collect();
        }
        self.display
            .as_ref()
            .map(|display| {
                self.topology_edges(display).iter().map(|edge| edge.id.clone()).collect()
            })
            .unwrap_or_default()
    }

    pub fn fit(&mut self) {
        if self.pending_display {
            self.fit_when_display_ready = true;
            return;
        }
        #[cfg(feature = "gpu-viewport")]
        if self.plan_set_staged
            && let Some((radius, center)) =
                self.gpu.as_ref().and_then(crate::gpu_viewport::GpuViewport::pending_display_bounds)
        {
            let (yaw, pitch) = StandardView::Iso.angles();
            self.camera.target = center;
            self.camera.yaw = -yaw;
            self.camera.pitch = pitch;
            self.camera.distance = (radius * 3.0).max(1.0);
            self.lens_shift = [0.0, 0.0];
            self.camera_fit = true;
            self.dirty = true;
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
        let tan_half = if self.camera.fov > 0.0 { (self.camera.fov * 0.5).tan() } else { 0.4 };
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

    /// Camera and view-matrix rows for the Debug window's Camera tab.
    ///
    /// These read the live camera and cost nothing until the window asks for
    /// them; the viewport no longer paints a diagnostics overlay.
    pub fn camera_debug_rows(&self) -> Vec<DebugRow> {
        let camera = &self.camera;
        let (right, up, back) = camera.basis();
        let target = camera.target;
        let eye = camera.eye();
        let mut rows = vec![
            DebugRow::new("Yaw", format!("{:.2}°", camera.yaw.to_degrees())),
            DebugRow::new("Pitch", format!("{:.2}°", camera.pitch.to_degrees())),
            DebugRow::new("Roll", format!("{:.2}°", camera.roll.to_degrees())),
            DebugRow::new("FOV", format!("{:.3}", camera.fov)),
            DebugRow::new("Distance", format!("{:.3}", camera.distance)),
            DebugRow::new("Half height", format!("{:.3}", camera.half_height())),
            DebugRow::new("Target", debug_vec3(target)),
            DebugRow::new("Eye", debug_vec3(eye)),
            DebugRow::new("Right", debug_vec3(right)),
            DebugRow::new("Up", debug_vec3(up)),
            DebugRow::new("Back", debug_vec3(back)),
            DebugRow::new(
                "Lens shift",
                format!("{:.4} {:.4}", self.lens_shift[0], self.lens_shift[1]),
            ),
        ];
        for (index, row) in camera.view().0.iter().enumerate() {
            rows.push(DebugRow::new(
                "View row",
                format!("{index}: {:.3} {:.3} {:.3} {:.3}", row[0], row[1], row[2], row[3]),
            ));
        }
        rows
    }

    /// GPU/frame rows for the Debug window's GPU tab.
    ///
    /// `frame_ms` is the profiler's existing CPU frame series; this method never
    /// starts a query or measures a frame itself.
    pub fn gpu_debug_rows(&self, frame_ms: f32) -> Vec<DebugRow> {
        let fps = if frame_ms > 0.0 { 1000.0 / frame_ms } else { 0.0 };
        let mut rows = Vec::new();
        if let Some(full) = self.last_full {
            rows.push(DebugRow::new(
                "Viewport",
                format!("{}×{}", full.width().round() as i32, full.height().round() as i32),
            ));
        }
        rows.push(DebugRow::new("Frame", format!("{frame_ms:.1} ms  {fps:.0} fps")));
        #[cfg(feature = "gpu-viewport")]
        {
            if let Some(gpu) = self.gpu.as_ref().and_then(|gpu| gpu.debug_info()) {
                rows.push(DebugRow::new(
                    "Adapter",
                    format!("{} ({}, {})", gpu.adapter, gpu.backend, gpu.device_type),
                ));
                rows.push(DebugRow::new(
                    "Object",
                    format!(
                        "{} tris  {} chunks  {:.1} MB",
                        gpu.triangles,
                        gpu.chunks,
                        gpu.object_bytes as f64 / (1024.0 * 1024.0)
                    ),
                ));
                rows.push(DebugRow::new(
                    "MSAA",
                    format!("{}x", crate::gpu_viewport::msaa_samples()),
                ));
            } else {
                rows.push(DebugRow::new("GPU", "viewport not initialized".to_owned()));
            }
        }
        #[cfg(not(feature = "gpu-viewport"))]
        rows.push(DebugRow::new("GPU", "viewport feature off".to_owned()));
        rows.push(DebugRow::new(
            "Tessellation warnings",
            self.tessellation_diagnostics.len().to_string(),
        ));
        for diagnostic in self.tessellation_diagnostics.iter().take(6) {
            rows.push(DebugRow::new("Warning", diagnostic.clone()));
        }
        rows
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
        let old_aspect =
            if snapshot.visible[1] > 1.0 { snapshot.visible[0] / snapshot.visible[1] } else { 1.0 };
        let new_aspect = visible.width() / visible.height().max(1.0);
        if new_aspect < old_aspect && new_aspect > 1.0e-6 {
            self.camera.distance *= f64::from(old_aspect / new_aspect);
            self.dirty = true;
        }
        if snapshot.lens_shift != [0.0, 0.0] {
            self.lens_shift = lens_shift_for_visible(full, visible);
        }
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
        if self.transform_gizmo.as_ref().map(|current| current.node_id.as_str())
            != gizmo.as_ref().map(|next| next.node_id.as_str())
        {
            self.gizmo_state.reset();
            // A selection change mid-drag must not leave the previewed geometry
            // displaced after the triad disappears.
            self.set_preview_transform(None, 1.0);
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
        let length: f64 = points.windows(2).map(|pair| (pair[1] - pair[0]).len()).sum();
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
            let matrix = crate::plan_math::instance_matrix(&instance.placement);
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
                    .map(|id| if multiple_bodies { format!("body:{body}:{id}") } else { id })
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
                    class: instance.edge_classes.get(index).copied().unwrap_or_default(),
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
            gpu.set_mesh(self.display.as_ref(), false);
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

    /// Show `transform` on the whole drawn body with `alpha`. The preview is
    /// one value: the object instances, plan uniform and scene projection all
    /// read it, so they cannot disagree during a drag.
    pub fn set_preview_transform(&mut self, transform: Option<[[f32; 4]; 4]>, alpha: f32) {
        let cpu_matches = self.preview_transform == transform && self.preview_alpha == alpha;
        // A staged activation can retire the unit's preview before this CPU
        // mirror syncs. Requiring both to match keeps an apparently unchanged
        // preview from being dropped just because the mirror is stale.
        #[cfg(feature = "gpu-viewport")]
        let gpu_matches = self.gpu.as_ref().is_none_or(|gpu| gpu.preview_matches(transform, alpha));
        #[cfg(not(feature = "gpu-viewport"))]
        let gpu_matches = true;
        if cpu_matches && gpu_matches {
            return;
        }
        self.preview_transform = transform;
        self.preview_alpha = alpha;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_preview_transform(transform, alpha);
        }
        self.dirty = true;
    }

    /// The top-right chrome's one hit test: the compass box, Home, the
    /// projection toggle and the display cog. `axis_gizmo` draws from the same
    /// [`NavigatorChrome`] value.
    fn navigator_chrome(&self, viewport: egui::Rect) -> NavigatorChrome {
        NavigatorChrome::at(viewport, self.chrome_right_inset)
    }

    /// Whether `pointer` is in the view navigator's chrome, which takes the
    /// press before viewport picking or navigation sees it.
    pub(crate) fn navigator_pointer_blocked(
        &self,
        viewport: egui::Rect,
        pointer: egui::Pos2,
    ) -> bool {
        self.chrome_visible && self.navigator_chrome(viewport).contains(pointer)
    }

    /// Whether `point` is over a triad handle. The triad's drawing and this
    /// hit test both go through [`viewer_gizmo`]'s `Proj` and `shapes`.
    fn transform_gizmo_contains_point(&self, rect: egui::Rect, point: egui::Pos2) -> bool {
        let Some(display) = self.display.as_ref() else { return false };
        let Some(gizmo) = self.transform_gizmo.as_ref() else { return false };
        viewer_gizmo::contains_pointer(
            gizmo,
            &self.camera,
            rect,
            self.lens_shift,
            display.radius,
            point,
        )
    }

    /// Whether viewport chrome owns `point`: the triad or the navigator block.
    fn chrome_contains_point(&self, rect: egui::Rect, point: egui::Pos2) -> bool {
        self.transform_gizmo_contains_point(rect, point)
            || (self.chrome_visible && self.navigator_chrome(rect).contains(point))
    }

    fn chrome_contains_pointer(&self, ui: &egui::Ui, rect: egui::Rect) -> bool {
        ui.ctx()
            .pointer_hover_pos()
            .is_some_and(|pointer| self.chrome_contains_point(rect, pointer))
    }

    /// Decide once, from the press position, whether viewport chrome owns the
    /// pointer gesture, and hold that decision until the button is released.
    ///
    /// Picking, selection misses and camera navigation all read
    /// [`Self::pointer_owned_by_chrome`] instead of re-testing the current
    /// hover position. A pointer that starts on a handle and ends over the
    /// model (or vice versa) therefore cannot hand the click to the viewport.
    fn update_chrome_pointer(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        overlay: egui::Rect,
        blockers: &[egui::Rect],
    ) {
        let (press, release, pointer_gone, focused, buttons_down) = ui.input(|input| {
            let mut press = None;
            let mut release = None;
            let mut gone = false;
            for event in &input.events {
                match event {
                    egui::Event::PointerButton { pos, button, pressed: true, .. }
                        if press.is_none() =>
                    {
                        press = Some((*button, *pos))
                    }
                    egui::Event::PointerButton { button, pressed: false, .. } => {
                        release = Some(*button)
                    }
                    egui::Event::PointerGone => gone = true,
                    _ => {}
                }
            }
            (
                press,
                release,
                gone,
                input.focused,
                [
                    input.pointer.button_down(egui::PointerButton::Primary),
                    input.pointer.button_down(egui::PointerButton::Middle),
                    input.pointer.button_down(egui::PointerButton::Secondary),
                ],
            )
        });
        let is_down = |button: egui::PointerButton| match button {
            egui::PointerButton::Primary => buttons_down[0],
            egui::PointerButton::Middle => buttons_down[1],
            egui::PointerButton::Secondary => buttons_down[2],
            _ => false,
        };
        if let Some(gesture) = self.chrome_pointer {
            let released = release == Some(gesture.button);
            // Hold through the release frame so the click's consumers see the
            // same decision as the press frame; drop it on the next frame or
            // when the gesture is lost.
            if pointer_gone || !focused || (!is_down(gesture.button) && !released) {
                self.chrome_pointer = None;
            }
        }
        if let Some((button, origin)) = press {
            let blocked = overlay.contains(origin)
                || blockers.iter().any(|blocker| blocker.contains(origin))
                || self.chrome_contains_point(rect, origin);
            self.chrome_pointer = Some(ChromePointer { button, origin, owned: blocked });
        }
    }

    /// Whether the pointer gesture in flight began on viewport chrome (or an
    /// overlay). Held for the whole press/drag/release.
    pub(crate) fn pointer_owned_by_chrome(&self) -> bool {
        self.chrome_pointer.is_some_and(|gesture| gesture.owned)
    }

    /// The held press position and ownership, for tests and diagnostics.
    #[cfg(test)]
    fn chrome_pointer(&self) -> Option<(egui::PointerButton, egui::Pos2, bool)> {
        self.chrome_pointer.map(|gesture| (gesture.button, gesture.origin, gesture.owned))
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
                if update.focus.is_some() {
                    pending.focus = update.focus;
                }
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
        self.sync_activated_display();
        self.poll_display_prep();
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
        // One decision for the whole gesture: did the press land on viewport
        // chrome or an overlay? Picking, selection and camera navigation read
        // this, not the current hover position.
        self.update_chrome_pointer(ui, rect, overlay, blockers);
        let chrome_owned = self.pointer_owned_by_chrome();

        let modifiers = ui.ctx().input(|input| input.modifiers);
        let middle = response.dragged_by(egui::PointerButton::Middle);
        let secondary = response.dragged_by(egui::PointerButton::Secondary);
        let orbit_drag = self.view.cad_mouse && middle && !modifiers.shift;
        let pan_drag = if self.view.cad_mouse {
            (middle && modifiers.shift) || secondary
        } else {
            middle || secondary
        };
        if (orbit_drag || pan_drag) && !chrome_owned {
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
        // Viewport chrome owns pointer buttons, not navigation gestures. A
        // press that starts on the triad or navigator (and its drag/release)
        // must not pick the model behind it, but a trackpad pan/orbit, pinch
        // or wheel over those controls is still a viewport camera gesture.
        // Only real overlay panels own gestures as well; the two rules stay
        // separate instead of reusing one over-broad flag.
        let pointer_button_blocked =
            pointer_over_overlay || chrome_owned || self.chrome_contains_pointer(ui, rect);
        let navigation_blocked = pointer_over_overlay;
        if response.hovered() && !navigation_blocked && gesture.has_navigation() {
            if (gesture.zoom - 1.0).abs() > f32::EPSILON {
                // egui's zoom_delta grows when pinching out; camera distance shrinks.
                self.camera.zoom_at(f64::from(1.0 / gesture.zoom), None);
            } else if gesture.wheel != egui::Vec2::ZERO {
                // Mouse wheel: preserve the existing zoom behaviour.
                self.camera.zoom_at(f64::from((-gesture.wheel.y * 0.0015).exp()), None);
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
        if !self.gpu_active() && (self.dirty || self.texture.is_none() || self.last_size != size) {
            if self.display.is_some() {
                self.render_texture(ui.ctx(), size);
            }
            self.last_size = size;
            self.dirty = false;
        }

        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_edge_selection(&self.selected_edges, self.hovered_edge.as_deref());
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
                self.pick_face(ui, rect, response.clicked(), pointer_button_blocked);
                self.update_edge_selection(ui, rect, response.clicked(), pointer_button_blocked);
                self.draw_face_selection(ui, rect);
                self.draw_face_hover(ui, rect);
                self.draw_comparison_overlay(ui, rect);
                // The feature-input edge field is a quiet guide; selected edges
                // are part of the GPU ribbon pass.
                if !self.gpu_active() {
                    self.draw_edit_ghost_edges(ui, rect);
                }
                self.draw_comparison_edges(ui, rect);
                self.draw_transform_gizmo(ui, rect);
            }
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
        if self.chrome_visible {
            self.axis_gizmo(ui, rect);
            // Top selection dock intentionally disabled; see inspector edit action.
            self.pick_face(ui, rect, response.clicked(), pointer_button_blocked);
            self.update_edge_selection(ui, rect, response.clicked(), pointer_button_blocked);
            self.draw_face_selection(ui, rect);
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
            if self.hovered_edge.take().is_some() {
                ui.ctx().request_repaint();
            }
            return;
        }
        let Some(pointer) = pointer else { return };
        #[cfg(feature = "gpu-viewport")]
        if self.gpu_active() && self.edge_picking_enabled {
            use crate::gpu_pick::{
                EdgeClickAction, PickKind, PickOutcome, accept_pick, edge_click_action,
                resolve_face_id,
            };
            let ppp = ui.ctx().pixels_per_point();
            let physical = [(pointer.x - viewport.min.x) * ppp, (pointer.y - viewport.min.y) * ppp];
            let physical_at = |pointer: egui::Pos2| {
                [(pointer.x - viewport.min.x) * ppp, (pointer.y - viewport.min.y) * ppp]
            };
            let (key, ghost_matrix) = self
                .gpu
                .as_ref()
                .map_or((0, None), |gpu| (gpu.pick_key(), gpu.ghost_edge_matrix()));
            let inputs = (physical, key, ghost_matrix, self.camera);
            let moved = self.last_gpu_pick_inputs != Some(inputs);
            self.last_gpu_pick_inputs = Some(inputs);
            let mut next_click = self.pending_gpu_pick;
            let mut next_empty = self.pending_face_empty;
            let mut pending_click = next_click.map(|(id, _, _)| id);
            let mut pending_empty = next_empty.map(|(id, _, _)| id);
            let mut outcome = PickOutcome::Ignore;
            if let Some(gpu) = self.gpu.as_mut() {
                // A display/ghost change bumps the pick key and drops the old
                // request. Re-ask for the still-wanted click at its own pointer
                // so it can resolve from the current scene.
                if let Some((_, pending_pointer, pending_key)) = next_click
                    && pending_key != key
                {
                    let physical = physical_at(pending_pointer);
                    let id = gpu.request_ghost_pick(physical, 7.0 * ppp, true, PickKind::Edges);
                    next_click = Some((id, pending_pointer, key));
                    pending_click = Some(id);
                }
                if let Some((_, pending_pointer, pending_key)) = next_empty
                    && pending_key != key
                {
                    let physical = physical_at(pending_pointer);
                    let id = gpu.request_ghost_pick(
                        physical,
                        crate::gpu_pick::FACE_PICK_RADIUS_PX * ppp,
                        true,
                        PickKind::Faces,
                    );
                    next_empty = Some((id, pending_pointer, key));
                    pending_empty = Some(id);
                }
                if clicked {
                    let id = gpu.request_ghost_pick(physical, 7.0 * ppp, true, PickKind::Edges);
                    next_click = Some((id, pointer, key));
                    pending_click = Some(id);
                    // A new click supersedes an unfinished empty check.
                    next_empty = None;
                    pending_empty = None;
                    self.latest_hover_pick = None;
                } else if self.pending_gpu_pick.is_none()
                    && self.pending_face_empty.is_none()
                    && (moved || self.latest_hover_pick.is_none())
                {
                    let id = gpu.request_ghost_pick(physical, 7.0 * ppp, false, PickKind::Edges);
                    self.latest_hover_pick = Some(id);
                }
                if let Some(result) = gpu.take_pick_result() {
                    if pending_empty == Some(result.id) && result.commit {
                        // The edge pick found nothing; item 2's face pass says
                        // whether the click was on the body at all.
                        next_empty = None;
                        if resolve_face_id(result.index).is_none() {
                            self.selection_miss = true;
                        }
                        ui.ctx().request_repaint();
                    } else {
                        let click_pointer = next_click.map_or(pointer, |(_, pointer, _)| pointer);
                        outcome = accept_pick(&result, key, pending_click);
                        match outcome {
                            PickOutcome::Hover(index) => {
                                if self.pending_gpu_pick.is_none()
                                    && self.pending_face_empty.is_none()
                                    && Some(result.id) == self.latest_hover_pick
                                {
                                    let id =
                                        crate::gpu_viewport::resolve_pick(&gpu.pick_data(), index)
                                            .map(|(id, _)| id);
                                    if id != self.hovered_edge {
                                        self.hovered_edge = id;
                                        ui.ctx().request_repaint();
                                    }
                                }
                            }
                            PickOutcome::Commit(index) => {
                                next_click = None;
                                match edge_click_action(outcome, index) {
                                    EdgeClickAction::Apply(index) => {
                                        if let Some((edge_id, center)) =
                                            crate::gpu_viewport::resolve_pick(
                                                &gpu.pick_data(),
                                                index,
                                            )
                                        {
                                            if !self.selected_edges.remove(&edge_id) {
                                                self.selected_edges.insert(edge_id.clone());
                                            }
                                            self.edge_picks.push(EdgePick { id: edge_id, center });
                                            ui.ctx().request_repaint();
                                        }
                                    }
                                    EdgeClickAction::CheckBody => {
                                        // Item 2's face pass answers whether the
                                        // click was on the body at all.
                                        let physical = physical_at(click_pointer);
                                        let id = gpu.request_ghost_pick(
                                            physical,
                                            crate::gpu_pick::FACE_PICK_RADIUS_PX * ppp,
                                            true,
                                            PickKind::Faces,
                                        );
                                        next_empty = Some((id, click_pointer, key));
                                    }
                                    EdgeClickAction::Retry | EdgeClickAction::Ignore => {}
                                }
                            }
                            PickOutcome::StaleCommit => {
                                // The pick scene changed under the click. Ask
                                // again for the current scene; the click still
                                // resolves from its own GPU pass.
                                let physical = physical_at(click_pointer);
                                let id = gpu.request_ghost_pick(
                                    physical,
                                    7.0 * ppp,
                                    true,
                                    PickKind::Edges,
                                );
                                next_click = Some((id, click_pointer, key));
                            }
                            PickOutcome::Ignore => {}
                        }
                    }
                }
            }
            self.pending_gpu_pick = next_click;
            self.pending_face_empty = next_empty;
            if std::env::var_os("NODECRAFT_TIMELINE").is_some()
                && matches!(outcome, PickOutcome::Hover(_) | PickOutcome::Commit(_))
            {
                eprintln!("[pick] edge={:?}", self.hovered_edge);
            }
            if self.pending_gpu_pick.is_some()
                || self.pending_face_empty.is_some()
                || self.gpu.as_ref().is_some_and(|gpu| gpu.pick_pending())
            {
                ui.ctx().request_repaint();
            }
            return;
        }
        #[cfg(not(feature = "gpu-viewport"))]
        {
            self.pending_gpu_pick = None;
            self.pending_face_empty = None;
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
    }

    #[cfg(not(feature = "gpu-viewport"))]
    fn apply_edge_click(
        &mut self,
        nearest: Option<(String, [f64; 3])>,
        pointer: egui::Pos2,
        viewport: egui::Rect,
        ui: &mut egui::Ui,
    ) {
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

    #[cfg(not(feature = "gpu-viewport"))]
    fn nearest_edge(
        &self,
        pointer: egui::Pos2,
        viewport: egui::Rect,
        threshold: f32,
    ) -> Option<(String, [f64; 3])> {
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
                    .filter_map(|point| self.project_to_viewport(*point, viewport))
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
                if edge.id.starts_with("preview-edge:") {
                    continue;
                }
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

    /// Project an overlay point through the same camera and preview transform
    /// as the shaded solid, so mesh-derived overlays cannot drift from it.
    fn project_to_viewport(
        &self,
        point: solidcore_record::Vec3,
        viewport: egui::Rect,
    ) -> Option<egui::Pos2> {
        let display = self.display.as_ref()?;
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let point = self.preview_transform.map_or(point, |matrix| {
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

    /// Paint every display triangle whose face id matches. Hover and committed
    /// selection use this one projection/fill path, so a selected face cannot
    /// highlight at a different place than the face under the pointer.
    fn draw_face_ids(
        &self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        colour: egui::Color32,
        matches: impl Fn(&str) -> bool,
    ) {
        let Some(display) = self.display.as_ref() else { return };
        let painter = ui.painter().with_clip_rect(viewport);
        for (index, triangle) in display.mesh.triangles.iter().enumerate() {
            let face = display.mesh.tri_face.get(index).copied().unwrap_or(u32::MAX) as usize;
            let Some(id) = display.face_ids.get(face).map(String::as_str) else {
                continue;
            };
            if !matches(id) {
                continue;
            }
            let points: Vec<egui::Pos2> = triangle
                .iter()
                .filter_map(|corner| {
                    let point = display.mesh.positions.get(*corner as usize)?;
                    self.project_to_viewport(*point, viewport)
                })
                .collect();
            if points.len() != 3 {
                continue;
            }
            let mut mesh = egui::Mesh::default();
            for point in points {
                mesh.colored_vertex(point, colour);
            }
            mesh.add_triangle(0, 1, 2);
            painter.add(egui::Shape::mesh(mesh));
        }
    }

    fn draw_face_selection(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        if self.selected_faces.is_empty() {
            return;
        }
        let theme = crate::theme::get();
        let colour = egui::Color32::from_rgba_unmultiplied(
            theme.accent[0],
            theme.accent[1],
            theme.accent[2],
            150,
        );
        self.draw_face_ids(ui, viewport, colour, |id| self.selected_faces.contains(id));
    }

    fn draw_face_hover(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(hovered) = self.hovered_face.as_deref() else { return };
        let theme = crate::theme::get();
        let colour = egui::Color32::from_rgba_unmultiplied(
            theme.highlight[0],
            theme.highlight[1],
            theme.highlight[2],
            90,
        );
        self.draw_face_ids(ui, viewport, colour, |id| id == hovered);
    }

    fn draw_comparison_overlay(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(comparison) = &display.comparison else { return };
        let theme = crate::theme::get();
        let painter = ui.painter().with_clip_rect(viewport);
        let draw_layer =
            |triangles: &[([solidcore_record::Vec3; 3], [solidcore_record::Vec3; 3])],
             color: egui::Color32| {
                for (points, _) in triangles {
                    let projected: Vec<egui::Pos2> = points
                        .iter()
                        .filter_map(|point| self.project_to_viewport(*point, viewport))
                        .collect();
                    if projected.len() != 3 {
                        continue;
                    }
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
        draw_layer(
            &comparison.added,
            egui::Color32::from_rgba_unmultiplied(add[0], add[1], add[2], overlay_alpha),
        );
        let cut = theme.preview_cut;
        draw_layer(
            &comparison.removed,
            egui::Color32::from_rgba_unmultiplied(cut[0], cut[1], cut[2], overlay_alpha),
        );
    }

    fn draw_comparison_edges(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(comparison) = &display.comparison else { return };
        if comparison.edges.is_empty() {
            return;
        }
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
        self.project_to_viewport(solidcore_record::Vec3::new(x, y, z), viewport)
    }

    fn draw_edit_ghost_edges(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let Some(matrix) = self.edit_ghost_transform else { return };
        if self.edit_ghost_edges.is_empty() {
            return;
        }
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
                    self.project_to_viewport(solidcore_record::Vec3::new(x, y, z), viewport)
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
                    .filter_map(|point| self.project_to_viewport(*point, viewport))
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

    /// CPU fallback for builds with no GPU viewport at all.
    #[cfg(not(feature = "gpu-viewport"))]
    fn pointer_hits_body(&self, pointer: egui::Pos2, viewport: egui::Rect) -> bool {
        let Some(display) = self.display.as_ref() else { return false };
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let x = f64::from(pointer.x - viewport.min.x) - f64::from(self.lens_shift[0]) * width * 0.5;
        let y =
            f64::from(pointer.y - viewport.min.y) + f64::from(self.lens_shift[1]) * height * 0.5;
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
        if !viewport.contains(pointer) || self.dock_rect.is_some_and(|dock| dock.contains(pointer))
        {
            self.hovered_face = None;
            return;
        }
        #[cfg(feature = "gpu-viewport")]
        if self.gpu_active() && self.edge_picking_enabled {
            use crate::gpu_pick::{PickKind, PickOutcome, accept_pick, resolve_face_id};
            let ppp = ui.ctx().pixels_per_point();
            let physical = [(pointer.x - viewport.min.x) * ppp, (pointer.y - viewport.min.y) * ppp];
            let key = self.gpu.as_ref().map_or(0, |gpu| gpu.pick_key());
            let physical_at = |pointer: egui::Pos2| {
                [(pointer.x - viewport.min.x) * ppp, (pointer.y - viewport.min.y) * ppp]
            };
            let mut pending_click = self.pending_gpu_pick.map(|(id, _, _)| id);
            if let Some(gpu) = self.gpu.as_mut() {
                // A scene change drops the old request. Re-ask for an
                // in-flight click so it still resolves from a GPU pass.
                if let Some((_, pending_pointer, pending_key)) = self.pending_gpu_pick
                    && pending_key != key
                {
                    let id = gpu.request_ghost_pick(
                        physical_at(pending_pointer),
                        crate::gpu_pick::FACE_PICK_RADIUS_PX * ppp,
                        true,
                        PickKind::Faces,
                    );
                    self.pending_gpu_pick = Some((id, pending_pointer, key));
                    pending_click = Some(id);
                }
                if clicked {
                    let id = gpu.request_ghost_pick(
                        physical,
                        crate::gpu_pick::FACE_PICK_RADIUS_PX * ppp,
                        true,
                        PickKind::Faces,
                    );
                    self.pending_gpu_pick = Some((id, pointer, key));
                    self.pending_face_empty = None;
                    self.latest_hover_pick = None;
                    pending_click = Some(id);
                } else if self.pending_gpu_pick.is_none() {
                    let id = gpu.request_ghost_pick(
                        physical,
                        crate::gpu_pick::FACE_PICK_RADIUS_PX * ppp,
                        false,
                        PickKind::Faces,
                    );
                    self.latest_hover_pick = Some(id);
                }
                if let Some(result) = gpu.take_pick_result() {
                    match accept_pick(&result, key, pending_click) {
                        PickOutcome::Hover(index) => {
                            if self.pending_gpu_pick.is_none()
                                && Some(result.id) == self.latest_hover_pick
                            {
                                let hovered = resolve_face_id(index);
                                if hovered != self.hovered_face {
                                    self.hovered_face = hovered;
                                    ui.ctx().request_repaint();
                                }
                            }
                        }
                        PickOutcome::Commit(index) => {
                            self.pending_gpu_pick = None;
                            match resolve_face_id(index) {
                                Some(face) => self.face_picks.push(face),
                                None => self.selection_miss = true,
                            }
                            ui.ctx().request_repaint();
                        }
                        PickOutcome::StaleCommit => {
                            // The pick scene changed under the click. The click
                            // resolves only from its own GPU pass; there is no
                            // CPU fallback to substitute.
                            self.pending_gpu_pick = None;
                        }
                        PickOutcome::Ignore => {}
                    }
                }
            }
            if self.pending_gpu_pick.is_some()
                || self.gpu.as_ref().is_some_and(|gpu| gpu.pick_pending())
            {
                ui.ctx().request_repaint();
            }
            return;
        }
        #[cfg(not(feature = "gpu-viewport"))]
        self.pick_face_cpu(ui, viewport, pointer, clicked);
        #[cfg(feature = "gpu-viewport")]
        if !self.gpu_active() && self.edge_picking_enabled {
            // A build with the GPU viewport can still run without an adapter
            // (headless tests, driver fallback). Picking then uses the same CPU
            // raycast as a build compiled without the feature.
            self.pick_face_cpu(ui, viewport, pointer, clicked);
        }
    }

    fn pick_face_cpu(
        &mut self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        pointer: egui::Pos2,
        clicked: bool,
    ) {
        let Some(display) = self.display.as_ref() else { return };
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let x = f64::from(pointer.x - viewport.min.x) - f64::from(self.lens_shift[0]) * width * 0.5;
        let y =
            f64::from(pointer.y - viewport.min.y) + f64::from(self.lens_shift[1]) * height * 0.5;
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

    /// SolidCraft's Axes view navigator in the viewport's top-right corner:
    /// six signed world axes and a drag anywhere on the navigator to orbit.
    /// Clicking a tip looks along that axis. NodeCraft's Home button sits
    /// under the compass, above the projection and display controls.
    ///
    /// Adapted from SolidCraft's view compass and `view.navigate` (synced with
    /// c2ffd48e, MIT OR Apache-2.0). NodeCraft keeps its own placement,
    /// camera, lens shift, theme and viewport chrome below the compass.
    fn axis_gizmo(&mut self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let theme = crate::theme::get();
        let chrome = self.navigator_chrome(viewport);
        let layout = chrome.layout;

        // A drag anywhere in the navigator's reserved box orbits; a stationary
        // press on a tip is the tip's click in `show`.
        let may_start = ui.is_enabled() && ui.rect_contains_pointer(layout.outer);
        let mut drag = self.navigator_drag;
        let orbit = crate::view_navigator::orbit(ui, &mut drag, may_start, |p| {
            crate::view_navigator::contains(&layout, p)
        });
        self.navigator_drag = drag;
        if let Some(step) = orbit.step
            && (step.dx != 0.0 || step.dy != 0.0)
        {
            self.camera.orbit(step.dx, step.dy);
            self.dirty = true;
            ui.ctx().request_repaint();
        }

        let hover = ui
            .input(|i| i.pointer.hover_pos())
            .filter(|p| ui.is_enabled() && ui.rect_contains_pointer(layout.outer));
        let compass = crate::view_navigator::Compass {
            clip: viewport,
            centre: layout.centre,
            axes: crate::view_navigator::view_axes(&self.camera),
            hover,
            orbiting: orbit.orbiting,
        };
        // The compass returns a tip or Home; its drag came from `orbit` above.
        match crate::view_navigator::show(ui, &compass) {
            Some(crate::view_navigator::ViewAction::Snap { direction, .. }) => {
                // NodeCraft has no in-flight camera glide, so the current
                // camera is the "from" a rapid second click starts from.
                self.camera = crate::view_navigator::snap(&self.camera, direction);
                self.dirty = true;
                ui.ctx().request_repaint();
            }
            None => {}
        }

        let painter = ui.painter().with_clip_rect(viewport);
        let icon_size = 18.0;

        // Home / isometric button directly under the compass, as before the
        // navigator port. Its right edge shares the compass box's right edge
        // and its 30-point hit area carries only a hover highlight.
        let icon_rect = navigator_home_icon_rect(layout.outer);
        let home_rect = chrome.home;
        let home = ui
            .interact(home_rect, egui::Id::new("viewport-axis-home"), egui::Sense::click())
            .on_hover_text("Home / isometric view");
        if home.hovered() {
            painter.circle_filled(icon_rect.center(), 14.0, theme.with_alpha(theme.highlight, 26));
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

        // Projection toggle directly below the Home button.
        let perspective_icon = navigator_projection_icon_rect(layout.outer);
        let perspective_rect = chrome.projection;
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
        let view_icon = navigator_view_icon_rect(layout.outer);
        let view_rect = chrome.display;
        let view = ui
            .interact(view_rect, egui::Id::new("viewport-view-modes"), egui::Sense::click())
            .on_hover_text("Display modes");
        if view.hovered() {
            painter.circle_filled(view_icon.center(), 14.0, theme.with_alpha(theme.highlight, 26));
            painter.circle_stroke(
                view_icon.center(),
                14.0,
                egui::Stroke::new(1.0, theme.with_alpha(theme.muted, 100)),
            );
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
        self.chrome_blockers = chrome.rects().to_vec();

        // One drag state owns the cursor: hide it only while the press has
        // passed the drag threshold; a release or a lost drag shows it again.
        if self.navigator_drag.moving {
            ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        } else if orbit.restored {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Default);
        }
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
                        egui::RichText::new(format!("{value:.decimals$}")).size(11.0).monospace(),
                    ),
                );
                ui.spacing_mut().slider_width = 118.0;
                let previous_style = (**ui.style()).clone();
                ui.style_mut().text_styles.insert(
                    egui::TextStyle::Body,
                    egui::FontId::proportional(Self::SETTINGS_SLIDER_FONT_SIZE),
                );
                ui.spacing_mut().interact_size.y = Self::SETTINGS_SLIDER_HEIGHT;
                changed = ui.add(egui::Slider::new(value, range).show_value(false)).changed();
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
                // Temporary tick row: to be replaced by the icon grid toggle (user-approved).
                changed |= crate::theme::menu_toggle(ui, &mut edited.grid, "XY grid");
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
                changed |=
                    Self::settings_slider(ui, "Roughness", &mut edited.roughness, 0.045..=1.0, 2);
                changed |=
                    Self::settings_slider(ui, "Metallic", &mut edited.metallic, 0.0..=1.0, 2);
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
                egui::pos2(
                    crate::gpu_backdrop::INNER_UV_MIN[0],
                    crate::gpu_backdrop::INNER_UV_MIN[1],
                ),
                egui::pos2(
                    crate::gpu_backdrop::INNER_UV_MAX[0],
                    crate::gpu_backdrop::INNER_UV_MAX[1],
                ),
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

        let scene = scene_for(display, &self.camera, self.lod_index, &self.view);
        let canvas = solidcraft_viewport::render(&scene, &self.camera, width, height);
        let image = egui::ColorImage::from_rgba_unmultiplied([canvas.w, canvas.h], &canvas.rgba);
        self.texture =
            Some(ctx.load_texture("nodecraft-viewport", image, egui::TextureOptions::LINEAR));
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

fn scene_for(
    display: &DisplayMesh,
    camera: &Camera,
    lod_index: usize,
    view: &ViewSettings,
) -> Scene {
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
    if draws_grid(view) {
        add_grid(&mut scene, display.radius, VIEW_ORIGIN, camera.fov > 0.0, ortho_fade);
    }
    scene.meshes.push(SceneMesh::solid(
        std::sync::Arc::new(display.render_mesh(lod_index).clone()),
        Rgb(151, 162, 176),
    ));
    for (index, edge) in display.mesh.edges.iter().enumerate() {
        let seam = display.mesh.seams.get(index).copied().unwrap_or(false);
        if seam && !view.show_seam_edges {
            continue;
        }
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

    struct AxisFrame {
        cursor: egui::CursorIcon,
        commands: Vec<egui::ViewportCommand>,
    }

    fn run_axis_frame_focused(
        ctx: &egui::Context,
        rect: egui::Rect,
        viewport: &mut Viewport,
        events: Vec<egui::Event>,
        focused: bool,
    ) -> AxisFrame {
        let mut output = ctx.run_ui(
            egui::RawInput { screen_rect: Some(rect), events, focused, ..Default::default() },
            |ui| viewport.axis_gizmo(ui, rect),
        );
        let cursor = output.platform_output.cursor_icon;
        let commands = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|viewport| viewport.commands.clone())
            .unwrap_or_default();
        output.textures_delta.clear();
        AxisFrame { cursor, commands }
    }

    fn run_axis_frame(
        ctx: &egui::Context,
        rect: egui::Rect,
        viewport: &mut Viewport,
        events: Vec<egui::Event>,
    ) -> AxisFrame {
        run_axis_frame_focused(ctx, rect, viewport, events, true)
    }

    fn navigator_press(at: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    fn cursor_warps(frame: &AxisFrame) -> Vec<egui::Pos2> {
        frame
            .commands
            .iter()
            .filter_map(|command| match command {
                egui::ViewportCommand::CursorPosition(at) => Some(*at),
                _ => None,
            })
            .collect()
    }

    fn navigator_drag_target(viewport: &Viewport, rect: egui::Rect) -> egui::Pos2 {
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        layout.centre + egui::vec2(35.0, 35.0)
    }

    fn test_quad_mesh() -> NodeMesh {
        NodeMesh {
            vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            ..Default::default()
        }
    }

    fn test_gizmo() -> Gizmo {
        Gizmo {
            node_id: "move".into(),
            pivot: [0.0, 0.0, 0.0],
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rx: 0.0,
            ry: 0.0,
            rz: 0.0,
            world_to_local: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            local_axes: false,
        }
    }

    /// A viewport with a display and a triad, so the full show path runs the
    /// same picking and chrome geometry as the app.
    fn framed_viewport() -> Viewport {
        let mut viewport = Viewport::default();
        viewport.set_mesh(Some(&test_quad_mesh()));
        viewport.set_transform_gizmo(Some(test_gizmo()));
        viewport
    }

    fn test_rect() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0))
    }

    fn pointer_button(at: egui::Pos2, button: egui::PointerButton, pressed: bool) -> egui::Event {
        egui::Event::PointerButton { pos: at, button, pressed, modifiers: Default::default() }
    }

    fn run_viewport_frame(
        ctx: &egui::Context,
        rect: egui::Rect,
        viewport: &mut Viewport,
        events: Vec<egui::Event>,
    ) {
        run_viewport_frame_blocked(ctx, rect, viewport, egui::Rect::NOTHING, &[], events);
    }

    fn run_viewport_frame_blocked(
        ctx: &egui::Context,
        rect: egui::Rect,
        viewport: &mut Viewport,
        overlay: egui::Rect,
        blockers: &[egui::Rect],
        events: Vec<egui::Event>,
    ) {
        let mut output = ctx.run_ui(
            egui::RawInput { screen_rect: Some(rect), events, focused: true, ..Default::default() },
            |ui| {
                viewport.show_with_backdrop_blockers(ui, overlay, blockers, rect);
            },
        );
        output.textures_delta.clear();
    }

    fn drag_from(
        ctx: &egui::Context,
        rect: egui::Rect,
        viewport: &mut Viewport,
        start: egui::Pos2,
        end: egui::Pos2,
    ) {
        let middle = egui::PointerButton::Middle;
        run_viewport_frame(ctx, rect, viewport, vec![egui::Event::PointerMoved(start)]);
        run_viewport_frame(ctx, rect, viewport, vec![pointer_button(start, middle, true)]);
        run_viewport_frame(
            ctx,
            rect,
            viewport,
            vec![egui::Event::PointerMoved(start.lerp(end, 0.5))],
        );
        run_viewport_frame(ctx, rect, viewport, vec![egui::Event::PointerMoved(end)]);
        run_viewport_frame(ctx, rect, viewport, vec![pointer_button(end, middle, false)]);
    }

    /// The navigation gestures winit/egui deliver without a pointer button:
    /// two-finger pan, two-finger orbit (shift-pan), pinch and mouse wheel.
    #[derive(Clone, Copy, Debug)]
    enum ViewGesture {
        Pan,
        Orbit,
        Pinch,
        Wheel,
    }

    impl ViewGesture {
        fn name(self) -> &'static str {
            match self {
                Self::Pan => "two-finger pan",
                Self::Orbit => "two-finger orbit",
                Self::Pinch => "pinch",
                Self::Wheel => "wheel",
            }
        }

        fn events(self) -> Vec<egui::Event> {
            let drag = |unit, delta| egui::Event::MouseWheel {
                unit,
                delta,
                phase: egui::TouchPhase::Move,
                modifiers: Default::default(),
            };
            match self {
                Self::Pan => vec![drag(egui::MouseWheelUnit::Point, egui::vec2(12.0, -6.0))],
                Self::Orbit => vec![
                    egui::Event::ModifiersChanged(egui::Modifiers {
                        shift: true,
                        ..Default::default()
                    }),
                    drag(egui::MouseWheelUnit::Point, egui::vec2(9.0, 4.0)),
                ],
                Self::Pinch => vec![egui::Event::Zoom(1.25)],
                Self::Wheel => vec![drag(egui::MouseWheelUnit::Line, egui::vec2(0.0, -2.0))],
            }
        }
    }

    /// Run one navigation gesture with the pointer resting at `at`, after the
    /// two settling frames the app gets from moving the pointer there. Returns
    /// the camera before and after, so a test can prove the gesture was live
    /// without assuming the fitted starting camera.
    fn gesture_camera(rect: egui::Rect, at: egui::Pos2, gesture: ViewGesture) -> (Camera, Camera) {
        gesture_camera_blocked(rect, at, gesture, egui::Rect::NOTHING, &[])
    }

    fn gesture_camera_blocked(
        rect: egui::Rect,
        at: egui::Pos2,
        gesture: ViewGesture,
        overlay: egui::Rect,
        blockers: &[egui::Rect],
    ) -> (Camera, Camera) {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let mut viewport = framed_viewport();
        run_viewport_frame_blocked(
            &ctx,
            rect,
            &mut viewport,
            overlay,
            blockers,
            vec![egui::Event::PointerMoved(at)],
        );
        run_viewport_frame_blocked(&ctx, rect, &mut viewport, overlay, blockers, Vec::new());
        let before = viewport.camera;
        run_viewport_frame_blocked(&ctx, rect, &mut viewport, overlay, blockers, gesture.events());
        (before, viewport.camera)
    }

    fn gesture_controls(rect: egui::Rect, viewport: &Viewport) -> Vec<(&'static str, egui::Pos2)> {
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        vec![
            ("gizmo centre ball", rect.center()),
            ("compass", chrome.layout.centre),
            ("navigator drag box", navigator_drag_target(viewport, rect)),
            ("Home", chrome.home.center()),
            ("projection toggle", chrome.projection.center()),
            ("display cog", chrome.display.center()),
        ]
    }

    #[test]
    fn every_navigator_control_blocks_viewport_picking() {
        let viewport = Viewport::default();
        let rect = test_rect();
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        for point in [
            chrome.layout.centre,
            chrome.layout.outer.min,
            chrome.home.center(),
            chrome.projection.center(),
            chrome.display.center(),
        ] {
            assert!(
                viewport.navigator_pointer_blocked(rect, point),
                "the navigator's drawn control at {point:?} must own the press",
            );
        }
        assert!(!viewport.navigator_pointer_blocked(rect, egui::pos2(20.0, 20.0)));
    }

    #[test]
    fn drawing_and_hit_testing_share_one_navigator_chrome_layout() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = test_rect();
        let mut viewport = Viewport::default();
        run_axis_frame(&ctx, rect, &mut viewport, Vec::new());
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        assert_eq!(
            viewport.interaction_blockers(),
            chrome.rects(),
            "the blockers must be exactly the rects axis_gizmo draws",
        );
        for point in [chrome.home.center(), chrome.projection.center(), chrome.display.center()] {
            assert!(
                viewport.navigator_pointer_blocked(rect, point),
                "the drawn control at {point:?} must also block picking",
            );
        }
    }

    #[test]
    fn a_middle_drag_starting_on_viewport_chrome_never_orbits_the_camera() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = test_rect();
        let mut viewport = framed_viewport();
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        let controls = [
            ("gizmo centre ball", rect.center()),
            ("navigator tip", chrome.layout.centre),
            ("navigator box", navigator_drag_target(&viewport, rect)),
            ("Home", chrome.home.center()),
            ("projection toggle", chrome.projection.center()),
            ("display cog", chrome.display.center()),
        ];
        for (name, start) in controls {
            let before = viewport.camera;
            drag_from(&ctx, rect, &mut viewport, start, rect.min + egui::vec2(40.0, 40.0));
            assert_eq!(
                viewport.camera, before,
                "a camera drag that starts on {name} must not orbit or pan the viewport",
            );
        }
    }

    #[test]
    fn chrome_ownership_is_taken_at_the_press_and_held_to_the_release() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = test_rect();
        let mut viewport = framed_viewport();
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        let controls = [
            ("gizmo centre ball", rect.center()),
            ("navigator tip", chrome.layout.centre),
            ("navigator box", navigator_drag_target(&viewport, rect)),
            ("Home", chrome.home.center()),
            ("projection toggle", chrome.projection.center()),
            ("display cog", chrome.display.center()),
        ];
        let primary = egui::PointerButton::Primary;
        for (name, start) in controls {
            run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(start, primary, true)],
            );
            assert_eq!(
                viewport.chrome_pointer(),
                Some((primary, start, true)),
                "the press on {name} is owned by chrome",
            );
            let elsewhere = rect.min + egui::vec2(40.0, 40.0);
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![egui::Event::PointerMoved(elsewhere)],
            );
            assert!(
                viewport.pointer_owned_by_chrome(),
                "{name}: ownership must survive moving the pointer into the viewport",
            );
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(elsewhere, primary, false)],
            );
            assert!(
                viewport.pointer_owned_by_chrome(),
                "{name}: the release frame still sees chrome ownership",
            );
            run_viewport_frame(&ctx, rect, &mut viewport, Vec::new());
            assert!(!viewport.pointer_owned_by_chrome(), "{name}: the next gesture is free");
        }

        // A press in the viewport is still a viewport gesture, even if the
        // pointer later moves over chrome.
        let start = rect.min + egui::vec2(80.0, 80.0);
        run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
        run_viewport_frame(&ctx, rect, &mut viewport, vec![pointer_button(start, primary, true)]);
        assert_eq!(viewport.chrome_pointer(), Some((primary, start, false)));
        run_viewport_frame(
            &ctx,
            rect,
            &mut viewport,
            vec![egui::Event::PointerMoved(rect.center())],
        );
        assert!(!viewport.pointer_owned_by_chrome());
    }

    #[cfg(not(feature = "gpu-viewport"))]
    #[test]
    fn primary_clicks_on_viewport_chrome_never_pick_or_clear_selection() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = test_rect();
        let mut viewport = framed_viewport();
        viewport.set_selection_mode(SelectionMode::Face);
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        let primary = egui::PointerButton::Primary;
        let gizmo_point = rect.center();
        // Clear of the triad's 110 px arrows, rings and ball, but still on the
        // body, so this point is a normal viewport click.
        let viewport_point = rect.center() + egui::vec2(0.0, 120.0);
        let controls = [
            ("gizmo centre ball", gizmo_point),
            ("navigator tip", chrome.layout.centre),
            ("navigator box", navigator_drag_target(&viewport, rect)),
            ("Home", chrome.home.center()),
            ("projection toggle", chrome.projection.center()),
            ("display cog", chrome.display.center()),
        ];
        let click = |ctx: &egui::Context, viewport: &mut Viewport, at: egui::Pos2| {
            run_viewport_frame(ctx, rect, viewport, vec![egui::Event::PointerMoved(at)]);
            run_viewport_frame(ctx, rect, viewport, vec![pointer_button(at, primary, true)]);
            run_viewport_frame(ctx, rect, viewport, vec![pointer_button(at, primary, false)]);
        };
        for (name, point) in controls {
            viewport.take_face_picks();
            viewport.take_selection_miss();
            click(&ctx, &mut viewport, point);
            assert!(
                viewport.take_face_picks().is_empty(),
                "{name}: a click on chrome must not pick a face",
            );
            assert!(
                !viewport.take_selection_miss(),
                "{name}: a click on chrome must not clear the selection or edit target",
            );
        }

        // The click threshold is wider than each control's hit test, so a
        // short drag can start on a control and end just outside it (or the
        // reverse) while egui still calls it a click. The press owner must
        // hold, and the release over a control must not pick either.
        let short = [
            (
                "gizmo ball edge",
                gizmo_point + egui::vec2(-8.0, 0.0),
                gizmo_point + egui::vec2(-13.0, 0.0),
            ),
            (
                "navigator box edge",
                chrome.layout.outer.min + egui::vec2(1.0, 1.0),
                chrome.layout.outer.min - egui::vec2(2.0, 2.0),
            ),
            (
                "Home edge",
                chrome.home.center() + egui::vec2(14.0, 0.0),
                chrome.home.center() + egui::vec2(19.0, 0.0),
            ),
            (
                "projection edge",
                chrome.projection.center() + egui::vec2(14.0, 0.0),
                chrome.projection.center() + egui::vec2(19.0, 0.0),
            ),
            (
                "display edge",
                chrome.display.center() + egui::vec2(14.0, 0.0),
                chrome.display.center() + egui::vec2(19.0, 0.0),
            ),
            (
                "Home edge from viewport",
                chrome.home.center() - egui::vec2(19.0, 0.0),
                chrome.home.center() - egui::vec2(14.0, 0.0),
            ),
            (
                "projection edge from viewport",
                chrome.projection.center() - egui::vec2(19.0, 0.0),
                chrome.projection.center() - egui::vec2(14.0, 0.0),
            ),
            (
                "display edge from viewport",
                chrome.display.center() - egui::vec2(19.0, 0.0),
                chrome.display.center() - egui::vec2(14.0, 0.0),
            ),
        ];
        for (name, press, release) in short {
            viewport.take_face_picks();
            viewport.take_selection_miss();
            run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(press)]);
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(press, primary, true)],
            );
            run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(release)]);
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(release, primary, false)],
            );
            assert!(
                viewport.take_face_picks().is_empty(),
                "{name}: a short drag across the control edge must not pick a face",
            );
            assert!(
                !viewport.take_selection_miss(),
                "{name}: a short drag across the control edge must not clear the selection",
            );
        }

        // Press on chrome, release in the viewport: the gesture still belongs
        // to the chrome, so no pick is requested at the release point.
        for (name, point) in
            [("Home", chrome.home.center()), ("projection toggle", chrome.projection.center())]
        {
            viewport.take_face_picks();
            viewport.take_selection_miss();
            run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(point)]);
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(point, primary, true)],
            );
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![egui::Event::PointerMoved(viewport_point)],
            );
            run_viewport_frame(
                &ctx,
                rect,
                &mut viewport,
                vec![pointer_button(viewport_point, primary, false)],
            );
            assert!(
                viewport.take_face_picks().is_empty(),
                "{name}: a press on chrome must not pick when released over the model",
            );
            assert!(!viewport.take_selection_miss());
        }

        // Press in the viewport, release on chrome: the release is over a
        // control, so it must not be treated as a viewport click either.
        viewport.take_face_picks();
        viewport.take_selection_miss();
        let point = chrome.projection.center();
        run_viewport_frame(
            &ctx,
            rect,
            &mut viewport,
            vec![egui::Event::PointerMoved(viewport_point)],
        );
        run_viewport_frame(
            &ctx,
            rect,
            &mut viewport,
            vec![pointer_button(viewport_point, primary, true)],
        );
        run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(point)]);
        run_viewport_frame(&ctx, rect, &mut viewport, vec![pointer_button(point, primary, false)]);
        assert!(viewport.take_face_picks().is_empty());
        assert!(!viewport.take_selection_miss());

        // A plain click elsewhere still picks normally.
        viewport.take_face_picks();
        viewport.take_selection_miss();
        click(&ctx, &mut viewport, viewport_point);
        assert!(
            !viewport.take_face_picks().is_empty(),
            "a click in the viewport must still pick the model",
        );
    }

    #[test]
    fn a_camera_drag_that_starts_in_the_viewport_still_orbits() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = test_rect();
        let mut viewport = framed_viewport();
        let before = viewport.camera;
        let start = rect.min + egui::vec2(80.0, 80.0);
        drag_from(&ctx, rect, &mut viewport, start, rect.center());
        assert_ne!(
            viewport.camera, before,
            "a drag that starts in the viewport keeps navigating even if it ends on chrome",
        );
    }

    #[test]
    fn navigation_gestures_over_viewport_chrome_move_the_camera_like_empty_viewport() {
        let rect = test_rect();
        let viewport = Viewport::default();
        let controls = gesture_controls(rect, &viewport);
        let empty = rect.min + egui::vec2(20.0, 20.0);
        for gesture in
            [ViewGesture::Pan, ViewGesture::Orbit, ViewGesture::Pinch, ViewGesture::Wheel]
        {
            let (empty_before, expected) = gesture_camera(rect, empty, gesture);
            assert_ne!(
                expected,
                empty_before,
                "{} over empty viewport must move the camera",
                gesture.name(),
            );
            for (name, at) in &controls {
                let (control_before, actual) = gesture_camera(rect, *at, gesture);
                assert_eq!(
                    control_before, empty_before,
                    "each run must start from the same fitted camera",
                );
                assert_eq!(
                    actual,
                    expected,
                    "a {} over {} must navigate the viewport exactly like empty space",
                    gesture.name(),
                    name,
                );
            }
        }
    }

    #[test]
    fn navigation_gestures_over_overlay_panels_still_do_not_move_the_camera() {
        let rect = test_rect();
        let overlay = egui::Rect::from_min_max(egui::pos2(300.0, rect.min.y), rect.max);
        let panel = egui::Rect::from_min_max(egui::pos2(600.0, 350.0), egui::pos2(780.0, 580.0));
        let at = panel.center();
        for gesture in
            [ViewGesture::Pan, ViewGesture::Orbit, ViewGesture::Pinch, ViewGesture::Wheel]
        {
            let (open_before, open) = gesture_camera(rect, at, gesture);
            assert_ne!(
                open,
                open_before,
                "{}: the same point off the panel must be a live viewport gesture",
                gesture.name(),
            );
            let (blocked_before, blocked) =
                gesture_camera_blocked(rect, at, gesture, overlay, &[panel]);
            assert_eq!(
                blocked,
                blocked_before,
                "a {} over an overlay panel must not move the camera",
                gesture.name(),
            );
        }
    }

    #[test]
    fn wheel_and_pinch_over_the_navigator_do_not_act_on_the_navigator() {
        let rect = test_rect();
        let viewport = Viewport::default();
        let chrome = NavigatorChrome::at(rect, viewport.chrome_right_inset);
        let controls = [
            ("compass", chrome.layout.centre),
            ("navigator drag box", navigator_drag_target(&viewport, rect)),
            ("Home", chrome.home.center()),
            ("projection toggle", chrome.projection.center()),
            ("display cog", chrome.display.center()),
        ];
        for gesture in [ViewGesture::Pinch, ViewGesture::Wheel] {
            for (name, at) in controls {
                let ctx = egui::Context::default();
                egui_extras::install_image_loaders(&ctx);
                let mut viewport = framed_viewport();
                viewport.camera.yaw = 0.4;
                viewport.camera.pitch = 0.2;
                let before = viewport.camera;
                let before_view = viewport.view;
                let before_drag = viewport.navigator_drag;
                run_viewport_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(at)]);
                run_viewport_frame(&ctx, rect, &mut viewport, Vec::new());
                run_viewport_frame(&ctx, rect, &mut viewport, gesture.events());
                assert_eq!(
                    viewport.navigator_drag,
                    before_drag,
                    "a {} over {} must not start or turn the navigator",
                    gesture.name(),
                    name,
                );
                assert_eq!(
                    viewport.view,
                    before_view,
                    "a {} over {} must not change the display settings",
                    gesture.name(),
                    name,
                );
                assert_eq!(
                    viewport.camera.target,
                    before.target,
                    "a {} over {} must not snap or orbit the view",
                    gesture.name(),
                    name,
                );
                assert_eq!(viewport.camera.yaw, before.yaw);
                assert_eq!(viewport.camera.pitch, before.pitch);
                assert_eq!(viewport.camera.fov, before.fov);
                assert_ne!(
                    viewport.camera.distance,
                    before.distance,
                    "the viewport still receives the {} over {}",
                    gesture.name(),
                    name,
                );
            }
        }
    }

    #[test]
    fn the_viewport_no_longer_paints_debug_text_overlays() {
        // The Debug window is the only place the camera/GPU diagnostics appear.
        // This guards against the old painter-only overlay coming back.
        let source = include_str!("viewport.rs");
        // Build the names so this test does not match its own literals.
        let layer = ["viewport", "-debug", "-overlay"].concat();
        let draw = ["draw", "_debug", "_overlays"].concat();
        let lines = ["gpu", "_info", "_lines"].concat();
        assert!(!source.contains(&layer));
        assert!(!source.contains(&draw));
        assert!(!source.contains(&lines));
    }

    #[test]
    fn debug_rows_read_the_live_camera_and_existing_gpu_snapshot() {
        let viewport = Viewport::default();
        let camera = viewport.camera_debug_rows();
        assert!(camera.iter().any(|row| row.label == "Yaw"));
        assert_eq!(camera.iter().filter(|row| row.label == "View row").count(), 4);
        let gpu = viewport.gpu_debug_rows(16.7);
        assert!(gpu.iter().any(|row| row.label == "Frame" && row.value.contains("16.7 ms")));
        assert!(gpu.iter().any(|row| row.label == "Tessellation warnings"));
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
    fn navigator_hit_area_blocks_viewport_picking() {
        let viewport = Viewport::default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        assert_eq!(layout.outer.width(), crate::view_navigator::AXIS_VIEW_SIZE);
        assert!(viewport.navigator_pointer_blocked(rect, layout.centre));
        assert!(
            viewport.navigator_pointer_blocked(rect, navigator_home_rect(layout.outer).center())
        );
        assert!(viewport.navigator_pointer_blocked(rect, layout.outer.min));
        assert!(viewport.navigator_pointer_blocked(rect, layout.outer.max));
        assert!(!viewport.navigator_pointer_blocked(rect, layout.outer.min - egui::vec2(1.0, 0.0)));
        assert!(!viewport.navigator_pointer_blocked(rect, egui::pos2(20.0, 20.0)));

        let mut hidden = Viewport::default();
        hidden.set_chrome_visible(false);
        assert!(!hidden.navigator_pointer_blocked(rect, layout.centre));
    }

    #[test]
    fn axis_gizmo_click_uses_the_ported_compass_and_records_its_blockers() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        viewport.last_full = Some(rect);
        viewport.last_visible = Some(rect);
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        let axes = crate::view_navigator::view_axes(&viewport.camera);
        let x_tip = crate::view_navigator::tips(&axes, layout.centre)
            .into_iter()
            .find(|tip| tip.0 == "x")
            .expect("x tip")
            .2;

        let hover =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(x_tip)]);
        assert_eq!(hover.cursor, egui::CursorIcon::PointingHand);
        let press = run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(x_tip, true)]);
        let release =
            run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(x_tip, false)]);
        assert_ne!(press.cursor, egui::CursorIcon::None);
        assert_ne!(release.cursor, egui::CursorIcon::None);
        assert!(cursor_warps(&release).is_empty(), "a plain tip click never warps the pointer");
        assert!(!viewport.navigator_drag.moving);
        assert!(
            viewport.camera.back().dist(solidcore_record::Vec3::X) < 1e-5,
            "the X tip looks from +X: {:?}",
            viewport.camera.back()
        );
        let blockers = viewport.interaction_blockers();
        assert!(blockers.contains(&layout.outer), "the whole navigator box blocks picking");
        let home_rect = navigator_home_rect(layout.outer);
        assert!(blockers.contains(&home_rect), "the Home button blocks picking");

        // Home returns to the isometric framing through our camera path.
        let hover = run_axis_frame(
            &ctx,
            rect,
            &mut viewport,
            vec![egui::Event::PointerMoved(home_rect.center())],
        );
        assert_eq!(hover.cursor, egui::CursorIcon::PointingHand);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(home_rect.center(), true)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(home_rect.center(), false)]);
        assert_eq!(viewport.camera.yaw, Camera::default().yaw);
        assert_eq!(viewport.camera.pitch, Camera::default().pitch);
        assert!(!viewport.navigator_drag.moving);
    }

    fn expected_home_rect(outer: egui::Rect) -> egui::Rect {
        let icon = egui::Rect::from_min_size(
            egui::pos2(outer.right() - 18.0, outer.bottom() + 10.0),
            egui::vec2(18.0, 18.0),
        );
        egui::Rect::from_center_size(icon.center(), egui::vec2(30.0, 30.0))
    }

    fn expected_perspective_rect(home: egui::Rect, outer: egui::Rect) -> egui::Rect {
        let icon = egui::Rect::from_min_size(
            egui::pos2(outer.right() - 18.0, home.bottom() + 10.0),
            egui::vec2(18.0, 18.0),
        );
        egui::Rect::from_center_size(icon.center(), egui::vec2(30.0, 30.0))
    }

    #[test]
    fn home_button_sits_under_the_navigator_and_above_the_projection_toggle() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        run_axis_frame(&ctx, rect, &mut viewport, Vec::new());
        let blockers = viewport.interaction_blockers();
        assert_eq!(blockers.len(), 4, "compass, Home, projection and display: {blockers:?}");
        let home = blockers[1];
        let perspective = blockers[2];
        assert_eq!(home, expected_home_rect(layout.outer));
        assert_eq!(perspective, expected_perspective_rect(home, layout.outer));
        let icon = egui::Rect::from_center_size(home.center(), egui::vec2(18.0, 18.0));
        assert_eq!(
            icon.right(),
            layout.outer.right(),
            "the icon shares the compass box's right edge"
        );
        assert_eq!(icon.top(), layout.outer.bottom() + 10.0, "10 points below the compass box");
        let perspective_icon =
            egui::Rect::from_center_size(perspective.center(), egui::vec2(18.0, 18.0));
        assert_eq!(
            perspective_icon.top(),
            home.bottom() + 10.0,
            "the projection icon follows Home's 30-point rect by 10 points"
        );
    }

    #[test]
    fn clicking_the_home_button_frames_the_isometric_view() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        viewport.last_full = Some(rect);
        viewport.last_visible = Some(rect);
        viewport.camera.yaw = 0.7;
        viewport.camera.pitch = 0.2;
        viewport.camera.distance = 900.0;
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        let home = expected_home_rect(layout.outer);
        run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(home.center())]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(home.center(), true)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(home.center(), false)]);
        assert_eq!(viewport.camera.yaw, Camera::default().yaw);
        assert_eq!(viewport.camera.pitch, Camera::default().pitch);
        assert_eq!(viewport.camera.distance, Camera::default().distance);
    }

    #[test]
    fn the_home_button_blocks_viewport_picking() {
        let viewport = Viewport::default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        let home = expected_home_rect(layout.outer);
        assert!(viewport.navigator_pointer_blocked(rect, home.center()));
    }

    #[test]
    fn the_compass_no_longer_owns_a_home_click() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        viewport.last_full = Some(rect);
        viewport.last_visible = Some(rect);
        viewport.camera.yaw = 0.7;
        viewport.camera.pitch = 0.2;
        viewport.camera.distance = 900.0;
        let before = viewport.camera;
        let layout = crate::view_navigator::layout(rect, viewport.chrome_right_inset);
        let old_chip = layout.centre + egui::vec2(-37.0, -37.0);
        run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(old_chip)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(old_chip, true)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(old_chip, false)]);
        assert_eq!(viewport.camera, before, "the compass has no Home hit area");
    }

    #[test]
    fn navigator_hover_shows_the_grab_cursor_over_its_box() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        let at = navigator_drag_target(&viewport, rect);
        let frame = run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(at)]);
        assert_eq!(frame.cursor, egui::CursorIcon::Grab);
    }

    #[test]
    fn navigator_drag_hides_the_pointer_and_release_warp_restores_it() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        let start = navigator_drag_target(&viewport, rect);
        let moved = start + egui::vec2(45.0, 8.0);
        let press = |at, pressed| navigator_press(at, pressed);

        let hover =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
        assert_eq!(hover.cursor, egui::CursorIcon::Grab);
        let pressed = run_axis_frame(&ctx, rect, &mut viewport, vec![press(start, true)]);
        assert_ne!(pressed.cursor, egui::CursorIcon::None, "a press alone shows the pointer");
        assert!(!viewport.navigator_drag.moving, "the threshold has not been passed");

        let dragging =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(moved)]);
        assert_eq!(dragging.cursor, egui::CursorIcon::None, "the orbit drag hides the pointer");
        assert!(viewport.navigator_drag.moving);
        assert_eq!(viewport.navigator_drag.start, Some(start));

        let held = run_axis_frame(&ctx, rect, &mut viewport, Vec::new());
        assert_eq!(held.cursor, egui::CursorIcon::None, "hidden on every frame of the drag");

        let released = run_axis_frame(&ctx, rect, &mut viewport, vec![press(moved, false)]);
        assert_ne!(released.cursor, egui::CursorIcon::None, "release shows the pointer again");
        assert!(!viewport.navigator_drag.moving);
        assert!(viewport.navigator_drag.at.is_none());
        assert_eq!(cursor_warps(&released), vec![start], "release warps to where the drag began");

        let after = run_axis_frame(&ctx, rect, &mut viewport, Vec::new());
        assert_ne!(after.cursor, egui::CursorIcon::None);
        assert!(cursor_warps(&after).is_empty());
    }

    #[test]
    fn a_lost_or_unfocused_drag_shows_the_pointer_again() {
        let ctx = egui::Context::default();
        egui_extras::install_image_loaders(&ctx);
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let mut viewport = Viewport::default();
        let start = navigator_drag_target(&viewport, rect);
        let moved = start + egui::vec2(45.0, 8.0);

        // The pointer is lost mid-drag.
        run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(start, true)]);
        let dragging =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(moved)]);
        assert_eq!(dragging.cursor, egui::CursorIcon::None);
        let gone = run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerGone]);
        assert_ne!(gone.cursor, egui::CursorIcon::None, "a lost drag shows the pointer");
        assert!(!viewport.navigator_drag.moving);
        assert!(viewport.navigator_drag.at.is_none());
        assert!(viewport.navigator_drag.start.is_none());

        // The window loses focus mid-drag.
        run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(start, true)]);
        let dragging =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(moved)]);
        assert_eq!(dragging.cursor, egui::CursorIcon::None);
        let unfocused = run_axis_frame_focused(&ctx, rect, &mut viewport, Vec::new(), false);
        assert_ne!(unfocused.cursor, egui::CursorIcon::None, "focus loss shows the pointer");
        assert!(!viewport.navigator_drag.moving);
        assert!(viewport.navigator_drag.at.is_none());

        // A drag state reset is the same one-state path: hiding follows it.
        run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(start)]);
        run_axis_frame(&ctx, rect, &mut viewport, vec![navigator_press(start, true)]);
        let dragging =
            run_axis_frame(&ctx, rect, &mut viewport, vec![egui::Event::PointerMoved(moved)]);
        assert_eq!(dragging.cursor, egui::CursorIcon::None);
        viewport.navigator_drag = crate::view_navigator::Drag::default();
        let reset = run_axis_frame(&ctx, rect, &mut viewport, Vec::new());
        assert_ne!(reset.cursor, egui::CursorIcon::None, "a reset drag shows the pointer");
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
    fn grid_defaults_on_and_old_settings_keep_it_on() {
        assert!(ViewSettings::default().grid);
        // Settings saved before the field existed must load with the grid on.
        let legacy: ViewSettings =
            serde_json::from_str(r#"{"edges":false}"#).expect("legacy settings parse");
        assert!(legacy.grid);
        let explicit: ViewSettings =
            serde_json::from_str(r#"{"grid":false}"#).expect("grid setting parses");
        assert!(!explicit.grid);
        assert!(explicit.any_extra(), "grid off is an active display override");
    }

    #[test]
    fn grid_visibility_is_one_shared_decision() {
        let mut view = ViewSettings::default();
        assert!(draws_grid(&view));
        view.grid = false;
        assert!(!draws_grid(&view), "the GPU grid and axis passes must stay off");
    }

    #[test]
    fn grid_off_produces_no_cpu_grid_lines() {
        let display = crate::adapter::DisplayMesh {
            mesh: std::sync::Arc::new(solidcraft_geom::Mesh::default()),
            lods: Vec::new(),
            face_ids: Vec::new(),
            exact_normals: false,
            face_colors: Vec::new(),
            body_ids: Vec::new(),
            edges: Vec::new(),
            comparison: None,
            selection_edges: None,
            radius: 10.0,
            center: Vec3::ZERO,
        };
        let camera = Camera::default();
        let mut view = ViewSettings::default();
        assert!(!scene_for(&display, &camera, 0, &view).grid.is_empty());
        view.grid = false;
        assert!(scene_for(&display, &camera, 0, &view).grid.is_empty());
    }

    #[test]
    fn toggling_grid_only_repaints_the_viewport() {
        let mut viewport = Viewport::default();
        viewport.dirty = false;
        let mut view = viewport.view();
        view.grid = false;
        viewport.set_view(view);
        assert!(viewport.dirty, "the grid toggle must repaint the viewport");
        assert!(!viewport.plan_set_staged, "the grid toggle must not restage the plan set");
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
        let shifted =
            viewport.project_to_viewport(point + shift, rect).expect("shifted point projects");
        let mut preview = [[0.0f32; 4]; 4];
        for (axis, column) in preview.iter_mut().enumerate() {
            column[axis] = 1.0;
        }
        preview[3][0] = shift.x as f32;
        viewport.preview_transform = Some(preview);
        let previewed =
            viewport.project_to_viewport(point, rect).expect("previewed point projects");
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

    #[cfg(not(feature = "gpu-viewport"))]
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

    /// Face-colours used to disable the plan path and fall back to the CPU
    /// object mesh. Plan-only evaluation leaves that mesh with edges but no
    /// triangles, so the fallback renders a wireframe. Face-colours is now a
    /// table fill and must keep the plan renderer enabled.
    #[test]
    fn face_colours_do_not_disable_the_plan_path() {
        let (enabled, colour) = plan_draw_state(false, true, false);
        assert!(enabled, "face-colours mode must keep drawing the plan set");
        assert_eq!(colour, None);
        assert!(!plan_draw_state(true, true, false).0, "no plans, no plan draw");
        assert_eq!(
            plan_draw_state(false, false, true).1,
            Some(DEFAULT_FACE_COLOUR),
            "ignore-look still replaces every face with the override",
        );
    }

    #[test]
    fn clearing_a_document_evicts_plan_state() {
        let mut viewport = Viewport::default();
        viewport.plans = vec![nodecraft_display::PlanInstance {
            target: "old-project".into(),
            plan: std::sync::Arc::new(nodecraft_layers::tess::TessPlan::default()),
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        }];
        viewport.set_feature_ghost_plans(vec![nodecraft_display::PlanInstance {
            target: "old-ghost".into(),
            plan: std::sync::Arc::new(nodecraft_layers::tess::TessPlan::default()),
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        }]);
        viewport.clear_document_display();
        assert!(viewport.plans.is_empty(), "closed project kept its plans");
        assert!(viewport.feature_ghost_plans.is_empty(), "closed project kept ghost plans");
        assert!(viewport.display.is_none(), "closed project kept its display mesh");
    }

    #[cfg(not(feature = "gpu-viewport"))]
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
            exact_normals: false,
            face_colors: Vec::new(),
            body_ids: Vec::new(),
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
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
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
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(vec![Some("edge:0".to_owned())]),
            edge_classes: std::sync::Arc::from(Vec::new()),
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
            exact_normals: false,
            face_colors: Vec::new(),
            body_ids: Vec::new(),
            edges: vec![crate::adapter::DisplayEdge {
                id: "edge:downstream".into(),
                points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X],
                midpoint: [0.5, 0.0, 0.0],
                class: nodecraft_layers::PlanEdgeClass::NORMAL,
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
            points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::Y],
            midpoint: [0.0, 0.5, 0.0],
            class: nodecraft_layers::PlanEdgeClass::NORMAL,
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
            exact_normals: false,
            face_colors: Vec::new(),
            body_ids: Vec::new(),
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

    #[test]
    fn edge_width_defaults_to_the_old_standard_width() {
        assert_eq!(ViewSettings::default().edge_width_px, 1.0);
        assert_eq!(default_edge_width_px(), 1.0);
    }

    #[test]
    fn old_settings_without_edge_width_get_the_default() {
        let view: ViewSettings =
            serde_json::from_value(serde_json::json!({})).expect("old settings");
        assert_eq!(view.edge_width_px, default_edge_width_px());
        assert!(
            !view.any_extra(),
            "the default edge width must not switch on an extra display path",
        );
    }

    #[test]
    fn old_settings_without_show_seam_edges_default_off() {
        let view: ViewSettings =
            serde_json::from_value(serde_json::json!({})).expect("old settings");
        assert!(!view.show_seam_edges);
        assert!(!view.any_extra(), "the default must not look like a custom view");
        let explicit: ViewSettings =
            serde_json::from_str(r#"{"show_seam_edges":true}"#).expect("seam setting parses");
        assert!(explicit.show_seam_edges);
        assert!(explicit.any_extra(), "showing seams is an active display override");
    }
}
