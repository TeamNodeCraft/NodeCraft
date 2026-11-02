//! CPU viewport built on SolidCraft's camera, scene and rasterizer.
//!
//! The wgpu path from `solidcraft-gpu` will replace the rasterizer once the viewport
//! adapter lands. Keeping the camera/scene/raster on SolidCraft types means the
//! switch is local to this module.

use solidcraft_render::{Camera, Rgb, Scene, SceneLine, SceneMesh, StandardView};
use nodecraft_protocol::Mesh as NodeMesh;

use crate::adapter::{self, DisplayMesh};
use crate::viewer_gizmo::{self, Gizmo, GizmoState};
use crate::viewport_grid::{grid_lines, GridKind};
#[cfg(feature = "gpu-viewport")]
use crate::gpu_viewport::GpuViewport;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ViewSettings {
    /// Draw the shaded solid with alpha blending.
    pub translucent: bool,
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
        self.translucent || !self.edges || self.triangles || self.normals || self.face_colours || self.zebra
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectionMode {
    #[default]
    Object,
    Face,
    Edge,
    Vertex,
}

impl SelectionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Object => "Object",
            Self::Face => "Face",
            Self::Edge => "Edge",
            Self::Vertex => "Vertex",
        }
    }
}

fn raycast_face(
    mesh: &solidcraft_geom::Mesh,
    face_ids: &[String],
    origin: solidcraft_geom::Vec3,
    direction: solidcraft_geom::Vec3,
) -> Option<String> {
    use solidcraft_geom::Vec3;
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

pub struct Viewport {
    pub camera: Camera,
    chrome_visible: bool,
    display: Option<DisplayMesh>,
    view: ViewSettings,
    /// NDC projection offset used to place the pivot at the visible viewport
    /// centre while leaving `camera.target` at the model centre.
    lens_shift: [f32; 2],
    texture: Option<egui::TextureHandle>,
    dirty: bool,
    last_size: [usize; 2],
    /// Right-hand chrome inset used when a floating panel covers the viewport.
    pub chrome_right_inset: f32,
    selection_mode: SelectionMode,
    face_picks: Vec<String>,
    dock_rect: Option<egui::Rect>,
    transform_gizmo: Option<Gizmo>,
    gizmo_state: GizmoState,
    gizmo_changes: Vec<(String, f64)>,
    #[cfg(feature = "gpu-viewport")]
    gpu: Option<GpuViewport>,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            chrome_visible: true,
            display: None,
            view: ViewSettings::default(),
            lens_shift: [0.0, 0.0],
            texture: None,
            dirty: true,
            last_size: [0, 0],
            chrome_right_inset: 0.0,
            selection_mode: SelectionMode::Object,
            face_picks: Vec::new(),
            dock_rect: None,
            transform_gizmo: None,
            gizmo_state: GizmoState::default(),
            gizmo_changes: Vec::new(),
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

    fn update_mesh(&mut self, mesh: Option<&NodeMesh>) {
        self.display = mesh.map(adapter::display_mesh);
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(self.view, self.display.as_ref());
            gpu.set_mesh(self.display.as_ref());
        }
    }

    pub fn set_view(&mut self, view: ViewSettings) {
        if self.view == view {
            return;
        }
        self.view = view;
        #[cfg(feature = "gpu-viewport")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_view(view, self.display.as_ref());
        }
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

    pub fn set_selection_mode(&mut self, mode: SelectionMode) {
        self.selection_mode = mode;
    }

    pub fn fit(&mut self) {
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
        self.dirty = true;
    }

    /// Fit the model into the unobstructed viewport region and center it there.
    /// `full` is the whole 3D viewport; `visible` is the part not covered by
    /// the floating node/tool panels.
    pub fn fit_visible(&mut self, full: egui::Rect, visible: egui::Rect) {
        let Some(display) = &self.display else {
            self.camera = Camera::default();
            self.dirty = true;
            return;
        };
        let visible = visible.intersect(full);
        if visible.width() <= 1.0 || visible.height() <= 1.0 {
            self.fit();
            return;
        }
        let (yaw, pitch) = StandardView::Iso.angles();
        self.camera.yaw = -yaw;
        self.camera.pitch = pitch;
        let aspect = f64::from(visible.width() / visible.height());
        let tan_half = if self.camera.fov > 0.0 {
            (self.camera.fov * 0.5).tan()
        } else {
            0.4
        };
        let vertical = display.radius / (tan_half * 0.9);
        let horizontal = display.radius / (tan_half * aspect * 0.9);
        self.camera.distance = vertical.max(horizontal).max(1.0);
        // Keep the pivot at the model bounds centre. Composition is performed by
        // the projection lens shift, not by moving the orbit pivot.
        self.camera.target = display.center;
        self.lens_shift = [
            2.0 * (visible.center().x - full.center().x) / full.width().max(1.0),
            -2.0 * (visible.center().y - full.center().y) / full.height().max(1.0),
        ];
        self.dirty = true;
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
        }
        self.transform_gizmo = gizmo;
    }

    pub fn take_gizmo_changes(&mut self) -> Vec<(String, f64)> {
        std::mem::take(&mut self.gizmo_changes)
    }

    fn draw_transform_gizmo(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let Some(display) = self.display.as_ref() else { return };
        let Some(gizmo) = self.transform_gizmo.as_mut() else { return };
        let changes = viewer_gizmo::show(
            ui,
            rect,
            &self.camera,
            self.lens_shift,
            display.radius,
            gizmo,
            &mut self.gizmo_state,
        );
        self.gizmo_changes.extend(changes);
    }

    pub fn show(&mut self, ui: &mut egui::Ui, overlay: egui::Rect) {
        self.show_with_backdrop(ui, overlay, overlay);
    }

    /// Draw the viewport with a separate rect driving the frosted-backdrop crop.
    /// `overlay` still controls which viewport areas absorb navigation gestures.
    pub fn show_with_backdrop(
        &mut self,
        ui: &mut egui::Ui,
        overlay: egui::Rect,
        backdrop_rect: egui::Rect,
    ) {
        self.show_with_backdrop_blockers(ui, overlay, &[overlay], backdrop_rect);
    }

    /// Draw the viewport while blocking navigation gestures over every floating
    /// panel, not just the node editor.
    pub fn show_with_backdrop_blockers(
        &mut self,
        ui: &mut egui::Ui,
        overlay: egui::Rect,
        blockers: &[egui::Rect],
        backdrop_rect: egui::Rect,
    ) {
        let available = ui.available_size();
        let width = available.x.max(32.0) as usize;
        let height = available.y.max(32.0) as usize;
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let visible = egui::Rect::from_min_max(
            egui::pos2(overlay.max.x.clamp(rect.min.x, rect.max.x), rect.min.y),
            rect.max,
        );

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
            blockers.iter().any(|blocker| blocker.contains(pointer))
                || overlay.contains(pointer)
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
                if gesture.shift {
                    // Shift + two fingers rotates the model.
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
            self.dirty = true;
        }

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
                self.axis_gizmo(ui, rect, visible);
                self.selection_dock(ui, rect, visible);
                self.pick_face(ui, rect, response.clicked(), pointer_over_overlay);
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
            self.axis_gizmo(ui, rect, visible);
            self.selection_dock(ui, rect, visible);
            self.pick_face(ui, rect, response.clicked(), pointer_over_overlay);
            self.draw_transform_gizmo(ui, rect);
        }
    }

    /// Pick the nearest display-mesh face under the pointer in face mode.
    fn pick_face(&mut self, ui: &mut egui::Ui, viewport: egui::Rect, clicked: bool, blocked: bool) {
        if !clicked || blocked || self.selection_mode != SelectionMode::Face {
            return;
        }
        let Some(pointer) = ui.ctx().pointer_interact_pos() else { return };
        if !viewport.contains(pointer) || self.dock_rect.is_some_and(|dock| dock.contains(pointer)) {
            return;
        }
        let Some(display) = self.display.as_ref() else { return };
        let width = f64::from(viewport.width()).max(1.0);
        let height = f64::from(viewport.height()).max(1.0);
        let x = f64::from(pointer.x - viewport.min.x) - f64::from(self.lens_shift[0]) * width * 0.5;
        let y = f64::from(pointer.y - viewport.min.y) + f64::from(self.lens_shift[1]) * height * 0.5;
        let (origin, direction) = self.camera.ray(x, y, width, height);
        let Some(face) = raycast_face(&display.mesh, display.face_ids.as_slice(), origin, direction) else {
            return;
        };
        self.face_picks.push(face);
        ui.ctx().request_repaint();
    }

    /// Top-of-viewport filter dock for face, edge and vertex selection.
    fn selection_dock(&mut self, ui: &mut egui::Ui, viewport: egui::Rect, visible: egui::Rect) {
        const MARGIN: f32 = 14.0;
        const BUTTON: f32 = 34.0;
        const HEIGHT: f32 = 30.0;
        const GAP: f32 = 4.0;
        let width = BUTTON * 3.0 + GAP * 2.0 + 12.0;
        // Centre between the node editor's right edge and the window's right edge.
        let available_left = visible.min.x.clamp(viewport.min.x, viewport.max.x);
        let available_right = viewport.max.x;
        let dock_center_x = (available_left + available_right) * 0.5;
        let dock = egui::Rect::from_center_size(
            egui::Pos2::new(dock_center_x, viewport.top() + MARGIN + HEIGHT * 0.5),
            egui::Vec2::new(width, HEIGHT),
        );
        // Keep the dock out from under the XYZ compass in narrow viewports.
        let right_limit = viewport.right() - self.chrome_right_inset - 94.0 - MARGIN * 2.0;
        let dock = if dock.right() > right_limit {
            egui::Rect::from_min_size(
                egui::Pos2::new(right_limit - dock.width(), dock.min.y),
                dock.size(),
            )
        } else {
            dock
        };
        let theme = crate::theme::get();
        self.dock_rect = Some(dock);
        let painter = ui.painter().with_clip_rect(viewport);
        painter.rect_filled(dock, 10.0, theme.with_alpha(theme.window, 205));
        painter.rect_stroke(
            dock,
            10.0,
            egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
            egui::StrokeKind::Inside,
        );
        let modes = [SelectionMode::Face, SelectionMode::Edge, SelectionMode::Vertex];
        let mut x = dock.min.x + 6.0;
        for mode in modes {
            let button = egui::Rect::from_min_size(
                egui::Pos2::new(x, dock.min.y + (HEIGHT - BUTTON) * 0.5),
                egui::Vec2::splat(BUTTON),
            );
            let response = ui
                .interact(
                    button,
                    egui::Id::new(("viewport-selection-mode", mode.label())),
                    egui::Sense::click(),
                )
                .on_hover_text(format!("{} selection", mode.label()));
            let active = self.selection_mode == mode;
            if active || response.hovered() {
                painter.rect_filled(
                    button,
                    7.0,
                    theme.with_alpha(if active { theme.highlight } else { theme.control }, 210),
                );
            }
            if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            if response.clicked() {
                self.selection_mode = if active { SelectionMode::Object } else { mode };
                ui.ctx().request_repaint();
            }
            let colour = if active {
                theme.color(theme.accent)
            } else {
                theme.color(theme.icon)
            };
            let center = button.center();
            match mode {
                SelectionMode::Face => {
                    let face = egui::Rect::from_center_size(center + egui::vec2(0.0, 1.0), egui::Vec2::splat(13.0));
                    painter.rect_filled(face, 2.0, theme.with_alpha(theme.icon, 60));
                    painter.rect_stroke(face, 2.0, egui::Stroke::new(1.5, colour), egui::StrokeKind::Inside);
                }
                SelectionMode::Edge => {
                    painter.line_segment(
                        [center + egui::vec2(-7.0, 5.0), center + egui::vec2(7.0, -5.0)],
                        egui::Stroke::new(2.0, colour),
                    );
                    painter.circle_filled(center + egui::vec2(-7.0, 5.0), 2.0, colour);
                    painter.circle_filled(center + egui::vec2(7.0, -5.0), 2.0, colour);
                }
                SelectionMode::Vertex => {
                    painter.circle_filled(center, 5.0, colour);
                    painter.circle_stroke(center, 7.0, egui::Stroke::new(1.2, colour));
                }
                SelectionMode::Object => {}
            }
            x += BUTTON + GAP;
        }
    }

    /// SolidCraft-style signed XYZ compass in the viewport's top-right corner.
    /// Clicking a positive tip chooses that standard view; dragging the compass orbits.
    fn axis_gizmo(&mut self, ui: &mut egui::Ui, viewport: egui::Rect, visible: egui::Rect) {
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
            ("x", solidcraft_geom::Vec3::X),
            ("-x", -solidcraft_geom::Vec3::X),
            ("y", solidcraft_geom::Vec3::Y),
            ("-y", -solidcraft_geom::Vec3::Y),
            ("z", solidcraft_geom::Vec3::Z),
            ("-z", -solidcraft_geom::Vec3::Z),
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
            self.fit_visible(viewport, visible);
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
        let svg = std::str::from_utf8(include_bytes!("../../../../vendor/fluent/perspective.svg"))
            .unwrap_or_default()
            .replace("#d8d8dc", &perspective_colour);
        let image = egui::Image::from_bytes("bytes://nodecraft-perspective.svg", svg.into_bytes())
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
    }

    /// Shared display-settings popup, used by the viewport cog and the top bar.
    pub fn show_view_settings_menu(&mut self, _ui: &mut egui::Ui, response: &egui::Response) {
        let mut edited = self.view;
        let mut changed = false;
        egui::Popup::menu(response)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .style(crate::theme::menu_style)
            .show(|ui| {
                ui.set_width(crate::theme::menu_width(
                    ui,
                    &[
                        "Translucent",
                        "Edges",
                        "Triangles",
                        "Face normals",
                        "Face colours",
                        "Zebra analysis",
                    ],
                ));
                ui.label(
                    egui::RichText::new("Display")
                        .size(11.0)
                        .color(crate::theme::get().color(crate::theme::get().muted)),
                );
                ui.add_space(2.0);
                changed |= crate::theme::menu_toggle(ui, &mut edited.translucent, "Translucent");
                changed |= crate::theme::menu_toggle(ui, &mut edited.edges, "Edges");
                changed |= crate::theme::menu_toggle(ui, &mut edited.triangles, "Triangles");
                changed |= crate::theme::menu_toggle(ui, &mut edited.normals, "Face normals");
                changed |= crate::theme::menu_toggle(ui, &mut edited.face_colours, "Face colours");
                changed |= crate::theme::menu_toggle(ui, &mut edited.zebra, "Zebra analysis");
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("Material")
                        .size(11.0)
                        .color(crate::theme::get().color(crate::theme::get().muted)),
                );
                ui.add_space(2.0);
                ui.spacing_mut().slider_width = 150.0;
                changed |= ui
                    .add(
                        egui::Slider::new(&mut edited.roughness, 0.045..=1.0)
                            .text("Roughness")
                            .max_decimals(2),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut edited.metallic, 0.0..=1.0)
                            .text("Metallic")
                            .max_decimals(2),
                    )
                    .changed();
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

        let scene = scene_for(display, &self.camera);
        let canvas = solidcraft_render::render(&scene, &self.camera, width, height);
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
    center: solidcraft_geom::Vec3,
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
                solidcraft_geom::Vec3::new(
                    center.x + line.start[0] as f64,
                    center.y + line.start[1] as f64,
                    center.z + line.start[2] as f64,
                ),
                solidcraft_geom::Vec3::new(
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

fn scene_for(display: &DisplayMesh, camera: &Camera) -> Scene {
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
        solidcraft_geom::Vec3::ZERO,
        camera.fov > 0.0,
        ortho_fade,
    );
    scene.meshes.push(SceneMesh::solid(display.mesh.clone(), Rgb(151, 162, 176)));
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
    fn projection_toggle_preserves_target_rotation_and_half_height() {
        let mut viewport = Viewport::default();
        viewport.camera.target = solidcraft_geom::Vec3::new(12.0, -4.0, 7.0);
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
    use super::*;

    #[test]
    fn zebra_is_an_active_display_override() {
        let mut view = ViewSettings::default();
        assert!(!view.any_extra());
        view.zebra = true;
        assert!(view.any_extra());
    }

    #[test]
    fn face_pick_returns_the_solidcore_face_group() {
        let mesh = solidcraft_geom::Mesh {
            positions: vec![
                solidcraft_geom::Vec3::new(-1.0, -1.0, 0.0),
                solidcraft_geom::Vec3::new(1.0, -1.0, 0.0),
                solidcraft_geom::Vec3::new(1.0, 1.0, 0.0),
            ],
            normals: vec![solidcraft_geom::Vec3::Z; 3],
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
            solidcraft_geom::Vec3::new(0.0, 0.0, 5.0),
            solidcraft_geom::Vec3::new(0.0, 0.0, -1.0),
        );
        assert_eq!(picked.as_deref(), Some("body:0:face:9"));
    }

    #[test]
    fn mesh_update_can_preserve_the_camera() {
        let mut viewport = Viewport::default();
        viewport.camera.target = solidcraft_geom::Vec3::new(12.0, -4.0, 3.0);
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
}
