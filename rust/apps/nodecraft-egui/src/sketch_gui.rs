//! First-pass sketch editor GUI. The sketch plane is world XY at Z = 0; the
//! upstream SolidCraft sketch crate owns points, curves, constraints and solving.
//! This module owns only the viewport interaction and projection.

use std::collections::HashSet;

use egui::{Pos2, Rect, Stroke, Vec2 as EVec2};
use solidcore_record_new::{Vec2, Vec3};
use solidcraft_geom::prelude::*;
use solidcraft_render::Camera;
use nodecraft_engine::sketch::{solve, ConstraintKind, CurveKind, Sketch, SolveReport};
use nodecraft_dock::{DockItem, DockMetrics, DockStyle, paint_background};
use crate::hover::HoverMap;
use crate::ui_shape;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    Line,
    Rectangle,
    Circle,
    Arc,
    Polyline,
    Ellipse,
    Polygon,
    Fillet,
    Trim,
}

impl Tool {
    const TOOLS: [Self; 10] = [
        Self::Select,
        Self::Line,
        Self::Rectangle,
        Self::Circle,
        Self::Arc,
        Self::Polyline,
        Self::Ellipse,
        Self::Polygon,
        Self::Fillet,
        Self::Trim,
    ];

    fn icon(self) -> (&'static str, &'static [u8]) {
        match self {
            Self::Select => ("select", include_bytes!("../../../../vendor/iconify-design/sketch/select.svg")),
            Self::Line => ("line", include_bytes!("../../../../vendor/iconify-design/sketch/line.svg")),
            Self::Rectangle => ("rectangle", include_bytes!("../../../../vendor/iconify-design/sketch/rectangle.svg")),
            Self::Circle => ("circle", include_bytes!("../../../../vendor/iconify-design/sketch/circle.svg")),
            Self::Arc => ("arc", include_bytes!("../../../../vendor/iconify-design/sketch/arc.svg")),
            Self::Polyline => ("polyline", include_bytes!("../../../../vendor/iconify-design/sketch/polyline.svg")),
            Self::Ellipse => ("ellipse", include_bytes!("../../../../vendor/iconify-design/sketch/ellipse.svg")),
            Self::Polygon => ("polygon", include_bytes!("../../../../vendor/iconify-design/sketch/polygon.svg")),
            Self::Fillet => ("fillet", include_bytes!("../../../../vendor/iconify-design/sketch/fillet.svg")),
            Self::Trim => ("trim", include_bytes!("../../../../vendor/iconify-design/sketch/trim.svg")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Select => "Select",
            Self::Line => "Line",
            Self::Rectangle => "Rectangle",
            Self::Circle => "Circle",
            Self::Arc => "3-pt Arc",
            Self::Polyline => "Polyline",
            Self::Ellipse => "Ellipse",
            Self::Polygon => "Polygon",
            Self::Fillet => "Fillet",
            Self::Trim => "Trim",
        }
    }

    fn draws(self) -> bool {
        matches!(self, Self::Line | Self::Rectangle | Self::Circle | Self::Arc)
    }
}

/// Additional gap above the horizontal constraints dock.
///
/// The 3D interaction rect is already inset from the workspace by the panel
/// margin, so this stays zero unless the viewport inset policy changes. This is
/// the single place to adjust that gap.
const CONSTRAINT_DOCK_TOP_GAP: f32 = 0.0;

struct SketchDockLayout {
    tools: Rect,
    constraints: Rect,
}

fn sketch_dock_layout(
    interaction: Rect,
    tool_height: f32,
    constraint_rows: usize,
) -> SketchDockLayout {
    let metrics = DockMetrics::DEFAULT;
    let edge = metrics.edge_offset;
    let tool_top = (interaction.center().y - tool_height * 0.5)
        .clamp(
            interaction.top() + edge,
            (interaction.bottom() - tool_height - edge).max(interaction.top() + edge),
        );
    let tools = Rect::from_min_size(
        Pos2::new(interaction.left(), tool_top),
        EVec2::new(metrics.width, tool_height),
    );

    let constraint_metrics = metrics.with_orientation(nodecraft_dock::DockOrientation::Horizontal);
    let constraint_size = constraint_metrics.dock_size(constraint_rows);
    let constraint_x = (interaction.center().x - constraint_size.x * 0.5).clamp(
        interaction.left(),
        (interaction.right() - constraint_size.x).max(interaction.left()),
    );
    let constraints = Rect::from_min_size(
        Pos2::new(constraint_x, interaction.top() + CONSTRAINT_DOCK_TOP_GAP),
        constraint_size,
    );
    SketchDockLayout { tools, constraints }
}

struct Constraint {
    label: &'static str,
    icon_name: &'static str,
    icon: &'static [u8],
}

const CONSTRAINTS: [Constraint; 12] = [
    Constraint { label: "Horizontal", icon_name: "horizontal", icon: include_bytes!("../../../../vendor/iconify-design/constraints/horizontal.svg") },
    Constraint { label: "Vertical", icon_name: "vertical", icon: include_bytes!("../../../../vendor/iconify-design/constraints/vertical.svg") },
    Constraint { label: "Coincident", icon_name: "coincident", icon: include_bytes!("../../../../vendor/iconify-design/constraints/coincident.svg") },
    Constraint { label: "Equal", icon_name: "equal", icon: include_bytes!("../../../../vendor/iconify-design/constraints/equal.svg") },
    Constraint { label: "Parallel", icon_name: "parallel", icon: include_bytes!("../../../../vendor/iconify-design/constraints/parallel.svg") },
    Constraint { label: "Perpendicular", icon_name: "perpendicular", icon: include_bytes!("../../../../vendor/iconify-design/constraints/perpendicular.svg") },
    Constraint { label: "Tangent", icon_name: "tangent", icon: include_bytes!("../../../../vendor/iconify-design/constraints/tangent.svg") },
    Constraint { label: "Concentric", icon_name: "concentric", icon: include_bytes!("../../../../vendor/iconify-design/constraints/concentric.svg") },
    Constraint { label: "Collinear", icon_name: "collinear", icon: include_bytes!("../../../../vendor/iconify-design/constraints/collinear.svg") },
    Constraint { label: "Length", icon_name: "length", icon: include_bytes!("../../../../vendor/iconify-design/constraints/length.svg") },
    Constraint { label: "Radius", icon_name: "radius", icon: include_bytes!("../../../../vendor/iconify-design/constraints/radius.svg") },
    Constraint { label: "Fix", icon_name: "fix", icon: include_bytes!("../../../../vendor/iconify-design/constraints/fix.svg") },
];

struct Draft {
    tool: Tool,
    start: Vec2,
    current: Vec2,
}


struct SketchDrag {
    start: Vec2,
    points: Vec<(usize, Vec2)>,
}

struct ThreePointArc {
    center: Vec2,
    start: Vec2,
    end: Vec2,
}

impl ThreePointArc {
    fn sample(&self, segments: usize) -> Vec<Vec2> {
        let start_angle = (self.start.y - self.center.y).atan2(self.start.x - self.center.x);
        let end_angle = (self.end.y - self.center.y).atan2(self.end.x - self.center.x);
        let mut sweep = end_angle - start_angle;
        while sweep < 0.0 {
            sweep += std::f64::consts::TAU;
        }
        let radius = self.center.dist(self.start);
        (0..=segments)
            .map(|index| {
                let angle = start_angle + sweep * index as f64 / segments as f64;
                self.center + Vec2::new(radius * angle.cos(), radius * angle.sin())
            })
            .collect()
    }
}

fn arc_through_three_points(start: Vec2, end: Vec2, through: Vec2) -> Option<ThreePointArc> {
    let d = 2.0 * (start.x * (end.y - through.y)
        + end.x * (through.y - start.y)
        + through.x * (start.y - end.y));
    if d.abs() <= 1.0e-9 {
        return None;
    }
    let start_sq = start.x * start.x + start.y * start.y;
    let end_sq = end.x * end.x + end.y * end.y;
    let through_sq = through.x * through.x + through.y * through.y;
    let center = Vec2::new(
        (start_sq * (end.y - through.y) + end_sq * (through.y - start.y) + through_sq * (start.y - end.y)) / d,
        (start_sq * (through.x - end.x) + end_sq * (start.x - through.x) + through_sq * (end.x - start.x)) / d,
    );
    let radius = center.dist(start);
    if !(radius.is_finite() && radius > 1.0e-9) {
        return None;
    }
    let start_angle = (start.y - center.y).atan2(start.x - center.x);
    let end_angle = (end.y - center.y).atan2(end.x - center.x);
    let through_angle = (through.y - center.y).atan2(through.x - center.x);
    let ccw = |from: f64, to: f64| {
        let mut delta = to - from;
        while delta < 0.0 {
            delta += std::f64::consts::TAU;
        }
        delta
    };
    // Sketch arcs are counter-clockwise. Order the endpoints so the third
    // point lies on the CCW sweep from start to end.
    if ccw(start_angle, through_angle) < ccw(start_angle, end_angle) {
        Some(ThreePointArc { center, start, end })
    } else {
        Some(ThreePointArc { center, start: end, end: start })
    }
}

struct ArcDraft {
    start: Vec2,
    end: Option<Vec2>,
    current: Vec2,
}

pub struct SketchGui {
    tool: Tool,
    active_node: Option<String>,
    sketch: Sketch,
    draft: Option<Draft>,
    arc_draft: Option<ArcDraft>,
    selected_point: Option<usize>,
    selected_curves: HashSet<usize>,
    drag: Option<SketchDrag>,
    /// Line type for new curves: 0 normal, 1 construction, 2 centerline, 3 infinite.
    draw_type: u8,
    clear_sketch_armed: bool,
    solve_report: Option<SolveReport>,
    fillet_radius: f64,
    dirty: bool,
    dock_measured_height: f32,
}

impl Default for SketchGui {
    fn default() -> Self {
        Self {
            tool: Tool::Select,
            active_node: None,
            sketch: Sketch::new(),
            draft: None,
            arc_draft: None,
            selected_point: None,
            selected_curves: HashSet::new(),
            drag: None,
            draw_type: 0,
            clear_sketch_armed: false,
            solve_report: None,
            fillet_radius: 5.0,
            dirty: false,
            dock_measured_height: 0.0,
        }
    }
}

impl SketchGui {
    /// Load the sketch from the selected node when the selection changes.
    pub fn sync(&mut self, node_id: &str, values: &serde_json::Value) {
        let loaded = values
            .get("sketch")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        if self.active_node.as_deref() == Some(node_id) && self.sketch == loaded {
            return;
        }
        self.active_node = Some(node_id.to_owned());
        self.sketch = loaded;
        self.draft = None;
        self.arc_draft = None;
        self.selected_point = None;
        self.selected_curves.clear();
        self.drag = None;
        self.clear_sketch_armed = false;
        self.solve_report = None;
        self.dirty = false;
    }

    fn add_three_point_arc(&mut self, start: Vec2, end: Vec2, through: Vec2) {
        let Some(arc) = arc_through_three_points(start, end, through) else { return };
        let first_new = self.sketch.curves.len();
        if self
            .sketch
            .add_arc(arc.center, arc.start, arc.end, [None; 3], None)
            .is_err()
        {
            return;
        }
        self.style_curve(first_new);
        self.resolve();
        self.selected_curves.clear();
        self.dirty = true;
    }

    pub fn take_value(&mut self) -> Option<serde_json::Value> {
        if !self.dirty {
            return None;
        }
        self.dirty = false;
        serde_json::to_value(&self.sketch).ok()
    }

    /// Draw the sketch chrome and handle canvas input.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        viewport: Rect,
        interaction: Rect,
        camera: &Camera,
        lens_shift: [f32; 2],
        blockers: &[Rect],
    ) -> Option<serde_json::Value> {
        if interaction.width() < 280.0 || interaction.height() < 220.0 {
            return None;
        }
        let theme = crate::theme::get();
        let mut palette_rects = blockers.to_vec();
        // The first rect is the whole workspace; interaction is the visible 3D
        // viewport after the node editor's width is subtracted.
        let metrics = DockMetrics::DEFAULT;
        let style = crate::app::tool_dock_style(&theme);
        let estimated_height = Tool::TOOLS.len() as f32 * metrics.row_size()
            + (Tool::TOOLS.len() as f32 - 1.0) * metrics.gap
            + metrics.padding * 2.0;
        let height = if self.dock_measured_height > 0.0 {
            self.dock_measured_height
        } else {
            estimated_height
        };
        let layout = sketch_dock_layout(
            interaction,
            height,
            CONSTRAINTS.len(),
        );
        let mut dock = layout.tools;
        let dock_content = egui::Area::new(egui::Id::new("sketch-tool-dock"))
            .order(egui::Order::Tooltip)
            .fixed_pos(dock.min + metrics.content_offset())
            .show(ctx, |ui| {
                ui.set_width(metrics.width);
                ui.spacing_mut().item_spacing = EVec2::ZERO;
                for (index, tool) in Tool::TOOLS.iter().enumerate() {
                    let active = self.tool == *tool;
                    let (name, source) = tool.icon();
                    let response = nodecraft_dock::icon_button(
                        ui,
                        metrics,
                        style,
                        DockItem::new(tool.label(), name, source).active(active),
                    );
                    if response.clicked() {
                        self.tool = *tool;
                        self.draft = None;
                        self.arc_draft = None;
                    }
                    if index + 1 < Tool::TOOLS.len() {
                        ui.add_space(metrics.gap);
                    }
                }
            });
        dock = nodecraft_dock::background_rect(dock_content.response.rect, metrics);
        let dock_painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("sketch-tool-dock-background"),
        ));
        paint_background(&dock_painter, dock, metrics, style);
        if (self.dock_measured_height - dock.height()).abs() > 0.5 {
            self.dock_measured_height = dock.height();
            ctx.request_repaint();
        }
        palette_rects.push(dock);

        if matches!(self.tool, Tool::Fillet | Tool::Arc) {
            let options_pos = Pos2::new(dock.max.x + 8.0, dock.center().y - 42.0);
            let options_response = egui::Area::new(egui::Id::new("sketch-tool-options"))
                .order(egui::Order::Tooltip)
                .fixed_pos(options_pos)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(theme.color(theme.popup))
                        .stroke(Stroke::new(1.0, theme.color(theme.border_strong)))
                        .corner_radius(egui::CornerRadius::same(12))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.set_width(156.0);
                            if self.tool == Tool::Arc {
                                ui.label(
                                    egui::RichText::new("Click start, end, then a point on the arc")
                                        .size(10.0)
                                        .color(theme.color(theme.muted)),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("FILLET RADIUS")
                                        .size(9.0)
                                        .color(theme.with_alpha(theme.muted, 190)),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut self.fillet_radius)
                                        .speed(0.1)
                                        .range(0.01..=10_000.0),
                                );
                                let ready = self.selected_curves.len() == 2;
                                let response = ui_shape::button(
                                    ui,
                                    if ready { "Apply Fillet" } else { "Select 2 lines" },
                                    140.0,
                                    24.0,
                                    if ready {
                                        theme.color(theme.accent)
                                    } else {
                                        theme.with_alpha(theme.control, 180)
                                    },
                                    Stroke::new(1.0, theme.color(theme.border)),
                                    theme.color(if ready { theme.text } else { theme.muted }),
                                );
                                if ready && response.clicked() {
                                    self.apply_fillet();
                                }
                            }
                        });
                });
            palette_rects.push(options_response.response.rect);
        }


        let constraint_metrics = DockMetrics::DEFAULT
            .with_orientation(nodecraft_dock::DockOrientation::Horizontal);
        let mut constraint_dock = layout.constraints;
        let mut constraint_to_apply = None;
        let constraint_content = egui::Area::new(egui::Id::new("sketch-constraints"))
            .order(egui::Order::Tooltip)
            .fixed_pos(constraint_dock.min + constraint_metrics.content_offset())
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing = EVec2::ZERO;
                ui.horizontal(|ui| {
                    for (index, constraint) in CONSTRAINTS.iter().enumerate() {
                        let response = nodecraft_dock::icon_button(
                            ui,
                            constraint_metrics,
                            style,
                            DockItem::new(
                                constraint.label,
                                constraint.icon_name,
                                constraint.icon,
                            ),
                        );
                        if response.clicked() {
                            self.tool = Tool::Select;
                            self.draft = None;
                            self.arc_draft = None;
                            constraint_to_apply = Some(index);
                        }
                        if index + 1 < CONSTRAINTS.len() {
                            ui.add_space(constraint_metrics.gap);
                        }
                    }
                });
            });
        constraint_dock = nodecraft_dock::background_rect(
            constraint_content.response.rect,
            constraint_metrics,
        );
        let constraint_painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("sketch-constraints-background"),
        ));
        paint_background(&constraint_painter, constraint_dock, constraint_metrics, style);
        palette_rects.push(constraint_dock);
        if let Some(index) = constraint_to_apply {
            self.apply_constraint(index);
        }

        self.handle_input(ctx, viewport, interaction, camera, lens_shift, &palette_rects);

        let mut painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("sketch-canvas"),
        ));
        painter.set_clip_rect(interaction);
        let line_colour = theme.color(theme.accent);
        let point_colour = theme.color(theme.text);
        for index in 0..self.sketch.curves.len() {
            if !self.curve_visible(index) {
                continue;
            }
            let points = self.sketch.polyline(index);
            let selected = self.selected_curves.contains(&index);
            draw_polyline(
                &painter,
                viewport,
                camera,
                lens_shift,
                &points,
                Stroke::new(
                    if selected { 3.2 } else { 2.0 },
                    if selected {
                        theme.color(theme.highlight)
                    } else {
                        line_colour
                    },
                ),
            );
        }
        for (index, point) in self.sketch.points.iter().enumerate() {
            if point.id == "origin" || self.sketch.view.hide_points {
                continue;
            }
            if let Some(screen) = project(viewport, camera, lens_shift, point.pos) {
                let selected = self.selected_point == Some(index);
                painter.circle_filled(
                    screen,
                    if selected { 5.0 } else { 3.0 },
                    if selected {
                        theme.color(theme.highlight)
                    } else {
                        point_colour
                    },
                );
            }
        }
        if let Some(draft) = &self.draft {
            for points in draft_paths(draft) {
                draw_polyline(
                    &painter,
                    viewport,
                    camera,
                    lens_shift,
                    &points,
                    Stroke::new(1.5, theme.with_alpha(theme.accent, 180)),
                );
            }
        }
        if let Some(arc) = &self.arc_draft {
            let points = match arc.end {
                Some(end) => arc_through_three_points(arc.start, end, arc.current)
                    .map(|arc| arc.sample(48))
                    .unwrap_or_else(|| vec![arc.start, arc.current]),
                None => vec![arc.start, arc.current],
            };
            draw_polyline(
                &painter,
                viewport,
                camera,
                lens_shift,
                &points,
                Stroke::new(1.5, theme.with_alpha(theme.accent, 180)),
            );
        }

        self.take_value()
    }

    fn handle_input(
        &mut self,
        ctx: &egui::Context,
        viewport: Rect,
        interaction: Rect,
        camera: &Camera,
        lens_shift: [f32; 2],
        palettes: &[Rect],
    ) {
        let pointer_position = ctx.pointer_latest_pos();
        let hover = HoverMap::new(interaction, palettes, pointer_position);
        if ctx.input(|input| {
            input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace)
        }) {
            self.delete_selection();
        }
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.draft = None;
            self.arc_draft = None;
            if hover.over_viewport() {
                self.tool = Tool::Select;
            }
            return;
        }
        let Some(pointer) = pointer_position else { return };
        if self.tool == Tool::Arc {
            let current = unproject(viewport, camera, lens_shift, pointer).map(|point| self.snap(point));
            if let (Some(draft), Some(current)) = (&mut self.arc_draft, current) {
                draft.current = current;
            }
            if ctx.input(|input| input.pointer.primary_clicked())
                && hover.over_viewport()
            {
                let Some(point) = unproject(viewport, camera, lens_shift, pointer).map(|point| self.snap(point)) else { return };
                match &mut self.arc_draft {
                    None => {
                        self.arc_draft = Some(ArcDraft {
                            start: point,
                            end: None,
                            current: point,
                        });
                    }
                    Some(draft) if draft.end.is_none() => {
                        if draft.start.dist(point) > 1.0e-6 {
                            draft.end = Some(point);
                            draft.current = point;
                        }
                    }
                    Some(draft) => {
                        let (start, end) = (draft.start, draft.end.expect("arc end checked"));
                        self.arc_draft = None;
                        self.add_three_point_arc(start, end, point);
                    }
                }
            }
            return;
        }
        if !self.tool.draws() {
            self.draft = None;
            if self.tool == Tool::Select {
                let pressed = ctx.input(|input| input.pointer.primary_pressed());
                let released = ctx.input(|input| input.pointer.primary_released());
                if pressed && hover.over_viewport() {
                    let additive = ctx.input(|input| input.modifiers.shift || input.modifiers.command);
                    self.select_at(viewport, interaction, camera, lens_shift, pointer, additive);
                    self.begin_drag(viewport, camera, lens_shift, pointer);
                }
                if self.drag.is_some()
                    && let Some(point) = unproject(viewport, camera, lens_shift, pointer)
                {
                    self.drag_to(point);
                }
                if released {
                    if self.drag.take().is_some() {
                        self.resolve();
                        self.dirty = true;
                    }
                }
            }
            return;
        }
        if ctx.input(|input| input.pointer.primary_pressed())
            && hover.over_viewport()
        {
            if let Some(start) = unproject(viewport, camera, lens_shift, pointer).map(|point| self.snap(point)) {
                self.draft = Some(Draft {
                    tool: self.tool,
                    start,
                    current: start,
                });
            }
        }
        if let Some(draft) = &mut self.draft {
            if let Some(current) = unproject(viewport, camera, lens_shift, pointer) {
                draft.current = current;
            }
            if ctx.input(|input| input.pointer.primary_released()) {
                let draft = self.draft.take().expect("draft was just read");
                self.commit(draft);
            }
        }
    }

    fn resolve(&mut self) {
        self.solve_report = Some(solve(&mut self.sketch));
    }

    fn snap(&self, point: Vec2) -> Vec2 {
        if self.sketch.view.no_snap {
            return point;
        }
        const STEP: f64 = 1.0;
        Vec2::new((point.x / STEP).round() * STEP, (point.y / STEP).round() * STEP)
    }

    fn curve_visible(&self, index: usize) -> bool {
        let Some(curve) = self.sketch.curves.get(index) else { return false };
        if curve.construction && self.sketch.view.hide_construction {
            return false;
        }
        if curve.link.is_some() && self.sketch.view.hide_projected {
            return false;
        }
        true
    }

    fn style_curve(&mut self, index: usize) {
        let style = self.draw_type;
        if let Some(curve) = self.sketch.curves.get_mut(index) {
            curve.construction = style >= 1;
            curve.centerline = style == 2;
            curve.infinite = style == 3 && matches!(curve.kind, CurveKind::Line { .. });
        }
    }

    fn set_line_type(&mut self, style: u8) {
        self.draw_type = style;
        if self.selected_curves.is_empty() {
            return;
        }
        let indices: Vec<usize> = self.selected_curves.iter().copied().collect();
        for index in indices {
            self.style_curve(index);
        }
        self.resolve();
        self.dirty = true;
    }

    fn line_type_icon(style: u8, active: bool) -> (egui::Image<'static>, &'static str) {
        let (bytes, name) = match style {
            0 => (include_bytes!("../../../../vendor/iconify-design/sketch/line-normal.svg").as_slice(), "normal"),
            1 => (include_bytes!("../../../../vendor/iconify-design/sketch/line-construction.svg").as_slice(), "construction"),
            2 => (include_bytes!("../../../../vendor/iconify-design/sketch/line-centerline.svg").as_slice(), "centerline"),
            _ => (include_bytes!("../../../../vendor/iconify-design/sketch/line-infinite.svg").as_slice(), "infinite"),
        };
        let theme = crate::theme::get();
        let colour = crate::theme::Theme::hex(if active {
            [255, 255, 255, 255]
        } else {
            theme.icon
        });
        let svg = crate::app::tint_svg(bytes, &colour);
        // Distinguish the tinted variants so egui's image cache cannot keep the
        // inactive colour on a segment that becomes active.
        let uri: &'static str = match (style, active) {
            (0, false) => "bytes://sketch-line-normal.svg",
            (1, false) => "bytes://sketch-line-construction.svg",
            (2, false) => "bytes://sketch-line-centerline.svg",
            (3, false) => "bytes://sketch-line-infinite.svg",
            (0, true) => "bytes://sketch-line-normal-active.svg",
            (1, true) => "bytes://sketch-line-construction-active.svg",
            (2, true) => "bytes://sketch-line-centerline-active.svg",
            _ => "bytes://sketch-line-infinite-active.svg",
        };
        (egui::Image::from_bytes(uri, svg.into_bytes()).fit_to_exact_size(egui::vec2(20.0, 20.0)), name)
    }

    /// One connected squircle segmented control for the four line types.
    fn line_type_selector(&mut self, ui: &mut egui::Ui) -> Option<u8> {
        let theme = crate::theme::get();
        let width = ui.available_width().min(4.0 * 44.0);
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(width, 30.0),
            egui::Sense::click(),
        );
        if !ui.is_rect_visible(rect) {
            return None;
        }
        let painter = ui.painter();
        painter.add(nodecraft_node_canvas::squircle::shape(
            rect,
            7.0,
            theme.color(theme.control),
            egui::Stroke::new(1.0, theme.with_alpha(theme.border, 190)),
        ));
        let segment_w = rect.width() / 4.0;
        let hovered = response
            .hover_pos()
            .filter(|point| rect.contains(*point))
            .map(|point| (((point.x - rect.left()) / segment_w).floor() as usize).min(3));
        let selected = self.draw_type as usize;
        if let Some(index) = hovered.filter(|index| *index != selected) {
            let segment = egui::Rect::from_min_size(
                egui::pos2(rect.left() + segment_w * index as f32, rect.top()),
                egui::vec2(segment_w, rect.height()),
            );
            let mut corners = egui::CornerRadius::ZERO;
            if index == 0 {
                corners.nw = 7;
                corners.sw = 7;
            } else if index == 3 {
                corners.ne = 7;
                corners.se = 7;
            }
            painter.rect_filled(segment, corners, theme.with_alpha(theme.highlight, 24));
        }
        let selected_rect = egui::Rect::from_min_size(
            egui::pos2(rect.left() + segment_w * selected as f32, rect.top()),
            egui::vec2(segment_w, rect.height()),
        );
        let mut selected_corners = egui::CornerRadius::ZERO;
        if selected == 0 {
            selected_corners.nw = 7;
            selected_corners.sw = 7;
        } else if selected == 3 {
            selected_corners.ne = 7;
            selected_corners.se = 7;
        }
        painter.rect_filled(selected_rect, selected_corners, theme.color(theme.accent));
        for index in 1..4 {
            let x = rect.left() + segment_w * index as f32;
            painter.line_segment(
                [egui::pos2(x, rect.top() + 1.0), egui::pos2(x, rect.bottom() - 1.0)],
                egui::Stroke::new(1.0, theme.with_alpha(theme.border, 210)),
            );
        }
        for index in 0..4 {
            let (image, name) = Self::line_type_icon(index as u8, index == selected);
            let icon_rect = egui::Rect::from_center_size(
                egui::pos2(rect.left() + segment_w * (index as f32 + 0.5), rect.center().y),
                egui::vec2(20.0, 20.0),
            );
            image.paint_at(ui, icon_rect);
            let _ = name;
        }
        if response.clicked()
            && let Some(point) = response.interact_pointer_pos()
            && rect.contains(point)
        {
            return Some((((point.x - rect.left()) / segment_w).floor() as usize).min(3) as u8);
        }
        let name = ["Normal", "Construction", "Centerline", "Infinite"][hovered.unwrap_or(selected)];
        response.on_hover_text(name);
        None
    }

    /// Sketch palette controls for the inspector while editing a sketch node.
    pub fn show_palette(&mut self, ui: &mut egui::Ui) {
        let theme = crate::theme::get();
        if self.solve_report.is_none() {
            self.resolve();
        }
        if let Some(report) = &self.solve_report {
            let text = if report.status == nodecraft_engine::sketch::SolveStatus::Failed {
                "Constraints conflict".to_owned()
            } else if self.sketch.curves.is_empty() {
                "No geometry yet".to_owned()
            } else if report.dof == 0 {
                "Fully constrained".to_owned()
            } else if report.dof == 1 {
                "1 degree of freedom".to_owned()
            } else {
                format!("{} degrees of freedom", report.dof)
            };
            ui.label(
                egui::RichText::new(text)
                    .size(11.0)
                    .color(theme.color(theme.muted)),
            );
        }
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Line Type").size(10.0).color(theme.color(theme.muted)));
        if let Some(style) = self.line_type_selector(ui) {
            self.set_line_type(style);
        }
        ui.add_space(6.0);
        let mut view = self.sketch.view.clone();
        let mut changed = false;
        for (label, hidden) in [
            ("Constraints", &mut view.hide_constraints),
            ("Dimensions", &mut view.hide_dimensions),
            ("Points", &mut view.hide_points),
            ("Snap", &mut view.no_snap),
        ] {
            let mut shown = !*hidden;
            if ui.checkbox(&mut shown, label).changed() {
                *hidden = !shown;
                changed = true;
            }
        }
        let mut construction_mode = self.draw_type == 1;
        if ui.checkbox(&mut construction_mode, "Construction mode").changed() {
            self.set_line_type(if construction_mode { 1 } else { 0 });
        }
        egui::CollapsingHeader::new("MORE")
            .default_open(false)
            .show(ui, |ui| {
                for (label, hidden) in [
                    ("Projected Geometries", &mut view.hide_projected),
                    ("Construction Geometries", &mut view.hide_construction),
                ] {
                    let mut shown = !*hidden;
                    if ui.checkbox(&mut shown, label).changed() {
                        *hidden = !shown;
                        changed = true;
                    }
                }
                for (label, on) in [
                    ("Slice", &mut view.slice),
                    ("3D Sketch", &mut view.three_d),
                ] {
                    if ui.checkbox(on, label).changed() {
                        changed = true;
                    }
                }
            });
        if changed {
            self.sketch.view = view;
            self.dirty = true;
        }
        ui.add_space(12.0);
        let danger = egui::Color32::from_rgb(245, 132, 122);
        if self.clear_sketch_armed {
            ui.label(
                egui::RichText::new("Clear all sketch geometry?")
                    .size(11.0)
                    .color(danger),
            );
            ui.horizontal(|ui| {
                if ui_shape::button(
                    ui,
                    "Cancel",
                    (ui.available_width() - 6.0) * 0.5,
                    28.0,
                    theme.with_alpha(theme.control, 220),
                    egui::Stroke::new(1.0, theme.with_alpha(theme.border, 210)),
                    theme.color(theme.text),
                )
                .clicked()
                {
                    self.clear_sketch_armed = false;
                }
                if ui_shape::button(
                    ui,
                    "Clear Sketch",
                    ui.available_width(),
                    28.0,
                    egui::Color32::from_rgba_unmultiplied(80, 28, 26, 140),
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(245, 132, 122, 180)),
                    danger,
                )
                .clicked()
                {
                    self.clear_sketch();
                }
            });
        } else if ui_shape::button(
            ui,
            "Clear Sketch",
            ui.available_width(),
            28.0,
            egui::Color32::from_rgba_unmultiplied(80, 28, 26, 90),
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(245, 132, 122, 120)),
            danger,
        )
        .clicked()
        {
            self.clear_sketch_armed = true;
        }
    }

    /// Non-interactive outline of the sketch while its node is selected but not
    /// being edited. Lifecycle and tool input stay with the node selection.
    pub fn show_selection_preview(
        &self,
        ctx: &egui::Context,
        viewport: Rect,
        interaction: Rect,
        camera: &Camera,
        lens_shift: [f32; 2],
    ) {
        if interaction.width() < 40.0 || interaction.height() < 40.0 {
            return;
        }
        let theme = crate::theme::get();
        let mut painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("sketch-selection-preview"),
        ));
        painter.set_clip_rect(interaction);
        let colour = theme.color(theme.highlight);
        for index in 0..self.sketch.curves.len() {
            if !self.curve_visible(index) {
                continue;
            }
            let points = self.sketch.polyline(index);
            draw_polyline(
                &painter,
                viewport,
                camera,
                lens_shift,
                &points,
                Stroke::new(2.0, colour),
            );
        }
        if !self.sketch.view.hide_points {
            for (index, point) in self.sketch.points.iter().enumerate() {
                if index == 0 {
                    continue;
                }
                if let Some(screen) = project(viewport, camera, lens_shift, point.pos) {
                    painter.circle_filled(screen, 2.5, colour);
                }
            }
        }
    }

    /// Start a point/curve drag from the current selection. A curve moves its
    /// own points; the origin and other fixed points stay put.
    fn begin_drag(
        &mut self,
        viewport: Rect,
        camera: &Camera,
        lens_shift: [f32; 2],
        pointer: Pos2,
    ) {
        let Some(start) = unproject(viewport, camera, lens_shift, pointer) else { return };
        let mut indices = Vec::new();
        if let Some(point) = self.selected_point {
            indices.push(point);
        } else {
            for curve in &self.selected_curves {
                if let Some(curve) = self.sketch.curves.get(*curve) {
                    indices.extend(curve.kind.point_ids());
                }
            }
        }
        indices.sort_unstable();
        indices.dedup();
        let points = indices
            .into_iter()
            .filter_map(|index| {
                let point = self.sketch.points.get(index)?;
                (!point.fixed).then_some((index, point.pos))
            })
            .collect::<Vec<_>>();
        if !points.is_empty() {
            self.drag = Some(SketchDrag { start, points });
        }
    }

    fn drag_to(&mut self, pointer: Vec2) {
        let Some(drag) = &self.drag else { return };
        let delta = pointer - drag.start;
        for (index, origin) in &drag.points {
            if let Some(point) = self.sketch.points.get_mut(*index) {
                point.pos = *origin + delta;
            }
        }
        // Keep constraints authoritative while the pointer moves. Free points
        // are unchanged; a constrained point stays on its locus.
        self.resolve();
    }

    fn clear_sketch(&mut self) {
        let view = self.sketch.view.clone();
        self.sketch = Sketch::default();
        self.sketch.view = view;
        self.draft = None;
        self.arc_draft = None;
        self.selected_point = None;
        self.selected_curves.clear();
        self.drag = None;
        self.draw_type = 0;
        self.clear_sketch_armed = false;
        self.resolve();
        self.dirty = true;
    }

    fn delete_selection(&mut self) {
        let changed = if let Some(point) = self.selected_point {
            let removable = self
                .sketch
                .points
                .get(point)
                .is_some_and(|point| point.id != "origin");
            if removable {
                self.sketch.remove_points(&[point]);
            }
            removable
        } else if !self.selected_curves.is_empty() {
            let curves: Vec<usize> = self.selected_curves.iter().copied().collect();
            self.sketch.remove_curves(&curves);
            true
        } else {
            false
        };
        if changed {
            self.selected_point = None;
            self.selected_curves.clear();
            self.resolve();
            self.dirty = true;
        }
    }

    /// Apply the constraint dock's button at `index` to the current selection.
    /// The dock is presentation; `solidcraft_sketch` owns validation and solving.
    fn apply_constraint(&mut self, index: usize) {
        let kind = match index {
            0 => self
                .selected_line()
                .map(|l| ConstraintKind::Horizontal { l }),
            1 => self
                .selected_line()
                .map(|l| ConstraintKind::Vertical { l }),
            2 => None, // Coincident needs two picked points; selection supports one for now.
            3 => self
                .selected_curve_pair()
                .map(|(a, b)| ConstraintKind::Equal { a, b }),
            4 => self
                .selected_line_pair()
                .map(|(a, b)| ConstraintKind::Parallel { a, b }),
            5 => self
                .selected_line_pair()
                .map(|(a, b)| ConstraintKind::Perpendicular { a, b }),
            6 => self
                .selected_curve_pair()
                .map(|(a, b)| ConstraintKind::Tangent { a, b }),
            7 => self
                .selected_round_pair()
                .map(|(a, b)| ConstraintKind::Concentric { a, b }),
            8 => self
                .selected_line_pair()
                .map(|(a, b)| ConstraintKind::Collinear { a, b }),
            9 => self.selected_line().and_then(|l| {
                let (a, b) = match self.sketch.curves.get(l)?.kind {
                    CurveKind::Line { a, b } => (a, b),
                    _ => return None,
                };
                let (pa, pb) = (self.sketch.point(a)?, self.sketch.point(b)?);
                Some(ConstraintKind::Length { l, value: pa.dist(pb) })
            }),
            10 => self.selected_round().map(|(c, value)| ConstraintKind::Radius { c, value }),
            11 => self.selected_point.map(|p| ConstraintKind::Fix { p }),
            _ => None,
        };
        let Some(kind) = kind else { return };
        if self.sketch.add_constraint(kind, None).is_ok() {
            self.resolve();
            self.dirty = true;
        }
    }

    fn selected_line(&self) -> Option<usize> {
        self.selected_curves.iter().copied().find(|index| {
            matches!(
                self.sketch.curves.get(*index).map(|curve| &curve.kind),
                Some(CurveKind::Line { .. })
            )
        })
    }

    fn selected_curve_pair(&self) -> Option<(usize, usize)> {
        let mut curves = self.selected_curves.iter().copied();
        let a = curves.next()?;
        let b = curves.next()?;
        (a != b).then_some((a, b))
    }

    fn selected_line_pair(&self) -> Option<(usize, usize)> {
        let mut lines = self.selected_curves.iter().copied().filter(|index| {
            matches!(
                self.sketch.curves.get(*index).map(|curve| &curve.kind),
                Some(CurveKind::Line { .. })
            )
        });
        let a = lines.next()?;
        let b = lines.next()?;
        (a != b).then_some((a, b))
    }

    fn selected_round(&self) -> Option<(usize, f64)> {
        self.selected_curves.iter().copied().find_map(|index| {
            match self.sketch.curves.get(index).map(|curve| &curve.kind) {
                Some(CurveKind::Circle { r, .. }) => Some((index, *r)),
                Some(CurveKind::Arc { c, a, .. }) => {
                    let center = self.sketch.point(*c)?;
                    Some((index, center.dist(self.sketch.point(*a)?)))
                }
                _ => None,
            }
        })
    }

    fn selected_round_pair(&self) -> Option<(usize, usize)> {
        let mut rounds = self.selected_curves.iter().copied().filter(|index| {
            matches!(
                self.sketch.curves.get(*index).map(|curve| &curve.kind),
                Some(CurveKind::Circle { .. } | CurveKind::Arc { .. })
            )
        });
        let a = rounds.next()?;
        let b = rounds.next()?;
        (a != b).then_some((a, b))
    }

    fn apply_fillet(&mut self) {
        if self.selected_curves.len() != 2 {
            return;
        }
        let mut indices: Vec<usize> = self.selected_curves.iter().copied().collect();
        indices.sort_unstable();
        let (first, second) = (indices[0], indices[1]);
        let (Some(kind1), Some(kind2)) = (
            self.sketch.curves.get(first).map(|curve| curve.kind.clone()),
            self.sketch.curves.get(second).map(|curve| curve.kind.clone()),
        ) else {
            return;
        };
        let (CurveKind::Line { a: a1, b: b1 }, CurveKind::Line { a: a2, b: b2 }) = (kind1, kind2) else {
            return;
        };
        let (Some(p1), Some(p2)) = (self.sketch.point(a1), self.sketch.point(b1)) else { return };
        let (Some(q1), Some(q2)) = (self.sketch.point(a2), self.sketch.point(b2)) else { return };
        let d1 = p2 - p1;
        let d2 = q2 - q1;
        let denominator = d1.x * d2.y - d1.y * d2.x;
        if denominator.abs() <= 1.0e-9 {
            return;
        }
        let offset = q1 - p1;
        let t = (offset.x * d2.y - offset.y * d2.x) / denominator;
        let s = (offset.x * d1.y - offset.y * d1.x) / denominator;
        let epsilon = 1.0e-4;
        if !(-epsilon..=1.0 + epsilon).contains(&t) || !(-epsilon..=1.0 + epsilon).contains(&s) {
            return;
        }
        let corner = p1 + d1 * t;
        let (shared1, other1, end1) = if p1.dist(corner) <= p2.dist(corner) {
            (a1, b1, p2)
        } else {
            (b1, a1, p1)
        };
        let (shared2, other2, end2) = if q1.dist(corner) <= q2.dist(corner) {
            (a2, b2, q2)
        } else {
            (b2, a2, q1)
        };
        let Some(u) = (end1 - corner).normalized() else { return };
        let Some(v) = (end2 - corner).normalized() else { return };
        let dot = u.dot(v).clamp(-1.0, 1.0);
        let angle = dot.acos();
        if angle <= 1.0e-3 || angle >= std::f64::consts::PI - 1.0e-3 {
            return;
        }
        let half = angle * 0.5;
        let length1 = corner.dist(end1);
        let length2 = corner.dist(end2);
        let max_radius = (length1.min(length2) * 0.49 * half.tan()).max(1.0e-6);
        let radius = self.fillet_radius.clamp(1.0e-6, max_radius);
        let tangent = radius / half.tan();
        if tangent <= 0.0 || tangent >= length1 || tangent >= length2 {
            return;
        }
        let Some(middle) = (u + v).normalized() else { return };
        let center = corner + middle * (radius / half.sin());
        let tangent1 = corner + u * tangent;
        let tangent2 = corner + v * tangent;
        let (Ok(point1), Ok(point2), Ok(center_point)) = (
            self.sketch.add_point(tangent1, None),
            self.sketch.add_point(tangent2, None),
            self.sketch.add_point(center, None),
        ) else {
            return;
        };
        if let Some(curve) = self.sketch.curves.get_mut(first) {
            curve.kind = if shared1 == a1 {
                CurveKind::Line { a: point1, b: other1 }
            } else {
                CurveKind::Line { a: other1, b: point1 }
            };
        }
        if let Some(curve) = self.sketch.curves.get_mut(second) {
            curve.kind = if shared2 == a2 {
                CurveKind::Line { a: point2, b: other2 }
            } else {
                CurveKind::Line { a: other2, b: point2 }
            };
        }
        let cross = u.x * v.y - u.y * v.x;
        let (start, end, start_point, end_point) = if cross >= 0.0 {
            (tangent2, tangent1, point2, point1)
        } else {
            (tangent1, tangent2, point1, point2)
        };
        let _ = self.sketch.add_arc(
            center,
            start,
            end,
            [Some(center_point), Some(start_point), Some(end_point)],
            None,
        );
        self.resolve();
        self.selected_curves.clear();
        self.dirty = true;
    }

    fn select_at(
        &mut self,
        viewport: Rect,
        interaction: Rect,
        camera: &Camera,
        lens_shift: [f32; 2],
        pointer: Pos2,
        additive: bool,
    ) {
        let point_radius = 9.0_f32;
        let mut best_point: Option<(f32, usize)> = None;
        for (index, point) in self.sketch.points.iter().enumerate() {
            if self.sketch.view.hide_points {
                continue;
            }
            let Some(screen) = project(viewport, camera, lens_shift, point.pos) else { continue };
            if !interaction.contains(screen) {
                continue;
            }
            let distance = screen.distance(pointer);
            if distance <= point_radius
                && best_point.is_none_or(|(best, _)| distance < best)
            {
                best_point = Some((distance, index));
            }
        }
        if let Some((_, index)) = best_point {
            self.selected_point = Some(index);
            self.selected_curves.clear();
            return;
        }
        let mut best_curve: Option<(f32, usize)> = None;
        for index in 0..self.sketch.curves.len() {
            if !self.curve_visible(index) {
                continue;
            }
            let projected: Vec<Pos2> = self
                .sketch
                .polyline(index)
                .into_iter()
                .filter_map(|point| project(viewport, camera, lens_shift, point))
                .collect();
            for pair in projected.windows(2) {
                let distance = point_segment_distance(pointer, pair[0], pair[1]);
                if !interaction.contains(pointer) {
                    continue;
                }
                if distance <= 7.0
                    && best_curve.is_none_or(|(best, _)| distance < best)
                {
                    best_curve = Some((distance, index));
                }
            }
        }
        if let Some((_, index)) = best_curve {
            self.selected_point = None;
            if additive {
                if !self.selected_curves.insert(index) {
                    self.selected_curves.remove(&index);
                }
            } else {
                self.selected_curves.clear();
                self.selected_curves.insert(index);
            }
        } else if !additive {
            self.selected_curves.clear();
            self.selected_point = None;
        }
    }

    fn commit(&mut self, draft: Draft) {
        if draft.start.dist(draft.current) <= 1.0e-6 {
            return;
        }
        let first_new = self.sketch.curves.len();
        match draft.tool {
            Tool::Line => add_line(&mut self.sketch, draft.start, draft.current),
            Tool::Rectangle => {
                let corners = [
                    draft.start,
                    Vec2::new(draft.current.x, draft.start.y),
                    draft.current,
                    Vec2::new(draft.start.x, draft.current.y),
                ];
                for index in 0..4 {
                    add_line(&mut self.sketch, corners[index], corners[(index + 1) % 4]);
                }
            }
            Tool::Circle => {
                let radius = draft.start.dist(draft.current);
                let _ = self.sketch.add_circle(draft.start, radius, None, None);
            }
            _ => {}
        }
        for index in first_new..self.sketch.curves.len() {
            self.style_curve(index);
        }
        self.resolve();
        // Newly drawn geometry must not inherit a stale curve selection; an old
        // index can otherwise land on one of the new edges and draw it thick.
        self.selected_curves.clear();
        self.selected_point = None;
        self.dirty = true;
    }
}

fn add_line(sketch: &mut Sketch, start: Vec2, end: Vec2) {
    let _ = sketch.add_line(start, end, None, None, None);
}

fn draft_paths(draft: &Draft) -> Vec<Vec<Vec2>> {
    match draft.tool {
        Tool::Line => vec![vec![draft.start, draft.current]],
        Tool::Rectangle => {
            let corners = [
                draft.start,
                Vec2::new(draft.current.x, draft.start.y),
                draft.current,
                Vec2::new(draft.start.x, draft.current.y),
            ];
            vec![corners.to_vec(), vec![corners[3], corners[0]]]
        }
        Tool::Circle => {
            let radius = draft.start.dist(draft.current);
            if radius <= 1.0e-6 {
                return Vec::new();
            }
            let points = (0..=64)
                .map(|step| {
                    let angle = std::f64::consts::TAU * step as f64 / 64.0;
                    draft.start + Vec2::new(radius * angle.cos(), radius * angle.sin())
                })
                .collect();
            vec![points]
        }
        _ => Vec::new(),
    }
}

fn project(viewport: Rect, camera: &Camera, lens_shift: [f32; 2], point: Vec2) -> Option<Pos2> {
    let (x, y, _) = camera.to_screen(
        Vec3::new(point.x, point.y, 0.0),
        f64::from(viewport.width()),
        f64::from(viewport.height()),
        camera.distance.max(1.0),
    )?;
    Some(
        viewport.min
            + EVec2::new(
                x as f32 + lens_shift[0] * viewport.width() * 0.5,
                y as f32 - lens_shift[1] * viewport.height() * 0.5,
            ),
    )
}

fn unproject(
    viewport: Rect,
    camera: &Camera,
    lens_shift: [f32; 2],
    pointer: Pos2,
) -> Option<Vec2> {
    let x = f64::from(
        pointer.x - viewport.min.x - lens_shift[0] * viewport.width() * 0.5,
    );
    let y = f64::from(
        pointer.y - viewport.min.y + lens_shift[1] * viewport.height() * 0.5,
    );
    let (origin, direction) = camera.ray(
        x,
        y,
        f64::from(viewport.width()),
        f64::from(viewport.height()),
    );
    if direction.z.abs() <= 1.0e-9 {
        return None;
    }
    let t = -origin.z / direction.z;
    if t <= 0.0 {
        return None;
    }
    let hit = origin + direction * t;
    Some(Vec2::new(hit.x, hit.y))
}

fn draw_polyline(
    painter: &egui::Painter,
    viewport: Rect,
    camera: &Camera,
    lens_shift: [f32; 2],
    points: &[Vec2],
    stroke: Stroke,
) {
    let projected: Vec<Option<Pos2>> = points
        .iter()
        .map(|point| project(viewport, camera, lens_shift, *point))
        .collect();
    for pair in projected.windows(2) {
        if let (Some(a), Some(b)) = (pair[0], pair[1]) {
            painter.line_segment([a, b], stroke);
        }
    }
}


fn point_segment_distance(point: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let length_sq = ab.length_sq();
    if length_sq <= f32::EPSILON {
        return point.distance(a);
    }
    let t = ((point - a).dot(ab) / length_sq).clamp(0.0, 1.0);
    point.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_tool_adds_a_solved_sketch_curve() {
        let mut gui = SketchGui {
            tool: Tool::Line,
            ..SketchGui::default()
        };
        gui.commit(Draft {
            tool: Tool::Line,
            start: Vec2::new(0.0, 0.0),
            current: Vec2::new(20.0, 0.0),
        });
        assert_eq!(gui.sketch.curves.len(), 1);
        let value = gui.take_value().expect("sketch change");
        assert!(value.get("curves").is_some());
    }

    #[test]
    fn clear_sketch_keeps_view_options_and_removes_geometry() {
        let mut gui = SketchGui::default();
        let _ = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), None, None, None);
        gui.sketch.view.hide_points = true;
        gui.clear_sketch_armed = true;
        gui.clear_sketch();
        assert!(gui.sketch.curves.is_empty());
        assert_eq!(gui.sketch.points.len(), 1);
        assert!(gui.sketch.view.hide_points);
        assert!(!gui.clear_sketch_armed);
        assert!(gui.dirty);
    }

    #[test]
    fn delete_removes_the_selected_sketch_curve() {
        let mut gui = SketchGui::default();
        let line = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), None, None, None).unwrap();
        gui.selected_curves.insert(line);
        gui.delete_selection();
        assert!(gui.sketch.curves.is_empty());
        assert!(gui.selected_curves.is_empty());
        assert!(gui.dirty);
    }

    #[test]
    fn drawn_curves_take_the_selected_line_type() {
        let mut gui = SketchGui {
            tool: Tool::Line,
            draw_type: 1,
            ..SketchGui::default()
        };
        gui.commit(Draft {
            tool: Tool::Line,
            start: Vec2::new(0.0, 0.0),
            current: Vec2::new(20.0, 0.0),
        });
        assert!(gui.sketch.curves[0].construction);
        assert!(!gui.sketch.curves[0].centerline);
    }

    #[test]
    fn select_tool_drag_moves_a_point() {
        let mut gui = SketchGui::default();
        let line = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), None, None, None).unwrap();
        let point = match gui.sketch.curves[line].kind {
            CurveKind::Line { b, .. } => b,
            _ => unreachable!(),
        };
        gui.selected_point = Some(point);
        gui.drag = Some(SketchDrag {
            start: Vec2::ZERO,
            points: vec![(point, gui.sketch.point(point).unwrap())],
        });
        gui.drag_to(Vec2::new(3.0, 4.0));
        assert!(gui.sketch.point(point).unwrap().dist(Vec2::new(13.0, 4.0)) < 1.0e-9);
    }

    #[test]
    fn constraint_dock_applies_line_and_dimension_constraints() {
        let mut gui = SketchGui::default();
        let line = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(20.0, 3.0), None, None, None).unwrap();
        gui.selected_curves.insert(line);
        gui.apply_constraint(0); // Horizontal
        assert!(matches!(
            gui.sketch.constraints.last().map(|constraint| &constraint.kind),
            Some(ConstraintKind::Horizontal { l }) if *l == line
        ));
        gui.apply_constraint(9); // Length
        assert!(matches!(
            gui.sketch.constraints.last().map(|constraint| &constraint.kind),
            Some(ConstraintKind::Length { l, .. }) if *l == line
        ));
        assert!(gui.take_value().is_some());
    }

    #[test]
    fn constraint_dock_uses_two_selected_lines() {
        let mut gui = SketchGui::default();
        let a = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), None, None, None).unwrap();
        let b = gui.sketch.add_line(Vec2::new(0.0, 10.0), Vec2::new(10.0, 10.0), None, None, None).unwrap();
        gui.selected_curves.insert(a);
        gui.selected_curves.insert(b);
        gui.apply_constraint(4); // Parallel
        assert!(matches!(
            gui.sketch.constraints.last().map(|constraint| &constraint.kind),
            Some(ConstraintKind::Parallel { .. })
        ));
    }

    #[test]
    fn fillet_joins_two_lines_with_a_tangent_arc() {
        let mut gui = SketchGui {
            tool: Tool::Fillet,
            ..SketchGui::default()
        };
        let a = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(20.0, 0.0), None, None, None).unwrap();
        let b = gui.sketch.add_line(Vec2::new(20.0, 0.0), Vec2::new(20.0, 20.0), None, None, None).unwrap();
        gui.selected_curves.insert(a);
        gui.selected_curves.insert(b);
        gui.apply_fillet();
        assert_eq!(gui.sketch.curves.len(), 3);
        assert!(matches!(gui.sketch.curves[2].kind, CurveKind::Arc { .. }));
        let arc = gui.sketch.polyline(2);
        let length: f64 = arc.windows(2).map(|pair| pair[0].dist(pair[1])).sum();
        assert!((length - std::f64::consts::FRAC_PI_2 * 5.0).abs() < 0.2, "arc length {length}");
        assert!(gui.take_value().is_some());
    }

    #[test]
    fn fillet_handles_lines_that_cross_at_ninety_degrees() {
        let mut gui = SketchGui {
            tool: Tool::Fillet,
            ..SketchGui::default()
        };
        let a = gui.sketch.add_line(Vec2::new(0.0, 0.0), Vec2::new(20.0, 0.0), None, None, None).unwrap();
        let b = gui.sketch.add_line(Vec2::new(10.0, -10.0), Vec2::new(10.0, 10.0), None, None, None).unwrap();
        gui.selected_curves.insert(a);
        gui.selected_curves.insert(b);
        gui.apply_fillet();
        assert_eq!(gui.sketch.curves.len(), 3);
        assert!(matches!(gui.sketch.curves[2].kind, CurveKind::Arc { .. }));
    }

    #[test]
    fn three_point_arc_passes_through_its_third_point() {
        let mut gui = SketchGui {
            tool: Tool::Arc,
            ..SketchGui::default()
        };
        gui.add_three_point_arc(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(5.0, 5.0),
        );
        assert_eq!(gui.sketch.curves.len(), 1);
        assert!(matches!(gui.sketch.curves[0].kind, CurveKind::Arc { .. }));
        let polyline = gui.sketch.polyline(0);
        let max_y = polyline.iter().map(|point| point.y).fold(f64::NEG_INFINITY, f64::max);
        assert!(max_y > 4.9, "arc did not pass through the third point: max y {max_y}");
        assert!(gui.take_value().is_some());
    }

    #[test]
    fn screen_ray_and_projection_round_trip_on_xy() {
        let camera = Camera::default();
        let viewport = Rect::from_min_size(Pos2::new(100.0, 50.0), EVec2::new(800.0, 600.0));
        let point = Vec2::new(12.0, -8.0);
        let screen = project(viewport, &camera, [0.0; 2], point).expect("project");
        let back = unproject(viewport, &camera, [0.0; 2], screen).expect("unproject");
        assert!((back.x - point.x).abs() < 1.0e-6);
        assert!((back.y - point.y).abs() < 1.0e-6);
    }

    #[test]
    fn drawing_a_rectangle_clears_stale_curve_selection() {
        let mut gui = SketchGui {
            tool: Tool::Rectangle,
            ..SketchGui::default()
        };
        gui.selected_curves.insert(0);
        gui.commit(Draft {
            tool: Tool::Rectangle,
            start: Vec2::new(0.0, 0.0),
            current: Vec2::new(20.0, 10.0),
        });
        assert_eq!(gui.sketch.curves.len(), 4);
        assert!(gui.selected_curves.is_empty());
        assert!(gui.selected_point.is_none());
    }

    #[test]
    fn constraint_dock_uses_the_viewport_top_without_double_margin() {
        let viewport = Rect::from_min_max(Pos2::new(100.0, 50.0), Pos2::new(900.0, 700.0));
        let layout = sketch_dock_layout(viewport, 300.0, CONSTRAINTS.len());
        assert_eq!(
            layout.constraints.top(),
            viewport.top() + CONSTRAINT_DOCK_TOP_GAP
        );
    }
}
