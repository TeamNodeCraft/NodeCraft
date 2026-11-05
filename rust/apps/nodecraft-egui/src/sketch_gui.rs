//! First-pass sketch editor GUI. The sketch plane is world XY at Z = 0; the
//! upstream SolidCraft sketch crate owns points, curves, constraints and solving.
//! This module owns only the viewport interaction and projection.

use std::collections::HashSet;

use egui::{Pos2, Rect, Stroke, Vec2 as EVec2};
use solidcore_record_new::{Vec2, Vec3};
use solidcraft_geom::prelude::*;
use solidcraft_render::Camera;
use nodecraft_engine::sketch::{solve, CurveKind, Sketch};
use nodecraft_dock::{DockItem, DockMetrics, DockStyle, paint_background};
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
        self.dirty = false;
    }

    fn add_three_point_arc(&mut self, start: Vec2, end: Vec2, through: Vec2) {
        let Some(arc) = arc_through_three_points(start, end, through) else { return };
        if self
            .sketch
            .add_arc(arc.center, arc.start, arc.end, [None; 3], None)
            .is_err()
        {
            return;
        }
        let _ = solve(&mut self.sketch);
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
    ) -> Option<serde_json::Value> {
        if interaction.width() < 280.0 || interaction.height() < 220.0 {
            return None;
        }
        let theme = crate::theme::get();
        let mut palette_rects = Vec::new();
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

        self.handle_input(ctx, viewport, interaction, camera, lens_shift, &palette_rects);

        let mut painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("sketch-canvas"),
        ));
        painter.set_clip_rect(interaction);
        let line_colour = theme.color(theme.accent);
        let point_colour = theme.color(theme.text);
        for index in 0..self.sketch.curves.len() {
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
            if point.id == "origin" {
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
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.draft = None;
            self.arc_draft = None;
            return;
        }
        let Some(pointer) = ctx.pointer_latest_pos() else { return };
        let over_palette = palettes.iter().any(|rect| rect.contains(pointer));
        if self.tool == Tool::Arc {
            if let Some(draft) = &mut self.arc_draft {
                if let Some(current) = unproject(viewport, camera, lens_shift, pointer) {
                    draft.current = current;
                }
            }
            if ctx.input(|input| input.pointer.primary_clicked())
                && interaction.contains(pointer)
                && !over_palette
            {
                let Some(point) = unproject(viewport, camera, lens_shift, pointer) else { return };
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
            if self.tool == Tool::Select
                && ctx.input(|input| input.pointer.primary_clicked())
                && interaction.contains(pointer)
                && !over_palette
            {
                let additive = ctx.input(|input| input.modifiers.shift || input.modifiers.command);
                self.select_at(viewport, interaction, camera, lens_shift, pointer, additive);
            }
            return;
        }
        if ctx.input(|input| input.pointer.primary_pressed())
            && interaction.contains(pointer)
            && !over_palette
        {
            if let Some(start) = unproject(viewport, camera, lens_shift, pointer) {
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
        let _ = solve(&mut self.sketch);
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
        let _ = solve(&mut self.sketch);
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
