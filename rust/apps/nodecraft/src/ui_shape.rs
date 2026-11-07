//! Shared app-chrome squircle geometry.
//!
//! The node editor owns its own squircle/theme settings. This module is only for
//! the surrounding application UI so all rounded rectangles use one curvature.

use egui::{Color32, Response, Sense, Stroke, Ui, Vec2};
use nodecraft_node_canvas::squircle;

pub const RADIUS: f32 = 12.0;

pub fn shape(rect: egui::Rect, fill: Color32, stroke: Stroke) -> egui::Shape {
    squircle::shape(rect, RADIUS, fill, stroke)
}

pub fn fill(painter: &egui::Painter, rect: egui::Rect, fill: Color32) {
    painter.add(shape(rect, fill, Stroke::NONE));
}

pub fn outline(painter: &egui::Painter, rect: egui::Rect, stroke: Stroke) {
    painter.add(shape(rect, Color32::TRANSPARENT, stroke));
}

pub fn image(painter: &egui::Painter, texture: egui::TextureId, rect: egui::Rect, uv: egui::Rect) {
    let points = squircle::points(rect, RADIUS);
    if points.len() < 3 {
        return;
    }
    let mut mesh = egui::Mesh::with_texture(texture);
    let centre = rect.center();
    let uv_at = |point: egui::Pos2| {
        egui::pos2(
            uv.min.x + (point.x - rect.min.x) / rect.width().max(1.0) * uv.width(),
            uv.min.y + (point.y - rect.min.y) / rect.height().max(1.0) * uv.height(),
        )
    };
    mesh.vertices.push(egui::epaint::Vertex {
        pos: centre,
        uv: uv_at(centre),
        color: Color32::WHITE,
    });
    for point in &points {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: *point,
            uv: uv_at(*point),
            color: Color32::WHITE,
        });
    }
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        mesh.add_triangle(0, index as u32 + 1, next as u32 + 1);
    }
    painter.add(egui::Shape::mesh(mesh));
}

pub fn button(
    ui: &mut Ui,
    label: &str,
    width: f32,
    height: f32,
    fill: Color32,
    stroke: Stroke,
    text_color: Color32,
) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let fill = if response.hovered() {
        fill.gamma_multiply(1.08)
    } else {
        fill
    };
    ui.painter().add(shape(rect, fill, stroke));
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(13.0),
        text_color,
    );
    response
}

pub fn callout(ui: &mut Ui, text: &str) {
    let theme = crate::theme::get();
    let text_color = theme.color(theme.muted);
    let width = ui.available_width();
    let font = egui::FontId::proportional(12.0);
    let galley = ui
        .painter()
        .layout(text.to_owned(), font, text_color, (width - 20.0).max(40.0));
    let height = galley.size().y + 20.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    ui.painter().add(shape(
        rect,
        theme.with_alpha(theme.control, 150),
        Stroke::new(1.0, theme.with_alpha(theme.border, 160)),
    ));
    ui.painter().galley(rect.min + Vec2::new(10.0, 10.0), galley, text_color);
}
