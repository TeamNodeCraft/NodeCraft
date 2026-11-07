//! Small cross-platform window-chrome primitives.
//!
//! The main and settings windows have deliberately different title-bar layouts.
//! These helpers only cover the shared mechanics: a clickable window button,
//! title text, and starting a native window drag.

use super::*;

pub(super) const CLOSE_LIGHT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 95, 86);
pub(super) const MINIMIZE_LIGHT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 189, 46);
pub(super) const ZOOM_LIGHT_COLOR: egui::Color32 = egui::Color32::from_rgb(39, 201, 63);

/// A circular window control that sends `command` when clicked.
///
/// It is intentionally platform-neutral: Windows/Linux use the same small
/// control while macOS uses it as a traffic light.
pub(super) fn window_button(
    ui: &mut egui::Ui,
    center: egui::Pos2,
    color: egui::Color32,
    id: egui::Id,
    command: egui::ViewportCommand,
) -> bool {
    let hit = egui::Rect::from_center_size(center, egui::Vec2::splat(20.0));
    let response = ui.interact(hit, id, egui::Sense::click());
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
        ui.ctx().send_viewport_cmd(command);
        true
    } else {
        false
    }
}

/// Draw the title-bar close light.
pub(super) fn close_light(ui: &mut egui::Ui, center: egui::Pos2, id: egui::Id) -> bool {
    window_button(ui, center, CLOSE_LIGHT_COLOR, id, egui::ViewportCommand::Close)
}

/// A title-bar surface that can start a native window drag from empty space.
pub(super) fn drag_response(
    ui: &mut egui::Ui,
    rect: egui::Rect,
    id: egui::Id,
) -> egui::Response {
    ui.interact(rect, id, egui::Sense::drag())
}

pub(super) fn start_drag(ui: &egui::Ui) {
    ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
}

/// Draw a title-bar label.
pub(super) fn title(
    ui: &egui::Ui,
    position: egui::Pos2,
    text: &str,
    color: egui::Color32,
) {
    ui.painter().text(
        position,
        egui::Align2::LEFT_CENTER,
        text,
        egui::FontId::proportional(13.0),
        color,
    );
}

pub(super) fn resize_direction(ctx: &egui::Context) -> Option<egui::viewport::ResizeDirection> {
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

pub(super) fn resize_cursor(direction: egui::viewport::ResizeDirection) -> egui::CursorIcon {
    use egui::viewport::ResizeDirection;
    match direction {
        ResizeDirection::North | ResizeDirection::South => egui::CursorIcon::ResizeVertical,
        ResizeDirection::East | ResizeDirection::West => egui::CursorIcon::ResizeHorizontal,
        ResizeDirection::NorthEast | ResizeDirection::SouthWest => egui::CursorIcon::ResizeNeSw,
        ResizeDirection::NorthWest | ResizeDirection::SouthEast => egui::CursorIcon::ResizeNwSe,
    }
}

