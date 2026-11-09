//! Shared macOS-style controls.

use egui::{Response, Sense, Ui, Vec2};

/// A macOS-style switch: a pill track with a sliding knob.
pub fn switch(ui: &mut Ui, value: &mut bool) -> Response {
    let theme = crate::theme::get();
    let desired = Vec2::new(34.0, 18.0);
    let (rect, mut response) = ui.allocate_exact_size(desired, Sense::click());
    response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
    if response.clicked() {
        *value = !*value;
        response.mark_changed();
    }
    let t = if *value { 1.0 } else { 0.0 };
    let track = theme.color(theme.control).lerp_to_gamma(theme.color(theme.accent), t);
    let knob = theme.color(theme.text).lerp_to_gamma(theme.color(theme.control), t * 0.25);
    ui.painter().rect_filled(rect, rect.height() * 0.5, track);
    let radius = rect.height() * 0.5 - 2.0;
    let center = egui::Pos2::new(
        rect.left() + radius + 2.0 + (rect.width() - 2.0 * radius - 4.0) * t,
        rect.center().y,
    );
    ui.painter().circle_filled(center, radius, knob);
    response
}

/// A settings row with a label on the left and a switch on the right.
pub fn switch_row(ui: &mut Ui, label: impl Into<egui::WidgetText>, value: &mut bool) -> Response {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| switch(ui, value),
        )
        .inner
    })
    .inner
}
