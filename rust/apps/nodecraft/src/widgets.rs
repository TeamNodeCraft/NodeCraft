//! Shared macOS-style controls.

use egui::{Response, Sense, Ui, Vec2};

/// The macOS switch's intrinsic size. Inspector rows reserve exactly this much
/// width for it so the label can use the rest of the row.
pub const SWITCH_SIZE: Vec2 = Vec2::new(34.0, 18.0);

/// A macOS-style switch: a pill track with a sliding knob.
///
/// When its `Ui` is disabled (inside `ui.add_enabled_ui(false, ..)`), the switch
/// paints a muted track/knob and ignores clicks. This is the one disabled look
/// every switch row uses.
pub fn switch(ui: &mut Ui, value: &mut bool) -> Response {
    let theme = crate::theme::get();
    let enabled = ui.is_enabled();
    let (rect, mut response) = ui.allocate_exact_size(SWITCH_SIZE, Sense::click());
    if enabled {
        response = response.on_hover_cursor(egui::CursorIcon::PointingHand);
        if response.clicked() {
            *value = !*value;
            response.mark_changed();
        }
    }
    let t = if *value { 1.0 } else { 0.0 };
    let (track, knob) = if enabled {
        (
            theme.color(theme.control).lerp_to_gamma(theme.color(theme.accent), t),
            theme.color(theme.text).lerp_to_gamma(theme.color(theme.control), t * 0.25),
        )
    } else {
        (
            theme.with_alpha(theme.control, 110),
            theme.with_alpha(theme.muted, 150),
        )
    };
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
