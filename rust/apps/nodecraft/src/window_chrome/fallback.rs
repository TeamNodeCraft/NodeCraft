//! Windows and Linux keep their native title bar and buttons.

use egui::ViewportBuilder;

pub(crate) fn configure_main_viewport(viewport: ViewportBuilder) -> ViewportBuilder {
    viewport
}

pub(crate) fn main_bar_controls_width() -> f32 {
    0.0
}

pub(crate) fn prepare_main_window() {}

pub(crate) fn settings_uses_native_close() -> bool {
    false
}

pub(crate) fn settings_title_start() -> f32 {
    42.0
}

pub(crate) fn prepare_settings_window() {}

pub(crate) fn set_main_window_draggable(_draggable: bool) {}
