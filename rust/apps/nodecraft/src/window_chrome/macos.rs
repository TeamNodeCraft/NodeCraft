//! macOS keeps the native traffic lights over a fullsize custom app bar.

use egui::ViewportBuilder;

pub(crate) fn configure_main_viewport(viewport: ViewportBuilder) -> ViewportBuilder {
    viewport
        .with_decorations(true)
        .with_fullsize_content_view(true)
        .with_title_shown(false)
        .with_titlebar_buttons_shown(true)
        .with_titlebar_shown(false)
}

/// Width reserved at the left of the custom app bar for the native buttons.
pub(crate) fn main_bar_controls_width() -> f32 {
    78.0
}

/// Show the native traffic lights after winit has created the window.
pub(crate) fn prepare_main_window() {
    nodecraft_macos::show_main_traffic_lights();
}

pub(crate) fn settings_uses_native_close() -> bool {
    true
}

pub(crate) fn settings_title_start() -> f32 {
    42.0
}

pub(crate) fn prepare_settings_window() {
    nodecraft_macos::show_settings_close_light();
}

pub(crate) fn set_main_window_draggable(draggable: bool) {
    nodecraft_macos::set_main_window_movable(draggable);
}
