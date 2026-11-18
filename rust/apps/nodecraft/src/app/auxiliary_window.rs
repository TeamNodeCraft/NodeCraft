//! Shared shell for the secondary native windows (Settings and Debug).
//!
//! Both windows use the same native viewport builder, macOS full-size content
//! treatment, title bar, close light and resize edges. Keep those decisions here
//! so a chrome change applies to both.

use super::titlebar;
use crate::theme::Theme;
use crate::window_chrome::AuxiliaryWindow;

pub(super) const AUXILIARY_TITLEBAR_HEIGHT: f32 = 40.0;

/// The one viewport builder for secondary native windows.
pub(super) fn auxiliary_viewport_builder(
    title: &str,
    size: egui::Vec2,
    min_size: egui::Vec2,
) -> egui::ViewportBuilder {
    let builder = egui::ViewportBuilder::default()
        .with_title(title)
        .with_inner_size(size)
        .with_min_inner_size(min_size)
        .with_resizable(true)
        // Keep the auxiliary window above NodeCraft even after the main window
        // is clicked or activated.
        .with_always_on_top()
        // Windows and Linux use the custom title bar below; macOS keeps the
        // native window but hides its title bar in favour of full-size content.
        .with_decorations(cfg!(target_os = "macos"));
    #[cfg(target_os = "macos")]
    let builder = builder
        .with_fullsize_content_view(true)
        .with_title_shown(false)
        .with_titlebar_buttons_shown(false)
        .with_titlebar_shown(false);
    builder
}

pub(super) struct AuxiliaryWindowShell {
    /// The area below the title bar where the window body goes.
    pub content: egui::Rect,
    /// The custom close light was pressed.
    pub close: bool,
}

/// Draw the shared auxiliary-window background, resize edges and title bar.
pub(super) fn auxiliary_window_shell(
    ui: &mut egui::Ui,
    theme: &Theme,
    full_rect: egui::Rect,
    title: &str,
    id_salt: &str,
    window: AuxiliaryWindow,
) -> AuxiliaryWindowShell {
    // A native viewport has no egui::Window background of its own. Paint the
    // same fill Settings uses instead of leaving the platform default.
    ui.painter()
        .rect_filled(full_rect, 0.0, theme.color(theme.window));

    if let Some(direction) = titlebar::resize_direction(ui.ctx()) {
        ui.ctx().set_cursor_icon(titlebar::resize_cursor(direction));
        if ui.ctx().input(|input| input.pointer.primary_pressed()) {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
        }
    }

    let titlebar = egui::Rect::from_min_max(
        full_rect.min,
        egui::pos2(full_rect.max.x, full_rect.min.y + AUXILIARY_TITLEBAR_HEIGHT),
    );
    ui.painter()
        .rect_filled(titlebar, 0.0, theme.color(theme.window));
    crate::window_chrome::prepare_auxiliary_window(window);
    let close_center = egui::pos2(titlebar.min.x + 22.0, titlebar.center().y);
    let close = if crate::window_chrome::auxiliary_window_uses_native_close() {
        false
    } else {
        titlebar::close_light(
            ui,
            close_center,
            ui.make_persistent_id(("auxiliary-title-close", id_salt)),
        )
    };
    let title_start = crate::window_chrome::auxiliary_window_title_start();
    let drag_rect = egui::Rect::from_min_max(egui::pos2(title_start, titlebar.min.y), titlebar.max);
    if titlebar::drag_response(
        ui,
        drag_rect,
        ui.make_persistent_id(("auxiliary-title-drag", id_salt)),
    )
    .drag_started()
    {
        titlebar::start_drag(ui);
    }
    titlebar::title(
        ui,
        egui::pos2(title_start + 4.0, titlebar.center().y),
        title,
        theme.color(theme.text),
    );
    crate::window_chrome::round_auxiliary_window(window);
    AuxiliaryWindowShell {
        content: egui::Rect::from_min_max(
            egui::pos2(full_rect.min.x, titlebar.max.y),
            full_rect.max,
        ),
        close,
    }
}
