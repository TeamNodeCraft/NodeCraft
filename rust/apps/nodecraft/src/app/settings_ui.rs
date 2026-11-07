//! The application settings window.

use super::*;
use super::theme_editor::*;

impl NodeCraftApp {
    pub(super) fn show_app_settings(&mut self, ctx: &egui::Context) {
        let mut theme = crate::theme::get();
        let theme_before = theme.clone();
        let mut theme_selection_changed = false;
        let mut theme_error: Option<String> = None;
        let mut theme_rename_id = self.theme_rename_id.clone();
        let mut theme_rename_name = self.theme_rename_name.clone();
        let mut theme_rename_focus = self.theme_rename_focus;
        let mut tab = self.settings_tab;
        let mut view = self.viewport.view();
        let mut show_view_matrix = self.viewport.show_view_matrix();
        let mut show_gpu_info = self.viewport.show_gpu_info();
        let mut engine_threads = self.engine_threads;
        let mut mesher_threads = self.mesher_threads;
        let mut workload_profiler = self.workload_profiler;
        let mut kernel_fixes = self.kernel_fixes;
        let mut activity = self.activity;
        let mut feature_overlay = self.feature_overlay;
        let mut cable_routing = self.cable_routing;
        let mut close = false;
        let mut builder = egui::ViewportBuilder::default()
            .with_title("NodeCraft Settings")
            .with_inner_size([760.0, 620.0])
            .with_min_inner_size([520.0, 400.0])
            .with_resizable(true)
            // Keep settings above the main NodeCraft window even if the main
            // window is clicked or activated.
            .with_always_on_top()
            // Native OS chrome: keep close and minimize, hide maximize. The
            // platform owns the button order/side and rounded window corners.
            // macOS verified: close/minimize at left, no zoom, rounded corners.
            // TODO(future agent): verify native button placement and rounded
            // corners on Windows and Linux; keep OS chrome, not custom controls.
            .with_decorations(true)
            .with_close_button(true)
            .with_minimize_button(false)
            .with_maximize_button(false);
        let parent_outer = ctx.input(|input| input.viewport().outer_rect);
        let monitor_size = ctx.input(|input| input.viewport().monitor_size);
        let mut center_once = !self.settings_open_last;
        if !self.settings_open_last {
            self.settings_open_last = true;
            if let Some(parent) = parent_outer {
                let settings_size = egui::vec2(760.0, 620.0);
                let mut pos = parent.center() - settings_size * 0.5;
                if let Some(monitor) = monitor_size {
                    pos.x = pos.x.clamp(0.0, (monitor.x - settings_size.x).max(0.0));
                    pos.y = pos.y.clamp(0.0, (monitor.y - settings_size.y).max(0.0));
                }
                builder = builder.with_position(pos);
            }
        }
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("nodecraft-app-settings"),
            builder,
            |ui, _class| {
                if ui.ctx().input(|input| input.viewport().close_requested()) {
                    close = true;
                    return;
                }
                if center_once {
                    if let (Some(parent), Some(child)) = (
                        parent_outer,
                        ui.ctx().input(|input| input.viewport().outer_rect),
                    ) {
                        let mut pos = parent.center() - child.size() * 0.5;
                        if let Some(monitor) = monitor_size {
                            pos.x = pos.x.clamp(0.0, (monitor.x - child.width()).max(0.0));
                            pos.y = pos.y.clamp(0.0, (monitor.y - child.height()).max(0.0));
                        }
                        ui.ctx()
                            .send_viewport_cmd(egui::ViewportCommand::OuterPosition(pos));
                        center_once = false;
                    }
                }
                let panel = ui.max_rect();
                let left = egui::Rect::from_min_max(
                    panel.min,
                    egui::pos2(panel.min.x + 124.0, panel.max.y),
                );
                let right = egui::Rect::from_min_max(
                    egui::pos2(left.max.x, panel.min.y),
                    panel.max,
                );
                ui.painter().rect_filled(left, 0.0, theme.color(theme.panel));
                ui.painter().rect_filled(right, 0.0, theme.color(theme.window));

                let tabs = [
                    SettingsTab::Appearance,
                    SettingsTab::Viewport,
                    SettingsTab::Engine,
                    SettingsTab::Diagnostics,
                    SettingsTab::Keys,
                ];
                let mut tab_y = left.min.y + 12.0;
                for candidate in tabs {
                    let button = egui::Rect::from_min_size(
                        egui::pos2(left.min.x + 10.0, tab_y),
                        egui::vec2(104.0, 30.0),
                    );
                    let response = ui.interact(
                        button,
                        ui.make_persistent_id(("settings-tab", candidate.label())),
                        egui::Sense::click(),
                    );
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }
                    if response.clicked() {
                        tab = candidate;
                    }
                    if tab == candidate {
                        ui.painter().rect_filled(
                            button,
                            egui::CornerRadius::same(7),
                            theme.color(theme.accent),
                        );
                    }
                    ui.painter().text(
                        egui::pos2(button.min.x + 10.0, button.center().y),
                        egui::Align2::LEFT_CENTER,
                        candidate.label(),
                        egui::FontId::proportional(13.0),
                        if tab == candidate {
                            theme.color(theme.window)
                        } else {
                            theme.color(theme.muted)
                        },
                    );
                    tab_y += 36.0;
                }

                let palette_row_width = (right.width() - 64.0).max(300.0);
                let mut content = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(right.shrink2(egui::vec2(18.0, 14.0)))
                        .layout(egui::Layout::top_down(egui::Align::Min)),
                );
                match tab {
                    SettingsTab::Appearance => {
                        content.heading("Appearance");
                        let mut selected_theme = crate::theme::active_preset();
                        content.horizontal(|ui| {
                            ui.label("Theme");
                            let combo_width = (palette_row_width
                                - 42.0
                                - 78.0
                                - ui.spacing().item_spacing.x * 4.0)
                                .max(120.0);
                            if let Some(rename_id) = theme_rename_id.clone() {
                                let response = ui.add_sized(
                                    [combo_width, 24.0],
                                    egui::TextEdit::singleline(&mut theme_rename_name)
                                        .hint_text("Theme name"),
                                );
                                if theme_rename_focus {
                                    response.request_focus();
                                    theme_rename_focus = false;
                                }
                                if ui.ctx().input(|input| input.key_pressed(egui::Key::Escape)) {
                                    theme_rename_id = None;
                                } else {
                                    let commit = response.lost_focus()
                                        || (response.has_focus()
                                            && ui.input(|input| {
                                                input.key_pressed(egui::Key::Enter)
                                            }));
                                    if commit && !theme_rename_name.trim().is_empty() {
                                        if let Some(preset) = crate::theme::available_themes()
                                            .into_iter()
                                            .find(|preset| preset.id == rename_id)
                                        {
                                            match crate::theme::rename_theme(
                                                &preset,
                                                &theme_rename_name,
                                            ) {
                                                Ok(renamed) => {
                                                    selected_theme = renamed;
                                                    theme_rename_id = None;
                                                    theme_selection_changed = true;
                                                }
                                                Err(error) => theme_error = Some(error),
                                            }
                                        }
                                    }
                                }
                            } else {
                                egui::ComboBox::from_id_salt("theme-preset")
                                    .width(combo_width)
                                    .popup_style(egui::style::StyleModifier::from(crate::theme::menu_style))
                                    .selected_text(crate::theme::active_name())
                                    .show_ui(ui, |ui| {
                                        for preset in crate::theme::available_themes() {
                                            if ui
                                                .selectable_label(
                                                    selected_theme.id == preset.id,
                                                    &preset.name,
                                                )
                                                .clicked()
                                                && selected_theme.id != preset.id
                                            {
                                                match crate::theme::load_preset(&preset) {
                                                    Ok(loaded) => {
                                                        theme = loaded;
                                                        selected_theme = preset;
                                                        theme_selection_changed = true;
                                                    }
                                                    Err(error) => theme_error = Some(error),
                                                }
                                            }
                                        }
                                    });
                            }
                            let can_reset = theme != crate::theme::active_defaults();
                            let reset = theme_revert_button(ui, can_reset)
                                .on_hover_text(if can_reset {
                                    "Reset current theme to defaults"
                                } else {
                                    "Theme already matches defaults"
                                });
                            if reset.clicked() && can_reset {
                                match crate::theme::reset_current_theme() {
                                    Ok(loaded) => {
                                        theme = loaded;
                                        selected_theme = crate::theme::active_preset();
                                        theme_selection_changed = true;
                                    }
                                    Err(error) => theme_error = Some(error),
                                }
                            }
                            let active_preset = crate::theme::active_preset();
                            let can_edit = active_preset.path.is_some();
                            let edit = theme_edit_button(ui, can_edit)
                                .on_hover_text(if can_edit {
                                    "Edit theme name"
                                } else {
                                    "Built-in themes cannot be renamed"
                                });
                            if edit.clicked() && can_edit {
                                theme_rename_id = Some(active_preset.id);
                                theme_rename_name = active_preset.name;
                                theme_rename_focus = true;
                            }
                            let add = theme_add_button(ui).on_hover_text("New theme");
                            if add.clicked() {
                                match crate::theme::create_theme("New Theme", &theme) {
                                    Ok(preset) => {
                                        selected_theme = preset.clone();
                                        theme_rename_id = Some(preset.id);
                                        theme_rename_name = preset.name;
                                        theme_rename_focus = true;
                                        theme_selection_changed = true;
                                    }
                                    Err(error) => {
                                        eprintln!("theme: cannot create user theme: {error}");
                                        theme_error = Some(error);
                                    }
                                }
                            }
                        });
                        if let Some(error) = &theme_error {
                            content.small(egui::RichText::new(error).color(theme.color(theme.text)));
                        }
                        content.small(
                            "Click a colour dot to edit. Edits apply live and are saved as \
                             NodeCraft Dark (custom).",
                        );
                        content.add_space(8.0);
                        content.horizontal(|ui| {
                            ui.label("Cable routing");
                            egui::ComboBox::from_id_salt("cable-routing")
                                .selected_text(match cable_routing {
                                    CableRouting::J => "j-routing",
                                    CableRouting::Bezier => "bezier-routing",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut cable_routing,
                                        CableRouting::J,
                                        "j-routing",
                                    );
                                    ui.selectable_value(
                                        &mut cable_routing,
                                        CableRouting::Bezier,
                                        "bezier-routing",
                                    );
                                });
                        });
                        content.small(
                            "Bezier routing keeps the original smooth cable curves.",
                        );
                        content.add_space(6.0);
                        content.scope(|ui| {
                            ui.style_mut().spacing.scroll = crate::theme::scroll_style_general();
                            ui.visuals_mut().widgets.inactive.bg_fill =
                                theme.with_alpha(theme.muted, 70);
                            ui.visuals_mut().widgets.hovered.bg_fill =
                                theme.with_alpha(theme.muted, 155);
                            ui.visuals_mut().widgets.active.bg_fill =
                                theme.with_alpha(theme.accent, 210);
                            ui.visuals_mut().widgets.inactive.corner_radius =
                                egui::CornerRadius::same(4);
                            ui.visuals_mut().widgets.hovered.corner_radius =
                                egui::CornerRadius::same(4);
                            ui.visuals_mut().widgets.active.corner_radius =
                                egui::CornerRadius::same(4);
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    for group in crate::theme::palette_groups(&theme) {
                                        egui::Frame::new()
                                            .fill(theme.color(theme.panel))
                                            .corner_radius(egui::CornerRadius::same(7))
                                            .inner_margin(egui::Margin::same(8))
                                            .show(ui, |ui| {
                                                ui.set_min_width(palette_row_width + 16.0);
                                                ui.label(egui::RichText::new(group.title).strong());
                                            ui.small(group.description);
                                            ui.add_space(3.0);
                                            for entry in &group.entries {
                                                theme_palette_row(ui, entry, &mut theme, palette_row_width);
                                            }
                                        });
                                    ui.add_space(6.0);
                                    }
                                });
                        });
                    }
                    SettingsTab::Viewport => {
                        content.heading("Viewport");
                        content.checkbox(&mut view.cad_mouse, "CAD mouse (middle orbit, Shift+middle pan)");
                        content.add(
                            egui::Slider::new(&mut view.pan_speed, 0.25..=4.0)
                                .text("Pan speed")
                                .max_decimals(2),
                        );
                        content.add(
                            egui::Slider::new(&mut view.orbit_speed, 0.25..=4.0)
                                .text("Orbit speed")
                                .max_decimals(2),
                        );
                        content.checkbox(&mut view.translucent, "Translucent");
                        content.checkbox(&mut view.edges, "Edges");
                        content.checkbox(&mut view.triangles, "Triangles");
                        content.checkbox(&mut view.normals, "Face normals");
                        content.checkbox(&mut view.face_colours, "Face colours");
                        content.checkbox(&mut view.zebra, "Zebra analysis");
                        content.checkbox(
                            &mut feature_overlay,
                            "Editing action overlay",
                        );
                        content.small("Show translucent added/removed surfaces while a feature node is selected.");
                    }
                    SettingsTab::Engine => {
                        content.heading("Engine");
                        content.small(
                            "Worker threads used by the SolidCore kernel pool.                              0 uses the available CPU parallelism. Changes apply on restart.",
                        );
                        let mut threads = engine_threads as u32;
                        content.add(
                            egui::Slider::new(&mut threads, 0..=64)
                                .text("Exact kernel threads (0 = auto)")
                                .integer(),
                        );
                        engine_threads = threads as usize;
                        let mut mesher = mesher_threads as u32;
                        content.add(
                            egui::Slider::new(&mut mesher, 0..=64)
                                .text("Mesher threads (0 = auto)")
                                .integer(),
                        );
                        mesher_threads = mesher as usize;
                        content.monospace(format!(
                            "Exact pool: {}",
                            if nodecraft_engine::kernel_pool::exact_pool_initialized() {
                                "already started"
                            } else {
                                "lazy"
                            }
                        ));
                        content.monospace(format!(
                            "Mesh pool: {}",
                            if nodecraft_engine::mesher::mesher_initialized() {
                                "already started"
                            } else {
                                "lazy"
                            }
                        ));
                        content.monospace(format!(
                            "Available CPU parallelism: {}",
                            std::thread::available_parallelism()
                                .map(|count| count.get())
                                .unwrap_or(1)
                        ));
                        content.add_space(8.0);
                        content.checkbox(
                            &mut workload_profiler,
                            "Adaptive workload dispatch (profiler)",
                        );
                        content.small(
                            "Chooses how many worker threads each evaluation uses. Narrow graphs use \
                             fewer threads to avoid overhead, so it can be slower than using every \
                             configured thread. Turn off for maximum throughput on complex graphs.",
                        );
                        content.add_space(8.0);
                        content.checkbox(
                            &mut kernel_fixes,
                            "Latest SolidCore kernel fixes",
                        );
                        content.small(
                            "Enables the newer offset, fillet and B-rep classification fixes. \
                             Applies when the kernel workers start; restart to change an existing pool.",
                        );
                    }
                    SettingsTab::Keys => {
                        content.heading("Key commands");
                        content.small(
                            "Tool dock keys only act after clicking the dock or pressing Command-M.",
                        );
                        content.add_space(10.0);
                        for (label, shortcut) in [
                            ("Tool dock keyboard navigation", &mut self.tool_dock_shortcut),
                            ("Search tools", &mut self.tool_search_shortcut),
                        ] {
                            content.horizontal(|ui| {
                                ui.add_sized(
                                    [210.0, 22.0],
                                    egui::Label::new(label),
                                );
                                let changed = ui
                                    .add_sized(
                                        [110.0, 24.0],
                                        egui::TextEdit::singleline(shortcut)
                                            .hint_text("Cmd+M"),
                                    )
                                    .changed();
                                if changed {
                                    self.view_save_due = Some(
                                        Instant::now()
                                            + std::time::Duration::from_millis(500),
                                    );
                                }
                            });
                        }
                    }
                    SettingsTab::Diagnostics => {
                        content.heading("Diagnostics");
                        content.monospace(format!("NodeCraft {}", env!("CARGO_PKG_VERSION")));
                        content.monospace(format!("NodeCraft Git Hash: {BUILD_GIT_HASH}"));
                        content.monospace(format!("SolidCore Git Hash: {SOLIDCORE_GIT_HASH}"));
                        #[cfg(feature = "gpu-viewport")]
                        content.monospace(format!("MSAA: {}x", crate::gpu_viewport::msaa_samples()));
                        content.checkbox(
                            &mut show_view_matrix,
                            "Show view matrix / camera debug",
                        );
                        content.checkbox(
                            &mut show_gpu_info,
                            "Show GPU / frame debug",
                        );
                        content.add_space(8.0);
                        content.heading("Node activity glow");
                        content.small(
                            "Diagnostic pulse shown while each node is processing. \n\
                             Hold is applied from node start and is at least 500 ms.",
                        );
                        content.add(
                            egui::Slider::new(&mut activity.fade_in_ms, 0..=500)
                                .text("Fade in (ms)"),
                        );
                        content.add(
                            egui::Slider::new(&mut activity.hold_ms, 16..=160)
                                .text("Hold (ms, 16–160)"),
                        );
                        content.add(
                            egui::Slider::new(&mut activity.fade_out_ms, 0..=500)
                                .text("Fade out (ms)"),
                        );
                    }
                }
                #[cfg(target_os = "macos")]
                nodecraft_macos::hide_settings_zoom_button();
            },
        );
        self.theme_rename_id = theme_rename_id;
        self.theme_rename_name = theme_rename_name;
        self.theme_rename_focus = theme_rename_focus;
        if theme_selection_changed {
            self.theme_dirty = false;
            self.theme_save_due = None;
        }
        if theme != theme_before {
            crate::theme::set(theme.clone());
            configure_style(ctx);
            self.node_editor.set_theme(canvas_theme(&theme));
            self.viewport.set_theme(&theme);
            if !theme_selection_changed {
                self.theme_dirty = true;
                self.theme_save_due =
                    Some(Instant::now() + std::time::Duration::from_millis(350));
            }
            ctx.request_repaint();
        }
        if view != self.viewport.view() {
            self.viewport.set_view(view);
        }
        self.viewport.set_show_view_matrix(show_view_matrix);
        self.viewport.set_show_gpu_info(show_gpu_info);
        if cable_routing != self.cable_routing {
            self.cable_routing = cable_routing;
            self.node_editor.set_cable_routing(cable_routing);
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if feature_overlay != self.feature_overlay {
            self.feature_overlay = feature_overlay;
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if engine_threads != self.engine_threads {
            self.engine_threads = engine_threads;
            nodecraft_engine::kernel_pool::configure_exact_threads(engine_threads);
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if mesher_threads != self.mesher_threads {
            self.mesher_threads = mesher_threads;
            nodecraft_engine::kernel_pool::configure_mesh_threads(mesher_threads);
            nodecraft_engine::mesher::configure_mesher_threads(mesher_threads);
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if workload_profiler != self.workload_profiler {
            self.workload_profiler = workload_profiler;
            nodecraft_engine::configure_workload_profiler(workload_profiler);
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if kernel_fixes != self.kernel_fixes {
            self.kernel_fixes = kernel_fixes;
            nodecraft_engine::configure_solidcore_fixes(kernel_fixes);
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if activity != self.activity {
            self.activity = activity;
            self.node_editor.set_activity_timings(
                u64::from(activity.fade_in_ms),
                u64::from(activity.hold_ms),
                u64::from(activity.fade_out_ms),
            );
            self.view_save_due = Some(Instant::now() + std::time::Duration::from_millis(350));
            ctx.request_repaint();
        }
        if close {
            self.settings_open = false;
            self.settings_open_last = false;
        }
        self.settings_tab = tab;
    }

}
