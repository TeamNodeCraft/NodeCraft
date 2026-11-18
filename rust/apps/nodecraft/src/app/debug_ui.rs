//! The single Debug native window: Frame, GPU and Camera tabs.
//!
//! The window replaces the three old Diagnostics switches (camera overlay, GPU
//! overlay and frame profiler). Its one persisted switch is `debug_window`; a
//! close request clears that same value. Profiling is tied to the window being
//! open, not to the selected tab, so the Frame graph keeps its history while
//! GPU or Camera is shown.

use super::settings_ui::paint_tab;
use super::*;
use crate::file_dialog::{FileDialogKind, FileDialogResult, PendingFileDialog};
use crate::gpu_profiler::{GpuProfilerSlot, SERIES, Series};
use crate::viewport::DebugRow;
use egui_plot::{HLine, Line, LineStyle, Plot, PlotPoints};

pub(super) const DEBUG_WINDOW_SIZE: egui::Vec2 = egui::vec2(620.0, 520.0);
pub(super) const DEBUG_WINDOW_MIN_SIZE: egui::Vec2 = egui::vec2(480.0, 400.0);
pub(super) const DEBUG_WINDOW_TITLE: &str = "Debug";
const DEBUG_TAB_WIDTH: f32 = 84.0;
const DEBUG_TAB_HEIGHT: f32 = 28.0;
const DEBUG_TAB_GAP: f32 = 6.0;

/// The Debug window's own OS viewport, distinct from the main window and
/// Settings.
pub(super) fn debug_window_viewport_id() -> egui::ViewportId {
    egui::ViewportId::from_hash_of("nodecraft-debug-window")
}

/// The embedded fallback's egui::Window id.
fn debug_window_window_id() -> egui::Id {
    egui::Id::new("nodecraft-debug-window")
}

fn debug_window_close_shortcut() -> egui::KeyboardShortcut {
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::W)
}

#[derive(Default)]
struct DebugWindowActions {
    save_clicked: bool,
    pause_request: Option<bool>,
}

pub(super) struct DebugWindowShell<T> {
    /// The OS window, its close light or Cmd-W asked to close.
    pub close_requested: bool,
    /// The body result, or `None` when the setting is off.
    pub body: Option<T>,
}

/// Draw the Debug window in its own viewport when the setting is on.
///
/// `enabled` is the app setting itself, so a close request clears the one
/// source of truth rather than a separate "window open" flag. When egui reports
/// embedded viewports (web or a backend without native windows), the fallback is
/// the ordinary egui::Window.
pub(super) fn show_debug_window_shell<T>(
    ctx: &egui::Context,
    enabled: &mut bool,
    mut body: impl FnMut(&mut egui::Ui) -> T,
) -> DebugWindowShell<T> {
    if !*enabled {
        return DebugWindowShell {
            close_requested: false,
            body: None,
        };
    }

    if ctx.embed_viewports() {
        return show_embedded_debug_window(ctx, enabled, body);
    }

    let theme = crate::theme::get();
    let mut close_requested = false;
    let mut body_result = None;
    ctx.show_viewport_immediate(
        debug_window_viewport_id(),
        auxiliary_viewport_builder(DEBUG_WINDOW_TITLE, DEBUG_WINDOW_SIZE, DEBUG_WINDOW_MIN_SIZE),
        |ui, _class| {
            if ui.ctx().input(|input| input.viewport().close_requested()) {
                close_requested = true;
                return;
            }
            if ui
                .ctx()
                .input_mut(|input| input.consume_shortcut(&debug_window_close_shortcut()))
            {
                close_requested = true;
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }

            let full_rect = ui.max_rect();
            let chrome = auxiliary_window_shell(
                ui,
                &theme,
                full_rect,
                DEBUG_WINDOW_TITLE,
                "debug-window",
                crate::window_chrome::AuxiliaryWindow::Debug,
            );
            if chrome.close {
                close_requested = true;
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
            let mut content = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(chrome.content.shrink2(egui::vec2(12.0, 10.0)))
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            body_result = Some(body(&mut content));
        },
    );
    if close_requested {
        *enabled = false;
    }
    DebugWindowShell {
        close_requested,
        body: body_result,
    }
}

fn show_embedded_debug_window<T>(
    ctx: &egui::Context,
    enabled: &mut bool,
    mut body: impl FnMut(&mut egui::Ui) -> T,
) -> DebugWindowShell<T> {
    let mut open = true;
    let mut body_result = None;
    egui::Window::new(DEBUG_WINDOW_TITLE)
        .id(debug_window_window_id())
        .default_size(DEBUG_WINDOW_SIZE)
        .min_size(DEBUG_WINDOW_MIN_SIZE)
        .resizable(true)
        .open(&mut open)
        .show(ctx, |ui| {
            body_result = Some(body(ui));
        });
    if !open {
        *enabled = false;
    }
    DebugWindowShell {
        close_requested: !open,
        body: body_result,
    }
}

impl NodeCraftApp {
    /// Turn the Debug window on/off. Called by the Diagnostics switch and when
    /// the native window's close button or Cmd-W is used.
    ///
    /// Turning it on enables timestamp profiling for as long as the window is
    /// open. The selected tab never gates profiling: switching to GPU or Camera
    /// leaves the Frame graph collecting history.
    pub(super) fn set_debug_window(&mut self, enabled: bool) {
        self.debug_window = enabled;
        if let Some(slot) = &self.gpu_profiler_slot
            && let Ok(mut slot) = slot.lock()
        {
            if enabled {
                slot.enable();
                slot.set_paused(false);
            } else {
                slot.disable();
            }
        }
        let available = self.gpu_profiler_slot.is_some();
        self.viewport.set_gpu_profiler_enabled(enabled && available);
    }

    /// Poll the asynchronous trace-save picker. One `Option` check when no
    /// picker is open.
    pub(super) fn poll_profiler_trace(&mut self) {
        let Some(result) = self
            .pending_profiler_trace
            .as_mut()
            .and_then(PendingFileDialog::try_finish)
        else {
            return;
        };
        self.pending_profiler_trace = None;
        let FileDialogResult::Chosen(path) = result else {
            self.profiler_window.status = Some("Trace save cancelled".to_owned());
            return;
        };
        if !self.debug_window {
            self.profiler_window.status = Some("Trace save cancelled".to_owned());
            return;
        }
        let result = match &self.gpu_profiler_slot {
            Some(slot) => match slot.lock() {
                Ok(slot) => slot.save_trace(&path),
                Err(_) => Err("profiler state is unavailable".to_owned()),
            },
            None => Err("profiler state is unavailable".to_owned()),
        };
        self.profiler_window.status = Some(match result {
            Ok(()) => format!("Saved trace: {}", path.display()),
            Err(error) => format!("Trace failed: {error}"),
        });
    }

    /// Start a profiler frame and harvest the previous frame's completed GPU
    /// results. One `bool` check when the window is off.
    pub(super) fn profiler_tick(&mut self, ctx: &egui::Context) {
        if !self.debug_window {
            return;
        }
        let frame_ms = ctx.input(|input| input.stable_dt).clamp(1.0e-4, 1.0) as f32 * 1000.0;
        let upload_bytes = self.viewport.gpu_object_bytes();
        let Some(slot) = self.gpu_profiler_slot.clone() else {
            return;
        };
        if let Ok(mut slot) = slot.lock() {
            slot.finish_frame();
            slot.begin_frame(frame_ms, upload_bytes);
        }
    }

    /// Record a CPU-side timing for the current profiler frame. One `bool`
    /// check when the window is off.
    pub(super) fn profiler_record(&self, series: Series, milliseconds: f64) {
        if !self.debug_window {
            return;
        }
        let Some(slot) = &self.gpu_profiler_slot else {
            return;
        };
        if let Ok(mut slot) = slot.lock() {
            slot.record(series, milliseconds as f32);
        }
    }

    /// The one Debug window. It exists only while the Diagnostics switch is on;
    /// a close request clears that same setting.
    pub(super) fn show_debug_window(&mut self, ctx: &egui::Context) {
        if !self.debug_window {
            return;
        }
        let Some(slot_arc) = self.gpu_profiler_slot.clone() else {
            return;
        };
        let Ok(mut slot) = slot_arc.lock() else {
            return;
        };
        let theme = crate::theme::get();
        let mut enabled = self.debug_window;
        let tab_before = self.debug_tab;
        let shell = {
            let window = &mut self.profiler_window;
            let viewport = &self.viewport;
            let tab = &mut self.debug_tab;
            show_debug_window_shell(ctx, &mut enabled, |ui| {
                debug_window_body(ui, tab, window, &mut slot, viewport, &theme)
            })
        };
        let close_requested = shell.close_requested;
        let actions = shell.body.unwrap_or_default();
        if let Some(paused) = actions.pause_request {
            slot.set_paused(paused);
        }
        drop(slot);

        if self.debug_tab != tab_before {
            self.schedule_debug_window_save(ctx);
        }
        if actions.save_clicked {
            self.save_profiler_trace();
        }
        if close_requested && self.debug_window {
            self.set_debug_window(false);
            self.schedule_debug_window_save(ctx);
        }
    }

    /// Persist the switch and selected tab through the same debounced settings
    /// writer that the Diagnostics switch and the window close use.
    pub(super) fn schedule_debug_window_save(&mut self, ctx: &egui::Context) {
        self.view_save_due = Some(Instant::now() + Duration::from_millis(350));
        ctx.request_repaint();
        ctx.request_repaint_after(Duration::from_millis(350));
    }

    fn save_profiler_trace(&mut self) {
        // `rfd`'s synchronous panel must not start inside winit's event
        // handler; `PendingFileDialog` owns the worker-thread pattern.
        self.pending_profiler_trace = Some(PendingFileDialog::spawn(
            FileDialogKind::SaveProfilerTrace {
                suggested_name: "nodecraft-frame-trace.json".to_owned(),
            },
        ));
        self.profiler_window.status = Some("Choose a trace path…".to_owned());
    }
}

/// The window body: one shared tab row, then the selected tab's content.
fn debug_window_body(
    ui: &mut egui::Ui,
    tab: &mut DebugTab,
    window: &mut crate::gpu_profiler::ProfilerWindow,
    slot: &mut GpuProfilerSlot,
    viewport: &Viewport,
    theme: &crate::theme::Theme,
) -> DebugWindowActions {
    ui.set_min_width(440.0);
    debug_tab_row(ui, theme, tab);
    ui.add_space(8.0);
    match *tab {
        DebugTab::Frame => frame_tab(ui, window, slot, theme),
        DebugTab::Gpu => gpu_tab(ui, slot, viewport, theme),
        DebugTab::Camera => camera_tab(ui, viewport, theme),
    }
}

/// The Debug window's tab row. It uses the same accent-filled tab painter as
/// Settings' sidebar so the two surfaces cannot drift apart.
fn debug_tab_row(ui: &mut egui::Ui, theme: &crate::theme::Theme, active: &mut DebugTab) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = DEBUG_TAB_GAP;
        for tab in DebugTab::ALL {
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(DEBUG_TAB_WIDTH, DEBUG_TAB_HEIGHT),
                egui::Sense::click(),
            );
            if response.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            paint_tab(ui, rect, tab.label(), *active == tab, theme);
            if response.clicked() {
                *active = tab;
            }
        }
    });
}

fn debug_rows_table(ui: &mut egui::Ui, id: &str, rows: &[DebugRow], theme: &crate::theme::Theme) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([18.0, 5.0])
        .striped(true)
        .show(ui, |ui| {
            for row in rows {
                ui.label(egui::RichText::new(row.label).color(theme.color(theme.muted)));
                ui.monospace(&row.value);
                ui.end_row();
            }
        });
}

/// GPU/frame diagnostics previously painted over the viewport. The frame time
/// comes from the profiler's CPU frame series, so this tab adds no measurement
/// of its own.
fn gpu_tab(
    ui: &mut egui::Ui,
    slot: &GpuProfilerSlot,
    viewport: &Viewport,
    theme: &crate::theme::Theme,
) -> DebugWindowActions {
    let frame = slot.stats(Series::CpuFrame);
    let rows = viewport.gpu_debug_rows(frame.current);
    egui::ScrollArea::vertical()
        .id_salt("debug-gpu-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            debug_rows_table(ui, "debug-gpu-rows", &rows, theme);
            ui.add_space(6.0);
            ui.small(
                "Frame time is the profiler's CPU frame series. GPU pass timings \
                 are in the Frame tab.",
            );
        });
    DebugWindowActions::default()
}

/// The camera and view-matrix diagnostics previously painted over the viewport.
fn camera_tab(
    ui: &mut egui::Ui,
    viewport: &Viewport,
    theme: &crate::theme::Theme,
) -> DebugWindowActions {
    let rows = viewport.camera_debug_rows();
    egui::ScrollArea::vertical()
        .id_salt("debug-camera-scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            debug_rows_table(ui, "debug-camera-rows", &rows, theme);
        });
    DebugWindowActions::default()
}

fn frame_tab(
    ui: &mut egui::Ui,
    window: &mut crate::gpu_profiler::ProfilerWindow,
    slot: &mut GpuProfilerSlot,
    theme: &crate::theme::Theme,
) -> DebugWindowActions {
    let mut actions = DebugWindowActions::default();
    if let Some(error) = slot.last_error() {
        ui.small(error);
    } else if !slot.timestamps_supported() {
        ui.small("This adapter has no GPU timestamp queries; showing CPU timings only.");
    }
    ui.horizontal(|ui| {
        if ui
            .button(if slot.is_paused() { "Resume" } else { "Pause" })
            .clicked()
        {
            actions.pause_request = Some(!slot.is_paused());
        }
        if ui
            .button("Save trace")
            .on_hover_text(
                "Chrome-trace JSON with the GPU pass scopes and CPU series, \
                 for Perfetto or chrome://tracing",
            )
            .clicked()
        {
            actions.save_clicked = true;
        }
        let frame = slot.stats(Series::CpuFrame);
        ui.small(format!(
            "last {:.2} ms   avg {:.2} ms",
            frame.current, frame.average
        ));
    });

    let plot_height = (ui.available_height() - 250.0).clamp(90.0, 180.0);
    // Fill the reusable buffers before the plot borrows them; the plot items
    // keep a borrow until the plot ends.
    for series in SERIES {
        if window.visible[series.index()] {
            let points = &mut window.plot_points[series.index()];
            if points.capacity() < crate::gpu_profiler::WINDOW_FRAMES {
                points.reserve(crate::gpu_profiler::WINDOW_FRAMES);
            }
            slot.plot_points(series, points);
        }
    }
    Plot::new("debug-frame-graph")
        .height(plot_height)
        .include_y(0.0)
        .include_y(16.7)
        .allow_zoom(false)
        .allow_drag(false)
        .allow_scroll(false)
        .show_axes([true, true])
        .show_grid([true, true])
        .x_axis_label("frames")
        .y_axis_label("ms")
        .show(ui, |plot_ui| {
            for series in SERIES {
                if !window.visible[series.index()] {
                    continue;
                }
                let points = &window.plot_points[series.index()];
                if points.is_empty() {
                    continue;
                }
                plot_ui.line(
                    Line::new(series.label(), PlotPoints::Borrowed(points.as_slice()))
                        .color(series.color(theme))
                        .width(1.5),
                );
            }
            plot_ui.hline(
                HLine::new("16.7 ms", 16.7)
                    .color(theme.color(theme.muted))
                    .width(1.0)
                    .style(LineStyle::dashed_dense()),
            );
        });

    ui.add_space(4.0);
    egui::Grid::new("debug-frame-series")
        .num_columns(5)
        .spacing([10.0, 3.0])
        .striped(true)
        .show(ui, |ui| {
            ui.label("");
            ui.label(egui::RichText::new("Series").strong());
            ui.label(egui::RichText::new("Now").strong());
            ui.label(egui::RichText::new("Avg").strong());
            ui.label(egui::RichText::new("Max").strong());
            ui.end_row();
            for series in SERIES {
                let stats = slot.stats(series);
                crate::widgets::switch(ui, &mut window.visible[series.index()]);
                ui.horizontal(|ui| {
                    ui.colored_label(series.color(theme), "●");
                    ui.label(series.label());
                });
                ui.monospace(format!("{:.2}", stats.current));
                ui.monospace(format!("{:.2}", stats.average));
                ui.monospace(format!("{:.2}", stats.maximum));
                ui.end_row();
            }
        });
    let counters = slot.counters();
    ui.small(format!(
        "Upload {:.1} KB/frame   ·   {} frames retained   ·   {} scopes / {} resolves",
        slot.upload_kb(),
        slot.retained_frames(),
        counters.scopes,
        counters.resolves,
    ));
    ui.small(
        "GPU series are pass-boundary timer queries on the passes NodeCraft                          encodes. egui's own pass is not ours to instrument; UI update covers it.",
    );
    if let Some(status) = &window.status {
        ui.small(status);
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    struct ViewportRequest {
        id: egui::ViewportId,
        title: Option<String>,
        inner_size: Option<egui::Vec2>,
        min_inner_size: Option<egui::Vec2>,
        resizable: Option<bool>,
    }

    fn install_renderer(requests: Arc<Mutex<Vec<ViewportRequest>>>, close_requested: bool) {
        egui::Context::set_immediate_viewport_renderer(move |ctx, mut viewport| {
            let builder = viewport.builder.clone();
            requests
                .lock()
                .expect("viewport request log")
                .push(ViewportRequest {
                    id: viewport.ids.this,
                    title: builder.title.clone(),
                    inner_size: builder.inner_size,
                    min_inner_size: builder.min_inner_size,
                    resizable: builder.resizable,
                });
            let screen = builder
                .inner_size
                .unwrap_or_else(|| egui::vec2(800.0, 600.0));
            let mut input = egui::RawInput {
                viewport_id: viewport.ids.this,
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen)),
                ..Default::default()
            };
            input.viewports.insert(
                viewport.ids.this,
                egui::ViewportInfo {
                    parent: Some(viewport.ids.parent),
                    title: builder.title.clone(),
                    events: if close_requested {
                        vec![egui::ViewportEvent::Close]
                    } else {
                        Vec::new()
                    },
                    ..Default::default()
                },
            );
            let mut output = ctx.run_ui(input, |ui| (viewport.viewport_ui_cb)(ui));
            output.textures_delta.clear();
        });
    }

    fn root_input() -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        }
    }

    fn run_shell(
        ctx: &egui::Context,
        enabled: &mut bool,
        body_draws: &mut usize,
    ) -> DebugWindowShell<()> {
        let mut shell = None;
        let mut output = ctx.run_ui(root_input(), |_ui| {
            shell = Some(show_debug_window_shell(ctx, enabled, |ui| {
                *body_draws += 1;
                ui.label("debug body");
            }));
        });
        output.textures_delta.clear();
        shell.expect("shell result")
    }

    fn frame_texts(frame: &egui::FullOutput) -> Vec<String> {
        fn collect(shape: &egui::Shape, texts: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(shape) => texts.push(shape.galley.text().to_owned()),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, texts);
                    }
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for clipped in &frame.shapes {
            collect(&clipped.shape, &mut texts);
        }
        texts
    }

    fn run_body(tab: DebugTab) -> Vec<String> {
        let ctx = egui::Context::default();
        let theme = crate::theme::get();
        let mut window = crate::gpu_profiler::ProfilerWindow::new();
        let mut slot = GpuProfilerSlot::new(None, None, false, None);
        slot.enable();
        let viewport = Viewport::default();
        let mut active = tab;
        let mut output = ctx.run_ui(root_input(), |ui| {
            debug_window_body(ui, &mut active, &mut window, &mut slot, &viewport, &theme);
        });
        output.textures_delta.clear();
        frame_texts(&output)
    }

    fn click_tab(active: &mut DebugTab, index: usize) -> bool {
        let ctx = egui::Context::default();
        let theme = crate::theme::get();
        let at = egui::pos2(
            index as f32 * (DEBUG_TAB_WIDTH + DEBUG_TAB_GAP) + DEBUG_TAB_WIDTH * 0.5,
            DEBUG_TAB_HEIGHT * 0.5,
        );
        let mut moved = root_input();
        moved.events.push(egui::Event::PointerMoved(at));
        let mut output = ctx.run_ui(moved, |ui| {
            debug_tab_row(ui, &theme, active);
        });
        output.textures_delta.clear();

        let mut press = root_input();
        press.events.push(egui::Event::PointerMoved(at));
        press.events.push(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        });
        let mut output = ctx.run_ui(press, |ui| {
            debug_tab_row(ui, &theme, active);
        });
        output.textures_delta.clear();

        let mut release = root_input();
        release.events.push(egui::Event::PointerMoved(at));
        release.events.push(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        let mut changed = false;
        let mut output = ctx.run_ui(release, |ui| {
            changed = {
                let before = *active;
                debug_tab_row(ui, &theme, active);
                *active != before
            };
        });
        output.textures_delta.clear();
        changed
    }

    #[test]
    fn one_switch_opens_one_debug_window() {
        let ctx = egui::Context::default();
        ctx.set_embed_viewports(false);
        let requests = Arc::new(Mutex::new(Vec::new()));
        install_renderer(requests.clone(), false);

        let mut enabled = false;
        let mut body_draws = 0;
        let shell = run_shell(&ctx, &mut enabled, &mut body_draws);
        assert!(shell.body.is_none());
        assert_eq!(body_draws, 0);
        assert!(requests.lock().expect("viewport request log").is_empty());

        enabled = true;
        let shell = run_shell(&ctx, &mut enabled, &mut body_draws);
        assert!(shell.body.is_some());
        assert_eq!(body_draws, 1);
        let requests = requests.lock().expect("viewport request log");
        assert_eq!(requests.len(), 1, "one switch opens one window");
        assert_eq!(requests[0].id, debug_window_viewport_id());
        assert_eq!(requests[0].title.as_deref(), Some(DEBUG_WINDOW_TITLE));
        assert_eq!(requests[0].inner_size, Some(DEBUG_WINDOW_SIZE));
        assert_eq!(requests[0].min_inner_size, Some(DEBUG_WINDOW_MIN_SIZE));
        assert_eq!(requests[0].resizable, Some(true));
    }

    #[test]
    fn a_close_request_on_the_debug_window_turns_the_setting_off() {
        let ctx = egui::Context::default();
        ctx.set_embed_viewports(false);
        let requests = Arc::new(Mutex::new(Vec::new()));
        install_renderer(requests.clone(), true);

        let mut enabled = true;
        let mut body_draws = 0;
        let shell = run_shell(&ctx, &mut enabled, &mut body_draws);
        assert!(shell.close_requested);
        assert!(!enabled, "closing the window clears the one setting");
        assert_eq!(body_draws, 0);

        let requested = requests.lock().expect("viewport request log").len();
        let shell = run_shell(&ctx, &mut enabled, &mut body_draws);
        assert!(shell.body.is_none());
        assert_eq!(
            requests.lock().expect("viewport request log").len(),
            requested,
            "an off setting must not request the viewport again"
        );
    }

    #[test]
    fn embedded_backends_draw_an_egui_window() {
        let ctx = egui::Context::default();
        ctx.set_embed_viewports(true);
        let mut enabled = true;
        let mut body_draws = 0;

        let shell = run_shell(&ctx, &mut enabled, &mut body_draws);
        assert!(shell.body.is_some());
        assert!(!shell.close_requested);
        assert_eq!(body_draws, 1);
        assert!(
            ctx.memory(|memory| memory.area_rect(debug_window_window_id()).is_some()),
            "the fallback is a real egui::Window, not a bare viewport"
        );
    }

    #[test]
    fn clicking_a_tab_switches_the_content() {
        let mut active = DebugTab::Frame;
        assert!(click_tab(&mut active, 1));
        assert_eq!(active, DebugTab::Gpu);
        assert!(click_tab(&mut active, 2));
        assert_eq!(active, DebugTab::Camera);
        assert!(click_tab(&mut active, 0));
        assert_eq!(active, DebugTab::Frame);
    }

    #[test]
    fn each_tab_renders_its_own_content() {
        let frame = run_body(DebugTab::Frame);
        assert!(frame.iter().any(|text| text == "Save trace"));
        assert!(frame.iter().any(|text| text == "Series"));

        let gpu = run_body(DebugTab::Gpu);
        assert!(gpu.iter().any(|text| text == "Tessellation warnings"));
        assert!(gpu.iter().any(|text| text == "Frame"));

        let camera = run_body(DebugTab::Camera);
        assert!(camera.iter().any(|text| text == "Yaw"));
        assert_eq!(
            camera
                .iter()
                .filter(|text| text.as_str() == "View row")
                .count(),
            4
        );
    }
}
