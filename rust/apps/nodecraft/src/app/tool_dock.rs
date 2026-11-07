//! Tool dock definitions, layout and interaction.

use super::*;
use nodecraft_dock::{DockItem, DockMetrics, DockStyle, paint_background};

const TOOL_METRICS: DockMetrics = DockMetrics::DEFAULT;

pub(crate) fn tool_dock_style(theme: &crate::theme::Theme) -> DockStyle {
    DockStyle {
        background: theme.color(theme.dock),
        border: theme.color(theme.dock_border),
        accent: theme.color(theme.accent),
        text: theme.color(theme.text),
        icon: theme.color(theme.icon),
        icon_hover: theme.color(theme.icon_hover),
        active_icon: egui::Color32::WHITE,
        shadow: egui::Color32::from_black_alpha(120),
    }
}
const TOOL_DOCK_GAP: f32 = 8.0;
const TOOL_SEARCH_CLEAR_AFTER: std::time::Duration = std::time::Duration::from_secs(5 * 60);

fn parse_shortcut(text: &str) -> Option<egui::KeyboardShortcut> {
    let mut modifiers = egui::Modifiers::NONE;
    let mut key = None;
    for part in text.split('+').map(str::trim).filter(|part| !part.is_empty()) {
        match part.to_lowercase().as_str() {
            "cmd" | "command" | "meta" | "super" => modifiers |= egui::Modifiers::COMMAND,
            "shift" => modifiers |= egui::Modifiers::SHIFT,
            "alt" | "option" => modifiers |= egui::Modifiers::ALT,
            "ctrl" | "control" => modifiers |= egui::Modifiers::CTRL,
            _ => key = egui::Key::from_name(part),
        }
    }
    key.map(|key| egui::KeyboardShortcut::new(modifiers, key))
}

fn shortcut_pressed(ctx: &egui::Context, text: &str) -> bool {
    parse_shortcut(text)
        .is_some_and(|shortcut| ctx.input_mut(|input| input.consume_shortcut(&shortcut)))
}

pub(super) fn tool_dock_height() -> f32 {
    // One place for the dock's vertical metrics: search, separator, every
    // category row, the grip, and the gaps between them.
    let mut height = TOOL_METRICS.padding;
    height += TOOL_METRICS.row_size(); // search
    height += TOOL_METRICS.gap;
    height += TOOL_METRICS.search_separator;
    height += TOOL_METRICS.gap;
    for index in 0..ToolSection::ALL.len() {
        height += TOOL_METRICS.row_size();
        if index + 1 < ToolSection::ALL.len() {
            height += TOOL_METRICS.gap;
        }
    }
    height += TOOL_METRICS.gap;
    height += TOOL_METRICS.grip_height;
    height += TOOL_METRICS.padding;
    height
}

pub(super) fn tool_dock_rect(panel: egui::Rect, offset_y: f32, height: f32) -> egui::Rect {
    let offset = TOOL_METRICS.edge_offset;
    let top = (panel.center().y - height * 0.5 + offset_y)
        .clamp(panel.top() + offset, (panel.bottom() - height - offset).max(panel.top() + offset));
    egui::Rect::from_min_size(
        egui::Pos2::new(panel.min.x + offset, top),
        egui::Vec2::new(TOOL_METRICS.width, height),
    )
}

#[derive(Clone, Copy)]
pub(super) struct ToolItem {
    pub(super) label: &'static str,
    pub(super) node_type: &'static str,
    pub(super) primitive: Option<&'static str>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolDockFocus {
    Search,
    Root { index: usize },
    Menu { section: ToolSection },
}

#[derive(Clone, Copy)]
pub(super) struct ToolGroup {
    heading: Option<&'static str>,
    tools: &'static [ToolItem],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolSection {
    Create,
    Modify,
    Boolean,
    Pattern,
    Organise,
    Animate,
    Io,
}

impl ToolSection {
    const ALL: [Self; 7] = [
        Self::Create,
        Self::Modify,
        Self::Boolean,
        Self::Pattern,
        Self::Organise,
        Self::Animate,
        Self::Io,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Create => "Create",
            Self::Modify => "Modify",
            Self::Boolean => "Boolean",
            Self::Pattern => "Pattern",
            Self::Organise => "Organise",
            Self::Animate => "Animate",
            Self::Io => "Input / Output",
        }
    }

    fn icon(self) -> (&'static str, &'static [u8]) {
        match self {
            Self::Create => ("iconify-create", include_bytes!("../../../../../vendor/iconify-design/create.svg")),
            Self::Modify => ("iconify-modify", include_bytes!("../../../../../vendor/iconify-design/modify.svg")),
            Self::Boolean => ("iconify-boolean", include_bytes!("../../../../../vendor/iconify-design/boolean.svg")),
            Self::Pattern => ("iconify-pattern", include_bytes!("../../../../../vendor/iconify-design/pattern.svg")),
            Self::Organise => ("iconify-organise", include_bytes!("../../../../../vendor/iconify-design/organise.svg")),
            Self::Animate => ("iconify-animate", include_bytes!("../../../../../vendor/iconify-design/animate.svg")),
            Self::Io => ("iconify-io", include_bytes!("../../../../../vendor/iconify-design/io.svg")),
        }
    }

    fn groups(self) -> &'static [ToolGroup] {
        const SKETCH: &[ToolItem] = &[
            ToolItem { label: "Sketch", node_type: "sketch", primitive: None },
            ToolItem { label: "Spline", node_type: "spline", primitive: None },
        ];
        const SOLIDS: &[ToolItem] = &[
            ToolItem { label: "Box", node_type: "solid", primitive: Some("box") },
            ToolItem { label: "Sphere", node_type: "solid", primitive: Some("sphere") },
            ToolItem { label: "Cylinder", node_type: "solid", primitive: Some("cylinder") },
            ToolItem { label: "Cone", node_type: "solid", primitive: Some("cone") },
            ToolItem { label: "Torus", node_type: "solid", primitive: Some("torus") },
            ToolItem { label: "Pyramid", node_type: "solid", primitive: Some("pyramid") },
            ToolItem { label: "Wedge", node_type: "solid", primitive: Some("wedge") },
            ToolItem { label: "Partial Sphere", node_type: "solid", primitive: Some("partialSphere") },
            ToolItem { label: "Partial Torus", node_type: "solid", primitive: Some("partialTorus") },
        ];
        const FEATURES: &[ToolItem] = &[
            ToolItem { label: "Extrude", node_type: "extrude", primitive: None },
            ToolItem { label: "Loft", node_type: "loft", primitive: None },
            ToolItem { label: "Surface Loft", node_type: "surfaceloft", primitive: None },
        ];
        const MODIFY: &[ToolItem] = &[
            ToolItem { label: "Transform", node_type: "translate", primitive: None },
            ToolItem { label: "Mirror", node_type: "mirror", primitive: None },
            ToolItem { label: "Chamfer", node_type: "chamfer", primitive: None },
            ToolItem { label: "Fillet", node_type: "fillet", primitive: None },
            ToolItem { label: "Shell", node_type: "shell", primitive: None },
            ToolItem { label: "Offset Faces", node_type: "offsetfaces", primitive: None },
        ];
        const BOOLEAN: &[ToolItem] = &[ToolItem { label: "Boolean", node_type: "boolean", primitive: None }];
        const PATTERN: &[ToolItem] = &[ToolItem { label: "Scatter", node_type: "scatter", primitive: None }];
        const ORGANISE: &[ToolItem] = &[
            ToolItem { label: "Combine to Part", node_type: "combine", primitive: None },
            ToolItem { label: "Group", node_type: "group", primitive: None },
            ToolItem { label: "Look", node_type: "look", primitive: None },
        ];
        const IO: &[ToolItem] = &[
            ToolItem { label: "Import STEP", node_type: "stepimporter", primitive: None },
            ToolItem { label: "Linked Part", node_type: "linkedpart", primitive: None },
            ToolItem { label: "Export STEP", node_type: "exporter", primitive: None },
        ];
        const CREATE: &[ToolGroup] = &[
            ToolGroup { heading: Some("Sketch"), tools: SKETCH },
            ToolGroup { heading: Some("Solid"), tools: SOLIDS },
            ToolGroup { heading: Some("Features"), tools: FEATURES },
        ];
        const MODIFY_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: MODIFY }];
        const BOOLEAN_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: BOOLEAN }];
        const PATTERN_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: PATTERN }];
        const ORGANISE_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: ORGANISE }];
        const ANIMATE_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: &[] }];
        const IO_GROUPS: &[ToolGroup] = &[ToolGroup { heading: None, tools: IO }];
        match self {
            Self::Create => CREATE,
            Self::Modify => MODIFY_GROUPS,
            Self::Boolean => BOOLEAN_GROUPS,
            Self::Pattern => PATTERN_GROUPS,
            Self::Organise => ORGANISE_GROUPS,
            Self::Animate => ANIMATE_GROUPS,
            Self::Io => IO_GROUPS,
        }
    }
}




impl NodeCraftApp {
    pub(super) fn active_tool_menu(&self) -> Option<ToolSection> {
        match self.tool_dock_focus {
            Some(ToolDockFocus::Menu { section }) => Some(section),
            _ => None,
        }
    }

    pub(super) fn tool_section_index(section: ToolSection) -> usize {
        ToolSection::ALL
            .iter()
            .position(|candidate| *candidate == section)
            .unwrap_or(0)
    }

    pub(super) fn open_tool_menu(&mut self, section: ToolSection, select_first: bool) {
        self.tool_dock_focus = Some(ToolDockFocus::Menu { section });
        self.tool_menu_index = select_first.then_some(0);
    }

    pub(super) fn close_tool_dock(&mut self) {
        self.tool_dock_focus = None;
        self.tool_menu_index = None;
        self.tool_search_result_index = None;
        self.tool_dock_keyboard_active = false;
    }

    pub(super) fn focus_tool_search(&mut self) {
        self.tool_dock_focus = Some(ToolDockFocus::Search);
        self.tool_menu_index = None;
        self.tool_dock_keyboard_active = true;
    }

    pub(super) fn focus_tool_root(&mut self, index: usize) {
        self.tool_dock_focus = Some(ToolDockFocus::Root { index });
        self.tool_menu_index = None;
        self.tool_dock_keyboard_active = true;
    }

    pub(super) fn show_tool_dock(
        &mut self,
        ctx: &egui::Context,
        workspace: egui::Rect,
        connect_source: Option<String>,
    ) {
        let theme = crate::theme::get();
        let panel = self.node_editor.panel_rect(workspace);
        let measured_height = if self.tool_dock_measured_height > 0.0 {
            self.tool_dock_measured_height
        } else {
            tool_dock_height()
        };
        let mut dock = tool_dock_rect(panel, self.tool_dock_offset_y, measured_height);

        let mut clicked_section = None;
        let mut hovered_section = None;
        // Only let pointer hover change the open category when the pointer has
        // actually moved. Otherwise a stationary pointer over the old icon
        // immediately undoes an arrow-key category change.
        let pointer_moved = ctx.input(|input| input.pointer.delta().length_sq() > 0.0);
        let mut section_rects = Vec::new();
        let mut first_section_rect = None;
        let dock_content = egui::Area::new(egui::Id::new("node-tool-dock"))
            .order(egui::Order::Tooltip)
            .fixed_pos(dock.min + TOOL_METRICS.content_offset())
            .show(ctx, |ui| {
                ui.set_width(TOOL_METRICS.width);
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                let search_active = self.tool_search_open
                    || matches!(self.tool_dock_focus, Some(ToolDockFocus::Search));
                let search_response = nodecraft_dock::icon_button(
                    ui,
                    TOOL_METRICS,
                    tool_dock_style(&theme),
                    DockItem::new(
                        "Search tools",
                        "search",
                        include_bytes!("../../../../../vendor/iconify-design/search.svg"),
                    )
                    .active(search_active),
                );
                let search_rect = search_response.rect;
                if search_response.clicked_by(egui::PointerButton::Primary) {
                    self.tool_search_open = !self.tool_search_open;
                    self.tool_search_result_index = None;
                    self.close_tool_dock();
                }
                ui.add_space(TOOL_METRICS.gap);
                ui.allocate_exact_size(
                    egui::Vec2::new(TOOL_METRICS.width, TOOL_METRICS.search_separator),
                    egui::Sense::hover(),
                );
                ui.add_space(TOOL_METRICS.gap);
                for section in ToolSection::ALL {
                    let active = match self.tool_dock_focus {
                        Some(ToolDockFocus::Menu { section: active }) => active == section,
                        Some(ToolDockFocus::Root { index }) => {
                            ToolSection::ALL.get(index) == Some(&section)
                        }
                        Some(ToolDockFocus::Search) | None => false,
                    };
                    let (name, source) = section.icon();
                    let response = nodecraft_dock::icon_button(
                        ui,
                        TOOL_METRICS,
                        tool_dock_style(&theme),
                        DockItem::new(section.label(), name, source).active(active),
                    );
                    let rect = response.rect;
                    if response.clicked_by(egui::PointerButton::Primary) {
                        clicked_section = Some(section);
                    } else if pointer_moved
                        && response.hovered()
                        && self.active_tool_menu().is_some()
                    {
                        hovered_section = Some(section);
                    }
                    if first_section_rect.is_none() {
                        first_section_rect = Some(rect);
                    }
                    section_rects.push((section, rect));
                    if section != ToolSection::ALL[ToolSection::ALL.len() - 1] {
                        ui.add_space(TOOL_METRICS.gap);
                    }
                }
                ui.add_space(TOOL_METRICS.gap);
                let (grip_rect, grip_response) = ui.allocate_exact_size(
                    egui::Vec2::new(TOOL_METRICS.width, TOOL_METRICS.grip_height),
                    egui::Sense::click_and_drag(),
                );
                if grip_response.double_clicked() {
                    self.tool_dock_offset_y = 0.0;
                    self.view_save_due =
                        Some(Instant::now() + std::time::Duration::from_millis(500));
                    ui.ctx().request_repaint();
                }
                if grip_response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
                if grip_response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                    let max_offset =
                        ((panel.height() - self.tool_dock_measured_height) * 0.5 - 12.0)
                            .max(0.0);
                    self.tool_dock_offset_y = (self.tool_dock_offset_y
                        + grip_response.drag_delta().y)
                        .clamp(-max_offset, max_offset);
                    self.view_save_due =
                        Some(Instant::now() + std::time::Duration::from_millis(500));
                    ui.ctx().request_repaint();
                }
                let dot = theme.with_alpha(theme.border, 220);
                for row in 0..2 {
                    for column in 0..3 {
                        let centre = grip_rect.center()
                            + egui::vec2(
                                (column as f32 - 1.0) * 5.0,
                                (row as f32 - 0.5) * 5.0,
                            );
                        ui.painter().circle_filled(centre, 1.4, dot);
                    }
                }
                if let Some(first) = first_section_rect {
                    let centre = egui::Pos2::new(
                        search_rect.center().x,
                        (search_rect.center().y + first.center().y) * 0.5,
                    );
                    ui.painter().line_segment(
                        [
                            egui::Pos2::new(centre.x - 12.0, centre.y),
                            egui::Pos2::new(centre.x + 12.0, centre.y),
                        ],
                        egui::Stroke::new(1.0, theme.with_alpha(theme.border, 160)),
                    );
                }
            });

        // The background is derived from the rows actually rendered, so its
        // height cannot drift from the icon stack.
        let content = dock_content.response.rect;
        dock = nodecraft_dock::background_rect(content, TOOL_METRICS);
        let background = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("node-tool-dock-background"),
        ));
        paint_background(
            &background,
            dock,
            TOOL_METRICS,
            tool_dock_style(&theme),
        );
        self.tool_dock_measured_height = dock.height();

        let category_clicked = clicked_section.is_some();
        if let Some(section) = clicked_section {
            self.tool_search_open = false;
            if self.active_tool_menu() == Some(section) {
                self.close_tool_dock();
            } else {
                self.open_tool_menu(section, false);
                self.tool_dock_keyboard_active = true;
            }
            // Opening the dock must not disturb the canvas selection. If the
            // click-away path cleared it this frame, restore the remembered
            // source the dock was opened for.
            if self.active_tool_menu().is_some()
                && self.selected_nodes.is_empty()
                && let Some(source) = self.tool_add_source.clone()
            {
                self.selected_nodes.insert(source);
            }
        } else if let Some(section) = hovered_section
            && self.active_tool_menu() != Some(section)
        {
            self.open_tool_menu(section, false);
        }

        // A root keyboard focus is not a menu, so the menu click-away path
        // below never runs while it is active. Any click that is not a
        // category activation should clear it.
        if self.active_tool_menu().is_none()
            && !category_clicked
            && ctx.input(|input| input.pointer.any_click())
        {
            self.close_tool_dock();
        }

        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.close_tool_dock();
            self.tool_search_open = false;
        }

        if shortcut_pressed(ctx, &self.tool_dock_shortcut) {
            if self.tool_dock_keyboard_active {
                self.close_tool_dock();
                self.tool_search_open = false;
            } else {
                self.focus_tool_search();
                self.tool_search_open = false;
            }
            self.tool_menu_rect = None;
        }
        if shortcut_pressed(ctx, &self.tool_search_shortcut) {
            self.tool_search_query.clear();
            self.tool_search_result_index = None;
            self.tool_search_clear_due = None;
            self.tool_search_open = true;
            self.focus_tool_search();
            self.tool_menu_rect = None;
        }

        if self.tool_search_open {
            let mut chosen = None;
            let search_centre_y =
                dock.min.y + TOOL_METRICS.padding + TOOL_METRICS.row_size() * 0.5;
            let search_popup_y = (search_centre_y - TOOL_METRICS.row_size() * 0.5 - 8.0)
                .clamp(panel.top() + 8.0, panel.bottom() - 48.0);
            let search_response = egui::Area::new(egui::Id::new("node-tool-search"))
                .order(egui::Order::Foreground)
                .fixed_pos(egui::Pos2::new(dock.max.x + TOOL_DOCK_GAP, search_popup_y))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(theme.color(theme.popup))
                        .stroke(egui::Stroke::new(1.0, theme.color(theme.border_strong)))
                        .corner_radius(egui::CornerRadius::same(12))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.set_width(260.0);
                            let query_frame = egui::Frame::new()
                                .fill(theme.color(theme.control))
                                .stroke(egui::Stroke::new(
                                    1.0,
                                    theme.color(theme.border_strong),
                                ))
                                .corner_radius(egui::CornerRadius::same(8))
                                .inner_margin(egui::Margin::symmetric(8, 6))
                                .show(ui, |ui| {
                                    let field_height =
                                        ui.text_style_height(&egui::TextStyle::Body) + 6.0;
                                    ui.add_sized(
                                        egui::Vec2::new(244.0, field_height),
                                        egui::TextEdit::singleline(&mut self.tool_search_query)
                                            .id(egui::Id::new("node-tool-search-field"))
                                            .frame(egui::Frame::NONE)
                                            .vertical_align(egui::Align::Center)
                                            .text_color(theme.color(theme.text))
                                            .hint_text(
                                                egui::RichText::new("Search tools")
                                                    .color(theme.with_alpha(theme.muted, 190)),
                                            ),
                                    )
                                });
                            let query = query_frame.inner;
                            // The field and the result list are two modes of the
                            // same search popup; None means text-entry mode.
                            if query.clicked() || query.gained_focus() {
                                self.tool_search_result_index = None;
                            }
                            let field_active = self.tool_search_result_index.is_none();
                            if field_active {
                                if !query.has_focus() {
                                    query.request_focus();
                                }
                                ui.painter().rect_stroke(
                                    query_frame.response.rect,
                                    egui::CornerRadius::same(8),
                                    egui::Stroke::new(1.5, theme.color(theme.accent)),
                                    egui::StrokeKind::Inside,
                                );
                            }
                            let needle = self.tool_search_query.trim().to_lowercase();
                            if needle.is_empty() {
                                return;
                            }
                            ui.add_space(4.0);
                            let mut found: Vec<(usize, ToolSection, ToolItem)> = Vec::new();
                            let pattern = format!("*{needle}*");
                            let threshold = (needle.chars().count() / 3).max(1);
                            for section in ToolSection::ALL {
                                for group in section.groups() {
                                    for tool in group.tools {
                                        let wildcard = nodecraft_search::wildcard_matches(&pattern, tool.label)
                                            || nodecraft_search::wildcard_matches(&pattern, tool.node_type);
                                        let distance = nodecraft_search::best_levenshtein(&needle, tool.label)
                                            .min(nodecraft_search::best_levenshtein(&needle, tool.node_type));
                                        if wildcard || distance <= threshold {
                                            found.push((if wildcard { 0 } else { distance }, section, *tool));
                                        }
                                    }
                                }
                            }
                            found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.label.cmp(b.2.label)));
                            if found.is_empty() {
                                ui.label(
                                    egui::RichText::new("No matching tools")
                                        .size(11.0)
                                        .color(theme.color(theme.muted)),
                                );
                                self.tool_search_result_index = None;
                            }
                            if query.changed() {
                                self.tool_search_result_index = None;
                                self.tool_search_clear_due =
                                    Some(Instant::now() + TOOL_SEARCH_CLEAR_AFTER);
                            }
                            let result_count = found.len().min(10);
                            // These keys must be consumed here. Otherwise an
                            // unhandled Up/Down can fall through to the native
                            // menu bar while the result list has focus.
                            let enter_pressed = ctx.input_mut(|input| {
                                input.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                            });
                            let results_focused = self.tool_search_result_index.is_some();
                            let (up_pressed, down_pressed) = if results_focused {
                                ctx.input_mut(|input| {
                                    (
                                        input.consume_key(
                                            egui::Modifiers::NONE,
                                            egui::Key::ArrowUp,
                                        ),
                                        input.consume_key(
                                            egui::Modifiers::NONE,
                                            egui::Key::ArrowDown,
                                        ),
                                    )
                                })
                            } else {
                                (false, false)
                            };
                            if result_count > 0 {
                                if enter_pressed {
                                    if let Some(index) = self.tool_search_result_index {
                                        chosen = found.get(index).map(|(_, _, tool)| *tool);
                                    } else {
                                        self.tool_search_result_index = Some(0);
                                        query.surrender_focus();
                                    }
                                }
                                if let Some(current) = self.tool_search_result_index {
                                    let mut index = current.min(result_count - 1);
                                    if down_pressed {
                                        index = (index + 1).min(result_count - 1);
                                    }
                                    let return_to_field = up_pressed && index == 0;
                                    if up_pressed && index > 0 {
                                        index -= 1;
                                    }
                                    self.tool_search_result_index =
                                        (!return_to_field).then_some(index);
                                    if return_to_field {
                                        query.request_focus();
                                    }
                                }
                            }
                            for (index, (_, section, tool)) in
                                found.iter().take(10).enumerate()
                            {
                                let selected = self.tool_search_result_index == Some(index);
                                let (rect, response) = ui.allocate_exact_size(
                                    egui::Vec2::new(260.0, 26.0),
                                    egui::Sense::click(),
                                );
                                if response.hovered() || selected {
                                    let fill = if selected {
                                        theme.with_alpha(theme.accent, 170)
                                    } else {
                                        theme.with_alpha(theme.control, 190)
                                    };
                                    ui_shape::fill(&ui.painter(), rect, fill);
                                    if response.hovered() {
                                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                    }
                                }
                                ui.painter().text(
                                    egui::Pos2::new(rect.min.x + 8.0, rect.center().y),
                                    egui::Align2::LEFT_CENTER,
                                    tool.label,
                                    egui::FontId::proportional(13.0),
                                    theme.color(theme.text),
                                );
                                ui.painter().text(
                                    egui::Pos2::new(rect.max.x - 8.0, rect.center().y),
                                    egui::Align2::RIGHT_CENTER,
                                    section.label(),
                                    egui::FontId::proportional(10.0),
                                    theme.color(theme.muted),
                                );
                                if response.clicked() {
                                    chosen = Some(*tool);
                                }
                            }
                        });
                });
            self.tool_menu_rect = Some(search_response.response.rect);
            // The click that opens the popup is outside it, but inside the dock;
            // do not treat that same click as click-away.
            let pointer_on_dock = ctx
                .pointer_latest_pos()
                .is_some_and(|pointer| dock.contains(pointer));
            if search_response.response.clicked_elsewhere() && !pointer_on_dock {
                self.tool_search_open = false;
                self.tool_search_clear_due = None;
                self.close_tool_dock();
                self.tool_menu_rect = None;
            }
            if let Some(tool) = chosen {
                self.tool_search_open = false;
                self.tool_search_query.clear();
                self.tool_search_clear_due = None;
                self.close_tool_dock();
                self.add_tool_node(tool, workspace, connect_source);
            }
            return;
        }

        if self.active_tool_menu().is_none() {
            if !self.tool_dock_keyboard_active {
                self.tool_menu_rect = None;
                return;
            }
            let keyboard_row_count = ToolSection::ALL.len() + 1;
            let focused = match self.tool_dock_focus {
                Some(ToolDockFocus::Search) => Some(0),
                Some(ToolDockFocus::Root { index }) => Some(index + 1),
                _ => None,
            };
            if ctx.input(|input| input.key_pressed(egui::Key::ArrowDown)) {
                let index = focused
                    .map(|index| (index + 1).min(keyboard_row_count - 1))
                    .unwrap_or(0);
                if index == 0 {
                    self.focus_tool_search();
                } else {
                    self.focus_tool_root(index - 1);
                }
            }
            if ctx.input(|input| input.key_pressed(egui::Key::ArrowUp)) {
                let index = focused
                    .map(|index| index.saturating_sub(1))
                    .unwrap_or(keyboard_row_count - 1);
                if index == 0 {
                    self.focus_tool_search();
                } else {
                    self.focus_tool_root(index - 1);
                }
            }
            if ctx.input(|input| input.key_pressed(egui::Key::ArrowRight))
                || ctx.input(|input| input.key_pressed(egui::Key::Enter))
            {
                match self.tool_dock_focus {
                    Some(ToolDockFocus::Search) => {
                        self.tool_search_open = true;
                        self.tool_menu_rect = None;
                        return;
                    }
                    Some(ToolDockFocus::Root { index }) => {
                        self.open_tool_menu(ToolSection::ALL[index], true);
                    }
                    _ => self.focus_tool_search(),
                }
            }
            if self.active_tool_menu().is_none() {
                self.tool_menu_rect = None;
                return;
            }
        }
        let Some(section) = self.active_tool_menu() else {
            self.tool_menu_rect = None;
            return;
        };
        if self.tool_dock_keyboard_active
            && ctx.input(|input| input.key_pressed(egui::Key::ArrowLeft))
        {
            self.focus_tool_root(Self::tool_section_index(section));
            self.tool_menu_rect = None;
            return;
        }
        let groups = section.groups();
        let items: Vec<&ToolItem> = groups
            .iter()
            .flat_map(|group| group.tools.iter())
            .collect();
        if !items.is_empty() {
            if let Some(index) = self.tool_menu_index.as_mut() {
                *index = (*index).min(items.len() - 1);
            }
            if self.tool_dock_keyboard_active
                && ctx.input(|input| input.key_pressed(egui::Key::ArrowDown))
            {
                self.tool_menu_index = Some(match self.tool_menu_index {
                    Some(index) => (index + 1).min(items.len() - 1),
                    None => 0,
                });
            }
            if self.tool_dock_keyboard_active
                && ctx.input(|input| input.key_pressed(egui::Key::ArrowUp))
            {
                self.tool_menu_index = Some(match self.tool_menu_index {
                    Some(index) => index.saturating_sub(1),
                    None => items.len() - 1,
                });
            }
        }
        let heading_count = groups.iter().filter(|group| group.heading.is_some()).count();
        let divider_count = groups.len().saturating_sub(1);
        let menu_height = 40.0
            + items.len() as f32 * 30.0
            + heading_count as f32 * 20.0
            + divider_count as f32 * 9.0
            + if items.is_empty() { 24.0 } else { 0.0 };
        let label_font = egui::FontId::proportional(13.0);
        let label_width = |text: &str| {
            ctx.fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap(text.to_owned(), label_font.clone(), theme.color(theme.text))
                    .size()
                    .x
            })
        };
        let longest = items
            .iter()
            .map(|tool| label_width(tool.label))
            .fold(0.0_f32, f32::max);
        // Leave room for the row's text padding but keep long labels readable
        // without letting the flyout grow across the canvas.
        let row_width = (longest + 20.0).clamp(140.0, 264.0);
        let anchor_y = section_rects
            .iter()
            .find(|(candidate, _)| *candidate == section)
            .map(|(_, rect)| rect.center().y)
            .unwrap_or(dock.min.y + 8.0);
        let y = (anchor_y - menu_height * 0.5).clamp(
            workspace.top() + 12.0,
            (workspace.bottom() - 12.0 - menu_height).max(workspace.top() + 12.0),
        );
        let menu_pos = egui::Pos2::new(dock.max.x + TOOL_DOCK_GAP, y);
        let keyboard_choice = if self.tool_dock_keyboard_active
            && ctx.input(|input| input.key_pressed(egui::Key::Enter))
        {
            self.tool_menu_index
                .and_then(|index| items.get(index))
                .map(|tool| **tool)
        } else {
            None
        };
        let mut chosen = keyboard_choice;
        let menu_response = egui::Area::new(egui::Id::new("node-tool-dock-menu"))
            .order(egui::Order::Foreground)
            .fixed_pos(menu_pos)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(theme.color(theme.popup))
                    .stroke(egui::Stroke::new(1.0, theme.color(theme.border_strong)))
                    .corner_radius(egui::CornerRadius::same(12))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_width(row_width);
                        ui.label(
                            egui::RichText::new(section.label().to_uppercase())
                                .size(10.0)
                                .color(theme.color(theme.muted)),
                        );
                        ui.add_space(3.0);
                        if items.is_empty() {
                            ui.label(
                                egui::RichText::new("No animation nodes yet")
                                    .size(11.0)
                                    .color(theme.color(theme.muted)),
                            );
                        }
                        let mut item_index = 0;
                        for (group_index, group) in groups.iter().enumerate() {
                            if group_index > 0 {
                                ui.add_space(4.0);
                                let (line, _) = ui.allocate_exact_size(
                                    egui::Vec2::new(row_width, 1.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().line_segment(
                                    [line.left_center(), line.right_center()],
                                    egui::Stroke::new(1.0, theme.with_alpha(theme.border, 140)),
                                );
                                ui.add_space(4.0);
                            }
                            if let Some(heading) = group.heading {
                                ui.add_space(2.0);
                                ui.label(
                                    egui::RichText::new(heading.to_uppercase())
                                        .size(9.0)
                                        .strong()
                                        .color(theme.with_alpha(theme.muted, 220)),
                                );
                                ui.add_space(2.0);
                            }
                            for tool in group.tools {
                            let selected = self.tool_menu_index == Some(item_index);
                            let (rect, response) = ui.allocate_exact_size(
                                egui::Vec2::new(row_width, 28.0),
                                egui::Sense::click(),
                            );
                            if response.hovered() || selected {
                                let fill = if selected {
                                    theme.with_alpha(theme.accent, 170)
                                } else {
                                    theme.with_alpha(theme.control, 190)
                                };
                                ui_shape::fill(&ui.painter(), rect, fill);
                                if response.hovered() {
                                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                                }
                            }
                            let text_width = label_width(tool.label);
                            let label = if text_width <= row_width - 20.0 {
                                tool.label.to_owned()
                            } else {
                                let ellipsis = "…";
                                let mut cut = tool.label;
                                while !cut.is_empty()
                                    && label_width(&format!("{cut}{ellipsis}")) > row_width - 20.0
                                {
                                    cut = &cut[..cut.len().saturating_sub(1)];
                                }
                                format!("{cut}{ellipsis}")
                            };
                            ui.painter().text(
                                egui::Pos2::new(rect.min.x + 10.0, rect.center().y),
                                egui::Align2::LEFT_CENTER,
                                label,
                                label_font.clone(),
                                theme.color(theme.text),
                            );
                            let response = response.on_hover_text(match tool.primitive {
                                Some(primitive) => format!("Add {} ({primitive})", tool.label),
                                None => format!("Add {}", tool.label),
                            });
                                if response.clicked() {
                                    chosen = Some(*tool);
                                }
                                item_index += 1;
                            }
                        }
                    });
            });
        self.tool_menu_rect = Some(menu_response.response.rect);
        let clicked_outside_menu = !category_clicked
            && ctx.input(|input| input.pointer.any_click())
            && ctx
                .pointer_latest_pos()
                .is_none_or(|pointer| !menu_response.response.rect.contains(pointer));
        if clicked_outside_menu {
            self.close_tool_dock();
            self.tool_add_source = None;
            self.tool_menu_rect = None;
        }
        if let Some(tool) = chosen {
            self.add_tool_node(tool, workspace, connect_source);
            self.close_tool_dock();
        }
    }

}
