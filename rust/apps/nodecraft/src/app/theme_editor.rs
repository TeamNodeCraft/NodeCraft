//! Theme editing widgets used by the settings UI.

use super::*;

pub(super) fn layout_roi(
    document: &GraphDocument,
    positions: &HashMap<String, (f64, f64)>,
) -> Option<egui::Rect> {
    let mut min = egui::Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = egui::Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for node in &document.nodes {
        let (x, y) = positions
            .get(&node.id)
            .copied()
            .unwrap_or((node.x, node.y));
        let x = x as f32;
        let y = y as f32;
        min.x = min.x.min(x - NODE_LEFT_PADDING);
        min.y = min.y.min(y);
        max.x = max.x.max(x + node_width(node) + NODE_RIGHT_PADDING);
        max.y = max.y.max(y + NODE_HEIGHT);
    }
    (min.x.is_finite() && max.x.is_finite()).then(|| egui::Rect::from_min_max(min, max))
}

pub(super) fn theme_color_dot(ui: &mut egui::Ui, rgba: [u8; 4]) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::click());
    if response.hovered() || response.clicked() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let theme = crate::theme::get();
    let center = rect.center();
    let radius = 8.5;
    ui.painter().circle_filled(center, radius, theme.color(theme.panel));
    ui.painter().circle_filled(center, radius, theme.color(rgba));
    ui.painter().circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, theme.color(theme.border_strong)),
    );
    if response.hovered() {
        ui.painter().circle_stroke(
            center,
            radius + 1.5,
            egui::Stroke::new(1.0, theme.color(theme.accent)),
        );
    }
    response
}

pub(super) fn theme_revert_button(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(26.0, 26.0),
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );
    let theme = crate::theme::get();
    let hot = enabled && response.hovered();
    if hot {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let tint = if !enabled {
        theme.icon
    } else if hot {
        theme.icon_hover
    } else {
        theme.icon
    };
    let svg = std::str::from_utf8(include_bytes!(
        "../../../../../vendor/fluent/arrow_clockwise.svg"
    ))
    .unwrap_or_default();
    let svg = tint_svg(svg.as_bytes(), &crate::theme::Theme::hex(tint));
    let image = egui::Image::from_bytes(
        "bytes://theme-revert-defaults.svg",
        svg.into_bytes(),
    )
    .fit_to_exact_size(egui::Vec2::splat(16.0));
    ui.put(rect.shrink(5.0), image);
    response
}

pub(super) fn theme_edit_button(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(26.0, 26.0),
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );
    let theme = crate::theme::get();
    let hot = enabled && response.hovered();
    if hot {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let tint = if !enabled {
        theme.icon
    } else if hot {
        theme.icon_hover
    } else {
        theme.icon
    };
    let svg = std::str::from_utf8(include_bytes!("../../../../../vendor/fluent/edit.svg"))
        .unwrap_or_default();
    let svg = tint_svg(svg.as_bytes(), &crate::theme::Theme::hex(tint));
    let image = egui::Image::from_bytes(
        "bytes://theme-edit-name.svg",
        svg.into_bytes(),
    )
    .fit_to_exact_size(egui::Vec2::splat(15.0));
    ui.put(rect.shrink(5.0), image);
    if !enabled {
        // Built-in themes cannot be renamed: show a crossed-out pencil rather
        // than merely dimming it.
        let slash = egui::Stroke::new(1.4, theme.color(theme.muted));
        ui.painter().line_segment(
            [
                rect.left_top() + egui::vec2(6.0, 6.0),
                rect.right_bottom() - egui::vec2(6.0, 6.0),
            ],
            slash,
        );
    }
    response
}

pub(super) fn theme_add_button(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::click());
    let theme = crate::theme::get();
    if response.hovered() {
        ui.painter().rect_filled(
            rect,
            egui::CornerRadius::same(6),
            theme.with_alpha(theme.highlight, 30),
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let color = if response.hovered() {
        theme.color(theme.icon_hover)
    } else {
        theme.color(theme.icon)
    };
    let center = rect.center();
    let arm = 6.0;
    let stroke = egui::Stroke::new(1.6, color);
    ui.painter().line_segment(
        [center + egui::vec2(-arm, 0.0), center + egui::vec2(arm, 0.0)],
        stroke,
    );
    ui.painter().line_segment(
        [center + egui::vec2(0.0, -arm), center + egui::vec2(0.0, arm)],
        stroke,
    );
    response
}

#[derive(Clone)]
struct ThemeHexDraft {
    text: String,
    rgba: [u8; 4],
}

/// Editable hex field for a palette colour. Accepts `#RGB`, `#RGBA`,
/// `#RRGGBB`, and `#RRGGBBAA`; six-digit input keeps the current alpha.
pub(super) fn theme_hex_edit(ui: &mut egui::Ui, key: &str, rgba: [u8; 4]) -> Option<[u8; 4]> {
    let data_id = ui.make_persistent_id(("theme-hex-color", key));
    let field_id = egui::Id::new(("theme-hex-field", key));
    let mut draft = ui
        .data(|data| data.get_temp::<ThemeHexDraft>(data_id))
        .unwrap_or_else(|| ThemeHexDraft {
            text: crate::theme::Theme::hex_rgba(rgba),
            rgba,
        });

    let response = ui
        .add_sized(
            [ui.available_width(), 18.0],
            egui::TextEdit::singleline(&mut draft.text)
                .id(field_id)
                .font(egui::TextStyle::Monospace)
                .horizontal_align(egui::Align::RIGHT)
                .hint_text("#rrggbbaa"),
        )
        .on_hover_text(
            "Hex colour: #RGB, #RGBA, #RRGGBB, or #RRGGBBAA\n             Six-digit input preserves the current alpha.",
        );

    let mut changed = None;
    if response.changed() {
        if let Some(next) = crate::theme::Theme::parse_hex(&draft.text, draft.rgba[3]) {
            draft.rgba = next;
            changed = Some(next);
        }
    }
    if response.has_focus() {
        // Keep partial/invalid input while the user is typing.
    } else if draft.rgba != rgba {
        draft.rgba = rgba;
        draft.text = crate::theme::Theme::hex_rgba(rgba);
    } else if response.lost_focus() {
        draft.text = crate::theme::Theme::hex_rgba(draft.rgba);
    }

    ui.data_mut(|data| data.insert_temp(data_id, draft));
    changed
}

pub(super) fn theme_palette_row(
    ui: &mut egui::Ui,
    entry: &crate::theme::ThemeEntry,
    theme: &mut crate::theme::Theme,
    row_width: f32,
) {
    const DOT_WIDTH: f32 = 26.0;
    const VALUE_WIDTH: f32 = 112.0;
    let spacing = ui.spacing().item_spacing.x;
    let label_width = (row_width - DOT_WIDTH - VALUE_WIDTH - spacing * 2.0).max(72.0);
    let value = match entry.value {
        crate::theme::ThemeValue::Color(_) => String::new(),
        crate::theme::ThemeValue::Number { value, .. } => format!("{value:.1}"),
    };

    let row_rect = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(row_width, 20.0),
    );
    let pointer_over_foreground = ui.ctx().pointer_latest_pos().is_some_and(|pointer| {
        ui.ctx()
            .layer_id_at(pointer)
            .is_some_and(|layer| layer.order >= egui::Order::Middle)
    });
    if !pointer_over_foreground && ui.rect_contains_pointer(row_rect) {
        ui.painter().rect_filled(
            row_rect.expand2(egui::vec2(4.0, 1.0)),
            egui::CornerRadius::same(4),
            theme.with_alpha(theme.highlight, 24),
        );
    }

    ui.scope_builder(
        egui::UiBuilder::new().max_rect(row_rect),
        |ui| {
            ui.horizontal(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(DOT_WIDTH, 20.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(DOT_WIDTH);
                if let crate::theme::ThemeValue::Color(rgba) = entry.value {
                    let dot = theme_color_dot(ui, rgba).on_hover_text(format!(
                        "theme.json: {}\n{}\nClick to edit",
                        entry.key, entry.description
                    ));
                    let popup_id = egui::Id::new(("theme-color-picker", entry.key));
                    let rect_id = egui::Id::new(("theme-color-picker-rect", entry.key));
                    let popup_open = egui::Popup::is_id_open(ui.ctx(), popup_id);
                    let previous_rect = ui.data(|data| data.get_temp::<egui::Rect>(rect_id));
                    if popup_open {
                        if previous_rect.is_some_and(|rect| {
                            pointer_interacted_outside(ui, rect)
                        }) {
                            egui::Popup::close_id(ui.ctx(), popup_id);
                        }
                        picker_pointer_blocker(ui, entry.key);
                    }
                    egui::Popup::menu(&dot)
                        .id(popup_id)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
                        .style(crate::theme::menu_style)
                        .show(|ui| {
                            let rect = ui.max_rect();
                            ui.data_mut(|data| data.insert_temp(rect_id, rect));
                            if let Some(color) = nodecraft_color_picker::color_picker(
                                ui,
                                egui::Id::new(("theme-color-wheel", entry.key)),
                                rgba,
                                nodecraft_color_picker::ColorPickerStyle {
                                    icon: crate::theme::get().icon,
                                    border: crate::theme::get().border_strong,
                                },
                            ) {
                                let _ = theme.set_color(entry.key, color);
                            }
                        });
                } else {
                    ui.allocate_exact_size(egui::vec2(20.0, 20.0), egui::Sense::hover());
                }
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(label_width, 20.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.set_width(label_width);
                ui.add(egui::Label::new(entry.label).truncate())
                    .on_hover_text(format!(
                        "theme.json: {}\n{}",
                        entry.key, entry.description
                    ));
            },
        );
        ui.allocate_ui_with_layout(
            egui::vec2(VALUE_WIDTH, 20.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.set_width(VALUE_WIDTH);
                if let crate::theme::ThemeValue::Number {
                    value: number,
                    min,
                    max,
                    step,
                } = entry.value
                {
                    let mut number = number;
                    let response = if entry.key == "backdrop_blur" {
                        ui.add(
                            egui::Slider::new(&mut number, min..=max)
                                .step_by(f64::from(step))
                                .show_value(true),
                        )
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut number)
                                .range(min..=max)
                                .speed(step)
                                .max_decimals(1),
                        )
                    };
                    if response.changed() {
                        let _ = theme.set_number(entry.key, number);
                    }
                } else if let crate::theme::ThemeValue::Color(rgba) = entry.value {
                    if let Some(color) = theme_hex_edit(ui, entry.key, rgba) {
                        let _ = theme.set_color(entry.key, color);
                    }
                } else {
                    ui.monospace(value);
                }
            },
        );
            });
        },
    );
}

/// Any pointer press or scroll outside the picker dismisses it.
fn pointer_interacted_outside(ui: &egui::Ui, popup: egui::Rect) -> bool {
    ui.ctx().input(|input| {
        let pointer = input.pointer.interact_pos();
        pointer.is_some_and(|pointer| !popup.contains(pointer))
            && (input.pointer.any_pressed()
                || input.smooth_scroll_delta != egui::Vec2::ZERO)
    })
}

/// While the picker is open, sit above the settings body so outside pointer
/// input is consumed by the dismissal layer instead of reaching a control.
fn picker_pointer_blocker(ui: &egui::Ui, key: &str) {
    let screen = ui.ctx().input(|input| input.viewport_rect());
    egui::Area::new(egui::Id::new(("theme-color-picker-blocker", key)))
        .order(egui::Order::Middle)
        .fixed_pos(screen.min)
        .show(ui.ctx(), |ui| {
            ui.allocate_exact_size(screen.size(), egui::Sense::click_and_drag());
        });
}
