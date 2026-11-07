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
    if ui.rect_contains_pointer(row_rect) {
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
                    egui::Popup::menu(&dot)
                        .id(egui::Id::new(("theme-color-picker", entry.key)))
                        .style(crate::theme::menu_style)
                        .show(|ui| {
                            if let Some(color) = theme_color_wheel(ui, entry.key, rgba) {
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

#[derive(Clone)]
struct WheelTexture {
    texture: egui::TextureHandle,
    value: f32,
}

fn make_color_wheel_image(radius: f32, size: f32, value: f32) -> egui::ColorImage {
    use egui::ecolor::Hsva;
    const WHEEL_SIZE: usize = 256;
    let radius_px = WHEEL_SIZE as f32 * (radius / size);
    let center_px = WHEEL_SIZE as f32 * 0.5;
    let value = value.clamp(0.0, 1.0);
    let mut pixels = Vec::with_capacity(WHEEL_SIZE * WHEEL_SIZE);
    for y in 0..WHEEL_SIZE {
        for x in 0..WHEEL_SIZE {
            let dx = x as f32 + 0.5 - center_px;
            let dy = y as f32 + 0.5 - center_px;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > radius_px {
                pixels.push(egui::Color32::TRANSPARENT);
            } else {
                let hue = dy.atan2(dx).rem_euclid(std::f32::consts::TAU)
                    / std::f32::consts::TAU;
                let saturation = (distance / radius_px).clamp(0.0, 1.0);
                // The wheel is a value map, never an alpha preview: keep it
                // opaque and apply the current luminosity through HSV value.
                pixels.push(egui::Color32::from(Hsva::new(hue, saturation, value, 1.0)));
            }
        }
    }
    egui::ColorImage::new([WHEEL_SIZE, WHEEL_SIZE], pixels)
}

/// Perceptual response for the luminosity slider: slider position is
/// gamma-decoded to HSV value, giving much finer control near black.
const LUMINOSITY_GAMMA: f32 = 2.2;

fn color_picker_slider(
    ui: &mut egui::Ui,
    icon: &'static [u8],
    image_id: &'static str,
    value: &mut f32,
) -> bool {
    let theme = crate::theme::get();
    let svg = crate::app::tint_svg(icon, &crate::theme::Theme::hex(theme.icon));
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::from_bytes(image_id, svg.into_bytes())
                .fit_to_exact_size(egui::Vec2::splat(16.0)),
        );
        ui.add(egui::Slider::new(value, 0.0..=1.0).show_value(true))
            .changed()
    })
    .inner
}

#[derive(Clone, Copy)]
struct ColorWheelState {
    hsva: egui::ecolor::Hsva,
    rgba: [u8; 4],
}

pub(super) fn theme_color_wheel(ui: &mut egui::Ui, key: &str, rgba: [u8; 4]) -> Option<[u8; 4]> {
    use egui::ecolor::Hsva;

    let state_id = ui.make_persistent_id(("theme-color-wheel-state", key));
    let mut state = ui
        .data(|data| data.get_temp::<ColorWheelState>(state_id))
        .filter(|state| state.rgba == rgba)
        .unwrap_or_else(|| {
            // Build HSV from opaque RGB. Converting the themed `Color32`
            // through premultiplied alpha makes low-alpha colours lose RGB
            // precision and can visibly saturate the wheel.
            ColorWheelState {
                hsva: hsva_from_rgba(rgba),
                rgba,
            }
        });

    let size = 188.0;
    let (wheel_rect, wheel_response) =
        ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
    let center = wheel_rect.center();
    let radius = size * 0.5 - 3.0;
    let mut changed = false;
    let mut hsva = state.hsva;

    // Circular HSV hue/saturation wheel at fixed full value. Hue/saturation,
    // luminosity and alpha are independent controls with no feedback path.
    let wheel_value = hsva.v.clamp(0.0, 1.0);
    let texture = {
        let ctx = ui.ctx().clone();
        let id = egui::Id::new("theme-color-wheel-gradient");
        let cached = ctx.data(|data| data.get_temp::<WheelTexture>(id));
        let mut texture = match cached {
            Some(cached) if (cached.value - wheel_value).abs() <= 1.0e-4 => cached.texture,
            Some(mut cached) => {
                // One texture with a full-precision value update avoids the
                // quantized 64-step luminosity ladder.
                cached.texture.set(
                    make_color_wheel_image(radius, size, wheel_value),
                    egui::TextureOptions::LINEAR,
                );
                cached.value = wheel_value;
                let texture = cached.texture.clone();
                ctx.data_mut(|data| data.insert_temp(id, cached));
                texture
            }
            None => {
                let texture = ctx.load_texture(
                    "theme-color-wheel-gradient",
                    make_color_wheel_image(radius, size, wheel_value),
                    egui::TextureOptions::LINEAR,
                );
                ctx.data_mut(|data| {
                    data.insert_temp(
                        id,
                        WheelTexture {
                            texture: texture.clone(),
                            value: wheel_value,
                        },
                    );
                });
                texture
            }
        };
        texture
    };
    ui.painter().image(
        texture.id(),
        wheel_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    ui.painter().circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, crate::theme::get().color(crate::theme::get().border_strong)),
    );

    if let Some(pointer) = wheel_response.interact_pointer_pos() {
        let delta = pointer - center;
        let distance = delta.length();
        if distance <= radius + 2.0 {
            hsva.h = delta.y.atan2(delta.x).rem_euclid(std::f32::consts::TAU)
                / std::f32::consts::TAU;
            hsva.s = (distance / radius).clamp(0.0, 1.0);
            changed = true;
        }
    }

    let marker = center
        + egui::vec2(
            (hsva.h * std::f32::consts::TAU).cos(),
            (hsva.h * std::f32::consts::TAU).sin(),
        ) * (hsva.s * radius);
    let marker_colour = if hsva.v > 0.55 {
        egui::Color32::BLACK
    } else {
        egui::Color32::WHITE
    };
    ui.painter().circle_stroke(
        marker,
        5.0,
        egui::Stroke::new(2.0, marker_colour),
    );
    ui.painter().circle_stroke(
        marker,
        7.0,
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(140)),
    );

    ui.add_space(6.0);
    ui.spacing_mut().slider_width = size - 26.0;
    let mut luminosity = hsva.v.powf(1.0 / LUMINOSITY_GAMMA);
    if color_picker_slider(
        ui,
        include_bytes!("../../../../../vendor/fluent/brightness_high.svg"),
        "bytes://theme-luminosity.svg",
        &mut luminosity,
    ) {
        hsva.v = luminosity.powf(LUMINOSITY_GAMMA);
        changed = true;
    }
    let mut alpha = f32::from(rgba[3]) / 255.0;
    let alpha_changed = color_picker_slider(
        ui,
        include_bytes!("../../../../../vendor/iconify-design/opacity.svg"),
        "bytes://theme-opacity.svg",
        &mut alpha,
    );
    let out_alpha = if alpha_changed {
        (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        rgba[3]
    };
    if changed || alpha_changed {
        let out = if changed {
            hsva_rgba(hsva, out_alpha)
        } else {
            // Alpha edits are byte-only: RGB stays exactly as the theme has it,
            // so the hex editor never shows a colour shift.
            replace_alpha(rgba, out_alpha)
        };
        hsva.a = f32::from(out_alpha) / 255.0;
        state.hsva = hsva;
        state.rgba = out;
        ui.data_mut(|data| data.insert_temp(state_id, state));
        return Some(out);
    }
    None
}


fn hsva_from_rgba(rgba: [u8; 4]) -> egui::ecolor::Hsva {
    let mut hsva = egui::ecolor::Hsva::from_srgba_unmultiplied([
        rgba[0],
        rgba[1],
        rgba[2],
        255,
    ]);
    hsva.a = f32::from(rgba[3]) / 255.0;
    hsva
}

fn hsva_rgba(hsva: egui::ecolor::Hsva, alpha: u8) -> [u8; 4] {
    let color = egui::Color32::from(egui::ecolor::Hsva::new(
        hsva.h,
        hsva.s,
        hsva.v,
        1.0,
    ));
    let [r, g, b, _] = color.to_srgba_unmultiplied();
    [r, g, b, alpha]
}

fn replace_alpha(rgba: [u8; 4], alpha: u8) -> [u8; 4] {
    [rgba[0], rgba[1], rgba[2], alpha]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_half_is_byte_128() {
        assert_eq!((0.5_f32 * 255.0).round() as u8, 128);
    }

    #[test]
    fn alpha_is_the_only_channel_replaced() {
        assert_eq!(replace_alpha([12, 34, 56, 78], 128), [12, 34, 56, 128]);
    }

    #[test]
    fn low_alpha_does_not_change_hsv_rgb() {
        let opaque = hsva_from_rgba([142, 142, 147, 255]);
        let faint = hsva_from_rgba([142, 142, 147, 13]);
        assert!((opaque.h - faint.h).abs() < f32::EPSILON);
        assert!((opaque.s - faint.s).abs() < f32::EPSILON);
        assert!((opaque.v - faint.v).abs() < f32::EPSILON);
        assert!((faint.a - 13.0 / 255.0).abs() < 1.0e-6);
    }

    #[test]
    fn hsva_rgba_keeps_the_requested_alpha() {
        let hsva = egui::ecolor::Hsva::new(0.3, 0.6, 0.7, 0.25);
        assert_eq!(hsva_rgba(hsva, 128)[3], 128);
    }
}
