//! HSV colour wheel with luminosity and opacity controls.
//!
//! The widget has no application/theme dependency: callers pass the current
//! RGBA value plus the icon and border colours used to draw the picker.

use egui::ecolor::Hsva;

/// Colours the picker needs from its host.
#[derive(Clone, Copy)]
pub struct ColorPickerStyle {
    /// Tint for the luminosity and opacity icons.
    pub icon: [u8; 4],
    /// Stroke colour around the wheel.
    pub border: [u8; 4],
}

/// Draw the colour picker. Returns a new RGBA value when the user changes the
/// wheel, luminosity or opacity.
pub fn color_picker(
    ui: &mut egui::Ui,
    id: egui::Id,
    rgba: [u8; 4],
    style: ColorPickerStyle,
) -> Option<[u8; 4]> {
    ui.spacing_mut().interact_size.y = 16.0;
    let state_id = id.with("state");
    let mut state = ui
        .data(|data| data.get_temp::<ColorWheelState>(state_id))
        .filter(|state| state.rgba == rgba)
        .unwrap_or_else(|| ColorWheelState {
            // Build HSV from opaque RGB. Converting a premultiplied Color32
            // through alpha makes low-alpha colours lose RGB precision.
            hsva: hsva_from_rgba(rgba),
            rgba,
        });

    let size = 188.0;
    let (wheel_rect, wheel_response) =
        ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
    let center = wheel_rect.center();
    let radius = size * 0.5 - 3.0;
    let mut changed = false;
    let mut hsva = state.hsva;

    let wheel_value = hsva.v.clamp(0.0, 1.0);
    let texture = wheel_texture(ui, id, radius, size, wheel_value);
    ui.painter().image(
        texture.id(),
        wheel_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    ui.painter().circle_stroke(
        center,
        radius,
        egui::Stroke::new(1.0, color(style.border)),
    );

    if let Some(pointer) = wheel_response.interact_pointer_pos() {
        let (hue, saturation) = wheel_hue_saturation(pointer, center, radius);
        hsva.h = hue;
        hsva.s = saturation;
        changed = true;
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
    // Soft, centered shadow keeps the selector visible on bright and dark
    // wheel regions alike.
    for (radius, alpha) in [(10.0, 22), (9.0, 38), (8.0, 62), (7.0, 90)] {
        ui.painter()
            .circle_filled(marker, radius, egui::Color32::from_black_alpha(alpha));
    }
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
    ui.set_width(size);
    let mut luminosity = hsva.v.powf(1.0 / LUMINOSITY_GAMMA);
    if picker_slider(
        ui,
        "Luminosity",
        include_bytes!("../../../../vendor/fluent/brightness_high.svg"),
        "bytes://nodecraft-color-picker-luminosity.svg",
        &mut luminosity,
        size,
        style.icon,
    ) {
        hsva.v = luminosity.powf(LUMINOSITY_GAMMA);
        changed = true;
    }
    let mut alpha = f32::from(rgba[3]) / 255.0;
    let alpha_changed = picker_slider(
        ui,
        "Opacity",
        include_bytes!("../../../../vendor/iconify-design/opacity.svg"),
        "bytes://nodecraft-color-picker-opacity.svg",
        &mut alpha,
        size,
        style.icon,
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
            // Alpha edits are byte-only: RGB stays exactly as supplied.
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

#[derive(Clone, Copy)]
struct ColorWheelState {
    hsva: Hsva,
    rgba: [u8; 4],
}

#[derive(Clone)]
struct WheelTexture {
    texture: egui::TextureHandle,
    value: f32,
}

fn wheel_texture(
    ui: &egui::Ui,
    id: egui::Id,
    radius: f32,
    size: f32,
    value: f32,
) -> egui::TextureHandle {
    let ctx = ui.ctx().clone();
    let texture_id = id.with("wheel-texture");
    let cached = ctx.data(|data| data.get_temp::<WheelTexture>(texture_id));
    match cached {
        Some(cached) if (cached.value - value).abs() <= 1.0e-4 => cached.texture,
        Some(mut cached) => {
            cached.texture.set(
                make_color_wheel_image(radius, size, value),
                egui::TextureOptions::LINEAR,
            );
            cached.value = value;
            let texture = cached.texture.clone();
            ctx.data_mut(|data| data.insert_temp(texture_id, cached));
            texture
        }
        None => {
            let texture = ctx.load_texture(
                format!("nodecraft-color-wheel-{id:?}"),
                make_color_wheel_image(radius, size, value),
                egui::TextureOptions::LINEAR,
            );
            ctx.data_mut(|data| {
                data.insert_temp(
                    texture_id,
                    WheelTexture {
                        texture: texture.clone(),
                        value,
                    },
                );
            });
            texture
        }
    }
}

fn make_color_wheel_image(radius: f32, size: f32, value: f32) -> egui::ColorImage {
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
                // The wheel is a value map, never an alpha preview.
                pixels.push(egui::Color32::from(Hsva::new(hue, saturation, value, 1.0)));
            }
        }
    }
    egui::ColorImage::new([WHEEL_SIZE, WHEEL_SIZE], pixels)
}

const LUMINOSITY_GAMMA: f32 = 2.2;

fn picker_slider(
    ui: &mut egui::Ui,
    name: &str,
    icon: &'static [u8],
    image_id: &'static str,
    value: &mut f32,
    width: f32,
    icon_tint: [u8; 4],
) -> bool {
    const ICON: f32 = 16.0;
    const VALUE: f32 = 44.0;
    let svg = tint_svg(icon, icon_tint);
    ui.spacing_mut().slider_width = (width - ICON - VALUE - ui.spacing().item_spacing.x * 2.0)
        .max(60.0);
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::from_bytes(image_id, svg.into_bytes())
                .fit_to_exact_size(egui::Vec2::splat(ICON)),
        )
        .on_hover_text(name);
        let response = ui.add(egui::Slider::new(value, 0.0..=1.0).show_value(false));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_sized(
                [VALUE, ui.spacing().interact_size.y],
                egui::Label::new(format!("{value:.2}")),
            );
        });
        response.changed()
    })
    .inner
}

fn color(rgba: [u8; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3])
}

fn hex(rgba: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgba[0], rgba[1], rgba[2])
}

fn tint_svg(source: &[u8], colour: [u8; 4]) -> String {
    let colour = hex(colour);
    std::str::from_utf8(source)
        .unwrap_or_default()
        .replace("#212121", &colour)
        .replace("currentColor", &colour)
}

fn wheel_hue_saturation(pointer: egui::Pos2, center: egui::Pos2, radius: f32) -> (f32, f32) {
    let delta = pointer - center;
    let distance = delta.length();
    let hue = delta.y.atan2(delta.x).rem_euclid(std::f32::consts::TAU)
        / std::f32::consts::TAU;
    (hue, (distance / radius.max(0.001)).clamp(0.0, 1.0))
}

fn hsva_from_rgba(rgba: [u8; 4]) -> Hsva {
    let mut hsva = Hsva::from_srgba_unmultiplied([rgba[0], rgba[1], rgba[2], 255]);
    hsva.a = f32::from(rgba[3]) / 255.0;
    hsva
}

fn hsva_rgba(hsva: Hsva, alpha: u8) -> [u8; 4] {
    let color = egui::Color32::from(Hsva::new(hsva.h, hsva.s, hsva.v, 1.0));
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
    fn selector_clamps_outside_the_rim() {
        let center = egui::Pos2::new(100.0, 100.0);
        let (_, saturation) =
            wheel_hue_saturation(egui::Pos2::new(400.0, 100.0), center, 80.0);
        assert_eq!(saturation, 1.0);
        let (_, inside) =
            wheel_hue_saturation(egui::Pos2::new(140.0, 100.0), center, 80.0);
        assert!((inside - 0.5).abs() < 1.0e-6);
    }

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
        let hsva = Hsva::new(0.3, 0.6, 0.7, 0.25);
        assert_eq!(hsva_rgba(hsva, 128)[3], 128);
    }
}
