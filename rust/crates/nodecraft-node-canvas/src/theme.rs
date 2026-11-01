//! Palette supplied by the host application to the node canvas.

use egui::Color32;

#[derive(Clone, Debug)]
pub struct CanvasTheme {
    pub window: [u8; 4],
    pub panel_glass: [u8; 4],
    pub control: [u8; 4],
    pub border: [u8; 4],
    pub text: [u8; 4],
    pub muted: [u8; 4],
    pub accent: [u8; 4],
    pub highlight: [u8; 4],
    pub port_input: [u8; 4],
    pub card: [u8; 4],
    pub card_border: [u8; 4],
    pub cable: [u8; 4],
    pub cable_outline: [u8; 4],
    pub cable_width: f32,
    pub cable_outline_width: f32,
    pub minimap_bg: [u8; 4],
    pub minimap_border: [u8; 4],
    pub minimap_node: [u8; 4],
    pub icon: [u8; 4],
    pub icon_hover: [u8; 4],
    pub popup: [u8; 4],
}

impl Default for CanvasTheme {
    fn default() -> Self {
        Self {
            window: [28, 28, 30, 255],
            panel_glass: [14, 14, 16, 184],
            control: [45, 45, 47, 255],
            border: [58, 58, 60, 255],
            text: [245, 245, 247, 255],
            muted: [152, 152, 157, 255],
            accent: [10, 132, 255, 255],
            highlight: [124, 156, 255, 255],
            port_input: [255, 159, 10, 255],
            card: [60, 60, 63, 255],
            card_border: [72, 72, 74, 255],
            cable: [142, 142, 147, 51],
            cable_outline: [0, 0, 0, 191],
            cable_width: 5.0,
            cable_outline_width: 2.0,
            minimap_bg: [36, 36, 38, 235],
            minimap_border: [58, 58, 60, 230],
            minimap_node: [152, 152, 157, 235],
            icon: [152, 152, 157, 255],
            icon_hover: [245, 245, 247, 255],
            popup: [30, 30, 32, 228],
        }
    }
}

impl std::hash::Hash for CanvasTheme {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::hash::Hash as _;
        for field in [
            self.window,
            self.panel_glass,
            self.control,
            self.border,
            self.text,
            self.muted,
            self.accent,
            self.highlight,
            self.port_input,
            self.card,
            self.card_border,
            self.cable,
            self.cable_outline,
            self.minimap_bg,
            self.minimap_border,
            self.minimap_node,
            self.icon,
            self.icon_hover,
            self.popup,
        ] {
            field.hash(state);
        }
        self.cable_width.to_bits().hash(state);
        self.cable_outline_width.to_bits().hash(state);
    }
}

impl CanvasTheme {
    pub fn color(&self, rgba: [u8; 4]) -> Color32 {
        Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3])
    }

    pub fn f32(&self, rgba: [u8; 4]) -> [f32; 4] {
        [
            f32::from(rgba[0]) / 255.0,
            f32::from(rgba[1]) / 255.0,
            f32::from(rgba[2]) / 255.0,
            f32::from(rgba[3]) / 255.0,
        ]
    }

    pub fn with_alpha(&self, rgba: [u8; 4], alpha: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], alpha)
    }
}
