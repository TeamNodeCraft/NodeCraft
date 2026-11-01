//! JSON-backed app theme.
//!
//! The embedded theme is the default. A user override can be supplied with
//! `NODECRAFT_THEME=/path/theme.json` or by placing `theme.json` in the app's
//! Application Support directory. Colours are RGBA byte arrays.

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Theme {
    pub window: [u8; 4],
    pub panel: [u8; 4],
    pub control: [u8; 4],
    pub border: [u8; 4],
    pub border_strong: [u8; 4],
    pub text: [u8; 4],
    pub muted: [u8; 4],
    pub accent: [u8; 4],
    pub highlight: [u8; 4],
    #[serde(default = "default_port_input")]
    pub port_input: [u8; 4],
    pub card: [u8; 4],
    pub card_border: [u8; 4],
    pub wire: [u8; 4],
    pub panel_glass: [u8; 4],
    pub minimap_bg: [u8; 4],
    pub minimap_border: [u8; 4],
    pub minimap_node: [u8; 4],
    pub icon: [u8; 4],
    pub icon_hover: [u8; 4],
    pub popup: [u8; 4],
    pub axis_x: [u8; 4],
    pub axis_y: [u8; 4],
    pub axis_z: [u8; 4],
}

fn default_port_input() -> [u8; 4] {
    [255, 159, 10, 255]
}

impl Default for Theme {
    fn default() -> Self {
        serde_json::from_str(include_str!("../assets/theme.json"))
            .expect("embedded theme.json must be valid")
    }
}

impl Theme {
    pub fn color(&self, rgba: [u8; 4]) -> egui::Color32 {
        egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3])
    }

    pub fn hex(rgba: [u8; 4]) -> String {
        format!("#{:02x}{:02x}{:02x}", rgba[0], rgba[1], rgba[2])
    }

    pub fn with_alpha(&self, rgba: [u8; 4], alpha: u8) -> egui::Color32 {
        egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], alpha)
    }

    fn env_override_path() -> Option<PathBuf> {
        std::env::var_os("NODECRAFT_THEME").map(PathBuf::from)
    }

    fn system_theme_path() -> Option<PathBuf> {
        if cfg!(target_os = "macos") {
            let home = std::env::var_os("HOME")?;
            return Some(
                PathBuf::from(home)
                    .join("Library/Application Support/NodeCraft")
                    .join("theme.json"),
            );
        }
        if cfg!(windows) {
            let app_data = std::env::var_os("APPDATA")?;
            return Some(PathBuf::from(app_data).join("NodeCraft").join("theme.json"));
        }
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
        Some(base.join("nodecraft").join("theme.json"))
    }

    fn load() -> Self {
        // `cargo test` must not create or repair the developer's real profile.
        #[cfg(test)]
        return Self::default();
        #[allow(unreachable_code)]
        if let Some(path) = Self::env_override_path() {
            match Self::load_file(&path) {
                Ok(Some(theme)) => return theme,
                Ok(None) => eprintln!("theme: override does not exist: {}", path.display()),
                Err(error) => eprintln!("theme: ignored invalid {}: {error}", path.display()),
            }
            return Self::default();
        }
        if let Some(path) = Self::system_theme_path() {
            return Self::load_or_initialize(&path);
        }
        Self::default()
    }

    fn load_file(path: &std::path::Path) -> Result<Option<Self>, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .map(Some)
                .map_err(|error| error.to_string()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }

    fn write_default(path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, include_str!("../assets/theme.json"))?;
        match std::fs::rename(&temp, path) {
            Ok(()) => Ok(()),
            Err(error) => {
                // Windows cannot rename over an existing file. This path is
                // only reached for a missing or freshly backed-up corrupt file.
                let _ = std::fs::remove_file(path);
                match std::fs::rename(&temp, path) {
                    Ok(()) => Ok(()),
                    Err(second_error) => {
                        let _ = std::fs::remove_file(&temp);
                        Err(if second_error.kind() == std::io::ErrorKind::AlreadyExists {
                            error
                        } else {
                            second_error
                        })
                    }
                }
            }
        }
    }

    fn load_or_initialize(path: &std::path::Path) -> Self {
        match Self::load_file(path) {
            Ok(Some(theme)) => theme,
            Ok(None) => {
                if let Err(error) = Self::write_default(path) {
                    eprintln!("theme: cannot create {}: {error}", path.display());
                }
                Self::default()
            }
            Err(error) => {
                eprintln!("theme: invalid {}: {error}; restoring default", path.display());
                let backup = path.with_extension("json.corrupt");
                if let Err(rename_error) = std::fs::rename(path, &backup) {
                    eprintln!(
                        "theme: cannot preserve corrupt {}: {rename_error}",
                        path.display()
                    );
                }
                if let Err(write_error) = Self::write_default(path) {
                    eprintln!("theme: cannot restore {}: {write_error}", path.display());
                }
                Self::default()
            }
        }
    }
}

static THEME: OnceLock<Theme> = OnceLock::new();

pub fn get() -> &'static Theme {
    THEME.get_or_init(Theme::load)
}

/// macOS-style menu styling shared by every app menu.
///
/// Keep row metrics and strokes identical across inactive/hovered/active/open
/// states: only the fill/foreground colour may change when an item is
/// highlighted, never its padding or border. That prevents the text from
/// shifting as the pointer moves between items.
pub fn menu_style(style: &mut egui::Style) {
    egui::containers::menu::menu_style(style);
    let theme = get();
    let popup = theme.color(theme.popup);
    let accent = theme.color(theme.accent);
    let text = theme.color(theme.text);
    let white = egui::Color32::WHITE;
    let radius = egui::CornerRadius::same(5);

    style.spacing.button_padding = egui::vec2(12.0, 5.0);
    style.spacing.item_spacing = egui::vec2(8.0, 2.0);
    style.spacing.menu_margin = egui::Margin::symmetric(6, 6);
    style.spacing.interact_size.y = 25.0;
    style.visuals.window_fill = popup;
    style.visuals.window_stroke = egui::Stroke::new(1.0, theme.color(theme.border));
    style.visuals.window_corner_radius = egui::CornerRadius::same(8);
    style.visuals.menu_corner_radius = egui::CornerRadius::same(8);
    style.visuals.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 0,
        color: egui::Color32::from_black_alpha(120),
    };

    for visuals in [
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        visuals.bg_stroke = egui::Stroke::NONE;
        visuals.corner_radius = radius;
        visuals.expansion = 0.0;
    }

    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
    style.visuals.widgets.inactive.bg_fill = theme.color(theme.control);
    style.visuals.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, text);

    for visuals in [
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        visuals.weak_bg_fill = accent;
        visuals.bg_fill = accent;
        visuals.fg_stroke = egui::Stroke::new(1.0, white);
    }

    style.visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, theme.color(theme.muted));
    style.visuals.selection.bg_fill = accent;
    style.visuals.selection.stroke = egui::Stroke::new(1.0, white);
}


/// Width for a menu content column, derived from its longest label.
///
/// The indicator column is always reserved, and the label gets a fixed amount
/// of trailing padding. This avoids a wide fixed menu when every item is short.
pub fn menu_width(ui: &egui::Ui, labels: &[&str]) -> f32 {
    let font = egui::FontId::proportional(13.0);
    let longest = labels
        .iter()
        .map(|label| {
            ui.ctx().fonts_mut(|fonts| {
                fonts
                    .layout_no_wrap((*label).to_owned(), font.clone(), egui::Color32::WHITE)
                    .size()
                    .x
            })
        })
        .fold(0.0_f32, f32::max);
    28.0 + longest + 12.0
}

/// Full-row menu item with a fixed indicator column.
///
/// The label is always painted at the same x, whether or not the checkmark is
/// visible. This keeps text from shifting when the selected item changes and
/// matches the layout used by the node cleanup menu.
pub fn menu_row(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response {
    let theme = get();
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 26.0),
        egui::Sense::click(),
    );
    if response.hovered() || response.clicked() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let text = if response.hovered() {
        egui::Color32::WHITE
    } else {
        theme.color(theme.text)
    };
    let fill = if response.hovered() {
        theme.color(theme.accent)
    } else {
        egui::Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, egui::CornerRadius::same(5), fill);
    if selected {
        ui.painter().text(
            egui::pos2(rect.left() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            "✓",
            egui::FontId::proportional(13.0),
            text,
        );
    }
    ui.painter().text(
        egui::pos2(rect.left() + 28.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(13.0),
        text,
    );
    response
}

/// Toggle a boolean menu item, returning whether it changed.
pub fn menu_toggle(ui: &mut egui::Ui, value: &mut bool, label: &str) -> bool {
    if menu_row(ui, *value, label).clicked() {
        *value = !*value;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_theme_path(name: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "nodecraft-theme-{name}-{}-{stamp}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("theme.json")
    }

    #[test]
    fn first_run_writes_the_default_theme() {
        let path = temp_theme_path("first-run");
        let theme = Theme::load_or_initialize(&path);
        assert_eq!(theme.window, Theme::default().window);
        assert!(path.is_file());
        assert!(Theme::load_file(&path).unwrap().is_some());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn corrupt_theme_is_backed_up_and_replaced() {
        let path = temp_theme_path("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{ not json").unwrap();
        let theme = Theme::load_or_initialize(&path);
        assert_eq!(theme.window, Theme::default().window);
        assert!(Theme::load_file(&path).unwrap().is_some());
        assert_eq!(
            std::fs::read_to_string(path.with_extension("json.corrupt")).unwrap(),
            "{ not json"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
