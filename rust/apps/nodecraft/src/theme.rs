//! JSON-backed app theme.
//!
//! The embedded theme is the default. A user override can be supplied with
//! `NODECRAFT_THEME=/path/theme.json` or by placing `theme.json` in the app's
//! Application Support directory. Colours are RGBA byte arrays.

use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Theme {
    pub window: [u8; 4],
    #[serde(default = "default_viewport_background")]
    pub viewport_background: [u8; 4],
    #[serde(default = "default_grid_minor")]
    pub grid_minor: [u8; 4],
    #[serde(default = "default_grid_major")]
    pub grid_major: [u8; 4],
    pub panel: [u8; 4],
    pub control: [u8; 4],
    #[serde(default = "default_dock")]
    pub dock: [u8; 4],
    pub border: [u8; 4],
    #[serde(default = "default_dock_border")]
    pub dock_border: [u8; 4],
    pub border_strong: [u8; 4],
    pub text: [u8; 4],
    pub muted: [u8; 4],
    pub accent: [u8; 4],
    pub highlight: [u8; 4],
    #[serde(default = "default_node_activity")]
    pub node_activity: [u8; 4],
    #[serde(default = "default_node_error")]
    pub node_error: [u8; 4],
    #[serde(default = "default_port_input")]
    pub port_input: [u8; 4],
    pub card: [u8; 4],
    pub card_border: [u8; 4],
    #[serde(alias = "wire")]
    pub cable: [u8; 4],
    #[serde(default = "default_cable_selected")]
    pub cable_selected: [u8; 4],
    #[serde(default = "default_cable_selected_outline")]
    pub cable_selected_outline: [u8; 4],
    #[serde(
        default = "default_cable_outline",
        alias = "cable_outline_opacity",
        deserialize_with = "deserialize_cable_outline"
    )]
    pub cable_outline: [u8; 4],
    #[serde(default = "default_cable_width")]
    pub cable_width: f32,
    #[serde(default = "default_cable_outline_width")]
    pub cable_outline_width: f32,
    #[serde(default = "default_backdrop_blur")]
    pub backdrop_blur: f32,
    #[serde(alias = "panel_glass")]
    pub blur_tint: [u8; 4],
    pub minimap_bg: [u8; 4],
    pub minimap_border: [u8; 4],
    pub minimap_node: [u8; 4],
    pub icon: [u8; 4],
    pub icon_hover: [u8; 4],
    pub popup: [u8; 4],
    pub axis_x: [u8; 4],
    pub axis_y: [u8; 4],
    pub axis_z: [u8; 4],
    #[serde(default = "default_preview_add")]
    pub preview_add: [u8; 4],
    #[serde(default = "default_preview_cut")]
    pub preview_cut: [u8; 4],
    #[serde(default = "default_ghost")]
    pub ghost: [u8; 4],
    #[serde(default = "default_ghost_edge")]
    pub ghost_edge: [u8; 4],
}

fn default_dock() -> [u8; 4] {
    [45, 45, 47, 255]
}

fn default_dock_border() -> [u8; 4] {
    [58, 58, 60, 255]
}

fn default_node_activity() -> [u8; 4] {
    [10, 132, 255, 255]
}

fn default_node_error() -> [u8; 4] {
    [255, 38, 38, 255]
}

fn default_preview_add() -> [u8; 4] {
    [74, 158, 255, 128]
}

fn default_preview_cut() -> [u8; 4] {
    [255, 92, 92, 128]
}

fn default_ghost() -> [u8; 4] {
    [97, 148, 199, 255]
}

fn default_ghost_edge() -> [u8; 4] {
    [24, 58, 122, 255]
}

fn default_port_input() -> [u8; 4] {
    [255, 159, 10, 255]
}

fn default_cable_outline() -> [u8; 4] {
    [0, 0, 0, 191]
}

fn default_cable_selected() -> [u8; 4] {
    [10, 132, 255, 255]
}

fn default_cable_selected_outline() -> [u8; 4] {
    [0, 0, 0, 191]
}

fn default_cable_width() -> f32 {
    5.0
}

fn default_cable_outline_width() -> f32 {
    2.0
}

fn default_backdrop_blur() -> f32 {
    18.0
}

fn default_viewport_background() -> [u8; 4] {
    [28, 28, 30, 255]
}

fn default_grid_minor() -> [u8; 4] {
    [152, 152, 157, 255]
}

fn default_grid_major() -> [u8; 4] {
    [245, 245, 247, 255]
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CableOutlineValue {
    Rgba([u8; 4]),
    Opacity(f32),
}

fn deserialize_cable_outline<'de, D>(deserializer: D) -> Result<[u8; 4], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = CableOutlineValue::deserialize(deserializer)?;
    Ok(match value {
        CableOutlineValue::Rgba(rgba) => rgba,
        CableOutlineValue::Opacity(opacity) => [
            0,
            0,
            0,
            (opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
        ],
    })
}

pub const DEFAULT_THEME_NAME: &str = "NodeCraft Dark";
pub const LIGHT_THEME_NAME: &str = "NodeCraft Light";
pub const FOREST_THEME_NAME: &str = "NodeCraft Forest";
pub const PUNK_THEME_NAME: &str = "NodeCraft Punk";
pub const CUSTOM_THEME_NAME: &str = "NodeCraft Dark (custom)";
pub const LIGHT_CUSTOM_THEME_NAME: &str = "NodeCraft Light (custom)";
pub const FOREST_CUSTOM_THEME_NAME: &str = "NodeCraft Forest (custom)";
pub const PUNK_CUSTOM_THEME_NAME: &str = "NodeCraft Punk (custom)";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThemePreset {
    pub name: String,
    /// `"default"` for the embedded palette, otherwise the theme file path.
    pub id: String,
    pub path: Option<PathBuf>,
}

fn default_preset() -> ThemePreset {
    ThemePreset {
        name: DEFAULT_THEME_NAME.to_owned(),
        id: "default".to_owned(),
        path: None,
    }
}

fn light_preset() -> ThemePreset {
    ThemePreset {
        name: LIGHT_THEME_NAME.to_owned(),
        id: "light".to_owned(),
        path: None,
    }
}

fn forest_preset() -> ThemePreset {
    ThemePreset {
        name: FOREST_THEME_NAME.to_owned(),
        id: "forest".to_owned(),
        path: None,
    }
}

fn punk_preset() -> ThemePreset {
    ThemePreset {
        name: PUNK_THEME_NAME.to_owned(),
        id: "punk".to_owned(),
        path: None,
    }
}

fn preset_from_path(path: PathBuf, name: Option<String>) -> ThemePreset {
    let id = path.to_string_lossy().into_owned();
    let name = name.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("Custom Theme")
            .replace(['-', '_'], " ")
    });
    ThemePreset { name, id, path: Some(path) }
}

#[derive(Clone, Debug)]
struct ThemeState {
    theme: Theme,
    preset: ThemePreset,
}

impl Default for ThemeState {
    fn default() -> Self {
        Self { theme: Theme::default(), preset: default_preset() }
    }
}

impl Default for Theme {
    fn default() -> Self {
        serde_json::from_str(include_str!("../assets/theme.json"))
            .expect("embedded theme.json must be valid")
    }
}

impl Theme {
    pub fn light() -> Self {
        serde_json::from_str(include_str!("../assets/theme-light.json"))
            .expect("embedded theme-light.json must be valid")
    }

    pub fn forest() -> Self {
        serde_json::from_str(include_str!("../assets/theme-forest.json"))
            .expect("embedded theme-forest.json must be valid")
    }

    pub fn punk() -> Self {
        serde_json::from_str(include_str!("../assets/theme-punk.json"))
            .expect("embedded theme-punk.json must be valid")
    }
}

impl Theme {
    pub fn color(&self, rgba: [u8; 4]) -> egui::Color32 {
        egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3])
    }

    pub fn hex(rgba: [u8; 4]) -> String {
        format!("#{:02x}{:02x}{:02x}", rgba[0], rgba[1], rgba[2])
    }

    /// Hex colour including alpha, used by editable settings fields.
    pub fn hex_rgba(rgba: [u8; 4]) -> String {
        format!(
            "#{:02x}{:02x}{:02x}{:02x}",
            rgba[0], rgba[1], rgba[2], rgba[3]
        )
    }

    /// Parse `#RGB`, `#RGBA`, `#RRGGBB` or `#RRGGBBAA`.
    ///
    /// Six-digit input keeps `fallback_alpha`; three/four-digit input expands
    /// each nibble. The optional leading `#` is accepted.
    pub fn parse_hex(text: &str, fallback_alpha: u8) -> Option<[u8; 4]> {
        let hex = text.trim().strip_prefix('#').unwrap_or(text.trim());
        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let nibble = |c: char| c.to_digit(16).map(|v| v as u8);
        let byte = |chars: &[char]| -> Option<u8> {
            Some(nibble(*chars.first()?)? << 4 | nibble(*chars.get(1)?)?)
        };
        let rgba = match hex.len() {
            3 | 4 => {
                let c: Vec<char> = hex.chars().collect();
                let expand = |x: char| nibble(x).map(|v| v * 17);
                [
                    expand(c[0])?,
                    expand(c[1])?,
                    expand(c[2])?,
                    if c.len() == 4 { expand(c[3])? } else { fallback_alpha },
                ]
            }
            6 | 8 => {
                let c: Vec<char> = hex.chars().collect();
                [
                    byte(&c[0..2])?,
                    byte(&c[2..4])?,
                    byte(&c[4..6])?,
                    if c.len() == 8 { byte(&c[6..8])? } else { fallback_alpha },
                ]
            }
            _ => return None,
        };
        Some(rgba)
    }

    pub fn with_alpha(&self, rgba: [u8; 4], alpha: u8) -> egui::Color32 {
        egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], alpha)
    }

    /// Test-covered mapping for the backdrop's dual-Kawase sample scale.
    #[cfg(test)]
    pub fn backdrop_blur_scale(&self) -> f32 {
        (self.backdrop_blur / 24.0).clamp(0.0, 2.5)
    }

    /// Test-covered number of pyramid levels used by the backdrop blur.
    #[cfg(test)]
    pub fn backdrop_blur_levels(&self) -> usize {
        if self.backdrop_blur <= 0.0 {
            return 0;
        }
        ((self.backdrop_blur / 18.0 * 5.0).ceil() as usize).clamp(1, 5)
    }

    /// Test-covered blend from the direct low-pass into the full pyramid.
    #[cfg(test)]
    pub fn backdrop_blur_mix(&self) -> f32 {
        let t = (self.backdrop_blur / 18.0).clamp(0.0, 1.0);
        t * t
    }

    /// Map the user-facing radius to the backdrop's dual-Kawase sample scale.
    /// 24 px is one full sample-scale unit; the default 18 px preserves the
    /// previous 0.75 backdrop blur.
    /// Number of pyramid levels used by the backdrop blur.
    ///
    /// One level is the smallest nonzero blur; five is the full chain. The
    /// 18 px reference keeps the previous full-strength appearance.
    /// Blend from the direct low-pass into the full pyramid.
    /// Update one palette entry by its JSON key. Returns false for unknown keys.
    pub fn set_color(&mut self, key: &str, value: [u8; 4]) -> bool {
        match key {
            "window" => self.window = value,
            "viewport_background" => self.viewport_background = value,
            "grid_minor" => self.grid_minor = value,
            "grid_major" => self.grid_major = value,
            "panel" => self.panel = value,
            "control" => self.control = value,
            "dock" => self.dock = value,
            "border" => self.border = value,
            "dock_border" => self.dock_border = value,
            "border_strong" => self.border_strong = value,
            "text" => self.text = value,
            "muted" => self.muted = value,
            "accent" => self.accent = value,
            "highlight" => self.highlight = value,
            "node_activity" => self.node_activity = value,
            "node_error" => self.node_error = value,
            "port_input" => self.port_input = value,
            "card" => self.card = value,
            "card_border" => self.card_border = value,
            "cable" => {
                let alpha_changed = self.cable[3] != value[3];
                self.cable = value;
                if alpha_changed {
                    self.cable_selected[3] = value[3];
                }
            }
            "cable_selected" => {
                let alpha_changed = self.cable_selected[3] != value[3];
                self.cable_selected = value;
                if alpha_changed {
                    self.cable[3] = value[3];
                }
            }
            "cable_outline" => {
                let alpha_changed = self.cable_outline[3] != value[3];
                self.cable_outline = value;
                if alpha_changed {
                    self.cable_selected_outline[3] = value[3];
                }
            }
            "cable_selected_outline" => {
                let alpha_changed = self.cable_selected_outline[3] != value[3];
                self.cable_selected_outline = value;
                if alpha_changed {
                    self.cable_outline[3] = value[3];
                }
            }
            "blur_tint" | "panel_glass" => self.blur_tint = value,
            "minimap_bg" => self.minimap_bg = value,
            "minimap_border" => self.minimap_border = value,
            "minimap_node" => self.minimap_node = value,
            "icon" => self.icon = value,
            "icon_hover" => self.icon_hover = value,
            "popup" => self.popup = value,
            "axis_x" => self.axis_x = value,
            "axis_y" => self.axis_y = value,
            "axis_z" => self.axis_z = value,
            "ghost" => self.ghost = value,
            "ghost_edge" => self.ghost_edge = value,
            _ => return false,
        }
        true
    }

    pub fn set_number(&mut self, key: &str, value: f32) -> bool {
        match key {
            "cable_width" => self.cable_width = value.clamp(1.0, 32.0),
            "cable_outline_width" => self.cable_outline_width = value.clamp(0.0, 15.0),
            "backdrop_blur" => self.backdrop_blur = value.clamp(0.0, 128.0),
            _ => return false,
        }
        true
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

    fn config_dir() -> Option<PathBuf> {
        Self::system_theme_path()?.parent().map(PathBuf::from)
    }

    fn themes_dir() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("themes"))
    }

    fn legacy_custom_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("nodecraft-dark-user.json"))
    }

    fn light_custom_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("nodecraft-light-user.json"))
    }

    fn forest_custom_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("nodecraft-forest-user.json"))
    }

    fn punk_custom_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("nodecraft-punk-user.json"))
    }

    fn selection_path() -> Option<PathBuf> {
        Self::config_dir().map(|dir| dir.join("theme-selection"))
    }

    fn write_theme(path: &std::path::Path, theme: &Theme) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let json = serde_json::to_string_pretty(theme).map_err(|error| error.to_string())?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, json).map_err(|error| error.to_string())?;
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            error.to_string()
        })
    }

    fn read_selection() -> Option<String> {
        Self::selection_path()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| text.trim().to_owned())
    }

    fn write_selection(preset: &ThemePreset) {
        if let Some(path) = Self::selection_path() {
            let _ = std::fs::write(path, &preset.id);
        }
    }

    fn load_state() -> ThemeState {
        // `cargo test` must not create or repair the developer's real profile.
        #[cfg(test)]
        return ThemeState::default();
        #[allow(unreachable_code)]
        if let Some(path) = Self::env_override_path() {
            return match Self::load_file(&path) {
                Ok(Some(theme)) => ThemeState {
                    preset: preset_from_path(path, None),
                    theme,
                },
                Ok(None) => ThemeState::default(),
                Err(error) => {
                    eprintln!("theme: ignored invalid {}: {error}", path.display());
                    ThemeState::default()
                }
            };
        }

        let selection = Self::read_selection();
        if let Some(selected) = selection.as_deref() {
            if selected == "light" {
                return ThemeState {
                    theme: Theme::light(),
                    preset: light_preset(),
                };
            }
            if selected == "forest" {
                return ThemeState {
                    theme: Theme::forest(),
                    preset: forest_preset(),
                };
            }
            if selected == "punk" {
                return ThemeState {
                    theme: Theme::punk(),
                    preset: punk_preset(),
                };
            }
            if selected != "default"
                && let Some(path) = user_theme_files()
                    .into_iter()
                    .find(|preset| preset.id == selected)
                    .and_then(|preset| preset.path)
                && let Ok(Some(theme)) = Self::load_file(&path)
            {
                return ThemeState {
                    preset: preset_from_path(path, None),
                    theme,
                };
            }
        } else if let Some(path) = Self::legacy_custom_path()
            && path.is_file()
            && let Ok(Some(theme)) = Self::load_file(&path)
        {
            return ThemeState {
                preset: preset_from_path(path, Some(CUSTOM_THEME_NAME.to_owned())),
                theme,
            };
        }

        // Migrate a legacy custom theme.json once.
        if let Some(base_path) = Self::system_theme_path() {
            let base = Self::load_or_initialize(&base_path);
            if base != Theme::default()
                && Self::legacy_custom_path().as_deref().is_none_or(|path| !path.is_file())
            {
                let preset = Self::legacy_custom_path()
                    .map(|path| preset_from_path(path, Some(CUSTOM_THEME_NAME.to_owned())))
                    .unwrap_or_else(default_preset);
                if let Some(path) = &preset.path {
                    let _ = Self::write_theme(path, &base);
                }
                Self::write_selection(&preset);
                return ThemeState { theme: base, preset };
            }
        }
        ThemeState::default()
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

static THEME: OnceLock<RwLock<ThemeState>> = OnceLock::new();

fn theme_cell() -> &'static RwLock<ThemeState> {
    THEME.get_or_init(|| RwLock::new(Theme::load_state()))
}

/// Snapshot of the current runtime theme.
///
/// Themes are small and the UI is the only writer, so cloning the palette per
/// frame is cheaper than threading a guard through every widget call.
pub fn get() -> Theme {
    theme_cell()
        .read()
        .map(|state| state.theme.clone())
        .unwrap_or_else(|_| Theme::default())
}

/// Replace the runtime theme after a live edit.
pub fn set(theme: Theme) {
    let mut state = theme_cell()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.theme = theme;
}

pub fn active_preset() -> ThemePreset {
    theme_cell()
        .read()
        .map(|state| state.preset.clone())
        .unwrap_or_else(|_| default_preset())
}

pub fn active_name() -> String {
    active_preset().name
}

/// The baseline palette for the active built-in theme.
pub fn active_defaults() -> Theme {
    let preset = active_preset();
    match preset.path.is_none().then_some(preset.id.as_str()) {
        Some("light") => Theme::light(),
        Some("forest") => Theme::forest(),
        Some("punk") => Theme::punk(),
        _ => Theme::default(),
    }
}

/// Does the live palette differ from the active theme's embedded baseline?
fn user_theme_files() -> Vec<ThemePreset> {
    let mut themes = Vec::new();
    if let Some(path) = Theme::legacy_custom_path()
        && path.is_file()
    {
        themes.push(preset_from_path(path, Some(CUSTOM_THEME_NAME.to_owned())));
    }
    if let Some(path) = Theme::light_custom_path()
        && path.is_file()
    {
        themes.push(preset_from_path(path, Some(LIGHT_CUSTOM_THEME_NAME.to_owned())));
    }
    if let Some(path) = Theme::forest_custom_path()
        && path.is_file()
    {
        themes.push(preset_from_path(path, Some(FOREST_CUSTOM_THEME_NAME.to_owned())));
    }
    if let Some(path) = Theme::punk_custom_path()
        && path.is_file()
    {
        themes.push(preset_from_path(path, Some(PUNK_CUSTOM_THEME_NAME.to_owned())));
    }
    if let Some(dir) = Theme::themes_dir()
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            if themes.iter().any(|preset| preset.path.as_deref() == Some(&path)) {
                continue;
            }
            themes.push(preset_from_path(path, None));
        }
    }
    themes.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    themes
}

pub fn available_themes() -> Vec<ThemePreset> {
    let mut themes = vec![default_preset(), light_preset(), forest_preset(), punk_preset()];
    themes.extend(user_theme_files());
    let active = active_preset();
    if !themes.iter().any(|preset| preset.id == active.id) {
        themes.push(active);
    }
    themes
}

pub fn load_preset(preset: &ThemePreset) -> Result<Theme, String> {
    let theme = match preset.path.as_deref() {
        None => match preset.id.as_str() {
            "light" => Theme::light(),
            "forest" => Theme::forest(),
            "punk" => Theme::punk(),
            _ => Theme::default(),
        },
        Some(path) => Theme::load_file(path)?
            .ok_or_else(|| format!("theme does not exist: {}", path.display()))?,
    };
    Theme::write_selection(preset);
    let mut state = theme_cell()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.theme = theme.clone();
    state.preset = preset.clone();
    Ok(theme)
}

/// Persist live edits to the active theme file, creating the legacy user theme
/// when the embedded default is being edited.
pub fn save(theme: &Theme) -> Result<(), String> {
    let mut preset = active_preset();
    let (path, custom_name) = match &preset.path {
        Some(path) => (path.clone(), CUSTOM_THEME_NAME),
        None if preset.id == "light" => (
            Theme::light_custom_path()
                .ok_or_else(|| "no user theme path available".to_owned())?,
            LIGHT_CUSTOM_THEME_NAME,
        ),
        None if preset.id == "forest" => (
            Theme::forest_custom_path()
                .ok_or_else(|| "no user theme path available".to_owned())?,
            FOREST_CUSTOM_THEME_NAME,
        ),
        None if preset.id == "punk" => (
            Theme::punk_custom_path()
                .ok_or_else(|| "no user theme path available".to_owned())?,
            PUNK_CUSTOM_THEME_NAME,
        ),
        None => (
            Theme::legacy_custom_path()
                .ok_or_else(|| "no user theme path available".to_owned())?,
            CUSTOM_THEME_NAME,
        ),
    };
    Theme::write_theme(&path, theme)?;
    if preset.path.is_none() {
        preset = preset_from_path(path, Some(custom_name.to_owned()));
    }
    Theme::write_selection(&preset);
    let mut state = theme_cell()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.theme = theme.clone();
    state.preset = preset;
    Ok(())
}

/// Create a named user theme from the current live palette.
pub fn create_theme(name: &str, theme: &Theme) -> Result<ThemePreset, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Theme name cannot be empty".to_owned());
    }
    let dir = Theme::themes_dir().ok_or_else(|| "no theme directory available".to_owned())?;
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let slug: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "theme" } else { slug };
    let mut path = dir.join(format!("{slug}.json"));
    let mut suffix = 2;
    while path.exists() {
        path = dir.join(format!("{slug}-{suffix}.json"));
        suffix += 1;
    }
    Theme::write_theme(&path, theme)?;
    let preset = preset_from_path(path, Some(name.to_owned()));
    Theme::write_selection(&preset);
    let mut state = theme_cell()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.theme = theme.clone();
    state.preset = preset.clone();
    Ok(preset)
}

/// Rename a user theme file and update the active preset if it is selected.
pub fn rename_theme(preset: &ThemePreset, name: &str) -> Result<ThemePreset, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Theme name cannot be empty".to_owned());
    }
    let Some(old_path) = &preset.path else {
        return Err("The default theme cannot be renamed".to_owned());
    };
    let slug: String = name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_lowercase() } else { '-' })
        .collect();
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "theme" } else { slug };
    let mut new_path = old_path.with_file_name(format!("{slug}.json"));
    let mut suffix = 2;
    while new_path != *old_path && new_path.exists() {
        new_path = old_path.with_file_name(format!("{slug}-{suffix}.json"));
        suffix += 1;
    }
    if new_path != *old_path {
        std::fs::rename(old_path, &new_path).map_err(|error| error.to_string())?;
    }
    let new_preset = preset_from_path(new_path, Some(name.to_owned()));
    if active_preset().id == preset.id {
        Theme::write_selection(&new_preset);
        let mut state = theme_cell()
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.preset = new_preset.clone();
    }
    Ok(new_preset)
}

/// Reset the active theme palette to the embedded defaults.
///
/// A custom theme keeps its name and file; only its colours are reset.
pub fn reset_current_theme() -> Result<Theme, String> {
    let preset = active_preset();
    let theme = match preset.path.is_none().then_some(preset.id.as_str()) {
        Some("light") => Theme::light(),
        Some("forest") => Theme::forest(),
        Some("punk") => Theme::punk(),
        _ => Theme::default(),
    };
    if let Some(path) = &preset.path {
        Theme::write_theme(path, &theme)?;
    }
    let mut state = theme_cell()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.theme = theme.clone();
    state.preset = preset;
    Ok(theme)
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
    fn every_palette_colour_key_is_editable() {
        for group in palette_groups(&Theme::default()) {
            for entry in group.entries {
                match entry.value {
                    ThemeValue::Color(_) => {
                        let mut theme = Theme::default();
                        assert!(
                            theme.set_color(entry.key, [1, 2, 3, 4]),
                            "palette colour key {:?} cannot be set",
                            entry.key
                        );
                    }
                    ThemeValue::Number { value, .. } => {
                        let mut theme = Theme::default();
                        assert!(
                            theme.set_number(entry.key, value),
                            "palette number key {:?} cannot be set",
                            entry.key
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn legacy_cable_outline_opacity_migrates_to_black_rgba() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../assets/theme.json")).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("cable_outline");
        object.insert("cable_outline_opacity".to_owned(), serde_json::json!(0.5));
        let theme: Theme = serde_json::from_value(value).unwrap();
        assert_eq!(theme.cable_outline, [0, 0, 0, 128]);
        assert_eq!(Theme::default().cable_outline, [0, 0, 0, 191]);
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

/// A theme value exposed in the settings palette viewer.
#[derive(Clone, Copy, Debug)]
pub enum ThemeValue {
    Color([u8; 4]),
    Number {
        value: f32,
        min: f32,
        max: f32,
        step: f32,
    },
}

/// One read-only entry in a theme palette category.
#[derive(Clone, Copy, Debug)]
pub struct ThemeEntry {
    pub key: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub value: ThemeValue,
}

/// A semantic category in the theme palette viewer.
pub struct ThemeGroup {
    pub title: &'static str,
    pub description: &'static str,
    pub entries: Vec<ThemeEntry>,
}

fn entry(
    key: &'static str,
    label: &'static str,
    description: &'static str,
    value: ThemeValue,
) -> ThemeEntry {
    ThemeEntry { key, label, description, value }
}

fn color(key: &'static str, label: &'static str, description: &'static str, value: [u8; 4]) -> ThemeEntry {
    entry(key, label, description, ThemeValue::Color(value))
}

fn number(
    key: &'static str,
    label: &'static str,
    description: &'static str,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
) -> ThemeEntry {
    entry(
        key,
        label,
        description,
        ThemeValue::Number { value, min, max, step },
    )
}

/// Theme data grouped by the part of the application it styles.
///
/// The settings UI renders this instead of dumping raw JSON: each field has a
/// semantic label and a visible swatch, while the JSON key remains available in
/// the tooltip for users editing `theme.json`.
pub fn palette_groups(theme: &Theme) -> Vec<ThemeGroup> {
    vec![
        ThemeGroup {
            title: "GUI",
            description: "Window, panels, controls and popup surfaces.",
            entries: vec![
                color("window", "Window", "Main application background", theme.window),
                color("panel", "Panel", "Standard panel background", theme.panel),
                color("control", "Control", "Buttons and inset controls", theme.control),
                color("dock", "Dock background", "Floating tool dock fill", theme.dock),
                color("border", "Border", "Default dividers and outlines", theme.border),
                color("dock_border", "Dock border", "Floating tool dock outline", theme.dock_border),
                color("border_strong", "Strong border", "Selected and emphasized outlines", theme.border_strong),
                color("popup", "Popup", "Menus and floating popup background", theme.popup),
                color("blur_tint", "Blur tint", "Tint applied to the blurred backdrop", theme.blur_tint),
                number(
                    "backdrop_blur",
                    "Backdrop blur",
                    "Approximate screen-space blur radius; 0 disables",
                    theme.backdrop_blur,
                    0.0,
                    128.0,
                    0.25,
                ),
            ],
        },
        ThemeGroup {
            title: "Text & selection",
            description: "Typography, accent and highlighted content.",
            entries: vec![
                color("text", "Text", "Primary text", theme.text),
                color("muted", "Muted text", "Secondary labels and hints", theme.muted),
                color("accent", "Accent", "Primary selection and action colour", theme.accent),
                color("highlight", "Highlight", "Secondary highlight and hover wash", theme.highlight),
                color("node_activity", "Node activity", "Node processing glow colour", theme.node_activity),
                color("node_error", "Node error", "Failed-node glow colour", theme.node_error),
            ],
        },
        ThemeGroup {
            title: "Node editor",
            description: "Cards, cables and node-canvas-specific surfaces.",
            entries: vec![
                color("card", "Card", "Node body background", theme.card),
                color("card_border", "Card border", "Node outline", theme.card_border),
                color("cable", "Cable", "Cable core colour", theme.cable),
                color(
                    "cable_outline",
                    "Cable outline",
                    "Dark cable outline",
                    theme.cable_outline,
                ),
                color(
                    "cable_selected",
                    "Selected cable",
                    "Cable core when selected or highlighted",
                    theme.cable_selected,
                ),
                color(
                    "cable_selected_outline",
                    "Selected outline",
                    "Outline for selected or highlighted cables",
                    theme.cable_selected_outline,
                ),
                number(
                    "cable_width",
                    "Cable width",
                    "Cable core diameter",
                    theme.cable_width,
                    1.0,
                    32.0,
                    0.5,
                ),
                number(
                    "cable_outline_width",
                    "Outline width",
                    "Added outside diameter",
                    theme.cable_outline_width,
                    0.0,
                    15.0,
                    0.5,
                ),
                color("port_input", "Data port", "Input port that carries data", theme.port_input),
            ],
        },
        ThemeGroup {
            title: "Minimap",
            description: "Node-overview map in the node editor.",
            entries: vec![
                color("minimap_bg", "Background", "Minimap card background", theme.minimap_bg),
                color("minimap_border", "Border", "Minimap card outline", theme.minimap_border),
                color("minimap_node", "Unselected node", "Node mark in the overview", theme.minimap_node),
            ],
        },
        ThemeGroup {
            title: "Icons",
            description: "Fluent icon tint states.",
            entries: vec![
                color("icon", "Icon", "Default icon tint", theme.icon),
                color("icon_hover", "Icon hover", "Hover/active icon tint", theme.icon_hover),
            ],
        },
        ThemeGroup {
            title: "3D viewport",
            description: "Background, ground grid and world-axis colours.",
            entries: vec![
                color(
                    "viewport_background",
                    "Background",
                    "3D viewport clear colour",
                    theme.viewport_background,
                ),
                color("grid_minor", "Grid minor", "Minor ground-grid lines", theme.grid_minor),
                color("grid_major", "Grid major", "Major ground-grid lines", theme.grid_major),
                color("axis_x", "X axis", "World X axis colour", theme.axis_x),
                color("axis_y", "Y axis", "World Y axis colour", theme.axis_y),
                color("axis_z", "Z axis", "World Z axis colour", theme.axis_z),
                color("ghost", "Edit ghost", "Transform edit ghost face colour", theme.ghost),
                color(
                    "ghost_edge",
                    "Edit ghost edges",
                    "Transform edit ghost B-rep edge colour",
                    theme.ghost_edge,
                ),
            ],
        },
    ]
}

#[cfg(test)]
mod palette_tests {
    use super::*;

    #[test]
    fn palette_groups_cover_every_theme_field_once() {
        for (file, theme) in [
            ("theme.json", Theme::default()),
            ("theme-light.json", Theme::light()),
        ] {
            let json: serde_json::Value =
                serde_json::from_str(if file == "theme-light.json" {
                    include_str!("../assets/theme-light.json")
                } else {
                    include_str!("../assets/theme.json")
                })
                .unwrap();
            let mut remaining: std::collections::BTreeSet<String> =
                json.as_object().unwrap().keys().cloned().collect();
            let mut seen = std::collections::BTreeSet::new();
            for group in palette_groups(&theme) {
                for entry in group.entries {
                    assert!(seen.insert(entry.key), "{file}: duplicate palette key {}", entry.key);
                    remaining.remove(entry.key);
                }
            }
            assert!(
                remaining.is_empty(),
                "{file} keys missing from settings palette: {remaining:?}"
            );
        }
    }

    #[test]
    fn backdrop_blur_maps_radius_to_sample_scale() {
        let mut theme = Theme::default();
        assert!((theme.backdrop_blur_scale() - 0.75).abs() < f32::EPSILON);
        theme.backdrop_blur = 0.0;
        assert_eq!(theme.backdrop_blur_scale(), 0.0);
        theme.backdrop_blur = 24.0;
        assert_eq!(theme.backdrop_blur_scale(), 1.0);
        theme.backdrop_blur = 1.0;
        assert_eq!(theme.backdrop_blur_levels(), 1);
        theme.backdrop_blur = 18.0;
        assert_eq!(theme.backdrop_blur_levels(), 5);
    }

    #[test]
    fn built_in_light_theme_is_distinct() {
        assert_ne!(Theme::light(), Theme::default());
        assert!(Theme::light().text[0] < Theme::default().text[0]);
        assert!(Theme::forest().accent[1] > Theme::forest().accent[0]);
        assert!(Theme::punk().accent[0] > Theme::punk().accent[1]);
        let themes = available_themes();
        assert!(themes.iter().any(|preset| preset.id == "light"));
        assert!(themes.iter().any(|preset| preset.id == "forest"));
        assert!(themes.iter().any(|preset| preset.id == "punk"));
        assert!(themes.iter().any(|preset| preset.id == "default"));
    }
}

/// Scrollbar style for general application UI.
///
/// Solid bars with a comfortable hit target; intentionally not applied globally
/// because on-canvas surfaces use the thin variant.
pub fn scroll_style_general() -> egui::style::ScrollStyle {
    let mut style = egui::style::ScrollStyle::solid();
    style.bar_width = 8.0;
    style.bar_inner_margin = 6.0;
    style.bar_outer_margin = 2.0;
    style.handle_min_length = 28.0;
    style
}

/// Thin floating scrollbar for inspector and node-canvas-adjacent surfaces.
///
/// The inspector's existing custom scrollbar remains untouched; this is the
/// shared style for future on-canvas scroll areas.
#[cfg(test)]
mod hex_color_tests {
    use super::Theme;

    #[test]
    fn cable_alpha_is_shared_between_normal_and_selected_states() {
        let mut theme = Theme::default();
        let mut cable = theme.cable;
        cable[3] = 128;
        assert!(theme.set_color("cable", cable));
        assert_eq!(theme.cable_selected[3], 128);

        let mut selected_outline = theme.cable_selected_outline;
        selected_outline[3] = 64;
        assert!(theme.set_color("cable_selected_outline", selected_outline));
        assert_eq!(theme.cable_outline[3], 64);
    }

    #[test]
    fn parses_six_digit_hex_and_preserves_alpha() {
        assert_eq!(Theme::parse_hex("#ff8800", 0x80), Some([255, 136, 0, 0x80]));
        assert_eq!(Theme::parse_hex("00ff00", 0x20), Some([0, 255, 0, 0x20]));
    }

    #[test]
    fn parses_short_and_alpha_hex() {
        assert_eq!(Theme::parse_hex("#f80", 0xff), Some([255, 136, 0, 0xff]));
        assert_eq!(Theme::parse_hex("#f80a", 0xff), Some([255, 136, 0, 0xaa]));
        assert_eq!(Theme::parse_hex("#ff880080", 0xff), Some([255, 136, 0, 0x80]));
    }

    #[test]
    fn rejects_bad_hex() {
        assert_eq!(Theme::parse_hex("#xyz", 0xff), None);
        assert_eq!(Theme::parse_hex("#12345", 0xff), None);
        assert_eq!(Theme::parse_hex("", 0xff), None);
    }
}
