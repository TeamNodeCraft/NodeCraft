//! Application settings persistence and startup document selection.

use super::*;

pub(super) fn default_document() -> Option<PathBuf> {
    let candidate = PathBuf::from("samples/step-import.nodecraft");
    candidate.is_file().then_some(candidate)
}

pub(super) fn choose_startup_document(
    explicit: Option<PathBuf>,
    last: Option<PathBuf>,
    fallback: Option<PathBuf>,
) -> Option<PathBuf> {
    explicit.or(last).or(fallback)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConfigPlatform {
    MacOs,
    Windows,
    Unix,
}

/// The platform config root, as a pure function so path behavior can be tested
/// for every target without running on it.
pub(super) fn config_dir_for(
    platform: ConfigPlatform,
    home: Option<&std::ffi::OsStr>,
    appdata: Option<&std::ffi::OsStr>,
    userprofile: Option<&std::ffi::OsStr>,
    xdg_config_home: Option<&std::ffi::OsStr>,
) -> Option<PathBuf> {
    match platform {
        ConfigPlatform::MacOs => home
            .map(|home| PathBuf::from(home).join("Library/Application Support/NodeCraft")),
        ConfigPlatform::Windows => appdata
            .map(|path| PathBuf::from(path).join("NodeCraft"))
            .or_else(|| {
                userprofile.map(|path| {
                    PathBuf::from(path)
                        .join("AppData")
                        .join("Roaming")
                        .join("NodeCraft")
                })
            }),
        ConfigPlatform::Unix => xdg_config_home
            .map(|path| PathBuf::from(path).join("nodecraft"))
            .or_else(|| home.map(|home| PathBuf::from(home).join(".config/nodecraft"))),
    }
}

pub(super) fn current_config_platform() -> ConfigPlatform {
    if cfg!(target_os = "macos") {
        ConfigPlatform::MacOs
    } else if cfg!(windows) {
        ConfigPlatform::Windows
    } else {
        ConfigPlatform::Unix
    }
}

pub(super) fn settings_file() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("NODECRAFT_CONFIG_DIR") {
        return Some(PathBuf::from(dir).join("settings.json"));
    }
    let home = std::env::var_os("HOME");
    let appdata = std::env::var_os("APPDATA");
    let userprofile = std::env::var_os("USERPROFILE");
    let xdg = std::env::var_os("XDG_CONFIG_HOME");
    Some(
        config_dir_for(
            current_config_platform(),
            home.as_deref(),
            appdata.as_deref(),
            userprofile.as_deref(),
            xdg.as_deref(),
        )?
        .join("settings.json"),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct ActivitySettings {
    pub(super) fade_in_ms: u32,
    pub(super) hold_ms: u32,
    pub(super) fade_out_ms: u32,
}

impl Default for ActivitySettings {
    fn default() -> Self {
        Self { fade_in_ms: 40, hold_ms: 40, fade_out_ms: 40 }
    }
}

#[derive(Default, serde::Serialize, serde::Deserialize)]
pub(super) struct AppSettings {
    #[serde(flatten)]
    pub(super) view: crate::viewport::ViewSettings,
    #[serde(default)]
    pub(super) theme: Option<String>,
    #[serde(default)]
    pub(super) activity: ActivitySettings,
    #[serde(default = "default_feature_overlay")]
    pub(super) feature_overlay: bool,
    #[serde(default)]
    pub(super) engine_threads: usize,
    #[serde(default, alias = "mesher_threads")]
    pub(super) tessellation_threads: usize,
    #[serde(default = "default_workload_profiler")]
    pub(super) workload_profiler: bool,
    #[serde(default = "default_kernel_fixes")]
    pub(super) kernel_fixes: bool,
    /// Cable routing style: "j-routing" or "bezier-routing".
    #[serde(default = "default_cable_routing")]
    pub(super) cable_routing: String,
    #[serde(default)]
    pub(super) tool_dock_offset_y: f32,
    #[serde(default = "default_tool_dock_shortcut")]
    pub(super) tool_dock_shortcut: String,
    #[serde(default = "default_tool_search_shortcut")]
    pub(super) tool_search_shortcut: String,
    #[serde(default)]
    pub(super) recent_files: Vec<String>,
}

pub(super) fn default_tool_dock_shortcut() -> String {
    "Cmd+M".to_owned()
}

pub(super) fn default_tool_search_shortcut() -> String {
    "Cmd+F".to_owned()
}

pub(super) fn default_cable_routing() -> String {
    CableRouting::J.setting_id().to_owned()
}

pub(super) fn default_workload_profiler() -> bool {
    true
}

pub(super) fn default_kernel_fixes() -> bool {
    true
}

pub(super) fn default_feature_overlay() -> bool {
    true
}

pub(super) fn load_app_settings() -> Option<AppSettings> {
    let text = std::fs::read_to_string(settings_file()?).ok()?;
    serde_json::from_str(&text).ok()
}

pub(super) fn save_app_settings(
    view: crate::viewport::ViewSettings,
    theme: Option<String>,
    activity: ActivitySettings,
    feature_overlay: bool,
    engine_threads: usize,
    tessellation_threads: usize,
    workload_profiler: bool,
    kernel_fixes: bool,
    cable_routing: CableRouting,
    tool_dock_offset_y: f32,
    tool_dock_shortcut: &str,
    tool_search_shortcut: &str,
    recent_files: &[PathBuf],
) -> Result<(), ()> {
    let path = settings_file().ok_or(())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|_| ())?;
    }
    let settings = AppSettings {
        view,
        theme,
        activity,
        feature_overlay,
        engine_threads,
        tessellation_threads,
        workload_profiler,
        kernel_fixes,
        cable_routing: cable_routing.setting_id().to_owned(),
        tool_dock_offset_y,
        tool_dock_shortcut: tool_dock_shortcut.to_owned(),
        tool_search_shortcut: tool_search_shortcut.to_owned(),
        recent_files: recent_files
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };
    let text = serde_json::to_string_pretty(&settings).map_err(|_| ())?;
    let temp = path.with_extension("json.tmp");
    std::fs::write(&temp, text).map_err(|_| ())?;
    std::fs::rename(&temp, &path).map_err(|_| ())?;
    Ok(())
}

/// Apply the saved theme id. Missing, moved or invalid themes intentionally fall
/// back to the embedded default dark palette.
pub(super) fn apply_saved_theme(id: Option<&str>) {
    let mut preset = id.and_then(|id| {
        crate::theme::available_themes()
            .into_iter()
            .find(|preset| preset.id == id)
    });
    if preset.is_none()
        && let Some(id) = id
        && id != "default"
        && id != "light"
        && id != "forest"
        && std::path::Path::new(id).is_file()
    {
        preset = Some(crate::theme::ThemePreset {
            name: id.to_owned(),
            id: id.to_owned(),
            path: Some(PathBuf::from(id)),
        });
    }
    let loaded = preset
        .as_ref()
        .is_some_and(|preset| crate::theme::load_preset(preset).is_ok());
    if !loaded {
        let default = crate::theme::ThemePreset {
            name: crate::theme::DEFAULT_THEME_NAME.to_owned(),
            id: "default".to_owned(),
            path: None,
        };
        let _ = crate::theme::load_preset(&default);
    }
}
