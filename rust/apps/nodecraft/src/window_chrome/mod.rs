//! Platform window-chrome policy with one entry point set per OS.

/// The secondary native windows that share the auxiliary chrome policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AuxiliaryWindow {
    Settings,
    Debug,
}

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::*;

#[cfg(not(target_os = "macos"))]
mod fallback;
#[cfg(not(target_os = "macos"))]
pub(crate) use fallback::*;
