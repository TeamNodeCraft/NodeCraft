//! Platform window-chrome policy with one entry point set per OS.

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub(crate) use macos::*;

#[cfg(not(target_os = "macos"))]
mod fallback;
#[cfg(not(target_os = "macos"))]
pub(crate) use fallback::*;
