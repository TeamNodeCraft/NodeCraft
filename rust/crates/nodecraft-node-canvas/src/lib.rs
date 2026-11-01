//! Reusable NodeCraft node canvas.
//!
//! The canvas owns view interaction, selection, z-order and retained geometry.
//! The application owns the session, evaluation, file handling and 3D viewport.

pub mod canvas;
pub mod gesture;
pub mod routing;
pub mod squircle;
pub mod theme;

pub mod gpu;
pub use gpu::install as install_gpu;
