//! The SolidCore kernel seam and error mapping.
//!
//! STEP and CAD IO is not here: `nodecraft-step` owns the cadmpeg/CADIR adapter.

pub mod error;
pub mod kernel;
pub use solidcore;
