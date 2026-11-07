//! Re-export the shared kernel profile.
//!
//! The implementation lives in `nodecraft-kernel-profile` so low-level crates
//! (mesh, STEP IO) can construct the same profiled kernel without depending on
//! `nodecraft-solidcore` and creating a cycle.

pub use nodecraft_kernel_profile::*;
