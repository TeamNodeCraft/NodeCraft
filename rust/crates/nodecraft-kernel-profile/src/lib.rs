//! Central kernel construction for NodeCraft.
//!
//! SolidCore ships new modelling fixes behind `Kernel::fixes` flags. Keeping every
//! kernel construction here means the evaluator, previews, STEP import and display
//! paths all see the same kernel profile, instead of each `Kernel::default()` site
//! silently choosing a different one.

use solidcore::kernel::{Kernel, KernelOptions};
use std::sync::atomic::{AtomicBool, Ordering};

static EXPERIMENTAL_FIXES: AtomicBool = AtomicBool::new(true);

/// The upstream fix names NodeCraft enables as one profile. The same names are
/// passed to the `solidcore-ops` session so kernel and ops work agree.
pub const EXPERIMENTAL_FIX_NAMES: [&str; 3] = [
    // B-rep point classification honours the current face trims while editing
    // (WR-732 / #2503).
    "brep_ray_classifies_current_trims",
    // Curved-face blend corners and their certified setback fills (#2343).
    "fillet_setback_beside_curved_faces",
    // Chamfers meeting at a vertex are closed by their own corner event (#2345).
    "fillet_chamfer_corner",
];

/// Configure the opt-in SolidCore fixes applied to newly created kernels.
pub fn configure_experimental_fixes(enabled: bool) {
    EXPERIMENTAL_FIXES.store(enabled, Ordering::Relaxed);
}

#[must_use]
pub fn experimental_fixes_enabled() -> bool {
    EXPERIMENTAL_FIXES.load(Ordering::Relaxed)
}

/// Apply NodeCraft's SolidCore fix profile to an existing kernel.
pub fn apply_fixes(kernel: &mut Kernel) {
    if !experimental_fixes_enabled() {
        return;
    }
    for name in EXPERIMENTAL_FIX_NAMES {
        if let Some(flag) = kernel.fixes.flag_mut(name) {
            *flag = true;
        }
    }
}

/// A NodeCraft kernel with the standard fix profile.
#[must_use]
pub fn kernel() -> Kernel {
    let mut kernel = Kernel::new(&KernelOptions::default());
    apply_fixes(&mut kernel);
    kernel
}

/// A NodeCraft kernel with explicit options and the standard fix profile.
#[must_use]
pub fn kernel_with_options(options: &KernelOptions) -> Kernel {
    let mut kernel = Kernel::new(options);
    apply_fixes(&mut kernel);
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_profile_enables_the_new_kernel_fixes() {
        let mut kernel = kernel();
        assert!(kernel.fixes.brep_ray_classifies_current_trims);
        assert!(kernel.fixes.fillet_setback_beside_curved_faces);
        assert!(kernel.fixes.fillet_chamfer_corner);
        for name in EXPERIMENTAL_FIX_NAMES {
            assert!(kernel.fixes.flag_mut(name).is_some(), "unknown fix {name}");
        }
    }
}
