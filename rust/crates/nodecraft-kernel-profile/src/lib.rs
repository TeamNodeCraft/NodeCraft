//! Central kernel construction for NodeCraft.
//!
//! SolidCore ships new modelling fixes behind `Kernel::fixes` flags. Keeping every
//! kernel construction here means the evaluator, previews, STEP import and display
//! paths all see the same kernel profile, instead of each `Kernel::default()` site
//! silently choosing a different one.

use solidcore::kernel::{Kernel, KernelOptions, UpstreamFixes};
use std::sync::atomic::{AtomicBool, Ordering};

/// Stack for a thread that runs kernel work.
///
/// A fillet nests deeply enough to overflow a thread's default stack (512 KiB on macOS,
/// 1 MiB on Windows); SolidCraft raises its app and CLI main stacks to 8 MiB for the same
/// reason (#15711). Every NodeCraft thread that runs kernel operations uses this size.
pub const KERNEL_THREAD_STACK: usize = 8 * 1024 * 1024;

static EXPERIMENTAL_FIXES: AtomicBool = AtomicBool::new(true);
static PROFILE_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// SolidCraft's profile: every opt-in upstream fix on. SolidCraft's `smlib` kernels run
/// `UpstreamFixes::all()` since it dropped its held-back list (SolidCraft #18937, in the pin).
#[must_use]
pub fn fixes() -> UpstreamFixes {
    UpstreamFixes::all()
}

/// Names that are on in [`fixes`], for the `solidcore-ops` session.
#[must_use]
pub fn fix_names() -> Vec<&'static str> {
    UpstreamFixes::NAMES.to_vec()
}

/// The fillet solver's finite-difference step for the whole process.
///
/// Upstream latches it once per process (a function static in `SmFilletSolver`): the first
/// fillet point evaluation settles it to `1e-3` or `1e-5`, and every later one takes it. Every
/// kernel made here shares this one cell ([`Kernel::process_fillet_point_step`]), as
/// `solidcore-ops`' session kernels and SolidCraft's share theirs, so a fillet's result does not
/// depend on which pool kernel builds it or what that kernel built first.
static FILLET_POINT_STEP: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

/// Configure the opt-in SolidCore fixes applied to newly created kernels.
pub fn configure_experimental_fixes(enabled: bool) {
    if EXPERIMENTAL_FIXES.swap(enabled, Ordering::Relaxed) != enabled {
        PROFILE_GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// Changes whenever the fix profile changes, so persisted caches can include it
/// in their keys and never reuse results from another profile.
#[must_use]
pub fn profile_generation() -> u64 {
    PROFILE_GENERATION.load(Ordering::Relaxed)
}

#[must_use]
pub fn experimental_fixes_enabled() -> bool {
    EXPERIMENTAL_FIXES.load(Ordering::Relaxed)
}

/// Apply NodeCraft's SolidCore fix profile to an existing kernel.
pub fn apply_fixes(kernel: &mut Kernel) {
    let enabled = experimental_fixes_enabled()
        && std::env::var("NODECRAFT_KERNEL_FIXES").map_or(true, |value| value != "0");
    kernel.fixes = if enabled { fixes() } else { UpstreamFixes::default() };
}

/// A NodeCraft kernel with the standard fix profile.
#[must_use]
pub fn kernel() -> Kernel {
    kernel_with_options(&KernelOptions::default())
}

/// A NodeCraft kernel with explicit options and the standard fix profile.
#[must_use]
pub fn kernel_with_options(options: &KernelOptions) -> Kernel {
    let mut kernel = Kernel::new(options);
    apply_fixes(&mut kernel);
    kernel.process_fillet_point_step = Some(&FILLET_POINT_STEP);
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fix_is_on() {
        let off: Vec<&str> = fixes().flags().filter(|(_, on)| !on).map(|(name, _)| name).collect();
        assert!(off.is_empty(), "UpstreamFixes flags are off: {off:?}");
        assert_eq!(fix_names(), UpstreamFixes::NAMES);
    }

    #[test]
    fn profile_off_is_pinned_parity() {
        let mut kernel = Kernel::new(&KernelOptions::default());
        kernel.fixes = UpstreamFixes::default();
        assert!(!kernel.fixes.brep_ray_classifies_current_trims);
    }

    /// The first latch on any NodeCraft kernel settles the fillet step for every other one,
    /// whichever step the other's first fillet point would have chosen.
    #[test]
    fn every_kernel_shares_the_process_fillet_step() {
        let first = kernel();
        let second = kernel_with_options(&KernelOptions::default());
        let settled = first.latch_fillet_point_step(1.0e-3);
        assert_eq!(second.latch_fillet_point_step(1.0e-5), settled);
        assert_eq!(kernel().latched_fillet_point_step(), Some(settled));
    }
}
