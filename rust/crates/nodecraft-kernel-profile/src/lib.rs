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

/// SolidCraft's held-back fixes at the pinned SolidCore revision. Keep these in
/// sync with `crates/smlib/src/core.rs` in SolidCraft on every pin bump.
pub const HELD_BACK: [&str; 2] = [
    // WR-707, solidcore#2311: sound geometry refused. With it on, the fillet rig's
    // 1668-fillet-both-sides-of-thin-wall-r7 is refused as breaking through the wall
    // ("largest radius that fits is 6.828 mm") where with it off it builds the closed form.
    "fillet_refuse_rounds_breaking_through",
    // WR-730, solidcore#2541: volumes wrong against their closed forms, and rounds too big
    // for their faces built instead of refused (fillet_over_a_cliff 721.832 vs 693.205, …).
    "fillet_round_ends_on_outline",
];

/// SolidCraft's profile: every opt-in upstream fix on, except [`HELD_BACK`].
#[must_use]
pub fn fixes() -> UpstreamFixes {
    let mut fixes = UpstreamFixes::all();
    for name in HELD_BACK {
        if let Some(flag) = fixes.flag_mut(name) {
            *flag = false;
        }
    }
    fixes
}

/// Names that are on in [`fixes`], for the `solidcore-ops` session.
#[must_use]
pub fn fix_names() -> Vec<&'static str> {
    UpstreamFixes::NAMES.iter().copied().filter(|name| !HELD_BACK.contains(name)).collect()
}

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
    fn every_fix_is_on_but_the_held_back() {
        let fixes = fixes();
        let flags: Vec<(&str, bool)> = fixes.flags().collect();
        let off: Vec<&str> = flags
            .iter()
            .filter(|(name, on)| !on && !HELD_BACK.contains(name))
            .map(|(name, _)| *name)
            .collect();
        assert!(off.is_empty(), "UpstreamFixes flags are off: {off:?}");
        for name in HELD_BACK {
            assert!(flags.contains(&(name, false)), "{name} is on or gone: drop it from HELD_BACK");
        }
    }

    #[test]
    fn profile_off_is_pinned_parity() {
        let mut kernel = Kernel::new(&KernelOptions::default());
        kernel.fixes = UpstreamFixes::default();
        assert!(!kernel.fixes.brep_ray_classifies_current_trims);
        assert_eq!(fix_names().len(), UpstreamFixes::NAMES.len() - HELD_BACK.len());
    }
}
