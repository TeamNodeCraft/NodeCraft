//! Process-wide thread budget for pools we do not own directly.

/// Half the available cores, at least two: display/kernel-nested work should
/// not consume every core alongside the exact kernel pool.
pub fn rayon_thread_budget() -> usize {
    std::thread::available_parallelism()
        .map(|threads| (threads.get() / 2).max(2))
        .unwrap_or(2)
}

/// Configure SolidCore's implicit rayon pool once, before kernel work.
pub fn init_rayon_pool() -> Result<(), String> {
    let threads = rayon_thread_budget();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .stack_size(nodecraft_kernel_profile::KERNEL_THREAD_STACK)
        .thread_name(|index| format!("solidcore-par-{index}"))
        .build_global()
        .map_err(|error| format!("rayon pool already configured: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_at_least_two() {
        assert!(rayon_thread_budget() >= 2);
    }

    #[test]
    fn local_pool_matches_budget() {
        let budget = rayon_thread_budget();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(budget)
            .build()
            .expect("local pool");
        assert_eq!(pool.current_num_threads(), budget);
    }
}
