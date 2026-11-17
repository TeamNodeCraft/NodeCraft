//! Local crash dialog and the app-facing panic boundary.
//!
//! The process-wide report store and the render-callback guard live in
//! `nodecraft_crash` so the app and the node canvas share one implementation.
//! This module owns the native dialog and re-exports the shared API, so callers
//! keep writing `crash_report::guard_render` / `crash_report::handle_payload`.

pub use nodecraft_crash::{
    Origin, begin_event, catch, guard_render, handle_payload, summary,
};

/// Install the shared panic hook and the native dialog used by the fatal path.
pub fn install_hook() {
    nodecraft_crash::install_hook();
    nodecraft_crash::install_fatal_dialog(show);
}

/// Show the local crash dialog. No report is sent or uploaded.
pub fn show(report: &str) {
    let _ = rfd::MessageDialog::new()
        .set_title("NodeCraft crashed")
        .set_description(summary(report))
        .set_level(rfd::MessageLevel::Error)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_crash::{HandleOutcome, record, take};

    #[test]
    fn summary_is_bounded_and_keeps_the_log_path() {
        let long = format!("NodeCraft crashed\n\n{}\n\nat src/lib.rs:1\n\n{}", "x".repeat(5000), "y".repeat(5000));
        let summary = summary(&long);
        assert!(summary.chars().count() < 1_200);
        assert!(summary.contains("Full details:"));
        assert!(summary.contains("src/lib.rs:1"));
    }

    #[test]
    fn first_panic_report_survives_the_cannot_unwind_follow_up() {
        begin_event();
        let original = "NodeCraft crashed\n\nwgpu error: Validation Error\n\nat gpu.rs:925";
        let follow_up = "NodeCraft crashed\n\npanic in a function that cannot unwind\n\nat panicking.rs:225";
        assert!(record(original));
        assert!(!record(follow_up));
        assert_eq!(take().as_deref(), Some(original));
    }

    #[test]
    fn per_event_reset_replaces_the_previous_report() {
        begin_event();
        assert!(record("NodeCraft crashed\n\nworker panic A"));
        begin_event();
        assert!(record("NodeCraft crashed\n\nrender panic B"));
        assert_eq!(
            take().as_deref(),
            Some("NodeCraft crashed\n\nrender panic B")
        );
    }

    #[test]
    fn render_origin_is_recovered_while_event_origin_stays_fatal() {
        begin_event();
        let outcome = handle_payload(Box::new("render boom"), Origin::Render);
        assert!(
            matches!(outcome, HandleOutcome::Recovered { .. }),
            "Origin::Render must not exit"
        );
        assert!(
            take().unwrap_or_default().contains("render boom"),
            "recovered render panic must remain recorded"
        );
        // The fatal path is classified, not executed: calling it would exit 101.
        assert!(Origin::Event.is_fatal());
        assert!(!Origin::Render.is_fatal());
    }
}
