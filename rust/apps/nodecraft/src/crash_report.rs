//! Local crash dialog.
//!
//! Panics are captured at the application boundary and shown as a native error
//! dialog. Nothing is sent anywhere; the report exists only for the user and
//! the local terminal/temp log.

use std::sync::{Mutex, OnceLock};

static LAST: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn last() -> &'static Mutex<Option<String>> {
    LAST.get_or_init(|| Mutex::new(None))
}

/// Install the panic hook: preserve the normal stderr report, but retain the
/// same text for the dialog shown after the event loop unwinds.
pub fn install_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|location| format!("{}:{}", location.file(), location.line()))
            .unwrap_or_else(|| "unknown location".to_owned());
        let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_owned()
        };
        let backtrace = std::backtrace::Backtrace::force_capture();
        let report = format!("NodeCraft crashed\n\n{message}\n\nat {location}\n\n{backtrace}");
        if let Ok(mut slot) = last().lock() {
            *slot = Some(report.clone());
        }
        let _ = std::fs::write(std::env::temp_dir().join("nodecraft-crash.log"), &report);
        default(info);
    }));
}

/// Take the last captured panic, if any.
pub fn take() -> Option<String> {
    last().lock().ok().and_then(|mut slot| slot.take())
}

/// A short summary for the native alert. The full backtrace stays in the log:
/// a system alert has no scroll area, so it must not try to show the whole
/// report and grow past the display.
#[must_use]
pub fn summary(report: &str) -> String {
    let mut sections = report.split("\n\n");
    let _title = sections.next();
    let message = sections.next().unwrap_or("NodeCraft crashed").trim();
    let location = sections.next().unwrap_or("").trim();
    let message = truncate(message, 700);
    let location = truncate(location, 200);
    format!(
        "{message}\n\n{location}\n\nFull details: {}",
        std::env::temp_dir().join("nodecraft-crash.log").display()
    )
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_owned()
    } else {
        format!("{}…", text.chars().take(limit).collect::<String>())
    }
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

/// Handle a panic that was caught inside the frame boundary. This is used
/// because winit callbacks cannot unwind past their C boundary to `main`.
pub fn handle_payload(payload: Box<dyn std::any::Any + Send>) -> ! {
    let _ = payload;
    let report = take().unwrap_or_else(|| "panic without a captured report".to_owned());
    show(&report);
    std::process::exit(101);
}

/// Run `body`, catching a panic so the process can show a dialog instead of
/// disappearing. Returns the body's result, or `None` after showing the crash.
pub fn catch<T>(body: impl FnOnce() -> T + std::panic::UnwindSafe) -> Option<T> {
    match std::panic::catch_unwind(body) {
        Ok(value) => Some(value),
        Err(_) => {
            let report = take().unwrap_or_else(|| "panic without a captured report".to_owned());
            show(&report);
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_bounded_and_keeps_the_log_path() {
        let long = format!("NodeCraft crashed\n\n{}\n\nat src/lib.rs:1\n\n{}", "x".repeat(5000), "y".repeat(5000));
        let summary = summary(&long);
        assert!(summary.chars().count() < 1_200);
        assert!(summary.contains("Full details:"));
        assert!(summary.contains("src/lib.rs:1"));
    }
}
