//! Shared crash-report store and render-callback panic guard.
//!
//! The UI/event thread and the wgpu render thread can both panic. Each thread
//! keeps its own per-event report slot, so a render callback starting a new
//! event cannot erase a fatal report the UI thread has not read yet (and a
//! recovered worker-thread panic cannot mask a later UI-thread panic). The set
//! of failed render units is process-wide: once a callback panics, every later
//! callback for that named unit is skipped instead of re-running the broken
//! body and re-panicking every frame.

use std::any::Any;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Where a caught panic came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// A `logic`/`ui`/main event boundary. Fatal: show the report and exit.
    Event,
    /// A render callback. Recovered: log/record it and keep the process alive.
    Render,
}

impl Origin {
    /// Whether this origin ends the process after reporting. The fatal branch
    /// is the only one that calls `process::exit`, so tests can assert the
    /// classification without triggering the exit.
    #[must_use]
    pub fn is_fatal(self) -> bool {
        matches!(self, Self::Event)
    }
}

/// Result of [`handle_payload`]. `Fatal` is never returned because the fatal
/// path calls `process::exit(101)`; it exists to make that contract explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleOutcome {
    Recovered { report: String },
    Fatal,
}

thread_local! {
    /// One report per event on this thread. [`begin_event`] clears it;
    /// [`record`] is first-wins within the event.
    static EVENT_REPORT: RefCell<Option<String>> = RefCell::new(None);
}

static FAILED_RENDER_UNITS: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
static FATAL_DIALOG: OnceLock<fn(&str)> = OnceLock::new();

fn failed_render_units() -> &'static Mutex<BTreeMap<String, String>> {
    FAILED_RENDER_UNITS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn lock_failed_render_units() -> std::sync::MutexGuard<'static, BTreeMap<String, String>> {
    match failed_render_units().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn crash_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("nodecraft-crash.log")
}

/// Start a new crash-report event on this thread, clearing the report left by
/// a previous event. UI/event boundaries call this before their `catch_unwind`;
/// [`guard_render`] calls it for each render callback.
pub fn begin_event() {
    EVENT_REPORT.with(|slot| *slot.borrow_mut() = None);
}

/// Alias for [`begin_event`] for call sites that describe the action as a reset.
pub fn reset_event() {
    begin_event();
}

/// Record `report` for this thread's current event. Returns `true` if stored.
///
/// First report wins within one event. That matters for the panic hook: a panic
/// raised inside an AppKit/winit callback cannot unwind, so Rust reports a
/// second `panic in a function that cannot unwind`. The follow-up must not
/// replace the original message that caused the abort.
///
/// The temp log is written by [`install_hook`], not here, so tests and callers
/// that stage a report cannot overwrite the user's last real crash log.
pub fn record(report: &str) -> bool {
    EVENT_REPORT.with(|slot| keep_first(&mut slot.borrow_mut(), report))
}

fn keep_first(slot: &mut Option<String>, report: &str) -> bool {
    if slot.is_some() {
        return false;
    }
    *slot = Some(report.to_owned());
    true
}

/// Take the current thread's captured panic report, if any.
pub fn take() -> Option<String> {
    EVENT_REPORT.with(|slot| slot.borrow_mut().take())
}

/// Install the panic hook: preserve the normal stderr report, retain the report
/// for this thread's current event, and write the temp crash log once per event.
pub fn install_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let report = report_from_info(info);
        if record(&report) {
            let _ = std::fs::write(crash_log_path(), &report);
        }
        default(info);
    }));
}

/// Install the native dialog used by the fatal event path. The shared crate is
/// dependency-free, so the app supplies the `rfd` dialog.
pub fn install_fatal_dialog(show: fn(&str)) {
    let _ = FATAL_DIALOG.set(show);
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
    format!("{message}\n\n{location}\n\nFull details: {}", crash_log_path().display())
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        text.to_owned()
    } else {
        format!("{}…", text.chars().take(limit).collect::<String>())
    }
}

/// Handle a panic that was caught at a boundary.
///
/// `Origin::Event` keeps the old fatal behaviour: show the local dialog (when
/// the app installed one) and exit 101. `Origin::Render` records/logs the
/// panic and returns a recovered result, so a render callback panic disables
/// drawing instead of terminating the process.
pub fn handle_payload(payload: Box<dyn Any + Send>, origin: Origin) -> HandleOutcome {
    let report = take().unwrap_or_else(|| report_from_payload(&*payload));
    // Re-record so the report stays available for this thread's event; callers
    // may read it after a recovered render panic. The panic hook owns the temp
    // log write.
    record(&report);
    if origin.is_fatal() {
        if let Some(show) = FATAL_DIALOG.get() {
            show(&report);
        }
        std::process::exit(101);
    }
    eprintln!(
        "NodeCraft: recovered from a render panic; the process continues.\n{}",
        summary(&report)
    );
    HandleOutcome::Recovered { report }
}

/// Run `body`, catching a panic so the process can show a dialog instead of
/// disappearing. Returns the body's result, or `None` after showing the crash.
pub fn catch<T>(body: impl FnOnce() -> T) -> Option<T> {
    begin_event();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => Some(value),
        Err(payload) => {
            let report = take().unwrap_or_else(|| report_from_payload(&*payload));
            record(&report);
            if let Some(show) = FATAL_DIALOG.get() {
                show(&report);
            }
            None
        }
    }
}

/// Guard one render callback.
///
/// `unit` is a stable name (`"viewport"`, `"node-canvas"`) rather than a
/// per-frame instance address: a per-frame name would run the broken body and
/// re-panic every frame. The first panic records the report, marks `unit`
/// failed process-wide, logs it once, and returns `None`; every later call for
/// that unit returns `None` without running `body`.
pub fn guard_render<T>(unit: &str, body: impl FnOnce() -> T) -> Option<T> {
    begin_event();
    if render_unit_failed(unit) {
        return None;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(value) => Some(value),
        Err(payload) => {
            // The installed hook normally recorded the report while unwinding.
            // Fall back to the payload so tests and hook-less callers still get
            // a message and a backtrace.
            let captured = take();
            let report = captured.clone().unwrap_or_else(|| report_from_payload(&*payload));
            if captured.is_none() {
                record(&report);
            }
            let first_failure = {
                let mut failed = lock_failed_render_units();
                if failed.contains_key(unit) {
                    false
                } else {
                    failed.insert(unit.to_owned(), report.clone());
                    true
                }
            };
            if first_failure {
                eprintln!(
                    "NodeCraft: render callback {unit:?} panicked; drawing is disabled for this unit for the rest of the process.\n{}",
                    summary(&report)
                );
            }
            None
        }
    }
}

/// Whether `unit` has panicked and been permanently disabled.
#[must_use]
pub fn render_unit_failed(unit: &str) -> bool {
    lock_failed_render_units().contains_key(unit)
}

/// The report captured when `unit` failed, if it has failed.
#[must_use]
pub fn failed_render_report(unit: &str) -> Option<String> {
    lock_failed_render_units().get(unit).cloned()
}

fn report_from_info(info: &std::panic::PanicHookInfo<'_>) -> String {
    let location = info
        .location()
        .map(|location| format!("{}:{}", location.file(), location.line()))
        .unwrap_or_else(|| "unknown location".to_owned());
    build_report(message_from_payload(info.payload()), &location)
}

fn report_from_payload(payload: &(dyn Any + Send)) -> String {
    build_report(message_from_payload(payload), "unknown location")
}

fn build_report(message: String, location: &str) -> String {
    let backtrace = std::backtrace::Backtrace::force_capture();
    format!("NodeCraft crashed\n\n{message}\n\nat {location}\n\n{backtrace}")
}

fn message_from_payload(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn first_panic_report_survives_the_cannot_unwind_follow_up() {
        let original = "NodeCraft crashed\n\nwgpu error: Validation Error\n\nat gpu.rs:925";
        let follow_up =
            "NodeCraft crashed\n\npanic in a function that cannot unwind\n\nat panicking.rs:225";
        let mut slot = None;
        assert!(keep_first(&mut slot, original));
        assert!(!keep_first(&mut slot, follow_up));
        assert_eq!(slot.as_deref(), Some(original));
    }

    #[test]
    fn beginning_a_new_event_replaces_the_previous_report() {
        begin_event();
        assert!(record("NodeCraft crashed\n\nworker panic A"));
        begin_event();
        assert!(record("NodeCraft crashed\n\nrender panic B"));
        assert_eq!(take().as_deref(), Some("NodeCraft crashed\n\nrender panic B"));
    }

    #[test]
    fn a_render_event_does_not_clear_another_threads_report() {
        begin_event();
        assert!(record("NodeCraft crashed\n\nUI fatal panic"));
        std::thread::spawn(|| {
            begin_event();
            assert!(record("NodeCraft crashed\n\nrender panic"));
            assert_eq!(take().as_deref(), Some("NodeCraft crashed\n\nrender panic"));
        })
        .join()
        .expect("render thread");
        assert_eq!(take().as_deref(), Some("NodeCraft crashed\n\nUI fatal panic"));
    }

    #[test]
    fn guard_render_catches_once_and_disables_the_named_unit() {
        let unit = "test-guard-render-unit";
        let calls = AtomicUsize::new(0);
        let first: Option<()> = guard_render(unit, || {
            calls.fetch_add(1, Ordering::SeqCst);
            panic!("injected callback panic");
        });
        assert!(first.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(render_unit_failed(unit));
        let report = failed_render_report(unit).expect("failed unit stores its report");
        assert!(report.contains("injected callback panic"), "{report}");
        assert!(report.contains("NodeCraft crashed"), "{report}");
        assert!(report.contains("at "), "{report}");

        // The body must not run again: a second callback for the same unit is
        // skipped before `catch_unwind` and before the counter increments.
        let second: Option<u32> = guard_render(unit, || {
            calls.fetch_add(1, Ordering::SeqCst);
            7
        });
        assert_eq!(second, None);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn render_origin_is_recovered_while_event_origin_stays_fatal() {
        begin_event();
        let outcome = handle_payload(Box::new("render boom"), Origin::Render);
        match outcome {
            HandleOutcome::Recovered { report } => {
                assert!(report.contains("render boom"), "{report}");
            }
            HandleOutcome::Fatal => panic!("render origin must not be fatal"),
        }
        assert!(
            take().unwrap_or_default().contains("render boom"),
            "recovered render panic must remain recorded"
        );
        // The fatal path is classified, not executed: calling it would exit 101.
        assert!(Origin::Event.is_fatal());
        assert!(!Origin::Render.is_fatal());
    }
}
