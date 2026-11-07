//! Native file dialogs that never block the winit event handler.
//!
//! `rfd`'s synchronous dialogs run an AppKit modal loop. Calling them from
//! `App::logic`/`App::ui` (which eframe runs inside winit's event handler) nests
//! that modal loop inside `handle_event`. Any window event raised by opening the
//! panel is then queued and can be delivered by the nested run loop while the
//! handler is still borrowed; winit's `EventHandler` treats that as re-entrancy
//! and panics with "tried to handle event while another event is currently being
//! handled". Because the panic unwinds through an Objective-C block, it aborts
//! the process instead of unwinding (macOS crash: SIGABRT, `panic_cannot_unwind`).
//!
//! Running the dialog on a worker thread avoids the nesting entirely: on macOS
//! `rfd` dispatches the panel to the main queue, so the modal loop starts between
//! event-loop turns, after winit's event handler has returned. The UI polls the
//! result and performs the action from a normal frame.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Which native dialog to show. File filters mirror the call sites.
pub enum FileDialogKind {
    OpenDocument,
    SaveDocument { suggested_name: String },
    OpenStepFile,
    ChooseDirectory,
}

pub enum FileDialogResult {
    Chosen(PathBuf),
    Cancelled,
}

/// A dialog running off the event thread; poll [`Self::try_finish`] once a frame.
pub struct PendingFileDialog {
    receiver: Receiver<FileDialogResult>,
}

impl PendingFileDialog {
    pub fn spawn(kind: FileDialogKind) -> Self {
        let (sender, receiver) = channel();
        std::thread::spawn(move || {
            // rfd's macOS backend dispatches this to the main queue and blocks
            // the worker until the panel closes.
            let result = show(kind);
            let _ = sender.send(result);
        });
        Self { receiver }
    }

    /// `None` while the dialog is open; `Some` on the frame it closes.
    pub fn try_finish(&mut self) -> Option<FileDialogResult> {
        match self.receiver.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            // The worker died without reporting (e.g. the process is shutting
            // down). Treat it as a cancel rather than blocking the UI forever.
            Err(TryRecvError::Disconnected) => Some(FileDialogResult::Cancelled),
        }
    }
}

#[cfg(test)]
impl PendingFileDialog {
    fn from_receiver(receiver: Receiver<FileDialogResult>) -> Self {
        Self { receiver }
    }
}

fn show(kind: FileDialogKind) -> FileDialogResult {
    let path = match kind {
        FileDialogKind::OpenDocument => rfd::FileDialog::new()
            .add_filter("NodeCraft graph", &["nodecraft"])
            .pick_file(),
        FileDialogKind::SaveDocument { suggested_name } => rfd::FileDialog::new()
            .add_filter("NodeCraft graph", &["nodecraft"])
            .set_file_name(suggested_name)
            .save_file(),
        FileDialogKind::OpenStepFile => rfd::FileDialog::new()
            .add_filter("STEP model", &["step", "stp"])
            .pick_file(),
        FileDialogKind::ChooseDirectory => rfd::FileDialog::new().pick_folder(),
    };
    match path {
        Some(path) => FileDialogResult::Chosen(path),
        None => FileDialogResult::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn polling_reports_pending_then_the_dialog_result() {
        let (sender, receiver) = channel();
        let mut pending = PendingFileDialog::from_receiver(receiver);
        assert!(pending.try_finish().is_none(), "an open dialog must not block");
        sender
            .send(FileDialogResult::Chosen(PathBuf::from("/tmp/saved.nodecraft")))
            .expect("receiver alive");
        match pending.try_finish() {
            Some(FileDialogResult::Chosen(path)) => {
                assert_eq!(path, PathBuf::from("/tmp/saved.nodecraft"));
            }
            _ => panic!("expected the chosen path"),
        }
        // A worker that dies without reporting is a cancel, not a hang.
        drop(sender);
        assert!(matches!(pending.try_finish(), Some(FileDialogResult::Cancelled)));
    }
}
