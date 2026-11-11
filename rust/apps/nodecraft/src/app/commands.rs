//! Application-level commands shared by menus and keyboard shortcuts.

use egui::KeyboardShortcut;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AppCommand {
    NewDocument,
    OpenDocument,
    SaveDocument,
    SaveDocumentAs,
    CloseTab,
}

impl AppCommand {
    pub(super) const ALL: [Self; 5] = [
        Self::NewDocument,
        Self::OpenDocument,
        Self::SaveDocument,
        Self::SaveDocumentAs,
        Self::CloseTab,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::NewDocument => "New",
            Self::OpenDocument => "Open…",
            Self::SaveDocument => "Save",
            Self::SaveDocumentAs => "Save As…",
            Self::CloseTab => "Close Tab",
        }
    }

    pub(super) fn shortcut_text(self) -> &'static str {
        match self {
            Self::NewDocument => "⌘N",
            Self::OpenDocument => "⌘O",
            Self::SaveDocument => "⌘S",
            Self::SaveDocumentAs => "⇧⌘S",
            Self::CloseTab => "⌘W",
        }
    }

    pub(super) fn shortcut(self) -> KeyboardShortcut {
        let key = match self {
            Self::NewDocument => egui::Key::N,
            Self::OpenDocument => egui::Key::O,
            Self::SaveDocument => egui::Key::S,
            Self::SaveDocumentAs => egui::Key::S,
            Self::CloseTab => egui::Key::W,
        };
        let modifiers = match self {
            Self::SaveDocumentAs => egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
            _ => egui::Modifiers::COMMAND,
        };
        KeyboardShortcut::new(modifiers, key)
    }

    pub(super) fn needs_document(self) -> bool {
        matches!(
            self,
            Self::SaveDocument | Self::SaveDocumentAs | Self::CloseTab
        )
    }
}

#[derive(Default)]
pub(super) struct CommandManager;

impl CommandManager {
    /// Consume every bound shortcut and return the commands that fired.
    pub(super) fn poll(&self, ctx: &egui::Context) -> Vec<AppCommand> {
        let mut commands = Vec::new();
        for command in AppCommand::ALL {
            if ctx.input_mut(|input| input.consume_shortcut(&command.shortcut())) {
                commands.push(command);
            }
        }
        commands
    }

    /// Draw a command as a menu item. Returns true when clicked.
    pub(super) fn menu_item(&self, ui: &mut egui::Ui, command: AppCommand, enabled: bool) -> bool {
        let clicked = ui
            .add_enabled(
                enabled,
                egui::Button::new(command.label()).shortcut_text(command.shortcut_text()),
            )
            .clicked();
        if clicked {
            ui.close();
        }
        clicked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_has_a_distinct_shortcut() {
        let mut shortcuts = Vec::new();
        for command in AppCommand::ALL {
            let shortcut = command.shortcut();
            assert!(!shortcuts.contains(&shortcut), "duplicate for {command:?}");
            shortcuts.push(shortcut);
        }
    }

    #[test]
    fn only_document_commands_need_a_session() {
        assert!(!AppCommand::NewDocument.needs_document());
        assert!(!AppCommand::OpenDocument.needs_document());
        assert!(AppCommand::SaveDocument.needs_document());
        assert!(AppCommand::SaveDocumentAs.needs_document());
        assert!(AppCommand::CloseTab.needs_document());
    }
}
