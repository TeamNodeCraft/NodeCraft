//! User-facing formatting for errors coming out of the modelling kernel.

use std::fmt::Display;

/// Format any kernel/blend error, adding guidance for failures that are
/// actionable in the editor (rather than raw status text).
#[must_use]
pub fn user_message(error: impl Display) -> String {
    let text = error.to_string();
    if text.contains("cannot be implanted") || text.contains("FaceTrimsUnavailable") {
        return format!(
            "{text} — the face trims could not be rebuilt while the body is being edited; heal or retrim the face and try again"
        );
    }
    if text.contains("status 1001") {
        return format!(
            "{text} — the operation could not close the resulting topology; check the feature size, edge selection and body validity"
        );
    }
    text
}

/// User-facing message for a B-rep operation that failed on a body.
#[must_use]
pub fn brep_message(operation: &str, error: impl Display) -> String {
    format!("{operation}: {}", user_message(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_failure_gets_actionable_guidance() {
        let message = user_message("ray_classify: the trims of face 4 cannot be implanted while it is edited");
        assert!(message.contains("heal or retrim"));
    }

    #[test]
    fn unrelated_errors_are_unchanged() {
        assert_eq!(user_message("invalid input"), "invalid input");
    }
}
