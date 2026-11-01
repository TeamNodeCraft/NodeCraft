//! Errors for the blend wrapper.
//!
//! Every failure is a value: the library never panics and never unwraps. The variants
//! separate "the caller's node values/selection are wrong" from "solidcore cannot do
//! this yet", so a NodeCraft Edge node can show the difference to the user.

use std::fmt;

/// Why an edge treatment could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlendError {
    /// The node values, the edge selection or the input mesh are malformed.
    InvalidInput(String),
    /// The requested feature is known but not implementable with the current
    /// solidcore port. `missing` lists the exact upstream/SM_API calls or port units
    /// that must land first.
    Unsupported {
        /// Short feature name, e.g. `"fillet G2"`.
        feature: &'static str,
        /// Why this input cannot be treated.
        reason: String,
        /// Upstream symbols / port units that are missing (empty when the geometry
        /// itself is unsupported, e.g. a curved edge).
        missing: Vec<&'static str>,
    },
    /// solidcore returned an error while reading the B-rep or its display mesh.
    Kernel(String),
}

impl BlendError {
    /// `Unsupported` with a reason and no missing-symbol list.
    #[must_use]
    pub fn unsupported(feature: &'static str, reason: impl Into<String>) -> Self {
        Self::Unsupported {
            feature,
            reason: reason.into(),
            missing: Vec::new(),
        }
    }

    /// `Unsupported` naming the missing solidcore API calls.
    #[must_use]
    pub fn missing_api(
        feature: &'static str,
        reason: impl Into<String>,
        missing: Vec<&'static str>,
    ) -> Self {
        Self::Unsupported {
            feature,
            reason: reason.into(),
            missing,
        }
    }

    /// True when the failure is a missing solidcore port rather than bad input.
    #[must_use]
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }

    /// The feature name of an `Unsupported` error.
    #[must_use]
    pub fn feature(&self) -> Option<&'static str> {
        match self {
            Self::Unsupported { feature, .. } => Some(feature),
            _ => None,
        }
    }
}

impl fmt::Display for BlendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Kernel(message) => write!(f, "solidcore: {message}"),
            Self::Unsupported {
                feature,
                reason,
                missing,
            } => {
                write!(f, "unsupported ({feature}): {reason}")?;
                if !missing.is_empty() {
                    write!(f, " [missing: {}]", missing.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for BlendError {}

impl From<solidcore::error::SolidError> for BlendError {
    fn from(error: solidcore::error::SolidError) -> Self {
        Self::Kernel(error.to_string())
    }
}
