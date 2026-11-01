//! Errors returned by the assembly parser.

use std::fmt;

/// Why a STEP assembly could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblyError {
    /// `nodecraft_step::StepFile::parse` rejected the file syntax.
    Step(String),
    /// The file is syntactically valid STEP but its product/assembly graph is not
    /// usable (missing products/definitions, a cycle, a singular placement, ...).
    Malformed(String),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step(message) => write!(formatter, "STEP parse error: {message}"),
            Self::Malformed(message) => write!(formatter, "assembly structure error: {message}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

impl From<std::io::Error> for AssemblyError {
    fn from(error: std::io::Error) -> Self {
        Self::Malformed(error.to_string())
    }
}
