//! Types shared by the exact blend wrappers.

use solidcore::topology::Brep;

/// A blended B-rep and the user-facing warnings from the blend.
#[derive(Debug)]
pub struct Blended {
    pub brep: Brep,
    pub warnings: Vec<String>,
}
