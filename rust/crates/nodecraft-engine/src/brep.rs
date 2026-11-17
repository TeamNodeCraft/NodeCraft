//! Shared B-rep copy, placement and snapshot helpers.
//!
//! Node geometry lives in `crate::nodes`; these are the engine-wide
//! primitives several non-node paths also use.

pub(crate) fn copy_brep(brep: &nodecraft_solidcore::solidcore::Brep) -> Result<nodecraft_solidcore::solidcore::Brep, String> {
    brep.clone_standing_for_self()
        .map_err(|error| format!("B-rep copy: {error}"))
}

pub(crate) fn snapshot_brep(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> Result<nodecraft_solidcore::solidcore::topology::Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(&mut kernel, brep)
        .map_err(|error| error.to_string())
}

