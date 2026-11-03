//! NodeCraft's Edge node (chamfer / fillet) over solidcore.
//!
//! # What is here
//!
//! * [`edge`] — exact edge selection on a solidcore [`Brep`], by approximate midpoint
//!   (the NodeCraft viewer's edge reference) or by ordinal.
//! * [`treatment`] — the Edge node's parameter model (`mode`, `chamferType`,
//!   `continuity`, `length`, `distance2`, `angle`) with strict parsing and validation.
//! * [`preview`] — a working, watertight mesh preview of a chamfer (equal,
//!   two-distance, distance-angle) and of a circular (G1) fillet on box-like edges,
//!   built by clipping the display mesh. This is the shortest path to a usable Edge
//!   node before solidcore's fillet executive is ported.
//! * [`exact`] — the exact B-rep signature plus the precise list of missing solidcore
//!   calls/port units; returns [`BlendError::Unsupported`] today.
//!
//! # Example
//!
//! ```no_run
//! use nodecraft_blend::{box_brep, edge, treatment::EdgeTreatment, preview::apply_edge_treatment};
//!
//! # fn main() -> Result<(), nodecraft_blend::BlendError> {
//! let (kernel, brep) = box_brep(40.0, 30.0, 20.0)?;
//! let edges = edge::edge_selections(&kernel, &brep)?;
//! let treatment = EdgeTreatment::default(); // chamfer, equal, 2.0
//! let mesh = apply_edge_treatment(&kernel, &brep, &edges[..1], &treatment)?;
//! assert!(!mesh.faces.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! # Status (solidcore `main`, `Cargo.lock` rev `6ecdaaa`)
//!
//! | Capability | State |
//! |---|---|
//! | edge selection by midpoint / ordinal | works (solidcore query API) |
//! | chamfer equal / two-distance / distance-angle, 90° box edges | works (mesh preview) |
//! | fillet circular G1, 90° box edges | works (mesh preview, [`preview::FILLET_SEGMENTS`] facets) |
//! | fillet G2 / G3 | explicit [`BlendError::Unsupported`] naming the missing fillet-law calls |
//! | exact B-rep chamfer/fillet | explicit [`BlendError::Unsupported`]; needs `blend.fillet_executive_p1` |
//! | curved-face or non-90° edges | explicit [`BlendError::Unsupported`] |
//!
//! The library never panics on bad node values, bad selections or unsupported geometry:
//! every path returns a [`BlendError`].

#![forbid(unsafe_code)]

pub mod edge;
pub mod error;
pub mod exact;
pub mod missing;
pub mod preview;
pub mod treatment;

pub use edge::{
    EdgeSelection, brep_center, edge_selections, select_edge_by_midpoint, select_edges_by_midpoints,
    select_edges_by_ordinals, select_edges_from_values,
};
pub use error::BlendError;
pub use exact::{apply_edge_treatment_brep, chamfer_brep, fillet_brep, fillet_brep_exact, fillet_brep_exact_with_kernel, missing_calls, port_units};
pub use preview::{FILLET_SEGMENTS, apply_edge_treatment, chamfer_brep_exact, chamfer_brep_exact_with_kernel, edge_treatment_preview};
pub use treatment::{ChamferType, Continuity, EdgeMode, EdgeTreatment};

use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives as api;
use solidcore::topology::Brep;

/// A solidcore box at the origin (B-rep coordinates), with its kernel.
///
/// This mirrors `nodecraft_solidcore::box_mesh`'s construction (`SmApiCreateBox` at
/// `(0, 0, 0)`) so tests and callers share one convention.
pub fn box_brep(length: f64, width: f64, height: f64) -> Result<(Kernel, Brep), BlendError> {
    if !(length.is_finite() && width.is_finite() && height.is_finite())
        || length <= 0.0
        || width <= 0.0
        || height <= 0.0
    {
        return Err(BlendError::InvalidInput(format!(
            "box dimensions must be positive finite numbers, got ({length}, {width}, {height})"
        )));
    }
    let mut kernel = Kernel::default();
    let brep = api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), length, width, height)
        .map_err(|error| BlendError::Kernel(format!("create_box: {error}")))?;
    Ok((kernel, brep))
}
