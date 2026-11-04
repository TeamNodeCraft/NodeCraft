//! The exact (B-rep) edge-treatment path and why it cannot run yet.
//!
//! Upstream SMLib reaches an edge treatment through the SM_API fillet entry points:
//!
//! ```text
//! SmApiFilletEdges / SmApiChamferFillet
//!     -> SmFilletExecutive::SetFilletParameters
//!     -> SmFilletExecutive::CreateFilletCorners
//!     -> SmFilletExecutive::DoFilleting
//!        -> SmFilletStandardSolver / SmFilletCorner / SmFilletGeom
//! ```
//!
//! At the locked solidcore revision (`Cargo.lock` rev `6ecdaaa`) none of the SM_API
//! blend functions is ported, and `blend.fillet_executive_p1` — which owns
//! `SetFilletParameters`, `CreateFilletCorners`, `DoPiecewiseMerge` and all seven
//! `SmApi*Fillet*` entry points — is explicitly listed as *not ported yet* in
//! `plan/PROGRESS.md`. The ported executive slice
//! (`blend::fillet_executive::p3::FilletExecutive`) can only be driven through the
//! `ExecutiveAccess` trait, whose only implementation lives in solidcore's private
//! `p3_tests` module and several of whose methods return
//! `SolidError::Unported`.
//!
//! [`apply_edge_treatment_brep`] therefore validates the request, names the missing
//! calls, and returns [`BlendError::Unsupported`]. It never panics and never returns a
//! silently wrong B-rep; when the upstream units land, this is the single function to
//! wire to the real executive. Until then [`crate::apply_edge_treatment`] provides the
//! mesh preview.
//!
//! ## Shortest path once the port lands
//!
//! 1. `blend.fillet_executive_p1`: `SetFilletParameters`, `CreateFilletCorners`,
//!    `DoFilleting`, `DoPiecewiseMerge`, `RemoveTopoEdges` + the seven SM_API
//!    entry points.
//! 2. An `ExecutiveAccess` adapter over `Brep`/`Kernel` (the open design question noted
//!    in `plan/units/blend.fillet_executive_p1.toml`: object identity between
//!    `SmFilletEdgeuse` records and B-rep edge use ids).
//! 3. `blend.fillet_standard_solver_p2` + `blend.fillet_corner_p1/p2` for corners.
//! 4. `blend.fillet_geom_p3/p5/p6` for the fillet surface/trim geometry.
//! 5. G2/G3 additionally need the B-spline fillet law (`SmBSplineFilletLaw`) and its
//!    `Evaluate`, which `blend.fillet_standard_solver_p2` owns.
//!
//! The mesh preview in [`crate::preview`] is the interim user-visible path: it uses
//! the same exact edge selection and the same node parameters, so switching to the
//! B-rep path later only replaces the geometry step.

use solidcore::kernel::Kernel;
use solidcore::topology::Brep;

use crate::edge::EdgeSelection;
use crate::error::BlendError;
use crate::missing::{CHAMFER_API, EXECUTIVE, FILLET_API, LAW_CONTINUITY, PORT_UNITS};
use crate::treatment::{Continuity, EdgeMode, EdgeTreatment};

/// The upstream calls an exact (B-rep) treatment of `mode` still needs.
#[must_use]
pub fn missing_calls(mode: EdgeMode, continuity: Continuity) -> Vec<&'static str> {
    let mut calls: Vec<&'static str> = match mode {
        EdgeMode::Chamfer => CHAMFER_API.to_vec(),
        EdgeMode::Fillet => FILLET_API.to_vec(),
    };
    if continuity != Continuity::Circular {
        calls.extend_from_slice(LAW_CONTINUITY);
    }
    calls.extend_from_slice(EXECUTIVE);
    calls
}

/// The solidcore port units that own those calls.
#[must_use]
pub const fn port_units() -> &'static [&'static str] {
    PORT_UNITS
}

/// Exact circular fillet of the selected edges from node `values`.
///
/// Uses solidcore's public `fillet_edges` entry point; this is the modeling
/// path, not the display-mesh preview.
pub fn fillet_brep_exact(brep: &Brep, values: &serde_json::Value) -> Result<Brep, String> {
    let mut kernel = nodecraft_solidcore::kernel::kernel();
    fillet_brep_exact_with_kernel(&mut kernel, brep, values)
}

/// [`fillet_brep_exact`] on a caller-owned (worker) kernel.
pub fn fillet_brep_exact_with_kernel(
    kernel: &mut Kernel,
    brep: &Brep,
    values: &serde_json::Value,
) -> Result<Brep, String> {
    let treatment = EdgeTreatment::from_node_values(values).map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    treatment.validate().map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if treatment.mode != EdgeMode::Fillet {
        return Err("exact fillet requires fillet mode".into());
    }
    if treatment.continuity != Continuity::Circular {
        return Err("G2/G3 fillets need a B-spline fillet law; only circular G1 is exact here".into());
    }
    let selections = crate::edge::select_edges_from_values(kernel, brep, values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if selections.is_empty() {
        return Err("an edge treatment needs at least one selected edge".into());
    }
    let edges: Vec<solidcore::topology::EdgeId> = selections.iter().map(|selection| selection.edge).collect();
    let mut result = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    solidcore::blend::fillet_executive::api::fillet_edges(
        kernel,
        &mut result,
        &edges,
        treatment.length,
        2, // exact circular cross-section
        1, // G1
        1.0,
    )
    .map_err(|error| {
        let text = error.to_string();
        if text.contains("status 1001") {
            format!(
                "solidcore fillet failed (status 1001): the radius may be too large for the selected edges or shell wall; reduce it ({text})"
            )
        } else {
            format!("solidcore fillet: {text}")
        }
    })?;
    Ok(result)
}

/// Exact B-rep chamfer/fillet of the selected edges.
///
/// Returns the missing solidcore API as [`BlendError::Unsupported`] at the locked
/// revision; the signature is the one the worker will call once the executive is
/// ported, so the surrounding plumbing does not change.
pub fn apply_edge_treatment_brep(
    _kernel: &Kernel,
    _brep: &Brep,
    selections: &[EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Brep, BlendError> {
    treatment.validate()?;
    if selections.is_empty() {
        return Err(BlendError::InvalidInput(
            "an exact edge treatment needs at least one selected edge".into(),
        ));
    }
    let feature: &'static str = match (treatment.mode, treatment.continuity) {
        (EdgeMode::Chamfer, _) => "exact chamfer",
        (EdgeMode::Fillet, Continuity::Circular) => "exact circular fillet (G1)",
        (EdgeMode::Fillet, Continuity::G2) => "exact fillet G2",
        (EdgeMode::Fillet, Continuity::G3) => "exact fillet G3",
    };
    let reason = format!(
        "solidcore has no ported SM_API edge-treatment entry point; the ported \
         blend::fillet_executive::p3 slice needs an ExecutiveAccess adapter and \
         CreateFilletCorners/SetFilletParameters/DoFilleting are still unported. \
         Use the mesh preview (apply_edge_treatment) until {} lands.",
        PORT_UNITS.join(", ")
    );
    Err(BlendError::missing_api(
        feature,
        reason,
        missing_calls(treatment.mode, treatment.continuity),
    ))
}

/// Exact circular chamfer of the selected edges; see [`apply_edge_treatment_brep`].
pub fn chamfer_brep(
    kernel: &Kernel,
    brep: &Brep,
    selections: &[EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Brep, BlendError> {
    if treatment.mode != EdgeMode::Chamfer {
        return Err(BlendError::InvalidInput(
            "chamfer_brep called with a fillet treatment".into(),
        ));
    }
    apply_edge_treatment_brep(kernel, brep, selections, treatment)
}

/// Exact fillet of the selected edges; see [`apply_edge_treatment_brep`].
pub fn fillet_brep(
    kernel: &Kernel,
    brep: &Brep,
    selections: &[EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Brep, BlendError> {
    if treatment.mode != EdgeMode::Fillet {
        return Err(BlendError::InvalidInput(
            "fillet_brep called with a chamfer treatment".into(),
        ));
    }
    apply_edge_treatment_brep(kernel, brep, selections, treatment)
}

#[cfg(test)]
mod exact_fillet_tests {
    use super::*;

    #[test]
    fn exact_circular_fillet_of_a_box_is_a_closed_solid() {
        let (_kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let values = serde_json::json!({
            "mode": "fillet",
            "continuity": "",
            "length": 4.0,
            "edges": [{"ordinal": 0}]
        });
        let filleted = fillet_brep_exact(&brep, &values).expect("exact fillet");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold"));
        assert!(!filleted.faces().is_empty());
    }
}
