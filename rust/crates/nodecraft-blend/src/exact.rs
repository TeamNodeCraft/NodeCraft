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
use crate::treatment::{Continuity, CornerTurn, EdgeMode, EdgeTreatment};

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

/// Reject a fillet result that the upstream validity checker does not accept.
/// The report text is the kernel's own; do not reduce it to a status number.
fn validate_result(kernel: &mut Kernel, brep: &Brep) -> Result<(), String> {
    let report = nodecraft_layers::check_kernel_brep(kernel, brep)?;
    if report.is_valid() {
        Ok(())
    } else {
        Err(format!("fillet result failed solidcore check: {report}"))
    }
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
    // A saved viewer pick carries both a display-space centre and an edge id.
    // After a Boolean the id can address a different edge entirely, so prefer
    // the centre; if SolidCore refuses that edge as smooth, retry by id.
    let primary = crate::edge::select_edges_from_values_center_first(kernel, brep, values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if primary.is_empty() {
        return Err("an edge treatment needs at least one selected edge".into());
    }
    // A refused edge is an upstream modelling refusal, not permission to round
    // a subset. Returning the error keeps the node from silently showing a
    // partial fillet around a join.
    // A swept corner is a session setting, so it needs the fuller ops blend path
    // even when the direct constant-radius call would succeed.
    let result = if treatment.corner_turn == CornerTurn::Swept {
        fillet_selection_ops(kernel, brep, &primary, &treatment)?
    } else {
        match fillet_selection(kernel, brep, &primary, &treatment) {
            Ok(result) => result,
            Err(error) if error.contains("status 1001") => {
                match fillet_selection_ops(kernel, brep, &primary, &treatment) {
                    Ok(result) => result,
                    Err(ops_error) => return Err(format!("{error}; ops blend: {ops_error}")),
                }
            }
            Err(error) => return Err(error),
        }
    };
    validate_result(kernel, &result)?;
    Ok(result)
}

/// Route the same round through `solidcore-ops::blend`, which drives the kernel's
/// full fillet session (corners and setbacks) rather than the direct `fillet_edges`
/// shortcut.
fn fillet_selection_ops(
    kernel: &mut Kernel,
    brep: &Brep,
    selections: &[crate::EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Brep, String> {
    use solidcore_ops::blend::{Blend, Corners, Turn, blend_ordered};
    let mut working = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| format!("{error:?}"))?;
    let (record, _) = solidcore_convert::brep::to_record(kernel, &mut working)
        .map_err(|error| format!("{error:?}"))?;
    let (body, order) = solidcore_ops::Body::from_record(&record, true)
        .map_err(|error| error.to_string())?;
    let mut edges = Vec::with_capacity(selections.len());
    for selection in selections {
        edges.push(
            order
                .edges
                .iter()
                .position(|edge| *edge == selection.edge)
                .ok_or_else(|| "selected edge is missing from the record order".to_owned())?,
        );
    }
    let kind = match treatment.continuity {
        Continuity::Circular => Blend::Round,
        Continuity::G2 => Blend::Curvature,
        Continuity::G3 => return Err("G3 needs the direct fillet path".into()),
    };
    let turn = match treatment.corner_turn {
        CornerTurn::Mitre => Turn::Mitre,
        CornerTurn::Swept => Turn::Swept,
    };
    if let Err(error) = blend_ordered(
        &body,
        &order,
        kind,
        &edges,
        &[treatment.length],
        &[],
        Corners::Filled,
        turn,
    ) {
        return Err(error.to_string());
    }
    let (record, _) = body.to_record().map_err(|error| error.to_string())?;
    let (result, _) = solidcore_convert::brep::to_kernel(kernel, &record, true)
        .map_err(|error| format!("{error:?}"))?;
    Ok(result)
}

fn fillet_selection(
    kernel: &mut Kernel,
    brep: &Brep,
    selections: &[crate::EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Brep, String> {
    let edges: Vec<solidcore::topology::EdgeId> =
        selections.iter().map(|selection| selection.edge).collect();
    let mut result = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    // SolidCore's constant-radius API takes the cross-section as an integer:
    // circular (anything except 0/2) or blend curve (2). G2/G3 use the
    // blend-curve section; continuity is 1/2/3 for G1/G2/G3.
    let (xsect_type, continuity) = match treatment.continuity {
        Continuity::Circular => (1, 1),
        Continuity::G2 => (2, 2),
        Continuity::G3 => (2, 3),
    };
    solidcore::blend::fillet_executive::api::fillet_edges(
        kernel,
        &mut result,
        &edges,
        treatment.length,
        xsect_type,
        continuity,
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
    fn exact_circular_fillet_accepts_two_adjacent_edges() {
        let (kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let selections = crate::select_edges_by_midpoints(
            &kernel,
            &brep,
            &[[40.0, 15.0, 20.0], [20.0, 30.0, 20.0]],
        )
        .expect("adjacent edges");
        let edges: Vec<_> = selections.iter().map(|selection| serde_json::json!({"ordinal": selection.ordinal})).collect();
        let values = serde_json::json!({"mode": "fillet", "continuity": "", "length": 2.0, "edges": edges});
        let filleted = fillet_brep_exact(&brep, &values).expect("adjacent exact fillet");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold"));
    }

    #[test]
    fn exact_g2_and_g3_fillets_produce_closed_solids() {
        for continuity in ["G2", "G3"] {
            let (_kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
            let values = serde_json::json!({
                "mode": "fillet",
                "continuity": continuity,
                "length": 2.0,
                "edges": [{"ordinal": 0}]
            });
            let filleted = fillet_brep_exact(&brep, &values)
                .unwrap_or_else(|error| panic!("{continuity} exact fillet: {error}"));
            assert!(solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold"));
        }
    }

    #[test]
    fn exact_circular_fillet_accepts_two_parallel_edges() {
        let (kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let selections = crate::select_edges_by_midpoints(
            &kernel,
            &brep,
            &[[40.0, 15.0, 20.0], [0.0, 15.0, 20.0]],
        )
        .expect("two edges");
        let edges: Vec<_> = selections
            .iter()
            .map(|selection| serde_json::json!({"ordinal": selection.ordinal}))
            .collect();
        let values = serde_json::json!({
            "mode": "fillet",
            "continuity": "",
            "length": 2.0,
            "edges": edges,
        });
        let filleted = fillet_brep_exact(&brep, &values).expect("two-edge exact fillet");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold"));
        assert!(!filleted.faces().is_empty());
    }

    #[test]
    fn ops_blend_rounds_a_box_edge() {
        let (mut kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let selection = crate::edge_selections(&kernel, &brep).expect("edges")[0].clone();
        let treatment = EdgeTreatment {
            mode: EdgeMode::Fillet,
            continuity: Continuity::Circular,
            length: 2.0,
            ..EdgeTreatment::default()
        };
        let result = fillet_selection_ops(&mut kernel, &brep, &[selection], &treatment)
            .expect("ops blend");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&result).expect("manifold"));
    }

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
