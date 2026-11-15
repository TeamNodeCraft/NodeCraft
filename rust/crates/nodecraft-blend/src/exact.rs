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
use crate::treatment::{Continuity, CornerTurn, EdgeMode, EdgeTreatment};

/// Reject a fillet result that the upstream validity checker does not accept.
/// The report text is the kernel's own; do not reduce it to a status number.
pub(crate) fn validate_result(kernel: &mut Kernel, brep: &Brep) -> Result<(), String> {
    // Full topology validation is a debug/test invariant. Release previews and
    // edits rely on the upstream operation's own validity contract; opt back in
    // with NODECRAFT_VALIDATE_RESULTS=1 when diagnosing a model.
    if !cfg!(any(test, debug_assertions))
        && std::env::var_os("NODECRAFT_VALIDATE_RESULTS").is_none()
    {
        return Ok(());
    }
    let report = nodecraft_ops::OpsSession::new(kernel)
        .check(brep, nodecraft_ops::CheckLevel::Geometry)
        .map_err(|error| format!("fillet result check: {error}"))?;
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
pub(crate) fn mark_made_blend_faces(
    kernel: &nodecraft_solidcore::solidcore::kernel::Kernel,
    result: &mut nodecraft_solidcore::solidcore::topology::Brep,
) -> Result<(), String> {
    // A fillet over several edges/tangent chains finishes one session per edge.
    // Keep every generated face, not just the last session's, or an additive
    // fillet would leave most of its new surface out of the overlay.
    let made: Vec<_> = kernel
        .blend_face_log
        .as_ref()
        .map(|log| {
            let mut made: Vec<nodecraft_solidcore::solidcore::blend::definition_check::MadeBlendFace> = Vec::new();
            for session in &log.sessions {
                for face in session {
                    if !made.iter().any(|candidate: &nodecraft_solidcore::solidcore::blend::definition_check::MadeBlendFace| candidate.face == face.face) {
                        made.push(*face);
                    }
                }
            }
            made
        })
        .unwrap_or_default();
    if made.is_empty() {
        return Ok(());
    }
    nodecraft_solidcore::solidcore::blend::definition_check::mark_blend_faces(result, &made)
        .map_err(|error| error.to_string())
}

pub fn fillet_brep_exact_with_kernel_warned(
    kernel: &mut Kernel,
    brep: &Brep,
    values: &serde_json::Value,
) -> Result<crate::ops_blend::Blended, String> {
    let treatment = EdgeTreatment::from_node_values(values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    treatment
        .validate()
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if treatment.mode != EdgeMode::Fillet {
        return Err("exact fillet requires fillet mode".into());
    }
    // A saved viewer pick carries both a display-space centre and an edge id.
    // After a Boolean the id can address a different edge entirely, so prefer
    // the centre; if SolidCore refuses that edge as smooth, retry by id.
    kernel.blend_face_log = Some(Default::default());
    let primary = crate::edge::select_edges_from_values(kernel, brep, values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    if primary.is_empty() {
        return Err("an edge treatment needs at least one selected edge".into());
    }
    // A Boolean join is split into several edges at seams even though the join
    // is one continuous curve. Parasolid-style edge treatment follows that
    // tangent run; without it a one-edge pick rounds only one segment and
    // leaves a visible gap.
    let tangent_chain = nodecraft_node_contract::resolve_bool("fillet", values, "tangentChain")
        .expect("tangentChain is a contract field");
    let primary = if tangent_chain {
        crate::edge::tangent_chain(kernel, brep, &primary, 0.5_f64.to_radians())
            .map_err(|error| nodecraft_solidcore::error::user_message(error))?
    } else {
        primary
    };
    if treatment.continuity != Continuity::G3 {
        use nodecraft_ops::CornerTurn as Turn;
        let kind = match treatment.continuity {
            Continuity::Circular => nodecraft_ops::OpsBlend::Round(match treatment.corner_turn {
                CornerTurn::Mitre => Turn::Mitre,
                CornerTurn::Swept => Turn::Swept,
            }),
            Continuity::G2 => nodecraft_ops::OpsBlend::Curvature,
            Continuity::G3 => unreachable!(),
        };
        let edges: Vec<_> = primary.iter().map(|selection| selection.edge).collect();
        let result = nodecraft_ops::OpsSession::new(kernel)
            .blend(brep, &edges, treatment.length, kind)
            .map_err(|error| error.to_string())?;
        validate_result(kernel, &result.brep)?;
        return Ok(crate::ops_blend::Blended { brep: result.brep, warnings: result.warnings });
    }
    // G3 has no solidcore-ops section; it keeps the direct kernel call.

    // Where three or more rounded edges meet, the direct path leaves a Mitre
    // corner (with a cusp). Use the kernel's set-back corner session so the
    // corner is closed by a patch instead.
    if let Some(result) =
        crate::fillet_corner::fillet_with_corner_patches(kernel, brep, &primary, &treatment)
            .map_err(|error| nodecraft_solidcore::error::user_message(error))?
    {
        validate_result(kernel, &result)?;
        let mut result = result;
        mark_made_blend_faces(kernel, &mut result)?;
        return Ok(crate::ops_blend::Blended { brep: result, warnings: Vec::new() });
    }
    // A refused edge is an upstream modelling refusal, not permission to round
    // a subset. Returning the error keeps the node from silently showing a
    // partial fillet around a join.
    // A swept corner is a session setting, so it needs the fuller ops blend path
    // even when the direct constant-radius call would succeed.
    let result = fillet_selection(kernel, brep, &primary, &treatment)?;
    validate_result(kernel, &result)?;
    let mut result = result;
    mark_made_blend_faces(kernel, &mut result)?;
    Ok(crate::ops_blend::Blended { brep: result, warnings: Vec::new() })
}

/// Fillet without returning warnings.
pub fn fillet_brep_exact_with_kernel(
    kernel: &mut Kernel,
    brep: &Brep,
    values: &serde_json::Value,
) -> Result<Brep, String> {
    fillet_brep_exact_with_kernel_warned(kernel, brep, values).map(|blended| blended.brep)
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
        // The G^n section refuses upstream's 1.0 thumbweight; use the kernel's
        // default for G2/G3 blend-curve cross-sections.
        if xsect_type == 2 {
            solidcore::blend::fillet_section::section::GN_DEFAULT_THUMBWEIGHT
        } else {
            1.0
        },
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
        let edges: Vec<_> = selections
            .iter()
            .map(|selection| serde_json::json!({"ordinal": selection.ordinal}))
            .collect();
        let values =
            serde_json::json!({"mode": "fillet", "continuity": "", "length": 2.0, "edges": edges});
        let filleted = fillet_brep_exact(&brep, &values).expect("adjacent exact fillet");
        assert!(
            solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold")
        );
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
            assert!(
                solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold")
            );
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
        assert!(
            solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold")
        );
        assert!(!filleted.faces().is_empty());
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
        assert!(
            solidcore::query::api_queries::brep_is_manifold_solid(&filleted).expect("manifold")
        );
        assert!(!filleted.faces().is_empty());
    }
    #[test]
    fn node_fillet_keeps_look_colour_and_leaves_the_round_uncoloured() {
        use solidcore::geom::attribute::{Attribute, Behavior};
        use solidcore::geom::vector3d::Vec3;
        let (kernel, mut brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let faces: Vec<_> = brep.faces().ids().collect();
        for face in &faces {
            brep.add_attribute(
                solidcore::EntityRef::Face(*face),
                Attribute::color(100, Vec3::new(1.0, 0.0, 0.0), Behavior::Copy),
            )
            .expect("colour");
        }
        let values = serde_json::json!({
            "mode": "fillet",
            "continuity": "",
            "length": 2.0,
            "edges": [{"ordinal": 0}]
        });
        let filleted = fillet_brep_exact(&brep, &values).expect("fillet");
        let coloured = filleted
            .faces()
            .iter()
            .filter(|(face, _)| filleted.find_attribute(solidcore::EntityRef::Face(*face), 100).is_some())
            .count();
        assert_eq!(coloured, 6, "the six box faces keep their colour");
        let marked = solidcore::blend::definition_check::blend_face_marks(&filleted)
            .expect("marks")
            .first()
            .expect("made round")
            .face;
        assert!(
            filleted
                .find_attribute(solidcore::EntityRef::Face(marked), 100)
                .is_none(),
            "the made round must be uncoloured"
        );
    }

    #[test]
    fn second_node_fillet_resolves_an_edge_on_the_new_body() {
        let (kernel, brep) = crate::box_brep(40.0, 30.0, 20.0).expect("box");
        let first = serde_json::json!({
            "mode": "fillet", "continuity": "", "length": 2.0, "edges": [{"ordinal": 0}]
        });
        let once = fillet_brep_exact(&brep, &first).expect("first fillet");
        let selections = crate::edge_selections(&kernel, &once).expect("edges");
        let edge = selections.iter().find(|selection| selection.edge == once.edge_list()[0])
            .or_else(|| selections.first())
            .expect("edge on the new body")
            .clone();
        let second = serde_json::json!({
            "mode": "fillet",
            "continuity": "",
            "length": 1.0,
            "edges": [{"center": edge.midpoint}]
        });
        let twice = fillet_brep_exact(&once, &second).expect("second fillet");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&twice).expect("manifold"));
    }

}
