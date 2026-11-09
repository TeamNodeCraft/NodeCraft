//! Setback filler corners for multi-edge fillets.
//!
//! A constant-radius fillet where three or more rounded edges meet at one vertex
//! needs a corner patch. The direct `fillet_edges` path leaves that corner as a
//! Mitre (or a rolling-ball sphere), which produces the folded/cusped corners on
//! box-style trihedral vertices. SolidCore's `FilletSession` has the set-back
//! and patch-fill machinery; this module drives it for NodeCraft's exact fillet.
//!
//! Only corners where every incident edge is selected are set back: a corner
//! with an unrounded edge has no patch boundary to close, and the kernel refuses
//! it. Corners that cannot be built return an error rather than silently falling
//! back to the cusped Mitre result.
//!
//! SolidCore's set-back patch is currently built from circular sections.
//! Curvature (G2/G3) sections would need the matched G2 patch that SolidCraft
//! has; this path refuses them instead of silently changing the requested
//! continuity. Single-edge and non-corner fillets keep their continuity.

use solidcore::blend::fillet_executive::session::FilletSession;
use solidcore::blend::fillet_solver::p1::FilletSurfaceGeneratorType;
use solidcore::kernel::Kernel;
use solidcore::topology::{Brep, EdgeId, VertexId};

use crate::{BlendError, Continuity, EdgeSelection, EdgeTreatment};

/// Default corner set-back as a multiple of the radius. The patch needs to start
/// past the point where the rounds would meet; at a square corner that is one
/// radius, and 1.5 leaves a patch wide enough to blend the rounds.
const SETBACK_FACTOR: f64 = 1.5;

/// Fillet through the kernel's set-back corner session when the selection has a
/// vertex where every incident edge is being rounded.
///
/// Returns `Ok(None)` when the selection has no such corner, so the caller can
/// use the ordinary direct fillet path.
pub(crate) fn fillet_with_corner_patches(
    kernel: &mut Kernel,
    brep: &Brep,
    selections: &[EdgeSelection],
    treatment: &EdgeTreatment,
) -> Result<Option<Brep>, BlendError> {
    if treatment.mode != crate::EdgeMode::Fillet {
        return Ok(None);
    }
    let corners = setback_vertices(brep, selections)?;
    if corners.is_empty() {
        return Ok(None);
    }
    if matches!(treatment.continuity, Continuity::G2 | Continuity::G3) {
        return Err(BlendError::unsupported(
            "fillet G2 corner",
            format!(
                "{} fillet: the matched G2 corner patch is not ported yet; use circular \
                 continuity for this multi-edge corner, or fillet the edges separately",
                treatment.continuity.as_str()
            ),
        ));
    }
    if treatment.corner_turn == crate::CornerTurn::Swept {
        // A swept turn is the direct/ops path's job; the set-back patch is not a
        // swept corner and must not silently replace it.
        return Ok(None);
    }

    let mut working = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| BlendError::Kernel(error.to_string()))?;
    let mut session = FilletSession::new(kernel, &mut working)
        .map_err(|error| BlendError::Kernel(format!("fillet session: {error}")))?;
    let edges: Vec<EdgeId> = selections.iter().map(|selection| selection.edge).collect();
    let radii = vec![treatment.length; edges.len()];
    // See the module docs: the portable set-back patch is circular. Keep the
    // requested G2/G3 sections for fillets without a multi-edge corner.
    let section = FilletSurfaceGeneratorType::Circular;
    session
        .prepare_per_edge_constant_radius(kernel, &edges, &radii, &[section], 1, 1.0)
        .map_err(|error| BlendError::Kernel(format!("fillet session prepare: {error}")))?;

    let setback = treatment.length * SETBACK_FACTOR;
    check_setback_room(brep, selections, &corners, setback)?;
    for vertex in &corners {
        session.add_set_back_vertex(*vertex);
    }
    session
        .create_corners(kernel, None)
        .map_err(|error| BlendError::Kernel(format!("fillet corners: {error}")))?;
    // Set the setback on the corners that were actually made at those vertices.
    for corner in session
        .fillet_corners()
        .map_err(|error| BlendError::Kernel(format!("fillet corners: {error}")))?
    {
        let record = session
            .corner(corner)
            .map_err(|error| BlendError::Kernel(format!("fillet corner: {error}")))?;
        if record
            .vertex
            .is_some_and(|vertex| corners.contains(&vertex))
        {
            session
                .set_corner_setback(corner, setback)
                .map_err(|error| BlendError::Kernel(format!("setback: {error}")))?;
        }
    }
    session.set_mark_corner_patches(true);
    session
        .do_filleting(kernel, None)
        .map_err(|error| BlendError::Kernel(format!("fillet session: {error}")))?;
    drop(session);
    // The set-back session leaves the corner patches as separate faces; weld and
    // orient them into the one closed solid.
    solidcore::topology::brep::p5::stitch_and_orient(
        kernel,
        &mut working,
        &solidcore::topology::brep::p5::StitchAndOrientOptions::default(),
    )
    .map_err(|error| BlendError::Kernel(format!("fillet corner stitch: {error}")))?;
    Ok(Some(working))
}

/// Refuse a set-back that cannot fit on the selected edges. The patch runs a
/// setback along each incident edge; two corners on the same edge must also
/// leave material between their setbacks.
fn check_setback_room(
    brep: &Brep,
    selections: &[EdgeSelection],
    corners: &[VertexId],
    setback: f64,
) -> Result<(), BlendError> {
    let lengths: std::collections::BTreeMap<EdgeId, f64> = selections
        .iter()
        .map(|selection| (selection.edge, selection.length))
        .collect();
    let mut corners_on_edge: std::collections::BTreeMap<EdgeId, usize> =
        std::collections::BTreeMap::new();
    for edge in brep.edge_list() {
        let record = brep
            .edge(*edge)
            .map_err(|error| BlendError::Kernel(error.to_string()))?;
        let at_corner = record
            .vertices
            .iter()
            .filter(|vertex| corners.contains(vertex))
            .count();
        if at_corner > 0 {
            let end_count = if record.vertices[0] == record.vertices[1] { 1 } else { 2 };
            let affected = at_corner.min(end_count);
            corners_on_edge.insert(*edge, affected);
        }
    }
    for (edge, count) in corners_on_edge {
        let Some(length) = lengths.get(&edge).copied() else {
            continue;
        };
        let needed = setback * count as f64;
        if !(needed < length) {
            return Err(BlendError::InvalidInput(format!(
                "fillet radius {:.6} needs a {needed:.6} setback on an edge {length:.6} long; \
                 reduce the radius or turn off tangent chain",
                setback / SETBACK_FACTOR
            )));
        }
    }
    Ok(())
}

/// Vertices where at least three selected edges meet and every incident edge is
/// selected. These are the corners a set-back patch can close.
fn setback_vertices(
    brep: &Brep,
    selections: &[EdgeSelection],
) -> Result<Vec<VertexId>, BlendError> {
    use std::collections::{BTreeMap, BTreeSet};
    let selected: BTreeSet<EdgeId> = selections.iter().map(|selection| selection.edge).collect();
    let mut incident: BTreeMap<VertexId, BTreeSet<EdgeId>> = BTreeMap::new();
    for edge in brep.edge_list() {
        let record = brep
            .edge(*edge)
            .map_err(|error| BlendError::Kernel(error.to_string()))?;
        for vertex in record.vertices {
            incident.entry(vertex).or_default().insert(*edge);
        }
    }
    Ok(incident
        .into_iter()
        .filter_map(|(vertex, edges)| {
            let selected_at = edges.intersection(&selected).count();
            (selected_at >= 3 && selected_at == edges.len()).then_some(vertex)
        })
        .collect())
}
