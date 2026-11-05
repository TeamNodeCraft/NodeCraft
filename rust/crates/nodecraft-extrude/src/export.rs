//! STEP-export bridge for exact extrude results.
//!
//! Turns a planar, polygonal solidcore [`Brep`] into the neutral topology the
//! `nodecraft-step` planar writer consumes, in both of its forms:
//!
//! * [`planar_brep_to_step_brep`] builds a
//!   [`StepBrep`](nodecraft_step::StepBrep) with one closed shell and one
//!   `MANIFOLD_SOLID_BREP`;
//!   [`nodecraft_step::write_step_from_step_brep`] then writes it.
//! * [`planar_brep_to_planar_body`] yields the equivalent
//!   [`PlanarBody`](nodecraft_step::PlanarBody), which
//!   [`nodecraft_step::write_planar_body`] writes directly and which the worker's
//!   planar import path already consumes.
//!
//! Both forms are produced in one call by [`extrude_planar_face_exact_for_export`],
//! and [`write_exact_extrude_step`] writes the same chain straight to a file.
//!
//! # Representable B-reps
//!
//! Only B-reps whose faces are planar and whose boundary edges are lines can be
//! represented: this is exactly the result of [`crate::extrude_planar_face_exact`] for
//! a polygonal profile (planar caps plus planar side faces with line bounds). Curved
//! profile extrusions (arc/circle boundaries) produce curved side faces and are
//! rejected with [`ExtrudeError::ExportUnsupported`]; they need an analytic/NURBS STEP
//! writer. A B-rep with live face records outside its face list (the legacy curve-sweep
//! routes) is rejected too: export would silently drop them.
//!
//! # Limits
//!
//! * one solid per file, no void shells (`BREP_WITH_VOIDS` is not representable);
//! * the emitted file is in millimetres (solidcore model units); every face is
//!   re-emitted as a `PLANE` through its loop and every edge as a `LINE` between its
//!   vertices, so original surface/curve entity ids and names are not preserved;
//! * the worker has no extrude/export node yet: full feature-chain export needs a
//!   worker feature node that calls [`write_exact_extrude_step`] (or exports a
//!   [`crate::extrude_planar_face_exact`] result with
//!   [`nodecraft_step::write_planar_body`] / [`nodecraft_step::write_step_from_step_brep`])
//!   and returns the path. This crate exposes and tests that B-rep side only.

use std::collections::BTreeMap;

use solidcore::geom::extent1d::Extent1d;
use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::query::api_queries::face_get_boundary_loops;
use solidcore::query::face::p1::evaluate_curve;
use solidcore::topology::face::p2 as face_p2;
use solidcore::{Brep, EdgeId, FaceId};
use nodecraft_step::topology::{
    StepBrep, StepEdge, StepFace, StepLoop, StepOrientedEdge, StepShell, StepSolid, StepVertex,
};
use nodecraft_step::PlanarBody;

use crate::ExtrudeError;

/// An exact extrude result with both neutral STEP-export representations.
#[derive(Debug)]
pub struct ExportableExtrude {
    /// The exact solidcore B-rep from the taper-extrude route.
    pub brep: Brep,
    /// The same solid as topology-first neutral STEP data, ready for
    /// [`nodecraft_step::write_step_from_step_brep`].
    pub step_brep: StepBrep,
    /// The same solid as the planar neutral body used by
    /// [`nodecraft_step::write_planar_body`] and the worker's planar import path.
    pub planar_body: PlanarBody,
}

/// Converts a planar, polygonal solidcore B-rep into a [`StepBrep`] with one closed
/// shell and one solid.
///
/// Faces are walked in `Brep::face_list()` order, each face's bound order is kept
/// (outer loop first), and each edge use keeps its `OrientType` as the STEP edge
/// orientation. Vertices are welded by position (`1e-7` quantization), so shared
/// solidcore edges become one `StepEdge` used by both adjacent faces.
///
/// # Errors
///
/// [`ExtrudeError::ExportUnsupported`] when the B-rep is empty, has live face records
/// outside its face list, has a non-planar face/surface, or has a boundary edge that
/// is not a line; solidcore query errors otherwise.
pub fn planar_brep_to_step_brep(k: &Kernel, brep: &Brep) -> Result<StepBrep, ExtrudeError> {
    ensure_exportable(k, brep)?;

    let mut topology = StepBrep::default();
    let mut next_id = 1u32;
    let mut vertex_of: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    let mut edge_of: BTreeMap<EdgeId, u32> = BTreeMap::new();
    let mut shell_faces = Vec::new();

    for &face_id in brep.face_list() {
        let loops = face_get_boundary_loops(brep, face_id).map_err(|error| {
            ExtrudeError::solid(
                "query::api_queries::face_get_boundary_loops",
                error,
            )
        })?;
        let mut cursor = 0usize;
        let mut bounds = Vec::with_capacity(loops.edge_counts.len());
        let mut outer_loop = None;
        for &edge_count in &loops.edge_counts {
            let end = cursor
                .checked_add(edge_count)
                .ok_or_else(|| ExtrudeError::ExportUnsupported {
                    why: format!("face {face_id:?} has a malformed boundary loop"),
                })?;
            let edge_ids = loops
                .edges
                .get(cursor..end)
                .ok_or_else(|| ExtrudeError::ExportUnsupported {
                    why: format!("face {face_id:?} has malformed boundary edges"),
                })?;
            let orientations = loops
                .orientations
                .get(cursor..end)
                .ok_or_else(|| ExtrudeError::ExportUnsupported {
                    why: format!("face {face_id:?} has malformed boundary orientations"),
                })?;

            let mut oriented_edges = Vec::with_capacity(edge_count);
            for (&edge_id, &orientation) in edge_ids.iter().zip(orientations) {
                let step_edge = if let Some(&step_edge) = edge_of.get(&edge_id) {
                    step_edge
                } else {
                    let edge = brep.edge(edge_id).map_err(|error| {
                        ExtrudeError::solid("topology::Brep::edge", error)
                    })?;
                    let curve = edge.curve.as_ref().ok_or_else(|| {
                        ExtrudeError::ExportUnsupported {
                            why: format!("face {face_id:?} boundary edge {edge_id:?} has no curve"),
                        }
                    })?;
                    let [start, _, _] =
                        evaluate_curve(k, Some(curve), edge.interval.min()).map_err(|error| {
                            ExtrudeError::solid("query::face::p1::evaluate_curve", error)
                        })?;
                    let [end, _, _] =
                        evaluate_curve(k, Some(curve), edge.interval.max()).map_err(|error| {
                            ExtrudeError::solid("query::face::p1::evaluate_curve", error)
                        })?;
                    let start_vertex = export_vertex(
                        &mut topology,
                        &mut next_id,
                        &mut vertex_of,
                        start,
                    );
                    let end_vertex =
                        export_vertex(&mut topology, &mut next_id, &mut vertex_of, end);
                    let id = next_id;
                    next_id += 1;
                    topology.edges.insert(
                        id,
                        StepEdge {
                            id,
                            start: start_vertex,
                            end: end_vertex,
                            curve: 0,
                            same_sense: true,
                        },
                    );
                    edge_of.insert(edge_id, id);
                    id
                };
                oriented_edges.push(StepOrientedEdge {
                    edge: step_edge,
                    orientation: orientation == OrientType::Same,
                });
            }

            let loop_id = next_id;
            next_id += 1;
            topology.loops.insert(
                loop_id,
                StepLoop {
                    id: loop_id,
                    oriented_edges,
                },
            );
            if outer_loop.is_none() {
                outer_loop = Some(loop_id);
            }
            bounds.push((loop_id, true));
            cursor = end;
        }
        let Some(outer_loop) = outer_loop else {
            return Err(ExtrudeError::ExportUnsupported {
                why: format!("face {face_id:?} has no boundary loops"),
            });
        };
        let step_face_id = next_id;
        next_id += 1;
        topology.faces.insert(
            step_face_id,
            StepFace {
                id: step_face_id,
                surface: 0,
                same_sense: true,
                bounds,
                outer_bounds: vec![outer_loop],
            },
        );
        shell_faces.push(step_face_id);
    }

    let shell_id = next_id;
    next_id += 1;
    topology.shells.insert(
        shell_id,
        StepShell {
            id: shell_id,
            open: false,
            faces: shell_faces,
        },
    );
    let solid_id = next_id;
    topology.solids.insert(
        solid_id,
        StepSolid {
            id: solid_id,
            shell: shell_id,
            voids: Vec::new(),
        },
    );
    Ok(topology)
}

/// Converts a planar, polygonal solidcore B-rep into a [`PlanarBody`] for
/// [`nodecraft_step::write_planar_body`].
///
/// This goes through [`planar_brep_to_step_brep`] and
/// [`nodecraft_step::step_brep_to_planar_body`], so it is the same export topology as
/// the `StepBrep` route.
///
/// # Errors
///
/// [`ExtrudeError::ExportUnsupported`] as for [`planar_brep_to_step_brep`];
/// [`ExtrudeError::Export`] when the planar writer's conversion rejects the body.
pub fn planar_brep_to_planar_body(k: &Kernel, brep: &Brep) -> Result<PlanarBody, ExtrudeError> {
    let step_brep = planar_brep_to_step_brep(k, brep)?;
    nodecraft_step::step_brep_to_planar_body(&step_brep)
        .map_err(|message| ExtrudeError::Export { message })
}

/// Writes a planar, polygonal solidcore B-rep as an AP214 STEP `MANIFOLD_SOLID_BREP`.
///
/// This is [`planar_brep_to_step_brep`] followed by
/// [`nodecraft_step::write_step_from_step_brep`].
///
/// # Errors
///
/// [`ExtrudeError::ExportUnsupported`] for a B-rep the planar writer cannot represent;
/// [`ExtrudeError::Export`] for a file write/convert failure.
pub fn write_planar_brep_step(
    k: &Kernel,
    brep: &Brep,
    path: &str,
) -> Result<(), ExtrudeError> {
    let step_brep = planar_brep_to_step_brep(k, brep)?;
    nodecraft_step::write_step_from_step_brep(path, &step_brep)
        .map_err(|message| ExtrudeError::Export { message })
}

/// Extrudes `face` exactly and returns the B-rep plus both STEP-export forms.
///
/// The B-rep is [`crate::extrude_planar_face_exact`]'s result; it must be a planar
/// polygon solid, which is the case for a polygonal face profile.
///
/// # Errors
///
/// Those of [`crate::extrude_planar_face_exact`] and
/// [`planar_brep_to_step_brep`].
pub fn extrude_planar_face_exact_for_export(
    k: &mut Kernel,
    source: &Brep,
    face: FaceId,
    distance: f64,
) -> Result<ExportableExtrude, ExtrudeError> {
    let brep = crate::extrude_planar_face_exact(k, source, face, distance)?;
    let step_brep = planar_brep_to_step_brep(k, &brep)?;
    let planar_body = nodecraft_step::step_brep_to_planar_body(&step_brep)
        .map_err(|message| ExtrudeError::Export { message })?;
    Ok(ExportableExtrude {
        brep,
        step_brep,
        planar_body,
    })
}

/// Extrudes `face` exactly and writes the result to `path` as AP214 STEP.
///
/// Returns the exact B-rep and its neutral export forms so callers can report
/// measures without re-reading the file.
///
/// # Errors
///
/// Those of [`extrude_planar_face_exact_for_export`] and
/// [`nodecraft_step::write_step_from_step_brep`].
pub fn write_exact_extrude_step(
    k: &mut Kernel,
    source: &Brep,
    face: FaceId,
    distance: f64,
    path: &str,
) -> Result<ExportableExtrude, ExtrudeError> {
    let export = extrude_planar_face_exact_for_export(k, source, face, distance)?;
    nodecraft_step::write_step_from_step_brep(path, &export.step_brep)
        .map_err(|message| ExtrudeError::Export { message })?;
    Ok(export)
}

fn ensure_exportable(k: &Kernel, brep: &Brep) -> Result<(), ExtrudeError> {
    if brep.face_list().is_empty() {
        return Err(ExtrudeError::ExportUnsupported {
            why: "B-rep has no faces".to_owned(),
        });
    }
    // Boolean results can retain deleted face records with no face use and no
    // loops. They are not part of the solid and are safe to omit; a live unlisted
    // face that still owns topology is still a hard error.
    let unlisted = brep
        .faces()
        .ids()
        .filter(|face| !brep.face_list().contains(face))
        .filter(|face| {
            brep.faces()
                .get(*face)
                .is_some_and(|face| face.faceuse.is_some() || !face.loops.is_empty())
        })
        .count();
    if unlisted > 0 {
        return Err(ExtrudeError::ExportUnsupported {
            why: format!(
                "B-rep has {unlisted} used face record(s) outside its face list; \
                 export would drop them (use the exact extrude route)"
            ),
        });
    }
    for &face_id in brep.face_list() {
        let surface = face_p2::surface(brep, face_id)
            .map_err(|error| ExtrudeError::solid("topology::face::p2::surface", error))?
            .ok_or_else(|| ExtrudeError::ExportUnsupported {
                why: format!("face {face_id:?} has no surface"),
            })?;
        let tolerance = crate::face_tolerance(k, brep, face_id)?;
        let planar = surface
            .is_planar(k, tolerance)
            .map_err(|error| ExtrudeError::solid("surface::Surface::is_planar", error))?;
        if !planar.is_planar {
            return Err(ExtrudeError::ExportUnsupported {
                why: format!(
                    "face {face_id:?} is not planar (deviation {}); the planar STEP writer \
                     only represents polygonal solids",
                    planar.deviation
                ),
            });
        }
        let loops = face_get_boundary_loops(brep, face_id).map_err(|error| {
            ExtrudeError::solid("query::api_queries::face_get_boundary_loops", error)
        })?;
        for &edge_id in &loops.edges {
            let edge = brep
                .edge(edge_id)
                .map_err(|error| ExtrudeError::solid("topology::Brep::edge", error))?;
            let Some(curve) = edge.curve.as_ref() else {
                return Err(ExtrudeError::ExportUnsupported {
                    why: format!("face {face_id:?} boundary edge {edge_id:?} has no curve"),
                });
            };
            if !is_straight_edge(k, curve, edge.interval, tolerance)? {
                return Err(ExtrudeError::ExportUnsupported {
                    why: format!(
                        "face {face_id:?} boundary edge {edge_id:?} is curved; the planar \
                         STEP writer only represents polygonal loops (curved profile \
                         extrusions need an analytic/NURBS writer)"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Whether the curve is a straight segment over `interval` (its midpoint lies on the
/// chord within tolerance). The exact extrude route represents straight profile edges
/// as two-control-point B-splines, not `Line` curves, so the geometry is checked rather
/// than the curve kind.
fn is_straight_edge(
    k: &Kernel,
    curve: &solidcore::curve::curve::Curve,
    interval: Extent1d,
    tolerance: f64,
) -> Result<bool, ExtrudeError> {
    let [start, _, _] = evaluate_curve(k, Some(curve), interval.min())
        .map_err(|error| ExtrudeError::solid("query::face::p1::evaluate_curve", error))?;
    let [middle, _, _] = evaluate_curve(
        k,
        Some(curve),
        interval.min() + (interval.max() - interval.min()) * 0.5,
    )
    .map_err(|error| ExtrudeError::solid("query::face::p1::evaluate_curve", error))?;
    let [end, _, _] = evaluate_curve(k, Some(curve), interval.max())
        .map_err(|error| ExtrudeError::solid("query::face::p1::evaluate_curve", error))?;
    let direction = end - start;
    let length = direction.length();
    if !length.is_finite() || length <= tolerance {
        return Ok(false);
    }
    let distance = (middle - start).cross(direction).length() / length;
    Ok(distance.is_finite() && distance <= tolerance)
}

fn export_vertex(
    topology: &mut StepBrep,
    next_id: &mut u32,
    vertex_of: &mut BTreeMap<[i64; 3], u32>,
    point: Point3,
) -> u32 {
    let key = quantize(point);
    if let Some(&id) = vertex_of.get(&key) {
        return id;
    }
    let id = *next_id;
    *next_id += 1;
    let point = [point.x, point.y, point.z];
    topology.points.insert(id, point);
    topology.vertices.insert(id, StepVertex { id, point });
    vertex_of.insert(key, id);
    id
}

fn quantize(point: Point3) -> [i64; 3] {
    [
        (point.x * 1.0e7).round() as i64,
        (point.y * 1.0e7).round() as i64,
        (point.z * 1.0e7).round() as i64,
    ]
}
