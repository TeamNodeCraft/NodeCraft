//! Write an AP214 STEP solid from the neutral [`StepBrep`] topology.
//!
//! [`crate::write_planar_body`] already emits STEP from a [`PlanarBody`]; this module
//! bridges the *topology-first* neutral model (`StepBrep`, as produced by
//! [`crate::read_topology`] / [`crate::read_topology_file`]) to that planar writer, so
//! an arbitrary planar topology graph can be exported without round-tripping through
//! solidcore.
//!
//! # What is preserved
//!
//! * solid → shell → face → bound/loop → oriented edge → vertex order, keeping the
//!   shell's face order and each face's bound (outer loop, then holes) order;
//! * edge and bound orientation flags are applied when walking a loop, so the emitted
//!   loop is the same oriented boundary as the source;
//! * a `FACE_OUTER_BOUND` becomes the outer loop; when a face has no outer bound the
//!   first `FACE_BOUND` is used (the STEP convention, matching
//!   [`crate::read_planar_solids`]);
//! * shared vertices are welded by position (1e-7 quantization), loops are validated
//!   as closed chains, and faces are validated as coplanar before writing.
//!
//! # Limits / blockers (why this is planar-only)
//!
//! [`StepBrep`] keeps *references* (`face.surface`, `edge.curve`) as numeric entity
//! ids but not the entity table that gives them names or geometry. Therefore the
//! writer cannot reproduce the original surface/curve names or entity ids: every face
//! is re-emitted as a `PLANE` through its loop and every edge as a `LINE` between its
//! vertices. Consequences:
//!
//! * **Curved geometry is not representable.** Cylindrical/conical/spherical/toroidal
//!   surfaces, B-spline surfaces and curved edges have no geometry in `StepBrep`; a
//!   face whose loops are not coplanar is rejected with an error instead of being
//!   silently faceted. Curved STEP solids must be imported through solidcore and
//!   exported with the analytic writers ([`crate::write_cylinder`] etc.).
//! * **Units are not carried.** `StepBrep` stores raw file coordinates; this writer
//!   emits a millimetre file. Use [`write_step_from_step_brep_scaled`] when the source
//!   file is not millimetres (25.4 for inch, 10.0 for centimetre, 0.001 for micrometre;
//!   the same factors as [`crate::read_planar_body`]'s unit handling).
//! * **One solid per file.** Multi-solid assemblies are rejected, and
//!   `BREP_WITH_VOIDS` solids are rejected because [`PlanarBody`] cannot represent a
//!   void shell. An open shell without a solid is rejected as well (sheets should use
//!   [`crate::write_open_shell`] with a `PlanarBody`).
//! * **`face.same_sense` is not used.** The boundary point order defines the emitted
//!   oriented face, which is the same convention as [`crate::read_planar_solids`]; the
//!   regenerated `PLANE` normal follows the loop winding and the face is written `.T.`.
//! * **Original ids/names are not preserved.** Only topology and points are carried.
//!
//! # Example
//!
//! ```no_run
//! use nodecraft_step::brep_export::write_step_from_step_brep;
//! use nodecraft_step::read_topology_file;
//!
//! let brep = read_topology_file("input.stp")?;
//! write_step_from_step_brep("output.step", &brep)?;
//! # Ok::<(), String>(())
//! ```

use std::collections::BTreeMap;

use crate::topology::{StepBrep, StepFace, StepShell};
use crate::{write_planar_body, PlanarBody};

/// Converts the single planar solid in `body` to a [`PlanarBody`], assuming the points
/// are already in the writer's millimetre units.
///
/// See [`step_brep_to_planar_body_scaled`] for foreign units and the module docs for
/// the exact structural guarantees and limits.
pub fn step_brep_to_planar_body(body: &StepBrep) -> Result<PlanarBody, String> {
    step_brep_to_planar_body_scaled(body, 1.0)
}

/// Converts the single planar solid in `body` to a [`PlanarBody`], scaling every point
/// by `scale` on the way out.
///
/// `scale` is applied after welding, so a scaled round-trip keeps the same topology as
/// an unscaled one. Faces are rejected unless all of their loops are coplanar within
/// `1e-6 * diagonal` (the same tolerance as [`crate::write_planar_body`]).
///
/// # Errors
///
/// Rejects an empty brep, multiple solids, `BREP_WITH_VOIDS`, an open/ambiguous shell,
/// missing topology references, loops that are not closed edge chains, loops with
/// fewer than three vertices, and non-coplanar faces.
pub fn step_brep_to_planar_body_scaled(body: &StepBrep, scale: f64) -> Result<PlanarBody, String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!("planar STEP export scale must be finite and positive, got {scale}"));
    }
    let (shell, owner) = select_planar_shell(body)?;
    if shell.faces.is_empty() {
        return Err(format!("{owner} has no faces"));
    }

    let mut planar = PlanarBody::default();
    let mut lookup: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    for face_id in &shell.faces {
        let face = body
            .faces
            .get(face_id)
            .ok_or_else(|| format!("{owner} references missing face #{face_id}"))?;
        if face.bounds.is_empty() {
            return Err(format!("face #{} has no bounds", face.id));
        }

        let mut loops: Vec<(Vec<usize>, bool)> = Vec::with_capacity(face.bounds.len());
        let mut outer_bounds = 0usize;
        for (loop_id, bound_orientation) in &face.bounds {
            let is_outer = face.outer_bounds.contains(loop_id);
            if is_outer {
                outer_bounds += 1;
            }
            let polygon = loop_polygon(
                body,
                face,
                *loop_id,
                *bound_orientation,
                &mut planar.points,
                &mut lookup,
                scale,
            )?;
            loops.push((polygon, is_outer));
        }
        if outer_bounds > 1 {
            return Err(format!(
                "face #{} has {outer_bounds} FACE_OUTER_BOUND loops; STEP allows at most one",
                face.id
            ));
        }
        // STEP permits a face whose outer boundary is a plain FACE_BOUND; in that case
        // the first bound in the entity list is the outer loop (same fallback as
        // read_planar_solids).
        let outer_index = loops.iter().position(|(_, is_outer)| *is_outer).unwrap_or(0);
        let (outer, _) = loops.remove(outer_index);
        let holes: Vec<Vec<usize>> = loops.into_iter().map(|(polygon, _)| polygon).collect();

        validate_planar_face(face.id, &planar.points, &outer, &holes)?;
        planar.push_face(outer, holes);
    }

    if planar.faces.is_empty() {
        return Err(format!("{owner} produced no planar faces"));
    }
    Ok(planar)
}

/// Writes `body` to `path` as a single AP214 planar `MANIFOLD_SOLID_BREP`, with points
/// interpreted as millimetres.
///
/// This is [`step_brep_to_planar_body`] followed by [`crate::write_planar_body`]; see the
/// module docs for what is preserved and which B-reps are unsupported.
pub fn write_step_from_step_brep(path: &str, body: &StepBrep) -> Result<(), String> {
    write_step_from_step_brep_scaled(path, body, 1.0)
}

/// Writes `body` to `path` with every point multiplied by `scale` (see
/// [`step_brep_to_planar_body_scaled`]).
///
/// Use this to keep a foreign-unit STEP file in millimetres, e.g. `scale = 25.4` for an
/// inch file, `scale = 10.0` for centimetres, `scale = 0.001` for micrometres.
pub fn write_step_from_step_brep_scaled(path: &str, body: &StepBrep, scale: f64) -> Result<(), String> {
    let planar = step_brep_to_planar_body_scaled(body, scale)?;
    write_planar_body(path, &planar)
}

/// Picks the one shell this writer can serialise.
fn select_planar_shell(body: &StepBrep) -> Result<(&StepShell, String), String> {
    if body.solids.is_empty() {
        if body.shells.is_empty() {
            return Err("STEP B-rep is empty: no solids and no shells".into());
        }
        if body.shells.len() > 1 {
            return Err(format!(
                "STEP B-rep has {} shells but no MANIFOLD_SOLID_BREP; the planar writer needs one closed shell",
                body.shells.len()
            ));
        }
        let (shell_id, shell) = body.shells.iter().next().expect("checked non-empty");
        if shell.open {
            return Err(format!(
                "shell #{shell_id} is an OPEN_SHELL; the planar solid writer needs a closed shell (use write_open_shell with a PlanarBody for sheets)"
            ));
        }
        return Ok((shell, format!("shell #{shell_id}")));
    }
    if body.solids.len() > 1 {
        return Err(format!(
            "STEP B-rep has {} MANIFOLD_SOLID_BREP solids; the planar writer emits one solid per file",
            body.solids.len()
        ));
    }
    let solid = body.solids.values().next().expect("checked non-empty");
    if !solid.voids.is_empty() {
        return Err(format!(
            "solid #{} has {} void shell(s); PlanarBody cannot represent BREP_WITH_VOIDS",
            solid.id,
            solid.voids.len()
        ));
    }
    let shell = body
        .shells
        .get(&solid.shell)
        .ok_or_else(|| format!("solid #{} references missing shell #{}", solid.id, solid.shell))?;
    if shell.open {
        return Err(format!(
            "solid #{} references OPEN_SHELL #{}; the planar solid writer needs a closed shell",
            solid.id, solid.shell
        ));
    }
    Ok((shell, format!("solid #{}", solid.id)))
}

/// Walks one face bound into a welded polygon of point indices, applying the edge and
/// bound orientation flags.
fn loop_polygon(
    body: &StepBrep,
    face: &StepFace,
    loop_id: u32,
    bound_orientation: bool,
    points: &mut Vec<[f64; 3]>,
    lookup: &mut BTreeMap<[i64; 3], usize>,
    scale: f64,
) -> Result<Vec<usize>, String> {
    let loop_ = body
        .loops
        .get(&loop_id)
        .ok_or_else(|| format!("face #{} references missing loop #{loop_id}", face.id))?;
    if loop_.oriented_edges.len() < 3 {
        return Err(format!(
            "loop #{loop_id} on face #{} has {} edge(s); a planar loop needs at least three",
            face.id,
            loop_.oriented_edges.len()
        ));
    }

    // Resolve each oriented edge into a directed vertex pair, then require the pieces to
    // form exactly one closed chain. This catches broken EDGE_LOOP references before we
    // emit a misleading polygon.
    let mut directed: Vec<(u32, u32)> = Vec::with_capacity(loop_.oriented_edges.len());
    for oriented_edge in &loop_.oriented_edges {
        let edge = body.edges.get(&oriented_edge.edge).ok_or_else(|| {
            format!(
                "loop #{loop_id} on face #{} references missing edge #{}",
                face.id, oriented_edge.edge
            )
        })?;
        let (start, end) = if oriented_edge.orientation {
            (edge.start, edge.end)
        } else {
            (edge.end, edge.start)
        };
        if start == end {
            return Err(format!(
                "loop #{loop_id} on face #{} uses degenerate edge #{} (both ends are vertex #{start})",
                face.id, edge.id
            ));
        }
        directed.push((start, end));
    }
    for position in 0..directed.len() {
        let (_, end) = directed[position];
        let (next_start, _) = directed[(position + 1) % directed.len()];
        if end != next_start {
            return Err(format!(
                "loop #{loop_id} on face #{} is not a closed chain: edge {} ends at vertex #{end} but the next edge starts at vertex #{next_start}",
                face.id,
                body.edges[&loop_.oriented_edges[position].edge].id
            ));
        }
    }

    let mut polygon: Vec<usize> = Vec::with_capacity(directed.len());
    for (start, _) in &directed {
        let vertex = body.vertices.get(start).ok_or_else(|| {
            format!(
                "loop #{loop_id} on face #{} references missing vertex #{start}",
                face.id
            )
        })?;
        let point = [
            vertex.point[0] * scale,
            vertex.point[1] * scale,
            vertex.point[2] * scale,
        ];
        let key = quantize(point);
        let index = *lookup.entry(key).or_insert_with(|| {
            points.push(point);
            points.len() - 1
        });
        if polygon.last().copied() != Some(index) {
            polygon.push(index);
        }
    }
    // A bound orientation of .F. means the bound's loop is reversed relative to the
    // face normal; reversing the walked points reproduces that handedness.
    if !bound_orientation {
        polygon.reverse();
    }
    if polygon.len() < 3 {
        return Err(format!(
            "loop #{loop_id} on face #{} has fewer than three distinct vertices after welding",
            face.id
        ));
    }
    Ok(polygon)
}

/// Validates that every loop of a face lies in the plane of its outer loop.
fn validate_planar_face(
    face_id: u32,
    points: &[[f64; 3]],
    outer: &[usize],
    holes: &[Vec<usize>],
) -> Result<(), String> {
    let normal = newell_normal(points, outer)
        .ok_or_else(|| format!("face #{face_id} outer loop is degenerate"))?;
    let origin = points[outer[0]];

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let diagonal = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    if !(diagonal > 0.0 && diagonal.is_finite()) {
        return Err("planar body has zero extent".into());
    }
    let tolerance = diagonal * 1e-6;

    for (loop_index, loop_points) in std::iter::once(outer).chain(holes.iter().map(Vec::as_slice)).enumerate() {
        for &index in loop_points {
            let point = points[index];
            let offset = (point[0] - origin[0]) * normal[0]
                + (point[1] - origin[1]) * normal[1]
                + (point[2] - origin[2]) * normal[2];
            if offset.abs() > tolerance {
                return Err(format!(
                    "face #{face_id} loop {loop_index} is not planar: point {index} is {offset} off the outer plane (tolerance {tolerance})"
                ));
            }
        }
        if loop_index > 0 && newell_normal(points, loop_points).is_none() {
            return Err(format!("face #{face_id} inner loop {loop_index} is degenerate"));
        }
    }
    Ok(())
}

fn quantize(point: [f64; 3]) -> [i64; 3] {
    [
        (point[0] * 1e7).round() as i64,
        (point[1] * 1e7).round() as i64,
        (point[2] * 1e7).round() as i64,
    ]
}

/// Unit normal of a closed planar loop (Newell, translation-stabilized).
fn newell_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Option<[f64; 3]> {
    let origin = points[*loop_points.first()?];
    let mut normal = [0.0f64; 3];
    for position in 0..loop_points.len() {
        let a = sub(points[loop_points[position]], origin);
        let b = sub(points[loop_points[(position + 1) % loop_points.len()]], origin);
        normal = add(normal, cross(a, b));
    }
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    Some([normal[0] / length, normal[1] / length, normal[2] / length])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{
        StepEdge, StepFace, StepLoop, StepOrientedEdge, StepShell, StepSolid, StepVertex,
    };
    use crate::{read_planar_solids, read_topology_file, write_planar_body};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEMP_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_path(tag: &str) -> PathBuf {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "sn-brep-export-{}-{tag}-{counter}.step",
            std::process::id()
        ))
    }

    fn fixture(relative: &str) -> Option<PathBuf> {
        let mut roots = Vec::new();
        if let Ok(root) = std::env::var("CARGO_MANIFEST_DIR") {
            roots.push(PathBuf::from(root));
        }
        // When the module is compiled standalone for verification (rustc --test with an
        // absolute #[path]), file!() is absolute and its grandparent is the crate dir.
        let source = Path::new(file!());
        if source.is_absolute() {
            if let Some(crate_dir) = source.parent().and_then(Path::parent) {
                roots.push(crate_dir.to_path_buf());
            }
        }
        if let Ok(current) = std::env::current_dir() {
            roots.push(current.clone());
            roots.push(current.join("rust/crates/nodecraft-step"));
        }
        for root in roots {
            let candidate = root.join("../../../tests").join(relative);
            if candidate.exists() {
                return Some(candidate);
            }
            let candidate = root.join("tests").join(relative);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
    }

    /// An axis-aligned box [-1, 1]^3 with a complete STEP-like topology:
    /// 8 points/vertices, 4 directed edges per face, one loop per face, one closed
    /// shell and one solid. Loop windings are outward.
    fn box_brep() -> StepBrep {
        let coords = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let mut brep = StepBrep::default();
        let mut point_ids = Vec::new();
        let mut vertex_ids = Vec::new();
        for (index, point) in coords.iter().enumerate() {
            let point_id = 100 + index as u32;
            brep.points.insert(point_id, *point);
            let vertex_id = 200 + index as u32;
            brep.vertices.insert(
                vertex_id,
                StepVertex {
                    id: vertex_id,
                    point: *point,
                },
            );
            point_ids.push(point_id);
            vertex_ids.push(vertex_id);
        }
        assert_eq!(point_ids.len(), coords.len());

        // Outward loop windings (right-hand rule pointing away from the box centre).
        let cycles: [[usize; 4]; 6] = [
            [0, 3, 2, 1], // z = -1
            [4, 5, 6, 7], // z = +1
            [0, 1, 5, 4], // y = -1
            [1, 2, 6, 5], // x = +1
            [2, 3, 7, 6], // y = +1
            [3, 0, 4, 7], // x = -1
        ];
        let mut faces = Vec::new();
        for (face_index, cycle) in cycles.iter().enumerate() {
            let mut oriented_edges = Vec::with_capacity(4);
            for position in 0..4 {
                let start = vertex_ids[cycle[position]];
                let end = vertex_ids[cycle[(position + 1) % 4]];
                let edge_id = 300 + brep.edges.len() as u32;
                brep.edges.insert(
                    edge_id,
                    StepEdge {
                        id: edge_id,
                        start,
                        end,
                        curve: 0,
                        same_sense: true,
                    },
                );
                oriented_edges.push(StepOrientedEdge {
                    edge: edge_id,
                    orientation: true,
                });
            }
            let loop_id = 400 + face_index as u32;
            brep.loops.insert(
                loop_id,
                StepLoop {
                    id: loop_id,
                    oriented_edges,
                },
            );
            let face_id = 500 + face_index as u32;
            brep.faces.insert(
                face_id,
                StepFace {
                    id: face_id,
                    surface: 900 + face_index as u32,
                    same_sense: true,
                    bounds: vec![(loop_id, true)],
                    outer_bounds: vec![loop_id],
                },
            );
            faces.push(face_id);
        }
        brep.shells.insert(
            600,
            StepShell {
                id: 600,
                open: false,
                faces,
            },
        );
        brep.solids.insert(
            700,
            StepSolid {
                id: 700,
                shell: 600,
                voids: Vec::new(),
            },
        );
        brep
    }

    fn push_cycle(points: &mut Vec<[f64; 3]>, coords: &[[f64; 3]; 4]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(4);
        for point in coords {
            let index = points.iter().position(|candidate| candidate == point).unwrap_or_else(|| {
                points.push(*point);
                points.len() - 1
            });
            indices.push(index);
        }
        indices
    }

    /// 4x4x1 slab with a 1x1 square through-hole: 10 faces, two of them with a hole.
    fn slab_with_hole() -> PlanarBody {
        let mut body = PlanarBody::default();
        let bottom_outer = push_cycle(
            &mut body.points,
            &[[-2.0, -2.0, -0.5], [2.0, -2.0, -0.5], [2.0, 2.0, -0.5], [-2.0, 2.0, -0.5]],
        );
        let top_outer = push_cycle(
            &mut body.points,
            &[[-2.0, -2.0, 0.5], [2.0, -2.0, 0.5], [2.0, 2.0, 0.5], [-2.0, 2.0, 0.5]],
        );
        let bottom_inner = push_cycle(
            &mut body.points,
            &[[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
        );
        let top_inner = push_cycle(
            &mut body.points,
            &[[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
        );

        // bottom (outward -Z) with reversed hole loop; top (outward +Z) with a hole.
        body.push_face(
            reverse(&bottom_outer),
            vec![bottom_inner.clone()],
        );
        body.push_face(
            top_outer.clone(),
            vec![reverse(&top_inner)],
        );
        // outer sides
        body.push_face(
            vec![bottom_outer[0], bottom_outer[1], top_outer[1], top_outer[0]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_outer[1], bottom_outer[2], top_outer[2], top_outer[1]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_outer[2], bottom_outer[3], top_outer[3], top_outer[2]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_outer[3], bottom_outer[0], top_outer[0], top_outer[3]],
            Vec::new(),
        );
        // hole walls (outward normals point into the hole)
        body.push_face(
            vec![bottom_inner[0], top_inner[0], top_inner[1], bottom_inner[1]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_inner[1], top_inner[1], top_inner[2], bottom_inner[2]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_inner[2], top_inner[2], top_inner[3], bottom_inner[3]],
            Vec::new(),
        );
        body.push_face(
            vec![bottom_inner[3], top_inner[3], top_inner[0], bottom_inner[0]],
            Vec::new(),
        );
        body
    }

    fn reverse(loop_points: &[usize]) -> Vec<usize> {
        loop_points.iter().rev().copied().collect()
    }

    #[test]
    fn box_brep_converts_and_round_trips_through_step() {
        let brep = box_brep();
        let planar = step_brep_to_planar_body(&brep).unwrap();
        assert_eq!(planar.faces.len(), 6);
        assert_eq!(planar.points.len(), 8);
        assert!(planar.holes.iter().all(Vec::is_empty));
        assert!((planar.volume() - 8.0).abs() < 1e-9, "volume {}", planar.volume());

        let path = temp_path("box");
        write_step_from_step_brep(path.to_str().unwrap(), &brep).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("MANIFOLD_SOLID_BREP"), "written file is not a solid");
        assert!(text.contains("PLANE("));
        assert!(text.contains("LINE("));

        let bodies = read_planar_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].faces.len(), 6);
        assert!((bodies[0].volume() - 8.0).abs() < 1e-9);

        let topology = read_topology_file(path.to_str().unwrap()).unwrap();
        assert_eq!(topology.faces.len(), 6);
        assert_eq!(topology.loops.len(), 6);
        assert_eq!(topology.vertices.len(), 8);
        assert_eq!(topology.edges.len(), 12);
        assert!(topology.validate().is_empty(), "{:?}", topology.validate());
        cleanup(&path);
    }

    #[test]
    fn holes_and_face_order_are_preserved() {
        let source = slab_with_hole();
        assert_eq!(source.faces.len(), 10);
        let first = temp_path("hole-source");
        write_planar_body(first.to_str().unwrap(), &source).unwrap();

        let brep = read_topology_file(first.to_str().unwrap()).unwrap();
        let converted = step_brep_to_planar_body(&brep).unwrap();
        assert_eq!(converted.faces.len(), source.faces.len());
        for face in 0..source.faces.len() {
            assert_eq!(
                converted.face_holes(face).len(),
                source.face_holes(face).len(),
                "face {face} hole count changed"
            );
        }
        assert_eq!(converted.holes.iter().filter(|holes| !holes.is_empty()).count(), 2);
        assert!((converted.volume().abs() - 15.0).abs() < 1e-9, "volume {}", converted.volume());

        let second = temp_path("hole-roundtrip");
        write_step_from_step_brep(second.to_str().unwrap(), &brep).unwrap();
        let bodies = read_planar_solids(second.to_str().unwrap()).unwrap();
        assert_eq!(bodies[0].faces.len(), 10);
        assert!((bodies[0].volume().abs() - 15.0).abs() < 1e-9);
        assert!(bodies[0].holes.iter().any(|holes| holes.len() == 1));
        cleanup(&first);
        cleanup(&second);
    }

    #[test]
    fn scaled_writer_emits_millimetres() {
        let brep = box_brep();
        let path = temp_path("scaled");
        write_step_from_step_brep_scaled(path.to_str().unwrap(), &brep, 25.4).unwrap();
        let bodies = read_planar_solids(path.to_str().unwrap()).unwrap();
        let (min, max) = bodies[0].bounds().unwrap();
        for axis in 0..3 {
            assert!((min[axis] + 25.4).abs() < 1e-9, "min {axis} {}", min[axis]);
            assert!((max[axis] - 25.4).abs() < 1e-9, "max {axis} {}", max[axis]);
        }
        assert!((bodies[0].volume() - 8.0 * 25.4f64.powi(3)).abs() < 1e-6);
        cleanup(&path);
    }

    #[test]
    fn bound_orientation_flips_the_written_loop() {
        let mut brep = box_brep();
        let base = step_brep_to_planar_body(&brep).unwrap();
        // Flip the first face's only bound to .F.
        brep.faces.get_mut(&500).unwrap().bounds[0].1 = false;
        let flipped = step_brep_to_planar_body(&brep).unwrap();
        let mut expected = base.faces[0].clone();
        expected.reverse();
        assert_eq!(flipped.faces[0], expected);
    }

    #[test]
    fn face_without_outer_bound_uses_the_first_bound() {
        let mut brep = box_brep();
        brep.faces.get_mut(&500).unwrap().outer_bounds.clear();
        let planar = step_brep_to_planar_body(&brep).unwrap();
        assert_eq!(planar.faces.len(), 6);
        assert!(planar.holes.iter().all(Vec::is_empty));
    }

    #[test]
    fn rejects_unsupported_breps() {
        assert!(step_brep_to_planar_body(&StepBrep::default())
            .unwrap_err()
            .contains("empty"));

        let mut multiple = box_brep();
        multiple.solids.insert(
            701,
            StepSolid {
                id: 701,
                shell: 600,
                voids: Vec::new(),
            },
        );
        assert!(step_brep_to_planar_body(&multiple)
            .unwrap_err()
            .contains("2 MANIFOLD_SOLID_BREP solids"));

        let mut voids = box_brep();
        voids.solids.get_mut(&700).unwrap().voids.push(600);
        assert!(step_brep_to_planar_body(&voids)
            .unwrap_err()
            .contains("void shell"));

        let mut open = box_brep();
        open.solids.clear();
        open.shells.get_mut(&600).unwrap().open = true;
        assert!(step_brep_to_planar_body(&open)
            .unwrap_err()
            .contains("OPEN_SHELL"));

        let mut ambiguous = box_brep();
        ambiguous.solids.clear();
        ambiguous.shells.insert(
            601,
            StepShell {
                id: 601,
                open: false,
                faces: vec![500],
            },
        );
        assert!(step_brep_to_planar_body(&ambiguous)
            .unwrap_err()
            .contains("no MANIFOLD_SOLID_BREP"));
    }

    #[test]
    fn rejects_broken_topology() {
        let mut missing_loop = box_brep();
        missing_loop.faces.get_mut(&500).unwrap().bounds[0].0 = 999;
        assert!(step_brep_to_planar_body(&missing_loop)
            .unwrap_err()
            .contains("missing loop #999"));

        let mut unclosed = box_brep();
        unclosed.loops.get_mut(&400).unwrap().oriented_edges.pop();
        assert!(step_brep_to_planar_body(&unclosed)
            .unwrap_err()
            .contains("not a closed chain"));

        let mut short_loop = box_brep();
        short_loop.loops.get_mut(&400).unwrap().oriented_edges.truncate(2);
        assert!(step_brep_to_planar_body(&short_loop)
            .unwrap_err()
            .contains("at least three"));

        let mut two_outers = box_brep();
        let face = two_outers.faces.get_mut(&500).unwrap();
        face.bounds.push(face.bounds[0]);
        face.outer_bounds.push(face.outer_bounds[0]);
        assert!(step_brep_to_planar_body(&two_outers)
            .unwrap_err()
            .contains("FACE_OUTER_BOUND"));

        let mut non_coplanar = box_brep();
        non_coplanar.vertices.get_mut(&200).unwrap().point[2] += 0.4;
        assert!(step_brep_to_planar_body(&non_coplanar)
            .unwrap_err()
            .contains("not planar"));

        let mut bad_scale = box_brep();
        assert!(step_brep_to_planar_body_scaled(&bad_scale, 0.0).is_err());
        bad_scale.solids.clear();
        assert!(write_step_from_step_brep_scaled(
            temp_path("bad-scale").to_str().unwrap(),
            &bad_scale,
            -1.0
        )
        .is_err());
    }

    #[test]
    fn real_block_fixture_round_trips() {
        let Some(source) = fixture("Sample AP203 Data Files/block.stp") else {
            return; // fixture corpus is optional in a bare checkout
        };
        let expected = crate::read_planar_body(source.to_str().unwrap()).unwrap();
        let brep = read_topology_file(source.to_str().unwrap()).unwrap();

        let path = temp_path("block");
        write_step_from_step_brep(path.to_str().unwrap(), &brep).unwrap();
        let bodies = read_planar_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].faces.len(), expected.faces.len());
        let expected_volume = expected.volume().abs();
        assert!(
            (bodies[0].volume().abs() - expected_volume).abs() <= expected_volume.max(1.0) * 1e-9,
            "volume {} vs {}",
            bodies[0].volume(),
            expected_volume
        );
        cleanup(&path);
    }
}
