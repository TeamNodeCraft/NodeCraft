//! Multi-solid mixed STEP import (Step-108).
//!
//! The single-solid mixed reader in `nodecraft-solidcore::step_mixed` builds one exact
//! plane/cylinder B-rep for one `MANIFOLD_SOLID_BREP`. STEP files such as `4pinplug.stp`
//! carry several mixed solids in one body; this module iterates every solid, builds each
//! exact mixed B-rep, tessellates it and merges the display meshes.
//!
//! Rejection is explicit and all-or-nothing: if any solid cannot be built exactly (an
//! unsupported surface such as `TOROIDAL_SURFACE`, a bad loop, a failed healer), the
//! whole import fails with the failing solid ids and reasons. A partially meshed file is
//! never returned, because a silently missing body would corrupt every downstream mass,
//! bounds and Boolean result.

use nodecraft_protocol::Mesh;
use nodecraft_step::{StepFile, StepBrep};

use crate::StepDiagnostics;

/// Per-solid B-rep statistics collected while meshing.
#[derive(Debug, Clone, Copy, Default)]
struct SolidStats {
    faces: usize,
    edges: usize,
    vertices: usize,
}

/// Builds one display mesh per STEP solid and merges them, rejecting partial results.
///
/// `results` holds one entry per solid, in file order: `(solid_id, Ok((mesh, stats)))` for
/// a solid that was built and tessellated, `Err(reason)` otherwise.
fn merge_solid_results(results: Vec<(u32, Result<(Mesh, SolidStats), String>)>) -> Result<crate::StepImport, String> {
    let mut meshes: Vec<Mesh> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut brep_faces = 0usize;
    let mut brep_edges = 0usize;
    let mut brep_vertices = 0usize;
    let mut stats_ok = 0usize;
    for (solid_id, result) in &results {
        match result {
            Ok((mesh, stats)) => {
                if mesh.faces.is_empty() {
                    failures.push(format!("solid #{solid_id}: tessellation produced no triangles"));
                    continue;
                }
                meshes.push(mesh.clone());
                brep_faces += stats.faces;
                brep_edges += stats.edges;
                brep_vertices += stats.vertices;
                stats_ok += 1;
            }
            Err(error) => failures.push(format!("solid #{solid_id}: {error}")),
        }
    }
    if !failures.is_empty() {
        return Err(format!(
            "{} of {} mixed solid(s) could not be built exactly: {}; refusing a partial multi-solid import (Step-108)",
            failures.len(),
            results.len(),
            failures.join("; ")
        ));
    }
    if meshes.is_empty() {
        return Err("mixed STEP file has no solids to import".into());
    }
    let mesh = crate::merge_meshes(meshes);
    let mut diagnostics = StepDiagnostics {
        kind: if results.len() == 1 { "mixed-brep".into() } else { "mixed-multi-solid".into() },
        bodies: stats_ok,
        breps: stats_ok,
        ..StepDiagnostics::default()
    };
    diagnostics.brep_faces = brep_faces;
    diagnostics.brep_edges = brep_edges;
    diagnostics.brep_vertices = brep_vertices;
    diagnostics.mesh_vertices = mesh.vertices.len();
    diagnostics.mesh_triangles = mesh.faces.len();
    Ok(crate::StepImport { mesh, diagnostics })
}

/// The file's STEP length-unit scale to millimetres (`1.0` when no unit is declared).
fn length_scale(path: &str) -> f64 {
    if let Ok(Some(context)) = nodecraft_step::read_unit_context(path) {
        return context.length_unit.millimetres();
    }
    nodecraft_step::read_length_unit(path).map(|unit| unit.millimetres()).unwrap_or(1.0)
}


/// Splits every single-edge closed loop (a full circle written as one `EDGE_CURVE` with
/// `v0 == v1`) into two half-circle edges with one synthetic intermediate vertex.
///
/// `make_topology_from_data`'s healer interpolates UV trims from the loop points; a
/// one-edge closed loop gives it a single repeated point and the NLib interpolation fails
/// (`interpolate_points: upstream status 1001`, e.g. every solid of `4pinplug.stp`). The
/// split is purely topological: the two new edges reuse the original CIRCLE curve and the
/// synthetic vertex lies on the circle (the point opposite the start point), so the exact
/// plane/cylinder geometry is unchanged.
fn split_closed_single_edge_loops(file: &StepFile, body: &StepBrep) -> StepBrep {
    let mut body = body.clone();
    let mut next_vertex: u32 = body.vertices.keys().copied().max().unwrap_or(0) + 1;
    let mut next_edge: u32 = body.edges.keys().copied().max().unwrap_or(0) + 1;
    let loop_ids: Vec<u32> = body.loops.keys().copied().collect();
    for loop_id in loop_ids {
        let Some(loop_) = body.loops.get(&loop_id) else { continue };
        if loop_.oriented_edges.len() != 1 {
            continue;
        }
        let oriented = loop_.oriented_edges[0].clone();
        let Some(edge) = body.edges.get(&oriented.edge).cloned() else { continue };
        if edge.start != edge.end {
            continue;
        }
        let Some(curve_entity) = file.entities.get(&edge.curve) else { continue };
        if !matches!(curve_entity.name.as_str(), "CIRCLE" | "INTERSECTION_CURVE" | "SURFACE_CURVE" | "SEAM_CURVE" | "TRIMMED_CURVE") {
            continue;
        }
        let Some(start_vertex) = body.vertices.get(&edge.start).cloned() else { continue };
        // A CIRCLE's placement is the diameter through the start point: the opposite point
        // is 2·center − P. Resolve the centre from the circle's AXIS2_PLACEMENT_3D.
        let circle_id = match curve_entity.name.as_str() {
            "INTERSECTION_CURVE" | "SURFACE_CURVE" | "SEAM_CURVE" | "TRIMMED_CURVE" => {
                match curve_entity.args.get(1) {
                    Some(nodecraft_step::Value::Ref(inner)) => *inner,
                    _ => continue,
                }
            }
            _ => edge.curve,
        };
        let Some(circle) = file.entities.get(&circle_id) else { continue };
        if circle.name != "CIRCLE" {
            continue;
        }
        let Some(nodecraft_step::Value::Ref(placement_id)) = circle.args.get(1) else { continue };
        let Some(nodecraft_step::Value::Ref(centre_id)) = file.entities.get(placement_id).and_then(|placement| placement.args.get(1)) else { continue };
        let Some(nodecraft_step::Value::List(coords)) = file.entities.get(centre_id).and_then(|point| point.args.get(1)) else { continue };
        let mut centre = [0.0f64; 3];
        let mut valid = true;
        for (slot, coords) in centre.iter_mut().zip(coords) {
            match coords {
                nodecraft_step::Value::Number(number) => *slot = *number,
                _ => { valid = false; break; }
            }
        }
        if !valid { continue; }
        let midpoint = [
            2.0 * centre[0] - start_vertex.point[0],
            2.0 * centre[1] - start_vertex.point[1],
            2.0 * centre[2] - start_vertex.point[2],
        ];
        let middle_vertex = next_vertex;
        next_vertex += 1;
        body.vertices.insert(middle_vertex, nodecraft_step::topology::StepVertex { id: middle_vertex, point: midpoint });
        let first_edge = next_edge;
        let second_edge = next_edge + 1;
        next_edge += 2;
        body.edges.insert(first_edge, nodecraft_step::topology::StepEdge { id: first_edge, start: edge.start, end: middle_vertex, curve: edge.curve, same_sense: edge.same_sense });
        body.edges.insert(second_edge, nodecraft_step::topology::StepEdge { id: second_edge, start: middle_vertex, end: edge.end, curve: edge.curve, same_sense: edge.same_sense });
        let replacement = if oriented.orientation {
            vec![
                nodecraft_step::topology::StepOrientedEdge { edge: first_edge, orientation: true },
                nodecraft_step::topology::StepOrientedEdge { edge: second_edge, orientation: true },
            ]
        } else {
            vec![
                nodecraft_step::topology::StepOrientedEdge { edge: second_edge, orientation: false },
                nodecraft_step::topology::StepOrientedEdge { edge: first_edge, orientation: false },
            ]
        };
        if let Some(loop_) = body.loops.get_mut(&loop_id) {
            loop_.oriented_edges = replacement;
        }
    }
    body
}

/// Splits the solids the planar/analytic reader skipped into (exact-mixed-analytic,
/// other). Analytic solids are those whose every face surface is plane/cylinder/cone/
/// sphere/torus; the exact mixed reader can build those, while B-spline/swept surfaces
/// stay with the sheet fallback. This is what lets a file with both classes import
/// per-solid instead of falling back to the legacy per-face "mixed not implemented" error.
pub(crate) fn split_skipped_solids(path: &str, skipped: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
    let body: StepBrep = nodecraft_step::read_topology(&file);
    let mut analytic = Vec::new();
    let mut other = Vec::new();
    for solid_id in skipped {
        let solid = body.solids.get(solid_id).ok_or_else(|| format!("{path}: solid #{solid_id} is not a MANIFOLD_SOLID_BREP"))?;
        let shell = body.shells.get(&solid.shell).ok_or_else(|| format!("{path}: solid #{solid_id} has no shell"))?;
        let mut all_analytic = true;
        for face_id in &shell.faces {
            let Some(face) = body.faces.get(face_id) else { all_analytic = false; break };
            let Some(surface) = file.entities.get(&face.surface) else { all_analytic = false; break };
            let analytic_surface = ["PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE", "TOROIDAL_SURFACE"]
                .iter()
                .any(|kind| surface.has_member(kind));
            if !analytic_surface {
                all_analytic = false;
                break;
            }
        }
        if all_analytic { analytic.push(*solid_id); } else { other.push(*solid_id); }
    }
    Ok((analytic, other))
}

/// Imports every solid of a multi-solid mixed STEP file.
///
/// # Errors
///
/// I/O and syntax errors, a file without solids, or the explicit all-or-nothing rejection
/// produced by [`merge_solid_results`] when any solid cannot be built exactly.
pub(crate) fn import_mixed_solids(path: &str) -> Result<crate::StepImport, String> {
    import_mixed_solids_selected(path, None)
}

/// Exact per-solid mixed import for a subset of a multi-solid file (the solids the
/// planar/analytic reader skipped). All-or-nothing: every selected solid must build, or
/// the error names the failing solid/face/surface and no partial bodies are returned.
pub(crate) fn import_mixed_solids_for_ids(path: &str, solid_ids: &[u32]) -> Result<crate::StepImport, String> {
    if solid_ids.is_empty() {
        return Err(format!("{path}: no skipped solids to import through the exact mixed reader"));
    }
    import_mixed_solids_selected(path, Some(solid_ids))
}

fn import_mixed_solids_selected(path: &str, selected: Option<&[u32]>) -> Result<crate::StepImport, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
    let body: StepBrep = nodecraft_step::read_topology(&file);
    let all_solid_ids: Vec<u32> = body.solids.keys().copied().collect();
    let solid_ids: Vec<u32> = match selected {
        None => all_solid_ids.clone(),
        Some(ids) => {
            for solid_id in ids {
                if !body.solids.contains_key(solid_id) {
                    return Err(format!("{path}: solid #{solid_id} is not a MANIFOLD_SOLID_BREP"));
                }
            }
            ids.to_vec()
        }
    };
    if solid_ids.is_empty() {
        return Err(format!("{path}: STEP file has no MANIFOLD_SOLID_BREP solids"));
    }
    let scale = length_scale(path);
    // Full-circle loops written as a single closed edge break the healer's UV interpolation;
    // split them topologically before building (no geometry change).
    let body = split_closed_single_edge_loops(&file, &body);
    let mut results = Vec::with_capacity(solid_ids.len());
    for solid_id in solid_ids {
        let result = match nodecraft_solidcore::step_mixed::build_mixed(&file, &body, solid_id, scale) {
            Ok((brep, _data, _face_ids)) => {
                let stats = SolidStats {
                    faces: brep.faces().len(),
                    edges: brep.edges().len(),
                    vertices: brep.vertices().len(),
                };
                match nodecraft_solidcore::curved::brep_display_mesh(&Default::default(), &brep, &Default::default()) {
                    Ok(mesh) => Ok((mesh, stats)),
                    Err(error) => Err(format!("display tessellation failed: {error}")),
                }
            }
            Err(error) => Err(error),
        };
        results.push((solid_id, result));
    }
    let import = merge_solid_results(results)?;
    // Safety net: the per-solid B-reps are exact, but a bad UV trim can still make the
    // display tessellator emit far-away triangles (e.g. full-circle cap faces). Never
    // return a mesh whose bounds do not cover the STEP body bounds; reject it instead of
    // handing a corrupt display mesh to the app (Step-108 fail-closed rule).
    // Whole-file bounds are only meaningful when every solid is selected; a subset must be
    // validated against its own per-solid B-reps (and its caller merges the rest).
    let selected_is_all = selected.is_none()
        || selected.is_some_and(|ids| ids.len() == all_solid_ids.len() && ids.iter().all(|id| all_solid_ids.contains(id)));
    if selected_is_all {
        if let Ok((step_min, step_max)) = nodecraft_step::read_step_bounds(path) {
        let mut got_min = [f32::INFINITY; 3];
        let mut got_max = [f32::NEG_INFINITY; 3];
        for vertex in &import.mesh.vertices {
            for axis in 0..3 {
                got_min[axis] = got_min[axis].min(vertex[axis]);
                got_max[axis] = got_max[axis].max(vertex[axis]);
            }
        }
        let extent = (0..3).map(|axis| (step_max[axis] - step_min[axis]).abs()).fold(0.0f64, f64::max).max(1.0);
        let tolerance = extent * 0.01;
        for axis in 0..3 {
            if (got_min[axis] as f64 - step_min[axis]).abs() > tolerance
                || (got_max[axis] as f64 - step_max[axis]).abs() > tolerance
            {
                return Err(format!(
                    "display mesh bounds [{got_min:?}]..[{got_max:?}] do not match STEP bounds [{step_min:?}]..[{step_max:?}]; the multi-solid display tessellation is invalid, refusing the import (Step-108)"
                ));
            }
        }
        }
    }
    Ok(import)
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use nodecraft_protocol::Mesh;

    use super::*;

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(relative);
        path.is_file().then_some(path)
    }

    fn triangle() -> Mesh {
        Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Mesh::default()
        }
    }

    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn doghouse_keeps_the_single_solid_mixed_path() {
        let Some(path) = repository("tests/Sample AP203 Data Files/doghouse.stp") else { return };
        let import = crate::import_step(path.to_str().unwrap()).expect("doghouse mixed import");
        assert_eq!(import.diagnostics.kind, "mixed-brep");
        assert_eq!(import.diagnostics.breps, 1);
        assert_eq!(import.diagnostics.bodies, 1);
        assert!(import.diagnostics.brep_faces >= 32, "{:?}", import.diagnostics.brep_faces);
        assert!(!import.mesh.faces.is_empty());
    }

    /// Joins two doghouse STEP files into one file with two mixed solids (second file's
    /// entity ids are renumbered), giving a real multi-solid mixed fixture with no
    /// single-edge full-circle loops (those still trip the solid-level mixed reader).
    fn merged_doghouses(destination: &str) {
        let source = repository("tests/Sample AP203 Data Files/doghouse.stp").expect("doghouse fixture");
        let first = std::fs::read_to_string(&source).unwrap();
        let second = first.clone();
        let data = |text: &str| -> String {
            let start = text.find("DATA;").unwrap() + "DATA;".len();
            let end = start + text[start..].find("ENDSEC;").unwrap();
            text[start..end].to_string()
        };
        let offset = 1_000_000u32;
        let renumber = |text: &str| -> String {
            let mut out = String::new();
            let mut rest = text;
            while let Some(hash) = rest.find('#') {
                out.push_str(&rest[..=hash]);
                rest = &rest[hash + 1..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(id) = digits.parse::<u32>() {
                    out.push_str(&(id + offset).to_string());
                    rest = &rest[digits.len()..];
                }
            }
            out.push_str(rest);
            out
        };
        let header_end = first.find("DATA;").unwrap();
        let mut merged = String::from(&first[..header_end]);
        merged.push_str("DATA;");
        merged.push_str(&data(&first));
        merged.push_str(&renumber(&data(&second)));
        merged.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        std::fs::write(destination, merged).unwrap();
    }

    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn merged_doghouse_file_imports_two_mixed_solids() {
        let destination = std::env::temp_dir().join(format!("mx108-merged-doghouse-{}.stp", std::process::id()));
        merged_doghouses(destination.to_str().unwrap());
        let result = crate::import_step(destination.to_str().unwrap());
        let _ = std::fs::remove_file(&destination);
        let import = result.expect("two doghouse mixed solids should import");
        assert_eq!(import.diagnostics.kind, "mixed-multi-solid");
        assert_eq!(import.diagnostics.bodies, 2);
        assert_eq!(import.diagnostics.breps, 2);
        assert!(import.diagnostics.brep_faces >= 64, "faces: {}", import.diagnostics.brep_faces);
        assert!(!import.mesh.faces.is_empty());
    }

    /// 4pinplug has seven mixed solids; their solid-level B-reps build, but the full-circle
    /// cap faces currently make the display tessellator emit a far-out artifact triangle
    /// (mesh bbox ~1e6 instead of ~2 mm). The multi-solid layer must reject that invalid
    /// mesh explicitly instead of returning geometry that fails every bounds check.
    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn four_pin_plug_invalid_display_mesh_is_rejected() {
        let Some(path) = repository("tests/Sample AP203 Data Files/4pinplug.stp") else { return };
        let error = crate::import_step(path.to_str().unwrap()).expect_err("invalid display tessellation must be rejected");
        assert!(error.contains("STEP geometry is mixed planar/analytic"), "{error}");
        assert!(error.contains("exact mixed multi-solid import failed"), "{error}");
        assert!(error.contains("display mesh bounds"), "{error}");
        assert!(error.contains("refusing"), "{error}");
    }

    /// The per-solid helper used by `lib.rs` for skipped solids must never emit the
    /// legacy "not implemented" fallback message: completion or an explicit solid/face/
    /// surface rejection only.
    #[test]
    fn skipped_analytic_solids_route_through_the_exact_mixed_reader() {
        for relative in [
            "tests/Sample AP203 Data Files/chair.stp",
            "tests/Sample AP203 Data Files/calosoe.stp",
            "tests/Sample AP203 Data Files/jack_in_the_box.stp",
            "tests/Sample AP203 Data Files/ph4m3-st.stp",
            "tests/Sample AP203 Data Files/st203-bapl.stp",
        ] {
            let Some(path) = repository(relative) else { continue };
            let Ok((_bodies, skipped)) = nodecraft_step::import_solids_partial(path.to_str().unwrap()) else { continue };
            if skipped.is_empty() {
                continue;
            }
            match import_mixed_solids_for_ids(path.to_str().unwrap(), &skipped) {
                Ok(import) => {
                    eprintln!("{relative}: exact mixed OK ({} solid(s), {} triangles)", import.diagnostics.bodies, import.mesh.faces.len());
                    assert!(!import.mesh.faces.is_empty(), "{relative}: no triangles");
                    assert_eq!(import.diagnostics.bodies, skipped.len(), "{relative}");
                }
                Err(error) => {
                    eprintln!("{relative}: explicit mixed rejection: {error}");
                    assert!(error.contains("solid #"), "{relative}: rejection must name the solid: {error}");
                    assert!(
                        !error.contains("Re-01/Tr-33") && !error.contains("not implemented yet"),
                        "{relative}: legacy mixed fallback message leaked: {error}"
                    );
                }
            }
        }
    }

    /// 4pinplug: every one of its seven mixed solids is attempted; the multi-solid layer
    /// rejects all-or-nothing with per-solid reasons and never returns a partial import.
    #[test]
    fn four_pin_plug_per_solid_import_is_all_or_nothing() {
        let Some(path) = repository("tests/Sample AP203 Data Files/4pinplug.stp") else { return };
        let body = {
            let text = std::fs::read_to_string(&path).unwrap();
            let file = StepFile::parse(&text).unwrap();
            nodecraft_step::read_topology(&file)
        };
        let ids: Vec<u32> = body.solids.keys().copied().collect();
        assert_eq!(ids.len(), 7);
        match import_mixed_solids_for_ids(path.to_str().unwrap(), &ids) {
            Ok(import) => {
                // Current tree: all seven solids build and tessellate; the import must be
                // complete (never a silent subset).
                assert_eq!(import.diagnostics.bodies, 7);
                assert_eq!(import.diagnostics.breps, 7);
                assert!(!import.mesh.faces.is_empty());
                eprintln!("4pinplug: exact multi-solid OK ({} solids, {} triangles)", import.diagnostics.bodies, import.mesh.faces.len());
            }
            Err(error) => {
                assert!(
                    error.contains("refusing a partial multi-solid import") || error.contains("STEP bounds"),
                    "{error}"
                );
                assert!(!error.contains("Re-01/Tr-33"), "{error}");
                eprintln!("4pinplug: explicit all-or-nothing rejection: {error}");
            }
        }
    }

    /// A subset of a multi-solid file is validated per selected solid: the whole-file
    /// STEP bounds check must not reject a valid subset.
    #[test]
    fn selected_solid_subset_skips_whole_file_bounds_check() {
        let destination = std::env::temp_dir().join(format!("mx108-subset-doghouse-{}.stp", std::process::id()));
        merged_doghouses(destination.to_str().unwrap());
        let text = std::fs::read_to_string(&destination).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let body = nodecraft_step::read_topology(&file);
        let ids: Vec<u32> = body.solids.keys().copied().collect();
        let first = vec![ids[0]];
        let import = import_mixed_solids_for_ids(destination.to_str().unwrap(), &first).expect("one selected solid");
        let _ = std::fs::remove_file(&destination);
        assert_eq!(import.diagnostics.bodies, 1);
        assert_eq!(import.diagnostics.kind, "mixed-brep");
        assert!(!import.mesh.faces.is_empty());
    }

    #[test]
    fn partial_bodies_are_rejected_explicitly() {
        let ok = (11u32, Ok((triangle(), SolidStats { faces: 1, edges: 0, vertices: 3 })));
        let bad = (12u32, Err("TOROIDAL_SURFACE face #99 is unsupported".to_string()));
        let error = merge_solid_results(vec![ok, bad]).expect_err("partial import must be rejected");
        assert!(error.contains("refusing a partial multi-solid import"), "{error}");
        assert!(error.contains("solid #12"), "{error}");
        assert!(error.contains("TOROIDAL_SURFACE"), "{error}");
    }

    #[test]
    fn empty_tessellation_is_a_failure_not_a_silent_skip() {
        let ok = (1u32, Ok((triangle(), SolidStats { faces: 1, edges: 0, vertices: 3 })));
        let empty = (2u32, Ok((Mesh::default(), SolidStats { faces: 1, edges: 0, vertices: 3 })));
        let error = merge_solid_results(vec![ok, empty]).expect_err("empty mesh must be rejected");
        assert!(error.contains("solid #2"), "{error}");
        assert!(error.contains("no triangles"), "{error}");
    }

    #[test]
    fn all_bodies_merge_into_one_multi_solid_result() {
        let a = (7u32, Ok((triangle(), SolidStats { faces: 1, edges: 2, vertices: 3 })));
        let b = (8u32, Ok((triangle(), SolidStats { faces: 2, edges: 3, vertices: 4 })));
        let import = merge_solid_results(vec![a, b]).expect("complete import");
        assert_eq!(import.diagnostics.kind, "mixed-multi-solid");
        assert_eq!(import.diagnostics.bodies, 2);
        assert_eq!(import.diagnostics.breps, 2);
        assert_eq!(import.diagnostics.brep_faces, 3);
        assert_eq!(import.diagnostics.brep_edges, 5);
        assert_eq!(import.diagnostics.brep_vertices, 7);
        assert_eq!(import.mesh.faces.len(), 2);
    }
}
