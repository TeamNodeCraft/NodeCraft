//! Display import for STEP sheet/shell models made of planar polygonal faces.
//!
//! `socks.stp` is a CATIA/StepCode "short name" file: its representation is a
//! `FACETED_BREP` over a closed shell of 768 `FACE_SURFACE`/`PLANE` faces with
//! `POLY_LOOP` boundaries (all planar polygons), and it has no `MANIFOLD_SOLID_BREP`.
//! The standard readers key off long Part 21 names, so `read_faceted_breps` and
//! `read_topology` see nothing and Stepio currently reports `no-solid-topology`.
//!
//! This module reads that sheet/shell geometry directly and emits a **precise display
//! mesh** (planar polygons ear-clipped exactly, per-face flat normals, boundary edge
//! loops). It also understands the Part 21 short entity names used by the legacy
//! CATIA/StepCode fixtures (`FCTBR` = `FACETED_BREP`, `CLSSHL` = `CLOSED_SHELL`,
//! `FCSRF` = `FACE_SURFACE`, `PLYLP` = `POLY_LOOP`, `CRTPNT` = `CARTESIAN_POINT`, ...)
//! and resolves the length unit from the representation's unit context, so an
//! inch-based model scales to millimetres exactly like the rest of Stepio.
//!
//! The result is display-only: there is no exact B-rep and no mass properties, because
//! the file has no solid topology. Curved sheet faces (B-spline, cylinder, ...) are
//! **not** approximated here: they produce a precise classification error instead.
//!
//! # Scope / integration
//!
//! Entry point: [`import_sheet`]. It returns `Ok(None)` when the file has no sheet
//! geometry this module understands, so it is safe to try from a dispatch fallback
//! (see the notes at the bottom of this file for the exact `lib.rs` insertion point).
//!
//! Only shell/body geometry is imported. Other surface-model entities in the same file
//! (for example the auxiliary B-spline surfaces in `socks.stp`) are recorded as a
//! warning and ignored, because the faceted shell is the display representation.

use std::collections::{BTreeMap, BTreeSet};

use nodecraft_protocol::{BrepEdge, Mesh};
use nodecraft_step::{Entity, PlanarBody, StepFile, Value};

/// Diagnostics `kind` reported for a sheet/shell display import.
pub const SHEET_KIND: &str = "sheet-brep";

/// A precise display import of a planar STEP sheet/shell model.
///
/// Several fields are diagnostic detail for tests and the next dispatch layer (mass
/// properties, exact-B-rep classification); the current `lib.rs` wire only needs the
/// mesh, kind, body count and warnings.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SheetImport {
    /// Display mesh in millimetres, per-face flat normals, lightweight.
    pub mesh: Mesh,
    /// Diagnostics kind, always [`SHEET_KIND`].
    pub kind: &'static str,
    /// Number of sheet bodies (FACETED_BREP / loose shells) imported.
    pub bodies: usize,
    /// Number of planar faces imported.
    pub faces: usize,
    /// Number of face bounds (outer loops + holes).
    pub loops: usize,
    /// Number of welded source vertices used by the mesh.
    pub vertices: usize,
    /// Number of triangles in the display mesh.
    pub triangles: usize,
    /// Closed shells imported.
    pub closed_shells: usize,
    /// Open shells imported.
    pub open_shells: usize,
    /// Surface type histogram over the imported faces (always `PLANE` today).
    pub surface_kinds: BTreeMap<String, usize>,
    /// Maximum distance (mm) of a loop point from its face's best-fit plane.
    ///
    /// Faceted shells may tessellate a curved skin, so their polygons are only
    /// approximately planar. The mesh uses the original 3D points; this value records
    /// how far the triangulation's planarity assumption is from exact.
    pub max_planarity_deviation: f64,
    /// Auxiliary surface entities present outside the imported shells (ignored).
    pub ignored_surface_entities: usize,
    /// Non-fatal notes for diagnostics.
    pub warnings: Vec<String>,
}

/// Part 21 short entity name -> the long name this reader keys on.
///
/// The mapping is intentionally small: only the topology/surface/unit entities a
/// planar sheet file needs. Unknown names are returned unchanged, so canonical files
/// work without any conversion.
fn canonical_name(name: &str) -> &str {
    match name {
        "FCTBR" => "FACETED_BREP",
        "CLSSHL" => "CLOSED_SHELL",
        "OPSHL" => "OPEN_SHELL",
        "FCSRF" => "FACE_SURFACE",
        "ADVNCFC" => "ADVANCED_FACE",
        "FCOTBN" => "FACE_OUTER_BOUND",
        "FCBND" => "FACE_BOUND",
        "PLYLP" => "POLY_LOOP",
        "EDGLP" => "EDGE_LOOP",
        "ORNEDG" => "ORIENTED_EDGE",
        "EDGCRV" => "EDGE_CURVE",
        "VTXPNT" => "VERTEX_POINT",
        "CRTPNT" => "CARTESIAN_POINT",
        "GUAC" => "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        "CNBSUN" => "CONVERSION_BASED_UNIT",
        "LNGUNT" => "LENGTH_UNIT",
        "SUNT" => "SI_UNIT",
        "NMDUNT" => "NAMED_UNIT",
        // Curved surface/curve entities are not meshed here, but naming them precisely
        // keeps the classification messages actionable.
        "BSSWK" => "B_SPLINE_SURFACE_WITH_KNOTS",
        "BSCWK" => "B_SPLINE_CURVE_WITH_KNOTS",
        "RCTRSR" => "RECTANGULAR_TRIMMED_SURFACE",
        "SL" => "SURFACE_OF_LINEAR_EXTRUSION",
        other => other,
    }
}

/// Whether an entity is (or complexly includes) `canonical`, honouring short names.
fn has_type(entity: &Entity, canonical: &str) -> bool {
    entity
        .complex_members()
        .iter()
        .any(|member| canonical_name(member) == canonical)
}

/// Human-readable type of an entity for diagnostics.
fn type_name(entity: &Entity) -> String {
    let members = entity.complex_members();
    if members.is_empty() {
        "<unknown>".into()
    } else {
        members.iter().map(|member| canonical_name(member)).collect::<Vec<_>>().join("+")
    }
}

fn entity<'a>(file: &'a StepFile, id: u32) -> Result<&'a Entity, String> {
    file.entities.get(&id).ok_or_else(|| format!("missing STEP entity #{id}"))
}

fn reference(value: &Value) -> Option<u32> {
    match value {
        Value::Ref(id) => Some(*id),
        _ => None,
    }
}

fn references(value: &Value) -> Vec<u32> {
    match value {
        Value::Ref(id) => vec![*id],
        Value::List(values) => values.iter().filter_map(reference).collect(),
        _ => Vec::new(),
    }
}

fn first_reference(entity: &Entity, index: usize) -> Option<u32> {
    entity.args.get(index).and_then(|value| references(value).into_iter().next())
}

fn bool_arg(value: &Value) -> Option<bool> {
    match value {
        Value::Enum(text) if text == "T" => Some(true),
        Value::Enum(text) if text == "F" => Some(false),
        _ => None,
    }
}

fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => Some(*number),
        _ => None,
    }
}

fn point_coordinates(file: &StepFile, id: u32) -> Result<[f64; 3], String> {
    let point = entity(file, id)?;
    if !has_type(point, "CARTESIAN_POINT") {
        return Err(format!("expected a CARTESIAN_POINT, found {} (#{id})", type_name(point)));
    }
    let Some(Value::List(values)) = point.args.get(1) else {
        return Err(format!("CARTESIAN_POINT #{id} has no coordinate list"));
    };
    let mut coordinates = [0.0; 3];
    for (slot, value) in coordinates.iter_mut().zip(values.iter()) {
        *slot = numeric(value).ok_or_else(|| format!("CARTESIAN_POINT #{id} has a non-numeric coordinate"))?;
    }
    Ok(coordinates)
}

/// Millimetres per model unit.
///
/// Thin compatibility wrapper; the unit logic lives in [`unit_scale_mm`].
fn model_scale(file: &StepFile, text: &str) -> f64 {
    unit_scale_mm(file, text).0
}

/// Single unit-scale entry point for Stepio.
///
/// The scale comes from `nodecraft-step`'s structured resolver: the representation's
/// `GLOBAL_UNIT_ASSIGNED_CONTEXT` (`GUAC`) is authoritative and conversion-based units
/// are resolved recursively (`CONVERSION_BASED_UNIT('METRE', #measure)` →
/// `LENGTH_MEASURE_WITH_UNIT(1.0, SI metre)` → 1000 mm/unit). A declared unit that is
/// not linked from a context is still used. When no declared unit can be resolved,
/// millimetres are assumed and `warning` is `Some` so callers can put it in
/// diagnostics instead of silently scaling by 1.
pub fn unit_scale_mm(file: &StepFile, text: &str) -> (f64, Option<String>) {
    let resolution = nodecraft_step::units::resolve_length_unit_from_file(file);
    let mut scale = resolution.millimetres_per_unit;
    let mut warning = resolution.warning;
    if matches!(
        resolution.source,
        nodecraft_step::units::UnitSource::AssumedMillimetres
    ) {
        // Padding-tolerant last resort for files whose unit entities the structural
        // parser cannot resolve (for example unusual short-name or partial files).
        if let Some((fallback_scale, description)) = nodecraft_step::units::length_scale_from_text(text) {
            scale = fallback_scale;
            warning = Some(format!(
                "length unit recovered from the file text ({description}); using {fallback_scale} mm per unit"
            ));
        }
    }
    (scale, warning)
}

/// A planar loop resolved to model (millimetre) coordinates.
struct ResolvedLoop {
    points: Vec<[f64; 3]>,
    orientation: bool,
    outer: bool,
}

fn resolve_loop(
    file: &StepFile,
    loop_id: u32,
    scale: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let loop_entity = entity(file, loop_id)?;
    if has_type(loop_entity, "POLY_LOOP") {
        let Some(Value::List(values)) = loop_entity.args.get(1) else {
            return Err(format!("POLY_LOOP #{loop_id} has no point list"));
        };
        let mut points = Vec::new();
        for value in values {
            let id = reference(value).ok_or_else(|| format!("POLY_LOOP #{loop_id} has a non-reference entry"))?;
            let point = point_coordinates(file, id)?;
            points.push([point[0] * scale, point[1] * scale, point[2] * scale]);
        }
        return Ok(points);
    }
    if has_type(loop_entity, "EDGE_LOOP") {
        let Some(Value::List(values)) = loop_entity.args.get(1) else {
            return Err(format!("EDGE_LOOP #{loop_id} has no edge list"));
        };
        // Directed vertex pairs from each oriented edge; then require a closed chain.
        let mut directed: Vec<([f64; 3], [f64; 3])> = Vec::new();
        for value in values {
            let use_id = reference(value).ok_or_else(|| format!("EDGE_LOOP #{loop_id} has a non-reference entry"))?;
            let use_entity = entity(file, use_id)?;
            if !has_type(use_entity, "ORIENTED_EDGE") {
                return Err(format!("EDGE_LOOP #{loop_id} entry #{use_id} is {}, expected ORIENTED_EDGE", type_name(use_entity)));
            }
            let edge_id = first_reference(use_entity, 3).ok_or_else(|| format!("ORIENTED_EDGE #{use_id} has no edge reference"))?;
            let orientation = use_entity.args.get(4).and_then(bool_arg).unwrap_or(true);
            let edge_entity = entity(file, edge_id)?;
            if !has_type(edge_entity, "EDGE_CURVE") {
                return Err(format!("ORIENTED_EDGE #{use_id} references {}, expected EDGE_CURVE", type_name(edge_entity)));
            }
            let start_id = first_reference(edge_entity, 1).ok_or_else(|| format!("EDGE_CURVE #{edge_id} has no start vertex"))?;
            let end_id = first_reference(edge_entity, 2).ok_or_else(|| format!("EDGE_CURVE #{edge_id} has no end vertex"))?;
            let vertex_point = |vertex_id: u32| -> Result<[f64; 3], String> {
                let vertex = entity(file, vertex_id)?;
                let point_id = first_reference(vertex, 1).ok_or_else(|| format!("VERTEX_POINT #{vertex_id} has no point"))?;
                let point = point_coordinates(file, point_id)?;
                Ok([point[0] * scale, point[1] * scale, point[2] * scale])
            };
            let (start, end) = if orientation {
                (vertex_point(start_id)?, vertex_point(end_id)?)
            } else {
                (vertex_point(end_id)?, vertex_point(start_id)?)
            };
            directed.push((start, end));
        }
        if directed.len() < 3 {
            return Err(format!("EDGE_LOOP #{loop_id} has fewer than three edges"));
        }
        let mut points = Vec::with_capacity(directed.len());
        for position in 0..directed.len() {
            let (start, end) = directed[position];
            let (next_start, _) = directed[(position + 1) % directed.len()];
            if end != next_start {
                return Err(format!("EDGE_LOOP #{loop_id} is not a closed chain"));
            }
            points.push(start);
        }
        return Ok(points);
    }
    Err(format!(
        "sheet loop #{loop_id} uses {}; only POLY_LOOP/EDGE_LOOP boundaries are supported",
        type_name(loop_entity)
    ))
}

/// Resolve the bounds of one face into loops (outer first, then holes).
fn resolve_face_loops(
    file: &StepFile,
    face: &Entity,
    face_id: u32,
    scale: f64,
) -> Result<Vec<ResolvedLoop>, String> {
    let Some(Value::List(bounds)) = face.args.get(1) else {
        return Err(format!("sheet face #{face_id} has no bound list"));
    };
    let mut loops = Vec::new();
    for value in bounds {
        let bound_id = reference(value).ok_or_else(|| format!("sheet face #{face_id} has a non-reference bound"))?;
        let bound = entity(file, bound_id)?;
        let outer = if has_type(bound, "FACE_OUTER_BOUND") {
            true
        } else if has_type(bound, "FACE_BOUND") {
            false
        } else {
            return Err(format!("sheet face #{face_id} bound #{bound_id} is {}, expected FACE_OUTER_BOUND/FACE_BOUND", type_name(bound)));
        };
        let loop_id = first_reference(bound, 1).ok_or_else(|| format!("bound #{bound_id} has no loop reference"))?;
        let orientation = bound.args.get(2).and_then(bool_arg).unwrap_or(true);
        loops.push(ResolvedLoop { points: resolve_loop(file, loop_id, scale)?, orientation, outer });
    }
    if loops.iter().filter(|loop_| loop_.outer).count() > 1 {
        return Err(format!("sheet face #{face_id} has more than one FACE_OUTER_BOUND"));
    }
    // STEP permits a plain FACE_BOUND as the outer loop; the first bound is then outer.
    if !loops.iter().any(|loop_| loop_.outer) {
        if let Some(first) = loops.first_mut() {
            first.outer = true;
        }
    }
    if loops.is_empty() {
        return Err(format!("sheet face #{face_id} has no bounds"));
    }
    Ok(loops)
}

fn dedupe_loop(mut points: Vec<[f64; 3]>) -> Result<Vec<[f64; 3]>, String> {
    let key = |point: [f64; 3]| [point[0].to_bits(), point[1].to_bits(), point[2].to_bits()];
    let mut deduped: Vec<[f64; 3]> = Vec::with_capacity(points.len());
    for point in points.drain(..) {
        if deduped.last().map(|last| key(*last)) != Some(key(point)) {
            deduped.push(point);
        }
    }
    if deduped.len() > 1 && key(deduped[0]) == key(*deduped.last().expect("non-empty")) {
        deduped.pop();
    }
    if deduped.len() < 3 {
        return Err("sheet loop has fewer than three distinct vertices".into());
    }
    Ok(deduped)
}

fn newell_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Option<[f64; 3]> {
    let origin = points[*loop_points.first()?];
    let mut normal = [0.0f64; 3];
    for position in 0..loop_points.len() {
        let a = [points[loop_points[position]][0] - origin[0], points[loop_points[position]][1] - origin[1], points[loop_points[position]][2] - origin[2]];
        let b = [points[loop_points[(position + 1) % loop_points.len()]][0] - origin[0], points[loop_points[(position + 1) % loop_points.len()]][1] - origin[1], points[loop_points[(position + 1) % loop_points.len()]][2] - origin[2]];
        normal = [
            normal[0] + a[1] * b[2] - a[2] * b[1],
            normal[1] + a[2] * b[0] - a[0] * b[2],
            normal[2] + a[0] * b[1] - a[1] * b[0],
        ];
    }
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !length.is_finite() || length <= 1e-15 {
        return None;
    }
    Some([normal[0] / length, normal[1] / length, normal[2] / length])
}

fn unit(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !length.is_finite() || length <= 1e-15 {
        return None;
    }
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
}

/// One sheet body: a shell's planar polygons plus metadata.
struct SheetBody {
    body: PlanarBody,
    face_ids: Vec<u32>,
    closed: bool,
    loop_count: usize,
    max_planarity_deviation: f64,
}

fn push_welded(
    body: &mut PlanarBody,
    lookup: &mut BTreeMap<[i64; 3], usize>,
    point: [f64; 3],
) -> usize {
    let key = [
        (point[0] * 1e7).round() as i64,
        (point[1] * 1e7).round() as i64,
        (point[2] * 1e7).round() as i64,
    ];
    *lookup.entry(key).or_insert_with(|| {
        body.points.push(point);
        body.points.len() - 1
    })
}

fn shell_faces(file: &StepFile, shell_id: u32) -> Result<Vec<u32>, String> {
    let shell = entity(file, shell_id)?;
    if !has_type(shell, "CLOSED_SHELL") && !has_type(shell, "OPEN_SHELL") {
        return Err(format!("sheet shell #{shell_id} is {}, expected CLOSED_SHELL/OPEN_SHELL", type_name(shell)));
    }
    let Some(Value::List(values)) = shell.args.get(1) else {
        return Err(format!("sheet shell #{shell_id} has no face list"));
    };
    Ok(values.iter().filter_map(reference).collect())
}

/// Resolve one shell into a planar `PlanarBody`, rejecting non-planar/unsupported faces
/// with a precise message instead of approximating them.
fn sheet_body(file: &StepFile, shell_id: u32, scale: f64, surface_kinds: &mut BTreeMap<String, usize>) -> Result<SheetBody, String> {
    let faces = shell_faces(file, shell_id)?;
    if faces.is_empty() {
        return Err(format!("sheet shell #{shell_id} has no faces"));
    }
    let mut body = PlanarBody::default();
    let mut lookup: BTreeMap<[i64; 3], usize> = BTreeMap::new();
    let mut face_ids = Vec::with_capacity(faces.len());
    let mut loop_count = 0usize;
    let mut non_planar = Vec::new();
    for face_id in &faces {
        let face = entity(file, *face_id)?;
        if !has_type(face, "FACE_SURFACE") && !has_type(face, "ADVANCED_FACE") && !has_type(face, "FACE") {
            return Err(format!("sheet shell #{shell_id} contains non-face entity #{face_id} ({})", type_name(face)));
        }
        // FACE_SURFACE/ADVANCED_FACE: (name, bounds, surface, same_sense). Plain FACE:
        // (name, bounds) with no surface, i.e. a polygonal face; treat as planar.
        if let Some(surface_id) = first_reference(face, 2) {
            let surface = entity(file, surface_id)?;
            let kind = type_name(surface);
            *surface_kinds.entry(kind.clone()).or_default() += 1;
            if !has_type(surface, "PLANE") {
                non_planar.push(format!("face #{face_id} uses {kind}"));
                continue;
            }
        } else {
            // A FACE without a surface entity is a polygonal (planar) face.
            *surface_kinds.entry("FACE".into()).or_default() += 1;
        }
        let loops = resolve_face_loops(file, face, *face_id, scale)?;
        let mut polygons: Vec<(Vec<usize>, bool)> = Vec::new();
        for resolved in loops {
            loop_count += 1;
            let mut points = dedupe_loop(resolved.points)?;
            if !resolved.orientation {
                points.reverse();
            }
            let indices: Vec<usize> = points.into_iter().map(|point| push_welded(&mut body, &mut lookup, point)).collect();
            // A loop that collapses during welding (duplicate vertices) is malformed.
            if indices.len() < 3 {
                return Err(format!("sheet face #{face_id} has a loop with fewer than three distinct vertices"));
            }
            polygons.push((indices, resolved.outer));
        }
        let outer_index = polygons.iter().position(|(_, outer)| *outer).unwrap_or(0);
        let (outer, _) = polygons.remove(outer_index);
        let holes: Vec<Vec<usize>> = polygons.into_iter().map(|(indices, _)| indices).collect();
        body.push_face(outer, holes);
        face_ids.push(*face_id);
    }
    if !non_planar.is_empty() {
        let kinds: BTreeSet<String> = surface_kinds.keys().cloned().collect();
        return Err(format!(
            "sheet shell #{shell_id} has {} non-planar face(s) ({}); surface kinds: {}; the planar sheet display path cannot represent it exactly",
            non_planar.len(),
            non_planar.iter().take(4).cloned().collect::<Vec<_>>().join(", "),
            kinds.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if body.faces.is_empty() {
        return Err(format!("sheet shell #{shell_id} has no planar faces"));
    }

    // Coplanarity gate: triangulation projects onto the loop plane, so a non-planar
    // loop would be silently flattened. Reject it with a precise message instead.
    let diagonal = {
        let (min, max) = body.bounds().ok_or_else(|| format!("sheet shell #{shell_id} has no points"))?;
        ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt()
    };
    if !(diagonal > 0.0 && diagonal.is_finite()) {
        return Err(format!("sheet shell #{shell_id} has zero extent"));
    }
    // Faceted shells frequently tessellate a curved skin, so their polygons are only
    // approximately planar. Allow 10 ppm of the model diagonal (socks: 0.75 um over a
    // 173 mm diagonal) and record the actual deviation; the emitted triangles always use
    // the original 3D points. Beyond that bound the face is not a usable facet.
    let tolerance = (diagonal * 1e-5).max(1e-9);
    let mut max_deviation = 0.0f64;
    for face in 0..body.faces.len() {
        let outer = &body.faces[face];
        let normal = newell_normal(&body.points, outer).ok_or_else(|| format!("sheet face #{} outer loop is degenerate", face_ids[face]))?;
        let origin = body.points[outer[0]];
        for loop_points in body.loops(face) {
            for &index in loop_points {
                let point = body.points[index];
                let offset = (point[0] - origin[0]) * normal[0] + (point[1] - origin[1]) * normal[1] + (point[2] - origin[2]) * normal[2];
                max_deviation = max_deviation.max(offset.abs());
                if offset.abs() > tolerance {
                    return Err(format!(
                        "sheet face #{} is not planar: point {index} is {offset} off the outer plane (tolerance {tolerance:.6} mm)",
                        face_ids[face]
                    ));
                }
            }
        }
    }

    let declared_closed = has_type(entity(file, shell_id)?, "CLOSED_SHELL");
    let mut valid_closed = false;
    if declared_closed && body.validate_closed().is_ok() {
        valid_closed = true;
        // A consistently oriented closed shell with negative signed volume is wound
        // inward; reverse every loop once (the centroid heuristic in
        // PlanarBody::repair_orientation is wrong for non-convex shells).
        if body.try_volume().map(|volume| volume < 0.0).unwrap_or(false) {
            for face in &mut body.faces {
                face.reverse();
            }
            for holes in &mut body.holes {
                for hole in holes {
                    hole.reverse();
                }
            }
        }
    }
    Ok(SheetBody { body, face_ids, closed: valid_closed, loop_count, max_planarity_deviation: max_deviation })
}

/// Build a display mesh (per-face flat normals, boundary edge loops) from a sheet body.
fn body_mesh(body: &SheetBody) -> Result<Mesh, String> {
    let mut mesh = Mesh::default();
    let mut edge_index = 0usize;
    for face in 0..body.body.faces.len() {
        let normal = newell_normal(&body.body.points, &body.body.faces[face])
            .ok_or_else(|| format!("sheet face #{} is degenerate", body.face_ids[face]))?;
        let normal = unit(normal).ok_or_else(|| format!("sheet face #{} has a degenerate normal", body.face_ids[face]))?;
        let normal32 = [normal[0] as f32, normal[1] as f32, normal[2] as f32];
        // One mesh vertex per distinct loop vertex of this face (flat shading).
        let mut local: BTreeMap<usize, u32> = BTreeMap::new();
        let mut loops: Vec<Vec<u32>> = Vec::new();
        for loop_points in body.body.loops(face) {
            let mut indices = Vec::with_capacity(loop_points.len());
            for &index in loop_points {
                let vertex = *local.entry(index).or_insert_with(|| {
                    let point = body.body.points[index];
                    mesh.vertices.push([point[0] as f32, point[1] as f32, point[2] as f32]);
                    mesh.normals.push(normal32);
                    mesh.vertices.len() as u32 - 1
                });
                indices.push(vertex);
            }
            loops.push(indices);
        }
        let triangles = body.body.face_triangles(face)?;
        let group = format!("sheet:face:{}", body.face_ids[face]);
        for triangle in triangles {
            let mapped = triangle.map(|index| local[&index]);
            let pa = mesh.vertices[mapped[0] as usize];
            let pb = mesh.vertices[mapped[1] as usize];
            let pc = mesh.vertices[mapped[2] as usize];
            let area = [
                (pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]),
                (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]),
                (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]),
            ];
            // Skip only truly degenerate triangles; face_triangles already validated
            // the polygon and its area budget.
            if area[0] == 0.0 && area[1] == 0.0 && area[2] == 0.0 {
                continue;
            }
            mesh.faces.push(mapped);
            mesh.face_groups.push(group.clone());
        }
        for indices in loops {
            let segments = (0..indices.len())
                .map(|index| [indices[index], indices[(index + 1) % indices.len()]])
                .collect();
            mesh.brep_edges.push(BrepEdge {
                id: format!("sheet-edge:{edge_index}"),
                kind: "sheet".into(),
                segments,
            });
            edge_index += 1;
        }
    }
    mesh.lightweight = true;
    Ok(mesh)
}

/// Identify the sheet bodies in a STEP file.
///
/// * `FACETED_BREP` (including short `FCTBR`) bodies are sheet targets.
/// * When there is no solid topology, loose `CLOSED_SHELL`/`OPEN_SHELL` shells are
///   sheet targets too.
fn sheet_targets(file: &StepFile) -> Result<Vec<(u32, bool)>, String> {
    let mut targets = Vec::new();
    for candidate in file.entities.values() {
        if !has_type(candidate, "FACETED_BREP") {
            continue;
        }
        let shell_id = first_reference(candidate, 1)
            .ok_or_else(|| format!("FACETED_BREP {} has no shell reference", type_name(candidate)))?;
        let shell = entity(file, shell_id)?;
        let closed = has_type(shell, "CLOSED_SHELL");
        if !closed && !has_type(shell, "OPEN_SHELL") {
            return Err(format!("FACETED_BREP #{shell_id} is not a shell ({})", type_name(shell)));
        }
        targets.push((shell_id, closed));
    }
    if !targets.is_empty() {
        return Ok(targets);
    }

    // No faceted brep: only loose shells in files without solid topology are sheets.
    let mut solid_shells = BTreeSet::new();
    let mut has_solids = false;
    for candidate in file.entities.values() {
        if has_type(candidate, "MANIFOLD_SOLID_BREP") || has_type(candidate, "BREP_WITH_VOIDS") {
            has_solids = true;
            if let Some(shell) = first_reference(candidate, 1) {
                solid_shells.insert(shell);
            }
            if let Some(Value::List(voids)) = candidate.args.get(2) {
                for value in voids {
                    if let Some(void) = reference(value) {
                        solid_shells.insert(void);
                    }
                }
            }
        }
    }
    if has_solids {
        return Ok(targets);
    }
    for (id, entity) in &file.entities {
        let closed = has_type(entity, "CLOSED_SHELL");
        let open = has_type(entity, "OPEN_SHELL");
        if (closed || open) && !solid_shells.contains(id) {
            targets.push((*id, closed));
        }
    }
    targets.sort_unstable();
    Ok(targets)
}

/// Surface entities present in the file but outside the imported shells, grouped by
/// canonical kind. This is the precise "not represented" side of a sheet import.
fn auxiliary_surface_kinds(file: &StepFile, used_faces: &BTreeSet<u32>) -> BTreeMap<String, usize> {
    let mut referenced_surfaces = BTreeSet::new();
    for face_id in used_faces {
        if let Some(face) = file.entities.get(face_id) {
            if let Some(surface) = first_reference(face, 2) {
                referenced_surfaces.insert(surface);
            }
        }
    }
    let surface_kinds = [
        "PLANE",
        "CYLINDRICAL_SURFACE",
        "CONICAL_SURFACE",
        "SPHERICAL_SURFACE",
        "TOROIDAL_SURFACE",
        "B_SPLINE_SURFACE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
        "RATIONAL_B_SPLINE_SURFACE",
        "SURFACE_OF_REVOLUTION",
        "SURFACE_OF_LINEAR_EXTRUSION",
        "RECTANGULAR_TRIMMED_SURFACE",
        "CURVE_BOUNDED_SURFACE",
        "OFFSET_SURFACE",
    ];
    let mut kinds = BTreeMap::new();
    for (id, entity) in &file.entities {
        if referenced_surfaces.contains(id) {
            continue;
        }
        for kind in surface_kinds {
            if has_type(entity, kind) {
                *kinds.entry(kind.to_string()).or_insert(0) += 1;
                break;
            }
        }
    }
    kinds
}

/// Try to import a STEP sheet/shell model as a precise display mesh.
///
/// Returns:
/// * `Ok(Some(import))` — every face of every sheet body is a planar polygon and the
///   mesh was built exactly (millimetres, per-face flat normals, boundary edges);
/// * `Ok(None)` — the file has no sheet geometry this module understands (no
///   FACETED_BREP and no loose shells without solid topology);
/// * `Err(message)` — sheet geometry exists but cannot be represented exactly (curved
///   faces, non-polygonal loops, non-coplanar loops, ...). The message names the shell,
///   face and surface kinds involved.
///
/// The mesh is display-only (`lightweight = true`): the file has no solid topology, so
/// no exact B-rep or mass properties are claimed.
pub fn import_sheet(path: &str) -> Result<Option<SheetImport>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: {error}"))?;
    let targets = sheet_targets(&file)?;
    if targets.is_empty() {
        return Ok(None);
    }
    let (scale, unit_warning) = unit_scale_mm(&file, &text);
    let mut surface_kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut closures: Vec<SheetBody> = Vec::new();
    let mut warnings = Vec::new();
    if let Some(warning) = unit_warning {
        warnings.push(warning);
    }
    for (shell_id, closed) in &targets {
        let body = sheet_body(&file, *shell_id, scale, &mut surface_kinds)?;
        if body.closed != *closed {
            warnings.push(format!("shell #{shell_id} is not edge-manifold; imported as a display sheet"));
        }
        closures.push(body);
    }

    let mut used_faces = BTreeSet::new();
    for body in &closures {
        used_faces.extend(body.face_ids.iter().copied());
    }
    let auxiliary_kinds = auxiliary_surface_kinds(&file, &used_faces);
    let ignored: usize = auxiliary_kinds.values().sum();
    if ignored > 0 {
        warnings.push(format!(
            "{ignored} auxiliary surface entity(ies) outside the imported shell(s) were ignored; the planar shell is the display representation"
        ));
    }

    let mut meshes = Vec::with_capacity(closures.len());
    let mut faces = 0usize;
    let mut loops = 0usize;
    let mut vertices = 0usize;
    let mut triangles = 0usize;
    let mut max_planarity_deviation = 0.0f64;
    for body in &closures {
        let mesh = body_mesh(body)?;
        faces += body.body.faces.len();
        loops += body.loop_count;
        vertices += body.body.points.len();
        triangles += mesh.faces.len();
        max_planarity_deviation = max_planarity_deviation.max(body.max_planarity_deviation);
        meshes.push(mesh);
    }
    if max_planarity_deviation > 1e-9 {
        warnings.push(format!(
            "faceted shell has slightly non-planar facets (max {max_planarity_deviation:.6} mm off the face plane); triangles use the original 3D points, so the display mesh is exact point geometry"
        ));
    }
    let mut mesh = Mesh::default();
    for (index, mut part) in meshes.into_iter().enumerate() {
        let offset = mesh.vertices.len() as u32;
        let prefix = format!("body:{index}:");
        mesh.vertices.append(&mut part.vertices);
        mesh.normals.append(&mut part.normals);
        mesh.faces.extend(part.faces.into_iter().map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]));
        mesh.face_groups.extend(part.face_groups.into_iter().map(|group| format!("{prefix}{group}")));
        mesh.brep_edges.extend(part.brep_edges.into_iter().map(|mut edge| {
            edge.id = format!("{prefix}{}", edge.id);
            for segment in &mut edge.segments {
                segment[0] += offset;
                segment[1] += offset;
            }
            edge
        }));
    }
    mesh.lightweight = true;
    if mesh.faces.is_empty() {
        return Err("sheet import produced no triangles".into());
    }
    let closed_shells = targets.iter().filter(|(_, closed)| *closed).count();
    let open_shells = targets.len() - closed_shells;
    warnings.push(format!(
        "display-only sheet import: {faces} planar face(s) from {closed_shells} closed / {open_shells} open shell(s); no solid topology or mass properties"
    ));
    Ok(Some(SheetImport {
        mesh,
        kind: SHEET_KIND,
        bodies: targets.len(),
        faces,
        loops,
        vertices,
        triangles,
        closed_shells,
        open_shells,
        surface_kinds,
        max_planarity_deviation,
        ignored_surface_entities: ignored,
        warnings,
    }))
}

// ---------------------------------------------------------------------------------
// lib.rs dispatch integration point (Step-107)
// ---------------------------------------------------------------------------------
//
// In `solidcore-stepio/src/lib.rs`, `import_step_inner`, inside the
// `match nodecraft_step::read_planar_solids(path)` arm
// `Err(error) if error.contains("no MANIFOLD_SOLID_BREP")`, before
// `import_solidcore_step(path)`:
//
// ```ignore
// match crate::step_sheet::import_sheet(path) {
//     Ok(Some(sheet)) => {
//         let mut diagnostics = StepDiagnostics {
//             kind: sheet.kind.into(),
//             bodies: sheet.bodies,
//             ..StepDiagnostics::default()
//         };
//         diagnostics.mesh_vertices = sheet.mesh.vertices.len();
//         diagnostics.mesh_triangles = sheet.mesh.faces.len();
//         diagnostics.warnings = sheet.warnings;
//         return Ok(StepImport { mesh: sheet.mesh, diagnostics });
//     }
//     Ok(None) => {}
//     Err(sheet_error) => {
//         // Optional: fold `sheet_error` into the final error if solidcore and the
//         // surface-model fallback also fail, so unsupported curved sheets get a
//         // precise classification.
//     }
// }
// ```
//
// `socks.stp` takes this arm (its canonical faceted reader sees no long-name
// `FACETED_BREP`). For files that reach the arm but have curved sheet faces,
// `import_sheet` returns `Err` and dispatch can fall through to the existing
// `surface_model` fallback (gear.stp) instead of approximating.

/// Exposed for dispatch diagnostics: the file's sheet geometry classification without
/// building a mesh. Returns `Ok(None)` when [`import_sheet`] would return `Ok(None)`.
#[allow(dead_code)]
pub fn classify_sheet(path: &str) -> Result<Option<SheetClassification>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: {error}"))?;
    let targets = sheet_targets(&file)?;
    if targets.is_empty() {
        return Ok(None);
    }
    let mut faces = 0usize;
    let mut surface_kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut planar_faces = 0usize;
    for (shell_id, _closed) in &targets {
        let face_ids = shell_faces(&file, *shell_id)?;
        faces += face_ids.len();
        for face_id in &face_ids {
            let face = entity(&file, *face_id)?;
            if let Some(surface_id) = first_reference(face, 2) {
                let surface = entity(&file, surface_id)?;
                let kind = type_name(surface);
                *surface_kinds.entry(kind.clone()).or_default() += 1;
                if has_type(surface, "PLANE") {
                    planar_faces += 1;
                }
            } else {
                *surface_kinds.entry("FACE".into()).or_default() += 1;
                planar_faces += 1;
            }
        }
    }
    let mut used_faces = BTreeSet::new();
    for (shell_id, _closed) in &targets {
        used_faces.extend(shell_faces(&file, *shell_id)?);
    }
    Ok(Some(SheetClassification {
        bodies: targets.len(),
        faces,
        planar_faces,
        surface_kinds,
        auxiliary_surface_kinds: auxiliary_surface_kinds(&file, &used_faces),
        scale: model_scale(&file, &text),
    }))
}

/// Structural classification of a sheet/shell model, without a mesh.
///
/// Public classification surface for dispatch diagnostics; tests use it today and the
/// curved-sheet path can use it when that classification is wired.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct SheetClassification {
    pub bodies: usize,
    pub faces: usize,
    pub planar_faces: usize,
    pub surface_kinds: BTreeMap<String, usize>,
    /// Surface entities outside the shells that this planar path cannot represent.
    pub auxiliary_surface_kinds: BTreeMap<String, usize>,
    /// Millimetres per model unit resolved from the representation unit context.
    pub scale: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEMP_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn fixture(relative: &str) -> Option<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(root) = std::env::var("CARGO_MANIFEST_DIR") {
            roots.push(PathBuf::from(root));
        }
        if let Ok(current) = std::env::current_dir() {
            roots.push(current.clone());
            roots.push(current.join("rust/crates/solidcore-stepio"));
        }
        let source = Path::new(file!());
        if source.is_absolute() {
            if let Some(crate_dir) = source.parent().and_then(Path::parent) {
                roots.push(crate_dir.to_path_buf());
            }
        }
        for root in roots {
            for prefix in ["../../../tests", "tests"] {
                let candidate = root.join(prefix).join(relative);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn temp_step(tag: &str, data: &str) -> PathBuf {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("sn-step-sheet-{}-{tag}-{counter}.step", std::process::id()));
        std::fs::write(&path, format!("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('test'),'1');\nFILE_NAME('','',(''),(''),'','','');\nFILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n")).unwrap();
        path
    }

    fn mesh_bounds(mesh: &Mesh) -> ([f64; 3], [f64; 3]) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis] as f64);
                max[axis] = max[axis].max(vertex[axis] as f64);
            }
        }
        (min, max)
    }

    fn assert_sane(mesh: &Mesh) {
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
        assert_eq!(mesh.faces.len(), mesh.face_groups.len());
        assert!(mesh.lightweight);
        for (index, normal) in mesh.normals.iter().enumerate() {
            let length = ((normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]) as f64).sqrt();
            assert!((length - 1.0).abs() < 1e-5, "normal {index} is not unit: {normal:?}");
        }
        for face in &mesh.faces {
            let points: Vec<[f64; 3]> = face.iter().map(|index| {
                let vertex = mesh.vertices[*index as usize];
                [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]
            }).collect();
            let u = [points[1][0] - points[0][0], points[1][1] - points[0][1], points[1][2] - points[0][2]];
            let v = [points[2][0] - points[0][0], points[2][1] - points[0][1], points[2][2] - points[0][2]];
            let area = ((u[1] * v[2] - u[2] * v[1]).powi(2) + (u[2] * v[0] - u[0] * v[2]).powi(2) + (u[0] * v[1] - u[1] * v[0]).powi(2)).sqrt() * 0.5;
            assert!(area > 1e-12, "degenerate triangle in mesh");
        }
        for edge in &mesh.brep_edges {
            for segment in &edge.segments {
                assert!((segment[0] as usize) < mesh.vertices.len());
                assert!((segment[1] as usize) < mesh.vertices.len());
            }
        }
    }

    /// Step-107 regression: socks.stp is a 768-face planar FACETED_BREP written with
    /// CATIA short entity names; it must import as a precise display sheet, not fail
    /// with no-solid-topology.
    #[test]
    fn socks_imports_as_precise_planar_sheet() {
        let Some(path) = fixture("Sample AP203 Data Files/socks.stp") else { return };
        let import = import_sheet(path.to_str().unwrap()).expect("socks sheet import").expect("socks has sheet geometry");
        assert_eq!(import.kind, SHEET_KIND);
        assert_eq!(import.bodies, 1);
        assert_eq!(import.faces, 768);
        assert_eq!(import.loops, 768);
        assert_eq!(import.closed_shells, 1);
        assert_eq!(import.open_shells, 0);
        assert_eq!(import.surface_kinds.get("PLANE"), Some(&768));
        assert_eq!(import.ignored_surface_entities, 10, "8 B-spline + 1 linear-extrusion + 1 trimmed tooling surface should be classified as ignored");
        assert!(import.warnings.iter().any(|warning| warning.contains("display-only")));
        assert!(import.max_planarity_deviation > 0.0 && import.max_planarity_deviation < 1e-3, "deviation {}", import.max_planarity_deviation);
        assert!(import.warnings.iter().any(|warning| warning.contains("non-planar facets")));
        assert_sane(&import.mesh);
        // 626 quads + 63 pentagons + ... : sum(n - 2) over all 768 polygons.
        assert_eq!(import.triangles, 2000);
        assert_eq!(import.mesh.faces.len(), 2000);
        // The mesh is the faceted shell (the part lump, 1002 welded points); the
        // large auxiliary tooling surfaces are ignored, see the warning assertion.
        assert_eq!(import.vertices, 1002);
        assert_eq!(import.mesh.vertices.len(), 3536);
        // The file's assigned unit is the INCH conversion unit, so mm = 25.4 * coords.
        let (min, max) = mesh_bounds(&import.mesh);
        let raw_min: [f64; 3] = [-1.01016756758, -2.87942620061, -1.22439191476];
        let raw_max: [f64; 3] = [0.880101504438, 2.75953328563, 2.0832993106];
        for axis in 0..3 {
            let want_min = raw_min[axis] * 25.4;
            let want_max = raw_max[axis] * 25.4;
            assert!((min[axis] - want_min).abs() < 1e-6 * want_max.abs().max(1.0), "min[{axis}] {} vs {}", min[axis], want_min);
            assert!((max[axis] - want_max).abs() < 1e-6 * want_max.abs().max(1.0), "max[{axis}] {} vs {}", max[axis], want_max);
        }
        // Closed, consistently oriented shell: signed mesh volume must be positive.
        let mut volume = 0.0;
        for triangle in &import.mesh.faces {
            let point = |index: u32| {
                let vertex = import.mesh.vertices[index as usize];
                [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]
            };
            let a = point(triangle[0]);
            let b = point(triangle[1]);
            let c = point(triangle[2]);
            volume += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        assert!(volume / 6.0 > 0.0, "socks shell is wound inward: volume {}", volume / 6.0);
    }

    #[test]
    fn classifies_socks_without_meshing() {
        let Some(path) = fixture("Sample AP203 Data Files/socks.stp") else { return };
        let classification = classify_sheet(path.to_str().unwrap()).unwrap().expect("sheet classification");
        assert_eq!(classification.bodies, 1);
        assert_eq!(classification.faces, 768);
        assert_eq!(classification.planar_faces, 768);
        assert_eq!(classification.scale, 25.4);
        assert_eq!(classification.surface_kinds.get("PLANE"), Some(&768));
        assert_eq!(classification.auxiliary_surface_kinds.get("B_SPLINE_SURFACE_WITH_KNOTS"), Some(&8));
        assert_eq!(classification.auxiliary_surface_kinds.get("RECTANGULAR_TRIMMED_SURFACE"), Some(&1));
        assert_eq!(classification.auxiliary_surface_kinds.get("SURFACE_OF_LINEAR_EXTRUSION"), Some(&1));
    }

    #[test]
    fn canonical_and_short_names_build_the_same_box_sheet() {
        let short = temp_step("short-box", "\
#20=(LNGUNT()SUNT(.MILLI.,.METRE.));\n\
#1=CRTPNT('',(0.,0.,0.));\n#2=CRTPNT('',(3.,0.,0.));\n#3=CRTPNT('',(3.,2.,0.));\n#4=CRTPNT('',(0.,2.,0.));\n#5=DRCTN('',(0.,0.,1.));\n#6=A2PL3D('',#1,#5,#2);\n#7=PLANE('',#6);\n#8=PLYLP('',(#1,#2,#3,#4));\n#9=FCOTBN('',#8,.T.);\n#10=FCSRF('',(#9),#7,.T.);\n#11=CLSSHL('',(#10));\n#12=FCTBR('sheet',#11);");
        let canonical = temp_step("long-box", "\
#20=(LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));\n\
#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(3.,0.,0.));\n#3=CARTESIAN_POINT('',(3.,2.,0.));\n#4=CARTESIAN_POINT('',(0.,2.,0.));\n#5=DIRECTION('',(0.,0.,1.));\n#6=AXIS2_PLACEMENT_3D('',#1,#5,#2);\n#7=PLANE('',#6);\n#8=POLY_LOOP('',(#1,#2,#3,#4));\n#9=FACE_OUTER_BOUND('',#8,.T.);\n#10=FACE_SURFACE('',(#9),#7,.T.);\n#11=CLOSED_SHELL('',(#10));\n#12=FACETED_BREP('sheet',#11);");
        let short_import = import_sheet(short.to_str().unwrap()).unwrap().expect("short sheet");
        let canonical_import = import_sheet(canonical.to_str().unwrap()).unwrap().expect("canonical sheet");
        assert_eq!(short_import.faces, 1);
        assert_eq!(short_import.mesh.faces.len(), 2, "quad -> two triangles");
        assert_eq!(short_import.mesh.vertices.len(), canonical_import.mesh.vertices.len());
        assert_eq!(short_import.mesh.faces, canonical_import.mesh.faces);
        let (min, max) = mesh_bounds(&short_import.mesh);
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [3.0, 2.0, 0.0]);
        let _ = std::fs::remove_file(short);
        let _ = std::fs::remove_file(canonical);
    }

    #[test]
    fn open_shell_without_solid_is_a_sheet() {
        let path = temp_step("open-shell", "\
#20=(LNGUNT()SUNT(.MILLI.,.METRE.));\n\
#1=CRTPNT('',(0.,0.,0.));\n#2=CRTPNT('',(1.,0.,0.));\n#3=CRTPNT('',(1.,1.,0.));\n#4=CRTPNT('',(0.,1.,0.));\n#5=PLYLP('',(#1,#2,#3,#4));\n#6=FCOTBN('',#5,.T.);\n#7=FCSRF('',(#6));\n#8=OPSHL('',(#7));");
        let import = import_sheet(path.to_str().unwrap()).unwrap().expect("open sheet");
        assert_eq!(import.open_shells, 1);
        assert_eq!(import.closed_shells, 0);
        assert_eq!(import.faces, 1);
        assert_eq!(import.mesh.faces.len(), 2);
        assert_sane(&import.mesh);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn curved_sheet_faces_are_classified_precisely() {
        let path = temp_step("curved", "\
#20=(LNGUNT()SUNT(.MILLI.,.METRE.));\n\
#1=CRTPNT('',(0.,0.,0.));\n#2=CRTPNT('',(1.,0.,0.));\n#3=CRTPNT('',(1.,1.,0.));\n#4=PLYLP('',(#1,#2,#3));\n#5=FCOTBN('',#4,.T.);\n#6=BSSWK('spline');\n#7=FCSRF('',(#5),#6,.T.);\n#8=CLSSHL('',(#7));\n#9=FCTBR('curved',#8);");
        let error = import_sheet(path.to_str().unwrap()).expect_err("curved sheet must not be approximated");
        assert!(error.contains("non-planar face"), "{error}");
        assert!(error.contains("face #7"), "{error}");
        assert!(error.contains("B_SPLINE_SURFACE_WITH_KNOTS"), "{error}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn solid_files_are_not_claimed_as_sheets() {
        let Some(path) = fixture("Sample AP203 Data Files/block.stp") else { return };
        assert!(import_sheet(path.to_str().unwrap()).unwrap().is_none());
    }
    #[test]
    fn unit_scale_resolves_conversion_based_metre_abc_fixtures() {
        for name in ["abc-0000", "abc-0006", "abc-0008", "abc-0035"] {
            let path = Path::new("/tmp/truck-resources/step").join(format!("{name}.step"));
            if !path.exists() {
                eprintln!("skipping {name}: missing {}", path.display());
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let file = StepFile::parse(&text).unwrap();
            let (scale, warning) = unit_scale_mm(&file, &text);
            assert!((scale - 1000.0).abs() < 1e-9, "{name}: {scale} mm/unit, expected 1000 (metre)");
            assert!(warning.is_none(), "{name}: unexpected warning {warning:?}");
        }
    }

    #[test]
    fn unit_scale_resolves_doghouse_inch_fixture() {
        let Some(path) = fixture("Sample AP203 Data Files/doghouse.stp") else { return };
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let (scale, warning) = unit_scale_mm(&file, &text);
        assert!((scale - 25.4).abs() < 1e-9, "doghouse: {scale} mm/unit, expected 25.4");
        assert!(warning.is_none(), "doghouse: unexpected warning {warning:?}");
    }

    #[test]
    fn unit_scale_resolves_occt_cube_millimetres() {
        let path = Path::new("/tmp/truck-resources/step/occt-cube.step");
        if !path.exists() {
            eprintln!("skipping occt-cube: missing {}", path.display());
            return;
        }
        let text = std::fs::read_to_string(path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let (scale, warning) = unit_scale_mm(&file, &text);
        assert!((scale - 1.0).abs() < 1e-12, "occt-cube: {scale} mm/unit, expected 1");
        assert!(warning.is_none(), "occt-cube: unexpected warning {warning:?}");
    }

    #[test]
    fn unit_scale_assumes_millimetres_with_warning_when_missing() {
        let path = temp_step("no-unit", "\
#1=CRTPNT('',(0.,0.,0.));\n#2=CRTPNT('',(1.,0.,0.));\n#3=CRTPNT('',(1.,1.,0.));\n#4=CRTPNT('',(0.,1.,0.));\n#5=PLYLP('',(#1,#2,#3,#4));\n#6=FCOTBN('',#5,.T.);\n#7=FCSRF('',(#6));\n#8=OPSHL('',(#7));");
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let (scale, warning) = unit_scale_mm(&file, &text);
        assert!((scale - 1.0).abs() < 1e-12, "unit-less scale {scale}; expected millimetres");
        let warning = warning.expect("unit-less files must warn");
        assert!(warning.contains("assuming millimetres"), "{warning}");
        let import = import_sheet(path.to_str().unwrap()).unwrap().expect("open sheet");
        assert!(import.warnings.iter().any(|note| note.contains("assuming millimetres")), "{:?}", import.warnings);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unit_scale_resolves_padded_conversion_metre_declaration() {
        let path = temp_step("padded-metre", "\
#20 = ( GUAC( ( #21 ) ) );\n\
#21 = ( CNBSUN( 'METRE', #22 ) LNGUNT( ) NMDUNT( * ) );\n\
#22 = LENGTH_MEASURE_WITH_UNIT( LENGTH_MEASURE( 1.0 ), #23 );\n\
#23 = ( LNGUNT( ) NMDUNT( * ) SUNT( $, .METRE. ) );");
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let (scale, warning) = unit_scale_mm(&file, &text);
        assert!((scale - 1000.0).abs() < 1e-9, "padded metre conversion: {scale} mm/unit");
        assert!(warning.is_none(), "padded metre: unexpected warning {warning:?}");
        let _ = std::fs::remove_file(path);
    }

}
