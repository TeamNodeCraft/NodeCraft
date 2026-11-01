//! Placed planar solids from a STEP assembly.
//!
//! `nodecraft_step::read_planar_solids` extracts planar [`PlanarBody`]s per solid in
//! file order, but it knows nothing about the product/assembly tree: which solid belongs
//! to which placed part, or where that part sits in the document. This module joins the
//! two public pieces:
//!
//! 1. [`parse_assembly_from_step_file`] builds the product/assembly tree with local and
//!    accumulated world transforms ([`crate::Node`]),
//! 2. leaf nodes (placed parts) are matched to their shape representation,
//! 3. the representation's **solid** items select the planar bodies returned by
//!    [`nodecraft_step::read_planar_solids`],
//! 4. each selected body is copied and its points moved by the leaf's world transform, so
//!    every returned body is in the shared document frame.
//!
//! `read_planar_solids` does not expose per-solid extraction, so the whole file has to be
//! planar for this path to work. A curved, faceted, voided or mapped-item assembly fails
//! with an explicit [`UNSUPPORTED_ERROR_PREFIX`] error instead of silently dropping
//! geometry. A single unsupported leaf fails the whole call rather than yielding a partial
//! import.
//!
//! # Units
//!
//! `read_planar_solids` converts body points to millimetres; the assembly parser keeps
//! transforms in the STEP file's declared length unit. This module therefore rescales the
//! transform translations to millimetres as well, inferring the file's scale from
//! [`nodecraft_step::read_step_bounds`] against the raw topology points (both are public;
//! the parser does not expose its unit table). [`PlanarInstance::world_transform`] is the
//! transform that was applied to the local body, in that same millimetre frame: applying it
//! to a pre-transform (model-space) body point reproduces the returned body point.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), String> {
//! for instance in step_assemblies::planar::planar_instances("assembly.step")? {
//!     // `instance.body` is in millimetres in the assembly/document frame.
//!     println!("{}: {} faces", instance.name, instance.body.faces.len());
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use nodecraft_step::{
    PlanarBody, StepBrep, StepFile, read_planar_solids, read_step_bounds, read_topology,
};

use crate::model::Representation;
use crate::{Node, Transform, parse_assembly_from_step_file};

/// Prefix of every error caused by geometry outside the planar display slice.
///
/// Callers can distinguish "this assembly is not supported yet" from I/O or structural
/// failures with [`str::starts_with`].
pub const UNSUPPORTED_ERROR_PREFIX: &str = "unsupported: ";

/// One placed planar solid: a leaf instance of the assembly tree applied to one solid of
/// its shape representation.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarInstance {
    /// Instance name from the assembly tree (occurrence/placement name, falling back to
    /// the product name).
    pub name: String,
    /// `PRODUCT` display name of the referenced product.
    pub product_name: String,
    /// Shape representation entity the solid item came from.
    pub representation_id: u32,
    /// `MANIFOLD_SOLID_BREP` / `BREP_WITH_VOIDS` entity id of this solid.
    pub solid_id: u32,
    /// The instance's world transform in the millimetre frame of [`Self::body`].
    pub world_transform: Transform,
    /// Planar B-rep with every point already transformed into the document frame
    /// (millimetres). Geometry is cloned per instance, so repeated parts do not share
    /// points.
    pub body: PlanarBody,
}

/// Build one [`PlanarInstance`] per planar solid in the leaf (placed-part) instances of
/// the STEP assembly at `path`.
///
/// # Errors
///
/// * I/O or syntax errors reading the file,
/// * assembly-structure errors from [`parse_assembly_from_step_file`],
/// * errors starting with [`UNSUPPORTED_ERROR_PREFIX`] when the file's solids are not all
///   planar (curved surfaces, faceted B-reps, void shells, mapped-item-only leaf
///   geometry, or a nodecraft-step planar-extraction limit),
/// * an error starting with [`UNSUPPORTED_ERROR_PREFIX`] when the tree has no leaf
///   instance carrying solid geometry.
pub fn planar_instances(path: &str) -> Result<Vec<PlanarInstance>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file =
        StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
    let assembly =
        parse_assembly_from_step_file(&file).map_err(|error| format!("{path}: {error}"))?;
    let topology = read_topology(&file);
    if let Some(external_error) = external_document_error(path, &file) {
        return Err(external_error);
    }

    // The planar extractor only works on the whole file; any non-planar solid in it makes
    // the whole assembly unsupported. That is deliberate: a partial import could place
    // the wrong parts without warning.
    let bodies =
        read_planar_solids(path).map_err(|error| planar_extraction_error(path, &error))?;
    let solid_bodies = bodies_by_solid_id(&topology, bodies)?;

    let representations: BTreeMap<u32, &Representation> = assembly
        .representations
        .iter()
        .map(|representation| (representation.entity_id, representation))
        .collect();
    let unit_scale_mm = length_unit_scale_mm(path, &topology)?;

    let mut instances = Vec::new();
    let mut unsupported: Vec<String> = Vec::new();
    let mut without_geometry: Vec<String> = Vec::new();

    for leaf in leaf_nodes(&assembly) {
        let Some(representation_id) = leaf.representation_id else {
            without_geometry.push(format!("{} (no shape representation)", leaf.name));
            continue;
        };
        let Some(representation) = representations.get(&representation_id) else {
            without_geometry.push(format!(
                "{} (shape representation #{representation_id} was not parsed)",
                leaf.name
            ));
            continue;
        };
        let selection = select_items(&file, representation);
        if let Some(problem) = selection.problem {
            unsupported.push(format!("{}: {problem}", leaf.name));
            continue;
        }
        if selection.solids.is_empty() {
            without_geometry.push(format!(
                "{} (representation #{representation_id} has no solid items)",
                leaf.name
            ));
            continue;
        }

        let world_transform =
            scaled_transform(&leaf.world_transform, unit_scale_mm).map_err(|error| {
                format!("{path}: cannot scale the transform of instance {}: {error}", leaf.name)
            })?;
        for solid_id in selection.solids {
            if let Some(solid) = topology.solids.get(&solid_id) {
                if !solid.voids.is_empty() {
                    unsupported.push(format!(
                        "{}: solid #{solid_id} has void shells, which the planar importer does not represent",
                        leaf.name
                    ));
                    continue;
                }
            }
            let Some(body) = solid_bodies.get(&solid_id) else {
                return Err(format!(
                    "{path}: internal error: solid #{solid_id} has no planar body"
                ));
            };
            let mut body = body.clone();
            for point in &mut body.points {
                *point = world_transform.transform_point(*point);
            }
            instances.push(PlanarInstance {
                name: leaf.name.clone(),
                product_name: leaf.product_name.clone(),
                representation_id,
                solid_id,
                world_transform,
                body,
            });
        }
    }

    if !unsupported.is_empty() {
        return Err(format!("{UNSUPPORTED_ERROR_PREFIX}{path}: {}", unsupported.join("; ")));
    }
    if instances.is_empty() {
        let detail = if without_geometry.is_empty() {
            "the assembly tree has no leaf instances".to_string()
        } else {
            format!("no leaf instance carries solid geometry: {}", without_geometry.join(", "))
        };
        return Err(format!("{UNSUPPORTED_ERROR_PREFIX}{path}: {detail}"));
    }
    Ok(instances)
}

// ---------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------

/// Which representation items are solid geometry, and why unsupported ones cannot be used.
struct Selection {
    /// Solid entity ids, in representation order.
    solids: Vec<u32>,
    /// Set when the representation contains geometry this slice cannot represent.
    problem: Option<String>,
}

/// Classify a leaf representation's items.
///
/// Placement items (`AXIS2_PLACEMENT_3D`, transformations) are ignored: the placement
/// itself already comes from the assembly tree. `MAPPED_ITEM` geometry is not resolved by
/// the assembly parser, so a representation that relies on it is reported as unsupported
/// rather than rendered at the wrong place.
fn select_items(file: &StepFile, representation: &Representation) -> Selection {
    let mut solids = Vec::new();
    let mut faceted = 0usize;
    let mut mapped = 0usize;
    for &item in &representation.items {
        let Some(entity) = file.entities.get(&item) else {
            continue;
        };
        match entity.name.as_str() {
            "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" => solids.push(item),
            "FACETED_BREP" => faceted += 1,
            "MAPPED_ITEM" => mapped += 1,
            _ => {}
        }
    }
    let problem = if faceted > 0 {
        Some(format!(
            "representation #{} contains faceted B-reps, which are not supported by the planar importer",
            representation.entity_id
        ))
    } else if solids.is_empty() && mapped > 0 {
        Some(format!(
            "representation #{} places its geometry through MAPPED_ITEM, which is not resolved by the assembly parser",
            representation.entity_id
        ))
    } else {
        None
    };
    Selection { solids, problem }
}

/// Leaf nodes of the assembly tree, in pre-order (roots that have no children included).
fn leaf_nodes(assembly: &crate::Assembly) -> Vec<&Node> {
    assembly.walk().into_iter().filter(|node| node.children.is_empty()).collect()
}

// ---------------------------------------------------------------------------------
// Body alignment / units
// ---------------------------------------------------------------------------------

/// Pair the planar bodies returned by `read_planar_solids` with their solid entity ids.
///
/// `read_planar_solids` walks `StepBrep::solids` (a `BTreeMap`, i.e. ascending entity id)
/// and pushes one body per solid in that order, so zipping the sorted ids with the bodies
/// recovers the mapping. The lengths are checked so a future change to either side fails
/// loudly instead of pairing the wrong bodies.
fn bodies_by_solid_id(
    topology: &StepBrep,
    bodies: Vec<PlanarBody>,
) -> Result<BTreeMap<u32, PlanarBody>, String> {
    if topology.solids.len() != bodies.len() {
        return Err(format!(
            "internal error: nodecraft-step returned {} planar bodies for {} solids",
            bodies.len(),
            topology.solids.len()
        ));
    }
    Ok(topology.solids.keys().copied().zip(bodies).collect())
}

/// Millimetres per STEP file unit, inferred without private nodecraft-step APIs.
///
/// `read_step_bounds` returns the same `CARTESIAN_POINT` set as `read_topology` but already
/// converted to millimetres by the file's declared unit. Comparing the extents of the two
/// recovers the conversion factor. The axis with the largest raw extent is used to avoid
/// flat models and cancellation; a degenerate/all-coincident point set reports `1.0`.
fn length_unit_scale_mm(path: &str, topology: &StepBrep) -> Result<f64, String> {
    let Some((raw_min, raw_max)) = raw_point_bounds(&topology.points) else {
        return Ok(1.0);
    };
    let (scaled_min, scaled_max) = read_step_bounds(path)
        .map_err(|error| format!("{path}: cannot determine the STEP length unit: {error}"))?;
    let mut largest_extent = 0.0f64;
    let mut scale = 1.0f64;
    for axis in 0..3 {
        let raw_extent = (raw_max[axis] - raw_min[axis]).abs();
        if raw_extent > largest_extent {
            largest_extent = raw_extent;
            let scaled_extent = (scaled_max[axis] - scaled_min[axis]).abs();
            if scaled_extent.is_finite() && scaled_extent > 0.0 {
                scale = scaled_extent / raw_extent;
            }
        }
    }
    if scale.is_finite() && scale > 0.0 { Ok(scale) } else { Ok(1.0) }
}

fn raw_point_bounds(points: &BTreeMap<u32, [f64; 3]>) -> Option<([f64; 3], [f64; 3])> {
    let mut points = points.values();
    let first = *points.next()?;
    let mut min = first;
    let mut max = first;
    for point in points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    Some((min, max))
}

/// Copy of an assembly transform with its translation converted from file units to
/// millimetres. The linear part is unit-independent for rigid/affine placements and is
/// left untouched (a uniform scale operator would already be part of it).
fn scaled_transform(transform: &Transform, unit_scale_mm: f64) -> Result<Transform, String> {
    if (unit_scale_mm - 1.0).abs() <= 1e-12 {
        return Ok(*transform);
    }
    let mut rows = transform.rows();
    for row in rows.iter_mut().take(3) {
        row[3] *= unit_scale_mm;
    }
    Transform::from_rows(rows).map_err(|error| error.to_string())
}

/// If the assembly keeps its component geometry in external STEP documents, report that
/// explicitly instead of the generic "no solid B-reps" error: the planar path reads one
/// file and cannot follow AP214 document references.
fn external_document_error(path: &str, file: &StepFile) -> Option<String> {
    let documents = crate::external::external_documents(file);
    if documents.is_empty() {
        return None;
    }
    let names: Vec<&str> = documents.iter().map(|document| document.file_name.as_str()).collect();
    Some(format!(
        "{UNSUPPORTED_ERROR_PREFIX}{path}: assembly geometry lives in external STEP document(s) \
         {} which the planar importer does not resolve",
        names.join(", ")
    ))
}

/// Turn a planar-extraction failure into an explicit unsupported-geometry error.
fn planar_extraction_error(path: &str, error: &str) -> String {
    if error.contains("curved surface") {
        format!(
            "{UNSUPPORTED_ERROR_PREFIX}{path}: assembly contains curved (non-planar) solids, \
             which the planar importer cannot represent ({error})"
        )
    } else if error.contains("no MANIFOLD_SOLID_BREP") {
        format!("{UNSUPPORTED_ERROR_PREFIX}{path}: assembly has no solid B-reps ({error})")
    } else {
        format!(
            "{UNSUPPORTED_ERROR_PREFIX}{path}: nodecraft-step cannot extract planar bodies from \
             this assembly ({error})"
        )
    }
}


#[cfg(test)]
mod external_document_tests {
    use std::path::{Path, PathBuf};

    use super::{UNSUPPORTED_ERROR_PREFIX, planar_instances};

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(relative);
        path.is_file().then_some(path)
    }

    #[test]
    fn assemblies_with_external_documents_report_the_documents() {
        let Some(path) = repository("tests/step-parts/MAINBODY.stp") else { return; };
        let error = planar_instances(path.to_str().unwrap()).expect_err("external references");
        assert!(error.starts_with(UNSUPPORTED_ERROR_PREFIX), "{error}");
        assert!(error.contains("external STEP document"), "{error}");
        assert!(error.contains("MAINBODY_FRONT.stp"), "{error}");
        assert!(error.contains("MAINBODY_BACK.stp"), "{error}");
    }
}
