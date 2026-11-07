//! External STEP document references and neutral occurrence/component enumeration.
//!
//! AP203/AP214/AP242 assemblies often keep component geometry in **separate** STEP
//! documents. The assembly file itself then contains only the product graph, the
//! placements, and a document reference chain:
//!
//! ```text
//! #33=DOCUMENT_FILE('MAINBODY_FRONT.stp','','',#34,'',$);
//! #35=APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT('MAINBODY_FRONT.stp',#36,#32,(#33));
//! #37=APPLIED_DOCUMENT_REFERENCE(#33,'',(#30));
//! #30=PRODUCT_DEFINITION(' ',' ',#29,#3);   // the occurrence's child definition
//! ```
//!
//! The occurrence itself already carries a placement (`NEXT_ASSEMBLY_USAGE_OCCURRENCE` →
//! `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` → `REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION`
//! → `ITEM_DEFINED_TRANSFORMATION`), which [`crate::parse_assembly_file`] resolves into a
//! local and world transform. What is missing for import is *where the geometry of that
//! occurrence comes from*: an external document, or one of the representations embedded in
//! the same file.
//!
//! This module adds that layer. [`external_documents`] extracts the document-reference
//! graph, [`component_instances`] walks the assembly tree and reports one
//! [`ComponentInstance`] per leaf occurrence with:
//!
//! * the geometry source ([`ComponentSource::ExternalDocument`] with the document name,
//!   the path resolved relative to the assembly file, and whether it exists, or
//!   [`ComponentSource::InFileRepresentation`] with the shared representation id),
//! * the local/world transforms in file units,
//! * the parent assembly's length-unit scale (millimetres per file unit) and, when the
//!   external document exists and declares one, the child document's scale,
//! * [`ComponentInstance::world_transform_mm`], a transform that takes child geometry
//!   coordinates *in millimetres* (as produced by a normal millimetre importer) into the
//!   assembly's world frame in millimetres.
//!
//! Units: STEP representation contexts carry their own length unit. The world placement
//! is expressed in the parent representation's unit, while the child geometry is written
//! in the child document's unit. `S(parent_mm) ∘ T ∘ S(1/child_mm)` therefore maps
//! millimetre child coordinates into the millimetre world frame; when both documents
//! declare the same unit this reduces to scaling the translation by `parent_mm`.
//!
//! This module is structure-only: it neither reads nor transforms geometry. Geometry
//! import (planar, analytic, B-spline, mixed) stays in `nodecraft-mesh`/`nodecraft-solidcore`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nodecraft_step::{StepFile, Value};

use crate::{Assembly, Node, Transform, parse_assembly_from_step_file};

/// One external STEP document referenced by the assembly file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalDocument {
    /// `DOCUMENT_FILE` entity id.
    pub document_file_id: u32,
    /// File name as written in the file (`APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT`
    /// name when present, otherwise the `DOCUMENT_FILE` name).
    pub file_name: String,
    /// `PRODUCT_DEFINITION` entity ids this document supplies geometry for.
    pub product_definition_ids: Vec<u32>,
}

/// Where the geometry of one placed occurrence lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentSource {
    /// Geometry lives in a separate STEP document.
    ExternalDocument {
        /// The parsed document-reference record.
        document: ExternalDocument,
        /// Path resolved relative to the assembly file (case-insensitive fallback).
        path: PathBuf,
        /// Whether the resolved path names an existing file.
        exists: bool,
    },
    /// Geometry is an in-file shape representation (curved and mixed bodies included).
    InFileRepresentation {
        /// `SHAPE_REPRESENTATION` (or subtype) entity id from the assembly tree.
        representation_id: u32,
    },
    /// The occurrence has neither an external document nor a representation.
    NoGeometry,
}

/// One placed occurrence (leaf of the assembly tree) and its geometry source.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentInstance {
    /// Occurrence/placement name (root product name for the root node).
    pub name: String,
    /// `PRODUCT` display name of the referenced product.
    pub product_name: String,
    /// Child `PRODUCT_DEFINITION` entity id.
    pub definition_id: Option<u32>,
    /// `NEXT_ASSEMBLY_USAGE_OCCURRENCE` id string, when the file has one.
    pub occurrence_id: Option<String>,
    /// Geometry source of this occurrence.
    pub source: ComponentSource,
    /// The occurrence's transform in its parent's coordinates (file units).
    pub local_transform: Transform,
    /// The occurrence in the document/world frame (file units).
    pub world_transform: Transform,
    /// Millimetres per parent-file unit (`1.0` when the unit is unknown).
    pub parent_unit_scale_mm: f64,
    /// Millimetres per external-document unit, when the document exists and declares one.
    pub child_unit_scale_mm: Option<f64>,
}

impl ComponentInstance {
    /// Transform from child geometry in **millimetres** to the assembly world frame in
    /// millimetres: `S(parent_mm) ∘ world_transform ∘ S(1/child_mm)`.
    ///
    /// With equal parent/child units this is the world transform whose translation is
    /// multiplied by `parent_unit_scale_mm`.
    #[must_use]
    pub fn world_transform_mm(&self) -> Transform {
        let parent = self.parent_unit_scale_mm;
        let child = self.child_unit_scale_mm.unwrap_or(parent);
        let ratio = if child.is_finite() && child > 0.0 {
            parent / child
        } else {
            1.0
        };
        let world = self.world_transform.rows();
        let mut rows = [[0.0f64; 4]; 4];
        for row in 0..3 {
            for column in 0..3 {
                rows[row][column] = world[row][column] * ratio;
            }
            rows[row][3] = world[row][3] * parent;
        }
        rows[3][3] = 1.0;
        Transform::from_rows(rows).unwrap_or(self.world_transform)
    }
}

/// One external document reference that could not be resolved to a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedDocument {
    /// The document name as written in the STEP file.
    pub file_name: String,
    /// Every path that was tried, in search order (deduplicated).
    pub searched: Vec<PathBuf>,
}

/// The leaf occurrences of an assembly plus the data a geometry importer needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Components {
    /// One entry per leaf occurrence, in assembly pre-order.
    pub instances: Vec<ComponentInstance>,
    /// External documents referenced by occurrences that could not be resolved to a file
    /// (deduplicated, in first-seen order).
    pub missing_documents: Vec<String>,
    /// Structured unresolved-reference details: the requested document name plus every
    /// candidate path that was searched (parent directory, environment search paths,
    /// explicit search paths and nearby sibling fixture directories).
    pub unresolved: Vec<UnresolvedDocument>,
    /// Non-fatal problems: assembly warnings plus directory scans that failed.
    pub warnings: Vec<String>,
    /// Millimetres per unit of the assembly file (`1.0` when unknown).
    pub unit_scale_mm: f64,
}

/// Extracts the external-document reference graph from an already parsed STEP file.
///
/// The documents are returned sorted by entity id. Documents that no
/// `APPLIED_DOCUMENT_REFERENCE` links to a product definition are still returned (with an
/// empty `product_definition_ids` list) so a caller can report unused references.
#[must_use]
pub fn external_documents(file: &StepFile) -> Vec<ExternalDocument> {
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    for (id, entity) in &file.entities {
        if entity.name == "DOCUMENT_FILE" {
            if let Some(Value::Str(name)) = entity.args.first() {
                names.insert(*id, name.clone());
            }
        }
    }
    // The identification assignment names the document as it appears on disk, which can
    // differ from the DOCUMENT_FILE name.
    for entity in file.entities.values() {
        if entity.name != "APPLIED_EXTERNAL_IDENTIFICATION_ASSIGNMENT" {
            continue;
        }
        let Some(Value::Str(file_name)) = entity.args.first() else {
            continue;
        };
        let Some(Value::List(items)) = entity.args.get(3) else {
            continue;
        };
        for item in items {
            if let Value::Ref(document_file_id) = item {
                names.insert(*document_file_id, file_name.clone());
            }
        }
    }
    // PRODUCT_DEFINITION_FORMATION ids -> the PRODUCT_DEFINITION ids that use them, so a
    // reference attached to a formation still reaches the occurrence's definition.
    let mut definitions_by_formation: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (id, entity) in &file.entities {
        if entity.name != "PRODUCT_DEFINITION" {
            continue;
        }
        if let Some(Value::Ref(formation)) = entity.args.get(2) {
            definitions_by_formation.entry(*formation).or_default().push(*id);
        }
    }
    let mut documents: BTreeMap<u32, ExternalDocument> = names
        .into_iter()
        .map(|(document_file_id, file_name)| {
            (
                document_file_id,
                ExternalDocument {
                    document_file_id,
                    file_name,
                    product_definition_ids: Vec::new(),
                },
            )
        })
        .collect();
    for entity in file.entities.values() {
        if entity.name != "APPLIED_DOCUMENT_REFERENCE" {
            continue;
        }
        let Some(Value::Ref(document_file_id)) = entity.args.first() else {
            continue;
        };
        let definitions: Vec<u32> = entity
            .args
            .iter()
            .filter_map(|value| match value {
                Value::List(items) => Some(items),
                _ => None,
            })
            .flat_map(|items| items.iter())
            .filter_map(|value| match value {
                Value::Ref(id) => Some(*id),
                _ => None,
            })
            .collect();
        let Some(document) = documents.get_mut(document_file_id) else {
            continue;
        };
        for referenced in definitions {
            let mut targets = vec![referenced];
            if let Some(derived) = definitions_by_formation.get(&referenced) {
                targets.extend(derived.iter().copied());
            }
            for definition in targets {
                if !document.product_definition_ids.contains(&definition) {
                    document.product_definition_ids.push(definition);
                }
            }
        }
    }
    documents.into_values().collect()
}

/// Millimetres per STEP file unit, from the file's unit context or declared unit.
///
/// # Errors
///
/// `Err` when neither a `GLOBAL_UNIT_ASSIGNED_CONTEXT` nor a supported length unit is
/// found (the file is not a normal millimetre/inch/... STEP document).
pub fn unit_scale_mm(path: &str) -> Result<f64, String> {
    if let Ok(Some(context)) = nodecraft_step::read_unit_context(path) {
        return Ok(context.length_unit.millimetres());
    }
    nodecraft_step::read_length_unit(path).map(|unit| unit.millimetres())
}

/// Walks the assembly at `path` and reports one [`ComponentInstance`] per leaf occurrence
/// with the geometry source, transforms and unit scales an importer needs.
///
/// External document names are resolved relative to the assembly file's directory; an
/// exact-name match is tried first, then a case-insensitive directory match (AP214
/// document names are frequently upper-cased on disk). Missing documents are collected in
/// [`Components::missing_documents`] instead of failing the call, because a partial
/// structure report is still useful and the importer may handle the absence itself.
///
/// # Errors
///
/// I/O and STEP syntax errors, assembly-structure errors from
/// [`parse_assembly_from_step_file`], and a non-finite transform.
pub fn component_instances(path: &str) -> Result<Components, String> {
    component_instances_with_search(path, &[], true)
}

/// Like [`component_instances`] but with explicit extra search directories (tried after
/// the assembly file's own directory and the `STEP_ASSEMBLY_SEARCH_PATHS` /
/// `NODECRAFT_STEP_SEARCH_PATHS` environment paths).
///
/// When `search_ancestor_siblings` is set, missing names are also looked for in the
/// immediate sub-directories of up to four ancestor directories of the assembly file.
/// This is the fallback that finds the sibling `tests/step-parts` directory referenced by
/// `tests/step-stepcode/s1-c5-214.stp` (`TAIL.stp`, `HEAD.stp`, `MAINBODY.stp`,
/// `FOOT.stp`). Pass `false` in strict callers/tests that must not search nearby
/// directories.
///
/// # Errors
///
/// I/O and STEP syntax errors, assembly-structure errors, and a non-finite transform.
pub fn component_instances_with_search(
    path: &str,
    extra_search_paths: &[PathBuf],
    search_ancestor_siblings: bool,
) -> Result<Components, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
    let assembly = parse_assembly_from_step_file(&file).map_err(|error| format!("{path}: {error}"))?;
    components_from_assembly(&assembly, &file, Path::new(path), extra_search_paths, search_ancestor_siblings)
}

fn components_from_assembly(
    assembly: &Assembly,
    file: &StepFile,
    assembly_path: &Path,
    extra_search_paths: &[PathBuf],
    search_ancestor_siblings: bool,
) -> Result<Components, String> {
    let path_label = assembly_path.display().to_string();
    let parent_directory = assembly_path.parent().unwrap_or_else(|| Path::new("."));
    let mut warnings = assembly.warnings.clone();
    let parent_unit_scale_mm = match unit_scale_mm(&path_label) {
        Ok(scale) => scale,
        Err(error) => {
            warnings.push(format!("cannot determine the assembly length unit ({error}); assuming millimetres"));
            1.0
        }
    };

    let documents = external_documents(file);
    let mut document_for_definition: BTreeMap<u32, &ExternalDocument> = BTreeMap::new();
    for document in &documents {
        for definition in &document.product_definition_ids {
            document_for_definition.insert(*definition, document);
        }
    }
    let directory_entries = std::fs::read_dir(parent_directory)
        .map(|entries| {
            entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut instances = Vec::new();
    let mut missing_documents: Vec<String> = Vec::new();
    let mut unresolved: Vec<UnresolvedDocument> = Vec::new();
    let multiple_ancestor_entries = if search_ancestor_siblings {
        ancestor_sibling_entries(parent_directory)
    } else {
        Vec::new()
    };
    let env_search_paths = environment_search_paths();
    let mut search_paths: Vec<PathBuf> = Vec::new();
    for directory in env_search_paths.into_iter().chain(extra_search_paths.iter().cloned()) {
        if !search_paths.contains(&directory) {
            search_paths.push(directory);
        }
    }
    for leaf in leaf_nodes(assembly) {
        let external = leaf
            .definition_id
            .and_then(|definition| document_for_definition.get(&definition).copied());
        let source = match external {
            Some(document) => {
                let (resolved, exists, searched) = resolve_document(
                    parent_directory,
                    &document.file_name,
                    &directory_entries,
                    &search_paths,
                    &multiple_ancestor_entries,
                );
                if !exists && !missing_documents.contains(&document.file_name) {
                    missing_documents.push(document.file_name.clone());
                    unresolved.push(UnresolvedDocument {
                        file_name: document.file_name.clone(),
                        searched,
                    });
                }
                ComponentSource::ExternalDocument {
                    document: document.clone(),
                    path: resolved,
                    exists,
                }
            }
            None => match leaf.representation_id {
                Some(representation_id) => ComponentSource::InFileRepresentation { representation_id },
                None => ComponentSource::NoGeometry,
            },
        };
        // The child unit is only meaningful for external documents that were found.
        let child_unit_scale_mm = match &source {
            ComponentSource::ExternalDocument { path, exists: true, .. } => {
                unit_scale_mm(&path.display().to_string()).ok()
            }
            _ => None,
        };
        if !leaf.world_transform.rows().iter().all(|row| row.iter().all(|value| value.is_finite())) {
            return Err(format!(
                "{path_label}: occurrence {:?} has a non-finite world transform",
                leaf.name
            ));
        }
        instances.push(ComponentInstance {
            name: leaf.name.clone(),
            product_name: leaf.product_name.clone(),
            definition_id: leaf.definition_id,
            occurrence_id: leaf.occurrence_id.clone(),
            source,
            local_transform: leaf.local_transform,
            world_transform: leaf.world_transform,
            parent_unit_scale_mm,
            child_unit_scale_mm,
        });
    }

    Ok(Components { instances, missing_documents, unresolved, warnings, unit_scale_mm: parent_unit_scale_mm })
}

/// Parses the environment search-path variables (`:` or `;` separated).
fn environment_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for variable in ["STEP_ASSEMBLY_SEARCH_PATHS", "NODECRAFT_STEP_SEARCH_PATHS"] {
        let Ok(value) = std::env::var(variable) else { continue };
        for part in value.split([':', ';']) {
            let part = part.trim();
            if !part.is_empty() {
                let path = PathBuf::from(part);
                if !paths.contains(&path) {
                    paths.push(path);
                }
            }
        }
    }
    paths
}

/// Lists `(directory, entry-name)` pairs for the immediate sub-directories of up to four
/// ancestors of `parent_directory` (used to find sibling fixture directories when a
/// referenced document lives next to the assembly's corpus rather than next to the file).
fn ancestor_sibling_entries(parent_directory: &Path) -> Vec<(PathBuf, Vec<String>)> {
    let mut directories = Vec::new();
    for ancestor in parent_directory.ancestors().skip(1).take(4) {
        let Ok(entries) = std::fs::read_dir(ancestor) else { continue };
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else { continue };
            if !file_type.is_dir() {
                continue;
            }
            let directory = entry.path();
            if directories.iter().any(|(existing, _): &(PathBuf, Vec<String>)| *existing == directory) {
                continue;
            }
            let names = std::fs::read_dir(&directory)
                .map(|children| {
                    children
                        .flatten()
                        .map(|child| child.file_name().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            directories.push((directory, names));
            if directories.len() >= 64 {
                return directories;
            }
        }
    }
    directories
}

/// Resolves `file_name` by trying, in order: the assembly's directory (exact then
/// case-insensitive), explicit/environment search paths, and ancestor sibling directories.
/// Returns the chosen path, whether it is a regular file, and every path tried.
fn resolve_document(
    directory: &Path,
    file_name: &str,
    entries: &[String],
    search_paths: &[PathBuf],
    ancestor_siblings: &[(PathBuf, Vec<String>)],
) -> (PathBuf, bool, Vec<PathBuf>) {
    let cleaned = file_name.replace('\\', "/");
    let name = cleaned.rsplit('/').next().unwrap_or(cleaned.as_str());
    let mut searched: Vec<PathBuf> = Vec::new();
    let try_directory = |directory: &Path, entries: &[String], searched: &mut Vec<PathBuf>| -> Option<PathBuf> {
        let exact = directory.join(name);
        if !searched.contains(&exact) {
            searched.push(exact.clone());
        }
        if exact.is_file() {
            return Some(exact);
        }
        if let Some(found) = entries.iter().find(|entry| entry.eq_ignore_ascii_case(name)) {
            let path = directory.join(found);
            if !searched.contains(&path) {
                searched.push(path.clone());
            }
            if path.is_file() {
                return Some(path);
            }
        }
        None
    };
    if let Some(found) = try_directory(directory, entries, &mut searched) {
        return (found, true, searched);
    }
    for search_path in search_paths {
        let entries = std::fs::read_dir(search_path)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if let Some(found) = try_directory(search_path, &entries, &mut searched) {
            return (found, true, searched);
        }
    }
    for (sibling, entries) in ancestor_siblings {
        if let Some(found) = try_directory(sibling, entries, &mut searched) {
            return (found, true, searched);
        }
    }
    let fallback = directory.join(name);
    if !searched.contains(&fallback) {
        searched.push(fallback.clone());
    }
    (fallback, false, searched)
}

fn leaf_nodes(assembly: &Assembly) -> Vec<&Node> {
    assembly.walk().into_iter().filter(|node| node.children.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(relative);
        path.is_file().then_some(path)
    }

    fn external(event: &ComponentInstance) -> (&ExternalDocument, &PathBuf, bool) {
        match &event.source {
            ComponentSource::ExternalDocument { document, path, exists } => (document, path, *exists),
            other => panic!("expected an external document source, got {other:?}"),
        }
    }

    #[test]
    fn mainbody_resolves_two_external_documents_and_units() {
        let Some(path) = repository("tests/step-parts/MAINBODY.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("MAINBODY components");
        assert_eq!(components.instances.len(), 2, "{:?}", components.instances);
        assert!(components.missing_documents.is_empty(), "{:?}", components.missing_documents);
        assert!((components.unit_scale_mm - 25.4).abs() < 1e-9, "unit scale {}", components.unit_scale_mm);

        let (front_document, front_path, front_exists) = external(&components.instances[0]);
        assert_eq!(front_document.file_name, "MAINBODY_FRONT.stp");
        assert_eq!(components.instances[0].product_name, "MAINBODY_FRONT");
        assert!(front_exists);
        assert!(front_path.is_file());
        assert!(front_path.ends_with("step-parts/MAINBODY_FRONT.stp"));
        let front_mm = components.instances[0].world_transform_mm().translation();
        assert!((front_mm[0] + 30.0).abs() < 1e-6 && front_mm[1].abs() < 1e-9 && front_mm[2].abs() < 1e-9, "{front_mm:?}");
        assert_eq!(components.instances[0].child_unit_scale_mm, Some(25.4));

        let (back_document, back_path, back_exists) = external(&components.instances[1]);
        assert_eq!(back_document.file_name, "MAINBODY_BACK.stp");
        assert!(back_exists && back_path.is_file());
        let back_mm = components.instances[1].world_transform_mm().translation();
        assert!((back_mm[0] - 50.0).abs() < 1e-6 && back_mm[1].abs() < 1e-9 && back_mm[2].abs() < 1e-9, "{back_mm:?}");
    }

    #[test]
    fn external_documents_extract_names_and_product_definitions() {
        let Some(path) = repository("tests/step-parts/MAINBODY.stp") else { return; };
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let documents = external_documents(&file);
        assert_eq!(documents.len(), 2, "{documents:?}");
        assert_eq!(documents[0].file_name, "MAINBODY_FRONT.stp");
        assert_eq!(documents[1].file_name, "MAINBODY_BACK.stp");
        assert!(!documents[0].product_definition_ids.is_empty());
        assert!(!documents[1].product_definition_ids.is_empty());
    }

    #[test]
    fn tail_and_foot_resolve_their_referenced_documents() {
        let Some(path) = repository("tests/step-parts/TAIL.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("TAIL components");
        assert_eq!(components.instances.len(), 3);
        assert!(components.missing_documents.is_empty());
        let files: Vec<&str> = components
            .instances
            .iter()
            .map(|instance| external(instance).0.file_name.as_str())
            .collect();
        assert!(files.contains(&"TAIL_TURBINE.stp"), "{files:?}");
        assert!(files.contains(&"TAIL_MIDDLE_PART.stp"), "{files:?}");
        // 1.61417322835 in = 41 mm, +/-0.944881889764 in = +/-24 mm, -4.0157480315 in = -102 mm.
        let translations: Vec<[f64; 3]> = components
            .instances
            .iter()
            .map(|instance| instance.world_transform_mm().translation())
            .collect();
        let close = |value: f64, expected: f64| (value - expected).abs() < 1e-6;
        assert!(translations.iter().any(|[x, y, z]| close(*x, 41.0) && close(*y, -24.0) && close(*z, 0.0)), "{translations:?}");
        assert!(translations.iter().any(|[x, y, z]| close(*x, -102.0) && close(*y, 0.0) && close(*z, 0.0)), "{translations:?}");

        let Some(path) = repository("tests/step-parts/FOOT.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("FOOT components");
        assert_eq!(components.instances.len(), 2);
        assert!(components.instances.iter().all(|instance| matches!(
            &instance.source,
            ComponentSource::ExternalDocument { exists: true, .. }
        )));
    }

    #[test]
    fn s1_referenced_documents_resolve_through_sibling_search() {
        // step-stepcode/s1-c5-214.stp references bare TAIL.stp/HEAD.stp/MAINBODY.stp/FOOT.stp;
        // those files live in tests/step-parts (a sibling of tests/step-stepcode), so the
        // ancestor-sibling search path resolves them.
        let Some(path) = repository("tests/step-stepcode/s1-c5-214.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("s1-c5-214 components");
        assert_eq!(components.instances.len(), 5);
        assert!(components.missing_documents.is_empty(), "{:?}", components.missing_documents);
        assert!(components.unresolved.is_empty(), "{:?}", components.unresolved);
        for instance in &components.instances {
            match &instance.source {
                ComponentSource::ExternalDocument { path, exists: true, .. } => {
                    assert!(path.ends_with("step-parts/TAIL.stp")
                        || path.ends_with("step-parts/HEAD.stp")
                        || path.ends_with("step-parts/MAINBODY.stp")
                        || path.ends_with("step-parts/FOOT.stp"), "{path:?}");
                }
                other => panic!("expected resolved external document, got {other:?}"),
            }
        }
    }

    #[test]
    fn strict_search_reports_missing_documents_and_searched_paths() {
        let Some(path) = repository("tests/step-stepcode/s1-c5-214.stp") else { return; };
        let components = component_instances_with_search(path.to_str().unwrap(), &[], false)
            .expect("s1-c5-214 components");
        assert_eq!(components.missing_documents.len(), 4, "{:?}", components.missing_documents);
        assert_eq!(components.unresolved.len(), 4, "{:?}", components.unresolved);
        for unresolved in &components.unresolved {
            assert!(!unresolved.searched.is_empty(), "{unresolved:?}");
        }
        assert!(components.instances.iter().all(|instance| matches!(
            &instance.source,
            ComponentSource::ExternalDocument { exists: false, .. }
        )));
    }

    #[test]
    fn in_file_assemblies_enumerate_representation_sources() {
        let Some(path) = repository("tests/Sample AP203 Data Files/as1_pe.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("as1_pe components");
        assert_eq!(components.instances.len(), 12, "as1_pe has 12 placed leaf bodies");
        assert!(components.missing_documents.is_empty());
        assert!(components.instances.iter().all(|instance| matches!(
            instance.source,
            ComponentSource::InFileRepresentation { .. }
        )));
        assert!((components.unit_scale_mm - 25.4).abs() < 1e-9, "as1_pe declares inches");
    }

    #[test]
    fn moon_buggy_occurrences_are_enumerated() {
        let Some(path) = repository("tests/Sample AP203 Data Files/moon_buggy_asm.stp") else { return; };
        let components = component_instances(path.to_str().unwrap()).expect("moon buggy components");
        // 30 nodes in the file: 11 internal/sub-assembly nodes and 19 placed leaves.
        assert_eq!(components.instances.len(), 19, "{} instances", components.instances.len());
        assert!(components.missing_documents.is_empty());
        assert!(components.instances.iter().all(|instance| matches!(
            instance.source,
            ComponentSource::InFileRepresentation { .. }
        )));
    }

    #[test]
    fn world_transform_mm_applies_both_unit_scales() {
        // Parent inch (25.4 mm/unit), child millimetre: a 1 inch translation is 25.4 mm.
        let transform = Transform::from_rows([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        let instance = ComponentInstance {
            name: "part".into(),
            product_name: "PART".into(),
            definition_id: Some(1),
            occurrence_id: None,
            source: ComponentSource::NoGeometry,
            local_transform: transform,
            world_transform: transform,
            parent_unit_scale_mm: 25.4,
            child_unit_scale_mm: Some(1.0),
        };
        let rows = instance.world_transform_mm().rows();
        assert!((rows[0][0] - 25.4).abs() < 1e-9, "{rows:?}");
        assert!((rows[0][3] - 25.4).abs() < 1e-9, "{rows:?}");
        assert!((rows[1][3] - 50.8).abs() < 1e-9, "{rows:?}");
        assert!((rows[3][3] - 1.0).abs() < 1e-12);
        // Missing child unit: fall back to the parent unit (translation only scaled).
        let mut unknown = instance.clone();
        unknown.child_unit_scale_mm = None;
        let rows = unknown.world_transform_mm().rows();
        assert!((rows[0][0] - 1.0).abs() < 1e-12, "{rows:?}");
        assert!((rows[0][3] - 25.4).abs() < 1e-9, "{rows:?}");
    }

    #[test]
    fn unit_scale_reads_inch_and_millimetre_files() {
        let Some(inch) = repository("tests/step-parts/MAINBODY.stp") else { return; };
        assert!((unit_scale_mm(inch.to_str().unwrap()).unwrap() - 25.4).abs() < 1e-9);
        let Some(mm) = repository("tests/Sample AP203 Data Files/block.stp") else { return; };
        assert!((unit_scale_mm(mm.to_str().unwrap()).unwrap() - 1.0).abs() < 1e-9);
    }
}
