//! Precise classification of STEP files that carry no solid topology.
//!
//! The solid readers fail with `STEP file has no MANIFOLD_SOLID_BREP` for every file
//! without a `MANIFOLD_SOLID_BREP`, which is true but unhelpful: a STEP file can be a
//! surface model (`CURVE_BOUNDED_SURFACE` / open shell), a wire/curve model, a
//! point cloud, or — as in STEPcode's `test/p21/select_segfault_2.stp` — *not a
//! geometric model at all* (a parser-regression fixture whose `FILE_SCHEMA` is
//! `select_segfault_2` and whose only entities are the unknown types `drc`, `cdl` and
//! `dre`).
//!
//! This module summarises what a no-solid STEP file actually contains and produces an
//! unsupported classification that names the file's real content instead of only
//! repeating the missing-solid message. Two failure classes are kept distinct on
//! purpose:
//!
//! * **no geometric content** — the file has no points/curves/surfaces/faces/shells/
//!   solids (unknown or product-only entities); reported as
//!   `STEP file has no geometry: not a geometric STEP model …`.
//! * **surface-model tessellation failed** — the file *does* carry sheet/surface
//!   geometry but the display-mesh path could not represent it; reported as
//!   `STEP file has no MANIFOLD_SOLID_BREP: the sheet/surface geometry could not be
//!   tessellated as a display mesh …`.
//!
//! Wire/curve-only and point-only files get their own precise messages. Every message
//! keeps the literal `no MANIFOLD_SOLID_BREP` phrase and leaves the path to the
//! caller's wrapper (`import_step` prefixes the file path), so existing corpus
//! classification and path requirements keep working.

use std::collections::BTreeSet;

use nodecraft_step::StepFile;

/// Counts of the content classes that decide the unsupported classification.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StepContentSummary {
    /// `FILE_SCHEMA` value, when the header can be read.
    pub schema: Option<String>,
    /// Number of DATA-section entities.
    pub entity_count: usize,
    pub points: usize,
    pub curves: usize,
    pub surfaces: usize,
    pub curve_bounded_surfaces: usize,
    pub faces: usize,
    pub shells: usize,
    pub solids: usize,
    pub surface_models: usize,
    /// Entity type names outside the STEP entity set this importer recognises
    /// (sorted, deduplicated; empty for a standard AP203/AP214 file).
    pub unrecognised_types: Vec<String>,
}

impl StepContentSummary {
    /// Reads and summarises `path`. Header problems are not fatal: the DATA section is
    /// what matters for the classification.
    pub(crate) fn read(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
        let file = StepFile::parse(&text)?;
        let schema = nodecraft_step::read_header(path).ok().and_then(|header| header.schema);
        Ok(Self::from_file(&file, schema))
    }

    pub(crate) fn from_file(file: &StepFile, schema: Option<String>) -> Self {
        let mut summary = Self { schema, entity_count: file.entities.len(), ..Self::default() };
        let mut unrecognised = BTreeSet::new();
        for entity in file.entities.values() {
            // Unknown typed member names also identify non-standard entities
            // (STEPcode's fixture nests `cdl` inside `drc`).
            for value in &entity.args {
                collect_unrecognised_typed_names(value, &mut unrecognised);
            }
            let name = entity.name.as_str();
            if name.is_empty() {
                // Complex instance (`(LENGTH_UNIT() …)`); its members are standard types.
                continue;
            }
            match name {
                "CARTESIAN_POINT" => summary.points += 1,
                "LINE" | "CIRCLE" | "ELLIPSE" | "HYPERBOLA" | "PARABOLA" | "B_SPLINE_CURVE"
                | "B_SPLINE_CURVE_WITH_KNOTS" | "RATIONAL_B_SPLINE_CURVE" | "TRIMMED_CURVE"
                | "COMPOSITE_CURVE" | "SURFACE_CURVE" | "SEAM_CURVE" | "PCURVE" | "POLYLINE" => {
                    summary.curves += 1;
                }
                "PLANE" | "CYLINDRICAL_SURFACE" | "CONICAL_SURFACE" | "SPHERICAL_SURFACE"
                | "TOROIDAL_SURFACE" | "B_SPLINE_SURFACE" | "B_SPLINE_SURFACE_WITH_KNOTS"
                | "RATIONAL_B_SPLINE_SURFACE" | "BEZIER_SURFACE" | "SURFACE_OF_REVOLUTION"
                | "SURFACE_OF_LINEAR_EXTRUSION" | "OFFSET_SURFACE"
                | "RECTANGULAR_TRIMMED_SURFACE" | "SWEPT_SURFACE" => summary.surfaces += 1,
                "CURVE_BOUNDED_SURFACE" => {
                    summary.surfaces += 1;
                    summary.curve_bounded_surfaces += 1;
                    summary.surface_models += 1;
                }
                "SHELL_BASED_SURFACE_MODEL" => summary.surface_models += 1,
                "ADVANCED_FACE" | "FACE_SURFACE" | "ORIENTED_FACE" => summary.faces += 1,
                "CLOSED_SHELL" | "OPEN_SHELL" | "ORIENTED_CLOSED_SHELL"
                | "ORIENTED_OPEN_SHELL" | "CONNECTED_FACE_SET" => summary.shells += 1,
                "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" | "FACETED_BREP"
                | "MANIFOLD_SOLID_BREP_WITH_VOIDS" => summary.solids += 1,
                other if !KNOWN_ENTITY_TYPES.contains(&other) => {
                    unrecognised.insert(other.to_string());
                }
                _ => {}
            }
        }
        summary.unrecognised_types = unrecognised.into_iter().collect();
        summary
    }

    /// Total geometric entities of any kind.
    pub(crate) fn geometry_count(&self) -> usize {
        self.points + self.curves + self.surfaces + self.faces + self.shells + self.solids
    }

    /// Nothing in the file is geometry: no points, curves, surfaces, faces, shells or solids.
    pub(crate) fn has_no_geometry(&self) -> bool {
        self.geometry_count() == 0
    }

    /// Sheet/surface (or face/shell) geometry that is not a solid.
    pub(crate) fn has_surface_geometry(&self) -> bool {
        self.surfaces > 0 || self.faces > 0 || self.shells > 0 || self.surface_models > 0
    }

    /// Curves but no faces/surfaces/shells/solids — wire/curve-only content.
    pub(crate) fn is_wire_only(&self) -> bool {
        self.curves > 0
            && self.surfaces == 0
            && self.faces == 0
            && self.shells == 0
            && self.solids == 0
            && self.surface_models == 0
    }

    /// A one-line factual description used in the classification messages.
    pub(crate) fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.points > 0 {
            parts.push(format!("{} CARTESIAN_POINT", self.points));
        }
        if self.curves > 0 {
            parts.push(format!("{} curve(s)", self.curves));
        }
        if self.surfaces > 0 {
            parts.push(format!("{} surface(s)", self.surfaces));
        }
        if self.curve_bounded_surfaces > 0 {
            parts.push(format!("{} CURVE_BOUNDED_SURFACE", self.curve_bounded_surfaces));
        }
        if self.faces > 0 {
            parts.push(format!("{} face(s)", self.faces));
        }
        if self.shells > 0 {
            parts.push(format!("{} shell(s)", self.shells));
        }
        if self.solids > 0 {
            parts.push(format!("{} solid(s)", self.solids));
        }
        let geometry = if parts.is_empty() {
            "0 geometric entities".to_string()
        } else {
            parts.join(", ")
        };
        let schema = self
            .schema
            .as_deref()
            .map(|schema| format!("schema '{schema}'; "))
            .unwrap_or_default();
        let unrecognised = if self.unrecognised_types.is_empty() {
            String::new()
        } else {
            let mut listed = self.unrecognised_types.iter().take(8).cloned().collect::<Vec<_>>();
            let remaining = self.unrecognised_types.len().saturating_sub(listed.len());
            if remaining > 0 {
                listed.push(format!("and {remaining} more"));
            }
            format!("; unrecognised entity types: {}", listed.join(", "))
        };
        format!("{schema}{} entities, {geometry}{unrecognised}", self.entity_count)
    }
}

/// Classification message for a file the solid/sheet readers could not import because it
/// has no solid topology.
///
/// `path` is read for the content summary; `cause` carries the earlier readers' errors
/// (and, for sheet files, the surface-model tessellation error). The caller adds the file
/// path to the final error.
pub(crate) fn unsupported_no_solid_message(path: &str, cause: &str) -> String {
    let summary = StepContentSummary::read(path).ok();
    match summary {
        Some(summary) if summary.solids == 0 && summary.has_surface_geometry() => format!(
            "STEP file has no MANIFOLD_SOLID_BREP: the sheet/surface geometry could not be \
             tessellated as a display mesh ({}); cause: {cause}",
            summary.describe()
        ),
        Some(summary) if summary.is_wire_only() => format!(
            "STEP file has no MANIFOLD_SOLID_BREP or sheet geometry: it contains only \
             wire/curve geometry ({}); wire display is not implemented; cause: {cause}",
            summary.describe()
        ),
        Some(summary) if summary.has_no_geometry() => format!(
            "STEP file has no geometry: not a geometric STEP model — {}; \
             no MANIFOLD_SOLID_BREP, FACE, SHELL or CURVE_BOUNDED_SURFACE geometry \
             (cause: {cause})",
            summary.describe()
        ),
        Some(summary) => format!(
            "STEP file has no MANIFOLD_SOLID_BREP: unsupported sheet/shell content ({}); \
             cause: {cause}",
            summary.describe()
        ),
        None => format!(
            "STEP file has no MANIFOLD_SOLID_BREP and no importable geometry (cause: {cause})"
        ),
    }
}

/// Adds the typed member names of `value` (recursively through lists/typed values) that
/// are not known STEP entity types.
fn collect_unrecognised_typed_names(value: &nodecraft_step::Value, out: &mut BTreeSet<String>) {
    match value {
        nodecraft_step::Value::Typed(name, values) => {
            if !name.is_empty() && name != "*" && !KNOWN_ENTITY_TYPES.contains(&name.as_str()) {
                out.insert(name.clone());
            }
            for value in values {
                collect_unrecognised_typed_names(value, out);
            }
        }
        nodecraft_step::Value::List(values) => {
            for value in values {
                collect_unrecognised_typed_names(value, out);
            }
        }
        _ => {}
    }
}

/// Entity type names of the STEP subset this importer understands. Names outside this set
/// are reported as unrecognised (e.g. STEPcode's `drc`/`cdl`/`dre` parser fuzz types) so a
/// non-geometric file is classified precisely instead of as a geometry problem.
const KNOWN_ENTITY_TYPES: &[&str] = &[
    // geometry / topology
    "CARTESIAN_POINT", "DIRECTION", "VECTOR", "LINE", "CIRCLE", "ELLIPSE", "HYPERBOLA",
    "PARABOLA", "B_SPLINE_CURVE", "B_SPLINE_CURVE_WITH_KNOTS", "RATIONAL_B_SPLINE_CURVE",
    "BEZIER_CURVE", "QUASI_UNIFORM_CURVE", "UNIFORM_CURVE", "TRIMMED_CURVE",
    "COMPOSITE_CURVE", "COMPOSITE_CURVE_SEGMENT", "SURFACE_CURVE", "SEAM_CURVE", "PCURVE",
    "POLYLINE", "PLANE", "CYLINDRICAL_SURFACE", "CONICAL_SURFACE", "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE", "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
    "RATIONAL_B_SPLINE_SURFACE", "BEZIER_SURFACE", "SURFACE_OF_REVOLUTION",
    "SURFACE_OF_LINEAR_EXTRUSION", "OFFSET_SURFACE", "RECTANGULAR_TRIMMED_SURFACE",
    "CURVE_BOUNDED_SURFACE", "SWEPT_SURFACE", "AXIS1_PLACEMENT", "AXIS2_PLACEMENT_2D",
    "AXIS2_PLACEMENT_3D", "VERTEX_POINT", "EDGE_CURVE", "ORIENTED_EDGE", "EDGE_LOOP",
    "POLY_LOOP", "VERTEX_LOOP", "PATH", "ORIENTED_PATH", "FACE_BOUND", "FACE_OUTER_BOUND",
    "ADVANCED_FACE", "FACE_SURFACE", "ORIENTED_FACE", "CLOSED_SHELL", "OPEN_SHELL",
    "ORIENTED_CLOSED_SHELL", "ORIENTED_OPEN_SHELL", "CONNECTED_FACE_SET", "SHELL_BASED_SURFACE_MODEL",
    "MANIFOLD_SOLID_BREP", "MANIFOLD_SOLID_BREP_WITH_VOIDS", "BREP_WITH_VOIDS", "FACETED_BREP",
    "GEOMETRIC_CURVE_SET", "GEOMETRIC_SET", "TRIMMED_CURVE",
    // representation / product structure
    "APPLICATION_CONTEXT", "APPLICATION_PROTOCOL_DEFINITION", "PRODUCT", "PRODUCT_CONTEXT",
    "PRODUCT_DEFINITION", "PRODUCT_DEFINITION_CONTEXT", "PRODUCT_DEFINITION_FORMATION",
    "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE", "PRODUCT_DEFINITION_SHAPE",
    "PRODUCT_DEFINITION_RELATIONSHIP", "PRODUCT_RELATED_PRODUCT_CATEGORY",
    "PRODUCT_CATEGORY", "PRODUCT_CATEGORY_RELATIONSHIP", "SHAPE_DEFINITION_REPRESENTATION",
    "SHAPE_REPRESENTATION", "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION", "FACETED_BREP_SHAPE_REPRESENTATION",
    "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION", "REPRESENTATION",
    "REPRESENTATION_CONTEXT", "REPRESENTATION_MAP", "REPRESENTATION_RELATIONSHIP",
    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION", "SHAPE_REPRESENTATION_RELATIONSHIP",
    "MAPPED_ITEM", "ITEM_DEFINED_TRANSFORMATION", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE", "ASSEMBLY_COMPONENT_USAGE", "PRODUCT_DEFINITION_USAGE",
    "GEOMETRIC_REPRESENTATION_CONTEXT", "GLOBAL_UNIT_ASSIGNED_CONTEXT",
    "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT", "UNCERTAINTY_MEASURE_WITH_UNIT",
    "MEASURE_WITH_UNIT", "LENGTH_MEASURE_WITH_UNIT", "LENGTH_UNIT", "NAMED_UNIT", "SI_UNIT",
    "PLANE_ANGLE_UNIT", "SOLID_ANGLE_UNIT", "CONVERSION_BASED_UNIT", "DIMENSIONAL_EXPONENTS",
    // presentation / styles
    "COLOUR_RGB", "DRAUGHTING_PRE_DEFINED_COLOUR", "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
    "STYLED_ITEM", "PRESENTATION_STYLE_ASSIGNMENT", "SURFACE_STYLE_USAGE",
    "SURFACE_SIDE_STYLE", "SURFACE_STYLE_FILL_AREA", "FILL_AREA_STYLE",
    "FILL_AREA_STYLE_COLOUR", "CURVE_STYLE", "DRAUGHTING_PRE_DEFINED_CURVE_FONT",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn repo_file(relative: &str) -> Option<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(3)?.to_path_buf();
        let path = root.join(relative);
        path.exists().then_some(path)
    }

    fn temp_path(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let serial = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "stepio-step-content-{}-{serial}-{tag}.stp",
            std::process::id()
        ))
    }

    fn write_temp(tag: &str, body: &str) -> PathBuf {
        let path = temp_path(tag);
        let text = format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
             ENDSEC;\nDATA;\n{body}\nENDSEC;\nEND-ISO-10303-21;\n"
        );
        std::fs::write(&path, text).unwrap();
        path
    }

    /// STEPcode's parser-regression fixture is not a geometric model: a precise
    /// non-geometric classification, not a missing-solid/topology failure.
    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn select_segfault_2_is_classified_as_non_geometric() {
        let Some(path) = repo_file("tests/step-stepcode/select_segfault_2.stp") else {
            eprintln!("skipping select_segfault_2.stp: fixture not present");
            return;
        };
        let path_text = path.to_string_lossy().into_owned();
        let summary = StepContentSummary::read(&path_text).expect("content summary");
        assert_eq!(summary.schema.as_deref(), Some("select_segfault_2"));
        assert_eq!(summary.entity_count, 3);
        assert!(summary.has_no_geometry(), "fixture unexpectedly has geometry: {summary:?}");
        assert!(summary.unrecognised_types.iter().any(|name| name == "drc"));
        assert!(summary.unrecognised_types.iter().any(|name| name == "cdl"));
        assert!(summary.unrecognised_types.iter().any(|name| name == "dre"));

        let error = crate::import_step(&path_text).expect_err("fixture must not import");
        // `import_step` prefixes the path: the classification must stay path-bearing.
        assert!(error.starts_with(&path_text), "{error}");
        assert!(error.contains("STEP file has no geometry"), "{error}");
        assert!(error.contains("not a geometric STEP model"), "{error}");
        assert!(error.contains("select_segfault_2"), "{error}");
        assert!(error.contains("drc"), "{error}");
        assert!(error.contains("dre"), "{error}");
        assert!(error.contains("no MANIFOLD_SOLID_BREP"), "{error}");
        // It must not be reported as a sheet/surface model or a mixed solid.
        assert!(!error.contains("sheet/surface geometry"), "{error}");
        assert!(!error.contains("mixed planar/analytic"), "{error}");
    }

    /// A file that really has sheet/surface geometry gets the surface-model failure
    /// classification, never "no geometry".
    #[test]
    fn surface_model_failure_is_classified_separately_from_no_geometry() {
        let path = write_temp(
            "surface-model-failure",
            "#1=CURVE_BOUNDED_SURFACE('',#2,#3);\n#2=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #3=LINE('',#2,#4);\n#4=VECTOR('',#5,1.);\n#5=DIRECTION('',(1.,0.,0.));",
        );
        let path_text = path.to_string_lossy().into_owned();
        let summary = StepContentSummary::read(&path_text).expect("summary");
        assert!(summary.has_surface_geometry());
        assert!(!summary.has_no_geometry());
        let message = unsupported_no_solid_message(
            &path_text,
            "surface model tessellation failed: face #1 has no boundary loops",
        );
        let _ = std::fs::remove_file(&path);
        assert!(message.contains("no MANIFOLD_SOLID_BREP"), "{message}");
        assert!(message.contains("sheet/surface geometry"), "{message}");
        assert!(message.contains("could not be tessellated as a display mesh"), "{message}");
        assert!(message.contains("CURVE_BOUNDED_SURFACE"), "{message}");
        assert!(message.contains("surface model tessellation failed"), "{message}");
        assert!(!message.contains("STEP file has no geometry"), "{message}");
        assert!(!message.contains("not a geometric STEP model"), "{message}");
    }

    #[test]
    fn wire_only_content_gets_a_wire_classification() {
        let path = write_temp(
            "wire-only",
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #3=DIRECTION('',(1.,0.,0.));\n#4=VECTOR('',#3,1.);\n#5=LINE('',#1,#4);\n\
             #6=EDGE_CURVE('',#7,#8,#5,.T.);\n#7=VERTEX_POINT('',#1);\n#8=VERTEX_POINT('',#2);",
        );
        let path_text = path.to_string_lossy().into_owned();
        let summary = StepContentSummary::read(&path_text).expect("summary");
        assert!(summary.is_wire_only(), "{summary:?}");
        let message = unsupported_no_solid_message(&path_text, "readers found no solid");
        let _ = std::fs::remove_file(&path);
        assert!(message.contains("no MANIFOLD_SOLID_BREP"), "{message}");
        assert!(message.contains("only wire/curve geometry"), "{message}");
        assert!(message.contains("wire display is not implemented"), "{message}");
        assert!(!message.contains("STEP file has no geometry"), "{message}");
    }

    #[test]
    fn unreadable_files_keep_the_missing_solid_phrase() {
        let path = temp_path("unreadable");
        std::fs::write(&path, "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1=BROKEN").unwrap();
        let path_text = path.to_string_lossy().into_owned();
        let message = unsupported_no_solid_message(&path_text, "STEP parse failed");
        let _ = std::fs::remove_file(&path);
        assert!(message.contains("no MANIFOLD_SOLID_BREP"), "{message}");
        assert!(message.contains("STEP parse failed"), "{message}");
    }
}
