//! Schema and reference validation for the STEP subset emitted by `nodecraft-step`.
//!
//! The NodeCraft STEP pipeline parses STEP text into a generic
//! [`StepFile`](nodecraft_step::StepFile): a `BTreeMap` of entity id to
//! [`Entity`](nodecraft_step::Entity) `{ name, args }`, where args are
//! [`Value`](nodecraft_step::Value)s (`Ref`, `Str`, `Number`, `Enum`, `List`,
//! `Typed`, `Null`). This crate adds the layer the parser deliberately leaves out:
//! does the file *mean* anything? It works purely from that public representation
//! and never re-parses or mutates the file.
//!
//! # What is checked
//!
//! | pass | code(s) | finds |
//! |---|---|---|
//! | parse (in [`validate_text`]) | `parse` | duplicate ids, truncated records, missing `DATA;` |
//! | references | `dangling-reference` | `#id` that names no entity, through lists and typed values |
//! | entity shape | `arg-count`, `arg-kind`, `empty-reference-list` | wrong arity, wrong value kind, empty required lists |
//! | reference types | `wrong-reference-type` | e.g. a shell that is not a `CLOSED_SHELL`, a bound that is not an `EDGE_LOOP` |
//! | flags / numbers | `invalid-boolean-flag`, `invalid-number`, `invalid-dimension`, `point-dimension` | `.MAYBE.` in a boolean slot, NaN, non-positive radii, 1-D or 4-D points |
//! | units | `invalid-si-prefix`, `invalid-unit` | `SI_UNIT(.BOGUS.,.METRE.)`, `.FURLONG.` |
//! | structure | `missing-product`, `missing-shape-representation`, `missing-product-definition`, `missing-length-unit` (warnings) | a solid with no product/representation/length unit |
//! | hygiene | `non-canonical-enum`, `unknown-entity`, `empty-file` (warnings) | `.milli.` casing, entities with no rule, an empty `DATA;` section |
//!
//! Every issue is an [`Issue`] with a [`Severity`], a stable `code`, the offending
//! entity id when there is one, and a human-readable message. Errors mean the file
//! cannot be trusted; warnings are legal-but-suspicious. [`Report::is_valid`] is
//! "no errors".
//!
//! # Example
//!
//! ```
//! use step_schema::validate_text;
//!
//! let report = validate_text(
//!     "DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));#2=VERTEX_POINT('',#1);ENDSEC;",
//! );
//! assert!(report.is_valid(), "{report}");
//! assert_eq!(report.error_count(), 0);
//! ```
//!
//! A dangling reference is an error:
//!
//! ```
//! use step_schema::{codes, validate_text};
//!
//! let report = validate_text("DATA;#1=VERTEX_POINT('',#9);ENDSEC;");
//! assert!(!report.is_valid());
//! assert!(report.has_error(codes::DANGLING_REFERENCE));
//! assert!(report.to_string().contains("#9"));
//! ```
//!
//! # Limitations
//!
//! * The parser keeps only the `DATA;` entities, so the ISO-10303-21 header
//!   (`FILE_NAME`, `FILE_SCHEMA`, …) cannot be checked here.
//! * Shape rules cover the entity set that `nodecraft-step` writes plus the common
//!   AP214 geometry/topology/product entities; a name with no rule only warns
//!   (`unknown-entity`), it never fails the file.
//! * Complex entity instances (`#5=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(…));`) are
//!   stored by the parser as an entity with an empty name. Their references are
//!   checked and their `SI_UNIT` enumerations are validated, but they have no
//!   argument-shape rule.
//! * Geometry is not validated for consistency (normals not unit, loops not closed,
//!   faces not forming a manifold). Those are model-level checks, not schema checks.

use nodecraft_step::StepFile;

mod checks;
mod schema;
mod units;
/// Stable issue codes, for matching instead of grepping messages.
///
/// Codes never change meaning; new codes may be added.
pub mod codes {
    /// Text could not be parsed (includes duplicate entity ids and truncation).
    pub const PARSE: &str = "parse";
    /// The file could not be read by [`crate::validate_file`].
    pub const IO: &str = "io";
    /// A `Ref` names an entity id that does not exist.
    pub const DANGLING_REFERENCE: &str = "dangling-reference";
    /// An entity has the wrong number of arguments.
    pub const ARG_COUNT: &str = "arg-count";
    /// An argument is of the wrong kind (string vs number vs reference …).
    pub const ARG_KIND: &str = "arg-kind";
    /// A reference points at the wrong entity type for its slot.
    pub const WRONG_REFERENCE_TYPE: &str = "wrong-reference-type";
    /// A list that must reference at least one entity is empty.
    pub const EMPTY_REFERENCE_LIST: &str = "empty-reference-list";
    /// A boolean slot holds something other than `.T.`, `.F.` or `.U.`.
    pub const INVALID_BOOLEAN_FLAG: &str = "invalid-boolean-flag";
    /// A number is NaN or infinite.
    pub const INVALID_NUMBER: &str = "invalid-number";
    /// A dimension (radius, magnitude, …) is zero or negative.
    pub const INVALID_DIMENSION: &str = "invalid-dimension";
    /// A point/direction coordinate list is not 2- or 3-dimensional.
    pub const POINT_DIMENSION: &str = "point-dimension";
    /// An `SI_UNIT` prefix is not a valid SI prefix.
    pub const INVALID_SI_PREFIX: &str = "invalid-si-prefix";
    /// An `SI_UNIT` unit symbol is not a known SI unit.
    pub const INVALID_UNIT: &str = "invalid-unit";
    /// An enumeration token is not canonical uppercase STEP.
    pub const NON_CANONICAL_ENUM: &str = "non-canonical-enum";
    /// An entity name has no shape rule in this schema subset.
    pub const UNKNOWN_ENTITY: &str = "unknown-entity";
    /// A solid B-rep exists but the file has no `PRODUCT`.
    pub const MISSING_PRODUCT: &str = "missing-product";
    /// A shape representation exists but is not linked to a product definition.
    pub const MISSING_PRODUCT_DEFINITION: &str = "missing-product-definition";
    /// Geometry exists but the file has no shape representation entity.
    pub const MISSING_SHAPE_REPRESENTATION: &str = "missing-shape-representation";
    /// Geometry exists but no SI length unit (`METRE`) is declared.
    pub const MISSING_LENGTH_UNIT: &str = "missing-length-unit";
    /// The `DATA;` section parsed but contains no entities.
    pub const EMPTY_FILE: &str = "empty-file";
}

/// How strongly an [`Issue`] affects the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// The file is structurally or semantically invalid.
    Error,
    /// The file is usable, but suspicious or non-canonical.
    Warning,
}

impl Severity {
    /// The lowercase name used by [`Display`](std::fmt::Display).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    /// Error or warning.
    pub severity: Severity,
    /// Stable code, see [`codes`].
    pub code: &'static str,
    /// Entity id the issue is attached to, when there is one.
    pub entity: Option<u32>,
    /// Human-readable description (already includes the entity name/id context).
    pub message: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {}: {}", self.severity, self.code, self.message)?;
        match self.entity {
            Some(id) => write!(formatter, " (entity #{id})"),
            None => Ok(()),
        }
    }
}

/// All findings of one validation run.
///
/// Errors and warnings are stored together, sorted errors-first and then by entity
/// id and code so reports are deterministic. Use [`Report::is_valid`] for the
/// headline verdict and [`Report::errors`] / [`Report::warnings`] to split them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// Findings, sorted by severity, entity id and code.
    pub issues: Vec<Issue>,
}

impl Report {
    /// True when the file has no errors (warnings are allowed).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|issue| issue.severity == Severity::Error)
    }

    /// True when the file has neither errors nor warnings.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Number of [`Severity::Error`] issues.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|issue| issue.severity == Severity::Error).count()
    }

    /// Number of [`Severity::Warning`] issues.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|issue| issue.severity == Severity::Warning).count()
    }

    /// Iterate over the errors, in report order.
    pub fn errors(&self) -> impl Iterator<Item = &Issue> + '_ {
        self.issues.iter().filter(|issue| issue.severity == Severity::Error)
    }

    /// Iterate over the warnings, in report order.
    pub fn warnings(&self) -> impl Iterator<Item = &Issue> + '_ {
        self.issues.iter().filter(|issue| issue.severity == Severity::Warning)
    }

    /// Whether any issue (error or warning) carries `code`.
    #[must_use]
    pub fn has_code(&self, code: &str) -> bool {
        self.issues.iter().any(|issue| issue.code == code)
    }

    /// Whether an error carries `code`.
    #[must_use]
    pub fn has_error(&self, code: &str) -> bool {
        self.errors().any(|issue| issue.code == code)
    }

    /// Whether a warning carries `code`.
    #[must_use]
    pub fn has_warning(&self, code: &str) -> bool {
        self.warnings().any(|issue| issue.code == code)
    }

    /// The issue attached to `entity`, if any.
    #[must_use]
    pub fn issues_for(&self, entity: u32) -> impl Iterator<Item = &Issue> + '_ {
        self.issues.iter().filter(move |issue| issue.entity == Some(entity))
    }

    /// `"N errors, M warnings"` (singular/plural handled).
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} {}, {} {}",
            self.error_count(),
            if self.error_count() == 1 { "error" } else { "errors" },
            self.warning_count(),
            if self.warning_count() == 1 { "warning" } else { "warnings" },
        )
    }

    pub(crate) fn push_error(
        &mut self,
        entity: Option<u32>,
        code: &'static str,
        message: impl Into<String>,
    ) {
        self.issues.push(Issue {
            severity: Severity::Error,
            code,
            entity,
            message: message.into(),
        });
    }

    pub(crate) fn push_warning(
        &mut self,
        entity: Option<u32>,
        code: &'static str,
        message: impl Into<String>,
    ) {
        self.issues.push(Issue {
            severity: Severity::Warning,
            code,
            entity,
            message: message.into(),
        });
    }

    pub(crate) fn finish(&mut self) {
        self.issues
            .sort_by_key(|issue| (issue.severity, issue.entity, issue.code));
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.issues.is_empty() {
            return formatter.write_str("valid STEP (no issues)");
        }
        for issue in &self.issues {
            writeln!(formatter, "- {issue}")?;
        }
        write!(formatter, "{}", self.summary())
    }
}

/// Validate an already-parsed STEP file.
///
/// Runs every pass in a fixed order and returns a sorted [`Report`]. The file is not
/// modified.
#[must_use]
pub fn validate(file: &StepFile) -> Report {
    let mut report = Report::default();
    if file.entities.is_empty() {
        report.push_warning(None, codes::EMPTY_FILE, "STEP DATA section has no entities");
    }

    checks::dangling_references(file, &mut report);
    checks::value_sanity(file, &mut report);

    for (id, entity) in &file.entities {
        if entity.name.is_empty() {
            // A complex entity instance; its SI units are checked below.
            continue;
        }
        if !schema::is_known_entity(&entity.name) {
            report.push_warning(
                Some(*id),
                codes::UNKNOWN_ENTITY,
                format!(
                    "entity {} has no schema rule in the NodeCraft STEP subset",
                    entity.name
                ),
            );
            continue;
        }
        schema::check_entity_shape(file, *id, entity, &mut report);
    }

    units::check_unit_enumerations(file, &mut report);
    checks::structure(file, &mut report);

    report.finish();
    report
}

/// Parse STEP text and validate it.
///
/// Parse failures are reported as a single error with code [`codes::PARSE`] instead
/// of being returned as `Err`, so malformed input flows through the same [`Report`]
/// as schema findings.
#[must_use]
pub fn validate_text(text: &str) -> Report {
    match StepFile::parse(text) {
        Ok(file) => validate(&file),
        Err(error) => {
            let mut report = Report::default();
            report.push_error(None, codes::PARSE, format!("STEP parse failed: {error}"));
            report.finish();
            report
        }
    }
}

/// Read a STEP file and validate it.
///
/// Read failures are reported as a single error with code [`codes::IO`].
#[must_use]
pub fn validate_file(path: &str) -> Report {
    match std::fs::read_to_string(path) {
        Ok(text) => validate_text(&text),
        Err(error) => {
            let mut report = Report::default();
            report.push_error(None, codes::IO, format!("cannot read {path}: {error}"));
            report.finish();
            report
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_is_a_parse_error() {
        let report = validate_text("HEADER;\nENDSEC;\n");
        assert!(!report.is_valid());
        assert!(report.has_error(codes::PARSE));
        assert_eq!(report.error_count(), 1);
    }

    #[test]
    fn empty_data_section_warns() {
        let report = validate_text("DATA;ENDSEC;");
        assert!(report.is_valid());
        assert!(report.has_warning(codes::EMPTY_FILE));
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn report_display_and_counts() {
        let mut report = Report::default();
        report.push_warning(Some(3), codes::EMPTY_FILE, "warning text");
        report.push_error(Some(1), codes::DANGLING_REFERENCE, "error text");
        report.finish();
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);
        assert_eq!(report.summary(), "1 error, 1 warning");
        let text = report.to_string();
        assert!(text.contains("error dangling-reference: error text (entity #1)"), "{text}");
        assert!(text.contains("warning empty-file: warning text (entity #3)"), "{text}");
        assert!(text.ends_with("1 error, 1 warning"), "{text}");
    }

    #[test]
    fn clean_report_display() {
        assert_eq!(Report::default().to_string(), "valid STEP (no issues)");
    }
    #[test]
    fn curve_bounded_surface_accepts_its_boundary_set_and_flag() {
        let text = "DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=AXIS2_PLACEMENT_3D('',#1,#2,#2);
#4=PLANE('',#3);
#5=CIRCLE('',#3,5.);
#6=CURVE_BOUNDED_SURFACE('',#4,(#5),.F.);
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::ARG_COUNT), "{report}");
        assert!(!report.has_error(codes::ARG_KIND), "{report}");
    }

    #[test]
    fn gear_fixture_has_no_curve_bounded_surface_shape_errors() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/Sample AP203 Data Files/gear.stp");
        if !path.exists() { return; }
        let report = validate_file(path.to_str().unwrap());
        assert!(!report.has_error(codes::ARG_COUNT), "{report}");
        assert!(!report.has_error(codes::ARG_KIND), "{report}");
    }

    #[test]
    fn surface_of_linear_extrusion_accepts_vector_axis() {
        let text = "DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=VECTOR('',#2,1.);
#4=CIRCLE('',#5,2.);
#5=AXIS2_PLACEMENT_3D('',#1,#2,#2);
#6=SURFACE_OF_LINEAR_EXTRUSION('',#4,#3);
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::WRONG_REFERENCE_TYPE), "{report}");
        // A DIRECTION is still invalid in the vector slot: the rule is not loosened to Any.
        let bad = text.replace("#6=SURFACE_OF_LINEAR_EXTRUSION('',#4,#3);", "#6=SURFACE_OF_LINEAR_EXTRUSION('',#4,#2);");
        assert!(validate_text(&bad).has_error(codes::WRONG_REFERENCE_TYPE));
    }

    #[test]
    fn plane_accepts_a2pl3d_placement() {
        let text = "DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=A2PL3D('',#1,#2,#2);
#4=PLANE('',#3);
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::WRONG_REFERENCE_TYPE), "{report}");
        assert!(!report.has_error(codes::ARG_COUNT), "{report}");
        assert!(!report.has_error(codes::ARG_KIND), "{report}");
        // A2PL3D is still shape-checked like AXIS2_PLACEMENT_3D.
        let bad = text.replace("#3=A2PL3D('',#1,#2,#2);", "#3=A2PL3D('',#1,#2);");
        assert!(validate_text(&bad).has_error(codes::ARG_COUNT));
    }

    #[test]
    fn legacy_short_entity_names_are_recognized_in_reference_slots() {
        let text = "DATA;
#1=CRTPNT('',(0.,0.,0.));
#2=DRCTN('',(0.,0.,1.));
#3=VCTR('',#2,1.);
#4=A2PL3D('',#1,#2,#2);
#5=PLANE('',#4);
#6=LINE('',#1,#3);
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::WRONG_REFERENCE_TYPE), "{report}");
        assert!(!report.has_error(codes::ARG_KIND), "{report}");
    }

    #[test]
    fn shape_definition_representation_accepts_property_definition() {        let text = "DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=PROPERTY_DEFINITION('',(),#1);
#3=SHAPE_REPRESENTATION('',(),#1);
#4=SHAPE_DEFINITION_REPRESENTATION(#2,#3);
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::WRONG_REFERENCE_TYPE), "{report}");
        // An unrelated definition type (a point) is still rejected.
        let bad = text.replace("#4=SHAPE_DEFINITION_REPRESENTATION(#2,#3);", "#4=SHAPE_DEFINITION_REPRESENTATION(#1,#3);");
        assert!(validate_text(&bad).has_error(codes::WRONG_REFERENCE_TYPE));
    }

    #[test]
    fn product_description_may_be_optional() {
        let text = "DATA;
#1=APPLICATION_CONTEXT('x');
#2=PRODUCT_CONTEXT('',#1,'mechanical');
#3=PRODUCT('id','name',$,(#2));
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::ARG_KIND), "{report}");
        // A non-text, non-$ description is still invalid.
        let bad = text.replace("PRODUCT('id','name',$,(#2))", "PRODUCT('id','name',7.,(#2))");
        assert!(validate_text(&bad).has_error(codes::ARG_KIND));
    }

    #[test]
    fn si_unit_accepts_degree_celsius() {
        // ISO 10303-41 `si_unit_name` includes `degree_celsius`; AP209 files use it
        // for thermodynamic temperature units.
        let text = "DATA;
#1=(NAMED_UNIT(*)SI_UNIT($,.DEGREE_CELSIUS.)THERMODYNAMIC_TEMPERATURE_UNIT());
ENDSEC;";
        let report = validate_text(text);
        assert!(!report.has_error(codes::INVALID_UNIT), "{report}");
        let bad = text.replace(".DEGREE_CELSIUS.", ".DEGREES_BOGUS.");
        assert!(validate_text(&bad).has_error(codes::INVALID_UNIT));
    }

    #[test]
    fn ats1_fixture_parses_without_parse_error() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-stepcode/ATS1-out.stp");
        if !path.exists() { return; }
        let report = validate_file(path.to_str().unwrap());
        assert!(!report.has_error(codes::PARSE), "{report}");
        // The file itself violates `products : SET [1:?] OF product` on
        // PRODUCT_RELATED_PRODUCT_CATEGORY('document',$,()), so validation still
        // reports that genuine file error; the rule is not relaxed.
        assert!(report.has_error(codes::EMPTY_REFERENCE_LIST), "{report}");
    }

}
