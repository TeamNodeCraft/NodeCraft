//! Validation tests over inline (often deliberately malformed) STEP text, plus
//! round-trips through the `nodecraft-step` writers.

use step_schema::{codes, validate_file, validate_text, Severity};

/// A schema-valid (if geometrically degenerate) planar solid without the AP214 product
/// wrapper: one triangle-less face made of a line edge. Structure checks warn about the
/// missing product/representation/length unit, but there are no errors.
const MINIMAL_SOLID: &str = "DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=PLANE('',#4);
#6=VERTEX_POINT('',#1);
#7=VERTEX_POINT('',#1);
#8=DIRECTION('',(0.,0.,1.));
#9=VECTOR('',#8,1.);
#10=LINE('',#1,#9);
#11=EDGE_CURVE('',#6,#7,#10,.T.);
#12=ORIENTED_EDGE('',*,*,#11,.T.);
#13=EDGE_LOOP('',(#12));
#14=FACE_OUTER_BOUND('',#13,.T.);
#15=ADVANCED_FACE('',(#14),#5,.T.);
#16=CLOSED_SHELL('',(#15));
#17=MANIFOLD_SOLID_BREP('solid',#16);
ENDSEC;
";

/// `MINIMAL_SOLID` with `append` inserted just before `ENDSEC;`.
fn with_appended(append: &str) -> String {
    MINIMAL_SOLID.replace("ENDSEC;", &format!("{append}\nENDSEC;"))
}

/// `MINIMAL_SOLID` with the first occurrence of `from` replaced by `to`.
fn mutated(from: &str, to: &str) -> String {
    assert!(MINIMAL_SOLID.contains(from), "test fixture no longer contains {from:?}");
    MINIMAL_SOLID.replacen(from, to, 1)
}

fn assert_error(report: &step_schema::Report, code: &str) {
    assert!(
        report.has_error(code),
        "expected error {code}, got:\n{report}",
    );
}

fn assert_warning(report: &step_schema::Report, code: &str) {
    assert!(
        report.has_warning(code),
        "expected warning {code}, got:\n{report}",
    );
}

#[test]
fn minimal_solid_has_no_errors_but_structure_warnings() {
    let report = validate_text(MINIMAL_SOLID);
    assert!(report.is_valid(), "unexpected errors:\n{report}");
    assert_eq!(report.error_count(), 0);
    assert_warning(&report, codes::MISSING_PRODUCT);
    assert_warning(&report, codes::MISSING_SHAPE_REPRESENTATION);
    assert_warning(&report, codes::MISSING_LENGTH_UNIT);
    assert_eq!(report.warning_count(), 3, "unexpected extra warnings:\n{report}");
    // The structure warnings point at the solid / first point, not at nothing.
    assert!(report.issues_for(17).any(|issue| issue.code == codes::MISSING_PRODUCT));
    assert!(report
        .issues_for(1)
        .any(|issue| issue.code == codes::MISSING_LENGTH_UNIT));
}

#[test]
fn dangling_reference_is_an_error_attached_to_the_referrer() {
    let report = validate_text(&with_appended("#18=VERTEX_POINT('',#99);"));
    assert!(!report.is_valid());
    assert_error(&report, codes::DANGLING_REFERENCE);
    assert_eq!(report.errors().count(), 1, "unexpected errors:\n{report}");
    let issue = report
        .issues_for(18)
        .next()
        .unwrap_or_else(|| panic!("no issue for #18:\n{report}"));
    assert_eq!(issue.severity, Severity::Error);
    assert!(issue.message.contains("#99"), "{}", issue.message);
}

#[test]
fn dangling_references_are_found_inside_lists_and_typed_values() {
    let report = validate_text(&with_appended(
        "#18=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(#97),#5,'x','');",
    ));
    assert!(report.issues_for(18).any(|issue| {
        issue.severity == Severity::Error
            && issue.code == codes::DANGLING_REFERENCE
            && issue.message.contains("#97")
    }), "unexpected report:\n{report}");

    let report = validate_text(&with_appended("#18=EDGE_LOOP('',(#12,#98));"));
    assert!(report.issues_for(18).any(|issue| {
        issue.code == codes::DANGLING_REFERENCE && issue.message.contains("#98")
    }), "unexpected report:\n{report}");
}

#[test]
fn duplicate_entity_ids_are_a_parse_error() {
    let report = validate_text(
        "DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));#1=CARTESIAN_POINT('',(1.,1.,1.));ENDSEC;",
    );
    assert!(!report.is_valid());
    assert_error(&report, codes::PARSE);
    let message = &report.errors().next().unwrap().message;
    assert!(message.contains("duplicate"), "{message}");
}

#[test]
fn truncated_or_data_less_files_are_parse_errors() {
    for text in ["HEADER;ENDSEC;", "DATA;#1=CARTESIAN_POINT('',(0.,0.,0.)"] {
        let report = validate_text(text);
        assert!(!report.is_valid(), "{text:?}");
        assert_error(&report, codes::PARSE);
    }
}

#[test]
fn wrong_reference_types_are_errors() {
    // A solid must reference a shell, not a plane.
    let report = validate_text(&mutated(
        "#17=MANIFOLD_SOLID_BREP('solid',#16);",
        "#17=MANIFOLD_SOLID_BREP('solid',#5);",
    ));
    assert_error(&report, codes::WRONG_REFERENCE_TYPE);
    let issue = report.issues_for(17).find(|issue| issue.code == codes::WRONG_REFERENCE_TYPE).unwrap();
    assert!(issue.message.contains("PLANE"), "{}", issue.message);

    // A closed shell must contain faces, not points.
    let report = validate_text(&mutated(
        "#16=CLOSED_SHELL('',(#15));",
        "#16=CLOSED_SHELL('',(#1));",
    ));
    assert_error(&report, codes::WRONG_REFERENCE_TYPE);

    // FACE_OUTER_BOUND must reference a loop.
    let report = validate_text(&mutated(
        "#14=FACE_OUTER_BOUND('',#13,.T.);",
        "#14=FACE_OUTER_BOUND('',#5,.T.);",
    ));
    assert_error(&report, codes::WRONG_REFERENCE_TYPE);

    // PRODUCT must reference PRODUCT_CONTEXT entities.
    let report = validate_text(&with_appended("#18=PRODUCT('p','p','',(#1));"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::WRONG_REFERENCE_TYPE));
}

#[test]
fn argument_count_and_kind_are_checked() {
    let report = validate_text(&with_appended("#18=CLOSED_SHELL('',(#15),.T.);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::ARG_COUNT));

    let report = validate_text(&with_appended("#18=CARTESIAN_POINT('',42);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::ARG_KIND));

    let report = validate_text(&with_appended("#18=PLANE(42,#4);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::ARG_KIND));
}

#[test]
fn invalid_boolean_flags_are_errors() {
    let report = validate_text(&mutated(
        "#11=EDGE_CURVE('',#6,#7,#10,.T.);",
        "#11=EDGE_CURVE('',#6,#7,#10,.MAYBE.);",
    ));
    assert_error(&report, codes::INVALID_BOOLEAN_FLAG);
    assert!(report.issues_for(11).any(|issue| issue.code == codes::INVALID_BOOLEAN_FLAG));
}

#[test]
fn lowercase_boolean_is_accepted_but_warned_as_non_canonical() {
    let report = validate_text(&mutated(
        "#11=EDGE_CURVE('',#6,#7,#10,.T.);",
        "#11=EDGE_CURVE('',#6,#7,#10,.t.);",
    ));
    assert!(report.is_valid(), "lowercase .t. should not be an error:\n{report}");
    assert_warning(&report, codes::NON_CANONICAL_ENUM);
}

#[test]
fn point_dimension_is_checked_but_two_d_points_are_allowed() {
    let report = validate_text(&with_appended("#18=CARTESIAN_POINT('',(1.,2.,3.,4.));"));
    assert_error(&report, codes::POINT_DIMENSION);

    let report = validate_text(&with_appended("#18=CARTESIAN_POINT('',(1.,2.));"));
    assert!(!report.has_error(codes::POINT_DIMENSION), "unexpected:\n{report}");
}

#[test]
fn non_finite_numbers_are_errors() {
    let report = validate_text(&with_appended("#18=CARTESIAN_POINT('',(1.,inf,3.));"));
    assert_error(&report, codes::INVALID_NUMBER);
}

#[test]
fn non_positive_dimensions_are_errors() {
    let report = validate_text(&with_appended("#18=CYLINDRICAL_SURFACE('',#4,0.);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::INVALID_DIMENSION));

    let report = validate_text(&with_appended("#18=SPHERICAL_SURFACE('',#4,-2.);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::INVALID_DIMENSION));
}

#[test]
fn empty_required_reference_lists_are_errors() {
    let report = validate_text(&mutated(
        "#16=CLOSED_SHELL('',(#15));",
        "#16=CLOSED_SHELL('',());",
    ));
    assert!(report.issues_for(16).any(|issue| issue.code == codes::EMPTY_REFERENCE_LIST));

    let report = validate_text(&with_appended("#18=ADVANCED_FACE('',(),#5,.T.);"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::EMPTY_REFERENCE_LIST));

    let report = validate_text(&with_appended("#18=PRODUCT('p','p','',());"));
    assert!(report.issues_for(18).any(|issue| issue.code == codes::EMPTY_REFERENCE_LIST));
}

#[test]
fn invalid_si_units_are_errors() {
    let report = validate_text(&with_appended(
        "#18=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.BOGUS.,.METRE.));",
    ));
    assert_error(&report, codes::INVALID_SI_PREFIX);

    let report = validate_text(&with_appended("#18=(NAMED_UNIT(*) SI_UNIT($,.FURLONG.));"));
    assert_error(&report, codes::INVALID_UNIT);
}

#[test]
fn complex_surface_instances_are_valid_face_surfaces() {
    // Rational/plain B-spline surfaces are complex entity instances (empty entity name);
    // an ADVANCED_FACE may reference them and must not be rejected as a wrong type.
    let report = validate_text(&mutated(
        "#5=PLANE('',#4);",
        "#5=(BOUNDED_SURFACE() B_SPLINE_SURFACE_WITH_KNOTS() SURFACE());",
    ));
    assert!(
        !report.has_error(codes::WRONG_REFERENCE_TYPE),
        "unexpected wrong-reference-type:\n{report}"
    );
    assert!(report.is_valid(), "unexpected errors:\n{report}");
}

#[test]
fn unknown_entities_are_warnings_not_errors() {
    let report = validate_text(&with_appended("#18=MYSTERY_ENTITY('x');"));
    assert!(report.is_valid(), "unknown entity should not fail the file:\n{report}");
    assert!(report.issues_for(18).any(|issue| {
        issue.severity == Severity::Warning && issue.code == codes::UNKNOWN_ENTITY
    }), "unexpected report:\n{report}");
}

#[test]
fn missing_length_unit_warns_when_geometry_has_no_metre() {
    let report = validate_text(MINIMAL_SOLID);
    assert_warning(&report, codes::MISSING_LENGTH_UNIT);

    // Adding a length unit removes the warning.
    let text = with_appended("#18=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));");
    let report = validate_text(&text);
    assert!(!report.has_warning(codes::MISSING_LENGTH_UNIT), "unexpected:\n{report}");
}

#[test]
fn adding_product_and_representation_clears_those_warnings() {
    let text = with_appended(
        "#18=APPLICATION_CONTEXT('x');\
#19=PRODUCT_CONTEXT('',#18,'mechanical');\
#20=PRODUCT('p','p','',(#19));\
#21=ADVANCED_BREP_SHAPE_REPRESENTATION('s',(#17),#18);",
    );
    let report = validate_text(&text);
    assert!(!report.has_warning(codes::MISSING_PRODUCT), "unexpected:\n{report}");
    assert!(
        !report.has_warning(codes::MISSING_SHAPE_REPRESENTATION),
        "unexpected:\n{report}"
    );
    // ... but the representation is still not linked to a product definition shape.
    assert_warning(&report, codes::MISSING_PRODUCT_DEFINITION);

    let text = with_appended(
        "#18=APPLICATION_CONTEXT('x');\
#19=PRODUCT_CONTEXT('',#18,'mechanical');\
#20=PRODUCT('p','p','',(#19));\
#21=ADVANCED_BREP_SHAPE_REPRESENTATION('s',(#17),#18);\
#22=PRODUCT_DEFINITION_SHAPE('','',#21);\
#23=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));",
    );
    let report = validate_text(&text);
    assert!(!report.has_warning(codes::MISSING_PRODUCT_DEFINITION), "unexpected:\n{report}");
    assert!(report.is_valid(), "unexpected errors:\n{report}");
}

#[test]
fn empty_data_section_warns_but_is_valid() {
    let report = validate_text("DATA;ENDSEC;");
    assert!(report.is_valid());
    assert_eq!(report.warning_count(), 1);
    assert_warning(&report, codes::EMPTY_FILE);
}

#[test]
fn report_api_helpers_agree() {
    let report = validate_text(MINIMAL_SOLID);
    assert_eq!(report.errors().count(), report.error_count());
    assert_eq!(report.warnings().count(), report.warning_count());
    assert_eq!(report.summary(), "0 errors, 3 warnings");
    assert!(report.has_code(codes::MISSING_PRODUCT));
    assert!(!report.has_code("no-such-code"));
    assert!(report.to_string().contains("missing-product"), "{report}");
    assert!(report.issues_for(9999).next().is_none());
}

#[test]
fn validate_file_reports_io_errors() {
    let path = std::env::temp_dir().join("step-schema-does-not-exist.step");
    let report = validate_file(path.to_str().unwrap());
    assert!(!report.is_valid());
    assert_error(&report, codes::IO);
}

#[test]
fn writer_output_has_no_schema_errors() {
    use nodecraft_step::{write_box, write_cone, write_cylinder, write_sphere, write_torus};

    let dir = std::env::temp_dir();
    let unique = std::process::id();
    let cases: Vec<(&str, Box<dyn Fn(&str) -> Result<(), String>>)> = vec![
        (
            "box",
            Box::new(|path: &str| write_box(path, 10.0, 4.0, 20.0)) as Box<dyn Fn(&str) -> Result<(), String>>,
        ),
        ("cylinder", Box::new(|path: &str| write_cylinder(path, 3.0, 8.0))),
        ("cone", Box::new(|path: &str| write_cone(path, 3.0, 1.0, 8.0))),
        ("sphere", Box::new(|path: &str| write_sphere(path, 4.0))),
        ("torus", Box::new(|path: &str| write_torus(path, 6.0, 2.0))),
    ];

    for (name, write) in cases {
        let path = dir.join(format!("step-schema-writer-{unique}-{name}.step"));
        let path_text = path.to_str().unwrap().to_string();
        write(&path_text).unwrap_or_else(|error| panic!("write {name}: {error}"));
        let report = validate_file(&path_text);
        assert!(report.is_clean(), "{name} should validate cleanly:\n{report}");
        let _ = std::fs::remove_file(&path);
    }
}
