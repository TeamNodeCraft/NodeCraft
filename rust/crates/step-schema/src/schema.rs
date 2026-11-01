//! Entity-shape rules: how many arguments each STEP entity takes, of what kind, and
//! which entity types its references must name.
//!
//! Only rules we are confident about are encoded here. An entity in the broader known
//! set ([`is_known_entity`]) without a slot table is accepted without shape checking;
//! an entity outside the known set only produces an `unknown-entity` warning. That way
//! the validator never rejects valid STEP that merely uses parts of AP214 this crate
//! does not model.

use nodecraft_step::{Entity, StepFile, Value};

use crate::{codes, Report};

/// Constraint on a numeric argument.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NumberRule {
    /// Any finite number.
    Finite,
    /// A finite number greater than zero (radii, vector magnitudes).
    Positive,
    /// A finite number greater than or equal to zero.
    NonNegative,
}

/// Constraint on a reference argument.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RefRule {
    /// Any existing entity.
    Any,
    /// An entity whose name is one of these.
    Named(&'static [&'static str]),
}

/// One argument slot of an entity.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Slot {
    /// A STEP string (`'text'`).
    Text,
    /// A STEP string or `$`.
    OptionalText,
    /// A finite number with the given rule.
    Number(NumberRule),
    /// A coordinate tuple: a list of 2 or 3 finite numbers (STEP allows 2-D points).
    Point,
    /// A required reference.
    Ref(RefRule),
    /// A reference or `$`.
    OptionalRef(RefRule),
    /// A non-empty list of references.
    RefList { rule: RefRule, min: usize },
    /// A STEP boolean enumeration (`.T.`, `.F.` or `.U.`).
    Boolean,
    /// The derived-attribute placeholder `*`.
    Wildcard,
}

// ---------------------------------------------------------------------------------
// Referenced-type name sets
// ---------------------------------------------------------------------------------

// Legacy exporters (notably CATIA and the StepCode stress fixtures) write the Part 21
// short entity names below. Treat them as the entities they abbreviate for reference
// rules; the short-name entities themselves stay outside the shape-checked subset and
// only produce the usual unknown-entity warning.
const POINT_NAMES: &[&str] = &["CARTESIAN_POINT", "CRTPNT"];
const DIRECTION_NAMES: &[&str] = &["DIRECTION", "DRCTN"];
const VECTOR_NAMES: &[&str] = &["VECTOR", "VCTR"];
const PLACEMENT_NAMES: &[&str] =
    &["AXIS2_PLACEMENT_3D", "AXIS2_PLACEMENT_2D", "AXIS1_PLACEMENT", "A2PL3D"];
const CURVE_NAMES: &[&str] = &[
    "LINE",
    "CIRCLE",
    "ELLIPSE",
    "HYPERBOLA",
    "PARABOLA",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_CURVE",
    "BEZIER_CURVE",
    "QUASI_UNIFORM_CURVE",
    "UNIFORM_CURVE",
    "RATIONAL_B_SPLINE_CURVE",
    "TRIMMED_CURVE",
    "COMPOSITE_CURVE",
    "SURFACE_CURVE",
    "SEAM_CURVE",
    "PCURVE",
    "INTERSECTION_CURVE",
    "OFFSET_CURVE_3D",
];
const SURFACE_NAMES: &[&str] = &[
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "B_SPLINE_SURFACE",
    "BEZIER_SURFACE",
    "RATIONAL_B_SPLINE_SURFACE",
    "SURFACE_OF_REVOLUTION",
    "SURFACE_OF_LINEAR_EXTRUSION",
    "OFFSET_SURFACE",
    "RECTANGULAR_TRIMMED_SURFACE",
    "CURVE_BOUNDED_SURFACE",
    "SWEPT_SURFACE",
];
const VERTEX_NAMES: &[&str] = &["VERTEX_POINT"];
const EDGE_NAMES: &[&str] = &["EDGE_CURVE", "ORIENTED_EDGE", "SUBEDGE", "EDGE"];
const LOOP_NAMES: &[&str] =
    &["EDGE_LOOP", "POLY_LOOP", "VERTEX_LOOP", "PATH", "ORIENTED_PATH"];
const BOUND_NAMES: &[&str] = &["FACE_BOUND", "FACE_OUTER_BOUND"];
const FACE_NAMES: &[&str] = &["ADVANCED_FACE", "FACE_SURFACE", "ORIENTED_FACE", "FACE"];
const SHELL_NAMES: &[&str] = &[
    "CLOSED_SHELL",
    "OPEN_SHELL",
    "ORIENTED_CLOSED_SHELL",
    "ORIENTED_OPEN_SHELL",
    "CONNECTED_FACE_SET",
];
const OPEN_SHELL_NAMES: &[&str] = &["OPEN_SHELL", "ORIENTED_OPEN_SHELL"];
const PRODUCT_CONTEXT_NAMES: &[&str] = &["PRODUCT_CONTEXT", "MECHANICAL_CONTEXT", "DESIGN_CONTEXT"];
const APPLICATION_CONTEXT_NAMES: &[&str] =
    &["APPLICATION_CONTEXT", "APPLICATION_PROTOCOL_DEFINITION"];
const PRODUCT_NAMES: &[&str] = &["PRODUCT"];
const REPRESENTATION_NAMES: &[&str] = &[
    "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "SHAPE_REPRESENTATION",
    "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
    "FACETED_BREP_SHAPE_REPRESENTATION",
];
const ORIENTED_EDGE_NAMES: &[&str] = &["ORIENTED_EDGE"];

// ---------------------------------------------------------------------------------
// Slot tables
// ---------------------------------------------------------------------------------

type S = Slot;

const CARTESIAN_POINT_SLOTS: [S; 2] = [S::Text, S::Point];
const DIRECTION_SLOTS: [S; 2] = [S::Text, S::Point];
const VECTOR_SLOTS: [S; 3] =
    [S::Text, S::Ref(RefRule::Named(DIRECTION_NAMES)), S::Number(NumberRule::Positive)];
const LINE_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(POINT_NAMES)),
    S::Ref(RefRule::Named(VECTOR_NAMES)),
];
const AXIS2_PLACEMENT_3D_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(POINT_NAMES)),
    S::OptionalRef(RefRule::Named(DIRECTION_NAMES)),
    S::OptionalRef(RefRule::Named(DIRECTION_NAMES)),
];
const AXIS2_PLACEMENT_2D_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(POINT_NAMES)),
    S::OptionalRef(RefRule::Named(DIRECTION_NAMES)),
];
const AXIS1_PLACEMENT_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(POINT_NAMES)),
    S::Ref(RefRule::Named(DIRECTION_NAMES)),
];
const CIRCLE_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
];
const PLANE_SLOTS: [S; 2] = [S::Text, S::Ref(RefRule::Named(PLACEMENT_NAMES))];
const CYLINDRICAL_SURFACE_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
];
const CONICAL_SURFACE_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::NonNegative),
    S::Number(NumberRule::NonNegative),
];
const SPHERICAL_SURFACE_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
];
const TOROIDAL_SURFACE_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
    S::Number(NumberRule::Positive),
];
const ELLIPSE_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
    S::Number(NumberRule::Positive),
];
const HYPERBOLA_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
    S::Number(NumberRule::Positive),
];
const PARABOLA_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
    S::Number(NumberRule::Positive),
];
const VERTEX_POINT_SLOTS: [S; 2] = [S::Text, S::Ref(RefRule::Named(POINT_NAMES))];
const EDGE_CURVE_SLOTS: [S; 5] = [
    S::Text,
    S::Ref(RefRule::Named(VERTEX_NAMES)),
    S::Ref(RefRule::Named(VERTEX_NAMES)),
    S::Ref(RefRule::Named(CURVE_NAMES)),
    S::Boolean,
];
const ORIENTED_EDGE_SLOTS: [S; 5] = [
    S::Text,
    S::Wildcard,
    S::Wildcard,
    S::Ref(RefRule::Named(EDGE_NAMES)),
    S::Boolean,
];
const EDGE_LOOP_SLOTS: [S; 2] =
    [S::Text, S::RefList { rule: RefRule::Named(ORIENTED_EDGE_NAMES), min: 1 }];
const FACE_OUTER_BOUND_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(LOOP_NAMES)),
    S::Boolean,
];
const FACE_BOUND_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(LOOP_NAMES)),
    S::Boolean,
];
const ADVANCED_FACE_SLOTS: [S; 4] = [
    S::Text,
    S::RefList { rule: RefRule::Named(BOUND_NAMES), min: 1 },
    S::Ref(RefRule::Named(SURFACE_NAMES)),
    S::Boolean,
];
const FACE_SURFACE_SLOTS: [S; 4] = [
    S::Text,
    S::RefList { rule: RefRule::Named(BOUND_NAMES), min: 1 },
    S::Ref(RefRule::Named(SURFACE_NAMES)),
    S::Boolean,
];
const CLOSED_SHELL_SLOTS: [S; 2] =
    [S::Text, S::RefList { rule: RefRule::Named(FACE_NAMES), min: 1 }];
const OPEN_SHELL_SLOTS: [S; 2] =
    [S::Text, S::RefList { rule: RefRule::Named(FACE_NAMES), min: 1 }];
const CONNECTED_FACE_SET_SLOTS: [S; 2] =
    [S::Text, S::RefList { rule: RefRule::Named(FACE_NAMES), min: 1 }];
const MANIFOLD_SOLID_BREP_SLOTS: [S; 2] = [S::Text, S::Ref(RefRule::Named(SHELL_NAMES))];
const BREP_WITH_VOIDS_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(SHELL_NAMES)),
    S::RefList { rule: RefRule::Named(OPEN_SHELL_NAMES), min: 0 },
];
const FACETED_BREP_SLOTS: [S; 2] = [S::Text, S::Ref(RefRule::Named(SHELL_NAMES))];
const SHELL_BASED_SURFACE_MODEL_SLOTS: [S; 2] =
    [S::Text, S::RefList { rule: RefRule::Named(SHELL_NAMES), min: 1 }];
const APPLICATION_CONTEXT_SLOTS: [S; 1] = [S::Text];
const PRODUCT_CONTEXT_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(APPLICATION_CONTEXT_NAMES)),
    S::Text,
];
const PRODUCT_SLOTS: [S; 4] = [
    S::Text,
    S::Text,
    S::OptionalText,
    S::RefList { rule: RefRule::Named(PRODUCT_CONTEXT_NAMES), min: 1 },
];
const PRODUCT_RELATED_PRODUCT_CATEGORY_SLOTS: [S; 3] = [
    S::Text,
    S::OptionalText,
    S::RefList { rule: RefRule::Named(PRODUCT_NAMES), min: 1 },
];
const PRODUCT_DEFINITION_FORMATION_SLOTS: [S; 3] = [
    S::Text,
    S::OptionalText,
    S::Ref(RefRule::Named(PRODUCT_NAMES)),
];
const PRODUCT_DEFINITION_SHAPE_SLOTS: [S; 3] = [S::Text, S::OptionalText, S::Ref(RefRule::Any)];
const SHAPE_DEFINITION_REPRESENTATION_SLOTS: [S; 2] = [
    // AP214/AP242 `representation_or_shape_definition`: both
    // PRODUCT_DEFINITION_SHAPE and the base PROPERTY_DEFINITION are legal.
    S::Ref(RefRule::Named(&["PRODUCT_DEFINITION_SHAPE", "PROPERTY_DEFINITION"])),
    S::Ref(RefRule::Any),
];
const ADVANCED_BREP_SHAPE_REPRESENTATION_SLOTS: [S; 3] =
    [S::Text, S::RefList { rule: RefRule::Any, min: 1 }, S::Ref(RefRule::Any)];
const SHAPE_REPRESENTATION_SLOTS: [S; 3] =
    [S::Text, S::RefList { rule: RefRule::Any, min: 0 }, S::Ref(RefRule::Any)];
const MANIFOLD_SURFACE_SHAPE_REPRESENTATION_SLOTS: [S; 3] =
    [S::Text, S::RefList { rule: RefRule::Any, min: 1 }, S::Ref(RefRule::Any)];
const REPRESENTATION_RELATIONSHIP_SLOTS: [S; 4] =
    [S::Text, S::OptionalText, S::Ref(RefRule::Any), S::Ref(RefRule::Any)];
const REPRESENTATION_MAP_SLOTS: [S; 3] =
    [S::Ref(RefRule::Any), S::Ref(RefRule::Any), S::Ref(RefRule::Any)];
const MAPPED_ITEM_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(&["REPRESENTATION_MAP"])),
    S::Ref(RefRule::Named(REPRESENTATION_NAMES)),
];
const ITEM_DEFINED_TRANSFORMATION_SLOTS: [S; 4] =
    [S::Text, S::OptionalText, S::Ref(RefRule::Any), S::Ref(RefRule::Any)];
const CONTEXT_DEPENDENT_SHAPE_REPRESENTATION_SLOTS: [S; 2] =
    [S::Ref(RefRule::Any), S::Ref(RefRule::Any)];
const SURFACE_OF_REVOLUTION_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(CURVE_NAMES)),
    S::Ref(RefRule::Named(PLACEMENT_NAMES)),
];
const SURFACE_OF_LINEAR_EXTRUSION_SLOTS: [S; 3] = [
    S::Text,
    S::Ref(RefRule::Named(CURVE_NAMES)),
    // EXPRESS: extrusion_axis : vector (not direction).
    S::Ref(RefRule::Named(VECTOR_NAMES)),
];
const OFFSET_SURFACE_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(SURFACE_NAMES)),
    S::Number(NumberRule::Finite),
    S::Boolean,
];
const RECTANGULAR_TRIMMED_SURFACE_SLOTS: [S; 6] = [
    S::Text,
    S::Ref(RefRule::Named(SURFACE_NAMES)),
    S::Number(NumberRule::Finite),
    S::Number(NumberRule::Finite),
    S::Number(NumberRule::Finite),
    S::Number(NumberRule::Finite),
];
const CURVE_BOUNDED_SURFACE_SLOTS: [S; 4] = [
    S::Text,
    S::Ref(RefRule::Named(SURFACE_NAMES)),
    S::RefList { rule: RefRule::Any, min: 1 },
    S::Boolean,
];

/// The slot table for a known entity, or `None` when the entity is known but not
/// shape-checked.
fn slots_for(name: &str) -> Option<&'static [Slot]> {
    Some(match name {
        "CARTESIAN_POINT" => &CARTESIAN_POINT_SLOTS,
        "DIRECTION" => &DIRECTION_SLOTS,
        "VECTOR" => &VECTOR_SLOTS,
        "LINE" => &LINE_SLOTS,
        "AXIS2_PLACEMENT_3D" => &AXIS2_PLACEMENT_3D_SLOTS,
        // `A2PL3D` is the AP203 short form of AXIS2_PLACEMENT_3D written by some
        // exporters (e.g. the socks.stp stress fixture): identical four arguments.
        "A2PL3D" => &AXIS2_PLACEMENT_3D_SLOTS,
        "AXIS2_PLACEMENT_2D" => &AXIS2_PLACEMENT_2D_SLOTS,
        "AXIS1_PLACEMENT" => &AXIS1_PLACEMENT_SLOTS,
        "CIRCLE" => &CIRCLE_SLOTS,
        "PLANE" => &PLANE_SLOTS,
        "CYLINDRICAL_SURFACE" => &CYLINDRICAL_SURFACE_SLOTS,
        "CONICAL_SURFACE" => &CONICAL_SURFACE_SLOTS,
        "SPHERICAL_SURFACE" => &SPHERICAL_SURFACE_SLOTS,
        "TOROIDAL_SURFACE" => &TOROIDAL_SURFACE_SLOTS,
        "ELLIPSE" => &ELLIPSE_SLOTS,
        "HYPERBOLA" => &HYPERBOLA_SLOTS,
        "PARABOLA" => &PARABOLA_SLOTS,
        "VERTEX_POINT" => &VERTEX_POINT_SLOTS,
        "EDGE_CURVE" => &EDGE_CURVE_SLOTS,
        "ORIENTED_EDGE" => &ORIENTED_EDGE_SLOTS,
        "EDGE_LOOP" => &EDGE_LOOP_SLOTS,
        "FACE_OUTER_BOUND" => &FACE_OUTER_BOUND_SLOTS,
        "FACE_BOUND" => &FACE_BOUND_SLOTS,
        "ADVANCED_FACE" => &ADVANCED_FACE_SLOTS,
        "FACE_SURFACE" => &FACE_SURFACE_SLOTS,
        "CLOSED_SHELL" => &CLOSED_SHELL_SLOTS,
        "OPEN_SHELL" => &OPEN_SHELL_SLOTS,
        "CONNECTED_FACE_SET" => &CONNECTED_FACE_SET_SLOTS,
        "MANIFOLD_SOLID_BREP" => &MANIFOLD_SOLID_BREP_SLOTS,
        "BREP_WITH_VOIDS" => &BREP_WITH_VOIDS_SLOTS,
        "FACETED_BREP" => &FACETED_BREP_SLOTS,
        "SHELL_BASED_SURFACE_MODEL" => &SHELL_BASED_SURFACE_MODEL_SLOTS,
        "APPLICATION_CONTEXT" => &APPLICATION_CONTEXT_SLOTS,
        "PRODUCT_CONTEXT" => &PRODUCT_CONTEXT_SLOTS,
        "PRODUCT" => &PRODUCT_SLOTS,
        "PRODUCT_RELATED_PRODUCT_CATEGORY" => &PRODUCT_RELATED_PRODUCT_CATEGORY_SLOTS,
        "PRODUCT_DEFINITION_FORMATION" => &PRODUCT_DEFINITION_FORMATION_SLOTS,
        "PRODUCT_DEFINITION_SHAPE" => &PRODUCT_DEFINITION_SHAPE_SLOTS,
        "SHAPE_DEFINITION_REPRESENTATION" => &SHAPE_DEFINITION_REPRESENTATION_SLOTS,
        "ADVANCED_BREP_SHAPE_REPRESENTATION" => &ADVANCED_BREP_SHAPE_REPRESENTATION_SLOTS,
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION" => {
            &MANIFOLD_SURFACE_SHAPE_REPRESENTATION_SLOTS
        }
        "SHAPE_REPRESENTATION" => &SHAPE_REPRESENTATION_SLOTS,
        "REPRESENTATION_RELATIONSHIP" => &REPRESENTATION_RELATIONSHIP_SLOTS,
        "REPRESENTATION_MAP" => &REPRESENTATION_MAP_SLOTS,
        "MAPPED_ITEM" => &MAPPED_ITEM_SLOTS,
        "ITEM_DEFINED_TRANSFORMATION" => &ITEM_DEFINED_TRANSFORMATION_SLOTS,
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" => {
            &CONTEXT_DEPENDENT_SHAPE_REPRESENTATION_SLOTS
        }
        "SURFACE_OF_REVOLUTION" => &SURFACE_OF_REVOLUTION_SLOTS,
        "SURFACE_OF_LINEAR_EXTRUSION" => &SURFACE_OF_LINEAR_EXTRUSION_SLOTS,
        "OFFSET_SURFACE" => &OFFSET_SURFACE_SLOTS,
        "RECTANGULAR_TRIMMED_SURFACE" => &RECTANGULAR_TRIMMED_SURFACE_SLOTS,
        "CURVE_BOUNDED_SURFACE" => &CURVE_BOUNDED_SURFACE_SLOTS,
        _ => return None,
    })
}

/// Entities that belong to the STEP subset but have no shape rule here (either their
/// argument list contains nested typed values, or the rule is too version-dependent
/// to assert safely). They are accepted without an `unknown-entity` warning.
const EXTRA_KNOWN: &[&str] = &[
    // measurements / representation infrastructure
    "UNCERTAINTY_MEASURE_WITH_UNIT",
    "MEASURE_WITH_UNIT",
    "LENGTH_MEASURE_WITH_UNIT",
    "REPRESENTATION_CONTEXT",
    "GEOMETRIC_REPRESENTATION_CONTEXT",
    "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT",
    "GLOBAL_UNIT_ASSIGNED_CONTEXT",
    "SHAPE_REPRESENTATION_RELATIONSHIP",
    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
    // product structure variants written by other exporters
    "PRODUCT_DEFINITION",
    "PRODUCT_DEFINITION_CONTEXT",
    "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
    "PRODUCT_DEFINITION_RELATIONSHIP",
    "PRODUCT_DEFINITION_WITH_ASSOCIATED_DOCUMENTS",
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
    "ASSEMBLY_COMPONENT_USAGE",
    "PRODUCT_CATEGORY",
    "PRODUCT_CATEGORY_RELATIONSHIP",
    "DESIGN_CONTEXT",
    "APPLICATION_PROTOCOL_DEFINITION",
    // curves / surfaces whose argument lists embed nested typed values
    "B_SPLINE_CURVE",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "BEZIER_CURVE",
    "QUASI_UNIFORM_CURVE",
    "UNIFORM_CURVE",
    "RATIONAL_B_SPLINE_CURVE",
    "TRIMMED_CURVE",
    "COMPOSITE_CURVE",
    "COMPOSITE_CURVE_SEGMENT",
    "PCURVE",
    "SURFACE_CURVE",
    "SEAM_CURVE",
    "B_SPLINE_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "BEZIER_SURFACE",
    "RATIONAL_B_SPLINE_SURFACE",
    "SWEPT_SURFACE",
    // topology
    "SUBEDGE",
    "EDGE",
    "ORIENTED_PATH",
    "PATH",
    "POLY_LOOP",
    "VERTEX_LOOP",
    "ORIENTED_FACE",
    "ORIENTED_CLOSED_SHELL",
    "ORIENTED_OPEN_SHELL",
    "MANIFOLD_SOLID_BREP_WITH_VOIDS",
    "SOLID_MODEL",
    "SOLID_MODEL_BY_EXTRUSION",
    "SOLID_MODEL_BY_REVOLUTION",
    "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
    // units (usually complex instances, listed for tolerance)
    "LENGTH_UNIT",
    "NAMED_UNIT",
    "SI_UNIT",
    "PLANE_ANGLE_UNIT",
    "SOLID_ANGLE_UNIT",
    "CONVERSION_BASED_UNIT",
    "DIMENSIONAL_EXPONENTS",
];

/// Whether the schema subset knows `name` (whether or not it has a slot rule).
pub(crate) fn is_known_entity(name: &str) -> bool {
    slots_for(name).is_some() || EXTRA_KNOWN.contains(&name)
}

/// Check one entity's argument shape, emitting issues into `report`.
pub(crate) fn check_entity_shape(
    file: &StepFile,
    id: u32,
    entity: &Entity,
    report: &mut Report,
) {
    let Some(slots) = slots_for(&entity.name) else {
        return;
    };
    if entity.args.len() != slots.len() {
        report.push_error(
            Some(id),
            codes::ARG_COUNT,
            format!(
                "{} expects {} argument{}, found {}",
                entity.name,
                slots.len(),
                if slots.len() == 1 { "" } else { "s" },
                entity.args.len()
            ),
        );
    }
    for (index, slot) in slots.iter().enumerate() {
        let Some(value) = entity.args.get(index) else {
            break;
        };
        check_slot(file, id, &entity.name, index, *slot, value, report);
    }
}

#[allow(clippy::too_many_lines)]
fn check_slot(
    file: &StepFile,
    id: u32,
    entity: &str,
    index: usize,
    slot: Slot,
    value: &Value,
    report: &mut Report,
) {
    let argument = index + 1;
    match slot {
        Slot::Text => {
            if !matches!(value, Value::Str(_)) {
                kind_error(report, id, entity, argument, "a string", value);
            }
        }
        Slot::OptionalText => {
            if !matches!(value, Value::Str(_) | Value::Null) {
                kind_error(report, id, entity, argument, "a string or `$`", value);
            }
        }
        Slot::Number(rule) => match value {
            Value::Number(number) => check_number(report, id, entity, argument, rule, value, *number),
            _ => kind_error(report, id, entity, argument, "a number", value),
        },
        Slot::Point => match value {
            Value::List(values) => {
                if !(2..=3).contains(&values.len()) {
                    report.push_error(
                        Some(id),
                        codes::POINT_DIMENSION,
                        format!(
                            "{entity} argument {argument} has {} coordinates; expected 2 or 3",
                            values.len()
                        ),
                    );
                }
                for (coordinate, value) in values.iter().enumerate() {
                    match value {
                        Value::Number(number) if number.is_finite() => {}
                        Value::Number(_) => report.push_error(
                            Some(id),
                            codes::INVALID_NUMBER,
                            format!(
                                "{entity} argument {argument} coordinate {} is not finite",
                                coordinate + 1
                            ),
                        ),
                        other => report.push_error(
                            Some(id),
                            codes::ARG_KIND,
                            format!(
                                "{entity} argument {argument} coordinate {} must be a number, found {}",
                                coordinate + 1,
                                describe(other)
                            ),
                        ),
                    }
                }
            }
            other => kind_error(report, id, entity, argument, "a coordinate list", other),
        },
        Slot::Ref(rule) => check_reference(file, id, entity, argument, rule, value, report, false),
        Slot::OptionalRef(rule) => {
            if matches!(value, Value::Null) {
                return;
            }
            check_reference(file, id, entity, argument, rule, value, report, false);
        }
        Slot::RefList { rule, min } => match value {
            Value::List(values) => {
                if values.len() < min {
                    report.push_error(
                        Some(id),
                        codes::EMPTY_REFERENCE_LIST,
                        format!(
                            "{entity} argument {argument} must reference at least {min} entit{}",
                            if min == 1 { "y" } else { "ies" }
                        ),
                    );
                }
                for value in values {
                    check_reference(file, id, entity, argument, rule, value, report, true);
                }
            }
            other => kind_error(report, id, entity, argument, "a list of references", other),
        },
        Slot::Boolean => match value {
            Value::Enum(flag)
                if flag.eq_ignore_ascii_case("T")
                    || flag.eq_ignore_ascii_case("F")
                    || flag.eq_ignore_ascii_case("U") => {}
            Value::Enum(flag) => report.push_error(
                Some(id),
                codes::INVALID_BOOLEAN_FLAG,
                format!(
                    "{entity} argument {argument} has boolean flag .{flag}.; expected .T., .F. or .U."
                ),
            ),
            other => kind_error(report, id, entity, argument, "a boolean flag", other),
        },
        Slot::Wildcard => {
            let is_wildcard = matches!(value, Value::Null)
                || matches!(value, Value::Typed(name, _) if name == "*");
            if !is_wildcard {
                kind_error(report, id, entity, argument, "the derived-value placeholder `*`", value);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_number(
    report: &mut Report,
    id: u32,
    entity: &str,
    argument: usize,
    rule: NumberRule,
    value: &Value,
    number: f64,
) {
    if !number.is_finite() {
        report.push_error(
            Some(id),
            codes::INVALID_NUMBER,
            format!("{entity} argument {argument} is not a finite number: {value:?}"),
        );
        return;
    }
    let problem = match rule {
        NumberRule::Finite => None,
        NumberRule::Positive if number <= 0.0 => Some("greater than zero"),
        NumberRule::NonNegative if number < 0.0 => Some("zero or greater"),
        NumberRule::Positive | NumberRule::NonNegative => None,
    };
    if let Some(requirement) = problem {
        report.push_error(
            Some(id),
            codes::INVALID_DIMENSION,
            format!("{entity} argument {argument} must be {requirement}, found {number}"),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn check_reference(
    file: &StepFile,
    id: u32,
    entity: &str,
    argument: usize,
    rule: RefRule,
    value: &Value,
    report: &mut Report,
    in_list: bool,
) {
    let Value::Ref(target) = value else {
        let message = if in_list {
            format!(
                "{entity} argument {argument} list item must be a reference, found {}",
                describe(value)
            )
        } else {
            format!(
                "{entity} argument {argument} must be a reference, found {}",
                describe(value)
            )
        };
        report.push_error(Some(id), codes::ARG_KIND, message);
        return;
    };
    let Some(target_entity) = file.entities.get(target) else {
        // The dangling-reference pass reports missing targets; nothing to add here.
        return;
    };
    let RefRule::Named(allowed) = rule else {
        return;
    };
    // A surface/curve may be written as a complex entity instance
    // (`#5=(BOUNDED_SURFACE() B_SPLINE_SURFACE(...) …);`, empty entity name). Match its
    // typed members, otherwise valid rational B-spline references would be rejected.
    let target_names = entity_type_names(target_entity);
    let target_label = if target_entity.name.is_empty() { "complex entity" } else { target_entity.name.as_str() };
    if !target_names.iter().any(|name| allowed.contains(name)) {
        report.push_error(
            Some(id),
            codes::WRONG_REFERENCE_TYPE,
            format!(
                "{entity} argument {argument} references #{target} ({target_label}); {}",
                expected_reference(allowed)
            ),
        );
    }
}

fn expected_reference(allowed: &[&str]) -> String {
    match allowed {
        [single] => format!("expected {single}"),
        _ => format!("expected one of {}", allowed.join(", ")),
    }
}

fn kind_error(
    report: &mut Report,
    id: u32,
    entity: &str,
    argument: usize,
    expected: &str,
    value: &Value,
) {
    report.push_error(
        Some(id),
        codes::ARG_KIND,
        format!("{entity} argument {argument} must be {expected}, found {}", describe(value)),
    );
}

/// The entity's own type name plus the typed member names of a complex instance.
///
/// Simple entities yield `[name]`; complex instances (`name == ""`) yield their flattened
/// `Type(...)` members such as `B_SPLINE_SURFACE_WITH_KNOTS`.
fn entity_type_names(entity: &Entity) -> Vec<&str> {
    let mut names = Vec::new();
    if !entity.name.is_empty() {
        names.push(entity.name.as_str());
    }
    for value in &entity.args {
        if let Value::Typed(name, _) = value {
            if !name.is_empty() && name != "*" {
                names.push(name.as_str());
            }
        }
    }
    names
}

/// Short description of a value for error messages.
fn describe(value: &Value) -> String {
    match value {
        Value::Ref(id) => format!("reference #{id}"),
        Value::Str(text) => format!("the string {text:?}"),
        Value::Number(number) => format!("the number {number}"),
        Value::Enum(name) => format!("the enumeration .{name}."),
        Value::List(_) => "a list".into(),
        Value::Typed(name, _) if name == "*" => "`*`".into(),
        Value::Typed(name, _) => format!("the typed value {name}(…)"),
        Value::Null => "`$`".into(),
    }
}
