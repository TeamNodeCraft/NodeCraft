//! Whole-file passes: reference integrity, value hygiene, and product-structure
//! warnings.

use nodecraft_step::{StepFile, Value};

use crate::units;
use crate::{codes, Report};

/// Report every `#id` reference that names an entity the file does not contain.
///
/// The walk is recursive through lists and typed values, so references nested in
/// aggregate arguments (`(#1,#2)`, `LENGTH_MEASURE(#3)`, …) are found too. One error is
/// reported per (referencing entity, missing id) pair.
pub(crate) fn dangling_references(file: &StepFile, report: &mut Report) {
    for (id, entity) in &file.entities {
        let mut missing = Vec::new();
        for value in &entity.args {
            collect_missing(file, value, &mut missing);
        }
        missing.sort_unstable();
        missing.dedup();
        for target in missing {
            report.push_error(
                Some(*id),
                codes::DANGLING_REFERENCE,
                format!("references undefined entity #{target}"),
            );
        }
    }
}

fn collect_missing(file: &StepFile, value: &Value, out: &mut Vec<u32>) {
    match value {
        Value::Ref(id) => {
            if !file.entities.contains_key(id) {
                out.push(*id);
            }
        }
        Value::List(values) => {
            for value in values {
                collect_missing(file, value, out);
            }
        }
        Value::Typed(_, values) => {
            for value in values {
                collect_missing(file, value, out);
            }
        }
        _ => {}
    }
}

/// Non-finite numbers and non-canonical enumeration casing.
pub(crate) fn value_sanity(file: &StepFile, report: &mut Report) {
    for (id, entity) in &file.entities {
        let mut non_finite = Vec::new();
        collect_non_finite(&entity.args, &mut non_finite);
        non_finite.sort_by(f64::total_cmp);
        non_finite.dedup();
        for number in non_finite {
            report.push_error(
                Some(*id),
                codes::INVALID_NUMBER,
                format!("entity {} has the non-finite number {number}", display_name(entity.name.as_str())),
            );
        }
        let mut seen = Vec::new();
        for name in units::non_canonical_enums(entity) {
            if seen.contains(&name) {
                continue;
            }
            seen.push(name);
            report.push_warning(
                Some(*id),
                codes::NON_CANONICAL_ENUM,
                format!("enumeration .{name}. is not canonical uppercase STEP"),
            );
        }
    }
}

fn collect_non_finite(values: &[Value], out: &mut Vec<f64>) {
    for value in values {
        match value {
            Value::Number(number) if !number.is_finite() => out.push(*number),
            Value::List(items) => collect_non_finite(items, out),
            Value::Typed(_, items) => collect_non_finite(items, out),
            _ => {}
        }
    }
}

/// Product/representation/unit structure warnings.
///
/// These are warnings rather than errors: a bare B-rep is still geometrically usable,
/// it just is not a complete AP214 product transfer.
pub(crate) fn structure(file: &StepFile, report: &mut Report) {
    if file.entities.is_empty() {
        return;
    }

    let mut solids = Vec::new();
    let mut product = None;
    let mut representation = None;
    let mut product_definition = None;
    let mut first_point = None;
    let mut geometry = false;

    for (id, entity) in &file.entities {
        match entity.name.as_str() {
            "MANIFOLD_SOLID_BREP"
            | "BREP_WITH_VOIDS"
            | "FACETED_BREP"
            | "MANIFOLD_SOLID_BREP_WITH_VOIDS" => solids.push(*id),
            "PRODUCT" => product = product.or(Some(*id)),
            "ADVANCED_BREP_SHAPE_REPRESENTATION"
            | "SHAPE_REPRESENTATION"
            | "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
            | "FACETED_BREP_SHAPE_REPRESENTATION"
            | "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION" => {
                representation = representation.or(Some(*id));
            }
            "PRODUCT_DEFINITION_SHAPE" | "SHAPE_DEFINITION_REPRESENTATION" => {
                product_definition = product_definition.or(Some(*id));
            }
            "CARTESIAN_POINT" => {
                first_point = first_point.or(Some(*id));
                geometry = true;
            }
            "VERTEX_POINT" | "EDGE_CURVE" | "ADVANCED_FACE" | "CLOSED_SHELL" => {
                geometry = true;
            }
            _ => {}
        }
    }

    if let Some(solid) = solids.first() {
        if product.is_none() {
            report.push_warning(
                Some(*solid),
                codes::MISSING_PRODUCT,
                "solid B-rep has no PRODUCT entity",
            );
        }
        match (representation, product_definition) {
            (None, _) => report.push_warning(
                Some(*solid),
                codes::MISSING_SHAPE_REPRESENTATION,
                "solid B-rep is not referenced by any shape representation entity",
            ),
            (Some(_), None) => report.push_warning(
                Some(*solid),
                codes::MISSING_PRODUCT_DEFINITION,
                "shape representation is not linked to a PRODUCT_DEFINITION_SHAPE",
            ),
            (Some(_), Some(_)) => {}
        }
    }

    if geometry && !units::has_length_unit(file) {
        let entity = first_point.or_else(|| solids.first().copied());
        report.push_warning(
            entity,
            codes::MISSING_LENGTH_UNIT,
            "geometry is present but the file declares no SI length unit (.METRE.)",
        );
    }
}

fn display_name(name: &str) -> &str {
    if name.is_empty() { "complex instance" } else { name }
}
