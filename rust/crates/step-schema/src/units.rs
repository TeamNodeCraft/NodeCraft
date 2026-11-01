//! Enumeration ("unit") checks for STEP enumerations that carry meaning.
//!
//! STEP stores most non-boolean enumerations inside complex entity instances, which the
//! parser flattens into an argument list like
//!
//! ```text
//! #5=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
//! ```
//!
//! so there is no entity named `SI_UNIT` to hang a shape rule on. This pass scans for
//! the flattened `SI_UNIT(…)` typed marker followed by its two-element argument list and
//! checks the prefix and unit symbols against the closed SI sets.

use nodecraft_step::{Entity, StepFile, Value};

use crate::{codes, Report};

/// `SI_UNIT` prefixes (`SM_...`/ISO-10303-41 `SI_PREFIX`).
pub(crate) const SI_PREFIXES: &[&str] = &[
    "EXA", "PETA", "TERA", "GIGA", "MEGA", "KILO", "HECTO", "DECA", "DECI", "CENTI",
    "MILLI", "MICRO", "NANO", "PICO", "FEMTO", "ATTO",
];

/// SI unit enumeration symbols allowed in `SI_UNIT` (ISO 10303-41 `si_unit_name`,
/// which includes `degree_celsius` for thermodynamic units).
pub(crate) const SI_UNITS: &[&str] = &[
    "METRE", "GRAM", "SECOND", "AMPERE", "KELVIN", "MOLE", "CANDELA", "RADIAN",
    "STERADIAN", "NEWTON", "PASCAL", "JOULE", "WATT", "COULOMB", "VOLT", "FARAD",
    "OHM", "HENRY", "TESLA", "WEBER", "SIEMENS", "HERTZ", "BECQUEREL", "GRAY",
    "SIEVERT", "LUMEN", "LUX", "DEGREE_CELSIUS",
];

/// Validate `SI_UNIT` arguments in complex instances and in any standalone `SI_UNIT`.
pub(crate) fn check_unit_enumerations(file: &StepFile, report: &mut Report) {
    for (id, entity) in &file.entities {
        if entity.name == "SI_UNIT" {
            check_si_unit(*id, "SI_UNIT", &entity.args, report);
            continue;
        }
        if !entity.name.is_empty() {
            // A plain entity; `SI_UNIT` is only valid as a complex-instance part, and
            // complex instances are the only place the marker appears.
            continue;
        }
        for unit_arguments in si_unit_argument_lists(&entity.args) {
            check_si_unit(*id, "SI_UNIT", unit_arguments, report);
        }
    }
}

/// The argument lists that follow each flattened `SI_UNIT` typed marker.
fn si_unit_argument_lists(args: &[Value]) -> Vec<&[Value]> {
    let mut lists = Vec::new();
    for (index, value) in args.iter().enumerate() {
        let Value::Typed(name, _) = value else {
            continue;
        };
        if name != "SI_UNIT" {
            continue;
        }
        if let Some(Value::List(items)) = args.get(index + 1) {
            lists.push(items.as_slice());
        }
    }
    lists
}

fn check_si_unit(id: u32, context: &str, items: &[Value], report: &mut Report) {
    if items.len() != 2 {
        report.push_error(
            Some(id),
            codes::ARG_COUNT,
            format!("{context} expects a prefix and a unit, found {} arguments", items.len()),
        );
        return;
    }
    match &items[0] {
        Value::Null => {}
        Value::Enum(prefix) if SI_PREFIXES.contains(&prefix.as_str()) => {}
        Value::Enum(prefix) => report.push_error(
            Some(id),
            codes::INVALID_SI_PREFIX,
            format!("SI_UNIT has invalid SI prefix .{prefix}."),
        ),
        other => report.push_error(
            Some(id),
            codes::ARG_KIND,
            format!("SI_UNIT prefix must be `.PREFIX.` or `$`, found {other:?}"),
        ),
    }
    match &items[1] {
        Value::Enum(unit) if SI_UNITS.contains(&unit.as_str()) => {}
        Value::Enum(unit) => report.push_error(
            Some(id),
            codes::INVALID_UNIT,
            format!("SI_UNIT has invalid SI unit .{unit}."),
        ),
        _ => report.push_error(
            Some(id),
            codes::INVALID_UNIT,
            "SI_UNIT must name an SI unit enumeration (e.g. .METRE.)",
        ),
    }
}

/// Whether the file declares an SI length unit somewhere.
///
/// A broad `.METRE.` search is deliberate: the length unit may be written as a complex
/// instance, a standalone entity, or inside a conversion-based unit, and a false
/// negative here would produce a spurious warning. Used for
/// [`codes::MISSING_LENGTH_UNIT`].
pub(crate) fn has_length_unit(file: &StepFile) -> bool {
    file.entities.values().any(|entity| values_contain_enum(&entity.args, "METRE"))
}

fn values_contain_enum(values: &[Value], target: &str) -> bool {
    values.iter().any(|value| match value {
        Value::Enum(name) => name == target,
        Value::List(items) => values_contain_enum(items, target),
        Value::Typed(_, items) => values_contain_enum(items, target),
        _ => false,
    })
}

/// Collect every non-boolean enumeration in `entity` for canonical-case checks.
pub(crate) fn non_canonical_enums(entity: &Entity) -> Vec<&str> {
    fn walk<'a>(values: &'a [Value], out: &mut Vec<&'a str>) {
        for value in values {
            match value {
                Value::Enum(name)
                    if !name.is_empty()
                        && !name
                            .chars()
                            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') =>
                {
                    out.push(name.as_str());
                }
                Value::List(items) => walk(items, out),
                Value::Typed(_, items) => walk(items, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    walk(&entity.args, &mut out);
    out
}
