//! Parse B-spline curves and surfaces out of a generic STEP entity table.
//!
//! Two shapes occur in the wild:
//!
//! * a plain entity such as `#1=B_SPLINE_CURVE_WITH_KNOTS('',3,(#5,#6,...),...)`, and
//! * a complex instance such as
//!   `#2=(BOUNDED_CURVE() B_SPLINE_CURVE(...) B_SPLINE_CURVE_WITH_KNOTS(...) ...
//!   RATIONAL_B_SPLINE_CURVE(...) ...)`.
//!
//! [`nodecraft_step::StepFile`] keeps the first with the entity type in
//! [`nodecraft_step::Entity::name`] and the second with an empty name whose arguments
//! alternate between [`nodecraft_step::Value::Typed`] member names and
//! [`nodecraft_step::Value::List`] argument lists. This module normalises both.

use std::collections::BTreeMap;

use nodecraft_step::{Entity, StepFile, Value};

use crate::{BsplineCurve, BsplineSurface, KnotSpec, Knots};

/// Parse the B-spline curve stored at entity `id`.
pub fn parse_curve(file: &StepFile, id: u32) -> Result<BsplineCurve, String> {
    let entity = file
        .entities
        .get(&id)
        .ok_or_else(|| format!("#{id}: no such entity"))?;
    curve_from_entity(file, id, entity)
}

/// Parse the B-spline surface stored at entity `id`.
pub fn parse_surface(file: &StepFile, id: u32) -> Result<BsplineSurface, String> {
    let entity = file
        .entities
        .get(&id)
        .ok_or_else(|| format!("#{id}: no such entity"))?;
    surface_from_entity(file, id, entity)
}

/// Parse every B-spline curve in the file, in entity-id order.
pub fn curves(file: &StepFile) -> Result<Vec<BsplineCurve>, String> {
    let mut out = Vec::new();
    for (id, entity) in &file.entities {
        if is_curve_entity(entity) {
            out.push(curve_from_entity(file, *id, entity)?);
        }
    }
    Ok(out)
}

/// Parse every B-spline surface in the file, in entity-id order.
pub fn surfaces(file: &StepFile) -> Result<Vec<BsplineSurface>, String> {
    let mut out = Vec::new();
    for (id, entity) in &file.entities {
        if is_surface_entity(entity) {
            out.push(surface_from_entity(file, *id, entity)?);
        }
    }
    Ok(out)
}

fn is_curve_entity(entity: &Entity) -> bool {
    match entity.name.as_str() {
        "B_SPLINE_CURVE_WITH_KNOTS" | "B_SPLINE_CURVE" => true,
        "" => complex_members(entity)
            .iter()
            .any(|member| member.name == "B_SPLINE_CURVE"),
        _ => false,
    }
}

fn is_surface_entity(entity: &Entity) -> bool {
    match entity.name.as_str() {
        "B_SPLINE_SURFACE_WITH_KNOTS" | "B_SPLINE_SURFACE" => true,
        "" => complex_members(entity)
            .iter()
            .any(|member| member.name == "B_SPLINE_SURFACE"),
        _ => false,
    }
}

/// One `NAME(args)` member of a STEP complex instance.
struct ComplexMember {
    name: String,
    args: Vec<Value>,
}

/// Split the alternating typed-name / argument-list representation of a complex
/// instance into named members.
fn complex_members(entity: &Entity) -> Vec<ComplexMember> {
    let mut members = Vec::new();
    let mut index = 0;
    while index < entity.args.len() {
        if let Value::Typed(name, inline) = &entity.args[index] {
            let mut args = inline.clone();
            if args.is_empty() {
                if let Some(Value::List(list)) = entity.args.get(index + 1) {
                    args = list.clone();
                    index += 1;
                }
            }
            members.push(ComplexMember {
                name: name.clone(),
                args,
            });
        }
        index += 1;
    }
    members
}

fn member<'a>(members: &'a [ComplexMember], name: &str) -> Option<&'a [Value]> {
    members
        .iter()
        .find(|member| member.name == name)
        .map(|member| member.args.as_slice())
}

// ---------------------------------------------------------------------------------
// curves
// ---------------------------------------------------------------------------------

type KnotData = Option<(Vec<usize>, Vec<f64>, KnotSpec)>;

fn curve_from_entity(file: &StepFile, id: u32, entity: &Entity) -> Result<BsplineCurve, String> {
    let context = |message: String| format!("#{id}: {message}");
    let (degree, pole_refs, weights, knot_data, closed, self_intersect): (
        usize,
        Vec<u32>,
        Option<Vec<f64>>,
        KnotData,
        bool,
        bool,
    ) = match entity.name.as_str() {
        // B_SPLINE_CURVE_WITH_KNOTS(name, degree, control_points, curve_form,
        //     closed_curve, self_intersect, knot_multiplicities, knots, knot_spec)
        "B_SPLINE_CURVE_WITH_KNOTS" => {
            let args = entity.args.get(1..).unwrap_or_default();
            let degree = index_arg(args, 0, "degree").map_err(context)?;
            let pole_refs =
                ref_list(arg(args, 1, "control_points").map_err(context)?).map_err(context)?;
            let closed =
                boolean(arg(args, 3, "closed_curve").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(args, 4, "self_intersect").map_err(context)?).map_err(context)?;
            let multiplicities =
                index_list(arg(args, 5, "knot_multiplicities").map_err(context)?)
                    .map_err(context)?;
            let values = number_list(arg(args, 6, "knots").map_err(context)?).map_err(context)?;
            let spec = knot_spec(arg(args, 7, "knot_spec").map_err(context)?).map_err(context)?;
            (
                degree,
                pole_refs,
                None,
                Some((multiplicities, values, spec)),
                closed,
                self_intersect,
            )
        }
        // B_SPLINE_CURVE(name, degree, control_points, curve_form, closed_curve,
        //     self_intersect): no knot data, uniform knots are implied.
        "B_SPLINE_CURVE" => {
            let args = entity.args.get(1..).unwrap_or_default();
            let degree = index_arg(args, 0, "degree").map_err(context)?;
            let pole_refs =
                ref_list(arg(args, 1, "control_points").map_err(context)?).map_err(context)?;
            let closed =
                boolean(arg(args, 3, "closed_curve").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(args, 4, "self_intersect").map_err(context)?).map_err(context)?;
            (
                degree,
                pole_refs,
                None,
                Some((Vec::new(), Vec::new(), KnotSpec::Uniform)),
                closed,
                self_intersect,
            )
        }
        "" => {
            let members = complex_members(entity);
            let base = member(&members, "B_SPLINE_CURVE")
                .ok_or_else(|| context("complex instance has no B_SPLINE_CURVE member".into()))?;
            let degree = index_arg(base, 0, "degree").map_err(context)?;
            let pole_refs =
                ref_list(arg(base, 1, "control_points").map_err(context)?).map_err(context)?;
            let closed =
                boolean(arg(base, 3, "closed_curve").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(base, 4, "self_intersect").map_err(context)?).map_err(context)?;
            let weights = match member(&members, "RATIONAL_B_SPLINE_CURVE") {
                Some(args) => Some(
                    number_list(arg(args, 0, "weights").map_err(context)?).map_err(context)?,
                ),
                None => None,
            };
            let knot_data = match member(&members, "B_SPLINE_CURVE_WITH_KNOTS") {
                Some(args) if !args.is_empty() => {
                    let spec = match args.get(2) {
                        Some(value) if !matches!(value, Value::Null) => {
                            knot_spec(value).map_err(context)?
                        }
                        _ => KnotSpec::Unspecified,
                    };
                    let multiplicities =
                        optional_index_list(args.first(), "knot_multiplicities")
                            .map_err(context)?;
                    let values =
                        optional_number_list(args.get(1), "knots").map_err(context)?;
                    Some((multiplicities, values, spec))
                }
                _ => None,
            };
            (
                degree,
                pole_refs,
                weights,
                knot_data,
                closed,
                self_intersect,
            )
        }
        other => {
            return Err(context(format!(
                "entity is a {other}, not a B-spline curve"
            )))
        }
    };

    let poles = resolve_points(file, &pole_refs).map_err(context)?;
    if poles.len() < degree + 1 {
        return Err(context(format!(
            "degree {degree} needs at least {} control points, got {}",
            degree + 1,
            poles.len()
        )));
    }
    let knots = build_knots(knot_data, degree, poles.len(), KnotSpec::QuasiUniform)
        .map_err(context)?;
    if let Some(weights) = &weights {
        validate_weights(weights, poles.len()).map_err(context)?;
    }

    Ok(BsplineCurve {
        degree,
        poles,
        weights,
        knots,
        closed,
        self_intersect,
    })
}

// ---------------------------------------------------------------------------------
// surfaces
// ---------------------------------------------------------------------------------

fn surface_from_entity(
    file: &StepFile,
    id: u32,
    entity: &Entity,
) -> Result<BsplineSurface, String> {
    let context = |message: String| format!("#{id}: {message}");
    #[allow(clippy::type_complexity)]
    let (
        u_degree,
        v_degree,
        pole_refs,
        weights,
        u_knot_data,
        v_knot_data,
        u_closed,
        v_closed,
        self_intersect,
    ): (
        usize,
        usize,
        Vec<Vec<u32>>,
        Option<Vec<Vec<f64>>>,
        KnotData,
        KnotData,
        bool,
        bool,
        bool,
    ) = match entity.name.as_str() {
        // B_SPLINE_SURFACE_WITH_KNOTS(name, u_degree, v_degree, control_points,
        //     surface_form, u_closed, v_closed, self_intersect, u_multiplicities,
        //     v_multiplicities, u_knots, v_knots, knot_spec)
        "B_SPLINE_SURFACE_WITH_KNOTS" => {
            let args = entity.args.get(1..).unwrap_or_default();
            let u_degree = index_arg(args, 0, "u_degree").map_err(context)?;
            let v_degree = index_arg(args, 1, "v_degree").map_err(context)?;
            let pole_refs =
                ref_grid(arg(args, 2, "control_points").map_err(context)?).map_err(context)?;
            let u_closed =
                boolean(arg(args, 4, "u_closed").map_err(context)?).map_err(context)?;
            let v_closed =
                boolean(arg(args, 5, "v_closed").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(args, 6, "self_intersect").map_err(context)?).map_err(context)?;
            let spec = knot_spec(arg(args, 11, "knot_spec").map_err(context)?).map_err(context)?;
            let u_multiplicities =
                optional_index_list(args.get(7), "u_multiplicities").map_err(context)?;
            let v_multiplicities =
                optional_index_list(args.get(8), "v_multiplicities").map_err(context)?;
            let u_values = optional_number_list(args.get(9), "u_knots").map_err(context)?;
            let v_values = optional_number_list(args.get(10), "v_knots").map_err(context)?;
            (
                u_degree,
                v_degree,
                pole_refs,
                None,
                Some((u_multiplicities, u_values, spec)),
                Some((v_multiplicities, v_values, spec)),
                u_closed,
                v_closed,
                self_intersect,
            )
        }
        // B_SPLINE_SURFACE(name, u_degree, v_degree, control_points, surface_form,
        //     u_closed, v_closed, self_intersect): no knots, uniform implied.
        "B_SPLINE_SURFACE" => {
            let args = entity.args.get(1..).unwrap_or_default();
            let u_degree = index_arg(args, 0, "u_degree").map_err(context)?;
            let v_degree = index_arg(args, 1, "v_degree").map_err(context)?;
            let pole_refs =
                ref_grid(arg(args, 2, "control_points").map_err(context)?).map_err(context)?;
            let u_closed =
                boolean(arg(args, 4, "u_closed").map_err(context)?).map_err(context)?;
            let v_closed =
                boolean(arg(args, 5, "v_closed").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(args, 6, "self_intersect").map_err(context)?).map_err(context)?;
            (
                u_degree,
                v_degree,
                pole_refs,
                None,
                Some((Vec::new(), Vec::new(), KnotSpec::Uniform)),
                Some((Vec::new(), Vec::new(), KnotSpec::Uniform)),
                u_closed,
                v_closed,
                self_intersect,
            )
        }
        "" => {
            let members = complex_members(entity);
            let base = member(&members, "B_SPLINE_SURFACE")
                .ok_or_else(|| context("complex instance has no B_SPLINE_SURFACE member".into()))?;
            let u_degree = index_arg(base, 0, "u_degree").map_err(context)?;
            let v_degree = index_arg(base, 1, "v_degree").map_err(context)?;
            let pole_refs =
                ref_grid(arg(base, 2, "control_points").map_err(context)?).map_err(context)?;
            let u_closed =
                boolean(arg(base, 4, "u_closed").map_err(context)?).map_err(context)?;
            let v_closed =
                boolean(arg(base, 5, "v_closed").map_err(context)?).map_err(context)?;
            let self_intersect =
                boolean(arg(base, 6, "self_intersect").map_err(context)?).map_err(context)?;
            let weights = match member(&members, "RATIONAL_B_SPLINE_SURFACE") {
                Some(args) => Some(
                    number_grid(arg(args, 0, "weights").map_err(context)?).map_err(context)?,
                ),
                None => None,
            };
            let (u_knot_data, v_knot_data) =
                match member(&members, "B_SPLINE_SURFACE_WITH_KNOTS") {
                    Some(args) if !args.is_empty() => {
                        let spec = match args.get(4) {
                            Some(value) if !matches!(value, Value::Null) => {
                                knot_spec(value).map_err(context)?
                            }
                            _ => KnotSpec::Unspecified,
                        };
                        let u_multiplicities =
                            optional_index_list(args.first(), "u_multiplicities")
                                .map_err(context)?;
                        let v_multiplicities =
                            optional_index_list(args.get(1), "v_multiplicities")
                                .map_err(context)?;
                        let u_values =
                            optional_number_list(args.get(2), "u_knots").map_err(context)?;
                        let v_values =
                            optional_number_list(args.get(3), "v_knots").map_err(context)?;
                        (
                            Some((u_multiplicities, u_values, spec)),
                            Some((v_multiplicities, v_values, spec)),
                        )
                    }
                    _ => (None, None),
                };
            (
                u_degree,
                v_degree,
                pole_refs,
                weights,
                u_knot_data,
                v_knot_data,
                u_closed,
                v_closed,
                self_intersect,
            )
        }
        other => {
            return Err(context(format!(
                "entity is a {other}, not a B-spline surface"
            )))
        }
    };

    let poles = resolve_point_grid(file, &pole_refs).map_err(context)?;
    let u_count = poles.len();
    let v_count = poles.first().map_or(0, Vec::len);
    if u_count < u_degree + 1 {
        return Err(context(format!(
            "u degree {u_degree} needs at least {} control rows, got {u_count}",
            u_degree + 1
        )));
    }
    if v_count < v_degree + 1 {
        return Err(context(format!(
            "v degree {v_degree} needs at least {} control points per row, got {v_count}",
            v_degree + 1
        )));
    }
    if poles.iter().any(|row| row.len() != v_count) {
        return Err(context("control-point grid is not rectangular".into()));
    }

    let u_knots = build_knots(u_knot_data, u_degree, u_count, KnotSpec::QuasiUniform)
        .map_err(context)?;
    let v_knots = build_knots(v_knot_data, v_degree, v_count, KnotSpec::QuasiUniform)
        .map_err(context)?;

    if let Some(weights) = &weights {
        if weights.len() != u_count || weights.iter().any(|row| row.len() != v_count) {
            return Err(context(format!(
                "weight grid is {}x{}, expected {u_count}x{v_count}",
                weights.len(),
                weights.first().map_or(0, Vec::len)
            )));
        }
        let flat: Vec<f64> = weights.iter().flatten().copied().collect();
        validate_weights(&flat, u_count * v_count).map_err(context)?;
    }

    Ok(BsplineSurface {
        u_degree,
        v_degree,
        poles,
        weights,
        u_knots,
        v_knots,
        u_closed,
        v_closed,
        self_intersect,
    })
}

fn build_knots(
    data: KnotData,
    degree: usize,
    control_count: usize,
    default_spec: KnotSpec,
) -> Result<Knots, String> {
    let (multiplicities, values, spec) = data.unwrap_or((Vec::new(), Vec::new(), default_spec));
    let knots = if multiplicities.is_empty() || values.is_empty() {
        Knots::default_for(degree, control_count, spec)?
    } else {
        Knots::explicit(&multiplicities, &values, spec)?
    };
    knots.validate_for(degree, control_count)?;
    Ok(knots)
}

fn validate_weights(weights: &[f64], expected: usize) -> Result<(), String> {
    if weights.len() != expected {
        return Err(format!(
            "{} weights for {expected} control points",
            weights.len()
        ));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err("weights must be finite".into());
    }
    if let Some(index) = weights.iter().position(|&weight| weight <= 0.0) {
        return Err(format!("weight {index} is not positive ({})", weights[index]));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------
// value helpers
// ---------------------------------------------------------------------------------

fn arg<'a>(args: &'a [Value], index: usize, what: &str) -> Result<&'a Value, String> {
    args.get(index)
        .ok_or_else(|| format!("missing {what} argument (index {index})"))
}

fn number(value: &Value, what: &str) -> Result<f64, String> {
    match value {
        Value::Number(number) if number.is_finite() => Ok(*number),
        Value::Number(number) => Err(format!("{what} is not finite ({number})")),
        other => Err(format!("{what} is not a number: {other:?}")),
    }
}

fn index_arg(args: &[Value], position: usize, what: &str) -> Result<usize, String> {
    let value = number(arg(args, position, what)?, what)?;
    if value < 0.0 || value.fract() != 0.0 {
        return Err(format!("{what} must be a non-negative integer, got {value}"));
    }
    Ok(value as usize)
}

/// STEP `LOGICAL` flags on B-splines (`.T.`, `.F.`, `.U.`); unknown reads as `false`.
fn boolean(value: &Value) -> Result<bool, String> {
    match value {
        Value::Enum(flag) if flag == "T" => Ok(true),
        Value::Enum(flag) if flag == "F" || flag == "U" => Ok(false),
        other => Err(format!("expected a STEP boolean, got {other:?}")),
    }
}

fn knot_spec(value: &Value) -> Result<KnotSpec, String> {
    KnotSpec::from_value(value).ok_or_else(|| format!("unknown knot_spec {value:?}"))
}

fn ref_list(value: &Value) -> Result<Vec<u32>, String> {
    let Value::List(items) = value else {
        return Err(format!("expected a list of references, got {value:?}"));
    };
    items
        .iter()
        .map(|item| match item {
            Value::Ref(id) => Ok(*id),
            other => Err(format!("expected a reference, got {other:?}")),
        })
        .collect()
}

fn number_list(value: &Value) -> Result<Vec<f64>, String> {
    let Value::List(items) = value else {
        return Err(format!("expected a list of numbers, got {value:?}"));
    };
    items
        .iter()
        .map(|item| number(item, "list entry"))
        .collect()
}

fn index_list(value: &Value) -> Result<Vec<usize>, String> {
    let Value::List(items) = value else {
        return Err(format!("expected a list of integers, got {value:?}"));
    };
    items
        .iter()
        .map(|item| {
            let number = number(item, "list entry")?;
            if number < 0.0 || number.fract() != 0.0 {
                return Err(format!("expected a non-negative integer, got {number}"));
            }
            Ok(number as usize)
        })
        .collect()
}

fn ref_grid(value: &Value) -> Result<Vec<Vec<u32>>, String> {
    let Value::List(rows) = value else {
        return Err(format!("expected a control-point grid, got {value:?}"));
    };
    rows.iter().map(ref_list).collect()
}

fn number_grid(value: &Value) -> Result<Vec<Vec<f64>>, String> {
    let Value::List(rows) = value else {
        return Err(format!("expected a weight grid, got {value:?}"));
    };
    rows.iter().map(number_list).collect()
}

fn optional_index_list(value: Option<&Value>, what: &str) -> Result<Vec<usize>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(value) => index_list(value).map_err(|error| format!("{what}: {error}")),
    }
}

fn optional_number_list(value: Option<&Value>, what: &str) -> Result<Vec<f64>, String> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(value) => number_list(value).map_err(|error| format!("{what}: {error}")),
    }
}

// ---------------------------------------------------------------------------------
// point resolution
// ---------------------------------------------------------------------------------

fn point(file: &StepFile, id: u32) -> Result<[f64; 3], String> {
    let entity = file
        .entities
        .get(&id)
        .ok_or_else(|| format!("#{id}: control point entity is missing"))?;
    if entity.name != "CARTESIAN_POINT" {
        return Err(format!(
            "#{id}: expected CARTESIAN_POINT, found {}",
            entity.name
        ));
    }
    // Classic STEP writes CARTESIAN_POINT(name, (x,y,z)); tolerate a missing name.
    let coordinates = match entity.args.first() {
        Some(Value::List(list)) => list,
        Some(_) => match entity.args.get(1) {
            Some(Value::List(list)) => list,
            _ => return Err(format!("#{id}: CARTESIAN_POINT has no coordinate list")),
        },
        None => return Err(format!("#{id}: CARTESIAN_POINT is empty")),
    };
    if !(coordinates.len() == 2 || coordinates.len() == 3) {
        return Err(format!(
            "#{id}: CARTESIAN_POINT has {} coordinates, expected 2 or 3",
            coordinates.len()
        ));
    }
    let mut point = [0.0; 3];
    for (slot, value) in point.iter_mut().zip(coordinates) {
        *slot = number(value, "coordinate")?;
    }
    Ok(point)
}

fn resolve_points(file: &StepFile, refs: &[u32]) -> Result<Vec<[f64; 3]>, String> {
    let mut cache: BTreeMap<u32, [f64; 3]> = BTreeMap::new();
    for &id in refs {
        if !cache.contains_key(&id) {
            let resolved = point(file, id)?;
            cache.insert(id, resolved);
        }
    }
    Ok(refs.iter().map(|id| cache[id]).collect())
}

fn resolve_point_grid(file: &StepFile, refs: &[Vec<u32>]) -> Result<Vec<Vec<[f64; 3]>>, String> {
    let flat: Vec<u32> = refs.iter().flatten().copied().collect();
    let resolved = resolve_points(file, &flat)?;
    let mut source = resolved.into_iter();
    refs.iter()
        .map(|row| {
            let row: Vec<[f64; 3]> = source.by_ref().take(row.len()).collect();
            Ok(row)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_complex_member_names_and_arguments() {
        let text = "DATA;#1=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(),.UNSPECIFIED.,.F.,.F.) \
                    CURVE() GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_CURVE((1.,1.)) \
                    REPRESENTATION_ITEM(''));ENDSEC;";
        let file = StepFile::parse(text).unwrap();
        let entity = &file.entities[&1];
        let members = complex_members(entity);
        let names: Vec<&str> = members.iter().map(|member| member.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "BOUNDED_CURVE",
                "B_SPLINE_CURVE",
                "CURVE",
                "GEOMETRIC_REPRESENTATION_ITEM",
                "RATIONAL_B_SPLINE_CURVE",
                "REPRESENTATION_ITEM"
            ]
        );
        let base = member(&members, "B_SPLINE_CURVE").unwrap();
        assert_eq!(base.len(), 5);
        assert_eq!(base[0], Value::Number(2.0));
    }

    #[test]
    fn index_list_rejects_fractions() {
        let value = Value::List(vec![Value::Number(3.5)]);
        assert!(index_list(&value).is_err());
    }

    #[test]
    fn boolean_accepts_step_flags() {
        assert!(boolean(&Value::Enum("T".into())).unwrap());
        assert!(!boolean(&Value::Enum("F".into())).unwrap());
        assert!(!boolean(&Value::Enum("U".into())).unwrap());
        assert!(boolean(&Value::Number(1.0)).is_err());
    }
}
