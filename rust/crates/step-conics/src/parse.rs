//! Reading `ELLIPSE`, `HYPERBOLA`, `PARABOLA` and their placements from a parsed STEP
//! file.

use nodecraft_step::{Entity, StepFile, Value};

use crate::{
    Axis2Placement3d, Conic, ConicError, ConicScene, Ellipse, Hyperbola,
    MasterRepresentation, Parabola, Trim, TrimmedConic,
};

const AXIS2_PLACEMENT_3D: &str = "AXIS2_PLACEMENT_3D";
const CARTESIAN_POINT: &str = "CARTESIAN_POINT";
const DIRECTION: &str = "DIRECTION";
const TRIMMED_CURVE: &str = "TRIMMED_CURVE";

/// Reads every `ELLIPSE`, `HYPERBOLA` and `PARABOLA` from `file`, together with the
/// `AXIS2_PLACEMENT_3D`s they reference and the conic-basis `TRIMMED_CURVE`s.
///
/// Non-conic entities are ignored, including `TRIMMED_CURVE`s on non-conic bases (for
/// example a `LINE`): those belong to other readers.
///
/// # Errors
///
/// [`ConicError::MissingEntity`] for a dangling reference,
/// [`ConicError::WrongEntity`]/[`ConicError::BadArgumentCount`]/
/// [`ConicError::BadValue`] for malformed entities, [`ConicError::InvalidPlacement`]
/// for a placement that cannot form a frame, [`ConicError::NonPositive`] for a
/// non-positive conic axis, and [`ConicError::UnsupportedTrim`] for a trim this crate
/// does not model (cartesian master representation, no parameter value, more than one
/// parameter value).
pub fn read_conics(file: &StepFile) -> Result<ConicScene, ConicError> {
    let mut scene = ConicScene::default();
    let mut trimmed_entities: Vec<(u32, &Entity)> = Vec::new();

    for (&id, entity) in &file.entities {
        match entity.name.as_str() {
            "ELLIPSE" | "HYPERBOLA" | "PARABOLA" => {
                let (placement_ref, conic) = parse_conic(file, id, entity)?;
                scene
                    .placements
                    .entry(placement_ref)
                    .or_insert_with(|| *conic.placement());
                scene.conics.insert(id, conic);
            }
            "TRIMMED_CURVE" => trimmed_entities.push((id, entity)),
            _ => {}
        }
    }

    for (id, entity) in trimmed_entities {
        if let Some(trimmed) = parse_trimmed_curve(file, id, entity, &scene)? {
            scene.trimmed_curves.insert(id, trimmed);
        }
    }
    Ok(scene)
}

/// Reads the `AXIS2_PLACEMENT_3D` with entity id `id`.
///
/// The placement is validated by resolving it to a [`crate::Frame`], so a degenerate
/// axis or reference direction is reported here rather than at use.
///
/// # Errors
///
/// [`ConicError::MissingEntity`] for a missing id,
/// [`ConicError::WrongEntity`]/[`ConicError::BadArgumentCount`]/
/// [`ConicError::BadValue`] for malformed records, and
/// [`ConicError::InvalidPlacement`] for a placement that does not resolve.
pub fn parse_placement(file: &StepFile, id: u32) -> Result<Axis2Placement3d, ConicError> {
    let entity = entity(file, id)?;
    if entity.name != AXIS2_PLACEMENT_3D {
        return Err(ConicError::WrongEntity {
            id,
            expected: AXIS2_PLACEMENT_3D,
            found: entity.name.clone(),
        });
    }
    expect_len(entity, id, 4)?;
    expect_name(entity, id)?;
    let location_id = expect_ref(arg(entity, id, 1)?, id, AXIS2_PLACEMENT_3D, "location")?;
    let location = parse_cartesian_point(file, location_id)?;
    let axis = parse_optional_direction(file, arg(entity, id, 2)?, id, "axis")?;
    let ref_direction = parse_optional_direction(file, arg(entity, id, 3)?, id, "ref_direction")?;
    let placement = Axis2Placement3d {
        location,
        axis,
        ref_direction,
    };
    placement
        .resolve()
        .map_err(|error| ConicError::InvalidPlacement {
            id,
            why: error.to_string(),
        })?;
    Ok(placement)
}

fn parse_conic(
    file: &StepFile,
    id: u32,
    entity: &Entity,
) -> Result<(u32, Conic), ConicError> {
    expect_name(entity, id)?;
    match entity.name.as_str() {
        "ELLIPSE" => {
            expect_len(entity, id, 4)?;
            let placement_id = expect_ref(arg(entity, id, 1)?, id, "ELLIPSE", "position")?;
            let semi_axis_1 = positive_number(entity, id, "semi_axis_1", arg(entity, id, 2)?)?;
            let semi_axis_2 = positive_number(entity, id, "semi_axis_2", arg(entity, id, 3)?)?;
            let placement = parse_placement(file, placement_id)?;
            Ok((
                placement_id,
                Conic::Ellipse(Ellipse {
                    placement,
                    semi_axis_1,
                    semi_axis_2,
                }),
            ))
        }
        "HYPERBOLA" => {
            expect_len(entity, id, 4)?;
            let placement_id = expect_ref(arg(entity, id, 1)?, id, "HYPERBOLA", "position")?;
            let semi_axis = positive_number(entity, id, "semi_axis", arg(entity, id, 2)?)?;
            let semi_imag_axis =
                positive_number(entity, id, "semi_imag_axis", arg(entity, id, 3)?)?;
            let placement = parse_placement(file, placement_id)?;
            Ok((
                placement_id,
                Conic::Hyperbola(Hyperbola {
                    placement,
                    semi_axis,
                    semi_imag_axis,
                }),
            ))
        }
        "PARABOLA" => {
            expect_len(entity, id, 3)?;
            let placement_id = expect_ref(arg(entity, id, 1)?, id, "PARABOLA", "position")?;
            let focal_distance =
                positive_number(entity, id, "focal_dist", arg(entity, id, 2)?)?;
            let placement = parse_placement(file, placement_id)?;
            Ok((
                placement_id,
                Conic::Parabola(Parabola {
                    placement,
                    focal_distance,
                }),
            ))
        }
        other => Err(ConicError::WrongEntity {
            id,
            expected: "ELLIPSE/HYPERBOLA/PARABOLA",
            found: other.to_owned(),
        }),
    }
}

fn parse_trimmed_curve(
    file: &StepFile,
    id: u32,
    entity: &Entity,
    scene: &ConicScene,
) -> Result<Option<TrimmedConic>, ConicError> {
    expect_len(entity, id, 6)?;
    expect_name(entity, id)?;
    let basis = expect_ref(
        arg(entity, id, 1)?,
        id,
        TRIMMED_CURVE,
        "basis_curve",
    )?;
    if !scene.conics.contains_key(&basis) {
        if file.entities.contains_key(&basis) {
            // A non-conic basis (a line, a B-spline, ...) is another reader's curve.
            return Ok(None);
        }
        return Err(ConicError::MissingEntity { id: basis });
    }
    let master = parse_master(arg(entity, id, 5)?, id)?;
    let trim_1 = parse_trim_endpoint(id, arg(entity, id, 2)?, "trim_1")?;
    let trim_2 = parse_trim_endpoint(id, arg(entity, id, 3)?, "trim_2")?;
    let sense_agreement = parse_sense(arg(entity, id, 4)?, id)?;
    Ok(Some(TrimmedConic {
        id,
        basis,
        trim: Trim {
            trim_1,
            trim_2,
            sense_agreement,
        },
        master,
    }))
}

fn parse_master(value: &Value, id: u32) -> Result<MasterRepresentation, ConicError> {
    let Value::Enum(text) = value else {
        return Err(ConicError::BadValue {
            id,
            entity: TRIMMED_CURVE.to_owned(),
            why: format!("master_representation must be an enumeration, found {value:?}"),
        });
    };
    match text.as_str() {
        "PARAMETER" => Ok(MasterRepresentation::Parameter),
        "UNSPECIFIED" => Ok(MasterRepresentation::Unspecified),
        "CARTESIAN" => Err(ConicError::UnsupportedTrim {
            id,
            why: "cartesian master representation; only parameter trims are modelled"
                .to_owned(),
        }),
        other => Err(ConicError::BadValue {
            id,
            entity: TRIMMED_CURVE.to_owned(),
            why: format!("unknown master_representation .{other}."),
        }),
    }
}

fn parse_sense(value: &Value, id: u32) -> Result<bool, ConicError> {
    match value {
        Value::Enum(text) if text == "T" => Ok(true),
        Value::Enum(text) if text == "F" => Ok(false),
        other => Err(ConicError::BadValue {
            id,
            entity: TRIMMED_CURVE.to_owned(),
            why: format!("sense_agreement must be .T. or .F., found {other:?}"),
        }),
    }
}

/// Extracts the single `PARAMETER_VALUE` from a `trim_1`/`trim_2` select set.
///
/// The `nodecraft-step` parser turns `PARAMETER_VALUE(0.)` into a `Typed` token
/// immediately followed by a separate `List([Number(0.)])`; both that shape and a
/// future `Typed("PARAMETER_VALUE", [Number])` shape are accepted. A cartesian point in
/// the set is ignored when a parameter value is present (files commonly give both
/// representations); a set without a parameter value is rejected explicitly.
fn parse_trim_endpoint(id: u32, value: &Value, field: &str) -> Result<f64, ConicError> {
    let Value::List(items) = value else {
        return Err(ConicError::UnsupportedTrim {
            id,
            why: format!("{field} must be a trim select set, found {value:?}"),
        });
    };
    let mut parameters: Vec<f64> = Vec::new();
    let mut index = 0usize;
    while index < items.len() {
        match &items[index] {
            Value::Typed(name, arguments) if name == "PARAMETER_VALUE" => {
                if let Some(Value::Number(number)) = arguments.first() {
                    parameters.push(*number);
                    index += 1;
                } else if let Some(Value::List(values)) = items.get(index + 1) {
                    match values.first() {
                        Some(Value::Number(number)) => {
                            parameters.push(*number);
                            index += 2;
                        }
                        _ => {
                            return Err(ConicError::UnsupportedTrim {
                                id,
                                why: format!("{field} PARAMETER_VALUE is not a single number"),
                            });
                        }
                    }
                } else {
                    return Err(ConicError::UnsupportedTrim {
                        id,
                        why: format!("{field} PARAMETER_VALUE carries no value"),
                    });
                }
            }
            Value::Number(number) => {
                parameters.push(*number);
                index += 1;
            }
            // A cartesian trim point; parameter-master trims use the parameter value.
            Value::Ref(_) => index += 1,
            other => {
                return Err(ConicError::UnsupportedTrim {
                    id,
                    why: format!("{field} contains unsupported trim item {other:?}"),
                });
            }
        }
    }
    if parameters.len() > 1 {
        return Err(ConicError::UnsupportedTrim {
            id,
            why: format!(
                "{field} has {} parameter values; one per trim end is supported",
                parameters.len()
            ),
        });
    }
    let Some(value) = parameters.first().copied() else {
        return Err(ConicError::UnsupportedTrim {
            id,
            why: format!("{field} has no PARAMETER_VALUE"),
        });
    };
    if !value.is_finite() {
        return Err(ConicError::NonFiniteTrim { id, value });
    }
    Ok(value)
}

fn parse_cartesian_point(file: &StepFile, id: u32) -> Result<[f64; 3], ConicError> {
    let entity = entity(file, id)?;
    if entity.name != CARTESIAN_POINT {
        return Err(ConicError::WrongEntity {
            id,
            expected: CARTESIAN_POINT,
            found: entity.name.clone(),
        });
    }
    expect_len(entity, id, 2)?;
    expect_name(entity, id)?;
    let Value::List(coordinates) = arg(entity, id, 1)? else {
        return Err(ConicError::BadValue {
            id,
            entity: CARTESIAN_POINT.to_owned(),
            why: "coordinates must be a list".to_owned(),
        });
    };
    if coordinates.len() != 3 {
        return Err(ConicError::BadValue {
            id,
            entity: CARTESIAN_POINT.to_owned(),
            why: format!(
                "a placement location needs 3 coordinates, found {}",
                coordinates.len()
            ),
        });
    }
    let mut point = [0.0_f64; 3];
    for (slot, value) in point.iter_mut().zip(coordinates) {
        *slot = finite_number(entity, id, "coordinate", value)?;
    }
    Ok(point)
}

fn parse_direction(file: &StepFile, id: u32) -> Result<[f64; 3], ConicError> {
    let entity = entity(file, id)?;
    if entity.name != DIRECTION {
        return Err(ConicError::WrongEntity {
            id,
            expected: DIRECTION,
            found: entity.name.clone(),
        });
    }
    expect_len(entity, id, 2)?;
    expect_name(entity, id)?;
    let Value::List(ratios) = arg(entity, id, 1)? else {
        return Err(ConicError::BadValue {
            id,
            entity: DIRECTION.to_owned(),
            why: "direction ratios must be a list".to_owned(),
        });
    };
    if ratios.len() != 3 {
        return Err(ConicError::BadValue {
            id,
            entity: DIRECTION.to_owned(),
            why: format!("a direction needs 3 ratios, found {}", ratios.len()),
        });
    }
    let mut direction = [0.0_f64; 3];
    for (slot, value) in direction.iter_mut().zip(ratios) {
        *slot = finite_number(entity, id, "ratio", value)?;
    }
    Ok(direction)
}

fn parse_optional_direction(
    file: &StepFile,
    value: &Value,
    placement_id: u32,
    field: &str,
) -> Result<Option<[f64; 3]>, ConicError> {
    match value {
        Value::Null => Ok(None),
        other => {
            let direction_id = expect_ref(other, placement_id, AXIS2_PLACEMENT_3D, field)?;
            Ok(Some(parse_direction(file, direction_id)?))
        }
    }
}

fn positive_number(
    entity: &Entity,
    id: u32,
    field: &str,
    value: &Value,
) -> Result<f64, ConicError> {
    let number = expect_number(value, id, entity, field)?;
    if !number.is_finite() {
        return Err(ConicError::NonFinite {
            id,
            entity: entity.name.clone(),
            field: field.to_owned(),
            value: number,
        });
    }
    if number <= 0.0 {
        return Err(ConicError::NonPositive {
            id,
            entity: entity.name.clone(),
            field: field.to_owned(),
            value: number,
        });
    }
    Ok(number)
}

fn finite_number(
    entity: &Entity,
    id: u32,
    field: &str,
    value: &Value,
) -> Result<f64, ConicError> {
    let number = expect_number(value, id, entity, field)?;
    if !number.is_finite() {
        return Err(ConicError::NonFinite {
            id,
            entity: entity.name.clone(),
            field: field.to_owned(),
            value: number,
        });
    }
    Ok(number)
}

fn expect_number(
    value: &Value,
    id: u32,
    entity: &Entity,
    field: &str,
) -> Result<f64, ConicError> {
    match value {
        Value::Number(number) => Ok(*number),
        other => Err(ConicError::BadValue {
            id,
            entity: entity.name.clone(),
            why: format!("{field} must be a number, found {other:?}"),
        }),
    }
}

fn entity(file: &StepFile, id: u32) -> Result<&Entity, ConicError> {
    file.entities
        .get(&id)
        .ok_or(ConicError::MissingEntity { id })
}

fn expect_len(entity: &Entity, id: u32, expected: usize) -> Result<(), ConicError> {
    if entity.args.len() == expected {
        Ok(())
    } else {
        Err(ConicError::BadArgumentCount {
            id,
            entity: entity.name.clone(),
            expected,
            found: entity.args.len(),
        })
    }
}

fn arg<'a>(entity: &'a Entity, id: u32, index: usize) -> Result<&'a Value, ConicError> {
    entity
        .args
        .get(index)
        .ok_or(ConicError::BadArgumentCount {
            id,
            entity: entity.name.clone(),
            expected: index + 1,
            found: entity.args.len(),
        })
}

fn expect_name(entity: &Entity, id: u32) -> Result<(), ConicError> {
    match arg(entity, id, 0)? {
        Value::Str(_) => Ok(()),
        other => Err(ConicError::BadValue {
            id,
            entity: entity.name.clone(),
            why: format!("first argument must be the entity name string, found {other:?}"),
        }),
    }
}

fn expect_ref(
    value: &Value,
    id: u32,
    entity: &str,
    field: &str,
) -> Result<u32, ConicError> {
    match value {
        Value::Ref(reference) => Ok(*reference),
        other => Err(ConicError::BadValue {
            id,
            entity: entity.to_owned(),
            why: format!("{field} must be an entity reference, found {other:?}"),
        }),
    }
}
