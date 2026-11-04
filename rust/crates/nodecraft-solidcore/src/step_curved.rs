//! STEP analytic curved surfaces → solidcore `Surface`.
//!
//! Maps one analytic STEP surface entity into the equivalent unbounded solidcore
//! surface, carrying over its `AXIS2_PLACEMENT_3D` (origin, axis, reference direction):
//!
//! | STEP entity | solidcore surface |
//! |---|---|
//! | `CYLINDRICAL_SURFACE(name, position, radius)` | `SurfaceKind::Cone` from `Cone::create_canonical_cylinder` (`is_cylinder() == true`) |
//! | `CONICAL_SURFACE(name, position, radius, semi_angle)` | `SurfaceKind::Cone` from `Cone::create_canonical_with_semi_angle`, unbounded, semi-angle radians → degrees |
//! | `SPHERICAL_SURFACE(name, position, radius)` | `SurfaceKind::Sphere` from `Sphere::create_canonical` |
//! | `TOROIDAL_SURFACE(name, position, major_radius, minor_radius)` | `SurfaceKind::Torus` from `Torus::create_canonical` |
//!
//! The STEP `AXIS2_PLACEMENT_3D` becomes the surface's local frame: `axis` is local Z
//! (default `+Z`) and `ref_direction` is local X (default `+X`), orthogonalized against
//! Z; local Y is `Z × X`. Missing `axis`/`ref_direction` attributes use the STEP
//! defaults. Values are used in STEP file units (no unit scaling) — the fixtures and
//! `nodecraft-step` writers are millimetres.
//!
//! The function converts a single surface only: it does not read face loops, edges or
//! trims, and it does not assemble a mixed B-rep.
//!
//! Entry point: [`analytic_surface_from_step`].

use solidcore::geom::axis2_placement::Axis2Placement;
use solidcore::math::{Point3, Vec3};
use solidcore::surface::cone::Cone;
use solidcore::surface::sphere::Sphere;
use solidcore::surface::surface::{Surface, SurfaceKind};
use solidcore::surface::torus::{Torus, TorusOptions};
use nodecraft_step::{Entity, StepFile, Value};

/// The STEP analytic surface entities this module understands.
pub const SUPPORTED_SURFACES: [&str; 4] = [
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE",
];

/// Converts the analytic STEP surface entity `surface_id` into a solidcore [`Surface`].
///
/// The returned surface is unbounded (a surface, not a face): the caller is responsible
/// for the face trims and `same_sense`. The STEP placement is preserved exactly.
///
/// # Errors
///
/// Unknown entity ids, non-analytic entities (`PLANE`, curves, ...), missing placements
/// or point/direction references, non-finite numbers, non-positive radii, a zero
/// placement axis, and cone semi-angles outside `(0, 90) degrees` are reported as
/// descriptive `String` errors.
pub fn analytic_surface_from_step(file: &StepFile, surface_id: u32) -> Result<Surface, String> {
    let entity = file
        .entities
        .get(&surface_id)
        .ok_or_else(|| format!("STEP entity #{surface_id} does not exist"))?;
    let (kind, args) = analytic_kind(entity).ok_or_else(|| {
        format!(
            "STEP entity #{surface_id} ({}) is not an analytic surface; supported: {}",
            entity_label(entity),
            SUPPORTED_SURFACES.join(", ")
        )
    })?;
    let placement = surface_placement(file, args, surface_id)?;
    let kernel = crate::kernel::kernel();
    let surface = match kind {
        "CYLINDRICAL_SURFACE" => {
            let radius = number_arg(args, 2, surface_id, "radius")?;
            require_positive(radius, surface_id, "cylinder radius")?;
            let cylinder = Cone::create_canonical_cylinder(&kernel, &placement, radius)
                .map_err(|error| format!("solidcore cylinder surface #{surface_id}: {error}"))?;
            Surface::new(SurfaceKind::Cone(Box::new(cylinder)))
        }
        "CONICAL_SURFACE" => {
            let radius = number_arg(args, 2, surface_id, "radius")?;
            let semi_angle = number_arg(args, 3, surface_id, "semi_angle")?;
            if !radius.is_finite() || radius < 0.0 {
                return Err(format!(
                    "conical surface #{surface_id} has a negative or non-finite radius ({radius})"
                ));
            }
            if !semi_angle.is_finite() || semi_angle <= 0.0 || semi_angle >= std::f64::consts::FRAC_PI_2
            {
                return Err(format!(
                    "conical surface #{surface_id} semi-angle {semi_angle} rad is outside (0, pi/2)"
                ));
            }
            let cone = Cone::create_canonical_with_semi_angle(
                &kernel,
                &placement,
                radius,
                semi_angle.to_degrees(),
                None,
                false,
                false,
            )
            .map_err(|error| format!("solidcore conical surface #{surface_id}: {error}"))?;
            Surface::new(SurfaceKind::Cone(Box::new(cone)))
        }
        "SPHERICAL_SURFACE" => {
            let radius = number_arg(args, 2, surface_id, "radius")?;
            require_positive(radius, surface_id, "sphere radius")?;
            let sphere = Sphere::create_canonical(&kernel, &placement, radius)
                .map_err(|error| format!("solidcore spherical surface #{surface_id}: {error}"))?;
            Surface::new(SurfaceKind::Sphere(Box::new(sphere)))
        }
        "TOROIDAL_SURFACE" => {
            let major = number_arg(args, 2, surface_id, "major_radius")?;
            let minor = number_arg(args, 3, surface_id, "minor_radius")?;
            require_positive(major, surface_id, "torus major radius")?;
            require_positive(minor, surface_id, "torus minor radius")?;
            let torus =
                Torus::create_canonical(&kernel, &placement, major, minor, TorusOptions::default(), None)
                    .map_err(|error| format!("solidcore toroidal surface #{surface_id}: {error}"))?;
            Surface::new(SurfaceKind::Torus(Box::new(torus)))
        }
        other => {
            return Err(format!(
                "STEP analytic surface #{surface_id} has unsupported kind {other}"
            ));
        }
    };
    Ok(surface)
}

// ---------------------------------------------------------------------------
// STEP entity helpers
// ---------------------------------------------------------------------------

/// The entity's analytic kind and its attribute slice (simple or complex instance).
fn analytic_kind<'a>(entity: &'a Entity) -> Option<(&'static str, &'a [Value])> {
    for (name, args) in components(entity) {
        if let Some(kind) = SUPPORTED_SURFACES.iter().find(|kind| **kind == name) {
            return Some((kind, args));
        }
    }
    None
}

/// Entity components: a plain entity is one `(name, args)` part; a complex instance
/// (`#n=(A(..) B(..))`) contributes one per typed part.
fn components(entity: &Entity) -> Vec<(&str, &[Value])> {
    if !entity.name.is_empty() {
        return vec![(entity.name.as_str(), entity.args.as_slice())];
    }
    let mut out = Vec::new();
    let mut index = 0;
    while index < entity.args.len() {
        let Value::Typed(name, inline) = &entity.args[index] else {
            index += 1;
            continue;
        };
        if !inline.is_empty() {
            out.push((name.as_str(), inline.as_slice()));
            index += 1;
        } else if let Some(Value::List(values)) = entity.args.get(index + 1) {
            out.push((name.as_str(), values.as_slice()));
            index += 2;
        } else {
            out.push((name.as_str(), &[]));
            index += 1;
        }
    }
    out
}

fn entity_label(entity: &Entity) -> String {
    if !entity.name.is_empty() {
        return entity.name.clone();
    }
    let names: Vec<&str> = components(entity).into_iter().map(|(name, _)| name).collect();
    if names.is_empty() {
        "complex instance".to_string()
    } else {
        names.join("+")
    }
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

/// Builds the surface's local frame from the entity's `AXIS2_PLACEMENT_3D` argument.
fn surface_placement(
    file: &StepFile,
    args: &[Value],
    surface_id: u32,
) -> Result<Axis2Placement, String> {
    let placement_id = ref_arg(args, 1, surface_id, "position")?;
    let placement_entity = file
        .entities
        .get(&placement_id)
        .ok_or_else(|| format!("surface #{surface_id} placement #{placement_id} does not exist"))?;
    let placement_args = components(placement_entity)
        .into_iter()
        .find(|(name, _)| *name == "AXIS2_PLACEMENT_3D")
        .map(|(_, args)| args)
        .ok_or_else(|| {
            format!(
                "surface #{surface_id} placement #{placement_id} is not an AXIS2_PLACEMENT_3D \
                 ({})",
                entity_label(placement_entity)
            )
        })?;
    let origin = match optional_ref(placement_args, 1) {
        Some(point) => point_coordinates(file, point, placement_id)?,
        None => [0.0, 0.0, 0.0],
    };
    let axis = match optional_ref(placement_args, 2) {
        Some(direction) => direction_ratios(file, direction, placement_id)?,
        None => [0.0, 0.0, 1.0],
    };
    let reference = match optional_ref(placement_args, 3) {
        Some(direction) => direction_ratios(file, direction, placement_id)?,
        None => [1.0, 0.0, 0.0],
    };
    let z = unit(axis, "placement axis")?;
    let mut x = orthogonalize(reference, z);
    if normalize(x).is_none() {
        // Reference missing/parallel to the axis: choose any perpendicular direction.
        let fallback = if z[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        x = orthogonalize(fallback, z);
    }
    let x = unit(x, "placement reference direction")?;
    let y = cross(z, x);
    Ok(Axis2Placement::new(
        Point3::new(origin[0], origin[1], origin[2]),
        Vec3::new(x[0], x[1], x[2]),
        Vec3::new(y[0], y[1], y[2]),
    ))
}

fn point_coordinates(file: &StepFile, id: u32, placement_id: u32) -> Result<[f64; 3], String> {
    let entity = file.entities.get(&id).ok_or_else(|| {
        format!("placement #{placement_id} references missing point #{id}")
    })?;
    if !type_names(entity).contains(&"CARTESIAN_POINT") {
        return Err(format!(
            "placement #{placement_id} point #{id} is not a CARTESIAN_POINT ({})",
            entity_label(entity)
        ));
    }
    let Some(Value::List(values)) = entity.args.get(1) else {
        return Err(format!("CARTESIAN_POINT #{id} has no coordinate list"));
    };
    Ok(number_list(values, id)?)
}

fn direction_ratios(file: &StepFile, id: u32, placement_id: u32) -> Result<[f64; 3], String> {
    let entity = file.entities.get(&id).ok_or_else(|| {
        format!("placement #{placement_id} references missing direction #{id}")
    })?;
    if !type_names(entity).contains(&"DIRECTION") {
        return Err(format!(
            "placement #{placement_id} direction #{id} is not a DIRECTION ({})",
            entity_label(entity)
        ));
    }
    let Some(Value::List(values)) = entity.args.get(1) else {
        return Err(format!("DIRECTION #{id} has no ratio list"));
    };
    Ok(number_list(values, id)?)
}

/// `(x, y, z)` from a number list, padding missing components with `0`.
fn number_list(values: &[Value], entity_id: u32) -> Result<[f64; 3], String> {
    let mut out = [0.0f64; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        let Value::Number(number) = value else {
            return Err(format!("entity #{entity_id} has a non-numeric coordinate: {value:?}"));
        };
        if !number.is_finite() {
            return Err(format!("entity #{entity_id} has a non-finite coordinate: {number}"));
        }
        *slot = *number;
    }
    Ok(out)
}

fn type_names(entity: &Entity) -> Vec<&str> {
    components(entity).into_iter().map(|(name, _)| name).collect()
}

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------

fn optional_ref(args: &[Value], index: usize) -> Option<u32> {
    match args.get(index) {
        Some(Value::Ref(id)) => Some(*id),
        _ => None,
    }
}

fn ref_arg(
    args: &[Value],
    index: usize,
    surface_id: u32,
    what: &str,
) -> Result<u32, String> {
    optional_ref(args, index)
        .ok_or_else(|| format!("analytic surface #{surface_id} has no {what} reference"))
}

fn number_arg(
    args: &[Value],
    index: usize,
    surface_id: u32,
    what: &str,
) -> Result<f64, String> {
    match args.get(index) {
        Some(Value::Number(number)) if number.is_finite() => Ok(*number),
        Some(other) => Err(format!(
            "analytic surface #{surface_id} {what} is not a finite number: {other:?}"
        )),
        None => Err(format!("analytic surface #{surface_id} has no {what}")),
    }
}

fn require_positive(value: f64, surface_id: u32, what: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "analytic surface #{surface_id} {what} must be positive and finite, got {value}"
        ))
    }
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !length.is_finite() || length == 0.0 {
        return None;
    }
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
}

fn unit(vector: [f64; 3], what: &str) -> Result<[f64; 3], String> {
    normalize(vector).ok_or_else(|| format!("{what} is zero or non-finite: {vector:?}"))
}

fn orthogonalize(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let projection = vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2];
    [
        vector[0] - axis[0] * projection,
        vector[1] - axis[1] * projection,
        vector[2] - axis[2] * projection,
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
