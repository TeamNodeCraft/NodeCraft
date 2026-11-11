//! Exact geometry passed from evaluation to display.
//!
//! These types are deliberately free of graph, display-plan and UI concerns.
//! A node evaluates to an exact [`Part`]-like value; the display service turns
//! that value into plans.

use std::sync::Arc;

/// Stable identity of one body in an exact snapshot.
///
/// `source` is the producing node's content key and `body` is the body's index
/// in that node's Part. Placement is separate instance data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyKey {
    pub source: u64,
    pub body: u32,
}

impl BodyKey {
    #[must_use]
    pub const fn new(source: u64, body: u32) -> Self {
        Self { source, body }
    }
}

/// Exact display input for one body: immutable B-rep plus placement.
#[derive(Clone, Debug)]
pub struct ExactBody {
    pub key: BodyKey,
    pub brep: Arc<solidcore::topology::Brep>,
    pub transforms: Vec<DisplayTransform>,
}

/// Exact display input for one Part. The exact engine returns these handles; it
/// does not tessellate them.
#[derive(Clone, Debug, Default)]
pub struct ExactPart {
    pub bodies: Vec<ExactBody>,
}

impl ExactBody {
    #[must_use]
    pub fn new(key: BodyKey, brep: Arc<solidcore::topology::Brep>) -> Self {
        Self { key, brep, transforms: Vec::new() }
    }
}

impl ExactPart {
    #[must_use]
    pub fn singleton(key: BodyKey, brep: Arc<solidcore::topology::Brep>) -> Self {
        Self { bodies: vec![ExactBody::new(key, brep)] }
    }

    /// Bake each body's placement/mirror/scale into standalone B-reps for
    /// export and exact consumers. Display planning never mutates these B-reps.
    pub fn realized_breps(&self) -> Result<Vec<solidcore::topology::Brep>, String> {
        let mut kernel = solidcore::kernel::Kernel::default();
        let mut breps = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            let mut brep = body
                .brep
                .clone_standing_for_self()
                .map_err(|error| format!("export B-rep copy: {error}"))?;
            for transform in &body.transforms {
                match transform {
                    DisplayTransform::Place { offset, rotation } => {
                        place_brep(&mut kernel, &mut brep, *offset, *rotation)?;
                    }
                    DisplayTransform::Mirror { plane, offset } => {
                        mirror_brep(&mut kernel, &mut brep, plane, *offset)?;
                    }
                    DisplayTransform::Scale { factor } => {
                        scale_brep(&mut brep, *factor)?;
                    }
                }
            }
            breps.push(brep);
        }
        Ok(breps)
    }
}

/// Placement applied after a body's local display plan/mesh is produced.
#[derive(Clone, Debug, PartialEq)]
pub enum DisplayTransform {
    Place { offset: [f64; 3], rotation: [f64; 3] },
    Mirror { plane: &'static str, offset: f64 },
    Scale { factor: f64 },
}

fn place_brep(
    kernel: &mut solidcore::kernel::Kernel,
    brep: &mut solidcore::topology::Brep,
    offset: [f64; 3],
    rotation: [f64; 3],
) -> Result<(), String> {
    use solidcore::geom::axis2_placement::Axis2Placement;
    use solidcore::math::{Point3, Vec3};
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    let x_axis = Vec3::new(cy * cz, cy * sz, -sy);
    let y_axis = Vec3::new(sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy);
    let placement = Axis2Placement::new(Point3::new(offset[0], offset[1], offset[2]), x_axis, y_axis);
    solidcore::topology::brep::p5::transform(kernel, brep, &placement, None, true)
        .map_err(|error| error.to_string())
}

fn mirror_brep(
    kernel: &mut solidcore::kernel::Kernel,
    brep: &mut solidcore::topology::Brep,
    plane: &str,
    offset: f64,
) -> Result<(), String> {
    use solidcore::{geom::axis2_placement::Axis2Placement, math::{Point3, Vec3}, topology::brep::mirror};
    let (origin, x_axis, y_axis) = match plane {
        "xz" => (Point3::new(0.0, offset, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        "yz" => (Point3::new(offset, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        _ => (Point3::new(0.0, 0.0, offset), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
    };
    let mut standalone = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| error.to_string())?;
    mirror::mirror(kernel, &mut standalone, &Axis2Placement::new(origin, x_axis, y_axis))
        .map_err(|error| error.to_string())?;
    *brep = standalone;
    Ok(())
}

fn scale_brep(brep: &mut solidcore::topology::Brep, factor: f64) -> Result<(), String> {
    use solidcore::geom::axis2_placement::Axis2Placement;
    use solidcore::math::{Point3, Vec3};
    let mut kernel = solidcore::kernel::Kernel::default();
    let placement = Axis2Placement::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    solidcore::topology::brep::p5::transform(
        &mut kernel,
        brep,
        &placement,
        Some(Vec3::new(factor, factor, factor)),
        true,
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_keys_identify_source_and_body() {
        assert_eq!(BodyKey::new(7, 2), BodyKey::new(7, 2));
        assert_ne!(BodyKey::new(7, 2), BodyKey::new(7, 3));
        assert_ne!(BodyKey::new(7, 2), BodyKey::new(8, 2));
    }
}
