//! Pure plan placement maths, independent of the GPU feature.

use glam::Mat4;
use nodecraft_display::{Placement, PlanInstance};

pub(crate) fn identity_f32() -> [[f32; 4]; 4] {
    Mat4::IDENTITY.to_cols_array_2d()
}

/// The GPU instance matrix for one body: the body's one composed placement
/// value, converted once. This is the only plan-matrix composition path.
pub(crate) fn instance_matrix(placement: &Placement) -> [[f32; 4]; 4] {
    placement.matrix().to_column_major_f32()
}

pub(crate) fn multiply_f32(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    (Mat4::from_cols_array_2d(&a) * Mat4::from_cols_array_2d(&b)).to_cols_array_2d()
}

fn hash_f64(hash: u64, value: f64) -> u64 {
    hash.rotate_left(5) ^ value.to_bits()
}

pub(crate) fn plan_identity_hash(plans: &[PlanInstance]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for instance in plans {
        instance.target.hash(&mut hasher);
        (std::sync::Arc::as_ptr(&instance.plan) as usize).hash(&mut hasher);
        let mut transform_hash = 0u64;
        for row in instance.placement.matrix().rows() {
            for value in row {
                transform_hash = hash_f64(transform_hash, value);
            }
        }
        transform_hash.hash(&mut hasher);
        for colour in &instance.face_colors {
            for value in colour {
                value.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}


#[cfg(test)]
mod tests {
    use super::*;

    use nodecraft_display::{DisplayTransform, Placement};

    fn place(offset: [f64; 3]) -> DisplayTransform {
        DisplayTransform::Place { offset, rotation: [0.0; 3] }
    }

    fn placement(ops: impl IntoIterator<Item = DisplayTransform>) -> Placement {
        Placement::from_ops(ops)
    }

    #[test]
    fn multiply_f32_matches_manual_product() {
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let b = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ];
        let actual = multiply_f32(a, b);
        let mut expected = [[0.0f32; 4]; 4];
        for column in 0..4 {
            for row in 0..4 {
                expected[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
            }
        }
        for column in 0..4 {
            for row in 0..4 {
                assert!((actual[column][row] - expected[column][row]).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn place_rotation_applies_x_then_y_then_z() {
        let m = instance_matrix(&placement([DisplayTransform::Place {
            offset: [0.0; 3],
            rotation: [90.0, 90.0, 0.0],
        }]));
        let p = glam::Mat4::from_cols_array_2d(&m).transform_point3(glam::Vec3::Y);
        assert!((p - glam::Vec3::X).length() < 1.0e-5, "{p:?}");
    }

    #[test]
    fn place_rotation_z_sign() {
        let m = instance_matrix(&placement([DisplayTransform::Place {
            offset: [0.0; 3],
            rotation: [0.0, 0.0, 90.0],
        }]));
        let p = glam::Mat4::from_cols_array_2d(&m).transform_point3(glam::Vec3::X);
        assert!((p - glam::Vec3::Y).length() < 1.0e-5, "{p:?}");
    }

    /// A mirror reverses orientation: every consumer that decides front faces
    /// from the instance matrix must see a negative determinant here, while a
    /// placement (and a positive scale) keeps it positive.
    #[test]
    fn mirror_instance_has_negative_determinant() {
        let mirror = glam::Mat4::from_cols_array_2d(&instance_matrix(&placement([
            DisplayTransform::Mirror { plane: "yz", offset: 10.0 },
        ])));
        assert!(
            (mirror.determinant() + 1.0).abs() < 1.0e-6,
            "a pure yz mirror is a reflection: {}",
            mirror.determinant(),
        );

        let placed = glam::Mat4::from_cols_array_2d(&instance_matrix(&placement([place([
            3.0, -2.0, 1.0,
        ])])));
        assert!(placed.determinant() > 0.0, "{}", placed.determinant());

        let scaled = glam::Mat4::from_cols_array_2d(&instance_matrix(&placement([
            DisplayTransform::Scale { factor: 2.5 },
        ])));
        assert!(scaled.determinant() > 0.0, "{}", scaled.determinant());
    }

    #[test]
    fn transforms_apply_first_to_last() {
        let matrix = instance_matrix(&placement([
            place([1.0, 0.0, 0.0]),
            DisplayTransform::Mirror { plane: "yz", offset: 0.0 },
        ]));
        let x = matrix[0][0] * 0.0
            + matrix[1][0] * 0.0
            + matrix[2][0] * 0.0
            + matrix[3][0];
        let y = matrix[0][1] * 0.0
            + matrix[1][1] * 0.0
            + matrix[2][1] * 0.0
            + matrix[3][1];
        assert!((x - -1.0).abs() < 1.0e-6, "{x}");
        assert!(y.abs() < 1.0e-6, "{y}");
    }
}
