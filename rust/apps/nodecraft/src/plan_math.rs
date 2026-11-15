//! Pure plan placement maths, independent of the GPU feature.

use glam::{DMat4, DVec3, EulerRot, Mat4};
use nodecraft_display::{DisplayTransform, PlanInstance};

fn place_matrix(offset: [f64; 3], rotation: [f64; 3]) -> DMat4 {
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    DMat4::from_translation(DVec3::from(offset))
        * DMat4::from_euler(EulerRot::ZYX, rz, ry, rx)
}

fn transform_matrix(transform: &DisplayTransform) -> DMat4 {
    match transform {
        DisplayTransform::Place { offset, rotation } => place_matrix(*offset, *rotation),
        DisplayTransform::Mirror { plane, offset } => {
            let mut matrix = DMat4::IDENTITY;
            match *plane {
                "xz" => {
                    matrix.y_axis.y = -1.0;
                    matrix.w_axis.y = 2.0 * offset;
                }
                "yz" => {
                    matrix.x_axis.x = -1.0;
                    matrix.w_axis.x = 2.0 * offset;
                }
                _ => {
                    matrix.z_axis.z = -1.0;
                    matrix.w_axis.z = 2.0 * offset;
                }
            }
            matrix
        }
        DisplayTransform::Scale { factor } => DMat4::from_scale(DVec3::splat(*factor)),
    }
}

pub(crate) fn identity_f32() -> [[f32; 4]; 4] {
    Mat4::IDENTITY.to_cols_array_2d()
}

pub(crate) fn instance_matrix(transforms: &[DisplayTransform]) -> [[f32; 4]; 4] {
    let mut matrix = DMat4::IDENTITY;
    for transform in transforms {
        matrix = transform_matrix(transform) * matrix;
    }
    matrix.as_mat4().to_cols_array_2d()
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
        for transform in &instance.transforms {
            transform_hash = match transform {
                DisplayTransform::Place { offset, rotation } => {
                    let mut h = transform_hash;
                    for value in offset.iter().chain(rotation.iter()) {
                        h = hash_f64(h, *value);
                    }
                    h.rotate_left(1) ^ 1
                }
                DisplayTransform::Mirror { plane, offset } => {
                    let mut h = hash_f64(transform_hash, *offset);
                    h ^= plane.len() as u64 ^ 2;
                    h
                }
                DisplayTransform::Scale { factor } => {
                    hash_f64(transform_hash, *factor).rotate_left(1) ^ 3
                }
            };
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

    fn place(offset: [f64; 3]) -> DisplayTransform {
        DisplayTransform::Place { offset, rotation: [0.0; 3] }
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
        let m = instance_matrix(&[DisplayTransform::Place {
            offset: [0.0; 3],
            rotation: [90.0, 90.0, 0.0],
        }]);
        let p = glam::Mat4::from_cols_array_2d(&m).transform_point3(glam::Vec3::Y);
        assert!((p - glam::Vec3::X).length() < 1.0e-5, "{p:?}");
    }

    #[test]
    fn place_rotation_z_sign() {
        let m = instance_matrix(&[DisplayTransform::Place {
            offset: [0.0; 3],
            rotation: [0.0, 0.0, 90.0],
        }]);
        let p = glam::Mat4::from_cols_array_2d(&m).transform_point3(glam::Vec3::X);
        assert!((p - glam::Vec3::Y).length() < 1.0e-5, "{p:?}");
    }

    /// A mirror reverses orientation: every consumer that decides front faces
    /// from the instance matrix must see a negative determinant here, while a
    /// placement (and a positive scale) keeps it positive.
    #[test]
    fn mirror_instance_has_negative_determinant() {
        let mirror = glam::Mat4::from_cols_array_2d(&instance_matrix(&[
            DisplayTransform::Mirror { plane: "yz", offset: 10.0 },
        ]));
        assert!(
            (mirror.determinant() + 1.0).abs() < 1.0e-6,
            "a pure yz mirror is a reflection: {}",
            mirror.determinant(),
        );

        let placed = glam::Mat4::from_cols_array_2d(&instance_matrix(&[place([
            3.0, -2.0, 1.0,
        ])]));
        assert!(placed.determinant() > 0.0, "{}", placed.determinant());

        let scaled = glam::Mat4::from_cols_array_2d(&instance_matrix(&[
            DisplayTransform::Scale { factor: 2.5 },
        ]));
        assert!(scaled.determinant() > 0.0, "{}", scaled.determinant());
    }

    #[test]
    fn transforms_apply_first_to_last() {
        let matrix = instance_matrix(&[
            place([1.0, 0.0, 0.0]),
            DisplayTransform::Mirror { plane: "yz", offset: 0.0 },
        ]);
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
