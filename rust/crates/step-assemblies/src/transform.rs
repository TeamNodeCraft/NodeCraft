//! Rigid/affine 4x4 transforms for assembly placements.
//!
//! Matrices are row-major with column-vector points: `p' = M * [x, y, z, 1]^T`, the
//! convention STEP's `AXIS2_PLACEMENT_3D` and `ITEM_DEFINED_TRANSFORMATION` use. The
//! bottom row is always `[0, 0, 0, 1]`.

use crate::AssemblyError;

/// A 4x4 assembly transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    rows: [[f64; 4]; 4],
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Builds a transform from a placement: `origin` is the local frame's origin and
    /// `axis`/`reference` are its local Z and X directions (STEP `AXIS2_PLACEMENT_3D`).
    ///
    /// `axis` is normalized; `reference` is orthogonalized against it (replaced by a
    /// perpendicular direction if it is zero or parallel), and local Y is `axis × X`.
    pub fn from_placement(
        origin: [f64; 3],
        axis: [f64; 3],
        reference: [f64; 3],
    ) -> Result<Self, AssemblyError> {
        let z = normalize(axis).ok_or_else(|| {
            AssemblyError::Malformed("placement axis direction is zero or non-finite".into())
        })?;
        let mut x = orthogonalize(reference, z);
        if normalize(x).is_none() {
            // Reference missing or parallel to the axis: pick any perpendicular.
            let fallback = if z[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            x = orthogonalize(fallback, z);
        }
        let x = normalize(x).ok_or_else(|| {
            AssemblyError::Malformed("placement reference direction is degenerate".into())
        })?;
        let y = cross(z, x);
        Ok(Self::from_columns(x, y, z, origin))
    }

    /// Builds a transform from its local X, Y, Z axes and origin (columns 0..2, 3).
    #[must_use]
    pub fn from_columns(x: [f64; 3], y: [f64; 3], z: [f64; 3], origin: [f64; 3]) -> Self {
        Self {
            rows: [
                [x[0], y[0], z[0], origin[0]],
                [x[1], y[1], z[1], origin[1]],
                [x[2], y[2], z[2], origin[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Builds a transform from a full 4x4 row-major matrix.
    pub fn from_rows(rows: [[f64; 4]; 4]) -> Result<Self, AssemblyError> {
        for (row_index, row) in rows.iter().enumerate() {
            for (column, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(AssemblyError::Malformed(format!(
                        "transform element [{row_index}][{column}] is not finite"
                    )));
                }
            }
        }
        let expected = [0.0, 0.0, 0.0, 1.0];
        if rows[3] != expected {
            return Err(AssemblyError::Malformed(
                "transform bottom row must be [0, 0, 0, 1]".into(),
            ));
        }
        Ok(Self { rows })
    }

    /// The raw row-major matrix.
    #[must_use]
    pub fn rows(&self) -> [[f64; 4]; 4] {
        self.rows
    }

    /// Matrix product `self * other` (apply `other` first, then `self`).
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let mut rows = [[0.0f64; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                rows[row][column] = (0..4)
                    .map(|k| self.rows[row][k] * other.rows[k][column])
                    .sum();
            }
        }
        Self { rows }
    }

    /// Inverse transform, or `None` when the linear part is singular.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let a = [
            [self.rows[0][0], self.rows[0][1], self.rows[0][2]],
            [self.rows[1][0], self.rows[1][1], self.rows[1][2]],
            [self.rows[2][0], self.rows[2][1], self.rows[2][2]],
        ];
        let determinant = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if !determinant.is_finite() || determinant.abs() < 1.0e-12 {
            return None;
        }
        let inverse = [
            [
                (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / determinant,
                (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / determinant,
                (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / determinant,
            ],
            [
                (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / determinant,
                (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / determinant,
                (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / determinant,
            ],
            [
                (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / determinant,
                (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / determinant,
                (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / determinant,
            ],
        ];
        let translation = [self.rows[0][3], self.rows[1][3], self.rows[2][3]];
        let inverse_translation = [
            -(inverse[0][0] * translation[0]
                + inverse[0][1] * translation[1]
                + inverse[0][2] * translation[2]),
            -(inverse[1][0] * translation[0]
                + inverse[1][1] * translation[1]
                + inverse[1][2] * translation[2]),
            -(inverse[2][0] * translation[0]
                + inverse[2][1] * translation[1]
                + inverse[2][2] * translation[2]),
        ];
        Some(Self {
            rows: [
                [inverse[0][0], inverse[0][1], inverse[0][2], inverse_translation[0]],
                [inverse[1][0], inverse[1][1], inverse[1][2], inverse_translation[1]],
                [inverse[2][0], inverse[2][1], inverse[2][2], inverse_translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
    }

    /// Applies the transform to a point.
    #[must_use]
    pub fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for row in 0..3 {
            out[row] = self.rows[row][0] * point[0]
                + self.rows[row][1] * point[1]
                + self.rows[row][2] * point[2]
                + self.rows[row][3];
        }
        out
    }

    /// Applies the transform to a direction (ignores translation).
    #[must_use]
    pub fn transform_vector(&self, vector: [f64; 3]) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for row in 0..3 {
            out[row] = self.rows[row][0] * vector[0]
                + self.rows[row][1] * vector[1]
                + self.rows[row][2] * vector[2];
        }
        out
    }

    /// The translation part.
    #[must_use]
    pub fn translation(&self) -> [f64; 3] {
        [self.rows[0][3], self.rows[1][3], self.rows[2][3]]
    }

    /// Element-wise closeness.
    #[must_use]
    pub fn approx_eq(&self, other: &Self, tolerance: f64) -> bool {
        self.rows.iter().zip(other.rows.iter()).all(|(left, right)| {
            left.iter()
                .zip(right.iter())
                .all(|(a, b)| (a - b).abs() <= tolerance)
        })
    }

    /// True when every element is within `tolerance` of the identity.
    #[must_use]
    pub fn is_identity(&self, tolerance: f64) -> bool {
        self.approx_eq(&Self::IDENTITY, tolerance)
    }
}

impl std::ops::Mul for Transform {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.multiply(&other)
    }
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length =
        (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !length.is_finite() || length == 0.0 {
        return None;
    }
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_translation_and_axes() {
        let transform =
            Transform::from_placement([10.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
                .unwrap();
        assert!(transform.approx_eq(&Transform::from_columns(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0]
        ), 1e-12));
        assert_eq!(transform.translation(), [10.0, 0.0, 0.0]);
        assert_eq!(transform.transform_point([1.0, 2.0, 3.0]), [11.0, 2.0, 3.0]);
    }

    #[test]
    fn rotation_about_z_maps_x_to_y() {
        // Reference along +Y with axis +Z gives local X = +Y, local Y = -X.
        let transform =
            Transform::from_placement([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
                .unwrap();
        let mapped = transform.transform_point([1.0, 0.0, 0.0]);
        assert!((mapped[0] - 0.0).abs() < 1e-12);
        assert!((mapped[1] - 1.0).abs() < 1e-12);
        assert!((mapped[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn multiply_and_inverse_round_trip() {
        let transform =
            Transform::from_placement([3.0, -4.0, 5.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
                .unwrap();
        let inverse = transform.inverse().unwrap();
        assert!((transform.multiply(&inverse)).is_identity(1e-12));
        let point = [1.0, 2.0, 3.0];
        assert!(inverse
            .transform_point(transform.transform_point(point))
            .iter()
            .zip(point.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12));
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let singular = Transform::from_rows([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn rejects_non_finite_elements() {
        assert!(Transform::from_rows([
            [f64::NAN, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .is_err());
    }
}
