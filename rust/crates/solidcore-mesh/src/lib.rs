//! Display tessellation plans for solidcore B-reps.
//!
//! A plan stores each face's triangulation in its own `(u, v)` parameters and the
//! surfaces to evaluate it on. Shared edge vertices are fixed in 3D, so neighbouring
//! faces meet exactly. [`TessPlan::evaluate`] turns the plan into a CPU mesh.
//!
//! The plan format and evaluator are ported from SolidCraft's `solidcraft-geom`
//! (MIT/Apache-2.0); see `NOTICE`.
#![allow(missing_docs)]
#![allow(clippy::pedantic)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::used_underscore_binding)]
#![allow(dead_code)]

use std::ops::{Add, Mul, Neg, Sub};

/// Highest B-spline degree the evaluator supports.
pub const MAX_SPLINE_DEGREE: usize = 7;

/// A 3D point or vector in millimetres, Z up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const X: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const Y: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn len2(self) -> f64 {
        self.dot(self)
    }

    pub fn len(self) -> f64 {
        self.len2().sqrt()
    }

    pub fn dist(self, other: Self) -> f64 {
        (self - other).len()
    }

    pub fn normalized(self) -> Option<Self> {
        let length = self.len();
        (length > 0.0 && length.is_finite()).then(|| self * (1.0 / length))
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor, self.z * factor)
    }
}

/// An indexed triangle mesh with per-vertex normals and B-rep face/edge metadata.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// B-rep face index per triangle.
    pub tri_face: Vec<u32>,
    /// B-rep edges as polylines.
    pub edges: Vec<Vec<Vec3>>,
    /// Per edge: the B-rep faces it bounds.
    pub edge_faces: Vec<Vec<u32>>,
}

mod tess;
pub use tess::{EvalCurve, EvalSurface, FacePlan, TessPlan};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_plan_evaluates_to_a_triangle() {
        let plan = TessPlan {
            surfaces: vec![EvalSurface::Plane {
                origin: Vec3::ZERO,
                du: Vec3::X,
                dv: Vec3::Y,
            }],
            faces: vec![FacePlan {
                face: 0,
                surface: Some(0),
                uv: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                fixed: vec![],
                normals: vec![],
                flip: false,
                triangles: vec![[0, 1, 2]],
            }],
            edges: vec![],
            edge_faces: vec![],
        };
        let mesh = plan.evaluate();
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
        assert_eq!(mesh.tri_face, vec![0]);
        assert!((mesh.normals[0] - Vec3::Z).len() < 1e-12);
    }
}
