//! Exact geometry passed from evaluation to display, and the one placement
//! value every consumer composes and reads.
//!
//! These types are deliberately free of graph, display-plan and UI concerns.
//! A node evaluates to an exact [`Part`]-like value; the display service turns
//! that value into plans.
//!
//! # One placement path
//!
//! A body's placement is a [`Placement`]: a shared, immutable chain node whose
//! composed 4x4 affine matrix is cached when the placement node is evaluated
//! (`child = local ∘ parent`, the old first-to-last chain product). Cloning a
//! body or handing its placement to the exact, display, GPU-plan, edge, pick
//! or gizmo path clones one [`Arc`], not a growing `Vec` of operations. The chain itself is kept as an `Arc` parent
//! link for the few consumers that genuinely need the individual operations
//! (provenance, and the degenerate-matrix fallback in [`realize_brep`]); no hot
//! path walks it.
//!
//! [`Affine3`] is the single composition helper. Kernel B-rep realisation, CPU
//! edge placement, mesh placement, GPU plan matrices and the session gizmo all
//! read this value instead of re-implementing rotate/mirror/scale maths.

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

/// Row-major affine transform: `p' = M * [x, y, z, 1]`.
///
/// This is the one composition helper. Every placement consumer reads the
/// matrix from a [`Placement`] and applies it here; none of them rebuilds the
/// rotate/mirror/scale product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3 {
    m: [[f64; 4]; 4],
}

impl Default for Affine3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine3 {
    pub const IDENTITY: Self = Self {
        m: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    };

    #[must_use]
    pub const fn from_rows(m: [[f64; 4]; 4]) -> Self {
        Self { m }
    }

    /// All four rows, row-major.
    #[must_use]
    pub const fn rows(self) -> [[f64; 4]; 4] {
        self.m
    }

    #[must_use]
    pub const fn row(self, row: usize) -> [f64; 4] {
        self.m[row]
    }

    /// One column of the linear 3x3 part.
    #[must_use]
    pub const fn linear_column(self, column: usize) -> [f64; 3] {
        [self.m[0][column], self.m[1][column], self.m[2][column]]
    }

    #[must_use]
    pub const fn translation(self) -> [f64; 3] {
        [self.m[0][3], self.m[1][3], self.m[2][3]]
    }

    /// Rotation only, matching the mesh and B-rep rotation order (X, then Y,
    /// then Z).
    #[must_use]
    pub fn from_rotation(rotation: [f64; 3]) -> Self {
        let (rx, ry, rz) =
            (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let x_axis = [cy * cz, cy * sz, -sy];
        let y_axis = [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy];
        let z_axis = [
            x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
            x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
            x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
        ];
        Self {
            m: [
                [x_axis[0], y_axis[0], z_axis[0], 0.0],
                [x_axis[1], y_axis[1], z_axis[1], 0.0],
                [x_axis[2], y_axis[2], z_axis[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Translation after X/Y/Z rotation, matching the mesh and B-rep placement
    /// paths.
    #[must_use]
    pub fn from_place(offset: [f64; 3], rotation: [f64; 3]) -> Self {
        let mut result = Self::from_rotation(rotation);
        result.m[0][3] = offset[0];
        result.m[1][3] = offset[1];
        result.m[2][3] = offset[2];
        result
    }

    /// Reflection through `plane`, then `offset`, matching the B-rep mirror.
    #[must_use]
    pub fn from_mirror(plane: &str, offset: f64) -> Self {
        let mut m = Self::IDENTITY.m;
        match plane {
            "xz" => {
                m[1][1] = -1.0;
                m[1][3] = 2.0 * offset;
            }
            "yz" => {
                m[0][0] = -1.0;
                m[0][3] = 2.0 * offset;
            }
            _ => {
                m[2][2] = -1.0;
                m[2][3] = 2.0 * offset;
            }
        }
        Self { m }
    }

    /// Uniform scale about the origin.
    #[must_use]
    pub fn from_scale(factor: f64) -> Self {
        let mut m = Self::IDENTITY.m;
        m[0][0] = factor;
        m[1][1] = factor;
        m[2][2] = factor;
        Self { m }
    }

    /// A stored 3x3 linear map plus translation, for sketch planes.
    #[must_use]
    pub fn from_linear_translation(linear: [[f64; 3]; 3], translation: [f64; 3]) -> Self {
        Self {
            m: [
                [linear[0][0], linear[0][1], linear[0][2], translation[0]],
                [linear[1][0], linear[1][1], linear[1][2], translation[1]],
                [linear[2][0], linear[2][1], linear[2][2], translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// The linear 3x3 part.
    #[must_use]
    pub const fn linear(self) -> [[f64; 3]; 3] {
        [
            [self.m[0][0], self.m[0][1], self.m[0][2]],
            [self.m[1][0], self.m[1][1], self.m[1][2]],
            [self.m[2][0], self.m[2][1], self.m[2][2]],
        ]
    }

    /// `self` applied after `other`.
    #[must_use]
    pub fn compose(self, other: Self) -> Self {
        let mut m = [[0.0; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                m[row][column] = (0..4).map(|k| self.m[row][k] * other.m[k][column]).sum();
            }
        }
        Self { m }
    }

    #[must_use]
    pub fn apply_point(self, point: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * point[0]
                + self.m[0][1] * point[1]
                + self.m[0][2] * point[2]
                + self.m[0][3],
            self.m[1][0] * point[0]
                + self.m[1][1] * point[1]
                + self.m[1][2] * point[2]
                + self.m[1][3],
            self.m[2][0] * point[0]
                + self.m[2][1] * point[1]
                + self.m[2][2] * point[2]
                + self.m[2][3],
        ]
    }

    #[must_use]
    pub fn apply_vector(self, vector: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * vector[0] + self.m[0][1] * vector[1] + self.m[0][2] * vector[2],
            self.m[1][0] * vector[0] + self.m[1][1] * vector[1] + self.m[1][2] * vector[2],
            self.m[2][0] * vector[0] + self.m[2][1] * vector[1] + self.m[2][2] * vector[2],
        ]
    }

    #[must_use]
    pub fn determinant(self) -> f64 {
        let r = self.linear();
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    }

    #[must_use]
    pub fn inverse_linear(self) -> [[f64; 3]; 3] {
        let inverse = self.inverse();
        [
            [inverse.m[0][0], inverse.m[0][1], inverse.m[0][2]],
            [inverse.m[1][0], inverse.m[1][1], inverse.m[1][2]],
            [inverse.m[2][0], inverse.m[2][1], inverse.m[2][2]],
        ]
    }

    /// Inverse of a rigid/reflected affine transform.
    #[must_use]
    pub fn inverse(self) -> Self {
        let r = self.linear();
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        let inv_det = 1.0 / det;
        let inv = [
            [
                (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
                (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
                (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
            ],
            [
                (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
                (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
                (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
            ],
            [
                (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
                (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
                (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
            ],
        ];
        let t = [self.m[0][3], self.m[1][3], self.m[2][3]];
        let it = [
            -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2]),
            -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2]),
            -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2]),
        ];
        Self {
            m: [
                [inv[0][0], inv[0][1], inv[0][2], it[0]],
                [inv[1][0], inv[1][1], inv[1][2], it[1]],
                [inv[2][0], inv[2][1], inv[2][2], it[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// A stored column-major matrix (the export/STEP layout).
    #[must_use]
    pub fn from_column_major(m: [[f64; 4]; 4]) -> Self {
        let mut rows = [[0.0; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                rows[row][column] = m[column][row];
            }
        }
        Self { m: rows }
    }

    /// Column-major `[[f64; 4]; 4]` (the export/STEP layout).
    #[must_use]
    pub fn to_column_major_f64(self) -> [[f64; 4]; 4] {
        let mut out = [[0.0f64; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[column][row] = self.m[row][column];
            }
        }
        out
    }

    /// Column-major `[[f32; 4]; 4]` for GPU uploads (WGSL `mat4x4<f32>` layout).
    #[must_use]
    pub fn to_column_major_f32(self) -> [[f32; 4]; 4] {
        let mut out = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[column][row] = self.m[row][column] as f32;
            }
        }
        out
    }

    /// `m * scale` (the uniform scale baked into one placement).
    #[must_use]
    pub fn scaled(self, scale: f64) -> Self {
        let mut m = self.m;
        for row in &mut m[..3] {
            for value in &mut row[..3] {
                *value *= scale;
            }
        }
        Self { m }
    }

    /// Negate the third linear column (used to split a reflection out of a
    /// placement matrix).
    #[must_use]
    pub fn with_negated_z_column(self) -> Self {
        let mut m = self.m;
        for row in &mut m[..3] {
            row[2] = -row[2];
        }
        Self { m }
    }
}

/// Placement applied after a body's local display plan/mesh is produced.
///
/// One operation. A body keeps its composition as a [`Placement`]; this type is
/// never stored as a growing vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DisplayTransform {
    Place { offset: [f64; 3], rotation: [f64; 3] },
    Mirror { plane: &'static str, offset: f64 },
    Scale { factor: f64 },
}

impl DisplayTransform {
    /// The local affine matrix of this one operation.
    #[must_use]
    pub fn matrix(self) -> Affine3 {
        match self {
            Self::Place { offset, rotation } => Affine3::from_place(offset, rotation),
            Self::Mirror { plane, offset } => Affine3::from_mirror(plane, offset),
            Self::Scale { factor } => Affine3::from_scale(factor),
        }
    }

    /// Whether this operation keeps the outward orientation of a body.
    #[must_use]
    pub fn is_rigid_place(self) -> bool {
        matches!(self, Self::Place { .. })
    }

    #[must_use]
    pub fn is_mirror(self) -> bool {
        matches!(self, Self::Mirror { .. })
    }
}

/// One shared link in a body's placement chain.
///
/// The composed matrix is cached here, so reading a placement is O(1). The
/// parent link keeps the individual operations in O(1) storage per node for
/// consumers that genuinely need the chain; it is never cloned into a body.
#[derive(Debug)]
struct PlacementNode {
    parent: Option<Arc<PlacementNode>>,
    /// `None` only for a base matrix (a sketch's own plane).
    local: Option<DisplayTransform>,
    matrix: Affine3,
    /// Every operation in this chain is a rigid translate/rotate.
    rigid: bool,
    /// The chain contains a mirror; mesh consumers must flip face winding.
    mirrored: bool,
    depth: usize,
}

/// A body's or sketch's placement: one shared chain node with the composed
/// affine matrix cached.
///
/// Cloning this value is O(1); it never copies the history. Equality compares
/// the composed matrix (with an Arc fast path), so plan identity does not walk
/// the chain.
#[derive(Default)]
pub struct Placement(Option<Arc<PlacementNode>>);

impl std::fmt::Debug for Placement {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Placement")
            .field("matrix", self.matrix())
            .field("depth", &self.depth())
            .finish()
    }
}

impl PartialEq for Placement {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(left), Some(right)) if Arc::ptr_eq(left, right) => true,
            _ => self.matrix() == other.matrix(),
        }
    }
}

impl Placement {
    #[must_use]
    pub const fn identity() -> Self {
        Self(None)
    }

    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.0.is_none()
    }

    /// The composed transform. O(1): the matrix was cached when this node was
    /// evaluated.
    #[must_use]
    pub fn matrix(&self) -> &Affine3 {
        self.0.as_ref().map_or(&Affine3::IDENTITY, |node| &node.matrix)
    }

    /// Compose `local` after this placement: `child = local ∘ parent`, so the
    /// new operation is applied to the parent's already-placed geometry.
    #[must_use]
    pub fn then(&self, local: DisplayTransform) -> Self {
        let parent = self.0.clone();
        // The new local operation is applied after the parent: matrix =
        // local * parent (function composition `local ∘ parent`). This matches
        // the old first-to-last chain product T_n * ... * T_1.
        let matrix = match &parent {
            Some(node) => local.matrix().compose(node.matrix),
            None => local.matrix(),
        };
        let rigid = parent.as_ref().is_none_or(|node| node.rigid) && local.is_rigid_place();
        let mirrored = parent.as_ref().is_some_and(|node| node.mirrored) ^ local.is_mirror();
        let depth = parent.as_ref().map_or(1, |node| node.depth + 1);
        placement_stats::note_composition();
        Self(Some(Arc::new(PlacementNode {
            parent,
            local: Some(local),
            matrix,
            rigid,
            mirrored,
            depth,
        })))
    }

    /// A placement whose composed value is `matrix` with no operation history.
    /// Used for a sketch's own plane, which is already a matrix, not a chain.
    #[must_use]
    pub fn from_matrix(matrix: Affine3) -> Self {
        if matrix == Affine3::IDENTITY {
            return Self::identity();
        }
        placement_stats::note_composition();
        Self(Some(Arc::new(PlacementNode {
            parent: None,
            local: None,
            matrix,
            rigid: false,
            mirrored: matrix.determinant() < 0.0,
            depth: 1,
        })))
    }

    /// Build a placement from a sequence of operations, first applied first.
    #[must_use]
    pub fn from_ops(ops: impl IntoIterator<Item = DisplayTransform>) -> Self {
        ops.into_iter().fold(Self::identity(), |placement, op| placement.then(op))
    }

    /// Whether every operation is a rigid translate/rotate (no mirror, no
    /// scale). The realised body keeps its source proof in that case.
    #[must_use]
    pub fn is_rigid(&self) -> bool {
        self.0.as_ref().is_none_or(|node| node.rigid)
    }

    /// Whether the chain contains a mirror; mesh consumers flip winding.
    #[must_use]
    pub fn has_mirror(&self) -> bool {
        self.0.as_ref().is_some_and(|node| node.mirrored)
    }

    /// Number of operations in the chain. O(1).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.0.as_ref().map_or(0, |node| node.depth)
    }

    /// The individual operations, first applied first. O(depth) and allocates;
    /// only provenance and the degenerate fallback use this.
    #[must_use]
    pub fn ops(&self) -> impl Iterator<Item = DisplayTransform> + '_ {
        let mut stack = Vec::with_capacity(self.depth());
        let mut node = self.0.as_ref();
        while let Some(current) = node {
            if let Some(local) = current.local {
                stack.push(local);
            }
            node = current.parent.as_ref();
        }
        stack.reverse();
        stack.into_iter()
    }
}

impl From<DisplayTransform> for Placement {
    fn from(transform: DisplayTransform) -> Self {
        Self::identity().then(transform)
    }
}

impl FromIterator<DisplayTransform> for Placement {
    fn from_iter<T: IntoIterator<Item = DisplayTransform>>(iter: T) -> Self {
        Self::from_ops(iter)
    }
}

impl Clone for Placement {
    fn clone(&self) -> Self {
        placement_stats::note_clone();
        Self(self.0.clone())
    }
}

/// Test-visible proof that per-edit placement work is O(N).
///
/// `matrix_elements_copied` counts the 16 affine elements logically copied per
/// composed placement node and per [`Placement`] clone. Sharing a parent node
/// keeps both O(1) per body, so a chain evaluation is O(N); a representation
/// that copied each body's operation history would grow quadratically.
pub mod placement_stats {
    use std::cell::Cell;

    thread_local! {
        static COMPOSITIONS: Cell<u64> = const { Cell::new(0) };
        static MATRIX_ELEMENTS_COPIED: Cell<u64> = const { Cell::new(0) };
        static NODES: Cell<u64> = const { Cell::new(0) };
    }

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct PlacementCounters {
        pub compositions: u64,
        pub matrix_elements_copied: u64,
        pub nodes: u64,
    }

    /// Reset this thread's counters. Placement work for a chain is done on the
    /// evaluating thread, so a test can measure one evaluation without other
    /// tests or worker threads adding to it.
    pub fn reset() {
        COMPOSITIONS.with(|value| value.set(0));
        MATRIX_ELEMENTS_COPIED.with(|value| value.set(0));
        NODES.with(|value| value.set(0));
    }

    #[must_use]
    pub fn counters() -> PlacementCounters {
        PlacementCounters {
            compositions: COMPOSITIONS.with(Cell::get),
            matrix_elements_copied: MATRIX_ELEMENTS_COPIED.with(Cell::get),
            nodes: NODES.with(Cell::get),
        }
    }

    pub(super) fn note_composition() {
        COMPOSITIONS.with(|value| value.set(value.get() + 1));
        MATRIX_ELEMENTS_COPIED.with(|value| value.set(value.get() + 16));
        NODES.with(|value| value.set(value.get() + 1));
    }

    pub(super) fn note_clone() {
        MATRIX_ELEMENTS_COPIED.with(|value| value.set(value.get() + 16));
    }
}

/// Exact display input for one body: immutable B-rep plus placement.
#[derive(Clone, Debug)]
pub struct ExactBody {
    pub key: BodyKey,
    pub brep: Arc<solidcore::topology::Brep>,
    pub placement: Placement,
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
        Self { key, brep, placement: Placement::identity() }
    }

    #[must_use]
    pub fn with_placement(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }
}

impl ExactPart {
    #[must_use]
    pub fn singleton(key: BodyKey, brep: Arc<solidcore::topology::Brep>) -> Self {
        Self { bodies: vec![ExactBody::new(key, brep)] }
    }

    /// Bake each body's placement into standalone B-reps for export and exact
    /// consumers. Display planning never mutates these B-reps.
    pub fn realized_breps(&self) -> Result<Vec<solidcore::topology::Brep>, String> {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut breps = Vec::with_capacity(self.bodies.len());
        for body in &self.bodies {
            let mut brep = body
                .brep
                .clone_standing_for_self()
                .map_err(|error| format!("export B-rep copy: {error}"))?;
            realize_brep(&mut kernel, &mut brep, &body.placement)?;
            breps.push(brep);
        }
        Ok(breps)
    }
}

/// Apply a body's one placement value to its B-rep.
///
/// The composed matrix is a uniform scale times an orthogonal map (the only
/// operations are rigid placement, reflection and uniform scale), so it splits
/// into an optional fixed reflection, a rotation and a uniform scale. The
/// reflection is applied first and the rest in one kernel transform call. This
/// is the exact B-rep realiser; it does not replay a per-body operation list.
pub fn realize_brep(
    kernel: &mut solidcore::kernel::Kernel,
    brep: &mut solidcore::topology::Brep,
    placement: &Placement,
) -> Result<(), String> {
    if placement.is_identity() {
        return Ok(());
    }
    let matrix = *placement.matrix();
    let determinant = matrix.determinant();
    let scale = determinant.abs().cbrt();
    // A zero/NaN scale (or a matrix with shear, which these operations cannot
    // produce) cannot be represented as rotation*scale. Replay the short chain
    // for that degenerate case only; normal placements never take this path.
    if !scale.is_finite() || scale <= 1.0e-12 {
        for transform in placement.ops() {
            apply_transform_brep(kernel, brep, transform)?;
        }
        return Ok(());
    }
    // Strip the uniform scale, leaving an orthogonal map. A negative
    // determinant is a reflection; split out a fixed z reflection so the
    // remaining axes are right-handed (the kernel's placement is right-handed).
    let mut rotation = matrix.scaled(1.0 / scale);
    let reflected = determinant < 0.0;
    if reflected {
        rotation = rotation.with_negated_z_column();
        mirror_z0_brep(kernel, brep)?;
    }
    let x_axis = rotation.linear_column(0);
    let y_axis = rotation.linear_column(1);
    let translation = matrix.translation();
    let axes = solidcore::geom::axis2_placement::Axis2Placement::new(
        solidcore::math::Point3::new(translation[0], translation[1], translation[2]),
        solidcore::math::Vec3::new(x_axis[0], x_axis[1], x_axis[2]),
        solidcore::math::Vec3::new(y_axis[0], y_axis[1], y_axis[2]),
    );
    solidcore::topology::brep::p5::transform(
        kernel,
        brep,
        &axes,
        Some(solidcore::math::Vec3::new(scale, scale, scale)),
        true,
    )
    .map_err(|error| error.to_string())
}

/// Apply one placement operation. Only the degenerate fallback uses this; body
/// placement goes through [`realize_brep`].
fn apply_transform_brep(
    kernel: &mut solidcore::kernel::Kernel,
    brep: &mut solidcore::topology::Brep,
    transform: DisplayTransform,
) -> Result<(), String> {
    use solidcore::geom::axis2_placement::Axis2Placement;
    use solidcore::math::{Point3, Vec3};
    match transform {
        DisplayTransform::Place { offset, rotation } => {
            let (rx, ry, rz) =
                (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
            let (sx, cx) = rx.sin_cos();
            let (sy, cy) = ry.sin_cos();
            let (sz, cz) = rz.sin_cos();
            let x_axis = Vec3::new(cy * cz, cy * sz, -sy);
            let y_axis = Vec3::new(sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy);
            let placement =
                Axis2Placement::new(Point3::new(offset[0], offset[1], offset[2]), x_axis, y_axis);
            solidcore::topology::brep::p5::transform(kernel, brep, &placement, None, true)
                .map_err(|error| error.to_string())
        }
        DisplayTransform::Mirror { plane, offset } => {
            let (origin, x_axis, y_axis) = match plane {
                "xz" => (
                    Point3::new(0.0, offset, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                ),
                "yz" => (
                    Point3::new(offset, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                ),
                _ => (
                    Point3::new(0.0, 0.0, offset),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                ),
            };
            let mut standalone = solidcore::topology::brep::p3::brep_copy(kernel, brep)
                .map_err(|error| error.to_string())?;
            solidcore::topology::brep::mirror::mirror(
                kernel,
                &mut standalone,
                &Axis2Placement::new(origin, x_axis, y_axis),
            )
            .map_err(|error| error.to_string())?;
            *brep = standalone;
            Ok(())
        }
        DisplayTransform::Scale { factor } => {
            let placement = Axis2Placement::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            solidcore::topology::brep::p5::transform(
                kernel,
                brep,
                &placement,
                Some(Vec3::new(factor, factor, factor)),
                true,
            )
            .map_err(|error| error.to_string())
        }
    }
}

fn mirror_z0_brep(
    kernel: &mut solidcore::kernel::Kernel,
    brep: &mut solidcore::topology::Brep,
) -> Result<(), String> {
    use solidcore::geom::axis2_placement::Axis2Placement;
    use solidcore::math::{Point3, Vec3};
    let mut standalone = solidcore::topology::brep::p3::brep_copy(kernel, brep)
        .map_err(|error| error.to_string())?;
    solidcore::topology::brep::mirror::mirror(
        kernel,
        &mut standalone,
        &Axis2Placement::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
    )
    .map_err(|error| error.to_string())?;
    *brep = standalone;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old per-operation point transform, kept as the equivalence oracle.
    /// Placement matrices must reproduce this to 1e-9.
    fn reference_point(mut point: [f64; 3], ops: &[DisplayTransform]) -> [f64; 3] {
        for op in ops {
            match *op {
                DisplayTransform::Place { offset, rotation } => {
                    let (rx, ry, rz) = (
                        rotation[0].to_radians(),
                        rotation[1].to_radians(),
                        rotation[2].to_radians(),
                    );
                    let (sx, cx) = rx.sin_cos();
                    let (sy, cy) = ry.sin_cos();
                    let (sz, cz) = rz.sin_cos();
                    (point[1], point[2]) =
                        (point[1] * cx - point[2] * sx, point[1] * sx + point[2] * cx);
                    (point[0], point[2]) =
                        (point[0] * cy + point[2] * sy, -point[0] * sy + point[2] * cy);
                    (point[0], point[1]) =
                        (point[0] * cz - point[1] * sz, point[0] * sz + point[1] * cz);
                    point[0] += offset[0];
                    point[1] += offset[1];
                    point[2] += offset[2];
                }
                DisplayTransform::Mirror { plane, offset } => match plane {
                    "xz" => point[1] = 2.0 * offset - point[1],
                    "yz" => point[0] = 2.0 * offset - point[0],
                    _ => point[2] = 2.0 * offset - point[2],
                },
                DisplayTransform::Scale { factor } => {
                    point[0] *= factor;
                    point[1] *= factor;
                    point[2] *= factor;
                }
            }
        }
        point
    }

    fn approx(left: [f64; 3], right: [f64; 3], context: &str) {
        for axis in 0..3 {
            assert!(
                (left[axis] - right[axis]).abs() <= 1.0e-9,
                "{context}: axis {axis}: {left:?} vs {right:?}",
            );
        }
    }

    fn sample_ops() -> Vec<DisplayTransform> {
        vec![
            DisplayTransform::Place { offset: [1.0, -2.0, 3.0], rotation: [10.0, 20.0, 30.0] },
            DisplayTransform::Scale { factor: 1.7 },
            DisplayTransform::Place { offset: [-4.0, 5.0, 6.0], rotation: [-15.0, 5.0, 70.0] },
            DisplayTransform::Mirror { plane: "yz", offset: 3.5 },
            DisplayTransform::Place { offset: [0.5, 0.25, -0.75], rotation: [45.0, -30.0, 90.0] },
        ]
    }

    #[test]
    fn composed_placement_matches_the_old_chain_product() {
        let ops = sample_ops();
        let placement = Placement::from_ops(ops.iter().copied());
        for point in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-5.0, 0.5, 8.0]] {
            approx(
                placement.matrix().apply_point(point),
                reference_point(point, &ops),
                "composed placement",
            );
        }
    }

    #[test]
    fn placement_clone_is_constant_time_and_shares_the_chain() {
        let placement = Placement::from_ops(sample_ops());
        let first = placement.matrix().rows();
        let clone = placement.clone();
        assert_eq!(clone.matrix().rows(), first);
        assert_eq!(clone.depth(), placement.depth());
    }

    #[test]
    fn placement_stats_count_per_composition_not_history() {
        placement_stats::reset();
        let mut placement = Placement::identity();
        for _ in 0..100 {
            placement = placement
                .then(DisplayTransform::Place { offset: [1.0, 0.0, 0.0], rotation: [0.0; 3] });
        }
        let cloned = placement.clone();
        assert_eq!(cloned.depth(), 100);
        let counters = placement_stats::counters();
        assert_eq!(counters.compositions, 100);
        assert_eq!(counters.nodes, 100);
        assert_eq!(
            counters.matrix_elements_copied,
            100 * 16 + 16,
            "one matrix per node plus one per body clone",
        );
    }

    #[test]
    fn mirror_and_rigid_flags_track_the_chain() {
        let placement = Placement::from_ops([DisplayTransform::Place {
            offset: [1.0, 0.0, 0.0],
            rotation: [0.0; 3],
        }]);
        assert!(placement.is_rigid());
        assert!(!placement.has_mirror());

        let mirrored = placement.then(DisplayTransform::Mirror { plane: "yz", offset: 2.0 });
        assert!(!mirrored.is_rigid());
        assert!(mirrored.has_mirror());

        let scaled = Placement::identity().then(DisplayTransform::Scale { factor: 2.0 });
        assert!(!scaled.is_rigid());
        assert!(!scaled.has_mirror());
    }

    #[test]
    fn realized_brep_matches_the_old_operation_replay() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = solidcore::primitive::api_primitives::create_box(
            &mut kernel,
            solidcore::math::Point3::new(-2.0, -3.0, -4.0),
            4.0,
            6.0,
            8.0,
        )
        .expect("box");
        let ops = sample_ops();
        let placement = Placement::from_ops(ops.iter().copied());

        let mut composed = brep.clone_standing_for_self().expect("copy");
        realize_brep(&mut kernel, &mut composed, &placement).expect("realise composed");

        let mut replayed = brep.clone_standing_for_self().expect("copy");
        for operation in &ops {
            apply_transform_brep(&mut kernel, &mut replayed, *operation).expect("replay");
        }

        let points = |brep: &solidcore::topology::Brep| {
            brep.vertex_list()
                .iter()
                .map(|vertex| {
                    let point = brep.vertex(*vertex).expect("vertex").point;
                    [point.x, point.y, point.z]
                })
                .collect::<Vec<_>>()
        };
        let composed_points = points(&composed);
        let replayed_points = points(&replayed);
        assert_eq!(composed_points.len(), replayed_points.len());
        for (left, right) in composed_points.iter().zip(&replayed_points) {
            approx(*left, *right, "realised B-rep vertex");
        }
    }

    #[test]
    fn body_keys_identify_source_and_body() {
        assert_eq!(BodyKey::new(7, 2), BodyKey::new(7, 2));
        assert_ne!(BodyKey::new(7, 2), BodyKey::new(7, 3));
        assert_ne!(BodyKey::new(7, 2), BodyKey::new(8, 2));
    }
}
