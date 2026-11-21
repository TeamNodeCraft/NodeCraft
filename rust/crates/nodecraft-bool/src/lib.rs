//! Safe B-rep wrappers over the solidcore Boolean APIs.
//!
//! NodeCraft's protocol exchanges display [`Mesh`]es, but the solidcore Boolean
//! entry points are B-rep-level: they consume and return
//! `solidcore::topology::Brep` values. This crate keeps that boundary explicit:
//!
//! * [`BooleanOp`] is the protocol-facing operation selector.
//! * [`boolean_breps`] wraps [`solidcore::boolean::api_brep::boolean`] (union,
//!   difference, intersection).
//! * [`display_mesh`] bridges any B-rep to a protocol [`Mesh`] through
//!   `nodecraft-layers`, and [`boolean_mesh`] combines an operation with that
//!   bridge.
//! * [`box_brep`] / [`box_brep_at`] construct solidcore box operands in B-rep
//!   coordinates.
//!
//! The shell wrappers (`shell_brep`, `shell_brep_with_options`,
//! `shell_brep_solid`, `shell_mesh`) were the legacy nodecraft-bool path. The
//! Shell node uses `nodecraft_ops::OpsSession::shell_body`, and the wrappers had
//! no product caller, so they are gone.
//!
//! # Current solidcore coverage (pinned `Cargo.lock` revision)
//!
//! Boolean union, difference and intersection run end to end for the box
//! operands exercised by the tests (disjoint, nested/containment, touching and
//! overlapping pairs), including the classification adapters that used to stop
//! at `SmEdge::GetTolerance` and the face-owned trimmed surface cache. One
//! representation gap remains: a disjoint intersection returns an empty result
//! whose B-rep has faces but no shell (`brep_is_manifold_solid` false, no
//! display mesh), so callers must not treat it as a valid empty solid.
//!
//! Errors are mapped to a [`String`] whose [`UNPORTED_PREFIX`] marks a missing
//! solidcore dependency rather than bad caller input; use [`is_unported_error`].

#![forbid(unsafe_code)]

use nodecraft_protocol::Mesh;
use solidcore::boolean::api_brep as boolean_api;
use solidcore::error::SolidError;
use solidcore::geom::a_object::BooleanOperation;
#[cfg(test)]
use solidcore::geom::attribute::AttributeStore;
use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives as api;
#[cfg(test)]
use solidcore::topology::topology::NotifyRecords;
use solidcore::topology::{Brep, FaceId};

/// Prefix of every error caused by a solidcore dependency that is not ported yet.
///
/// Callers that want to tell "this kernel cannot do it yet" apart from "these operands
/// are invalid" should use [`is_unported_error`].
pub const UNPORTED_PREFIX: &str = "solidcore unported: ";

/// Boolean operation selector for [`boolean_breps`] and [`boolean_mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// `A ∪ B`.
    Union,
    /// `A \ B` (the second operand is subtracted from the first).
    Difference,
    /// `A ∩ B`.
    Intersection,
}

impl BooleanOp {
    /// The solidcore merge operation this selector maps to.
    #[must_use]
    pub const fn to_solidcore(self) -> BooleanOperation {
        match self {
            Self::Union => BooleanOperation::Union,
            Self::Difference => BooleanOperation::Difference,
            Self::Intersection => BooleanOperation::Intersection,
        }
    }
}

impl std::fmt::Display for BooleanOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Union => "union",
            Self::Difference => "difference",
            Self::Intersection => "intersection",
        })
    }
}

/// Turns a solidcore error into a caller-facing message, keeping unported-dependency
/// failures distinguishable from invalid input.
#[must_use]
pub fn map_error(context: &str, error: &SolidError) -> String {
    match error {
        SolidError::Unported { upstream, unit } => {
            format!("{UNPORTED_PREFIX}{context}: {upstream} ({unit}) is not ported")
        }
        _ => format!("{context}: {error}"),
    }
}

/// Whether an error message produced by this crate reports an unported solidcore
/// dependency (as opposed to invalid operands or an ordinary kernel failure).
#[must_use]
pub fn is_unported_error(message: &str) -> bool {
    message.starts_with(UNPORTED_PREFIX)
}

/// Creates the solidcore box of `length` (X) by `width` (Y) by `height` (Z) with its
/// lower-left corner at `origin`, in B-rep coordinates.
///
/// This is the operand constructor for [`boolean_breps`] and [`shell_brep`]; display
/// meshes produced from its results are therefore not centered (unlike
/// [`nodecraft_solidcore::box_mesh`], which centers the primitive at the origin).
///
/// # Errors
///
/// A message from solidcore (non-finite or non-positive dimensions, or an unported
/// constructor dependency).
pub fn box_brep_at(origin: [f64; 3], length: f64, width: f64, height: f64) -> Result<Brep, String> {
    with_wrapper_kernel(|kernel| {
        api::create_box(kernel, Point3::new(origin[0], origin[1], origin[2]), length, width, height)
            .map_err(|error| map_error("solidcore create_box", &error))
    })
}

thread_local! {
    /// One long-lived wrapper kernel for callers that do not own a worker
    /// kernel. Reusing it keeps memoised surfaces and tessellation warm.
    static WRAPPER_KERNEL: std::cell::RefCell<Option<Kernel>> = const { std::cell::RefCell::new(None) };
}

fn with_wrapper_kernel<T>(f: impl FnOnce(&mut Kernel) -> Result<T, String>) -> Result<T, String> {
    WRAPPER_KERNEL.with(|cell| {
        let mut slot = cell.borrow_mut();
        let kernel = slot.get_or_insert_with(nodecraft_kernel_profile::kernel);
        f(kernel)
    })
}

/// [`box_brep_at`] with the box's lower-left corner at the origin.
///
/// # Errors
///
/// A message from solidcore (non-finite or non-positive dimensions, or an unported
/// constructor dependency).
pub fn box_brep(length: f64, width: f64, height: f64) -> Result<Brep, String> {
    box_brep_at([0.0, 0.0, 0.0], length, width, height)
}

/// The face ids of `brep` in solidcore's owning-list order.
///
/// # Errors
///
/// A message if solidcore cannot enumerate the B-rep's faces.
pub fn brep_faces(brep: &Brep) -> Result<Vec<FaceId>, String> {
    use solidcore::topology::brep::p1 as bp1;
    bp1::faces(brep, None).map_err(|error| map_error("solidcore brep faces", &error))
}

/// Booleans two B-reps (`A op B`) and returns the resulting B-rep.
///
/// The operands are consumed: `a` becomes the result and `b` is deleted by the
/// solidcore merge, matching `SmApiBoolean`.
///
/// **Coverage caveat:** see the module documentation. Boolean results are exact for
/// the box cases covered by the tests, but a disjoint intersection is represented as
/// an invalid empty B-rep, and non-box geometry is not covered by these tests yet.
///
/// # Errors
///
/// A message from solidcore. An unported merge dependency is reported with
/// [`UNPORTED_PREFIX`]; see [`is_unported_error`].
/// N-ary Boolean over lists of B-reps, using SolidCore's `boolean_lists`.
/// This avoids chaining hundreds of pairwise Booleans through intermediate
/// topology, which is where long difference chains fail with status 1001.
pub fn boolean_lists_breps(
    kernel: &mut Kernel,
    a: &[Brep],
    b: &[Brep],
    op: BooleanOp,
) -> Result<Vec<Brep>, String> {
    let operation = match op {
        BooleanOp::Union => 1,
        BooleanOp::Intersection => 0,
        BooleanOp::Difference => 3,
    };
    let result = solidcore::boolean::api_brep::boolean_lists(kernel, a, &[], b, &[], operation)
        .map_err(|error| map_error("solidcore boolean lists", &error))?;
    Ok(result.breps)
}

/// Batched list Boolean. One huge list can exceed the merge's intermediate
/// topology budget; chunks of a modest size keep every merge in the range the
/// kernel handles reliably while still avoiding pairwise chaining.
pub fn boolean_lists_batched(
    kernel: &mut Kernel,
    a: &[Brep],
    b: &[Brep],
    op: BooleanOp,
) -> Result<Vec<Brep>, String> {
    const BATCH: usize = 4;
    if b.len() <= BATCH {
        return boolean_lists_breps(kernel, a, b, op);
    }
    let mut current: Vec<Brep> = a
        .iter()
        .map(|brep| brep.clone_standing_for_self())
        .collect::<Result<_, _>>()
        .map_err(|error| map_error("Boolean input copy", &error))?;
    for (index, chunk) in b.chunks(BATCH).enumerate() {
        current = boolean_lists_breps(kernel, &current, chunk, op)
            .map_err(|error| format!("Boolean batch {index} ({} tools): {error}", chunk.len()))?;
        if current.is_empty() {
            break;
        }
    }
    Ok(current)
}

pub fn boolean_breps(op: BooleanOp, a: Brep, b: Brep) -> Result<Brep, String> {
    with_wrapper_kernel(|kernel| boolean_breps_with_kernel(kernel, op, a, b))
}

/// Boolean B-rep operation using the caller's kernel.
///
/// Display tessellation of the result must use the same kernel that built it:
/// SolidCore surfaces/topology can retain kernel-owned state, and a fresh
/// kernel can display the pre-Boolean surfaces.
pub fn boolean_breps_with_kernel(
    kernel: &mut Kernel,
    op: BooleanOp,
    a: Brep,
    b: Brep,
) -> Result<Brep, String> {
    boolean_with_kernel(kernel, op, a, b)
}

fn boolean_with_kernel(
    kernel: &mut Kernel,
    op: BooleanOp,
    a: Brep,
    b: Brep,
) -> Result<Brep, String> {
    let outcome = boolean_api::boolean(kernel, a, b, op.to_solidcore())
        .map_err(|error| map_error(&format!("solidcore boolean {op}"), &error))?;
    Ok(outcome.result)
}

/// Tessellates a solidcore B-rep into a protocol [`Mesh`] in B-rep coordinates.
///
/// Uses the upstream `solidcore-tess` path through `nodecraft-layers`. The mesh is a
/// display approximation: seam vertices may be duplicated and the index buffer is not
/// welded. The result is not re-centered.
///
/// # Errors
///
/// A message when tessellation fails or the B-rep has no faces.
pub fn display_mesh(brep: &Brep) -> Result<Mesh, String> {
    if brep.faces().len() == 0 {
        return Err("display tessellation produced no triangles".into());
    }
    with_wrapper_kernel(|kernel| {
        let mut copy =
            brep.clone_standing_for_self().map_err(|error| format!("display copy: {error}"))?;
        let mut mesh = nodecraft_layers::tessellate_kernel(kernel, &mut copy, 0.05)
            .map_err(|error| format!("solidcore-tess display mesh: {error}"))?
            .mesh;
        if mesh.faces.is_empty() {
            return Err("display tessellation produced no triangles".into());
        }
        mesh.lightweight = true;
        Ok(mesh)
    })
}

/// [`boolean_breps`] followed by [`display_mesh`].
///
/// This is the Mesh-facing wrapper a NodeCraft node needs. It inherits every
/// [`boolean_breps`] caveat.
///
/// # Errors
///
/// The Boolean error, or a display-tessellation error.
pub fn boolean_mesh(op: BooleanOp, a: Brep, b: Brep) -> Result<Mesh, String> {
    let brep = with_wrapper_kernel(|kernel| boolean_with_kernel(kernel, op, a, b))?;
    display_mesh(&brep)
}

#[cfg(test)]
mod test_identity {
    use solidcore::error::EntityRef;
    use solidcore::geom::attribute::ObjectKey;
    use solidcore::geom::object::ObjectRef;

    /// Reserved identity namespace for source (input) topology.
    const ORIGINAL_NAMESPACE: u64 = 100_000_000_000;
    /// Per-object-kind stride inside a namespace (`type_code * ENTITY_STRIDE + index`).
    const ENTITY_STRIDE: u64 = 1_000_000;
    /// `SmFaceuse_TYPE` (solidcore's `ShellType::Faceuse` code; no public constant).
    const FACEUSE_TYPE: i64 = 16022;

    fn type_code_of(entity: EntityRef) -> i64 {
        use solidcore::topology as topo;
        match entity {
            EntityRef::Brep(_) => solidcore::geom::a_object::BREP_TYPE,
            EntityRef::Shell(_) => topo::Shell::type_code(),
            EntityRef::Face(_) => topo::face::p2::TYPE,
            EntityRef::Faceuse(_) => FACEUSE_TYPE,
            EntityRef::Edge(_) => topo::edge::TYPE,
            EntityRef::Edgeuse(_) => topo::edgeuse::p1::TYPE,
            EntityRef::Vertex(_) => topo::vertex::TYPE,
            EntityRef::Vertexuse(_) => topo::vertexuse::TYPE,
            EntityRef::Loop(_) => topo::loopuse::LOOP_TYPE,
            EntityRef::Loopuse(_) => topo::loopuse::TYPE,
            EntityRef::Region(_) => topo::Region::type_code(),
        }
    }

    fn index_of(entity: EntityRef) -> u64 {
        u64::from(match entity {
            EntityRef::Brep(id) => id.index(),
            EntityRef::Shell(id) => id.index(),
            EntityRef::Face(id) => id.index(),
            EntityRef::Faceuse(id) => id.index(),
            EntityRef::Edge(id) => id.index(),
            EntityRef::Edgeuse(id) => id.index(),
            EntityRef::Vertex(id) => id.index(),
            EntityRef::Vertexuse(id) => id.index(),
            EntityRef::Loop(id) => id.index(),
            EntityRef::Loopuse(id) => id.index(),
            EntityRef::Region(id) => id.index(),
        })
    }

    pub(super) fn original_key_of(entity: EntityRef) -> ObjectRef {
        let kind = u64::try_from(type_code_of(entity)).unwrap_or_default();
        ObjectRef {
            type_code: type_code_of(entity),
            key: ObjectKey(ORIGINAL_NAMESPACE + kind * ENTITY_STRIDE + index_of(entity)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_identity::original_key_of;
    use super::*;

    const BOUNDS_TOL: f32 = 1.0e-4;

    /// Vertex bounds of a display mesh.
    fn mesh_bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        (min, max)
    }

    fn assert_bounds(mesh: &Mesh, expected_min: [f32; 3], expected_max: [f32; 3]) {
        let (min, max) = mesh_bounds(mesh);
        for axis in 0..3 {
            assert!(
                (min[axis] - expected_min[axis]).abs() <= BOUNDS_TOL,
                "min[{axis}] = {} != {}",
                min[axis],
                expected_min[axis]
            );
            assert!(
                (max[axis] - expected_max[axis]).abs() <= BOUNDS_TOL,
                "max[{axis}] = {} != {}",
                max[axis],
                expected_max[axis]
            );
        }
    }

    /// Exact B-rep volume through solidcore's mass-properties integrator.
    fn volume(brep: &Brep) -> Result<f64, String> {
        let mut copy = brep.clone_standing_for_self().map_err(|error| error.to_string())?;
        let store = AttributeStore::default();
        let attributes = solidcore::mass::integrator::properties::PropertyAttributes {
            store: &store,
            key_of: &|entity| original_key_of(entity).key,
        };
        solidcore::mass::integrator::properties::brep_volume(
            &nodecraft_kernel_profile::kernel(),
            &mut copy,
            1.0e-6,
            &attributes,
        )
        .map_err(|error| format!("brep_volume: {error}"))
    }

    fn disjoint_boxes() -> (Brep, Brep) {
        (
            box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box A"),
            box_brep_at([20.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box B"),
        )
    }

    #[test]
    fn boolean_op_maps_to_the_solidcore_operation() {
        assert_eq!(BooleanOp::Union.to_solidcore(), BooleanOperation::Union);
        assert_eq!(BooleanOp::Difference.to_solidcore(), BooleanOperation::Difference);
        assert_eq!(BooleanOp::Intersection.to_solidcore(), BooleanOperation::Intersection);
        assert_eq!(BooleanOp::Union.to_string(), "union");
        assert_eq!(BooleanOp::Difference.to_string(), "difference");
        assert_eq!(BooleanOp::Intersection.to_string(), "intersection");
    }

    #[test]
    fn error_mapping_separates_unported_dependencies_from_invalid_input() {
        let unported = map_error(
            "solidcore boolean union",
            &SolidError::unported("SmEdge::GetTolerance", "topology.edge"),
        );
        assert!(is_unported_error(&unported), "{unported}");
        assert!(unported.contains("SmEdge::GetTolerance"), "{unported}");
        assert!(unported.contains("topology.edge"), "{unported}");

        let invalid = map_error(
            "solidcore boolean union",
            &SolidError::invalid_input("boolean::api_brep", "bad operand"),
        );
        assert!(!is_unported_error(&invalid), "{invalid}");
    }

    #[test]
    fn two_boxes_are_solidcore_boxes_with_expected_bounds() {
        let a = box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box A");
        let b = box_brep_at([5.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box B");
        assert_eq!(brep_faces(&a).expect("faces A").len(), 6);
        assert_eq!(brep_faces(&b).expect("faces B").len(), 6);
        let mesh_a = display_mesh(&a).expect("display mesh A");
        let mesh_b = display_mesh(&b).expect("display mesh B");
        assert_bounds(&mesh_a, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert_bounds(&mesh_b, [5.0, 0.0, 0.0], [15.0, 10.0, 10.0]);
    }

    /// Disjoint operands are exact (volumes 2000 and 1000 for two 10-cubes).
    #[test]
    fn boolean_disjoint_union_and_difference_are_exact() {
        let (a, b) = disjoint_boxes();
        let union = boolean_breps(BooleanOp::Union, a, b).expect("disjoint union");
        assert_eq!(union.faces().len(), 12);
        assert_eq!(volume(&union).expect("union volume"), 2000.0);
        let mesh = display_mesh(&union).expect("union display mesh");
        assert_bounds(&mesh, [0.0, 0.0, 0.0], [30.0, 10.0, 10.0]);
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&union).expect("manifold"));

        let (a, b) = disjoint_boxes();
        let difference = boolean_breps(BooleanOp::Difference, a, b).expect("disjoint difference");
        assert_eq!(difference.faces().len(), 6);
        assert_eq!(volume(&difference).expect("difference volume"), 1000.0);
        let mesh = display_mesh(&difference).expect("difference display mesh");
        assert_bounds(&mesh, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    }

    /// A disjoint intersection is geometrically empty (volume 0), but solidcore
    /// represents it as faces with no shell: manifold query false and no display mesh.
    /// The wrapper must not be consumed as a valid empty solid today.
    #[test]
    fn boolean_disjoint_intersection_is_empty_but_invalid_topology() {
        let (a, b) = disjoint_boxes();
        let intersection =
            boolean_breps(BooleanOp::Intersection, a, b).expect("disjoint intersection call");
        assert_eq!(volume(&intersection).expect("intersection volume"), 0.0);
        assert_eq!(intersection.shells().len(), 0);
        assert!(
            !solidcore::query::api_queries::brep_is_manifold_solid(&intersection)
                .expect("manifold query")
        );
        assert!(display_mesh(&intersection).is_err());
    }

    /// Nested operands are classified correctly on the current Boolean adapter.
    #[test]
    fn boolean_nested_operands_are_classified() {
        let nested = || {
            (
                box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("outer box"),
                box_brep_at([2.0, 2.0, 2.0], 4.0, 4.0, 4.0).expect("inner box"),
            )
        };
        let (a, b) = nested();
        let union = boolean_breps(BooleanOp::Union, a, b).expect("nested union call");
        assert_eq!(union.faces().len(), 6);
        assert_eq!(volume(&union).expect("union volume"), 1000.0);

        let (a, b) = nested();
        let difference =
            boolean_breps(BooleanOp::Difference, a, b).expect("nested difference call");
        assert_eq!(volume(&difference).expect("difference volume"), 936.0);

        let (a, b) = nested();
        let intersection =
            boolean_breps(BooleanOp::Intersection, a, b).expect("nested intersection call");
        assert_eq!(volume(&intersection).expect("intersection volume"), 64.0);
    }

    /// Overlapping boxes produce real Boolean results; the display mesh must use the
    /// live faces only (Boolean edits leave deleted face records in the arena).
    #[test]
    fn boolean_overlapping_boxes_are_classified() {
        for (op, expected_volume, expected_min, expected_max) in [
            (BooleanOp::Union, 1500.0, [0.0, 0.0, 0.0], [15.0, 10.0, 10.0]),
            (BooleanOp::Difference, 500.0, [0.0, 0.0, 0.0], [5.0, 10.0, 10.0]),
            (BooleanOp::Intersection, 500.0, [5.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        ] {
            let a = box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box A");
            let b = box_brep_at([5.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box B");
            let result = boolean_breps(op, a, b).unwrap_or_else(|error| panic!("{op}: {error}"));
            assert!(
                (volume(&result).expect("result volume") - expected_volume).abs() < 1e-6,
                "{op}: wrong volume"
            );
            let mesh = display_mesh(&result).expect("result display mesh");
            assert_bounds(&mesh, expected_min, expected_max);
        }
    }

    /// Touching, coplanar-face operands now go through the trimmed-surface cache.
    #[test]
    fn boolean_touching_boxes_are_classified() {
        for (op, expected_volume) in [
            (BooleanOp::Union, 2000.0),
            (BooleanOp::Difference, 1000.0),
            (BooleanOp::Intersection, 0.0),
        ] {
            let a = box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box A");
            let b = box_brep_at([10.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box B");
            let result = boolean_breps(op, a, b).expect("touching boolean");
            assert!(
                (volume(&result).expect("touching volume") - expected_volume).abs() < 1e-6,
                "{op}"
            );
        }
    }

    /// Evidence that `SmMerge` construction succeeds on the current solidcore. The
    /// manifold stage may now return a defined result instead of an unported error;
    /// either outcome is accepted, but a panic is not.
    #[test]
    fn boolean_construction_succeeds_before_the_manifold_intersection_fails() {
        use solidcore::boolean::merge::p3::{BooleanDriver, DependencyMerge};
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut primary = Some(box_brep(10.0, 10.0, 10.0).expect("box A"));
        let mut other = Some(box_brep(10.0, 10.0, 10.0).expect("box B"));
        let mut driver = DependencyMerge::default();
        driver
            .construct(&mut kernel, &mut primary, &mut other, 0.0, 0.0, None)
            .expect("SmMerge construction is ported");
        let mut manifold = true;
        let mut lamina_edges = 0u64;
        match driver.manifold_boolean(
            &mut kernel,
            &mut primary,
            &mut other,
            &mut manifold,
            &mut lamina_edges,
            BooleanOperation::Union,
            None,
        ) {
            Ok(_) => {}
            Err(error) => assert!(error.is_unported(), "{error:?}"),
        }
    }

    #[test]
    fn boolean_mesh_returns_the_overlap_union() {
        let a = box_brep_at([0.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box A");
        let b = box_brep_at([5.0, 0.0, 0.0], 10.0, 10.0, 10.0).expect("box B");
        let mesh = boolean_mesh(BooleanOp::Union, a, b).expect("overlap mesh");
        assert!(!mesh.vertices.is_empty());
        assert_bounds(&mesh, [0.0, 0.0, 0.0], [15.0, 10.0, 10.0]);
    }

    /// Stale-claim check: `SmApiShellBrepFull` hardcodes `SmBrep::SmBrep`
    /// (`topology.brep_p3`), but the B-rep copy constructor is ported as
    /// `topology::brep::p3::copy_brep` and copies a box.
    #[test]
    fn brep_copy_constructor_is_ported_despite_the_api_shell_stub() {
        let source = box_brep(10.0, 10.0, 10.0).expect("box");
        let mut records = NotifyRecords::new();
        let mut attributes = AttributeStore::default();
        let mut env = solidcore::topology::edge::NotifyEnv {
            records: &mut records,
            attributes: &mut attributes,
            context: None,
            key_of: &original_key_of,
        };
        let copied = solidcore::topology::brep::p3::copy_brep(
            &mut nodecraft_kernel_profile::kernel(),
            &mut env,
            &source,
            &original_key_of,
            None,
            None,
            true,
            true,
        )
        .expect("B-rep copy constructor is ported");
        assert_eq!(copied.faces().len(), 6);
        assert!(!copied.edges().is_empty());
    }

    #[test]
    fn display_mesh_of_a_box_is_in_brep_coordinates() {
        let mesh = display_mesh(&box_brep(10.0, 4.0, 20.0).expect("box")).expect("mesh");
        assert!(!mesh.faces.is_empty());
        assert_eq!(mesh.vertices.len(), mesh.normals.len());
        assert!(mesh.lightweight);
        assert_bounds(&mesh, [0.0, 0.0, 0.0], [10.0, 4.0, 20.0]);
    }

    #[test]
    fn display_mesh_of_an_empty_brep_is_an_error() {
        let error = display_mesh(&Brep::new()).expect_err("empty B-rep has no triangles");
        assert!(error.contains("produced no triangles"), "{error}");
    }

    #[test]
    fn invalid_box_is_not_reported_as_an_unported_dependency() {
        let error = box_brep(0.0, 1.0, 1.0).expect_err("zero length is invalid");
        assert!(!is_unported_error(&error), "{error}");
        assert!(error.contains("create_box"), "{error}");
    }
}
