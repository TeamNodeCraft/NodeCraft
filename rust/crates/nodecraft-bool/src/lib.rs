//! Safe B-rep wrappers over the solidcore Boolean and Shell APIs.
//!
//! NodeCraft's protocol exchanges display [`Mesh`]es, but the solidcore Boolean and
//! offset/shell entry points are B-rep-level: they consume and return
//! `solidcore::topology::Brep` values. This crate keeps that boundary explicit:
//!
//! * [`BooleanOp`] is the protocol-facing operation selector.
//! * [`boolean_breps`] wraps [`solidcore::boolean::api_brep::boolean`] (union,
//!   difference, intersection) and [`shell_brep`] / [`shell_brep_with_options`] wrap
//!   [`solidcore::offset::offset_executive::p2`]'s `OffsetExecutive::shell_brep`
//!   (hollowing with removed "opening" faces).
//! * [`display_mesh`] bridges any B-rep to a protocol [`Mesh`] through
//!   [`nodecraft_solidcore::curved::brep_display_mesh`], and [`boolean_mesh`] /
//!   [`shell_mesh`] combine an operation with that bridge.
//! * [`box_brep`] / [`box_brep_at`] construct solidcore box operands in B-rep
//!   coordinates; [`brep_faces`] lists the face ids a shell operation can open.
//!
//! The wrappers take and return plain `Brep`s rather than `nodecraft_protocol::Mesh`:
//! a display mesh cannot be converted back into exact B-rep topology, so a Boolean or
//! shell operation has to start from B-rep handles. The Mesh-facing functions exist
//! for the last step only.
//!
//! # Current solidcore coverage (pinned `Cargo.lock` revision)
//!
//! * **Boolean:** union, difference and intersection run end to end for the box
//!   operands exercised by the tests (disjoint, nested/containment, touching and
//!   overlapping pairs), including the classification adapters that used to stop at
//!   `SmEdge::GetTolerance` and the face-owned trimmed surface cache. One
//!   representation gap remains: a disjoint intersection returns an empty result whose
//!   B-rep has faces but no shell (`brep_is_manifold_solid` false, no display mesh), so
//!   callers must not treat it as a valid empty solid.
//! * **Shell:** rounded and extended `shell_brep` calls complete through
//!   `SmOffsetExecutive::DoSolidOffset` / `do_extended_offset` and return the offset
//!   shell B-rep (an offset surface shell, not a hollow solid, unless
//!   `create_offset_solid` is requested). Creating an offset solid relative to a
//!   manifold input still hits the native proto-topology edge-imprint guard
//!   (`SM_ERR_NULL_POINTER`, status 1006).
//! * The merging `PiecewiseMerge` inside the offset executive requires the caller's
//!   [`solidcore::sweep::topology_sweep::store::SweepIdentities`]
//!   (`OffsetNotifications::merge_identities`); the shell notification fixture supplies them
//!   in reserved namespaces.
//!
//! Errors are mapped to a [`String`] whose [`UNPORTED_PREFIX`] marks a missing
//! solidcore dependency rather than bad caller input; use [`is_unported_error`].

#![forbid(unsafe_code)]


use solidcore::boolean::api_brep as boolean_api;
use solidcore::error::{EntityRef, SolidError};
use solidcore::geom::a_object::BooleanOperation;
use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::geom::object::ObjectRef;
use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::offset::offset_executive::p1::OffsetExecutive;
use solidcore::offset::offset_executive::p2::{OffsetNotifications, OffsetOptions, ShellOptions};
use solidcore::curve::curve::Curve;
use solidcore::primitive::api_primitives as api;
use solidcore::sweep::topology_sweep::store::SweepIdentities;
use solidcore::topology::topology::{NotifyRecords, Topology};
use solidcore::topology::{Brep, FaceId};
use nodecraft_protocol::Mesh;
use nodecraft_solidcore::curved::{CurvedTessOptions, brep_display_mesh};

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
    let mut kernel = Kernel::default();
    api::create_box(
        &mut kernel,
        Point3::new(origin[0], origin[1], origin[2]),
        length,
        width,
        height,
    )
    .map_err(|error| map_error("solidcore create_box", &error))
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
/// The ids name faces in `brep` and are what [`shell_brep`] expects for its `openings`.
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
    let mut current = a.to_vec();
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
    let mut kernel = Kernel::default();
    boolean_breps_with_kernel(&mut kernel, op, a, b)
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
/// Uses the nodecraft-solidcore display bridge
/// ([`nodecraft_solidcore::curved::brep_display_mesh`]) with its default grid density.
/// The mesh is a display approximation: seam vertices may be duplicated and the index
/// buffer is not welded. The result is not re-centered.
///
/// # Errors
///
/// A message when the bridge cannot tessellate a face or the B-rep has no faces.
pub fn display_mesh(brep: &Brep) -> Result<Mesh, String> {
    display_mesh_with_options(brep, &CurvedTessOptions::default())
}

/// [`display_mesh`] with an explicit display-grid density.
///
/// # Errors
///
/// A message when the bridge cannot tessellate a face or the B-rep has no faces.
pub fn display_mesh_with_options(brep: &Brep, options: &CurvedTessOptions) -> Result<Mesh, String> {
    let kernel = Kernel::default();
    brep_display_mesh(&kernel, brep, options)
        .map_err(|error| format!("nodecraft_solidcore curved display mesh: {error}"))
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
    let mut kernel = Kernel::default();
    let brep = boolean_with_kernel(&mut kernel, op, a, b)?;
    brep_display_mesh(&kernel, &brep, &CurvedTessOptions::default())
        .map_err(|error| format!("nodecraft_solidcore curved display mesh: {error}"))
}

// ---------------------------------------------------------------------------------
// Shell (offset) wrapper
// ---------------------------------------------------------------------------------

/// Reserved identity namespace for source (input) topology in the notification stores.
const ORIGINAL_NAMESPACE: u64 = 100_000_000_000;
/// Reserved identity namespace for result topology in the notification stores.
const RESULT_NAMESPACE: u64 = 200_000_000_000;
/// Reserved identity namespace for result edge-curve identities.
const RESULT_CURVE_NAMESPACE: u64 = 300_000_000_000;
/// Reserved identity namespace for source edge-curve identities.
const ORIGINAL_CURVE_NAMESPACE: u64 = 400_000_000_000;
/// Reserved identity namespace for Boolean source-list handles.
const SOURCE_LIST_NAMESPACE: u64 = 500_000_000_000;
/// Reserved identity namespace for the B-reps a merging offset `PiecewiseMerge`
/// creates (see `OffsetNotifications::merge_identities`). The merge namespaces sit
/// far above the source/result stores so an entity kind (up to `16_022 * 1_000_000`)
/// and a body index never alias another store's key.
const MERGE_TOPOLOGY_NAMESPACE: u64 = 1_000_000_000_000_000_000;
/// Reserved identity namespace for merge-body edge curves.
const MERGE_CURVE_NAMESPACE: u64 = 2_000_000_000_000_000_000;
/// Reserved identity namespace for merge-body Boolean source-list handles.
const MERGE_SOURCE_LIST_NAMESPACE: u64 = 3_000_000_000_000_000_000;
/// Per-merge-body stride inside the merge identity namespaces.
const MERGE_BREP_STRIDE: u64 = 1_000_000_000_000;
/// Per-object-kind stride inside a namespace (`type_code * ENTITY_STRIDE + index`).
const ENTITY_STRIDE: u64 = 1_000_000;

/// `SmFaceuse_TYPE` (solidcore's `ShellType::Faceuse` code; no public constant).
const FACEUSE_TYPE: i64 = 16022;

/// The upstream `Sm*_TYPE` code of an entity, used to keep notification identities
/// distinct between object kinds inside one namespace.
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

fn notification_ref(entity: EntityRef, namespace: u64) -> ObjectRef {
    // Every `Sm*_TYPE` code is a small positive constant (12060..=16022 here), so the
    // signed-to-unsigned conversion is lossless.
    let kind = u64::try_from(type_code_of(entity)).unwrap_or_default();
    ObjectRef {
        type_code: type_code_of(entity),
        key: ObjectKey(namespace + kind * ENTITY_STRIDE + index_of(entity)),
    }
}

fn original_key_of(entity: EntityRef) -> ObjectRef {
    notification_ref(entity, ORIGINAL_NAMESPACE)
}

fn result_key_of(entity: EntityRef) -> ObjectRef {
    notification_ref(entity, RESULT_NAMESPACE)
}

fn curve_ref(edge: solidcore::EdgeId, curve_type: i64, namespace: u64) -> ObjectRef {
    ObjectRef {
        type_code: curve_type,
        key: ObjectKey(namespace + u64::from(edge.index())),
    }
}

fn original_curve_ref(edge: solidcore::EdgeId, curve_type: i64) -> ObjectRef {
    curve_ref(edge, curve_type, ORIGINAL_CURVE_NAMESPACE)
}

fn result_curve_ref(edge: solidcore::EdgeId, curve_type: i64) -> ObjectRef {
    curve_ref(edge, curve_type, RESULT_CURVE_NAMESPACE)
}

fn boolean_source_lists(result_arena: bool, region: solidcore::topology::RegionId) -> [ObjectRef; 2] {
    let base = if result_arena {
        SOURCE_LIST_NAMESPACE + 10_000_000_000
    } else {
        SOURCE_LIST_NAMESPACE
    };
    [0, 1].map(|list| ObjectRef {
        key: ObjectKey(base + u64::from(region.index()) * 2 + list),
        type_code: 0,
    })
}

fn merge_topology_ref(brep: solidcore::BrepId, entity: EntityRef) -> ObjectRef {
    let kind = u64::try_from(type_code_of(entity)).unwrap_or_default();
    ObjectRef {
        type_code: type_code_of(entity),
        key: ObjectKey(
            MERGE_TOPOLOGY_NAMESPACE
                + u64::from(brep.index()) * MERGE_BREP_STRIDE
                + kind * ENTITY_STRIDE
                + index_of(entity),
        ),
    }
}

fn merge_curve_ref(brep: solidcore::BrepId, edge: solidcore::EdgeId, type_code: i64) -> ObjectRef {
    ObjectRef {
        type_code,
        key: ObjectKey(
            MERGE_CURVE_NAMESPACE
                + u64::from(brep.index()) * MERGE_BREP_STRIDE
                + u64::from(edge.index()),
        ),
    }
}

/// Identities of the B-reps a merging offset `PiecewiseMerge` creates (its merge
/// B-rep and stand-alone face B-reps), in namespaces disjoint from the source/result
/// stores above.
struct MergeIdentities;

impl SweepIdentities for MergeIdentities {
    fn topology(&self, brep: solidcore::BrepId, entity: EntityRef) -> ObjectRef {
        merge_topology_ref(brep, entity)
    }

    fn owned_auxiliary_objects(&self, brep: solidcore::BrepId, model: &Brep) -> Vec<ObjectKey> {
        // Every edge curve of the merge body is exclusively owned by it; the offset
        // temporary (handle 0) keeps the caller's result identities instead.
        model
            .edges()
            .iter()
            .filter(|(_, edge)| edge.curve.is_some())
            .map(|(edge, _)| merge_curve_ref(brep, edge, 0).key)
            .collect()
    }

    fn curve(&self, brep: solidcore::BrepId, edge: solidcore::EdgeId, curve: &Curve) -> ObjectRef {
        merge_curve_ref(brep, edge, curve.type_code())
    }

    fn new_curve(
        &self,
        brep: solidcore::BrepId,
        edge: solidcore::EdgeId,
        type_code: i64,
    ) -> Result<ObjectRef, SolidError> {
        Ok(merge_curve_ref(brep, edge, type_code))
    }

    fn source_lists(
        &self,
        brep: solidcore::BrepId,
        region: solidcore::topology::RegionId,
    ) -> Result<[ObjectRef; 2], SolidError> {
        Ok([0, 1].map(|list| ObjectRef {
            key: ObjectKey(
                MERGE_SOURCE_LIST_NAMESPACE
                    + u64::from(brep.index()) * MERGE_BREP_STRIDE
                    + u64::from(region.index()) * 2
                    + list,
            ),
            type_code: 0,
        }))
    }
}

/// Owns the notification stores the offset executive writes through.
///
/// Solidcore's shell entry point takes a caller-owned notification capability
/// (`OffsetNotifications`) instead of reaching for global state. This fixture is the
/// minimal implementation of that contract for a standalone call: every source entity
/// has a registered record, result intersections get disjoint identities, the result
/// B-rep key is reserved, and no attributes are attached. Real callers that need
/// parametric re-execution (tags/attributes carried across a rebuild) will have to
/// pass their own stable keys; see `NOTES.md`.
struct ShellNotifications {
    records: NotifyRecords,
    attributes: AttributeStore,
    result_brep: ObjectRef,
    merge_identities: MergeIdentities,
}

impl ShellNotifications {
    fn new(original: &Brep) -> Self {
        let mut records = NotifyRecords::new();
        records.insert(
            original_key_of(EntityRef::Brep(solidcore::BrepId::from_index(0))).key,
            Topology::new(),
        );
        let entities = original
            .shells()
            .iter()
            .map(|(id, _)| EntityRef::Shell(id))
            .chain(original.faces().iter().map(|(id, _)| EntityRef::Face(id)))
            .chain(original.faceuses().iter().map(|(id, _)| EntityRef::Faceuse(id)))
            .chain(original.edges().iter().map(|(id, _)| EntityRef::Edge(id)))
            .chain(original.edgeuses().iter().map(|(id, _)| EntityRef::Edgeuse(id)))
            .chain(original.vertices().iter().map(|(id, _)| EntityRef::Vertex(id)))
            .chain(original.vertexuses().iter().map(|(id, _)| EntityRef::Vertexuse(id)))
            .chain(original.loopuses().iter().map(|(id, _)| EntityRef::Loopuse(id)))
            .chain(original.regions().iter().map(|(id, _)| EntityRef::Region(id)));
        for entity in entities {
            records.insert(original_key_of(entity).key, Topology::new());
        }
        let result_brep = result_key_of(EntityRef::Brep(solidcore::BrepId::from_index(0)));
        records.insert(result_brep.key, Topology::new());
        Self {
            records,
            attributes: AttributeStore::default(),
            result_brep,
            merge_identities: MergeIdentities,
        }
    }

    fn context(&mut self) -> OffsetNotifications<'_> {
        OffsetNotifications {
            records: &mut self.records,
            attributes: &mut self.attributes,
            original_key_of: &original_key_of,
            result_key_of: &result_key_of,
            original_curve_ref: &original_curve_ref,
            result_curve_ref: &result_curve_ref,
            boolean_source_lists: &boolean_source_lists,
            result_brep: self.result_brep,
            merge_identities: Some(&self.merge_identities),
        }
    }
}

/// Hollows `brep` by `thickness`, treating `openings` as faces to remove (open the
/// shell), and returns the resulting B-rep.
///
/// This wraps `OffsetExecutive::shell_brep` (`SmOffsetExecutive::ShellBrep`) with the
/// default `ShellOptions`: the signed offset distance is forwarded unchanged (negative
/// insets, the usual hollowing direction), no lamina side walls are added, and the
/// rounded (non-extended) offset path is used. The caller owns the returned B-rep; the
/// input is consumed.
///
/// **Current coverage:** the rounded offset shell completes in solidcore without
/// side walls. Closed offset-solid side walls still report an upstream construction
/// guard; the returned B-rep is an offset shell, not a hollow solid.
///
/// # Errors
///
/// A message from solidcore. An unported offset/Boolean dependency is reported with
/// [`UNPORTED_PREFIX`]; see [`is_unported_error`]. A distance smaller than twice the
/// B-rep tolerance is rejected by solidcore as invalid input.
pub fn shell_brep(brep: Brep, thickness: f64, openings: &[FaceId]) -> Result<Brep, String> {
    shell_brep_with_options(
        brep,
        ShellOptions {
            offset: OffsetOptions {
                distance: thickness,
                ..OffsetOptions::default()
            },
            create_offset_solid: false,
            merge_sheet_side_walls: false,
        },
        openings,
    )
}

/// [`shell_brep`] with explicit `ShellOptions`, mainly to reach the extended-offset
/// path (`extend_convex_edges`) and the offset-solid side walls
/// (`create_offset_solid`).
///
/// **Current coverage:** rounded and extended offset shells complete. Since the
/// upstream shell construction fixes, `create_offset_solid` produces a manifold
/// hollow solid rather than an open offset sheet.
///
/// # Errors
///
/// A message from solidcore; unported dependencies carry [`UNPORTED_PREFIX`].
/// Shell with sharp offset walls rather than rounded convex corners.
pub fn shell_brep_solid(brep: Brep, thickness: f64, openings: &[FaceId]) -> Result<Brep, String> {
    shell_brep_with_options(
        brep,
        ShellOptions {
            offset: OffsetOptions {
                distance: thickness,
                extend_convex_edges: true,
                merge_results: true,
            },
            create_offset_solid: true,
            merge_sheet_side_walls: false,
        },
        openings,
    )
}

pub fn shell_brep_solid_with_kernel(
    kernel: &mut Kernel,
    brep: Brep,
    thickness: f64,
    openings: &[FaceId],
) -> Result<Brep, String> {
    shell_brep_with_options_with_kernel(
        kernel,
        brep,
        ShellOptions {
            offset: OffsetOptions {
                distance: thickness,
                extend_convex_edges: true,
                merge_results: true,
            },
            create_offset_solid: true,
            merge_sheet_side_walls: false,
        },
        openings,
    )
}

pub fn shell_brep_with_options(
    brep: Brep,
    options: ShellOptions,
    openings: &[FaceId],
) -> Result<Brep, String> {
    let mut kernel = Kernel::default();
    shell_brep_with_options_with_kernel(&mut kernel, brep, options, openings)
}

/// [`shell_brep_with_options`] on a caller-owned (worker) kernel.
pub fn shell_brep_with_options_with_kernel(
    kernel: &mut Kernel,
    brep: Brep,
    options: ShellOptions,
    openings: &[FaceId],
) -> Result<Brep, String> {
    let mut notifications = ShellNotifications::new(&brep);
    let outcome = OffsetExecutive::shell_brep(
        kernel,
        brep,
        options,
        openings,
        &mut notifications.context(),
    );
    outcome
        .result
        .map_err(|error| map_error("solidcore shell_brep", &error))
}

/// [`shell_brep`] followed by [`display_mesh`].
///
/// # Errors
///
/// The shell error, or a display-tessellation error.
pub fn shell_mesh(brep: Brep, thickness: f64, openings: &[FaceId]) -> Result<Mesh, String> {
    let brep = shell_brep(brep, thickness, openings)?;
    display_mesh(&brep)
}

#[cfg(test)]
mod tests {
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
        let mut copy = brep.clone();
        let store = AttributeStore::default();
        let attributes = solidcore::mass::integrator::properties::PropertyAttributes {
            store: &store,
            key_of: &|entity| original_key_of(entity).key,
        };
        solidcore::mass::integrator::properties::brep_volume(
            &Kernel::default(),
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
        assert_eq!(
            BooleanOp::Intersection.to_solidcore(),
            BooleanOperation::Intersection
        );
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
        assert!(
            solidcore::query::api_queries::brep_is_manifold_solid(&union).expect("manifold")
        );

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
        assert!(!solidcore::query::api_queries::brep_is_manifold_solid(&intersection)
            .expect("manifold query"));
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
        let difference = boolean_breps(BooleanOp::Difference, a, b).expect("nested difference call");
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
            assert!((volume(&result).expect("touching volume") - expected_volume).abs() < 1e-6, "{op}");
        }
    }

    /// Evidence that `SmMerge` construction succeeds on the current solidcore. The
    /// manifold stage may now return a defined result instead of an unported error;
    /// either outcome is accepted, but a panic is not.
    #[test]
    fn boolean_construction_succeeds_before_the_manifold_intersection_fails() {
        use solidcore::boolean::merge::p3::{BooleanDriver, DependencyMerge};
        let mut kernel = Kernel::default();
        let mut primary = Some(box_brep(10.0, 10.0, 10.0).expect("box A"));
        let mut other = Some(box_brep(10.0, 10.0, 10.0).expect("box B"));
        let mut driver = DependencyMerge::default();
        driver
            .construct(
                &mut kernel,
                &mut primary,
                &mut other,
                0.0,
                0.0,
                None,
            )
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

    #[test]
    fn shell_solid_is_a_closed_manifold_with_openings() {
        for openings in [0usize, 1] {
            let brep = box_brep(10.0, 10.0, 10.0).expect("box");
            let faces = brep_faces(&brep).expect("faces");
            let shell = shell_brep_solid(brep, -1.0, &faces[..openings]).expect("shell solid");
            assert!(
                solidcore::query::api_queries::brep_is_manifold_solid(&shell).expect("manifold query"),
                "{openings}: shell is not a closed manifold solid"
            );
            assert!(!shell.faces().is_empty(), "{openings}: shell has no faces");
        }
    }

    /// Rounded shelling a real box now completes in solidcore.
    #[test]
    fn shell_box_produces_an_offset_shell() {
        for openings in [0usize, 1] {
            let brep = box_brep(10.0, 10.0, 10.0).expect("box");
            let faces = brep_faces(&brep).expect("faces");
            let shell = shell_brep(brep, -1.0, &faces[..openings]).expect("shell is ported");
            assert!(!shell.faces().is_empty(), "{openings}: shell has no faces");
        }
    }

    /// Stale-claim check: `topology::brep::p4::shrink_geometry` (the symbol the shell
    /// stop marker names) is implemented and does nothing harmful to a valid box.
    #[test]
    fn shrink_geometry_symbol_is_ported_despite_the_shell_marker() {
        let mut brep = box_brep(10.0, 10.0, 10.0).expect("box");
        solidcore::topology::brep::p4::shrink_geometry(&Kernel::default(), &mut brep)
            .expect("SmBrep::ShrinkGeometry is ported");
        assert_eq!(brep.faces().len(), 6);
    }

    /// A closed offset solid still hits a native construction guard: the shell
    /// difference reaches proto-topology edge imprint, which reports
    /// `SM_ERR_NULL_POINTER` (1006) when a proto vertex has no final vertex for the
    /// definition. This is kernel behavior, not an unported dependency.
    #[test]
    fn shell_with_side_walls_reports_upstream_guard() {
        let brep = box_brep(10.0, 10.0, 10.0).expect("box");
        let error = shell_brep_with_options(
            brep,
            ShellOptions {
                offset: OffsetOptions { distance: -1.0, ..OffsetOptions::default() },
                create_offset_solid: true,
                merge_sheet_side_walls: false,
            },
            &[],
        )
        .expect_err("offset-solid construction is not usable");
        assert!(!is_unported_error(&error), "{error}");
        // Upstream now reaches the Boolean merge before the old imprint guard.
        assert!(error.contains("1001") || error.contains("1006"), "{error}");
    }

    /// Extended shell options now complete through offset geometry creation.
    #[test]
    fn extended_shell_completes() {
        let brep = box_brep(10.0, 10.0, 10.0).expect("box");
        let shell = shell_brep_with_options(
            brep,
            ShellOptions {
                offset: OffsetOptions {
                    distance: -1.0,
                    extend_convex_edges: true,
                    merge_results: false,
                },
                create_offset_solid: false,
                merge_sheet_side_walls: false,
            },
            &[],
        )
        .expect("extended shell is ported");
        assert!(!shell.faces().is_empty(), "extended shell has no faces");
    }

    #[test]
    fn shell_zero_thickness_is_invalid_input_not_an_unported_dependency() {
        let brep = box_brep(10.0, 10.0, 10.0).expect("box");
        let error = shell_brep(brep, 0.0, &[]).expect_err("zero shell distance is invalid");
        assert!(!is_unported_error(&error), "{error}");
        assert!(error.contains("1007"), "{error}");
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
            &mut Kernel::default(),
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

