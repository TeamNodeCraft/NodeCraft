//! A solid body: NodeCraft's immutable record of a solid, with the proved kernel body beside it.
//!
//! The record is an `Arc<Brep>`: display, tessellation and picking share one allocation. The
//! kernel body (`solidcore_ops::Body`, which holds the proof that the B-rep is a valid outward
//! solid) is built from that record on first use and shared by clones through an
//! `Arc<OnceLock<..>>`; a result made by an operation stores the body it made, with its proof,
//! directly. Volume and the exact box are memos, shared by clones.
//!
//! A [`SolidBody`] is never mutated in place. Operations take one by reference and copy the
//! kernel body (`Body::copy_held_on`), which keeps a proof that still holds for the unchanged
//! B-rep under the same kernel settings; the consumer re-proves only when the settings differ.
//!
//! Every body also carries a shared [`ProofState`]: whether a valid-outward-solid proof has
//! been taken for it yet. Clones and every reader of one body share it, so a body is proved
//! at most once. The engine's background proof queue and the blocking callers (export, check,
//! orientation-dependent operations) consult it instead of re-proving a body they have
//! already seen.

use std::sync::{Arc, Mutex, OnceLock};

use solidcore::EntityRef;
use solidcore::kernel::Kernel;
use solidcore::topology::Brep;
use solidcore::topology::brep::p3::brep_copy_mapped;
use solidcore_ops::{Body, Order};

use crate::SessionError;
use crate::error_map::{OpKind, is_undecided_test, map_error, undecided_proof};

/// An exact axis-aligned box: the least corner, then the greatest.
pub type Bounds = ([f64; 3], [f64; 3]);

/// The proof standing for one body: unknown, proved a valid outward solid, refused with the
/// failing test's message, or undecided with the kernel's honest sentence.
///
/// A body starts [`ProofState::Unknown`]. An engine evaluation defers proofs (ops'
/// `Validation::EachNode`), so a body it makes stays unknown until the background proof queue
/// or a blocking caller proves it. A result made under `Validation::EachStep` is already
/// proved and starts [`ProofState::Proved`]. `Refused` and `Undecided` are kept: the same body
/// is not proved again after the kernel has answered. `Undecided` is not a refusal: no test
/// showed the body is not a solid, but the kernel could not prove that it is, so an operation
/// that needs a proved operand cannot use it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofState {
    Unknown,
    Proved,
    Refused(String),
    Undecided(String),
}

/// One solid body: the shared B-rep, the shared (proved) kernel body, their memos and the
/// shared proof state.
///
/// Cheap to clone; never mutated in place. A sheet is allowed (it simply never holds a proof);
/// an edges-only B-rep is allowed too, and [`SolidBody::new`] does not touch it until an
/// operation first needs the kernel body.
#[derive(Clone, Debug)]
pub struct SolidBody {
    brep: Arc<Brep>,
    /// The record as ops' body, made on first use (or kept from the operation that made it,
    /// with its proof), shared by clones because they share the record. It is the operand every
    /// operation copies: it keeps the exact blend definitions of rounds, which the display
    /// record `brep` may not (see [`SolidBody::from_ops`]). The [`Order`] names, for each face,
    /// edge and vertex of `brep` in list order, the body's own entity.
    ops: Arc<OnceLock<(Body, Order)>>,
    /// The last volume measured, shared by clones because they share the record.
    volume: Arc<Mutex<Option<f64>>>,
    /// The exact box, measured on first use and shared by clones because they share the record.
    bounds: Arc<Mutex<Option<Bounds>>>,
    /// The shared proof state: one body, one proof, however many readers it has.
    /// `None` is [`ProofState::Unknown`]; a proof is written at most once, so a
    /// `OnceLock` is the whole state (a proof or refusal is never replaced).
    proof: Arc<OnceLock<ProofState>>,
    /// How many proof attempts this body has made. Normally at most one: the state above is
    /// the memo, and a test can see that a second caller did not touch the kernel.
    attempts: Arc<std::sync::atomic::AtomicUsize>,
}

impl SolidBody {
    /// A solid body of `brep`, with no kernel body or proof yet.
    #[must_use]
    pub fn new(brep: Brep) -> SolidBody {
        SolidBody::from_arc(Arc::new(brep))
    }

    /// A solid body sharing `brep` (one display body, many placed instances).
    #[must_use]
    pub fn from_arc(brep: Arc<Brep>) -> SolidBody {
        SolidBody {
            brep,
            ops: Arc::new(OnceLock::new()),
            volume: Arc::new(Mutex::new(None)),
            bounds: Arc::new(Mutex::new(None)),
            proof: Arc::new(OnceLock::new()),
            attempts: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// A result an ops operation made: its B-rep is copied for the shared display record, and
    /// the body keeps the proof ops made on it.
    ///
    /// Under `Validation::EachStep` the result is proved by ops itself and starts
    /// [`ProofState::Proved`] when it is a closed solid (a sheet result never holds a proof).
    /// Under `Validation::EachNode` the proof is deferred to the background queue, so the
    /// result starts [`ProofState::Unknown`].
    ///
    /// # Errors
    /// Those of [`Body::to_brep`].
    pub fn from_ops(body: Body) -> Result<SolidBody, SessionError> {
        // `to_brep` is a display copy (same ids and order) whose surfaces are evaluation copies
        // without blend definitions; the body itself, kept below, is the operand and keeps them.
        let brep = body.to_brep().map_err(|error| map_error(OpKind::Body, error))?;
        let order = Order::of(&brep);
        let solid = SolidBody::from_arc(Arc::new(brep));
        let proved = crate::validation() == crate::Validation::EachStep
            && solidcore::topology::brep::p2::is_manifold_solid(solid.brep()).unwrap_or(false);
        if proved {
            solid.write_proof(ProofState::Proved);
        }
        let _ = solid.ops.set((body, order));
        Ok(solid)
    }

    /// This body's shared proof state.
    #[must_use]
    pub fn proof_state(&self) -> ProofState {
        self.proof.get().cloned().unwrap_or(ProofState::Unknown)
    }

    /// Whether the body is already proved a valid outward solid.
    #[must_use]
    pub fn is_proved(&self) -> bool {
        matches!(self.proof_state(), ProofState::Proved)
    }

    /// Whether no proof or refusal has been decided for this body yet. This is
    /// the same shared state read without cloning it, for a queue deciding
    /// whether a body needs a job.
    #[must_use]
    pub fn needs_proof(&self) -> bool {
        self.proof.get().is_none()
    }

    /// Whether the body was refused, and the test the proof named.
    #[must_use]
    pub fn refused(&self) -> Option<String> {
        match self.proof_state() {
            ProofState::Refused(reason) => Some(reason),
            ProofState::Unknown | ProofState::Proved | ProofState::Undecided(_) => None,
        }
    }

    /// Whether the kernel could not decide the body's proof, and the honest sentence for it.
    #[must_use]
    pub fn undecided(&self) -> Option<String> {
        match self.proof_state() {
            ProofState::Undecided(reason) => Some(reason),
            ProofState::Unknown | ProofState::Proved | ProofState::Refused(_) => None,
        }
    }

    /// How many times a proof was actually attempted for this body (not how many callers
    /// asked): a body already proved or refused makes no new attempt.
    #[must_use]
    pub fn proof_attempts(&self) -> usize {
        self.attempts.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A stable identity for the shared proof state, so a queue can prove one body once even
    /// when several parts of a graph hold clones of it.
    #[must_use]
    pub fn proof_id(&self) -> usize {
        Arc::as_ptr(&self.proof) as usize
    }

    fn write_proof(&self, state: ProofState) {
        // A proof is written once; a later caller that saw Unknown may race, but
        // both write the same decision, and the first decision is kept.
        let _ = self.proof.set(state);
    }

    /// Give this body `state` for a test of the memo, without running the kernel. A real
    /// undecided proof needs a body the kernel cannot decide, which no cheap test body gives.
    #[cfg(test)]
    pub(crate) fn write_proof_for_test(&self, state: ProofState) {
        self.write_proof(state);
    }

    /// Give this body `source`'s proof state when it is still unknown: a proof copy (a rigid
    /// transform of the same B-rep, or a deep copy of it) carries the source's proof. A refusal
    /// is never overwritten, and a proved proof is never dropped.
    pub fn carry_proof_from(&self, source: &SolidBody) {
        if !matches!(source.proof_state(), ProofState::Proved) {
            return;
        }
        let _ = self.proof.set(ProofState::Proved);
    }

    /// The shared B-rep: the one accessor for display, tessellation and picking.
    #[must_use]
    pub fn brep(&self) -> &Brep {
        &self.brep
    }

    /// The shared allocation itself, for a consumer that keeps an `Arc<Brep>` (the display
    /// plan's `ExactBody`).
    #[must_use]
    pub fn shared(&self) -> Arc<Brep> {
        Arc::clone(&self.brep)
    }

    /// The B-rep as an owning value, for a consumer that mutates it. The shared record is
    /// taken when this is its last owner, so a body no one else holds costs no copy; the
    /// kernel body (and its proof) is dropped, as an edit would drop the proof anyway.
    ///
    /// # Errors
    /// Those of the kernel's copy, for a record another owner still holds.
    pub fn into_brep(self) -> Result<Brep, SessionError> {
        match Arc::try_unwrap(self.brep) {
            Ok(brep) => Ok(brep),
            Err(shared) => shared
                .clone_standing_for_self()
                .map_err(|error| SessionError::Kernel(error.to_string())),
        }
    }

    /// The kernel body and its [`Order`], built once from this record and shared by every
    /// later caller.
    ///
    /// The body is the kernel's own copy of the record (`brep_copy_mapped`), which keeps each
    /// surface's exact blend definition; its ids are new, so the order maps the record's lists
    /// to them through the copy's map. Not `clone_standing_for_self`: that copy's surfaces are
    /// evaluation copies, which carry no definition, so every earlier round would reach the next
    /// operation as a generic NURBS face (and a later blend along it is certified from scratch).
    fn kernel_body(&self, kernel: &mut Kernel) -> Result<&(Body, Order), SessionError> {
        if let Some(held) = self.ops.get() {
            return Ok(held);
        }
        let (copy, map) = brep_copy_mapped(kernel, &self.brep)
            .map_err(|error| SessionError::Kernel(error.to_string()))?;
        let record = Order::of(&self.brep);
        let mapped = |entity: EntityRef| map.get_at(Some(entity)).ok().flatten();
        let missing = || SessionError::Kernel("the kernel's copy lost an entity".to_owned());
        let order = Order {
            faces: record
                .faces
                .iter()
                .map(|&f| match mapped(EntityRef::Face(f)) {
                    Some(EntityRef::Face(copy)) => Ok(copy),
                    _ => Err(missing()),
                })
                .collect::<Result<_, _>>()?,
            edges: record
                .edges
                .iter()
                .map(|&e| match mapped(EntityRef::Edge(e)) {
                    Some(EntityRef::Edge(copy)) => Ok(copy),
                    _ => Err(missing()),
                })
                .collect::<Result<_, _>>()?,
            vertices: record
                .vertices
                .iter()
                .map(|&v| match mapped(EntityRef::Vertex(v)) {
                    Some(EntityRef::Vertex(copy)) => Ok(copy),
                    _ => Err(missing()),
                })
                .collect::<Result<_, _>>()?,
            // the copy's own lists, mapped one to one: no edge is split
            split_edges: record.split_edges,
        };
        let body = Body::from_brep(kernel, copy).map_err(|error| map_error(OpKind::Body, error))?;
        Ok(self.ops.get_or_init(|| (body, order)))
    }

    /// A kernel body for one operation that consumes it by value: a deep copy that keeps a
    /// proof that still holds. The record is unchanged, so the copy is the same B-rep under the
    /// same settings; the operation's own `Expect::of_held` re-checks the settings.
    ///
    /// # Errors
    /// Those of [`Body::from_brep`] and [`Body::copy_held_on`].
    pub fn ops_copy(&self, kernel: &mut Kernel) -> Result<Body, SessionError> {
        self.kernel_body(kernel)?
            .0
            .copy_held_on(kernel)
            .map_err(|error| map_error(OpKind::Body, error))
    }

    /// [`Self::ops_copy`] with the [`Order`] that names, for each face, edge and vertex of
    /// [`Self::brep`] in list order, the copy's own entity: for an operation that picks entities
    /// by their place in the record (a blend's edges, a shell's open faces).
    ///
    /// # Errors
    /// Those of [`Body::from_brep`] and [`Body::copy_with_order`].
    pub fn ops_copy_ordered(&self, kernel: &mut Kernel) -> Result<(Body, Order), SessionError> {
        let (body, order) = self.kernel_body(kernel)?;
        body.copy_with_order(order).map_err(|error| map_error(OpKind::Body, error))
    }

    /// Prove this body now, under `kernel`'s settings, and keep the proof: the check the
    /// background proof queue and a blocking caller (export, check, an orientation-dependent
    /// operation) run. A body already decided is not proved again; a refused body stays refused
    /// and returns the test it failed, and an undecided body stays undecided and returns the
    /// honest sentence for a proof the kernel could not decide.
    ///
    /// # Errors
    /// [`SessionError::Invalid`] when the body is not a valid outward solid, or when the kernel
    /// could not decide it ([`ProofState::Undecided`]); those of [`Body::from_brep`]. Only an
    /// answer from the solid test is remembered, so a kernel status or a cancellation is retried
    /// later.
    pub fn prove(&self, kernel: &mut Kernel) -> Result<(), SessionError> {
        match self.proof_state() {
            ProofState::Proved => return Ok(()),
            ProofState::Refused(reason) | ProofState::Undecided(reason) => {
                return Err(SessionError::Invalid(reason));
            }
            ProofState::Unknown => {}
        }
        self.attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let proved = self.kernel_body(kernel)?.0.prove_on(kernel);
        match proved {
            Ok(()) => {
                self.write_proof(ProofState::Proved);
                Ok(())
            }
            Err(error) => {
                // A test the kernel could not decide is not a refusal: the body is kept
                // undecided, and no later consumer proves it again.
                if let Some(test) = error.not_solid_test()
                    && is_undecided_test(test)
                {
                    let mapped = undecided_proof(test);
                    if let SessionError::Invalid(reason) = &mapped {
                        self.write_proof(ProofState::Undecided(reason.clone()));
                    }
                    return Err(mapped);
                }
                let mapped = map_error(OpKind::Body, error);
                if let SessionError::Invalid(reason) = &mapped {
                    self.write_proof(ProofState::Refused(reason.clone()));
                }
                Err(mapped)
            }
        }
    }

    /// The body's volume, measured once with ops' exact volume at `1e-8` and shared by clones.
    /// The Boolean path never asks for it.
    ///
    /// # Errors
    /// Those of [`Self::ops_copy`] and [`Body::volume`].
    pub fn volume(&self, kernel: &mut Kernel) -> Result<f64, SessionError> {
        if let Some(volume) = *self.volume.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Ok(volume);
        }
        let volume = self
            .ops_copy(kernel)?
            .volume(1.0e-8)
            .map_err(|error| map_error(OpKind::Body, error))?;
        *self.volume.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(volume);
        Ok(volume)
    }

    /// The exact axis-aligned box (`SmBrep::CalculateTightBoundingBox`), measured once and
    /// shared by clones. The decision a boolean's "apart" early-out makes uses this, never a
    /// mesh box.
    ///
    /// # Errors
    /// Those of [`Self::ops_copy`] and [`Body::bounds`].
    pub fn bounds(&self, kernel: &mut Kernel) -> Result<Bounds, SessionError> {
        if let Some(bounds) = *self.bounds.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            return Ok(bounds);
        }
        let bounds =
            self.ops_copy(kernel)?.bounds().map_err(|error| map_error(OpKind::Body, error))?;
        *self.bounds.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(bounds);
        Ok(bounds)
    }
}

impl From<Brep> for SolidBody {
    fn from(brep: Brep) -> SolidBody {
        SolidBody::new(brep)
    }
}
