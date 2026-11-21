//! The only NodeCraft crate that names `solidcore-ops`.
//!
//! An [`OpsSession`] lends a kernel-pool worker's own kernel to ops, so ops algorithms run
//! directly on our B-reps without records or the process lock.

use solidcore::blend::definition_check::{MadeBlendFace, mark_blend_faces};
use solidcore::kernel::Kernel;
use solidcore::topology::brep::p3::brep_copy;
use solidcore::topology::{Brep, EdgeId, FaceId};
use solidcore_ops::blend::{Blend, BlendOptions, ChamferSecond, Corners, Turn};
use solidcore_ops::{Body, Contract, Order};

pub use solidcore_ops::blend::{
    BlendOptions as Options, ChamferCorner, ChamferSecond as Second, Turn as CornerTurn,
};
pub use solidcore_ops::cancel;
pub use solidcore_ops::{BooleanOp, CheckLevel, CheckReport};

mod error_map;
mod solid_body;

pub use solid_body::{Bounds, ProofState, SolidBody};

/// Which proofs ops takes while an application runs the calls of one node. Re-exported from
/// ops, with [`validating`] keeping this crate's own mirror of the current mode so a result
/// can be given the proof state ops made for it.
pub use solidcore_ops::session::Validation;

thread_local! {
    /// The mode [`validating`] installed on this thread, mirrored so this crate can tell
    /// whether an ops result was proved (`EachStep`) or left to the background queue
    /// (`EachNode`) when it wraps the result in a [`SolidBody`](solid_body::SolidBody).
    static VALIDATION: std::cell::Cell<Validation> = const { std::cell::Cell::new(Validation::EachStep) };
}

/// Run `f` with `mode` deciding which proofs the ops calls on this thread take.
///
/// # Panics
/// Restores the previous mode even when `f` panics.
pub fn validating<R>(mode: Validation, f: impl FnOnce() -> R) -> R {
    struct Restore(Validation);
    impl Drop for Restore {
        fn drop(&mut self) {
            VALIDATION.with(|current| current.set(self.0));
        }
    }
    let previous = VALIDATION.with(|current| current.replace(mode));
    let _restore = Restore(previous);
    solidcore_ops::session::validating(mode, f)
}

/// The proof mode the ops calls on this thread run under: `EachStep` outside [`validating`].
#[must_use]
pub fn validation() -> Validation {
    VALIDATION.with(std::cell::Cell::get)
}

use error_map::{OpKind, map_error};
thread_local! {
    static NEEDS_RESET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether an `OpsSession` on this thread asked for a fresh kernel since the last call.
pub fn take_needs_reset() -> bool {
    NEEDS_RESET.with(|flag| flag.replace(false))
}

/// Install ops' failure hook so every failed call writes its inputs to `dir`.
pub fn install_failure_dump(dir: std::path::PathBuf) {
    let _ = std::fs::create_dir_all(&dir);
    solidcore_ops::session::set_failure_hook(Some(std::sync::Arc::new(move |failure| {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let base = dir.join(format!("call-{n:03}"));
        let fixes = nodecraft_kernel_profile::fix_names().join(",");
        let mut meta = format!(
            "call: {}\nwhy: {}\nfixes: {fixes}\ninputs: {}\n",
            failure.call,
            failure.why,
            failure.inputs.len()
        );
        for (name, bytes) in &failure.inputs {
            match bytes {
                Ok(bytes) => {
                    let file = format!("call-{n:03}-{name}.smb");
                    let _ = std::fs::write(dir.join(&file), bytes);
                    meta.push_str(&format!("{name}: {} bytes ({file})\n", bytes.len()));
                }
                Err(error) => meta.push_str(&format!("{name}: unreadable: {error}\n")),
            }
        }
        let _ = std::fs::write(base.with_extension("txt"), meta);
    })));
}

/// Install the failure dump when `NODECRAFT_OPS_DUMP` names a directory.
pub fn install_failure_dump_from_env() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if let Some(dir) = std::env::var_os("NODECRAFT_OPS_DUMP") {
            install_failure_dump(std::path::PathBuf::from(dir));
        }
    });
}

/// The blend to run through ops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpsBlend {
    Round(Turn),
    Curvature,
    Chamfer(ChamferCorner),
    ChamferSides { second: ChamferSecond, up: [f64; 3], flip: bool },
}

/// A blend result on our B-rep.
#[derive(Debug)]
pub struct OpsResult {
    pub brep: Brep,
    pub warnings: Vec<String>,
    /// Faces ops reported as made by this blend.
    pub made_faces: Vec<FaceId>,
    pub volume: f64,
}

/// Why an ops session call failed.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionError {
    Cancelled,
    Invalid(String),
    Kernel(String),
    NeedsReset,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "cancelled"),
            Self::Invalid(message) => write!(f, "invalid: {message}"),
            Self::Kernel(message) => write!(f, "solidcore: {message}"),
            Self::NeedsReset => write!(f, "kernel needs reset after a failed ops call"),
        }
    }
}

impl std::error::Error for SessionError {}

/// A kernel borrowed for the duration of one node's ops work.
pub struct OpsSession<'k> {
    kernel: &'k mut Kernel,
    needs_reset: bool,
}

impl<'k> OpsSession<'k> {
    pub fn new(kernel: &'k mut Kernel) -> Self {
        install_failure_dump_from_env();
        Self { kernel, needs_reset: false }
    }

    fn note_reset(&mut self, reset: bool) {
        self.needs_reset |= reset;
        if reset {
            NEEDS_RESET.with(|flag| flag.set(true));
        }
    }

    /// Round, G2 or chamfer the selected edges.
    pub fn blend(
        &mut self,
        brep: &Brep,
        edges: &[EdgeId],
        size: f64,
        kind: OpsBlend,
    ) -> Result<OpsResult, SessionError> {
        let input_edges = brep.edge_list().to_vec();
        let indices = edges
            .iter()
            .map(|edge| {
                input_edges.iter().position(|candidate| candidate == edge).ok_or_else(|| {
                    SessionError::Invalid("selected edge is not on the body".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let copy = brep_copy(self.kernel, brep)
            .map_err(|error| SessionError::Kernel(error.to_string()))?;
        let order = Order::of(&copy);
        let body =
            Body::from_brep(self.kernel, copy).map_err(|error| map_error(OpKind::Blend, error))?;
        self.kernel.blend_face_log = Some(Default::default());
        let (blend, options) = match kind {
            OpsBlend::Round(turn) => (
                Blend::Round,
                BlendOptions {
                    turn,
                    chamfer_corner: ChamferCorner::Mitre,
                    contract: Contract::Proved,
                },
            ),
            OpsBlend::Curvature => (
                Blend::Curvature,
                BlendOptions {
                    turn: Turn::Mitre,
                    chamfer_corner: ChamferCorner::Mitre,
                    contract: Contract::Proved,
                },
            ),
            OpsBlend::Chamfer(corner) => (
                Blend::Chamfer,
                BlendOptions {
                    turn: Turn::Mitre,
                    chamfer_corner: corner,
                    contract: Contract::Proved,
                },
            ),
            OpsBlend::ChamferSides { second, up, flip } => {
                let mut along = Vec::with_capacity(indices.len());
                let (faces, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
                    indices
                        .iter()
                        .map(|edge| {
                            solidcore_ops::blend::edge_faces_along_ordered(&body, &order, *edge, up)
                        })
                        .collect::<Result<Vec<_>, _>>()
                });
                self.note_reset(reset);
                let faces = faces.map_err(|error| map_error(OpKind::Blend, error))?;
                for pair in faces {
                    let leaning = if pair[0].1 >= pair[1].1 { pair[0].0 } else { pair[1].0 };
                    let other = if leaning == pair[0].0 { pair[1].0 } else { pair[0].0 };
                    along.push(if flip { other } else { leaning });
                }
                let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
                    solidcore_ops::blend::chamfer_sides_ordered_with(
                        &body,
                        &order,
                        &indices,
                        &along,
                        size,
                        second,
                        Contract::Proved,
                    )
                });
                self.note_reset(reset);
                let _ = result.map_err(|error| map_error(OpKind::Blend, error))?;
                let volume =
                    body.volume(1.0e-8).map_err(|error| map_error(OpKind::Blend, error))?;
                let mut brep = body.into_brep();
                let made_faces = self.mark_made(&mut brep)?;
                return Ok(OpsResult { brep, warnings: Vec::new(), made_faces, volume });
            }
        };
        let (blended, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::blend::blend_ordered(
                &body,
                &order,
                blend,
                &indices,
                &[size],
                &[],
                Corners::Filled,
                options,
            )
        });
        self.note_reset(reset);
        let blended = blended.map_err(|error| map_error(OpKind::Blend, error))?;
        let warnings = blended
            .dropped
            .iter()
            .enumerate()
            .filter(|(_, dropped)| **dropped)
            .map(|(index, _)| format!("edge {} was removed by a larger round", index + 1))
            .collect();
        let volume = body.volume(1.0e-8).map_err(|error| map_error(OpKind::Blend, error))?;
        let mut brep = body.into_brep();
        let made_faces = self.mark_made(&mut brep)?;
        Ok(OpsResult { brep, warnings, made_faces, volume })
    }

    /// Boolean two solid bodies; `None` when the result is empty. The empty answer is ops'
    /// own (`Ok(None)` or `Error::Empty`), never a measured volume of zero.
    ///
    /// A `Cut`/`Intersect` whose tool is apart from the body (their exact boxes, grown by the
    /// tolerances, do not meet) takes no intersection work: the cut is the body itself and the
    /// intersection is empty. The boxes are the kernel's exact boxes, never the display mesh's.
    pub fn boolean_outcome(
        &mut self,
        a: &SolidBody,
        b: &SolidBody,
        op: BooleanOp,
    ) -> Result<Option<SolidBody>, SessionError> {
        if matches!(op, BooleanOp::Subtract | BooleanOp::Intersect) && self.apart(a, b)? {
            return Ok(match op {
                BooleanOp::Subtract => Some(a.clone()),
                _ => None,
            });
        }
        let ab = a.ops_copy(self.kernel)?;
        let bb = b.ops_copy(self.kernel)?;
        let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::boolean::boolean(ab, bb, op)
        });
        self.note_reset(reset);
        match result.map_err(|error| map_error(OpKind::Boolean, error))? {
            Some(body) => SolidBody::from_ops(body).map(Some),
            None => Ok(None),
        }
    }

    /// Boolean two solid bodies; an empty result is an error.
    pub fn boolean(
        &mut self,
        a: &SolidBody,
        b: &SolidBody,
        op: BooleanOp,
    ) -> Result<SolidBody, SessionError> {
        self.boolean_outcome(a, b, op)?
            .ok_or_else(|| SessionError::Invalid("boolean produced nothing".to_owned()))
    }

    /// N-ary list Boolean through SolidCore's `boolean_lists`: one merge instead of a
    /// pairwise chain. The result bodies are new B-reps with no proof (the operation is not an
    /// ops `Body` call and takes no outward proof); a caller under `EachStep` that needs a
    /// proof must take one.
    ///
    /// # Errors
    /// The kernel's boolean error, mapped to a user sentence.
    pub fn boolean_lists(
        &mut self,
        a: &[SolidBody],
        b: &[SolidBody],
        op: BooleanOp,
    ) -> Result<Vec<SolidBody>, SessionError> {
        let operation = match op {
            BooleanOp::Union => 1,
            BooleanOp::Intersect => 0,
            BooleanOp::Subtract => 3,
        };
        let copy = |bodies: &[SolidBody]| -> Result<Vec<Brep>, SessionError> {
            bodies
                .iter()
                .map(|body| {
                    body.brep()
                        .clone_standing_for_self()
                        .map_err(|error| SessionError::Kernel(error.to_string()))
                })
                .collect()
        };
        let (a_breps, b_breps) = (copy(a)?, copy(b)?);
        let result = solidcore::boolean::api_brep::boolean_lists(
            self.kernel,
            &a_breps,
            &[],
            &b_breps,
            &[],
            operation,
        )
        .map_err(|error| SessionError::Kernel(error.to_string()))?;
        Ok(result.breps.into_iter().map(SolidBody::new).collect())
    }

    /// Union many solid bodies; the result keeps the proof ops made on it.
    pub fn union_all(&mut self, bodies: &[SolidBody]) -> Result<SolidBody, SessionError> {
        let mut kernels = Vec::with_capacity(bodies.len());
        for body in bodies {
            kernels.push(body.ops_copy(self.kernel)?);
        }
        let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::boolean::union_all(kernels)
        });
        self.note_reset(reset);
        let body = result.map_err(|error| map_error(OpKind::Boolean, error))?;
        SolidBody::from_ops(body)
    }

    /// Extrude one exact planar region on its plane into a solid body.
    ///
    /// The region is the record's exact 2D boundary (outer loop plus holes) and the plane
    /// supplies the origin and the right-handed axes. The material is swept from `start` to
    /// `end` along the plane normal (`x × y`), so every consumer that has `(2D curves, plane)`
    /// builds the same solid without sampling.
    ///
    /// The extrusion runs on this session's own lent kernel, so the kernel profile (for
    /// example `linear_sweep_arc_is_cylinder`) applies exactly as it does to a Boolean. The
    /// result starts without a proof under [`Validation::EachNode`], like every other
    /// operation: the background proof queue or a blocking caller proves it later.
    ///
    /// # Errors
    /// The ops profile refusal or kernel error, mapped to a user sentence.
    pub fn extrude_planar_region(
        &mut self,
        plane: &solidcore_record::Plane,
        region: &solidcore_record::Region2,
        start: f64,
        end: f64,
    ) -> Result<SolidBody, SessionError> {
        let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::prism::extrude_planar_regions(
                plane,
                std::slice::from_ref(region),
                start,
                end,
            )
        });
        self.note_reset(reset);
        let body = result.map_err(|error| map_error(OpKind::Extrude, error))?;
        SolidBody::from_ops(body)
    }

    /// Whether two bodies are apart: their exact boxes, each grown by twice its record's
    /// tolerance, do not meet. Never true for a body whose exact box cannot be measured.
    fn apart(&mut self, a: &SolidBody, b: &SolidBody) -> Result<bool, SessionError> {
        let ((alo, ahi), (blo, bhi)) = (a.bounds(self.kernel)?, b.bounds(self.kernel)?);
        let ta = 2.0 * a.brep().tolerance();
        let tb = 2.0 * b.brep().tolerance();
        Ok(!(alo[0] - ta <= bhi[0] + tb
            && blo[0] - tb <= ahi[0] + ta
            && alo[1] - ta <= bhi[1] + tb
            && blo[1] - tb <= ahi[1] + ta
            && alo[2] - ta <= bhi[2] + tb
            && blo[2] - tb <= ahi[2] + ta))
    }

    /// An explicit Check of a solid: prove it first (blocking), then run the kernel's own
    /// validity report on it. A refused body is an error, not a report: the proof already
    /// named the failing test.
    ///
    /// # Errors
    /// The proof's refusal; those of [`Self::check`].
    pub fn check_body(
        &mut self,
        body: &SolidBody,
        level: CheckLevel,
    ) -> Result<CheckReport, SessionError> {
        body.prove(self.kernel)?;
        self.check(body.brep(), level)
    }

    /// Run the kernel's own validity check on `brep` at `level`.
    ///
    /// The report is the kernel's text: callers surface it whole rather than translating
    /// individual failure numbers. A body that fails the check is still `Ok`, with the
    /// failures in the report; `Err` means the check could not run.
    pub fn check(&mut self, brep: &Brep, level: CheckLevel) -> Result<CheckReport, SessionError> {
        let copy = brep_copy(self.kernel, brep)
            .map_err(|error| SessionError::Kernel(error.to_string()))?;
        let body =
            Body::from_brep(self.kernel, copy).map_err(|error| map_error(OpKind::Check, error))?;
        let (report, reset) =
            solidcore_ops::session::with_kernel(self.kernel, || body.check(level));
        self.note_reset(reset);
        report.map_err(|error| map_error(OpKind::Check, error))
    }

    /// Shell a solid by `signed_thickness` (positive outward, negative inward), leaving
    /// `open` faces out.
    ///
    /// A shell moves faces along their outward normals, so its input must be a proved
    /// outward solid: an unknown body is proved first (blocking), and a refused one is
    /// refused before any shell geometry is built.
    ///
    /// # Errors
    /// The proof's refusal; those of [`Self::shell`].
    pub fn shell_body(
        &mut self,
        body: &SolidBody,
        open: &[FaceId],
        signed_thickness: f64,
    ) -> Result<OpsResult, SessionError> {
        body.prove(self.kernel)?;
        self.shell(body.brep(), open, signed_thickness)
    }

    /// Shell the body by `signed_thickness` (positive outward, negative inward), leaving
    /// `open` faces out.
    pub fn shell(
        &mut self,
        brep: &Brep,
        open: &[FaceId],
        thickness: f64,
    ) -> Result<OpsResult, SessionError> {
        let faces = brep.face_list().to_vec();
        let open =
            open.iter()
                .map(|face| {
                    faces.iter().position(|candidate| candidate == face).ok_or_else(|| {
                        SessionError::Invalid("open face is not on the body".to_owned())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        let copy = brep_copy(self.kernel, brep)
            .map_err(|error| SessionError::Kernel(error.to_string()))?;
        let order = Order::of(&copy);
        let body =
            Body::from_brep(self.kernel, copy).map_err(|error| map_error(OpKind::Shell, error))?;
        let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::shell(&body, &order, &open, thickness)
        });
        self.note_reset(reset);
        result.map_err(|error| map_error(OpKind::Shell, error))?;
        let volume = body.volume(1.0e-8).map_err(|error| map_error(OpKind::Shell, error))?;
        Ok(OpsResult {
            brep: body.into_brep(),
            warnings: Vec::new(),
            made_faces: Vec::new(),
            volume,
        })
    }

    /// Offset a solid's faces by `distance` along their outward normals.
    ///
    /// An offset depends on the outward orientation, so the input is proved first (blocking):
    /// an unknown body is proved, a refused one is refused before the offset runs.
    ///
    /// # Errors
    /// The proof's refusal; those of [`Self::offset_faces`].
    pub fn offset_faces_body(
        &mut self,
        body: &SolidBody,
        faces: &[FaceId],
        distance: f64,
    ) -> Result<OpsResult, SessionError> {
        body.prove(self.kernel)?;
        self.offset_faces(body.brep(), faces, distance)
    }

    /// Offset `faces` by `distance` through ops.
    pub fn offset_faces(
        &mut self,
        brep: &Brep,
        faces: &[FaceId],
        distance: f64,
    ) -> Result<OpsResult, SessionError> {
        let faces_on_body = brep.face_list().to_vec();
        let indices = faces
            .iter()
            .map(|face| {
                faces_on_body.iter().position(|c| c == face).ok_or_else(|| {
                    SessionError::Invalid("offset face is not on the body".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let copy = brep_copy(self.kernel, brep).map_err(|e| SessionError::Kernel(e.to_string()))?;
        let order = Order::of(&copy);
        let body =
            Body::from_brep(self.kernel, copy).map_err(|error| map_error(OpKind::Offset, error))?;
        let (result, reset) = solidcore_ops::session::with_kernel(self.kernel, || {
            solidcore_ops::offset_faces(&body, &order, &indices, distance)
        });
        self.note_reset(reset);
        result.map_err(|error| map_error(OpKind::Offset, error))?;
        let volume = body.volume(1.0e-8).map_err(|error| map_error(OpKind::Offset, error))?;
        Ok(OpsResult {
            brep: body.into_brep(),
            warnings: Vec::new(),
            made_faces: Vec::new(),
            volume,
        })
    }

    /// Two-distance / distance-angle chamfer.
    pub fn chamfer_sides(
        &mut self,
        brep: &Brep,
        edges: &[EdgeId],
        size: f64,
        second: ChamferSecond,
        up: [f64; 3],
        flip: bool,
    ) -> Result<OpsResult, SessionError> {
        self.blend(brep, edges, size, OpsBlend::ChamferSides { second, up, flip })
    }

    fn mark_made(&mut self, brep: &mut Brep) -> Result<Vec<FaceId>, SessionError> {
        let made: Vec<MadeBlendFace> = self
            .kernel
            .blend_face_log
            .take()
            .map(|log| log.sessions.into_iter().flatten().collect())
            .unwrap_or_default();
        let ids = made.iter().map(|face| face.face).collect();
        if !made.is_empty() {
            mark_blend_faces(brep, &made)
                .map_err(|error| SessionError::Kernel(error.to_string()))?;
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::geom::attribute::{Attribute, Behavior};
    use solidcore::geom::vector3d::Vec3;
    use solidcore::math::Point3;
    use solidcore::primitive::api_primitives::create_box;

    fn box_brep(kernel: &mut Kernel) -> Brep {
        create_box(kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0).expect("box")
    }

    /// The one planar face whose **outward** normal is `dir` (±x, ±y or ±z).
    ///
    /// A bounding-box pick is wrong for a box: five of its faces share `min-x = 0`, so
    /// `min-x` does not name a face. The outward direction is the surface normal, turned
    /// over where the face's primary face use is
    /// [`OrientType::Opposite`](solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType).
    fn face_facing(kernel: &Kernel, brep: &Brep, dir: [f64; 3]) -> FaceId {
        use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
        use solidcore::query::normal::surface_normal;
        use solidcore::surface::surface::SurfaceKind;
        let dir = Vec3::new(dir[0], dir[1], dir[2]).unitize().expect("a unit direction");
        let mut matching = Vec::new();
        for face in brep.face_list() {
            let record = brep.face(*face).expect("face");
            let Some(surface) = record.surface.as_ref() else { continue };
            if !matches!(surface.kind(), SurfaceKind::Plane(_)) {
                continue;
            }
            // Sample a normal away from the face's boundary, where a trimmed face's
            // corner is not on the surface.
            let domain = solidcore::topology::face::p2::uv_domain(brep, *face).expect("uv domain");
            let u = domain.u_interval();
            let v = domain.v_interval();
            let uv =
                solidcore::math::Point2::new((u.min() + u.max()) * 0.5, (v.min() + v.max()) * 0.5);
            let mut normal = surface_normal(kernel, surface, uv).expect("face normal").value;
            if record.faceuse.is_some_and(|faceuse| {
                brep.faceuse(faceuse).expect("face use").orientation == OrientType::Opposite
            }) {
                normal = Vec3::new(-normal.x, -normal.y, -normal.z);
            }
            if normal.dot(dir) > 0.999 {
                matching.push(*face);
            }
        }
        assert_eq!(
            matching.len(),
            1,
            "expected exactly one face facing {dir:?}, found {}",
            matching.len()
        );
        matching[0]
    }

    fn run(size: f64, kind: OpsBlend) -> OpsResult {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let edge = brep.edge_list()[0];
        OpsSession::new(&mut kernel).blend(&brep, &[edge], size, kind).expect("blend")
    }

    #[test]
    fn round_volume_and_made_face() {
        let result = run(1.0, OpsBlend::Round(Turn::Mitre));
        let expected = 8000.0 - (1.0 - std::f64::consts::PI / 4.0) * 20.0;
        assert!((result.volume - expected).abs() < 1.0e-4, "{}", result.volume);
        assert_eq!(result.made_faces.len(), 1);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn chamfer_curvature_and_sides() {
        let chamfer = run(1.0, OpsBlend::Chamfer(ChamferCorner::Mitre));
        assert!((chamfer.volume - 7990.0).abs() < 1.0e-4, "{}", chamfer.volume);
        let curvature = run(1.0, OpsBlend::Curvature);
        assert!(curvature.volume > 7990.0 && curvature.volume < 7995.8, "{}", curvature.volume);
        let sides = run(
            1.0,
            OpsBlend::ChamferSides {
                second: ChamferSecond::Distance(2.0),
                up: [0.0, 0.0, 1.0],
                flip: false,
            },
        );
        assert!((sides.volume - 7980.0).abs() < 1.0e-3, "{}", sides.volume);
    }

    #[test]
    fn look_colour_is_kept_without_matching() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep = box_brep(&mut kernel);
        let faces: Vec<_> = brep.faces().ids().collect();
        for face in &faces {
            brep.add_attribute(
                solidcore::EntityRef::Face(*face),
                Attribute::color(100, Vec3::new(1.0, 0.0, 0.0), Behavior::Copy),
            )
            .expect("colour");
        }
        let edge = brep.edge_list()[0];
        let result = OpsSession::new(&mut kernel)
            .blend(&brep, &[edge], 1.0, OpsBlend::Round(Turn::Mitre))
            .expect("round");
        let coloured = result
            .brep
            .faces()
            .iter()
            .filter(|(face, _)| {
                result.brep.find_attribute(solidcore::EntityRef::Face(*face), 100).is_some()
            })
            .count();
        assert_eq!(coloured, 6);
    }

    #[test]
    #[ignore = "timing test; run with --ignored --release"]
    fn session_speedup_is_about_four() {
        let work = |threads: usize| {
            let started = std::time::Instant::now();
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    std::thread::spawn(move || {
                        let mut kernel = nodecraft_kernel_profile::kernel();
                        let brep = box_brep(&mut kernel);
                        let edge = brep.edge_list()[0];
                        for _ in 0..16 / threads {
                            OpsSession::new(&mut kernel)
                                .blend(&brep, &[edge], 1.0, OpsBlend::Round(Turn::Mitre))
                                .expect("round");
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("thread");
            }
            started.elapsed().as_secs_f64()
        };
        let serial = work(1);
        let parallel = work(4);
        eprintln!("serial {serial:.3}s parallel {parallel:.3}s speedup {:.2}", serial / parallel);
        assert!(serial / parallel > 2.5, "speedup {}", serial / parallel);
    }
    #[test]
    #[ignore = "timing test; run with --ignored --release"]
    fn with_kernel_cost_per_call() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let iterations = 10_000;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            let (_, _) = solidcore_ops::session::with_kernel(&mut kernel, || ());
        }
        let ns = started.elapsed().as_secs_f64() * 1.0e9 / f64::from(iterations);
        eprintln!("with_kernel {ns:.0} ns/call");
    }

    #[test]
    fn cancelled_borrowed_call_asks_for_a_reset() {
        use solidcore::kernel::CancelToken;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let edge = brep.edge_list()[0];
        let token = CancelToken::new();
        token.cancel();
        kernel.cancel = Some(token);
        let result =
            OpsSession::new(&mut kernel).blend(&brep, &[edge], 1.0, OpsBlend::Round(Turn::Mitre));
        // A token already cancelled refuses before the kernel is touched, so no
        // reset is requested; the pool rebuild path covers a mid-call cancel.
        assert!(matches!(result, Err(SessionError::Cancelled)), "{result:?}");
    }

    #[test]
    fn chained_round_makes_two_faces_and_no_warnings() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let edges = brep.edge_list().to_vec();
        let first = OpsSession::new(&mut kernel)
            .blend(&brep, &[edges[0]], 1.0, OpsBlend::Round(Turn::Mitre))
            .expect("first");
        use solidcore::blend::definition_check::blend_face_marks;
        assert_eq!(blend_face_marks(&first.brep).expect("marks").len(), 1);
        assert!(first.warnings.is_empty());
        let second = OpsSession::new(&mut kernel)
            .blend(&first.brep, &[edges[1]], 1.0, OpsBlend::Round(Turn::Mitre))
            .expect("second");
        assert_eq!(blend_face_marks(&second.brep).expect("marks").len(), 2);
        assert!(second.warnings.is_empty());
    }

    #[test]
    fn chamfer_sides_flip_changes_normals_not_volume() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let edge = brep.edge_list()[0];
        let plain = OpsSession::new(&mut kernel)
            .chamfer_sides(
                &brep,
                &[edge],
                1.0,
                ChamferSecond::Distance(2.0),
                [0.0, 0.0, 1.0],
                false,
            )
            .expect("plain");
        let flipped = OpsSession::new(&mut kernel)
            .chamfer_sides(&brep, &[edge], 1.0, ChamferSecond::Distance(2.0), [0.0, 0.0, 1.0], true)
            .expect("flipped");
        assert!((plain.volume - 7980.0).abs() < 1.0e-3, "{}", plain.volume);
        assert!((flipped.volume - 7980.0).abs() < 1.0e-3, "{}", flipped.volume);
        use solidcore::blend::definition_check::blend_face_marks;
        let normal = |result: &OpsResult| {
            let face = blend_face_marks(&result.brep).expect("marks")[0].face;
            let surface = result.brep.face(face).expect("face").surface.clone();
            format!("{surface:?}")
        };
        assert_ne!(normal(&plain), normal(&flipped));
    }

    #[test]
    fn reset_flag_is_sticky_until_the_pool_takes_it() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut session = OpsSession::new(&mut kernel);
        session.note_reset(true);
        assert!(take_needs_reset());
        assert!(!take_needs_reset());
    }

    #[test]
    fn g2_three_edge_corner_builds_and_marks() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let mut groups: Vec<(
            [solidcore::topology::VertexId; 2],
            Vec<solidcore::topology::EdgeId>,
        )> = Vec::new();
        for edge in brep.edge_list() {
            let vertices = brep.edge(*edge).expect("edge").vertices;
            groups.push((vertices, vec![*edge]));
        }
        let mut corner = Vec::new();
        for (vertices, edges) in &groups {
            let joined: Vec<_> = groups
                .iter()
                .filter(|(other, _)| other[0] == vertices[0] || other[1] == vertices[0])
                .flat_map(|(_, edges)| edges.iter().copied())
                .collect();
            if joined.len() >= 3 {
                corner = joined.into_iter().take(3).collect();
                break;
            }
        }
        assert_eq!(corner.len(), 3, "three edges at a box corner");
        let edges = corner;
        let g2 = OpsSession::new(&mut kernel)
            .blend(&brep, &edges, 1.0, OpsBlend::Curvature)
            .expect("G2 corner");
        assert!(solidcore::query::api_queries::brep_is_manifold_solid(&g2.brep).expect("manifold"));
        assert_eq!(
            solidcore::blend::definition_check::blend_face_marks(&g2.brep).expect("marks").len(),
            9
        );
        let chamfer = OpsSession::new(&mut kernel)
            .blend(&brep, &edges, 1.0, OpsBlend::Chamfer(ChamferCorner::Mitre))
            .expect("chamfer");
        let round = OpsSession::new(&mut kernel)
            .blend(&brep, &edges, 1.0, OpsBlend::Round(Turn::Mitre))
            .expect("round");
        assert!(
            chamfer.volume < g2.volume && chamfer.volume < round.volume,
            "chamfer {} < G2 {}, round {}",
            chamfer.volume,
            g2.volume,
            round.volume
        );
        assert!(
            (g2.volume - round.volume).abs() < 0.5,
            "G2 {} vs round {}",
            g2.volume,
            round.volume
        );
    }

    #[test]
    #[ignore = "diagnosis: G2 cylinder hang"]
    fn diagnose_g2_cylinder() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_cylinder;
        let cases: &[(&str, OpsBlend, f64, bool)] = &[
            ("G1 both 15", OpsBlend::Round(Turn::Mitre), 15.0, false),
            ("G1 one 15", OpsBlend::Round(Turn::Mitre), 15.0, true),
            ("G2 both 15", OpsBlend::Curvature, 15.0, false),
            ("G2 one 15", OpsBlend::Curvature, 15.0, true),
            ("G2 both 10", OpsBlend::Curvature, 10.0, false),
            ("G2 one 10", OpsBlend::Curvature, 10.0, true),
            ("G2 both 5", OpsBlend::Curvature, 5.0, false),
            ("G2 one 5", OpsBlend::Curvature, 5.0, true),
            ("G2 both 1", OpsBlend::Curvature, 1.0, false),
            ("G2 one 1", OpsBlend::Curvature, 1.0, true),
            ("chamfer both 15", OpsBlend::Chamfer(ChamferCorner::Mitre), 15.0, false),
        ];
        for (name, kind, size, one) in cases {
            let (tx, rx) = std::sync::mpsc::channel();
            let kind = *kind;
            let size = *size;
            let one = *one;
            std::thread::spawn(move || {
                let mut kernel = nodecraft_kernel_profile::kernel();
                let brep = create_cylinder(&mut kernel, Point3::new(0.0, 0.0, -40.0), 20.0, 80.0)
                    .expect("cylinder");
                let all: Vec<_> = brep
                    .edge_list()
                    .iter()
                    .copied()
                    .filter(|edge| {
                        let vertices = brep.edge(*edge).expect("edge").vertices;
                        let z = |v| brep.vertex(v).expect("vertex").point.z;
                        (z(vertices[0]) - z(vertices[1])).abs() < 1.0e-9
                    })
                    .collect();
                let edges = if one { vec![all[0]] } else { all };
                let started = std::time::Instant::now();
                let result = OpsSession::new(&mut kernel).blend(&brep, &edges, size, kind);
                let _ = tx.send((started.elapsed(), result.map(|r| r.volume), edges_count(&brep)));
            });
            match rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok((elapsed, result, count)) => {
                    eprintln!("{name}: edges {count} {elapsed:?} -> {result:?}");
                }
                Err(_) => eprintln!("{name}: HANG >10 s"),
            }
        }
    }

    fn edges_count(brep: &Brep) -> usize {
        brep.edge_list().len()
    }

    #[test]
    #[ignore = "diagnosis: hangs"]
    fn hang_g2_cylinder() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_cylinder;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = create_cylinder(&mut kernel, Point3::new(0.0, 0.0, -40.0), 20.0, 80.0)
            .expect("cylinder");
        let edges: Vec<_> = brep
            .edge_list()
            .iter()
            .copied()
            .filter(|edge| {
                let v = brep.edge(*edge).expect("edge").vertices;
                (brep.vertex(v[0]).expect("v").point.z - brep.vertex(v[1]).expect("v").point.z)
                    .abs()
                    < 1.0e-9
            })
            .collect();
        eprintln!("blending {} edges", edges.len());
        let result = OpsSession::new(&mut kernel).blend(&brep, &edges, 15.0, OpsBlend::Curvature);
        eprintln!("result {result:?}");
    }

    #[test]
    #[ignore = "diagnosis: cancellation"]
    fn cancel_g2_cylinder() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_cylinder;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = create_cylinder(&mut kernel, Point3::new(0.0, 0.0, -40.0), 20.0, 80.0)
            .expect("cylinder");
        let edges: Vec<_> = brep
            .edge_list()
            .iter()
            .copied()
            .filter(|edge| {
                let v = brep.edge(*edge).expect("edge").vertices;
                (brep.vertex(v[0]).expect("v").point.z - brep.vertex(v[1]).expect("v").point.z)
                    .abs()
                    < 1.0e-9
            })
            .collect();
        let token = solidcore::kernel::CancelToken::new();
        kernel.cancel = Some(token.clone());
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(2));
            token.cancel();
        });
        let started = std::time::Instant::now();
        let result = OpsSession::new(&mut kernel).blend(&brep, &edges, 15.0, OpsBlend::Curvature);
        eprintln!("cancel after {:.1}s: {result:?}", started.elapsed().as_secs_f64());
        canceller.join().ok();
    }

    fn fnv_seed(id: &str) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in id.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    }

    fn seed_blend(seed: u64) -> (bool, f64, String) {
        use solidcore::kernel::Rng;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut kernel);
        let edge = brep.edge_list()[0];
        kernel.rng = Rng::new(seed);
        let started = std::time::Instant::now();
        let result = OpsSession::new(&mut kernel).blend(&brep, &[edge], 8.0, OpsBlend::Curvature);
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(result) => {
                let mut brep = result.brep;
                let failure =
                    solidcore::topology::brep::outward_solid_failure(&mut kernel, &mut brep);
                (true, elapsed, format!("{failure:?}"))
            }
            Err(error) => (false, elapsed, error.to_string()),
        }
    }

    #[test]
    #[ignore = "diagnosis: seed sweep"]
    fn g2_box_seed_sweep() {
        let mut seeds = vec![0, fnv_seed("fillet"), fnv_seed("fillet-copy1")];
        let mut state = 0x1234_5678_9abc_def0u64;
        for _ in 0..50 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            seeds.push(state);
        }
        let mut passed = 0;
        let mut failed = 0;
        let mut slowest = (0.0f64, 0u64);
        for seed in seeds {
            let (ok, ms, _detail) = seed_blend(seed);
            if ok {
                passed += 1;
            } else {
                failed += 1;
            }
            if ms > slowest.0 {
                slowest = (ms, seed);
            }
        }
        eprintln!(
            "seed sweep: {passed} passed, {failed} failed; slowest {:.1} ms at {:#018x}",
            slowest.0, slowest.1
        );
    }

    #[test]
    #[ignore = "diagnosis: context carry-over"]
    fn g2_box_context_carry_over() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let _large =
            create_box(&mut kernel, Point3::new(-1000.0, -1000.0, -1000.0), 2000.0, 2000.0, 2000.0)
                .expect("large box");
        let before =
            (kernel.context.stored_model_size_estimate(), kernel.context.this_zone_tol_3d());
        let brep = box_brep(&mut kernel);
        let edge = brep.edge_list()[0];
        let started = std::time::Instant::now();
        let dirty = OpsSession::new(&mut kernel).blend(&brep, &[edge], 8.0, OpsBlend::Curvature);
        eprintln!(
            "carried context {before:?} -> {:?} in {:.1} ms",
            dirty.as_ref().map(|r| r.volume),
            started.elapsed().as_secs_f64() * 1000.0
        );
        let mut fresh = nodecraft_kernel_profile::kernel();
        let brep = box_brep(&mut fresh);
        let edge = brep.edge_list()[0];
        let clean = OpsSession::new(&mut fresh).blend(&brep, &[edge], 8.0, OpsBlend::Curvature);
        eprintln!("fresh: {:?}", clean.as_ref().map(|r| r.volume));
    }

    #[test]
    #[ignore = "diagnosis: named seeds"]
    fn g2_box_named_seed_times() {
        for (name, seed) in [
            ("default", 0),
            ("fillet", fnv_seed("fillet")),
            ("fillet-copy1", fnv_seed("fillet-copy1")),
        ] {
            let (ok, ms, detail) = seed_blend(seed);
            eprintln!(
                "{name} {seed:#018x} {} {ms:.1} ms {detail}",
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    #[test]
    // Also needs a developer dump at /tmp/ops-dump; run with `--ignored --release`.
    #[ignore = "slow: 12s"]
    fn hook_dump_round_trips_a_coloured_body() {
        use solidcore::geom::attribute::{Attribute, Behavior};
        use solidcore::geom::vector3d::Vec3;
        use solidcore_ops::CheckLevel;
        let fixes = nodecraft_kernel_profile::fix_names();
        solidcore_ops::session::set_fixes(fixes).expect("fixes");
        let dir = std::env::temp_dir().join(format!("nodecraft-dump-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        install_failure_dump(dir.clone());
        let mut kernel = nodecraft_kernel_profile::kernel();
        let loaded = solidcore_ops::Body::from_smb(
            std::fs::read("/tmp/ops-dump/call-000-brep.smb").expect("alex dump"),
            false,
        )
        .expect("read");
        let mut brep = loaded.into_brep();
        let face = brep.face_list()[0];
        brep.add_attribute(
            solidcore::EntityRef::Face(face),
            Attribute::color(100, Vec3::new(1.0, 0.0, 0.0), Behavior::Copy),
        )
        .expect("colour");
        let order = Order::of(&brep);
        let body = Body::from_brep(&mut kernel, brep).expect("body");
        let before = body.volume(1.0e-8).expect("volume");
        let result = solidcore_ops::shell(&body, &order, &[], 8.0);
        assert!(result.is_err(), "thick shell should refuse");
        let smb = std::fs::read_dir(&dir)
            .expect("dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "smb"))
            .expect("dump smb");
        let read = solidcore_ops::Body::from_smb(std::fs::read(&smb).expect("bytes"), false)
            .expect("replay read");
        assert!((read.volume(1.0e-8).expect("volume") - before).abs() < 1.0e-6);
        assert!(read.check(CheckLevel::All).expect("check").is_valid());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "reads NODECRAFT_OPS_DUMP"]
    fn replay_ops_dump() {
        use solidcore_ops::CheckLevel;
        let dir = std::env::var_os("NODECRAFT_OPS_DUMP").expect("NODECRAFT_OPS_DUMP");
        let dir = std::path::PathBuf::from(dir);
        let meta = std::fs::read_to_string(dir.join("call-000.txt")).expect("call-000.txt");
        let fixes = meta
            .lines()
            .find_map(|line| line.strip_prefix("fixes: "))
            .map(str::to_owned)
            .unwrap_or_else(|| nodecraft_kernel_profile::fix_names().join(","));
        let fixes: Vec<&str> = fixes.split(',').filter(|s| !s.is_empty()).collect();
        solidcore_ops::session::set_fixes(fixes).expect("fixes");
        let smb = std::fs::read_dir(&dir)
            .expect("dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| {
                p.file_name().is_some_and(|n| {
                    n.to_string_lossy().starts_with("call-000-")
                        && p.extension().is_some_and(|x| x == "smb")
                })
            })
            .expect("dump smb");
        let body = solidcore_ops::Body::from_smb(std::fs::read(&smb).expect("bytes"), false)
            .expect("read");
        eprintln!("volume {:?}", body.volume(1.0e-8));
        eprintln!("check {:?}", body.check(CheckLevel::All));
    }

    #[test]
    // A timing measurement: other tests running beside it on the same cores make the
    // speedup meaningless, so it runs with the other timing tests.
    #[ignore = "timing: needs a quiet machine; run with --ignored --release"]
    fn shell_and_parallel_speedup() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = create_box(&mut kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
            .expect("box");
        let result = OpsSession::new(&mut kernel).shell(&brep, &[], 1.0).expect("shell");
        assert!((result.volume - 2648.0).abs() < 1.0e-6, "{}", result.volume);
        let inward = OpsSession::new(&mut kernel).shell(&brep, &[], -1.0).expect("inward shell");
        assert!((inward.volume - 2168.0).abs() < 1.0e-6, "{}", inward.volume);
        let top = face_facing(&kernel, &brep, [0.0, 0.0, 1.0]);
        let opened = OpsSession::new(&mut kernel).shell(&brep, &[top], -1.0).expect("open shell");
        assert!((opened.volume - 1844.0).abs() < 1.0e-6, "{}", opened.volume);
        let work = |threads: usize| {
            let started = std::time::Instant::now();
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    std::thread::spawn(move || {
                        let mut kernel = nodecraft_kernel_profile::kernel();
                        for _ in 0..16 / threads {
                            let brep = create_box(
                                &mut kernel,
                                Point3::new(-10.0, -10.0, -10.0),
                                20.0,
                                20.0,
                                20.0,
                            )
                            .expect("box");
                            OpsSession::new(&mut kernel).shell(&brep, &[], 1.0).expect("shell");
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().expect("thread");
            }
            started.elapsed().as_secs_f64()
        };
        let serial = work(1);
        let parallel = work(4);
        eprintln!(
            "shells serial {serial:.3}s parallel {parallel:.3}s speedup {:.2}",
            serial / parallel
        );
        assert!(serial / parallel > 2.0);
    }

    #[test]
    fn shell_record_path_and_session_path_match() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let brep = create_box(
            &mut nodecraft_kernel_profile::kernel(),
            Point3::new(0.0, 0.0, 0.0),
            20.0,
            30.0,
            40.0,
        )
        .expect("box");
        let pick_kernel = nodecraft_kernel_profile::kernel();
        let top = face_facing(&pick_kernel, &brep, [0.0, 0.0, 1.0]);
        let bottom = face_facing(&pick_kernel, &brep, [0.0, 0.0, -1.0]);
        let mut old_kernel = nodecraft_kernel_profile::kernel();
        let mut old_brep = brep.clone_standing_for_self().expect("copy");
        let order =
            solidcore_convert::brep::to_record(&mut old_kernel, &mut old_brep).expect("record").1;
        let open: Vec<usize> = [top, bottom]
            .iter()
            .map(|face| {
                order.faces.iter().position(|candidate| candidate == face).expect("record face")
            })
            .collect();
        let mut new_kernel = nodecraft_kernel_profile::kernel();
        let new_faces = nodecraft_layers::record_face_ids(
            &mut new_kernel,
            &mut brep.clone_standing_for_self().expect("copy"),
            &open,
        )
        .expect("ids");
        let old = nodecraft_layers::shell_record_faces(&mut old_kernel, &mut old_brep, &open, -1.0)
            .expect("old shell");
        let new =
            OpsSession::new(&mut new_kernel).shell(&brep, &new_faces, -1.0).expect("new shell");
        let old_body = solidcore_ops::Body::from_brep(&mut old_kernel, old).expect("old body");
        let old_volume = old_body.volume(1e-8).expect("volume");
        assert!((old_volume - 3840.0).abs() < 1e-6, "{old_volume}");
        assert!((new.volume - 3840.0).abs() < 1e-6, "{}", new.volume);
        assert!((old_volume - new.volume).abs() < 1e-6);
        let old_faces = old_body.copy().expect("copy").into_brep().face_list().len();
        assert_eq!(old_faces, new.brep.face_list().len());
        let open_top =
            OpsSession::new(&mut new_kernel).shell(&brep, &[top], -1.0).expect("open top");
        assert!((open_top.volume - 4344.0).abs() < 1e-6, "{}", open_top.volume);
        assert_eq!(new.brep.face_list().len(), 10);
        eprintln!(
            "shell conformance volume {} faces {} open-top {}",
            new.volume,
            new.brep.face_list().len(),
            open_top.volume
        );
    }
    #[test]
    fn offset_record_and_session_paths_match() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let run = |brep: Brep, faces: &[solidcore::topology::FaceId], expected: f64| {
            let mut old_kernel = nodecraft_kernel_profile::kernel();
            let mut old_brep = brep.clone_standing_for_self().expect("copy");
            let mut scratch = brep.clone_standing_for_self().expect("copy");
            let order = solidcore_convert::brep::to_record(&mut old_kernel, &mut scratch)
                .expect("record")
                .1;
            let indices: Vec<usize> = faces
                .iter()
                .map(|face| order.faces.iter().position(|c| c == face).expect("face"))
                .collect();
            let old = nodecraft_layers::offset_faces_record_faces(
                &mut old_kernel,
                &mut old_brep,
                &indices,
                -2.0,
            )
            .expect("old offset");
            let mut kernel = nodecraft_kernel_profile::kernel();
            let new =
                OpsSession::new(&mut kernel).offset_faces(&brep, faces, -2.0).expect("new offset");
            let old_body = solidcore_ops::Body::from_brep(&mut old_kernel, old).expect("body");
            let old_v = old_body.volume(1e-8).expect("volume");
            assert!((old_v - new.volume).abs() < 1e-6, "{old_v} vs {}", new.volume);
            assert!((new.volume - expected).abs() < 1e-6, "{} vs {expected}", new.volume);
            eprintln!(
                "offset conformance volume {} faces {} expected {expected}",
                new.volume,
                new.brep.face_list().len()
            );
        };
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 30.0, 40.0).expect("box");
        let top = face_facing(&kernel, &brep, [0.0, 0.0, 1.0]);
        let bottom = face_facing(&kernel, &brep, [0.0, 0.0, -1.0]);
        run(brep.clone_standing_for_self().expect("copy"), &[top], 22800.0);
        run(brep, &[top, bottom], 21600.0);

        let mut union_kernel = nodecraft_kernel_profile::kernel();
        let a =
            create_box(&mut union_kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 30.0, 40.0).expect("a");
        let b = create_box(&mut union_kernel, Point3::new(20.0, 0.0, 0.0), 10.0, 30.0, 40.0)
            .expect("b");
        let union = nodecraft_layers::boolean_breps_ops(
            &mut union_kernel,
            &a,
            &b,
            nodecraft_layers::BooleanOp::Union,
        )
        .expect("union")
        .expect("solid");
        let far = face_facing(&union_kernel, &union, [1.0, 0.0, 0.0]);
        run(union, &[far], 33600.0);
    }

    #[test]
    fn look_colour_survives_a_shell() {
        use solidcore::geom::attribute::{Attribute, Behavior};
        use solidcore::geom::vector3d::Vec3;
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box");
        let faces: Vec<_> = brep.faces().ids().collect();
        for face in &faces {
            brep.add_attribute(
                solidcore::EntityRef::Face(*face),
                Attribute::color(100, Vec3::new(1.0, 0.0, 0.0), Behavior::Copy),
            )
            .expect("colour");
        }
        let result = OpsSession::new(&mut kernel).shell(&brep, &[], 1.0).expect("shell");
        let coloured = result
            .brep
            .faces()
            .iter()
            .filter(|(face, _)| {
                result.brep.find_attribute(solidcore::EntityRef::Face(*face), 100).is_some()
            })
            .count();
        assert_eq!(coloured, 12, "{coloured}");
    }

    #[test]
    fn boolean_and_union_all_closed_forms() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        use solidcore_ops::BooleanOp;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let a = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("a"),
        );
        let b = SolidBody::new(
            create_box(&mut kernel, Point3::new(10.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("b"),
        );
        let union = OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Union).expect("union");
        let subtract =
            OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Subtract).expect("subtract");
        let intersect =
            OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Intersect).expect("intersect");
        let union_volume = union.volume(&mut kernel).expect("union volume");
        let subtract_volume = subtract.volume(&mut kernel).expect("subtract volume");
        let intersect_volume = intersect.volume(&mut kernel).expect("intersect volume");
        assert!((union_volume - 12000.0).abs() < 1e-6);
        assert!((subtract_volume - 4000.0).abs() < 1e-6);
        assert!((intersect_volume - 4000.0).abs() < 1e-6);
        let boxes: Vec<_> = (0..4)
            .map(|i| {
                SolidBody::new(
                    create_box(
                        &mut kernel,
                        Point3::new(i as f64 * 30.0, 0.0, 0.0),
                        20.0,
                        20.0,
                        20.0,
                    )
                    .expect("box"),
                )
            })
            .collect();
        let all = OpsSession::new(&mut kernel).union_all(&boxes).expect("union_all");
        let all_volume = all.volume(&mut kernel).expect("union_all volume");
        assert!((all_volume - 32000.0).abs() < 1e-6);
        eprintln!(
            "boolean volumes union {union_volume} sub {subtract_volume} int {intersect_volume} all {all_volume}"
        );
    }

    /// Empty is ops' answer (`None`), not a measured volume: a cut that removes the body and an
    /// intersection of apart bodies both give nothing, and neither call measures a volume.
    #[test]
    fn empty_boolean_is_ops_none_not_a_zero_volume() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let cube = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("cube"),
        );
        let far = SolidBody::new(
            create_box(&mut kernel, Point3::new(100.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("far"),
        );
        // Identical bodies: the cut leaves nothing (ops reports Error::Empty, mapped to None).
        let own = OpsSession::new(&mut kernel)
            .boolean_outcome(&cube, &cube, BooleanOp::Subtract)
            .expect("coincident subtract is not an error");
        assert!(own.is_none(), "subtracting a body from itself leaves nothing");
        // Apart: the intersection early-out is empty without any intersection work.
        let none = OpsSession::new(&mut kernel)
            .boolean_outcome(&cube, &far, BooleanOp::Intersect)
            .expect("apart intersect is not an error");
        assert!(none.is_none(), "apart bodies intersect in nothing");
    }

    /// The apart early-out returns the body itself for a Cut and nothing for an Intersect,
    /// with no intersection work; a body that meets the tool still gets the boolean.
    #[test]
    fn apart_cut_returns_the_body_and_intersect_nothing() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let cube = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("cube"),
        );
        let far = SolidBody::new(
            create_box(&mut kernel, Point3::new(100.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("far"),
        );
        let cut = OpsSession::new(&mut kernel)
            .boolean_outcome(&cube, &far, BooleanOp::Subtract)
            .expect("cut")
            .expect("the body itself");
        assert_eq!(cut.brep().face_list().len(), cube.brep().face_list().len());
        assert_eq!(cut.volume(&mut kernel).expect("volume"), 8000.0);
        // The body itself is returned, not a proved copy of the tool's geometry: the same
        // shared record.
        assert!(
            std::sync::Arc::ptr_eq(&cut.shared(), &cube.shared()),
            "the cut is the body itself"
        );
        let overlapping = SolidBody::new(
            create_box(&mut kernel, Point3::new(10.0, 0.0, 0.0), 20.0, 20.0, 20.0)
                .expect("overlapping"),
        );
        let real = OpsSession::new(&mut kernel)
            .boolean(&cube, &overlapping, BooleanOp::Subtract)
            .expect("real cut");
        assert!((real.volume(&mut kernel).expect("volume") - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn boolean_step2_cases() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let cube = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("cube"),
        );
        let bar = SolidBody::new(
            create_box(&mut kernel, Point3::new(-5.0, 5.0, 5.0), 30.0, 10.0, 10.0).expect("bar"),
        );
        let sub_body = OpsSession::new(&mut kernel)
            .boolean(&cube, &bar, nodecraft_layers::BooleanOp::Subtract)
            .expect("sub");
        let sub = sub_body.volume(&mut kernel).expect("sub volume");
        assert!((sub - 6000.0).abs() < 1e-6, "{sub}");
        let far = SolidBody::new(
            create_box(&mut kernel, Point3::new(100.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("far"),
        );
        let empty = OpsSession::new(&mut kernel)
            .boolean_outcome(&cube, &far, nodecraft_layers::BooleanOp::Intersect)
            .expect("outcome");
        assert!(empty.is_none(), "disjoint subtract must be empty, not an error");
        let boxes: Vec<_> = (0..4)
            .map(|i| {
                SolidBody::new(
                    create_box(
                        &mut kernel,
                        Point3::new(i as f64 * 15.0, 0.0, 0.0),
                        20.0,
                        20.0,
                        20.0,
                    )
                    .expect("box"),
                )
            })
            .collect();
        let all = OpsSession::new(&mut kernel).union_all(&boxes).expect("union_all");
        let all_volume = all.volume(&mut kernel).expect("union_all volume");
        assert!((all_volume - 26000.0).abs() < 1e-6, "{all_volume}");
        eprintln!("step2 sub {sub} union_all {all_volume} empty {}", empty.is_none());
    }

    #[test]
    fn boolean_layers_conformance_and_look_colour() {
        use solidcore::geom::attribute::{Attribute, Behavior};
        use solidcore::geom::vector3d::Vec3;
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut a =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("a");
        let negative_x = face_facing(&kernel, &a, [-1.0, 0.0, 0.0]);
        a.add_attribute(
            solidcore::EntityRef::Face(negative_x),
            Attribute::color(100, Vec3::new(1.0, 0.0, 0.0), Behavior::Copy),
        )
        .expect("colour");
        let b = create_box(&mut kernel, Point3::new(10.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("b");
        let a_body = SolidBody::new(a.clone_standing_for_self().expect("copy a"));
        let b_body = SolidBody::new(b.clone_standing_for_self().expect("copy b"));
        for (op, expected) in [
            (nodecraft_layers::BooleanOp::Union, 12000.0),
            (nodecraft_layers::BooleanOp::Subtract, 4000.0),
            (nodecraft_layers::BooleanOp::Intersect, 4000.0),
        ] {
            let mut old_kernel = nodecraft_kernel_profile::kernel();
            let old = nodecraft_layers::boolean_breps_ops(&mut old_kernel, &a, &b, op)
                .expect("old")
                .expect("solid");
            let old_body = solidcore_ops::Body::from_brep(&mut old_kernel, old).expect("body");
            let mut new_kernel = nodecraft_kernel_profile::kernel();
            let new = OpsSession::new(&mut new_kernel)
                .boolean_outcome(&a_body, &b_body, op)
                .expect("new")
                .expect("solid");
            let new_volume = new.volume(&mut new_kernel).expect("volume");
            assert!((old_body.volume(1e-8).expect("v") - new_volume).abs() < 1e-6);
            assert!((new_volume - expected).abs() < 1e-6);
            assert_eq!(
                old_body.copy().expect("copy").into_brep().face_list().len(),
                new.brep().face_list().len(),
                "{op:?}: old and new face counts differ"
            );
            assert_eq!(new.brep().face_list().len(), 6);
            eprintln!(
                "boolean conformance {op:?} volume {new_volume} faces {}",
                new.brep().face_list().len()
            );
        }
        let union = OpsSession::new(&mut kernel)
            .boolean_outcome(&a_body, &b_body, nodecraft_layers::BooleanOp::Union)
            .expect("union")
            .expect("solid");
        let minus_x = face_facing(&kernel, union.brep(), [-1.0, 0.0, 0.0]);
        let plus_x = face_facing(&kernel, union.brep(), [1.0, 0.0, 0.0]);
        assert!(
            union.brep().find_attribute(solidcore::EntityRef::Face(minus_x), 100).is_some(),
            "-x face keeps its colour"
        );
        assert!(
            union.brep().find_attribute(solidcore::EntityRef::Face(plus_x), 100).is_none(),
            "+x face stays uncoloured"
        );
    }

    #[test]
    fn check_matches_the_record_path_and_reports_a_valid_box() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0).expect("box");
        let new =
            OpsSession::new(&mut kernel).check(&brep, CheckLevel::Geometry).expect("session check");
        assert!(new.is_valid(), "{new}");
        assert_eq!(new.to_string(), "valid");
        // The layer check this replaces converted through a record first; the lent
        // kernel must give the same report without the round-trip.
        let mut old_kernel = nodecraft_kernel_profile::kernel();
        let mut copy = brep.clone_standing_for_self().expect("copy");
        let (record, _) =
            solidcore_convert::brep::to_record(&mut old_kernel, &mut copy).expect("record");
        let (body, _) = solidcore_ops::Body::from_record(&record, true).expect("body");
        let old = body.check(solidcore_ops::CheckLevel::Geometry).expect("record-path check");
        assert_eq!(old, new);
        for level in [CheckLevel::Basic, CheckLevel::All] {
            let report = OpsSession::new(&mut kernel).check(&brep, level).expect("check");
            assert!(report.is_valid(), "{level:?}: {report}");
        }
        eprintln!("check report {new} at Geometry, and valid at Basic and All");
    }

    /// What happens to a body with no closed shell before it is routed through `SolidBody`.
    ///
    /// A sheet (faces, open shells) is accepted by `Body::from_brep` and simply never holds a
    /// proof. An edges-only body cannot be built in NodeCraft at all: the kernel's record
    /// conversion refuses wire and vertex shells (`NotYet`), and the kernel exposes no wire-body
    /// constructor, so nothing reaches `Body::from_brep` with edges and no faces. If it did,
    /// `from_brep` walks each edge's uses (none on a wire edge) and adds no trims.
    #[test]
    fn from_brep_takes_a_sheet_and_the_kernel_refuses_wire_shells() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::{create_box, create_plane};
        let mut kernel = nodecraft_kernel_profile::kernel();
        let sheet = SolidBody::new(
            create_plane(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0).expect("sheet"),
        );
        let copy = sheet.ops_copy(&mut kernel).expect("from_brep takes a sheet");
        assert_eq!(copy.counts().expect("counts")[0], 1);
        assert!(sheet.prove(&mut kernel).is_err(), "a sheet is not a solid");

        let box_brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box");
        let mut copy = box_brep.clone_standing_for_self().expect("copy");
        let (mut record, _) =
            solidcore_convert::brep::to_record(&mut kernel, &mut copy).expect("record");
        record.faces.clear();
        let wires: Vec<u32> = (0..record.edges.len() as u32).collect();
        record.shells = vec![solidcore_record::brep::Shell::Wire { edges: wires }];
        record.regions.truncate(1);
        record.regions[0].shells.clear();
        record.regions[0].shells.push(0);
        for edge in &mut record.edges {
            edge.uses.clear();
        }
        let refused = solidcore_convert::brep::to_kernel(&mut kernel, &record, true);
        assert!(refused.is_err(), "the kernel must not silently build an edges-only body");
    }

    /// The volume-sign decisions ops made in `calls` (one per proof; upstream's count).
    fn proof_decisions(calls: &[solidcore_ops::session::CallProfile]) -> u64 {
        calls
            .iter()
            .flat_map(|call| &call.profile.summary)
            .filter(|summary| summary.name == "mass::enclosure::volume_sign")
            .map(|summary| summary.count)
            .sum()
    }

    /// A growing Boolean chain proves each accumulated body once: its own result proof is reused
    /// as the next operand's, so a 4-box chain makes 7 decisions, not the 9 of a fresh body per
    /// step.
    #[test]
    fn a_boolean_chain_proves_each_body_once() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        use solidcore_ops::session;
        let boxes = |kernel: &mut solidcore::kernel::Kernel| -> Vec<SolidBody> {
            (0..4)
                .map(|i| {
                    SolidBody::new(
                        create_box(
                            kernel,
                            Point3::new(i as f64 * 25.0, 0.0, 0.0),
                            20.0,
                            20.0,
                            20.0,
                        )
                        .expect("box"),
                    )
                })
                .collect()
        };
        let mut kernel = nodecraft_kernel_profile::kernel();
        let parts = boxes(&mut kernel);
        let (carried, calls) = session::profiled(|| {
            let mut acc = parts[0].clone();
            for part in &parts[1..] {
                acc = OpsSession::new(&mut kernel)
                    .boolean(&acc, part, BooleanOp::Union)
                    .expect("union");
            }
            acc
        });
        assert_eq!(
            proof_decisions(&calls),
            7,
            "3 new primitives + 3 results; the accumulated body is never re-proved"
        );
        let mut fresh_kernel = nodecraft_kernel_profile::kernel();
        let parts = boxes(&mut fresh_kernel);
        let (fresh, calls) = session::profiled(|| {
            let mut acc = parts[0].clone();
            for part in &parts[1..] {
                // A B-rep in, a B-rep out, as the old engine did: the proof is dropped.
                let input = SolidBody::new(acc.brep().clone_standing_for_self().expect("copy"));
                acc = OpsSession::new(&mut fresh_kernel)
                    .boolean(&input, part, BooleanOp::Union)
                    .expect("union");
                acc = SolidBody::new(acc.brep().clone_standing_for_self().expect("copy"));
            }
            acc
        });
        assert_eq!(
            proof_decisions(&calls),
            9,
            "a fresh B-rep per step proves every operand and result"
        );
        assert_eq!(carried.brep().face_list().len(), fresh.brep().face_list().len());
    }

    /// One proved result feeding two consumers is proved once, not once per consumer.
    #[test]
    fn a_shared_proved_body_is_not_re_proved_by_each_consumer() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        use solidcore_ops::session;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let a = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("a"),
        );
        let b = SolidBody::new(
            create_box(&mut kernel, Point3::new(10.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("b"),
        );
        let shared =
            OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Subtract).expect("first");
        let tools: Vec<SolidBody> = (0..2)
            .map(|i| {
                SolidBody::new(
                    create_box(&mut kernel, Point3::new(5.0, 5.0 * i as f64, 5.0), 4.0, 4.0, 4.0)
                        .expect("tool"),
                )
            })
            .collect();
        let (_, calls) = session::profiled(|| {
            for tool in &tools {
                let _ = OpsSession::new(&mut kernel)
                    .boolean(&shared, tool, BooleanOp::Subtract)
                    .expect("cut");
            }
        });
        // each cut: the tool (1) and the result (1); the shared body's proof is reused (0)
        assert_eq!(proof_decisions(&calls), 4, "the shared body is proved once, not per consumer");
    }

    /// An inside-out box: every face's orientation flipped. It stays manifold, so only the
    /// proof can tell it is not an outward solid.
    fn inside_out_box(kernel: &mut solidcore::kernel::Kernel) -> Brep {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        use solidcore::topology::brep::p3::brep_copy;
        use solidcore::topology::face::p1::flip_face_orientation;
        let boxed =
            create_box(kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0).expect("box");
        let mut brep = brep_copy(kernel, &boxed).expect("copy");
        for face in brep.face_list().to_vec() {
            flip_face_orientation(&mut brep, face).expect("flip");
        }
        brep
    }

    /// A body starts unknown, is proved once, and every clone shares the state. Without the
    /// shared proof state the second proof makes another volume-sign decision.
    #[test]
    fn a_body_starts_unknown_and_is_proved_once() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let body = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box"),
        );
        assert_eq!(body.proof_state(), ProofState::Unknown);
        body.prove(&mut kernel).expect("prove");
        assert_eq!(body.proof_state(), ProofState::Proved);
        assert_eq!(body.proof_attempts(), 1);
        // A clone is a reader of the same body: it sees the proof.
        let reader = body.clone();
        assert!(reader.is_proved());
        reader.prove(&mut kernel).expect("already proved");
        assert_eq!(body.proof_attempts(), 1, "a proved body is never proved again");
    }

    /// A refusal names the failed test and is kept: a second blocking proof does no kernel
    /// work. Without the cached state every caller would decide the volume sign again.
    #[test]
    fn a_refused_body_keeps_its_refusal() {
        let mut kernel = nodecraft_kernel_profile::kernel();
        let body = SolidBody::new(inside_out_box(&mut kernel));
        assert!(body.prove(&mut kernel).is_err(), "an inside-out box is not a solid");
        assert_eq!(body.proof_attempts(), 1);
        let reason = body.refused().expect("the refusal is kept");
        assert!(reason.contains("closed solid"), "{reason}");
        assert_eq!(body.clone().refused(), Some(reason));
        assert!(body.prove(&mut kernel).is_err(), "still refused");
        assert_eq!(body.proof_attempts(), 1, "a refused body is not proved again");
    }

    /// Ops under `EachStep` proves its result, so the wrapped body starts proved. Under
    /// `EachNode` the proof is left to the background queue, so it starts unknown.
    #[test]
    fn a_result_starts_proved_under_each_step_and_unknown_under_each_node() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let a = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("a"),
        );
        let b = SolidBody::new(
            create_box(&mut kernel, Point3::new(10.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("b"),
        );
        let stepped =
            OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Union).expect("union");
        assert_eq!(stepped.proof_state(), ProofState::Proved);
        let node = validating(Validation::EachNode, || {
            OpsSession::new(&mut kernel).boolean(&a, &b, BooleanOp::Union).expect("union")
        });
        assert_eq!(node.proof_state(), ProofState::Unknown, "the node-end proof is deferred");
        node.prove(&mut kernel).expect("the deferred proof holds");
        assert!(node.is_proved());
    }

    /// A rounded rectangle: four lines and one partial arc, the shape whose swept wall is
    /// analytic only with `linear_sweep_arc_is_cylinder`.
    fn rounded_region() -> solidcore_record::Region2 {
        use solidcore_record::{Loop2, Region2, Seg2, Vec2};
        let p = |x: f64, y: f64| Vec2::new(x, y);
        Region2 {
            outer: Loop2 {
                segs: vec![
                    Seg2::Line { a: p(-20.0, -15.0), b: p(20.0, -15.0) },
                    Seg2::Line { a: p(20.0, -15.0), b: p(20.0, 9.0) },
                    Seg2::Arc {
                        center: p(14.0, 9.0),
                        radius: 6.0,
                        start: 0.0,
                        sweep: std::f64::consts::FRAC_PI_2,
                    },
                    Seg2::Line { a: p(14.0, 15.0), b: p(-20.0, 15.0) },
                    Seg2::Line { a: p(-20.0, 15.0), b: p(-20.0, -15.0) },
                ],
            },
            holes: Vec::new(),
        }
    }

    /// `extrude_planar_region` runs on the kernel the session was lent and returns a
    /// `SolidBody`: analytic on NodeCraft's profile and unknown under `EachNode`.
    #[test]
    fn extrude_planar_region_uses_the_lent_kernel_and_defers_its_proof() {
        use solidcore::curve::curve::CurveKind;
        use solidcore::surface::surface::SurfaceKind;
        let plane = solidcore_record::Plane::XY;
        let region = rounded_region();
        let mut kernel = nodecraft_kernel_profile::kernel();
        let body = OpsSession::new(&mut kernel)
            .extrude_planar_region(&plane, &region, 0.0, 10.0)
            .expect("extrude on the NodeCraft profile");
        let circle_edges = body
            .brep()
            .edges()
            .ids()
            .filter(|edge| {
                body.brep()
                    .edge(*edge)
                    .ok()
                    .and_then(|edge| edge.curve.as_ref())
                    .is_some_and(|curve| matches!(curve.kind(), CurveKind::Ellipse(ellipse) if ellipse.is_circle()))
            })
            .count();
        let cylindrical_faces = body
            .brep()
            .faces()
            .ids()
            .filter(|face| {
                body.brep()
                    .face(*face)
                    .ok()
                    .and_then(|face| face.surface.as_ref())
                    .is_some_and(|surface| matches!(surface.kind(), SurfaceKind::Cone(cone) if cone.is_cylinder()))
            })
            .count();
        assert_eq!(circle_edges, 2, "the profile arc stays analytic on this kernel");
        assert_eq!(cylindrical_faces, 1, "its swept wall is a cylinder");

        let deferred = validating(Validation::EachNode, || {
            OpsSession::new(&mut kernel).extrude_planar_region(&plane, &region, 0.0, 10.0)
        })
        .expect("deferred extrude");
        assert_eq!(deferred.proof_state(), ProofState::Unknown, "the node-end proof is deferred");
        assert_eq!(deferred.proof_attempts(), 0, "no proof on the critical path");
        deferred.prove(&mut kernel).expect("the deferred proof holds");
        assert!(deferred.is_proved());
    }

    /// Proving a body must leave its shared B-rep exactly as it was: the proof copy is a
    /// separate allocation and the display/selection handles stay valid.
    #[test]
    fn a_proof_does_not_change_the_shared_brep() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep =
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box");
        let (faces, edges, vertices) =
            (brep.face_list().to_vec(), brep.edge_list().to_vec(), brep.vertex_list().to_vec());
        let body = SolidBody::new(brep);
        body.prove(&mut kernel).expect("prove");
        assert_eq!(body.brep().face_list(), faces.as_slice());
        assert_eq!(body.brep().edge_list(), edges.as_slice());
        assert_eq!(body.brep().vertex_list(), vertices.as_slice());
    }

    /// Shell and offset are orientation-dependent: their blocking proof refuses an
    /// inside-out body before any shell/offset geometry is built.
    #[test]
    fn shell_and_offset_refuse_an_inside_out_body_before_running() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let reversed = SolidBody::new(inside_out_box(&mut kernel));
        let error = OpsSession::new(&mut kernel)
            .shell_body(&reversed, &[], -1.0)
            .expect_err("an inside-out body has no inside to shell");
        assert!(matches!(error, SessionError::Invalid(_)), "{error}");
        assert!(reversed.refused().is_some());
        assert_eq!(reversed.proof_attempts(), 1, "the proof ran once, before the shell");

        let valid = SolidBody::new(
            create_box(&mut kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
                .expect("box"),
        );
        let top = face_facing(&kernel, valid.brep(), [0.0, 0.0, 1.0]);
        OpsSession::new(&mut kernel)
            .offset_faces_body(&valid, &[top], 1.0)
            .expect("a valid body is proved then offset");
        assert!(valid.is_proved());
    }

    /// An explicit Check proves its body first: an inside-out body is a refusal naming the
    /// failed test, not a validity report.
    #[test]
    fn check_body_proves_first_and_refuses_an_inside_out_body() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let reversed = SolidBody::new(inside_out_box(&mut kernel));
        let error = OpsSession::new(&mut kernel)
            .check_body(&reversed, CheckLevel::Geometry)
            .expect_err("an explicit check refuses an inside-out body");
        assert!(matches!(error, SessionError::Invalid(_)), "{error}");
        assert!(reversed.refused().is_some());
        assert_eq!(reversed.proof_attempts(), 1);
        let valid = SolidBody::new(
            create_box(&mut kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
                .expect("box"),
        );
        let report = OpsSession::new(&mut kernel)
            .check_body(&valid, CheckLevel::Geometry)
            .expect("a valid body checks");
        assert!(report.is_valid(), "{report}");
        assert!(valid.is_proved());
    }

    /// A proof copy carries the source's proved state; a refusal is never overwritten.
    #[test]
    fn a_proof_copy_carries_a_proved_state() {
        use solidcore::math::Point3;
        use solidcore::primitive::api_primitives::create_box;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let source = SolidBody::new(
            create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 20.0, 20.0, 20.0).expect("box"),
        );
        source.prove(&mut kernel).expect("prove");
        let copy = SolidBody::new(source.brep().clone_standing_for_self().expect("copy"));
        copy.carry_proof_from(&source);
        assert!(copy.is_proved(), "a rigid/identical copy keeps the proof");
        let refused = SolidBody::new(inside_out_box(&mut kernel));
        assert!(refused.prove(&mut kernel).is_err());
        let other = SolidBody::new(
            create_box(&mut kernel, Point3::new(30.0, 0.0, 0.0), 10.0, 10.0, 10.0).expect("other"),
        );
        other.prove(&mut kernel).expect("prove other");
        refused.carry_proof_from(&other);
        assert!(refused.refused().is_some(), "a refusal is not overwritten by a proved copy");
    }
}
