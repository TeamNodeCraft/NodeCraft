//! Owned exact-node tasks.
//!
//! A task is everything a worker kernel needs: no `&mut Eval`, no caches, no
//! graph borrows. The coordinator keeps cache/error ownership.

use crate::kernel_pool::shared_pool;
use nodecraft_ops::SolidBody;
use nodecraft_protocol::Node;
use nodecraft_solidcore::solidcore::{
    kernel::{CancelToken, Kernel},
    topology::Brep,
};


/// Typed value crossing the node boundary. Node ids are always flat graph ids;
/// UI nesting is presentation only.
#[derive(Clone)]
#[allow(dead_code)] // Part/Sketch/Mesh outputs are the planned typed-input path.
pub(crate) enum NodeOutput {
    Shape(SolidBody),
    Shapes(Vec<SolidBody>),
    Part(std::sync::Arc<crate::Part>),
    Sketch(std::sync::Arc<crate::SketchOutput>),
    Mesh(std::sync::Arc<nodecraft_protocol::Mesh>),
    Empty,
}

#[allow(dead_code)]
impl NodeOutput {
    pub(crate) fn into_shape(self) -> Option<SolidBody> {
        match self {
            Self::Shape(shape) => Some(shape),
            _ => None,
        }
    }

    pub(crate) fn into_shapes(self) -> Option<Vec<SolidBody>> {
        match self {
            Self::Shapes(shapes) => Some(shapes),
            _ => None,
        }
    }

    pub(crate) fn into_brep(self) -> Option<Brep> {
        match self {
            Self::Shape(shape) => shape.into_brep().ok(),
            _ => None,
        }
    }

    /// The shared B-rep of a shape output, for the display paths that keep `Arc<Brep>`.
    pub(crate) fn into_shared_brep(self) -> Option<std::sync::Arc<Brep>> {
        match self {
            Self::Shape(shape) => Some(shape.shared()),
            _ => None,
        }
    }

    pub(crate) fn into_sketch(self) -> Option<std::sync::Arc<crate::SketchOutput>> {
        match self {
            Self::Sketch(sketch) => Some(sketch),
            _ => None,
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Shape(_) => "shape",
            Self::Shapes(_) => "shapes",
            Self::Part(_) => "part",
            Self::Sketch(_) => "sketch",
            Self::Mesh(_) => "mesh",
            Self::Empty => "empty",
        }
    }
}

/// A node task's output plus any user-facing warnings it produced.
pub(crate) struct TaskResult {
    pub output: NodeOutput,
    pub warnings: Vec<String>,
}

impl From<NodeOutput> for TaskResult {
    fn from(output: NodeOutput) -> Self {
        Self { output, warnings: Vec::new() }
    }
}

/// Inputs are collected by the flat graph's input port names.
#[allow(dead_code)] // Planned typed-input path for scheduler dispatch.
pub(crate) struct NodeInputs {
    values: std::collections::BTreeMap<String, NodeOutput>,
}

#[allow(dead_code)]
impl NodeInputs {
    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(crate) fn new() -> Self {
        Self { values: std::collections::BTreeMap::new() }
    }

    pub(crate) fn insert(&mut self, port: &str, value: NodeOutput) {
        self.values.insert(port.to_owned(), value);
    }

    pub(crate) fn get(&self, port: &str) -> Option<&NodeOutput> {
        self.values.get(port)
    }
}

/// Exact work that can execute on a single kernel worker.
pub(crate) enum ExactNodeTask {
    Primitive(Node),
    Boolean {
        op: nodecraft_bool::BooleanOp,
        a: Vec<SolidBody>,
        b: Vec<SolidBody>,
    },
    Extrude {
        body: SolidBody,
        face_index: usize,
        amount: f64,
    },
    ExtrudeSketch {
        output: std::sync::Arc<crate::SketchOutput>,
        amount: f64,
    },
    Translate {
        brep: Brep,
        offset: [f64; 3],
        rotation: [f64; 3],
    },
    Fillet {
        body: SolidBody,
        values: serde_json::Value,
    },
    Shell {
        body: SolidBody,
        signed_thickness: f64,
        openings: Vec<usize>,
    },
    Mirror {
        brep: Brep,
        values: serde_json::Value,
    },
    LoftSketch {
        bottom: crate::SketchOutput,
        top: crate::SketchOutput,
        ruled: bool,
    },
    Chamfer {
        body: SolidBody,
        values: serde_json::Value,
    },
    SurfaceLoft {
        sections: Vec<crate::SketchOutput>,
        ruled: bool,
    },
    BooleanLists {
        op: nodecraft_bool::BooleanOp,
        a: Vec<SolidBody>,
        b: Vec<SolidBody>,
    },
    ExtrudeFeature {
        body: SolidBody,
        face_index: usize,
        amount: f64,
        operation: String,
    },
    OffsetFaces {
        body: SolidBody,
        faces: Vec<usize>,
        distance: f64,
    },
    Look {
        brep: std::sync::Arc<Brep>,
        colour: [f64; 3],
    },
    LookMany {
        breps: Vec<std::sync::Arc<Brep>>,
        colour: [f64; 3],
    },
    Realize {
        body: SolidBody,
        transforms: Vec<crate::BodyTransform>,
    },
    ImportStep {
        path: String,
    },
    FaceLoft {
        bottom: Brep,
        top: Brep,
        ruled: bool,
        cap_ends: bool,
    },
}

/// Import one CAD file as a multi-body Part through the single exact import
/// route. Warnings name every refused body; the Part keeps the bodies that
/// converted, in CADIR order and with their names and colours.
pub(crate) fn import_step_part(path: &str) -> Result<TaskResult, String> {
    let import = nodecraft_mesh::import_step_breps(path)?;
    let (part, color_warnings) = crate::Part::from_imported(import.bodies);
    let mut warnings = import.warnings;
    warnings.extend(color_warnings);
    Ok(TaskResult {
        output: NodeOutput::Part(std::sync::Arc::new(part)),
        warnings,
    })
}

impl ExactNodeTask {
    /// Node types this task layer can dispatch today. The descriptor table
    /// remains the long-term capability source; this is the implemented subset.
    #[allow(dead_code)] // Implemented subset; descriptor table remains the authority.
    pub(crate) fn supports(node_type: &str) -> bool {
        matches!(node_type, "solid" | "primitive" | "boolean" | "fuse" | "extrude" | "fillet" | "shell" | "mirror" | "loft" | "surfaceloft" | "chamfer")
    }

    pub(crate) fn run(self, kernel: &mut Kernel) -> Result<NodeOutput, String> {
        match self {
            Self::Primitive(node) => crate::primitive_brep_with_kernel(kernel, &node)
                .map(|brep| NodeOutput::Shape(SolidBody::new(brep))),
            Self::Translate { mut brep, offset, rotation } => {
                crate::apply_brep_placement_with_kernel(kernel, &mut brep, offset, rotation)?;
                Ok(NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::ExtrudeSketch { output, amount } => {
                crate::sketch_extrude_brep(kernel, &output, amount).map(NodeOutput::Shape)
            }
            Self::Extrude { body, face_index, amount } => {
                // Extrusion follows the face's outward normal, so an unknown or inside-out
                // body must be decided before the geometry is built.
                body.prove(kernel).map_err(|error| format!("extrude: {error}"))?;
                let face = body
                    .brep()
                    .faces()
                    .ids()
                    .nth(face_index)
                    .ok_or_else(|| format!("Extrude face {face_index} does not exist"))?;
                nodecraft_extrude::extrude_planar_face_exact(kernel, body.brep(), face, amount)
                    .map(|brep| NodeOutput::Shape(SolidBody::new(brep)))
                    .map_err(|error| format!("extrude: {error}"))
            }
            Self::Chamfer { body, values } => {
                let (brep, _) = run_edge_session(kernel, &body, &values)?;
                Ok(NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::SurfaceLoft { sections, ruled } => {
                crate::multi_section_surface_loft(kernel, &sections, ruled, false)
                    .map(|brep| NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::LoftSketch { bottom, top, ruled } => {
                crate::loft_sketch(kernel, &bottom, &top, ruled, true)
                    .map(|brep| NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::Mirror { brep, values } => {
                let mut brep = brep;
                crate::mirror_brep_with_kernel(kernel, &mut brep, &values)
                    .map(|()| NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::Shell { body, signed_thickness, openings } => {
                // A shell offsets faces along their outward normals: prove the input first.
                let mut scratch = body
                    .brep()
                    .clone_standing_for_self()
                    .map_err(|error| format!("Shell: {error}"))?;
                let faces = nodecraft_layers::record_face_ids(kernel, &mut scratch, &openings)
                    .map_err(|error| format!("Shell: {error}"))?;
                let result = nodecraft_ops::OpsSession::new(kernel)
                    .shell_body(&body, &faces, signed_thickness)
                    .map_err(|error| format!("Shell: {error}"))?;
                Ok(NodeOutput::Shape(SolidBody::new(result.brep)))
            }
            Self::Fillet { body, values } => {
                let (brep, _) = run_edge_session(kernel, &body, &values)?;
                Ok(NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::BooleanLists { op, a, b } => {
                let (results, _warnings) = boolean_solids(kernel, &a, &b, op)?;
                Ok(NodeOutput::Shapes(results))
            }
            Self::ExtrudeFeature { body, face_index, amount, operation } => {
                body.prove(kernel).map_err(|error| format!("extrude: {error}"))?;
                let face = body
                    .brep()
                    .faces()
                    .ids()
                    .nth(face_index)
                    .ok_or_else(|| format!("Extrude face {face_index} does not exist"))?;
                let tool = nodecraft_extrude::extrude_planar_face_exact(kernel, body.brep(), face, amount)
                    .map_err(|error| format!("extrude: {error}"))?;
                let op = match operation.as_str() {
                    "remove" => nodecraft_bool::BooleanOp::Difference,
                    "intersect" => nodecraft_bool::BooleanOp::Intersection,
                    _ => nodecraft_bool::BooleanOp::Union,
                };
                let result = crate::boolean_breps_ops_with_kernel(kernel, body.brep(), &tool, op)?;
                Ok(NodeOutput::Shape(SolidBody::new(result)))
            }
            Self::OffsetFaces { body, faces, distance } => {
                // Offsets move faces along their outward normals: prove the input first.
                let mut scratch = body
                    .brep()
                    .clone_standing_for_self()
                    .map_err(|error| format!("Offset Faces: {error}"))?;
                let ids = nodecraft_layers::record_face_ids(kernel, &mut scratch, &faces)
                    .map_err(|error| format!("Offset Faces: {error}"))?;
                let result = nodecraft_ops::OpsSession::new(kernel)
                    .offset_faces_body(&body, &ids, distance)
                    .map_err(|error| format!("Offset Faces: {error}"))?;
                Ok(NodeOutput::Shape(SolidBody::new(result.brep)))
            }
            Self::Look { brep, colour } => {
                let mut brep = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(kernel, &brep)
                    .map_err(|error| error.to_string())?;
                crate::look::apply_look_color(&mut brep, colour)?;
                Ok(NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::LookMany { breps, colour } => {
                let mut out = Vec::with_capacity(breps.len());
                for brep in breps {
                    let mut brep = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(kernel, &brep)
                        .map_err(|error| error.to_string())?;
                    crate::look::apply_look_color(&mut brep, colour)?;
                    out.push(SolidBody::new(brep));
                }
                Ok(NodeOutput::Shapes(out))
            }
            Self::Realize { body, transforms } => {
                let mut brep = nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy(kernel, body.brep())
                    .map_err(|error| error.to_string())?;
                for transform in &transforms {
                    match transform {
                        crate::BodyTransform::Place { offset, rotation } => {
                            crate::apply_brep_placement_with_kernel(kernel, &mut brep, *offset, *rotation)?;
                        }
                        crate::BodyTransform::Mirror { plane, offset } => {
                            let values = serde_json::json!({"plane": plane, "offset": offset});
                            crate::mirror_brep_with_kernel(kernel, &mut brep, &values)?;
                        }
                        crate::BodyTransform::Scale { factor } => crate::scale_brep(&mut brep, *factor)?,
                    }
                }
                Ok(NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::ImportStep { path } => {
                import_step_part(&path).map(|result| result.output)
            }
            Self::FaceLoft { bottom, top, ruled, cap_ends } => {
                crate::loft_breps(kernel, bottom, top, ruled, cap_ends)
                    .map(|brep| NodeOutput::Shape(SolidBody::new(brep)))
            }
            Self::Boolean { op, a, b } => {
                let (results, _warnings) = boolean_solids(kernel, &a, &b, op)?;
                boolean_shape(kernel, results)
            }
        }
    }
}

/// The warning a Boolean node shows when its operation leaves nothing.
pub(crate) const EMPTY_BOOLEAN_WARNING: &str = "The result is empty: the shapes don't overlap";

/// Map NodeCraft's Boolean selector to the ops one.
fn ops_boolean(op: nodecraft_bool::BooleanOp) -> nodecraft_ops::BooleanOp {
    match op {
        nodecraft_bool::BooleanOp::Union => nodecraft_ops::BooleanOp::Union,
        nodecraft_bool::BooleanOp::Difference => nodecraft_ops::BooleanOp::Subtract,
        nodecraft_bool::BooleanOp::Intersection => nodecraft_ops::BooleanOp::Intersect,
    }
}

/// Standalone copies of solids, for handing them to ops. A [`SolidBody`] clone is cheap and
/// shares its proved kernel body, so a body that feeds several consumers is not re-proved by
/// each.
fn copy_solids(solids: &[SolidBody]) -> Vec<SolidBody> {
    solids.to_vec()
}

/// One side of a Boolean: a Part's solids as one body.
///
/// A side holding several solids is a Combine, and the user means the Part as a whole, so
/// its pieces are unioned before the operation applies to them. `None` is an empty side.
fn union_side(session: &mut nodecraft_ops::OpsSession<'_>, solids: &[SolidBody]) -> Result<Option<SolidBody>, String> {
    match solids {
        [] => Ok(None),
        [solid] => Ok(Some(solid.clone())),
        solids => session
            .union_all(solids)
            .map(Some)
            .map_err(|error| format!("Boolean: {error}")),
    }
}

/// Boolean two lists of solids through `OpsSession` on this worker kernel.
///
/// Each side is a Part. For a union both sides merge in **one** `union_all`, level by level,
/// rather than one Boolean per piece (a long union chain such as forest's then costs one
/// tree merge per node instead of one per operand piece). For a Subtract or Intersect the B
/// side is unioned into one body first and the operation runs per A solid:
/// - `A − (b₁ ∪ b₂)`, not `A − b₁ − b₂` (the same thing),
/// - `A ∩ (b₁ ∪ b₂)`, **not** `A ∩ b₁ ∩ b₂`: a Combine of two apart solids intersects A
///   with their union, where sequential intersection would wrongly return nothing.
///
/// Separate A solids stay separate results: this deliberately restores the
/// original list Boolean (dcb094b44; see agents/review/2026-11-13-p3-review.md
/// §2), not the later merged single shape. For Difference/Intersect overlapping
/// A pieces therefore stay separate (not merged), while Union returns one merged
/// body. An empty list means the operation left nothing: a warning on the node,
/// not an error.
fn boolean_solids(
    kernel: &mut Kernel,
    a: &[SolidBody],
    b: &[SolidBody],
    op: nodecraft_bool::BooleanOp,
) -> Result<(Vec<SolidBody>, Vec<String>), String> {
    if a.is_empty() {
        return Err("Boolean operand A is empty".to_owned());
    }
    let mut session = nodecraft_ops::OpsSession::new(kernel);
    if matches!(op, nodecraft_bool::BooleanOp::Union) {
        let (a_list, b_list) = (copy_solids(a), copy_solids(b));
        if a_list.is_empty() && b_list.is_empty() {
            return Err("Boolean operand A is empty".to_owned());
        }
        let body = match (a_list.len(), b_list.len()) {
            (1, 0) => a_list.into_iter().next().expect("one solid"),
            (0, 1) => b_list.into_iter().next().expect("one solid"),
            // Keep the two lists apart for the n-ary union: the kernel merges the B tools
            // into A, which is cheaper than appending them to A. `boolean_lists` needs both
            // sides; a one-sided run is `union_all`.
            _ => {
                let mut results = if a_list.is_empty() || b_list.is_empty() {
                    let all: Vec<_> = a_list.into_iter().chain(b_list).collect();
                    vec![session
                        .union_all(&all)
                        .map_err(|error| format!("Boolean: {error}"))?]
                } else {
                    session
                        .boolean_lists(&a_list, &b_list, nodecraft_ops::BooleanOp::Union)
                        .map_err(|error| format!("Boolean: {error}"))?
                };
                match results.len() {
                    1 => results.pop().expect("one solid"),
                    // A list union normally merges into one body; if the kernel returns
                    // separate pieces, merge them as the node's single output.
                    _ => session
                        .union_all(&results)
                        .map_err(|error| format!("Boolean: {error}"))?,
                }
            }
        };
        return Ok((vec![body], Vec::new()));
    }
    // A multi-tool Difference/Intersection is one n-ary list Boolean rather than a union of
    // the tools followed by a per-A cut: the kernel's list merge is the same operation with
    // one merge tree, and it is what the direct path used.
    if b.len() > 1 {
        let mut results = session
            .boolean_lists(a, b, ops_boolean(op))
            .map_err(|error| format!("Boolean: {error}"))?;
        if results.is_empty() {
            return Ok((results, vec![EMPTY_BOOLEAN_WARNING.to_owned()]));
        }
        return Ok((results, Vec::new()));
    }
    let Some(b_side) = union_side(&mut session, b)? else {
        // No B solids. Subtract keeps A (A − ∅ = A); Intersect has nothing to
        // keep (A ∩ ∅ = ∅). Union already merged both sides above.
        return if matches!(op, nodecraft_bool::BooleanOp::Intersection) {
            Ok((Vec::new(), vec![EMPTY_BOOLEAN_WARNING.to_owned()]))
        } else {
            Ok((copy_solids(a), Vec::new()))
        };
    };
    let mut results = Vec::with_capacity(a.len());
    for piece in a {
        if let Some(result) = session
            .boolean_outcome(piece, &b_side, ops_boolean(op))
            .map_err(|error| format!("Boolean: {error}"))?
        {
            results.push(result);
        }
    }
    if results.is_empty() {
        return Ok((results, vec![EMPTY_BOOLEAN_WARNING.to_owned()]));
    }
    Ok((results, Vec::new()))
}

/// A single-result Boolean from the per-solid results: none is empty, one is a shape, and
/// several (separate A solids that survived) merge into one body for the display path, as
/// the list Boolean's extra shapes were merged before.
fn boolean_shape(kernel: &mut Kernel, results: Vec<SolidBody>) -> Result<NodeOutput, String> {
    match results.len() {
        0 => Ok(NodeOutput::Empty),
        1 => Ok(NodeOutput::Shape(results.into_iter().next().expect("one solid"))),
        _ => nodecraft_ops::OpsSession::new(kernel)
            .union_all(&results)
            .map(NodeOutput::Shape)
            .map_err(|error| format!("Boolean: {error}")),
    }
}

/// Execute an owned exact task on the shared persistent-kernel pool.
/// Fillet/chamfer the node's selected edges through `OpsSession` on this worker kernel.
fn run_edge_session(
    kernel: &mut Kernel,
    source: &SolidBody,
    values: &serde_json::Value,
) -> Result<(Brep, Vec<String>), String> {
    // A blend is built from the faces' outward directions: the kernel classifies each edge
    // by the turn between its two faces, offsets each side along that face's outward normal,
    // and places the rolling-ball section on that side (solidcore-ops `blend.rs`). Ops'
    // `Expect::of` only decides closedness under `EachNode`, so without a blocking proof an
    // inside-out body reaches the blend and is treated as a face blend. Prove through the
    // same shared helper Extrude/shell/offset use before any selection or geometry runs; a
    // refused body is remembered and named.
    source.prove(kernel).map_err(|error| format!("blend: {error}"))?;
    let source = source.brep();
    use nodecraft_blend::treatment::{ChamferType, Continuity, EdgeMode};
    let treatment = nodecraft_blend::treatment::EdgeTreatment::from_node_values(values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    treatment
        .validate()
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    // G3 has no ops section yet; it keeps the direct kernel path.
    if treatment.mode == EdgeMode::Fillet && treatment.continuity == Continuity::G3 {
        let blended = nodecraft_blend::fillet_brep_exact_with_kernel_warned(kernel, source, values)?;
        return Ok((blended.brep, blended.warnings));
    }
    let selections = nodecraft_blend::edge::select_edges_from_values(kernel, source, values)
        .map_err(|error| nodecraft_solidcore::error::user_message(error))?;
    // Chamfer and fillet both declare `tangentChain`; resolving it through
    // either field set gives the same contract default and stored encoding.
    let selections = if crate::contract_bool("fillet", values, "tangentChain") {
        nodecraft_blend::edge::tangent_chain(kernel, source, &selections, 0.5_f64.to_radians())
            .map_err(|error| nodecraft_solidcore::error::user_message(error))?
    } else {
        selections
    };
    let edges: Vec<_> = selections.iter().map(|selection| selection.edge).collect();
    let Some((kind, size)) = treatment.to_ops_blend() else {
        unreachable!("G3 returned above");
    };
    let result = nodecraft_ops::OpsSession::new(kernel)
        .blend(source, &edges, size, kind)
        .map_err(|error| error.to_string())?;
    Ok((result.brep, result.warnings))
}

impl ExactNodeTask {
    /// Run a task that can produce warnings (fillet/chamfer, and a Boolean whose result
    /// is empty); every other task is delegated to `run` and has no warnings.
    fn run_warned(self, kernel: &mut Kernel) -> Result<TaskResult, String> {
        match self {
            Self::ImportStep { path } => import_step_part(&path),
            Self::Fillet { body, values } => {
                let (brep, warnings) = run_edge_session(kernel, &body, &values)?;
                Ok(TaskResult {
                    output: NodeOutput::Shape(SolidBody::new(brep)),
                    warnings,
                })
            }
            Self::Chamfer { body, values } => {
                let (brep, warnings) = run_edge_session(kernel, &body, &values)?;
                Ok(TaskResult {
                    output: NodeOutput::Shape(SolidBody::new(brep)),
                    warnings,
                })
            }
            Self::BooleanLists { op, a, b } => {
                let (results, warnings) = boolean_solids(kernel, &a, &b, op)?;
                Ok(TaskResult {
                    output: NodeOutput::Shapes(results),
                    warnings,
                })
            }
            Self::Boolean { op, a, b } => {
                // The proof is not on the evaluation path: ops runs under `EachNode` (see
                // `execute_exact_warned`) and the output stays unknown for the background
                // proof queue. The kernel Boolean and its closedness test still run here.
                let (results, warnings) = boolean_solids(kernel, &a, &b, op)?;
                let output = boolean_shape(kernel, results)?;
                Ok(TaskResult { output, warnings })
            }
            other => other.run(kernel).map(TaskResult::from),
        }
    }
}

pub(crate) fn execute_exact_warned(
    task: ExactNodeTask,
    _node_id: &str,
    seed: u64,
) -> Result<TaskResult, String> {
    // Every ops call the engine makes during evaluation runs under `EachNode`: the cheap
    // kernel status and closedness test still run (an open or failed result is still an
    // error), but no validity/volume-sign proof is taken. The background proof queue (or a
    // blocking caller: export, check, an orientation-dependent operation) proves each body
    // once, off this path. Direct `OpsSession` callers keep `EachStep`.
    shared_pool().execute(seed, move |kernel| {
        nodecraft_ops::validating(nodecraft_ops::Validation::EachNode, || task.run_warned(kernel))
    })
}

pub(crate) fn execute_exact(
    task: ExactNodeTask,
    _node_id: &str,
    seed: u64,
) -> Result<NodeOutput, String> {
    execute_exact_warned(task, _node_id, seed).map(|result| result.output)
}

/// Revision cancellation token for a task group.
#[derive(Clone)]
#[allow(dead_code)] // Planned cancellation path for scheduler dispatch.
pub(crate) struct TaskCancellation {
    pub token: CancelToken,
}

#[allow(dead_code)]
impl TaskCancellation {
    pub(crate) fn new() -> Self {
        Self { token: CancelToken::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn primitive(length: f64) -> Node {
        Node {
            id: "p".into(),
            node_type: "solid".into(),
            values: json!({"primitiveType": "box", "length": length, "width": 10.0, "height": 10.0}),
            suppressed: false,
            cache: None,
        }
    }

    /// An inside-out box: every face's orientation flipped. It stays manifold,
    /// so only the proof can tell it is not an outward solid.
    fn inside_out_box(kernel: &mut nodecraft_solidcore::solidcore::kernel::Kernel) -> Brep {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;
        use nodecraft_solidcore::solidcore::topology::brep::p3::brep_copy;
        use nodecraft_solidcore::solidcore::topology::face::p1::flip_face_orientation;
        let boxed = create_box(kernel, Point3::new(-10.0, -10.0, -10.0), 20.0, 20.0, 20.0)
            .expect("box");
        let mut brep = brep_copy(kernel, &boxed).expect("copy");
        for face in brep.face_list().to_vec() {
            flip_face_orientation(&mut brep, face).expect("flip");
        }
        brep
    }

    /// A blend is orientation-dependent (turn classification, offset side and
    /// rolling-ball side), so an inside-out input must be refused by the same
    /// blocking proof shell/offset/extrude use, before edge selection or any
    /// blend geometry runs.
    #[test]
    fn fillet_refuses_an_inside_out_body_before_running() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let brep = inside_out_box(&mut kernel);
        let edge = nodecraft_blend::edge::edge_selections(&kernel, &brep)
            .expect("edge selections")
            .into_iter()
            .next()
            .expect("a box edge");
        let body = SolidBody::new(brep);
        let values = json!({
            "mode": "fillet",
            "continuity": "circular",
            "length": 1.0,
            "edges": [{"center": edge.midpoint}],
        });
        let error = run_edge_session(&mut kernel, &body, &values)
            .expect_err("an inside-out body must be refused before the blend");
        assert!(error.contains("not a closed solid"), "the proof names the refusal: {error}");
        assert!(body.refused().is_some(), "the refusal is remembered");
        assert_eq!(body.proof_attempts(), 1, "the proof ran once, before the blend");
    }

    #[test]
    fn primitive_task_matches_direct_primitive() {
        let node = primitive(20.0);
        let direct = crate::primitive_brep(&node).expect("direct");
        let task = exact_via_pool(node, 7);
        let task = task.into_shape().expect("shape output");
        assert_eq!(direct.face_list().len(), task.brep().face_list().len());
        assert_eq!(direct.edge_list().len(), task.brep().edge_list().len());
    }

    fn exact_via_pool(node: Node, seed: u64) -> NodeOutput {
        execute_exact(ExactNodeTask::Primitive(node), "p", seed).expect("task")
    }

    #[test]
    fn node_output_keeps_typed_values_for_flat_node_ids() {
        let inputs = NodeInputs::new();
        assert!(inputs.get("shape").is_none());
        assert_eq!(NodeOutput::Empty.kind(), "empty");
    }

    #[test]
    fn supported_types_are_limited_to_implemented_entry_points() {
        assert!(ExactNodeTask::supports("solid"));
        assert!(ExactNodeTask::supports("boolean"));
        assert!(ExactNodeTask::supports("fuse"));
        assert!(ExactNodeTask::supports("extrude"));
        assert!(ExactNodeTask::supports("fillet"));
    }

    #[test]
    fn intersect_with_empty_b_is_empty_and_warns() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let a = vec![SolidBody::new(crate::primitive_brep(&primitive(10.0)).expect("box A"))];
        let (results, warnings) =
            boolean_solids(&mut kernel, &a, &[], nodecraft_bool::BooleanOp::Intersection)
                .expect("intersect with empty B");
        assert!(results.is_empty(), "A ∩ ∅ must be empty");
        assert_eq!(warnings, vec![EMPTY_BOOLEAN_WARNING.to_owned()]);
    }

    #[test]
    fn subtract_with_empty_b_keeps_a() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let a = vec![SolidBody::new(crate::primitive_brep(&primitive(10.0)).expect("box A"))];
        let (results, warnings) =
            boolean_solids(&mut kernel, &a, &[], nodecraft_bool::BooleanOp::Difference)
                .expect("subtract with empty B");
        assert_eq!(results.len(), 1, "A − ∅ must keep A");
        assert!(warnings.is_empty(), "A − ∅ must not warn");
    }

    /// `EachNode` proofs only what the operation expected to be a solid: a Boolean of two
    /// sheets stays a sheet under both modes and is not refused by the node-end proof.
    #[test]
    fn each_node_boolean_of_sheets_is_allowed() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_plane;
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let a = create_plane(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0).expect("sheet A");
        let b = create_plane(&mut kernel, Point3::new(5.0, 0.0, 0.0), 10.0, 10.0).expect("sheet B");
        let task = ExactNodeTask::Boolean {
            op: nodecraft_bool::BooleanOp::Union,
            a: vec![SolidBody::new(a)],
            b: vec![SolidBody::new(b)],
        };
        let result = task
            .run_warned(&mut kernel)
            .expect("a sheet Boolean behaves as under EachStep");
        let NodeOutput::Shape(body) = result.output else {
            panic!("a sheet Boolean has a shape output");
        };
        assert_eq!(body.brep().face_list().len(), 1, "the union is still a sheet");
    }

    /// Engine evaluation leaves the proof of a solid Boolean to the background queue: the
    /// kernel Boolean and its closedness test run, but the output is Unknown with no proof
    /// attempt. Without the change ops ran under EachStep and the node end called `prove_on`,
    /// so the output arrived Proved.
    #[test]
    fn engine_evaluation_leaves_the_output_proof_to_the_background_queue() {
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let a = SolidBody::new(crate::primitive_brep(&primitive(10.0)).expect("box A"));
        let b = SolidBody::new(crate::primitive_brep(&primitive(10.0)).expect("box B"));
        let task = ExactNodeTask::Boolean {
            op: nodecraft_bool::BooleanOp::Union,
            a: vec![a],
            b: vec![b],
        };
        let output = execute_exact(task, "boolean", 7).expect("solid Boolean");
        let NodeOutput::Shape(body) = output else {
            panic!("a solid Boolean has a shape output");
        };
        assert_eq!(
            body.proof_state(),
            nodecraft_ops::ProofState::Unknown,
            "no proof on the evaluation path"
        );
        assert_eq!(body.proof_attempts(), 0, "evaluation did not touch the proof");
        body.prove(&mut kernel).expect("the deferred proof holds");
        assert!(body.is_proved());
    }
}
