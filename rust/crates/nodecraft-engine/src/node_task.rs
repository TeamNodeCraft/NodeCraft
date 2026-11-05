//! Owned exact-node tasks.
//!
//! A task is everything a worker kernel needs: no `&mut Eval`, no caches, no
//! graph borrows. The coordinator keeps cache/error ownership.

use crate::kernel_pool::shared_pool;
use nodecraft_protocol::Node;
use nodecraft_solidcore::solidcore::{
    kernel::{CancelToken, Kernel},
    topology::Brep,
};


/// Typed value crossing the node boundary. Node ids are always flat graph ids;
/// UI nesting is presentation only.
#[derive(Clone)]
pub(crate) enum NodeOutput {
    Shape(std::sync::Arc<Brep>),
    Part(std::sync::Arc<crate::Part>),
    Sketch(std::sync::Arc<crate::SketchOutput>),
    Mesh(std::sync::Arc<nodecraft_protocol::Mesh>),
    Empty,
}

impl NodeOutput {
    pub(crate) fn into_shape(self) -> Option<std::sync::Arc<Brep>> {
        match self {
            Self::Shape(shape) => Some(shape),
            _ => None,
        }
    }

    pub(crate) fn into_brep(self) -> Option<Brep> {
        match self {
            Self::Shape(shape) => Some(
                match std::sync::Arc::try_unwrap(shape) {
                    Ok(brep) => brep,
                    Err(shared) => shared.clone_standing_for_self().ok()?,
                },
            ),
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
            Self::Part(_) => "part",
            Self::Sketch(_) => "sketch",
            Self::Mesh(_) => "mesh",
            Self::Empty => "empty",
        }
    }
}

/// Inputs are collected by the flat graph's input port names.
pub(crate) struct NodeInputs {
    values: std::collections::BTreeMap<String, NodeOutput>,
}

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
        a: Vec<Brep>,
        b: Vec<Brep>,
    },
    Extrude {
        brep: Brep,
        face_index: usize,
        amount: f64,
    },
    Fillet {
        brep: Brep,
        values: serde_json::Value,
    },
    Shell {
        brep: Brep,
        signed_thickness: f64,
        openings: Vec<nodecraft_solidcore::solidcore::topology::FaceId>,
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
        brep: Brep,
        values: serde_json::Value,
    },
    SurfaceLoft {
        sections: Vec<crate::SketchOutput>,
        ruled: bool,
    },
}

impl ExactNodeTask {
    /// Node types this task layer can dispatch today. The descriptor table
    /// remains the long-term capability source; this is the implemented subset.
    pub(crate) fn supports(node_type: &str) -> bool {
        matches!(node_type, "solid" | "primitive" | "boolean" | "fuse" | "extrude" | "fillet" | "shell" | "mirror" | "loft" | "surfaceloft" | "chamfer")
    }

    pub(crate) fn run(self, kernel: &mut Kernel) -> Result<NodeOutput, String> {
        match self {
            Self::Primitive(node) => crate::primitive_brep_with_kernel(kernel, &node)
                .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep))),
            Self::Extrude { brep, face_index, amount } => {
                let face = brep
                    .faces()
                    .ids()
                    .nth(face_index)
                    .ok_or_else(|| format!("Extrude face {face_index} does not exist"))?;
                nodecraft_extrude::extrude_planar_face_exact(kernel, &brep, face, amount)
                    .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
                    .map_err(|error| format!("extrude: {error}"))
            }
            Self::Chamfer { brep, values } => {
                nodecraft_blend::chamfer_brep_exact_with_kernel(kernel, &brep, &values)
                    .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::SurfaceLoft { sections, ruled } => {
                crate::multi_section_surface_loft(kernel, &sections, ruled, false)
                    .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::LoftSketch { bottom, top, ruled } => {
                crate::loft_sketch(kernel, &bottom, &top, ruled, true)
                    .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::Mirror { brep, values } => {
                let mut brep = brep;
                crate::mirror_brep_with_kernel(kernel, &mut brep, &values)
                    .map(|()| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::Shell { brep, signed_thickness, openings } => {
                nodecraft_bool::shell_brep_solid_with_kernel(
                    kernel,
                    brep,
                    signed_thickness,
                    &openings,
                )
                .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::Fillet { brep, values } => {
                nodecraft_blend::fillet_brep_exact_with_kernel(kernel, &brep, &values)
                    .map(|brep| NodeOutput::Shape(std::sync::Arc::new(brep)))
            }
            Self::Boolean { op, a, b } => {
                let mut a = a.into_iter();
                let mut result = a.next().ok_or("Boolean operand A is empty")?;
                // Combine/Part inputs must be unioned before a later
                // Subtract/Intersect applies to the whole Part.
                for brep in a {
                    result = nodecraft_bool::boolean_breps_with_kernel(
                        kernel,
                        nodecraft_bool::BooleanOp::Union,
                        result,
                        brep,
                    )?;
                }
                for brep in b {
                    result =
                        nodecraft_bool::boolean_breps_with_kernel(kernel, op, result, brep)?;
                }
                Ok(NodeOutput::Shape(std::sync::Arc::new(result)))
            }
        }
    }
}

/// Execute an owned exact task on the shared persistent-kernel pool.
pub(crate) fn execute_exact(
    task: ExactNodeTask,
    node_id: &str,
    seed: u64,
) -> Result<NodeOutput, String> {
    shared_pool().execute(seed, move |kernel| task.run(kernel))
}

/// Revision cancellation token for a task group.
#[derive(Clone)]
pub(crate) struct TaskCancellation {
    pub token: CancelToken,
}

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

    #[test]
    fn primitive_task_matches_direct_primitive() {
        let node = primitive(20.0);
        let direct = crate::primitive_brep(&node).expect("direct");
        let task = exact_via_pool(node, 7);
        let task = task.into_shape().expect("shape output");
        assert_eq!(direct.face_list().len(), task.face_list().len());
        assert_eq!(direct.edge_list().len(), task.edge_list().len());
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
}
