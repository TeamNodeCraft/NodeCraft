//! Contract and template for evaluation nodes that produce exact geometry.
//!
//! A node result has two channels:
//!
//! - `mesh`: display/selection output.
//! - `exact`: optional evaluated B-rep or B-reps.
//!
//! Downstream exact consumers must use `exact`, never rebuild it by walking the
//! graph. If a node can produce an exact solid, it must populate `exact` on every
//! evaluation, including cache hits.
//!
//! # Template
//!
//! ```ignore
//! use nodecraft_engine::node_plugin::{ExactGeometry, NodePlugin, NodeResult};
//! use nodecraft_protocol::Node;
//!
//! struct MyNode;
//!
//! impl NodePlugin for MyNode {
//!     const NODE_TYPE: &'static str = "myNode";
//!
//!     fn evaluate(&self, _node: &Node, inputs: &[NodeResult]) -> Result<NodeResult, String> {
//!         let Some(input) = inputs.first() else {
//!             return Err("myNode needs an input".into());
//!         };
//!         let Some(brep) = input.exact.first() else {
//!             return Err("myNode requires an exact input".into());
//!         };
//!
//!         // Build the node's display mesh here.
//!         let mesh = input.mesh.clone();
//!
//!         // Keep the exact result in the same node output. Do not make the
//!         // exporter reconstruct it from the request graph.
//!         Ok(NodeResult::new(mesh).with_brep(brep.clone()))
//!     }
//! }
//! ```

use std::sync::Arc;

use nodecraft_protocol::{Mesh, Node};
use nodecraft_solidcore::solidcore::topology::Brep;

/// The exact geometry channel of a node output.
#[derive(Clone, Default)]
pub enum ExactGeometry {
    /// No exact body (for example a sketch or a display-only node).
    #[default]
    None,
    /// One exact solid body.
    Brep(Arc<Brep>),
    /// Multiple exact bodies, such as a Combine/Part result.
    Breps(Vec<Arc<Brep>>),
}

impl ExactGeometry {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::None => true,
            Self::Brep(_) => false,
            Self::Breps(breps) => breps.is_empty(),
        }
    }

    pub fn first(&self) -> Option<&Arc<Brep>> {
        match self {
            Self::None => None,
            Self::Brep(brep) => Some(brep),
            Self::Breps(breps) => breps.first(),
        }
    }

    pub fn as_slice(&self) -> &[Arc<Brep>] {
        match self {
            Self::None => &[],
            Self::Brep(brep) => std::slice::from_ref(brep),
            Self::Breps(breps) => breps,
        }
    }
}

/// One evaluated node output.
#[derive(Clone, Default)]
pub struct NodeResult {
    pub mesh: Mesh,
    pub exact: ExactGeometry,
}

impl NodeResult {
    pub fn new(mesh: Mesh) -> Self {
        Self { mesh, exact: ExactGeometry::None }
    }

    pub fn with_brep(mut self, brep: Arc<Brep>) -> Self {
        self.exact = ExactGeometry::Brep(brep);
        self
    }

    pub fn with_breps(mut self, breps: Vec<Arc<Brep>>) -> Self {
        self.exact = if breps.len() == 1 {
            ExactGeometry::Brep(breps.into_iter().next().expect("one brep"))
        } else {
            ExactGeometry::Breps(breps)
        };
        self
    }
}

/// Interface a node implementation must satisfy.
///
/// `inputs` are already-evaluated upstream results. A node must pass exact
/// geometry through or replace it with its own exact result; it must never ask
/// an exporter or Boolean to reconstruct the graph.
pub trait NodePlugin {
    const NODE_TYPE: &'static str;

    fn evaluate(&self, node: &Node, inputs: &[NodeResult]) -> Result<NodeResult, String>;
}
