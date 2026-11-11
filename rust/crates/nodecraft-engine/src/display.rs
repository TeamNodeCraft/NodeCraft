//! Display-tessellation boundary types.
//!
//! SolidCore owns exact B-reps. The tessellation worker owns display meshes. The engine
//! describes a display target as a request of local body meshes plus instance
//! transforms; the viewport can draw those instances without merging them.

use crate::kernel_pool::RevisionToken;
use nodecraft_protocol::Mesh;
use std::sync::Arc;

pub use nodecraft_display::{DisplayTransform, PlanInstance, face_colors_by_index};

/// Bump when the meshing algorithm changes so cached meshes are not reused.
pub const TESSELLATION_VERSION: u32 = 1;

/// Stable key for one display mesh.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshKey {
    pub target: String,
    pub content_hash: String,
    pub tolerance_bits: u64,
    pub preview_detail: Option<u8>,
    pub tessellation_version: u32,
}

impl MeshKey {
    #[must_use]
    pub fn new(
        target: impl Into<String>,
        content_hash: impl Into<String>,
        tolerance: f64,
        preview_detail: Option<u8>,
    ) -> Self {
        Self {
            target: target.into(),
            content_hash: content_hash.into(),
            tolerance_bits: tolerance.to_bits(),
            preview_detail,
            tessellation_version: TESSELLATION_VERSION,
        }
    }
}

/// A target that needs a display mesh.
#[derive(Clone, Debug)]
pub struct DisplayTarget {
    pub node_id: String,
    pub key: MeshKey,
    /// True when the target is also an exact Part consumer. The tessellation worker still
    /// only receives B-reps; it never sees this flag.
    pub exact: bool,
}

/// Local-space mesh plus placement for one display instance.
#[derive(Clone, Debug)]
pub struct MeshInstance {
    pub mesh: Arc<Mesh>,
    /// The plan the CPU mesh was evaluated from, when the body has one. The
    /// viewport can evaluate this plan on the GPU; transforms remain separate.
    pub plan: Option<Arc<solidcore_tess::TessPlan>>,
    /// Kernel edge id by record edge index for deriving the CPU mesh from the plan.
    pub plan_edge_ids: Vec<Option<String>>,
    /// Planner diagnostics for this body, when it was just tessellated.
    pub stats: Option<nodecraft_layers::TessStats>,
    pub transforms: Vec<DisplayTransform>,
}

/// A body sent to the tessellation worker. The tessellation worker never mutates the B-rep.
#[derive(Clone, Debug)]
pub struct MeshBody {
    pub brep: Arc<nodecraft_solidcore::solidcore::topology::Brep>,
    pub transforms: Vec<DisplayTransform>,
}

/// Job sent to an independent tessellation worker.
#[derive(Clone, Debug)]
pub struct MeshRequest {
    pub key: MeshKey,
    pub revision: u64,
    pub body: MeshBody,
}

/// Completed display content: local meshes and their instance transforms.
#[derive(Clone, Debug, Default)]
pub struct MeshAsset {
    pub instances: Vec<MeshInstance>,
}

impl MeshAsset {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Compatibility adapter for renderers that still consume one merged mesh.
    #[must_use]
    pub fn merged(&self) -> Mesh {
        let mut meshes = Vec::with_capacity(self.instances.len());
        for instance in &self.instances {
            let mut mesh = (*instance.mesh).clone();
            for transform in &instance.transforms {
                mesh = match transform {
                    DisplayTransform::Place { offset, rotation } => crate::mesh_placement(&mesh, *offset, *rotation),
                    DisplayTransform::Mirror { plane, offset } => crate::mesh_mirror(&mesh, plane, *offset),
                    DisplayTransform::Scale { factor } => crate::mesh_scale(&mesh, *factor),
                };
            }
            meshes.push(mesh);
        }
        if meshes.len() == 1 {
            meshes.pop().expect("one mesh")
        } else {
            nodecraft_mesh::merge_meshes(meshes)
        }
    }
}

/// Cancellation and revision guard for tessellation requests.
#[derive(Clone, Debug)]
pub struct MeshRevision {
    pub revision: u64,
    pub token: RevisionToken,
}

impl MeshRevision {
    #[must_use]
    pub fn new(revision: u64, token: RevisionToken) -> Self {
        Self { revision, token }
    }

    #[must_use]
    pub fn is_current(&self) -> bool {
        !self.token.is_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::Mesh;

    #[test]
    fn mesh_asset_keeps_local_mesh_and_applies_instance_transform() {
        let mut local = Mesh::default();
        local.vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        local.faces = vec![[0, 1, 2]];
        let asset = MeshAsset {
            instances: vec![MeshInstance {
                mesh: Arc::new(local),
                plan: None,
                plan_edge_ids: Vec::new(),
                stats: None,
                transforms: vec![DisplayTransform::Place { offset: [10.0, 0.0, 0.0], rotation: [0.0; 3] }],
            }],
        };
        let merged = asset.merged();
        assert_eq!(merged.vertices[0][0] as f64, 10.0);
        assert_eq!(asset.instance_count(), 1);
    }
}
