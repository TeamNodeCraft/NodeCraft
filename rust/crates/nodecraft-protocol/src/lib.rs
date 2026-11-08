//! The JSON worker protocol shared by the Python and Rust evaluation workers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn default_run_exporters() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Request {
    pub request_id: u64,
    #[serde(default)]
    pub tab_id: Option<String>,
    #[serde(default)]
    pub graph_revision: Option<String>,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub mesh_targets: Option<Vec<String>>,
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub connections: Vec<Connection>,
    #[serde(default)]
    pub binary_mesh: bool,
    #[serde(default)]
    pub project_directory: Option<String>,
    #[serde(default)]
    pub progress: bool,
    /// Run exporter side effects (write STEP files). Normal UI evaluation
    /// leaves this false so editing export metadata cannot write files.
    #[serde(default = "default_run_exporters")]
    pub run_exporters: bool,
    /// Display max-sag override for LOD. `None` uses the viewport default.
    #[serde(default)]
    pub mesh_tolerance: Option<f64>,
    /// `[before, after]` feature targets for an exact comparison payload.
    #[serde(default)]
    pub comparison_targets: Option<Vec<String>>,
    /// Developer rebuild: ignore every cache entry, so each node is evaluated
    /// again even when a fresh cache would be populated within the same pass.
    #[serde(default)]
    pub bypass_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub values: serde_json::Value,
    #[serde(default)]
    pub suppressed: bool,
    #[serde(default)]
    pub cache: Option<CacheState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheState {
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Connection {
    pub from: String,
    pub to: String,
    #[serde(rename = "in")]
    pub input: String,
    #[serde(default)]
    pub out: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mesh {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vertices: Vec<[f32; 3]>,
    #[serde(default)]
    pub faces: Vec<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub normals: Vec<[f32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertices_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normals_data: Option<String>,
    #[serde(default)]
    pub face_groups: Vec<String>,
    /// Optional per-face sRGB colour (parallel to `face_groups`). Empty means
    /// the renderer's default body colour.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub face_colors: Vec<[f32; 3]>,
    #[serde(default)]
    pub brep_edges: Vec<BrepEdge>,
    #[serde(default)]
    pub lightweight: bool,
    /// Optional additive/removal tool preview drawn over the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_preview: Option<Box<Mesh>>,
    /// `join`/`new` draw the preview additively; `remove`/`intersect` cut it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_preview_operation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrepEdge {
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub segments: Vec<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshPayload {
    pub key: Option<String>,
    pub mesh: Mesh,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    pub request_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tab_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_revision: Option<String>,
    #[serde(default)]
    pub meshes: BTreeMap<String, MeshPayload>,
    #[serde(default)]
    pub empty_targets: Vec<String>,
    #[serde(default)]
    pub node_errors: BTreeMap<String, String>,
    #[serde(default)]
    pub evaluation_order: Vec<String>,
    #[serde(default)]
    pub timings: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub brep_timings: BTreeMap<String, f64>,
    #[serde(default)]
    pub evaluation_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_stats: Option<CacheStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<ComparisonPayload>,
    pub engine: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComparisonPayload {
    pub added: Mesh,
    pub removed: Mesh,
    /// True when the ghost came from the display-surface fallback rather than
    /// the exact feature Boolean.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub approximate: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheStats {
    #[serde(default)]
    pub hits: u32,
    #[serde(default)]
    pub misses: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    #[serde(rename = "type")]
    pub kind: String,
    pub phase: String,
    pub request_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    #[serde(default)]
    pub cached: bool,
}
