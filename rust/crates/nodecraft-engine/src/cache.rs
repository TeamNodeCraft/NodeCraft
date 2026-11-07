//! Content-addressed caches owned by one evaluation worker.
//!
//! Cache keys are `(node identity, content hash)` pairs. A node marked dirty
//! must bypass every cache even if a caller supplied a stale key. Failed nodes
//! store a failure entry and must never be answered by a healthy mesh entry.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use nodecraft_protocol::{Mesh, Node};

use crate::Part;
use crate::comparison::DifferenceMeshes;

/// Shared key for every content-addressed cache entry: node identity plus the
/// node's content hash. Dirty nodes must bypass lookups using this key.
pub(crate) type CacheKey = (String, String);

const MESH_CACHE_BYTES: usize = 256 << 20;
const BREP_CACHE_BYTES: usize = 256 << 20;
const DEFAULT_PART_CACHE_ENTRIES: usize = 8192;
const MAX_PART_CACHE_ENTRIES: usize = 65_536;
const BODY_MESH_CACHE_BYTES: usize = 256 << 20;
const FAILURE_CACHE_ENTRIES: usize = 4096;
const COMPARISON_CACHE_BYTES: usize = 128 << 20;

/// A node marked dirty must not be answered from any content-addressed cache,
/// even when a caller supplied a key that matches an older result.
pub(crate) fn node_cache_is_fresh(node: &Node) -> bool {
    node.cache.as_ref().and_then(|cache| cache.state.as_deref()) != Some("dirty")
}

#[derive(Clone, Debug)]
pub struct EngineCache {
    meshes: HashMap<CacheKey, Mesh>,
    mesh_order: std::collections::VecDeque<CacheKey>,
    mesh_bytes: usize,
    mesh_limit: usize,
    breps: HashMap<CacheKey, Arc<nodecraft_solidcore::solidcore::topology::Brep>>,
    brep_order: std::collections::VecDeque<CacheKey>,
    brep_bytes: usize,
    brep_limit: usize,
    brep_hits: u64,
    /// Exact multi-solid Part results. Parts are independently cached from
    /// meshes so display-quality changes and viewport eviction cannot discard
    /// exact export data.
    parts: HashMap<CacheKey, Arc<Part>>,
    part_order: std::collections::VecDeque<CacheKey>,
    part_limit: usize,
    parts_hits: u64,
    /// Display meshes keyed by the identity of the immutable body B-rep and
    /// tessellation tolerance. A transform only changes placement, so reusing
    /// the body's local mesh avoids retessellating a large Part on every drag.
    body_meshes: HashMap<(usize, u64), (Arc<nodecraft_solidcore::solidcore::topology::Brep>, Arc<Mesh>)>,
    body_mesh_order: std::collections::VecDeque<(usize, u64)>,
    body_mesh_bytes: usize,
    failures: HashMap<CacheKey, String>,
    failure_order: std::collections::VecDeque<CacheKey>,
    failure_hits: u64,
    comparisons: HashMap<CacheKey, DifferenceMeshes>,
    comparison_order: VecDeque<CacheKey>,
    comparison_failures: HashMap<CacheKey, String>,
    comparison_bytes: usize,
    comparison_hits: u64,
    imports: HashMap<(String, u64, u64), Arc<nodecraft_mesh::StepImport>>,
    import_order: std::collections::VecDeque<(String, u64, u64)>,
    import_hits: u64,
}

impl Default for EngineCache {
    fn default() -> Self { Self::new() }
}

impl EngineCache {
    #[must_use]
    pub fn new() -> Self {
        Self { meshes: HashMap::new(), mesh_order: std::collections::VecDeque::new(), mesh_bytes: 0, mesh_limit: MESH_CACHE_BYTES, breps: HashMap::new(), brep_order: std::collections::VecDeque::new(), brep_bytes: 0, brep_limit: BREP_CACHE_BYTES, brep_hits: 0, parts: HashMap::new(), part_order: std::collections::VecDeque::new(), part_limit: DEFAULT_PART_CACHE_ENTRIES, parts_hits: 0, body_meshes: HashMap::new(), body_mesh_order: std::collections::VecDeque::new(), body_mesh_bytes: 0, failures: HashMap::new(), failure_order: std::collections::VecDeque::new(), failure_hits: 0, comparisons: HashMap::new(), comparison_order: VecDeque::new(), comparison_failures: HashMap::new(), comparison_bytes: 0, comparison_hits: 0, imports: HashMap::new(), import_order: std::collections::VecDeque::new(), import_hits: 0 }
    }

    #[must_use]
    pub fn with_limits(mesh_limit: usize, brep_limit: usize) -> Self {
        let mut cache = Self::new();
        cache.mesh_limit = mesh_limit;
        cache.brep_limit = brep_limit;
        cache
    }

    pub(crate) fn note_brep_hit(&mut self) { self.brep_hits = self.brep_hits.saturating_add(1); }
    pub(crate) fn note_part_hit(&mut self) { self.parts_hits = self.parts_hits.saturating_add(1); }
    pub(crate) fn note_failure_hit(&mut self) { self.failure_hits = self.failure_hits.saturating_add(1); }
    pub(crate) fn note_import_hit(&mut self) { self.import_hits = self.import_hits.saturating_add(1); }

    #[must_use]
    pub(crate) fn difference(&self, key: &CacheKey) -> Option<DifferenceMeshes> {
        self.comparisons.get(key).cloned()
    }

    #[must_use]
    pub(crate) fn difference_failure(&self, key: &CacheKey) -> Option<String> {
        self.comparison_failures.get(key).cloned()
    }

    pub(crate) fn insert_difference_failure(&mut self, key: CacheKey, error: String) {
        if self.comparison_failures.len() >= 1024 {
            if let Some(oldest) = self.comparison_failures.keys().next().cloned() {
                self.comparison_failures.remove(&oldest);
            }
        }
        self.comparison_failures.insert(key, error);
    }

    pub(crate) fn insert_difference(&mut self, key: CacheKey, difference: DifferenceMeshes) {
        self.comparison_failures.remove(&key);
        let bytes = difference_bytes(&difference);
        if let Some(old) = self.comparisons.insert(key.clone(), difference) {
            self.comparison_bytes = self.comparison_bytes.saturating_sub(difference_bytes(&old));
        } else {
            self.comparison_order.push_back(key);
        }
        self.comparison_bytes = self.comparison_bytes.saturating_add(bytes);
        while self.comparison_bytes > COMPARISON_CACHE_BYTES && self.comparisons.len() > 1 {
            let Some(oldest) = self.comparison_order.pop_front() else { break; };
            if let Some(old) = self.comparisons.remove(&oldest) {
                self.comparison_bytes = self.comparison_bytes.saturating_sub(difference_bytes(&old));
            }
        }
    }

    pub(crate) fn note_difference_hit(&mut self) { self.comparison_hits = self.comparison_hits.saturating_add(1); }

    #[must_use]
    pub fn difference_entries(&self) -> usize { self.comparisons.len() }

    #[must_use]
    pub fn difference_hits(&self) -> u64 { self.comparison_hits }

    #[must_use]
    pub fn mesh(&self, key: &CacheKey) -> Option<Mesh> { self.meshes.get(key).cloned() }

    pub fn remove_mesh(&mut self, key: &CacheKey) {
        if let Some(mesh) = self.meshes.remove(key) {
            self.mesh_bytes = self.mesh_bytes.saturating_sub(mesh_bytes(&mesh));
        }
    }

    pub fn insert_mesh(&mut self, key: CacheKey, mesh: Mesh) {
        let bytes = mesh_bytes(&mesh);
        if let Some(old) = self.meshes.insert(key.clone(), mesh) {
            self.mesh_bytes = self.mesh_bytes.saturating_sub(mesh_bytes(&old));
        } else {
            self.mesh_order.push_back(key);
        }
        self.mesh_bytes = self.mesh_bytes.saturating_add(bytes);
        while self.mesh_bytes > self.mesh_limit && self.meshes.len() > 1 {
            let Some(oldest) = self.mesh_order.pop_front() else { break; };
            if let Some(old) = self.meshes.remove(&oldest) { self.mesh_bytes = self.mesh_bytes.saturating_sub(mesh_bytes(&old)); }
        }
    }

    #[must_use]
    pub fn brep(&self, key: &CacheKey) -> Option<Arc<nodecraft_solidcore::solidcore::topology::Brep>> { self.breps.get(key).cloned() }

    pub fn insert_brep(&mut self, key: CacheKey, brep: Arc<nodecraft_solidcore::solidcore::topology::Brep>) {
        let bytes = brep_bytes(&brep);
        if let Some(old) = self.breps.insert(key.clone(), brep) {
            self.brep_bytes = self.brep_bytes.saturating_sub(brep_bytes(&old));
        } else {
            self.brep_order.push_back(key);
        }
        self.brep_bytes = self.brep_bytes.saturating_add(bytes);
        while self.brep_bytes > self.brep_limit && self.breps.len() > 1 {
            let Some(oldest) = self.brep_order.pop_front() else { break; };
            if let Some(old) = self.breps.remove(&oldest) { self.brep_bytes = self.brep_bytes.saturating_sub(brep_bytes(&old)); }
        }
    }

    #[must_use]
    pub fn mesh_entries(&self) -> usize { self.meshes.len() }

    #[must_use]
    pub fn brep_entries(&self) -> usize { self.breps.len() }

    #[must_use]
    pub fn brep_hits(&self) -> u64 { self.brep_hits }

    #[must_use]
    pub(crate) fn part(&self, key: &CacheKey) -> Option<Arc<Part>> {
        self.parts.get(key).cloned()
    }

    #[must_use]
    pub fn body_mesh(
        &self,
        brep: &Arc<nodecraft_solidcore::solidcore::topology::Brep>,
        tolerance: f64,
    ) -> Option<Arc<Mesh>> {
        let key = (Arc::as_ptr(brep) as usize, tolerance.to_bits());
        let (stored, mesh) = self.body_meshes.get(&key)?;
        // The allocator can reuse a pointer after a B-rep is dropped; keep the
        // Arc in the entry and verify identity before trusting the mesh.
        Arc::ptr_eq(stored, brep).then(|| Arc::clone(mesh))
    }

    pub fn insert_body_mesh(
        &mut self,
        brep: Arc<nodecraft_solidcore::solidcore::topology::Brep>,
        tolerance: f64,
        mesh: Arc<Mesh>,
    ) {
        let key = (Arc::as_ptr(&brep) as usize, tolerance.to_bits());
        let bytes = mesh_bytes(&mesh);
        if let Some((_, old)) = self.body_meshes.insert(key, (brep, mesh)) {
            self.body_mesh_bytes = self.body_mesh_bytes.saturating_sub(mesh_bytes(&old));
        } else {
            self.body_mesh_order.push_back(key);
        }
        self.body_mesh_bytes = self.body_mesh_bytes.saturating_add(bytes);
        while self.body_mesh_bytes > BODY_MESH_CACHE_BYTES {
            let Some(oldest) = self.body_mesh_order.pop_front() else { break; };
            if let Some((_, mesh)) = self.body_meshes.remove(&oldest) {
                self.body_mesh_bytes = self.body_mesh_bytes.saturating_sub(mesh_bytes(&mesh));
            }
        }
    }

    /// Size the Part cache for the current graph. A large flat forest can have
    /// more Part-producing nodes than the default budget; evicting them between
    /// transform edits makes upstream combines recompute on every drag.
    pub fn ensure_part_capacity(&mut self, node_count: usize) {
        self.part_limit = node_count
            .saturating_add(2048)
            .clamp(DEFAULT_PART_CACHE_ENTRIES, MAX_PART_CACHE_ENTRIES);
        while self.parts.len() > self.part_limit {
            let Some(oldest) = self.part_order.pop_front() else { break; };
            self.parts.remove(&oldest);
        }
    }

    pub(crate) fn insert_part(&mut self, key: CacheKey, part: Arc<Part>) {
        if self.parts.insert(key.clone(), part).is_none() {
            self.part_order.push_back(key);
        }
        while self.parts.len() > self.part_limit {
            let Some(oldest) = self.part_order.pop_front() else { break; };
            self.parts.remove(&oldest);
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn part_entries(&self) -> usize { self.parts.len() }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn part_hits(&self) -> u64 { self.parts_hits }

    #[must_use]
    pub fn failure(&self, key: &CacheKey) -> Option<String> { self.failures.get(key).cloned() }

    pub fn insert_failure(&mut self, key: CacheKey, error: String) {
        if self.failures.insert(key.clone(), error).is_none() {
            self.failure_order.push_back(key);
        }
        while self.failure_order.len() > FAILURE_CACHE_ENTRIES {
            if let Some(oldest) = self.failure_order.pop_front() { self.failures.remove(&oldest); }
        }
    }

    pub fn remove_failure(&mut self, key: &CacheKey) { self.failures.remove(key); }

    #[must_use]
    pub fn failure_entries(&self) -> usize { self.failures.len() }

    #[must_use]
    pub fn failure_hits(&self) -> u64 { self.failure_hits }

    #[must_use]
    pub fn import(&self, key: &(String, u64, u64)) -> Option<Arc<nodecraft_mesh::StepImport>> { self.imports.get(key).cloned() }

    pub fn insert_import(&mut self, key: (String, u64, u64), import: Arc<nodecraft_mesh::StepImport>) {
        if self.imports.insert(key.clone(), import).is_none() { self.import_order.push_back(key); }
        while self.import_order.len() > 32 { if let Some(oldest) = self.import_order.pop_front() { self.imports.remove(&oldest); } }
    }

    #[must_use]
    pub fn import_hits(&self) -> u64 { self.import_hits }
}

/// Every Stepio failure names the file it was operating on.
pub(crate) fn stepio_error(path: &str, error: String) -> String {
    if path.is_empty() || error.starts_with(path) { error } else { format!("{path}: {error}") }
}

pub(crate) fn import_cache_key(path: &str) -> (String, u64, u64) {
    std::fs::metadata(path)
        .ok()
        .map(|metadata| {
            let modified = metadata.modified().ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map(|duration| duration.as_nanos() as u64).unwrap_or(0);
            (path.to_string(), metadata.len(), modified)
        })
        .unwrap_or_else(|| (path.to_string(), 0, 0))
}

fn mesh_bytes(mesh: &Mesh) -> usize {
    mesh.vertices.len() * 24 + mesh.normals.len() * 24 + mesh.faces.len() * 12 + mesh.face_groups.iter().map(|group| group.len()).sum::<usize>() + 512
}

fn brep_bytes(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> usize {
    brep.face_list().len() * 1024 + brep.edge_list().len() * 512 + brep.vertex_list().len() * 128 + 4096
}

fn difference_bytes(difference: &DifferenceMeshes) -> usize {
    mesh_bytes(&difference.added) + mesh_bytes(&difference.removed)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_cache_capacity_grows_for_large_graphs() {
        let mut cache = EngineCache::new();
        cache.ensure_part_capacity(10_000);
        assert!(cache.part_limit >= 10_000, "large graph should size the Part cache");
        cache.ensure_part_capacity(1);
        assert_eq!(cache.part_limit, DEFAULT_PART_CACHE_ENTRIES);
    }

    #[test]
    fn removing_a_mesh_releases_its_cache_budget() {
        let mut cache = EngineCache::new();
        let key = ("node".to_owned(), "hash".to_owned());
        cache.insert_mesh(key.clone(), Mesh::default());
        cache.remove_mesh(&key);
        assert_eq!(cache.mesh_entries(), 0);
        assert!(cache.mesh(&key).is_none());
    }

    #[test]
    fn failure_entries_are_bounded() {
        let mut cache = EngineCache::new();
        for index in 0..(FAILURE_CACHE_ENTRIES + 4) {
            cache.insert_failure((format!("node-{index}"), String::new()), "failed".into());
        }
        assert_eq!(cache.failure_entries(), FAILURE_CACHE_ENTRIES);
    }
}
