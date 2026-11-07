//! Cached flat evaluation graph for linked documents.
//!
//! Nesting is a UI/presentation concept. The engine always receives one flat
//! graph. This cache keeps that flat graph until a source document changes.

use crate::{expand_linked_parts_with, ContentHash, GraphDocument, SessionError};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// A flattened graph plus the source fingerprint it was built from.
#[derive(Clone, Debug)]
pub struct FlatGraph {
    document: GraphDocument,
    fingerprint: ContentHash,
}

impl FlatGraph {
    #[must_use]
    pub fn document(&self) -> &GraphDocument {
        &self.document
    }

    #[must_use]
    pub fn fingerprint(&self) -> ContentHash {
        self.fingerprint
    }
}

/// Per-tab flatten cache. Linked-document content is part of the fingerprint,
/// so editing a linked document (including a nested one) invalidates the flat
/// graph for every root that depends on it.
#[derive(Default)]
pub struct FlatGraphCache {
    current: Option<FlatGraph>,
}

impl FlatGraphCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the flat graph for the current source documents, rebuilding only
    /// when a source document's content changed.
    pub fn resolve(
        &mut self,
        document: &GraphDocument,
        linked_documents: &BTreeMap<PathBuf, GraphDocument>,
    ) -> Result<&FlatGraph, SessionError> {
        let fingerprint = source_fingerprint(document, linked_documents);
        if self
            .current
            .as_ref()
            .is_none_or(|flat| flat.fingerprint != fingerprint)
        {
            let flattened = expand_linked_parts_with(document, linked_documents)?;
            self.current = Some(FlatGraph {
                document: flattened,
                fingerprint,
            });
        }
        Ok(self.current.as_ref().expect("flat graph inserted"))
    }

    pub fn clear(&mut self) {
        self.current = None;
    }
}

/// Stable fingerprint of the root and every linked source document.
#[must_use]
pub fn source_fingerprint(
    document: &GraphDocument,
    linked_documents: &BTreeMap<PathBuf, GraphDocument>,
) -> ContentHash {
    let mut combined: u128 = 0x9e37_79b9_7f4a_7c15;
    let mut mix = |combined: &mut u128, hash: ContentHash| {
        *combined = combined
            .rotate_left(17)
            .wrapping_add(hash.0)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9);
    };
    mix(
        &mut combined,
        crate::store::document_hash(document).unwrap_or(ContentHash(0)),
    );
    for (path, linked) in linked_documents {
        mix(
            &mut combined,
            crate::store::document_hash(linked).unwrap_or(ContentHash(0)),
        );
        for byte in path.to_string_lossy().as_bytes() {
            combined = (combined ^ u128::from(*byte)).wrapping_mul(0x100_0000_01b3);
        }
    }
    ContentHash(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphConnection, GraphNode, GraphInterface};
    use serde_json::json;
    use std::path::PathBuf;

    fn document(id: &str, value: f64) -> GraphDocument {
        GraphDocument {
            schema_version: 1,
            id: id.into(),
            nodes: vec![GraphNode { expressions: Default::default(),
                id: "p".into(),
                node_type: "primitive".into(),
                values: json!({"primitiveType": "box", "length": value}),
                suppressed: false,
                x: 0.0,
                y: 0.0,
            }],
            connections: Vec::<GraphConnection>::new(),
            interface: GraphInterface::default(),
        }
    }

    #[test]
    fn flat_graph_reuses_until_linked_content_changes() {
        let root = document("root", 1.0);
        let path = PathBuf::from("/tmp/flat-cache-child.nodecraft");
        let mut linked = BTreeMap::new();
        linked.insert(path.clone(), document("child", 1.0));
        let mut cache = FlatGraphCache::new();
        let first = cache.resolve(&root, &linked).expect("flatten").fingerprint();
        let second = cache.resolve(&root, &linked).expect("reuse").fingerprint();
        assert_eq!(first, second);

        linked.insert(path, document("child", 2.0));
        let third = cache.resolve(&root, &linked).expect("rebuild").fingerprint();
        assert_ne!(first, third);
    }
}
