//! The neutral product/assembly tree produced by the parser.

use crate::Transform;

/// Parsed STEP product/assembly structure plus the raw product metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Assembly {
    /// Top-level instances (usually one; several for multi-root files).
    pub roots: Vec<Node>,
    /// Every `PRODUCT` entity in file order.
    pub products: Vec<Product>,
    /// Every `PRODUCT_DEFINITION` and its shape representation, in file order.
    pub definitions: Vec<Definition>,
    /// Every shape representation (`SHAPE_REPRESENTATION` and subtypes), in file order.
    pub representations: Vec<Representation>,
    /// Every `PRODUCT_DEFINITION_SHAPE`, in entity order.
    pub product_definition_shapes: Vec<ProductDefinitionShape>,
    /// Non-fatal structure problems (missing placement relationship, unknown operator,
    /// unplaced part, ...). Parsing still succeeds.
    pub warnings: Vec<String>,
}

/// One `PRODUCT` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Product {
    pub entity_id: u32,
    /// `PRODUCT` id string (first attribute).
    pub product_id: String,
    /// Display name (second attribute, falling back to the id string).
    pub name: String,
    pub description: String,
}

/// One `PRODUCT_DEFINITION` and its shape representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    pub entity_id: u32,
    /// Product this definition belongs to, when resolvable.
    pub product_entity: Option<u32>,
    pub product_name: String,
    /// Shape representation linked through `PRODUCT_DEFINITION_SHAPE` /
    /// `SHAPE_DEFINITION_REPRESENTATION`.
    pub representation: Option<u32>,
}

/// One `PRODUCT_DEFINITION_SHAPE`: the named "shape" of a product definition or of an
/// assembly usage occurrence (the latter carries the occurrence's placement name).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductDefinitionShape {
    pub entity_id: u32,
    pub name: String,
    pub description: String,
    /// The `PRODUCT_DEFINITION` or `NEXT_ASSEMBLY_USAGE_OCCURRENCE` it describes.
    pub defined: Option<u32>,
}

/// One shape representation and its items.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Representation {
    pub entity_id: u32,
    pub name: String,
    /// Entity type (`SHAPE_REPRESENTATION`, `ADVANCED_BREP_SHAPE_REPRESENTATION`, ...).
    pub kind: String,
    /// Representation items (placement, solid, mapped item, ...) in file order.
    pub items: Vec<u32>,
}

/// One instance in the assembly tree: a placed child (or the root product).
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Display name: occurrence name when the file names one, else the product name.
    pub name: String,
    /// `PRODUCT` display name of the referenced product.
    pub product_name: String,
    /// `NEXT_ASSEMBLY_USAGE_OCCURRENCE` id string (occurrences only).
    pub occurrence_id: Option<String>,
    /// `NEXT_ASSEMBLY_USAGE_OCCURRENCE` name (occurrences only).
    pub occurrence_name: Option<String>,
    /// Name of the occurrence's `PRODUCT_DEFINITION_SHAPE` ("Placement #0", ...).
    pub placement_name: Option<String>,
    /// Child `PRODUCT_DEFINITION` entity id.
    pub definition_id: Option<u32>,
    /// Child shape representation entity id.
    pub representation_id: Option<u32>,
    /// The occurrence's transform in its parent's coordinates (identity for the root).
    pub local_transform: Transform,
    /// `parent world * local`: the occurrence in the document/world frame (identity
    /// transform at the root).
    pub world_transform: Transform,
    /// Nested occurrences, in file order.
    pub children: Vec<Node>,
}

impl Assembly {
    /// The first root instance, if any.
    #[must_use]
    pub fn root(&self) -> Option<&Node> {
        self.roots.first()
    }

    /// Number of nodes in the tree, roots included.
    #[must_use]
    pub fn node_count(&self) -> usize {
        fn count(node: &Node) -> usize {
            1 + node.children.iter().map(count).sum::<usize>()
        }
        self.roots.iter().map(count).sum()
    }

    /// All nodes in pre-order (root, then descendants).
    #[must_use]
    pub fn walk(&self) -> Vec<&Node> {
        fn push<'a>(node: &'a Node, out: &mut Vec<&'a Node>) {
            out.push(node);
            for child in &node.children {
                push(child, out);
            }
        }
        let mut out = Vec::with_capacity(self.node_count());
        for root in &self.roots {
            push(root, &mut out);
        }
        out
    }

    /// The parsed product record for a product entity id.
    #[must_use]
    pub fn product(&self, entity_id: u32) -> Option<&Product> {
        self.products.iter().find(|product| product.entity_id == entity_id)
    }
}

impl Node {
    /// Depth-first search for the first node with `name`.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&Node> {
        if self.name == name {
            return Some(self);
        }
        self.children.iter().find_map(|child| child.find(name))
    }
}
