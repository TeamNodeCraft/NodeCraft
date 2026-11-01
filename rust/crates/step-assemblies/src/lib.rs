//! STEP product/assembly structure parser.
//!
//! Reads the AP203/AP214/AP242 product graph from a
//! [`nodecraft_step::StepFile`] and exposes a neutral tree of named occurrences with
//! 4x4 local transforms and accumulated world transforms:
//!
//! ```no_run
//! use step_assemblies::{parse_assembly_file, Transform};
//!
//! # fn main() -> Result<(), step_assemblies::AssemblyError> {
//! let assembly = parse_assembly_file("assembly.step")?;
//! for root in &assembly.roots {
//!     for node in std::iter::once(root).chain(root.children.iter()) {
//!         println!("{} at {:?}", node.name, node.world_transform.translation());
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Supported STEP structure: `PRODUCT`, `PRODUCT_DEFINITION`,
//! `PRODUCT_DEFINITION_SHAPE`, `SHAPE_REPRESENTATION` (and subtypes),
//! `SHAPE_DEFINITION_REPRESENTATION`, `NEXT_ASSEMBLY_USAGE_OCCURRENCE` (also
//! `ASSEMBLY_COMPONENT_USAGE` / `PRODUCT_DEFINITION_USAGE`),
//! `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`,
//! `REPRESENTATION_RELATIONSHIP` / `..._WITH_TRANSFORMATION`, and transformations via
//! `ITEM_DEFINED_TRANSFORMATION`, `CARTESIAN_TRANSFORMATION_OPERATOR_3D` /
//! `CARTESIAN_TRANSFORMATION_OPERATOR`, with `AXIS2_PLACEMENT_3D` placements.
//!
//! Semantics and the parent/child-side heuristic follow smlibpy's `step_reader.py`
//! (CAx-IF convention by default, corrected by the product structure). Local transforms
//! map child-representation coordinates into the parent's; world transforms accumulate
//! as `parent * local`.

#![forbid(unsafe_code)]

pub mod error;
pub mod external;
pub mod model;
pub mod parse;
pub mod planar;
pub mod transform;
pub mod write;

pub use error::AssemblyError;
pub use external::{
    ComponentInstance, ComponentSource, Components, ExternalDocument, component_instances,
    external_documents, unit_scale_mm,
};
pub use model::{Assembly, Definition, Node, Product, ProductDefinitionShape, Representation};
pub use parse::{parse_assembly, parse_assembly_file, parse_assembly_from_step_file};
pub use planar::{PlanarInstance, planar_instances};
pub use write::{AssemblyNode, PlanarPart, assembly_tree_text, planar_assembly_text, write_assembly_tree, write_planar_assembly};
pub use transform::Transform;
