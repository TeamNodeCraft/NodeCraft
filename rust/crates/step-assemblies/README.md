# step-assemblies

STEP (AP203 / AP214 / AP242) **product/assembly structure** parser for NodeCraft.
Uses only the public `nodecraft_step::{StepFile, Value}` parser and turns the product
graph into a neutral tree of named occurrences with 4x4 transforms.

```rust
use step_assemblies::parse_assembly_file;

let assembly = parse_assembly_file("assembly.step")?;
let root = assembly.root().expect("a product");
println!("{}: {} nodes", root.name, assembly.node_count());
for node in assembly.walk() {
    println!(
        "{} [{}] world translation {:?}",
        node.name,
        node.product_name,
        node.world_transform.translation(),
    );
}
```

## Parsed STEP entities

- `PRODUCT`, `PRODUCT_DEFINITION`, `PRODUCT_DEFINITION_FORMATION*`,
  `PRODUCT_DEFINITION_SHAPE`, `SHAPE_DEFINITION_REPRESENTATION`
- `SHAPE_REPRESENTATION` and subtypes (`ADVANCED_BREP_SHAPE_REPRESENTATION`,
  `MANIFOLD_SURFACE_SHAPE_REPRESENTATION`, ...) with their item lists
- `NEXT_ASSEMBLY_USAGE_OCCURRENCE` (also `ASSEMBLY_COMPONENT_USAGE` /
  `PRODUCT_DEFINITION_USAGE`): one occurrence = one parent/child instance
- `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`: links an occurrence's
  `PRODUCT_DEFINITION_SHAPE` to its representation relationship
- `REPRESENTATION_RELATIONSHIP` (plain, complex instance, and
  `REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION`)
- transformations: `ITEM_DEFINED_TRANSFORMATION` (with two `AXIS2_PLACEMENT_3D`
  items), `CARTESIAN_TRANSFORMATION_OPERATOR_3D`,
  `CARTESIAN_TRANSFORMATION_OPERATOR`
- `AXIS2_PLACEMENT_3D` placements (location + local Z + local X, Y = Z × X)

## API

| Item | Meaning |
|---|---|
| `parse_assembly(text)` | parse ISO-10303-21 text |
| `parse_assembly_file(path)` | read + parse a file |
| `parse_assembly_from_step_file(&StepFile)` | reuse an already-parsed file |
| `Assembly { roots, products, definitions, representations, product_definition_shapes, warnings }` | neutral product graph; `roots` is the instance tree |
| `Node { name, product_name, occurrence_id, occurrence_name, placement_name, definition_id, representation_id, local_transform, world_transform, children }` | one placed occurrence (the root node has no occurrence ids and identity local transform) |
| `Transform` | row-major 4x4, column-vector points: `from_placement`, `from_columns`, `multiply`, `inverse`, `transform_point`, `translation`, `approx_eq`, `is_identity`, `rows` |
| `Assembly::{root, node_count, walk, product}`; `Node::find` | tree helpers |

World transforms accumulate as `world = parent_world * local`, so a node's
`world_transform` maps its own child-representation coordinates straight into the
document frame.

## Transform direction

Exporters disagree on which side of a `REPRESENTATION_RELATIONSHIP` is the parent, so the
product structure decides: the occurrence's parent/child `PRODUCT_DEFINITION`s (from
`NEXT_ASSEMBLY_USAGE_OCCURRENCE`) select the relationship side, with the CAx-IF reading
(`rep_1` = child, `rep_2` = parent) as fallback. The local transform is
`placement(parent item) * placement(child item)^-1`, mapping child representation
coordinates into the parent's. This mirrors smlibpy's `step_reader.py`
(`_representation_transforms` / `_item_transform`).

## Validation

`cargo test -p step-assemblies` → 5 unit tests + 12 integration tests + 1 doctest:

- parent/child placement: translation, rotation (local X → parent Y), point mapping;
- multiple occurrences of the same part: same `PRODUCT`/definition/representation,
  two nodes at different transforms;
- nested assemblies: root → sub-assembly → part, `world = parent * local`
  (10 + 5 = 15), checked recursively for every test tree;
- relationships with the parent representation listed first (the `as1_pe.stp` layout)
  produce the same placement;
- missing `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` → identity + warning; plain
  supplemental `SHAPE_REPRESENTATION_RELATIONSHIP` links are ignored (no phantom
  occurrence); cycles, missing definitions and unparseable input are explicit errors;
- leaf-only files (no occurrences) become a single root;
- real repository fixtures: `as1_pe.stp` (1 root, 15 nodes, 12 placed bodies, transforms
  cross-checked against smlibpy), `moon_buggy_asm.stp` (20 products, 30 nodes),
  `weldment_asm_solid.stp` (34 products, 109 nodes), `rear.stp` (49 products, 50 nodes),
  `cubcylso.stp` (2 nodes). The fixtures are skipped when not present.

## Limits

- Transforms are in **STEP file units**; no unit conversion (smlibpy reports mm, so
  `as1_pe.stp` — inches — matches smlibpy's placements multiplied by 25.4).
- Assemblies whose structure is expressed only through `MAPPED_ITEM` /
  `REPRESENTATION_MAP` (no `NEXT_ASSEMBLY_USAGE_OCCURRENCE`) are not expanded into a
  product tree; only representation metadata is exposed.
- An occurrence without a placement relationship keeps an identity transform and adds a
  warning (the file still parses).
- Geometry is not read: this crate is only the product/assembly structure. Colors,
  validation properties and PMI are ignored.
- Cyclic parent/child usage, missing product definitions and singular placements are
  errors; other recoverable problems land in `Assembly::warnings`.
