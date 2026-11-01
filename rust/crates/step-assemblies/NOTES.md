# step-assemblies: product/assembly structure notes

Status: **implemented and tested.** `cargo test -p step-assemblies` → 5 unit tests + 12
integration tests + 1 doctest, no warnings. The parser reads the AP203/AP214/AP242
product graph through the public `nodecraft_step::{StepFile, Value}` API and exposes a
neutral occurrence tree with local and accumulated world 4x4 transforms.

## Design

| file | contents |
|---|---|
| `src/transform.rs` | row-major 4x4 `Transform` (column-vector points): placement → matrix, multiply, inverse (3x3 adjugate), point/vector mapping, tolerance compares. |
| `src/model.rs` | neutral data: `Assembly`, `Node`, `Product`, `Definition`, `Representation`, `ProductDefinitionShape`, tree helpers (`root`, `node_count`, `walk`, `find`). |
| `src/parse.rs` | entity collection, occurrence construction, transform direction heuristics, tree assembly, warnings/errors. |
| `src/error.rs` | `AssemblyError::{Step, Malformed}`. |

The parser is deliberately two-layered:

1. **Collection** (one pass per entity kind): products + formations, product
   definitions, representations and their items, product-definition shapes,
   shape-definition representations, usage occurrences, representation relationships,
   placements (lazily resolved).
2. **Interpretation**: each `NEXT_ASSEMBLY_USAGE_OCCURRENCE` becomes one occurrence;
   the occurrence's `PRODUCT_DEFINITION_SHAPE` links through
   `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` to a `REPRESENTATION_RELATIONSHIP`, whose
   `ITEM_DEFINED_TRANSFORMATION` yields the local matrix. The same child product can be
   used by any number of occurrences (a separate node each time), and occurrences can
   nest (a child definition that is itself a parent expands recursively).

## Transform semantics

- Local = `M(parent item) * inverse(M(child item))`: maps child-representation
  coordinates into the parent representation. World = `parent world * local`, identity
  at each root.
- Direction of `ITEM_DEFINED_TRANSFORMATION`: the relationship's representation order is
  not trusted first; the occurrence's parent/child product definitions from the product
  structure pick the parent side. Fallback is the CAx-IF convention (`rep_1` = child,
  `rep_2` = parent), then item membership in the representation item lists decides which
  of the two placements belongs to the parent. This is a direct port of the logic in
  smlibpy's `step_reader.py` (`_representation_transforms`, `_item_transform`) so both
  readers agree on the same files.
- Unsupported operators (anything other than `ITEM_DEFINED_TRANSFORMATION`,
  `CARTESIAN_TRANSFORMATION_OPERATOR_3D`, `CARTESIAN_TRANSFORMATION_OPERATOR`) keep an
  identity local transform and add a warning; a missing
  `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION` likewise warns and uses identity. Nothing is
  silently skipped.
- Placements are `AXIS2_PLACEMENT_3D(location, axis, ref_direction)`; defaults are
  `(0,0,0)`, `+Z`, `+X`; a missing/parallel reference direction is replaced by a
  perpendicular fallback so the frame stays orthonormal.

## Errors vs warnings

Errors (`AssemblyError::Malformed`): no `PRODUCT` entities, no `PRODUCT_DEFINITION`, a
usage occurrence referencing a missing product definition, cyclic parent/child usage, a
transformation item that is not an `AXIS2_PLACEMENT_3D`, a singular child placement,
non-finite transform entries. Syntax errors come back as `AssemblyError::Step`.

Warnings (`Assembly::warnings`, parse succeeds): occurrence without a placement
relationship, unknown transformation operator, usage occurrence without resolvable
parent/child. A synthetic test asserts the identity+warning fallback.

## Real-file evidence

Parsed with `examples/inspect` (also covered by tests when the fixtures are present):

| fixture | products | roots | nodes | notes |
|---|---:|---:|---:|---|
| `tests/Sample AP203 Data Files/as1_pe.stp` | 7 | 1 (`AS1_ASM`) | 15 | 4 top-level instances; the two `L-BRACKET_ASM` occurrences each expand their 5 nested parts (1+4+2×5). 12 placed bodies, no warnings. |
| `tests/Sample AP203 Data Files/moon_buggy_asm.stp` | 20 | 1 (`Moon Buggy`) | 30 | AP203 file has no `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`; 3 occurrences fall back to identity with warnings, the rest are placed by relationships. |
| `tests/Sample AP203 Data Files/weldment_asm_solid.stp` | 34 | 1 (`WELDMENT_ASM`) | 109 | assembly-of-assemblies, all placements resolved. |
| `tests/Sample AP203 Data Files/rear.stp` | 49 | 1 (`REAR_LEFT_QUARTER_ASM`) | 50 | one level, 49 placed children. |
| `tests/Sample AP203 Data Files/cubcylso.stp` | 2 | 1 (`*MASTER`) | 2 | `DETAIL` placed at `(200, 0, 0)`. |

Cross-check for `as1_pe.stp` (the file declares **inch** units, our transforms stay in
file units): the world translations our parser computes, multiplied by 25.4, are exactly
the placements smlibpy's `step_reader.interpret` reports for the same 12 bodies — e.g.
`L-BRACKET` `35 → 889`, `L-BRACKET` `-135 → -3429`, `BOLT` `(15, 10, 0) → (381, 254, 0)`,
`NUT` `(35, 40, 0) → (889, 1016, 0)`, `ROD` `(-152, 40, 0) → (-3860.8, 1016, 0)`. The test
asserts those file-unit translations for `as1_pe.stp`, so the direction/rotation logic is
locked without depending on smlibpy at test time.

## Limits / next steps

- **No unit conversion.** Transforms are in STEP file units. A length-unit-aware wrapper
  would multiply translations (and child geometry) by the representation context's unit
  factor; smlibpy already reports mm.
- **Product tree only.** `MAPPED_ITEM`/`REPRESENTATION_MAP`-only assemblies (no
  `NEXT_ASSEMBLY_USAGE_OCCURRENCE`) are not expanded; the representations they form are
  still listed. AP242 "occurrence" variants other than the usage-occurrence family are
  not interpreted.
- **No geometry.** Colors, validation properties and PMI are ignored; the crate returns
  structure + transforms, not meshes.
- **Worker integration is not wired.** `nodecraft-worker` still imports a single
  representation; using this tree would mean asking the worker for each
  `representation_id` and applying `world_transform`, which is outside this crate.
- Multiple roots are supported (a file with several unparented top products), and a
  leaf-only file (no occurrences) becomes a single root.
