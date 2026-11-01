# Philosophy

NodeCraft is an exact, node-based CAD editor. The graph is the model; the
viewport is a view of evaluated results.

## Core invariants

- **Exact geometry is authoritative.** The worker evaluates a flat DAG into
  B-reps and meshes. The viewport must never invent or repair topology.
- **One transform path.** Card geometry, hit testing, labels, icons, ports,
  cables, lasso and minimap mapping share the canvas transform. See
  `rust/crates/nodecraft-node-canvas/src/canvas.rs` and its regression test
  `card_labels_icons_and_hit_tests_share_one_transform`.
- **No symptom patches.** Do not add offsets, per-node exceptions, arbitrary
  delays, fade-outs, or clamps to hide a stale or incorrect source value. Fix
  the transform, z-order, identity, invalidation, or action ordering.
- **Node state is explicit.** A red node means an expected reference could not
  be resolved. A missing/zero selection is not automatically an error.
- **Exact work and display work are separate.** B-rep construction is not a mesh
  job. Do not hide a threading problem by putting the monolithic evaluator
  behind a second pool.
- **Palette-driven presentation.** Colours come from the theme; new UI should not
  introduce unrelated hard-coded colours. Geometry/display settings are
  platform-independent.
- **Small, testable boundaries.** Prefer a pure helper and a regression test over
  a broad UI special case.

## When a bug appears

1. Identify the shared invariant that is wrong.
2. Make every consumer use the same source of truth.
3. Add a test that fails before the fix.
4. Only then tune appearance.
