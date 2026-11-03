# TODO

Priority order is approximate.

## Recently completed

- [x] Scope engine caches per tab instead of one global worker cache.
- [x] Cache the flattened linked-document graph per tab.
- [x] Publish live document revisions from tabs into the shared document store.
- [x] Open linked `.nodecraft` documents as editable tabs.
- [x] Add the shared `nodecraft-node-contract` crate with descriptors and ports.
- [x] Add typed `NodeInputs` / `NodeOutput` values including Shape, Part and Sketch.
- [x] Add the persistent SolidCore kernel worker pool.
- [x] Inject deterministic per-task RNG seeds.
- [x] Wire revision cancellation tokens into pooled kernel tasks.
- [x] Add owned exact `NodeTask` values and migrate the exact BRep node branches.
- [x] Route primitive, boolean/fuse, translate, mirror, extrude, fillet, shell,
      chamfer, loft, surface loft and STEP import through the kernel pool.
- [x] Add the dependency-level, dataflow-capable parallel scheduler.
- [x] Dispatch primitive/translate/mirror/boolean/extrude/fillet/shell/chamfer/
      STEP import/loft/surfaceloft tasks through the scheduler prewarm path.
- [x] Make scatter exact Part-backed; remove the mesh-only scatter implementation.
- [x] Run exporters only on explicit export requests.
- [x] Add NodeCraft/SolidCore hashes to Diagnostics (short, non-persistent debug overlay).
- [x] Add tab drag-and-drop reordering.
- [x] Reduce viewport edge width to 1 px.
- [x] Document the presentation scene graph and MeshAsset boundary.

## Next: remove tessellation from node branches

Attempted a direct reroute; it broke progress, mesh-cache, BRep-cache and failure
parity tests, so it was reverted. Do this as a dedicated extraction:

- [ ] Add a `DisplayTarget` mesher API that preserves:
  - stable mesh cache keys
  - `node-start` / `node-complete` progress
  - error attribution
  - feature-preview payloads
- [ ] Route `evaluate_part_display` and mesh-only targets through `DisplayTarget`.
- [ ] Delete `brep_display_mesh` / `part_display_mesh` calls from node branches.
- [ ] Add parity tests before deleting each branch.
- [ ] Replace scatter's remaining display-specific path with the same display stage.

## Evaluation architecture

- [ ] Split the engine API into exact evaluation and meshing:
  `evaluate_exact` then `mesh_exact`. This is the prerequisite for real
  parallelism; do not just move the monolithic evaluator onto a second pool.
- [ ] Give exact and mesh work independent bounded worker pools and caches.
- [ ] Make the scheduler the primary exact path and remove duplicate recursive
  exact work from `Eval`.
- [ ] Move cache, error and progress commits into the coordinator.
- [ ] Route every exact node through descriptor entry points and universal
  `NodeInputs`; remove `match node_type` as dispatch authority.
- [ ] Add kernel-aware STEP import or explicit importer kernel ownership.
- [ ] Give loft/surface loft real sketch dependencies in the scheduler instead of
  pre-resolving them as leaf tasks.
- [ ] Prioritise visible/selected mesh work; drop stale revisions.
- [ ] Prove a slow mesh job cannot block a graph edit.

## Mesh and presentation runtime

See [Presentation scene graph](../docs/presentation-scene-graph.md).

- [x] Define the presentation graph separately from the CAD DAG.
- [ ] Implement `MeshAsset` keyed by geometry hash + tolerance/LOD.
- [ ] Implement the independent CPU mesher boundary; GPU mesher later.
- [ ] Preserve scatter instance identity instead of merging into one display mesh.
- [ ] Implement `SceneGraph` core: nodes, hierarchy, stable ids, frame snapshot.
- [ ] Add `Time`, `SceneTransform` and `SceneItem` scene concepts.
- [ ] Run animation/presentation scheduling independently of SolidCore/mesher
      pools.
- [ ] Keep selection and picking in world space during animation.
- [ ] Ensure animation updates transforms only, with no exact or mesh work.

## Exact geometry and STEP

- [ ] Clear the remaining official import regressions:
  - NIST ctc-01 unpaired edges
  - `MAPPED_ITEM` assembly transforms
  - `nist_ctc_05` B-rep loop-gap failure
- [ ] Remove the legacy STEP reader/writer once the official path passes the
      full matrix.
- [ ] Add regression coverage for every fixed import/export case.
- [ ] Separate exporter side effects from evaluation entirely; export only on
      explicit export requests.
- [ ] Remove heal/export fallback kernels once the owning task boundary exists.

## Viewer breakpoint and scopes

- [x] Make the breakpoint an explicit scope root independent of selection.
- [ ] Pin and persist it per tab.
- [ ] Model preview and export as sibling scopes of the breakpoint.
- [ ] Cancel stale scope work when the breakpoint moves while retaining caches.
- [ ] Return exact and display results for the scope, never downstream work.

## Typed node contract and data flow

- [x] Introduce typed runtime values, port specs and a node descriptor.
- [x] Keep JSON as the serialization format; convert at the document boundary.
- [ ] Add explicit item/list/tree access and distinguish missing, null and empty.
- [ ] Make inputs immutable everywhere and thread cancellation through all long
      operations (routed exact tasks are done; remaining branches are not).
- [ ] Add solution identity so stale results are rejected.
- [ ] Add per-item metadata that survives list/tree restructuring.
- [ ] Phase the work: typed contract, list/tree semantics, exact/mesh split,
      metadata/migrations, then schema-driven editor surfaces.

## Editor and viewport

- [ ] Add the missing Swift/WebView editor behaviours: save, connection
      creation and compact node-library placement.
- [ ] GPU viewport: selection picking, display-plan tessellation and section
      planes.
- [ ] Fix grid zoom crossfade with a hierarchical line model.
- [ ] Fix single-downstream cable insertion jumps.
- [ ] Finish explicit LOD tiers without corrupting topology.
- [ ] Add parity/regression coverage for tab reordering and linked-document tabs.

## Cross-platform and release

- [ ] Exercise Linux and Windows builds in CI, then run a real interactive smoke
      test.
- [ ] Add platform packaging (Windows/Linux desktop launch, macOS app bundle).
- [ ] Keep the public snapshot free of private paths, corpora and generated
      reports.

## Formal verification research

- [ ] Investigate using Lean 4 to check selected NodeCraft invariants in future.
- [ ] Start with small, pure Rust modules and evaluate Rust-to-Lean extraction
      tools such as Aeneas or hax before committing to a large proof effort.
- [ ] Compare Lean against Rust-specific options (Kani, Prusti, Creusot) and
      record which invariants are practical to prove.
- [ ] Keep Lean out of the build and runtime until a concrete, bounded proof
      target justifies the tooling cost.
