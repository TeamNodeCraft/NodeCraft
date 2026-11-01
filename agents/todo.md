# TODO

Priority order is approximate.

## Exact geometry and STEP

- [ ] Clear the remaining official import regressions:
  - NIST ctc-01 unpaired edges
  - `MAPPED_ITEM` assembly transforms
  - `nist_ctc_05` B-rep loop-gap failure
- [ ] Remove the legacy STEP reader/writer once the official path passes the
      full matrix.
- [ ] Add regression coverage for every fixed import/export case.

## Evaluation architecture

- [ ] Split the engine API into exact evaluation and meshing:
  `evaluate_exact` then `mesh_exact`. This is the prerequisite for real
  parallelism; do not just move the monolithic evaluator onto a second pool.
- [ ] Give exact and mesh work independent bounded worker pools and caches.
- [ ] Prioritise visible/selected mesh work; drop stale revisions.
- [ ] Prove a slow mesh job cannot block a graph edit.

## Editor and viewport

- [ ] Add the missing Swift/WebView editor behaviours: save, connection
      creation and compact node-library placement.
- [ ] GPU viewport: selection picking, display-plan tessellation and section
      planes.
- [ ] Fix grid zoom crossfade with a hierarchical line model.
- [ ] Fix single-downstream cable insertion jumps.
- [ ] Finish explicit LOD tiers without corrupting topology.

## Presentation runtime

- [ ] Define a presentation graph separate from the CAD DAG.
- [ ] Add `Time`, `SceneTransform` and `SceneItem` scene concepts.
- [ ] Run animation/presentation scheduling independently of SolidCore/mesher
      pools.
- [ ] Keep selection and picking in world space during animation.

## Viewer breakpoint and scopes

- [ ] Make the breakpoint an explicit scope root independent of selection.
- [ ] Pin and persist it per tab.
- [ ] Model preview and export as sibling scopes of the breakpoint.
- [ ] Cancel stale scope work when the breakpoint moves while retaining caches.
- [ ] Return exact and display results for the scope, never downstream work.

## Typed node contract and data flow

- [ ] Introduce typed runtime values, port specs and a node descriptor.
- [ ] Keep JSON as the serialization format; convert at the document boundary.
- [ ] Add explicit item/list/tree access and distinguish missing, null and empty.
- [ ] Make inputs immutable and thread cancellation through long operations.
- [ ] Add solution identity so stale results are rejected.
- [ ] Add per-item metadata that survives list/tree restructuring.
- [ ] Phase the work: typed contract, list/tree semantics, exact/mesh split,
      metadata/migrations, then schema-driven editor surfaces.

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
