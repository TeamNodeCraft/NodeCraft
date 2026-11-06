# TODO

Priority order is approximate.

## Recently completed

- [x] Prepare the upstream layer adoption locally: delete the unused
      `crates/solidcore-mesh` port, add the pinned `nodecraft-layers` crate with a
      record -> `solidcore-tess` adapter and a copied record fixture test, and an
      ops primitive/smb smoke test. The kernel B-rep <-> record bridge is still
      missing upstream, so `record_bridge()` stays `Missing` and the engine is
      untouched.
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
- [x] Add GPU instance transforms for viewport objects (`660ae76`), with a
      display-only drag preview that re-evaluates only on release (`35d3401`).
- [x] Scope drag previews to the exact Part body branch (`0a6e149`).
- [x] Make whole-mesh previews a GPU-only scene projection change (`225f4cb`).
- [x] Run native file dialogs off the winit event handler (`b3b8151`).
- [x] Render open surfaces (Surface Loft and similar) two-sided instead of
      culling their winding-away back faces.
- [x] Route upward node cables as a subway-map wrap: straight drops and runs
      joined by geometry that reads smoothly — each corner is a cubic
      approximation of a circular quarter arc (0.5523 * radius controls) between
      exact straight segments, and the wrap picks its horizontal side from the
      lane helper. Normal downward cables keep vertical port tangents with
      clamped handles so short aligned drops no longer cusp.
- [x] Skip the GPU transform preview when a transform has a sketch/spline
      upstream: those nodes have no exact Part body layout, so the preview scope
      fell back to the whole display mesh and moved unrelated geometry. Without
      a GPU preview the drag now lets the kernel evaluate the edit live instead.
- [x] Recover the node-canvas cable performance lost during the routing work:
      hit testing now uses the routed cubic control hull for its broad phase and
      an allocation-free sampled distance instead of building a 64-point Vec per
      cable, and the GPU cable vertex budget follows the style (288 vertices for
      a single-cubic bezier cable instead of always issuing 1152). Hit testing
      also follows the selected style now.
- [x] Add an app setting (persisted in settings.json, not the theme) for cable
      routing: `j-routing` (horizontal/vertical rounded corners) and
      `bezier-routing` (the original smooth single-cubic curves). It is exposed
      in Settings > Appearance and threaded through the retained cable geometry,
      hit testing, drag previews and auto-placement so every consumer agrees.
- [x] Prefer horizontal and vertical cable connectors that turn the input from
      the source-facing side: an upward route bends just outside the target
      card's near edge plus padding, runs along the card top and drops into the
      port, so it does not wind around the whole card. The route is deterministic
      and shared by CPU hit testing, placement and the GPU instance.
- [x] Snap cable shapes deterministically when a structured corner would be too
      sharp: downward corner radii below the stroke-safe threshold and cramped
      upward wraps fall back to one smooth cubic (vertical port tangents, with a
      small bow when the ports are nearly aligned). The decision is pure port and
      card geometry, so the CPU hit-test route and the GPU control points always
      agree.
- [x] Make cable routing node-size aware through one shared maths function:
      `route_nodes` takes the source and target card rects, places the wrap lane
      outside the wider card on the target's side, and `arc_handle` supplies the
      same quarter-arc control distance to every corner. CPU hit testing, the CPU
      fallback, placement paths and the GPU cable instance all consume the
      resulting control points, and the near-zero-drop fade lets cables pass
      under the cards so the up/down shapes transition continuously.
- [x] Keep the final cable ribbon vertex on the routed end point: the GPU
      segment index is clamped before computing the local `t`, so `t == 1` no
      longer snaps the last quad back to the segment start and leaves a stray
      vertex.
- [x] Give downward node cables the same subway-map vocabulary: straight drops
      and runs joined by quarter-circle corner cubics. Aligned drops stay
      straight vertical lines, and small drops blend the corners away so the
      cable does not snap when a node crosses the source.
- [x] Smooth corners that are too tight for the stroke: when the quarter-arc
      radius is below a threshold the path blends towards one smooth cubic
      split into the same number of segments, and the GPU derives a tangent from
      the neighbouring segments when a degenerate segment's analytic tangent is
      zero, preventing the ribbon from pinching into a notch.

## Viewport transform-gizmo preview

- [ ] Move partial-body preview edge lines onto the GPU. Scoped drags still
      rebuild the SolidCraft edge scene and re-upload its buffers every frame,
      because a `GpuScene` has no per-body line instances. Either add instance
      transforms to the SolidCraft line path or render the moved bodies' edges
      with an app-owned GPU line renderer.
      Acceptance: dragging one tree in `samples/gigatest-forest-5000-2.nodecraft`
      performs no per-frame scene rebuild and stays interactive.
- [ ] Make face picking respect the active preview. `pick_face` and
      `pointer_hits_body` raycast the un-previewed CPU display mesh, so clicks
      during a drag (or while the release evaluation runs) can hit the stale
      position. Transform the ray by the inverse preview matrix for affected
      bodies, or disable face picking while a preview is active. Add a
      regression test.
- [ ] Exact scoped preview for merged-body targets. Boolean/fuse,
      fillet/chamfer with edges, shell and similar nodes collapse provenance into
      one body, so the preview moves that whole body. Decide between the
      documented proxy ghost (`docs/viewer-gizmo-proxy.md`) and per-body source
      provenance in the engine display payload; document the approximation until
      then.
- [ ] Cover the remaining `preview_body_scope` layout cases with tests: STEP
      importer, Scatter nested inside a Combine, sketch/spline curve instances,
      group/linked-part expansion (currently falls back to whole mesh), and
      suppressed nodes. The flattened body-index map must stay in lockstep with
      the engine's Part display ordering.
- [ ] Coalesce a whole gizmo drag into one undo entry. The 700 ms value
      coalescing window splits a slow drag with pauses into multiple undo steps.
- [ ] Verify deferred document publishing. Publishing is skipped during a drag
      and flushed on release; confirm linked-part roots, backdrop readers and
      other tabs stay consistent, and add a test around tab switches mid-drag.
- [ ] Verify preview rendering in every view mode: transparent, zebra, face
      colours, orthographic, and the new world-Z-axis renderer (colour, fade and
      depth behaviour must match the old scene axis).
- [ ] Add performance guards. Expose counters for scene uploads and cached
      preview scopes/ranges, assert that a whole-mesh preview performs zero scene
      uploads, and keep a small drag benchmark so the fast path cannot regress.
- [ ] Cache the downstream placement chain used by `selected_transform_gizmo`.
      It still recomputes the `transform_chain` BFS and rebuilds the triad every
      frame; cache the chain per selection/target.
- [ ] Regression-test the async file dialogs. Only Save As has been exercised
      live; verify Open, linked-part picker, STEP file and STEP folder dialogs
      (open, cancel and complete) so the winit event-handler re-entrancy crash
      cannot reappear through another call site.
- [ ] Update `docs/viewer-gizmo-proxy.md` for the egui implementation: exact Part
      body flattening, preview scopes, the GPU projection fast path, the
      partial-preview limitation and non-blocking dialogs.
- [ ] Housekeeping: remove or fix the untracked, non-compiling
      `rust/crates/nodecraft-engine/src/bin/step_report_mesh.rs` (it breaks
      full-workspace builds), decide on the untracked
      `samples/gigatest-forest-5000-2.nodecraft` and the long-modified
      `samples/gigatest-forest-5000.nodecraft`, and confirm no preview debug hooks
      remain.

## Refactor `app.rs`

`rust/apps/nodecraft-egui/src/app.rs` is ~8,100 lines. Split by lifecycle and
ownership, not by line count. Extract one unit per commit, keep `NodeCraftApp`
as the owner, and keep behavior unchanged. Target: roughly 3,500–4,500 lines
with the central draw order still in one place.

- [ ] **Evaluation** → `evaluation.rs`: `EvalJob`, `TabEvalState`,
      `EvalWorker`, evaluation-key/dirty plumbing, `evaluate` /
      `evaluate_with_cache_policy`, `poll_evaluation`. Highest-value, lowest-UI
      coupling.
- [ ] **Settings persistence** → `settings.rs`: `AppSettings`,
      `ActivitySettings`, platform/config paths, load/save, defaults, startup
      document selection.
- [ ] **Settings and theme UI** → `settings_ui.rs` / `theme_editor.rs`:
      `show_app_settings`, theme add/revert/edit buttons, hex editor, palette
      rows, colour wheel.
- [ ] **Tool dock** → `tool_dock.rs`: `ToolSection`, `ToolGroup`, `ToolItem`,
      menu/search/focus state and `show_tool_dock`.
- [ ] **Tabs and document lifecycle** → `tabs.rs`: `DocumentTab`,
      open/save/restore/activate/close, session installation, document tabs and
      file dialogs/drop handling.
- [ ] **Pure document operations** → `document_ops.rs`: clipboard/paste plan,
      delete/disconnect command builders, `target_closure`,
      `downstream_nodes`, `initial_viewer_target`, node position/placement,
      tab reorder, path/document-id helpers.
- [ ] **Central UI** last: keep one `ui()` orchestration function and extract
      one panel per module (`tabs`, `viewport`, `node_editor`, `inspector`,
      `sketch`, `tool_dock`, overlays). Do not extract before the units above
      leave the file.

Acceptance:

- each extraction is a separate, behavior-preserving commit;
- no duplicated helpers or visibility-only copies;
- unit tests move with their module, and app-level tests stay in `app.rs`;
- `cargo test -p nodecraft-egui` stays green after every step;
- `app.rs` ends up containing the app struct, shared state, central `ui()`
      draw order and high-level coordination only.

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
- [ ] Add a closedness flag to display meshes so back-face culling can be
      restored for bodies known to be closed; open surfaces (Surface Loft,
      open shells) must stay two-sided.
- [ ] Add parity/regression coverage for tab reordering and linked-document tabs.

## Appearance and materials

- [ ] Extend Look from colour-only to a named material definition shared by
      multiple Look nodes: base colour, roughness, metallic, opacity, optional
      clearcoat/sheen and a display name.
- [ ] Store material identity on the Part/body boundary, separate from the
      SolidCore colour attribute; keep colour as the one value the kernel and
      STEP exporter can understand today.
- [ ] Add per-body or per-face material parameters to the display mesh and GPU
      object shader. The shader currently has global roughness/metallic
      uniforms, so plastic materials need those values in the vertex/instance
      payload or a second material buffer.
- [ ] Define material precedence through Combine/Boolean: same material keeps
      it, mixed materials clear to a neutral material and emit a yellow warning.
- [ ] Add regression tests for material survival through translate, mirror,
      shell, chamfer, fillet and Boolean, plus material clearing on mixed
      Boolean results.
- [ ] Do not add textures until the display mesh carries UVs end to end:
      protocol mesh, `DisplayMesh`, GPU vertex format, `solidcore-tess` output,
      and a per-face UV mapping/conversion boundary.
- [ ] Decide the UV source for imported STEP faces, primitive faces and
      post-Boolean/post-fillet faces. Boolean and fillet rebuild faces, so a
      texture assignment needs a provenance or projection rule, not a stale UV
      array.
- [ ] Package texture image files with the `.nodecraft` document or reference a
      project-relative path; absolute user paths must not leak into the public
      snapshot or make a document non-portable.
- [ ] Add sRGB/linear texture sampling and colour-space tests; a colour Look and
      a textured Look must agree on how base colour is encoded.
- [ ] Treat STEP texture/material export as separate exporter work. AP242
      presentation styles can carry colour and some surface-style data, but PBR
      roughness/metallic and arbitrary textures are not portable in the same
      way; document what is written and what viewers will ignore.
- [ ] Keep textures display-only until a STEP round-trip target is specified,
      and never feed display materials into exact geometry or measurement.

## SolidCore ops and blend

- [ ] Upstream: one half of the two-cylinder union join is classified `Smooth`
      by `topology::edge::blend_edgeuse_or_why`, although both adjacent faces
      are the two cylinder surfaces. Neither `ssi_split_curves_at_singular_points`
      nor `ssi_tangent_curve_tracer` changes it. See
      `docs/upstream-solidcore-cylinder-join.md`.
- [ ] Add the upstream regression that builds the two-cylinder union and
      asserts both join edges are blendable. Filleting both is what NodeCraft
      needs for an all-round join.
- [ ] Until upstream resolves it, keep skipping the Smooth half in NodeCraft so
      the graph evaluates instead of failing the whole fillet.

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

## Display LOD and solidcore-tess

- [x] Bridge kernel B-reps to `solidcore-tess` through `solidcore-convert`
      (`to_record`/`to_kernel`) and use the upstream planner/evaluator in the
      mesher workers.
- [x] Add a legacy-vs-tess parity harness
      (`nodecraft-engine --test tess_parity`) comparing bounds, face groups,
      edge counts and triangle counts for box, cylinder and sphere.
- [ ] Choose a display tolerance/angle target: the upstream tessellator currently
      produces 10–20× more triangles than the legacy bridge on the same
      primitives while matching bounds, faces and edges.
- [ ] Treat `solidcore-tess` as still under active upstream development. Review
      the pinned revision's planner and changelog before deleting the legacy
      mesher; the upstream API/behaviour may still change between pins.
- [ ] Find out why the density is unchanged: scaling the NodeCraft chord
      tolerance by 2×, 4× and 8× left box/cylinder/sphere triangle counts at
      124/672/960. Inspect `TessPlan`/`PlanStats` and the planner's display
      rules to identify whether there is an internal floor, a planar-face
      subdivision rule, or a missing plan option.
- [x] Check whether the density comes from the kernel→record conversion:
      upstream's own `box_10x6x4` fixture record and the converted kernel box
      both tessellate to 124 triangles, so conversion is faithful and the
      density is the upstream planner's own display output.
- [ ] Ask upstream (or inspect the planner changelog) whether a coarser display
      plan is intended for planar faces; do not treat the 124-triangle box as a
      NodeCraft conversion bug.
- [ ] Do not compensate for that density with LOD alone; fix the display plan
      first, then use LOD for distance/screen-size reduction.
- [ ] Add the upstream commit/date to the pin review so a change in tessellator
      behaviour is visible when updating SolidCore.
- [ ] Wire `solidcore_tess::simplify::levels` into the display path for LOD
      tiers, keyed by full-mesh key plus screen-size targets.
- [ ] Keep simplified levels display-only: they carry positions, triangles and a
      maximum surface error, but no `tri_face`, face groups, edge polylines or
      seam/smooth flags.
- [ ] Keep picking, selection, edge rendering and export on the full
      tessellation until upstream preserves face provenance through
      simplification, or simplify per face group.
- [ ] Add LOD regression coverage: each level must report finite error, stay
      within its deviation budget, and never be used for exact output.
- [ ] Delete the legacy `nodecraft-mesh` display path only after the STEP,
      primitive and sketch/spline mesh-only paths are measured against the
      parity harness.
