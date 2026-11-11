# Changelog

## Unreleased — 2026-11-09

### Dependencies and build

- SolidCore, its STEP/tessellation modules and the SolidCraft viewport/sketch
  crates now come from the vendored SolidCraft tree at
  `c314b6d26ca756184ef2bdd1742f5056854d3d07`. The separate SolidCore pin and the
  `solidcore-record-new` alias are gone; one lockfile revision covers all of
  them.
- The SolidCraft split moved the shared camera/scene/rasterizer to
  `solidcraft-viewport` and its wgpu pipelines to `solidcraft-viewport-gpu`.
  NodeCraft now imports those crates, converts `Aabb3` to
  `solidcraft_viewport::vec::Bounds` at the camera seam, and no longer depends on
  `solidcraft-render`/`solidcraft-gpu`.
- The new tessellation and blend/Boolean/STEP fixes are in. The
  `1000_Bool_Circ` display mesh reduced from about 157k to 45k faces and evaluates
  in ~167 ms on this machine.
- Removed the dead-code set left by earlier refactors and applied workspace-wide
  unused-import/mut cleanups. `cargo check --workspace --all-targets` and the
  release app build are warning-free.

### Inspector

- All inspector control rows now share one label/control layout: equal left and
  right insets, one row height and gap, number rows sized from the full control
  column, and a scroll bar placed by ScrollArea layout rather than a negative
  margin. The bar keeps its normal hover/drag hit area.
- Long labels truncate with the full label on hover. Exporter and STEP-import
  fields use the same rows.

### Architecture

- Documented the target graph definition/instance model: groups, loaded
  `.nodecraft` definitions and reusable base features are one concept; graph
  instances flatten to a single flat evaluation graph; canonical definitions are
  edited once and shared by instances. The engine still receives no nested
  evaluator, cache owner or kernel pool. Status: target design, with current
  `group`/`linkedpart` gaps listed in the plan.

- Graph Input/Output marker inspectors now edit the root document interface
  outside definition tabs, resizing markers and removing dropped-port wires
  atomically.

- Extracted the root-interface resize commands and added a regression covering
  marker port resizing, removed wires and undo.

- Graph instances now expose **Make unique** for local definitions: the
  definition is cloned with a fresh id/name and only the selected instance is
  repointed.

- External graph instances now expose **Make local**: the external definition and
  its reachable local definition closure are imported into the current document
  with fresh ids/names, and only the selected instance is repointed.

- Added an end-to-end flat-cache regression: definition geometry edits rebuild
  the flat graph, while definition renames and layout moves reuse it.

- Introduced `ViewKey(document_id, definition_id)` and moved graph-tab identity
  metadata onto it as the first workspace-view groundwork step.

- Added definition-scoped session commands: edits use body node ids and land in
  the parent document as one atomic, undoable `SetDefinition`. This is the
  session primitive for definition-backed graph views.

- Definition-tab edits now route through the parent document session via
  `apply_definition_commands`; the visible child session is refreshed as a
  projection without owning the edit history. The projection itself remains
  until the canvas can render definition views directly.

- Definition-tab undo/redo now operates on the parent session and only accepts
  the next parent history entry for that definition body, so root edits are not
  consumed from a graph view.

- Definition tabs no longer flush a child session on close: routed edits already
  live in the parent session, and projection refreshes use derived commands with
  no child history.

- Undo/redo enablement now follows definition-scoped parent history when a graph
  tab is active, so definition-tab history is available in the menus.

- Inactive definition tabs no longer keep a projection session in
  `DocumentStore`; the projection is rebuilt from the parent definition when the
  tab is activated. This removes the second-document lifetime for inactive
  graph views.

- Definition projections are no longer registered in `DocumentStore`, even while
  active; only the in-memory `self.session` projection exists for rendering.

- Replaced the inline-payload definition flatten bridge with direct expansion:
  group and graph-instance nodes now expand in one recursive pass, including
  nested local and external definitions, with cycle/missing-definition errors.
  Added nested-definition coverage.

- Graph Input/Output markers inside definition tabs now resize the parent
  definition interface and propagate to instance caches/wires, making IO edits
  safe from either the root or a graph view.

- Root-tab undo/redo now refuses to consume a pending definition-body history
  entry, and menu enablement reflects that scope boundary.

- Parent session history entries now carry a scope tag (`root` or
  `definition:<id>`); definition undo/redo and root/graph conflict checks use the
  tag rather than command shape.

- Added a per-tab **Nodes only view** toggle that hides the 3D viewport, as the
  first optional-3D editor step.

- Nodes-only tabs now expand the node editor across the workspace while keeping
  inspector space, instead of leaving the normal split-panel width.

- Tabs now choose Split, Nodes only, or 3D only view modes; 3D-only hides the
  node editor while retaining the inspector.

- Node-editor pane width is now saved and restored per tab, so each view keeps
  its own layout.

- Added **Duplicate Tab** for root documents. Multiple tabs can now view one
  stored document session with independent layout, selection, camera, breakpoint
  and tab-scoped undo. History entries are tagged with the originating view id.

- Duplicate Tab now also duplicates definition views; definition history is
  scoped per graph tab, so two views of one definition no longer share an undo
  boundary.

- Persist the inspector pane width per tab alongside node-editor width, and add
  scoped-history coverage for independent tab undo boundaries.

- Evaluation caches are now partitioned by document id instead of tab id, so
  duplicate root/definition views of one document share a cache partition.

- Documented the remaining definition projection as a rebuildable view cache
  rather than a second document session: edits, history and persistence remain
  on the parent document.

- Added an interleaved root/definition history regression: each view consumes
  only its own top entries, and the other view resumes after it is resolved.

- The node-editor header now shows the active definition name as a view
  breadcrumb.

- The view breadcrumb now shows `document / definition` for graph views and the
  document name for root views.

- Breadcrumb separators are drawn as chevrons aligned to the text centre line;
  no extra symbol font is bundled.

### Workspace and graph-model integration

- Display plan instances now carry kernel edge ids; ghost-edge selection uses
  those ids and arc-length midpoints instead of record indices/middle samples.


- Keep ghost edges alive after the moving layer upload, clear them when the edit
  ends, restore theme colour/alpha, and apply ghost instance transforms to
  their topology points.


- Ghost edges use a dark near-opaque material and keep their GPU moving-layer
  matrix in sync whenever the edge set is rebuilt.


- Edit-ghost edges now use a GPU scene/moving layer keyed by ghost-set changes,
  so a drag frame updates only the ghost transform matrix.


- The edit ghost now uses its own GPU uniform matrix and one instanced draw;
  dragging no longer rebuilds ghost chunks or instance buffers.


- Instance plan buffers are keyed by pass and colouring so two draw groups of
  the same plan cannot alias; colour-cache hits now update LRU order.


- GPU plan cache now keys instance buffers by draw pass, bounds colour variants
  with a small LRU, and gates instance-hash diagnostics behind the timeline flag.


- GPU plan path now draws each part with one instanced call and caches colour
  and instance buffers in resident plan meshes, updating instances in place
  when only matrices change. Gizmo previews use a per-frame uniform matrix
  instead of rebuilding plan instances.


- Tab restore now restores the quality tier, keeps pending framing rects across
  rapid switches, and relies on explicit frame preservation instead of a
  lingering preserve flag.


- Only the active document tab shows its close button; inactive tabs no longer
  expose or hit-test a close X.


- Tabs save and restore the complete 3D view (camera, lens shift, fitted/quality
  state and framing rects), with resize-aware distance and pivot correction.


- Extracted pure resident-plan byte/recency bookkeeping with A→B→A and
  over-budget eviction tests; recorded a timed tab-switch log showing 0 uploads
  on returning to both documents.


- Replaced the resident plan LRU deque with O(1) recency stamps, maintained
  resident bytes at insert/remove sites, and introduced a named `PlanDebug`
  struct with residency diagnostics and logic tests.


- Corrected the resident plan LRU so used entries are MRU and removed
  per-frame order rebuilding; eviction now runs only over the byte budget.


- Added UI-thread/gizmo/edit-overlay timing under `NODECRAFT_TIMELINE=1` and
  stopped rebuilding the merged breakpoint mesh on the UI thread when resident
  plans can carry the edit ghost.


- Added env-gated (`NODECRAFT_TIMELINE=1`) evaluation/display/plan timeline
  logging. A small-forrest first edit measured 18 ms exact, 1.4 ms display,
  zero plan uploads and 677 µs apply, so the reported 5 s delay is not on the
  control-path edit pipeline.


- Missing display overlays no longer remove the breakpoint, deferred display
  frames expire after 250 ms, and each tab restores its last applied display
  frame before re-evaluating.


- Reset the shared display pipeline at the session-install choke point and
  require the newest snapshot before submitting camera-quality frames, preventing
  one tab's display state or quality request from leaking into another.


- Stage display frames until their plan set is resident so edges, picking and
  triangles activate together; rekey the test display-eval cache by content
  hash; cover unsupported GPU-plan detection with a display test.


- Added painted plan-chunk/instance counters. Forest boot instrumentation shows
  the plan pass draws the correct chunks and instances while the shaded result
  is not visible in the capture, pointing at draw order/depth rather than
  plan upload.


- Do not submit camera-quality display frames while an exact display frame is
  in flight; the single latest-frame slot previously let a quality request
  overwrite the first editable frame after load.


- Show the full CPU breakpoint mesh while the background GPU tessellation
  pipeline is still compiling. Previously the plan path had no chunks for the
  first ~30 s, so only the edge outline updated.


- Quality-tier display frames now carry the snapshot id they were planned from
  and are discarded if a newer exact snapshot has arrived. This prevents the
  first post-load edits from being overwritten by stale-quality geometry.


- Apply the all-scope transform preview to GPU plan instance matrices. Gizmo
  drags previously moved the edges and ghost while leaving the plan-shaded mesh
  at the old placement, because the object renderer is skipped when plans are
  active.


- Added an uploaded-instance-buffer hash to plan diagnostics. Forest before/after
  hashes change together with the active plan hash, showing plan instance upload
  is not the stalled step.


- Added plan identity/transform hashing to the dev state endpoint. Forest
  measurements show the active PlanState hash changes after a transform with no
  pending/failed plans, so plan upload is not the stalled step.


- Staged GPU plan swaps now settle refused plans instead of stalling forever,
  count only real uploads against the per-frame budget, cancel a pending future
  set on edit-back, preserve shared preparation progress and fall back to the
  CPU mesh for unsupported plan sets. Added chunk/failed diagnostics.


- Added GPU plan diagnostics (pending/active plan count, chunk triangle count,
  instance count) to the dev state endpoint while investigating the unresolved
  forest mesh refresh issue.


- Keep requesting repaints while a staged GPU plan swap is incomplete. Without
  this, multi-plan swaps (for example a forest transform/dimension change)
  prepared only one plan and then stopped, leaving the old mesh visible while
  the new edge outline moved.


- GPU plan review: `set_plans` no longer restarts an in-progress staged swap
  when the same request is republished; documented the remaining failure,
  initial-load, ghost and CPU-LOD gaps in the plan update path.


- Fixed the evaluation progress meter: it is now `BrepComplete / total` (0–100%)
  instead of assuming a second display phase and a doubled denominator. Removed
  the obsolete display progress counter.


- GPU plan quality swaps are staged. New plans prepare resident meshes while the
  previous chunks remain visible; the swap happens only once every new plan is
  resident, at most one new plan per frame. This removes the single long repack
  pause when zoom crosses a quality tier.


- Exact node operations now dispatch through the kernel pool: Boolean/fuse,
  extrude feature booleans, Offset Faces, Look, placement realization and STEP
  import. Dependency chains stay consecutive; independent branches remain
  schedulable. Fixed realization to use a full B-rep copy so downstream fillets
  keep their pcurves.


- Node activity now follows the engine directly: `Start`/`BrepStart` begin a
  node's glow and `Complete`/`BrepComplete` end it. Removed the display-phase
  deferral (`activity_display_phase`/`activity_display_targets`) and the final
  pulse; leftover active nodes are only faded out as a safety net when the
  exact result arrives.


- One exact result per job. Non-displayable evaluations publish one empty
  snapshot/frame, and the app derives pending display state from the snapshot
  channel. Removed the `exact_final` flag and the `awaiting_display` hand-off
  that could cancel an empty frame before it applied.


- Display frames now track their own `displayed_target`, so a breakpoint change
  refits while ordinary edits preserve the camera. Display errors update the
  status line. Evaluations with no displayable part publish an empty snapshot
  and frame, so stale geometry is cleared and a later zoom cannot bring the old
  body back.


- Addressed the display-ownership review: quality frames are matched by their
  request revision (not snapshot id alone), exact reuse keys include comparison,
  edit-preview and feature-ghost inputs, `eval_target` is set by the exact
  handler, display failures use a separate display error field, and a missing
  breakpoint frame clears stale viewport geometry.


- The app now owns display snapshots through a `DisplaySnapshot` channel; the
  exact `Evaluation` carries no display fields or snapshot. Display requests
  carry a stable `snapshot_id` plus quality, and frames are matched by snapshot
  id rather than evaluation revision. Deleted the legacy display branch and the
  `display_complete`/display-only flags (the remaining exact flag is
  `exact_final`).


- Removed the legacy `EvalWorker` display branch and its `mesh_part` /
  `brep_display` / engine-display paths. The exact worker now returns only exact
  results; exporters stay on the exact path. `Evaluation` no longer carries
  display meshes, plans, overlays, tessellation diagnostics or display timings.
- The headless `--check` path now plans its mesh through the display service
  planner instead of reading a display field from the exact result.


- Split the service path into `apply_exact_result` (node/evaluation state) and
  `apply_display_frame` (viewport plans, meshes, overlays, diagnostics and
  picking). Both the normal deferred frame and a zoom quality frame use the
  shared display handler, so neither display path resets node state.
- Display service failures now publish an error frame for the exact revision;
  the fixed 10-second wait is gone.


- Display-only quality frames now have their own apply path. They update only
  plans, meshes, overlays, diagnostics and topology picking; they do not touch
  evaluation status, errors, activity, glow or `eval_pending`. Removed the
  `pulse_glow` flag that was masking this boundary.


- Quality-only zoom changes now reuse the exact snapshot's overlays too, not
  just the breakpoint. The exact pre-pass carries its `ExactSnapshot` and
  feature-ghost flag into the app; both plain and comparison/edit-overlay
  sessions submit the new quality straight to the display worker. Verified with
  `/state`: overlay zoom keeps `evalPending` and `nodeActivity` at zero.


- A camera quality-tier change now submits a `DisplayRequest` directly to the
  display worker using the stored exact snapshot. It no longer calls
  `evaluate()`, sets `eval_pending`, resets node activity or enters the legacy
  planner. Added quality-tier hysteresis and a `/zoom` development endpoint.
  Verified with `/state`: zoom changes `previewDetail` while `evalPending` stays
  `None` and `nodeActivity` stays zero.


- Addressed the follow-up body-identity review. Prewarm is limited to leaf
  producers so transforms keep shareable plans; body provenance is `Option`
  rather than a zero sentinel; raw requests without cache keys derive a
  recursive content seed; difference overlays require engine keys instead of
  constant fallbacks; and added end-to-end scatter and exact-difference key
  tests.


- Fixed body-plan identity after the display-service review. `BodyInstance`
  carries the producing node's content key and body index through placement,
  combine and scatter, so instances of one B-rep share one plan and one resident
  GPU mesh; a transform no longer replans downstream geometry. `plan_part`
  deduplicates equal keys before its parallel planning pass, and `PlanCache`
  now has a non-zero default capacity.
- Difference overlays use engine-provided keys derived from the before/after
  content keys, so inspecting a different feature cannot hit the previous
  difference's cached plan. Added regression tests for shared instance plans and
  transform-preserved body keys.


- Updated SolidCore/SolidCraft to `525171eb861f172bdc1e23c7c5c070a44a8fc831`
  (all pins and lockfile together). Adapted the viewport frames to the new
  `Frame::turned_depth` field. Kernel changes include typed failure reasons,
  blend/boolean/offset fixes and B-spline surface handling.


- Exact Part body keys now use the producing node's content hash instead of its
  node id. A transform changes the B-rep while the node id stays stable; with an
  id-based key the display plan cache returned the old breakpoint plan, so the
  edited branch ghost moved but the breakpoint mesh did not. Added a dev
  `/set` control endpoint for direct transform diagnosis.


- The exact pass now evaluates the edited node's input and feature-ghost input.
  These targets were still being passed as `None` after the preview worker was
  removed, so the display service had no exact edit geometry to plan and
  transform drags had no branch ghost. The dev state endpoint also reports the
  edit preview target and gizmo drag state for diagnosis.


- Transform drags attach the service-published edit ghost when the drag starts.
  The ghost can arrive before the gizmo interaction, and the old attach path only
  ran when a preview result arrived, leaving the drag with no branch preview.


- Resident GPU meshes are now evicted when their plan leaves the current normal
  or ghost draw, and each entry pins its plan Arc. This prevents a freed plan's
  address being reused by another plan while its stale GPU mesh is still cached,
  which could draw the wrong body after a transform/plan change.


- Added a resident GPU plan-mesh cache. Each plan is evaluated and recorded once,
  then reused across poses, ghost matrices and colour changes; only the
  instance/colour buffers are rebuilt. This removes full-scene `TessGpu`
  repacks from interaction frames.


- Removed the dedicated preview worker. The main exact request now includes the
  edited input, and the display service plans its `EditInput` overlay alongside
  the breakpoint, comparison and feature ghosts. Transform/edge edit previews
  now consume that service frame instead of a second exact evaluation.


- Moved comparison, feature-ghost and exact difference overlays into the
  display service. `DisplayRequest` now carries exact overlay parts and a
  feature-ghost flag; `DisplayFrame` returns overlay meshes/plans and the
  fillet/chamfer feature plans. The app merges these into one evaluation, with
  the old combined pass retained for exporters and transform edit previews.


- The app now polls the display service directly. The exact pre-pass marks a
  deferred display, the app stores that exact evaluation, and `poll_evaluation`
  merges the matching `DisplayFrame` through the shared final-evaluation path.
  Removed the per-display bridge thread; exact and display work are now fully
  decoupled.


- The exact worker no longer waits for display planning. It publishes the exact
  pre-pass, submits the breakpoint `DisplayRequest` and immediately accepts the
  next edit; a small frame bridge waits for the matching `DisplayFrame` and
  returns the final evaluation. Superseded bridge threads abort through an
  atomic revision.


- Routed the plain breakpoint display through the lock-free display worker.
  The exact pass now hands a `DisplayRequest` to the service and consumes its
  latest published `DisplayFrame`; comparison, ghost and edit-preview paths
  still use the old combined pass for now.


- Added the lock-free display worker boundary: latest request slot, latest
  `DisplayFrame` slot, atomic revision checks, repaint callback and worker-owned
  plan cache. The app is not wired to it yet; this is the service runtime the
  exact worker will hand breakpoint/overlay snapshots to.


- Added the display-service contract: `ExactSnapshot`, geometry-carrying
  `OverlayRequest`s, `DisplayRequest`, `DisplayFrame`, and a synchronous
  `DisplayService` entry point. The contract has no graph or viewport state and
  plans breakpoint and overlay bodies through one planner.


- Started the display-service split. Exact Part/BRep handles and transforms now
  live in `nodecraft-exact`; `nodecraft-display` re-exports them for
  compatibility. Plan cache entries are keyed by stable `BodyKey` plus quality
  and no longer pin the B-rep, so plans survive engine geometry eviction without
  keeping every body alive.


- Clear node activity when an evaluation completes. A cached plan-only display
  could finish without a final display-phase progress event, leaving the viewed
  node's activity glow stuck on.


- In feature edge-selection mode, the candidate edge field is drawn at one
  third opacity, and selected edges are painted fully opaque on top. This makes
  the current selection clear while keeping the input topology visible.


- Fillet/chamfer edge selection is limited to the feature's input B-rep. While
  the input is shown as the edit ghost, hit-testing, stored id checks, centre
  matching and the visible-edge set use only the ghost's B-rep edges; the
  downstream breakpoint's visible edges are no longer selectable.


- Skip pole-fan triangles in the triangle-edge overlay. Their edges all meet at
  one point, which piled into a dark disc at sphere poles; the rest of the
  triangulation stays visible.


- Reconnected Translucent to the main PBR path. Plans stay enabled with a
  transparent PBR pipeline (no depth writes) instead of falling back to the
  edge-only object mesh, and the object path no longer draws a duplicate solid
  when a plan is active.


- Fixed the plan triangle-edge overlay mapping: each face's triangle indices are
  local to that face, so the line buffer now offsets them by the face's first
  global `GpuMesh` vertex instead of one part-wide base. The Triangles setting
  draws on the surface rather than connecting unrelated vertices.


- The viewport Triangles setting now works on the GPU plan path. Plan triangle
  edges are built from each `TessPlan` face's triangle list into a line-index
  buffer and drawn by a flat line-list pipeline over the PBR result.


- Ghosts now use a flat, unlit shader instead of the PBR object shader. The
  shader keeps the authored ghost colour and multiplies the instance alpha by
  the live opacity uniform, so Ghost opacity remains connected.


- Removed the standalone CPU/object ghost renderer. Edit and feature ghosts are
  now drawn only by the plan GPU pass; a branch ghost waits for its display
  plans instead of uploading a second CPU mesh. Plan edges cover the ghost
  overlay when plans are present.


- Updated SolidCraft/SolidCore to `3bdb61a` and migrated STEP import/export from
  the removed `step-import`/`step-brep` crates to `solidcore-step`.
- G2/G3 fillets now pass the kernel's `GN_DEFAULT_THUMBWEIGHT`; the current
  upstream G^n section refuses the old 1.0 value with status 1007.


- Restored the breadcrumb above the node canvas and raised the File menu instead:
  breadcrumb uses Foreground (same as the canvas, created after it), while the
  File popup is a Tooltip-order popup so it still covers breadcrumb, tabs and
  node content.


- Fixed the breadcrumb bar painting over the File menu by giving all panel
  overlays one shared `Middle` layer constant below Foreground menus/popups.
  The tab strip now uses the same constant.


- Fixed the document tab strip painting over the File menu. Tabs now use the
  Middle layer, above the title-bar drag interaction but below Foreground
  menus/popups, with a regression test for the layer invariant.


- Added `nodecraft-bench`, a timeout-safe benchmark runner. It keeps one worker
  alive, content-addresses node cache keys so a one-field edit only invalidates
  affected branches, and reports cold/warm/edit wall and evaluation timings,
  triangles, response bytes and cache hits. It is wired to the 500-instance
  forest fixture in `benchmark/`.


- Added a bounded benchmark folder with `benchmark/forest_500.nodecraft`, a
  500-instance forest for the inner loop. The 5,000-instance samples stay
  stress-only.


- Review cleanup: removed the unused multi-shape `NodeOutput` variant, repaired
  the operand doc comment, and added `Part::translated` so deferred translation
  shares the same instance-level path as deferred mirroring.


- Centralized the Part mirror path in `Part::mirrored`: a big Part keeps its
  bodies and only records the mirror transform, so a mirror costs no B-rep
  copies. Realization still runs `ExactNodeTask::Mirror`, the same task the
  operand path uses. Added a regression test that the deferred Part mirror and
  the direct mirror task produce the same bounds.


- The task Boolean path now calls the same n-ary `boolean_lists_batched` core as
  the exact Part path, then folds disjoint result pieces back into the single
  shape required by operand/export callers. Legacy operand mirror and translate
  now dispatch through `ExactNodeTask` too.


- Added `ExactNodeTask::ExtrudeSketch` and routed the Part path, the test
  display path and the legacy operand path through it. Sketch extrusion no
  longer has three separate implementations.


- Unified the heavy Part operations behind `ExactNodeTask`: fillet, chamfer,
  extrude and shell in `evaluate_part` now dispatch through the same task
  implementation used by the kernel pool, and the shell task uses the record
  shell back end only. Multi-solid fillet/chamfer/extrude operands now fail
  explicitly instead of silently using the first body.


- Started collapsing duplicate node-operation paths. Removed the unreachable
  Boolean/fillet/chamfer/shell/extrude builders from `prewarm_primitives`; those
  nodes are evaluated only by the exact Part path. Gated the legacy
  `operand_brep`/`operand_breps` implementation to tests, leaving the batched
  Part Boolean path as the production path.


- The viewer's Ghost opacity control now updates the GPU plan ghost instance
  alpha, not only the legacy object ghost uniform. Moving the slider rebuilds
  the ghost plan chunks at the new alpha for both edit ghosts and feature
  ghosts.


- The transform edit ghost now applies the live gizmo matrix to its GPU plan
  chunks, so the ghost body is drawn at the preview placement instead of being
  hidden at its old position. Main evaluation updates no longer clear a live
  edit ghost between preview frames.


- LOD/quality changes now send a display-only job carrying the app's cached
  exact Evaluation. The worker skips the exact pass entirely and plans the
  cached B-rep into a new TessPlan/GPU mesh. Exact/cache timings are cleared so
  the status line reports only the live display swap.


- Cache the exact `Part`/B-rep result per document request. Camera/LOD quality
  changes (window resize, orbit) now reuse the cached exact handles and only
  re-plan display data instead of walking the DAG again.


- Parallelized plan-only body meshing across up to 16 threads. The 5,000-tree
  forest sample dropped from ~5.9 s to ~0.54 s mesh time; total evaluation is now
  ~0.6 s. Plans retain their edge polylines for CPU picking, while the viewport
  renders the GPU `GpuMesh`.


- Added `nodecraft_layers::plan_kernel` and
  `nodecraft_display::plan_part`: the plain display path now produces TessPlans
  and plan edge polylines without evaluating a CPU triangle mesh. The GPU plan
  renderer draws the mesh; CPU mesh generation remains on comparison, ghost and
  export paths. Cup sample display mesh time dropped to ~6 ms.


- Feature-only blend plans are now opt-in (`mesh_part_with_features`) and built
  only while a fillet/chamfer is selected. Previously every display mesh clone
  searched blend marks and cloned the whole `TessPlan` for feature ghosts, which
  could make large meshes appear to hang. Display sizing now uses the kernel's
  approximate bounding box: a vertex-only extent under-sizes curved bodies and
  the tight exact box can spend tens of seconds on a large Boolean/imported body.


- Profiled the torus/cylinder/fillet graph: Boolean work and duplicate validity
  checks dominated. Prewarm now schedules only cheap independent B-reps; heavy
  Boolean/fillet/chamfer/shell/extrude nodes use the exact Part path's batched
  Boolean instead of the generic ops prewarm path. Display body size uses the
  vertex extent instead of a tight exact bounding box. Full validity checks are
  debug/test/`NODECRAFT_VALIDATE_RESULTS=1` only; release saves 21% on the graph
  before the prewarm fix, and the prewarm fix takes it from ~950 ms to ~157 ms.


- Removed the display plan cache from `EngineCache`; the worker and session
  runtime own `nodecraft_display::PlanCache` beside exact evaluation.
- Production engine evaluation no longer meshes: exact mirror Boolean operands
  now use exact mirrored B-reps instead of a display tessellation, and the
  engine display methods/helpers are test-only. The engine production API is
  exact Part/BRep only.


- Documented the display/export scheduling decision: interactive display stays on
  the evaluation thread for now; a future shared executor should isolate an
  export lane so fine export plans cannot starve viewport planning.


- Removed `Response.meshes` from the engine protocol. The worker and session
  runtime build JSON `meshes` from `nodecraft-display`; the CLI and STEP scripts
  keep reading the same JSON field.
- `nodecraft_engine::evaluate` is now test-only; production engine evaluation is
  exact only. The app bridge, app exporter, worker, session runtime and
  `step_report_mesh` all use the display/export service.
- The display plan cache is no longer part of the engine response path; the
  worker and session runtime own display plan caches beside exact evaluation.


- Migrated `step_report_mesh` to `evaluate_exact` plus
  `nodecraft-display::protocol_meshes`. The report tooling no longer asks the
  engine for a mesh.


- The app export action now runs the exact pass, finds the exporter input's
  exact Part and calls `nodecraft-display::export_step`. Exporter runs no longer
  use the engine display pass; the viewport mesh for the export action is built
  from the display service.


- Added exact STEP export to `nodecraft-display::export_step`: realized B-reps
  are written, re-imported for validation, and export metadata (units, header,
  schema, product, layer, material, colour) is applied without engine display
  state. A regression exports a box from an `ExactPart`.


- Added `ExactPart::realized_breps` to the display/export service: placement,
  mirror and scale are baked into standalone B-reps for exact export without
  going back through the engine. This is the exact-geometry half of moving the
  STEP exporter off the engine display pass.


- Added the display/export service boundary: `nodecraft-display::ExportOptions`
  and `export_mesh` request export tolerances and optional closed-topology
  validation independently of the viewport. Export plans are cached under their
  own quality key and are not reused from a coarser display plan.


- The Rust worker now runs `evaluate_exact` and builds its JSON `meshes` through
  `nodecraft-display::protocol_meshes`, so the worker no longer calls the engine
  display pass. `Response.meshes` is becoming a display-service compatibility
  payload rather than an engine-produced mesh response.


- Removed `EngineOutput.plans`; the exact engine no longer exposes display
  plans. Normal/ghost/preview/comparison display plans come from
  `nodecraft-display`; the transitional engine display fallback returns no plans.


- Exact comparison now returns before/after difference B-reps from the exact
  pass (`ExactComparison`), and the app meshes the before/after bodies and
  difference through `nodecraft-display`. The normal target, ghost, preview and
  comparison display paths no longer need the engine's display pass.


- `PreviewWorker` now uses the exact pass plus
  `nodecraft_display::mesh_part` instead of the engine display pass, so edit
  ghosts and their input-body plan edges come from the display service. The main
  ghost overlay also goes through `mesh_part`; comparison/difference remains on
  the transitional engine path.


- The app's normal target display path now runs through
  `nodecraft-display::mesh_part` using exact Part/BRep handles from the exact
  engine pass. Plain targets no longer call the engine display pass; comparison,
  ghost and edit-preview cases still use the transitional engine path.


- `nodecraft-display` now consumes exact Part/BRep handles directly:
  `ExactPart`/`ExactBody`, plan cache, plan-derived CPU mesh and GPU
  `PlanInstance`s. `EngineOutput.exact_parts` uses those display-side types, so
  the display service can mesh an exact result without a graph or engine display
  pass.


- Moved `DisplayTransform`, `PlanInstance` and face-colour indexing into
  `nodecraft-display`. The app consumes the display types directly instead of
  reaching through `nodecraft-engine`; the engine only re-exports them for its
  transitional display pass.


- Added `nodecraft-display`, the display-plan service crate. It owns
  `DisplayQuality`, the plan-only `PlanCache`, body tessellation and derived plan
  evaluation helpers. The engine's transitional display path now uses that cache;
  no graph evaluation lives in the display service.


- Removed the engine's target display-mesh cache. Display meshes are now derived
  from body plans each display pass; the engine caches no CPU meshes. The
  forest remains within budget (~1.06 s mesh) while the cache holds plans only.


- The body display cache is now plan-only. `EngineCache` stores
  `(BRep identity, quality) -> TessPlan + stats + kernel edge-id mapping`;
  cached display meshes are derived from the plan with `evaluate_plan`, which
  preserves B-rep edge ids. This removes the cached per-body CPU mesh and keeps
  the display plan as the source of truth.


- Started the engine/display boundary refactor: `EngineOutput` now carries
  `exact_parts` (immutable B-rep + placement handles) from the exact pass. The
  exact pass is proven not to create meshes; the display service can consume
  these handles next.


- Recorded the engine/display boundary refactor: the engine should return exact
  Part/BRep handles only; a separate display service should own plan creation,
  quality, `PlanStats` and a plan-only cache. The redundant body
  `(BRep, Mesh, Plan, Stats)` cache is to be removed, with CPU meshes/edges
  derived in the view layer.


- Included `preview_detail` in the display-mesh cache identity. A camera quality
  change can no longer reuse the previous merged mesh; body/plan caches remain
  keyed separately by their quality.


- Kept resident GPU plan chunks when an evaluation returns the same plan Arcs,
  transforms and face colours. Re-evaluating a breakpoint for an unchanged plan
  no longer rebuilds the GpuMesh; only a new plan/quality or draw mode does.


- Removed the per-body size clamp from preview plans and made view detail tiers
  explicit (close 8, then 6/4/2, far 1) with matching facet angles. This removes
  the case where a small body got full-quality facets while a large nearby body
  got coarse facets in the same view. Screen-space per-body LOD is still needed
  for fully consistent quality across a large assembly.


- Added view-dependent coarse display plans. The engine accepts a
  `preview_detail` request, tessellates each body at `solidcore_tess::preview`
  density (clamped so small bodies keep full quality), and the app selects a
  detail tier from the camera. `gigatest-forest-5000` drops from ~3.24M to
  ~2.31M faces and mesh time from ~1.15 s to ~0.94 s at the fitted view. Picking,
  ghosts, full edges and export remain on full-quality data.


- Surfaced the full `PlanStats` diagnostics: lenient, over-tolerance,
  unsampled/capped/strict edges and averaged poles now flow through `TessStats`,
  the body cache, `EngineOutput`, app evaluation, the viewport debug panel and
  `/state`.


- Recorded SolidCore/SolidCraft integration opportunities for the new pin:
  `PlanStats` diagnostics, `solidcore_tess::preview` bounded coarse-first
  display, per-body plan profiling, continuity-backed edge chains, typed
  boolean/check failures and sweep stations.


- Updated SolidCraft/SolidCore to `7e902aa47351bac0bc4235b7210ad93527479c5d`.
  The 26 kernel changes include the moved-mesher pole/face handling, the 50k-face
  plate performance work, sweep stations, projected curve-pair maxima, fillet
  fixes and typed boolean failure reasons. Moved app plan types behind the
  `nodecraft-layers::tess` boundary so the app no longer depends on
  `solidcore-tess` directly.


- Documented that GPU plan evaluation does not lower tessellation density: the
  forest still plans ~3.2M faces. Added a TODO for per-body `PlanStats`, a real
  chord/angle display-quality tier, view-dependent coarse plans and a density
  regression.


- Fixed GPU plan activation bookkeeping: plans are matched against the raw
  evaluation target (so nested/definition targets are not dropped), and view
  changes refresh the plan renderer state so leaving face-colour/translucency
  modes turns the GPU path back on.


- Face picking now derives triangle-to-face ids from the display target's
  `TessPlan` faces, so the selection result is the B-rep face id rather than a
  display-mesh group label. Added a regression for the mapping.


- Recorded the completed GPU plan/B-rep picking work and the remaining
  enhancements (analytic edge-curve intersection, plan-based face picking,
  hardware mesh-parity regression and separate plan/evaluation profiling).


- Cleared the pick-input plan cache with the edit ghost and added regressions
  proving display plans carry B-rep edge polylines and that the input-body ghost
  exposes body-scoped plan edge ids.


- GPU plan drawing now keeps each body plan in local space and applies its
  placement, mirror or scale as a per-plan instance matrix. This removes the
  reflection/scale CPU fallback and avoids cloning plans just to place them.


- The GPU plan pipeline compiles its evaluation shader in the background
  (`TessLater`) and starts on the CPU display mesh until the plan evaluator is
  ready, matching SolidCraft's startup path.


- Added the GPU display-plan path. The viewport now evaluates captured
  `TessPlan`s with `solidcraft-viewport-gpu` (`TessGpu`/`GpuMesh`) and draws the
  result through NodeCraft's existing PBR shader; plans are chunked at the
  upstream vertex limit, per-face colours are carried by a per-vertex buffer, and
  the CPU display mesh remains the fallback for reflections/scales, face-colour
  modes, translucency, picking and edges.
- Feature edge selection now prefers the edit-input body's B-rep plan edges
  (`TessPlan::edges`) from the preview worker, with body-scoped ids, instead of
  only the displayed result/display mesh. This is still sampled B-rep edge
  geometry, not analytic curve intersection.


- Display tessellation now retains and exposes its `TessPlan`. `TessDisplay`,
  the tessellation pool, body-mesh cache and `EngineOutput` carry plans (with
  instance transforms) to the app alongside the existing CPU mesh; the JSON
  response is unchanged. This is the plan-creation half of GPU tessellation;
  the viewport does not evaluate plans on the GPU yet.


- Extended the pure B-rep picking TODO: feature edge/face selection must target
  the edited node's exact input body (including multi-body, placement/mirror,
  suppression and nested-instance cases), independent of the displayed
  breakpoint/result mesh.


- Recorded the integration boundary for GPU plan tessellation: keep NodeCraft's
  PBR/ghost/comparison/overlay renderer and picking, and port only the
  `TessPlan`/`TessGpu`/`GpuMesh` tessellation path, plan cache, background compile
  and CPU fallback. Per-face colouring must be carried from `vertex_face` without
  rebuilding CPU object vertices.


- Recorded the SolidCraft reference implementations for view-dependent LOD,
  display quality, `GpuMesh` plan caching, viewport plan selection and picking.
  These are guidance for the next development order; no code was copied.


- Added follow-up TODOs for pure B-rep edge/face picking and GPU tessellation
  from `TessPlan` (`TessGpu`/`GpuMesh`), including exact selection independent of
  LOD, plan caching, CPU fallback and the distinction between plan generation
  cost and GPU plan evaluation.


- Recorded the coarse-first tessellation experiment: a 1.2 tolerance / 60° initial
  plan reduced the forest by only ~1% of faces and added a refinement pass, so it
  was reverted. The note calls for per-body plan statistics before trying
  view-dependent plans or a coarse preview.


- Combine operands feeding a Boolean are now expanded directly from their
  connected shapes. This avoids materializing an intermediate Part, copying every
  body through `realized_solids`, and fixes multi-input Combine as a Boolean
  operand. Standalone Combine-to-Part still uses the normal Part path; its main
  cost is tessellation/merging, not the body-vector concatenation.


- Replaced the Mirror node icon with the Andrsrxn butterfly icon.


- Documented the remaining wgpu upload risk: display preparation is threaded, but
  the UI/render thread still creates/uploads all pending object buffers in one
  frame. The note records progressive chunk upload, coarse-first upload,
  worker-side device/queue upload options and the need to time the first frame
  separately.


- Added a display-preparation worker. Protocol-mesh conversion, crease-normal
  splitting, edge extraction and GPU object-vertex/index preparation now run off
  the UI thread for large meshes; only buffer creation/upload and drawing remain
  on the UI/render thread. Large loads keep the old display until the prepared
  mesh is ready, and stale preparations are discarded by generation.


- LOD generation is now lazy and off the UI thread: loading a large document no
  longer simplifies millions of triangles before showing the full mesh. Levels
  are generated on the first zoom-out and applied when ready. `gigatest-forest`
  evaluates in about 1.1 s without the previous synchronous LOD stall.


- Evaluation status now reports display mesh counts (target, comparison, ghost)
  and cache reuse by kind: BRep/Part hits versus display-mesh reuse. The engine
  exposes per-evaluation BRep, Part and display-mesh hit counters instead of only
  an aggregate.


- Status now reports cache reuse per phase: BRep hits/misses from the exact
  pre-pass and display hits/misses separately, instead of showing only the final
  display pass’s counters.


- Re-evaluate now bypasses only the exact pre-pass caches; the display pass
  consumes the freshly warmed exact results instead of bypassing them and
  rebuilding the graph serially. On `1000_Bool_Circ` re-evaluate dropped from
  about 2 s to ~58 ms after the exact pre-pass.


- Documented that display meshes are planned once at a fixed tolerance and LOD
  only removes triangles; view-dependent retessellation is not implemented. The
  note records the upstream view-based tessellation primitives and the cache,
  threshold and provenance work required to adopt them.


- LOD now covers small primitives too (≥500 triangles instead of 20k), and fit uses
  the full mesh; zooming out selects coarser levels, so zooming back in visibly
  restores triangles.


- Wired solidcore-tess display LOD into the viewport. Large monochrome display
  meshes now carry simplified levels; the renderer selects one from camera
  distance / model radius. Edges, picking, selection and export stay on the full
  mesh. Levels are display-only and retain their measured deviation.


- Renamed NodeCraft's display worker from "mesher" to "tessellation": the
  implementation lives in `solidcore-tess`, while `TessellationPool` only
  schedules body tessellation. Removed the unused duplicate mesh kernel pool and
  updated settings/docs naming; old `mesher_threads` settings still load.


- Documented the open re-evaluate contract questions: cache bypass versus free,
  prewarm reuse, exporter bypass, progress/activity phases, nested-view targets,
  cold-versus-warm cache reporting, thread scheduling, cancellation and tests.


- The exact pre-pass now prewarms the target closure through the kernel pool and
  reuses those B-reps in the serial exact loop. Wide graphs parallelize independent
  branches, so exact thread settings affect interactive evaluation; `1000_Bool_Circ`
  measured about 194 ms with auto versus about 540 ms with one worker.
- Prewarmed B-rep reuse is restricted to nodes produced by the prewarm scheduler,
  so display/export evaluation cannot pick up an unrelated operand cache entry.


- Engine settings clarify that worker counts affect parallel exact dispatch and
  display-mesh bodies; serial Boolean chains and single-body displays may show no
  difference between 1 and auto.


- Engine settings now report the actual worker counts of the initialized exact
  pool and display mesher, making it possible to verify that the configured
  thread count reached the pool.


- The re-evaluate circular indicator is determinate across both passes: exact
  progress fills the first half, display progress the second half. No spinner
  state remains.


- The re-evaluate icon becomes a same-size circular progress indicator while an
  evaluation is pending: a rotating arc during the exact pass and a determinate
  arc during the display pass. It returns to the arrow when idle.


- Node activity now reflects the node's real processing span: process start begins
  the glow; mesh/display targets stay lit through their display pass and end on
  display Complete. Nodes that are not display targets fall back to ending at
  process stop.


- Restored the original activity handling: only the display pass emits
  Start/Complete activity events, giving one glow per node. Exact-pass progress is
  not sent to the canvas activity layer.


- Added an exact/display progress boundary so node activity starts once in the
  exact pass, stays lit through the display pass, and ends on the display
  Complete event. Boolean nodes no longer pulse twice.


- Activity is simplified to one glow per processing pass: node start starts the
  glow and node end ends it. The display pass no longer replays activity for
  nodes already swept by the exact pass.


- Node activity now holds from the exact-pass start through to the node’s display
  Complete event. Exact `BrepComplete` no longer ends the glow early.


- Node activity is deduplicated per evaluation: the exact pass provides the
  single sweep, and the following display pass no longer replays Start/Complete
  for nodes already swept.


- Removed the dead pause before interactive evaluation starts: exact-pass
  progress now drives node activity, and interactive requests skip the serial
  prewarm scheduler (measured as pure overhead on the 5,000-node boolean chain).
- Workload profiling and prewarm dependency propagation are linear; part-target
  ancestor checks use an id/type index instead of rescanning nodes.


- Target-closure discovery now builds reverse adjacency once instead of scanning
  every connection for every visited node. Large boolean chains no longer pause
  for seconds before the kernel phase; initial viewer target selection is linear
  too.


- Re-evaluate now bypasses cache lookups instead of dropping every engine/flat
  cache and resetting kernel/mesher pools. This keeps the fresh-evaluation
  guarantee without paying deallocation/reset latency on large B-rep and mesh
  caches; content hashes still invalidate changed nested graphs.


- Graph-node copy/paste now carries the referenced local definition closure,
  remaps definition ids/names on paste and rewrites nested graph-node references.
  Hand-written clipboard JSON that omits definitions synthesizes a passthrough
  definition from the graph node’s cached interface instead of failing
  `Missing graph definition`.


- Re-evaluate now clears every evaluation flat graph/cache partition, not only
  the active document’s top-level cache; kernel and mesher caches are cleared with
  it, and preview worker state is dropped.


- Transform previews now follow the same instance-context mapping as subgraph
  evaluation. Preview requests use the parent document and prefixed target;
  first-level Graph Input markers bind to the owning graph node’s upstream source,
  and preview targets map back to the body view.


- Graph-definition views now evaluate in the context of their owning flat graph:
  the tab’s instance path (`g1::g2`) is used to map body selections/breakpoints to
  prefixed flat targets, so parent inputs are evaluated inside the subgraph.
  Displayed target, node errors, activity and dirty nodes map back to the body.
  Transform previews and topology picking still need the same mapping.


- Recorded the missing contextual subgraph evaluation work: definition tabs
  evaluate their body without binding the parent instance’s inputs, so Graph
  Input has no external source inside the subgraph view.


- Fixed graph-node breakpoints for passthrough definitions: a graph whose body
  wires Graph Input directly to Graph Output now gets a synthetic flat combine at
  the graph card id, wired from the external input, so the viewer breakpoint has
  a target.


- Recorded a concrete autosave/recovery TODO, including dirty tracking, debounce,
  external-edit conflicts, untitled drafts, external definition flushes and the
  interaction with definition-cascade deletion.


- Deleting graph instances now computes local definition reference counts and
  cascades removal of uniquely-owned descendant definitions; tabs for removed
  definitions close. Shared definitions remain and their tabs stay open.


- Breadcrumb hover chips are fill-only with no outline.


- Breadcrumb hover chips use a smaller squircle radius so the highlight reads as
  a compact chip instead of a stretched pill.


- Breadcrumb layers are now interactive: each layer gets a hover squircle and
  clicking it navigates the current tab to that document/definition instead of
  opening another tab.


- Node layout updates in graph views no longer trigger evaluation or clear the
  cached mesh. Position writes persist/undo as editor state but are not
  evaluation input; node hashes and flat-graph fingerprints already ignore
  positions.


- Fixed an invalid auto-connect created when a Graph Output marker was selected:
  Graph Output has no output port, but the generic auto-connect policy treated it
  as a shape source and connected it into the next added node.


- Fixed graph-view interaction after adding a nested graph: definition projections
  now carry the parent definition table, so derived root updates can resolve the
  new graph node’s reference instead of failing hash validation and leaving the
  canvas frozen.


- Switching to a graph tab now clears stale canvas drag/resize/lasso/context
  state, preventing a previous view’s transient pointer state from blocking the
  new canvas.


- Structural edits in a graph view now rebuild the projection session
  immediately, so a newly added nested graph node appears without switching
  tabs; positional edits still use the lightweight derived update.


- Graph views now carry their full breadcrumb ancestry (document ❯ parent graph ❯
  child graph) and fit the canvas after adding a nested graph, so a newly created
  child node is visible immediately.


- Nested graph creation now remaps a conflicting definition id/name inside the
  definition-view transaction and rewrites the new graph-node reference, so a
  stale projection cannot block adding a graph inside a graph.


- Fixed “Duplicate graph definition …” when adding a graph inside a graph:
  definition ids/names are now allocated from the parent document, not the
  projection’s empty definition table.


- Nested graph creation now works from graph views: definition-scoped commands
  can add/update/remove definitions in the parent document, so a graph can
  contain another graph definition and undo/redo treats it as one transaction.


- Removed the legacy web viewer (`index.html`, `styles.css`, `app.js`,
  `viewer.js`, its evaluation planner/test and Three.js/Inter web assets). The
  native egui app and public site snapshot are unaffected.


- Inter is embedded in the executable on every platform. No separate bundle
  resource or runtime font-file probe is needed.


- Added the `andrsrxn:share-network` icon for graph nodes and graph-view tabs.
  Graph cards use it in the canvas; definition tabs show it left of the title.


- New graph definitions place their interface markers in one vertical column,
  spanning the available editor height, so first entry shows Graph Input at the
  top and Graph Output at the bottom.


- Completed the workspace/graph-model transition scope: definitions and
  instances, direct nested flattening, definition-aware validation/hashing,
  same-session graph views, tab-scoped undo, duplicate views, split/nodes-only/
  3D-only layouts, per-tab pane widths, breadcrumbs and per-document evaluation
  cache partitions. Multi-pane-per-tab, multi-scope evaluation, iterator, fold,
  externalise/pack and the public snapshot remain deferred.


- Added an external graph-reference save/load/flatten regression covering the
  canonical external document resolver.

### Workspace and graph-model integration

- Load/save now validate definition ids, unique names, interface ports and owned
  node/connection topology. Fixed the legacy group conversion duplicating
  existing definitions, which the new validation exposed. New graph creation
  also allocates unique definition names.

### Workspace and graph-model integration

- Extracted the graph-definition port command builder and added a regression
  covering root + nested instance cache updates, removed wires and undo.

### Workspace and graph-model integration

- Renaming a graph definition now refreshes cached labels on every referencing
  graph instance in the source root and nested definitions, not just the edited
  card.

### Workspace and graph-model integration

- Graph-instance inspector renaming now works for external definitions when the
  external document is open.

### Workspace and graph-model integration

- Port resizing now also updates graph instances nested inside other definitions,
  refreshes open containing-definition tabs, and drops their removed-port wires.

### Workspace and graph-model integration

- Fixed graph-instance cards not redrawing their ports after inspector port
  changes. The node-canvas geometry cache now includes graph instance
  input/output names, with a regression test.

### Workspace and graph-model integration

- Graph port counts can now edit external definitions when the external document
  is open; graph caches, open definition tabs and parent wires are refreshed the
  same way as local definitions.

### Workspace and graph-model integration

- Definition port resizing now works with an open definition tab: the body edits
  are flushed, the interface resized, and the open body session replaced with the
  resized definition so marker nodes stay consistent.

### Workspace and graph-model integration

- Restored `Num inputs`/`Num outputs` for local graph instances. Resizing updates
  the referenced definition interface, marker nodes, owned connections, graph
  instance caches and parent connections, through undoable commands. Editing is
  guarded while an open graph-definition tab exists; external and nested
  instance coverage remains.

### Workspace and graph-model integration

- Recorded the known regression that graph definition IO counts are temporarily
  read-only after moving ports from the legacy group payload to definitions.
  The definition-interface edit path is the next visible interface task.

### Workspace and graph-model integration

- Fold/accumulator execution is explicitly deferred. The iterator plan remains
  valid, and the definition kind/name may be reserved, but no fold
  implementation is part of the current transition.

### Workspace and graph-model integration

- Graph-instance inspector names are editable for local definitions; renaming
  updates the definition, cached graph-node label and graph tab title.

### Workspace and graph-model integration

- The tool dock Graph entry now creates a graph-instance node plus a new local
  definition with a default shape input/output. Added undoable
  `AddDefinition`/`RemoveDefinition` session commands and coverage.

### Workspace and graph-model integration

- Graph instances have an inspector **Open graph** action in addition to
  double-click activation.

### Workspace and graph-model integration

- External non-root graph references now load/activate the canonical external
  document, then open the referenced definition in a transitional child tab.

### Workspace and graph-model integration

- External graph nodes now open their root definition by activating or loading
  the canonical `.nodecraft` document tab. Non-root external definitions remain
  future view-path work.

### Workspace and graph-model integration

- Double-clicking a local graph node now opens its definition in a tab. Edits
  sync back through a `SetDefinition` session command, invalidate the parent
  graph-node hash and re-evaluate the parent when activated. This is a
  transitional child-session path; external graph opening and same-session
  views remain.

### Workspace and graph-model integration

- The definition flatten bridge now resolves external graph references through
  the canonical store, with missing-file/definition and cycle errors, and an
  external-definition flatten regression.

### Workspace and graph-model integration

- Graph-node content hashes now include the referenced local definition body,
  while excluding definition names and definition node positions. Definition
  geometry edits dirty the graph node and downstream consumers; renames/layout
  moves do not.

### Workspace and graph-model integration

- Document loading now converts inline group payloads before canonicalization,
  so a legacy inline graph opens as a graph-instance node with a local
  definition and flattens for evaluation. Linked parts keep their existing
  resolver until external graph references are executable.

### Workspace and graph-model integration

- Store canonicalization now covers nodes inside every definition for both load
  and save, and a migrated local graph definition is round-tripped through
  save/load and flattened in a regression test.

### Workspace and graph-model integration

- Migrated graph nodes now cache their definition name and input/output interface,
  render as Graph cards with the correct ports, and show a read-only graph-instance
  inspector. Local definitions flatten through the evaluation bridge.

### Workspace and graph-model integration

- Added a local-definition flatten bridge: graph-instance nodes are materialized
  into inline payloads, checked for definition cycles/missing ids, flattened and
  wired into both `FlatGraphCache` and `engine_bridge`. This lets migrated local
  definitions evaluate through the existing group path while external refs,
  graph-node UI and direct definition flattening remain pending.

### Workspace and graph-model integration

- Added the graph-instance reference model and a legacy conversion helper:
  inline `group` payloads become local `GraphDefinition`s with scoped ids, and
  `linkedpart` nodes become external graph references. Migration tests cover
  local, nested and external cases. The helper is not yet wired into load or app
  evaluation.

### Workspace and graph-model integration

- Documented the Phase 0 transition state and added an actionable checklist in
  `docs/workspace-transition-todo.md`.

### Workspace and graph-model integration

- Phase 0 started: `GraphDocument` now has a root `GraphDefinition` plus a
  `definitions` list, without bumping `schemaVersion` or adding a legacy loader.
  Existing root graph operations are forwarded to the root definition.
- Converted the repository `.nodecraft` files to the root-definition shape and
  updated store save/load plus the worker document protocol.
- Added a root/definitions serialization round-trip test.

### Graph nodes

- Renamed the visible inline-subpatch node from **Group** to **Graph**; the
  serialized node type stays `group` for file compatibility. Card titles now use
  the same display names as the inspector.
- Graph cards render every declared body input/output port, and the inspector
  exposes editable **Num inputs** / **Num outputs** counts. Resizing ports
  atomically updates one Graph Input and one Graph Output marker, their child
  wires and parent connections; card geometry invalidates immediately.
- Graph interface markers are single-sided (Graph Input has outputs only, Graph
  Output has inputs only) and cannot be deleted. Interface cables, including the
  default passthrough, can still be disconnected.
- Graph nodes have an inspector **Name** field; the graph-editing tab uses that
  name, with a node-id fallback for unnamed legacy graphs.
- With nothing selected inside a graph body, the inspector shows the owning graph
  node instead of an empty panel. Name edits route to the owning document; port
  counts are displayed read-only there to avoid desynchronising the body session.
- Inline graphs are now expanded before evaluation in the app flatten cache, and
  a graph card has a synthetic combined preview target instead of failing as an
  unimplemented `group` node.

### Workspace and graph-model integration

- Added the workspace integration plan: one document session per document, tabs
  and editor panes as views, flat owner-scoped graph definitions, tab-local undo,
  optional 3D viewports with per-view breakpoints, and one evaluation runtime.

### Tool dock

- The Create menu's Solid entry is now a split item: the primary row adds the
  default box solid, and the chevron opens a second flyout with the custom
  primitives (sphere, cylinder, cone, torus, pyramid, wedge and partial
  variants). Both rows keep the normal menu hover, click and keyboard styling.

### Feature edge editing

- Chamfer/fillet edge selection keeps the downstream breakpoint visible and
  previews the feature's input body as the transform-edit ghost; the ghost edges
  are pickable. This removes the old target swap that replaced a viewed Part
  with one upstream body.

### Known issues

- Fillet/chamfer ghost edit mode is **not 100% of the desired interaction yet**.
  The breakpoint + input ghost path works, but selection/overlay behaviour still
  needs another pass; it is not considered finished.
- A constructive chamfer around a feature can still remove more material than
  intended on some combined/looped shapes. This is a kernel result issue; the
  app has no local workaround.
- True G2 multi-edge fillet corners refuse until the matched G2 corner patch is
  ported into SolidCore.

## Unreleased — 2026-11-07

### SolidCore integration

- Re-evaluate now clears the tab's flattened-graph and preview caches, resets
  kernel memoisation, and bypasses every cache read so a coalesced or fresh
  cache cannot answer any node during the rebuild.
- Home/ISO now centres the shared world origin (0,0,0) in the unobstructed 3D
  viewport and uses that same origin as the orbit pivot; the construction grid
  consumes the same constant, so the grid centre and view centre cannot
  diverge. Trackpad
  navigation latches orbit or pan at gesture start, so releasing Shift during a
  two-finger orbit no longer turns the remaining momentum into a pivot-moving
  pan.
- STEP import shows the detected application protocol and `FILE_SCHEMA` in the
  importer inspector, with an explicit AP203/AP214/AP242 support note.
- Removed the last display-tessellation compatibility wrappers; Boolean display
  meshes now call `nodecraft-layers`/`solidcore-tess` directly.
- Circular fillets now expose SolidCore's swept corner turn: two rounds at a
  sharp side edge can meet as mitred (default) or swept sections.
- Removed the old `solidcore-stepio` STEP façade and the local B-rep mesher.
  STEP import now uses the official `step-import`/`step-brep` reader and all
  display tessellation goes through `nodecraft-layers` / `solidcore-tess`.
  `nodecraft-mesh` now keeps only STEP import, neutral mesh utilities and
  thin compatibility wrappers.
- Extracted the SolidCore kernel fix profile into a dependency-light crate so
  primitive, STEP and Boolean construction use the same `UpstreamFixes` profile
  without a dependency cycle through `nodecraft-solidcore`.
- Multi-edge G2 fillet corners use SolidCore's set-back corner session and are
  welded into the body, avoiding the folded Mitre corner. The patch currently
  uses circular sections; a true G2 edge-section patch still needs the
  SolidCraft `fill_matched`/`corner_g2` port.
- Equal-distance chamfer now falls back to the kernel blend session for curved
  faces, so a chamfer on a Boolean/torus join no longer stops at the planar
  preview limit. Fillet now has a **Tangent chain** inspector toggle and expands the selected
  edge through SolidCore's tangent-chain definition before rounding. This closes
  the seam-split torus/cylinder join for a normal join pick.
- Viewport edge references now carry the underlying BRep edge ordinal, and a
  multi-edge pick is resolved as a unique set with smooth/tangent edges excluded.
  A pick that only hits tangent edges now reports "reselect the join" instead of
  silently rounding an unrelated segment.
- The upstream fix profile now also configures the `solidcore-ops` session, so
  Boolean, shell, offset and blend calls run with the same fixes as direct
  kernel work.
- STEP import now tessellates imported bodies through
  `nodecraft-layers` / `solidcore-tess`, like every evaluated Part. The legacy
  STEP mesher remains only as a fallback for analytic edge curves
  `solidcore-convert` cannot yet record.
- Closed primitive display (`box`, `cylinder`, `cone`, `sphere`, `torus`,
  `pyramid`, `sphereNoPole`) now uses the same upstream tessellator.
- Partial sphere and partial torus now have exact SolidCore B-reps, so they
  participate in exact evaluation, Boolean and export as well as upstream
  display tessellation.
- The wedge now has an exact B-rep from `solidcore-ops` planar prism extrusion;
  every primitive display path uses `solidcore-tess`.
- The STEP legacy fallback is limited to the record layer's unrepresentable
  analytic-curve refusal; other upstream tessellation errors are not masked.
- Extrude/prism and loft/sweep volume parity tests now cover the upstream ops
  paths. Display evaluation cancels cooperatively between nodes when a newer
  edit arrives.
- Exact fillet, Boolean, union, shell and offset results are validated through
  `solidcore-ops::check`; invalid results report the upstream `CheckReport`
  text.
- A selected fillet edge refused by SolidCore as `Smooth` is no longer dropped
  to produce a partial round; the refusal reaches the node error.

### Ghost editing

- Transform ghosts come from a dedicated branch-only `PreviewWorker` with its
  own cache. The breakpoint evaluation no longer delays the preview.
- The ghost solid keeps its transform across mesh re-uploads.
- Ghost face and edge colours moved into the theme palette.

### Viewport

- Zebra analysis is a striped-skybox reflection with crease-aware smoothed
  normals and derivative antialiasing.
- Added `Ignore Look colours` for display without Look-authored face colours.
- G2/G3 fillet overlays now take the exact generated blend faces from the
  fillet result. The faces are identified by surface equality and mapped
  through the record face order, so the overlay shows only the fillet surface
  rather than the whole body. Their exact B-rep boundary edges are drawn too.
  The display-surface classifier remains the fallback when the exact result
  has no generated faces.
- G2/G3 fillet overlays skip the unsupported exact Boolean difference directly;
  failed exact comparisons are cached too. Selecting the test fillet graph went
  from a 1.27 s failed exact comparison to a ~0.1 s cached evaluation.
- The display popup has a fixed width, aligned columns and no slider jitter.

### Viewport

- Zero-length edge segments are skipped in GPU and 2D edge rendering, removing
  the vertex marker/“ball” that appeared at duplicated polyline points.

### App

- Release builds now keep line tables (`debug = "line-tables-only"`) and the app
  bundle no longer strips them, so release crash reports name file:line frames.
- Added a local crash dialog: a panic is caught at the frame boundary and shown
  with its message, location and backtrace. Nothing is sent or uploaded; the
  full text is also written to the system temp directory. The native alert shows
  only a short summary so it cannot grow past the display. Developers can test it
  with `NODECRAFT_DEV=1 nodecraft --crash-test`.
- Fixed a mesh crash when a degenerate triangle left a vertex without a normal
  cluster: the shading split now uses a fallback cluster and tolerates the
  degenerate face.
- Numeric fields accept SolidCraft-compatible expressions such as `40/2`,
  `2 in` and `90 deg`, including inspector fields, spline points and the sketch
  fillet radius. The parser and dimensional evaluator live in the new
  `nodecraft-expr` crate; one small adapter supplies the unit-kind-specific
  drag-value widget. While the field is focused it shows the source expression;
  Return or focus loss switches to the evaluated number, and clicking again
  restores the expression. Dragging or using the step buttons clears it. The
  click is detected from the field's previous rect so DragValue cannot seed its
  editor with the formatted number first. Expressions are saved in the node's
  JSON at the node level as `expressions`, so the source survives save/reload.
  All tracked test documents are now `schemaVersion: 1`. The legacy
  `.solidnode` and `values._expressions` compatibility paths were removed.
  Invalid expressions do not change the stored number (no NaN/0 is committed);
  SolidCraft stores the source and blocks apply, while this numeric-value adapter
  discards invalid text on focus loss. The expression cache lives in an input-only thread-local
  map rather than egui's global data map, so it cannot collide with DragValue's
  edit state; stale entries are invalidated without touching egui data.
- Fixed the theme colour picker so alpha is a byte-only edit (0.5 is 50%),
  derives HSV from opaque RGB so low alpha cannot saturate the hue, preserves
  RGB in the hex field, and does not tint the round picker. Cable/outline alpha
  is shared between normal and selected states so selection no longer pins the
  cable opaque. The
  luminosity control now changes the wheel's own brightness. The CPU cable
  fallback now uses the themed cable/outline colours, alpha and widths.
- Removed the retired Swift/WKWebView front end, worker queue, Swift test and
  legacy build script. The Rust app and `scripts/build-app.sh` are the single
  macOS build path.
- The tab-strip "+" stays anchored to the rightmost tab slot while a tab is
  dragged; the drag ghost no longer expands the strip.
- Dragged document tabs stay flat in the tab strip instead of lifting upward;
  the drop shadow remains for separation.
- Native macOS window movement is disabled while the pointer is over the
  document tab strip or a tab drag is active, so tab reordering no longer
  moves the window.
- The settings bar uses only the native macOS close light (minimize/zoom hidden)
  via the same per-OS window-chrome entry points; Windows/Linux keep the custom
  close control.
- macOS native traffic lights are centered vertically in the 50 px app bar,
  spaced 23 px apart with a 16 px left inset.
- macOS now uses the native traffic lights over the custom app bar instead of
  flat custom circles. The window-chrome helper has per-OS modules with the
  same main-window entry points; Windows and Linux keep native chrome.
- Slider rails have circular end caps as well as round handles; the colour picker uses the
  same compact slider sizing inside its menu-styled popup.
- Extracted the HSV colour wheel and luminosity/opacity controls into the
  `nodecraft-color-picker` crate; the app keeps the popup, hex field and theme
  integration.
- Extracted cross-platform title-bar button, drag, resize and title helpers;
  split the settings window shell, title bar and navigation out of the main
  settings function. The settings custom bar now works on Windows and Linux as
  well as macOS, with the same single close control and title.
- The settings window now uses the app's custom macOS title bar and the same
  16 px rounded, transparent window corners as the main window, with a single
  close control and the unchanged “NodeCraft Settings” title.
- The colour picker dismisses on any mouse press or scroll outside its popup,
  while inside clicks/drags remain inside the picker. A transparent middle
  layer consumes outside pointer input so settings controls do not hover or
  receive the dismissing click.
- The colour wheel selector keeps moving along the rim when a drag leaves the
  wheel; the pointer distance is clamped to the radius. Canvas wire/lasso and
  trackpad gestures are suppressed while the pointer is over a foreground
  picker or popup.
- The colour picker uses luminosity and opacity icons beside its sliders,
  with the wheel above and the two slider rows (icon, track, value) beneath it.
- Cable edge anti-aliasing is now pixel-sized instead of proportional to cable
  width, so thick zoomed-in cables stay sharp. The colour wheel updates its
  luminosity continuously instead of through a 64-step texture cache.
- Document saves use compact pretty JSON: short arrays and small objects stay
  inline while node and large collection structures remain expanded.
- Node evaluation errors now include a copy icon in their floating header.
- Node evaluation errors now float at the bottom of the 3D viewer instead of
  occupying the top of the inspector. The error still remains in the status
  bar for copy/context.
- The application crate now has `#![forbid(unsafe_code)]`. macOS Objective-C
  interop moved to the small `nodecraft-macos` boundary crate, which exposes
  only safe functions; document-open handoff and window chrome remain there.
- Renamed the desktop application package and binary from `nodecraft-egui` to
  `nodecraft`; the macOS bundle is now `NodeCraft.app` with the `NodeCraft`
  executable.
- Registered `.nodecraft` documents with macOS. Finder opens
  them as app tabs, including documents opened while the app is already
  running.
- The inspector disclosure chevron was replaced by a themed, centred reset
  control.


## Unreleased — 2026-11-05

### Build

- Updated SolidCore to `a39659e3` (current `main`).

### Graph

- Node errors now stay visible across evaluations. A selected node outside the
  viewed target is evaluated as a validation target for its error glow. Failed
  nodes no longer cache their input-bypass mesh as a successful result, and the
  failure cache is checked before the mesh cache, so a node cannot silently
  return to "no error" on the next evaluation.

- Node descriptors gain a `preview_input` capability. Fillet and Chamfer declare
  `shape`: selecting either shows the viewed breakpoint's final body plus the
  selected feature's difference from its input, without changing the breakpoint.
  The exact removed/added difference is used when available. If the exact
  surface intersection is unavailable (some G2 curved fillets), the engine falls
  back to a localized display-surface difference instead of ghosting the whole
  input body. Expanding to another feature is a descriptor entry, not a UI
  special case.

- Closed circular edge references resolve from the saved edge centroid as well
  as from points on the curve. Older documents stored the circle centre, which
  previously resolved to the wrong ordinal or a smooth seam; new edge picks also
  store a point on the edge.

- Changed nodes are sent to the engine as dirty. Every content-addressed cache
  path honours that state: worker mesh/failure, exact B-rep, Part, Part display,
  Part-only, primitive B-rep, Boolean operand, warm-prefetch and STEP import.
  Dragging an inspector value can no longer reuse the previous mesh and report
  a 0.0 ms cache hit. The body-mesh cache remains pointer-identity based and
  verifies `Arc::ptr_eq`, so it is not a content-cache bypass.

- Inspector value edits never move the viewed breakpoint. The edited node is
  marked dirty and the current viewed target is re-evaluated; an unrelated edit
  leaves the breakpoint exactly where the user put it.

- Evaluation identity now includes a session generation, bumped whenever the
  active session is installed. Replacing a project cannot reuse the previous
  graph's result even when its id, revision and target match.

- A zero extrude amount is rejected with "extrude amount must be greater than
  zero (got 0)" instead of a generic non-finite-distance message.

- Add a cylinder regression for a whole-circle sketch profile, covering its
  volume, centre, radius, height and side-wall radius.

- A whole-circle sketch profile is now accepted by Extrude: a single closed
  periodic curve is a valid profile and the extruder already knows how to split
  it into halves when needed.

- Add a confirmed Clear Sketch action that removes geometry and constraints
  while keeping the sketch display options.

- Selecting a sketch node outlines its curves and points in the viewport without
  opening the editor or capturing input.

- Centralise pointer-region checks in a shared `HoverMap` used by the viewport
  and sketch editor.
- Delete / Backspace in sketch editing removes the selected sketch points or
  curves instead of deleting the sketch node.

- The line-type picker is one connected squircle segmented control with white
  selected icons.
- Add the sketch palette to the inspector while editing: degrees-of-freedom
  status, line type (normal, construction, centerline, infinite), constraints /
  dimensions / points / snap toggles, construction mode, and a More section for
  projected and construction geometry, slice and 3D sketch. The 3D viewport
  keeps its own grid and scene; the sketch editor no longer draws a second XY
  floor.

- Sketches now enter and leave editing from the inspector's Edit sketch /
  Finish sketch button instead of opening automatically when the node is
  selected.
- The Select tool drags points and whole curves; fixed points stay put and the
  constraint solver runs during the drag.

- Upgrade the sketch model/solver to SolidCraft `d7cad1005` while keeping the
  NodeCraft egui sketch UI.
- Wire the constraint dock to `solidcraft-sketch`: horizontal, vertical,
  equal, parallel, perpendicular, tangent, concentric, collinear, length,
  radius and fix constraints apply to the current curve/point selection.

- Fillet now skips a smooth edge inside a mixed join selection and rounds the
  remaining blendable edges instead of failing the whole feature.

### Viewport

- Selected and hovered edges draw only their stroke; the per-vertex dots are
  gone, so dense curved edge selections stay legible.

- Add an opt-in GPU / frame diagnostics overlay (Settings > Diagnostics): frame
  time, adapter/backend, object triangles, GPU chunks and uploaded object memory.
  It stacks above the existing camera debug overlay; both background boxes are
  more transparent and remain non-interactive.
- The bottom-left evaluation log now splits total time into kernel and display
  meshing time.
- Uniform-colour display meshes keep their indexed positions/indices through the
  GPU object renderer instead of expanding every triangle into three vertices;
  the 10k forest viewport preparation drops from ~714 ms to ~367 ms.
- The frosted backdrop keeps at most 120k PBR triangles, since it is blurred
  heavily; that removes another ~200 ms from the 10k forest viewport preparation.

- Entering or leaving an inspector topology selection keeps the current camera;
  editing a fillet/chamfer/shell selection no longer refits the 3D viewport to
  the feature's input body.

### Node editor

- Viewer edge ids resolve by edge-list ordinal, matching the ids the display
  mesh writes.

### Node editor

- Fillet now retries with the stored edge centre when a Boolean has changed the
  edge ordinals behind a viewer edge id and SolidCore refuses the id-resolved
  edge as smooth.

### Node editor tool dock

- Reorganised the dock into Create, Modify, Boolean, Pattern, Organise, Animate
  and Input / Output sections.
- Grouped the Create menu into Sketch, Solid and Features, with labelled
  dividers.
- Added dedicated Iconify icons for every dock section and detailed modelling
  and primitive node types.
- Boolean nodes now show union, subtract A − B, subtract B − A or intersect
  icons according to their mode, using the existing app assets.
- Added a magnifying-glass tool search with CHOC-style wildcard and
  Levenshtein matching, themed result rows and keyboard result navigation.
- The search field now has a themed focus outline, centred text, a blinking
  caret, a five-minute stale-text timer and its own search shortcut.
- The dock is sized from its icon rows, has consistent square squircle hover
  and active highlights, a six-dot drag grip, saved vertical position and
  double-click recentre.
- Added keyboard navigation: Command-M toggles dock keyboard mode, the arrows
  move through search, categories and tools, and the shortcuts are editable in
  Settings → Keys.

### Node editor

- Shake a node or multi-node selection left and right to detach it from cables
  crossing the selection boundary; cables inside the selection stay connected.

### Performance

- Split the 3D object renderer into bounded vertex/index chunks so very large
  merged scenes (for example the 10,000-tree forest) stay under wgpu's per-buffer
  limit instead of aborting the app.
- Stop allocating and serializing a per-face colour channel when a graph has
  no Look nodes. Large scatter graphs no longer pay an extra colour vector for
  every display face.

### Appearance

- Added a pass-through Look node that attaches a per-face colour attribute to
  the incoming Part. SolidCore carries the colour through modelling operations
  and Boolean merges, and the viewport renders it.

### Sketch editor

- Updated the SolidCraft stack to 0.40.0-rc.2 (`481687ff`), bringing the newer
  sketch text, dimension, profile and solver work into the typed sketch output.
- Updated the SolidCore kernel and its ops/convert/tess/STEP crates to the
  latest main revision while bridging the SolidCraft 0.40 record types at the
  sketch/engine boundary.
- Adapted the GPU frame setup to the newer SolidCraft view fields.

- Replaced the sketch tool palette with a vertical icon dock on the left edge of
  the 3D viewport, using the same row size, squircle highlight, spacing and
  hover treatment as the node editor dock.
- Added sketch-tool icons and a compact tool-options flyout for Fillet and Arc.
- Moved sketch constraints into a horizontal dock at the top of the visible 3D
  viewport, centred between the node editor and the window edge.
- Extracted the shared dock metrics, squircle highlights, icon buttons and
  background chrome into the reusable `nodecraft-dock` crate, so both docks
  render through one path.
- Added a DockOrientation flag so the same generator can lay out vertical or
  horizontal docks.

### Other GUI and application changes

- Renamed the Primitive node type to Solid, keeping `primitive` as a document
  compatibility alias.
- Added the Keys settings tab and persisted tool-dock keyboard bindings.
- Replaced the legacy FluidCAD-derived sketch icon paths in the web viewer and
  removed the FluidCAD notice.
- Recorded CHOC in the third-party notices and documented the Rust ports in
  `nodecraft-search`.

## Earlier releases

See the public repository history for earlier snapshots.
