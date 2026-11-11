# TODO

Priority order is approximate.

## Current state (2026-11-08)

- SolidCore is vendored in SolidCraft at `c314b6d26ca756184ef2bdd1742f5056854d3d07`.
- Every SolidCraft module, including the sketch stack, is pinned to
  `c314b6d26ca756184ef2bdd1742f5056854d3d07`.
- Evaluated Part, comparison and STEP import display meshes use
  `nodecraft-layers` / `solidcore-tess`. STEP keeps a legacy fallback only for
  analytic edge curves that `solidcore-convert` cannot yet record.
- The app evaluates in two phases: exact result first, then display against the
  warmed Part cache. A separate branch-only `PreviewWorker` supplies transform
  ghost meshes without waiting for the breakpoint pass.
- Display tessellation retains each body's `TessPlan`; the viewport evaluates
  plans on the GPU through `TessGpu`/`GpuMesh` (background compile, chunked
  batches, per-plan placement/mirror/scale and per-vertex face colours) while
  keeping the CPU display mesh as the authority for edges, faces, ghosts,
  comparison and fallback.
- Feature edge selection uses the edit-input body's `TessPlan` edge polylines
  from `PreviewWorker`, with body-scoped ids, so picking does not depend on the
  displayed result or LOD.
- The `solidcore-ops` session receives the same upstream fix profile as
  NodeCraft kernels.
- Fillet refusals are no longer converted into partial successes. Exact fillet
  results are checked through `solidcore-ops::check`; invalid results report the
  upstream `CheckReport` text.
- A torus/cylinder join now rounds continuously from one picked edge through the
  fillet tangent chain. SolidCore #2826 remains open for callers that pass a
  bare edge without the chain.
- All primitives, including wedge, partial sphere and partial torus, now have
  exact B-reps and use `solidcore-tess`; the local B-rep mesher and the
  `solidcore-stepio` façade have been removed.
- The public snapshot is generated from committed `HEAD`; republish after
  integration work with `./scripts/publish-public.sh`.
- Latest SolidCore pin targeted gates: app 134, engine 125 (+1 ignored), session
  59 and layers pin tests pass. The last full workspace baseline before the pin
  bump was 879 passed / 1 known `nodecraft-mesh` sphere seam failure / 11
  ignored.

## SolidCraft reference implementations

When implementing the GPU/LOD/picking work, read SolidCraft at the pinned revision
first. Port the structure, not the app-specific code.

- `crates/ui-egui/src/display_lod.rs` — view-dependent refinement: screen-space
  error, hysteresis, still-camera delay, per-body refined `TessPlan` + CPU mesh,
  off-thread planning, triangle budget and cache.
- `crates/doc/src/display_quality.rs` — `DisplayQuality`, chord/angle refinement,
  `MAX_REFINE`, `refinement_for` and hysteresis.
- `crates/viewport-gpu/src/view.rs` — `Batches`/resident `GpuMesh` cache keyed by
  plan identity, plan upload and GPU evaluation.
- `crates/viewport-gpu/src/lib.rs` — `TessGpu`, `GpuMesh`, `prepare_many`,
  `evaluate`, `record`.
- `crates/ui-egui/src/viewport.rs` — how a viewport chooses plans, builds
  scenes and draws refined bodies.
- `crates/ui-egui/src/view_pick.rs` — screen/ray picking, edge pixel tolerance
  and exact body/face/point resolution.
- `crates/engine/src/cmd/edit.rs` / `measure_sel.rs` — resolving picks to exact
  `Sel::Face`/`Sel::Edge` indices and points.
- `crates/kernel/src/continuity.rs` — `pick_edges` for exact edge handling after
  a hit.

Caveats: upstream LOD **refines up from the chosen quality, it does not start
coarse**; upstream picking is mesh-candidate-first even when the final selection
is exact. NodeCraft currently receives one merged protocol mesh, while SolidCraft
works per `ModelBody` with plans, so the body-level plan map is the main adapter
difference.

## Display tessellation density (in progress)

- [x] Add a coarse display tier (`preview_detail`) selected from the camera.
      Forest fitted view: ~3.24M -> ~2.31M faces, mesh ~1.15 s -> ~0.94 s.
- [ ] Refine only visible bodies/screen regions with per-body screen-space error
      instead of one target-wide quality tier.
- [ ] Tune the tier thresholds and verify zoom transitions do not re-evaluate
      unnecessarily.

The GPU plan path does **not** reduce tessellation density or plan-generation
time. It only moves `TessPlan::evaluate` to the GPU. `gigatest-forest-5000` still
plans/evaluates a full-quality mesh:

```text
faces    3,195,208
vertices 1,735,641
mesh     ~1.1 s
```

The LOD simplifier runs after this full mesh exists, so it improves rendered
frame cost but not meshing cost. A chord/angle coarse-first experiment saved
only ~1% of triangles (~3,157,928 faces) and added a refinement pass.

Plan:

- [x] Collect `PlanStats` per body and expose lenient/over-tolerance/capped/
      unsampled/averaged-pole diagnostics through the engine, app status and
      `/state`. Per-node-type aggregation remains open.
- [ ] Identify which bodies produce most of the 3.2M triangles and whether their
      plan honors the requested chord/angle.
- [ ] Add a real display-quality tier (chord + facet angle) to the engine
      request and include it in the plan/mesh cache key.
- [ ] Choose a coarse plan for far/large views and refine view-dependently on
      zoom, rather than simplifying after a full mesh.
- [ ] Keep exact edge/picking/ghost/fallback data on the full B-rep side.
- [ ] Add a triangle-budget regression for the forest so density cannot silently
      regress.

## SolidCore integration opportunities

See [SolidCore/SolidCraft integration opportunities](../docs/solidcore-integration-opportunities.md).
Prioritized: `PlanStats` diagnostics, `solidcore_tess::preview` coarse-first
display, per-body plan profiling, continuity-backed edge chains, typed
boolean/check failures, then sweep stations.

## Engine/display boundary refactor (in progress)

The engine currently owns exact evaluation, Part/BRep caches, the tessellation
pool, `TessPlan` creation, CPU `Mesh` caches and the protocol mesh response.
That is too many layers. Plans are display artifacts, not graph-evaluation
results.

Target boundary:

```text
nodecraft-engine        exact graph evaluation -> Part/BRep handles
nodecraft-display       BRep + quality -> plan cache / PlanStats
viewport                plan -> GpuMesh cache, CPU fallback/picking data
```

Plan caching:

- cache the **plan**, not the mesh;
- `plan.evaluate()` is cheap and the GPU path does not need a CPU mesh;
- derive CPU meshes/edges/face topology on demand in the display/view layer;
- remove the `(BRep, Mesh, Plan, Stats)` body cache from `EngineCache`;
- keep exact BRep/Part caches in the engine.

Migration checklist:

- [x] Add `EngineOutput.exact_parts` (B-rep + placement handles) from the exact
      pass; regression proves the exact pass returns no meshes.
- [x] Add `nodecraft-display` with quality selection, plan-only cache, body
      tessellation and derived plan evaluation; move the engine's plan cache onto
      it. Still needs the workers/cache ownership moved fully out of the engine.
- [x] `EngineOutput.exact_parts` exposes exact display-side Part/BRep handles;
      `nodecraft-display::mesh_part` meshes them without a graph. Switching the
      app display path off the engine is still open.
- [x] Store body plans only in `EngineCache`; derive CPU meshes from cached
      plans. Move the plan cache itself out of the engine next.
- [x] Display types live in `nodecraft-display`; the app's normal target display
      path now meshes exact handles through it. Comparison/ghost/edit-preview and
      the engine response path remain transitional.
- [ ] Keep exact B-rep picking independent of plan/mesh caches.
- [x] Remove `EngineOutput.plans`. Normal/ghost/preview/comparison display plans
      come from `nodecraft-display`; only `Response.meshes` remains for exporter
      runs.
- [x] ExactPart can realize world B-reps for exporters without engine meshing;
      `export_step` writes/validates exact STEP with metadata.
- [x] Add display/export quality separation: `ExportOptions`/`export_mesh` with
      export tolerances, closed-topology checking and a distinct plan-cache key.
- [x] Removed `Response.meshes`; worker/session JSON meshes come from the display
      service and `evaluate()` is test-only.
- [x] Removed the plan cache from `EngineCache`; worker/session own display plan
      caches. Engine display methods are test-only and exact mirror Booleans use
      exact B-reps.
- [x] The Rust worker uses the exact engine + `nodecraft-display::protocol_meshes`;
      `Response.meshes` is now display-service compatibility output for the
      worker/CLI/STEP scripts, not engine display output. The app exporter now
      uses `export_step`; `step_report_mesh` now uses the display service too.
      Only engine tests still use `evaluate()` for mesh assertions.

## GPU plan / B-rep picking follow-ups

Implemented (2026-11-10):

- [x] Capture and carry `TessPlan` per body through mesh asset, body cache,
  `EngineOutput` and app evaluation.
- [x] Chunked `TessGpu`/`GpuMesh` rendering through the NodeCraft PBR shader.
- [x] Background `TessLater` compile with CPU display-mesh fallback.
- [x] Per-plan placement/mirror/scale instance matrices.
- [x] Per-vertex face colours from `GpuMesh::vertex_face` and B-rep face colours.
- [x] Input-body plan edges for feature edge picking.

Still worth doing:

- [ ] Analytic edge-curve intersection. Current picking uses the B-rep plan's
  edge polylines; they are exact body topology but sampled geometry. This is an
  accuracy enhancement, not a dependency of selection.
- [x] Plan-based face picking: face mode now derives triangle→face ids from the
  display target's `TessPlan` faces (`face:N` / `body:B:face:N`) before
  raycasting the CPU display mesh.
- [ ] A hardware regression that asserts `GpuMesh` output matches
  `TessPlan::evaluate` on a simple body.
- [ ] Measure plan generation vs GPU evaluation separately on the forest.

## Development order (current)

Work in this order; do not start a later item before its dependency is settled.

1. [ ] **Per-body tessellation profiling** — instrument `PlanStats`, plan
       generation, `TessPlan::evaluate`, display preparation and GPU upload
       separately. Determine whether the forest cost is CPU planning, CPU
       evaluation, mesh merge or upload.
2. [x] **B-rep picking contract** — the preview worker returns the edit-input
       body's `TessPlan`; edge picking uses its B-rep edge polylines with
       body-scoped ids. Face picking uses the input body's exact face ids from
       its display mesh. Picking does not depend on display LOD.
3. [x] **GPU tessellation from `TessPlan`** — carry plans through the display
       path, use `TessGpu`/`GpuMesh`, cache by plan identity, and retain the CPU
       `TessPlan::evaluate` + protocol-mesh fallback.
4. [ ] **Coarse-first / view-dependent display** — use the profiling data to pick a
       real coarse plan or preview, define quality tiers, refine when the camera
       settles, and keep exact picking/edges. Do not retry chord/angle-only tuning
       without plan statistics.
5. [ ] **Progressive GPU upload** — upload a bounded number of vertex/index chunks
       per frame and draw chunks as they become resident.
6. [ ] **Re-evaluate/cache semantics** — settle bypass versus free, phase
       counters, cold-versus-warm reporting, nested-view targets and progress.
7. [ ] **Autosave and recovery** — dirty tracking, debounced atomic saves,
       conflicts, untitled recovery and cascade-deletion undo.
8. [ ] **Deferred graph/runtime work** — multiple panes/viewports, iterator,
       externalise/pack lifecycle, fold/accumulator.

Dependency: **1 → 2 → 3 → 4**. Building GPU tessellation or another LOD tier
before profiling and picking risks optimizing the wrong layer.

## Workspace integration (start here)

Transition checklist: [Workspace transition TODO](../docs/workspace-transition-todo.md).
Full sequence: [Workspace and graph-model integration plan](../docs/workspace-integration-plan.md).

- [x] **Phase 0 groundwork:** add `GraphDocument.root` + `GraphDefinition`
      list without a schemaVersion bump; convert current `.nodecraft` files;
      update store/worker save-load; root field access remains compatible.
- [ ] **Phase 0A:** define graph-instance values and migrate `group` payloads to
      local definitions.
- [ ] **Phase 0B:** migrate `linkedpart` to external definition references.
- [ ] **Phase 0C:** validate/hash/canonicalize definitions and round-trip a
      migrated document.
- [ ] **Phase 1:** definition-scoped APIs and one local/external flattener.
- [ ] **Phase 2:** workspace/view/tab/pane state; graph tabs as views; remove
      child-session sync.
- [ ] **Phase 3:** tab-owned undo journals and cross-tab conflict checks.
- [ ] **Phase 4+:** panes/3D, runtime ownership, iterator, linked lifecycle.
- Fold/accumulator execution is deferred for the current transition.

## Re-evaluate semantics (open questions)

The current button bypasses cache reads, prewarms the exact closure, reuses warmed
B-reps and runs a non-export display pass. This works but the contract, cache
reporting, progress and activity behaviour need a design pass. See
[Re-evaluate behaviour: open questions](../docs/re-evaluate-questions.md).

- [ ] Decide the user-facing contract (refresh vs cold rebuild vs debug action).
- [ ] Decide whether re-evaluate frees caches or only bypasses reads; split a
      separate clear-cache action if needed.
- [ ] Define exact/prewarm/display progress and cold-vs-reuse cache statistics.
- [ ] Define re-evaluate behaviour for nested/definition views and multiple tabs.
- [ ] Add phase timings and regression coverage.
- [ ] Revisit whether the button should remain user-visible.

## Document autosave (planned)

Autosave is not implemented: edits currently reach disk only through an explicit
Save/Save As. The definition-cascade cleanup also removes definitions from the
in-memory document immediately but only from the file on the next save.

- [ ] Track per-document dirty state from the saved content hash/revision, shared
      by all tabs/views over that document. Show a dirty marker per tab and
      document title.
- [ ] Debounced atomic autosave for documents with a path, using the existing
      temp-file + rename path. Skip while a gizmo/node drag, file dialog, export,
      or other blocking operation is active; flush on idle, focus loss, tab close
      and app quit.
- [ ] Flush the owning document when a definition view edits a definition, and
      when an external definition is edited through its canonical document.
- [ ] Before overwriting, compare the on-disk hash with the loaded hash. On a
      conflict, pause autosave and surface the same conflict flow as manual save.
- [ ] Pause autosave while evaluation/export writes are in flight; never
      autosave a partially published or partially restored session.
- [ ] Untitled documents: store a recovery draft in the app data directory rather
      than inventing a filename, and offer recovery on next launch.
- [ ] Setting for autosave enable/disable and debounce interval; default policy
      to be decided after recovery/conflict tests pass.
- [ ] Make definition cascade/GC safe under autosave: deletion becomes durable
      quickly, so undo must be able to restore the definitions and closed tabs
      before a later flush.
- [ ] Tests: dirty/save transitions, debounce coalescing, failed atomic write,
      external-edit conflict, quiescence during drag, tab switch, external
      definition document, untitled recovery and undo after autosave.
- [ ] Document the autosave/recovery lifecycle in the README and architecture
      notes once the behavior is settled.

## Contextual subgraph evaluation (planned)

Opening a graph definition tab currently evaluates that definition body in
isolation. A Graph Input marker has no external source there, so the body (and
its breakpoint) sees no parent geometry. This is not a renderer bug; it is the
missing evaluation-context binding.

- [x] Track the instance path for each graph view (`g1::g2`, with graph-node ids),
      alongside the existing `ViewKey`.
- [x] Evaluate the owning flat graph in context: use the parent instance path as
      the prefix, and map a body breakpoint/selection to the prefixed flat node or
      the graph-card combine for a Graph Output marker.
- [x] Map displayed target, node errors, activity and dirty nodes back to the body
      view; comparison/ghost targets use the prefixed selection.
- [x] Support nested paths and external definition documents; shared definitions
      use the path of the instance the view was opened from.
- [x] Map transform previews through the prefix. First-level Graph Input markers
      bind to the owning graph node's upstream source; deeper input markers and
      topology-picking sessions remain.
- [ ] Add end-to-end regressions for passthrough/transformed inputs, nested bodies
      and a definition used by two differently-fed instances.
- [ ] Remove the isolated-body fallback once preview/picking mapping is stable.
- [x] Re-evaluate bypasses cache lookups for the fresh run instead of freeing all
      flat/engine caches; content hashes already invalidate changed nested graphs.

## Coarse-first tessellation profiling (planned)

A coarse initial tolerance/angle pass on `gigatest-forest-5000` reduced faces by
only ~1% and added a refinement evaluation, so it was reverted. LOD after the full
mesh cannot reduce tessellation cost.

- [ ] Collect per-body `PlanStats` for the forest: faces/triangles by node type,
      by tree group vs STEP import.
- [ ] Check whether `MeshTolerance` angle/chord are actually reducing those
      bodies' plans; the 30°/60° experiment barely changed the total.
- [ ] Identify whether the dominant meshing cost is solidcore-tess or the STEP
      import/display path.
- [ ] Only then consider view-dependent plans, a true coarse preview, or GPU plan
      evaluation.

## GPU upload profiling (planned)

The display-preparation worker moves mesh conversion off the UI thread, but
`ObjectRenderer::prepare` still creates and uploads all pending vertex/index
buffers in one frame. Large forests can hitch on the first upload even when
evaluation is threaded.

- [ ] Time first-frame upload separately from evaluation/display preparation.
- [ ] Add a per-frame upload budget and progressive draw for large meshes.
- [ ] Consider uploading a coarse LOD first, then upgrading.
- [ ] Evaluate using `Device`/`Queue` clones from a worker while keeping draw
      state synchronized.
- [ ] Add a large-mesh regression that asserts the UI frame does not stall during
      buffer upload.

## Pure B-rep picking and GPU plan tessellation (planned)

Two connected follow-ups for the display/picking boundary.

### Pure B-rep edge/face picking

The display mesh should be allowed to be a coarse LOD or GPU-evaluated plan while
picking remains exact and independent of that mesh.

- [ ] Keep exact record/kernel edge curves and stable edge ids available for
      picking. Edge picking should use analytic curve/screen projection, not
      triangle adjacency or a sampled display polyline.
- [ ] Decide the UI boundary: send exact edge curves (plus body/face ids) to the
      UI, or run edge picking in the engine worker and return selection ids.
- [ ] Preserve exact face/provenance for face picking: surface-domain ray tests
      or a dedicated exact face-picking service; do not infer faces from LOD
      triangles.
- [ ] Verify selection remains correct at high zoom on curved edges and with the
      display at a coarse LOD.
- [ ] Keep hover/selection overlays display-only; returning the exact edge/face
      id is the authoritative operation.
- [ ] Add regressions for edge/face picking across LOD changes and body
      placements, including transformed/mirrored bodies.
- [x] Pick edges against the **selected feature node's input body**, not the
      viewport breakpoint/output mesh: the preview worker now returns that body's
      `TessPlan`, and the viewport derives pickable B-rep plan edges from it with
      body-scoped ids. Face picking still uses the display mesh.
- [ ] Handle multi-body inputs, placement/mirror transforms, suppressed bodies and
      nested graph/instance paths when resolving the edited node's input body.
- [x] Keep the displayed result/ghost separate from the pick-input body; the
      preview worker returns the input body's plan and edge selection uses it
      independently of the viewer breakpoint.

### GPU tessellation from `TessPlan`

`solidcraft-viewport-gpu` exposes `TessGpu`/`GpuMesh`: upload a `TessPlan`, evaluate
vertices/normals in a compute shader, and render the resulting buffers directly.

- [x] Carry `TessPlan` through the display request/cache instead of only the
      CPU-evaluated protocol mesh.
- [x] Use `TessGpu::prepare_many`/`record` and the shared `GpuMesh` path; retain
      the CPU `TessPlan::evaluate` + protocol-mesh fallback for headless, export
      and environments without GPU plan support.
- [x] Keep edge polylines, face ids/face groups and exact picking data on the
      CPU/B-rep side; GPU mesh vertex/index buffers are rendering-only.
- [ ] Cache `GpuMesh` by `TessPlan` identity and discard plans/buffers on graph
      or viewport changes.
- [ ] Measure whether the forest cost is plan generation/triangulation (CPU) or
      plan evaluation (GPU candidate). GPU tessellation does not remove plan
      generation cost.
- [ ] Add plan/`PlanStats` instrumentation per body and a GpuMesh regression on
      a large document.

### Integration boundary

Keep NodeCraft's PBR object shader, transparent/ghost layers, comparison meshes,
edge/face overlays and picking. Port only the tessellation architecture:

- engine/display returns a `TessPlan` per display body plus exact edge data,
  instead of only one merged protocol mesh;
- a new plan renderer owns `TessGpu`, `GpuMesh` residency and the background
  `TessReady`/`TessLater` compile, with CPU `TessPlan::evaluate` fallback;
- drawing the main body from `GpuMesh` must preserve NodeCraft's per-face colours:
  `GpuMesh::vertex_face` can drive a compact colour buffer/table instead of
  rebuilding CPU `ObjectVertex` positions;
- ghosts, comparison layers and edge/face overlays remain on the existing
  NodeCraft renderer/full data;
- NodeCraft picking stays independent of the plan mesh and authoritative on the
  B-rep/full display data.

Do not replace the PBR/ghost/layer renderer wholesale with SolidCraft's viewport;
the portable part is plan generation, GPU plan evaluation, `GpuMesh` buffers and
their cache lifecycle.

## GPU tessellation plan path (planned)

NodeCraft currently plans with `solidcore_tess::plan` and evaluates the plan on
the CPU in `nodecraft-layers`, then uploads the protocol mesh. The upstream
viewport GPU crates can evaluate the same plan in a compute shader.

- [ ] Carry `TessPlan` through the display request instead of only the
      CPU-evaluated mesh.
- [ ] Upload/evaluate the plan through `solidcraft-viewport-gpu`'s compute path
      while retaining the CPU fallback and protocol-mesh path.
- [ ] Preserve edge, face-group and picking provenance in the GPU path.
- [ ] Measure CPU evaluation versus GPU plan evaluation on large Parts and compare
      first-frame latency and memory.

## Graph definitions, instances and flattening (planned)

See [Graph definitions, instances and flattening](../docs/graph-definition-instance-plan.md).

- [ ] Formalize `GraphDefinition` and `GraphInstance` in the document model:
      typed interface ports, parameters, definition reference and output mapping.
- [ ] Unify `group` and `linkedpart` into the same graph-instance semantics.
- [ ] Add a canonical `DocumentStore` keyed by definition path/id so opening the
      same `.nodecraft` from multiple graphs edits one session.
- [ ] Replace special-case linked expansion with one recursive flattener:
      instance-prefixed ids, cycle/missing-file errors, interface binding.
- [ ] Make the flattener cache key include the root + all reachable definition
      revisions/content hashes.
- [ ] Move cache ownership to one `EvaluationRuntime` with logical per-document
      partitions; keep tabs/source/undo outside the runtime.
- [ ] Add graph-instance inspector actions: open definition, expose parameter,
      inline/externalize, replace/localize.
- [ ] Add breadcrumb navigation and definition-open shared by groups and linked
      files.
- [ ] Regression tests: inline vs external graph equivalence, definition edit
      invalidation, parameter hashes, cycle/missing files, canonical edit dedupe,
      and placement-only instance changes.
- [ ] Implement iterator graph nodes:
      `Input`/`Output`/`Iteration Index` interface nodes, instance iteration
      count, ordered sequence outputs, outside aggregation, and concurrent
      prefixed flattening. Plan: [Iterator nodes](../docs/node/iterator.md).

## Feature edge ghost editing (in progress)

- [ ] **Fillet/chamfer ghost edit mode is not 100% how we want it yet.** The
      breakpoint + transform-edit input ghost path works and the ghost edges are
      pickable, but the selection/overlay behaviour still needs another pass.
      Do not treat this as finished.
- [ ] A constructive chamfer around a feature can remove most of the body on
      some combined/looped shapes. The session returns a technically valid
      result; this needs an upstream result guard or fix, not an app workaround.
- [ ] True G2 multi-edge fillet corners are refused until the matched G2 corner
      patch port in `agents/port-g2-corner-fill.md` lands.

## SolidCore a445bda pin bump (completed)

Updated the workspace pin from `0249e8e` to `a445bda`; targeted app, engine,
layers and session tests pass. This predates the vendored SolidCraft layout; the
workspace now has one record revision from the SolidCraft tree.

Upstream history from `3ed61947` to `f8685cdc`: +491 commits to `5458b142`,
then +7 blend end-section/smooth-vertex fixes, then +6 blend/offset/surface
fixes. The pin is applied.

- [x] Update every workspace SolidCore pin together; member manifests stay on
      `workspace = true`.
- [x] Run app, engine, layers, solidcore and mesh gates.
- [x] Carry the kernel fix profile into the `solidcore-ops` session.
- [x] Update the SolidCraft sketch stack to `871014f9`.
- [x] Pull the next 7 upstream commits (blend end-section and smooth-vertex
      run-on fixes) and validate Boolean/union/shell/offset results through
      `solidcore-ops::check`.
- [ ] Revisit Part-target Boolean/shell migration when upstream exposes public
      provenance attach/read or otherwise preserves application attributes and
      edge identity.

## V1 document format: node-level expressions

The old compatibility shape kept source expressions under a reserved key inside
the node's free-form `values` object:

```json
{
  "type": "chamfer",
  "values": {
    "length": 2,
    "distance2": 0.8,
    "_expressions": {
      "length": "2/1"
    }
  }
}
```

The converter has been removed. Existing tracked test documents were rewritten
to the node-level format, and new documents use `schemaVersion: 1`.

Problems with the old shape:

- `_expressions` collides with the namespace of node value keys.
- Expression metadata is sent to the engine inside `values`, even though the
  engine only needs the evaluated number.
- Node content hashes and cache invalidation currently include presentation
  source text.
- The format is not self-describing or schema-versioned.
- Nested values use ad-hoc dotted keys such as `points.0.x`.
- Invalid source has to be silently discarded or repaired because there is no
  first-class place to preserve it.

Target V1 shape:

```json
{
  "schemaVersion": 1,
  "nodes": [
    {
      "id": "n76",
      "type": "chamfer",
      "values": {
        "length": 2,
        "distance2": 0.8
      },
      "expressions": {
        "length": "2/1"
      }
    }
  ]
}
```

Rules for the migration:

- [x] Add `expressions: BTreeMap<String, String>` as a node-level sibling of
      `values`; omit it when empty.
- [x] Define canonical expression paths for nested fields (`points.0.x`,
      `points.0.y`); a structured representation remains an option before V1.
- [x] Keep expressions out of the worker `Node.values` conversion; the engine
      continues to receive finite numbers only.
- [x] Separate geometry/content hashes from document-source hashes so editing
      only an expression does not invalidate geometry caches unnecessarily.
- [ ] Preserve invalid source and report it in the UI; do not silently drop or
      repair user-edited expressions on load.
- [x] Remove the legacy converter and rewrite the existing tracked documents to
      the node-level format.
- [x] Add round-trip tests for expressions and node hashes.
- [ ] Decide whether expression source belongs in linked-part content hashes or
      only in the local project revision.
- [x] Keep `schemaVersion: 1`; no version churn before V1 freezes the file
      format.

### SolidCraft document comparison (options for V1)

SolidCraft's design JSON is a parametric feature document. NodeCraft's is a
node graph, so feature/assembly fields are not directly transferable, but its
top-level document fields are a useful checklist.

SolidCraft top-level fields include:

- `format` (`solidcraft/N`) plus `format_pending`, with an incremental
  migration ladder;
- `name`, `units` and free-form `metadata`;
- `params` (`name`, `expr`, `unit`, `comment`, `model`);
- `features`, plus optional `marker` and `next_id`;
- optional view/document state such as `named_views`, `visibility`, `render`,
  appearances, materials, assemblies and configurations.

NodeCraft currently has `schemaVersion`, `id`, `nodes`, `connections` and
`interface`. Candidate additions, in priority order:

- [ ] Add optional document `name`.
- [ ] Add optional `units` (default `"mm"`) for display and project metadata.
- [ ] Add optional free-form `metadata` for author/project/plugin data.
- [ ] Add document-level `parameters` so named formulas can be reused by numbox
      expressions. Compare with parameter nodes before choosing the model.
- [ ] Add optional node `name`; today the inspector title is derived from type.
- [ ] Decide whether camera/visibility state belongs in the document or only in
      tab/app settings. SolidCraft saves named views and browser visibility.
- [ ] Decide between the current dual `values` + `expressions` storage, a
      source-only expression form, and explicit `{value, expr}` pairs.
- [ ] Decide whether to keep numeric `schemaVersion` or adopt a
      `"nodecraft/N"` format string before V1 freezes the format.

Do not copy SolidCraft's feature-specific model wholesale:

- `features`, `param_names`, `body_names`, `edge_names`/`face_names`;
- `components`, `occurrences`, `assembly`, joints;
- sheet metal, plastic, configurations, user materials;
- timeline `marker` rollback.

Preferred near-term order:

1. `name`, `units`, `metadata`;
2. document parameters and expression lookup;
3. optional node names;
4. view/visibility decision;
5. expression-storage and versioning decision.

## G2 corner fill

- [ ] Port the matched G2 corner fill from SolidCraft into SolidCore and expose
      it as a corner mode in `solidcore_ops`. Full plan:
      [port-g2-corner-fill.md](port-g2-corner-fill.md).

## Exact measure UI

- [ ] Show exact kernel measures in the inspector/properties: edge length,
      face area, face centroid and planar face normal via the
      `solidcore_ops::measure` APIs. Edge selection already uses the exact
      kernel edge length.

## Remaining after this pass

- [ ] Republish the public snapshot with the docs and integration work.
- [ ] Make display work truly concurrent. The `EvalWorker` is still sequential
      after the exact phase; a newer edit now cancels display cooperatively
      between nodes/mesher bodies, but display still runs on the exact worker.
- [x] Remove the STEP legacy mesher fallback; STEP display uses solidcore-tess.
- [x] Multi-edge torus/cylinder join fillets run continuously from one picked
      edge via the fillet tangent-chain expansion. Keep the upstream #2826
      regression for callers that pass a bare edge.
- [ ] Manually smoke-test heavy-target edits, rapid edits, tab switches during
      display, failed-node glow, export/clear-cache and cache growth.
- [ ] Finish the Part-target Boolean and shell ops migration once upstream
      provenance/attribute support is available.
- [ ] Fillet/chamfer ops migration: keep the current direct-plus-ops path until
      a parity matrix covers G1/G2/G3, mixed edges, corners/setbacks,
      two-cylinder joins, chamfer angle/distance and failures.
- [x] Check `solidcore-ops::prism`/`sweep` against the extrude/loft workflows.
      Extrude/prism and loft volume parity tests pass; migration waits on a
      larger topology/attribute parity matrix.
- [ ] Keep dependency-crate warning cleanup as a separate low-priority item.

## Recently completed

- [x] Bump SolidCore to `3ed61947` across every workspace crate and re-run the
      layer, engine, worker, mesh, step and comparison gates.
- [x] Move comparison ghosts to record/ops Booleans and `solidcore-tess`.
- [x] Move Boolean operand/export chains and extrude operation Booleans to
      `solidcore-ops`, with a box-volume parity test.
- [x] Decouple selection from viewport editing: double-click enters a node edit
      target, selection alone never opens a gizmo, and the transform gizmo
      requires the selected node to be the explicit edit target.
- [x] Add branch-scoped transform previews: when the breakpoint merges the edit
      branch, fetch the transform input mesh, draw it over a faint breakpoint
      mesh, and move only that branch while dragging; the breakpoint mesh itself
      never moves.
- [x] Add `evaluate_exact` and wire the app worker into exact → display phases;
      publish exact errors/status before meshing, keep the old mesh visible,
      cancel stale display after the exact phase, keep exports single-pass and
      make progress display-owned.
- [x] Split `app.rs` from ~8,183 to ~4,527 lines into evaluation, settings,
      settings UI, theme editor, tool dock, tabs and document-ops modules.
- [x] Split the engine into `exact_eval.rs`, `display_eval.rs` plus cache,
      primitive, look, bypass, face-reference and comparison modules.
- [x] Route all engine display-only branches and comparison ghosts through
      `nodecraft-layers` / `solidcore-tess`; engine no longer calls
      `brep_display_mesh`.
- [x] Move Boolean operand/export chains and extrude operation booleans to
      record/ops Booleans, with a box-volume parity test. The Part-target Boolean
      node path is held back by ops attribute/topology parity.
- [x] Add failure-cache and exact-comparison caching, `CacheKey`, approximate
      ghost labelling and selected-node inspector errors.
- [x] Centralize inspector metadata, topology selection and primitive field
      metadata in `nodecraft-node-contract`.
- [x] Add SolidCore pin and app/engine layer-boundary tests, run layers tests in
      CI, and centralize workspace pins.
- [x] Refresh the SolidCore alignment/sync docs for the live record/convert/ops/
      tess bridge.

- [x] Prepare the upstream layer adoption locally: delete the unused
      `crates/solidcore-mesh` port, add the pinned `nodecraft-layers` crate with a
      record -> `solidcore-tess` adapter and a copied record fixture test, and an
      ops primitive/smb smoke test. Historical note: the kernel B-rep <-> record
      bridge was missing at that point; it has since landed and is used by
      `nodecraft-layers`.
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

`rust/apps/nodecraft/src/app.rs` was ~8,100 lines. It is now ~4,500
lines after extracting lifecycle/ownership units into `app/` modules. The
target was roughly 3,500–4,500 lines with the central draw order still in one
place; that target is met.

- [x] **Evaluation** → `app/evaluation.rs`: `EvalJob`, `TabEvalState`,
      `EvalWorker`, `evaluate`, `evaluate_with_cache_policy` and
      `poll_evaluation`. Evaluation-key construction remains with the key type
      in `app.rs` for now.
- [x] **Settings persistence** → `app/settings.rs`: `AppSettings`,
      `ActivitySettings`, platform/config paths, load/save, defaults and startup
      document selection.
- [x] **Settings and theme UI** → `app/settings_ui.rs` / `app/theme_editor.rs`:
      settings window, theme add/revert/edit buttons, hex editor, palette rows
      and colour wheel.
- [x] **Tool dock** → `app/tool_dock.rs`: `ToolSection`, `ToolGroup`,
      `ToolItem`, menu/search/focus state, layout helpers and `show_tool_dock`.
- [x] **Tabs and document lifecycle** → `app/tabs.rs`: `DocumentTab`,
      `TabDrag`, open/save/restore/activate/close, session installation,
      document tabs, linked-part opening and native save/open dialogs.
- [x] **Pure document operations** → `app/document_ops.rs`: clipboard/paste
      plan, delete/disconnect command builders, `target_closure`,
      `downstream_nodes`, `initial_viewer_target`, node position tracking,
      tab reorder, path/document-id helpers and document extension helpers.
- [x] **Central UI** last: `ui()` remains the single orchestration point and
      the central draw order stays in `app.rs`. Further panel extraction is
      optional follow-up, not required to meet the size target.

Acceptance:

- each extraction is a separate, behavior-preserving commit;
- no duplicated helpers or visibility-only copies;
- unit tests move with their module, and app-level tests stay in `app.rs`;
- `cargo test -p nodecraft` stays green after every step;
- `app.rs` ends up containing the app struct, shared state, central `ui()`
      draw order and high-level coordination only.

## Code quality review (2026-11-06)

Review carried out at private HEAD `7a29d14`, public `951251b`.
These are cross-cutting improvements, not individual feature requests.

### P0 — restore a trustworthy baseline

- [x] Free workspace disk and add a target-cleanup policy. `rust/target/debug`
      was removed (40 GB total target, 33 GB debug), freeing enough space to
      link. The policy is in [Agent Guide](README.md#build-hygiene): remove build
      artifacts only, never source or the nested public checkout.
- [x] Record the actual workspace baseline after cleanup. Fast tier:
      `cargo test --workspace --no-fail-fast`
      → 808 passed, 1 failed (known sphere seam), 11 ignored. Results are in
      [Current state](current-state.md).
- [x] Make public CI match its README: add `nodecraft-engine --lib` and
      `nodecraft-session` to CI. The engine library tests use only public
      fixtures; the long STEP-corpus tests remain private/slow.
- [x] Refresh documentation drift: SolidCore `a39659e3`, SolidCraft
      `d7cad100`, current test counts, explicit viewer breakpoint semantics,
      known shell failure and node-error discovery boundary.

### P1 — structural clean-up

- [x] Split `nodecraft-engine/src/lib.rs` (~7,200 lines) the way `app.rs` is
      split: exact evaluation, display evaluation, caches, comparison and error
      attribution behind explicit interfaces. Exact evaluation is now
      `exact_eval.rs`, display evaluation is `display_eval.rs`; cache,
      primitive, look, bypass, face-reference and comparison concerns already
      have their own modules. `lib.rs` is the coordinator.
- [x] Make node metadata authoritative in one place. Ports, defaults, selection
      capabilities, preview inputs and inspector fields now live in
      `nodecraft-node-contract`, including per-primitive fields. The custom
      stepimporter and exporter editors remain dedicated file-picker UI.
- [x] Formalise cache keys and entries. `CacheKey` names the shared
      node-id/content-hash key used by every content cache. Invariant tests cover
      mesh failure-bypass, dirty Part/B-rep, session identity and exact-comparison
      cache behaviour.
- [x] Define the boundary of node-error discovery. Selected nodes outside the
      viewed closure are validated; the inspector now shows the selected node's
      error text as well as its canvas glow. Unviewed, unselected branches are
      not evaluated, and global background validation remains out of scope.
- [x] Distinguish exact feature differences from approximate display-surface
      fallbacks in the comparison payload and UI. The payload carries
      `approximate`, the status bar labels approximate feature ghosts, and exact
      comparison results are cached by the two target content keys.
- [ ] Improve shell coverage/validation for curved fillets and top-face
      openings without reintroducing a silent fallback. Malformed and
      out-of-range opening references now fail explicitly, but the curved/
      top-face shell limitation is blocked on SolidCore shell topology and
      remains an explicit error. See the SolidCore pin-bump retest item below.
- [x] Remove warning/dead-code debt from the main targets. `nodecraft-engine`
      and the app source check warning-free. Dependency-crate warnings remain a
      separate clean-up item.

### P2 — longer-term quality

- [ ] Add a CI warning budget and eventually `-D warnings`; remove dead research
      modules or wire them in.
- [ ] Add benchmark/regression coverage for large graphs, exact meshing and
      display tessellation; keep performance changes measurable.
- [ ] Complete descriptor-driven dispatch so node types cannot drift between
      evaluation, inspector and public APIs.
- [ ] Add public release tags/versioning to the snapshot workflow when a release
      is intended; the current public history has snapshots but no release tag.
- [ ] Keep the public exporter allowlist reviewed whenever a new sample or
      required data file is added.

## Next: remove tessellation from node branches

Display-only mesh branches in `display_eval.rs` (boolean, extrude, loft,
surfaceloft and extrude preview) now tessellate through
`nodecraft_layers::tessellate_kernel` / `solidcore-tess` instead of the local
`brep_display_mesh`. `comparison.rs` now uses record/ops Booleans and
`solidcore-tess` for feature ghosts. Exact evaluation is unchanged.

The engine node branches no longer call `brep_display_mesh`. Remaining work is
the shared display scheduler and removing the old mesh-only branches themselves:

- [ ] Add a `DisplayTarget` mesher API that preserves:
  - stable mesh cache keys
  - `node-start` / `node-complete` progress
  - error attribution
  - feature-preview payloads
- [ ] Route `evaluate_part_display` and mesh-only targets through `DisplayTarget`.
- [x] Delete `brep_display_mesh` / `part_display_mesh` calls from engine node
      branches and comparison; both now use `nodecraft-layers`.
- [ ] Add parity tests before deleting each branch.
- [ ] Replace scatter's remaining display-specific path with the same display stage.

## Evaluation architecture

- [x] Add an exact-only engine entry point: `evaluate_exact` fills the Part cache
      and returns no display meshes.
- [x] Wire the app's EvalWorker as a two-phase evaluation: publish the exact
      result immediately (errors/status while the old mesh stays visible), then
      run the display pass against the warmed cache. A newer queued edit
      abandons the stale display pass after the exact phase. Export jobs skip the
      exact pre-pass so exporters run once, and progress is driven by the display
      pass only.

- [x] Add the exact-only engine entry point and the first two-phase app worker:
      exact result published before display, stale display abandoned after the
      exact phase, exports single-pass, display-owned progress.
- [ ] Split the engine API into exact evaluation and meshing:
  `evaluate_exact` then `mesh_exact`. This is the prerequisite for real
  parallelism; do not just move the monolithic evaluator onto a second pool.
- [ ] Add a dedicated display job/result channel and a display worker; exact and
      display must not share one blocking worker thread.
- [ ] Revision-stamp display results and reject stale/tab-mismatched meshes.
- [ ] Coordinate cache ownership between exact and display workers without
      duplicating the Part/record cache.
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

- [ ] Remove the top-bar debug Re-evaluate button once automatic content-hash
      invalidation (and any record cache) is the production path. Do not couple
      record-cache work to this temporary button.

- [ ] When STEP import works again, verify the Step importer inspector shows the
      detected application protocol and `FILE_SCHEMA`, and that its hover note
      describes the AP203/AP214/AP242 support boundary. Add a UI smoke/regression
      check for the label and cache refresh.
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

## SolidCore record cache (deferred)

Wait for upstream record/convert/ops/tess APIs to settle before making records
the authoritative exact value. Current kernel BReps remain authoritative; the
record is currently rebuilt and discarded on every display/ops conversion.
The next-generation coordinator below supersedes this as the target design.

- [ ] When upstream stabilises, add **one** exact-cache entry shape rather than
      two independent caches:
      `ExactEntry { record, kernel: Option<...> }`, with one explicit authority.
- [ ] Start with a derived record cache: cache `Arc<solidcore_record::brep::Brep>`
      beside the kernel BRep, keyed by the same body identity as `body_meshes`
      (pointer identity plus `Arc::ptr_eq`).
- [ ] Pass the record into `MeshRequest`/`MeshBody` and tessellate it directly
      with `solidcore-tess`; stop converting kernel → record per mesh request.
- [ ] Keep `nodecraft-layers` as the only conversion seam and add tessellation
      parity tests (kernel-converted record vs cached record).
- [ ] Do not flip `Part`/cache authority to records until upstream ops owns the
      remaining workflows: boolean, extrude, loft, fillet and chamfer.
- [ ] Only then promote the record to authoritative, derive the kernel on demand,
      and measure conversion cost on real graphs before deleting the kernel
      working representation.
- [ ] Do not add a second independent record cache: one entry, one invalidation
      path, one memory budget.

## Next-generation cache coordinator (outside the worker pool)

Goal: one high-speed, content-addressed result cache owned by a coordinator or
actor, not by a kernel thread. Kernels remain disposable per-worker scratch
state; killing and respawning a worker must not lose or invalidate authoritative
cache entries.

### Invariants

- The authoritative cache never lives in `Kernel`, a kernel caches object, a
  thread-local, or an `Arc` pointer identity.
- Cache values are immutable records or validated BReps owned by the coordinator.
- Cache keys are derived from the record/job input and every behavior-changing
  option, never from a worker id, thread id, allocation address or timestamp.
- A worker may only commit a result after the job's cancellation token is still
  current and validation has passed.
- The debug rebuild is a cache-bypass escape hatch, not the production
  invalidation path.

### Key shape

```text
CacheKey = hash(
    record schema/version,
    canonical record bytes or record content hash,
    operation identity,
    operation parameters,
    units and tolerances,
    kernel options and fixes,
    RNG seed
)
```

- Canonicalize before hashing; hash once per immutable record and carry the hash
  with it.
- Include every input that can change the result. Missing one creates a stale
  hit, which is worse than a miss.
- Version the key format; old entries must be unreadable after a schema or
  behavior change.
- Use a strong, deterministic hash; never use pointer identity or `HashMap`
  iteration order.

### Coordinator API

- `lookup(key) -> Option<Arc<Value>>`
- `get_or_reserve(key) -> Hit | Reserved` with single-flight/dedup so concurrent
  identical jobs compute once
- `commit(key, generation, Arc<Value>)` only when current
- `abandon(key, generation)` on cancellation/death; never leaves a poisoned entry
- `evict` by byte budget and a clock/LRU policy
- optional disk tier keyed by the same `CacheKey`, with a validated schema version

### Worker lifecycle

- Worker = kernel + job loop. No authoritative cache state.
- Cancelling a job cancels its token; a late result is dropped, not committed.
- Killing/respawning a worker creates a fresh kernel; it derives the kernel from
  the record on demand.
- In-flight reservations must be released if the worker dies; the coordinator
  must not wait forever for a dead reservation.
- Kernel memoisation caches are discarded with the worker and are never used to
  answer a cache lookup.

### Performance

- Keep cache values as `Arc`; copy only at the kernel boundary.
- Shard or actor-own the map to avoid a global mutex; no worker holds the lock
  while executing.
- Single-flight duplicate jobs; do not compute the same exact body twice.
- Hash records once, not per lookup; cache the record hash beside the record.
- Separate exact and display tiers, with independent byte budgets and eviction.
- Admit by size/benefit: avoid caching one-off huge or cheap results if they
  evict hot entries.
- Zero-copy record → `solidcore-tess` where possible; remove per-request kernel →
  record conversion.
- Measure hit latency, miss overhead, peak bytes and worker restart cost on real
  graphs, not microbenchmarks only.

### Migration stages

1. Define canonical record hashing and a versioned `CacheKey` type.
2. Build the coordinator API and tests with a synthetic value type.
3. Route one `EngineCache` family (exact BReps) through the coordinator.
4. Replace pointer-keyed `body_mesh` entries with record-hash keys.
5. Move Part/BRep/import/display tiers one at a time; keep single-flight and
   cancellation semantics at every step.
6. Add an optional disk tier only after in-memory behavior is stable.
7. Remove per-tab/per-worker result caches only once the coordinator is the sole
   authority.

### Acceptance

- The same record produces the same key across threads, tabs, worker respawns and
  process restarts.
- Killing a worker mid-job leaves no partial/poisoned cache entry.
- Two concurrent identical jobs produce one computation and one cache entry.
- A cache hit never needs a kernel; a miss can run on any worker.
- Exact and display caches have independent, bounded memory budgets and measurable
  eviction.
- Debug rebuild bypasses all reads and still produces the same result.
- A benchmark shows acceptable hit latency and no regression in full rebuild
  time.

### Open design questions

- Per-document cache versus process-global cache; how to isolate tabs.
- Whether the record itself or `Arc<Brep>` is the cached value, and where kernel
  conversion is allowed.
- Cost and location of record validation before commit.
- Persistence format, versioning and disk eviction policy.
- Whether the coordinator owns job scheduling or only cache reservations.
- How to handle multiple placements/instances of the same body without
  duplicating exact geometry.

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

- [ ] Upstream: expose public provenance attach/read (currently `pub(crate)`) or
      preserve application attributes through ops Boolean/shell. NodeCraft's
      Look colours and saved edge identity need this before the Part-target
      Boolean and shell paths can move off the kernel implementation.
- [ ] Fillet/chamfer `solidcore-ops::blend` migration is on hold; keep the
      current working `nodecraft_blend` path for now.

- [ ] **On every SolidCore pin bump, retest the G2-fillet top-face shell.**
      Reproduction at current pin `c314b6d26ca756184ef2bdd1742f5056854d3d07`:
      solid cylinder → G2 fillet (edge `edge:1`, centre `[-20, 0, -20]`, length
      `12.5`) → Shell (`face:2`, thickness `3`, inward). The shell currently
      fails with `Shell: solidcore-ops shell: the result is not a closed solid`
      while other face ids shell successfully. Check the SolidCore changelog and
      shell/offset tests when pulling a new revision; if it passes, add a
      NodeCraft regression and remove the limitation note. Do not add a local
      shell fallback to mask it.
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
- [ ] Add view-dependent retessellation: pass view/screen tolerance into the
      display plan, cache by tolerance/angle/view bucket, and re-plan on zoom
      thresholds. LOD alone cannot add detail; see
      [Mesher backend](../docs/mesher-backend.md#view-dependent-retessellation-not-implemented).
- [ ] Add the upstream commit/date to the pin review so a change in tessellator
      behaviour is visible when updating SolidCore.
- [x] Wire `solidcore_tess::simplify::levels` into the display path for LOD
      tiers. Initial selection uses camera distance / model radius; full-mesh
      key plus screen-size targets remain the refinement.
- [ ] Keep simplified levels display-only: they carry positions, triangles and a
      maximum surface error, but no `tri_face`, face groups, edge polylines or
      seam/smooth flags.
- [x] Keep picking, selection, edge rendering and export on the full
      tessellation until upstream preserves face provenance through
      simplification, or simplify per face group.
- [x] Add initial LOD regression coverage: large meshes produce coarser levels
      with finite error, and only the display object path consumes them.
- [ ] Delete the legacy `nodecraft-mesh` display path only after the STEP,
      primitive and sketch/spline mesh-only paths are measured against the
      parity harness.
- [ ] Try `Frame::turned_depth` from the camera's f64 view-projection for
      close-zoom depth precision. Supply it only when reversed depth is active,
      and leave it `None` when the scene projection is composed with a preview
      transform unless the turned depth is derived from that same matrix. Verify
      whether it fixes the broken/zi-fighting perspective presentation when
      zoomed in; it will not change texture interpolation or LOD.
