# Current State

NodeCraft is a native Rust application built with egui 0.36, eframe/wgpu and
SolidCore/SMLib for exact geometry.

## Working

- native Rust application and macOS `NodeCraft.app` bundle build; the
  application crate is `#![forbid(unsafe_code)]` and the audited
  `nodecraft-macos` crate is the only local FFI boundary
- the application has a local panic dialog; panic text is shown in-process and
  written to the temp directory, never sent
- `nodecraft-expr` carries the SolidCraft-compatible parameter expression
  grammar and dimensional evaluator; numeric fields use a thin unit-kind
  adapter that shows the source expression while focused and the evaluated
  number otherwise; expressions persist in each node's `expressions` map
  (`schemaVersion: 1`, with legacy `values._expressions` migrated on load)
- Inter is embedded in the executable for all platforms
- saves are explicit; autosave and untitled-document recovery are not yet
  implemented
- `.nodecraft` document association and Finder open events on
  macOS
- flat node graph with stable IDs, undo/redo and linked documents
- scoped evaluation through an explicit per-tab viewer breakpoint; selecting a
  node does not move it
- node canvas with card/cable selection, delete and layout modes
- GPU viewport with SolidCraft rendering, grid, axes, display modes and theme
- STEP import through the official reader; imported bodies and every primitive
  (including wedge, partial sphere and partial torus) display through
  `nodecraft-layers` / `solidcore-tess` only
- JSON-backed dark theme, first-run creation and corrupt-theme repair
- native OS settings viewport with display and diagnostic sections
- macOS-style menus with modeless popup rows
- public snapshot build and GitHub Actions matrix for macOS/Linux/Windows
- branch-only transform preview on a dedicated `PreviewWorker` with its own
  evaluation/cache, so the edit ghost does not wait behind the breakpoint pass
- Geometry graph value carrying exact BRep solids and sketches
- Geometry-native primitive, translate, mirror, combine, Boolean, STEP import,
  extrude, sketch, spline, loft, surfaceloft, shell and scatter evaluation
- rendered viewport targets evaluate through the Geometry evaluator
- Part/Geometry cache reuse and exact cache-hit reporting
- face/edge/vertex selection filter dock in the viewport
- shell opening-face selection from the viewport, with inspector chips
- sharp-wall shell attempt using SolidCore's extended offset path
- SolidCore vendored in SolidCraft at `c314b6d26ca756184ef2bdd1742f5056854d3d07`;
  every SolidCore module, including the record crate, comes from that tree
- SolidCraft sketch stack is pinned to the same revision. The latest update moved
  the shared camera/scene/rasterizer to `solidcraft-viewport` and the wgpu
  pipelines to `solidcraft-viewport-gpu`; NodeCraft has been migrated at that
  seam.
- The workspace and release app build are warning-free; inspector controls share
  one layout/inset and the scroll bar uses normal ScrollArea hit geometry.
- Feature edge editing uses a breakpoint + input-body edit ghost. This is still
  work in progress; see `agents/bugs.md`.
- `configure_solidcore_fixes` applies the same upstream fix profile to our
  kernels and the `solidcore-ops` session (Boolean/shell/offset/blend)
- exact fillet, Boolean, union, shell and offset results are validated through
  `solidcore-ops::check`; an invalid result reports the upstream
  `CheckReport` text
- G2/G3 feature ghosts use the exact fillet result's generated blend faces,
  mapped through the record face order; the display-surface classifier remains
  the fallback, so a fillet highlights its own surface rather than the body
- refused fillet edges are reported, not silently dropped from a multi-edge
  selection
- circular fillets expose SolidCore's mitred or swept corner turn
- ghost face/edge colours live in the theme palette; the inspector reset action
  replaces the old disclosure chevron
- STEP planar import normalizes closed-solid winding after new face-use-aware exports

## Test baseline

Targeted after the `solidcore-tess` migration:

```text
nodecraft          134 passed
nodecraft-engine   125 passed / 1 ignored
nodecraft-mesh      7 passed
nodecraft-solidcore 48 passed / 1 ignored
nodecraft-bool      20 passed
```

STEP import now uses the official reader and `solidcore-tess`; the old
`solidcore-stepio` façade and local mesher have been removed.

The public snapshot does not include the large STEP corpora used by the private
matrix. Some private fixture tests may be skipped there.

## Runtime shape

The app is a command client over a flat, acyclic graph. The session owns the
document revision and dirty set. Evaluation runs in two phases on the app
`EvalWorker`: exact B-rep work first, then display tessellation against the
warmed Part cache. A separate branch-only `PreviewWorker` resolves the transform
input for the edit ghost, and a bounded tessellation pool performs body tessellation.
See [How NodeCraft works](architecture.md).

The evaluator is split internally: `exact_eval.rs` owns Part/B-rep construction
and `display_eval.rs` owns display meshing and mesh caching, with cache,
primitive, look, bypass and comparison concerns in their own modules. Display work is not yet fully concurrent: the `EvalWorker` is still sequential
after the exact phase. A newer edit cancels the display pass cooperatively; the
evaluator checks the revision token between nodes and the tessellation worker checks it per
body.

## Workspace transition

The workspace/graph transition is complete for its current scope. Graph
definitions and instances, direct local/external/nested flattening, definition
validation/hashing, graph tabs as views over one stored document session,
definition-scoped and tab-scoped undo, duplicate views, split/nodes-only/3D-only
layouts, per-tab pane widths and view breadcrumbs are implemented. Definition
projections are rebuildable view caches, not stored document sessions.

Deferred beyond this scope: multiple simultaneous panes per tab, true
multi-scope evaluation, iterator execution, fold, externalise/pack lifecycle and
the public snapshot republish. See
[Workspace transition TODO](../docs/workspace-transition-todo.md).

## Planned graph model

The target document model is a canonical graph definition plus graph instances:
groups, loaded `.nodecraft` files and base features use the same interface,
parameters and flattening rules. The engine receives one flat graph; nesting is
navigation, not runtime nesting. The app still uses the current `group` payload
and read-only `linkedpart` paths. Design:
[Graph definitions, instances and flattening](../docs/graph-definition-instance-plan.md).

## Known shell limitations

Shelling is not yet general. Current state:

- box shelling works with sharp inner/outer walls
- cylinder and sphere shell through the curved path but are slow and need
  volume/orientation validation
- cone shelling currently fails at curved display tessellation
  (`solidcore curved tessellation produced no triangles`)
- inner-shell orientation still needs a correctness audit
- shelling the top face after a G2 fillet fails with
  `the result is not a closed solid`; this remains an explicit error, not a
  silent fallback
- openings are accepted as body-scoped face references and resolved to
  SolidCore `FaceId`s
- the offset options (`extend_convex_edges`, `merge_results`,
  `create_offset_solid`, `merge_sheet_side_walls`) remain engine policy, not
  inspector controls

The next shell work is per-input option selection, volume/area validation for
curved primitives, and contextual face editing from the inspector.

## Known geometry limitations

- The torus/cylinder join fillet is continuous in the app: one picked join edge
  expands through the SolidCore tangent chain and rounds the full run. Upstream
  #2826 remains open for callers that pass a single edge without the chain.
- STEP bodies whose edge curves have no B-spline form cannot yet be
  recorded by `solidcore-convert`; STEP import falls back to the legacy display
  tessellator for those bodies only.
- `nodecraft-mesh`'s pole-free sphere tessellation still fails its manifold seam
  assertion; this is the single workspace test failure above.

## Node error discovery

Selected nodes outside the viewed target closure are evaluated as validation
targets so their errors reach the canvas glow without changing the breakpoint.
Unviewed, unselected branches are not evaluated and therefore cannot report
errors; global background validation is not implemented.

## Platforms

macOS is the daily-driver and packaged target. Linux and Windows are kept
compile-friendly through CI; they have not been interactively verified on this
machine. macOS-specific AppKit behaviour is behind `cfg(target_os = "macos")`.

## Packaging

The public snapshot is a sanitized source export. It intentionally excludes
private reference checkouts, large test corpora, generated reports and local
configuration.
