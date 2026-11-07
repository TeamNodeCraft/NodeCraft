# Current State

NodeCraft is a native Rust application built with egui 0.36, eframe/wgpu and
SolidCore/SMLib for exact geometry.

## Working

- native Rust application and macOS `NodeCraft.app` bundle build
- `.nodecraft` / `.solidnode` document association and Finder open events on
  macOS
- flat node graph with stable IDs, undo/redo and linked documents
- scoped evaluation through an explicit per-tab viewer breakpoint; selecting a
  node does not move it
- node canvas with card/cable selection, delete and layout modes
- GPU viewport with SolidCraft rendering, grid, axes, display modes and theme
- STEP import through the SolidCore stack; imported bodies and every primitive
  (including wedge, partial sphere and partial torus) display through
  `nodecraft-layers` / `solidcore-tess` (with an explicit legacy fallback for
  STEP analytic edge curves that `solidcore-convert` cannot yet record)
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
- SolidCore pinned at `a00676f5c6e3a1e86cdb3b6d733ad662cf10202e` (latest
  `main` as of 2026-11-07)
- SolidCraft sketch stack pinned at `871014f955a76fd308ce156e68c93e6372906005`
- `configure_solidcore_fixes` applies the same upstream fix profile to our
  kernels and the `solidcore-ops` session (Boolean/shell/offset/blend)
- exact fillet, Boolean, union, shell and offset results are validated through
  `solidcore-ops::check`; an invalid result reports the upstream
  `CheckReport` text
- the approximate feature ghost is face-classified with a spatial surface index
  and drawn as a see-through overlay, so a re-meshed curved face does not ghost
  through the result in triangle slivers
- refused fillet edges are reported, not silently dropped from a multi-edge
  selection
- ghost face/edge colours live in the theme palette; the inspector reset action
  replaces the old disclosure chevron
- STEP planar import normalizes closed-solid winding after new face-use-aware exports

## Test baseline

Verified on 2026-11-07:

```text
cargo test --workspace --exclude solidcore-stepio --no-fail-fast
  846 passed / 1 failed / 11 ignored

the one failure is the documented sphere seam in nodecraft-mesh:
  remaining_ported_primitives_tessellate
```

Individual suites:

```text
nodecraft          123 passed
nodecraft-engine        121 passed / 1 ignored
nodecraft-mesh           18 passed / 1 failed / 1 ignored (lib)
nodecraft-mesh           mesh_quality 4 passed
nodecraft-blend          8 unit + integration suites
nodecraft-node-canvas    full canvas suite
nodecraft-session        full session suite
```

`solidcore-stepio` contains a 46-fixture SMLib mass-reference test that takes
substantially longer than the rest of the workspace suite. Keep it in a separate
slow tier rather than requiring it for every change.

The public snapshot does not include the large STEP corpora used by the private
matrix. Some private fixture tests may be skipped there.

## Runtime shape

The app is a command client over a flat, acyclic graph. The session owns the
document revision and dirty set. Evaluation runs in two phases on the app
`EvalWorker`: exact B-rep work first, then display tessellation against the
warmed Part cache. A separate branch-only `PreviewWorker` resolves the transform
input for the edit ghost, and a bounded mesher pool performs body tessellation.
See [How NodeCraft works](architecture.md).

The evaluator is split internally: `exact_eval.rs` owns Part/B-rep construction
and `display_eval.rs` owns display meshing and mesh caching, with cache,
primitive, look, bypass and comparison concerns in their own modules. Display work is not yet fully concurrent: the `EvalWorker` is still sequential
after the exact phase. A newer edit cancels the display pass cooperatively; the
evaluator checks the revision token between nodes and the mesher checks it per
body.

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

- A partial fillet around a multi-edge torus/cylinder join is still an upstream
  seam/edge-use problem (SolidCore #2826 remains open; #2863 is merged). NodeCraft
  now reports the upstream `Smooth` refusal instead of silently landing a
  partial round.
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
