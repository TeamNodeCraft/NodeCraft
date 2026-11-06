# Current State

NodeCraft is a native Rust application built with egui 0.36, eframe/wgpu and
SolidCore/SMLib for exact geometry.

## Working

- native egui application and macOS bundle build
- flat node graph with stable IDs, undo/redo and linked documents
- scoped evaluation through an explicit per-tab viewer breakpoint; selecting a
  node does not move it
- node canvas with card/cable selection, delete and layout modes
- GPU viewport with SolidCraft rendering, grid, axes, display modes and theme
- STEP import through the SolidCore stack; older reader paths remain for
  comparison
- JSON-backed dark theme, first-run creation and corrupt-theme repair
- native OS settings viewport with display and diagnostic sections
- macOS-style menus with modeless popup rows
- public snapshot build and GitHub Actions matrix for macOS/Linux/Windows
- Geometry graph value carrying exact BRep solids and sketches
- Geometry-native primitive, translate, mirror, combine, Boolean, STEP import,
  extrude, sketch, spline, loft, surfaceloft, shell and scatter evaluation
- rendered viewport targets evaluate through the Geometry evaluator
- Part/Geometry cache reuse and exact cache-hit reporting
- face/edge/vertex selection filter dock in the viewport
- shell opening-face selection from the viewport, with inspector chips
- sharp-wall shell attempt using SolidCore's extended offset path
- SolidCore pinned at `3ed6194755cc2a3298882373ff465adeb443485a`
- SolidCraft sketch stack pinned at `d7cad1005a7ab8ed948d7c8cca666230e988c579`
- STEP planar import normalizes closed-solid winding after new face-use-aware exports

## Test baseline

Latest local baseline after the 2026-11-06 cleanup:

```text
cargo test --workspace --exclude solidcore-stepio --no-fail-fast
  808 passed / 1 failed / 11 ignored

the one failure is the documented sphere seam in nodecraft-mesh:
  remaining_ported_primitives_tessellate

targeted:
  nodecraft-egui          108 passed
  nodecraft-node-canvas    54 passed / 4 ignored
  nodecraft-session        53 passed
  nodecraft-engine        114 passed / 1 ignored
  nodecraft-node-contract   5 passed
  nodecraft-blend           8 unit + 5 integration
```

`solidcore-stepio` contains a 46-fixture SMLib mass-reference test that takes
substantially longer than the rest of the workspace suite. Keep it in a separate
slow tier rather than requiring it for every change.

The public snapshot does not include the large STEP corpora used by the private
matrix. Some private fixture tests may be skipped there.

## Runtime shape

The app is a command client over a flat, acyclic graph. The session owns the
document revision and dirty set; an evaluation worker owns the persistent engine
cache and currently performs exact B-rep work plus display tessellation in one
call. See [How NodeCraft works](architecture.md).

The evaluator is split internally now: `exact_eval.rs` owns Part/B-rep
construction and `display_eval.rs` owns display meshing and mesh caching, with
cache, primitive, look, bypass and comparison concerns in their own modules.
The remaining architectural work is to give exact and display work independent
bounded pools and a scheduler rather than wrapping one worker call.

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
