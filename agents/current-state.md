# Current State

NodeCraft is a native Rust application built with egui 0.36, eframe/wgpu and
SolidCore/SMLib for exact geometry.

## Working

- native egui application and macOS bundle build
- flat node graph with stable IDs, undo/redo and linked documents
- scoped evaluation through a viewer breakpoint (currently selection-driven)
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
- SolidCore pinned at `d85f648466a752e00d6e1ce8f546e0be0db0896c`
- STEP planar import normalizes closed-solid winding after new face-use-aware exports

## Test baseline

Latest local baseline for the public subset:

```text
nodecraft-egui          62 passed
nodecraft-node-canvas   19 passed
nodecraft-engine        75 passed / 1 ignored
```

The public snapshot does not include the large STEP corpora used by the private
matrix. Some private fixture tests may be skipped there.

## Runtime shape

The app is a command client over a flat, acyclic graph. The session owns the
document revision and dirty set; an evaluation worker owns the persistent engine
cache and currently performs exact B-rep work plus display tessellation in one
call. See [How NodeCraft works](architecture.md).

The monolithic evaluator is the main architectural debt. The next stage is to
split exact evaluation from display meshing and give them independent bounded
pools and caches, not to wrap the current monolith in a second thread pool.

## Known shell limitations

Shelling is not yet general. Current state:

- box shelling works with sharp inner/outer walls
- cylinder and sphere shell through the curved path but are slow and need
  volume/orientation validation
- cone shelling currently fails at curved display tessellation
  (`solidcore curved tessellation produced no triangles`)
- inner-shell orientation still needs a correctness audit
- openings are accepted as body-scoped face references and resolved to
  SolidCore `FaceId`s
- the offset options (`extend_convex_edges`, `merge_results`,
  `create_offset_solid`, `merge_sheet_side_walls`) remain engine policy, not
  inspector controls

The next shell work is per-input option selection, volume/area validation for
curved primitives, and contextual face editing from the inspector.

## Platforms

macOS is the daily-driver and packaged target. Linux and Windows are kept
compile-friendly through CI; they have not been interactively verified on this
machine. macOS-specific AppKit behaviour is behind `cfg(target_os = "macos")`.

## Packaging

The public snapshot is a sanitized source export. It intentionally excludes
private reference checkouts, large test corpora, generated reports and local
configuration.
