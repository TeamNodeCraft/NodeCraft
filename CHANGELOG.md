# Changelog

## Unreleased — 2026-11-07

### SolidCore integration

- Updated SolidCore to `a00676f5` (latest `main`) and the SolidCraft sketch
  stack to `871014f9`. The SolidCore delta includes the blend end-section and
  smooth-vertex run-on fixes.
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
- The approximate feature-difference overlay now classifies whole faces with a
  spatial surface index instead of per-triangle centroid tests. Curved faces
  re-meshed by a fillet no longer produce a jagged triangle-soup outline, and
  the overlay still shows the feature from both sides.
- G2/G3 fillet overlays skip the unsupported exact Boolean difference directly;
  failed exact comparisons are cached too. Selecting the test fillet graph went
  from a 1.27 s failed exact comparison to a ~0.1 s cached evaluation.
- The display popup has a fixed width, aligned columns and no slider jitter.

### Viewport

- Zero-length edge segments are skipped in GPU and 2D edge rendering, removing
  the vertex marker/“ball” that appeared at duplicated polyline points.

### App

- Renamed the desktop application package and binary from `nodecraft-egui` to
  `nodecraft`; the macOS bundle is now `NodeCraft.app` with the `NodeCraft`
  executable.
- Registered `.nodecraft` and `.solidnode` documents with macOS. Finder opens
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
