# Changelog

## Unreleased — 2026-11-05

### Build

- Updated SolidCore to `d93a19bd` (current `main`).

### Sketch editor

- Fillet now skips a smooth edge inside a mixed join selection and rounds the
  remaining blendable edges instead of failing the whole feature.

### Node editor

- Viewer edge ids now resolve by SolidCore `EdgeId` index rather than edge-list
  ordinal. After a Boolean those differ, so a saved join edge was filleting a
  cylinder cap edge instead.

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
