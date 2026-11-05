# Changelog

## Unreleased — 2026-11-05

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
