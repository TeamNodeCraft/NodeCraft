# nodecraft-egui

First slice of the Swift/WebView front-end port to a native Rust egui application.

## Status

Working:

- left node-editor overlay over the full 3D viewport
- `lyon`-tessellated node cards, ports and smooth Bézier wires with retained GPU buffers
- `glyphon`/`cosmic-text` GPU text labels
- node cards, connection curves, node drag/select, graph pan/zoom
- eframe/egui shell: menu bar, node editor overlay, full model viewport, inspector, status bar
- wgpu viewport through `solidcraft-gpu` (`solidcraft-render` CPU fallback retained)
- `.nodecraft` graph loading through `nodecraft-session::GraphDocument`
- evaluation through the existing `nodecraft-engine` (the authority; no UI geometry work)
- CPU viewport built on `solidcraft-render` (`Camera`, `Scene`, `Canvas`)
- NodeCraft mesh/edge conversion through `solidcraft-geom`
- fit / iso / front / top camera commands
- headless smoke check: `nodecraft-egui --check path.nodecraft`

Not yet ported:

- real-time worker/session stream (the first slice evaluates synchronously)
- save/undo/redo and project handling
- broader wgpu viewport features: selection picking, section, MSAA/SSAA, GPU display-plan evaluation

## Node editing shortcuts

- `Cmd/Ctrl+C` — copy selected nodes as JSON
- `Cmd/Ctrl+X` — cut selected nodes as JSON
- `Cmd/Ctrl+V` — paste node JSON at the mouse position
- `Delete` / `Backspace` — delete selected nodes and cables
- drag on empty canvas — live lasso selection, updating node highlights while the lasso grows

The clipboard payload contains the selected `GraphNode` descriptions and any
connections whose endpoints are both selected. Paste accepts the same payload,
a bare node array, or a single node description, gives new nodes collision-free
IDs, and offsets them from their original positions.

## Run

```sh
cargo run -p nodecraft-egui -- ../samples/step-import.nodecraft
cargo run -p nodecraft-egui -- --check ../samples/step-import.nodecraft
```

## Dependency policy

This branch deliberately keeps the geometry authority and upstream boundaries
separate:

```text
solidcore                 immutable upstream git dependency; no local patches
solidcraft-geom           upstream git dependency
solidcraft-render         upstream git dependency
solidcraft-gpu            upstream git dependency (optional until the wgpu viewport lands)
nodecraft-*               this repository
```

The SolidCraft dependencies are consumed **unmodified** from
`https://github.com/tangency-labs/solidcraft`, branch `develop`. `Cargo.lock`
pins the exact commit; update with:

```sh
./scripts/update-solidcraft-crates.sh
```

Do not edit files under `.cargo/git/checkouts` or copy `solidcraft-geom`,
`solidcraft-render` or `solidcraft-gpu` into this repository. UI changes belong in
`rust/apps/nodecraft-egui`.

If a SolidCraft UI component must be modified, vendor only the UI crate (for
example `solidcraft-ui-egui`) under `rust/vendor/`, record its upstream revision
in `UPSTREAM_REV`, and keep the diff limited to UI concerns. Geometry, kernel and
solidcore code stay upstream and are changed only with an upstream PR.

## Port map

| Swift/WebView | egui port |
|---|---|
| AppKit window and menus | eframe viewport / egui `MenuBar` |
| graph canvas (`app.js`) | future egui node editor |
| Three.js viewer (`viewer.js`) | `solidcraft-gpu` wgpu viewport (CPU `solidcraft-render` fallback now) |
| inspector | `Panel::right` |
| node browser | `Panel::left` |
| settings dialog | upstream `solidcraft-ui-egui` patterns where reusable |
| worker HTTP/session bridge | direct `nodecraft-engine`/`nodecraft-session` calls |


## Theme

The UI colours live in `assets/theme.json`. The embedded JSON is the default.
Override it for a local build or run with either:

```sh
NODECRAFT_THEME=/path/to/theme.json ./build/nodecraft-egui.app/Contents/MacOS/nodecraft-egui
```

or place `theme.json` in:

```text
~/Library/Application Support/NodeCraft/theme.json
```

`selected`/`accent` is the selected blue used for selected nodes and their
upstream/downstream cables; `highlight` is the related hover/highlight colour.
`port_input` is the complementary orange used for input eyelets; output eyelets
use `accent`.
