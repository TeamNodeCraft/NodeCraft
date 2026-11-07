# NodeCraft app model–view semantics

## Ownership

`NodeCraftApp` owns exactly one authoritative model: `Option<nodecraft_session::Session>`.

- Reads of the graph go through `Session::document()`.
- Mutations go through `Session::apply_commands` / `apply_commands_coalesced`.
- Undo/redo is owned by `Session`; the app never stores a parallel document.
- The engine worker receives a snapshot of the session document and returns a
  tagged result. Results for an older request ID are discarded.

## View

`nodecraft-node-canvas` is a reusable view over `&GraphDocument`. Its `NodeEditor` never mutates the graph.

Interaction produces `NodeEditorAction` values:

- `SelectNode { id, additive }`
- `SelectCable { key, additive }`
- `SetLassoSelection { nodes, additive }`
- `MoveNodes { positions }`
- `SetPositions(Vec<(id, x, y)>)`
- `CleanupLayout { mode }`
- `Deselect`

The app translates those actions into session `Command`s. Group drags are
coalesced under `move:nodes`; layout cleanup is one `cleanup-layout` transaction.

## Identity

- Node selection is by stable node ID.
- Cable selection is by `(from, to, input, out)` `ConnectionKey`.
- Undo/redo remaps selection to surviving IDs/keys instead of clearing it.
- Graph view fitting follows document identity, not node count or vector index.

## Theme

The app owns the JSON theme at
`rust/apps/nodecraft/assets/theme.json` and exposes it through
`crate::theme`. The canvas receives a `CanvasTheme` palette; it does not load
or read the app theme file itself.

## Evaluation

- Evaluation runs on a background worker with the engine cache resident there.
- Exact and display are separate phases: the exact result reports errors and
  status first, then display tessellation runs against the warmed Part cache.
- Only the newest request ID is applied.
- The renderer keeps the previous mesh until a newer result is accepted.
- Stale results can never overwrite a newer graph revision.
- The transform edit ghost uses a separate branch-only `PreviewWorker`; it does
  not move or wait for the breakpoint mesh.
- Refused fillet edges and failed upstream validity checks propagate as node
  errors rather than partial display geometry.

## GPU fallbacks

- MSAA defaults to 4× on macOS and 1× elsewhere. Override with
  `NODECRAFT_MSAA=1` or `NODECRAFT_MSAA=4`.
- The frosted backdrop uses `Rgba16Float` when the adapter supports
  filterable render-attachment usage, otherwise it falls back to
  `Rgba8Unorm`. `NODECRAFT_FORCE_RGBA8=1` exercises the fallback.
- If the node renderer cannot initialize, `NodeEditor` draws a CPU fallback
  of cards, ports, cables and labels. `NODECRAFT_CPU_NODES=1` exercises it.
- The viewport already has a CPU raster fallback when GPU installation is
  unavailable.
