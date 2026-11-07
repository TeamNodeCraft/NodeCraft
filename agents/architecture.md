# How NodeCraft Works

## Authority chain

```text
.nodecraft file
   -> Rust Session (authoritative revision, hashes, dirty set, undo)
   -> evaluation engine (one graph revision)
   -> worker response (meshes, errors, timings)
   -> egui app / node canvas / GPU viewport
```

The UI is a command client. It does not own a second graph or send an entire graph
on every edit.

## The DAG

A document is one flat `GraphDocument`:

```text
GraphDocument
  schemaVersion: u32
  id: String
  nodes: Vec<GraphNode>
  connections: Vec<GraphConnection>
  interface: GraphInterface
```

A node has:

```text
id          stable string identity
type        node type name
values      operation parameters, currently JSON
suppressed  disabled state
x, y        canvas position, intentionally excluded from geometry hashes
```

A connection has:

```text
from     producer node id
to       consumer node id
input    named consumer input
out      named producer output
```

The graph is a **directed acyclic graph**. Cycles are rejected by the session's
topological hash pass. Evaluation follows the target closure in topological order;
downstream nodes consume already-computed outputs and no branch can re-enter a
node.

Content hashes include node type, values, suppression and sorted upstream input
hashes. They exclude node positions. Editing a value marks that node and its
downstream consumers dirty; layout edits do not dirty geometry.

## Session and cache

`nodecraft-session` owns:

- the revisioned `GraphDocument`
- content hashes
- dirty-node propagation
- command transactions and undo/redo
- linked-part expansion
- the persistent `EngineCache` used across edits

Commands are explicit (`SetValue`, `SetPosition`, `SetSuppressed`, `Connect`,
`Disconnect`, `AddNode`, `RemoveNode`, and related operations). A transaction
records inverse commands and can roll back without cloning the whole document.
Undo/redo restores the authoritative document revision.

## Evaluation today

The app runs two independent workers plus a bounded mesher pool.

### EvalWorker: exact, then display

```text
UI edit -> EvalJob(session document + linked documents)
        -> newest job wins; older queued jobs are discarded
        -> expand linked parts into one flat graph
        -> exact phase: SolidCore B-rep / Part evaluation
        -> exact result (errors, status, previous mesh retained)
        -> display phase: solidcore-tess against the warmed Part cache
        -> mesh/error/timing response
        -> UI replaces the displayed result
```

The exact and display phases are separate calls on the same worker thread. The
exact phase is the authority; display meshes and their caches are disposable.
Display work is not yet fully concurrent: it is cancelled only when a queued job
exists at the exact/display boundary.

### PreviewWorker

The transform edit ghost does not wait for the breakpoint evaluation. Selecting
a translate node sends a branch-only preview request to its own worker:

```text
preview target = transform input node
  -> own thread
  -> own EngineCache / FlatGraphCache
  -> exact + display for that branch only
  -> ghost mesh
```

The main breakpoint mesh is never moved. The ghost is a standalone object
renderer with its own buffers, opacity uniform and depth state, composited over
a dimmed breakpoint. Crease-smoothed normals drive the zebra reflection analysis.

### Mesher pool

`nodecraft-engine` owns a bounded `MesherPool`. Body tessellation is submitted
per body and awaited by the evaluation worker. Evaluated Part/Boolean/shell and
STEP import display paths use `nodecraft-layers` / `solidcore-tess`; the legacy
`nodecraft-mesh` curved tessellator remains only as a STEP fallback for analytic
edge curves that `solidcore-convert` cannot yet record.

The worker catches kernel panics, drops a poisoned cache and reports a node error
rather than killing the process.

## Linked documents and nesting

The engine only ever receives a flat graph. Linked parts are expanded by one
shared resolution boundary with missing-file and cycle checks.

Nesting is primarily a view concept:

```text
view:       collapsed groups, breadcrumbs, group frames
territory:  one flat GraphDocument and one flat DAG
```

Collapsing/expanding a group must not change evaluation, hashes, dirty nodes or
undo history. Group membership is editor metadata.

Reusable component definitions, instances, interfaces and linked `.nodecraft`
files are real semantic nodes. They are resolved and flattened before the engine
boundary. A linked part may present as one collapsed node; its expanded nodes
are the territory.

Invariants:

- engine requests contain only flat nodes and connections
- linked resolution is shared by app, worker and CLI
- cycle and missing-file errors are reported before evaluation
- view-only grouping never affects content hashes

## Viewer breakpoint and evaluation scope

The viewer breakpoint is the node whose result the viewport is currently
inspecting. It is explicit and stored per tab. Opening a document pins it to the
initial viewer target; the node's eye control changes it; selecting a node does
not move it. A request targets just that node.

The engine evaluates the **target closure**: the breakpoint and its upstream
dependencies, not the whole graph and not downstream nodes. This is what makes a
small feature inspectable without rebuilding unrelated geometry.

Current behaviour:

```text
explicit viewer_breakpoint -> target request -> upstream closure
```

Current semantics:

- the breakpoint is an explicit **scope root**, independent of selection
- it is pinned and persisted per tab
- moving it cancels stale work for the old scope but keeps reusable cache entries
- a selected node outside the closure is evaluated only as a validation target,
  so its error reaches the canvas glow without changing the displayed mesh
- downstream nodes are never evaluated merely because the breakpoint moved

Planned semantics:

- breakpoint, preview and export are sibling scopes
- a breakpoint request returns both exact B-rep work and its display mesh

The scoped DAG is a core modelling concept, not a debug workaround.

## Future thread-pool plan

Current state is one monolithic engine thread. The plan is not to put that same
call behind a second pool; that would only make the pools compete for the same
job.

The future boundary is:

```text
evaluate_exact(request) -> ExactOutput
mesh_exact(exact, meshTargets, lod) -> Meshes
```

Then:

- **exact pool** — bounded SolidCore work: imports, booleans, extrude, heal
- **mesh pool** — bounded display tessellation and LOD generation
- **coordinator** — serializes session commands on the exact side, overlaps mesh
  jobs with exact jobs, cancels by graph revision, prioritises visible/selected
  work, and drops stale results
- **separate caches** — exact B-rep/import cache and mesh cache keyed by graph
  revision + node + LOD
- **protocol extension** — exact results may arrive before display meshes,
  matched by revision/node/LOD

Required tests:

- exact output is unchanged by the split
- chained `evaluate_exact` + `mesh_exact` matches the old single-call mesh
- a slow mesh job cannot block a graph edit
- cancelled/stale jobs are ignored

Presentation/animation will get its own runtime and scheduler later. It must not
drive B-rep evaluation or occupy the exact/mesh pools.

## Typed contracts and data-flow research direction

The flat DAG and content-hash cache are a good foundation. The next step we want
to take from our node-editor research is a real runtime contract instead of
untyped JSON values and string ports.

### Target concepts

```text
NodeValue   typed runtime value
Pear        value + per-item metadata
Twig        ordered list of pears
Tree        ordered map of path -> twig
PortSpec    value kind + access + requirement + default
```

- `NodeValue` covers null, booleans, integers, numbers, text, points, vectors,
  planes, transforms, curves, surfaces, B-reps, meshes and sketch profiles.
- Geometry is immutable and shared (`Arc`), not copied through the graph.
- JSON remains the durable file format; conversion happens at the document
  boundary, not inside evaluation.
- Ports declare `Item`, `Twig`/list or `Tree` access.
- Missing input, a null input and an empty list are distinct states.
- Inputs are immutable; long operations receive a cancellation token.
- A solution has an identity, so stale results can be rejected.
- Per-node warnings/errors and preview flags are explicit outputs.
- A node/parameter descriptor is the single source of truth for ports,
  defaults, optionality, UI metadata and documentation.

### Phased rollout

1. **Typed contract** — value enum, port specs, descriptor, JSON migration.
2. **List/tree semantics** — explicit access, replication/lacing and per-item
   metadata, with deterministic flattening at the port boundary.
3. **Exact/mesh split** — the pool architecture above, with solution identity and
   cancellation.
4. **Metadata and migration** — stable component metadata and document-version
   upgrades.
5. **Editor surface** — search/documentation metadata and a schema-driven
   inspector, only where the descriptor cannot express the UI.

We are not adopting an external plugin SDK, a second geometry kernel, or
document/UI coupling from that prior art. We are adopting the contracts that
make a node system predictable.
