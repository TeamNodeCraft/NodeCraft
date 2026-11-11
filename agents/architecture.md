# How NodeCraft Works

## Authority chain

The implementation ordering and final ownership model are tracked in
[Workspace and graph-model integration plan](../docs/workspace-integration-plan.md).

```text
.nodecraft files       durable graph definitions
DocumentStore           canonical editable definitions
Rust Session            revision, hashes, dirty set, undo per definition
graph-instance flatten  one flat evaluation graph
evaluation engine       pure evaluator for one graph revision
worker response         meshes, errors, timings
egui app                command client, canvas and GPU viewport
```

The target ownership model is shown above. Today the app still flattens only the
`group`/`linkedpart` subset and its evaluation worker owns per-tab caches; the
migration is tracked in
[Graph definitions, instances and flattening](../docs/graph-definition-instance-plan.md).

The UI is a command client. It does not own a second graph or send an entire graph
on every edit.

A document is both a graph and a reusable definition. It can be used at the root,
embedded as an inline group, or loaded as a graph-instance node from another
`.nodecraft`. Those are presentations of one definition/instance model; the full
contract is in
[Graph definitions, instances and flattening](../docs/graph-definition-instance-plan.md).

## Source graph and evaluation graph

A source `GraphDocument` describes what the user authored, including
`GraphInstance` nodes that reference other definitions:

```text
GraphDocument
  schemaVersion: u32
  id: String
  nodes: Vec<GraphNode>
  connections: Vec<GraphConnection>
  interface: GraphInterface
```

Before evaluation, every reachable graph instance is flattened into one
evaluation `GraphDocument`. The engine only ever receives that flat graph. The
source graph is the source of truth; the flat graph is a cacheable derived view.

The live app still flattens the current `group`/`linkedpart` subset; the general
definition/instance migration is tracked in the plan linked above.

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
- graph-instance flattening (currently group payloads + linked parts)
- persistent evaluation caches owned by the evaluation runtime

Commands are explicit (`SetValue`, `SetPosition`, `SetSuppressed`, `Connect`,
`Disconnect`, `AddNode`, `RemoveNode`, and related operations). A transaction
records inverse commands and can roll back without cloning the whole document.
Undo/redo restores the authoritative document revision.

## Evaluation today

The app runs two independent workers plus a bounded tessellation pool.

### EvalWorker: exact, then display

```text
UI edit -> EvalJob(canonical document + reachable definitions)
        -> newest job wins; older queued jobs are discarded
        -> resolve and flatten graph instances into one flat graph
        -> exact phase: SolidCore B-rep / Part evaluation
        -> exact result (errors, status, previous mesh retained)
        -> display phase: solidcore-tess against the warmed Part cache
        -> mesh/error/timing response
        -> UI replaces the displayed result
```

Today the flattening step resolves inline group payloads and read-only
`linkedpart` references. The general definition/instance flattener and canonical
document editing are the target described in the plan; this section does not
claim they are implemented.

The exact and display phases are separate calls on the same worker thread. A
separate display-preparation worker converts the protocol mesh, splits crease
normals, extracts edges and prepares GPU object vertices off the UI thread. The
viewport also captures each body's `TessPlan` and evaluates it through
`solidcraft-viewport-gpu`'s `TessGpu`/`GpuMesh`, drawing the plan buffers through
the NodeCraft PBR pipeline; the CPU mesh remains the fallback and the source for
edges, picking, ghosts and comparison layers. The
exact phase is the authority; display meshes and their caches are disposable.
Display work is not yet fully concurrent. A newer edit cancels it cooperatively:
the display evaluator checks the revision token between nodes and the tessellation worker
checks it per body, but display still runs on the exact worker.

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

### Tessellation pool

`nodecraft-engine` owns a bounded `TessellationPool`. Body tessellation is
submitted per body and awaited by the evaluation worker. Evaluated Part/Boolean/shell,
primitive and STEP import display paths use `nodecraft-layers` /
`solidcore-tess`; the legacy `nodecraft-mesh` tessellator remains only for STEP
analytic edge curves that `solidcore-convert` cannot yet record.

The worker catches kernel panics, drops a poisoned cache and reports a node error
rather than killing the process.

## Graph definitions, instances and nesting

The engine only ever receives a flat graph. Graph instances are expanded by one
shared resolver with missing-file and cycle checks.

Nesting has two presentations:

```text
view nesting:       collapsed groups, breadcrumbs, frames
reference nesting:  a GraphInstance node pointing at a definition
```

View nesting is editor metadata. Collapsing/expanding it must not change
evaluation, hashes, dirty nodes or undo history. Reference nesting is semantic at
the node, but its definition is flattened before evaluation; it never creates a
nested evaluator, cache or kernel pool.

A definition has an interface (typed inputs/outputs and parameters). The instance
binds parent connections and values to that interface. A group, a loaded
`.nodecraft` and a base feature are the same concept:

- inline definition: an embedded group payload;
- file-backed definition: a `.nodecraft` loaded as a graph-instance node;
- root document: an implicit instance with no parent.

The canonical editable definition lives once in the `DocumentStore`. Opening the
same definition from another graph focuses that document; edits produce a new
definition revision and invalidate only dependent flattened graphs.

Invariants:

- engine requests contain only flat nodes and connections;
- definition resolution is shared by app, worker and CLI;
- cycle and missing-file errors are reported before evaluation;
- view-only grouping never affects content hashes;
- interface ports are the only legal cross-definition dependencies;
- no nested engine runtime, cache owner or kernel pool.

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

One evaluation runtime owns the caches, scheduler and kernel pool. Documents and
tabs do not own caches. The runtime keeps logical per-document cache partitions
so a large tab cannot evict another tab's working set, and all evaluation jobs
carry immutable flat-graph snapshots.

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
