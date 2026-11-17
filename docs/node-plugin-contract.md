# Built-in node contract

One built-in node is one module under
`rust/crates/nodecraft-engine/src/nodes/`, implementing one trait and listed
once in the registry in `nodes/mod.rs`.

The old `node_plugin.rs` template ("mesh + exact channel", implemented by zero
nodes) is gone. It predated `Part`, `SolidBody` and proofs, `NodeOutput` /
`NodeInputs`, `OpsSession`, shared plans, warnings and revision cancellation, so
it described an engine that no longer exists.

## What a node implements

```rust
pub(crate) trait BuiltinNode: Sync {
    fn node_types(&self) -> &'static [&'static str];

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String>;

    fn effect(
        &self,
        node: NodeView<'_>,
        ctx: &mut EffectContext<'_>,
    ) -> Result<Option<NodeEffect>, String> { Ok(None) }

    fn exact_weight(&self, node: NodeView<'_>) -> usize;
    fn diagnostics_scope(&self) -> DiagnosticsScope { DiagnosticsScope::Node }
    fn export_terminal(&self) -> bool { false }
    fn output_kind(&self) -> OutputKind { OutputKind::Part }
    fn mesh_bodies(&self, node: NodeView<'_>) -> usize { 0 }
    fn suppression_source(
        &self,
        node: NodeView<'_>,
        connections: &[Connection],
    ) -> Option<String>;
    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> { None }
    fn repeats_input_bodies(&self) -> bool { false }
    fn transform_role(&self) -> Option<TransformRole> { None }
    fn merge_provenance(
        &self,
        node: NodeView<'_>,
        connections: &[Connection],
        affected: &HashSet<String>,
    ) -> MergeProvenance { MergeProvenance::Preserved }
}
```

Each method exists because at least one ported node needs it today:

| Method | Question it answers | Today's dispatch it replaces |
|---|---|---|
| `node_types` | Which contract ids share this implementation? | registry key |
| `evaluate` | What is the node's exact `Part`? | `exact_eval::evaluate_part` arms |
| `exact_weight` | How expensive is one exact run? | `workload::exact_weight` |
| `mesh_bodies` | How many display bodies does it estimate? | `workload::mesh_bodies` |
| `suppression_source` | What does a suppressed node pass through? | `bypass::suppression_source` |
| `body_layout` | Which display bodies does its Part produce, given per-port input layouts? | session `placement::part_body_layout` |
| `repeats_input_bodies` | Is it a repeater for drag preview? | session `placement::repeats_input_bodies` |
| `transform_role` | Is it the translate or mirror step of the preview chain? | session `placement::Affine3::from_node` |
| `merge_provenance` | Does a merge keep its operands' provenance? | session preview's Boolean `a`/`b` rule |
| `effect` | Does the node write an output file, and under which policy? | display evaluator's exporter arm and the app's export write |
| `diagnostics_scope` | Is this node's debug diagnostics collected in the normal pass or an isolated one? | the worker's `stepimporter` match |
| `export_terminal` | Can this node terminate an export placement chain? | `lib.rs::resolve_part_chain`'s type list |
| `output_kind` | Is the output a Part, a Sketch, or an effect? | display/overlay and export-demand type checks |

`PartEval` context methods stay narrower than the request graph:

- `input_state(port) -> InputState` answers `Unconnected`, `Inactive` or
  `Active` using the one `effective_output_suppressed` walk, so a Boolean never
  evaluates an inactive operand just to discover it contributes nothing.
- `union_run(node) -> Option<UnionRun>` resolves the existing
  `fused_union_leaves` plan and evaluates its leaves, so a node receives only
  the leaf Parts in chain order. The engine still owns
  `union_run_intermediates`; the node never sees the request graph.
- `step_import(node) -> Result<StepImport, String>` keeps path/file identity,
  the display-import cache and counters, and returns every converted body with
  its warnings and diagnostics; the importer node builds `Part::from_imported`.
- `record_diagnostics(Vec<NodeDiagnostic>)` records a node's debug diagnostics
  in the same JSON object shape the inspector reads.
- `export_input(node) -> ExportInput` resolves the exporter's input Part, its
  placement chain and any Combine members once in the engine. `EffectContext`
  carries `run_exporters`, the project directory and the current cancellation
  token; validation/ghost passes never write and a cancelled revision writes
  nothing.

## One node view and one exact representation

`NodeView<'a>` is the engine-owned borrowed node shape: `id`, `node_type`,
`values` and `suppressed`. The protocol `Node` converts into it with `From`, and
the session converts its `GraphNode` with `NodeView::new`; neither clones the
values, and the engine still does not depend on `nodecraft-session`.

`body_layout` takes a slice of `(port, upstream layout)` in connection order.
A one-input node takes `"shape"`; Combine concatenates every `"shape"` input.
This is what lets a multi-input node express its Part body order without a
node-type branch in the session.

There is one exact operand representation: a node's evaluated `Part`, realised
and required to be a single body. The old raw-operand route
(`exact_operand_brep` on the trait, the test-only `operand_brep`) is gone.
`lib.rs::exact_operand_brep` and its plural collector both call
`Eval::evaluate_part`, so a registered node and an unported node take the same
path, and a multi-body Part still errors with "an exact operand needs exactly
one body" instead of being collapsed to its first shape.

There is no test-only `display_mesh` arm. Production display and tests both
consume the exact `Part`; the display evaluator tessellates the Part instead of
building a second mesh-only value.

The trait is object-safe and has no per-node special cases. A node module knows
only its own `node_types`; it never matches another node's type string. The
engine asks `nodes::find(node_type)` and calls the trait.

## Values, proofs and jobs

- `evaluate` returns the same immutable `Part` the engine uses: shared
  `SolidBody` bodies plus placement chains, provenance (`source`/`source_body`),
  sketches and imported names/colours. Do not invent a parallel value type.
- Inputs are read by port through `PartEval::input_part`, which calls
  `Eval::evaluate_part`. That keeps the Part cache, prewarmed B-reps, bypass,
  provenance tagging and warnings on the one existing path.
- Kernel work is an `ExactNodeTask` run with `PartEval::execute_warned`
  (`execute_exact_warned`): the shared persistent kernel pool, the current
  revision cancellation token, `OpsSession` under `Validation::EachNode`, and
  the task's warnings. `OpsSession` blend/shell/offset operations prove their
  input at the operation boundary (`SolidBody::prove`), and each body keeps its
  shared proof state, so one proof serves every reader.
- `PartEval::warn` appends the task's warnings to the node's entry in
  `Response::node_warnings`.
- Field values and defaults come only from `nodecraft-node-contract`:
  `resolve_number` / `resolve_bool` / `resolve_text` / `resolve_choice` (or the
  engine's `contract_number` / `contract_bool` / `contract_text` wrappers). The
  `contract-defaults-only` ast-grep rule must stay at zero findings. Selector
  keys that are not fields (`primitiveType`, `edges`, `faces`, `path`, ...) are
  read directly; they have no contract default.

## Crate placement

Engine behaviour lives in `nodecraft-engine/src/nodes/`. A node module is a
child of the engine crate, so it can use `Part`, `BodyInstance`,
`BodyTransform` and `ExactNodeTask` directly; the public boundary is the two
small context traits, not `Eval`.

UI presentation metadata is data in `nodecraft-node-contract`:

- `metadata::NODE_METADATA` is the one row per node type: icon key, inspector
  badge, section heading, inspector label, card label, dock label, description
  kind and tool-dock placement/order;
- the UI crates keep the egui code, the icon assets and the key-to-asset map.
  The canvas maps a row's icon key to its SVG; the inspector renders a row's
  badge/section/label/description; the tool dock builds its sections, groups and
  Solid submenu from the row's dock placement;
- value-selected metadata stays data too: `PRIMITIVE_METADATA` owns primitive
  icon keys and dock order, while `BOOLEAN_ICONS` / `BOOLEAN_DESCRIPTIONS` own
  the mode variants. `description()` resolves the Boolean and Group text from
  the node's values;
- fields and defaults already come from this crate, and per-field editor widgets
  and node-specific editors (the sketch editor, topology selection) stay in the
  UI;
- the engine has no egui dependency, and the contract carries no icon bytes.

`metadata::tests` checks every contract descriptor has a row; the engine's
registry test checks every registered `BuiltinNode` id has one too.

Session document policy that needs the session's own `GraphNode`
(`defaults::canonicalize_node`, group/graph traversal, linked-part expansion)
cannot move into the engine without making the engine depend on
`nodecraft-session` — a cycle — so it stays in the session crate. Session
*placement* can and does call the registry.

## Registry and the structural lint

`nodes/mod.rs` owns the one registry table. `nodes::find` is the only
node-type lookup in the engine. Add a node by:

1. creating `nodes/<name>.rs` with `impl BuiltinNode`;
2. adding it to `NODES`;
3. removing its arms from the dispatch files.

`ast-grep/rules/no-node-type-dispatch-in-node-modules.yml` forbids node-type
dispatch patterns inside `nodes/`, so a node cannot grow a branch for another
node's type (or its own: its ids are data in `node_types`). The engine dispatch
files remain listed in `no-node-type-dispatch-outside-registry.yml` until their
last arm is ported; each port removes its arms, and a file with no arms left
comes off that allowed list.

## Current ports

- `primitive` (contract ids `solid`/`primitive`): the whole primitive family is
  one code path selected by `primitiveType`, so one module covers Box and its
  siblings.
- `circularpattern`: exact Part instances, cost, bypass, body layout and the
  repeater role.
- `fillet`: `OpsSession` blend through `ExactNodeTask::Fillet`, warnings,
  revision-token cancellation through the kernel pool, and proofs.
- `translate`: shared placement maths, cost, bypass, body layout and the
  preview-chain transform role.
- `look`: shared colour attribute path, cost, bypass, body layout and source
  body-plan reuse.
- `mirror`: shared reflection maths, combining/non-combining bodies, body
  layout, conditional repeater role and preview-chain transform role.
- `scatter`: contract-clamped count, deterministic placement, cost, generator
  suppression, body layout and repeater role.
- `combine`: concatenating Part, multi-input body layout through the per-port
  `body_layout`, and per-operand error tolerance.
- `chamfer`: the shared `node_task::run_edge_session` blend path with Fillet,
  the no-edges pass-through, the one-body requirement, warnings, cost, bypass
  and the edge-selection body layout.
- `shell`: `OpsSession::shell_body` through `ExactNodeTask::Shell`, one shelled
  body per input body in Part order, positive-thickness validation, the
  `face:N` / `body:B:face:N` opening references, per-body filtering, cost,
  bypass and the source body layout.
- `offsetfaces`: `OpsSession::offset_faces_body` through
  `ExactNodeTask::OffsetFaces`, per-body `face:N` / `body:B:face:N` filtering,
  the no-faces pass-through, cost, and the reviewer-decided shape-modifier
  behaviour: a suppressed node passes its input through and its body layout is
  the source layout.
- `boolean` (contract ids `boolean`/`fuse`): one exact Boolean path for both
  ids, `ExactNodeTask::BooleanLists`, operand suppression pass-through, warnings,
  the both-operands body layout, and the merge-provenance declaration. Union
  fusion stays the engine's `fused_union_leaves` helper: `PartEval::union_run`
  resolves the plan and hands the node leaf Parts only.
- `stepimporter`: `PartEval::step_import`, `Part::from_imported`, every body,
  names/colours, per-body warnings, the import cache, the diagnostics JSON and
  the `IsolatedProcess` scope.
- `exporter`: an `OutputKind::Effect` node. Its `effect` owns file naming,
  overwrite, faceted/exact/Combine-assembly/imported-re-export formats and the
  one writer call; the engine resolves its input and proves bodies first.

The remaining document-only nodes still use the old dispatch arms. Do not port
them without a gate review.
