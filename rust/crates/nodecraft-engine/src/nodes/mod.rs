//! Built-in nodes: one module per node, one registry.
//!
//! A built-in node implements [`BuiltinNode`] once and is listed in the
//! registry below. The engine's dispatch points ask the registry by node-type
//! id instead of matching type strings; a node module therefore knows its own
//! type only (its `node_types` list) and never another node's.
//!
//! The trait mirrors today's engine, not a future value model:
//!
//! - [`PartEval::input_part`] evaluates an upstream port through the exact Part
//!   path (`Eval::evaluate_part`), so values stay `Part`/`SolidBody`/`Brep` and
//!   keep their provenance (`source`/`source_body`), proofs and caches.
//! - [`PartEval::execute_warned`] runs an [`ExactNodeTask`] on the shared
//!   kernel pool through `execute_exact_warned`, so the revision cancellation
//!   token, `OpsSession` (`EachNode` validation) and warnings behave as they do
//!   for a hard-coded engine arm.
//! - Field defaults and choices are read only through `nodecraft-node-contract`
//!   resolvers. The node module never invents a fallback.
//! - `bypass`, the workload estimate, the session's body layout and a node's
//!   repeat/transform role are answered here because the ported nodes already
//!   answer those questions in the dispatch files. Every method reads a
//!   [`NodeView`], the borrowed node shape shared by the protocol and session.
//!
//! Future work is deliberately not on the trait yet (see
//! `agents/claude-notes/2026-11-16-node-platform-plan.md`, stage 1): cost tier
//! (`scalar`/`list`/`heavy`), change kind (`geometry`/`placement-only`) and WASM
//! compatibility are declarations for later stages. Adding them now would give
//! the trait executable policy that no built-in node needs today; the current
//! `exact_weight`/`mesh_bodies` pair is the whole cost vocabulary the engine
//! uses.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use nodecraft_protocol::{Connection, Node};
use nodecraft_solidcore::solidcore::topology::Brep;

use crate::node_task::{ExactNodeTask, TaskResult};
use crate::{Eval, Part, RequestIndex, SketchOutput};

mod boolean;
mod chamfer;
mod circularpattern;
mod combine;
mod exporter;
mod extrude;
mod fillet;
mod loft;
mod look;
mod mirror;
mod offsetfaces;
mod primitive;
mod scatter;
mod shell;
mod sketch;
mod spline;
mod stepimporter;
mod surfaceloft;
mod translate;

pub(crate) use primitive::{primitive_brep, primitive_brep_with_kernel};

/// A borrowed view of one graph node.
///
/// This is the engine-owned node shape: the protocol [`Node`] and the session's
/// `GraphNode` both convert into it without cloning the values. A node module
/// reads only these four fields; engine caches and request routing stay on
/// [`PartEval`].
#[derive(Clone, Copy, Debug)]
pub struct NodeView<'a> {
    pub id: &'a str,
    pub node_type: &'a str,
    pub values: &'a serde_json::Value,
    pub suppressed: bool,
}

impl<'a> NodeView<'a> {
    #[must_use]
    pub fn new(
        id: &'a str,
        node_type: &'a str,
        values: &'a serde_json::Value,
        suppressed: bool,
    ) -> Self {
        Self { id, node_type, values, suppressed }
    }
}

impl<'a> From<&'a Node> for NodeView<'a> {
    fn from(node: &'a Node) -> Self {
        Self::new(&node.id, &node.node_type, &node.values, node.suppressed)
    }
}

/// How a node places its input in the session's drag-preview chain.
///
/// The session owns the affine math ([`crate::nodes`] cannot depend on it); the
/// node declares which transform it is so no session code matches its type id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformRole {
    Translate,
    Mirror,
}

/// What value a node publishes as its exact output.
///
/// Display and overlay policy ask this declaration instead of comparing a
/// node-type string. The default is a Part; Sketch and Spline override it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    Part,
    Sketch,
    /// The node performs an effect (for example writing a STEP file) and does
    /// not publish a geometry value of its own.
    Effect,
}

/// Whether a connected operand still contributes to a node.
///
/// The three states keep "there is no connection" apart from "there is a
/// connection, but its effective output is suppressed". A Boolean must not
/// evaluate the second kind just to discover it contributes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputState {
    /// The port has no connection.
    Unconnected,
    /// The port is connected, but the connection's effective output is
    /// suppressed (a suppressed generator, or a pass-through chain ending in
    /// one).
    Inactive,
    /// The port is connected and its effective output is live.
    Active,
}

/// Whether a node's debug diagnostics need an isolated process.
///
/// The worker asks this declaration instead of matching the importer's type
/// string. `Node` diagnostics are collected in the normal engine pass;
/// `IsolatedProcess` means the worker must not reuse its evaluation cache while
/// collecting them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticsScope {
    Node,
    IsolatedProcess,
}

/// Whether a node keeps the provenance of its operand bodies.
///
/// A plain feature replaces one body in place or maps every source body in
/// order, so a drag preview can still attribute the displayed bodies. A node
/// that merges several operands into one body loses that attribution: the
/// session must fall back to a full evaluation instead of moving only the
/// edited branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeProvenance {
    Preserved,
    LosesProvenance,
}

/// One connected section of a multi-section input, kept in connection order
/// with the upstream node id that produced it.
pub(crate) struct InputSection {
    pub from: String,
    pub part: Result<std::sync::Arc<Part>, String>,
}

/// One imported CAD file as exact bodies plus the import's diagnostics.
///
/// The engine owns path resolution, the display-import cache and its counters;
/// the importer node owns turning the converted bodies into a `Part` (names,
/// colours and per-body warnings are part of that constructor).
pub(crate) struct StepImport {
    pub bodies: Vec<nodecraft_mesh::StepBrepBody>,
    pub warnings: Vec<String>,
    pub diagnostics: nodecraft_mesh::StepDiagnostics,
}

/// A node's debug diagnostics, in the one response shape the app/inspector
/// reads. Today only STEP import reports diagnostics; the enum keeps the map
/// open for other node kinds without changing the JSON object shape.
#[derive(Debug, Clone)]
pub enum NodeDiagnostic {
    Step(nodecraft_mesh::StepDiagnostics),
}

impl NodeDiagnostic {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Step(diagnostics) => diagnostics.to_json(),
        }
    }
}

/// What a node's effect did, when it reports a result.
#[derive(Clone, Debug)]
pub(crate) enum NodeEffect {
    /// The node wrote its output file.
    WroteFile,
}

/// Engine-owned context for a node's side effect.
///
/// Normal preview/validation passes never write files (`run_exporters` is
/// false), and a cancelled revision never writes a stale file. The node receives
/// only the resolved input and the three policy values; it never reaches the
/// request graph or the evaluator caches.
pub(crate) struct EffectContext<'a> {
    run_exporters: bool,
    project_directory: Option<&'a str>,
    token: crate::CancelToken,
    eval: &'a mut dyn PartEval,
}

impl<'a> EffectContext<'a> {
    pub(crate) fn new(
        run_exporters: bool,
        project_directory: Option<&'a str>,
        token: crate::CancelToken,
        eval: &'a mut dyn PartEval,
    ) -> Self {
        Self { run_exporters, project_directory, token, eval }
    }

    pub(crate) fn run_exporters(&self) -> bool {
        self.run_exporters
    }

    pub(crate) fn project_directory(&self) -> Option<&str> {
        self.project_directory
    }

    pub(crate) fn token(&self) -> &crate::CancelToken {
        &self.token
    }

    /// The exporter's input Part and the placement chain the engine resolved
    /// once for this effect.
    pub(crate) fn export_input(&mut self, node: NodeView<'_>) -> ExportInput {
        self.eval.export_input(node)
    }
}

/// The exporter input, resolved by the engine.
pub(crate) struct ExportInput {
    /// The evaluated Part of the exporter's `shape` input.
    pub part: Result<std::sync::Arc<Part>, String>,
    /// Node type of the direct input (kept as data; routing uses `combine`).
    pub source_type: String,
    /// The direct input is a Combine, so the assembly route applies.
    pub combine: bool,
    /// The terminal part chain of the direct input, when the engine can resolve
    /// one (STEP import, Extrude, Boolean, primitive).
    pub chain: Option<ExportChain>,
    /// The chain and exact B-rep of each Combine member, in connection order.
    pub combine_members: Vec<ExportMember>,
    /// Every realized body of the input Part, or the exact-operand error.
    pub breps: Result<Vec<Brep>, String>,
}

impl ExportInput {
    fn missing(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self {
            part: Err(reason.clone()),
            source_type: String::new(),
            combine: false,
            chain: None,
            combine_members: Vec::new(),
            breps: Err(reason),
        }
    }
}

/// A resolved placement chain ending at an exportable source node.
pub(crate) struct ExportChain {
    pub node: Node,
    pub offset: [f64; 3],
    pub rotation: [f64; 3],
    /// The chain ends at a STEP import, so the imported re-export route applies.
    pub imported_step: bool,
}

/// One Combine member: its placement chain and its single exact B-rep.
pub(crate) struct ExportMember {
    pub index: usize,
    pub chain: Option<ExportChain>,
    pub brep: Result<Brep, String>,
}

/// The resolved operands of one fused Union run.
///
/// The engine owns the request-graph walk (`exact_eval::fused_union_leaves`)
/// and hands the node the leaf Parts only, in the order a node-by-node union
/// chain would apply them. The node turns them into one `union_all`; it never
/// sees the flat request, the intermediate ids, or why the run was foldable.
pub(crate) struct UnionRun {
    operands: Vec<Result<std::sync::Arc<Part>, String>>,
}

impl UnionRun {
    /// Take the leaf operands, in chain order. A failed leaf keeps its error so
    /// the node can propagate the first one, as the exact arm did.
    pub(crate) fn into_operands(self) -> Vec<Result<std::sync::Arc<Part>, String>> {
        self.operands
    }
}

/// A single-profile input, classified by the shared section validation.
pub(crate) enum ProfileInput {
    Sketches(Vec<std::sync::Arc<SketchOutput>>),
    Body(std::sync::Arc<Part>),
}

/// The validated shape of one Part used as a section or profile.
#[derive(Debug)]
enum SectionValue {
    Sketches(Vec<std::sync::Arc<SketchOutput>>),
    Body,
}

/// The one validation every sketch/section/profile input shares.
///
/// A Part is either bodies or sketches, never both, and never empty. The
/// exact-one-sketch rule is applied by the section consumers (`input_sketch`
/// and `input_sections`); a profile (Extrude) intentionally accepts several
/// sketches from a Combine and turns each into its own body.
fn validated_section(part: &Part, label: &str) -> Result<SectionValue, String> {
    let has_bodies = !part.bodies.is_empty();
    let has_sketches = !part.sketches.is_empty();
    if !has_bodies && !has_sketches {
        return Err(format!("{label} has no geometry"));
    }
    if has_bodies && has_sketches {
        return Err(format!("{label} contains both bodies and sketches; connect one or the other"));
    }
    if has_bodies {
        return Ok(SectionValue::Body);
    }
    Ok(SectionValue::Sketches(
        part.realized_sketches().into_iter().map(std::sync::Arc::new).collect(),
    ))
}

/// The one sketch of a validated section, or `None` when the section is a body.
pub(super) fn section_sketch(part: &Part) -> Option<std::sync::Arc<SketchOutput>> {
    if !part.bodies.is_empty() || part.sketches.len() != 1 {
        return None;
    }
    Some(std::sync::Arc::new(part.sketches[0].realized()))
}

fn exactly_one_sketch(
    sketches: Vec<std::sync::Arc<SketchOutput>>,
    label: &str,
) -> Result<std::sync::Arc<SketchOutput>, String> {
    if sketches.len() != 1 {
        return Err(format!(
            "{label} has {} sketches; a section must be exactly one sketch",
            sketches.len()
        ));
    }
    Ok(sketches.into_iter().next().expect("one sketch"))
}

/// One built-in node implementation.
///
/// The trait is object-safe: the registry stores `&'static dyn BuiltinNode`, so
/// a node implementation is reached only through the registry. Methods are
/// added only when a ported node needs them today.
pub(crate) trait BuiltinNode: Sync {
    /// Every node-type id this implementation answers for. The primitive family
    /// answers both its contract ids (`solid` and `primitive`); most nodes
    /// answer one.
    fn node_types(&self) -> &'static [&'static str];

    /// Normal exact evaluation: the node's immutable [`Part`] value.
    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String>;

    /// Run this node's side effect, if it has one and the engine asks for it.
    ///
    /// The default is no effect. The context carries the write policy, the
    /// project directory and the current revision token; the engine resolves
    /// the exporter input once through [`PartEval::export_input`].
    fn effect(
        &self,
        _node: NodeView<'_>,
        _ctx: &mut EffectContext<'_>,
    ) -> Result<Option<NodeEffect>, String> {
        Ok(None)
    }

    /// Relative exact-work estimate, the engine's current cost vocabulary.
    fn exact_weight(&self, node: NodeView<'_>) -> usize;

    /// The value kind this node publishes. Display/overlay policy reads this
    /// declaration instead of comparing the node-type string.
    fn output_kind(&self) -> OutputKind {
        OutputKind::Part
    }

    /// Where this node's debug diagnostics are collected. The default is the
    /// normal engine pass; the importer overrides it.
    fn diagnostics_scope(&self) -> DiagnosticsScope {
        DiagnosticsScope::Node
    }

    /// Whether this node can terminate an export placement chain.
    ///
    /// The exporter resolves its input through the same declaration instead of
    /// matching a list of source type strings. The default is false; sources
    /// that expose one exact Part (primitive, Boolean, Extrude, STEP import)
    /// override it.
    fn export_terminal(&self) -> bool {
        false
    }

    /// Rough display-body estimate, used by the mesh pool.
    fn mesh_bodies(&self, node: NodeView<'_>) -> usize {
        let _ = node;
        0
    }

    /// The input a suppressed node passes through. `None` means the node is a
    /// generator and suppression turns it off.
    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String>;

    /// Node ids of the display bodies this node's Part produces, in Part body
    /// order, for the session's drag preview.
    ///
    /// `inputs` carries every connected input port with its upstream layout, in
    /// connection order, so a multi-input node (Combine) can express the
    /// flattened Part body order and a one-input node can repeat or transform
    /// its source layout. `None` means the engine cannot reproduce that layout
    /// exactly.
    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        let _ = (node, inputs);
        None
    }

    /// Whether this node turns one input body into several displayed copies, so
    /// a drag preview anchors at copy 0 (the untouched source body). Mirror is
    /// conditional on its `combine` value, so the view is required.
    fn repeats_input_bodies(&self, node: NodeView<'_>) -> bool {
        let _ = node;
        false
    }

    /// Which transform this node is for the session's drag-preview chain.
    /// `None` for everything that is not a placement.
    fn transform_role(&self) -> Option<TransformRole> {
        None
    }

    /// Whether this node keeps the provenance of the operand bodies it merges.
    ///
    /// `affected` is the set of node ids downstream of the edited branch (or
    /// just the selected id when asking about the target itself). The default
    /// preserves provenance; a merging node reports `LosesProvenance` when one
    /// of its operands is outside that set.
    fn merge_provenance(
        &self,
        node: NodeView<'_>,
        connections: &[Connection],
        affected: &HashSet<String>,
    ) -> MergeProvenance {
        let _ = (node, connections, affected);
        MergeProvenance::Preserved
    }
}

/// The layout of one connected input port, when the node has one.
pub(crate) fn input_layout<'a>(
    inputs: &'a [(&str, Vec<String>)],
    port: &str,
) -> Option<&'a [String]> {
    inputs.iter().find(|(name, _)| *name == port).map(|(_, layout)| layout.as_slice())
}

/// Whether an edge-feature node has a saved edge selection.
///
/// A fresh Fillet/Chamfer has no edges yet and passes its input through; only
/// a non-empty `edges` array runs the blend.
pub(crate) fn has_edges(values: &serde_json::Value) -> bool {
    values.get("edges").and_then(serde_json::Value::as_array).is_some_and(|edges| !edges.is_empty())
}

/// An edge blend's (Fillet's, Chamfer's) result as the node's Part: the body as the task made
/// it, so the next node's operand is its kernel body with every round's exact blend definition.
/// Never through a B-rep copy: `SolidBody::into_brep` falls back to an evaluation copy, which
/// carries no definitions, whenever the B-rep is shared.
pub(crate) fn edge_blend_part(
    output: crate::node_task::NodeOutput,
    node_name: &str,
) -> Result<Part, String> {
    match output {
        crate::node_task::NodeOutput::Shape(body) => Ok(Part::from_bodies([body])),
        _ => Err(format!("{node_name} task returned no shape")),
    }
}

/// What a node may ask the exact Part evaluator for.
///
/// This is intentionally narrower than `Eval`: a node cannot reach the caches,
/// the request graph or another node's implementation. Everything it needs is a
/// port evaluated through the same Part path the engine already uses.
pub(crate) trait PartEval {
    /// Evaluate the first connected port among `ports`, or `None` when none is
    /// connected. Evaluator errors (including cancellation) propagate.
    fn input_part(&mut self, ports: &[&str]) -> Result<Option<std::sync::Arc<Part>>, String>;

    /// Import the CAD file named by `node`, through the engine's one import
    /// path. The engine keeps path/key resolution, `bypass_cache`, the display
    /// tolerance check, cache counters and the file identity; the node turns
    /// the returned bodies into a `Part`.
    /// Resolve and import one CAD source. `None` means the importer has no
    /// stored path yet: it is not configured and produces an empty Part, not
    /// an error.
    fn step_import(&mut self, node: NodeView<'_>) -> Result<Option<StepImport>, String>;

    /// Resolve the exporter's input Part and placement chain once, for the
    /// node's effect. The node never walks the request graph itself.
    fn export_input(&mut self, node: NodeView<'_>) -> ExportInput;

    /// Whether `port` is unconnected, connected to a suppressed effective
    /// output, or connected to a live one. This is the one existing suppression
    /// walk (`bypass::effective_output_suppressed`): a suppressed pass-through
    /// feature stays active while its source is active, and a suppressed
    /// generator does not.
    fn input_state(&mut self, port: &str) -> InputState;

    /// Resolve the shared Union-run plan for `node`, if this node is the top of
    /// a foldable run of Union nodes.
    ///
    /// The engine walks the request graph and evaluates the leaves; the node
    /// receives only their Parts. `None` means the node must evaluate its A/B
    /// operands normally (not a Union, reverse difference, or fewer than three
    /// leaves). The engine's `union_run_intermediates` keeps the folded
    /// intermediates out of the exact loop, so this returns the same leaves the
    /// node-by-node chain would union.
    fn union_run(&mut self, node: NodeView<'_>) -> Option<UnionRun>;

    /// Evaluate every connection to `port`, in connection order, keeping each
    /// operand's result separate. Combine tolerates one failed operand while
    /// keeping the others; a single `Result` would stop at the first failure.
    fn input_parts(&mut self, port: &str) -> Vec<Result<std::sync::Arc<Part>, String>>;

    /// The one sketch connected to `port`, or `None` when the port is
    /// unconnected or its Part is a body. A Part carrying several sketches (or
    /// both bodies and sketches) is an error: a section is exactly one sketch.
    fn input_sketch(&mut self, port: &str) -> Result<Option<std::sync::Arc<SketchOutput>>, String>;

    /// Every connection to `port`, in connection order, source-labelled. Each
    /// section is validated as exactly one sketch or a body, never mixed; a
    /// failed section keeps its source label and error.
    fn input_sections(&mut self, port: &str) -> Vec<InputSection>;

    /// The first connected port among `ports`, classified as a profile: one or
    /// more sketches, or a body. Mixed and empty profiles are errors.
    fn input_profile(&mut self, ports: &[&str]) -> Result<ProfileInput, String>;

    /// Publish a node's sketch output for downstream overlay lookup. This is
    /// the one path a node records a sketch, replacing direct map inserts.
    fn publish_sketch(&mut self, node: NodeView<'_>, sketch: std::sync::Arc<SketchOutput>);

    /// The shared exact B-rep cache for a generating node: a prewarmed/brep
    /// cached value if present, otherwise the node's `Primitive` task. The view
    /// identifies the node; the engine looks up its request entry for the cache
    /// key.
    fn cached_brep(&mut self, node: NodeView<'_>) -> Result<std::sync::Arc<Brep>, String>;

    /// Run an exact task on the shared kernel pool through
    /// `execute_exact_warned` (revision token, `EachNode` validation, warnings).
    fn execute_warned(&mut self, task: ExactNodeTask) -> Result<TaskResult, String>;

    /// Append user-facing warnings to this node's entry.
    fn warn(&mut self, warnings: Vec<String>);

    /// Record debug diagnostics for this node. The map and its JSON shape stay
    /// the one the app/inspector already reads.
    fn record_diagnostics(&mut self, diagnostics: Vec<NodeDiagnostic>);

    /// Record a non-fatal error on this node. Combine keeps the operands that
    /// did evaluate but still reports the failed ones, as it did before the
    /// port.
    fn record_error(&mut self, message: String);
}

/// One source file for an import: either the user's path or a temp file that
/// holds a decoded ingested copy. The temp file is removed on drop.
enum SourceFile {
    Existing(String),
    Temporary(std::path::PathBuf),
}

impl SourceFile {
    fn path(&self) -> &str {
        match self {
            Self::Existing(path) => path,
            Self::Temporary(path) => path.to_str().unwrap_or_default(),
        }
    }
}

impl Drop for SourceFile {
    fn drop(&mut self) {
        if let Self::Temporary(path) = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn write_temp_source(bytes: &[u8]) -> Result<std::path::PathBuf, String> {
    use std::io::Write as _;
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("nodecraft-ingest-{}-{id}.step", std::process::id()));
    let mut file =
        std::fs::File::create(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    file.write_all(bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    file.sync_all().map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(path)
}

/// The engine-backed [`PartEval`] handed to a node's `evaluate`.
pub(crate) struct EnginePartEval<'a, 'req, 'prog> {
    eval: &'a mut Eval<'req, 'prog>,
    node_id: String,
}

impl<'a, 'req, 'prog> EnginePartEval<'a, 'req, 'prog> {
    pub(crate) fn new(eval: &'a mut Eval<'req, 'prog>, node: &Node) -> Self {
        Self { eval, node_id: node.id.clone() }
    }
}

impl PartEval for EnginePartEval<'_, '_, '_> {
    fn input_part(&mut self, ports: &[&str]) -> Result<Option<std::sync::Arc<Part>>, String> {
        let Some(link) = self.eval.index.first_incoming_any(&self.node_id, ports) else {
            return Ok(None);
        };
        let source = link.from.clone();
        self.eval.evaluate_part(&source).map(Some)
    }

    fn step_import(&mut self, node: NodeView<'_>) -> Result<Option<StepImport>, String> {
        let ingest = nodecraft_node_contract::resolve_bool(
            "stepimporter",
            node.values,
            crate::import_io::INGEST_KEY,
        )
        .unwrap_or(false);
        let document_directory = self.eval.imports.document_directory.clone();
        let Some(resolved) = crate::import_io::resolve_import(
            node.values,
            document_directory.as_deref(),
            &self.eval.imports.ingested_sources,
            ingest,
        )?
        else {
            return Ok(None);
        };
        // A missing source that falls back to the embedded copy is imported
        // from a private temp file; the guard removes it when this call ends.
        let source = match &resolved.path {
            Some(path) => SourceFile::Existing(path.display().to_string()),
            None => {
                let bytes = resolved.bytes.as_deref().ok_or("STEP import resolved no bytes")?;
                SourceFile::Temporary(write_temp_source(bytes)?)
            }
        };
        let path = source.path();
        let default_quality =
            (nodecraft_mesh::display_tolerance() - nodecraft_mesh::DEFAULT_DISPLAY_TOLERANCE).abs()
                < 1.0e-9;
        let options = if default_quality { "display:default" } else { "display:custom" };
        let key = crate::import_cache_key(&resolved.content_hash, options);
        let full =
            self.eval.index.node(node.id).ok_or("STEP import node missing from the request")?;
        let fresh = crate::cache::node_cache_is_fresh(full);
        let (diagnostics, breps) = if !self.eval.bypass_cache
            && default_quality
            && fresh
            && let Some(import) = self.eval.cache.import(&key)
        {
            // The display mesh comes from the cache; the exact route decodes
            // the file once for its kernel bodies.
            self.eval.cache.note_import_hit();
            let diagnostics = import.diagnostics.clone();
            let breps = nodecraft_mesh::import_step_breps(path)?;
            (diagnostics, breps)
        } else {
            // One decode serves both the display mesh and the exact bodies.
            let combined = nodecraft_mesh::import_step_with_breps(path)?;
            let diagnostics = combined.display.diagnostics.clone();
            if default_quality {
                self.eval.cache.insert_import(key, std::sync::Arc::new(combined.display));
            }
            (diagnostics, combined.breps)
        };
        let mut warnings = breps.warnings;
        if let Some(warning) = resolved.warning {
            warnings.push(warning);
        }
        if let Some(warning) =
            node.values.get(crate::import_io::PATH_WARNING_KEY).and_then(serde_json::Value::as_str)
            && !warning.is_empty()
        {
            warnings.push(warning.to_owned());
        }
        Ok(Some(StepImport { bodies: breps.bodies, warnings, diagnostics }))
    }

    fn export_input(&mut self, node: NodeView<'_>) -> ExportInput {
        let Some(link) = self.eval.index.first_incoming(node.id, "shape") else {
            return ExportInput::missing("Export node has no shape input.");
        };
        let source = link.from.clone();
        let source_type =
            self.eval.index.node(&source).map(|entry| entry.node_type.clone()).unwrap_or_default();
        let combine = crate::is_combine_node(
            self.eval.index.node(&source).expect("export source is in the request"),
        );
        let chain =
            crate::resolve_part_chain(&self.eval.index, &source).map(|(node, offset, rotation)| {
                ExportChain {
                    node: node.clone(),
                    offset,
                    rotation,
                    imported_step: crate::is_stepimporter_node(node),
                }
            });
        let part = self.eval.evaluate_part(&source);
        let breps = crate::exact_operand_breps(self.eval, &source);
        let member_sources: Vec<String> = if combine {
            self.eval
                .index
                .incoming(&source)
                .filter(|link| link.input == "shape")
                .map(|link| link.from.clone())
                .collect()
        } else {
            Vec::new()
        };
        let combine_members = member_sources
            .into_iter()
            .enumerate()
            .map(|(index, member)| {
                let chain = crate::resolve_part_chain(&self.eval.index, &member).map(
                    |(node, offset, rotation)| ExportChain {
                        node: node.clone(),
                        offset,
                        rotation,
                        imported_step: crate::is_stepimporter_node(node),
                    },
                );
                // The placement is carried separately, so the B-rep comes from
                // the chain's source node, not the already-placed member.
                let brep = match &chain {
                    Some(chain) => crate::exact_operand_brep(self.eval, &chain.node.id),
                    None => Err("assembly export requires STEP import or Extrude parts".into()),
                };
                ExportMember { index, chain, brep }
            })
            .collect();
        ExportInput { part, source_type, combine, chain, combine_members, breps }
    }

    fn input_state(&mut self, port: &str) -> InputState {
        let Some(link) = self.eval.index.first_incoming(&self.node_id, port) else {
            return InputState::Unconnected;
        };
        if crate::bypass::effective_output_suppressed(&self.eval.index, &link.from) {
            InputState::Inactive
        } else {
            InputState::Active
        }
    }

    fn union_run(&mut self, node: NodeView<'_>) -> Option<UnionRun> {
        if !matches!(
            crate::boolean_mode(node.values),
            Ok((nodecraft_bool::BooleanOp::Union, false))
        ) {
            return None;
        }
        if self.input_state("a") != InputState::Active
            || self.input_state("b") != InputState::Active
        {
            return None;
        }
        let leaves = {
            let index = &self.eval.index;
            let a = index.first_incoming(node.id, "a")?.from.clone();
            let b = index.first_incoming(node.id, "b")?.from.clone();
            crate::exact_eval::fused_union_leaves(index, node.id, &a, &b)?
        };
        let operands = leaves.into_iter().map(|leaf| self.eval.evaluate_part(&leaf)).collect();
        Some(UnionRun { operands })
    }

    fn cached_brep(&mut self, node: NodeView<'_>) -> Result<std::sync::Arc<Brep>, String> {
        let full = self.eval.index.node(node.id).cloned().ok_or("node missing from the request")?;
        self.eval.cached_primitive_brep_arc(&full)
    }

    fn input_parts(&mut self, port: &str) -> Vec<Result<std::sync::Arc<Part>, String>> {
        let sources: Vec<String> = self
            .eval
            .index
            .incoming(&self.node_id)
            .filter(|link| link.input == port)
            .map(|link| link.from.clone())
            .collect();
        sources.into_iter().map(|source| self.eval.evaluate_part(&source)).collect()
    }

    fn input_sketch(&mut self, port: &str) -> Result<Option<std::sync::Arc<SketchOutput>>, String> {
        let Some(source) =
            self.eval.index.first_incoming(&self.node_id, port).map(|link| link.from.clone())
        else {
            return Ok(None);
        };
        let part = self.eval.evaluate_part(&source)?;
        match validated_section(&part, &format!("Sketch input {source}"))? {
            SectionValue::Sketches(sketches) => {
                exactly_one_sketch(sketches, &format!("Sketch input {source}")).map(Some)
            }
            SectionValue::Body => Ok(None),
        }
    }

    fn input_sections(&mut self, port: &str) -> Vec<InputSection> {
        let sources: Vec<String> = self
            .eval
            .index
            .incoming(&self.node_id)
            .filter(|link| link.input == port)
            .map(|link| link.from.clone())
            .collect();
        sources
            .into_iter()
            .map(|from| {
                let part =
                    self.eval.evaluate_part(&from).and_then(|part| {
                        match validated_section(&part, &format!("Section {from}"))? {
                            SectionValue::Sketches(sketches) => {
                                exactly_one_sketch(sketches, &format!("Section {from}"))?;
                                Ok(part)
                            }
                            SectionValue::Body => Ok(part),
                        }
                    });
                InputSection { from, part }
            })
            .collect()
    }

    fn input_profile(&mut self, ports: &[&str]) -> Result<ProfileInput, String> {
        let Some(source) =
            self.eval.index.first_incoming_any(&self.node_id, ports).map(|link| link.from.clone())
        else {
            return Err("Profile input has no connected source".into());
        };
        let part = self.eval.evaluate_part(&source)?;
        match validated_section(&part, &format!("Profile input {source}"))? {
            SectionValue::Sketches(sketches) => Ok(ProfileInput::Sketches(sketches)),
            SectionValue::Body => Ok(ProfileInput::Body(part)),
        }
    }

    fn publish_sketch(&mut self, node: NodeView<'_>, sketch: std::sync::Arc<SketchOutput>) {
        self.eval.publish_sketch_output(node.id, sketch);
    }

    fn execute_warned(&mut self, task: ExactNodeTask) -> Result<TaskResult, String> {
        crate::node_task::execute_exact_warned(task, &self.node_id, crate::fnv_seed(&self.node_id))
    }

    fn warn(&mut self, warnings: Vec<String>) {
        if !warnings.is_empty() {
            self.eval.warnings.entry(self.node_id.clone()).or_default().extend(warnings);
        }
    }

    fn record_diagnostics(&mut self, diagnostics: Vec<NodeDiagnostic>) {
        if !diagnostics.is_empty() {
            self.eval.diagnostics.entry(self.node_id.clone()).or_default().extend(diagnostics);
        }
    }

    fn record_error(&mut self, message: String) {
        self.eval.errors.insert(self.node_id.clone(), message);
    }
}

static REGISTRY: OnceLock<HashMap<&'static str, &'static dyn BuiltinNode>> = OnceLock::new();

static NODES: &[&dyn BuiltinNode] = &[
    &primitive::Primitive,
    &boolean::Boolean,
    &circularpattern::CircularPattern,
    &chamfer::Chamfer,
    &fillet::Fillet,
    &translate::Translate,
    &look::Look,
    &mirror::Mirror,
    &offsetfaces::OffsetFaces,
    &scatter::Scatter,
    &shell::Shell,
    &sketch::SketchNode,
    &spline::Spline,
    &stepimporter::StepImporter,
    &combine::Combine,
    &extrude::Extrude,
    &exporter::Exporter,
    &loft::Loft,
    &surfaceloft::SurfaceLoft,
];

fn registry() -> &'static HashMap<&'static str, &'static dyn BuiltinNode> {
    REGISTRY.get_or_init(|| {
        let mut by_type = HashMap::new();
        for node in NODES {
            for node_type in node.node_types() {
                let previous = by_type.insert(*node_type, *node);
                assert!(
                    previous.is_none(),
                    "duplicate built-in node registry entry for {node_type:?}",
                );
            }
        }
        by_type
    })
}

/// The implementation registered for `node_type`, if it has been ported.
#[must_use]
pub(crate) fn find(node_type: &str) -> Option<&'static dyn BuiltinNode> {
    registry().get(node_type).copied()
}

/// Whether a node type is served by a [`BuiltinNode`] yet.
#[must_use]
pub fn handles(node_type: &str) -> bool {
    find(node_type).is_some()
}

/// Display body layout for the session's drag preview, when the node is ported.
#[must_use]
pub fn body_layout(node: NodeView<'_>, inputs: &[(&str, Vec<String>)]) -> Option<Vec<String>> {
    find(node.node_type)?.body_layout(node, inputs)
}

/// The value kind the registry's implementation publishes. Unported nodes
/// default to Part; a node that is not registered can only reach this query
/// through a caller's own fallback.
#[must_use]
pub fn output_kind(node: NodeView<'_>) -> OutputKind {
    if let Some(builtin) = find(node.node_type) {
        return builtin.output_kind();
    }
    // An unported node still answers the display/overlay question from the
    // contract descriptor until its module is registered.
    match nodecraft_node_contract::descriptor(node.node_type).map(|descriptor| descriptor.output) {
        Some(nodecraft_node_contract::ValueKind::Sketch) => OutputKind::Sketch,
        _ => OutputKind::Part,
    }
}

/// The diagnostics scope of a node type. Unported nodes use the normal pass;
/// a node overrides this through the registry when it needs isolation.
#[must_use]
pub fn diagnostics_scope(node_type: &str) -> DiagnosticsScope {
    find(node_type).map_or(DiagnosticsScope::Node, |builtin| builtin.diagnostics_scope())
}

/// Whether a node type can terminate an export placement chain.
#[must_use]
pub fn export_terminal(node_type: &str) -> bool {
    find(node_type).is_some_and(|builtin| builtin.export_terminal())
}

/// The transform role the session's drag-preview chain should apply, if any.
#[must_use]
pub fn transform_role(node: NodeView<'_>) -> Option<TransformRole> {
    find(node.node_type)?.transform_role()
}

/// Whether the node repeats its input bodies (drag preview anchors at copy 0).
#[must_use]
pub fn repeats_input_bodies(node: NodeView<'_>) -> bool {
    find(node.node_type).is_some_and(|builtin| builtin.repeats_input_bodies(node))
}

/// Whether the node keeps operand provenance when it builds its Part.
///
/// The session's preview scope calls this instead of knowing a Boolean's port
/// names. An unported node preserves provenance until its module declares
/// otherwise.
#[must_use]
pub fn merge_provenance(
    node: NodeView<'_>,
    connections: &[Connection],
    affected: &HashSet<String>,
) -> MergeProvenance {
    find(node.node_type).map_or(MergeProvenance::Preserved, |builtin| {
        builtin.merge_provenance(node, connections, affected)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_protocol::Request;
    use serde_json::json;

    fn sketch() -> crate::sketch::Sketch {
        let mut sketch = crate::sketch::Sketch::new();
        sketch.add_circle(solidcore_record::Vec2::new(0.0, 0.0), 10.0, None, None).expect("circle");
        sketch
    }

    fn sketch_part() -> Part {
        Part::from_sketch(std::sync::Arc::new(SketchOutput::new(sketch())))
    }

    fn body_part() -> Part {
        let node = Node {
            id: "box".into(),
            node_type: "primitive".into(),
            values: json!({"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}),
            suppressed: false,
            cache: None,
        };
        Part::from_brep(primitive_brep(&node).expect("box"))
    }

    fn mixed_part() -> Part {
        let mut part = body_part();
        part.sketches = sketch_part().sketches;
        part
    }

    fn node(id: &str, node_type: &str, values: serde_json::Value) -> Node {
        Node { id: id.into(), node_type: node_type.into(), values, suppressed: false, cache: None }
    }

    /// `step_import` is the engine-owned import route: it returns every body,
    /// the per-body warnings and the same diagnostics the display reader
    /// produced. The importer node only builds the `Part`.
    #[test]
    fn step_import_returns_every_body_and_the_diagnostics() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../samples/rust-export.step");
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["s"], "meshTargets": [],
            "nodes": [{"id": "s", "type": "stepimporter", "values": {"path": path.to_string_lossy()}}],
            "connections": []
        }))
        .expect("import request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes[0].clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        let import = ctx.step_import(NodeView::from(&target)).expect("import").expect("configured");
        assert_eq!(import.bodies.len(), 6, "every body in the file is returned");
        assert_eq!(import.diagnostics.bodies, 6);
        assert!(import.diagnostics.mesh_triangles > 0);
    }

    /// `record_diagnostics` keeps the JSON object the inspector already reads:
    /// one node id maps to the STEP diagnostics object, with no renamed keys.
    #[test]
    fn record_diagnostics_keeps_the_existing_json_shape() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [{"id": "n", "type": "primitive", "values": {"primitiveType": "box"}}],
            "connections": []
        }))
        .expect("diagnostics request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes[0].clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        ctx.record_diagnostics(vec![NodeDiagnostic::Step(nodecraft_mesh::StepDiagnostics {
            kind: "solidcore-brep".into(),
            bodies: 3,
            brep_faces: 18,
            ..nodecraft_mesh::StepDiagnostics::default()
        })]);
        drop(ctx);
        let recorded = eval.diagnostics.get("n").expect("diagnostics recorded");
        assert_eq!(recorded.len(), 1);
        let json = recorded[0].to_json();
        assert_eq!(json["kind"], "solidcore-brep");
        assert_eq!(json["bodies"], 3);
        assert_eq!(json["brepFaces"], 18, "the camelCase response key is unchanged");
    }

    /// `export_input` resolves the input Part and the placement chain once in
    /// the engine, so the exporter effect never walks the graph itself.
    #[test]
    fn export_input_resolves_the_part_and_placement_chain() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "p", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "t", "type": "translate", "values": {"x": 5.0}},
                {"id": "e", "type": "exporter", "values": {"fileName": "out"}}
            ],
            "connections": [
                {"from": "p", "to": "t", "in": "shape", "out": "shape"},
                {"from": "t", "to": "e", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("export input request");
        let mut eval = crate::Eval::new(&request, None);
        let exporter = request.nodes.iter().find(|node| node.id == "e").expect("exporter").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &exporter);
        let input = ctx.export_input(NodeView::from(&exporter));
        assert_eq!(input.source_type, "translate");
        assert_eq!(input.part.expect("input part").bodies.len(), 1);
        assert_eq!(input.breps.expect("exact operands").len(), 1);
        let chain = input.chain.expect("translate chain resolves to its primitive");
        assert_eq!(chain.node.id, "p");
        assert_eq!(chain.offset, [5.0, 0.0, 0.0]);
        assert!(input.combine_members.is_empty());
    }

    /// A Combine exporter input resolves one member per connection, in order.
    #[test]
    fn export_input_resolves_combine_members_in_order() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["e"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "c", "type": "combine", "values": {}},
                {"id": "e", "type": "exporter", "values": {"fileName": "out"}}
            ],
            "connections": [
                {"from": "a", "to": "c", "in": "shape", "out": "shape"},
                {"from": "b", "to": "c", "in": "shape", "out": "shape"},
                {"from": "c", "to": "e", "in": "shape", "out": "shape"}
            ]
        }))
        .expect("combine export input request");
        let mut eval = crate::Eval::new(&request, None);
        let exporter = request.nodes.iter().find(|node| node.id == "e").expect("exporter").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &exporter);
        let input = ctx.export_input(NodeView::from(&exporter));
        assert_eq!(input.source_type, "combine");
        assert_eq!(input.combine_members.len(), 2);
        assert_eq!(input.combine_members[0].index, 0);
        assert_eq!(input.combine_members[1].index, 1);
        assert!(input.combine_members.iter().all(|member| member.brep.is_ok()));
    }

    /// The shared validation keeps the existing error wording: empty, mixed,
    /// and a multi-sketch section are each refused rather than guessed.
    #[test]
    fn shared_validation_errors_keep_their_wording() {
        let empty = Part::empty();
        assert_eq!(
            validated_section(&empty, "Section s").unwrap_err(),
            "Section s has no geometry",
        );
        assert_eq!(
            validated_section(&mixed_part(), "Section s").unwrap_err(),
            "Section s contains both bodies and sketches; connect one or the other",
        );

        let mut two = sketch_part();
        two.sketches.push(two.sketches[0].clone());
        assert_eq!(
            exactly_one_sketch(
                match validated_section(&two, "Section s").expect("sketches") {
                    SectionValue::Sketches(sketches) => sketches,
                    SectionValue::Body => panic!("two sketches are not a body"),
                },
                "Section s",
            )
            .unwrap_err(),
            "Section s has 2 sketches; a section must be exactly one sketch",
        );
    }

    /// `input_sections` keeps connection order and each upstream id, even when
    /// one section fails to evaluate.
    #[test]
    fn input_sections_preserve_order_and_source_labels() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [
                {"id": "s1", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "s2", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}},
                {"id": "bad", "type": "stepimporter", "values": {"path": "/nonexistent/none.step"}},
                {"id": "n", "type": "surfaceloft", "values": {}}
            ],
            "connections": [
                {"from": "s1", "to": "n", "in": "sections", "out": "sketch"},
                {"from": "b", "to": "n", "in": "sections", "out": "shape"},
                {"from": "bad", "to": "n", "in": "sections", "out": "shape"},
                {"from": "s2", "to": "n", "in": "sections", "out": "sketch"}
            ]
        }))
        .expect("sections request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "n").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        let sections = ctx.input_sections("sections");
        assert_eq!(
            sections.iter().map(|section| section.from.as_str()).collect::<Vec<_>>(),
            ["s1", "b", "bad", "s2"],
        );
        assert!(sections[0].part.is_ok(), "a sketch section is one section");
        assert!(sections[1].part.is_ok(), "a body section is one section");
        assert!(sections[2].part.is_err(), "a failed section keeps its error");
        assert!(sections[3].part.is_ok());
    }

    /// `input_sketch` accepts exactly one sketch, treats a body as no sketch,
    /// and refuses a Part carrying several sketches.
    #[test]
    fn input_sketch_classifies_one_sketch_or_a_body() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [
                {"id": "s", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "s2", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}},
                {"id": "c", "type": "combine", "values": {}},
                {"id": "n", "type": "loft", "values": {}}
            ],
            "connections": [
                {"from": "s", "to": "n", "in": "bottom", "out": "sketch"},
                {"from": "s2", "to": "c", "in": "shape", "out": "sketch"},
                {"from": "s2", "to": "c", "in": "shape", "out": "sketch"}
            ]
        }))
        .expect("sketch input request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "n").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        let bottom = ctx.input_sketch("bottom").expect("one sketch").expect("a sketch");
        assert_eq!(bottom.profiles.len(), 1);
        assert!(ctx.input_sketch("top").expect("unconnected").is_none());

        let body_request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "n", "type": "loft", "values": {}}
            ],
            "connections": [{"from": "b", "to": "n", "in": "bottom", "out": "shape"}]
        }))
        .expect("body input request");
        let mut eval = crate::Eval::new(&body_request, None);
        let target = body_request.nodes.iter().find(|node| node.id == "n").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        assert!(ctx.input_sketch("bottom").expect("body is not a sketch").is_none());
    }

    /// `input_state` is the existing effective-suppression walk in three
    /// states: unconnected, connected-but-inactive, and live. A suppressed
    /// pass-through around a live source stays active, but a suppressed
    /// generator does not.
    #[test]
    fn input_state_distinguishes_unconnected_inactive_and_active() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [
                {"id": "dead", "type": "primitive", "suppressed": true, "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "values": {"x": 5}},
                {"id": "live", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "n", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [
                {"from": "dead", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "n", "in": "a", "out": "shape"},
                {"from": "live", "to": "n", "in": "b", "out": "shape"}
            ]
        }))
        .expect("state request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "n").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        assert_eq!(
            ctx.input_state("a"),
            InputState::Inactive,
            "a suppressed generator is inactive through a translate",
        );
        assert_eq!(ctx.input_state("b"), InputState::Active, "a live operand is active");
        assert_eq!(
            ctx.input_state("shape"),
            InputState::Unconnected,
            "an unconnected port is distinct from an inactive one",
        );
    }

    /// A suppressed pass-through whose source is live is Active, not Inactive:
    /// the state walk follows the effective output, not the node's own flag.
    #[test]
    fn input_state_follows_a_pass_through_to_its_live_source() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["n"], "meshTargets": [],
            "nodes": [
                {"id": "live", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "suppressed": true, "values": {"x": 5}},
                {"id": "n", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [
                {"from": "live", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "n", "in": "a", "out": "shape"}
            ]
        }))
        .expect("pass-through request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "n").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        assert_eq!(ctx.input_state("a"), InputState::Active);
    }

    /// The engine resolves a Union run into leaf Parts; the node gets no ids,
    /// no ports and no request graph. A non-Union node gets no plan.
    #[test]
    fn union_run_returns_the_fused_leaf_operands() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["u3"], "meshTargets": [],
            "nodes": [
                {"id": "a", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "c", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "d", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "u1", "type": "boolean", "values": {"mode": "union"}},
                {"id": "u2", "type": "boolean", "values": {"mode": "union"}},
                {"id": "u3", "type": "boolean", "values": {"mode": "union"}}
            ],
            "connections": [
                {"from": "a", "to": "u1", "in": "a", "out": "shape"},
                {"from": "b", "to": "u1", "in": "b", "out": "shape"},
                {"from": "u1", "to": "u2", "in": "a", "out": "shape"},
                {"from": "c", "to": "u2", "in": "b", "out": "shape"},
                {"from": "u2", "to": "u3", "in": "a", "out": "shape"},
                {"from": "d", "to": "u3", "in": "b", "out": "shape"}
            ]
        }))
        .expect("union run request");
        let mut eval = crate::Eval::new(&request, None);
        let top = request.nodes.iter().find(|node| node.id == "u3").expect("top").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &top);
        let run = ctx.union_run(NodeView::from(&top)).expect("u3 is a fused run");
        let operands = run.into_operands();
        drop(ctx);
        assert_eq!(operands.len(), 4, "a, b, c, d");
        assert!(operands.iter().all(Result::is_ok));
        assert!(eval.parts.get("u1").is_none(), "the plan skips intermediates");
        assert!(eval.parts.get("u2").is_none(), "the plan skips intermediates");

        let single = request.nodes.iter().find(|node| node.id == "u1").expect("u1").clone();
        let mut eval = crate::Eval::new(&request, None);
        let mut ctx = EnginePartEval::new(&mut eval, &single);
        assert!(ctx.union_run(NodeView::from(&single)).is_none(), "two leaves do not fuse");
    }

    /// `input_profile` returns every profile sketch, or the body Part, and
    /// refuses the mixed Part.
    #[test]
    fn input_profile_classifies_sketches_or_a_body() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["x"], "meshTargets": [],
            "nodes": [
                {"id": "s", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}},
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "x", "type": "extrude", "values": {"amount": 5.0}}
            ],
            "connections": [{"from": "s", "to": "x", "in": "profile", "out": "sketch"}]
        }))
        .expect("profile request");
        let mut eval = crate::Eval::new(&request, None);
        let target = request.nodes.iter().find(|node| node.id == "x").expect("target").clone();
        {
            let mut ctx = EnginePartEval::new(&mut eval, &target);
            match ctx.input_profile(&["profile", "shape"]).expect("sketch profile") {
                ProfileInput::Sketches(sketches) => assert_eq!(sketches.len(), 1),
                ProfileInput::Body(_) => panic!("a sketch profile is not a body"),
            }
        }

        let body_request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["x"], "meshTargets": [],
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "x", "type": "extrude", "values": {"amount": 5.0}}
            ],
            "connections": [{"from": "b", "to": "x", "in": "shape", "out": "shape"}]
        }))
        .expect("body profile request");
        let mut eval = crate::Eval::new(&body_request, None);
        let target = body_request.nodes.iter().find(|node| node.id == "x").expect("target").clone();
        let mut ctx = EnginePartEval::new(&mut eval, &target);
        assert!(matches!(
            ctx.input_profile(&["profile", "shape"]).expect("body profile"),
            ProfileInput::Body(_),
        ));
    }

    /// Publishing a sketch is the one insert path; the overlay lookup reads the
    /// same output back under the node id.
    #[test]
    fn publish_sketch_is_visible_to_overlay_lookup() {
        let request: Request = serde_json::from_value(json!({
            "requestId": 1, "targets": ["s"], "meshTargets": [],
            "nodes": [
                {"id": "s", "type": "sketch", "values": {"sketch": serde_json::to_value(sketch()).unwrap()}}
            ],
            "connections": []
        }))
        .expect("publish request");
        let mut eval = crate::Eval::new(&request, None);
        let node = request.nodes[0].clone();
        let output = std::sync::Arc::new(SketchOutput::new(sketch()));
        {
            let mut ctx = EnginePartEval::new(&mut eval, &node);
            ctx.publish_sketch(NodeView::from(&node), output.clone());
        }
        let published = eval.sketches.get("s").expect("published sketch");
        assert!(std::sync::Arc::ptr_eq(published, &output));
        let looked_up = eval.part_sketch_output("s").expect("overlay lookup");
        assert!((looked_up.point(10.0, 0.0)[0] - 10.0).abs() < 1.0e-9);
    }

    /// `Part::from_bodies` keeps Part order and the source/source_body tagging
    /// contract: generated bodies start untagged and the engine stamps them.
    #[test]
    fn from_bodies_preserves_order_and_tagging() {
        let box_node = node(
            "box",
            "primitive",
            json!({"primitiveType": "box", "length": 10.0, "width": 10.0, "height": 10.0}),
        );
        let first = primitive_brep(&box_node).expect("box");
        let second = primitive_brep(&box_node).expect("box");
        let part = Part::from_bodies(vec![
            nodecraft_ops::SolidBody::new(first),
            nodecraft_ops::SolidBody::new(second),
        ]);
        assert_eq!(part.bodies.len(), 2);
        assert_eq!(part.bodies[0].source_body, 0);
        assert_eq!(part.bodies[1].source_body, 1);
        assert!(part.bodies.iter().all(|body| body.source.is_none()));
        let tagged = part.tagged(77);
        assert!(tagged.bodies.iter().all(|body| body.source == Some(77)));
        assert_eq!(tagged.bodies[1].source_body, 1);
    }

    #[test]
    fn output_kind_declaration_wins_and_unported_nodes_use_the_contract() {
        let values = json!({});
        let view = NodeView::new("x", "not-a-node", &values, false);
        assert_eq!(output_kind(view), OutputKind::Part);
        let sketch_view = NodeView::new("s", "sketch", &values, false);
        assert_eq!(output_kind(sketch_view), OutputKind::Sketch);
    }

    /// Every registry id has a presentation row in the contract, so the UI never
    /// has to grow a per-type arm for a newly registered node.
    #[test]
    fn every_registered_builtin_node_has_contract_metadata() {
        for node in NODES {
            for node_type in node.node_types() {
                assert!(
                    nodecraft_node_contract::metadata(node_type).is_some(),
                    "registered built-in {node_type:?} has no contract metadata row",
                );
            }
        }
    }

    /// The registry replaces the exported type lists: the diagnostics scope and
    /// the export-chain terminal are declarations, not string matches.
    #[test]
    fn diagnostics_and_export_terminal_are_registry_declarations() {
        assert_eq!(diagnostics_scope("primitive"), DiagnosticsScope::Node);
        assert_eq!(diagnostics_scope("not-a-node"), DiagnosticsScope::Node);
        assert!(export_terminal("primitive"));
        assert!(export_terminal("boolean"));
        assert!(export_terminal("fuse"));
        assert!(export_terminal("extrude"));
        assert!(!export_terminal("fillet"), "a feature is not an export source");
        assert!(!export_terminal("not-a-node"));
    }
}
