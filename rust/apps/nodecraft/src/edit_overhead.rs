//! Per-edit bookkeeping benchmark (measurement only).
//!
//! This module is compiled only for the app crate's unit-test target and its
//! one test is `#[ignore]`d, so it never runs in the default suite. It times the
//! real app path for one small value edit on a cheap-node graph:
//!
//! 1. session: `Session::apply_commands` (the one whole-document hash pass for
//!    the new revision and the dirty-set computation),
//! 2. app: the document clone the app takes for the worker job
//!    (`evaluation.rs`, `let job_document = eval_document.clone()`), plus the
//!    `DocumentStore::publish_document` clone the app also pays per edit,
//! 3. bridge: `Session::hashes_with_sources` (the Arc reuse path; no hash pass)
//!    + `engine_bridge::build_exact_request` (the real materialisation of every
//!    node and connection into a `Request`),
//! 4. engine: `evaluate_exact_with_imports` on that request once all closure
//!    nodes are cache hits, split by the engine's own inert phase capture
//!    (`nodecraft_engine::bench_support`),
//! 5. total: 1 + 2 + 3 + 4.
//!
//! The kernel-increment cost of the edit is measured separately: the pre-edit
//! document is evaluated first, then the post-edit request is evaluated against
//! a clone of that cache with the production dirty set, so only the nodes whose
//! content key actually changed miss (median of three such passes).
//!
//! Run with:
//!   cargo test -p nodecraft --release edit_overhead -- --ignored --nocapture
//! Optional filters: `EDIT_BENCH_NS=100,1000` and
//! `EDIT_BENCH_SHAPES=chain,fan,mix`.

use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

use nodecraft_engine::{EngineCache, ImportContext, bench_support};
use nodecraft_protocol::Request;
use nodecraft_session::store::DocumentStore;
use nodecraft_session::{Command, GraphConnection, GraphDocument, GraphNode, Session};
use serde_json::{Value, json};

use crate::engine_bridge;

const WARMUP: usize = 5;
const RUNS: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Chain,
    Fan,
    Mix,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Fan => "fan",
            Self::Mix => "mix",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Location {
    Leaf,
    Middle,
    Root,
}

impl Location {
    fn name(self) -> &'static str {
        match self {
            Self::Leaf => "leaf",
            Self::Middle => "middle",
            Self::Root => "root",
        }
    }
}

struct Fixture {
    shape: Shape,
    n: usize,
    edges: usize,
    document: GraphDocument,
    target: String,
    edit_node: String,
    edit_field: &'static str,
}

fn node(id: String, node_type: &str, values: Value) -> GraphNode {
    GraphNode {
        id,
        node_type: node_type.to_owned(),
        values,
        suppressed: false,
        x: 0.0,
        y: 0.0,
        expressions: BTreeMap::new(),
    }
}

fn translate(id: String) -> GraphNode {
    node(id, "translate", json!({"x": 1.0}))
}

fn edge(from: &str, to: &str) -> GraphConnection {
    GraphConnection {
        from: from.to_owned(),
        to: to.to_owned(),
        input: "shape".to_owned(),
        out: "shape".to_owned(),
    }
}

/// The three shapes all share one Box primitive and only use Translate above it.
///
/// `chain`: box -> t1 -> ... -> t(N-1). Every node is upstream of the target.
/// `fan`:   box -> t_i for i in 1..N. Each Translate is a leaf; the target is the
///          last one, so only box + target are in the target closure.
/// `mix`:   box -> s1 -> ... -> s(N/2), then s(N/2) -> t_j for the rest; the
///          target is the last fan leaf. The target closure is the whole spine.
fn fixture(shape: Shape, n: usize) -> Fixture {
    assert!(n >= 8);
    let mut nodes = Vec::with_capacity(n);
    let mut connections = Vec::with_capacity(n - 1);
    nodes.push(node(
        "box".to_owned(),
        "primitive",
        json!({"primitiveType": "box", "length": 40.0, "width": 10.0, "height": 20.0}),
    ));
    let spine = n / 2;
    match shape {
        Shape::Chain => {
            for index in 1..n {
                nodes.push(translate(format!("t{index}")));
                let from = if index == 1 { "box".to_owned() } else { format!("t{}", index - 1) };
                connections.push(edge(&from, &format!("t{index}")));
            }
        }
        Shape::Fan => {
            for index in 1..n {
                nodes.push(translate(format!("t{index}")));
                connections.push(edge("box", &format!("t{index}")));
            }
        }
        Shape::Mix => {
            for index in 1..=spine {
                nodes.push(translate(format!("t{index}")));
                let from = if index == 1 { "box".to_owned() } else { format!("t{}", index - 1) };
                connections.push(edge(&from, &format!("t{index}")));
            }
            for index in spine + 1..n {
                nodes.push(translate(format!("t{index}")));
                connections.push(edge(&format!("t{spine}"), &format!("t{index}")));
            }
        }
    }
    let target = format!("t{}", n - 1);
    let mut document = GraphDocument::new("edit-overhead-bench");
    document.root.nodes = nodes;
    document.root.connections = connections;
    document.root.interface.outputs = vec![target.clone()];
    Fixture { shape, n, edges: n - 1, document, target, edit_node: String::new(), edit_field: "x" }
}

fn edit_target(fixture: &Fixture, location: Location) -> (String, &'static str) {
    match (fixture.shape, location) {
        (_, Location::Root) => ("box".to_owned(), "length"),
        // In a fan every Translate is a leaf, so the "middle" node has no
        // downstream either; it is still a different node from the target.
        (Shape::Fan, Location::Middle) => (format!("t{}", fixture.n / 2), "x"),
        (Shape::Mix, Location::Middle) => (format!("t{}", fixture.n / 4), "x"),
        (_, Location::Middle) => (format!("t{}", fixture.n / 2), "x"),
        (_, Location::Leaf) => (format!("t{}", fixture.n - 1), "x"),
    }
}

#[derive(Default)]
struct CaseResult {
    shape: &'static str,
    n: usize,
    edges: usize,
    location: &'static str,
    dirty: usize,
    session_med: f64,
    session_p90: f64,
    clone_med: f64,
    clone_p90: f64,
    publish_med: f64,
    publish_p90: f64,
    hash_med: f64,
    hash_p90: f64,
    build_med: f64,
    build_p90: f64,
    engine_total_med: f64,
    engine_total_p90: f64,
    engine_prewarm_med: f64,
    engine_walk_med: f64,
    engine_order_med: f64,
    engine_loop_med: f64,
    engine_tail_med: f64,
    total_med: f64,
    total_p90: f64,
    cold_ms: f64,
    cold_misses: u32,
    cold_hits: u32,
    warm_hits: u32,
    warm_misses: u32,
    document_bytes: usize,
    request_bytes: usize,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 { (sorted[middle - 1] + sorted[middle]) / 2.0 } else { sorted[middle] }
}

/// Nearest-rank p90 (the sample at ceil(0.9 * n), 1-based).
fn p90(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((sorted.len() as f64) * 0.9).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Approximate heap bytes owned by a JSON value. BTreeMap node overhead is
/// estimated as 16 bytes per entry; the report marks this as an estimate.
fn value_heap(value: &Value) -> usize {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => 0,
        Value::String(text) => text.capacity(),
        Value::Array(items) => {
            items.capacity() * std::mem::size_of::<Value>()
                + items.iter().map(value_heap).sum::<usize>()
        }
        Value::Object(map) => {
            map.len() * (std::mem::size_of::<String>() + std::mem::size_of::<Value>() + 16)
                + map.iter().map(|(key, value)| key.capacity() + value_heap(value)).sum::<usize>()
        }
    }
}

fn node_heap(node: &GraphNode) -> usize {
    std::mem::size_of::<GraphNode>()
        + node.id.capacity()
        + node.node_type.capacity()
        + value_heap(&node.values)
        + node.expressions.len() * (std::mem::size_of::<String>() * 2 + 16)
        + node
            .expressions
            .iter()
            .map(|(key, value)| key.capacity() + value.capacity())
            .sum::<usize>()
}

fn document_heap(document: &GraphDocument) -> usize {
    std::mem::size_of::<GraphDocument>()
        + document.id.capacity()
        + std::mem::size_of::<nodecraft_session::GraphDefinition>()
        + document.root.id.capacity()
        + document.root.name.capacity()
        + document.root.kind.capacity()
        + document
            .root
            .interface
            .inputs
            .iter()
            .chain(document.root.interface.outputs.iter())
            .map(String::capacity)
            .sum::<usize>()
        + document.root.nodes.capacity() * std::mem::size_of::<GraphNode>()
        + document.root.nodes.iter().map(node_heap).sum::<usize>()
        + document.root.connections.capacity() * std::mem::size_of::<GraphConnection>()
        + document
            .root
            .connections
            .iter()
            .map(|link| {
                link.from.capacity()
                    + link.to.capacity()
                    + link.input.capacity()
                    + link.out.capacity()
            })
            .sum::<usize>()
        + document.definitions.capacity()
            * std::mem::size_of::<nodecraft_session::GraphDefinition>()
        + document.ingested_sources.capacity()
            * std::mem::size_of::<nodecraft_protocol::IngestRecord>()
}

fn empty_dirty() -> HashSet<String> {
    HashSet::new()
}

#[allow(clippy::too_many_lines)]
fn run_case(shape: Shape, n: usize, location: Location) -> CaseResult {
    let mut fixture = fixture(shape, n);
    let (edit_node, edit_field) = edit_target(&fixture, location);
    fixture.edit_node = edit_node.clone();
    fixture.edit_field = edit_field;

    let mut session = Session::new(fixture.document);
    assert_eq!(
        engine_bridge::target_for(session.document(), None).as_deref(),
        Some(fixture.target.as_str())
    );
    let target = fixture.target.clone();
    let imports = ImportContext::new(None, Vec::new());

    // Pre-edit evaluation: the cache a real edit starts from.
    let pre_hashes =
        session.hashes_with_sources(&imports.source_hashes).expect("pre-edit session hashes");
    let pre_request = engine_bridge::build_exact_request(
        session.document(),
        &target,
        &pre_hashes,
        &empty_dirty(),
        None,
        None,
        None,
        &[],
        None,
        false,
        None,
        false,
        false,
    );
    let pre = nodecraft_engine::evaluate_exact_with_imports(
        pre_request,
        EngineCache::new(),
        None,
        imports.clone(),
    );
    assert!(pre.response.node_errors.is_empty(), "pre-edit errors: {:?}", pre.response.node_errors);
    let mut cache = pre.cache;

    // 1. session: one small value edit through the real command path.
    let mut session_ms = Vec::with_capacity(RUNS);
    let mut change = None;
    for index in 0..WARMUP + RUNS {
        let value = if edit_field == "length" {
            json!(40.0 + (index as f64 + 1.0) * 0.25)
        } else {
            json!(1.0 + (index as f64 + 1.0) * 0.25)
        };
        let command =
            Command::SetValue { node: edit_node.clone(), field: edit_field.to_owned(), value };
        let started = Instant::now();
        let result = session.apply_commands(&[command]).expect("benchmark edit applies");
        let elapsed = ms(started);
        assert!(result.changed);
        if index >= WARMUP {
            session_ms.push(elapsed);
        }
        change = Some(result);
    }
    let change = change.expect("at least one edit");
    let dirty: HashSet<String> = change.dirty_nodes.iter().cloned().collect();
    assert!(!dirty.is_empty(), "edit must dirty at least itself");

    // 2. app: the per-job snapshot clone (evaluation.rs) and the store publish
    // clone (store.rs).
    let document = session.document();
    let mut clone_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        let copy = document.clone();
        let elapsed = ms(started);
        black_box(&copy);
        clone_ms.push(elapsed);
    }
    let mut store = DocumentStore::new();
    store.insert("bench", document.clone()).expect("store insert");
    let mut publish_ms = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        store.publish_document("bench", document);
        publish_ms.push(ms(started));
    }

    // 3. bridge: reuse the session's revision hashes (an Arc clone, no hash
    // pass), then materialise the Request. This is the exact sequence
    // `evaluate_flat_exact_with_imports_and_hashes` runs on the evaluation
    // worker when it receives the session map.
    let mut hash_ms = Vec::with_capacity(RUNS);
    let mut build_ms = Vec::with_capacity(RUNS);
    let mut post_hashes = pre_hashes.clone();
    let mut post_request = None;
    let pre_merge = engine_bridge::pre_merge_sources(document, &target);
    let pre_merge = pre_merge.as_ref().map(|(a, b)| (a.as_str(), b.as_str()));
    for _ in 0..RUNS {
        let started = Instant::now();
        let hashes =
            session.hashes_with_sources(&imports.source_hashes).expect("post-edit session hashes");
        hash_ms.push(ms(started));
        let started = Instant::now();
        let request = engine_bridge::build_exact_request(
            document,
            &target,
            &hashes,
            &dirty,
            None,
            pre_merge,
            None,
            &[],
            None,
            false,
            None,
            false,
            false,
        );
        build_ms.push(ms(started));
        black_box(&request);
        post_hashes = hashes;
        post_request = Some(request);
    }
    let post_request = post_request.expect("one built request");
    let request_bytes = serde_json::to_vec(&post_request).expect("request serialises").len();

    // 4a. kernel increment: evaluate the post-edit request against the pre-edit
    // cache, so only the nodes whose content key changed miss. Three samples,
    // each against a clone of the same pre-edit cache; the cache clone itself is
    // outside the timed region.
    let pre_edit_cache = cache.clone();
    let mut cold_samples = Vec::with_capacity(3);
    let mut cold_stats = nodecraft_protocol::CacheStats::default();
    for _ in 0..3 {
        let input = pre_edit_cache.clone();
        let started = Instant::now();
        let cold = nodecraft_engine::evaluate_exact_with_imports(
            post_request.clone(),
            input,
            None,
            imports.clone(),
        );
        cold_samples.push(ms(started));
        cold_stats = cold.response.cache_stats.unwrap_or_default();
        assert!(
            cold.response.node_errors.is_empty(),
            "post-edit errors: {:?}",
            cold.response.node_errors
        );
        cache = cold.cache;
    }
    let cold_ms = median(&cold_samples);

    // 4b. engine bookkeeping: same document, empty dirty set, so every closure
    // node is a cache hit. Warm once, then measure the all-hit pass.
    let warm_request = engine_bridge::build_exact_request(
        document,
        &target,
        &post_hashes,
        &empty_dirty(),
        None,
        pre_merge,
        None,
        &[],
        None,
        false,
        None,
        false,
        false,
    );
    let warm = nodecraft_engine::evaluate_exact_with_imports(
        warm_request.clone(),
        std::mem::take(&mut cache),
        None,
        imports.clone(),
    );
    let warm_stats = warm.response.cache_stats.unwrap_or_default();
    assert_eq!(warm_stats.misses, 0, "warm-up pass must be all hits");
    cache = warm.cache;

    let mut requests: Vec<Request> = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        requests.push(warm_request.clone());
    }
    let mut engine_total = Vec::with_capacity(RUNS);
    let mut engine_prewarm = Vec::with_capacity(RUNS);
    let mut engine_walk = Vec::with_capacity(RUNS);
    let mut engine_order = Vec::with_capacity(RUNS);
    let mut engine_loop = Vec::with_capacity(RUNS);
    let mut engine_tail = Vec::with_capacity(RUNS);
    let mut warm_hits = 0u32;
    let mut warm_misses = 0u32;
    for request in requests {
        let taken = std::mem::take(&mut cache);
        let (output, mut phases) = bench_support::capture_eval_phases(|| {
            nodecraft_engine::evaluate_exact_with_imports(request, taken, None, imports.clone())
        });
        cache = output.cache;
        let stats = output.response.cache_stats.unwrap_or_default();
        assert_eq!(stats.misses, 0, "measured engine pass must be all hits");
        assert!(
            output.response.node_errors.is_empty(),
            "warm errors: {:?}",
            output.response.node_errors
        );
        warm_hits = warm_hits.saturating_add(stats.hits);
        warm_misses = warm_misses.saturating_add(stats.misses);
        let phase = phases.pop().expect("one phase record per evaluation");
        engine_total.push(phase.total_ms);
        engine_prewarm.push(phase.prewarm_ms);
        engine_walk.push(phase.walk_ms);
        engine_order.push(phase.order_ms);
        engine_loop.push(phase.loop_ms);
        engine_tail.push(phase.tail_ms);
    }

    let mut total = Vec::with_capacity(RUNS);
    for index in 0..RUNS {
        total.push(
            session_ms[index]
                + clone_ms[index]
                + hash_ms[index]
                + build_ms[index]
                + engine_total[index],
        );
    }

    CaseResult {
        shape: shape.name(),
        n,
        edges: fixture.edges,
        location: location.name(),
        dirty: dirty.len(),
        session_med: median(&session_ms),
        session_p90: p90(&session_ms),
        clone_med: median(&clone_ms),
        clone_p90: p90(&clone_ms),
        publish_med: median(&publish_ms),
        publish_p90: p90(&publish_ms),
        hash_med: median(&hash_ms),
        hash_p90: p90(&hash_ms),
        build_med: median(&build_ms),
        build_p90: p90(&build_ms),
        engine_total_med: median(&engine_total),
        engine_total_p90: p90(&engine_total),
        engine_prewarm_med: median(&engine_prewarm),
        engine_walk_med: median(&engine_walk),
        engine_order_med: median(&engine_order),
        engine_loop_med: median(&engine_loop),
        engine_tail_med: median(&engine_tail),
        total_med: median(&total),
        total_p90: p90(&total),
        cold_ms,
        cold_misses: cold_stats.misses,
        cold_hits: cold_stats.hits,
        warm_hits,
        warm_misses,
        document_bytes: document_heap(document),
        request_bytes,
    }
}

fn env_usize_list(name: &str) -> Option<Vec<usize>> {
    std::env::var(name).ok().map(|value| {
        value.split(',').filter_map(|item| item.trim().parse::<usize>().ok()).collect()
    })
}

fn env_shapes() -> Vec<Shape> {
    let Some(value) = std::env::var("EDIT_BENCH_SHAPES").ok() else {
        return vec![Shape::Chain, Shape::Fan, Shape::Mix];
    };
    value
        .split(',')
        .filter_map(|item| match item.trim() {
            "chain" => Some(Shape::Chain),
            "fan" => Some(Shape::Fan),
            "mix" => Some(Shape::Mix),
            _ => None,
        })
        .collect()
}

#[test]
#[ignore = "benchmark: per-edit bookkeeping; run with --release --ignored --nocapture"]
fn per_edit_bookkeeping_measurements() {
    let sizes = env_usize_list("EDIT_BENCH_NS").unwrap_or_else(|| vec![100, 1_000, 5_000]);
    println!(
        "RESULT\tshape\tn\tedges\tedit\tdirty\tsession_med\tsession_p90\tclone_med\tclone_p90\tpublish_med\tpublish_p90\thash_med\thash_p90\tbuild_med\tbuild_p90\tengine_med\tengine_p90\tprewarm_med\twalk_med\torder_med\tloop_med\ttail_med\ttotal_med\ttotal_p90\tcold_ms\tcold_misses\tcold_hits\twarm_hits\twarm_misses\tdoc_bytes\trequest_bytes"
    );
    for shape in env_shapes() {
        for n in sizes.iter().copied() {
            for location in [Location::Leaf, Location::Middle, Location::Root] {
                let case = run_case(shape, n, location);
                println!(
                    "RESULT\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}",
                    case.shape,
                    case.n,
                    case.edges,
                    case.location,
                    case.dirty,
                    case.session_med,
                    case.session_p90,
                    case.clone_med,
                    case.clone_p90,
                    case.publish_med,
                    case.publish_p90,
                    case.hash_med,
                    case.hash_p90,
                    case.build_med,
                    case.build_p90,
                    case.engine_total_med,
                    case.engine_total_p90,
                    case.engine_prewarm_med,
                    case.engine_walk_med,
                    case.engine_order_med,
                    case.engine_loop_med,
                    case.engine_tail_med,
                    case.total_med,
                    case.total_p90,
                    case.cold_ms,
                    case.cold_misses,
                    case.cold_hits,
                    case.warm_hits,
                    case.warm_misses,
                    case.document_bytes,
                    case.request_bytes,
                );
            }
        }
    }
}
