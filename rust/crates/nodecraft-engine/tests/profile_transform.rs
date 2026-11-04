//! Profile the second evaluation of a transform edit on a large graph.
//!
//! `cargo test -p nodecraft-engine --test profile_transform -- --ignored --nocapture`
//!
//! Optionally set `NODECRAFT_PROFILE_GRAPH=/path/to/file.nodecraft`.

use nodecraft_engine::{EngineCache, evaluate};
use nodecraft_protocol::{CacheState, Connection, Node, Request};
use std::path::PathBuf;

fn graph_path() -> PathBuf {
    std::env::var_os("NODECRAFT_PROFILE_GRAPH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../samples/gigatest-forest-5000.nodecraft")
        })
}

fn node(id: &str, node_type: &str, values: &serde_json::Value, suppressed: bool, key: &str) -> Node {
    Node {
        id: id.into(),
        node_type: node_type.into(),
        values: values.clone(),
        suppressed,
        cache: Some(CacheState { key: Some(key.into()), state: Some("cached".into()) }),
    }
}

fn request(nodes: Vec<Node>, connections: Vec<Connection>, target: &str, revision: u64) -> Request {
    Request {
        request_id: revision,
        tab_id: None,
        graph_revision: Some(revision.to_string()),
        targets: vec![target.into()],
        mesh_targets: Some(vec![target.into()]),
        nodes,
        connections,
        binary_mesh: false,
        project_directory: None,
        progress: false,
        run_exporters: false,
        mesh_tolerance: None,
        comparison_targets: None,
    }
}

#[test]
#[ignore = "profiling probe; run explicitly with --ignored --nocapture"]
fn transform_redraw_breaks_down_into_brep_and_mesh() {
    let path = graph_path();
    let text = std::fs::read_to_string(&path).expect("graph file");
    let raw: serde_json::Value = serde_json::from_str(&text).expect("valid graph");
    let raw_nodes = raw["nodes"].as_array().expect("nodes");
    let raw_connections = raw["connections"].as_array().expect("connections");

    let target = raw_nodes
        .iter()
        .filter(|node| {
            let id = node["id"].as_str().unwrap_or("");
            !raw_connections.iter().any(|link| link["from"].as_str() == Some(id))
        })
        .filter_map(|node| Some((node["id"].as_str()?.to_owned(), node["type"].as_str()?.to_owned())))
        .filter(|(_, node_type)| {
            matches!(node_type.as_str(), "translate" | "combine" | "boolean" | "fuse")
        })
        .next_back()
        .map(|(id, _)| id)
        .expect("a downstream transform/combine target");

    let base_nodes: Vec<(String, String, serde_json::Value, bool)> = raw_nodes
        .iter()
        .map(|node| {
            (
                node["id"].as_str().unwrap_or_default().to_owned(),
                node["type"].as_str().unwrap_or_default().to_owned(),
                node["values"].clone(),
                node["suppressed"].as_bool().unwrap_or(false),
            )
        })
        .collect();
    let connections: Vec<Connection> = raw_connections
        .iter()
        .map(|link| Connection {
            from: link["from"].as_str().unwrap_or_default().to_owned(),
            to: link["to"].as_str().unwrap_or_default().to_owned(),
            input: link["in"].as_str().unwrap_or_default().to_owned(),
            out: link["out"].as_str().unwrap_or_default().to_owned(),
        })
        .collect();

    let build = |target_revision: u64| {
        base_nodes
            .iter()
            .map(|(id, node_type, values, suppressed)| {
                let revision = if id == &target { target_revision } else { 0 };
                node(id, node_type, values, *suppressed, &format!("{id}-{revision}"))
            })
            .collect()
    };

    let first = evaluate(request(build(0), connections.clone(), &target, 1), EngineCache::new(), None);
    let print = |label: &str, output: &nodecraft_engine::EngineOutput| {
        let response = &output.response;
        let target_ms = response.timings.get(&target).copied().unwrap_or(0.0);
        let target_brep = response.brep_timings.get(&target).copied().unwrap_or(0.0);
        let total_brep: f64 = response.brep_timings.values().sum();
        let total_mesh = response.timings.values().sum::<f64>() - total_brep;
        println!(
            "{label}: total={:.1}ms target={:.1}ms target_brep={:.1}ms brep_sum={:.1}ms mesh_sum={:.1}ms cache_hits={} cache_misses={}",
            response.evaluation_ms,
            target_ms,
            target_brep,
            total_brep,
            total_mesh,
            response.cache_stats.as_ref().map_or(0, |stats| stats.hits),
            response.cache_stats.as_ref().map_or(0, |stats| stats.misses),
        );
    };
    print("warm", &first);
    let cache = first.cache;

    let second = evaluate(request(build(1), connections, &target, 2), cache, None);
    print("transform", &second);
}
