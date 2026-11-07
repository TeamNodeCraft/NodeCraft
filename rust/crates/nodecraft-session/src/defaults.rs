//! Canonical node values.
//!
//! The editor must not mutate graph values while rendering. Missing structural
//! defaults are applied once when the session accepts a document or command node.

use crate::{GraphDocument, GraphNode};
use serde_json::{json, Value};

/// Applies structural defaults to every node in `document`.
pub fn canonicalize_document(document: &mut GraphDocument) {
    for node in &mut document.nodes { canonicalize_node(node); }
}

/// Applies structural defaults to one node. Numeric primitive dimensions are left
/// alone: missing dimensions are an engine concern, while these fields define the
/// node's mode/space/identity and are needed by the UI and content hash.
pub fn canonicalize_node(node: &mut GraphNode) {
    if !node.values.is_object() { node.values = json!({}); }
    // Legacy documents stored expressions under values._expressions. Move them
    // to the node-level map and never serialize the reserved value key again.
    if let Some(legacy) = node.values.as_object_mut().and_then(|values| values.remove("_expressions")) {
        if let Value::Object(legacy) = legacy {
            for (field, source) in legacy {
                if let Value::String(source) = source {
                    node.expressions.entry(field).or_insert(source);
                }
            }
        }
    }
    // Older documents stored fillets as chamfer nodes with mode: "fillet".
    if node.node_type == "chamfer"
        && node.values.get("mode").and_then(serde_json::Value::as_str) == Some("fillet")
    {
        node.node_type = "fillet".to_owned();
        node.values.as_object_mut().map(|values| values.remove("mode"));
    }
    match node.node_type.as_str() {
        "primitive" => {
            if node.values.get("primitiveType").is_none() { node.values["primitiveType"] = json!("box"); }
        }
        "translate" => {
            for field in ["x", "y", "z", "rx", "ry", "rz"] {
                if !node.values.get(field).and_then(serde_json::Value::as_f64).is_some_and(f64::is_finite) {
                    node.values[field] = json!(0.0);
                }
            }
            if node.values.get("space").is_none() { node.values["space"] = json!("world"); }
        }
        "chamfer" => {
            if node.values.get("mode").is_none() { node.values["mode"] = json!("chamfer"); }
            if node.values.get("chamferType").is_none() { node.values["chamferType"] = json!("equal"); }
        }
        "fillet" => {
            if node.values.get("continuity").is_none() { node.values["continuity"] = json!(""); }
        }
        "look" => {
            if node.values.get("color").and_then(serde_json::Value::as_str).is_none() {
                node.values["color"] = json!("#7fa0cf");
            }
        }
        "surfaceloft" => {
            if node.values.get("ruled").is_none() { node.values["ruled"] = json!(0.0); }
        }
        "spline" => {
            if node.values.get("points").and_then(serde_json::Value::as_array).is_none_or(|points| points.len() < 2) {
                node.values["points"] = json!([
                    {"x": -80.0, "y": 0.0},
                    {"x": -25.0, "y": 45.0},
                    {"x": 25.0, "y": -45.0},
                    {"x": 80.0, "y": 0.0}
                ]);
            }
            if node.values.get("degree").and_then(serde_json::Value::as_u64).is_none_or(|degree| !(1..=9).contains(&degree)) {
                node.values["degree"] = json!(3);
            }
            if node.values.get("control").is_none() { node.values["control"] = json!(false); }
            if node.values.get("periodic").is_none() { node.values["periodic"] = json!(false); }
        }
        "extrude" => {
            if node.values.get("operation").is_none() { node.values["operation"] = json!("new"); }
        }
        "shell" => {
            if node.values.get("direction").is_none() { node.values["direction"] = json!("inward"); }
        }
        "boolean" | "fuse" => {
            if node.values.get("mode").is_none() { node.values["mode"] = json!("union"); }
        }
        "stepimporter" => {
            if node.values.get("path").is_none() { node.values["path"] = json!(""); }
            if node.values.get("heal").is_none() { node.values["heal"] = json!(true); }
        }
        "exporter" => {
            let name = node.values.get("fileName").and_then(serde_json::Value::as_str).unwrap_or("model");
            node.values["fileName"] = json!(name.trim_end_matches(".step").trim_end_matches(".STEP"));
            if node.values["fileName"].as_str().is_none_or(str::is_empty) { node.values["fileName"] = json!("model"); }
            if node.values.get("overwrite").is_none() { node.values["overwrite"] = json!(false); }
        }
        _ => {}
    }
}
