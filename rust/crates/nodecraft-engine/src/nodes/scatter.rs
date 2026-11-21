//! Scatter: repeat an exact Part on a seeded grid/random field.
//!
//! Every instance clones the source Part, so copies of one body keep the
//! source's body key and share one display plan. Copy 0 is the first placed
//! instance; the drag preview anchors at the source frame (a repeater).

use super::{BuiltinNode, NodeView, PartEval};
use crate::{BodyTransform, Part, RequestIndex};

/// Registered for the `scatter` contract id.
pub(crate) struct Scatter;

impl BuiltinNode for Scatter {
    fn node_types(&self) -> &'static [&'static str] {
        &["scatter"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Scatter has no source".into());
        };
        if source.bodies.is_empty() && source.sketches.is_empty() {
            return Err("Scatter source Part is empty".into());
        }
        let count = count_values(node.values);
        let span_x = crate::contract_number("scatter", node.values, "spanX").max(1.0);
        let span_y = crate::contract_number("scatter", node.values, "spanY").max(1.0);
        let seed = crate::contract_number("scatter", node.values, "seed").to_bits();
        let min_scale = crate::contract_number("scatter", node.values, "minScale").max(0.05);
        let max_scale = crate::contract_number("scatter", node.values, "maxScale").max(min_scale);
        let distribution = crate::contract_text("scatter", node.values, "distribution");
        let columns = (count as f64).sqrt().ceil() as usize;
        let rows = (count + columns - 1) / columns;
        let cell_x = span_x / columns as f64;
        let cell_y = span_y / rows as f64;
        let mut rng = crate::ScatterRng::new(seed);
        let mut bodies = Vec::new();
        let mut sketches = Vec::new();
        for index in 0..count {
            let col = index % columns;
            let row = index / columns;
            let grid_x = if columns > 1 {
                -span_x / 2.0 + span_x * col as f64 / (columns - 1) as f64
            } else {
                0.0
            };
            let grid_y = if rows > 1 {
                -span_y / 2.0 + span_y * row as f64 / (rows - 1) as f64
            } else {
                0.0
            };
            let (x, y) = match distribution.as_ref() {
                "grid" => (grid_x, grid_y),
                "stratified" => (
                    -span_x / 2.0 + (col as f64 + rng.unit()) * cell_x,
                    -span_y / 2.0 + (row as f64 + rng.unit()) * cell_y,
                ),
                _ => ((rng.unit() - 0.5) * span_x, (rng.unit() - 0.5) * span_y),
            };
            let factor = min_scale + (max_scale - min_scale) * rng.unit();
            let mut instance_bodies = source.bodies.clone();
            for body in &mut instance_bodies {
                body.push_transform(BodyTransform::Scale { factor });
                body.push_transform(BodyTransform::Place {
                    offset: [x, y, 0.0],
                    rotation: [0.0; 3],
                });
            }
            bodies.extend(instance_bodies);
            let mut instance_sketches = source.sketches.clone();
            for sketch in &mut instance_sketches {
                sketch.placement = sketch.placement.then(BodyTransform::Scale { factor });
                sketch.placement = sketch
                    .placement
                    .then(BodyTransform::Place { offset: [x, y, 0.0], rotation: [0.0; 3] });
            }
            sketches.extend(instance_sketches);
        }
        Ok(Part { bodies, sketches })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        8
    }

    fn mesh_bodies(&self, node: NodeView<'_>) -> usize {
        count_values(node.values)
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // Scatter is a generator: suppressing it turns it off.
        None
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        let source = super::input_layout(inputs, "shape")?;
        let count = count_values(node.values);
        let mut layout = Vec::with_capacity(source.len() * count);
        for _ in 0..count {
            layout.extend(source.iter().cloned());
        }
        Some(layout)
    }

    fn repeats_input_bodies(&self, _node: NodeView<'_>) -> bool {
        true
    }
}

/// Scatter repeat count through the contract, clamped like the old arm.
fn count_values(values: &serde_json::Value) -> usize {
    crate::contract_number("scatter", values, "count").round().clamp(1.0, 5000.0) as usize
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn request(count: f64) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["scatter"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 4, "width": 4, "height": 4}},
                {"id": "scatter", "type": "scatter", "cache": {"key": "scatter"},
                 "values": {"count": count, "distribution": "grid", "spanX": 20.0, "spanY": 20.0,
                            "minScale": 1.0, "maxScale": 1.0}}
            ],
            "connections": [{"from": "box", "to": "scatter", "in": "shape", "out": "shape"}]
        }))
        .expect("scatter request")
    }

    /// Removing the registry entry makes the engine reject `scatter`, so this
    /// test is the entry's fail proof.
    #[test]
    fn registry_routes_scatter_and_its_repeater_role() {
        let builtin = crate::nodes::find("scatter").expect("scatter registered");
        assert_eq!(builtin.node_types(), &["scatter"]);
        let values = serde_json::json!({"count": 3});
        let view = crate::nodes::NodeView::new("scatter", "scatter", &values, false);
        assert!(builtin.repeats_input_bodies(view));
        assert_eq!(builtin.mesh_bodies(view), 3);
    }

    /// The count is the contract value, rounded and clamped to 1..=5000; a
    /// missing count takes the contract default (1000), not an old private one.
    #[test]
    fn count_is_clamped_through_the_contract() {
        assert_eq!(super::count_values(&serde_json::json!({"count": 0})), 1);
        assert_eq!(super::count_values(&serde_json::json!({"count": -3})), 1);
        assert_eq!(super::count_values(&serde_json::json!({"count": 1.0e9})), 5000);
        assert_eq!(super::count_values(&serde_json::json!({})), 1000);

        let output = crate::evaluate_exact(request(2.0), crate::EngineCache::new(), None);
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("scatter").expect("scatter exact part");
        assert_eq!(part.bodies.len(), 2);
        let keys: std::collections::HashSet<_> = part.bodies.iter().map(|body| body.key).collect();
        assert_eq!(keys.len(), 1, "every instance shares its source body plan");
    }

    /// The display layout is the source layout repeated once per instance.
    #[test]
    fn body_layout_is_the_source_repeated_by_count() {
        let builtin = crate::nodes::find("scatter").expect("scatter registered");
        let values = serde_json::json!({"count": 3});
        let view = crate::nodes::NodeView::new("scatter", "scatter", &values, false);
        assert_eq!(
            builtin.body_layout(view, &[("shape", vec!["box".to_owned(), "sphere".to_owned()])],),
            Some(vec![
                "box".to_owned(),
                "sphere".to_owned(),
                "box".to_owned(),
                "sphere".to_owned(),
                "box".to_owned(),
                "sphere".to_owned(),
            ]),
        );
    }
}
