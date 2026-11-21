//! STEP/CAD import: one file, every body.
//!
//! The engine owns the path, file identity, display-import cache and
//! diagnostics; this module turns the converted bodies into a `Part` through
//! the one importer constructor, so names, colours and per-body warnings keep
//! their existing shape.

use super::{BuiltinNode, DiagnosticsScope, NodeDiagnostic, NodeView, PartEval};
use crate::{Part, RequestIndex};

/// Registered for the `stepimporter` contract id.
pub(crate) struct StepImporter;

impl BuiltinNode for StepImporter {
    fn node_types(&self) -> &'static [&'static str] {
        &["stepimporter"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(import) = ctx.step_import(node)? else {
            // No path yet: the neutral unconfigured state. An empty Part keeps
            // downstream nodes behaving as they do for any empty input, without
            // an error card, banner or failed status.
            return Ok(Part::empty());
        };
        ctx.record_diagnostics(vec![NodeDiagnostic::Step(import.diagnostics)]);
        let (part, color_warnings) = Part::from_imported(import.bodies);
        let mut warnings = import.warnings;
        warnings.extend(color_warnings);
        ctx.warn(warnings);
        Ok(part)
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        8
    }

    fn mesh_bodies(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn diagnostics_scope(&self) -> DiagnosticsScope {
        DiagnosticsScope::IsolatedProcess
    }

    fn export_terminal(&self) -> bool {
        true
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // An import is a source; suppressing it turns it off.
        None
    }

    fn body_layout(
        &self,
        node: NodeView<'_>,
        _inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // Every imported body is produced by this node.
        Some(vec![node.id.to_owned()])
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    use super::*;

    fn request(path: &str) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["s"],
            "meshTargets": [],
            "nodes": [{"id": "s", "type": "stepimporter", "values": {"path": path}, "cache": {"key": "s"}}],
            "connections": []
        }))
        .expect("import request")
    }

    /// Removing the registry entry makes the engine reject `stepimporter`, so
    /// this test is the entry's fail proof.
    #[test]
    fn registry_routes_stepimporter() {
        assert!(crate::nodes::handles("stepimporter"));
        let builtin = crate::nodes::find("stepimporter").expect("stepimporter registered");
        assert_eq!(builtin.node_types(), &["stepimporter"]);
        assert_eq!(builtin.diagnostics_scope(), DiagnosticsScope::IsolatedProcess);
        assert!(builtin.export_terminal());
        let values = serde_json::json!({});
        let view = NodeView::new("s", "stepimporter", &values, false);
        assert_eq!(builtin.body_layout(view, &[]), Some(vec!["s".to_owned()]));
    }

    /// An importer with no path is not configured yet: it yields an empty Part
    /// with no node error and no warning. Downstream nodes then see the same
    /// empty input they would from any other empty Part.
    #[test]
    fn empty_path_is_an_unconfigured_empty_part_not_an_error() {
        let output = crate::evaluate_exact(request(""), crate::EngineCache::new(), None);
        assert!(
            output.response.node_errors.is_empty(),
            "an empty path must not be an error: {:?}",
            output.response.node_errors
        );
        assert!(
            !output.response.node_warnings.contains_key("s"),
            "an empty path must not warn: {:?}",
            output.response.node_warnings
        );
        let part = output.exact_parts.get("s").expect("empty importer part");
        assert!(part.bodies.is_empty(), "an unconfigured importer has no bodies");
    }

    /// The neutral state is only for an empty path: a non-empty missing source
    /// stays a node error, unchanged.
    #[test]
    fn missing_non_empty_path_is_still_an_error() {
        let output = crate::evaluate_exact(
            request("/nonexistent/sn-import-missing.step"),
            crate::EngineCache::new(),
            None,
        );
        let error = output.response.node_errors.get("s").expect("missing source error");
        assert!(error.contains("Source file not found"), "unexpected error: {error}");
        assert!(
            output.exact_parts.get("s").is_none_or(|part| part.bodies.is_empty()),
            "a failed import yields no bodies"
        );
    }

    /// Every body in the file becomes a Part body, with no first-body collapse.
    #[test]
    fn three_body_import_produces_three_bodies() {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives::create_box;

        let directory =
            std::env::temp_dir().join(format!("sn-import-three-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("temp directory");
        let path = directory.join("three.step");
        let mut kernel = nodecraft_solidcore::kernel::kernel();
        let breps: Vec<_> = (0..3)
            .map(|index| {
                create_box(
                    &mut kernel,
                    Point3::new(index as f64 * 20.0, 0.0, 0.0),
                    10.0,
                    10.0,
                    10.0,
                )
                .expect("box")
            })
            .collect();
        nodecraft_step::write_breps_exact(path.to_str().expect("utf-8"), &breps).expect("write");

        let output = crate::evaluate_exact(
            request(path.to_str().expect("utf-8")),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("s").expect("imported part");
        assert_eq!(part.bodies.len(), 3, "the importer must keep every body");
        let diagnostics = output.diagnostics.get("s").expect("diagnostics recorded");
        let NodeDiagnostic::Step(diagnostics) = &diagnostics[0];
        assert_eq!(diagnostics.bodies, 3);
        assert_eq!(diagnostics.breps, 3);

        let _ = std::fs::remove_dir_all(&directory);
    }
}
