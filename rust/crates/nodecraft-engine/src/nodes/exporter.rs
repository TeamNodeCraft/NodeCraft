//! STEP export: the one writer node.
//!
//! The engine resolves the input Part, the placement chain and the Combine
//! members once (`EffectContext::export_input`); this module owns the file
//! naming, overwrite/format rules and the one writer call per format. Nothing
//! here walks the request graph.

use std::path::{Path, PathBuf};

use super::{
    BuiltinNode, EffectContext, ExportChain, ExportInput, NodeEffect, NodeView, OutputKind,
    PartEval,
};
use crate::{Part, RequestIndex};

/// Registered for the `exporter` contract id.
pub(crate) struct Exporter;

impl BuiltinNode for Exporter {
    fn node_types(&self) -> &'static [&'static str] {
        &["exporter"]
    }

    fn evaluate(&self, _node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        // The exporter has no geometry of its own. Display asks its input Part
        // through the one exact path; the write happens in `effect`, never here.
        match ctx.input_part(&["shape"])? {
            Some(part) => Ok((*part).clone()),
            None => Err("Export node has no shape input.".into()),
        }
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        1
    }

    fn output_kind(&self) -> OutputKind {
        OutputKind::Effect
    }

    fn suppression_source(&self, _node: NodeView<'_>, _index: &RequestIndex<'_>) -> Option<String> {
        // An exporter is an effect terminal; suppressing it turns it off.
        None
    }

    fn effect(
        &self,
        node: NodeView<'_>,
        ctx: &mut EffectContext<'_>,
    ) -> Result<Option<NodeEffect>, String> {
        if !ctx.run_exporters() || ctx.token().is_cancelled() {
            return Ok(None);
        }
        let input = ctx.export_input(node);
        let directory =
            node.values.get("directory").and_then(serde_json::Value::as_str).unwrap_or("");
        let root = std::env::var("NODECRAFT_PROJECT_ROOT").unwrap_or_else(|_| ".".into());
        let project_folder =
            ctx.project_directory().filter(|value| !value.trim().is_empty()).map(PathBuf::from);
        let mut output = if directory.is_empty() {
            project_folder.unwrap_or_else(|| PathBuf::from(root.clone()).join("output"))
        } else {
            PathBuf::from(directory)
        };
        if output.is_relative() {
            output = PathBuf::from(root).join(output);
        }
        let stem = node
            .values
            .get("fileName")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("model")
            .trim_end_matches(".step")
            .trim();
        let stem = if stem.is_empty() { "model" } else { stem };
        let path = output.join(format!("{stem}.step"));
        let path_text = path.to_str().unwrap_or("model.step").to_owned();
        let result = if let Err(error) = std::fs::create_dir_all(&output) {
            Err(format!("{}: {error}", output.display()))
        } else {
            match step_options(node.values) {
                Err(error) => Err(error),
                Ok(options) => {
                    let format = node.values.get("format").and_then(serde_json::Value::as_str);
                    if format == Some("faceted") {
                        export_faceted(&input, &path_text, &options)
                    } else if input.combine {
                        export_combine_assembly(&input, &path_text, &options)
                    } else if input.chain.as_ref().is_some_and(|chain| chain.imported_step) {
                        export_imported_step(&input, &path_text, &options)
                    } else {
                        export_exact_or_part(&input, &path_text, &options)
                    }
                }
            }
        };
        match result {
            Ok(()) => Ok(Some(NodeEffect::WroteFile)),
            Err(error) => Err(stepio_error(&path_text, error)),
        }
    }
}

/// Every Stepio failure names the file it was operating on.
fn stepio_error(path: &str, error: String) -> String {
    if path.is_empty() || error.starts_with(path) { error } else { format!("{path}: {error}") }
}

/// The exporter's presentation options. One resolver, matching the display
/// writer and the old engine test options.
fn step_options(values: &serde_json::Value) -> Result<nodecraft_step::StepExportOptions, String> {
    let unit =
        values.get("unit").and_then(serde_json::Value::as_str).unwrap_or("mm").to_ascii_lowercase();
    if !matches!(unit.as_str(), "" | "mm" | "millimetre" | "millimeter") {
        return Err(format!("STEP export writes millimetres; {unit:?} output is not supported"));
    }
    let color = values
        .get("color")
        .and_then(serde_json::Value::as_str)
        .map(|color| {
            let hex = color.trim_start_matches('#');
            if hex.len() != 6 {
                return Err("color must be #RRGGBB".to_string());
            }
            let component = |range| {
                u8::from_str_radix(&hex[range], 16)
                    .map(|value| f64::from(value) / 255.0)
                    .map_err(|error| error.to_string())
            };
            Ok([component(0..2)?, component(2..4)?, component(4..6)?])
        })
        .transpose()?;
    Ok(nodecraft_step::StepExportOptions {
        product_name: values
            .get("productName")
            .or_else(|| values.get("fileName"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("solid")
            .to_string(),
        schema: values
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("ap242")
            .to_string(),
        author: values.get("author").and_then(serde_json::Value::as_str).unwrap_or("").to_string(),
        organization: values
            .get("organization")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("NodeCraft")
            .to_string(),
        color,
        layer: values.get("layer").and_then(serde_json::Value::as_str).map(str::to_string),
    })
}

/// The exact path: every realised body, or the Part's own body metadata when
/// the input does not expose one exact operand.
///
/// Exactness is authoritative: a writer failure, or a Part with no exact B-rep
/// bodies, is an error. Faceted output exists only through the exporter's
/// explicit `format: "faceted"` route; this path never tessellates.
fn export_exact_or_part(
    input: &ExportInput,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let part = input.part.as_ref().map_err(Clone::clone)?;
    match &input.breps {
        Ok(breps) if !breps.is_empty() => {
            nodecraft_step::write_breps_exact_with_options(path, breps, options)
                .map_err(|error| format!("exact STEP write failed: {error}"))
        }
        Ok(_) => Err("no exact B-rep bodies to export".into()),
        Err(error) => export_part_bodies(part, path, options)
            .map_err(|body_error| format!("{error}; exact body export failed: {body_error}")),
    }
}

/// Write the evaluated Part as one STEP body per Part body, keeping imported
/// names and colours. This is the route for a Part that is not one exact
/// operand (Circular Pattern is the common case).
fn export_part_bodies(
    part: &Part,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let solids = part.realized_bodies()?;
    if solids.is_empty() {
        return Err("Part has no solids".into());
    }
    let stem =
        Path::new(path).file_stem().and_then(|stem| stem.to_str()).unwrap_or("model").to_string();
    let mut bodies = Vec::with_capacity(solids.len());
    for (index, solid) in solids.iter().enumerate() {
        let instance = part.bodies.get(index);
        let name = instance.and_then(|body| body.name.clone()).unwrap_or_else(|| {
            if solids.len() == 1 { stem.clone() } else { format!("{stem}_{}", index + 1) }
        });
        bodies.push(nodecraft_step::ExportBody {
            name,
            color: instance.and_then(|body| body.color),
            placement: None,
            sheet: false,
            brep: nodecraft_step::record_of_brep(solid.brep())?,
        });
    }
    let mut options = options.clone();
    if options.product_name.trim().is_empty() {
        options.product_name = stem;
    }
    nodecraft_step::write_step_with_options(path, &bodies, &options)
}

/// Explicit `format: "faceted"` export: tessellate the same Part and write
/// every triangle. This is never a fallback for the exact route.
fn export_faceted(
    input: &ExportInput,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let part = input.part.as_ref().map_err(Clone::clone)?;
    let exact = crate::exact_part_from(part.as_ref(), 0);
    let mut cache = nodecraft_display::PlanCache::new(8);
    let mesh = nodecraft_display::export_mesh(
        &exact,
        nodecraft_display::ExportOptions { tolerance: 0.05, require_closed: false },
        &mut cache,
    )?;
    nodecraft_step::write_step_faceted(path, &mesh.mesh.vertices, &mesh.mesh.faces, options)
}

/// Combine assembly export: one body per member, with the member's placement.
fn export_combine_assembly(
    input: &ExportInput,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let mut bodies = Vec::new();
    for member in &input.combine_members {
        let chain =
            member.chain.as_ref().ok_or("assembly export requires STEP import or Extrude parts")?;
        let brep = member.brep.as_ref().map_err(Clone::clone)?;
        let record = nodecraft_step::record_of_brep(brep)?;
        let moved = chain.offset != [0.0; 3] || chain.rotation != [0.0; 3];
        bodies.push(nodecraft_step::ExportBody {
            name: format!("Part {}", member.index),
            color: None,
            placement: moved.then(|| crate::placement_of(chain.offset, chain.rotation)),
            sheet: false,
            brep: record,
        });
    }
    if bodies.len() < 2 {
        return Err("assembly export needs at least two parts".into());
    }
    nodecraft_step::write_step_with_options(path, &bodies, options)
}

/// Imported STEP re-export: keep the original CADIR bodies (names, colours,
/// placements) and apply the resolved translate chain.
fn export_imported_step(
    input: &ExportInput,
    path: &str,
    options: &nodecraft_step::StepExportOptions,
) -> Result<(), String> {
    let Some(ExportChain { node: importer, offset, rotation, imported_step: true }) = &input.chain
    else {
        return Err("no STEP import in chain".into());
    };
    let source = importer.values.get("path").and_then(serde_json::Value::as_str).unwrap_or("");
    if source.is_empty() {
        return Err("STEP import has no path".into());
    }
    let bytes = std::fs::read(source).map_err(|error| format!("{source}: {error}"))?;
    let import = nodecraft_step::read_cad(&bytes, source)?;
    let mut bodies = Vec::with_capacity(import.bodies.len());
    for mut body in import.bodies {
        let moved = *offset != [0.0; 3] || *rotation != [0.0; 3];
        if moved {
            let placement = body.placement.map(|placement| {
                crate::compose_placement(crate::placement_of(*offset, *rotation), placement)
            });
            body.placement = placement.or_else(|| Some(crate::placement_of(*offset, *rotation)));
        }
        bodies.push(nodecraft_step::ExportBody {
            name: body.name,
            color: body.color,
            placement: body.placement,
            sheet: body.sheet,
            brep: body.brep,
        });
    }
    if bodies.is_empty() {
        return Err(format!("{source}: no STEP bodies to export"));
    }
    nodecraft_step::write_step_with_options(path, &bodies, options)
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::{Node, Request};
    use nodecraft_solidcore::solidcore::topology::Brep;

    use super::*;

    fn box_part() -> crate::Part {
        let node = Node {
            id: "b".into(),
            node_type: "primitive".into(),
            values: serde_json::json!({
                "primitiveType": "box",
                "length": 10,
                "width": 10,
                "height": 10
            }),
            suppressed: false,
            cache: None,
        };
        let brep = crate::nodes::primitive_brep(&node).expect("box brep");
        crate::Part::from_brep(brep)
    }

    fn exact_export_input(breps: Result<Vec<Brep>, String>) -> ExportInput {
        ExportInput {
            part: Ok(std::sync::Arc::new(box_part())),
            source_type: "primitive".into(),
            combine: false,
            chain: None,
            combine_members: Vec::new(),
            breps,
        }
    }

    /// Exactness is authoritative: when the exact writer refuses a body the
    /// exporter reports that error and writes nothing. The old fallback
    /// tessellated the Part and wrote a faceted STEP file instead.
    #[test]
    fn exact_write_failure_is_an_error_and_writes_no_file() {
        let directory =
            std::env::temp_dir().join(format!("sn-export-exact-failure-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("create output directory");
        let path = directory.join("exact.step");
        let path_text = path.to_str().expect("utf-8 path");
        let input = exact_export_input(Ok(vec![Brep::default()]));

        let error =
            export_exact_or_part(&input, path_text, &nodecraft_step::StepExportOptions::default())
                .expect_err("an exact writer failure must not fall back to faceted output");
        let error = stepio_error(path_text, error);
        assert!(
            error.to_ascii_lowercase().contains("step")
                || error.to_ascii_lowercase().contains("geometry"),
            "error should name the exact writer failure: {error}"
        );
        assert!(error.starts_with(path_text), "error should carry the file path: {error}");
        assert!(!path.exists(), "exact-write failure wrote a faceted file: {error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// A Part with no exact B-rep bodies is an error, not a license to
    /// tessellate: the old code silently wrote a faceted file.
    #[test]
    fn no_exact_breps_is_an_error_and_writes_no_file() {
        let directory =
            std::env::temp_dir().join(format!("sn-export-no-breps-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("create output directory");
        let path = directory.join("exact.step");
        let path_text = path.to_str().expect("utf-8 path");
        let input = exact_export_input(Ok(Vec::new()));

        let error =
            export_exact_or_part(&input, path_text, &nodecraft_step::StepExportOptions::default())
                .expect_err("no exact bodies must not fall back to faceted output");
        assert!(
            error.contains("no exact B-rep bodies"),
            "error should explain that no exact bodies exist: {error}"
        );
        assert!(!path.exists(), "missing exact breps wrote a faceted file: {error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    fn export_request(directory: &Path, run_exporters: bool, targets: &[&str]) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": targets,
            "meshTargets": [],
            "runExporters": run_exporters,
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "b"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "once"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "b", "to": "e", "in": "shape", "out": "shape"}]
        }))
        .expect("export request")
    }

    /// Removing the registry entry makes the engine reject `exporter`.
    #[test]
    fn registry_routes_exporter() {
        assert!(crate::nodes::handles("exporter"));
        let builtin = crate::nodes::find("exporter").expect("exporter registered");
        assert_eq!(builtin.node_types(), &["exporter"]);
        assert_eq!(builtin.output_kind(), OutputKind::Effect);
    }

    /// A normal export run writes the file once, even when the exporter's input
    /// is also a target and the display pass also looks at the exporter.
    #[test]
    fn exporter_writes_once_per_run() {
        let directory = std::env::temp_dir().join(format!("sn-export-once-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let output = crate::evaluate_exact(
            export_request(&directory, true, &["b", "e"]),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(directory.join("once.step").is_file(), "export file missing");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// Validation/ghost passes set `runExporters = false`; they must never
    /// write a file.
    #[test]
    fn validation_does_not_write_an_export_file() {
        assert!(crate::nodes::handles("exporter"), "exporter registered");
        let directory =
            std::env::temp_dir().join(format!("sn-export-validation-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let output = crate::evaluate_exact(
            export_request(&directory, false, &["e"]),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        assert!(!directory.join("once.step").exists(), "validation wrote a file");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// An errored input is displayable through bypass but is not a
    /// successful export: no file and the exporter carries the input's error.
    #[test]
    fn an_errored_input_does_not_write() {
        let directory =
            std::env::temp_dir().join(format!("sn-export-errored-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "runExporters": true,
            "nodes": [
                {"id": "i", "type": "stepimporter", "values": {"path": "/nonexistent/batch5.step"}, "cache": {"key": "i"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "once"}, "cache": {"key": "e"}}
            ],
            "connections": [{"from": "i", "to": "e", "in": "shape", "out": "shape"}]
        }))
        .expect("errored export request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        assert!(!directory.join("once.step").exists(), "errored input wrote a file");
        assert!(
            output.response.node_errors.get("e").is_some_and(|error| error.contains("batch5.step")),
            "{:?}",
            output.response.node_errors,
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The effect boundary reads *every* incoming connection, not a port named
    /// `shape`: an errored input on any port blocks the write and its error is
    /// recorded on the effect node. The valid `shape` input alone would write.
    #[test]
    fn an_errored_input_on_another_port_does_not_write() {
        let directory =
            std::env::temp_dir().join(format!("sn-export-other-port-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let request: Request = serde_json::from_value(serde_json::json!({
            "requestId": 1,
            "targets": ["e"],
            "meshTargets": [],
            "runExporters": true,
            "nodes": [
                {"id": "b", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "b"}},
                {"id": "i", "type": "stepimporter", "values": {"path": "/nonexistent/port-agnostic.step"}, "cache": {"key": "i"}},
                {"id": "e", "type": "exporter", "values": {"directory": directory.to_string_lossy(), "fileName": "once"}, "cache": {"key": "e"}}
            ],
            "connections": [
                {"from": "b", "to": "e", "in": "shape", "out": "shape"},
                {"from": "i", "to": "e", "in": "payload", "out": "shape"}
            ]
        }))
        .expect("port-agnostic export request");
        let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
        assert!(!directory.join("once.step").exists(), "an errored non-shape input wrote a file");
        assert!(
            output
                .response
                .node_errors
                .get("e")
                .is_some_and(|error| error.contains("port-agnostic.step")),
            "{:?}",
            output.response.node_errors,
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// A cancelled revision never writes a stale file, even with
    /// `runExporters = true`.
    #[test]
    fn a_cancelled_revision_writes_nothing() {
        assert!(crate::nodes::handles("exporter"), "exporter registered");
        let directory =
            std::env::temp_dir().join(format!("sn-export-cancelled-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let token = crate::kernel_pool::RevisionToken::new();
        token.cancel();
        crate::kernel_pool::set_current_revision_token(token);
        let output = crate::evaluate_exact(
            export_request(&directory, true, &["e"]),
            crate::EngineCache::new(),
            None,
        );
        assert!(!directory.join("once.step").exists(), "cancelled revision wrote a file");
        assert!(!output.response.node_errors.contains_key("e"));
        crate::kernel_pool::set_current_revision_token(crate::kernel_pool::RevisionToken::new());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
