//! Shell: hollow every body of an exact Part by a signed wall thickness.
//!
//! The kernel operation stays `node_task::ExactNodeTask::Shell`
//! (`OpsSession::shell_body`). This module owns the node-shaped parts: the
//! shape input, the multi-body order, positive-thickness validation, the
//! `face:N` / `body:B:face:N` opening references, per-body opening filtering
//! and the display body layout.
//!
//! References go through the shared `face_ref` resolver: an unscoped `face:N`
//! opens face `N` on every body that has it (the existing multi-body meaning),
//! and a body-scoped `body:B:face:N` opens it on body `B` alone. An opening no
//! body has, a body that is not in the Part, a malformed entry and an opening
//! that would remove every face are all errors before geometry.

use super::{BuiltinNode, NodeView, PartEval};
use crate::face_ref::{parse_face_references, resolve_face_references};
use crate::node_task::ExactNodeTask;
use crate::{Part, RequestIndex};

/// Registered for the `shell` contract id.
pub(crate) struct Shell;

impl BuiltinNode for Shell {
    fn node_types(&self) -> &'static [&'static str] {
        &["shell"]
    }

    fn evaluate(&self, node: NodeView<'_>, ctx: &mut dyn PartEval) -> Result<Part, String> {
        let Some(source) = ctx.input_part(&["shape"])? else {
            return Err("Shell has no shape input".into());
        };
        let solids = source.realized_bodies()?;
        if solids.is_empty() {
            return Err("Shell source Part is empty".into());
        }
        let thickness = crate::contract_number("shell", node.values, "thickness");
        if !(thickness.is_finite() && thickness > 0.0) {
            return Err("Shell thickness must be positive".into());
        }
        // The contract owns the direction options. One shared resolver accepts
        // the canonical values case-insensitively and refuses anything else, so
        // a misspelling can never mean "not inward" (outward).
        let direction =
            nodecraft_node_contract::resolve_choice_value("shell", node.values, "direction")
                .map_err(|error| format!("Shell: {error}"))?;
        let signed = match direction {
            "inward" => -thickness,
            "outward" => thickness,
            other => {
                return Err(format!(
                    "Shell direction {other:?} is not supported; expected inward or outward."
                ));
            }
        };
        // Parse and resolve every opening before touching geometry. A malformed
        // or unresolvable reference is an error; silently dropping one would
        // turn the requested shell into a closed shell of the wrong shape.
        let references = parse_face_references(node.values)?;
        let body_faces: Vec<usize> = solids
            .iter()
            .map(|solid| solid.brep().faces().ids().count())
            .collect();
        let openings = resolve_face_references(&references, &body_faces)?;
        for (body, count) in body_faces.iter().enumerate() {
            if *count > 0 && openings[body].len() == *count {
                return Err(
                    "Can't remove every face — keep at least one face to make a wall.".to_owned(),
                );
            }
        }
        let mut bodies = Vec::with_capacity(solids.len());
        for (solid, openings) in solids.into_iter().zip(openings) {
            let result = ctx.execute_warned(ExactNodeTask::Shell {
                body: solid,
                signed_thickness: signed,
                openings,
            })?;
            let crate::node_task::TaskResult { output, warnings } = result;
            ctx.warn(warnings);
            let shelled = output
                .into_shape()
                .ok_or_else(|| "Shell task returned no shape".to_owned())?;
            bodies.push(crate::BodyInstance {
                body: shelled,
                placement: nodecraft_exact::Placement::identity(),
                realized: Default::default(),
                source: None,
                source_body: 0,
                name: None,
                color: None,
            });
        }
        Ok(Part { bodies, sketches: Vec::new() })
    }

    fn exact_weight(&self, _node: NodeView<'_>) -> usize {
        12
    }

    fn suppression_source(&self, node: NodeView<'_>, index: &RequestIndex<'_>) -> Option<String> {
        index
            .first_incoming_any(&node.id, &["shape", "profile"])
            .map(|link| link.from.clone())
    }

    fn body_layout(
        &self,
        _node: NodeView<'_>,
        inputs: &[(&str, Vec<String>)],
    ) -> Option<Vec<String>> {
        // One shelled body per input body, in the same order.
        super::input_layout(inputs, "shape").map(<[String]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use nodecraft_protocol::Request;

    fn link(from: &str, to: &str) -> serde_json::Value {
        serde_json::json!({"from": from, "to": to, "in": "shape", "out": "shape"})
    }

    fn two_box_request_with(
        faces: serde_json::Value,
        thickness: f64,
        direction: &str,
    ) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "big", "type": "primitive", "cache": {"key": "big"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "small", "type": "primitive", "cache": {"key": "small"},
                 "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}},
                {"id": "move", "type": "translate", "cache": {"key": "move"},
                 "values": {"x": 30.0}},
                {"id": "part", "type": "combine", "cache": {"key": "part"}, "values": {}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": direction, "faces": faces}}
            ],
            "connections": [
                link("big", "part"),
                link("small", "move"),
                link("move", "part"),
                link("part", "shell")
            ]
        }))
        .expect("shell request")
    }

    fn two_box_request(faces: serde_json::Value, thickness: f64) -> Request {
        two_box_request_with(faces, thickness, "inward")
    }

    fn single_box_request_with(
        faces: serde_json::Value,
        thickness: f64,
        direction: &str,
    ) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 2, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": direction, "faces": faces}}
            ],
            "connections": [link("box", "shell")]
        }))
        .expect("shell request")
    }

    fn single_box_request(faces: serde_json::Value, thickness: f64) -> Request {
        single_box_request_with(faces, thickness, "inward")
    }

    fn cylinder_request(
        faces: serde_json::Value,
        thickness: f64,
        direction: &str,
    ) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 3, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "cylinder", "type": "primitive", "cache": {"key": "cylinder"},
                 "values": {"primitiveType": "cylinder", "radius": 10.0, "height": 20.0}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": direction, "faces": faces}}
            ],
            "connections": [link("cylinder", "shell")]
        }))
        .expect("shell request")
    }

    /// A 20 mm box with four edges filleted at r=2, then shelled. The fillet
    /// face order is the record order the viewport picker writes.
    fn filleted_box_request(
        shell_faces: serde_json::Value,
        thickness: f64,
        direction: &str,
    ) -> Request {
        serde_json::from_value(serde_json::json!({
            "requestId": 4, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "cache": {"key": "box"},
                 "values": {"primitiveType": "box", "length": 20, "width": 20, "height": 20}},
                {"id": "fillet", "type": "fillet", "cache": {"key": "fillet"},
                 "values": {"continuity": "", "length": 2.0, "tangentChain": false,
                            "edges": [{"ordinal": 0}, {"ordinal": 1}, {"ordinal": 2}, {"ordinal": 3}]}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": direction, "faces": shell_faces}}
            ],
            "connections": [link("box", "fillet"), link("fillet", "shell")]
        }))
        .expect("shell request")
    }

    fn step_import_request(
        relative: &str,
        faces: serde_json::Value,
        thickness: f64,
        direction: &str,
    ) -> Request {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative);
        serde_json::from_value(serde_json::json!({
            "requestId": 5, "targets": ["shell"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "import", "type": "stepimporter", "cache": {"key": "import"},
                 "values": {"path": path.to_string_lossy(), "heal": true}},
                {"id": "shell", "type": "shell", "cache": {"key": "shell"},
                 "values": {"thickness": thickness, "direction": direction, "faces": faces}}
            ],
            "connections": [link("import", "shell")]
        }))
        .expect("shell request")
    }

    fn node_error(output: &crate::EngineOutput) -> String {
        output
            .response
            .node_errors
            .get("shell")
            .cloned()
            .unwrap_or_default()
    }

    fn single_body_volume(output: &crate::EngineOutput) -> f64 {
        let part = output.exact_parts.get("shell").expect("shell exact part");
        assert_eq!(part.bodies.len(), 1);
        volume(&part.bodies[0].brep)
    }

    fn volume(brep: &nodecraft_solidcore::solidcore::topology::Brep) -> f64 {
        let record = nodecraft_step::record_of_brep(brep).expect("brep record");
        nodecraft_step::record_volume(&record).expect("brep volume")
    }

    fn bounds(part: &nodecraft_exact::ExactPart, index: usize) -> ([f64; 3], [f64; 3]) {
        let brep = part.realized_breps().expect("realized body");
        let record = nodecraft_step::record_of_brep(&brep[index]).expect("body record");
        let (min, max) = nodecraft_step::record_bounds(&record).expect("body bounds");
        ([min[0], min[1], min[2]], [max[0], max[1], max[2]])
    }

    /// Removing the registry entry makes the engine reject `shell`, so this
    /// test is the entry's fail proof. It also pins the moved cost estimate.
    #[test]
    fn registry_routes_shell() {
        let builtin = crate::nodes::find("shell").expect("shell registered");
        assert_eq!(builtin.node_types(), &["shell"]);
        let values = serde_json::Value::Null;
        assert_eq!(
            builtin.exact_weight(crate::nodes::NodeView::new("s", "shell", &values, false)),
            12
        );
    }

    /// Every input body is shelled, in Part body order. The first box stays
    /// around the origin and the second stays at x=30, so both the bounds and
    /// the larger first volume prove the order.
    #[test]
    fn multi_body_shell_keeps_the_source_body_order() {
        let output = crate::evaluate_exact(
            two_box_request(serde_json::json!([]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("shell").expect("shell exact part");
        assert_eq!(part.bodies.len(), 2, "one shelled body per input body");
        let (first_min, first_max) = bounds(part, 0);
        let (second_min, second_max) = bounds(part, 1);
        assert!(first_min[0] < second_min[0], "{first_min:?} vs {second_min:?}");
        assert!((first_min[0] + 10.0).abs() < 1.0e-6, "{first_min:?}");
        assert!((first_max[0] - 10.0).abs() < 1.0e-6, "{first_max:?}");
        assert!((second_min[0] - 25.0).abs() < 1.0e-6, "{second_min:?}");
        assert!((second_max[0] - 35.0).abs() < 1.0e-6, "{second_max:?}");
        let first_volume = volume(&part.bodies[0].brep);
        let second_volume = volume(&part.bodies[1].brep);
        assert!(first_volume > second_volume, "{first_volume} vs {second_volume}");
        assert!(first_volume < 8_000.0, "the big box was shelled: {first_volume}");
        assert!(second_volume < 1_000.0, "the small box was shelled: {second_volume}");
    }

    /// A shell needs a positive, finite wall thickness; zero and negative
    /// values are errors instead of a closed or inverted body.
    #[test]
    fn shell_rejects_non_positive_thickness() {
        for thickness in [0.0, -2.0] {
            let output = crate::evaluate_exact(
                single_box_request(serde_json::json!([]), thickness),
                crate::EngineCache::new(),
                None,
            );
            let error = output.response.node_errors.get("shell").expect("thickness error");
            assert!(error.contains("thickness must be positive"), "{thickness}: {error}");
        }
    }

    /// Both opening reference forms resolve: `face:N` and `body:B:face:N`.
    #[test]
    fn shell_accepts_face_and_body_scoped_opening_references() {
        for reference in ["face:1", "body:0:face:1"] {
            let output = crate::evaluate_exact(
                single_box_request(serde_json::json!([{"id": reference}]), 1.0),
                crate::EngineCache::new(),
                None,
            );
            assert!(
                output.response.node_errors.is_empty(),
                "{reference}: {:?}",
                output.response.node_errors
            );
            let part = output.exact_parts.get("shell").expect("shell exact part");
            assert_eq!(part.bodies.len(), 1);
            assert!(volume(&part.bodies[0].brep) > 0.0);
        }
    }

    /// A body-scoped opening only opens that body. Body 0 must match a closed
    /// shell of the same source; body 1 must lose the open face's wall.
    #[test]
    fn body_scoped_openings_are_filtered_per_body() {
        let closed = crate::evaluate_exact(
            two_box_request(serde_json::json!([]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        let opened = crate::evaluate_exact(
            two_box_request(serde_json::json!([{"id": "body:1:face:1"}]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(closed.response.node_errors.is_empty(), "{:?}", closed.response.node_errors);
        assert!(opened.response.node_errors.is_empty(), "{:?}", opened.response.node_errors);
        let closed = closed.exact_parts.get("shell").expect("closed shell part");
        let opened = opened.exact_parts.get("shell").expect("opened shell part");
        let closed_first = volume(&closed.bodies[0].brep);
        let opened_first = volume(&opened.bodies[0].brep);
        assert!(
            (closed_first - opened_first).abs() < 1.0e-6,
            "body 0 must stay closed: {closed_first} vs {opened_first}"
        );
        let closed_second = volume(&closed.bodies[1].brep);
        let opened_second = volume(&opened.bodies[1].brep);
        assert!(
            closed_second - opened_second > 1.0,
            "body 1 must lose the open face's wall: {closed_second} vs {opened_second}"
        );
    }

    /// A shell keeps the source display layout: one body in, one body out, in
    /// the same order.
    #[test]
    fn body_layout_is_the_source_layout() {
        let builtin = crate::nodes::find("shell").expect("shell registered");
        let values = serde_json::json!({});
        let view = crate::nodes::NodeView::new("shell", "shell", &values, false);
        assert_eq!(
            builtin.body_layout(view, &[("shape", vec!["big".to_owned(), "small".to_owned()])]),
            Some(vec!["big".to_owned(), "small".to_owned()]),
        );
    }

    /// Every saved opening must resolve. A face beyond the body, a non-string
    /// id, an empty entry, a `faces` value that is not a list and a malformed
    /// body prefix are all errors before geometry; none may become a closed
    /// shell of the wrong shape.
    #[test]
    fn shell_rejects_unresolved_malformed_and_non_list_openings() {
        let cases = [
            ("face beyond the body", serde_json::json!([{"id": "face:99"}]), "face:99"),
            ("first missing face", serde_json::json!([{"id": "face:6"}]), "face:6"),
            ("later missing face", serde_json::json!([{"id": "face:7"}]), "face:7"),
            ("non-string id", serde_json::json!([{"id": 5}]), "entry 1"),
            ("missing id", serde_json::json!([{}]), "entry 1"),
            ("faces is not a list", serde_json::json!({"id": "face:0"}), "list"),
            ("malformed body prefix", serde_json::json!([{"id": "body:x:face:1"}]), "body:x:face:1"),
        ];
        for (case, faces, expected) in cases {
            let output = crate::evaluate_exact(
                single_box_request(faces, 1.0),
                crate::EngineCache::new(),
                None,
            );
            let error = node_error(&output);
            assert!(!error.is_empty(), "{case}: the opening was silently ignored");
            assert!(error.contains(expected), "{case}: {error}");
            assert!(
                error.contains("Reselect the faces to remove"),
                "{case}: {error}"
            );
        }
    }

    /// A body-scoped reference applies to its body alone; a body that is not in
    /// the Part, including a malformed body prefix, is an error rather than a
    /// downgrade to a global face.
    #[test]
    fn shell_body_scoped_references_only_open_their_body_and_reject_unknown_bodies() {
        for (case, reference) in [
            ("malformed body", "body:x:face:1"),
            ("unknown body", "body:9:face:1"),
        ] {
            let output = crate::evaluate_exact(
                single_box_request(serde_json::json!([{"id": reference}]), 1.0),
                crate::EngineCache::new(),
                None,
            );
            let error = node_error(&output);
            assert!(!error.is_empty(), "{case}: no node error");
            assert!(error.contains(reference), "{case}: {error}");
        }

        // An unscoped face applies to every body that has it, as the existing
        // multi-body tests pin; a body-scoped one applies to that body alone.
        let closed = crate::evaluate_exact(
            two_box_request(serde_json::json!([]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        let global = crate::evaluate_exact(
            two_box_request(serde_json::json!([{"id": "face:1"}]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        let scoped = crate::evaluate_exact(
            two_box_request(serde_json::json!([{"id": "body:1:face:1"}]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(closed.response.node_errors.is_empty(), "{:?}", closed.response.node_errors);
        assert!(global.response.node_errors.is_empty(), "{:?}", global.response.node_errors);
        assert!(scoped.response.node_errors.is_empty(), "{:?}", scoped.response.node_errors);
        let closed = closed.exact_parts.get("shell").expect("closed shell part");
        let global = global.exact_parts.get("shell").expect("global shell part");
        let scoped = scoped.exact_parts.get("shell").expect("scoped shell part");
        for body in 0..2 {
            let closed_volume = volume(&closed.bodies[body].brep);
            let global_volume = volume(&global.bodies[body].brep);
            assert!(
                closed_volume - global_volume > 1.0,
                "body {body} must lose the global face's wall: {closed_volume} vs {global_volume}"
            );
        }
        let closed_first = volume(&closed.bodies[0].brep);
        let scoped_first = volume(&scoped.bodies[0].brep);
        assert!(
            (closed_first - scoped_first).abs() < 1.0e-6,
            "body 0 must stay closed for body:1:face:1: {closed_first} vs {scoped_first}"
        );
        let closed_second = volume(&closed.bodies[1].brep);
        let scoped_second = volume(&scoped.bodies[1].brep);
        assert!(
            closed_second - scoped_second > 1.0,
            "body 1 must lose the scoped face's wall: {closed_second} vs {scoped_second}"
        );
    }

    /// The contract's canonical direction values win, matched case-insensitively.
    /// A value outside the contract is an error, never a silent outward shell.
    #[test]
    fn shell_direction_accepts_canonical_values_case_insensitively_and_rejects_unknown() {
        let inward = crate::evaluate_exact(
            single_box_request_with(serde_json::json!([]), 1.0, "inward"),
            crate::EngineCache::new(),
            None,
        );
        let outward = crate::evaluate_exact(
            single_box_request_with(serde_json::json!([]), 1.0, "outward"),
            crate::EngineCache::new(),
            None,
        );
        assert!(inward.response.node_errors.is_empty(), "{:?}", inward.response.node_errors);
        assert!(outward.response.node_errors.is_empty(), "{:?}", outward.response.node_errors);
        let inward_volume = single_body_volume(&inward);
        let outward_volume = single_body_volume(&outward);
        assert!((inward_volume - 2168.0).abs() < 1.0e-6, "{inward_volume}");
        assert!((outward_volume - 2648.0).abs() < 1.0e-6, "{outward_volume}");

        for direction in ["Inward", "INWARD"] {
            let output = crate::evaluate_exact(
                single_box_request_with(serde_json::json!([]), 1.0, direction),
                crate::EngineCache::new(),
                None,
            );
            assert!(
                output.response.node_errors.is_empty(),
                "{direction}: {:?}",
                output.response.node_errors
            );
            let volume = single_body_volume(&output);
            assert!(
                (volume - inward_volume).abs() < 1.0e-6,
                "{direction} must mean inward: {volume} vs {inward_volume}"
            );
        }
        for direction in ["Outward", "OUTWARD"] {
            let output = crate::evaluate_exact(
                single_box_request_with(serde_json::json!([]), 1.0, direction),
                crate::EngineCache::new(),
                None,
            );
            assert!(
                output.response.node_errors.is_empty(),
                "{direction}: {:?}",
                output.response.node_errors
            );
            let volume = single_body_volume(&output);
            assert!(
                (volume - outward_volume).abs() < 1.0e-6,
                "{direction} must mean outward: {volume} vs {outward_volume}"
            );
        }
        for direction in ["inwrad", "outside"] {
            let output = crate::evaluate_exact(
                single_box_request_with(serde_json::json!([]), 1.0, direction),
                crate::EngineCache::new(),
                None,
            );
            let error = node_error(&output);
            assert!(!error.is_empty(), "{direction}: no node error");
            assert!(error.contains("direction"), "{direction}: {error}");
            assert!(error.contains("inward"), "{direction}: {error}");
            assert!(error.contains("outward"), "{direction}: {error}");
        }
    }

    /// An inward wall thicker than the part is a thickness refusal, not a
    /// fillet's "blend fewer edges" advice.
    #[test]
    fn shell_too_thick_reports_shell_not_blend() {
        let cases = [
            (
                "box 50 inward",
                single_box_request(serde_json::json!([]), 50.0),
            ),
            (
                "cylinder 10.1 inward",
                cylinder_request(serde_json::json!([]), 10.1, "inward"),
            ),
        ];
        for (case, request) in cases {
            let output = crate::evaluate_exact(request, crate::EngineCache::new(), None);
            let error = node_error(&output);
            assert!(!error.is_empty(), "{case}: no node error");
            let lower = error.to_lowercase();
            assert!(
                lower.contains("wall is thicker") || lower.contains("too thick"),
                "{case}: {error}"
            );
            assert!(!lower.contains("blend"), "{case}: blend wording: {error}");
            assert!(!lower.contains("edges"), "{case}: edge wording: {error}");
        }
    }

    /// Opening a face on a filleted body must be actionable: no raw kernel
    /// status, and any refusal names the shell or the face. Every record face
    /// is tried because the picker can reach all of them.
    #[test]
    fn shell_opening_on_a_filleted_face_is_actionable_and_never_a_raw_status() {
        let closed = crate::evaluate_exact(
            filleted_box_request(serde_json::json!([]), 1.0, "inward"),
            crate::EngineCache::new(),
            None,
        );
        assert!(closed.response.node_errors.is_empty(), "{:?}", closed.response.node_errors);
        let fillet = closed.exact_parts.get("fillet").expect("fillet exact part");
        let face_count = fillet.bodies[0].brep.faces().ids().count();
        assert!(face_count > 6, "the fillet must add faces: {face_count}");
        let mut refusals = 0;
        for face in 0..face_count {
            for direction in ["inward", "outward"] {
                let output = crate::evaluate_exact(
                    filleted_box_request(
                        serde_json::json!([{"id": format!("face:{face}")}]),
                        1.0,
                        direction,
                    ),
                    crate::EngineCache::new(),
                    None,
                );
                if let Some(error) = output.response.node_errors.get("shell") {
                    refusals += 1;
                    let lower = error.to_lowercase();
                    assert!(
                        !lower.contains("kernel status") && !lower.contains("status 100"),
                        "face {face} {direction}: raw status: {error}"
                    );
                    assert!(
                        lower.contains("shell")
                            || lower.contains("wall")
                            || lower.contains("face"),
                        "face {face} {direction}: no actionable advice: {error}"
                    );
                } else {
                    let volume = single_body_volume(&output);
                    assert!(volume > 0.0, "face {face} {direction}: empty shell");
                }
            }
        }
        assert!(refusals > 0, "the filleted box should refuse at least one opening");
    }

    /// The STEP fixture's refusals must not suggest a 0.00 mm wall and must not
    /// leak a raw kernel status, for either thickness sign.
    #[test]
    fn shell_moon_buggy_refusals_are_actionable_and_never_suggest_zero() {
        const MOON_BUGGY: &str = "tests/Sample AP203 Data Files/moon_buggy.stp";
        for (thickness, direction) in [(1.0, "inward"), (0.001, "inward"), (1.0, "outward")] {
            let output = crate::evaluate_exact(
                step_import_request(MOON_BUGGY, serde_json::json!([]), thickness, direction),
                crate::EngineCache::new(),
                None,
            );
            let error = node_error(&output);
            assert!(!error.is_empty(), "{thickness} {direction}: no node error");
            let lower = error.to_lowercase();
            assert!(!error.contains("0.00 mm"), "{thickness} {direction}: {error}");
            assert!(
                !lower.contains("kernel status") && !lower.contains("status 100"),
                "{thickness} {direction}: raw status: {error}"
            );
            assert!(
                lower.contains("shell") || lower.contains("wall"),
                "{thickness} {direction}: not shell wording: {error}"
            );
        }
    }

    /// The same face twice is the same opening: duplicates are de-duplicated
    /// before the kernel, both in the unscoped and the body-scoped form.
    #[test]
    fn shell_duplicate_openings_are_one_opening() {
        let single = crate::evaluate_exact(
            single_box_request(serde_json::json!([{"id": "face:0"}]), 1.0),
            crate::EngineCache::new(),
            None,
        );
        assert!(single.response.node_errors.is_empty(), "{:?}", single.response.node_errors);
        let single_volume = single_body_volume(&single);
        for (case, faces) in [
            (
                "the same face twice",
                serde_json::json!([{"id": "face:0"}, {"id": "face:0"}]),
            ),
            (
                "scoped and unscoped",
                serde_json::json!([{"id": "face:0"}, {"id": "body:0:face:0"}]),
            ),
        ] {
            let output = crate::evaluate_exact(
                single_box_request(faces, 1.0),
                crate::EngineCache::new(),
                None,
            );
            assert!(
                output.response.node_errors.is_empty(),
                "{case}: {:?}",
                output.response.node_errors
            );
            let volume = single_body_volume(&output);
            assert!(
                (volume - single_volume).abs() < 1.0e-6,
                "{case}: {volume} vs {single_volume}"
            );
        }
    }

    /// Opening every face leaves no wall. That is a selection mistake, not an
    /// empty result or a size problem.
    #[test]
    fn shell_rejects_opening_every_face() {
        let faces = (0..6)
            .map(|face| serde_json::json!({"id": format!("face:{face}")}))
            .collect::<Vec<_>>();
        for direction in ["inward", "outward"] {
            let output = crate::evaluate_exact(
                single_box_request_with(serde_json::json!(faces), 1.0, direction),
                crate::EngineCache::new(),
                None,
            );
            let error = node_error(&output);
            assert!(!error.is_empty(), "{direction}: no node error");
            assert!(
                error.contains("Can't remove every face"),
                "{direction}: {error}"
            );
            assert!(
                error.contains("keep at least one face to make a wall"),
                "{direction}: {error}"
            );
            assert!(!error.contains("result is empty"), "{direction}: {error}");
        }
    }

    /// Both signs are the same wall on opposite sides of the surface.
    #[test]
    fn shell_inward_and_outward_volumes_differ_by_the_wall() {
        let inward = crate::evaluate_exact(
            single_box_request_with(serde_json::json!([]), 1.0, "inward"),
            crate::EngineCache::new(),
            None,
        );
        let outward = crate::evaluate_exact(
            single_box_request_with(serde_json::json!([]), 1.0, "outward"),
            crate::EngineCache::new(),
            None,
        );
        assert!(inward.response.node_errors.is_empty(), "{:?}", inward.response.node_errors);
        assert!(outward.response.node_errors.is_empty(), "{:?}", outward.response.node_errors);
        let inward_volume = single_body_volume(&inward);
        let outward_volume = single_body_volume(&outward);
        assert!((inward_volume - 2168.0).abs() < 1.0e-6, "{inward_volume}");
        assert!((outward_volume - 2648.0).abs() < 1.0e-6, "{outward_volume}");
    }
}
