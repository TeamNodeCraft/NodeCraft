//! Affine placements for graph control points.
//!
//! A node's values are expressed in its parent frame; controls displayed at a
//! downstream breakpoint must be placed through every downstream transform and
//! edits must be mapped back through the inverse chain.

use nodecraft_protocol::Connection;

use crate::{GraphDocument, GraphNode};

/// Read a contract field through the contract resolver. The panic path is a
/// programming error: every call below names a field the contract defines.
fn contract_number(node_type: &str, values: &serde_json::Value, key: &str) -> f64 {
    nodecraft_node_contract::resolve_number(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no number field {node_type}.{key}"))
}

fn contract_text<'a>(
    node_type: &str,
    values: &'a serde_json::Value,
    key: &str,
) -> std::borrow::Cow<'a, str> {
    nodecraft_node_contract::resolve_text(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no text field {node_type}.{key}"))
}

/// The contract-resolved fields of a Translate node.
///
/// Geometry placement (`Affine3::from_translate`), the viewport gizmo, and the
/// edit ghost all read these six numbers through this one helper. Keeping the
/// resolution here means a missing field takes the same contract default in the
/// viewport that it takes in the engine and session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslateValues {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub rx: f64,
    pub ry: f64,
    pub rz: f64,
}

/// Resolve a Translate node's fields through the node contract.
#[must_use]
pub fn translate_values(values: &serde_json::Value) -> TranslateValues {
    TranslateValues {
        x: contract_number("translate", values, "x"),
        y: contract_number("translate", values, "y"),
        z: contract_number("translate", values, "z"),
        rx: contract_number("translate", values, "rx"),
        ry: contract_number("translate", values, "ry"),
        rz: contract_number("translate", values, "rz"),
    }
}

/// The gizmo's affine placement value.
///
/// A thin view over [`nodecraft_exact::Affine3`]: composition, inversion and
/// point mapping are the one implementation in `nodecraft-exact`, not a second
/// gizmo copy. The constructors below only resolve the node contract's fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3(nodecraft_exact::Affine3);

impl Affine3 {
    pub fn identity() -> Self {
        Self(nodecraft_exact::Affine3::IDENTITY)
    }

    /// Rotation only, matching the mesh/BRep rotation order.
    pub fn rotation(values: &serde_json::Value) -> Self {
        let translate = translate_values(values);
        Self(nodecraft_exact::Affine3::from_rotation([translate.rx, translate.ry, translate.rz]))
    }

    /// Translation after X/Y/Z rotation, matching the mesh and BRep transform paths.
    /// `space` only changes the manipulator frame; it never changes geometry.
    pub fn from_translate(values: &serde_json::Value) -> Self {
        let translate = translate_values(values);
        Self(nodecraft_exact::Affine3::from_place(
            [translate.x, translate.y, translate.z],
            [translate.rx, translate.ry, translate.rz],
        ))
    }

    pub fn from_mirror(values: &serde_json::Value) -> Self {
        let plane = contract_text("mirror", values, "plane");
        let offset = contract_number("mirror", values, "offset");
        Self(nodecraft_exact::Affine3::from_mirror(plane.as_ref(), offset))
    }

    pub fn from_node(node: &GraphNode) -> Option<Self> {
        let view = nodecraft_engine::nodes::NodeView::new(
            &node.id,
            &node.node_type,
            &node.values,
            node.suppressed,
        );
        match nodecraft_engine::nodes::transform_role(view)? {
            nodecraft_engine::nodes::TransformRole::Translate => {
                Some(Self::from_translate(&node.values))
            }
            nodecraft_engine::nodes::TransformRole::Mirror => Some(Self::from_mirror(&node.values)),
        }
    }

    /// `self` applied after `other`.
    pub fn compose(self, other: Self) -> Self {
        Self(self.0.compose(other.0))
    }

    pub fn apply_point(self, point: [f64; 3]) -> [f64; 3] {
        self.0.apply_point(point)
    }

    pub fn apply_vector(self, vector: [f64; 3]) -> [f64; 3] {
        self.0.apply_vector(vector)
    }

    pub fn inverse_linear(self) -> [[f64; 3]; 3] {
        self.0.inverse_linear()
    }

    /// Inverse of a rigid/reflected affine transform.
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }

    /// Column-major `[[f32; 4]; 4]` for GPU uploads (WGSL `mat4x4<f32>` layout).
    pub fn to_column_major_f32(self) -> [[f32; 4]; 4] {
        self.0.to_column_major_f32()
    }

    /// The shared placement value, for consumers that take it directly.
    pub fn placement(self) -> nodecraft_exact::Placement {
        nodecraft_exact::Placement::from_matrix(self.0)
    }
}

impl From<nodecraft_exact::Affine3> for Affine3 {
    fn from(matrix: nodecraft_exact::Affine3) -> Self {
        Self(matrix)
    }
}

impl From<Affine3> for nodecraft_exact::Affine3 {
    fn from(value: Affine3) -> Self {
        value.0
    }
}

/// Display bodies that an edit to `selected` moves at the viewed `target`.
///
/// The exact Part display path flattens a target's solids into one merged mesh
/// and names each one `body:{index}:` in Part body order (nested Combines do not
/// nest the names). This rebuilds that same flattened order and returns the
/// body prefixes downstream of `selected`:
///
/// - `None` when `selected` does not feed `target`, or when `target` merges the
///   edited branch with unrelated geometry and per-body provenance is lost,
/// - [`PreviewBodyScope::All`] when every displayed body is downstream,
/// - [`PreviewBodyScope::Prefixes`] for the subset that moves,
/// - [`PreviewBodyScope::Unknown`] when downstream merging destroyed the body
///   layout. Callers can still offer a branch/edit-point preview, but not a
///   whole-mesh transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreviewBodyScope {
    All,
    Prefixes(Vec<String>),
    Unknown,
}

pub fn preview_body_scope(
    document: &GraphDocument,
    selected: &str,
    target: &str,
) -> Option<PreviewBodyScope> {
    // Resolve the whole ancestor/descendant relation once. Walking the graph per
    // body marker is quadratic on assemblies with thousands of bodies.
    let mut ancestors = std::collections::HashSet::from([selected.to_owned()]);
    let mut stack = vec![selected.to_owned()];
    while let Some(id) = stack.pop() {
        for link in document.connections.iter().filter(|link| link.to == id) {
            if ancestors.insert(link.from.clone()) {
                stack.push(link.from.clone());
            }
        }
    }
    let mut descendants = std::collections::HashSet::from([selected.to_owned()]);
    let mut stack = vec![selected.to_owned()];
    while let Some(id) = stack.pop() {
        for link in document.connections.iter().filter(|link| link.from == id) {
            if descendants.insert(link.to.clone()) {
                stack.push(link.to.clone());
            }
        }
    }
    if !descendants.contains(target) {
        return None;
    }
    // The registry owns which nodes merge operands and how. The protocol
    // connection shape keeps the engine's declaration independent of the
    // session's document types.
    let registry_connections: Vec<Connection> = document
        .connections
        .iter()
        .map(|link| Connection {
            from: link.from.clone(),
            to: link.to.clone(),
            input: link.input.clone(),
            out: link.out.clone(),
        })
        .collect();
    if selected != target
        && let Some(node) = document.nodes.iter().find(|node| node.id == target)
    {
        let view = nodecraft_engine::nodes::NodeView::new(
            &node.id,
            &node.node_type,
            &node.values,
            node.suppressed,
        );
        // Asking about the target itself: only the selected upstream branch is
        // attributed to the edit, so a node that merges a second operand loses
        // provenance. A drag on one branch cannot move only its own geometry;
        // there is no display-scoped fast preview, so the caller lets evaluation
        // produce the new body.
        let affected = std::collections::HashSet::from([selected.to_owned()]);
        if nodecraft_engine::nodes::merge_provenance(view, &registry_connections, &affected)
            == nodecraft_engine::nodes::MergeProvenance::LosesProvenance
        {
            return None;
        }
    }
    let Some(layout) = part_body_layout(document, target) else {
        // Downstream merging or an unsupported display chain hid the body
        // layout. This is not the same as "all bodies are downstream": the
        // caller must not move the whole breakpoint mesh here.
        return Some(PreviewBodyScope::Unknown);
    };
    // A merge node downstream of the edit produced one body from both
    // operands. If any operand is outside the edit's downstream branch, that
    // body contains geometry the drag cannot attribute to the edit, so the
    // preview must not move it as a whole. The registry decides which nodes
    // merge and which ports carry the operands; the session only supplies the
    // affected set.
    let merge_loses_provenance = |marker: &str| -> bool {
        if marker == selected || !descendants.contains(marker) {
            return false;
        }
        let Some(node) = document.nodes.iter().find(|node| node.id == marker) else {
            return false;
        };
        let view = nodecraft_engine::nodes::NodeView::new(
            &node.id,
            &node.node_type,
            &node.values,
            node.suppressed,
        );
        nodecraft_engine::nodes::merge_provenance(view, &registry_connections, &descendants)
            == nodecraft_engine::nodes::MergeProvenance::LosesProvenance
    };
    let mut prefixes = Vec::new();
    let mut all_affected = !layout.is_empty();
    for (index, marker) in layout.iter().enumerate() {
        if merge_loses_provenance(marker) {
            return Some(PreviewBodyScope::Unknown);
        }
        if ancestors.contains(marker) || descendants.contains(marker) {
            prefixes.push(format!("body:{index}:"));
        } else {
            all_affected = false;
        }
    }
    if all_affected {
        Some(PreviewBodyScope::All)
    } else {
        Some(PreviewBodyScope::Prefixes(prefixes))
    }
}

/// Node ids of the display bodies the exact Part path produces at `target`, in
/// Part body order. `None` when that layout cannot be reproduced exactly.
fn part_body_layout(document: &GraphDocument, target: &str) -> Option<Vec<String>> {
    if !part_display_chain(document, target) {
        return None;
    }
    let node = document.nodes.iter().find(|node| node.id == target)?;
    if node.suppressed {
        return None;
    }
    if nodecraft_engine::nodes::handles(&node.node_type) {
        // The registry owns the layout rule. Pass every connected input port
        // with its upstream layout, in connection order, so Combine can flatten
        // its operands and one-input nodes can repeat or transform the source.
        let view = nodecraft_engine::nodes::NodeView::new(
            &node.id,
            &node.node_type,
            &node.values,
            node.suppressed,
        );
        let inputs = document
            .connections
            .iter()
            .filter(|link| link.to == target)
            .map(|link| Some((link.input.as_str(), part_body_layout(document, &link.from)?)))
            .collect::<Option<Vec<_>>>()?;
        return nodecraft_engine::nodes::body_layout(view, &inputs);
    }
    // Every Part display type is served by the registry.
    None
}

/// Whether the engine builds this target's display mesh on the exact Part path:
/// the target and every ancestor must be Part nodes.
fn part_display_chain(document: &GraphDocument, target: &str) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![target.to_owned()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let Some(node) = document.nodes.iter().find(|node| node.id == id) else {
            return false;
        };
        if node.suppressed || !nodecraft_engine::part_display_supported(&node.node_type) {
            return false;
        }
        for link in document.connections.iter().filter(|link| link.to == id) {
            stack.push(link.from.clone());
        }
    }
    true
}

/// Whether a node turns one input body into several displayed copies.
///
/// A drag preview has a single affine, so it cannot follow every copy. The
/// shared rule is to anchor at the node's input frame — copy 0, the untouched
/// source body — and let re-evaluation place all copies after the edit.
///
/// A non-combining Mirror is not a repeater: its only displayed body is the
/// reflected copy, so its placement still belongs in the chain.
fn repeats_input_bodies(node: &GraphNode) -> bool {
    let view = nodecraft_engine::nodes::NodeView::new(
        &node.id,
        &node.node_type,
        &node.values,
        node.suppressed,
    );
    nodecraft_engine::nodes::repeats_input_bodies(view)
}

/// World-space transform that carries display geometry evaluated with the
/// transform node's `start` values to the same geometry under its `current`
/// values.
///
/// `chain` is the downstream placement chain from the edited node to the
/// displayed target (see [`transform_chain`]), so
/// `chain ∘ from_translate(values)` is the node's placement in target space.
/// The result maps the old placement onto the new one, which is exactly what a
/// drag preview needs before the kernel re-evaluates the real geometry.
pub fn preview_transform(
    chain: Affine3,
    start: &serde_json::Value,
    current: &serde_json::Value,
) -> Affine3 {
    let start = chain.compose(Affine3::from_translate(start));
    let current = chain.compose(Affine3::from_translate(current));
    current.compose(start.inverse())
}

/// Transform chain from `from` to `to`, excluding `from` itself.
///
/// A drag preview can only apply one affine chain. Return `None` when the graph
/// contains more than one path from `from` to `to` (the edit affects several
/// branches that may transform differently). A node that repeats its input body
/// contributes the identity: the chain anchors at copy 0, and the other copies
/// are placed again by evaluation when the drag ends.
pub fn transform_chain(document: &GraphDocument, from: &str, to: &str) -> Option<Affine3> {
    if from == to {
        return Some(Affine3::identity());
    }
    let path = unique_document_path(document, from, to)?;
    let mut chain = Affine3::identity();
    for id in path.iter().skip(1) {
        let node = document.nodes.iter().find(|node| &node.id == id)?;
        if node.suppressed {
            // Suppressed nodes are bypassed by evaluation; their transform must
            // not leak into the preview chain.
            continue;
        }
        if repeats_input_bodies(node) {
            continue;
        }
        if let Some(transform) = Affine3::from_node(node) {
            chain = transform.compose(chain);
        }
    }
    Some(chain)
}

/// The one path from `from` to `to`, or `None` when there are zero or multiple
/// paths. The reverse reachability set keeps the search bounded on large graphs.
fn unique_document_path(document: &GraphDocument, from: &str, to: &str) -> Option<Vec<String>> {
    let mut reaches_to = std::collections::HashSet::from([to.to_owned()]);
    let mut scan = vec![to.to_owned()];
    while let Some(id) = scan.pop() {
        for link in document.connections.iter().filter(|link| link.to == id) {
            if reaches_to.insert(link.from.clone()) {
                scan.push(link.from.clone());
            }
        }
    }
    if !reaches_to.contains(from) {
        return None;
    }
    let mut stack = vec![(from.to_owned(), vec![from.to_owned()])];
    let mut found: Option<Vec<String>> = None;
    while let Some((id, path)) = stack.pop() {
        if id == to {
            if found.is_some() {
                return None;
            }
            found = Some(path);
            continue;
        }
        for link in document.connections.iter().filter(|link| link.from == id) {
            if !reaches_to.contains(&link.to) || path.contains(&link.to) {
                continue;
            }
            let mut next = path.clone();
            next.push(link.to.clone());
            stack.push((link.to.clone(), next));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{GraphConnection, GraphDocument, GraphNode};

    fn node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
        GraphNode {
            expressions: Default::default(),
            id: id.into(),
            node_type: node_type.into(),
            values,
            suppressed: false,
            x: 0.0,
            y: 0.0,
        }
    }

    fn link(from: &str, to: &str) -> GraphConnection {
        GraphConnection {
            from: from.into(),
            to: to.into(),
            input: "shape".into(),
            out: "shape".into(),
        }
    }

    #[test]
    fn translate_chain_places_and_inverse_maps_world_delta() {
        let translate =
            Affine3::from_translate(&serde_json::json!({"x": 10.0, "y": 0.0, "z": 0.0}));
        assert_eq!(translate.apply_point([0.0, 0.0, 0.0]), [10.0, 0.0, 0.0]);
        assert_eq!(translate.inverse().apply_point([10.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn translate_values_use_contract_defaults_when_fields_are_missing() {
        let missing = serde_json::json!({});
        let translate = translate_values(&missing);
        assert_eq!(
            translate,
            TranslateValues { x: 0.0, y: 0.0, z: 0.0, rx: 0.0, ry: 0.0, rz: 0.0 }
        );
        // All six Translate defaults are 0.0, so the contract default is
        // observably equal to the old private fallback. Pin the resolver path
        // itself: the frame, placement and gizmo all consume this helper, and
        // the promoted contract-defaults-only rule rejects the old read.
        for key in ["x", "y", "z", "rx", "ry", "rz"] {
            assert_eq!(
                nodecraft_node_contract::resolve_number("translate", &missing, key),
                Some(0.0),
                "translate.{key}"
            );
        }
        assert_eq!(Affine3::from_translate(&missing).apply_point([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
        let explicit = serde_json::json!({
            "x": 2.0, "y": 3.0, "z": 4.0, "rx": 5.0, "ry": 6.0, "rz": 7.0
        });
        assert_eq!(
            translate_values(&explicit),
            TranslateValues { x: 2.0, y: 3.0, z: 4.0, rx: 5.0, ry: 6.0, rz: 7.0 }
        );
    }

    #[test]
    fn downstream_rotation_and_non_combining_mirror_compose() {
        let mut document = GraphDocument::new("placement");
        document.nodes = vec![
            node("a", "translate", serde_json::json!({"x": 2.0})),
            node("b", "translate", serde_json::json!({"z": 5.0})),
            node(
                "c",
                "mirror",
                serde_json::json!({"plane": "yz", "offset": 10.0, "combine": false}),
            ),
        ];
        document.connections = vec![link("a", "b"), link("b", "c")];
        let chain = transform_chain(&document, "a", "c").expect("chain");
        // a point is translated to z=5, then mirrored across x=10.
        let placed = chain.apply_point([0.0, 0.0, 0.0]);
        assert!((placed[0] - 20.0).abs() < 1e-9, "{placed:?}");
        assert!((placed[2] - 5.0).abs() < 1e-9, "{placed:?}");
        let world_delta = chain.inverse().apply_vector([1.0, 0.0, 0.0]);
        assert!((world_delta[0] + 1.0).abs() < 1e-9, "{world_delta:?}");
    }

    /// A transform upstream of a repeater is activated at copy 0, the source
    /// body. Without the shared copy-boundary rule the chain either refuses the
    /// path (Circular Pattern) or anchors to the reflected copy (Mirror), so the
    /// gizmo is hidden or detached from the body the user selected.
    #[test]
    fn transform_chain_anchors_at_the_source_copy_of_a_repeater() {
        let mut document = GraphDocument::new("repeater-anchor");
        document.nodes = vec![
            node("box", "solid", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 34.0})),
            node("pattern", "circularpattern", serde_json::json!({"count": 4})),
        ];
        document.connections = vec![link("box", "move"), link("move", "pattern")];

        // `solid` is a body in the exact Part layout, so the pattern has four
        // displayed copies in body order.
        assert_eq!(part_body_layout(&document, "pattern"), Some(vec!["box".to_owned(); 4]),);
        assert_eq!(preview_body_scope(&document, "move", "pattern"), Some(PreviewBodyScope::All),);
        let chain = transform_chain(&document, "move", "pattern").expect("pattern chain");
        assert_eq!(chain.apply_point([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);

        // A combining Mirror appends its reflected copy after the source copy,
        // so copy 0 is still the untouched source frame.
        document.nodes[2] = node(
            "pattern",
            "mirror",
            serde_json::json!({"plane": "yz", "offset": 10.0, "combine": true}),
        );
        assert_eq!(
            part_body_layout(&document, "pattern"),
            Some(vec!["box".to_owned(), "box".to_owned()]),
        );
        assert_eq!(preview_body_scope(&document, "move", "pattern"), Some(PreviewBodyScope::All),);
        let chain = transform_chain(&document, "move", "pattern").expect("mirror chain");
        assert_eq!(chain.apply_point([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn space_does_not_change_translation() {
        let world = Affine3::from_translate(&serde_json::json!({
            "space": "world", "x": 10.0, "y": 0.0, "z": 0.0, "rz": 90.0
        }));
        let local = Affine3::from_translate(&serde_json::json!({
            "space": "local", "x": 10.0, "y": 0.0, "z": 0.0, "rz": 90.0
        }));
        assert_eq!(world.apply_point([0.0, 0.0, 0.0]), local.apply_point([0.0, 0.0, 0.0]));
        assert!((local.apply_point([0.0, 0.0, 0.0])[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn preview_transform_moves_geometry_with_the_edited_node() {
        let start = serde_json::json!({"x": 1.0, "y": 2.0, "z": 3.0});
        let current = serde_json::json!({"x": 4.0, "y": 6.0, "z": 8.0});
        let preview = preview_transform(Affine3::identity(), &start, &current);
        assert_eq!(preview.apply_point([0.0, 0.0, 0.0]), [3.0, 4.0, 5.0]);
        // A point on the evaluated body moves with the node.
        assert_eq!(preview.apply_point([10.0, 0.0, 0.0]), [13.0, 4.0, 5.0]);
    }

    #[test]
    fn preview_transform_is_identity_for_unchanged_values() {
        let values =
            serde_json::json!({"x": 1.0, "y": -2.0, "z": 0.5, "rx": 10.0, "ry": 20.0, "rz": 30.0});
        let preview = preview_transform(Affine3::identity(), &values, &values);
        for point in [[0.0, 0.0, 0.0], [3.0, -4.0, 5.0], [-2.0, 1.0, 7.0]] {
            let moved = preview.apply_point(point);
            for axis in 0..3 {
                assert!((moved[axis] - point[axis]).abs() < 1e-9, "{point:?} {moved:?}");
            }
        }
    }

    #[test]
    fn preview_transform_rotates_about_the_node_origin_through_the_chain() {
        // The edited node rotates 90 degrees about Z at its own origin, then a
        // downstream translate places the result 10 units along X. The node
        // origin therefore sits at world (10, 0, 0), and (11, 0, 0) must swing
        // to (10, 1, 0).
        let start = serde_json::json!({"x": 0.0, "y": 0.0, "z": 0.0, "rz": 0.0});
        let current = serde_json::json!({"x": 0.0, "y": 0.0, "z": 0.0, "rz": 90.0});
        let downstream = Affine3::from_translate(&serde_json::json!({"x": 10.0}));
        let preview = preview_transform(downstream, &start, &current);
        let moved = preview.apply_point([11.0, 0.0, 0.0]);
        assert!((moved[0] - 10.0).abs() < 1e-9, "{moved:?}");
        assert!((moved[1] - 1.0).abs() < 1e-9, "{moved:?}");
        assert!(moved[2].abs() < 1e-9, "{moved:?}");
        // The pivot itself does not move under pure rotation.
        let pivot = preview.apply_point([10.0, 0.0, 0.0]);
        assert!((pivot[0] - 10.0).abs() < 1e-9, "{pivot:?}");
        assert!(pivot[1].abs() < 1e-9, "{pivot:?}");
    }

    #[test]
    fn branch_preview_delta_uses_the_downstream_chain_frame() {
        // This is the deep-ghost case: the node rotates, a downstream translate
        // places the branch, and the drag changes the node's X value. The ghost
        // matrix must be chain · Tcurrent, not Tcurrent · Tstart⁻¹ · chain · Tstart.
        let downstream = Affine3::from_translate(&serde_json::json!({"y": 22.0}));
        let start = serde_json::json!({"rx": 90.0, "z": 6.0, "rz": 0.0});
        let current = serde_json::json!({"rx": 90.0, "z": 6.0, "rz": 35.0});
        let base = downstream.compose(Affine3::from_translate(&start));
        let matrix = preview_transform(downstream, &start, &current).compose(base);
        let expected = downstream.compose(Affine3::from_translate(&current));
        for point in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            let got = matrix.apply_point(point);
            let want = expected.apply_point(point);
            for axis in 0..3 {
                assert!((got[axis] - want[axis]).abs() < 1.0e-9, "{point:?}: {got:?} != {want:?}");
            }
        }
        let wrong = preview_transform(Affine3::identity(), &start, &current).compose(base);
        assert!(
            (wrong.apply_point([1.0, 0.0, 0.0])[0] - expected.apply_point([1.0, 0.0, 0.0])[0])
                .abs()
                > 1.0e-6
        );
    }

    #[test]
    fn preview_transform_reports_column_major_gpu_layout() {
        let moved = preview_transform(
            Affine3::identity(),
            &serde_json::json!({"x": 0.0}),
            &serde_json::json!({"x": 2.0, "y": 3.0}),
        );
        let matrix = moved.to_column_major_f32();
        // Translation lives in the last column of a column-major matrix.
        assert_eq!(matrix[3][0], 2.0);
        assert_eq!(matrix[3][1], 3.0);
        assert_eq!(matrix[3][2], 0.0);
        assert_eq!(matrix[3][3], 1.0);
    }

    fn combine_graph(move_branch: usize) -> GraphDocument {
        let mut document = GraphDocument::new("preview-scope");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("combine", "combine", serde_json::json!({})),
        ];
        let mut shape_links = vec![link("move", "combine"), link("sphere", "combine")];
        if move_branch == 1 {
            shape_links.swap(0, 1);
        }
        document.connections = vec![link("box", "move")];
        document.connections.extend(shape_links);
        document
    }

    fn prefixes(names: &[&str]) -> Option<PreviewBodyScope> {
        Some(PreviewBodyScope::Prefixes(names.iter().map(|name| (*name).to_owned()).collect()))
    }

    #[test]
    fn preview_scope_is_unknown_after_a_boolean_and_fillet() {
        let mut document = GraphDocument::new("preview-unknown");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("boolean", "boolean", serde_json::json!({"mode": "union"})),
            node("fillet", "fillet", serde_json::json!({})),
            node("look", "look", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "move"),
            GraphConnection {
                from: "move".into(),
                to: "boolean".into(),
                input: "a".into(),
                out: "shape".into(),
            },
            GraphConnection {
                from: "sphere".into(),
                to: "boolean".into(),
                input: "b".into(),
                out: "shape".into(),
            },
            link("boolean", "fillet"),
            link("fillet", "look"),
        ];
        assert_eq!(preview_body_scope(&document, "move", "look"), Some(PreviewBodyScope::Unknown));
    }

    /// The Boolean merge rule lives in the registry, not in the session. Both
    /// operands lose the target's provenance; a one-operand pass-through keeps
    /// the source layout and previews the source body.
    #[test]
    fn preview_scope_uses_the_registry_merge_rule() {
        let mut document = GraphDocument::new("registry-merge");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("boolean", "boolean", serde_json::json!({"mode": "union"})),
        ];
        document.connections = vec![
            link("box", "move"),
            GraphConnection {
                from: "move".into(),
                to: "boolean".into(),
                input: "a".into(),
                out: "shape".into(),
            },
            GraphConnection {
                from: "sphere".into(),
                to: "boolean".into(),
                input: "b".into(),
                out: "shape".into(),
            },
        ];
        let boolean = document.nodes.iter().find(|node| node.id == "boolean").expect("boolean");
        let view = nodecraft_engine::nodes::NodeView::new(
            &boolean.id,
            &boolean.node_type,
            &boolean.values,
            boolean.suppressed,
        );
        let connections: Vec<nodecraft_protocol::Connection> = document
            .connections
            .iter()
            .map(|link| nodecraft_protocol::Connection {
                from: link.from.clone(),
                to: link.to.clone(),
                input: link.input.clone(),
                out: link.out.clone(),
            })
            .collect();
        let affected: std::collections::HashSet<String> = ["move".to_owned()].into_iter().collect();
        assert_eq!(
            nodecraft_engine::nodes::merge_provenance(view, &connections, &affected),
            nodecraft_engine::nodes::MergeProvenance::LosesProvenance,
        );
        assert_eq!(preview_body_scope(&document, "move", "boolean"), None);

        // B is missing: the Boolean is the one-operand pass-through and the
        // source body keeps its own layout.
        let mut one_document = document.clone();
        one_document.connections.retain(|link| !(link.to == "boolean" && link.input == "b"));
        let one_boolean =
            one_document.nodes.iter().find(|node| node.id == "boolean").expect("boolean");
        let one_view = nodecraft_engine::nodes::NodeView::new(
            &one_boolean.id,
            &one_boolean.node_type,
            &one_boolean.values,
            one_boolean.suppressed,
        );
        let one_operand: Vec<nodecraft_protocol::Connection> = one_document
            .connections
            .iter()
            .map(|link| nodecraft_protocol::Connection {
                from: link.from.clone(),
                to: link.to.clone(),
                input: link.input.clone(),
                out: link.out.clone(),
            })
            .collect();
        assert_eq!(
            nodecraft_engine::nodes::merge_provenance(one_view, &one_operand, &affected),
            nodecraft_engine::nodes::MergeProvenance::Preserved,
        );
        assert_eq!(
            preview_body_scope(&one_document, "move", "boolean"),
            Some(PreviewBodyScope::All),
        );
    }

    #[test]
    fn preview_scope_isolates_the_connected_combine_branch() {
        let document = combine_graph(0);
        assert_eq!(preview_body_scope(&document, "move", "combine"), prefixes(&["body:0:"]));
        let document = combine_graph(1);
        assert_eq!(preview_body_scope(&document, "move", "combine"), prefixes(&["body:1:"]));
    }

    #[test]
    fn preview_scope_flattens_nested_combines() {
        // The exact Part path numbers bodies in flattened order, so a nested
        // Combine does not produce nested `body:0:body:0:` names. Only the first
        // body is downstream of the transform.
        let mut document = GraphDocument::new("preview-scope");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("inner", "combine", serde_json::json!({})),
            node("other", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("outer", "combine", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "move"),
            link("move", "inner"),
            link("sphere", "inner"),
            link("inner", "outer"),
            link("other", "outer"),
        ];
        assert_eq!(preview_body_scope(&document, "move", "outer"), prefixes(&["body:0:"]));
    }

    #[test]
    fn preview_scope_moves_every_body_when_the_transform_covers_the_whole_part() {
        // Regression for a forest-style assembly: the top-level transform sits
        // after the Combine that produces every body, so every `body:N:` moves.
        let mut document = GraphDocument::new("preview-scope");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("inner", "combine", serde_json::json!({})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("outer", "combine", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "inner"),
            link("sphere", "inner"),
            link("inner", "move"),
            link("move", "outer"),
        ];
        assert_eq!(preview_body_scope(&document, "move", "outer"), Some(PreviewBodyScope::All));
    }

    #[test]
    fn preview_scope_is_all_bodies_for_the_displayed_transform() {
        let document = combine_graph(0);
        assert_eq!(preview_body_scope(&document, "move", "move"), Some(PreviewBodyScope::All));
        // A Boolean merges both operands into one body; moving only one branch
        // has no display-scoped preview.
        let mut document = combine_graph(0);
        document.nodes.push(node("boolean", "boolean", serde_json::json!({})));
        document.connections.push(GraphConnection {
            from: "move".into(),
            to: "boolean".into(),
            input: "a".into(),
            out: "shape".into(),
        });
        document.connections.push(GraphConnection {
            from: "sphere".into(),
            to: "boolean".into(),
            input: "b".into(),
            out: "shape".into(),
        });
        assert_eq!(preview_body_scope(&document, "move", "boolean"), None);
    }

    #[test]
    fn preview_scope_accounts_for_combining_mirrors_and_scatter_instances() {
        // A combining Mirror appends the mirrored copy after the originals, so
        // both body 0 and body 1 belong to the edited branch.
        let mut document = GraphDocument::new("preview-scope");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("mirror", "mirror", serde_json::json!({"combine": true, "plane": "yz"})),
            node("other", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("combine", "combine", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "move"),
            link("move", "mirror"),
            link("mirror", "combine"),
            link("other", "combine"),
        ];
        assert_eq!(
            preview_body_scope(&document, "move", "combine"),
            prefixes(&["body:0:", "body:1:"])
        );

        // Scatter repeats its source bodies once per instance.
        let mut document = GraphDocument::new("preview-scope");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("move", "translate", serde_json::json!({"x": 10.0})),
            node("scatter", "scatter", serde_json::json!({"count": 3})),
            node("other", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("combine", "combine", serde_json::json!({})),
        ];
        document.connections = vec![
            link("box", "move"),
            link("move", "scatter"),
            link("scatter", "combine"),
            link("other", "combine"),
        ];
        assert_eq!(
            preview_body_scope(&document, "move", "combine"),
            prefixes(&["body:0:", "body:1:", "body:2:"])
        );
    }

    /// The inspector shows Combine on for a fresh mirror; the session body
    /// layout must agree even when the saved document has no `combine` key.
    #[test]
    fn mirror_missing_combine_keeps_the_source_in_body_layout() {
        let mut document = GraphDocument::new("mirror-combine-default");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("mirror", "mirror", serde_json::json!({"plane": "yz", "offset": 10.0})),
        ];
        document.connections = vec![link("box", "mirror")];
        let combined = Some(vec!["box".to_owned(), "box".to_owned()]);
        assert_eq!(part_body_layout(&document, "mirror"), combined);

        document.nodes[1].values["combine"] = serde_json::json!(true);
        assert_eq!(part_body_layout(&document, "mirror"), combined);

        document.nodes[1].values["combine"] = serde_json::json!(false);
        assert_eq!(part_body_layout(&document, "mirror"), Some(vec!["box".to_owned()]));
    }

    /// Fillet follows its source layout until edges are selected, then the
    /// blend is one new body. The rule lives on the node; this pins it through
    /// the registry.
    #[test]
    fn fillet_body_layout_is_the_source_until_edges_are_selected() {
        assert!(nodecraft_engine::nodes::handles("fillet"));
        let mut document = GraphDocument::new("fillet-layout");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("fillet", "fillet", serde_json::json!({})),
        ];
        document.connections = vec![link("box", "fillet")];
        assert_eq!(part_body_layout(&document, "fillet"), Some(vec!["box".to_owned()]));

        document.nodes[1].values = serde_json::json!({"edges": [{"ordinal": 0}]});
        assert_eq!(part_body_layout(&document, "fillet"), Some(vec!["fillet".to_owned()]));
    }

    /// Chamfer follows its source layout until edges are selected, then the
    /// bevel is one new body. The rule lives on the node; this pins it through
    /// the registry.
    #[test]
    fn chamfer_body_layout_is_the_source_until_edges_are_selected() {
        assert!(nodecraft_engine::nodes::handles("chamfer"));
        let mut document = GraphDocument::new("chamfer-layout");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("chamfer", "chamfer", serde_json::json!({})),
        ];
        document.connections = vec![link("box", "chamfer")];
        assert_eq!(part_body_layout(&document, "chamfer"), Some(vec!["box".to_owned()]));

        document.nodes[1].values = serde_json::json!({"edges": [{"ordinal": 0}]});
        assert_eq!(part_body_layout(&document, "chamfer"), Some(vec!["chamfer".to_owned()]));
    }

    /// A multi-body shell keeps the source layout: one shelled body per input
    /// body, in the same order. The rule lives on the node; this pins it
    /// through the registry.
    #[test]
    fn shell_body_layout_is_the_source_layout() {
        assert!(nodecraft_engine::nodes::handles("shell"));
        let mut document = GraphDocument::new("shell-layout");
        document.nodes = vec![
            node("big", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("small", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("part", "combine", serde_json::json!({})),
            node("shell", "shell", serde_json::json!({"thickness": 1.0})),
        ];
        document.connections =
            vec![link("big", "part"), link("small", "part"), link("part", "shell")];
        assert_eq!(
            part_body_layout(&document, "shell"),
            Some(vec!["big".to_owned(), "small".to_owned()]),
        );
    }

    /// Offset Faces keeps the source layout: it replaces edited bodies in
    /// place. This is the reviewer's decision, and before the port the node had
    /// no layout arm, so the session could not reproduce its Part.
    #[test]
    fn offset_faces_body_layout_is_the_source_layout() {
        assert!(nodecraft_engine::nodes::handles("offsetfaces"));
        let mut document = GraphDocument::new("offset-faces-layout");
        document.nodes = vec![
            node("big", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("small", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("part", "combine", serde_json::json!({})),
            node(
                "offset",
                "offsetfaces",
                serde_json::json!({"distance": 1.0, "faces": [{"id": "body:1:face:1"}]}),
            ),
        ];
        document.connections =
            vec![link("big", "part"), link("small", "part"), link("part", "offset")];
        assert_eq!(
            part_body_layout(&document, "offset"),
            Some(vec!["big".to_owned(), "small".to_owned()]),
        );
    }

    /// A mirror with no saved plane reflects through XZ, the contract default.
    #[test]
    fn mirror_missing_plane_uses_the_xz_transform() {
        let mirror = Affine3::from_mirror(&serde_json::json!({}));
        assert_eq!(mirror.apply_point([1.0, 2.0, 3.0]), [1.0, -2.0, 3.0]);
        let explicit = Affine3::from_mirror(&serde_json::json!({"plane": "xz"}));
        assert_eq!(mirror.apply_point([4.0, 5.0, 6.0]), explicit.apply_point([4.0, 5.0, 6.0]));
    }

    /// Missing scatter count uses the contract default (1000), not the old
    /// session reader's 100.
    #[test]
    fn scatter_missing_count_uses_the_contract_default() {
        let mut document = GraphDocument::new("scatter-count-default");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("scatter", "scatter", serde_json::json!({})),
        ];
        document.connections = vec![link("box", "scatter")];
        assert_eq!(part_body_layout(&document, "scatter").map(|layout| layout.len()), Some(1000));
    }

    /// A Circular Pattern repeats the whole source layout once per copy, so a
    /// two-body source with three copies has six display bodies.
    #[test]
    fn circular_pattern_body_layout_repeats_the_source_count_times() {
        assert!(nodecraft_engine::nodes::handles("circularpattern"));
        let mut document = GraphDocument::new("circular-pattern-layout");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("part", "combine", serde_json::json!({})),
            node("pattern", "circularpattern", serde_json::json!({"count": 3})),
        ];
        document.connections =
            vec![link("box", "part"), link("sphere", "part"), link("part", "pattern")];
        assert_eq!(
            part_body_layout(&document, "pattern"),
            Some(vec![
                "box".to_owned(),
                "sphere".to_owned(),
                "box".to_owned(),
                "sphere".to_owned(),
                "box".to_owned(),
                "sphere".to_owned(),
            ]),
        );

        // A fresh node has no saved count and uses the contract default (6).
        document.nodes[3].values = serde_json::json!({});
        assert_eq!(part_body_layout(&document, "pattern").map(|layout| layout.len()), Some(12),);

        // Copies have per-instance rotations, so the chain anchors at copy 0
        // (the source frame) instead of trying to encode every copy.
        let chain = transform_chain(&document, "part", "pattern").expect("copy-0 chain");
        assert_eq!(chain.apply_point([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn preview_scope_is_none_when_the_node_does_not_feed_the_target() {
        let mut document = combine_graph(0);
        document.nodes.push(node("loose", "translate", serde_json::json!({"x": 1.0})));
        assert_eq!(preview_body_scope(&document, "loose", "combine"), None);
    }
    #[test]
    fn transform_chain_rejects_ambiguous_paths_and_anchors_scatter() {
        // Two paths from the transform to the target merge; one affine preview
        // cannot represent both downstream placements.
        let mut document = GraphDocument::new("branch-preview");
        document.nodes = vec![
            node("move", "translate", serde_json::json!({"x": 1.0})),
            node("a", "translate", serde_json::json!({"x": 10.0})),
            node("b", "translate", serde_json::json!({"x": 20.0})),
            node("merge", "boolean", serde_json::json!({"mode": "union"})),
        ];
        document.connections =
            vec![link("move", "a"), link("move", "b"), link("a", "merge"), link("b", "merge")];
        assert!(transform_chain(&document, "move", "merge").is_none());

        // A scatter below the edited node also repeats the body; the chain
        // anchors at its input frame (copy 0's source) so the gizmo can resolve.
        let mut document = GraphDocument::new("scatter-preview");
        document.nodes = vec![
            node("move", "translate", serde_json::json!({"x": 1.0})),
            node("scatter", "scatter", serde_json::json!({"count": 4})),
        ];
        document.connections = vec![link("move", "scatter")];
        let chain = transform_chain(&document, "move", "scatter").expect("scatter chain");
        assert_eq!(chain.apply_point([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);

        // A suppressed downstream transform is bypassed, not composed.
        let mut document = GraphDocument::new("suppressed-chain");
        document.nodes = vec![
            node("move", "translate", serde_json::json!({"x": 1.0})),
            GraphNode {
                expressions: Default::default(),
                id: "down".into(),
                node_type: "translate".into(),
                values: serde_json::json!({"x": 10.0}),
                suppressed: true,
                x: 0.0,
                y: 0.0,
            },
        ];
        document.connections = vec![link("move", "down")];
        let chain = transform_chain(&document, "move", "down").expect("chain");
        assert_eq!(chain.apply_point([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);

        // A single chain still resolves.
        let mut document = GraphDocument::new("one-chain");
        document.nodes = vec![
            node("move", "translate", serde_json::json!({"x": 1.0})),
            node("down", "translate", serde_json::json!({"x": 10.0})),
        ];
        document.connections = vec![link("move", "down")];
        assert!(transform_chain(&document, "move", "down").is_some());
    }
}
