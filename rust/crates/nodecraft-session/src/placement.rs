//! Affine placements for graph control points.
//!
//! A node's values are expressed in its parent frame; controls displayed at a
//! downstream breakpoint must be placed through every downstream transform and
//! edits must be mapped back through the inverse chain.

use crate::{GraphConnection, GraphDocument, GraphNode};

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

fn contract_bool(node_type: &str, values: &serde_json::Value, key: &str) -> bool {
    nodecraft_node_contract::resolve_bool(node_type, values, key)
        .unwrap_or_else(|| panic!("node contract has no switch field {node_type}.{key}"))
}

/// Row-major affine transform: `p' = M * [x, y, z, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct Affine3 {
    m: [[f64; 4]; 4],
}

impl Affine3 {
    pub fn identity() -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Rotation only, matching the mesh/BRep rotation order.
    pub fn rotation(values: &serde_json::Value) -> Self {
        let value = |key: &str| contract_number("translate", values, key);
        let (rx, ry, rz) = (value("rx").to_radians(), value("ry").to_radians(), value("rz").to_radians());
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let x_axis = [cy * cz, cy * sz, -sy];
        let y_axis = [sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy];
        let z_axis = [
            x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
            x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
            x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
        ];
        Self {
            m: [
                [x_axis[0], y_axis[0], z_axis[0], 0.0],
                [x_axis[1], y_axis[1], z_axis[1], 0.0],
                [x_axis[2], y_axis[2], z_axis[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Translation after X/Y/Z rotation, matching the mesh and BRep transform paths.
    /// `space` only changes the manipulator frame; it never changes geometry.
    pub fn from_translate(values: &serde_json::Value) -> Self {
        let value = |key: &str| contract_number("translate", values, key);
        let mut result = Self::rotation(values);
        result.m[0][3] = value("x");
        result.m[1][3] = value("y");
        result.m[2][3] = value("z");
        result
    }

    pub fn from_mirror(values: &serde_json::Value) -> Self {
        let plane = contract_text("mirror", values, "plane");
        let offset = contract_number("mirror", values, "offset");
        let mut m = Self::identity().m;
        match plane.as_ref() {
            "xy" => { m[2][2] = -1.0; m[2][3] = 2.0 * offset; }
            "yz" => { m[0][0] = -1.0; m[0][3] = 2.0 * offset; }
            _ => { m[1][1] = -1.0; m[1][3] = 2.0 * offset; }
        }
        Self { m }
    }

    pub fn from_node(node: &GraphNode) -> Option<Self> {
        match node.node_type.as_str() {
            "translate" => Some(Self::from_translate(&node.values)),
            "mirror" => Some(Self::from_mirror(&node.values)),
            _ => None,
        }
    }

    /// `self` applied after `other`.
    pub fn compose(self, other: Self) -> Self {
        let mut m = [[0.0; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                m[row][column] = (0..4).map(|k| self.m[row][k] * other.m[k][column]).sum();
            }
        }
        Self { m }
    }

    pub fn apply_point(self, point: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * point[0] + self.m[0][1] * point[1] + self.m[0][2] * point[2] + self.m[0][3],
            self.m[1][0] * point[0] + self.m[1][1] * point[1] + self.m[1][2] * point[2] + self.m[1][3],
            self.m[2][0] * point[0] + self.m[2][1] * point[1] + self.m[2][2] * point[2] + self.m[2][3],
        ]
    }

    pub fn apply_vector(self, vector: [f64; 3]) -> [f64; 3] {
        [
            self.m[0][0] * vector[0] + self.m[0][1] * vector[1] + self.m[0][2] * vector[2],
            self.m[1][0] * vector[0] + self.m[1][1] * vector[1] + self.m[1][2] * vector[2],
            self.m[2][0] * vector[0] + self.m[2][1] * vector[1] + self.m[2][2] * vector[2],
        ]
    }

    pub fn inverse_linear(self) -> [[f64; 3]; 3] {
        let inverse = self.inverse();
        [
            [inverse.m[0][0], inverse.m[0][1], inverse.m[0][2]],
            [inverse.m[1][0], inverse.m[1][1], inverse.m[1][2]],
            [inverse.m[2][0], inverse.m[2][1], inverse.m[2][2]],
        ]
    }

    /// Inverse of a rigid/reflected affine transform.
    pub fn inverse(self) -> Self {
        let r = [
            [self.m[0][0], self.m[0][1], self.m[0][2]],
            [self.m[1][0], self.m[1][1], self.m[1][2]],
            [self.m[2][0], self.m[2][1], self.m[2][2]],
        ];
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        let inv_det = 1.0 / det;
        let inv = [
            [
                (r[1][1] * r[2][2] - r[1][2] * r[2][1]) * inv_det,
                (r[0][2] * r[2][1] - r[0][1] * r[2][2]) * inv_det,
                (r[0][1] * r[1][2] - r[0][2] * r[1][1]) * inv_det,
            ],
            [
                (r[1][2] * r[2][0] - r[1][0] * r[2][2]) * inv_det,
                (r[0][0] * r[2][2] - r[0][2] * r[2][0]) * inv_det,
                (r[0][2] * r[1][0] - r[0][0] * r[1][2]) * inv_det,
            ],
            [
                (r[1][0] * r[2][1] - r[1][1] * r[2][0]) * inv_det,
                (r[0][1] * r[2][0] - r[0][0] * r[2][1]) * inv_det,
                (r[0][0] * r[1][1] - r[0][1] * r[1][0]) * inv_det,
            ],
        ];
        let t = [self.m[0][3], self.m[1][3], self.m[2][3]];
        let it = [
            -(inv[0][0] * t[0] + inv[0][1] * t[1] + inv[0][2] * t[2]),
            -(inv[1][0] * t[0] + inv[1][1] * t[1] + inv[1][2] * t[2]),
            -(inv[2][0] * t[0] + inv[2][1] * t[1] + inv[2][2] * t[2]),
        ];
        Self {
            m: [
                [inv[0][0], inv[0][1], inv[0][2], it[0]],
                [inv[1][0], inv[1][1], inv[1][2], it[1]],
                [inv[2][0], inv[2][1], inv[2][2], it[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Column-major `[[f32; 4]; 4]` for GPU uploads (WGSL `mat4x4<f32>` layout).
    pub fn to_column_major_f32(self) -> [[f32; 4]; 4] {
        let mut out = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[column][row] = self.m[row][column] as f32;
            }
        }
        out
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
    if selected != target {
        let merges_operands = document
            .nodes
            .iter()
            .find(|node| node.id == target)
            .is_some_and(|node| matches!(node.node_type.as_str(), "boolean" | "fuse"))
            && document
                .connections
                .iter()
                .filter(|link| link.to == target)
                .map(|link| link.input.as_str())
                .collect::<std::collections::HashSet<_>>()
                .is_superset(&["a", "b"].into_iter().collect());
        if merges_operands {
            // Both operands are unioned into one result body; provenance is
            // lost inside it. A drag on one branch cannot move only its own
            // geometry, so there is no display-scoped fast preview; the caller
            // lets evaluation produce the new body.
            return None;
        }
    }
    let Some(layout) = part_body_layout(document, target) else {
        // Downstream merging or an unsupported display chain hid the body
        // layout. This is not the same as "all bodies are downstream": the
        // caller must not move the whole breakpoint mesh here.
        return Some(PreviewBodyScope::Unknown);
    };
    let mut prefixes = Vec::new();
    let mut all_affected = !layout.is_empty();
    for (index, marker) in layout.iter().enumerate() {
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
    let source = |inputs: &[&str]| -> Option<&GraphConnection> {
        document
            .connections
            .iter()
            .find(|link| link.to == target && inputs.contains(&link.input.as_str()))
    };
    let layout = match node.node_type.as_str() {
        "solid" | "primitive" | "stepimporter" => vec![target.to_owned()],
        "translate" => part_body_layout(document, &source(&["shape"])?.from)?,
        "mirror" => {
            let source_layout = part_body_layout(document, &source(&["shape"])?.from)?;
            if mirror_combines(&node.values) {
                let mut combined = source_layout.clone();
                combined.extend(source_layout);
                combined
            } else {
                source_layout
            }
        }
        "combine" => {
            let mut layout = Vec::new();
            for link in document
                .connections
                .iter()
                .filter(|link| link.to == target && link.input == "shape")
            {
                layout.extend(part_body_layout(document, &link.from)?);
            }
            layout
        }
        "boolean" | "fuse" => {
            let a = source(&["a"]).map(|link| link.from.clone());
            let b = source(&["b"]).map(|link| link.from.clone());
            match (a, b) {
                // Both operands are unioned into one result body; provenance is
                // lost inside it, so the body moves as a whole.
                (Some(_), Some(_)) => vec![target.to_owned()],
                (Some(a), None) => part_body_layout(document, &a)?,
                (None, Some(b)) => part_body_layout(document, &b)?,
                (None, None) => return None,
            }
        }
        "extrude" | "loft" | "surfaceloft" => vec![target.to_owned()],
        "shell" => {
            // One shelled body per input body, in the same order.
            part_body_layout(document, &source(&["shape"])?.from)?
        }
        "chamfer" | "fillet" => {
            let source_layout = part_body_layout(document, &source(&["shape"])?.from)?;
            let has_edges = node
                .values
                .get("edges")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|edges| !edges.is_empty());
            if has_edges {
                vec![target.to_owned()]
            } else {
                source_layout
            }
        }
        "scatter" | "circularpattern" => {
            let source_layout = part_body_layout(document, &source(&["shape"])?.from)?;
            let count = contract_number(&node.node_type, &node.values, "count")
                .round()
                .clamp(1.0, 5000.0) as usize;
            let mut layout = Vec::with_capacity(source_layout.len() * count);
            for _ in 0..count {
                layout.extend(source_layout.iter().cloned());
            }
            layout
        }
        // Sketches/splines contribute no bodies (their curves are appended after
        // the bodies as separate display instances).
        "sketch" | "spline" => Vec::new(),
        _ => return None,
    };
    Some(layout)
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

/// Mirror value that enables `merge(mesh, mirrored)` in the display evaluator.
fn mirror_combines(values: &serde_json::Value) -> bool {
    contract_bool("mirror", values, "combine")
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
    match node.node_type.as_str() {
        "mirror" => mirror_combines(&node.values),
        "scatter" | "circularpattern" => true,
        _ => false,
    }
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
fn unique_document_path(
    document: &GraphDocument,
    from: &str,
    to: &str,
) -> Option<Vec<String>> {
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
        GraphNode { expressions: Default::default(), id: id.into(), node_type: node_type.into(), values, suppressed: false, x: 0.0, y: 0.0 }
    }

    fn link(from: &str, to: &str) -> GraphConnection {
        GraphConnection { from: from.into(), to: to.into(), input: "shape".into(), out: "shape".into() }
    }

    #[test]
    fn translate_chain_places_and_inverse_maps_world_delta() {
        let translate = Affine3::from_translate(&serde_json::json!({"x": 10.0, "y": 0.0, "z": 0.0}));
        assert_eq!(translate.apply_point([0.0, 0.0, 0.0]), [10.0, 0.0, 0.0]);
        assert_eq!(translate.inverse().apply_point([10.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
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
        assert_eq!(
            part_body_layout(&document, "pattern"),
            Some(vec!["box".to_owned(); 4]),
        );
        assert_eq!(
            preview_body_scope(&document, "move", "pattern"),
            Some(PreviewBodyScope::All),
        );
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
        assert_eq!(
            preview_body_scope(&document, "move", "pattern"),
            Some(PreviewBodyScope::All),
        );
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
        let values = serde_json::json!({"x": 1.0, "y": -2.0, "z": 0.5, "rx": 10.0, "ry": 20.0, "rz": 30.0});
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
        assert!((wrong.apply_point([1.0, 0.0, 0.0])[0] - expected.apply_point([1.0, 0.0, 0.0])[0]).abs() > 1.0e-6);
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
            GraphConnection { from: "move".into(), to: "boolean".into(), input: "a".into(), out: "shape".into() },
            GraphConnection { from: "sphere".into(), to: "boolean".into(), input: "b".into(), out: "shape".into() },
            link("boolean", "fillet"),
            link("fillet", "look"),
        ];
        assert_eq!(
            preview_body_scope(&document, "move", "look"),
            Some(PreviewBodyScope::Unknown)
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
        assert_eq!(
            preview_body_scope(&document, "move", "outer"),
            Some(PreviewBodyScope::All)
        );
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
        let mut document = GraphDocument::new("circular-pattern-layout");
        document.nodes = vec![
            node("box", "primitive", serde_json::json!({"primitiveType": "box"})),
            node("sphere", "primitive", serde_json::json!({"primitiveType": "sphere"})),
            node("part", "combine", serde_json::json!({})),
            node("pattern", "circularpattern", serde_json::json!({"count": 3})),
        ];
        document.connections = vec![
            link("box", "part"),
            link("sphere", "part"),
            link("part", "pattern"),
        ];
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
        assert_eq!(
            part_body_layout(&document, "pattern").map(|layout| layout.len()),
            Some(12),
        );

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
        document.connections = vec![
            link("move", "a"),
            link("move", "b"),
            link("a", "merge"),
            link("b", "merge"),
        ];
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
            GraphNode { expressions: Default::default(),
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
