//! Affine placements for graph control points.
//!
//! A node's values are expressed in its parent frame; controls displayed at a
//! downstream breakpoint must be placed through every downstream transform and
//! edits must be mapped back through the inverse chain.

use crate::{GraphDocument, GraphNode};

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
        let value = |key: &str| values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
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
        let value = |key: &str| values.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let mut result = Self::rotation(values);
        result.m[0][3] = value("x");
        result.m[1][3] = value("y");
        result.m[2][3] = value("z");
        result
    }

    pub fn from_mirror(values: &serde_json::Value) -> Self {
        let plane = values.get("plane").and_then(serde_json::Value::as_str).unwrap_or("xy");
        let offset = values.get("offset").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
        let mut m = Self::identity().m;
        match plane {
            "xz" => { m[1][1] = -1.0; m[1][3] = 2.0 * offset; }
            "yz" => { m[0][0] = -1.0; m[0][3] = 2.0 * offset; }
            _ => { m[2][2] = -1.0; m[2][3] = 2.0 * offset; }
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
}

/// Transform chain from `from` to `to`, excluding `from` itself.
///
/// Non-transform nodes (Boolean, combine, scatter, ...) contribute identity; a
/// scatter instance cannot be represented by one affine chain.
pub fn transform_chain(document: &GraphDocument, from: &str, to: &str) -> Option<Affine3> {
    if from == to {
        return Some(Affine3::identity());
    }
    let path = document_path(document, from, to)?;
    let mut chain = Affine3::identity();
    for id in path.iter().skip(1) {
        let node = document.nodes.iter().find(|node| &node.id == id)?;
        if let Some(transform) = Affine3::from_node(node) {
            chain = transform.compose(chain);
        }
    }
    Some(chain)
}

fn document_path(document: &GraphDocument, from: &str, to: &str) -> Option<Vec<String>> {
    let mut queue = std::collections::VecDeque::from([vec![from.to_owned()]]);
    let mut seen = std::collections::HashSet::from([from.to_owned()]);
    while let Some(path) = queue.pop_front() {
        let Some(last) = path.last() else { continue };
        for connection in document.connections.iter().filter(|link| &link.from == last) {
            if !seen.insert(connection.to.clone()) {
                continue;
            }
            let mut next = path.clone();
            next.push(connection.to.clone());
            if connection.to == to {
                return Some(next);
            }
            queue.push_back(next);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphConnection, GraphDocument, GraphNode};

    fn node(id: &str, node_type: &str, values: serde_json::Value) -> GraphNode {
        GraphNode { id: id.into(), node_type: node_type.into(), values, suppressed: false, x: 0.0, y: 0.0 }
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
    fn downstream_rotation_and_mirror_compose() {
        let mut document = GraphDocument::new("placement");
        document.nodes = vec![
            node("a", "translate", serde_json::json!({"x": 2.0})),
            node("b", "translate", serde_json::json!({"z": 5.0})),
            node("c", "mirror", serde_json::json!({"plane": "yz", "offset": 10.0})),
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
}
