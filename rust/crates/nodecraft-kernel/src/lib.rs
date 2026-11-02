//! Kernel boundary for the Rust experiment.
//!
//! The first slice is deliberately procedural (box/cylinder + translate) so the
//! protocol, mesh transport and conformance harness can be proven before solidcore
//! geometry is wired in. Replace `ProceduralKernel` with a solidcore implementation
//! behind this trait; nothing else needs to change.

use nodecraft_protocol::{BrepEdge, Mesh};

pub trait Kernel {
    fn primitive(&mut self, kind: &str, values: &serde_json::Value) -> Result<Mesh, String>;
    fn translate(&mut self, mesh: &Mesh, values: &serde_json::Value) -> Result<Mesh, String>;
}

#[derive(Default)]
pub struct ProceduralKernel;

fn number(values: &serde_json::Value, key: &str, default: f64) -> f64 {
    values
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default)
}

fn box_mesh(length: f64, width: f64, height: f64) -> Mesh {
    let (x, y, z) = (length / 2.0, width / 2.0, height / 2.0);
    let corners = [
        [-x, -y, -z],
        [x, -y, -z],
        [x, y, -z],
        [-x, y, -z],
        [-x, -y, z],
        [x, -y, z],
        [x, y, z],
        [-x, y, z],
    ];
    let mut mesh = Mesh::default();
    for quad in [
        ([0usize, 1, 2, 3], [0.0, 0.0, -1.0], 0usize),
        ([4, 5, 6, 7], [0.0, 0.0, 1.0], 1),
        ([0, 1, 5, 4], [0.0, -1.0, 0.0], 2),
        ([1, 2, 6, 5], [1.0, 0.0, 0.0], 3),
        ([2, 3, 7, 6], [0.0, 1.0, 0.0], 4),
        ([3, 0, 4, 7], [-1.0, 0.0, 0.0], 5),
    ] {
        let (indices, normal, face) = quad;
        let base = mesh.vertices.len() as u32;
        for index in indices {
            let corner = corners[index];
            mesh.vertices.push([corner[0] as f32, corner[1] as f32, corner[2] as f32]);
            mesh.normals.push(normal);
        }
        mesh.faces.push([base, base + 1, base + 2]);
        mesh.faces.push([base, base + 2, base + 3]);
        mesh.face_groups.push(format!("face:{face}"));
        mesh.face_groups.push(format!("face:{face}"));
    }
    mesh.brep_edges = (0..12)
        .map(|index| BrepEdge {
            id: format!("mesh-edge:{index}"),
            kind: "mesh".into(),
            segments: Vec::new(),
        })
        .collect();
    mesh.lightweight = true;
    mesh
}

fn cylinder_mesh(radius: f64, height: f64, segments: usize) -> Mesh {
    let segments = segments.max(3);
    let mut mesh = Mesh::default();
    let mut top = Vec::with_capacity(segments);
    let mut bottom = Vec::with_capacity(segments);
    for index in 0..segments {
        let angle = std::f64::consts::TAU * index as f64 / segments as f64;
        let (sin, cos) = angle.sin_cos();
        let normal = [cos as f32, sin as f32, 0.0];
        mesh.vertices.push([(radius * cos) as f32, (radius * sin) as f32, (height / 2.0) as f32]);
        mesh.normals.push(normal);
        top.push(mesh.vertices.len() as u32 - 1);
        mesh.vertices.push([(radius * cos) as f32, (radius * sin) as f32, (-height / 2.0) as f32]);
        mesh.normals.push(normal);
        bottom.push(mesh.vertices.len() as u32 - 1);
    }
    for index in 0..segments {
        let next = (index + 1) % segments;
        let base = mesh.vertices.len() as u32;
        for vertex in [bottom[index], bottom[next], top[next], top[index]] {
            let point = mesh.vertices[vertex as usize];
            let normal = mesh.normals[vertex as usize];
            mesh.vertices.push(point);
            mesh.normals.push(normal);
        }
        mesh.faces.push([base, base + 1, base + 2]);
        mesh.faces.push([base, base + 2, base + 3]);
        mesh.face_groups.push("face:side".into());
        mesh.face_groups.push("face:side".into());
    }
    let top_center = mesh.vertices.len() as u32;
    mesh.vertices.push([0.0, 0.0, (height / 2.0) as f32]);
    mesh.normals.push([0.0, 0.0, 1.0]);
    let bottom_center = mesh.vertices.len() as u32;
    mesh.vertices.push([0.0, 0.0, (-height / 2.0) as f32]);
    mesh.normals.push([0.0, 0.0, -1.0]);
    for index in 0..segments {
        let next = (index + 1) % segments;
        mesh.faces.push([top[index], top[next], top_center]);
        mesh.face_groups.push("face:top".into());
        mesh.faces.push([bottom[next], bottom[index], bottom_center]);
        mesh.face_groups.push("face:bottom".into());
    }
    mesh.lightweight = true;
    mesh
}

impl Kernel for ProceduralKernel {
    fn primitive(&mut self, kind: &str, values: &serde_json::Value) -> Result<Mesh, String> {
        match kind {
            "box" => Ok(box_mesh(
                number(values, "length", 40.0),
                number(values, "width", 40.0),
                number(values, "height", 40.0),
            )),
            "cylinder" => Ok(cylinder_mesh(
                number(values, "radius", 20.0),
                number(values, "height", 40.0),
                32,
            )),
            other => Err(format!("Rust experiment does not implement primitive {other:?} yet")),
        }
    }

    fn translate(&mut self, mesh: &Mesh, values: &serde_json::Value) -> Result<Mesh, String> {
        let (x, y, z) = (
            number(values, "x", 0.0),
            number(values, "y", 0.0),
            number(values, "z", 0.0),
        );
        let (rx, ry, rz) = (
            number(values, "rx", 0.0).to_radians(),
            number(values, "ry", 0.0).to_radians(),
            number(values, "rz", 0.0).to_radians(),
        );
        let rotate = |point: [f64; 3]| -> [f64; 3] {
            let (mut x, mut y, mut z) = (point[0], point[1], point[2]);
            let (sx, cx) = rx.sin_cos();
            let (sy, cy) = ry.sin_cos();
            let (sz, cz) = rz.sin_cos();
            (y, z) = (y * cx - z * sx, y * sx + z * cx);
            (x, z) = (x * cy + z * sy, -x * sy + z * cy);
            (x, y) = (x * cz - y * sz, x * sz + y * cz);
            [x, y, z]
        };
        let (x, y, z) = if values.get("space").and_then(serde_json::Value::as_str) == Some("local") {
            let offset = rotate([x, y, z]);
            (offset[0], offset[1], offset[2])
        } else {
            (x, y, z)
        };
        let mut transformed = mesh.clone();
        for vertex in &mut transformed.vertices {
            let point = rotate([vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]);
            vertex[0] = (point[0] + x) as f32;
            vertex[1] = (point[1] + y) as f32;
            vertex[2] = (point[2] + z) as f32;
        }
        for normal in &mut transformed.normals {
            let rotated = rotate([normal[0] as f64, normal[1] as f64, normal[2] as f64]);
            *normal = [rotated[0] as f32, rotated[1] as f32, rotated[2] as f32];
        }
        Ok(transformed)
    }
}

pub fn bounds(mesh: &Mesh) -> Option<([f32; 3], [f32; 3])> {
    let first = *mesh.vertices.first()?;
    let mut min = first;
    let mut max = first;
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_has_expected_bounds_and_faces() {
        let mesh = ProceduralKernel
            .primitive("box", &serde_json::json!({"length": 10, "width": 4, "height": 20}))
            .unwrap();
        assert_eq!(mesh.faces.len(), 12);
        let (min, max) = bounds(&mesh).unwrap();
        assert_eq!(min, [-5.0, -2.0, -10.0]);
        assert_eq!(max, [5.0, 2.0, 10.0]);
    }

    #[test]
    fn translate_moves_every_vertex() {
        let mut kernel = ProceduralKernel;
        let mesh = kernel.primitive("box", &serde_json::json!({})).unwrap();
        let moved = kernel.translate(&mesh, &serde_json::json!({"x": 5, "y": -2, "z": 3})).unwrap();
        let (min, _) = bounds(&moved).unwrap();
        assert_eq!(min[0], -20.0 + 5.0);
        assert_eq!(min[1], -20.0 - 2.0);
        assert_eq!(min[2], -20.0 + 3.0);
    }
    #[test]
    fn local_translation_uses_rotated_axes() {
        let mesh = ProceduralKernel.primitive("box", &serde_json::json!({"length": 2, "width": 2, "height": 2})).unwrap();
        let moved = ProceduralKernel
            .translate(&mesh, &serde_json::json!({"space": "local", "x": 10, "y": 0, "z": 0, "rz": 90}))
            .unwrap();
        let (min, max) = bounds(&moved).unwrap();
        assert!((min[0] + 1.0).abs() < 1e-4 && (max[0] - 1.0).abs() < 1e-4, "{min:?} {max:?}");
        assert!((min[1] - 9.0).abs() < 1e-4 && (max[1] - 11.0).abs() < 1e-4, "{min:?} {max:?}");
    }

    #[test]
    fn transform_rotates_then_translates() {
        let mesh = ProceduralKernel.primitive("box", &serde_json::json!({"length": 10, "width": 4, "height": 20})).unwrap();
        let rotated = ProceduralKernel.translate(&mesh, &serde_json::json!({"x": 100, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 90})).unwrap();
        let (min, max) = bounds(&rotated).unwrap();
        assert!((min[0] - 98.0).abs() < 1e-4 && (max[0] - 102.0).abs() < 1e-4, "{min:?} {max:?}");
        assert!((min[1] + 5.0).abs() < 1e-4 && (max[1] - 5.0).abs() < 1e-4, "{min:?} {max:?}");
    }

}
