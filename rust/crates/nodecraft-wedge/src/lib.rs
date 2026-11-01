//! Analytic display mesh for the NodeCraft wedge primitive.
//!
//! The wedge matches `smlibpy.parts.Wedge` (and therefore the Python worker's `wedge`
//! node): a `length` (X) by `width` (Y) base rectangle with a vertical high face,
//! `height` tall, at `x = -length / 2`, and a slope running down to the far bottom
//! edge at `x = +length / 2`, `z = -height / 2`. The optional `top_length`, measured
//! from the high `-X` end, replaces the sharp top edge with a flat top:
//!
//! * `top_length == 0` is a triangular prism (5 faces, 8 triangles);
//! * `0 < top_length < length` is a trapezoidal prism (6 faces, 12 triangles).
//!
//! The primitive is centered on the origin, matching the other NodeCraft display
//! meshes: the bounding box is `(-length/2, -width/2, -height/2)` to
//! `(length/2, width/2, height/2)`.
//!
//! Solidcore has no wedge constructor and no simple profile-extrusion entry point, so
//! the wedge is emitted as exact analytic planar triangles instead of being derived
//! from a B-rep. The result is display-only and marked [`Mesh::lightweight`]; bounds,
//! volume (`length * width * height / 2` sharp, `(length + top_length) * width *
//! height / 2` flat) and closedness are covered by the tests below.
//!
//! Faces are emitted in this order, and each becomes one face group:
//!
//! | group | face |
//! | --- | --- |
//! | `face:0` | bottom (`z = -height / 2`) |
//! | `face:1` | high vertical face (`x = -length / 2`) |
//! | `face:2` | slope |
//! | `face:3` | side at `y = -width / 2` |
//! | `face:4` | side at `y = +width / 2` |
//! | `face:5` | flat top, only when `top_length > 0` |
//!
//! Every face gets its own (unwelded) vertices with a flat outward normal, and
//! [`Mesh::brep_edges`] carries one closed perimeter loop per face for display edges.
//!
//! # Example
//!
//! ```
//! use nodecraft_wedge::wedge_mesh;
//!
//! // A ramp 20 long (X), 10 wide (Y), 8 high (Z).
//! let ramp = wedge_mesh(20.0, 10.0, 8.0, 0.0).unwrap();
//! assert_eq!(ramp.faces.len(), 8);
//! assert!(ramp.lightweight);
//!
//! // A flat top 5 long instead of the sharp edge: (20 + 5) / 2 * 8 * 10 = 1000.
//! let step = wedge_mesh(20.0, 10.0, 8.0, 5.0).unwrap();
//! assert_eq!(step.faces.len(), 12);
//! ```

use nodecraft_protocol::{BrepEdge, Mesh};

/// Build the centered display mesh for the NodeCraft wedge primitive.
///
/// `length` is along X, `width` along Y and `height` along Z, all positive. The high
/// vertical side is at `x = -length / 2`; the slope runs down towards `+X`.
/// `top_length` is the length of the flat top measured from the high `-X` side and
/// must satisfy `0 <= top_length < length`. `top_length == 0` keeps the sharp top
/// edge, giving a triangular prism.
///
/// # Errors
///
/// Returns a message matching the Python worker's validation when a dimension is not
/// finite, `length`, `width` or `height` is not positive, or `top_length` is outside
/// `[0, length)`.
pub fn wedge_mesh(
    length: f64,
    width: f64,
    height: f64,
    top_length: f64,
) -> Result<Mesh, String> {
    if !(length.is_finite() && width.is_finite() && height.is_finite() && top_length.is_finite())
    {
        return Err("wedge dimensions must be finite".into());
    }
    if !(length > 0.0 && width > 0.0 && height > 0.0) {
        return Err("wedge length, width and height must be positive".into());
    }
    if !(top_length >= 0.0 && top_length < length) {
        return Err("wedge top_length must be at least 0 and less than the length".into());
    }

    let half_length = length * 0.5;
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let x_high = -half_length;
    let x_low = half_length;
    // X of the far end of the flat top, measured from the high `-X` side.
    let x_top = x_high + top_length;
    let y_min = -half_width;
    let y_max = half_width;
    let z_min = -half_height;
    let z_max = half_height;

    let mut mesh = Mesh::default();
    let mut face_index = 0usize;
    let mut edge_index = 0usize;

    // Bottom rectangle at z = -height / 2.
    push_face(
        &mut mesh,
        &[
            [x_high, y_min, z_min],
            [x_low, y_min, z_min],
            [x_low, y_max, z_min],
            [x_high, y_max, z_min],
        ],
        [0.0, 0.0, -1.0],
        &mut face_index,
        &mut edge_index,
    );

    // High vertical face at x = -length / 2.
    push_face(
        &mut mesh,
        &[
            [x_high, y_min, z_min],
            [x_high, y_min, z_max],
            [x_high, y_max, z_max],
            [x_high, y_max, z_min],
        ],
        [-1.0, 0.0, 0.0],
        &mut face_index,
        &mut edge_index,
    );

    // Slope from the high top edge down to the far bottom edge. Its outward normal is
    // the XZ vector perpendicular to the slope direction (length - top_length, -height).
    push_face(
        &mut mesh,
        &[
            [x_top, y_min, z_max],
            [x_low, y_min, z_min],
            [x_low, y_max, z_min],
            [x_top, y_max, z_max],
        ],
        [height, 0.0, length - top_length],
        &mut face_index,
        &mut edge_index,
    );

    if top_length > 0.0 {
        // Side faces at y = -width / 2 and y = +width / 2 (trapezoids).
        push_face(
            &mut mesh,
            &[
                [x_high, y_min, z_min],
                [x_low, y_min, z_min],
                [x_top, y_min, z_max],
                [x_high, y_min, z_max],
            ],
            [0.0, -1.0, 0.0],
            &mut face_index,
            &mut edge_index,
        );
        push_face(
            &mut mesh,
            &[
                [x_high, y_max, z_min],
                [x_low, y_max, z_min],
                [x_top, y_max, z_max],
                [x_high, y_max, z_max],
            ],
            [0.0, 1.0, 0.0],
            &mut face_index,
            &mut edge_index,
        );

        // Flat top at z = +height / 2.
        push_face(
            &mut mesh,
            &[
                [x_high, y_min, z_max],
                [x_top, y_min, z_max],
                [x_top, y_max, z_max],
                [x_high, y_max, z_max],
            ],
            [0.0, 0.0, 1.0],
            &mut face_index,
            &mut edge_index,
        );
    } else {
        // Side faces at y = -width / 2 and y = +width / 2 (triangles).
        push_face(
            &mut mesh,
            &[
                [x_high, y_min, z_min],
                [x_low, y_min, z_min],
                [x_high, y_min, z_max],
            ],
            [0.0, -1.0, 0.0],
            &mut face_index,
            &mut edge_index,
        );
        push_face(
            &mut mesh,
            &[
                [x_high, y_max, z_min],
                [x_low, y_max, z_min],
                [x_high, y_max, z_max],
            ],
            [0.0, 1.0, 0.0],
            &mut face_index,
            &mut edge_index,
        );
    }

    mesh.lightweight = true;
    Ok(mesh)
}

/// Push one planar polygon as a fan triangulation with flat outward normals.
///
/// The polygon is reversed when its geometric winding disagrees with `outward`, so the
/// emitted triangles always wind outwards. The face's own vertices and normals are
/// appended, followed by the triangles, one face-group label per triangle, and a
/// perimeter loop in [`Mesh::brep_edges`].
fn push_face(
    mesh: &mut Mesh,
    polygon: &[[f64; 3]],
    outward: [f64; 3],
    face_index: &mut usize,
    edge_index: &mut usize,
) {
    debug_assert!(polygon.len() >= 3);
    let mut polygon = polygon.to_vec();
    let normal = normalize(outward);

    if dot(triangle_normal(polygon[0], polygon[1], polygon[2]), normal) < 0.0 {
        polygon.reverse();
    }

    let group = format!("face:{}", *face_index);
    let base = mesh.vertices.len() as u32;
    for point in &polygon {
        mesh.vertices
            .push([point[0] as f32, point[1] as f32, point[2] as f32]);
        mesh.normals
            .push([normal[0] as f32, normal[1] as f32, normal[2] as f32]);
    }

    for index in 1..polygon.len() - 1 {
        mesh.faces
            .push([base, base + index as u32, base + index as u32 + 1]);
        mesh.face_groups.push(group.clone());
    }

    let count = polygon.len() as u32;
    mesh.brep_edges.push(BrepEdge {
        id: format!("mesh-edge:{}", *edge_index),
        kind: "mesh".into(),
        segments: (0..count).map(|index| [base + index, base + (index + 1) % count]).collect(),
    });

    *face_index += 1;
    *edge_index += 1;
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    cross(
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
    )
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = dot(vector, vector).sqrt();
    if length > 0.0 {
        [vector[0] / length, vector[1] / length, vector[2] / length]
    } else {
        vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    fn bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = mesh.vertices[0];
        let mut max = mesh.vertices[0];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        (min, max)
    }

    fn assert_bounds(mesh: &Mesh, expected_min: [f32; 3], expected_max: [f32; 3]) {
        let (min, max) = bounds(mesh);
        for axis in 0..3 {
            assert!(
                (min[axis] - expected_min[axis]).abs() < 1e-5,
                "min[{axis}] is {} expected {}",
                min[axis],
                expected_min[axis]
            );
            assert!(
                (max[axis] - expected_max[axis]).abs() < 1e-5,
                "max[{axis}] is {} expected {}",
                max[axis],
                expected_max[axis]
            );
        }
    }

    fn point(mesh: &Mesh, index: u32) -> [f64; 3] {
        let vertex = mesh.vertices[index as usize];
        [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64]
    }

    fn centroid(mesh: &Mesh, triangle: [u32; 3]) -> [f64; 3] {
        let a = point(mesh, triangle[0]);
        let b = point(mesh, triangle[1]);
        let c = point(mesh, triangle[2]);
        [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ]
    }

    fn group_index(mesh: &Mesh, triangle_index: usize) -> usize {
        mesh.face_groups[triangle_index]
            .strip_prefix("face:")
            .expect("face group should use the face:<index> form")
            .parse()
            .expect("face group index should be a number")
    }

    /// Signed volume of the closed triangle mesh (divergence theorem). Positive means
    /// the triangles wind outwards.
    fn signed_volume(mesh: &Mesh) -> f64 {
        let mut accumulated = 0.0;
        for triangle in &mesh.faces {
            let a = point(mesh, triangle[0]);
            let b = point(mesh, triangle[1]);
            let c = point(mesh, triangle[2]);
            accumulated += dot(a, cross(b, c));
        }
        accumulated / 6.0
    }

    /// Assert the mesh is closed: after welding vertices by their exact `f32`
    /// coordinates, every undirected triangle edge must be shared by exactly two
    /// triangles, and no triangle may be degenerate.
    fn assert_closed(mesh: &Mesh) {
        let key = |vertex: [f32; 3]| -> [u32; 3] {
            vertex.map(|value| if value == 0.0 { 0.0f32.to_bits() } else { value.to_bits() })
        };
        let mut edges: HashMap<([u32; 3], [u32; 3]), usize> = HashMap::new();
        for (triangle_index, triangle) in mesh.faces.iter().enumerate() {
            let keys = triangle.map(|index| key(mesh.vertices[index as usize]));
            let normal = triangle_normal(
                point(mesh, triangle[0]),
                point(mesh, triangle[1]),
                point(mesh, triangle[2]),
            );
            assert!(
                dot(normal, normal) > 0.0,
                "triangle {triangle_index} is degenerate"
            );
            for (start, end) in [(0usize, 1usize), (1, 2), (2, 0)] {
                let (first, second) = if keys[start] <= keys[end] {
                    (keys[start], keys[end])
                } else {
                    (keys[end], keys[start])
                };
                *edges.entry((first, second)).or_default() += 1;
            }
        }
        assert!(!edges.is_empty(), "mesh has no edges");
        for (edge, count) in edges {
            assert_eq!(count, 2, "edge {edge:?} is used by {count} triangles");
        }
    }

    /// Assert every triangle faces away from an interior reference point.
    fn assert_outward(mesh: &Mesh) {
        let count = mesh.vertices.len() as f64;
        let mut interior = [0.0f64; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                interior[axis] += vertex[axis] as f64;
            }
        }
        for axis in 0..3 {
            interior[axis] /= count;
        }
        for (triangle_index, triangle) in mesh.faces.iter().enumerate() {
            let normal = normalize(triangle_normal(
                point(mesh, triangle[0]),
                point(mesh, triangle[1]),
                point(mesh, triangle[2]),
            ));
            let center = centroid(mesh, *triangle);
            let offset = [
                center[0] - interior[0],
                center[1] - interior[1],
                center[2] - interior[2],
            ];
            assert!(
                dot(normal, offset) > 0.0,
                "triangle {triangle_index} does not face outwards"
            );
        }
    }

    fn assert_flat_normals(mesh: &Mesh) {
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        for triangle in &mesh.faces {
            let geometric = normalize(triangle_normal(
                point(mesh, triangle[0]),
                point(mesh, triangle[1]),
                point(mesh, triangle[2]),
            ));
            for index in triangle {
                let normal = mesh.normals[*index as usize];
                let normal = [normal[0] as f64, normal[1] as f64, normal[2] as f64];
                assert!((dot(normal, normal) - 1.0).abs() < 1e-5, "normal is not a unit vector");
                assert!(
                    dot(normal, geometric) > 1.0 - 1e-5,
                    "vertex normal disagrees with the triangle winding"
                );
            }
        }
    }

    fn assert_perimeter_edges(mesh: &Mesh, expected: usize) {
        assert_eq!(mesh.brep_edges.len(), expected);
        let mut ids = BTreeSet::new();
        for edge in &mesh.brep_edges {
            assert!(ids.insert(edge.id.clone()), "duplicate edge id {}", edge.id);
            assert_eq!(edge.kind, "mesh");
            assert!(!edge.segments.is_empty(), "edge {} has no segments", edge.id);
            for segment in &edge.segments {
                for index in segment {
                    assert!(
                        (*index as usize) < mesh.vertices.len(),
                        "edge {} references vertex {index}",
                        edge.id
                    );
                }
            }
        }
    }

    /// Check each face group's triangles against the expected plane and outward normal.
    fn assert_face_planes(mesh: &Mesh, length: f64, width: f64, height: f64, top_length: f64) {
        let half_length = length * 0.5;
        let half_width = width * 0.5;
        let half_height = height * 0.5;
        let slope_normal = normalize([height, 0.0, length - top_length]);
        let slope_zero = height * (-half_length + top_length) + (length - top_length) * half_height;
        let slope_scale = (height * height + (length - top_length).powi(2)).sqrt();

        for (triangle_index, triangle) in mesh.faces.iter().enumerate() {
            let group = group_index(mesh, triangle_index);
            let normal = normalize(triangle_normal(
                point(mesh, triangle[0]),
                point(mesh, triangle[1]),
                point(mesh, triangle[2]),
            ));
            let center = centroid(mesh, *triangle);
            let (expected_normal, on_plane) = match group {
                0 => ([0.0, 0.0, -1.0], (center[2] + half_height).abs() < 1e-5),
                1 => ([-1.0, 0.0, 0.0], (center[0] + half_length).abs() < 1e-5),
                2 => (
                    slope_normal,
                    (height * center[0] + (length - top_length) * center[2] - slope_zero).abs()
                        / slope_scale
                        < 1e-5,
                ),
                3 => ([0.0, -1.0, 0.0], (center[1] + half_width).abs() < 1e-5),
                4 => ([0.0, 1.0, 0.0], (center[1] - half_width).abs() < 1e-5),
                5 => ([0.0, 0.0, 1.0], (center[2] - half_height).abs() < 1e-5),
                other => panic!("unexpected face group face:{other}"),
            };
            assert!(
                dot(normal, expected_normal) > 1.0 - 1e-5,
                "triangle {triangle_index} in face:{group} has the wrong normal"
            );
            assert!(
                on_plane,
                "triangle {triangle_index} in face:{group} is off its face plane"
            );
        }
    }

    #[test]
    fn sharp_wedge_matches_smlibpy_triangular_prism() {
        // smlibpy: Wedge(20, 10, 8) -> bbox (-10, -5, -4)..(10, 5, 4), volume 800.
        let mesh = wedge_mesh(20.0, 10.0, 8.0, 0.0).unwrap();
        assert!(mesh.lightweight);
        assert_eq!(mesh.faces.len(), 8);
        assert_eq!(mesh.vertices.len(), 18);
        assert_eq!(mesh.normals.len(), 18);
        assert_eq!(mesh.face_groups.len(), mesh.faces.len());
        assert_bounds(&mesh, [-10.0, -5.0, -4.0], [10.0, 5.0, 4.0]);

        // Five planar faces: bottom, high, slope and the two triangular sides.
        let groups: BTreeSet<&str> = mesh.face_groups.iter().map(String::as_str).collect();
        assert_eq!(
            groups,
            BTreeSet::from(["face:0", "face:1", "face:2", "face:3", "face:4"])
        );
        assert_perimeter_edges(&mesh, 5);

        assert_face_planes(&mesh, 20.0, 10.0, 8.0, 0.0);
        assert_flat_normals(&mesh);
        assert_outward(&mesh);
        assert_closed(&mesh);

        let volume = signed_volume(&mesh);
        assert!((volume - 800.0).abs() < 1e-3, "volume is {volume}, expected 800");

        // The sharp top edge sits at x = -length / 2, z = +height / 2.
        for vertex in &mesh.vertices {
            if (vertex[2] - 4.0).abs() < 1e-6 {
                assert!((vertex[0] + 10.0).abs() < 1e-6, "top edge is not at x = -10");
            }
        }
    }

    #[test]
    fn flat_top_wedge_matches_smlibpy_trapezoid() {
        // smlibpy: Wedge(20, 10, 8, 5) -> bbox unchanged, volume (20 + 5) / 2 * 8 * 10.
        let mesh = wedge_mesh(20.0, 10.0, 8.0, 5.0).unwrap();
        assert!(mesh.lightweight);
        assert_eq!(mesh.faces.len(), 12);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.normals.len(), 24);
        assert_eq!(mesh.face_groups.len(), mesh.faces.len());
        assert_bounds(&mesh, [-10.0, -5.0, -4.0], [10.0, 5.0, 4.0]);

        let groups: BTreeSet<&str> = mesh.face_groups.iter().map(String::as_str).collect();
        assert_eq!(
            groups,
            BTreeSet::from(["face:0", "face:1", "face:2", "face:3", "face:4", "face:5"])
        );
        assert_perimeter_edges(&mesh, 6);

        assert_face_planes(&mesh, 20.0, 10.0, 8.0, 5.0);
        assert_flat_normals(&mesh);
        assert_outward(&mesh);
        assert_closed(&mesh);

        let volume = signed_volume(&mesh);
        assert!((volume - 1000.0).abs() < 1e-3, "volume is {volume}, expected 1000");

        // The flat top (face:5) spans x = -10 .. -5 at z = +4; the slope starts there.
        let mut top_vertices = 0;
        for (triangle_index, triangle) in mesh.faces.iter().enumerate() {
            if group_index(&mesh, triangle_index) != 5 {
                continue;
            }
            for index in triangle {
                let vertex = mesh.vertices[*index as usize];
                assert!((vertex[2] - 4.0).abs() < 1e-6, "top face is not at z = 4");
                assert!(
                    (-10.0f32..=-5.0f32).contains(&vertex[0]),
                    "top face x = {} is outside -10..-5",
                    vertex[0]
                );
                top_vertices += 1;
            }
        }
        assert_eq!(top_vertices, 6, "flat top should be two triangles");
    }

    #[test]
    fn top_length_zero_is_a_sharp_edge_even_when_negative_zero() {
        let sharp = wedge_mesh(20.0, 10.0, 8.0, 0.0).unwrap();
        let negative_zero = wedge_mesh(20.0, 10.0, 8.0, -0.0).unwrap();
        assert_eq!(sharp.faces.len(), negative_zero.faces.len());
        assert_eq!(sharp.face_groups, negative_zero.face_groups);
        assert_eq!(sharp.vertices, negative_zero.vertices);
        assert!((signed_volume(&sharp) - 800.0).abs() < 1e-3);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        for (length, width, height) in [
            (0.0, 10.0, 8.0),
            (-20.0, 10.0, 8.0),
            (20.0, 0.0, 8.0),
            (20.0, -10.0, 8.0),
            (20.0, 10.0, 0.0),
            (20.0, 10.0, -8.0),
        ] {
            assert!(
                wedge_mesh(length, width, height, 0.0).is_err(),
                "({length}, {width}, {height}) should be rejected"
            );
        }

        for top_length in [-0.5, 20.0, 21.0] {
            assert!(
                wedge_mesh(20.0, 10.0, 8.0, top_length).is_err(),
                "top_length {top_length} should be rejected"
            );
        }

        assert!(wedge_mesh(f64::NAN, 10.0, 8.0, 0.0).is_err());
        assert!(wedge_mesh(20.0, f64::NAN, 8.0, 0.0).is_err());
        assert!(wedge_mesh(20.0, 10.0, f64::NAN, 0.0).is_err());
        assert!(wedge_mesh(20.0, 10.0, 8.0, f64::NAN).is_err());
        assert!(wedge_mesh(f64::INFINITY, 10.0, 8.0, 0.0).is_err());
        assert!(wedge_mesh(20.0, 10.0, f64::INFINITY, 0.0).is_err());

        // Boundaries that are still valid.
        assert!(wedge_mesh(20.0, 10.0, 8.0, 0.0).is_ok());
        assert!(wedge_mesh(20.0, 10.0, 8.0, 19.5).is_ok());
    }

    #[test]
    fn all_face_groups_are_contiguous_and_parallel_to_faces() {
        for top_length in [0.0, 2.0, 19.5] {
            let mesh = wedge_mesh(20.0, 10.0, 8.0, top_length).unwrap();
            assert_eq!(mesh.face_groups.len(), mesh.faces.len());
            let mut counts = [0usize; 6];
            for triangle_index in 0..mesh.faces.len() {
                counts[group_index(&mesh, triangle_index)] += 1;
            }
            let expected = if top_length > 0.0 {
                [2, 2, 2, 2, 2, 2]
            } else {
                [2, 2, 2, 1, 1, 0]
            };
            assert_eq!(counts, expected, "top_length {top_length}");
        }
    }

    #[test]
    fn face_groups_keep_stable_indices_for_both_shapes() {
        let sharp = wedge_mesh(20.0, 10.0, 8.0, 0.0).unwrap();
        let flat = wedge_mesh(20.0, 10.0, 8.0, 5.0).unwrap();
        // face:0..face:4 always mean bottom, high, slope, -Y side, +Y side; the
        // optional top is always the last group (face:5).
        for (mesh, slope_normal) in [
            (&sharp, normalize([8.0, 0.0, 20.0])),
            (&flat, normalize([8.0, 0.0, 15.0])),
        ] {
            let expected_normals = [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                slope_normal,
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ];
            for (group, expected) in expected_normals.iter().enumerate() {
                for (triangle_index, triangle) in mesh.faces.iter().enumerate() {
                    if group_index(mesh, triangle_index) != group {
                        continue;
                    }
                    let normal = normalize(triangle_normal(
                        point(mesh, triangle[0]),
                        point(mesh, triangle[1]),
                        point(mesh, triangle[2]),
                    ));
                    assert!(dot(normal, *expected) > 1.0 - 1e-5);
                }
            }
        }
    }
}
