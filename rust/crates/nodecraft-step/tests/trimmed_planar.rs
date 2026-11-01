//! Trimmed planar faces: holes in the model, reader, writer and triangulation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nodecraft_step::{
    PlanarBody, StepFile, read_planar_body, write_planar_body,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Bare face in the `z = 0` plane: 4x4 outer rectangle with a 2x2 hole.
/// Outer is counter-clockwise, the hole clockwise (STEP inner-bound convention).
fn plate_with_one_hole() -> PlanarBody {
    PlanarBody {
        points: vec![
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 1.0, 0.0],
        ],
        faces: vec![vec![0, 1, 2, 3]],
        holes: vec![vec![vec![4, 5, 6, 7]]],
    }
}

/// Closed slab (16 outer area, 2x2 through-hole, height 1) built from 10 faces.
///
/// front/back are the trimmed end faces, plus four outer side walls and four hole
/// side walls. Winding is outward everywhere (hole walls face into the void).
fn slab_with_through_hole() -> PlanarBody {
    let points = vec![
        // z = 0 outer + hole
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
        [1.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [3.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
        // z = 1 outer + hole (same order)
        [0.0, 0.0, 1.0],
        [4.0, 0.0, 1.0],
        [4.0, 4.0, 1.0],
        [0.0, 4.0, 1.0],
        [1.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [3.0, 3.0, 1.0],
        [1.0, 3.0, 1.0],
    ];
    let mut faces = Vec::new();
    let mut holes = Vec::new();
    // front (-Z): outer clockwise seen from +Z, hole counter-clockwise.
    faces.push(vec![3, 2, 1, 0]);
    holes.push(vec![vec![4, 5, 6, 7]]);
    // back (+Z): outer counter-clockwise, hole clockwise.
    faces.push(vec![8, 9, 10, 11]);
    holes.push(vec![vec![12, 15, 14, 13]]);
    // Outer side walls: [b, a, a_back, b_back] for every front outer edge.
    let front_outer = [3usize, 2, 1, 0];
    for index in 0..front_outer.len() {
        let a = front_outer[index];
        let b = front_outer[(index + 1) % front_outer.len()];
        faces.push(vec![b, a, a + 8, b + 8]);
        holes.push(Vec::new());
    }
    // Hole side walls: [b, a, a_back, b_back] on the front hole loop.
    let front_hole = [4usize, 5, 6, 7];
    for index in 0..front_hole.len() {
        let a = front_hole[index];
        let b = front_hole[(index + 1) % front_hole.len()];
        faces.push(vec![b, a, a + 8, b + 8]);
        holes.push(Vec::new());
    }
    PlanarBody { points, faces, holes }
}

/// Concave L-shaped face in `z = 0` with one rectangular hole.
fn l_shape_with_hole() -> PlanarBody {
    PlanarBody {
        points: vec![
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 6.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [1.5, 0.5, 0.0],
        ],
        faces: vec![vec![0, 1, 2, 3, 4, 5]],
        holes: vec![vec![vec![6, 7, 8, 9]]],
    }
}

fn face_area(body: &PlanarBody, triangles: &[[usize; 3]]) -> f64 {
    triangles
        .iter()
        .map(|[a, b, c]| {
            let a = body.points[*a];
            let b = body.points[*b];
            let c = body.points[*c];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
        })
        .sum()
}

/// Even-odd point-in-polygon in 3D on a planar loop (projected by dropping `axis`).
fn loop_contains(loop_points: &[[f64; 3]], point: [f64; 3]) -> bool {
    let axis = (0..3)
        .min_by(|left, right| {
            let extent = |axis: usize| {
                loop_points.iter().map(|p| p[axis]).fold(f64::NEG_INFINITY, f64::max)
                    - loop_points.iter().map(|p| p[axis]).fold(f64::INFINITY, f64::min)
            };
            extent(*left).partial_cmp(&extent(*right)).unwrap()
        })
        .unwrap();
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let mut inside = false;
    for index in 0..loop_points.len() {
        let a = loop_points[index];
        let b = loop_points[(index + 1) % loop_points.len()];
        if (a[v] > point[v]) != (b[v] > point[v]) {
            let x = a[u] + (point[v] - a[v]) / (b[v] - a[v]) * (b[u] - a[u]);
            if point[u] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn triangle_centroid(body: &PlanarBody, triangle: &[usize; 3]) -> [f64; 3] {
    let mut centroid = [0.0; 3];
    for index in triangle {
        for axis in 0..3 {
            centroid[axis] += body.points[*index][axis] / 3.0;
        }
    }
    centroid
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("nodecraft-step-trimmed-{}-{name}", std::process::id()))
}

/// Locate the workspace `.venv` interpreter (override with `NODECRAFT_PYTHON`).
fn python() -> Option<PathBuf> {
    if let Ok(configured) = std::env::var("NODECRAFT_PYTHON") {
        let path = PathBuf::from(configured);
        return path.exists().then_some(path);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.ancestors().nth(3)?;
    let candidate = root.join(".venv").join("bin").join("python");
    candidate.exists().then_some(candidate)
}

const IMPORT_SCRIPT: &str = r"
import glob, os, sys
root = sys.argv[1]
for candidate in glob.glob(os.path.join(root, '.venv', 'lib', 'python*', 'site-packages')):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
import smlibpy
path = sys.argv[2]
part = smlibpy.import_step(path, heal=False)
bbox = part.bbox()
fields = [
    str(len(part.faces())), str(len(part.edges())), str(len(part.vertices())),
    '1' if part.is_valid() else '0',
    repr(float(part.volume)), repr(float(part.area)),
    repr(float(bbox.min.x)), repr(float(bbox.min.y)), repr(float(bbox.min.z)),
    repr(float(bbox.max.x)), repr(float(bbox.max.y)), repr(float(bbox.max.z)),
]
print('|'.join(fields))
";

#[test]
fn smlibpy_reimports_the_trimmed_slab() {
    let Some(python) = python() else {
        eprintln!("skipping smlibpy trimmed round-trip: workspace .venv/bin/python not found");
        return;
    };
    let path = tmp_path("slab-smlibpy.step");
    write_planar_body(path.to_str().unwrap(), &slab_with_through_hole()).expect("write");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.ancestors().nth(3).expect("workspace root");
    let output = Command::new(&python)
        .arg("-c")
        .arg(IMPORT_SCRIPT)
        .arg(root)
        .arg(&path)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {python:?}: {error}"));
    assert!(
        output.status.success(),
        "smlibpy import failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let fields: Vec<&str> = stdout.trim().split('|').collect();
    assert_eq!(fields.len(), 12, "malformed smlibpy row: {stdout}");
    assert_eq!(fields[0].parse::<usize>().unwrap(), 10, "faces");
    assert_eq!(fields[1].parse::<usize>().unwrap(), 24, "edges");
    assert_eq!(fields[2].parse::<usize>().unwrap(), 16, "vertices");
    assert_eq!(fields[3], "1", "is_valid");
    assert!((fields[4].parse::<f64>().unwrap() - 12.0).abs() < 1e-9, "volume");
    assert!((fields[5].parse::<f64>().unwrap() - 48.0).abs() < 1e-9, "area");
    let bounds: Vec<f64> = fields[6..12].iter().map(|field| field.parse().unwrap()).collect();
    assert!((bounds[0] - 0.0).abs() < 1e-6 && (bounds[1] - 0.0).abs() < 1e-6 && (bounds[2] - 0.0).abs() < 1e-6);
    assert!((bounds[3] - 4.0).abs() < 1e-6 && (bounds[4] - 4.0).abs() < 1e-6 && (bounds[5] - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Triangulation
// ---------------------------------------------------------------------------

#[test]
fn triangulates_a_rectangular_face_with_one_hole() {
    let body = plate_with_one_hole();
    let triangles = body.triangulate().expect("triangulate");
    // Outer + hole + bridge duplicates: 4 + 4 + 2 vertices -> 8 triangles.
    assert_eq!(triangles.len(), 8);
    assert!((face_area(&body, &triangles) - 12.0).abs() < 1e-12);
    assert!((body.area() - 12.0).abs() < 1e-12);
    let hole = &body.holes[0][0];
    for triangle in &triangles {
        let centroid = triangle_centroid(&body, triangle);
        let hole_points: Vec<[f64; 3]> = hole.iter().map(|index| body.points[*index]).collect();
        assert!(
            !loop_contains(&hole_points, centroid),
            "triangle centroid {centroid:?} fell inside the hole"
        );
    }
    // Dropping the hole must change the area: the hole is really used.
    let mut solid_face = body.clone();
    solid_face.holes = vec![Vec::new()];
    assert!((solid_face.area() - 16.0).abs() < 1e-12);
}

#[test]
fn triangulates_a_face_with_two_holes() {
    let body = PlanarBody {
        points: vec![
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [4.0, 3.0, 0.0],
            [5.0, 3.0, 0.0],
            [5.0, 1.0, 0.0],
        ],
        faces: vec![vec![0, 1, 2, 3]],
        holes: vec![vec![vec![4, 5, 6, 7], vec![8, 9, 10, 11]]],
    };
    let triangles = body.triangulate().expect("triangulate");
    assert!((face_area(&body, &triangles) - 21.0).abs() < 1e-12);
    assert!((body.area() - 21.0).abs() < 1e-12);
    for hole in &body.holes[0] {
        let hole_points: Vec<[f64; 3]> = hole.iter().map(|index| body.points[*index]).collect();
        for triangle in &triangles {
            let centroid = triangle_centroid(&body, triangle);
            assert!(
                !loop_contains(&hole_points, centroid),
                "triangle centroid {centroid:?} fell inside hole {hole:?}"
            );
        }
    }
    // The two holes are distinct: each contributes its own 1x1 area.
    assert!((body.try_area().unwrap() - 21.0).abs() < 1e-12);
}

#[test]
fn triangulates_a_concave_outer_face_with_a_hole() {
    let body = l_shape_with_hole();
    let triangles = body.triangulate().expect("triangulate");
    let expected = (6.0 * 2.0 + 2.0 * 4.0) - 1.0; // L-shape area 20 minus 1x1 hole
    assert!((face_area(&body, &triangles) - expected).abs() < 1e-12);
    let outer: Vec<[f64; 3]> = body.faces[0].iter().map(|index| body.points[*index]).collect();
    let hole: Vec<[f64; 3]> = body.holes[0][0].iter().map(|index| body.points[*index]).collect();
    for triangle in &triangles {
        let centroid = triangle_centroid(&body, triangle);
        assert!(loop_contains(&outer, centroid), "triangulation escaped the outer loop");
        assert!(!loop_contains(&hole, centroid), "triangulation crossed the hole");
    }
}

#[test]
fn rejects_degenerate_holes_and_missing_points() {
    let mut body = plate_with_one_hole();
    body.holes[0][0] = vec![4, 5];
    assert!(body.triangulate().is_err(), "two-point hole must fail");

    let mut body = plate_with_one_hole();
    body.holes[0][0] = vec![4, 5, 99];
    assert!(body.triangulate().is_err(), "missing point must fail");

    let mut body = plate_with_one_hole();
    body.faces[0] = vec![0, 1];
    assert!(body.triangulate().is_err(), "two-point outer loop must fail");
}

// ---------------------------------------------------------------------------
// Writer / reader round-trip
// ---------------------------------------------------------------------------

#[test]
fn writes_and_reads_outer_and_inner_bounds() {
    let path = tmp_path("slab.step");
    let body = slab_with_through_hole();
    write_planar_body(path.to_str().unwrap(), &body).expect("write");
    let text = fs::read_to_string(&path).unwrap();
    let file = StepFile::parse(&text).unwrap();
    let count = |name: &str| file.entities.values().filter(|entity| entity.name == name).count();
    assert_eq!(count("ADVANCED_FACE"), 10);
    assert_eq!(count("FACE_OUTER_BOUND"), 10);
    assert_eq!(count("FACE_BOUND"), 2);
    assert!(file.dangling_refs().is_empty());

    let read = read_planar_body(path.to_str().unwrap()).expect("read");
    assert_eq!(read.faces.len(), 10);
    assert_eq!(read.holes.len(), 10, "holes stay index-aligned with faces");
    assert_eq!(read.holes.iter().filter(|holes| !holes.is_empty()).count(), 2);
    assert_eq!(read.holes[0].len(), 1);
    assert_eq!(read.holes[1].len(), 1);
    assert!((read.area() - 48.0).abs() < 1e-9);
    assert!((read.volume() - 12.0).abs() < 1e-9);
    assert!((body.area() - 48.0).abs() < 1e-12);
    assert!((body.volume() - 12.0).abs() < 1e-12);
}

#[test]
fn writer_preserves_loop_orientation_exactly() {
    let path = tmp_path("orientation.step");
    let body = plate_with_one_hole();
    write_planar_body(path.to_str().unwrap(), &body).expect("write");
    let read = read_planar_body(path.to_str().unwrap()).expect("read");
    assert_eq!(read.faces[0], body.faces[0], "outer loop order must round-trip");
    assert_eq!(read.holes[0][0], body.holes[0][0], "hole loop order must round-trip");
    assert_eq!(read, body);
}

#[test]
fn reader_honours_bound_orientation_flags() {
    let path = tmp_path("flags.step");
    let body = plate_with_one_hole();
    write_planar_body(path.to_str().unwrap(), &body).expect("write");
    let text = fs::read_to_string(&path).unwrap();
    let outer_bound = text
        .lines()
        .find(|line| line.contains("FACE_OUTER_BOUND"))
        .expect("outer bound");
    let inner_bound = text
        .lines()
        .find(|line| line.contains("FACE_BOUND"))
        .expect("inner bound");
    let flipped = text
        .replace(outer_bound, &outer_bound.replace(".T.);", ".F.);"))
        .replace(inner_bound, &inner_bound.replace(".T.);", ".F.);"));
    fs::write(&path, flipped).unwrap();
    let read = read_planar_body(path.to_str().unwrap()).expect("read");
    assert_eq!(
        read.faces[0],
        body.faces[0].iter().rev().copied().collect::<Vec<_>>(),
        "outer FACE_OUTER_BOUND .F. must reverse the loop"
    );
    assert_eq!(
        read.holes[0][0],
        body.holes[0][0].iter().rev().copied().collect::<Vec<_>>(),
        "FACE_BOUND .F. must reverse the hole loop"
    );
}

#[test]
fn reader_treats_first_face_bound_as_outer_when_no_outer_bound_exists() {
    let path = tmp_path("no-outer.step");
    write_planar_body(path.to_str().unwrap(), &plate_with_one_hole()).expect("write");
    let text = fs::read_to_string(&path).unwrap();
    let text = text.replace("FACE_OUTER_BOUND", "FACE_BOUND");
    fs::write(&path, text).unwrap();
    let read = read_planar_body(path.to_str().unwrap()).expect("first FACE_BOUND is the outer loop");
    assert_eq!(read.faces.len(), 1);
    assert_eq!(read.faces[0], vec![0, 1, 2, 3]);
    assert_eq!(read.holes.len(), 1);
    assert_eq!(read.holes[0].len(), 1);
}

#[test]
fn writer_rejects_malformed_trimmed_bodies() {
    let path = tmp_path("invalid.step").to_str().unwrap().to_string();

    let mut body = plate_with_one_hole();
    body.holes[0][0] = vec![4, 5];
    let error = write_planar_body(&path, &body).expect_err("short hole");
    assert!(error.contains("three vertices"), "{error}");

    let mut body = plate_with_one_hole();
    body.faces[0].push(99);
    let error = write_planar_body(&path, &body).expect_err("missing point");
    assert!(error.contains("missing point"), "{error}");

    let mut body = plate_with_one_hole();
    body.points[4][2] = 1.0; // move one hole vertex off the face plane
    let error = write_planar_body(&path, &body).expect_err("not coplanar");
    assert!(error.contains("not planar"), "{error}");

    let mut body = plate_with_one_hole();
    body.faces[0] = vec![0, 0, 1, 2];
    let error = write_planar_body(&path, &body).expect_err("repeated point");
    assert!(error.contains("repeats point"), "{error}");
}

#[test]
fn box_gains_no_holes_and_stays_green() {
    let path = tmp_path("box.step");
    nodecraft_step::write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).expect("write box");
    let body = read_planar_body(path.to_str().unwrap()).expect("read box");
    assert_eq!(body.faces.len(), 6);
    assert_eq!(body.holes.len(), 6);
    assert!(body.holes.iter().all(Vec::is_empty));
    assert!((body.area() - 640.0).abs() < 1e-9);
    assert!((body.volume() - 800.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Multi-hole caps (NIST CTC-01 regression shapes)
// ---------------------------------------------------------------------------

/// A closed 2D loop from a list of `z = 0` points appended to `body`.
fn add_loop(body: &mut PlanarBody, points: &[[f64; 2]]) -> Vec<usize> {
    points
        .iter()
        .map(|&[x, y]| {
            body.points.push([x, y, 0.0]);
            body.points.len() - 1
        })
        .collect()
}

/// A single planar face at `z = 0` from an outer loop and hole loops.
fn face_from_loops(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> PlanarBody {
    let mut body = PlanarBody::default();
    let outer = add_loop(&mut body, outer);
    let holes: Vec<Vec<usize>> = holes.iter().map(|hole| add_loop(&mut body, hole)).collect();
    body.push_face(outer, holes);
    body
}

fn regular_polygon(cx: f64, cy: f64, radius: f64, segments: usize) -> Vec<[f64; 2]> {
    (0..segments)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / segments as f64;
            [cx + radius * angle.cos(), cy + radius * angle.sin()]
        })
        .collect()
}

/// A capsule/stadium loop: two half-circle ends joined by straight sides.
fn stadium(cx: f64, cy: f64, half_length: f64, radius: f64, arc_segments: usize) -> Vec<[f64; 2]> {
    let mut points = Vec::with_capacity(arc_segments * 2);
    for index in 0..arc_segments {
        let angle = -std::f64::consts::FRAC_PI_2
            + std::f64::consts::PI * index as f64 / arc_segments as f64;
        points.push([cx + half_length + radius * angle.cos(), cy + radius * angle.sin()]);
    }
    for index in 0..arc_segments {
        let angle = std::f64::consts::FRAC_PI_2
            + std::f64::consts::PI * index as f64 / arc_segments as f64;
        points.push([cx - half_length + radius * angle.cos(), cy + radius * angle.sin()]);
    }
    points
}

fn area_2d(points: &[[f64; 2]]) -> f64 {
    let count = points.len();
    0.5 * (0..count)
        .map(|index| {
            points[index][0] * points[(index + 1) % count][1]
                - points[(index + 1) % count][0] * points[index][1]
        })
        .sum::<f64>()
        .abs()
}

fn point_in_triangle_2d(point: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let cross = |u: [f64; 2], v: [f64; 2], w: [f64; 2]| {
        (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    };
    let first = cross(a, b, point);
    let second = cross(b, c, point);
    let third = cross(c, a, point);
    (first > 0.0 && second > 0.0 && third > 0.0)
        || (first < 0.0 && second < 0.0 && third < 0.0)
}

/// Every sample inside the face is covered by exactly one triangle and no sample inside
/// a hole is covered. This is the "watertight cap with holes, no hole dropped" check.
fn assert_face_coverage(body: &PlanarBody, triangles: &[[usize; 3]], resolution: usize) {
    let outer: Vec<[f64; 3]> = body.faces[0].iter().map(|&index| body.points[index]).collect();
    let holes: Vec<Vec<[f64; 3]>> = body.holes[0]
        .iter()
        .map(|hole| hole.iter().map(|&index| body.points[index]).collect())
        .collect();
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for point in &outer {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    let mut covered = 0usize;
    let mut hole_samples = 0usize;
    for row in 0..resolution {
        for column in 0..resolution {
            // Offset by half a cell so samples never land exactly on an axis-aligned
            // triangle edge.
            let x = min_x + (column as f64 + 0.5) / resolution as f64 * (max_x - min_x);
            let y = min_y + (row as f64 + 0.5) / resolution as f64 * (max_y - min_y);
            let point = [x, y, 0.0];
            if !loop_contains(&outer, point) {
                continue;
            }
            if holes.iter().any(|hole| loop_contains(hole, point)) {
                hole_samples += 1;
                for triangle in triangles {
                    let a = body.points[triangle[0]];
                    let b = body.points[triangle[1]];
                    let c = body.points[triangle[2]];
                    assert!(
                        !point_in_triangle_2d(
                            [x, y],
                            [a[0], a[1]],
                            [b[0], b[1]],
                            [c[0], c[1]]
                        ),
                        "triangle {triangle:?} covers a hole at ({x}, {y})"
                    );
                }
                continue;
            }
            // Samples on a shared triangle edge belong to neither triangle under a
            // strict test; skip them rather than treating them as a gap.
            let edge_tolerance = (max_x - min_x + max_y - min_y) * 1.0e-6;
            let mut on_edge = false;
            for triangle in triangles {
                let a = body.points[triangle[0]];
                let b = body.points[triangle[1]];
                let c = body.points[triangle[2]];
                for [first, second] in [
                    [[a[0], a[1]], [b[0], b[1]]],
                    [[b[0], b[1]], [c[0], c[1]]],
                    [[c[0], c[1]], [a[0], a[1]]],
                ] {
                    let ux = second[0] - first[0];
                    let uy = second[1] - first[1];
                    let length_squared = ux * ux + uy * uy;
                    if length_squared <= 0.0 {
                        continue;
                    }
                    let t = (((x - first[0]) * ux + (y - first[1]) * uy) / length_squared)
                        .clamp(0.0, 1.0);
                    let dx = x - (first[0] + ux * t);
                    let dy = y - (first[1] + uy * t);
                    if (dx * dx + dy * dy).sqrt() <= edge_tolerance {
                        on_edge = true;
                        break;
                    }
                }
                if on_edge {
                    break;
                }
            }
            if on_edge {
                continue;
            }
            let mut count = 0usize;
            for triangle in triangles {
                let a = body.points[triangle[0]];
                let b = body.points[triangle[1]];
                let c = body.points[triangle[2]];
                if point_in_triangle_2d([x, y], [a[0], a[1]], [b[0], b[1]], [c[0], c[1]]) {
                    count += 1;
                }
            }
            assert_eq!(count, 1, "sample ({x}, {y}) covered {count} times");
            covered += 1;
        }
    }
    assert!(covered > resolution, "coverage test saw too little of the face");
    assert!(hole_samples > 0, "coverage test saw no hole area");
}

#[test]
fn triangulates_a_face_with_a_dense_grid_of_round_holes() {
    const COLUMNS: usize = 5;
    const ROWS: usize = 4;
    const SEGMENTS: usize = 12;
    const RADIUS: f64 = 4.0;
    let size = 120.0;
    let outer = vec![[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]];
    let mut holes = Vec::new();
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let cx = 15.0 + column as f64 * (size - 30.0) / (COLUMNS - 1) as f64;
            let cy = 15.0 + row as f64 * (size - 30.0) / (ROWS - 1) as f64;
            let mut hole = regular_polygon(cx, cy, RADIUS, SEGMENTS);
            // Mixed winding must be normalized, not trusted.
            if (row + column) % 2 == 0 {
                hole.reverse();
            }
            holes.push(hole);
        }
    }
    let body = face_from_loops(&outer, &holes);
    let triangles = body.triangulate().expect("dense hole grid triangulates");

    let expected = size * size
        - (COLUMNS * ROWS) as f64 * area_2d(&regular_polygon(0.0, 0.0, RADIUS, SEGMENTS));
    assert!((face_area(&body, &triangles) - expected).abs() < 1e-9);
    assert!((body.area() - expected).abs() < 1e-9);
    for hole in &body.holes[0] {
        let hole_points: Vec<[f64; 3]> = hole.iter().map(|&index| body.points[index]).collect();
        for triangle in &triangles {
            let centroid = triangle_centroid(&body, triangle);
            assert!(
                !loop_contains(&hole_points, centroid),
                "triangle centroid {centroid:?} fell inside a hole"
            );
        }
    }
    assert_face_coverage(&body, &triangles, 48);
}

#[test]
fn triangulates_a_large_cap_with_corner_holes_and_slots() {
    // Mirrors the NIST CTC-01 face that used to fail: four corner holes and two
    // rounded slots on a large rectangular cap.
    let outer = vec![
        [-200.0, -130.0],
        [200.0, -130.0],
        [200.0, 130.0],
        [-200.0, 130.0],
    ];
    let holes = vec![
        regular_polygon(-160.0, 90.0, 22.0, 40),
        regular_polygon(160.0, 90.0, 22.0, 40),
        regular_polygon(-160.0, -90.0, 22.0, 40),
        regular_polygon(160.0, -90.0, 22.0, 40),
        stadium(-40.0, 0.0, 50.0, 20.0, 20),
        stadium(90.0, 0.0, 30.0, 20.0, 20),
    ];
    let body = face_from_loops(&outer, &holes);
    let triangles = body.triangulate().expect("corner holes + slots triangulate");

    let expected = area_2d(&outer)
        - holes.iter().map(|hole| area_2d(hole)).sum::<f64>();
    assert!((face_area(&body, &triangles) - expected).abs() < 1e-8);
    assert_face_coverage(&body, &triangles, 64);
}

#[test]
fn triangulates_holes_that_share_a_bridge_anchor() {
    // Two vertically aligned holes with a thin ligament. The nearest visible ring
    // vertex for both is the same, which produces a weakly simple bridged ring; the
    // triangulator must still cut both holes out.
    let outer = vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
    let holes = vec![
        regular_polygon(50.0, 35.0, 15.0, 20),
        regular_polygon(50.0, 68.0, 10.0, 20),
        regular_polygon(20.0, 20.0, 8.0, 16),
    ];
    let body = face_from_loops(&outer, &holes);
    let triangles = body
        .triangulate()
        .expect("holes sharing a bridge anchor triangulate");
    let expected = 10_000.0 - holes.iter().map(|hole| area_2d(hole)).sum::<f64>();
    assert!((face_area(&body, &triangles) - expected).abs() < 1e-8);
    assert_face_coverage(&body, &triangles, 48);
}

