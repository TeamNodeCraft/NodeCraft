//! Generates the 5x5 STEP test set in `STEP_TEST_TESS/`.
//!
//! These are deliberately small, simplified versions of the corpus cases: each file is
//! one body so a failure names the feature directly. Run from the repository root:
//! `cargo run -p nodecraft-step --example step_test_tess`.

use std::path::PathBuf;

use nodecraft_step::{
    write_box, write_box_with_void, write_cone, write_cylinder, write_open_shell, write_planar_body,
    write_pyramid, write_sphere, write_torus, write_wedge, PlanarBody,
};

fn out(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .join("STEP_TEST_TESS");
    std::fs::create_dir_all(&root).unwrap();
    root.join(name).to_string_lossy().into_owned()
}

fn prism(path: &str, polygon: &[[f64; 2]], height: f64, open: bool) -> Result<(), String> {
    let mut points = Vec::new();
    for &[x, y] in polygon {
        points.push([x, y, -height / 2.0]);
    }
    for &[x, y] in polygon {
        points.push([x, y, height / 2.0]);
    }
    let n = polygon.len();
    let mut faces = Vec::new();
    faces.push((0..n).rev().collect::<Vec<_>>());
    if !open {
        faces.push((n..2 * n).collect::<Vec<_>>());
    }
    for i in 0..n {
        let j = (i + 1) % n;
        faces.push(vec![i, j, n + j, n + i]);
    }
    write_planar_body(
        path,
        &PlanarBody {
            points,
            faces,
            ..PlanarBody::default()
        },
    )
}

fn circle(cx: f64, cy: f64, radius: f64, segments: usize) -> Vec<[f64; 2]> {
    (0..segments)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / segments as f64;
            [cx + radius * angle.cos(), cy + radius * angle.sin()]
        })
        .collect()
}

fn plate_with_holes(
    path: &str,
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    thickness: f64,
) -> Result<(), String> {
    let mut body = PlanarBody::default();
    let mut bottom_outer = Vec::new();
    let mut top_outer = Vec::new();
    for &[x, y] in outer {
        bottom_outer.push(body.points.len());
        body.points.push([x, y, -thickness / 2.0]);
    }
    for &[x, y] in outer {
        top_outer.push(body.points.len());
        body.points.push([x, y, thickness / 2.0]);
    }
    let mut bottom_holes = Vec::new();
    let mut top_holes = Vec::new();
    for hole in holes {
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for &[x, y] in hole {
            bottom.push(body.points.len());
            body.points.push([x, y, -thickness / 2.0]);
        }
        for &[x, y] in hole {
            top.push(body.points.len());
            body.points.push([x, y, thickness / 2.0]);
        }
        bottom_holes.push(bottom);
        top_holes.push(top);
    }
    let mut bottom = bottom_outer.clone();
    bottom.reverse();
    body.push_face(bottom, bottom_holes.iter().map(|loop_| loop_.iter().rev().copied().collect()).collect());
    body.push_face(top_outer.clone(), top_holes.iter().map(|loop_| loop_.iter().rev().copied().collect()).collect());

    let add_walls = |body: &mut PlanarBody, loop_bottom: &[usize], loop_top: &[usize], reverse: bool| {
        for index in 0..loop_bottom.len() {
            let next = (index + 1) % loop_bottom.len();
            let face = if reverse {
                vec![loop_bottom[next], loop_bottom[index], loop_top[index], loop_top[next]]
            } else {
                vec![loop_bottom[index], loop_bottom[next], loop_top[next], loop_top[index]]
            };
            body.push_face(face, Vec::new());
        }
    };
    add_walls(&mut body, &bottom_outer, &top_outer, false);
    for (bottom, top) in bottom_holes.iter().zip(&top_holes) {
        add_walls(&mut body, bottom, top, true);
    }
    write_planar_body(path, &body)
}

fn main() -> Result<(), String> {
    // Row 1: the simple solids.
    write_box(&out("r1c1_box.step"), 10.0, 10.0, 10.0)?;
    write_box(&out("r1c2_thin_plate.step"), 30.0, 30.0, 1.0)?;
    write_box(&out("r1c3_tall_box.step"), 5.0, 5.0, 30.0)?;
    write_wedge(&out("r1c4_wedge.step"), 20.0, 10.0, 10.0, 8.0)?;
    write_pyramid(&out("r1c5_pyramid.step"), 20.0, 15.0)?;

    // Row 2: planar topology and concave loops.
    write_box_with_void(&out("r2c1_box_with_void.step"), [20.0, 20.0, 20.0], [8.0, 8.0, 8.0])?;
    prism(&out("r2c2_l_prism.step"), &[[-7.0, -7.0], [7.0, -7.0], [7.0, 0.0], [0.0, 0.0], [0.0, 7.0], [-7.0, 7.0]], 8.0, false)?;
    prism(&out("r2c3_u_prism.step"), &[[-8.0, -8.0], [8.0, -8.0], [8.0, 8.0], [4.0, 8.0], [4.0, -2.0], [-4.0, -2.0], [-4.0, 8.0], [-8.0, 8.0]], 8.0, false)?;
    prism(&out("r2c4_plus_prism.step"), &[[-2.0, -7.0], [2.0, -7.0], [2.0, -2.0], [7.0, -2.0], [7.0, 2.0], [2.0, 2.0], [2.0, 7.0], [-2.0, 7.0], [-2.0, 2.0], [-7.0, 2.0], [-7.0, -2.0], [-2.0, -2.0]], 6.0, false)?;
    prism(&out("r2c5_stepped.step"), &[[-8.0, -6.0], [0.0, -6.0], [0.0, -3.0], [8.0, -3.0], [8.0, 6.0], [-8.0, 6.0]], 8.0, false)?;

    // Row 3: analytic curved surfaces.
    write_cylinder(&out("r3c1_cylinder.step"), 5.0, 10.0)?;
    write_cylinder(&out("r3c2_thin_cylinder.step"), 2.0, 20.0)?;
    write_cone(&out("r3c3_cone.step"), 6.0, 2.0, 10.0)?;
    write_sphere(&out("r3c4_sphere.step"), 5.0)?;
    write_torus(&out("r3c5_torus.step"), 8.0, 2.0)?;

    // Row 4: transforms and open topology.
    write_box(&out("r4c1_rotated_box.step"), 12.0, 4.0, 8.0)?;
    write_cylinder(&out("r4c2_tilted_cylinder.step"), 4.0, 12.0)?;
    write_sphere(&out("r4c3_off_origin_sphere.step"), 4.0)?;
    let mut open = PlanarBody::default();
    open.points = vec![
        [-5.0, -5.0, -5.0], [5.0, -5.0, -5.0], [5.0, 5.0, -5.0], [-5.0, 5.0, -5.0],
        [-5.0, -5.0, 5.0], [5.0, -5.0, 5.0], [5.0, 5.0, 5.0], [-5.0, 5.0, 5.0],
    ];
    open.faces = vec![
        vec![3, 2, 1, 0], vec![0, 1, 5, 4], vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7],
    ];
    write_open_shell(&out("r4c4_open_box.step"), &open)?;
    prism(&out("r4c5_skewed_prism.step"), &[[-6.0, -4.0], [3.0, -6.0], [7.0, -1.0], [2.0, 6.0], [-5.0, 4.0]], 7.0, false)?;

    // Row 5: scale and tolerance edge cases.
    write_box(&out("r5c1_tiny_box.step"), 0.2, 0.1, 0.3)?;
    write_box(&out("r5c2_large_box.step"), 2000.0, 1200.0, 800.0)?;
    write_cylinder(&out("r5c3_tiny_cylinder.step"), 0.25, 0.4)?;
    write_sphere(&out("r5c4_large_sphere.step"), 500.0)?;
    write_torus(&out("r5c5_flat_torus.step"), 10.0, 0.5)?;

    // Offset files so opening one is not always centred at the origin.
    for (name, rotation, offset) in [
        ("r4c1_rotated_box.step", [0.0, 0.0, 30.0], [20.0, 0.0, 0.0]),
        ("r4c2_tilted_cylinder.step", [30.0, 0.0, 0.0], [-20.0, 0.0, 0.0]),
        ("r4c3_off_origin_sphere.step", [0.0, 0.0, 0.0], [0.0, 25.0, 15.0]),
    ] {
        let path = out(name);
        nodecraft_step::transform_step_file(&path, rotation, offset)?;
    }

    // Row 6: moon-buggy-style face and loop stress cases.
    let rect = [[-40.0, -30.0], [40.0, -30.0], [40.0, 30.0], [-40.0, 30.0]];
    let mut perforated = Vec::new();
    for row in 0..4 {
        for column in 0..5 {
            perforated.push(circle(-28.0 + column as f64 * 14.0, -20.0 + row as f64 * 13.0, 4.0, 16));
        }
    }
    plate_with_holes(&out("r6c1b_one_hole.step"), &rect, &[circle(0.0, 0.0, 8.0, 20)], 4.0)?;
    plate_with_holes(&out("r6c1_perforated_plate.step"), &rect, &perforated, 4.0)?;

    let mut thin_holes = Vec::new();
    for index in 0..12 {
        thin_holes.push(circle(-30.0 + index as f64 * 5.5, 0.0, 2.0, 12));
    }
    plate_with_holes(&out("r6c2_thin_perforated.step"), &rect, &thin_holes, 1.2)?;

    let mut slots = Vec::new();
    for row in 0..4 {
        let cy = -18.0 + row as f64 * 12.0;
        for column in 0..3 {
            let cx = -24.0 + column as f64 * 24.0;
            slots.push(vec![[cx - 8.0, cy - 1.5], [cx + 8.0, cy - 1.5], [cx + 8.0, cy + 1.5], [cx - 8.0, cy + 1.5]]);
        }
    }
    plate_with_holes(&out("r6c3_slotted_plate.step"), &rect, &slots, 5.0)?;

    let l_outer = [[-35.0, -25.0], [25.0, -25.0], [25.0, -10.0], [-10.0, -10.0], [-10.0, 25.0], [-35.0, 25.0]];
    let l_holes = vec![circle(-22.0, -17.0, 4.0, 16), circle(6.0, -17.0, 4.0, 16), circle(-22.0, 8.0, 4.0, 16)];
    plate_with_holes(&out("r6c4_l_bracket_holes.step"), &l_outer, &l_holes, 4.0)?;

    let tangent = vec![circle(-4.05, 0.0, 4.0, 20), circle(4.05, 0.0, 4.0, 20), circle(0.0, 17.0, 5.0, 20)];
    plate_with_holes(&out("r6c5_tangent_holes.step"), &rect, &tangent, 4.0)?;

    // Row 7: harder loop/tessellation stress cases.
    let big_rect = [[-60.0, -40.0], [60.0, -40.0], [60.0, 40.0], [-60.0, 40.0]];
    let mut dense = Vec::new();
    for row in 0..10 {
        for column in 0..15 {
            dense.push(circle(-56.0 + column as f64 * 8.0, -36.0 + row as f64 * 8.0, 1.5, 10));
        }
    }
    plate_with_holes(&out("r7c1_dense_150_holes.step"), &big_rect, &dense, 3.0)?;

    let mut thin_ligaments = Vec::new();
    for row in 0..8 {
        for column in 0..10 {
            thin_ligaments.push(circle(-27.0 + column as f64 * 6.2, -21.0 + row as f64 * 6.2, 3.0, 14));
        }
    }
    plate_with_holes(&out("r7c2_thin_ligaments.step"), &[[-34.0, -28.0], [34.0, -28.0], [34.0, 28.0], [-34.0, 28.0]], &thin_ligaments, 3.0)?;

    let star: Vec<[f64; 2]> = (0..20)
        .map(|index| {
            let angle = std::f64::consts::TAU * index as f64 / 20.0;
            let radius = if index % 2 == 0 { 42.0 } else { 24.0 };
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect();
    let star_holes = vec![circle(-14.0, 0.0, 4.0, 16), circle(14.0, 0.0, 4.0, 16), circle(0.0, 15.0, 4.0, 16), circle(0.0, -15.0, 4.0, 16)];
    plate_with_holes(&out("r7c3_star_plate.step"), &star, &star_holes, 4.0)?;

    let mut edge_holes = Vec::new();
    for row in 0..8 {
        edge_holes.push(circle(-39.45, -28.0 + row as f64 * 8.0, 3.0, 14));
        edge_holes.push(circle(39.45, -28.0 + row as f64 * 8.0, 3.0, 14));
    }
    plate_with_holes(&out("r7c4_edge_holes.step"), &big_rect, &edge_holes, 3.0)?;

    let mut lattice = Vec::new();
    for row in 0..7 {
        for column in 0..11 {
            let cx = -50.0 + column as f64 * 10.0;
            let cy = -30.0 + row as f64 * 10.0;
            lattice.push(vec![[cx - 4.2, cy - 1.2], [cx + 4.2, cy - 1.2], [cx + 4.2, cy + 1.2], [cx - 4.2, cy + 1.2]]);
        }
    }
    plate_with_holes(&out("r7c5_slot_lattice.step"), &[[-58.0, -38.0], [58.0, -38.0], [58.0, 38.0], [-58.0, 38.0]], &lattice, 3.0)?;

    // Row 8: wheel-focused isolation cases.
    let wheel_outer = circle(0.0, 0.0, 20.0, 48);
    let wheel_inner = circle(0.0, 0.0, 12.0, 48);
    plate_with_holes(&out("r8c1_wheel_ring.step"), &wheel_outer, &[wheel_inner], 6.0)?;
    write_torus(&out("r8c2_wheel_torus.step"), 18.0, 6.0)?;
    write_cylinder(&out("r8c3_wheel_hub.step"), 8.0, 6.0)?;
    write_torus(&out("r8c4_wheel_torus_thin.step"), 20.0, 2.0)?;
    let rim_outer = circle(0.0, 0.0, 22.0, 64);
    let rim_inner = circle(0.0, 0.0, 18.0, 64);
    plate_with_holes(&out("r8c5_wheel_rim_thin.step"), &rim_outer, &[rim_inner], 2.0)?;

    std::fs::write(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .join("STEP_TEST_TESS/README.md"),
        "# STEP test grid\n\n5x5 files generated by `step_test_tess.rs`. Each row/column is one feature family:\n\n- row 1: box, plate, tall box, wedge, pyramid\n- row 2: void, concave L/U/plus, stepped prism\n- row 3: cylinder, thin cylinder, cone, sphere, torus\n- row 4: rotated, tilted, off-origin, open shell, skewed prism\n- row 5: tiny/large scale and flat torus\n\nLoad files individually; a failure names the feature in the filename.\n",
    )
    .unwrap();
    Ok(())
}
