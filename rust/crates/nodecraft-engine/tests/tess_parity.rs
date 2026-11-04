//! Parity harness: the legacy display bridge versus the upstream tessellator.
//!
//! Run with `cargo test -p nodecraft-engine --test tess_parity -- --nocapture` to
//! see the measured counts before deleting the legacy path.

use nodecraft_solidcore::solidcore::kernel::Kernel;
use nodecraft_solidcore::solidcore::math::Point3;
use nodecraft_solidcore::solidcore::primitive::api_primitives as api;
use nodecraft_solidcore::solidcore::topology::Brep;

fn bounds(vertices: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (min, max)
}

fn compare(name: &str, mut brep: Brep) {
    let old_kernel = Kernel::default();
    let old = nodecraft_solidcore::curved::brep_display_mesh(
        &old_kernel,
        &brep,
        &nodecraft_solidcore::curved::CurvedTessOptions::default(),
    )
    .unwrap_or_else(|error| panic!("{name}: legacy mesh: {error}"));

    let mut new_kernel = Kernel::default();
    let new = nodecraft_layers::tessellate_kernel(&mut new_kernel, &mut brep, 0.05)
        .unwrap_or_else(|error| panic!("{name}: tess: {error}"));
    let new_mesh = &new.mesh;

    let old_bounds = bounds(&old.vertices);
    let new_bounds = bounds(&new_mesh.vertices);
    for axis in 0..3 {
        assert!((old_bounds.0[axis] - new_bounds.0[axis]).abs() < 0.02, "{name}: min axis {axis}");
        assert!((old_bounds.1[axis] - new_bounds.1[axis]).abs() < 0.02, "{name}: max axis {axis}");
    }

    let new_faces: std::collections::BTreeSet<_> = new_mesh.face_groups.iter().collect();
    assert_eq!(new_faces.len(), new.stats.faces, "{name}: face groups");
    assert!(!new_mesh.faces.is_empty(), "{name}: no triangles");
    assert_eq!(new_mesh.vertices.len(), new_mesh.normals.len(), "{name}: normal count");
    assert!(!new.edge_polylines.is_empty(), "{name}: no edge polylines");
    assert!(!new_mesh.brep_edges.is_empty(), "{name}: no protocol edges");

    println!(
        "{name}: old tris {} edges {} | tess tris {} faces {} edges {}",
        old.faces.len(),
        old.brep_edges.len(),
        new_mesh.faces.len(),
        new.stats.faces,
        new.edge_polylines.len()
    );
}

#[test]
fn legacy_and_tess_agree_on_primitive_bounds() {
    let mut kernel = Kernel::default();
    compare(
        "box",
        api::create_box(&mut kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 6.0, 4.0).expect("box"),
    );
    let mut kernel = Kernel::default();
    compare(
        "cylinder",
        api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 3.0, 8.0).expect("cylinder"),
    );
    let mut kernel = Kernel::default();
    compare(
        "sphere",
        api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), 4.0).expect("sphere"),
    );
}
