use std::path::PathBuf;

use solidcore_stepio::import_step;

fn fixture(relative: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative).to_string_lossy().into_owned()
}

#[test]
fn imports_planar_block() {
    let imported = import_step(&fixture("../../../tests/Sample AP203 Data Files/block.stp")).unwrap();
    assert_eq!(imported.diagnostics.kind, "solidcore-brep");
    assert_eq!(imported.diagnostics.brep_faces, 6);
    assert!(!imported.mesh.faces.is_empty());
}

#[test]
fn imports_multi_solid_sample() {
    let imported = import_step(&fixture("../../../samples/rust-export.step")).unwrap();
    assert_eq!(imported.diagnostics.kind, "solidcore-brep");
    assert_eq!(imported.diagnostics.bodies, 6);
    assert!(!imported.mesh.faces.is_empty());
}

#[test]
fn imports_bspline_solid_fallback() {
    let imported = import_step(&fixture("../../../tests/Sample AP203 Data Files/moon_buggy.stp")).unwrap();
    assert_eq!(imported.diagnostics.kind, "solidcore-brep");
    assert!(!imported.mesh.faces.is_empty());
    assert!(!imported.mesh.brep_edges.is_empty(), "fallback solids need display edges");

}

#[test]
fn merged_meshes_namespace_groups_and_edges() {
    use nodecraft_protocol::{BrepEdge, Mesh};
    let mut first = Mesh::default();
    first.vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    first.normals = vec![[0.0, 0.0, 1.0]; 3];
    first.faces = vec![[0, 1, 2]];
    first.face_groups = vec!["face:0".into()];
    first.brep_edges = vec![BrepEdge { id: "step-edge:0".into(), kind: "step".into(), segments: vec![[0, 1]] }];
    let mut second = first.clone();
    second.vertices.iter_mut().for_each(|point| point[0] += 2.0);
    let merged = solidcore_stepio::merge_meshes(vec![first, second]);
    assert_eq!(merged.face_groups, ["body:0:face:0", "body:1:face:0"]);
    assert_eq!(merged.brep_edges[0].id, "body:0:step-edge:0");
    assert_eq!(merged.brep_edges[1].id, "body:1:step-edge:0");
    assert_eq!(merged.brep_edges[1].segments[0], [3, 4]);
}

#[test]
fn bspline_fixture_trims_cover_most_faces() {
    for path in [
        "../../../tests/Sample AP203 Data Files/moon_buggy.stp",
        "../../../tests/Sample AP203 Data Files/piston.stp",
        "../../../tests/Sample AP203 Data Files/tork.stp",
    ] {
        let imported = import_step(&fixture(path)).unwrap_or_else(|error| panic!("{path}: {error}"));
        assert_eq!(imported.diagnostics.kind, "solidcore-brep", "{path}");
        assert!(!imported.mesh.faces.is_empty(), "{path}");
    }
}
