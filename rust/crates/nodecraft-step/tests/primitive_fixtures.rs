use nodecraft_step::{read_topology_file, write_box, write_wedge};

#[test]
fn box_topology_fixture() {
    let path = std::env::temp_dir().join(format!("sn-fixture-box-{}.step", std::process::id()));
    write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.shells.len(), 1);
    assert_eq!(brep.faces.len(), 6);
    assert_eq!(brep.edges.len(), 12);
    assert_eq!(brep.vertices.len(), 8);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());
}

#[test]
fn wedge_topology_fixture() {
    let path = std::env::temp_dir().join(format!("sn-fixture-wedge-{}.step", std::process::id()));
    write_wedge(path.to_str().unwrap(), 10.0, 4.0, 6.0, 2.0).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.faces.len(), 6);
    assert_eq!(brep.edges.len(), 12);
    assert_eq!(brep.vertices.len(), 8);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());
}

use nodecraft_step::{write_cone, write_cylinder, write_sphere, write_torus};

#[test]
fn curved_primitive_topology_fixtures() {
    let dir = std::env::temp_dir().join(format!("sn-fixture-curved-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let cases: [(&str, Box<dyn Fn(&str) -> Result<(), String>>, usize, usize, usize); 4] = [
        ("cylinder", Box::new(|path| write_cylinder(path, 5.0, 10.0)), 3, 3, 2),
        ("cone", Box::new(|path| write_cone(path, 5.0, 2.0, 10.0)), 3, 3, 2),
        ("sphere", Box::new(|path| write_sphere(path, 5.0)), 1, 1, 2),
        ("torus", Box::new(|path| write_torus(path, 8.0, 2.0)), 1, 2, 1),
    ];
    for (name, write, faces, edges, vertices) in cases {
        let path = dir.join(format!("{name}.step"));
        write(path.to_str().unwrap()).unwrap();
        let brep = read_topology_file(path.to_str().unwrap()).unwrap();
        assert_eq!(brep.faces.len(), faces, "{name} faces");
        assert_eq!(brep.edges.len(), edges, "{name} edges");
        assert_eq!(brep.vertices.len(), vertices, "{name} vertices");
        assert!(brep.validate().is_empty(), "{name}: {:?}", brep.validate());
    }
}

#[test]
fn hollow_box_topology_fixture() {
    let path = std::env::temp_dir().join(format!("sn-fixture-hollow-{}.step", std::process::id()));
    nodecraft_step::write_box_with_void(path.to_str().unwrap(), [10.0, 10.0, 10.0], [4.0, 4.0, 4.0]).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.shells.len(), 2);
    assert_eq!(brep.faces.len(), 12);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());
}
