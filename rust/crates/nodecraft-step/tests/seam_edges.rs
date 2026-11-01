use nodecraft_step::{read_topology_file, write_sphere, write_torus};

#[test]
fn sphere_seam_topology_is_consistent() {
    let path = std::env::temp_dir().join(format!("sn-seam-sphere-{}.step", std::process::id()));
    write_sphere(path.to_str().unwrap(), 5.0).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.faces.len(), 1);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());
}

#[test]
fn torus_seam_topology_is_consistent() {
    let path = std::env::temp_dir().join(format!("sn-seam-torus-{}.step", std::process::id()));
    write_torus(path.to_str().unwrap(), 8.0, 2.0).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.faces.len(), 1);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());
}
