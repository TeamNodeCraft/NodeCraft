//! Derived/product/context entity output tests (Truck output/derive parity).

use std::fs;

use nodecraft_step::{read_header, read_topology, write_box, StepFile};

#[test]
fn box_output_has_product_context_and_topology_entities() {
    let path = std::env::temp_dir().join(format!("sn-derived-{}.step", std::process::id()));
    write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    let file = StepFile::parse(&text).unwrap();
    let brep = read_topology(&file);

    assert_eq!(brep.solids.len(), 1);
    assert_eq!(brep.shells.len(), 1);
    assert_eq!(brep.faces.len(), 6);
    assert_eq!(brep.edges.len(), 12);
    assert_eq!(brep.vertices.len(), 8);
    assert!(brep.validate().is_empty(), "{:?}", brep.validate());

    let names: Vec<&str> = file.entities.values().map(|entity| entity.name.as_str()).collect();
    for required in [
        "APPLICATION_CONTEXT",
        "PRODUCT_CONTEXT",
        "PRODUCT",
        "PRODUCT_DEFINITION",
        "PRODUCT_DEFINITION_SHAPE",
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        "MANIFOLD_SOLID_BREP",
        "CLOSED_SHELL",
        "ADVANCED_FACE",
        "AXIS2_PLACEMENT_3D",
        "PLANE",
    ] {
        assert!(names.contains(&required), "missing {required}");
    }
    let header = read_header(path.to_str().unwrap()).unwrap();
    assert_eq!(header.schema.as_deref(), Some("CONFIG_CONTROL_DESIGN"));
    assert!(header.has_iso_start && header.has_data && header.has_iso_end);
}
