use step_assemblies::{AssemblyNode, parse_assembly_file, planar_instances, write_assembly_tree, Transform};

fn translation(x: f64) -> Transform {
    Transform::from_placement([x, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).unwrap()
}

#[test]
fn writes_and_parses_nested_assembly() {
    let dir = std::env::temp_dir().join(format!("sn-nested-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("part.step");
    nodecraft_step::write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
    let body = nodecraft_step::read_planar_body(path.to_str().unwrap()).unwrap();

    let root = [
        AssemblyNode::part("A", body.clone()).placed(translation(10.0)),
        AssemblyNode::group("B", vec![AssemblyNode::part("C", body).placed(translation(5.0))]).placed(translation(20.0)),
    ];
    let out = dir.join("nested.step");
    write_assembly_tree(out.to_str().unwrap(), "ROOT", &root).unwrap();

    let assembly = parse_assembly_file(&out).unwrap();
    let root_node = assembly.root().unwrap();
    assert_eq!(root_node.children.len(), 2, "{:?}", root_node);
    assert_eq!(root_node.children[0].product_name, "A");
    assert_eq!(root_node.children[0].world_transform.translation(), [10.0, 0.0, 0.0]);
    assert_eq!(root_node.children[1].product_name, "B");
    assert_eq!(root_node.children[1].world_transform.translation(), [20.0, 0.0, 0.0]);
    assert_eq!(root_node.children[1].children.len(), 1);
    assert_eq!(root_node.children[1].children[0].product_name, "C");
    assert_eq!(root_node.children[1].children[0].world_transform.translation(), [25.0, 0.0, 0.0]);
    assert_eq!(planar_instances(out.to_str().unwrap()).unwrap().len(), 2);
}
