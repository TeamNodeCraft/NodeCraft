use nodecraft_step::{read_topology_file, write_box};

#[test]
fn reads_reversed_oriented_edge_flag() {
    let path = std::env::temp_dir().join(format!("sn-reversed-edge-{}.step", std::process::id()));
    write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
    let mut text = std::fs::read_to_string(&path).unwrap();
    let marker = "ORIENTED_EDGE('',*,*,";
    let start = text.find(marker).expect("oriented edge");
    let tail = &text[start..];
    let flag = tail.find(",.T.);").expect("orientation flag");
    let index = start + flag;
    text.replace_range(index..index + 5, ",.F.)");
    std::fs::write(&path, text).unwrap();
    let brep = read_topology_file(path.to_str().unwrap()).unwrap();
    assert!(brep.loops.values().any(|loop_| loop_.oriented_edges.iter().any(|use_| !use_.orientation)));
}
