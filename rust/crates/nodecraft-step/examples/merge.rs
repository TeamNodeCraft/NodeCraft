fn main() {
    let dir = std::env::temp_dir().join("sn-merge");
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.step"); let b = dir.join("b.step"); let out = dir.join("merged.step");
    nodecraft_step::write_box(a.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
    nodecraft_step::write_cylinder(b.to_str().unwrap(), 5.0, 10.0).unwrap();
    nodecraft_step::merge_step_files(&[a.to_string_lossy().into(), b.to_string_lossy().into()], out.to_str().unwrap()).unwrap();
    println!("{}", out.display());
}
