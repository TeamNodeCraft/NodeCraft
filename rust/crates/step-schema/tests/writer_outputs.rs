use step_schema::validate_file;

#[test]
fn primitive_writer_outputs_are_schema_valid() {
    let dir = std::env::temp_dir().join(format!("sn-schema-out-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let writers: [(&str, fn(&str) -> Result<(), String>); 6] = [
        ("box", |path| nodecraft_step::write_box(path, 10.0, 4.0, 20.0)),
        ("cylinder", |path| nodecraft_step::write_cylinder(path, 5.0, 10.0)),
        ("cone", |path| nodecraft_step::write_cone(path, 5.0, 2.0, 10.0)),
        ("sphere", |path| nodecraft_step::write_sphere(path, 5.0)),
        ("torus", |path| nodecraft_step::write_torus(path, 8.0, 2.0)),
        ("wedge", |path| nodecraft_step::write_wedge(path, 10.0, 4.0, 6.0, 2.0)),
    ];
    for (name, write) in writers {
        let path = dir.join(format!("{name}.step"));
        write(path.to_str().unwrap()).unwrap();
        let report = validate_file(path.to_str().unwrap());
        assert!(report.is_valid(), "{name}: {:?}", report.errors().map(ToString::to_string).collect::<Vec<_>>());
    }
}
