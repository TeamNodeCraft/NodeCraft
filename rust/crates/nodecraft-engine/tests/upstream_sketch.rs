#[test]
fn upstream_sketch_crate_is_available() {
    let sketch = nodecraft_engine::sketch::Sketch::new();
    assert_eq!(sketch.points.len(), 1);
    assert_eq!(sketch.points[0].id, "origin");
}
