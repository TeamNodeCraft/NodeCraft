use solidcore::kernel::Kernel;

#[test]
fn a_box_passes_the_upstream_geometry_check() {
    let body = solidcore_ops::make_box([0.0, 0.0, 0.0], [10.0, 6.0, 4.0]).expect("box");
    let mut kernel = Kernel::default();
    let brep = nodecraft_layers::body_to_kernel(&mut kernel, &body).expect("kernel brep");
    let report = nodecraft_layers::check_kernel_brep(&mut kernel, &brep).expect("check");
    assert!(report.is_valid(), "{report}");
}
