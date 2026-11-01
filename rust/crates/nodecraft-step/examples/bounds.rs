fn main() {
    let path = std::env::args().nth(1).unwrap();
    println!("{:?}", nodecraft_step::read_step_bounds(&path));
    println!("{:?}", nodecraft_step::detect_analytic_solid(&path));
}
