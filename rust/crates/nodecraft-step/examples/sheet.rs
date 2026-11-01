fn main() {
    let path = std::env::args().nth(1).unwrap();
    let body = nodecraft_step::PlanarBody {
        points: vec![[0.,0.,0.],[4.,0.,0.],[4.,4.,0.],[0.,4.,0.]],
        faces: vec![vec![0,1,2,3]],
        holes: vec![vec![]],
    };
    nodecraft_step::write_open_shell(&path, &body).unwrap();
    println!("{path}");
}
