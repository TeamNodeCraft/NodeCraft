fn main(){let p=std::env::args().nth(1).unwrap();nodecraft_step::write_box_with_void(&p,[10.0,10.0,10.0],[4.0,4.0,4.0]).unwrap();println!("{p}");}
