use std::time::Instant;
fn ms<T>(label: &str, f: impl FnOnce() -> T) -> T { let start=Instant::now(); let out=f(); eprintln!("{label}: {:.1} ms", start.elapsed().as_secs_f64()*1000.0); out }
fn main(){let path=std::env::args().nth(1).unwrap();
 let _ = ms("import_step", || solidcore_stepio::import_step(&path).map(|i| (i.mesh.faces.len(), i.diagnostics.kind)));
 ms("classify_step", || nodecraft_step::classify_step(&path).ok());
 ms("classify_solids", || nodecraft_step::classify_solids(&path).ok());
 ms("read_topology_file", || nodecraft_step::read_topology_file(&path).ok());
 ms("read_header", || nodecraft_step::read_header(&path).ok());
 ms("read_solid_names", || nodecraft_step::read_solid_names(&path).ok());
 ms("read_unit_context", || nodecraft_step::read_unit_context(&path).ok());
 ms("read_tolerance", || nodecraft_step::read_tolerance(&path).ok());
}
