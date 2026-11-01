//! Mesh quality: watertight primitives, and the NIST reference solid's volume, area
//! and display-mesh closure.

use std::collections::HashMap;
use std::path::PathBuf;

use nodecraft_mesh::{
    cone_mesh, cylinder_mesh, import_step, pyramid_mesh, sphere_mesh, torus_mesh,
};

fn metrics(mesh: &nodecraft_protocol::Mesh) -> (f64, f64) {
    let mut volume = 0.0f64;
    let mut area = 0.0f64;
    for face in &mesh.faces {
        let [a, b, c] = [face[0] as usize, face[1] as usize, face[2] as usize];
        let (a, b, c) = (mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
        volume += (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64)
            - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64)
            + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64))
            / 6.0;
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            u[1] as f64 * v[2] as f64 - u[2] as f64 * v[1] as f64,
            u[2] as f64 * v[0] as f64 - u[0] as f64 * v[2] as f64,
            u[0] as f64 * v[1] as f64 - u[1] as f64 * v[0] as f64,
        ];
        area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    }
    (volume, area)
}

/// Count undirected edges that do not have exactly one triangle in each direction.
///
/// A watertight, consistently wound mesh has every undirected edge twice: `a -> b` in
/// one triangle and `b -> a` in its neighbour. The returned excess is the number of
/// directed uses left over after cancelling the opposites, so 0 means watertight.
fn unpaired_edges(mesh: &nodecraft_protocol::Mesh) -> usize {
    let mut census: HashMap<(u32, u32), i32> = HashMap::new();
    for face in &mesh.faces {
        for (from, to) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            if from == to {
                continue;
            }
            let (lo, hi) = (from.min(to), from.max(to));
            *census.entry((lo, hi)).or_insert(0) += if from == lo { 1 } else { -1 };
        }
    }
    census
        .values()
        .map(|balance| balance.unsigned_abs() as usize)
        .sum()
}

#[test]
fn primitive_meshes_are_watertight() {
    let meshes = [
        ("cylinder", cylinder_mesh(2.0, 5.0)),
        ("cone", cone_mesh(2.0, 1.0, 5.0)),
        ("sphere", sphere_mesh(3.0)),
        ("torus", torus_mesh(5.0, 1.0)),
        ("pyramid", pyramid_mesh(10.0, 20.0)),
    ];
    for (name, mesh) in meshes {
        let mesh = mesh.unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(!mesh.faces.is_empty(), "{name}: no triangles");
        assert_eq!(unpaired_edges(&mesh), 0, "{name}: open boundary edges");
    }
}

/// The NIST CTC-01 file's independent SMLib reference metrics.
const NIST_CTC_01_VOLUME: f64 = 14_642_612.388_154_887;
const NIST_CTC_01_AREA: f64 = 807_951.914_531_810_2;

fn nist_path() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/step-nist/nist_ctc_01_asme1_rd.stp");
    path.is_file().then_some(path)
}

#[test]
fn nist_ctc_01_volume_and_area_match_smlib_within_one_percent() {
    let Some(path) = nist_path() else {
        eprintln!("skipping nist_ctc_01_volume_and_area_match_smlib_within_one_percent: fixture not present");
        return;
    };
    let import = import_step(path.to_str().expect("UTF-8 path")).expect("NIST import");
    let (volume, area) = metrics(&import.mesh);
    let volume_error = (volume - NIST_CTC_01_VOLUME).abs() / NIST_CTC_01_VOLUME;
    let area_error = (area - NIST_CTC_01_AREA).abs() / NIST_CTC_01_AREA;
    assert!(
        volume_error < 0.01,
        "volume {volume:.1} vs SMLib {NIST_CTC_01_VOLUME:.1} ({:.3}%)",
        volume_error * 100.0
    );
    assert!(
        area_error < 0.01,
        "area {area:.1} vs SMLib {NIST_CTC_01_AREA:.1} ({:.3}%)",
        area_error * 100.0
    );
    eprintln!(
        "NIST ctc_01: volume {volume:.1} ({:.4}%), area {area:.1} ({:.4}%)",
        volume_error * 100.0,
        area_error * 100.0
    );
}

/// The tessellated area must equal the B-rep's own integrated area. This is what the
/// curved trim clipping fixes: dropping partial boundary cells lost about 3.6% of the
/// total (14.5% of the curved area).
#[test]
fn nist_ctc_01_tessellated_area_matches_the_brep_area() {
    let Some(path) = nist_path() else {
        eprintln!("skipping nist_ctc_01_tessellated_area_matches_the_brep_area: fixture not present");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let model = step_import::import(&bytes, &step_import::Limits::default()).expect("official model");
    let body = model.bodies.first().expect("one body");
    let mut brep = step_brep::build(body).expect("B-rep build");
    let kernel = solidcore::kernel::Kernel::default();
    let exact = solidcore::mass::integrator::properties::brep_area(&kernel, &mut brep, 1.0e-3)
        .expect("B-rep area");
    let import = import_step(path.to_str().expect("UTF-8 path")).expect("NIST import");
    let (_, area) = metrics(&import.mesh);
    let error = (area - exact).abs() / exact;
    assert!(
        error < 0.005,
        "tessellated area {area:.1} vs B-rep {exact:.1} ({:.3}%)",
        error * 100.0
    );
    eprintln!("NIST ctc_01 area: tessellated {area:.1} vs exact {exact:.1} ({:.4}%)", error * 100.0);
}

/// Regression guard for display-mesh closure. Every undirected edge should be used
/// twice; the remaining few are cracks at three-face corners where the source B-rep
/// represents one edge by two curves that are up to several millimetres apart. They are
/// reported rather than hidden.
#[test]
fn nist_ctc_01_unpaired_edge_regression() {
    let Some(path) = nist_path() else {
        eprintln!("skipping nist_ctc_01_unpaired_edge_regression: fixture not present");
        return;
    };
    let import = import_step(path.to_str().expect("UTF-8 path")).expect("NIST import");
    let unpaired = unpaired_edges(&import.mesh);
    // Connected components over the face vertices: a welded display mesh of one solid
    // should be a single piece.
    let mut parent: Vec<u32> = (0..import.mesh.vertices.len() as u32).collect();
    fn find(parent: &mut [u32], mut index: u32) -> u32 {
        while parent[index as usize] != index {
            let grand = parent[parent[index as usize] as usize];
            parent[index as usize] = grand;
            index = grand;
        }
        index
    }
    for face in &import.mesh.faces {
        let (a, b, c) = (face[0], face[1], face[2]);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra as usize] = rb;
        }
        let rc = find(&mut parent, c);
        let rb = find(&mut parent, b);
        if rc != rb {
            parent[rc as usize] = rb;
        }
    }
    let roots: std::collections::BTreeSet<u32> = import
        .mesh
        .faces
        .iter()
        .flat_map(|face| face.iter().copied())
        .map(|index| find(&mut parent, index))
        .collect();
    eprintln!(
        "NIST ctc_01: unpaired directed edge uses {unpaired}, connected components {}",
        roots.len()
    );
    assert!(unpaired <= 60, "unpaired directed edges {unpaired} (target 0)");
}
