use std::path::PathBuf;

use solidcore_stepio::import_step;


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
    (volume.abs(), area)
}

/// When a NIST file becomes importable, its display mesh must agree with the
/// independent SMLib reference volume/area recorded in `reference.json`.
/// Unsupported files are skipped here and remain covered by `nist_corpus.rs`.
#[test]
fn nist_geometry_matches_smlib_reference_when_supported() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-nist");
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(corpus.join("reference.json")).expect("NIST reference"),
    )
    .expect("valid reference json");
    let files = reference["files"].as_array().expect("reference files");
    assert_eq!(files.len(), 11, "expected 11 NIST reference entries");
    let mut compared = 0;
    for entry in files {
        let name = entry["file"].as_str().expect("file name");
        let expected_volume = entry["volume"].as_f64().expect("reference volume");
        let expected_area = entry["area"].as_f64().expect("reference area");
        assert!(expected_volume > 0.0 && expected_area > 0.0, "{name}: invalid reference");
        let path = corpus.join(name);
        let Ok(import) = import_step(path.to_str().expect("UTF-8 path")) else {
            continue;
        };
        let (volume, area) = metrics(&import.mesh);
        let volume_error = (volume - expected_volume).abs() / expected_volume;
        let area_error = (area - expected_area).abs() / expected_area;
        assert!(volume_error < 0.02, "{name}: volume {volume} vs {expected_volume} ({volume_error:.3})");
        assert!(area_error < 0.02, "{name}: area {area} vs {expected_area} ({area_error:.3})");
        compared += 1;
    }
    eprintln!("NIST geometry compared: {compared}/11 supported");
}
