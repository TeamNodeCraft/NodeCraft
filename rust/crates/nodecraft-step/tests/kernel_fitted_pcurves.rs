//! Kernel-fitted pcurves on a STEP B-spline face: `moon_buggy` face #37 is the first face
//! whose trim has no closed form. The importer's fallback sends the face to the kernel as a
//! one-face sheet; every use must come back with a pcurve that lifts onto the surface within
//! the edge tolerance the importer records.

use nodecraft_step::read_cad;
use solidcore_record::brep::Loop;

const MOON_BUGGY: &str = "tests/Sample AP203 Data Files/moon_buggy.stp";

fn fixture(relative: &str) -> String {
    format!("{}/../../../{relative}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn moon_buggy_face_37_pcurves_lift_onto_the_surface() {
    let path = fixture(MOON_BUGGY);
    let bytes = std::fs::read(&path).expect("read moon_buggy.stp");
    let import = read_cad(&bytes, &path).expect("moon_buggy.stp imports");
    assert_eq!(import.bodies.len(), 1, "{:?}", import.failures);
    assert!(import.failures.is_empty(), "{:?}", import.failures);
    let body = &import.bodies[0];
    assert!(!body.sheet, "moon_buggy imported as a surface body");
    assert_eq!(body.brep.faces.len(), 77);

    let index =
        body.face_origin.iter().position(|&origin| origin == 37).expect("moon_buggy face #37");
    let face = &body.brep.faces[index];
    let surface = solidcore_convert::surface(face.nurbs.as_ref().unwrap_or(&face.surface))
        .expect("face surface");
    let surface = solidcore::eval::SurfaceEval::new(&surface).expect("surface evaluator");
    let mut uses = 0usize;
    for loop_ in &face.loops {
        let Loop::Edges { edges } = loop_ else {
            continue;
        };
        for use_ in edges {
            uses += 1;
            let edge = &body.brep.edges[use_.edge as usize];
            let curve = solidcore_convert::curve(edge.nurbs.as_ref().unwrap_or(&edge.curve))
                .expect("edge curve");
            let curve = solidcore::eval::CurveEval::new(&curve).expect("curve evaluator");
            let [start, end] = edge.interval;
            let mut gap: f64 = 0.0;
            for step in 0..=16 {
                let t = if step == 16 { end } else { start + (end - start) * (step as f64 / 16.0) };
                let uv = solidcore_ops::pcurve::eval(&use_.pcurve, t)
                    .unwrap_or_else(|| panic!("pcurve of edge {} has no point at {t}", use_.edge));
                let lifted =
                    surface.point(uv).unwrap_or_else(|| panic!("surface has no point at {uv:?}"));
                let on_edge =
                    curve.point(t).unwrap_or_else(|| panic!("edge curve has no point at {t}"));
                gap = gap.max(lifted.distance(on_edge));
            }
            assert!(
                gap <= edge.tolerance,
                "face #37 use edge {}: pcurve lifts {gap} mm off its edge, tolerance {} mm",
                use_.edge,
                edge.tolerance
            );
            assert!(
                gap <= 1.0,
                "face #37 use edge {}: pcurve lifts {gap} mm off its edge, past MAX_TOLERANCE",
                use_.edge
            );
        }
    }
    assert_eq!(uses, 4, "face #37 has four edge uses");
}
