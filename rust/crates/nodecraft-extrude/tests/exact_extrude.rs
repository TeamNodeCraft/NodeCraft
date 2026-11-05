//! Integration tests for the exact `create_taper_extrude` route.

use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::kernel::Kernel;
use solidcore::mass::integrator::properties::{
    PropertyAttributes, brep_area, brep_volume,
};
use solidcore::math::{Point3, Vec3};
use solidcore::primitive::api_primitives as api;
use solidcore::query::api_queries::{brep_is_manifold_solid, face_get_boundary_loops};
use solidcore::sweep::api_brep::create_taper_extrude;
use solidcore::{Brep, FaceId};
use nodecraft_extrude::{
    ExtrudeError, brep_is_connected_solid, extrude_planar_face, extrude_planar_face_exact,
    extrude_profile_curves, face_boundary, face_plane,
};

fn measure(k: &Kernel, brep: &Brep) -> (f64, f64) {
    let store = AttributeStore::default();
    let attributes = PropertyAttributes {
        store: &store,
        key_of: &|_| ObjectKey(0),
    };
    let mut copy = brep.clone_standing_for_self().expect("standing copy");
    let area = brep_area(k, &mut copy, 1e-4).expect("area");
    let mut copy = brep.clone_standing_for_self().expect("standing copy");
    let volume = brep_volume(k, &mut copy, 1e-4, &attributes).expect("volume");
    (area, volume)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} != {expected} (tolerance {tolerance})"
    );
}

fn box_brep(kernel: &mut Kernel) -> Brep {
    api::create_box(kernel, Point3::new(0.0, 0.0, 0.0), 10.0, 4.0, 20.0).expect("box")
}

fn face_with_normal(k: &Kernel, brep: &Brep, normal: Vec3) -> FaceId {
    for (face, _) in brep.faces().iter() {
        if let Ok(plane) = face_plane(k, brep, face) {
            if plane.normal.dot(normal) > 0.999 {
                return face;
            }
        }
    }
    panic!("no face with normal {normal:?}");
}

fn bounds(k: &Kernel, brep: &Brep) -> ([f64; 3], [f64; 3]) {
    let extent = solidcore::query::brep::calculate_bounding_box(
        k,
        brep,
        solidcore::query::brep::BoundingBoxOptions {
            approx: false,
            tight: true,
            expand_by_zone_tol_3d: false,
        },
    )
    .expect("bounds");
    (
        [extent.min_at(0), extent.min_at(1), extent.min_at(2)],
        [extent.max_at(0), extent.max_at(1), extent.max_at(2)],
    )
}

/// A closed exact solid has no face-arena records outside the face list, every face is
/// used by a face use and a single boundary loop, and every loop edge is shared by
/// exactly two faces.
fn assert_closed_solid(brep: &Brep, faces: usize, edges: usize, vertices: usize) {
    assert_eq!(brep.face_list().len(), faces, "listed faces");
    assert_eq!(
        brep.faces().len(),
        faces,
        "no unlisted face records may remain"
    );
    assert_eq!(brep.edge_list().len(), edges, "edges");
    assert_eq!(brep.vertex_list().len(), vertices, "vertices");
    assert!(brep_is_manifold_solid(brep).expect("manifold query"));
    assert!(brep_is_connected_solid(brep), "connected solid diagnostic");

    let mut loop_edges = 0usize;
    let mut unique = std::collections::BTreeSet::new();
    for &face in brep.face_list() {
        let record = brep.face(face).expect("face");
        assert!(record.faceuse.is_some(), "face {face:?} has no face use");
        assert_eq!(record.loops.len(), 1, "face {face:?} has one outer loop");
        let loops = face_get_boundary_loops(brep, face).expect("face boundary loops");
        assert_eq!(loops.edge_counts.len(), 1);
        assert!(!loops.edges.is_empty());
        for &edge in &loops.edges {
            unique.insert(edge);
            loop_edges += 1;
        }
    }
    assert_eq!(unique.len(), edges, "all edges appear in face loops");
    assert_eq!(loop_edges, 2 * edges, "every edge used by exactly two faces");
}

#[test]
fn exact_extrude_box_top_face_is_a_closed_solid() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    let solid = extrude_planar_face_exact(&mut k, &brep, top, 5.0).expect("exact extrude");
    assert_closed_solid(&solid, 6, 12, 8);
    let (area, volume) = measure(&k, &solid);
    assert_close(area, 220.0, 1.0e-9);
    assert_close(volume, 200.0, 1.0e-9);
    let (min, max) = bounds(&k, &solid);
    assert_close(min[0], 0.0, 1.0e-9);
    assert_close(min[1], 0.0, 1.0e-9);
    assert_close(min[2], 20.0, 1.0e-9);
    assert_close(max[0], 10.0, 1.0e-9);
    assert_close(max[1], 4.0, 1.0e-9);
    assert_close(max[2], 25.0, 1.0e-9);
}

#[test]
fn exact_extrude_box_side_face_uses_outward_normal() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let side = face_with_normal(&k, &brep, Vec3::new(1.0, 0.0, 0.0));

    let solid = extrude_planar_face_exact(&mut k, &brep, side, 2.0).expect("exact extrude");
    assert_closed_solid(&solid, 6, 12, 8);
    let (area, volume) = measure(&k, &solid);
    assert_close(area, 2.0 * 80.0 + 48.0 * 2.0, 1.0e-9);
    assert_close(volume, 160.0, 1.0e-9);
    let (min, max) = bounds(&k, &solid);
    assert_close(min[0], 10.0, 1.0e-9);
    assert_close(max[0], 12.0, 1.0e-9);
    assert_close(min[2], 0.0, 1.0e-9);
    assert_close(max[2], 20.0, 1.0e-9);
}

#[test]
fn exact_extrude_distances_have_exact_area_and_volume() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    for distance in [1.0, 5.0, 12.5] {
        let solid = extrude_planar_face_exact(&mut k, &brep, top, distance).expect("extrude");
        assert_closed_solid(&solid, 6, 12, 8);
        let (area, volume) = measure(&k, &solid);
        // Top face 10x4, perimeter 28.
        assert_close(area, 2.0 * 40.0 + 28.0 * distance, 1.0e-9);
        assert_close(volume, 40.0 * distance, 1.0e-9);
    }
}

#[test]
fn exact_extrude_matches_the_box_primitive_measures() {
    let mut k = Kernel::default();
    let source = box_brep(&mut k);
    let top = face_with_normal(&k, &source, Vec3::new(0.0, 0.0, 1.0));

    // Extruding the top face by the box height reproduces the box exactly.
    let extruded = extrude_planar_face_exact(&mut k, &source, top, 20.0).expect("extrude");
    let primitive = api::create_box(&mut k, Point3::new(0.0, 0.0, 20.0), 10.0, 4.0, 20.0)
        .expect("box");
    assert_closed_solid(&extruded, 6, 12, 8);
    let (extruded_area, extruded_volume) = measure(&k, &extruded);
    let (primitive_area, primitive_volume) = measure(&k, &primitive);
    assert_close(extruded_area, primitive_area, 1.0e-9);
    assert_close(extruded_volume, primitive_volume, 1.0e-9);
    assert_close(primitive_volume, 800.0, 1.0e-9);
}

#[test]
fn exact_extrude_triangle_profile_is_closed_and_exact() {
    let mut k = Kernel::default();
    let pyramid = api::create_pyramid(&mut k, Point3::new(0.0, 0.0, 0.0), 8.0, 6.0)
        .expect("pyramid");
    let mut triangle = None;
    for (face, _) in pyramid.faces().iter() {
        if let Ok(loops) = face_get_boundary_loops(&pyramid, face) {
            if loops.edge_counts == [3] {
                triangle = Some(face);
            }
        }
    }
    let triangle = triangle.expect("triangular side face");

    let profile = nodecraft_extrude::planar_face_profile(&k, &pyramid, triangle)
        .expect("triangle profile");
    let points = &profile.loops[0].points;
    let triangle_area = (points[1] - points[0])
        .cross(points[2] - points[0])
        .length()
        * 0.5;
    let perimeter = (points[1] - points[0]).length()
        + (points[2] - points[1]).length()
        + (points[0] - points[2]).length();

    let solid = extrude_planar_face_exact(&mut k, &pyramid, triangle, 2.0).expect("extrude");
    assert_closed_solid(&solid, 5, 9, 6);
    let (area, volume) = measure(&k, &solid);
    assert_close(area, 2.0 * triangle_area + perimeter * 2.0, 1.0e-6);
    assert_close(volume, triangle_area * 2.0, 1.0e-6);
}

#[test]
fn exact_extrude_curve_route_matches_the_face_route() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    let loops = face_boundary(&k, &brep, top).expect("boundary");
    let curves = loops[0].oriented_curves(&k).expect("oriented curves");
    let from_curves = extrude_profile_curves(&mut k, &curves, 5.0).expect("curve profile extrude");
    let from_face = extrude_planar_face_exact(&mut k, &brep, top, 5.0).expect("face extrude");

    assert_closed_solid(&from_curves, 6, 12, 8);
    assert_eq!(
        measure(&k, &from_curves),
        measure(&k, &from_face),
        "same profile, same exact measures"
    );
}

/// A single closed periodic profile now extrudes through solidcore's merge path.
#[test]
fn periodic_profile_extrudes_successfully() {
    let mut k = Kernel::default();
    let cylinder = api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0)
        .expect("cylinder");
    let cap = face_with_normal(&k, &cylinder, Vec3::new(0.0, 0.0, 1.0));
    let loops = face_boundary(&k, &cylinder, cap).expect("cap boundary");
    assert_eq!(loops[0].segments.len(), 1, "a full circle is one closed curve");
    let curves = loops[0].oriented_curves(&k).expect("cap curve");
    let refs: Vec<Option<&solidcore::curve::curve::Curve>> = curves.iter().map(Some).collect();

    let result = create_taper_extrude(&mut k, &refs, 3.0, 0.0, 3, false)
        .expect("periodic-profile extrude");
    assert!(!result.face_list().is_empty(), "no faces");
    assert!(
        solidcore::query::api_queries::brep_is_manifold_solid(&result).expect("manifold"),
        "periodic-profile extrude is not a manifold solid"
    );
    let (area, volume) = measure(&k, &result);
    assert!(area > 0.0 && volume > 0.0, "area {area}, volume {volume}");
}

#[test]
fn exact_extrude_circular_profile_uses_the_split_fallback() {
    let mut k = Kernel::default();
    let cylinder = api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0)
        .expect("cylinder");
    let cap = face_with_normal(&k, &cylinder, Vec3::new(0.0, 0.0, 1.0));

    // The wrapper splits the single closed circle into two half-arcs and succeeds.
    let solid = extrude_planar_face_exact(&mut k, &cylinder, cap, 3.0).expect("exact extrude");
    assert!(brep_is_connected_solid(&solid));
    assert!(brep_is_manifold_solid(&solid).expect("manifold"));

    let store = AttributeStore::default();
    let attributes = PropertyAttributes {
        store: &store,
        key_of: &|_| ObjectKey(0),
    };
    let mut volume_copy = solid.clone_standing_for_self().expect("standing copy"); let volume = brep_volume(&k, &mut volume_copy, 1e-6, &attributes).expect("volume");
    assert_close(volume, std::f64::consts::PI * 4.0 * 3.0, 1.0e-4);
    // `brep_area` clamps its integration accuracy to 1e-4; the seam split keeps the
    // area within that accuracy of the analytic cylinder.
    let mut area_copy = solid.clone_standing_for_self().expect("standing copy"); let area = brep_area(&k, &mut area_copy, 1e-4).expect("area");
    assert_close(
        area,
        std::f64::consts::TAU * 4.0 + std::f64::consts::TAU * 2.0 * 3.0,
        1.0e-2,
    );
}

#[test]
fn exact_extrude_rejects_invalid_input() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    for distance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = extrude_planar_face_exact(&mut k, &brep, top, distance)
            .expect_err("reject invalid distance");
        assert!(matches!(error, ExtrudeError::InvalidDistance(_)), "got {error}");
    }
    assert!(matches!(
        extrude_profile_curves(&mut k, &[], 1.0),
        Err(ExtrudeError::EmptyProfile)
    ));

    let cylinder = api::create_cylinder(&mut k, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0)
        .expect("cylinder");
    let mut lateral = None;
    for (face, _) in cylinder.faces().iter() {
        if matches!(
            face_plane(&k, &cylinder, face),
            Err(ExtrudeError::NotPlanar { .. })
        ) {
            lateral = Some(face);
        }
    }
    let lateral = lateral.expect("cylinder side face");
    assert!(matches!(
        extrude_planar_face_exact(&mut k, &cylinder, lateral, 1.0),
        Err(ExtrudeError::NotPlanar { .. })
    ));
}

#[test]
fn extrude_planar_face_prefers_the_exact_route() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    let solid = extrude_planar_face(&mut k, &brep, top, 5.0).expect("extrude");
    // The old fallback returned a B-rep with unlisted cap-face records; the exact route
    // returns a clean closed solid.
    assert!(brep_is_connected_solid(&solid));
    assert_closed_solid(&solid, 6, 12, 8);
    let (area, volume) = measure(&k, &solid);
    assert_close(area, 220.0, 1.0e-9);
    assert_close(volume, 200.0, 1.0e-9);
}

/// The legacy curve-sweep route is still exposed for comparison. Solidcore now
/// closes its B-rep, so it should be a connected solid as well as geometrically exact.
#[test]
fn legacy_linear_sweep_measures_exactly_and_is_connected() {
    let mut k = Kernel::default();
    let brep = box_brep(&mut k);
    let top = face_with_normal(&k, &brep, Vec3::new(0.0, 0.0, 1.0));

    let legacy = nodecraft_extrude::extrude_planar_face_linear_sweep(&mut k, &brep, top, 5.0)
        .expect("legacy linear sweep");
    let (area, volume) = measure(&k, &legacy);
    assert_close(area, 220.0, 1.0e-9);
    assert_close(volume, 200.0, 1.0e-9);
    assert!(brep_is_connected_solid(&legacy));
    assert_eq!(legacy.faces().len(), legacy.face_list().len());
}
