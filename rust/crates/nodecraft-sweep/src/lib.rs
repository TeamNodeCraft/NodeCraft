//! Sketch profile segments as the curves `solidcore-ops` sweeps and lofts take.
//!
//! The generic part only uses upstream types and is a candidate for
//! `solidcore-record`/`solidcore-ops`: [`curve_from_seg2`]. NodeCraft-specific
//! sketch adapters stay outside this crate.

use solidcore_ops::sweep::Curve;
use solidcore_record::profile::Seg2;
use solidcore_record::Plane;

/// One planar profile segment as a sweep curve, through the upstream converter.
///
/// Lines and arcs stay analytic; cubic, conic and Bézier segments become NURBS
/// curves. Unsupported or degenerate segments return `None`.
#[must_use]
pub fn curve_from_seg2(segment: &Seg2, plane: &Plane) -> Option<Curve> {
    Curve::from_seg2(segment, plane)
}

/// A closed profile on `plane`, in order.
#[must_use]
pub fn section_from_seg2(segments: &[Seg2], plane: &Plane) -> Option<Vec<Curve>> {
    segments
        .iter()
        .map(|segment| curve_from_seg2(segment, plane))
        .collect()
}

/// Loft closed profile sections placed on their own record planes.
///
/// # Errors
///
/// [`solidcore_ops::Error::Argument`] when a segment has no sweep curve form,
/// or the loft's own refusals.
pub fn loft_on_planes(
    sections: &[(Vec<Seg2>, Plane)],
    degree: u32,
    cap: bool,
) -> Result<solidcore_ops::Body, solidcore_ops::Error> {
    let sections: Option<Vec<Vec<Curve>>> = sections
        .iter()
        .map(|(segments, plane)| section_from_seg2(segments, plane))
        .collect();
    let sections = sections.ok_or(solidcore_ops::Error::Argument)?;
    solidcore_ops::sweep::loft(&sections, degree, cap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore_record::{Vec2, Vec3};

    fn square(size: f64) -> Vec<Seg2> {
        let h = size * 0.5;
        let p = [Vec2::new(-h, -h), Vec2::new(h, -h), Vec2::new(h, h), Vec2::new(-h, h)];
        (0..4).map(|i| Seg2::Line { a: p[i], b: p[(i + 1) % 4] }).collect()
    }

    #[test]
    fn line_and_arc_map_into_the_plane() {
        let line = curve_from_seg2(
            &Seg2::Line { a: Vec2::new(0.0, 0.0), b: Vec2::new(2.0, 0.0) },
            &Plane::XY,
        )
        .expect("line");
        assert!(matches!(line, Curve::Line { start: [0.0, 0.0, 0.0], end: [2.0, 0.0, 0.0] }));

        let arc = curve_from_seg2(
            &Seg2::Arc { center: Vec2::new(0.0, 0.0), radius: 1.0, start: 0.0, sweep: std::f64::consts::FRAC_PI_2 },
            &Plane::XY,
        )
        .expect("arc");
        let Curve::Arc { axis, start, end, .. } = arc else { panic!("arc variant") };
        assert_eq!(axis, [0.0, 0.0, 1.0]);
        assert!((end - start - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }

    #[test]
    fn curved_segments_become_nurbs() {
        let cubic = curve_from_seg2(
            &Seg2::Cubic {
                p0: Vec2::new(0.0, 0.0),
                p1: Vec2::new(1.0, 0.0),
                p2: Vec2::new(1.0, 1.0),
                p3: Vec2::new(0.0, 1.0),
            },
            &Plane::XY,
        )
        .expect("cubic");
        let Curve::Nurbs { degree, points, .. } = cubic else { panic!("cubic NURBS") };
        assert_eq!(degree, 3);
        assert_eq!(points[3], [0.0, 1.0, 0.0, 1.0]);

        let conic = curve_from_seg2(
            &Seg2::Conic {
                a: Vec2::new(0.0, 0.0),
                apex: Vec2::new(1.0, 1.0),
                b: Vec2::new(2.0, 0.0),
                w: 0.5,
            },
            &Plane::XY,
        )
        .expect("conic");
        let Curve::Nurbs { points, .. } = conic else { panic!("conic NURBS") };
        assert_eq!(points[1], [1.0, 1.0, 0.0, 0.5]);
    }

    #[test]
    fn two_sections_loft_to_a_closed_box() {
        let upper_plane = Plane { origin: Vec3::new(0.0, 0.0, 6.0), ..Plane::XY };
        let body = loft_on_planes(&[(square(4.0), Plane::XY), (square(4.0), upper_plane)], 1, true)
            .expect("loft");
        assert_eq!(body.counts().expect("counts"), [6, 12, 8]);
        assert!((body.volume(1.0e-3).expect("volume") - 96.0).abs() < 1.0e-6);
    }

    #[test]
    fn loft_between_two_square_sections_has_the_expected_volume() {
        let square = square(10.0);
        let bottom = Plane::XY;
        let top = Plane {
            origin: Vec3::new(0.0, 0.0, 10.0),
            ..Plane::XY
        };
        let body = loft_on_planes(&[(square.clone(), bottom), (square, top)], 1, true)
            .expect("ops loft");
        let volume = body.volume(1.0e-7).expect("volume");
        assert!((volume - 1000.0).abs() < 1.0e-6, "{volume}");
    }
}
