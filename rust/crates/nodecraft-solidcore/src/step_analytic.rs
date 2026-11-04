//! Exact analytic STEP primitives → solidcore Breps.

use solidcore::kernel::Kernel;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives as api;
use solidcore::topology::Brep;

/// Builds the solidcore Brep described by an analytic primitive STEP file.
///
/// The detected primitive is dimension-validated before any solidcore constructor runs,
/// so a misdetected or malformed file fails with a precise category instead of a
/// downstream `create_*` error.
pub fn brep_from_analytic_step(path: &str) -> Result<Brep, String> {
    let solid = nodecraft_step::detect_analytic_solid(path)?
        .ok_or_else(|| "STEP file is not a single analytic primitive".to_string())?;
    validate_primitive(&solid)?;
    let mut kernel = crate::kernel::kernel();
    match solid {
        nodecraft_step::AnalyticSolid::Cylinder { radius, height, center } => {
            api::create_cylinder(&mut kernel, Point3::new(center[0], center[1], center[2] - height / 2.0), radius, height).map_err(|e| format!("solidcore cylinder: {e}"))
        }
        nodecraft_step::AnalyticSolid::Cone { radius_base, radius_top, height, center } => {
            api::create_cone(&mut kernel, Point3::new(center[0], center[1], center[2] - height / 2.0), radius_base, radius_top, height).map_err(|e| format!("solidcore cone: {e}"))
        }
        nodecraft_step::AnalyticSolid::Sphere { radius, center } => {
            api::create_sphere(&mut kernel, Point3::new(center[0], center[1], center[2]), radius).map_err(|e| format!("solidcore sphere: {e}"))
        }
        nodecraft_step::AnalyticSolid::Torus { major, minor, center } => {
            api::create_torus(&mut kernel, Point3::new(center[0], center[1], center[2]), major, minor).map_err(|e| format!("solidcore torus: {e}"))
        }
    }
}

/// Rejects detected primitives whose dimensions cannot describe a solid.
fn validate_primitive(solid: &nodecraft_step::AnalyticSolid) -> Result<(), String> {
    use nodecraft_step::AnalyticSolid;
    let positive = |field: &str, value: f64| -> Result<(), String> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(format!("invalid analytic primitive: {field} must be finite and positive, got {value}"))
        }
    };
    let non_negative = |field: &str, value: f64| -> Result<(), String> {
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(format!("invalid analytic primitive: {field} must be finite and non-negative, got {value}"))
        }
    };
    let center = |point: [f64; 3]| -> Result<(), String> {
        if point.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(format!("invalid analytic primitive: center must be finite, got {point:?}"))
        }
    };
    match solid {
        AnalyticSolid::Cylinder { radius, height, center: point } => {
            positive("cylinder radius", *radius)?;
            positive("cylinder height", *height)?;
            center(*point)
        }
        AnalyticSolid::Cone { radius_base, radius_top, height, center: point } => {
            non_negative("cone base radius", *radius_base)?;
            non_negative("cone top radius", *radius_top)?;
            if *radius_base <= 0.0 && *radius_top <= 0.0 {
                return Err("invalid analytic primitive: cone base and top radii are both zero".into());
            }
            positive("cone height", *height)?;
            center(*point)
        }
        AnalyticSolid::Sphere { radius, center: point } => {
            positive("sphere radius", *radius)?;
            center(*point)
        }
        AnalyticSolid::Torus { major, minor, center: point } => {
            positive("torus major radius", *major)?;
            positive("torus minor radius", *minor)?;
            center(*point)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcore::mass::integrator::properties::{brep_area, brep_volume, PropertyAttributes};
    use solidcore::geom::attribute::AttributeStore;

    #[test]
    fn cylinder_step_becomes_exact_brep() {
        let path = std::env::temp_dir().join(format!("sn-analytic-{}.step", std::process::id()));
        nodecraft_step::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).unwrap();
        let mut brep = brep_from_analytic_step(path.to_str().unwrap()).unwrap();
        let kernel = crate::kernel::kernel();
        let store = AttributeStore::new();
        let attrs = PropertyAttributes { store: &store, key_of: &|entity| solidcore::geom::attribute::ObjectKey(1_000_000 + entity_type(entity)) };
        let volume = brep_volume(&kernel, &mut brep, 1e-7, &attrs).unwrap().abs();
        let area = brep_area(&kernel, &mut brep, 1e-7).unwrap();
        assert!((volume - std::f64::consts::PI * 25.0 * 10.0).abs() < 1e-3, "volume {volume}");
        assert!((area - (2.0 * std::f64::consts::PI * 5.0 * 10.0 + 2.0 * std::f64::consts::PI * 25.0)).abs() < 2e-2, "area {area}");
    }

    fn entity_type(entity: solidcore::error::EntityRef) -> u64 {
        use solidcore::error::EntityRef::*;
        match entity { Face(_) => 1, Edge(_) => 2, Edgeuse(_) => 3, Vertex(_) => 4, Vertexuse(_) => 5, Loop(_) => 6, Loopuse(_) => 7, Faceuse(_) => 8, Shell(_) => 9, Region(_) => 10, Brep(_) => 11 }
    }


    #[test]
    fn rejects_multi_solid_lookalike_file_cleanly() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/Sample AP203 Data Files/4pinplug.stp");
        if !path.exists() {
            return;
        }
        let error = brep_from_analytic_step(path.to_str().unwrap())
            .expect_err("4pinplug is a multi-solid assembly, not one analytic primitive");
        assert!(
            error.contains("not a single analytic primitive"),
            "expected the precise non-primitive category, got: {error}"
        );
        assert!(!error.contains("create_cylinder"), "file-level detection reached a constructor: {error}");
        assert!(!error.contains("invalid cone dimensions"), "file-level detection reached a constructor: {error}");
    }

    #[test]
    fn rejects_invalid_detected_primitive_dimensions() {
        use nodecraft_step::AnalyticSolid;
        let origin = [0.0, 0.0, 0.0];
        let invalid = [
            AnalyticSolid::Cylinder { radius: 0.0, height: 1.0, center: origin },
            AnalyticSolid::Cylinder { radius: 1.0, height: -1.0, center: origin },
            AnalyticSolid::Cylinder { radius: f64::NAN, height: 1.0, center: origin },
            AnalyticSolid::Cylinder { radius: 1.0, height: 1.0, center: [f64::INFINITY, 0.0, 0.0] },
            AnalyticSolid::Cone { radius_base: 0.0, radius_top: 0.0, height: 1.0, center: origin },
            AnalyticSolid::Cone { radius_base: -1.0, radius_top: 1.0, height: 1.0, center: origin },
            AnalyticSolid::Cone { radius_base: 1.0, radius_top: 1.0, height: 0.0, center: origin },
            AnalyticSolid::Sphere { radius: 0.0, center: origin },
            AnalyticSolid::Sphere { radius: 1.0, center: [0.0, f64::NAN, 0.0] },
            AnalyticSolid::Torus { major: -1.0, minor: 1.0, center: origin },
            AnalyticSolid::Torus { major: 8.0, minor: 0.0, center: origin },
        ];
        for solid in &invalid {
            let error = validate_primitive(solid).expect_err("invalid primitive dimensions must be rejected");
            assert!(error.contains("invalid analytic primitive"), "{solid:?}: {error}");
        }
        for solid in [
            AnalyticSolid::Cylinder { radius: 5.0, height: 10.0, center: origin },
            AnalyticSolid::Cone { radius_base: 2.0, radius_top: 5.0, height: 10.0, center: origin },
            AnalyticSolid::Sphere { radius: 5.0, center: origin },
            AnalyticSolid::Torus { major: 8.0, minor: 2.0, center: origin },
        ] {
            validate_primitive(&solid).expect("valid primitive dimensions");
        }
    }
}
