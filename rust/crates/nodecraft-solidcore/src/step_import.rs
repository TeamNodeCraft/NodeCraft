//! General STEP → solidcore B-rep plug-in.
//!
//! This is the single entry point that plugs the new solidcore STEP readers into the
//! NodeCraft import pipeline. The readers each own one class of STEP geometry:
//!
//! | reader | geometry it builds |
//! |---|---|
//! | [`crate::step_nurbs_faces`] | one B-spline sheet face with a UV-rectangular trim |
//! | [`crate::step_analytic`] | cylinder / cone / sphere / torus primitives |
//!
//! Mixed planar/cylindrical solids (doghouse-style) are tried first through
//! [`crate::step_mixed`]. That reader currently rejects them explicitly: the prototype
//! builds the exact 32-face topology but solidcore's analytic-plane area/volume
//! integration is not exact yet, so returning a B-rep would silently report wrong
//! metrics. The dispatcher surfaces that reason instead of falling back to a planar shell.
//!
//! [`brep_from_step`] parses the STEP file once per candidate reader and returns the
//! first exact B-rep, tagging which reader produced it. When no reader can handle the
//! file it returns a single error that lists every reader's reason, so the caller (the
//! Rust worker's `stepimporter` node) can report a precise unsupported-feature message.
//!
//! # Scope / limits
//!
//! * **One B-rep per file.** Assemblies and multi-solid files are not compounded into
//!   one Brep yet; the readers reject them explicitly.
//! * **Mixed planar/cylindrical solids are rejected explicitly** by the
//!   [`crate::step_mixed`] stub until solidcore's analytic-plane area/volume integration
//!   and `face_internal_point` are exact for those B-reps; no inexact B-rep is returned.
//! * Readers reject geometry they cannot represent *exactly* (unknown surface kinds,
//!   non-rectangular sheet trims, voids, ...) instead of guessing.
//! * Coordinates are millimetres where the reader handles units (`step_mixed`,
//!   `step_analytic`); see each reader for its own unit rules.

use solidcore::kernel::Kernel;
use solidcore::topology::Brep;
/// Which solidcore STEP reader produced a [`StepBrepImport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepBrepKind {
    /// Mixed planar/cylindrical solid reader (temporarily absent from this build; kept
    /// for API stability while the reader is re-landed).
    MixedPlanarCylindrical,
    /// `step_nurbs_faces`: one B-spline sheet face with a UV-rectangular trim.
    BsplineSheet,
    /// `step_analytic`: cylinder / cone / sphere / torus primitive.
    AnalyticPrimitive,
    /// Exact planar solid reader (faces + UV trim curves).
    PlanarSolid,
}

impl StepBrepKind {
    /// Stable short name used in diagnostics (`kind` field of the worker response).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MixedPlanarCylindrical => "mixed-planar-cylindrical",
            Self::BsplineSheet => "bspline-sheet",
            Self::AnalyticPrimitive => "analytic-primitive",
            Self::PlanarSolid => "planar-solid",
        }
    }
}

impl std::fmt::Display for StepBrepKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An exact solidcore B-rep imported from STEP, plus the reader that built it.
#[derive(Debug)]
pub struct StepBrepImport {
    /// The imported B-rep (millimetres where the reader handles units).
    pub brep: Brep,
    /// Which reader produced [`Self::brep`].
    pub kind: StepBrepKind,
}

/// Import a STEP file with the first solidcore reader that can represent it exactly.
///
/// # Errors
///
/// * the file cannot be read or parsed,
/// * no reader supports the file's geometry; the error lists every reader's reason
///   (`no solidcore STEP importer can read ...`).
pub fn brep_from_step(path: &str) -> Result<StepBrepImport, String> {
    let mut attempts: Vec<String> = Vec::new();

    match nodecraft_step::read_planar_solids(path) {
        Ok(bodies) if !bodies.is_empty() => {
            match crate::mesh_brep::brep_from_planar_body_exact(&bodies[0]) {
                Ok(brep) => return Ok(StepBrepImport { brep, kind: StepBrepKind::PlanarSolid }),
                Err(error) => attempts.push(format!("planar solid: {error}")),
            }
        }
        Ok(_) => attempts.push("planar solid: no solids".into()),
        Err(error) => attempts.push(format!("planar solid: {error}")),
    }
    match crate::step_nurbs_faces::brep_from_bspline_sheet_file(path) {
        Ok(brep) => return Ok(StepBrepImport { brep, kind: StepBrepKind::BsplineSheet }),
        Err(error) => attempts.push(format!("B-spline sheet: {error}")),
    }
    match crate::step_analytic::brep_from_analytic_step(path) {
        Ok(brep) => return Ok(StepBrepImport { brep, kind: StepBrepKind::AnalyticPrimitive }),
        Err(error) => attempts.push(format!("analytic primitive: {error}")),
    }
    match crate::step_mixed::brep_from_mixed_step(path) {
        Ok(brep) => return Ok(StepBrepImport { brep, kind: StepBrepKind::MixedPlanarCylindrical }),
        Err(error) => attempts.push(format!("mixed planar/cylindrical: {error}")),
    }

    Err(format!(
        "no solidcore STEP importer can read {path} (tried: {})",
        attempts.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name: &str) -> Option<String> {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for directory in ["Sample AP203 Data Files"] {
            let path = manifest.join("../../../tests").join(directory).join(name);
            if path.exists() {
                return Some(path.to_string_lossy().into_owned());
            }
        }
        None
    }

    /// Mixed planar/cylindrical solids are imported exactly by the validated mixed
    /// reader; the dispatcher must report the mixed category and the exact B-rep.
    #[test]
    fn mixed_planar_cylindrical_solid_imports_exactly() {
        let Some(path) = fixture("doghouse.stp") else {
            eprintln!("skipping doghouse.stp: fixture not present");
            return;
        };
        let import = brep_from_step(&path).expect("mixed reader succeeds");
        assert_eq!(import.kind, StepBrepKind::MixedPlanarCylindrical);
        assert_eq!(import.brep.faces().len(), 32);
        assert_eq!(import.brep.edges().len(), 87);
        assert_eq!(import.brep.vertices().len(), 58);
        use solidcore::geom::attribute::{AttributeStore, ObjectKey};
        use solidcore::kernel::Kernel;
        use solidcore::mass::integrator::properties::{PropertyAttributes, brep_mass_properties};
        let kernel = crate::kernel::kernel();
        let store = AttributeStore::new();
        let key = |_| ObjectKey(2_000_000);
        let attributes = PropertyAttributes { store: &store, key_of: &key };
        let mut brep = import.brep.clone_standing_for_self().expect("standing copy");
        let mass = brep_mass_properties(&kernel, &mut brep, 1e-4, 1.0, solidcore::math::Point3::new(0.0, 0.0, 0.0), &attributes).expect("mass");
        assert!((mass.properties.volume - 12_629_528_647.66).abs() / 12_629_528_647.66 < 0.005, "volume {}", mass.properties.volume);
        assert!((mass.properties.area - 201_468_494.547).abs() / 201_468_494.547 < 0.005, "area {}", mass.properties.area);
    }

    /// A minimal single-face B-spline sheet (the same fixture shape the nurbs-face tests
    /// use) must be picked up by the sheet reader when the mixed reader rejects it.
    #[test]
    fn unsupported_curved_solid_lists_all_reader_attempts() {
        let Some(path) = fixture("piston.stp") else {
            eprintln!("skipping piston.stp: fixture not present");
            return;
        };
        let error = brep_from_step(&path).expect_err("piston.stp has B-spline faces");
        assert!(error.contains("no solidcore STEP importer can read"), "{error}");
        assert!(error.contains("B-spline sheet"), "{error}");
        assert!(error.contains("analytic primitive"), "{error}");
    }

    #[test]
    fn missing_file_is_an_import_error() {
        let path = std::env::temp_dir().join("nodecraft-step-import-missing.step");
        let error = brep_from_step(path.to_str().unwrap()).expect_err("missing file");
        assert!(error.contains("no solidcore STEP importer can read"), "{error}");
    }
}
