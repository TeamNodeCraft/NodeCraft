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
//! [`step_display_mesh`] goes one step further: it tessellates the imported B-rep with
//! [`crate::curved::brep_display_mesh`] and returns the display [`Mesh`] together with
//! the topology counters and any warnings/fallbacks the worker diagnostics need.
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
use nodecraft_protocol::Mesh;

use crate::curved::{CurvedTessOptions, brep_display_mesh};

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

/// A STEP file imported into solidcore and tessellated into a display mesh.
#[derive(Debug)]
pub struct StepMeshImport {
    /// Display mesh tessellated from the imported B-rep.
    pub mesh: Mesh,
    /// Which solidcore STEP reader produced the B-rep.
    pub kind: StepBrepKind,
    /// Number of faces in the imported B-rep.
    pub brep_faces: usize,
    /// Number of edges in the imported B-rep.
    pub brep_edges: usize,
    /// Number of vertices in the imported B-rep.
    pub brep_vertices: usize,
    /// Non-fatal issues met while importing/tessellating.
    pub warnings: Vec<String>,
    /// Paths that fell back to a less exact representation (none today; reserved for
    /// readers that later gain approximate fallbacks).
    pub fallbacks: Vec<String>,
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

/// Import and tessellate a STEP file through solidcore, with default display options.
///
/// # Errors
///
/// Those of [`brep_from_step`] plus display-tessellation failures (the imported B-rep is
/// reported as not renderable rather than silently replaced by a different mesh).
pub fn step_display_mesh(path: &str) -> Result<StepMeshImport, String> {
    step_display_mesh_with_options(path, &CurvedTessOptions::default())
}

/// [`step_display_mesh`] with explicit display-grid density.
pub fn step_display_mesh_with_options(
    path: &str,
    options: &CurvedTessOptions,
) -> Result<StepMeshImport, String> {
    let import = brep_from_step(path)?;
    let kernel = Kernel::default();
    let mesh = brep_display_mesh(&kernel, &import.brep, options).map_err(|error| {
        format!(
            "{} STEP B-rep imported from {path}, but display tessellation failed: {error}",
            import.kind
        )
    })?;
    if mesh.faces.is_empty() {
        return Err(format!(
            "{} STEP B-rep imported from {path} produced no display triangles",
            import.kind
        ));
    }
    Ok(StepMeshImport {
        mesh,
        kind: import.kind,
        brep_faces: import.brep.faces().len(),
        brep_edges: import.brep.edges().len(),
        brep_vertices: import.brep.vertices().len(),
        warnings: Vec::new(),
        fallbacks: Vec::new(),
    })
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

    fn mesh_bounds(mesh: &Mesh) -> ([f32; 3], [f32; 3]) {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        (min, max)
    }

    fn assert_mesh_sane(mesh: &Mesh) {
        assert!(!mesh.vertices.is_empty(), "mesh has no vertices");
        assert!(!mesh.faces.is_empty(), "mesh has no triangles");
        assert_eq!(mesh.normals.len(), mesh.vertices.len(), "normal count");
        for vertex in &mesh.vertices {
            assert!(vertex.iter().all(|value| value.is_finite()), "non-finite vertex {vertex:?}");
        }
        for normal in &mesh.normals {
            let length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-3, "non-unit normal {normal:?}");
        }
        for face in &mesh.faces {
            for index in face {
                assert!((*index as usize) < mesh.vertices.len(), "face index out of range");
            }
        }
    }

    /// Analytic primitive STEP files are accepted by the dispatcher's analytic reader
    /// (the worker normally routes these earlier; this proves the dispatcher branch).
    #[test]
    fn analytic_cylinder_imports_through_dispatcher() {
        let path = std::env::temp_dir().join(format!(
            "nodecraft-step-import-cylinder-{}.step",
            std::process::id()
        ));
        nodecraft_step::write_cylinder(path.to_str().unwrap(), 5.0, 10.0).expect("write cylinder");
        let result = step_display_mesh(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let mesh_import = result.expect("analytic cylinder display mesh");
        assert_eq!(mesh_import.kind, StepBrepKind::AnalyticPrimitive);
        assert_eq!(mesh_import.brep_faces, 3);
        assert_mesh_sane(&mesh_import.mesh);
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
        let kernel = Kernel::default();
        let store = AttributeStore::new();
        let key = |_| ObjectKey(2_000_000);
        let attributes = PropertyAttributes { store: &store, key_of: &key };
        let mut brep = import.brep.clone();
        let mass = brep_mass_properties(&kernel, &mut brep, 1e-4, 1.0, solidcore::math::Point3::new(0.0, 0.0, 0.0), &attributes).expect("mass");
        assert!((mass.properties.volume - 12_629_528_647.66).abs() / 12_629_528_647.66 < 0.005, "volume {}", mass.properties.volume);
        assert!((mass.properties.area - 201_468_494.547).abs() / 201_468_494.547 < 0.005, "area {}", mass.properties.area);
    }

    /// A minimal single-face B-spline sheet (the same fixture shape the nurbs-face tests
    /// use) must be picked up by the sheet reader when the mixed reader rejects it.
    #[test]
    fn bspline_sheet_imports_through_sheet_reader() {
        let body = "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(0.,1.,0.));\n\
             #3=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(1.,1.,0.));\n\
             #10=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
                  (2,2),(2,2),(0.,1.),(0.,1.),.UNSPECIFIED.);\n\
             #20=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#3),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n\
             #21=B_SPLINE_CURVE_WITH_KNOTS('',1,(#3,#4),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n\
             #22=B_SPLINE_CURVE_WITH_KNOTS('',1,(#2,#4),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n\
             #23=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n\
             #30=VERTEX_POINT('',#1);\n\
             #31=VERTEX_POINT('',#3);\n\
             #32=VERTEX_POINT('',#4);\n\
             #33=VERTEX_POINT('',#2);\n\
             #40=EDGE_CURVE('',#30,#31,#20,.T.);\n\
             #41=EDGE_CURVE('',#31,#32,#21,.T.);\n\
             #42=EDGE_CURVE('',#33,#32,#22,.T.);\n\
             #43=EDGE_CURVE('',#30,#33,#23,.T.);\n\
             #50=ORIENTED_EDGE('',*,*,#40,.T.);\n\
             #51=ORIENTED_EDGE('',*,*,#41,.T.);\n\
             #52=ORIENTED_EDGE('',*,*,#42,.F.);\n\
             #53=ORIENTED_EDGE('',*,*,#43,.F.);\n\
             #60=EDGE_LOOP('',(#50,#51,#52,#53));\n\
             #61=FACE_OUTER_BOUND('',#60,.T.);\n\
             #70=ADVANCED_FACE('',(#61),#10,.T.);\n";
        let text = format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\n\
             FILE_NAME('','',(''),(''),'nodecraft-solidcore-test','nodecraft-solidcore-test','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{body}ENDSEC;\nEND-ISO-10303-21;\n"
        );
        let path = std::env::temp_dir().join(format!(
            "nodecraft-step-import-sheet-{}.step",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        let result = step_display_mesh(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let mesh_import = result.expect("B-spline sheet display mesh");
        assert_eq!(mesh_import.kind, StepBrepKind::BsplineSheet);
        assert_eq!(mesh_import.brep_faces, 1);
        assert_eq!(mesh_import.brep_edges, 4);
        assert_eq!(mesh_import.brep_vertices, 4);
        assert_mesh_sane(&mesh_import.mesh);
        let (min, max) = mesh_bounds(&mesh_import.mesh);
        for axis in 0..2 {
            assert!((min[axis] - 0.0).abs() < 0.05, "min {min:?}");
            assert!((max[axis] - 1.0).abs() < 0.05, "max {max:?}");
        }
    }

    /// Genuinely unsupported curved solids report every reader's reason instead of a
    /// partial or guessed B-rep.
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
