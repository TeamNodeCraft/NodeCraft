//! STEP geometry-type classification for import dispatch and diagnostics.

use std::collections::BTreeMap;

use crate::{read_topology_file, StepFile};

/// Counts of the geometry types that decide which import path can run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepTypeReport {
    pub solids: usize,
    pub faces: usize,
    pub planes: usize,
    pub cylinders: usize,
    pub cones: usize,
    pub spheres: usize,
    pub tori: usize,
    pub bspline_surfaces: usize,
    pub faceted_breps: usize,
    pub other_surfaces: usize,
    pub assemblies: bool,
}

/// The importer path selected for a STEP file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportPath {
    Assembly,
    Faceted,
    Planar,
    Bspline,
    AnalyticPrimitive,
    MixedUnsupported,
    Unsupported,
}

impl StepTypeReport {
    /// True when every face is planar.
    pub fn all_planar(&self) -> bool { self.faces > 0 && self.planes == self.faces }
    /// True when the file has mixed planar/curved faces.
    pub fn mixed_surfaces(&self) -> bool { self.planes > 0 && self.faces > self.planes }

    /// The importer path this classification selects.
    pub fn import_path(&self) -> ImportPath {
        if self.assemblies { return ImportPath::Assembly; }
        if self.faceted_breps > 0 { return ImportPath::Faceted; }
        if self.faces == 0 { return ImportPath::Unsupported; }
        if self.all_planar() { return ImportPath::Planar; }
        if self.bspline_surfaces > 0 { return ImportPath::Bspline; }
        let analytic = self.cylinders + self.cones + self.spheres + self.tori;
        if self.solids == 1 && analytic == 1 && self.faces <= 3 { return ImportPath::AnalyticPrimitive; }
        if self.mixed_surfaces() { return ImportPath::MixedUnsupported; }
        if self.solids == 1 && analytic > 0 { return ImportPath::AnalyticPrimitive; }
        ImportPath::Unsupported
    }
}

/// Per-solid geometry classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolidTypeReport {
    pub solid: u32,
    pub faces: usize,
    pub planes: usize,
    pub cylinders: usize,
    pub cones: usize,
    pub spheres: usize,
    pub tori: usize,
    pub bsplines: usize,
    pub faceted: bool,
}

impl SolidTypeReport {
    /// True when this solid mixes planar and curved faces.
    pub fn mixed(&self) -> bool { self.planes > 0 && self.faces > self.planes }
}

/// Classifies each `MANIFOLD_SOLID_BREP` separately.
pub fn classify_solids(path: &str) -> Result<Vec<SolidTypeReport>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let brep = crate::read_topology_file(path)?;
    let mut out = Vec::new();
    for solid in brep.solids.values() {
        let shell = brep.shells.get(&solid.shell).ok_or_else(|| format!("solid #{} has no shell", solid.id))?;
        let mut report = SolidTypeReport { solid: solid.id, faces: 0, planes: 0, cylinders: 0, cones: 0, spheres: 0, tori: 0, bsplines: 0, faceted: false };
        for face_id in &shell.faces {
            let Some(face) = brep.faces.get(face_id) else { continue };
            report.faces += 1;
            let Some(surface) = file.entities.get(&face.surface) else { continue };
            let members = surface.complex_members();
            if members.contains(&"PLANE") { report.planes += 1; }
            if members.contains(&"CYLINDRICAL_SURFACE") { report.cylinders += 1; }
            if members.contains(&"CONICAL_SURFACE") { report.cones += 1; }
            if members.contains(&"SPHERICAL_SURFACE") { report.spheres += 1; }
            if members.contains(&"TOROIDAL_SURFACE") { report.tori += 1; }
            if members.iter().any(|member| *member == "B_SPLINE_SURFACE" || *member == "B_SPLINE_SURFACE_WITH_KNOTS" || *member == "RATIONAL_B_SPLINE_SURFACE") { report.bsplines += 1; }
        }
        out.push(report);
    }
    Ok(out)
}

/// Classifies the solids/faces in a STEP file without building a Brep.
pub fn classify_step(path: &str) -> Result<StepTypeReport, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let brep = read_topology_file(path)?;
    let mut report = StepTypeReport { solids: brep.solids.len(), faces: brep.faces.len(), assemblies: text.contains("NEXT_ASSEMBLY_USAGE_OCCURRENCE"), faceted_breps: file.entities.values().filter(|entity| entity.name == "FACETED_BREP").count(), ..StepTypeReport::default() };
    let mut surfaces: BTreeMap<&str, usize> = BTreeMap::new();
    for face in brep.faces.values() {
        let Some(entity) = file.entities.get(&face.surface) else { report.other_surfaces += 1; continue };
        for member in entity.complex_members() {
            *surfaces.entry(member).or_default() += 1;
        }
    }
    for (name, count) in surfaces {
        match name {
            "PLANE" => report.planes = count,
            "CYLINDRICAL_SURFACE" => report.cylinders = count,
            "CONICAL_SURFACE" => report.cones = count,
            "SPHERICAL_SURFACE" => report.spheres = count,
            "TOROIDAL_SURFACE" => report.tori = count,
            "B_SPLINE_SURFACE" | "B_SPLINE_SURFACE_WITH_KNOTS" | "RATIONAL_B_SPLINE_SURFACE" => report.bspline_surfaces += count,
            _ => report.other_surfaces += count,
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write_box;
    #[test]
    fn classifies_mixed_planar_curved() {
        let mixed = StepTypeReport { solids: 1, faces: 6, planes: 4, cylinders: 2, ..StepTypeReport::default() };
        assert!(mixed.mixed_surfaces());
        assert_eq!(mixed.import_path(), ImportPath::MixedUnsupported);
    }

    #[test]
    fn classifies_solids_separately() {
        let path = std::env::temp_dir().join(format!("sn-classify-solids-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let solids = classify_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(solids.len(), 1);
        assert_eq!(solids[0].faces, 6);
        assert_eq!(solids[0].planes, 6);
        assert!(!solids[0].mixed());
    }

    #[test]
    fn classifies_planar_box() {
        let path = std::env::temp_dir().join(format!("sn-classify-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let report = classify_step(path.to_str().unwrap()).unwrap();
        assert_eq!(report.solids, 1);
        assert_eq!(report.faces, 6);
        assert_eq!(report.planes, 6);
        assert!(report.all_planar());
        assert!(!report.mixed_surfaces());
        assert_eq!(report.import_path(), ImportPath::Planar);
    }
}
