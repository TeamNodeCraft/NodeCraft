//! Neutral STEP B-rep topology: points, vertices, edges, loops, faces, shells, solids.
//!
//! Geometry references are preserved (surface/curve entity ids); this layer does not
//! evaluate curved geometry. It is the dependency-ordered skeleton used by the future
//! general STEP -> solidcore importer.

use std::collections::BTreeMap;

use crate::{Entity, StepFile, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct StepVertex { pub id: u32, pub point: [f64; 3] }
#[derive(Debug, Clone, PartialEq)]
pub struct StepEdge { pub id: u32, pub start: u32, pub end: u32, pub curve: u32, pub same_sense: bool }
#[derive(Debug, Clone, PartialEq)]
pub struct StepOrientedEdge { pub edge: u32, pub orientation: bool }
#[derive(Debug, Clone, PartialEq)]
pub struct StepLoop { pub id: u32, pub oriented_edges: Vec<StepOrientedEdge> }
#[derive(Debug, Clone, PartialEq)]
pub struct StepFace { pub id: u32, pub surface: u32, pub same_sense: bool, pub bounds: Vec<(u32, bool)>, pub outer_bounds: Vec<u32> }
#[derive(Debug, Clone, PartialEq)]
pub struct StepShell { pub id: u32, pub open: bool, pub faces: Vec<u32> }
#[derive(Debug, Clone, PartialEq)]
pub struct StepSolid { pub id: u32, pub shell: u32, pub voids: Vec<u32> }

#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepBrep {
    pub points: BTreeMap<u32, [f64; 3]>,
    pub vertices: BTreeMap<u32, StepVertex>,
    pub edges: BTreeMap<u32, StepEdge>,
    pub loops: BTreeMap<u32, StepLoop>,
    pub faces: BTreeMap<u32, StepFace>,
    pub shells: BTreeMap<u32, StepShell>,
    pub solids: BTreeMap<u32, StepSolid>,
}

fn reference(entity: &Entity, index: usize) -> Option<u32> {
    match entity.args.get(index)? { Value::Ref(id) => Some(*id), _ => None }
}

fn string_flag(entity: &Entity, index: usize) -> bool {
    matches!(entity.args.get(index), Some(Value::Enum(flag)) if flag == "T")
}

/// Extracts the topology graph from an already-parsed STEP file.
pub fn read_topology(file: &StepFile) -> StepBrep {
    let mut brep = StepBrep::default();
    // Dependency order: points -> vertices -> edges -> loops -> faces -> shells -> solids.
    for (id, entity) in &file.entities {
        if entity.name != "CARTESIAN_POINT" { continue; }
        if let Some(Value::List(coords)) = entity.args.get(1) {
            let mut point = [0.0; 3];
            for (slot, value) in point.iter_mut().zip(coords) {
                if let Value::Number(number) = value { *slot = *number; }
            }
            brep.points.insert(*id, point);
        }
    }
    for (id, entity) in &file.entities {
        if entity.name != "VERTEX_POINT" { continue; }
        if let Some(point) = reference(entity, 1).and_then(|point| brep.points.get(&point)) {
            brep.vertices.insert(*id, StepVertex { id: *id, point: *point });
        }
    }
    for (id, entity) in &file.entities {
        if entity.name != "EDGE_CURVE" { continue; }
        if let (Some(start), Some(end), Some(curve)) = (reference(entity, 1), reference(entity, 2), reference(entity, 3)) {
            if brep.vertices.contains_key(&start) && brep.vertices.contains_key(&end) {
                brep.edges.insert(*id, StepEdge { id: *id, start, end, curve, same_sense: string_flag(entity, 4) });
            }
        }
    }
    for (id, entity) in &file.entities {
        if entity.name != "EDGE_LOOP" { continue; }
        let Some(Value::List(uses)) = entity.args.get(1) else { continue };
        let oriented_edges = uses.iter().filter_map(|value| {
            let Value::Ref(use_id) = value else { return None };
            let use_entity = file.entities.get(use_id)?;
            let edge = reference(use_entity, 3)?;
            if !brep.edges.contains_key(&edge) { return None; }
            Some(StepOrientedEdge { edge, orientation: string_flag(use_entity, 4) })
        }).collect();
        brep.loops.insert(*id, StepLoop { id: *id, oriented_edges });
    }
    let mut bounds: BTreeMap<u32, (u32, bool, bool)> = BTreeMap::new();
    for (id, entity) in &file.entities {
        if entity.name != "FACE_OUTER_BOUND" && entity.name != "FACE_BOUND" { continue; }
        if let Some(loop_id) = reference(entity, 1) {
            if brep.loops.contains_key(&loop_id) { bounds.insert(*id, (loop_id, string_flag(entity, 2), entity.name == "FACE_OUTER_BOUND")); }
        }
    }
    for (id, entity) in &file.entities {
        if entity.name != "ADVANCED_FACE" && entity.name != "FACE_SURFACE" { continue; }
        let Some(surface) = reference(entity, 2) else { continue };
        let mut face_bounds = Vec::new();
        let mut outer_bounds = Vec::new();
        if let Some(Value::List(values)) = entity.args.get(1) {
            for value in values {
                let Value::Ref(bound) = value else { continue };
                let Some((loop_id, orientation, outer)) = bounds.get(bound).copied() else { continue };
                face_bounds.push((loop_id, orientation));
                if outer { outer_bounds.push(loop_id); }
            }
        }
        brep.faces.insert(*id, StepFace { id: *id, surface, same_sense: string_flag(entity, 3), bounds: face_bounds, outer_bounds });
    }
    let base_faces: Vec<(u32, StepFace)> = brep.faces.iter().map(|(id, face)| (*id, face.clone())).collect();
    for (id, entity) in &file.entities {
        if entity.name != "ORIENTED_FACE" { continue; }
        let Some(base_id) = reference(entity, 1) else { continue };
        let Some((_, base)) = base_faces.iter().find(|(candidate, _)| *candidate == base_id) else { continue };
        let orientation = string_flag(entity, 2);
        brep.faces.insert(*id, StepFace { id: *id, surface: base.surface, same_sense: if orientation { base.same_sense } else { !base.same_sense }, bounds: base.bounds.clone(), outer_bounds: base.outer_bounds.clone() });
    }
    for (id, entity) in &file.entities {
        if entity.name != "CLOSED_SHELL" && entity.name != "OPEN_SHELL" { continue; }
        let faces = match entity.args.get(1) {
            Some(Value::List(values)) => values.iter().filter_map(|value| match value {
                Value::Ref(face) if brep.faces.contains_key(face) => Some(*face), _ => None,
            }).collect(),
            _ => Vec::new(),
        };
        brep.shells.insert(*id, StepShell { id: *id, open: entity.name == "OPEN_SHELL", faces });
    }
    for (id, entity) in &file.entities {
        if entity.name != "MANIFOLD_SOLID_BREP" && entity.name != "BREP_WITH_VOIDS" { continue; }
        if let Some(shell) = reference(entity, 1) {
            if brep.shells.contains_key(&shell) {
                let voids = match entity.args.get(2) {
                    Some(Value::List(values)) => values.iter().filter_map(|value| match value {
                        Value::Ref(void) if brep.shells.contains_key(void) => Some(*void), _ => None,
                    }).collect(),
                    _ => Vec::new(),
                };
                brep.solids.insert(*id, StepSolid { id: *id, shell, voids });
            }
        }
    }
    brep
}


impl StepBrep {
    /// Structural checks that do not evaluate curved geometry.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for edge in self.edges.values() {
            if !self.vertices.contains_key(&edge.start) { issues.push(format!("edge #{} has missing start vertex #{}", edge.id, edge.start)); }
            if !self.vertices.contains_key(&edge.end) { issues.push(format!("edge #{} has missing end vertex #{}", edge.id, edge.end)); }
            // A closed edge (start == end) is a legitimate seam edge when its curve is
            // closed; topological validation of the shell use counts below rejects a
            // solitary zero-length edge. Keep it here instead of rejecting seams.  
            let _ = edge.start == edge.end;
        }
        for loop_ in self.loops.values() {
            let closed_loop = !loop_.oriented_edges.is_empty() && loop_.oriented_edges.iter().all(|use_| {
                self.edges.get(&use_.edge).is_some_and(|edge| edge.start == edge.end)
            });
            let seam_pair = loop_.oriented_edges.len() == 2
                && loop_.oriented_edges[0].edge == loop_.oriented_edges[1].edge
                && loop_.oriented_edges[0].orientation != loop_.oriented_edges[1].orientation;
            if loop_.oriented_edges.len() < 3 && !closed_loop && !seam_pair { issues.push(format!("loop #{} has fewer than 3 edges", loop_.id)); }
            let mut first_start = None;
            let mut previous_end = None;
            let mut used_edges: std::collections::BTreeMap<u32, bool> = std::collections::BTreeMap::new();
            for use_ in &loop_.oriented_edges {
                if let Some(previous) = used_edges.insert(use_.edge, use_.orientation) {
                    if previous == use_.orientation { issues.push(format!("loop #{} uses edge #{} twice in the same direction", loop_.id, use_.edge)); }
                }
                let Some(edge) = self.edges.get(&use_.edge) else {
                    issues.push(format!("loop #{} references missing edge #{}", loop_.id, use_.edge));
                    continue;
                };
                let (start, end) = if use_.orientation { (edge.start, edge.end) } else { (edge.end, edge.start) };
                if first_start.is_none() { first_start = Some(start); }
                if let Some(previous) = previous_end {
                    if previous != start { issues.push(format!("loop #{} is not closed: edge #{} starts at vertex {} after vertex {}", loop_.id, edge.id, start, previous)); }
                }
                previous_end = Some(end);
            }
            if let (Some(first), Some(last)) = (first_start, previous_end) {
                if first != last { issues.push(format!("loop #{} does not close: starts at vertex {first}, ends at vertex {last}", loop_.id)); }
            }
        }
        let mut used_surfaces = std::collections::BTreeSet::new();
        for face in self.faces.values() {
            used_surfaces.insert(face.surface);
            if face.bounds.is_empty() { issues.push(format!("face #{} has no bounds", face.id)); }
            for (loop_id, _) in &face.bounds {
                if !self.loops.contains_key(loop_id) { issues.push(format!("face #{} references missing loop #{}", face.id, loop_id)); }
            }
        }
        for shell in self.shells.values() {
            if shell.faces.is_empty() { issues.push(format!("shell #{} has no faces", shell.id)); }
            let mut seen = std::collections::BTreeSet::new();
            for face in &shell.faces {
                if !seen.insert(*face) { issues.push(format!("shell #{} lists face #{} more than once", shell.id, face)); }
            }
            let mut uses: std::collections::BTreeMap<(u32, u32), usize> = std::collections::BTreeMap::new();
            for face_id in &shell.faces {
                let Some(face) = self.faces.get(face_id) else { issues.push(format!("shell #{} references missing face #{}", shell.id, face_id)); continue };
                for (loop_id, _) in &face.bounds {
                    let Some(loop_) = self.loops.get(loop_id) else { continue };
                    for use_ in &loop_.oriented_edges {
                        let Some(edge) = self.edges.get(&use_.edge) else { continue };
                        let key = if edge.start < edge.end { (edge.start, edge.end) } else { (edge.end, edge.start) };
                        *uses.entry(key).or_default() += 1;
                    }
                }
            }
            if !shell.open {
                for (edge, count) in uses {
                    let valid = if edge.0 == edge.1 { count >= 2 && count % 2 == 0 } else { count == 2 };
                    if !valid { issues.push(format!("closed shell #{} edge ({},{}) used {count} times", shell.id, edge.0, edge.1)); }
                }
            }
        }
        for solid in self.solids.values() {
            if !self.shells.contains_key(&solid.shell) { issues.push(format!("solid #{} references missing shell #{}", solid.id, solid.shell)); }
            for void in &solid.voids { if !self.shells.contains_key(void) { issues.push(format!("solid #{} references missing void shell #{}", solid.id, void)); } }
        }
        issues
    }

    /// The STEP `same_sense` flag of face `id`, when the face exists.
    pub fn face_same_sense(&self, id: u32) -> Option<bool> {
        self.faces.get(&id).map(|face| face.same_sense)
    }


    /// Edge ids used an odd number of times across the shells (open-boundary candidates).
    pub fn open_edges(&self) -> Vec<u32> {
        let mut uses: BTreeMap<u32, usize> = BTreeMap::new();
        for face in self.faces.values() {
            for (loop_id, _) in &face.bounds {
                let Some(loop_) = self.loops.get(loop_id) else { continue };
                for use_ in &loop_.oriented_edges { *uses.entry(use_.edge).or_default() += 1; }
            }
        }
        uses.into_iter().filter(|(edge, count)| { self.edges.get(edge).is_none_or(|edge| edge.start != edge.end) && count % 2 == 1 }).map(|(edge, _)| edge).collect()
    }


    /// One-line inventory used in import diagnostics.
    pub fn summary(&self) -> String {
        format!("{} solid(s), {} shell(s), {} face(s), {} loop(s), {} edge(s), {} vertex(ies)", self.solids.len(), self.shells.len(), self.faces.len(), self.loops.len(), self.edges.len(), self.vertices.len())
    }
}

/// Reads topology from a STEP file path.
pub fn read_topology_file(path: &str) -> Result<StepBrep, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    Ok(read_topology(&file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_has_complete_topology() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/Sample AP203 Data Files/block.stp");
        if !std::path::Path::new(path).exists() { return; }
        let brep = read_topology_file(path).unwrap();
        assert_eq!(brep.solids.len(), 1);
        assert_eq!(brep.shells.len(), 1);
        assert_eq!(brep.faces.len(), 6);
        assert_eq!(brep.loops.len(), 6);
        assert_eq!(brep.edges.len(), 12);
        assert_eq!(brep.vertices.len(), 8);
        assert!(brep.solids.values().all(|solid| brep.shells[&solid.shell].faces.len() == 6));
        assert!(brep.validate().is_empty(), "{:?}", brep.validate());
    }

    #[test]
    fn validation_detects_duplicate_shell_face() {
        let mut brep = StepBrep::default();
        brep.shells.insert(1, StepShell { id: 1, open: false, faces: vec![2, 2] });
        brep.faces.insert(2, StepFace { id: 2, surface: 1, same_sense: true, bounds: vec![], outer_bounds: vec![] });
        let issues = brep.validate();
        assert!(issues.iter().any(|issue| issue.contains("more than once")), "{issues:?}");
    }

    #[test]
    fn validation_detects_repeated_loop_edge() {
        let mut brep = StepBrep::default();
        for id in 1..=3 { brep.vertices.insert(id, StepVertex { id, point: [id as f64, 0.0, 0.0] }); }
        brep.edges.insert(10, StepEdge { id: 10, start: 1, end: 2, curve: 1, same_sense: true });
        brep.edges.insert(11, StepEdge { id: 11, start: 2, end: 3, curve: 2, same_sense: true });
        brep.loops.insert(20, StepLoop { id: 20, oriented_edges: vec![StepOrientedEdge { edge: 10, orientation: true }, StepOrientedEdge { edge: 11, orientation: true }, StepOrientedEdge { edge: 10, orientation: true }] });
        let issues = brep.validate();
        assert!(issues.iter().any(|issue| issue.contains("same direction")), "{issues:?}");
    }

    #[test]
    fn validation_allows_opposite_seam_edge() {
        let mut brep = StepBrep::default();
        for id in 1..=2 { brep.vertices.insert(id, StepVertex { id, point: [id as f64, 0.0, 0.0] }); }
        brep.edges.insert(10, StepEdge { id: 10, start: 1, end: 2, curve: 1, same_sense: true });
        brep.loops.insert(20, StepLoop { id: 20, oriented_edges: vec![StepOrientedEdge { edge: 10, orientation: true }, StepOrientedEdge { edge: 10, orientation: false }] });
        let issues = brep.validate();
        assert!(!issues.iter().any(|issue| issue.contains("twice in the same direction")), "{issues:?}");
    }

    #[test]
    fn validation_detects_open_loop() {
        let mut brep = StepBrep::default();
        for id in 1..=3 { brep.vertices.insert(id, StepVertex { id, point: [id as f64, 0.0, 0.0] }); }
        brep.edges.insert(10, StepEdge { id: 10, start: 1, end: 2, curve: 1, same_sense: true });
        brep.edges.insert(11, StepEdge { id: 11, start: 3, end: 2, curve: 2, same_sense: true });
        brep.loops.insert(20, StepLoop { id: 20, oriented_edges: vec![StepOrientedEdge { edge: 10, orientation: true }, StepOrientedEdge { edge: 11, orientation: true }, StepOrientedEdge { edge: 10, orientation: true }] });
        let issues = brep.validate();
        assert!(issues.iter().any(|issue| issue.contains("is not closed") || issue.contains("does not close")), "{issues:?}");
    }

    #[test]
    fn validation_reports_broken_edges_and_shells() {
        let mut brep = StepBrep::default();
        brep.edges.insert(1, StepEdge { id: 1, start: 99, end: 100, curve: 1, same_sense: true });
        brep.loops.insert(2, StepLoop { id: 2, oriented_edges: vec![StepOrientedEdge { edge: 1, orientation: true }] });
        brep.faces.insert(3, StepFace { id: 3, surface: 1, same_sense: true, bounds: vec![(2, true)], outer_bounds: vec![2] });
        brep.shells.insert(4, StepShell { id: 4, open: false, faces: vec![3] });
        brep.solids.insert(5, StepSolid { id: 5, shell: 4, voids: vec![] });
        let issues = brep.validate();
        assert!(issues.iter().any(|issue| issue.contains("missing start vertex")));
        assert!(issues.iter().any(|issue| issue.contains("fewer than 3 edges")));
    }

    #[test]
    fn turbine_exposes_curved_topology_without_panicking() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/Sample AP203 Data Files/turbine.stp");
        if !std::path::Path::new(path).exists() { return; }
        let brep = read_topology_file(path).unwrap();
        assert!(brep.faces.len() > 100, "faces {}", brep.faces.len());
        assert!(!brep.solids.is_empty());
    }
}

/// Reads each closed planar solid as a separate [`crate::PlanarBody`].
///
/// Curved faces or non-planar solids are rejected explicitly; holes are preserved from
/// the topology's `FACE_OUTER_BOUND` / `FACE_BOUND` grouping.
pub fn read_planar_solids(path: &str) -> Result<Vec<crate::PlanarBody>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = crate::unit_scale(&text);
    let tolerance = crate::read_tolerance(path).ok().flatten().unwrap_or(1e-7).clamp(1e-9, 1e-3);
    let file = StepFile::parse(&text)?;
    let brep = read_topology(&file);
    if brep.solids.is_empty() { return Err("STEP file has no MANIFOLD_SOLID_BREP".into()); }
    let mut bodies = Vec::new();
    for solid in brep.solids.values() {
        let shell = brep.shells.get(&solid.shell).ok_or_else(|| format!("solid #{} has missing shell #{}", solid.id, solid.shell))?;
        let mut body = crate::PlanarBody::default();
        let mut lookup: BTreeMap<[i64; 3], usize> = BTreeMap::new();
        for face_id in &shell.faces {
            let face = brep.faces.get(face_id).ok_or_else(|| format!("shell #{} has missing face #{face_id}", shell.id))?;
            let surface = file.entities.get(&face.surface).ok_or_else(|| format!("face #{} has missing surface #{}", face.id, face.surface))?;
            if surface.name != "PLANE" {
                return Err(format!("solid #{} contains curved surface {} on face #{}", solid.id, surface.name, face.id));
            }
            let mut loops: Vec<(Vec<usize>, bool)> = Vec::new();
            for (loop_id, orientation) in &face.bounds {
                let loop_ = brep.loops.get(loop_id).ok_or_else(|| format!("face #{} has missing loop #{loop_id}", face.id))?;
                let mut points = Vec::new();
                for use_ in &loop_.oriented_edges {
                    let edge = brep.edges.get(&use_.edge).ok_or_else(|| format!("loop #{loop_id} has missing edge #{}", use_.edge))?;
                    let vertex = if use_.orientation { edge.start } else { edge.end };
                    let vertex = brep.vertices.get(&vertex).ok_or_else(|| format!("edge #{} has missing vertex", edge.id))?;
                    let point = [vertex.point[0] * scale, vertex.point[1] * scale, vertex.point[2] * scale];
                    let key = [(point[0] * 1e7).round() as i64, (point[1] * 1e7).round() as i64, (point[2] * 1e7).round() as i64];
                    let index = *lookup.entry(key).or_insert_with(|| { body.points.push(point); body.points.len() - 1 });
                    if points.last().copied() != Some(index) { points.push(index); }
                }
                if points.first() == points.last() { points.pop(); }
                if !*orientation { points.reverse(); }
                if points.len() >= 3 { loops.push((points, face.outer_bounds.contains(loop_id))); }
                else if face.outer_bounds.contains(loop_id) { return Err(format!("face #{} has a non-polygonal outer loop", face.id)); }
                else { return Err(format!("face #{} has a short inner loop", face.id)); }
            }
            if loops.is_empty() { continue; }
            let outer_index = loops.iter().position(|(_, outer)| *outer).unwrap_or(0);
            let outer = loops.remove(outer_index).0;
            body.push_face(outer, loops.into_iter().map(|(points, _)| points).collect());
        }
        if body.faces.is_empty() { return Err(format!("solid #{} has no planar faces", solid.id)); }
        body.weld_points(tolerance);
        if body.volume().is_finite() && body.volume() < 0.0 { body.repair_orientation()?; }
        bodies.push(body);
    }
    Ok(bodies)
}

#[cfg(test)]
mod multi_body_tests {
    use super::*;
    #[test]
    fn import_repairs_inside_out_solid() {
        let path = std::env::temp_dir().join(format!("sn-orient-import-{}.step", std::process::id()));
        let source = std::env::temp_dir().join(format!("sn-orient-source-{}.step", std::process::id()));
        crate::write_box(source.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let mut body = crate::read_planar_body(source.to_str().unwrap()).unwrap();
        for face in &mut body.faces { face.reverse(); }
        crate::write_planar_body(path.to_str().unwrap(), &body).unwrap();
        let bodies = read_planar_solids(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert!((bodies[0].volume() - 8.0).abs() < 1e-9, "{}", bodies[0].volume());
    }

    #[test]
    fn merged_boxes_read_as_two_planar_solids() {
        let dir = std::env::temp_dir().join(format!("sn-multi-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.step"); let b = dir.join("b.step"); let out = dir.join("both.step");
        crate::write_box(a.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        crate::write_box(b.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        crate::merge_step_files(&[a.to_string_lossy().into(), b.to_string_lossy().into()], out.to_str().unwrap()).unwrap();
        let bodies = read_planar_solids(out.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 2);
        assert!((bodies[0].volume() - 8.0).abs() < 1e-6);
        assert!((bodies[1].volume() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn hollow_box_exposes_void_shell() {
        let path = std::env::temp_dir().join(format!("sn-void-topology-{}.step", std::process::id()));
        crate::write_box_with_void(path.to_str().unwrap(), [10.0, 10.0, 10.0], [4.0, 4.0, 4.0]).unwrap();
        let brep = read_topology_file(path.to_str().unwrap()).unwrap();
        assert_eq!(brep.solids.len(), 1);
        let solid = brep.solids.values().next().unwrap();
        assert_eq!(solid.voids.len(), 1);
        assert_eq!(brep.shells.len(), 2);
        assert!(brep.validate().is_empty(), "{:?}", brep.validate());
    }

    #[test]
    fn block_reads_as_one_planar_solid() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../tests/Sample AP203 Data Files/block.stp");
        if !std::path::Path::new(path).exists() { return; }
        let bodies = read_planar_solids(path).unwrap();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0].faces.len(), 6);
        assert!((bodies[0].volume() - 60.0).abs() < 1e-6, "{}", bodies[0].volume());
    }
    #[test]
    fn exposes_face_same_sense() {
        let mut brep = StepBrep::default();
        brep.faces.insert(1, StepFace { id: 1, surface: 1, same_sense: false, bounds: vec![], outer_bounds: vec![] });
        assert_eq!(brep.face_same_sense(1), Some(false));
        assert_eq!(brep.face_same_sense(2), None);
    }

    #[test]
    fn detects_open_edges() {
        use crate::write_open_shell;
        let path = std::env::temp_dir().join(format!("sn-open-edges-{}.step", std::process::id()));
        let body = crate::PlanarBody { points: vec![[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]], faces: vec![vec![0,1,2,3]], holes: vec![vec![]] };
        write_open_shell(path.to_str().unwrap(), &body).unwrap();
        let brep = read_topology_file(path.to_str().unwrap()).unwrap();
        assert_eq!(brep.open_edges().len(), 4);
    }

}
