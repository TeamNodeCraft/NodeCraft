//! Representation-scoped STEP import for in-file assembly components (Step-106).
//!
//! An assembly file keeps the geometry of its placed parts in *shape representations*
//! inside the same file. The product/assembly tree (see `step-assemblies`) resolves the
//! occurrence tree, placements and external document references; this module turns one
//! representation id into the display mesh of exactly the solids/facets that
//! representation (or a representation related to it) contains, using the same readers as
//! the whole-file importer:
//!
//! * planar and analytic-primitive solids come from `nodecraft_step::import_solids_partial`
//!   restricted to the representation's item list,
//! * `FACETED_BREP` lumps come from `nodecraft_step::read_faceted_breps`,
//! * a sheet/surface representation that is the only geometry in the file falls back to the
//!   general solidcore B-spline/sheet reader (`crate::import_solidcore_step`),
//! * solids the planar/analytic extractor skipped are reported as **mixed/curved solids
//!   blocked by Mx-72** instead of a generic "not supported".
//!
//! Two STEP details make representation lookup non-trivial:
//!
//! 1. **Related representations.** An `ADVANCED_BREP_SHAPE_REPRESENTATION` (geometry) is
//!    often linked to the `SHAPE_REPRESENTATION` (placements) a product definition names
//!    through a plain `SHAPE_REPRESENTATION_RELATIONSHIP` with no transformation
//!    (`cubcylso.stp`: the child occurrence names `#302`, geometry is `#301`). The lookup
//!    follows those untransformed relationships; transformation-carrying relationships are
//!    excluded because they are occurrence placements (`CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`),
//!    not geometry links.
//! 2. **Placement-only representations.** Sub-assembly representations list only
//!    `AXIS2_PLACEMENT_3D` items; those are skipped for internal graph nodes and reported as
//!    "no solid geometry" only for leaf occurrences that are supposed to carry geometry.
//!
//! Every failure names the representation id and the reason category; mixed/curved solids
//! explicitly say `Mx-72` so the remaining blocker is not confused with this mechanism.

use std::collections::{BTreeMap, BTreeSet};

use nodecraft_step::{PlanarBody, StepBody, StepFile, Value};

use crate::StepImport;

/// One representation's resolved geometry, before meshing.
#[derive(Debug, Default)]
struct Geometry {
    solids: Vec<u32>,
    faceted: Vec<u32>,
    sheets: Vec<u32>,
    mapped_items: usize,
    source_representations: Vec<u32>,
}

impl Geometry {
    fn has_any(&self) -> bool {
        !self.solids.is_empty() || !self.faceted.is_empty() || !self.sheets.is_empty()
    }
}

/// Shared parse state for importing several representations of one file.
pub(crate) struct RepresentationContext {
    path: String,
    file: StepFile,
    /// Parsed topology (needed by the per-solid exact mixed reader).
    topology: nodecraft_step::StepBrep,
    /// Millimetres per file unit, used when building mixed solids.
    unit_scale_mm: f64,
    /// Solids extracted by the planar/analytic reader, in ascending solid-id order.
    bodies: Vec<StepBody>,
    /// Solid ids the planar/analytic reader could not mesh (mixed/curved/B-spline).
    skipped: BTreeSet<u32>,
    /// Solid id -> index into `bodies`.
    body_by_solid: BTreeMap<u32, usize>,
    /// Faceted lumps, in ascending `FACETED_BREP` id order (when the counts line up).
    faceted_bodies: Vec<PlanarBody>,
    faceted_ids: Vec<u32>,
    /// Undirected untransformed representation-relationship pairs.
    relations: BTreeMap<u32, BTreeSet<u32>>,
    /// Entity ids that are shape representations with an item list.
    representations: BTreeSet<u32>,
    /// Number of representations in the file that resolve to actual geometry.
    geometry_representation_count: usize,
}

impl RepresentationContext {
    /// Parses `path` and prepares the solids/facets/relationships needed for lookups.
    ///
    /// # Errors
    ///
    /// I/O and STEP syntax errors. A file without solids is *not* an error: its faceted or
    /// sheet representations can still be imported.
    pub(crate) fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
        let file = StepFile::parse(&text).map_err(|error| format!("{path}: STEP parse error: {error}"))?;
        let topology = nodecraft_step::read_topology(&file);
        let unit_scale_mm = if let Ok(Some(context)) = nodecraft_step::read_unit_context(path) {
            context.length_unit.millimetres()
        } else {
            nodecraft_step::read_length_unit(path).map(|unit| unit.millimetres()).unwrap_or(1.0)
        };
        let (bodies, skipped) = match nodecraft_step::import_solids_partial(path) {
            Ok((bodies, skipped)) => (bodies, skipped),
            Err(_) => (Vec::new(), Vec::new()),
        };
        let skipped: BTreeSet<u32> = skipped.into_iter().collect();
        let mut body_by_solid = BTreeMap::new();
        let mut body_index = 0usize;
        for solid_id in topology.solids.keys() {
            if skipped.contains(solid_id) {
                continue;
            }
            body_by_solid.insert(*solid_id, body_index);
            body_index += 1;
        }
        let faceted_ids: Vec<u32> = file
            .entities
            .iter()
            .filter(|(_, entity)| entity.name == "FACETED_BREP")
            .map(|(id, _)| *id)
            .collect();
        let faceted_bodies = nodecraft_step::read_faceted_breps(path).unwrap_or_default();
        let mut context = Self {
            path: path.to_string(),
            file,
            topology,
            unit_scale_mm,
            bodies,
            skipped,
            body_by_solid,
            faceted_bodies,
            faceted_ids,
            relations: BTreeMap::new(),
            representations: BTreeSet::new(),
            geometry_representation_count: 0,
        };
        context.collect_representations_and_relations();
        let mut geometry_count = 0usize;
        for representation in context.representations.iter().copied().collect::<Vec<_>>() {
            if context.resolve_geometry(representation).has_any() {
                geometry_count += 1;
            }
        }
        context.geometry_representation_count = geometry_count;
        Ok(context)
    }

    /// True when `representation_id` (directly or through an untransformed related
    /// representation) carries solid, faceted or sheet geometry.
    pub(crate) fn has_geometry(&self, representation_id: u32) -> bool {
        self.resolve_geometry(representation_id).has_any()
    }

    /// Imports exactly the geometry of `representation_id`.
    ///
    /// # Errors
    ///
    /// * unknown or non-representation entity ids,
    /// * mixed/curved solids that the planar/analytic extractor skipped (message contains
    ///   `Mx-72`),
    /// * `MAPPED_ITEM` geometry (not resolved by the assembly parser),
    /// * sheet geometry when the file has more than one geometry representation (a
    ///   whole-file sheet reader would import the wrong part),
    /// * placement-only representations ("no solid geometry").
    pub(crate) fn import(&self, representation_id: u32) -> Result<StepImport, String> {
        let geometry = self.resolve_geometry(representation_id);
        if !self.representations.contains(&representation_id) {
            return Err(format!("representation #{representation_id} does not exist in {}", self.path));
        }
        if !geometry.has_any() {
            if geometry.mapped_items > 0 {
                return Err(format!(
                    "representation #{representation_id} places its geometry through MAPPED_ITEM, which the assembly parser does not resolve"
                ));
            }
            return Err(format!("representation #{representation_id} has no solid geometry (placement-only representation)"));
        }

        let mut failures: Vec<String> = Vec::new();
        let mut bodies: Vec<StepBody> = Vec::new();
        let mut skipped_solids: Vec<u32> = Vec::new();
        for solid_id in &geometry.solids {
            if self.skipped.contains(solid_id) {
                skipped_solids.push(*solid_id);
                continue;
            }
            match self.body_by_solid.get(solid_id) {
                Some(index) => match self.bodies.get(*index) {
                    Some(body) => bodies.push(body.clone()),
                    None => failures.push(format!("internal error: solid #{solid_id} body index {index} is missing")),
                },
                None => failures.push(format!("internal error: solid #{solid_id} has no extracted body")),
            }
        }
        if self.faceted_bodies.len() != self.faceted_ids.len() && !geometry.faceted.is_empty() {
            failures.push(format!(
                "{} faceted lump(s) could not be matched to their FACETED_BREP entities",
                geometry.faceted.len()
            ));
        } else {
            for faceted_id in &geometry.faceted {
                match self.faceted_ids.iter().position(|id| id == faceted_id) {
                    Some(index) => match self.faceted_bodies.get(index) {
                        Some(body) => bodies.push(StepBody::Planar(body.clone())),
                        None => failures.push(format!("internal error: faceted lump #{faceted_id} has no body")),
                    },
                    None => failures.push(format!("faceted lump #{faceted_id} is not in this file")),
                }
            }
        }

        if !failures.is_empty() {
            return Err(format!(
                "representation #{representation_id} cannot be meshed yet: {}",
                failures.join("; ")
            ));
        }

        // Representation-scoped mixed children: the planar/analytic reader skipped these
        // solids, so build each one with the exact per-solid mixed reader (Step-108),
        // tessellate it and merge with the planar/analytic mesh. All-or-nothing: any
        // failed solid rejects the representation (no partial body).
        let mut mixed_meshes: Vec<nodecraft_protocol::Mesh> = Vec::new();
        let mut mixed_face_count = 0usize;
        let mut mixed_edge_count = 0usize;
        let mut mixed_vertex_count = 0usize;
        for solid_id in &skipped_solids {
            match nodecraft_solidcore::step_mixed::build_mixed(&self.file, &self.topology, *solid_id, self.unit_scale_mm) {
                Ok((brep, _data, _face_ids)) => {
                    let mesh = nodecraft_solidcore::curved::brep_display_mesh(
                        &Default::default(),
                        &brep,
                        &Default::default(),
                    )
                    .map_err(|error| {
                        format!("representation #{representation_id} solid #{solid_id}: display tessellation failed: {error}")
                    })?;
                    if mesh.faces.is_empty() {
                        failures.push(format!("representation #{representation_id} solid #{solid_id}: display tessellation produced no triangles"));
                        continue;
                    }
                    mixed_face_count += brep.faces().len();
                    mixed_edge_count += brep.edges().len();
                    mixed_vertex_count += brep.vertices().len();
                    mixed_meshes.push(mesh);
                }
                Err(error) => failures.push(format!(
                    "representation #{representation_id} solid #{solid_id} requires mixed planar/analytic import and the exact mixed build failed: {error}"
                )),
            }
        }
        if !failures.is_empty() {
            return Err(failures.join("; "));
        }

        if bodies.is_empty() && mixed_meshes.is_empty() {
            // Sheet/surface representation: only safe to use the whole-file sheet reader
            // when it cannot import some other part by mistake.
            if self.geometry_representation_count == 1 {
                let mut import = crate::import_solidcore_step(&self.path)
                    .map_err(|error| format!("representation #{representation_id} sheet import: {error}"))?;
                import.diagnostics.kind = "representation-sheet".into();
                import.diagnostics.bodies = 1;
                return Ok(import);
            }
            return Err(format!(
                "representation #{representation_id} holds sheet/surface geometry; representation-scoped B-spline/sheet import is not implemented ({})",
                self.geometry_representation_count
            ));
        }

        let mut import = if bodies.is_empty() {
            crate::StepImport {
                mesh: nodecraft_protocol::Mesh::default(),
                diagnostics: crate::StepDiagnostics::default(),
            }
        } else {
            crate::import_multi_bodies(&bodies)
                .map_err(|error| format!("representation #{representation_id}: {error}"))?
        };
        if !mixed_meshes.is_empty() {
            let mut all = Vec::with_capacity(mixed_meshes.len() + 1);
            all.push(import.mesh.clone());
            all.extend(mixed_meshes);
            import.mesh = crate::merge_meshes(all);
            import.diagnostics.kind = if bodies.is_empty() { "representation-mixed".into() } else { "representation-planar-mixed".into() };
            import.diagnostics.breps += skipped_solids.len();
            import.diagnostics.brep_faces += mixed_face_count;
            import.diagnostics.brep_edges += mixed_edge_count;
            import.diagnostics.brep_vertices += mixed_vertex_count;
        } else {
            import.diagnostics.kind = "representation".into();
        }
        import.diagnostics.bodies = bodies.len() + skipped_solids.len();
        import.diagnostics.mesh_vertices = import.mesh.vertices.len();
        import.diagnostics.mesh_triangles = import.mesh.faces.len();
        // Note: no whole-file bbox guard here. A representation's mesh is in the
        // component's own coordinates (possibly a small part inside a large assembly), so
        // STEP file bounds are not a valid reference; the assembly-level bbox check and the
        // per-solid checks in mixed_multi cover corrupt tessellation instead.
        Ok(import)
    }

    /// All geometry item ids of a representation, following untransformed representation
    /// relationships but not occurrence (transformation-carrying) links.
    fn resolve_geometry(&self, representation_id: u32) -> Geometry {
        let mut geometry = Geometry::default();
        let mut visited: BTreeSet<u32> = BTreeSet::new();
        let mut queue: Vec<(u32, usize)> = vec![(representation_id, 0)];
        while let Some((current, depth)) = queue.pop() {
            if !visited.insert(current) || depth > 4 {
                continue;
            }
            let Some(entity) = self.file.entities.get(&current) else { continue };
            let Some(Value::List(items)) = entity.args.get(1) else { continue };
            let mut found_here = false;
            for item in items {
                let Value::Ref(item_id) = item else { continue };
                let Some(item_entity) = self.file.entities.get(item_id) else { continue };
                match item_entity.name.as_str() {
                    "MANIFOLD_SOLID_BREP" | "BREP_WITH_VOIDS" => {
                        geometry.solids.push(*item_id);
                        found_here = true;
                    }
                    "FACETED_BREP" => {
                        geometry.faceted.push(*item_id);
                        found_here = true;
                    }
                    "OPEN_SHELL" | "SHELL_BASED_SURFACE_MODEL" | "CURVE_BOUNDED_SURFACE" | "ADVANCED_FACE" => {
                        geometry.sheets.push(*item_id);
                        found_here = true;
                    }
                    "MAPPED_ITEM" => geometry.mapped_items += 1,
                    _ => {}
                }
            }
            if found_here {
                geometry.source_representations.push(current);
            }
            if let Some(related) = self.relations.get(&current) {
                for next in related {
                    queue.push((*next, depth + 1));
                }
            }
        }
        geometry
    }

    fn collect_representations_and_relations(&mut self) {
        // Occurrence relationships carry a transformation and describe placements, not
        // geometry identity: never treat them as geometry links.
        let mut occurrence_relations: BTreeSet<u32> = BTreeSet::new();
        for (id, entity) in &self.file.entities {
            if entity.name != "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" {
                continue;
            }
            for value in &entity.args {
                collect_refs(value, &mut occurrence_relations);
            }
            occurrence_relations.insert(*id);
        }
        for (id, entity) in &self.file.entities {
            let name = entity.name.as_str();
            let is_rep = name.ends_with("SHAPE_REPRESENTATION")
                && !name.contains("RELATIONSHIP")
                || name == "SHAPE_REPRESENTATION"
                || (name.is_empty() && has_member(entity, "SHAPE_REPRESENTATION"));
            if is_rep && matches!(entity.args.get(1), Some(Value::List(_))) {
                self.representations.insert(*id);
            }
            if occurrence_relations.contains(id) {
                continue;
            }
            let relation_member = name.contains("REPRESENTATION_RELATIONSHIP") || has_member(entity, "REPRESENTATION_RELATIONSHIP");
            if !relation_member {
                continue;
            }
            let transformed = name.contains("WITH_TRANSFORMATION") || has_member(entity, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION");
            if transformed {
                continue;
            }
            let mut refs: Vec<u32> = Vec::new();
            collect_representation_refs(&self.file, entity, &mut refs);
            if refs.len() >= 2 {
                self.relations.entry(refs[0]).or_default().insert(refs[1]);
                self.relations.entry(refs[1]).or_default().insert(refs[0]);
            }
        }
    }
}

fn has_member(entity: &nodecraft_step::Entity, member: &str) -> bool {
    entity.args.iter().any(|value| matches!(value, Value::Typed(name, _) if name == member))
}

fn collect_refs(value: &Value, out: &mut BTreeSet<u32>) {
    match value {
        Value::Ref(id) => {
            out.insert(*id);
        }
        Value::List(values) => {
            for value in values {
                collect_refs(value, out);
            }
        }
        Value::Typed(_, values) => {
            for value in values {
                collect_refs(value, out);
            }
        }
        _ => {}
    }
}

fn collect_representation_refs(file: &StepFile, entity: &nodecraft_step::Entity, out: &mut Vec<u32>) {
    fn is_representation(file: &StepFile, id: u32) -> bool {
        file.entities.get(&id).is_some_and(|entity| {
            let name = entity.name.as_str();
            (name.ends_with("SHAPE_REPRESENTATION") && !name.contains("RELATIONSHIP"))
                || name == "SHAPE_REPRESENTATION"
        })
    }
    fn walk(file: &StepFile, value: &Value, out: &mut Vec<u32>) {
        match value {
            Value::Ref(id) if is_representation(file, *id) => {
                if !out.contains(id) {
                    out.push(*id);
                }
            }
            Value::List(values) => {
                for value in values {
                    walk(file, value, out);
                }
            }
            Value::Typed(_, values) => {
                for value in values {
                    walk(file, value, out);
                }
            }
            _ => {}
        }
    }
    for value in &entity.args {
        walk(file, value, out);
    }
}

/// Imports the geometry of one in-file assembly representation.
///
/// The returned mesh is in the representation's own coordinates (millimetres); the caller
/// positions it with the occurrence's `world_transform_mm`.
///
/// # Errors
///
/// See [`RepresentationContext::import`]; every message names the representation id and a
/// precise reason (mixed/curved solids say `Mx-72`).
pub fn import_representation(path: &str, representation_id: u32) -> Result<StepImport, String> {
    RepresentationContext::load(path)?.import(representation_id)
}

/// Human-readable summary of which representations of a file carry geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationSummary {
    /// Representation entity ids with geometry (solids/facets/sheets).
    pub with_geometry: Vec<u32>,
    /// Representation entity ids without solid/faceted/sheet items.
    pub without_geometry: Vec<u32>,
}

/// Lists which representations in `path` carry geometry, resolving related representations.
///
/// # Errors
///
/// I/O and STEP syntax errors.
pub fn representation_summary(path: &str) -> Result<RepresentationSummary, String> {
    let context = RepresentationContext::load(path)?;
    let mut with_geometry = Vec::new();
    let mut without_geometry = Vec::new();
    for representation in &context.representations {
        if context.has_geometry(*representation) {
            with_geometry.push(*representation);
        } else {
            without_geometry.push(*representation);
        }
    }
    Ok(RepresentationSummary { with_geometry, without_geometry })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use nodecraft_step::write_box;

    use super::*;

    fn repository(relative: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").join(relative);
        path.is_file().then_some(path)
    }

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("stepio-representation-{}-{name}.step", std::process::id()))
    }

    /// Joins two `write_box` files into one STEP file (second file's entity ids are
    /// renumbered) so one file has two representations, each with its own solid.
    fn merged_boxes(first: &str, second: &str, destination: &str) {
        let first_text = std::fs::read_to_string(first).unwrap();
        let second_text = std::fs::read_to_string(second).unwrap();
        let data = |text: &str| -> String {
            let start = text.find("DATA;").unwrap() + "DATA;".len();
            let end = start + text[start..].find("ENDSEC;").unwrap();
            text[start..end].to_string()
        };
        let offset = 1_000_000u32;
        let renumber = |text: &str| -> String {
            let mut out = String::new();
            let mut rest = text;
            while let Some(hash) = rest.find('#') {
                out.push_str(&rest[..=hash]);
                rest = &rest[hash + 1..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(id) = digits.parse::<u32>() {
                    out.push_str(&(id + offset).to_string());
                    rest = &rest[digits.len()..];
                }
            }
            out.push_str(rest);
            out
        };
        let header_end = first_text.find("DATA;").unwrap();
        let mut merged = String::from(&first_text[..header_end]);
        merged.push_str("DATA;");
        merged.push_str(&data(&first_text));
        merged.push_str(&renumber(&data(&second_text)));
        merged.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        std::fs::write(destination, merged).unwrap();
    }

    /// Finds the first representation whose item list contains `solid_id`.
    fn representation_for_solid(file: &StepFile, solid_id: u32) -> u32 {
        for (id, entity) in &file.entities {
            let is_rep = (entity.name.ends_with("SHAPE_REPRESENTATION") && !entity.name.contains("RELATIONSHIP"))
                || entity.name == "SHAPE_REPRESENTATION";
            if !is_rep {
                continue;
            }
            let Some(Value::List(items)) = entity.args.get(1) else { continue };
            if items.iter().any(|item| matches!(item, Value::Ref(item_id) if *item_id == solid_id)) {
                return *id;
            }
        }
        panic!("no representation contains solid #{solid_id}");
    }

    #[test]
    fn imports_only_the_representation_item_solids() {
        let box_a = temp("a");
        let box_b = temp("b");
        let merged = temp("merged");
        write_box(box_a.to_str().unwrap(), 10.0, 10.0, 10.0).unwrap();
        write_box(box_b.to_str().unwrap(), 20.0, 20.0, 20.0).unwrap();
        merged_boxes(box_a.to_str().unwrap(), box_b.to_str().unwrap(), merged.to_str().unwrap());
        let text = std::fs::read_to_string(&merged).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let solids: Vec<u32> = file
            .entities
            .iter()
            .filter(|(_, entity)| entity.name == "MANIFOLD_SOLID_BREP")
            .map(|(id, _)| *id)
            .collect();
        assert_eq!(solids.len(), 2, "two box solids");
        for (solid, expected_half_extent) in [(solids[0], 5.0f32), (solids[1], 10.0f32)] {
            let representation = representation_for_solid(&file, solid);
            let import = import_representation(merged.to_str().unwrap(), representation)
                .unwrap_or_else(|error| panic!("representation #{representation}: {error}"));
            assert!(!import.mesh.faces.is_empty());
            assert_eq!(import.diagnostics.kind, "representation");
            let max = import
                .mesh
                .vertices
                .iter()
                .map(|vertex| vertex[0].abs().max(vertex[1].abs()).max(vertex[2].abs()))
                .fold(0.0f32, f32::max);
            assert!((max - expected_half_extent).abs() < 1e-3, "solid #{solid}: max {max}, expected {expected_half_extent}");
        }
        for path in [&box_a, &box_b, &merged] {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn follows_untransformed_representation_relationships() {
        // cubcylso's DETAIL occurrence names a placement-only representation (#302); the
        // cylinder geometry is in #301, linked by a plain SHAPE_REPRESENTATION_RELATIONSHIP.
        let Some(path) = repository("tests/Sample AP203 Data Files/cubcylso.stp") else { return };
        let path = path.to_str().unwrap();
        let components = step_assemblies::component_instances(path).expect("components");
        let representation = components
            .instances
            .iter()
            .find_map(|instance| match &instance.source {
                step_assemblies::ComponentSource::InFileRepresentation { representation_id } => Some(*representation_id),
                _ => None,
            })
            .expect("in-file representation");
        let summary = representation_summary(path).unwrap();
        assert!(summary.with_geometry.contains(&representation), "geometry relation not resolved: {summary:?}");
        let import = import_representation(path, representation).expect("DETAIL representation");
        assert!(!import.mesh.faces.is_empty(), "cylinder mesh expected");
    }

    #[test]
    fn mixed_representation_imports_all_solids_exactly() {
        let Some(path) = repository("tests/Sample AP203 Data Files/as1_pe.stp") else { return };
        let path = path.to_str().unwrap();
        let components = step_assemblies::component_instances(path).expect("components");
        let representation = components
            .instances
            .iter()
            .find_map(|instance| match &instance.source {
                step_assemblies::ComponentSource::InFileRepresentation { representation_id } => Some(*representation_id),
                _ => None,
            })
            .expect("in-file representation");
        let import = import_representation(path, representation).expect("mixed representation imports");
        assert!(import.diagnostics.kind.contains("mixed"), "{:?}", import.diagnostics.kind);
        assert!(import.diagnostics.bodies >= 1, "{:?}", import.diagnostics);
        assert!(import.diagnostics.breps >= 1, "{:?}", import.diagnostics);
        assert!(!import.mesh.faces.is_empty());
    }

    #[test]
    fn placement_only_representation_reports_no_geometry() {
        let Some(path) = repository("tests/step-parts/MAINBODY.stp") else { return };
        let path = path.to_str().unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        let representation = file
            .entities
            .iter()
            .find(|(_, entity)| {
                let is_rep = (entity.name.ends_with("SHAPE_REPRESENTATION") && !entity.name.contains("RELATIONSHIP"))
                    || entity.name == "SHAPE_REPRESENTATION";
                let Some(Value::List(items)) = entity.args.get(1) else { return false };
                is_rep && items.iter().all(|item| matches!(item, Value::Ref(id) if file.entities.get(id).is_some_and(|entity| entity.name == "AXIS2_PLACEMENT_3D")))
            })
            .map(|(id, _)| *id)
            .expect("placement-only representation");
        let error = import_representation(path, representation).expect_err("no geometry");
        assert!(error.contains("no solid geometry"), "{error}");
    }

    #[test]
    fn unknown_representation_errors() {
        let Some(path) = repository("tests/Sample AP203 Data Files/cubcylso.stp") else { return };
        let error = import_representation(path.to_str().unwrap(), 999_999).expect_err("missing representation");
        assert!(error.contains("does not exist"), "{error}");
    }

    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn assembly_with_related_representation_imports_end_to_end() {
        let Some(path) = repository("tests/Sample AP203 Data Files/cubcylso.stp") else { return };
        let import = crate::import_step(path.to_str().unwrap()).expect("cubcylso assembles");
        assert_eq!(import.diagnostics.kind, "assembly");
        assert!(!import.mesh.faces.is_empty());
        assert!(import.diagnostics.bodies >= 1, "{:?}", import.diagnostics.warnings);
    }

    #[ignore = "legacy stepio expectation; official reader handles this now"]
    #[test]
    fn assembly_imports_mixed_children() {
        let Some(path) = repository("tests/Sample AP203 Data Files/as1_pe.stp") else { return };
        let import = crate::import_step(path.to_str().unwrap()).expect("as1_pe assembles mixed children");
        assert_eq!(import.diagnostics.kind, "assembly");
        assert_eq!(import.diagnostics.bodies, 12, "diagnostics: {:?}", import.diagnostics);
        assert!(!import.mesh.faces.is_empty());
        assert!(import.diagnostics.mesh_triangles > 1000, "{:?}", import.diagnostics);
    }
}
