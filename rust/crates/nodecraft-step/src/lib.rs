//! Minimal STEP (ISO-10303-21 / AP214) reader and writer for the NodeCraft primitives.
//!
//! Modeled on smlibpy's `step_reader.py` / `step.py`: a generic entity parser plus an
//! exact B-rep writer.
//!
//! Written primitives, all centered on the origin in millimeters with `+Z` as the
//! primitive axis (matching smlibpy's `Box`/`Cylinder`/`Cone`/`Sphere`/`Torus`):
//!
//! * [`write_box`] — six planar faces (planar reader round-trip).
//! * [`write_cylinder`] — two planar caps plus one `CYLINDRICAL_SURFACE`, seam edge.
//! * [`write_cone`] — frustum or apex cone via `CONICAL_SURFACE` (both taper
//!   directions; either radius may be zero).
//! * [`write_sphere`] — one `SPHERICAL_SURFACE` with a polar seam loop.
//! * [`write_torus`] — one `TOROIDAL_SURFACE` with two seam circles.
//!
//! The analytic writers are validated against smlibpy's `import_step` in
//! `tests/analytic_primitives.rs` (faces/edges/vertices, volume, area, bounds).
//!
//! Planar B-reps support trimmed faces: each [`PlanarBody`] face has one outer loop and
//! any number of inner loops (holes). The reader groups `FACE_OUTER_BOUND` /
//! `FACE_BOUND` by `ADVANCED_FACE` and honours the bounds' orientation flags; the writer
//! emits the outer bound plus one `FACE_BOUND` per hole in the stored order. Area,
//! volume and the triangle mesh are computed with an ear-clipping triangulation that
//! bridges holes into the outer loop (`triangulate.rs`); holes are never dropped.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

pub mod solidcore_export;
pub use step_brep::brep_to_model;
pub use solidcore_export::write_brep_exact;
mod analytic;
pub mod units;
pub mod style;
pub mod header;
pub mod classify;
pub use classify::{classify_solids, classify_step, ImportPath, SolidTypeReport, StepTypeReport};
pub mod faceted;
pub use faceted::read_faceted_breps;
pub mod multi;
pub use multi::{import_solids, import_solids_partial, StepBody};
pub use header::{read_header, validate_header, StepHeader};
pub use style::{read_color, read_colors, read_layers, read_material_properties, read_material_property_values, read_materials, read_product_name, read_solid_names, read_style_names, set_color, set_layer, set_material, set_material_property, set_product_name};
pub mod brep_export;
pub use brep_export::{step_brep_to_planar_body, write_step_from_step_brep};
pub mod topology;
pub use topology::{read_planar_solids, read_topology, read_topology_file, StepBrep};
pub use units::{convert_step_units, read_length_unit, read_tolerance, read_unit_context, set_header, set_schema, LengthUnit, UnitContext};
pub mod triangulate;

pub use analytic::{write_cone, write_cylinder, write_sphere, write_torus};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Ref(u32),
    Str(String),
    Number(f64),
    Enum(String),
    List(Vec<Value>),
    Typed(String, Vec<Value>),
    Null,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub name: String,
    pub args: Vec<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct StepFile {
    pub entities: BTreeMap<u32, Entity>,
}


impl Entity {
    /// Type names in a complex entity instance (`(A() B() ...)`) plus the simple name.
    pub fn complex_members(&self) -> Vec<&str> {
        let mut members = Vec::new();
        if !self.name.is_empty() { members.push(self.name.as_str()); }
        for arg in &self.args { if let Value::Typed(name, _) = arg { members.push(name.as_str()); } }
        members
    }

    /// Whether this entity or complex instance contains `name`.
    pub fn has_member(&self, name: &str) -> bool {
        self.complex_members().iter().any(|member| *member == name)
    }
}

impl StepFile {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut file = StepFile::default();
        let data = text
            .split_once("DATA;")
            .map(|(_, rest)| rest.split_once("ENDSEC;").map_or(rest, |(data, _)| data))
            .ok_or("STEP file has no DATA section")?;
        let mut parser = Parser { bytes: data.as_bytes(), pos: 0 };
        while let Some(id) = parser.entity_start()? {
            parser.expect(b'=')?;
            let name = parser.identifier()?;
            parser.expect(b'(')?;
            let args = parser.arguments()?;
            parser.expect(b';')?;
            if file.entities.insert(id, Entity { name, args }).is_some() {
                return Err(format!("duplicate STEP entity #{id}"));
            }
        }
        Ok(file)
    }


    /// Whether entity `id` is (or complexly includes) type `name`.
    pub fn is_type(&self, id: u32, name: &str) -> bool {
        self.entities.get(&id).is_some_and(|entity| entity.has_member(name))
    }

    pub fn points(&self) -> impl Iterator<Item = [f64; 3]> + '_ {
        self.entities.values().filter(|entity| entity.name == "CARTESIAN_POINT").filter_map(|entity| {
            let coordinates = entity.args.get(1)?;
            let Value::List(values) = coordinates else { return None };
            let mut point = [0.0; 3];
            for (slot, value) in point.iter_mut().zip(values) {
                let Value::Number(number) = value else { return None };
                *slot = *number;
            }
            Some(point)
        })
    }

    pub fn dangling_refs(&self) -> Vec<u32> {
        fn collect(value: &Value, out: &mut Vec<u32>) {
            match value {
                Value::Ref(id) => out.push(*id),
                Value::List(values) | Value::Typed(_, values) => for value in values { collect(value, out); },
                _ => {}
            }
        }
        let mut refs = Vec::new();
        for entity in self.entities.values() { for value in &entity.args { collect(value, &mut refs); } }
        refs.retain(|id| !self.entities.contains_key(id));
        refs.sort_unstable(); refs.dedup(); refs
    }


    /// Sorted ids of every entity with the given STEP type name.
    pub fn entities_named(&self, name: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = self.entities.iter().filter(|(_, entity)| entity.name == name).map(|(id, _)| *id).collect();
        ids.sort_unstable(); ids
    }


    /// Ordered `(id, entity type)` table for AST/schema consumers.
    pub fn entity_table(&self) -> Vec<(u32, String)> {
        self.entities.iter().map(|(id, entity)| (*id, entity.name.clone())).collect()
    }

    /// Entity-type histogram for AST/table inspection.
    pub fn type_counts(&self) -> std::collections::BTreeMap<String, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for entity in self.entities.values() { *counts.entry(entity.name.clone()).or_default() += 1; }
        counts
    }

    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut points = self.points();
        let first = points.next()?;
        let mut min = first;
        let mut max = first;
        for point in points {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        Some((min, max))
    }
}

/// Millimetres per declared STEP length unit (SI prefixes and inch tokens).
pub(crate) fn unit_scale(text: &str) -> f64 {
    if text.contains("INCH") || text.contains(".INCH.") { 25.4 }
    else if text.contains(".MILLI.,.METRE.") { 1.0 }
    else if text.contains(".CENTI.,.METRE.") { 10.0 }
    else if text.contains(".MICRO.,.METRE.") { 0.001 }
    else if text.contains(".KILO.,.METRE.") { 1_000_000.0 }
    else { 1_000.0 }
}


/// Writes a planar body as a `FACETED_BREP` (polygon loops, no analytic surfaces).
pub fn write_faceted_brep(path: &str, body: &PlanarBody) -> Result<(), String> {
    return solidcore_export::write_faceted_body(path, body);
    #[allow(unreachable_code)]
    {
    if body.points.is_empty() || body.faces.is_empty() { return Err("planar body is empty".into()); }
    body.validate_closed()?;
    let mut writer = Writer::default();
    let context = writer.header();
    let point_ids: Vec<u32> = body.points.iter().map(|[x, y, z]| writer.add(format!("CARTESIAN_POINT('',({},{},{}))", real(*x), real(*y), real(*z)))).collect();
    let mut face_ids = Vec::new();
    for face in 0..body.faces.len() {
        let mut bound_ids = Vec::new();
        for (loop_index, loop_points) in body.loops(face).enumerate() {
            let refs = loop_points.iter().map(|index| format!("#{}", point_ids[*index])).collect::<Vec<_>>().join(",");
            let loop_id = writer.add(format!("POLY_LOOP('',({refs}))"));
            bound_ids.push(writer.add(format!("{}( '',#{loop_id},.T.)", if loop_index == 0 { "FACE_OUTER_BOUND" } else { "FACE_BOUND" })));
        }
        let refs = bound_ids.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(",");
        face_ids.push(writer.add(format!("FACE('',({refs}))")));
    }
    let shell = writer.add(format!("CLOSED_SHELL('',({}))", face_ids.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(",")));
    let brep = writer.add(format!("FACETED_BREP('solid',#{shell})"));
    let representation = writer.add(format!("FACETED_BREP_SHAPE_REPRESENTATION('solid',(#{brep}),#{context})"));
    let definition = writer.product_definition.ok_or("STEP writer has no product definition")?;
    let shape = writer.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
    writer.add(format!("SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"));
    fs::write(path, writer.text()).map_err(|error| format!("{path}: {error}"))
    }
}

/// Bounds of the vertices actually referenced by STEP solids, in millimetres.
///
/// `read_step_bounds` scans every `CARTESIAN_POINT`, including unreferenced construction
/// geometry. This topology-aware variant is the right reference for validating an
/// imported B-rep. It falls back to the raw point bounds when the file has no solids.
pub fn read_topology_bounds(path: &str) -> Result<([f64; 3], [f64; 3]), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let body = topology::read_topology(&file);
    let scale = unit_scale(&text);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut found = false;
    for solid in body.solids.values() {
        let Some(shell) = body.shells.get(&solid.shell) else { continue };
        for face_id in &shell.faces {
            let Some(face) = body.faces.get(face_id) else { continue };
            for (loop_id, _) in &face.bounds {
                let Some(loop_) = body.loops.get(loop_id) else { continue };
                for use_ in &loop_.oriented_edges {
                    let Some(edge) = body.edges.get(&use_.edge) else { continue };
                    for vertex_id in [edge.start, edge.end] {
                        let Some(vertex) = body.vertices.get(&vertex_id) else { continue };
                        found = true;
                        for axis in 0..3 {
                            let value = vertex.point[axis] * scale;
                            min[axis] = min[axis].min(value);
                            max[axis] = max[axis].max(value);
                        }
                    }
                }
            }
        }
    }
    if found { Ok((min, max)) } else { read_step_bounds(path) }
}

pub fn read_step_bounds(path: &str) -> Result<([f64; 3], [f64; 3]), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = unit_scale(&text);
    StepFile::parse(&text)?.bounds().map(|(min, max)| {
        (min.map(|value| value * scale), max.map(|value| value * scale))
    }).ok_or_else(|| "STEP file has no CARTESIAN_POINT entities".into())
}


/// Counts the STEP features the planar/analytic importer can run into.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepFeatures {
    pub advanced_faces: usize,
    pub plane_surfaces: usize,
    pub analytic_surfaces: usize,
    pub bspline_surfaces: usize,
    pub conics: usize,
    pub assemblies: usize,
    pub products: usize,
}

impl StepFeatures {
    pub fn unsupported_summary(&self) -> String {
        let mut parts = Vec::new();
        if self.bspline_surfaces > 0 { parts.push(format!("{} B-spline surface(s)", self.bspline_surfaces)); }
        if self.conics > 0 { parts.push(format!("{} conic(s)", self.conics)); }
        if self.assemblies > 0 { parts.push(format!("{} assembly usage(s)", self.assemblies)); }
        if self.analytic_surfaces > 0 { parts.push(format!("{} curved analytic surface(s)", self.analytic_surfaces)); }
        if parts.is_empty() { format!("{} advanced face(s), {} product(s)", self.advanced_faces, self.products) } else { parts.join(", ") }
    }
}

/// Inspects a STEP file's geometry/product features without importing geometry.
pub fn inspect_features(path: &str) -> Result<StepFeatures, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let count = |name: &str| file.entities.values().filter(|entity| entity.name == name).count();
    Ok(StepFeatures {
        advanced_faces: count("ADVANCED_FACE"),
        plane_surfaces: count("PLANE"),
        analytic_surfaces: count("CYLINDRICAL_SURFACE") + count("CONICAL_SURFACE") + count("SPHERICAL_SURFACE") + count("TOROIDAL_SURFACE"),
        bspline_surfaces: count("B_SPLINE_SURFACE_WITH_KNOTS") + count("B_SPLINE_SURFACE") + count("RATIONAL_B_SPLINE_SURFACE"),
        conics: count("ELLIPSE") + count("HYPERBOLA") + count("PARABOLA"),
        assemblies: count("NEXT_ASSEMBLY_USAGE_OCCURRENCE") + count("ASSEMBLY_COMPONENT_USAGE") + count("PRODUCT_DEFINITION_USAGE"),
        products: count("PRODUCT"),
    })
}

/// A neutral planar B-rep: shared points, one outer loop per face and optional holes.
///
/// `faces[i]` is the outer loop of face `i` and `holes[i]` its inner loops (in STEP
/// bound order). The two vectors are index-aligned when the body is produced by this
/// crate's reader/writer or [`PlanarBody::push_face`]; bodies built by older callers may
/// leave `holes` shorter, which [`PlanarBody::face_holes`] treats as "no holes".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanarBody {
    pub points: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    /// Inner loops (holes) of each face, index-aligned with [`Self::faces`].
    pub holes: Vec<Vec<Vec<usize>>>,
}

impl PlanarBody {
    /// The holes of `face`, or an empty slice for a face without holes.
    pub fn face_holes(&self, face: usize) -> &[Vec<usize>] {
        self.holes.get(face).map_or(&[], Vec::as_slice)
    }

    /// Appends a face with its outer loop and holes, keeping the two vectors aligned.
    pub fn push_face(&mut self, outer: Vec<usize>, holes: Vec<Vec<usize>>) {
        self.faces.push(outer);
        self.holes.push(holes);
    }

    /// Replaces the holes of an existing face (padding `holes` to the face count).
    pub fn set_face_holes(&mut self, face: usize, holes: Vec<Vec<usize>>) -> Result<(), String> {
        if face >= self.faces.len() {
            return Err(format!(
                "face {face} does not exist (body has {} faces)",
                self.faces.len()
            ));
        }
        self.holes.resize_with(self.faces.len(), Vec::new);
        self.holes[face] = holes;
        Ok(())
    }

    /// Outer loop followed by every hole loop of `face`.
    pub fn loops(&self, face: usize) -> impl Iterator<Item = &Vec<usize>> {
        std::iter::once(&self.faces[face]).chain(self.face_holes(face).iter())
    }

    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut points = self.points.iter().copied();
        let first = points.next()?;
        let mut min = first;
        let mut max = first;
        for point in points {
            for axis in 0..3 { min[axis] = min[axis].min(point[axis]); max[axis] = max[axis].max(point[axis]); }
        }
        Some((min, max))
    }

    /// Repairs loop winding for a closed body: outer loops face away from the body
    /// centroid and hole loops face into their face. This is a centroid heuristic; it
    /// is intended for primitive/star-shaped solids, not arbitrary non-convex shells.
    pub fn repair_orientation(&mut self) -> Result<(), String> {
        if self.points.is_empty() || self.faces.is_empty() { return Err("planar body is empty".into()); }
        let mut centroid = [0.0; 3];
        for point in &self.points { for axis in 0..3 { centroid[axis] += point[axis] / self.points.len() as f64; } }
        for face in 0..self.faces.len() {
            let outer = self.faces[face].clone();
            let normal = polygon_normal(&self.points, &outer).ok_or_else(|| format!("face {face} is degenerate"))?;
            let center = polygon_center(&self.points, &outer);
            let outward = [center[0]-centroid[0], center[1]-centroid[1], center[2]-centroid[2]];
            if dot3(normal, outward) < 0.0 { self.faces[face].reverse(); }
            let outer_normal = polygon_normal(&self.points, &self.faces[face]).ok_or_else(|| format!("face {face} is degenerate"))?;
            if face < self.holes.len() {
                for hole in &mut self.holes[face] {
                    if let Some(hole_normal) = polygon_normal(&self.points, hole) {
                        if dot3(hole_normal, outer_normal) > 0.0 { hole.reverse(); }
                    }
                }
            }
        }
        self.validate_closed()
    }


    /// Merges coincident points within `tolerance` and remaps all loops.
    /// Returns the number of removed points.
    pub fn weld_points(&mut self, tolerance: f64) -> usize {
        let tolerance = tolerance.max(1e-12);
        let mut remap = vec![0usize; self.points.len()];
        let mut points: Vec<[f64; 3]> = Vec::new();
        for (index, point) in self.points.iter().enumerate() {
            let mut target = None;
            for (candidate_index, candidate) in points.iter().enumerate() {
                let distance = ((point[0]-candidate[0]).powi(2) + (point[1]-candidate[1]).powi(2) + (point[2]-candidate[2]).powi(2)).sqrt();
                if distance <= tolerance { target = Some(candidate_index); break; }
            }
            remap[index] = target.unwrap_or_else(|| { points.push(*point); points.len()-1 });
        }
        let removed = self.points.len().saturating_sub(points.len());
        let clean = |loop_points: &mut Vec<usize>| {
            for index in loop_points.iter_mut() { *index = remap[*index]; }
            loop_points.dedup();
            if loop_points.len() > 1 && loop_points.first() == loop_points.last() { loop_points.pop(); }
        };
        for face in &mut self.faces { clean(face); }
        for holes in &mut self.holes { for hole in holes { clean(hole); } }
        self.points = points;
        removed
    }

    /// Structural closed-shell check: every undirected edge must be used exactly twice.
    pub fn validate_closed(&self) -> Result<(), String> {
        let mut uses: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for face in 0..self.faces.len() {
            for loop_points in self.loops(face) {
                if loop_points.len() < 3 { return Err(format!("face {face} has a loop with fewer than 3 vertices")); }
                for index in 0..loop_points.len() {
                    let a = loop_points[index];
                    let b = loop_points[(index + 1) % loop_points.len()];
                    if a == b { return Err(format!("face {face} has a degenerate edge")); }
                    let key = if a < b { (a, b) } else { (b, a) };
                    *uses.entry(key).or_default() += 1;
                }
            }
        }
        for (edge, count) in uses {
            if count != 2 { return Err(format!("edge ({}, {}) is used {count} times, expected 2", edge.0, edge.1)); }
        }
        Ok(())
    }

    /// Signed volume of the closed body (holes included), or `NaN` if triangulation
    /// fails; use [`Self::try_volume`] for the error.
    pub fn volume(&self) -> f64 {
        self.try_volume().unwrap_or(f64::NAN)
    }

    /// Signed volume, computed from the triangulated faces (see [`Self::triangulate`]).
    pub fn try_volume(&self) -> Result<f64, String> {
        let mut total = 0.0;
        for triangle in self.triangulate()? {
            let a = self.points[triangle[0]];
            let b = self.points[triangle[1]];
            let c = self.points[triangle[2]];
            total += a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        Ok(total / 6.0)
    }

    /// Total surface area (holes included), or `NaN` if triangulation fails; use
    /// [`Self::try_area`] for the error.
    pub fn area(&self) -> f64 {
        self.try_area().unwrap_or(f64::NAN)
    }

    /// Total surface area from the triangulated faces.
    pub fn try_area(&self) -> Result<f64, String> {
        let mut total = 0.0;
        for triangle in self.triangulate()? {
            let a = self.points[triangle[0]];
            let b = self.points[triangle[1]];
            let c = self.points[triangle[2]];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            total += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        }
        Ok(total)
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalyticSolid {
    Cylinder { radius: f64, height: f64, center: [f64; 3] },
    Cone { radius_base: f64, radius_top: f64, height: f64, center: [f64; 3] },
    Sphere { radius: f64, center: [f64; 3] },
    Torus { major: f64, minor: f64, center: [f64; 3] },
}

fn numeric_args(entity: &Entity) -> Vec<f64> {
    entity.args.iter().filter_map(|value| match value {
        Value::Number(number) => Some(*number),
        _ => None,
    }).collect()
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn finite_point(point: [f64; 3]) -> bool {
    point.iter().all(|value| value.is_finite())
}

/// The placement axis of a surface (`AXIS2_PLACEMENT_3D` axis direction), defaulting
/// to `+Z` for `$`. `None` means the placement/axis reference is malformed.
fn placement_axis(file: &StepFile, surface: &Entity) -> Option<[f64; 3]> {
    let Value::Ref(placement_id) = surface.args.get(1)? else { return None };
    let placement = file.entities.get(placement_id)?;
    if placement.name != "AXIS2_PLACEMENT_3D" { return None; }
    match placement.args.get(2)? {
        Value::Null => Some([0.0, 0.0, 1.0]),
        Value::Ref(axis_id) => {
            let direction = file.entities.get(axis_id)?;
            if direction.name != "DIRECTION" { return None; }
            let Value::List(values) = direction.args.get(1)? else { return None };
            let mut axis = [0.0; 3];
            for (slot, value) in axis.iter_mut().zip(values) {
                let Value::Number(number) = value else { return None };
                *slot = *number;
            }
            Some(axis)
        }
        _ => None,
    }
}

/// Whether `axis` points along `±Z` (the only orientation an `AnalyticSolid`
/// cylinder/cone can represent).
fn axis_is_world_z(axis: [f64; 3]) -> bool {
    if !finite_point(axis) { return false; }
    let length = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if !finite_positive(length) { return false; }
    let lateral = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt() / length;
    lateral <= 1.0e-6 && (axis[2] / length).abs() >= 1.0 - 1.0e-6
}

/// Recognizes the analytic primitive STEP files smlibpy writes.
///
/// Detection requires a genuine single-primitive signature, not just the presence of
/// one analytic surface: exactly one `MANIFOLD_SOLID_BREP`, one closed shell, the
/// primitive's face/surface/circle counts, a `±Z` cylinder/cone axis, positive
/// dimensions, and (for cylinders) both cap circles matching the surface radius. Files
/// that only look like a primitive are rejected with `Ok(None)` so callers can fall
/// through to their general import paths instead of building invalid geometry.
pub fn detect_analytic_solid(path: &str) -> Result<Option<AnalyticSolid>, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let file = StepFile::parse(&text)?;
    let scale = unit_scale(&text);
    let point = |id: u32| -> Option<[f64; 3]> {
        let entity = file.entities.get(&id)?;
        if entity.name != "CARTESIAN_POINT" { return None; }
        let Value::List(coords) = &entity.args[1] else { return None };
        let mut out = [0.0; 3];
        for (slot, value) in out.iter_mut().zip(coords) { let Value::Number(number) = value else { return None }; *slot = *number; }
        Some(out)
    };
    let placement_origin = |surface: &Entity| -> Option<[f64; 3]> {
        let Value::Ref(placement_id) = surface.args.get(1)? else { return None };
        let placement = file.entities.get(placement_id)?;
        let Value::Ref(point_id) = placement.args.get(1)? else { return None };
        point(*point_id)
    };
    let count = |name: &str| file.entities.values().filter(|entity| entity.name == name).count();
    // A sheet or a multi-solid file may contain primitive-shaped surfaces; it is not a
    // single analytic primitive.
    if count("MANIFOLD_SOLID_BREP") != 1 || count("BREP_WITH_VOIDS") != 0 || count("CLOSED_SHELL") != 1 {
        return Ok(None);
    }
    let faces = count("ADVANCED_FACE");
    let planes = count("PLANE");
    let cylinders = count("CYLINDRICAL_SURFACE");
    let cones = count("CONICAL_SURFACE");
    let spheres = count("SPHERICAL_SURFACE");
    let tori = count("TOROIDAL_SURFACE");
    let circle_count = count("CIRCLE");
    let primitive_signature = |kind: &str| match kind {
        "cylinder" => cylinders == 1 && cones == 0 && spheres == 0 && tori == 0 && faces == 3 && planes == 2 && circle_count == 2,
        "cone" => cones == 1 && cylinders == 0 && spheres == 0 && tori == 0 && faces == 3 && planes == 2 && circle_count == 2,
        "sphere" => spheres == 1 && cylinders == 0 && cones == 0 && tori == 0 && faces == 1 && circle_count == 1,
        "torus" => tori == 1 && cylinders == 0 && cones == 0 && spheres == 0 && faces == 1 && circle_count == 2,
        _ => false,
    };
    let circles: Vec<([f64; 3], f64)> = file.entities.values().filter(|entity| entity.name == "CIRCLE").filter_map(|entity| {
        let Value::Ref(placement_id) = entity.args.get(1)? else { return None };
        let placement = file.entities.get(placement_id)?;
        let Value::Ref(point_id) = placement.args.get(1)? else { return None };
        let Value::Number(radius) = entity.args.get(2)? else { return None };
        Some((point(*point_id)?, *radius))
    }).collect();
    let scaled_center = |center: [f64; 3]| center.map(|value| value * scale);
    for entity in file.entities.values() {
        match entity.name.as_str() {
            "CYLINDRICAL_SURFACE" if primitive_signature("cylinder") => {
                let Some(raw_radius) = numeric_args(entity).first().copied() else { return Ok(None) };
                let radius = raw_radius * scale;
                let Some(axis) = placement_axis(&file, entity) else { return Ok(None) };
                if !axis_is_world_z(axis) { return Ok(None); }
                if circles.len() != 2
                    || circles.iter().any(|(_, circle_radius)| {
                        !finite_positive(*circle_radius)
                            || (*circle_radius - raw_radius).abs() > raw_radius.abs() * 1.0e-6 + 1.0e-9
                    })
                {
                    return Ok(None);
                }
                let Some(origin) = placement_origin(entity) else { return Ok(None) };
                let zmin = circles.iter().map(|(circle_point, _)| circle_point[2]).fold(f64::INFINITY, f64::min);
                let zmax = circles.iter().map(|(circle_point, _)| circle_point[2]).fold(f64::NEG_INFINITY, f64::max);
                let height = (zmax - zmin) * scale;
                if !finite_positive(radius) || !finite_positive(height) { return Ok(None); }
                let center = scaled_center([origin[0], origin[1], (zmin + zmax) / 2.0]);
                if !finite_point(center) { return Ok(None); }
                return Ok(Some(AnalyticSolid::Cylinder { radius, height, center }));
            }
            "CONICAL_SURFACE" if primitive_signature("cone") => {
                let Some(raw_radius) = numeric_args(entity).first().copied() else { return Ok(None) };
                let Some(axis) = placement_axis(&file, entity) else { return Ok(None) };
                if !axis_is_world_z(axis) || !finite_positive(raw_radius * scale) { return Ok(None); }
                if circles.len() != 2
                    || circles.iter().any(|(_, circle_radius)| !finite_non_negative(*circle_radius))
                {
                    return Ok(None);
                }
                let Some(origin) = placement_origin(entity) else { return Ok(None) };
                let mut ordered = circles.clone();
                ordered.sort_by(|a, b| a.0[2].partial_cmp(&b.0[2]).unwrap_or(std::cmp::Ordering::Equal));
                let radius_base = ordered[0].1 * scale;
                let radius_top = ordered[1].1 * scale;
                let height = (ordered[1].0[2] - ordered[0].0[2]) * scale;
                if !finite_non_negative(radius_base) || !finite_non_negative(radius_top)
                    || (radius_base <= 0.0 && radius_top <= 0.0)
                    || !finite_positive(height)
                {
                    return Ok(None);
                }
                let center = scaled_center([origin[0], origin[1], (ordered[0].0[2] + ordered[1].0[2]) / 2.0]);
                if !finite_point(center) { return Ok(None); }
                return Ok(Some(AnalyticSolid::Cone { radius_base, radius_top, height, center }));
            }
            "SPHERICAL_SURFACE" if primitive_signature("sphere") => {
                let Some(raw_radius) = numeric_args(entity).first().copied() else { return Ok(None) };
                let radius = raw_radius * scale;
                let Some(origin) = placement_origin(entity) else { return Ok(None) };
                let center = scaled_center(origin);
                if !finite_positive(radius) || !finite_point(center) { return Ok(None); }
                return Ok(Some(AnalyticSolid::Sphere { radius, center }));
            }
            "TOROIDAL_SURFACE" if primitive_signature("torus") => {
                let values = numeric_args(entity);
                if values.len() < 2 { return Ok(None); }
                let major = values[0] * scale;
                let minor = values[1] * scale;
                let Some(origin) = placement_origin(entity) else { return Ok(None) };
                let center = scaled_center(origin);
                if !finite_positive(major) || !finite_positive(minor) || !finite_point(center) {
                    return Ok(None);
                }
                return Ok(Some(AnalyticSolid::Torus { major, minor, center }));
            }
            _ => {}
        }
    }
    Ok(None)
}


/// Parses planar STEP faces (VERTEX_POINT / EDGE_CURVE / EDGE_LOOP / ADVANCED_FACE).
pub fn read_planar_body(path: &str) -> Result<PlanarBody, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = unit_scale(&text);
    let file = StepFile::parse(&text)?;
    let dangling = file.dangling_refs();
    if !dangling.is_empty() { return Err(format!("STEP file has dangling entity references: {dangling:?}")); }
    let mut body = planar_body_from_file(&file)?;
    for point in &mut body.points { for value in point.iter_mut() { *value *= scale; } }
    Ok(body)
}

/// A `FACE_OUTER_BOUND` or `FACE_BOUND` of a planar face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlanarBoundKind {
    Outer,
    Inner,
}

#[derive(Debug, Clone, Copy)]
struct PlanarBound {
    loop_id: u32,
    kind: PlanarBoundKind,
    /// The bound's STEP orientation flag: `false` means the loop is reversed.
    orientation: bool,
}

fn planar_body_from_file(file: &StepFile) -> Result<PlanarBody, String> {
    let mut body = PlanarBody::default();
    let mut point_index = BTreeMap::new();
    let mut vertex_index = BTreeMap::new();
    let mut edge_vertices = BTreeMap::new();
    let mut oriented_edges = BTreeMap::new();
    let mut loops = BTreeMap::new();
    let mut bounds: BTreeMap<u32, PlanarBound> = BTreeMap::new();
    for (id, entity) in &file.entities {
        if entity.name != "CARTESIAN_POINT" { continue; }
        let Value::List(coords) = &entity.args[1] else { continue };
        let mut point = [0.0; 3];
        for (slot, value) in point.iter_mut().zip(coords) {
            let Value::Number(number) = value else { continue };
            *slot = *number;
        }
        point_index.insert(*id, body.points.len());
        body.points.push(point);
    }
    for (id, entity) in &file.entities {
        if entity.name != "VERTEX_POINT" { continue; }
        if let Some(Value::Ref(point)) = entity.args.get(1) {
            if let Some(index) = point_index.get(point) { vertex_index.insert(*id, *index); }
        }
    }
    for (id, entity) in &file.entities {
        match entity.name.as_str() {
            "CARTESIAN_POINT" | "VERTEX_POINT" => {}
            "EDGE_CURVE" => {
                if let (Some(Value::Ref(a)), Some(Value::Ref(b))) = (entity.args.get(1), entity.args.get(2)) {
                    if let (Some(a), Some(b)) = (vertex_index.get(a), vertex_index.get(b)) { edge_vertices.insert(*id, (*a, *b)); }
                }
            }
            "ORIENTED_EDGE" => {
                if let Some(Value::Ref(edge)) = entity.args.get(3) {
                    let forward = entity.args.get(4) == Some(&Value::Enum("T".into()));
                    oriented_edges.insert(*id, (*edge, forward));
                }
            }
            "EDGE_LOOP" => {
                let Some(Value::List(uses)) = entity.args.get(1) else { continue };
                let mut points = Vec::new();
                for value in uses {
                    let Value::Ref(use_id) = value else { continue };
                    let Some((edge, forward)) = oriented_edges.get(use_id) else { continue };
                    let Some((a, b)) = edge_vertices.get(edge) else { continue };
                    points.push(if *forward { *a } else { *b });
                }
                loops.insert(*id, points);
            }
            "FACE_OUTER_BOUND" | "FACE_BOUND" => {
                let Some(Value::Ref(loop_id)) = entity.args.get(1) else { continue };
                let orientation = match entity.args.get(2) {
                    Some(Value::Enum(flag)) if flag == "T" => true,
                    Some(Value::Enum(flag)) if flag == "F" => false,
                    None | Some(Value::Null) => true,
                    Some(other) => {
                        return Err(format!(
                            "{} orientation is not .T./.F.: {other:?}",
                            entity.name
                        ));
                    }
                };
                let kind = if entity.name == "FACE_OUTER_BOUND" {
                    PlanarBoundKind::Outer
                } else {
                    PlanarBoundKind::Inner
                };
                bounds.insert(*id, PlanarBound { loop_id: *loop_id, kind, orientation });
            }
            _ => {}
        }
    }
    // Second pass: edge loops can reference oriented edges defined later in the file.
    for (id, entity) in &file.entities {
        if entity.name != "EDGE_LOOP" { continue; }
        let Some(Value::List(uses)) = entity.args.get(1) else { continue };
        let mut points = Vec::new();
        for value in uses {
            let Value::Ref(use_id) = value else { continue };
            let Some((edge, forward)) = oriented_edges.get(use_id) else { continue };
            let Some((a, b)) = edge_vertices.get(edge) else { continue };
            points.push(if *forward { *a } else { *b });
        }
        loops.insert(*id, points);
    }
    for entity in file.entities.values() {
        if entity.name != "ADVANCED_FACE" { continue; }
        let Some(Value::List(bound_refs)) = entity.args.get(1) else { continue };
        let mut outer: Option<Vec<usize>> = None;
        let mut holes: Vec<Vec<usize>> = Vec::new();
        let mut saw_outer = false;
        for bound_ref in bound_refs {
            let Value::Ref(bound_id) = bound_ref else { continue };
            let Some(bound) = bounds.get(bound_id) else { continue };
            let Some(loop_points) = loops.get(&bound.loop_id) else { continue };
            let mut points = loop_points.clone();
            // STEP bound orientation: .F. means the loop is used reversed.
            if !bound.orientation { points.reverse(); }
            match bound.kind {
                PlanarBoundKind::Outer => {
                    if saw_outer {
                        return Err("ADVANCED_FACE has more than one FACE_OUTER_BOUND".into());
                    }
                    saw_outer = true;
                    if points.len() < 3 {
                        // A non-polygonal outer bound (e.g. a single closed curve);
                        // the planar reader does not support this face.
                        continue;
                    }
                    outer = Some(points);
                }
                PlanarBoundKind::Inner => {
                    holes.push(points);
                }
            }
        }
        if outer.is_none() && !holes.is_empty() {
            // STEP permits FACE_BOUND for the outer loop; when no FACE_OUTER_BOUND is
            // present the first bound is the outer loop and the rest are holes.
            outer = Some(holes.remove(0));
        }
        if let Some(outer) = outer {
            if outer.len() < 3 {
                // Non-polygonal outer bound (for example a single closed curve).
                continue;
            }
            for hole in &holes {
                if hole.len() < 3 {
                    return Err("FACE_BOUND inner loop has fewer than three vertices".into());
                }
            }
            body.push_face(outer, holes);
        }
    }
    if body.faces.is_empty() { return Err("STEP file has no planar face loops".into()); }
    let declared = file.entities.values().filter(|entity| entity.name == "ADVANCED_FACE").count();
    if body.faces.len() < declared {
        return Err(format!(
            "STEP file has {} unsupported curved or non-polygonal face(s); refusing a partial planar import",
            declared - body.faces.len()
        ));
    }
    Ok(body)
}

/// Writes a neutral planar body (outer loops + holes) as AP214 planar B-rep faces.
///
/// Every face becomes one `ADVANCED_FACE` with a `FACE_OUTER_BOUND` and one
/// `FACE_BOUND` per hole, in the stored loop order. Loop orientation is preserved:
/// the writer never reverses or reorders a loop, so an inward-facing hole loop stays
/// inward-facing in the file (bind the `.T.` orientation flag to keep the handedness).
///
/// # Errors
///
/// Empty/degenerate loops, out-of-range point indices, and faces whose loops are not
/// coplanar are rejected; nothing is silently dropped.
pub fn write_planar_body(path: &str, body: &PlanarBody) -> Result<(), String> {
    solidcore_export::write_planar_body(path, body)
}

/// Writes a planar sheet as an `OPEN_SHELL` (not a closed solid).
pub fn write_open_shell(path: &str, body: &PlanarBody) -> Result<(), String> {
    write_planar_shell(path, body, true)
}

fn write_planar_shell(path: &str, body: &PlanarBody, open: bool) -> Result<(), String> {
    if body.points.is_empty() || body.faces.is_empty() { return Err("planar body is empty".into()); }
    let repaired;
    let body = if !open && body.volume().is_finite() && body.volume() < 0.0 {
        repaired = { let mut copy = body.clone(); copy.repair_orientation()?; copy };
        &repaired
    } else { body };
    let normals = planar_face_normals(body)?;
    let mut w = Writer::default();
    let ctx = w.header();
    let point_ids: Vec<u32> = body.points.iter().map(|[x, y, z]| w.add(format!("CARTESIAN_POINT('',({},{},{}))", real(*x), real(*y), real(*z)))).collect();
    let vertex_ids: Vec<u32> = point_ids.iter().map(|id| w.add(format!("VERTEX_POINT('',#{id})"))).collect();
    let mut edges: BTreeMap<(usize, usize), (u32, bool)> = BTreeMap::new();
    for face_index in 0..body.faces.len() {
        for loop_points in body.loops(face_index) {
            for index in 0..loop_points.len() {
                let a = loop_points[index];
                let b = loop_points[(index + 1) % loop_points.len()];
                let key = if a < b { (a, b) } else { (b, a) };
                if let std::collections::btree_map::Entry::Vacant(entry) = edges.entry(key) {
                    let pa = body.points[a];
                    let pb = body.points[b];
                    let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if length <= 0.0 { return Err("degenerate planar edge".into()); }
                    let direction = w.add(format!("DIRECTION('',({},{},{}))", real(d[0] / length), real(d[1] / length), real(d[2] / length)));
                    let vector = w.add(format!("VECTOR('',#{direction},{})", real(length)));
                    let line = w.add(format!("LINE('',#{},#{vector})", point_ids[a]));
                    entry.insert((w.add(format!("EDGE_CURVE('',#{},#{},#{line},.T.)", vertex_ids[a], vertex_ids[b])), a < b));
                }
            }
        }
    }
    let mut face_display_ids = Vec::new();
    for face_index in 0..body.faces.len() {
        let normal = normals[face_index];
        let mut bounds = Vec::new();
        let mut outer_bound = true;
        for loop_points in body.loops(face_index) {
            let mut oriented = Vec::new();
            for index in 0..loop_points.len() {
                let a = loop_points[index];
                let b = loop_points[(index + 1) % loop_points.len()];
                let key = if a < b { (a, b) } else { (b, a) };
                let (edge_id, creation_starts_at_min) = *edges.get(&key).ok_or("missing planar edge")?;
                let current_starts_at_min = a < b;
                let forward = current_starts_at_min == creation_starts_at_min;
                let flag = if forward { "T" } else { "F" };
                oriented.push(w.add(format!("ORIENTED_EDGE('',*,*,#{edge_id},.{flag}.)")));
            }
            let loop_id = w.add(format!("EDGE_LOOP('',({}))", oriented.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(",")));
            let kind = if outer_bound { "FACE_OUTER_BOUND" } else { "FACE_BOUND" };
            let bound = w.add(format!("{kind}('',#{loop_id},.T.)"));
            bounds.push(format!("#{bound}"));
            outer_bound = false;
        }
        let axis = w.add(format!("DIRECTION('',({},{},{}))", real(normal[0]), real(normal[1]), real(normal[2])));
        let candidate = if normal[0].abs() > 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let reference = unit(orthogonalize(candidate, normal), "plane reference direction")?;
        let reference = w.add(format!("DIRECTION('',({},{},{}))", real(reference[0]), real(reference[1]), real(reference[2])));
        let placement = w.add(format!("AXIS2_PLACEMENT_3D('',#{},#{axis},#{reference})", point_ids[body.faces[face_index][0]]));
        let plane = w.add(format!("PLANE('',#{placement})"));
        face_display_ids.push(w.add(format!("ADVANCED_FACE('',({}),#{plane},.T.)", bounds.join(","))));
    }
    let shell_faces = face_display_ids.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(",");
    let text = if open {
        let shell = w.add(format!("OPEN_SHELL('',({shell_faces}))"));
        let model = w.add(format!("SHELL_BASED_SURFACE_MODEL('sheet',(#{shell}))"));
        let representation = w.add(format!("MANIFOLD_SURFACE_SHAPE_REPRESENTATION('sheet',(#{model}),#{ctx})"));
        let definition = w.product_definition.unwrap_or(representation);
        let shape = w.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
        w.add(format!("SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"));
        w.text()
    } else {
        let shell = w.add(format!("CLOSED_SHELL('',({shell_faces}))"));
        let solid = w.add(format!("MANIFOLD_SOLID_BREP('solid',#{shell})"));
        w.finish_solid(ctx, "solid", solid)
    };
    fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}


fn polygon_center(points: &[[f64; 3]], loop_points: &[usize]) -> [f64; 3] {
    let mut center = [0.0; 3];
    for index in loop_points { for axis in 0..3 { center[axis] += points[*index][axis] / loop_points.len() as f64; } }
    center
}

fn polygon_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Option<[f64; 3]> {
    let origin = points[loop_points[0]];
    let mut normal = [0.0; 3];
    for index in 1..loop_points.len() - 1 {
        let a = sub3(points[loop_points[index]], origin);
        let b = sub3(points[loop_points[index + 1]], origin);
        normal = add3(normal, cross3(a, b));
    }
    unit(normal, "polygon normal").ok()
}

/// Unit outward-ish normal of every planar face, validating loops and coplanarity.
///
/// The normal follows the outer loop's winding (Newell), so `ADVANCED_FACE(..., .T.)`
/// keeps the loop handedness. Holes must lie in the same plane; their winding is not
/// required to be opposite here (the writer preserves it and the file states it).
fn planar_face_normals(body: &PlanarBody) -> Result<Vec<[f64; 3]>, String> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in &body.points {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let diagonal = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    if !(diagonal > 0.0 && diagonal.is_finite()) { return Err("planar body has zero extent".into()); }
    let tolerance = diagonal * 1e-6;
    let mut normals = Vec::with_capacity(body.faces.len());
    for face_index in 0..body.faces.len() {
        let outer = &body.faces[face_index];
        if outer.len() < 3 {
            return Err(format!("face {face_index} outer loop has fewer than three vertices"));
        }
        // Validate every loop before touching point data (no index panics).
        for (loop_index, loop_points) in body.loops(face_index).enumerate() {
            if loop_points.len() < 3 {
                return Err(format!(
                    "face {face_index} {} loop has fewer than three vertices",
                    if loop_index == 0 { "outer" } else { "inner" }
                ));
            }
            for (position, &index) in loop_points.iter().enumerate() {
                if index >= body.points.len() {
                    return Err(format!("face {face_index} references missing point {index}"));
                }
                if index == loop_points[(position + 1) % loop_points.len()] {
                    return Err(format!(
                        "face {face_index} loop {loop_index} repeats point {index} consecutively"
                    ));
                }
            }
        }
        let normal = newell_normal(&body.points, outer)
            .ok_or_else(|| format!("face {face_index} outer loop is degenerate"))?;
        let origin = body.points[outer[0]];
        for (loop_index, loop_points) in body.loops(face_index).enumerate() {
            for &index in loop_points {
                let relative = sub3(body.points[index], origin);
                let offset = dot3(relative, normal);
                if offset.abs() > tolerance {
                    return Err(format!(
                        "face {face_index} loop {loop_index} is not planar: point {index} is {offset} off the outer plane"
                    ));
                }
            }
            if loop_index > 0 && newell_normal(&body.points, loop_points).is_none() {
                return Err(format!("face {face_index} inner loop {loop_index} is degenerate"));
            }
        }
        normals.push(normal);
    }
    Ok(normals)
}

/// Unit normal of a closed planar loop (Newell, translation-stabilized).
pub(crate) fn newell_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Option<[f64; 3]> {
    let origin = points[*loop_points.first()?];
    let mut normal = [0.0f64; 3];
    for position in 0..loop_points.len() {
        let a = sub3(points[loop_points[position]], origin);
        let b = sub3(points[loop_points[(position + 1) % loop_points.len()]], origin);
        normal = add3(normal, cross3(a, b));
    }
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !length.is_finite() || length <= 0.0 { return None; }
    Some([normal[0] / length, normal[1] / length, normal[2] / length])
}

pub(crate) fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub(crate) fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub(crate) fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}




/// Rotates all points and directions and translates points in an existing STEP file.
pub fn transform_step_file(path: &str, rotation: [f64; 3], offset: [f64; 3]) -> Result<(), String> {
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let rotate = |p: [f64; 3]| {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let (x, y, z) = (p[0], p[1], p[2]);
        let (y, z) = (y * cx - z * sx, y * sx + z * cx);
        let (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        (x * cz - y * sz, x * sz + y * cz, z)
    };
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = unit_scale(&text);
    let offset = [offset[0] / scale, offset[1] / scale, offset[2] / scale];
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut replacement = String::new();
        for prefix in ["CARTESIAN_POINT('',(", "DIRECTION('',("] {
            let Some(start) = line.find(prefix) else { continue };
            let Some(end) = line[start..].find("))") else { continue };
            let numbers = &line[start + prefix.len()..start + end];
            let values: Vec<f64> = numbers.split(',').filter_map(|value| value.trim().parse().ok()).collect();
            if values.len() != 3 { continue; }
            let rotated = rotate([values[0], values[1], values[2]]);
            let values = if prefix.starts_with("CARTESIAN") {
                [rotated.0 + offset[0], rotated.1 + offset[1], rotated.2 + offset[2]]
            } else {
                [rotated.0, rotated.1, rotated.2]
            };
            replacement.push_str(&line[..start + prefix.len()]);
            replacement.push_str(&format!("{},{},{})", real(values[0]), real(values[1]), real(values[2])));
            replacement.push_str(&line[start + end + 1..]);
            break;
        }
        if replacement.is_empty() { out.push_str(line); } else { out.push_str(&replacement); }
        out.push('\n');
    }
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}

/// Translates every `CARTESIAN_POINT` in an existing STEP file by `offset`.
pub fn translate_points_in_file(path: &str, offset: [f64; 3]) -> Result<(), String> {
    if offset == [0.0, 0.0, 0.0] { return Ok(()); }
    let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
    let scale = unit_scale(&text);
    let offset = [offset[0] / scale, offset[1] / scale, offset[2] / scale];
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if let Some(start) = line.find("CARTESIAN_POINT('',(") {
            let prefix = &line[..start + "CARTESIAN_POINT('',(".len()];
            if let Some(end) = line[start..].find("))") {
                let numbers = &line[start + "CARTESIAN_POINT('',(".len()..start + end];
                let values: Vec<f64> = numbers.split(',').filter_map(|value| value.trim().parse().ok()).collect();
                if values.len() == 3 {
                    out.push_str(prefix);
                    out.push_str(&format!("{},{},{})", real(values[0] + offset[0]), real(values[1] + offset[1]), real(values[2] + offset[2])));
                    out.push_str(&line[start + end + 1..]);
                    out.push('\n');
                    continue;
                }
            }
        }
        out.push_str(line); out.push('\n');
    }
    fs::write(path, out).map_err(|error| format!("{path}: {error}"))
}


/// Writes a centered box with a centered rectangular void as `BREP_WITH_VOIDS`.
pub fn write_box_with_void(path: &str, outer: [f64; 3], inner: [f64; 3]) -> Result<(), String> {
    if outer.iter().any(|value| !(*value > 0.0)) || inner.iter().any(|value| !(*value > 0.0)) {
        return Err("box dimensions must be positive".into());
    }
    if (0..3).any(|axis| inner[axis] >= outer[axis]) { return Err("void must be smaller than the outer box".into()); }
    let mut w = Writer::default();
    let ctx = w.header();
    let outer_shell = add_box_shell(&mut w, outer, false)?;
    let inner_shell = add_box_shell(&mut w, inner, true)?;
    let solid = w.add(format!("BREP_WITH_VOIDS('solid',#{outer_shell},(#{inner_shell}))"));
    fs::write(path, w.finish_solid(ctx, "solid", solid)).map_err(|error| format!("{path}: {error}"))
}

fn add_box_shell(w: &mut Writer, dimensions: [f64; 3], inward: bool) -> Result<u32, String> {
    let (x, y, z) = (dimensions[0] / 2.0, dimensions[1] / 2.0, dimensions[2] / 2.0);
    let points = [[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]];
    let mut faces = vec![vec![0,3,2,1], vec![4,5,6,7], vec![0,1,5,4], vec![1,2,6,5], vec![2,3,7,6], vec![3,0,4,7]];
    if inward { for face in &mut faces { face.reverse(); } }
    let point_ids: Vec<u32> = points.iter().map(|point| w.point(*point)).collect();
    let vertex_ids: Vec<u32> = point_ids.iter().map(|id| w.add(format!("VERTEX_POINT('',#{id})"))).collect();
    let mut edges: BTreeMap<(usize, usize), (u32, bool)> = BTreeMap::new();
    for face in &faces {
        for index in 0..face.len() {
            let a = face[index]; let b = face[(index + 1) % face.len()];
            let key = if a < b { (a, b) } else { (b, a) };
            if let std::collections::btree_map::Entry::Vacant(entry) = edges.entry(key) {
                let pa = points[a]; let pb = points[b];
                let direction = unit([pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]], "box edge")?;
                let direction = w.direction(direction)?;
                let vector = w.add(format!("VECTOR('',#{direction},{})", real(((pb[0]-pa[0]).powi(2)+(pb[1]-pa[1]).powi(2)+(pb[2]-pa[2]).powi(2)).sqrt())));
                let line = w.add(format!("LINE('',#{},#{vector})", point_ids[a]));
                entry.insert((w.add(format!("EDGE_CURVE('',#{},#{},#{line},.T.)", vertex_ids[a], vertex_ids[b])), a < b));
            }
        }
    }
    let mut face_ids = Vec::new();
    for face in &faces {
        let mut oriented = Vec::new();
        for index in 0..face.len() {
            let a = face[index]; let b = face[(index + 1) % face.len()];
            let key = if a < b { (a, b) } else { (b, a) };
            let (edge, creation_starts_min) = edges[&key];
            let flag = if (a < b) == creation_starts_min { "T" } else { "F" };
            oriented.push(w.add(format!("ORIENTED_EDGE('',*,*,#{edge},.{flag}.)")));
        }
        let loop_id = w.add(format!("EDGE_LOOP('',({}))", oriented.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(",")));
        let bound = w.add(format!("FACE_OUTER_BOUND('',#{loop_id},.T.)"));
        let normal = unit(face_normal(&points, face), "box face normal")?;
        let axis = w.direction(normal)?;
        let candidate = if normal[0].abs() > 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let reference = w.direction(unit(orthogonalize(candidate, normal), "box reference")?)?;
        let placement = w.add(format!("AXIS2_PLACEMENT_3D('',#{},#{axis},#{reference})", point_ids[face[0]]));
        let plane = w.add(format!("PLANE('',#{placement})"));
        face_ids.push(w.add(format!("ADVANCED_FACE('',(#{bound}),#{plane},.T.)")));
    }
    Ok(w.add(format!("CLOSED_SHELL('',({}))", face_ids.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(","))))
}

fn face_normal(points: &[[f64; 3]], face: &[usize]) -> [f64; 3] {
    let origin = points[face[0]];
    let mut normal = [0.0; 3];
    for index in 1..face.len() - 1 {
        let a = sub3(points[face[index]], origin);
        let b = sub3(points[face[index + 1]], origin);
        normal = add3(normal, cross3(a, b));
    }
    normal
}

/// Writes a centered box as AP214 `MANIFOLD_SOLID_BREP` (planar faces only).
pub fn write_box(path: &str, length: f64, width: f64, height: f64) -> Result<(), String> {
    if !(length > 0.0 && width > 0.0 && height > 0.0) { return Err("box dimensions must be positive".into()); }
    let (x, y, z) = (length / 2.0, width / 2.0, height / 2.0);
    let points = vec![
        [-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
        [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z],
    ];
    let faces = vec![
        vec![0,3,2,1], vec![4,5,6,7], vec![0,1,5,4],
        vec![1,2,6,5], vec![2,3,7,6], vec![3,0,4,7],
    ];
    write_planar_body(path, &PlanarBody { points, faces, ..PlanarBody::default() })
}

/// A STEP REAL literal: always contains a decimal point, uses STEP `E` exponents.
pub(crate) fn real(value: f64) -> String {
    if value == 0.0 {
        return "0.".into();
    }
    let text = format!("{value:?}");
    match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => {
            let mantissa = if mantissa.contains('.') {
                mantissa.to_string()
            } else {
                format!("{mantissa}.")
            };
            format!("{mantissa}E{exponent}")
        }
        None => {
            if text.contains('.') {
                text
            } else {
                format!("{text}.")
            }
        }
    }
}

/// STEP entity writer: sequential `#id=ENTITY(...);` lines plus the AP214 header.
#[derive(Default)]
pub(crate) struct Writer {
    next: u32,
    lines: Vec<String>,
    product_definition: Option<u32>,
}

impl Writer {
    pub(crate) fn add(&mut self, body: impl AsRef<str>) -> u32 {
        self.next += 1;
        self.lines.push(format!("#{}={};", self.next, body.as_ref()));
        self.next
    }

    /// Writes the product/context header entities and returns the geometric
    /// representation context id.
    pub(crate) fn header(&mut self) -> u32 {
        let app = self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')");
        let pctx = self.add(format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
        let prod = self.add(format!("PRODUCT('solid','solid','',(#{pctx}))"));
        self.add(format!("PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))"));
        let form = self.add(format!("PRODUCT_DEFINITION_FORMATION('','',#{prod})"));
        let dctx = self.add(format!("PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"));
        let pdef = self.add(format!("PRODUCT_DEFINITION('design','',#{form},#{dctx})"));
        self.product_definition = Some(pdef);
        let unit = self.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
        let angle = self.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
        let solid_angle = self.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())");
        let uncertainty = self.add(format!(
            "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{unit},'distance_accuracy_value','')"
        ));
        self.add(format!(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) \
             GLOBAL_UNIT_ASSIGNED_CONTEXT((#{unit},#{angle},#{solid_angle})) REPRESENTATION_CONTEXT('',''))"
        ))
    }

    /// Adds `ADVANCED_BREP_SHAPE_REPRESENTATION` + `PRODUCT_DEFINITION_SHAPE` for one
    /// solid and returns the finished ISO-10303-21 text.
    pub(crate) fn finish_solid(mut self, context: u32, name: &str, solid: u32) -> String {
        let representation = self.add(format!(
            "ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{solid}),#{context})"
        ));
        let definition = self.product_definition.unwrap_or(representation);
        let shape = self.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
        self.add(format!("SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"));
        self.text()
    }

    fn text(self) -> String {
        let mut out = String::new();
        let _ = write!(out, "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION((''),'2;1');\nFILE_NAME('','',(''),(''),'nodecraft-step','nodecraft-step','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n");
        for line in self.lines { out.push_str(&line); out.push('\n'); }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }

    // -- geometry -----------------------------------------------------------

    /// `CARTESIAN_POINT('',(x,y,z))`.
    pub(crate) fn point(&mut self, point: [f64; 3]) -> u32 {
        self.add(format!(
            "CARTESIAN_POINT('',({},{},{}))",
            real(point[0]),
            real(point[1]),
            real(point[2])
        ))
    }

    /// `DIRECTION`, normalized. Fails on a zero/non-finite vector.
    pub(crate) fn direction(&mut self, direction: [f64; 3]) -> Result<u32, String> {
        let unit = unit(direction, "direction")?;
        Ok(self.add(format!(
            "DIRECTION('',({},{},{}))",
            real(unit[0]),
            real(unit[1]),
            real(unit[2])
        )))
    }

    /// `AXIS2_PLACEMENT_3D` with `z` as the axis and `x` orthogonalized against it.
    pub(crate) fn axis2(
        &mut self,
        origin: [f64; 3],
        z: [f64; 3],
        x: [f64; 3],
    ) -> Result<u32, String> {
        let z = unit(z, "placement axis")?;
        let x = unit(orthogonalize(x, z), "placement reference direction")?;
        let point = self.point(origin);
        let axis = self.direction(z)?;
        let reference = self.direction(x)?;
        Ok(self.add(format!(
            "AXIS2_PLACEMENT_3D('',#{point},#{axis},#{reference})"
        )))
    }

    /// `CIRCLE` in the placement plane, with the placement axis as the circle normal.
    pub(crate) fn circle(
        &mut self,
        origin: [f64; 3],
        z: [f64; 3],
        x: [f64; 3],
        radius: f64,
    ) -> Result<u32, String> {
        let placement = self.axis2(origin, z, x)?;
        Ok(self.add(format!("CIRCLE('',#{placement},{})", real(radius))))
    }

    /// `LINE` through `start` with direction `delta` (the vector length is `|delta|`).
    pub(crate) fn line(&mut self, start: [f64; 3], delta: [f64; 3]) -> Result<u32, String> {
        let length = norm(delta);
        let direction = unit(delta, "line direction")?;
        let vector_direction = self.direction(direction)?;
        let vector = self.add(format!(
            "VECTOR('',#{vector_direction},{})",
            real(length)
        ));
        let point = self.point(start);
        Ok(self.add(format!("LINE('',#{point},#{vector})")))
    }

    /// `PLANE` with `normal` as the placement axis.
    pub(crate) fn plane(
        &mut self,
        origin: [f64; 3],
        normal: [f64; 3],
        reference: [f64; 3],
    ) -> Result<u32, String> {
        let placement = self.axis2(origin, normal, reference)?;
        Ok(self.add(format!("PLANE('',#{placement})")))
    }

    /// `CYLINDRICAL_SURFACE` with the given axis, origin on the axis, and radius.
    pub(crate) fn cylindrical_surface(
        &mut self,
        origin: [f64; 3],
        axis: [f64; 3],
        reference: [f64; 3],
        radius: f64,
    ) -> Result<u32, String> {
        let placement = self.axis2(origin, axis, reference)?;
        Ok(self.add(format!(
            "CYLINDRICAL_SURFACE('',#{placement},{})",
            real(radius)
        )))
    }

    /// `CONICAL_SURFACE`: `radius` at `origin`, half-angle `semi_angle` radians.
    pub(crate) fn conical_surface(
        &mut self,
        origin: [f64; 3],
        axis: [f64; 3],
        reference: [f64; 3],
        radius: f64,
        semi_angle: f64,
    ) -> Result<u32, String> {
        let placement = self.axis2(origin, axis, reference)?;
        Ok(self.add(format!(
            "CONICAL_SURFACE('',#{placement},{},{})",
            real(radius),
            real(semi_angle)
        )))
    }

    /// `SPHERICAL_SURFACE` centred on the placement origin.
    pub(crate) fn spherical_surface(
        &mut self,
        centre: [f64; 3],
        axis: [f64; 3],
        reference: [f64; 3],
        radius: f64,
    ) -> Result<u32, String> {
        let placement = self.axis2(centre, axis, reference)?;
        Ok(self.add(format!(
            "SPHERICAL_SURFACE('',#{placement},{})",
            real(radius)
        )))
    }

    /// `TOROIDAL_SURFACE` centred on the placement origin.
    pub(crate) fn toroidal_surface(
        &mut self,
        centre: [f64; 3],
        axis: [f64; 3],
        reference: [f64; 3],
        major_radius: f64,
        minor_radius: f64,
    ) -> Result<u32, String> {
        let placement = self.axis2(centre, axis, reference)?;
        Ok(self.add(format!(
            "TOROIDAL_SURFACE('',#{placement},{},{})",
            real(major_radius),
            real(minor_radius)
        )))
    }
}

/// Unit vector, rejecting zero/non-finite input.
pub(crate) fn unit(vector: [f64; 3], what: &str) -> Result<[f64; 3], String> {
    let length = norm(vector);
    if !length.is_finite() || length == 0.0 {
        return Err(format!("{what} is degenerate: {vector:?}"));
    }
    Ok([vector[0] / length, vector[1] / length, vector[2] / length])
}

/// `vector` with the `axis` component removed.
pub(crate) fn orthogonalize(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let projection = vector[0] * axis[0] + vector[1] * axis[1] + vector[2] * axis[2];
    [
        vector[0] - axis[0] * projection,
        vector[1] - axis[1] * projection,
        vector[2] - axis[2] * projection,
    ]
}

pub(crate) fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

struct Parser<'a> { bytes: &'a [u8], pos: usize }
impl<'a> Parser<'a> {
    /// Skip whitespace and ISO-10303-21 `/* … */` comments.
    ///
    /// Comments are whitespace in Part 21 and may appear between any tokens, including
    /// between `#id=` and the entity instance or inside an argument list. Files such as
    /// the StepCode ATS1 fixture rely on that; without this the parser fails at the
    /// first comment with `expected '('`.
    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.bytes.get(self.pos) == Some(&b'/') && self.bytes.get(self.pos + 1) == Some(&b'*') {
                self.pos += 2;
                while self.pos < self.bytes.len() {
                    if self.bytes[self.pos] == b'*' && self.bytes.get(self.pos + 1) == Some(&b'/') {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }
    fn entity_start(&mut self) -> Result<Option<u32>, String> {
        loop {
            self.skip_ws();
            if self.pos >= self.bytes.len() { return Ok(None); }
            if self.bytes[self.pos] == b'#' {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() { self.pos += 1; }
                let id = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?.parse().map_err(|e: std::num::ParseIntError| e.to_string())?;
                return Ok(Some(id));
            }
            self.pos += 1;
        }
    }
    fn expect(&mut self, byte: u8) -> Result<(), String> {
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&byte) { self.pos += 1; Ok(()) } else { Err(format!("expected '{}' at {}", byte as char, self.pos)) }
    }
    fn identifier(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len() && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_') { self.pos += 1; }
        String::from_utf8(self.bytes[start..self.pos].to_vec()).map_err(|e| e.to_string())
    }
    fn arguments(&mut self) -> Result<Vec<Value>, String> {
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b')') => { self.pos += 1; return Ok(values); }
                Some(b',') => { self.pos += 1; }
                Some(_) => values.push(self.value()?),
                None => return Err("unterminated arguments".into()),
            }
        }
    }
    fn value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.bytes.get(self.pos).copied() {
            Some(b'#') => { self.pos += 1; let start=self.pos; while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) { self.pos+=1; } Ok(Value::Ref(std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e|e.to_string())?.parse().map_err(|e:std::num::ParseIntError|e.to_string())?)) }
            Some(b'\'') => { self.pos += 1; let mut text=String::new(); while let Some(&c)=self.bytes.get(self.pos) { self.pos+=1; if c==b'\'' { if self.bytes.get(self.pos)==Some(&b'\''){self.pos+=1;text.push('\'')} else {break} } else { text.push(c as char); } } Ok(Value::Str(text)) }
            Some(b'(') => { self.pos += 1; Ok(Value::List(self.arguments()?)) }
            Some(b'$') => { self.pos += 1; Ok(Value::Null) }
            Some(b'.') => { self.pos += 1; let start=self.pos; while self.bytes.get(self.pos)!=Some(&b'.') { self.pos+=1; } let enum_text=String::from_utf8_lossy(&self.bytes[start..self.pos]).to_string(); self.pos+=1; Ok(Value::Enum(enum_text)) }
            Some(_) => {
                let start = self.pos;
                while let Some(&c)=self.bytes.get(self.pos) { if c.is_ascii_alphanumeric()||matches!(c,b'+'|b'-'|b'.'|b'E'|b'e'|b'_') { self.pos+=1; } else { break; } }
                if self.pos == start {
                    let symbol = self.bytes[self.pos] as char;
                    self.pos += 1;
                    return Ok(Value::Typed(symbol.to_string(), Vec::new()));
                }
                let token = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| e.to_string())?;
                if let Ok(number)=token.parse::<f64>() { Ok(Value::Number(number)) } else { Ok(Value::Typed(token.into(), Vec::new())) }
            }
            None => Err("unexpected end of STEP data".into()),
        }
    }
}



/// Merges several STEP files into one DATA section with unique entity ids.
pub fn merge_step_files(paths: &[String], output: &str) -> Result<(), String> {
    if paths.is_empty() { return Err("no STEP files to merge".into()); }
    let first = fs::read_to_string(&paths[0]).map_err(|error| format!("{}: {error}", paths[0]))?;
    let header = first.split_once("DATA;").map(|(head, _)| format!("{head}DATA;\n")).ok_or("missing DATA section")?;
    let mut lines = Vec::new();
    let mut offset = 0u32;
    for path in paths {
        let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
        let data = text.split_once("DATA;").and_then(|(_, rest)| rest.split_once("ENDSEC;").map(|(data, _)| data)).ok_or("missing DATA section")?;
        let mut local_max = 0u32;
        for line in data.lines().filter(|line| line.trim_start().starts_with('#')) {
            if let Some(end) = line.find('=') {
                if let Ok(id) = line[1..end].trim().parse::<u32>() { local_max = local_max.max(id); }
            }
            let mut shifted = String::new();
            let bytes = line.as_bytes();
            let mut index = 0;
            while index < bytes.len() {
                if bytes[index] == b'#' {
                    let mut end = index + 1;
                    while end < bytes.len() && bytes[end].is_ascii_digit() { end += 1; }
                    if end > index + 1 {
                        let id: u32 = line[index+1..end].parse().map_err(|error: std::num::ParseIntError| error.to_string())?;
                        shifted.push('#');
                        shifted.push_str(&(id + offset).to_string());
                        index = end;
                        continue;
                    }
                }
                shifted.push(bytes[index] as char);
                index += 1;
            }
            lines.push(shifted);
        }
        offset += local_max;
    }
    let mut out = header;
    for line in lines { out.push_str(&line); out.push('\n'); }
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    fs::write(output, out).map_err(|error| format!("{output}: {error}"))
}

/// Writes an indexed triangle mesh as a planar STEP B-rep, welding shared coordinates.
pub fn write_mesh_planar(path: &str, vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<(), String> {
    if vertices.is_empty() || faces.is_empty() { return Err("mesh is empty".into()); }
    let mut body = PlanarBody::default();
    let mut lookup = BTreeMap::new();
    for vertex in vertices {
        let p = [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64];
        let key = [(p[0] * 1e7).round() as i64, (p[1] * 1e7).round() as i64, (p[2] * 1e7).round() as i64];
        lookup.entry(key).or_insert_with(|| { body.points.push(p); body.points.len() - 1 });
    }
    for face in faces {
        let indexes: Vec<usize> = face.iter().filter_map(|index| {
            let p = vertices.get(*index as usize)?;
            let key = [(p[0] as f64 * 1e7).round() as i64, (p[1] as f64 * 1e7).round() as i64, (p[2] as f64 * 1e7).round() as i64];
            lookup.get(&key).copied()
        }).collect();
        if indexes.len() == 3 { body.faces.push(indexes); }
    }
    write_faceted_brep(path, &body)
}

/// Writes a centered NodeCraft wedge as planar faces.
pub fn write_wedge(path: &str, length: f64, width: f64, height: f64, top_length: f64) -> Result<(), String> {
    if !(length > 0.0 && width > 0.0 && height > 0.0) { return Err("wedge length, width and height must be positive".into()); }
    if !(top_length >= 0.0 && top_length < length) { return Err("wedge top_length must be at least 0 and less than the length".into()); }
    let (hl, hw, hh) = (length/2.0, width/2.0, height/2.0);
    let (xh, xl, xt) = (-hl, hl, -hl + top_length);
    let (ymin, ymax, zmin, zmax) = (-hw, hw, -hh, hh);
    let mut points = vec![
        [xh, ymin, zmin], [xl, ymin, zmin], [xl, ymax, zmin], [xh, ymax, zmin],
        [xh, ymin, zmax], [xh, ymax, zmax],
    ];
    let mut faces: Vec<Vec<usize>> = Vec::new();
    // bottom, reversed to face outward -Z
    faces.push(vec![3, 2, 1, 0]);
    // high -X face
    faces.push(vec![0, 4, 5, 3]);
    if top_length > 0.0 {
        points.push([xt, ymin, zmax]); // 6
        points.push([xt, ymax, zmax]); // 7
        faces.push(vec![6, 1, 2, 7]); // slope
        faces.push(vec![0, 1, 6, 4]); // -Y trapezoid
        faces.push(vec![3, 5, 7, 2]); // +Y trapezoid
        faces.push(vec![4, 6, 7, 5]); // flat top
    } else {
        faces.push(vec![4, 1, 2, 5]); // slope triangle pair as quad? sharp top: vertices 4,1,2,5 are coplanar? yes slope quad with top edge 4-5 and bottom 1-2, but top edge length width at xh; okay
        faces.push(vec![0, 1, 4]); // -Y triangle
        faces.push(vec![3, 5, 2]); // +Y triangle
    }
    write_planar_body(path, &PlanarBody { points, faces, ..PlanarBody::default() })
}

/// Builds a centered NodeCraft square pyramid as planar faces.
pub fn pyramid_body(length: f64, height: f64) -> Result<PlanarBody, String> {
    if !(length > 0.0 && height > 0.0) { return Err("pyramid length and height must be positive".into()); }
    let (hl, hh) = (length / 2.0, height / 2.0);
    let points = vec![
        [-hl, -hl, -hh], [hl, -hl, -hh], [hl, hl, -hh], [-hl, hl, -hh],
        [0.0, 0.0, hh],
    ];
    let faces = vec![
        vec![3, 2, 1, 0], // base, outward -Z
        vec![0, 1, 4],
        vec![1, 2, 4],
        vec![2, 3, 4],
        vec![3, 0, 4],
    ];
    Ok(PlanarBody { points, faces, ..PlanarBody::default() })
}

/// Writes a centered NodeCraft square pyramid as planar faces.
pub fn write_pyramid(path: &str, length: f64, height: f64) -> Result<(), String> {
    write_planar_body(path, &pyramid_body(length, height)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_box_and_roundtrips_topology() {
        let path = std::env::temp_dir().join("nodecraft-step-box.step");
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        let body = read_planar_body(path.to_str().unwrap()).unwrap();
        assert_eq!(body.faces.len(), 6);
        assert_eq!(body.points.len(), 8);
        assert!((body.volume() - 800.0).abs() < 1e-9);
        assert!((body.area() - 640.0).abs() < 1e-9);
        assert_eq!(body.bounds(), Some(([-5.0,-2.0,-10.0],[5.0,2.0,10.0])));
    }

    #[test]
    fn converts_centimetre_units_to_millimetres() {
        let path = std::env::temp_dir().join("nodecraft-step-box-cm.step");
        write_box(path.to_str().unwrap(), 10.0, 4.0, 20.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT(.CENTI.,.METRE.)");
        std::fs::write(&path, text).unwrap();
        let (min, max) = read_step_bounds(path.to_str().unwrap()).unwrap();
        assert_eq!(min, [-50.0, -20.0, -100.0]);
        assert_eq!(max, [50.0, 20.0, 100.0]);
    }

    #[test]
    fn detects_dangling_references() {
        let file = StepFile::parse("DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));#2=VERTEX_POINT('',#99);ENDSEC;").unwrap();
        assert_eq!(file.dangling_refs(), vec![99]);
    }

    #[test]
    fn parses_block_comments_between_tokens() {
        let text = "DATA;\n#1= /* entity comment */ CARTESIAN_POINT('',(1.,2.,3.));\n#2=CARTESIAN_POINT('',/* before list */(0.,0.,0.));\n#3=VERTEX_POINT('',#2);\nENDSEC;";
        let file = StepFile::parse(text).unwrap();
        assert_eq!(file.entities.len(), 3);
        assert_eq!(file.entities.get(&1).unwrap().name, "CARTESIAN_POINT");
        assert_eq!(
            file.entities.get(&2).unwrap().args[1],
            Value::List(vec![Value::Number(0.0), Value::Number(0.0), Value::Number(0.0)])
        );
    }

    #[test]
    fn parses_stepcode_ats1_fixture_with_comments() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/step-stepcode/ATS1-out.stp");
        if !path.exists() { return; }
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        // 186 `#id=` definitions, including complex instances behind /* … */ comments.
        assert_eq!(file.entities.len(), 186, "ATS1 parsed {} entities", file.entities.len());
        assert!(file.entities.values().any(|entity| entity.name == "NODE"));
    }

    #[test]
    fn rejects_missing_or_truncated_data() {
        assert!(StepFile::parse("HEADER;
ENDSEC;").is_err());
        assert!(StepFile::parse("DATA;#1=CARTESIAN_POINT('',(0.,0.,0.)").is_err());
    }

    #[test]
    fn rejects_duplicate_entity_ids() {
        assert!(StepFile::parse("DATA;#1=CARTESIAN_POINT('',(0.,0.,0.));#1=CARTESIAN_POINT('',(1.,1.,1.));ENDSEC;").is_err());
    }

    #[test]
    fn repairs_inside_out_box_orientation() {
        let path = std::env::temp_dir().join("nodecraft-step-orient.step");
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let mut body = read_planar_body(path.to_str().unwrap()).unwrap();
        for face in &mut body.faces { face.reverse(); }
        assert!(body.volume() < 0.0);
        body.repair_orientation().unwrap();
        assert!((body.volume() - 8.0).abs() < 1e-9, "{}", body.volume());
    }

    #[test]
    fn validates_closed_box_and_rejects_open_plate() {
        let path = std::env::temp_dir().join("nodecraft-step-closed-box.step");
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let body = read_planar_body(path.to_str().unwrap()).unwrap();
        body.validate_closed().unwrap();
        let plate = PlanarBody { points: vec![[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]], faces: vec![vec![0,1,2,3]], holes: vec![vec![]] };
        assert!(plate.validate_closed().is_err());
    }

    #[test]
    fn writes_hollow_box_with_void() {
        let path = std::env::temp_dir().join("nodecraft-step-void.step");
        write_box_with_void(path.to_str().unwrap(), [10.0, 10.0, 10.0], [4.0, 4.0, 4.0]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("BREP_WITH_VOIDS("));
        assert!(text.contains("CLOSED_SHELL("));
    }

    #[test]
    fn writes_open_shell_sheet() {
        let path = std::env::temp_dir().join("nodecraft-step-open-sheet.step");
        let body = PlanarBody { points: vec![[0.,0.,0.],[4.,0.,0.],[4.,4.,0.],[0.,4.,0.]], faces: vec![vec![0,1,2,3]], holes: vec![vec![]] };
        write_open_shell(path.to_str().unwrap(), &body).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("OPEN_SHELL("));
        assert!(text.contains("SHELL_BASED_SURFACE_MODEL("));
        assert!(text.contains("MANIFOLD_SURFACE_SHAPE_REPRESENTATION("));
        assert!(!text.contains("MANIFOLD_SOLID_BREP"));
    }

    #[test]
    fn parses_references_and_lists() {
        let file = StepFile::parse("DATA;#1=CARTESIAN_POINT('',(1.,2.,3.));#2=VERTEX_POINT('',#1);ENDSEC;").unwrap();
        assert_eq!(file.entities[&2].args[1], Value::Ref(1));
        assert_eq!(file.bounds(), Some(([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])));
    }
    #[test]
    fn writer_repairs_reversed_solid() {
        let path = std::env::temp_dir().join(format!("sn-reversed-{}.step", std::process::id()));
        let body = PlanarBody {
            points: vec![[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]],
            faces: vec![vec![0,3,2,1],vec![4,5,6,7],vec![0,1,5,4],vec![1,2,6,5],vec![2,3,7,6],vec![3,0,4,7]],
            holes: vec![vec![];6],
        };
        let reversed = PlanarBody { faces: body.faces.iter().map(|face| face.iter().rev().copied().collect()).collect(), ..body.clone() };
        assert!(reversed.volume() < 0.0);
        write_planar_body(path.to_str().unwrap(), &reversed).unwrap();
        let written = read_planar_solids(path.to_str().unwrap()).unwrap();
        assert!((written[0].volume() - 1.0).abs() < 1e-9, "{}", written[0].volume());
    }

    #[test]
    fn rewrites_ap203_and_ap242_schemas() {
        for (schema, expected) in [("ap203", "CONFIG_CONTROL_DESIGN"), ("ap214", "AUTOMOTIVE_DESIGN"), ("ap242", "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF")] {
            let path = std::env::temp_dir().join(format!("sn-schema-{schema}-{}.step", std::process::id()));
            write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
            set_schema(path.to_str().unwrap(), schema).unwrap();
            assert_eq!(read_header(path.to_str().unwrap()).unwrap().schema.as_deref(), Some(expected));
            assert!(read_planar_solids(path.to_str().unwrap()).is_ok());
        }
    }

    #[test]
    fn exposes_entity_type_tables() {
        let path = std::env::temp_dir().join(format!("sn-tables-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let file = StepFile::parse(&text).unwrap();
        assert_eq!(file.entities_named("MANIFOLD_SOLID_BREP").len(), 1);
        assert_eq!(file.entities_named("ADVANCED_FACE").len(), 6);
        assert_eq!(file.entities_named("CARTESIAN_POINT").len(), 8);
        let counts = file.type_counts();
        assert_eq!(counts.get("PRODUCT"), Some(&1));
        assert_eq!(counts.get("CLOSED_SHELL"), Some(&1));
    }

    #[test]
    fn welds_coincident_points() {
        let mut body = PlanarBody::default();
        body.points = vec![[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]];
        body.push_face(vec![0,1,2,3], Vec::new());
        body.push_face(vec![4,5,6], Vec::new());
        assert_eq!(body.weld_points(1e-6), 3);
        assert_eq!(body.points.len(), 4);
        assert!(body.faces.iter().all(|face| face.iter().all(|index| *index < 4)));
    }

    #[test]
    fn inspects_complex_entity_members() {
        let file = StepFile::parse("DATA;#1=(BOUNDED_SURFACE() B_SPLINE_SURFACE() SURFACE());ENDSEC;").unwrap();
        let entity = file.entities.get(&1).unwrap();
        assert!(entity.has_member("BOUNDED_SURFACE"));
        assert!(entity.has_member("SURFACE"));
        assert!(!entity.has_member("PLANE"));
        assert!(file.is_type(1, "B_SPLINE_SURFACE"));
        assert!(!file.is_type(1, "PLANE"));
    }

    #[test]
    fn reads_face_surface_entities() {
        let path = std::env::temp_dir().join(format!("sn-facesurface-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let text = std::fs::read_to_string(&path).unwrap().replace("ADVANCED_FACE(", "FACE_SURFACE(");
        std::fs::write(&path, text).unwrap();
        let brep = read_topology_file(path.to_str().unwrap()).unwrap();
        assert_eq!(brep.faces.len(), 6);
        assert_eq!(brep.solids.len(), 1);
    }

    #[test]
    fn reads_oriented_face_entities() {
        let path = std::env::temp_dir().join(format!("sn-orientedface-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let mut text = std::fs::read_to_string(&path).unwrap().replace("ADVANCED_FACE(", "FACE_SURFACE(");
        let mut next = 900u32;
        let mut oriented = String::new();
        for line in text.lines().filter(|line| line.contains("=FACE_SURFACE(")).map(str::to_string).collect::<Vec<_>>() {
            if let Some(id) = line.strip_prefix('#').and_then(|rest| rest.split('=').next()).and_then(|id| id.parse::<u32>().ok()) {
                oriented.push_str(&format!("#{next}=ORIENTED_FACE('',#{id},.T.);\n")); next += 1;
            }
        }
        let index = text.rfind("ENDSEC;").unwrap();
        text.insert_str(index, &oriented);
        std::fs::write(&path, text).unwrap();
        let brep = read_topology_file(path.to_str().unwrap()).unwrap();
        assert!(brep.faces.len() >= 12, "{}", brep.faces.len());
    }

    #[test]
    fn writes_and_reads_faceted_brep() {
        let path = std::env::temp_dir().join(format!("sn-faceted-write-{}.step", std::process::id()));
        let body = PlanarBody {
            points: vec![[0.,0.,0.],[2.,0.,0.],[2.,2.,0.],[0.,2.,0.],[0.,0.,2.],[2.,0.,2.],[2.,2.,2.],[0.,2.,2.]],
            faces: vec![vec![0,3,2,1],vec![4,5,6,7],vec![0,1,5,4],vec![1,2,6,5],vec![2,3,7,6],vec![3,0,4,7]],
            holes: vec![vec![];6],
        };
        write_faceted_brep(path.to_str().unwrap(), &body).unwrap();
        let bodies = read_faceted_breps(path.to_str().unwrap()).unwrap();
        assert_eq!(bodies.len(), 1);
        assert!((bodies[0].volume() - 8.0).abs() < 1e-9, "{}", bodies[0].volume());
    }

    #[test]
    fn exposes_ordered_entity_table() {
        let path = std::env::temp_dir().join(format!("sn-entity-table-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        let file = StepFile::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let table = file.entity_table();
        assert_eq!(table.len(), file.entities.len());
        assert!(table.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert_eq!(table.first().map(|(_, name)| name.as_str()), Some("APPLICATION_CONTEXT"));
    }

    #[test]
    fn transforms_inch_file_in_millimetres() {
        let path = std::env::temp_dir().join(format!("sn-inch-transform-{}.step", std::process::id()));
        write_box(path.to_str().unwrap(), 2.0, 2.0, 2.0).unwrap();
        convert_step_units(path.to_str().unwrap(), LengthUnit::Millimetre, LengthUnit::Inch).unwrap();
        let before = read_step_bounds(path.to_str().unwrap()).unwrap();
        transform_step_file(path.to_str().unwrap(), [0.0; 3], [10.0, 0.0, 0.0]).unwrap();
        let after = read_step_bounds(path.to_str().unwrap()).unwrap();
        assert!((after.0[0] - (before.0[0] + 10.0)).abs() < 1e-6, "{before:?} {after:?}");
        assert!((after.1[0] - (before.1[0] + 10.0)).abs() < 1e-6, "{before:?} {after:?}");
    }

    #[test]
    fn writes_pyramid_and_roundtrips_topology() {
        let path = std::env::temp_dir().join("nodecraft-step-pyramid.step");
        write_pyramid(path.to_str().unwrap(), 20.0, 30.0).unwrap();
        let body = read_planar_body(path.to_str().unwrap()).unwrap();
        assert_eq!(body.faces.len(), 5);
        assert_eq!(body.points.len(), 5);
        assert!((body.volume() - 4000.0).abs() < 1e-9);
        assert!((body.area() - (400.0 + 2.0 * 20.0 * 1000.0_f64.sqrt())).abs() < 1e-9);
    }



    fn analytic_temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nodecraft-step-analytic-{tag}-{}.step", std::process::id()))
    }

    /// A single-solid, primitive-shaped cylinder STEP file (counts only; detection is
    /// signature-based, the references need not form real topology).
    fn synthetic_cylinder_step(
        axis: [f64; 3],
        surface_radius: f64,
        bottom_radius: f64,
        top_radius: f64,
        bottom_z: f64,
        top_z: f64,
        extra: &str,
    ) -> String {
        format!(
            "ISO-10303-21;\n/* .MILLI.,.METRE. */\nDATA;\n\
             #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=DIRECTION('',({},{},{}));\n\
             #3=DIRECTION('',(1.,0.,0.));\n\
             #4=AXIS2_PLACEMENT_3D('',#1,#2,#3);\n\
             #5=CYLINDRICAL_SURFACE('',#4,{surface_radius});\n\
             #10=CARTESIAN_POINT('',(0.,0.,{bottom_z}));\n\
             #11=AXIS2_PLACEMENT_3D('',#10,#2,#3);\n\
             #12=CIRCLE('',#11,{bottom_radius});\n\
             #20=CARTESIAN_POINT('',(0.,0.,{top_z}));\n\
             #21=AXIS2_PLACEMENT_3D('',#20,#2,#3);\n\
             #22=CIRCLE('',#21,{top_radius});\n\
             #8=PLANE('',#4);\n\
             #9=PLANE('',#4);\n\
             #30=ADVANCED_FACE('',(#12),#8,.F.);\n\
             #31=ADVANCED_FACE('',(#22),#9,.F.);\n\
             #32=ADVANCED_FACE('',(#12),#5,.F.);\n\
             #40=CLOSED_SHELL('',(#30,#31,#32));\n\
             #41=MANIFOLD_SOLID_BREP('',#40);\n\
             {extra}\
             ENDSEC;\nEND-ISO-10303-21;\n",
            axis[0], axis[1], axis[2],
        )
    }

    fn detect_text(text: &str, tag: &str) -> Option<AnalyticSolid> {
        let path = analytic_temp_path(tag);
        std::fs::write(&path, text).unwrap();
        let result = detect_analytic_solid(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        result
    }

    #[test]
    fn analytic_detection_accepts_written_primitives() {
        let cylinder_path = analytic_temp_path("written-cylinder");
        write_cylinder(cylinder_path.to_str().unwrap(), 5.0, 10.0).unwrap();
        let cylinder = detect_analytic_solid(cylinder_path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&cylinder_path).ok();
        match cylinder {
            Some(AnalyticSolid::Cylinder { radius, height, center }) => {
                assert!((radius - 5.0).abs() < 1e-9, "radius {radius}");
                assert!((height - 10.0).abs() < 1e-9, "height {height}");
                assert_eq!(center, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected a cylinder, got {other:?}"),
        }

        // A written cone's base/top radii follow the circle Z order, so an inverted
        // cone keeps its orientation instead of being sorted by radius.
        let cone_path = analytic_temp_path("written-cone");
        write_cone(cone_path.to_str().unwrap(), 2.0, 5.0, 10.0).unwrap();
        let cone = detect_analytic_solid(cone_path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&cone_path).ok();
        match cone {
            Some(AnalyticSolid::Cone { radius_base, radius_top, height, center }) => {
                assert!((radius_base - 2.0).abs() < 1e-9, "base {radius_base}");
                assert!((radius_top - 5.0).abs() < 1e-9, "top {radius_top}");
                assert!((height - 10.0).abs() < 1e-9, "height {height}");
                assert_eq!(center, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected a cone, got {other:?}"),
        }

        let sphere_path = analytic_temp_path("written-sphere");
        write_sphere(sphere_path.to_str().unwrap(), 5.0).unwrap();
        let sphere = detect_analytic_solid(sphere_path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&sphere_path).ok();
        match sphere {
            Some(AnalyticSolid::Sphere { radius, center }) => {
                assert!((radius - 5.0).abs() < 1e-9, "radius {radius}");
                assert_eq!(center, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected a sphere, got {other:?}"),
        }

        let torus_path = analytic_temp_path("written-torus");
        write_torus(torus_path.to_str().unwrap(), 8.0, 2.0).unwrap();
        let torus = detect_analytic_solid(torus_path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&torus_path).ok();
        match torus {
            Some(AnalyticSolid::Torus { major, minor, center }) => {
                assert!((major - 8.0).abs() < 1e-9, "major {major}");
                assert!((minor - 2.0).abs() < 1e-9, "minor {minor}");
                assert_eq!(center, [0.0, 0.0, 0.0]);
            }
            other => panic!("expected a torus, got {other:?}"),
        }
    }

    #[test]
    fn analytic_detection_requires_a_genuine_primitive_signature() {
        // The reference synthetic file is accepted with exact dimensions.
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step([0.0, 0.0, 1.0], 5.0, 5.0, 5.0, -5.0, 5.0, ""),
                "valid",
            ),
            Some(AnalyticSolid::Cylinder { radius: 5.0, height: 10.0, center: [0.0, 0.0, 0.0] })
        );
        // A cylinder whose axis is not ±Z is a rotated primitive this API cannot
        // represent; it must not be reported as an axis-aligned cylinder.
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step([1.0, 0.0, 0.0], 5.0, 5.0, 5.0, -5.0, 5.0, ""),
                "horizontal",
            ),
            None
        );
        // Non-positive dimensions are not a primitive.
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step([0.0, 0.0, 1.0], -5.0, 5.0, 5.0, -5.0, 5.0, ""),
                "negative-radius",
            ),
            None
        );
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step([0.0, 0.0, 1.0], 5.0, 5.0, 5.0, 0.0, 0.0, ""),
                "zero-height",
            ),
            None
        );
        // Both cap circles must match the surface radius.
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step([0.0, 0.0, 1.0], 5.0, 4.0, 5.0, -5.0, 5.0, ""),
                "mismatched-circles",
            ),
            None
        );
        // More than one solid means this is not a whole-file primitive.
        assert_eq!(
            detect_text(
                &synthetic_cylinder_step(
                    [0.0, 0.0, 1.0],
                    5.0,
                    5.0,
                    5.0,
                    -5.0,
                    5.0,
                    "#42=MANIFOLD_SOLID_BREP('',#40);\n",
                ),
                "multi-solid",
            ),
            None
        );
    }

    #[test]
    fn four_pin_plug_is_not_a_whole_file_analytic_primitive() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/Sample AP203 Data Files/4pinplug.stp");
        if !path.exists() {
            return;
        }
        let report = classify_step(path.to_str().unwrap()).unwrap();
        assert_eq!(report.solids, 7, "4pinplug is a multi-solid file");
        // Whole-file primitive detection must stay out of the multi-solid path; this
        // assertion does not mask the per-solid `multi::import_solids_partial` reader
        // (that path is owned separately and needs the shell-local signature fix).
        assert_eq!(detect_analytic_solid(path.to_str().unwrap()).unwrap(), None);
    }
}
