//! STEP assembly export for planar bodies.
//!
//! Builds a small, valid AP214 product graph that places several
//! [`nodecraft_step::PlanarBody`] parts with rigid transforms:
//!
//! * one `PRODUCT` / `PRODUCT_DEFINITION` / `ADVANCED_BREP_SHAPE_REPRESENTATION`
//!   chain per part, whose representation items are an identity placement and a
//!   `MANIFOLD_SOLID_BREP` built from the body (real polygon geometry, never faceted
//!   approximations);
//! * one assembly `PRODUCT` / `SHAPE_REPRESENTATION` whose items are the part
//!   placements;
//! * one `NEXT_ASSEMBLY_USAGE_OCCURRENCE` per instance with a
//!   `PRODUCT_DEFINITION_SHAPE`, `CONTEXT_DEPENDENT_SHAPE_REPRESENTATION`,
//!   `REPRESENTATION_RELATIONSHIP` (+ `..._WITH_TRANSFORMATION`) and
//!   `ITEM_DEFINED_TRANSFORMATION`.
//!
//! The local transform of each occurrence is emitted as the parent placement with an
//! identity child placement (`local = parent * child⁻¹ = parent`), which is exactly what
//! [`crate::parse_assembly_file`] and [`crate::planar_instances`] read back.
//!
//! # Round trip
//!
//! ```no_run
//! use step_assemblies::planar_instances;
//! use step_assemblies::write::{PlanarPart, write_planar_assembly};
//!
//! # fn main() -> Result<(), String> {
//! // Any planar solid body can be a part; here the crate's own box writer makes one.
//! nodecraft_step::write_box("block.step", 10.0, 4.0, 20.0)?;
//! let body = nodecraft_step::read_planar_body("block.step")?;
//!
//! let parts = [PlanarPart::new("block", body)];
//! write_planar_assembly("assembly.step", &parts)?;
//! assert_eq!(planar_instances("assembly.step")?.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! # Limitations (deliberate, not silent)
//!
//! * **Rigid transforms only.** Each [`PlanarPart::transform`] must be a right-handed
//!   rotation + translation; scale/shear/reflection is rejected with an
//!   `unsupported:` error because an `AXIS2_PLACEMENT_3D` cannot represent it.
//! * **Flat assemblies.** One root assembly containing the parts directly; nested
//!   sub-assemblies are not produced.
//! * **No instancing.** Every instance gets its own copy of the part geometry, so N
//!   instances of the same body write N solids (a `MAPPED_ITEM`/shared-representation
//!   exporter could deduplicate later).
//! * **Millimetres.** Body coordinates and transforms are written as-is into a
//!   millimetre file (`SI_UNIT(.MILLI.,.METRE.)`); convert beforehand if needed.
//! * **Planar only.** The B-rep emitter copies the conventions of
//!   `nodecraft_step::write_planar_body` (planar faces, straight polygon edges, holes
//!   preserved); curved surface geometry cannot be represented by `PlanarBody`. Every
//!   body is written as `MANIFOLD_SOLID_BREP` like that writer does, so a planar sheet
//!   body stays a solid B-rep rather than being downgraded to a surface model.
//! * Bodies with no faces, out-of-range indices, degenerate/coplanar-violating loops,
//!   or repeated loop points are rejected rather than exported damaged.

use std::collections::BTreeMap;

use nodecraft_step::PlanarBody;

use crate::Transform;

/// Prefix of errors caused by geometry/transforms outside this writer's slice.
pub const UNSUPPORTED_ERROR_PREFIX: &str = "unsupported: ";

/// One placed planar part for [`planar_assembly_text`] / [`write_planar_assembly`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarPart {
    /// Product name; also used for the instance placement name in the assembly tree.
    pub name: String,
    /// Planar B-rep in the part's local coordinates (millimetres).
    pub body: PlanarBody,
    /// World placement of this instance (millimetres). Identity for an unplaced part.
    pub transform: Transform,
}

impl PlanarPart {
    /// A part at the identity transform.
    #[must_use]
    pub fn new(name: impl Into<String>, body: PlanarBody) -> Self {
        Self { name: name.into(), body, transform: Transform::IDENTITY }
    }

    /// A part placed by `transform`.
    #[must_use]
    pub fn placed(name: impl Into<String>, body: PlanarBody, transform: Transform) -> Self {
        Self { name: name.into(), body, transform }
    }
}

/// The STEP text of a flat assembly containing `parts`, with a default root name.
///
/// # Errors
///
/// Empty input, invalid bodies, or non-rigid transforms (see the module docs).
pub fn planar_assembly_text(parts: &[PlanarPart]) -> Result<String, String> {
    planar_assembly_text_named("ASSEMBLY", parts)
}

/// The STEP text of a flat assembly containing `parts`, rooted at a product named
/// `root_name`.
pub fn planar_assembly_text_named(root_name: &str, parts: &[PlanarPart]) -> Result<String, String> {
    if parts.is_empty() {
        return Err("assembly export needs at least one part".into());
    }
    let mut writer = Writer::default();
    let context = writer.header();

    // Per-part geometry and the identity child placements used by ITEM_DEFINED_TRANSFORMATION.
    let mut solids = Vec::with_capacity(parts.len());
    for part in parts {
        let solid = writer.planar_solid(&part.body, &part.name)?;
        let child = writer.placement(&Transform::IDENTITY, "identity placement")?;
        solids.push((solid, child));
    }

    // Parent placements (one per instance) live in the root representation, so they must
    // exist before the root product chain is emitted.
    let mut parent_placements = Vec::with_capacity(parts.len());
    for part in parts {
        parent_placements.push(writer.placement(
            &part.transform,
            &format!("transform of part {:?}", part.name),
        )?);
    }

    // One product chain per part: representation items = identity placement + solid.
    let mut parts_graph = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let (solid, child) = solids[index];
        let (definition, representation) = writer.product(
            &format!("part-{index}"),
            &part.name,
            RepresentationKind::Brep,
            &[child, solid],
            context,
        );
        parts_graph.push((definition, representation));
    }

    // Root assembly: its representation carries the parent placements.
    let (root_definition, root_representation) = writer.product(
        "assembly",
        root_name,
        RepresentationKind::Shape,
        &parent_placements,
        context,
    );

    // Product structure: every part is a placed child of the root assembly.
    for (index, part) in parts.iter().enumerate() {
        let (part_definition, part_representation) = parts_graph[index];
        let parent_placement = parent_placements[index];
        let child_placement = solids[index].1;
        let nauo = writer.add(format!(
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE('occ-{index}','{}','',#{root_definition},#{part_definition},$)",
            step_string(&part.name)
        ));
        let occurrence_shape = writer.add(format!(
            "PRODUCT_DEFINITION_SHAPE('Placement {index}','',#{nauo})"
        ));
        let transformation = writer.add(format!(
            "ITEM_DEFINED_TRANSFORMATION('','',#{child_placement},#{parent_placement})"
        ));
        let relationship = writer.add(format!(
            "(REPRESENTATION_RELATIONSHIP('','',#{part_representation},#{root_representation})\n\
             REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transformation})\n\
             SHAPE_REPRESENTATION_RELATIONSHIP())"
        ));
        writer.add(format!(
            "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relationship},#{occurrence_shape})"
        ));
    }

    Ok(writer.text())
}

/// Write a flat planar assembly to `path` (millimetres, AP214).
///
/// # Errors
///
/// Those of [`planar_assembly_text`] plus file write errors.
pub fn write_planar_assembly(path: &str, parts: &[PlanarPart]) -> Result<(), String> {
    let text = planar_assembly_text(parts)?;
    std::fs::write(path, text).map_err(|error| format!("{path}: {error}"))
}

// ---------------------------------------------------------------------------------
// STEP entity writer
// ---------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepresentationKind {
    /// `ADVANCED_BREP_SHAPE_REPRESENTATION` — items include a solid.
    Brep,
    /// `SHAPE_REPRESENTATION` — items are placements only.
    Shape,
}

/// Sequential `#id=ENTITY(...);` lines plus the shared AP214 header/context.
#[derive(Default)]
struct Writer {
    next: u32,
    lines: Vec<String>,
    /// Product context id produced by [`Writer::header`], used by [`Writer::product`].
    context: Option<u32>,
}

impl Writer {
    fn add(&mut self, body: impl AsRef<str>) -> u32 {
        self.next += 1;
        self.lines.push(format!("#{}={};", self.next, body.as_ref()));
        self.next
    }

    /// Product/context header entities; returns the representation context id.
    fn header(&mut self) -> u32 {
        let application =
            self.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')");
        let product_context = self.add(format!("PRODUCT_CONTEXT('',#{application},'mechanical')"));
        let length = self.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))");
        let angle = self.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))");
        let solid_angle = self.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())");
        let uncertainty = self.add(format!(
            "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-6),#{length},'distance_accuracy_value','')"
        ));
        self.context = Some(product_context);
        self.add(format!(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) \
             GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) REPRESENTATION_CONTEXT('',''))"
        ))
    }

    /// `AXIS2_PLACEMENT_3D` for a rigid transform.
    fn placement(&mut self, transform: &Transform, what: &str) -> Result<u32, String> {
        let frame = RigidFrame::from_transform(transform)
            .map_err(|error| format!("{UNSUPPORTED_ERROR_PREFIX}{what}: {error}"))?;
        let point = self.add(format!(
            "CARTESIAN_POINT('',({},{},{}))",
            real(frame.origin[0]),
            real(frame.origin[1]),
            real(frame.origin[2])
        ));
        let axis = self.add(format!(
            "DIRECTION('',({},{},{}))",
            real(frame.axis[0]),
            real(frame.axis[1]),
            real(frame.axis[2])
        ));
        let reference = self.add(format!(
            "DIRECTION('',({},{},{}))",
            real(frame.reference[0]),
            real(frame.reference[1]),
            real(frame.reference[2])
        ));
        Ok(self.add(format!(
            "AXIS2_PLACEMENT_3D('',#{point},#{axis},#{reference})"
        )))
    }

    /// `PRODUCT` -> formation -> definition -> `PRODUCT_DEFINITION_SHAPE` -> representation
    /// -> `SHAPE_DEFINITION_REPRESENTATION`; returns `(definition, representation)`.
    fn product(
        &mut self,
        product_id: &str,
        name: &str,
        kind: RepresentationKind,
        representation_items: &[u32],
        context: u32,
    ) -> (u32, u32) {
        let product_context = self
            .context
            .expect("Writer::header must run before Writer::product");
        let product = self.add(format!(
            "PRODUCT('{}','{}','',(#{product_context}))",
            step_string(product_id),
            step_string(name)
        ));
        let formation = self.add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
        let definition = self.add(format!("PRODUCT_DEFINITION('design','',#{formation},$)"));
        let shape = self.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"));
        let items: Vec<String> = representation_items
            .iter()
            .map(|item| format!("#{item}"))
            .collect();
        let kind_name = match kind {
            RepresentationKind::Brep => "ADVANCED_BREP_SHAPE_REPRESENTATION",
            RepresentationKind::Shape => "SHAPE_REPRESENTATION",
        };
        let representation = self.add(format!(
            "{kind_name}('{}',({}),#{context})",
            step_string(name),
            items.join(",")
        ));
        self.add(format!(
            "SHAPE_DEFINITION_REPRESENTATION(#{shape},#{representation})"
        ));
        (definition, representation)
    }

    /// Emit a `MANIFOLD_SOLID_BREP` for `body` (planar topology only) and return its id.
    ///
    /// Mirrors the conventions of `nodecraft_step::write_planar_body`: every point becomes
    /// a vertex, shared straight edges are de-duplicated, the first loop of each face is
    /// the `FACE_OUTER_BOUND`, further loops become `FACE_BOUND` holes, and the `.T.`
    /// orientation flags keep the stored loop handedness.
    fn planar_solid(&mut self, body: &PlanarBody, name: &str) -> Result<u32, String> {
        let normals = planar_face_normals(body)?;
        let point_ids: Vec<u32> = body
            .points
            .iter()
            .map(|[x, y, z]| {
                self.add(format!(
                    "CARTESIAN_POINT('',({},{},{}))",
                    real(*x),
                    real(*y),
                    real(*z)
                ))
            })
            .collect();
        let vertex_ids: Vec<u32> = point_ids
            .iter()
            .map(|point| self.add(format!("VERTEX_POINT('',#{point})")))
            .collect();

        // Shared edges, keyed by the unordered vertex pair; the bool records whether the
        // first creation walked from the lower to the higher point index.
        let mut edges: BTreeMap<(usize, usize), (u32, bool)> = BTreeMap::new();
        for face_index in 0..body.faces.len() {
            for loop_points in body.loops(face_index) {
                for index in 0..loop_points.len() {
                    let a = loop_points[index];
                    let b = loop_points[(index + 1) % loop_points.len()];
                    let key = if a < b { (a, b) } else { (b, a) };
                    if edges.contains_key(&key) {
                        continue;
                    }
                    let pa = body.points[a];
                    let pb = body.points[b];
                    let delta = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let length = (delta[0] * delta[0]
                        + delta[1] * delta[1]
                        + delta[2] * delta[2])
                        .sqrt();
                    if !length.is_finite() || length <= 0.0 {
                        return Err(format!("face {face_index} has a degenerate edge"));
                    }
                    let direction = self.add(format!(
                        "DIRECTION('',({},{},{}))",
                        real(delta[0] / length),
                        real(delta[1] / length),
                        real(delta[2] / length)
                    ));
                    let vector = self.add(format!("VECTOR('',#{direction},{})", real(length)));
                    let line = self.add(format!("LINE('',#{},#{vector})", point_ids[a]));
                    let edge = self.add(format!(
                        "EDGE_CURVE('',#{},#{},#{line},.T.)",
                        vertex_ids[a], vertex_ids[b]
                    ));
                    edges.insert(key, (edge, a < b));
                }
            }
        }

        let mut face_ids = Vec::with_capacity(body.faces.len());
        for face_index in 0..body.faces.len() {
            let normal = normals[face_index];
            let mut bounds = Vec::new();
            let mut outer = true;
            for loop_points in body.loops(face_index) {
                let mut oriented = Vec::with_capacity(loop_points.len());
                for index in 0..loop_points.len() {
                    let a = loop_points[index];
                    let b = loop_points[(index + 1) % loop_points.len()];
                    let key = if a < b { (a, b) } else { (b, a) };
                    let (edge, creation_starts_at_min) = *edges
                        .get(&key)
                        .ok_or_else(|| format!("face {face_index} is missing a planar edge"))?;
                    let forward = (a < b) == creation_starts_at_min;
                    oriented.push(self.add(format!(
                        "ORIENTED_EDGE('',*,*,#{edge},.{}.)",
                        if forward { "T" } else { "F" }
                    )));
                }
                let loop_id = self.add(format!(
                    "EDGE_LOOP('',({}))",
                    oriented
                        .iter()
                        .map(|edge| format!("#{edge}"))
                        .collect::<Vec<_>>()
                        .join(",")
                ));
                let kind = if outer { "FACE_OUTER_BOUND" } else { "FACE_BOUND" };
                bounds.push(format!(
                    "#{}",
                    self.add(format!("{kind}('',#{loop_id},.T.)"))
                ));
                outer = false;
            }

            let axis = self.add(format!(
                "DIRECTION('',({},{},{}))",
                real(normal[0]),
                real(normal[1]),
                real(normal[2])
            ));
            let candidate = if normal[0].abs() > 0.9 {
                [0.0, 1.0, 0.0]
            } else {
                [1.0, 0.0, 0.0]
            };
            let reference = unit(orthogonalize(candidate, normal))
                .ok_or_else(|| format!("face {face_index} plane reference is degenerate"))?;
            let reference = self.add(format!(
                "DIRECTION('',({},{},{}))",
                real(reference[0]),
                real(reference[1]),
                real(reference[2])
            ));
            let placement = self.add(format!(
                "AXIS2_PLACEMENT_3D('',#{},#{axis},#{reference})",
                point_ids[body.faces[face_index][0]]
            ));
            let plane = self.add(format!("PLANE('',#{placement})"));
            face_ids.push(self.add(format!(
                "ADVANCED_FACE('',({}),#{plane},.T.)",
                bounds.join(",")
            )));
        }

        let faces = face_ids
            .iter()
            .map(|face| format!("#{face}"))
            .collect::<Vec<_>>()
            .join(",");
        let shell = self.add(format!("CLOSED_SHELL('',({faces}))"));
        Ok(self.add(format!(
            "MANIFOLD_SOLID_BREP('{}',#{shell})",
            step_string(name)
        )))
    }

    fn text(&self) -> String {
        let mut out = String::with_capacity(self.lines.len() * 64 + 512);
        out.push_str(
            "ISO-10303-21;\nHEADER;\n\
             FILE_DESCRIPTION(('nodecraft planar assembly'),'2;1');\n\
             FILE_NAME('assembly.step','',(''),(''),'nodecraft-step','nodecraft','');\n\
             FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
             ENDSEC;\nDATA;\n",
        );
        for line in &self.lines {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
        out
    }
}

// ---------------------------------------------------------------------------------
// Rigid frame handling
// ---------------------------------------------------------------------------------

/// The translation and local axes of a rigid 4x4 transform.
struct RigidFrame {
    origin: [f64; 3],
    axis: [f64; 3],
    reference: [f64; 3],
}

impl RigidFrame {
    /// Decompose a right-handed rigid transform into the `AXIS2_PLACEMENT_3D` fields
    /// (origin, local Z, local X).
    fn from_transform(transform: &Transform) -> Result<Self, String> {
        let rows = transform.rows();
        let origin = [rows[0][3], rows[1][3], rows[2][3]];
        let x = [rows[0][0], rows[1][0], rows[2][0]];
        let y = [rows[0][1], rows[1][1], rows[2][1]];
        let z = [rows[0][2], rows[1][2], rows[2][2]];
        for (label, vector) in [("X", x), ("Y", y), ("Z", z)] {
            let length = dot(vector, vector).sqrt();
            if !length.is_finite() || (length - 1.0).abs() > 1e-9 {
                return Err(format!(
                    "transform column {label} is not a unit vector (length {length}); only rigid rotation+translation placements can be written"
                ));
            }
        }
        for (left, right, dot_name) in [
            (x, y, "X·Y"),
            (x, z, "X·Z"),
            (y, z, "Y·Z"),
        ] {
            let product = dot(left, right);
            if !product.is_finite() || product.abs() > 1e-9 {
                return Err(format!(
                    "transform is not a rotation: columns are not orthogonal ({dot_name} = {product})"
                ));
            }
        }
        let handedness = cross(x, y);
        if !(0..3).all(|axis| (handedness[axis] - z[axis]).abs() <= 1e-9) {
            return Err(
                "transform is left-handed or sheared; only right-handed rigid placements can be written"
                    .into(),
            );
        }
        if !origin.iter().all(|value| value.is_finite()) {
            return Err("transform translation is not finite".into());
        }
        Ok(Self { origin, axis: z, reference: x })
    }
}

// ---------------------------------------------------------------------------------
// Planar body validation (mirrors nodecraft_step::write_planar_body)
// ---------------------------------------------------------------------------------

/// Validate every face/loop of `body` and return one unit normal per face.
///
/// Rejects empty bodies, short/out-of-range/repeated loop points, degenerate loops and
/// loops whose points do not lie on the outer loop's plane.
fn planar_face_normals(body: &PlanarBody) -> Result<Vec<[f64; 3]>, String> {
    if body.points.is_empty() || body.faces.is_empty() {
        return Err("planar body is empty: no points or no faces".into());
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in &body.points {
        if !point.iter().all(|value| value.is_finite()) {
            return Err(format!("planar body has a non-finite point: {point:?}"));
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let diagonal = ((max[0] - min[0]).powi(2)
        + (max[1] - min[1]).powi(2)
        + (max[2] - min[2]).powi(2))
    .sqrt();
    if !(diagonal > 0.0 && diagonal.is_finite()) {
        return Err("planar body has zero extent".into());
    }
    let tolerance = diagonal * 1e-6;
    let mut normals = Vec::with_capacity(body.faces.len());
    for face_index in 0..body.faces.len() {
        let outer = &body.faces[face_index];
        if outer.len() < 3 {
            return Err(format!("face {face_index} outer loop has fewer than three vertices"));
        }
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
                let relative = [
                    body.points[index][0] - origin[0],
                    body.points[index][1] - origin[1],
                    body.points[index][2] - origin[2],
                ];
                let offset = dot(relative, normal);
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

// ---------------------------------------------------------------------------------
// Small helpers (same conventions as nodecraft_step's planar writer)
// ---------------------------------------------------------------------------------

/// Newell normal of a closed, planar polygon (translation-stabilized).
fn newell_normal(points: &[[f64; 3]], loop_points: &[usize]) -> Option<[f64; 3]> {
    let origin = points[*loop_points.first()?];
    let mut normal = [0.0f64; 3];
    for position in 0..loop_points.len() {
        let a = sub(points[loop_points[position]], origin);
        let b = sub(points[loop_points[(position + 1) % loop_points.len()]], origin);
        normal = add(normal, cross(a, b));
    }
    unit(normal)
}

fn orthogonalize(vector: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    let projection = dot(vector, axis);
    [
        vector[0] - axis[0] * projection,
        vector[1] - axis[1] * projection,
        vector[2] - axis[2] * projection,
    ]
}

fn unit(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(vector, vector).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A STEP string literal body, escaping quotes and flattening control characters.
fn step_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\'' => out.push_str("''"),
            '\n' | '\r' | '\t' => out.push(' '),
            character if character.is_control() => out.push(' '),
            character => out.push(character),
        }
    }
    out
}

/// A STEP REAL literal: always contains a decimal point, uses STEP `E` exponents.
fn real(value: f64) -> String {
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

// ---------------------------------------------------------------------------------
// Nested assembly tree
// ---------------------------------------------------------------------------------

/// One node of a nested assembly tree: either a placed part (`body`) or a
/// sub-assembly (`children`), never both.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyNode {
    pub name: String,
    /// Placement relative to the parent node. Ignored for the synthetic root.
    pub transform: Transform,
    pub body: Option<PlanarBody>,
    pub children: Vec<AssemblyNode>,
}

impl AssemblyNode {
    /// A leaf part at the identity transform.
    #[must_use]
    pub fn part(name: impl Into<String>, body: PlanarBody) -> Self {
        Self { name: name.into(), transform: Transform::IDENTITY, body: Some(body), children: Vec::new() }
    }

    /// A sub-assembly with children.
    #[must_use]
    pub fn group(name: impl Into<String>, children: Vec<Self>) -> Self {
        Self { name: name.into(), transform: Transform::IDENTITY, body: None, children }
    }

    /// Places this node relative to its parent.
    #[must_use]
    pub fn placed(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Adds a child and returns the updated node.
    #[must_use]
    pub fn with_child(mut self, child: Self) -> Self {
        self.children.push(child);
        self
    }
}

struct Emitted {
    definition: u32,
    representation: u32,
    identity: u32,
}

fn add_occurrence(
    writer: &mut Writer,
    parent: &Emitted,
    child: &Emitted,
    parent_placement: u32,
    index: usize,
    name: &str,
) -> Result<(), String> {
    let nauo = writer.add(format!(
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE('occ-{index}','{}','',#{},#{},$)",
        step_string(name), parent.definition, child.definition
    ));
    let occurrence_shape = writer.add(format!("PRODUCT_DEFINITION_SHAPE('Placement {index}','',#{nauo})"));
    let transformation = writer.add(format!(
        "ITEM_DEFINED_TRANSFORMATION('','',#{},#{parent_placement})", child.identity
    ));
    let relationship = writer.add(format!(
        "(REPRESENTATION_RELATIONSHIP('','',#{},#{})\n\
         REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{transformation})\n\
         SHAPE_REPRESENTATION_RELATIONSHIP())",
        child.representation, parent.representation
    ));
    writer.add(format!("CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{relationship},#{occurrence_shape})"));
    Ok(())
}

fn emit_node(writer: &mut Writer, node: &AssemblyNode, context: u32, counter: &mut usize) -> Result<Emitted, String> {
    match (&node.body, node.children.is_empty()) {
        (Some(_), false) => return Err(format!("assembly node {:?} has both a body and children", node.name)),
        (None, true) => return Err(format!("assembly group {:?} has no children", node.name)),
        _ => {}
    }
    *counter += 1;
    let key = format!("node-{}", *counter);
    let identity = writer.placement(&Transform::IDENTITY, "identity placement")?;
    if let Some(body) = &node.body {
        let solid = writer.planar_solid(body, &node.name)?;
        let (definition, representation) = writer.product(&key, &node.name, RepresentationKind::Brep, &[identity, solid], context);
        return Ok(Emitted { definition, representation, identity });
    }

    let mut emitted = Vec::with_capacity(node.children.len());
    for child in &node.children {
        emitted.push((child, emit_node(writer, child, context, counter)?));
    }
    let placements: Vec<u32> = emitted.iter().enumerate().map(|(index, (child, _))| {
        writer.placement(&child.transform, &format!("occurrence {index} of {:?}", child.name))
    }).collect::<Result<_, _>>()?;
    let mut items = vec![identity];
    items.extend(placements.iter().copied());
    let (definition, representation) = writer.product(&key, &node.name, RepresentationKind::Shape, &items, context);
    let parent = Emitted { definition, representation, identity };
    for (index, ((child, child_emitted), parent_placement)) in emitted.iter().zip(placements).enumerate() {
        add_occurrence(writer, &parent, child_emitted, parent_placement, index, &child.name)?;
    }
    Ok(parent)
}

/// STEP text for a nested assembly rooted at `root_name`.
pub fn assembly_tree_text(root_name: &str, nodes: &[AssemblyNode]) -> Result<String, String> {
    if nodes.is_empty() { return Err("assembly export needs at least one node".into()); }
    let root = AssemblyNode::group(root_name, nodes.to_vec());
    let mut writer = Writer::default();
    let context = writer.header();
    let mut counter = 0;
    emit_node(&mut writer, &root, context, &mut counter)?;
    Ok(writer.text())
}

/// Writes a nested planar assembly to `path` (millimetres, AP214).
pub fn write_assembly_tree(path: &str, root_name: &str, nodes: &[AssemblyNode]) -> Result<(), String> {
    std::fs::write(path, assembly_tree_text(root_name, nodes)?).map_err(|error| format!("{path}: {error}"))
}
