//! Records to CADIR, the mirror of `solidcore-cadir`'s geometry mapping, and on to
//! cadmpeg's STEP encoder.
//!
//! A schema-4 record becomes one CADIR body: its vertices and edges on their curves,
//! its faces on their surfaces, its loops as coedge rings and its shells under their
//! regions. The mapping prefers the face's and edge's NURBS form when the record carries
//! one, because that is the parameterization its trims and intervals use. Analytic
//! carriers map to CADIR's analytic vocabulary; carriers with no neutral form (an
//! intersection curve, a placed revolution without a NURBS form, a blend definition)
//! refuse the export with the reason. Pcurves are not written: cadmpeg's STEP encoder
//! writes the 3D topology and its carriers.
//!
//! One body per NodeCraft body, with its name, colour and rigid placement. STEP is
//! written by cadmpeg's `StepCodec`; nothing here writes Part 21 text.

use cadmpeg_codec_step::{StepCodec, StepSchema, StepWriteOptions};
use cadmpeg_ir::CadIr;
use cadmpeg_ir::codec::write::target::TargetRequest;
use cadmpeg_ir::codec::write::{EncodeInput, Encoder};
use cadmpeg_ir::document::Model;
use cadmpeg_ir::features::FinitePoint3;
use cadmpeg_ir::geometry::Surface;
use cadmpeg_ir::geometry::analytic::{
    CircleCurve, ConeSurface, CylinderSurface, EllipseCurve, LineCurve, PlaneSurface,
    SphereSurface, TorusSurface,
};
use cadmpeg_ir::geometry::nurbs::{
    NurbsCurve, NurbsPoleGrid, NurbsPoles3, NurbsSurface, NurbsSurfaceAxis, WeightedPole3,
};
use cadmpeg_ir::geometry::{
    CurveGeometry, SolvedCurveGeometry, SolvedSurfaceGeometry, SurfaceGeometry,
};
use cadmpeg_ir::ids::{
    BodyId, CoedgeId, CurveId, EdgeId, FaceId, LoopId, OccurrenceId, PointId, ProductDefinitionId,
    RegionId, ShellId, SurfaceId, VertexId,
};
use cadmpeg_ir::math::{Point3, Vector3};
use cadmpeg_ir::products::{
    Occurrence, OccurrenceParent, ProductDefinition, ProductDefinitionKind, PrototypeReference,
};
use cadmpeg_ir::scalar::{Angle, NonNegativeLength, NonZeroReal, PositiveReal};
use cadmpeg_ir::topology::{
    Body, BodyKind, Coedge, Edge, EdgeCarrier, Face, FaceLoops, Loop, LoopBoundary, LoopRing,
    Point, Region, Sense, Shell, Vertex,
};
use cadmpeg_ir::units::Tolerances;
use solidcore_record::brep as record;

/// One body to write.
#[derive(Clone, Debug)]
pub struct ExportBody {
    /// The body's name (the product and body label).
    pub name: String,
    /// Surface colour, linear RGB in 0..1.
    pub color: Option<[f64; 3]>,
    /// Where the body goes (column-major 4 x 4, millimetres), when it moves.
    pub placement: Option<[[f64; 4]; 4]>,
    /// No closed shell: a surface body.
    pub sheet: bool,
    /// The record.
    pub brep: record::Brep,
}

impl ExportBody {
    /// A body with no colour and no placement.
    pub fn new(name: impl Into<String>, brep: record::Brep) -> Self {
        Self { name: name.into(), color: None, placement: None, sheet: false, brep }
    }
}

fn unsupported(what: impl std::fmt::Display) -> String {
    format!("not supported yet: {what}")
}

fn point(p: [f64; 3]) -> Point3 {
    Point3::new(p[0], p[1], p[2])
}

fn vector(v: [f64; 3]) -> Vector3 {
    Vector3::new(v[0], v[1], v[2])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalized(v: [f64; 3]) -> Result<[f64; 3], String> {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if length.is_finite() && length > 0.0 {
        Ok([v[0] / length, v[1] / length, v[2] / length])
    } else {
        Err("a degenerate direction in the record".into())
    }
}

fn axis_of(frame: &record::Frame) -> Result<[f64; 3], String> {
    normalized(cross(frame.x_axis, frame.y_axis))
}

fn curve_geometry(curve: &record::Curve) -> Result<SolvedCurveGeometry, String> {
    match curve {
        record::Curve::Line { start, end } => {
            let direction = normalized([end[0] - start[0], end[1] - start[1], end[2] - start[2]])?;
            let line = LineCurve::try_new(point(*start), vector(direction))
                .map_err(|error| format!("line: {error}"))?;
            Ok(SolvedCurveGeometry::Line(line))
        }
        record::Curve::Circle { frame, radius } => {
            let axis = axis_of(frame)?;
            let circle = CircleCurve::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                *radius,
            )
            .map_err(|error| format!("circle: {error}"))?;
            Ok(SolvedCurveGeometry::Circle(circle))
        }
        record::Curve::Ellipse { frame, radii } => {
            if radii[0] < radii[1] {
                return Err(unsupported("an ellipse whose record x axis is its minor axis"));
            }
            let axis = axis_of(frame)?;
            let ellipse = EllipseCurve::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                radii[0],
                radii[1],
            )
            .map_err(|error| format!("ellipse: {error}"))?;
            Ok(SolvedCurveGeometry::Ellipse(ellipse))
        }
        record::Curve::Polyline { points } => {
            if points.len() < 2 {
                return Err("a polyline with fewer than two points".into());
            }
            // Degree one on unit-spaced knots: segment i spans [i, i + 1].
            let mut knots = vec![0.0];
            knots.extend((1..points.len()).map(|value| value as f64));
            knots.push((points.len() - 1) as f64);
            let curve = NurbsCurve::new(
                1,
                knots,
                NurbsPoles3::Polynomial { points: points.iter().map(|p| point(*p)).collect() },
                false,
            )
            .map_err(|error| format!("polyline: {error}"))?;
            Ok(SolvedCurveGeometry::Nurbs(curve))
        }
        record::Curve::Spline { knots, points } => {
            let degree = knots
                .len()
                .checked_sub(points.len().saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a spline with no degree".to_string())?;
            let curve = NurbsCurve::new(
                degree,
                knots.clone(),
                NurbsPoles3::Polynomial { points: points.iter().map(|p| point(*p)).collect() },
                false,
            )
            .map_err(|error| format!("spline curve: {error}"))?;
            Ok(SolvedCurveGeometry::Nurbs(curve))
        }
        record::Curve::RationalSpline { knots, homogeneous } => {
            let degree = knots
                .len()
                .checked_sub(homogeneous.len().saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a rational spline with no degree".to_string())?;
            let mut poles = Vec::with_capacity(homogeneous.len());
            for pole in homogeneous {
                let w = pole[3];
                let weight = NonZeroReal::new(w)
                    .ok_or_else(|| "a rational spline with a zero weight".to_string())?;
                poles.push(WeightedPole3 {
                    point: point([pole[0] / w, pole[1] / w, pole[2] / w]),
                    weight,
                });
            }
            let curve = NurbsCurve::new(
                degree,
                knots.clone(),
                NurbsPoles3::Rational { points: poles },
                false,
            )
            .map_err(|error| format!("rational spline curve: {error}"))?;
            Ok(SolvedCurveGeometry::Nurbs(curve))
        }
        record::Curve::SurfaceIntersection { .. } => {
            Err(unsupported("a surface-intersection curve with no NURBS form"))
        }
    }
}

fn surface_geometry(surface: &record::Surface) -> Result<SolvedSurfaceGeometry, String> {
    match surface {
        record::Surface::Plane { origin, u_axis, v_axis } => {
            let normal = normalized(cross(*u_axis, *v_axis))?;
            let plane = PlaneSurface::try_new(point(*origin), vector(normal), vector(*u_axis))
                .map_err(|error| format!("plane: {error}"))?;
            Ok(SolvedSurfaceGeometry::Plane(plane))
        }
        record::Surface::PlaneThrough { origin, u_point, v_point } => {
            let u = [u_point[0] - origin[0], u_point[1] - origin[1], u_point[2] - origin[2]];
            let v = [v_point[0] - origin[0], v_point[1] - origin[1], v_point[2] - origin[2]];
            let normal = normalized(cross(u, v))?;
            let plane = PlaneSurface::try_new(point(*origin), vector(normal), vector(u))
                .map_err(|error| format!("plane: {error}"))?;
            Ok(SolvedSurfaceGeometry::Plane(plane))
        }
        record::Surface::Cylinder { frame, radius } => {
            let axis = axis_of(frame)?;
            let cylinder = CylinderSurface::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                *radius,
            )
            .map_err(|error| format!("cylinder: {error}"))?;
            Ok(SolvedSurfaceGeometry::Cylinder(cylinder))
        }
        record::Surface::Cone { frame, radius_at_origin, half_angle } => {
            if *radius_at_origin < 0.0 {
                return Err(unsupported("a cone whose radius at the origin is negative"));
            }
            let axis = axis_of(frame)?;
            let radius = NonNegativeLength::new(*radius_at_origin)
                .ok_or_else(|| "a cone radius that is not finite".to_string())?;
            let ratio = PositiveReal::new(1.0).ok_or_else(|| "a cone ratio".to_string())?;
            let angle = Angle::new(*half_angle)
                .ok_or_else(|| "a cone half angle that is not finite".to_string())?;
            let cone = ConeSurface::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                radius.get(),
                ratio.get(),
                angle.get(),
            )
            .map_err(|error| format!("cone: {error}"))?;
            Ok(SolvedSurfaceGeometry::Cone(cone))
        }
        record::Surface::Sphere { frame, radius } => {
            let axis = axis_of(frame)?;
            let sphere = SphereSurface::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                *radius,
            )
            .map_err(|error| format!("sphere: {error}"))?;
            Ok(SolvedSurfaceGeometry::Sphere(sphere))
        }
        record::Surface::Torus { frame, major_radius, minor_radius, tube_start } => {
            if *tube_start != 0.0 {
                return Err(unsupported("a torus whose tube starts away from its frame axis"));
            }
            let axis = axis_of(frame)?;
            let torus = TorusSurface::try_new(
                point(frame.origin),
                vector(axis),
                vector(frame.x_axis),
                *major_radius,
                *minor_radius,
            )
            .map_err(|error| format!("torus: {error}"))?;
            Ok(SolvedSurfaceGeometry::Torus(torus))
        }
        record::Surface::Spline { u_knots, v_knots, points } => {
            let u_degree = u_knots
                .len()
                .checked_sub(points.len().saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a spline surface with no u degree".to_string())?;
            let rows = points.first().map_or(0, Vec::len);
            let v_degree = v_knots
                .len()
                .checked_sub(rows.saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a spline surface with no v degree".to_string())?;
            let surface = NurbsSurface::new(
                NurbsSurfaceAxis::new(u_degree, u_knots.clone(), false),
                NurbsSurfaceAxis::new(v_degree, v_knots.clone(), false),
                NurbsPoleGrid::Polynomial {
                    rows: points
                        .iter()
                        .map(|row| row.iter().map(|p| point(*p)).collect())
                        .collect(),
                },
                false,
            )
            .map_err(|error| format!("spline surface: {error}"))?;
            Ok(SolvedSurfaceGeometry::Nurbs(surface))
        }
        record::Surface::RationalSpline { u_knots, v_knots, homogeneous } => {
            let u_degree = u_knots
                .len()
                .checked_sub(homogeneous.len().saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a rational surface with no u degree".to_string())?;
            let rows = homogeneous.first().map_or(0, Vec::len);
            let v_degree = v_knots
                .len()
                .checked_sub(rows.saturating_add(1))
                .and_then(|degree| u32::try_from(degree).ok())
                .ok_or_else(|| "a rational surface with no v degree".to_string())?;
            let mut grid = Vec::with_capacity(homogeneous.len());
            for row in homogeneous {
                let mut mapped = Vec::with_capacity(row.len());
                for pole in row {
                    let w = pole[3];
                    let weight = NonZeroReal::new(w)
                        .ok_or_else(|| "a rational surface with a zero weight".to_string())?;
                    mapped.push(WeightedPole3 {
                        point: point([pole[0] / w, pole[1] / w, pole[2] / w]),
                        weight,
                    });
                }
                grid.push(mapped);
            }
            let surface = NurbsSurface::new(
                NurbsSurfaceAxis::new(u_degree, u_knots.clone(), false),
                NurbsSurfaceAxis::new(v_degree, v_knots.clone(), false),
                NurbsPoleGrid::Rational { rows: grid },
                false,
            )
            .map_err(|error| format!("rational surface: {error}"))?;
            Ok(SolvedSurfaceGeometry::Nurbs(surface))
        }
        record::Surface::BallPatch { .. } => Err(unsupported("a ball patch without a NURBS form")),
        record::Surface::PlacedRevolution { .. } => {
            Err(unsupported("a placed revolution without a NURBS form"))
        }
        record::Surface::Blend { .. } => Err(unsupported("a blend surface without a NURBS form")),
    }
}

/// The least distance uncertainty written (mm).
const MIN_UNCERTAINTY: f64 = 1e-6;

fn uncertainty(brep: &record::Brep) -> f64 {
    std::iter::once(brep.tolerance)
        .chain(brep.edges.iter().map(|edge| edge.tolerance))
        .chain(brep.vertices.iter().map(|vertex| vertex.tolerance))
        .filter(|tolerance| tolerance.is_finite())
        .fold(MIN_UNCERTAINTY, f64::max)
}

fn positive(value: f64) -> Option<PositiveReal> {
    PositiveReal::new(value)
}

fn transform_of(
    placement: [[f64; 4]; 4],
) -> Result<Option<cadmpeg_ir::transform::Transform>, String> {
    let rows = [
        [placement[0][0], placement[1][0], placement[2][0], placement[3][0]],
        [placement[0][1], placement[1][1], placement[2][1], placement[3][1]],
        [placement[0][2], placement[1][2], placement[2][2], placement[3][2]],
    ];
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err("a placement that is not finite".into());
    }
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let linear = rows.map(|row| [row[0], row[1], row[2]]);
    let translation = [rows[0][3], rows[1][3], rows[2][3]];
    let unmoved = linear == identity && translation == [0.0; 3];
    if unmoved {
        return Ok(None);
    }
    Ok(Some(
        cadmpeg_ir::transform::Transform::affine(rows)
            .ok_or_else(|| "a placement the neutral model refuses".to_string())?,
    ))
}

fn color_of(color: [f64; 3]) -> Option<cadmpeg_ir::topology::Color> {
    cadmpeg_ir::topology::Color::new(color[0] as f32, color[1] as f32, color[2] as f32, 1.0)
}

struct BodyWriter<'a> {
    model: &'a mut Model,
    brep: &'a record::Brep,
    body: BodyId,
    index: usize,
    sheet: bool,
}

impl BodyWriter<'_> {
    fn id(&self, kind: &str, key: impl std::fmt::Display) -> String {
        format!("nodecraft:export:{kind}#{}_{key}", self.index)
    }

    fn write(&mut self) -> Result<(), String> {
        let tolerance = uncertainty(self.brep);
        // Vertices and their points.
        let mut vertex_ids = Vec::with_capacity(self.brep.vertices.len());
        for (index, vertex) in self.brep.vertices.iter().enumerate() {
            let point_id =
                PointId::mint(self.id("point", index)).map_err(|error| error.to_string())?;
            let position = FinitePoint3::new(point(vertex.point))
                .ok_or_else(|| "a vertex with a non-finite coordinate".to_string())?;
            self.model.points.push(Point::new(point_id.clone(), position, None));
            let vertex_id =
                VertexId::mint(self.id("vertex", index)).map_err(|error| error.to_string())?;
            self.model.vertices.push(Vertex {
                id: vertex_id.clone(),
                point: point_id,
                tolerance: positive(vertex.tolerance),
            });
            vertex_ids.push(vertex_id);
        }

        // Edge curves and edges.
        let mut edge_ids = Vec::with_capacity(self.brep.edges.len());
        for (index, edge) in self.brep.edges.iter().enumerate() {
            let carrier_curve = edge.nurbs.as_ref().unwrap_or(&edge.curve);
            let geometry = curve_geometry(carrier_curve)?;
            let curve_id =
                CurveId::mint(self.id("curve", index)).map_err(|error| error.to_string())?;
            self.model.curves.push(cadmpeg_ir::geometry::Curve {
                id: curve_id.clone(),
                geometry: CurveGeometry::Solved(geometry),
                parameter_range: None,
                source_object: None,
            });
            let start = vertex_ids
                .get(edge.vertices[0] as usize)
                .ok_or_else(|| "an edge naming a missing start vertex".to_string())?
                .clone();
            let end = vertex_ids
                .get(edge.vertices[1] as usize)
                .ok_or_else(|| "an edge naming a missing end vertex".to_string())?
                .clone();
            let carrier = EdgeCarrier::new(Some(curve_id), Some(edge.interval))
                .map_err(|error| format!("edge interval: {error}"))?;
            let edge_id =
                EdgeId::mint(self.id("edge", index)).map_err(|error| error.to_string())?;
            self.model.edges.push(Edge {
                id: edge_id.clone(),
                carrier,
                start,
                end,
                tolerance: positive(edge.tolerance),
            });
            edge_ids.push(edge_id);
        }

        // Which faces each shell uses, and the first side each face is seen from.
        let mut shell_faces: std::collections::BTreeMap<usize, Vec<(usize, bool)>> =
            std::collections::BTreeMap::new();
        let mut shell_wires: std::collections::BTreeMap<usize, Vec<EdgeId>> =
            std::collections::BTreeMap::new();
        let mut shell_vertices: std::collections::BTreeMap<usize, Vec<VertexId>> =
            std::collections::BTreeMap::new();
        let mut face_shell: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        let mut face_forward: std::collections::BTreeMap<usize, bool> =
            std::collections::BTreeMap::new();
        for region_index in 0..self.brep.regions.len() {
            let region = &self.brep.regions[region_index];
            if region_index == 0 {
                continue;
            }
            for shell_index in &region.shells {
                let shell = self
                    .brep
                    .shells
                    .get(*shell_index as usize)
                    .ok_or_else(|| "a region naming a missing shell".to_string())?;
                match shell {
                    record::Shell::Faces { faces, wires } => {
                        let entry = shell_faces.entry(*shell_index as usize).or_default();
                        for use_ in faces {
                            let face = use_.face as usize;
                            entry.push((face, use_.forward));
                            face_shell.entry(face).or_insert(*shell_index as usize);
                            face_forward.entry(face).or_insert(use_.forward);
                        }
                        if !wires.is_empty() {
                            let wire_edges = shell_wires.entry(*shell_index as usize).or_default();
                            for wire in wires {
                                if let Some(edge) = edge_ids.get(*wire as usize) {
                                    wire_edges.push(edge.clone());
                                }
                            }
                        }
                    }
                    record::Shell::Wire { edges } => {
                        let wire_edges = shell_wires.entry(*shell_index as usize).or_default();
                        for edge in edges {
                            if let Some(edge) = edge_ids.get(*edge as usize) {
                                wire_edges.push(edge.clone());
                            }
                        }
                    }
                    record::Shell::Vertex { vertex } => {
                        if let Some(vertex) = vertex_ids.get(*vertex as usize) {
                            shell_vertices
                                .entry(*shell_index as usize)
                                .or_default()
                                .push(vertex.clone());
                        }
                    }
                }
            }
        }

        // Regions and their shells.
        let mut region_ids = Vec::new();
        for region_index in 1..self.brep.regions.len() {
            let region = &self.brep.regions[region_index];
            if region.void {
                continue;
            }
            let region_id = RegionId::mint(self.id("region", region_index))
                .map_err(|error| error.to_string())?;
            let mut shells = Vec::new();
            for shell_index in &region.shells {
                let shell_id = ShellId::mint(self.id("shell", shell_index))
                    .map_err(|error| error.to_string())?;
                if self.model.shells.iter().any(|shell| shell.id == shell_id) {
                    shells.push(shell_id);
                    continue;
                }
                let faces = shell_faces
                    .get(&(*shell_index as usize))
                    .map(|uses| {
                        uses.iter()
                            .filter_map(|(face, _)| FaceId::mint(self.id("face", face)).ok())
                            .collect()
                    })
                    .unwrap_or_default();
                let wires = shell_wires.get(&(*shell_index as usize)).cloned().unwrap_or_default();
                let vertices =
                    shell_vertices.get(&(*shell_index as usize)).cloned().unwrap_or_default();
                let shell = Shell::new(shell_id.clone(), region_id.clone(), faces, wires, vertices)
                    .map_err(|error| format!("shell: {error:?}"))?;
                self.model.shells.push(shell);
                shells.push(shell_id);
            }
            self.model.regions.push(Region {
                id: region_id.clone(),
                body: self.body.clone(),
                shells: shells.clone(),
            });
            region_ids.push(region_id);
        }
        if region_ids.is_empty() {
            // A sheet body has no bounded region: keep every shell in one region.
            let region_id =
                RegionId::mint(self.id("region", 0)).map_err(|error| error.to_string())?;
            let mut shells = Vec::new();
            for shell_index in 0..self.brep.shells.len() {
                let shell_id = ShellId::mint(self.id("shell", shell_index))
                    .map_err(|error| error.to_string())?;
                let faces: Vec<FaceId> = shell_faces
                    .get(&shell_index)
                    .map(|uses| {
                        uses.iter()
                            .filter_map(|(face, _)| FaceId::mint(self.id("face", face)).ok())
                            .collect()
                    })
                    .unwrap_or_default();
                let wires = shell_wires.get(&shell_index).cloned().unwrap_or_default();
                let vertices = shell_vertices.get(&shell_index).cloned().unwrap_or_default();
                if faces.is_empty() && wires.is_empty() && vertices.is_empty() {
                    continue;
                }
                let shell = Shell::new(shell_id.clone(), region_id.clone(), faces, wires, vertices)
                    .map_err(|error| format!("sheet shell: {error:?}"))?;
                self.model.shells.push(shell);
                shells.push(shell_id);
            }
            self.model.regions.push(Region {
                id: region_id.clone(),
                body: self.body.clone(),
                shells,
            });
            region_ids.push(region_id);
        }

        // Faces, their surfaces and loops.
        for face_index in 0..self.brep.faces.len() {
            if !face_shell.contains_key(&face_index) {
                continue;
            }
            let face = &self.brep.faces[face_index];
            let carrier = face.nurbs.as_ref().unwrap_or(&face.surface);
            let geometry = surface_geometry(carrier)?;
            let surface_id = SurfaceId::mint(self.id("surface", face_index))
                .map_err(|error| error.to_string())?;
            self.model.surfaces.push(Surface {
                id: surface_id.clone(),
                geometry: SurfaceGeometry::Solved(geometry),
                source_object: None,
            });
            let face_id =
                FaceId::mint(self.id("face", face_index)).map_err(|error| error.to_string())?;
            let shell_index = face_shell[&face_index];
            let shell_id =
                ShellId::mint(self.id("shell", shell_index)).map_err(|error| error.to_string())?;
            let mut loop_ids = Vec::new();
            for (loop_index, loop_) in face.loops.iter().enumerate() {
                let loop_id = LoopId::mint(self.id("loop", format!("{face_index}_{loop_index}")))
                    .map_err(|error| error.to_string())?;
                let boundary = match loop_ {
                    record::Loop::Edges { edges } => {
                        // Record loops run relative to the surface's natural normal. A
                        // shell use with a reversed face sees the boundary the other way
                        // round, and the STEP face sense then agrees with the loop.
                        let reversed = !face_forward.get(&face_index).copied().unwrap_or(true);
                        let ordered: Vec<(usize, &record::OrientedEdge)> = if reversed {
                            edges.iter().enumerate().rev().collect()
                        } else {
                            edges.iter().enumerate().collect()
                        };
                        let mut coedges = Vec::with_capacity(edges.len());
                        for (position, oriented) in ordered {
                            let coedge_id = CoedgeId::mint(
                                self.id("coedge", format!("{face_index}_{loop_index}_{position}")),
                            )
                            .map_err(|error| error.to_string())?;
                            let edge = match self.brep.edges.get(oriented.edge as usize) {
                                Some(edge) => edge,
                                None => return Err("a loop naming a missing edge".into()),
                            };
                            let radial_next = edge
                                .uses
                                .iter()
                                .position(|use_| {
                                    use_.face as usize == face_index
                                        && use_.r#loop as usize == loop_index
                                        && use_.position as usize == position
                                })
                                .and_then(|use_index| {
                                    if edge.uses.len() <= 1 {
                                        return None;
                                    }
                                    let next = &edge.uses[(use_index + 1) % edge.uses.len()];
                                    CoedgeId::mint(self.id(
                                        "coedge",
                                        format!("{}_{}_{}", next.face, next.r#loop, next.position),
                                    ))
                                    .ok()
                                })
                                .unwrap_or_else(|| coedge_id.clone());
                            let forward =
                                if reversed { !oriented.forward } else { oriented.forward };
                            self.model.coedges.push(Coedge {
                                id: coedge_id.clone(),
                                owner_loop: loop_id.clone(),
                                edge: edge_ids
                                    .get(oriented.edge as usize)
                                    .ok_or_else(|| "a loop naming a missing edge".to_string())?
                                    .clone(),
                                radial_next,
                                sense: if forward { Sense::Forward } else { Sense::Reversed },
                                pcurves: Vec::new(),
                                use_curve: None,
                            });
                            coedges.push(coedge_id);
                        }
                        LoopBoundary::Ring(
                            LoopRing::new(coedges, Vec::new())
                                .map_err(|error| format!("loop ring: {error}"))?,
                        )
                    }
                    record::Loop::Vertex { vertex } => LoopBoundary::Vertex {
                        vertex: vertex_ids
                            .get(*vertex as usize)
                            .ok_or_else(|| "a vertex loop naming a missing vertex".to_string())?
                            .clone(),
                        pcurves: Vec::new(),
                    },
                };
                self.model.loops.push(Loop {
                    id: loop_id.clone(),
                    face: face_id.clone(),
                    boundary,
                });
                loop_ids.push(loop_id);
            }
            let loops = if loop_ids.is_empty() {
                FaceLoops::unspecified(Vec::new())
            } else {
                let outer = loop_ids[0].clone();
                FaceLoops::classified(outer, loop_ids[1..].to_vec())
            };
            self.model.faces.push(Face {
                id: face_id.clone(),
                shell: shell_id,
                surface: surface_id,
                sense: if face_forward.get(&face_index).copied().unwrap_or(true) {
                    Sense::Forward
                } else {
                    Sense::Reversed
                },
                loops,
                name: None,
                color: None,
                tolerance: positive(tolerance),
            });
        }

        self.model.bodies.push(Body {
            id: self.body.clone(),
            kind: if self.sheet { BodyKind::Sheet } else { BodyKind::Solid },
            regions: region_ids,
            transform: None,
            name: None,
            color: None,
            visible: None,
        });
        Ok(())
    }
}

/// Build CADIR for `bodies`; `product` names the document.
pub fn cadir(bodies: &[ExportBody], product: &str) -> Result<CadIr, String> {
    if bodies.is_empty() {
        return Err("STEP export has no bodies".into());
    }
    let tolerance =
        bodies.iter().map(|body| uncertainty(&body.brep)).fold(MIN_UNCERTAINTY, f64::max);
    let mut ir = CadIr::empty();
    ir.tolerances =
        Tolerances::new(tolerance, 1.0e-6).map_err(|error| format!("export tolerance: {error}"))?;
    let mut model = Model::default();
    for (index, body) in bodies.iter().enumerate() {
        let id = BodyId::mint(format!("nodecraft:export:body#{index}"))
            .map_err(|error| error.to_string())?;
        let transform = body.placement.map(transform_of).transpose()?.flatten();
        let name = if body.name.trim().is_empty() {
            format!("{product}_{}", index + 1)
        } else {
            body.name.clone()
        };
        let mut writer = BodyWriter {
            model: &mut model,
            brep: &body.brep,
            body: id.clone(),
            index,
            sheet: body.sheet,
        };
        writer.write()?;
        if let Some(body) = model.bodies.last_mut() {
            body.name = Some(name.clone());
            body.color = bodies[index].color.and_then(color_of);
            // A root placement belongs to the occurrence, not the body: the STEP writer
            // then states a placed instance (a real assembly) instead of a moved body.
            body.transform = None;
        }
        let definition = ProductDefinitionId::mint(format!("nodecraft:export:product#{index}"))
            .map_err(|error| error.to_string())?;
        model.product_definitions.push(ProductDefinition {
            id: definition.clone(),
            kind: ProductDefinitionKind::Part,
            source_name: Some(name.clone()),
            label: Some(name.clone()),
            description: None,
            part_number: None,
            bom_properties: Default::default(),
            bodies: vec![id.clone()],
            sketches: Vec::new(),
            native_ref: None,
        });
        let occurrence = OccurrenceId::mint(format!("nodecraft:export:occurrence#{index}"))
            .map_err(|error| error.to_string())?;
        model.occurrences.push(Occurrence {
            id: occurrence,
            prototype: PrototypeReference::Local { definition },
            parent: OccurrenceParent::Root {},
            ordinal: index as u32,
            transform: transform.unwrap_or(cadmpeg_ir::transform::Transform::identity()),
            linked_prototype: None,
            scale: [cadmpeg_ir::scalar::FiniteReal::ONE; 3],
            name: Some(name),
            visible: None,
            link: None,
            native_ref: None,
        });
    }
    ir.model = model;
    Ok(ir)
}

/// Presentation and schema options for one STEP write.
#[derive(Clone, Debug)]
pub struct StepExportOptions {
    /// The `FILE_NAME` and root product name.
    pub product_name: String,
    /// `ap203`, `ap214` or `ap242` (AP242 edition 1).
    pub schema: String,
    /// The `FILE_NAME` author entry.
    pub author: String,
    /// The `FILE_NAME` organization entry.
    pub organization: String,
    /// Apply this body colour to every body, overriding per-body colours.
    pub color: Option<[f64; 3]>,
    /// Put every body on a presentation layer with this name.
    pub layer: Option<String>,
}

impl Default for StepExportOptions {
    fn default() -> Self {
        Self {
            product_name: "model".to_string(),
            schema: "ap242".to_string(),
            author: String::new(),
            organization: "NodeCraft".to_string(),
            color: None,
            layer: None,
        }
    }
}

fn schema_and_target(schema: &str) -> Result<(StepSchema, &'static str), String> {
    let schema = match schema.trim().to_ascii_lowercase().as_str() {
        "" | "ap242" | "ap242e1" | "ap242e2" | "ap242e3" => StepSchema::Ap242Edition1,
        "ap214" | "214" | "automotive_design" => StepSchema::Ap214,
        "ap203" | "203" | "ap203e1" | "config_control_design" => StepSchema::Ap203Edition1,
        other => return Err(format!("unsupported STEP schema {other:?}")),
    };
    let target = match schema {
        StepSchema::Ap203Edition1 => "ap203e1",
        StepSchema::Ap214 => "ap214",
        _ => "ap242e1",
    };
    Ok((schema, target))
}

/// Write `bodies` to `path` with NodeCraft's presentation options.
///
/// Every body is one product definition; a placement is carried by that definition's
/// root occurrence, so a Combine with placed parts is written as a STEP assembly.
pub fn write_step_with_options(
    path: &str,
    bodies: &[ExportBody],
    options: &StepExportOptions,
) -> Result<(), String> {
    let product =
        if options.product_name.trim().is_empty() { "model" } else { options.product_name.trim() };
    let mut ir = cadir(bodies, product)?;
    if let Some(color) = options.color {
        let color = color_of(color);
        for body in &mut ir.model.bodies {
            body.color = color;
        }
    }
    if let Some(layer) = options.layer.as_deref().filter(|name| !name.trim().is_empty()) {
        let id = cadmpeg_ir::ids::LayerId::mint("nodecraft:export:layer#0")
            .map_err(|error| error.to_string())?;
        ir.model.presentation_layers.push(cadmpeg_ir::presentation::PresentationLayer {
            id,
            name: layer.trim().to_string(),
            description: None,
            visible: None,
            items: ir
                .model
                .bodies
                .iter()
                .map(|body| cadmpeg_ir::presentation::PresentationItem::Body {
                    body: body.id.clone(),
                })
                .collect(),
        });
    }
    let (_, target) = schema_and_target(&options.schema)?;
    let writer_options = StepWriteOptions {
        product_name: product.to_string(),
        author: options.author.clone(),
        organization: if options.organization.trim().is_empty() {
            "NodeCraft".to_string()
        } else {
            options.organization.clone()
        },
        timestamp: None,
        originating_system: "NodeCraft".to_string(),
    };
    let plan = StepCodec { options: writer_options }
        .plan(EncodeInput::new(&ir, None), TargetRequest::Explicit(target))
        .map_err(|error| format!("STEP export of '{product}': {error}"))?;
    let mut bytes = Vec::new();
    plan.write_to(&mut bytes).map_err(|error| format!("STEP export of '{product}': {error}"))?;
    if bytes.is_empty() {
        return Err("STEP export produced no bytes".into());
    }
    std::fs::write(path, bytes).map_err(|error| format!("{path}: {error}"))
}

/// Write `bodies` to `path` as one STEP part through cadmpeg's encoder.
///
/// `schema` is `ap203`, `ap214` or `ap242` (the AP242 edition 1 long form).
pub fn write_step(
    path: &str,
    bodies: &[ExportBody],
    product: &str,
    schema: &str,
) -> Result<(), String> {
    write_step_with_options(
        path,
        bodies,
        &StepExportOptions {
            product_name: product.to_string(),
            schema: schema.to_string(),
            ..StepExportOptions::default()
        },
    )
}

/// The record form of one kernel B-rep.
pub fn record_of_brep(brep: &solidcore::topology::Brep) -> Result<record::Brep, String> {
    let mut kernel = nodecraft_kernel_profile::kernel();
    let mut copy =
        brep.clone_standing_for_self().map_err(|error| format!("STEP export copy: {error}"))?;
    let (record, _order) = solidcore_convert::brep::to_record(&mut kernel, &mut copy)
        .map_err(|error| format!("STEP export record: {error}"))?;
    Ok(record)
}

fn body_stem(path: &str) -> String {
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = base
        .strip_suffix(".step")
        .or_else(|| base.strip_suffix(".stp"))
        .or_else(|| base.strip_suffix(".STEP"))
        .or_else(|| base.strip_suffix(".STP"))
        .unwrap_or(base);
    if stem.trim().is_empty() { "part".to_string() } else { stem.trim().to_string() }
}

/// The `ExportBody` list for one or more kernel B-reps.
pub fn bodies_from_breps(
    stem: &str,
    breps: &[solidcore::topology::Brep],
) -> Result<Vec<ExportBody>, String> {
    if breps.is_empty() {
        return Err("Part has no solids".into());
    }
    let mut bodies = Vec::with_capacity(breps.len());
    for (index, brep) in breps.iter().enumerate() {
        let record = record_of_brep(brep)?;
        if record.regions.len() <= 1 && record.shells.is_empty() {
            return Err(format!("body {} has no STEP geometry", index + 1));
        }
        let name =
            if breps.len() == 1 { stem.to_string() } else { format!("{stem}_{}", index + 1) };
        bodies.push(ExportBody::new(name, record));
    }
    Ok(bodies)
}

/// Write an exact SolidCore B-rep as one STEP part.
pub fn write_brep_exact(path: &str, brep: &solidcore::topology::Brep) -> Result<(), String> {
    write_breps_exact(path, std::slice::from_ref(brep))
}

/// Write several exact SolidCore solids as one STEP part.
pub fn write_breps_exact(path: &str, breps: &[solidcore::topology::Brep]) -> Result<(), String> {
    let stem = body_stem(path);
    let bodies = bodies_from_breps(&stem, breps)?;
    write_step(path, &bodies, &stem, "ap242")
}

/// Write several exact SolidCore solids with presentation options.
pub fn write_breps_exact_with_options(
    path: &str,
    breps: &[solidcore::topology::Brep],
    options: &StepExportOptions,
) -> Result<(), String> {
    let stem = body_stem(path);
    let bodies = bodies_from_breps(&stem, breps)?;
    let mut options = options.clone();
    if options.product_name.trim().is_empty() {
        options.product_name = stem;
    }
    write_step_with_options(path, &bodies, &options)
}

/// A closed faceted body from an indexed triangle mesh: one planar triangular face per
/// triangle, with line carriers on shared edges.
fn record_from_mesh(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Result<record::Brep, String> {
    use std::collections::BTreeMap;

    fn key(point: [f64; 3]) -> [i64; 3] {
        [
            (point[0] * 1e7).round() as i64,
            (point[1] * 1e7).round() as i64,
            (point[2] * 1e7).round() as i64,
        ]
    }
    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    }
    fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
        let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (length.is_finite() && length > 0.0).then(|| [v[0] / length, v[1] / length, v[2] / length])
    }

    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut point_index: BTreeMap<[i64; 3], u32> = BTreeMap::new();
    let mut remap = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let point = [f64::from(vertex[0]), f64::from(vertex[1]), f64::from(vertex[2])];
        if point.iter().any(|value| !value.is_finite()) {
            return Err("a mesh vertex is not finite".into());
        }
        let index = *point_index.entry(key(point)).or_insert_with(|| {
            points.push(point);
            (points.len() - 1) as u32
        });
        remap.push(index);
    }

    let mut vertices_record = Vec::with_capacity(points.len());
    for point in &points {
        vertices_record.push(record::Vertex { point: *point, tolerance: 1e-7 });
    }
    let mut edges: Vec<record::Edge> = Vec::new();
    let mut edge_index: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    let mut faces = Vec::new();
    let mut oriented_faces = Vec::new();
    for triangle in triangles {
        let Some(&a) = triangle.first().and_then(|index| remap.get(*index as usize)) else {
            continue;
        };
        let Some(&b) = triangle.get(1).and_then(|index| remap.get(*index as usize)) else {
            continue;
        };
        let Some(&c) = triangle.get(2).and_then(|index| remap.get(*index as usize)) else {
            continue;
        };
        if a == b || b == c || c == a {
            continue;
        }
        let pa = points[a as usize];
        let pb = points[b as usize];
        let pc = points[c as usize];
        let Some(u) = normalize(sub(pb, pa)) else {
            continue;
        };
        let Some(normal) = normalize(cross(sub(pb, pa), sub(pc, pa))) else {
            continue;
        };
        let Some(v) = normalize(cross(normal, u)) else {
            continue;
        };
        let ua = 0.0;
        let ub = (pb[0] - pa[0]) * u[0] + (pb[1] - pa[1]) * u[1] + (pb[2] - pa[2]) * u[2];
        let uc = (pc[0] - pa[0]) * u[0] + (pc[1] - pa[1]) * u[1] + (pc[2] - pa[2]) * u[2];
        let va = 0.0;
        let vb = (pb[0] - pa[0]) * v[0] + (pb[1] - pa[1]) * v[1] + (pb[2] - pa[2]) * v[2];
        let vc = (pc[0] - pa[0]) * v[0] + (pc[1] - pa[1]) * v[1] + (pc[2] - pa[2]) * v[2];
        let mut loop_edges = Vec::with_capacity(3);
        for (start, end) in [(a, b), (b, c), (c, a)] {
            let (low, high) = if start < end { (start, end) } else { (end, start) };
            let edge = *edge_index.entry((low, high)).or_insert_with(|| {
                let start_point = points[low as usize];
                let end_point = points[high as usize];
                edges.push(record::Edge {
                    curve: record::Curve::Line { start: start_point, end: end_point },
                    nurbs: None,
                    interval: [0.0, 1.0],
                    vertices: [low, high],
                    tolerance: 1e-7,
                    uses: Vec::new(),
                });
                (edges.len() - 1) as u32
            });
            let pcurve = record::Curve2::Spline {
                knots: vec![0.0, 0.0, 1.0, 1.0],
                points: vec![[ua, va], [ub, vb]],
            };
            loop_edges.push(record::OrientedEdge { edge, forward: start == low, pcurve });
        }
        oriented_faces.push(record::OrientedFace { face: faces.len() as u32, forward: true });
        faces.push(record::Face {
            surface: record::Surface::Plane { origin: pa, u_axis: u, v_axis: v },
            nurbs: None,
            domain: [
                [ua.min(ub).min(uc), ua.max(ub).max(uc)],
                [va.min(vb).min(vc), va.max(vb).max(vc)],
            ],
            loops: vec![record::Loop::Edges { edges: loop_edges }],
        });
    }
    if faces.is_empty() {
        return Err("faceted export has no non-degenerate triangles".into());
    }
    Ok(record::Brep {
        version: record::VERSION,
        tolerance: 1e-6,
        regions: vec![
            record::Region { shells: Vec::new(), void: false },
            record::Region { shells: vec![0], void: false },
        ],
        shells: vec![record::Shell::Faces { faces: oriented_faces, wires: Vec::new() }],
        faces,
        edges,
        vertices: vertices_record,
    })
}

/// Write an indexed display mesh as a faceted STEP body through CADIR.
///
/// CADIR has no `FACETED_BREP` entity, so the mesh becomes a closed shell of planar
/// triangular faces with line carriers; cadmpeg writes it as an AP203/AP214/AP242 B-rep.
pub fn write_step_faceted(
    path: &str,
    vertices: &[[f32; 3]],
    triangles: &[[u32; 3]],
    options: &StepExportOptions,
) -> Result<(), String> {
    if vertices.is_empty() || triangles.is_empty() {
        return Err("cannot export an empty mesh".into());
    }
    let stem = body_stem(path);
    let mut options = options.clone();
    if options.product_name.trim().is_empty() {
        options.product_name = stem.clone();
    }
    let brep = record_from_mesh(vertices, triangles)?;
    let bodies = vec![ExportBody { name: stem, color: None, placement: None, sheet: false, brep }];
    write_step_with_options(path, &bodies, &options)
}

/// Signed volume of a schema-4 record, through the ops body.
pub fn record_volume(brep: &record::Brep) -> Result<f64, String> {
    let (body, _) =
        solidcore_ops::Body::from_record(brep, false).map_err(|error| error.to_string())?;
    body.volume(1e-9).map_err(|error| error.to_string())
}

/// The record's vertex bounds.
pub fn record_bounds(brep: &record::Brep) -> Option<([f64; 3], [f64; 3])> {
    let mut vertices = brep.vertices.iter();
    let first = vertices.next()?.point;
    let mut min = first;
    let mut max = first;
    for vertex in vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex.point[axis]);
            max[axis] = max[axis].max(vertex.point[axis]);
        }
    }
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plane_maps_to_cadir() {
        let surface = record::Surface::Plane {
            origin: [1.0, 2.0, 3.0],
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
        };
        assert!(matches!(surface_geometry(&surface), Ok(SolvedSurfaceGeometry::Plane(_))));
    }

    #[test]
    fn a_line_maps_to_cadir() {
        let curve = record::Curve::Line { start: [0.0, 0.0, 0.0], end: [0.0, 0.0, 4.0] };
        assert!(matches!(curve_geometry(&curve), Ok(SolvedCurveGeometry::Line(_))));
    }
}
