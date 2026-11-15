//! The cadmpeg seam: the one NodeCraft crate that names cadmpeg or `solidcore-cadir`.
//!
//! Files are decoded by cadmpeg into its neutral model (CADIR) and read straight into
//! schema-4 B-rep records by `solidcore_cadir`. Nothing passes through STEP text. CADIR
//! supplies names, colours, the product occurrence placements and the document
//! tolerance; `solidcore_cadir` supplies one record per body region and a failure per
//! region it refused.
//!
//! Only the codecs for open formats are compiled in (`step`, `fcstd`, `rhino`), matching
//! the pins in `docs/solidcore-sync.md`. Hostile input is bounded by the service resource
//! limits and a 512 MiB input cap; panics inside cadmpeg become errors.

use std::collections::BTreeMap;
use std::io::Cursor;

use cadmpeg_core::decode::{
    DecodeArena, DecodeContext, DecodeDepth, DecodeMode, DecodePolicy, ResourceLimits, View,
};
use cadmpeg_ir::codec::{Codec, DecodeFailure, DecodeOptions, DecodeResult};
use cadmpeg_ir::document::CadIr;
use cadmpeg_ir::products::{Occurrence, OccurrenceParent, ProductDefinition, PrototypeReference};
use cadmpeg_ir::transform::Transform;
use cadmpeg_ir::validate::validate_neutral_for_decode;
use cadmpeg_registry::{
    DETECTION_PREFIX_LEN, InputCatalog, ResolveSourceError, ResolvedSource, forced_input,
};

/// The largest CAD file NodeCraft reads (the STEP/import cap SolidCraft uses).
pub const MAX_INPUT_BYTES: u64 = 512 << 20;

/// The codecs we accept, by cadmpeg codec id.
const ACCEPTED: [&str; 3] = ["step", "fcstd", "rhino"];

/// File extensions read through cadmpeg (lower case).
pub const CAD_EXTS: [&str; 4] = ["step", "stp", "fcstd", "3dm"];

/// Is this a file cadmpeg reads (`.step`, `.stp`, `.FCStd`, `.3dm`, any case)?
pub fn is_cad_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            CAD_EXTS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// One body region read into a record.
#[derive(Clone, Debug)]
pub struct CadBody {
    /// cadmpeg's body id.
    pub id: String,
    /// Which of the body's regions this is, in the body's order.
    pub region: usize,
    /// The body's display name (else the placing product's, else the id).
    pub name: String,
    /// The placing product's path, root first.
    pub product: Vec<String>,
    /// Surface colour, linear RGB in 0..1.
    pub color: Option<[f64; 3]>,
    /// The source hides the body.
    pub hidden: bool,
    /// Where the body goes (column-major 4 x 4, millimetres), when it moves.
    pub placement: Option<[[f64; 4]; 4]>,
    /// The record, in its own coordinates.
    pub brep: solidcore_record::brep::Brep,
    /// The source face of each of the record's faces.
    pub face_origin: Vec<u64>,
    /// No closed shell: a surface body.
    pub sheet: bool,
    /// What the reader repaired or noted on the way.
    pub notes: Vec<String>,
    /// The record faces turned over, in increasing order.
    pub turned: Vec<u32>,
}

/// A body region that could not be read, and why.
#[derive(Clone, Debug)]
pub struct CadFailure {
    /// cadmpeg's body id.
    pub id: String,
    /// Which of the body's regions this is (from 0).
    pub region: usize,
    /// Why it was refused.
    pub message: String,
}

/// Counts of the geometry kinds in a CADIR document, for import diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CadTypeReport {
    /// Top-level bodies.
    pub solids: usize,
    /// Body faces.
    pub faces: usize,
    /// Planar faces.
    pub planes: usize,
    /// Cylindrical faces.
    pub cylinders: usize,
    /// Conical faces.
    pub cones: usize,
    /// Spherical faces.
    pub spheres: usize,
    /// Toroidal faces.
    pub tori: usize,
    /// B-spline or NURBS faces.
    pub bspline_surfaces: usize,
    /// Source tessellations with no exact B-rep body.
    pub faceted_breps: usize,
    /// Faces on a carrier this report does not classify.
    pub other_surfaces: usize,
    /// The document states product occurrences.
    pub assemblies: bool,
}

impl CadTypeReport {
    /// True when every face is planar.
    pub fn all_planar(&self) -> bool {
        self.faces > 0 && self.planes == self.faces
    }

    /// True when the document mixes planar and curved faces.
    pub fn mixed_surfaces(&self) -> bool {
        self.planes > 0 && self.faces > self.planes
    }
}

/// Per-body geometry counts, for import diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CadSolidTypeReport {
    /// Zero-based body index in the CADIR arena.
    pub solid: u32,
    /// Body faces.
    pub faces: usize,
    /// Planar faces.
    pub planes: usize,
    /// Cylindrical faces.
    pub cylinders: usize,
    /// Conical faces.
    pub cones: usize,
    /// Spherical faces.
    pub spheres: usize,
    /// Toroidal faces.
    pub tori: usize,
    /// B-spline or NURBS faces.
    pub bsplines: usize,
    /// The body carries no exact B-rep faces.
    pub faceted: bool,
}

impl CadSolidTypeReport {
    /// True when this body mixes planar and curved faces.
    pub fn mixed(&self) -> bool {
        self.planes > 0 && self.faces > self.planes
    }
}

/// A CAD file read into records.
#[derive(Clone, Debug, Default)]
pub struct CadImport {
    /// The `FILE_NAME` name, when the file states one.
    pub file_name: String,
    /// The detected cadmpeg format id (`step`, `fcstd`, `rhino`).
    pub format: String,
    /// The program that wrote the file, when known.
    pub system: String,
    /// CADIR's linear tolerance, in millimetres.
    pub tolerance: f64,
    /// The body regions read.
    pub bodies: Vec<CadBody>,
    /// The body regions refused, each with why.
    pub failures: Vec<CadFailure>,
    /// What cadmpeg could not bring over or repaired.
    pub warnings: Vec<String>,
    /// Product occurrences in the CADIR tree.
    pub instances: usize,
    /// Occurrences whose definition supplied a record.
    pub placed_instances: usize,
    /// Occurrences not placed (external documents, unresolved prototypes, external only).
    pub unplaced_instances: usize,
    /// Geometry-kind counts over the decoded CADIR document.
    pub type_report: CadTypeReport,
    /// Per-body geometry-kind counts.
    pub solid_types: Vec<CadSolidTypeReport>,
}

/// The pcurves a face's uses need when the file gives no closed form, made by the
/// kernel, as both STEP readers made them.
struct KernelTrims;

impl solidcore_ops::pcurve::Trims for KernelTrims {
    fn trims(
        &self,
        face: &solidcore_ops::pcurve::TrimFace<'_>,
    ) -> Result<Vec<Vec<solidcore_record::brep::Curve2>>, String> {
        use solidcore_ops::pcurve::{EdgeUse, pcurves_within};
        let mut loops: Vec<Vec<EdgeUse<'_>>> = Vec::with_capacity(face.loops.len());
        for loop_ in &face.loops {
            let mut uses = Vec::with_capacity(loop_.len());
            for use_ in loop_ {
                let edge = face
                    .edges
                    .get(use_.edge as usize)
                    .ok_or_else(|| format!("face use names missing edge {}", use_.edge))?;
                uses.push(EdgeUse {
                    edge: use_.edge,
                    curve: edge.curve,
                    interval: edge.interval,
                    forward: use_.forward,
                    tolerance: edge.tolerance,
                    given: use_.pcurve.as_ref(),
                });
            }
            loops.push(uses);
        }
        pcurves_within(face.surface, &loops, face.tolerance, Some(face.domain))
            .map_err(|error| error.to_string())
    }
}

/// The decode policy: cadmpeg's tight service profile under our input cap.
fn policy() -> DecodePolicy {
    DecodePolicy {
        mode: DecodeMode::Salvage,
        limits: ResourceLimits {
            max_input_bytes: MAX_INPUT_BYTES,
            ..ResourceLimits::service()
        },
    }
}

/// The codec cadmpeg detects from the bytes, else the one the extension forces.
fn detect<'a>(
    catalog: &'a InputCatalog,
    policy: &DecodePolicy,
    extension: &str,
    bytes: &[u8],
) -> Result<Option<&'a dyn Codec>, String> {
    let arena = DecodeArena::new();
    let (ctx, _) =
        DecodeContext::from_root_bytes(&[], &arena, policy).map_err(|e| e.to_string())?;
    let prefix = bytes.get(..DETECTION_PREFIX_LEN).unwrap_or(bytes);
    let resolved = catalog.resolve_seekable_source(
        &ctx,
        View::over_retained(prefix),
        &mut Cursor::new(bytes),
        None,
    );
    let resolved = match resolved {
        Ok(ResolvedSource::Unrecognized | ResolvedSource::Cadir)
        | Err(ResolveSourceError::Ambiguous(_)) => {
            match forced_input(extension).map_err(|e| e.to_string())? {
                Some(forced) => {
                    catalog.resolve_source(&ctx, View::over_retained(prefix), Some(forced))
                }
                None => Ok(ResolvedSource::Unrecognized),
            }
        }
        other => other,
    };
    ctx.finish_session().map_err(|e| e.to_string())?;
    match resolved.map_err(|e| e.to_string())? {
        ResolvedSource::Native { codec, .. } => Ok(Some(codec)),
        _ => Ok(None),
    }
}

fn decode_at(
    name: &str,
    extension: &str,
    bytes: &[u8],
    accepted: &[&str],
) -> Result<(String, DecodeResult), String> {
    let policy = policy();
    let catalog = InputCatalog::with_builtins();
    let codec = detect(&catalog, &policy, extension, bytes)
        .map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: unrecognised file contents"))?;
    let format = codec.id().as_str().to_string();
    if !accepted.contains(&format.as_str()) {
        return Err(format!(
            "{name}: {format} files are not supported (STEP .step/.stp, FreeCAD .FCStd and Rhino .3dm only)"
        ));
    }
    let options = DecodeOptions {
        depth: DecodeDepth::Full,
        policy,
    };
    let decoded = codec
        .decode(&mut Cursor::new(bytes), &options)
        .map_err(|error| match error {
            DecodeFailure::Codec(error) => {
                format!("{name}: could not read this {format} file: {error}")
            }
            other => format!("{name}: {other}"),
        })?;
    Ok((format, decoded))
}

/// The matrix product `left * right` of two affine rows matrices.
fn matrix_mul(left: [[f64; 4]; 4], right: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            out[row][column] = (0..4).map(|k| left[row][k] * right[k][column]).sum();
        }
    }
    out
}

fn column_major(rows: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    (rows != identity).then(|| {
        let mut out = [[0.0; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[column][row] = rows[row][column];
            }
        }
        out
    })
}

/// The occurrence placements of one definition, world-frame.
struct Instances<'a> {
    by_definition: BTreeMap<&'a str, Vec<OccurrenceWorld<'a>>>,
}

struct OccurrenceWorld<'a> {
    occurrence: &'a Occurrence,
    rows: [[f64; 4]; 4],
}

impl<'a> Instances<'a> {
    fn of(ir: &'a CadIr) -> (Self, usize, usize, Vec<String>) {
        let definitions: BTreeMap<&str, &ProductDefinition> = ir
            .model
            .product_definitions
            .iter()
            .map(|definition| (definition.id.as_str(), definition))
            .collect();
        let mut warnings = Vec::new();
        let mut unplaced = 0usize;
        let mut world: BTreeMap<&str, [[f64; 4]; 4]> = BTreeMap::new();
        // Resolve root-first; a cycle or a missing parent is a warning, never a panic.
        let mut pending: Vec<&Occurrence> = ir.model.occurrences.iter().collect();
        for _ in 0..=pending.len() {
            let mut progressed = false;
            let mut next = Vec::new();
            for occurrence in pending {
                let rows = match &occurrence.parent {
                    OccurrenceParent::Root {} => Some(occurrence.transform.rows()),
                    OccurrenceParent::Occurrence { occurrence: parent } => world
                        .get(parent.as_str())
                        .map(|parent_rows| matrix_mul(*parent_rows, occurrence.transform.rows())),
                };
                match rows {
                    Some(rows) => {
                        world.insert(occurrence.id.as_str(), rows);
                        progressed = true;
                    }
                    None => next.push(occurrence),
                }
            }
            pending = next;
            if pending.is_empty() || !progressed {
                break;
            }
        }
        for occurrence in &pending {
            unplaced += 1;
            warnings.push(format!(
                "assembly occurrence {:?}: its parent occurrence was not resolved; the instance was not placed",
                occurrence.name.clone().unwrap_or_else(|| occurrence.id.to_string())
            ));
        }
        let mut by_definition: BTreeMap<&str, Vec<OccurrenceWorld<'_>>> = BTreeMap::new();
        for occurrence in &ir.model.occurrences {
            let rows = match world.get(occurrence.id.as_str()) {
                Some(rows) => *rows,
                None => continue,
            };
            match &occurrence.prototype {
                PrototypeReference::Local { definition } => {
                    let Some(definition) = definitions.get(definition.as_str()) else {
                        unplaced += 1;
                        warnings.push(format!(
                            "assembly occurrence {:?}: its product definition was not found; the instance was not placed",
                            occurrence.name.clone().unwrap_or_else(|| occurrence.id.to_string())
                        ));
                        continue;
                    };
                    by_definition
                        .entry(definition.id.as_str())
                        .or_default()
                        .push(OccurrenceWorld { occurrence, rows });
                }
                PrototypeReference::External { document, .. } => {
                    unplaced += 1;
                    let document = match document {
                        cadmpeg_ir::products::ExternalDocument::Path { path } => path.to_string(),
                        cadmpeg_ir::products::ExternalDocument::DocumentId { document_id } => {
                            document_id.to_string()
                        }
                        cadmpeg_ir::products::ExternalDocument::Missing {} => {
                            "an unnamed document".to_string()
                        }
                    };
                    warnings.push(format!(
                        "assembly occurrence {:?}: external document {document} is not resolved by CADIR; the instance was not placed",
                        occurrence.name.clone().unwrap_or_else(|| occurrence.id.to_string())
                    ));
                }
                PrototypeReference::Unresolved {} => {
                    unplaced += 1;
                    warnings.push(format!(
                        "assembly occurrence {:?}: its prototype is unresolved; the instance was not placed",
                        occurrence.name.clone().unwrap_or_else(|| occurrence.id.to_string())
                    ));
                }
            }
        }
        (
            Self { by_definition },
            ir.model.occurrences.len(),
            unplaced,
            warnings,
        )
    }

    fn for_definition(&self, definition: &str) -> &[OccurrenceWorld<'a>] {
        self.by_definition
            .get(definition)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// The definition that owns `body`, with the names on its path from the root.
fn owner<'a>(ir: &'a CadIr, body: &str) -> Option<(&'a ProductDefinition, Vec<String>)> {
    let definition = ir
        .model
        .product_definitions
        .iter()
        .find(|definition| definition.bodies.iter().any(|id| id.as_str() == body))?;
    let mut path = Vec::new();
    if let Some(label) = definition
        .label
        .clone()
        .or_else(|| definition.source_name.clone())
        .filter(|label| !label.trim().is_empty())
    {
        path.push(label);
    }
    Some((definition, path))
}

/// The colour CADIR recorded for `body`: its own, else the one colour every face it
/// owns carries (STEP styling is per face, and exporters commonly style every face).
fn body_color(ir: &CadIr, body: &cadmpeg_ir::topology::Body) -> Option<[f64; 3]> {
    if let Some(color) = body.color {
        return Some([
            f64::from(color.r()),
            f64::from(color.g()),
            f64::from(color.b()),
        ]);
    }
    let faces: BTreeMap<&str, &cadmpeg_ir::topology::Face> = ir
        .model
        .faces
        .iter()
        .map(|face| (face.id.as_str(), face))
        .collect();
    let shells: BTreeMap<&str, &cadmpeg_ir::topology::Shell> = ir
        .model
        .shells
        .iter()
        .map(|shell| (shell.id.as_str(), shell))
        .collect();
    let regions: BTreeMap<&str, &cadmpeg_ir::topology::Region> = ir
        .model
        .regions
        .iter()
        .map(|region| (region.id.as_str(), region))
        .collect();
    let mut found: Option<[f64; 3]> = None;
    for region in body
        .regions
        .iter()
        .filter_map(|id| regions.get(id.as_str()))
    {
        for shell in region
            .shells
            .iter()
            .filter_map(|id| shells.get(id.as_str()))
        {
            for face in shell.faces().iter().filter_map(|id| faces.get(id.as_str())) {
                let Some(color) = face.color else { continue };
                let color = [
                    f64::from(color.r()),
                    f64::from(color.g()),
                    f64::from(color.b()),
                ];
                match found {
                    Some(previous) if previous != color => return None,
                    Some(_) => {}
                    None => found = Some(color),
                }
            }
        }
    }
    found
}

#[derive(Clone, Copy, Default)]
struct SurfaceKinds {
    planes: usize,
    cylinders: usize,
    cones: usize,
    spheres: usize,
    tori: usize,
    bsplines: usize,
    other: usize,
}

impl SurfaceKinds {
    fn add(&mut self, other: Self) {
        self.planes += other.planes;
        self.cylinders += other.cylinders;
        self.cones += other.cones;
        self.spheres += other.spheres;
        self.tori += other.tori;
        self.bsplines += other.bsplines;
        self.other += other.other;
    }

    fn faces(self) -> usize {
        self.planes + self.cylinders + self.cones + self.spheres + self.tori + self.bsplines
            + self.other
    }
}

fn surface_kinds(geometry: &cadmpeg_ir::geometry::SolvedSurfaceGeometry) -> SurfaceKinds {
    use cadmpeg_ir::geometry::SolvedSurfaceGeometry;
    match geometry {
        SolvedSurfaceGeometry::Plane(_) => SurfaceKinds {
            planes: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Cylinder(_) => SurfaceKinds {
            cylinders: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Cone(_) => SurfaceKinds {
            cones: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Sphere(_) => SurfaceKinds {
            spheres: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Torus(_) => SurfaceKinds {
            tori: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Nurbs(_) => SurfaceKinds {
            bsplines: 1,
            ..SurfaceKinds::default()
        },
        SolvedSurfaceGeometry::Transformed(placed) => surface_kinds(placed.basis()),
        _ => SurfaceKinds {
            other: 1,
            ..SurfaceKinds::default()
        },
    }
}

fn face_kinds(ir: &CadIr) -> BTreeMap<&str, SurfaceKinds> {
    let surfaces: BTreeMap<&str, SurfaceKinds> = ir
        .model
        .surfaces
        .iter()
        .map(|surface| {
            let kinds = surface
                .geometry
                .solved()
                .map(surface_kinds)
                .unwrap_or_default();
            (surface.id.as_str(), kinds)
        })
        .collect();
    ir.model
        .faces
        .iter()
        .map(|face| {
            (
                face.id.as_str(),
                surfaces
                    .get(face.surface.as_str())
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect()
}

fn type_reports(ir: &CadIr) -> (CadTypeReport, Vec<CadSolidTypeReport>) {
    let faces = face_kinds(ir);
    let regions: BTreeMap<&str, &cadmpeg_ir::topology::Region> = ir
        .model
        .regions
        .iter()
        .map(|region| (region.id.as_str(), region))
        .collect();
    let shells: BTreeMap<&str, &cadmpeg_ir::topology::Shell> = ir
        .model
        .shells
        .iter()
        .map(|shell| (shell.id.as_str(), shell))
        .collect();
    let mut total = SurfaceKinds::default();
    let mut solids = Vec::with_capacity(ir.model.bodies.len());
    for (index, body) in ir.model.bodies.iter().enumerate() {
        let mut kinds = SurfaceKinds::default();
        for region in body
            .regions
            .iter()
            .filter_map(|id| regions.get(id.as_str()))
        {
            for shell in region
                .shells
                .iter()
                .filter_map(|id| shells.get(id.as_str()))
            {
                for face in shell.faces() {
                    kinds.add(faces.get(face.as_str()).copied().unwrap_or_default());
                }
            }
        }
        total.add(kinds);
        solids.push(CadSolidTypeReport {
            solid: index as u32,
            faces: kinds.faces(),
            planes: kinds.planes,
            cylinders: kinds.cylinders,
            cones: kinds.cones,
            spheres: kinds.spheres,
            tori: kinds.tori,
            bsplines: kinds.bsplines,
            faceted: kinds.faces() == 0,
        });
    }
    (
        CadTypeReport {
            solids: ir.model.bodies.len(),
            faces: total.faces(),
            planes: total.planes,
            cylinders: total.cylinders,
            cones: total.cones,
            spheres: total.spheres,
            tori: total.tori,
            bspline_surfaces: total.bsplines,
            faceted_breps: ir.model.tessellations.len(),
            other_surfaces: total.other,
            assemblies: !ir.model.occurrences.is_empty(),
        },
        solids,
    )
}

/// Read the STEP schema identifier from a CAD file's decoded header.
///
/// The identifier comes from cadmpeg's classified source metadata, never from text
/// parsed here. Non-STEP formats return `None`.
pub fn read_schema(path: &str) -> Result<Option<String>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !CAD_EXTS.contains(&extension.as_str()) {
        return Err(format!(
            "{path}: not supported (STEP .step/.stp, FreeCAD .FCStd and Rhino .3dm only)"
        ));
    }
    let policy = policy();
    let catalog = InputCatalog::with_builtins();
    let codec = detect(&catalog, &policy, &extension, &bytes)
        .map_err(|error| format!("{path}: {error}"))?
        .ok_or_else(|| format!("{path}: unrecognised file contents"))?;
    let options = DecodeOptions {
        depth: DecodeDepth::Container,
        policy,
    };
    let decoded = codec
        .decode(&mut Cursor::new(&bytes), &options)
        .map_err(|error| format!("{path}: {error}"))?;
    let (ir, _, _) = decoded.into_parts();
    Ok(ir.source.and_then(|source| {
        source
            .attributes
            .iter()
            .find(|(key, _)| key.as_str() == "schema")
            .map(|(_, value)| value.clone())
    }))
}

/// The application protocol a STEP schema identifier names.
pub fn application_protocol(schema: &str) -> Option<&'static str> {
    let schema = schema.to_ascii_uppercase();
    if schema.contains("CONFIG_CONTROL_DESIGN") || schema.contains("AP203") {
        Some("AP203")
    } else if schema.contains("AUTOMOTIVE_DESIGN") || schema.contains("AP214") {
        Some("AP214")
    } else if schema.contains("AP242") {
        Some("AP242")
    } else {
        None
    }
}

/// Read `bytes` (a file called `name`) through cadmpeg and `solidcore-cadir`.
///
/// Panics inside cadmpeg or the reader become errors. A body region that cannot be
/// read is reported in [`CadImport::failures`]; the other regions still import.
pub fn read_cad(bytes: &[u8], name: &str) -> Result<CadImport, String> {
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!(
            "{name}: file is {} MiB, larger than the {} MiB import limit",
            bytes.len() >> 20,
            MAX_INPUT_BYTES >> 20
        ));
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_cad_inner(bytes, name))).map_err(
        |panic| {
            let detail = panic
                .downcast_ref::<&str>()
                .map(|detail| (*detail).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            format!("{name}: cadmpeg import panicked: {detail}")
        },
    )?
}

fn read_cad_inner(bytes: &[u8], name: &str) -> Result<CadImport, String> {
    let extension = std::path::Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if !CAD_EXTS.contains(&extension.as_str()) {
        return Err(format!(
            "{name}: not supported (STEP .step/.stp, FreeCAD .FCStd and Rhino .3dm only)"
        ));
    }
    let (format, decoded) = decode_at(name, &extension, bytes, &ACCEPTED)?;
    let policy = policy();
    let (ir, report, _) = decoded.into_parts();
    if !report.geometry_transferred() {
        return Err(format!(
            "{name}: no geometry could be read from this {format} file"
        ));
    }
    let mut warnings: Vec<String> = report
        .losses
        .iter()
        .filter(|loss| !format!("{}", loss.code).starts_with("step/decode.opaque-record-preserved"))
        .map(|loss| format!("{}: {}", loss.code, loss.message))
        .collect();

    // Validate the neutral model, as cadmpeg's own conversion does.
    let arena = DecodeArena::new();
    let ctx = DecodeContext::for_loaded_input(&arena, &policy, bytes.len() as u64)
        .map_err(|error| error.to_string())?;
    let validation =
        validate_neutral_for_decode(&ctx, &ir, Vec::new()).map_err(|error| error.to_string())?;
    ctx.finish_session().map_err(|error| error.to_string())?;
    warnings.extend(
        validation
            .findings
            .iter()
            .map(|finding| format!("{}: {}", finding.severity, finding.message)),
    );
    if !validation.is_ok() {
        return Err(format!(
            "{name}: the {format} model is invalid ({} errors)",
            validation.error_count()
        ));
    }

    let (instance_index, instance_count, unplaced_instances, instance_warnings) =
        Instances::of(&ir);
    warnings.extend(instance_warnings);
    let tolerance = solidcore_cadir::tolerance(&ir);
    let (type_report, solid_types) = type_reports(&ir);
    let read = solidcore_cadir::read_bodies(&ir, Some(&KernelTrims));
    let by_id: BTreeMap<&str, &cadmpeg_ir::topology::Body> = ir
        .model
        .bodies
        .iter()
        .map(|body| (body.id.as_str(), body))
        .collect();

    let mut bodies = Vec::with_capacity(read.read.len());
    let mut placed_instances = 0usize;
    for region in read.read {
        let Some(body) = by_id.get(region.body.as_str()) else {
            continue;
        };
        let (definition, path) = match owner(&ir, &region.body) {
            Some(owner) => (Some(owner.0), owner.1),
            None => (None, Vec::new()),
        };
        let worlds = definition
            .map(|definition| instance_index.for_definition(definition.id.as_str()))
            .unwrap_or(&[]);
        let body_name = body.name.clone().filter(|name| !name.trim().is_empty());
        let definition_name = definition.and_then(|definition| {
            definition
                .label
                .clone()
                .or_else(|| definition.source_name.clone())
                .filter(|name| !name.trim().is_empty())
        });
        let color = body_color(&ir, body);
        let base = body_name
            .or(definition_name)
            .unwrap_or_else(|| region.body.clone());
        let body_placement = body.transform.map(Transform::rows);
        if worlds.is_empty() {
            // No occurrence places this body: keep it where its own placement says.
            if !ir.model.occurrences.is_empty() {
                warnings.push(format!(
                    "body {:?} is not placed by any assembly occurrence; it was imported in its own frame",
                    base
                ));
            }
            bodies.push(CadBody {
                id: region.body.clone(),
                region: region.region,
                name: base,
                product: path,
                color,
                hidden: body.visible == Some(false),
                placement: body_placement.and_then(column_major),
                brep: region.solid.brep,
                face_origin: region.solid.face_origin,
                sheet: region.solid.sheet,
                notes: region.solid.notes,
                turned: region.solid.turned,
            });
            continue;
        }
        placed_instances += 1;
        for world in worlds {
            let name = world
                .occurrence
                .name
                .clone()
                .filter(|name| !name.trim().is_empty())
                .unwrap_or_else(|| base.clone());
            let mut rows = world.rows;
            if let Some(placement) = body_placement {
                rows = matrix_mul(rows, placement);
            }
            let mut instance_path = path.clone();
            if let Some(instance) = world
                .occurrence
                .name
                .clone()
                .filter(|name| !name.trim().is_empty())
            {
                instance_path.push(instance);
            }
            bodies.push(CadBody {
                id: region.body.clone(),
                region: region.region,
                name,
                product: instance_path,
                color,
                hidden: body.visible == Some(false),
                placement: column_major(rows),
                brep: region.solid.brep.clone(),
                face_origin: region.solid.face_origin.clone(),
                sheet: region.solid.sheet,
                notes: region.solid.notes.clone(),
                turned: region.solid.turned.clone(),
            });
        }
    }
    let failures: Vec<CadFailure> = read
        .failures
        .iter()
        .map(|failure| CadFailure {
            id: failure.body.clone(),
            region: failure.region,
            message: failure.message.clone(),
        })
        .collect();
    for failure in &failures {
        warnings.push(format!(
            "body {:?} region {} was not read: {}",
            failure.id, failure.region, failure.message
        ));
    }
    let system = match format.as_str() {
        "fcstd" => "FreeCAD".to_string(),
        "rhino" => "Rhino".to_string(),
        "step" => "STEP".to_string(),
        other => other.to_string(),
    };
    let file_name = std::path::Path::new(name)
        .file_stem()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| name.to_string());
    Ok(CadImport {
        file_name,
        format,
        system,
        tolerance,
        bodies,
        failures,
        warnings,
        instances: instance_count,
        placed_instances,
        unplaced_instances,
        type_report,
        solid_types,
    })
}
