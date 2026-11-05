//! Display tessellation for curved solidcore faces (cylinder, cone, sphere, torus).
//!
//! [`nodecraft_solidcore::box_mesh`] handles the planar box through the B-rep boundary loops. This
//! module adds the cheapest viable display path for faces with curved surfaces using
//! only solidcore public query APIs:
//!
//! * [face UV domains](solidcore::topology::face::p2::uv_domain),
//! * the face's UV trim curves
//!   ([`uv_trim_curve`](solidcore::topology::edgeuse::p1::uv_trim_curve)) sampled
//!   through [`evaluate_curve`](solidcore::query::face::p1::evaluate_curve),
//! * surface point/normal sampling
//!   ([`Surface::evaluate_point`](solidcore::surface::surface::Surface::evaluate_point),
//!   [`Surface::evaluate_normal`](solidcore::surface::surface::Surface::evaluate_normal)).
//!
//! It deliberately does **not** use `SmApiTessellate` / `SmApiPoly*` (out of solidcore
//! scope, see `plan/closure.md`) or `query::tess::TessSrfCache` (it needs poly-B-rep
//! topology that the port does not build yet). See `CURVED_NOTES.md` for the exact
//! remaining gap.
//!
//! # Strategy
//!
//! Faces are split by shape, not by surface class:
//!
//! * **Planar faces** (cylinder/cone caps, box-like faces): sample the boundary loops
//!   in 3D and ear-clip the outer loop with its holes. Windings are oriented against
//!   the face's outward surface normal.
//! * **Curved faces** (cylinder/cone lateral surface, sphere, torus): sample the
//!   face's UV rectangle on a grid. The primitive curved faces are *untrimmed* (their
//!   trim loops lie entirely on the UV-domain boundary: seams and end circles), so the
//!   whole rectangle is filled. A face whose loops run through the interior is clipped
//!   cell by cell in (u, v): whole cells are in or out by a crossing test and a cell
//!   the boundary crosses is cut along the loop and ear-clipped ([`trim`]), so the face
//!   ends exactly at its trim, not at the last cell wholly inside it.
//!
//! Every face samples a boundary edge at the same parameters (one uniform set per
//! edge), so two faces sharing an edge start from the same 3D points. After all faces
//! are tessellated, [`stitch`] welds coincident vertices, splits T-junctions, caps the
//! small slits three trimmed faces can leave at a corner, and orients each shell so
//! the display mesh closes and faces outward. Normals are evaluated per vertex on the
//! surface and recomputed from the final triangles where welding merged a seam.

pub mod step_import;
mod stitch;
mod trim;
pub use step_import::{import_step, import_step_brep, merge_meshes, StepDiagnostics, StepImport};

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use solidcore::kernel::Kernel;
use solidcore::math::{Point2, Point3, Vec3};
use solidcore::nurbs::b_spline_surface_n_lib::p2::OrientType;
use solidcore::nurbs::knots::Side;
use solidcore::primitive::api_primitives as api;
use solidcore::query::face::p1::evaluate_curve;
use solidcore::nurbs::b_spline_surface::p1::SurfParamType;
use solidcore::surface::surface::p2::SurfaceKind;
use solidcore::surface::surface::Surface;
pub(crate) use nurbs_grid_mesh;
use nurbs_grid_mesh::{mesh_trimmed, LoopPoint, MeshOptions, NurbsSurface};
use solidcore::topology::edgeuse::p1 as eu;
use solidcore::topology::face::p2 as fp2;
use solidcore::topology::face::p3 as fp3;
use solidcore::topology::trimming_tools;
use solidcore::Brep;
use nodecraft_protocol::{BrepEdge, Mesh};
use nodecraft_step::PlanarBody;

/// Two coordinates closer than this are treated as the same point.
const POINT_EPS: f64 = 1.0e-9;

thread_local! {
    /// Display max-sag override for this evaluation (LOD). The engine sets it per
    /// request; the default is the normal viewport quality.
    static DISPLAY_TOLERANCE: Cell<Option<f64>> = const { Cell::new(None) };
    /// False inside an outer body-level mesher worker, where spawning another
    /// face-level pool would oversubscribe the machine.
    static FACE_PARALLEL: Cell<bool> = const { Cell::new(true) };
}

/// Enable or disable nested face-level worker threads on this thread.
pub fn set_face_parallel(enabled: bool) {
    FACE_PARALLEL.with(|slot| slot.set(enabled));
}

fn face_parallel_enabled() -> bool {
    FACE_PARALLEL.with(Cell::get)
}

/// Set the max-sag used by [`CurvedTessOptions::default`] for the current thread.
pub fn set_display_tolerance(tolerance: Option<f64>) {
    DISPLAY_TOLERANCE.with(|slot| slot.set(tolerance.filter(|value| value.is_finite() && *value > 0.0)));
}

/// Current display max-sag in model units.
#[must_use]
pub fn display_tolerance() -> f64 {
    DISPLAY_TOLERANCE.with(Cell::get).unwrap_or(DEFAULT_DISPLAY_TOLERANCE)
}

/// Normal viewport max sag.
pub const DEFAULT_DISPLAY_TOLERANCE: f64 = 0.35;

/// Grid density for the curved-face display mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvedTessOptions {
    /// Grid segments across the face's first (U) parameter direction. For all
    /// primitive curved faces this is the "around the axis" direction.
    pub u_segments: usize,
    /// Grid segments across the face's second (V) parameter direction.
    pub v_segments: usize,
    /// Minimum polyline segments for one boundary edge; the actual count is increased
    /// until the edge's chord sag meets `bspline_tolerance`.
    pub loop_segments: usize,
    /// Chord tolerance for B-spline trimmed faces, in model units.
    pub bspline_tolerance: f64,
    /// Maximum normal turn for curved analytic faces. Zero keeps the explicit
    /// `u_segments`/`v_segments` grid; imports set this to the Shapr3D-like 4 degrees.
    pub max_normal_angle_deg: f64,
}

impl Default for CurvedTessOptions {
    fn default() -> Self {
        Self {
            u_segments: 16,
            v_segments: 4,
            loop_segments: 8,
            bspline_tolerance: display_tolerance(),
            max_normal_angle_deg: 0.0,
        }
    }
}

/// Shared 3D samples for each topological B-rep edge, keyed by edge-list ordinal.
type EdgeSampleCache = BTreeMap<usize, Vec<[f64; 3]>>;

/// A solidcore cylinder converted to the centered NodeCraft primitive convention.
///
/// The cylinder is created about +Z from the origin and re-centered on its bounding
/// box, matching [`nodecraft_solidcore::box_mesh`].
pub fn cylinder_mesh(radius: f64, height: f64) -> Result<Mesh, String> {
    cylinder_mesh_with_options(radius, height, &CurvedTessOptions::default())
}

/// [`cylinder_mesh`] with an explicit display grid density.
pub fn cylinder_mesh_with_options(
    radius: f64,
    height: f64,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), radius, height)
        .map_err(|error| format!("solidcore create_cylinder: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, options)?)
}

/// A solidcore cone converted to the centered NodeCraft primitive convention.
pub fn cone_mesh(radius_base: f64, radius_top: f64, height: f64) -> Result<Mesh, String> {
    cone_mesh_with_options(
        radius_base,
        radius_top,
        height,
        &CurvedTessOptions::default(),
    )
}

/// [`cone_mesh`] with an explicit display grid density.
pub fn cone_mesh_with_options(
    radius_base: f64,
    radius_top: f64,
    height: f64,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_cone(
        &mut kernel,
        Point3::new(0.0, 0.0, 0.0),
        radius_base,
        radius_top,
        height,
    )
    .map_err(|error| format!("solidcore create_cone: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, options)?)
}

/// A solidcore sphere converted to the centered NodeCraft primitive convention.
pub fn sphere_mesh(radius: f64) -> Result<Mesh, String> {
    sphere_mesh_with_options(radius, &CurvedTessOptions::default())
}

/// [`sphere_mesh`] with an explicit display grid density.
pub fn sphere_mesh_with_options(
    radius: f64,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), radius)
        .map_err(|error| format!("solidcore create_sphere: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, options)?)
}

/// A solidcore torus converted to the centered NodeCraft primitive convention.
pub fn torus_mesh(major_radius: f64, minor_radius: f64) -> Result<Mesh, String> {
    torus_mesh_with_options(major_radius, minor_radius, &CurvedTessOptions::default())
}

/// [`torus_mesh`] with an explicit display grid density.
pub fn torus_mesh_with_options(
    major_radius: f64,
    minor_radius: f64,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_torus(
        &mut kernel,
        Point3::new(0.0, 0.0, 0.0),
        major_radius,
        minor_radius,
    )
    .map_err(|error| format!("solidcore create_torus: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, options)?)
}

/// A solidcore square pyramid converted to the centered NodeCraft convention.
pub fn pyramid_mesh(length: f64, height: f64) -> Result<Mesh, String> {
    pyramid_mesh_with_options(length, height, &CurvedTessOptions::default())
}

/// [`pyramid_mesh`] with an explicit display grid density.
pub fn pyramid_mesh_with_options(
    length: f64,
    height: f64,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_pyramid(
        &mut kernel,
        Point3::new(0.0, 0.0, 0.0),
        length,
        height,
    )
    .map_err(|error| format!("solidcore create_pyramid: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, options)?)
}

/// A solidcore pole-free sphere converted to the centered NodeCraft convention.
pub fn sphere_no_pole_mesh(radius: f64) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_sphere_no_pole(&mut kernel, Point3::new(0.0, 0.0, 0.0), radius)
        .map_err(|error| format!("solidcore create_sphere_no_pole: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, &CurvedTessOptions::default())?)
}

/// A solidcore spherical sector converted to the centered NodeCraft convention.
pub fn partial_sphere_mesh(
    radius: f64,
    start_angle_deg: f64,
    end_angle_deg: f64,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_partial_sphere(
        &mut kernel,
        Point3::new(0.0, 0.0, 0.0),
        radius,
        start_angle_deg,
        end_angle_deg,
    )
    .map_err(|error| format!("solidcore create_partial_sphere: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, &CurvedTessOptions::default())?)
}

/// A solidcore partial torus converted to the centered NodeCraft convention.
pub fn partial_torus_mesh(
    major_radius: f64,
    minor_radius: f64,
    start_angle_deg: f64,
    end_angle_deg: f64,
) -> Result<Mesh, String> {
    let mut kernel = Kernel::default();
    let brep = api::create_partial_torus(
        &mut kernel,
        Point3::new(0.0, 0.0, 0.0),
        major_radius,
        minor_radius,
        start_angle_deg,
        end_angle_deg,
    )
    .map_err(|error| format!("solidcore create_partial_torus: {error}"))?;
    centered(brep_display_mesh(&kernel, &brep, &CurvedTessOptions::default())?)
}

/// Tessellate every face of `brep` into a display [`Mesh`] in B-rep coordinates.
///
/// The result is not re-centered; use a convenience wrapper like [`cylinder_mesh`] for
/// the centered NodeCraft convention. Unreferenced seam vertices may appear when a
/// closed surface's UV grid duplicates a seam row.
pub fn brep_display_mesh(
    kernel: &Kernel,
    brep: &Brep,
    options: &CurvedTessOptions,
) -> Result<Mesh, String> {
    brep_display_mesh_with_senses(kernel, brep, options, None)
}

/// [`brep_display_mesh`], with the STEP `same_sense` flag of every face.
///
/// `senses[i]` belongs to `brep.face_list()[i]`, which is the order the caller built the
/// faces in. When the counts disagree (a face was split or removed) the flags cannot be
/// mapped and the mesh keeps the orientation its face uses and stitching gave it.
/// Tessellate independent faces on a bounded worker pool.
///
/// Shared edge polylines are deterministic per edge (same tolerance/segment count), so
/// faces can build local meshes independently and be merged in face order. Stitching,
/// orientation and edge merging happen after the merge.
fn tessellate_faces_parallel(
    kernel: &Kernel,
    brep: &Brep,
    face_ids: &[solidcore::FaceId],
    options: &CurvedTessOptions,
) -> Vec<(usize, Mesh)> {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(face_ids.len().max(1));
    let mut meshes = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                let kernel = Kernel::default();
                let mut local = Vec::new();
                for index in (worker..face_ids.len()).step_by(workers) {
                    let mut mesh = Mesh::default();
                    let mut edge_index = 0usize;
                    let mut edge_cache = EdgeSampleCache::new();
                    match tessellate_face(
                        &kernel,
                        brep,
                        face_ids[index],
                        index,
                        options,
                        &mut mesh,
                        &mut edge_index,
                        &mut edge_cache,
                    ) {
                        Ok(()) => local.push((index, mesh)),
                        Err(error) => eprintln!("face {index}: {error}"),
                    }
                }
                local
            }));
        }
        for handle in handles {
            meshes.extend(handle.join().expect("face mesher thread panicked"));
        }
    });
    meshes.sort_by_key(|(index, _)| *index);
    meshes
}

/// Merge per-face local meshes into one body mesh, preserving face groups and edge IDs.
fn merge_face_meshes(face_meshes: Vec<(usize, Mesh)>) -> Mesh {
    let has_colors = face_meshes.iter().any(|(_, mesh)| !mesh.face_colors.is_empty());
    let mut merged = Mesh::default();
    let mut edges: std::collections::BTreeMap<String, BrepEdge> = std::collections::BTreeMap::new();
    for (face_index, mesh) in face_meshes {
        let offset = merged.vertices.len() as u32;
        merged.vertices.extend(mesh.vertices);
        merged.normals.extend(mesh.normals);
        let face_count = mesh.faces.len();
        merged.faces.extend(mesh.faces.into_iter().map(|face| [face[0] + offset, face[1] + offset, face[2] + offset]));
        merged.face_groups.extend(mesh.face_groups);
        if has_colors {
            if mesh.face_colors.is_empty() {
                merged.face_colors.extend(std::iter::repeat_n([0.0; 3], face_count));
            } else {
                merged.face_colors.extend(mesh.face_colors);
            }
        }
        for mut edge in mesh.brep_edges {
            if edge.id.starts_with("mesh-edge:") {
                edge.id = format!("face:{face_index}:{}", edge.id);
            }
            for segment in &mut edge.segments {
                segment[0] += offset;
                segment[1] += offset;
            }
            let id = edge.id.clone();
            edges.entry(id.clone()).or_insert_with(|| BrepEdge { id, kind: edge.kind.clone(), segments: Vec::new() }).segments.extend(edge.segments);
        }
    }
    // Adjacent faces contribute the same topological edge twice. Deduplicate by
    // geometric endpoints after the vertex offsets are applied; otherwise the edge
    // polyline doubles back on itself and the mitered display edge degenerates.
    let key = |index: u32| {
        let point = merged.vertices[index as usize];
        [
            (point[0] as f64 * 1.0e5).round() as i64,
            (point[1] as f64 * 1.0e5).round() as i64,
            (point[2] as f64 * 1.0e5).round() as i64,
        ]
    };
    merged.brep_edges = edges
        .into_values()
        .filter_map(|mut edge| {
            let mut seen = std::collections::HashSet::new();
            edge.segments.retain(|[a, b]| {
                if (*a as usize) >= merged.vertices.len() || (*b as usize) >= merged.vertices.len() {
                    return false;
                }
                let (a, b) = (key(*a), key(*b));
                seen.insert(if a <= b { (a, b) } else { (b, a) })
            });
            (!edge.segments.is_empty()).then_some(edge)
        })
        .collect();
    merged.lightweight = true;
    merged
}

pub fn brep_display_mesh_with_senses(
    kernel: &Kernel,
    brep: &Brep,
    options: &CurvedTessOptions,
    senses: Option<&[bool]>,
) -> Result<Mesh, String> {
    // B-spline faces from the legacy import path can lack UV trims; create them only
    // on those faces, on a working copy, so the trimmed mesher has loops to clip to.
    let missing_bspline_trims = brep.face_list().iter().any(|&face| {
        matches!(
            fp2::surface(brep, face).ok().flatten().map(Surface::kind),
            Some(solidcore::surface::surface::p2::SurfaceKind::BSpline(_))
        ) && !trimming_tools::face_uv_curves_present(brep, face).unwrap_or(false)
    });
    let mut trimmed_brep = None;
    if missing_bspline_trims {
        let mut copy = brep
            .clone_standing_for_self()
            .map_err(|error| format!("display mesh copy: {error}"))?;
        for face in copy.face_list().to_vec() {
            if matches!(
                fp2::surface(&copy, face).ok().flatten().map(Surface::kind),
                Some(solidcore::surface::surface::p2::SurfaceKind::BSpline(_))
            ) && !trimming_tools::face_uv_curves_present(&copy, face).unwrap_or(false)
            {
                let _ = fp3::create_uv_trim_curves(kernel, &mut copy, face, false, None);
            }
        }
        trimmed_brep = Some(copy);
    }
    let brep = trimmed_brep.as_ref().unwrap_or(brep);
    // Tessellate faces independently. The B-rep face list is the authority: boolean
    // and topology edits can leave stale arena slots, and tessellating those would add
    // geometry that is no longer part of the solid.
    let face_ids = brep.face_list().to_vec();
    let face_meshes = if face_ids.len() >= 16 && face_parallel_enabled() {
        tessellate_faces_parallel(kernel, brep, &face_ids, options)
    } else {
        let mut local = Vec::with_capacity(face_ids.len());
        for (face_index, &face_id) in face_ids.iter().enumerate() {
            let mut mesh = Mesh::default();
            let mut edge_index = 0usize;
            let mut edge_cache = EdgeSampleCache::new();
            match tessellate_face(kernel, brep, face_id, face_index, options, &mut mesh, &mut edge_index, &mut edge_cache) {
                Ok(()) => local.push((face_index, mesh)),
                Err(error) => eprintln!("face {face_index}: {error}"),
            }
        }
        local
    };
    let mut mesh = merge_face_meshes(face_meshes);
    // Put every face on the side the STEP file gives it first; the stitching pass then
    // makes neighbouring faces agree and picks each shell's outward sign. Doing it the
    // other way round lets a face that disagrees with the file's `same_sense` stay
    // inverted, and a single large inverted face corrupts the signed volume.
    if let Some(senses) = senses {
        orient_mesh_to_senses(kernel, brep, &mut mesh, senses);
    }
    // Shared edge samples put adjacent face boundaries on identical positions. The
    // spatial stitch below is residual cleanup for T-junctions and tiny corner gaps, not
    // the primary edge-alignment mechanism. `NODECRAFT_DISABLE_WELD=1` is a build-time
    // inspection switch; the implementation stays compiled either way.
    const ENABLE_DISPLAY_STITCH: bool = option_env!("NODECRAFT_DISABLE_WELD").is_none();
    if ENABLE_DISPLAY_STITCH {
        stitch::stitch_display_mesh(&mut mesh, kernel, brep);
        orient_closed_shells(&mut mesh);
    }
    if mesh.faces.is_empty() {
        return Err("solidcore curved tessellation produced no triangles".into());
    }
    Ok(mesh)
}

/// Wind every face group so its triangles face the direction STEP gives it.
///
/// `brep_display_mesh` already follows the face-use orientation, but stitching can expose
/// the downward use for a face and the B-rep face list is not guaranteed to keep the
/// file's face order once a face has been split. The file's `same_sense` flag is the
/// authority: it says whether the outward normal agrees with the surface normal.
fn orient_mesh_to_senses(kernel: &Kernel, brep: &Brep, mesh: &mut Mesh, senses: &[bool]) {
    let faces = brep.face_list();
    if faces.len() != senses.len() {
        return;
    }
    for (index, (&face_id, &same_sense)) in faces.iter().zip(senses).enumerate() {
        let Ok(domain) = fp2::uv_domain(brep, face_id) else {
            continue;
        };
        let Ok(Some(surface)) = fp2::surface(brep, face_id) else {
            continue;
        };
        let u = (domain.u_interval().min() + domain.u_interval().max()) * 0.5;
        let v = (domain.v_interval().min() + domain.v_interval().max()) * 0.5;
        let Ok(normal) = surface.evaluate_normal(kernel, Point2::new(u, v), Side::Left, Side::Left)
        else {
            continue;
        };
        let Ok(normal) = normal.unitize() else {
            continue;
        };
        let desired = if same_sense {
            normal
        } else {
            Vec3::new(-normal.x, -normal.y, -normal.z)
        };
        let group = format!("face:{index}");
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        for (triangle, triangle_group) in mesh.faces.iter().zip(mesh.face_groups.iter()) {
            if triangle_group != &group {
                continue;
            }
            let pa = point_from_vertex(mesh.vertices[triangle[0] as usize]);
            let pb = point_from_vertex(mesh.vertices[triangle[1] as usize]);
            let pc = point_from_vertex(mesh.vertices[triangle[2] as usize]);
            let u = Vec3::new(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
            let v = Vec3::new(pc.x - pa.x, pc.y - pa.y, pc.z - pa.z);
            sum = sum + u.cross(v);
        }
        if sum.dot(desired) < 0.0 {
            flip_face_group(mesh, index);
        }
    }
}

/// Reverses the triangles and normals of one `face:{index}` display group.
fn flip_face_group(mesh: &mut Mesh, face_index: usize) {
    let group = format!("face:{face_index}");
    let Mesh { faces, face_groups, normals, .. } = mesh;
    let mut vertices: BTreeSet<u32> = BTreeSet::new();
    for (triangle, triangle_group) in faces.iter_mut().zip(face_groups.iter()) {
        if triangle_group != &group {
            continue;
        }
        triangle.swap(1, 2);
        vertices.extend(triangle.iter().copied());
    }
    for vertex in vertices {
        if let Some(normal) = normals.get_mut(vertex as usize) {
            *normal = [-normal[0], -normal[1], -normal[2]];
        }
    }
}

// ---------------------------------------------------------------------------------
// Per-face dispatch
// ---------------------------------------------------------------------------------

fn tessellate_face(
    kernel: &Kernel,
    brep: &Brep,
    face_id: solidcore::FaceId,
    face_index: usize,
    options: &CurvedTessOptions,
    mesh: &mut Mesh,
    edge_index: &mut usize,
    edge_cache: &mut EdgeSampleCache,
) -> Result<(), String> {
    let surface = fp2::surface(brep, face_id)
        .map_err(|error| format!("solidcore face surface: {error}"))?
        .ok_or_else(|| "face has no surface".to_string())?;
    let face = brep
        .face(face_id)
        .map_err(|error| format!("solidcore face: {error}"))?;
    let domain = fp2::uv_domain(brep, face_id)
        .map_err(|error| format!("solidcore face uv domain: {error}"))?;
    let (u0, u1, v0, v1) = checked_domain(&domain)?;
    let flip_normal = match face.faceuse {
        Some(faceuse) => brep
            .faceuse(faceuse)
            .map_err(|error| format!("solidcore faceuse: {error}"))?
            .orientation
            == OrientType::Opposite,
        None => false,
    };

    let boundary = face_loops(kernel, brep, face_id, options.loop_segments.max(1), options.bspline_tolerance.max(1.0e-4), edge_cache)?;
    if boundary.loops.is_empty() {
        // Partial solids can expose a face whose trim loops are implicit in the UV
        // domain (no explicit boundary loopuses). Render the domain grid instead of
        // failing the whole solid; this is the same untrimmed curved-face path.
        // A B-rep with no shells is not a valid solid, so keep rejecting it.
        if brep.shells().is_empty() {
            return Err("face has no boundary loops and the B-rep has no shells".into());
        }
        return tessellate_curved_face(
            kernel,
            brep,
            face_id,
            surface,
            (u0, u1, v0, v1),
            flip_normal,
            face_index,
            options,
            &boundary,
            mesh,
            edge_index,
        );
    }

    if let Some(result) = tessellate_bspline_face(
        kernel, face_index, surface, flip_normal, &boundary.loops, &boundary.edge_ids, mesh, edge_index, options,
    ) {
        return result;
    }

    let result = if is_planar(kernel, surface, u0, u1, v0, v1) {
        tessellate_planar_face(
            kernel,
            surface,
            (u0, u1, v0, v1),
            flip_normal,
            face_index,
            &boundary.loops,
            &boundary.edge_ids,
            mesh,
            edge_index,
        )
    } else {
        tessellate_curved_face(
            kernel,
            brep,
            face_id,
            surface,
            (u0, u1, v0, v1),
            flip_normal,
            face_index,
            options,
            &boundary,
            mesh,
            edge_index,
        )
    };
    result
}

/// Flip closed shells whose triangles face inward.
///
/// The analytic curved paths build triangles from the surface's UV frame; whether that
/// frame points out of the solid depends on the face use. Stitching makes neighbours
/// agree but does not choose the shell's outward sign. A closed component with negative
/// signed volume is inside-out, which renders dark and exports inverted; flip it and its
/// normals. Open sheets are left alone.
fn orient_closed_shells(mesh: &mut Mesh) {
    let count = mesh.vertices.len();
    if count == 0 || mesh.faces.is_empty() {
        return;
    }
    let mut parent: Vec<usize> = (0..count).collect();
    fn find(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        index
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut edge_use: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
    for face in &mesh.faces {
        union(&mut parent, face[0] as usize, face[1] as usize);
        union(&mut parent, face[1] as usize, face[2] as usize);
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let key = (a.min(b), a.max(b));
            *edge_use.entry(key).or_insert(0) += 1;
        }
    }
    let mut volume: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for face in &mesh.faces {
        let a = mesh.vertices[face[0] as usize];
        let b = mesh.vertices[face[1] as usize];
        let c = mesh.vertices[face[2] as usize];
        let cross = [
            b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64,
            b[2] as f64 * c[0] as f64 - b[0] as f64 * c[2] as f64,
            b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64,
        ];
        let signed = (a[0] as f64 * cross[0] + a[1] as f64 * cross[1] + a[2] as f64 * cross[2]) / 6.0;
        *volume.entry(find(&mut parent, face[0] as usize)).or_insert(0.0) += signed;
    }
    let mut closed: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
    for ((a, _b), uses) in edge_use {
        let root = find(&mut parent, a as usize);
        let entry = closed.entry(root).or_insert(true);
        *entry &= uses == 2;
    }
    let mut flip: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
    for face in &mut mesh.faces {
        let root = find(&mut parent, face[0] as usize);
        let should_flip = closed.get(&root).copied().unwrap_or(false)
            && volume.get(&root).copied().unwrap_or(0.0) < 0.0;
        if should_flip {
            face.swap(1, 2);
        }
        flip.insert(root, should_flip);
    }
    for index in 0..mesh.vertices.len() {
        let root = find(&mut parent, index);
        if flip.get(&root).copied().unwrap_or(false) {
            if let Some(normal) = mesh.normals.get_mut(index) {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
            }
        }
    }
}

/// Meshes a B-spline face with the trimmed grid mesher, which clips the grid to the
/// face's loops and puts boundary vertices on the shared edge polylines.
///
/// Returns `None` for non-B-spline surfaces so the caller keeps the existing path.
#[allow(clippy::too_many_arguments)]
/// Largest angle between surface normals sampled around a face's trim loops.
///
/// A bent sheet has a large normal span even when its chord sag is small; using this to
/// tighten the sag near bends keeps the triangles aligned with the bend instead of
/// letting the grid switch triangle diagonal across the curve.
fn boundary_normal_span(kernel: &Kernel, surface: &Surface, loops: &[Vec<LoopPoint>]) -> f64 {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for loop_points in loops {
        for point in loop_points {
            if point.u.is_finite() && point.v.is_finite() {
                min[0] = min[0].min(point.u);
                min[1] = min[1].min(point.v);
                max[0] = max[0].max(point.u);
                max[1] = max[1].max(point.v);
            }
        }
    }
    if !min[0].is_finite() || !max[0].is_finite() {
        return 0.0;
    }
    let samples = 7usize;
    let mut normals = Vec::with_capacity(samples * samples);
    for iv in 0..samples {
        for iu in 0..samples {
            let u = min[0] + (max[0] - min[0]) * iu as f64 / (samples - 1) as f64;
            let v = min[1] + (max[1] - min[1]) * iv as f64 / (samples - 1) as f64;
            let normal = surface
                .evaluate_normal(kernel, Point2::new(u, v), Side::Left, Side::Left)
                .ok()
                .and_then(|normal| normal.unitize().ok());
            normals.push(normal);
        }
    }
    let mut span = 0.0f64;
    for iv in 0..samples {
        for iu in 0..samples {
            let index = iv * samples + iu;
            if let Some(normal) = normals[index] {
                for next in [index + 1, index + samples] {
                    if (next % samples == 0 && next == index + 1) || next >= normals.len() {
                        continue;
                    }
                    if let Some(other) = normals[next] {
                        span = span.max(normal.dot(other).clamp(-1.0, 1.0).acos());
                    }
                }
            }
        }
    }
    span
}

fn tessellate_bspline_face(
    kernel: &Kernel,
    face_index: usize,
    surface: &Surface,
    flip_normal: bool,
    loops: &[Vec<LoopPoint>],
    edge_ids: &[Vec<usize>],
    mesh: &mut Mesh,
    edge_index: &mut usize,
    options: &CurvedTessOptions,
) -> Option<Result<(), String>> {
    let SurfaceKind::BSpline(bspline) = surface.kind() else {
        return None;
    };
    if loops.is_empty() {
        return None;
    }
    let Ok(nurbs) = nurbs_surface(bspline) else {
        return None;
    };
    // Do not use the global display tolerance on a tiny feature: a spike/cone apex
    // has very short trim segments, and a 0.5 mm chord there cuts the tip into steps.
    // Refine relative to the shortest trim segment, capped by the display tolerance.
    // Max-sag display tolerance in model units. The mesher measures cell midpoints
    // against the true surface. Small trimmed features need a proportionally finer sag
    // so the trim boundary is followed, while large faces use the requested maximum.
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for loop_points in loops {
        for point in loop_points {
            for axis in 0..3 { min[axis] = min[axis].min(point.p[axis]); max[axis] = max[axis].max(point.p[axis]); }
        }
    }
    let extent = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    // Max-angle refinement: a 20-degree normal span is the target. Tightening the
    // max-sag proportional to the actual span puts more, axis-aligned grid lines across
    // a bend instead of leaving a few diagonally-split cells with visible sheen.
    let span = boundary_normal_span(kernel, surface, loops);
    let target = 20.0f64.to_radians();
    let bend_factor = (target / span.max(target)).clamp(0.4, 1.0);
    let local = if extent.is_finite() { extent * 0.001 } else { options.bspline_tolerance };
    let tolerance = (local * bend_factor)
        .min(options.bspline_tolerance)
        .max(1.0e-9);
    // Large barrels can need far more than the grid mesher's normal 64 lines
    // per knot span at close LOD; let max-sag decide up to a high ceiling.
    let options = MeshOptions {
        max_lines_per_span: 512,
        ..MeshOptions::new(tolerance)
    };
    let trimmed = match mesh_trimmed(&nurbs, loops, flip_normal, &options) {
        Ok(mesh) => mesh,
        // Closed surfaces with a degenerate seam can defeat the loop closure check;
        // keep the existing grid fallback for those rather than failing the solid.
        Err(_) => return None,
    };
    if trimmed.triangles.is_empty() {
        return None;
    }
    let group = format!("face:{face_index}");
    let base = mesh.vertices.len() as u32;
    let mut normals = Vec::with_capacity(trimmed.positions.len());
    for (position, uv) in trimmed.positions.iter().zip(&trimmed.uvs) {
        let point = Point3::new(position[0], position[1], position[2]);
        let normal = surface
            .evaluate_normal(kernel, Point2::new(uv[0], uv[1]), Side::Left, Side::Left)
            .ok()
            .and_then(|normal| normal.unitize().ok())
            .map(|normal| if flip_normal { -normal } else { normal })
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        normals.push(normal);
        push_vertex(mesh, point, normal);
    }
    for triangle in &trimmed.triangles {
        let [a, b, c] = *triangle;
        if let (Some(&na), Some(&nb), Some(&nc)) = (
            normals.get(a as usize),
            normals.get(b as usize),
            normals.get(c as usize),
        ) {
            let pa = Point3::new(trimmed.positions[a as usize][0], trimmed.positions[a as usize][1], trimmed.positions[a as usize][2]);
            let pb = Point3::new(trimmed.positions[b as usize][0], trimmed.positions[b as usize][1], trimmed.positions[b as usize][2]);
            let pc = Point3::new(trimmed.positions[c as usize][0], trimmed.positions[c as usize][1], trimmed.positions[c as usize][2]);
            let cross = (pb - pa).cross(pc - pa);
            if cross.length() <= 1.0e-9 {
                continue;
            }
            let Some(triangle_normal) = triangle_normal(pa, pb, pc) else { continue };
            let normal = ((na + nb + nc) * (1.0 / 3.0)).unitize().unwrap_or(triangle_normal);
            push_triangle(mesh, base + a, base + b, base + c, normal, &group);
        }
    }
    let fallback = normals.first().copied().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let loops_3d: Vec<Vec<Point3>> = loops
        .iter()
        .map(|loop_points| {
            loop_points
                .iter()
                .map(|point| Point3::new(point.p[0], point.p[1], point.p[2]))
                .collect()
        })
        .collect();
    add_loop_edges(mesh, &loops_3d, edge_ids, fallback, edge_index);
    Some(Ok(()))
}

fn full_knots(knots: &[f64], mults: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for (&knot, &mult) in knots.iter().zip(mults) {
        out.extend(std::iter::repeat_n(knot, mult));
    }
    out
}

fn nurbs_surface(bspline: &solidcore::nurbs::b_spline_surface::BSplineSurface) -> Result<NurbsSurface, String> {
    let net = bspline
        .control_point_net()
        .map_err(|error| format!("NURBS control points: {error}"))?;
    let (u_knots, u_mults) = bspline
        .knots(SurfParamType::U, None)
        .map_err(|error| format!("NURBS u knots: {error}"))?;
    let (v_knots, v_mults) = bspline
        .knots(SurfParamType::V, None)
        .map_err(|error| format!("NURBS v knots: {error}"))?;
    let poles: Vec<[f64; 3]> = net
        .points
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect();
    NurbsSurface::new(
        bspline.degree(SurfParamType::U),
        bspline.degree(SurfParamType::V),
        full_knots(&u_knots, &u_mults),
        full_knots(&v_knots, &v_mults),
        net.u_count,
        net.v_count,
        poles,
        (!net.weights.is_empty()).then_some(net.weights.as_slice()),
    )
    .map_err(|error| format!("NURBS surface: {error}"))
}

/// The face's boundary loops sampled once, shared by every tessellation path.
///
/// One sample per edge is taken from the edge's own 3D curve and the same parameter is
/// evaluated on the face's UV trim curve, so the two adjacent faces of an edge sample
/// exactly the same 3D points (the edge parameter, not the pcurve's, sets the count).
struct FaceLoops {
    loops: Vec<Vec<LoopPoint>>,
    /// Topological edge ordinal for every loop segment (parallel to each loop's points,
    /// where point `i` owns the segment `i -> i+1`).
    edge_ids: Vec<Vec<usize>>,
    /// False when any edge use lacks a UV trim curve; clipping cannot be evaluated then.
    has_uv: bool,
}

/// Smallest power-of-two sample count whose midpoint chord sag is within `tolerance`
/// for this edge curve.
fn edge_segment_count(
    kernel: &Kernel,
    curve: &solidcore::curve::curve::Curve,
    low: f64,
    high: f64,
    tolerance: f64,
    min_segments: usize,
    max_segments: usize,
) -> usize {
    let natural = curve.natural_interval();
    let (low, high) = (low.clamp(natural.min(), natural.max()), high.clamp(natural.min(), natural.max()));
    let mut count = min_segments.max(4).next_power_of_two();
    let limit = max_segments.max(count);
    loop {
        let mut sag = 0.0f64;
        let mut previous: Option<Point3> = None;
        for step in 0..=count {
            let t = step as f64 / count as f64;
            let parameter = low + (high - low) * t;
            let Ok(evaluated) = evaluate_curve(kernel, Some(curve), parameter) else { return count };
            let Some(point) = evaluated.first() else { return count };
            let point = Point3::new(point.x, point.y, point.z);
            if let Some(previous) = previous {
                let mid = Point3::new((previous.x + point.x) * 0.5, (previous.y + point.y) * 0.5, (previous.z + point.z) * 0.5);
                let parameter_mid = low + (high - low) * (step as f64 - 0.5) / count as f64;
                if let Ok(evaluated) = evaluate_curve(kernel, Some(curve), parameter_mid) {
                    if let Some(surface_point) = evaluated.first() {
                        let surface_point = Point3::new(surface_point.x, surface_point.y, surface_point.z);
                        sag = sag.max(surface_point.distance(mid));
                    }
                }
            }
            previous = Some(point);
        }
        if sag <= tolerance || count >= limit {
            return count;
        }
        count = (count * 2).min(limit);
    }
}

fn face_loops(
    kernel: &Kernel,
    brep: &Brep,
    face_id: solidcore::FaceId,
    segments: usize,
    tolerance: f64,
    edge_cache: &mut EdgeSampleCache,
) -> Result<FaceLoops, String> {
    let face = brep
        .face(face_id)
        .map_err(|error| format!("solidcore face: {error}"))?;
    let Some(faceuse) = face.faceuse else {
        return Ok(FaceLoops { loops: Vec::new(), edge_ids: Vec::new(), has_uv: true });
    };
    let loopuses = eu::faceuse_loopuses(brep, faceuse, false, None)
        .map_err(|error| format!("solidcore faceuse loops: {error}"))?;
    let mut has_uv = true;
    let mut loops = Vec::new();
    let mut all_edge_ids = Vec::new();
    for loopuse in loopuses {
        let edgeuses = eu::loopuse_edgeuses(brep, loopuse)
            .map_err(|error| format!("solidcore loop edges: {error}"))?;
        if edgeuses.is_empty() {
            continue;
        }
        // Every edge gets the same number of samples. The count must not depend on the
        // rest of the loop: the two faces of an edge have different loops (a cylinder
        // band also has its end circles and seam), and any loop-wide normalisation would
        // sample the same edge at different parameters on each side, leaving the display
        // mesh open.
        let mut points: Vec<LoopPoint> = Vec::new();
        let mut loop_edge_ids: Vec<usize> = Vec::new();
        let mut last_edge_ordinal = 0usize;
        for &edgeuse in &edgeuses {
            let Some(edge_id) = eu::edge(brep, edgeuse)
                .map_err(|error| format!("solidcore edgeuse edge: {error}"))?
            else {
                continue;
            };
            let edge = brep
                .edge(edge_id)
                .map_err(|error| format!("solidcore edge: {error}"))?;
            let Some(edge_curve) = edge.curve.as_ref() else {
                continue;
            };
            let edge_ordinal = brep.edge_list().iter().position(|candidate| *candidate == edge_id).unwrap_or(usize::MAX);
            last_edge_ordinal = edge_ordinal;
            let orientation = eu::orientation(brep, edgeuse)
                .map_err(|error| format!("solidcore edgeuse orientation: {error}"))?;
            let uv_curve = eu::uv_trim_curve(brep, edgeuse)
                .map_err(|error| format!("solidcore UV trim curve: {error}"))?
                .0;
            if uv_curve.is_none() {
                has_uv = false;
            }
            let natural = edge_curve.natural_interval();
            let (low, high) = (edge.interval.min(), edge.interval.max());
            let count = edge_segment_count(kernel, edge_curve, low, high, tolerance, segments, 512);
            // Sample the edge once in its natural direction. Every adjacent face reads the
            // same 3D samples, so shared boundaries cannot drift apart.
            let samples = if let Some(cached) = edge_cache.get(&edge_ordinal) {
                cached.clone()
            } else {
                let mut edge_points: Vec<LoopPoint> = Vec::with_capacity(count + 1);
                for step in 0..=count {
                    let fraction = step as f64 / count as f64;
                    let parameter = (low + (high - low) * fraction)
                        .clamp(natural.min(), natural.max());
                    let evaluated = evaluate_curve(kernel, Some(edge_curve), parameter)
                        .map_err(|error| format!("solidcore boundary curve: {error}"))?;
                    let edge_point = evaluated
                        .first()
                        .copied()
                        .ok_or_else(|| "boundary curve returned no points".to_string())?;
                    let point = LoopPoint {
                        u: f64::NAN,
                        v: f64::NAN,
                        p: [edge_point.x, edge_point.y, edge_point.z],
                    };
                    let duplicate = edge_points.last().is_some_and(|last: &LoopPoint| {
                        same_point(
                            Point3::new(last.p[0], last.p[1], last.p[2]),
                            Point3::new(point.p[0], point.p[1], point.p[2]),
                        )
                    });
                    if !duplicate {
                        edge_points.push(point);
                    }
                }
                simplify_collinear(&mut edge_points);
                let samples: Vec<[f64; 3]> = edge_points.into_iter().map(|point| point.p).collect();
                edge_cache.insert(edge_ordinal, samples.clone());
                samples
            };
            let uv_interval = uv_curve.map(|curve| curve.natural_interval());
            let last = samples.len().saturating_sub(1).max(1) as f64;
            for (step, _sample) in samples.iter().enumerate() {
                let fraction = step as f64 / last;
                let (u, v) = match (uv_curve, uv_interval) {
                    (Some(curve), Some(interval)) => {
                        let uv_parameter = if orientation == OrientType::Same {
                            interval.min() + (interval.max() - interval.min()) * fraction
                        } else {
                            interval.max() - (interval.max() - interval.min()) * fraction
                        }
                        .clamp(interval.min(), interval.max());
                        let uv = evaluate_curve(kernel, Some(curve), uv_parameter)
                            .map_err(|error| format!("solidcore UV trim curve: {error}"))?;
                        let uv = uv
                            .first()
                            .copied()
                            .ok_or_else(|| "UV trim curve returned no points".to_string())?;
                        (uv.x, uv.y)
                    }
                    _ => (f64::NAN, f64::NAN),
                };
                let natural_index = if orientation == OrientType::Same { step } else { samples.len().saturating_sub(1 + step) };
                let point = LoopPoint {
                    u,
                    v,
                    p: samples.get(natural_index).copied().unwrap_or(samples[0]),
                };
                let duplicate = points.last().is_some_and(|last: &LoopPoint| {
                    same_point(
                        Point3::new(last.p[0], last.p[1], last.p[2]),
                        Point3::new(point.p[0], point.p[1], point.p[2]),
                    )
                });
                if !duplicate {
                    let had_previous = !points.is_empty();
                    points.push(point);
                    if had_previous {
                        loop_edge_ids.push(edge_ordinal);
                    }
                }
            }
        }
        if points.len() >= 2
            && same_point(
                Point3::new(points[0].p[0], points[0].p[1], points[0].p[2]),
                Point3::new(
                    points[points.len() - 1].p[0],
                    points[points.len() - 1].p[1],
                    points[points.len() - 1].p[2],
                ),
            )
        {
            points.pop();
            loop_edge_ids.pop();
        }
        if points.len() >= 3 {
            // The closing segment (last -> first) belongs to the last edge.
            loop_edge_ids.push(last_edge_ordinal);
            loops.push(points);
            all_edge_ids.push(loop_edge_ids);
        }
    }
    Ok(FaceLoops { loops, edge_ids: all_edge_ids, has_uv })
}

/// Validate and unpack a face UV domain into `(u_min, u_max, v_min, v_max)`.
fn checked_domain(domain: &solidcore::geom::extent2d::Extent2d) -> Result<(f64, f64, f64, f64), String> {
    let u0 = domain.u_interval().min();
    let u1 = domain.u_interval().max();
    let v0 = domain.v_interval().min();
    let v1 = domain.v_interval().max();
    if !(u0.is_finite() && u1.is_finite() && v0.is_finite() && v1.is_finite()) {
        return Err(format!("non-finite UV domain ({u0},{u1})x({v0},{v1})"));
    }
    if u1 <= u0 || v1 <= v0 {
        return Err(format!("empty UV domain ({u0},{u1})x({v0},{v1})"));
    }
    Ok((u0, u1, v0, v1))
}

/// Whether the face's surface is flat over its UV domain.
///
/// Uses the surface's own normals (no `SurfaceKind` matching, so a planar B-spline or
/// planar trivially bounded face is recognized too). Evaluation failures count as
/// "not planar" so the caller falls back to the curved-grid path.
fn is_planar(kernel: &Kernel, surface: &Surface, u0: f64, u1: f64, v0: f64, v1: f64) -> bool {
    let fractions = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5), (1.0, 0.5), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)];
    let mut points = Vec::with_capacity(fractions.len());
    let mut normals = Vec::with_capacity(fractions.len());
    for (fu, fv) in fractions {
        let uv = Point2::new(u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv);
        let Ok(point) = surface.evaluate_point(kernel, uv) else {
            return false;
        };
        let Ok(normal) = surface.evaluate_normal(kernel, uv, Side::Left, Side::Left) else {
            return false;
        };
        let Ok(normal) = normal.unitize() else {
            return false;
        };
        points.push(point);
        normals.push(normal);
    }
    let reference = normals[0];
    if !normals
        .iter()
        .all(|normal| normal.dot(reference).abs() >= 1.0 - 1.0e-5)
    {
        return false;
    }
    let scale = points
        .iter()
        .map(|point| point.x.abs().max(point.y.abs()).max(point.z.abs()))
        .fold(0.0f64, f64::max)
        .max(1.0);
    points
        .iter()
        .all(|point| ((point.x - points[0].x) * reference.x
            + (point.y - points[0].y) * reference.y
            + (point.z - points[0].z) * reference.z)
            .abs()
            <= 1.0e-6 * scale)
}

// ---------------------------------------------------------------------------------
// Planar faces: boundary-loop ear clipping
// ---------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn tessellate_planar_face(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    flip_normal: bool,
    face_index: usize,
    loops: &[Vec<LoopPoint>],
    edge_ids: &[Vec<usize>],
    mesh: &mut Mesh,
    edge_index: &mut usize,
) -> Result<(), String> {
    let loops_3d: Vec<Vec<Point3>> = loops
        .iter()
        .map(|path| {
            path.iter()
                .map(|point| Point3::new(point.p[0], point.p[1], point.p[2]))
                .collect()
        })
        .collect();
    let loops = &loops_3d;
    let mid = Point2::new((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    let mut normal = surface
        .evaluate_normal(kernel, mid, Side::Left, Side::Left)
        .map_err(|error| format!("solidcore plane normal: {error}"))?;
    normal = normal
        .unitize()
        .map_err(|_| "solidcore plane normal is degenerate".to_string())?;
    if flip_normal {
        normal = Vec3::new(-normal.x, -normal.y, -normal.z);
    }

    // Triangulate with the shared planar-face ear clipper so holes cut the cap instead
    // of being covered by a centroid fan.
    //
    // The B-rep's boundary-loop order is not defined to start with the outer loop:
    // STEP faces may list inner `FACE_BOUND`s before `FACE_OUTER_BOUND`, and the
    // topology bridge preserves that order. Passing a hole as the outer loop makes the
    // real outer loop look like an unbridgeable hole, so choose the loop with the
    // largest area instead of trusting the position in `loops`.
    let areas: Vec<f64> = loops
        .iter()
        .map(|path| loop_area_along(path, normal))
        .collect();
    let outer_index = areas
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut face: Vec<usize> = Vec::new();
    let mut holes: Vec<Vec<usize>> = Vec::new();
    for (loop_index, path) in loops.iter().enumerate() {
        let mut indices = Vec::with_capacity(path.len());
        for point in path {
            indices.push(points.len());
            points.push([point.x, point.y, point.z]);
        }
        if loop_index == outer_index {
            face = indices;
        } else {
            holes.push(indices);
        }
    }
    // Keep the cap winding consistent with the face normal.
    if areas.get(outer_index).is_some_and(|area| *area < 0.0) {
        face.reverse();
    }
    
    let planar = PlanarBody { points, faces: vec![face], holes: vec![holes] };
    let triangles = match planar.face_triangles(0) {
        Ok(triangles) => triangles,
        Err(error) => {
            // The bridged ear clipper can reject a weakly-simple cap ring. Dropping the
            // whole face leaves a visible hole in the model, so leave a last-resort fan
            // over the outer loop; the spatial stitch closes its boundary.
            let outer = &planar.faces[0];
            let mut fallback = Vec::with_capacity(outer.len().saturating_sub(2));
            for i in 1..outer.len().saturating_sub(1) {
                fallback.push([outer[0], outer[i], outer[i + 1]]);
            }
            let _ = error;
            eprintln!("face {face_index}: planar face triangulation failed; using cap fan");
            fallback
        }
    };
    let group = format!("face:{face_index}");
    let base = mesh.vertices.len() as u32;
    for point in &planar.points {
        push_vertex(mesh, Point3::new(point[0], point[1], point[2]), normal);
    }
    for triangle in triangles {
        push_triangle(
            mesh,
            base + triangle[0] as u32,
            base + triangle[1] as u32,
            base + triangle[2] as u32,
            normal,
            &group,
        );
    }

    add_loop_edges(mesh, loops, edge_ids, normal, edge_index);
    Ok(())
}

/// Signed area of a closed 3D polyline measured along `normal` (positive when the loop
/// winds counter-clockwise about it). Used to identify the outer boundary of a face.
fn loop_area_along(points: &[Point3], normal: Vec3) -> f64 {
    let mut acc = Vec3::new(0.0, 0.0, 0.0);
    for index in 0..points.len() {
        let a = points[index];
        let b = points[(index + 1) % points.len()];
        acc = acc
            + Vec3::new(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            );
    }
    0.5 * acc.dot(normal)
}

// ---------------------------------------------------------------------------------
// Curved faces: UV grid
// ---------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn tessellate_curved_face(
    kernel: &Kernel,
    _brep: &Brep,
    _face_id: solidcore::FaceId,
    surface: &Surface,
    domain: (f64, f64, f64, f64),
    flip_normal: bool,
    face_index: usize,
    options: &CurvedTessOptions,
    boundary: &FaceLoops,
    mesh: &mut Mesh,
    edge_index: &mut usize,
) -> Result<(), String> {
    let (u0, u1, v0, v1) = domain;
    // Angle-driven grids match Shapr3D on doubly-curved analytic surfaces. The
    // one-curvature cone/cylinder trims still use their explicit grid: the planar caps
    // are triangulated from the loop segments, and a mismatched angular grid opens the
    // rim by a few dozen edges.
    let options = if matches!(
        surface.kind(),
        solidcore::surface::surface::p2::SurfaceKind::Sphere(_)
            | solidcore::surface::surface::p2::SurfaceKind::Torus(_)
    ) {
        *options
    } else {
        CurvedTessOptions {
            max_normal_angle_deg: 0.0,
            ..*options
        }
    };
    let options = &options;
    // The primitive curved faces have their trims on the UV-domain boundary (seams and
    // end circles), where the whole rectangle is the face. Only an interior trim needs
    // cutting; those cells used to be dropped, which lost area and opened the mesh.
    let untrimmed = !boundary.has_uv
        || boundary.loops.is_empty()
        || loops_on_domain_boundary(&boundary.loops, u0, u1, v0, v1);
    let loops_3d: Vec<Vec<Point3>> = boundary
        .loops
        .iter()
        .map(|loop_points| {
            loop_points
                .iter()
                .map(|point| Point3::new(point.p[0], point.p[1], point.p[2]))
                .collect()
        })
        .collect();
    if untrimmed {
        return tessellate_curved_grid(
            kernel, surface, domain, flip_normal, face_index, options, &loops_3d,
            &boundary.edge_ids, mesh, edge_index,
        );
    }
    match tessellate_curved_trimmed(
        kernel, surface, domain, flip_normal, face_index, options, boundary, &loops_3d, mesh,
        edge_index,
    ) {
        Ok(()) => Ok(()),
        // A torus (or another doubly periodic surface) can have pcurves whose seams
        // defeat the chain walk. Fall back to a fine grid keeping cells whose centre is
        // inside the trim: approximate at the boundary, but it fills the face instead of
        // dropping it and losing all of its area.
        Err(error) => {
            eprintln!("face {face_index}: {error}; using an approximate trimmed grid");
            tessellate_curved_grid_inside(
                kernel, surface, domain, flip_normal, face_index, options, boundary, &loops_3d,
                mesh, edge_index,
            )
        }
    }
}

/// Approximate trimmed fill for a curved face: a fine grid, keeping whole cells whose
/// centre is inside the face's trim loops.
#[allow(clippy::too_many_arguments)]
fn tessellate_curved_grid_inside(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    flip_normal: bool,
    face_index: usize,
    options: &CurvedTessOptions,
    boundary: &FaceLoops,
    loops_3d: &[Vec<Point3>],
    mesh: &mut Mesh,
    edge_index: &mut usize,
) -> Result<(), String> {
    let uv_loops: Vec<Vec<[f64; 2]>> = boundary
        .loops
        .iter()
        .map(|loop_points| loop_points.iter().map(|point| [point.u, point.v]).collect())
        .collect();
    let nu = curved_segments(kernel, surface, (u0, u1, v0, v1), true, options, 512);
    let nv = curved_segments(kernel, surface, (u0, u1, v0, v1), false, options, 256);
    let nodes_u = nu + 1;
    let mut points = Vec::with_capacity(nodes_u * (nv + 1));
    let mut normals = Vec::with_capacity(nodes_u * (nv + 1));
    let mut normal_ok = Vec::with_capacity(nodes_u * (nv + 1));
    for iv in 0..=nv {
        let v = v0 + (v1 - v0) * (iv as f64) / (nv as f64);
        for iu in 0..=nu {
            let u = u0 + (u1 - u0) * (iu as f64) / (nu as f64);
            let uv = Point2::new(u, v);
            let point = surface
                .evaluate_point(kernel, uv)
                .map_err(|error| format!("solidcore surface point at ({u},{v}): {error}"))?;
            let normal = surface
                .evaluate_normal(kernel, uv, Side::Left, Side::Left)
                .ok()
                .and_then(|normal| normal.unitize().ok());
            let mut sampled = normal.unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            if flip_normal {
                sampled = Vec3::new(-sampled.x, -sampled.y, -sampled.z);
            }
            points.push(point);
            normals.push(sampled);
            normal_ok.push(normal.is_some());
        }
    }
    let base = mesh.vertices.len() as u32;
    for index in 0..points.len() {
        push_vertex(mesh, points[index], normals[index]);
    }
    let group = format!("face:{face_index}");
    for iv in 0..nv {
        let vc = v0 + (v1 - v0) * ((iv as f64) + 0.5) / (nv as f64);
        for iu in 0..nu {
            let uc = u0 + (u1 - u0) * ((iu as f64) + 0.5) / (nu as f64);
            if !uv_inside(uc, vc, &uv_loops) {
                continue;
            }
            let a = iv * nodes_u + iu;
            let b = a + 1;
            let d = a + nodes_u;
            let c = d + 1;
            emit_grid_triangle(mesh, base, &points, &normals, &normal_ok, a, b, c, &group);
            emit_grid_triangle(mesh, base, &points, &normals, &normal_ok, a, c, d, &group);
        }
    }
    let fallback_normal = normals
        .iter()
        .zip(normal_ok.iter())
        .find_map(|(normal, ok)| ok.then_some(*normal))
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    add_loop_edges(mesh, loops_3d, &boundary.edge_ids, fallback_normal, edge_index);
    Ok(())
}

/// Even-odd point-in-loops test in UV space.
fn uv_inside(u: f64, v: f64, loops: &[Vec<[f64; 2]>]) -> bool {
    let mut inside = false;
    for polygon in loops {
        if polygon.len() < 3 {
            continue;
        }
        let mut previous = polygon.len() - 1;
        for current in 0..polygon.len() {
            let [xi, yi] = polygon[current];
            let [xj, yj] = polygon[previous];
            if (yi > v) != (yj > v) {
                let x = (xj - xi) * (v - yi) / (yj - yi) + xi;
                if u < x {
                    inside = !inside;
                }
            }
            previous = current;
        }
    }
    inside
}

/// Fill the whole UV rectangle: the untrimmed curved-face path.
#[allow(clippy::too_many_arguments)]
/// Choose grid segments for one parameter direction from the normal-angle span.
///
/// Shapr3D's display tessellation is angle driven: a sphere uses about 90 x 45 cells, a
/// torus about 90 x 90, and a cylinder about 90 x 2, i.e. a maximum normal turn of 4
/// degrees. Chord-sag alone under-tessellates the small-radius direction of a torus and
/// over-tessellates a cone, so use the angle target for curved analytic surfaces.
fn curved_segments(
    kernel: &Kernel,
    surface: &Surface,
    domain: (f64, f64, f64, f64),
    along_u: bool,
    options: &CurvedTessOptions,
    max_segments: usize,
) -> usize {
    if options.max_normal_angle_deg > 0.0 {
        let min = if along_u {
            options.loop_segments.max(4)
        } else {
            1
        };
        angle_segments(kernel, surface, domain, along_u, min, max_segments, options.max_normal_angle_deg)
    } else {
        let (min, max) = if along_u {
            (options.loop_segments.max(4), max_segments)
        } else {
            (options.v_segments.max(2), max_segments)
        };
        sag_segments(kernel, surface, domain, along_u, min, max, options.bspline_tolerance)
    }
}

fn angle_segments(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    along_u: bool,
    min_segments: usize,
    max_segments: usize,
    max_angle_deg: f64,
) -> usize {
    let samples = 16usize;
    let mut span: f64 = 0.0;
    for other in 0..=4 {
        let other_fraction = other as f64 / 4.0;
        let mut line_span: f64 = 0.0;
        let mut previous: Option<Vec3> = None;
        for step in 0..=samples {
            let t = step as f64 / samples as f64;
            let uv = if along_u {
                Point2::new(u0 + (u1 - u0) * t, v0 + (v1 - v0) * other_fraction)
            } else {
                Point2::new(u0 + (u1 - u0) * other_fraction, v0 + (v1 - v0) * t)
            };
            let normal = surface
                .evaluate_normal(kernel, uv, Side::Left, Side::Left)
                .ok()
                .and_then(|normal| normal.unitize().ok());
            if let (Some(previous), Some(normal)) = (previous, normal) {
                line_span += previous.dot(normal).clamp(-1.0, 1.0).acos();
            }
            previous = normal;
        }
        span = span.max(line_span);
    }
    let target = max_angle_deg.to_radians().max(1.0e-6);
    ((span.max(0.0) / target).ceil() as usize).clamp(min_segments.max(2), max_segments.max(2))
}

/// Choose segments for one parameter direction so the chord sag matches the display
/// tolerance. Starts from a small grid and doubles until the sampled iso-curve midpoints
/// are within tolerance.
fn sag_segments(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    along_u: bool,
    min_segments: usize,
    max_segments: usize,
    tolerance: f64,
) -> usize {
    let mut count = min_segments.max(2).next_power_of_two();
    let limit = max_segments.max(count);
    loop {
        let mut sag: f64 = 0.0;
        for other in 0..=4 {
            let other_fraction = other as f64 / 4.0;
            for cell in 0..count {
                let t0 = cell as f64 / count as f64;
                let t1 = (cell + 1) as f64 / count as f64;
                let mid = (t0 + t1) * 0.5;
                let sample = |t: f64| {
                    if along_u {
                        surface.evaluate_point(kernel, Point2::new(u0 + (u1 - u0) * t, v0 + (v1 - v0) * other_fraction))
                    } else {
                        surface.evaluate_point(kernel, Point2::new(u0 + (u1 - u0) * other_fraction, v0 + (v1 - v0) * t))
                    }
                };
                let (Ok(a), Ok(m), Ok(b)) = (sample(t0), sample(mid), sample(t1)) else {
                    return count;
                };
                let chord = [
                    (a.x + b.x) * 0.5 - m.x,
                    (a.y + b.y) * 0.5 - m.y,
                    (a.z + b.z) * 0.5 - m.z,
                ];
                sag = sag.max((chord[0] * chord[0] + chord[1] * chord[1] + chord[2] * chord[2]).sqrt());
            }
        }
        if sag <= tolerance || count >= limit {
            return count;
        }
        count = (count * 2).min(limit);
    }
}

fn tessellate_curved_grid(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    flip_normal: bool,
    face_index: usize,
    options: &CurvedTessOptions,
    loops_3d: &[Vec<Point3>],
    edge_ids: &[Vec<usize>],
    mesh: &mut Mesh,
    edge_index: &mut usize,
) -> Result<(), String> {
    let nu = curved_segments(kernel, surface, (u0, u1, v0, v1), true, options, 256);
    let nv = curved_segments(kernel, surface, (u0, u1, v0, v1), false, options, 128);
    let nodes_u = nu + 1;
    let node_count = nodes_u * (nv + 1);
    let mut points = Vec::with_capacity(node_count);
    let mut normals = Vec::with_capacity(node_count);
    let mut normal_ok = Vec::with_capacity(node_count);
    for iv in 0..=nv {
        let v = v0 + (v1 - v0) * (iv as f64) / (nv as f64);
        for iu in 0..=nu {
            let u = u0 + (u1 - u0) * (iu as f64) / (nu as f64);
            let uv = Point2::new(u, v);
            let point = surface
                .evaluate_point(kernel, uv)
                .map_err(|error| format!("solidcore surface point at ({u},{v}): {error}"))?;
            let normal = surface
                .evaluate_normal(kernel, uv, Side::Left, Side::Left)
                .ok()
                .and_then(|normal| normal.unitize().ok());
            let mut sampled = normal.unwrap_or(Vec3::new(0.0, 0.0, 0.0));
            if flip_normal {
                sampled = Vec3::new(-sampled.x, -sampled.y, -sampled.z);
            }
            points.push(point);
            normals.push(sampled);
            normal_ok.push(normal.is_some());
        }
    }

    // Register the grid vertices (normals are patched from triangle geometry when the
    // surface normal is singular, e.g. at a cone apex or a sphere pole).
    let base = mesh.vertices.len() as u32;
    for index in 0..node_count {
        push_vertex(mesh, points[index], normals[index]);
    }

    let group = format!("face:{face_index}");
    for iv in 0..nv {
        for iu in 0..nu {
            let a = iv * nodes_u + iu;
            let b = a + 1;
            let d = a + nodes_u;
            let c = d + 1;
            emit_grid_triangle(
                mesh, base, &points, &normals, &normal_ok, a, b, c, &group,
            );
            emit_grid_triangle(
                mesh, base, &points, &normals, &normal_ok, a, c, d, &group,
            );
        }
    }

    // Display-edge vertices are not shaded through the face; give them a unit normal
    // from the grid so `normals` stays a valid unit-normal array.
    let fallback_normal = normals
        .iter()
        .zip(normal_ok.iter())
        .find_map(|(normal, ok)| ok.then_some(*normal))
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    add_loop_edges(mesh, loops_3d, edge_ids, fallback_normal, edge_index);
    Ok(())
}

/// Cut the UV grid against the face's interior trim loops.
#[allow(clippy::too_many_arguments)]
fn tessellate_curved_trimmed(
    kernel: &Kernel,
    surface: &Surface,
    (u0, u1, v0, v1): (f64, f64, f64, f64),
    flip_normal: bool,
    face_index: usize,
    options: &CurvedTessOptions,
    boundary: &FaceLoops,
    loops_3d: &[Vec<Point3>],
    mesh: &mut Mesh,
    edge_index: &mut usize,
) -> Result<(), String> {
    let handed = uv_handedness(kernel, surface, (u0, u1, v0, v1));
    let mut loops = boundary.loops.clone();
    orient_uv_loops(&mut loops, handed);

    // Grid the loop's own UV extent at the same chord density as the full domain, so a
    // small trim is not meshed by one or two cells.
    let (lu0, lu1, lv0, lv1) = uv_extent(&loops);
    let du = (u1 - u0).abs().max(1.0e-12);
    let dv = (v1 - v0).abs().max(1.0e-12);
    let frac_u = ((lu1 - lu0).abs() / du).clamp(1.0e-6, 1.0);
    let frac_v = ((lv1 - lv0).abs() / dv).clamp(1.0e-6, 1.0);
    let full_u = curved_segments(kernel, surface, (u0, u1, v0, v1), true, options, 512);
    let full_v = curved_segments(kernel, surface, (u0, u1, v0, v1), false, options, 256);
    let nu = ((full_u as f64) * frac_u).round().max(2.0) as usize;
    let nv = ((full_v as f64) * frac_v).round().max(2.0) as usize;
    let u_lines: Vec<f64> = (0..=nu)
        .map(|i| lu0 + (lu1 - lu0) * (i as f64) / (nu as f64))
        .collect();
    let v_lines: Vec<f64> = (0..=nv)
        .map(|i| lv0 + (lv1 - lv0) * (i as f64) / (nv as f64))
        .collect();
    let eval = |u: f64, v: f64| {
        surface
            .evaluate_point(kernel, Point2::new(u, v))
            .map(|point| [point.x, point.y, point.z])
            .map_err(|error| format!("solidcore surface point at ({u},{v}): {error}"))
    };
    let mut grid = None;
    let mut last_error = String::new();
    for attempt in 0..2 {
        let mut attempt_loops = loops.clone();
        if attempt == 1 {
            // A periodic surface's pcurves can wind the other way round from its UV
            // frame; try the other handedness before giving up on the face.
            orient_uv_loops(&mut attempt_loops, !handed);
        }
        let mut attempt_u = u_lines.clone();
        let mut attempt_v = v_lines.clone();
        match trim::mesh_trimmed_grid(&attempt_loops, &mut attempt_u, &mut attempt_v, eval) {
            Ok(result) => {
                grid = Some(result);
                break;
            }
            Err(error) => last_error = error,
        }
    }
    let Some(grid) = grid else {
        return Err(format!("curved trim failed: {last_error}"));
    };
    if grid.triangles.is_empty() {
        return Err("trimmed curved face produced no triangles".into());
    }

    let group = format!("face:{face_index}");
    let base = mesh.vertices.len() as u32;
    let mut normals = Vec::with_capacity(grid.positions.len());
    let mut normal_ok = Vec::with_capacity(grid.positions.len());
    for (position, uv) in grid.positions.iter().zip(&grid.uvs) {
        let sampled = surface
            .evaluate_normal(kernel, Point2::new(uv[0], uv[1]), Side::Left, Side::Left)
            .ok()
            .and_then(|normal| normal.unitize().ok());
        let normal = sampled
            .map(|normal| if flip_normal { -normal } else { normal })
            .unwrap_or(Vec3::new(0.0, 0.0, 0.0));
        normals.push(normal);
        normal_ok.push(sampled.is_some());
        push_vertex(mesh, Point3::new(position[0], position[1], position[2]), normal);
    }
    for triangle in &grid.triangles {
        let [a, b, c] = *triangle;
        let pa = Point3::new(
            grid.positions[a as usize][0],
            grid.positions[a as usize][1],
            grid.positions[a as usize][2],
        );
        let pb = Point3::new(
            grid.positions[b as usize][0],
            grid.positions[b as usize][1],
            grid.positions[b as usize][2],
        );
        let pc = Point3::new(
            grid.positions[c as usize][0],
            grid.positions[c as usize][1],
            grid.positions[c as usize][2],
        );
        let Some(triangle_normal) = triangle_normal(pa, pb, pc) else {
            continue;
        };
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        let mut count = 0usize;
        for (node, ok) in [(a, normal_ok[a as usize]), (b, normal_ok[b as usize]), (c, normal_ok[c as usize])] {
            if ok {
                sum = sum + normals[node as usize];
                count += 1;
            }
        }
        let target = if count > 0 {
            (sum * (1.0 / count as f64)).unitize().unwrap_or(triangle_normal)
        } else {
            triangle_normal
        };
        for node in [a, b, c] {
            if !normal_ok[node as usize] {
                mesh.normals[base as usize + node as usize] = to_f32(target);
            }
        }
        push_triangle(mesh, base + a, base + b, base + c, target, &group);
    }
    let fallback_normal = normals
        .iter()
        .zip(normal_ok.iter())
        .find_map(|(normal, ok)| ok.then_some(*normal))
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    add_loop_edges(mesh, loops_3d, &boundary.edge_ids, fallback_normal, edge_index);
    Ok(())
}

/// Whether the surface normal follows the UV frame (`dS/du x dS/dv`).
///
/// A face's loops are wound with the face on their left in this frame, so the outer loop
/// must be anticlockwise in (u, v) when this is true and clockwise when it is false.
fn uv_handedness(kernel: &Kernel, surface: &Surface, (u0, u1, v0, v1): (f64, f64, f64, f64)) -> bool {
    let u = (u0 + u1) * 0.5;
    let v = (v0 + v1) * 0.5;
    let h_u = (u1 - u0) * 1.0e-4;
    let h_v = (v1 - v0) * 1.0e-4;
    let point = |u: f64, v: f64| surface.evaluate_point(kernel, Point2::new(u, v));
    let (Ok(p0), Ok(pu), Ok(pv)) = (point(u, v), point(u + h_u, v), point(u, v + h_v)) else {
        return true;
    };
    let cross = (pu - p0).cross(pv - p0);
    let Ok(normal) = surface.evaluate_normal(kernel, Point2::new(u, v), Side::Left, Side::Left)
    else {
        return true;
    };
    let Ok(normal) = normal.unitize() else {
        return true;
    };
    cross.dot(normal) >= 0.0
}

fn signed_area_uv(loop_points: &[LoopPoint]) -> f64 {
    loop_points
        .iter()
        .zip(loop_points.iter().cycle().skip(1))
        .map(|(a, b)| a.u * b.v - b.u * a.v)
        .sum::<f64>()
        / 2.0
}

fn orient_uv_loops(loops: &mut [Vec<LoopPoint>], handed: bool) {
    let areas: Vec<f64> = loops.iter().map(|loop_points| signed_area_uv(loop_points)).collect();
    let outer = areas
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map_or(0, |(index, _)| index);
    for (index, loop_points) in loops.iter_mut().enumerate() {
        let want_positive = if index == outer { handed } else { !handed };
        if areas.get(index).is_some_and(|area| (*area > 0.0) != want_positive) {
            loop_points.reverse();
        }
    }
}

fn uv_extent(loops: &[Vec<LoopPoint>]) -> (f64, f64, f64, f64) {
    let mut u0 = f64::INFINITY;
    let mut u1 = f64::NEG_INFINITY;
    let mut v0 = f64::INFINITY;
    let mut v1 = f64::NEG_INFINITY;
    for point in loops.iter().flatten() {
        u0 = u0.min(point.u);
        u1 = u1.max(point.u);
        v0 = v0.min(point.v);
        v1 = v1.max(point.v);
    }
    (u0, u1, v0, v1)
}

/// Emit one grid triangle, oriented against the quad's sampled normals and dropping
/// degenerate triangles (poles of a sphere, seam duplicates).
#[allow(clippy::too_many_arguments)]
fn emit_grid_triangle(
    mesh: &mut Mesh,
    base: u32,
    points: &[Point3],
    normals: &[Vec3],
    normal_ok: &[bool],
    a: usize,
    b: usize,
    c: usize,
    group: &str,
) {
    let pa = points[a];
    let pb = points[b];
    let pc = points[c];
    let Some(mut geometric_normal) = triangle_normal(pa, pb, pc) else {
        return;
    };
    let reference = average_normal(normals, normal_ok, [a, b, c]).unwrap_or(geometric_normal);
    let mut b = b;
    let mut c = c;
    if geometric_normal.dot(reference) < 0.0 {
        core::mem::swap(&mut b, &mut c);
        let Some(flipped) = triangle_normal(points[a], points[b], points[c]) else {
            return;
        };
        geometric_normal = flipped;
    }
    for node in [a, b, c] {
        if !normal_ok[node] {
            mesh.normals[base as usize + node] = to_f32(geometric_normal);
        }
    }
    mesh.faces.push([
        base + a as u32,
        base + b as u32,
        base + c as u32,
    ]);
    mesh.face_groups.push(group.to_string());
}

fn average_normal(normals: &[Vec3], normal_ok: &[bool], nodes: [usize; 3]) -> Option<Vec3> {
    let mut sum = Vec3::new(0.0, 0.0, 0.0);
    let mut count = 0;
    for node in nodes {
        if normal_ok[node] {
            sum = Vec3::new(
                sum.x + normals[node].x,
                sum.y + normals[node].y,
                sum.z + normals[node].z,
            );
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let divisor = count as f64;
    Vec3::new(sum.x / divisor, sum.y / divisor, sum.z / divisor).unitize().ok()
}

// ---------------------------------------------------------------------------------
// Boundary loops
// ---------------------------------------------------------------------------------

/// Drop interior samples of a polyline that lie on the line through their neighbours.
///
/// The criterion uses only the sampled points, so every face sharing an edge keeps the
/// same subset. Endpoints are always kept.
fn simplify_collinear(points: &mut Vec<LoopPoint>) {
    if points.len() < 3 {
        return;
    }
    let mut out: Vec<LoopPoint> = Vec::with_capacity(points.len());
    out.push(points[0]);
    for index in 1..points.len() - 1 {
        let a = out[out.len() - 1];
        let b = points[index];
        let c = points[index + 1];
        let ab = [b.p[0] - a.p[0], b.p[1] - a.p[1], b.p[2] - a.p[2]];
        let ac = [c.p[0] - a.p[0], c.p[1] - a.p[1], c.p[2] - a.p[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let cross_length = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let length = (ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2]).sqrt();
        // Distance from b to the line a-c; a straight edge's rounding is far below this.
        if length > 0.0 && cross_length / length > 1.0e-6 * length.max(1.0) {
            out.push(b);
        }
    }
    out.push(points[points.len() - 1]);
    *points = out;
}

/// True when every trim sample lies on one of the four UV-domain boundary lines.
///
/// This is the signature of the primitive curved faces: their only trims are seams
/// (`u = u_min` / `u = u_max`) and end circles (`v = v_min` / `v = v_max`). A genuinely
/// trimmed curved face has trim samples in the domain interior.
fn loops_on_domain_boundary(
    loops: &[Vec<LoopPoint>],
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
) -> bool {
    let eps_u = (u1 - u0).abs() * 1.0e-6;
    let eps_v = (v1 - v0).abs() * 1.0e-6;
    loops.iter().flatten().all(|point| {
        (point.u - u0).abs() <= eps_u
            || (point.u - u1).abs() <= eps_u
            || (point.v - v0).abs() <= eps_v
            || (point.v - v1).abs() <= eps_v
    })
}

// ---------------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------------

fn centered(mut mesh: Mesh) -> Result<Mesh, String> {
    if mesh.vertices.is_empty() {
        return Err("tessellation produced no vertices".into());
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    for vertex in &mut mesh.vertices {
        for axis in 0..3 {
            vertex[axis] -= center[axis];
        }
    }
    Ok(mesh)
}

/// Register the boundary loops' polylines as display edges.
fn add_loop_edges(
    mesh: &mut Mesh,
    loops: &[Vec<Point3>],
    edge_ids: &[Vec<usize>],
    fallback_normal: Vec3,
    edge_index: &mut usize,
) {
    for (loop_index, loop_points) in loops.iter().enumerate() {
        if loop_points.len() < 2 {
            continue;
        }
        let base = mesh.vertices.len() as u32;
        for point in loop_points {
            push_vertex(mesh, *point, fallback_normal);
        }
        let ids = edge_ids.get(loop_index).filter(|ids| ids.len() == loop_points.len());
        if let Some(ids) = ids {
            // One display edge per topological B-rep edge. This is what makes edge
            // selection and chamfer references address a single edge instead of the
            // whole face boundary.
            let mut groups: std::collections::BTreeMap<usize, Vec<[u32; 2]>> = std::collections::BTreeMap::new();
            for index in 0..loop_points.len() {
                let next = (index + 1) % loop_points.len();
                groups.entry(ids[index]).or_default().push([base + index as u32, base + next as u32]);
            }
            for (edge_id, segments) in groups {
                let id = format!("edge:{edge_id}");
                if let Some(existing) = mesh.brep_edges.iter_mut().find(|edge| edge.id == id) {
                    for segment in segments {
                        if !existing.segments.contains(&segment) {
                            existing.segments.push(segment);
                        }
                    }
                } else {
                    mesh.brep_edges.push(BrepEdge { id, kind: "edge".into(), segments });
                }
            }
        } else {
            // No topology available (domain-grid fallback): keep the loop as one edge.
            let mut segments = Vec::with_capacity(loop_points.len());
            for index in 0..loop_points.len() {
                let next = (index + 1) % loop_points.len();
                segments.push([base + index as u32, base + next as u32]);
            }
            mesh.brep_edges.push(BrepEdge {
                id: format!("mesh-edge:{}", *edge_index),
                kind: "mesh".into(),
                segments,
            });
            *edge_index += 1;
        }
    }
    // Adjacent faces contribute the same geometric edge chain with different vertex
    // indices. Keep one copy so the overlay does not draw doubled edges.
    for edge in &mut mesh.brep_edges {
        let mut seen = std::collections::HashSet::new();
        edge.segments.retain(|[a, b]| {
            let key = |index: u32| {
                let point = mesh.vertices[index as usize];
                [
                    (point[0] as f64 * 1.0e5).round() as i64,
                    (point[1] as f64 * 1.0e5).round() as i64,
                    (point[2] as f64 * 1.0e5).round() as i64,
                ]
            };
            let (a, b) = (key(*a), key(*b));
            seen.insert(if a <= b { (a, b) } else { (b, a) })
        });
        edge.segments.retain(|segment| (segment[0] as usize) < mesh.vertices.len() && (segment[1] as usize) < mesh.vertices.len());
    }
}

fn push_vertex(mesh: &mut Mesh, point: Point3, normal: Vec3) {
    mesh.vertices.push([point.x as f32, point.y as f32, point.z as f32]);
    mesh.normals.push([normal.x as f32, normal.y as f32, normal.z as f32]);
}

/// Push a triangle, flipping its winding so it faces `normal`, and skipping degenerate
/// triangles. Normals are expected to be already registered by the caller.
fn push_triangle(mesh: &mut Mesh, a: u32, b: u32, c: u32, normal: Vec3, group: &str) {
    let pa = point_from_vertex(mesh.vertices[a as usize]);
    let pb = point_from_vertex(mesh.vertices[b as usize]);
    let pc = point_from_vertex(mesh.vertices[c as usize]);
    let Some(triangle) = triangle_normal(pa, pb, pc) else {
        return;
    };
    if triangle.dot(normal) >= 0.0 {
        mesh.faces.push([a, b, c]);
    } else {
        mesh.faces.push([a, c, b]);
    }
    mesh.face_groups.push(group.to_string());
}

fn point_from_vertex(vertex: [f32; 3]) -> Point3 {
    Point3::new(vertex[0] as f64, vertex[1] as f64, vertex[2] as f64)
}

fn triangle_normal(a: Point3, b: Point3, c: Point3) -> Option<Vec3> {
    let u = Vec3::new(b.x - a.x, b.y - a.y, b.z - a.z);
    let v = Vec3::new(c.x - a.x, c.y - a.y, c.z - a.z);
    let normal = u.cross(v);
    let length = normal.length();
    let scale = u.length().max(v.length()).max(1.0);
    if length <= 1.0e-12 * scale * scale {
        return None;
    }
    normal.unitize().ok()
}

fn to_f32(normal: Vec3) -> [f32; 3] {
    [normal.x as f32, normal.y as f32, normal.z as f32]
}

fn same_point(a: Point3, b: Point3) -> bool {
    (a.x - b.x).abs() <= POINT_EPS
        && (a.y - b.y).abs() <= POINT_EPS
        && (a.z - b.z).abs() <= POINT_EPS
}

/// Render-only split of welded display vertices at shading creases.
///
/// SolidCore's stitched display mesh shares one vertex across face boundaries,
/// which is right for topology and measurement but wrong for shading: a cap and
/// its wall need two normals. This adapter keeps the source mesh untouched and
/// returns a copy whose vertices are split by smooth group, so sharp rims keep
/// their crease while tangent faces share a normal.
#[must_use]
pub fn split_shading_normals(mesh: &Mesh) -> Mesh {
    if mesh.faces.is_empty() {
        return mesh.clone();
    }

    // Geometric triangle normals, flipped onto the same side as the source
    // surface normals. The source normals are often already welded/averaged at
    // this point, but their sign is still useful.
    let mut triangle_normals = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let [a, b, c] = face.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) = (mesh.vertices.get(a), mesh.vertices.get(b), mesh.vertices.get(c)) else {
            triangle_normals.push(None);
            continue;
        };
        let Some(mut normal) = shading_triangle_normal(*pa, *pb, *pc) else {
            triangle_normals.push(None);
            continue;
        };
        let mut source = [0.0_f32; 3];
        for index in [a, b, c] {
            if let Some(vertex_normal) = mesh.normals.get(index) {
                for axis in 0..3 {
                    source[axis] += vertex_normal[axis];
                }
            }
        }
        if normal[0] * source[0] + normal[1] * source[1] + normal[2] * source[2] < 0.0 {
            for value in &mut normal {
                *value = -*value;
            }
        }
        triangle_normals.push(Some(normal));
    }

    // Split per vertex by normal direction, not by B-rep face group. A cap and a
    // wall can be part of one B-rep face/group in the display mesh, but they must
    // still use separate normals at the rim.
    const CREASE_COSINE: f32 = 0.819_152_1; // 35 degrees
    let mut incidents: Vec<Vec<(usize, usize, [f32; 3])>> = vec![Vec::new(); mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let Some(normal) = triangle_normals[face_index] else { continue };
        for (slot, vertex) in face.iter().enumerate() {
            if let Some(list) = incidents.get_mut(*vertex as usize) {
                list.push((face_index, slot, normal));
            }
        }
    }

    let mut cluster_sums = vec![Vec::<[f32; 3]>::new(); mesh.vertices.len()];
    let mut corner_cluster = vec![[0_usize; 3]; mesh.faces.len()];
    for (vertex, incident) in incidents.iter().enumerate() {
        for (face_index, slot, normal) in incident {
            let mut assigned = None;
            for (cluster, sum) in cluster_sums[vertex].iter().enumerate() {
                let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                if length > 1.0e-12
                    && (sum[0] * normal[0] + sum[1] * normal[1] + sum[2] * normal[2]) / length >= CREASE_COSINE
                {
                    assigned = Some(cluster);
                    break;
                }
            }
            let cluster = assigned.unwrap_or_else(|| {
                cluster_sums[vertex].push([0.0; 3]);
                cluster_sums[vertex].len() - 1
            });
            for axis in 0..3 {
                cluster_sums[vertex][cluster][axis] += normal[axis];
            }
            corner_cluster[*face_index][*slot] = cluster;
        }
    }

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut vertex_map: HashMap<(usize, usize), u32> = HashMap::new();
    let mut first_for_old = vec![u32::MAX; mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let mut corners = [0_u32; 3];
        for (slot, old) in face.iter().enumerate() {
            let key = (*old as usize, corner_cluster[face_index][slot]);
            let index = if let Some(index) = vertex_map.get(&key) {
                *index
            } else {
                let index = positions.len() as u32;
                positions.push(mesh.vertices[*old as usize]);
                let sum = cluster_sums[*old as usize][corner_cluster[face_index][slot]];
                let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
                normals.push(if length > 1.0e-12 {
                    [sum[0] / length, sum[1] / length, sum[2] / length]
                } else {
                    [0.0, 0.0, 1.0]
                });
                if first_for_old[*old as usize] == u32::MAX {
                    first_for_old[*old as usize] = index;
                }
                vertex_map.insert(key, index);
                index
            };
            corners[slot] = index;
        }
        faces.push(corners);
    }

    let mut render = mesh.clone();
    render.vertices = positions;
    render.normals = normals;
    render.faces = faces;
    for edge in &mut render.brep_edges {
        edge.segments.retain_mut(|segment| {
            let a = first_for_old.get(segment[0] as usize).copied().unwrap_or(u32::MAX);
            let b = first_for_old.get(segment[1] as usize).copied().unwrap_or(u32::MAX);
            if a == u32::MAX || b == u32::MAX || a == b {
                return false;
            }
            *segment = [a, b];
            true
        });
    }
    render.vertices_data = None;
    render.normals_data = None;
    render
}

fn shading_triangle_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<[f32; 3]> {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    (length > 1.0e-12).then(|| [normal[0] / length, normal[1] / length, normal[2] / length])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn test_options() -> CurvedTessOptions {
        CurvedTessOptions {
            u_segments: 16,
            v_segments: 4,
            loop_segments: 16,
            ..CurvedTessOptions::default()
        }
    }

    fn group_counts(mesh: &Mesh) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for group in &mesh.face_groups {
            *counts.entry(group.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn assert_mesh_sane(mesh: &Mesh) {
        assert!(!mesh.vertices.is_empty(), "mesh has no vertices");
        assert!(!mesh.faces.is_empty(), "mesh has no faces");
        assert_eq!(mesh.normals.len(), mesh.vertices.len(), "normals not parallel to vertices");
        assert_eq!(mesh.face_groups.len(), mesh.faces.len(), "face groups not parallel to faces");
        for vertex in &mesh.vertices {
            for value in vertex {
                assert!(value.is_finite(), "non-finite vertex {vertex:?}");
            }
        }
        for normal in &mesh.normals {
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!((length - 1.0).abs() < 1.0e-3, "normal not unit: {normal:?}");
        }
        for face in &mesh.faces {
            for index in face {
                assert!((*index as usize) < mesh.vertices.len(), "face index out of range");
            }
            let a = mesh.vertices[face[0] as usize];
            let b = mesh.vertices[face[1] as usize];
            let c = mesh.vertices[face[2] as usize];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let area2 = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            assert!(area2 > 1.0e-9, "degenerate triangle {face:?}");
        }
        for edge in &mesh.brep_edges {
            for segment in &edge.segments {
                for index in segment {
                    assert!((*index as usize) < mesh.vertices.len(), "edge index out of range");
                }
            }
        }
    }

    #[test]
    fn loop_area_along_selects_the_outer_loop_winding() {
        // A 10x10 square in the z = 0 plane, traversed counter-clockwise about +z.
        let square = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        assert!((loop_area_along(&square, up) - 100.0).abs() < 1.0e-9);
        let mut reversed = square;
        reversed.reverse();
        assert!((loop_area_along(&reversed, up) + 100.0).abs() < 1.0e-9);
        // The larger loop is the outer boundary even when it is listed second; this is
        // exactly what STEP faces with inner bounds before the outer bound rely on.
        let small = [
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(2.0, 4.0, 0.0),
            Point3::new(4.0, 4.0, 0.0),
            Point3::new(4.0, 2.0, 0.0),
        ];
        assert!(loop_area_along(&square, up).abs() > loop_area_along(&small, up).abs());
    }

    #[test]
    fn cylinder_mesh_has_lateral_surface_and_two_caps() {
        let mesh = cylinder_mesh_with_options(2.0, 5.0, &test_options()).unwrap();
        assert_mesh_sane(&mesh);

        let counts = group_counts(&mesh);
        assert_eq!(counts.len(), 3, "expected three face groups, got {counts:?}");
        // 16 u segments x 4 v segments x 2 triangles for the lateral surface.
        assert_eq!(counts.get("face:0"), Some(&(16 * 4 * 2)));
        // One fan triangle per boundary-loop segment for each circular cap.
        assert_eq!(counts.get("face:1"), Some(&14));
        assert_eq!(counts.get("face:2"), Some(&14));

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in &mesh.vertices {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        assert!((min[0] + 2.0).abs() < 1.0e-4, "x min {}", min[0]);
        assert!((max[0] - 2.0).abs() < 1.0e-4, "x max {}", max[0]);
        assert!((min[1] + 2.0).abs() < 1.0e-4, "y min {}", min[1]);
        assert!((max[1] - 2.0).abs() < 1.0e-4, "y max {}", max[1]);
        assert!((min[2] + 2.5).abs() < 1.0e-4, "z min {}", min[2]);
        assert!((max[2] - 2.5).abs() < 1.0e-4, "z max {}", max[2]);

        // Caps stay exactly on their planes and side normals are radial (no z part).
        for (face_index, group) in mesh.face_groups.iter().enumerate() {
            let face = mesh.faces[face_index];
            match group.as_str() {
                "face:1" => {
                    for index in face {
                        assert!(mesh.vertices[index as usize][2] < -2.5 + 1.0e-4);
                    }
                }
                "face:2" => {
                    for index in face {
                        assert!(mesh.vertices[index as usize][2] > 2.5 - 1.0e-4);
                    }
                }
                "face:0" => {
                    // The lateral surface's geometric normal is radial even though a
                    // welded rim vertex averages in the cap's normal.
                    let a = mesh.vertices[face[0] as usize];
                    let b = mesh.vertices[face[1] as usize];
                    let c = mesh.vertices[face[2] as usize];
                    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let normal = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let length =
                        (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
                            .sqrt();
                    assert!(
                        (normal[2] / length).abs() < 1.0e-3,
                        "lateral triangle normal not radial: {normal:?}"
                    );
                }
                other => panic!("unexpected face group {other}"),
            }
        }
    }

    #[test]
    fn cylinder_triangles_face_outward() {
        let mesh = cylinder_mesh_with_options(2.0, 5.0, &test_options()).unwrap();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let a = mesh.vertices[face[0] as usize];
            let b = mesh.vertices[face[1] as usize];
            let c = mesh.vertices[face[2] as usize];
            let normal = triangle_normal(
                Point3::new(a[0] as f64, a[1] as f64, a[2] as f64),
                Point3::new(b[0] as f64, b[1] as f64, b[2] as f64),
                Point3::new(c[0] as f64, c[1] as f64, c[2] as f64),
            )
            .unwrap();
            let centroid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            let outward = normal.x as f32 * centroid[0]
                + normal.y as f32 * centroid[1]
                + normal.z as f32 * centroid[2];
            assert!(outward > 1.0e-6, "triangle {face_index} faces inward");
        }
    }

    #[test]
    fn cylinder_vertices_sit_on_the_radius() {
        let mesh = cylinder_mesh_with_options(2.0, 5.0, &test_options()).unwrap();
        for (face_index, group) in mesh.face_groups.iter().enumerate() {
            if group != "face:0" {
                continue;
            }
            for index in mesh.faces[face_index] {
                let vertex = mesh.vertices[index as usize];
                let radius = (vertex[0] * vertex[0] + vertex[1] * vertex[1]).sqrt();
                assert!((radius - 2.0).abs() < 1.0e-4, "lateral vertex radius {radius}");
            }
        }
    }

    #[test]
    fn sphere_mesh_vertices_lie_on_the_sphere() {
        let mesh = sphere_mesh_with_options(3.0, &test_options()).unwrap();
        assert_mesh_sane(&mesh);
        assert_eq!(group_counts(&mesh).len(), 1);
        for vertex in &mesh.vertices {
            let radius =
                (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt();
            assert!((radius - 3.0).abs() < 1.0e-3, "sphere vertex radius {radius}");
        }
    }

    #[test]
    fn torus_mesh_vertices_lie_on_the_torus() {
        let mesh = torus_mesh_with_options(5.0, 1.0, &test_options()).unwrap();
        assert_mesh_sane(&mesh);
        assert_eq!(group_counts(&mesh).len(), 1);
        for vertex in &mesh.vertices {
            let radial = (vertex[0] * vertex[0] + vertex[1] * vertex[1]).sqrt();
            let distance = ((radial - 5.0) * (radial - 5.0) + vertex[2] * vertex[2]).sqrt();
            assert!((distance - 1.0).abs() < 1.0e-3, "torus vertex distance {distance}");
        }
    }

    #[test]
    fn cone_mesh_spans_both_radii() {
        let mut kernel = Kernel::default();
        let brep = api::create_cone(&mut kernel, Point3::new(0.0, 0.0, 0.0), 2.0, 1.0, 5.0)
            .unwrap();
        let mesh = brep_display_mesh(&kernel, &brep, &test_options()).unwrap();
        assert_mesh_sane(&mesh);
        let counts = group_counts(&mesh);
        assert_eq!(counts.len(), 3, "cone should have lateral surface plus two caps: {counts:?}");
        assert_eq!(counts.get("face:0"), Some(&(16 * 4 * 2)));
        for (face_index, group) in mesh.face_groups.iter().enumerate() {
            if group != "face:0" {
                continue;
            }
            for index in mesh.faces[face_index] {
                let vertex = mesh.vertices[index as usize];
                let radius = (vertex[0] * vertex[0] + vertex[1] * vertex[1]).sqrt();
                assert!(
                    radius >= 1.0 - 1.0e-4 && radius <= 2.0 + 1.0e-4,
                    "cone lateral radius {radius}"
                );
            }
        }
    }

    #[test]
    fn merging_uncoloured_meshes_keeps_the_colour_channel_empty() {
        let mesh = || Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            face_groups: vec!["face:0".into()],
            ..Mesh::default()
        };
        let merged = crate::merge_meshes(vec![mesh(), mesh()]);
        assert!(merged.face_colors.is_empty(), "no-look meshes must not allocate a colour per face");
    }

    #[test]
    fn shading_split_separates_flat_cap_from_wall() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            normals: vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            face_groups: vec!["face:0".into(), "face:0".into()],
            brep_edges: Vec::new(),
            ..Mesh::default()
        };
        let split = split_shading_normals(&mesh);
        assert!(split.vertices.len() > mesh.vertices.len());
        assert!(split.normals.iter().any(|normal| normal[2] > 0.99));
        assert!(split.normals.iter().any(|normal| normal[0] < -0.99));
    }

    #[test]
    fn brep_display_mesh_keeps_solidcore_coordinates() {
        let mut kernel = Kernel::default();
        let brep = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 2.0, 5.0)
            .unwrap();
        let mesh = brep_display_mesh(&kernel, &brep, &test_options()).unwrap();
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for vertex in &mesh.vertices {
            min_z = min_z.min(vertex[2]);
            max_z = max_z.max(vertex[2]);
        }
        assert!(min_z.abs() < 1.0e-4, "uncentered cylinder should start at z=0, got {min_z}");
        assert!((max_z - 5.0).abs() < 1.0e-4, "uncentered cylinder should end at z=5, got {max_z}");
    }

    #[test]
    fn options_control_grid_resolution() {
        let mut kernel = Kernel::default();
        let brep = api::create_cylinder(&mut kernel, Point3::new(0.0, 0.0, 0.0), 1.0, 1.0)
            .unwrap();
        let coarse = brep_display_mesh(
            &kernel,
            &brep,
            &CurvedTessOptions { u_segments: 8, v_segments: 2, loop_segments: 8, ..CurvedTessOptions::default() },
        )
        .unwrap();
        let fine = brep_display_mesh(
            &kernel,
            &brep,
            &CurvedTessOptions { u_segments: 32, v_segments: 8, loop_segments: 32, ..CurvedTessOptions::default() },
        )
        .unwrap();
        assert!(fine.faces.len() > coarse.faces.len());
        assert_eq!(group_counts(&coarse).get("face:0"), Some(&(8 * 2 * 2)));
        assert_eq!(group_counts(&fine).get("face:0"), Some(&(32 * 8 * 2)));
    }
    #[test]
    fn angle_grid_matches_shapr3d_sphere_and_torus_density() {
        let options = CurvedTessOptions { max_normal_angle_deg: 4.0, ..CurvedTessOptions::default() };
        let mut kernel = Kernel::default();
        let sphere = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), 14.0).unwrap();
        let sphere_mesh = brep_display_mesh(&kernel, &sphere, &options).unwrap();
        let torus = api::create_torus(&mut kernel, Point3::new(0.0, 0.0, 0.0), 16.0, 5.0).unwrap();
        let torus_mesh = brep_display_mesh(&kernel, &torus, &options).unwrap();
        // Shapr3D's one-shot corpus export at these radii.
        assert_eq!(sphere_mesh.faces.len(), 7_920, "sphere density");
        assert_eq!(torus_mesh.faces.len(), 16_200, "torus density");
        let volume = |mesh: &Mesh| -> f64 {
            mesh.faces.iter().map(|face| {
                let a = mesh.vertices[face[0] as usize];
                let b = mesh.vertices[face[1] as usize];
                let c = mesh.vertices[face[2] as usize];
                (a[0] as f64 * (b[1] as f64 * c[2] as f64 - b[2] as f64 * c[1] as f64)
                    - a[1] as f64 * (b[0] as f64 * c[2] as f64 - b[2] as f64 * c[0] as f64)
                    + a[2] as f64 * (b[0] as f64 * c[1] as f64 - b[1] as f64 * c[0] as f64)) / 6.0
            }).sum()
        };
        assert!(volume(&sphere_mesh) > 0.0, "sphere shell must face outward");
        assert!(volume(&torus_mesh) > 0.0, "torus shell must face outward");
    }

    #[test]
    fn remaining_ported_primitives_have_analytic_volumes() {
        use solidcore::geom::attribute::{AttributeStore, ObjectKey};
        use solidcore::mass::integrator::properties::{brep_volume, PropertyAttributes};
        let store = AttributeStore::new();
        let key = |_| ObjectKey(0);
        let attrs = PropertyAttributes { store: &store, key_of: &key };
        let volume = |brep: solidcore::Brep| {
            let kernel = Kernel::default();
            brep_volume(&kernel, &mut brep.clone_standing_for_self().expect("standing copy"), 1e-7, &attrs).unwrap().abs()
        };
        let mut kernel = Kernel::default();
        let pyramid = api::create_pyramid(&mut kernel, Point3::new(0.0, 0.0, 0.0), 5.0, 20.0).unwrap();
        assert!((volume(pyramid) - 5.0 * 5.0 * 20.0 / 3.0).abs() < 1e-6);
        let sphere = api::create_sphere_no_pole(&mut kernel, Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        let sphere_volume = volume(sphere);
        assert!((sphere_volume - 4.0 / 3.0 * std::f64::consts::PI * 27.0).abs() < 1e-4, "sphere volume {sphere_volume}");
        let partial_sphere = api::create_partial_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), 3.0, 0.0, 90.0).unwrap();
        let partial_sphere_volume = volume(partial_sphere); assert!((partial_sphere_volume - std::f64::consts::PI * 27.0 / 3.0).abs() < 1e-4, "partial sphere volume {partial_sphere_volume}");
        let partial_torus = api::create_partial_torus(&mut kernel, Point3::new(0.0, 0.0, 0.0), 5.0, 1.5, 0.0, 90.0).unwrap();
        let partial_torus_volume = volume(partial_torus); assert!((partial_torus_volume - 2.0 * std::f64::consts::PI * std::f64::consts::PI * 5.0 * 1.5 * 1.5 / 4.0).abs() < 1e-3, "partial torus volume {partial_torus_volume}");
    }

    #[test]
    fn remaining_ported_primitives_tessellate() {
        for (name, mesh) in [
            ("pyramid", pyramid_mesh(10.0, 20.0)),
            ("sphere_no_pole", sphere_no_pole_mesh(3.0)),
            ("partial_sphere", partial_sphere_mesh(3.0, 0.0, 90.0)),
            ("partial_torus", partial_torus_mesh(5.0, 1.5, 0.0, 90.0)),
        ] {
            let mesh = mesh.unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_mesh_sane(&mesh);
            assert!(!mesh.vertices.is_empty(), "{name}: no vertices");
            assert!(!mesh.brep_edges.is_empty(), "{name}: no edges");
            if name == "sphere_no_pole" {
                let mut edges: BTreeMap<(u32, u32), usize> = BTreeMap::new();
                for face in &mesh.faces {
                    for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                        let key = if a < b { (a, b) } else { (b, a) };
                        *edges.entry(key).or_default() += 1;
                    }
                }
                assert!(edges.values().all(|count| *count == 2), "sphere_no_pole has open or non-manifold seam edges");
            }
        }
    }


    #[test]
    #[ignore = "sample-scale STEP imports are slow in debug; run explicitly"]
    fn unbounded_cone_intersection_samples_are_bounded() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../tests/Sample AP203 Data Files");
        for name in ["1797609in.stp", "2827056.stp"] {
            let path = root.join(name);
            if !path.exists() {
                continue;
            }
            let import = import_step(path.to_str().unwrap())
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let max = import.mesh.vertices.iter().flat_map(|vertex| vertex.iter()).fold(0.0_f32, |max, value| max.max(value.abs()));
            assert!(max < 1_000.0, "{name}: display mesh escaped the model bounds (max {max})");
            assert!(import.mesh.faces.len() > 1_000, "{name}: display mesh collapsed ({} triangles)", import.mesh.faces.len());
            assert!(!import.mesh.vertices.is_empty(), "{name}: no vertices");
        }
    }

}
