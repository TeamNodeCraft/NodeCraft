//! wgpu viewport glue around SolidCraft's GPU renderer.
//!
//! The renderer, scene format and pipelines belong to `solidcraft-viewport-gpu`; this module only
//! converts a NodeCraft display mesh into a `GpuScene`, owns the `SceneSlot`, and implements
//! the `egui_wgpu` paint callback.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use egui_wgpu::{CallbackTrait, RenderState, wgpu};
use solidcore_record::Vec3;
use solidcraft_geom::{Aabb3, Mesh as CraftMesh};

use crate::viewport::{ViewSettings, draws_grid};
use solidcraft_viewport::Camera;
use solidcraft_viewport::surface::OpaqueSurface;
use solidcraft_viewport_gpu::depth_format;
use solidcraft_viewport_gpu::view::{
    FACE_BIAS, depth_bias, depth_compare, gpu_view_proj, reversed_depth,
};
use solidcraft_viewport_gpu::view::{
    Frame, GpuScene, GpuTarget, Palette, PaletteIndex, Renderer, SceneSlot, clip_hatches,
    clip_planes,
};

use crate::adapter::DisplayMesh;
use crate::edge_ribbon::EdgeCpuBatch;
use crate::gpu_backdrop::BackdropRenderer;
use crate::viewport_grid::grid_step;
use solidcraft_geom::prelude::*;

const EMPTY_GRID_RADIUS: f64 = 100.0;

/// The defocused backdrop is drawn far below display resolution and blurred, so
/// its PBR copy does not need every triangle. Capping it keeps the per-mesh CPU
/// and GPU backdrop cost bounded on large assemblies without changing the solid
/// viewport object.
const BACKDROP_PBR_TRIANGLE_BUDGET: usize = 120_000;

/// Triangle stride for the backdrop's PBR copy: 1 for a small mesh, larger only
/// when the mesh exceeds [`BACKDROP_PBR_TRIANGLE_BUDGET`].
fn backdrop_pbr_stride(triangle_count: usize) -> usize {
    triangle_count.div_ceil(BACKDROP_PBR_TRIANGLE_BUDGET).max(1)
}

/// Flat palette background for the viewport callback. SolidCraft clears its own
/// offscreen frame to transparent; this quad makes the central app surface show the
/// palette's darkest colour instead of whatever the window compositor leaves behind.
struct ViewportClear {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl ViewportClear {
    fn set_background(&self, queue: &wgpu::Queue, rgba: [u8; 4]) {
        let color = [
            f32::from(rgba[0]) / 255.0,
            f32::from(rgba[1]) / 255.0,
            f32::from(rgba[2]) / 255.0,
            f32::from(rgba[3]) / 255.0,
        ];
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&color));
    }
}

const CLEAR_SHADER: &str = r#"
struct ClearUniform {
  color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> clear: ClearUniform;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return clear.color;
}
"#;

/// MSAA used by the egui pass and all custom viewport/node pipelines.
///
/// macOS defaults to 4x; other platforms default to 1x because the surface
/// sample count is chosen before the adapter is queried. Set NODECRAFT_MSAA=4
/// to opt in on a platform that supports it. The Metal Bgra8Unorm format used
/// by the egui pipeline advertises 1x/2x/4x here; 8x is not a valid surface
/// sample count, so the canvas must not request it.
pub fn msaa_samples() -> u32 {
    std::env::var("NODECRAFT_MSAA")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|samples| matches!(samples, 1 | 4))
        .unwrap_or(if cfg!(target_os = "macos") { 4 } else { 1 })
}

/// The scene pass's color attachment and resolve target. wgpu rejects a resolve
/// from a single-sampled attachment, so without MSAA the pass draws directly into
/// the texture the viewport later blits from.
fn scene_color_attachment<'a, T>(
    samples: u32,
    color: &'a T,
    resolve: &'a T,
) -> (&'a T, Option<&'a T>) {
    if samples > 1 { (color, Some(resolve)) } else { (resolve, None) }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GridUniforms {
    eye: [f32; 4],
    target: [f32; 4],
    center: [f32; 4],
    origin: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    back: [f32; 4],
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    params: [f32; 4],
    grid: [f32; 4],
    minor: [f32; 4],
    major: [f32; 4],
    axis_x: [f32; 4],
    axis_y: [f32; 4],
    options: [f32; 4],
}

/// Offset the projected scene in NDC while leaving the camera pivot unchanged.
fn shift_view_proj(mut vp: [[f32; 4]; 4], shift: [f32; 2]) -> [[f32; 4]; 4] {
    // `shift` is an NDC offset. Add it multiplied by clip-space w so the
    // perspective divide leaves a constant screen-space shift. A constant clip
    // offset would scale with 1/w and move the pivot when projection changes.
    for column in &mut vp {
        column[0] += shift[0] * column[3];
        column[1] += shift[1] * column[3];
    }
    vp
}

fn invert4x4(columns: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut rows = [[0.0_f64; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            rows[row][column] = f64::from(columns[column][row]);
        }
        rows[row][4 + row] = 1.0;
    }
    for column in 0..4 {
        let mut pivot = column;
        for row in column + 1..4 {
            if rows[row][column].abs() > rows[pivot][column].abs() {
                pivot = row;
            }
        }
        if rows[pivot][column].abs() < 1.0e-12 {
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
        }
        rows.swap(column, pivot);
        let scale = rows[column][column];
        for cell in &mut rows[column] {
            *cell /= scale;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let factor = rows[row][column];
            for index in 0..8 {
                rows[row][index] -= factor * rows[column][index];
            }
        }
    }
    let mut inverse = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            inverse[column][row] = rows[row][4 + column] as f32;
        }
    }
    inverse
}

const GRID_SHADER: &str = r#"
struct Uniforms {
    eye: vec4<f32>,
    target_pos: vec4<f32>,
    center: vec4<f32>,
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    back: vec4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    params: vec4<f32>,
    grid: vec4<f32>,
    minor: vec4<f32>,
    major: vec4<f32>,
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
    options: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[index];
    var out: VsOut;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = position * 0.5 + 0.5;
    return out;
}

fn line_mask(coord: vec2<f32>, thickness: f32) -> f32 {
    let width = max(fwidth(coord), vec2<f32>(1.0e-6));
    let distance = abs(fract(coord - 0.5) - 0.5) / width;
    return 1.0 - smoothstep(thickness, thickness + 1.0, min(distance.x, distance.y));
}

struct FragOut {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VsOut) -> FragOut {
    let ndc = in.uv * 2.0 - 1.0;
    let near_h = u.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = u.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    if near_h.w <= 0.0 || far_h.w <= 0.0 {
        discard;
    }
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    let direction = far - near;
    if abs(direction.z) < 1.0e-6 {
        discard;
    }
    let t = (u.center.z - near.z) / direction.z;
    if t <= 0.0 {
        discard;
    }
    let world = near + direction * t;
    // Anchor the lattice to the world origin, not the camera-following centre.
    // The centre only moves the fade region, so panning moves the grid with the
    // model instead of keeping the pattern fixed to the screen.
    let coord = world.xy / u.grid.x;
    let major_coord = coord / u.options.x;
    let fine_weight = u.options.w;
    let coarse_weight = 1.0 - fine_weight;
    // Crossfade two complete grid levels, not just the major-line mask:
    //   fine level:   F minor + 5F major
    //   coarse level: 5F minor + 25F major
    let fine_minor = line_mask(coord, 0.62) * fine_weight;
    let fine_major = line_mask(major_coord, 0.88) * fine_weight;
    let coarse_minor = line_mask(major_coord, 0.62) * coarse_weight;
    let coarse_major = line_mask(major_coord * 0.2, 0.88) * coarse_weight;
    let minor = fine_minor + coarse_minor;
    let major = fine_major + coarse_major;
    let total = max(minor + major, 1.0e-6);
    // Upstream-style distance fade on the grid plane: disappear before the
    // grid edge and before the camera's far plane cuts the floor.
    var fade = 1.0;
    if (u.right.w > 0.5) {
        // Perspective: fade along view depth so the floor dissolves toward the
        // horizon consistently at every yaw/pitch. `axis_y.w` carries the far
        // plane; keep the fade comfortably inside it.
        let depth = dot(world - u.eye.xyz, -u.back.xyz);
        let start = min(u.params.y * 1.75, u.axis_y.w * 0.55);
        let end = max(
            start + u.params.y * 0.5,
            min(u.params.y * 8.0, u.axis_y.w * 0.92),
        );
        fade = 1.0 - smoothstep(start, end, depth);
    } else {
        // Orthographic uses a finite grid with a radial boundary fade, plus
        // an edge-on fade for the always-XY floor.
        fade = select(
            0.0,
            1.0 - smoothstep(u.options.y, u.options.z, length(world.xy - u.center.xy)),
            u.options.z > u.options.y,
        );
        let low_angle = smoothstep(0.04, 0.22, abs(u.back.z));
        fade *= low_angle;
    }

    let world_x = abs(world.y - u.origin.y) / max(fwidth(world.y), 1.0e-6);
    let world_y = abs(world.x - u.origin.x) / max(fwidth(world.x), 1.0e-6);
    let x_axis = select(
        0.0,
        1.0 - smoothstep(0.5, 1.5, world_x),
        world.x >= u.origin.x,
    );
    let y_axis = select(
        0.0,
        1.0 - smoothstep(0.5, 1.5, world_y),
        world.y >= u.origin.y,
    );
    let axis = max(x_axis, y_axis);

    // Blend the role itself, not just two independent masks: a line that is
    // becoming major smoothly gains major colour as it loses minor weight.
    var colour = mix(u.minor.rgb, u.major.rgb, clamp(major / total, 0.0, 1.0));
    var alpha = (minor * u.grid.y + major * u.grid.z) * fade;
    if x_axis > y_axis {
        colour = mix(colour, u.axis_x.rgb, x_axis);
    } else {
        colour = mix(colour, u.axis_y.rgb, y_axis);
    }
    alpha = max(alpha, axis * 0.9 * fade);
    if alpha < 0.004 {
        discard;
    }
    let clip = u.view_proj * vec4<f32>(world, 1.0);
    if clip.w <= 0.0 {
        discard;
    }
    var out: FragOut;
    out.colour = vec4<f32>(colour, alpha);
    out.depth = clip.z / clip.w;
    return out;
}
"#;

/// Unlit ghost shader. Ghosts are an overlay, not material: keep the authored
/// colour and opacity, with no lighting, roughness or metallic response.
pub(crate) const GHOST_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct GhostIn {
    @location(2) colour: vec3<f32>,
    @location(3) alpha: f32,
};

fn encode(c: vec3<f32>) -> vec3<f32> {
    return select(c * 12.92, 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), c > vec3<f32>(0.0031308));
}

@fragment
fn fs(in: GhostIn) -> @location(0) vec4<f32> {
    return vec4<f32>(encode(in.colour), clamp(u.params.x * in.alpha, 0.0, 1.0));
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectVertex {
    position: [f32; 3],
    normal: [f32; 3],
    colour: [f32; 3],
}

/// CPU-side object geometry prepared away from the UI thread. `face_ids` is
/// parallel to `vertices` and carries the same pick encoding the plan chunks
/// use, so the object renderer can write face ids without any per-pick work.
#[derive(Default)]
pub(crate) struct PreparedObject {
    pub(crate) vertices: Vec<ObjectVertex>,
    pub(crate) face_ids: Vec<u32>,
    pub(crate) indices: Vec<u32>,
}

/// Pick ids for the triangles of one display mesh, derived from the mesh's own
/// face strings (`face_ids` selected by `tri_face`). A merged multi-body mesh
/// carries `body:N:` prefixes and therefore uses the same `body:N:face:M`
/// strings the plan chunks produce for multiple bodies.
struct ObjectFaceIds {
    /// Pick id by display face index (`tri_face` value).
    by_face: Vec<u32>,
    /// Body index by display face index, parsed from the `body:N:` prefix.
    by_face_body: Vec<u32>,
    /// B-rep face number by display face index (`face:N`). Synthetic names
    /// fall back to the display face index, matching the pick id fallback.
    by_face_number: Vec<u32>,
    /// Whether this display mesh names more than one body.
    multiple: bool,
}

impl ObjectFaceIds {
    fn from_display(display: &DisplayMesh) -> Self {
        // A merged mesh always carries `body:N:` prefixes, even when it holds a
        // single body. Count the bodies rather than the prefixes so the
        // `multiple` bit matches the plan path's `plans.len() > 1`.
        let bodies: std::collections::BTreeSet<u32> =
            display.face_ids.iter().map(|name| display_face_body(name)).collect();
        let multiple = bodies.len() > 1;
        let mut by_face = Vec::with_capacity(display.face_ids.len());
        let mut by_face_body = Vec::with_capacity(display.face_ids.len());
        let mut by_face_number = Vec::with_capacity(display.face_ids.len());
        for (index, name) in display.face_ids.iter().enumerate() {
            let (body, face) = parse_display_face(name).unwrap_or_else(|| {
                // A synthetic group name has no numeric B-rep face. Keep the
                // body pickable by its display face index instead of dropping
                // every id in the mesh.
                (display_face_body(name), index as u32)
            });
            by_face.push(crate::gpu_pick::face_pick_id(body, face, multiple));
            by_face_body.push(body);
            by_face_number.push(face);
        }
        Self { by_face, by_face_body, by_face_number, multiple }
    }

    /// A mesh with no display face strings (the coarse LOD upload). Its
    /// `tri_face` is the only identity available.
    fn from_mesh(mesh: &CraftMesh) -> Self {
        let faces = mesh.tri_face.iter().copied().max().unwrap_or(0);
        let by_face =
            (0..=faces).map(|face| crate::gpu_pick::face_pick_id(0, face, false)).collect();
        Self {
            by_face,
            by_face_body: vec![0; faces as usize + 1],
            by_face_number: (0..=faces).collect(),
            multiple: false,
        }
    }

    fn pick(&self, mesh: &CraftMesh, triangle: usize) -> u32 {
        let face = self.face_index(mesh, triangle);
        self.by_face
            .get(face as usize)
            .copied()
            .unwrap_or_else(|| crate::gpu_pick::face_pick_id(0, face, self.multiple))
    }

    /// `(body index, B-rep face number)` for a triangle.
    fn body_face(&self, mesh: &CraftMesh, triangle: usize) -> (u32, u32) {
        let face = self.face_index(mesh, triangle) as usize;
        (
            self.by_face_body.get(face).copied().unwrap_or(0),
            self.by_face_number.get(face).copied().unwrap_or(face as u32),
        )
    }

    fn face_index(&self, mesh: &CraftMesh, triangle: usize) -> u32 {
        mesh.tri_face.get(triangle).copied().unwrap_or(triangle as u32)
    }
}

/// A display mesh without plan identities (an import or a coarse LOD) still
/// needs deterministic per-body colours. This synthetic key keeps bodies
/// distinct; plan-backed meshes carry the real identity in `body_ids`.
fn fallback_body_id(body: u32) -> u64 {
    nodecraft_display::body_instance_id(nodecraft_display::BodyKey::new(0, body), 0)
}

/// The stable identity for one body of a display mesh.
fn display_body_id(display: &DisplayMesh, body: u32) -> u64 {
    display.body_ids.get(body as usize).copied().unwrap_or_else(|| fallback_body_id(body))
}

/// The body a display face string belongs to, even when its face part is not
/// numeric (`body:2:blend:x` -> body 2). Unprefixed strings belong to body 0.
fn display_face_body(name: &str) -> u32 {
    name.strip_prefix("body:")
        .and_then(|rest| rest.split_once(':'))
        .and_then(|(body, _)| body.parse().ok())
        .unwrap_or(0)
}

/// Parse a display face string in the numeric B-rep form the pick encoding can
/// round-trip: `face:N` or `body:N:face:M`.
fn parse_display_face(name: &str) -> Option<(u32, u32)> {
    if let Some(rest) = name.strip_prefix("body:") {
        let (body, rest) = rest.split_once(':')?;
        let face = rest.strip_prefix("face:")?;
        if face.contains(':') {
            return None;
        }
        Some((body.parse().ok()?, face.parse().ok()?))
    } else {
        let face = name.strip_prefix("face:")?;
        if face.contains(':') {
            return None;
        }
        Some((0, face.parse().ok()?))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ObjectUniforms {
    view_proj: [[f32; 4]; 4],
    back: [f32; 4],
    eye: [f32; 4],
    // x = object alpha, yzw reserved.
    params: [f32; 4],
    /// Interactive placement applied after the per-instance model matrix.
    extra_model: [[f32; 4]; 4],
}

impl ObjectUniforms {
    pub(crate) fn with_extra(&self, extra_model: [[f32; 4]; 4]) -> Self {
        Self { extra_model, ..*self }
    }

    pub(crate) fn view_proj(&self) -> [[f32; 4]; 4] {
        self.view_proj
    }

    pub(crate) fn pick_uniforms(
        &self,
        view_proj: [[f32; 4]; 4],
        extra_model: [[f32; 4]; 4],
    ) -> Self {
        Self { view_proj, extra_model, ..*self }
    }
}

const OBJECT_VERTEX_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct In {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) colour: vec3<f32>,
    // Per-instance transform, supplied as four column vectors because WGSL
    // vertex inputs are scalars/vectors. Locations 3..6 are the model matrix
    // (column-major) and 7..10 the inverse-transpose normal matrix.
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) normal_0: vec4<f32>,
    @location(8) normal_1: vec4<f32>,
    @location(9) normal_2: vec4<f32>,
    @location(10) normal_3: vec4<f32>,
    @location(11) instance_alpha: f32,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) colour: vec3<f32>,
    @location(3) alpha: f32,
};

@vertex
fn vs(in: In) -> Out {
    let model = u.extra_model * mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let normal_matrix = u.extra_model * mat4x4<f32>(in.normal_0, in.normal_1, in.normal_2, in.normal_3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: Out;
    out.position = u.view_proj * world;
    out.normal = (normal_matrix * vec4<f32>(in.normal, 0.0)).xyz;
    out.world = world.xyz;
    out.colour = in.colour;
    out.alpha = in.instance_alpha;
    return out;
}
"#;

/// Plan vertex shader. Instead of a per-chunk per-vertex colour buffer it reads
/// the B-rep face from the pick id the plan already carries and the body from
/// the per-instance body buffer, then looks the colour up in one shared table.
const PLAN_VERTEX_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
    extra_model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var<storage, read> colours: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> body_offsets: array<u32>;

struct In {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // The plan pick id: FACE_BASE and the multi-body flag in the high bits, the
    // B-rep face in the low 20 bits.
    @location(2) face_id: u32,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) normal_0: vec4<f32>,
    @location(8) normal_1: vec4<f32>,
    @location(9) normal_2: vec4<f32>,
    @location(10) normal_3: vec4<f32>,
    @location(11) instance_alpha: f32,
    @location(12) body: u32,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) colour: vec3<f32>,
    @location(3) alpha: f32,
};

@vertex
fn vs(in: In) -> Out {
    let model = u.extra_model * mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let normal_matrix = u.extra_model * mat4x4<f32>(in.normal_0, in.normal_1, in.normal_2, in.normal_3);
    let world = model * vec4<f32>(in.position, 1.0);
    var out: Out;
    out.position = u.view_proj * world;
    out.normal = (normal_matrix * vec4<f32>(in.normal, 0.0)).xyz;
    out.world = world.xyz;
    // The face id buffer is the pick encoding; its low 20 bits are the B-rep
    // face. Each body's table slice is sized for its plan's face count, so the
    // index is always in range (a missing face id is 0, i.e. face 0).
    let face = in.face_id & 0xFFFFFu;
    out.colour = colours[body_offsets[in.body] + face].rgb;
    out.alpha = in.instance_alpha;
    return out;
}
"#;

const OBJECT_FRAGMENT_SHADER: &str = r#"
fn encode(c: vec3<f32>) -> vec3<f32> {
    return select(c * 12.92, 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), c > vec3<f32>(0.0031308));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn distribution_ggx(nh: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

fn geometry_schlick_ggx(nv: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return nv / (nv * (1.0 - k) + k);
}

fn fresnel_schlick(cosine: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

fn eval_pbr(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    base: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let nv = max(dot(n, v), 1.0e-4);
    let nl = max(dot(n, l), 0.0);
    if nl <= 0.0 { return vec3<f32>(0.0); }
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);

    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let fresnel = fresnel_schlick(vh, f0);
    let distribution = distribution_ggx(nh, roughness);
    let geometry = geometry_schlick_ggx(nv, roughness)
        * geometry_schlick_ggx(nl, roughness);
    let specular = (distribution * geometry) / max(4.0 * nv * nl, 1.0e-4);
    let diffuse = base * (1.0 - metallic) * (1.0 / 3.14159265);
    return (diffuse * (vec3<f32>(1.0) - fresnel) + specular * fresnel) * nl;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    var v = normalize(u.back.xyz);
    if (u.eye.w > 0.5) { v = normalize(u.eye.xyz - in.world); }
    if (dot(n, v) < 0.0) { n = -n; }

    if (u.params.y > 0.5) {
        // Zebra analysis is a perfect mirror of a procedural striped skybox.
        // Sample the reflected view ray in spherical coordinates instead of
        // shading the surface: the environment itself is high-contrast vertical
        // bands, so a smooth surface reflects smooth bands and a crease makes
        // them break. This is deliberately independent of metallic/roughness.
        let reflected = reflect(-v, n);
        let longitude = atan2(reflected.x, -reflected.z);
        let latitude = asin(clamp(reflected.y, -1.0, 1.0));
        let phase = longitude * 14.0 + latitude * 3.0;
        let wave = sin(phase);
        let aa = clamp(fwidth(wave) * 0.9, 0.004, 0.3);
        let stripe = smoothstep(-aa, aa, wave);
        let tone = mix(0.0, 1.0, stripe);
        return vec4<f32>(encode(vec3<f32>(tone)), u.params.x * in.alpha);
    }

    let base = in.colour;
    let metallic = clamp(u.params.w, 0.0, 1.0);
    let roughness = clamp(u.params.z, 0.045, 1.0);
    let nv = max(dot(n, v), 1.0e-4);
    let f0 = mix(vec3<f32>(0.04), base, metallic);

    // Swift viewer light rig: hemisphere + key/fill/rim/bounce.
    let sky = vec3<f32>(0.95, 0.94, 0.925);
    let ground = vec3<f32>(0.23, 0.23, 0.23);
    let hemisphere = mix(ground, sky, clamp(n.z * 0.5 + 0.5, 0.0, 1.0));

    var radiance = vec3<f32>(0.0);
    let lights = array<vec4<f32>, 4>(
        vec4<f32>(normalize(vec3<f32>(-5.0, 7.0, 8.0)), 3.0),
        vec4<f32>(normalize(vec3<f32>(6.0, -3.0, 4.0)), 0.62),
        vec4<f32>(normalize(vec3<f32>(3.0, 5.0, -7.0)), 1.02),
        vec4<f32>(normalize(vec3<f32>(-2.0, -7.0, -2.0)), 0.3),
    );
    let light_colours = array<vec3<f32>, 4>(
        vec3<f32>(1.0, 0.976, 0.949),
        vec3<f32>(0.871, 0.871, 0.878),
        vec3<f32>(0.957, 0.949, 0.933),
        vec3<f32>(0.812, 0.812, 0.812),
    );
    for (var index = 0u; index < 4u; index += 1u) {
        radiance += eval_pbr(n, v, lights[index].xyz, base, metallic, roughness)
            * light_colours[index] * lights[index].w;
    }

    // Cheap analytic ambient: diffuse irradiance plus a rough environment
    // specular term, appropriate for the studio rig and a CAD viewer.
    let diffuse_ambient = base * (1.0 - metallic) * hemisphere * 0.55;
    let reflection = reflect(-v, n);
    let environment = mix(ground, sky, clamp(reflection.z * 0.5 + 0.5, 0.0, 1.0));
    let ambient_specular = environment * fresnel_schlick(nv, f0)
        * (1.0 - roughness * 0.8) * 1.1;

    let mapped = aces(radiance + diffuse_ambient + ambient_specular + vec3<f32>(0.005));
    return vec4<f32>(encode(mapped), u.params.x * in.alpha);
}
"#;

/// The object renderer's PBR shader: per-vertex colour, no face table.
pub(crate) fn object_shader() -> String {
    [OBJECT_VERTEX_SHADER, OBJECT_FRAGMENT_SHADER].concat()
}

/// The plan renderer's PBR shader: the same fragment code, but the vertex stage
/// looks the colour up from the shared face-colour table.
pub(crate) fn plan_object_shader() -> String {
    [PLAN_VERTEX_SHADER, OBJECT_FRAGMENT_SHADER].concat()
}

pub(crate) use crate::viewport::DEFAULT_FACE_COLOUR;

/// HSL hue step between adjacent faces. The golden-ratio conjugate spreads
/// successive face indices as far apart as possible, so a fan of faces never
/// lands on two neighbouring hues.
const FACE_HUE_STEP: f64 = 0.618_033_988_749_894_9;
/// Saturation and lightness stay fixed inside the old palette's brightness
/// band, so bodies and faces vary in hue only (the renderer's PBR lighting then
/// supplies shading).
const FACE_SATURATION: f64 = 0.58;
const FACE_LIGHTNESS: f64 = 0.52;

fn hsl_to_rgb(hue: f64, saturation: f64, lightness: f64) -> [f32; 3] {
    let chroma = (1.0 - (2.0 * lightness - 1.0).abs()) * saturation;
    let sector = hue.rem_euclid(1.0) * 6.0;
    let x = chroma * (1.0 - (sector % 2.0 - 1.0).abs());
    let (red, green, blue) = match sector as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let offset = lightness - chroma * 0.5;
    [(red + offset) as f32, (green + offset) as f32, (blue + offset) as f32]
}

/// One body's face colour in Face colours mode.
///
/// The hue is a hash of the stable body instance identity (the Part's
/// provenance key plus the copy ordinal, see
/// [`nodecraft_display::body_instance_id`]) offset by golden-ratio steps per
/// face. Colours therefore differ per body *and* per face, and the same
/// `(body, face)` keeps its colour when the plan list is rebuilt or reordered.
pub(crate) fn face_colour(body: u64, face: u32) -> [f32; 3] {
    // SplitMix64 finalizer: a stable, well-mixed hue seed from the body id.
    let mut seed = body;
    seed ^= seed >> 30;
    seed = seed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed ^= seed >> 27;
    seed = seed.wrapping_mul(0x94d0_49bb_1331_11eb);
    seed ^= seed >> 31;
    let body_hue = seed as f64 / u64::MAX as f64;
    let hue = body_hue + f64::from(face) * FACE_HUE_STEP;
    hsl_to_rgb(hue, FACE_SATURATION, FACE_LIGHTNESS)
}

/// Split a display mesh into chunks small enough for wgpu's per-buffer limit.
///
/// Indices are remapped to each chunk's local vertex array. A vertex shared by
/// triangles that land in different chunks is duplicated there.
///
/// This is a temporary workaround for uploading one full-resolution mesh. It
/// should become unnecessary once display LOD selects a bounded mesh per view,
/// because the camera will never need every triangle of a huge scene at once.
fn chunk_object_mesh(
    vertices: Vec<ObjectVertex>,
    face_ids: Vec<u32>,
    indices: Vec<u32>,
) -> Vec<(Vec<ObjectVertex>, Vec<u32>, Vec<u32>)> {
    const MAX_CHUNK_VERTICES: usize = 5_000_000;
    chunk_object_mesh_with_limit(vertices, face_ids, indices, MAX_CHUNK_VERTICES)
}

fn chunk_object_mesh_with_limit(
    vertices: Vec<ObjectVertex>,
    face_ids: Vec<u32>,
    indices: Vec<u32>,
    max_vertices: usize,
) -> Vec<(Vec<ObjectVertex>, Vec<u32>, Vec<u32>)> {
    let mut chunks: Vec<(Vec<ObjectVertex>, Vec<u32>, Vec<u32>)> = Vec::new();
    let mut chunk_vertices: Vec<ObjectVertex> = Vec::new();
    let mut chunk_face_ids: Vec<u32> = Vec::new();
    let mut chunk_indices: Vec<u32> = Vec::new();
    let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for triangle in indices.chunks(3) {
        if chunk_vertices.len() + triangle.len() > max_vertices && !chunk_indices.is_empty() {
            chunks.push((
                std::mem::take(&mut chunk_vertices),
                std::mem::take(&mut chunk_face_ids),
                std::mem::take(&mut chunk_indices),
            ));
            remap.clear();
        }
        for &source in triangle {
            let Some(vertex) = vertices.get(source as usize) else { continue };
            let local = *remap.entry(source).or_insert_with(|| {
                chunk_vertices.push(*vertex);
                chunk_face_ids.push(face_ids.get(source as usize).copied().unwrap_or(0));
                (chunk_vertices.len() - 1) as u32
            });
            chunk_indices.push(local);
        }
    }
    if !chunk_indices.is_empty() {
        chunks.push((chunk_vertices, chunk_face_ids, chunk_indices));
    }
    chunks
}

pub(crate) fn prepare_object(display: &DisplayMesh, view: &ViewSettings) -> PreparedObject {
    let mesh: &CraftMesh = &display.mesh;
    let faces = ObjectFaceIds::from_display(display);
    // The app draws the solid material itself (see ObjectRenderer) so it can use
    // the Swift viewer's light rig and tone mapping. The SolidCraft scene keeps
    // only the edges; the blurred backdrop keeps a PBR copy for defocused colour.
    let ignore_look = view.ignore_look_colors;
    let uniform = if view.face_colours {
        None
    } else if ignore_look || display.face_colors.is_empty() {
        Some(DEFAULT_FACE_COLOUR)
    } else {
        let first = display.face_colors[0];
        display.face_colors.iter().all(|colour| *colour == first).then_some(first)
    };
    // The smoothed builder repairs coarse tessellation shading by averaging
    // triangle normals. A plan's evaluated normals are already the surface's
    // per-face normals (including one copy per triangle at a pole); rebuilding
    // them would collapse a cone apex back onto its axis. Keep those normals and
    // use the per-triangle path below.
    if let Some(prepared) = uniform
        .filter(|_| !display.exact_normals)
        .and_then(|colour| smoothed_object_mesh(mesh, colour, &faces))
    {
        return prepared;
    }
    let mut vertices = Vec::new();
    let mut face_ids = Vec::new();
    let mut indices = Vec::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let [a, b, c] = triangle.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) =
            (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
        else {
            continue;
        };
        let fallback = (*pb - *pa).cross(*pc - *pa);
        let fallback = fallback.normalized().unwrap_or(solidcore_record::Vec3::Z);
        let colour = if view.face_colours {
            let (body, face) = faces.body_face(mesh, triangle_index);
            face_colour(display_body_id(display, body), face)
        } else if ignore_look {
            DEFAULT_FACE_COLOUR
        } else {
            display.face_colors.get(triangle_index).copied().unwrap_or(DEFAULT_FACE_COLOUR)
        };
        let face = faces.pick(mesh, triangle_index);
        for (index, point) in [(a, pa), (b, pb), (c, pc)] {
            let normal = mesh
                .normals
                .get(index)
                .copied()
                .filter(|normal| normal.len() > 1.0e-9)
                .unwrap_or(fallback);
            indices.push(vertices.len() as u32);
            vertices.push(ObjectVertex {
                position: point.to_f32(),
                normal: normal.to_f32(),
                colour,
            });
            face_ids.push(face);
        }
    }
    PreparedObject { vertices, face_ids, indices }
}

struct GpuMeshAsset {
    key: u64,
    vertices: wgpu::Buffer,
    /// One pick id per vertex, parallel to `vertices`.
    face_ids: wgpu::Buffer,
    indices: wgpu::Buffer,
}

struct GpuDrawItem {
    asset_key: u64,
    /// Column-major model matrix. Identity until the display scene supplies
    /// per-instance placement.
    transform: [[f32; 4]; 4],
    /// Index range within the asset. A gizmo preview splits the merged mesh into
    /// moved and stationary body ranges.
    index_start: u32,
    index_count: u32,
    /// Explicit ghost-layer membership. This is intentionally independent of
    /// alpha: the ghost draw must use the ghost pipeline even if the opacity is
    /// temporarily 1.0 (for example while a setting is being initialised).
    ghost: bool,
}

impl GpuDrawItem {
    /// GPU instance payload for this draw: model matrix plus the
    /// inverse-transpose normal matrix that keeps lighting correct under
    /// non-uniform scale.
    fn instance(&self) -> GpuInstance {
        // The dedicated ghost uniform carries opacity; instance alpha stays 1 so
        // the shader does not square the ghost value.
        GpuInstance::from_model(self.transform, 1.0)
    }
}

/// Per-draw instance data uploaded to the object renderer's instance buffer.
///
/// Both matrices are column-major `[[f32; 4]; 4]`, matching WGSL `mat4x4<f32>`
/// constructor order. `normal` is the inverse transpose of `model`'s upper-left
/// 3x3 with an identity translation, so it transforms directions, not points.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuInstance {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 4],
    alpha: f32,
    _padding: [f32; 3],
}

impl GpuInstance {
    pub(crate) fn from_model(model: [[f32; 4]; 4], alpha: f32) -> Self {
        Self { model, normal: normal_matrix(model), alpha, _padding: [0.0; 3] }
    }
}

/// Vertex-buffer layout for `GpuInstance`, shared by the opaque and transparent
/// object pipelines. Attribute locations 3 through 10 follow the vertex
/// attributes (0..=2) so the same WGSL `In` struct feeds both pipelines.
const OBJECT_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 9] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 64, shader_location: 7 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 80, shader_location: 8 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 96, shader_location: 9 },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 10,
    },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 128, shader_location: 11 },
];

pub(crate) fn object_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &OBJECT_INSTANCE_ATTRIBUTES,
    }
}

fn object_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ObjectVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
        ],
    }
}

/// Position-only view of the object vertex buffer for the face-id pass. The
/// pick shader reads the face id from its own parallel buffer at location 2, so
/// this layout must not declare the object's colour at that location.
pub(crate) fn object_pick_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    }];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ObjectVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Indexed object geometry for a display mesh with one colour.
///
/// Returns `None` when the mesh has unusable normals; the caller then falls back
/// to the per-triangle expansion that can synthesise a face normal.
/// Non-indexed fallback for meshes without reliable per-vertex normals.
fn flat_object_mesh(
    mesh: &CraftMesh,
    colour: [f32; 3],
    faces: &ObjectFaceIds,
) -> Option<PreparedObject> {
    let mut vertices = Vec::new();
    let mut face_ids = Vec::new();
    let mut indices = Vec::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let [a, b, c] = triangle.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) =
            (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
        else {
            continue;
        };
        let fallback = (*pb - *pa).cross(*pc - *pa);
        let normal = fallback.normalized().unwrap_or(solidcore_record::Vec3::Z).to_f32();
        let face = faces.pick(mesh, triangle_index);
        for point in [pa, pb, pc] {
            indices.push(vertices.len() as u32);
            vertices.push(ObjectVertex { position: point.to_f32(), normal, colour });
            face_ids.push(face);
        }
    }
    Some(PreparedObject { vertices, face_ids, indices })
}

/// Indexed object geometry with crease-aware smooth normals.
///
/// Tessellators often provide faceted normals; zebra analysis then exposes every
/// triangle boundary. Average incident face normals inside a crease angle so
/// smooth faces shade continuously, while splitting vertices at genuine creases.
fn smoothed_object_mesh(
    mesh: &CraftMesh,
    colour: [f32; 3],
    faces: &ObjectFaceIds,
) -> Option<PreparedObject> {
    // Tessellated smooth surfaces can have coarse facets; treat anything under
    // 60° as smooth. Real CAD creases (box edges, rims) are near 90° and still
    // split, so zebra shows genuine discontinuities rather than tessellation
    // seams.
    const CREASE_COS: f64 = 0.5; // 60 degrees
    let key = |p: &Vec3| {
        ((p.x * 1.0e3).round() as i64, (p.y * 1.0e3).round() as i64, (p.z * 1.0e3).round() as i64)
    };
    // Creases follow B-rep faces, not just the angle between triangles. A chamfer
    // face is only 45° from the faces beside it: an angular threshold alone
    // averages them and makes the flat bevel look rounded and dark.
    let group_of = |triangle: usize| mesh.tri_face.get(triangle).copied().unwrap_or(0);
    let mut incident: std::collections::HashMap<((i64, i64, i64), u32), Vec<Vec3>> =
        std::collections::HashMap::new();
    let mut face_normals = Vec::with_capacity(mesh.triangles.len());
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let [a, b, c] = triangle.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) =
            (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
        else {
            face_normals.push(None);
            continue;
        };
        let cross = (*pb - *pa).cross(*pc - *pa);
        let normal = cross.normalized();
        face_normals.push(normal);
        if let Some(normal) = normal {
            let group = group_of(triangle_index);
            for point in [pa, pb, pc] {
                incident.entry((key(point), group)).or_default().push(normal);
            }
        }
    }
    let mut vertices: Vec<ObjectVertex> = Vec::new();
    let mut face_ids: Vec<u32> = Vec::new();
    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    let mut corners: std::collections::HashMap<((i64, i64, i64), u32, [i64; 3]), u32> =
        std::collections::HashMap::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let Some(face) = face_normals.get(triangle_index).copied().flatten() else { continue };
        let pick = faces.pick(mesh, triangle_index);
        for index in triangle {
            let point = mesh.positions.get(*index as usize)?;
            let k = (key(point), group_of(triangle_index));
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for normal in incident.get(&k).into_iter().flatten() {
                if normal.dot(face) >= CREASE_COS {
                    sum = sum + *normal;
                }
            }
            let normal = sum.normalized().unwrap_or(face);
            let normal_key = [
                (normal.x * 1.0e3).round() as i64,
                (normal.y * 1.0e3).round() as i64,
                (normal.z * 1.0e3).round() as i64,
            ];
            let vertex_index = *corners.entry((k.0, k.1, normal_key)).or_insert_with(|| {
                let index = vertices.len() as u32;
                vertices.push(ObjectVertex {
                    position: point.to_f32(),
                    normal: normal.to_f32(),
                    colour,
                });
                face_ids.push(pick);
                index
            });
            indices.push(vertex_index);
        }
    }
    Some(PreparedObject { vertices, face_ids, indices })
}

#[cfg(test)]
fn indexed_object_mesh(
    mesh: &CraftMesh,
    colour: [f32; 3],
    faces: &ObjectFaceIds,
) -> Option<PreparedObject> {
    if mesh.normals.len() < mesh.positions.len() {
        return None;
    }
    let mut vertices = Vec::with_capacity(mesh.positions.len());
    for (point, normal) in mesh.positions.iter().zip(&mesh.normals) {
        if !(normal.len() > 1.0e-9) {
            return None;
        }
        vertices.push(ObjectVertex { position: point.to_f32(), normal: normal.to_f32(), colour });
    }
    let mut face_ids = vec![0u32; vertices.len()];
    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let pick = faces.pick(mesh, triangle_index);
        for index in triangle {
            if *index as usize >= vertices.len() {
                return None;
            }
            face_ids[*index as usize] = pick;
            indices.push(*index);
        }
    }
    Some(PreparedObject { vertices, face_ids, indices })
}

struct PendingObjectMesh {
    key: u64,
    vertices: Vec<ObjectVertex>,
    /// One pick id per vertex, parallel to `vertices`.
    face_ids: Vec<u32>,
    indices: Vec<u32>,
}

struct ObjectRenderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    ghost_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    ghost_bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    ghost_uniforms: wgpu::Buffer,
    ghost_alpha: f32,
    assets: Vec<GpuMeshAsset>,
    /// One entry per uploaded chunk: (asset key, triangle count). Large meshes
    /// are split so no single wgpu buffer exceeds the device limit.
    chunk_spans: Vec<(u64, u32)>,
    draws: Vec<GpuDrawItem>,
    /// One `GpuInstance` per draw, in draw order. Rebuilt in `prepare` whenever
    /// the draw list changes.
    instances: Option<wgpu::Buffer>,
    instances_dirty: bool,
    pending: Option<Vec<PendingObjectMesh>>,
    next_key: u64,
    uploaded_triangles: u64,
    uploaded_object_bytes: u64,
}

impl ObjectRenderer {
    fn install(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
        reversed: bool,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-object"),
            source: wgpu::ShaderSource::Wgsl(object_shader().into()),
        });
        let ghost_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-object-ghost"),
            source: wgpu::ShaderSource::Wgsl(GHOST_SHADER.into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-object-uniforms"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-object-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-object-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let ghost_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-object-ghost-uniforms"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ghost_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-object-ghost-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ghost_uniforms.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-object-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-object-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(object_vertex_layout()), Some(object_instance_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Surfaces are two-sided. An open sheet (Surface Loft, etc.) can
                // wind away from the camera, and back-face culling would make it
                // disappear; closed solids are unaffected because their back
                // faces are hidden by depth. The fragment shader flips the
                // normal for back-facing fragments.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(opaque_face_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let transparent_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-object-transparent-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(object_vertex_layout()), Some(object_instance_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(translucent_face_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-object-ghost-layer-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(object_vertex_layout()), Some(object_instance_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ghost_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(ghost_face_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            transparent_pipeline,
            ghost_pipeline,
            bind_group,
            ghost_bind_group,
            uniforms,
            ghost_uniforms,
            ghost_alpha: 1.0,
            assets: Vec::new(),
            chunk_spans: Vec::new(),
            draws: Vec::new(),
            instances: None,
            instances_dirty: false,
            pending: None,
            next_key: 1,
            uploaded_triangles: 0,
            uploaded_object_bytes: 0,
        }
    }

    fn set_mesh(&mut self, vertices: Vec<ObjectVertex>, face_ids: Vec<u32>, indices: Vec<u32>) {
        debug_assert_eq!(
            vertices.len(),
            face_ids.len(),
            "the object mesh must carry one face id per vertex",
        );
        // A 10k-tree scene can exceed wgpu's 256 MiB buffer limit in one vertex
        // buffer. Split the mesh into bounded chunks; each chunk gets its own
        // vertex/index/face-id buffers and draw range, then the instance buffer
        // keeps them in one logical mesh.
        let uploaded_triangles = indices.len() as u64 / 3;
        let uploaded_object_bytes = (vertices.len() * std::mem::size_of::<ObjectVertex>()
            + face_ids.len() * std::mem::size_of::<u32>()
            + indices.len() * std::mem::size_of::<u32>())
            as u64;
        let chunks = chunk_object_mesh(vertices, face_ids, indices);
        self.chunk_spans.clear();
        self.draws.clear();
        self.ghost_alpha = 1.0;
        let mut pending = Vec::with_capacity(chunks.len());
        for (chunk_vertices, chunk_face_ids, chunk_indices) in chunks {
            let key = self.next_key;
            self.next_key = self.next_key.wrapping_add(1).max(1);
            let index_count = chunk_indices.len() as u32;
            self.draws.push(GpuDrawItem {
                asset_key: key,
                transform: identity_matrix(),
                index_start: 0,
                index_count,
                ghost: false,
            });
            self.chunk_spans.push((key, index_count / 3));
            pending.push(PendingObjectMesh {
                key,
                vertices: chunk_vertices,
                face_ids: chunk_face_ids,
                indices: chunk_indices,
            });
        }
        self.pending = Some(pending);
        self.instances_dirty = true;
        self.uploaded_triangles = uploaded_triangles;
        self.uploaded_object_bytes = uploaded_object_bytes;
    }

    /// Displace the existing draw list without touching the uploaded mesh. The
    /// instance buffer is re-uploaded in `prepare`; vertex/index buffers stay.
    fn set_transform(&mut self, transform: Option<[[f32; 4]; 4]>, ghost_alpha: f32) {
        self.ghost_alpha = ghost_alpha;
        if self.chunk_spans.is_empty() {
            return;
        }
        self.draws = self
            .chunk_spans
            .iter()
            .map(|(asset_key, triangles)| GpuDrawItem {
                asset_key: *asset_key,
                transform: transform.unwrap_or_else(identity_matrix),
                index_start: 0,
                index_count: triangles * 3,
                ghost: false,
            })
            .collect();
        self.instances_dirty = true;
    }

    fn set_ghost_alpha(&mut self, alpha: f32) {
        self.ghost_alpha = alpha;
    }

    fn debug_info(&self, adapter: &str, backend: &str, device_type: &str) -> GpuDebugInfo {
        GpuDebugInfo {
            adapter: adapter.to_owned(),
            backend: backend.to_owned(),
            device_type: device_type.to_owned(),
            chunks: self.chunk_spans.len(),
            triangles: self.uploaded_triangles,
            object_bytes: self.uploaded_object_bytes,
        }
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &ObjectUniforms) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
        let mut ghost = *uniforms;
        ghost.params[0] = self.ghost_alpha;
        queue.write_buffer(&self.ghost_uniforms, 0, bytemuck::bytes_of(&ghost));
        if let Some(chunks) = self.pending.take() {
            self.assets.clear();
            for mesh in chunks {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }
                let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-vertices"),
                    size: (mesh.vertices.len() * std::mem::size_of::<ObjectVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
                // Face ids upload with their vertices, so a pick can never read
                // ids from an older activation's geometry.
                let face_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-face-ids"),
                    size: (mesh.face_ids.len().max(1) * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if !mesh.face_ids.is_empty() {
                    queue.write_buffer(&face_buffer, 0, bytemuck::cast_slice(&mesh.face_ids));
                }
                let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-indices"),
                    size: (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
                self.assets.push(GpuMeshAsset {
                    key: mesh.key,
                    vertices: vertex_buffer,
                    face_ids: face_buffer,
                    indices: index_buffer,
                });
            }
        }
        if self.instances_dirty {
            self.instances_dirty = false;
            self.instances = if self.draws.is_empty() {
                None
            } else {
                let instances: Vec<GpuInstance> =
                    self.draws.iter().map(GpuDrawItem::instance).collect();
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-instances"),
                    size: (instances.len() * std::mem::size_of::<GpuInstance>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&instances));
                Some(buffer)
            };
        }
    }

    fn paint(&self, pass: &mut wgpu::RenderPass<'static>, transparent: bool, draw_opaque: bool) {
        let Some(instances) = self.instances.as_ref() else { return };
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(1, instances.slice(..));
        let emit = |draw: &GpuDrawItem, pass: &mut wgpu::RenderPass<'static>, index: usize| {
            let Some(asset) = self.assets.iter().find(|asset| asset.key == draw.asset_key) else {
                return;
            };
            pass.set_vertex_buffer(0, asset.vertices.slice(..));
            pass.set_index_buffer(asset.indices.slice(..), wgpu::IndexFormat::Uint32);
            // Each draw reads its own slot in the shared instance buffer.
            pass.draw_indexed(
                draw.index_start..draw.index_start + draw.index_count,
                0,
                index as u32..index as u32 + 1,
            );
        };
        if transparent {
            pass.set_pipeline(&self.transparent_pipeline);
            for (index, draw) in self.draws.iter().enumerate() {
                emit(draw, pass, index);
            }
            return;
        }
        if draw_opaque {
            pass.set_pipeline(&self.pipeline);
            for (index, draw) in self.draws.iter().enumerate() {
                if !draw.ghost {
                    emit(draw, pass, index);
                }
            }
        }
        // Ghost layer: blended, depth-test-disabled, after the opaque scene.
        pass.set_pipeline(&self.ghost_pipeline);
        pass.set_bind_group(0, &self.ghost_bind_group, &[]);
        for (index, draw) in self.draws.iter().enumerate() {
            if draw.ghost {
                emit(draw, pass, index);
            }
        }
    }

    /// Draw this mesh's face ids with the pick pass's ID pipeline. The vertex,
    /// face-id and instance buffers are the ones the visible object draw uses,
    /// so the picked face is exactly the face under the cursor regardless of the
    /// preview matrix currently held in the instance buffer.
    fn paint_face_ids(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
    ) {
        let Some(instances) = self.instances.as_ref() else { return };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.set_vertex_buffer(2, instances.slice(..));
        for (index, draw) in self.draws.iter().enumerate() {
            if draw.ghost {
                continue;
            }
            let Some(asset) = self.assets.iter().find(|asset| asset.key == draw.asset_key) else {
                continue;
            };
            pass.set_vertex_buffer(0, asset.vertices.slice(..));
            pass.set_vertex_buffer(1, asset.face_ids.slice(..));
            pass.set_index_buffer(asset.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                draw.index_start..draw.index_start + draw.index_count,
                0,
                index as u32..index as u32 + 1,
            );
        }
    }
}

/// Draw the face ids for this frame from the one renderer that drew the faces.
/// `plan_active` is the same `PlanState::is_available()` value that selected the
/// plan pass over the object pass, so picking cannot disagree with the picture.
fn paint_pick_face_ids(
    plan_active: bool,
    plan_edges: bool,
    plan: Option<&crate::gpu_plan::PlanState>,
    object: Option<&ObjectRenderer>,
    pass: &mut wgpu::RenderPass<'_>,
    passes: &crate::gpu_pick::PickPasses<'_>,
    binds: &crate::gpu_pick::PickBinds<'_>,
) {
    if plan_active {
        if let Some(plan) = plan {
            plan.paint_face_ids(pass, passes.face_id, binds.main);
            plan.paint_ghost_depth(pass, passes.ghost_depth, binds.ghost);
            if plan_edges {
                plan.paint_edge_ids(pass, passes.plan_edge, binds.plan_edge);
            }
        }
    } else if let Some(object) = object {
        object.paint_face_ids(pass, passes.object_face_id, binds.main);
    }
}

pub(crate) fn identity_matrix() -> [[f32; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

/// A pipeline's depth state, kept as a value so tests can assert the model
/// directly instead of inspecting a compiled pipeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DepthPlan {
    pub write: bool,
    pub compare: wgpu::CompareFunction,
    pub bias: wgpu::DepthBiasState,
}

impl DepthPlan {
    pub(crate) fn state(self, format: wgpu::TextureFormat) -> wgpu::DepthStencilState {
        wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(self.write),
            depth_compare: Some(self.compare),
            stencil: wgpu::StencilState::default(),
            bias: self.bias,
        }
    }
}

/// Highlights sit on the face they copy, so they are pulled a few depth steps
/// toward the eye. SolidCraft's `HIGHLIGHT_BIAS` is the same value; it is
/// private there, so derive it from the public face bias instead of copying a
/// second magic number.
pub(crate) const HIGHLIGHT_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: -FACE_BIAS.constant,
    slope_scale: -FACE_BIAS.slope_scale,
    clamp: 0.0,
};

/// Removed preview material is pushed twice as far as an opaque face, so it
/// stays behind the material it shares with the new shape. Matches SolidCraft's
/// private `GHOST_BIAS`.
pub(crate) const GHOST_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 2 * FACE_BIAS.constant,
    slope_scale: 2.0 * FACE_BIAS.slope_scale,
    clamp: 0.0,
};

/// Opaque faces are pushed away from the eye, so edges and other things on the
/// face draw over them at any angle while an edge behind another body stays
/// hidden. This is SolidCraft's `FACE_BIAS` model (#1434) applied to the
/// NodeCraft plan and object passes.
pub(crate) fn opaque_face_depth(reversed: bool) -> DepthPlan {
    DepthPlan {
        write: true,
        compare: depth_compare(reversed, wgpu::CompareFunction::LessEqual),
        bias: depth_bias(reversed, FACE_BIAS),
    }
}

/// Ghost faces are an overlay: NodeCraft has always drawn them with `Always`
/// and no depth write so the breakpoint body underneath cannot z-fight with
/// them. The map in agents/review/2026-11-18-edge-ribbon-unify-map.md records
/// this as the one deliberate difference from SolidCraft's ghost pass.
pub(crate) fn ghost_face_depth(_reversed: bool) -> DepthPlan {
    DepthPlan {
        write: false,
        compare: wgpu::CompareFunction::Always,
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Translucent faces are depth-tested but write no depth, so edges behind them
/// remain visible. No bias: they are not a surface edges must win against.
pub(crate) fn translucent_face_depth(reversed: bool) -> DepthPlan {
    DepthPlan {
        write: false,
        compare: depth_compare(reversed, wgpu::CompareFunction::LessEqual),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Edges are tested against the same depth buffer as the faces but never
/// write or bias it: the faces are pushed back, not the lines pulled forward.
pub(crate) fn edge_depth(reversed: bool) -> DepthPlan {
    DepthPlan {
        write: false,
        compare: depth_compare(reversed, wgpu::CompareFunction::LessEqual),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Selected and hovered edge copies are pulled slightly toward the eye so they
/// cover the body's own edge at the same place.
pub(crate) fn edge_highlight_depth(reversed: bool) -> DepthPlan {
    DepthPlan {
        write: false,
        compare: depth_compare(reversed, wgpu::CompareFunction::LessEqual),
        bias: depth_bias(reversed, HIGHLIGHT_BIAS),
    }
}

/// Inverse transpose of the upper-left 3x3 of a column-major model matrix.
///
/// Normals transform by `(M^-1)^T` so lighting stays correct under non-uniform
/// scale. The translation column is dropped (normals are directions) and a
/// degenerate model falls back to identity rather than poisoning the shader
/// with NaNs.
fn normal_matrix(model: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Row-major copy of the upper-left 3x3.
    let m = [
        [model[0][0], model[1][0], model[2][0]],
        [model[0][1], model[1][1], model[2][1]],
        [model[0][2], model[1][2], model[2][2]],
    ];
    let cofactor = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
        ],
        [
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
        ],
        [
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ];
    let determinant =
        m[0][0] * cofactor[0][0] + m[0][1] * cofactor[0][1] + m[0][2] * cofactor[0][2];
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return identity_matrix();
    }
    let inverse_determinant = 1.0 / determinant;
    let mut out = identity_matrix();
    // (M^-1)^T = cofactor(M) / det, stored column-major like the model.
    for row in 0..3 {
        for column in 0..3 {
            out[column][row] = cofactor[row][column] * inverse_determinant;
        }
    }
    out
}

/// `matrix * point` for a column-major affine matrix.
fn transform_point(matrix: [[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2] + matrix[3][0],
        matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2] + matrix[3][1],
        matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2] + matrix[3][2],
    ]
}

/// Axis-aligned bounds of a box after an affine transform.
fn transform_bounds(bounds: Aabb3, matrix: [[f32; 4]; 4]) -> Aabb3 {
    let corners = [
        [bounds.min.x, bounds.min.y, bounds.min.z],
        [bounds.max.x, bounds.min.y, bounds.min.z],
        [bounds.min.x, bounds.max.y, bounds.min.z],
        [bounds.max.x, bounds.max.y, bounds.min.z],
        [bounds.min.x, bounds.min.y, bounds.max.z],
        [bounds.max.x, bounds.min.y, bounds.max.z],
        [bounds.min.x, bounds.max.y, bounds.max.z],
        [bounds.max.x, bounds.max.y, bounds.max.z],
    ];
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for corner in corners {
        let point = transform_point(matrix, [corner[0] as f32, corner[1] as f32, corner[2] as f32]);
        for axis in 0..3 {
            let value = f64::from(point[axis]);
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    Aabb3 { min: Vec3::new(min[0], min[1], min[2]), max: Vec3::new(max[0], max[1], max[2]) }
}

/// `a * b` for column-major 4x4 matrices (apply `b` first, then `a`).
fn multiply_matrix(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

struct GridRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl GridRenderer {
    fn install(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-grid"),
            source: wgpu::ShaderSource::Wgsl(GRID_SHADER.into()),
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-grid-uniforms"),
            size: std::mem::size_of::<GridUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-grid-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-grid-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-grid-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-grid-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, bind_group, buffer }
    }

    fn prepare(&self, queue: &wgpu::Queue, uniforms: &GridUniforms) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(uniforms));
    }

    fn paint(&self, pass: &mut wgpu::RenderPass<'static>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

const AXIS_SHADER: &str = r#"
struct A {
    view_proj: mat4x4<f32>,
    start: vec4<f32>,
    end: vec4<f32>,
    colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> a: A;

struct V {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vertex: u32) -> V {
    let point = select(a.start, a.end, vertex == 1u);
    var out: V;
    out.position = a.view_proj * point;
    return out;
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return a.colour;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AxisUniforms {
    view_proj: [[f32; 4]; 4],
    start: [f32; 4],
    end: [f32; 4],
    colour: [f32; 4],
}

/// The world Z axis. It is drawn from two uniform points so it never has to be
/// baked into the mesh scene: the scene's projection can then be transformed on
/// the GPU for whole-mesh previews while the world frame stays fixed.
struct AxisRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl AxisRenderer {
    fn install(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-axis"),
            source: wgpu::ShaderSource::Wgsl(AXIS_SHADER.into()),
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-axis-uniforms"),
            size: std::mem::size_of::<AxisUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-axis-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-axis-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-axis-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-axis-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, bind_group, buffer }
    }

    fn prepare(&self, queue: &wgpu::Queue, uniforms: &AxisUniforms) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(uniforms));
    }

    fn paint(&self, pass: &mut wgpu::RenderPass<'static>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..2, 0..1);
    }
}

fn colour(rgba: [u8; 4], alpha: f32) -> [f32; 4] {
    [f32::from(rgba[0]) / 255.0, f32::from(rgba[1]) / 255.0, f32::from(rgba[2]) / 255.0, alpha]
}

const BLIT_SHADER: &str = r#"
struct V {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var out: V;
  out.pos = vec4<f32>(p[i], 0.0, 1.0);
  out.uv = vec2((p[i].x + 1.0) * 0.5, 1.0 - (p[i].y + 1.0) * 0.5);
  return out;
}

@fragment
fn fs(in: V) -> @location(0) vec4<f32> {
  return textureSample(src, smp, in.uv);
}
"#;

struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

pub(crate) struct SceneTarget {
    size: [u32; 2],
    color_view: wgpu::TextureView,
    resolve_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    blit: BlitPipeline,
}

impl SceneTarget {
    fn new(
        device: &wgpu::Device,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
        samples: u32,
    ) -> Self {
        let size = [size[0].max(1), size[1].max(1)];
        let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 };
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport-scene-msaa"),
            size: extent,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolve = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport-scene-resolve"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport-scene-depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format: depth,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let resolve_view = resolve.create_view(&Default::default());
        let depth_view = depth_texture.create_view(&Default::default());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-blit-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport-blit-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-blit-bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-blit-pipeline-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-blit-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // Match egui's depth attachment; the blit neither tests nor writes.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        Self {
            size,
            color_view,
            resolve_view,
            depth_view,
            blit: BlitPipeline { pipeline, bind_group },
        }
    }
}

/// Display-only drag preview state: one world transform and its alpha. This is
/// the single preview value of the GPU display unit: the plan uniform, the
/// scene edges and the drawn projections all derive from it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PreviewState {
    pub(crate) transform: [[f32; 4]; 4],
    pub(crate) alpha: f32,
}

impl PreviewState {
    /// The plan-shader matrix for this preview.
    fn plan_transform(&self) -> [[f32; 4]; 4] {
        self.transform
    }

    /// Apply the preview to a base scene projection.
    fn scene_view_proj(&self, base: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        multiply_matrix(base, self.transform)
    }

    /// The defocused backdrop follows the same preview as the scene.
    fn backdrop_view_proj(&self, base: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        multiply_matrix(base, self.transform)
    }
}

/// The preview one drawn frame uses. It captures the base projections when the
/// callback is built; a staged activation can replace `state` in `prepare`, and
/// every consumer (plan uniform, scene projection, backdrop) is derived from
/// that resulting value, never from a separate owner.
#[derive(Clone, Debug)]
struct DrawnPreview {
    base_view_proj: [[f32; 4]; 4],
    base_backdrop_view_proj: [[f32; 4]; 4],
    state: Option<PreviewState>,
}

impl DrawnPreview {
    fn new(
        base_view_proj: [[f32; 4]; 4],
        base_backdrop_view_proj: [[f32; 4]; 4],
        state: Option<PreviewState>,
    ) -> Self {
        Self { base_view_proj, base_backdrop_view_proj, state }
    }

    fn set_state(&mut self, state: Option<PreviewState>) {
        self.state = state;
    }

    fn plan_transform(&self) -> Option<[[f32; 4]; 4]> {
        self.state.as_ref().map(PreviewState::plan_transform)
    }

    fn scene_view_proj(&self) -> [[f32; 4]; 4] {
        self.state
            .as_ref()
            .map_or(self.base_view_proj, |preview| preview.scene_view_proj(self.base_view_proj))
    }

    fn backdrop_view_proj(&self) -> [[f32; 4]; 4] {
        self.state.as_ref().map_or(self.base_backdrop_view_proj, |preview| {
            preview.backdrop_view_proj(self.base_backdrop_view_proj)
        })
    }
}

/// Make the drawn frame's projection and the plan uniform follow the unit's one
/// preview value. Activation calls this after taking the staged payload, so the
/// scene key, edge scene, plan chunks and preview all change in one `prepare`.
fn sync_drawn_preview(drawn: &mut DrawnPreview, frame: &mut Frame, state: Option<PreviewState>) {
    drawn.set_state(state);
    frame.view_proj = drawn.scene_view_proj();
}

/// Stable GPU/display facts for the debug overlay.
#[derive(Clone, Debug)]
pub struct GpuDebugInfo {
    pub adapter: String,
    pub backend: String,
    pub device_type: String,
    pub chunks: usize,
    pub triangles: u64,
    pub object_bytes: u64,
}

pub(crate) fn resolve_pick(data: &GhostPickData, index: u32) -> Option<(String, [f64; 3])> {
    if index == 0 {
        return None;
    }
    data.edges.get(index as usize).cloned()
}

#[derive(Clone, Default)]
pub(crate) struct GhostPickData {
    pub multi_body: bool,
    /// POD segments; `id` is the pick index (0 = none).
    pub segments: std::sync::Arc<[crate::gpu_pick::PickSegment]>,
    /// Pick index 0 is reserved for "none"; use `resolve_pick`.
    pub edges: Vec<(String, [f64; 3])>,
}

pub(crate) fn build_ghost_pick_data(edges: &[crate::adapter::DisplayEdge]) -> GhostPickData {
    let mut segments: Vec<crate::gpu_pick::PickSegment> = Vec::new();
    let mut data = GhostPickData {
        segments: std::sync::Arc::from(Vec::new()),
        edges: vec![(String::new(), [0.0; 3])],
        multi_body: false,
    };
    for edge in edges {
        if edge.id.starts_with("preview-edge:") {
            continue;
        }
        let index = data.edges.len() as u32;
        if let Some(body) = edge.id.strip_prefix("body:")
            && let Some((number, _)) = body.split_once(':')
            && number != "0"
        {
            data.multi_body = true;
        }
        data.edges.push((edge.id.clone(), edge.midpoint));
        for pair in edge.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let d = b - a;
            if d.dot(d) <= 1.0e-20 {
                continue;
            }
            segments.push(crate::gpu_pick::PickSegment {
                a: [a.x as f32, a.y as f32, a.z as f32],
                id: index,
                b: [b.x as f32, b.y as f32, b.z as f32],
                class: edge.class.bits(),
            });
        }
    }
    data.segments = std::sync::Arc::from(segments);
    data
}

/// A display payload that must become visible in the same activation as its
/// plan set: edge draw data, object mesh, picking display, diagnostics and the
/// displayed target. The payload waits for `PlanState::activation` instead of
/// being installed by the app on a later frame.
struct PendingDisplay {
    wait_activation: u64,
    edges: StagedEdges,
    source_mesh: Option<nodecraft_protocol::Mesh>,
    prepared: Option<PreparedObject>,
    diagnostics: Option<Vec<String>>,
    target: Option<String>,
    /// The preview to show with this payload when it activates. `None` retires
    /// it in the same step that installs the authoritative geometry.
    preview: Option<PreviewState>,
}

/// Edge data and its GPU draw payload for one exact snapshot. Today `scene`
/// carries the edge lines uploaded by the SolidCraft scene renderer. A planned
/// edge-ribbon generator can replace `scene` with its own vertex/index buffers
/// here without changing how the payload pairs with the plan set.
struct StagedEdges {
    /// Exact object state that produced these edge polylines. It must equal the
    /// plan set's snapshot when the pair activates.
    snapshot: u64,
    /// Scene key allocated when this payload was staged. The frame that
    /// activates the pair carries this key, so the renderer uploads the new
    /// scene in the same `prepare` call instead of keeping the old key.
    key: u64,
    /// Allocation order of `key`. Sync keeps a newer immediate upload rather
    /// than rolling its key back to an activation that has already been drawn.
    key_serial: u64,
    display: Option<Arc<DisplayMesh>>,
    scene: GpuScene,
    /// World-space fallback edge data for the CPU renderer. The plan path
    /// carries its own local-space edge buffer with each resident plan.
    edge_batch: EdgeCpuBatch,
}

#[derive(Clone)]
pub(crate) struct DisplayActivation {
    pub(crate) snapshot: u64,
    pub(crate) key: u64,
    pub(crate) key_serial: u64,
    pub(crate) display: Option<Arc<DisplayMesh>>,
    pub(crate) source_mesh: Option<nodecraft_protocol::Mesh>,
    pub(crate) diagnostics: Option<Vec<String>>,
    pub(crate) target: Option<String>,
    /// The display unit's preview after this activation. The CPU mirror adopts
    /// it so a retired preview cannot linger as stale app-side state.
    pub(crate) preview: Option<PreviewState>,
}

/// Let the current frame adopt an activation's scene key unless a newer scene
/// has already been handed to the renderer. A later frame must keep the newer
/// key; rolling back would make the renderer re-upload the stale slot and hide
/// the scene that upload is meant to show.
fn adopt_scene_key(key: &mut u64, key_serial: &mut u64, active: &DisplayActivation) {
    if active.key_serial > *key_serial {
        *key = active.key;
        *key_serial = active.key_serial;
    }
}

/// Hand an activated pair to the frame that is about to upload it. The scene
/// and its key travel together: the renderer skips a scene whose key it already
/// has, so a key captured before activation would leave the new edges unsent.
fn apply_activated_display(pending: &PendingDisplay, frame: &mut Frame) {
    frame.key = pending.edges.key;
    if let Ok(mut slot) = frame.slot.lock() {
        *slot = Some(pending.edges.scene.clone());
    }
}

#[derive(Default)]
pub(crate) struct DisplaySwap {
    pending: Option<PendingDisplay>,
    active: Option<(u64, DisplayActivation)>,
    serial: u64,
    /// Snapshot of the edge/object data currently on the GPU. It changes only
    /// with the plan set's activation (or with an immediate resident commit).
    committed_snapshot: u64,
    /// The drag preview currently drawn by this display unit. It is the one
    /// source for the plan uniform and the scene projection of a frame; a
    /// staged payload may replace it in the same activation as its geometry.
    preview: Option<PreviewState>,
}

impl DisplaySwap {
    fn stage(&mut self, pending: PendingDisplay) {
        self.pending = Some(pending);
    }

    /// Take the staged payload once its plan set has activated. The active
    /// copy keeps the CPU mirror and timeline in step with the GPU swap.
    fn take_ready(&mut self, plan_activation: u64, plan_snapshot: u64) -> Option<PendingDisplay> {
        let ready = self.pending.as_ref().is_some_and(|pending| {
            pending.wait_activation <= plan_activation && pending.edges.snapshot == plan_snapshot
        });
        if !ready {
            return None;
        }
        let pending = self.pending.take()?;
        self.serial = self.serial.wrapping_add(1);
        self.committed_snapshot = pending.edges.snapshot;
        // The preview is part of the pair: the new geometry, edges, scene key
        // and preview change in this one activation, so the display cannot snap
        // back between them.
        self.preview.clone_from(&pending.preview);
        self.active = Some((
            self.serial,
            DisplayActivation {
                snapshot: pending.edges.snapshot,
                key: pending.edges.key,
                key_serial: pending.edges.key_serial,
                display: pending.edges.display.clone(),
                source_mesh: pending.source_mesh.clone(),
                diagnostics: pending.diagnostics.clone(),
                target: pending.target.clone(),
                preview: self.preview.clone(),
            },
        ));
        Some(pending)
    }

    fn clear_pending(&mut self) {
        self.pending = None;
    }

    fn commit_immediate(&mut self, snapshot: u64) {
        self.committed_snapshot = snapshot;
    }

    fn take_activated(&mut self, synced: &mut u64) -> Option<DisplayActivation> {
        let (serial, active) = self.active.as_ref()?;
        if *serial <= *synced {
            return None;
        }
        *synced = *serial;
        Some(active.clone())
    }
}

pub struct GpuViewport {
    adapter: String,
    backend: String,
    device_type: String,
    slot: SceneSlot,
    /// Allocator for scene keys. A staged pair reserves its key here without
    /// becoming the key of frames that still show the previous scene.
    next_key: u64,
    /// Key carried by newly built frames: the scene last handed to the
    /// renderer. It adopts an activated pair's key on the next display sync.
    key: u64,
    /// Allocation order of `key`, so an already-drawn activation cannot roll
    /// back a newer immediate upload during that sync.
    key_serial: u64,
    ghost_edge_matrix: Option<[[f32; 4]; 4]>,
    ghost_edge_present: bool,
    ghost_pick: GhostPickData,
    /// Pick segments/ids for the displayed body's topology edges. The edit
    /// ghost's own set lives in `ghost_pick`.
    display_pick: GhostPickData,
    /// Generation of the pick scene (display or ghost edges). Requests carry it
    /// so a result from an older scene is dropped rather than mapped onto new
    /// ids.
    pick_key: u64,
    pick: std::sync::Arc<Mutex<crate::gpu_pick::PickRenderer>>,
    pick_request: std::sync::Arc<std::sync::Mutex<Option<crate::gpu_pick::PickRequest>>>,
    pick_next: u64,
    backdrop: Arc<Mutex<BackdropRenderer>>,
    backdrop_id: egui::TextureId,
    grid_extent: f64,
    grid_step: f32,
    /// Display radius of the current mesh, used for the world axis extent.
    display_radius: f64,
    /// Bounds of the uploaded display mesh. Preview frames reuse this instead of
    /// walking every vertex again.
    display_bounds: Aabb3,
    bounds: Aabb3,
    #[allow(dead_code)] // Kept for future resource creation after GPU init.
    device: wgpu::Device,
    object: Arc<Mutex<ObjectRenderer>>,
    /// GPU-evaluated plan mesh; falls back to the CPU object above.
    plan: Arc<Mutex<crate::gpu_plan::PlanState>>,
    /// Display payload staged with a non-resident plan set. Activation happens
    /// inside the render callback, where the plan swap happens.
    display_swap: Arc<Mutex<DisplaySwap>>,
    /// Last activation serial copied into the Viewport's CPU display mirror.
    display_synced: u64,
    /// Keeps the background plan-evaluation shader compile alive.
    #[cfg(not(target_arch = "wasm32"))]
    _tess_later: Option<solidcraft_viewport_gpu::view::TessLater>,
    scene: Arc<Mutex<Option<SceneTarget>>>,
    view: ViewSettings,
    chrome_visible: bool,
    background: [u8; 4],
    target_format: wgpu::TextureFormat,
    reversed_depth: bool,
    /// Mirrors Settings > Diagnostics so the callback can skip all profiler
    /// resources with a plain bool check when the switch is off.
    profiler_enabled: bool,
}

impl GpuViewport {
    /// Install the SolidCraft renderer into eframe's wgpu callback resources.
    pub fn install(rs: &RenderState) -> Option<Self> {
        let scene = GpuTarget {
            format: rs.target_format,
            depth: Some(depth_format(&rs.device, true)),
            samples: msaa_samples(),
            tess: false,
            oit: false,
            edges: false,
        };
        let window = GpuTarget {
            format: rs.target_format,
            depth: Some(depth_format(&rs.device, true)),
            samples: msaa_samples(),
            tess: false,
            oit: false,
            edges: false,
        };
        let renderer = Renderer::with_window_target(&rs.device, scene, window, None);
        rs.renderer.write().callback_resources.insert(renderer);
        let backdrop = BackdropRenderer::install(rs);
        let backdrop_id = backdrop.lock().ok()?.texture_id();
        rs.renderer.write().callback_resources.insert(backdrop.clone());
        let adapter_info = rs.adapter.get_info();
        let scene = Arc::new(Mutex::new(None));
        rs.renderer.write().callback_resources.insert(scene.clone());

        let module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-clear"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_SHADER.into()),
        });
        let bind_group_layout =
            rs.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport-clear-bind-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let theme = crate::theme::get();
        let background = [
            f32::from(theme.window[0]) / 255.0,
            f32::from(theme.window[1]) / 255.0,
            f32::from(theme.window[2]) / 255.0,
            f32::from(theme.window[3]) / 255.0,
        ];
        let clear_buffer = rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewport-clear-background"),
            contents: bytemuck::cast_slice(&background),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let clear_bind_group = rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-clear-bind"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: clear_buffer.as_entire_binding(),
            }],
        });
        let layout = rs.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-clear-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-clear"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: rs.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // Match eframe's depth attachment; the clear itself neither tests nor writes.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format(&rs.device, true),
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        rs.renderer.write().callback_resources.insert(ViewportClear {
            pipeline,
            bind_group: clear_bind_group,
            buffer: clear_buffer,
        });
        rs.renderer.write().callback_resources.insert(GridRenderer::install(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
        ));
        rs.renderer.write().callback_resources.insert(AxisRenderer::install(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
        ));
        let reversed_depth = reversed_depth(Some(depth_format(&rs.device, true)));
        let object = Arc::new(Mutex::new(ObjectRenderer::install(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
            reversed_depth,
        )));
        rs.renderer.write().callback_resources.insert(object.clone());
        #[cfg(not(target_arch = "wasm32"))]
        let tess_later = Some(solidcraft_viewport_gpu::view::TessLater::start(
            &rs.adapter,
            &rs.device,
            &rs.queue,
            solidcraft_viewport_gpu::SHADER,
            false,
        ));
        #[cfg(not(target_arch = "wasm32"))]
        let tess_ready = tess_later.as_ref().map(|later| later.ready());
        #[cfg(target_arch = "wasm32")]
        let tess_ready = None;
        let plan = Arc::new(Mutex::new(crate::gpu_plan::PlanState::new(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
            reversed_depth,
            tess_ready,
        )));
        Some(Self {
            adapter: adapter_info.name,
            backend: format!("{:?}", adapter_info.backend),
            device_type: format!("{:?}", adapter_info.device_type),
            slot: Arc::new(Mutex::new(None)),
            next_key: 1,
            key: 0,
            key_serial: 0,
            ghost_edge_matrix: None,
            ghost_edge_present: false,
            ghost_pick: GhostPickData::default(),
            display_pick: GhostPickData::default(),
            pick_key: 0,
            pick: std::sync::Arc::new(Mutex::new(crate::gpu_pick::PickRenderer::install(
                &rs.device,
                depth_format(&rs.device, true),
                reversed_depth,
            ))),
            pick_request: std::sync::Arc::new(std::sync::Mutex::new(None)),
            pick_next: 1,
            backdrop,
            backdrop_id,
            grid_extent: 0.0,
            grid_step: 1.0,
            display_radius: EMPTY_GRID_RADIUS,
            display_bounds: Aabb3 {
                min: Vec3::new(-EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS),
                max: Vec3::new(EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS),
            },
            bounds: Aabb3 {
                min: Vec3::new(-EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS),
                max: Vec3::new(EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS),
            },
            device: rs.device.clone(),
            object,
            plan,
            display_swap: Arc::new(Mutex::new(DisplaySwap::default())),
            display_synced: 0,
            #[cfg(not(target_arch = "wasm32"))]
            _tess_later: tess_later,
            scene,
            view: ViewSettings::default(),
            chrome_visible: true,
            background: crate::theme::get().viewport_background,
            target_format: rs.target_format,
            reversed_depth,
            profiler_enabled: false,
        })
    }

    pub fn set_gpu_profiler_enabled(&mut self, enabled: bool) {
        self.profiler_enabled = enabled;
    }

    pub fn plan_ready(&self) -> bool {
        self.plan.lock().ok().is_some_and(|plan| plan.tess_ready())
    }

    pub fn plan_debug(&self) -> Option<crate::gpu_plan::PlanDebug> {
        self.plan.lock().ok().map(|plan| plan.debug_counts())
    }

    /// Reserve a fresh scene key. Returns the key and its allocation order.
    /// Staged and immediate uploads take keys from this one counter, so a
    /// staged key can never be reused by a scene uploaded before activation.
    fn reserve_scene_key(&mut self) -> (u64, u64) {
        let key = self.next_key;
        self.next_key = self.next_key.wrapping_add(1);
        (key, self.next_key)
    }

    /// Reserve the key for a scene this frame uploads immediately.
    fn take_scene_key(&mut self) -> u64 {
        let (key, key_serial) = self.reserve_scene_key();
        self.key = key;
        self.key_serial = key_serial;
        key
    }

    pub fn set_plans(
        &mut self,
        plans: Vec<nodecraft_display::PlanInstance>,
        enabled: bool,
        colour: Option<[f32; 3]>,
        snapshot: u64,
    ) -> crate::gpu_plan::PlanSetState {
        let status = self
            .plan
            .lock()
            .map(|mut plan| plan.set_plans(plans, enabled, colour, snapshot))
            .unwrap_or(crate::gpu_plan::PlanSetState::Immediate);
        if status == crate::gpu_plan::PlanSetState::Immediate {
            // A set that activated immediately (or an edit-back that cancelled
            // the staged set) must not leave a display payload waiting for an
            // activation that will never happen.
            self.clear_pending_display();
        }
        status
    }

    pub(crate) fn stage_display(
        &mut self,
        display: Option<DisplayMesh>,
        source_mesh: Option<nodecraft_protocol::Mesh>,
        diagnostics: Option<Vec<String>>,
        target: Option<String>,
        snapshot: u64,
        retire_preview: bool,
    ) {
        let wait_activation =
            self.plan.lock().map(|plan| plan.activation().wrapping_add(1)).unwrap_or(0);
        let prepared = display.as_ref().map(|display| prepare_object(display, &self.view));
        // The payload carries the preview it will show with. Retiring it here,
        // rather than immediately after the app stages the display, keeps the
        // current geometry previewed until this pair activates. A live drag
        // keeps the unit's current preview, so the new geometry and its edges
        // still agree on activation.
        let preview = if retire_preview {
            None
        } else {
            self.display_swap.lock().ok().and_then(|swap| swap.preview.clone())
        };
        let mut scene = Self::build_scene(display.as_ref(), &self.view, true);
        let edge_batch = display
            .as_ref()
            .map(|display| EdgeCpuBatch::from_display_edges(&display.edges))
            .unwrap_or_default();
        // The custom object renderer owns the main solid, as in set_prepared_mesh.
        scene.pbr.clear();
        // Reserve the key with the payload, but do not make it the key of frames
        // built before activation: those still show the previous scene. The
        // activation frame carries it explicitly.
        let (key, key_serial) = self.reserve_scene_key();
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            let (edges, radius) =
                display.as_ref().map_or((0, 0.0), |display| (display.edges.len(), display.radius));
            eprintln!(
                "[display] stage key={key} wait_activation={wait_activation} snapshot={snapshot} edges={edges} radius={radius:.3} target={target:?}",
            );
        }
        if let Ok(mut swap) = self.display_swap.lock() {
            swap.stage(PendingDisplay {
                wait_activation,
                edges: StagedEdges {
                    snapshot,
                    key,
                    key_serial,
                    display: display.map(Arc::new),
                    scene,
                    edge_batch,
                },
                source_mesh,
                prepared,
                diagnostics,
                target,
                preview,
            });
        }
    }

    pub(crate) fn stage_diagnostics(&mut self, diagnostics: Vec<String>) {
        if let Ok(mut swap) = self.display_swap.lock()
            && let Some(pending) = swap.pending.as_mut()
        {
            pending.diagnostics = Some(diagnostics);
        }
    }

    fn clear_pending_display(&mut self) {
        if let Ok(mut swap) = self.display_swap.lock() {
            swap.clear_pending();
        }
    }

    pub(crate) fn take_activated_display(&mut self) -> Option<DisplayActivation> {
        let active = self.display_swap.lock().ok()?.take_activated(&mut self.display_synced)?;
        // The staging frame's activation handed the renderer `active.key`.
        // Later frames must use that same key (unless a newer upload already
        // advanced it), or the renderer would reject the next scene as stale.
        adopt_scene_key(&mut self.key, &mut self.key_serial, &active);
        if let Some(display) = active.display.as_ref() {
            self.display_pick = build_ghost_pick_data(&display.edges);
            self.pick_key = self.pick_key.wrapping_add(1);
        }
        Some(active)
    }

    pub(crate) fn commit_immediate_snapshot(&mut self, snapshot: u64) {
        if let Ok(mut swap) = self.display_swap.lock() {
            swap.commit_immediate(snapshot);
        }
    }

    pub(crate) fn committed_snapshot(&self) -> u64 {
        self.display_swap.lock().map(|swap| swap.committed_snapshot).unwrap_or(0)
    }

    pub(crate) fn pending_display_bounds(&self) -> Option<(f64, solidcore_record::Vec3)> {
        let swap = self.display_swap.lock().ok()?;
        let display = swap.pending.as_ref()?.edges.display.as_ref()?;
        Some((display.radius, display.center))
    }

    pub fn set_ghost_plans(
        &mut self,
        plans: Vec<nodecraft_display::PlanInstance>,
        alpha: f32,
        colour: Option<[f32; 3]>,
        enabled: bool,
    ) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_ghost_plans(plans, alpha, colour, enabled);
        }
    }

    pub fn set_ghost_matrix(&mut self, matrix: Option<[[f32; 4]; 4]>) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_ghost_matrix(matrix);
        }
    }

    pub fn bounds(&self) -> &Aabb3 {
        &self.bounds
    }

    pub fn set_chrome_visible(&mut self, visible: bool) {
        self.chrome_visible = visible;
    }

    pub fn set_background(&mut self, rgba: [u8; 4]) {
        self.background = rgba;
    }

    pub fn set_view(&mut self, view: ViewSettings, display: Option<&DisplayMesh>) {
        if self.view == view {
            return;
        }
        let rebuild = self.view.face_colours != view.face_colours
            || self.view.edges != view.edges
            || self.view.triangles != view.triangles
            || self.view.normals != view.normals;
        self.view = view;
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_triangles(view.triangles);
            plan.set_translucent(view.translucent);
            plan.set_edge_width(view.edge_width_px);
            plan.set_edges_enabled(view.edges && !view.zebra);
            plan.set_show_seam_edges(view.show_seam_edges);
            // Face-colours is a table fill, not a different renderer: the plan
            // chunks keep their meshes and instance buffers and only the
            // colour table is rewritten.
            plan.set_face_colours(view.face_colours);
        }
        if rebuild {
            self.set_mesh(display, true);
        }
    }

    pub fn set_mesh(&mut self, display: Option<&DisplayMesh>, retire_preview: bool) {
        let prepared = display.map(|display| prepare_object(display, &self.view));
        self.set_prepared_mesh(display, prepared, retire_preview);
    }

    /// Apply display geometry whose CPU mesh conversion already ran on a worker.
    /// `retire_preview` travels with the mesh: authoritative geometry retires the
    /// drag preview in the same step, while a live drag keeps it.
    pub(crate) fn set_prepared_mesh(
        &mut self,
        display: Option<&DisplayMesh>,
        prepared: Option<PreparedObject>,
        retire_preview: bool,
    ) {
        if retire_preview {
            // A full display mesh is authoritative geometry: it already sits at
            // the edited placement, so any drag preview is retired here, with
            // the mesh.
            self.set_preview_unit(None);
        }
        self.take_scene_key();
        self.display_pick =
            display.map(|display| build_ghost_pick_data(&display.edges)).unwrap_or_default();
        self.pick_key = self.pick_key.wrapping_add(1);
        let radius = display.map_or(EMPTY_GRID_RADIUS, |display| display.radius);
        self.display_radius = radius;
        self.grid_extent = crate::viewport_grid::grid_extent(radius);
        self.grid_step = grid_step(radius) as f32;
        self.display_bounds = display.map_or_else(
            || Aabb3 {
                min: Vec3::new(-radius, -radius, -radius),
                max: Vec3::new(radius, radius, radius),
            },
            |display| display.mesh.bounds(),
        );
        let preview = if retire_preview { None } else { self.preview_unit() };
        self.bounds = match &preview {
            Some(preview) => transform_bounds(self.display_bounds, preview.transform),
            None => self.display_bounds,
        };
        let (object_vertices, object_face_ids, object_indices) = prepared
            .map(|prepared| (prepared.vertices, prepared.face_ids, prepared.indices))
            .unwrap_or_default();
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_mesh(object_vertices, object_face_ids, object_indices);
            // A live preview keeps moving the new geometry. A retired one is
            // already at identity through `set_mesh`.
            if let Some(preview) = &preview {
                renderer.set_transform(Some(preview.transform), preview.alpha);
            }
        }
        if let Ok(mut plan) = self.plan.lock() {
            let edges = display
                .map(|display| EdgeCpuBatch::from_display_edges(&display.edges))
                .unwrap_or_default();
            plan.set_world_edges(edges);
        }
        let mut scene = Self::build_scene(display, &self.view, true);
        if let Ok(mut backdrop) = self.backdrop.lock() {
            backdrop.set_scene(scene.clone());
        }
        // The custom object renderer owns the main solid; SolidCraft only draws the
        // edges/grid over it.
        scene.pbr.clear();
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(scene);
        }
    }

    /// Replace only the displayed object geometry with a coarser LOD. Edges and
    /// the backdrop scene stay on the full display mesh, as does picking.
    pub fn set_lod_mesh(&mut self, mesh: &CraftMesh) {
        let colour = [0.17, 0.18, 0.195];
        let faces = ObjectFaceIds::from_mesh(mesh);
        let geometry = smoothed_object_mesh(mesh, colour, &faces)
            .or_else(|| flat_object_mesh(mesh, colour, &faces))
            .unwrap_or_default();
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_mesh(geometry.vertices, geometry.face_ids, geometry.indices);
        }
    }

    pub fn clear_plan_transient(&mut self) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.clear_transient();
        }
    }

    pub fn clear_plan_document(&mut self) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.clear_document();
        }
    }

    /// Generation of the active pick scene (display or edit-ghost edges).
    pub fn pick_key(&self) -> u64 {
        self.pick_key
    }

    pub fn ghost_edge_matrix(&self) -> Option<[[f32; 4]; 4]> {
        self.ghost_edge_matrix
    }

    pub fn pick_pending(&self) -> bool {
        self.pick_request.lock().is_ok_and(|request| request.is_some())
            || self.pick.lock().is_ok_and(|pick| pick.in_flight())
    }

    pub fn request_ghost_pick(
        &mut self,
        px: [f32; 2],
        radius_px: f32,
        commit: bool,
        kind: crate::gpu_pick::PickKind,
    ) -> u64 {
        let mut slot = self.pick_request.lock().expect("pick request lock");
        if !commit
            && let Some(existing) = slot.as_ref()
            && existing.commit
        {
            return existing.id;
        }
        let id = self.pick_next;
        self.pick_next = self.pick_next.wrapping_add(1);
        *slot = Some(crate::gpu_pick::PickRequest {
            id,
            px,
            radius_px,
            ghost_key: self.pick_key,
            commit,
            kind,
        });
        id
    }

    pub fn cancel_pick(&mut self, id: u64) {
        if let Ok(mut slot) = self.pick_request.lock()
            && slot.as_ref().is_some_and(|request| request.id == id)
        {
            *slot = None;
        }
    }

    pub fn take_pick_result(&mut self) -> Option<crate::gpu_pick::PickResult> {
        self.pick.lock().ok()?.take_result()
    }

    /// Hand the current selection/hover to the ribbon pass. This is id lookup
    /// only: no buffer is rebuilt when the pointer moves.
    pub fn set_edge_selection(
        &mut self,
        selected: &std::collections::BTreeSet<String>,
        hovered: Option<&str>,
    ) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_edge_selection(
                selected.iter().cloned().collect(),
                hovered.map(ToOwned::to_owned),
            );
        }
    }

    pub fn set_ghost_edges(&mut self, edges: &[crate::adapter::DisplayEdge]) {
        let batch = EdgeCpuBatch::from_display_edges(edges);
        self.ghost_edge_present = !batch.is_empty();
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_world_ghost_edges(batch);
        }
        self.ghost_pick = build_ghost_pick_data(edges);
        self.pick_key = self.pick_key.wrapping_add(1);
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[ghost-edges] rebuild {} edges", edges.len());
        }
    }

    pub(crate) fn ghost_pick_data(&self) -> &GhostPickData {
        &self.ghost_pick
    }

    /// The edge ids and segments the pick pass is currently drawing: the edit
    /// ghost while it is shown, otherwise the displayed body's edges. Plan
    /// edges are local-space and pick through the same buffers as the visible
    /// ribbons, so their id table comes from the plan state.
    pub(crate) fn pick_data(&self) -> GhostPickData {
        if self.ghost_edge_present {
            return self.ghost_pick.clone();
        }
        if let Ok(plan) = self.plan.lock() {
            let data = plan.edge_pick_data();
            if data.edges.len() > 1 {
                return data.clone();
            }
        }
        self.display_pick.clone()
    }

    pub fn set_ghost_edge_matrix(&mut self, matrix: Option<[[f32; 4]; 4]>) {
        self.ghost_edge_matrix = matrix;
    }

    pub fn clear_ghost_edges(&mut self) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_world_ghost_edges(EdgeCpuBatch::default());
        }
        self.ghost_edge_matrix = None;
        self.ghost_edge_present = false;
        self.ghost_pick = GhostPickData::default();
        self.pick_key = self.pick_key.wrapping_add(1);
    }

    pub fn clear_ghost(&mut self) {
        self.clear_ghost_edges();
        self.set_ghost_plans(Vec::new(), 0.45, None, false);
        self.set_ghost_matrix(None);
    }

    /// Preview a transform edit without touching the merged mesh or the scene
    /// geometry.
    ///
    /// The preview is applied by transforming the scene projection in `frame()`,
    /// so a drag frame only updates the object instance matrix and a uniform —
    /// no CPU geometry work and no GPU re-upload.
    pub fn set_preview_transform(&mut self, transform: Option<[[f32; 4]; 4]>, alpha: f32) {
        let next = transform.map(|transform| PreviewState { transform, alpha });
        // The ghost alpha uniform must be current even when the preview state
        // itself is unchanged (e.g. the static overlay was already installed).
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_ghost_alpha(alpha);
        }
        let previous = self.preview_unit();
        if previous == next {
            return;
        }
        self.set_preview_unit(next.clone());
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_transform(
                next.as_ref().map(|preview| preview.transform),
                next.as_ref().map_or(1.0, |preview| preview.alpha),
            );
        }
        let base_bounds = self.display_bounds;
        self.bounds = match &next {
            Some(preview) => transform_bounds(base_bounds, preview.transform),
            None => base_bounds,
        };
    }

    /// The display unit's current preview. This is the single source every
    /// per-frame consumer reads.
    fn preview_unit(&self) -> Option<PreviewState> {
        self.display_swap.lock().ok().and_then(|swap| swap.preview.clone())
    }

    fn set_preview_unit(&mut self, preview: Option<PreviewState>) {
        if let Ok(mut swap) = self.display_swap.lock() {
            swap.preview = preview;
        }
    }

    /// Compare the app mirror with the display unit. An activation may have
    /// retired the preview before the app synced its CPU copy; forwarding an
    /// apparently unchanged preview must still reach the unit.
    pub(crate) fn preview_matches(&self, transform: Option<[[f32; 4]; 4]>, alpha: f32) -> bool {
        match (transform, self.preview_unit()) {
            (None, None) => true,
            (Some(transform), Some(preview)) => {
                preview.transform == transform && preview.alpha == alpha
            }
            _ => false,
        }
    }

    /// Edges and analysis overlays for the current display mesh. The preview's
    /// matrix moves geometry; the world-space Z axis marker stays put because it
    /// belongs to the world frame, not to the edited body.
    fn build_scene(
        display: Option<&DisplayMesh>,
        view: &ViewSettings,
        include_pbr: bool,
    ) -> GpuScene {
        let theme = crate::theme::get();
        let palette = Palette::with_stable(vec![
            [151, 162, 176, 255],
            [0, 0, 0, 255],
            [theme.axis_z[0], theme.axis_z[1], theme.axis_z[2], 230],
            [148, 148, 148, 140],
            [76, 201, 245, 230],
        ]);
        let mut scene = GpuScene::with_palette(palette);
        let triangle_edge = PaletteIndex(3);
        let face_normal = PaletteIndex(4);
        if let Some(display) = display {
            let mesh: &CraftMesh = &display.mesh;
            if include_pbr {
                let surface = OpaqueSurface::new(Vec3::new(0.32, 0.34, 0.37), 0.0, 0.44, 1.62)
                    .unwrap_or_else(|_| {
                        OpaqueSurface::new(Vec3::new(0.6, 0.6, 0.6), 0.0, 0.44, 1.5)
                            .expect("fallback surface")
                    });
                let face_ids = view.face_colours.then(|| ObjectFaceIds::from_display(display));
                for (triangle_index, triangle) in mesh
                    .triangles
                    .iter()
                    .enumerate()
                    .step_by(backdrop_pbr_stride(mesh.triangles.len()))
                {
                    let [a, b, c] = triangle.map(|index| index as usize);
                    let (Some(pa), Some(pb), Some(pc)) =
                        (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
                    else {
                        continue;
                    };
                    let colour = if let Some(face_ids) = face_ids.as_ref() {
                        let (body, face) = face_ids.body_face(mesh, triangle_index);
                        Some(face_colour(display_body_id(display, body), face))
                    } else {
                        display.face_colors.get(triangle_index).copied()
                    };
                    let surface = colour
                        .and_then(|color| {
                            OpaqueSurface::new(
                                Vec3::new(
                                    f64::from(color[0]),
                                    f64::from(color[1]),
                                    f64::from(color[2]),
                                ),
                                0.0,
                                0.44,
                                1.62,
                            )
                            .ok()
                        })
                        .unwrap_or(surface);
                    let fallback = (*pb - *pa).cross(*pc - *pa);
                    let fallback = fallback.normalized().unwrap_or(solidcore_record::Vec3::Z);
                    scene.pbr_tri(pa.to_f32(), fallback.to_f32(), surface);
                    scene.pbr_tri(pb.to_f32(), fallback.to_f32(), surface);
                    scene.pbr_tri(pc.to_f32(), fallback.to_f32(), surface);
                }
            }
            if view.triangles {
                let mut seen = std::collections::HashSet::new();
                for triangle in mesh.triangles.iter() {
                    for pair in [
                        [triangle[0], triangle[1]],
                        [triangle[1], triangle[2]],
                        [triangle[2], triangle[0]],
                    ] {
                        let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                        if !seen.insert(key) {
                            continue;
                        }
                        let (Some(a), Some(b)) = (
                            mesh.positions.get(pair[0] as usize),
                            mesh.positions.get(pair[1] as usize),
                        ) else {
                            continue;
                        };
                        scene.line(a.to_f32(), b.to_f32(), triangle_edge, 0.7, false);
                    }
                }
            }
            if view.normals {
                // Aggregate triangles by display face, then draw one area-weighted
                // face normal from each face centroid.
                let mut faces: std::collections::HashMap<u32, (Vec3, Vec3, f64)> =
                    std::collections::HashMap::new();
                for (index, triangle) in mesh.triangles.iter().enumerate() {
                    let [a, b, c] = triangle.map(|index| index as usize);
                    let (Some(pa), Some(pb), Some(pc)) =
                        (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
                    else {
                        continue;
                    };
                    let cross = (*pb - *pa).cross(*pc - *pa);
                    let area = cross.len() * 0.5;
                    if area <= 1.0e-12 {
                        continue;
                    }
                    let normal = cross.normalized().unwrap_or(Vec3::Z);
                    let centroid = (*pa + *pb + *pc) / 3.0;
                    let face = mesh.tri_face.get(index).copied().unwrap_or(index as u32);
                    let entry = faces.entry(face).or_insert((Vec3::ZERO, Vec3::ZERO, 0.0));
                    entry.0 = entry.0 + normal * area;
                    entry.1 = entry.1 + centroid * area;
                    entry.2 += area;
                }
                let length = display.radius * 0.075;
                for (_face, (normal, centroid, area)) in faces {
                    if area <= 1.0e-12 {
                        continue;
                    }
                    let Some(normal) = normal.normalized() else { continue };
                    let center = centroid / area;
                    let tip = centroid / area + normal * length;
                    scene.line(center.to_f32(), tip.to_f32(), face_normal, 1.5, false);
                }
            }
        }
        scene
    }

    pub fn debug_info(&self) -> Option<GpuDebugInfo> {
        let object = self.object.lock().ok()?;
        Some(object.debug_info(&self.adapter, &self.backend, &self.device_type))
    }

    pub fn backdrop_texture_id(&self) -> egui::TextureId {
        self.backdrop_id
    }

    pub fn frame(
        &self,
        camera: &Camera,
        lens_shift: [f32; 2],
        width: f32,
        height: f32,
        pixels_per_point: f32,
        bounds: &solidcraft_geom::Aabb3,
        backdrop_crop: [f32; 4],
        backdrop_blur_scale: [f32; 4],
    ) -> GpuCallback {
        let bounds = solidcraft_viewport::vec::Bounds { min: bounds.min, max: bounds.max };
        let aspect = f64::from(width) / f64::from(height.max(1.0));
        let half_height = camera.half_height();
        let (grid_minor, grid_major, grid_fine_weight) =
            crate::viewport_grid::adaptive_grid_crossfade(half_height);
        let back_vec = camera.back();
        let slant = back_vec.z;
        let grid_extent =
            crate::viewport_grid::adaptive_grid_extent(half_height, grid_major, slant);
        let grid_center = crate::viewport_grid::adaptive_grid_center(
            [camera.target.x, camera.target.y, camera.target.z],
            [back_vec.x, back_vec.y, back_vec.z],
            grid_extent,
            grid_major,
        );
        let clip_far = camera.clip_bounds(&bounds).1;
        let far_reach = crate::viewport_grid::adaptive_grid_far_reach(
            [camera.eye().x, camera.eye().y, camera.eye().z],
            [back_vec.x, back_vec.y, back_vec.z],
            clip_far,
            grid_center,
        );
        let (fade_start, fade_end) =
            crate::viewport_grid::adaptive_grid_fade(half_height, slant, grid_extent, far_reach);
        // Keep the model's normal near/far planes. Expanding them for the grid
        // destroys depth precision and makes large perspective models flicker.
        let view_proj =
            shift_view_proj(camera.view_proj_bounds(aspect, &bounds).to_f32(), lens_shift);
        // The display unit's preview is the one source for this frame. It
        // transforms the scene projection instead of rebuilding edge geometry.
        // `prepare` may replace this value in the same step as a staged
        // activation.
        let unit_preview = self.display_swap.lock().ok().and_then(|swap| swap.preview.clone());
        let drawn =
            DrawnPreview::new(view_proj, crop_view_proj(view_proj, backdrop_crop), unit_preview);
        let scene_view_proj = drawn.scene_view_proj();
        let eye = camera.eye().to_f32();
        let back = camera.back().to_f32();
        let (right, up, _) = camera.basis();
        let (grid, axis) = if draws_grid(&self.view) {
            let theme = crate::theme::get();
            let grid = GridUniforms {
                eye: [eye[0], eye[1], eye[2], 1.0],
                target: {
                    let t = camera.target.to_f32();
                    [t[0], t[1], t[2], 1.0]
                },
                center: [grid_center[0] as f32, grid_center[1] as f32, grid_center[2] as f32, 1.0],
                origin: [0.0, 0.0, 0.0, 1.0],
                right: {
                    let r = right.to_f32();
                    [r[0], r[1], r[2], if camera.fov > 0.0 { 1.0 } else { 0.0 }]
                },
                up: {
                    let u = up.to_f32();
                    [u[0], u[1], u[2], 0.0]
                },
                back: [back[0], back[1], back[2], 0.0],
                view_proj,
                inv_view_proj: invert4x4(view_proj),
                params: [
                    camera.half_height() as f32,
                    camera.distance as f32,
                    aspect as f32,
                    if camera.fov > 0.0 { 1.0 } else { 0.0 },
                ],
                grid: [grid_minor as f32, 0.07, 0.22, grid_extent as f32],
                minor: colour(theme.grid_minor, 1.0),
                major: colour(theme.grid_major, 1.0),
                axis_x: {
                    let mut colour = colour(theme.axis_x, 1.0);
                    colour[3] = crate::viewport_grid::GRID_MAJOR_STEP as f32;
                    colour
                },
                axis_y: {
                    let mut colour = colour(theme.axis_y, 1.0);
                    colour[3] = camera.clip_bounds(&bounds).1 as f32;
                    colour
                },
                options: [
                    (grid_major / grid_minor) as f32,
                    fade_start as f32,
                    fade_end as f32,
                    grid_fine_weight,
                ],
            };
            let axis = AxisUniforms {
                view_proj,
                start: [0.0, 0.0, 0.0, 1.0],
                end: [0.0, 0.0, self.grid_extent as f32, 1.0],
                colour: colour(theme.axis_z, 230.0 / 255.0),
            };
            (Some(grid), Some(axis))
        } else {
            (None, None)
        };
        let object = ObjectUniforms {
            view_proj: gpu_view_proj(view_proj, self.reversed_depth),
            back: [back[0], back[1], back[2], 0.0],
            eye: [eye[0], eye[1], eye[2], if camera.fov > 0.0 { 1.0 } else { 0.0 }],
            params: [
                if self.view.translucent { 0.34 } else { 1.0 },
                if self.view.zebra { 1.0 } else { 0.0 },
                self.view.roughness,
                self.view.metallic,
            ],
            extra_model: identity_matrix(),
        };
        let plan_available = self.plan.lock().map(|plan| plan.is_available()).unwrap_or(false);
        let (pick_segments, ghost_edges) = if self.ghost_edge_present {
            (self.ghost_pick.segments.clone(), true)
        } else if plan_available {
            // Plan edges are picked from their local-space buffers with the
            // plan instances, not from world-space CPU segments.
            (std::sync::Arc::<[crate::gpu_pick::PickSegment]>::from(Vec::new()), false)
        } else {
            (self.display_pick.segments.clone(), false)
        };
        let pick_frame = Some(crate::gpu_pick::PickFrame {
            request: None,
            ghost_key: self.pick_key,
            segments: pick_segments,
            ghost_edges,
            face_matrix: self
                .preview_unit()
                .map(|preview| preview.transform)
                .unwrap_or_else(identity_matrix),
            ghost_matrix: self.ghost_edge_matrix.unwrap_or_else(identity_matrix),
            size_px: [width * pixels_per_point, height * pixels_per_point],
            edge_width_px: self.view.edge_width_px,
            show_seam_edges: self.view.show_seam_edges,
        });
        GpuCallback {
            object,
            background: self.background,
            view: self.view,
            chrome_visible: self.chrome_visible,
            frame: Mutex::new(Frame {
                product: false,
                preview_bounds: None,
                // Ghost edges are part of the same ribbon pass: the ghost matrix
                // is a uniform, so there is no separate moving scene slot.
                moving: None,
                product_lights: None,
                upload_error: None,
                clouds: Vec::new(),
                key: self.key,
                turned_depth: None,
                slot: self.slot.clone(),
                sketch_key: 0,
                sketch_slot: Arc::new(Mutex::new(None)),
                hl_key: 0,
                hl_slot: Arc::new(Mutex::new(None)),
                pv_key: 0,
                pv_slot: Arc::new(Mutex::new(None)),
                pv_hide: false,
                view_proj: scene_view_proj,
                back,
                size_px: [width * pixels_per_point, height * pixels_per_point],
                clip: clip_planes(&[]),
                clip_hatch: clip_hatches(&[]),
                cap: [0.0; 4],
                images: Vec::new(),
                analysis: [0.0; 8],
                analysis_colors: [0.0; 36],
                access: None,
                outline: false,
                eye: [eye[0], eye[1], eye[2], if camera.fov > 0.0 { 1.0 } else { 0.0 }],
                ao_radius: 0.0,
                ao_samples: solidcraft_viewport_gpu::view::AO_SAMPLES,
                ssaa: 1.0,
                hover_tint: [0.0; 3],
                selection_tint: [0.0; 3],
                hatch: Default::default(),
                sky: [0.0, 1.0, 0.0, 0.0],
            }),
            preview: Mutex::new(drawn),
            backdrop_blur_scale,
            backdrop_crop,
            target_format: self.target_format,
            scene: self.scene.clone(),
            plan: self.plan.clone(),
            display_swap: self.display_swap.clone(),
            grid,
            axis,
            pick: self.pick.clone(),
            pick_frame,
            pick_request: self.pick_request.clone(),
            profiler_enabled: self.profiler_enabled,
        }
    }
}

fn crop_view_proj(vp: [[f32; 4]; 4], crop: [f32; 4]) -> [[f32; 4]; 4] {
    // SolidCraft matrices are column-major. Remap the panel's NDC sub-rectangle
    // to [-1, 1] for the blurred texture.
    let nx0 = 2.0 * crop[0] - 1.0;
    let nx1 = 2.0 * (crop[0] + crop[2]) - 1.0;
    let ny0 = 1.0 - 2.0 * (crop[1] + crop[3]);
    let ny1 = 1.0 - 2.0 * crop[1];
    let sx = 2.0 / (nx1 - nx0).max(1.0e-6);
    let sy = 2.0 / (ny1 - ny0).max(1.0e-6);
    let tx = -sx * (nx0 + nx1) * 0.5;
    let ty = -sy * (ny0 + ny1) * 0.5;
    let m = [[sx, 0.0, 0.0, 0.0], [0.0, sy, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [tx, ty, 0.0, 1.0]];
    multiply_matrix(m, vp)
}

/// One viewport draw as an egui paint callback.
pub struct GpuCallback {
    pub object: ObjectUniforms,
    pub background: [u8; 4],
    pub view: ViewSettings,
    #[allow(dead_code)] // Reserved for GPU chrome rendering.
    pub chrome_visible: bool,
    /// The frame handed to the renderer. The target scene of a staged pair is
    /// activated inside `prepare`, so its key is set here before that upload.
    pub frame: Mutex<Frame>,
    /// The preview source for this draw. A staged activation replaces its state
    /// in `prepare`; the plan uniform, scene projection and backdrop all derive
    /// from the resulting value.
    preview: Mutex<DrawnPreview>,
    pub backdrop_blur_scale: [f32; 4],
    pub backdrop_crop: [f32; 4],
    pub target_format: wgpu::TextureFormat,
    pub scene: Arc<Mutex<Option<SceneTarget>>>,
    pub plan: Arc<Mutex<crate::gpu_plan::PlanState>>,
    /// Display payload staged with the plan set. It activates in the same
    /// `prepare` call that swaps the plan chunks.
    pub(crate) display_swap: Arc<Mutex<DisplaySwap>>,
    pub pick: Arc<Mutex<crate::gpu_pick::PickRenderer>>,
    pub pick_frame: Option<crate::gpu_pick::PickFrame>,
    pub pick_request: Arc<Mutex<Option<crate::gpu_pick::PickRequest>>>,
    pub grid: Option<GridUniforms>,
    pub axis: Option<AxisUniforms>,
    /// Copied from the viewport each frame; checked before touching any
    /// profiler resource.
    pub profiler_enabled: bool,
}

impl GpuCallback {
    fn prepare_impl(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Off: one bool check, no resources lookup and no lock. On: one lock
        // for the whole callback, then pass scopes on the shared slot.
        let profiler_slot = self.profiler_enabled.then(|| {
            resources
                .get::<std::sync::Arc<std::sync::Mutex<crate::gpu_profiler::GpuProfilerSlot>>>()
                .cloned()
        });
        let profiler_slot = profiler_slot.flatten();
        let mut profiler_guard = profiler_slot.as_ref().and_then(|slot| slot.lock().ok());
        let mut profiler = profiler_guard.as_deref_mut();
        if let Some(clear) = resources.get::<ViewportClear>() {
            clear.set_background(queue, self.background);
        }
        if let Some(grid) = resources.get::<GridRenderer>()
            && let Some(uniforms) = self.grid.as_ref()
        {
            grid.prepare(queue, uniforms);
        }
        if let Some(axis) = resources.get::<AxisRenderer>()
            && let Some(uniforms) = self.axis.as_ref()
        {
            axis.prepare(queue, uniforms);
        }
        // A staged pair activates inside this call and hands the renderer both
        // the new scene and its own key. The frame captured when the callback
        // was built may still carry the key of the previous scene.
        let Ok(mut frame) = self.frame.lock() else {
            return Vec::new();
        };
        // Advance the plan swap first: it decides whether the staged display
        // payload activates in this same `prepare` call.
        let (plan_activation, plan_snapshot, plan_active) = self
            .plan
            .lock()
            .map(|mut plan| {
                plan.prepare_chunks(device, queue);
                (plan.activation(), plan.display_snapshot(), plan.is_available())
            })
            .unwrap_or((0, 0, false));
        // The display unit decides the preview for this frame. An activation
        // installs its payload's preview in the same step as the scene and key.
        let (pending, unit_preview) = {
            let Ok(mut swap) = self.display_swap.lock() else {
                return Vec::new();
            };
            let pending = swap.take_ready(plan_activation, plan_snapshot);
            (pending, swap.preview.clone())
        };
        if let Some(pending) = pending {
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                let edges = pending.edges.display.as_ref().map_or(0, |display| display.edges.len());
                eprintln!(
                    "[display] activate key={} plan_activation={plan_activation} snapshot={} edges={edges} target={:?}",
                    pending.edges.key, pending.edges.snapshot, pending.target,
                );
            }
            // The scene and its key must reach the renderer in one call: a new
            // scene under the captured key is skipped as already uploaded.
            apply_activated_display(&pending, &mut frame);
            if let Ok(mut plan) = self.plan.lock() {
                plan.set_world_edges(pending.edges.edge_batch.clone());
            }
            // Object vertices and the edge scene become drawable in this same
            // callback, before the paint pass runs.
            if let Some(object) = resources.get::<Arc<Mutex<ObjectRenderer>>>()
                && let Ok(mut object) = object.lock()
            {
                let (vertices, face_ids, indices) = pending
                    .prepared
                    .map(|prepared| (prepared.vertices, prepared.face_ids, prepared.indices))
                    .unwrap_or_default();
                object.set_mesh(vertices, face_ids, indices);
                // A live drag keeps its preview over the activated mesh. Apply
                // the same value that `drawn` will use for the plan uniform and
                // projections in this prepare.
                if let Some(preview) = &unit_preview {
                    object.set_transform(Some(preview.transform), preview.alpha);
                }
            }
            if let Some(backdrop) = resources.get::<Arc<Mutex<BackdropRenderer>>>()
                && let Ok(mut backdrop) = backdrop.lock()
            {
                backdrop.set_scene(pending.edges.scene);
            }
        }
        let Ok(mut drawn) = self.preview.lock() else {
            return Vec::new();
        };
        sync_drawn_preview(&mut drawn, &mut frame, unit_preview);
        // Both consumers take the same value for this frame: the plan shader
        // matrix and the scene projection come from `drawn`.
        if let Ok(plan) = self.plan.lock() {
            plan.write_uniforms(queue, &self.object, drawn.plan_transform(), frame.size_px);
        }
        if let Some(object) = resources.get::<Arc<Mutex<ObjectRenderer>>>() {
            if let Ok(mut object) = object.lock() {
                object.prepare(device, queue, &self.object);
            }
        }
        {
            let Some(renderer) = resources.get_mut::<Renderer>() else {
                return Vec::new();
            };
            renderer.prepare(device, queue, encoder, &frame);
        }
        let samples = msaa_samples();
        let size =
            [frame.size_px[0].round().max(1.0) as u32, frame.size_px[1].round().max(1.0) as u32];
        let clear = resources.get::<ViewportClear>();
        let object = resources.get::<Arc<Mutex<ObjectRenderer>>>();
        let renderer = resources.get::<Renderer>();
        let grid = resources.get::<GridRenderer>();
        let axis = resources.get::<AxisRenderer>();
        let mut external_view = None;
        if let (Some(clear), Some(object), Some(renderer), Some(grid), Some(axis)) =
            (clear, object, renderer, grid, axis)
        {
            if let Ok(mut scene) = self.scene.lock() {
                if scene.as_ref().is_none_or(|target| target.size != size) {
                    *scene = Some(SceneTarget::new(
                        device,
                        size,
                        self.target_format,
                        depth_format(device, true),
                        samples,
                    ));
                }
                if let Some(target) = scene.as_ref() {
                    external_view = Some(target.resolve_view.clone());
                    let (view, resolve_target) =
                        scene_color_attachment(samples, &target.color_view, &target.resolve_view);
                    crate::gpu_profiler::render_pass(
                        profiler.as_deref_mut(),
                        crate::gpu_profiler::pass::SCENE,
                        encoder,
                        wgpu::RenderPassDescriptor {
                            label: Some(crate::gpu_profiler::pass::SCENE),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view,
                                depth_slice: None,
                                resolve_target,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &target.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Discard,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                        |pass| {
                            pass.set_viewport(0.0, 0.0, size[0] as f32, size[1] as f32, 0.0, 1.0);
                            pass.set_pipeline(&clear.pipeline);
                            pass.set_bind_group(0, &clear.bind_group, &[]);
                            pass.draw(0..3, 0..1);
                            if plan_active && let Ok(plan) = self.plan.lock() {
                                plan.paint(pass);
                            }
                            if let Ok(object) = object.lock() {
                                object.paint(
                                    pass,
                                    self.view.translucent && !plan_active,
                                    !plan_active,
                                );
                            }
                            if let Ok(mut plan) = self.plan.lock() {
                                plan.paint_edges(pass, plan_active);
                            }
                            renderer.paint(
                                &frame,
                                pass,
                                [0.0, 0.0, size[0] as f32, size[1] as f32],
                            );
                            // The axis lines belong to the XY ground grid: the
                            // grid toggle hides the grid shader's X/Y axes and
                            // the world Z axis together.
                            if self.grid.is_some() {
                                grid.paint(pass);
                            }
                            if self.axis.is_some() {
                                axis.paint(pass);
                            }
                        },
                    );
                }
            }
        }
        if let Ok(mut pick) = self.pick.lock() {
            pick.service(device);
        }
        if let Some(pick_frame) = &self.pick_frame {
            let mut pick_frame = pick_frame.clone();
            // Plan chunks carry the preview in the uniform. The object mesh
            // already carries the same drawn preview in its instance buffer
            // (set from the same DrawnPreview value), so it must use identity
            // here or the pick pass would apply the preview twice.
            pick_frame.face_matrix = if plan_active {
                drawn.plan_transform().unwrap_or_else(identity_matrix)
            } else {
                identity_matrix()
            };
            if let Ok(mut pick) = self.pick.lock() {
                let request = self.pick_request.lock().ok().and_then(|mut slot| {
                    match crate::gpu_pick::take_decision(
                        slot.as_ref(),
                        pick_frame.ghost_key,
                        pick.has_free_slot(),
                    ) {
                        crate::gpu_pick::TakeDecision::Take => slot.take(),
                        crate::gpu_pick::TakeDecision::Drop => {
                            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                                eprintln!("[pick] drop stale request");
                            }
                            *slot = None;
                            None
                        }
                        crate::gpu_pick::TakeDecision::Keep => None,
                    }
                });
                if let Some(request) = request {
                    let uniforms = crate::gpu_pick::PickUniforms {
                        view_proj: self.object.view_proj(),
                        back: [0.0; 4],
                        eye: [0.0; 4],
                        params: [0.0; 4],
                        extra_model: identity_matrix(),
                    };
                    let plan_state = self.plan.lock().ok();
                    let object_state = resources
                        .get::<Arc<Mutex<ObjectRenderer>>>()
                        .and_then(|object| object.lock().ok());
                    pick.encode(
                        device,
                        queue,
                        encoder,
                        &uniforms,
                        &pick_frame,
                        request,
                        |pass, passes, binds| {
                            paint_pick_face_ids(
                                plan_active,
                                plan_active
                                    && !self
                                        .pick_frame
                                        .as_ref()
                                        .is_some_and(|frame| frame.ghost_edges),
                                plan_state.as_deref(),
                                object_state.as_deref(),
                                pass,
                                passes,
                                binds,
                            );
                        },
                        profiler.as_deref_mut(),
                    );
                }
            }
        }
        if let Some(backdrop) = resources.get_mut::<Arc<Mutex<BackdropRenderer>>>() {
            if let Ok(mut backdrop) = backdrop.lock() {
                backdrop.prepare(
                    device,
                    queue,
                    encoder,
                    &frame,
                    drawn.backdrop_view_proj(),
                    self.backdrop_blur_scale,
                    external_view.as_ref().map(|view| (view, self.backdrop_crop)),
                    profiler.as_deref_mut(),
                );
            }
        }
        Vec::new()
    }

    fn paint_impl(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(scene) = self.scene.lock().ok() else { return };
        let Some(scene) = scene.as_ref() else { return };
        let viewport = info.viewport_in_pixels();
        pass.set_viewport(
            viewport.left_px as f32,
            viewport.top_px as f32,
            viewport.width_px as f32,
            viewport.height_px as f32,
            0.0,
            1.0,
        );
        pass.set_pipeline(&scene.blit.pipeline);
        pass.set_bind_group(0, &scene.blit.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl CallbackTrait for GpuCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        crate::crash_report::guard_render("viewport", || {
            self.prepare_impl(device, queue, screen, encoder, resources)
        })
        .unwrap_or_default()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let _ = crate::crash_report::guard_render("viewport", || {
            self.paint_impl(info, pass, resources);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_mesh_is_chunked_with_local_indices() {
        let vertices: Vec<ObjectVertex> = (0..8)
            .map(|i| ObjectVertex {
                position: [i as f32, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                colour: [1.0, 0.0, 0.0],
            })
            .collect();
        let face_ids: Vec<u32> = (0..8).map(|i| 100 + i).collect();
        let indices = vec![0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0];
        let chunks = chunk_object_mesh_with_limit(vertices, face_ids, indices, 3);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks.iter().map(|(_, _, indices)| indices.len()).sum::<usize>(), 12);
        for (vertices, face_ids, indices) in &chunks {
            assert!(vertices.len() <= 3);
            assert_eq!(face_ids.len(), vertices.len(), "face ids travel with their vertices");
            assert!(indices.iter().all(|index| (*index as usize) < vertices.len()));
            assert!(face_ids.iter().all(|id| *id >= 100));
        }
    }

    #[test]
    fn lens_shift_is_constant_after_perspective_divide() {
        // clip.x = x, clip.w = z; after divide ndc.x = x/z.
        let vp = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let shifted = shift_view_proj(vp, [0.25, -0.5]);
        let apply = |m: [[f32; 4]; 4], x: f32, z: f32| {
            let mut out = [0.0f32; 4];
            for row in 0..4 {
                out[row] = m[0][row] * x + m[2][row] * z + m[3][row];
            }
            out
        };
        for (x, z) in [(0.0, 1.0), (2.0, 4.0), (-3.0, 7.0)] {
            let clip = apply(shifted, x, z);
            let ndc_x = clip[0] / clip[3];
            assert!((ndc_x - (x / z + 0.25)).abs() < 1e-6, "{x} {z} {ndc_x}");
        }
    }

    #[test]
    fn scene_pass_only_resolves_when_multisampled() {
        let (color, resolve) = ("color", "resolve");
        assert_eq!(scene_color_attachment(4, &color, &resolve), (&color, Some(&resolve)));
        assert_eq!(scene_color_attachment(1, &color, &resolve), (&resolve, None));
    }

    #[test]
    fn backdrop_pbr_copy_is_budgeted() {
        assert_eq!(backdrop_pbr_stride(0), 1);
        assert_eq!(backdrop_pbr_stride(BACKDROP_PBR_TRIANGLE_BUDGET), 1);
        let stride = backdrop_pbr_stride(3_356_630);
        assert!(stride > 1);
        assert!(3_356_630 / stride <= BACKDROP_PBR_TRIANGLE_BUDGET);
    }

    #[test]
    fn uniform_colour_mesh_keeps_its_indices() {
        use nodecraft_protocol::Mesh as NodeMesh;
        let source = NodeMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            faces: vec![[0, 1, 2]],
            ..NodeMesh::default()
        };
        let display = crate::adapter::display_mesh(&source);
        let faces = ObjectFaceIds::from_display(&display);
        let prepared =
            indexed_object_mesh(&display.mesh, [0.2, 0.3, 0.4], &faces).expect("indexed mesh");
        assert_eq!(prepared.vertices.len(), 3);
        assert_eq!(prepared.face_ids.len(), 3);
        assert_eq!(prepared.indices, vec![0, 1, 2]);
    }

    #[test]
    fn exact_normals_bypass_the_object_smoother() {
        use nodecraft_protocol::Mesh as NodeMesh;
        // The evaluated normal is not the triangle's flat normal. The object
        // renderer's uniform-colour smoother must not rebuild it.
        let exact = [0.6, 0.0, 0.8];
        let source = NodeMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![exact; 3],
            faces: vec![[0, 1, 2]],
            face_groups: vec!["face:0".into()],
            exact_normals: true,
            ..NodeMesh::default()
        };
        let display = crate::adapter::display_mesh(&source);
        assert!(display.exact_normals);
        let prepared = prepare_object(&display, &ViewSettings::default());
        assert_eq!(prepared.vertices.len(), 3);
        assert!(prepared.vertices.iter().all(|vertex| {
            (vertex.normal[0] - exact[0]).abs() < 1.0e-6
                && (vertex.normal[1] - exact[1]).abs() < 1.0e-6
                && (vertex.normal[2] - exact[2]).abs() < 1.0e-6
        }));
        assert!(
            prepared
                .face_ids
                .iter()
                .all(|id| { crate::gpu_pick::resolve_face_id(*id).as_deref() == Some("face:0") })
        );
    }

    #[test]
    fn shading_creases_follow_brep_faces_not_just_triangle_angle() {
        use nodecraft_protocol::Mesh as NodeMesh;
        // A horizontal and a 45-degree face share an edge. Their normals are
        // 45 degrees apart, so an angular-only smoother merges them and makes
        // a flat chamfer look rounded.
        let source = NodeMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            faces: vec![[0, 1, 2], [0, 1, 3]],
            face_groups: vec!["face:0".to_owned(), "face:1".to_owned()],
            ..NodeMesh::default()
        };
        let display = crate::adapter::display_mesh(&source);
        let faces = ObjectFaceIds::from_display(&display);
        let prepared = smoothed_object_mesh(&display.mesh, [0.2, 0.3, 0.4], &faces).expect("mesh");
        assert_eq!(prepared.vertices.len(), 6, "the two B-rep faces must not share normals");
    }

    #[test]
    fn identity_draw_uploads_identity_model_and_normal_matrix() {
        let draw = GpuDrawItem {
            asset_key: 1,
            transform: identity_matrix(),
            index_start: 0,
            index_count: 3,
            ghost: false,
        };
        let instance = draw.instance();
        assert_eq!(instance.model, identity_matrix());
        assert_eq!(instance.normal, identity_matrix());
    }

    #[test]
    fn object_shader_applies_per_instance_alpha() {
        assert!(
            object_shader().contains("u.params.x * in.alpha"),
            "the fragment shader must multiply the layer alpha by the instance alpha"
        );
        let draw = GpuDrawItem {
            asset_key: 1,
            transform: identity_matrix(),
            index_start: 0,
            index_count: 3,
            ghost: true,
        };
        assert_eq!(draw.instance().alpha, 1.0);
    }

    #[test]
    fn instance_payload_is_two_column_major_matrices() {
        // Two column-major matrices (128 bytes), then the per-instance ghost
        // alpha with three padding floats.
        assert_eq!(std::mem::size_of::<GpuInstance>(), 144);
        assert_eq!(OBJECT_INSTANCE_ATTRIBUTES[4].offset, 64);
        assert_eq!(OBJECT_INSTANCE_ATTRIBUTES[7].offset, 112);
        for attribute in &OBJECT_INSTANCE_ATTRIBUTES[..8] {
            assert_eq!(attribute.format, wgpu::VertexFormat::Float32x4);
        }
        assert_eq!(OBJECT_INSTANCE_ATTRIBUTES[8].offset, 128);
        assert_eq!(OBJECT_INSTANCE_ATTRIBUTES[8].format, wgpu::VertexFormat::Float32);
        assert_eq!(object_instance_layout().step_mode, wgpu::VertexStepMode::Instance);
    }

    #[test]
    fn normal_matrix_inverts_non_uniform_scale() {
        let mut model = identity_matrix();
        model[0][0] = 2.0;
        model[1][1] = 4.0;
        model[2][2] = 0.5;
        let normal = normal_matrix(model);
        assert!((normal[0][0] - 0.5).abs() < 1.0e-6);
        assert!((normal[1][1] - 0.25).abs() < 1.0e-6);
        assert!((normal[2][2] - 2.0).abs() < 1.0e-6);
        assert_eq!(normal[3], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn normal_matrix_of_rotation_is_the_rotation() {
        // 90 degrees about Z: orthogonal, so (M^-1)^T = M.
        let model = [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let normal = normal_matrix(model);
        for column in 0..4 {
            for row in 0..4 {
                assert!(
                    (normal[column][row] - model[column][row]).abs() < 1.0e-6,
                    "{column} {row}"
                );
            }
        }
    }

    #[test]
    fn normal_matrix_reflects_normals_for_a_mirrored_instance() {
        // The same instance matrix the plan pass uses: yz reflection with the
        // plane at x = 10. (M^-1)^T equals M for a reflection, so the +x
        // normal turns to -x while +z is unchanged; the sign of the normal
        // matrix's determinant stays with the placement, never dropped.
        let model = crate::plan_math::instance_matrix(&nodecraft_display::Placement::from(
            nodecraft_display::DisplayTransform::Mirror { plane: "yz", offset: 10.0 },
        ));
        let normal = normal_matrix(model);
        let transformed = |vector: [f32; 3]| {
            [
                normal[0][0] * vector[0] + normal[1][0] * vector[1] + normal[2][0] * vector[2],
                normal[0][1] * vector[0] + normal[1][1] * vector[1] + normal[2][1] * vector[2],
                normal[0][2] * vector[0] + normal[1][2] * vector[1] + normal[2][2] * vector[2],
            ]
        };
        assert_eq!(transformed([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0]);
        assert_eq!(transformed([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]);
        let determinant = glam::Mat4::from_cols_array_2d(&normal).determinant();
        assert!(determinant < 0.0, "the normal matrix keeps the reflection: {determinant}");
    }

    #[test]
    fn normal_matrix_ignores_translation_and_falls_back_on_degenerate_models() {
        let mut translated = identity_matrix();
        translated[3] = [5.0, -3.0, 2.0, 1.0];
        assert_eq!(normal_matrix(translated), identity_matrix());

        // Zero scale would make the inverse explode; identity is the safe fallback.
        let degenerate = [[0.0; 4]; 4];
        assert_eq!(normal_matrix(degenerate), identity_matrix());
    }

    fn ghost_edge(points: Vec<solidcore_record::Vec3>) -> crate::adapter::DisplayEdge {
        crate::adapter::DisplayEdge {
            id: "edge:0".into(),
            midpoint: [0.0, 0.0, 0.0],
            points,
            class: nodecraft_layers::PlanEdgeClass::NORMAL,
        }
    }

    #[test]
    fn multi_body_ghosts_are_flagged() {
        let edge = |id: &str| crate::adapter::DisplayEdge {
            id: id.into(),
            points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X],
            midpoint: [0.5, 0.0, 0.0],
            class: nodecraft_layers::PlanEdgeClass::NORMAL,
        };
        assert!(build_ghost_pick_data(&[edge("body:0:edge:0"), edge("body:1:edge:0")]).multi_body);
        assert!(!build_ghost_pick_data(&[edge("body:0:edge:0")]).multi_body);
    }

    #[test]
    fn resolve_pick_returns_none_for_index_zero() {
        let edge = ghost_edge(vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X]);
        let data = build_ghost_pick_data(&[edge]);
        assert!(resolve_pick(&data, 0).is_none());
        assert_eq!(resolve_pick(&data, 1).map(|entry| entry.0), Some("edge:0".to_owned()));
    }

    #[test]
    fn ghost_pick_indices_resolve_skipped_preview_edges() {
        let edges = vec![
            ghost_edge(vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X]),
            crate::adapter::DisplayEdge {
                id: "preview-edge:1".into(),
                points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::Y],
                midpoint: [0.0, 0.5, 0.0],
                class: nodecraft_layers::PlanEdgeClass::NORMAL,
            },
            crate::adapter::DisplayEdge {
                id: "edge:2".into(),
                points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::Z],
                midpoint: [0.0, 0.0, 0.5],
                class: nodecraft_layers::PlanEdgeClass::NORMAL,
            },
        ];
        let data = build_ghost_pick_data(&edges);
        assert_eq!(data.edges.get(2).map(|entry| entry.0.as_str()), Some("edge:2"));
    }

    fn staged_display(
        edge_id: &str,
        wait_activation: u64,
        snapshot: u64,
        key: u64,
        preview: Option<PreviewState>,
    ) -> PendingDisplay {
        PendingDisplay {
            wait_activation,
            edges: StagedEdges {
                snapshot,
                key,
                key_serial: key.wrapping_add(1),
                display: Some(Arc::new(DisplayMesh {
                    mesh: Arc::new(CraftMesh::default()),
                    lods: Vec::new(),
                    face_ids: Vec::new(),
                    exact_normals: false,
                    face_colors: Vec::new(),
                    body_ids: Vec::new(),
                    edges: vec![crate::adapter::DisplayEdge {
                        id: edge_id.to_owned(),
                        points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X],
                        midpoint: [0.5, 0.0, 0.0],
                        class: nodecraft_layers::PlanEdgeClass::NORMAL,
                    }],
                    comparison: None,
                    selection_edges: None,
                    radius: 1.0,
                    center: solidcore_record::Vec3::ZERO,
                })),
                edge_batch: EdgeCpuBatch::from_display_edges(&[crate::adapter::DisplayEdge {
                    id: edge_id.to_owned(),
                    points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X],
                    midpoint: [0.5, 0.0, 0.0],
                    class: nodecraft_layers::PlanEdgeClass::NORMAL,
                }]),
                scene: GpuScene::default(),
            },
            source_mesh: None,
            prepared: None,
            diagnostics: None,
            target: Some(edge_id.to_owned()),
            preview,
        }
    }

    fn test_frame(key: u64) -> Frame {
        Frame {
            product: false,
            preview_bounds: None,
            moving: None,
            product_lights: None,
            upload_error: None,
            clouds: Vec::new(),
            key,
            turned_depth: None,
            slot: Arc::new(Mutex::new(None)),
            sketch_key: 0,
            sketch_slot: Arc::new(Mutex::new(None)),
            hl_key: 0,
            hl_slot: Arc::new(Mutex::new(None)),
            pv_key: 0,
            pv_slot: Arc::new(Mutex::new(None)),
            pv_hide: false,
            view_proj: identity_matrix(),
            back: [0.0; 3],
            size_px: [1.0, 1.0],
            clip: clip_planes(&[]),
            clip_hatch: clip_hatches(&[]),
            cap: [0.0; 4],
            images: Vec::new(),
            analysis: [0.0; 8],
            analysis_colors: [0.0; 36],
            access: None,
            outline: false,
            eye: [0.0; 4],
            ao_radius: 0.0,
            ao_samples: solidcraft_viewport_gpu::view::AO_SAMPLES,
            ssaa: 1.0,
            hover_tint: [0.0; 3],
            selection_tint: [0.0; 3],
            hatch: Default::default(),
            sky: [0.0, 1.0, 0.0, 0.0],
        }
    }

    fn active_edge_now(swap: &DisplaySwap) -> String {
        swap.active
            .as_ref()
            .and_then(|(_, active)| active.display.as_ref())
            .map(|display| display.edges[0].id.clone())
            .unwrap_or_default()
    }

    fn active_edge(swap: &mut DisplaySwap, synced: &mut u64) -> String {
        swap.take_activated(synced)
            .and_then(|active| active.display)
            .map(|display| display.edges[0].id.clone())
            .unwrap_or_default()
    }

    /// The resize bug: a staged plan set must keep both the old triangles and
    /// the old edge mesh. A slow upload can delay activation, but it can never
    /// let the new edges overtake the new plan chunks.
    #[test]
    fn a_staged_display_waits_for_its_plan_activation_and_switches_edges_with_it() {
        let mut swap = DisplaySwap::default();
        swap.active = Some((
            1,
            DisplayActivation {
                snapshot: 1,
                key: 1,
                key_serial: 2,
                display: Some(Arc::new(DisplayMesh {
                    mesh: Arc::new(CraftMesh::default()),
                    lods: Vec::new(),
                    face_ids: Vec::new(),
                    exact_normals: false,
                    face_colors: Vec::new(),
                    body_ids: Vec::new(),
                    edges: vec![crate::adapter::DisplayEdge {
                        id: "old".to_owned(),
                        points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::X],
                        midpoint: [0.5, 0.0, 0.0],
                        class: nodecraft_layers::PlanEdgeClass::NORMAL,
                    }],
                    comparison: None,
                    selection_edges: None,
                    radius: 1.0,
                    center: solidcore_record::Vec3::ZERO,
                })),
                source_mesh: None,
                diagnostics: None,
                target: Some("old".to_owned()),
                preview: None,
            },
        ));
        swap.serial = 1;
        swap.stage(staged_display("new", 3, 2, 3, None));
        let mut synced = 0;
        assert_eq!(active_edge(&mut swap, &mut synced), "old");
        // Two prepare calls upload chunks but do not finish the plan set.
        assert!(swap.take_ready(1, 1).is_none());
        assert_eq!(active_edge_now(&swap), "old");
        assert!(swap.take_ready(2, 2).is_none());
        assert_eq!(active_edge_now(&swap), "old");
        // Activation switches the payload in the same step as the plan set.
        assert!(swap.take_ready(3, 2).is_some());
        assert_eq!(active_edge(&mut swap, &mut synced), "new");
        assert_eq!(swap.committed_snapshot, 2);
        assert!(swap.pending.is_none());
    }

    /// The lag bug: `stage_display` used to stage only the scene, never a key.
    /// `prepare` then handed the renderer the key captured when the callback
    /// was built, so the renderer skipped the upload and kept drawing the old
    /// edges until an unrelated key bump happened later. The activated pair
    /// must hand over the staged key and scene together.
    #[test]
    fn a_staged_activation_hands_the_renderer_the_new_scene_and_its_key() {
        let mut swap = DisplaySwap::default();
        // The renderer already holds the scene uploaded under this key.
        let uploaded_key = 1;
        swap.active = Some((
            1,
            DisplayActivation {
                snapshot: 1,
                key: uploaded_key,
                key_serial: 2,
                display: None,
                source_mesh: None,
                diagnostics: None,
                target: Some("old".to_owned()),
                preview: None,
            },
        ));
        swap.serial = 1;

        let pending = staged_display("new", 2, 2, 2, None);
        assert!(!pending.edges.edge_batch.is_empty(), "the staged pair carries its edges");
        swap.stage(pending);

        // The frame built for this paint still carries the old key.
        let mut frame = test_frame(uploaded_key);
        let pending = swap.take_ready(2, 2).expect("plan and display snapshots agree");
        apply_activated_display(&pending, &mut frame);

        assert_ne!(frame.key, uploaded_key, "the renderer would skip an unchanged key");
        assert_eq!(frame.key, pending.edges.key);
        let activated = frame.slot.lock().unwrap().take().expect("activated scene");
        assert!(activated.overlay.is_empty(), "edges no longer travel in the scene");

        // The next sync adopts the activated key so later frames keep it.
        let mut synced = 0;
        let active = swap.take_activated(&mut synced).expect("active pair");
        assert_eq!(active.key, 2);
        let (mut key, mut key_serial) = (uploaded_key, 2);
        adopt_scene_key(&mut key, &mut key_serial, &active);
        assert_eq!((key, key_serial), (2, 3));
        // A newer immediate upload is not rolled back to the older activation.
        let (mut key, mut key_serial) = (4, 5);
        adopt_scene_key(&mut key, &mut key_serial, &active);
        assert_eq!((key, key_serial), (4, 5));
    }

    fn translation(x: f32) -> [[f32; 4]; 4] {
        let mut matrix = identity_matrix();
        matrix[3] = [x, 0.0, 0.0, 1.0];
        matrix
    }

    fn whole_preview(matrix: [[f32; 4]; 4]) -> PreviewState {
        PreviewState { transform: matrix, alpha: 1.0 }
    }

    /// The snap-back bug: the app used to clear the plan preview right after
    /// `set_display`, while the already-built frame still carried the preview in
    /// its scene projection. The preview now travels with the staged pair, so
    /// before activation both layers keep it and activation drops both together
    /// with the new scene.
    #[test]
    fn a_staged_activation_retires_the_preview_with_the_new_scene() {
        let base = identity_matrix();
        let preview_matrix = translation(2.0);
        let mut swap = DisplaySwap::default();
        swap.preview = Some(whole_preview(preview_matrix));

        // The frame built while the pair is still staged: one preview value for
        // the plan uniform and the scene projection.
        let mut drawn = DrawnPreview::new(base, base, swap.preview.clone());
        let mut frame = test_frame(1);
        sync_drawn_preview(&mut drawn, &mut frame, swap.preview.clone());
        assert_eq!(drawn.plan_transform(), Some(preview_matrix));
        assert_eq!(frame.view_proj, multiply_matrix(base, preview_matrix));

        let pending = staged_display("new", 2, 2, 2, None);
        assert!(!pending.edges.edge_batch.is_empty());
        swap.stage(pending);

        // One prepare: the staged pair installs the new scene and its key and
        // retires the preview for the same frame.
        let pending = swap.take_ready(2, 2).expect("plan and display snapshots agree");
        apply_activated_display(&pending, &mut frame);
        sync_drawn_preview(&mut drawn, &mut frame, swap.preview.clone());

        assert_eq!(swap.preview, None, "the staged pair retires the preview");
        assert_eq!(drawn.plan_transform(), None, "plan preview must drop");
        assert_eq!(frame.view_proj, base, "scene projection must drop it too");
        assert_eq!(frame.key, pending.edges.key);
        let activated = frame.slot.lock().unwrap().take().expect("activated scene");
        assert!(activated.overlay.is_empty(), "edges are a ribbon pass, not scene lines");
    }

    /// During a drag the plan uniform and the scene projection of one prepared
    /// frame must be two uses of the same preview matrix, never two owners.
    #[test]
    fn a_drag_frame_derives_plan_and_scene_projection_from_one_matrix() {
        let base = identity_matrix();
        let preview_matrix = translation(3.0);
        let preview = whole_preview(preview_matrix);
        let mut drawn = DrawnPreview::new(base, base, Some(preview.clone()));
        let mut frame = test_frame(1);
        sync_drawn_preview(&mut drawn, &mut frame, Some(preview));

        let plan = drawn.plan_transform().expect("whole-mesh preview");
        assert_eq!(plan, preview_matrix);
        assert_eq!(frame.view_proj, multiply_matrix(base, plan));
        assert_eq!(drawn.backdrop_view_proj(), multiply_matrix(base, plan));
    }

    #[test]
    fn a_newer_snapshot_replaces_the_whole_staged_pair() {
        let mut swap = DisplaySwap::default();
        // A continuous drag produces N, N+1, N+2 before the first upload
        // finishes. Each arrival replaces the previous staged pair whole.
        swap.stage(staged_display("edges-2", 2, 2, 2, None));
        swap.stage(staged_display("edges-3", 3, 3, 3, None));
        swap.stage(staged_display("edges-4", 4, 4, 4, None));
        assert!(swap.take_ready(2, 2).is_none());
        assert_eq!(active_edge_now(&swap), "");
        assert_eq!(swap.committed_snapshot, 0);
        // A plan set from an older snapshot must not consume the newer edge
        // payload (or vice versa): no activation until both ids agree.
        assert!(swap.take_ready(4, 2).is_none());
        assert_eq!(swap.committed_snapshot, 0);
        assert!(swap.take_ready(4, 4).is_some());
        assert_eq!(active_edge_now(&swap), "edges-4");
        assert_eq!(swap.committed_snapshot, 4);
    }

    #[test]
    fn a_display_activated_after_its_plan_set_is_not_replayed() {
        let mut swap = DisplaySwap::default();
        swap.stage(staged_display("new", 2, 2, 2, None));
        assert!(swap.take_ready(2, 2).is_some());
        let mut synced = 0;
        assert_eq!(active_edge(&mut swap, &mut synced), "new");
        // The CPU mirror consumes the activation once.
        assert_eq!(active_edge(&mut swap, &mut synced), "");
    }

    #[test]
    fn ghost_pick_data_skips_preview_edges_and_degenerate_segments() {
        let edges = vec![
            ghost_edge(vec![
                solidcore_record::Vec3::ZERO,
                solidcore_record::Vec3::X,
                solidcore_record::Vec3::X,
            ]),
            crate::adapter::DisplayEdge {
                id: "preview-edge:1".into(),
                points: vec![solidcore_record::Vec3::ZERO, solidcore_record::Vec3::Y],
                midpoint: [0.0, 0.5, 0.0],
                class: nodecraft_layers::PlanEdgeClass::NORMAL,
            },
        ];
        let data = build_ghost_pick_data(&edges);
        assert_eq!(data.segments.len(), 1);
        assert_eq!(data.segments[0].id, 1);
        assert_eq!(data.edges.len(), 2);
        assert_eq!(data.edges[1].0, "edge:0");
    }

    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .ok()
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))
            .ok()
        })?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn perspective_rh(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        glam::Mat4::perspective_rh(fov_y, aspect, near, far).to_cols_array_2d()
    }

    fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        glam::Mat4::look_at_rh(eye.into(), target.into(), up.into()).to_cols_array_2d()
    }

    fn project_px(vp: [[f32; 4]; 4], p: [f32; 3], size: [f32; 2]) -> [f32; 2] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            out[row] = (0..4)
                .map(|column| vp[column][row] * if column == 3 { 1.0 } else { p[column] })
                .sum();
        }
        let ndc = [out[0] / out[3], out[1] / out[3]];
        [(ndc[0] * 0.5 + 0.5) * size[0], (1.0 - (ndc[1] * 0.5 + 0.5)) * size[1]]
    }

    /// A cube whose display faces all map to one known pick string, so the
    /// object path is identifiable even when the plan path also draws a box.
    fn face_box_display(face_ids: Vec<String>, tri_face: Vec<u32>) -> DisplayMesh {
        let corners: [[f64; 3]; 8] = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];
        let faces: [[usize; 4]; 6] = [
            [4, 5, 6, 7], // +z
            [1, 0, 3, 2], // -z
            [5, 1, 2, 6], // +x
            [0, 4, 7, 3], // -x
            [3, 7, 6, 2], // +y
            [0, 1, 5, 4], // -y
        ];
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        for quad in faces {
            let base = positions.len() as u32;
            for corner in quad {
                let p = corners[corner];
                positions.push(solidcore_record::Vec3::new(p[0], p[1], p[2]));
            }
            triangles.push([base, base + 1, base + 2]);
            triangles.push([base, base + 2, base + 3]);
        }
        assert_eq!(triangles.len(), tri_face.len());
        let mesh = CraftMesh {
            positions,
            normals: Vec::new(),
            triangles,
            tri_face,
            edges: Vec::new(),
            display_only_edges: false,
            seams: Vec::new(),
            smooth_edges: Vec::new(),
            sharp_edges: Vec::new(),
            edge_faces: Vec::new(),
        };
        DisplayMesh {
            mesh: Arc::new(mesh),
            lods: Vec::new(),
            face_ids,
            exact_normals: false,
            face_colors: Vec::new(),
            body_ids: Vec::new(),
            edges: Vec::new(),
            comparison: None,
            selection_edges: None,
            radius: 1.0,
            center: solidcore_record::Vec3::ZERO,
        }
    }

    fn box_plan_sized(size: f64) -> std::sync::Arc<nodecraft_layers::tess::TessPlan> {
        use nodecraft_solidcore::solidcore::math::Point3;
        use nodecraft_solidcore::solidcore::primitive::api_primitives as api;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep = api::create_box(
            &mut kernel,
            Point3::new(-size * 0.5, -size * 0.5, -size * 0.5),
            size,
            size,
            size,
        )
        .expect("box");
        nodecraft_layers::tessellate_kernel_preview(&mut kernel, &mut brep, 2)
            .expect("box preview plan")
            .plan
    }

    fn run_face_pick(
        picker: &mut crate::gpu_pick::PickRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &crate::gpu_pick::PickUniforms,
        frame: &crate::gpu_pick::PickFrame,
        request: crate::gpu_pick::PickRequest,
        plan: Option<&crate::gpu_plan::PlanState>,
        object: Option<&ObjectRenderer>,
        plan_active: bool,
    ) -> Option<crate::gpu_pick::PickResult> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("face-pick test"),
        });
        picker.encode(
            device,
            queue,
            &mut encoder,
            uniforms,
            frame,
            request,
            |pass, passes, binds| {
                paint_pick_face_ids(plan_active, plan_active, plan, object, pass, passes, binds);
            },
            None,
        );
        queue.submit([encoder.finish()]);
        for _ in 0..200 {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            picker.service(device);
            if let Some(result) = picker.take_result() {
                return Some(result);
            }
        }
        None
    }

    fn pick_faces_at(
        picker: &mut crate::gpu_pick::PickRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &crate::gpu_pick::PickUniforms,
        frame: &crate::gpu_pick::PickFrame,
        plan: Option<&crate::gpu_plan::PlanState>,
        object: Option<&ObjectRenderer>,
        plan_active: bool,
        id: u64,
        point: [f32; 3],
    ) -> u32 {
        let px = project_px(uniforms.view_proj, point, frame.size_px);
        let request = crate::gpu_pick::PickRequest {
            id,
            px,
            radius_px: 1.0,
            ghost_key: frame.ghost_key,
            commit: false,
            kind: crate::gpu_pick::PickKind::Faces,
        };
        run_face_pick(picker, device, queue, uniforms, frame, request, plan, object, plan_active)
            .map(|result| result.index)
            .unwrap_or(0)
    }

    /// One plan, one mesh, four placements: the original body, two translated
    /// copies and a `yz` mirror. All bodies share the same `Arc<TessPlan>`
    /// exactly as a combined mirror does (it clones a body instance and appends
    /// a mirror transform), so the plan's resident mesh and colouring are
    /// reused while every placement stays in the group's instance list.
    fn shared_plan_placements() -> Vec<nodecraft_display::Placement> {
        vec![
            nodecraft_display::Placement::identity(),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [-20.0, 0.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [40.0, 0.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Mirror {
                plane: "yz",
                offset: 10.0,
            }),
        ]
    }

    /// The B-rep face whose triangle centre points furthest along `direction`.
    /// Tests use it to sample a known plan face without assuming box face
    /// order.
    fn plan_face_toward(
        plan: &std::sync::Arc<nodecraft_layers::tess::TessPlan>,
        direction: [f32; 3],
    ) -> u32 {
        let display = nodecraft_layers::evaluate_plan(plan, &[]);
        let mesh = &display.mesh;
        let mut best = (f64::NEG_INFINITY, 0u32);
        for (triangle, group) in mesh.faces.iter().zip(&mesh.face_groups) {
            let face =
                group.strip_prefix("face:").and_then(|tail| tail.parse::<u32>().ok()).unwrap_or(0);
            let mut centre = [0.0f64; 3];
            let mut count = 0usize;
            for index in triangle {
                if let Some(point) = mesh.vertices.get(*index as usize) {
                    centre[0] += f64::from(point[0]);
                    centre[1] += f64::from(point[1]);
                    centre[2] += f64::from(point[2]);
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let distance = centre[0] / count as f64 * f64::from(direction[0])
                + centre[1] / count as f64 * f64::from(direction[1])
                + centre[2] / count as f64 * f64::from(direction[2]);
            if distance > best.0 {
                best = (distance, face);
            }
        }
        best.1
    }

    /// Render one plan set into an offscreen colour target and read it back.
    fn render_plan_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plan: &crate::gpu_plan::PlanState,
        size: [u32; 2],
        uniforms: &ObjectUniforms,
    ) -> (Vec<u8>, u32) {
        plan.write_uniforms(queue, uniforms, None, [64.0, 64.0]);
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plan colour test colour"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let colour_view = colour.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plan colour test depth"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plan colour test"),
        });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("plan colour test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.1, b: 0.15, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            plan.paint(&mut pass);
        }
        let bytes_per_row = size[0] * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plan colour test readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &colour,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.recv().expect("map callback").expect("map");
        let pixels = slice.get_mapped_range().expect("map range").to_vec();
        readback.unmap();
        (pixels, bytes_per_row)
    }

    fn pixel_at(
        pixels: &[u8],
        bytes_per_row: u32,
        vp: [[f32; 4]; 4],
        point: [f32; 3],
        size: [u32; 2],
    ) -> [u8; 4] {
        let px = project_px(vp, point, [size[0] as f32, size[1] as f32]);
        let x = px[0].round().clamp(0.0, size[0] as f32 - 1.0) as usize;
        let y = px[1].round().clamp(0.0, size[1] as f32 - 1.0) as usize;
        let at = y * bytes_per_row as usize + x * 4;
        [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
    }

    /// Three bodies share one `Arc<TessPlan>` but author different per-face
    /// colours, including one with no authored colour. They must still be one
    /// chunk and one instanced draw, and every body's face must show its own
    /// table colour.
    #[test]
    fn shared_plan_bodies_with_different_face_colours_draw_in_one_instanced_chunk() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let size = [384u32, 128u32];
        let eye = [0.0, 0.0, 260.0];
        let view = look_at_rh(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = gpu_view_proj(multiply_matrix(perspective_rh(0.5, 3.0, 0.1, 1000.0), view), false);
        let uniforms = ObjectUniforms {
            view_proj: vp,
            back: [0.0, 0.0, 1.0, 1.0],
            eye: [eye[0], eye[1], eye[2], 1.0],
            params: [1.0, 0.0, 0.45, 0.0],
            extra_model: identity_matrix(),
        };
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(10.0);
        let faces = shared.faces.len();
        let plus_z = plan_face_toward(&shared, [0.0, 0.0, 1.0]) as usize;
        let mut red = vec![DEFAULT_FACE_COLOUR; faces];
        red[plus_z] = [0.8, 0.04, 0.04];
        let mut green = vec![DEFAULT_FACE_COLOUR; faces];
        green[plus_z] = [0.04, 0.8, 0.04];
        let instance = |face_colors: Vec<[f32; 3]>, x: f64| nodecraft_display::PlanInstance {
            body: 0,
            target: "test".into(),
            plan: std::sync::Arc::clone(&shared),
            face_colors,
            placement: nodecraft_display::Placement::from(
                nodecraft_display::DisplayTransform::Place {
                    offset: [x, 0.0, 0.0],
                    rotation: [0.0; 3],
                },
            ),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        plan.set_plans(
            vec![instance(red, -25.0), instance(green, 0.0), instance(Vec::new(), 25.0)],
            true,
            None,
            1,
        );
        plan.prepare_chunks(&device, &queue);
        // Exercise the triangle-edge pipeline too: it shares the plan layout
        // and must bind the same colour table without disturbing the surface.
        plan.set_triangles(true);
        plan.prepare_chunks(&device, &queue);
        let debug = plan.debug_counts();
        assert_eq!(debug.chunks, 1, "different face colours share one instanced chunk");
        assert_eq!(debug.instances, 3, "all three bodies are instances of that chunk");

        // Draw once with the triangle overlay to exercise its pipeline and
        // table binding; then sample the shaded surface without its dark
        // triangle edges crossing the face centre.
        let _ = render_plan_pixels(&device, &queue, &plan, size, &uniforms);
        plan.set_triangles(false);
        plan.prepare_chunks(&device, &queue);
        let (pixels, row) = render_plan_pixels(&device, &queue, &plan, size, &uniforms);
        let red_px = pixel_at(&pixels, row, vp, [-25.0, 0.0, 5.0], size);
        let green_px = pixel_at(&pixels, row, vp, [0.0, 0.0, 5.0], size);
        let default_px = pixel_at(&pixels, row, vp, [25.0, 0.0, 5.0], size);
        for (name, pixel) in [("red", red_px), ("green", green_px), ("default", default_px)] {
            assert_eq!(pixel[3], 255, "the {name} body must be opaque");
        }
        assert!(
            red_px[0] > red_px[1].saturating_add(30) && red_px[0] > red_px[2].saturating_add(30),
            "the red body's +z face must use its own table slice: {red_px:?}",
        );
        assert!(
            green_px[1] > green_px[0].saturating_add(30)
                && green_px[1] > green_px[2].saturating_add(30),
            "the green body's +z face must use its own table slice: {green_px:?}",
        );
        let spread = default_px[..3].iter().copied().max().unwrap()
            - default_px[..3].iter().copied().min().unwrap();
        assert!(
            spread < 40,
            "the body with no authored colour must use the default table colour: {default_px:?}",
        );
    }

    /// Face-colours mode is a table fill on the plan path: adjacent faces of one
    /// box must render with the palette's distinct colours, and no CPU object
    /// mesh is involved.
    #[test]
    fn face_colours_view_mode_renders_adjacent_faces_from_the_plan_table() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let size = [256u32, 256u32];
        let eye = [30.0, -20.0, 30.0];
        let view = look_at_rh(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = gpu_view_proj(multiply_matrix(perspective_rh(0.7, 1.0, 0.1, 1000.0), view), false);
        let uniforms = ObjectUniforms {
            view_proj: vp,
            back: [0.0, 0.0, 1.0, 1.0],
            eye: [eye[0], eye[1], eye[2], 1.0],
            params: [1.0, 0.0, 0.45, 0.0],
            extra_model: identity_matrix(),
        };
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(10.0);
        let plus_z = plan_face_toward(&shared, [0.0, 0.0, 1.0]);
        let plus_x = plan_face_toward(&shared, [1.0, 0.0, 0.0]);
        let body = 0x1234_5678_9abc_def0;
        assert_ne!(
            face_colour(body, plus_z),
            face_colour(body, plus_x),
            "the fixture's adjacent faces need distinct palette colours",
        );
        plan.set_plans(
            vec![nodecraft_display::PlanInstance {
                target: "test".into(),
                plan: std::sync::Arc::clone(&shared),
                face_colors: Vec::new(),
                body,
                placement: nodecraft_display::Placement::identity(),
                edge_ids: std::sync::Arc::from(Vec::new()),
                edge_classes: std::sync::Arc::from(Vec::new()),
            }],
            true,
            None,
            1,
        );
        plan.set_face_colours(true);
        plan.prepare_chunks(&device, &queue);
        let debug = plan.debug_counts();
        assert!(debug.face_colours);
        assert_eq!(debug.chunks, 1, "face-colours mode stays on the plan path");

        let (pixels, row) = render_plan_pixels(&device, &queue, &plan, size, &uniforms);
        let z_px = pixel_at(&pixels, row, vp, [0.0, 0.0, 5.0], size);
        let x_px = pixel_at(&pixels, row, vp, [5.0, 0.0, 0.0], size);
        assert_eq!(z_px[3], 255, "the +z face must be drawn by the plan path");
        assert_eq!(x_px[3], 255, "the +x face must be drawn by the plan path");
        let difference = (0..3)
            .map(|channel| (i32::from(z_px[channel]) - i32::from(x_px[channel])).unsigned_abs())
            .sum::<u32>();
        assert!(
            difference > 20,
            "adjacent faces must get their distinct table colours: +z={z_px:?} +x={x_px:?}",
        );
    }

    /// The CPU object mesh and the plan colour table must derive the same
    /// colour from the same `(body, face)` pair. The display mesh carries the
    /// plan's body identity, and both paths parse the B-rep face number out of
    /// the display face string.
    #[test]
    fn cpu_and_plan_face_colours_agree_for_the_same_body_and_face() {
        let body = 0xfeed_face_cafe_beef;
        let face = 3u32;
        let mut display = face_box_display(vec![format!("face:{face}")], vec![0; 12]);
        display.body_ids = vec![body];
        let prepared = prepare_object(
            &display,
            &ViewSettings { face_colours: true, ..ViewSettings::default() },
        );
        let plan = std::sync::Arc::new(nodecraft_layers::tess::TessPlan {
            faces: vec![Default::default(); face as usize + 1],
            ..Default::default()
        });
        let (colours, offsets) = crate::gpu_plan::colour_table_data(
            &[nodecraft_display::PlanInstance {
                target: "test".into(),
                plan,
                face_colors: Vec::new(),
                body,
                placement: nodecraft_display::Placement::identity(),
                edge_ids: std::sync::Arc::from(Vec::new()),
                edge_classes: std::sync::Arc::from(Vec::new()),
            }],
            None,
            true,
        );
        let expected = colours[offsets[0] as usize + face as usize];
        assert_eq!(
            prepared.vertices[0].colour,
            [expected[0], expected[1], expected[2]],
            "CPU and plan paths must agree on (body, face)",
        );
    }

    /// `immediate` installs the shared-plan set through the resident/no-staging
    /// path after a one-body set, which is what a combine toggle back to `true`
    /// does. If that path folds the bodies of the shared plan into one
    /// placement, the render and pick tests below lose a body.
    fn mirrored_plan_pair(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        immediate: bool,
    ) -> crate::gpu_plan::PlanState {
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let plan_arc = box_plan_sized(10.0);
        let instance = |placement| nodecraft_display::PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&plan_arc),
            face_colors: Vec::new(),
            body: 0,
            placement,
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let four = || shared_plan_placements().into_iter().map(instance).collect::<Vec<_>>();
        if immediate {
            plan.set_plans(vec![instance(nodecraft_display::Placement::identity())], true, None, 0);
            plan.prepare_chunks(device, queue);
            let state = plan.set_plans(four(), true, None, 1);
            assert_eq!(
                state,
                crate::gpu_plan::PlanSetState::Immediate,
                "an all-resident shared-plan set must activate without staging",
            );
        } else {
            plan.set_plans(four(), true, None, 1);
        }
        plan.prepare_chunks(device, queue);
        assert!(plan.is_available(), "plan must be drawable: {:?}", plan.debug_counts());
        let debug = plan.debug_counts();
        assert_eq!(
            debug.chunks, 1,
            "four bodies of one plan are one chunk (immediate={immediate})",
        );
        assert_eq!(
            debug.instances, 4,
            "the one chunk has every body as an instance (immediate={immediate})",
        );
        plan
    }

    /// Every instance of the shared plan must draw from its own placement. If
    /// the instances share one matrix, the last placement overwrites the rest
    /// and the other bodies have no shaded pixels at all (only their separately
    /// transformed edges remain).
    fn assert_mirrored_plan_render(device: &wgpu::Device, queue: &wgpu::Queue, immediate: bool) {
        let size = [256u32, 128u32];
        let view = look_at_rh([10.0, 0.0, 200.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = gpu_view_proj(multiply_matrix(perspective_rh(0.2, 2.0, 0.1, 1000.0), view), false);
        let uniforms = ObjectUniforms {
            view_proj: vp,
            back: [0.0, 0.0, 1.0, 1.0],
            eye: [10.0, 0.0, 200.0, 1.0],
            params: [1.0, 0.0, 0.45, 0.0],
            extra_model: identity_matrix(),
        };
        let plan = mirrored_plan_pair(&device, &queue, immediate);
        plan.write_uniforms(&queue, &uniforms, None, [64.0, 64.0]);

        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mirror render test color"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mirror render test depth"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mirror render test"),
        });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mirror render test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.1, b: 0.15, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            plan.paint(&mut pass);
        }
        let bytes_per_row = size[0] * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mirror render test readback"),
            size: (bytes_per_row * size[1]) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.recv().expect("map callback").expect("map");
        let pixels = slice.get_mapped_range().expect("map range").to_vec();
        readback.unmap();
        let sample = |point: [f32; 3]| {
            let px = project_px(vp, point, [size[0] as f32, size[1] as f32]);
            let x = px[0].round().clamp(0.0, size[0] as f32 - 1.0) as usize;
            let y = px[1].round().clamp(0.0, size[1] as f32 - 1.0) as usize;
            let at = y * bytes_per_row as usize + x * 4;
            [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
        };
        // Every instance samples the +z face at its own placement. The mirror
        // must shade like the original's corresponding face; the translated
        // copies only need to be opaque and cover their own face.
        let original = sample([0.0, 0.0, 5.0]);
        let left = sample([-20.0, 0.0, 5.0]);
        let right = sample([40.0, 0.0, 5.0]);
        let mirrored = sample([20.0, 0.0, 5.0]);
        let background = sample([10.0, 0.0, 80.0]);
        for (name, pixel) in [
            ("original", original),
            ("left translate", left),
            ("right translate", right),
            ("mirrored", mirrored),
        ] {
            assert_eq!(pixel[3], 255, "the {name} body must be opaque");
            assert_ne!(
                pixel, background,
                "the {name} body must cover its placement, not show the background",
            );
        }
        for channel in 0..3 {
            assert!(
                (original[channel] as i32 - mirrored[channel] as i32).abs() <= 2,
                "mirrored face must shade like the original's corresponding face: \
                 original={original:?} mirrored={mirrored:?} (immediate={immediate})",
            );
        }
    }

    #[test]
    fn mirrored_plan_instance_draws_opaque_faces_at_its_own_placement() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        for immediate in [false, true] {
            assert_mirrored_plan_render(&device, &queue, immediate);
        }
    }

    fn assert_mirrored_plan_pick(device: &wgpu::Device, queue: &wgpu::Queue, immediate: bool) {
        let size = [256.0f32, 128.0];
        let view = look_at_rh([10.0, 0.0, 200.0], [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = gpu_view_proj(multiply_matrix(perspective_rh(0.2, 2.0, 0.1, 1000.0), view), false);
        let plan = mirrored_plan_pair(&device, &queue, immediate);
        let uniforms = crate::gpu_pick::PickUniforms {
            view_proj: vp,
            back: [0.0, 0.0, 1.0, 1.0],
            eye: [10.0, 0.0, 200.0, 1.0],
            params: [0.0; 4],
            extra_model: identity_matrix(),
        };
        let frame = crate::gpu_pick::PickFrame {
            request: None,
            ghost_key: 71,
            segments: Arc::from(Vec::new()),
            ghost_edges: false,
            face_matrix: identity_matrix(),
            ghost_matrix: identity_matrix(),
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let mut picker = crate::gpu_pick::PickRenderer::install(
            &device,
            wgpu::TextureFormat::Depth32Float,
            false,
        );
        let points = [[0.0, 0.0, 5.0], [-20.0, 0.0, 5.0], [40.0, 0.0, 5.0], [20.0, 0.0, 5.0]];
        for (body, point) in points.iter().enumerate() {
            let id = pick_faces_at(
                &mut picker,
                &device,
                &queue,
                &uniforms,
                &frame,
                Some(&plan),
                None,
                true,
                body as u64 + 1,
                *point,
            );
            let resolved = crate::gpu_pick::resolve_face_id(id);
            assert!(
                resolved.as_deref().is_some_and(|id| id.starts_with(&format!("body:{body}:face:"))),
                "instance {body} must pick its own face string, got {resolved:?} ({id}) \
                 (immediate={immediate})",
            );
        }
    }

    /// Picking uses the same instance buffer and per-instance body list as the
    /// shaded plan pass. Every body of the shared plan must return its own
    /// `body:N:face:M`, not another instance's or nothing because its placement
    /// never reached the GPU. Both the staged activation and the resident
    /// immediate commit must pass.
    #[test]
    fn mirrored_plan_instance_faces_are_pickable_at_its_own_placement() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        for immediate in [false, true] {
            assert_mirrored_plan_pick(&device, &queue, immediate);
        }
    }

    #[test]
    fn object_face_ids_keep_the_display_body_and_face_strings() {
        let mut display = face_box_display(vec!["face:7".into()], vec![0; 12]);
        // Both object-mesh builders (the per-triangle preview/face-colour path
        // and the smoothed uniform-colour path) must carry the same ids.
        for view in [
            ViewSettings { face_colours: true, ..ViewSettings::default() },
            ViewSettings::default(),
        ] {
            let prepared = prepare_object(&display, &view);
            assert!(!prepared.face_ids.is_empty());
            assert!(
                prepared.face_ids.iter().all(|id| {
                    crate::gpu_pick::resolve_face_id(*id).as_deref() == Some("face:7")
                })
            );
        }

        // A merged one-body mesh still carries `body:0:` prefixes; the plan
        // path does not add a body prefix until there is a second body.
        display.face_ids = vec!["body:0:face:4".into()];
        let view = ViewSettings::default();
        let prepared = prepare_object(&display, &view);
        assert!(!prepared.face_ids.is_empty());
        assert!(
            prepared
                .face_ids
                .iter()
                .all(|id| { crate::gpu_pick::resolve_face_id(*id).as_deref() == Some("face:4") })
        );

        // Two bodies switch to the plan path's `body:N:face:M` form.
        display.face_ids = vec!["body:1:face:4".into(), "body:2:face:5".into()];
        let prepared = prepare_object(&display, &view);
        assert!(prepared.face_ids.iter().all(|id| {
            crate::gpu_pick::resolve_face_id(*id).as_deref() == Some("body:1:face:4")
        }));

        // A synthetic face group still keeps its body; only the face falls back
        // to the display face index.
        display.face_ids = vec!["body:2:blend:7".into(), "body:3:face:1".into()];
        let prepared = prepare_object(&display, &view);
        assert!(prepared.face_ids.iter().all(|id| {
            crate::gpu_pick::resolve_face_id(*id).as_deref() == Some("body:2:face:0")
        }));
    }

    /// A body drawn by the object renderer must write face ids with the mesh's
    /// own face strings and the drawn instance/preview transform. When plan
    /// chunks are available they, not the object mesh, supply the id.
    #[test]
    fn object_mesh_face_ids_pick_the_previewed_face_and_plans_take_over() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let size = [128.0f32, 128.0];
        let view = look_at_rh([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let vp = gpu_view_proj(multiply_matrix(perspective_rh(0.8, 1.0, 0.1, 100.0), view), false);
        let object_face = 42u32;
        let object_pick = crate::gpu_pick::face_pick_id(0, object_face, false);
        let display = face_box_display(vec![format!("face:{object_face}")], vec![0; 12]);
        let prepared = prepare_object(
            &display,
            &ViewSettings { face_colours: true, ..ViewSettings::default() },
        );
        assert_eq!(
            prepared.face_ids.len(),
            prepared.vertices.len(),
            "the upload must carry one face id per vertex",
        );
        let mut object = ObjectRenderer::install(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
        );
        object.set_mesh(prepared.vertices, prepared.face_ids, prepared.indices);
        let object_uniforms = ObjectUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: identity_matrix(),
        };
        object.prepare(&device, &queue, &object_uniforms);
        let frame = crate::gpu_pick::PickFrame {
            request: None,
            ghost_key: 41,
            segments: Arc::from(Vec::new()),
            ghost_edges: false,
            face_matrix: identity_matrix(),
            ghost_matrix: identity_matrix(),
            size_px: size,
            edge_width_px: 1.0,
            show_seam_edges: false,
        };
        let uniforms = crate::gpu_pick::PickUniforms {
            view_proj: vp,
            back: [0.0; 4],
            eye: [0.0; 4],
            params: [0.0; 4],
            extra_model: identity_matrix(),
        };
        let mut picker = crate::gpu_pick::PickRenderer::install(
            &device,
            wgpu::TextureFormat::Depth32Float,
            false,
        );

        // Plans are off (face-colour mode / empty plan set): the object mesh
        // supplies the id.
        let picked = pick_faces_at(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            None,
            Some(&object),
            false,
            1,
            [0.0, 0.0, 0.5],
        );
        assert_eq!(picked, object_pick);
        assert_eq!(crate::gpu_pick::resolve_face_id(picked).as_deref(), Some("face:42"),);
        // Off the body: the target clears to zero, so there is no id.
        let miss = pick_faces_at(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            None,
            Some(&object),
            false,
            2,
            [1.2, 1.2, 0.5],
        );
        assert_eq!(miss, 0);

        // The object renderer's draw instance carries the preview matrix; the
        // pick pass reads that same instance.
        let mut preview = identity_matrix();
        preview[3] = [0.2, 0.0, 0.0, 1.0];
        object.set_transform(Some(preview), 1.0);
        object.prepare(&device, &queue, &object_uniforms);
        let picked = pick_faces_at(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            None,
            Some(&object),
            false,
            3,
            [0.2, 0.0, 0.5],
        );
        assert_eq!(picked, object_pick, "pick must follow the drawn preview instance");

        // Plans available: the plan chunks supply the id, even though the
        // object mesh is still uploaded underneath.
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        plan.set_plans(
            vec![nodecraft_display::PlanInstance {
                target: "test".into(),
                plan: box_plan_sized(1.0),
                face_colors: Vec::new(),
                body: 0,
                placement: nodecraft_display::Placement::identity(),
                edge_ids: std::sync::Arc::from(Vec::new()),
                edge_classes: std::sync::Arc::from(Vec::new()),
            }],
            true,
            None,
            1,
        );
        plan.prepare_chunks(&device, &queue);
        assert!(plan.is_available(), "the plan set must be drawable");
        let plan_id = pick_faces_at(
            &mut picker,
            &device,
            &queue,
            &uniforms,
            &frame,
            Some(&plan),
            Some(&object),
            true,
            4,
            [0.0, 0.0, 0.5],
        );
        assert_ne!(plan_id, 0, "the plan box must be under the cursor");
        assert_ne!(plan_id, object_pick, "plan chunks must own the face id when active");
        assert!(crate::gpu_pick::resolve_face_id(plan_id).is_some());
    }

    #[test]
    fn depth_model_matches_solidcraft() {
        use wgpu::CompareFunction::{Always, Greater, GreaterEqual, LessEqual};

        let face = opaque_face_depth(false);
        assert!(face.write, "opaque faces write depth");
        assert_eq!(face.compare, LessEqual);
        assert_eq!(face.bias.constant, 8, "SolidCraft FACE_STEPS");
        assert_eq!(face.bias.slope_scale, 1.0, "SolidCraft FACE_SLOPE");

        let reversed = opaque_face_depth(true);
        assert_eq!(reversed.compare, GreaterEqual);
        assert_eq!(reversed.bias.constant, -8, "reversed depth keeps the push away from the eye",);
        assert_eq!(reversed.bias.slope_scale, -1.0);

        let translucent = translucent_face_depth(false);
        assert!(!translucent.write, "translucent faces must not hide edges behind them");
        assert_eq!(translucent.compare, LessEqual);
        assert_eq!(translucent.bias, wgpu::DepthBiasState::default());

        let edge = edge_depth(false);
        assert!(!edge.write, "edges never write depth");
        assert_eq!(edge.compare, LessEqual, "edges are tested against the shaded model");
        assert_eq!(
            edge.bias,
            wgpu::DepthBiasState::default(),
            "edges are not pulled toward the eye; the face is pushed back instead",
        );

        let highlight = edge_highlight_depth(false);
        assert!(!highlight.write);
        assert_eq!(highlight.compare, LessEqual);
        assert_eq!(highlight.bias.constant, -8, "highlights move toward the eye");
        assert_eq!(edge_highlight_depth(true).bias.constant, 8);

        let ghost = ghost_face_depth(false);
        assert!(!ghost.write, "the ghost overlay writes no depth");
        assert_eq!(ghost.compare, Always, "the ghost overlay never z-fights the body");
        assert_eq!(GHOST_BIAS.constant, 16, "twice FACE_STEPS");
        assert_eq!(GHOST_BIAS.slope_scale, 2.0, "twice FACE_SLOPE");

        assert_eq!(depth_compare(false, Greater), Greater);
        assert_eq!(depth_compare(true, Greater), wgpu::CompareFunction::Less);
    }

    /// A one-pixel ribbon that falls between two rows of pixel centres covers
    /// none of them and the edge disappears for that segment. The shared ribbon
    /// now draws a narrow edge at least 1.3 px wide, so a curved edge covers
    /// every segment; SolidCraft's standard edge uses the same floor.
    #[test]
    fn a_one_pixel_edge_covers_a_curved_polyline() {
        use crate::edge_ribbon::{
            EdgeCpuBatch, EdgeRenderer, EdgeStyle, EdgeUniforms, GpuEdgeBatch,
        };
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let size = [128u32, 128u32];
        let vp = identity_matrix();
        let mut renderer = EdgeRenderer::install(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            true,
        );
        let black = [0.0, 0.0, 0.0, 1.0];
        renderer.write_uniforms(
            &queue,
            EdgeUniforms::new(vp, identity_matrix(), [128.0, 128.0], 1.0, black, true),
            EdgeUniforms::new(vp, identity_matrix(), [128.0, 128.0], 2.0, black, false),
            EdgeUniforms::new(vp, identity_matrix(), [128.0, 128.0], 3.0, black, true),
            EdgeUniforms::new(vp, identity_matrix(), [128.0, 128.0], 2.5, black, true),
        );
        // A closed 24-segment circle: every screen angle and phase appears.
        let points: Vec<solidcore_record::Vec3> = (0..=24)
            .map(|step| {
                let angle = std::f64::consts::TAU * f64::from(step) / 24.0;
                solidcore_record::Vec3::new(0.75 * angle.cos(), 0.75 * angle.sin(), 0.0)
            })
            .collect();
        let edge = crate::adapter::DisplayEdge {
            id: "circle".into(),
            points: points.clone(),
            midpoint: [0.0; 3],
            class: nodecraft_layers::PlanEdgeClass::NORMAL,
        };
        let cpu = EdgeCpuBatch::from_display_edges(&[edge]);
        let Some(batch) = GpuEdgeBatch::upload(&mut renderer, &device, &queue, &cpu) else {
            return;
        };
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("edge coverage colour"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let colour_view = colour.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("edge coverage depth"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("edge coverage"),
        });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("edge coverage pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.9, g: 0.9, b: 0.9, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            renderer.paint_world(&mut pass, &batch, EdgeStyle::Standard);
        }
        let row = size[0] * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge coverage readback"),
            size: (row * size[1]) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &colour,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.recv().expect("map callback").expect("map");
        let pixels = slice.get_mapped_range().expect("map range").to_vec();
        readback.unmap();
        let at = |x: f32, y: f32| -> [u8; 4] {
            let x = x.round().clamp(0.0, size[0] as f32 - 1.0) as usize;
            let y = y.round().clamp(0.0, size[1] as f32 - 1.0) as usize;
            let i = y * row as usize + x * 4;
            [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
        };
        let mut total = 0usize;
        let mut covered = 0usize;
        for pair in points.windows(2) {
            let a = [pair[0].x as f32, pair[0].y as f32];
            let b = [pair[1].x as f32, pair[1].y as f32];
            for step in 1..=7 {
                let t = step as f32 / 8.0;
                let px = project_px(
                    vp,
                    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0],
                    [size[0] as f32, size[1] as f32],
                );
                let mut dark = false;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let c = at(px[0] + dx as f32, px[1] + dy as f32);
                        if c[0] < 70 && c[1] < 70 && c[2] < 70 {
                            dark = true;
                        }
                    }
                }
                total += 1;
                if dark {
                    covered += 1;
                }
            }
        }
        let coverage = covered as f64 / total as f64;
        assert!(coverage >= 0.95, "curved 1 px edge coverage {coverage:.3} ({covered}/{total})");
    }

    #[test]
    fn body_edges_no_longer_use_the_scene_lifted_line_path() {
        let source = include_str!("gpu_viewport.rs");
        let lifted = ["scene", ".lifted", "_line"].concat();
        let ghost_scene = ["ghost", "_edge", "_scene"].concat();
        let ghost_width = ["GHOST", "_EDGE", "_WIDTH"].concat();
        assert!(
            !source.contains(&lifted),
            "standard B-rep edges must go through the GPU ribbon, not scene lines",
        );
        assert!(
            !source.contains(&ghost_scene),
            "ghost edges must not build a second GpuScene layer",
        );
        assert!(
            !source.contains(&ghost_width),
            "ghost width is a ratio of the one edge-width setting",
        );
    }

    /// The cone in the seam report: one primitive cone with the contract's
    /// defaults, planned through the same preview path the viewport uses, with
    /// the edge classes that path derives.
    fn seam_cone_display() -> nodecraft_layers::TessDisplay {
        use nodecraft_solidcore::solidcore::{math::Point3, primitive::api_primitives as api};
        let mut kernel = nodecraft_kernel_profile::kernel();
        let mut brep = api::create_cone(&mut kernel, Point3::new(0.0, 0.0, -20.0), 20.0, 0.0, 40.0)
            .expect("cone");
        nodecraft_layers::tessellate_kernel_preview(&mut kernel, &mut brep, 2)
            .expect("cone preview plan")
    }

    /// The report's orthographic camera: yaw -90, pitch 0, up Z, looking along
    /// -x from (103.923, 11.259, -20.709), with a sub-pixel target offset. The
    /// report fixes the camera but not which pixel centre the seam lands on;
    /// the tests sweep that phase so a phase-dependent rasterisation defect
    /// cannot hide. The target is 1664x1640 physical, the screenshot's
    /// viewport at two device pixels per logical pixel.
    fn seam_camera_phase(aspect: f64, target_y_offset: f64) -> [[f32; 4]; 4] {
        let camera = solidcraft_viewport::Camera {
            target: solidcore_record::Vec3::new(0.0, 11.259 + target_y_offset, -20.709),
            yaw: -std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
            roll: 0.0,
            distance: 103.923,
            fov: 0.0,
            up: solidcraft_viewport::UpAxis::Z,
        };
        let mut vp = identity_matrix();
        for (column, values) in camera.view_proj(aspect, 28.284_271).0.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                vp[column][row] = *value as f32;
            }
        }
        gpu_view_proj(vp, false)
    }

    fn read_scene_colour(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        colour: &wgpu::Texture,
        size: [u32; 2],
        row: u32,
    ) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cone seam readback"),
            size: (row * size[1]) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cone seam readback"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: colour,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size[1]),
                },
            },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.recv().expect("map callback").expect("map");
        let pixels = slice.get_mapped_range().expect("map range").to_vec();
        readback.unmap();
        pixels
    }

    /// Draw plans and their edges exactly as the viewport's callback does:
    /// faces first, then the one ribbon pass.
    fn render_seam_scene(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plans: Vec<nodecraft_display::PlanInstance>,
        size: [u32; 2],
        edge_width_px: f32,
        translucent: bool,
        reversed: bool,
        phase: f64,
        show_seams: bool,
    ) -> (Vec<u8>, [[f32; 4]; 4]) {
        let tess = solidcraft_viewport_gpu::TessGpu::new("test", device.clone(), queue.clone())
            .expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut plan = crate::gpu_plan::PlanState::new(
            device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            reversed,
            Some(ready),
        );
        let aspect = f64::from(size[0]) / f64::from(size[1]);
        let normal = seam_camera_phase(aspect, phase);
        let vp = if reversed { gpu_view_proj(normal, true) } else { normal };
        let uniforms = ObjectUniforms {
            view_proj: vp,
            back: [1.0, 0.0, 0.0, 1.0],
            eye: [103.923, 11.259, -20.709, 1.0],
            params: [1.0, 0.0, 0.45, 0.0],
            extra_model: identity_matrix(),
        };
        plan.set_plans(plans, true, None, 1);
        plan.prepare_chunks(device, queue);
        plan.set_edge_width(edge_width_px);
        plan.set_show_seam_edges(show_seams);
        plan.set_translucent(translucent);
        plan.write_uniforms(queue, &uniforms, None, [size[0] as f32, size[1] as f32]);
        let colour = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cone seam colour"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let colour_view = colour.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cone seam depth"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cone seam scene"),
        });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cone seam pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.9, g: 0.9, b: 0.9, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if reversed { 0.0 } else { 1.0 }),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            plan.paint(&mut pass);
            plan.paint_edges(&mut pass, true);
        }
        queue.submit([encoder.finish()]);
        let row = size[0] * 4;
        let pixels = read_scene_colour(device, queue, &colour, size, row);
        (pixels, vp)
    }

    fn seam_plan_instance(
        plan: std::sync::Arc<nodecraft_layers::tess::TessPlan>,
        edge_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]>,
        body: u64,
        placement: nodecraft_display::Placement,
    ) -> nodecraft_display::PlanInstance {
        nodecraft_display::PlanInstance {
            target: "seam".into(),
            plan,
            face_colors: Vec::new(),
            body,
            placement,
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes,
        }
    }

    /// Every pixel of the ribbon's cross-section at this point is dark. A
    /// single dark neighbour is not enough: a gap on one side of a four-pixel
    /// band must still count as a gap.
    fn seam_is_dark(pixels: &[u8], size: [u32; 2], vp: [[f32; 4]; 4], point: [f32; 3]) -> bool {
        let px = project_px(vp, point, [size[0] as f32, size[1] as f32]);
        for dx in -1..=1 {
            for dy in -1..=1 {
                let x = (px[0] + dx as f32).round().clamp(0.0, size[0] as f32 - 1.0) as usize;
                let y = (px[1] + dy as f32).round().clamp(0.0, size[1] as f32 - 1.0) as usize;
                let at = y * size[0] as usize * 4 + x * 4;
                if !(pixels[at] < 70 && pixels[at + 1] < 70 && pixels[at + 2] < 70) {
                    return false;
                }
            }
        }
        true
    }

    /// The seam is the cone's straight apex-to-base edge. At a wide edge width
    /// the old segment quads carried each endpoint's depth half a width past
    /// the vertex, which bent the depth ramp behind the face for the first
    /// pixels after every ring row: the seam read as a dashed line. The ribbon
    /// now extrapolates the depth along the segment, so every sample between
    /// apex and base draws over the face.
    #[test]
    fn a_cone_seam_covers_every_ring_row_at_a_wide_edge_width() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let display = seam_cone_display();
        let plan = std::sync::Arc::clone(&display.plan);
        let classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(display.edge_classes.clone());
        let seam_index = display
            .edge_classes
            .iter()
            .position(|class| class.is_seam())
            .expect("the cone has a seam");
        let seam = plan.edges[seam_index].clone();
        assert!(seam.len() > 2, "the cone seam is a polyline through ring rows");
        let size = [1664u32, 1640u32];
        for reversed in [false, true] {
            for phase in [-0.4f64, 0.0, 0.4] {
                let (pixels, vp) = render_seam_scene(
                    &device,
                    &queue,
                    vec![seam_plan_instance(
                        std::sync::Arc::clone(&plan),
                        std::sync::Arc::clone(&classes),
                        0,
                        nodecraft_display::Placement::identity(),
                    )],
                    size,
                    4.0,
                    false,
                    reversed,
                    phase,
                    true,
                );
                let mut total = 0usize;
                let mut covered = 0usize;
                let mut row_gaps = Vec::new();
                for (index, pair) in seam.windows(2).enumerate() {
                    for step in 0..=8 {
                        let t = step as f64 / 8.0;
                        let point = [
                            (pair[0].x + (pair[1].x - pair[0].x) * t) as f32,
                            (pair[0].y + (pair[1].y - pair[0].y) * t) as f32,
                            (pair[0].z + (pair[1].z - pair[0].z) * t) as f32,
                        ];
                        total += 1;
                        if seam_is_dark(&pixels, size, vp, point) {
                            covered += 1;
                        } else if step == 0 {
                            row_gaps.push(index);
                        }
                    }
                }
                let coverage = covered as f64 / total as f64;
                assert!(
                    row_gaps.is_empty(),
                    "reversed={reversed}: the seam is missing at ring rows {row_gaps:?}",
                );
                assert!(
                    coverage >= 0.98,
                    "reversed={reversed} phase={phase}: cone seam coverage {coverage:.3} ({covered}/{total})",
                );
            }
        }
    }

    /// The default setting hides the wall seam but keeps the base circle: both
    /// are ordinary plan edges, classified once and toggled by the uniform mask.
    #[test]
    fn the_default_setting_hides_only_the_cone_seam() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let display = seam_cone_display();
        let plan = std::sync::Arc::clone(&display.plan);
        let classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(display.edge_classes.clone());
        let seam_index = display
            .edge_classes
            .iter()
            .position(|class| class.is_seam())
            .expect("the cone has a seam");
        let base_index = (0..plan.edges.len())
            .find(|index| !display.edge_classes[*index].is_seam())
            .expect("the cone has an end circle");
        let seam_point = plan.edges[seam_index][plan.edges[seam_index].len() / 2];
        let base_point = plan.edges[base_index][plan.edges[base_index].len() / 2];
        let seam_point = [seam_point.x as f32, seam_point.y as f32, seam_point.z as f32];
        let base_point = [base_point.x as f32, base_point.y as f32, base_point.z as f32];
        let size = [832u32, 820u32];
        let instance = || {
            vec![seam_plan_instance(
                std::sync::Arc::clone(&plan),
                std::sync::Arc::clone(&classes),
                0,
                nodecraft_display::Placement::identity(),
            )]
        };
        let (hidden, vp) =
            render_seam_scene(&device, &queue, instance(), size, 4.0, false, false, 0.0, false);
        assert!(
            !seam_is_dark(&hidden, size, vp, seam_point),
            "the default setting must not draw the cone's seam",
        );
        assert!(
            seam_is_dark(&hidden, size, vp, base_point),
            "the cone's base circle is not a seam and stays drawn",
        );
        let (shown, vp) =
            render_seam_scene(&device, &queue, instance(), size, 4.0, false, false, 0.0, true);
        assert!(
            seam_is_dark(&shown, size, vp, seam_point),
            "the setting draws the same edge without re-uploading it",
        );
    }

    /// The base circle is seen edge-on in the same view. Its vertices sit on
    /// the same shared-node depth path, so every sample over the full circle
    /// must draw, with no break at a polyline vertex.
    #[test]
    fn a_cone_base_circle_covers_every_ring_row_at_a_wide_edge_width() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let display = seam_cone_display();
        let plan = std::sync::Arc::clone(&display.plan);
        let classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(display.edge_classes.clone());
        let base_index = (0..plan.edges.len())
            .find(|index| !display.edge_classes[*index].is_seam())
            .expect("the cone has an end circle");
        let base = plan.edges[base_index].clone();
        assert!(base.len() > 2, "the base circle is a closed polyline");
        let size = [1664u32, 1640u32];
        let (pixels, vp) = render_seam_scene(
            &device,
            &queue,
            vec![seam_plan_instance(
                std::sync::Arc::clone(&plan),
                classes,
                0,
                nodecraft_display::Placement::identity(),
            )],
            size,
            4.0,
            false,
            false,
            0.0,
            true,
        );
        let mut total = 0usize;
        let mut covered = 0usize;
        for pair in base.windows(2) {
            for step in 0..=8 {
                let t = step as f64 / 8.0;
                let point = [
                    (pair[0].x + (pair[1].x - pair[0].x) * t) as f32,
                    (pair[0].y + (pair[1].y - pair[0].y) * t) as f32,
                    (pair[0].z + (pair[1].z - pair[0].z) * t) as f32,
                ];
                total += 1;
                if seam_is_dark(&pixels, size, vp, point) {
                    covered += 1;
                }
            }
        }
        let coverage = covered as f64 / total as f64;
        assert!(coverage >= 0.98, "cone base circle coverage {coverage:.3} ({covered}/{total})");
    }

    /// An opaque body between the camera and the seam still hides it. The
    /// ribbon only changes its screen footprint; it must not disable depth.
    #[test]
    fn an_edge_behind_an_opaque_body_stays_hidden() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let cone_display = seam_cone_display();
        let cone = std::sync::Arc::clone(&cone_display.plan);
        let cone_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(cone_display.edge_classes.clone());
        let box_plan = box_plan_sized(20.0);
        let box_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(vec![
                nodecraft_layers::PlanEdgeClass::NORMAL;
                box_plan.edges.len()
            ]);
        let box_placement =
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [40.0, 0.0, 0.0],
                rotation: [0.0; 3],
            });
        let size = [1664u32, 1640u32];
        let (pixels, vp) = render_seam_scene(
            &device,
            &queue,
            vec![
                seam_plan_instance(
                    std::sync::Arc::clone(&cone),
                    cone_classes,
                    0,
                    nodecraft_display::Placement::identity(),
                ),
                seam_plan_instance(box_plan, box_classes, 1, box_placement),
            ],
            size,
            4.0,
            false,
            false,
            0.0,
            true,
        );
        // (20, 0, 0) is on the seam and behind the box's near face at x = 30.
        assert!(
            !seam_is_dark(&pixels, size, vp, [20.0, 0.0, 0.0]),
            "an edge behind an opaque body must be hidden by that body's depth",
        );
    }

    /// A translucent body writes no depth, so the edge behind it stays
    /// visible. This is the viewport's existing translucent contract.
    #[test]
    fn a_translucent_body_lets_the_edge_behind_it_show() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let cone_display = seam_cone_display();
        let cone = std::sync::Arc::clone(&cone_display.plan);
        let cone_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(cone_display.edge_classes.clone());
        let box_plan = box_plan_sized(20.0);
        let box_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]> =
            std::sync::Arc::from(vec![
                nodecraft_layers::PlanEdgeClass::NORMAL;
                box_plan.edges.len()
            ]);
        let box_placement =
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [40.0, 0.0, 0.0],
                rotation: [0.0; 3],
            });
        let size = [1664u32, 1640u32];
        let (pixels, vp) = render_seam_scene(
            &device,
            &queue,
            vec![
                seam_plan_instance(
                    std::sync::Arc::clone(&cone),
                    cone_classes,
                    0,
                    nodecraft_display::Placement::identity(),
                ),
                seam_plan_instance(box_plan, box_classes, 1, box_placement),
            ],
            size,
            4.0,
            true,
            false,
            0.0,
            true,
        );
        assert!(
            seam_is_dark(&pixels, size, vp, [20.0, 0.0, 0.0]),
            "a translucent body must not hide the edge behind it",
        );
    }
}
