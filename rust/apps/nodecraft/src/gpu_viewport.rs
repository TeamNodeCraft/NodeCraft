//! wgpu viewport glue around SolidCraft's GPU renderer.
//!
//! The renderer, scene format and pipelines belong to `solidcraft-viewport-gpu`; this module only
//! converts a NodeCraft display mesh into a `GpuScene`, owns the `SceneSlot`, and implements
//! the `egui_wgpu` paint callback.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use egui_wgpu::{wgpu, CallbackTrait, RenderState};
use solidcraft_geom::{Aabb3, Mesh as CraftMesh};
use solidcore_record::Vec3;

use crate::viewport::{PreviewScope, ViewSettings};
use solidcraft_viewport_gpu::depth_format;
use solidcraft_viewport_gpu::view::{gpu_view_proj, reversed_depth};
use solidcraft_viewport_gpu::view::{Frame, GpuScene, GpuTarget, MovingLayer, Palette, PaletteIndex, Renderer, SceneSlot};
use solidcraft_viewport::surface::OpaqueSurface;
use solidcraft_viewport::Camera;

use crate::adapter::DisplayMesh;
use crate::viewport_grid::grid_step;
use crate::gpu_backdrop::BackdropRenderer;
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
fn scene_color_attachment<'a, T>(samples: u32, color: &'a T, resolve: &'a T) -> (&'a T, Option<&'a T>) {
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
            return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
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

/// CPU-side object geometry prepared away from the UI thread.
pub(crate) struct PreparedObject {
    pub(crate) vertices: Vec<ObjectVertex>,
    pub(crate) indices: Vec<u32>,
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
}

pub(crate) const OBJECT_SHADER: &str = r#"
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

fn face_colour(face: u32) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 12] = [
        [0.78, 0.25, 0.25],
        [0.25, 0.50, 0.82],
        [0.30, 0.65, 0.38],
        [0.86, 0.58, 0.18],
        [0.55, 0.30, 0.72],
        [0.20, 0.65, 0.65],
        [0.82, 0.40, 0.55],
        [0.40, 0.55, 0.75],
        [0.65, 0.55, 0.25],
        [0.50, 0.40, 0.70],
        [0.25, 0.55, 0.55],
        [0.70, 0.45, 0.35],
    ];
    PALETTE[(face as usize) % PALETTE.len()]
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
    indices: Vec<u32>,
) -> Vec<(Vec<ObjectVertex>, Vec<u32>)> {
    const MAX_CHUNK_VERTICES: usize = 5_000_000;
    chunk_object_mesh_with_limit(vertices, indices, MAX_CHUNK_VERTICES)
}

fn chunk_object_mesh_with_limit(
    vertices: Vec<ObjectVertex>,
    indices: Vec<u32>,
    max_vertices: usize,
) -> Vec<(Vec<ObjectVertex>, Vec<u32>)> {
    let mut chunks: Vec<(Vec<ObjectVertex>, Vec<u32>)> = Vec::new();
    let mut chunk_vertices: Vec<ObjectVertex> = Vec::new();
    let mut chunk_indices: Vec<u32> = Vec::new();
    let mut remap: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for triangle in indices.chunks(3) {
        if chunk_vertices.len() + triangle.len() > max_vertices && !chunk_indices.is_empty() {
            chunks.push((std::mem::take(&mut chunk_vertices), std::mem::take(&mut chunk_indices)));
            remap.clear();
        }
        for &source in triangle {
            let Some(vertex) = vertices.get(source as usize) else { continue };
            let local = *remap.entry(source).or_insert_with(|| {
                chunk_vertices.push(*vertex);
                (chunk_vertices.len() - 1) as u32
            });
            chunk_indices.push(local);
        }
    }
    if !chunk_indices.is_empty() {
        chunks.push((chunk_vertices, chunk_indices));
    }
    chunks
}

pub(crate) fn prepare_object(display: &DisplayMesh, view: &ViewSettings) -> PreparedObject {
    let mesh: &CraftMesh = &display.mesh;
    // The app draws the solid material itself (see ObjectRenderer) so it can use
    // the Swift viewer's light rig and tone mapping. The SolidCraft scene keeps
    // only the edges; the blurred backdrop keeps a PBR copy for defocused colour.
    let ignore_look = view.ignore_look_colors;
    let uniform = if view.face_colours {
        None
    } else if ignore_look || display.face_colors.is_empty() {
        Some([0.17, 0.18, 0.195])
    } else {
        let first = display.face_colors[0];
        display.face_colors.iter().all(|colour| *colour == first).then_some(first)
    };
    if let Some((vertices, indices)) = uniform.and_then(|colour| smoothed_object_mesh(mesh, colour)) {
        return PreparedObject { vertices, indices };
    }
    let mut vertices = Vec::new();
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
            face_colour(mesh.tri_face.get(triangle_index).copied().unwrap_or(0))
        } else if ignore_look {
            [0.17, 0.18, 0.195]
        } else {
            display
                .face_colors
                .get(triangle_index)
                .copied()
                .unwrap_or([0.17, 0.18, 0.195])
        };
        for (index, point) in [(a, pa), (b, pb), (c, pc)] {
            let normal = mesh.normals.get(index).copied().filter(|normal| normal.len() > 1.0e-9).unwrap_or(fallback);
            indices.push(vertices.len() as u32);
            vertices.push(ObjectVertex { position: point.to_f32(), normal: normal.to_f32(), colour });
        }
    }
    PreparedObject { vertices, indices }
}

struct GpuMeshAsset {
    key: u64,
    vertices: wgpu::Buffer,
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
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 112, shader_location: 10 },
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
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
        ],
    }
}

/// Indexed object geometry for a display mesh with one colour.
///
/// Returns `None` when the mesh has unusable normals; the caller then falls back
/// to the per-triangle expansion that can synthesise a face normal.
/// Non-indexed fallback for meshes without reliable per-vertex normals.
fn flat_object_mesh(mesh: &CraftMesh, colour: [f32; 3]) -> Option<(Vec<ObjectVertex>, Vec<u32>)> {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for triangle in &mesh.triangles {
        let [a, b, c] = triangle.map(|index| index as usize);
        let (Some(pa), Some(pb), Some(pc)) = (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c)) else { continue };
        let fallback = (*pb - *pa).cross(*pc - *pa);
        let normal = fallback.normalized().unwrap_or(solidcore_record::Vec3::Z).to_f32();
        for point in [pa, pb, pc] {
            indices.push(vertices.len() as u32);
            vertices.push(ObjectVertex { position: point.to_f32(), normal, colour });
        }
    }
    Some((vertices, indices))
}

/// Indexed object geometry with crease-aware smooth normals.
///
/// Tessellators often provide faceted normals; zebra analysis then exposes every
/// triangle boundary. Average incident face normals inside a crease angle so
/// smooth faces shade continuously, while splitting vertices at genuine creases.
fn smoothed_object_mesh(mesh: &CraftMesh, colour: [f32; 3]) -> Option<(Vec<ObjectVertex>, Vec<u32>)> {
    // Tessellated smooth surfaces can have coarse facets; treat anything under
    // 60° as smooth. Real CAD creases (box edges, rims) are near 90° and still
    // split, so zebra shows genuine discontinuities rather than tessellation
    // seams.
    const CREASE_COS: f64 = 0.5; // 60 degrees
    let key = |p: &Vec3| {
        (
            (p.x * 1.0e3).round() as i64,
            (p.y * 1.0e3).round() as i64,
            (p.z * 1.0e3).round() as i64,
        )
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
        let (Some(pa), Some(pb), Some(pc)) = (
            mesh.positions.get(a),
            mesh.positions.get(b),
            mesh.positions.get(c),
        ) else {
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
    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    let mut corners: std::collections::HashMap<((i64, i64, i64), u32, [i64; 3]), u32> =
        std::collections::HashMap::new();
    for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
        let Some(face) = face_normals.get(triangle_index).copied().flatten() else { continue };
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
                index
            });
            indices.push(vertex_index);
        }
    }
    Some((vertices, indices))
}

#[cfg(test)]
fn indexed_object_mesh(mesh: &CraftMesh, colour: [f32; 3]) -> Option<(Vec<ObjectVertex>, Vec<u32>)> {
    if mesh.normals.len() < mesh.positions.len() {
        return None;
    }
    let mut vertices = Vec::with_capacity(mesh.positions.len());
    for (point, normal) in mesh.positions.iter().zip(&mesh.normals) {
        if !(normal.len() > 1.0e-9) {
            return None;
        }
        vertices.push(ObjectVertex {
            position: point.to_f32(),
            normal: normal.to_f32(),
            colour,
        });
    }
    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    for triangle in &mesh.triangles {
        for index in triangle {
            if *index as usize >= vertices.len() {
                return None;
            }
            indices.push(*index);
        }
    }
    Some((vertices, indices))
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
    pending: Option<Vec<(u64, Vec<ObjectVertex>, Vec<u32>)>>,
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
            source: wgpu::ShaderSource::Wgsl(OBJECT_SHADER.into()),
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
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
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
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ghost_uniforms.as_entire_binding() }],
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
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                depth_compare: Some(if reversed {
                    wgpu::CompareFunction::GreaterEqual
                } else {
                    wgpu::CompareFunction::LessEqual
                }),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                // The ghost is an overlay layer: never z-fight with the opaque
                // breakpoint body underneath it.
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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

    fn set_mesh(&mut self, vertices: Vec<ObjectVertex>, indices: Vec<u32>) {
        // A 10k-tree scene can exceed wgpu's 256 MiB buffer limit in one vertex
        // buffer. Split the mesh into bounded chunks; each chunk gets its own
        // vertex/index buffers and draw range, then the instance buffer keeps
        // them in one logical mesh.
        let uploaded_triangles = indices.len() as u64 / 3;
        let uploaded_object_bytes = (vertices.len() * std::mem::size_of::<ObjectVertex>()
            + indices.len() * std::mem::size_of::<u32>()) as u64;
        let chunks = chunk_object_mesh(vertices, indices);
        self.chunk_spans.clear();
        self.draws.clear();
        self.ghost_alpha = 1.0;
        let mut pending = Vec::with_capacity(chunks.len());
        for (chunk_vertices, chunk_indices) in chunks {
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
            pending.push((key, chunk_vertices, chunk_indices));
        }
        self.pending = Some(pending);
        self.instances_dirty = true;
        self.uploaded_triangles = uploaded_triangles;
        self.uploaded_object_bytes = uploaded_object_bytes;
    }

    /// Displace the existing draw list without touching the uploaded mesh. The
    /// instance buffer is re-uploaded in `prepare`; vertex/index buffers stay.
    fn set_transform(
        &mut self,
        transform: Option<[[f32; 4]; 4]>,
        affected: Option<&[(u32, u32)]>,
        ghost_alpha: f32,
    ) {
        self.ghost_alpha = ghost_alpha;
        if self.chunk_spans.is_empty() {
            return;
        }
        let chunks = self.chunk_spans.clone();
        self.draws = match (transform, affected) {
            (Some(matrix), Some(affected)) => {
                let total = chunks.iter().map(|(_, triangles)| *triangles).sum::<u32>();
                let (moved, stationary) = preview_draw_ranges(affected, total);
                let mut draws = Vec::new();
                let mut first_triangle = 0;
                for (asset_key, triangles) in chunks {
                    let last_triangle = first_triangle + triangles;
                    for (ranges, matrix, _alpha, ghost) in [
                        (&moved, matrix, ghost_alpha, true),
                        (&stationary, identity_matrix(), 1.0, false),
                    ] {
                        for &(start, end) in ranges {
                            let start = start.max(first_triangle);
                            let end = end.min(last_triangle);
                            if end <= start {
                                continue;
                            }
                            draws.push(GpuDrawItem {
                                asset_key,
                                transform: matrix,
                                index_start: (start - first_triangle) * 3,
                                index_count: (end - start) * 3,
                                ghost,
                            });
                        }
                    }
                    first_triangle = last_triangle;
                }
                draws
            }
            (transform, _) => chunks
                .into_iter()
                .map(|(asset_key, triangles)| GpuDrawItem {
                    asset_key,
                    transform: transform.unwrap_or_else(identity_matrix),
                    index_start: 0,
                    index_count: triangles * 3,
                    ghost: false,
                })
                .collect(),
        };
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
            for (key, vertices, indices) in chunks {
                if vertices.is_empty() || indices.is_empty() {
                    continue;
                }
                let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-vertices"),
                    size: (vertices.len() * std::mem::size_of::<ObjectVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
                let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-object-indices"),
                    size: (indices.len() * std::mem::size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&indices));
                self.assets.push(GpuMeshAsset {
                    key,
                    vertices: vertex_buffer,
                    indices: index_buffer,
                });
            }
        }
        if self.instances_dirty {
            self.instances_dirty = false;
            self.instances = if self.draws.is_empty() {
                None
            } else {
                let instances: Vec<GpuInstance> = self.draws.iter().map(GpuDrawItem::instance).collect();
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
            let Some(asset) = self.assets.iter().find(|asset| asset.key == draw.asset_key) else { return };
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
}

pub(crate) fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
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
    let determinant = m[0][0] * cofactor[0][0] + m[0][1] * cofactor[0][1] + m[0][2] * cofactor[0][2];
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

/// `matrix * direction` for a column-major matrix, ignoring translation.
fn transform_direction(matrix: [[f32; 4]; 4], direction: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * direction[0] + matrix[1][0] * direction[1] + matrix[2][0] * direction[2],
        matrix[0][1] * direction[0] + matrix[1][1] * direction[1] + matrix[2][1] * direction[2],
        matrix[0][2] * direction[0] + matrix[1][2] * direction[1] + matrix[2][2] * direction[2],
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
        let point = transform_point(
            matrix,
            [corner[0] as f32, corner[1] as f32, corner[2] as f32],
        );
        for axis in 0..3 {
            let value = f64::from(point[axis]);
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    Aabb3 {
        min: Vec3::new(min[0], min[1], min[2]),
        max: Vec3::new(max[0], max[1], max[2]),
    }
}

/// Smallest box containing both inputs.
fn union_bounds(a: Aabb3, b: Aabb3) -> Aabb3 {
    Aabb3 {
        min: Vec3::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y), a.min.z.min(b.min.z)),
        max: Vec3::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y), a.max.z.max(b.max.z)),
    }
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

/// Split triangle ranges into moved and stationary draw ranges. Affected ranges
/// may arrive unsorted or overlapping; they are clipped to `total` and the
/// complement is emitted for the bodies that stay put.
fn preview_draw_ranges(affected: &[(u32, u32)], total: u32) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let mut clipped: Vec<(u32, u32)> = affected
        .iter()
        .map(|&(start, end)| (start.min(total), end.min(total)))
        .filter(|(start, end)| end > start)
        .collect();
    clipped.sort_unstable();
    let mut moved: Vec<(u32, u32)> = Vec::with_capacity(clipped.len());
    for (start, end) in clipped {
        match moved.last_mut() {
            Some((_, last_end)) if start <= *last_end => *last_end = (*last_end).max(end),
            _ => moved.push((start, end)),
        }
    }
    let mut stationary = Vec::new();
    let mut cursor = 0;
    for (start, end) in &moved {
        if *start > cursor {
            stationary.push((cursor, *start));
        }
        cursor = *end;
    }
    if cursor < total {
        stationary.push((cursor, total));
    }
    (moved, stationary)
}

/// Convert a half-open triangle range into an index-buffer draw range.

#[cfg(test)]
fn draw_index_range((start, end): (u32, u32)) -> (u32, u32) {
    (start * 3, (end - start) * 3)
}

/// Triangle ranges of the display bodies a preview scope moves.
///
/// The object renderer emits one triangle per display triangle in order, so a
/// display triangle index maps directly to its index-buffer range.
fn affected_triangle_ranges(display: &DisplayMesh, scope: &PreviewScope) -> Option<Vec<(u32, u32)>> {
    if matches!(scope, PreviewScope::All) {
        return None;
    }
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for index in 0..display.mesh.triangles.len() {
        let face = display.mesh.tri_face.get(index).copied().unwrap_or(0) as usize;
        let moves = display.face_ids.get(face).is_some_and(|id| scope.is_affected(id));
        if !moves {
            continue;
        }
        let index = index as u32;
        match ranges.last_mut() {
            Some((_, end)) if *end == index => *end = index + 1,
            _ => ranges.push((index, index + 1)),
        }
    }
    Some(ranges)
}

struct GridRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl GridRenderer {
    fn install(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32, depth: wgpu::TextureFormat) -> Self {
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
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
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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
    fn install(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32, depth: wgpu::TextureFormat) -> Self {
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
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
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
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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
    [
        f32::from(rgba[0]) / 255.0,
        f32::from(rgba[1]) / 255.0,
        f32::from(rgba[2]) / 255.0,
        alpha,
    ]
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
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
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
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
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

/// Display-only drag preview state: the world transform plus which display
/// bodies it applies to.
#[derive(Clone, Debug, PartialEq)]
struct PreviewState {
    transform: [[f32; 4]; 4],
    scope: PreviewScope,
    alpha: f32,
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

pub struct GpuViewport {
    adapter: String,
    backend: String,
    device_type: String,
    slot: SceneSlot,
    key: u64,
    ghost_edge_slot: SceneSlot,
    ghost_edge_key: u64,
    ghost_edge_matrix: Option<[[f32; 4]; 4]>,
    ghost_edge_present: bool,
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
    /// Keeps the background plan-evaluation shader compile alive.
    #[cfg(not(target_arch = "wasm32"))]
    _tess_later: Option<solidcraft_viewport_gpu::view::TessLater>,
    scene: Arc<Mutex<Option<SceneTarget>>>,
    view: ViewSettings,
    chrome_visible: bool,
    background: [u8; 4],
    target_format: wgpu::TextureFormat,
    reversed_depth: bool,
    /// Display-only drag preview for gizmo edits. Applied to the object
    /// instances and the scene edges of the affected bodies; the merged mesh
    /// stays uploaded.
    preview: Option<PreviewState>,
    /// Cached triangle ranges for a partial-body preview. The scan runs once per
    /// scope, not once per drag frame.
    preview_ranges: Option<(PreviewScope, Option<Vec<(u32, u32)>>)>,
}

const GHOST_EDGE_WIDTH: f32 = 2.0;

pub(crate) fn ghost_edge_scene(
    edges: &[crate::adapter::DisplayEdge],
    rgba: [u8; 4],
    width: f32,
) -> (GpuScene, usize) {
    let mut scene = GpuScene::with_palette(Palette::with_stable(vec![rgba]));
    let colour = PaletteIndex(0);
    let mut segments = 0;
    for edge in edges {
        for pair in edge.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let d = b - a;
            if d.dot(d) <= 1.0e-20 {
                continue;
            }
            scene.line(a.to_f32(), b.to_f32(), colour, width, true);
            segments += 1;
        }
    }
    (scene, segments)
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
        };
        let window = GpuTarget {
            format: rs.target_format,
            depth: Some(depth_format(&rs.device, true)),
            samples: msaa_samples(),
            tess: false,
            oit: false,
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
            rs.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            multisample: wgpu::MultisampleState { count: msaa_samples(), mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        rs.renderer
            .write()
            .callback_resources
            .insert(ViewportClear {
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
            key: 0,
            ghost_edge_slot: Arc::new(Mutex::new(None)),
            ghost_edge_key: 1 << 60,
            ghost_edge_matrix: None,
            ghost_edge_present: false,
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
            #[cfg(not(target_arch = "wasm32"))]
            _tess_later: tess_later,
            scene,
            view: ViewSettings::default(),
            chrome_visible: true,
            background: crate::theme::get().viewport_background,
            target_format: rs.target_format,
            reversed_depth,
            preview: None,
            preview_ranges: None,
        })
    }

    pub fn plan_ready(&self) -> bool {
        self.plan.lock().ok().is_some_and(|plan| plan.tess_ready())
    }

    pub fn plan_debug(&self) -> Option<crate::gpu_plan::PlanDebug> {
        self.plan.lock().ok().map(|plan| plan.debug_counts())
    }

    pub fn set_plans(
        &mut self,
        plans: Vec<nodecraft_display::PlanInstance>,
        enabled: bool,
        colour: Option<[f32; 3]>,
    ) {
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_plans(plans, enabled, colour);
        }
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
        }
        if rebuild {
            self.set_mesh(display);
        }
    }

    pub fn set_mesh(&mut self, display: Option<&DisplayMesh>) {
        let prepared = display.map(|display| prepare_object(display, &self.view));
        self.set_prepared_mesh(display, prepared);
    }

    /// Apply display geometry whose CPU mesh conversion already ran on a worker.
    pub(crate) fn set_prepared_mesh(
        &mut self,
        display: Option<&DisplayMesh>,
        prepared: Option<PreparedObject>,
    ) {
        // A full display mesh is authoritative geometry: it already sits at the
        // edited placement, so any drag preview is retired here.
        self.preview = None;
        self.preview_ranges = None;
        self.key = self.key.wrapping_add(1);
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
        self.bounds = self.display_bounds;
        let (object_vertices, object_indices) = prepared
            .map(|prepared| (prepared.vertices, prepared.indices))
            .unwrap_or_default();
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_mesh(object_vertices, object_indices);
        }
        let mut scene = self.build_scene(display, None, &PreviewScope::All, true);
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
        let geometry = smoothed_object_mesh(mesh, colour)
            .or_else(|| flat_object_mesh(mesh, colour))
            .unwrap_or_default();
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_mesh(geometry.0, geometry.1);
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

    pub fn set_ghost_edges(&mut self, edges: &[crate::adapter::DisplayEdge]) {
        let theme = crate::theme::get().ghost_edge;
        let (scene, n) = ghost_edge_scene(edges, theme, GHOST_EDGE_WIDTH);
        if let Ok(mut slot) = self.ghost_edge_slot.lock() {
            *slot = (n > 0).then_some(scene);
        }
        self.ghost_edge_present = n > 0;
        self.ghost_edge_key = self.ghost_edge_key.wrapping_add(1);
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[ghost-edges] rebuild {n} segments");
        }
    }

    pub fn set_ghost_edge_matrix(&mut self, matrix: Option<[[f32; 4]; 4]>) {
        self.ghost_edge_matrix = matrix;
    }

    pub fn clear_ghost_edges(&mut self) {
        if let Ok(mut slot) = self.ghost_edge_slot.lock() {
            *slot = None;
        }
        self.ghost_edge_matrix = None;
        self.ghost_edge_present = false;
        self.ghost_edge_key = self.ghost_edge_key.wrapping_add(1);
    }

    pub fn clear_ghost(&mut self) {
        self.clear_ghost_edges();
        self.set_ghost_plans(Vec::new(), 0.45, None, false);
        self.set_ghost_matrix(None);
    }

    /// Preview a transform edit without touching the merged mesh or the scene
    /// geometry.
    ///
    /// A whole-mesh preview is applied by transforming the scene projection in
    /// `frame()`, so a drag frame only updates the object instance matrix and a
    /// 128-byte uniform — no CPU geometry work and no GPU re-upload. Only a
    /// partial-body preview rebuilds the edge scene, and the affected triangle
    /// ranges are cached per scope.
    pub fn set_preview_transform(
        &mut self,
        transform: Option<[[f32; 4]; 4]>,
        display: Option<&DisplayMesh>,
        scope: PreviewScope,
        alpha: f32,
    ) {
        let Some(display) = display else {
            self.preview = transform.map(|transform| PreviewState { transform, scope, alpha });
            if let Ok(mut renderer) = self.object.lock() {
                renderer.set_transform(transform, None, 1.0);
            }
            return;
        };
        // A body scope that happens to cover every triangle can use the
        // whole-mesh path as well.
        let scope = self.scope_for_display(display, scope);
        let next = transform.map(|transform| PreviewState { transform, scope, alpha });
        // The ghost alpha uniform must be current even when the preview state
        // itself is unchanged (e.g. the static overlay was already installed).
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_ghost_alpha(alpha);
        }
        if self.preview == next {
            return;
        }
        let previous = std::mem::replace(&mut self.preview, next);
        let previous_scoped = previous
            .as_ref()
            .is_some_and(|preview| !matches!(preview.scope, PreviewScope::All));
        let scoped = self
            .preview
            .as_ref()
            .is_some_and(|preview| !matches!(preview.scope, PreviewScope::All));
        let preview_scope = self.preview.as_ref().map(|preview| preview.scope.clone());
        let affected = preview_scope
            .as_ref()
            .and_then(|scope| self.affected_ranges(display, scope));
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_transform(
                transform,
                affected.as_deref(),
                self.preview.as_ref().map_or(1.0, |preview| preview.alpha),
            );
        }
        let base_bounds = self.display_bounds;
        self.bounds = match &self.preview {
            Some(preview) if matches!(preview.scope, PreviewScope::All) => {
                transform_bounds(base_bounds, preview.transform)
            }
            Some(preview) => union_bounds(
                base_bounds,
                transform_bounds(base_bounds, preview.transform),
            ),
            None => base_bounds,
        };
        let plan_preview = match &self.preview {
            Some(preview) if matches!(preview.scope, PreviewScope::All) => Some(preview.transform),
            _ => None,
        };
        if let Ok(mut plan) = self.plan.lock() {
            plan.set_preview_transform(plan_preview);
        }
        if scoped {
            let (transform, scope) = match &self.preview {
                Some(preview) => (Some(preview.transform), preview.scope.clone()),
                None => (None, PreviewScope::All),
            };
            self.upload_scene(display, transform, &scope);
        } else if previous_scoped {
            // Leaving a partial preview: put the static edges back once.
            self.upload_scene(display, None, &PreviewScope::All);
        }
    }

    /// Cached triangle ranges for a scoped preview.
    fn affected_ranges(
        &mut self,
        display: &DisplayMesh,
        scope: &PreviewScope,
    ) -> Option<Vec<(u32, u32)>> {
        if matches!(scope, PreviewScope::All) {
            return None;
        }
        if let Some((cached_scope, ranges)) = &self.preview_ranges {
            if cached_scope == scope {
                return ranges.clone();
            }
        }
        let ranges = affected_triangle_ranges(display, scope);
        self.preview_ranges = Some((scope.clone(), ranges.clone()));
        ranges
    }

    /// Collapse a body scope that covers the whole mesh into the zero-rebuild
    /// whole-mesh path.
    fn scope_for_display(&mut self, display: &DisplayMesh, scope: PreviewScope) -> PreviewScope {
        let triangles = display.mesh.triangles.len() as u32;
        let PreviewScope::Bodies(_) = &scope else { return scope };
        if triangles == 0 {
            return PreviewScope::All;
        }
        match self.affected_ranges(display, &scope) {
            Some(ranges) if ranges.len() == 1 && ranges[0] == (0, triangles) => PreviewScope::All,
            _ => scope,
        }
    }

    /// Replace the scene slot. Only used when entering/leaving a partial-body
    /// preview or when the display mesh changes.
    fn upload_scene(
        &mut self,
        display: &DisplayMesh,
        transform: Option<[[f32; 4]; 4]>,
        scope: &PreviewScope,
    ) {
        let mut scene = self.build_scene(Some(display), transform, scope, false);
        scene.pbr.clear();
        self.key = self.key.wrapping_add(1);
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(scene);
        }
    }

    /// Edges and analysis overlays for the current display mesh. `transform`
    /// moves geometry for a drag preview; the world-space Z axis marker stays put
    /// because it belongs to the world frame, not to the edited body.
    fn build_scene(
        &self,
        display: Option<&DisplayMesh>,
        transform: Option<[[f32; 4]; 4]>,
        scope: &PreviewScope,
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
        let edge = PaletteIndex(1);
        let edge_width = 1.0;
        let triangle_edge = PaletteIndex(3);
        let face_normal = PaletteIndex(4);
        // Normals follow the inverse transpose so a mirrored or non-uniform
        // preview chain cannot flip the edge lighting offsets.
        let normal_transform = transform.map(normal_matrix);
        let moves = |body: &str| transform.is_some() && scope.is_affected(body);
        let world = |point: Vec3, moves: bool| {
            if !moves {
                return point;
            }
            transform.map_or(point, |matrix| {
                let moved = transform_point(matrix, point.to_f32());
                Vec3::new(f64::from(moved[0]), f64::from(moved[1]), f64::from(moved[2]))
            })
        };
        let direction = |normal: [f32; 3], moves: bool| {
            if !moves {
                return normal;
            }
            normal_transform.map_or(normal, |matrix| {
                let moved = transform_direction(matrix, normal);
                let length = (moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2]).sqrt();
                if length > 1.0e-12 {
                    [moved[0] / length, moved[1] / length, moved[2] / length]
                } else {
                    normal
                }
            })
        };
        if let Some(display) = display {
            let mesh: &CraftMesh = &display.mesh;
            if include_pbr {
                let surface = OpaqueSurface::new(Vec3::new(0.32, 0.34, 0.37), 0.0, 0.44, 1.62)
                    .unwrap_or_else(|_| OpaqueSurface::new(Vec3::new(0.6, 0.6, 0.6), 0.0, 0.44, 1.5).expect("fallback surface"));
                for (triangle_index, triangle) in mesh.triangles.iter().enumerate().step_by(backdrop_pbr_stride(mesh.triangles.len())) {
                    let [a, b, c] = triangle.map(|index| index as usize);
                    let (Some(pa), Some(pb), Some(pc)) =
                        (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
                    else {
                        continue;
                    };
                    let surface = display
                        .face_colors
                        .get(triangle_index)
                        .and_then(|color| {
                            OpaqueSurface::new(
                                Vec3::new(f64::from(color[0]), f64::from(color[1]), f64::from(color[2])),
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
            // Zebra is a continuity view: B-rep edge overlays are suppressed so
            // their dark strokes are not mistaken for stripe breaks.
            if self.view.edges && !self.view.zebra {
                for display_edge in &display.edges {
                    let moved = moves(&display_edge.id);
                    for (index, pair) in display_edge.points.windows(2).enumerate() {
                        let [a, b] = [pair[0], pair[1]];
                        let start = world(a, moved);
                        let end = world(b, moved);
                        // A zero-length segment is a vertex marker in the line
                        // shader: it draws a square/ball at the vertex.
                        let delta = end - start;
                        if delta.dot(delta) <= 1.0e-20 {
                            continue;
                        }
                        let normals = [
                            direction(display_edge.normals.get(index).copied().unwrap_or([0.0; 3]), moved),
                            direction(display_edge.normals.get(index + 1).copied().unwrap_or([0.0; 3]), moved),
                        ];
                        scene.lifted_line(
                            start.to_f32(),
                            end.to_f32(),
                            edge,
                            edge_width,
                            normals,
                        );
                    }
                }
            }
            if self.view.triangles {
                let mut seen = std::collections::HashSet::new();
                for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
                    let face = mesh.tri_face.get(triangle_index).copied().unwrap_or(0) as usize;
                    let moved = display.face_ids.get(face).is_some_and(|id| scope.is_affected(id));
                    for pair in [[triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]]] {
                        let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                        if !seen.insert(key) { continue; }
                        let (Some(a), Some(b)) = (mesh.positions.get(pair[0] as usize), mesh.positions.get(pair[1] as usize)) else { continue };
                        scene.line(world(*a, moved).to_f32(), world(*b, moved).to_f32(), triangle_edge, 0.7, false);
                    }
                }
            }
            if self.view.normals {
                // Aggregate triangles by display face, then draw one area-weighted
                // face normal from each face centroid.
                let mut faces: std::collections::HashMap<u32, (Vec3, Vec3, f64)> = std::collections::HashMap::new();
                for (index, triangle) in mesh.triangles.iter().enumerate() {
                    let [a, b, c] = triangle.map(|index| index as usize);
                    let (Some(pa), Some(pb), Some(pc)) = (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c)) else { continue };
                    let cross = (*pb - *pa).cross(*pc - *pa);
                    let area = cross.len() * 0.5;
                    if area <= 1.0e-12 { continue; }
                    let normal = cross.normalized().unwrap_or(Vec3::Z);
                    let centroid = (*pa + *pb + *pc) / 3.0;
                    let face = mesh.tri_face.get(index).copied().unwrap_or(index as u32);
                    let entry = faces.entry(face).or_insert((Vec3::ZERO, Vec3::ZERO, 0.0));
                    entry.0 = entry.0 + normal * area;
                    entry.1 = entry.1 + centroid * area;
                    entry.2 += area;
                }
                let length = display.radius * 0.075;
                for (face, (normal, centroid, area)) in faces {
                    if area <= 1.0e-12 { continue; }
                    let Some(normal) = normal.normalized() else { continue };
                    let moved = display.face_ids.get(face as usize).is_some_and(|id| scope.is_affected(id));
                    let center = world(centroid / area, moved);
                    let tip = world(centroid / area + normal * length, moved);
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
        let bounds = solidcraft_viewport::vec::Bounds {
            min: bounds.min,
            max: bounds.max,
        };
        let aspect = f64::from(width) / f64::from(height.max(1.0));
        let half_height = camera.half_height();
        let (grid_minor, grid_major, grid_fine_weight) =
            crate::viewport_grid::adaptive_grid_crossfade(half_height);
        let back_vec = camera.back();
        let slant = back_vec.z;
        let grid_extent = crate::viewport_grid::adaptive_grid_extent(half_height, grid_major, slant);
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
        let view_proj = shift_view_proj(
            camera.view_proj_bounds(aspect, &bounds).to_f32(),
            lens_shift,
        );
        // A whole-mesh preview transforms the scene projection instead of
        // rebuilding edge geometry: the scene buffers stay uploaded and the GPU
        // does all the work per frame.
        let scene_view_proj = match &self.preview {
            Some(preview) if matches!(preview.scope, PreviewScope::All) => {
                multiply_matrix(view_proj, preview.transform)
            }
            _ => view_proj,
        };
        // The backdrop keeps its last scene while a preview is active; moving
        // the projection keeps the defocused copy under the previewed geometry.
        let backdrop_view_proj = {
            let cropped = crop_view_proj(view_proj, backdrop_crop);
            match &self.preview {
                Some(preview) => multiply_matrix(cropped, preview.transform),
                None => cropped,
            }
        };
        let eye = camera.eye().to_f32();
        let back = camera.back().to_f32();
        let (right, up, _) = camera.basis();
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
                [
                    r[0],
                    r[1],
                    r[2],
                    if camera.fov > 0.0 { 1.0 } else { 0.0 },
                ]
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
            grid: [
                grid_minor as f32,
                0.07,
                0.22,
                grid_extent as f32,
            ],
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
        GpuCallback {
            object,
            background: self.background,
            view: self.view,
            chrome_visible: self.chrome_visible,
            frame: Frame {
            product: false,
            preview_bounds: None,
            moving: self
                .ghost_edge_matrix
                .filter(|_| self.ghost_edge_present)
                .map(|transform| MovingLayer {
                    key: self.ghost_edge_key,
                    slot: self.ghost_edge_slot.clone(),
                    transform,
                }),
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
            clip: [0.0; 4],
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
            },
            backdrop_view_proj,
            backdrop_blur_scale,
            backdrop_crop,
            target_format: self.target_format,
            scene: self.scene.clone(),
            plan: self.plan.clone(),
            grid,
            axis,
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
    let m = [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx, ty, 0.0, 1.0],
    ];
    multiply_matrix(m, vp)
}

/// One viewport draw as an egui paint callback.
pub struct GpuCallback {
    pub object: ObjectUniforms,
    pub background: [u8; 4],
    pub view: ViewSettings,
    #[allow(dead_code)] // Reserved for GPU chrome rendering.
    pub chrome_visible: bool,
    pub frame: Frame,
    pub backdrop_view_proj: [[f32; 4]; 4],
    pub backdrop_blur_scale: [f32; 4],
    pub backdrop_crop: [f32; 4],
    pub target_format: wgpu::TextureFormat,
    pub scene: Arc<Mutex<Option<SceneTarget>>>,
    pub plan: Arc<Mutex<crate::gpu_plan::PlanState>>,
    pub grid: GridUniforms,
    pub axis: AxisUniforms,
}

impl CallbackTrait for GpuCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(clear) = resources.get::<ViewportClear>() {
            clear.set_background(queue, self.background);
        }
        if let Some(grid) = resources.get::<GridRenderer>() {
            grid.prepare(queue, &self.grid);
        }
        if let Some(axis) = resources.get::<AxisRenderer>() {
            axis.prepare(queue, &self.axis);
        }
        if let Some(object) = resources.get::<Arc<Mutex<ObjectRenderer>>>() {
            if let Ok(mut object) = object.lock() {
                object.prepare(device, queue, &self.object);
            }
        }
        let plan_active = self
            .plan
            .lock()
            .map(|mut plan| {
                plan.prepare(device, queue, encoder, &self.object);
                plan.is_available()
            })
            .unwrap_or(false);
        {
            let Some(renderer) = resources.get_mut::<Renderer>() else {
                return Vec::new();
            };
            renderer.prepare(device, queue, encoder, &self.frame);
        }
        let samples = msaa_samples();
        let size = [
            self.frame.size_px[0].round().max(1.0) as u32,
            self.frame.size_px[1].round().max(1.0) as u32,
        ];
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
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("viewport-scene-offscreen"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            depth_slice: None,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
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
                    pass.set_viewport(0.0, 0.0, size[0] as f32, size[1] as f32, 0.0, 1.0);
                    pass.set_pipeline(&clear.pipeline);
                    pass.set_bind_group(0, &clear.bind_group, &[]);
                    pass.draw(0..3, 0..1);
                    if plan_active {
                        if let Ok(plan) = self.plan.lock() {
                            plan.paint(&mut pass);
                        }
                    }
                    if let Ok(object) = object.lock() {
                        object.paint(
                            &mut pass,
                            self.view.translucent && !plan_active,
                            !plan_active,
                        );
                    }
                    renderer.paint(
                        &self.frame,
                        &mut pass,
                        [0.0, 0.0, size[0] as f32, size[1] as f32],
                    );
                    grid.paint(&mut pass);
                    axis.paint(&mut pass);
                }
            }
        }
        if let Some(backdrop) = resources.get_mut::<Arc<Mutex<BackdropRenderer>>>() {
            if let Ok(mut backdrop) = backdrop.lock() {
                backdrop.prepare(
                    device,
                    queue,
                    encoder,
                    &self.frame,
                    self.backdrop_view_proj,
                    self.backdrop_blur_scale,
                    external_view.as_ref().map(|view| (view, self.backdrop_crop)),
                );
            }
        }
        Vec::new()
    }

    fn paint(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_mesh_is_chunked_with_local_indices() {
        let vertices: Vec<ObjectVertex> = (0..8)
            .map(|i| ObjectVertex { position: [i as f32, 0.0, 0.0], normal: [0.0, 0.0, 1.0], colour: [1.0, 0.0, 0.0] })
            .collect();
        let indices = vec![0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 0];
        let chunks = chunk_object_mesh_with_limit(vertices, indices, 3);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks.iter().map(|(_, indices)| indices.len()).sum::<usize>(), 12);
        for (vertices, indices) in &chunks {
            assert!(vertices.len() <= 3);
            assert!(indices.iter().all(|index| (*index as usize) < vertices.len()));
        }
    }

    #[test]
    fn lens_shift_is_constant_after_perspective_divide() {
        // clip.x = x, clip.w = z; after divide ndc.x = x/z.
        let vp = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]];
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
        let (vertices, indices) = indexed_object_mesh(&display.mesh, [0.2, 0.3, 0.4]).expect("indexed mesh");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn shading_creases_follow_brep_faces_not_just_triangle_angle() {
        use nodecraft_protocol::Mesh as NodeMesh;
        // A horizontal and a 45-degree face share an edge. Their normals are
        // 45 degrees apart, so an angular-only smoother merges them and makes
        // a flat chamfer look rounded.
        let source = NodeMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            faces: vec![[0, 1, 2], [0, 1, 3]],
            face_groups: vec!["face:0".to_owned(), "face:1".to_owned()],
            ..NodeMesh::default()
        };
        let display = crate::adapter::display_mesh(&source);
        let (vertices, _) =
            smoothed_object_mesh(&display.mesh, [0.2, 0.3, 0.4]).expect("mesh");
        assert_eq!(vertices.len(), 6, "the two B-rep faces must not share normals");
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
    fn branch_ghost_scope_selects_only_the_branch_triangles() {
        use nodecraft_protocol::Mesh as NodeMesh;
        let tri = |group: &str| NodeMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            faces: vec![[0, 1, 2]],
            face_groups: vec![group.to_owned()],
            ..NodeMesh::default()
        };
        let merged = nodecraft_mesh::merge_meshes(vec![tri("face:0"), tri("face:0")]);
        let display = crate::adapter::display_mesh(&merged);
        let ranges = affected_triangle_ranges(
            &display,
            &PreviewScope::Bodies(vec!["body:0:".to_owned()]),
        );
        assert_eq!(ranges, Some(vec![(0, 1)]), "face ids: {:?}", display.face_ids);
    }

    #[test]
    fn object_shader_applies_per_instance_alpha() {
        assert!(
            OBJECT_SHADER.contains("u.params.x * in.alpha"),
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
                assert!((normal[column][row] - model[column][row]).abs() < 1.0e-6, "{column} {row}");
            }
        }
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

    #[test]
    fn preview_draw_ranges_split_moved_and_stationary_bodies() {
        // Affected ranges arrive unsorted and overlapping; the stationary ranges
        // are the complement clipped to the mesh.
        assert_eq!(
            preview_draw_ranges(&[(4, 6), (1, 3), (2, 5)], 8),
            (vec![(1, 6)], vec![(0, 1), (6, 8)])
        );
        assert_eq!(preview_draw_ranges(&[(2, 4)], 6), (vec![(2, 4)], vec![(0, 2), (4, 6)]));
        assert_eq!(preview_draw_ranges(&[], 5), (vec![], vec![(0, 5)]));
        assert_eq!(preview_draw_ranges(&[(0, 99)], 5), (vec![(0, 5)], vec![]));
    }

    #[test]
    fn affected_triangle_ranges_follow_display_body_prefixes() {
        use nodecraft_protocol::Mesh as NodeMesh;
        let mesh = NodeMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [3, 4, 5]],
            face_groups: vec!["body:0:face:0".to_owned(), "body:1:face:0".to_owned()],
            ..NodeMesh::default()
        };
        let display = crate::adapter::display_mesh(&mesh);
        let first = PreviewScope::Bodies(vec!["body:0:".to_owned()]);
        assert_eq!(affected_triangle_ranges(&display, &first), Some(vec![(0, 1)]));
        let second = PreviewScope::Bodies(vec!["body:1:".to_owned()]);
        assert_eq!(affected_triangle_ranges(&display, &second), Some(vec![(1, 2)]));
        assert_eq!(affected_triangle_ranges(&display, &PreviewScope::All), None);
    }

    #[test]
    fn preview_draw_ranges_cover_each_index_once_and_stay_in_bounds() {
        // Regression: the stationary range must use (end - start) as its
        // triangle count, not the end index, or the draw reads past the index
        // buffer (a wgpu validation panic).
        let total = 972;
        let (moved, stationary) = preview_draw_ranges(&[(0, 12)], total);
        let mut index_ranges: Vec<(u32, u32)> = moved
            .into_iter()
            .chain(stationary)
            .map(draw_index_range)
            .collect();
        index_ranges.sort_unstable();
        let mut cursor = 0;
        for (start, count) in index_ranges {
            assert_eq!(start, cursor, "ranges must tile the index buffer");
            assert!(count > 0, "empty draw");
            cursor += count;
        }
        assert_eq!(cursor, total * 3);
    }
    fn ghost_edge(points: Vec<solidcore_record::Vec3>) -> crate::adapter::DisplayEdge {
        crate::adapter::DisplayEdge {
            id: "edge:0".into(),
            midpoint: [0.0, 0.0, 0.0],
            normals: Vec::new(),
            points,
        }
    }

    #[test]
    fn ghost_edge_scene_counts_non_degenerate_segments() {
        let edge = ghost_edge(vec![
            solidcore_record::Vec3::new(0.0, 0.0, 0.0),
            solidcore_record::Vec3::new(1.0, 0.0, 0.0),
            solidcore_record::Vec3::new(1.0, 0.0, 0.0),
            solidcore_record::Vec3::new(1.0, 1.0, 0.0),
        ]);
        let (_, segments) = ghost_edge_scene(&[edge], [255, 255, 255, 80], 1.6);
        assert_eq!(segments, 2);
    }

    #[test]
    fn ghost_edge_scene_is_empty_for_no_edges() {
        let (_, segments) = ghost_edge_scene(&[], [255, 255, 255, 80], 1.6);
        assert_eq!(segments, 0);
    }

}
