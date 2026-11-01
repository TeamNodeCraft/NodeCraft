//! wgpu viewport glue around SolidCraft's GPU renderer.
//!
//! The renderer, scene format and pipelines belong to `solidcraft-gpu`; this module only
//! converts a NodeCraft display mesh into a `GpuScene`, owns the `SceneSlot`, and implements
//! the `egui_wgpu` paint callback.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use egui_wgpu::{wgpu, CallbackTrait, RenderState};
use solidcraft_geom::{Aabb3, Mesh as CraftMesh, Vec3};

use crate::viewport::ViewSettings;
use solidcraft_gpu::depth_format;
use solidcraft_gpu::view::{Frame, GpuScene, GpuTarget, Palette, PaletteIndex, Renderer, SceneSlot};
use solidcraft_render::surface::OpaqueSurface;
use solidcraft_render::Camera;

use crate::adapter::DisplayMesh;
use crate::viewport_grid::{grid_lines, grid_step, GridKind};
use crate::gpu_backdrop::BackdropRenderer;

const EMPTY_GRID_RADIUS: f64 = 100.0;

/// Flat palette background for the viewport callback. SolidCraft clears its own
/// offscreen frame to transparent; this quad makes the central app surface show the
/// palette's darkest colour instead of whatever the window compositor leaves behind.
struct ViewportClear {
    pipeline: wgpu::RenderPipeline,
}

const CLEAR_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(28.0 / 255.0, 28.0 / 255.0, 30.0 / 255.0, 1.0);
}
"#;

/// MSAA used by the egui pass and all custom viewport/node pipelines.
///
/// macOS defaults to 4x; other platforms default to 1x because the surface
/// sample count is chosen before the adapter is queried. Set NODECRAFT_MSAA=4
/// to opt in on a platform that supports it.
pub fn msaa_samples() -> u32 {
    std::env::var("NODECRAFT_MSAA")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|samples| matches!(samples, 1 | 4))
        .unwrap_or(if cfg!(target_os = "macos") { 4 } else { 1 })
}


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridUniforms {
    eye: [f32; 4],
    target: [f32; 4],
    center: [f32; 4],
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
    let coord = (world.xy - u.center.xy) / u.grid.x;
    let minor_width = max(fwidth(coord).x, fwidth(coord).y);
    let major_coord = coord / u.options.x;
    let major_width = max(fwidth(major_coord).x, fwidth(major_coord).y);
    // Keep both weights present at every adaptive zoom level; the adaptive step
    // already keeps the cell size readable.
    // Crossfade a whole coarse grid into the next finer grid over one decade.
    // The finer level's majors are the coarser level's minors, so both line
    // counts and their meanings stay continuous through the transition.
    // Stagger the two levels: the finer grid arrives first, then the
    // outgoing coarse grid clears. They do not dim to half strength together.
    let coarse_weight = u.options.y;
    let fine_weight = u.options.z;
    let coarse_minor = line_mask(coord, 0.62) * coarse_weight;
    let coarse_major = line_mask(major_coord, 0.88) * coarse_weight;
    let fine_minor = line_mask(coord * 10.0, 0.62) * fine_weight;
    let fine_major = line_mask(coord, 0.88) * fine_weight;
    let minor = coarse_minor + fine_minor;
    let major = coarse_major + fine_major;
    let total = max(minor + major, 1.0e-6);
    // params.w is the perspective flag: 1 = perspective, 0 = orthographic.
    let perspective = u.params.w > 0.5;
    let view_ray = normalize(world - u.eye.xyz);
    let horizon = smoothstep(0.03, 0.22, abs(view_ray.z));
    // The grid plane may extend past the model's far plane. Fade it before that
    // clip so perspective keeps a horizon rather than a hard clipped edge.
    let distance_fade = 1.0 - smoothstep(
        u.axis_y.w * 0.32,
        u.axis_y.w * 0.78,
        length(world - u.eye.xyz),
    );
    // Orthographic views have no horizon. Fade the whole floor as the view
    // approaches the XY plane so the finite camera cannot sit on grid lines.
    let low_angle = smoothstep(0.04, 0.22, abs(u.back.z));
    let perspective_fade = horizon * distance_fade * low_angle;
    let fade = select(low_angle, perspective_fade, perspective);

    let world_x = abs(world.y - u.center.y) / max(fwidth(world.y), 1.0e-6);
    let world_y = abs(world.x - u.center.x) / max(fwidth(world.x), 1.0e-6);
    let x_axis = select(
        0.0,
        1.0 - smoothstep(0.5, 1.5, world_x),
        world.x >= u.center.x,
    );
    let y_axis = select(
        0.0,
        1.0 - smoothstep(0.5, 1.5, world_y),
        world.y >= u.center.y,
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


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectVertex {
    position: [f32; 3],
    normal: [f32; 3],
    colour: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ObjectUniforms {
    view_proj: [[f32; 4]; 4],
    back: [f32; 4],
    eye: [f32; 4],
    // x = object alpha, yzw reserved.
    params: [f32; 4],
}

const OBJECT_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    back: vec4<f32>,
    eye: vec4<f32>,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct In {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) colour: vec3<f32>,
};
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) colour: vec3<f32>,
};

@vertex
fn vs(in: In) -> Out {
    var out: Out;
    out.position = u.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.world = in.position;
    out.colour = in.colour;
    return out;
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    return select(c * 12.92, 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), c > vec3<f32>(0.0031308));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gaussian_spec(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let nv = max(dot(n, v), 0.0);
    let nl = max(dot(n, l), 0.0);
    if nv <= 0.0 || nl <= 0.0 { return 0.0; }
    let a = roughness * roughness;
    let a2 = a * a;
    let d = a2 / (3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let gv = 2.0 / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    let gl = 2.0 / (nl + sqrt(a2 + (1.0 - a2) * nl * nl));
    return d * 0.25 * gv * gl;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    var v = normalize(u.back.xyz);
    if (u.eye.w > 0.5) { v = normalize(u.eye.xyz - in.world); }
    if (dot(n, v) < 0.0) { n = -n; }

    if (u.params.y > 0.5) {
        // Zebra analysis: reflect the view vector and stripe the resulting
        // direction. Curvature changes bend the reflected direction, breaking
        // the stripes; G1/G2 joins are read by whether stripes kink or flow.
        let reflected = reflect(-v, n);
        let angle = atan2(reflected.y, reflected.x);
        let stripe = sin(angle * 12.0);
        let tone = mix(0.035, 0.965, smoothstep(-0.045, 0.045, stripe));
        return vec4<f32>(encode(vec3<f32>(tone)), u.params.x);
    }

    let base = in.colour;
    let roughness = 0.44;
    let f0 = vec3<f32>(0.04) * 0.55;
    let view = v;
    var diffuse = vec3<f32>(0.0);
    var specular = vec3<f32>(0.0);

    // Swift viewer light rig: hemisphere + key/fill/rim/bounce.
    let sky = vec3<f32>(0.95, 0.94, 0.925);
    let ground = vec3<f32>(0.23, 0.23, 0.23);
    let hemisphere = mix(ground, sky, clamp(n.z * 0.5 + 0.5, 0.0, 1.0)) * 0.86;
    diffuse += base * hemisphere * 0.3183099;

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
        let l = lights[index].xyz;
        let strength = lights[index].w * max(dot(n, l), 0.0);
        diffuse += base * light_colours[index] * strength * 0.3183099;
        specular += light_colours[index] * gaussian_spec(n, view, l, roughness) * strength;
    }
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(n, view), 0.0), 5.0);
    specular *= fresnel;

    let mapped = aces((diffuse + specular * 0.55 + vec3<f32>(0.015)) * 1.0);
    return vec4<f32>(encode(mapped), u.params.x);
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

struct ObjectRenderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    vertices: Option<wgpu::Buffer>,
    indices: Option<wgpu::Buffer>,
    index_count: u32,
    pending: Option<(Vec<ObjectVertex>, Vec<u32>)>,
}

impl ObjectRenderer {
    fn install(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32, depth: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-object"),
            source: wgpu::ShaderSource::Wgsl(OBJECT_SHADER.into()),
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
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ObjectVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
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
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ObjectVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                    ],
                })],
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
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, transparent_pipeline, bind_group, uniforms, vertices: None, indices: None, index_count: 0, pending: None }
    }

    fn set_mesh(&mut self, vertices: Vec<ObjectVertex>, indices: Vec<u32>) {
        self.pending = Some((vertices, indices));
    }

    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: &ObjectUniforms) {
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(uniforms));
        if let Some((vertices, indices)) = self.pending.take() {
            if vertices.is_empty() || indices.is_empty() {
                self.vertices = None;
                self.indices = None;
                self.index_count = 0;
                return;
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
            self.vertices = Some(vertex_buffer);
            self.indices = Some(index_buffer);
            self.index_count = indices.len() as u32;
        }
    }

    fn paint(&self, pass: &mut wgpu::RenderPass<'static>, transparent: bool) {
        let (Some(vertices), Some(indices)) = (&self.vertices, &self.indices) else { return };
        pass.set_pipeline(if transparent { &self.transparent_pipeline } else { &self.pipeline });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
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

fn colour(rgba: [u8; 4], alpha: f32) -> [f32; 4] {
    [
        f32::from(rgba[0]) / 255.0,
        f32::from(rgba[1]) / 255.0,
        f32::from(rgba[2]) / 255.0,
        alpha,
    ]
}

pub struct GpuViewport {
    slot: SceneSlot,
    key: u64,
    backdrop: Arc<Mutex<BackdropRenderer>>,
    backdrop_id: egui::TextureId,
    grid_extent: f64,
    grid_step: f32,
    grid_center: [f32; 3],
    bounds: Aabb3,
    device: wgpu::Device,
    object: Arc<Mutex<ObjectRenderer>>,
    view: ViewSettings,
    chrome_visible: bool,
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

        let module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-clear"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_SHADER.into()),
        });
        let layout = rs.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-clear-layout"),
            bind_group_layouts: &[],
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
        rs.renderer.write().callback_resources.insert(ViewportClear { pipeline });
        rs.renderer.write().callback_resources.insert(GridRenderer::install(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
        ));
        let object = Arc::new(Mutex::new(ObjectRenderer::install(
            &rs.device,
            rs.target_format,
            msaa_samples(),
            depth_format(&rs.device, true),
        )));
        rs.renderer.write().callback_resources.insert(object.clone());
        Some(Self {
            slot: Arc::new(Mutex::new(None)),
            key: 0,
            backdrop,
            backdrop_id,
            grid_extent: 0.0,
            grid_step: 1.0,
            grid_center: [0.0; 3],
            bounds: Aabb3 {
                min: Vec3::new(-EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS, -EMPTY_GRID_RADIUS),
                max: Vec3::new(EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS, EMPTY_GRID_RADIUS),
            },
            device: rs.device.clone(),
            object,
            view: ViewSettings::default(),
            chrome_visible: true,
        })
    }

    pub fn bounds(&self) -> &Aabb3 {
        &self.bounds
    }

    pub fn set_chrome_visible(&mut self, visible: bool) {
        self.chrome_visible = visible;
    }

    pub fn set_view(&mut self, view: ViewSettings, display: Option<&DisplayMesh>) {
        if self.view == view {
            return;
        }
        self.view = view;
        self.set_mesh(display);
    }

    pub fn set_mesh(&mut self, display: Option<&DisplayMesh>) {
        self.key = self.key.wrapping_add(1);
        let radius = display.map_or(EMPTY_GRID_RADIUS, |display| display.radius);
        self.grid_extent = crate::viewport_grid::grid_extent(radius);
        self.grid_step = grid_step(radius) as f32;
        self.bounds = display.map_or_else(
            || Aabb3 {
                min: Vec3::new(-radius, -radius, -radius),
                max: Vec3::new(radius, radius, radius),
            },
            |display| display.mesh.bounds(),
        );
        let theme = crate::theme::get();
        let palette = Palette::with_stable(vec![
            [151, 162, 176, 255],
            [0, 0, 0, 255],
            [theme.axis_z[0], theme.axis_z[1], theme.axis_z[2], 230],
            [148, 148, 148, 140],
            [76, 201, 245, 230],
        ]);
        let mut scene = GpuScene::with_palette(palette);
        let _body = PaletteIndex(0);
        let edge = PaletteIndex(1);
        let edge_width = 2.2;
        let triangle_edge = PaletteIndex(3);
        let face_normal = PaletteIndex(4);
        let mut object_vertices = Vec::new();
        let mut object_indices = Vec::new();
        if let Some(display) = display {
            let mesh: &CraftMesh = &display.mesh;
            let draw_object = true;
            // The app draws the solid material itself (see ObjectRenderer) so it can use
            // the Swift viewer's light rig and tone mapping. The SolidCraft scene keeps
            // only the edges; the blurred backdrop keeps a PBR copy for defocused colour.
            let surface = OpaqueSurface::new(Vec3::new(0.32, 0.34, 0.37), 0.0, 0.44, 1.62)
                .unwrap_or_else(|_| OpaqueSurface::new(Vec3::new(0.6, 0.6, 0.6), 0.0, 0.44, 1.5).expect("fallback surface"));
            if draw_object {
                for (triangle_index, triangle) in mesh.triangles.iter().enumerate() {
                    let [a, b, c] = triangle.map(|index| index as usize);
                    let (Some(pa), Some(pb), Some(pc)) =
                        (mesh.positions.get(a), mesh.positions.get(b), mesh.positions.get(c))
                    else {
                        continue;
                    };
                    let fallback = (*pb - *pa).cross(*pc - *pa);
                    let fallback = fallback.normalized().unwrap_or(solidcraft_geom::Vec3::Z);
                    let colour = if self.view.face_colours {
                        face_colour(mesh.tri_face.get(triangle_index).copied().unwrap_or(0))
                    } else {
                        [0.17, 0.18, 0.195]
                    };
                    for (index, point) in [(a, pa), (b, pb), (c, pc)] {
                        let normal = mesh.normals.get(index).copied().filter(|normal| normal.len() > 1.0e-9).unwrap_or(fallback);
                        object_indices.push(object_vertices.len() as u32);
                        object_vertices.push(ObjectVertex { position: point.to_f32(), normal: normal.to_f32(), colour });
                    }
                    scene.pbr_tri(pa.to_f32(), fallback.to_f32(), surface);
                    scene.pbr_tri(pb.to_f32(), fallback.to_f32(), surface);
                    scene.pbr_tri(pc.to_f32(), fallback.to_f32(), surface);
                }
            }
            if self.view.edges {
                for edge_points in &mesh.edges {
                    for pair in edge_points.windows(2) {
                        let [a, b] = [pair[0], pair[1]];
                        scene.line(a.to_f32(), b.to_f32(), edge, edge_width, false);
                    }
                }
            }
            if self.view.triangles {
                let mut seen = std::collections::HashSet::new();
                for triangle in &mesh.triangles {
                    for pair in [[triangle[0], triangle[1]], [triangle[1], triangle[2]], [triangle[2], triangle[0]]] {
                        let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                        if !seen.insert(key) { continue; }
                        let (Some(a), Some(b)) = (mesh.positions.get(pair[0] as usize), mesh.positions.get(pair[1] as usize)) else { continue };
                        scene.line(a.to_f32(), b.to_f32(), triangle_edge, 0.7, false);
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
                for (_, (normal, centroid, area)) in faces {
                    if area <= 1.0e-12 { continue; }
                    let Some(normal) = normal.normalized() else { continue };
                    let center = centroid / area;
                    scene.line(center.to_f32(), (center + normal * length).to_f32(), face_normal, 1.5, false);
                }
            }
        }
        if let Ok(mut renderer) = self.object.lock() {
            renderer.set_mesh(object_vertices, object_indices);
        }
        if let Some(axis) = grid_lines(radius)
            .into_iter()
            .find(|line| line.kind == GridKind::AxisZ)
        {
            scene.under_line(
                [
                    self.grid_center[0] + axis.start[0],
                    self.grid_center[1] + axis.start[1],
                    self.grid_center[2] + axis.start[2],
                ],
                [
                    self.grid_center[0] + axis.end[0],
                    self.grid_center[1] + axis.end[1],
                    self.grid_center[2] + axis.end[2],
                ],
                PaletteIndex(2),
                2.2,
            );
        }
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

    pub fn backdrop_texture_id(&self) -> egui::TextureId {
        self.backdrop_id
    }

    pub fn frame(
        &self,
        camera: &Camera,
        width: f32,
        height: f32,
        pixels_per_point: f32,
        bounds: &solidcraft_geom::Aabb3,
        backdrop_crop: [f32; 4],
        backdrop_blur_scale: [f32; 2],
    ) -> GpuCallback {
        let aspect = f64::from(width) / f64::from(height.max(1.0));
        let (grid_step, grid_level) = crate::viewport_grid::adaptive_grid_level(camera.distance);
        let (coarse_weight, fine_weight) = crate::viewport_grid::grid_level_weights(grid_level);
        // Keep the model's normal near/far planes. Expanding them for the grid
        // destroys depth precision and makes large perspective models flicker.
        let view_proj = camera.view_proj_bounds(aspect, bounds).to_f32();
        let backdrop_view_proj = crop_view_proj(view_proj, backdrop_crop);
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
            center: [self.grid_center[0], self.grid_center[1], self.grid_center[2], 1.0],
            right: {
                let r = right.to_f32();
                [r[0], r[1], r[2], 0.0]
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
                grid_step as f32,
                0.07,
                0.22,
                self.grid_extent as f32,
            ],
            minor: colour(theme.muted, 1.0),
            major: colour(theme.icon_hover, 1.0),
            axis_x: {
                let mut colour = colour(theme.axis_x, 1.0);
                colour[3] = crate::viewport_grid::GRID_MAJOR_STEP as f32;
                colour
            },
            axis_y: {
                let mut colour = colour(theme.axis_y, 1.0);
                colour[3] = camera.clip_bounds(bounds).1 as f32;
                colour
            },
            options: [
                crate::viewport_grid::GRID_MAJOR_EVERY as f32,
                coarse_weight,
                fine_weight,
                0.0,
            ],
        };
        let object = ObjectUniforms {
            view_proj,
            back: [back[0], back[1], back[2], 0.0],
            eye: [eye[0], eye[1], eye[2], if camera.fov > 0.0 { 1.0 } else { 0.0 }],
            params: [
                if self.view.translucent { 0.34 } else { 1.0 },
                if self.view.zebra { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        GpuCallback {
            object,
            view: self.view,
            chrome_visible: self.chrome_visible,
            frame: Frame {
            product: false,
            key: self.key,
            slot: self.slot.clone(),
            sketch_key: 0,
            sketch_slot: Arc::new(Mutex::new(None)),
            hl_key: 0,
            hl_slot: Arc::new(Mutex::new(None)),
            pv_key: 0,
            pv_slot: Arc::new(Mutex::new(None)),
            pv_hide: false,
            view_proj,
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
            ssaa: 1.0,
            hover_tint: [0.0; 3],
            selection_tint: [0.0; 3],
            hatch: Default::default(),
            },
            backdrop_view_proj,
            backdrop_blur_scale,
            grid,
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
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = (0..4).map(|k| m[k][row] * vp[col][k]).sum();
        }
    }
    out
}

/// One viewport draw as an egui paint callback.
pub struct GpuCallback {
    pub object: ObjectUniforms,
    pub view: ViewSettings,
    pub chrome_visible: bool,
    pub frame: Frame,
    pub backdrop_view_proj: [[f32; 4]; 4],
    pub backdrop_blur_scale: [f32; 2],
    pub grid: GridUniforms,
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
        if let Some(grid) = resources.get::<GridRenderer>() {
            grid.prepare(queue, &self.grid);
        }
        if let Some(object) = resources.get::<Arc<Mutex<ObjectRenderer>>>() {
            if let Ok(mut object) = object.lock() {
                object.prepare(device, queue, &self.object);
            }
        }
        if let Some(renderer) = resources.get_mut::<Renderer>() {
            renderer.prepare(device, queue, encoder, &self.frame);
            if let Some(backdrop) = resources.get_mut::<Arc<Mutex<BackdropRenderer>>>() {
                if let Ok(mut backdrop) = backdrop.lock() {
                    backdrop.prepare(device, queue, encoder, &self.frame, self.backdrop_view_proj, self.backdrop_blur_scale);
                }
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(renderer) = resources.get::<Renderer>() else { return };
        let viewport = info.viewport_in_pixels();
        if let Some(clear) = resources.get::<ViewportClear>() {
            pass.set_viewport(
                viewport.left_px as f32,
                viewport.top_px as f32,
                viewport.width_px as f32,
                viewport.height_px as f32,
                0.0,
                1.0,
            );
            pass.set_pipeline(&clear.pipeline);
            pass.draw(0..3, 0..1);
        }
        if let Some(object) = resources.get::<Arc<Mutex<ObjectRenderer>>>() {
            if let Ok(object) = object.lock() {
                object.paint(pass, self.view.translucent);
            }
        }
        renderer.paint(
            &self.frame,
            pass,
            [
                viewport.left_px as f32,
                viewport.top_px as f32,
                viewport.width_px as f32,
                viewport.height_px as f32,
            ],
        );
        if let Some(grid) = resources.get::<GridRenderer>() {
            grid.paint(pass);
        }
    }
}
