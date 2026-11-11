//! GPU evaluation and rendering of `solidcore_tess::TessPlan`s.
//!
//! Plans come from the engine worker with local-body transforms. This module
//! evaluates the plans with `solidcraft-viewport-gpu`'s `TessGpu`, applies each
//! body's placement/mirror/scale as a per-plan instance matrix, and draws the
//! resulting `GpuMesh` through NodeCraft's existing PBR shader. The CPU display
//! mesh remains the authority for edges, faces, ghosts, picking and the fallback
//! path.

use egui_wgpu::wgpu;
use solidcraft_viewport_gpu::{GpuMesh, TessGpu};
use nodecraft_layers::tess::TessPlan;

use nodecraft_display::{DisplayTransform, PlanInstance};

use crate::gpu_viewport::{object_instance_layout, GpuInstance, ObjectUniforms, GHOST_SHADER, OBJECT_SHADER};

/// Flat triangle-edge overlay for the plan mesh. It shares the plan vertex and
/// instance layouts, but none of the PBR inputs.
const TRIANGLE_SHADER: &str = r#"
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
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};

@vertex
fn vs(in: In) -> @builtin(position) vec4<f32> {
    let model = u.extra_model * mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    return u.view_proj * model * vec4<f32>(in.position, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.12, 0.14, 0.17, 0.72);
}
"#;

/// Plan vertex layout in `GpuMesh`: position, normal.
fn plan_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 1 },
    ];
    wgpu::VertexBufferLayout {
        array_stride: solidcraft_viewport_gpu::VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Colour is a one-instance vertex input at location 2, so the existing PBR
/// shader works unchanged with a single body colour.
fn plan_colour_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 2,
    }];
    wgpu::VertexBufferLayout {
        array_stride: 16,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

struct PlanRenderer {
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    triangle_pipeline: wgpu::RenderPipeline,
    ghost_pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    ghost_uniforms: wgpu::Buffer,
    ghost_bind_group: wgpu::BindGroup,
}

impl PlanRenderer {
    fn install(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
        reversed: bool,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-plan"),
            source: wgpu::ShaderSource::Wgsl(OBJECT_SHADER.into()),
        });
        let ghost_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-plan-ghost"),
            source: wgpu::ShaderSource::Wgsl(GHOST_SHADER.into()),
        });
        let triangle_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-plan-triangles"),
            source: wgpu::ShaderSource::Wgsl(TRIANGLE_SHADER.into()),
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-plan-uniforms"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-plan-bind-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-plan-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-plan-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-plan-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(plan_vertex_layout()), Some(plan_colour_layout()), Some(object_instance_layout())],
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
            label: Some("viewport-plan-transparent-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(plan_vertex_layout()), Some(plan_colour_layout()), Some(object_instance_layout())],
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
        let triangle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-plan-triangle-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &triangle_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(plan_vertex_layout()), Some(object_instance_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &triangle_module,
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
            label: Some("viewport-plan-ghost-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(plan_vertex_layout()), Some(plan_colour_layout()), Some(object_instance_layout())],
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
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });
        let ghost_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-plan-ghost-uniforms"),
            size: std::mem::size_of::<ObjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ghost_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-plan-ghost-bind-group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ghost_uniforms.as_entire_binding() }],
        });
        Self {
            pipeline,
            transparent_pipeline,
            triangle_pipeline,
            ghost_pipeline,
            uniforms,
            bind_group,
            ghost_uniforms,
            ghost_bind_group,
        }
    }

    fn prepare(
        &self,
        queue: &wgpu::Queue,
        uniforms: &ObjectUniforms,
        extra_model: Option<[[f32; 4]; 4]>,
        ghost_extra: Option<[[f32; 4]; 4]>,
    ) {
        let main = uniforms.with_extra(extra_model.unwrap_or_else(crate::gpu_viewport::identity_matrix));
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&main));
        let ghost = uniforms.with_extra(ghost_extra.unwrap_or_else(crate::gpu_viewport::identity_matrix));
        queue.write_buffer(&self.ghost_uniforms, 0, bytemuck::bytes_of(&ghost));
    }

    fn paint(
        &self,
        pass: &mut wgpu::RenderPass<'static>,
        mesh: &GpuMesh,
        colour: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        translucent: bool,
        instance_count: u32,
    ) {
        pass.set_pipeline(if translucent { &self.transparent_pipeline } else { &self.pipeline });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, colour.slice(..));
        pass.set_vertex_buffer(2, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        let Some(part) = mesh.parts.first() else { return };
        pass.draw_indexed(part.indices.clone(), 0, 0..instance_count);
    }

    fn paint_triangle_edges(
        &self,
        pass: &mut wgpu::RenderPass<'static>,
        mesh: &GpuMesh,
        instances: &wgpu::Buffer,
        lines: &TriangleLines,
        instance_count: u32,
    ) {
        pass.set_pipeline(&self.triangle_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(lines.indices.slice(..), wgpu::IndexFormat::Uint32);
        let Some(&(start, count)) = lines.ranges.first() else { return };
        if count == 0 {
            return;
        }
        pass.draw_indexed(start..start + count, 0, 0..instance_count);
    }

    fn paint_ghost(
        &self,
        pass: &mut wgpu::RenderPass<'static>,
        mesh: &GpuMesh,
        colour: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        instance_count: u32,
    ) {
        pass.set_pipeline(&self.ghost_pipeline);
        pass.set_bind_group(0, &self.ghost_bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, colour.slice(..));
        pass.set_vertex_buffer(2, instances.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        let Some(part) = mesh.parts.first() else { return };
        pass.draw_indexed(part.indices.clone(), 0, 0..instance_count);
    }
}

fn place_matrix(offset: [f64; 3], rotation: [f64; 3]) -> [[f64; 4]; 4] {
    let (rx, ry, rz) = (rotation[0].to_radians(), rotation[1].to_radians(), rotation[2].to_radians());
    let rotate = |p: [f64; 3]| {
        let (mut x, mut y, mut z) = (p[0], p[1], p[2]);
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        (y, z) = (y * cx - z * sx, y * sx + z * cx);
        (x, z) = (x * cy + z * sy, -x * sy + z * cy);
        (x, y) = (x * cz - y * sz, x * sz + y * cz);
        [x, y, z]
    };
    let x = rotate([1.0, 0.0, 0.0]);
    let y = rotate([0.0, 1.0, 0.0]);
    let z = rotate([0.0, 0.0, 1.0]);
    [
        [x[0], x[1], x[2], 0.0],
        [y[0], y[1], y[2], 0.0],
        [z[0], z[1], z[2], 0.0],
        [offset[0], offset[1], offset[2], 1.0],
    ]
}

fn transform_matrix(transform: &DisplayTransform) -> [[f64; 4]; 4] {
    match transform {
        DisplayTransform::Place { offset, rotation } => place_matrix(*offset, *rotation),
        DisplayTransform::Mirror { plane, offset } => {
            let mut matrix = identity_f64();
            match *plane {
                "xz" => {
                    matrix[1][1] = -1.0;
                    matrix[3][1] = 2.0 * offset;
                }
                "yz" => {
                    matrix[0][0] = -1.0;
                    matrix[3][0] = 2.0 * offset;
                }
                _ => {
                    matrix[2][2] = -1.0;
                    matrix[3][2] = 2.0 * offset;
                }
            }
            matrix
        }
        DisplayTransform::Scale { factor } => {
            let mut matrix = identity_f64();
            matrix[0][0] = *factor;
            matrix[1][1] = *factor;
            matrix[2][2] = *factor;
            matrix
        }
    }
}

fn identity_f64() -> [[f64; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

fn multiply(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

pub(crate) fn instance_matrix(transforms: &[DisplayTransform]) -> [[f32; 4]; 4] {
    let mut matrix = identity_f64();
    for transform in transforms {
        matrix = multiply(transform_matrix(transform), matrix);
    }
    let mut out = [[0.0_f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = matrix[column][row] as f32;
        }
    }
    out
}

fn multiply_f32(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

struct TriangleLines {
    indices: wgpu::Buffer,
    /// Per plan part: index range in the line buffer. The part index is also the
    /// instance index for the draw.
    ranges: Vec<(u32, u32)>,
}

fn prepare_triangle_lines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &GpuMesh,
    batch: &[(std::sync::Arc<TessPlan>, PlanInstance, [[f32; 4]; 4])],
) -> Option<TriangleLines> {
    let mut indices: Vec<u32> = Vec::new();
    let mut ranges = Vec::with_capacity(batch.len());
    for (part_index, (plan, _, _)) in batch.iter().enumerate() {
        let Some(part) = mesh.parts.get(part_index) else {
            ranges.push((indices.len() as u32, 0));
            continue;
        };
        let start = indices.len() as u32;
        // A plan face's triangle indices are local to that face's vertices.
        // The GpuMesh stores every face's vertices in that face's order, so find
        // each face's first global vertex instead of assuming one part-wide base.
        let mut face_bases = std::collections::BTreeMap::new();
        for global in part.vertices.clone() {
            let Some(face) = mesh.vertex_face.get(global as usize) else { continue };
            if global >= part.vertices.start && global < part.vertices.end {
                face_bases.entry(*face).or_insert(global);
            }
        }
        for face in &plan.faces {
            let Some(&base) = face_bases.get(&face.face) else { continue };
            let poles: std::collections::BTreeSet<u32> =
                face.poles.iter().map(|pole| pole.node).collect();
            // A pole fan can be represented by several coincident vertices (one
            // per seam side), so face.poles alone does not catch every fan. A
            // vertex shared by many triangles is a fan centre; skip its
            // triangles so their converging edges cannot become a dark disc.
            let mut valence = vec![0u32; face.uv.len()];
            for triangle in &face.triangles {
                for node in triangle {
                    if let Some(count) = valence.get_mut(*node as usize) {
                        *count += 1;
                    }
                }
            }
            for triangle in &face.triangles {
                let [a, b, c] = *triangle;
                let fan_centre = [a, b, c].iter().any(|node| {
                    poles.contains(node)
                        || valence.get(*node as usize).is_some_and(|count| *count >= 10)
                });
                if fan_centre {
                    continue;
                }
                // Each triangle contributes its three boundary edges.
                for (from, to) in [(a, b), (b, c), (c, a)] {
                    let vertex_count = part.vertices.end - base;
                    if from >= vertex_count || to >= vertex_count {
                        continue;
                    }
                    indices.push(base + from);
                    indices.push(base + to);
                }
            }
        }
        ranges.push((start, indices.len() as u32 - start));
    }
    if indices.is_empty() {
        return None;
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport-plan-triangle-lines"),
        size: (indices.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&indices));
    Some(TriangleLines { indices: buffer, ranges })
}

struct ResidentMesh {
    /// Strong handle for the plan identity. Besides describing the mesh, this
    /// prevents a freed plan's address being reused by a different plan while
    /// the resident entry is still alive.
    plan: std::sync::Arc<TessPlan>,
    mesh: std::sync::Arc<GpuMesh>,
    lines: Option<std::sync::Arc<TriangleLines>>,
    colours: std::collections::HashMap<u64, std::sync::Arc<wgpu::Buffer>>,
    colour_order: std::collections::VecDeque<u64>,
    instances: std::collections::HashMap<(InstanceSlot, u64), InstanceBuffer>,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum InstanceSlot {
    Main,
    Ghost,
}

struct InstanceBuffer {
    buffer: std::sync::Arc<wgpu::Buffer>,
    capacity: u32,
}

#[derive(Debug, Default)]
struct Residency {
    entries: std::collections::HashMap<usize, (u64, u64)>,
    bytes: u64,
    tick: u64,
    budget: u64,
}

impl Residency {
    fn new(budget: u64) -> Self {
        Self { budget, ..Self::default() }
    }

    fn insert(&mut self, key: usize, bytes: u64) {
        self.tick = self.tick.wrapping_add(1);
        if let Some((old, _)) = self.entries.insert(key, (bytes, self.tick)) {
            self.bytes = self.bytes.saturating_sub(old);
        }
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn add_bytes(&mut self, key: usize, extra: u64) {
        if let Some((bytes, _)) = self.entries.get_mut(&key) {
            *bytes = bytes.saturating_add(extra);
            self.bytes = self.bytes.saturating_add(extra);
        }
    }

    fn touch(&mut self, key: usize) {
        if let Some((_, stamp)) = self.entries.get_mut(&key) {
            *stamp = self.tick;
        }
    }

    fn touch_set(&mut self, keys: &[usize]) {
        self.tick = self.tick.wrapping_add(1);
        for key in keys {
            self.touch(*key);
        }
    }

    fn remove_bytes(&mut self, key: usize, bytes: u64) {
        if let Some((entry, _)) = self.entries.get_mut(&key) {
            *entry = entry.saturating_sub(bytes);
        }
        self.bytes = self.bytes.saturating_sub(bytes);
    }

    fn remove(&mut self, key: usize) -> Option<u64> {
        let (bytes, _) = self.entries.remove(&key)?;
        self.bytes = self.bytes.saturating_sub(bytes);
        Some(bytes)
    }

    /// Evict unused entries oldest-first until under budget. Returns evicted keys.
    fn evict(&mut self, used: &std::collections::HashSet<usize>) -> Vec<usize> {
        if self.bytes <= self.budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, usize)> = self
            .entries
            .iter()
            .filter(|(key, _)| !used.contains(key))
            .map(|(key, (_, stamp))| (*stamp, *key))
            .collect();
        candidates.sort_unstable();
        let mut evicted = Vec::new();
        for (_, key) in candidates {
            if self.bytes <= self.budget {
                break;
            }
            if self.remove(key).is_some() {
                evicted.push(key);
            }
        }
        evicted
    }
}

struct DrawGroup {
    plan: std::sync::Arc<TessPlan>,
    face_colors: Vec<[f32; 3]>,
    matrices: Vec<[[f32; 4]; 4]>,
}

fn prepare_triangle_lines_for_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &GpuMesh,
    plan: &TessPlan,
) -> Option<TriangleLines> {
    let mut indices: Vec<u32> = Vec::new();
    // A resident mesh contains exactly one plan, so there is one part and one
    // line range; every instance reuses that range.
    let part = mesh.parts.first()?;
    let start = 0u32;
    let mut face_bases = std::collections::BTreeMap::new();
    for global in part.vertices.clone() {
        let Some(face) = mesh.vertex_face.get(global as usize) else { continue };
        face_bases.entry(*face).or_insert(global);
    }
    for face in &plan.faces {
        let Some(&base) = face_bases.get(&face.face) else { continue };
        let poles: std::collections::BTreeSet<u32> =
            face.poles.iter().map(|pole| pole.node).collect();
        let mut valence = vec![0u32; face.uv.len()];
        for triangle in &face.triangles {
            for node in triangle {
                if let Some(count) = valence.get_mut(*node as usize) {
                    *count += 1;
                }
            }
        }
        for triangle in &face.triangles {
            let [a, b, c] = *triangle;
            let fan_centre = [a, b, c].iter().any(|node| {
                poles.contains(node)
                    || valence.get(*node as usize).is_some_and(|count| *count >= 10)
            });
            if fan_centre {
                continue;
            }
            for (from, to) in [(a, b), (b, c), (c, a)] {
                let vertex_count = part.vertices.end - base;
                if from >= vertex_count || to >= vertex_count {
                    continue;
                }
                indices.push(base + from);
                indices.push(base + to);
            }
        }
    }
    if indices.is_empty() {
        return None;
    }
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport-plan-triangle-lines"),
        size: (indices.len() * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&indices));
    Some(TriangleLines { indices: buffer, ranges: vec![(start, indices.len() as u32)] })
}

fn resident_mesh(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    queue: &wgpu::Queue,
    tess: &TessGpu,
    cache: &mut std::collections::HashMap<usize, ResidentMesh>,
    plan: &std::sync::Arc<TessPlan>,
    triangles: bool,
    residency: &mut Residency,
) -> Option<std::sync::Arc<GpuMesh>> {
    let key = std::sync::Arc::as_ptr(plan) as usize;
    if let Some(resident) = cache.get_mut(&key) {
        residency.touch(key);
        if triangles && resident.lines.is_none() {
            resident.lines = prepare_triangle_lines_for_plan(device, queue, &resident.mesh, plan.as_ref())
                .map(std::sync::Arc::new);
            if let Some(lines) = resident.lines.as_ref() {
                let bytes = lines.indices.size() + lines.ranges.len() as u64 * 8;
                resident.bytes += bytes;
                residency.add_bytes(key, bytes);
            }
        }
        return Some(std::sync::Arc::clone(&resident.mesh));
    }
    let mesh = match tess.prepare(plan.as_ref()) {
        Ok(mesh) => mesh,
        Err(error) => {
            eprintln!("GPU plan preparation failed: {error}");
            return None;
        }
    };
    tess.record(encoder, &mesh);
    let mesh = std::sync::Arc::new(mesh);
    let lines = triangles
        .then(|| prepare_triangle_lines_for_plan(device, queue, &mesh, plan.as_ref()))
        .flatten()
        .map(std::sync::Arc::new);
    let lines_bytes = lines
        .as_ref()
        .map_or(0, |lines| lines.indices.size() + lines.ranges.len() as u64 * 8);
    let bytes = mesh.vertex_count as u64 * solidcraft_viewport_gpu::VERTEX_STRIDE
        + mesh.index_count as u64 * 4
        + lines_bytes;
    cache.insert(
        key,
        ResidentMesh {
            plan: std::sync::Arc::clone(plan),
            mesh: std::sync::Arc::clone(&mesh),
            lines,
            colours: std::collections::HashMap::new(),
            colour_order: std::collections::VecDeque::new(),
            instances: std::collections::HashMap::new(),
            bytes,
        },
    );
    residency.insert(key, bytes);
    Some(mesh)
}

fn prepare_plan_chunks(
    tess: &TessGpu,
    encoder: &mut wgpu::CommandEncoder,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut std::collections::HashMap<usize, ResidentMesh>,
    plans: &[PlanInstance],
    colour: Option<[f32; 3]>,
    alpha: f32,
    extra: Option<[[f32; 4]; 4]>,
    triangles: bool,
    pass: InstanceSlot,
    residency: &mut Residency,
) -> Vec<PlanChunk> {
    let mut groups: Vec<DrawGroup> = Vec::new();
    for instance in plans {
        let matrix = instance_matrix(&instance.transforms);
        let matrix = extra.map_or(matrix, |extra| multiply_f32(extra, matrix));
        if let Some(group) = groups.iter_mut().find(|group| {
            std::sync::Arc::ptr_eq(&group.plan, &instance.plan)
                && group.face_colors == instance.face_colors
        }) {
            group.matrices.push(matrix);
        } else {
            groups.push(DrawGroup {
                plan: std::sync::Arc::clone(&instance.plan),
                face_colors: instance.face_colors.clone(),
                matrices: vec![matrix],
            });
        }
    }
    let mut chunks = Vec::with_capacity(groups.len());
    for group in groups {
        let Some(mesh) = resident_mesh(
            device,
            encoder,
            queue,
            tess,
            cache,
            &group.plan,
            triangles,
            residency,
        ) else {
            continue;
        };
        let colour_hash = colour_cache_key(mesh.vertex_count, colour, &group.face_colors);
        let plan_key = std::sync::Arc::as_ptr(&group.plan) as usize;
        let cached_colour = cache
            .get(&plan_key)
            .and_then(|resident| resident.colours.get(&colour_hash))
            .cloned();
        let colour_buffer = if let Some(buffer) = cached_colour {
            // A hit is MRU: keep an in-use colouring from being evicted.
            if let Some(resident) = cache.get_mut(&plan_key) {
                resident.colour_order.retain(|key| *key != colour_hash);
                resident.colour_order.push_back(colour_hash);
            }
            buffer
        } else {
            let mut colours = vec![[0.17_f32, 0.18, 0.195]; mesh.vertex_count as usize];
            for (index, slot) in colours.iter_mut().enumerate() {
                let face = mesh.vertex_face.get(index).copied().unwrap_or(0) as usize;
                *slot = colour.unwrap_or_else(|| {
                    group.face_colors.get(face).copied().unwrap_or([0.17, 0.18, 0.195])
                });
            }
            let padded: Vec<[f32; 4]> =
                colours.into_iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
            let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-plan-colours"),
                size: (padded.len() * std::mem::size_of::<[f32; 4]>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&padded));
            let bytes = buffer.size();
            if let Some(resident) = cache.get_mut(&plan_key) {
                resident.colours.insert(colour_hash, std::sync::Arc::clone(&buffer));
                resident.colour_order.push_back(colour_hash);
                resident.bytes = resident.bytes.saturating_add(bytes);
                // Keep a tiny colouring LRU; slider drags must not grow the
                // resident mesh without bound.
                while resident.colour_order.len() > 3 {
                    let Some(old) = resident.colour_order.pop_front() else { break };
                    if old == colour_hash {
                        resident.colour_order.push_back(old);
                        break;
                    }
                    if let Some(removed) = resident.colours.remove(&old) {
                        // Active chunks may still hold an Arc to this buffer;
                        // the accounting is conservative but safe.
                        let freed = removed.size();
                        resident.bytes = resident.bytes.saturating_sub(freed);
                        residency.remove_bytes(plan_key, freed);
                    }
                }
            }
            residency.add_bytes(plan_key, bytes);
            buffer
        };
        let instances: Vec<GpuInstance> = group
            .matrices
            .iter()
            .map(|matrix| GpuInstance::from_model(*matrix, alpha))
            .collect();
        let instance_hash = if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            use std::hash::{Hash, Hasher};
            let mut instance_hasher = std::collections::hash_map::DefaultHasher::new();
            for instance in &instances {
                bytemuck::bytes_of(instance).hash(&mut instance_hasher);
            }
            instance_hasher.finish()
        } else {
            0
        };
        let slot_key = (pass, colour_hash);
        let needed = instances.len() as u32;
        let instance_bytes = std::mem::size_of::<GpuInstance>() as u64;
        let instance_buffer = if let Some(slot) = cache
            .get_mut(&plan_key)
            .and_then(|resident| resident.instances.get_mut(&slot_key))
        {
            if slot.capacity < needed {
                let bytes = u64::from(needed) * instance_bytes;
                let old = u64::from(slot.capacity) * instance_bytes;
                let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-plan-instances"),
                    size: bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                slot.buffer = std::sync::Arc::clone(&buffer);
                slot.capacity = needed;
                if bytes > old {
                    if let Some(resident) = cache.get_mut(&plan_key) {
                        resident.bytes = resident.bytes.saturating_add(bytes - old);
                    }
                    residency.add_bytes(plan_key, bytes - old);
                }
                buffer
            } else {
                std::sync::Arc::clone(&slot.buffer)
            }
        } else {
            let bytes = u64::from(needed) * instance_bytes;
            let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-plan-instances"),
                size: bytes.max(instance_bytes),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            if let Some(resident) = cache.get_mut(&plan_key) {
                resident.instances.insert(
                    slot_key,
                    InstanceBuffer {
                        buffer: std::sync::Arc::clone(&buffer),
                        capacity: needed.max(1),
                    },
                );
                resident.bytes = resident.bytes.saturating_add(buffer.size());
            }
            residency.add_bytes(plan_key, buffer.size());
            buffer
        };
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        let key = std::sync::Arc::as_ptr(&group.plan) as usize;
        let lines = cache.get(&key).and_then(|resident| resident.lines.clone());
        chunks.push(PlanChunk {
            mesh,
            colour: colour_buffer,
            instances: instance_buffer,
            instance_count: instances.len() as u32,
            instance_hash,
            lines,
        });
    }
    // Keep resident meshes across passes so pose and ghost-matrix edits only
    // rebuild instance/colour buffers. Eviction happens in `PlanState::prepare`
    // after both the normal and ghost passes have marked their plans used.
    chunks
}

struct PlanChunk {
    mesh: std::sync::Arc<GpuMesh>,
    colour: std::sync::Arc<wgpu::Buffer>,
    instances: std::sync::Arc<wgpu::Buffer>,
    instance_count: u32,
    instance_hash: u64,
    lines: Option<std::sync::Arc<TriangleLines>>,
}

const RESIDENT_BUDGET: u64 = 512 << 20;
const UPLOAD_VERTEX_BUDGET: usize = 2_000_000;
const UPLOAD_TIME_BUDGET: std::time::Duration = std::time::Duration::from_millis(4);

fn hash_f64(hash: u64, value: f64) -> u64 {
    hash.rotate_left(5) ^ value.to_bits()
}

pub(crate) fn plan_identity_hash(plans: &[PlanInstance]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for instance in plans {
        instance.target.hash(&mut hasher);
        (std::sync::Arc::as_ptr(&instance.plan) as usize).hash(&mut hasher);
        let mut transform_hash = 0u64;
        for transform in &instance.transforms {
            transform_hash = match transform {
                DisplayTransform::Place { offset, rotation } => {
                    let mut h = transform_hash;
                    for value in offset.iter().chain(rotation.iter()) {
                        h = hash_f64(h, *value);
                    }
                    h.rotate_left(1) ^ 1
                }
                DisplayTransform::Mirror { plane, offset } => {
                    let mut h = hash_f64(transform_hash, *offset);
                    h ^= plane.len() as u64 ^ 2;
                    h
                }
                DisplayTransform::Scale { factor } => {
                    hash_f64(transform_hash, *factor).rotate_left(1) ^ 3
                }
            };
        }
        transform_hash.hash(&mut hasher);
        for colour in &instance.face_colors {
            for value in colour {
                value.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn plans_ready(plans: &[PlanInstance], prepared: &std::collections::HashSet<usize>) -> bool {
    plans.iter().all(|instance| {
        prepared.contains(&(std::sync::Arc::as_ptr(&instance.plan) as usize))
    })
}

fn colour_cache_key(
    vertex_count: u32,
    colour: Option<[f32; 3]>,
    face_colors: &[[f32; 3]],
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    vertex_count.hash(&mut hasher);
    if let Some(value) = colour {
        for channel in value {
            channel.to_bits().hash(&mut hasher);
        }
    }
    for value in face_colors {
        for channel in value {
            channel.to_bits().hash(&mut hasher);
        }
    }
    hasher.finish()
}

fn same_plan_set(a: &[PlanInstance], b: &[PlanInstance]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(old, new)| {
            old.target == new.target
                && std::sync::Arc::ptr_eq(&old.plan, &new.plan)
                && old.transforms == new.transforms
                && old.face_colors == new.face_colors
        })
}

struct PendingPlans {
    plans: Vec<PlanInstance>,
    enabled: bool,
    colour: Option<[f32; 3]>,
    triangles: bool,
    prepared: std::collections::HashSet<usize>,
    failed: std::collections::HashSet<usize>,
}

/// GPU plan state shared between the viewport callback and `GpuViewport`.
pub(crate) struct PlanState {
    renderer: Option<PlanRenderer>,
    tess: Option<TessGpu>,
    tess_ready: Option<std::sync::Arc<std::sync::OnceLock<TessGpu>>>,
    plans: Vec<PlanInstance>,
    chunks: Vec<PlanChunk>,
    /// New geometry is prepared gradually while the previous chunks stay
    /// visible; the swap happens only when every new plan is resident.
    pending: Option<PendingPlans>,
    resident: std::collections::HashMap<usize, ResidentMesh>,
    residency: Residency,
    last_prepare_us: u64,
    last_uploads: u32,
    painted_chunks: std::sync::atomic::AtomicU32,
    painted_instances: std::sync::atomic::AtomicU32,
    preview: Option<[[f32; 4]; 4]>,
    colour_override: Option<[f32; 3]>,
    dirty: bool,
    enabled: bool,
    translucent: bool,
    triangles: bool,
    /// Ghost plans are drawn by the same GPU mesher with a blended pipeline.
    ghost_plans: Vec<PlanInstance>,
    ghost_chunks: Vec<PlanChunk>,
    ghost_colour: Option<[f32; 3]>,
    ghost_alpha: f32,
    ghost_dirty: bool,
    ghost_enabled: bool,
    ghost_matrix: Option<[[f32; 4]; 4]>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PlanDebug {
    pub enabled: bool,
    pub pending: bool,
    pub chunks: usize,
    pub plans: usize,
    pub triangles: u32,
    pub instances: u32,
    pub failed: usize,
    pub plan_hash: u64,
    pub instance_hash: u64,
    pub prepare_us: u64,
    pub uploads: u32,
    pub painted_chunks: u32,
    pub painted_instances: u32,
    pub resident_count: usize,
    pub resident_bytes: u64,
}

impl PlanState {
    pub(crate) fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
        reversed: bool,
        tess_ready: Option<std::sync::Arc<std::sync::OnceLock<TessGpu>>>,
    ) -> Self {
        Self {
            renderer: Some(PlanRenderer::install(device, format, samples, depth, reversed)),
            tess: tess_ready.as_ref().and_then(|ready| ready.get().cloned()),
            tess_ready,
            plans: Vec::new(),
            chunks: Vec::new(),
            pending: None,
            resident: std::collections::HashMap::new(),
            residency: Residency::new(RESIDENT_BUDGET),
            last_prepare_us: 0,
            last_uploads: 0,
            painted_chunks: std::sync::atomic::AtomicU32::new(0),
            painted_instances: std::sync::atomic::AtomicU32::new(0),
            preview: None,
            colour_override: None,
            dirty: true,
            enabled: false,
            translucent: false,
            triangles: false,
            ghost_plans: Vec::new(),
            ghost_chunks: Vec::new(),
            ghost_colour: None,
            ghost_alpha: 0.45,
            ghost_dirty: false,
            ghost_enabled: false,
            ghost_matrix: None,
        }
    }

    fn touch_resident(&mut self, key: usize) {
        self.residency.touch(key);
    }

    pub(crate) fn set_plans(&mut self, plans: Vec<PlanInstance>, enabled: bool, colour: Option<[f32; 3]>) {
        // The display plan cache reuses the same plan Arcs across evaluations.
        // Keep the resident GpuMesh when neither the geometry nor the draw mode
        // changed; only a genuinely new plan/quality needs GPU preparation.
        // Do not restart an in-progress staged swap when the same request is
        // republished on the next frame.
        if let Some(pending) = self.pending.as_ref()
            && same_plan_set(&pending.plans, &plans)
            && pending.enabled == enabled
            && pending.colour == colour
        {
            return;
        }
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[plan] set {} active {} chunks {} pending={}",
                plans.len(),
                self.plans.len(),
                self.chunks.len(),
                self.pending.is_some(),
            );
        }
        let same_plans = same_plan_set(&self.plans, &plans);
        if same_plans && self.enabled == enabled && self.colour_override == colour {
            // An edit-back must cancel a still-pending future set; otherwise the
            // stale pending swap would activate after this frame.
            self.pending = None;
            self.dirty = true;
            return;
        }
        // Keep preparation progress for plans shared with the replaced set.
        let prepared = self
            .pending
            .as_ref()
            .map(|pending| {
                pending
                    .prepared
                    .iter()
                    .copied()
                    .filter(|key| {
                        plans.iter().any(|instance| {
                            std::sync::Arc::as_ptr(&instance.plan) as usize == *key
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.pending = Some(PendingPlans {
            plans,
            enabled,
            colour,
            triangles: self.triangles,
            prepared,
            failed: std::collections::HashSet::new(),
        });
        if let Some(pending) = self.pending.as_ref() {
            let keys: Vec<usize> = pending
                .plans
                .iter()
                .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
                .collect();
            self.residency.touch_set(&keys);
        }
    }

    pub(crate) fn set_translucent(&mut self, translucent: bool) {
        self.translucent = translucent;
    }

    pub(crate) fn set_triangles(&mut self, enabled: bool) {
        if self.triangles == enabled {
            return;
        }
        self.triangles = enabled;
        if let Some(pending) = self.pending.as_mut() {
            pending.triangles = enabled;
        } else {
            self.dirty = true;
        }
    }

    pub(crate) fn set_ghost_plans(
        &mut self,
        plans: Vec<PlanInstance>,
        alpha: f32,
        colour: Option<[f32; 3]>,
        enabled: bool,
    ) {
        let same = self.ghost_enabled == enabled
            && (self.ghost_alpha - alpha).abs() < f32::EPSILON
            && self.ghost_colour == colour
            && self.ghost_plans.len() == plans.len()
            && self.ghost_plans.iter().zip(&plans).all(|(old, new)| {
                old.target == new.target
                    && std::sync::Arc::ptr_eq(&old.plan, &new.plan)
                    && old.transforms == new.transforms
                    && old.face_colors == new.face_colors
            });
        if same {
            return;
        }
        let keys: Vec<usize> = plans
            .iter()
            .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
            .collect();
        self.residency.touch_set(&keys);
        self.ghost_plans = plans;
        self.ghost_alpha = alpha;
        self.ghost_colour = colour;
        self.ghost_enabled = enabled;
        self.ghost_dirty = true;
    }

    pub(crate) fn set_ghost_matrix(&mut self, matrix: Option<[[f32; 4]; 4]>) {
        if self.ghost_matrix == matrix {
            return;
        }
        self.ghost_matrix = matrix;
    }

    /// Drop the outgoing document's draw state now, without staging an empty
    /// pending set. The next document then activates with an unlimited budget.
    pub(crate) fn clear_document(&mut self) {
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[plan] clear document");
        }
        self.pending = None;
        self.plans.clear();
        self.chunks.clear();
        self.enabled = false;
        self.colour_override = None;
        self.clear_transient();
        self.dirty = false;
        // Resident meshes deliberately survive; they hold their plan Arcs and
        // are bounded by the LRU budget (Step 2).
    }

    pub(crate) fn clear_transient(&mut self) {
        self.preview = None;
        self.dirty = true;
        self.ghost_matrix = None;
        self.ghost_plans.clear();
        self.ghost_chunks.clear();
        self.ghost_enabled = false;
        self.ghost_dirty = false;
    }

    pub(crate) fn is_available(&self) -> bool {
        self.enabled && !self.chunks.is_empty()
    }

    pub(crate) fn debug_counts(&self) -> PlanDebug {
        let triangles = self.chunks.iter().map(|chunk| chunk.mesh.index_count / 3).sum();
        let instances = self.chunks.iter().map(|chunk| chunk.instance_count).sum();
        let failed = self.pending.as_ref().map_or(0, |pending| pending.failed.len());
        let mut instance_hash = 0u64;
        for chunk in &self.chunks {
            instance_hash = instance_hash.rotate_left(5) ^ chunk.instance_hash;
        }
        PlanDebug {
            enabled: self.enabled,
            pending: self.pending.is_some(),
            chunks: self.chunks.len(),
            plans: self.plans.len(),
            triangles,
            instances,
            failed,
            plan_hash: plan_identity_hash(&self.plans),
            instance_hash,
            prepare_us: self.last_prepare_us,
            uploads: self.last_uploads,
            painted_chunks: self.painted_chunks.load(std::sync::atomic::Ordering::Relaxed),
            painted_instances: self.painted_instances.load(std::sync::atomic::Ordering::Relaxed),
            resident_count: self.resident.len(),
            resident_bytes: self.residency.bytes,
        }
    }

    pub(crate) fn tess_ready(&self) -> bool {
        self.tess.is_some() || self
            .tess_ready
            .as_ref()
            .is_some_and(|ready| ready.get().is_some())
    }

    pub(crate) fn set_preview_transform(&mut self, preview: Option<[[f32; 4]; 4]>) {
        if self.preview == preview {
            return;
        }
        self.preview = preview;
        // The preview is a shader uniform; no instance/colour rebuild needed.
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &ObjectUniforms,
    ) {
        let started = std::time::Instant::now();
        self.last_uploads = 0;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.prepare(queue, uniforms, self.preview, self.ghost_matrix);
        }
        if self.tess.is_none() {
            self.tess = self.tess_ready.as_ref().and_then(|ready| ready.get().cloned());
        }
        let Some(tess) = self.tess.clone() else {
            self.chunks.clear();
            self.ghost_chunks.clear();
            return;
        };
        if let Some(mut pending) = self.pending.take() {
            if !pending.enabled || pending.plans.is_empty() {
                self.plans = pending.plans;
                self.enabled = pending.enabled;
                self.colour_override = pending.colour;
                self.chunks.clear();
                self.dirty = false;
            } else {
                // Spread first-time plan preparation over frames so a quality
                // swap never stalls one render callback. Keep drawing the old
                // chunks until every new plan is resident.
                let unlimited = self.chunks.is_empty();
                let mut uploaded = 0usize;
                let mut uploaded_vertices = 0usize;
                for instance in &pending.plans {
                    if !unlimited
                        && uploaded > 0
                        && (uploaded_vertices >= UPLOAD_VERTEX_BUDGET
                            || started.elapsed() >= UPLOAD_TIME_BUDGET)
                    {
                        break;
                    }
                    let key = std::sync::Arc::as_ptr(&instance.plan) as usize;
                    if pending.prepared.contains(&key) || pending.failed.contains(&key) {
                        continue;
                    }
                    if self.resident.contains_key(&key) {
                        // A cache hit needs no upload work and does not consume
                        // the per-frame preparation budget.
                        pending.prepared.insert(key);
                        continue;
                    }
                    if resident_mesh(
                        device,
                        encoder,
                        queue,
                        &tess,
                        &mut self.resident,
                        &instance.plan,
                        pending.triangles,
                        &mut self.residency,
                    )
                    .is_some()
                    {
                        pending.prepared.insert(key);
                        uploaded += 1;
                        uploaded_vertices += instance.plan.vertex_count();
                        self.last_uploads = self.last_uploads.saturating_add(1);
                        self.touch_resident(key);
                    } else {
                        // A refused plan must not block the whole swap forever.
                        // It is settled as failed and the remaining plans can
                        // activate; diagnostics still report it.
                        pending.failed.insert(key);
                        pending.prepared.insert(key);
                    }
                }
                let ready = plans_ready(&pending.plans, &pending.prepared);
                if ready {
                    if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                        eprintln!(
                            "[plan] activate {} instances {} uploads {}",
                            pending.plans.len(),
                            pending.plans.len(),
                            self.last_uploads,
                        );
                    }
                    self.chunks = prepare_plan_chunks(
                        &tess,
                        encoder,
                        device,
                        queue,
                        &mut self.resident,
                        &pending.plans,
                        pending.colour,
                        1.0,
                        None,
                        pending.triangles,
                        InstanceSlot::Main,
                        &mut self.residency,
                    );
                    self.plans = pending.plans;
                    self.enabled = pending.enabled;
                    self.colour_override = pending.colour;
                    self.dirty = false;
                    let keys: Vec<usize> = self
                        .plans
                        .iter()
                        .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
                        .collect();
                    self.residency.touch_set(&keys);
                } else {
                    self.pending = Some(pending);
                }
            }
        } else if self.enabled && self.dirty {
            self.dirty = false;
            self.chunks = prepare_plan_chunks(
                &tess,
                encoder,
                device,
                queue,
                &mut self.resident,
                &self.plans,
                self.colour_override,
                1.0,
                None,
                self.triangles,
                InstanceSlot::Main,
                &mut self.residency,
            );
        } else if !self.enabled {
            self.chunks.clear();
        }
        if self.ghost_dirty {
            self.ghost_dirty = false;
            self.ghost_chunks = if self.ghost_enabled {
                prepare_plan_chunks(
                    &tess,
                    encoder,
                    device,
                    queue,
                    &mut self.resident,
                    &self.ghost_plans,
                    self.ghost_colour,
                    self.ghost_alpha,
                    None,
                    false,
                    InstanceSlot::Ghost,
                    &mut self.residency,
                )
            } else {
                Vec::new()
            };
            let keys: Vec<usize> = self
                .ghost_plans
                .iter()
                .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
                .collect();
            self.residency.touch_set(&keys);
        }
        // Bytes and recency are maintained by `Residency`; evict only when
        // over budget, oldest unused first.
        if self.residency.bytes > self.residency.budget {
            let mut used = std::collections::HashSet::new();
            for instance in self.plans.iter().chain(self.ghost_plans.iter()) {
                used.insert(std::sync::Arc::as_ptr(&instance.plan) as usize);
            }
            if let Some(pending) = self.pending.as_ref() {
                for instance in &pending.plans {
                    used.insert(std::sync::Arc::as_ptr(&instance.plan) as usize);
                }
            }
            for key in self.residency.evict(&used) {
                self.resident.remove(&key);
                if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                    eprintln!("[plan] evict {key}");
                }
            }
        }
        self.last_prepare_us = started.elapsed().as_micros() as u64;
    }

    pub(crate) fn paint(&self, pass: &mut wgpu::RenderPass<'static>) {
        let Some(renderer) = self.renderer.as_ref() else { return };
        self.painted_chunks.store(
            self.chunks.len() as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.painted_instances.store(
            self.chunks.iter().map(|chunk| chunk.instance_count).sum(),
            std::sync::atomic::Ordering::Relaxed,
        );
        for chunk in &self.chunks {
            renderer.paint(pass, &chunk.mesh, &chunk.colour, &chunk.instances, self.translucent, chunk.instance_count);
        }
        if self.triangles {
            for chunk in &self.chunks {
                if let Some(lines) = &chunk.lines {
                    renderer.paint_triangle_edges(pass, &chunk.mesh, &chunk.instances, lines, chunk.instance_count);
                }
            }
        }
        for chunk in &self.ghost_chunks {
            renderer.paint_ghost(pass, &chunk.mesh, &chunk.colour, &chunk.instances, chunk.instance_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance() -> PlanInstance {
        PlanInstance {
            target: String::new(),
            plan: std::sync::Arc::new(TessPlan::default()),
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(Vec::new()),
        }
    }

    #[test]
    fn instance_cache_slots_include_pass_and_colour() {
        let a = (InstanceSlot::Main, 1);
        let b = (InstanceSlot::Main, 2);
        let c = (InstanceSlot::Ghost, 1);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn colour_key_changes_with_colours_and_is_stable_otherwise() {
        let faces = [[0.2, 0.3, 0.4]];
        assert_eq!(
            colour_cache_key(8, None, &faces),
            colour_cache_key(8, None, &faces)
        );
        assert_ne!(
            colour_cache_key(8, None, &faces),
            colour_cache_key(8, Some([1.0, 0.0, 0.0]), &faces)
        );
        assert_ne!(
            colour_cache_key(8, None, &faces),
            colour_cache_key(8, None, &[[0.2, 0.3, 0.5]])
        );
    }

    #[test]
    fn residency_remove_bytes_updates_the_total() {
        let mut residency = Residency::new(100);
        residency.insert(1, 40);
        residency.remove_bytes(1, 10);
        assert_eq!(residency.bytes, 30);
    }

    #[test]
    fn residency_keeps_the_most_recent_entry_across_a_b_a() {
        let mut residency = Residency::new(130);
        residency.insert(0, 30); // C, oldest
        residency.insert(1, 60); // A
        residency.touch_set(&[1]);
        residency.insert(2, 60); // B
        residency.touch_set(&[2]);
        let used = std::collections::HashSet::from([2]);
        assert_eq!(residency.evict(&used), vec![0]);
        assert!(residency.entries.contains_key(&1));
        assert!(residency.entries.contains_key(&2));
        assert!(residency.bytes <= residency.budget);
    }

    #[test]
    fn residency_never_evicts_used_entries() {
        let mut residency = Residency::new(10);
        residency.insert(1, 20);
        let used = std::collections::HashSet::from([1]);
        assert!(residency.evict(&used).is_empty());
        assert_eq!(residency.bytes, 20);
    }

    #[test]
    fn residency_bytes_follow_insert_and_remove() {
        let mut residency = Residency::new(100);
        residency.insert(1, 40);
        residency.add_bytes(1, 10);
        assert_eq!(residency.bytes, 50);
        residency.remove(1);
        assert_eq!(residency.bytes, 0);
    }

    #[test]
    fn readiness_requires_every_plan_and_refused_plans_settle() {
        let a = instance();
        let b = instance();
        let mut prepared = std::collections::HashSet::new();
        prepared.insert(std::sync::Arc::as_ptr(&a.plan) as usize);
        assert!(!plans_ready(&[a.clone(), b.clone()], &prepared));
        prepared.insert(std::sync::Arc::as_ptr(&b.plan) as usize);
        assert!(plans_ready(&[a, b], &prepared));
    }
}
