//! GPU evaluation and rendering of `solidcore_tess::TessPlan`s.
//!
//! Plans come from the engine worker with local-body transforms. This module
//! evaluates the plans with `solidcraft-viewport-gpu`'s `TessGpu`, applies each
//! body's placement/mirror/scale as a per-plan instance matrix, and draws the
//! resulting `GpuMesh` through NodeCraft's existing PBR shader. B-rep edges are
//! uploaded once per plan in local space and drawn as screen-width ribbons with
//! the same per-instance placement matrices; picking reuses those buffers and
//! the same ribbon vertex function. The CPU object path remains only as the
//! fallback for plans the GPU mesher refuses.

use egui_wgpu::wgpu;
use solidcraft_viewport_gpu::{GpuMesh, TessGpu};
use nodecraft_layers::tess::TessPlan;

use nodecraft_display::PlanInstance;
use crate::edge_ribbon::{
    EdgeCpuBatch, EdgeRenderer, EdgeStyle, EdgeUniforms, GpuEdgeBatch,
};
use crate::gpu_viewport::GhostPickData;
use crate::plan_math::{instance_matrix, multiply_f32, plan_identity_hash};

use crate::gpu_viewport::{
    edge_depth, face_colour, ghost_face_depth, identity_matrix, object_instance_layout,
    opaque_face_depth, plan_object_shader, translucent_face_depth, GpuInstance, ObjectUniforms,
    DEFAULT_FACE_COLOUR, GHOST_SHADER,
};

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

/// The one edge-width source. Every edge kind's width is a ratio of
/// `ViewSettings::edge_width_px`, using the ratios the previous fixed widths
/// had (standard 1.0, ghost 2.0, selected 3.0, hover 2.5).
pub(crate) fn edge_widths(base_px: f32) -> (f32, f32, f32, f32) {
    (
        base_px,
        base_px * 2.0,
        base_px * 3.0,
        base_px * 2.5,
    )
}

fn theme_colour(colour: [u8; 4]) -> [f32; 4] {
    [
        f32::from(colour[0]) / 255.0,
        f32::from(colour[1]) / 255.0,
        f32::from(colour[2]) / 255.0,
        f32::from(colour[3]) / 255.0,
    ]
}

/// Plan vertex layout in `GpuMesh`: position, normal.
pub(crate) fn plan_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
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

/// Per-vertex plan pick id, read by both the pick pass and the shaded plan
/// shader. Its low 20 bits are the B-rep face the colour table is indexed by.
pub(crate) fn plan_face_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: 2,
    }];
    wgpu::VertexBufferLayout {
        array_stride: 4,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

/// Per-instance global body index. The plan shader uses it to find that body's
/// slice in the shared colour table; the pick pass uses the same value for the
/// `body:N` prefix.
pub(crate) fn plan_body_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRS: [wgpu::VertexAttribute; 1] = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: 12,
    }];
    wgpu::VertexBufferLayout {
        array_stride: 4,
        step_mode: wgpu::VertexStepMode::Instance,
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
    /// Layout for the shared colour table: one storage array of colours and one
    /// of per-body offsets. Tables are created per plan set, not per chunk.
    colour_bind_layout: wgpu::BindGroupLayout,
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
            source: wgpu::ShaderSource::Wgsl(plan_object_shader().into()),
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
        let colour_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-plan-colour-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-plan-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_layout), Some(&colour_bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-plan-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(plan_vertex_layout()), Some(plan_face_layout()), Some(object_instance_layout()), Some(plan_body_layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(opaque_face_depth(reversed).state(depth)),
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
                buffers: &[Some(plan_vertex_layout()), Some(plan_face_layout()), Some(object_instance_layout()), Some(plan_body_layout())],
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
            depth_stencil: Some(translucent_face_depth(reversed).state(depth)),
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
            depth_stencil: Some(edge_depth(reversed).state(depth)),
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
                buffers: &[Some(plan_vertex_layout()), Some(plan_face_layout()), Some(object_instance_layout()), Some(plan_body_layout())],
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
            depth_stencil: Some(ghost_face_depth(reversed).state(depth)),
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
            colour_bind_layout,
        }
    }

    /// Create the single shared colour table for a plan set. The table is a
    /// storage array of `vec4` colours plus one `u32` start offset per body;
    /// the shader indexes `colours[offsets[body] + face]`.
    fn create_colour_table(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        colours: &[[f32; 4]],
        offsets: &[u32],
    ) -> PlanColourTable {
        let colours_size = (colours.len().max(1) * std::mem::size_of::<[f32; 4]>()) as u64;
        let offsets_size = (offsets.len().max(1) * std::mem::size_of::<u32>()) as u64;
        let colour_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-plan-colour-table"),
            size: colours_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&colour_buffer, 0, bytemuck::cast_slice(colours));
        let offset_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-plan-colour-offsets"),
            size: offsets_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&offset_buffer, 0, bytemuck::cast_slice(offsets));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport-plan-colour-bind-group"),
            layout: &self.colour_bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: colour_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offset_buffer.as_entire_binding() },
            ],
        });
        PlanColourTable {
            colours: colour_buffer,
            offsets: offset_buffer,
            bind_group,
            colour_capacity: colours.len().max(1) as u32,
            offset_capacity: offsets.len().max(1) as u32,
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
        pass: &mut wgpu::RenderPass<'_>,
        mesh: &GpuMesh,
        face_ids: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        bodies: &wgpu::Buffer,
        table: &PlanColourTable,
        translucent: bool,
        instance_count: u32,
    ) {
        pass.set_pipeline(if translucent { &self.transparent_pipeline } else { &self.pipeline });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &table.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, face_ids.slice(..));
        pass.set_vertex_buffer(2, instances.slice(..));
        pass.set_vertex_buffer(3, bodies.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        let Some(part) = mesh.parts.first() else { return };
        pass.draw_indexed(part.indices.clone(), 0, 0..instance_count);
    }

    fn paint_triangle_edges(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh: &GpuMesh,
        instances: &wgpu::Buffer,
        table: &PlanColourTable,
        lines: &TriangleLines,
        instance_count: u32,
    ) {
        pass.set_pipeline(&self.triangle_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, &table.bind_group, &[]);
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
        pass: &mut wgpu::RenderPass<'_>,
        mesh: &GpuMesh,
        face_ids: &wgpu::Buffer,
        instances: &wgpu::Buffer,
        bodies: &wgpu::Buffer,
        table: &PlanColourTable,
        instance_count: u32,
    ) {
        pass.set_pipeline(&self.ghost_pipeline);
        pass.set_bind_group(0, &self.ghost_bind_group, &[]);
        pass.set_bind_group(1, &table.bind_group, &[]);
        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
        pass.set_vertex_buffer(1, face_ids.slice(..));
        pass.set_vertex_buffer(2, instances.slice(..));
        pass.set_vertex_buffer(3, bodies.slice(..));
        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
        let Some(part) = mesh.parts.first() else { return };
        pass.draw_indexed(part.indices.clone(), 0, 0..instance_count);
    }
}

struct TriangleLines {
    indices: wgpu::Buffer,
    /// Per plan part: index range in the line buffer. The part index is also the
    /// instance index for the draw.
    ranges: Vec<(u32, u32)>,
}

/// Build the triangle-edge index list shared by the batch and resident line
/// overlays.
///
/// `vertex_face` is `GpuMesh::vertex_face` and `part_vertices` is the plan's
/// slice of it. A plan face's triangle indices are local to that face's
/// vertices, and the GPU mesh stores every face's vertices contiguously in plan
/// order, so recover each face's first global vertex instead of assuming one
/// part-wide base.
fn triangle_line_indices(
    plan: &TessPlan,
    vertex_face: &[u32],
    part_vertices: std::ops::Range<u32>,
) -> Vec<u32> {
    let mut indices: Vec<u32> = Vec::new();
    let mut face_bases = std::collections::BTreeMap::new();
    for global in part_vertices.clone() {
        let Some(face) = vertex_face.get(global as usize) else { continue };
        face_bases.entry(*face).or_insert(global);
    }
    for face in &plan.faces {
        let Some(&base) = face_bases.get(&face.face) else { continue };
        let vertex_count = part_vertices.end - base;
        // Every triangle edge is legitimate displayed geometry: the pole fan is
        // a real part of the display mesh, not an artifact to hide. The old
        // valence >= 10 / declared-pole skip blanked the whole sphere cap; the
        // fan's converging edges only looked like a dark disc because shared
        // triangle edges were emitted twice. Emit each undirected edge once per
        // face so duplicate overdraw goes away without hiding any geometry.
        let mut edges = std::collections::BTreeSet::new();
        for triangle in &face.triangles {
            let [a, b, c] = *triangle;
            for (from, to) in [(a, b), (b, c), (c, a)] {
                edges.insert((from.min(to), from.max(to)));
            }
        }
        for (from, to) in edges {
            if from >= vertex_count || to >= vertex_count {
                continue;
            }
            indices.push(base + from);
            indices.push(base + to);
        }
    }
    indices
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
        indices.extend_from_slice(&triangle_line_indices(
            plan,
            &mesh.vertex_face,
            part.vertices.clone(),
        ));
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
    /// Local-space edge ribbons, uploaded once for this plan and shared by
    /// every body instance.
    edges: Option<std::sync::Arc<GpuEdgeBatch>>,
    lines: Option<std::sync::Arc<TriangleLines>>,
    /// One buffer pair per pass. A plan is one instanced draw per pass now that
    /// face colouring is looked up in the shared table.
    instances: std::collections::HashMap<InstanceSlot, InstanceBuffer>,
    /// Per-vertex face ids, keyed by whether the display has multiple bodies.
    /// The body itself is per instance, so it is not part of this key.
    faces: std::collections::HashMap<u64, std::sync::Arc<wgpu::Buffer>>,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum InstanceSlot {
    Main,
    Ghost,
}

struct InstanceBuffer {
    /// One `GpuInstance` per body of the group, in body order.
    buffer: std::sync::Arc<wgpu::Buffer>,
    /// One body index per instance, parallel to `buffer`.
    bodies: std::sync::Arc<wgpu::Buffer>,
    capacity: u32,
}

/// One GPU colour table for a whole plan set. `colours` holds every body's
/// face colours contiguously; `offsets[body]` is the first entry for that body.
/// A colour-only change rewrites these buffers in place, leaving every chunk,
/// mesh and instance buffer untouched.
struct PlanColourTable {
    colours: wgpu::Buffer,
    offsets: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    colour_capacity: u32,
    offset_capacity: u32,
}

impl PlanColourTable {
    fn can_hold(&self, colours: &[[f32; 4]], offsets: &[u32]) -> bool {
        colours.len() as u32 <= self.colour_capacity
            && offsets.len() as u32 <= self.offset_capacity
    }

    fn write(&self, queue: &wgpu::Queue, colours: &[[f32; 4]], offsets: &[u32]) {
        queue.write_buffer(&self.colours, 0, bytemuck::cast_slice(colours));
        queue.write_buffer(&self.offsets, 0, bytemuck::cast_slice(offsets));
    }
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
    /// Global body indices in body order. Face ids use them as the `body:N`
    /// prefix, one instance at a time.
    bodies: Vec<u32>,
    plan: std::sync::Arc<TessPlan>,
    /// The one class per plan edge, shared with the plan and uploaded once.
    edge_classes: std::sync::Arc<[nodecraft_layers::PlanEdgeClass]>,
    /// One placement per body, in the same order as `bodies`. A group never
    /// loses a body: `matrices.len() == bodies.len()` is the invariant that
    /// keeps a shared plan from collapsing a mirror back onto its input.
    matrices: Vec<[[f32; 4]; 4]>,
}

/// Split a plan set into draw groups keyed only by the resident
/// `Arc<TessPlan>`. Colour is no longer a grouping input: the shader looks each
/// vertex's colour up from the shared table using the per-instance body index,
/// so bodies with different face colours still share one instanced draw. Groups
/// and bodies stay in body order.
fn plan_draw_groups(
    plans: &[PlanInstance],
    extra: Option<[[f32; 4]; 4]>,
) -> Vec<DrawGroup> {
    let mut groups: Vec<DrawGroup> = Vec::new();
    let mut by_plan: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for (body, instance) in plans.iter().enumerate() {
        let matrix = instance_matrix(&instance.placement);
        let matrix = extra.map_or(matrix, |extra| multiply_f32(extra, matrix));
        let plan_key = std::sync::Arc::as_ptr(&instance.plan) as usize;
        let existing = by_plan.get(&plan_key).copied().filter(|index| {
            std::sync::Arc::ptr_eq(&groups[*index].plan, &instance.plan)
        });
        match existing {
            Some(index) => {
                groups[index].bodies.push(body as u32);
                groups[index].matrices.push(matrix);
            }
            None => {
                by_plan.insert(plan_key, groups.len());
                groups.push(DrawGroup {
                    bodies: vec![body as u32],
                    plan: std::sync::Arc::clone(&instance.plan),
                    edge_classes: std::sync::Arc::clone(&instance.edge_classes),
                    matrices: vec![matrix],
                });
            }
        }
    }
    groups
}

fn prepare_triangle_lines_for_plan(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &GpuMesh,
    plan: &TessPlan,
) -> Option<TriangleLines> {
    // A resident mesh contains exactly one plan, so there is one part and one
    // line range; every instance reuses that range.
    let part = mesh.parts.first()?;
    let indices = triangle_line_indices(plan, &mesh.vertex_face, part.vertices.clone());
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
    Some(TriangleLines { indices: buffer, ranges: vec![(0, indices.len() as u32)] })
}

/// Compute fills created during one prepare pass.
///
/// A fill writes the mesh's vertex buffer, so it must be submitted even when
/// the frame that prepared it never reaches the GPU: `wgpu` surface acquisition
/// can skip a frame after the renderer prepared its callbacks. Riding on that
/// frame's encoder would leave a resident mesh with an empty vertex buffer
/// forever. A separate encoder and queue submission makes the fill part of the
/// mesh's residency, not of an unrelated paint frame.
struct PlanFills {
    encoder: Option<wgpu::CommandEncoder>,
}

impl PlanFills {
    fn new() -> Self {
        Self { encoder: None }
    }

    fn record(&mut self, device: &wgpu::Device, tess: &TessGpu, mesh: &GpuMesh) {
        let encoder = self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport-plan-fill"),
            })
        });
        tess.record(encoder, mesh);
    }

    fn submit(self, queue: &wgpu::Queue) {
        if let Some(encoder) = self.encoder {
            queue.submit(Some(encoder.finish()));
        }
    }
}

fn resident_mesh(
    device: &wgpu::Device,
    fills: &mut PlanFills,
    queue: &wgpu::Queue,
    tess: &TessGpu,
    cache: &mut std::collections::HashMap<usize, ResidentMesh>,
    plan: &std::sync::Arc<TessPlan>,
    edge_classes: &[nodecraft_layers::PlanEdgeClass],
    triangles: bool,
    edge_renderer: &mut EdgeRenderer,
    residency: &mut Residency,
) -> Option<std::sync::Arc<GpuMesh>> {
    let key = std::sync::Arc::as_ptr(plan) as usize;
    if let Some(resident) = cache.get_mut(&key) {
        residency.touch(key);
        if resident.edges.is_none() {
            let batch = EdgeCpuBatch::from_plan(plan.as_ref(), &[], edge_classes);
            resident.edges = GpuEdgeBatch::upload(edge_renderer, device, queue, &batch);
            if let Some(edges) = resident.edges.as_ref() {
                resident.bytes = resident.bytes.saturating_add(edges.bytes);
                residency.add_bytes(key, edges.bytes);
            }
        }
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
    fills.record(device, tess, &mesh);
    let mesh = std::sync::Arc::new(mesh);
    let lines = triangles
        .then(|| prepare_triangle_lines_for_plan(device, queue, &mesh, plan.as_ref()))
        .flatten()
        .map(std::sync::Arc::new);
    let lines_bytes = lines
        .as_ref()
        .map_or(0, |lines| lines.indices.size() + lines.ranges.len() as u64 * 8);
    let edge_batch = EdgeCpuBatch::from_plan(plan.as_ref(), &[], edge_classes);
    let edges = GpuEdgeBatch::upload(edge_renderer, device, queue, &edge_batch);
    let edge_bytes = edges.as_ref().map_or(0, |edges| edges.bytes);
    let bytes = mesh.vertex_count as u64 * solidcraft_viewport_gpu::VERTEX_STRIDE
        + mesh.index_count as u64 * 4
        + lines_bytes
        + edge_bytes;
    cache.insert(
        key,
        ResidentMesh {
            plan: std::sync::Arc::clone(plan),
            mesh: std::sync::Arc::clone(&mesh),
            edges,
            lines,
            instances: std::collections::HashMap::new(),
            faces: std::collections::HashMap::new(),
            bytes,
        },
    );
    residency.insert(key, bytes);
    Some(mesh)
}

/// Per-vertex face pick ids for one plan's resident mesh. Each id carries the
/// B-rep face and whether the display has multiple bodies; the body index is
/// added per instance from the group's body list. The same resident plan can
/// therefore be drawn by every body of the group without rebuilding this array.
fn prepare_face_ids(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut std::collections::HashMap<usize, ResidentMesh>,
    plan: &std::sync::Arc<TessPlan>,
    mesh: &GpuMesh,
    multiple: bool,
    residency: &mut Residency,
) -> std::sync::Arc<wgpu::Buffer> {
    let plan_key = std::sync::Arc::as_ptr(plan) as usize;
    let face_key = u64::from(multiple);
    if let Some(buffer) = cache
        .get(&plan_key)
        .and_then(|resident| resident.faces.get(&face_key))
        .cloned()
    {
        return buffer;
    }
    let ids: Vec<u32> = (0..mesh.vertex_count as usize)
        .map(|index| {
            let face = mesh.vertex_face.get(index).copied().unwrap_or(u32::MAX);
            crate::gpu_pick::face_pick_id(0, face, multiple)
        })
        .collect();
    let bytes = (ids.len().max(1) * std::mem::size_of::<u32>()) as u64;
    let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport-plan-face-ids"),
        size: bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&ids));
    if let Some(resident) = cache.get_mut(&plan_key) {
        resident.faces.insert(face_key, std::sync::Arc::clone(&buffer));
        resident.bytes = resident.bytes.saturating_add(bytes);
    }
    residency.add_bytes(plan_key, bytes);
    buffer
}

/// Build one pass's GPU chunk list from a plan set. This is the single builder
/// for both ways a set is installed: the staged activation (`pending.plans`)
/// and the immediate resident commit (`self.plans`). Both therefore produce the
/// same chunks: one per plan draw group, with every body of that plan in body
/// order and one instance placement per body. The colour table is shared by
/// every chunk in the set.
fn prepare_plan_chunks(
    tess: &TessGpu,
    fills: &mut PlanFills,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cache: &mut std::collections::HashMap<usize, ResidentMesh>,
    plans: &[PlanInstance],
    table: std::sync::Arc<PlanColourTable>,
    alpha: f32,
    extra: Option<[[f32; 4]; 4]>,
    triangles: bool,
    pass: InstanceSlot,
    edge_renderer: &mut EdgeRenderer,
    mut edge_pick_next: Option<&mut u32>,
    residency: &mut Residency,
) -> Vec<PlanChunk> {
    let multiple_faces = plans.len() > 1;
    let groups = plan_draw_groups(plans, extra);
    let mut chunks = Vec::with_capacity(groups.len());
    for group in groups {
        let Some(mesh) = resident_mesh(
            device,
            fills,
            queue,
            tess,
            cache,
            &group.plan,
            &group.edge_classes,
            triangles,
            edge_renderer,
            residency,
        ) else {
            continue;
        };
        let face_ids = prepare_face_ids(
            device,
            queue,
            cache,
            &group.plan,
            &mesh,
            multiple_faces,
            residency,
        );
        let plan_key = std::sync::Arc::as_ptr(&group.plan) as usize;
        let instances: Vec<GpuInstance> = group
            .matrices
            .iter()
            .map(|matrix| GpuInstance::from_model(*matrix, alpha))
            .collect();
        let needed = instances.len() as u32;
        let instance_hash = if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            use std::hash::{Hash, Hasher};
            let mut instance_hasher = std::collections::hash_map::DefaultHasher::new();
            bytemuck::cast_slice::<GpuInstance, u8>(&instances).hash(&mut instance_hasher);
            instance_hasher.finish()
        } else {
            0
        };
        let slot_key = pass;
        let instance_bytes = std::mem::size_of::<GpuInstance>() as u64;
        let one_body_bytes = std::mem::size_of::<u32>() as u64;
        let (instance_buffer, body_buffer) = if let Some(slot) = cache
            .get_mut(&plan_key)
            .and_then(|resident| resident.instances.get_mut(&slot_key))
        {
            if slot.capacity < needed {
                let bytes = u64::from(needed) * instance_bytes;
                let body_bytes = u64::from(needed) * one_body_bytes;
                let old = u64::from(slot.capacity) * (instance_bytes + one_body_bytes);
                let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-plan-instances"),
                    size: bytes,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
                let bodies = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("viewport-plan-instance-bodies"),
                    size: body_bytes,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }));
                slot.buffer = std::sync::Arc::clone(&buffer);
                slot.bodies = std::sync::Arc::clone(&bodies);
                slot.capacity = needed;
                let bytes = bytes + body_bytes;
                if bytes > old {
                    if let Some(resident) = cache.get_mut(&plan_key) {
                        resident.bytes = resident.bytes.saturating_add(bytes - old);
                    }
                    residency.add_bytes(plan_key, bytes - old);
                }
                (buffer, bodies)
            } else {
                (
                    std::sync::Arc::clone(&slot.buffer),
                    std::sync::Arc::clone(&slot.bodies),
                )
            }
        } else {
            let bytes = u64::from(needed) * instance_bytes;
            let body_bytes = u64::from(needed) * one_body_bytes;
            let buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-plan-instances"),
                size: bytes.max(instance_bytes),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            let bodies = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-plan-instance-bodies"),
                size: body_bytes.max(one_body_bytes),
                usage: wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            if let Some(resident) = cache.get_mut(&plan_key) {
                resident.instances.insert(
                    slot_key,
                    InstanceBuffer {
                        buffer: std::sync::Arc::clone(&buffer),
                        bodies: std::sync::Arc::clone(&bodies),
                        capacity: needed.max(1),
                    },
                );
                resident.bytes = resident
                    .bytes
                    .saturating_add(buffer.size() + bodies.size());
            }
            residency.add_bytes(plan_key, buffer.size() + bodies.size());
            (buffer, bodies)
        };
        queue.write_buffer(&instance_buffer, 0, bytemuck::cast_slice(&instances));
        queue.write_buffer(&body_buffer, 0, bytemuck::cast_slice(&group.bodies));
        let key = std::sync::Arc::as_ptr(&group.plan) as usize;
        let lines = cache.get(&key).and_then(|resident| resident.lines.clone());
        let edges = cache.get(&key).and_then(|resident| resident.edges.clone());
        let edge_count = edges.as_ref().map_or(0, |edges| edges.ranges.len() as u32);
        let bases: Vec<u32> = {
            let mut next = 0;
            if let Some(value) = edge_pick_next.as_deref() {
                next = *value;
            }
            (0..needed)
                .map(|slot| next + slot * edge_count)
                .collect()
        };
        if let Some(value) = edge_pick_next.as_deref_mut() {
            *value = value.saturating_add(needed.saturating_mul(edge_count));
        }
        let edge_bases = (edge_count > 0 && pass == InstanceSlot::Main).then(|| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewport-plan-edge-pick-bases"),
                size: (bases.len().max(1) * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&bases));
            std::sync::Arc::new(buffer)
        });
        let edge_pick_base = bases.first().copied().unwrap_or(0);
        chunks.push(PlanChunk {
            mesh,
            face_ids,
            instances: instance_buffer,
            bodies: body_buffer,
            body_list: group.bodies.clone(),
            edges,
            edge_bases,
            edge_count,
            edge_pick_base,
            table: std::sync::Arc::clone(&table),
            instance_count: needed,
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
    /// Per-vertex pick id carrying the B-rep face and the multi-body flag. The
    /// same buffer feeds the pick pass and the shader's colour table lookup.
    face_ids: std::sync::Arc<wgpu::Buffer>,
    instances: std::sync::Arc<wgpu::Buffer>,
    /// Global body index per instance, parallel to `instances`.
    bodies: std::sync::Arc<wgpu::Buffer>,
    /// Global body indices in instance order, for edge highlights and picking.
    body_list: Vec<u32>,
    /// Local-space ribbons shared by every body of this plan.
    edges: Option<std::sync::Arc<GpuEdgeBatch>>,
    /// Edge pick id base per instance, parallel to `instances`.
    edge_bases: Option<std::sync::Arc<wgpu::Buffer>>,
    /// Number of edges in the plan and this chunk's first pick index.
    edge_count: u32,
    edge_pick_base: u32,
    /// Shared colour table for the whole plan set. It changes only on a colour
    /// edit, never on a transform or mesh edit.
    table: std::sync::Arc<PlanColourTable>,
    instance_count: u32,
    instance_hash: u64,
    lines: Option<std::sync::Arc<TriangleLines>>,
}

const RESIDENT_BUDGET: u64 = 512 << 20;
const UPLOAD_VERTEX_BUDGET: usize = 2_000_000;
const UPLOAD_TIME_BUDGET: std::time::Duration = std::time::Duration::from_millis(4);

fn plans_ready(plans: &[PlanInstance], prepared: &std::collections::HashSet<usize>) -> bool {
    plans.iter().all(|instance| {
        prepared.contains(&(std::sync::Arc::as_ptr(&instance.plan) as usize))
    })
}

/// Geometry and placement identity. A change here needs new chunks; a change
/// only in `face_colors` does not.
fn same_plan_geometry(a: &[PlanInstance], b: &[PlanInstance]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(old, new)| {
            old.target == new.target
                && old.body == new.body
                && std::sync::Arc::ptr_eq(&old.plan, &new.plan)
                && old.edge_classes == new.edge_classes
                && old.placement == new.placement
        })
}

fn same_plan_colours(a: &[PlanInstance], b: &[PlanInstance]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(old, new)| old.face_colors == new.face_colors)
}

#[cfg(test)]
fn same_plan_set(a: &[PlanInstance], b: &[PlanInstance]) -> bool {
    same_plan_geometry(a, b) && same_plan_colours(a, b)
}

/// Flatten one plan set's face colours into the shared table. Each body owns a
/// contiguous slice starting at `offsets[body]`; a body's slice covers its
/// plan's face count even when the display only authored some faces. Face
/// colouring and the ignore-look override are table fills too, so every draw
/// path uses the same lookup.
pub(crate) fn colour_table_data(
    plans: &[PlanInstance],
    colour: Option<[f32; 3]>,
    face_colours: bool,
) -> (Vec<[f32; 4]>, Vec<u32>) {
    let mut colours = Vec::new();
    let mut offsets = Vec::with_capacity(plans.len());
    for instance in plans {
        offsets.push(colours.len() as u32);
        let count = instance
            .face_colors
            .len()
            .max(instance.plan.faces.len())
            .max(1);
        for face in 0..count {
            let value = if face_colours {
                face_colour(instance.body, face as u32)
            } else if let Some(value) = colour {
                value
            } else {
                instance
                    .face_colors
                    .get(face)
                    .copied()
                    .unwrap_or(DEFAULT_FACE_COLOUR)
            };
            colours.push([value[0], value[1], value[2], 0.0]);
        }
    }
    (colours, offsets)
}

/// Write the table in place when it still fits, otherwise replace it. A
/// colour-only edit therefore rewrites two storage buffers and leaves every
/// chunk, mesh and instance buffer alone.
fn refresh_colour_table(
    renderer: &PlanRenderer,
    cache: &mut Option<std::sync::Arc<PlanColourTable>>,
    uploads: &mut u64,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plans: &[PlanInstance],
    colour: Option<[f32; 3]>,
    face_colours: bool,
) -> std::sync::Arc<PlanColourTable> {
    let (colours, offsets) = colour_table_data(plans, colour, face_colours);
    if let Some(table) = cache.as_ref()
        && table.can_hold(&colours, &offsets)
    {
        table.write(queue, &colours, &offsets);
        *uploads = uploads.saturating_add(1);
        return std::sync::Arc::clone(table);
    }
    let table = std::sync::Arc::new(renderer.create_colour_table(
        device,
        queue,
        &colours,
        &offsets,
    ));
    *cache = Some(std::sync::Arc::clone(&table));
    *uploads = uploads.saturating_add(1);
    table
}

/// Whether every plan in `plans` already has a resident GPU mesh. A set whose
/// plans are all resident needs no staged swap: preparing it uploads nothing,
/// so `prepare` can activate it in the current frame. An empty set is also
/// ready (it has nothing to upload).
fn plans_resident(
    plans: &[PlanInstance],
    resident: impl Fn(usize) -> bool,
) -> bool {
    plans.iter().all(|instance| {
        resident(std::sync::Arc::as_ptr(&instance.plan) as usize)
    })
}

struct PendingPlans {
    plans: Vec<PlanInstance>,
    /// Exact object state that produced this plan set (and its edge payload).
    snapshot: u64,
    enabled: bool,
    colour: Option<[f32; 3]>,
    triangles: bool,
    prepared: std::collections::HashSet<usize>,
    failed: std::collections::HashSet<usize>,
}

/// Whether `PlanState::set_plans` installed the set now or staged it for a
/// later activation. The viewport stages the matching display payload with the
/// same wait: both activate in `prepare`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanSetState {
    Immediate,
    Staged,
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
    /// Bumped whenever the plans that `paint` draws change. The viewport's
    /// staged display waits for the matching activation, so edges and plan
    /// triangles never switch on different frames.
    activation: u64,
    /// Exact object state whose plans are currently drawn. The staged edge
    /// payload must carry the same id to activate with this plan set.
    display_snapshot: u64,
    resident: std::collections::HashMap<usize, ResidentMesh>,
    residency: Residency,
    last_prepare_us: u64,
    last_uploads: u32,
    painted_chunks: std::sync::atomic::AtomicU32,
    painted_instances: std::sync::atomic::AtomicU32,
    colour_override: Option<[f32; 3]>,
    /// One table per pass. Chunks hold the same Arc, so a colour-only edit
    /// rewrites it in place and leaves the chunks untouched.
    colour_table: Option<std::sync::Arc<PlanColourTable>>,
    ghost_colour_table: Option<std::sync::Arc<PlanColourTable>>,
    colour_table_dirty: bool,
    /// Distinct face colours for the whole body (the View menu's Face colours
    /// mode). It is a table fill, not a renderer switch.
    face_colours: bool,
    /// Diagnostics: chunk list rebuilds and colour-table uploads are counted
    /// separately so a colour edit can prove it did not rebuild geometry.
    chunk_builds: u64,
    table_uploads: u64,
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
    /// Standard edge width in screen pixels. It reaches the ribbon shader through
    /// a uniform only: changing it never touches a plan, chunk or edge buffer.
    edge_width_px: f32,
    /// `ViewSettings.edges && !zebra`, the same gate the old scene path used.
    edges_enabled: bool,
    /// `ViewSettings.show_seam_edges`, reaching the ribbon shader as a uniform
    /// mask only. Changing it never rebuilds a plan, chunk or edge buffer.
    show_seam_edges: bool,
    /// The one ribbon renderer for plan edges, world fallback edges and
    /// selection/hover highlights.
    edges: EdgeRenderer,
    world_edges: Option<std::sync::Arc<GpuEdgeBatch>>,
    world_edge_ids: Vec<String>,
    world_ghost_edges: Option<std::sync::Arc<GpuEdgeBatch>>,
    world_ghost_edge_ids: Vec<String>,
    /// New world-space edge data, uploaded by the next `prepare_chunks` when
    /// the device and queue are available. Plan edges do not use this slot.
    pending_world_edges: Option<EdgeCpuBatch>,
    pending_world_ghost_edges: Option<EdgeCpuBatch>,
    selected_edges: Vec<String>,
    hovered_edge: Option<String>,
    /// Pick-id table for the plan's local edges. Pick index 0 is "none"; the
    /// rest follow chunk order, body slot and plan edge order.
    edge_pick: GhostPickData,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PlanDebug {
    pub enabled: bool,
    pub pending: bool,
    pub activation: u64,
    pub snapshot: u64,
    pub pending_snapshot: u64,
    pub chunks: usize,
    pub plans: usize,
    pub triangles: u32,
    pub instances: u32,
    pub failed: usize,
    pub plan_hash: u64,
    pub instance_hash: u64,
    pub prepare_us: u64,
    pub uploads: u32,
    pub face_colours: bool,
    pub chunk_builds: u64,
    pub table_uploads: u64,
    pub painted_chunks: u32,
    pub painted_instances: u32,
    pub edge_width_px: f32,
    pub edge_uploads: u64,
    pub edge_draws: u64,
    pub edge_pick_entries: usize,
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
            activation: 0,
            display_snapshot: 0,
            resident: std::collections::HashMap::new(),
            residency: Residency::new(RESIDENT_BUDGET),
            last_prepare_us: 0,
            last_uploads: 0,
            painted_chunks: std::sync::atomic::AtomicU32::new(0),
            painted_instances: std::sync::atomic::AtomicU32::new(0),
            colour_override: None,
            colour_table: None,
            ghost_colour_table: None,
            colour_table_dirty: true,
            face_colours: false,
            chunk_builds: 0,
            table_uploads: 0,
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
            edge_width_px: crate::viewport::default_edge_width_px(),
            edges_enabled: true,
            show_seam_edges: false,
            edges: EdgeRenderer::install(device, format, samples, depth, reversed),
            world_edges: None,
            world_edge_ids: Vec::new(),
            world_ghost_edges: None,
            world_ghost_edge_ids: Vec::new(),
            pending_world_edges: None,
            pending_world_ghost_edges: None,
            selected_edges: Vec::new(),
            hovered_edge: None,
            edge_pick: GhostPickData::default(),
        }
    }

    fn touch_resident(&mut self, key: usize) {
        self.residency.touch(key);
    }

    pub(crate) fn set_plans(
        &mut self,
        plans: Vec<PlanInstance>,
        enabled: bool,
        colour: Option<[f32; 3]>,
        snapshot: u64,
    ) -> PlanSetState {
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[plan] set {} active {} chunks {} pending={} snapshot={snapshot}",
                plans.len(),
                self.plans.len(),
                self.chunks.len(),
                self.pending.is_some(),
            );
        }
        // A staged set that still has the same geometry only needs its colour
        // request updated. Do not restart its mesh preparation, and do not
        // rebuild anything: the table is filled when the set activates.
        if let Some(pending) = self.pending.as_mut()
            && pending.enabled == enabled
            && same_plan_geometry(&pending.plans, &plans)
        {
            pending.plans = plans;
            pending.colour = colour;
            pending.snapshot = snapshot;
            return PlanSetState::Staged;
        }
        // The display plan cache reuses the same plan Arcs across evaluations.
        // Keep chunks, meshes and instance buffers when only the colouring
        // changed; a colour-only edit rewrites the shared table in `prepare`.
        if same_plan_geometry(&self.plans, &plans) && self.enabled == enabled {
            let colour_changed =
                !same_plan_colours(&self.plans, &plans) || self.colour_override != colour;
            // An edit-back must cancel a still-pending future set; otherwise the
            // stale pending swap would activate after this frame.
            self.pending = None;
            self.plans = plans;
            self.colour_override = colour;
            self.display_snapshot = snapshot;
            if colour_changed {
                self.colour_table_dirty = true;
            }
            if enabled && self.chunks.is_empty() {
                self.dirty = true;
            }
            return PlanSetState::Immediate;
        }
        // A plan set whose meshes are all resident needs no staged swap: its
        // preparation is instance/table churn over cached GPU meshes, with zero
        // vertex uploads. Activate it in this frame instead of deferring the
        // exact frame and replaying it once the "swap" is ready. The chunks are
        // rebuilt by `prepare` from `self.plans` (dirty), so edges, picking and
        // plan triangles still switch in this activation.
        if !enabled || plans_resident(&plans, |key| self.resident.contains_key(&key)) {
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!(
                    "[plan] resident set {} no staging",
                    plans.len(),
                );
            }
            let keys: Vec<usize> = plans
                .iter()
                .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
                .collect();
            self.residency.touch_set(&keys);
            self.plans = plans;
            self.enabled = enabled;
            self.colour_override = colour;
            self.pending = None;
            self.display_snapshot = snapshot;
            self.activation = self.activation.wrapping_add(1);
            // A resident/empty enabled set still needs `prepare` to rebuild its
            // chunks in this frame; a disabled set only clears them there.
            self.dirty = enabled;
            self.colour_table_dirty = enabled;
            return PlanSetState::Immediate;
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
            snapshot,
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
        PlanSetState::Staged
    }

    pub(crate) fn activation(&self) -> u64 {
        self.activation
    }

    pub(crate) fn display_snapshot(&self) -> u64 {
        self.display_snapshot
    }

    pub(crate) fn set_translucent(&mut self, translucent: bool) {
        self.translucent = translucent;
    }

    /// A width change is a uniform value, not geometry. The plan chunks, their
    /// edge buffers and every instance buffer stay exactly as they are.
    pub(crate) fn set_edge_width(&mut self, width_px: f32) {
        self.edge_width_px = width_px.clamp(0.1, 16.0);
    }

    pub(crate) fn set_edges_enabled(&mut self, enabled: bool) {
        self.edges_enabled = enabled;
    }

    /// Seams are a uniform mask over the class already in each edge vertex. No
    /// plan, chunk, edge buffer or pick table changes.
    pub(crate) fn set_show_seam_edges(&mut self, show: bool) {
        self.show_seam_edges = show;
    }

    /// Install world-space fallback edges (the CPU renderer or a comparison
    /// overlay). Plan edges never come through here: they are already resident
    /// with their plan.
    pub(crate) fn set_world_edges(&mut self, edges: EdgeCpuBatch) {
        self.pending_world_edges = Some(edges);
    }

    pub(crate) fn set_world_ghost_edges(&mut self, edges: EdgeCpuBatch) {
        self.pending_world_ghost_edges = Some(edges);
    }

    /// The drawn selection and hover. This is CPU-side id lookup; the GPU only
    /// gets a handful of extra range draws, never a rebuilt buffer.
    pub(crate) fn set_edge_selection(&mut self, selected: Vec<String>, hovered: Option<String>) {
        if self.selected_edges == selected && self.hovered_edge == hovered {
            return;
        }
        self.selected_edges = selected;
        self.hovered_edge = hovered;
    }

    pub(crate) fn edge_pick_data(&self) -> &GhostPickData {
        &self.edge_pick
    }

    /// Rebuild the plan-edge pick id table. The index of an edge here is the
    /// same number the plan edge pick shader writes, so a readback index resolves
    /// without any CPU world-space point list.
    fn rebuild_edge_pick_data(&mut self) {
        let multiple = self.plans.len() > 1;
        let mut edges = vec![(String::new(), [0.0; 3])];
        for chunk in &self.chunks {
            if chunk.edge_count == 0 {
                continue;
            }
            for (slot, body) in chunk.body_list.iter().enumerate() {
                let Some(instance) = self.plans.get(*body as usize) else { continue };
                debug_assert_eq!(
                    chunk.edge_pick_base as usize + slot * chunk.edge_count as usize,
                    edges.len(),
                    "edge pick bases must follow the resolution table order",
                );
                for edge_index in 0..chunk.edge_count as usize {
                    let id = instance
                        .edge_ids
                        .get(edge_index)
                        .and_then(Clone::clone)
                        .unwrap_or_else(|| format!("edge:{edge_index}"));
                    let id = if multiple {
                        format!("body:{body}:{id}")
                    } else {
                        id
                    };
                    edges.push((id, plan_edge_midpoint(instance, edge_index)));
                }
            }
        }
        self.edge_pick = GhostPickData {
            multi_body: multiple,
            segments: std::sync::Arc::from(Vec::new()),
            edges,
        };
    }

    /// Draw plan-local edge ids into the pick target with the shared ribbon
    /// vertex function. Every body instance carries its own pick base.
    pub(crate) fn paint_edge_ids(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        for chunk in &self.chunks {
            let (Some(edges), Some(bases)) = (chunk.edges.as_ref(), chunk.edge_bases.as_ref()) else {
                continue;
            };
            pass.set_vertex_buffer(0, edges.buffer.slice(..));
            pass.set_vertex_buffer(1, chunk.instances.slice(..));
            pass.set_vertex_buffer(2, bases.slice(..));
            pass.draw(0..edges.vertex_count, 0..chunk.instance_count);
        }
    }

    /// The View menu's face-colours mode is a table fill on the plan path. It
    /// does not rebuild chunks, meshes or instance buffers.
    pub(crate) fn set_face_colours(&mut self, enabled: bool) {
        if self.face_colours == enabled {
            return;
        }
        self.face_colours = enabled;
        self.colour_table_dirty = true;
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
                    && old.placement == new.placement
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
        self.display_snapshot = 0;
        self.activation = self.activation.wrapping_add(1);
        self.enabled = false;
        self.colour_override = None;
        self.colour_table = None;
        self.colour_table_dirty = false;
        self.clear_transient();
        self.edge_pick = GhostPickData::default();
        self.dirty = false;
        // Resident meshes deliberately survive; they hold their plan Arcs and
        // are bounded by the LRU budget (Step 2).
    }

    pub(crate) fn clear_transient(&mut self) {
        self.dirty = true;
        self.ghost_matrix = None;
        self.ghost_plans.clear();
        self.ghost_chunks.clear();
        self.ghost_colour_table = None;
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
            activation: self.activation,
            snapshot: self.display_snapshot,
            pending_snapshot: self.pending.as_ref().map_or(0, |pending| pending.snapshot),
            chunks: self.chunks.len(),
            plans: self.plans.len(),
            triangles,
            instances,
            failed,
            plan_hash: plan_identity_hash(&self.plans),
            instance_hash,
            prepare_us: self.last_prepare_us,
            uploads: self.last_uploads,
            face_colours: self.face_colours,
            chunk_builds: self.chunk_builds,
            table_uploads: self.table_uploads,
            painted_chunks: self.painted_chunks.load(std::sync::atomic::Ordering::Relaxed),
            painted_instances: self.painted_instances.load(std::sync::atomic::Ordering::Relaxed),
            edge_width_px: self.edge_width_px,
            edge_uploads: self.edges.uploads,
            edge_draws: self.edges.draws,
            edge_pick_entries: self.edge_pick.edges.len(),
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

    /// Upload the plan uniforms for one drawn frame. `preview` is the display
    /// unit's single preview value for that frame; there is no second preview
    /// owner in this state.
    pub(crate) fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        uniforms: &ObjectUniforms,
        preview: Option<[[f32; 4]; 4]>,
        size_px: [f32; 2],
    ) {
        if let Some(renderer) = self.renderer.as_ref() {
            renderer.prepare(queue, uniforms, preview, self.ghost_matrix);
        }
        let theme = crate::theme::get();
        let main_transform = preview.unwrap_or_else(identity_matrix);
        let ghost_transform = self.ghost_matrix.unwrap_or_else(identity_matrix);
        let (standard, ghost, selected, hover) = edge_widths(self.edge_width_px);
        self.edges.write_uniforms(
            queue,
            EdgeUniforms::new(
                uniforms.view_proj(),
                main_transform,
                size_px,
                standard,
                [0.0, 0.0, 0.0, 1.0],
                true,
            )
            .show_seams(self.show_seam_edges),
            EdgeUniforms::new(
                uniforms.view_proj(),
                ghost_transform,
                size_px,
                ghost,
                theme_colour(theme.ghost_edge),
                false,
            )
            .show_seams(self.show_seam_edges),
            EdgeUniforms::new(
                uniforms.view_proj(),
                main_transform,
                size_px,
                selected,
                theme_colour(theme.accent),
                true,
            )
            .show_seams(self.show_seam_edges),
            EdgeUniforms::new(
                uniforms.view_proj(),
                main_transform,
                size_px,
                hover,
                theme_colour(theme.highlight),
                true,
            )
            .show_seams(self.show_seam_edges),
        );
    }

    /// Advance the staged plan set. The uniform upload is a separate step
    /// ([`PlanState::write_uniforms`]) so the display unit can decide the one
    /// preview value for the frame *after* an activation in this same
    /// `prepare`, then hand that same value to the plan shader and the scene
    /// projection.
    pub(crate) fn prepare_chunks(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let started = std::time::Instant::now();
        self.last_uploads = 0;
        if let Some(edges) = self.pending_world_edges.take() {
            self.world_edge_ids.clone_from(&edges.ids);
            self.world_edges = GpuEdgeBatch::upload(&mut self.edges, device, queue, &edges);
        }
        if let Some(edges) = self.pending_world_ghost_edges.take() {
            self.world_ghost_edge_ids.clone_from(&edges.ids);
            self.world_ghost_edges = GpuEdgeBatch::upload(&mut self.edges, device, queue, &edges);
        }
        if self.tess.is_none() {
            self.tess = self.tess_ready.as_ref().and_then(|ready| ready.get().cloned());
        }
        let Some(tess) = self.tess.clone() else {
            self.chunks.clear();
            self.ghost_chunks.clear();
            return;
        };
        let mut fills = PlanFills::new();
        let mut edge_pick_next = 1u32;
        if let Some(mut pending) = self.pending.take() {
            if !pending.enabled || pending.plans.is_empty() {
                self.plans = pending.plans;
                self.enabled = pending.enabled;
                self.colour_override = pending.colour;
                self.chunks.clear();
                self.dirty = false;
                self.colour_table_dirty = false;
                self.display_snapshot = pending.snapshot;
                self.activation = self.activation.wrapping_add(1);
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
                        &mut fills,
                        queue,
                        &tess,
                        &mut self.resident,
                        &instance.plan,
                        &instance.edge_classes,
                        pending.triangles,
                        &mut self.edges,
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
                    let table = {
                        let Some(renderer) = self.renderer.as_ref() else { return };
                        refresh_colour_table(
                            renderer,
                            &mut self.colour_table,
                            &mut self.table_uploads,
                            device,
                            queue,
                            &pending.plans,
                            pending.colour,
                            self.face_colours,
                        )
                    };
                    self.chunks = prepare_plan_chunks(
                        &tess,
                        &mut fills,
                        device,
                        queue,
                        &mut self.resident,
                        &pending.plans,
                        table,
                        1.0,
                        None,
                        pending.triangles,
                        InstanceSlot::Main,
                        &mut self.edges,
                        Some(&mut edge_pick_next),
                        &mut self.residency,
                    );
                    self.chunk_builds = self.chunk_builds.saturating_add(1);
                    self.plans = pending.plans;
                    self.enabled = pending.enabled;
                    self.colour_override = pending.colour;
                    self.dirty = false;
                    self.colour_table_dirty = false;
                    self.display_snapshot = pending.snapshot;
                    self.activation = self.activation.wrapping_add(1);
                    let keys: Vec<usize> = self
                        .plans
                        .iter()
                        .map(|instance| std::sync::Arc::as_ptr(&instance.plan) as usize)
                        .collect();
                    self.residency.touch_set(&keys);
                    self.rebuild_edge_pick_data();
                } else {
                    self.pending = Some(pending);
                }
            }
        } else if self.enabled && self.dirty {
            self.dirty = false;
            let table = {
                let Some(renderer) = self.renderer.as_ref() else { return };
                refresh_colour_table(
                    renderer,
                    &mut self.colour_table,
                    &mut self.table_uploads,
                    device,
                    queue,
                    &self.plans,
                    self.colour_override,
                    self.face_colours,
                )
            };
            self.chunks = prepare_plan_chunks(
                &tess,
                &mut fills,
                device,
                queue,
                &mut self.resident,
                &self.plans,
                table,
                1.0,
                None,
                self.triangles,
                InstanceSlot::Main,
                &mut self.edges,
                Some(&mut edge_pick_next),
                &mut self.residency,
            );
            self.chunk_builds = self.chunk_builds.saturating_add(1);
            self.colour_table_dirty = false;
            self.rebuild_edge_pick_data();
        } else if self.enabled && self.colour_table_dirty {
            self.colour_table_dirty = false;
            let table = {
                let Some(renderer) = self.renderer.as_ref() else { return };
                refresh_colour_table(
                    renderer,
                    &mut self.colour_table,
                    &mut self.table_uploads,
                    device,
                    queue,
                    &self.plans,
                    self.colour_override,
                    self.face_colours,
                )
            };
            // Normally the table was rewritten in place, so every chunk already
            // points at the updated colours. Only a capacity change replaces
            // the table object, and then the chunks adopt the new handle.
            for chunk in &mut self.chunks {
                if !std::sync::Arc::ptr_eq(&chunk.table, &table) {
                    chunk.table = std::sync::Arc::clone(&table);
                }
            }
        } else if !self.enabled {
            self.chunks.clear();
        }
        if self.ghost_dirty {
            self.ghost_dirty = false;
            self.ghost_chunks = if self.ghost_enabled {
                let table = {
                    let Some(renderer) = self.renderer.as_ref() else { return };
                    refresh_colour_table(
                        renderer,
                        &mut self.ghost_colour_table,
                        &mut self.table_uploads,
                        device,
                        queue,
                        &self.ghost_plans,
                        self.ghost_colour,
                        self.face_colours,
                    )
                };
                let chunks = prepare_plan_chunks(
                    &tess,
                    &mut fills,
                    device,
                    queue,
                    &mut self.resident,
                    &self.ghost_plans,
                    table,
                    self.ghost_alpha,
                    None,
                    false,
                    InstanceSlot::Ghost,
                    &mut self.edges,
                    None,
                    &mut self.residency,
                );
                self.chunk_builds = self.chunk_builds.saturating_add(1);
                chunks
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
        // Submit every new mesh's fill now, before the frame can be dropped
        // and before any chunk draws it. `ResidentMesh` entries above are only
        // observable as drawable once this returns.
        fills.submit(queue);
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
        if self.last_prepare_us > 0 && std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[plan] prepare_us={} chunks={} instances={} uploads={} table_uploads={}",
                self.last_prepare_us,
                self.chunks.len(),
                self.chunks.iter().map(|chunk| chunk.instance_count).sum::<u32>(),
                self.last_uploads,
                self.table_uploads,
            );
        }
    }

    /// Draw the main plan chunks' B-rep face ids into the pick target with the
    /// same vertices, instance buffers and plan preview matrix as the visible
    /// plan pass.
    /// Draw the main plan chunks' B-rep face ids into the pick target with the
    /// same vertices, instance buffers, per-instance body indices and plan
    /// preview matrix as the visible plan pass.
    pub(crate) fn paint_face_ids(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        for chunk in &self.chunks {
            pass.set_vertex_buffer(0, chunk.mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, chunk.face_ids.slice(..));
            pass.set_vertex_buffer(2, chunk.instances.slice(..));
            pass.set_vertex_buffer(3, chunk.bodies.slice(..));
            pass.set_index_buffer(chunk.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            if let Some(part) = chunk.mesh.parts.first() {
                pass.draw_indexed(part.indices.clone(), 0, 0..chunk.instance_count);
            }
        }
    }

    pub(crate) fn paint_ghost_depth(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        bind: &wgpu::BindGroup,
    ) {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        for chunk in &self.ghost_chunks {
            pass.set_vertex_buffer(0, chunk.mesh.vertices.slice(..));
            pass.set_vertex_buffer(1, chunk.instances.slice(..));
            pass.set_index_buffer(chunk.mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            if let Some(part) = chunk.mesh.parts.first() {
                pass.draw_indexed(part.indices.clone(), 0, 0..chunk.instance_count);
            }
        }
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
            renderer.paint(
                pass,
                &chunk.mesh,
                &chunk.face_ids,
                &chunk.instances,
                &chunk.bodies,
                &chunk.table,
                self.translucent,
                chunk.instance_count,
            );
        }
        if self.triangles {
            for chunk in &self.chunks {
                if let Some(lines) = &chunk.lines {
                    renderer.paint_triangle_edges(
                        pass,
                        &chunk.mesh,
                        &chunk.instances,
                        &chunk.table,
                        lines,
                        chunk.instance_count,
                    );
                }
            }
        }
        for chunk in &self.ghost_chunks {
            renderer.paint_ghost(
                pass,
                &chunk.mesh,
                &chunk.face_ids,
                &chunk.instances,
                &chunk.bodies,
                &chunk.table,
                chunk.instance_count,
            );
        }
    }

    /// Draw every B-rep edge through the one ribbon pipeline. Plan edges are
    /// local-space buffers instanced with the plan's own instance buffer; world
    /// fallback edges use the identity instance. Selection and hover ranges are
    /// extra draws over the same buffers, never a second expansion.
    pub(crate) fn paint_edges(
        &mut self,
        pass: &mut wgpu::RenderPass<'static>,
        plan_active: bool,
    ) {
        if self.edges_enabled {
            if plan_active {
                for chunk in &self.chunks {
                    if let Some(edges) = &chunk.edges {
                        self.edges.paint(
                            pass,
                            edges,
                            &chunk.instances,
                            chunk.instance_count,
                            EdgeStyle::Standard,
                        );
                    }
                }
                for chunk in &self.ghost_chunks {
                    if let Some(edges) = &chunk.edges {
                        self.edges.paint(
                            pass,
                            edges,
                            &chunk.instances,
                            chunk.instance_count,
                            EdgeStyle::Ghost,
                        );
                    }
                }
            } else {
                if let Some(edges) = &self.world_edges {
                    self.edges.paint_world(pass, edges, EdgeStyle::Standard);
                }
                if let Some(edges) = &self.world_ghost_edges {
                    self.edges.paint_world(pass, edges, EdgeStyle::Ghost);
                }
            }
        }
        self.paint_edge_highlights(pass, plan_active);
    }

    fn paint_edge_highlights(
        &mut self,
        pass: &mut wgpu::RenderPass<'static>,
        plan_active: bool,
    ) {
        let selected = self.selected_edges.clone();
        let hovered = self
            .hovered_edge
            .clone()
            .filter(|id| !selected.iter().any(|selected| selected == id));
        if plan_active {
            for id in &selected {
                self.paint_plan_edge_id(pass, id, EdgeStyle::Selected);
            }
            if let Some(id) = hovered.as_deref() {
                self.paint_plan_edge_id(pass, id, EdgeStyle::Hover);
            }
        } else {
            for id in &selected {
                self.paint_world_edge_id(pass, id, EdgeStyle::Selected);
            }
            if let Some(id) = hovered.as_deref() {
                self.paint_world_edge_id(pass, id, EdgeStyle::Hover);
            }
        }
    }

    fn paint_plan_edge_id(&mut self, pass: &mut wgpu::RenderPass<'static>, id: &str, style: EdgeStyle) {
        let Some((batch, instances, instance, range)) = self.plan_edge_id(id) else {
            return;
        };
        self.edges.paint_range(pass, &batch, &instances, instance, range, style);
    }

    fn plan_edge_id(
        &self,
        id: &str,
    ) -> Option<(std::sync::Arc<GpuEdgeBatch>, std::sync::Arc<wgpu::Buffer>, u32, (u32, u32))> {
        let multiple = self.plans.len() > 1;
        let (body, edge_id) = split_body_id(id, multiple);
        let instance = self.plans.get(body)?;
        let edge_index = plan_edge_index(instance, edge_id, 0)?;
        for chunk in &self.chunks {
            let Some(slot) = chunk.body_list.iter().position(|candidate| *candidate as usize == body) else {
                continue;
            };
            let batch = chunk.edges.as_ref()?;
            let range = batch.range(edge_index)?;
            return Some((
                std::sync::Arc::clone(batch),
                std::sync::Arc::clone(&chunk.instances),
                slot as u32,
                range,
            ));
        }
        None
    }

    fn paint_world_edge_id(
        &mut self,
        pass: &mut wgpu::RenderPass<'static>,
        id: &str,
        style: EdgeStyle,
    ) {
        let Some(batch) = self.world_edges.as_ref() else { return };
        let Some(index) = self.world_edge_ids.iter().position(|candidate| candidate == id) else {
            return;
        };
        let Some(range) = batch.range(index) else { return };
        self.edges.paint_world_range(pass, batch, range, style);
    }
}

fn plan_edge_midpoint(instance: &PlanInstance, edge_index: usize) -> [f64; 3] {
    let Some(points) = instance.plan.edges.get(edge_index) else {
        return [0.0; 3];
    };
    let Some(point) = points.get(points.len() / 2).copied().or_else(|| points.first().copied()) else {
        return [0.0; 3];
    };
    let matrix = instance_matrix(&instance.placement);
    [
        f64::from(matrix[0][0]) * point.x
            + f64::from(matrix[1][0]) * point.y
            + f64::from(matrix[2][0]) * point.z
            + f64::from(matrix[3][0]),
        f64::from(matrix[0][1]) * point.x
            + f64::from(matrix[1][1]) * point.y
            + f64::from(matrix[2][1]) * point.z
            + f64::from(matrix[3][1]),
        f64::from(matrix[0][2]) * point.x
            + f64::from(matrix[1][2]) * point.y
            + f64::from(matrix[2][2]) * point.z
            + f64::from(matrix[3][2]),
    ]
}

/// `body:N:edge:M` for a multi-body display; otherwise the bare edge id.
fn split_body_id(id: &str, multiple: bool) -> (usize, &str) {
    if multiple && let Some(rest) = id.strip_prefix("body:")
        && let Some((body, edge)) = rest.split_once(':')
        && let Ok(body) = body.parse()
    {
        return (body, edge);
    }
    (0, id)
}

fn plan_edge_index(instance: &PlanInstance, id: &str, fallback_base: usize) -> Option<usize> {
    if let Some(index) = instance
        .edge_ids
        .iter()
        .position(|candidate| candidate.as_deref() == Some(id))
    {
        return Some(index);
    }
    id.strip_prefix("edge:")
        .and_then(|index| index.parse::<usize>().ok())
        .filter(|index| *index >= fallback_base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance() -> PlanInstance {
        PlanInstance {
            target: String::new(),
            plan: std::sync::Arc::new(TessPlan::default()),
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        }
    }

    #[test]
    fn draw_groups_instance_every_body_of_a_shared_plan() {
        let plan = std::sync::Arc::new(TessPlan::default());
        let instance = |placement| PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&plan),
            face_colors: Vec::new(),
            body: 0,
            placement,
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let placements = vec![
            nodecraft_display::Placement::identity(),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [10.0, 0.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [0.0, 20.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Mirror {
                plane: "yz",
                offset: 10.0,
            }),
        ];
        let plans: Vec<PlanInstance> = placements.iter().cloned().map(instance).collect();
        let groups = plan_draw_groups(&plans, None);
        assert_eq!(
            groups.len(),
            1,
            "all bodies of one plan and colouring are one instanced draw group",
        );
        assert_eq!(
            groups[0].bodies,
            vec![0, 1, 2, 3],
            "the group keeps every body, in body order",
        );
        assert!(
            std::sync::Arc::ptr_eq(&groups[0].plan, &plan),
            "the resident mesh is shared, not duplicated",
        );
        assert_eq!(groups[0].matrices.len(), 4);
        for (matrix, placement) in groups[0].matrices.iter().zip(&placements) {
            assert_eq!(*matrix, instance_matrix(placement));
        }
    }

    /// Every copy of one box shares one plan, so the copies are one chunk with
    /// one instance per copy. Each instance matrix must place the plan exactly
    /// where the CPU display path places its edges.
    #[test]
    fn circular_pattern_copies_are_one_instanced_draw_group_matching_the_cpu_transform() {
        use nodecraft_display::{DisplayQuality, PlanCache};

        let request: nodecraft_protocol::Request = serde_json::from_value(serde_json::json!({
            "requestId": 1, "targets": ["pattern"], "meshTargets": [], "binaryMesh": false,
            "nodes": [
                {"id": "box", "type": "primitive", "values": {"primitiveType": "box", "length": 10, "width": 10, "height": 10}, "cache": {"key": "box"}},
                {"id": "move", "type": "translate", "values": {"x": 20}, "cache": {"key": "move"}},
                {"id": "pattern", "type": "circularpattern", "values": {"count": 4, "angle": 360, "axis": "z"}, "cache": {"key": "pattern"}}
            ],
            "connections": [
                {"from": "box", "to": "move", "in": "shape", "out": "shape"},
                {"from": "move", "to": "pattern", "in": "shape", "out": "shape"}
            ]
        })).expect("circular pattern request");
        let output = nodecraft_engine::evaluate_exact(
            request,
            nodecraft_engine::EngineCache::new(),
            None,
        );
        assert!(output.response.node_errors.is_empty(), "{:?}", output.response.node_errors);
        let part = output.exact_parts.get("pattern").expect("circular pattern exact part").clone();
        let mut cache = PlanCache::new(16);
        let planned = nodecraft_display::plan_part(&part, DisplayQuality::Full(0.35), &mut cache)
            .expect("display plan");
        assert_eq!(planned.plans.len(), 4);
        for instance in &planned.plans {
            assert!(
                std::sync::Arc::ptr_eq(&planned.plans[0].plan, &instance.plan),
                "every copy must instance the same plan",
            );
        }

        let groups = plan_draw_groups(&planned.plans, None);
        assert_eq!(groups.len(), 1, "four copies of one box are one chunk");
        assert_eq!(groups[0].bodies, vec![0, 1, 2, 3]);
        assert_eq!(groups[0].matrices.len(), 4, "one instance per copy");
        for (matrix, instance) in groups[0].matrices.iter().zip(&planned.plans) {
            assert_eq!(*matrix, instance_matrix(&instance.placement));
        }

        // Edge geometry is no longer transformed per point on the CPU. Each
        // copy shares the plan's local-space polylines and the same instance
        // matrix that places its triangles.
        assert_eq!(
            planned.bounds_mesh.vertices.len(),
            8,
            "the display payload is a bounds carrier, not an edge copy",
        );
        assert!(planned.bounds_mesh.brep_edges.is_empty());
        let edge_segments = |instance: &PlanInstance| -> usize {
            instance.plan.edges.iter().map(|edge| edge.len().saturating_sub(1)).sum()
        };
        let shared_segments = edge_segments(&planned.plans[0]);
        assert!(shared_segments > 0);
        for instance in &planned.plans {
            assert_eq!(edge_segments(instance), shared_segments);
            assert!(std::sync::Arc::ptr_eq(&planned.plans[0].plan, &instance.plan));
        }
    }

    #[test]
    fn draw_groups_instance_bodies_with_different_colourings() {
        let plan = std::sync::Arc::new(TessPlan::default());
        let instance = |face_colors: Vec<[f32; 3]>| PlanInstance {
            body: 0,
            target: "test".into(),
            plan: std::sync::Arc::clone(&plan),
            face_colors,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let red = vec![[1.0, 0.0, 0.0]];
        let green = vec![[0.0, 1.0, 0.0]];
        let plans = vec![
            instance(red),
            instance(green),
            instance(Vec::new()),
        ];
        let groups = plan_draw_groups(&plans, None);
        assert_eq!(
            groups.len(),
            1,
            "colour is a shader lookup, not a grouping key",
        );
        assert_eq!(groups[0].bodies, vec![0, 1, 2]);
        assert_eq!(groups[0].matrices.len(), 3);
    }

    #[test]
    fn same_plan_with_a_new_placement_is_not_the_same_display() {
        let plan = std::sync::Arc::new(TessPlan::default());
        let base = PlanInstance {
            target: "t".into(),
            plan: std::sync::Arc::clone(&plan),
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let moved = PlanInstance {
            placement: nodecraft_display::Placement::from_ops([nodecraft_display::DisplayTransform::Place {
                offset: [5.0, 0.0, 0.0],
                rotation: [0.0; 3],
            }]),
            ..base.clone()
        };
        assert!(same_plan_set(std::slice::from_ref(&base), std::slice::from_ref(&base)));
        assert!(
            !same_plan_set(std::slice::from_ref(&base), std::slice::from_ref(&moved)),
            "a placement change must not be treated as the same GPU display",
        );
    }

    #[test]
    fn instance_cache_slots_are_keyed_by_pass() {
        let main = InstanceSlot::Main;
        let ghost = InstanceSlot::Ghost;
        assert_ne!(main, ghost, "the ghost and main passes own separate instance data");
    }

    #[test]
    fn colour_table_gives_every_body_its_own_face_slice() {
        let plan = std::sync::Arc::new(TessPlan {
            faces: vec![Default::default(); 2],
            ..Default::default()
        });
        let instance = |body: u64, face_colors: Vec<[f32; 3]>| PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&plan),
            face_colors,
            body,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let red = [1.0, 0.0, 0.0];
        let green = [0.0, 1.0, 0.0];
        let blue = [0.0, 0.0, 1.0];
        let plans = vec![
            instance(11, vec![red, green]),
            instance(22, vec![blue]),
            instance(33, Vec::new()),
        ];
        let (colours, offsets) = colour_table_data(&plans, None, false);
        assert_eq!(offsets, vec![0, 2, 4], "each body starts its own slice");
        assert_eq!(
            colours,
            vec![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.17, 0.18, 0.195, 0.0],
                [0.17, 0.18, 0.195, 0.0],
                [0.17, 0.18, 0.195, 0.0],
            ],
            "authored faces keep their colour and missing faces use the default",
        );

        // The ignore-look override is the same table fill for every face.
        let (colours, _) = colour_table_data(&plans, Some([0.4, 0.5, 0.6]), false);
        assert!(colours.iter().all(|colour| *colour == [0.4, 0.5, 0.6, 0.0]));

        // Face-colours mode fills the table from the shared hash, per body and
        // per face.
        let (colours, offsets) = colour_table_data(&plans, None, true);
        for (body, instance) in plans.iter().enumerate() {
            for face in 0..2u32 {
                let expected = face_colour(instance.body, face);
                assert_eq!(
                    colours[offsets[body] as usize + face as usize],
                    [expected[0], expected[1], expected[2], 0.0],
                    "body {body} face {face} must use its own hashed colour",
                );
            }
        }
        assert_ne!(
            face_colour(plans[0].body, 0),
            face_colour(plans[1].body, 0),
            "two bodies must not share face 0's colour",
        );
        assert_ne!(
            face_colour(plans[0].body, 0),
            face_colour(plans[0].body, 1),
            "adjacent faces of one body must differ",
        );
    }

    /// Face colours are a function of the body identity and face number, not of
    /// the plan's position in the list. Two evaluations of the same plan set,
    /// and the same set with its plan list reordered, must produce identical
    /// tables.
    #[test]
    fn face_colour_table_is_deterministic_across_evaluations_and_plan_order() {
        let shared = std::sync::Arc::new(TessPlan {
            faces: vec![Default::default(); 2],
            ..Default::default()
        });
        let instance = |body: u64| PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&shared),
            face_colors: Vec::new(),
            body,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let first = vec![instance(1), instance(2)];
        let again = vec![instance(1), instance(2)];
        let reordered = vec![instance(2), instance(1)];
        let table = |plans: &[PlanInstance]| {
            let (colours, offsets) = colour_table_data(plans, None, true);
            plans
                .iter()
                .enumerate()
                .map(|(index, instance)| {
                    let start = offsets[index] as usize;
                    (instance.body, colours[start..start + 2].to_vec())
                })
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        let first_table = table(&first);
        assert_eq!(first_table, table(&again), "the same identities re-evaluate");
        assert_eq!(first_table, table(&reordered), "reordering must not recolour");
        assert_ne!(
            first_table[&1][0],
            first_table[&2][0],
            "two instances of one shared plan differ on face 0",
        );
        assert_ne!(
            first_table[&1][0],
            first_table[&1][1],
            "adjacent faces of one instance differ",
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
    fn a_resident_plan_set_activates_without_a_staged_swap() {
        let a = instance();
        let b = instance();
        let a_key = std::sync::Arc::as_ptr(&a.plan) as usize;
        let b_key = std::sync::Arc::as_ptr(&b.plan) as usize;
        let resident = std::collections::HashSet::from([a_key, b_key]);
        assert!(
            plans_resident(&[a.clone(), b.clone()], |key| resident.contains(&key)),
            "a set whose GPU meshes are all resident needs no staging",
        );
        assert!(
            !plans_resident(&[a, b], |key| key == a_key),
            "one missing GPU mesh still needs preparation",
        );
        assert!(
            plans_resident(&[], |_| false),
            "an empty set has nothing to upload and activates immediately",
        );
    }

    /// A non-resident set must keep the old plan chunks drawn until the new
    /// chunks are resident. The viewport stages its edge payload against this
    /// same activation counter, so edges cannot overtake the plan swap.
    #[test]
    fn a_staged_plan_set_keeps_the_old_chunks_until_activation() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let old = box_plan_sized(10.0);
        let instance = |plan: std::sync::Arc<TessPlan>| PlanInstance {
            target: "test".into(),
            plan,
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        state.set_plans(vec![instance(std::sync::Arc::clone(&old))], true, None, 1);
        state.prepare_chunks(&device, &queue);
        let old_activation = state.activation();
        assert_eq!(state.chunks.len(), 1);
        assert!(std::sync::Arc::ptr_eq(&state.plans[0].plan, &old));

        let new = box_plan_sized(20.0);
        let status = state.set_plans(vec![instance(new)], true, None, 2);
        assert_eq!(status, PlanSetState::Staged);
        assert_eq!(
            state.activation(),
            old_activation,
            "staging a non-resident set must not activate the display",
        );
        assert!(
            std::sync::Arc::ptr_eq(&state.plans[0].plan, &old),
            "the old chunks stay painted while the new set uploads",
        );

        state.prepare_chunks(&device, &queue);
        assert!(
            state.activation() > old_activation,
            "the prepared set activates as one step",
        );
        assert_eq!(state.display_snapshot(), 2);
        assert!(!std::sync::Arc::ptr_eq(&state.plans[0].plan, &old));
    }

    /// A colour-only edit must rewrite the shared table, not the chunk list.
    /// The same plan drawn by three bodies with different face colours is one
    /// instanced chunk before and after the edit; its mesh, instance, body and
    /// face-id buffers are byte-for-byte the same allocations.
    #[test]
    fn a_colour_only_edit_keeps_the_chunk_and_instance_buffers() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(10.0);
        let faces = shared.faces.len();
        let instance = |face_colors: Vec<[f32; 3]>, x: f64| PlanInstance {
            body: 0,
            target: "test".into(),
            plan: std::sync::Arc::clone(&shared),
            face_colors,
            placement: nodecraft_display::Placement::from_ops([nodecraft_display::DisplayTransform::Place {
                offset: [x, 0.0, 0.0],
                rotation: [0.0; 3],
            }]),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let red = vec![[0.8, 0.05, 0.05]; faces];
        let green = vec![[0.05, 0.8, 0.05]; faces];
        let blue = vec![[0.05, 0.05, 0.8]; faces];
        let mut plans = vec![
            instance(red, -20.0),
            instance(green, 0.0),
            instance(blue, 20.0),
        ];
        state.set_plans(plans.clone(), true, None, 1);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.debug_counts().chunks, 1);
        assert_eq!(state.debug_counts().instances, 3);
        let before = state.debug_counts();
        let mesh = std::sync::Arc::clone(&state.chunks[0].mesh);
        let instances = std::sync::Arc::clone(&state.chunks[0].instances);
        let bodies = std::sync::Arc::clone(&state.chunks[0].bodies);
        let face_ids = std::sync::Arc::clone(&state.chunks[0].face_ids);
        let table = std::sync::Arc::clone(&state.chunks[0].table);

        let yellow = [0.9, 0.8, 0.1];
        plans[1].face_colors = vec![yellow; faces];
        assert_eq!(
            state.set_plans(plans, true, None, 1),
            PlanSetState::Immediate,
            "a colour-only edit must not stage geometry",
        );
        state.prepare_chunks(&device, &queue);

        let after = state.debug_counts();
        assert_eq!(after.chunks, 1, "colour does not split the chunk");
        assert_eq!(after.instances, 3);
        assert_eq!(
            after.chunk_builds, before.chunk_builds,
            "a colour-only edit must not rebuild the chunk list",
        );
        assert_eq!(after.uploads, 0, "no mesh vertex upload is needed");
        assert!(
            after.table_uploads > before.table_uploads,
            "the shared table is the only upload",
        );
        assert!(std::sync::Arc::ptr_eq(&state.chunks[0].mesh, &mesh));
        assert!(std::sync::Arc::ptr_eq(&state.chunks[0].instances, &instances));
        assert!(std::sync::Arc::ptr_eq(&state.chunks[0].bodies, &bodies));
        assert!(std::sync::Arc::ptr_eq(&state.chunks[0].face_ids, &face_ids));
        assert!(
            std::sync::Arc::ptr_eq(&state.chunks[0].table, &table),
            "the table is rewritten in place, so its bind group stays valid",
        );
        let bytes = read_buffer(&device, &queue, &state.chunks[0].table.colours);
        let colours: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        assert_eq!(colours[faces], [yellow[0], yellow[1], yellow[2], 0.0]);
        assert_eq!(colours[0], [0.8, 0.05, 0.05, 0.0]);
    }

    fn chunk_models(
        state: &PlanState,
        index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<[[f32; 4]; 4]> {
        let chunk = &state.chunks[index];
        let instance_floats = std::mem::size_of::<GpuInstance>() / std::mem::size_of::<f32>();
        let bytes = read_buffer(device, queue, &chunk.instances);
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        (0..chunk.instance_count as usize)
            .map(|instance| {
                let start = instance * instance_floats;
                let mut model = [[0.0f32; 4]; 4];
                for (column, values) in model.iter_mut().enumerate() {
                    values.copy_from_slice(&floats[start + column * 4..start + column * 4 + 4]);
                }
                model
            })
            .collect()
    }

    fn chunk_bodies(
        state: &PlanState,
        index: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<u32> {
        let chunk = &state.chunks[index];
        let bytes = read_buffer(device, queue, &chunk.bodies);
        let ids: &[u32] = bytemuck::cast_slice(&bytes);
        ids[..chunk.instance_count as usize].to_vec()
    }

    #[test]
    fn shared_plan_bodies_stay_instances_through_staged_and_immediate_commits() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(10.0);
        let instance = |placement| PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&shared),
            face_colors: Vec::new(),
            body: 0,
            placement,
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let mirror = nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Mirror {
            plane: "yz",
            offset: 10.0,
        });
        let one = || vec![instance(nodecraft_display::Placement::identity())];
        let two = || vec![instance(nodecraft_display::Placement::identity()), instance(mirror.clone())];
        let expected = vec![
            instance_matrix(&nodecraft_display::Placement::identity()),
            instance_matrix(&mirror),
        ];
        state.set_plans(two(), true, None, 2);
        state.prepare_chunks(&device, &queue);
        assert_eq!(
            state.chunks.len(),
            1,
            "staged activation builds one chunk for both bodies of the shared plan",
        );
        assert_eq!(state.chunks[0].instance_count, 2);
        assert_eq!(chunk_models(&state, 0, &device, &queue), expected);
        assert_eq!(chunk_bodies(&state, 0, &device, &queue), vec![0, 1]);

        assert_eq!(state.set_plans(one(), true, None, 3), PlanSetState::Immediate);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.chunks.len(), 1);
        assert_eq!(state.chunks[0].instance_count, 1);
        assert_eq!(chunk_models(&state, 0, &device, &queue), vec![expected[0]]);

        assert_eq!(state.set_plans(two(), true, None, 4), PlanSetState::Immediate);
        state.prepare_chunks(&device, &queue);
        assert_eq!(
            state.chunks.len(),
            1,
            "the resident immediate path must build the same one chunk as staging",
        );
        assert_eq!(state.debug_counts().instances, 2);
        assert_eq!(state.chunks[0].instance_count, 2);
        assert_eq!(
            chunk_models(&state, 0, &device, &queue),
            expected,
            "the 2 -> 1 -> 2 toggle must keep both placements in body order",
        );
        assert_eq!(chunk_bodies(&state, 0, &device, &queue), vec![0, 1]);
    }

    #[test]
    fn shared_plan_keeps_two_translates_and_a_mirror_as_one_instance_list() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(10.0);
        let instance = |placement| PlanInstance {
            target: "test".into(),
            plan: std::sync::Arc::clone(&shared),
            face_colors: Vec::new(),
            body: 0,
            placement,
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let placements = vec![
            nodecraft_display::Placement::identity(),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [10.0, 0.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                offset: [0.0, 20.0, 0.0],
                rotation: [0.0; 3],
            }),
            nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Mirror {
                plane: "yz",
                offset: 10.0,
            }),
        ];
        let expected: Vec<[[f32; 4]; 4]> =
            placements.iter().map(instance_matrix).collect();
        let four = || placements.iter().cloned().map(instance).collect::<Vec<_>>();

        // Staged activation (the first set has to upload the resident mesh).
        state.set_plans(four(), true, None, 1);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.chunks.len(), 1, "one chunk for the whole shared plan");
        assert_eq!(state.chunks[0].instance_count, 4);
        assert_eq!(chunk_models(&state, 0, &device, &queue), expected);
        assert_eq!(chunk_bodies(&state, 0, &device, &queue), vec![0, 1, 2, 3]);

        // Immediate resident commit after a one-body set.
        state.set_plans(vec![instance(nodecraft_display::Placement::identity())], true, None, 2);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.set_plans(four(), true, None, 3), PlanSetState::Immediate);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.chunks.len(), 1);
        assert_eq!(state.chunks[0].instance_count, 4);
        assert_eq!(
            chunk_models(&state, 0, &device, &queue),
            expected,
            "the immediate path must keep every placement, including the mirror",
        );
        assert_eq!(chunk_bodies(&state, 0, &device, &queue), vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_thousand_bodies_sharing_a_plan_are_one_chunk_and_one_draw() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let shared = box_plan_sized(1.0);
        let plans: Vec<PlanInstance> = (0..1000)
            .map(|body| PlanInstance {
                target: "scatter".into(),
                plan: std::sync::Arc::clone(&shared),
                face_colors: Vec::new(),
                body: 0,
                placement: if body == 0 {
                    nodecraft_display::Placement::identity()
                } else {
                    nodecraft_display::Placement::from(nodecraft_display::DisplayTransform::Place {
                        offset: [body as f64 * 2.0, 0.0, 0.0],
                        rotation: [0.0; 3],
                    })
                },
                edge_ids: std::sync::Arc::from(Vec::new()),
                edge_classes: std::sync::Arc::from(Vec::new()),
            })
            .collect();
        state.set_plans(plans, true, None, 1);
        state.prepare_chunks(&device, &queue);
        let debug = state.debug_counts();
        assert_eq!(debug.plans, 1000);
        assert_eq!(debug.chunks, 1, "1000 bodies of one plan are one chunk");
        assert_eq!(debug.instances, 1000, "the chunk has one instance per body");
        assert_eq!(state.chunks.len(), 1, "one chunk means one draw per pass");
        assert_eq!(state.chunks[0].instance_count, 1000);
        let plan_key = std::sync::Arc::as_ptr(&shared) as usize;
        assert_eq!(
            state.resident[&plan_key].instances.len(),
            1,
            "one instance buffer for the group, not one per body",
        );
        assert_eq!(
            chunk_bodies(&state, 0, &device, &queue),
            (0..1000).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn a_newer_staged_plan_set_replaces_the_older_one_before_activation() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let instance = |plan: std::sync::Arc<TessPlan>, snapshot: u64| PlanInstance {
            target: format!("snapshot-{snapshot}"),
            plan,
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::identity(),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        let first = box_plan_sized(10.0);
        state.set_plans(
            vec![instance(std::sync::Arc::clone(&first), 1)],
            true,
            None,
            1,
        );
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.display_snapshot(), 1);

        // Two newer states arrive before either can upload. Only the newest
        // pending pair may survive and activate as a whole.
        state.set_plans(vec![instance(box_plan_sized(20.0), 2)], true, None, 2);
        state.set_plans(vec![instance(box_plan_sized(30.0), 3)], true, None, 3);
        assert_eq!(state.display_snapshot(), 1);
        assert_eq!(state.debug_counts().pending_snapshot, 3);
        state.prepare_chunks(&device, &queue);
        assert_eq!(state.display_snapshot(), 3);
        assert_eq!(state.plans[0].target, "snapshot-3");
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

    fn box_plan_sized(size: f64) -> std::sync::Arc<TessPlan> {
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

    fn box_plan() -> std::sync::Arc<TessPlan> {
        box_plan_sized(10.0)
    }

    /// Every undirected triangle edge is drawn exactly once, including the
    /// sphere pole fans the old valence skip blanked. The expected edges come
    /// straight from the plan's faces, so the test is independent of the
    /// helper's emission order.
    #[test]
    fn triangle_lines_emit_every_unique_sphere_edge_including_pole_fans() {
        use nodecraft_solidcore::solidcore::{math::Point3, primitive::api_primitives as api};
        const RADIUS: f64 = 20.0;
        let mut kernel = nodecraft_kernel_profile::kernel();
        let brep = api::create_sphere(&mut kernel, Point3::new(0.0, 0.0, 0.0), RADIUS)
            .expect("sphere");
        // 0.025 keeps the pole fan well above the old valence >= 10 threshold.
        let plan = nodecraft_layers::plan_kernel(&mut kernel, &brep, 0.025, None)
            .expect("sphere display plan")
            .plan;

        // GpuMesh packs each plan face's vertices contiguously in plan order.
        let mut vertex_face = Vec::new();
        let mut face_bases = Vec::with_capacity(plan.faces.len());
        for face in &plan.faces {
            face_bases.push(vertex_face.len() as u32);
            vertex_face.extend(std::iter::repeat_n(face.face, face.uv.len()));
        }
        let indices = triangle_line_indices(&plan, &vertex_face, 0..vertex_face.len() as u32);

        let mut expected: std::collections::BTreeSet<(u32, u32)> =
            std::collections::BTreeSet::new();
        let mut triangles = 0usize;
        let mut max_valence = 0usize;
        let mut fan_node = 0u32;
        for (face, &base) in plan.faces.iter().zip(&face_bases) {
            let mut valence = vec![0usize; face.uv.len()];
            for triangle in &face.triangles {
                for node in triangle {
                    if let Some(count) = valence.get_mut(*node as usize) {
                        *count += 1;
                    }
                }
            }
            if let Some((node, &count)) =
                valence.iter().enumerate().max_by_key(|(_, count)| **count)
            {
                if count > max_valence {
                    max_valence = count;
                    fan_node = base + node as u32;
                }
            }
            for triangle in &face.triangles {
                let [a, b, c] = *triangle;
                for (from, to) in [(a, b), (b, c), (c, a)] {
                    expected.insert((base + from.min(to), base + from.max(to)));
                }
                triangles += 1;
            }
        }
        assert!(triangles > 0, "sphere plan has no triangles");
        assert!(
            max_valence >= 10,
            "sphere pole fan must exercise the old valence >= 10 skip (max valence {max_valence})"
        );

        assert_eq!(
            indices.len(),
            2 * expected.len(),
            "one index pair per unique triangle edge: {} indices for {} unique edges ({} triangles)",
            indices.len(),
            expected.len(),
            triangles,
        );
        assert!(
            indices.len() > 3 * triangles / 2,
            "unique-edge emission must not duplicate shared edges: {} indices for {} triangles",
            indices.len(),
            triangles,
        );

        let emitted: std::collections::BTreeSet<(u32, u32)> = indices
            .chunks_exact(2)
            .map(|pair| (pair[0].min(pair[1]), pair[0].max(pair[1])))
            .collect();
        assert_eq!(emitted, expected, "emitted edges differ from the plan's unique edges");

        // Every edge incident on the pole copy is a fan spoke or cap-ring edge;
        // the old skip emitted none of them.
        let expected_fan: std::collections::BTreeSet<(u32, u32)> = expected
            .iter()
            .filter(|(a, b)| *a == fan_node || *b == fan_node)
            .copied()
            .collect();
        assert!(!expected_fan.is_empty(), "pole node {fan_node} has no plan edges");
        let emitted_fan = expected_fan.iter().filter(|edge| emitted.contains(edge)).count();
        assert_eq!(
            emitted_fan,
            expected_fan.len(),
            "pole fan edges are missing from the wire indices (node {fan_node}, valence {max_valence})",
        );
    }

    fn read_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
    ) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plan test readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("plan test readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        receiver.recv().expect("map callback").expect("map");
        let bytes = slice.get_mapped_range().expect("map range").to_vec();
        readback.unmap();
        bytes
    }

    /// A GPU mesh's fill must not ride on a paint frame's encoder: if that frame
    /// is dropped (surface acquire skips it), the resident entry would keep an
    /// empty vertex buffer and the solid disappears while its edge overlay still
    /// draws. `prepare` therefore submits the fill itself.
    #[test]
    fn a_resident_mesh_is_filled_even_when_the_prepare_frame_is_dropped() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let plan = box_plan();
        state.set_plans(
            vec![PlanInstance {
                target: "test".into(),
                plan,
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
        // This encoder stands for the paint frame that is never submitted.
        let frame = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dropped prepare frame"),
        });
        state.prepare_chunks(&device, &queue);
        drop(frame);
        let mesh = state.chunks.first().expect("activated chunk").mesh.clone();
        let bytes = read_buffer(&device, &queue, &mesh.vertices);
        let stride = solidcraft_viewport_gpu::VERTEX_STRIDE as usize;
        let filled = bytes.chunks_exact(stride).any(|vertex| {
            let component = |at: usize| {
                f32::from_ne_bytes(vertex[at..at + 4].try_into().expect("component"))
            };
            component(0) != 0.0 || component(4) != 0.0 || component(8) != 0.0
        });
        assert!(
            filled,
            "the resident GPU mesh must be filled even when no paint frame is submitted",
        );
    }

    #[test]
    fn changing_the_edge_width_only_updates_the_uniform_value() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        state.set_plans(vec![instance()], true, None, 1);
        state.prepare_chunks(&device, &queue);
        let before = state.debug_counts();
        state.set_edge_width(3.5);
        let after = state.debug_counts();
        assert_eq!(after.edge_width_px, 3.5);
        assert_eq!(
            after.chunk_builds, before.chunk_builds,
            "an edge-width change must not rebuild a chunk",
        );
        assert_eq!(after.uploads, before.uploads);
    }

    #[test]
    fn toggling_show_seam_edges_only_updates_the_uniform_mask() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let plan = box_plan();
        state.set_plans(
            vec![PlanInstance {
                target: "seams".into(),
                plan,
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
        state.prepare_chunks(&device, &queue);
        let before = state.debug_counts();
        state.set_show_seam_edges(true);
        let after = state.debug_counts();
        assert_eq!(
            after.chunk_builds, before.chunk_builds,
            "a seam setting must not rebuild a chunk",
        );
        assert_eq!(
            after.edge_uploads, before.edge_uploads,
            "a seam setting must not re-upload an edge",
        );
        assert_eq!(after.uploads, before.uploads);
        assert_eq!(after.edge_pick_entries, before.edge_pick_entries);
    }


    #[test]
    fn moving_a_body_uploads_no_edge_data() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let plan = box_plan_sized(10.0);
        let at = |x: f64| PlanInstance {
            target: "move".into(),
            plan: std::sync::Arc::clone(&plan),
            face_colors: Vec::new(),
            body: 0,
            placement: nodecraft_display::Placement::from_ops([
                nodecraft_display::DisplayTransform::Place {
                    offset: [x, 0.0, 0.0],
                    rotation: [0.0; 3],
                },
            ]),
            edge_ids: std::sync::Arc::from(Vec::new()),
            edge_classes: std::sync::Arc::from(Vec::new()),
        };
        state.set_plans(vec![at(0.0)], true, None, 1);
        state.prepare_chunks(&device, &queue);
        let before = state.debug_counts();
        assert!(before.edge_uploads > 0, "the plan's edges upload once");
        assert!(before.edge_pick_entries > 1, "plan edges are pickable");

        state.set_plans(vec![at(25.0)], true, None, 2);
        state.prepare_chunks(&device, &queue);
        let after = state.debug_counts();
        assert_eq!(
            after.edge_uploads, before.edge_uploads,
            "a placement-only move must not upload edge data",
        );
        assert_eq!(after.edge_pick_entries, before.edge_pick_entries);
        assert!(after.chunk_builds > before.chunk_builds, "instances were rebuilt");
    }

    #[test]
    fn all_edge_draws_come_from_the_ribbon_pipeline() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("skip: no adapter");
            return;
        };
        let tess = TessGpu::new("test", device.clone(), queue.clone()).expect("tess pipeline");
        let ready = std::sync::Arc::new(std::sync::OnceLock::new());
        ready.set(tess).ok().expect("tess once");
        let mut state = PlanState::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            1,
            wgpu::TextureFormat::Depth32Float,
            false,
            Some(ready),
        );
        let plan = box_plan_sized(10.0);
        state.set_plans(
            vec![PlanInstance {
                target: "edges".into(),
                plan,
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
        state.prepare_chunks(&device, &queue);
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("edge-draw-color"),
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("edge-draw-depth"),
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("edge-draw"),
        });
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("edge-draw-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
            state.paint_edges(&mut pass, true);
            state.set_edge_selection(vec!["edge:0".into()], None);
            state.paint_edges(&mut pass, true);
        }
        queue.submit([encoder.finish()]);
        let debug = state.debug_counts();
        assert!(debug.edge_draws >= 2, "standard and highlighted edges both draw");
    }


    #[test]
    fn every_edge_kind_width_derives_from_the_one_setting() {
        assert_eq!(edge_widths(1.0), (1.0, 2.0, 3.0, 2.5));
        assert_eq!(edge_widths(2.0), (2.0, 4.0, 6.0, 5.0));
    }

}
