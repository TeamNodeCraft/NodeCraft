//! The one GPU ribbon path for every B-rep edge the viewport draws.
//!
//! Edge polylines are uploaded once per plan in local space and instanced with
//! the same body placement matrix as the plan's triangles. The vertex shader
//! expands each segment into a screen-width ribbon; the width reaches it as a
//! uniform, so a slider move never rebuilds or re-uploads anything. Picking
//! shares [`EDGE_RIBBON_WGSL`] so the drawn and pickable bands have exactly the
//! same geometry.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;

use crate::adapter::DisplayEdge;
use crate::gpu_viewport::{
    GpuInstance, edge_depth, edge_highlight_depth, ghost_face_depth, identity_matrix,
};
use nodecraft_layers::tess::TessPlan;

/// The narrowest band ever drawn, in physical pixels. A one-pixel-wide quad can
/// fall exactly between two pixel centres along a segment and cover none of
/// them, which reads as a dashed edge; SolidCraft's standard edge is 1.3 px for
/// the same reason. The requested width still controls the opacity, so a 1 px
/// setting draws a 1.3 px band at 1/1.3 alpha instead of dropping out.
pub(crate) const EDGE_MIN_DRAWN_PX: f32 = 1.3;

/// The ribbon vertex function. `vp` is the GPU-space view-projection, `ca`/`cb`
/// are the segment's clip-space ends, `screen` is the target size in physical
/// pixels and `width_px` is the desired band width in those same pixels. The
/// returned position is clip space, so the offset stays constant on screen at
/// every depth; `alpha` fades sub-pixel lines the way SolidCraft does.
///
/// The same source is concatenated into the visible edge shader and the pick
/// edge shader. Do not copy this function into either shader.
pub(crate) const EDGE_RIBBON_WGSL: &str = r#"
// The narrowest band drawn, in physical pixels. Keep this equal to
// EDGE_MIN_DRAWN_PX: a 1 px quad can fall between pixel centres and vanish.
const MIN_DRAWN_PX: f32 = 1.3;

// Per-edge class bits. Keep these equal to PlanEdgeClass::SEAM/SMOOTH.
const EDGE_CLASS_SEAM: u32 = 1u;

// The one draw/pick rule: a seam is hidden unless the uniform mask asks for it.
// Smooth/tangent joins are not masked yet and stay drawn.
fn edge_class_hidden(class_bits: u32, mask: u32) -> bool {
    return (class_bits & EDGE_CLASS_SEAM) != 0u && (mask & EDGE_CLASS_SEAM) == 0u;
}

struct EdgeRibbon {
    position: vec4<f32>,
    alpha: f32,
};

fn edge_ribbon(ca_in: vec4<f32>, cb_in: vec4<f32>, screen: vec2<f32>, width_px: f32, vi: u32) -> EdgeRibbon {
    var ca = ca_in;
    var cb = cb_in;
    var visible = 1.0;
    if (ca.w < 1e-4 && cb.w < 1e-4) {
        visible = 0.0;
    } else if (ca.w < 1e-4) {
        let k = (1e-4 - ca.w) / (cb.w - ca.w);
        ca = mix(ca, cb, k);
    } else if (cb.w < 1e-4) {
        let k = (1e-4 - cb.w) / (ca.w - cb.w);
        cb = mix(cb, ca, k);
    }

    let half = max(screen * 0.5, vec2<f32>(0.5));
    let sa = ca.xy / max(ca.w, 1e-6) * half;
    let sb = cb.xy / max(cb.w, 1e-6) * half;
    var dir = sb - sa;
    let len = length(dir);
    if (len < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / len;
    }
    let nrm = vec2<f32>(-dir.y, dir.x);

    var ts = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var ss = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let corner = vi % 6u;
    let t = ts[corner];
    let s = ss[corner];

    // A line thinner than a pixel is drawn 1.3 px wide and faded by the width
    // it lacks, so sub-pixel edges do not flicker without MSAA. A 1 px request
    // gets the same floor (and the same fade) because a 1 px quad can fall
    // between pixel centres on a curved segment and vanish.
    let wd = max(width_px, MIN_DRAWN_PX);
    // Along-segment half extension: + at b, - at a.
    let s_ext = select(-wd * 0.5, wd * 0.5, t > 0.5);
    // The quad runs half a width past each end so neighbouring segments
    // overlap at a joint instead of leaving a notch. Its clip position must be
    // extrapolated along the segment, not copied from the endpoint: copying the
    // endpoint's z/w bends the depth ramp over `L + wd` px, and on a wide band
    // the face wins back the first pixels after every shared vertex (the
    // dashed seam). Extrapolating x, y, z and w keeps the ramp exactly on the
    // segment's depth plane while the extension still closes the joint.
    var t_ext = t;
    if (len > 1e-6) {
        t_ext = t + s_ext / len;
    }
    let base_ext = mix(ca, cb, t_ext);
    let off = nrm * s * wd * 0.5;

    var out: EdgeRibbon;
    if (visible < 0.5 || base_ext.w < 1e-6) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.alpha = 0.0;
        return out;
    }
    out.position = vec4<f32>(base_ext.xy + off / half * base_ext.w, base_ext.z, base_ext.w);
    out.alpha = clamp(width_px / wd, 0.0, 1.0);
    return out;
}
"#;

/// One ribbon vertex: both segment ends, the edge's pick index and its one
/// display class. Six copies of a segment are drawn with `draw(0..6*n)`; `id`
/// and `class` are flat across all six.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct EdgeVertex {
    pub a: [f32; 3],
    pub b: [f32; 3],
    /// Plan edge index, or a world edge's 1-based pick index. The edge pick
    /// shader adds the per-instance body base to this.
    pub id: u32,
    /// [`nodecraft_layers::PlanEdgeClass`] bits, uploaded once with the edge.
    pub class: u32,
}

const EDGE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 24, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 28, shader_location: 12 },
];

pub(crate) fn edge_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<EdgeVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &EDGE_VERTEX_ATTRIBUTES,
    }
}

/// The subset of `GpuInstance` the ribbon needs: the model matrix and alpha.
/// The stride and the two locations must stay identical to the plan's instance
/// buffer, which is what makes one instance buffer serve both meshes and edges.
const EDGE_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 4 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 5 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 6 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 128, shader_location: 11 },
];

pub(crate) fn edge_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &EDGE_INSTANCE_ATTRIBUTES,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct EdgeUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub extra_model: [[f32; 4]; 4],
    /// x, y: target size in physical pixels. z: band width in those pixels.
    /// w: 1 to multiply by the instance alpha, 0 to ignore it (ghost edges keep
    /// their own opacity rather than the ghost face layer's).
    pub screen: [f32; 4],
    /// Straight sRGB colour; alpha is the layer alpha.
    pub colour: [f32; 4],
    /// x: [`nodecraft_layers::PlanEdgeClass`] bits the setting shows. It is the
    /// only state a seam toggle writes; the edge buffer never changes.
    pub mask: [u32; 4],
}

impl EdgeUniforms {
    pub(crate) fn new(
        view_proj: [[f32; 4]; 4],
        extra_model: [[f32; 4]; 4],
        size_px: [f32; 2],
        width_px: f32,
        colour: [f32; 4],
        use_instance_alpha: bool,
    ) -> Self {
        Self {
            view_proj,
            extra_model,
            screen: [
                size_px[0],
                size_px[1],
                width_px,
                if use_instance_alpha { 1.0 } else { 0.0 },
            ],
            colour,
            // Seams are hidden unless the caller opts in (the default setting).
            mask: [0; 4],
        }
    }

    /// Show or hide seam-class edges. This is a uniform value only.
    #[must_use]
    pub(crate) fn show_seams(mut self, show: bool) -> Self {
        self.mask[0] = nodecraft_layers::PlanEdgeClass::mask(show);
        self
    }
}

const EDGE_RENDER_SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    extra_model: mat4x4<f32>,
    screen: vec4<f32>,
    colour: vec4<f32>,
    mask: vec4<u32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct In {
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(11) alpha: f32,
    @location(12) edge_class: u32,
};

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec4<f32>,
};

@vertex
fn vs(in: In, @builtin(vertex_index) vi: u32) -> Out {
    var out: Out;
    // The same class/mask rule the pick shader uses: hidden means not drawn.
    if (edge_class_hidden(in.edge_class, u.mask.x)) {
        out.position = vec4<f32>(0.0);
        out.colour = vec4<f32>(0.0);
        return out;
    }
    let model = u.extra_model * mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let ca = u.view_proj * model * vec4<f32>(in.a, 1.0);
    let cb = u.view_proj * model * vec4<f32>(in.b, 1.0);
    let ribbon = edge_ribbon(ca, cb, u.screen.xy, u.screen.z, vi);
    out.position = ribbon.position;
    let instance_alpha = mix(1.0, in.alpha, u.screen.w);
    out.colour = vec4<f32>(u.colour.rgb, u.colour.a * instance_alpha * ribbon.alpha);
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return in.colour;
}
"#;

/// CPU mirror of [`EDGE_RIBBON_WGSL`], used by the unit tests and nothing in the
/// render loop. Keeping the two formulas side by side is deliberate: a shader
/// change must change this function too, and the width/HiDPI tests then catch a
/// drift between drawn and pickable bands.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RibbonCorner {
    pub position: [f32; 4],
    pub alpha: f32,
}

#[cfg(test)]
pub(crate) fn ribbon_corner(
    ca_in: [f32; 4],
    cb_in: [f32; 4],
    screen: [f32; 2],
    width_px: f32,
    vi: u32,
) -> RibbonCorner {
    let mut ca = ca_in;
    let mut cb = cb_in;
    let mut visible = 1.0;
    if ca[3] < 1.0e-4 && cb[3] < 1.0e-4 {
        visible = 0.0;
    } else if ca[3] < 1.0e-4 {
        let k = (1.0e-4 - ca[3]) / (cb[3] - ca[3]);
        ca = mix4(ca, cb, k);
    } else if cb[3] < 1.0e-4 {
        let k = (1.0e-4 - cb[3]) / (ca[3] - cb[3]);
        cb = mix4(cb, ca, k);
    }
    let half = [(screen[0] * 0.5).max(0.5), (screen[1] * 0.5).max(0.5)];
    let sa = [ca[0] / ca[3].max(1.0e-6) * half[0], ca[1] / ca[3].max(1.0e-6) * half[1]];
    let sb = [cb[0] / cb[3].max(1.0e-6) * half[0], cb[1] / cb[3].max(1.0e-6) * half[1]];
    let mut dir = [sb[0] - sa[0], sb[1] - sa[1]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    if len < 1.0e-6 {
        dir = [1.0, 0.0];
    } else {
        dir = [dir[0] / len, dir[1] / len];
    }
    let nrm = [-dir[1], dir[0]];
    let corner = (vi % 6) as usize;
    let ts = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let ss = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let t = ts[corner];
    let s = ss[corner];
    let wd = width_px.max(EDGE_MIN_DRAWN_PX);
    // The joint overlap runs `wd / 2` px past each endpoint, so its clip
    // position is the linear extrapolation of the segment, not the endpoint
    // repeated: the rasterizer interpolates the quad's depth over its full
    // length, and repeating the endpoint bends that ramp behind the face near
    // every shared vertex.
    let s_ext = if t > 0.5 { wd * 0.5 } else { -wd * 0.5 };
    let t_ext = if len > 1.0e-6 { t + s_ext / len } else { t };
    let base = mix4(ca, cb, t_ext);
    let off = [nrm[0] * s * wd * 0.5, nrm[1] * s * wd * 0.5];
    if visible < 0.5 || base[3] < 1.0e-6 {
        return RibbonCorner { position: [0.0; 4], alpha: 0.0 };
    }
    RibbonCorner {
        position: [
            base[0] + off[0] / half[0] * base[3],
            base[1] + off[1] / half[1] * base[3],
            base[2],
            base[3],
        ],
        alpha: (width_px / wd).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
fn mix4(a: [f32; 4], b: [f32; 4], k: f32) -> [f32; 4] {
    std::array::from_fn(|index| a[index] + (b[index] - a[index]) * k)
}

/// CPU mirror of the shader's `edge_class_hidden`, for the draw/pick rule test.
#[cfg(test)]
pub(crate) fn edge_class_hidden(class: u32, mask: u32) -> bool {
    const SEAM: u32 = nodecraft_layers::PlanEdgeClass::SEAM.bits();
    class & SEAM != 0 && mask & SEAM == 0
}

/// CPU-side edge data, built once per plan or per display snapshot.
#[derive(Clone, Debug, Default)]
pub(crate) struct EdgeCpuBatch {
    pub vertices: Vec<EdgeVertex>,
    /// Vertex range by edge index, `(start, count)`. Empty edges have count 0.
    pub ranges: Vec<(u32, u32)>,
    /// Edge id by edge index, used for selection/hover draws.
    pub ids: Vec<String>,
    /// Whether this edge may produce a pick id (synthetic preview ids do not).
    pub pickable: Vec<bool>,
}

impl EdgeCpuBatch {
    /// Build local-space plan edges. One buffer per `Arc<TessPlan>`; every body
    /// placement is an instance matrix and never touches this data.
    pub(crate) fn from_plan(
        plan: &TessPlan,
        edge_ids: &[Option<String>],
        edge_classes: &[nodecraft_layers::PlanEdgeClass],
    ) -> Self {
        let mut batch = Self::default();
        for (edge_index, polyline) in plan.edges.iter().enumerate() {
            let class = edge_classes
                .get(edge_index)
                .copied()
                .unwrap_or_default()
                .bits();
            let start = batch.vertices.len() as u32;
            for pair in polyline.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let a = [a.x as f32, a.y as f32, a.z as f32];
                let b = [b.x as f32, b.y as f32, b.z as f32];
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= 1.0e-20 {
                    continue;
                }
                // Six vertices share both ends; the shader picks the corner.
                for _ in 0..6 {
                    batch.vertices.push(EdgeVertex { a, b, id: edge_index as u32, class });
                }
            }
            batch.ranges.push((start, batch.vertices.len() as u32 - start));
            batch.ids.push(
                edge_ids
                    .get(edge_index)
                    .and_then(Clone::clone)
                    .unwrap_or_else(|| format!("edge:{edge_index}")),
            );
            batch.pickable.push(true);
        }
        batch
    }

    /// Build a world-space fallback batch for the CPU renderer or a comparison
    /// overlay. This is not used by the plan path; plan edges stay local.
    pub(crate) fn from_display_edges(edges: &[DisplayEdge]) -> Self {
        let mut batch = Self::default();
        for (edge_index, edge) in edges.iter().enumerate() {
            let start = batch.vertices.len() as u32;
            for pair in edge.points.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let a = [a.x as f32, a.y as f32, a.z as f32];
                let b = [b.x as f32, b.y as f32, b.z as f32];
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= 1.0e-20 {
                    continue;
                }
                for _ in 0..6 {
                    batch.vertices.push(EdgeVertex {
                        a,
                        b,
                        id: edge_index as u32 + 1,
                        class: edge.class.bits(),
                    });
                }
            }
            batch.ranges.push((start, batch.vertices.len() as u32 - start));
            batch.ids.push(edge.id.clone());
            batch.pickable.push(!edge.id.starts_with("preview-edge:"));
        }
        batch
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    #[must_use]
    pub(crate) fn bytes(&self) -> u64 {
        (self.vertices.len() * std::mem::size_of::<EdgeVertex>()) as u64
    }
}

/// Edge data on the GPU. Shared by every body of one plan.
pub(crate) struct GpuEdgeBatch {
    pub buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub ranges: Vec<(u32, u32)>,
    pub bytes: u64,
}

impl GpuEdgeBatch {
    pub(crate) fn upload(
        renderer: &mut EdgeRenderer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &EdgeCpuBatch,
    ) -> Option<Arc<Self>> {
        if batch.is_empty() {
            return None;
        }
        let bytes = batch.bytes();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-edge-ribbon"),
            size: bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&batch.vertices));
        renderer.uploads = renderer.uploads.saturating_add(1);
        Some(Arc::new(Self {
            buffer,
            vertex_count: batch.vertices.len() as u32,
            ranges: batch.ranges.clone(),
            bytes,
        }))
    }

    pub(crate) fn range(&self, edge_index: usize) -> Option<(u32, u32)> {
        self.ranges.get(edge_index).copied().filter(|(_, count)| *count > 0)
    }
}

/// Which uniform colour/width a draw uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EdgeStyle {
    Standard,
    Ghost,
    Selected,
    Hover,
}

struct EdgePass {
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub(crate) struct EdgeRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Ghost edges are an `Always` overlay today (the map records this as the
    /// one deliberate difference from SolidCraft's depth-tested ghost lines).
    ghost_pipeline: wgpu::RenderPipeline,
    highlight_pipeline: wgpu::RenderPipeline,
    standard: EdgePass,
    ghost: EdgePass,
    selected: EdgePass,
    hover: EdgePass,
    identity_instance: wgpu::Buffer,
    pub uploads: u64,
    pub draws: u64,
}

impl EdgeRenderer {
    pub(crate) fn install(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: wgpu::TextureFormat,
        reversed: bool,
    ) -> Self {
        let source = format!("{EDGE_RIBBON_WGSL}\n{EDGE_RENDER_SHADER}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport-edge-ribbon"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport-edge-bind-layout"),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport-edge-layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-edge-ribbon-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(edge_vertex_layout()), Some(edge_instance_layout())],
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
            depth_stencil: Some(edge_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let highlight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-edge-highlight-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(edge_vertex_layout()), Some(edge_instance_layout())],
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
            depth_stencil: Some(edge_highlight_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport-edge-ghost-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(edge_vertex_layout()), Some(edge_instance_layout())],
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
            depth_stencil: Some(ghost_face_depth(reversed).state(depth)),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });
        let pass = |label: &str| {
            let uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<EdgeUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                }],
            });
            EdgePass { uniform, bind_group }
        };
        let identity_instance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport-edge-identity-instance"),
            size: std::mem::size_of::<GpuInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The identity instance is written by `write_uniforms`, which always
        // runs before the first paint.
        Self {
            pipeline,
            ghost_pipeline,
            highlight_pipeline,
            standard: pass("viewport-edge-standard-uniforms"),
            ghost: pass("viewport-edge-ghost-uniforms"),
            selected: pass("viewport-edge-selected-uniforms"),
            hover: pass("viewport-edge-hover-uniforms"),
            identity_instance,
            uploads: 0,
            draws: 0,
        }
    }

    pub(crate) fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        standard: EdgeUniforms,
        ghost: EdgeUniforms,
        selected: EdgeUniforms,
        hover: EdgeUniforms,
    ) {
        queue.write_buffer(&self.standard.uniform, 0, bytemuck::bytes_of(&standard));
        queue.write_buffer(&self.ghost.uniform, 0, bytemuck::bytes_of(&ghost));
        queue.write_buffer(&self.selected.uniform, 0, bytemuck::bytes_of(&selected));
        queue.write_buffer(&self.hover.uniform, 0, bytemuck::bytes_of(&hover));
        let identity = GpuInstance::from_model(identity_matrix(), 1.0);
        queue.write_buffer(&self.identity_instance, 0, bytemuck::bytes_of(&identity));
    }

    fn visible_pipeline(&self, style: EdgeStyle) -> &wgpu::RenderPipeline {
        if style == EdgeStyle::Ghost {
            &self.ghost_pipeline
        } else {
            &self.pipeline
        }
    }

    fn bind(&self, style: EdgeStyle) -> &wgpu::BindGroup {
        match style {
            EdgeStyle::Standard => &self.standard.bind_group,
            EdgeStyle::Ghost => &self.ghost.bind_group,
            EdgeStyle::Selected => &self.selected.bind_group,
            EdgeStyle::Hover => &self.hover.bind_group,
        }
    }

    /// Draw every body of one plan. The instance buffer is the plan's own; the
    /// edge buffer is shared by every body of that plan.
    pub(crate) fn paint(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        batch: &GpuEdgeBatch,
        instances: &wgpu::Buffer,
        instance_count: u32,
        style: EdgeStyle,
    ) {
        if batch.vertex_count == 0 || instance_count == 0 {
            return;
        }
        pass.set_pipeline(self.visible_pipeline(style));
        pass.set_bind_group(0, self.bind(style), &[]);
        pass.set_vertex_buffer(0, batch.buffer.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.draw(0..batch.vertex_count, 0..instance_count);
        self.draws = self.draws.saturating_add(1);
    }

    /// Draw a world-space fallback batch with the identity instance.
    pub(crate) fn paint_world(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        batch: &GpuEdgeBatch,
        style: EdgeStyle,
    ) {
        if batch.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(self.visible_pipeline(style));
        pass.set_bind_group(0, self.bind(style), &[]);
        pass.set_vertex_buffer(0, batch.buffer.slice(..));
        pass.set_vertex_buffer(1, self.identity_instance.slice(..));
        pass.draw(0..batch.vertex_count, 0..1);
        self.draws = self.draws.saturating_add(1);
    }

    /// Draw one world-space edge range for selection/hover.
    pub(crate) fn paint_world_range(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        batch: &GpuEdgeBatch,
        range: (u32, u32),
        style: EdgeStyle,
    ) {
        if range.1 == 0 {
            return;
        }
        pass.set_pipeline(&self.highlight_pipeline);
        pass.set_bind_group(0, self.bind(style), &[]);
        pass.set_vertex_buffer(0, batch.buffer.slice(..));
        pass.set_vertex_buffer(1, self.identity_instance.slice(..));
        pass.draw(range.0..range.0 + range.1, 0..1);
        self.draws = self.draws.saturating_add(1);
    }

    /// Draw one edge range for one body. Used by selection and hover highlights:
    /// a handful of ranges, no rebuilt index buffer.
    pub(crate) fn paint_range(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        batch: &GpuEdgeBatch,
        instances: &wgpu::Buffer,
        instance: u32,
        range: (u32, u32),
        style: EdgeStyle,
    ) {
        if range.1 == 0 {
            return;
        }
        pass.set_pipeline(&self.highlight_pipeline);
        pass.set_bind_group(0, self.bind(style), &[]);
        pass.set_vertex_buffer(0, batch.buffer.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.draw(range.0..range.0 + range.1, instance..instance + 1);
        self.draws = self.draws.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A horizontal segment one pixel wide at the centre of a 100 px target.
    #[test]
    fn ribbon_width_is_screen_pixels_not_clip_units() {
        let ca = [-0.5, 0.0, 0.0, 1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let corner = ribbon_corner(ca, cb, [100.0, 100.0], 2.0, 0);
        // The first corner is side -1 and t=0: half a width below the segment.
        assert!((corner.position[1] + 2.0 / 100.0).abs() < 1.0e-6, "{corner:?}");
        let opposite = ribbon_corner(ca, cb, [100.0, 100.0], 2.0, 2);
        assert!((opposite.position[1] - 2.0 / 100.0).abs() < 1.0e-6, "{opposite:?}");
        assert_eq!(corner.alpha, 1.0);
    }

    #[test]
    fn doubling_the_width_doubles_the_screen_band() {
        let ca = [-0.5, 0.0, 0.0, 1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let one = ribbon_corner(ca, cb, [200.0, 200.0], 2.0, 0);
        let four = ribbon_corner(ca, cb, [200.0, 200.0], 4.0, 0);
        assert!((four.position[1] - one.position[1] * 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn hidpi_screen_size_keeps_the_requested_physical_width() {
        // The caller passes physical pixels for both the target and the width;
        // a 2x target still yields exactly two physical pixels.
        let ca = [-0.5, 0.0, 0.0, 1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let low = ribbon_corner(ca, cb, [100.0, 100.0], 2.0, 0);
        let high = ribbon_corner(ca, cb, [200.0, 200.0], 2.0, 0);
        assert!((low.position[1] + 2.0 / 100.0).abs() < 1.0e-6);
        assert!((high.position[1] + 2.0 / 200.0).abs() < 1.0e-6);
    }

    #[test]
    fn subpixel_width_is_one_pixel_faded() {
        let ca = [-0.5, 0.0, 0.0, 1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let corner = ribbon_corner(ca, cb, [100.0, 100.0], 0.25, 0);
        assert!(
            (corner.position[1] + EDGE_MIN_DRAWN_PX / 100.0).abs() < 1.0e-6,
            "the coverage floor is the drawn width",
        );
        assert!((corner.alpha - 0.25 / EDGE_MIN_DRAWN_PX).abs() < 1.0e-6, "faded by the missing width");
    }

    #[test]
    fn a_one_pixel_edge_is_drawn_wider_than_a_pixel_to_cover_a_centre() {
        let ca = [-0.5, 0.0, 0.0, 1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let corner = ribbon_corner(ca, cb, [100.0, 100.0], 1.0, 0);
        let drawn_width_px = (corner.position[1].abs()) * 100.0;
        assert!(
            drawn_width_px > 1.0,
            "a 1 px request must draw more than 1 px so it cannot fall between pixel centres",
        );
        assert!((corner.alpha - 1.0 / EDGE_MIN_DRAWN_PX).abs() < 1.0e-6);
    }

    /// The joint overlap runs half a width past the endpoint, so its clip
    /// position must be the segment's linear extrapolation. Repeating the
    /// endpoint's depth makes the rasterizer interpolate the quad's depth over
    /// `length + width` instead of `length`; at wide widths the face then
    /// occludes the first pixels after every shared vertex (the dashed seam).
    #[test]
    fn a_segment_extension_extrapolates_depth_instead_of_repeating_the_endpoint() {
        let ca = [-0.5, 0.0, 0.2, 1.0];
        let cb = [0.5, 0.0, 0.6, 1.0];
        // A 2 px wide band on a 100 px target: half a width is 1 px, and the
        // segment is 50 px long, so the extension is 1/50 of its parameter.
        let start = ribbon_corner(ca, cb, [100.0, 100.0], 2.0, 0);
        let end = ribbon_corner(ca, cb, [100.0, 100.0], 2.0, 2);
        let k = 1.0 / 50.0;
        assert!(
            (start.position[2] - (0.2 - k * 0.4)).abs() < 1.0e-6,
            "start depth must extrapolate along the segment: {start:?}",
        );
        assert!(
            (end.position[2] - (0.6 + k * 0.4)).abs() < 1.0e-6,
            "end depth must extrapolate along the segment: {end:?}",
        );
        // The along-screen footprint is unchanged (still half a width past
        // each end); only the depth is extrapolated.
        assert!((start.position[0] - (-0.5 - k)).abs() < 1.0e-6, "{start:?}");
        assert!((end.position[0] - (0.5 + k)).abs() < 1.0e-6, "{end:?}");
    }

    #[test]
    fn the_shader_and_cpu_mirror_share_the_minimum_drawn_width() {
        let declaration = EDGE_RIBBON_WGSL
            .lines()
            .find(|line| line.trim_start().starts_with("const MIN_DRAWN_PX"))
            .expect("the shader declares its coverage floor");
        assert!(
            declaration.contains("1.3"),
            "shader floor {declaration:?} drifted from EDGE_MIN_DRAWN_PX {EDGE_MIN_DRAWN_PX}",
        );
        assert_eq!(EDGE_MIN_DRAWN_PX, 1.3);
    }

    #[test]
    fn a_segment_with_an_end_behind_the_eye_is_clipped_not_flipped() {
        let ca = [-0.5, 0.0, 0.0, -1.0];
        let cb = [0.5, 0.0, 0.0, 1.0];
        let corner = ribbon_corner(ca, cb, [100.0, 100.0], 1.0, 0);
        assert!(corner.position.iter().all(|value| value.is_finite()));
        assert!(corner.position[3] > 0.0);
    }

    #[test]
    fn a_degenerate_segment_still_has_a_finite_direction() {
        let p = [0.2, -0.3, 0.4, 1.0];
        let corner = ribbon_corner(p, p, [64.0, 64.0], 2.0, 3);
        assert!(corner.position.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn plan_edges_are_uploaded_once_per_plan_and_indexed_by_edge() {
        let plan = TessPlan {
            edges: vec![
                vec![
                    solidcore_record::Vec3::new(0.0, 0.0, 0.0),
                    solidcore_record::Vec3::new(1.0, 0.0, 0.0),
                    solidcore_record::Vec3::new(1.0, 1.0, 0.0),
                ],
                vec![
                    solidcore_record::Vec3::new(0.0, 0.0, 1.0),
                    solidcore_record::Vec3::new(1.0, 0.0, 1.0),
                ],
            ],
            ..TessPlan::default()
        };
        let ids = [Some("edge:a".to_owned()), None];
        let classes = [
            nodecraft_layers::PlanEdgeClass::NORMAL,
            nodecraft_layers::PlanEdgeClass::SEAM,
        ];
        let batch = EdgeCpuBatch::from_plan(&plan, &ids, &classes);
        assert_eq!(batch.ranges.len(), 2);
        assert_eq!(batch.vertices.len(), 6 * 3);
        assert_eq!(batch.ranges[0], (0, 12));
        assert_eq!(batch.ranges[1], (12, 6));
        assert_eq!(batch.ids, vec!["edge:a".to_owned(), "edge:1".to_owned()]);
        assert_eq!(batch.vertices[0].id, 0);
        assert_eq!(batch.vertices[12].id, 1);
        assert_eq!(batch.vertices[0].class, 0, "an ordinary edge has no class bits");
        assert!(
            edge_class_hidden(
                batch.vertices[12].class,
                nodecraft_layers::PlanEdgeClass::mask(false),
            ),
            "a seam vertex carries the class that the default mask hides",
        );
        assert!(
            !edge_class_hidden(
                batch.vertices[12].class,
                nodecraft_layers::PlanEdgeClass::mask(true),
            ),
            "the same vertex is drawn and pickable when the mask asks for seams",
        );
        assert!(!edge_class_hidden(
            batch.vertices[0].class,
            nodecraft_layers::PlanEdgeClass::mask(false),
        ));
    }

    #[test]
    fn ghost_edges_ignore_the_ghost_face_opacity() {
        let standard = EdgeUniforms::new(identity_matrix(), identity_matrix(), [100.0, 100.0], 1.0, [0.0; 4], true);
        let ghost = EdgeUniforms::new(identity_matrix(), identity_matrix(), [100.0, 100.0], 2.0, [0.0; 4], false);
        assert_eq!(standard.screen[3], 1.0);
        assert_eq!(ghost.screen[3], 0.0, "ghost edge alpha is the theme alpha, not ghost_opacity");
    }

}
