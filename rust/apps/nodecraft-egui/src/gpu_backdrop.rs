//! GPU-rendered frosted backdrop for the node-panel glass.
//!
//! Large radii use the standard dual-Kawase down/up pyramid: a 4-sample
//! downsample chain followed by an 8-sample upsample chain. This is the
//! production-compositor approach used by browser/macOS-style blur.

use std::sync::{Arc, Mutex};

use egui_wgpu::wgpu;
use solidcraft_gpu::view::{Frame, GpuScene, GpuTarget, Renderer, SceneSlot};

// The framebuffer is rendered wider/taller than the displayed panel so the
// blur can sample content just outside the crop instead of clamping at its edge.
const WIDTH: u32 = 512;
const HEIGHT: u32 = 1024;
pub const INNER_UV_MIN: [f32; 2] = [0.0, 0.0];
pub const INNER_UV_MAX: [f32; 2] = [1.0, 1.0];

const SHADER: &str = r#"
struct O {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct Params {
  scale: vec4<f32>,
  tint: vec4<f32>,
};
@group(0) @binding(2) var<uniform> params: Params;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> O {
  var p = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
  var out: O;
  out.pos = vec4(p[i], 0., 1.);
  out.uv = vec2((p[i].x + 1.) * 0.5, 1. - (p[i].y + 1.) * 0.5);
  return out;
}

fn fetch(uv: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(src, smp, uv, 0.0).rgb;
}

fn decode_srgb(c: vec3<f32>) -> vec3<f32> {
  return select(c / 12.92, pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c > vec3<f32>(0.04045));
}

fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
  return select(c * 12.92, 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c > vec3<f32>(0.0031308));
}

fn downsample(uv: vec2<f32>) -> vec3<f32> {
  let h = 0.5 / vec2<f32>(textureDimensions(src)) * params.scale.xy;
  return 0.25 * (
    fetch(uv + vec2(-h.x, -h.y)) +
    fetch(uv + vec2( h.x, -h.y)) +
    fetch(uv + vec2(-h.x,  h.y)) +
    fetch(uv + vec2( h.x,  h.y))
  );
}

fn upsample(uv: vec2<f32>) -> vec3<f32> {
  let h = 1.0 / vec2<f32>(textureDimensions(src)) * params.scale.xy;
  return (
    fetch(uv + vec2(-2.0 * h.x, 0.0)) * 2.0 +
    fetch(uv + vec2( 2.0 * h.x, 0.0)) * 2.0 +
    fetch(uv + vec2(0.0, -2.0 * h.y)) * 2.0 +
    fetch(uv + vec2(0.0,  2.0 * h.y)) * 2.0 +
    fetch(uv + vec2(-h.x, -h.y)) +
    fetch(uv + vec2( h.x, -h.y)) +
    fetch(uv + vec2(-h.x,  h.y)) +
    fetch(uv + vec2( h.x,  h.y))
  ) / 12.0;
}

@fragment
fn down_gamma(in: O) -> @location(0) vec4<f32> {
  return vec4<f32>(decode_srgb(downsample(in.uv)), 1.0);
}

@fragment
fn down(in: O) -> @location(0) vec4<f32> {
  return vec4<f32>(downsample(in.uv), 1.0);
}

@fragment
fn up(in: O) -> @location(0) vec4<f32> {
  return vec4<f32>(upsample(in.uv), 1.0);
}

@fragment
fn up_gamma(in: O) -> @location(0) vec4<f32> {
  // Tint with the theme's dark glass colour, composited in linear light.
  let c = mix(upsample(in.uv), params.tint.rgb, params.tint.a);
  // A barely-there darkening of the finished panel film.
  return vec4<f32>(encode_srgb(c * 0.985), 1.0);
}
"#;

fn srgb_to_linear(v: u8) -> f32 {
    let c = f32::from(v) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

type Pair = (wgpu::Texture, wgpu::TextureView);

pub struct BackdropRenderer {
    renderer: Renderer,
    slot: SceneSlot,
    _scene: wgpu::Texture,
    scene_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    levels: [Pair; 5],
    _final_texture: wgpu::Texture,
    final_view: wgpu::TextureView,
    down_gamma: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    up_gamma: wgpu::RenderPipeline,
    bind_scene: wgpu::BindGroup,
    bind_down: [wgpu::BindGroup; 5],
    bind_levels: [wgpu::BindGroup; 5],
    params: wgpu::Buffer,
    texture_id: egui::TextureId,
}

fn make_texture(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat, width: u32, height: u32) -> Pair {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

impl BackdropRenderer {
    pub fn install(rs: &egui_wgpu::RenderState) -> Arc<Mutex<Self>> {
        // Linear-light scratch for the whole pyramid. The final pass writes an
        // sRGB texture so egui displays it correctly.
        // Prefer the linear HDR path, but fall back to 8-bit textures when the
        // adapter cannot both render and filter Rgba16Float.
        let force_rgba8 = std::env::var_os("NODECRAFT_FORCE_RGBA8").is_some();
        let float_features = rs
            .adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let linear = if !force_rgba8
            && float_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
            && float_features.allowed_usages.contains(
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        {
            wgpu::TextureFormat::Rgba16Float
        } else {
            eprintln!("node backdrop: Rgba16Float unsupported, falling back to Rgba8Unorm");
            wgpu::TextureFormat::Rgba8Unorm
        };
        // Keep the finished buffer at the same precision as the scratch.
        let display = linear;
        let scene_format = wgpu::TextureFormat::Rgba8Unorm;
        let depth = solidcraft_gpu::depth_format(&rs.device, true);
        // egui's window target is Bgra8Unorm: values are gamma-encoded and written
        // raw. Render the scene into the same raw format, decode once into the
        // linear float scratch, and encode only on the final display pass.
        let (scene, scene_view) = make_texture(&rs.device, "node-backdrop-scene", scene_format, WIDTH, HEIGHT);
        let depth_texture = rs.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("node-backdrop-depth"),
            size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&Default::default());
        let levels = [
            make_texture(&rs.device, "node-backdrop-down-1", linear, WIDTH / 2, HEIGHT / 2),
            make_texture(&rs.device, "node-backdrop-down-2", linear, WIDTH / 4, HEIGHT / 4),
            make_texture(&rs.device, "node-backdrop-down-3", linear, WIDTH / 8, HEIGHT / 8),
            make_texture(&rs.device, "node-backdrop-down-4", linear, WIDTH / 16, HEIGHT / 16),
            make_texture(&rs.device, "node-backdrop-down-5", linear, WIDTH / 32, HEIGHT / 32),
        ];
        let (_final_texture, final_view) = make_texture(&rs.device, "node-backdrop-final", display, WIDTH, HEIGHT);

        let target = GpuTarget { format: scene_format, depth: Some(depth), samples: 1, tess: false, oit: false };
        let renderer = Renderer::with_window_target(&rs.device, target, target, None);

        let params = rs.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node-backdrop-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = rs.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("node-backdrop-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bgl = rs.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("node-backdrop-bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind = |view: &wgpu::TextureView| {
            rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("node-backdrop-bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                ],
            })
        };
        let bind_scene = bind(&scene_view);
        let bind_down = [bind(&levels[0].1), bind(&levels[1].1), bind(&levels[2].1), bind(&levels[3].1), bind(&levels[4].1)];
        let bind_levels = bind_down.clone();

        let module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("node-backdrop-dual-kawase"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = rs.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("node-backdrop-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, label: &str, format: wgpu::TextureFormat| {
            rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let down_gamma = pipeline("down_gamma", "node-backdrop-down-gamma", linear);
        let down = pipeline("down", "node-backdrop-down", linear);
        let up = pipeline("up", "node-backdrop-up", linear);
        let up_gamma = pipeline("up_gamma", "node-backdrop-up-gamma", display);
        let texture_id = {
            let mut egui = rs.renderer.write();
            egui.register_native_texture(&rs.device, &final_view, wgpu::FilterMode::Linear)
        };

        Arc::new(Mutex::new(Self {
            renderer,
            slot: Arc::new(Mutex::new(None)),
            _scene: scene,
            scene_view,
            depth_view,
            levels,
            _final_texture,
            final_view,
            down_gamma,
            down,
            up,
            up_gamma,
            bind_scene,
            bind_down,
            bind_levels,
            params,
            texture_id,
        }))
    }

    pub fn set_scene(&mut self, scene: GpuScene) {
                if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(scene);
        }
    }

    pub fn texture_id(&self) -> egui::TextureId {
        self.texture_id
    }

    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, frame: &Frame, view_proj: [[f32; 4]; 4], blur_scale: [f32; 2]) {
                let theme = crate::theme::get();
        // A neutral theme surface used as the frosted film; high coverage keeps
        // the blurred scene subtle rather than reading as a dark see-through panel.
        let glass = theme.panel;
        let tint = [
            srgb_to_linear(glass[0]),
            srgb_to_linear(glass[1]),
            srgb_to_linear(glass[2]),
            0.80,
        ];
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[blur_scale[0], blur_scale[1], 0.0, 0.0, tint[0], tint[1], tint[2], tint[3]]),
        );
        let backdrop = Frame {
            product: false,
            key: frame.key,
            slot: self.slot.clone(),
            sketch_key: 0,
            sketch_slot: Arc::new(Mutex::new(None)),
            hl_key: 0,
            hl_slot: Arc::new(Mutex::new(None)),
            pv_key: 0,
            pv_slot: Arc::new(Mutex::new(None)),
            pv_hide: false,
            view_proj,
            back: frame.back,
            size_px: [WIDTH as f32, HEIGHT as f32],
            clip: [0.0; 4],
            cap: [0.0; 4],
            images: Vec::new(),
            analysis: [0.0; 8],
            analysis_colors: [0.0; 36],
            access: None,
            outline: false,
            eye: frame.eye,
            ao_radius: 0.0,
            ssaa: 1.0,
            hover_tint: [0.0; 3],
            selection_tint: [0.0; 3],
            hatch: Default::default(),
        };
        self.renderer.prepare(device, queue, encoder, &backdrop);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("node-backdrop-scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 28.0 / 255.0,
                            g: 28.0 / 255.0,
                            b: 30.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.renderer.paint(&backdrop, &mut pass, [0.0, 0.0, WIDTH as f32, HEIGHT as f32]);
        }

        let blur_pass = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, pipeline: &wgpu::RenderPipeline, bind: &wgpu::BindGroup, label: &str| {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        };

        // 512x1024 -> 256x512 -> 128x256 -> 64x128 -> 32x64 -> 16x32.
        blur_pass(encoder, &self.levels[0].1, &self.down_gamma, &self.bind_scene, "node-backdrop-down-1");
        blur_pass(encoder, &self.levels[1].1, &self.down, &self.bind_down[0], "node-backdrop-down-2");
        blur_pass(encoder, &self.levels[2].1, &self.down, &self.bind_down[1], "node-backdrop-down-3");
        blur_pass(encoder, &self.levels[3].1, &self.down, &self.bind_down[2], "node-backdrop-down-4");
        blur_pass(encoder, &self.levels[4].1, &self.down, &self.bind_down[3], "node-backdrop-down-5");
        // 16x32 -> 32x64 -> 64x128 -> 128x256 -> 256x512 -> 512x1024.
        blur_pass(encoder, &self.levels[3].1, &self.up, &self.bind_levels[4], "node-backdrop-up-32");
        blur_pass(encoder, &self.levels[2].1, &self.up, &self.bind_levels[3], "node-backdrop-up-64");
        blur_pass(encoder, &self.levels[1].1, &self.up, &self.bind_levels[2], "node-backdrop-up-128");
        blur_pass(encoder, &self.levels[0].1, &self.up, &self.bind_levels[1], "node-backdrop-up-256");
        blur_pass(encoder, &self.final_view, &self.up_gamma, &self.bind_levels[0], "node-backdrop-up-final");
        let _ = (&self._scene, &self._final_texture);
    }
}
