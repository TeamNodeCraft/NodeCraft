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
const WIDTH: u32 = 2048;
const HEIGHT: u32 = 4096;
pub const BLUR_WIDTH: u32 = WIDTH / 4;
pub const BLUR_HEIGHT: u32 = HEIGHT / 4;
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
  crop: vec4<f32>,
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

fn fetch_source(uv: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(src, smp, params.crop.xy + uv * params.crop.zw, 0.0).rgb;
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

fn fetch_box(uv: vec2<f32>, span: f32) -> vec3<f32> {
  // Average eight source positions across one quarter-resolution texel.
  var sum = vec3<f32>(0.0);
  for (var i = 0; i < 8; i++) {
    let x = (f32(i) + 0.5) / 8.0 - 0.5;
    sum += fetch(uv + vec2(span * x, 0.0));
  }
  return sum / 8.0;
}

fn gaussian_h(in: O) -> vec3<f32> {
  let uv = params.crop.xy + in.uv * params.crop.zw;
  // One quarter-resolution texel is the minimum horizontal kernel: below
  // that the pass point-samples the full-res scene and aliases into stripes.
  let radius = max(params.scale.x, 1.0);
  let sigma = max(radius * 0.5, 0.001);
  let step = params.crop.z / params.scale.z;
  let quarter = step;
  let limit = min(ceil(radius * 3.0), 64.0);
  var sum = vec3<f32>(0.0);
  var weight = 0.0;
  for (var i = 0; i <= 64; i++) {
    if f32(i) > limit { break; }
    let x = f32(i);
    let w = exp(-0.5 * (x / sigma) * (x / sigma));
    let o = vec2<f32>(step * x, 0.0);
    var pair = fetch(uv + o) + fetch(uv - o);
    if (radius < 4.0) {
      pair = fetch_box(uv + o, quarter) + fetch_box(uv - o, quarter);
    }
    sum += pair * w;
    weight += 2.0 * w;
  }
  return sum / max(weight, 0.0001);
}

@fragment
fn gaussian_h_frag(in: O) -> @location(0) vec4<f32> {
  let c = gaussian_h(in);
  return vec4<f32>(decode_srgb(c), 1.0);
}

@fragment
fn gaussian_v_frag(in: O) -> @location(0) vec4<f32> {
  let radius = max(params.scale.x, 1.0);
  let sigma = max(radius * 0.5, 0.001);
  let step = 1.0 / params.scale.w;
  let limit = min(ceil(radius * 3.0), 64.0);
  var sum = vec3<f32>(0.0);
  var weight = 0.0;
  for (var i = 0; i <= 64; i++) {
    if f32(i) > limit { break; }
    let y = f32(i);
    let w = exp(-0.5 * (y / sigma) * (y / sigma));
    let o = vec2<f32>(0.0, step * y);
    sum += (fetch(in.uv + o) + fetch(in.uv - o)) * w;
    weight += 2.0 * w;
  }
  return vec4<f32>(sum / max(weight, 0.0001), 1.0);
}

@fragment
fn gaussian_final(in: O) -> @location(0) vec4<f32> {
  let c = mix(fetch(in.uv), params.tint.rgb, params.tint.a);
  return vec4<f32>(encode_srgb(c * 0.985), 1.0);
}

@fragment
fn down_gamma(in: O) -> @location(0) vec4<f32> {
  let h = 0.5 / vec2<f32>(textureDimensions(src)) * params.scale.xy;
  let uv = params.crop.xy + in.uv * params.crop.zw;
  let c = 0.25 * (
    fetch_source(uv + vec2(-h.x, -h.y)) +
    fetch_source(uv + vec2( h.x, -h.y)) +
    fetch_source(uv + vec2(-h.x,  h.y)) +
    fetch_source(uv + vec2( h.x,  h.y))
  );
  return vec4<f32>(decode_srgb(c), 1.0);
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

@fragment
fn scene_gamma(in: O) -> @location(0) vec4<f32> {
  // Zero blur: copy the sharp scene through the same glass tint without
  // running any down/up passes.
  let c = mix(decode_srgb(fetch_source(in.uv)), params.tint.rgb, params.tint.a);
  return vec4<f32>(encode_srgb(c * 0.985), 1.0);
}

@fragment
fn mix_gamma(in: O) -> @location(0) vec4<f32> {
  // Blend the pyramid result over the direct low-pass base.
  let c = mix(upsample(in.uv), params.tint.rgb, params.tint.a);
  return vec4<f32>(encode_srgb(c * 0.985), params.scale.z);
}

@fragment
fn low_gamma(in: O) -> @location(0) vec4<f32> {
  // Sub-pyramid radii: blur directly from the full-res crop so the
  // half-resolution pyramid cannot shift the backdrop phase.
  let uv = params.crop.xy + in.uv * params.crop.zw;
  let h = params.scale.xy / vec2<f32>(textureDimensions(src));
  var c = fetch(uv) * 4.0;
  c += fetch(uv + vec2(h.x, 0.0)) * 2.0;
  c += fetch(uv - vec2(h.x, 0.0)) * 2.0;
  c += fetch(uv + vec2(0.0, h.y)) * 2.0;
  c += fetch(uv - vec2(0.0, h.y)) * 2.0;
  c += fetch(uv + vec2(h.x, h.y));
  c += fetch(uv + vec2(h.x, -h.y));
  c += fetch(uv + vec2(-h.x, h.y));
  c += fetch(uv + vec2(-h.x, -h.y));
  let tinted = mix(decode_srgb(c / 16.0), params.tint.rgb, params.tint.a);
  return vec4<f32>(encode_srgb(tinted * 0.985), 1.0);
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

#[allow(dead_code)] // GPU resources are retained for their lifetime.
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
    scene_gamma: wgpu::RenderPipeline,
    low_gamma: wgpu::RenderPipeline,
    mix_gamma: wgpu::RenderPipeline,
    gaussian_h: wgpu::RenderPipeline,
    gaussian_v: wgpu::RenderPipeline,
    gaussian_final: wgpu::RenderPipeline,
    blur_a_view: wgpu::TextureView,
    blur_b_view: wgpu::TextureView,
    bind_blur_a: wgpu::BindGroup,
    bind_blur_b: wgpu::BindGroup,
    bind_scene: wgpu::BindGroup,
    bind_down: [wgpu::BindGroup; 5],
    bind_levels: [wgpu::BindGroup; 5],
    params: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
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
        let (_blur_a, blur_a_view) = make_texture(&rs.device, "node-backdrop-blur-a", linear, BLUR_WIDTH, BLUR_HEIGHT);
        let (_blur_b, blur_b_view) = make_texture(&rs.device, "node-backdrop-blur-b", linear, BLUR_WIDTH, BLUR_HEIGHT);
        let (_final_texture, final_view) = make_texture(&rs.device, "node-backdrop-final", display, WIDTH, HEIGHT);

        let target = GpuTarget { format: scene_format, depth: Some(depth), samples: 1, tess: false, oit: false };
        let renderer = Renderer::with_window_target(&rs.device, target, target, None);

        let params = rs.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node-backdrop-params"),
            size: 48,
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
        let bind_blur_a = bind(&blur_a_view);
        let bind_blur_b = bind(&blur_b_view);

        let module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("node-backdrop-dual-kawase"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = rs.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("node-backdrop-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, label: &str, format: wgpu::TextureFormat, blend: bool| {
            rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: if blend { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let down_gamma = pipeline("down_gamma", "node-backdrop-down-gamma", linear, false);
        let down = pipeline("down", "node-backdrop-down", linear, false);
        let up = pipeline("up", "node-backdrop-up", linear, false);
        let up_gamma = pipeline("up_gamma", "node-backdrop-up-gamma", display, false);
        let scene_gamma = pipeline("scene_gamma", "node-backdrop-scene-gamma", display, false);
        let low_gamma = pipeline("low_gamma", "node-backdrop-low-gamma", display, false);
        let mix_gamma = pipeline("mix_gamma", "node-backdrop-mix-gamma", display, true);
        let gaussian_h = pipeline("gaussian_h_frag", "node-backdrop-gaussian-h", linear, false);
        let gaussian_v = pipeline("gaussian_v_frag", "node-backdrop-gaussian-v", linear, false);
        let gaussian_final = pipeline("gaussian_final", "node-backdrop-gaussian-final", display, false);
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
            scene_gamma,
            low_gamma,
            mix_gamma,
            gaussian_h,
            gaussian_v,
            gaussian_final,
            blur_a_view,
            blur_b_view,
            bind_blur_a,
            bind_blur_b,
            bind_scene,
            bind_down,
            bind_levels,
            params,
            bgl,
            sampler,
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

    fn bind_source(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("node-backdrop-external-bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params.as_entire_binding(),
                },
            ],
        })
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        view_proj: [[f32; 4]; 4],
        blur_scale: [f32; 4],
        external_source: Option<(&wgpu::TextureView, [f32; 4])>,
    ) {
                let theme = crate::theme::get();
        // A neutral theme surface used as the frosted film; high coverage keeps
        // the blurred scene subtle rather than reading as a dark see-through panel.
        let glass = theme.blur_tint;
        let tint = [
            srgb_to_linear(glass[0]),
            srgb_to_linear(glass[1]),
            srgb_to_linear(glass[2]),
            f32::from(glass[3]) / 255.0,
        ];
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[
                blur_scale[0],
                blur_scale[1],
                blur_scale[2],
                blur_scale[3],
                tint[0],
                tint[1],
                tint[2],
                tint[3],
                external_source.map_or(0.0, |(_, crop)| crop[0]),
                external_source.map_or(0.0, |(_, crop)| crop[1]),
                external_source.map_or(1.0, |(_, crop)| crop[2]),
                external_source.map_or(1.0, |(_, crop)| crop[3]),
            ]),
        );
        let backdrop = Frame {
            product: false,
            preview_bounds: None,
            moving: None,
            product_lights: None,
            upload_error: None,
            clouds: Vec::new(),
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
            ao_samples: solidcraft_gpu::view::AO_SAMPLES,
            ssaa: 1.0,
            hover_tint: [0.0; 3],
            selection_tint: [0.0; 3],
            hatch: Default::default(),
            sky: [0.0, 1.0, 0.0, 0.0],
        };
        let external_bind = external_source.map(|(view, _)| self.bind_source(device, view));
        if external_source.is_none() {
            self.renderer.prepare(device, queue, encoder, &backdrop);
        }
        if external_source.is_none() {
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

        let bind_scene = external_bind.as_ref().unwrap_or(&self.bind_scene);
        if blur_scale[1] <= f32::EPSILON {
            blur_pass(
                encoder,
                &self.final_view,
                &self.scene_gamma,
                bind_scene,
                "node-backdrop-sharp",
            );
            return;
        }

        // Apple-style backdrop: quarter-resolution separable Gaussian,
        // normalized edges, then a tinted upscale. No multi-level
        // reconstruction, so blur cannot accumulate apparent scale.
        blur_pass(encoder, &self.blur_a_view, &self.gaussian_h, bind_scene, "node-backdrop-gaussian-h");
        blur_pass(encoder, &self.blur_b_view, &self.gaussian_v, &self.bind_blur_a, "node-backdrop-gaussian-v");
        blur_pass(encoder, &self.final_view, &self.gaussian_final, &self.bind_blur_b, "node-backdrop-gaussian-final");
        let _ = (&self._scene, &self._final_texture, &self.levels, &self.blur_a_view, &self.blur_b_view);        let _ = (&self._scene, &self._final_texture);
    }
}
