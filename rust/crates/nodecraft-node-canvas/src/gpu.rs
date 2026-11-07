//! Lyon-tessellated node geometry plus glyphon text, retained in wgpu buffers.
//!
//! Graph-space card, port and wire geometry is tessellated only when the graph or
//! selection changes. Navigation updates a small transform uniform; it does not
//! re-upload vertex data. Glyphon owns the text atlas and shapes labels.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};


use egui_wgpu::{wgpu, CallbackTrait, RenderState};
use glyphon::{
    Attrs, Buffer as TextBuffer, Cache, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use lyon::math::point;
use lyon::path::{Path, Winding};
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
};

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Default)]
pub struct NodeGeometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// One cable instance: up to four routed cubic segments in graph space, packed
/// as thirteen shared points (one vertex attribute pair per point), colours and
/// graph-space outline width. The vertex shader only evaluates the supplied
/// cubics: routing and hit testing share `crate::routing`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CableVertex {
    /// Thirteen `vec4<f32>` vertex attributes hold the twenty-five points of
    /// the eight straight/arc segments.
    pub points: [[f32; 4]; 13],
    pub core: [f32; 4],
    pub outline: [f32; 4],
    /// `x`: outline width, `y`: core ratio, `z`: segment count.
    pub params: [f32; 4],
}

impl CableVertex {
    pub fn new(
        routing: &crate::routing::WireRouting,
        core: [f32; 4],
        outline: [f32; 4],
        width: f32,
        core_ratio: f32,
    ) -> Self {
        // Segments share their join points, so segment `i` starts at point
        // `3 * i`. Twenty-five points cover eight segments.
        // 25 points = 50 floats; two padding slots complete thirteen vec4s.
        let mut flat = [0.0_f32; 52];
        for (index, cubic) in routing.segments.iter().take(routing.count).enumerate() {
            let base = index * 3;
            for (offset, point) in [cubic.p0, cubic.p1, cubic.p2, cubic.p3].iter().enumerate() {
                let at = (base + offset) * 2;
                flat[at] = point.x;
                flat[at + 1] = point.y;
            }
        }
        let mut points = [[0.0_f32; 4]; 13];
        for (index, packed) in points.iter_mut().enumerate() {
            for (lane, value) in packed.iter_mut().enumerate() {
                *value = flat[index * 4 + lane];
            }
        }
        Self {
            points,
            core,
            outline,
            params: [width, core_ratio, routing.count as f32, 0.0],
        }
    }
}

/// One SDF card/port/button instance in graph space.
///
/// Cards use `kind == 0` (rounded box); ports and buttons use `kind == 1`
/// (circle). The shader expands a shared quad and evaluates the signed-distance
/// field in the fragment stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShapeInstance {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub fill: [f32; 4],
    pub stroke: [f32; 4],
    pub radius: f32,
    pub stroke_width: f32,
    pub kind: f32,
    pub _pad: f32,
}

#[derive(Clone)]
pub struct NodeText {
    pub text: String,
    pub center_x: f32,
    pub top: f32,
    pub size: f32,
    pub color: [u8; 4],
    pub bounds: TextBounds,
}

pub struct NodeDraw {
    pub geometry_key: u64,
    pub geometry: Option<Arc<NodeGeometry>>,
    pub shapes: Option<Arc<Vec<ShapeInstance>>>,
    pub glows: Vec<ShapeInstance>,
    pub cables: Option<Arc<Vec<CableVertex>>>,
    /// Vertex count issued per cable instance. The cable shader can evaluate up
    /// to `CABLE_MAX_PATH_SEGMENTS`, but shorter routes should not pay for
    /// vertices they never use.
    pub cable_vertex_budget: u32,
    pub transform: [f32; 4], // scale, offset x, offset y, unused
    pub clip: [f32; 4],
    pub clip_radius: f32,
    pub ppp: f32,
    pub curve_resolution: f32,
    pub glow: [f32; 4],
    pub texts: Vec<NodeText>,
}

const SHADER: &str = r#"
struct Globals {
  screen_scale: vec4<f32>,
  offset: vec4<f32>,
  clip_rect: vec4<f32>,
  clip_params: vec4<f32>,
  glow: vec4<f32>,
};

fn inside_rounded(p: vec2<f32>) -> bool {
  let r = max(globals.clip_params.x, 0.001);
  let inner_min = globals.clip_rect.xy + vec2<f32>(r);
  let inner_max = globals.clip_rect.zw - vec2<f32>(r);
  let nearest = clamp(p, inner_min, inner_max);
  let q = abs(p - nearest) / r;
  // Match the panel's squircle: |x|^4 + |y|^4 <= 1.
  let q2 = q * q;
  return dot(q2, q2) <= 1.0;
}
@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexIn {
  @location(0) position: vec2<f32>,
  @location(1) color: vec4<f32>,
};
struct VertexOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) screen: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs(in: VertexIn) -> VertexOut {
  // Keep geometry on subpixel coordinates so MSAA can antialias card edges,
  // wire curves and ports. The pan itself is still pixel-quantized, which is
  // enough to keep labels stable without snapping every vertex.
  let p = in.position * globals.screen_scale.z + globals.offset.xy;
  var out: VertexOut;
  out.clip = vec4(
    p.x / globals.screen_scale.x * 2.0 - 1.0,
    1.0 - p.y / globals.screen_scale.y * 2.0,
    0.0,
    1.0
  );
  out.screen = p;
  out.color = in.color;
  return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
  if (!inside_rounded(in.screen)) { discard; }
  return in.color;
}
"#;

const CABLE_SHADER: &str = r#"
struct Globals {
  screen_scale: vec4<f32>,
  offset: vec4<f32>,
  clip_rect: vec4<f32>,
  clip_params: vec4<f32>,
  glow: vec4<f32>,
};

fn inside_rounded(p: vec2<f32>) -> bool {
  let r = max(globals.clip_params.x, 0.001);
  let inner_min = globals.clip_rect.xy + vec2<f32>(r);
  let inner_max = globals.clip_rect.zw - vec2<f32>(r);
  let nearest = clamp(p, inner_min, inner_max);
  let q = abs(p - nearest) / r;
  // Match the panel's squircle: |x|^4 + |y|^4 <= 1.
  let q2 = q * q;
  return dot(q2, q2) <= 1.0;
}
@group(0) @binding(0) var<uniform> globals: Globals;

struct Instance {
  @location(0) v0: vec4<f32>,
  @location(1) v1: vec4<f32>,
  @location(2) v2: vec4<f32>,
  @location(3) v3: vec4<f32>,
  @location(4) v4: vec4<f32>,
  @location(5) v5: vec4<f32>,
  @location(6) v6: vec4<f32>,
  @location(7) v7: vec4<f32>,
  @location(8) v8: vec4<f32>,
  @location(9) v9: vec4<f32>,
  @location(10) v10: vec4<f32>,
  @location(11) v11: vec4<f32>,
  @location(12) v12: vec4<f32>,
  @location(13) core: vec4<f32>,
  @location(14) outline: vec4<f32>,
  // x: width, y: core ratio, z: segment count.
  @location(15) params: vec4<f32>,
};
struct Out {
  @builtin(position) clip: vec4<f32>,
  @location(0) screen: vec2<f32>,
  @location(1) core: vec4<f32>,
  @location(2) outline: vec4<f32>,
  @location(3) side: f32,
  @location(4) core_ratio: f32,
};

const BASE_SEGMENTS: f32 = 12.0;
const MAX_SEGMENTS: f32 = 48.0;
const MAX_PATH_SEGMENTS: f32 = 192.0;
const ANGULAR_RESOLUTION_FACTOR: f32 = 2.0;

fn packed(inst: Instance, index: u32) -> vec4<f32> {
  var values = array<vec4<f32>, 13>(
    inst.v0, inst.v1, inst.v2, inst.v3, inst.v4, inst.v5, inst.v6,
    inst.v7, inst.v8, inst.v9, inst.v10, inst.v11, inst.v12
  );
  return values[index];
}

fn point_at(inst: Instance, index: u32) -> vec2<f32> {
  let value = packed(inst, index / 2u);
  if (index % 2u == 0u) {
    return value.xy;
  }
  return value.zw;
}

fn bezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
  let u = 1.0 - t;
  return p0 * (u*u*u) + p1 * (3.0*u*u*t) + p2 * (3.0*u*t*t) + p3 * (t*t*t);
}

fn tangent(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
  let u = 1.0 - t;
  return (p1 - p0) * (3.0*u*u) + (p2 - p1) * (6.0*u*t) + (p3 - p2) * (3.0*t*t);
}

fn segment_index(inst: Instance, t: f32) -> u32 {
  let count = u32(inst.params.z);
  let scaled = t * f32(count);
  return min(u32(floor(scaled)), count - 1u);
}

fn segment_local(inst: Instance, t: f32) -> f32 {
  let count = u32(inst.params.z);
  let scaled = t * f32(count);
  let index = min(u32(floor(scaled)), count - 1u);
  // Use the clamped index: at t == 1 the unclamped floor would be `count`, which
  // made the final ribbon vertex snap back to the segment start (a stray
  // vertex at the cable end).
  return scaled - f32(index);
}

// Evaluate the routed path. Normal cables are one cubic; upward wraps are four
// joined cubics (straight runs and rounded corners).
fn path_point(inst: Instance, t: f32) -> vec2<f32> {
  let segment = segment_index(inst, t);
  let local = segment_local(inst, t);
  let base = segment * 3u;
  return bezier(
    point_at(inst, base),
    point_at(inst, base + 1u),
    point_at(inst, base + 2u),
    point_at(inst, base + 3u),
    local
  );
}

fn path_tangent(inst: Instance, t: f32) -> vec2<f32> {
  let segment = segment_index(inst, t);
  let local = segment_local(inst, t);
  let base = segment * 3u;
  return tangent(
    point_at(inst, base),
    point_at(inst, base + 1u),
    point_at(inst, base + 2u),
    point_at(inst, base + 3u),
    local
  );
}

// A degenerate straight/arc segment (zero length, which rounding makes possible
// when a turn needs almost no run) has a zero analytic tangent. Derive the
// direction from the neighbouring segments instead of falling back to a fixed
// axis, otherwise the ribbon pinches into a sharp notch at that point.
fn path_normal_tangent(inst: Instance, t: f32) -> vec2<f32> {
  let analytic = path_tangent(inst, t);
  if (length(analytic) > 1.0e-5) {
    return analytic;
  }
  let step = 0.5 / max(inst.params.z, 1.0);
  let before = path_point(inst, max(t - step, 0.0));
  let after = path_point(inst, min(t + step, 1.0));
  let derived = after - before;
  if (length(derived) > 1.0e-5) {
    return derived;
  }
  return vec2<f32>(0.0, 1.0);
}

// Cables that arrive almost perpendicular to their chord need more samples
// than the straight/vertical default. Compute this per instance so only sharp
// bends pay for the extra vertices. Wrapped cables get the same budget per
// cubic, multiplied by the segment count.
fn segment_count(inst: Instance) -> f32 {
  let start = point_at(inst, 0u);
  let last = u32(inst.params.z) * 3u;
  let end = point_at(inst, last);
  let end_control = point_at(inst, last - 1u);
  let chord = normalize(end - start + vec2<f32>(0.00001, 0.0));
  let start_tangent = normalize(point_at(inst, 1u) - start + vec2<f32>(0.00001, 0.0));
  let end_tangent = normalize(end - end_control + vec2<f32>(0.00001, 0.0));
  let start_angle = acos(clamp(dot(chord, start_tangent), -1.0, 1.0));
  let end_angle = acos(clamp(dot(chord, end_tangent), -1.0, 1.0));
  let angle = max(start_angle, end_angle);
  // Continuous angular factor: a small bend gets a small increase, 45 degrees
  // gets a moderate increase, and stronger bends keep increasing. There is no
  // single threshold where resolution suddenly jumps.
  let angular_factor = 1.0 + (angle / 3.14159265) * ANGULAR_RESOLUTION_FACTOR;

  // The chord length is a cheap screen-space upper bound on the arc length.
  // At far zoom the curve is small, so fewer segments are enough. At close
  // zoom spend more of the budget on smoothness and multiply by the angular
  // factor.
  let screen_chord = length(end - start) * globals.screen_scale.z;
  let length_budget = max(BASE_SEGMENTS, screen_chord / 12.0 * globals.screen_scale.w);
  let per_segment = round(clamp(length_budget * angular_factor, BASE_SEGMENTS, MAX_SEGMENTS));
  return min(per_segment * inst.params.z, MAX_PATH_SEGMENTS);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, inst: Instance) -> Out {
  let seg = f32(vi / 6u);
  let corner = vi % 6u;
  let at_end = corner == 2u || corner == 4u || corner == 5u;
  let segments = segment_count(inst);
  let is_active = seg < segments;
  var t = (seg + select(0.0, 1.0, at_end)) / segments;
  if (!is_active) {
    t = 1.0;
  }
  let p = path_point(inst, t);
  let d = path_normal_tangent(inst, t);
  var n = vec2<f32>(-d.y, d.x);
  if (length(n) <= 0.000001) { n = vec2<f32>(0.0, 1.0); }
  n = normalize(n);
  var side = -1.0;
  if (corner == 1u || corner == 2u || corner == 4u) { side = 1.0; }
  var base_screen = p * globals.screen_scale.z + globals.offset.xy;
  if (!is_active) {
    base_screen = vec2<f32>(1.0e9, 1.0e9);
  }
  // Keep cables from collapsing below one physical pixel when zoomed out.
  // MSAA smooths the edge, but a sub-pixel ribbon cannot be reconstructed by
  // multisampling alone.
  let screen_width = max(inst.params.x * globals.screen_scale.z, 1.5);
  let half_width = screen_width * 0.5;
  let screen = base_screen + n * half_width * side;
  var out: Out;
  out.clip = vec4<f32>(
    screen.x / globals.screen_scale.x * 2.0 - 1.0,
    1.0 - screen.y / globals.screen_scale.y * 2.0,
    0.0,
    1.0
  );
  out.screen = screen;
  out.core = inst.core;
  out.outline = inst.outline;
  out.side = side;
  out.core_ratio = inst.params.y;
  return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  if (!inside_rounded(in.screen)) { discard; }
  let d = abs(in.side);
  // `side` is a normalized ribbon coordinate. Using a fixed fraction of it
  // makes the AA band scale with cable width, so a wide zoomed-in cable looks
  // soft. fwidth gives one pixel in the same coordinate space.
  let aa = max(fwidth(d), 0.0001);
  let outer = 1.0 - smoothstep(1.0 - aa, 1.0, d);
  let core_mix = smoothstep(in.core_ratio - aa, in.core_ratio + aa, d);
  let colour = mix(in.core, in.outline, core_mix);
  return vec4<f32>(colour.rgb, colour.a * outer);
}
"#;


const SHAPE_SHADER: &str = r#"
struct Globals {
  screen_scale: vec4<f32>,
  offset: vec4<f32>,
  clip_rect: vec4<f32>,
  clip_params: vec4<f32>,
  glow: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals: Globals;

fn inside_rounded(p: vec2<f32>) -> bool {
  let r = max(globals.clip_params.x, 0.001);
  let inner_min = globals.clip_rect.xy + vec2<f32>(r);
  let inner_max = globals.clip_rect.zw - vec2<f32>(r);
  let nearest = clamp(p, inner_min, inner_max);
  let q = abs(p - nearest) / r;
  // Match the panel's squircle: |x|^4 + |y|^4 <= 1.
  let q2 = q * q;
  return dot(q2, q2) <= 1.0;
}

struct Instance {
  @location(0) center: vec2<f32>,
  @location(1) size: vec2<f32>,
  @location(2) fill: vec4<f32>,
  @location(3) stroke: vec4<f32>,
  @location(4) radius: f32,
  @location(5) stroke_width: f32,
  @location(6) kind: f32,
  @location(7) suppressed: f32,
};

struct Out {
  @builtin(position) clip: vec4<f32>,
  @location(0) screen: vec2<f32>,
  @location(1) local: vec2<f32>,
  @location(2) half_size: vec2<f32>,
  @location(3) fill: vec4<f32>,
  @location(4) stroke: vec4<f32>,
  @location(5) radius: f32,
  @location(6) stroke_width: f32,
  @location(7) kind: f32,
  @location(8) suppressed: f32,
};

fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
  let q = abs(p) - b + vec2<f32>(r);
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32, inst: Instance) -> Out {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
  );
  let half_size = inst.size * 0.5;
  // Cover the antialias fringe and the stroke's outer half. Stroke width is
  // in graph units, so convert its half-width to screen pixels before adding
  // the constant AA fringe, then convert the total back to graph units.
  let scale = max(globals.screen_scale.z, 0.0001);
  // Glow falls off over its full radius, not half of a stroke; otherwise the
  // quad ends while alpha is still non-zero and the glow has a square edge.
  let stroke_pad = select(inst.stroke_width * 0.5, inst.stroke_width, inst.kind > 1.5);
  let pad = 2.5 / scale + stroke_pad;
  let local = half_size + vec2<f32>(pad);
  let graph = inst.center + corners[vi] * local;
  let screen = graph * globals.screen_scale.z + globals.offset.xy;
  var out: Out;
  out.clip = vec4<f32>(
    screen.x / globals.screen_scale.x * 2.0 - 1.0,
    1.0 - screen.y / globals.screen_scale.y * 2.0,
    0.0,
    1.0
  );
  out.screen = screen;
  out.local = corners[vi] * local;
  out.half_size = half_size;
  out.fill = inst.fill;
  out.stroke = inst.stroke;
  out.radius = inst.radius;
  out.stroke_width = inst.stroke_width;
  out.kind = inst.kind;
  out.suppressed = inst.suppressed;
  return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  if (!inside_rounded(in.screen)) { discard; }
  // kind == 2: analytic glow behind a card. The instance is the card rect
  // expanded by the glow radius; stroke_width carries that radius.
  if (in.kind > 1.5) {
    let glow = max(in.stroke_width, 0.001);
    let inner_half = max(in.half_size - vec2<f32>(glow), vec2<f32>(0.5));
    let inner_radius = max(in.radius - glow, 0.0);
    let d = sd_round_box(in.local, inner_half, inner_radius);
    let t = clamp(1.0 - d / glow, 0.0, 1.0);
    // kind 2 is the selection pulse and follows the global envelope.
    // kind 3 is per-node activity; its strength is in stroke.a.
    let intensity = select(globals.glow.x, in.stroke.a, in.kind > 2.5);
    return vec4<f32>(in.fill.rgb, in.fill.a * t * t * intensity);
  }
  var distance: f32;
  if (in.kind > 0.5) {
    distance = length(in.local) - in.half_size.x;
  } else {
    distance = sd_round_box(in.local, in.half_size, in.radius);
  }
  let aa = max(fwidth(distance), 0.0001);
  let fill_alpha = 1.0 - smoothstep(-aa, aa, distance);
  var stroke_alpha = 1.0 - smoothstep(-aa, aa, abs(distance) - in.stroke_width * 0.5);
  if (in.stroke_width <= 0.0) {
    stroke_alpha = 0.0;
  }
  let stroke_mix = clamp(stroke_alpha, 0.0, 1.0);
  var alpha = max(fill_alpha * in.fill.a, stroke_alpha * in.stroke.a);
  var colour = mix(in.fill.rgb, in.stroke.rgb, stroke_mix);
  // Suppressed cards carry a 45-degree hatch overlay. It lives in the card
  // shader so it follows the card transform without extra geometry.
  if (in.kind < 0.5 && in.suppressed > 0.5) {
    // Use card-local units, not screen pixels, so the hatch scales with the
    // node like its border and text. One 36-unit period is split into an equal
    // 18-unit stripe and 18-unit gap. The hard square wave keeps the edges
    // crisp instead of feathering them with a smoothstep.
    let diagonal = (in.local.x + in.local.y) / 36.0;
    let hatch = select(0.0, 1.0, fract(diagonal) < 0.5);
    colour = mix(colour, in.stroke.rgb, hatch * 0.9);
    alpha = max(alpha, hatch * 0.7 * fill_alpha);
  }
  return vec4<f32>(colour, alpha);
}
"#;

struct NodeRenderer {
    pipeline: wgpu::RenderPipeline,
    shape_pipeline: wgpu::RenderPipeline,
    cable_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    vertex_capacity: u64,
    index_capacity: u64,
    index_count: u32,
    cable_buffer: Option<wgpu::Buffer>,
    cable_capacity: u64,
    cable_count: u32,
    cable_vertex_budget: u32,
    cable_key: Option<u64>,
    shape_buffer: Option<wgpu::Buffer>,
    shape_capacity: u64,
    shape_count: u32,
    shape_key: Option<u64>,
    glow_buffer: Option<wgpu::Buffer>,
    glow_capacity: u64,
    glow_count: u32,
    geometry_key: Option<u64>,
}

const CABLE_MAX_SEGMENTS: u32 = 48;
/// Longest total path budget (see `MAX_PATH_SEGMENTS` in the shader).
const CABLE_MAX_PATH_SEGMENTS: u32 = 192;

struct GlyphonState {
    font_system: FontSystem,
    swash_cache: SwashCache,
    _cache: Cache,
    viewport: Viewport,
    atlas: TextAtlas,
    renderer: TextRenderer,
    /// Shaped label buffers retained across frames; positions are applied per frame
    /// through TextArea, so pan/zoom does not reshape text.
    buffers: HashMap<(String, u32), TextBuffer>,
}

fn depth_state(format: wgpu::TextureFormat) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn load_font_system() -> FontSystem {
    let mut db = glyphon::cosmic_text::fontdb::Database::new();
    db.load_font_data(include_bytes!("../assets/InterVariable.ttf").to_vec());
    db.set_sans_serif_family("Inter");
    FontSystem::new_with_locale_and_db("en-US".to_string(), db)
}

pub fn install(rs: &RenderState, msaa_samples: u32, depth: wgpu::TextureFormat) -> bool {

    let uniform = rs.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("node-lyon-globals"),
        size: 80,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_layout = rs.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("node-lyon-bgl"),
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
    let bind = rs.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("node-lyon-bg"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
    });
    let module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("node-lyon"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = rs.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("node-lyon-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("node-lyon"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: rs.target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_state(depth)),
        multisample: wgpu::MultisampleState { count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    });

    let shape_module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("node-shape"),
        source: wgpu::ShaderSource::Wgsl(SHAPE_SHADER.into()),
    });
    let shape_pipeline = rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("node-shape"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shape_module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<ShapeInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x2,
                    1 => Float32x2,
                    2 => Float32x4,
                    3 => Float32x4,
                    4 => Float32,
                    5 => Float32,
                    6 => Float32,
                    7 => Float32
                ],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shape_module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: rs.target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_state(depth)),
        multisample: wgpu::MultisampleState { count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    });

    let cable_module = rs.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("node-cable"),
        source: wgpu::ShaderSource::Wgsl(CABLE_SHADER.into()),
    });
    let cable_pipeline = rs.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("node-cable"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &cable_module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CableVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x4,
                    1 => Float32x4,
                    2 => Float32x4,
                    3 => Float32x4,
                    4 => Float32x4,
                    5 => Float32x4,
                    6 => Float32x4,
                    7 => Float32x4,
                    8 => Float32x4,
                    9 => Float32x4,
                    10 => Float32x4,
                    11 => Float32x4,
                    12 => Float32x4,
                    13 => Float32x4,
                    14 => Float32x4,
                    15 => Float32x4
                ],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &cable_module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: rs.target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_state(depth)),
        multisample: wgpu::MultisampleState { count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    });

    let cache = Cache::new(&rs.device);
    let mut atlas = TextAtlas::new(&rs.device, &rs.queue, &cache, rs.target_format);
    let text_renderer = TextRenderer::new(
        &mut atlas,
        &rs.device,
        wgpu::MultisampleState { count: msaa_samples, mask: !0, alpha_to_coverage_enabled: false },
        Some(depth_state(depth)),
    );
    let viewport = Viewport::new(&rs.device, &cache);
    let font_system = load_font_system();
    let swash_cache = SwashCache::new();

    {
        let mut renderer = rs.renderer.write();
        renderer.callback_resources.insert(Arc::new(Mutex::new(NodeRenderer {
            pipeline,
            cable_pipeline,
            uniform,
            bind,
            shape_pipeline,
            vertex_buffer: None,
            index_buffer: None,
            vertex_capacity: 0,
            index_capacity: 0,
            index_count: 0,
            cable_buffer: None,
            cable_capacity: 0,
            cable_count: 0,
            cable_vertex_budget: 6,
            cable_key: None,
            shape_buffer: None,
            shape_capacity: 0,
            shape_count: 0,
            shape_key: None,
            glow_buffer: None,
            glow_capacity: 0,
            glow_count: 0,
            geometry_key: None,
        })));
        renderer.callback_resources.insert(Arc::new(Mutex::new(GlyphonState {
            font_system,
            swash_cache,
            _cache: cache,
            viewport,
            atlas,
            renderer: text_renderer,
            buffers: HashMap::new(),
        })));
    }
    true
}

pub struct NodeCallback(pub NodeDraw);

impl CallbackTrait for NodeCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let draw = &self.0;
        let Some(renderer) = resources.get::<Arc<Mutex<NodeRenderer>>>() else { return Vec::new() };
        if let Ok(mut renderer) = renderer.lock() {
            // Two vec4s avoid any vec2/struct padding ambiguity in WGSL uniforms.
            let globals = [
                screen.size_in_pixels[0].max(1) as f32,
                screen.size_in_pixels[1].max(1) as f32,
                draw.transform[0] * draw.ppp,
                draw.curve_resolution,
                draw.transform[1] * draw.ppp,
                draw.transform[2] * draw.ppp,
                0.0,
                0.0,
                draw.clip[0] * draw.ppp,
                draw.clip[1] * draw.ppp,
                draw.clip[2] * draw.ppp,
                draw.clip[3] * draw.ppp,
                draw.clip_radius * draw.ppp,
                0.0,
                0.0,
                0.0,
                draw.glow[0],
                draw.glow[1],
                draw.glow[2],
                draw.glow[3],
            ];
            queue.write_buffer(&renderer.uniform, 0, &globals.map(f32::to_le_bytes).concat());
            if let Some(geometry) = &draw.geometry {
                if renderer.geometry_key != Some(draw.geometry_key) {
                    let vertex_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            geometry.vertices.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(geometry.vertices.as_slice()),
                        )
                    };
                    let index_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            geometry.indices.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(geometry.indices.as_slice()),
                        )
                    };
                    if !vertex_bytes.is_empty() && !index_bytes.is_empty() {
                        // Keep buffers alive across edits/drags. Grow only when the
                        // current capacity is insufficient; otherwise update in place.
                        if renderer.vertex_buffer.is_none()
                            || renderer.vertex_capacity < vertex_bytes.len() as u64
                        {
                            let capacity = (vertex_bytes.len() as u64).next_power_of_two().max(4096);
                            renderer.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("node-lyon-vertices"),
                                size: capacity,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }));
                            renderer.vertex_capacity = capacity;
                        }
                        queue.write_buffer(renderer.vertex_buffer.as_ref().unwrap(), 0, vertex_bytes);

                        if renderer.index_buffer.is_none()
                            || renderer.index_capacity < index_bytes.len() as u64
                        {
                            let capacity = (index_bytes.len() as u64).next_power_of_two().max(4096);
                            renderer.index_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("node-lyon-indices"),
                                size: capacity,
                                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }));
                            renderer.index_capacity = capacity;
                        }
                        queue.write_buffer(renderer.index_buffer.as_ref().unwrap(), 0, index_bytes);
                        renderer.index_count = geometry.indices.len() as u32;
                    } else {
                        renderer.index_count = 0;
                    }
                    renderer.geometry_key = Some(draw.geometry_key);
                }
            }
            let glow_bytes = bytemuck::cast_slice(draw.glows.as_slice());
            if !glow_bytes.is_empty() {
                if renderer.glow_buffer.is_none()
                    || renderer.glow_capacity < glow_bytes.len() as u64
                {
                    let capacity = (glow_bytes.len() as u64).next_power_of_two().max(4096);
                    renderer.glow_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("node-activity-glows"),
                        size: capacity,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                    renderer.glow_capacity = capacity;
                }
                queue.write_buffer(renderer.glow_buffer.as_ref().unwrap(), 0, glow_bytes);
                renderer.glow_count = draw.glows.len() as u32;
            } else {
                renderer.glow_count = 0;
            }
            renderer.cable_vertex_budget = draw.cable_vertex_budget.max(6);
            if let Some(cables) = &draw.cables {
                if renderer.cable_key != Some(draw.geometry_key) {
                    let bytes = bytemuck::cast_slice(cables.as_slice());
                    if !bytes.is_empty() {
                        if renderer.cable_buffer.is_none()
                            || renderer.cable_capacity < bytes.len() as u64
                        {
                            let capacity = (bytes.len() as u64).next_power_of_two().max(4096);
                            renderer.cable_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("node-cable-instances"),
                                size: capacity,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }));
                            renderer.cable_capacity = capacity;
                        }
                        queue.write_buffer(renderer.cable_buffer.as_ref().unwrap(), 0, bytes);
                        renderer.cable_count = cables.len() as u32;
                    } else {
                        renderer.cable_count = 0;
                    }
                    renderer.cable_key = Some(draw.geometry_key);
                }
            }

            if let Some(shapes) = &draw.shapes {
                if renderer.shape_key != Some(draw.geometry_key) {
                    let bytes = bytemuck::cast_slice(shapes.as_slice());
                    if !bytes.is_empty() {
                        if renderer.shape_buffer.is_none()
                            || renderer.shape_capacity < bytes.len() as u64
                        {
                            let capacity = (bytes.len() as u64).next_power_of_two().max(4096);
                            renderer.shape_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("node-shape-instances"),
                                size: capacity,
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            }));
                            renderer.shape_capacity = capacity;
                        }
                        queue.write_buffer(renderer.shape_buffer.as_ref().unwrap(), 0, bytes);
                        renderer.shape_count = shapes.len() as u32;
                    } else {
                        renderer.shape_count = 0;
                    }
                    renderer.shape_key = Some(draw.geometry_key);
                }
            }
        }

        let Some(glyphon) = resources.get::<Arc<Mutex<GlyphonState>>>() else { return Vec::new() };
        if let Ok(mut glyphon) = glyphon.lock() {
            glyphon.viewport.update(
                queue,
                Resolution {
                    width: screen.size_in_pixels[0].max(1),
                    height: screen.size_in_pixels[1].max(1),
                },
            );
            let GlyphonState {
                font_system,
                swash_cache,
                viewport,
                atlas,
                renderer,
                buffers,
                ..
            } = &mut *glyphon;
            if buffers.len() > 4096 {
                buffers.clear();
            }
            for text in &draw.texts {
                let size = ((text.size * draw.ppp) * 4.0).round() / 4.0;
                let key = (text.text.clone(), size.to_bits());
                if !buffers.contains_key(&key) {
                    let mut buffer = TextBuffer::new(
                        font_system,
                        Metrics::new(size, size * 1.25),
                    );
                    buffer.set_size(None, None);
                    buffer.set_text(
                        &text.text,
                        &Attrs::new().family(Family::SansSerif),
                        Shaping::Advanced,
                        None,
                    );
                    buffer.shape_until_scroll(font_system, false);
                    buffers.insert(key, buffer);
                }
            }
            let areas: Vec<TextArea<'_>> = draw
                .texts
                .iter()
                .filter_map(|text| {
                    let size = ((text.size * draw.ppp) * 4.0).round() / 4.0;
                    let key = (text.text.clone(), size.to_bits());
                    let buffer = buffers.get(&key)?;
                    let width = buffer
                        .layout_runs()
                        .map(|run| run.line_w)
                        .fold(0.0_f32, f32::max);
                    Some(TextArea {
                        buffer,
                        left: (text.center_x * draw.ppp - width * 0.5).round(),
                        top: (text.top * draw.ppp).round(),
                        scale: 1.0,
                        bounds: text.bounds,
                        default_color: glyphon::Color::rgba(
                            text.color[0],
                            text.color[1],
                            text.color[2],
                            text.color[3],
                        ),
                        custom_glyphs: &[],
                    })
                })
                .collect();
            let _ = renderer.prepare(
                device,
                queue,
                font_system,
                atlas,
                viewport,
                areas,
                swash_cache,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        // Egui paint callbacks are not scissored automatically; without this
        // the GPU cards/cables escape the floating panel rectangle.
        let clip = info.clip_rect_in_pixels();
        if clip.width_px <= 0 || clip.height_px <= 0 {
            return;
        }
        pass.set_scissor_rect(
            clip.left_px.max(0) as u32,
            clip.top_px.max(0) as u32,
            clip.width_px as u32,
            clip.height_px as u32,
        );
        if let Some(renderer) = resources.get::<Arc<Mutex<NodeRenderer>>>() {
            if let Ok(renderer) = renderer.lock() {
                if let Some(gb) = &renderer.glow_buffer {
                    if renderer.glow_count > 0 {
                        pass.set_pipeline(&renderer.shape_pipeline);
                        pass.set_bind_group(0, &renderer.bind, &[]);
                        pass.set_vertex_buffer(0, gb.slice(..));
                        pass.draw(0..6, 0..renderer.glow_count);
                    }
                }
                if let Some(cb) = &renderer.cable_buffer {
                    if renderer.cable_count > 0 {
                        pass.set_pipeline(&renderer.cable_pipeline);
                        pass.set_bind_group(0, &renderer.bind, &[]);
                        pass.set_vertex_buffer(0, cb.slice(..));
                        pass.draw(0..renderer.cable_vertex_budget, 0..renderer.cable_count);
                    }
                }
                if let Some(sb) = &renderer.shape_buffer {
                    if renderer.shape_count > 0 {
                        pass.set_pipeline(&renderer.shape_pipeline);
                        pass.set_bind_group(0, &renderer.bind, &[]);
                        pass.set_vertex_buffer(0, sb.slice(..));
                        pass.draw(0..6, 0..renderer.shape_count);
                    }
                }
                if let (Some(vb), Some(ib)) = (&renderer.vertex_buffer, &renderer.index_buffer) {
                    pass.set_pipeline(&renderer.pipeline);
                    pass.set_bind_group(0, &renderer.bind, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..renderer.index_count, 0, 0..1);
                }
            }
        }
        if let Some(glyphon) = resources.get::<Arc<Mutex<GlyphonState>>>() {
            if let Ok(glyphon) = glyphon.lock() {
                let _ = glyphon.renderer.render(&glyphon.atlas, &glyphon.viewport, pass);
            }
        }
    }
}

// --- Lyon geometry construction -------------------------------------------------

pub struct GeometryBuilder {
    buffers: VertexBuffers<Vertex, u32>,
    fill: FillTessellator,
    stroke: StrokeTessellator,
}

impl Default for GeometryBuilder {
    fn default() -> Self {
        Self {
            buffers: VertexBuffers::new(),
            fill: FillTessellator::new(),
            stroke: StrokeTessellator::new(),
        }
    }
}

impl GeometryBuilder {
    pub fn fill_path(&mut self, path: &Path, color: [f32; 4]) {
        let mut builder = BuffersBuilder::new(&mut self.buffers, |v: FillVertex<'_>| {
            let p = v.position();
            Vertex { position: [p.x, p.y], color }
        });
        let _ = self.fill.tessellate_path(path, &FillOptions::default(), &mut builder);
    }

    pub fn stroke_path(&mut self, path: &Path, color: [f32; 4], width: f32) {
        let mut builder = BuffersBuilder::new(&mut self.buffers, |v: StrokeVertex<'_, '_>| {
            let p = v.position();
            Vertex { position: [p.x, p.y], color }
        });
        let options = StrokeOptions::default()
            .with_line_width(width)
            .with_line_cap(LineCap::Round)
            .with_line_join(LineJoin::Round);
        let _ = self.stroke.tessellate_path(path, &options, &mut builder);
    }

    pub fn finish(self) -> NodeGeometry {
        NodeGeometry { vertices: self.buffers.vertices, indices: self.buffers.indices }
    }
}

pub fn squircle_rect_path(x: f32, y: f32, w: f32, h: f32, radius: f32) -> Path {
    let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h));
    let points = crate::squircle::points(rect, radius);
    let mut builder = Path::builder();
    let Some(first) = points.first() else { return builder.build() };
    builder.begin(point(first.x, first.y));
    for p in &points[1..] {
        builder.line_to(point(p.x, p.y));
    }
    builder.end(true);
    builder.build()
}


pub fn circle_path(cx: f32, cy: f32, radius: f32) -> Path {
    let mut builder = Path::builder();
    builder.add_circle(point(cx, cy), radius, Winding::Positive);
    builder.build()
}

pub fn wire_path(x0: f32, y0: f32, x1: f32, y1: f32) -> Path {
    let points = crate::routing::route(x0, y0, x1, y1, 1.0).polyline(32);
    let mut builder = Path::builder();
    let Some(first) = points.first() else { return builder.build() };
    builder.begin(point(first.x, first.y));
    for next in &points[1..] {
        builder.line_to(point(next.x, next.y));
    }
    builder.end(false);
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyon_builds_card_and_wire_geometry() {
        let mut builder = GeometryBuilder::default();
        builder.fill_path(&squircle_rect_path(0.0, 0.0, 232.0, 104.0, 26.0), [1.0, 1.0, 1.0, 1.0]);
        builder.stroke_path(&wire_path(10.0, 10.0, 300.0, 300.0), [1.0, 1.0, 1.0, 1.0], 2.4);
        let geometry = builder.finish();
        assert!(!geometry.vertices.is_empty());
        assert!(!geometry.indices.is_empty());
    }
}
