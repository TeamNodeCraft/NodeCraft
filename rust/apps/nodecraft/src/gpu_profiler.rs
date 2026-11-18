//! Optional live GPU/CPU frame profiler.
//!
//! The profiler is off by default. While it is off, every entry point in this
//! module is a single `bool`/`AtomicBool` check followed by a return: no
//! `QuerySet` is created, no timestamp is written, nothing is resolved, copied
//! or mapped, and no ring-buffer slot is touched. The app still owns a cheap
//! [`GpuProfilerSlot`] so the setting can be switched on without recreating the
//! wgpu device; the slot holds no GPU query resources until [`GpuProfilerSlot::enable`].
//!
//! GPU timings come from [`wgpu_profiler`] timer-query scopes placed on render
//! passes. Only the pass-boundary feature (`TIMESTAMP_QUERY`) is requested, so
//! every backend - including Apple GPUs, which cannot sample timestamps inside
//! a pass - produces the same series. Results are read back asynchronously a
//! few frames later; the app never blocks on the GPU.
//!
//! Pass labels are defined once in [`pass`] and shared by the `label` on the
//! wgpu pass descriptor and the profiler scope. Do not duplicate these strings
//! at call sites.
//!
//! Two boundaries are outside this module's control: egui-wgpu owns the final
//! egui pass and exposes no timestamp hook, so its cost is the CPU `UI update`
//! series; plan GPU tessellation runs in the display worker's own device and is
//! reported through the existing `Display plan` CPU counter instead of a render
//! pass.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use egui_wgpu::wgpu;
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Number of frames kept per series. The ring is allocated once.
pub const WINDOW_FRAMES: usize = 600;

/// wgpu-profiler keeps at most three frames in flight. This is only a safety
/// cap so a stalled map cannot grow our frame-id queue without bound.
const MAX_PENDING_GPU_FRAMES: usize = 8;

/// Makes the temporary GPU-trace file unique even if two saves overlap (or two
/// tests run in parallel in one process).
static TRACE_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The labels of every render pass this app owns.
///
/// These are the single source of truth: pass descriptors use these constants
/// and so does the profiler scope, so a pass can never be recorded under a
/// different name than it is encoded with.
pub mod pass {
    /// The offscreen viewport pass. Opaque/translucent/ghost faces, ribbon
    /// edges, GPU plan draws, grid and axes share this pass, so they share one
    /// pass-boundary series on every backend (Apple cannot sample inside it).
    pub const SCENE: &str = "viewport-scene-offscreen";
    pub const PICK: &str = "pick-pass";
    pub const BACKDROP_SCENE: &str = "node-backdrop-scene";
    pub const BACKDROP_SHARP: &str = "node-backdrop-sharp";
    pub const BACKDROP_BLUR_H: &str = "node-backdrop-gaussian-h";
    pub const BACKDROP_BLUR_V: &str = "node-backdrop-gaussian-v";
    pub const BACKDROP_BLUR_FINAL: &str = "node-backdrop-gaussian-final";
}

/// One render pass and the series its GPU time contributes to.
pub struct PassDef {
    pub label: &'static str,
    pub series: Series,
}

/// Every pass that is scoped. Keeping this table next to [`pass`] is what makes
/// the pass set testable without a GPU.
pub const PASSES: [PassDef; 7] = [
    PassDef {
        label: pass::SCENE,
        series: Series::GpuScene,
    },
    PassDef {
        label: pass::PICK,
        series: Series::GpuPick,
    },
    PassDef {
        label: pass::BACKDROP_SCENE,
        series: Series::GpuBackdrop,
    },
    PassDef {
        label: pass::BACKDROP_SHARP,
        series: Series::GpuBlur,
    },
    PassDef {
        label: pass::BACKDROP_BLUR_H,
        series: Series::GpuBlur,
    },
    PassDef {
        label: pass::BACKDROP_BLUR_V,
        series: Series::GpuBlur,
    },
    PassDef {
        label: pass::BACKDROP_BLUR_FINAL,
        series: Series::GpuBlur,
    },
];

/// A graph/table series. The order is the table order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Series {
    CpuFrame = 0,
    UiUpdate = 1,
    EvalSubmit = 2,
    DisplayApply = 3,
    PlanBuild = 4,
    GpuTotal = 5,
    GpuScene = 6,
    GpuPick = 7,
    GpuBackdrop = 8,
    GpuBlur = 9,
}

pub const SERIES_COUNT: usize = 10;
pub const SERIES: [Series; SERIES_COUNT] = [
    Series::CpuFrame,
    Series::UiUpdate,
    Series::EvalSubmit,
    Series::DisplayApply,
    Series::PlanBuild,
    Series::GpuTotal,
    Series::GpuScene,
    Series::GpuPick,
    Series::GpuBackdrop,
    Series::GpuBlur,
];

impl Series {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn key(self) -> &'static str {
        match self {
            Self::CpuFrame => "cpu.frame",
            Self::UiUpdate => "cpu.ui",
            Self::EvalSubmit => "cpu.eval_submit",
            Self::DisplayApply => "cpu.display_apply",
            Self::PlanBuild => "cpu.display_plan",
            Self::GpuTotal => "gpu.total",
            Self::GpuScene => "gpu.scene",
            Self::GpuPick => "gpu.pick",
            Self::GpuBackdrop => "gpu.backdrop",
            Self::GpuBlur => "gpu.blur",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuFrame => "Frame (CPU)",
            Self::UiUpdate => "UI update",
            Self::EvalSubmit => "Evaluate submit",
            Self::DisplayApply => "Display apply",
            Self::PlanBuild => "Display plan",
            Self::GpuTotal => "GPU total",
            Self::GpuScene => "GPU scene",
            Self::GpuPick => "GPU pick",
            Self::GpuBackdrop => "GPU backdrop",
            Self::GpuBlur => "GPU blur",
        }
    }

    pub const fn default_visible(self) -> bool {
        matches!(self, Self::CpuFrame | Self::GpuTotal)
    }

    /// The series a pass label contributes to, or `None` for an unknown label.
    pub fn for_pass_label(label: &str) -> Option<Self> {
        PASSES
            .iter()
            .find(|pass| pass.label == label)
            .map(|pass| pass.series)
    }

    pub fn color(self, theme: &crate::theme::Theme) -> egui::Color32 {
        let rgba = match self {
            Self::CpuFrame => theme.text,
            Self::UiUpdate => theme.highlight,
            Self::EvalSubmit => theme.node_activity,
            Self::DisplayApply => theme.port_input,
            Self::PlanBuild => theme.node_warning,
            Self::GpuTotal => theme.accent,
            Self::GpuScene => theme.axis_x,
            Self::GpuPick => theme.axis_y,
            Self::GpuBackdrop => theme.axis_z,
            Self::GpuBlur => theme.cable,
        };
        theme.color(rgba)
    }
}

/// Current/average/max of one series over the retained window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SeriesStats {
    pub current: f32,
    pub average: f32,
    pub maximum: f32,
    pub samples: usize,
}

/// Query/ring work actually issued. Off-path tests assert these stay zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfilerCounters {
    pub scopes: u64,
    pub timestamp_writes: u64,
    pub resolves: u64,
    pub end_frames: u64,
}

/// One frame in the fixed ring.
#[derive(Clone, Copy)]
struct Sample {
    seq: u64,
    values: [f32; SERIES_COUNT],
    present: u32,
    /// Bytes uploaded since the previous frame, for the trace counter lane.
    upload_bytes: u64,
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            seq: 0,
            values: [0.0; SERIES_COUNT],
            present: 0,
            upload_bytes: 0,
        }
    }
}

impl Sample {
    fn set(&mut self, series: Series, value: f32) {
        let index = series.index();
        self.values[index] = value;
        self.present |= 1 << index;
    }

    fn add(&mut self, series: Series, value: f32) {
        let index = series.index();
        self.values[index] += value;
        self.present |= 1 << index;
    }

    fn get(&self, series: Series) -> Option<f32> {
        (self.present & (1 << series.index()) != 0).then_some(self.values[series.index()])
    }
}

/// Fixed-capacity, allocation-free frame ring.
struct SampleRing {
    samples: Vec<Sample>,
    len: usize,
    next_seq: u64,
}

impl SampleRing {
    fn new() -> Self {
        // Deliberately empty: an off profiler must not allocate its ring.
        Self {
            samples: Vec::new(),
            len: 0,
            next_seq: 0,
        }
    }

    fn ensure_capacity(&mut self) {
        if self.samples.is_empty() {
            self.samples = vec![Sample::default(); WINDOW_FRAMES];
        }
    }

    fn clear(&mut self) {
        self.samples = Vec::new();
        self.len = 0;
        self.next_seq = 0;
    }

    fn begin(&mut self) -> u64 {
        self.ensure_capacity();
        let seq = self.next_seq;
        self.next_seq += 1;
        let index = seq as usize % WINDOW_FRAMES;
        self.samples[index] = Sample {
            seq,
            ..Sample::default()
        };
        self.len = (self.len + 1).min(WINDOW_FRAMES);
        seq
    }

    fn get_mut(&mut self, seq: u64) -> Option<&mut Sample> {
        if !self.contains(seq) {
            return None;
        }
        let index = seq as usize % WINDOW_FRAMES;
        (self.samples[index].seq == seq).then_some(&mut self.samples[index])
    }

    fn contains(&self, seq: u64) -> bool {
        seq < self.next_seq && self.next_seq - seq <= self.len as u64
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Oldest to newest.
    fn iter(&self) -> impl Iterator<Item = &Sample> {
        let start = if self.len == WINDOW_FRAMES {
            self.next_seq as usize % WINDOW_FRAMES
        } else {
            0
        };
        (0..self.len).map(move |offset| &self.samples[(start + offset) % WINDOW_FRAMES])
    }
}

/// CPU-side state plus the fixed ring. This type has no wgpu dependency so the
/// zero-cost and recording rules are tested without a GPU.
struct ProfilerCore {
    enabled: bool,
    timestamps_supported: bool,
    paused: bool,
    ring: SampleRing,
    current_seq: Option<u64>,
    pending_gpu: VecDeque<u64>,
    frame_has_scopes: bool,
    counters: ProfilerCounters,
    pass_counts: [u64; PASSES.len()],
    trace: VecDeque<TraceFrame>,
    last_upload_bytes: Option<u64>,
    upload_bytes: u64,
    last_error: Option<String>,
}

impl ProfilerCore {
    fn new(enabled: bool, timestamps_supported: bool) -> Self {
        Self {
            enabled,
            timestamps_supported,
            paused: false,
            ring: SampleRing::new(),
            current_seq: None,
            pending_gpu: VecDeque::new(),
            frame_has_scopes: false,
            counters: ProfilerCounters::default(),
            pass_counts: [0; PASSES.len()],
            trace: VecDeque::new(),
            last_upload_bytes: None,
            upload_bytes: 0,
            last_error: None,
        }
    }

    fn recording(&self) -> bool {
        self.enabled && !self.paused
    }

    fn reset(&mut self) {
        self.paused = false;
        self.ring.clear();
        self.current_seq = None;
        self.pending_gpu = VecDeque::new();
        self.frame_has_scopes = false;
        self.counters = ProfilerCounters::default();
        self.pass_counts = [0; PASSES.len()];
        self.trace = VecDeque::new();
        self.last_upload_bytes = None;
        self.upload_bytes = 0;
        self.last_error = None;
    }

    /// Allocate the ring and trace storage on first enable, never while off.
    fn ensure_storage(&mut self) {
        self.ring.ensure_capacity();
        if self.trace.capacity() == 0 {
            self.trace = VecDeque::with_capacity(WINDOW_FRAMES);
        }
    }

    fn begin_frame(&mut self, frame_ms: f32, upload_bytes: Option<u64>) {
        if !self.recording() {
            return;
        }
        self.upload_bytes = 0;
        if let Some(bytes) = upload_bytes {
            let uploaded = self
                .last_upload_bytes
                .map_or(0, |previous| bytes.saturating_sub(previous));
            self.last_upload_bytes = Some(bytes);
            self.upload_bytes = uploaded;
        }
        let seq = self.ring.begin();
        self.current_seq = Some(seq);
        if let Some(sample) = self.ring.get_mut(seq) {
            sample.set(Series::CpuFrame, frame_ms);
            sample.upload_bytes = self.upload_bytes;
        }
    }

    fn record(&mut self, series: Series, value_ms: f32) {
        if !self.recording() || !value_ms.is_finite() {
            return;
        }
        let Some(seq) = self.current_seq else { return };
        if let Some(sample) = self.ring.get_mut(seq) {
            sample.set(series, value_ms.max(0.0));
        }
    }

    /// Called by the render pass helper only when a real query is opened.
    fn note_scope(&mut self, label: &'static str) {
        if !self.recording() {
            return;
        }
        if let Some(index) = PASSES.iter().position(|pass| pass.label == label) {
            self.pass_counts[index] += 1;
        }
        self.counters.scopes += 1;
        // A pass-boundary query pair is a beginning and an end timestamp.
        self.counters.timestamp_writes += 2;
        self.frame_has_scopes = true;
    }

    fn note_resolve(&mut self) {
        if self.enabled {
            self.counters.resolves += 1;
        }
    }

    fn note_end_frame(&mut self) {
        if self.enabled {
            self.counters.end_frames += 1;
        }
    }

    fn attach_gpu(&mut self, seq: u64, results: Vec<wgpu_profiler::GpuTimerQueryResult>) {
        if !self.enabled {
            return;
        }
        let mut total = 0.0f32;
        let mut any = false;
        let mut per_series = [0.0f32; SERIES_COUNT];
        for query in &results {
            let Some(time) = &query.time else { continue };
            let ms = ((time.end - time.start) * 1000.0).max(0.0) as f32;
            any = true;
            total += ms;
            if let Some(series) = Series::for_pass_label(&query.label) {
                per_series[series.index()] += ms;
            }
        }
        if let Some(sample) = self.ring.get_mut(seq)
            && any
        {
            sample.set(Series::GpuTotal, total);
            for series in SERIES {
                if per_series[series.index()] > 0.0 {
                    sample.add(series, per_series[series.index()]);
                }
            }
        }
        self.trace.push_back(TraceFrame { results });
        while self.trace.len() > WINDOW_FRAMES {
            self.trace.pop_front();
        }
    }

    fn series_stats(&self, series: Series) -> SeriesStats {
        let mut stats = SeriesStats::default();
        let mut sum = 0.0f64;
        for sample in self.ring.iter() {
            let Some(value) = sample.get(series) else {
                continue;
            };
            stats.current = value;
            stats.maximum = stats.maximum.max(value);
            sum += f64::from(value);
            stats.samples += 1;
        }
        if stats.samples > 0 {
            stats.average = (sum / stats.samples as f64) as f32;
        }
        stats
    }

    fn plot_points(&self, series: Series, out: &mut Vec<egui_plot::PlotPoint>) {
        out.clear();
        for (index, sample) in self.ring.iter().enumerate() {
            if let Some(value) = sample.get(series) {
                out.push(egui_plot::PlotPoint {
                    x: index as f64,
                    y: f64::from(value),
                });
            }
        }
    }

    fn upload_kb(&self) -> f64 {
        self.upload_bytes as f64 / 1024.0
    }
}

struct TraceFrame {
    results: Vec<wgpu_profiler::GpuTimerQueryResult>,
}

/// The app-owned profiler state. `device`/`queue` are `None` only when eframe
/// has no wgpu render state (for example a CPU-only test build).
pub struct GpuProfilerSlot {
    core: ProfilerCore,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    #[cfg_attr(not(feature = "tracy"), allow(dead_code))]
    backend: Option<wgpu::Backend>,
    gpu: Option<GpuProfiler>,
}

impl GpuProfilerSlot {
    pub fn new(
        device: Option<wgpu::Device>,
        queue: Option<wgpu::Queue>,
        timestamps_supported: bool,
        backend: Option<wgpu::Backend>,
    ) -> Self {
        let timestamps_available = timestamps_supported && device.is_some() && queue.is_some();
        Self {
            core: ProfilerCore::new(false, timestamps_available),
            device,
            queue,
            backend,
            gpu: None,
        }
    }

    /// Test/headless constructor: CPU series only, no wgpu device.
    #[cfg(test)]
    fn headless(enabled: bool) -> Self {
        let mut slot = Self::new(None, None, false, None);
        if enabled {
            slot.enable();
        }
        slot
    }

    #[cfg(test)]
    pub fn is_enabled(&self) -> bool {
        self.core.enabled
    }

    pub fn timestamps_supported(&self) -> bool {
        self.core.timestamps_supported && self.gpu.is_some()
    }

    pub fn last_error(&self) -> Option<&str> {
        self.core.last_error.as_deref()
    }

    pub fn counters(&self) -> ProfilerCounters {
        self.core.counters
    }

    /// Turn recording on. The first call creates the wgpu-profiler and its
    /// query resources lazily; while off they do not exist at all.
    pub fn enable(&mut self) {
        if self.core.enabled {
            return;
        }
        self.core.enabled = true;
        self.core.paused = false;
        self.core.last_error = None;
        self.core.ensure_storage();
        if !self.core.timestamps_supported {
            return;
        }
        let (Some(device), Some(queue)) = (self.device.as_ref(), self.queue.as_ref()) else {
            self.core.timestamps_supported = false;
            return;
        };
        let settings = GpuProfilerSettings {
            enable_timer_queries: true,
            // Debug groups are not needed for the graph and would add command
            // encoder work on every pass.
            enable_debug_groups: false,
            max_num_pending_frames: 3,
        };
        #[cfg(feature = "tracy")]
        let created = match self.backend {
            Some(backend) => GpuProfiler::new_with_tracy_client(settings, backend, device, queue),
            None => GpuProfiler::new(device, settings),
        };
        #[cfg(not(feature = "tracy"))]
        let created = {
            let _ = queue;
            GpuProfiler::new(device, settings)
        };
        match created {
            Ok(gpu) => self.gpu = Some(gpu),
            Err(error) => {
                self.core.timestamps_supported = false;
                self.core.last_error = Some(format!("GPU timestamp queries unavailable: {error}"));
            }
        }
    }

    pub fn disable(&mut self) {
        self.core.enabled = false;
        // Dropping the profiler releases its query sets and buffers.
        self.gpu = None;
        self.core.reset();
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.core.paused = paused;
    }

    pub fn is_paused(&self) -> bool {
        self.core.paused
    }

    pub fn begin_frame(&mut self, frame_ms: f32, upload_bytes: Option<u64>) {
        self.core.begin_frame(frame_ms, upload_bytes);
    }

    /// Close the previous rendered frame, resolve its timestamps on a command
    /// encoder submitted after the render work, and harvest any completed
    /// results. Never waits for the GPU.
    pub fn finish_frame(&mut self) {
        if !self.core.enabled {
            return;
        }
        let Some(queue) = self.queue.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else { return };
        if self.core.frame_has_scopes {
            let Some(device) = self.device.as_ref() else {
                return;
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-profiler-resolve"),
            });
            gpu.resolve_queries(&mut encoder);
            self.core.note_resolve();
            queue.submit(Some(encoder.finish()));
            let seq = self.core.current_seq;
            match gpu.end_frame() {
                Ok(()) => {
                    self.core.note_end_frame();
                    if let Some(seq) = seq {
                        self.core.pending_gpu.push_back(seq);
                        if self.core.pending_gpu.len() > MAX_PENDING_GPU_FRAMES {
                            self.core.pending_gpu.pop_front();
                        }
                    }
                }
                Err(error) => {
                    self.core.last_error = Some(format!("GPU profiler frame: {error}"));
                }
            }
            self.core.frame_has_scopes = false;
        }
        if let Some(results) = gpu.process_finished_frame(queue.get_timestamp_period())
            && let Some(seq) = self.core.pending_gpu.pop_front()
        {
            self.core.attach_gpu(seq, results);
        }
    }

    pub fn record(&mut self, series: Series, value_ms: f32) {
        self.core.record(series, value_ms);
    }

    pub fn stats(&self, series: Series) -> SeriesStats {
        self.core.series_stats(series)
    }

    pub fn plot_points(&self, series: Series, out: &mut Vec<egui_plot::PlotPoint>) {
        self.core.plot_points(series, out);
    }

    pub fn upload_kb(&self) -> f64 {
        self.core.upload_kb()
    }

    pub fn retained_frames(&self) -> usize {
        self.core.ring.len()
    }

    pub fn save_trace(&self, path: &Path) -> Result<(), String> {
        write_merged_trace(path, &self.core.ring, &self.core.trace)
    }

    /// Encode one render pass through the profiler, or call the draw closure
    /// directly when profiling is off/paused. The pass label is the same value
    /// used for the wgpu descriptor and the profiler scope.
    pub fn render_pass(
        &mut self,
        label: &'static str,
        encoder: &mut wgpu::CommandEncoder,
        mut descriptor: wgpu::RenderPassDescriptor<'_>,
        draw: impl FnOnce(&mut wgpu::RenderPass<'static>),
    ) {
        if !self.core.recording() {
            descriptor.label = Some(label);
            let pass = encoder.begin_render_pass(&descriptor);
            let mut pass = pass.forget_lifetime();
            draw(&mut pass);
            return;
        }
        let Some(gpu) = self.gpu.as_ref() else {
            descriptor.label = Some(label);
            let pass = encoder.begin_render_pass(&descriptor);
            let mut pass = pass.forget_lifetime();
            draw(&mut pass);
            return;
        };
        self.core.note_scope(label);
        let query = gpu.begin_pass_query(label, encoder);
        // Rebuild the descriptor with the query lifetime so the caller's
        // attachments can be reborrowed for this pass only. Assigning the
        // timestamp writes into the caller's descriptor would tie its whole
        // lifetime to the local query.
        let pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: descriptor.color_attachments,
            depth_stencil_attachment: descriptor.depth_stencil_attachment,
            timestamp_writes: query.render_pass_timestamp_writes(),
            occlusion_query_set: descriptor.occlusion_query_set,
            multiview_mask: descriptor.multiview_mask,
        };
        let pass = encoder.begin_render_pass(&pass_descriptor);
        let mut pass = pass.forget_lifetime();
        draw(&mut pass);
        drop(pass);
        drop(pass_descriptor);
        gpu.end_query(encoder, query);
    }
}

/// Pass-site entry point that works with an optional profiler slot.
pub fn render_pass(
    profiler: Option<&mut GpuProfilerSlot>,
    label: &'static str,
    encoder: &mut wgpu::CommandEncoder,
    descriptor: wgpu::RenderPassDescriptor<'_>,
    draw: impl FnOnce(&mut wgpu::RenderPass<'static>),
) {
    match profiler {
        Some(profiler) => profiler.render_pass(label, encoder, descriptor, draw),
        None => {
            let mut descriptor = descriptor;
            descriptor.label = Some(label);
            let pass = encoder.begin_render_pass(&descriptor);
            let mut pass = pass.forget_lifetime();
            draw(&mut pass);
        }
    }
}

/// Per-series visibility/status state for the frame profiler window.
pub struct ProfilerWindow {
    pub visible: [bool; SERIES_COUNT],
    pub status: Option<String>,
    /// Reused point buffers, one per series; `clear` + `push` keeps the live
    /// graph allocation-free after the first frame.
    pub plot_points: [Vec<egui_plot::PlotPoint>; SERIES_COUNT],
}

impl ProfilerWindow {
    pub fn new() -> Self {
        let mut visible = [false; SERIES_COUNT];
        for series in SERIES {
            visible[series.index()] = series.default_visible();
        }
        // Empty until the window first draws; an off profiler allocates none.
        let plot_points: [Vec<egui_plot::PlotPoint>; SERIES_COUNT] =
            std::array::from_fn(|_| Vec::new());
        Self {
            visible,
            status: None,
            plot_points,
        }
    }
}

/// wgpu-profiler's Chrome-trace export written to a temporary file, then
/// extended with our CPU series (and an upload counter) before being written
/// to `path`.
fn write_merged_trace(
    path: &Path,
    ring: &SampleRing,
    trace: &VecDeque<TraceFrame>,
) -> Result<(), String> {
    let mut gpu_results = Vec::new();
    for frame in trace {
        gpu_results.extend(frame.results.iter().cloned());
    }
    let temp = std::env::temp_dir().join(format!(
        "nodecraft-frame-trace-{}-{}-{}.json",
        std::process::id(),
        ring.next_seq,
        TRACE_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let write = wgpu_profiler::chrometrace::write_chrometrace(&temp, &gpu_results);
    let read = std::fs::read_to_string(&temp);
    let _ = std::fs::remove_file(&temp);
    write.map_err(|error| format!("write GPU trace: {error}"))?;
    let text = read.map_err(|error| format!("read GPU trace: {error}"))?;
    let mut root: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("parse GPU trace: {error}"))?;
    let events = root
        .get_mut("traceEvents")
        .and_then(serde_json::Value::as_array_mut)
        .ok_or_else(|| "GPU trace has no traceEvents array".to_owned())?;

    let gpu_base = events
        .iter()
        .filter_map(|event| event.get("ts").and_then(serde_json::Value::as_f64))
        .fold(f64::INFINITY, f64::min);
    let mut clock_us = if gpu_base.is_finite() { gpu_base } else { 0.0 };
    for sample in ring.iter() {
        for series in SERIES {
            let Some(value) = sample.get(series) else {
                continue;
            };
            events.push(serde_json::json!({
                "pid": 0,
                "tid": 0,
                "ts": clock_us,
                "dur": f64::from(value) * 1000.0,
                "ph": "X",
                "name": series.key(),
            }));
        }
        let frame_us = sample
            .get(Series::CpuFrame)
            .map_or(0.0, |ms| f64::from(ms) * 1000.0);
        if sample.upload_bytes > 0 {
            events.push(serde_json::json!({
                "pid": 0,
                "tid": 0,
                "ts": clock_us,
                "ph": "C",
                "name": "upload_bytes",
                "args": { "bytes": sample.upload_bytes },
            }));
        }
        clock_us += frame_us;
    }

    let text = serde_json::to_string_pretty(&root)
        .map_err(|error| format!("serialize merged trace: {error}"))?;
    std::fs::write(path, text).map_err(|error| format!("write trace: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timed(label: &str, milliseconds: f64) -> wgpu_profiler::GpuTimerQueryResult {
        let start = 1.0;
        wgpu_profiler::GpuTimerQueryResult {
            label: label.to_owned(),
            pid: 1,
            tid: std::thread::current().id(),
            time: Some(start..start + milliseconds / 1000.0),
            nested_queries: Vec::new(),
        }
    }

    #[test]
    fn profiler_off_issues_no_queries_or_ring_writes() {
        let mut core = ProfilerCore::new(false, true);
        core.begin_frame(16.0, Some(4096));
        for pass in PASSES {
            core.note_scope(pass.label);
        }
        core.note_resolve();
        core.note_end_frame();
        core.attach_gpu(0, vec![timed(pass::SCENE, 1.0)]);

        assert_eq!(core.counters, ProfilerCounters::default());
        assert_eq!(core.ring.len(), 0);
        assert!(core.ring.samples.is_empty());
        assert!(core.trace.is_empty());
        assert_eq!(core.trace.capacity(), 0);
        assert!(core.current_seq.is_none());
    }

    #[test]
    fn profiler_on_records_each_named_pass_and_series() {
        let mut core = ProfilerCore::new(true, true);
        core.begin_frame(16.0, None);
        for pass in PASSES {
            core.note_scope(pass.label);
        }
        assert_eq!(core.counters.scopes, PASSES.len() as u64);
        assert_eq!(core.counters.timestamp_writes, PASSES.len() as u64 * 2);
        assert_eq!(core.pass_counts, [1; PASSES.len()]);
        assert!(core.frame_has_scopes);

        let seq = core.current_seq.expect("current frame");
        core.attach_gpu(
            seq,
            vec![
                timed(pass::SCENE, 2.0),
                timed(pass::PICK, 0.5),
                timed(pass::BACKDROP_SCENE, 0.2),
                timed(pass::BACKDROP_BLUR_H, 0.1),
                timed(pass::BACKDROP_BLUR_V, 0.1),
                timed(pass::BACKDROP_BLUR_FINAL, 0.05),
            ],
        );

        for series in [
            Series::GpuTotal,
            Series::GpuScene,
            Series::GpuPick,
            Series::GpuBackdrop,
            Series::GpuBlur,
        ] {
            let stats = core.series_stats(series);
            assert!(stats.samples > 0, "{series:?} should have a sample");
            assert!(stats.current > 0.0, "{series:?} should be non-zero");
        }
        assert!((core.series_stats(Series::GpuTotal).current - 2.95).abs() < 1.0e-4);
        assert!((core.series_stats(Series::GpuBlur).current - 0.25).abs() < 1.0e-4);
        assert_eq!(core.trace.len(), 1);
    }

    #[test]
    fn ring_never_grows_past_the_window() {
        let mut core = ProfilerCore::new(true, true);
        for frame in 0..(WINDOW_FRAMES + 137) {
            core.begin_frame(frame as f32, None);
            core.record(Series::UiUpdate, 1.0);
        }
        assert_eq!(core.ring.len(), WINDOW_FRAMES);
        let stats = core.series_stats(Series::CpuFrame);
        assert_eq!(stats.samples, WINDOW_FRAMES);
        assert_eq!(core.ring.samples.len(), WINDOW_FRAMES);
    }

    #[test]
    fn paused_recording_leaves_the_ring_alone() {
        let mut core = ProfilerCore::new(true, true);
        core.begin_frame(16.0, None);
        let before = core.ring.len();
        core.paused = true;
        core.begin_frame(16.0, None);
        core.record(Series::UiUpdate, 3.0);
        assert_eq!(core.ring.len(), before);
    }

    #[test]
    fn chrome_trace_export_is_valid_json_with_cpu_and_gpu_events() {
        let mut core = ProfilerCore::new(true, true);
        core.begin_frame(16.7, Some(2048));
        core.record(Series::UiUpdate, 4.0);
        core.record(Series::EvalSubmit, 2.0);
        core.record(Series::DisplayApply, 1.0);
        for pass in PASSES {
            core.note_scope(pass.label);
        }
        let seq = core.current_seq.expect("current frame");
        core.attach_gpu(
            seq,
            vec![
                timed(pass::SCENE, 5.0),
                timed(pass::PICK, 1.0),
                timed(pass::BACKDROP_SCENE, 0.5),
            ],
        );

        let path = std::env::temp_dir().join(format!(
            "nodecraft-profiler-test-{}.json",
            std::process::id()
        ));
        write_merged_trace(&path, &core.ring, &core.trace).expect("write trace");
        let text = std::fs::read_to_string(&path).expect("read trace");
        let _ = std::fs::remove_file(&path);
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let events = value
            .get("traceEvents")
            .and_then(serde_json::Value::as_array)
            .expect("traceEvents");
        let names: Vec<&str> = events
            .iter()
            .filter_map(|event| event.get("name").and_then(serde_json::Value::as_str))
            .collect();
        assert!(names.contains(&pass::SCENE));
        assert!(names.contains(&"cpu.frame"));
        assert!(names.contains(&"cpu.ui"));
        assert!(names.contains(&"gpu.total"));
        assert!(events.iter().all(|event| event.get("ph").is_some()));
    }

    #[test]
    fn disabled_slot_does_nothing_even_when_driven() {
        let mut slot = GpuProfilerSlot::headless(false);
        slot.begin_frame(16.0, Some(4096));
        slot.record(Series::UiUpdate, 3.0);
        slot.finish_frame();
        assert_eq!(slot.stats(Series::CpuFrame).samples, 0);
        assert_eq!(slot.counters(), ProfilerCounters::default());
        assert!(!slot.is_enabled());
    }

    #[test]
    fn cpu_only_trace_is_valid_json() {
        let mut core = ProfilerCore::new(true, false);
        core.begin_frame(16.0, None);
        core.record(Series::UiUpdate, 2.0);
        let path = std::env::temp_dir().join(format!(
            "nodecraft-profiler-cpu-trace-{}.json",
            std::process::id()
        ));
        write_merged_trace(&path, &core.ring, &core.trace).expect("write trace");
        let text = std::fs::read_to_string(&path).expect("read trace");
        let _ = std::fs::remove_file(&path);
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let names: Vec<&str> = value
            .get("traceEvents")
            .and_then(serde_json::Value::as_array)
            .expect("traceEvents")
            .iter()
            .filter_map(|event| event.get("name").and_then(serde_json::Value::as_str))
            .collect();
        assert!(names.contains(&"cpu.frame"));
        assert!(names.contains(&"cpu.ui"));
        assert!(!names.iter().any(|name| name.starts_with("gpu.")));
    }

    #[test]
    fn headless_slot_keeps_cpu_series_without_gpu_queries() {
        let mut slot = GpuProfilerSlot::headless(true);
        assert!(slot.is_enabled());
        assert!(!slot.timestamps_supported());
        slot.begin_frame(16.0, Some(1024));
        slot.record(Series::UiUpdate, 2.0);
        slot.finish_frame();
        let stats = slot.stats(Series::CpuFrame);
        assert_eq!(stats.samples, 1);
        assert!(slot.stats(Series::GpuTotal).samples == 0);
        assert_eq!(slot.counters().resolves, 0);
        slot.disable();
        assert!(!slot.is_enabled());
    }
}
