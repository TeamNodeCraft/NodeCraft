//! Display-service contract.
//!
//! Evaluation produces exact Parts; this module describes the display work
//! derived from them. The types are deliberately independent of the node graph
//! and of the viewport.

use std::collections::BTreeMap;
use std::sync::Arc;

use nodecraft_exact::ExactPart;

use crate::{DisplayQuality, PlanCache, PlanInstance, PartPlans};

/// Exact inputs for one display revision.
///
/// The breakpoint map is keyed by the evaluated target id; overlay parts are
/// already resolved by the exact runtime and never require a DAG walk here.
#[derive(Clone, Debug, Default)]
pub struct ExactSnapshot {
    pub breakpoint: BTreeMap<String, ExactPart>,
    pub overlays: Vec<OverlayRequest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OverlayKind {
    EditInput,
    FeatureGhost,
    ComparisonBefore,
    ComparisonAfter,
    DifferenceAdded,
    DifferenceRemoved,
}

/// One overlay's exact geometry. The service only plans this body; it does not
/// resolve node ids or evaluate the graph.
#[derive(Clone, Debug)]
pub struct OverlayRequest {
    pub kind: OverlayKind,
    pub part: Arc<ExactPart>,
}

/// The app clears this while the GPU tessellation pipeline is still compiling.
/// Breakpoint frames then carry a full CPU mesh so the object renderer can draw
/// until the plan path is ready. Defaults to true so library users/tests use
/// the plan path directly.
static GPU_TESS_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_gpu_tess_ready(ready: bool) {
    GPU_TESS_READY.store(ready, std::sync::atomic::Ordering::Relaxed);
}

#[must_use]
pub fn gpu_tess_ready() -> bool {
    GPU_TESS_READY.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Debug)]
pub struct DisplayRequest {
    pub revision: u64,
    /// Stable id of the exact snapshot this request displays. Camera quality
    /// requests keep the snapshot id and change only quality/revision.
    pub snapshot_id: u64,
    pub quality: DisplayQuality,
    /// Build fillet/chamfer added-face plans for the breakpoint result.
    pub feature_ghost: bool,
    pub snapshot: Arc<ExactSnapshot>,
}

#[derive(Clone, Debug, Default)]
pub struct OverlayPlans {
    pub kind: Option<OverlayKind>,
    pub plans: Vec<PlanInstance>,
    pub mesh: nodecraft_protocol::Mesh,
    pub edge_mesh: nodecraft_protocol::Mesh,
    pub stats: Vec<nodecraft_layers::TessStats>,
}

#[derive(Clone, Debug, Default)]
pub struct DisplayFrame {
    pub revision: u64,
    pub snapshot_id: u64,
    pub plan_us: u64,
    pub error: Option<String>,
    pub breakpoint: BTreeMap<String, PartPlans>,
    pub feature_plans: Vec<PlanInstance>,
    pub overlays: Vec<OverlayPlans>,
}

/// Keep in sync with `solidcraft_viewport_gpu::MAX_BATCH`.
const GPU_MAX_BATCH: usize = 1 << 20;

/// Plans the pinned GPU evaluator refuses are drawn by the CPU object path for
/// the whole frame. Mixing a CPU body into a plan-active scene would require a
/// second object pass, which the viewport does not have.
fn gpu_plans_supported(plans: &[PlanInstance]) -> bool {
    plans.iter().all(|instance| {
        let plan = &instance.plan;
        plan.vertex_count() <= GPU_MAX_BATCH
            && plan.faces.iter().all(|face| {
                face.poles.is_empty()
                    && face.surface.is_some_and(|surface| (surface as usize) < plan.surfaces.len())
            })
    })
}

/// Synchronous display planner. A later worker wrapper publishes its result
/// through a latest-wins slot; this is the single entry point both paths use.
#[derive(Clone, Debug, Default)]
pub struct DisplayService {
    pub cache: PlanCache,
}

impl DisplayService {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self { cache: PlanCache::new(limit) }
    }

    pub fn plan(&mut self, request: &DisplayRequest) -> Result<DisplayFrame, String> {
        let started = std::time::Instant::now();
        let timeline = std::time::Instant::now();
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!("[display {}] plan start", request.revision);
        }
        let mut frame = DisplayFrame { revision: request.revision, ..DisplayFrame::default() };
        for (target, part) in &request.snapshot.breakpoint {
            if request.feature_ghost {
                let display = crate::mesh_part_with_features(part, request.quality, &mut self.cache)?;
                frame.feature_plans = display.feature_plans;
                frame.breakpoint.insert(
                    target.clone(),
                    PartPlans { plans: display.plans, stats: display.stats, edge_mesh: display.mesh },
                );
            } else if !gpu_tess_ready() {
                // The background GPU tessellation compile is still running.
                // Supply the full CPU mesh so the breakpoint is visible and
                // editable immediately; the plans it carries become resident
                // when the pipeline is ready.
                let display = crate::mesh_part(part, request.quality, &mut self.cache)?;
                frame.breakpoint.insert(
                    target.clone(),
                    PartPlans {
                        plans: display.plans,
                        stats: display.stats,
                        edge_mesh: display.mesh,
                    },
                );
            } else {
                let planned = crate::plan_part(part, request.quality, &mut self.cache)?;
                if gpu_plans_supported(&planned.plans) {
                    frame.breakpoint.insert(target.clone(), planned);
                } else {
                    // A pole face or oversized dispatch cannot run through the
                    // GPU planner. Use the CPU object path for this target; an
                    // empty plan set disables plan rendering for the frame.
                    let display = crate::mesh_part(part, request.quality, &mut self.cache)?;
                    frame.breakpoint.insert(
                        target.clone(),
                        PartPlans { plans: Vec::new(), stats: display.stats, edge_mesh: display.mesh },
                    );
                }
            }
        }
        for overlay in &request.snapshot.overlays {
            let overlay_started = std::time::Instant::now();
            let display = crate::mesh_part(overlay.part.as_ref(), request.quality, &mut self.cache)?;
            if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
                eprintln!(
                    "[display {}] overlay {:?} {}ms",
                    request.revision,
                    overlay.kind,
                    overlay_started.elapsed().as_millis()
                );
            }
            frame.overlays.push(OverlayPlans {
                kind: Some(overlay.kind),
                plans: display.plans,
                mesh: display.mesh,
                edge_mesh: nodecraft_protocol::Mesh::default(),
                stats: display.stats,
            });
        }
        frame.plan_us = started.elapsed().as_micros() as u64;
        if std::env::var_os("NODECRAFT_TIMELINE").is_some() {
            eprintln!(
                "[display {}] plan done +{:?} ({} breakpoints)",
                request.revision,
                timeline.elapsed(),
                frame.breakpoint.len(),
            );
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nodecraft_solidcore::solidcore::math::Point3;
    use nodecraft_solidcore::solidcore::primitive::api_primitives as api;

    fn box_part() -> ExactPart {
        let mut kernel = nodecraft_solidcore::solidcore::kernel::Kernel::default();
        let brep = api::create_box(
            &mut kernel,
            Point3::new(-5.0, -5.0, -5.0),
            10.0,
            10.0,
            10.0,
        )
        .expect("box");
        ExactPart {
            bodies: vec![nodecraft_exact::ExactBody::new(
                nodecraft_exact::BodyKey::new(1, 0),
                Arc::new(brep),
            )],
        }
    }

    #[test]
    fn plans_breakpoint_and_overlay_geometry() {
        let part = Arc::new(box_part());
        let snapshot = ExactSnapshot {
            breakpoint: BTreeMap::from([("look".into(), box_part())]),
            overlays: vec![OverlayRequest { kind: OverlayKind::EditInput, part: Arc::clone(&part) }],
        };
        let mut service = DisplayService::new(8);
        let frame = service
            .plan(&DisplayRequest {
                revision: 7,
                snapshot_id: 7,
                quality: DisplayQuality::Full(0.35),
                feature_ghost: false,
                snapshot: Arc::new(snapshot),
            })
            .expect("display frame");
        assert_eq!(frame.revision, 7);
        assert_eq!(frame.breakpoint.len(), 1);
        assert_eq!(frame.overlays.len(), 1);
        assert!(!frame.breakpoint["look"].plans.is_empty());
        assert!(!frame.overlays[0].plans.is_empty());
    }

    #[test]
    fn plans_without_a_surface_are_not_sent_to_the_gpu() {
        let plan = std::sync::Arc::new(solidcore_tess::TessPlan {
            faces: vec![solidcore_tess::FacePlan::default()],
            ..Default::default()
        });
        let instance = PlanInstance {
            target: "x".into(),
            plan,
            face_colors: Vec::new(),
            transforms: Vec::new(),
            edge_ids: std::sync::Arc::from(Vec::new()),
        };
        assert!(!gpu_plans_supported(&[instance]));
    }

    #[test]
    fn worker_publishes_the_latest_frame() {
        let snapshot = ExactSnapshot {
            breakpoint: BTreeMap::from([("look".into(), box_part())]),
            overlays: Vec::new(),
        };
        let worker = DisplayWorker::start(8, || {}).expect("display worker");
        worker.submit(DisplayRequest {
            revision: 11,
            snapshot_id: 11,
            quality: DisplayQuality::Full(0.35),
            feature_ghost: false,
            snapshot: Arc::new(snapshot),
        });
        let mut frame = None;
        for _ in 0..100 {
            frame = worker.latest_frame();
            if frame.as_ref().is_some_and(|frame| frame.revision == 11) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let frame = frame.expect("published frame");
        assert_eq!(frame.revision, 11);
        assert!(!frame.breakpoint["look"].plans.is_empty());
    }
}

/// Latest-value atomic slot. Writers never lock; readers only load an `Arc`.
#[derive(Debug)]
pub struct LatestSlot<T> {
    value: arc_swap::ArcSwapOption<T>,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self { value: arc_swap::ArcSwapOption::empty() }
    }
}

impl<T> LatestSlot<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn store(&self, value: Arc<T>) {
        self.value.store(Some(value));
    }

    #[must_use]
    pub fn load(&self) -> Option<Arc<T>> {
        self.value.load_full()
    }

    pub fn clear(&self) {
        self.value.store(None);
    }
}

/// Lock-free display worker boundary.
///
/// The UI stores the newest `DisplayRequest`; the service takes it, plans it,
/// and publishes a `DisplayFrame`. The worker never holds a lock that the UI
/// needs, and it never publishes a frame whose request has been superseded.
pub struct DisplayWorker {
    request: Arc<LatestSlot<DisplayRequest>>,
    frame: Arc<LatestSlot<DisplayFrame>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    thread: std::thread::Thread,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for DisplayWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DisplayWorker").finish_non_exhaustive()
    }
}

impl Drop for DisplayWorker {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        self.thread.unpark();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl DisplayWorker {
    /// Start the worker. `repaint` runs after a frame is published; the app
    /// passes `egui::Context::request_repaint` there.
    pub fn start(
        cache_limit: usize,
        repaint: impl Fn() + Send + 'static,
    ) -> std::io::Result<Self> {
        let request = Arc::new(LatestSlot::<DisplayRequest>::new());
        let frame = Arc::new(LatestSlot::<DisplayFrame>::new());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_request = Arc::clone(&request);
        let worker_frame = Arc::clone(&frame);
        let worker_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("nodecraft-display".into())
            .spawn(move || {
                let mut service = DisplayService::new(cache_limit);
                let mut last_planned = None;
                while !worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::park();
                    if worker_stop.load(std::sync::atomic::Ordering::Acquire) {
                        break;
                    }
                    let Some(next) = worker_request.load() else { continue };
                    let revision = next.revision;
                    if last_planned == Some(revision) {
                        continue;
                    }
                    match service.plan(&next) {
                        Ok(mut frame) => {
                            // Do not publish a frame the service has already
                            // superseded while it was planning.
                            if worker_request
                                .load()
                                .is_some_and(|current| current.revision != revision)
                            {
                                std::thread::yield_now();
                                continue;
                            }
                            frame.revision = revision;
                            frame.snapshot_id = next.snapshot_id;
                            last_planned = Some(revision);
                            worker_frame.store(Arc::new(frame));
                            repaint();
                        }
                        Err(error) => {
                            // Publish the failure for this exact revision so the
                            // UI can retire its pending request without guessing
                            // from a timeout.
                            eprintln!("display service failed revision {revision}: {error}");
                            last_planned = Some(revision);
                            worker_frame.store(Arc::new(DisplayFrame {
                                revision,
                                snapshot_id: next.snapshot_id,
                                error: Some(error),
                                ..DisplayFrame::default()
                            }));
                            repaint();
                        }
                    }
                }
            })?;
        Ok(Self {
            request,
            frame,
            stop,
            thread: handle.thread().clone(),
            handle: Some(handle),
        })
    }

    pub fn submit(&self, request: DisplayRequest) {
        self.request.store(Arc::new(request));
        self.thread.unpark();
    }

    pub fn latest_frame(&self) -> Option<Arc<DisplayFrame>> {
        self.frame.load()
    }
}
