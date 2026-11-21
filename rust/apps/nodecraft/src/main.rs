#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

//! NodeCraft native front end.
//!
//! This is the first slice of the Swift/WebView front-end port. The authority is
//! still the existing NodeCraft session/evaluation engine; the UI only sends graph
//! requests and displays the resulting mesh.

mod adapter;
mod app;
mod control_server;
mod crash_report;
#[cfg(feature = "gpu-viewport")]
mod edge_ribbon;
mod engine_bridge;
mod expression_input;
mod file_dialog;
#[cfg(feature = "gpu-viewport")]
mod gpu_backdrop;
#[cfg(feature = "gpu-viewport")]
mod gpu_pick;
#[cfg(feature = "gpu-viewport")]
mod gpu_plan;
mod gpu_profiler;
#[cfg(feature = "gpu-viewport")]
mod gpu_viewport;
mod hover;
mod plan_math;
mod sketch_gui;
mod source_service;
mod theme;
mod ui_shape;
// Measurement-only harness; its test is `#[ignore]`d and never runs by default.
#[cfg(test)]
mod edit_overhead;
mod inspector;
mod view_navigator;
mod viewer_gizmo;
mod viewport;
mod viewport_grid;
mod widgets;
mod window_chrome;

use std::path::PathBuf;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "gpu-viewport")]
use crate::gpu_viewport::msaa_samples;

#[cfg(not(feature = "gpu-viewport"))]
fn msaa_samples() -> u32 {
    1
}

fn main() -> eframe::Result {
    crash_report::install_hook();
    match crash_report::catch(run) {
        Some(result) => result,
        None => {
            std::process::exit(101);
        }
    }
}

fn run() -> eframe::Result {
    if let Err(error) = nodecraft_engine::thread_pools::init_rayon_pool() {
        eprintln!("{error}");
    }
    #[cfg(target_os = "macos")]
    nodecraft_macos::install_observer();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--check") {
        return check(args.get(1).map(PathBuf::from));
    }
    let crash_test = args.iter().any(|arg| arg == "--crash-test");
    if crash_test && std::env::var_os("NODECRAFT_DEV").is_none() {
        eprintln!("--crash-test is a developer check; set NODECRAFT_DEV=1 to enable it");
        std::process::exit(2);
    }
    args.retain(|arg| arg != "--crash-test");
    let render_model = args.iter().any(|arg| arg == "--render-model");
    let screenshot_flag = if render_model { "--render-model" } else { "--screenshot" };
    let screenshot = args
        .iter()
        .position(|arg| arg == screenshot_flag)
        .and_then(|index| args.get(index + 1).cloned())
        .map(PathBuf::from)
        .or_else(|| {
            if render_model {
                None
            } else {
                std::env::var_os("NODECRAFT_EGUI_SCREENSHOT").map(PathBuf::from)
            }
        });
    if let Some(index) = args.iter().position(|arg| arg == screenshot_flag) {
        args.remove(index);
        if index < args.len() {
            args.remove(index);
        }
    }
    if render_model && screenshot.is_none() {
        eprintln!("usage: nodecraft --render-model <output.png> <document.nodecraft>");
        std::process::exit(2);
    }
    let initial = args.first().cloned().map(PathBuf::from);
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([1440.0, 900.0])
        .with_min_inner_size([960.0, 600.0])
        .with_title("NodeCraft");
    let viewport = window_chrome::configure_main_viewport(viewport);
    let options = eframe::NativeOptions {
        viewport,
        wgpu_options: wgpu_options_with_timestamps(),
        // The SolidCraft viewport needs a depth buffer in eframe's pass. Matching the
        // scene and window depth targets lets it draw directly into the egui pass
        // instead of relying on an offscreen blit that can stay hidden on first show.
        depth_buffer: 24,
        stencil_buffer: 8,
        multisampling: msaa_samples() as u16,
        ..Default::default()
    };
    eframe::run_native(
        "NodeCraft",
        options,
        Box::new(move |cc| {
            Ok(Box::new(app::NodeCraftApp::new(cc, initial, screenshot, render_model, crash_test)))
        }),
    )
}

/// Ask for the pass-boundary timestamp feature only when this adapter exposes
/// it. The frame profiler is optional, but a device's enabled features cannot
/// change after creation, so this request is made unconditionally (and cheaply)
/// for every device that can support it. Backends without it (GL/WebGL) simply
/// leave the bit out and the profiler shows CPU timings only.
fn wgpu_options_with_timestamps() -> egui_wgpu::WgpuConfiguration {
    let mut options = egui_wgpu::WgpuConfiguration::default();
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_setup {
        let base = setup.device_descriptor.clone();
        setup.device_descriptor = std::sync::Arc::new(move |adapter| {
            let mut descriptor = base(adapter);
            if adapter.features().contains(eframe::wgpu::Features::TIMESTAMP_QUERY) {
                descriptor.required_features |= eframe::wgpu::Features::TIMESTAMP_QUERY;
            }
            descriptor
        });
    }
    options
}

/// Headless smoke check used by CI and by the first-port acceptance test.
fn check(path: Option<PathBuf>) -> eframe::Result {
    let Some(path) = path else {
        eprintln!("usage: nodecraft --check <document.nodecraft>");
        std::process::exit(2);
    };
    let document: nodecraft_session::GraphDocument =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|error| {
            eprintln!("{}: {error}", path.display());
            std::process::exit(2);
        }))
        .unwrap_or_else(|error| {
            eprintln!("{}: {error}", path.display());
            std::process::exit(2);
        });
    let Some(target) = engine_bridge::target_for(&document, None) else {
        eprintln!("{}: document has no target", path.display());
        std::process::exit(2);
    };
    let (evaluation, _) = engine_bridge::evaluate_flat_exact(
        &document,
        target.clone(),
        nodecraft_engine::EngineCache::new(),
        None,
        None,
        false,
        &Default::default(),
        None,
        &[],
        None,
        false,
    );
    let Some(evaluation) = evaluation else {
        eprintln!("{}: document has no target", path.display());
        std::process::exit(2);
    };
    let mut plan_cache = nodecraft_display::PlanCache::new(1024);
    let mesh = evaluation
        .exact_parts
        .get(&target)
        .and_then(|part| {
            nodecraft_display::mesh_part(
                part,
                nodecraft_display::default_quality(),
                &mut plan_cache,
            )
            .ok()
        })
        .map(|display| display.mesh);
    println!(
        "{} target={} ok={} verts={} tris={} edges={} elapsed_ms={:.1}",
        path.display(),
        evaluation.target,
        evaluation.error.is_none(),
        mesh.as_ref().map_or(0, |mesh| mesh.vertices.len()),
        mesh.as_ref().map_or(0, |mesh| mesh.faces.len()),
        mesh.as_ref().map_or(0, |mesh| mesh.brep_edges.len()),
        evaluation.elapsed_ms
    );
    if let Some(error) = evaluation.error {
        eprintln!("{error}");
        std::process::exit(1);
    }
    Ok(())
}
