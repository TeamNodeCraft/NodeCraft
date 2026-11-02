#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! NodeCraft egui front end.
//!
//! This is the first slice of the Swift/WebView front-end port. The authority is
//! still the existing NodeCraft session/evaluation engine; the UI only sends graph
//! requests and displays the resulting mesh.

mod adapter;
mod app;
mod control_server;
mod engine_bridge;
mod theme;
mod sketch_gui;
mod ui_shape;
#[cfg(feature = "gpu-viewport")]
mod gpu_backdrop;
#[cfg(feature = "gpu-viewport")]
mod gpu_viewport;
mod inspector;
mod viewport;
mod viewport_grid;
mod viewer_gizmo;

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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--check") {
        return check(args.get(1).map(PathBuf::from));
    }
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
        eprintln!("usage: nodecraft-egui --render-model <output.png> <document.nodecraft>");
        std::process::exit(2);
    }
    let initial = args.first().cloned().map(PathBuf::from);
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([1440.0, 900.0])
        .with_min_inner_size([960.0, 600.0])
        .with_title("NodeCraft");
    #[cfg(target_os = "macos")]
    let viewport = viewport
        .with_fullsize_content_view(true)
        .with_title_shown(false)
        .with_titlebar_buttons_shown(false)
        .with_titlebar_shown(false);
    let options = eframe::NativeOptions {
        viewport,
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
            Ok(Box::new(app::NodeCraftApp::new(
                cc,
                initial,
                screenshot,
                render_model,
            )))
        }),
    )
}

/// Headless smoke check used by CI and by the first-port acceptance test.
fn check(path: Option<PathBuf>) -> eframe::Result {
    let Some(path) = path else {
        eprintln!("usage: nodecraft-egui --check <document.nodecraft>");
        std::process::exit(2);
    };
    let document: nodecraft_session::GraphDocument = serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|error| {
            eprintln!("{}: {error}", path.display());
            std::process::exit(2);
        }),
    )
    .unwrap_or_else(|error| {
        eprintln!("{}: {error}", path.display());
        std::process::exit(2);
    });
    let (evaluation, _) = engine_bridge::evaluate(&document, nodecraft_engine::EngineCache::new(), None);
    let Some(evaluation) = evaluation else {
        eprintln!("{}: document has no target", path.display());
        std::process::exit(2);
    };
    let mesh = evaluation.mesh.as_ref();
    println!(
        "{} target={} ok={} verts={} tris={} edges={} elapsed_ms={:.1}",
        path.display(),
        evaluation.target,
        evaluation.error.is_none(),
        mesh.map_or(0, |mesh| mesh.vertices.len()),
        mesh.map_or(0, |mesh| mesh.faces.len()),
        mesh.map_or(0, |mesh| mesh.brep_edges.len()),
        evaluation.elapsed_ms
    );
    if let Some(error) = evaluation.error {
        eprintln!("{error}");
        std::process::exit(1);
    }
    Ok(())
}
