# NodeCraft

NodeCraft is a native node-based CAD editor built with Rust, egui and wgpu.
It uses SolidCore/SMLib B-reps for exact geometry and a node canvas for
constructive modelling.

## What's new

- Inspector-driven topology selection for chamfer, fillet and shell features.
- Exact chamfer evaluation, including chained chamfer operations.
- Feature comparison overlay showing removed and added material around a
  selected operation; display-only and excluded from picking.
- Viewer breakpoint controls with explicit downstream stepping.
- Auto-placement and fade-in for tool-dock nodes, with cable-aware spacing.
- Persisted editing-overlay setting under Viewport preferences.

## Requirements

- Rust toolchain (stable)
- macOS 13+ for the `.app` bundle helper
- Linux: X11/Wayland development packages required by eframe/winit
- Windows: a standard Rust MSVC toolchain

The cargo binary builds on macOS, Linux and Windows; the `.app` helper is
macOS-only.

## Build

```sh
cargo build --release -p nodecraft-egui
```

Build a macOS app bundle:

```sh
./scripts/build-nodecraft-egui-app.sh
open build/nodecraft-egui.app
```

## Run

```sh
cargo run --release -p nodecraft-egui -- samples/spline-showcase.nodecraft
```

## Tests

```sh
cargo test -p nodecraft-egui
cargo test -p nodecraft-node-canvas
cargo test -p nodecraft-engine --lib
```

Some integration tests in the private development tree use large STEP
corpora that are not included in this public source snapshot.

## Repository layout

```text
rust/apps/nodecraft-egui   desktop application
rust/crates/               engine, node canvas, geometry, I/O crates
vendor/fluent              icon assets
samples/                   small runnable example documents
agents/                    engineering notes
docs/                      project documentation
```

Private planning, research, reports and reference code are intentionally not
included.

## Documentation

- [Architecture](agents/architecture.md)
- [egui model/view](docs/egui-model-view.md)
- [Node plugin contract](docs/node-plugin-contract.md)
- [Agent guide](agents/README.md)
- [Website](https://nodecraft.com.au)

## License

NodeCraft is dual-licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
