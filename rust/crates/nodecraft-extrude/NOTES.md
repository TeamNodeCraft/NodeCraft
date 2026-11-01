# Exact Extrude from a planar profile — findings and wrapper

Investigated against the pinned solidcore checkout
(`rust/Cargo.lock` → `6ecdaaa21`, `.cargo/git/checkouts/solidcore-a854643c1e2d477c/6ecdaaa/`)
on 2026-10-24.

## Verdict: yes (with one named unported symbol)

An **exact solidcore B-rep** can be produced from a planar profile with

```rust
solidcore::sweep::api_brep::create_taper_extrude(
    k, &curves, height, 0.0 /* draft deg */, 3 /* caps both */, false,
)
```

i.e. the ported upstream `SmApiCreateTaperExtrude` extrude feature at zero draft. It is
wrapped as:

* `extrude_planar_face_exact(&mut Kernel, &Brep, FaceId, distance) -> Result<Brep, ExtrudeError>`
* `extrude_profile_curves(&mut Kernel, &[Curve], distance) -> Result<Brep, ExtrudeError>`
  (straight from profile curves, e.g. a STEP loop, without an existing face)

`extrude_planar_face` now tries this route first; the older curve sweeps remain as
fallbacks/for comparison.

The one blocker is a **single closed periodic profile curve** (a full `CIRCLE`/`ELLIPSE`
as one curve): solidcore's direct path falls into its offset/merge fallback and stops at
the unported `SmMerge::SmMerge` (`boolean.merge_p2`). The exact wrapper splits such a
curve into two half-arcs and retries, which is geometrically exact (the side wall becomes
two faces at a seam). Precise symbol below.

## Evidence (pinned 6ecdaaa, `cargo test -p nodecraft-extrude`)

`listed` = `Brep::face_list()` (faces solidcore queries use); `arena` =
`Brep::faces()` (all live records). Box = `10 x 4 x 20` at the origin; top face at
`z = 20`; extrude `d = 5` unless noted.

| route | listed / arena faces | edges | vertices | area | volume | manifold |
|---|---|---|---|---|---|---|
| `create_box` baseline | 6 / 6 | 12 | 8 | 640 | 800 | yes |
| `create_taper_extrude` top face | 6 / 6 | 12 | 8 | **220** | **200** | yes |
| `create_taper_extrude` +X side face (d=2) | 6 / 6 | 12 | 8 | **256** | **160** | yes |
| `create_taper_extrude` triangle (pyramid side, d=2) | 5 / 5 | 9 | 6 | closed form | closed form | yes |
| `create_linear_sweep` capped (legacy) | 6 / 8 | 12 | 8 | 220 | 200 | yes |
| `create_linear_sweep` uncapped | 4 / 4 | 12 | 8 | 140 | 0 | no (open sheet) |
| `create_linear_sweep_with_repetitions` reps=2 uncapped | 8 / 10 | – | – | 280 | 0 | no (open sheet) |
| `create_curve_sweep_from_edges` capped (legacy) | 6 / 8 | 12 | 8 | 220.00000000000003 | 200.00000000000003 | yes |

Shape checks on the exact route for the box top face: 6 faces each with one 4-edge loop,
12 unique edge uses in loops / 24 loop-edge occurrences (every edge shared by exactly two
faces), `brep_is_manifold_solid == true`, bounds `(0,0,20)..(10,4,25)`. Extruding the top
face by `d = 20` reproduces `create_box(10,4,20)` measures exactly (640 / 800).

Distances 1 / 5 / 12.5 give `area = 2*40 + 28*d`, `volume = 40*d` to `< 1e-9`
(plane integration is exact). The triangular pyramid-side profile matches
`2*triangle_area + perimeter*d` / `triangle_area*d` to `< 1e-6`.

The legacy `create_linear_sweep` / `create_curve_sweep_from_edges` routes measure the
**same exact area and volume**, but their B-rep keeps one live face record without a face
use per cap (`arena faceuse == None`, `loops == []`) in the infinite region:
`brep_is_connected_solid` is false. The exact route has no such records.

### Periodic profiles

* Full circle (cylinder cap face, or a single `create_circle` curve), zero draft:
  `Unported: upstream SmMerge::SmMerge (unit boolean.merge_p2) is not ported yet`.
* Full ellipse (single `create_ellipse` curve): same error.
* Full circle split into **2** half-arcs (trimmed copies): exact solid; volume converges
  to `12π = 37.6991118…` (`brep_volume` accuracy 1e-6 gives 37.6991369, 1e-8 gives
  37.6991117); area `20π = 62.831853…`, measured 62.83822 because `brep_area` clamps its
  integration accuracy to `1e-4` (the geometry is exact; the numeric area query is not).
* Full ellipse split into 2: area 108.39353 vs closed form 108.39353; volume 75.39808
  vs `πab·h = 75.39822` (mass-query accuracy).
* Full circle split into **4** quarter-arcs: hits the same `SmMerge` error. The wrapper
  therefore uses 2 halves.
* Two semicircle curves and a D-shape (semicircle + chord) work directly.
* `create_taper_extrude` with a non-zero draft (5°) runs (area 200.27, volume 170.65 for
  the 10×4 rectangle) but corner/draft semantics were not validated here; the wrapper
  exposes zero draft only.
* Distance `0` returns a degenerate double-sided plate (area 80, volume 0) and a negative
  distance extrudes inward/overlapping (area 108, volume 40); the wrapper rejects
  non-positive/non-finite distances as before.

## Precise missing/unported symbols

1. **`SmMerge::SmMerge` — unit `boolean.merge_p2`.**
   Stub: `primitive::primitive_creation::model::PrimitiveModel::new_merge`
   (`crates/solidcore/src/primitive/primitive_creation/model.rs`) returns
   `SolidError::unported("SmMerge::SmMerge", "boolean.merge_p2")`; `delete_merge`,
   `merge_operands`, `piecewise_merge` and `union` are stubs around it.
   Reached from the extrude path: `primitive::primitive_creation::p2::taper::taper`
   sets `state.missing_faces` for a single closed periodic profile, then calls
   `primitive::primitive_creation::p2::taper::surface_intersection_method`, whose
   `PrimitiveCreationAccess::with_merge` → `new_merge` is unported. Porting
   `SmMerge::SmMerge` (and the merge stages it drives) is what removes the periodic
   profile limitation and the split workaround.
   Pinned port status (`plan/STATUS.md`): `sweep.api_brep` 573 cases, 514 pass,
   59 deferred; blockers include `boolean.merge_p2` (8), `surface.trim_srf_cache` (45),
   `topology.edgeuse_p3` (6).
2. **Legacy curve-sweep cap cleanup is incomplete (not an unported symbol).**
   `create_linear_sweep` → `PrimitiveCreation::create_repeated_linear_sweep` →
   `PrimitiveCreation::create_linear_sweep` creates caps through
   `planar_faces` → `topology::brep::p8::create_planar_faces_with_3d_curves` /
   `create_faces_with_curves` and stitches with `topology::brep::p5::stitch_and_orient`;
   one face record per cap is left live but unlisted (no face use). Exact measures are
   unaffected. Prefer the taper-extrude route; no separate symbol to port.

## Wrapper API and limits

* `extrude_planar_face_exact` — planar face → single boundary loop → oriented curves →
  taper extrude; rejects holed faces with `MultipleLoopsUnsupported { face, loops }`
  (upstream taper extrude consumes one chain), non-planar faces with `NotPlanar`, and
  non-positive/non-finite distances with `InvalidDistance`.
* `extrude_profile_curves` — the same route from a curve array (STEP profile loops),
  `EmptyProfile` for an empty array.
* Periodic single-curve profiles are split into two halves automatically (only when the
  direct call returns a `SolidError::Unported`); if the split also fails, the original
  unported error is returned so callers still see `SmMerge::SmMerge`.
* `extrude_planar_face` order is now exact → `create_linear_sweep` →
  `create_curve_sweep_from_edges`; the two legacy functions remain public for comparison
  and are still the only place `brep_is_connected_solid` is false.
* Not exposed/validated: non-zero draft angle, hole loops, and profiles that are not one
  closed planar chain (the caller must pass chain-ordered curves; face-derived loops are
  already in order).

## STEP export bridge (second wave)

An exact extrude result is a planar, polygonal solid, so it is directly representable by
the new `nodecraft-step` planar writers. `src/export.rs` converts the solidcore B-rep to
both neutral forms and writes the file:

| entry point | result |
|---|---|
| `planar_brep_to_step_brep(&Kernel, &Brep)` | `StepBrep` with one closed shell + one solid, ready for `nodecraft_step::write_step_from_step_brep` |
| `planar_brep_to_planar_body(&Kernel, &Brep)` | `PlanarBody` ready for `nodecraft_step::write_planar_body` (and the worker's planar import path) |
| `write_planar_brep_step(&Kernel, &Brep, path)` | AP214 `MANIFOLD_SOLID_BREP` file |
| `extrude_planar_face_exact_for_export(...)` | `ExportableExtrude { brep, step_brep, planar_body }` in one call |
| `write_exact_extrude_step(..., path)` | exact extrude + STEP write, returns the `ExportableExtrude` |

Mapping: faces are walked in `Brep::face_list()` order, each face’s bound order is kept
(outer loop first), each edge use keeps its `OrientType` as the STEP edge orientation,
vertices are welded by position (`1e-7`). The export code also verifies the B-rep is
representable: every listed face planar, every boundary edge straight (geometric
straightness test — the exact extrude route represents straight profile edges as
two-control-point B-splines, not `Line` curves), and no live face records outside the
face list (the legacy curve sweeps fail this on purpose rather than dropping faces).

Evidence (`tests/export.rs`, box top face 10×4 extruded 5 mm):

* `StepBrep`: 1 solid, 1 shell, 6 faces, 12 edges, 8 vertices, `StepBrep::validate()`
  empty; `PlanarBody`: 6 faces, 8 points, `validate_closed()` ok, volume **200**, area
  **220**, bounds `(0,0,20)..(10,4,25)`.
* written STEP re-read via `nodecraft_step::read_planar_body`: 6 faces, volume 200, area
  220, same bounds; re-read via `nodecraft_step::read_topology_file`: 1 solid / 1 shell /
  6 faces / 12 edges / 8 vertices, `validate()` empty. The re-read PlanarBody equals the
  in-memory one.
* +X side face extruded 2 mm exports volume 160 / area 256.
* triangle (pyramid side) profile exports 5 faces / 9 edges / 6 vertices with exact
  closed-form volume/area to < 1e-9.
* `planar_brep_to_planar_body` equals the analogous `nodecraft_solidcore::planar::planar_body`
  bridge (same faces/points; same volume/area).
* rejected with `ExtrudeError::ExportUnsupported`: a cylinder (curved faces), an exact
  extrude of a circular profile (curved side faces), and a legacy capped linear sweep
  (live face records outside the face list).

### Reported limits of the planar writers

* line-bounded **planar** faces only; curved surfaces/edges are rejected, never faceted
  (curved profile extrusions need an analytic/NURBS STEP writer);
* one solid per file, no `BREP_WITH_VOIDS` (PlanarBody cannot represent voids);
* output is in millimetres; every face is re-emitted as `PLANE` and every edge as `LINE`,
  so original STEP surface/curve ids and names are not preserved.

### Full feature-chain export: exact worker API gap (not fixed here)

The crate only owns the B-rep side. Full feature-chain export (feature node → STEP file)
needs a worker feature node that:

1. evaluates the feature to a solidcore `Brep`
   (`nodecraft_extrude::extrude_planar_face_exact` for an extrude, or a `Brep` produced by
   the STEP→solidcore adapter), then
2. calls `nodecraft_extrude::write_exact_extrude_step(&mut kernel, &source_brep, face,
   distance, path)` (or `write_planar_brep_step(&kernel, &brep, path)`), and
3. returns the written path (and optionally `ExportableExtrude` measures) through the
   protocol response.

No worker/app/dashboard files were edited in this wave; that node + protocol field is
the remaining integration.

## Tests

`cargo test -p nodecraft-extrude` (25 tests + 1 doctest):

* `tests/exact_extrude.rs` (11 tests): exact box-top and box-side solids (topology
  counts, shared edges, bounds, exact area/volume), distances 1/5/12.5, triangle
  pyramid-side profile, exact route vs `create_box` measures, curve-array route vs face
  route, circular cap split fallback (volume vs `12π`), the pinned `SmMerge::SmMerge`
  evidence test (calls solidcore directly and asserts the unported unit), invalid-input
  rejection, `extrude_planar_face` preferring the exact route, and the legacy
  linear-sweep "exact measures but not connected" comparison.
* `tests/export.rs` (5 tests): StepBrep + PlanarBody export forms, STEP file write and
  re-read through both readers, generic `write_planar_brep_step`, triangle-profile export,
  and curved/unwired export rejections.
* `src/lib.rs` unit tests (9 + 1 ignored): face plane/boundary helpers, preview mesh,
  legacy sweep route behavior.
* The existing ignored test still pins the legacy curve sweeps' unwired cap faces.
