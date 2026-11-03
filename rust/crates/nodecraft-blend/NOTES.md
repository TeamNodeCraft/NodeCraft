# nodecraft-blend: NodeCraft Edge node (chamfer + fillet) over solidcore

Status: **compiles, 24 tests + 1 doctest pass; the exact B-rep path is blocked at the
solidcore port, and a working mesh preview covers the node today.**

`cargo test -p nodecraft-blend` → 5 (edge selection) + 5 (exact path) + 7 (node values)
+ 7 (preview) + 1 doctest, 0 failed, no warnings.

solidcore revision: git `main`, `Cargo.lock` rev `6ecdaaa`.

> **2026-11 status — exact BRep fillet is live.** SolidCore is now at
> `fb937dd8d991072cf5b6fa7d16cf46ea110ea238`. The shell path uses upstream
> `create_offset_solid` and returns a closed manifold hollow solid, and
> `solidcore::blend::fillet_executive::api::fillet_edges` completes for a box.
>
> The exact fillet integration copies the input with SolidCore's owning
> `brep::p3::brep_copy()` before calling the kernel. It replaced the former
> `clone_standing_for_self()` workaround; the owning copy is safe to edit and
> does not share geometry owners with the input.
>
> `fillet_brep_exact` now routes through that API, and the engine evaluates
> fillet as a Part. Mesh preview remains display-only.

## API

| Item | Purpose |
|---|---|
| `box_brep(length, width, height)` | solidcore box at the origin plus its `Kernel` (same convention as `nodecraft_solidcore::box_mesh`). |
| `edge::edge_selections(&Kernel, &Brep)` | all edges in `SmBrep::GetEdges` order with exact start/end/midpoint/length. |
| `edge::select_edge_by_midpoint(&Kernel, &Brep, [f64; 3])` | nearest edge to an approximate midpoint (B-rep coordinates); rejects picks farther than 25% of the B-rep diagonal from every edge. |
| `edge::select_edges_by_midpoints`, `select_edges_by_ordinals` | multi-select; duplicates collapse; out-of-range ordinals are errors. |
| `edge::select_edges_from_values(&Kernel, &Brep, &Value)` | resolves the node's `values.edges` references (`center`/`point` in centred mesh coordinates, else `ordinal`). |
| `treatment::EdgeTreatment::{from_node_values, validate, chamfer_distances}` | the node parameter model (`mode`, `chamferType`, `continuity`, `length`, `distance2`, `angle`) with strict parsing. |
| `preview::apply_edge_treatment(&Kernel, &Brep, &[EdgeSelection], &EdgeTreatment)` | **working** watertight mesh preview in centred coordinates. |
| `preview::edge_treatment_preview(&Kernel, &Brep, &Value)` | one-call worker entry point: parse values + select edges + preview. |
| `exact::apply_edge_treatment_brep`, `chamfer_brep`, `fillet_brep` | the exact B-rep signature; returns `BlendError::Unsupported` naming the missing calls/units. |
| `exact::missing_calls(mode, continuity)`, `port_units()` | the blocker lists, verbatim upstream names. |
| `BlendError::{InvalidInput, Unsupported{feature, reason, missing}, Kernel}` | every failure is a value; the library has `#![forbid(unsafe_code)]` and no `unwrap`. |

The worker integration is one function: `edge_treatment_preview(&kernel, &brep,
&node.values)` → `nodecraft_protocol::Mesh`. `values` is exactly the Edge node's
`values` object, so no new protocol work is needed.

## What works today (verified by tests)

* **Edge selection** by approximate midpoint and by ordinal on a solidcore box; 12 edges,
  lengths `{40 × 4, 30 × 4, 20 × 4}`; re-selecting with a jittered pick returns the same
  edge (`tests/edge_selection.rs`).
* **Chamfer, equal distance**: 45° plane at `length/√2` from the edge; volume
  `24000 − 0.5·d²·L`; watertight (`tests/preview.rs`).
* **Chamfer, two-distance**: asymmetric cut-back with distances 4 and 1 on the two faces;
  plane distance `d1·d2/√(d1²+d2²)` (`tests/preview.rs`).
* **Chamfer, distance-angle**: plane distance `d1·sin(angle)` (θ = 30° → 1.0;
  45° reproduces the equal-distance chamfer) (`tests/preview.rs`).
* **Fillet, circular (G1)**: built as `FILLET_SEGMENTS = 8` tangent planes along the arc
  (tangent, so the facet never gouges), watertight, volume below the box and above the
  ideal circular fillet (`tests/preview.rs`).
* **Two non-adjacent parallel edges at once**; volume adds up, watertight.
* **Node values end to end** (`tests/node_values.rs`): ordinal references and
  centred-`center` references both select the right edge; defaults match the app.

The preview builds its own corner-only planar display mesh from the B-rep boundary loops.
`brep_display_mesh` samples each face boundary independently, which leaves T-junctions
between adjacent faces (288 welded edges appear once on an untreated box), so it cannot
be clipped into a verifiably watertight result. Restricting the preview to planar-faced
solids also gives an early, explicit rejection of curved bodies.

## The exact path: blocked, and exactly where

No SM_API edge-treatment entry point is ported at rev `6ecdaaa`:

* `plan/closure.md` lists `SmApiChamferFillet`, `SmApiCircularFillet`,
  `SmApiFilletEdges`, `SmApiFilletEdgesPerEdge`, `SmApiSetBevelCorners`,
  `SmApiSurfaceSurfaceFillet`, `SmApiVariableRadiusFillet` as the fillet roots.
* `plan/PROGRESS.md` marks every `blend.fillet_*` unit as partial or in progress;
  `plan/units/blend.fillet_executive_p1.toml` ends with *"Not ported yet (…):
  DoPiecewiseMerge, TestAndConvertBigRadFillet, SetFilletParameters, RemoveTopoEdges,
  LoadFilletSolver, CreateSmoothEdgeSequence, UpdateTopologyChanges, CreateFilletCorners,
  ~SmFilletExecutive and the SmApi*Fillet entry points."*
* `crates/solidcore/src/blend/` has no `api_fillets`/SM_API module at all. The ported
  `blend::fillet_executive::p3::FilletExecutive` is driven only through the
  `ExecutiveAccess` trait; its only implementation is in solidcore's private
  `blend/fillet_executive/p3_tests.rs` (`#[cfg(test)]`), and several trait methods return
  `SolidError::Unported` (e.g. `corner_solvers` → `SmFilletCorner::GetFilletSolvers`).
  `FilletExecutive::do_fillet(kernel, access)` cannot be called from outside the crate.

Upstream signatures (from `harness/callgraph/units.json`, upstream commit `c9c5979`):

```
SmApiFilletEdges(SmBrep*, SmTArray<SmEdge*>*, double, long, long, double)
SmApiFilletEdgesPerEdge(SmBrep*, SmTArray<SmEdge*>*, SmTArray<double>*, long, double)
SmApiChamferFillet(SmBrep*, SmTArray<SmEdge*>*, double, long, long, double)
SmApiSetBevelCorners(SmBrep*, SmTArray<SmVertex*>*, long)
SmApiVariableRadiusFillet(...)
```

Two tempting shortcuts do **not** work at this revision:

1. **Boolean cut** (chamfer = difference with a half-space prism): the Boolean wrapper is
   itself runtime-blocked (`nodecraft-bool/NOTES.md`:
   `solidcore unported: … SmEdge::GetTolerance (topology.edge)`), so a chamfer-through-
   Boolean is blocked one level lower.
2. **`api_primitives::create_blend_primitive`** (the only ported blend primitive call)
   documents `SolidError::Unported` (`SmBrep::CreateFaceFromSurface`), and it adds a
   blend face between two boundary edges — it does not trim the adjacent faces, so it is
   not an edge treatment even if it ran.

## Shortest path (recommended order)

1. **User-visible now:** wire `edge_treatment_preview` into the worker's `chamfer` node
   case (worker is owned by the lead). Validation covers equal/two-distance/
   distance-angle chamfer and G1 fillet; G2/G3, curved faces and non-90° dihedrals return
   `BlendError::Unsupported` with the missing API named, which the node can surface as its
   error text.
2. **Exact B-rep, in this order** (each step has difftest fixtures in solidcore):
   1. `blend.fillet_executive_p1` — `SetFilletParameters`, `CreateFilletCorners`,
      `DoFilleting`, `DoPiecewiseMerge`, plus the seven `SmApi*Fillet*` entry points.
   2. `ExecutiveAccess` adapter over `Brep`/`Kernel`: map `SmFilletEdgeuse` records to
      B-rep edge-use ids (the open identity question noted in the unit's TOML).
   3. `blend.fillet_standard_solver_p2` (constant-radius solvers) and
      `blend.fillet_corner_p1/p2` (corner patches).
   4. `blend.fillet_geom_p3/p5/p6` (fillet surface, rails, trims).
   5. G2/G3: `SmBSplineFilletLaw` + the executive law path
      (`blend.fillet_executive_p1`/`p2`, `blend.fillet_standard_solver_p2`).
3. Replace `preview::apply_edge_treatment` with `exact::apply_edge_treatment_brep` +
   `nodecraft_solidcore::brep_display_mesh`. The node parameter model, edge selection and
   tests stay unchanged, so this is a geometry-only swap.

Estimated effort to an exact box chamfer: `fillet_executive_p1` + the
`ExecutiveAccess` adapter + one solver/corner/geom slice. Everything above it
(SM_API plumbing, node wiring, selection) already exists here.

## Preview limitations (explicit errors, never wrong geometry)

* planar faces only, 90° dihedral, straight edges, convex body (all box edges; a
  cylinder rim returns `Unsupported`);
* at most one treatment per edge; two edges sharing a corner are not merged (the second
  cut may consume the first cap — currently it errors rather than producing a wrong
  body);
* distance-angle limited to `0 < angle < 90` (convex chamfer);
* G2/G3 fillets are rejected (`SmBSplineFilletLaw` missing);
* output is a *preview mesh*: no `brep_edges`, so downstream Edge-node face/edge
  selection and Boolean consumers still need the exact path;
* fillet is a faceted tangent-plane approximation (8 facets), not exact NURBS.

## Coordinate conventions

* B-rep coordinates are the solidcore coordinates (`create_box` at the origin).
* `apply_edge_treatment` / `edge_treatment_preview` return meshes in NodeCraft's centred
  primitive coordinates (B-rep minus the tight bbox centre), matching `box_mesh` /
  `curved` primitives, so the viewer and existing offset selection references line up.
* Node `edges[].center`/`point` are centred mesh coordinates; `select_edges_from_values`
  adds the bbox centre before sampling the exact edge. `ordinal` is interpreted as the
  index in `SmBrep::GetEdges` order.
