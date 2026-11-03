# nodecraft-bool: exact Boolean/Shell feature B-rep APIs — evidence report

Status: **the public solidcore APIs are callable but no general exact Boolean/Shell
feature path exists yet.** The B-rep wrappers in `src/lib.rs` compile against the pinned
solidcore revision and expose the operations; their tests pin both the partial
capability and every current blocker. `cargo test -p nodecraft-bool` → **18 passed, 0
failed, no warnings**.

This file supersedes the earlier blocker notes. Three earlier "hard blocker" claims were
stale/imprecise and are corrected below.

## Revisions checked

| Source | Revision | Notes |
|--------|----------|-------|
| `Cargo.lock` dependency (compiled) | `6ecdaaa` (2026-10-24 01:26Z, `main`) | cargo checkout under `.cargo/git/checkouts/solidcore-a854643c1e2d477c/6ecdaaa` |
| Latest `main` read-only checkout | `0ffc493` (2026-10-24 02:00Z) | `/tmp/solidcore-check`; one merge later |

`diff -rq` between the two checkouts: only 5 files differ, all
`blend/fillet_*` plus `topology/face/p2.rs`. Every file involved in the findings below
(`boolean/*`, `offset/*`, `topology/edge.rs`, `topology/brep/p3.rs`,
`topology/brep/p4.rs`, `topology/face/p3.rs`, `intersect/point_class.rs`,
`surface/surface_cache/*`, `surface/trim_srf_cache.rs`) is **byte-identical**, so all
blockers are current on `main` as well as on the pinned rev.

Method: `cargo test` probes in this crate (public APIs only) plus `lldb` line
breakpoints in the debug test binary to capture exact call chains. All probe outputs
below are reproduced by the tests listed in "Evidence map".

## Public API inventory (all public, all callable)

Boolean (`solidcore::boolean::api_brep`, `SmApiBoolean*`): `boolean`,
`boolean_union`, `boolean_difference`, `boolean_intersection`, `boolean_merge`,
`boolean_with_options`, `merge_breps`, `boolean_lists`, `boolean_trees`; lower level
`boolean::merge::p3::{DependencyMerge, BooleanDriver, piecewise_merge, ...}` and
`boolean::merge::p4::{manifold_boolean, KernelManifoldIntersector, ...}`.

Shell/offset (`solidcore::offset::offset_executive::p2`, `SmOffsetExecutive` /
`SmApiBrep`): `OffsetExecutive::shell_brep`, `offset_brep`,
`api_shell_brep_full`, `api_offset_brep_full`, plus the p1 executive methods
(`do_solid_offset`, `do_extended_offset`, `build_offset_of_edgeuses`, ...).

The wrappers here use exactly `boolean::api_brep::boolean` and
`OffsetExecutive::shell_brep`.

## What works today (partial Boolean only)

Two disjoint `10³` solidcore boxes (`box_brep_at([0,0,0])` and `box_brep_at([20,0,0])`):

| Operation | Result | Volume | Faces | Evidence |
|-----------|--------|--------|-------|----------|
| union | Ok(Brep), bounds `(0,0,0)..(30,10,10)`, `brep_is_manifold_solid` true, displayable | `2000.0` (correct) | 12 | `boolean_disjoint_union_and_difference_are_exact` |
| difference | Ok(Brep), bounds `(0,0,0)..(10,10,10)` | `1000.0` (correct) | 6 | same test |
| intersection | Ok(Brep) but **invalid topology**: 0 shells, `brep_is_manifold_solid` false, no display mesh | `0.0` (geometrically empty) | 6 unlinked | `boolean_disjoint_intersection_is_empty_but_invalid_topology` |

So the only trustworthy feature today is **combining disjoint bodies via union** (and
difference when the second body is disjoint, which returns the first unchanged). The
empty intersection is represented incorrectly and must not be consumed as an empty solid.

Two nested boxes (`10³` outer, `4³` inner at `[2,2,2]`) — no surface intersections,
but the merge does not classify containment:

| Operation | Result volume | Expected | Faces | Evidence |
|-----------|---------------|----------|-------|----------|
| union | `1064.0` | `1000.0` | 12 (inner box faces retained) | `boolean_nested_operands_are_misclassified` |
| difference | `1000.0` | `936.0` | 6 (cavity faces dropped) | same test |
| intersection | `0.0` | `64.0` | empty/invalid | same test |

Conclusion: do not use `boolean_breps` as a general exact Boolean feature yet. Its
`Ok` results are only trustworthy for genuinely disjoint operands.

## Boolean blocker 1 — intersecting operands: `SmEdge::GetTolerance`

Overlapping boxes (`[0,10]³` vs `[5,15]×[0,10]²`) fail for union, difference and
intersection with:

```
solidcore unported: solidcore boolean <op>: SmEdge::GetTolerance (topology.edge) is not ported
```

`lldb` backtrace from `DependencyMerge::manifold_boolean` (exact chain, pinned rev):

```
boolean::merge::p4::manifold_boolean                        boolean/merge/p4.rs:1421
→ ManifoldKernelAdapter::intersect_insert_relate            boolean/merge/p4/kernel_adapter.rs:742
→ TopologyIntersector::intersect_insert_relate              boolean/topology_intersector/p2.rs:1213
→ TopologyIntersector::intersect_topology                   boolean/topology_intersector/p2.rs:1346
→ face_face_intersect_curves                                boolean/topology_intersector/p1.rs:726
→ ArenaPair::face_intersect                                 boolean/merge/p4/arena_pair.rs:912
→ topology::face::p3::face_intersect_breps                  topology/face/p3.rs
→ query::face::p2::curve_on_classify[_cached]                query/face/p2.rs:1284/1391
→ query::face::p2::intersect_edges                          query/face/p2.rs:1541
→ CurveClassification::intersect_edges                      intersect/curve_class/p9.rs:274
→ curve_class::p5::insert_solutions[_in]                    intersect/curve_class/p5.rs:1221/1281
→ point_class::object_zone_tol_3d (ClassObjectKind::Edge)   intersect/point_class.rs:1693
→ topology_of fallback                                      intersect/point_class.rs:1944
      .ok_or(SolidError::unported("SmEdge::GetTolerance", "topology.edge"))
```

**Correction (stale claim):** `SmEdge::GetTolerance` *is* ported —
`topology::edge::tolerance` returns `m_sZoneTol3d` (`topology/edge.rs:335`). The error is
the generic `topology_of` fallback: the classification adapter could not resolve an
`IntersectObject::Topology(Edge(id), side)` handle to its `(Brep, EntityRef::Edge)`.

**Exact missing implementation:** `impl ClassGeometry<IntersectObject> for
IntersectObjects<'_>` at `topology/face/p3.rs:1484` implements `curve`, `surface`,
`volume`, `point`, `classification_vertex_point` and `topology_zone_tol_3d` (Vertex and
Face arms only) but **not** `topology_entity` / `topology_handle`, whose trait defaults
return `None` (`intersect/point_class.rs:343-352`). An edge classification object
(`IntersectObjects::edge()` → `Topology(EntityRef::Edge(e), side)`,
`topology/face/p3.rs:1447`) therefore cannot be resolved. Fix candidates (kernel-side,
~20 lines):
1. implement `topology_entity`/`topology_handle` for `IntersectObjects` (map
   `Topology(e, side)` → `(self.brep(side), e)`, and the curve/edgeuse objects to their
   owning edge/edgeuse);
2. and/or add `Edge`/`Edgeuse` arms to `topology_zone_tol_3d` that answer from
   `brep(side).edge(e).tolerance` directly.

Nothing here needs a new algorithm port; it is adapter plumbing.

## Boolean blocker 2 — touching/coincident faces: face-owned trimmed surface cache

Touching boxes that share the `x=10` plane (`[0,10]³` and `[10,20]×[0,10]²`, all three
ops) fail with a *different* unported dependency:

```
solidcore unported: solidcore boolean <op>: TrimSrfCache::build_tree_with_face for a face-owned surface (surface.trim_srf_cache) is not ported
```

`lldb` chain:

```
face_intersect_breps                          topology/face/p3.rs
→ Surface::global_surface_intersect           surface/surface/p2.rs
→ base_global_surface_intersect_in            surface/surface/p2.rs:3189
→ surface_intersector::p5::do_intersection    intersect/surface_intersector/p5.rs:351
→ intersection_surface_caches[_in]            intersect/surface_intersector/p5.rs:115/157
→ SurfaceCache::get_tree                      surface/surface_cache/p3.rs:1237
   (self.trimmed_dispatch && self.surface().face()?.is_some())
→ Err(Unported("TrimSrfCache::build_tree_with_face for a face-owned surface",
               "surface.trim_srf_cache"))
```

A mirror guard exists in `TrimSrfCache::build_tree_body`
(`surface/trim_srf_cache.rs`, around lines 625–634). The trimmed cache cannot follow a
face-owned surface's `GetFace()` owner to the B-rep because no `CacheFace` handle
adapter is supplied on this path; it therefore refuses to build rather than silently
treat the face as an untrimmed rectangle. The fix is to pass the caller's
`CacheFace`/face handle through `face_intersect`'s coincidence path
(`TrimSrfCache::build_tree_with_face`), not to port a new algorithm.

## Shell/offset support (SolidCore a9ef4c70)

The earlier "every shell path is unported" claim is obsolete.

- `shell_brep` (rounded, `create_offset_solid = false`) completes and returns an
  offset shell.
- `shell_brep_solid` (extended edges, `create_offset_solid = true`) completes and
  returns a **closed manifold hollow solid**, including with opening faces.
  Covered by `shell_solid_is_a_closed_manifold_with_openings`.
- `shell_brep_with_options` exposes both modes.

`offset_brep`, `api_shell_brep_full` and `api_offset_brep_full` have not been
re-verified in this revision.

## Corrections to prior claims (summary)

1. **"Boolean blocked because `SmEdge::GetTolerance` (`topology.edge`) is unported."**
   Imprecise. `topology::edge::tolerance` is ported; the error is the generic
   `topology_of` fallback for an `IntersectObject` edge handle that
   `IntersectObjects` (topology.face.p3) does not resolve. Fix: adapter
   `topology_entity`/`topology_handle` (and edge arms in `topology_zone_tol_3d`).
2. **"Shell blocked because `SmBrep::ShrinkGeometry` (`topology.brep_p4`) is
   unported."** Wrong. `shrink_geometry` is ported and succeeds; the error is a
   placeholder for the unported `SmOffsetExecutive::DoSolidOffset` continuation.
3. **"`SmApiShellBrepFull` blocked because the `SmBrep` copy constructor is
   unported."** Wrong as of this revision. `topology::brep::p3::copy_brep` is ported;
   the API stub is stale. It remains blocked only by the `DoSolidOffset` continuation.

Additionally, the earlier assumption that boolean *intersection* of disjoint solids is
usable should not be made: it returns an empty volume but an invalid B-rep (0 shells).

## Evidence map (tests that pin each finding)

| Finding | Test |
|---------|------|
| disjoint union/difference exact (volumes 2000/1000) | `boolean_disjoint_union_and_difference_are_exact` |
| disjoint intersection empty but invalid | `boolean_disjoint_intersection_is_empty_but_invalid_topology` |
| nested containment misclassified (1064/1000/0) | `boolean_nested_operands_are_misclassified` |
| overlapping → edge classification adapter (`SmEdge::GetTolerance`) | `boolean_overlapping_boxes_hit_the_edge_classification_adapter` |
| construction OK, failure inside manifold intersection | `boolean_construction_succeeds_before_the_manifold_intersection_fails` |
| touching → face-owned trimmed surface cache | `boolean_touching_boxes_hit_the_face_owned_surface_cache` |
| rounded shell placeholder marker | `shell_box_reports_the_known_unported_shrink_geometry` |
| `ShrinkGeometry` ported despite marker | `shrink_geometry_symbol_is_ported_despite_the_shell_marker` |
| extended shell → `BuildOffsetOfEdgeuses` adapter | `extended_shell_hits_the_build_offset_of_edgeuses_adapter` |
| sub-tolerance shell distance → 1007, not Unported | `shell_zero_thickness_is_invalid_input_not_an_unported_dependency` |
| B-rep copy constructor ported despite API stub | `brep_copy_constructor_is_ported_despite_the_api_shell_stub` |
| mesh bridge on a box (bounds, B-rep coords) | `display_mesh_of_a_box_is_in_brep_coordinates` |

## Wrapper API map

| Item | Returns | Notes |
|------|---------|-------|
| `BooleanOp` | — | Union / Difference / Intersection → `BooleanOperation` |
| `boolean_breps(op, a, b)` | `Result<Brep, String>` | `api_brep::boolean`; consumes operands |
| `boolean_mesh(op, a, b)` | `Result<Mesh, String>` | boolean + `display_mesh` |
| `shell_brep(brep, thickness, openings)` | `Result<Brep, String>` | rounded shell, default `ShellOptions` |
| `shell_brep_solid(brep, thickness, openings)` | `Result<Brep, String>` | sharp shell using extended convex edges + merged results |
| `shell_brep_with_options(brep, options, openings)` | `Result<Brep, String>` | reaches extended path / side walls |
| `shell_mesh(brep, thickness, openings)` | `Result<Mesh, String>` | shell + `display_mesh` |
| `display_mesh(&Brep)` / `display_mesh_with_options` | `Result<Mesh, String>` | generic B-rep → protocol mesh bridge (not centered) |
| `box_brep` / `box_brep_at` | `Result<Brep, String>` | operand constructor (B-rep coordinates) |
| `brep_faces(&Brep)` | `Result<Vec<FaceId>, String>` | shell `openings` ids |
| `map_error` / `is_unported_error` / `UNPORTED_PREFIX` | — | unported dependency vs invalid input |

## Shell status and known issues

`nodecraft_bool::shell_brep_solid` is the current feature-node shell entry point.
It uses:

```text
extend_convex_edges = true
merge_results       = true
create_offset_solid = false
merge_sheet_side_walls = false
```

Current observations:

| Input | Result |
|-------|--------|
| box, inward/outward | works; sharp inner and outer walls; openings resolve by `FaceId` |
| cylinder | completes but slow (~1.2 s in the probe) and needs volume/orientation checks |
| sphere | completes but slow (~0.3 s in the probe) and needs volume/orientation checks |
| cone | fails display tessellation: `solidcore curved tessellation produced no triangles` |

Open concerns:

- inner-shell orientation is not yet validated against SolidCore mass properties
- curved shell results need volume/area and watertightness checks
- `shell_brep` and `shell_brep_with_options` still expose rounded-shell defaults;
  feature nodes should use `shell_brep_solid`
- offset options are internal policy, not inspector controls
- shell opening selection is ordinal/body-scoped; persistent topological naming
  across upstream edits remains outstanding
- the contextual inspector selection plan is documented privately and is not
  part of this public snapshot

## Next steps when solidcore lands the missing pieces

1. `topology.face_p3`: implement the `IntersectObjects` `ClassGeometry` topology
   adapter (or edge arms in `topology_zone_tol_3d`); then re-run the overlap tests —
   the pinned `SmEdge::GetTolerance` assertions will fail and should be replaced by
   bounds/volume checks (a success branch already exists in the disjoint/nested tests).
2. `surface.trim_srf_cache` / `surface.surface_cache_p3`: supply the `CacheFace`
   adapter on the coincident-face path; then drop the `TrimSrfCache` assertions in
   `boolean_touching_boxes_hit_the_face_owned_surface_cache`.
3. `offset.offset_executive_p2`: port the `DoSolidOffset` continuation (cpp:648–1051)
   and remove the placeholder `SmBrep::ShrinkGeometry` marker; wire
   `api_shell_brep_full` to `copy_brep`. `offset.offset_executive_p1`: port the
   `BuildOffsetOfEdgeuses` geometry-creation adapter and `SmMerge::PiecewiseMerge`.
4. Only then evaluate whether `boolean_breps`/`shell_brep` can be promoted as exact
   feature nodes in the worker; validate results with `brep_volume`/`brep_area` and the
   STEP writer/reader round-trip.

## Reproduce

```sh
export RUSTUP_HOME="$PWD/.rustup" CARGO_HOME="$PWD/.cargo" PATH="$PWD/.cargo/bin:$PATH"
cd rust && cargo test -p nodecraft-bool
```

Call-chain evidence (optional, macOS lldb; debug test binary):

```sh
BIN=$(ls -t target/debug/deps/nodecraft_bool-* | grep -v '\.' | head -1)
SC=../.cargo/git/checkouts/solidcore-a854643c1e2d477c/6ecdaaa/crates/solidcore/src
lldb -b -o "breakpoint set -f $SC/intersect/point_class.rs -l 1693" \
        -o 'run <test-name> --nocapture' -o 'bt 25' -o 'quit' "$BIN"
```

(The lldb probe scenario is exercised by
`boolean_overlapping_boxes_hit_the_edge_classification_adapter`; run that test name in a
temporary copy to reproduce the backtrace.)

Note: `cargo clippy` is not installed in the workspace-local toolchain.
