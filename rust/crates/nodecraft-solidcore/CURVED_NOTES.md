# Curved display tessellation — status and remaining blocker

`src/curved.rs` produces display meshes for the curved solidcore primitives
(cylinder, cone, sphere, torus) from public solidcore APIs only. It is wired into
`lib.rs` as `pub mod curved;` and is covered by 8 unit tests.

## What works today

| entry point | result |
|---|---|
| `curved::cylinder_mesh(r, h)` | centered mesh, 3 face groups: lateral grid + two planar-cap fans |
| `curved::cone_mesh(rb, rt, h)` | same, lateral grid ramps between the radii |
| `curved::sphere_mesh(r)` | one closed face, UV grid on the sphere face |
| `curved::torus_mesh(r, R)` | one closed face, UV grid on the torus face |
| `curved::brep_display_mesh(&kernel, &brep, &opts)` | any B-rep, raw coordinates, face groups `face:{index}` |
| `*_with_options` variants | explicit `u_segments` / `v_segments` / `loop_segments` |

Each mesh has `vertices`, unit `normals`, `faces`, one `face_groups` entry per
triangle, `brep_edges` (boundary-loop polylines referencing valid vertices) and
`lightweight = true`, matching `box_mesh`.

## How it works (the cheap path)

Per face:

1. `topology::face::p2::uv_domain` → the face's parameter rectangle.
2. `topology::edgeuse::p1::faceuse_loopuses` + `loopuse_edgeuses` →
   `edgeuse::p1::uv_trim_curve`, sampled with
   `query::face::p1::evaluate_curve` and reversed when the edge use is
   `OrientType::Opposite`. This reconstructs the trim loop in UV space.
3. `Surface::evaluate_normal(...)` at five domain samples decides **planar vs
   curved** (no `SurfaceKind` matching, so trim-plane B-spline faces work too).
4. Planar faces: `face.faceuse` loop edges are sampled in 3D (`evaluate_curve`,
   parameter span shared across the loop) and fan-triangulated around the loop
   centroid.
5. Curved faces: the UV grid is sampled with `Surface::evaluate_point` /
   `evaluate_normal`. If every trim sample lies on the domain boundary — the case
   for all four primitive curved faces, whose only trims are seams and end circles
   — the whole rectangle is filled. Otherwise a conservative even-odd
   point-in-trim-polygon test keeps grid cells whose four corners are inside.
6. Winding is oriented against the face's outward normal (surface normal flipped
   when the primary face use is `OrientType::Opposite`), and degenerate triangles
   (sphere poles, seam duplicates) are dropped.

## Exact blocker for a general tessellator

This is a *display* path, not the real `SmApiTessellate` pipeline:

- `SmApiTessellate`, `SmApiTessellateBoundaries` and `SmApiPoly*` are **out of
  solidcore scope** (see `plan/closure.md`, "Out of scope"), so they cannot be called.
- `query::tess::TessSrfCache` is ported, but its constructor needs
  `poly_brep: Option<PolyBrep>` and the trim/implant steps need a poly B-rep built
  from the B-rep (`SmPolyBrep::Build` in `SmPolyBrep.cpp`). That converter is **not
  ported** and is not reachable from an in-scope root; the only `PolyBrep`s today are
  built vertex-by-vertex by tests/ray-tracing helpers, and `TessSrfCache::new` also
  wants to take ownership of a `Surface`.
- Consequently there is no tolerance-driven, silhouette-aware, exactly-trimmed
  tessellation available. Its absence is the only reason for the approximations below.

### Next API calls when that gap closes

1. Build a `PolyBrep` for the face (upstream `SmPolyBrep::Build`) from the face's UV
   trim curves — the exact same `uv_trim_curve` data used here.
2. `query::tess::TessSrfCache::new(face_key, surface, Some(poly_brep),
   Some(SurfaceTessDriver::new(...)), SurfaceCacheOptions { chord_height_tolerance,
   ang_tol_rad, .. })`, then `build_tree_with_subdivision()` and
   `mark_face_containment()`.
3. Walk the cache tree leaves (`Inside`/`Outside`/`OnBoundary`) and emit the
   `Inside`/`OnBoundary` surfaces as triangles, using the cache's UV nodes instead of
   a uniform grid.
4. Or, upstream-equivalent: implement `SmApiTessellate` as an in-scope root in
   solidcore and call it from `brep_display_mesh`.

A cheaper intermediate: `surface_cache` subdivision over a `Surface` we construct
with `primitive::api_primitives::create_*_surface` (no poly B-rep needed) would give
tolerance-driven untrimmed grids — an incremental improvement on the fixed grid here,
but still no exact trim boundary.

## Known limitations (accepted for this slice)

- **Trimmed curved faces are approximate**: only cells whose four corners are inside
  survive, so the mesh boundary is one grid cell inside the true trim. No primitive in
  this slice hits that path (their curved faces are untrimmed).
- **Planar faces** use a centroid fan: correct for convex outlines (box, primitive
  caps); non-convex or multi-loop planar faces would need ear clipping and hole
  bridging instead.
- **Seams are not welded**: closed surfaces duplicate the `u = u_min` / `u = u_max`
  rows. Display-only; no watertight index buffer.
- Normals are sampled per grid node; no smoothing/crease logic beyond that.

## Integration

`nodecraft-worker` currently dispatches `"primitive" | "box" | "cylinder"` and falls
back to `nodecraft-kernel`'s procedural cylinder for `cylinder`. To use the B-rep
path, call `nodecraft_solidcore::curved::cylinder_mesh(...)` (or the other wrappers)
from the `primitive` arm in `rust/crates/nodecraft-worker/src/main.rs`.
