# nodecraft-step

Rust STEP (ISO-10303-21 / AP214) reader/writer modeled on smlibpy's
`step_reader.py` and `step.py`.

## Implemented

- `PlanarBody` neutral planar B-rep model: shared points, one outer loop per face
  (`faces`) plus optional inner loops (`holes`, index-aligned; `push_face`,
  `face_holes`, `set_face_holes`). `bounds`, `area`/`try_area` and
  `volume`/`try_volume` account for holes.
- Planar STEP writer (`write_planar_body`) with shared edges and loop orientation:
  `FACE_OUTER_BOUND` for the outer loop and one `FACE_BOUND` per hole, in stored order.
- Planar STEP reader (`read_planar_body`) reconstructing faces from
  VERTEX_POINT / EDGE_CURVE / ORIENTED_EDGE / EDGE_LOOP / ADVANCED_FACE; it groups
  `FACE_OUTER_BOUND` / `FACE_BOUND` per advanced face and honours their `.T./.F.`
  orientation flags (a `.F.` bound reverses its loop).
- Ear-clipping triangulation (`PlanarBody::triangulate`, `face_triangles`) that bridges
  every hole into the outer loop, validates the result against the loop areas, and is
  used by `area`/`volume`; holes are never dropped.
- Generic STEP entity parser (`StepFile::parse`) with refs, strings, numbers,
  enums, lists, typed values and nulls.
- `CARTESIAN_POINT` extraction and `bounds()` / `read_step_bounds(path)`.
- `write_box(path, length, width, height)`: centered box, planar faces,
  `PLANE` + `LINE` geometry, `CLOSED_SHELL` + `MANIFOLD_SOLID_BREP`.
- Analytic curved primitives, centered on the origin in millimeters, `+Z` axis
  (same convention as smlibpy's primitives):
  - `write_cylinder(path, radius, height)` — 2 `PLANE` caps + 1
    `CYLINDRICAL_SURFACE`, seam `LINE`, 2 cap `CIRCLE`s.
  - `write_cone(path, radius_base, radius_top, height)` — frustum, apex cone
    (either radius zero; no cap on that end) or zero-taper cone, via
    `CONICAL_SURFACE`; the surface is placed at the wide end with a positive
    semi-angle, so both taper directions are supported.
  - `write_sphere(path, radius)` — 1 `SPHERICAL_SURFACE`, polar seam `CIRCLE`
    used forwards and backwards by the face loop.
  - `write_torus(path, major_radius, minor_radius)` — 1 `TOROIDAL_SURFACE`,
    tube-seam and outer-equator-seam `CIRCLE`s (`major_radius > minor_radius > 0`).
- `Writer`/`Geometry` helpers: normalized `DIRECTION`s, orthonormalized
  `AXIS2_PLACEMENT_3D`, `CIRCLE`/`LINE`/`PLANE`/analytic surfaces, STEP-safe
  REAL formatting (always a decimal point, `E` exponents), and explicit errors
  for degenerate directions/dimensions.

## Validation

`cargo test -p nodecraft-step` runs unit tests plus three integration test files:
`tests/analytic_primitives.rs` (curved writers) and `tests/trimmed_planar.rs`
(trimmed planar faces with holes). The planar/trimmed coverage:

1. **Structure** (`files_have_expected_topology_and_analytic_surfaces`): parses
   every emitted file, checks declared `ADVANCED_FACE` / `VERTEX_POINT` /
   `EDGE_CURVE` / `CIRCLE` / `LINE` / `CLOSED_SHELL` / `MANIFOLD_SOLID_BREP`
   counts, the analytic surface kind and its numeric arguments, that every `#`
   reference resolves, and that no number is non-finite.
2. **smlibpy round-trip** (`smlibpy_reimport_matches_closed_form_metrics`):
   imports every file with `smlibpy.import_step(path, heal=False)` and compares
   faces/edges/vertices, `is_valid()`, volume, area and bounds with the
   closed-form values (volume/area relative tolerance `1e-9`, bounds `1e-6`).
   Skips with a message if the workspace `.venv` interpreter is absent
   (`NODECRAFT_PYTHON` overrides the path).
3. **Invalid dimensions** are explicit errors (zero/negative/NaN radius, zero
   height, `minor >= major`, both cone radii zero).

Trimmed planar faces (`tests/trimmed_planar.rs`, 11 tests):

- rectangle with one hole and with two holes: triangle areas add up to
  `outer - holes`, no triangle centroid falls inside a hole;
- concave L-shaped outer loop with a hole (ear clipping beyond rectangles);
- writer → reader round-trip: 10-face slab with a through-hole keeps one outer + one
  inner bound per trimmed end face, loop order round-trips exactly, area 48, volume 12;
- reader honours `FACE_BOUND` / `FACE_OUTER_BOUND` `.F.` orientation flags by reversing
  the loop, and rejects faces whose holes have no `FACE_OUTER_BOUND`;
- writer rejects short/degenerate loops, out-of-range points, non-coplanar loops and
  consecutive repeated vertices (explicit errors, no panics);
- the trimmed slab re-imports through `smlibpy.import_step(heal=False)` as a valid
  solid: 10 faces, 24 edges, 16 vertices, volume 12, area 48, bounds
  `(0,0,0)..(4,4,1)`;
- box (6 faces, no holes) still passes unchanged.

Measured imports (smlibpy 0.4.0, solidcore `main`):

| primitive | faces | edges | vertices | valid | volume | area |
|---|---:|---:|---:|:---:|---:|---:|
| `write_cylinder(5, 10)` | 3 | 2 | 2 | yes | 785.398163 | 471.238898 |
| `write_cylinder(2.5, 4)` | 3 | 2 | 2 | yes | 78.539816 | 102.101761 |
| `write_cone(5, 2, 10)` | 3 | 2 | 2 | yes | 408.407045 | 320.700519 |
| `write_cone(2, 5, 10)` (inverted) | 3 | 2 | 2 | yes | 408.407045 | 320.700519 |
| `write_cone(5, 0, 10)` (apex top) | 2 | 1 | 2 | yes | 261.799388 | 254.160185 |
| `write_cone(0, 5, 10)` (apex bottom) | 2 | 1 | 2 | yes | 261.799388 | 254.160185 |
| `write_cone(3, 3, 6)` (zero taper) | 3 | 2 | 2 | yes | 169.646003 | 169.646003 |
| `write_sphere(5)` | 1 | 0 | 2 | yes | 523.598776 | 314.159265 |
| `write_torus(8, 2)` | 1 | 0 | 1 | yes | 631.654682 | 631.654682 |

Note the smlibpy kernel counts fewer edges than the files declare: seam and
degenerate edges are not separate kernel edges (a sphere reports 0 kernel edges
for its 1 seam `EDGE_CURVE`; a torus reports 0 for its 2 seam `EDGE_CURVE`s).
That matches what smlibpy itself writes for the same primitives.

## Worker wiring

`nodecraft-worker` supports:
- `stepimporter` for planar STEP files (read_planar_body → display mesh).
- `exporter` for a direct box source (AP214 write_box).

The curved writers above are crate-level only; wiring the worker `exporter` to
`write_cylinder`/`write_cone`/`write_sphere`/`write_torus` belongs to the worker
crate (not edited here). A worker exporter change only has to match on
`primitiveType` and call the matching writer.

## Not yet implemented

- Solidcore `Brep` → STEP faces/curves for arbitrary solids (trimmed planar faces are
  supported through `PlanarBody`; nodecraft-solidcore still pushes every B-rep loop as a
  separate face).
- STEP → solidcore `Brep` topology construction.
- Nested holes (island-within-hole) and holes touching the outer loop; the
  triangulation rejects what it cannot bridge/clip with an explicit error.
- B-spline curves/surfaces, colors, assemblies, units other than mm.
- Curved STEP *import* (the planar reader still rejects curved files; smlibpy
  handles those in the Python worker).
