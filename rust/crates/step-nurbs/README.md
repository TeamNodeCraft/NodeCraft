# step-nurbs

B-spline / NURBS curve and surface support for STEP (ISO 10303-21) data, built on the
generic entity parser in [`nodecraft-step`](../nodecraft-step).

The crate turns the two STEP spellings of a B-spline into one evaluable struct:

* plain entities — `B_SPLINE_CURVE_WITH_KNOTS(...)`, `B_SPLINE_SURFACE_WITH_KNOTS(...)`;
* **complex instances** — STEP writes rational entities as a parenthesised list of
  partial entities, which the generic parser keeps as alternating typed member names and
  argument lists:

  ```text
  #20=(BOUNDED_CURVE() B_SPLINE_CURVE(2,(#1,#2,#3),.UNSPECIFIED.,.F.,.F.)
       B_SPLINE_CURVE_WITH_KNOTS((3,3),(0.,1.),.UNSPECIFIED.)
       CURVE() GEOMETRIC_REPRESENTATION_ITEM()
       RATIONAL_B_SPLINE_CURVE((1.,0.7071067811865476,1.))
       REPRESENTATION_ITEM(''));
  ```

## Usage

```rust
use nodecraft_step::StepFile;
use step_nurbs::{curves, bspline_curve_entity, curve_point};

let text = std::fs::read_to_string("part.step")?;
let file = StepFile::parse(&text)?;

for curve in curves(&file)? {
    let (low, high) = curve.domain();
    let point = curve_point(&curve, 0.5 * (low + high))?;
    println!(
        "degree {} with {} poles, midpoint {point:?}",
        curve.degree,
        curve.poles.len()
    );
}

// Emit the same curve again as a STEP entity body (add the `#id=...;` yourself).
let knots = step_nurbs::Knots::quasi_uniform(3, 4)?;
let body = bspline_curve_entity(3, &[1, 2, 3, 4], &knots, None, false, false)?;
```

## API

| Item | Purpose |
| --- | --- |
| `curves(&StepFile)`, `surfaces(&StepFile)` | Parse every B-spline entity in the file, in entity-id order. |
| `parse_curve(&StepFile, id)`, `parse_surface(&StepFile, id)` | Parse one entity by id. |
| `BsplineCurve` | `degree`, `poles`, `weights: Option<Vec<f64>>`, `knots`, `closed`, `self_intersect`, `domain()`, `point_at(u)`, `is_rational()`. |
| `BsplineSurface` | `u_degree`, `v_degree`, `poles[u][v]`, `weights: Option<Vec<Vec<f64>>>`, `u_knots`, `v_knots`, `u_closed`, `v_closed`, `self_intersect`, `u_domain()`, `v_domain()`, `point_at(u, v)`. |
| `curve_point`, `surface_point` | Free-function evaluation (clamped to the parameter domain). |
| `Knots` | `multiplicities`, distinct `values`, `spec`, `derived`; `expanded()`, `validate_for(degree, control_count)`, `quasi_uniform`, `uniform`, `piecewise_bezier`, `default_for`. |
| `KnotSpec` | `.UNSPECIFIED.`, `.UNIFORM.`, `.QUASI_UNIFORM.`, `.PIECEWISE_BEZIER.`. |
| `bspline_curve_entity`, `bspline_surface_entity` | Emit the STEP entity body for polynomial (simple entity) or rational (complex instance) data. |
| `format_real`, `format_real_list`, `format_int_list`, `format_ref_list`, `format_ref_grid`, `format_real_grid` | STEP literal/list formatting used by the emitters (upper-case `E`, always a decimal point). |

## Parsing details

* Control points are resolved through `CARTESIAN_POINT` references. 2D points are read
  as `z = 0`; a missing or non-point reference is an error naming the entity id.
* Poles are indexed `poles[u][v]` for surfaces (STEP's first index is `u`).
* `RATIONAL_B_SPLINE_CURVE` / `RATIONAL_B_SPLINE_SURFACE` members set `weights`;
  polynomial entities leave `weights` as `None`.
* Weights must be finite and strictly positive; the count must match the poles.
* STEP `LOGICAL` flags (`.T.`, `.F.`, `.U.`) are read as booleans with `.U.` (unknown)
  mapped to `false`.
* Knot vectors are validated against `degree` and control-point count
  (`Σ multiplicities == control_count + degree + 1`, strictly increasing values,
  multiplicity `<= degree + 1`).

### Default knots

When a file omits the knot data (`$` for the multiplicities and/or knot values, a
`B_SPLINE_SURFACE` without the `_WITH_KNOTS` member, or a `B_SPLINE_CURVE`), the vector
is derived from the degree and control-point count and `Knots::derived` is set:

| `knot_spec` | Derived vector | Domain |
| --- | --- | --- |
| `.QUASI_UNIFORM.` / `.UNSPECIFIED.` | clamped: ends multiplicity `degree+1`, evenly spaced interior knots, `[0, 1]` | `[0, 1]` |
| `.UNIFORM.` | multiplicity 1 everywhere, integer values `0 ..= control_count+degree` | `[degree, control_count]` |
| `.PIECEWISE_BEZIER.` | ends `degree+1`, interior multiplicity `degree`; requires `(control_count-1) % degree == 0` | `[0, segments]` |

`.UNSPECIFIED.` defaults to the clamped quasi-uniform vector because that is what most
writers mean when they leave the knots out. A knotless `B_SPLINE_CURVE` uses uniform
knots (the entity has no `knot_spec` attribute).

## Evaluation

Curves and surfaces are evaluated with the de Boor algorithm (*The NURBS Book*,
`FindSpan`/`BasisFuns`/`CurvePoint`), using homogeneous poles for rational entities and
dividing by the accumulated weight. Parameters outside the domain are clamped to its
ends; non-finite parameters, mismatched knot vectors and zero weight sums are errors.
The parameter domain is `[U[degree], U[control_count]]` of the expanded knot vector, so
it is `[0, 1]` for clamped knots and `[degree, control_count]` for uniform knots.

## Limits

* No unit conversion — coordinates are returned exactly as stored in the file.
* No trimming: `TRIMMED_CURVE` / face bounds are not resolved here; the crate parses the
  B-spline entities themselves.
* Only the `knot_spec` spellings listed above are accepted.
* `format_real` uses the shortest round-trip Rust representation with a STEP-compatible
  decimal point and upper-case exponent.

## Tests

```bash
cargo test -p step-nurbs
```

The suite covers inline STEP fixtures for polynomial curves (clamped cubic Bézier,
uniform knotless quadratic, 2D control points), rational complex-instance curves (quarter
circle), polynomial and rational bilinear surfaces, `$`-defaulted surface knots,
emit→parse roundtrips, discovery/ordering, malformed input, and two real AP203 fixtures
(`tests/Sample AP203 Data Files/gear.stp` polynomial curves,
`tests/Sample AP203 Data Files/jack_in_the_box.stp` rational surfaces) when the checkout
contains them.
