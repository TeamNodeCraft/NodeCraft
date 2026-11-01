# Analytic conic STEP support — status and explicit limits

`step-conics` reads and writes the analytic conic curve entities of ISO-10303-21
(AP214) on top of the public `nodecraft_step::{StepFile, Value}` reader. It does not
depend on the B-rep, tessellation or kernel crates and it does not modify
`nodecraft-step`.

## What works

| area | status |
|---|---|
| `ELLIPSE`, `HYPERBOLA`, `PARABOLA` | parsed with placement, semi-axes/focal distance |
| `AXIS2_PLACEMENT_3D` | parsed with optional `$` axis/reference direction; resolved to a right-handed orthonormal `Frame` using the STEP defaults |
| `DIRECTION`, `CARTESIAN_POINT` | parsed (3D directions and points; a 2D placement location is an explicit error) |
| `TRIMMED_CURVE` on a conic basis | parsed into `TrimmedConic { trim_1, trim_2, sense_agreement, master_representation }` |
| writer strings | `Conic::step_entity`, `Trim::step_trimmed_curve`, `ConicWriter::{push_placement, push_conic, push_trimmed_conic, entities, finish}`, `write_conics` |
| round-trip | `write_conics(&[Conic])` → `ConicScene::parse` reproduces the same `Conic`s; trimmed conics round-trip their raw parameters, sense and `.PARAMETER.` master |
| trim limits | `validate_trim(conic, trim, AngleUnit) -> TrimRange` / `ConicLimit` |

All maps in `ConicScene` are keyed by STEP entity id (`BTreeMap`) so iteration is
deterministic. `ConicScene::trim_for(conic_id)` returns the first conic-basis trim in
entity-id order.

## Explicit limits

**Parsing**

* Only 3D conic definitions are read. A conic reached through `SURFACE_CURVE`,
  `PCURVE` or `SEAM_CURVE` is returned as the 3D curve; no UV/parametric
  representation is reconstructed.
* Only parameter trims are modelled. A `TRIMMED_CURVE` with
  `master_representation .CARTESIAN.`, with no `PARAMETER_VALUE` in a trim set, or
  with more than one parameter value per trim end fails with
  `ConicError::UnsupportedTrim`. A cartesian point given *alongside* a parameter value
  is accepted and ignored (the common export shape
  `(#point,PARAMETER_VALUE(0.))`).
* A `TRIMMED_CURVE` whose basis is not a conic (a line, a B-spline, …) is ignored:
  other readers own it. A dangling basis reference is `ConicError::MissingEntity`.
* Non-finite STEP reals are rejected on read (`ConicError::NonFinite` /
  `NonFiniteTrim`), as are non-positive conic axes (`NonPositive`) and placements that
  do not resolve (`InvalidPlacement`).
* `ConicScene::parse` is all-or-nothing: one malformed conic fails the whole scene
  rather than returning partial data.

**Periodicity / trimming**

* `ELLIPSE` is periodic (`2π` rad / `360°`); `HYPERBOLA` and `PARABOLA` are open,
  single-branch curves (`ConicDomain`).
* STEP does not record whether ellipse trim values are radians or degrees. The crate
  never guesses: the caller passes `AngleUnit::Radians` or `AngleUnit::Degrees` to
  `validate_trim`.
* `validate_trim` rejects non-finite trims, `trim_2 < trim_1`
  (`TrimReversed`; use `sense_agreement` for direction) and empty intervals
  (`EmptyTrim`). For an ellipse it also rejects spans beyond one period
  (`TrimSpanExceedsPeriod`) and reports `full_period` when the span is exactly one
  period. Seam-crossing trims are not normalized; they are rejected as reversed
  intervals because STEP gives no seam rule.
* Open conics accept any increasing interval (the branch is unbounded).
* Trimmed conics are **not evaluated or tessellated** here; `TrimRange` is a
  validated description, not a curve sampler.

**Writer**

* A conic-only ISO-10303-21 file is emitted (minimal header, no product/B-rep
  section). Entities are numbered from `#1`.
* Trim entities are always parameter trims (`.PARAMETER.`) with typed
  `PARAMETER_VALUE` selects; cartesian trim points are not emitted.
* Non-finite or non-positive geometry is rejected before any line is added, so a
  failed `push_*` leaves the writer unchanged.
* Reals use the shortest round-tripping decimal form (`10.0`, `0.`, `1.5`,
  `1.E-7`); non-finite values are `InvalidGeometry`.

## Tests

`cargo test -p step-conics`:

* 14 integration tests in `tests/conics.rs` — inline STEP parsing of all three conics
  and placements, STEP placement defaults and degenerate placements, malformed
  entities, unsupported/cartesian/multi-point trims, non-conic trims ignored,
  writer entity strings and writer determinism, write→parse round-trip for full and
  trimmed conics, writer rejection without partial entities, and the explicit
  periodic/open trim limits.
* 1 doctest in `src/lib.rs` (parse + trim validation).

## Remaining integration (out of scope here)

Nothing consumes `ConicScene` yet: `nodecraft-step` does not detect conic-only parts
and `nodecraft-worker` does not display conics. The natural next step is a
`nodecraft-step` entry point (for example `detect_conics(path)`) that calls
`step_conics::ConicScene::parse`, analogous to `detect_analytic_solid`, plus a
display/tessellation path for trimmed conics once `TrimRange` is validated.
