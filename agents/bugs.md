# Known Bugs and Risky Areas

These are current known failures or fragile areas. Verify before changing them.

## Sphere seam

`nodecraft-mesh` has a known failure:

```text
sphere_no_pole has open or non-manifold seam edges
```

Do not suppress the seam check to make the test green; fix the seam topology.

## Torus/cylinder join fillet (resolved in the app)

A constant-radius fillet on the torus/cylinder join used to stop part-way around
the join when only one seam-split edge was picked. Fillet now expands the pick
through SolidCore's tangent chain, so one picked join edge rounds the continuous
run without a gap. Upstream issue:
[tangency-labs/solidcore#2826](https://github.com/tangency-labs/solidcore/issues/2826).
Keep this regression covered when changing edge selection or the blend path.

## Feature edge ghost editing is not finished

Fillet/chamfer edge selection now keeps the downstream breakpoint and previews
the feature input as the transform-edit ghost. The basic path works (ghost
edges are pickable, the breakpoint is not replaced), but the selection and
overlay interaction is **not 100% of the desired behaviour yet**. Treat this as
work in progress when changing selection, comparison overlays or the edit-ghost
path.

## Constructive chamfer can cut away a feature

A constructive chamfer around a boss/feature can remove most of the body on some
combined shapes. SolidCore currently returns a technically valid result, so the
app has no error to show. This is a kernel result issue; add validation only
together with an upstream regression, not as a local visual workaround.

## STEP analytic edge fallback

`solidcore-convert` cannot record STEP edge curves that have no B-spline form.
STEP import now uses `solidcore-tess` first and falls back to the legacy
`nodecraft-mesh` tessellator only for those bodies; remove the fallback when the
record conversion supports them.

## Grid zoom crossfade

Major/minor grid line weights still pop or change density around decade
boundaries. The intended fix is a single hierarchical line-level model derived
from world coordinate and continuous zoom, not another mask patch.

## Shift-drag cable insertion

Insertion can jump when retargeting a single downstream coalescing node. The
existing multi-branch transition does not cover this case. Fix the preview and
document shift model together.

## Cross-platform

Windows and Linux compile in CI but have no interactive verification yet.
Native window chrome, file dialogs, clipboard and GPU surface behaviour are the
highest-risk areas.

## STEP coverage

STEP import and display tessellation now use the official reader plus
`solidcore-tess`; there is no separate slow Stepio tier. The known sphere seam
remains the only failure in the fast workspace tier.

## Public snapshot gaps

Large STEP corpora, reference checkouts, reports and the full test matrix are
not public. Do not interpret a missing fixture as "no regression".
