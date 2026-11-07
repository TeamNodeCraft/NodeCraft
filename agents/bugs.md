# Known Bugs and Risky Areas

These are current known failures or fragile areas. Verify before changing them.

## Sphere seam

`nodecraft-mesh` has a known failure:

```text
sphere_no_pole has open or non-manifold seam edges
```

Do not suppress the seam check to make the test green; fix the seam topology.

## Partial multi-edge join fillet

A constant-radius fillet on the torus/cylinder join can stop part-way around the
join. SolidCore reports `EdgeNotBlendable { reason: Smooth }` for some join
edges; the application now reports that refusal instead of dropping the refused
edges and returning a partial round. Upstream issue:
[tangency-labs/solidcore#2826](https://github.com/tangency-labs/solidcore/issues/2826).
`#2863` is merged and present but does not cover this remaining case.

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
