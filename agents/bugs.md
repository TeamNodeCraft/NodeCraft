# Known Bugs and Risky Areas

These are current known failures or fragile areas. Verify before changing them.

## Exact extrusion connectivity

`rust/crates/nodecraft-extrude/tests/exact_extrude.rs`:

```text
legacy_linear_sweep_measures_exactly_but_is_not_connected
```

The test currently fails the connectivity assertion. Investigate the B-rep
topology produced by the legacy linear sweep rather than weakening the test.

## Sphere seam

`nodecraft-mesh` has a known failure:

```text
sphere_no_pole has open or non-manifold seam edges
```

Do not suppress the seam check to make the test green; fix the seam topology.

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

## Public snapshot gaps

Large STEP corpora, reference checkouts, reports and the full test matrix are
not public. Do not interpret a missing fixture as "no regression".
