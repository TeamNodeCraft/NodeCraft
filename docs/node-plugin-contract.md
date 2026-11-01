# Node plugin exact-geometry contract

Every evaluation node returns two channels:

```rust
NodeResult {
    mesh: Mesh,           // viewport and selection
    exact: ExactGeometry, // evaluated B-rep(s), if the node is exact
}
```

The canonical Rust template lives in
`rust/crates/nodecraft-engine/src/node_plugin.rs`.

## Required for new nodes

1. Implement `NodePlugin` (or follow the same contract in the hardcoded engine
   arm) and return `NodeResult`.
2. If the node produces a solid, cache its exact B-rep on the **same evaluation**
   that produces the display mesh:
   - one body → `ExactGeometry::Brep`
   - several bodies → `ExactGeometry::Breps`
3. Preserve exact input geometry where the operation is exact. Do not expose only
   a mesh and later reconstruct the operation in an exporter.
4. Populate the exact channel on cache hits too. A cached display mesh must not
   suppress the exact result.
5. Exact consumers (Boolean, Extrude, Loft, STEP export) consume the cached
   exact channel. They must not walk the request graph to rebuild upstream
   nodes.

## Sketch exception

A Sketch returns profiles, not a solid B-rep. Loft/Extrude convert those
profiles into their own exact B-rep output. Downstream nodes see the Loft or
Extrude B-rep, not the Sketch.

## Exporter rule

The exporter writes the evaluated exact result when one is present. Graph
re-resolution is a legacy fallback only for stateless assembly/export paths and
should not be added to new nodes.
