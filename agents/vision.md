# Vision

## What we want to build

A local-first, node-based CAD application where users construct exact solids by
connecting operations, inspect the resulting B-rep and mesh in a fast GPU
viewport, and reuse designs through linked documents.

NodeCraft should feel direct: edit the graph, see exact results, orbit the model,
and move between part and assembly contexts without losing work.

## Why

- Existing node CAD tools often trade away exactness or hide evaluation behind a
  slow round trip.
- A flat DAG plus explicit worker boundaries makes evaluation understandable,
  cacheable and debuggable.
- A native Rust/egui application gives us one code path for the editor, viewport
  and native window chrome.
- The project should remain usable without a cloud service or proprietary file
  format.

## Long-term direction

- exact B-rep operations with clear failure reporting
- linked parts and assembly transforms
- responsive GPU picking and display-quality LOD
- a presentation runtime for animation and scene transforms, separate from CAD
  evaluation
- macOS, Linux and Windows builds from the same source
