# Agent Guide

This directory is for humans and agents working on NodeCraft. Read this before
changing geometry, the node canvas, or the viewport.

## Documents

- [How NodeCraft works](architecture.md) — DAG, session, evaluation and planned threading.
- [Workspace transition TODO](../docs/workspace-transition-todo.md) — current checklist.
- [Workspace and graph-model integration plan](../docs/workspace-integration-plan.md) — implementation order.
- [Graph definitions, instances and flattening](../docs/graph-definition-instance-plan.md) — target nesting/reuse model.
- [Iterator nodes](../docs/node/iterator.md) — target iteration/map model.
- [Philosophy](philosophy.md) — invariants and rules for changes.
- [Vision](vision.md) — what NodeCraft is and why it exists.
- [Current state](current-state.md) — what works today.
- [SolidCore dependencies](../docs/solidcore-sync.md) — vendored kernel, pins and update procedure.
- [Bugs](bugs.md) — known failures and risky areas.
- [TODO](todo.md) — current priorities.
- [Performance baseline](performance-baseline.md) — measured canvas bottlenecks.

## Working rules

1. Read the relevant notes before touching the code.
2. Fix the shared invariant, not the visible symptom.
3. Add a regression test when fixing a real bug.
4. Keep exact geometry and display geometry separate.
5. Run the smallest relevant tests first, then the wider suite.
6. Keep build artifacts out of the workspace budget. `rust/target/debug` can
   grow to tens of gigabytes; remove that directory when disk pressure blocks
   linking. Never delete source, samples, reference checkouts, reports or other
   working files to make room. Do not run `git clean -xfd` while the nested
   public checkout exists.

## Build hygiene

The workspace and public snapshot share one disk. Before a long build:

```sh
df -h .
du -sh rust/target rust/target/debug rust/target/release
```

If linking fails with `No space left on device`, remove build artifacts only:

```sh
rm -rf rust/target/debug
```

That is the supported cleanup path. It forces a debug rebuild but leaves the
release app, source, samples, caches and public checkout untouched.

The private development tree contains large STEP and vendor corpora that are not
included in this public snapshot. Missing fixture tests may be skipped locally;
that is not a reason to weaken the code path.
