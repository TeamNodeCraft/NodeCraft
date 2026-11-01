# Agent Guide

This directory is for humans and agents working on NodeCraft. Read this before
changing geometry, the node canvas, or the viewport.

## Documents

- [How NodeCraft works](architecture.md) — DAG, session, evaluation and planned threading.
- [Philosophy](philosophy.md) — invariants and rules for changes.
- [Vision](vision.md) — what NodeCraft is and why it exists.
- [Current state](current-state.md) — what works today.
- [Bugs](bugs.md) — known failures and risky areas.
- [TODO](todo.md) — current priorities.

## Working rules

1. Read the relevant notes before touching the code.
2. Fix the shared invariant, not the visible symptom.
3. Add a regression test when fixing a real bug.
4. Keep exact geometry and display geometry separate.
5. Run the smallest relevant tests first, then the wider suite.

The private development tree contains large STEP and vendor corpora that are not
included in this public snapshot. Missing fixture tests may be skipped locally;
that is not a reason to weaken the code path.
