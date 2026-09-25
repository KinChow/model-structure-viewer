# Hierarchical ELK layout POC (2026-09-25)

## Purpose

This POC verifies the proposed B design:

> each visible hierarchy owner gets its own ELK layout, while edges between
> siblings are submitted to that owner's layout.

It is an evidence experiment only. It does not change production layout, Graph
IR v2, checkpoint truth, cost accounting, or React Flow rendering.

Run:

```bash
node scripts/evidence/structure/poc-hierarchical-elk.mjs
```

Output:

```text
artifacts/architecture-repair/hierarchical-elk-poc-2026-09-25.json
```

## What was tested

Representative fully expanded models:

- DeepSeek-V4.1-Flash
- Qwen3.8-Flash-Next
- GLM-5.2
- Kimi-K3

For each model the script runs two variants:

1. `intrinsic`: parent ELK nodes use the size produced by their nested layout.
2. `proxy`: parent ELK nodes use the visible module box size while nested
   layouts are still solved independently.

The script also records:

- visible Graph IR node and edge counts;
- number of independent hierarchy layout runs;
- same-level edges submitted to the root layout;
- cross-hierarchy edges that require boundary handling;
- top-level root extent and positions.

## Result

The initial `intrinsic` variant is not sufficient. Although it gives every
hierarchy its own ELK run, expanded descendants still determine the parent
box size. On the representative models this reproduces the original failure
class: the root extent remains tens or hundreds of thousands of pixels high.

The `proxy` variant keeps the root layout compact because nested content does
not participate in the parent solve. This confirms the important refinement:

```text
independent per-level layout
  + parent-level geometry contract
  + explicit boundary ports/routes
```

is materially different from simply recursively calling ELK and passing the
intrinsic child size upward.

The proxy result is not yet production-ready. It intentionally does not render
the nested coordinates into React Flow and does not route cross-hierarchy
segments. It only proves that the proposed separation can prevent expanded
internal detail from moving the top-level spine.

The follow-up implementation POC added fixed shells, local-coordinate
scaling, synthetic boundary ports, and React Flow rendering. It passed the
finite-geometry and parent-spine unit checks, but failed the browser rendering
gate: expanded internal edges crossed visible descendant tiles and the
scaling made deep modules too small to read. This is not an acceptable
production result. The production UI was therefore kept on the previous
ELK-only implementation while this POC remains isolated.

## Decision

The B direction remains technically reasonable, but the terminal design cannot
be “one ELK call per level” alone. The required production contract is:

1. solve each owner with ELK using same-level edges only;
2. use a stable parent-level box/anchor contract for the outer solve;
3. solve expanded descendants in the owner's local coordinate system;
4. map boundary ports to the nearest visible ancestor;
5. route cross-hierarchy edges through explicit boundary segments;
6. verify that collapsing and expanding a descendant changes only its local
   geometry and the necessary boundary routes, not the semantic top-level
   order.

This is still ELK-native layout. The additional code is a coordinator and
boundary mapping layer, not a replacement layout engine.

## Current conclusion

The POC does not justify switching the production renderer to “all visible
levels in one canvas.” The best implementation direction is now a **hybrid
hierarchical viewer**:

1. Keep the top-level overview compact and stable.
2. Use ELK hierarchical/local layouts for one focused module at a time.
3. Keep the focused module's children in a local coordinate system without
   shrinking them to fit a fixed global shell.
4. Route edges crossing the focused module through explicit boundary ports.
5. Restore the overview positions when focus is closed.

This is still B in terms of hierarchy ownership, but it avoids the failed
requirement that every expanded descendant be rendered simultaneously inside
the global overview. It also matches the existing viewer's “Focus canvas”
interaction and is a better fit for Kimi-K3 / large decoder trees than either
global recursive scaling or a huge one-page compound graph.

The next production POC should therefore be a **focused-module detail
viewport**, not another global coordinate/scaling experiment. At this stage the
existing ELK-only renderer remains the overview production path; the focused
viewport is the bounded detail path described below.

## Focused-module implementation result

The focused-module viewport is now implemented without changing Graph IR v2 or
the default overview layout:

- the overview continues to use the production single-compound ELK layout;
- selecting an expandable module exposes `打开模块详情` / `Open module detail`;
- detail mode keeps the selected top-level module's visible descendants and
  other top-level modules as boundary tiles;
- external edges are projected to the nearest included module boundary, while
  original source and target paths remain on the edge metadata;
- the focused shell is normalized to a stable viewport anchor, and synthetic
  input/output terminals are placed outside the shell; descendant expansion
  therefore does not translate the terminals or the final-norm boundary;
- closing detail mode restores the normal overview graph.

The implementation is in
`frontend/src/diagram/focusGraph.js`; its independent graph tests are in
`frontend/src/diagram/focusGraph.test.js`. The browser audit is
`scripts/evidence/structure/audit-focused-module.mjs`.

Chrome validation on **September 25, 2026**:

- all 60 built-in models entered focused detail successfully;
- 60/60 had a ready layout and at least one rendered edge;
- 0 page errors;
- 0 sampled edge occlusions;
- representative DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, GLM-5.2, and
  Kimi-K3 focused views were checked independently.

This solves the verified failure mode without pretending that all expanded
descendants can remain readable in one global canvas. Full deep inspection is
now explicitly local to the focused module.

## Overview direction correction

The production overview now uses a linear top-level ELK hierarchy: visible
top-level modules are direct children of the model frame, the model frame and
outer wrapper use `RIGHT` direction, and nested operator contents remain local
`DOWN` layouts. This keeps the semantic input-to-output order on the x-axis
without changing Graph IR node paths or dataflow endpoints. Auxiliary DSpark /
MTP branches remain in the same ELK solve and may occupy a secondary row, but
they no longer force the main input/output spine into a vertical stack.

## External implementation research

The official ELK documentation describes a native feature that directly targets
this failure mode: **Topdown Layout**. Unlike the normal bottom-up behavior,
Topdown Layout computes the root first and then recursively lays out children
inside the space provided by their parents. Child layouts are scaled to fit
instead of enlarging the parent. A hierarchical node must use
`org.eclipse.elk.nodeSize.fixedGraphSize=true`, and the documented mode is not
compatible with `INCLUDE_CHILDREN`. See:

- https://eclipse.dev/elk/reference/options/org-eclipse-elk-topdownLayout.html
- https://eclipse.dev/elk/blog/posts/2023/23-04-11-topdown-layout.html

The local `elkjs` 0.12.0 bundle contains these options, and a synthetic
multilevel graph was successfully laid out with:

```text
elk.topdownLayout = true
elk.topdown.nodeType = ROOT_NODE / HIERARCHICAL_NODE
elk.nodeSize.fixedGraphSize = true
```

This is a stronger candidate than merely enabling partitioning. Partitioning
can constrain sibling ordering, but it does not by itself define whether an
expanded child is allowed to enlarge its parent.

React Flow also has a native parent/child sub-flow model: child positions are
relative to the parent, and sub-flows can connect to nodes outside their
parent. This makes a local-coordinate implementation compatible with the
renderer, provided the viewer uses `parentId`/relative coordinates or applies
the equivalent transform consistently. See:

- https://reactflow.dev/learn/layouting/sub-flows

## Revised candidate

The most promising mature design is therefore **B+ / ELK Topdown-inspired
hierarchical layout**:

1. ELK lays out stable parent shells at the outer level.
2. Each expanded shell receives an independent local ELK layout.
3. The shell has a fixed geometry contract; child intrinsic size cannot resize
   the outer shell.
4. Cross-hierarchy edges use explicit boundary ports and are routed in the
   nearest common owner layout.
5. The renderer stores local child coordinates relative to the shell.
6. If the local layout is larger than the shell, either the ELK topdown scale
   factor is rendered or an equivalent local transform is applied.

Native ELK Topdown should be tested before implementing a custom coordinator.
It is not yet accepted as the production solution because the current viewer
does not consume ELK's `topdown.scaleFactor`, and excessive scaling could make
deeply expanded modules unreadable. If that renderer gate fails, the fallback
is the explicit B+ coordinator with the same fixed-shell contract but no
implicit scaling.

This gives the decision order:

```text
1. Native ELK Topdown + fixed shells + renderer scale support
2. Independent per-owner ELK + fixed shells + explicit local transform
3. Current single compound ELK only as rollback
```
