# Frontend Graph Sources

## Decision

MSV should use mature graph primitives for the canvas rather than expanding the
custom SVG interaction layer. The target stack is:

- `@xyflow/react` for nodes, handles, selection, pan, zoom, and minimap.
- `elkjs` for compound graph layout and routed edge sections.
- A thin MSV adapter that maps the existing structure IR, formula metadata,
  Tensor Shape data, Cost Lens data, and PD data into graph nodes and edges.

## Reference Source

The graph interaction and component design are informed by:

- `modelmap`: https://github.com/lizhaoliu/modelmap
- Local reference checkout: `/Users/zhouzijian01/Desktop/workspace/code/kinchow/modelmap`
- License: MIT, see the reference checkout `LICENSE`.
- Relevant reference files: `web/src/graph/layout.ts`,
  `web/src/graph/nodes.tsx`, `web/src/graph/edges.tsx`, and
  `web/src/flow/engine.ts`.

MSV does not copy modelmap's model extraction, trace data, cost planner, or
flow replay engine. Those parts depend on modelmap-specific data contracts and
would conflict with MSV's config/safetensors truth, formulas, Cost Lens, and
PD analysis.

## ONNX Boundary

ONNX may be accepted as an optional graph input in a future adapter, but it is
not the source of truth for MSV model structure. Many supported models do not
ship an ONNX graph, and ONNX does not contain MSV's weight truth, formula
metadata, hardware cost assumptions, or PD deployment semantics.

## Attribution Rule

Any code ported from the modelmap checkout must retain the MIT copyright and
license notice, and the porting commit must identify the original file. Ideas
or behavior reimplemented against MSV's own data model are documented as
inspiration, not as copied source code.
