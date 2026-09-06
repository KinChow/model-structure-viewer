# Frontend Graph Sources

## Decision

MSV should use mature graph primitives for the canvas rather than expanding the
custom SVG interaction layer. The target stack is:

- `@xyflow/react` for nodes, handles, selection, pan, zoom, and minimap.
- `elkjs` for compound graph node placement and container sizing.
- `@tisoap/react-flow-smart-edge` for obstacle-aware paths between leaf nodes.
- A thin MSV adapter that maps the existing structure IR, formula metadata,
  Tensor Shape data, Cost Lens data, and PD data into graph nodes and edges.

ELK edge sections are intentionally not passed to React Flow. ELK and React
Flow use different coordinate representations for nested nodes and compound
frames; Smart Edge reads the current controlled React Flow nodes instead. Edges
whose source or target is a compound frame use React Flow's native geometry so
they enter the frame directly rather than routing around the frame's children.

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

## Edge Semantics and Routing

Model builders declare stable child-id pairs in `dataflow_edges` for common
MLP, GQA/MLA, and MoE modules. The graph materializer resolves those pairs to
tree paths, keeping model semantics out of the canvas renderer. A legacy
semantic matcher remains as a compatibility fallback for specialized variants
that have not migrated to declarations yet. Parent-child containment is
represented by React Flow `parentId` and compound frames. `react-flow-smart-edge`
owns only the final obstacle-aware SVG path for edges between ordinary nodes.
The dependency is MIT licensed; see its package documentation and repository:

- https://github.com/tisoap/react-flow-smart-edge
- https://reactflow.dev/learn/layouting/layouting#routing-edges

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
