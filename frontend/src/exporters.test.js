import assert from "node:assert/strict";
import test from "node:test";
import { exportStructure } from "./exporters.js";
import { materializeStructureGraph } from "./structure/graph/materializeStructureGraph.js";

// P7（步骤 7）：夹具从 legacy root 换成 Graph IR（导出只消费图）。
const structure = {
  summary: { strategy: "frontend-architecture-template" },
  source: {},
  graph: materializeStructureGraph({
    id: "model",
    name: "Model",
    type: "model",
    attributes: {},
    children: [
      {
        id: "decoder",
        name: "Decoder Layers",
        type: "decoder",
        repeat: 2,
        attributes: {},
        children: [],
      },
    ],
  }),
};

test("exports structure as JSON on the frontend", () => {
  const text = exportStructure(structure, "json");
  assert.equal(JSON.parse(text).graph.nodes[0].canonical_id, "model");
});

test("exports structure as Mermaid on the frontend", () => {
  const text = exportStructure(structure, "mermaid");
  assert.match(text, /^flowchart TD/);
  assert.match(text, /Decoder Layers x2/);
});

test("exports structure as DOT on the frontend", () => {
  const text = exportStructure(structure, "dot");
  assert.match(text, /^digraph ModelStructure/);
  assert.match(text, /rankdir=LR/);
});

test("exports Graph IR only and ignores foreign legacy fields", () => {
  const text = exportStructure({
    root: { name: "stale", type: "model", children: [] },
    graph: {
      version: 2,
      schema_version: 2,
      root_id: "root",
      nodes: [
        { id: "root", name: "Graph Model", type: "model" },
        { id: "root.0", parent_id: "root", order: 0, name: "Graph Decoder", type: "decoder" },
      ],
      edges: [{ id: "declared", source: "root", target: "root.0", kind: "dataflow", evidence: "declared" }],
    },
  }, "mermaid");
  assert.match(text, /Graph Model/);
  assert.match(text, /Graph Decoder/);
  assert.doesNotMatch(text, /stale/);
});

test("empty graph yields a header-only diagram instead of falling back to a tree", () => {
  assert.equal(exportStructure({ graph: { version: 2, schema_version: 2, root_id: "root", nodes: [], edges: [] } }, "mermaid"), "flowchart TD\n");
  assert.equal(
    exportStructure({ graph: { version: 2, schema_version: 2, root_id: "root", nodes: [], edges: [] } }, "dot"),
    'digraph ModelStructure {\n  rankdir=LR;\n  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica"];\n  edge [color="#64748b"];\n}\n',
  );
});
