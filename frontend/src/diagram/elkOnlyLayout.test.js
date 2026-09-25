import assert from "node:assert/strict";
import test from "node:test";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElkOnly } from "./elkOnlyLayout.js";
import { materializeStructureGraph } from "../structure/graph/materializeStructureGraph.js";

function view(tree, expanded = new Set(["root"])) {
  return layoutGraph({ graph: materializeStructureGraph(tree) }, expanded);
}

function assertFiniteGraph(graph) {
  assert.equal(graph.layoutEngine, "elk-only-poc");
  assert.ok(graph.nodes.every((node) =>
    [node.x, node.y, node.width, node.height].every(Number.isFinite),
  ));
  assert.ok(graph.edges
    .filter((edge) => edge.kind === "dataflow")
    .every((edge) => edge.routePoints?.length >= 2));
}

function overlap(a, b) {
  const x = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
  const y = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  return x > 2 && y > 2;
}

test("ELK-only layout keeps multimodal and auxiliary lanes in one compound solve", async () => {
  const graph = await layoutGraphWithElkOnly(view({
    id: "model",
    type: "model",
    attributes: {
      dataflow_edges: [
        ["vision", "merge"],
        ["tokens", "merge"],
        ["merge", "decoder"],
        ["decoder", "final_norm"],
        ["final_norm", "lm_head"],
        ["decoder", "dspark"],
        ["dspark", "lm_head"],
      ],
    },
    children: [
      { id: "vision", type: "vision-encoder", children: [] },
      { id: "tokens", type: "embedding", children: [] },
      { id: "merge", type: "merger", children: [] },
      { id: "decoder", type: "decoder", children: [{ id: "layer", type: "layer", children: [] }] },
      { id: "dspark", type: "dspark", children: [{ id: "projection", type: "operator", children: [] }] },
      { id: "final_norm", type: "normalization", children: [] },
      { id: "lm_head", type: "output", children: [] },
    ],
  }, new Set(["root", "root.3", "root.4"])));
  assertFiniteGraph(graph);
  const get = (path) => graph.nodes.find((node) => node.path === path);
  assert.ok(get("root.4").y > get("root.3").y, "auxiliary lane should be below the main lane");
  assert.ok(get("root.3").x < get("root.5").x && get("root.5").x < get("root.6").x);
  const siblings = graph.nodes.filter((node) => node.path.split(".").length === 2);
  for (let left = 0; left < siblings.length; left += 1) {
    for (let right = left + 1; right < siblings.length; right += 1) {
      assert.equal(overlap(siblings[left], siblings[right]), false,
        `${siblings[left].path} overlaps ${siblings[right].path}`);
    }
  }
});

test("ELK-only layout routes an expanded internal branch without post-layout mutation", async () => {
  const graph = await layoutGraphWithElkOnly(view({
    id: "model",
    type: "model",
    children: [{
      id: "attention",
      type: "attention",
      attributes: { dataflow_edges: [["q", "scores"], ["k", "scores"], ["scores", "softmax"], ["v", "context"], ["softmax", "context"]] },
      children: [
        { id: "q", type: "operator", children: [] },
        { id: "k", type: "operator", children: [] },
        { id: "v", type: "operator", children: [] },
        { id: "scores", type: "operator", children: [] },
        { id: "softmax", type: "operator", children: [] },
        { id: "context", type: "operator", children: [] },
      ],
    }],
  }, new Set(["root", "root.0"])));
  assertFiniteGraph(graph);
  assert.deepEqual(graph.nodes.map((node) => node.path), [
    "root", "root.0", "root.0.0", "root.0.1", "root.0.2", "root.0.3", "root.0.4", "root.0.5",
  ]);
  const q = graph.nodes.find((node) => node.path === "root.0.0");
  const context = graph.nodes.find((node) => node.path === "root.0.5");
  assert.ok(q.y < context.y);
});
