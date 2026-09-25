import assert from "node:assert/strict";
import test from "node:test";
import { buildFocusedLayoutGraph, focusTopLevelPath, normalizeFocusedElkGraph } from "./focusGraph.js";
import { materializeStructureGraph } from "../structure/graph/materializeStructureGraph.js";

function structure() {
  return {
    graph: materializeStructureGraph({
      id: "model",
      type: "model",
      attributes: {
        dataflow_edges: [
          ["embedding", "decoder"],
          ["decoder", "final_norm"],
          ["final_norm", "lm_head"],
          ["decoder", "sidecar"],
          ["sidecar", "lm_head"],
        ],
      },
      children: [
        { id: "embedding", type: "embedding", children: [] },
        {
          id: "decoder",
          type: "decoder",
          children: [{
            id: "layer",
            type: "layer",
            children: [
              { id: "attention", type: "attention", children: [] },
              { id: "mlp", type: "mlp", children: [] },
            ],
          }],
        },
        { id: "final_norm", type: "normalization", children: [] },
        { id: "lm_head", type: "output", children: [] },
        { id: "sidecar", type: "dspark", children: [{ id: "deep", type: "operator", children: [] }] },
      ],
    }),
  };
}

test("focusTopLevelPath resolves nested nodes to their module", () => {
  assert.equal(focusTopLevelPath("root.1.0.1"), "root.1");
  assert.equal(focusTopLevelPath("root"), null);
});

test("focused graph keeps one subtree and projects external edges to boundaries", async () => {
  const graph = buildFocusedLayoutGraph(
    structure(),
    new Set(["root", "root.1", "root.1.0"]),
    "root.1.0.1",
  );
  assert.equal(graph.focusMode, true);
  assert.equal(graph.focusPath, "root.1");
  assert.ok(graph.nodes.some((node) => node.path === "root.1.0.0"));
  assert.ok(graph.nodes.some((node) => node.path === "root.1.0.1"));
  assert.ok(graph.nodes.some((node) => node.path === "root"));
  assert.equal(graph.nodes.some((node) => node.path === "root.0"), false);
  assert.equal(graph.nodes.some((node) => node.path === "root.4"), false);
  assert.ok(graph.nodes.some((node) => node.synthetic));
  assert.ok(graph.edges.some((edge) => edge.source === "root.1" && edge.target.startsWith("root.9")));
  assert.ok(graph.edges.some((edge) => edge.source.startsWith("root.9") && edge.target === "root.1"));
  assert.ok(graph.containerFrames.some((frame) => frame.id === "root"));
});

test("without focus the regular overview graph is unchanged", () => {
  const graph = buildFocusedLayoutGraph(structure(), new Set(["root"]), null);
  assert.equal(graph.focusMode, undefined);
  assert.equal(graph.focusPath, undefined);
  assert.ok(graph.nodes.some((node) => node.path === "root"));
});

test("focused normalization anchors the module shell and keeps boundary terminals stable", () => {
  const graph = normalizeFocusedElkGraph({
    focusMode: true,
    focusPath: "root.1",
    nodes: [
      { path: "root", x: 0, y: 0, width: 800, height: 600 },
      { path: "root.1.0", x: 60, y: 80, width: 120, height: 64 },
      { path: "root.9000", x: 20, y: 30, width: 220, height: 64, synthetic: true, boundaryDirection: "in" },
      { path: "root.9001", x: 20, y: 30, width: 220, height: 64, synthetic: true, boundaryDirection: "out" },
    ],
    containerFrames: [
      { id: "root", x: 0, y: 0, width: 900, height: 700 },
      { id: "root.1", x: 10, y: 20, width: 400, height: 240 },
    ],
    edges: [
      { id: "inside", source: "root.1.0", target: "root.1.0", routePoints: [{ x: 1, y: 2 }] },
      { id: "in", source: "root.9000", target: "root.1.0", routePoints: [{ x: 1, y: 2 }] },
    ],
  });
  assert.equal(graph.containerFrames.find((frame) => frame.id === "root.1").x, 420);
  assert.equal(graph.containerFrames.find((frame) => frame.id === "root.1").y, 120);
  assert.deepEqual(
    graph.nodes.find((node) => node.path === "root.9000").x,
    80,
  );
  assert.deepEqual(
    graph.nodes.find((node) => node.path === "root.9001").x,
    1240,
  );
  assert.ok(graph.edges.find((edge) => edge.id === "in").routePoints.length >= 2);
});
