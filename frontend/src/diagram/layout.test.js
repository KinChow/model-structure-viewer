import assert from "node:assert/strict";
import test from "node:test";
import { layoutGraph } from "./layout.js";

test("layoutGraph exposes independent visible nodes and edges", () => {
  const graph = layoutGraph({
    name: "model",
    type: "model",
    children: [
      { name: "embed", type: "embedding", children: [] },
      { name: "layers", type: "layer-group", repeat: 2, children: [
        { name: "attention", type: "attention", children: [] },
      ] },
    ],
  }, new Set(["root", "root.1"]));

  assert.deepEqual(graph.edges.map(({ source, target }) => [source, target]), [
    ["root", "root.0"],
    ["root", "root.1"],
    ["root.1", "root.1.0"],
  ]);
  assert.equal(graph.nodes.find((node) => node.path === "root.1").children, undefined);
  assert.equal(graph.containerFrames.length, 2);
});
