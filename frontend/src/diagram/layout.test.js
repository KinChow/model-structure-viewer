import assert from "node:assert/strict";
import test from "node:test";
import { layoutGraph } from "./layout.js";
import { layoutGraphWithElk } from "./elkLayout.js";

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

test("ELK lays out the graph without changing stable node paths", async () => {
  const graph = layoutGraph({
    name: "model", type: "model", children: [
      { name: "a", type: "module", children: [] },
      { name: "b", type: "module", children: [] },
    ],
  }, new Set(["root"]));
  const laidOut = await layoutGraphWithElk(graph);
  assert.deepEqual(laidOut.nodes.map((node) => node.path), ["root", "root.0", "root.1"]);
  assert.ok(laidOut.nodes.every((node) => Number.isFinite(node.x) && Number.isFinite(node.y)));
  assert.deepEqual(laidOut.edges, graph.edges);
  assert.deepEqual(laidOut.containerFrames, []);
});
