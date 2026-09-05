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

  assert.deepEqual(graph.edges.filter((edge) => edge.kind === "structure").map(({ source, target }) => [source, target]), [
    ["root", "root.0"],
    ["root", "root.1"],
    ["root.1", "root.1.0"],
  ]);
  assert.equal(graph.nodes.find((node) => node.path === "root.1").children, undefined);
  assert.equal(graph.nodes.find((node) => node.path === "root.0").stage, "input");
  assert.equal(graph.nodes.find((node) => node.path === "root.1.0").stage, "decoder");
  assert.deepEqual(graph.edges.filter((edge) => edge.evidence === "module-order").map(({ source, target }) => [source, target]), [["root.0", "root.1"]]);
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
  assert.deepEqual(JSON.parse(JSON.stringify(laidOut.edges.map(({ id, source, target, kind, evidence }) => ({ id, source, target, kind, evidence })))), graph.edges);
  assert.ok(laidOut.edges.some((edge) => edge.sections.length > 0));
  assert.deepEqual(laidOut.containerFrames, []);
});

test("layoutGraph adds dataflow edges only when tensor shapes match", () => {
  const graph = layoutGraph({
    name: "block", type: "module", children: [
      { name: "gate", type: "operator", input_shape: [1, 4], output_shape: [1, 8], children: [] },
      { name: "activation", type: "operator", input_shape: [1, 8], output_shape: [1, 8], children: [] },
      { name: "same-shape", type: "operator", input_shape: [1, 8], output_shape: [1, 8], children: [] },
      { name: "other", type: "operator", input_shape: [1, 16], output_shape: [1, 16], children: [] },
    ],
  }, new Set(["root"]));
  assert.deepEqual(graph.edges.filter((edge) => edge.kind === "dataflow" && edge.evidence !== "module-order").map(({ source, target }) => [source, target]), [
    ["root.0", "root.1"],
    ["root.1", "root.2"],
  ]);
});
