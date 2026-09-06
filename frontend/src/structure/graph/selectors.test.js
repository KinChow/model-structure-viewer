import test from "node:test";
import assert from "node:assert/strict";
import { graphChildren, graphNodeAt, graphViewNode } from "./selectors.js";

const graph = {
  root_id: "root",
  nodes: [
    { id: "root", canonical_id: "model", name: "Model", parent_id: null, order: 0, type: "model" },
    { id: "root.1", canonical_id: "head", name: "Head", parent_id: "root", order: 1, type: "output" },
    { id: "root.0", canonical_id: "embed", name: "Embed", parent_id: "root", order: 0, type: "embedding" },
    { id: "root.1.0", canonical_id: "proj", name: "Projection", parent_id: "root.1", order: 0, type: "linear" },
  ],
};

test("graph selectors preserve stable graph paths while exposing view children", () => {
  assert.deepEqual(graphChildren(graph, "root").map((node) => node.id), ["root.0", "root.1"]);
  assert.equal(graphNodeAt(graph, "root.1.0").canonical_id, "proj");
  const root = graphViewNode(graph, "root");
  assert.equal(root.id, "model");
  assert.equal(root.path, "root");
  assert.equal(root.children[1].path, "root.1");
  assert.equal(root.children[1].children[0].path, "root.1.0");
});
