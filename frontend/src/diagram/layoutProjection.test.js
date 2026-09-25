import assert from "node:assert/strict";
import test from "node:test";
import { buildLayoutProjection, LANES, VIRTUAL_ROOT } from "./layoutProjection.js";

test("layout projection adds virtual lanes without changing Graph IR paths", () => {
  const graph = {
    version: 2,
    nodes: [
      { path: "root", node: { type: "model" } },
      { path: "root.0", node: { type: "vision-encoder" } },
      { path: "root.1", node: { type: "decoder" } },
      { path: "root.2", node: { type: "dspark" } },
      { path: "root.1.0", node: { type: "attention" } },
    ],
    edges: [],
  };
  const projection = buildLayoutProjection(graph);
  assert.equal(projection.root_id, VIRTUAL_ROOT);
  assert.deepEqual([...projection.laneIds.keys()], LANES);
  assert.equal(projection.layoutParentByPath.get("root.0"), projection.laneIds.get("input"));
  assert.equal(projection.layoutParentByPath.get("root.1"), projection.laneIds.get("main"));
  assert.equal(projection.layoutParentByPath.get("root.2"), projection.laneIds.get("auxiliary"));
  assert.equal(projection.laneByPath.get("root.1.0"), "main");
  assert.equal(projection.layoutParentByPath.get("root.0"), projection.laneIds.get("input"));
  assert.equal(projection.layoutParentByPath.get(projection.laneIds.get("input")), "root");
  assert.deepEqual(projection.layoutChildrenByParent.get(projection.laneIds.get("main")), ["root.1"]);
  assert.ok(!graph.nodes.some((node) => node.virtual));
});

test("projection does not replace fact nodes or infer dataflow edges", () => {
  const graph = {
    version: 2,
    nodes: [
      { path: "root", node: { type: "model" } },
      { path: "root.0", node: { type: "module" } },
    ],
    edges: [{ id: "declared", source: "root.0", target: "root.0", kind: "dataflow", evidence: "declared" }],
  };
  const projection = buildLayoutProjection(graph);
  assert.equal(graph.edges.length, 1);
  assert.deepEqual(graph.edges[0], { id: "declared", source: "root.0", target: "root.0", kind: "dataflow", evidence: "declared" });
  assert.equal(projection.virtualNodes.length, 4);
});
