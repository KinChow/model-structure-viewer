import assert from "node:assert/strict";
import test from "node:test";
import { buildElkHierarchyEdges } from "./elkHierarchyEdges.js";

function graph(paths, pairs) {
  return {
    nodes: paths.map((path) => ({ path })),
    edges: pairs.map(([source, target], index) => ({
      id: `edge-${index}`, kind: "dataflow", source, target,
    })),
  };
}

test("cross-container edge is decomposed at the common ancestor without changing its endpoints", () => {
  const result = buildElkHierarchyEdges(graph(
    ["root", "root.0", "root.0.0", "root.0.1", "root.1", "root.1.0"],
    [["root.0.1", "root.1.0"]],
  ));
  assert.deepEqual(result.segmentsByEdge.get("edge-0"), [
    "edge-0::out::root.0", "edge-0::core", "edge-0::in::root.1",
  ]);
  assert.deepEqual(result.partsByOwner.get("root"), [{
    id: "edge-0::core",
    source: "root.0::bridge::edge-0::out",
    target: "root.1::bridge::edge-0::in",
  }]);
});

test("edge to its ancestor terminates on the ancestor border port", () => {
  const result = buildElkHierarchyEdges(graph(
    ["root", "root.0", "root.0.0", "root.0.0.0"],
    [["root.0.0.0", "root.0.0"], ["root.0", "root.0.0.0"]],
  ));
  assert.deepEqual(result.partsByOwner.get("root.0.0").map((part) => part.id), [
    "edge-0::core", "edge-1::in::root.0.0",
  ]);
  assert.equal(result.partsByOwner.get("root.0")[0].id, "edge-1::core");
  assert.ok(result.portsByNode.get("root.0.0").some((port) => port.direction === "in"));
  assert.ok(result.portsByNode.get("root.0").some((port) => port.direction === "out"));
});
