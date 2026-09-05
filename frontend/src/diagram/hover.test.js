import assert from "node:assert/strict";
import test from "node:test";
import { isEdgeRelated, isGraphEdgeRelated, isPathRelated, relatedDataflowEdgeIds } from "./hover.js";

test("悬停节点关联自身、祖先和后代", () => {
  assert.equal(isPathRelated("root.1", "root.1"), true);
  assert.equal(isPathRelated("root.1.2", "root.1"), true);
  assert.equal(isPathRelated("root", "root.1.2"), true);
  assert.equal(isPathRelated("root.2", "root.1"), false);
});

test("悬停节点关联经过它的边", () => {
  assert.equal(isEdgeRelated("root", "root.1", "root.1"), true);
  assert.equal(isEdgeRelated("root.2", "root.3", "root.1"), false);
});

test("图节点聚焦时直接高亮相连的兄弟边", () => {
  assert.equal(isGraphEdgeRelated("root.1.0", "root.1.1", "root.1.0"), true);
  assert.equal(isGraphEdgeRelated("root.1.2", "root.1.3", "root.1.0"), false);
});

test("图节点聚焦时高亮完整的数据流连通路径", () => {
  const edges = [
    { id: "a", kind: "dataflow", source: "root.1", target: "root.1.0" },
    { id: "b", kind: "dataflow", source: "root.1.0", target: "root.1.1" },
    { id: "c", kind: "dataflow", source: "root.1.1", target: "root.2" },
    { id: "d", kind: "dataflow", source: "root.3", target: "root.4" },
  ];
  assert.deepEqual([...relatedDataflowEdgeIds(edges, "root.1")], ["a", "b", "c"]);
});
