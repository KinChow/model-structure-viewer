import assert from "node:assert/strict";
import test from "node:test";
import { isEdgeRelated, isGraphEdgeRelated, isPathRelated } from "./hover.js";

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
