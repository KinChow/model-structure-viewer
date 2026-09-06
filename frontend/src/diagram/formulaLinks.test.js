import assert from "node:assert/strict";
import test from "node:test";
import { collectFormulaLinks } from "./formulaLinks.js";

test("收集公式节点路径并保留解释", () => {
  const links = collectFormulaLinks({ attributes: {}, children: [{ attributes: { formula_id: "linear", explanation: "矩阵乘" }, children: [] }] });
  assert.deepEqual(links, [{ path: "root.0", formulaId: "linear", explanation: "矩阵乘" }]);
});

test("Graph IR formula links use stable node paths", () => {
  const links = collectFormulaLinks({
    nodes: [{ id: "root.2.0", attributes: { formula_id: "attention", explanation: "QK" } }],
  });
  assert.deepEqual(links, [{ path: "root.2.0", formulaId: "attention", explanation: "QK" }]);
});
