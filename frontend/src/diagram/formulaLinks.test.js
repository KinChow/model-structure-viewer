import assert from "node:assert/strict";
import test from "node:test";
import { collectFormulaLinks } from "./formulaLinks.js";

test("收集公式节点路径并保留解释", () => {
  const links = collectFormulaLinks({ attributes: {}, children: [{ attributes: { formula_id: "linear", explanation: "矩阵乘" }, children: [] }] });
  assert.deepEqual(links, [{ path: "root.0", formulaId: "linear", explanation: "矩阵乘" }]);
});
