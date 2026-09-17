import assert from "node:assert/strict";
import test from "node:test";
import katex from "katex";
import { FORMULAS } from "../structure/operators/formulas/index.js";
import { FORMULA_TEX, formulaTex } from "./formulaTex.js";

test("every registered formula has KaTeX-parseable LaTeX（双向覆盖 + 可解析）", () => {
  // 双向锁定：FORMULA_TEX 键集 == FORMULAS 键集（新增/删除算子必须同步）。
  assert.deepEqual(Object.keys(FORMULA_TEX).sort(), Object.keys(FORMULAS).sort());
  for (const [id, tex] of Object.entries(FORMULA_TEX)) {
    assert.equal(formulaTex(id), tex, `${id}: formulaTex lookup mismatch`);
    // throwOnError:true —— 任何不可解析的 LaTeX 立即红，逼迫转写正确。
    assert.doesNotThrow(
      () => katex.renderToString(tex, { throwOnError: true, displayMode: false }),
      `${id}: LaTeX 不可解析 -> ${tex}`,
    );
  }
});

test("formulaTex 未登记算子返回 null（调用方回退 ASCII）", () => {
  assert.equal(formulaTex("nonexistent_op"), null);
  assert.equal(formulaTex(""), null);
});
