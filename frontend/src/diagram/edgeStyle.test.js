import assert from "node:assert/strict";
import test from "node:test";
import { edgeStrokeWidth } from "./edgeStyle.js";

test("edgeStrokeWidth 弱化顺序推断边，声明边按张量规模取宽", () => {
  // §2.2：module-order 是推断，declared 是事实；推断边不得比声明边醒目。
  assert.equal(edgeStrokeWidth({ kind: "dataflow", evidence: "module-order" }, {}), 1.5);
  const declaredSmall = edgeStrokeWidth({ kind: "dataflow" }, { node: { output_shape: [-1, -1, 8] } });
  const declaredLarge = edgeStrokeWidth({ kind: "dataflow" }, { node: { output_shape: [1, 4096, 4096] } });
  assert.equal(declaredSmall, 1.8);
  assert.ok(declaredLarge > 2);
  // 即使是极小张量的声明边，也应不弱于推断边。
  assert.ok(declaredSmall > edgeStrokeWidth({ kind: "dataflow", evidence: "module-order" }, { node: { output_shape: [1, 4096, 4096] } }));
});
