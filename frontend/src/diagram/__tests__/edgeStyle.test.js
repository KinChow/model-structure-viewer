import assert from "node:assert/strict";
import test from "node:test";
import { edgePresentation, edgeStrokeWidth } from "../edgeStyle.js";

test("三类 evidence 的展示互不相同（§2.2）", () => {
  const source = { output_shape: [-1, -1, 4096] };
  const declared = edgePresentation({ evidence: "declared" }, source);
  const order = edgePresentation({ evidence: "module-order" }, source);
  const shape = edgePresentation({ evidence: "shape-match" }, source);
  assert.equal(new Set([declared.width, order.width, shape.width]).size, 3, "宽度三值互异");
  assert.equal(new Set([declared.className, order.className, shape.className]).size, 3, "类名三值互异");
  assert.equal(new Set([declared.hint, order.hint, shape.hint]).size, 3, "hover 提示三值互异");
  assert.ok(order.width < declared.width, "推断边不得比声明边醒目（§2.2）");
  assert.equal(declared.className, "", "declared 走默认类");
});

test("tensor 感知宽度：宽度随声明边源张量规模增长且有上界", () => {
  const small = edgePresentation({ evidence: "declared" }, { output_shape: [-1, -1, 512] });
  const large = edgePresentation({ evidence: "declared" }, { output_shape: [-1, -1, 65536] });
  assert.ok(large.width > small.width);
  assert.ok(large.width <= 2.8);
});

test("兼容出口 edgeStrokeWidth 与 presentation 宽度一致", () => {
  const edge = { evidence: "module-order" };
  assert.equal(edgeStrokeWidth(edge, {}), edgePresentation(edge, {}).width);
});
