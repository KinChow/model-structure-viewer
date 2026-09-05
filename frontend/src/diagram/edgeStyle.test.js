import assert from "node:assert/strict";
import test from "node:test";
import { edgeStrokeWidth } from "./edgeStyle.js";

test("edgeStrokeWidth keeps main flow prominent and scales known tensors", () => {
  assert.equal(edgeStrokeWidth({ kind: "structure" }, {}), 1.5);
  assert.equal(edgeStrokeWidth({ kind: "dataflow", evidence: "module-order" }, {}), 2.4);
  assert.equal(edgeStrokeWidth({ kind: "dataflow" }, { node: { output_shape: [-1, -1, 8] } }), 1.8);
  assert.ok(edgeStrokeWidth({ kind: "dataflow" }, { node: { output_shape: [1, 4096, 4096] } }) > 2);
});
