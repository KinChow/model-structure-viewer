import assert from "node:assert/strict";
import test from "node:test";
import { computeNodeCosts, linearMacs } from "../compute.js";

test("packed qweight is unknown without logical shape metadata", () => {
  assert.equal(linearMacs({ weight_shapes: { qweight: [4, 1] } }, { batch: 1, sequence: 1 }), null);
});

test("MoE expert MACs use per-layer active fraction", () => {
  const root = { children: [{ id: "decoder.0.mlp.experts.0", weight_shapes: { weight: [4, 2] }, children: [] }] };
  const rows = computeNodeCosts(root, { experts: 8, expertsPerToken: 2, layerSchedule: ["moe"] }, { batch: 1, sequence: 1 });
  assert.equal(rows[1].macs, 2);
});
