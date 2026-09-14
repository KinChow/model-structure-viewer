import assert from "node:assert/strict";
import test from "node:test";
import { materializeStructureGraph } from "../../structure/graph/materializeStructureGraph.js";
import { memoryBreakdown } from "../memory.js";
import { classifyRoofline } from "../roofline.js";

test("KV bytes 使用用户可调假设，不计入无来源常数", () => {
  const graph = materializeStructureGraph({
    id: "model",
    children: [{ id: "layers.0.sdpa", attributes: { cache_kv_elements: 2 }, children: [] }],
  });
  const result = memoryBreakdown({ weightBytes: 0, graph, tokens: 2, kvBytes: 1, bufferBytes: 0 });
  assert.equal(result.kvBytes, 4);
  assert.equal(result.totalBytes, 4);
});

test("roofline 使用用户提供的效率因子", () => {
  // M11-P0-4：迁移到 actions 形状
  const result = classifyRoofline({
    actions: { matrix: 10, vector: 0, sfu: 0, bytes: { weights: 10, actIn: 0, actOut: 0 } },
  }, { peak_flops: { bf16: 100 }, memory_bandwidth: 10 }, { efficiency: { flops: 0.5, hbm: 0.5 } });
  assert.equal(result.times.matrix, 0.4); // W5-2：五路 max 命名
  assert.equal(result.times.memory, 2);
});
