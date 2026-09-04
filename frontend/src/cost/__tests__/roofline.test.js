import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_EFFICIENCY, resolveEfficiency } from "../efficiency.js";
import { classifyRoofline } from "../roofline.js";

const CHIP = {
  memory_bandwidth: 100,
  peak_flops: { bf16: 1000 },
  interconnect: { intra_node: { bandwidth: 50 }, inter_node: { bandwidth: 20 } },
};

test("效率因子使用文献默认值并支持芯片覆盖", () => {
  assert.deepEqual(resolveEfficiency(), DEFAULT_EFFICIENCY);
  assert.equal(resolveEfficiency({ efficiency: { flops: 0.5 } }).flops, 0.5);
  assert.equal(resolveEfficiency({ efficiency: { flops: 2 } }).flops, DEFAULT_EFFICIENCY.flops);
});

test("F17 三类时间各除对应效率因子后取最大值作为 bound", () => {
  const result = classifyRoofline({ macs: 100, weightBytes: 100, actInBytes: 0, actOutBytes: 0, commBytes: 10000 }, CHIP);
  assert.equal(result.bound, "comm");
  assert.equal(result.times.compute, 200 / 700);
  assert.equal(result.times.memory, 100 / 90);
  assert.equal(result.times.comm, 10000 / 40);
  assert.equal(result.arithmeticIntensity, 2);
  assert.equal(result.ridgePoint, 1000 * 0.7 / (100 * 0.9));
});

test("缺少带宽或 MACs 时显式 unknown，不伪造分类", () => {
  const result = classifyRoofline({ macs: null, weightBytes: 100 }, { peak_flops: { bf16: 1000 } });
  assert.equal(result.bound, "unknown");
  assert.deepEqual(result.missing, ["macs", "memory_bandwidth"]);
});

test("跨节点通信使用 inter_node 带宽与效率因子", () => {
  const result = classifyRoofline({ macs: 1, weightBytes: 1, commBytes: 100 }, CHIP, { interNode: true });
  assert.equal(result.times.comm, 100 / (20 * 0.6));
});

test("只有 memory 时间时不做不完整的 bound 分类", () => {
  const result = classifyRoofline({ macs: null, weightBytes: 100 }, { memory_bandwidth: 100 });
  assert.equal(result.times.memory, 100 / 90);
  assert.equal(result.bound, "unknown");
});

test("存在通信量但缺链路带宽时 bound 为 unknown", () => {
  const result = classifyRoofline({ macs: 10, weightBytes: 100, commBytes: 20 }, { memory_bandwidth: 100, peak_flops: { bf16: 1000 } });
  assert.equal(result.bound, "unknown");
});
