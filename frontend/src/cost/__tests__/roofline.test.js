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
  // M11-P0-4：迁移到 actions 形状（旧 macs/weightBytes 形状不再伪造 vector/sfu=0）
  const result = classifyRoofline({
    actions: { matrix: 100, vector: 0, sfu: 0, bytes: { weights: 100, actIn: 0, actOut: 0 }, commBytes: 10000 },
  }, CHIP);
  assert.equal(result.bound, "comm");
  assert.equal(result.times.matrix, 200 / 700); // W5-2：五路 max，矩阵单元取代旧"compute"单类
  assert.equal(result.times.memory, 100 / 70);
  assert.equal(result.times.comm, 10000 / 40);
  assert.equal(result.arithmeticIntensity, 2);
  assert.equal(result.ridgePoint, 1000 * 0.7 / (100 * 0.7));
});

test("缺少带宽或 MACs 时显式 unknown，不伪造分类", () => {
  const result = classifyRoofline({ macs: null, weightBytes: 100 }, { peak_flops: { bf16: 1000 } });
  assert.equal(result.bound, "unknown");
  // M11-P0-4：旧形状的 vector/sfu 是未知（null）；actIn/actOut 未知使
  // bytesMoved 未知（missing 记 bytes_moved 而非 memory_bandwidth）
  assert.deepEqual(result.missing, ["matrix", "vector", "sfu", "bytes_moved"]);
});

test("跨节点通信使用 inter_node 带宽与效率因子", () => {
  const result = classifyRoofline({
    actions: { matrix: 1, vector: 0, sfu: 0, bytes: { weights: 1, actIn: 0, actOut: 0 }, commBytes: 100 },
  }, CHIP, { interNode: true });
  assert.equal(result.times.comm, 100 / (20 * 0.6));
});

test("只有 memory 时间时不做不完整的 bound 分类", () => {
  const result = classifyRoofline({
    actions: { matrix: null, vector: null, sfu: null, bytes: { weights: 100, actIn: 0, actOut: 0 } },
  }, { memory_bandwidth: 100 });
  assert.equal(result.times.memory, 100 / 70);
  assert.equal(result.bound, "unknown");
});

test("存在通信量但缺链路带宽时 bound 为 unknown", () => {
  const result = classifyRoofline({
    actions: { matrix: 10, vector: 0, sfu: 0, bytes: { weights: 100, actIn: 0, actOut: 0 }, commBytes: 20 },
  }, { memory_bandwidth: 100, peak_flops: { bf16: 1000 } });
  assert.equal(result.bound, "unknown");
});

test("W5-2 同一动作向量换卡只走表乘法：时间比 = 费率倒数比", () => {
  const actions = { matrix: 1000, vector: 200, sfu: 0, bytes: { weights: 100, actIn: 50, actOut: 50 } };
  const cardA = { peak_flops: { bf16: 1000 }, vector_flops: 500, memory_bandwidth: 100 };
  const cardB = { peak_flops: { bf16: 2000 }, vector_flops: 1000, memory_bandwidth: 200 };
  const a = classifyRoofline({ actions }, cardA);
  const b = classifyRoofline({ actions }, cardB);
  // 费率翻倍 → 时间减半（效率因子相同）；无任何重新计算
  assert.equal(b.times.matrix, a.times.matrix / 2);
  assert.equal(b.times.vector, a.times.vector / 2);
  assert.equal(b.times.memory, a.times.memory / 2);
});

test("W5-2 softmax 定性分类：prefill 落 memory-bound，矩阵已知零不伪造 compute 主导", () => {
  // softmax：matrix=0（精确陈述），vector=exp 计数，bytes 激活流量
  const cost = { actions: { matrix: 0, vector: 48, sfu: 0, bytes: { weights: 0, actIn: 96, actOut: 96 } } };
  const chip = { peak_flops: { bf16: 1000 }, vector_flops: 500, memory_bandwidth: 100 };
  const result = classifyRoofline(cost, chip);
  assert.equal(result.bound, "memory");
  assert.equal(result.times.matrix, 0); // 已知零：0 是精确陈述（§3.3）
});

test("W5-2 sfu 单元参与五路 max", () => {
  const cost = { actions: { matrix: 10, vector: 0, sfu: 10000, bytes: { weights: 0, actIn: 0, actOut: 0 } } };
  const chip = { peak_flops: { bf16: 1000 }, sfu_ops: 1000 };
  const result = classifyRoofline(cost, chip);
  assert.equal(result.bound, "sfu");
  assert.equal(result.times.sfu, 10);
});
