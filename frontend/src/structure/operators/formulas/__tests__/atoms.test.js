// 18 原子的手算 exact 单测（W1）。
//
// 判据：每个原子一条测试，参数取 T=2 / d=2 / heads=1 量级，期望值**手算写死**，
// 不引用被测实现的任何常量。这是整套对账体系的地基——原子对 + 分解恒等
// => 叶模块对，把「41 算子 x 59 模型」的 2400 格降到 18 个手算点。
import assert from "node:assert/strict";
import test from "node:test";
import {
  ATOMS,
  add,
  conv1d,
  decayScan,
  div,
  evaluateDecomposition,
  gather,
  matmul,
  mul,
  permuteCopy,
  reduceMax,
  reduceSum,
  relu,
  rope,
  rsqrt,
  scale,
  scatter,
  sigmoid,
  silu,
  softmax,
  sumActions,
  topk,
} from "../atoms.js";

const B = 2; // bytesPerElement

function expect(actual, { matrix = 0, vector = 0, sfu = 0, weights = 0, actIn = 0, actOut = 0 }, label) {
  assert.deepEqual(actual, { matrix, vector, sfu, bytes: { weights, actIn, actOut } }, label);
}

test("A1 matmul: 激活 x 激活 [1,2,2]x[1,2,2]", () => {
  // MACs = 1·2·2·2 = 8；读 lhs 4 元素 + rhs 4 元素 = 8·2B = 16B；写 4 元素 = 8B
  expect(matmul({ batch: 1, m: 2, k: 2, n: 2, bytesPerElement: B }), { matrix: 8, actIn: 16, actOut: 8 });
});

test("A1 matmul: 激活 x 权重（权重不随 batch 复制）", () => {
  // batch=2, m=2, k=2, n=3 -> MACs = 2·2·2·3 = 24
  // 权重 k·n = 6 元素 = 12B（读一遍）；actIn = batch·m·k = 8 元素 = 16B
  // actOut = batch·m·n = 12 元素 = 24B
  expect(matmul({ batch: 2, m: 2, k: 2, n: 3, bytesPerElement: B, rhs: "weight" }),
    { matrix: 24, weights: 12, actIn: 16, actOut: 24 });
});

test("A1 matmul: density 只缩放 matrix 与输出（因果/块稀疏用）", () => {
  // batch=1,m=4,k=2,n=4,density=0.625（因果 10/16）-> MACs = 4·2·4·0.625 = 20
  // actIn = m·k + k·n = 8+8 = 16 元素 = 32B；actOut = 4·4·0.625 = 10 元素 = 20B
  expect(matmul({ batch: 1, m: 4, k: 2, n: 4, bytesPerElement: B, density: 0.625 }),
    { matrix: 20, actIn: 32, actOut: 20 });
});

test("A1 matmul: 调用方可声明真实操作数足迹（GQA 共享 K）", () => {
  // 4 个 query 头共享 2 个 KV 头：rhsElements 按 kvHeads 声明
  expect(matmul({ batch: 4, m: 2, k: 2, n: 3, bytesPerElement: B, lhsElements: 16, rhsElements: 12, outElements: 24 }),
    { matrix: 48, actIn: 56, actOut: 48 });
});

test("A2 softmax: 4 个元素", () => {
  // vector = 3·4 = 12；sfu = 2·4 = 8；读写各 4 元素 = 8B
  expect(softmax({ elements: 4, bytesPerElement: B }), { vector: 12, sfu: 8, actIn: 8, actOut: 8 });
});

test("A3 scale: 4 个元素乘常量", () => {
  expect(scale({ elements: 4, bytesPerElement: B }), { vector: 4, actIn: 8, actOut: 8 });
});

test("A4 add: 3 操作数 4 元素 = 2 次加法/元素", () => {
  // vector = 4·(3-1) = 8；actIn = 3·4·2B = 24B；actOut = 8B
  expect(add({ elements: 4, bytesPerElement: B, operands: 3 }), { vector: 8, actIn: 24, actOut: 8 });
});

test("A5 mul: 2 操作数 4 元素", () => {
  expect(mul({ elements: 4, bytesPerElement: B }), { vector: 4, actIn: 16, actOut: 8 });
});

test("A5 mul: 末操作数是学习参数时计入 weights 而非 actIn（norm weight）", () => {
  // elements=4（T·H）、weightElements=2（H）：读 x 4 元素=8B，权重 2 元素=4B
  expect(mul({ elements: 4, bytesPerElement: B, weightElements: 2 }),
    { vector: 4, weights: 4, actIn: 8, actOut: 8 });
});

test("A6 relu: 4 元素比较选择，无 SFU", () => {
  expect(relu({ elements: 4, bytesPerElement: B }), { vector: 4, actIn: 8, actOut: 8 });
});

test("A7 sigmoid: 4 元素 = 8 SFU，vector 为零（乘法归 mul 原子）", () => {
  expect(sigmoid({ elements: 4, bytesPerElement: B }), { sfu: 8, actIn: 8, actOut: 8 });
});

test("A8 silu: 4 元素 = 8 SFU + 4 乘", () => {
  expect(silu({ elements: 4, bytesPerElement: B }), { vector: 4, sfu: 8, actIn: 8, actOut: 8 });
});

test("A9 reduce_max: 8 元素分 2 组 = 每组 3 次比较", () => {
  // vector = 8-2 = 6；actIn = 8·2B = 16B；actOut = 2·2B = 4B
  expect(reduceMax({ elements: 8, groups: 2, bytesPerElement: B }), { vector: 6, actIn: 16, actOut: 4 });
});

test("A10 reduce_sum: 8 元素分 4 组 = 每组 1 次加法", () => {
  expect(reduceSum({ elements: 8, groups: 4, bytesPerElement: B }), { vector: 4, actIn: 16, actOut: 8 });
});

test("A11 rsqrt: 2 元素 = 2 SFU", () => {
  expect(rsqrt({ elements: 2, bytesPerElement: B }), { sfu: 2, actIn: 4, actOut: 4 });
});

test("A19 div: 4 元素 = 4 SFU（W1 补：首版原子表缺 div，topk 归一化无法表达）", () => {
  expect(div({ elements: 4, bytesPerElement: B }), { sfu: 4, actIn: 16, actOut: 8 });
});

test("A12 rope: 4 元素 = 12 flop，sfu 精确零（A3 查表）", () => {
  // actIn 含 sin/cos 同宽读入 = 2·4·2B = 16B；actOut = 4·2B = 8B
  expect(rope({ elements: 4, bytesPerElement: B }), { vector: 12, actIn: 16, actOut: 8 });
});

test("A13 gather: 2 行 x 宽 3 = 按真实拷贝元素计（G1）", () => {
  expect(gather({ rows: 2, width: 3, bytesPerElement: B }), { actIn: 12, actOut: 12 });
});

test("A13 gather: writeOut=false（KV cache 读进寄存器直接消费）", () => {
  expect(gather({ rows: 2, width: 3, bytesPerElement: B, writeOut: false }), { actIn: 12 });
});

test("A14 scatter: 与 gather 对称", () => {
  expect(scatter({ rows: 2, width: 3, bytesPerElement: B }), { actIn: 12, actOut: 12 });
});

test("A14 scatter: readIn=false（刚算出的 K/V 写回 cache）", () => {
  expect(scatter({ rows: 2, width: 3, bytesPerElement: B, readIn: false }), { actOut: 12 });
});

test("A15 permute_copy: 6 元素物化拷贝", () => {
  expect(permuteCopy({ elements: 6, bytesPerElement: B }), { actIn: 12, actOut: 12 });
});

test("A16 conv1d: T=2 channels=2 kernel=2，无 state", () => {
  // MACs = 2·2·2 = 8；weights = 2·2 = 4 元素 = 8B；actIn/actOut = 2·2 = 4 元素 = 8B
  expect(conv1d({ tokens: 2, channels: 2, kernel: 2, bytesPerElement: B }),
    { matrix: 8, weights: 8, actIn: 8, actOut: 8 });
});

test("A16 conv1d: decode 相位带 conv state（channels·(kernel-1)=2）", () => {
  expect(conv1d({ tokens: 1, channels: 2, kernel: 2, bytesPerElement: B, stateElements: 2 }),
    { matrix: 4, weights: 8, actIn: 8, actOut: 8 });
});

test("A17 topk: 2 行 x 4 候选选 2，写出 values + int32 索引", () => {
  // vector = 2·4 = 8；actIn = 8 元素 = 16B
  // actOut = values 2·2·2B + indices 2·2·4B = 8+16 = 24B
  expect(topk({ rows: 2, candidates: 4, k: 2, bytesPerElement: B }), { vector: 8, actIn: 16, actOut: 24 });
});

test("A17 topk: k 大于候选数时按候选数截断", () => {
  // selected = 1·2；actOut = 2·2B + 2·4B = 12
  expect(topk({ rows: 1, candidates: 2, k: 8, bytesPerElement: B }), { vector: 2, actIn: 4, actOut: 12 });
});

test("A18 decay_scan: 2 步 x state 4，单头每步 1 次 exp", () => {
  // vector = 2·4 = 8；sfu = 2·1·1 = 2；actIn = 2·2·4·2B = 32B；actOut = 2·4·2B = 16B
  expect(decayScan({ steps: 2, state: 4, heads: 1, bytesPerElement: B }),
    { vector: 8, sfu: 2, actIn: 32, actOut: 16 });
});

test("原子注册表恰好 19 条，且 evaluateDecomposition 与手工求和一致", () => {
  assert.equal(Object.keys(ATOMS).length, 19);
  const steps = [
    { atom: "matmul", args: { batch: 1, m: 2, k: 2, n: 2, bytesPerElement: B } },
    { atom: "softmax", args: { elements: 4, bytesPerElement: B } },
  ];
  assert.deepEqual(
    evaluateDecomposition(steps),
    sumActions(matmul({ batch: 1, m: 2, k: 2, n: 2, bytesPerElement: B }), softmax({ elements: 4, bytesPerElement: B })),
  );
});

test("evaluateDecomposition 对未注册原子直接报错，不静默记零", () => {
  assert.throws(() => evaluateDecomposition([{ atom: "flash_attention", args: {} }]), /unknown atom/);
});
