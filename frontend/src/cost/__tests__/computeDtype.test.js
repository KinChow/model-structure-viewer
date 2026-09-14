// 逐 op compute-dtype（N2-1）：mHC 的 TF32 pre-GEMM 在 roofline 里的混合费率。
// 背景：Hopper/Blackwell + DeepGEMM 时 mHC pre-GEMM 在 TF32 tensor core 运行
//（bf16 激活 upcast，vLLM deepseek_v4 tilelang_kernels.py:686-711 取证）；
// 芯片表 tf32 行已就位（chips/public.js）。actions.matrix 是总量，
// actions.matrixTf32 是声明为 tf32 的子集，matrix 时间 = 两段费率之和。
import assert from "node:assert/strict";
import test from "node:test";
import { classifyRoofline } from "../roofline.js";

const CHIP = {
  id: "mixed-probe", memory_bytes: 80e9, memory_bandwidth: 2e12,
  peak_flops: { bf16: 312e12, tf32: 156e12 }, vector_flops: 19.5e12, sfu_ops: 4.875e12,
};

test("混合 dtype：矩阵时间 = bf16 段/312 + tf32 段/156（手算）", () => {
  // matrix 总量 1000，其中 400 声明 tf32：
  //   bf16 段 600 → 600/312e12·2（MACs→FLOPs 已在费率里折半）……直接用时间核对：
  //   matrixTime = 600 / (312e12/2) + 400 / (156e12/2)
  const { bound, times } = classifyRoofline(
    { actions: { matrix: 1000, matrixTf32: 400, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } } },
    CHIP,
    { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } },
  );
  // 矩阵时间 = bf16 段 600/(312e12/2) + tf32 段 400/(156e12/2)
  const expected = 600 / (312e12 / 2) + 400 / (156e12 / 2);
  assert.equal(times.matrix, expected);
  assert.equal(bound, "matrix");
});

test("无 tf32 桶：行为与旧口径逐位一致（回归守卫）", () => {
  const actions = { matrix: 1000, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } };
  const withKey = classifyRoofline({ actions: { ...actions, matrixTf32: 0 } }, CHIP, { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } });
  const without = classifyRoofline({ actions }, CHIP, { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } });
  assert.equal(withKey.times.matrix, without.times.matrix);
  assert.equal(withKey.bound, without.bound);
});

test("芯片无 tf32 行：整段回退全局费率，不制造 missing", () => {
  const chip = { peak_flops: { bf16: 312e12 }, memory_bandwidth: 2e12 };
  const { missing, times } = classifyRoofline(
    { actions: { matrix: 1000, matrixTf32: 400, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } } },
    chip,
    { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } },
  );
  assert.equal(times.matrix, 1000 / (312e12 / 2));
  assert.ok(!missing.some((m) => m.includes("tf32")));
});

test("computeDtypes.tf32 与扁平 matrixTf32 同口径（aggregate 生产形状）", () => {
  const expected = 600 / (312e12 / 2) + 400 / (156e12 / 2);
  const fromBucket = classifyRoofline(
    { actions: { matrix: 1000, computeDtypes: { tf32: 400 }, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } } },
    CHIP,
    { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } },
  );
  const fromFlat = classifyRoofline(
    { actions: { matrix: 1000, matrixTf32: 400, vector: 0, sfu: 0, bytes: { weights: 0, actIn: 0, actOut: 0 } } },
    CHIP,
    { dtype: "bf16", efficiency: { flops: 1, hbm: 1 } },
  );
  assert.equal(fromBucket.times.matrix, expected);
  assert.equal(fromBucket.times.matrix, fromFlat.times.matrix);
});
