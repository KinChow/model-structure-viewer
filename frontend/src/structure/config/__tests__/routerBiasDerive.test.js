// 派生标志 routerCorrectionBias / routerBiasVl 的行为守护。
// correction bias ⇔ topk_method=="noaux_tc" 或 sigmoid 路由（scoring_func sigmoid / 配方 sigmoidRouter）。
// bias_vl ⇔ 修正 bias × 视觉塔 × 存在 compress_ratios——checkpoint 实证 V4.1-Flash /
// V4-Flash-Vision-Exp 有、V4-Flash-0731(无视觉)/MiniMax-M3(无 compress_ratios 视觉) 无。
import assert from "node:assert/strict";
import test from "node:test";
import { normalizeConfig } from "../normalize.js";

test("routerBiasVl：有 compress_ratios + 视觉塔（V4.1 / V4-Flash-Vision）两者真", () => {
  for (const arch of ["DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM"]) {
    const n = normalizeConfig({
      architectures: [arch],
      topk_method: "noaux_tc",
      scoring_func: "sqrtsoftplus",
      compress_ratios: [0, 4, 128],
      vision_config: { hidden_size: 1024, num_hidden_layers: 8 },
    });
    assert.equal(n.routerCorrectionBias, true, arch);
    assert.equal(n.routerBiasVl, true, arch);
  }
});

test("routerBiasVl：有 compress_ratios 无视觉（V4-Flash-0731）correction真/vl省略", () => {
  const n = normalizeConfig({
    architectures: ["DeepseekV4ForCausalLM"],
    topk_method: "noaux_tc",
    scoring_func: "sqrtsoftplus",
    compress_ratios: [0, 4, 128],
  });
  assert.equal(n.routerCorrectionBias, true);
  assert.equal(n.routerBiasVl, undefined);
});

test("routerBiasVl：无 compress_ratios 的视觉 MoE（MiniMax-M3 类，sigmoid+视觉）vl省略", () => {
  const n = normalizeConfig({
    architectures: ["MiniMaxM3SparseForConditionalGeneration"],
    scoring_func: "sigmoid",
    vision_config: { hidden_size: 1024, num_hidden_layers: 8 },
  });
  assert.equal(n.routerCorrectionBias, true);
  assert.equal(n.routerBiasVl, undefined);
});

test("routerCorrectionBias 派生：V3(noaux_tc, 无视觉) correction真/vl省略", () => {
  const n = normalizeConfig({
    architectures: ["DeepseekV3ForCausalLM"],
    model_type: "deepseek_v3",
    topk_method: "noaux_tc",
    scoring_func: "sigmoid",
  });
  assert.equal(n.routerCorrectionBias, true);
  assert.equal(n.routerBiasVl, undefined);
});

test("routerCorrectionBias/routerBiasVl 派生：非 noaux_tc(Qwen) 两者省略", () => {
  const n = normalizeConfig({
    architectures: ["Qwen3MoeForCausalLM"],
    model_type: "qwen3_moe",
  });
  assert.equal(n.routerCorrectionBias, undefined);
  assert.equal(n.routerBiasVl, undefined);
});

test("routerCorrectionBias 派生：sigmoid 路由(scoring_func) correction真/vl省略（MiniMax-M2）", () => {
  const n = normalizeConfig({
    architectures: ["MiniMaxM2ForCausalLM"],
    model_type: "minimax_m2",
    scoring_func: "sigmoid",
  });
  assert.equal(n.routerCorrectionBias, true);
  assert.equal(n.routerBiasVl, undefined);
});

test("routerCorrectionBias 派生：配方 sigmoidRouter(GLM4-MoE，topk_method 缺省) correction真", () => {
  const n = normalizeConfig({
    architectures: ["Glm4MoeForCausalLM"],
    model_type: "glm4_moe",
  });
  assert.equal(n.routerCorrectionBias, true);
  assert.equal(n.routerBiasVl, undefined);
});
