import assert from "node:assert/strict";
import test from "node:test";
import { dsparkLayerCount, mtpModuleCount } from "./deepseek_mtp.js";
import { assembleDeepseekV4 } from "./deepseek_v4.js";
import { assembleQwen3_5 } from "./qwen3_5.js";
import { assembleQwen4Exp } from "./qwen4_exp.js";
import { assembleDeepseekV3 } from "./deepseek_v3.js";

test("DSpark config suppresses MTP module count", () => {
  assert.equal(mtpModuleCount({ mtpModules: 1, dsparkTargetLayerIds: [40, 41, 42] }), 0);
  assert.equal(mtpModuleCount({ mtpModules: 1, dsparkTargetLayerIds: [] }), 1);
  assert.equal(dsparkLayerCount({ dsparkTargetLayerIds: [58, 59, 60] }), 3);
  assert.equal(dsparkLayerCount({ dsparkTargetLayerIds: [] }), 0);
});

test("checkpoint 真值抑制幻影 MTP：config 声明 MTP 但权重无 MTP 张量则不计", () => {
  // MiniMax-M3 / M2.7：config num_nextn_predict_layers>0，但 checkpoint mtp_tensor_count===0。
  assert.equal(mtpModuleCount({ mtpModules: 1, checkpointMtpTensorCount: 0 }), 0);
  assert.equal(mtpModuleCount({ mtpModules: 3, checkpointMtpTensorCount: 0 }), 0);
  // checkpoint 里确有 MTP 张量 → 保留 config 声明的模块数（DeepSeek/GLM/Qwen）。
  assert.equal(mtpModuleCount({ mtpModules: 1, checkpointMtpTensorCount: 5 }), 1);
  // 真值缺席（离线 golden / 取证失败）→ 信任 config，向后兼容。
  assert.equal(mtpModuleCount({ mtpModules: 1 }), 1);
  assert.equal(mtpModuleCount({ mtpModules: 2, checkpointMtpTensorCount: undefined }), 2);
});

test("各架构文件自己挂对应 vLLM 投机头，不经 draftClassOf 分派", () => {
  const resolved = { architecture: "test" };
  const treeClass = (assemble, normalized) => assemble(resolved, normalized).children.find((n) => n.id === "mtp")?.attributes.class;

  assert.equal(treeClass(assembleDeepseekV4, { dsparkTargetLayerIds: [40, 41, 42], mtpModules: 1, hiddenSize: 8, layers: 1 }), "DSparkDeepseekV4Model");
  assert.equal(treeClass(assembleDeepseekV4, { mtpModules: 1, compressRatios: [0, 4, 128], hiddenSize: 8, layers: 1 }), "DeepSeekV4MultiTokenPredictorLayer");
  assert.equal(treeClass(assembleQwen4Exp, { mtpModules: 1, hyperConnectionCount: 4, hiddenSize: 8, layers: 1, experts: 8 }), "Qwen4ExpMultiTokenPredictor");
  assert.equal(treeClass(assembleQwen3_5, { mtpModules: 1, hiddenSize: 8, layers: 1 }), "Qwen3_5MultiTokenPredictor");
  assert.equal(treeClass(assembleDeepseekV3, { mtpModules: 1, kvLoraRank: 512, hiddenSize: 8, layers: 1 }), "DeepSeekMultiTokenPredictorLayer");
  assert.equal(treeClass(assembleDeepseekV3, { mtpModules: 0, hiddenSize: 8, layers: 1 }), undefined);
});
