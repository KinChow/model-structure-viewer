import assert from "node:assert/strict";
import test from "node:test";
import { draftClassOf, dsparkLayerCount, mtpModuleCount } from "./mtp.js";

test("DSpark config suppresses MTP module count", () => {
  assert.equal(mtpModuleCount({ mtpModules: 1, dsparkTargetLayerIds: [40, 41, 42] }), 0);
  assert.equal(mtpModuleCount({ mtpModules: 1, dsparkTargetLayerIds: [] }), 1);
  assert.equal(dsparkLayerCount({ dsparkTargetLayerIds: [58, 59, 60] }), 3);
  assert.equal(dsparkLayerCount({ dsparkTargetLayerIds: [] }), 0);
});

test("draftClassOf 按字段分派到 vLLM 类名，不按家族名", () => {
  assert.equal(draftClassOf({ dsparkTargetLayerIds: [40, 41, 42], mtpModules: 1 }), "DSparkDeepseekV4Model");
  assert.equal(draftClassOf({ mtpModules: 1, compressRatios: [0, 4, 128] }), "DeepSeekV4MultiTokenPredictorLayer");
  assert.equal(draftClassOf({ mtpModules: 1, hyperConnectionCount: 4, linearAttentionMode: "qwen4_exp" }), "Qwen4ExpMultiTokenPredictor");
  assert.equal(draftClassOf({ mtpModules: 1, linearAttentionMode: "qwen3_5" }), "Qwen3_5MultiTokenPredictor");
  assert.equal(draftClassOf({ mtpModules: 1, kvLoraRank: 512 }), "DeepSeekMultiTokenPredictorLayer");
  assert.equal(draftClassOf({ mtpModules: 0 }), null);
});
