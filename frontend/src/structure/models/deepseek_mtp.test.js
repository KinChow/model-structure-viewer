import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { buildStructureFromArtifacts } from "../buildStructure.js";
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

test("DeepSeek-style MTP binds tail-layer checkpoint paths to the draft branch", () => {
  for (const [modelId, layerIndex] of [
    ["deepseek-ai/DeepSeek-R1", 61],
    ["zai-org/GLM-4.7", 92],
  ]) {
    const dir = new URL(`../../../../models/${modelId}/`, import.meta.url);
    const config = JSON.parse(fs.readFileSync(new URL("config.json", dir), "utf8"));
    const prefix = `model.layers.${layerIndex}`;
    const tensors = [
      [`${prefix}.embed_tokens.weight`, [config.vocab_size, config.hidden_size]],
      [`${prefix}.enorm.weight`, [config.hidden_size]],
      [`${prefix}.hnorm.weight`, [config.hidden_size]],
      [`${prefix}.eh_proj.weight`, [config.hidden_size, config.hidden_size * 2]],
      [`${prefix}.input_layernorm.weight`, [config.hidden_size]],
      [`${prefix}.shared_head.norm.weight`, [config.hidden_size]],
      [`${prefix}.shared_head.head.weight`, [config.vocab_size, config.hidden_size]],
    ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
    const { graph } = buildStructureFromArtifacts({
      config,
      modelId,
      checkpointTruth: { tensors, mtp_tensor_count: tensors.length },
    });
    const owners = new Map();
    for (const tensor of tensors) {
      const matches = graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
      assert.equal(matches.length, 1, `${modelId}: ${tensor.name}`);
      owners.set(tensor.name, matches[0].canonical_id);
    }
    assert.equal(owners.get(`${prefix}.embed_tokens.weight`), "mtp.embed_tokens");
    assert.equal(owners.get(`${prefix}.enorm.weight`), "mtp.enorm");
    assert.equal(owners.get(`${prefix}.hnorm.weight`), "mtp.hnorm");
    assert.equal(owners.get(`${prefix}.eh_proj.weight`), "mtp.eh_proj");
    assert.equal(owners.get(`${prefix}.input_layernorm.weight`), "mtp.layer.input_layernorm");
    assert.equal(owners.get(`${prefix}.shared_head.norm.weight`), "mtp.shared_head.norm");
    assert.equal(owners.get(`${prefix}.shared_head.head.weight`), "mtp.shared_head.head");
  }
});

test("GLM-5.3-Flash MTP binds the published tail layer for both released variants", () => {
  for (const modelId of ["zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"]) {
    const dir = new URL(`../../../../models/${modelId}/`, import.meta.url);
    const config = JSON.parse(fs.readFileSync(new URL("config.json", dir), "utf8"));
    const prefix = `model.language_model.layers.${config.text_config.num_hidden_layers}`;
    const tensors = [
      [`${prefix}.enorm.weight`, [config.text_config.hidden_size]],
      [`${prefix}.hnorm.weight`, [config.text_config.hidden_size]],
      [`${prefix}.eh_proj.weight`, [config.text_config.hidden_size, config.text_config.hidden_size * 2]],
      [`${prefix}.input_layernorm.weight`, [config.text_config.hidden_size]],
      [`${prefix}.post_attention_layernorm.weight`, [config.text_config.hidden_size]],
      [`${prefix}.self_attn.q_a_layernorm.weight`, [config.text_config.q_lora_rank]],
      [`${prefix}.mlp.gate.weight`, [config.text_config.n_routed_experts, config.text_config.hidden_size]],
      [`${prefix}.shared_head.norm.weight`, [config.text_config.hidden_size]],
    ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
    const structure = buildStructureFromArtifacts({
      config,
      modelId,
      checkpointTruth: { tensors, mtp_tensor_count: tensors.length },
    });
    const { graph } = structure;
    for (const tensor of tensors) {
      const matches = graph.nodes.filter(node => node.tensor_names?.includes(tensor.name));
      assert.equal(matches.length, 1, `${modelId}: ${tensor.name}`);
      assert.ok(matches[0].canonical_id.startsWith("mtp."), `${modelId}: ${tensor.name}`);
    }
    assert.deepEqual(structure.source.diagnostics.template_gaps, []);
    assert.equal(graph.nodes.some(node => node.canonical_id === "mtp.embed_tokens"), false);
    assert.equal(graph.nodes.some(node => node.canonical_id === "mtp.shared_head.head"), false);
    assert.equal(graph.nodes.some(node => node.canonical_id === "mtp.layer.mhc_attn_pre"), false);
    assert.equal(graph.nodes.some(node => node.canonical_id === "mtp.layer.mhc_ffn_pre"), false);
    assert.ok(graph.nodes.some(node => node.canonical_id === "mtp.layer.input_layernorm"));
    assert.ok(graph.nodes.some(node => node.canonical_id === "mtp.layer.post_attention_layernorm"));
  }
});

test("GLM-5.3-Flash aggregates published MTP expert tensors onto the fused expert leaf", () => {
  const modelId = "zai-org/GLM-5.3-Flash-BF16";
  const dir = new URL(`../../../../models/${modelId}/`, import.meta.url);
  const config = JSON.parse(fs.readFileSync(new URL("config.json", dir), "utf8"));
  const prefix = `model.language_model.layers.${config.text_config.num_hidden_layers}.mlp.experts`;
  const tensors = [
    [`${prefix}.0.gate_proj.weight`, [config.text_config.moe_intermediate_size, config.text_config.hidden_size]],
    [`${prefix}.0.up_proj.weight`, [config.text_config.moe_intermediate_size, config.text_config.hidden_size]],
    [`${prefix}.1.down_proj.weight`, [config.text_config.hidden_size, config.text_config.moe_intermediate_size]],
  ].map(([name, shape]) => ({ name, shape, dtype: "BF16" }));
  const structure = buildStructureFromArtifacts({
    config,
    modelId,
    checkpointTruth: { tensors, mtp_tensor_count: tensors.length },
  });
  const expert = structure.graph.nodes.find(node => node.canonical_id === "mtp.layer.mlp.expert_mlp");
  assert.ok(expert);
  assert.deepEqual(structure.source.diagnostics.template_gaps, []);
  assert.deepEqual(expert.tensor_names.sort(), tensors.map(tensor => tensor.name).sort());
  assert.equal(expert.params, tensors.reduce((sum, tensor) => sum + tensor.shape.reduce((a, b) => a * b, 1), 0));
});
