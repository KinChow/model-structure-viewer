import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec, weightMatrixDecl } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass } from "../archs/index.js";
import { ngramEmbeddingModule } from "./embedding.js";

// P4-2：长尾复合算子的权重声明。每组的 shape/quantizable/param_dtype 与
// formulas/index.js 对应 counts 的组成逐项同源（锚 1 执法），分片亲和按
// vLLM 源码取证：
// - hyper_connection：vLLM qwen4_exp/common/hyperconnection.py:176-193 用的是
//   **raw nn.Linear**（注释原文 "raw Linear weights (checkpoint-compatible)"，
//   无 ColumnParallel/RowParallel 包装）→ 每卡完整持有 = replicated；
// - mHC（deepseek_v4/amd/model.py:712-753）：hc_*_fn/base/scale 全是裸
//   nn.Parameter（torch.float32，无 weight_loader 分片）→ replicated；
// - KDA/GDN 衰减参数：kimi_gdn_linear_attn.py:241,268 用 sharded_weight_loader /
//   a_log_weight_loader 沿头维切 → tp。

/** mHC 的 fn [mix_hc, hc_dim]：mix_hc = (2+hc_mult)·hc_mult、hc_dim = hc_mult·H
 *  （vLLM deepseek_v4/amd/model.py:709-711，与 extractor 的 mhcMixRows/mhcDim 同式）。 */
function mhcGroups(normalized, hiddenWidth) {
  const mult = normalized.mhcNumResidualStreams || 0;
  const mixRows = (2 + mult) * mult;
  const hcDim = mult * hiddenWidth;
  return [
    weightMatrixDecl("replicated", { shape: [mixRows, hcDim], param_dtype: "mhc_fn", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [mixRows], param_dtype: "mhc_base", quantizable: false }),
    weightMatrixDecl("replicated", { shape: [3], param_dtype: "mhc_scale", quantizable: false }),
    // attn_norm / ffn_norm 的 RMSNorm 权重融进 mhc 内核（model.py:704-705），
    // 结构树里没有独立 norm 叶 —— bf16，记在本叶（extractor mhc ctx 的 norm 项）。
    weightMatrixDecl("replicated", { shape: [hiddenWidth], quantizable: false }),
  ];
}

export function hyperConnectionModule(id, normalized, phase = "branch") {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const names = {
    attn_mix: "HyperConnection attention mix",
    mlp_combine_mix: "HyperConnection MLP combine + mix",
    final: "HyperConnection final mixer",
    branch: id.split(".").at(-1) === "mixer" ? "Hyper Connection Mixer" : id.split(".").at(-1).replaceAll("_", " "),
  };
  return withShapeDims(moduleSpec(
    id,
    names[phase] || names.branch,
    "hyper-connection",
    {
      class: hfNamedClass(normalized, "gatedResidualClass", "GatedResidual"),
      hc_phase: phase,
      hc_count: normalized.hyperConnectionCount,
      hc_lowrank: normalized.hyperConnectionLowrank,
      state_handoff: phase === "mlp_combine_mix" ? "from_previous_layer" : phase === "attn_mix" ? "to_next_layer" : undefined,
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [operatorSpec(`${id}.${phase}`, names[phase] || "hyper-connection mix", "hyper_connection", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}, injection`, shapes.hidden),
      hc_phase: phase,
      hc_count: normalized.hyperConnectionCount,
      hc_lowrank: normalized.hyperConnectionLowrank,
      // GatedResidual 的 use_combine（vLLM qwen4_exp/common/hyperconnection.py:157-158、
      // 188-193）：最终 mixer 只做 mix（把多流收成单流），没有 combine，因此
      // **没有** block_inject_weight（hc_count × hyper_hidden）。此前一律按有
      // combine 记，Flash-Next 多算 40,960 参数（2026-09-09 权重字节逐层归因）。
      hc_use_combine: phase !== "final",
      // P4-2：hc_norm[grouped] + W_down + W_up + W_inject 的组成与 extractor 的
      // hyper_connection ctx 逐项同源；raw nn.Linear 无并行包装 → replicated。
      weightMatrices: [
        weightMatrixDecl("replicated", { shape: [(normalized.hyperConnectionCount || 1) * (normalized.hiddenSize || 0)], quantizable: false }),
        weightMatrixDecl("replicated", { shape: [normalized.hyperConnectionLowrank || 0, (normalized.hyperConnectionCount || 1) * (normalized.hiddenSize || 0)], quantizable: false }),
        weightMatrixDecl("replicated", { shape: [(normalized.hyperConnectionCount || 1) * (normalized.hiddenSize || 0), normalized.hyperConnectionLowrank || 0], quantizable: false }),
        ...(phase !== "final"
          ? [weightMatrixDecl("replicated", { shape: [normalized.hyperConnectionCount || 0, (normalized.hyperConnectionCount || 1) * (normalized.hiddenSize || 0)], quantizable: false })]
          : []),
      ],
    }, { input: dims.hidden, output: dims.hidden })],
  ), dims.hidden, dims.hidden);
}

export function pleModule(id, normalized, { layerIndex = 0 } = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const found = (normalized.pleLayerIds || []).indexOf(layerIndex + 1);
  const pleLayerIndex = found >= 0 ? found : 0;
  const embedOut = [-1, -1, normalized.pleEmbedDim || 0];
  return withShapeDims(moduleSpec(
    id,
    "PLE",
    "ple",
    {
      class: hfNamedClass(normalized, "pleClass", "PLELayer"),
      embed_dim: normalized.pleEmbedDim,
      ngram_size: normalized.pleNgramSize,
      heads_per_ngram: normalized.pleHeadsPerNgram,
      conv_kernel_size: normalized.pleConvKernelSize,
      conv_dilation: normalized.pleNgramSize,
      key_projection_size: normalized.hiddenSize * (normalized.hyperConnectionCount || 1),
      value_projection_size: normalized.hiddenSize,
      implementation: ["ngram_embedding", "kv_proj", "grouped_norm", "gated_output", "dilated_short_conv"],
      dataflow_edges: [["ple_embedding", "inject"]],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      withShapeDims(moduleSpec(
        `${id}.ple_embedding`,
        "PLE ngram embedding",
        "ngram-embedding",
        {
          class: "Qwen4ExpTextNGramEmbedding",
          ...shapeFlow(shapes.tokenIds, shapes.hidden),
        },
        [ngramEmbeddingModule(`${id}.ple_embedding.ngram_embedding`, normalized, { pleLayerIndex })],
      ), dims.tokenIds, embedOut),
      operatorSpec(`${id}.inject`, "PLE injection", "ple", {
        ...shapeFlow(`${shapes.hidden}, input_ids, ngram_context`, shapes.hidden),
        embed_dim: normalized.pleEmbedDim,
        // P4-2：inject 叶只声明 W_kv / conv / norm。ngram 表是
        // `ple.ple_embedding.ngram_embedding` 的 nn.Embedding（modeling_qwen4_exp.py:1111），
        // 容量走 type=embedding 子叶，不进本叶 weightMatrices。
        weightMatrices: [
          weightMatrixDecl("tp", { shape: [2 * (normalized.pleEmbedDim || 0), normalized.hiddenSize || 0], split: "output", quantizable: false }),
          weightMatrixDecl("tp", { shape: [normalized.pleEmbedDim || 0, normalized.pleNgramSize || 1], split: "output", quantizable: false }),
          weightMatrixDecl("replicated", { shape: [normalized.pleEmbedDim || 0], quantizable: false }),
        ],
      }, { input: dims.hidden, output: dims.hidden }),
    ],
  ), dims.hidden, dims.hidden);
}

export function sharedExpertGateModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "Shared Expert Gate",
    "shared-expert-gate",
    { class: "SharedExpertGate", ...shapeFlow(shapes.hidden, shapes.hidden) },
    [operatorSpec(`${id}.gate`, "shared expert gate", "shared_expert_gate", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden })],
  ), dims.hidden, dims.hidden);
}

export function multiHyperConnectionModule(id, normalized, phase = "pre") {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const streams = normalized.mhcNumResidualStreams;
  const formulaByPhase = {
    pre: "mhc_pre",
    fused_post_pre: "mhc_fused_post_pre",
    post: "mhc_post",
    contract: "mhc_contract",
  };
  const names = {
    pre: "mHC attention pre",
    fused_post_pre: "mHC fused post + FFN pre",
    post: "mHC final post",
    contract: "mHC contract",
  };
  const formulaId = formulaByPhase[phase] || formulaByPhase.pre;
  const inputShape = phase === "pre"
    ? `[residual streams=${streams}, ${shapes.hidden}]`
    : phase === "contract"
      ? `[residual streams=${streams}, ${shapes.hidden}]`
      : `${shapes.hidden}, [residual streams=${streams}, ${shapes.hidden}]`;
  const outputShape = phase === "post" ? `[residual streams=${streams}, ${shapes.hidden}]` : shapes.hidden;
  const numericInput = phase === "pre" || phase === "contract" ? [-1, streams, ...dims.hidden.slice(1)] : dims.hidden;
  const numericOutput = phase === "post" ? [-1, streams, ...dims.hidden.slice(1)] : dims.hidden;
  return withShapeDims(moduleSpec(
    id,
    names[phase] || id.split(".").at(-1).replaceAll("_", " "),
    "multi-hyper-connection",
    {
      class: phase === "fused_post_pre" ? "MHCFusedPostPreOp" : phase === "post" ? "MHCPostOp" : phase === "contract" ? "HCContract" : "MHCPreOp",
      mhc_phase: phase,
      streams,
      sinkhorn_iterations: normalized.mhcSinkhornIterations,
      tau: normalized.mhcTau,
      hc_eps: normalized.mhcEps,
      post_mult_value: normalized.mhcPostMultValue,
      state_handoff: phase === "fused_post_pre" ? "from_previous_layer" : phase === "pre" ? "to_next_layer" : undefined,
      ...shapeFlow(inputShape, outputShape),
    },
    [operatorSpec(`${id}.${phase}`, names[phase] || "mHC operation", formulaId, {
      ...shapeFlow(inputShape, outputShape),
      streams,
      sinkhorn_iterations: normalized.mhcSinkhornIterations,
      tau: normalized.mhcTau,
      hc_eps: normalized.mhcEps,
      post_mult_value: normalized.mhcPostMultValue,
      // P4-2：pre 与 fused_post_pre 持有 hc_*_fn/base/scale + 融合 norm 的权重
      // （extractor mhc ctx 的 matrix/base/scale/norm 项）；post 的 combine 是
      // weightsShared（复用最后一层 ffn 参数，vLLM model.py:1074-1097），
      // contract 无自有权重 —— 两者的 counts.bytes.weights=0，不在覆盖判据内。
      ...(phase === "pre" || phase === "fused_post_pre"
        ? { weightMatrices: mhcGroups(normalized, normalized.hiddenSize || 0) }
        : {}),
    }, { input: numericInput, output: numericOutput })],
  ), dims.hidden, dims.hidden);
}
