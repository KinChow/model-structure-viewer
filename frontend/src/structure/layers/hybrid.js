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
  // PLE 投影/归一化宽度（对齐 transformers modular_qwen4_exp.py:730-742）：
  //   key_proj  : Linear(ple_embed_dim → hc_hidden = hidden·hc_count)
  //   value_proj: Linear(ple_embed_dim → hidden)
  //   norm_key / norm_query / norm_conv: 各 RMSNorm(hc_hidden)
  //   conv1d    : depthwise Conv1d(hc_hidden, hc_hidden, kernel=ple_conv_kernel_size, groups=hc_hidden)
  // checkpoint 实证（Qwen3.8-Flash-Next）：key_proj[10240,2560]、value_proj[2560,2560]、
  //   conv1d[10240,1,4]、3×norm[10240]。此前误按 [2·ple_embed] 合并投影 + 单 norm[ple_embed] 建模，
  //   漏计了 key 的 hc_count 因子、value 分支与另外两个 norm（逐 checkpoint 对账抓出）。
  const pleEmbed = normalized.pleEmbedDim || 0;
  const hcHidden = (normalized.hiddenSize || 0) * (normalized.hyperConnectionCount || 1);
  const pleNorm = () => weightMatrixDecl("replicated", { shape: [hcHidden], quantizable: false });
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
        // P4-2：inject 叶声明 key_proj / value_proj / conv1d(depthwise) / 3×RMSNorm。ngram 表是
        // `ple.ple_embedding.ngram_embedding` 的 nn.Embedding（modeling_qwen4_exp.py:1111），
        // 容量走 type=embedding 子叶，不进本叶 weightMatrices。
        weightMatrices: [
          weightMatrixDecl("tp", { shape: [hcHidden, pleEmbed], split: "output", quantizable: false }),
          weightMatrixDecl("tp", { shape: [normalized.hiddenSize || 0, pleEmbed], split: "output", quantizable: false }),
          weightMatrixDecl("tp", { shape: [hcHidden, normalized.pleConvKernelSize || 1], split: "output", quantizable: false }),
          pleNorm(), pleNorm(), pleNorm(),
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

// DeepSeek V4.1 Engram —— n-gram 哈希记忆按门控写回 hc_mult 条残差流（随附
// model.py Engram / ParallelEngramEmbedding）。挂在 engramLayerIds 命中层的入口，
// 三步：哈希表查行（embed，fp8，无 MAC）→ wkv 投影出 hc_mult 个 key + 1 个 value →
// 归一化点积门控写回残差流（engram_gate）。engram_num_embeddings 逐层不同，按
// engramLayerIds 下标取当前层表宽（V4.1 主导权重来源）。
export function engramModule(id, normalized, { layerIndex = 0 } = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layerIds = normalized.engramLayerIds || [];
  const found = layerIds.indexOf(layerIndex);
  const engramIndex = found >= 0 ? found : 0;
  const numEmbeddings = (normalized.engramNumEmbeddings || [])[engramIndex] || 0;
  const headDim = normalized.engramHeadDim || 0;
  const nHeads = normalized.engramNHeads || 0;
  const maxNgram = normalized.engramMaxNgramSize || 0;
  const hcMult = normalized.mhcNumResidualStreams || 0;
  const hidden = normalized.hiddenSize || 0;
  // n_hash_cols = (max_ngram_size - 1) · n_heads（model.py Engram.__init__）。
  const nHashCols = Math.max((maxNgram - 1) * nHeads, 0);
  // wkv 输出 = dim·(hc_mult+1)：hc_mult 份 key + 1 份共享 value。
  const kvOut = hidden * (hcMult + 1);
  const streamDisplay = `[residual streams=${hcMult}, ${shapes.hidden}]`;
  const streamNumeric = [-1, hcMult, ...dims.hidden.slice(1)];
  const hashDisplay = `[batch, sequence, n-gram hash columns=${nHashCols}]`;
  const gatheredDisplay = `[batch, sequence, n-gram hash columns=${nHashCols}, engram head dimension=${headDim}]`;
  const kvDisplay = `[batch, sequence, engram kv=${kvOut}]`;
  return withShapeDims(moduleSpec(
    id,
    "Engram",
    "engram",
    {
      class: hfNamedClass(normalized, "engramClass", "Engram"),
      engram_num_embeddings: numEmbeddings,
      engram_head_dim: headDim,
      engram_n_heads: nHeads,
      engram_max_ngram_size: maxNgram,
      n_hash_columns: nHashCols,
      implementation: ["ngram_hash_lookup", "kv_projection", "match_gated_residual_write"],
      dataflow_edges: [["embed", "wkv"], ["wkv", "engram_gate"]],
      ...shapeFlow(streamDisplay, streamDisplay),
    },
    [
      // ParallelEngramEmbedding：按行分片、fp8 存储（查表时反量化）。gather 无 MAC，
      // 容量 = num_embeddings × engram_head_dim。类型 embedding → 走 embedGatherCounts。
      withShapeDims(moduleSpec(`${id}.embed`, "engram n-gram embedding", "embedding", {
        class: "ParallelEngramEmbedding",
        // 锚 1（modelIdentities）对 embedding 叶按 vocab_size·hidden_size 对账声明容量，
        // 沿用 embed_tokens / ngram 表的字段名：vocab_size=哈希表行数、hidden_size=每行宽。
        vocab_size: numEmbeddings,
        hidden_size: headDim,
        weightMatrices: [weightMatrixDecl("vocab", { shape: [numEmbeddings, headDim] })],
        ...shapeFlow(hashDisplay, gatheredDisplay),
      }), [-1, -1, nHashCols], [-1, -1, nHashCols, headDim]),
      operatorSpec(`${id}.wkv`, "engram key/value projection", "linear", {
        ...shapeFlow(gatheredDisplay, kvDisplay),
        projection_role: "engram_wkv",
        weightMatrices: [weightMatrixDecl("replicated", { shape: [kvOut, nHashCols * headDim] })],
        implementation: ["model.py Engram.wkv"],
      }, { input: [-1, -1, nHashCols * headDim], output: [-1, -1, kvOut] }),
      operatorSpec(`${id}.engram_gate`, "engram match-gated write", "engram_gate", {
        ...shapeFlow(`${streamDisplay}, ${gatheredDisplay}`, streamDisplay),
        engram_head_dim: headDim,
        hc_mult: hcMult,
        // q_weight / k_weight（[hc_mult, dim] 各一，fp32 nn.Parameter，仅作乘积使用）。
        weightMatrices: [
          weightMatrixDecl("replicated", { shape: [hcMult, hidden], quantizable: false }),
          weightMatrixDecl("replicated", { shape: [hcMult, hidden], quantizable: false }),
        ],
        implementation: ["model.py Engram.forward: normalized dot + signed-sqrt sigmoid gate"],
      }, { input: streamNumeric, output: streamNumeric }),
    ],
  ), streamNumeric, streamNumeric);
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
