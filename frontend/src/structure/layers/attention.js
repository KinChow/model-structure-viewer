import { moduleSpec, withShapeDims } from "./base.js";
import { attentionOperatorSpecs, deepseekV4AttentionOperatorSpecs, linearAttentionOperatorSpecs, minimaxDenseAttentionOperatorSpecs, minimaxM2AttentionOperatorSpecs, minimaxSparseAttentionOperatorSpecs, mlaAttentionOperatorSpecs, qsaAttentionOperatorSpecs, qwen35FullAttentionOperatorSpecs } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass, recipeFlag, recipeLinearAttentionMode } from "../archs/index.js";

// 组件表：attentionKind × 字段/配方匹配 → { name, ops, edges }。
// ops 与 edges 必须取自同一表项 —— children 改了 edges 没跟着改在结构上不可能。
// 条目按序首个匹配生效；edges 允许 undefined（未声明组合仍依赖 module-order，D 阶段收口）。
const ATTENTION_COMPONENTS = [
  {
    kind: "linear",
    ops: (id, normalized) => linearAttentionOperatorSpecs(id, normalized),
    edges: (normalized) => {
      const mode = recipeLinearAttentionMode(normalized);
      if (mode === "qwen3_5" || mode === "qwen4_exp") {
        return [["qkv_projection", "qkvz_split"], ["qkvz_split", "short_conv"], ["beta_projection", "state_update"], ["decay_projection", "state_update"], ["short_conv", "state_update"], ["state_update", "output_gate_norm"], ["output_gate_norm", "out_proj"]];
      }
      if (mode === "kimi_k3" || mode === "glm5_next") {
        const lowRankGate = mode === "glm5_next";
        return [
          ["qkv_projection", "short_conv"],
          ["qkv_projection", "f_b_proj"],
          ["f_b_proj", "state_update"],
          ...(lowRankGate
            ? [["qkv_projection", "g_b_proj"], ["g_b_proj", "output_gate_norm"]]
            : [["qkv_projection", "output_gate_norm"]]),
          ["short_conv", "state_update"],
          ["state_update", "output_gate_norm"],
          ["output_gate_norm", "out_proj"],
        ];
      }
      return undefined;
    },
  },
  {
    kind: "sparse",
    match: (normalized) => Boolean(normalized.sparseTopkBlocks),
    ops: (id, normalized, layerIndex) => minimaxSparseAttentionOperatorSpecs(id, normalized, layerIndex),
    edges: () => [["qkv_index_proj", "qkv_index_split"], ["qkv_index_split", "q_norm"], ["qkv_index_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["qkv_index_split", "index_q_norm"], ["qkv_index_split", "index_k_norm"], ["index_q_norm", "index_rope"], ["index_k_norm", "index_rope"], ["index_rope", "indexer"], ["rope", "sparse_attention"], ["indexer", "sparse_attention"], ["sparse_attention", "o_proj"]],
  },
  {
    kind: "gqa",
    match: (normalized) => Boolean(normalized.sparseTopkBlocks),
    ops: (id, normalized) => minimaxDenseAttentionOperatorSpecs(id, normalized),
    edges: () => [["qkv_index_proj", "qkv_index_split"], ["qkv_index_split", "q_norm"], ["qkv_index_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "sdpa"], ["sdpa", "o_proj"]],
  },
  {
    kind: "gqa",
    match: (normalized) => recipeFlag(normalized, "fusedQkv") && !normalized.sparseTopkBlocks,
    ops: (id, normalized) => minimaxM2AttentionOperatorSpecs(id, normalized),
    edges: () => [["qkv_proj", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "sdpa"], ["sdpa", "o_proj"]],
  },
  {
    kind: "qwen35_full",
    name: () => "Qwen3.5 Full Attention",
    ops: (id, normalized) => qwen35FullAttentionOperatorSpecs(id, normalized),
    edges: () => [["qkv_gate_proj", "qkv_gate_split"], ["qkv_gate_split", "q_norm"], ["qkv_gate_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "sdpa"], ["sdpa", "output_gate"], ["output_gate", "o_proj"]],
  },
  {
    kind: "qsa",
    ops: (id, normalized, layerIndex) => qsaAttentionOperatorSpecs(id, normalized, layerIndex),
    edges: (normalized) => {
      // DSA over MLA：有 kv_lora_rank。逐头 QSA：没有。
      if ((normalized.kvLoraRank || 0) > 0) {
        return [["q_a_proj", "q_a_norm"], ["q_a_norm", "q_b_proj"], ["kv_a_proj", "kv_split"], ["kv_split", "kv_a_norm"], ["kv_a_norm", "kv_b_proj"], ["q_b_proj", "rope"], ["kv_b_proj", "rope"], ["q_a_norm", "q_proj"], ["q_proj", "indexer"], ["wk_weights_proj", "k_norm"], ["k_norm", "indexer"], ["indexer", "sparse_attention"], ["rope", "sparse_attention"], ["sparse_attention", "o_proj"]];
      }
      if (recipeLinearAttentionMode(normalized) === "qwen4_exp" || normalized.qsaIndexerHeads) {
        return [["qkv_proj", "q_norm"], ["qkv_proj", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["indexer", "sparse_attention"], ["rope", "sparse_attention"], ["sparse_attention", "out_proj"]];
      }
      return undefined;
    },
  },
  {
    kind: "dsv4",
    ops: (id, normalized, layerIndex) => deepseekV4AttentionOperatorSpecs(id, normalized, layerIndex),
    edges: (normalized) => Number(normalized.compressRatios?.[0]) > 0
      ? [["fused_wqa_wkv", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "kv_norm"], ["q_norm", "q_proj"], ["q_proj", "rope"], ["kv_norm", "rope"], ["compressor", "attention"], ["rope", "attention"], ["attention", "inverse_rope"], ["inverse_rope", "wo_a"], ["wo_a", "wo_b"]]
      : [["fused_wqa_wkv", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "kv_norm"], ["q_norm", "q_proj"], ["q_proj", "rope"], ["kv_norm", "rope"], ["rope", "attention"], ["attention", "inverse_rope"], ["inverse_rope", "wo_a"], ["wo_a", "wo_b"]],
  },
  {
    kind: "mla",
    ops: (id, normalized) => mlaAttentionOperatorSpecs(id, normalized),
    edges: () => [["q_a_proj", "q_a_norm"], ["q_a_norm", "q_b_proj"], ["kv_a_proj", "kv_split"], ["kv_split", "kv_a_norm"], ["kv_a_norm", "kv_b_proj"], ["q_b_proj", "rope"], ["kv_b_proj", "rope"], ["rope", "sdpa"], ["sdpa", "o_proj"]],
  },
  {
    kind: "gqa",
    ops: (id, normalized) => attentionOperatorSpecs(id, "gqa", normalized),
    edges: () => [["q_proj", "rope"], ["k_proj", "rope"], ["rope", "sdpa"], ["v_proj", "sdpa"], ["sdpa", "o_proj"]],
  },
  {
    // 兜底：未列出组合的 children 走默认 GQA 链（attentionKind 透传给 scores 标注），
    // edges 未声明（undefined）。
    ops: (id, normalized, layerIndex, kind) => attentionOperatorSpecs(id, kind, normalized),
    edges: () => undefined,
  },
];

function matchAttentionComponent(kind, normalized) {
  return ATTENTION_COMPONENTS.find((entry) =>
    (entry.kind === undefined || entry.kind === kind)
    && (!entry.match || entry.match(normalized)));
}

export function attentionModule(id, normalized, attentionKind, layerIndex = 0) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const component = matchAttentionComponent(attentionKind, normalized);
  // 调度 kind "qsa" 在有 kv_lora 时是 DSA over MLA，模块标注跟叶口径对齐。
  const moduleKind = attentionKind === "qsa" && (normalized.kvLoraRank || 0) > 0
    ? "dsa_sparse_mla"
    : attentionKind;
  const displayName = component.name
    ? component.name(attentionKind)
    : `${attentionKind.toUpperCase()} Attention`;
  const children = component.ops(id, normalized, layerIndex, attentionKind);
  const declaredEdges = component.edges(normalized);
  return withShapeDims(moduleSpec(
    id,
    displayName,
    "attention",
    {
      class: hfNamedClass(normalized, "attentionClass", "Attention", "Attention", { kind: attentionKind }),
      attention_kind: moduleKind,
      model_variant: normalized.modelType,
      hidden_size: normalized.hiddenSize,
      num_attention_heads: normalized.attentionHeads,
      num_key_value_heads: normalized.kvHeads,
      compress_ratio: attentionKind === "dsv4" ? normalized.compressRatios?.[layerIndex] : undefined,
      attention_variant: attentionKind === "dsv4" ? "deepseek_v4" : undefined,
      ...shapeFlow(shapes.hidden, shapes.hidden, {
        query_shape: shapes.attentionQuery,
        key_shape: shapes.attentionKey,
        value_shape: shapes.attentionValue,
      }),
      dataflow_edges: declaredEdges,
    },
    children,
  ), dims.hidden, dims.hidden);
}
