import { moduleSpec, withShapeDims } from "./base.js";
import { attentionOperatorSpecs, deepseekV4AttentionOperatorSpecs, linearAttentionOperatorSpecs, minimaxDenseAttentionOperatorSpecs, minimaxM2AttentionOperatorSpecs, minimaxSparseAttentionOperatorSpecs, mlaAttentionOperatorSpecs, qsaAttentionOperatorSpecs, qwen35FullAttentionOperatorSpecs } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass, mlaPaths, recipeFlag, recipeLinearAttentionMode } from "../archs/index.js";
import { indexerScheduleOf, isIndexShareConfig } from "./schedule.js";

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
        return [["qkv_projection", "qkvz_split"], ["qkvz_split", "short_conv"], ["qkvz_split", "output_gate_norm"], ["beta_projection", "state_update"], ["decay_projection", "state_update"], ["short_conv", "state_update"], ["state_update", "output_gate_norm"], ["output_gate_norm", "out_proj"]];
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
    edges: () => [["qkv_index_proj", "qkv_index_split"], ["qkv_index_split", "q_norm"], ["qkv_index_split", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["qkv_index_split", "index_q_norm"], ["qkv_index_split", "index_k_norm"], ["index_q_norm", "index_rope"], ["index_k_norm", "index_rope"], ["index_rope", "indexer"], ["rope", "sparse_attention"], ["indexer.valid_block_ids", "sparse_attention"], ["sparse_attention", "o_proj"]],
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
    edges: () => [["qkv_gate_proj", "qkv_gate_split"], ["qkv_gate_split", "q_norm"], ["qkv_gate_split", "k_norm"], ["qkv_gate_split", "output_gate"], ["q_norm", "rope"], ["k_norm", "rope"], ["rope", "sdpa"], ["sdpa", "output_gate"], ["output_gate", "o_proj"]],
  },
  {
    kind: "qsa",
    ops: (id, normalized, layerIndex) => qsaAttentionOperatorSpecs(id, normalized, layerIndex),
    edges: (normalized, layerIndex = 0) => {
      // DSA over MLA：有 kv_lora_rank。逐头 QSA：没有。
      if ((normalized.kvLoraRank || 0) > 0) {
        const shared = isIndexShareConfig(normalized) && indexerScheduleOf(normalized)?.[layerIndex] === "reuse";
        const common = [
          ["q_a_proj", "q_a_norm"], ["q_a_norm", "q_b_proj"], ["kv_a_proj", "kv_split"],
          ["kv_split", "kv_a_norm"], ["kv_a_norm", "kv_b_proj"], ["q_b_proj", "rope"],
          ["kv_b_proj", "rope"],
        ];
        return shared
          ? [...common, ["rope", "sparse_attention"], ["index_reuse", "sparse_attention"], ["sparse_attention", "o_proj"]]
          : [...common, ["q_a_norm", "q_proj"], ["q_proj", "indexer"], ["wk_weights_proj", "k_norm"], ["k_norm", "indexer"], ["indexer", "sparse_attention"], ["rope", "sparse_attention"], ["sparse_attention", "o_proj"]];
      }
      if (recipeLinearAttentionMode(normalized) === "qwen4_exp" || normalized.qsaIndexerHeads) {
        return [["q_proj", "q_gate_split"], ["q_gate_split", "q_norm"], ["q_gate_split", "output_gate"],
          ["k_proj", "k_norm"], ["q_norm", "rope"], ["k_norm", "rope"], ["indexer", "sparse_attention"],
          ["rope", "sparse_attention"], ["v_proj", "sparse_attention"],
          ["sparse_attention", "output_gate"], ["output_gate", "o_proj"]];
      }
      return undefined;
    },
  },
  {
    kind: "dsv4",
    ops: (id, normalized, layerIndex) => deepseekV4AttentionOperatorSpecs(id, normalized, layerIndex),
    // 逐层连线（与 ops 的 compressor/indexer 发射条件同源）：只有 kv_source 层有 compressor、
    // index_source(或 ratio===4) 层有 indexer——此前用 compressRatios[0] 全局判据导致这些子算子
    // 在 Full/Reindex 层悬空无边。ratio、kv_source、index_source 与 ops/index.js 完全一致。
    edges: (normalized, layerIndex = 0) => {
      const ratio = Number(normalized.compressRatios?.[layerIndex] ?? 0);
      const kvSrc = normalized.kvSourceLayerIds;
      const idxSrc = normalized.indexSourceLayerIds;
      const emitCompressor = Array.isArray(kvSrc) ? kvSrc.includes(layerIndex) : ratio > 1;
      const emitIndexer = (Array.isArray(idxSrc) && idxSrc.includes(layerIndex)) || ratio === 4;
      const e = [
        ["fused_wqa_wkv", "qkv_split"], ["qkv_split", "q_norm"], ["qkv_split", "kv_norm"],
        ["q_norm", "q_proj"], ["q_proj", "rope"], ["kv_norm", "rope"],
        ["rope", "attention"], ["attention", "inverse_rope"], ["inverse_rope", "wo_a"], ["wo_a", "wo_b"],
      ];
      // 压缩 KV：compressor 读模块 hidden（与 fused_wqa_wkv 同为入口源），输出汇入 attention
      //（compressor -> attention 已登记 slice）。仅 kv_source 层有。
      if (emitCompressor) e.push(["compressor", "attention"]);
      // 稀疏索引器：q 潜表 → indexer.q_proj（同 q_proj 连续），weights_proj（入口源）与 q_proj
      // 汇入 indexer（fused-in），indexer 选择信号 → attention（control）。仅 index_source/ratio4 层有。
      if (emitIndexer) e.push(["q_norm", "indexer.q_proj"], ["indexer.q_proj", "indexer"], ["indexer.weights_proj", "indexer"], ["indexer", "attention"]);
      return e;
    },
  },
  {
    kind: "mla",
    ops: (id, normalized) => mlaAttentionOperatorSpecs(id, normalized),
    edges: (normalized) => {
      const paths = mlaPaths(normalized);
      const compressedQuery = normalized.qLoraRank != null;
      return [
        ...(compressedQuery ? [["q_a_proj", paths.qNorm], [paths.qNorm, "q_b_proj"]] : []),
        [paths.kvProjection, "kv_split"], ["kv_split", paths.kvNorm], [paths.kvNorm, "kv_b_proj"],
        ...(normalized.mlaUseNope
          ? [[compressedQuery ? "q_b_proj" : "q_proj", "sdpa"], ["kv_b_proj", "sdpa"], ["kv_split", "sdpa"]]
          : [[compressedQuery ? "q_b_proj" : "q_proj", "rope"], ["kv_b_proj", "rope"], ["rope", "sdpa"]]),
        ...(normalized.mlaUseOutputGate
          ? [["g_proj", "output_gate"], ["sdpa", "output_gate"], ["output_gate", "o_proj"]]
          : [["sdpa", "o_proj"]]),
      ];
    },
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
  const declaredEdges = component.edges(normalized, layerIndex);
  const edgeRelations = [];
  if (attentionKind === "mla" && normalized.mlaUseNope) {
    edgeRelations.push(
      { from: "kv_split", to: "sdpa", label: "shared key channels (NoPE)" },
      { from: "kv_b_proj", to: "sdpa", label: "content K, V" },
    );
  }
  if (attentionKind === "mla" && normalized.mlaUseOutputGate) {
    edgeRelations.push({ from: "g_proj", to: "output_gate", label: "gate logits" },
      { from: "sdpa", to: "output_gate", label: "attention output" });
  }
  if (Array.isArray(declaredEdges)) {
    if (declaredEdges.some(([from, to]) => from === "qkvz_split" && to === "short_conv")) {
      edgeRelations.push({ from: "qkvz_split", to: "short_conv", label: "q, k, v" });
    }
    if (declaredEdges.some(([from, to]) => from === "qkvz_split" && to === "output_gate_norm")) {
      edgeRelations.push({ from: "qkvz_split", to: "output_gate_norm", label: "z" });
    }
    if (declaredEdges.some(([from, to]) => from === "qkv_gate_split" && to === "output_gate")) {
      edgeRelations.push({ from: "qkv_gate_split", to: "output_gate", label: "gate" });
    }
    if (declaredEdges.some(([from, to]) => from === "q_gate_split" && to === "output_gate")) {
      edgeRelations.push({ from: "q_gate_split", to: "output_gate", label: "gate" });
    }
  }
  return withShapeDims(moduleSpec(
    id,
    displayName,
    "attention",
    {
      class: hfNamedClass(normalized, "attentionClass", "Attention", "Attention", { kind: attentionKind }),
      attention_kind: moduleKind,
      model_variant: normalized.modelType,
      ...(attentionKind === "mla" ? { position_encoding: normalized.mlaUseNope ? "none" : "rope" } : {}),
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
      ...(edgeRelations.length ? { dataflow_edge_relations: edgeRelations } : {}),
    },
    children,
  ), dims.hidden, dims.hidden);
}
