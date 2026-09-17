// 构建链四级词汇（出处对照见 .comate/specs/naming-cleanup/doc.md 与 docs/details/modules.md）：
//   spec    —— builder 声明式产物（本仓已定稿：networkSpec / operatorSpec / spec.kind）
//   network —— spec 树（transformers/vLLM 惯例：模型 = nn.Module 组合树；IR v3 字段名）
//   model   —— 瞬态 StructureNode 树根，仅用于物化（transformers/vLLM：根模块即 model）
//   graph   —— 唯一载荷 Graph IR（llama.cpp ggml_cgraph / torch.fx Graph 先例）
import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { rmsNormModule } from "../layers/norm.js";
import { outputAttentionResidualModule } from "../layers/residual.js";
import { projectorModule } from "../layers/projector.js";
import { visionTowerModule } from "../layers/vision.js";
import { hyperConnectionModule } from "../layers/hybrid.js";
import { hfLayersAttr, recipeVisionInternalMerger } from "../archs/index.js";

export function networkSpec(id, name, architecture, children, attributes = {}) {
  return {
    kind: "network",
    id,
    name,
    architecture,
    attributes,
    children,
  };
}

/**
 * 投机头是**并行草稿分支**，不在主干顺序链上（对标 vLLM deepseek_mtp.py /
 * SGLang deepseek_nextn.py 的 forward）：
 *   MTP.forward(previous_hidden_states, inputs_embeds)
 *     inputs_embeds   = enorm(embed_tokens(input_ids))   // 与主模型共享 token 嵌入
 *     previous_hidden = hnorm(主干末层 hidden，final norm 之前)
 *     → eh_proj(cat[...]) → mtp_block → shared_head（草稿 logits）
 * 即 MTP 与主模型同源输入（embedding + decoder 输出两路 fan-in），输出走自己的
 * shared_head，**不回流**主干 final norm / lm_head。DSpark 仅从主干目标层 hidden
 * 取输入（单路 fan-in）。因此顶层不能把草稿串进 embed→decoder→draft→norm 的顺序
 * 链，需显式声明数据流边：主干串行 + 草稿 fan-in，草稿输出为末端不接主干。
 * children 里草稿仍置于 decoder 之后（默认布局顺序），边由 id 声明决定拓扑。
 */
export function networkSpecWithDraft(id, name, architecture, children, draft) {
  // 主干（embed → decoder → final norm → lm_head，含 vision/projector 前置）是真实
  // 顺序数据流（HF/vLLM forward 逐模块串行），应声明为 declared 实线边。此前无草稿
  // 模型走 { sequence: true }，该标记不产出 declared 边，物化时退化成 module-order
  // 虚线，导致「有草稿=实线 / 无草稿=虚线」的顶层连线风格不一致（同一条主干却两种
  // 画法）。统一：无论是否有草稿，主干都显式声明串行边；草稿只是在此基础上追加
  // fan-in / 出口边。
  const trunk = children.filter((child) => child !== draft);
  const edges = [];
  for (let index = 0; index < trunk.length - 1; index += 1) {
    edges.push([trunk[index].id, trunk[index + 1].id]);
  }
  if (!draft) return networkSpec(id, name, architecture, children, { dataflow_edges: edges });
  const decoder = children.find((child) => child.type === "decoder");
  const embed = children.find((child) => child.type === "embedding");
  const outputHead = children.find((child) => child.type === "output");
  // 主干末层 hidden → 草稿（MTP/DSpark 皆有）
  if (decoder) edges.push([decoder.id, draft.id]);
  // token 嵌入 → 草稿（仅 MTP：与主模型共享 embedding；DSpark 不吃 embedding）
  if (draft.type === "mtp" && embed) edges.push([embed.id, draft.id]);
  // 草稿 logits 出口（对标 SGLang/vLLM 的两种投机头权重实装）：
  //   - MTP：SharedHead 自带 head（checkpoint 有 shared_head.head.weight，
  //     tie_word_embeddings=false），草稿在自身 shared_head 内落 logits，不回主干；
  //   - DSpark：SGLang deepseek_v4_dspark._logits_from_x_post_hc 复用主干
  //     self.lm_head（attach_shared_modules 挂 target lm_head，_remap 对 head./
  //     lm_head. 一律 return None——无自带 head），故 DSpark 输出经共享边接主干
  //     lm_head。
  if (draft.type === "dspark" && outputHead) edges.push([draft.id, outputHead.id]);
  return networkSpec(id, name, architecture, children, { dataflow_edges: edges });
}

/** 投机头由调用方传入（对标 vLLM 各模型文件自己挂 mtp/dspark，不是共享 dispatcher）。 */
export function textDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind, draft } = {}) {
  const children = [
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, { attentionKind, defaultLayerKind }),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    ...(draft ? [draft] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ];
  return networkSpecWithDraft("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, children, draft);
}

export function multimodalDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind, draft } = {}) {
  const children = [
    visionTowerModule(normalized),
    ...(normalized.hasVisionProjector && !recipeVisionInternalMerger(normalized) ? [projectorModule(normalized)] : []),
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, { attentionKind, defaultLayerKind }),
    ...(draft ? [draft] : []),
    ...(normalized.hyperConnectionCount ? [hyperConnectionModule("hyper_connection_mixer", normalized, "final")] : []),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ];
  return networkSpecWithDraft("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, children, draft);
}
