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

/** 投机头由调用方传入（对标 vLLM 各模型文件自己挂 mtp/dspark，不是共享 dispatcher）。 */
export function textDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind, draft } = {}) {
  // §2.1：网络级子节点（embed → decoder → draft → norm → lm_head）显式声明顺序执行。
  // 投机头挂点对标 vLLM 各模型文件 children 顺序：decoder 之后、final norm 之前。
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, [
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, { attentionKind, defaultLayerKind }),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    ...(draft ? [draft] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}

export function multimodalDecoderNetwork(resolved, normalized, { attentionKind, defaultLayerKind, draft } = {}) {
  return networkSpec("model", resolved.architecture || normalized.modelType || "Model", resolved.architecture, [
    visionTowerModule(normalized),
    ...(normalized.hasVisionProjector && !recipeVisionInternalMerger(normalized) ? [projectorModule(normalized)] : []),
    embeddingModule("embed_tokens", normalized),
    decoderStackNetwork(hfLayersAttr(normalized), normalized, { attentionKind, defaultLayerKind }),
    ...(draft ? [draft] : []),
    ...(normalized.hyperConnectionCount ? [hyperConnectionModule("hyper_connection_mixer", normalized, "final")] : []),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ], { sequence: true });
}
