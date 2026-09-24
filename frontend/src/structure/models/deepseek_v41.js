// DeepSeek-V4.1（DeepseekV41ForCausalLM）—— 上游确认为**独立架构**，非 V4 的子类/继承：
//   SGLang 侧 V4.1 有独立的 vision tower(deepseek_v41_vit.py)、image processor、chat encoding(encoding_dsv41)、
//   function-call/reasoning parser、template detection、`is_deepseek_v41_arch`（model_type "deepseek_v41"）；
//   `deepseek_v4.py` 的 `EntryClass = [DeepseekV4ForCausalLM]` **只含 V4**，无 DeepseekV41(DeepseekV4) 继承，
//   config 注册表也无 deepseek_v41 别名。故此处是**独立入口**，仅按需复用 V4 家族的共享构件（draft/装配），
//   而非把整套 V4 顶层装配器当作自身。
//
// 与 V4 的结构差异均由 config/normalize/ops/decoderStack 驱动、自动生效（已对账，实测 V4.1 与 V4 渲染不同）：
//   - CSA2 跨层 KV 共享：kv_source/index_source + FP4 宽度 → **KV 890 B/token**（V4-Flash 3,440）；
//   - Engram（n-gram 哈希记忆）：engramLayerIds 每层注入（V4 无）；
//   - CSA2 compress_ratios 含 ratio=2 → dsv4_sparse_mla（V4 只有 0/4/128）；无 hash 层（sqrtsoftplus 路由）；
//   - 草稿头：dsparkLayerCount>0 → DSpark（main_proj/main_norm/mhc），与 V4-Flash 的标准 MTP(enorm/hnorm) 不同，
//     由 v4Draft 按 config 分支（已实测各自正确）。
import { multimodalDecoderNetwork, networkSpecWithDraft, textDecoderNetwork } from "./common.js";
import { v4Draft } from "./deepseek_v4.js";
import { decoderStackNetwork } from "../layers/decoderStack.js";
import { embeddingModule } from "../layers/embedding.js";
import { lmHeadModule } from "../layers/outputHead.js";
import { rmsNormModule } from "../layers/norm.js";
import { multimodalEntry } from "../layers/multimodalEntry.js";
import { hyperConnectionModule } from "../layers/hybrid.js";
import { outputAttentionResidualModule } from "../layers/residual.js";
import { hfLayersAttr } from "../archs/index.js";

// CED 分界 = 解码器首个 Full 层（candidate_source_layer_id）。需为有效中段切点，
// 且 config 显式给出 kv_source_layer_ids（否则不是 V4.1 CSA2）。缺证据时返回 null →
// 退回既有单栈路径，行为不变。
function cedBoundary(normalized) {
  const cs = normalized.candidateSourceLayerId;
  const layers = normalized.layers || 0;
  if (!Array.isArray(normalized.kvSourceLayerIds)) return null;
  if (!Number.isFinite(cs) || cs <= 0 || cs >= layers) return null;
  return cs;
}

export function assembleDeepseekV41(resolved, normalized) {
  const draft = v4Draft(normalized);
  const boundary = cedBoundary(normalized);
  if (boundary == null) {
    const opts = { draft };
    return normalized.hasVision
      ? multimodalDecoderNetwork(resolved, normalized, opts)
      : textDecoderNetwork(resolved, normalized, opts);
  }
  // 官方 CED：20 层因果编码器 + 20 层解码器，解码器全局 KV 由末端编码器隐状态投影。
  const layersAttr = hfLayersAttr(normalized);
  const encoderId = `${layersAttr}.encoder`;
  const decoderId = `${layersAttr}.decoder`;
  const lastLayer = (normalized.layers || 0) - 1;
  const encoder = decoderStackNetwork(encoderId, normalized, { range: [0, boundary - 1], name: "Causal Encoder", type: "encoder" });
  const decoder = decoderStackNetwork(decoderId, normalized, { range: [boundary, lastLayer], name: "Decoder", type: "decoder" });
  const entry = normalized.hasVision ? multimodalEntry(normalized) : null;
  const children = [
    ...(entry ? entry.children : [embeddingModule("embed_tokens", normalized)]),
    encoder,
    decoder,
    ...(draft ? [draft] : []),
    ...(normalized.hyperConnectionCount ? [hyperConnectionModule("hyper_connection_mixer", normalized, "final")] : []),
    ...(normalized.attnResBlockSize ? [outputAttentionResidualModule("output_attn_residual", normalized)] : []),
    rmsNormModule("norm", "final norm", normalized),
    lmHeadModule("lm_head", normalized),
  ];
  const architecture = resolved.architecture || normalized.modelType || "Model";
  // CED 关系边：编码器 → 解码器的主干边额外标注"全局 KV 投影"，渲染为虚线并带提示。
  const edgeMeta = { [`${encoderId}=>${decoderId}`]: { relation: "kv-projection" } };
  return networkSpecWithDraft("model", architecture, architecture, children, draft, edgeMeta, entry);
}
