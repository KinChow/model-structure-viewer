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
import { multimodalDecoderNetwork, textDecoderNetwork } from "./common.js";
import { v4Draft } from "./deepseek_v4.js";

export function assembleDeepseekV41(resolved, normalized) {
  const draft = v4Draft(normalized);
  const opts = { draft };
  return normalized.hasVision
    ? multimodalDecoderNetwork(resolved, normalized, opts)
    : textDecoderNetwork(resolved, normalized, opts);
}
