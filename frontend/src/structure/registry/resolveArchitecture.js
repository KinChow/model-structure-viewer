import { ARCHITECTURE_ALIASES } from "./aliases.js";
import { multimodalVariant } from "./architectureCatalog.js";

function withVision(normalized, canonicalArchitecture) {
  return normalized.hasVision
    ? multimodalVariant(canonicalArchitecture) || canonicalArchitecture
    : canonicalArchitecture;
}

export function resolveArchitecture(normalized, options = {}) {
  if (normalized.architecture && ARCHITECTURE_ALIASES[normalized.architecture]) {
    return {
      canonicalArchitecture: withVision(normalized, ARCHITECTURE_ALIASES[normalized.architecture]),
      architecture: normalized.architecture,
      resolution: "architecture-alias",
    };
  }

  // W5：删除原「architecture + model_type + modelId 拼串做 includes」的家族兜底。
  // 该兜底会把任何 id 里带 "qwen"/"deepseek"/"minimax" 的模型硬塞进某个模板，
  // 属于猜测而非判定；vLLM 的做法是 `_raise_for_unsupported` —— 精确表认不出
  // 就报 unsupported，由 collectDiagnostics 出前端告警，绝不伪造结构。
  // 实测：删除前 57/59 走 architecture-alias，仅 Qwen3_5MoeForCausalLM 系 2 个
  // 落兜底，已补进 aliases.js 精确表。
  if (normalized.layers) {
    return { canonicalArchitecture: "generic-decoder", architecture: normalized.architecture, resolution: "field-inference" };
  }
  return { canonicalArchitecture: "generic-config", architecture: normalized.architecture, resolution: "generic-config" };
}
