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
  // 就报 unsupported。
  // 步骤 2（结构正确性收口）：再删 layers 字段推断兜底（generic-decoder）——
  // 用 layers/hidden/heads 拼通用网络同样属于伪造结构。前端的对应物 =
  // 空网络走完管线 + collectDiagnostics 枚举支持项。
  return { canonicalArchitecture: "unsupported", architecture: normalized.architecture, resolution: "unsupported" };
}
