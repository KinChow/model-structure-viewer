// 查找键 = config.architectures[0] 原字符串。
// 对标：vLLM vllm/model_executor/models/registry.py
//   _TEXT_GENERATION_MODELS / _MULTIMODAL_MODELS 以 HF 类名为键；
//   ModelRegistry._try_load_model_cls / _raise_for_unsupported。
// 对标：SGLang python/sglang/srt/models/registry.py
//   _ModelRegistry.models 以 model_arch 为键；_raise_for_unsupported。
// 没有 architectures[0] 或不在 MODELS 表里 → unsupported，不编造结构。

import { MODELS } from "../models/index.js";

export function resolveArchitecture(normalized) {
  const architecture = normalized?.architecture;
  if (architecture && MODELS[architecture]) {
    return {
      architecture,
      resolution: "architecture-alias",
    };
  }
  return {
    architecture,
    resolution: "unsupported",
  };
}

export function hasModelArchitecture(architecture) {
  return Boolean(architecture && MODELS[architecture]);
}
