export const ARCHITECTURE_ALIASES = {
  DeepseekV3ForCausalLM: "mla-moe-decoder",
  DeepseekV32ForCausalLM: "mla-moe-decoder",
  DeepseekV4ForCausalLM: "mla-moe-decoder",
  Glm4MoeForCausalLM: "gqa-moe-decoder",
  GlmMoeDsaForCausalLM: "mla-moe-decoder",
  Qwen3ForCausalLM: "gqa-decoder",
  Qwen3_5ForConditionalGeneration: "gqa-decoder",
  Qwen3_5MoeForConditionalGeneration: "gqa-moe-decoder",
  // W5：此前靠 resolveArchitecture 的子串兜底命中（probe.includes("qwen")），
  // 现补进精确表。Qwen3.8-2.4T-A95B 系用纯文本 MoE 架构名。
  Qwen3_5MoeForCausalLM: "gqa-moe-decoder",
  Qwen3MoeForCausalLM: "gqa-moe-decoder",
  Qwen4ExpForConditionalGeneration: "multimodal-gqa-moe-decoder",
  KimiK25ForConditionalGeneration: "mla-moe-decoder",
  KimiK3ForConditionalGeneration: "hybrid-multimodal-moe-decoder",
  Glm5NextForConditionalGeneration: "hybrid-multimodal-moe-decoder",
  MiniMaxM2ForCausalLM: "gqa-moe-decoder",
  MiniMaxM3SparseForConditionalGeneration: "multimodal-sparse-moe-decoder",
};
