// 参数字节宽登记表（v1 范围：vLLM 显式声明 torch.float32 的**标量/一维**参数）。
//
// 为什么是一张表：叶 counts 与 weightMatrices 声明必须对同一个参数用同一个字节宽，
// 否则锚 1 失配。dtype 和公式一样是「两侧各写一遍就会漂移」的知识，所以定在这里。
//
// 覆盖范围（2026-09-09 用户裁决「一次性修完」）：
// - 全部 vLLM 显式声明 torch.float32 的参数（含大矩阵 hc_*_fn）；
// - 量化张量（FP8 块量化 / MXFP8 / GPTQ）的权重与 scale 走 cost/quantBytes.js
//   的 per-matrix 精确计算，不进本表（本表只管**未量化**参数的 dtype）；
// - 其余参数默认跟随 torch_dtype（B=2）。
// 终态方案：checkpoint 证据在场时以 safetensors 头部的逐 tensor dtype 为准
// （truth/skeleton.js 的 weight_dtypes 已是该机制）。

/** vLLM 显式声明 torch.float32 的参数组 -> 每元素字节。未登记的默认 2（bf16/fp16）。 */
export const FP32_PARAMS = Object.freeze({
  // GDN/KDA 的衰减参数（dt_bias + A_log）。形状按家族不同，元素数由
  // extractor 的 gatedDeltaStateCtx / counts.gatedDeltaStateCounts 给出：
  // - qwen GDN：两者都是 num_v_heads（qwen_gdn_linear_attn.py:467-475）
  // - glm5next：dt_bias = projection_size、A_log = num_heads（kda.py:205-243）
  // - kimi_k3：同 glm5next（kimi_k3/amd/kda.py:138-195）
  gdn_decay: 4,
  // mHC 的 per-mix 基准与 scale 标量：hc_{attn,ffn}_base [mix_hc]、
  // hc_{attn,ffn}_scale [3]（deepseek_v4/amd/model.py:728-753，
  // torch.float32、requires_grad=False）。
  mhc_base: 4,
  mhc_scale: 4,
  // mHC 的大矩阵 hc_{attn,ffn}_fn [mix_hc, hc_dim]：上游同样是 fp32 buffer
  //（deepseek_v4/amd/model.py:714-727，torch.float32、requires_grad=False），
  // 但它是**密读 GEMM 操作数**，走权重字节恒等式（不同于 tid2eid 的散读）。
  mhc_fn: 4,
  // DSpark confidence_head.proj 是 ReplicatedLinear(params_dtype=float32)
  // （vLLM qwen3_dspark.py:176-182）。
  dspark_confidence: 4,
});

export const DEFAULT_PARAM_BYTES = 2;

export function paramBytes(key) {
  return FP32_PARAMS[key] ?? DEFAULT_PARAM_BYTES;
}
