// 参数字节宽登记表（v1 范围：vLLM 显式声明 torch.float32 的**标量/一维**参数）。
//
// 为什么是一张表：权重字节恒等式的两侧（结构树 counts 与 derivedWeights 闭式公式）
// 必须对同一个参数用同一个字节宽，否则恒等式失配。dtype 和公式一样是「两侧各写
// 一遍就会漂移」的知识，所以定在这里，两边 import。
//
// v1 刻意不覆盖的（显式登记，不是遗漏）：
// - mHC 的大矩阵 hc_{attn,ffn}_fn [mix_hc, hc_dim]：上游同样是 fp32 buffer
//   （deepseek_v4/amd/model.py:714-727），但它是大矩阵，先跟随 torch_dtype 口径计；
// - 量化 scale 张量（FP8 逐块 scale 等）：需要引入量化块形状模型，另行立项；
// - 其余一切参数：默认跟随 torch_dtype（B=2）。
// 完整方案（per-tensor dtype 取自 checkpoint safetensors 头部）见
// operators_reference.md §7 与 MAINTENANCE 的登记。

/** vLLM 显式声明 torch.float32 的参数组 -> 每元素字节。未登记的默认 2（bf16/fp16）。 */
export const FP32_PARAMS = Object.freeze({
  // GDN/KDA 的衰减参数（dt_bias + A_log）。形状按家族不同，元素数由
  // derivedWeights.gdnDecayElements / extractor 的 stateUpdateCounts 给出：
  // - qwen GDN：两者都是 num_v_heads（qwen_gdn_linear_attn.py:467-475）
  // - glm5next：dt_bias = projection_size、A_log = num_heads（kda.py:205-243）
  // - kimi_k3：同 glm5next（kimi_k3/amd/kda.py:138-195）
  gdn_decay: 4,
  // mHC 的 per-mix 基准与 scale 标量：hc_{attn,ffn}_base [mix_hc]、
  // hc_{attn,ffn}_scale [3]（deepseek_v4/amd/model.py:728-753，
  // torch.float32、requires_grad=False）。
  mhc_base: 4,
  mhc_scale: 4,
});

export const DEFAULT_PARAM_BYTES = 2;

export function paramBytes(key) {
  return FP32_PARAMS[key] ?? DEFAULT_PARAM_BYTES;
}
