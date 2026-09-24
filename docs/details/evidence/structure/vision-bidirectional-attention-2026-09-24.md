# 多模态 Vision Tower 的 attention mask 复核

日期：2026-09-24。目的：避免把语言主干的 causal prefill 计数
错误套到视觉编码器。

## 发布实现证据

当前仓库收录的发布实现中，以下视觉 attention 明确使用双向注意力：

- DeepSeek V4/V4.1：`inference/vision.py` 调用
  `torch.nn.functional.scaled_dot_product_attention(q, k, v)`，没有
  `is_causal` 或 causal mask。
- MiniMax M3：`MiniMaxM3VLVisionAttention.is_causal = False`。
- Qwen3.8-Flash-Next：`Qwen4ExpVisionAttention.is_causal = False`，
  eager/SDPA 路径均传 `is_causal=False`。
- GLM-5.3-Flash：`Glm5NextVisionAttention.is_causal = False`，
  eager/SDPA 路径均传 `is_causal=False`。
- Kimi K2.5/K3 的视觉/图像 attention 实现使用
  `flash_attn_varlen_func(..., causal=False)`；Kimi K3 的语言线性
  attention 仍然是 causal，两者不能混用。

这些不是由 model_type 推断，而是读取本地随条目冻结的发布实现。对应文件：

```text
models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py
models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py
models/zai-org/GLM-5.3-Flash/modeling_glm5_next.py
models/moonshotai/Kimi-K2.5/modeling_kimi_k25.py
models/moonshotai/Kimi-K3/modeling_kimi_k3.py
models/moonshotai/Kimi-K3/modeling_kimi_linear.py
```

## Graph IR 与费用规则

所有 39 个多模态条目的视觉 SDPA 现在带有：

```text
attention_mask_kind = bidirectional
```

`scoredPairs` 仍默认按 causal 计算，只有节点明确声明
`bidirectional` 时才使用完整 `query_tokens × key_tokens` 矩形。这样：

- 文本主干保持现有 causal 计数；
- Vision Tower 使用完整图像 token attention；
- SDPA 父算子、scores/context 解释叶和 softmax 使用同一 mask；
- 不通过修改总量 golden 掩盖局部计数差异。

覆盖测试：

- 39 个视觉模型配置路径；
- 39 个视觉模型 artifacts 路径；
- DeepSeek V4/V4.1 发布 ViT 精确 MAC；
- MiniMax/Qwen/GLM/Kimi 视觉实现的 Graph IR mask 属性；
- 全量 586 个 frontend 单测；
- Chrome 桌面和移动全量 60 模型成本扫描。
