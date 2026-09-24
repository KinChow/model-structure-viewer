# DeepSeek V4/V4.1 视觉塔：双向注意力的计数修正

日期：2026-09-24。发布实现锚点与权重 fixture 见
`deepseek_v4_vision_sources.json` 和
`frontend/src/structure/models/__fixtures__/deepseek-v4-vision-release-header.json`。

官方 `inference/vision.py` 中，`ViT` 的每个 `Block.attn` 调用
`F.scaled_dot_product_attention(q, k, v)`，**没有**设置 `is_causal` 或
`attn_mask`。PyTorch 官方 API 的 `is_causal` 默认是 `False`，
而 vLLM 官方移植文档也明确称 DeepSeek-V4 ViT 为
“full bidirectional attention per image, 2D RoPE”：

- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/vision.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/vision.py`
- `https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html`
- `https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/common/vision/`

此前结构节点是正确的 SDPA，但成本注册表的通用 `scoredPairs`
在 prefill 对**所有**注意力套用了因果三角形：
四个图像 patch、16 个 head、head dim 64 时错误地算
`10 × 16 × (64 + 64) = 20,480 MAC`；实际完整矩形是
`16 × 16 × (64 + 64) = 32,768 MAC`。视觉塔 32 层的差距不能
靠改模型总 MAC 的黄金快照掩盖。

现在只给发布实现有明确证据的 V4/V4.1 SDPA 和三个展开子节点声明
`attention_mask_kind=bidirectional`。`scoredPairs` 增加可选
`causal` 输入，默认保持现有文本行为；聚合 SDPA、展开的
scores/context 及 softmax 在相同 mask 下计费。成本 oracle
在 `extractor.identity.test.js` 按独立发布实现的结构断言对账，
并有小尺寸精确 MAC 测试。其他视觉塔暂不凭直觉外推，需继续按
各自官方前向审计其 attention mask。

这仍是理论动作计数，不声称 GPU kernel 实测性能；视觉图像数、
切块/填充和执行实例位置的差异也没有被此修正消除。
