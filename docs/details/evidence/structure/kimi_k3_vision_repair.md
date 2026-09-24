# Kimi-K3 视觉塔：MoonViT 发布路径对齐

日期：2026-09-24。Graph IR 保持 v2。

## 证据

本地固定 revision 的发布实现和 checkpoint index：

- `models/moonshotai/Kimi-K3/modeling_kimi_k3.py`
- `models/moonshotai/Kimi-K3/config.json`
- `models/moonshotai/Kimi-K3/index.json`
- `models/moonshotai/Kimi-K3/source-ref.json`

`MoonViT3dPretrainedModel` 明确创建 `patch_embed` 和 `encoder`。
`MoonVision3dPatchEmbed` 是带可学习二维 position weight 的 Conv2d；
`MoonViTEncoderLayer` 使用 `norm0 → wqkv → QKV reshape/RoPE →
attention → wo → residual`，随后 `norm1 → mlp.fc0 → GELU →
mlp.fc1 → residual`。其 `qkv_hidden_size=1536` 独立于视觉 hidden
`1024`，最终由 `encoder.final_layernorm` 收束。视觉塔的 position
embedding 是 `64 × 64 × 1024` 的可学习参数，时间方向的正弦项是
buffer，不应另计为训练权重。

## 原问题

原通用视觉配方虽然形状上能够生成 fused QKV，但使用了不存在的模块路径：

- `vision_tower.0.qkv_proj`
- `vision_tower.0.qkv_split`
- `vision_tower.0.input_norm/post_norm`
- `vision_tower.0.fc1/fc2`

同时没有体现 `patch_embed.pos_emb.weight`、`encoder.blocks.*`、
`encoder.final_layernorm`，并将视觉塔 token 预算和 projector 合并后的
token 数混用。

## 修复

- 增加 Kimi-K3 专用视觉配方，不改变通用视觉族；
- 使用发布路径 `patch_embed.proj/pos_emb`、
  `encoder.blocks.0.norm0/wqkv/wo/norm1/mlp.fc0/fc1` 和
  `encoder.final_layernorm`；
- 保留 fused `wqkv`，但显式展开 QKV split、reshape、共享 2D RoPE、
  V 未旋转输入和 attention context merge；
- 按 `attn_bias=false`、`linear_bias=false`、`norm_type=rmsnorm` 建立权重
  和费用归属；
- 新增 `vision_token_source=patch_tokens`，让视觉塔 attention/MLP 使用
  4096 个 patch token，而 patch merge/projector 仍使用合并后的 1024 token；
- `vision_position` 支持可学习 position weight 的驻留和执行读流量；
- 继续保持视觉 attention 双向，不将 MoonViT 改成因果 attention。

## 独立验证

`frontend/src/structure/models/multimodalEntry.test.js` 固化：

- 真实模块路径和禁止的旧路径；
- `qkv_hidden_size=1536`、patch 输入 `3×14×14`；
- position weight `[64,64,1024]`；
- 视觉塔参数容量 `401,214,464`；
- 共享 RoPE 的 `index-control` 边和 QKV→SDPA 数据边。

`frontend/e2e/kimi-vision.spec.js` 使用真实 Chrome 桌面和移动尺寸，
展开生产页面，检查节点、SVG 连线和截图。

本批不声称 Kimi-K3 全模型所有量化权重均已完成逐 tensor 对账；
视觉 index 中的路径证据和发布实现已用于结构修复，其他未核验项继续保留
在全量状态台账中。
