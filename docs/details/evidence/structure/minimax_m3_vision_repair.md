# MiniMax-M3 视觉塔：从通用融合 QKV 改为发布实现

日期：2026-09-24。Graph IR 保持 v2。

## 证据

本地入库的 Transformers v5.16.1 前向实现：

- `models/MiniMaxAI/MiniMax-M3/modeling_minimax_m3_vl.py`
- `models/MiniMaxAI/MiniMax-M3/source-ref.json`
- `models/MiniMaxAI/MiniMax-M3/config.json`

`MiniMaxM3VLVisionAttention` 明确实例化独立的
`q_proj`、`k_proj`、`v_proj`、`out_proj`，并设置 `is_causal = False`；
`MiniMaxM3VLVisionEncoderLayer` 使用 `layer_norm1 → attention → residual`
和 `layer_norm2 → MLP → residual`；视觉塔使用 `Conv3d` patch embedding、
`pre_layrnorm` 和轴向 3D RoPE。`MiniMaxM3VLMultiModalProjector` 则是
逐 patch 两层 GELU 投影，再由独立的 patch-merge MLP 聚合。

这与 Qwen/GLM 等可使用 fused QKV 的通用视觉配方不同，不能用
`qkv_proj → split` 仅凭配置形状代替真实 checkpoint 模块。

## 原问题

MiniMax-M3 和 MiniMax-M3-MXFP8 之前复用了通用视觉组网，产生了不存在的：

- `vision_tower.0.qkv_proj`
- `vision_tower.0.qkv_split`
- `vision_tower.0.input_norm`
- `vision_tower.0.post_norm`
- `vision_tower.0.fc1/fc2` 的错误层级

同时没有展示独立的三路投影、LayerNorm 残差和轴向位置输入，导致
source-ref 只能通过前缀近似“覆盖”，不能证明模块路径和费用归属正确。

## 修复

新增明确的 MiniMax 视觉配方，不扩展 Graph IR 协议：

- checkpoint 路径改为 `vision_tower.embeddings.proj`、
  `pre_layrnorm`、`layers.0.self_attn.q_proj/k_proj/v_proj/out_proj`、
  `layers.0.layer_norm1/2`、`layers.0.mlp.fc1/fc2`；
- 增加只改变视图的 head reshape、Q/K RoPE、V 未旋转输入、
  attention context merge 和两处 residual add；
- LayerNorm 按 affine scale+bias 计权重；Q/K/V/O 与 MLP 线性层按发布 bias
  计权重；
- `temporal_patch_size` 从嵌套 `img_token_compression_config` 归一化为 2，
  patch embedding 权重输入宽度为 `3 × 2 × 14 × 14 = 1176`；
- position angle provider 作为现有 `index-control` 关系边展示，实际 Q/K
  rotation 只在每层 rope 叶计费，避免重复计费；
- 保持视觉 attention 双向，保持现有 MiniMax 两阶段 projector 和文本入口。

## 独立验证

- `frontend/src/structure/models/multimodalEntry.test.js`：
  两个发布变体、config-only 和 production artifacts；断言不存在 fused
  QKV、精确模块路径、Conv3d 形状、视觉参数恒等式
  `631,185,920`，以及 V→SDPA 的真实边。
- `frontend/src/structure/builtinModels.test.js`：60 条目基础检查。
- `frontend/e2e/minimax-vision.spec.js`：真实 Chrome 桌面页面展开、节点、
  SVG 边路径和截图。

本修复不声称已完成 MiniMax 视觉完整 checkpoint tensor-by-tensor 对账；
header-truth 仍是总量级证据，逐模块权重路径以 source-ref 和配置形状为主。
