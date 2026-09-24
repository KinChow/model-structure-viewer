# Qwen3.5 / Qwen4-Exp 视觉塔：对齐 `visual.blocks` 发布结构

日期：2026-09-24。Graph IR 保持 v2。

## 证据

本地固定版本 source-ref sidecar 指向 Transformers v5.16.1
`modeling_qwen3_5.py`，并记录了以下发布模块：

- `visual.patch_embed.proj`：Conv3d patch embedding；
- `visual.pos_embed`：learned embedding；
- `visual.rotary_pos_emb`：无参数的视觉 RoPE provider；
- `visual.blocks.0.norm1/norm2`；
- `visual.blocks.0.attn.qkv/proj`；
- `visual.blocks.0.mlp.linear_fc1/linear_fc2/act_fn`；
- `visual.merger.norm/linear_fc1/act_fn/linear_fc2`。

覆盖的配置变体包括 Qwen3.5、Qwen3.5-MoE 和 Qwen4-Exp/Flash-Next。

## 原问题

旧的通用视觉组网把这些模型表示为：

```text
visual.0.input_norm
visual.0.qkv_proj
visual.0.qkv_split
visual.0.out_proj
visual.0.post_norm
visual.0.fc1/fc2
```

这不仅是显示名称不同，还导致 block 层级、LayerNorm 参数、position
embedding、merger 参数和 checkpoint canonical path 无法逐项对账。

## 修复

- 增加 Qwen blocks 专用视觉配方；
- 使用 `visual.blocks.0` 和发布的 `attn/MLP` 子路径；
- 展开 QKV reshape、Q/K RoPE、V 未旋转输入、context merge 和双残差；
- `LayerNorm` 按 affine scale+bias 计权重；
- patch embedding 使用 Conv3d 输入宽度
  `channels × temporal_patch × patch²`；
- `visual.pos_embed` 计 learned position table 的驻留和执行读取；
- merger 使用 `linear_fc1/linear_fc2`，而不是旧的 `fc1/fc2`；
- Qwen3.5、Qwen4-Exp 和 Flash-Next 共用发布结构，但保留各自的文本主干
  和融合配方，不把视觉结构差异扩散到 Graph IR 协议。

## 独立验证

`frontend/src/structure/models/multimodalEntry.test.js` 覆盖三个代表变体，
断言精确 canonical path、禁止旧路径、Conv3d/position shape、双向 attention
和控制边。

`frontend/src/structure/builtinModels.test.js`、全量前端测试和现有 60 条目
验证用于检查其余变体不回归。Chrome 专项将覆盖 Qwen3.8-27B、
Qwen3.8-Flash-Next 的桌面和移动展开。

本批不声称所有 Qwen 量化 checkpoint 的 packed scale 已完成逐 tensor 审计；
视觉塔拓扑和静态参数归属已按 source-ref 与配置证据修复。
