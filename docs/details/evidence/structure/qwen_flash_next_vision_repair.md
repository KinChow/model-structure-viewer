# Qwen3.8 Flash-Next 视觉塔语义补强

状态：**已修复并通过 config / production artifacts / 浏览器回归**。日期：
2026-09-24。Graph IR 保持 v2。

## 触发问题

此前 Flash-Next 的视觉图已经有 `patch_embed → pos_embed → blocks → merger`
和双向视觉注意力，但只表达了普通视觉 token 流，遗漏了发布实现中影响架构
语义的三个条件：

1. 视觉 attention 使用 `grid_thw` 产生的 `cu_seqlens` 对多图/视频 patch 做
   packed variable-length attention，而不是把所有输入假设为一个固定方阵；
2. 学习位置表不是简单查表后相加，forward 会依据图像/视频网格做 bilinear
   interpolation，并使用 `align_corners=True`；
3. `PatchMerger(use_postshuffle_norm=False)` 在 merge 前对 hidden width 做
   LayerNorm；不能把它误标成 merge 后 norm。

这些差异不会凭空新增权重或改变 Graph IR 协议，但会影响结构解释、可展开节点
属性和后续成本消费者的 token 域判断。

## 外部/发布实现证据

- pinned Transformers source：
  `models/Qwen/Qwen3.8-Flash-Next/modeling_qwen4_exp.py`
  - `Qwen4ExpVisionModel.forward`：patch embedding、位置插值、`grid_thw`
    position IDs、packed `cu_seqlens`、vision blocks、merger；
  - `Qwen4ExpVisionRotaryEmbedding`：axial 2D RoPE；
  - `Qwen4ExpVisionAttention`：`is_causal=False`，按 `cu_seqlens` 分块；
  - `Qwen4ExpVisionPatchMerger`：`use_postshuffle_norm=False`。
- 对应 source-ref 和变体 revision：
  `models/Qwen/Qwen3.8-Flash-Next/source-ref.json`、
  `models/Qwen/Qwen3.8-Flash-Next-FP8/source-ref.json`。
- 官方技术资料和模型卡索引见：
  `docs/details/evidence/structure/qwen_flash_next_qsa_repair.md`、
  `qwen_flash_next_gr_sources.json`。

## 修复

在现有架构配方中增加：

- `visionPackedAttention`
- `visionPositionInterpolation`
- `visionPositionAlignCorners`
- `visionMergerPostshuffleNorm`

Graph IR 节点增加结构化 attributes：

- 视觉 score/context：`packed_variable_length=true`、
  `attention_sequence_source="grid_thw"`；
- `visual.pos_embed`：`interpolation="bilinear"`、
  `interpolation_align_corners=true`、`position_grid_source="grid_thw"`；
- `visual.merger.norm`：`norm_stage="pre_shuffle"`。

没有把 packed attention 虚构成新的算子，也没有按每张图片拆成伪造的权重
模块；这些属性描述真实 forward 的控制/布局条件。

## 验证

- `qwenFlashNextQsa.test.js`：config 与 production artifacts **8/8**；
- frontend 全量单测：**601/601**；
- docs check、build、principles gate：通过；
- 两个发布变体的 QSA、位置插值、packed attention 和 merger 属性均有断言；
- 后续 Chrome 回归覆盖 Flash-Next 多模态入口和全量模型成本/图扫描。

仍保留的边界：

- packed attention 的 kernel 实际性能不由静态图推断；
- 视觉融合 scatter 的实际搬运量继续按既有规则保留 unknown；
- 不把本批视觉语义修复外推到其他视觉家族。
