# Kimi-K3 发布视觉投影器：PatchMergerMLPV2

日期：2026-09-24。此专项由 MiniMax-M3 发布权重审计触发，是对
“其它家族是否也被泛化 projector 误建模”的横向检查。

## 一手证据

- 发布 `modeling_kimi_k3.py` revision
  `f831ab66814297da540d832a5235f8e904f29d06`，`PatchMergerMLPV2`
  定义于 783–815 行：`proj.0`/`proj.2` 均 `bias=False`，先展开
  `merge_kernel_size²` 个视觉 patch，经 `proj.0 → GELU → proj.2`
  后执行 `post_norm=RMSNorm(text_hidden)`；**不是** K2.5 的
  `pre_norm=LayerNorm(mm_hidden)`。
- 官方 `model.safetensors.index.json` SHA-256
  `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967d`，
  `mm_projector` 仅有 `proj.0.weight`、`proj.2.weight`、
  `post_norm.weight` 三个张量。
- 发布 shard 95 的 safetensors header（HTTP Range，未下载权重）分别
  给出 `[4096,4096]`、`[7168,4096]`、`[7168]`，三者均 BF16。
  精确键名和形状冻结在
  `frontend/src/structure/models/__fixtures__/kimi-k3-projector-header.json`。

## 修复与边界

公共 projector 构造器按架构配方中的 `visionProjectorKind=patchmergerv2`
选择 V2 分支：不发射不存在的 `pre_norm` 和 bias 参数，发射
`proj.0 → proj.1 → proj.2 → post_norm`；K2.5 的 V1 分支仍保留
前置 LayerNorm 和有 bias 的两层 Linear。Graph IR 仍为 v2。

机制测试分别从 config-only 与真实 header skeleton 构图，验证精确
canonical ID、数据依赖、参数声明和唯一 checkpoint 绑定。图面/费用
变化仅限 K3 V2，更新 golden 前应审阅逐模块差异，且需浏览器展开验收。

此处的 header 只证明投影器三个张量；K3 全部视觉塔与 93 层 packed
专家权重仍未完成逐张量审计。融合搬运和 GPU 性能依然 unknown。
