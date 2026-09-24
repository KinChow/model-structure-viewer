# 多模态 projector canonical path 与内部结构修复

日期：2026-09-24。Graph IR 保持 v2。

## 发现的问题

此前所有外部视觉投影器都使用显示层通用 ID `projector`。这会掩盖不同
发布实现的真实模块路径，并使 source-ref/checkpoint 绑定出现以下缺口：

| 模型家族 | 发布实现路径 | 旧图路径 |
|---|---|---|
| MiniMax-M3 | `multi_modal_projector` | `projector` |
| Kimi-K2.5 / Kimi-K3 | `mm_projector` | `projector` |
| DeepSeek-V4 / V4.1 Vision | `aligner` | `projector` |

这不是显示名称问题。官方 forward/source-ref 还显示内部结构并不相同：

- MiniMax-M3 是 `linear_1 → GELU → linear_2`，再经过
  `merge_linear_1 → GELU → merge_linear_2`；
- Kimi PatchMerger 使用 `pre_norm`、`proj.0/1/2`；
- DeepSeek Aligner 使用 `w1 → GELU → w2`。

如果仅把节点显示名改成对应术语而不改变 canonical path，真实 projector
权重仍可能被当作 gap；如果只把单个线性层改名，又会漏掉发布实现中的
第二段投影。

## 修复

架构配方新增 `visionProjectorPath` / `visionProjectorKind`，不在 builder 中
根据模型显示名猜测：

```text
MiniMax M3        multi_modal_projector + six internal steps
Kimi K2/K3        mm_projector + PatchMergerMLP paths
DeepSeek V4/V4.1  aligner + w1/GELU/w2
```

显示角色仍为 `projector`，但 Graph IR canonical ID 和 source-ref 绑定路径
使用发布实现的物理模块名。纯语义融合节点仍保持零参数，不伪造 checkpoint
模块。

## 独立断言

`multimodalEntry.test.js` 验证五个代表条目的精确 canonical IDs：

- MiniMax M3 的 7 个 projector/module paths；
- Kimi-K2.5、Kimi-K3 的 `mm_projector.proj.0/1/2`；
- DeepSeek-V4 Vision、V4.1 的 `aligner.w1/activation/w2`。

`multimodalSourceCoverage.test.js` 继续以 source-ref 为期望，验证所有视觉塔
和 projector source modules 都能由 Graph IR 的精确节点或折叠祖先表示。

## 证据边界

本修复证明了 canonical path 与发布 forward/module definition 的一致性，
并展开了已由源码确认的 projector 计算链；它不等于完整 safetensors 每个
张量已逐一读取。量化 scale、packed layout、融合搬运量和 GPU 性能仍按
unknown/待审计边界处理。
