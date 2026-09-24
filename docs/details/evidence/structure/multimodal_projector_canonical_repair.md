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

- MiniMax-M3 发布 checkpoint 是 `multi_modal_projector.linear_1/2`
  后接独立的 `patch_merge_mlp.linear_1/2`；两者之间还有 patch regroup。
  较新的 Transformers 合并类把后者命名为 `merge_linear_1/2`，但该路径
  **不存在于这两个发布版本的权重索引**；
- Kimi PatchMerger 使用 `pre_norm`、`proj.0/1/2`；
- DeepSeek Aligner 使用 `w1 → GELU → w2`。

如果仅把节点显示名改成对应术语而不改变 canonical path，真实 projector
权重仍可能被当作 gap；如果只把单个线性层改名，又会漏掉发布实现中的
第二段投影。

## 修复

架构配方新增 `visionProjectorPath` / `visionProjectorKind`，不在 builder 中
根据模型显示名猜测：

```text
MiniMax M3        multi_modal_projector → patch_merge_mlp
Kimi K2/K3        mm_projector + PatchMergerMLP paths
DeepSeek V4/V4.1  aligner + w1/GELU/w2
```

显示角色仍为 `projector`，但 Graph IR canonical ID 和 source-ref 绑定路径
使用发布实现的物理模块名。纯语义融合节点仍保持零参数，不伪造 checkpoint
模块。

## 独立断言

`multimodalEntry.test.js` 验证五个代表条目的精确 canonical IDs：

- MiniMax M3 的两个物理模块与各自的 `linear_1/2`；
- Kimi-K2.5、Kimi-K3 的 `mm_projector.proj.0/1/2`；
- DeepSeek-V4 Vision、V4.1 的 `aligner.w1/activation/w2`。

`multimodalSourceCoverage.test.js` 继续以 source-ref 为期望，验证所有视觉塔
和 projector source modules 都能由 Graph IR 的精确节点或折叠祖先表示。

## 证据边界

2026-09-24 追加发布权重索引与 shards 26/59 safetensors header 的独立
取证：MiniMax-M3 revision `f0e1c1e04d40177e4673a22097036854f536e9c0`
索引 SHA-256 `54dbde502126d07f6999077437a06b5df1f71e317518956d0aad1c8197df524e`。
四个投影各含 weight+bias，`patch_merge_mlp.linear_1.weight` 的
`[6144,24576]` 形状证实四块文本宽度合并；较新的库实现与已发布
checkpoint 路径不一致，必须以发布权重为准。

MXFP8 revision `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba` 的官方
索引 SHA-256 `41126751cfdb44c2fb33522bcaaa6f6ca057462838b3493a347678e6e1ca24ee`
同样列出这两个独立模块的 8 个张量名。只从其索引推断**模块存在与路径**，
不从索引推断打包形状与 dtype；冻结键名在
`frontend/src/structure/models/__fixtures__/minimax-m3-published-projector-index.json`。

这修正了本页上一批“`multi_modal_projector` 内有 six internal steps”的
过度结论：该六步只描述较新 Transformers 库的融合类，不代表两个发布
checkpoint 的模块边界。公共入口支持由配方声明的连续投影模块，仍使用
Graph IR v2 和已有 canonical ID；无须设计新协议或按显示名推断。

此专项只读取上述 8 个张量的名字和 header，并未读取全部视觉/语言权重。
MXFP8 的索引同样有这两个独立模块，量化 scale/packed layout、融合搬运量
和 GPU 性能仍按 unknown/待审计边界处理。
