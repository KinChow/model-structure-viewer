# source-ref 精确路径绑定修复

日期：2026-09-24
状态：已修复精确路径被展示 class 误过滤的问题；未宣称所有 graph 节点均已有
source-ref。

## 问题

DeepSeek-V4 的生产 `source-ref.json` 使用 Transformers 发布模块路径和实现类名：

```text
root.layers.2.self_attn.compressor
class_name = DeepseekV4CSACompressor
```

模板图为了展示语义使用：

```text
layers.2.self_attn.compressor
attributes.class = compressed KV/state compressor
```

原绑定器先按 canonical path 命中，再用 graph 的 `attributes.class` 过滤。
因此即使路径精确相同，也会因为展示 class 与发布实现 class 不同而丢失 source-ref。

## 修复规则

1. 先比较去掉 `root/model/language_model` 包装后的**未折叠精确路径**；
2. 精确路径唯一命中时，路径优先，不要求展示 class 等于实现 class；
3. 精确路径未命中时，才使用去掉层号/折叠段的路径，并要求 class 一致；
4. 最后的 class-only fallback 保持原行为；
5. 折叠视觉子节点不能因 `visual.0 → visual` 错绑定到 `VisionModel` 的 source-ref。

## 验证

- 单元测试：source-ref binder 与 DeepSeek-V4 机制测试 **18/18**；
- 60 个内置条目均有正向 source-ref 绑定；
- 2026-09-24 DeepSeek V4 topology 修复后的全量统计：
  `bound=6798`、`unmatched=41113`。新增的发布子模块使 V4 的精确路径可绑定；
  unmatched 总量上升是因为图中同时物化了更多真实节点，不能单看总数判断变差。

`unmatched` 仍然存在是预期的审计信号：当前模板没有为每个发布模块都物化
独立 graph node，尤其是已知 DeepSeek-V4 compressor/indexer 内部的
`kv_proj`、`gate_proj`、`kv_norm`、`rotary_emb`、`q_b_proj` 和 `scorer`。
这部分不能用“source-ref 已有绑定”掩盖，下一批需要按 Transformers 发布路径
扩展真实内部节点并继续做 checkpoint 对账。
