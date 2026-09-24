# DeepSeekV3/Kimi/GLM MLA checkpoint path repair

- 日期：2026-09-24
- Graph IR：保持 v2，不增加协议字段

## 发现

`source-ref.json` 对 DeepSeek-R1、DeepSeek-V3.1、DeepSeek-V3.2、Kimi-K2 Base/Thinking 和 GLM-5 的发布实现都给出了以下真实模块属性：

- `self_attn.q_a_proj`
- `self_attn.q_a_layernorm`
- `self_attn.q_b_proj`
- `self_attn.kv_a_proj_with_mqa`
- `self_attn.kv_a_layernorm`
- `self_attn.kv_b_proj`

旧的通用 MLA 配方把其中三个路径缩写成 `q_a_norm`、`kv_a_proj`、`kv_a_norm`。这不是数学语义错误，但会使 Graph IR canonical ID 与发布模块路径脱节，阻断 source/checkpoint 的精确绑定，也使 DSA indexer 的跨节点边引用了不存在的端点。

## 修复

通用 `mlaPaths()` 现在默认使用发布实现的精确属性名；已有 Kimi-K3 专用路径继续保留。DSA/qsa 的组网边也改为消费同一组 `mlaPaths()`，不再硬编码旧别名。

这只修正真值路径和边解析，不把语义节点误认为额外权重，也不改变 MLA 的计算公式、KV cache 口径或 Graph IR 协议。

## 验证

- DSA indexer checkpoint/config 双路径测试：20/20 通过；
- DeepSeek-R1/V3.1 机制断言确认 `q_a_layernorm`、`kv_a_layernorm`、`kv_a_proj_with_mqa` 存在且旧别名不存在；
- Kimi-K2 MLA 结构断言确认真实路径；
- 本批修改后将重新执行全量 frontend、docs、principles 和浏览器回归。
