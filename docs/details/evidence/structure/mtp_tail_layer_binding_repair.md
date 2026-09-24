# DeepSeek-style MTP tail-layer binding repair

- 日期：2026-09-24
- Graph IR：保持 v2

## 发现

公开 vLLM `deepseek_mtp.py` 和 safetensors index manifest 显示，DeepSeek-style MTP 并不是一个只存在于前端的 `mtp.*` 权重树：

- runtime 有独立的 MTP `embed_tokens`；
- 发布 state dict 把 MTP layer 放在 `model.layers.<num_hidden_layers>`；
- MTP layer 下包含 `enorm`、`hnorm`、`eh_proj`、`input_layernorm`、`self_attn`、`mlp`；
- `shared_head` 也位于同一个尾层路径下。

例如：

```text
model.layers.61.embed_tokens.weight
model.layers.61.eh_proj.weight
model.layers.61.shared_head.norm.weight
model.layers.61.shared_head.head.weight
```

GLM-4.7 对应尾层为 `model.layers.92.*`。

旧图虽然显示了 `mtp.enorm/eh_proj/layer/shared_head`，但没有 MTP 本地 `embed_tokens`，且 header-only 场景无法验证这些真实尾层路径是否能绑定到草稿分支。

## 修复

- 在 DeepSeek-style MTP 草稿树中增加 checkpoint-local `mtp.embed_tokens`；
- 保留主干 embedding 到 MTP runtime input 的 fan-in，不把 resident fallback embedding 错画成另一条 active tensor-flow copy；
- 增加动态 tail-layer path aliases，把发布的 `layers.<N>.*` 精确绑定到 `mtp.*` / `mtp.layer.*`；
- 只对 DeepSeek-style MTP 架构启用该映射，不影响 Qwen 等本来就发布为 `mtp.*` 的架构；
- MTP 本地 embedding 的驻留权重进入容量计算，避免 header 总量与模板漏项不一致。

## 验证

新增机制测试覆盖 DeepSeek-R1（尾层 61）和 GLM-4.7（尾层 92），确认以下 tensor 每个只绑定一个 Graph IR 节点：

- `embed_tokens`
- `enorm`
- `hnorm`
- `eh_proj`
- `input_layernorm`
- `shared_head.norm`
- `shared_head.head`

完整 MTP experts 的逐张量绑定仍受当前 aggregate expert 叶设计限制，继续保持 unknown，不把这批测试扩大解释为完整 MTP truth。
