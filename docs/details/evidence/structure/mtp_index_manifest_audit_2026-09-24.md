# MTP index-manifest audit

- 日期：2026-09-24
- 方法：只下载 `model.safetensors.index.json`，不下载模型权重；通过 ModelScope 的公开 resolve endpoint 读取 `weight_map`。
- 目的：独立验证 config 的 MTP 声明是否落到发布权重，避免只用配置推断。

## 结果

| model | index tensor count | MTP-like keys | 结论 |
|---|---:|---:|---|
| `deepseek-ai/DeepSeek-R1` | 91,991 | 2 | 有 `model.layers.61.shared_head.norm.weight` 和 `model.layers.61.shared_head.head.weight`，保留 1 个 MTP 模块 |
| `deepseek-ai/DeepSeek-V3.1` | 91,991 | 2 | 与 R1 相同，保留 1 个 MTP 模块 |
| `moonshotai/Kimi-K2-Base` | 139,644 | 0 | 不生成 MTP |
| `moonshotai/Kimi-K2-Thinking` | 208,276 | 0 | 不生成 MTP |
| `MiniMaxAI/MiniMax-M2.7` | 96,103 | 0 | 不生成 MTP，尽管 config 声明 `use_mtp/num_mtp_modules` |
| `zai-org/GLM-4.7` | 44,691 | 2 `shared_head.*` | 保留 1 个 MTP 模块；header `mtp_tensor_count=502` |

MTP-like key 的筛选同时覆盖 `mtp`、`nextn` 和 `shared_head`；DeepSeek 两个模型的实际发布路径是 `model.layers.61.shared_head.*`，不是配置中直接出现的 `mtp.*`。

## 对当前实现的影响

现有规则“config 声明模块数 + checkpoint `mtp_tensor_count` 决定是否实例化”仍然正确：

- DeepSeek-R1/V3.1 的 header `mtp_tensor_count=1564` 与 index manifest 的两个 shared-head 参数一致，保留 `DeepSeekMultiTokenPredictorLayer`；
- Kimi-K2 Base/Thinking 的 header `mtp_tensor_count=0` 与 index manifest 一致，抑制 config/继承路径可能产生的幻影 MTP；
- MiniMax-M2.7 的 header/index 都没有 MTP 权重，继续抑制其 config-only 的 `use_mtp/num_mtp_modules` 声明；
- GLM-4.7 的 header/index 都有 shared-head 权重，当前生产图保留 1 个 MTP 模块；
- 不能仅以 `shared_head` 两个参数推出完整 MTP 内部逐层权重，因此 MTP 子图的完整 checkpoint 绑定仍保留 unknown。

## 来源

- `https://modelscope.cn/models/deepseek-ai/DeepSeek-R1/resolve/master/model.safetensors.index.json`
- `https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1/resolve/master/model.safetensors.index.json`
- `https://modelscope.cn/models/moonshotai/Kimi-K2-Base/resolve/master/model.safetensors.index.json`
- `https://modelscope.cn/models/moonshotai/Kimi-K2-Thinking/resolve/master/model.safetensors.index.json`
- `https://hf-mirror.com/MiniMaxAI/MiniMax-M2.7/resolve/main/model.safetensors.index.json`
- `https://hf-mirror.com/zai-org/GLM-4.7/resolve/main/model.safetensors.index.json`

该审计只读取索引，不声称完成完整权重或 GPU 执行验证。
