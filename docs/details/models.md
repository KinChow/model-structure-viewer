# 实现细节：模型

## 模型清单

模型清单的唯一事实来源是 [`models/catalog.json`](../../models/catalog.json)。每个条目至少包含：

- `model_id`：来源仓库 ID，也是模型配置目录的相对路径。
- `config_path`：仓库内 `models/<org>/<model>/config.json` 的路径。
- `model_type`、`architectures`：模型识别和 registry 解析输入。
- `release_time`：公开模型仓库创建时间的代理值。
- `metadata_files`、`verified` 和 `validation_report`：轻量元数据和验证状态。

新增模型的最小流程是：添加配置和必要元数据，运行 catalog 生成，补充架构 alias/builder（如果现有结构不能复用），运行前端组网验证，再更新模型专项说明。

## 当前已支持模型

当前 `models/catalog.json` 收录 59 个内置模型。下表按运行时 `resolveArchitecture` 得到的 canonical architecture 分组；这些模型均可通过 `builtin` 入口读取仓库配置并进入前端结构生成链路，模型列表以 catalog 为准。

### `gqa-decoder`：15 个

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-27B-FP8`
- `Qwen/Qwen3.5-27B-GPTQ-Int4`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-4B-Base`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.8-27B`
- `Qwen/Qwen3.8-27B-FP8`

### `gqa-moe-decoder`：16 个

- `MiniMaxAI/MiniMax-M2.7`
- `Qwen/Qwen3.5-122B-A10B`
- `Qwen/Qwen3.5-122B-A10B-FP8`
- `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-35B-A3B-Base`
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-397B-A17B-FP8`
- `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.8-2.4T-A95B`
- `Qwen/Qwen3.8-2.4T-A95B-FP8`
- `zai-org/GLM-4.7`

### `mla-moe-decoder`：21 个

- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V4-Flash`
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- `deepseek-ai/DeepSeek-V4-Pro`
- `deepseek-ai/DeepSeek-V4-Pro-0813`
- `moonshotai/Kimi-K2-Base`
- `moonshotai/Kimi-K2-Instruct`
- `moonshotai/Kimi-K2-Instruct-0905`
- `moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`
- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.7-Code`
- `zai-org/GLM-5`
- `zai-org/GLM-5.1`
- `zai-org/GLM-5.2`
- `zai-org/GLM-5.2-FP8`
- `zai-org/GLM-5.3`
- `zai-org/GLM-5.3-BF16`

### `multimodal-sparse-moe-decoder`：2 个

- `MiniMaxAI/MiniMax-M3`
- `MiniMaxAI/MiniMax-M3-MXFP8`

### `multimodal-gqa-moe-decoder`：2 个

- `Qwen/Qwen3.8-Flash-Next`
- `Qwen/Qwen3.8-Flash-Next-FP8`

### `hybrid-multimodal-moe-decoder`：3 个

- `moonshotai/Kimi-K3`
- `zai-org/GLM-5.3-Flash`
- `zai-org/GLM-5.3-Flash-BF16`

列表校验命令：

```bash
npm --prefix frontend run verify:models
```

该命令验证前端 builtin 组网，不代表每个模型都能通过后端 transformers meta-device 验证，也不代表权重存在或可以运行推理。没有出现在 catalog 的模型仍可能通过 `hf`、`config` 或 generic fallback 生成部分结构，但不属于上述内置支持列表。

## 支持来源

| 来源 | 入口 | 主要行为 |
|---|---|---|
| `builtin` | `models/` 和静态 `/models/` | 使用仓库内置配置，不需要后端 |
| `local` | `MODEL_ROOT/<org>/<model>/config.json` | 读取本地配置，需要 API |
| `hf` | Hugging Face 或 ModelScope | 读取公开配置和允许的轻量元数据，不下载权重 |
| `auto` | 前端/后端默认模式 | 按 builtin、local、远程来源回退 |
| `config` | 粘贴或上传 JSON | 直接使用用户配置，必要时读取本地 header |

来源解析、缓存策略和 API 参数见 [`source_resolution.md`](models/source_resolution.md)。

## 发布时间

`release_time` 当前保存为 ISO 8601 UTC 字符串。它优先使用 ModelScope 的 `Data.CreatedTime`；ModelScope 没有对应仓库时使用 Hugging Face 镜像 API 的 `createdAt`。这是仓库创建时间代理值，不等同于官方公告或论文发布时间。

详细查询时间、接口和字段边界见 [`release_metadata.md`](models/release_metadata.md)。

## 目录与缓存边界

允许缓存的轻量文件包括：

- `config.json`
- `README.md`
- `model.safetensors.index.json`
- `configuration_*.py`、`modeling_*.py`、`tokenization_*.py`

工具不缓存 `.safetensors`、`.bin`、`.gguf`、`.pt`、`.pth`、`.onnx` 等权重或推理文件。后端验证可能执行本地 remote code，但这是验证路径，不改变前端静态路径的轻量边界。

## 模型适配归属

- 模型 ID 或官方 architecture 别名：`frontend/src/structure/registry/aliases.js`
- canonical architecture：`frontend/src/structure/registry/architectureCatalog.js`
- 顶层组网：`frontend/src/structure/model_executor/models/`
- 可复用层：`frontend/src/structure/model_executor/layers/`
- 专项模型说明：`docs/details/models/`
