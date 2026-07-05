# 内置模型配置

这个目录保存已经适配过的模型配置，方便直接启动和验证。

目录结构保持 Hugging Face model id 的形式：

```text
models/<org>/<model>/config.json
```

这里只放轻量文件：

- `config.json`
- `configuration_*.py`
- `modeling_*.py`
- `tokenization_*.py`

这里不放权重文件。`.safetensors`、`.bin`、`.gguf`、`.pt` 和 `.onnx` 已经在 `.gitignore` 里拦住。

`catalog.json` 是前端静态部署使用的模型索引。新增或删除模型后，在仓库根目录执行：

```bash
node scripts/generate-model-catalog.mjs
```

## 快速使用

在仓库根目录执行：

```bash
.venv/bin/msv list --root ./models
.venv/bin/msv serve --root ./models --port 8000
```

前端启动后选择 `builtin` 或 `auto`，model id 直接填下面这些值即可。
如果需要验证后端本地模型目录，选择 `local`。

## 当前模型

### Qwen

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-0.8B-Base`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-2B-Base`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-4B-Base`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.5-27B`
- `Qwen/Qwen3.5-27B-FP8`
- `Qwen/Qwen3.5-27B-GPTQ-Int4`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-35B-A3B-Base`
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`
- `Qwen/Qwen3.5-122B-A10B`
- `Qwen/Qwen3.5-122B-A10B-FP8`
- `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `Qwen/Qwen3.5-397B-A17B`
- `Qwen/Qwen3.5-397B-A17B-FP8`
- `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4`
- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`

### DeepSeek

- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-V3.1`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V4-Flash`
- `deepseek-ai/DeepSeek-V4-Pro`

### GLM

- `zai-org/GLM-4.7`
- `zai-org/GLM-5`
- `zai-org/GLM-5.1`
- `zai-org/GLM-5.2`

### Kimi

- `moonshotai/Kimi-K2-Base`
- `moonshotai/Kimi-K2-Instruct`
- `moonshotai/Kimi-K2-Instruct-0905`
- `moonshotai/Kimi-K2-Thinking`
- `moonshotai/Kimi-K2.5`
- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.7-Code`

### MiniMax

- `MiniMaxAI/MiniMax-M2.7`
- `MiniMaxAI/MiniMax-M3`
- `MiniMaxAI/MiniMax-M3-MXFP8`
