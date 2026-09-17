# 实现细节：模型

## 模型清单

模型清单的唯一事实来源是 [`models/catalog.json`](../../models/catalog.json)。每个条目至少包含：

- `model_id`：来源仓库 ID，也是模型配置目录的相对路径。
- `config_path`：仓库内 `models/<org>/<model>/config.json` 的路径。
- `model_type`、`architectures`：模型识别和 registry 解析输入。
- `release_time`：公开模型仓库创建时间的代理值。
- `metadata_files`、`verified` 和 `validation_report`：轻量元数据和验证状态。

新增模型的最小流程是：添加配置和必要元数据，运行 catalog 生成，补充架构 alias/builder（如果现有结构不能复用），运行前端组网验证，再更新模型专项说明。

## 结构类台账（W1）

`canonical architecture` 是**组网模板**的分组；开发与验收还需要更细的一层：
**结构类**。判据全部取 `config.json` 的结构性字段存在性（对标 vLLM
`ModelRegistry` 用 `architectures[0]` 精确键 + 显式字段派生逐层方案，不用
`model_type` 子串），量化正交剔除（只影响 bytes 与参数字节）。

59 个内置模型去重后 **16 个结构类**。开发单元是**部件**（22 个），结构类只是
部件的组合；每个结构类挑一个代表模型做验收。

### 部件清单（22 个）

- 注意力 8：`GQA(+outgate)` · `MLA` · `DSA` · `DSA-kpool` · `QSA` ·
  `DSV4(compress+swa)` · `MSA(block)` · `LinAttn(GDN/KDA)`
- FFN 5：`DenseMLP` · `MoE` · `+shared` · `+hashroute` · `+latentMoE`
- 残差 4：`plain` · `MHC` · `HyperConn+PLE` · `AttnResBlock`
- 视觉 4：Qwen-VL 系（含 merger）· Kimi ViT · MiniMax projector · GLM-Flash
- MTP 1（层数为参数：1 或 3）

### 结构类与代表模型

| 类 | architectures[0] | 注意力 | FFN | 残差 | 视觉 | MTP | 模型数 | 代表模型 |
|---|---|---|---|---|---|---|---|---|
| S01 | `Qwen3_5ForConditionalGeneration` | GQA+outgate / LinAttn | DenseMLP | plain | 有 | 1 | 15 | `Qwen/Qwen3.5-0.8B` |
| S02 | `Qwen3_5MoeForConditionalGeneration` | GQA+outgate / LinAttn | MoE+shared | plain | 有 | 1 | 12 | `Qwen/Qwen3.5-35B-A3B` |
| S03 | `GlmMoeDsaForCausalLM` | DSA | MoE+shared | plain | 无 | 1 | 6 | `zai-org/GLM-5` |
| S04 | `DeepseekV3ForCausalLM` | MLA | MoE+shared | plain | 无 | 无 | 4 | `moonshotai/Kimi-K2-Instruct` |
| S05 | `DeepseekV4ForCausalLM` | DSV4 | MoE+shared+hash | MHC | 无 | 1 | 4 | `deepseek-ai/DeepSeek-V4-Pro` |
| S06 | `KimiK25ForConditionalGeneration` | MLA | MoE+shared | plain | 有 | 无 | 3 | `moonshotai/Kimi-K2.5` |
| S07 | `DeepseekV3ForCausalLM` | MLA | MoE+shared | plain | 无 | 1 | 2 | `deepseek-ai/DeepSeek-V3.1` |
| S08 | `Glm5NextForConditionalGeneration` | DSA-kpool / LinAttn | MoE+shared | MHC | 有 | 1 | 2 | `zai-org/GLM-5.3-Flash` |
| S09 | `MiniMaxM3SparseForConditionalGeneration` | MSA | MoE+shared | plain | 有 | 1 | 2 | `MiniMaxAI/MiniMax-M3` |
| S10 | `Qwen3_5MoeForCausalLM` | GQA+outgate / LinAttn | MoE+shared | plain | 无 | 1 | 2 | `Qwen/Qwen3.8-2.4T-A95B` |
| S11 | `Qwen4ExpForConditionalGeneration` | QSA / LinAttn | MoE+shared | HyperConn+PLE | 有 | 1 | 2 | `Qwen/Qwen3.8-Flash-Next` |
| S12 | `DeepseekV32ForCausalLM` | DSA | MoE+shared | plain | 无 | 1 | 1 | `deepseek-ai/DeepSeek-V3.2` |
| S13 | `DeepseekV4ForCausalLM` | DSV4 | MoE+shared+hash | MHC | 有（平铺） | 3 | 1 | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` |
| S14 | `Glm4MoeForCausalLM` | GQA | MoE+shared | plain | 无 | 1 | 1 | `zai-org/GLM-4.7` |
| S15 | `KimiK3ForConditionalGeneration` | MLA / LinAttn | MoE+shared+latentMoE | AttnResBlock | 有 | 无 | 1 | `moonshotai/Kimi-K3` |
| S16 | `MiniMaxM2ForCausalLM` | GQA | MoE | plain | 无 | 3 | 1 | `MiniMaxAI/MiniMax-M2.7` |

判据字段（可机械复现，不含家族名）：

- `sparse_attention_config.sparse_block_size` → MSA 块稀疏
- `compress_ratios` 数组 → DSV4 压缩/滑窗混合
- `index_topk` + `kv_lora_rank` + `q_lora_rank` → DSA over MLA；再看
  `index_kpool > 1` 决定是否 kpool 变体
- `indexer_budget` + `indexer_kv_heads` + `indexer_compress_ratio` → QSA
- `linear_attn_config` 或 `linear_num_key_heads` → 线性注意力层
- `num_nextn_predict_layers` / `mtp_num_hidden_layers` / `num_mtp_modules` → MTP

### S13 的视觉判定（W1 核查结论）

`DeepSeek-V4-Flash-Vision-Exp` **没有**嵌套 `vision_config`，视觉参数是平铺的
`vision_n_layers` / `vision_dim` / `vision_n_heads` / `vision_inter_dim` /
`vision_patch_size`。`normalizeConfig` 的 `flatVisionConfig` 分支已正确识别
（实测 `hasVision=true`、`visionLayers=32`、`visionHiddenSize=1024`），
`resolveArchitecture` 也归入 `multimodal-mla-moe-decoder`——**不是代码缺陷**，
早前台账脚本只查嵌套键才误判为无视觉。

真实缺口是另一条：该模型 `visionTokens` 为 `undefined`（平铺配置没有
`image_size`，`visionPatchTokenCount` 的三条回退全不命中），视觉域成本因此按
`visionTokens || 1` 计。config 里的 `vision_max_n_token` 未被消费，是补齐这项的
候选来源。登记为 W4 视觉部件项。

<!-- BEGIN GENERATED: details-models -->

> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**
> 当前 `models/catalog.json` 收录 60 个内置模型，按 `architectures[0]` 分组。

## 当前已支持模型（60 个，按 architectures[0]）

### `Qwen3_5ForConditionalGeneration`：15 个

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

### `Qwen3_5MoeForConditionalGeneration`：12 个

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

### `DeepseekV3ForCausalLM`：6 个

- `deepseek-ai/DeepSeek-R1`
- `deepseek-ai/DeepSeek-V3.1`
- `moonshotai/Kimi-K2-Base`
- `moonshotai/Kimi-K2-Instruct`
- `moonshotai/Kimi-K2-Instruct-0905`
- `moonshotai/Kimi-K2-Thinking`

### `GlmMoeDsaForCausalLM`：6 个

- `zai-org/GLM-5`
- `zai-org/GLM-5.1`
- `zai-org/GLM-5.2`
- `zai-org/GLM-5.2-FP8`
- `zai-org/GLM-5.3`
- `zai-org/GLM-5.3-BF16`

### `DeepseekV4ForCausalLM`：5 个

- `deepseek-ai/DeepSeek-V4-Flash`
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- `deepseek-ai/DeepSeek-V4-Pro`
- `deepseek-ai/DeepSeek-V4-Pro-0813`

### `KimiK25ForConditionalGeneration`：3 个

- `moonshotai/Kimi-K2.5`
- `moonshotai/Kimi-K2.6`
- `moonshotai/Kimi-K2.7-Code`

### `Glm5NextForConditionalGeneration`：2 个

- `zai-org/GLM-5.3-Flash`
- `zai-org/GLM-5.3-Flash-BF16`

### `MiniMaxM3SparseForConditionalGeneration`：2 个

- `MiniMaxAI/MiniMax-M3`
- `MiniMaxAI/MiniMax-M3-MXFP8`

### `Qwen3_5MoeForCausalLM`：2 个

- `Qwen/Qwen3.8-2.4T-A95B`
- `Qwen/Qwen3.8-2.4T-A95B-FP8`

### `Qwen4ExpForConditionalGeneration`：2 个

- `Qwen/Qwen3.8-Flash-Next`
- `Qwen/Qwen3.8-Flash-Next-FP8`

### `DeepseekV32ForCausalLM`：1 个

- `deepseek-ai/DeepSeek-V3.2`

### `DeepseekV41ForCausalLM`：1 个

- `deepseek-ai/DeepSeek-V4.1-Flash`

### `Glm4MoeForCausalLM`：1 个

- `zai-org/GLM-4.7`

### `KimiK3ForConditionalGeneration`：1 个

- `moonshotai/Kimi-K3`

### `MiniMaxM2ForCausalLM`：1 个

- `MiniMaxAI/MiniMax-M2.7`

<!-- END GENERATED: details-models -->

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
| `auto` | CLI/API 兼容模式 | 按 builtin、local、远程来源回退；网页入口使用明确端点 |
| `config` | 粘贴或上传 JSON | 直接使用用户配置，必要时读取本地 header |

来源解析、缓存策略和 API 参数见 [`source_resolution.md`](models/source_resolution.md)。

## 发布时间

`release_time` 当前保存为 ISO 8601 UTC 字符串。它优先使用 ModelScope 的 `Data.CreatedTime`；ModelScope 没有对应仓库时使用 Hugging Face 镜像 API 的 `createdAt`。这是仓库创建时间代理值，不等同于官方公告或论文发布时间。

详细查询时间、接口和字段边界见 [`release_metadata.md`](models/release_metadata.md)。

## 目录与缓存边界

内置静态资源只发布：

- `config.json`
- `catalog.json`

网页运行时按远程端点读取 safetensors header，不下载权重数据区。后端缓存和本地模型目录可以保留 `README.md`、自定义 Python 代码等验证辅助文件；`auto_fetch_remote_code=false` 时不会联网获取这些代码。

## 模型适配归属

- 官方 architecture 查找键：`frontend/src/structure/models/index.js` 的 `MODELS`
- 顶层组网：`frontend/src/structure/models/`
- 可复用层：`frontend/src/structure/layers/`
- 算子公式与工厂：`frontend/src/structure/operators/`
- 专项模型说明：`docs/details/models/`


## 模型目录内的证据文件（M8-V2 起）

部分模型目录除 `config.json` 外还带有校准证据文件（HF hub 单模型仓库惯例：
模型相关文件同仓）：

- `modeling_*.py` / `configuration_*.py`：官方 modeling 源码（L2 结构语义证据，
  公式对照依据）；
- `model.safetensors.index.json`：权重清单原件（L3，体积大不入 git，
  由 manifest 的来源 URL 可重下）；
- `index-summary.json`：index 派生摘要——逐层张量模式（入 git）；
- `evidence-manifest.json`：证据清单——文件名/级别（L1 config、L2 modeling、
  L3 index）/来源 URL/下载日期；
- `kimi-linear-analysis.md` 等分析报告。

取证入口：`node scripts/fetch-evidence.mjs <org>/<id> --probe <文件名列表>`。
用途与纪律见 [`MAINTENANCE.md`](../MAINTENANCE.md) 纪律 3b/3c 与
[`identity_calibration.md`](identity_calibration.md)。
