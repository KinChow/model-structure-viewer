# 架构台账参考（Architectures Reference）

> **定位**：架构识别三层登记的机器台账——别名表（`architectures[0]` 精确键）→
> canonical architecture（模板分组与 multimodal 变体）→ 配方位（写不出 config
> 字段判据的家族知识）。机器段由 `node scripts/gen-model-reference.mjs` 生成，
> 经 `npm --prefix frontend run docs:check` 逐字节守护。

## 人工段：识别链与纪律

- 识别链：`config.architectures[0]` → `ARCHITECTURE_ALIASES` 精确匹配（**不做子串**，
  家族名兜底已按 principles §8.1 删除，认不出即 `unsupported`）→ canonical
  architecture（`ARCHITECTURE_CATALOG`）；`hasVision` 为真时升格 multimodal 变体
  （`resolveArchitecture.withVision`）。
- `ARCH_RECIPES` 只收「没有 config 字段判据」的配方位（`archs/index.js` 头注）；
  能用字段表达的一律走 `config/plan.js` 的字段判据，不进配方表——配方表条数
  只许收敛，不许用登记替代判据。
- 新家族接入触发 [`MAINTENANCE.md`](MAINTENANCE.md) 变更纪律 2/3b（先家族知识
  收口再接模型）；别名/配方位改动后机器段随之漂移，跑生成器重写即可，
  `docs:check` 保证台账不落后于源文件。

<!-- BEGIN GENERATED: architectures -->

> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**
> 事实源 = `frontend/src/structure/registry/aliases.js`（别名表：key=`architectures[0]` 原字符串，
> 精确匹配不做子串，principles §8.1 / vLLM `ModelRegistry` 对标）+
> `registry/architectureCatalog.js`（canonical → 模板能力与 multimodal 变体）+
> `structure/archs/index.js` 的 `ARCH_RECIPES`（只收「写不出 config 字段判据」的配方位）。
> `catalog 命中` = 59 内置模型的精确计数（命中 0 = 内置 catalog 无消费者，hf / config 等外部来源仍可命中该别名）；别名/配方顺序 = 源文件声明顺序。
> 别名命中后实际解析还会按 `hasVision` 升格 multimodal 变体（`resolveArchitecture.withVision`），
> 故别名列的 canonical 与模型台账列可能相差一个 `multimodal-` 前缀。

## 别名表 ARCHITECTURE_ALIASES（16 条）

| architectures[0] | canonical architecture（模板） | catalog 命中 |
|---|---|---|
| `DeepseekV3ForCausalLM` | `mla-moe-decoder` | 6 |
| `DeepseekV32ForCausalLM` | `mla-moe-decoder` | 1 |
| `DeepseekV4ForCausalLM` | `mla-moe-decoder` | 5 |
| `Glm4MoeForCausalLM` | `gqa-moe-decoder` | 1 |
| `GlmMoeDsaForCausalLM` | `mla-moe-decoder` | 6 |
| `Qwen3ForCausalLM` | `gqa-decoder` | 0 |
| `Qwen3_5ForConditionalGeneration` | `gqa-decoder` | 15 |
| `Qwen3_5MoeForConditionalGeneration` | `gqa-moe-decoder` | 12 |
| `Qwen3_5MoeForCausalLM` | `gqa-moe-decoder` | 2 |
| `Qwen3MoeForCausalLM` | `gqa-moe-decoder` | 0 |
| `Qwen4ExpForConditionalGeneration` | `multimodal-gqa-moe-decoder` | 2 |
| `KimiK25ForConditionalGeneration` | `mla-moe-decoder` | 3 |
| `KimiK3ForConditionalGeneration` | `hybrid-multimodal-moe-decoder` | 1 |
| `Glm5NextForConditionalGeneration` | `hybrid-multimodal-moe-decoder` | 2 |
| `MiniMaxM2ForCausalLM` | `gqa-moe-decoder` | 1 |
| `MiniMaxM3SparseForConditionalGeneration` | `multimodal-sparse-moe-decoder` | 2 |

## canonical architecture 目录 ARCHITECTURE_CATALOG（9 条）

| canonical | 有模板 | multimodal 变体 | catalog 模型数 |
|---|---|---|---|
| `gqa-decoder` | ✓ | `multimodal-gqa-decoder` | 0 |
| `gqa-moe-decoder` | ✓ | `multimodal-gqa-moe-decoder` | 4 |
| `mla-moe-decoder` | ✓ | `multimodal-mla-moe-decoder` | 17 |
| `multimodal-gqa-decoder` | ✓ | — | 15 |
| `multimodal-sparse-moe-decoder` | ✓ | — | 2 |
| `multimodal-gqa-moe-decoder` | ✓ | — | 14 |
| `multimodal-mla-moe-decoder` | ✓ | — | 4 |
| `hybrid-multimodal-moe-decoder` | ✓ | — | 3 |
| `unsupported` | — | — | 0 |

## 配方表 ARCH_RECIPES（14 条）

> 四个配方位（normMode / linearAttentionMode / visionInternalMerger / sharedExpertsAreFused）在 config 里没有对应字段，属人工登记的
> 家族知识（archs/index.js 头注：显式声明比藏在 `model_type.includes(...)` 里诚实）；
> 能用 config 字段表达的判据一律走 `config/plan.js`，不进配方表。

| architectures[0] | normMode | linearAttentionMode | visionInternalMerger | sharedExpertsAreFused |
|---|---|---|---|---|
| `DeepseekV3ForCausalLM` | — | — | — | — |
| `DeepseekV32ForCausalLM` | — | — | — | — |
| `DeepseekV4ForCausalLM` | — | — | — | — |
| `Glm4MoeForCausalLM` | — | — | — | — |
| `GlmMoeDsaForCausalLM` | — | — | — | — |
| `Glm5NextForConditionalGeneration` | — | `glm5_next` | ✓ | — |
| `KimiK25ForConditionalGeneration` | — | `kimi` | — | — |
| `KimiK3ForConditionalGeneration` | — | `kimi_k3` | — | ✓ |
| `MiniMaxM2ForCausalLM` | — | — | — | — |
| `MiniMaxM3SparseForConditionalGeneration` | `gemma_rmsnorm` | — | — | — |
| `Qwen3_5ForConditionalGeneration` | `gemma_rmsnorm` | `qwen3_5` | ✓ | — |
| `Qwen3_5MoeForCausalLM` | `gemma_rmsnorm` | `qwen3_5` | — | — |
| `Qwen3_5MoeForConditionalGeneration` | `gemma_rmsnorm` | `qwen3_5` | ✓ | — |
| `Qwen4ExpForConditionalGeneration` | — | `qwen4_exp` | ✓ | — |

<!-- END GENERATED: architectures -->
