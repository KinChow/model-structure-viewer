# 架构台账参考（Architectures Reference）

> **定位**：架构识别的机器台账——`MODELS` 注册表（`architectures[0]` 精确键，
> 对标 vLLM `_TEXT_GENERATION_MODELS` / SGLang `_ModelRegistry.models`）+
> 配方表（类名 / 路径例外）。机器段由 `node scripts/gen-model-reference.mjs` 生成，
> 经 `npm --prefix frontend run docs:check` 逐字节守护。

## 人工段：识别链与纪律

- 识别链：`config.architectures[0]` 精确查 `MODELS`（**不做子串**，认不出即 `unsupported`）。
  视觉塔是该建模函数内部的可选子模块，不升格第二种架构名。
- `ARCH_RECIPES` 只收类名 / 路径例外与没有 config 字段判据的配方位；
  能用字段表达的一律走 `layers/schedule.js`。
- 新架构接入：registry 一行 + 一份组装函数；台账随后由生成器重写，
  `docs:check` 保证不落后于源文件。

<!-- BEGIN GENERATED: architectures -->

> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**
> 事实源 = `frontend/src/structure/models/index.js` 的 `MODELS`
>（key=`architectures[0]` 原字符串，对标 vLLM `_TEXT_GENERATION_MODELS` / SGLang `_ModelRegistry.models`）+
> `structure/archs/index.js` 的 `ARCH_RECIPES`（类名 / 路径例外）。
> `catalog 命中` = 59 内置模型的精确计数。

## MODELS 注册表（16 条）

| architectures[0] | catalog 命中 |
|---|---|
| `DeepseekV3ForCausalLM` | 6 |
| `DeepseekV32ForCausalLM` | 1 |
| `DeepseekV4ForCausalLM` | 5 |
| `Glm4MoeForCausalLM` | 1 |
| `GlmMoeDsaForCausalLM` | 6 |
| `Qwen3ForCausalLM` | 0 |
| `Qwen3_5ForConditionalGeneration` | 15 |
| `Qwen3_5MoeForConditionalGeneration` | 12 |
| `Qwen3_5MoeForCausalLM` | 2 |
| `Qwen3MoeForCausalLM` | 0 |
| `Qwen4ExpForConditionalGeneration` | 2 |
| `KimiK25ForConditionalGeneration` | 3 |
| `KimiK3ForConditionalGeneration` | 1 |
| `Glm5NextForConditionalGeneration` | 2 |
| `MiniMaxM2ForCausalLM` | 1 |
| `MiniMaxM3SparseForConditionalGeneration` | 2 |

## 配方表 ARCH_RECIPES（11 条）

> 四个配方位（normMode / linearAttentionMode / visionInternalMerger / sharedExpertsAreFused）在 config 里没有对应字段，属人工登记的
> 家族知识（archs/index.js 头注：显式声明比藏在 `model_type.includes(...)` 里诚实）；
> 能用 config 字段表达的判据一律走 `layers/schedule.js`，不进配方表。

| architectures[0] | normMode | linearAttentionMode | visionInternalMerger | sharedExpertsAreFused | moeClass | hashMoE | visionMergerMlp | visionAttr | decoderLayerClass | mlpClass | rmsNormClass | attentionClass | modelClass | ffn | latentMoE | fusedQkv | sigmoidRouter | visionBlockClass | visionModelClass | layersAttr | swigluVariant | attention | gatedResidualClass | layerMix | pleClass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `DeepseekV4ForCausalLM` | — | — | — | — | `DeepseekV4SparseMoeBlock` | ✓ | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `Glm5NextForConditionalGeneration` | — | `glm5_next` | ✓ | — | `Glm5NextTextMoE` | — | ✓ | `visual` | `Glm5NextTextDecoderLayer` | `Glm5NextTextMLP` | `Glm5NextTextRMSNorm` | `linear=Glm5NextTextLinearAttention, gqa=Glm5NextTextAttention, qwen35_full=Glm5NextTextAttention` | — | — | — | — | — | — | — | — | — | — | — | — | — |
| `KimiK25ForConditionalGeneration` | — | `kimi` | — | — | `DeepseekV3MoE` | — | — | `vision_tower` | `DeepseekV3DecoderLayer` | `DeepseekV3MLP` | `DeepseekV3RMSNorm` | `DeepseekV3Attention` | `DeepseekV3Model` | — | — | — | — | — | — | — | — | — | — | — | — |
| `KimiK3ForConditionalGeneration` | — | `kimi_k3` | — | ✓ | `KimiSparseMoeBlock` | — | — | `vision_tower` | `KimiDecoderLayer` | `KimiMLP` | `KimiRMSNorm` | `mla=KimiMLAAttention, linear=KimiDeltaAttention` | `KimiLinearModel` | `moe=block_sparse_moe` | ✓ | — | — | — | — | — | — | — | — | — | — |
| `MiniMaxM2ForCausalLM` | — | — | — | — | `MiniMaxM2SparseMoeBlock` | — | — | — | — | — | — | — | — | `block_sparse_moe` | — | ✓ | ✓ | — | — | — | — | — | — | — | — |
| `Glm4MoeForCausalLM` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | ✓ | ✓ | — | — | — | — | — | — | — | — |
| `MiniMaxM3SparseForConditionalGeneration` | `gemma_rmsnorm` | — | — | — | `MiniMaxM3VLSparseMoeBlock` | — | — | `vision_tower` | `MiniMaxM3VLDecoderLayer` | `MiniMaxM3VLDenseMLP` | `MiniMaxM3VLRMSNorm` | `MiniMaxM3VLAttention` | `MiniMaxM3VLTextModel` | — | — | — | ✓ | `MiniMaxM3VLVisionEncoderLayer` | `MiniMaxM3VLVisionModel` | `language_model` | `swigluoai` | — | — | — | — |
| `Qwen3_5ForConditionalGeneration` | `gemma_rmsnorm` | `qwen3_5` | ✓ | — | — | — | — | `visual` | `Qwen3_5DecoderLayer` | `Qwen3_5MLP` | `Qwen3_5RMSNorm` | `linear=Qwen3_5GatedDeltaNet, qwen35_full=Qwen3_5Attention` | `Qwen3_5TextModel` | — | — | — | — | — | — | — | — | `linear=linear_attn` | — | — | — |
| `Qwen3_5MoeForCausalLM` | `gemma_rmsnorm` | `qwen3_5` | — | — | `Qwen3_5MoeSparseMoeBlock` | — | — | — | `Qwen3_5MoeDecoderLayer` | `Qwen3_5MoeMLP` | `Qwen3_5MoeRMSNorm` | `linear=Qwen3_5MoeGatedDeltaNet, qwen35_full=Qwen3_5MoeAttention` | — | — | — | — | — | — | — | — | — | `linear=linear_attn` | — | — | — |
| `Qwen3_5MoeForConditionalGeneration` | `gemma_rmsnorm` | `qwen3_5` | ✓ | — | `Qwen3_5MoeSparseMoeBlock` | — | — | `visual` | `Qwen3_5MoeDecoderLayer` | `Qwen3_5MoeMLP` | `Qwen3_5MoeRMSNorm` | `linear=Qwen3_5MoeGatedDeltaNet, qwen35_full=Qwen3_5MoeAttention` | `Qwen3_5MoeTextModel` | — | — | — | — | — | — | — | — | `linear=linear_attn` | — | — | — |
| `Qwen4ExpForConditionalGeneration` | — | `qwen4_exp` | ✓ | — | `Qwen4ExpTextSparseMoeBlock` | — | — | `visual` | `Qwen4ExpTextDecoderLayer` | `Qwen4ExpTextMLP` | `Qwen4ExpTextRMSNorm` | `linear=Qwen4ExpTextGatedDeltaNet, qsa=Qwen4ExpTextAttention, gqa=Qwen4ExpTextAttention` | — | — | — | — | — | — | — | — | — | `linear=linear_attn` | `Qwen4ExpTextGatedResidual` | `hyper_connection` | `Qwen4ExpTextPLELayer` |

<!-- END GENERATED: architectures -->
