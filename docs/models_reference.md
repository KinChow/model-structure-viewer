# 模型台账参考（Models Reference）

> **定位**：59 个内置模型的机器台账（清单事实）+ 人工段（解读入口）。
> 机器段由 `node scripts/gen-model-reference.mjs` 生成，经 `npm --prefix frontend
> run docs:check` 逐字节守护；带锚注解的人工视图（结构类台账 16 类/22 部件、
> 按 canonical 分组的模型列表、模型专项说明）住在
> [`details/models.md`](details/models.md)，两处分工：本页是「账」，details 是「读」。

## 人工段：怎么读这张表

- `architectures[0]` 是运行时 `resolveArchitecture` 的查找键（vLLM `_MODELS`）；
  更细的**结构类**（16 类，判据全部取 config 结构性字段）与代表模型见
  details/models.md 的「结构类台账（W1）」。
- `参数量级（图声明）` 来自 `graphWeightCapacity(graph).elements`：walk 图上的 `weightMatrices`，
  与 UI 摘要（SummaryChips）同一函数。无 checkpoint、无闭式。
- `证据库` 指向 `models/<org>/<id>/`：`manifest` = 该目录带 `evidence-manifest.json`
  （L1 config / L2 modeling / L3 index 摘要），取证入口与纪律见
  [`MAINTENANCE.md`](MAINTENANCE.md) 变更纪律 3b 与
  [`details/identity_calibration.md`](details/identity_calibration.md)。
- 表由脚本生成意味着：新增/删除模型只需进 catalog（`npm --prefix frontend run catalog`），
  台账随后由 `docs:check` 强制同步，不设第二份手写清单。
- 口径权威链：本表 `architectures[0]` 列来自 catalog + 运行时实跑（与 `verify:models` 同源）。若与
  details/models.md 的人工分组出现出入，以本表为准并回改人工段（照 operators_reference.md 的权威链惯例）。

<!-- BEGIN GENERATED: models -->

> **本节由 `node scripts/gen-model-reference.mjs` 生成，请勿手改。**
> 数据源 = `models/catalog.json` + 前端运行时链路（`normalizeConfig` → `resolveArchitecture`，
> 与 `frontend/src/structure/builtinModels.test.js` 同口径）；`参数量级` =
> `graphWeightCapacity(graph).elements`（walk 图声明，与 UI SummaryChips 同一函数；
> 无 checkpoint 时不编造闭式。取整到参数个位，缩写沿用 formatters.js 的 B/M 档位并补 T 档）；
> `证据库` = `models/<org>/<id>/` 目录存在性，
> `manifest` = 目录内另有 `evidence-manifest.json`（见 details/models.md「模型目录内的证据文件」）；
> `release_time` 取 catalog ISO 串的日期段，完整时刻以 `models/catalog.json` 为准。

## 模型台账（59 个内置模型，按 model_id 码元序）

| 模型 ID | family（model_type） | architectures[0] | 参数量级（图声明） | 证据库 | release_time |
|---|---|---|---|---|---|
| `MiniMaxAI/MiniMax-M2.7` | `minimax_m2` | `MiniMaxM2ForCausalLM` | 239,752,106,240（239.8B） | 有 | 2026-04-09 |
| `MiniMaxAI/MiniMax-M3` | `minimax_m3_vl` | `MiniMaxM3SparseForConditionalGeneration` | 432,565,191,424（432.6B） | manifest | 2026-06-02 |
| `MiniMaxAI/MiniMax-M3-MXFP8` | `minimax_m3_vl` | `MiniMaxM3SparseForConditionalGeneration` | 432,565,191,424（432.6B） | 有 | 2026-06-02 |
| `Qwen/Qwen3.5-0.8B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 871,564,608（871.6M） | 有 | 2026-03-01 |
| `Qwen/Qwen3.5-0.8B-Base` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 871,564,608（871.6M） | 有 | 2026-03-01 |
| `Qwen/Qwen3.5-122B-A10B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 125,083,351,808（125.1B） | 有 | 2026-02-24 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 125,083,351,808（125.1B） | 有 | 2026-02-25 |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 125,083,351,808（125.1B） | 有 | 2026-03-03 |
| `Qwen/Qwen3.5-27B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-02-24 |
| `Qwen/Qwen3.5-27B-FP8` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-02-25 |
| `Qwen/Qwen3.5-27B-GPTQ-Int4` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-03-03 |
| `Qwen/Qwen3.5-2B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 2,271,435,072（2.3B） | 有 | 2026-03-01 |
| `Qwen/Qwen3.5-2B-Base` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 2,271,435,072（2.3B） | 有 | 2026-03-01 |
| `Qwen/Qwen3.5-35B-A3B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-02-24 |
| `Qwen/Qwen3.5-35B-A3B-Base` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-02-24 |
| `Qwen/Qwen3.5-35B-A3B-FP8` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-02-25 |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-03-03 |
| `Qwen/Qwen3.5-397B-A17B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 403,394,683,392（403.4B） | 有 | 2026-02-16 |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 403,394,683,392（403.4B） | 有 | 2026-02-18 |
| `Qwen/Qwen3.5-397B-A17B-GPTQ-Int4` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 403,394,683,392（403.4B） | 有 | 2026-03-03 |
| `Qwen/Qwen3.5-4B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 4,657,229,824（4.7B） | 有 | 2026-02-27 |
| `Qwen/Qwen3.5-4B-Base` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 4,657,229,824（4.7B） | 有 | 2026-02-27 |
| `Qwen/Qwen3.5-9B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 9,650,108,672（9.7B） | 有 | 2026-02-27 |
| `Qwen/Qwen3.5-9B-Base` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 9,650,108,672（9.7B） | 有 | 2026-02-27 |
| `Qwen/Qwen3.6-27B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-04-22 |
| `Qwen/Qwen3.6-27B-FP8` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-04-22 |
| `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-04-16 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 35,948,745,088（35.9B） | 有 | 2026-04-16 |
| `Qwen/Qwen3.8-2.4T-A95B` | `qwen3_5_moe_text` | `Qwen3_5MoeForCausalLM` | 2,446,181,963,648（2.45T） | 有 | 2026-08-08 |
| `Qwen/Qwen3.8-2.4T-A95B-FP8` | `qwen3_5_moe_text` | `Qwen3_5MoeForCausalLM` | 2,446,181,963,648（2.45T） | 有 | 2026-08-08 |
| `Qwen/Qwen3.8-27B` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-08-12 |
| `Qwen/Qwen3.8-27B-FP8` | `qwen3_5` | `Qwen3_5ForConditionalGeneration` | 27,778,431,232（27.8B） | 有 | 2026-08-13 |
| `Qwen/Qwen3.8-Flash-Next` | `qwen4_exp` | `Qwen4ExpForConditionalGeneration` | 179,510,522,496（179.5B） | manifest | 2026-08-24 |
| `Qwen/Qwen3.8-Flash-Next-FP8` | `qwen4_exp` | `Qwen4ExpForConditionalGeneration` | 179,510,522,496（179.5B） | 有 | 2026-08-24 |
| `deepseek-ai/DeepSeek-R1` | `deepseek_v3` | `DeepseekV3ForCausalLM` | 682,636,481,024（682.6B） | 有 | 2025-01-20 |
| `deepseek-ai/DeepSeek-V3.1` | `deepseek_v3` | `DeepseekV3ForCausalLM` | 682,636,481,024（682.6B） | 有 | 2025-08-21 |
| `deepseek-ai/DeepSeek-V3.2` | `deepseek_v32` | `DeepseekV32ForCausalLM` | 683,501,948,672（683.5B） | 有 | 2025-12-01 |
| `deepseek-ai/DeepSeek-V4-Flash` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 290,893,582,669（290.9B） | manifest | 2026-04-22 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 304,129,384,121（304.1B） | 有 | 2026-07-31 |
| `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 304,453,013,177（304.5B） | 有 | 2026-08-31 |
| `deepseek-ai/DeepSeek-V4-Pro` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 1,598,716,624,153（1.60T） | 有 | 2026-04-22 |
| `deepseek-ai/DeepSeek-V4-Pro-0813` | `deepseek_v4` | `DeepseekV4ForCausalLM` | 1,650,374,885,253（1.65T） | 有 | 2026-08-13 |
| `moonshotai/Kimi-K2-Base` | `kimi_k2` | `DeepseekV3ForCausalLM` | 1,026,408,209,408（1.03T） | 有 | 2025-07-10 |
| `moonshotai/Kimi-K2-Instruct` | `kimi_k2` | `DeepseekV3ForCausalLM` | 1,026,408,209,408（1.03T） | 有 | 2025-07-11 |
| `moonshotai/Kimi-K2-Instruct-0905` | `kimi_k2` | `DeepseekV3ForCausalLM` | 1,026,408,209,408（1.03T） | 有 | 2025-09-04 |
| `moonshotai/Kimi-K2-Thinking` | `kimi_k2` | `DeepseekV3ForCausalLM` | 1,026,408,209,408（1.03T） | 有 | 2025-11-04 |
| `moonshotai/Kimi-K2.5` | `kimi_k25` | `KimiK25ForConditionalGeneration` | 1,026,874,297,344（1.03T） | 有 | 2026-01-25 |
| `moonshotai/Kimi-K2.6` | `kimi_k25` | `KimiK25ForConditionalGeneration` | 1,026,874,297,344（1.03T） | 有 | 2026-04-17 |
| `moonshotai/Kimi-K2.7-Code` | `kimi_k25` | `KimiK25ForConditionalGeneration` | 1,026,874,297,344（1.03T） | 有 | 2026-06-12 |
| `moonshotai/Kimi-K3` | `kimi_k3` | `KimiK3ForConditionalGeneration` | 2,780,984,527,968（2.78T） | manifest | 2026-07-27 |
| `zai-org/GLM-4.7` | `glm4_moe` | `Glm4MoeForCausalLM` | 356,785,884,416（356.8B） | 有 | 2025-12-22 |
| `zai-org/GLM-5` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-02-11 |
| `zai-org/GLM-5.1` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-04-03 |
| `zai-org/GLM-5.2` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-06-16 |
| `zai-org/GLM-5.2-FP8` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-06-16 |
| `zai-org/GLM-5.3` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-08-25 |
| `zai-org/GLM-5.3-BF16` | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | 753,864,109,440（753.9B） | 有 | 2026-08-25 |
| `zai-org/GLM-5.3-Flash` | `glm5_next` | `Glm5NextForConditionalGeneration` | 321,300,400,436（321.3B） | manifest | 2026-08-25 |
| `zai-org/GLM-5.3-Flash-BF16` | `glm5_next` | `Glm5NextForConditionalGeneration` | 321,300,400,436（321.3B） | 有 | 2026-08-25 |

## 按 architectures[0] 汇总（生成物）

- `Qwen3_5ForConditionalGeneration`：15 个
- `Qwen3_5MoeForConditionalGeneration`：12 个
- `DeepseekV3ForCausalLM`：6 个
- `GlmMoeDsaForCausalLM`：6 个
- `DeepseekV4ForCausalLM`：5 个
- `KimiK25ForConditionalGeneration`：3 个
- `Glm5NextForConditionalGeneration`：2 个
- `MiniMaxM3SparseForConditionalGeneration`：2 个
- `Qwen3_5MoeForCausalLM`：2 个
- `Qwen4ExpForConditionalGeneration`：2 个
- `DeepseekV32ForCausalLM`：1 个
- `Glm4MoeForCausalLM`：1 个
- `KimiK3ForConditionalGeneration`：1 个
- `MiniMaxM2ForCausalLM`：1 个

证据库 manifest（evidence-manifest.json）覆盖：**5 / 59**

<!-- END GENERATED: models -->
