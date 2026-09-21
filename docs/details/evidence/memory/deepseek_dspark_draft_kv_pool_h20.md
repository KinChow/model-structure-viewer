# DSpark 草稿 KV 池 —— H20 在机对账 MSV draftKvBytesPerToken（R2/R3 draft 侧，2026-09-21 现跑）

> **2026-09-21 更正**：下文“草稿复用目标池 SWA / 草稿显存只有权重”的推断不成立。
> 本轮逐张量取证确认 target/draft SWA storage 指针不同，draft 独占
> `34,594,560 B/rank`；`c4_size=0` 只证明没有独立压缩 KV，不证明没有独立 SWA。
> 原始日志事实保留，当前归属结论见
> [framework runtime validation](framework_runtime_validation_20260921.md)。

> 复现（2026-09-21，本人现跑非复用旧服务；GPU 从 0 起、跑后清零）：H20 `10.98.95.16`
> （8×H20-3e SM90）容器 `dsv41_zzj_deploy`（`lmsysorg/sglang:dev-dsv41`，`deep_ep 2.1.0`、torch 2.13+cu130、CUDA 13.0）。
> V4.1：`/ssd1/models/DeepSeek-V4.1-Flash-Attn-W8A8-MoE-W4A8-INT8-Dynamic`（int8-dynamic 代理 ckpt、`kv_cache_dtype=fp8_e4m3`）。
> 起服务 `--attention-backend dsv4 --speculative-algorithm DSPARK --speculative-dspark-block-size 5`
> `--speculative-num-draft-tokens 6 --speculative-num-steps 1 --speculative-eagle-topk 1 --moe-runner-backend marlin`。

## MSV 建模基线（前端，本地可复现）

`node scripts/evidence/memory/deepseek_dspark_draft_kv.mjs`（调 `frontend/src/cost/memory.js: draftKvBytesPerToken`）：

| 模型 | draft KV/token（设计口径） | 草稿子树叶（dtype×elements） | 按 fp8 KV 重算 |
|---|---|---|---|
| DeepSeek-V4-Flash | 512 B | 1 层 `F8_E4M3`(1B)×512 | 512 B（设计已是 fp8） |
| DeepSeek-V4.1-Flash | 768 B | 3 层 `F4`(0.5B)×512 = 256×3 | 1536 B（fp4→fp8 翻倍，idx=0） |
| DeepSeek-V4-Pro | 512 B | 1 层 `F8_E4M3`(1B)×512 | 512 B |

口径：MSV `draftKvBytesPerToken` 把 MTP/DSpark 草稿子树建模为独立、随上下文线性增长的每 token 常驻 KV 池
（与全量 EAGLE/MTP 草稿同口径）。

## 真机运行时（V4.1 DSpark，fresh serve zzj_fresh_v41_fp8.log）

| 量 | 值 | 来源行 |
|---|---|---|
| 目标 `bytes_per_full_token` | 1670.75 B（fp8 KV） | `DSV4 memory calculation`（TP4 复现 TP8 旧值 → 与 TP 无关） |
| 目标 KV 池 | `swa=19456 c4_size=1544768 c128_size=48274 c4_state=2432 c128_state=512` | `Initialize DeepSeekV4TokenToKVPool`（第 1 次） |
| 草稿 KV 池 | `swa=19456 c4_size=0 c4_logical_size=0 c128_size=0 c4_state=0 c128_state=0` | `Initialize DeepSeekV4TokenToKVPool`（第 2 次） |
| DSpark 草稿 runner | `markov_head=DSparkV4MarkovHead, gamma=5, verify_num_draft_tokens=6, query_token_num=5` | `Initialized DSpark draft runner` |
| 草稿权重 | 1.94 GB/卡（`DeepseekV4ForCausalLMDSpark`, compressed-tensors） | `Load weight end` |
| target verify CUDA graph | `num_tokens_per_req=6` | Capture 行 |
| draft verify CUDA graph | `num_tokens_per_req=5, mem usage≈0.00 GB` | Capture 行 |
| `/generate` | "The capital of France is Paris."（正确） | 端到端 |

## 真机运行时（V4-Flash-0731 DSpark，fresh serve zzj_fresh_v4flash0731_fp8_b.log，TP8/EP8）

> V4-Flash 需 **0731 版本**才 bundle DSpark（`/ssd2/models/DeepSeek-V4-Flash-0731-fp8`，config 含 `dspark`、
> `num_nextn_predict_layers:1`、index 含 `mtp.0/1/2.*`）。`DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic` 无 DSpark 头。

| 量 | 值 | 来源行 |
|---|---|---|
| 目标 `bytes_per_full_token` | 3939.79 B（fp8 KV） | `DSV4 memory calculation`（full_token=1290240） |
| 目标 KV 池 | `swa=707840 c4_size=322560 c128_size=10080 c4_state=88480 c128_state=65792` | `Initialize DeepSeekV4TokenToKVPool`（第 1 次） |
| 草稿 KV 池 | `swa=707840 c4_size=0 c4_logical_size=0 c128_size=0 c4_state=0 c128_state=0` | `Initialize DeepSeekV4TokenToKVPool`（第 2 次） |
| DSpark 草稿 runner | `markov_head=DSparkV4MarkovHead, gamma=5, verify_num_draft_tokens=6, query_token_num=5` | `Initialized DSpark draft runner` |
| `/generate` | "...Paris. The capital of Spain is Madrid. The capital of Italy is Rome."（正确） | 端到端 |

> 注：marlin MoE runner 对 0731（`Fp8MoEMethod`）的 DSpark 路径会 `AttributeError: 'NoneType' ... runner_backend`；
> 去掉 `--moe-runner-backend marlin`（用 auto）即跑通。V4.1（W4A8）需 marlin，V4-Flash-0731（fp8 MoE）用默认。

## 对账结论（据实登记）

- 关键发现：DSpark 草稿 KV 池 `c4_size=0`（无独立压缩 KV）。真机第二个 `DeepSeekV4TokenToKVPool`（草稿池）
  的所有压缩/满 token 分量为 0，草稿只复用目标池的 SWA（同一 `swa=19456`）。DSpark 是 Markov 头 + block 验证
  （`DSparkV4MarkovHead`），其显存成本 = 草稿权重（1.94 GB/卡）+ verify CUDA graph（≈0 GB），没有随上下文增长的
  独立每 token 草稿 KV 池。
- **V4-Flash-0731 与 V4.1 同结论**：两模型的草稿池均 `c4_size=0` → DSpark 家族**统一**把草稿并入目标池、
  草稿层不额外常驻压缩 KV（不随 V4-Flash/V4.1、TP4/TP8 变）。
- 因此 MSV 的 `draftKvBytesPerToken`（V4.1 768 B/token、V4-Flash 512 B/token）所建模的"独立、随上下文增长的草稿
  常驻 KV 池"，在 DSpark 运行时不对应一个独立池——这是建模口径与 DSpark 运行时机制的 regime 差异，如实登记：
  MSV 对 DSpark 家族的草稿 KV 属于上界式建模（相当于把草稿层当成拥有自有 KV 的全量草稿）；DSpark 实际把草稿并入
  目标池、草稿层不额外常驻压缩 KV。对全量 EAGLE/MTP（草稿自带解码层 KV）MSV 的独立池口径才成立。
- 目标侧 `bytes_per_full_token=1670.75`(fp8) 与 MSV 主干 V4.1 KV 890(fp4) 的 1.877× ≈ fp4→fp8 关系不变
  （见 `deepseek_v41_dspark_runtime_h20.md`）。

## 边界（仍未闭合）

- ~~V4-Flash 草稿池 blocked-on-ckpt~~ **已解**：用 `DeepSeek-V4-Flash-0731-fp8`（bundle DSpark）现跑闭合（见上表）。
  非 0731 的 `DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic` 无 DSpark 头（`requires --speculative-draft-model-path`）。
- fp4 精确逐字节（768/512 命中 0 容差）hard-blocked：`--enable-deepseek-v4-fp4-indexer requires SM100, SM120,`
  `or gfx95 GPUs`；H20=SM90(Hopper) 被直接拒绝 → 原生 fp4 KV 口径需 Blackwell(SM100)，H20 不可得（见 R1/R2）。
  故草稿池的精确 fp4 数值也随之留 Blackwell。
- 用的仍是 int8-dynamic 代理 ckpt + KV=fp8，非 MSV 内置原始 fp8/fp4；精确对账留 Blackwell + 原始 ckpt。

全程临时日志 `/ssd1/models/zzj_fresh_v41_fp8.log`（现跑取证），GPU 跑后清零（8×0 MiB 已核）。
