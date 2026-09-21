# DeepEP 在 H20 前向 + shared-expert 是否折叠进 all-to-all（R4，2026-09-21 现跑）

> 复现（H20 `10.98.95.16` 8×H20-3e SM90，容器 `dsv41_zzj_deploy`，`deep_ep 2.1.0`/CUDA13/torch2.13）：
> 减层 dummy DeepSeek-V3（`architectures=[DeepseekV3ForCausalLM]`、`n_routed_experts=256`、`n_shared_experts=1`、
> `num_experts_per_tok=8`、2 层、`--load-format dummy --skip-tokenizer-init`），SGLang `--tp-size 2 --ep-size 2 --moe-a2a-backend deepep`
> （`--deepep-mode normal` 与 `low_latency` 各一次）。日志 `/ssd1/models/zzj_dsv3_deepep*.log`。

## 结论 1：DeepEP dispatch 在 H20 跑通（A100 ABI 障碍解除）

- normal 模式：前向进入 `models/deepseek_v2.py:1499 forward_deepep` → `dispatcher.dispatch(...)` → `run_moe_core(dispatch_output)`
  —— DeepEP dispatch 实际执行；随后停在 `NotImplementedError: Unquantized DeepEP MoE currently supports low_latency mode only`
  （未量化 dummy 权重限制，非通信问题）。
- low_latency 模式：构造 + CUDA graph capture 进行；停在 `RuntimeError: q_v is only supported for hdim_v >= 256`
  （减层 dummy 的 MLA `v_head_dim=64` 违反 decode kernel 维度约束，非 DeepEP 问题）。
- 对照 A100：预编 kernel `layout.cu:128 'named symbol not found'`、源码重编卡 NVSHMEM/CUDA-13（`deepep_source_build_attempt.md`、`etp_deepep.md`）。
  H20（Hopper=DeepEP 目标架构）上 `deep_ep 2.1.0`+`Buffer` import 通过、dispatch 执行 → ABI/构建障碍解除。
  未取完整出 token：减层 dummy 的 MLA 维度/量化与真机 kernel 约束不合（需真·DeepSeek-V3 维度或量化权重），非通信路径问题。

## 结论 2（关键）：DeepEP 默认不把 shared expert 折进 all-to-all，前端 C3c 高估

运行时日志逐次打印：`DeepEP: fusion off by default (use --enforce-shared-experts-fusion to enable). Shared experts fusion optimization is disabled.`

源码 `models/deepseek_v2.py: shared_experts_fusion_disable_reason`（构造期一次性决定 `num_fused_shared_experts`）：

- `is_deepep_class_backend()` → DeepEP 默认关（除非 `--enforce-shared-experts-fusion`）。
- `moe_ep_size>1` 且 NV → 关（"under expert parallelism" 仅 AMD gfx942+ 可开）。
- 仅 `DeepseekV3ForCausalLM` + `n_routed_experts∈{256,384}` + `n_shared_experts==1` + capability≥80 + 非 EP + 非 DeepEP 才 `num_fused_shared_experts=n_shared`。

`num_fused_shared_experts=0` 时 shared expert 是独立本地 MLP、不进 a2a dispatch；dispatch topk 仅 = `num_experts_per_tok`。

因此：在 NV H20 的 DeepEP / EP>1 常规场景，shared expert 不折进 all-to-all → 真机 dispatch 字节 = `B·T·topk·H·b`（routed only）。
而前端 `cost/comm.js:44-45` 对 SGLang 无条件 `+sharedExperts`（`sharedFused = frameworkProfile==="sglang" && sharedExperts>0`）
→ C3c 在默认 DeepEP/EP 场景高估 all-to-all 字节。仅当显式 `--enforce-shared-experts-fusion`（且 DeepSeek-V3/R1 合规 + capability≥80）时折叠才成立。

## 前端修复（已落地，方案 A：默认关 + 显式 opt-in）

`cost/comm.js` 的 C3c fold 已改：默认**不**折叠（匹配 DeepEP/EP 默认 fusion off），仅当
`enforceSharedExpertsFusion===true`（`options` 或 `plan.enforceSharedExpertsFusion` / `plan.enforce_shared_experts_fusion`
任一，对应 SGLang `--enforce-shared-experts-fusion`）且 `frameworkProfile==="sglang"` 且 `sharedExperts>0` 时才 `+n_shared`。
vLLM/neutral 恒不折叠。`planCommunicationBytes` 增 `enforceSharedExpertsFusion` 入参透传。

- 单测 `cost/__tests__/comm.test.js` 更新：SGLang 默认 = 16（不折叠）、`enforceSharedExpertsFusion:true` = 24（折叠）、
  非 sglang/无 shared 恒不折叠。
- `node --test` 438/438、`verify:models` 60/60、`docs:check` 全绿；**无 golden 变更**（cost_counts 等生成器走 neutral profile，从不触发该 fold）。
- all-to-all 基础字节口径 `B·T·topk·H·b` 仍由 A100 NCCL `alltoall_bench.py` 收口（比值 1.000）。
- 备注：该开关目前经 `plan`/`options` 传入（默认关即正确默认）；如需 UI 可切换，可在 framework 设置里加一个 sglang-only 的 enforce 勾选（未做，非必需）。

全程 GPU 跑后清零（8×0 MiB 已核）；临时件 `/ssd1/models/_dsv3_tiny`、`zzj_dsv3_deepep*.log`。
