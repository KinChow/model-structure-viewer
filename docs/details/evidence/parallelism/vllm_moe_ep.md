# vLLM MoE 专家并行（EP）vs MSV expertShardDivisor —— 真机对账（A100, 2026-09-20）

> 复现：容器 `vllm-0920`（A100，vLLM `0.28.1rc1.dev278`），模型 `/ssd2/models/DeepSeek/DeepSeek-V2-Lite`
> （`DeepseekV2ForCausalLM`，64 routed experts、moe_intermediate 1408、2 shared、bf16），tp2 ±`--enable-expert-parallel`。
> 补 vLLM 侧 MoE 并行真机（此前 vLLM 仅验 GQA 稠密）。

## 真机（vLLM FusedMoE 配置 + expert placement）

| 模式 | worker | EP 状态 | FusedMoE | 每 rank 专家 | 每卡权重 |
|---|---|---|---|---|---|
| **EP-on** `tp2 --enable-expert-parallel` | `Worker_TP0_EP0` | `[EP Rank 0/2] Local/global experts: 32/64`（linear 0→0..31→31） | **E=32, N=1408** | 14.73 GiB |
| **EP-off** 纯 `tp2` | `Worker_TP0` | 无 EP | **E=64, N=704** | 14.73 GiB |

- EP-on：每 rank 持 **32 = 64/ep 个完整专家**、intermediate 全长 1408（不切）→ `ep_size = tp×dp = 2×1 = 2`、expert-TP=1。
- EP-off：全 64 专家在每卡、每专家 intermediate 1408→704（expert-TP=2）。
- 两者每卡权重相同（14.73 GiB）：EP 按专家数减半、TP 按 intermediate 减半，总 MoE 参数都减半。

## 对账 MSV（`cost/sharding.js:expertShardDivisor`）

- **vLLM EP-on** ⇔ plan `{ep:2, moeTp:1}` → `epOn=true, divisor = epSize×moeTp = 2×1 = 2`、`setDegree=epSize=2`
  → 每 rank 64/2 = **32 完整专家、moe_tp=1 不切 intermediate** —— 与 vLLM `E=32,N=1408` **逐点一致**。
- **vLLM EP-off（纯 TP）** ⇔ plan `{ep:1, tp:2}` → `epOn=false, moeTp 缺省=tp=2, divisor = moeTp×dp = 2`
  → 全专家、每专家 intermediate ÷2 —— 与 vLLM `E=64,N=704` **逐点一致**。
- **SGLang EP×moe_tp**（混合 ETP）由显式 `moe_tp` 表达（divisor = ep×moe_tp），已由 A100 SGLang `ep48.md`/`etp_deepep.md` 验证。
- **shared expert**：MSV 复用 dense MLP 声明（class `tp`，÷tp、跨 EP rank 复制），**不并入 routed ep 除数** —— 与真机
  「EP 下 shared experts fusion disabled、shared 专家保持独立 TP 切」一致（`sglang_dsv41_dspark_h20.md` 亦见此行）。

## 结论

- **MSV 的 MoE 并行建模（EP 整专家÷ep / TP intermediate÷tp / 混合 ETP÷(ep·moe_tp) / shared 独立÷tp）与 vLLM 真机逐点一致**，
  且能同时表达 vLLM(ep=tp×dp, moe_tp=1)、SGLang(ep×moe_tp)、TRT-LLM(显式 moe_tp) 三框架 —— 经 plan 参数区分，无需分叉代码。
- 即 frontend_problem_inventory 的 **[B] MoE EP×moe_tp / shared-expert 分叉不是前端 bug**：MSV 已按 plan 参数正确建模，两框架差异可表达。
  「framework profile」（neutral/vLLM/SGLang 预设选择器）是在既有正确 plan 参数上加的**可选 UX 预设**，非正确性修复 → 按需增强、非本目标必需。
- 未覆盖：vLLM all-to-all 字节直测（SGLang 侧已 `etp_deepep.md`/`alltoall_bench.py` 验）、DP-attention（attnMode=dp）跨框架。

## 附：同轮 vLLM MLA KV/token 对账（内存维，免费副产）

V2-Lite 是 MLA（`kv_lora_rank=512` + `qk_rope=64`，27 层全 MLA）。vLLM EP-on KV 池 845,152 tok / 24.48 GiB →
**KV/token = 31,101 B**；MSV/设计 = 27 层 × (512+64) × 2B(bf16) = **31,104 B**（0.008%，24.48 两位小数舍入）。
与 SGLang V2-Lite（`runtime_profiles/sglang_v2lite.md`：MLA latent 31,104 B/token）**逐值一致** →
**vLLM MLA latent KV == SGLang == MSV**（GQA 112KiB 之外，MLA 家族的 KV 内存口径亦跨框架一致）。

## 追加：C3b `resolveFrameworkPlan` 真机确认（A100, 2026-09-21）

复现：A100 `10.55.87.81` 容器 `vllm-0920`，`vllm serve /ssd2/models/DeepSeek/DeepSeek-V2-Lite --tensor-parallel-size 2
--enable-expert-parallel --gpu-memory-utilization 0.4 --trust-remote-code`。日志 `/ssd2/models/_reduced/vllm_v2lite_c3b.log`。

- 真机：`[EP Rank 0/2] ... Local/global number of experts: **32/64**`（每 rank 32 完整专家 = 64/ep(2)、moe_tp=1、intermediate 不切）。
- MSV C3b：`resolveFrameworkPlan({tp:2,dp:1}, "vllm") = {ep:2, moeTp:1}` → `expertShardDivisor` → `epSize=2, moeTp=1, divisor=2, setDegree=2`
  → 每 rank 64/2 = **32 完整专家、N=1408 不切** —— **与 vLLM 真机 32/64 逐点一致**。
- 即框架预设 `vllm` 自动落到 `ep=tp×dp、moeTp=1`，产出的 plan 与 vLLM 真机 EP-on 专家分片口径一致（C3b 端到端坐实）。跑完已 `pkill` vLLM、GPU 复位。
