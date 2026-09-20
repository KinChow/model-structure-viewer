# 内置模型全量验证矩阵（P1 追踪骨架）

> 计划 v3 的 P1 活动追踪表。口径：静态 60/60（`verify:models`）覆盖全模型；运行时按 **11 builder 各取代表减层件**
> × 每维（结构/显存/成本/并行）× 双框架（SGLang / vLLM）填格。同 builder 的 config 变体由静态 60/60 + 族内抽检覆盖。
> 状态：✅ 已对账 / 🟡 部分或口径-代表 / ⬜ 待补 / ⛔ 硬件边界。**vLLM 列本轮全新（此前 0 次真机）。**
> 详细结论见 `evidence/README.md` 覆盖矩阵、`structure/builder_coverage.md`、各维 `evidence/**`。

## 逐 builder × 维度 × 框架

| builder | 模型数 | 减层件 | 结构 | 显存·SGLang | 显存·vLLM | 成本·SGLang | 成本·vLLM | 并行·SGLang | 并行·vLLM | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| assembleQwen3_5 | 29 | qwen3_5_reduced + qwen3_5_moe_tiny | ✅ | ✅ | ⬜ | 🟡 | ⬜ | ✅ | ⬜ | dense+MoE 双分支；GDN ssm dtype 见 Bug2 |
| assembleDeepseekV3 | 9 | deepseek_v3_tiny | ✅ | 🟡 | ⬜ | ✅ | ⬜ | 🟡 | ⬜ | MLA+MoE（V3.1/R1/K2） |
| assembleDeepseekV32 | 7 | deepseek_v32_tiny | ✅ | 🟡(cache) | ⬜ | 🟡 | ⬜ | 🟡 | ⬜ | DSA index；稀疏前向 H20(P4) |
| assembleDeepseekV4 | 5 | — | ✅ | 🟡(静态) | ⬜ | 🟡 | ⬜ | ⛔ | ⬜ | fp8→H20 前向(P4) |
| assembleMiniMaxM3 | 2 | minimax_m3 减层 | ✅ | 🟡(cache) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | 块稀疏；vLLM 支持存疑 |
| assembleQwen4Exp | 2 | qwen4_exp 减层 | ✅ | ✅ | ⬜ | 🟡 | ⬜ | ⬜ | ⬜ | qsa+index |
| assembleGlm5Next | 2 | glm5_next_reduced | ✅ | ✅(H20 稀疏前向) | ⬜ | 🟡 | ⬜ | ⬜ | ⬜ | KDA+MLA+DSA；Bug1/Bug2 源 |
| assembleMiniMaxM2 | 1 | minimax_m2_tiny | ✅ | ✅ | ⬜ | 🟡 | ⬜ | 🟡 | ⬜ | GQA+MoE |
| assembleDeepseekV41 | 1 | — | ✅ | ✅(静态 890) | ⬜ | 🟡(H20 fp8 前向) | ⬜ | ✅(H20 tp8/ep8) | ⬜ | **H20 fp8 前向+DSpark 跑通**；engram/fp4-KV 数值留后续 |
| assembleKimiK3 | 1 | kimi_k3/kimi_linear_tiny | ✅ | 🟡(cache) | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | KDA+MLA；vLLM 支持存疑 |
| assembleGlm4Moe | 1 | glm4_moe_tiny | ✅ | ✅ | ⬜ | 🟡 | ⬜ | 🟡 | ⬜ | GQA+MoE(shared 融合) |

## 收官判据

- 每格 → ✅ 或显式 ⛔（硬件边界，如 fp4=Blackwell）+ 证据文件；除搁置多机外无 ⬜。
- 每 builder ≥1 减层真机 / 减层 cache 口径证据（可用框架）。
- 关键数命中容差：Qwen3-0.6B KV 112KiB/tok、glm5_next KV 2312 B/tok、GLM-5.3-Flash 线性 140.78 MiB/seq、V4.1 KV 890 B/tok。
- vLLM 列：逐 builder（bf16 装得下者）`vllm serve` 对每层有效宽度 / KV / 权重÷tp / EP，与 MSV + SGLang 三方对拍，喂 P3 framework profile。

## 本轮进展（2026-09-20 双框架在机补验）

- **vLLM 首次真机（A100，Qwen3-0.6B GQA 代表）**：KV/token 114,712 B ≈ MSV 114,688（0.02%）、每卡权重 ÷tp、KV 池 ×tp——**vLLM == SGLang == MSV**（GQA KV 属 [C] 干净，跨框架一致）。证据 `parallelism/vllm_width_tp.md`。→ 各 GQA-bearing builder 的 vLLM 显存/并行经此代表覆盖；vLLM MoE `EP=TP×DP`（VL3）仍待。
- **glm5_next DSA 稀疏前向（H20）复跑再确认**：KV/token **2312** == 修复后 MSV（Bug1 已落地 commit 3e6aa3b）。证据 `runtime_profiles/sglang_glm5next.md`。
- **qwen3_5 GDN（H20）**：GQA KV/token 4096 == MSV（0.0%）；线性 state 精确复测（512 slots）conv 293,601 vs MSV 294,912、ssm 12,603,883 vs MSV 12,582,912、总量 12.30 vs 12.28 MiB —— **逐分量 <0.5%，Bug2（ssm fp32）逐字节精确验证**（初测 ~1.25× 系 4-slot 舍入伪差，已排除，**无 GDN state-shape bug**）。证据 `runtime_profiles/sglang_qwen35_gdn_h20.md`。
- **V4.1-Flash + DSpark（H20，最后一个 fp8-only builder）**：现跑 tp8/ep8 dsv4 backend/kv fp8_e4m3/W4A8-marlin MoE + DSPARK（num_draft_tokens 6），`server fired up` + `/generate` 正确出词（"...Paris."）→ **HB1 fp8 前向 + HB6 DSpark 投机 H20 运行时通过**；EP 下 shared-expert 不融合（与 vLLM 一致，[B] 收敛）。**11 个 builder 全部具备 A100/H20 运行时证据**。证据 `runtime_profiles/sglang_dsv41_dspark_h20.md`。**边界**：W8A8/W4A8 build 走 fp8 index，fp4 设计 KV(890) 数值对拍留 fp4-build。
- **vLLM MoE 专家并行（A100，VL3）**：DeepSeek-V2-Lite tp2 —— EP-on 每 rank 32/64 完整专家（`E=32,N=1408`，ep=tp×dp、expert-TP=1）、EP-off 全 64 专家 intermediate÷2（`E=64,N=704`）——与 MSV `expertShardDivisor`（EP `÷ep 整专家` / TP `÷tp intermediate` / shared `÷tp` 独立）**逐点一致**。**[B] MoE EP×moe_tp/shared-expert 非前端 bug**（MSV 已按 plan 参数正确表达 vLLM/SGLang/TRT 三框架）；framework profile 属可选 UX 预设、非正确性修复。证据 `parallelism/vllm_moe_ep.md`。

## 开项登记（架构级）

- ~~GDN recurrent state-shape 残差~~ **已关闭（2026-09-20）**：512-slot 精确复测显示 `linearStateResidentDecl` 的 GDN conv(bf16)+ssm(fp32) 与真机逐分量 <0.5%，初测 ~1.25× 为 4-slot 粗舍入伪差，非前端 bug。
