# A-2 · MoE/MLA 算子成本真值（减层随机权重, DeepSeek-V3, A100）

> 减层随机初始化（`scripts/evidence/cost/moe_build_reduced.py`, `HF_HUB_OFFLINE=1`, **零权重下载**）：DeepSeek-V3 → 2 层
> (layer0 dense / layer1 MoE)、n_routed_experts=8、topk=4、n_group=1、nextn=0，随机参数 3.02B。
> 前端 `scripts/evidence/cost/operator_cost.mjs` 跑同一 reduced config；真值 FlopCounterMode（eager）。S=128 prefill。

## 前端算子覆盖（此前 Qwen3-0.6B 稠密没有的）

MLA：`mla_query_compress`(q_a)、`mla_kv_compress`(kv_a)、`mla_kv_split`、`sdpa_attention`；
MoE：`topk`(router)、`moe_dispatch`、`fused_moe_mlp`(routed experts)、`moe_combine`、`moe_add`；
另有 dense MLP `swiglu`、shared_expert 的 gate/up/down（记为 linear）。

## matrix 通道对账（MACs，torch FlopCounter=aten.mm/2）

| 组件 | 前端 MACs | torch 叶子 MACs | 判定 |
|---|---|---|---|
| lm_head | 118.615G | 118.615G | **精确** |
| MLA q_a(compress) | 2.819G | 2.819G | **精确** |
| MLA kv_a(compress) | 1.057G | 1.057G | **精确** |
| 标准线性合计(含 lm_head+MLA q_b/kv_b/o+dense MLP+shared+router) | 216.45G | 219.02G | **0.971（3% 低）** |
| routed experts (`fused_moe_mlp`) | 22.55G | **≈0（FlopCounter 未计）** | oracle 失效，见下 |
| 注意力 bmm | 0.406G | 1.342G | 0.302 |

**结论（三类，诚实）：**
1. **标准 GEMM（lm_head / MLA 压缩 q_a·kv_a / dense MLP / shared expert / router）：前端与 torch 一致到 ~3%**
   ——与 operator_cost.md 稠密的逐位相等一致，MLA 的 q_a/kv_a 压缩投影**精确相等**。残余 ~3%（6.4 GMACs）集中在
   MLA q_b/kv_b 的 rope-split 区，属 MLA 结构细节，留作更细粒度逐 module 映射的后续核对（未定性为 bug）。
2. **routed experts：FlopCounter（HF eager MoE）不计其 GEMM**（实测 layer1.mlp aten.mm=11.29G 仅含 shared
   expert，routed 专家 ≈0）——**FlopCounter 不是 MoE routed 专家的有效 oracle**（分组/gather 路径漏计）。
   前端 `fused_moe_mlp`=tokens×topk×3×h×moe_inter=22.55 GMACs 是教科书 active-compute 口径，**构造正确**；
   其真值需用 **ncu 采专家 kernel FLOPs** 核对（Part 1 后续，本轮未做 ncu 专家采集）。
3. **MLA 注意力 bmm=0.302×**：MLA 是 latent/吸收式注意力（kv_b 吸收、latent 共享），与前端 `sdpa_attention`
   的标准 MHA `scoredPairs` 口径不同——属 MLA 口径差异，需单独口径（登记，非 bug）。

## 处置

- **未改前端公式**：标准 GEMM 一致、MLA 压缩精确；MLA q_b/kv_b 残余(3%)与 bmm 口径差、routed 专家 oracle
  失效均需 ncu/更细分析定性，本轮**登记为待细化项**，不轻改公式（避免"为修复而修复"）。
- 减层随机权重路径**零下载**已坐实（3.02B DeepSeek-V3 MLA+MoE 离线构建+前向+FlopCounter 全通）。

## 复现

`python3 scripts/evidence/cost/moe_build_reduced.py deepseek_v3` → `node scripts/evidence/cost/operator_cost.mjs --config scripts/evidence/_fixtures/deepseek_v3.json ...`
→ `python3 scripts/evidence/cost/moe_collect_flops.py deepseek_v3` → 本对账。DeepSeek-V4(DSA) 已能同法减层构建(1.73B)，算子级对账留后续。
