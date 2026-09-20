# Part 2c · 混合 ETP（moe_tp>1）与 DeepEP all-to-all（8×A100）

> 补并行验证最后的"未验证"项。减层随机 qwen3_moe（8 experts/topk2/4层，`scripts/evidence/parallelism/build_ep_ckpt.py`，零下载）。
> 验证前端 `cost/sharding.js: expertShardDivisor`（ep×moe_tp）与 `cost/comm.js: expertAllToAllBytes`（moe_tp 无关）。

## ETP（expert tensor parallel, moe_tp>1）—— 已验证

SGLang `--tp-size 4 --ep-size 2`：4 个 TP rank 分成 2 个 EP 组（`TP0/TP1→EP0`、`TP2/TP3→EP1`），
即每组内 2 路张量并行 → **moe_tp = tp/ep = 2**。MoE runner 打印每 rank 专家形状：

| 计划 | 前端 `expertShardDivisor` | 每 rank E（=experts/ep） | 每 rank N（=moe_inter/moe_tp） | SGLang 实测 | 判定 |
|---|---|---|---|---|---|
| ep=2（moe_tp=1，旧证据） | divisor=2, moeTp=1 | 4 | 768 | `E=4,N=768` | ✓ |
| ep=2 + moe_tp=2（tp4/ep2） | **divisor=4**, moeTp=2 | 4 | **384** | **`E=4,N=384`** | ✓ 逐点一致 |

- **N 从 768 → 384 恰好减半**，正是每个专家的 intermediate 被 moe_tp=2 张量并行切分；E 仍为 4（每组持 experts/ep=4 个完整专家）。
- 前端 `expertShardDivisor({ep:2,moeTp:2})` = epSize×moeTp = **4**（=TRT-LLM 混合 ETP 语义：每卡 E/ep 完整专家、专家权重再 ÷moe_tp），
  与真机 `E=4,N=384` 逐点吻合。**补齐了此前"混合 ETP（moe_tp>1）未验证"。** 证据由 `scripts/evidence/parallelism/build_ep_ckpt.py` 构建 ckpt + SGLang `--tp-size 4 --ep-size 2` 重跑重生。

## all-to-all 字节口径（moe_tp 无关）

- 前端 `expertAllToAllBytes = B·T·expertsPerToken·H·b` **不含 moe_tp 因子**：ETP 切的是专家**权重/计算**，
  不改变 token dispatch 的**载荷条数**（每 token 仍发往其 topk 专家所在的 EP 组）。Part 2b（`scripts/evidence/parallelism/alltoall_bench.py`）
  已在 ep=2/4/8 字节直测该载荷（16.78MB，比值 1.000，与 ep 无关）——**该结论对 moe_tp 同样成立**（载荷口径与传输后端解耦）。

## DeepEP —— 框架层生效，kernel ABI 不匹配（本环境边界）

SGLang `--moe-a2a-backend deepep --deepep-mode normal`：
- **后端在框架层生效**：`moe_a2a_backend='deepep'`、`deepep_mode='normal'`，走 DeepEP 专属路径（cuda graph 因 normal 模式禁用、
  "Only use 20 SMs for DeepEP communication"），并进入 `qwen3_moe.py:362 forward_deepep → dispatcher.dispatch(...)`。
- **但 DeepEP 编译 kernel 与本机 CUDA 13.0/驱动 ABI 不匹配**：`RuntimeError: CUDA error .../deep-ep-build/csrc/kernels/layout.cu:128
  'named symbol not found'` → 无法完成一次前向，故 **DeepEP 字节级直测在本环境不可得**。这是**环境/构建**限制（deep_ep 预编译 kernel
  vs CUDA 13 运行时），非前端/建模问题。证据由 SGLang `--moe-a2a-backend deepep --deepep-mode normal` 重跑重生。
- 口径说明：DeepEP 只是同一 token 载荷的**优化传输**，字节口径与传输后端无关，已由 Part 2b 的 NCCL all_to_all_single 直测覆盖
  （比值 1.000）。**DeepEP 专属字节级直测留换环境**（需可运行的 DeepEP 构建；SGLang 注明 DeepEP 面向 DeepSeek-V3/R1 EP≥2 场景）。

## 判定与边界（诚实）

- **混合 ETP（moe_tp>1）已验证**：真机 `E=4,N=384` == 前端 expertShardDivisor(ep×moe_tp=4)，该缺口收口。
- all-to-all 字节口径与 moe_tp/传输后端解耦，已由 Part 2b 字节直测。
- **DeepEP**：框架层生效并进入 dispatch，但本机 kernel ABI 不匹配无法前向；字节级直测留换环境。
- 仍未验证（需换环境）：**真·多机**（跨节点 NCCL / PD 分离）、可运行 DeepEP 构建下的字节直测。
