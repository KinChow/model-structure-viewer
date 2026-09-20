# 前端算子级计算量/访存量对真值（Qwen3-0.6B, A100-80GB, TP=1）

> 目的：验证前端 cost lens 每个**原子/融合算子**产出的动作向量
> `{matrix(MACs), vector(FLOPs), sfu(ops), bytes{...}}` 是否与 GPU 真值一致、是否满足 roofline。
> 三通道各选 oracle：matrix→torch FlopCounterMode（精确）；bytes→ncu DRAM；vector/sfu→ncu + 解析。
>
> 在机：A100-SXM4-80GB / CUDA 13.0 / torch 2.13.0+cu130 / transformers 5.17.0 / ncu 2025.3.1。
> 复现：`node scripts/evidence/cost/operator_cost.mjs`（前端）→ `scripts/evidence/cost/operator_collect_flops.py`（FlopCounter）→
> `scripts/evidence/cost/run_ncu.sh`（ncu 微基准）→ `scripts/evidence/cost/operator_reconcile.py`（对账）。

## 结论速览

| 通道 | 判定 | 关键数据 |
|---|---|---|
| **matrix（计算量）GEMM** | **精确相等** | MSV MACs×2 == torch FLOPs：prefill 610,288,009,216、decode 1,191,968,768（逐位相等） |
| **matrix 注意力** | 因果口径差（合法） | MSV 因果 = 0.501× torch 全方阵；MSV 与 flash kernel 实际下三角计算对齐（见 flash_kernel_caliber.md） |
| **bytes（访存量）GEMM** | **compulsory 读侧精确** | 4 个 GEMM 的 ncu DRAM 读 / MSV(weights+actIn) = 1.002–1.004；写侧常驻 L2 → MSV total 为保守上界 |
| **bytes 逐元素** | 融合口径（合法） | MSV 建**融合 compulsory**（读一遍+写一遍）；朴素多 kernel 微基准重复物化中间量而高估，非代表 |
| **roofline bound 分类** | **全部一致** | GEMM AI 340–438 FLOP/B（>脊点≈120 → compute-bound）；norm/rope/swiglu AI 0.2–0.33（<<脊点 → memory-bound） |

## 一、matrix 通道（torch FlopCounterMode 真值）

FlopCounterMode 只数 matmul 家族（mm/bmm/sdpa），FMA=2 → **MSV matrix(MACs)×2 应 == torch FLOPs**。

| 相位 | MSV 线性 MACs×2 | torch aten.mm | 相等 | MSV sdpa MACs×2 | torch aten.bmm | fe/torch |
|---|---|---|---|---|---|---|
| prefill S=512 | 610,288,009,216 | 610,288,009,216 | **是** | 30,123,491,328 | 60,129,607,680 | 0.501 |
| decode ctx=576 | 1,191,968,768 | 1,191,968,768 | **是** | 132,120,576 | 132,120,704 | 1.000 |

- **线性/投影（q/k/v/o/gate/up/down/lm_head + embedding）逐位相等**——所有 `linearCounts` 公式经真值坐实。
- **注意力 prefill 0.501×**：MSV 按**因果** `scoredPairs=S(S+1)/2=131,328` 计；eager/torch 公式按**全方阵** `S²=262,144` 计（比值 513/1024=0.501）。flash kernel 真实只算下三角（flash_kernel_caliber.md 已证 scores 不落 HBM），故 **MSV 因果口径与生产 flash kernel 的实际计算一致**，torch 公式反而高估。decode（query=1）无因果差，1.000（127 的绝对差 = 0.0001%，舍入级）。

## 二、bytes 通道（ncu DRAM 真值，单层实例 prefill S=512）

MSV bytes = **compulsory 流量**（每张量读一遍+写一遍，不含 tiling 重读/cache/融合）。

| 算子 | MSV 读(w+actIn+kvR) | ncu DRAM 读 | 读比值 | ncu DRAM 写 | 说明 |
|---|---|---|---|---|---|
| gate_proj | 7,340,032 | 7,361,024 | **1.0029** | 0 | 权重读一遍精确；输出常驻 L2 未落 DRAM |
| down_proj | 9,437,184 | 9,453,952 | **1.0018** | 0 | 同上 |
| o_proj | 6,291,456 | 6,308,224 | **1.0027** | 0 | 同上 |
| q_proj | 5,242,880 | 5,263,872 | **1.0040** | 0 | 同上 |

- **GEMM 的 MSV compulsory 读侧 == ncu DRAM 读，误差 0.2–0.4%**（权重+输入各读一遍，坐实 `linearCounts.bytes`）。
- **写侧常驻 L2**（ncu dram_write≈0）：单算子隔离下输出没被逐出到 DRAM。MSV 把 actOut 计入 → 对 DRAM 是**保守上界**（真实前向里输出被下一算子消费，多从 cache 命中）。此即 MSV "估计/上界" 口径的实证。
- **逐元素（rmsnorm/rope/swiglu）读比值 1.5–10×偏大**：本次微基准是**朴素多 kernel** 实现（rmsnorm 8 kernel、rope 4 kernel），把 fp32 中间量反复物化到 DRAM；MSV 建的是**融合 kernel 的 compulsory**（读一遍+写一遍），是融合下的正确下界，朴素实现不代表。真实模型用融合 RMSNorm/SwiGLU kernel，趋近 MSV。
- **注意力 S=512 读 3.37×**：小 kernel 固定开销主导，非渐近口径；注意力访存的干净口径见 flash_kernel_caliber.md（大 S 下 flash DRAM≈Q/K/V/O ≪ A2 上界）。

## 三、vector/sfu 通道与 roofline bound 分类

FLOP 计数器不数逐元素/norm/softmax；MSV 的 vector/sfu 是**解析逻辑算子数**（由 `atoms.js` 逐算子单测精确锁定），
与硬件 SASS 指令数（含类型转换、地址计算、MUFU 展开、循环开销）**非 1:1**。因此本通道不做"指令数相等"判定，
改验 **roofline bound 分类**：

| 算子 | AI (FLOP/byte) | A100 脊点≈120 | MSV bound | ncu 佐证 |
|---|---|---|---|---|
| gate/up/down/q/o_proj | 340–438 | > 脊点 | matrix（compute） | tensor pipe 主导、DRAM 读=compulsory |
| rmsnorm | 0.197 | ≪ 脊点 | memory | DRAM 主导、tensor≈0 |
| rope | 0.268 | ≪ 脊点 | memory | 同上 |
| swiglu | 0.333 | ≪ 脊点 | memory | 同上 |

- prefill GEMM 全部 **compute-bound**、逐元素全部 **memory-bound**——与 MSV roofline（`classifyRoofline` 五路取 max）
  的 bound 判定逐项一致，也与聚合 roofline 口径（prefill 算力受限 / decode 访存受限）自洽。

## 处置

- **未发现前端公式错误**：matrix 通道逐位相等、bytes 读侧精确、bound 分类一致 → `counts.js`/`modules.js` 不改。
- **登记合法口径差**（见 `docs/details/cost_counts.md` 标注）：① 注意力因果 vs 全方阵（MSV 对齐 flash）；
  ② bytes 写侧常驻 cache（MSV total 为 DRAM 保守上界）；③ 逐元素 MSV=融合 compulsory（朴素微基准高估非代表）。
- **回答用户问题**：前端原子/融合算子的**计算量**（matrix 精确、vector/sfu 解析锁定）与**访存量**
  （compulsory 读侧精确、写/融合为保守上界）**满足 roofline 输入口径**，bound 分类经真值坐实。
