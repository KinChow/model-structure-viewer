# 线 B · TP 并行策略：前端投影 vs SGLang 真机（Qwen3-0.6B, A100-80GB）

> 目的：验证前端并行策略代码（`cost/sharding.js` 权重按 TP 折宽、`cost/parallel.js` `kvBytesPerCard`
> GQA 分片、`cost/comm.js` all-reduce）在真机 TP>1 下与实测一致——此前只在 TP=1（退化恒等）验过。
>
> 在机：A100-SXM4-80GB ×4 / SGLang / Qwen3-0.6B / mem-fraction-static 0.8。
> 前端预测：`node scripts/evidence/parallelism/tp_projection.mjs`（投影 JSON 由该脚本重生）；
> 真机：SGLang `--tp-size {2,4}` 启动日志每卡显存分解（启动日志重跑重生）。

## 每卡权重（weight ÷ tp）

| TP | 前端预测/卡 | 实测/卡（Load weight mem usage） | 比值(vs TP1) | 判定 |
|---|---|---|---|---|
| 1 | 1192 MB | ~1.16–1.20 GB（=2×0.58=4×0.30） | ×1.00 | 基线 |
| 2 | 596 MB | **0.58 GB** | ×0.50 | ✓ |
| 4 | 298 MB | **0.30 GB** | ×0.25 | ✓ |

前端 `declaredWeightBytesPerCard`（TP-sharded 类 ÷tp、norm/embedding 按类）预测每卡 596/298 MB，
实测 580/300 MB，**误差 ~3%，×1/tp 折宽成立**。

## KV 分片（GQA：shardFactor = min(tp, kv_heads=8)）

| TP | 前端 kvShard | 实测 KV 池 max_total_num_tokens | 比值(vs TP1) | K/V 每卡 |
|---|---|---|---|---|
| 1 | 1 | 578,608 | ×1.00 | — |
| 2 | 2 | 1,158,357 | **×2.002** | 30.93+30.93 GB（kv_heads 8→4） |
| 4 | 4 | 2,318,630 | **×4.008** | 30.96+30.96 GB（kv_heads 8→2） |

每卡 KV/token 随 tp 减半/四分之一 → 同等 KV 预算下 token 容量精确 ×2/×4，**与前端 `kvBytesPerCard`
的 `min(tp, kv_heads)` 分片逐点一致**（kv_heads=8 ≥ tp，未触发 min 截断）。

## 通信（all-reduce）

- 前端 `ringAllReduceBytes`：每层 2 次 all-reduce（attn o_proj + MLP down_proj 后），
  bytes = 2×2(tp-1)/tp·B·T·H·b；预测 TP=2 总 469.8 MB、TP=4 704.6 MB（B=1,T=4096）。
- 真机结构一致：SGLang TP 走 Megatron 式**每层两段 all-reduce**（日志见 `multimem all-gather disabled
  (world_size=2)`，走 ring/all-reduce 路径）。
- **口径边界（诚实）**：本轮未用 nsys/NCCL profiler 直接测 all-reduce 字节数，仅确认**公式结构与框架
  行为一致**（2 次/层、随 (tp-1)/tp 缩放）。字节级实测留作后续（需 nsys nvtx 抓 nccl kernel）。

## 判定

- **前端 TP 折叠（权重 ÷tp、KV min(tp,kv_heads) 分片）在 TP=2/4 真机验证通过**：权重每卡误差 ~3%、
  KV 池容量精确 ×tp。补齐了此前只有 TP=1 退化恒等的空缺。
- all-reduce 通信公式**结构**已确认（Megatron 每层两段），字节级实测为后续项。
- 未验证：EP（专家并行，需 SGLang 支持的 MoE + 可加载 checkpoint）、多机 inter-node。
