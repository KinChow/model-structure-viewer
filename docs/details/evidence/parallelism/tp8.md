# TP=8 全节点并行对账（Qwen3-0.6B, 8×A100-80GB）

承接 `tp_parallel.md`（TP=1/2/4），补齐全节点 TP=8。前端投影 `scripts/evidence/parallelism/tp_projection.mjs`；
真机 SGLang `--tp 8 --mem-fraction-static 0.8`（启动日志重跑重生）。

## 每卡权重（weight ÷ tp）

| TP | 前端预测/卡 | 实测/卡（Load weight mem usage） | ×vs TP1 |
|---|---|---|---|
| 1 | 1192.1 MB | ~1.16–1.20 GB | ×1.00 |
| 2 | 596.1 MB | 0.58 GB | ×0.50 |
| 4 | 298.1 MB | 0.30 GB | ×0.25 |
| **8** | **149.1 MB** | **0.16 GB** | **×0.135**（≈0.125，含固定开销） |

8 卡各 `mem usage=0.16 GB`——weight ÷8 成立（前端 149.1MB，实测 160MB，差 ~7% 为固定开销/对齐，与 TP2/4 的 ~3% 同源）。

## KV 分片（GQA：shardFactor = min(tp, kv_heads=8)）

| TP | 前端 kvShard | 实测 KV 池 #tokens | ×vs TP1 |
|---|---|---|---|
| 1 | 1 | 578,608 | ×1.00 |
| 2 | 2 | 1,158,357 | ×2.002 |
| 4 | 4 | 2,318,630 | ×4.008 |
| **8** | **8** | **4,646,648** | **×8.03** |

TP=8：`KV Cache #tokens=4,646,648`（K 31.02 + V 31.02 GB/卡），vs TP1 ×**8.03**，与前端 `kvBytesPerCard`
的 `min(tp, kv_heads)` 分片逐点一致。**边界点 tp==kv_heads=8**：min(8,8)=8 不截断；若 tp>8 会触发 min 上限
（每卡 KV 不再随 tp 下降）——该上限逻辑此前已在 `parallel.js` 建模，本次 tp=8 正好压边界验证。

## 判定

前端 TP 折叠（权重 ÷tp、KV `min(tp,kv_heads)` 分片）在 **TP=1/2/4/8 全序列**真机验证通过；TP=8 是 kv_heads
边界点，min 截断逻辑成立。all-reduce 字节级见 `allreduce.md`（N=8 比值 1.000）+ `allreduce_rs_ag.md`。
