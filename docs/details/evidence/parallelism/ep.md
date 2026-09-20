# Part 2 · EP 专家并行：前端投影 vs SGLang 真机（reduced qwen3_moe, A100×2）

> 减层随机 qwen3_moe（8 experts, topk=2, 4 层, 256M 随机权重, `scripts/evidence/parallelism/build_ep_ckpt.py`, **零权重下载**，
> save_pretrained 到本地 + 复制 Qwen3-0.6B tokenizer）→ SGLang `--tp-size 2 --ep-size 2` 加载。
> 前端投影：`expertShardDivisor` + `expertAllToAllBytes`（`scripts/evidence/parallelism/tp_projection.mjs` 同源 API）。

## 专家分片（expertShardDivisor）

| 量 | 前端预测(ep=2) | SGLang 真机 | 判定 |
|---|---|---|---|
| 每 rank 专家数 | experts/ep = 8/2 = **4** | MoE kernel config **`E=4,N=768`**（每 rank 跑 4 专家） | **✓ 逐点一致** |
| expertShardDivisor.divisor | 2（epSize×moeTp=2×1） | 专家权重 ÷2/rank | ✓ |
| EP 模式 | ep>1 → EP 分片 | 日志 `TP0 EP0`/`TP1 EP1`、`ep_size=2` | ✓ |

`E=4` 是 SGLang MoE runner 打印的每 rank 专家数——**8 experts / ep=2 = 4，与前端 `expertShardDivisor`
（每 rank 拥 E/ep 个完整专家）逐点一致**。每卡权重 0.38GB（tp=2 attention 分片 + ep=2 专家分片）。

## all-to-all（expertAllToAllBytes）

- 前端预测：每层 dispatch+combine = 2·B·T·topk·H·b（ep>1 触发）；本配置 T=4096,H=1024,topk=2 → **33.55 MB/层**。
- SGLang：`moe_a2a_backend='none'`（小模型未启 DeepEP），EP dispatch/combine 走标准 all-to-all——**结构一致**
  （ep>1 才有 all-to-all，与前端 `ep<=1 → 0` 的门控一致）。
- **字节级直测（`scripts/evidence/parallelism/alltoall_bench.py`, NCCL all_to_all_single, ep=2/4/8）**：搬运的 dispatch token-expert 载荷逐 ep
  均 **16.78 MB == 前端 `B·T·topk·H·b`（比值 1.000）**，dispatch+combine = 33.55 MB/层与预测一致；载荷量**与 ep 无关**
  （ep 只改延迟/带宽：busbw 130→262→306 GB/s，ep=2/4/8），印证前端公式用"每 token 激活专家数"而非专家总数、且无隐藏 ep 因子。

## 判定

- **前端 EP 专家分片（每 rank E/ep 个完整专家）在真机验证通过**：SGLang ep=2 → E=4/rank 逐点一致，
  补齐了此前 EP 完全未验的空缺。
- **all-to-all 字节级已从"仅结构+门控"升级为"字节直测"**：ep=2/4/8 搬运载荷 == 前端 dispatch 字节（比值 1.000），
  与 ep 无关；触发门控（ep>1）亦一致。
- 口径边界（诚实）：微基准按等分 all_to_all_single 的 dispatch 载荷构造，验证的是**载荷字节量**与前端公式一致；
  更强的 nsys per-kernel NVLink 字节见 `nsys.md`。
- 未验证：混合 ETP（moe_tp>1）、DeepEP、多机。
