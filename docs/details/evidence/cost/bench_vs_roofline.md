# MSV 理论 roofline vs SGLang 实测 serving benchmark（Qwen3-0.6B, A100-80GB, TP=1）

> 口径范围（用户指定）：仅 SGLang、仅 `Qwen3-0.6B`。
> 目的：验证 MSV 的 roofline 口径（`frontend/src/cost/roofline.js` + `chips/rates.js`）
> 在真实 A100 上给出的是**物理合理的下界**——每一项实测延迟都应落在对应
> roofline 地板之上，且量级、bound 分类（prefill 算力受限 / decode 访存受限）正确。

## 复现

- roofline 计算：`scripts/evidence/cost/roofline.py`（输出由该脚本重生）；
- 实测：`python -m sglang.benchmark.serving`（原始逐请求结果 + 完整 server_info 由该 benchmark 命令重跑重生）。

benchmark 命令（localhost 访问需绕过代理）：

```bash
no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost \
python -m sglang.benchmark.serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --dataset-name random --num-prompts 100 \
  --random-input-len 512 --random-output-len 128 \
  --model /ssd2/models/Qwen/Qwen3-0.6B
```

> 注：SGLang 启动时 `qwen3_asr already used by a Transformers config` 报错，
> 根因是共享安装的 `configs/qwen3_asr.py` 在 AutoConfig 重复注册。已在 SGLang
> **源码** `qwen3_asr.py:167-168` 改为 `exist_ok=True`（该修复属 SGLang 安装，
> 不在 MSV 仓库内，会传播到所有 spawn 子进程）。

## MSV roofline 口径（与代码逐行对应）

| 量 | 公式 | 代码 | A100 取值 |
| --- | --- | --- | --- |
| matrixPerSecond | `peak_flops[dtype]·η_flops/2` | rates.js:30 | 312e12·0.7/2 = **109.2 TMACs/s**（=218.4 TFLOP/s eff） |
| bytesPerSecond | `memory_bandwidth·η_hbm` | rates.js:35 | 2039e9·0.9 = **1835.1 GB/s** |
| 单路时间 | `quantity / rate` | roofline.js:80 | — |
| bound | 五路取 max（overlap 上界） | roofline.js:118-120 | — |

芯片规格来源 `chips/public.js: nvidia-a100-80gb-sxm`；效率因子来源
`efficiency.js: DEFAULT_EFFICIENCY {flops:0.7, hbm:0.9}`。模型 FLOPs/bytes 由
config.json 推导后作为 action 向量喂入（roofline 本身不含模型知识）。

## Qwen3-0.6B 结构量（config.json 推导）

- params 总计 596,049,920（非 embedding 440,467,456）；tie_word_embeddings=true 无独立 lm_head；
- 权重 bf16 = **1.192 GB**；
- 每 token KV = 28×8×128×2×2 = **112 KiB**；
- 每 token 线性 MACs（全层）+ attention MACs 见脚本。

## A. 单序列 PREFILL（S=512）→ 算力受限

| MACs | bytes | t_matrix | t_memory | bound | t |
| --- | --- | --- | --- | --- | --- |
| 255.55 G | 1.251 GB | **2.340 ms** | 0.682 ms | matrix | 2.340 ms |

prefill 计算主导（2.34ms > 0.68ms），bound=matrix，与 roofline 对 prefill 的
算力受限判定一致。

## B. 每步 DECODE（context~576）→ 访存受限

| batch B | bytes | t_matrix | t_memory | bound | TPOT | out tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.258 GB | 0.005 | 0.686 | memory | 0.69 ms | 1,459 |
| 58 | 5.024 GB | 0.269 | 2.738 | memory | 2.74 ms | 21,187 |
| 100 | 7.798 GB | 0.464 | 4.249 | memory | 4.25 ms | 23,533 |

decode 全 batch 访存主导（权重 1.19GB + KV 随 batch/context 线性增长），bound=memory。

## C. 聚合 roofline（100 req × in512/out128）

- prefill 总 MACs 25.56 T → 算力地板 **234 ms**；
- decode 总字节 997 GB（KV 845 + 权重 153，context 512→639 积分）→ 访存地板 **544 ms**；
- **MSV roofline 地板（prefill+decode）= 778 ms**。

## D. 实测（`python -m sglang.benchmark.serving`）

| 指标 | 值 |
| --- | --- |
| Successful requests | 100 |
| Benchmark duration | 1.55 s |
| Output token throughput | 8276.88 tok/s |
| Total token throughput | 41384.41 tok/s |
| Mean / Median TTFT | 584.26 / 571.72 ms |
| Mean / Median TPOT | 7.03 / 7.49 ms |
| Mean ITL | 7.03 ms |
| Concurrency | 95.50 |

## 对照与偏差归因

| 对照 | 理论（MSV roofline） | 实测 | 比值 | 归因 |
| --- | --- | --- | --- | --- |
| 端到端时长 | 778 ms（地板） | 1550 ms | **1.99×** | roofline 是下界；实测含调度/排队/kernel 启动/图切换开销，实现效率 ~50% 相对地板（地板本身已含 η_flops=0.7, η_hbm=0.9） |
| TPOT | 4.25 ms（B=100） | 7.03 ms | **1.65×** | decode step 未 100% 打满带宽；context 增长使 KV 读逐步变大；采样+Python 调度每步开销；非所有请求每步都在 decode（prefill 波次占并发但不产出 token） |
| 输出吞吐 | ~23.5k tok/s（B=100 地板） | 8.28k tok/s | 0.35× | 同上；且实际瞬时 batch < 100（吞吐/TPOT 反推 ~58） |
| TTFT | 234 ms（100 req prefill 算力地板，末位请求下界） | 584 ms | 2.5× | rate=inf 下 100 请求瞬时到达，chunked prefill 分波 + decode 交织 → TTFT 含排队等待 |

**结论**：每一项实测延迟均落在对应 MSV roofline 地板**之上**（时长 1.99×、
TPOT 1.65×、TTFT 2.5×），且 bound 分类正确（prefill 算力受限、decode 访存受限），
量级正确。这验证了 MSV roofline 是物理成立的下界口径（overlap 上界 / 保守地板），
而非拟合值。偏差来自 roofline 明确不建模的部分：调度、排队、kernel 启动、
非满带宽利用、context 增长——与文档中 roofline "估计/下界" 的标注自洽。
