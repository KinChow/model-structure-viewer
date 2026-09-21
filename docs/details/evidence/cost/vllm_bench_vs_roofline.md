# vLLM 实测 serving benchmark vs MSV roofline（Qwen3-0.6B, A100-80GB, TP=1）——cost 维跨框架补验

> 复现：容器 `vllm-0920`（A100，vLLM `0.28.1rc1.dev278`），`vllm bench serve --backend vllm --model /ssd2/models/Qwen/Qwen3-0.6B
> --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 200 --request-rate inf`（synthetic，无下载）。
> 目的：把 `cost/bench_vs_roofline.md` 的 roofline 下界验证从 **SGLang 单框架**扩到 **vLLM**——每项实测延迟应落在 MSV 地板之上、bound 分类正确。
> MSV 地板口径与 A100 取值同 `bench_vs_roofline.md`（matrixPerSecond 109.2 TMACs/s、bytesPerSecond 1835 GB/s；prefill 算力受限、decode 访存受限）。

## 真机（200 req × in512/out128，rate=inf 饱和/最大吞吐区）

| 指标 | vLLM 实测 |
|---|---|
| Successful / Failed | 200 / 0 |
| Benchmark duration | 2.71 s |
| Output token throughput | **9,451 tok/s**（peak 12,412） |
| Total token throughput | 47,257 tok/s |
| Mean / Median TTFT | **703 / 664 ms**（P99 1347） |
| Mean / Median TPOT | **14.41 / 14.89 ms**（P99 16.6） |
| GPU KV cache | 348,960 tokens / 37.27 GiB |

## 对照 MSV roofline 地板（同 `bench_vs_roofline.md` 口径）

- **KV/token**：37.27 GiB / 348,960 = **114,712 B ≈ MSV 114,688**（0.02%）——与 VL1（`../parallelism/vllm_width_tp.md`）逐值一致。
- **TPOT（decode 访存受限）**：MSV 地板 B=100 = 4.25 ms、B=1 = 0.69 ms（随 batch/context 单调增）；vLLM 实测 **14.41 ms 落在地板之上**，
  bound=memory（decode 访存主导），与 MSV decode 判定一致。
- **TTFT（prefill 算力受限）**：MSV prefill 地板（100 req 234 ms；rate=inf 200 req 瞬时到达约 ×2 ≈ 468 ms 末位下界）；vLLM 实测 **703 ms 在地板之上**（含 admission 排队）。
- **输出吞吐**：实测 9,451 tok/s < MSV decode 理论峰（B=100 地板 ~23.5k），实现效率 ~40%，与 roofline 明确不建模的调度/kernel 启动/非满带宽一致。
- **量级 + bound 分类正确**：prefill→算力/TTFT、decode→访存/TPOT，与 SGLang（`bench_vs_roofline.md`：时长 1.99×、TPOT 1.65×、TTFT 2.5×）**同模式**（vLLM 本轮 200-req 饱和、SGLang 100-req，绝对值不直接可比，但均在地板之上、bound 一致）。

## 结论

**MSV roofline 是 vLLM 与 SGLang 共同的物理下界**：两框架实测延迟均落在 MSV 地板之上、bound 分类一致（cost 维跨框架一致，非拟合值）。
至此 **cost（算力/访存/roofline）维在 vLLM + SGLang 双框架均已验证**。边界：nightly `vllm bench serve` 不再强制 temperature=0（不影响 token-count roofline）；跨节点通信费率仍留多机。

## vLLM KV 口径覆盖扩展（2026-09-21，A100 vllm-0920）

补 vLLM 侧每卡 KV 池对账（前端 `kvBytesPerCard` / per-token KV），覆盖两大架构族：

| 模型 | 架构族 | vLLM 启动 KV 池 | per-token KV（前端口径） | 判定 |
|---|---|---|---|---|
| Qwen3-0.6B | GQA | 282,224 tokens（tp1, gmu 0.4） | 28·8·128·2·2 = 114,688 B | == MSV |
| DeepSeek-V2-Lite | MLA+MoE | 656,112 tokens/rank（DP2+EP，见 R6 节） | 27·(512+64)·2 = 31,104 B（MLA latent） | == MSV |

- GQA（Qwen3-0.6B）：vLLM per-token KV = 114,688 B，与前端 GQA `kvBytesPerToken`（= SGLang 侧 112 KiB/token）逐位一致。
- MLA+MoE（V2-Lite）：vLLM DP2+EP 每 DP rank 独立满宽 MLA latent KV = 31,104 B/token（见 R6 DP-attention），EP `E=32/64` == `expertShardDivisor`。
- **诚实边界**：Qwen3.5-4B / Qwen3.8-Flash-Next（线性 hybrid, `Qwen3_5`/`Qwen4Exp`）本轮 vLLM 启动因空闲 GPU 显存竞争失败，
  **未确认 vLLM 对该 exotic arch 的支持**（非架构判定，属环境）；这两族的运行时 KV 已在 SGLang 侧验证（`structure/runtime_profiles/sglang_qwen4exp_linear.md`
  等，双通道 Mamba+KV cache），不在 vLLM 侧强凑。full serving throughput/roofline per-model 亦留干净 GPU 专跑。
