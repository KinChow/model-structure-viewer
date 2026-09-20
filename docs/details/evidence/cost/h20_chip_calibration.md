# H20-3e 芯片规格微基准校准 —— 验证 MSV H20 roofline 速率常数（不同硬件·算力）

> 复现：H20 `10.98.95.16` 容器 `sglang-dev-20260918-20518d85`，GPU0，torch 2.13+cu130。
> `python -c` 微基准：bf16 GEMM（n=8192，2·n³ FLOP，100 迭代取时）+ HBM 带宽（1e9 元素 bf16 clone，读+写=numel·2·2 字节，50 迭代）。
> 目的：此前把 `H20-3e` 加进 `frontend/public/chips.local.json`（本地 gitignore）时规格为**占位值、bound 标 untrusted**；本项微基准校准坐实之，
> 使 MSV 在 H20 上的 roofline（`cost/roofline.js` + `chips/rates.js`）速率可信——把 A100 已验的 roofline 口径扩到 **H20 硬件**。

## 实测（NVIDIA H20-3e）

| 量 | 实测 | chips.local.json 规格(peak) | 实测/peak | 判定 |
|---|---|---|---|---|
| bf16 GEMM | **138.3 TFLOPS** | 148 TFLOPS | 93.5% | ✅ peak≈148 坐实（纯 GEMM 达峰 ~93%，实测<peak 合理） |
| HBM 带宽 | **3874 GB/s** | 4800 GB/s (4.8 TB/s HBM3e) | 80.7% | ✅ peak≈4.8TB/s 坐实（clone 达 ~81%，实测<peak 合理） |

## 与 MSV roofline 口径对账

- MSV `matrixPerSecond = peak_bf16·η_flops/2`（η_flops=0.7）= 148e12·0.7/2 = 51.8 TMACs/s（103.6 TFLOP/s eff）。
  实测纯 GEMM 138.3 TFLOPS > 103.6 → **η_flops=0.7 对纯 kernel 是保守值**，MSV 计算地板（用 0.7）对 H20 serving（含开销）是**有效下界**（同 A100 bench 结论：实测 serving ≥ 地板）。
- MSV `bytesPerSecond = peak_bw·η_hbm`（η_hbm=0.9）= 4800·0.9 = 4320 GB/s。实测 clone 3874 GB/s（81%）< 4320 → η_hbm=0.9 略乐观（与 `principles.md` 既有标注一致），
  但访存地板 time=bytes/4320 < bytes/3874 = 实测 → **仍是有效下界（实测 ≥ 地板）**。
- **结论**：H20-3e 的 peak bf16 / HBM 规格经微基准坐实（实测均在 peak 之下、比例合理）；MSV 的 **H20 roofline 速率常数可信、bound 不再是占位**。
  算力/roofline 维**从 A100 扩到 H20（不同硬件）**：A100 有 serving bench（`bench_vs_roofline.md`）+ vLLM（`vllm_bench_vs_roofline.md`），H20 有本项规格校准坐实速率地板。

## 边界

- 本项校准 peak 速率 + η 口径；H20 端到端 serving-vs-roofline bench 未单跑（A100 双框架已验 serving≥地板，roofline 为框架无关物理下界，H20 规格既已坐实即可信）。
- fp8/fp4 峰值张量核速率未微基准（fp8 前向已在 V4.1 dsv4 验证跑通；fp4=Blackwell 不可得）。
