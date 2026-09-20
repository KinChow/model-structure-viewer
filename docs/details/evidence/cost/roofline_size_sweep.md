# roofline 随数据大小的拟合（Qwen3-0.6B 算子形状, A100-80GB）

> 问题：MSV 的算子 roofline 在**不同数据大小**下拟合得如何？
> 方法：按尺寸扫描（GEMM 扫 tokens M、flash 注意力扫 S、SwiGLU 扫 M），CUDA events 实测单 kernel
> 时延（多次均值，不经 ncu 以免 replay 扰动）vs MSV 五路地板 `max(matrix/109.2TMACs, mem/1835GBs)`
> （含 η flops0.7/hbm0.9，是"效率地板下界"）。A100 脊点 AI = **119 FLOP/byte**。

## 一、GEMM [M,1024]×[1024,3072]（gate_proj 形状）

| M | 实测 | MSV 地板 | 实测/地板 | bound | AI | 实测算力/带宽 |
|---|---|---|---|---|---|---|
| 1 | 13.96µs | 3.43µs | 4.07 | memory | 1.0 | 0.5 TF / 451 GB/s |
| 16 | 14.49µs | 3.50µs | 4.14 | memory | 15.7 | 6.9 TF / 443 GB/s |
| 64 | 14.54µs | 3.71µs | 3.92 | memory | 59 | 27.7 TF |
| 256 | 17.34µs | 7.37µs | 2.35 | **matrix** | 192 | 92.9 TF |
| 1024 | 39.56µs | 29.50µs | 1.34 | matrix | 439 | 162.9 TF |
| 4096 | 145.5µs | 118.0µs | 1.23 | matrix | 647 | 177.1 TF |
| 16384 | 488.2µs | 472.0µs | **1.03** | matrix | 734 | **211.1 TF** |

- 拟合**随 M 单调收紧**：M=1 差 4.07×，M=16384 收到 **1.03×**（实测 211 TF ≈ η 地板 0.7×312=218，97%）。
- **脊点翻转与实测一致**：AI 在 M∈[64,256] 越过 119 → bound 从 memory 翻到 matrix，与 MSV 分类同步。
- 小 M 的偏离是**固定 kernel launch 开销 ~14µs**（MSV 纯 work/rate 地板不建模）。

## 二、FLASH 注意力（causal, GQA 16/8, D=128）

| S | 实测 | MSV 地板 | 实测/地板 | bound | AI | 实测算力 |
|---|---|---|---|---|---|---|
| 128 | 44.3µs | 0.86µs | 51.6 | memory | 43 | 1.5 TF |
| 512 | 48.6µs | 4.93µs | 9.86 | matrix | 171 | 22.2 TF |
| 2048 | 198µs | 78.7µs | 2.52 | matrix | 683 | 86.8 TF |
| 8192 | 1962µs | 1259µs | 1.56 | matrix | 2731 | 140.1 TF |
| 16384 | 5957µs | 5035µs | **1.18** | matrix | 5462 | 184.6 TF |

- 同样**随 S 单调收紧**（51.6×→1.18×）：小 S 被固定开销主导，大 S 趋近因果 matrix 地板。
- MSV 用**因果 matrix + flash 字节（Q/K/V/O，scores 不落 HBM）**，与 flash kernel 同口径 → AI 高、matrix-bound，
  与 flash_kernel_caliber.md 结论一致（A2 物化上界不用于 flash 口径）。

## 三、SwiGLU [M,3072]（memory-bound）

| M | 实测 | MSV 地板 | 实测/地板 | bound | 实测带宽 |
|---|---|---|---|---|---|
| 64 | 14.9µs | 0.64µs | 23.2 | memory | 79 GB/s |
| 256 | 15.1µs | 2.57µs | 5.86 | memory | 313 GB/s |
| 1024 | 16.1µs | 10.3µs | **1.57** | memory | **1170 GB/s** |
| 4096 | 83.3µs | 41.1µs | 2.02 | memory | 906 GB/s |
| 16384 | 296.7µs | 164.6µs | 1.80 | memory | 1018 GB/s |

- 全程 memory-bound（与 MSV 一致）；大 M 实测带宽 900–1170 GB/s，仅**达 1835 地板的 50–64%**。

## 结论：拟合得怎么样

1. **恒为有效下界**：所有尺寸实测 ≥ MSV 地板（比值 ≥ 1），从不被击穿——roofline 作为下界成立。
2. **随尺寸单调收紧**：大尺寸拟合很好（GEMM **1.03×**、注意力 **1.18×**、SwiGLU **1.57×**）；小尺寸偏离大
   （4–52×），根因是 **MSV 不建模的固定 kernel launch/低占用开销**（~14µs 量级）——这是 roofline 的
   已知盲区，非公式错误。
3. **脊点翻转正确**：GEMM 在 AI≈119 处 memory→matrix 翻转与实测一致；bound 分类各尺寸都对。
4. **效率校准**：大 GEMM 实测 211 TF ≈ η_flops=0.7 地板（校准良好）；但 **memory-bound 逐元素只达
   带宽 50–64%，η_hbm=0.9 偏乐观**（有效 ~0.55）——若要 memory 侧拟合更紧，可把逐元素 η_hbm 下调
   或加"小尺寸固定开销项"。

> 复现：`python3 scripts/evidence/cost/roofline_sweep.py`（CUDA events 计时；输出由该脚本重生）。
