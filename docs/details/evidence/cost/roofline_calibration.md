# roofline η_hbm 校准结果（A100/Qwen3-0.6B）

> 校准：`DEFAULT_EFFICIENCY.hbm` 0.9 → **0.7**（efficiency.js）。目标：memory-bound 拟合到量级一致、
> 保持下界、可泛化到其他环境。η_flops 保持 0.7（大 GEMM 实测 97% 已坐实）。

## 前后对照（实测/地板 比值；实测时延不变，仅地板口径变）

| 算子 | bound | η_hbm=0.9（前） | η_hbm=0.7（后） | 变化 |
|---|---|---|---|---|
| SwiGLU M=1024 | memory | 1.57 | **1.24** | 收紧 |
| SwiGLU M=16384 | memory | 1.80 | **1.40** | 收紧 |
| GEMM M=1（小，memory） | memory | 4.07 | 3.19 | 收紧（余量为 launch 开销） |
| 注意力 S=128（小，memory） | memory | 51.6 | 40.5 | 收紧（余量为 launch 开销） |
| GEMM M=1024/16384 | **matrix** | 1.34 / 1.03 | 1.34 / 1.03 | **不变**（matrix-bound 不受 η_hbm 影响） |
| 注意力 S=16384 | **matrix** | 1.18 | 1.18 | 不变 |

## 判定

- **memory-bound 拟合收紧到量级一致**（SwiGLU 1.57–1.80 → 1.24–1.40），且**仍是有效下界**
  （比值 > 1；有效带宽地板 1427 GB/s > 实测最好点 1170 GB/s，未击穿）。
- **matrix-bound 预测不变**：η_flops 未动、GEMM/大 S 注意力比值不变——校准不牺牲已验证的算力侧。
- **脊点 119 → 153 FLOP/byte**：memory→matrix 翻转点右移，但 GEMM 的翻转仍落在 M∈(64,256]，与实测一致。
- **泛化性**：0.7 是跨厂商可辩护的可达 HBM 比例（非 overfit 到 A100 逐元素 0.44–0.57）；默认值仍可被
  `chip.efficiency` / UI 覆盖。小尺寸残余偏离（launch 开销）不建模——设备相关、不可泛化，作为已知 caveat。

## 复现

`python3 scripts/evidence/cost/roofline_sweep.py`（已含 η_hbm=0.7；校准输出与 0.9 基线均由该脚本重生）。
回归：node --test 410 / verify:models 60/60 / docs:check 全绿。
