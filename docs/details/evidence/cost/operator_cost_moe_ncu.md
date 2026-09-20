# A-2 ncu 补齐 · MoE routed-expert matrix + MoE/MLA bytes 通道（DeepSeek-V3 减层, A100）

> 补齐 Part1 两个诚实缺项：① routed-expert matrix（FlopCounter 对 HF eager MoE loop 盲区）；
> ② MoE/MLA bytes 通道。方法：孤立 GEMM FlopCounter（matrix）+ ncu DRAM（bytes），DeepSeek-V3 减层形状
> （hidden 7168, moe_inter 2048, q_lora 1536, kv_a 576, S=128, topk=4, active=512）。

## 缺项① routed-expert matrix：孤立 GEMM FlopCounter（`scripts/evidence/cost/moe_iso_flops.py`）

FlopCounter 数不了 HF MoE loop 的分组专家 GEMM；但**孤立 GEMM 能被正确计数**——用它证明前端 active-compute
公式正确（HF loop 只是对计数器隐藏该计算，计算本身=公式）：

| 算子 | torch FLOPs（孤立 GEMM） | 前端 MACs×2 | 判定 |
|---|---|---|---|
| MoE routed active（gate+up+down, 512 pairs） | 45,097,156,608 | 45,097,156,608 | **精确相等** |
| MLA q_a compress | 2,818,572,288 | 2,818,572,288 | **精确相等** |
| MLA kv_a compress | 1,056,964,608 | 1,056,964,608 | **精确相等** |

→ **前端 `fused_moe_mlp`（tokens×topk×3×h×moe_inter）与 MLA 压缩 matrix 公式经孤立 GEMM 真值坐实为精确**；
Part1 的 "routed experts FlopCounter 漏计" 是 HF loop 的工具局限，非前端错误。

## 缺项② bytes 通道：ncu DRAM 读（`scripts/evidence/cost/moe_bench.py` + `scripts/evidence/cost/run_ncu.sh`）

per-GEMM MSV compulsory 读(weights+actIn) vs ncu DRAM 读：

| 算子(GEMM) | MSV 读 | ncu DRAM 读 | 比值 | 说明 |
|---|---|---|---|---|
| moe_gate [512,7168]×[7168,2048] | 36,700,160 | 36,723,712 | **1.0006** | 大 GEMM，精确 |
| moe_down [512,2048]×[2048,7168] | 31,457,280 | 32,433,792 | 1.031 | 3% |
| mla_q_a [128,7168]×[7168,1536] | 23,855,104 | 27,031,936 | 1.133 | 小 GEMM，固定开销占比大 |
| mla_kv_a [128,7168]×[7168,576] | 10,092,544 | 11,298,304 | 1.120 | 同上 |

→ **MSV compulsory 读 ≤ ncu DRAM 读**（有效下界），大 GEMM(moe_gate)近精确(1.0006)，小 GEMM(MLA 压缩)
偏 12–13%（与 roofline 尺寸扫描一致：小 kernel 固定 DRAM 读开销占比大）。写侧部分常驻 cache（moe/q_a 有写、
kv_a=0），MSV total 仍为保守上界。

## 结论（缺项闭合）

- **matrix**：MoE routed-expert + MLA 压缩公式**精确**（孤立 GEMM 真值）——闭合 FlopCounter 盲区。
- **bytes**：MoE/MLA GEMM 的 MSV compulsory 读 = ncu DRAM 读（大算子精确、小算子界内 <13%）——闭合 bytes 通道。
- 仍未做（诚实）：MLA 注意力 bmm 0.30× 的 latent/吸收口径细化；MoE 聚合 all-expert 常驻权重字节（vs 单专家 GEMM）。
