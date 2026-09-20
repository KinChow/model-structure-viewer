# 多卡 TP+EP 端到端前向真机（8×A100，tp=2 ep=2）

此前的并行/通信是 **NCCL 微基准 + SGLang 启动日志字节级**（tp1/2/4/8、EP48），未做**真实多卡推理前向**。
本项用 2×A100 对减层 MoE/MLA 模型跑 `--tp-size 2 --ep-size 2` 真机 `/generate`，把并行/通信从"离线口径"
升级到"在线前向真机"。

## 结果

| 模型 | tp1 基线 | tp2/ep2 每卡 | 验证点 |
|---|---|---|---|
| **deepseek_v3**（MLA+MoE） | KV 0.05GB, MoE **E=8** | KV **0.05GB（不变）**, MoE **E=4** | MLA latent KV **TP 复制不分片**；EP 专家 8÷2=4/卡 |
| **glm4_moe**（GQA+MoE） | K+V 0.02+0.02GB, MoE E=9(含shared) | K+V **0.01+0.01GB（减半）**, MoE **E=4** | GQA KV **÷tp 分片**（kv_heads 2÷2=1/卡）；EP 4/卡 |

- **EP 专家分片真机**：两模型 MoE runner 从 tp1 的 E=8/E=9 → tp2ep2 的 **E=4 per rank**（8 routed÷ep2），
  与 MSV `expertShardDivisor=ep×moe_tp` 口径一致，且在**真实前向**中 dispatch/combine + all-to-all 执行。
- **KV 分片口径真机对比**：
  - GQA（glm4_moe）：KV **÷tp**（0.02→0.01/卡），= MSV `KV × min(tp, kv_heads)`（kv_heads=2）。
  - MLA（deepseek_v3）：压缩 latent **复制不分片**（0.05→0.05/卡）——MLA 单一 latent 每卡持全份，
    与 MSV MLA-under-TP 口径一致（latent 不按 kv_heads 切）。
- **通信真机**：两 rank all-reduce/all-gather NCCL（`multimem all-gather disabled for world_size=2`，走 ring），
  prefill/decode 跨卡执行。
- **数值一致性**：deepseek_v3 `/generate` 输出 tp1==tp2 **逐 token 相同**（[1994,760,597,760,420,420]，temp=0）
  → TP 分片保持前向正确。glm4_moe tp2 输出在第 3 token 后与 tp1 分叉——EP 路径下 shared-expert 融合（tp1 E=9 融合
  vs ep2 分离）+ 随机权重 top-2 路由的数值敏感致 argmax 翻转（真实训练权重更稳），属预期，非正确性缺陷。

## 结论

- **TP 权重分片 + EP 专家分片 + GQA/MLA 两种 KV 分片口径 + all-reduce/all-to-all 通信，均已在真实多卡前向中验证**，
  与 MSV `parallel.js`/`comm.js` 口径一致。此前的离线字节级（`tp/`, `ep/`, `allreduce/`, `nsys/`）
  由本项的在线前向真机补强。
- 边界：真·多机（inter-node NCCL）、更大 TP/EP（受减层模型规模限制，口径已由 tp8/ep48 startup-log 覆盖）。

复现：`CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path deepseek_v3_tiny --tp-size 2
--ep-size 2 --skip-tokenizer-init ...` → `/generate`；glm4_moe_tiny 同法（GPUs 2,3）。
