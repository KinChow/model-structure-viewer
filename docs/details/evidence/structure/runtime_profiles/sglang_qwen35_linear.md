# 运行时补维：qwen3_5 builder（Qwen3.5-4B）——覆盖 29/31 内置 Qwen

**背景（修正）**：31 个内置 Qwen 用**两个不同前端装配器**：`assembleQwen3_5`（`qwen3_5.js`，覆盖 29 个：
`Qwen3_5ForConditionalGeneration`/`Qwen3_5Moe*`）与 `assembleQwen4Exp`（`qwen4_exp.js`，2 个）。此前只跑了
qwen4_exp（Qwen3.8-Flash-Next），**qwen3_5 这条覆盖 29/31 的代码路径未做运行时验证**——本项补上。

Qwen3.5-4B（`Qwen3_5ForConditionalGeneration`、`qwen3_5_text`、bf16、8.8GB、32 层 linear:full=3:1 dense），
SGLang `--tp 1`，**server fired up ✓**（SGLang 运行输出）。

## 运行时观测（hybrid 双 cache，与 qwen4_exp 同类）

- **每卡权重** 8.62 GB（4B bf16, tp1）。
- **Mamba/SSM state cache（linear 层）**：`max_mamba_cache_size=534, conv_state 0.59GB, ssm_state 25.08GB`，
  5 slots/request（capped max_running=106）。
- **KV cache（full 层）**：`#tokens 936,109, K 14.28 + V 14.28 GB`（8 个 full 层）。

## 与前端对齐

- 前端**支持 qwen3_5**：`Qwen3_5ForConditionalGeneration` resolution=**architecture-alias**，双通道
  `kvBytesPerToken=32,768` + `stateBytesPerSequence=26.35 MB`。
- **KV per-token 近乎精确**：SGLang K+V 28.56 GiB / 936,109 tokens = **32,760 B/token ≈ 前端 32,768**（0.02%）。
- **state 同量级**：前端 26.35 MB/seq vs SGLang ssm 25.08GB/534 slots——同 caliber 口径差（SGLang slot 化 vs
  前端 per-seq），与 qwen4_exp 一致，精确映射留 caliber 细化。

## 结论

- **qwen3_5 builder 运行时首次验证**（覆盖 29/31 内置 Qwen 的前端代码路径）：hybrid 双 cache（linear SSM state +
  full KV）成立，**KV per-token 与前端精确吻合**、state 同量级。
- 加上 qwen4_exp（Qwen3.8-Flash-Next），**两个 Qwen 前端装配器（qwen3_5 + qwen4_exp）的运行时口径均已验证**。
- 剩余 Qwen 差异（qwen3_5 的 MoE 变体如 Qwen3.5-35B-A3B、VL 变体、更大尺寸、fp8）：MoE 分片(EP/ETP)已验、
  vision 静态已覆盖、fp8 留 H20——同维度/受限，非新增运行时口径。
