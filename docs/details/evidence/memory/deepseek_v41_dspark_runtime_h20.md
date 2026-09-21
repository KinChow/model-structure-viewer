# DeepSeek-V4.1-Flash DSpark 运行时 KV/接受率 —— H20 在机记账（R2 增量）

> 复现：H20 `10.98.95.16` 容器 `dsv41_zzj_deploy`（lmsysorg/sglang:dev-dsv41），SGLang serve
> `--speculative-algorithm DSPARK --speculative-dspark-block-size 5 --tp 8 --ep-size 8`，
> ckpt `/ssd1/models/DeepSeek-V4.1-Flash-Attn-W8A8-MoE-W4A8-INT8-Dynamic`（int8-dynamic **代理** ckpt，
> 非 MSV 内置的原始 fp8/fp4），`kv_cache_dtype=fp8_e4m3`。数据取自既有 serve 日志 `/ssd1/models/dsv41_serve.log`
> （2026-09-19，只读，未重起服务）。

## 运行时记账（SGLang 自打印）

| 量 | 值 | 来源行 |
|---|---|---|
| `bytes_per_full_token` | **1670.75 B** | `DSV4 memory calculation` |
| full-token 容量 | 4,448,512 | 同上 |
| SWA fixed | 15.78 GB | 同上 |
| `swa_tokens` | 707,840（mode=cap, prefix_tails=1024） | `DSV4 SWA sizing` |
| DSpark | `block_size=5, num_draft_tokens=6, num_steps=1, eagle_topk=1` | server_args |
| DSpark 接受 | `accept len≈2.0, accept rate≈0.2` | Decode batch |
| 生成吞吐 | ≈63–67 tok/s（bs=1） | Decode batch |
| target verify CUDA graph | mem 11.09 GB / 卡 | Capture 行 |

## 对账 MSV 前端（V4.1 KV）

- MSV 内置 DeepSeek-V4.1-Flash 前端建模 **KV = 890 B/token**（CSA2 边际 + **fp4** 逐 dtype 口径，
  `evidence/structure/deepseek_v41_tensor_identity.md` 静态精确命中）。
- 运行时 `bytes_per_full_token = 1670.75 B`，但 **KV 走 fp8_e4m3（1 B/elem）而非 fp4（≈0.56 B/elem）**。
  **1670.75 / 890 = 1.877 ≈ fp4→fp8 字节翻倍比** → MSV 的逐 token KV **结构**在 fp8 dtype 下被运行时**定性坐实**，
  偏差主因是 dtype 选择、非结构错误。
- **DSpark**：MSV 前端建模 DSpark 草稿结构（draft head + block size），运行时 `block_size=5/num_draft_tokens=6` 与前端一致；
  `accept rate≈0.2` 属**运行时属性、MSV 不预测**，仅登记为运行时证据。

## 边界（仍未闭合，见 validation_status R2/R3）

- 用的是 **int8-dynamic 代理 ckpt**（非 MSV 内置的原始 fp8/fp4），且 KV=fp8 而 MSV 口径=fp4 →
  **精确逐字节闭合需**：① 用原始 `/ssd4/models/DeepSeek-V4.1-Flash`（fp8/fp4，H20 上 fp4 expert 需回退）复跑；
  ② MSV 按 fp8 KV 口径重算一版，与 1670.75 逐字节对拍。1.877× 目前是**定性**佐证、非精确对账。
- engram 常驻/预取显存、fp4 expert 真实 footprint 仍需 Hopper + 完整权重。
