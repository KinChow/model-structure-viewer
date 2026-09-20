# 跨 builder KV/state 逐字节 dtype 审计（H20，本机 /ssd*/models）

**目的**：不只验"元素口径对上"，而是把 KV/state cache 拆到**逐字节（dtype × 元素）**对前端 memory lens，
挖 dtype 不一致的前端 bug。锚点环境 8× H20-3e(SM90)/CUDA13/sglang dev；减层 `--load-format dummy` 真机取运行时
dtype，SGLang cache-param 代码路径（scheduler 分配同一段）取字节。

## 运行时 dtype（SGLang 真值） vs 前端 dtype（`cost/memory.js`）

前端 `residentMemoryFromGraph`：带 `cache_kv_dtype` 的叶按逐 dtype 算（仅 dsv4 家族传了），**其余叶 KV/index/state
一律按统一 `kvBytes`（默认 bf16=2B）**（`memory.js:177-188`）。运行时各 cache 的**架构强制 dtype**：

| cache | 运行时 dtype（SGLang） | 前端 dtype | 结论 |
|---|---|---|---|
| GQA KV（K/V 分列） | bf16（`kv_cache_dtype`，量化 ckpt 除外） | bf16 | ✅ 一致 |
| MLA latent（kv_lora+qk_rope） | bf16 | bf16 | ✅ 一致 |
| dsv4/v41 压缩 KV + index | fp4/fp8 + 尺度（前端传 `indexDtype`） | fp4/fp8 | ✅ 一致（V4.1 890 B/token 精确） |
| **DSA index**（`dsa_sparse_mla`） | **fp8(uint8 1B) + E8M0 尺度(4B/128)** | **bf16(2B)** | ❌ **Bug 1** |
| **KDA/GDN recurrent(ssm) state** | **fp32(4B)**（`mamba2_state_dtype` 默认，qwen3_5 config 显式 float32） | **bf16(2B)** | ❌ **Bug 2** |
| KDA/GDN conv state | bf16(2B) | bf16(2B) | ✅ 一致 |

## Bug 1 — DSA index 按 bf16 计（实为 fp8）→ KV-per-token 高估

`dsa_sparse_mla`（`ops/index.js:1146-1149`，glm5_next + deepseek_v32 共用）不传 index dtype → 前端 index=128×2B=256 B/层；
运行时 `index_key_cache.py` 存 fp8 128×1B + 4B 尺度 = 132 B/层（**1.94×**）。**H20 实测锚**：减层 glm5_next(2 DSA 层)
KV 池 `73.71 GB / 34,233,856 tok = 2312 B/token`，与 fp8-index 模型精确吻合；前端 bf16-index 会给 2560 → **高估 +10.7%**。
对比 `dsv4_sparse_mla`（`ops/index.js:768-779`）正确传了 `indexDtype`，故此 bug 仅在 glm5_next/deepseek_v32(9 模型)。

## Bug 2 — KDA/GDN recurrent state 按 bf16 计（实为 fp32）→ 显存低估 ~48%（影响面最大）

`linearStateResidentDecl` 返回单一 `state_elements`，`memory.js:188` 一律 ×bf16(2B)；但 SGLang
`mamba2_state_dtype` 的 **temporal(ssm) 默认 fp32(4B)**、conv bf16(2B)（`mamba_cache_per_req = conv×2 + ssm×4`）。
ssm 占绝对多数 → 前端**低估线性 state ~1.934×（48.3%）**。逐 builder（SGLang cache-param 真值 vs 前端 bf16）：

- **glm5_next**：conv 73,728 + ssm 1,048,576 → 运行时 4,341,760 vs 前端 2,244,608 B/层（1.934×）。真实 GLM-5.3-Flash 34 KDA 层：**140.78 vs 72.78 MiB/seq**。
- **kimi_k3**（num_heads96/head_dim128）：运行时 6,512,640 vs 前端 3,366,912 B/层（1.934×）。
- **qwen3_5**（GDN，config `mamba_ssm_dtype=float32`）：H20 减层真机 `Mamba Cache ssm_state` + `FlashInfer/Triton GDN kernels` 起服务确认；同类低估。

## 观察 3 — W8A8C8 量化 ckpt 强制 int8 KV（前端默认 bf16）

`Qwen3-30B/Qwen3.5-35B-A3B-INT8-W8A8C8`、`DeepSeek-*-W8A8-INT8` 的 config `quantization_config.kv_cache_scheme
= {num_bits:8, type:int, strategy:attn_head}` → 运行时 KV int8(1B)。前端 KV dtype 是用户可选参数、**不自动读 ckpt 的
kv_cache_scheme** → 默认 bf16 时对这些 ckpt 高估 KV 2×。属"用户口径 vs ckpt 强制口径"，非纯 bug，登记为可增强项。

## 已核对为一致（clean）

- GQA KV：qwen3_5 减层 H20 实测 `K/V 40.17GB 各 / 21,061,889 tok / 2 full 层 = 2047.9 B/层` == 前端 `2·kv_heads·head_dim·bf16 = 2048`（0.0%）。
- MLA latent、dsv4/v41 压缩 KV+index：前端已逐 dtype 建模（V4.1 890 精确），无偏差。

## 影响与修法（未在本机改——无 node，改后须 `node --test`/`verify:models`/重生成 golden）

- Bug 2 影响最大（线性 hybrid 模型的最大 cache，低估近 2×，波及 glm5_next/kimi_k3/qwen3_5/qwen4_exp）。修：`linearStateResidentDecl`
  拆 conv/recurrent 元素 + `memory.js` 按 conv=bf16 / ssm=fp32 分别计（参照 dsv4 的逐 dtype 写法）。
- Bug 1 修：`ops/index.js:1146` 的 `cacheResidentDecl` 补 `kvDtype`(bf16)+`indexDtype`(fp8)+growth，对齐 768-779。
- 二者同根：memory lens 对"架构强制 dtype"（fp8 index、fp32 ssm）套了用户统一 KV dtype，仅 dsv4 家族做了逐 dtype。

## 复现

减层件 `scripts/evidence/structure/glm5_next_reduce.py`（glm5_next）+ 本文档内 qwen3_5 减层器；
`sglang.launch_server --load-format dummy --tp1`；字节真值 `KimiLinearStateShape`+`mamba2_state_dtype`（ssm fp32）
与 `index_key_cache.py`（fp8+尺度）。

## 配置可变性核实（回答"DSA index 能否配置设置"）+ dsv4 澄清

- **DSA index dtype = 框架硬编码 fp8，不可配置**：`DSATokenToKVPool.index_k_with_scale_buffer_dtype = torch.uint8`
  （`memory_pool.py:4820`，class 变量）/ `DeepSeekV4IndexerPool`（`deepseek_v4_memory_pool.py:336`）。index_buf =
  128×fp8(1B) + 1×fp32 scale(4B) = 132 B/token。**不受 `--kv-cache-dtype` 影响**（那个只改 MLA latent 存储），
  无 env / server-arg 开关。→ **Bug 1 是确定的前端错，配置改不掉运行时 fp8，前端必须建模 fp8+尺度**。
- **ssm state dtype = 默认 fp32，但可配置**：`mamba2_state_dtype()` 默认 temporal=fp32、conv=bf16；temporal 可经
  config `mamba_ssm_dtype` / env `SGLANG_MAMBA_SSM_DTYPE` / `--mamba-ssm-dtype` 改 bf16/fp16。**conv 两侧恒 bf16（clean）**。
  → **Bug 2 正确修法 = 前端读 `config.mamba_ssm_dtype`（缺省 fp32），只对 recurrent/temporal 分量套该 dtype、conv 保持
  bf16**；不是简单"全改 fp32"。qwen3_5 config 显式 float32。
- **前端只对 dsv4 读 config dtype**（`quantization_config.expert_dtype`/`quant_method` → F4/F8_E4M3）；**不读**
  `mamba_ssm_dtype` / `kv_cache_scheme` / 任何 index dtype（grep 零命中）。用户仅有一个统一 `kvElementBytes` 选择器（默认 2）。
- **dsv4/v41 澄清（不是 bug，撤回"另一个方向的 index bug"担忧）**：前端按 V4.1 **设计的 fp4** 计 KV/index（对官方
  890 B/token 精确、已闭合）；但**本 SGLang dev build 默认关 fp4 indexer**（`enable_deepseek_v4_fp4_indexer=False`，
  实验/HIP-only）→ 该 build 默认走 fp8 index(132B)+bf16 压缩 KV，footprint 比设计值大。这是"前端对齐模型设计口径
  vs 本 build 未启 fp4"的**参考基准差异**，非前端错；启 fp4 的 build/官方部署即对齐。
- **其它 cache 复核（clean / 缺项）**：conv state（bf16 两侧）、SWA 窗口 KV（随 kvDtype 两侧）、MiniMax-M3 块稀疏
  index（`index_dtype` 缺省随主 KV=bf16，两侧一致）—— 均 **clean**。**MTP/投机 draft 的 state/KV** 前端完全不建模
  （`speculative_algorithm` 开时运行时另分配 draft 状态缓冲）——静态工具缺项，登记（是否建模属设计取舍）。
