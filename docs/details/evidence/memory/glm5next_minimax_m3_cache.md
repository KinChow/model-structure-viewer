# assembleGlm5Next & assembleMiniMaxM3 cache 口径运行时对账（+ 一处 MSV 口径修正）

延续减层法（不建全权重，直取 SGLang 自身 cache-param 代码路径 = scheduler 分配 KV/state/index 池的同一段），
对最后两个未验 builder 收口。配置用仓库内置真值：`models/zai-org/GLM-5.3-Flash-BF16/config.json`、
`models/MiniMaxAI/MiniMax-M3/config.json`。

## glm5_next（GLM-5.3-Flash-BF16，Glm5NextForConditionalGeneration）

45 层 = KDA 线性 34 层 + DSA 稀疏 MLA 11 层（[3,7,…,43]）。三种 cache：

| 量 | SGLang 运行时 | MSV 前端 | 差 |
|---|---|---|---|
| KDA state /层/请求 | 1,122,304（conv 73,728 + temporal 1,048,576） | 1,122,304 | **0.0%** |
| MLA latent kv /token/层 | 512（kv_lora_rank 512 + qk_rope 0） | 512 | **0.0%** |
| DSA indexer k /token/层 | 128（index_head_dim；fp8 存另 +4B/128 尺度） | 128 | **0.0%** |

- KDA：SGLang `Glm5NextTextConfig.mamba2_cache_params` → `KimiLinearStateShape.create(num_heads=64,
  head_dim=128,conv=4)` = conv (3,24576)=73,728 + temporal (64,128,128)=1,048,576 → **1,122,304**，与 MSV
  `linearStateResidentDecl` 同式（与 kimi_k3 复用同一 shape）。
- DSA index：SGLang `index_key_cache.py` 每 token = `index_head_dim + index_head_dim//quant_block_size*4`
  = 128 + 4（uint8，fp8 索引每 128 维一枚尺度）；核心元素 128 = MSV `indexElements=index_head_dim`。

## minimax_m3（MiniMax-M3，MiniMaxM3SparseForConditionalGeneration）

60 层 GQA（kv_heads=4, head_dim=128）+ 块稀疏（layers 3-59，`sparse_disable_index_value=1` → indexer 仅存 K）。

| 量 | SGLang 运行时 | MSV 前端 | 差 |
|---|---|---|---|
| GQA kv /token/层 | 1,024（2·kv_heads·head_dim=2·4·128） | 1,024 | **0.0%** |
| 稀疏 indexer /token/稀疏层 | 128（单头 K；`head_num=1, head_dim=idx_head_dim=128`） | 128 | **0.0%**（修正后） |

**发现并修正一处 MSV 口径 bug**：`ops/index.js` minimax sparse 分支此前把 index cache 记成
`sparseIndexHeads · sparseIndexDim = 4·128 = 512`——用的是 **query 侧** index 头数。但 SGLang
`MiniMaxSparseKVPool`（`memory_pool.py:5433-5461`）的 index_kv/index_k 池均 **`head_num=1`**（单头共享 K，
V 仅当 `disable_index_value=0` 时再存一份），且本文件 874-881 行早已按单头建模 index_k/index_v，line 940
自相矛盾。已改为 `indexElements = indexKeyProjection + indexValueProjection`（= idx_head_dim + (disable?0:idx_head_dim)），
本配置稀疏层 disable=1 → **128**，与 SGLang 逐点一致。修正仅影响 MiniMax-M3/M3-MXFP8 两模型的 spec 树。

## 回归

- `node --test`：**410/410 pass**（含 ops-spec-tree diff、W2/W5 capacity↔indexRead 不变量、declaration 边集）。
- 重生成 `ops-spec-tree.golden.json` / `ops-edge.golden.json`（仅 MiniMax-M3 两模型 hash 变）。
- `verify:models` **60/60，failed=0**；`docs:check`（operators/models/cost）全一致。

## 边界

- glm5_next 的 DSA 稀疏**前向** kernel = SM90a/SM100f（与 deepseek_v32 同），A100(SM80) 只能到 cache 口径；
  minimax_m3 块稀疏前向亦待真机。本项收口的是三类 cache 的**口径**（scheduler 分配所用同一段代码），已 0.0%。
- fp8 索引的 +4B/128 尺度、MXFP8 权重打包等 dtype 细节属边际字节，不改元素口径。

复现：`msv_predict_glm5_mm3.mjs`（前端）+ 本文档内 `Glm5NextTextConfig` / `KimiLinearStateShape.create`
与 `MiniMaxSparseKVPool` 代码路径（SGLang 口径）。
