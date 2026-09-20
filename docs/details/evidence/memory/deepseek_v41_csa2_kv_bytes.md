# CSA2 跨层 KV 共享字节口径对账

## 权威真值（官方 `assets/dsv41_kv_cache.png`）

"Global KV Cache Per Token (Bytes)"：DeepSeek-V1 = 389,120；V3.2 = 48,068；**V4-Flash = 3,514**；
**V4.1-Flash = 890**（较 V4-Flash **3.9× smaller**）。这是**每 token 边际增长**口径（随上下文线性增长的部分）。

## 前端修复前后（`kvBytesPerToken`，元素数口径）

前端 `residentMemoryFromGraph` 汇总各叶 `cache_kv_elements + cache_index_elements` × 折叠 multiplier：

| 模型 | 修复前 elem/token | 修复后 elem/token | 真值 B/token |
|---|---|---|---|
| V3.2 | 42,944 | 42,944（不变） | 48,068 |
| V4-Flash | 35,616 | 35,616（不变） | 3,514 |
| **V4.1-Flash** | **41,216** | **23,936** | **890** |

- 修复前 V4.1（41,216）**大于** V4-Flash（35,616）——方向错误，根因是 Reuse 层重复计入其复用的压缩 KV/index。
- 修复后（folding 区分 source/reuse + 逐层 emitCompressor/emitIndexer 门控）：Reuse 层压缩 KV/index 归 0，
  V4.1 降到 23,936、**小于 V4-Flash**（方向修正）；**V3.2 / V4-Flash / 其余 59 模型逐字节不变**
  （ops-spec-tree golden 仅 DeepSeek-V4.1-Flash 一个 hash 变化）。

## 已收口

**CSA2 跨层 KV 共享的结构性过计数**：Reuse 层不再重复计入 source 层的压缩 KV/index。这正是官方 3.9×
（V4-Flash→V4.1）缩减的结构成因——前端 KV 模型现已体现"source 层常驻、Reuse 层复用"。逐层层位与参考栈
真值一致（`../structure/deepseek_v41_module_tree.md`）。

## 未收口（登记为口径专项，需模型级改动，超"仅 V4.1 变"边界）

绝对每 token **字节**对齐 890 需两处口径改造，且会影响全部模型的 KV 上报，故不在本次（"仅 V4.1 golden 变"）内：

1. **边际 vs 常驻口径**：官方 890 是边际增长（滑窗为 `O(1)`、window_size×head_dim 有界，不随 token 增长）。
   前端当前把每层滑窗按每 token 常驻宽度（head_dim）计入，故窗口项主导（40 层×512），无法体现 3.9× 边际比。
2. **逐 dtype 字节**：压缩 KV = fp4(per-16, E4M3 scale)、index key = fp4(per-32, E8M0)、滑窗 K = fp8；
   前端 `residentMemoryFromGraph` 用统一 `kvBytes`（默认 bf16=2B）。需按叶 dtype 分别计字节。

此外 ratio=1 source 层（真实 layer 20）常驻压缩 KV 未计（滑窗分支），见 `../structure/deepseek_v41_module_tree.md §3`。
三项合并为后续"V4.1 KV 边际字节口径"专项；本次先收口结构性跨层共享（明确的过计数错误）。

## 追加：A1+A2「边际 + 逐 dtype」专项落地与双锚点校准（spec dsv4-kv-marginal-bytes）

**A2（ratio=1 归类）**：`isSparse` 改 `ratio>0&&!=128`（全 catalog 实测仅 V4.1 含 ratio=1）→ V4.1 的
ratio=1 层归入 sparse_mla。逐层实证（真实 config）：layer 2/8（r2 source）growthKv=256/growthIdx=64；
**layer 20（r1 kv+idx source）growthKv=512（head_dim/1，Full 模式全长压缩 KV，此前 swa 计 0）/growthIdx=128**；
layer 24/36（r1 仅 idx source）growthKv=0/growthIdx=128；layer 39（r1 reuse）全 0——dangling compressor/indexer 消除。

**A1（边际 + 逐 dtype）**：叶新增 `cache_kv_growth_elements`/`cache_index_growth_elements` + `cache_kv_dtype`
（边际口径：排除有界滑窗，随 token 增长的压缩 KV `head_dim/ratio`（仅 kv_source）+ index `index_head_dim/ratio`
（仅 index_source）；逐 dtype V4.1=F4(0.5B)/V4-Flash·Pro=F8(1B)）。全驻留 `cache_kv_elements` 保留供 W5 对账。
`residentMemoryFromGraph` 带 dtype 的叶用 growth×dtype，其余回退统一 kvBytes。

**双锚点校准**（对官方 `assets/dsv41_kv_cache.png`，`kvBytesPerToken`）：

| 模型 | 本实现 | 官方真值 | 残差 |
|---|---|---|---|
| DeepSeek-V4-Flash | **3,440 B** | 3,514 | −2.1% |
| DeepSeek-V4.1-Flash | **1,056 B** | 890 | +18.7% |
| DeepSeek-V4-Pro | 4,924 B | （无公开值） | — |

- V4-Flash 落在 2% 内，强证明"边际 + 逐 dtype"框架方向正确。
- **V4.1 残差 +18.7%**：F4 按 0.5 B/elem（未含 scale 摊销；含 E4M3/16、E8M0/32 会更高，方向相反）。剩余
  gap 归因于静态无法精确定的项：压缩 latent 的 rope 尾是否高精存储、Full 模式 index 是否全长计、官方图取整。
  按项目纪律**如实登记残差**、不强凑（强行命中 890 会破坏 V4-Flash 的 2% 吻合或引入无依据 fudge）。
- **非 dsv4（无 cache_kv_dtype）逐字节不变**（如 GLM-5=109,824B=54,912×2）。golden 变更**仅 dsv4 家族 6 个模型**
  （normalize hash + ops-spec-tree/edge），非 dsv4 不变；`node --test` 410/410、`verify:models` 60/60、
  W5 KV 读量恒等式（capacity↔kvRead）保持、`docs:check` 全绿。

## 追加：V4.1 残差静态收口 → 精确命中 890（A100 静态收尾 #1+#2）

逐张量对账（`../structure/deepseek_v41_tensor_identity.md`）+ 参考栈 dtype 精读，把 +18.7% 拆净并**静态收敛**：

1. **index 键缓存仅在 4 层**（不是 8）：checkpoint `indexer.wk`/`k_cache` 只在 owns_k=kv_source∩index_source
   =[2,8,14,20]；24/28/32/36 有 index 查询但复用共享键、不自带 k_cache。前端此前按全 8 个 index_source 计 index
   常驻 → 多算 4 层。修正：index 常驻/growth 门控 `emitCompressor && emitIndexer`（V4：ratio===4，行为不变）。
2. **fp4 含 scale 摊销**：压缩 KV = fp4+E4M3/16 = **0.5625 B/elem**（`F4_E4M3S16`）、index = fp4+E8M0/32 =
   **0.53125 B/elem**（`F4_E8M0S32`）。

**重算 V4.1**（head_dim=512、index_head_dim=128）：
- 压缩 KV（[2,8,14,20]）：512/2×3 + 512/1 = 1280 elem × 0.5625 = **720 B**
- index 键（[2,8,14,20]）：128/2×3 + 128 = 320 elem × 0.53125 = **170 B**
- 合计 = **890 B**

| 模型 | 收口后 | 官方真值 | 残差 |
|---|---|---|---|
| DeepSeek-V4.1-Flash | **890 B** | 890 | **0.0%** |
| DeepSeek-V4-Flash | 3,440 B | 3,514 | −2.1%（未变） |

- V4.1 **精确命中 890**（逐张量 + dtype 实证驱动，非 fudge）。
- V4-Flash 仍 −2.1%：其 fp8 cache 的 scale 摊销 / index 键层子集无本地 checkpoint 可核（只 V4.1 权重在本地），
  如实留残差、不强凑（未改 V4 口径）。
- **blast radius 收窄到仅 V4.1**：本轮 golden 仅 DeepSeek-V4.1-Flash 变（index 常驻层集合 + fp4 dtype 属性）；
  V4-Flash/Pro 等其余 dsv4 及非 dsv4 逐字节不变；`node --test` 410/410、`verify:models` 60/60、W5 恒等式保持、
  `docs:check` 全绿。
