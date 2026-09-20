# DeepSeek-V4-Flash 逐张量结构对账 + KV 残差归因（A100 纯 I/O）

从 BOS 下 V4-Flash 的 `config.json` + `model.safetensors.index.json`（69,187 张量，纯 I/O，无需 fp8 前向）核其
KV 组成，试收 `dsv4-kv-marginal-bytes` 留的 V4-Flash −2.1%（3,440 vs 官方 3,514）。config：`deepseek_v4` bf16 +
fp8(weight_block 128)、43 层、head_dim=512、q_lora_rank=1024、n_routed_experts=256、**无 kv/index_source_layer_ids**
（用 ratio 0/4/128 启发式）、ratio 分布 {0:3, 4:21, 128:20}。

## 逐张量结构（name 级）

| 组件 | checkpoint 层集合 | 前端判据 | 一致性 |
|---|---|---|---|
| `compressor.wkv`（压缩 KV） | **全 41 个 ratio>0 层 [2..42]**（含 ratio=4 与 128） | `emitCompressor=ratio>1` | ✓ 每层自带、无跨层共享（无 source_ids） |
| `indexer.wq_b`（index 查询） | **21 个 ratio==4 层** | `emitIndexer=ratio===4` | ✓ |
| **`indexer.wk`（index 键 cache）** | **空（无该张量）** | —— | V4-Flash indexer **无独立键投影/k_cache**（≠ V4.1 的 owns_k）；键由压缩 latent 派生 |
| `engram` | **空** | —— | ✓ engram 是 V4.1 专属，V4-Flash 无 |

**结论（结构）**：V4-Flash 前端 `emitCompressor=ratio>1`（41 层）、`emitIndexer=ratio===4`（21 层）判据与
checkpoint 逐层一致；V4-Flash 无 source_layer_ids（每压缩层自带 KV）、无独立 index 键 cache、无 engram——均与前端建模吻合。

## KV 字节残差归因（−2.1%）

前端 V4-Flash = 3,440 elem×F8(1B) = 压缩 KV（ratio4: 512/4×21=2688 + ratio128: 512/128×20=80）+ index（128/4×21=672）
= 3,440 B。官方 3,514 → **前端 UNDER 74 B（−2.1%）**。

- 试过的方向不成立：移除 index（因 `indexer.wk` 空）→ 3,440−672=2,768（−21%，远劣），说明 **V4-Flash 的 index 功能上仍占
  KV**（键从压缩 latent 派生、无独立 wk 权重）——前端计入 index 反而更接近真值。
- 74B 缺口方向 = **加**（不是减）：属 **fp8 cache 的 ue8m0 scale 摊销 + rope 尾精度**等小口径量，加 fp8 scale
  （per-128）只补 ~27B、仍差；**精确拆分需 V4-Flash 参考推理栈（本地无，只有 V4.1 的 `inference/`）**。
- **处置（据实、不强凑）**：V4-Flash 结构已逐张量验证无误，−2.1% 是**小 fp8-caliber 残差、非结构错误**，
  在无 V4-Flash 参考栈的前提下**不精确收口、不为 2% 猜测改前端**（会动 V4 家族 golden）。与 V4.1（有参考栈→精确 890）
  形成对照：**能精确的已精确，缺参考栈的据实留残差**。

清理：BOS 下的 config/index 临时文件用后删除；未改任何前端代码/ golden。
