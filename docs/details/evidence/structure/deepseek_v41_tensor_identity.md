# DeepSeek-V4.1-Flash 逐张量权重恒等式对账（A100 静态收尾 #1）

真值源：`$MODELS/DeepSeek/DeepSeek-V4.1-Flash/model.safetensors.index.json`（96,085 张量，
`total_size=510,286,023,000`）。纯 JSON name 级归类，无需 GPU。

## name 级组件层位（逐张量归类 → 层集合）

| 组件（张量名模式） | checkpoint 层集合 | config/前端判据 | 一致性 |
|---|---|---|---|
| `layers.N.attn.compressor.wkv` | **[2,8,14,20]** | `kv_source_layer_ids` | ✓ |
| `layers.N.attn.compressor.wgate` | **[2,8,14]** | 仅 ratio>1（layer 20 ratio=1 无 gate，见 Compressor@444-448） | ✓ |
| `layers.N.attn.indexer.wq_b` / `weights_proj`（index 查询） | **[2,8,14,20,24,28,32,36]** | `index_source_layer_ids`（8 层） | ✓ |
| `layers.N.attn.indexer.wk` / `k_norm`（index **键**，owns k_cache） | **[2,8,14,20]**（仅 4 层！） | = `kv_source`（owns_k） | **差异见下** |
| `layers.N.engram.*` | **[1,14]** | `engram_layer_ids` | ✓ |
| `layers.N.ffn.experts.*` | 全 40 层各 **384** routed | `n_routed_experts=384` | ✓ |
| `layers.N.ffn.shared_experts.*` | 全 40 层 | `n_shared_experts=1` | ✓ |
| 层数 | 40 主干 + 3 MTP + 32 vision | config | ✓ |

## 关键发现：index **键缓存**仅在 4 层（≠ 8 个 index_source）

- checkpoint：**`indexer.wk`（索引键投影 + `k_cache`）只在 [2,8,14,20]**（= kv_source∩index_source），
  而 `indexer.wq_b`（索引查询）在全 8 个 index_source 层。即：**24/28/32/36 层有索引查询但复用共享键、
  不自带 index k_cache**。参考栈 `Indexer.owns_k` 与 kv_source 同址（键由压缩 KV latent 派生）。
- **对前端 KV 边际字节的影响**：`dsv4-kv-marginal-bytes` 把 index 常驻按全 8 个 index_source 计（含
  24/28/36/32），**多算 4 层**——正是 V4.1 +18.7% 残差的主因之一。修正：index 常驻/growth 应门控在
  **owns_k = kv_source∩index_source**（`emitCompressor && emitIndexer`），仅 [2,8,14,20]。
- 其余 name 层位与前端声明**逐组件一致**（compressor/indexer 查询/engram/experts/MTP/vision）。

## 结论

- V4.1 逐张量 name 层位与结构声明对账：**除 index 键缓存层集合（前端多计 24/28/32/36）外，全部一致**。
- 该差异转 #2（KV 残差静态收紧）修正；shape 级对账见下节 / `§shape`。

## §shape 逐张量 shape 级对账（读分片头部，代表性张量 vs config）

config：dim=5120、head_dim=512、q_lora=1280、o_lora=1024、o_groups=8、n_heads=64、moe_int=2304、
index_head_dim=128、index_n_heads=32、vocab=129280、engram head_dim=256。

| 张量 | checkpoint dtype/shape | config 期望 | ✓ |
|---|---|---|---|
| `attn.wq_a.weight` | F8_E4M3 [1280,5120] | [q_lora, dim] | ✓ |
| `attn.wq_b.weight` | F8_E4M3 [32768,1280] | [n_heads·head_dim, q_lora] | ✓ |
| `attn.wkv.weight` | F8_E4M3 [512,5120] | [head_dim, dim] | ✓ |
| `attn.wo_a.weight` | F8_E4M3 [8192,4096] | [o_groups·o_lora, n_heads·head_dim/o_groups] | ✓ |
| `attn.wo_b.weight` | F8_E4M3 [5120,8192] | [dim, o_groups·o_lora] | ✓ |
| `compressor.wkv.weight` | **BF16** [512,5120] | [head_dim, dim]（权重 bf16，非 fp8） | ✓ |
| `indexer.wk.weight` | **BF16** [128,512] | [index_head_dim, head_dim] | ✓ |
| `indexer.wq_b.weight` | F8_E4M3 [4096,1280] | [index_n_heads·index_head_dim, q_lora] | ✓ |
| `ffn.experts.0.w1/w3` | **I8** [2304,2560] | [moe_int, dim/2]（fp4 packed 2/byte） | ✓ |
| `ffn.experts.0.w2` | **I8** [5120,1152] | [dim, moe_int/2] | ✓ |
| `ffn.shared_experts.w1` | **F8_E4M3** [2304,5120] | [moe_int, dim]（shared=fp8，非 fp4） | ✓ |
| `engram.embed.weight` | F8_E4M3 [384006168,256] | [engram_num_embeddings[0], engram_head_dim] | ✓ |
| `embed/head.weight` | BF16 [129280,5120] | [vocab, dim] | ✓ |

**补充发现（dtype 分布）**：dense 注意力投影 = fp8；**compressor.wkv / indexer.wk 权重 = bf16**；
**routed experts = fp4（I8 packed，dim/2）**、**shared expert = fp8**；engram embed = fp8 巨表。
代表性 shape 与 config **逐点一致**（含 fp4/fp8 packing）。**KV cache dtype**（≠权重 dtype）由 `_compress_kv`/
`_window_kv` 决定：压缩 KV/index = fp4、滑窗 = fp8——与 `dsv4-kv-marginal-bytes` 的 F4/F8 建模一致。
