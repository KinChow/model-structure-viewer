# DeepSeek-V4.1-Flash 减层真实模块树对账（per_model_reconcile.md 第 60 模型收口）

环境：`.venv`(Py3.12) + torch 2.14.0+cu130 + tilelang 0.1.8（`apache-tvm-ffi` 从 0.1.14.post0 降到
0.1.8.post2 后可 import，见 §环境）；8×A100-80GB / CUDA 13.0。本项**不需完整权重**：用官方参考栈
`inference/model.py` 的减层小模型（随机权重）取真实 nn.Module 树与 KV buffer shape。

## 1. 参考栈真值（`scripts/evidence/memory/deepseek_v41_kv_shapes.py` 只读探针，随机权重构造）

自测小模型 `ModelArgs`：`n_layers=5`、`compress_ratios=(0,2,2,1,1,0)`、`kv_source_layers=(1,3)`、
`index_source_layers=(1,3)`。`named_buffers()` 实测（由 `scripts/evidence/memory/deepseek_v41_kv_shapes.py` 重生）：

| 层 | window_kv_cache | compress_kv_cache | indexer.k_cache | 角色 |
|---|---|---|---|---|
| 0 (r0) | [4,128,128] | — | — | 滑窗 |
| 1 (r2) | [4,128,128] | **[4,2048,128]** | **[4,2048,64]** | kv+index source |
| 2 (r2) | [4,128,128] | — | — | **Reuse（0 常驻）** |
| 3 (r1) | [4,128,128] | **[4,4096,128]** | **[4,4096,64]** | kv+index source |
| 4 (r1) | [4,128,128] | — | — | **Reuse（0 常驻）** |

- `layers_owning_compress_kv_cache=[1,3]==kv_source_layers`、`layers_owning_index_k_cache=[1,3]==index_source_layers`
  → `csa2_kv_share_ok=true`、`indexer_source_ok=true`。
- 证实参考实现语义（`model.py@618` 注释）：**`compress_ratio>0` 不代表该层压缩自有 KV，只有 kv_source
  层压缩、其余读同一 cache**；压缩 KV latent 宽度 = 单份 `head_dim`（非 K/V 分开），index k_cache 宽度 =
  `index_head_dim`；均只在 source 层常驻。

## 2. 前端 `assembleDeepseekV41` 减层对账（同 config：5 层/ratios(0,2,2,1,1)/source=[1,3]）

逐层调用 `deepseekV4AttentionOperatorSpecs`（head_dim=512、index_head_dim=128）：

| 层 | ratio | 前端 compressor | 真值 | 前端 indexer | 真值 | resident_kv | resident_idx |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 否 | 否 ✓ | 否 | 否 ✓ | 512 | 0 |
| 1 | 2 | **是** | 是 ✓ | **是** | 是 ✓ | 1536 | 128 |
| 2 | 2 | 否 | 否 ✓ | 否 | 否 ✓ | **512** | **0** |
| 3 | 1 | 是 | 是 ✓ | 是 | 是 ✓ | 512 | 0 |
| 4 | 1 | 否 | 否 ✓ | 否 | 否 ✓ | 512 | 0 |

- **compressor / indexer 层位：前端 == 参考栈，5/5 逐层一致**（source-only）。
- **CSA2 跨层 KV 共享（本次修复核心）**：ratio=2 的 Reuse 层（layer 2）resident_kv 从 1536→**512**（仅滑窗）、
  resident_idx→**0**，与"复用 source、0 常驻"真值一致。折叠签名 `:kvsrc/:kvreuse/:idxsrc` 使 source 层与
  Reuse 层不再折叠合并（否则以 source 值 ×range 计，抹平门控）。

## 3. 残留 caliber（登记，非本次收口）

- **ratio=1 source 层的常驻压缩 KV 未计**：layer 3（真实 layer 20，ratio=1 的 kv/index source）前端因
  `isSparse=ratio>1` 落在滑窗分支，resident_kv=512（缺 source 自有压缩 KV `head_dim/1` + index）。
  compressor/indexer **层位**已对（按 source 集合 emit），但滑窗分支不计其常驻压缩字节 → 该层偏低。
  正确建模需把 V4.1 的 ratio=1 归入压缩/稀疏路径（改动算子类型，需另立并重生 golden）。
- **绝对每 token 字节口径**：见 `../memory/deepseek_v41_csa2_kv_bytes.md`。

## 环境

`apache-tvm-ffi 0.1.14.post0` 与 tilelang 0.1.8 的 py3.12 反射注册冲突
（`AttributeError: attribute '__dict__' of 'type' objects is not writable`）；降到 `0.1.8.post2`
（tilelang 0.1.8 声明 `~=0.1.0`）后 import 通过。构造模型/读 buffer 不触发 kernel，故 shape 真值不受
kernel 环境边界影响（后者见 `deepseek_v41_engram_dspark.md`）。

## 全模块树对账（A100 静态收尾 #3）

探针 `scripts/evidence/structure/deepseek_v41_module_tree.py` 构造减层小模型（默认 `ModelArgs`，随机权重，无 GEMM），dump 完整
`named_modules/named_children`（由 `scripts/evidence/structure/deepseek_v41_module_tree.py` 重生）：

- **顶层**：`embed:ParallelEmbedding` / `layers:ModuleList` / `norm:RMSNorm` / `head:ParallelHead` /
  `mtp:ModuleList` —— 与前端 `assembleDeepseekV41`（embedding + decoder layers + final norm + head + MTP/投机头）
  顶层结构一致。
- **模块类直方图**：`Attention`×5、`MoE`×5、`Gate`×5、`Expert`×45、`Compressor`×2、`Indexer`×2、`Block`×5、
  `ColumnParallelLinear`/`RowParallelLinear`/`RMSNorm`/`Linear` 若干 —— 与前端节点类型（dsv4 注意力链 +
  routed MoE(gate/dispatch/experts/combine) + shared expert + compressor + indexer）逐类对应。
- **每 backbone 层子模块**：`[attn, attn_norm, ffn, ffn_norm]`（`hc_*` 是 Parameter 非子模块，故不在
  named_children，但 `layers.N.hc_attn_base/...` 张量在 checkpoint 已实证——见
  `deepseek_v41_tensor_identity.md`，对应前端 MHC 建模）。

**engram / DSpark / vision 结构**：默认 `ModelArgs` 关闭这三者（`engram_layer_ids=()`、`vision_n_layers=0`、
`dspark_target_layer_ids=()`）；显式开启 engram 需真实 tokenizer（`NgramHashState→build_compressed_token_map`），
不在纯构造路径内。故这三者的结构对账走 **#1 checkpoint 名+shape**（权威、已覆盖）：
- **engram**：`layers.{1,14}.engram.{embed,q_weight,k_weight,wkv}`（embed=fp8 [384006168,256]）。
- **DSpark**：`mtp.N.{markov_head.embed/head, confidence_head.proj, main_proj}`（投机头在 MTP 层内）。
- **vision**：`vision.blocks.0..31.{norm1,attn.wqkv/wo,norm2,...}` + `vision.patch_embed` + `aligner`（32 层）。

**结论（#3）**：backbone 模块树（Attention/MoE/Gate/Expert/Compressor/Indexer/Block/norms/MHC）与前端节点
类型逐类一致；engram/DSpark/vision 结构经 #1 checkpoint 名+shape 全覆盖对账、逐点一致。注意力之外无结构缺口。
