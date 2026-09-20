# 修复 V4.1 CSA2 注意力被误判为 V4 的 SWA

## 问题（联网 + config + 代码 + 实测 确认）

- **CSA vs CSA2 不同机制**（联网）：V4 = Compressed Sparse Attention(CSA) + HCA 逐层混合；
  V4.1-Flash = **CSA2**，每层静态 Full/Reindex/Reuse 模式、跨层共享/复用 KV 与稀疏索引（KV ~890 B/token）。
- **config**：V4.1 `text_config.compress_ratios=[0,0,2,2,…]`（ratio=2），有 `kv_source_layer_ids=[2,8,14,20]`/
  `index_source_layer_ids=[2,8,14,20,24,28,32,36]`/`index_n_heads=32`；V4 是 ratio 0/4/128、无 source_layer_ids、index_n_heads=64。
- **代码 bug**：`deepseek_v41.js` 逐字复用 `assembleDeepseekV4`；`ops/index.js` 注意力类型硬编码
  `ratio===4→sparse / ===128→HCA / else→SWA`。V4.1 的 ratio=2 落 else → **41/43 层被误判为滑窗 MQA**，
  且 source 层 emit 的 indexer 悬挂（注意力不消费稀疏选择）。
- **实测**（修复前）：真实 V4.1 前端注意力算子全是 `dsv4_swa_attention`，无 sparse/compressed。

## 修复

`ops/index.js:deepseekV4AttentionOperatorSpecs` 引入 `isSparse = ratio > 1 && ratio !== 128`
（压缩稀疏 MLA：V4 CSA ratio=4 + V4.1 CSA2 ratio=2 都是"压缩 KV + indexer 选择"），替换硬编码 `ratio===4`：
- 注意力类型：`if (isSparse) → dsv4_sparse_mla`（消费 indexer 的 top-k 选择），`===128 → 压缩稠密`，`else → 滑窗`。
- compressor 宽度 / sparse cacheResident 的 `(ratio===4?2:1)` → `(isSparse?2:1)`。

## 验证

- V4.1：`dsv4_sparse_mla` 从 0 → 出现（ratio=2 层修正为稀疏）；V4：算子分布**逐项不变**（sparse_mla 21 /
  compressed 20 / swa 2）。
- ops-spec-tree golden：**仅 DeepSeek-V4.1-Flash 一个 hash 变化**，其余 59 模型不变。
- 回归全绿：node --test 410、verify:models 60/60、docs:check（golden/reference 已随意图变更重生）。

## 仍未收口（本轮边界，需 CSA2 规范/框架）

- **CSA2 跨层 KV 共享（Full/Reindex/Reuse）**：Reuse 层复用 source 层的压缩 KV，前端仍按**每层**计
  resident KV → V4.1 KV cache 字节偏高（未体现 890 B/token 的跨层复用）。
- **ratio=2 压缩 KV 精确宽度**：现按 isSparse 与 V4 CSA 同形（保守），准确口径需 V4.1 CSA2 modeling。
- 因 `deepseek_v41` 无框架支持，无法整模型对真值——本修复只纠正**注意力类型分类**（明确的错），
  caliber 细节待真实 checkpoint/框架。

## 附：CSA2 跨层 KV 共享建模尝试（本轮尝试→回退，记录原因）

尝试把"压缩 KV 仅 kv_source 层常驻、Reuse 层 resident=0"做进 sparse_mla 的 `cacheResidentDecl`（按
`kvSourceLayerIds.includes(layerIndex)` 门控）。**回退**，原因是**与层折叠冲突**：

- `compactRanges` 按结构签名折叠连续层，把 kv_source 层（如 layer 2，owns KV=1536）与其后的 Reuse 层
  （3–13，应 0）折成**一个 range、以 source 层为代表 ×range_size**（实测 `rep layer=2 mult=12 kvEl=1536`）。
- 于是逐层门控被折叠"抹平"——resident KV = source 层值 × 整段层数，而非"每段只算 source 层一次"。
- 结果：V4.1 每 token KV 元素几乎不降（31232 vs V4 32928），达不到 ~890 B/token 的 4× 缩减。

**正确建模需**：① 折叠签名区分 source/reuse 模式（使二者不折在一起），或改成模型级"按 kv_source 层数计一次"
的 KV 口径；② FP4 KV（E2M1，per-16-channel）等 CSA2 精确 caliber。二者都需 CSA2 规范/可运行框架
（`deepseek_v41` 无框架支持，无法对真值验证），故**跨层 KV 共享字节口径仍挂起**（本轮边界）。本轮只保留
已验证的**注意力类型修复**（ratio=2 → sparse_mla）。

## 后续验证前提（是否需要完整权重？）

分三类，**不是所有收口项都需要完整训练权重**——大多数被"框架支持"卡住，而非"权重"：

| 待验证项 | 需完整训练权重？ | 真正的前置依赖 | 说明 |
| --- | --- | --- | --- |
| CSA2 跨层 KV 共享字节口径（890 B/token） | **否** | `deepseek_v41` 框架 + CSA2 实现（真值来源）；前端折叠签名区分 source/reuse | KV 大小由 shape/config 决定，与权重数值无关。减层随机 checkpoint（`from_config` + random init）即可对 shape 真值。 |
| ratio=2 压缩 KV 精确宽度 / FP4 KV caliber | **否** | 同上（框架 + CSA2 规范） | 宽度是 dtype/结构量，随机权重可验。 |
| 前端折叠签名修复（source/reuse 不折在一起） | **否** | 无外部依赖，纯前端 | 可立即做，只受"缺 CSA2 真值无法回归对账"影响——修完仍需框架侧对拍才算收口。 |
| engram / DSpark 运行时（接受率 / 显存 / 吞吐） | **是** | 完整训练权重 + 可运行框架 + GPU | 行为依赖权重数值（草稿接受率等），减层随机权重无意义。 |
| 后端 transformers 构造整模型对账 | **是（或补齐 remote code）** | `deepseek_v41` 框架支持或 remote code | 需能实例化整模型；若只验 shape，减层随机权重可行。 |

**结论**：CSA2 KV 共享/宽度的收口**不缺权重、缺框架**——一旦 `deepseek_v41` 有可运行框架（或补齐 remote code），
用减层随机 checkpoint 即可对 shape/字节真值，无需下载完整权重。仅 engram/DSpark 的**运行时行为**验证才必须完整权重。

## 框架状态更新（2026-09-18，联网确认代码）

**"缺框架"这一前置已解除——vLLM main 与 SGLang 均已合入 `deepseek_v41`。** 逐条核对上游代码：

- **vLLM `registry.py`（main）已含条目**（原文）：
  - `"DeepseekV41ForCausalLM": ("vllm.models.deepseek_v41", "DeepseekV41ForCausalLM")`
  - 旁证同族：`"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", ...)`、`"DeepseekV32ForCausalLM": ("vllm.models.deepseek_v32", ...)`、
    `"DeepseekV3ForCausalLM": ("deepseek_v2", ...)`
  - engram/DSpark 草稿头：`"DSparkV41DraftModel": ("vllm.models.deepseek_v41", "DSparkDeepseekV4ForCausalLM")`（V4 对应 `DSparkDraftModel`）。
- **keystone PR [vllm#56214] `[Model] Support DeepSeek-V4.1-Flash` 已 merge**（`ywang96 merged 8 commits into main from dsv41-feat`）——
  加 registry 条目 + config + tokenizer 接线；模型定义 PR #56228 已 merge；稀疏 indexer PR #56254（DeepGEMM sparse MQA logits → V4.1 indexer）、
  注意力 megakernel #56344 等 kernel 在 tracker #56217。追踪 issue #56400 明确 **`DeepseekV41ForCausalLM` 与 V4 是不同架构**
  （独立 tree / tokenizer / parser / config class）——与本轮"V4.1 不可逐字复用 V4 组网"的判断一致。
  （注：issue #56400 正文仍写"in review"，晚于 PR 页 merged 状态，以 PR 页为准。）
- **SGLang**：官方 X 公告"DeepSeek V4.1 Flash weights are out… day-0 inference and RL support in SGLang and Miles. V4.1 extends the V4 stack with compressed KV shared…"，
  changelog 有 DSV4 DSpark/缓存管理相关合入；静态文档页尚滞后（仍列到 V3.1/R1）。

**对本项目的影响**：

1. 上游"V4.1 是独立架构、不复用 V4"直接印证前端 bug——`deepseek_v41.js` 目前 `export { assembleDeepseekV4 as assembleDeepseekV41 }`
   正是上游告诫的反模式；本轮 `isSparse`（ratio=2 → `dsv4_sparse_mla`）修复方向正确，但**逐字复用整条组网仍应拆开**（后续项）。
2. #56254 "sparse MQA logits → indexer" 佐证 V4.1 走**稀疏 indexer 喂稀疏 MLA**，与修复后的 `dsv4_sparse_mla` 分类一致。
3. **CSA2 跨层 KV 共享字节口径（890 B/token）验证现已解锁**：框架存在 → 用减层随机 checkpoint 实例化 `DeepseekV41ForCausalLM`，
   dump KV cache 结构/字节即可对真值，**无需完整权重**。上表"真正的前置依赖=框架"已满足，转为**可执行**。
4. engram/DSpark 运行时（`DSparkV41DraftModel`）仍需**完整权重 + GPU**（行为依赖数值）；SGLang 称权重已放出，如需可下载后验。
