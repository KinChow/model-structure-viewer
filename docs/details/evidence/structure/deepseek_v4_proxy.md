# B · deepseek_v4 减层结构对账（CSA2 最近可跑代理）

> V4.1 的 CSA2 本环境无框架（无 `deepseek_v41`）。V4 是 transformers 5.17 原生架构，可减层随机实例化（零下载），
> 且上游 "V4.1 extends the V4 stack"——用 V4 对账前端 dsv4 注意力算子的**逐层类型/indexer 布局**，是 CSA2 之外最接近的真值。
> 脚本：`scripts/evidence/structure/v4_proxy_backend.py`（transformers 真值）、`scripts/evidence/structure/v4_proxy_frontend.mjs`（前端算子）。

## 后端真值（transformers deepseek_v4）

- ratio→layer_type 映射**取自 transformers 自带常量** `_COMPRESS_RATIO_TO_LAYER_TYPE`（非硬编码）：
  `{0: sliding_attention, 4: compressed_sparse_attention(CSA), 128: heavily_compressed_attention(HCA)}`。
- 真实 DeepSeek-V4-Flash（43 主层）layer_types 计数：**sliding=2, CSA=21, HCA=20**。
- 减层随机 `from_config`（6 层 `[0,0,4,128,4,128]`，meta device，零下载）：`cfg.layer_types` == 实例化
  `self_attn.layer_type` == ratio 映射，**逐层一致（ok=True）**——证明模块确按映射构建，非仅配置声明。
- transformers 源码：CSA 层含 Lightning Indexer（top-`index_topk`），**HCA 层无 indexer**，sliding 层为 shared-KV MQA 滑窗。

## 前端算子（DeepSeek-V4-Flash）

`buildStructureFromConfig` → 注意力算子 operator_id × 折叠 multiplier 求和：

| operator_id | Σmultiplier |
|---|---|
| dsv4_sparse_mla | 21 |
| dsv4_compressed_attention | 20 |
| dsv4_swa_attention | 3 |
| dsv4_indexer | 21 |
| dsv4_hash_route | 3（MoE，非注意力） |

## 对账

| 类型 | 前端 | transformers 主层(43) | 判定 |
|---|---|---|---|
| CSA / sparse_mla | dsv4_sparse_mla **21** | CSA **21** | ✓ 逐点一致 |
| HCA / compressed | dsv4_compressed_attention **20** | HCA **20** | ✓ |
| sliding / swa | dsv4_swa_attention **3** | sliding **2** | ✓（差 1 = MTP 层，见下） |
| indexer 布局 | dsv4_indexer **21**（= CSA 层数） | 仅 CSA 有 indexer（HCA 无） | ✓ 逐点一致 |

- **swa 差 1 已解释、非 bug**：V4-Flash `num_nextn_predict_layers=1`，`compress_ratios` 44 项 = 43 主层 + 1 MTP 层，
  `ratios[43]=0`（MTP 为 sliding）。前端把 MTP 建模为额外一层 → swa=2(主)+1(MTP)=3；transformers 主模型只建 43 层
  （MTP 单独处理）→ sliding=2。**主层分布逐点一致**；前端多出的 1 个 swa 正是 MTP 草稿层（ratio=0，与 ce860a MTP 修复
  一致——纯 MLA/滑窗 MTP 不误判为稀疏）。

## 结论与 V4.1 边界（诚实）

- **前端 dsv4 注意力的 ratio→类型映射与 indexer 布局，与 transformers deepseek_v4 真值逐点一致**（含减层实例化确认模块构建）。
- **V4.1 边界**：transformers V4 映射常量只有 `{0,4,128}`，**不含 ratio=2**——V4 无法表示 V4.1 的 CSA2（ratio=2），
  这正是需要独立 `deepseek_v41` 的原因（已确认 vLLM main / SGLang 合入，见 `deepseek_v41_csa2_attention_fix.md` 框架状态更新）。
  前端当前把 ratio=2 归为 `dsv4_sparse_mla`（isSparse，最近代理），**跨层 KV 共享 890 B/token 精确口径仍需 deepseek_v41 框架**
  （本环境无，留换环境/升级框架后验）。
