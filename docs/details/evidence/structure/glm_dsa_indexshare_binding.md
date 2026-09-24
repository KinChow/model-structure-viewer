# GLM DSA / IndexShare checkpoint 绑定复核

日期：2026-09-24。Graph IR 保持 v2。

## 结论

GLM-5、GLM-5.1、GLM-5.2、GLM-5.2-FP8、GLM-5.3、GLM-5.3-BF16
以及 GLM-5.3-Flash 两个 BF16/FP8 变体的 DSA indexer 已修正为使用
发布实现中的物理模块路径。此前图中使用 `q_proj` /
`wk_weights_proj` 这样的通用别名；生产 checkpoint/source-ref 使用的是：

```text
self_attn.indexer.wq_b
self_attn.indexer.wk
self_attn.indexer.k_norm
self_attn.indexer.weights_proj
```

这不是显示名称问题。checkpoint truth 按 canonical module path 精确绑定，
通用别名会使真实 `wq_b`、`wk`、`weights_proj` 张量变成 gap，或者在未来
完整 skeleton 加载时无法绑定到对应节点。QSA、DeepSeek-V4 C4Indexer
等使用不同发布模块的路径没有被这次改名影响。

## 外部与发布证据

1. Hugging Face Transformers v5.16.1 的 `GlmMoeDsaIndexer`：
   `wq_b = Linear(q_lora_rank, index_n_heads * index_head_dim)`、
   `wk = Linear(hidden_size, index_head_dim)`、
   `k_norm = LayerNorm(index_head_dim)`、
   `weights_proj = Linear(hidden_size, index_n_heads)`。
   实现还明确：`indexer_types[layer_idx] == "shared"` 时不实例化
   indexer，并从前一个 full 层接收 `prev_topk_indices`。
2. GLM-5.2 官方模型卡说明 IndexShare 在每四个 sparse-attention 层中
   复用同一个 indexer；GLM-5.2 config 为 78 层、21 个 `full` 和
   57 个 `shared`。前三层为 full，层 3–5 复用层 2，之后按发布数组
   的 full/source 边界处理，不能机械按显示折叠区间猜源层。
3. 下载的 GLM-5.2 safetensors index map 中，indexer 的五类 tensor 只
   出现在 full 层：0、1、2、6、10、…、74；共享层没有对应 indexer
   权重。GLM-5.3 解析后的 index map 同样只在 full 层出现 indexer
   权重，并有 FP8 的 scale tensor。
4. IndexCache 论文 `arXiv:2603.12201` 的算法描述是 full 层生成
   top-k、shared 层复用最近的 preceding full 层，并且复用的是当前
   index tensor，不新增一份持久 KV cache。实现继续保留每层独立 MLA
   latent/KV；IndexShare 不被误画成 MLA KV 共享。

## 代码修复

- DSA builder 的 canonical ID 改为 `wq_b`、`wk`、`weights_proj`；
  `k_norm` 保持原物理路径。
- attention dataflow 同步改为 `wq_b → indexer`、
  `wk → k_norm → indexer`、`weights_proj → indexer`。
- V4 C4Indexer 的 `indexer.q_proj` / `indexer.weights_proj` 保留，
  因为它对应另一套 SGLang C4Indexer 物理实现，不可用 DSA 的改名规则
  全局替换。
- folded layer 的 `index_source_layer_id`、19 条跨层 `index-reuse`
  关系和 GLM-5.3-Flash 的 k-pool 独立路径均保持不变。

## 测试

`frontend/src/structure/models/dsaIndexerCheckpoint.test.js` 覆盖：

- DeepSeek-V3.2、GLM-5/5.1/5.2/5.3、GLM-5.3-Flash 的
  config-only 与 production-artifacts；
- 物理投影形状、DSA indexer fan-in、计算公式；
- GLM-5.2 的 21 个物理 indexer 与 57 个共享层；
- synthetic tensor/header 与 skeleton 两种绑定路径，每个张量只绑定一次；
- 共享层无 indexer 权重副本，仍保留独立 MLA attention。

本批定向结果：**30/30 通过**。

证据文件、读取时间和 SHA256 见本地
`artifacts/architecture-repair/indexshare-*`；完整权重未下载到仓库，
这些 artifacts 不作为产品运行时依赖。
