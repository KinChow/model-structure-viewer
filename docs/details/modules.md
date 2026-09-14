# 实现细节：模块、关系与公式

> 最后对齐：2026-09-10（P8/P4/P3/P6 后）。

本文是当前代码的模块级实现文档。它回答四个问题：模块如何组合、算子如何连接、公式从哪里来、成本分析如何从结构树得到。系统级边界见 [`../architecture.md`](../architecture.md)，模型来源见 [`models.md`](models.md)。

## 1. 统一对象模型

MSV 有三种相互转换但职责不同的对象：

```text
原始输入
  config.json
  checkpoint truth: tensor name / dtype / shape / parameter count
        |
        v
规范化配置 + 架构解析结果
  normalized config
  resolved canonical architecture
        |
        v
结构 IR
  network -> module -> operator
  formula metadata / diagnostics / source status
        |
        v
Materialized ModelStructure
  summary + source + graph + extra_config
        |
        +--> React Flow diagram / Layers / Inspector
        +--> JSON / Mermaid / DOT / SVG export
        +--> memory / MACs / parallel / communication / roofline
```

`ModelStructure` 的结构载荷只有 `graph`（Graph IR）：P8 起 legacy `root` 树字段已停产（modelStructure.js:88-127，`root_id` 仅作为 graph 协议契约字段保留）。树形视图（Layers/Inspector 的父子层级）不持有独立数据，由 `graph/selectors.js` 的 `graphViewNode` 按 `parent_id`/`order` 按需重建（selectors.js:16-25）。

边界规则：

- `config` 只提供模型字段，不直接生成 UI。
- `registry` 只负责识别 canonical architecture，不负责成本计算。
- `models` 负责顶层组网（查找键 = `config.architectures[0]`）。
- `layers` 负责可复用模块和执行顺序。
- `operators/ops` 负责算子节点、shape flow、公式 ID 和参考实现名。
- `operators/formulas` 负责公式文本、输入输出和解释，不执行计算。
- `IR` 负责稳定的中间协议。
- `materializer` 只负责把模板网络物化为 Graph IR（`materializeStructureGraph`）；checkpoint truth 合并在 `structure/truth/graphTruth.js`：`enrichGraphWithTruth = bindTruthToGraph（路径绑定）+ appendGraphGaps（缺口补挂）`，materializer 在物化后调用它（modelStructure.js:77-84、graphTruth.js:210-274）。
- `cost` 从 materialized structure 和 normalized config 计算理论成本。
- `diagram` 只负责把结构转换成图并响应交互。

## 2. 结构生成调用关系

### 2.1 配置路径

代码入口：`frontend/src/structure/buildStructure.js`。

```text
buildStructureFromConfig(config, options)
  -> normalizeConfig(config)
  -> resolveArchitecture(normalized, options)
  -> buildNetwork(resolved, normalized)
  -> createStructureIr({ network, normalized, resolved, options })
  -> materializeModelStructure(ir)
```

### 2.2 带 checkpoint truth 的路径

```text
config + model/source metadata + checkpointTruth
  -> buildStructureFromArtifacts(artifacts)
  -> buildStructureFromConfig(config, truth options)
  -> buildNetwork：模板 network（无模板 = unsupported 空网络）
  -> createStructureIr -> materializeModelStructure
  -> materializeStructureGraph：模板 Graph IR
  -> enrichGraphWithTruth(graph, truth)：bindTruthToGraph + appendGraphGaps
  -> materialized ModelStructure（graph 为唯一结构载荷）
```

checkpoint truth 的获取和结构骨架分成两件事：

1. `cost/weights.js` 优先调用 `@huggingface/hub` 的 `parseSafetensorsMetadata` 计算参数量和 dtype 分布，失败时使用 `safetensorsReader.js` 读取 header。
2. `structure/truth/skeleton.js` 根据 tensor name 建 trie，得到含参模块树；`structure/truth/graphTruth.js（mergeSemantics 已于 W3-D 删除）` 再把它和模板语义对齐。

因此模板不是含参模块的唯一来源：模板负责无参算子、执行顺序、语义标签和公式；truth 负责 tensor、shape、dtype 和参数量。

## 3. Registry 与 canonical architecture

相关代码：

- `frontend/src/structure/config/normalize.js`
- `frontend/src/structure/registry/resolveArchitecture.js`
- `frontend/src/structure/models/index.js`

解析关系：

```text
architectures[0]
  -> MODELS 精确表
  -> 该架构的组装函数（内部决定有无 vision）
```

查找键的单源是 `models/index.js` 的 `MODELS`（对标 vLLM `_TEXT_GENERATION_MODELS`、SGLang `_ModelRegistry.models`）。`hasModelArchitecture` 供 materializer 判定。未命中 → `unsupported`，空网络 + 诊断枚举 `SUPPORTED_MODEL_ARCHITECTURES`（vLLM `ModelRegistry._raise_for_unsupported`）。

新增模型：registry 一行 + 一份组装函数；参数差异复用已有函数；顶层接线不同才新写组装；局部 attention / MoE 不同则扩展 layer；只有现有 IR 无法表达时才改协议。

## 4. 顶层网络关系

顶层 builder 由 `models/index.js` 的 `MODELS` 分派（对标 vLLM `_TEXT_GENERATION_MODELS`：一 `architectures[0]` 一模块）。投机头写在该架构文件里（`deepseek_v4.js` 挂 MTP/DSpark，`qwen3_5.js` / `qwen4_exp.js` 挂各自 MTP，其余引用 `deepseek_mtp.js` 的 SharedHead / eh_proj），位置在 decoder 之后、final norm / HC mixer 之前。对标 vLLM：`models/<arch>/mtp.py`、`dspark.py` 跟主干同包，不是共享 dispatcher。树 id 仍是 `mtp`（checkpoint）。零件名 = 权重/成员名。decoder kind 写在该类组网里（V4 MTP/DSpark 强制 SWA；Qwen3.5 强制 full attention；Qwen4Exp 强制 QSA 且关 PLE），不抄主干最后一层。`repeat: 0` 不算每次前向；参数仍占显存。一份模板 × N 写 `modules=N`；已展开 stage（DSpark `mtp.0/1/2`）写 `modules=1`，`residentRepeat` 不再乘。

### 4.1 Dense/GQA decoder

```text
model
  ├── embed_tokens
  ├── decoder
  │   └── decoder layers [dense]
  │       ├── attention [GQA]
  │       └── mlp
  ├── optional MTP
  ├── optional output_attn_residual
  ├── final norm
  └── lm_head
```

实现：`models/common.js`、`models/qwen3_5.js`、`layers/decoderStack.js`、`layers/decoderLayer.js`。

### 4.2 MoE decoder

```text
decoder layer
  ├── attention
  └── routed MoE
      ├── router
      ├── routed experts
      │   ├── gate projection
      │   ├── up projection
      │   ├── activation
      │   └── down projection
      ├── optional shared expert
      └── optional shared expert gate / branch add
```

实现：`layers/moe.js`。专家路径在成本计算中使用 `expertsPerToken / experts` 的活跃比例；专家权重在 EP 投影中另行按平均/最坏区间处理。

fused shared expert 的判定单源（P3）：shared expert 是否为"融合"形态的判定权归 `structure/archs/index.js:37` 的 `ARCH_RECIPES.sharedExpertsAreFused` 配方（当前仅 KimiK3 登记），`config/normalize.js` 只归一字段，不再持有第二份 `model_type` 子串判定（normalize.js:155-162）。取证（docs/details/sharding_matrix.md 方案更正节）：K3 checkpoint 每个 MoE 层只有 `shared_experts.{gate,up,down}_proj.weight` 各一个（92 层 × 3 = 276 张量，k3-index.json），`modeling_kimi_linear.py:797-801` 先把 intermediate_size 放大 `num_shared_experts` 倍再实例化**单个** MLP——"融合" = 单个更宽的 MLP，不是打包张量，也不涉及 ep 亲和（shared expert 唯一分片语义 = ÷tp，parallel_protocol Q5）。因此模板保持三叶形态（out = 模块宽）与 checkpoint 1:1 对应，无需独立 operator_id，也无需 ep+tp 双组声明。

### 4.3 MLA decoder

```text
decoder layer
  ├── MLA attention
  │   ├── query down projection -> query latent norm -> query up projection
  │   ├── KV compression projection -> latent/rope split -> latent norm
  │   ├── rotary position path
  │   ├── attention score/context or sparse index path
  │   └── output projection
  └── MoE or dense MLP
```

实现：`models/deepseek_v3.js`、`layers/attention.js`、`ops/index.js` 的 MLA/DSA 分支。MLA 的 KV cache 不是普通的 K/V 两份 head 张量，而是 `kv_lora_rank + qk_rope_head_dim` 的 latent/rotary 组合；这是内存和 TP 投影的关键分支。

### 4.4 MiniMax multimodal sparse decoder

```text
model
  ├── vision tower
  ├── multimodal projector
  ├── text decoder
  │   └── layers
  │       ├── dense or sparse attention
  │       │   ├── fused main/index QKV projection
  │       │   ├── main/index split
  │       │   ├── Q/K norm + RoPE
  │       │   ├── block indexer [sparse layer]
  │       │   └── block-sparse GQA [sparse layer]
  │       └── dense or routed MoE
  ├── optional MTP
  └── lm_head
```

实现：`models/minimax_m3.js`、`layers/vision.js`、`layers/projector.js`、`ops/index.js` 的 MiniMax attention 分支。

### 4.5 Multimodal/hybrid decoder

```text
model
  ├── vision tower
  ├── projector
  ├── embed_tokens
  ├── decoder stack
  │   ├── GQA / linear attention / QSA / hybrid attention
  │   ├── MLP or MoE
  │   └── optional hyper connection / PLE / attention residual
  ├── optional final hyper connection mixer
  ├── optional output attention residual
  ├── optional MTP
  ├── final norm
  └── lm_head
```

实现：`models/qwen4_exp.js`、`layers/hybrid.js`、`layers/residual.js`、`layers/vision.js`。

### 4.6 Generic fallback

无 builder 时（canonical architecture = `unsupported`，`ARCHITECTURE_CATALOG` 中 hasBuilder=false）MSV 不再伪造结构：`buildNetwork` 返回只含根节点的空网络，让管线走完、诊断可达；`collectDiagnostics` 产出 `unsupported-architecture` 诊断并枚举 `SUPPORTED_MODEL_ARCHITECTURES` 支持项（vLLM `_raise_for_unsupported` 模式；models/index.js:53-62、`diagnostics/collectDiagnostics.js:22-25`），前端以 banner 告警。

checkpoint truth 在场时（`truth.skeleton` 离线骨架文件形态或 `truth.tensors` 形态），`enrichGraphWithTruth` 的无 builder 分支直接以 checkpoint 骨架图作为结构：`skeletonTruthGraph` 把含参模块树转成 Graph IR，root 重写为模型名，strategy = `skeleton-truth` / `skeleton-truth-file`（graphTruth.js:216-229,249-261）；有 builder 时走 template+truth 绑定与缺口补挂（strategy = `template+truth` / `template+truth-file`）。fallback 不声称拥有架构特有的执行语义和公式，来源状态必须通过 `strategy` 和 `diagnostics` 暴露。

## 5. Layer 分类与关系

| 分类 | 实现文件 | 上游 | 下游 | 主要公式/状态 |
|---|---|---|---|---|
| Embedding | `layers/embedding.js` | token IDs | hidden state | `linear` |
| Decoder stack | `layers/decoderStack.js` | embedding | repeated decoder layers | `repeat`、layer range |
| Decoder layer | `layers/decoderLayer.js` | hidden state | residual hidden state | attention + MLP/MoE |
| Attention | `layers/attention.js`、`ops/index.js` | hidden state | attention output | GQA、MLA、QSA、KDA、RoPE |
| MLP | `layers/mlp.js` | attention residual | layer residual | Linear、SwiGLU |
| MoE | `layers/moe.js` | layer hidden state | routed/shared output | TopK、dispatch/combine |
| Normalization | `layers/norm.js` | hidden state | normalized state | RMSNorm/Gemma RMSNorm |
| Vision tower | `layers/vision.js` | image/video input | visual features | patch embedding、attention、norm |
| Projector | `layers/projector.js` | visual features | text hidden width | Linear/projector activation |
| Residual | `layers/residual.js` | residual streams | mixed hidden state | attention residual |
| Hybrid | `layers/hybrid.js` | multi-stream hidden state | mixed/contracted state | mHC、PLE、Hyper Connection |
| MTP | 各架构文件（`deepseek_mtp.js` / `deepseek_v4.js` / `qwen3_5.js` / `qwen4_exp.js`） | 主干最后一层 hidden | SharedHead.norm / hc_head / mixer | 对标 vLLM `models/<arch>/mtp.py`（repeat=0） |
| DSpark | `models/deepseek_v4.js` | 主干 `dspark_target_layer_ids` hidden | hc_head hidden + Markov bias | 对标 vLLM `models/deepseek_v4/nvidia/dspark.py`（repeat=0） |
| LM head | `layers/outputHead.js` | final hidden state | logits | Linear |

普通 decoder layer 的**同级**逻辑关系（原则 §2.5，对标 Gallery）：

```text
layer_in → PreNorm → Attention → ⊕ → PreNorm → MLP/MoE → ⊕
    └──── skip ─────┘              └──── skip ────┘
```

`attn_residual_add` / `ffn_residual_add` 是与 Attention / FFN 并列的兄弟节点（`layers/decoderLayer.js`，公式 `residual_add`）。skip 的源是残差主干，不是子层输出；展开 Attention / FFN 看不到 skip。**现状代码只有顺序 residual_add、尚无 skip 边**（实现债）。特殊层由 `layerSchedule`、`attentionSchedule`、`hyperConnectionCount` 和 `attnResBlockSize` 决定。

结构树中的 `repeat` 用于压缩同构层。成本遍历必须传递 repeat 乘数，但子节点已有显式 repeat 时不能再次相乘。图、Layers、Inspector、公式和成本都使用同一 `root.0.1...` 节点路径。

## 6. Operator 关系

`operators/ops/index.js` 的 `operatorSpec` 是算子统一入口：

```text
operatorSpec(id, name, operatorId, attributes, numericShapes)
  -> formulaForOperator(operatorId)
  -> formula / explanation / inputs / outputs
  -> input_shape / output_shape
  -> optional implementation / model-specific attributes
```

| 字段 | 作用 |
|---|---|
| `operator_id` | 连接公式元数据和成本逻辑；UI 公式索引同一键 |
| `input_shape` / `output_shape` | 数值 shape，供成本和 Inspector 使用 |
| `attributes` | 可读 shape、attention kind、split sizes 等 |
| `implementation` | vLLM/SGLang 参考实现名，不在浏览器执行 |
| `source_fields` | 参与生成的配置字段 |

### 6.1 Operator spec dispatch

`layers/attention.js` 根据 `attentionKind` 选择下列 operator spec；同一层的 `layerIndex` 用于读取按层变化的 schedule。

| Dispatcher | 触发条件 | 主要 operator chain |
|---|---|---|
| `attentionOperatorSpecs` | 普通 GQA/attention | q/k/v -> RoPE -> score -> softmax -> context -> o |
| `linearAttentionOperatorSpecs` | `attentionKind=linear`（通用槽位） | qkv/gate/decay projection -> short conv -> state update -> output gate |
| `canonicalKdaOperatorSpecs` | `attentionKind=linear` 且 `linearAttentionMode` ∈ {kimi_k3, kimi, glm5_next, qwen4_exp, qwen3_5}，在 `linearAttentionOperatorSpecs` 内分派（ops/index.js:244-251,267） | 融合 qkvg(a/b/f) 投影 -> short conv -> KDA state update（`gated_delta_attention`）-> output gate norm -> o；KDA 是同一语义结构，框架融合方式只记录在投影属性 |
| `qwen35FullAttentionOperatorSpecs` | `qwen35_full` | fused qkvz -> q/k norm -> RoPE -> score/context -> output gate -> o |
| `mlaAttentionOperatorSpecs` | `attentionKind=mla` | q compression + KV compression -> split -> norm/RoPE -> MLA score/context -> o |
| `qsaAttentionOperatorSpecs` | `attentionKind=qsa`（Qwen 逐头 QSA） | qkv/norm -> indexer -> selected sparse attention -> o |
| `dsaAttentionOperatorSpecs` | `attentionKind=qsa` 且 modelType ∈ {deepseek_v32, glm_moe_dsa, glm5_next}（`qsaAttentionOperatorSpecs` 内分派，ops/index.js:669,864；模块级 `attention_kind` 对齐为 `dsa_sparse_mla`，attention.js:118） | q/kv compression -> indexer（`dsa_indexer`/`dsa_kpool_indexer`）-> sparse MLA -> o |
| `deepseekV4AttentionOperatorSpecs` | `attentionKind=dsv4` | DSV4 compressor/indexer/SWA or compressed MLA -> output projection |
| `minimaxDenseAttentionOperatorSpecs` | MiniMax dense layer | fused QKV -> norm/RoPE -> dense score/context -> o |
| `minimaxSparseAttentionOperatorSpecs` | MiniMax sparse layer | fused main/index QKV -> split/norm/RoPE -> block indexer -> sparse GQA -> o |
| `minimaxM2AttentionOperatorSpecs` | MiniMax M2/GLM4 variant | fused QKV -> norm/RoPE -> dense score/context -> o |
| `mlpOperatorSpecs` | dense MLP | gate projection + up projection -> SwiGLU -> down projection |
| `moeOperatorSpecs` | 普通 MoE | router -> topk -> dispatch -> expert MLP -> combine |
| `deepseekV4MoeOperatorSpecs` | DeepSeek V4 MoE | hash route 或 router/topk -> dispatch -> expert -> combine |
| `kimiK3MoeOperatorSpecs` | Kimi K3 MoE | router/topk -> latent down -> dispatch -> latent expert -> combine/norm/up |

operator chain 只是可解释的结构语义，不表示 MSV 会调用 vLLM/SGLang kernel。具体实现参考通过 operator 的 `implementation` 属性保存。

## 7. 完整公式目录

公式元数据的代码来源是 `frontend/src/structure/operators/formulas/index.js`（`FORMULAS` 注册表，在用 49 个键）。表中公式按 Markdown 做了必要的排版归一，但含义和公式 ID 与代码一致；它们是当前页面展示的架构语义，不等同于某个 runtime kernel 的源码。

### 7.1 基础算子

| ID | 公式 | 关系 |
|---|---|---|
| `linear` | `Y = XW^T + b` | q/k/v、MLP、output/projector |
| `matmul` | `Y = A B` | attention score/context |
| `softmax` | `softmax(x_i) = exp(x_i) / sum_j exp(x_j)` | score -> probability |
| `split` | `[q,k,v,b,f_a,g_a] = split(z; 3P,H,D,D)` | fused projection split |
| `causal_conv1d` | `x'_t = SiLU(Conv1D(x_{t-w+1:t}; w))` | linear attention history |
| `rope` | `q', k' = rotate(q, k, position)` | positional encoding |
| `rmsnorm` | `y = x / sqrt(mean(x^2) + eps) * weight` | standard norm |
| `gemma_rmsnorm` | `y = x / sqrt(mean(x^2) + eps) * (1 + weight)` | Gemma-style norm |
| `swiglu` | `y = SiLU(xW_gate) * (xW_up)` | gated MLP |
| `residual_add` | `h = x + sublayer(x)` | decoder 层 attention/FFN 后的残差加（formulas/index.js:188） |

### 7.2 MoE 和 linear attention

| ID | 公式 | 关系 |
|---|---|---|
| `topk` | `experts = topk(router_logits, k)` | routed expert selection |
| `moe_dispatch` | `x_e = dispatch(x, expert_ids)` | token -> expert inputs |
| `moe_combine` | `y = sum_e weight_e * expert_e(x_e)` | expert outputs -> hidden |
| `fused_moe_mlp` | `y = (SiLU(x W_gate^T) ⊙ (x W_up^T)) W_down^T` | routed expert 融合前馈，gate/up/down 三段 GEMM 单叶计数（对标 vLLM FusedMoE 打包 w13/w2；formulas/index.js:175） |
| `moe_add` | `y = y_routed + y_shared` | shared/routed merge |
| `linear_attention` | `S_t = decay_t*S_{t-1} + k_t^T v_t; y_t = q_t S_t` | generic recurrent attention |
| `linear_attention_gate` | `y_t = gate(z_t) * y_t` | gated recurrent output |
| `gated_delta_attention` | `beta=sigmoid(beta_raw); v'=beta(v-S_prev k); S=exp(g)S_prev+v'k^T; o=S q` | canonical KDA |
| `gated_rmsnorm` | `y = RMSNorm(o, weight) * phi(g_2)` | KDA output norm/gate |

### 7.3 Multi-stream、MLA 和 QSA

| ID | 公式 | 关系 |
|---|---|---|
| `mhc_pre` | `p=sigmoid(M_a s_a+b_a)+eps; C=Sinkhorn(softmax(M_c s_c+b_c)+eps); x=sum_i p_i H_i` | mHC input mixing |
| `mhc_fused_post_pre` | `(H',post',C',x') = MHCPre(MHCPost(x,H,post,C); F,scale,base)` | fused layer boundary |
| `mhc_post` | `H'_j = post_j*x + sum_i C_ij H_i` | mHC output injection |
| `mhc_contract` | `h = (1/n) * sum_i H_i` | stream contraction |
| `mla_query_compress` | `c_q=W_qa x; q=W_qb RMSNorm(c_q)` | MLA query latent |
| `mla_kv_compress` | `[c_KV,k_R] = W_kv x` | MLA KV latent |
| `mla_kv_split` | `[c_KV,k_R] = split(z; kv_lora_rank, rope_dim)` | MLA cache split |
| `mla_output_gate` | `O' = sigmoid(W_g x) * O` | gated MLA output |
| `attention_residual` | `s_i=<RMSNorm(x_i),w>; p=softmax(s); y=RMSNorm(sum_i p_i x_i)` | residual bank selection |
| `hyper_connection` | `x_n=GroupedRMSNorm(H); gate=W_up SiLU(W_down x_n); H'=Combine(H,block_output,injection)` | multi-stream connection |
| `ple` | `[k,v]=W_kv e; y=ShortConv(GatedNorm(k,v,RMSNorm(H)))` | PLE inject（ngram 表是 sibling Embedding） |
| `shared_expert_gate` | `y = y_routed + sigmoid(W_g x) * y_shared` | gated shared expert |
| `qsa_indexer` | `I = topk((W_q x)(W_k K)^T/sqrt(d_i), budget)` | sparse token index |
| `qsa_sparse_attention` | `O = softmax(Q K_I^T/sqrt(d)) V_I` | selected sparse attention |
| `qwen_qkvz_split` | `[q,k,v,z] = split(W_qkvz x; q,k,v,z)` | Qwen GDN split |

### 7.4 Sparse、DSA/DSV4 和特殊注意力

| ID | 公式 | 关系 |
|---|---|---|
| `attention_qkv_split` | `[q,k,v] = split(W_qkv x; q,k,v)` | MiniMax/GLM fused QKV |
| `attention_output_gate` | `O' = sigmoid(G) * O` | full-attention output gate |
| `minimax_sparse_indexer` | `B = topk_blocks(score_type(Q_i K_i^T/sqrt(d_i)), k)` | block selection |
| `minimax_sparse_attention` | `O = softmax(Q K_B^T/sqrt(d)) V_B` | selected block GQA |
| `dsa_indexer` | `s_t = sum_h w_h ReLU(q_h k_s)/sqrt(d_i); I = topk(s_t, index_topk)` | DeepSeek/GLM DSA lightning indexer（formulas/index.js:421） |
| `dsa_kpool_indexer` | `s_t = sum_h w_h ReLU(q_h pool(k))/sqrt(d_i); I = topk(s_t, index_topk/kpool)*kpool + tail` | GLM-5.3-Flash k-pool 变体（formulas/index.js:436） |
| `dsv4_indexer` | `s_t = sum_h w_h ReLU(q_h k^c_s)/sqrt(d_i); I = topk(s_t, index_topk)` | DeepSeek V4 C4 压缩 latent 打分（formulas/index.js:451） |
| `dsa_sparse_mla` | `O = softmax(Q W_kv_b C_I^T/sqrt(d)) C_I W_v_b` | DSA 主注意力：选中位置上的吸收式 MLA（formulas/index.js:476） |
| `dsv4_sparse_mla` | `O = softmax(Q [K^c_I ; K_{t-w:t}]^T/sqrt(d)) [V^c_I ; V_{t-w:t}]` | DSV4 C4 压缩 + 滑窗混合读（formulas/index.js:490） |
| `dsv4_hash_route` | `expert_ids = hash_table[input_ids]` | hash MoE route |
| `dsv4_swa_attention` | `O = softmax(Q K_{t-w:t}^T/sqrt(d)) V_{t-w:t}` | sliding-window MQA |
| `dsv4_compressed_attention` | `O = softmax(Q C_KV^T/sqrt(d)) C_KV` | compressed MLA |
| `dsv4_output_projection` | `O_{hidden} = W_{o_b}(W_{o_a}(RoPE^{-1}(O)))` | DSV4 low-rank output |

### 7.5 逐公式来源映射

来源等级不是置信度高低，而是说明公式怎样进入 MSV：

| 来源 | 定义 | 代码证据 |
|---|---|---|
| S1 标准算子 | 通用数学或 Transformer 算子定义，由 MSV 维护展示形式 | `formulas/index.js` + 通用 operator spec |
| S2 参考实现语义 | 根据当前 vLLM/SGLang 模型实现链抽象为架构语义 | `ops/index.js` 的模型分支和 `implementation` 字段 |
| S3 MSV 组合关系 | MSV 为图和解释层定义的分支组合，不对应单一 runtime kernel | layer/builder 组合关系 + `formulas/index.js` |

| 公式 ID | 来源 | 说明 |
|---|---|---|
| `linear`、`matmul`、`softmax`、`residual_add` | S1 | 通用线性代数、归一化和残差加定义（residual_add 为一等 aten 锚点） |
| `rope`、`rmsnorm`、`gemma_rmsnorm`、`swiglu` | S1 | 标准 Transformer 位置、归一化和激活语义 |
| `topk`、`moe_dispatch`、`moe_combine` | S1 | 通用稀疏 MoE 路由关系 |
| `split`、`causal_conv1d`、`linear_attention` | S2 | GLM/Qwen/Kimi linear-attention 投影、卷积和状态链 |
| `gated_delta_attention`、`gated_rmsnorm` | S2 | Qwen/GLM/Kimi 共用的 canonical KDA 语义 |
| `fused_moe_mlp` | S2 | vLLM FusedMoE（w13/w2 打包）与 SGLang fused_moe 专家内核 |
| `mhc_pre`、`mhc_fused_post_pre`、`mhc_post`、`mhc_contract` | S2 | vLLM mHC pre/post/contract 实现关系 |
| `mla_query_compress`、`mla_kv_compress`、`mla_kv_split`、`mla_output_gate` | S2 | vLLM/SGLang MLA projection/cache 路径 |
| `attention_residual`、`hyper_connection`、`ple` | S2 | Kimi/Qwen/GLM 多流和位置增强实现关系 |
| `qsa_indexer`、`dsa_indexer`、`dsa_kpool_indexer`、`dsv4_indexer` | S2 | QSA/DSA/DSV4 indexer（打分选 token/块，无 value 无 softmax） |
| `qsa_sparse_attention`、`dsa_sparse_mla`、`dsv4_sparse_mla` | S2 | QSA/DSA/DSV4 主注意力（选中位置上的 GQA/吸收式 MLA） |
| `qwen_qkvz_split`、`attention_qkv_split`、`attention_output_gate` | S2 | fused 投影和 full-attention gate 实现关系 |
| `minimax_sparse_indexer`、`minimax_sparse_attention` | S2 | MiniMax block indexer 和 sparse backend 语义 |
| `dsv4_hash_route`、`dsv4_swa_attention`、`dsv4_compressed_attention`、`dsv4_output_projection` | S2 | DeepSeek V4 hash MoE、SWA、compressed MLA 路径 |
| `moe_add`、`linear_attention_gate`、`shared_expert_gate` | S3 | MSV 对分支合并和门控关系的显式建模 |

## 8. 公式来源和可信度

| 来源类型 | 当前代码位置 | 含义 |
|---|---|---|
| MSV 语义公式 | `structure/operators/formulas/index.js` | 页面展示公式，由 MSV 按结构语义维护 |
| 参考实现 | `operators/ops/index.js` 的 `implementation` | vLLM/SGLang 类、算子或 backend 名称，不在浏览器执行 |
| 成本方法论 | `cost/*.js` 注释和函数 | 逐算子 counts 对标 FlopCounterMode / onnx-tool（shape → 公式）。容量 walk 图（§3.8）。并行除法规则见 `details/parallel_protocol.md`（GQA `min(TP, kv_heads)`、MLA 不切、DP-attention 复制）。**不再参考 llm-analysis**。FlopCounterMode 独立算子夹具在 `verification/flop_counter.py`（Linear / BMM / 深度可分 Conv1d）。 |
| shape/协议事实 | safetensors、`@huggingface/hub`、模型 config | 参数量、dtype、shape、cache 和模型识别事实 |

外部参考：

- [PyTorch FlopCounterMode](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)：逐算子 shape → FLOP 公式（mm/bmm/conv/SDPA）；msv counts 的机制对标。
- [onnx-tool](https://github.com/ThanatosShinji/onnx-tool)：ONNX shape inference + 每节点 MACs。
- [vLLM](https://github.com/vllm-project/vllm)：模型执行、KDA state、attention/MoE 算子和实现名称参考。
- [SGLang](https://github.com/sgl-project/sglang)：模型执行、attention/MoE/backend 实现名称参考。
- [safetensors](https://github.com/huggingface/safetensors)：header、dtype、shape、offset 和 tensor key 事实来源。
- [@huggingface/hub](https://github.com/huggingface/huggingface.js)：浏览器端 safetensors metadata 和参数量解析实现。

限制：`formulas/index.js` 当前没有逐公式的外部 URL 字段。上述外部来源是方法论或参考实现来源，不表示每条公式都逐行复制自外部函数。新增公式时应同步记录来源类别。**不再参考 llm-analysis**（闭式容量、per-GPU 函数、效率因子出处均不引用）。

## 9. 成本公式关系

```text
materialized graph + normalized config + load + plan + chip
  -> shape 沿 dataflow 传播（运行时维 B/S；静态维不动）
  -> 节点公式只吃本节点 in/out/weight shape → counts
  -> 计费主语（原则 §2.4）：父有 counts 用父，否则用叶子；禁止双计
  -> computeNodeCosts / memoryBreakdown
  -> projectPlan / projectPdFit
  -> planCommunicationBytes
  -> classifyRoofline
  -> CostSummary / node Cost Lens
```

### 9.1 权重字节

```text
weight_bytes = sum(parameter_count[dtype] * bytes_per_dtype(dtype))
node_weight_bytes = sum(product(shape) * bytes_per_dtype(tensor_dtype))
```

代码：`cost/weights.js`、`cost/memory.js`、`cost/aggregate.js`。无 checkpoint 时 walk 图声明（`graphWeightCapacity`）。聚合结果携带 `weightSource`：`checkpoint` / `node`（图声明汇总）/ `derived-quantized`（图声明 + 量化）/ `what-if` / `empty`。

### 9.2 KV cache 和 request state

```text
GQA KV/token = 2 * layers * kv_heads * head_dim * bytes
MLA KV/token = layers * (kv_lora_rank + qk_rope_head_dim) * bytes
KV          = batch * tokens * KV/token

KDA state/layer = conv_history_elements
                + value_heads * value_dim * key_dim
```

线性 attention 的 recurrent state 是 request state，不是 token KV。代码：`cost/memory.js`。KV/KDA **容量** walk 叶声明（`cache_*_elements` / `state_elements`）；除法规则见 §9.4。KDA state shape 对标 vLLM `MambaStateShapeCalculator.kda_state_shape`。

### 9.3 MACs/FLOPs

```text
MACs_linear = tokens * input_width * output_width * expert_fraction
FLOPs       = 2 * MACs
tokens      = batch * sequence  # prefill
tokens      = batch             # decode

MACs_attention = query_tokens * heads * context_tokens
                 * (query_key_dim + value_dim)
```

QSA 使用 `min(sequence, indexer_budget)`；MiniMax sparse 使用 `(topk + init + local) * block_size`；linear attention 按各模型投影、卷积、状态更新和输出投影分解。公式唯一来源是 `structure/operators/formulas/`——`formulas/extractor.js` 的 `countsForNode` 查 `FORMULAS` counts 注册表；`cost/compute.js` 只做 Graph IR 遍历（`walkStructure`）和 repeat 倍乘，旧 nodeMacs 分派链已于 W5-1 删除。

### 9.4 并行投影

```text
GQA KV shard = min(TP, kv_heads)
MLA KV shard = 1
DP-attention KV shard = 1
```

并行计划的归一化与校验单源是 `cost/parallelPlan.js`（`normalizeParallelPlan`，协议 Q8）：`world_size == tp*pp*dp`；`ep <= experts`、`moe_ep <= experts` 且 experts 能被 moe_ep 整除；EP 启用时专家域闭合 `moe_ep * moe_tp == ep * tp`（TRT-LLM Hybrid ETP）；无 EP 时 `moe_tp == tp`；`attn_mode` 只能是 tp 或 dp（parallelPlan.js:21-75）。

权重在单卡上的归属唯一入口是叶 `attributes.weightMatrices` 声明（schema v2：`class / shape / count / matrices / split / quantizable / param_dtype`，ops/index.js:66-100）：norm scale/bias = replicated 且不参与量化；lm_head/embedding = vocab 轴（受 vocab_parallel 开关支配）；router = replicated；gate_up/qkv 沿 output 切、down/o_proj 沿 input 切（与 vLLM `MergedColumnParallelLinear`/`RowParallelLinear` 一一对应）；fused_moe_mlp = ep 组。`cost/parallel.js` 的路径正则规则表已删除（P5）：无声明带权叶返回 `axis: "unknown"`——诚实缺项、不再按路径猜归属，覆盖率护栏保证内置模型不触达（parallel.js:64-83）。norm/embedding/lm_head/router 的复制与 vocab 出处都由声明 class 表达，不再是 parallel 侧的正则判定。routed expert 权重按 EP 切分并给出平均/最坏区间；PP 只负责 stage 归属。代码：`cost/parallel.js`、`cost/sharding.js`、`cost/parallelPlan.js`。

### 9.5 通信量

```text
TP all-reduce = operations * 2 * (TP-1)/TP * B * T * H * bytes
EP all-to-all = operations * B * T * experts_per_token * H * bytes
PP boundary   = max(PP-1, 0) * B * T * H * bytes
```

无 EP 且 `dp > 1` 时 dispatch/combine 仍触发 all-to-all（DP-shards-experts，协议 Q6+Q4；EP 启用时走 ep 分支；无 moe_dp 轴，用 dp 近似，口径标注近似；comm.js:58-66）。KV 侧 `kv_keep_ratio`（decode 侧实际驻留 KV 比例，协议 Q7③，缺省 1）在 `parallelPlan.js` 归一校验（:32-35,47）、`kvBytesPerCard` 按比例折减（parallel.js:40-42）。PD transfer 使用 decode rank 数和两侧较小链路带宽，并给出闭式 `transferSeconds = (per-rank KV + state 字节) / 链路带宽`（协议 Q7②；comm.js:98-102）；布局不同只标记重排。代码：`cost/comm.js`。不建模 overlap、调度和真实拥塞。

### 9.6 Roofline

```text
t_compute = 2 * MACs / (peak_flops * eta_flops)
t_memory  = bytes_moved / (memory_bandwidth * eta_hbm)
t_comm    = comm_bytes / (link_bandwidth * eta_comm)
bound     = max(matrix, vector, sfu, memory, comm)   # 五路时间取 max（W5-2）
```

bound 是五路时间（matrix/vector/sfu/memory/comm）取 max 的 **overlap 静态上限**：comm 与 compute 取最大而非求和是闭式不等式口径，结果带 `overlapUpperBound: true` 标记（P10，roofline.js:119-125）。费率单源在 `cost/chips/rates.js`（`chipRates`），效率因子来自 `cost/efficiency.js`。动作向量声明了 `computeDtype` 时（N2-1，如 mHC 的 TF32 pre-GEMM），矩阵时间拆两段费率：tf32 桶按 `peak_flops.tf32` 计、其余按全局 dtype 费率；芯片无 tf32 行时整段回退全局费率（roofline.js:19-21,84-95）。数量未知或费率缺失时对应路为 null，bound 返回 `unknown`——已知零参与 max 但不主导，未知不得冒充零。

### 9.7 显存：总量、每卡峰值、节点 VRAM 三套数字，禁止混比

主语仍是图（§9 数据流）。Fit 不另开公式，只读投影。

```text
# 产品层 CostSummary（放得下吗）
Total VRAM = memoryBreakdown.totalBytes
           = 未分片（weights + 图 buffer + KV + KDA）
             # 图能证明的驻留下界。activation workspace / CUDA runtime /
             # comm scratch 无法从 config 得到，不计（runtime-unknown）

stage.totalBytes = 该 PP stage 已切权重 + 已切 KV + 已切 KDA

Peak / card = max_s stage.totalBytes          # projectPlan 与 projectPdFit 同式
Fit / card  = Peak / card <= chip.memory_bytes
              # 拓扑（requiredGpus <= totalGpus）走 planStatus，不进 Fit
              # plan 无效 → Fit 未知，不得谎报显存不足
              # 结论是图内驻留下界，不是「含框架开销刚好能跑」

# 节点层 Cost Lens（算子可解释）
ownWeightPerCard     = nodeCostPerCard(nodeWeightCapacityBytes × resident)
                       # 只切本节点 weightMatrices；无声明则 unknown/0
subtreeWeightPerCard = aggregate_weightBytes   # 先切后上卷，复用 aggregateNodeCosts
nodeActivation       = 本节点 dataflow 边界
                       （counts.bytes.actIn+actOut，否则 input/output_shape）
vramBytes            = subtreeWeightPerCard + nodeActivation
```

纪律：

- 不提供 Activation peak / Runtime / Comm buffer 旋钮，也不把它们默认成 0 再加进 Fit。真实峰值显存是 runtime-unknown。
- 节点激活禁止 Σ 子节点。容器展示的是子树驻留权重 + 本节点边界，不是子树执行峰值。
- 切分是叶的事。禁止对已切分的 `aggregate_weightBytes` 再送进 `nodeCostPerCard`（会 `/TP²`）。
- 算力倍率用 `multiplier`，容量用 `resident`（MTP/DSpark `repeat=0` 仍占显存）。
- 权重入口唯一：`nodeWeightCapacityBytes`（checkpoint shape 优先，否则 `weightMatrices`，`shared` 跳过）。`projectNodePlan` 与 `computeNodeCosts` 都调它，禁止写死 `*2`。
- `projectPlan` 给每个 stage 写 `totalBytes`（抄 `projectPdFit` 已有那一行）。CostSummary 不内联加法，不新增 `peakStageMemoryBytes`。
- 不单开「节点总显存」投影。芯片行已有单卡容量；拓扑行已有 nodes × GPUs。

测试锁：叶子 `vramBytes` 只按声明轴切一次；父 = Σ 子每卡权重 + 自身边界激活；Fit 读 `stage.totalBytes` 不读 `cost.memory.totalBytes`；MTP `repeat=0` 权重非 0。

## 10. UI、导出和诊断关系

```text
ModelStructure.graph（Graph IR 唯一结构载荷；App.jsx:96、DetailWorkspace.jsx:54,138）
  ├── SummaryChips      -> 模型摘要、来源、状态
  ├── ArchitectureTab   -> React Flow、公式索引、Cost Lens
  ├── StructureSearchBox + React Flow -> 展开、折叠、选择和数据流
  ├── NodeDetailPanel   -> truth、公式、shape、属性、node lens
  ├── ExportTab         -> Mermaid / DOT / JSON
  └── RawConfigTab      -> extra_config
```

树形视图（Inspector 顶层模块、选中节点的父子层级）没有独立载荷，由 `graph/selectors.js` 的 `graphViewNode` 按 `parent_id`/`order` 按需重建（selectors.js:16-25）。

节点路径是 Architecture、公式按钮、Layers、搜索、Inspector 和 node Cost Lens 的唯一联动键。模块 builder 可通过 `attributes.dataflow_edges` 声明稳定的子算子数据流；`diagnostics` 必须说明结构来源、truth 状态、fallback 和 repair；UI 不能只根据是否有结构树判断验证成功。

## 11. 测试对应关系

| 实现层 | 主要测试 |
|---|---|
| config/registry/builder/layer | `structure/modelArchitecture.test.js`、`structure/builtinModels.test.js` |
| truth 骨架/绑定/graph merge | `structure/truth/__tests__/`：`graphTruth`、`roleBinding`、`skeleton`、`skeletonTruthFile`、`truthDiagnosticsSeam`（`mergeSemantics.test.js` 已随 W3-D graph 化退役，不存在） |
| 算子树/权重声明/unsupported | `structure/operators/__tests__/`：`declaration`、`ops-spec-tree.diff`、`unsupportedArchitecture` |
| 公式与 counts 恒等 | `structure/operators/formulas/__tests__/`：`atoms`、`counts`、`identities`、`modelIdentities`、`extractor.identity`、`countsAtomsConsistency` |
| shape/dtype/safetensors | `cost/__tests__/dims.test.js`、`safetensorsReader.test.js` |
| 成本链（counts/computeDtype/量化/字节/roofline/声明接缝） | `cost/__tests__/`：`compute`、`computeDtype`、`quantBytes`、`sharding`、`bytesCompleteness`、`rooflineChain`、`memory`、`aggregateWeights`（aggregate 接缝）、`traverse` |
| TP/PP/EP/DP/PD | `cost/__tests__/parallel.test.js`、`comm.test.js`、`pdSummary.test.js` |
| 芯片来源和缺项 | `cost/__tests__/publicChips.test.js`、`coverage.test.js`、`manualChip.test.js` |
| 图和交互 | `diagram/__tests__/edgeStyle.test.js`、`components/ModelEntry.test.js`、`hooks/useStructure.test.js`、`diagnostics.test.js` |
| 导出 | `frontend/src/exporters.test.js` |

新增或修改公式至少要同时更新公式元数据、operator spec、成本函数（如果参与成本）、单元测试和本文公式/来源表。

## 12. 修改规则

1. 配置字段变化改 `normalize.js`，不把厂商判断散落到 UI。
2. 官方架构名称变化改 `models/index.js` 的 `MODELS` 表。
3. 顶层拓扑变化改 `models/`。
4. 可复用模块变化改 `layers/`；最小算子语义改 `operators/ops/` 和 `operators/formulas/`。
5. 参数和 shape 真值改 truth/skeleton/merge 路径，不在模板中复制 checkpoint 事实。
6. 成本公式改 `cost/`，同时更新来源注释和成本单测。
7. IR 字段改动必须同步 materializer、API schema、导出、UI 和测试。
8. `implementation` 字段只用于解释和参考映射，不是运行时依赖。
