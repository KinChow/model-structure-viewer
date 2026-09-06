# 实现细节：模块、关系与公式

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
  summary + source + root + extra_config
        |
        +--> React Flow diagram / Layers / Inspector
        +--> JSON / Mermaid / DOT / SVG export
        +--> memory / MACs / parallel / communication / roofline
```

边界规则：

- `config` 只提供模型字段，不直接生成 UI。
- `registry` 只负责识别 canonical architecture，不负责成本计算。
- `model_executor/models` 负责顶层组网。
- `layers` 负责可复用模块和执行顺序。
- `ops` 负责算子节点、shape flow、公式 ID 和参考实现名。
- `formulas` 负责公式文本、输入输出和解释，不执行计算。
- `IR` 负责稳定的中间协议。
- `materializer` 负责模板语义与 checkpoint truth 合并。
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
  -> template network + skeleton trie
  -> enrichNetworkWithTruth
  -> materialized StructureNode
```

checkpoint truth 的获取和结构骨架分成两件事：

1. `cost/weights.js` 优先调用 `@huggingface/hub` 的 `parseSafetensorsMetadata` 计算参数量和 dtype 分布，失败时使用 `safetensorsReader.js` 读取 header。
2. `cost/skeleton.js` 根据 tensor name 建 trie，得到含参模块树；`cost/mergeSemantics.js` 再把它和模板语义对齐。

因此模板不是含参模块的唯一来源：模板负责无参算子、执行顺序、语义标签和公式；truth 负责 tensor、shape、dtype 和参数量。

## 3. Registry 与 canonical architecture

相关代码：

- `frontend/src/structure/config/normalize.js`
- `frontend/src/structure/registry/aliases.js`
- `frontend/src/structure/registry/architectureCatalog.js`
- `frontend/src/structure/registry/resolveArchitecture.js`

解析关系：

```text
model_type / architectures[0] / model id
  -> aliases
  -> canonical architecture
  -> hasTemplate
  -> model builder
```

当前 canonical architecture 与 builder：

| Canonical architecture | 模板 | Builder | 结构特征 |
|---|---:|---|---|
| `gqa-decoder` | 是 | `buildGqaDecoderNetwork` | dense decoder + GQA |
| `gqa-moe-decoder` | 是 | `buildGqaMoeDecoderNetwork` | decoder + routed/shared MoE |
| `mla-moe-decoder` | 是 | `buildMlaMoeDecoderNetwork` | MLA + MoE |
| `multimodal-sparse-moe-decoder` | 是 | `buildMiniMaxM3Network` | vision + projector + sparse text decoder |
| `multimodal-gqa-moe-decoder` | 是 | `buildQwenMultimodalNetwork` | vision + projector + GQA/MoE decoder |
| `hybrid-multimodal-moe-decoder` | 是 | `buildHybridMultimodalNetwork` | multimodal decoder + hybrid connection |
| `generic-decoder` | 否 | `buildGenericDecoderNetwork` | 通用 decoder stack |
| `generic-config` | 否 | `buildGenericConfigNetwork` | 只有配置级根节点 |

新增模型的决策顺序：先补 alias；参数差异复用 canonical architecture；顶层组网不同才新增 builder；局部 attention、MoE 或层序列不同则扩展 layer/ops/formulas；只有现有 IR 无法表达时才修改协议。

## 4. 顶层网络关系

### 4.1 Dense/GQA decoder

```text
model
  ├── embed_tokens
  ├── decoder
  │   └── decoder layers [dense]
  │       ├── attention [GQA]
  │       └── mlp
  ├── optional output_attn_residual
  ├── final norm
  └── lm_head
```

实现：`models/common.js`、`models/qwen.js`、`layers/decoderStack.js`、`layers/decoderLayer.js`。

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

MLA 的 KV cache 不是普通的 K/V 两份 head 张量，而是 `kv_lora_rank + qk_rope_head_dim` 的 latent/rotary 组合；这是内存和 TP 投影的关键分支。

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
  └── lm_head
```

实现：`models/minimax.js`、`layers/vision.js`、`layers/projector.js`、`ops/index.js` 的 MiniMax attention 分支。

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
  ├── final norm
  └── lm_head
```

实现：`models/qwen.js`、`layers/hybrid.js`、`layers/residual.js`、`layers/vision.js`。

### 4.6 Generic fallback

无专用模板时，config 生成 generic decoder 或根节点；checkpoint truth 可通过 skeleton trie 生成含参模块树。generic fallback 不声称拥有架构特有的执行语义和公式，来源状态必须通过 `strategy` 和 `diagnostics` 暴露。

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
| LM head | `layers/outputHead.js` | final hidden state | logits | Linear |

普通 decoder layer 的逻辑关系是 `hidden -> pre-attention norm -> attention -> residual -> pre-MLP norm -> MLP/MoE -> residual -> next layer`。特殊层由 `layerSchedule`、`attentionSchedule`、`hyperConnectionCount` 和 `attnResBlockSize` 决定。

结构树中的 `repeat` 用于压缩同构层。成本遍历必须传递 repeat 乘数，但子节点已有显式 repeat 时不能再次相乘。图、Layers、Inspector、公式和成本都使用同一 `root.0.1...` 节点路径。

## 6. Operator 关系

`model_executor/ops/index.js` 的 `operatorSpec` 是算子统一入口：

```text
operatorSpec(id, name, operatorId, attributes, numericShapes)
  -> formulaForOperator(operatorId)
  -> formula_id / formula / explanation / inputs / outputs
  -> input_shape / output_shape
  -> optional implementation / model-specific attributes
```

| 字段 | 作用 |
|---|---|
| `operator_id` | 连接公式元数据和成本逻辑 |
| `formula_id` | UI 公式索引和节点联动键 |
| `input_shape` / `output_shape` | 数值 shape，供成本和 Inspector 使用 |
| `attributes` | 可读 shape、attention kind、split sizes 等 |
| `implementation` | vLLM/SGLang 参考实现名，不在浏览器执行 |
| `source_fields` | 参与生成的配置字段 |

### 6.1 Operator spec dispatch

`layers/attention.js` 根据 `attentionKind` 选择下列 operator spec；同一层的 `layerIndex` 用于读取按层变化的 schedule。

| Dispatcher | 触发条件 | 主要 operator chain |
|---|---|---|
| `attentionOperatorSpecs` | 普通 GQA/attention | q/k/v -> RoPE -> score -> softmax -> context -> o |
| `linearAttentionOperatorSpecs` | `attentionKind=linear` | qkv/gate/decay projection -> short conv -> state update -> output gate |
| `qwen35FullAttentionOperatorSpecs` | `qwen35_full` | fused qkvz -> q/k norm -> RoPE -> score/context -> output gate -> o |
| `mlaAttentionOperatorSpecs` | `attentionKind=mla` | q compression + KV compression -> split -> norm/RoPE -> MLA score/context -> o |
| `qsaAttentionOperatorSpecs` | `attentionKind=qsa` | qkv/norm -> indexer -> selected sparse attention -> o |
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

公式元数据的代码来源是 `frontend/src/structure/formulas/index.js`。表中公式按 Markdown 做了必要的排版归一，但含义和公式 ID 与代码一致；它们是当前页面展示的架构语义，不等同于某个 runtime kernel 的源码。

### 7.1 基础算子

| ID | 公式 | 关系 |
|---|---|---|
| `linear` | `Y = XW^T + b` | q/k/v、MLP、output/projector |
| `matmul` | `Y = A B` | attention score/context |
| `softmax` | `softmax(x_i) = exp(x_i) / sum_j exp(x_j)` | score -> probability |
| `sigmoid` | `sigmoid(x) = 1 / (1 + exp(-x))` | router/gate/beta |
| `split` | `[q,k,v,b,f_a,g_a] = split(z; 3P,H,D,D)` | fused projection split |
| `causal_conv1d` | `x'_t = SiLU(Conv1D(x_{t-w+1:t}; w))` | linear attention history |
| `kda_decay` | `g_t = lower_bound * sigmoid(exp(A_log) * (g_t^{raw} + dt_bias))` | KDA safe decay |
| `rope` | `q', k' = rotate(q, k, position)` | positional encoding |
| `rmsnorm` | `y = x / sqrt(mean(x^2) + eps) * weight` | standard norm |
| `gemma_rmsnorm` | `y = x / sqrt(mean(x^2) + eps) * (1 + weight)` | Gemma-style norm |
| `swiglu` | `y = SiLU(xW_gate) * (xW_up)` | gated MLP |

### 7.2 MoE 和 linear attention

| ID | 公式 | 关系 |
|---|---|---|
| `topk` | `experts = topk(router_logits, k)` | routed expert selection |
| `moe_dispatch` | `x_e = dispatch(x, expert_ids)` | token -> expert inputs |
| `moe_combine` | `y = sum_e weight_e * expert_e(x_e)` | expert outputs -> hidden |
| `moe_add` | `y = y_routed + y_shared` | shared/routed merge |
| `linear_attention` | `S_t = decay_t*S_{t-1} + k_t^T v_t; y_t = q_t S_t` | generic recurrent attention |
| `linear_attention_gate` | `y_t = gate(z_t) * y_t` | gated recurrent output |
| `kimi_kda` | `beta=sigmoid(beta_raw); v'=beta(v-S_prev k); S=exp(g)S_prev+v'k^T; o=S q` | Kimi KDA |
| `kimi_kda_output_gate` | `y = RMSNorm(o, sigmoid(g))` | Kimi output gate |
| `kimi_fused_qkvg_split` | `[q,k,v,g] = split(z; P,P,P,P)` | Kimi fused projection |
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
| `ple` | `e=HashNGram(ids,context); [k,v]=W_kv e; y=ShortConv(GatedNorm(k,v,RMSNorm(H)))` | position learning enhancement |
| `shared_expert_gate` | `y = y_routed + sigmoid(W_g x) * y_shared` | gated shared expert |
| `qsa_indexer` | `I = topk((W_q x)(W_k K)^T/sqrt(d_i), budget)` | sparse token index |
| `qsa_attention` | `O = softmax(Q K_I^T/sqrt(d)) V_I` | selected sparse attention |
| `qwen_qkvz_split` | `[q,k,v,z] = split(W_qkvz x; q,k,v,z)` | Qwen GDN split |

### 7.4 Sparse、DSV4 和特殊注意力

| ID | 公式 | 关系 |
|---|---|---|
| `attention_qkv_split` | `[q,k,v] = split(W_qkv x; q,k,v)` | MiniMax/GLM fused QKV |
| `attention_output_gate` | `O' = sigmoid(G) * O` | full-attention output gate |
| `minimax_sparse_indexer` | `B = topk_blocks(score_type(Q_i K_i^T/sqrt(d_i)), k)` | block selection |
| `minimax_sparse_attention` | `O = softmax(Q K_B^T/sqrt(d)) V_B` | selected block GQA |
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
| `linear`、`matmul`、`softmax`、`sigmoid` | S1 | 通用线性代数和归一化定义 |
| `rope`、`rmsnorm`、`gemma_rmsnorm`、`swiglu` | S1 | 标准 Transformer 位置、归一化和激活语义 |
| `topk`、`moe_dispatch`、`moe_combine` | S1 | 通用稀疏 MoE 路由关系 |
| `split`、`causal_conv1d`、`kda_decay`、`linear_attention` | S2 | GLM/Qwen/Kimi linear-attention 投影、卷积和状态链 |
| `kimi_kda`、`kimi_kda_output_gate`、`kimi_fused_qkvg_split` | S2 | Kimi KDA 模型实现语义 |
| `gated_delta_attention`、`gated_rmsnorm` | S2 | Qwen/GLM/Kimi 共用的 canonical KDA 语义 |
| `mhc_pre`、`mhc_fused_post_pre`、`mhc_post`、`mhc_contract` | S2 | vLLM mHC pre/post/contract 实现关系 |
| `mla_query_compress`、`mla_kv_compress`、`mla_kv_split`、`mla_output_gate` | S2 | vLLM/SGLang MLA projection/cache 路径 |
| `attention_residual`、`hyper_connection`、`ple` | S2 | Kimi/Qwen/GLM 多流和位置增强实现关系 |
| `qsa_indexer`、`qsa_attention`、`qwen_qkvz_split` | S2 | QSA/DSA/GDN 模型实现关系 |
| `attention_qkv_split`、`attention_output_gate` | S2 | fused QKV 和 full-attention gate 实现关系 |
| `minimax_sparse_indexer`、`minimax_sparse_attention` | S2 | MiniMax block indexer 和 sparse backend 语义 |
| `dsv4_hash_route`、`dsv4_swa_attention`、`dsv4_compressed_attention`、`dsv4_output_projection` | S2 | DeepSeek V4 hash MoE、SWA、compressed MLA 路径 |
| `moe_add`、`linear_attention_gate`、`shared_expert_gate` | S3 | MSV 对分支合并和门控关系的显式建模 |

## 8. 公式来源和可信度

| 来源类型 | 当前代码位置 | 含义 |
|---|---|---|
| MSV 语义公式 | `structure/formulas/index.js` | 页面展示公式，由 MSV 按结构语义维护 |
| 参考实现 | `model_executor/ops/index.js` 的 `implementation` | vLLM/SGLang 类、算子或 backend 名称，不在浏览器执行 |
| 成本方法论 | `cost/*.js` 注释和函数 | 借鉴 llm-analysis 的解析形式和效率因子，不承诺精度等价 |
| shape/协议事实 | safetensors、`@huggingface/hub`、模型 config | 参数量、dtype、shape、cache 和模型识别事实 |

外部参考：

- [llm-analysis](https://github.com/cli99/llm-analysis)：线性层/attention FLOPs、KV cache、并行和效率因子的解析模型参考。
- [vLLM](https://github.com/vllm-project/vllm)：模型执行、KDA state、attention/MoE 算子和实现名称参考。
- [SGLang](https://github.com/sgl-project/sglang)：模型执行、attention/MoE/backend 实现名称参考。
- [safetensors](https://github.com/huggingface/safetensors)：header、dtype、shape、offset 和 tensor key 事实来源。
- [@huggingface/hub](https://github.com/huggingface/huggingface.js)：浏览器端 safetensors metadata 和参数量解析实现。

限制：`formulas/index.js` 当前没有逐公式的外部 URL 字段。上述外部来源是方法论或参考实现来源，不表示每条公式都逐行复制自外部函数。新增公式时应同步记录来源类别。

## 9. 成本公式关系

```text
materialized root + normalized config + load + plan + chip
  -> computeNodeCosts
  -> memoryBreakdown
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

代码：`cost/weights.js`、`cost/memory.js`、`cost/aggregate.js`。无 truth 时 `derivedWeights.js` 根据 config 做 fallback，必须标为 `derived`。

### 9.2 KV cache 和 request state

```text
GQA KV/token = 2 * layers * kv_heads * head_dim * bytes
MLA KV/token = layers * (kv_lora_rank + qk_rope_head_dim) * bytes
KV          = batch * tokens * KV/token

KDA state/layer = conv_history_elements
                + value_heads * value_dim * key_dim
```

线性 attention 的 recurrent state 是 request state，不是 token KV。代码：`cost/memory.js`。KV 注释来源为 llm-analysis `get_memory_kv_cache_per_layer`；KDA state shape 来源为 vLLM `MambaStateShapeCalculator.kda_state_shape`。

### 9.3 MACs/FLOPs

```text
MACs_linear = tokens * input_width * output_width * expert_fraction
FLOPs       = 2 * MACs
tokens      = batch * sequence  # prefill
tokens      = batch             # decode

MACs_attention = query_tokens * heads * context_tokens
                 * (query_key_dim + value_dim)
```

QSA 使用 `min(sequence, indexer_budget)`；MiniMax sparse 使用 `(topk + init + local) * block_size`；linear attention 按各模型投影、卷积、状态更新和输出投影分解。代码：`cost/compute.js`。方法论注释指向 llm-analysis 的 linear/attention FLOPs 函数。

### 9.4 并行投影

```text
GQA KV shard = min(TP, kv_heads)
MLA KV shard = 1
DP-attention KV shard = 1
```

routed expert 权重按 EP 切分；norm 和非 vocab-parallel embedding/lm_head 复制；普通 projection 按 TP 切分；PP 只负责 stage 归属。代码：`cost/parallel.js`。

### 9.5 通信量

```text
TP all-reduce = operations * 2 * (TP-1)/TP * B * T * H * bytes
EP all-to-all = operations * B * T * experts_per_token * H * bytes
PP boundary   = max(PP-1, 0) * B * T * H * bytes
```

PD transfer 使用 decode rank 数和两侧较小链路带宽；布局不同只标记重排。代码：`cost/comm.js`。不建模 overlap、调度和真实拥塞。

### 9.6 Roofline

```text
t_compute = 2 * MACs / (peak_flops * eta_flops)
t_memory  = bytes_moved / (memory_bandwidth * eta_hbm)
t_comm    = comm_bytes / (link_bandwidth * eta_comm)
bound     = max(available times)
```

缺少必要输入时返回 `unknown`。效率因子来源于 llm-analysis 方法论，默认值和芯片覆盖由 `cost/efficiency.js` 管理。

### 9.7 显存 fit

```text
total = weights + KV + request_state
      + activation_peak + runtime_const + communication_buffer
```

`activation_peak`、`runtime_const` 和 `communication_buffer` 是显式用户假设，不是权重百分比，也不代表 allocator 实测峰值。

## 10. UI、导出和诊断关系

```text
StructureNode
  ├── SummaryChips      -> 模型摘要、来源、状态
  ├── ArchitectureTab   -> React Flow、公式索引、Cost Lens
  ├── StructureSearchBox + React Flow -> 展开、折叠、选择和数据流
  ├── NodeDetailPanel   -> truth、公式、shape、属性、node lens
  ├── ExportTab         -> Mermaid / DOT / JSON
  └── RawConfigTab      -> extra_config
```

节点路径是 Architecture、公式按钮、Layers、搜索、Inspector 和 node Cost Lens 的唯一联动键。`diagnostics` 必须说明结构来源、truth 状态、fallback 和 repair；UI 不能只根据是否有结构树判断验证成功。

## 11. 测试对应关系

| 实现层 | 主要测试 |
|---|---|
| config/registry/builder/layer | `structure/modelArchitecture.test.js`、`builtinModels.test.js` |
| IR/materializer/truth merge | `structure/truth/__tests__/mergeSemantics.test.js`、structure tests |
| shape/dtype/safetensors | `dims.test.js`、`safetensorsReader.test.js`、`skeleton.test.js` |
| 公式和成本 | `compute.test.js`、`memory.test.js`、`derivedWeights.test.js`、`roofline.test.js` |
| TP/PP/EP/DP/PD | `parallel.test.js`、`comm.test.js`、`pdSummary.test.js` |
| 芯片来源和缺项 | `publicChips.test.js`、`coverage.test.js`、`manualChip.test.js` |
| 图和交互 | `diagram/*.test.js`、`ModelEntry.test.js`、`useStructure.test.js` |
| 导出 | `frontend/src/exporters.test.js` |

新增或修改公式至少要同时更新公式元数据、operator spec、成本函数（如果参与成本）、单元测试和本文公式/来源表。

## 12. 修改规则

1. 配置字段变化改 `normalize.js`，不把厂商判断散落到 UI。
2. 官方架构名称变化改 `aliases.js`。
3. 顶层拓扑变化改 `model_executor/models/`。
4. 可复用模块变化改 `layers/`；最小算子语义改 `ops/` 和 `formulas/`。
5. 参数和 shape 真值改 truth/skeleton/merge 路径，不在模板中复制 checkpoint 事实。
6. 成本公式改 `cost/`，同时更新来源注释和成本单测。
7. IR 字段改动必须同步 materializer、API schema、导出、UI 和测试。
8. `implementation` 字段只用于解释和参考映射，不是运行时依赖。
