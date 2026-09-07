# 开发原则

本文档是 msv 的**强制约束**，开发时必须遵守。

**文档优先级**：本文档 > `architecture.md` / `ui_interaction.md` / `details/*`。
本文档与代码冲突时，视为代码缺陷，改代码或走 §10 例外登记；不得静默偏离。

每条原则的格式为：**陈述** → **判据**（怎么算违反）→ **检查**（怎么发现违反）。

---

## 1. 定位：五支柱与越界判据

msv 只做五件事：

1. **零权重下载**拿到完整可动画结构图
2. **"放得下吗"**：显存分解、逐卡 fit、并行投影、多芯片（含国产芯片）
3. **transformers 校验**：结构与真实框架对齐
4. **算子级可解释**：算子做什么、代价多大、瓶颈在算力/带宽/通信
5. **轻量**：静态可部署，前端不装 torch

**判据**：任何新能力必须能明确映射到其中一支，否则是越界，不做。

**明确不做**（提出即驳回，需要时指向 [Vidur](https://github.com/microsoft/vidur)）：
plan 搜索 / 推荐最优配置、吞吐与延迟与 TTFT 预测、与实测对齐的校准闭环、
训练内存规划（优化器状态/梯度/激活重算）、运行时 trace、前端装 torch、
调度与批处理动态、脉冲回放式动画。

**检查**：PR 描述必须写明本次改动服务哪一支。写不出来的先讨论范围，不写代码。

---

## 2. 结构原则：节点即图，递归至算子

### 2.1 节点**可以**含图；含图的节点**必须显式声明边**

模型由节点组成，节点内部可以是一张图，递归向下直到叶子节点对应算子。
"可以"是关键——参照 MLIR（`operations **may** have regions`）与 Model Explorer
（layer 可展开），普遍递归是能力而非义务。

**判据**：
- 构造了 `children` 且这些子节点之间存在真实数据流的模块，**必须**在 attributes 里
  声明 `dataflow_edges`。
- **禁止**新增任何依赖**显示名**（`node.name`）推断拓扑的代码。
  `materializeStructureGraph.js` 中残留的 `legacySemanticEdges` 是历史遗留，
  已不可达，不得复活、不得参照其写法扩展。

**检查**：新增 layer builder 的测试必须断言其 `dataflow_edges` 非空，或显式标注
该模块为顺序执行（见 2.2）。

### 2.2 每条边必须携带来源等级 `evidence`，且 UI 必须能区分

一张图里"声明的边"和"推断的边"可信度不同，用户必须能看出区别。

| `evidence` | 含义 | 可信度 |
|---|---|---|
| `declared` | builder 通过 `dataflow_edges` 显式声明 | 事实 |
| `shape-match` | 由 input/output shape 相等推断 | 启发式 |
| `module-order` | 仅由兄弟节点顺序推断 | 弱假设 |

**判据**：
- 边的 `evidence` **不得为空**。
- UI **必须**对非 `declared` 的边做视觉区分（虚线/弱化/hover 提示），
  **禁止**把推断边画成与声明边完全一致的样子。

**检查**：`materializeStructureGraph` 的测试断言所有输出边均带 `evidence`；
diagram 层测试断言 `module-order` 边的样式与 `declared` 不同。

### 2.3 算子是终结节点

叶子节点对应模型算子，算子**不再含图**。算子的实现差异（vLLM/SGLang 的 fused kernel）
写进 `attributes.implementation`，**不得**为同一语义拆出多个节点。

**判据**：出现"同一个数学操作因为框架实现不同而产生两个节点"即违反。

### 2.4 父节点只汇总，公式归叶子

含参与计算的公式一律挂在叶子；父节点的成本是子树聚合值，用独立字段
（`aggregate_macs` / `aggregate_weightBytes`）承载。

**判据**：父子同时计入模型总量即为重复计费，属 bug。

**检查**：`cost/__tests__` 必须有一条"父节点自身 compute_macs 为 0"的断言。

### 2.5 残差与跨层级回连是已知缺口，不得伪造

当前 IR 无法表达"从父模块入口绕到子模块之后"的边，因此**残差旁路画不出来**。
这是嵌套图 + 反馈边的固有难点（MLIR 亦为此把 graph region 限制为单 block）。

**判据**：**禁止**用"并列一个 residual 兄弟节点"来假装表达了残差边，
也**禁止**在文档或 UI 上声称已支持残差拓扑。要做就扩 IR，不做就留空并标注。

---

## 3. 成本原则：算子 → 公式 → 成本

### 3.1 公式表是算子的**唯一注册点**，条目产出**动作向量**

参照 PyTorch `torch.utils.flop_counter` 的 `flop_registry`（按 aten op 注册、入参仅 shape、
未注册先分解再记 0），但**有意超越它**：flop_counter 只数矩阵系 FLOPs 且明确不数
elementwise（softmax/norm 贡献 0）；本仓的注册条目产出**四维动作向量**
（多单元动作模型即 Accelergy 的 Action Counts 原生形态）：

```js
// frontend/src/structure/formulas/index.js
softmax: {
  title, formula, explanation, inputs, outputs,
  aten: "aten._softmax",            // 对照锚点；无对应则省略，用 counts 注释声明分解
  // ref: torch flop_counter 明确不数 softmax（有意超越，依据 Accelergy 多单元动作模型）；
  //      SFU 吞吐见 chip.sfu_ops（CUDA guide：16/SM/clk vs FP32 128）
  // 假设：融合单遍实现（FlashAttention 式），logits 读 1 遍；未融合的多遍读放大不建模（§2.3）
  counts: ({ tokens, vocab, bytesPerElement }) => ({
    matrix: 0,                      // 精确陈述：不用矩阵单元
    vector: 3 * tokens * vocab,     // max/sum 归约 + 逐元素乘
    sfu:    2 * tokens * vocab,     // exp + div
    bytes:  { weights: 0,
              actIn:  tokens * vocab * bytesPerElement,
              actOut: tokens * vocab * bytesPerElement },
  }),
}
```

**条目三分类**：计算+访存（matrix>0）/ 仅访存（matrix=0，vector/sfu/bytes 非零）/
分解声明（counts 写成已知 op 的组合，分解假设显式标注）。
**不存在"未实现"类**：每条必须终止于 counts；null 唯一来源是芯片缺字段（§7 降级），
那是硬件信息问题，不是实现缺口。

**约定**：
- `counts` 入参**只含结构化 shape 参数**，拿不到 `node` 与显示名（§3.2 在结构上不可违反）。
- `matrix` 存 MACs；aten 公式是 FLOPs（含 2×），抄公式时显式换算并注明。
  `vector` 存 flop，`sfu` 存操作次数——单位不同，逐条注明。
- `bytes` 是**每次前向的 compulsory traffic**（权重读一遍 + 输入 + 输出），**无 phase 分支**：
  decode 的 memory-bound 现象由 seq=1 自然涌现（activations 与 matrix 变小、weights 不变）。

**判据**：新增算子时，若需要在 `formulas/index.js` **之外**再改一处分派逻辑
才能让成本生效，即违反。

**检查**：
- CI 断言每个条目三选一（counts / 纯 traffic / 分解声明），无白名单；
- **整模型恒等式作为 matrix 维度的外部 oracle**：对每个内置模型，
  counts 聚合的 matrix FLOPs ≈ `2 × 参数量 × tokens`（dense；MoE 按 expertFraction 缩放）
  ——训练 6ND / 推理 2ND 的标准 invariant，独立于实现，能抓住 /TP 写错、漏 2×、单位错；
  decode 场景补充恒等式：seq=1 时 traffic ≈ 权重字节数（强度 ~1-2，memory-bound）。

### 3.2 禁止显示名参与任何数值计算

**判据**：`cost/` 下出现对 `node.name` 的正则或字符串匹配用于选择公式、
判断分支或计算数值，即违反。分派只允许基于 `type`、`attributes.operator_id`
和结构化 attributes。

**理由**：改一个中文/英文显示名就静默改变成本结果，且无任何测试会拦住。

**检查**：CI 用 grep 断言 `frontend/src/cost/**` 不含 `node?.name` 参与计算的模式。

### 3.3 `null` 与 `0` 必须区分——且按单元区分

`matrix = 0` 是**精确陈述**："该算子不使用矩阵单元"（softmax / norm / rope）——
它的 vector / sfu / bytes 通常非零，不得因 matrix 为 0 而宣称"无成本"。
任一单元 `null` 表示"未实现或无法确定"。

**判据**：把未实现当成 0 计入总量即违反——它会让不完整的总量看起来像完整的。
把"matrix 为 0"渲染成"零成本"同样违反。

**检查**：汇总条必须展示"N 个算子成本未覆盖"（按单元缺失分别计数），
`aggregate.js` 的 null 传播有单测。

### 3.4 "做多少事"与"每件事多贵"必须分离

参照 Accelergy 的 **ERT（单位动作代价，只跟硬件有关）× Action Counts
（动作次数，只跟模型和负载有关）**。

**判据**：芯片参数（算力、带宽、互联）出现在 action counts 的计算路径里即违反。
换一张卡只应触发"表乘法"，不应触发结构遍历与 MAC 重算。

**理由**：msv 的核心卖点是"换卡后瓶颈移到哪"，这正是 ERT 变、counts 不变。
两者纠缠会让多芯片对比变成 N 次全量重算。

### 3.5 每个公式实现处必须写来源注释

```js
// ref: llm-analysis LLMAnalysis.get_num_flops_fwd_per_layer_attn
```

**判据**：新增或修改数值公式而无 `// ref:` 注释，不予合入。

### 3.6 不追精度，但守量级正确

成本模块是**理论分析**，不是仿真。效率因子给文献默认值 + UI 可调 + 明示假设，
然后停手；不做实测校准闭环。

**但要区分"准确"与"正确"**：倍数级错误会**改变结论本身**，那是 bug 不是精度问题，
必须有单测。已知三类高频错误各须有单测：

- GQA：`kv_per_card = kv / min(TP, num_kv_heads)`，**不是** `/ TP`
- **MLA 的 KV 无法按 TP 切分**（单个压缩 latent，无头维度）→ TP 下全量复制
- DP-attention：每个 DP rank 持有完整 KV，单卡 KV 不随 DP 下降

**判据**：验收标准写"能得出正确的定性结论"，**不写**"与实测偏差 < X%"。

### 3.7 算力按单元分：矩阵 / 向量 / SFU，瓶颈取多路 max

算力不是一种资源。逐算子模型必须区分：

| 单元 | 内容 | 芯片字段 | 吞吐特征 |
|---|---|---|---|
| 矩阵 | GEMM/CONV（MACs） | `peak_flops[dtype]` | 最高，`η_flops=0.7` |
| 向量 | elementwise、归约加法、逐元素乘（flop） | `vector_flops`（即 FP32 吞吐） | NVIDIA 64/SM/clk @sm80、128 @sm90（CUDA guide 吞吐表） |
| SFU | `exp`/`rsqrt`/`sin`/除法（操作次数） | `sfu_ops`（新增） | 16/SM/clk（CUDA guide）；官方比值 × fp32 rate 推导，source 标注 |
| 访存 | compulsory bytes | `memory_bandwidth` | — |

**时间模型**：单算子时间 = `max(矩阵, 向量, SFU, 访存, 通信)` 各路除以对应 rate
（§3.4 的 ERT ⋈ counts）。瓶颈分类随之细化为五类。

**跨芯片分化有公开数据支撑**：NVIDIA 向量：矩阵 ≈ 1:4~1:8（CUDA guide），
昇腾 910A/B/C ≈ **1:32 ~ 1:128**（Cube FP16 256/294.9/378.9 vs Vector FP32 2/9.2/11.8
TFLOPS，arXiv 2607.20120）。同一 RMSNorm/softmax 在两类芯片上会落进不同瓶颈类
——这正是多芯片对比要暴露的东西。昇腾无独立 SFU（超越函数在向量单元执行），
`sfu_ops` 缺失时可选 per-chip 单元映射（sfu→vector rate，语义映射非估算），W5 定。

**规约的处理**：reduce = N−1 次向量加法 + 访存（读 N 写 ~0），
强度 ≈ 1 flop / 4~8 byte，**在 roofline 分类里几乎必然落 memory-bound**——
不发明"规约单元"，分类交给模型算出来。
softmax 等 score 归一化按**融合单遍实现**假设（logits 读 1 遍，2026-09-07 拍板）；
多遍未融合读放大不建模（§2.3，实现差异进 implementation 属性）。

**效率因子**：矩阵沿用 `η_flops=0.7`；向量/SFU 初始 `η=1.0`（下界语义 + UI 可调 + 明示假设，
不做校准流程，§3.6）。

**缺项降级（§7）**：芯片缺 `sfu_ops` / `vector_flops` → 对应单元的时间不可判，
`coverage.js` 关闭相应能力门控，绝不估算。

---

## 4. 真值优先

### 4.1 数值一律取 checkpoint 真值，不用模板公式推导

- 模型级参数量：`@huggingface/hub` 的 `parameterTotal`
- 节点级：safetensors header range read（单分片约 9KB）→ 逐张量 dtype/shape
- 参数量换算**必须**用 `@huggingface/hub` 的 `parseSafetensorsMetadata`，
  **禁止手写**（子字节量化打包宽度、bitsandbytes 前缀、exponent-only dtype 等边界手写必错）

**建树可以自己写**（`truth/skeleton.js`，trie 无边界问题），**换算不可以**。这两件事不要混。

### 4.2 每个数值必须标注 `value_source`

`checkpoint` / `derived` / `introspect`。UI 必须能区分真值与推导值。

**判据**：新增数值字段而不设 `value_source` 即违反。

### 4.3 适配产物的形态：两张表 + 一份代码

结构的**骨架来自适配**（人工 / agent 对照**该模型的 modeling 文件**组织关系），不来自
config 推导，也不来自 checkpoint 自动推断。运行时**不做结构推断**。

**语义标准的层级**（2026-09-07 定）：

1. **数学定义**——counts 的真正标准，与实现无关；
2. **该模型的 modeling 文件**（版本锚定）——语义分解与执行顺序的规范参考实现。
   多数模型 checkpoint 里**没有** model.py，定义在库内，因此按**五级来源阶梯**取源
   （与 §7 的 confidence 阶梯同构；排序判据 = 与 msv 所读 config 的契约契合度 +
   可锚定性 + "作者 > 转述者"；优先级 2026-09-07 定）：
   - ① **checkpoint 自带的** `modeling_*.py`（remote code）——模型作者所写，随权重分发，
     天然自锚定。识别方式：config.json 的 `auto_map` 字段（如 Kimi-K3：
     `"AutoModelForCausalLM": "modeling_kimi_k3.KimiK3ForConditionalGeneration"`）；
   - ② **transformers 库内**该架构的 `modeling_<arch>.py`——HF 收录时的转写，
     版本钉死，首选锚为 config.json 自带的 `transformers_version` 字段（catalog 可覆盖）；
   - ③ **原始发布仓库**（官方 GitHub 等模型作者自己的实现）——裸 commit 锚定；
     原始配置格式与 HF config.json 的字段映射需人工桥接，映射关系入档；
   - ④ **vLLM / SGLang 的模型实现**（`vllm/model_executor/models/*.py` 等）——引擎作者
     的转写。必须标注引擎 + 版本；
   - ⑤ 都没有 → 未适配（§4.5），不出语义结构。

   任意两级来源之间发现**语义冲突**：记录冲突进 diagnostics，人工裁决后登记，
   **不得静默选一**——阶梯只是执行顺序，不预设谁对。
   `models/` 目录已 vendored 内置模型的 modeling 文件（覆盖 ①），适配离线可做；
   版本漂移的发现机制 = 旁路 B source_ref 再生的 diff 分级（class_name 变化即上游重构告警）。
3. **vLLM / SGLang**——下游优化实现，只提供 `implementation` 属性与融合假设的来源
   （A2/A3），**不得作为语义标准**：融合把语义做没了，且两个引擎互相不一致。

形态照抄成熟方案，分三层：

**层 1｜canonical 节点角色表（全局共享，声明式）**
对齐 llama.cpp 的 `MODEL_TENSOR` 枚举与 `MODEL_TENSORS[arch]` 列表。
角色（`attn_q` / `attn_qkv` / `attn_norm` / `ffn_gate` / …）是**跨模型可比性的唯一载体**：
无论 checkpoint 里叫 `input_layernorm`、`ln_1` 还是 `norm_1`，角色都是 `attn_norm`。

角色与算子类型是两个维度，都要有：
```
{ role: "attn_q",    operator_id: "linear" }
{ role: "attn_norm", operator_id: "rmsnorm" }
```

**层 2｜per-arch checkpoint 映射表（声明式）**
对齐 llama.cpp `gguf-py/gguf/tensor_mapping.py` 的 `block_mappings_cfg`
（候选列表 + `{bid}` 占位）与 transformers 的 `WeightRenaming` / `WeightConverter`
（可组合、可逆的 `ConversionOps`：`Chunk`/`Concatenate`、`MergeModulelist`/`SplitModulelist`、
`Transpose`、`PermuteForRope`）。

```js
attn_out:    ["model.layers.{bid}.self_attn.o_proj",
              "model.layers.{bid}.self_attn.out_proj"],   // 别名用候选列表，不用启发式匹配
attn_qkv:    { from: "model.layers.{bid}.self_attn.qkv_proj", op: Chunk(0) },
ffn_experts: { from: "model.layers.{bid}.mlp.experts.{eid}.gate_proj", op: MergeModulelist(0) },
```

**层 3｜图构建（代码，每 arch 一份）**
对齐 llama.cpp `src/models/<arch>.cpp`、transformers `modeling_*.py`、vLLM `models/*.py`。
负责执行顺序、无参算子插入、条件分支。msv 现有的 `layers/*.js` + `ops/index.js` 就是这一层。

**组件配方是一级概念，家族不是。** 现在的模型都是 transformer，变化只在组件选型
（attention / norm / FFN / 位置编码 / 附加结构），不同家族会采用同一配方
（代码内证据：`minimax_m2` 与 `glm4_moe` 共用同一条 attention 算子链，`attention.js:17-19`）。
**模型 = 配方 + 数字 + 逐层调度。** 新增模型应当是"选一个已有配方 + 填数字"；
只有发明新组件方案才写新代码。家族名只允许出现在"模型 → 配方"的薄解析层。

**判据**：
- 适配产物里出现"为了得到参数量 / shape 而写的公式分支"即违反——那些必须来自真值（§4.1）。
- checkpoint 名与结构节点的对应**必须显式声明在层 2**，**禁止**运行时用路径归一化做相等匹配去猜。
- **禁止**把层 3 改成纯 JSON/YAML 数据文件。图构建含条件分支（MoE/dense 交替、
  `layer_types` 调度、逐层 `compress_ratio`、PLE/MHC 存在性、有无 bias），
  纯数据表达必然要发明 mini-DSL，那是自研且更难维护。
  llama.cpp / transformers / vLLM / TensorRT-LLM 无一例外用代码表达图构建。

**检查**：新增 arch 时，层 1、层 2 各一处声明，层 3 一个 builder；三者之外不得再有该 arch 的分支。

### 4.4 真值缺失或冲突必须可见，禁止静默丢弃

`mergeSemantics` 在同一路径匹配到多个候选时会放弃绑定并记入 `ambiguous_truth_matches`；
`template_gaps` 记录 trie 里有而适配产物未声明的含参模块。

**判据**：这两类信号**必须**在 UI 上可见。只写进 diagnostics 而 UI 不展示，
等于真值静默缺失——用户看到的是一个"看起来完整"的错结构。

层 2 落地后，`ambiguous` 应降为 0（因为对应关系是声明的）；仍出现即说明映射表写错。

### 4.5 适配状态决定能给出什么，禁止伪造

trie（safetensors header）只做两件事：**给已适配模型提供精确数值**、
**给未适配模型提供一个诚实的层级视图**。它不提供骨架、不提供顺序、不提供无参算子。

| | 有 safetensors header | 无 header |
|---|---|---|
| **已适配** | 适配产物骨架 + trie 真值（完整） | 适配产物骨架 + config 推导数值，标 `value_source=derived` |
| **未适配** | 仅 trie 层级与数值，**明确标注"未适配、无语义"** | **不出结构图**，提示需要适配 |

"无 header"不是边缘场景：`cost/weights.js` 的文档注释已把"无 safetensors / gated / 网络"
列为预期情况；`models/` 内置 catalog **只有 config.json，零 safetensors**；
GGUF / `pytorch_model.bin` / `.pth` / MLX 格式模型永远读不到 safetensors header。

**判据**：**禁止**用 generic 兜底给未适配模型编造一个看起来完整的结构图。
未适配就显式说未适配。

### 4.6 映射表必须可逆校验

照 transformers `WeightConverter.reverse_transform()` 的可逆设计：每个 `ConversionOps`
都有 `reverse_op`，一份声明双向可用。

**检查**：层 2 写完后必须有一条测试——用映射把 trie 反推成角色集合，再正推回 checkpoint 名，
与原始 header 逐项对比。对不上即映射错误。这比"看图对不对"可靠得多，是映射表的唯一质量闸门。

### 4.7 config 只供数，方案由组网决定

config 解读包含两种性质不同的工作，归属不同：

- **字段归一**（别名吸收、默认值）：合法的独立层，对应 transformers 的 `Config` 类。
  终态是一个瘦的 config 视图。
- **方案解读**（这个模型的 attention 用哪个方案、逐层怎么调度、归一化用哪种 norm）：
  这是**骨架决策**，属于组网（builder / 配方表），**禁止**在 config 归一层预先决定。
  transformers 的分工即如此：`Config` 类管字段，`modeling_*.py` 的 `__init__` 管决定。

**判据**：config 归一层出现家族条件分支、或输出"selected scheme"类字段即违反。

**检查**：config 视图的输出对象不含方案类字段；逐层调度只在组网入口被消费。

**存量偏离（W3b 收口）**：`normalize.js` 的 `attentionSchedule` / `layerSchedule` /
`linearAttentionMode` / `normMode` 等方案类字段与全部家族条件分支；
`cost/compute.js:308` 对 `layerSchedule` 的跨层消费。

---

## 5. 源码可追溯

### 5.1 由 transformers 实例得到的节点必须携带 `source_ref`

支柱④"算子级可解释"不只是给公式，还要能跳到**框架里真正实现它的那段代码**。

```
source_ref: {
  file: "transformers/models/qwen2/modeling_qwen2.py",
  line: 1456,
  url:  "https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/qwen2/modeling_qwen2.py#L1456",
  framework: "transformers",
  version: "4.40.0",
}
```

### 5.2 采集机制（参照 modelmap `src/modelmap/annotate.py:116-134`）

对每个 `nn.Module` 实例：

```python
cls  = type(module)
file = inspect.getsourcefile(cls)
line = inspect.getsourcelines(cls)[1]
# 用 (包目录, github repo, f"v{__version__}") 前缀匹配，拼 blob 永久链接
```

**这必须在有框架实例的进程里做** —— `inspect` 需要真实的类对象。
因此这是**后端唯一不可替代的能力之一**（见 §6.1），前端拿不到。

### 5.3 `source_ref` 是**离线产物**，静态部署也要能用

后端把 `(module_path, class_name, source_ref, has_params)` 生成为随 catalog 发布的
静态 JSON（与 `models/` 下的内置 config 同级）。

**判据**：把 `source_ref` 做成"必须在线调后端才有"的能力即违反支柱⑤。

**字段必须按腐坏速度分成两半**：

- **稳定部分**（跨版本几乎不变，是持久 key）：`framework` / `module_path` / `class_name` / `file`
- **易腐部分**（绑在一起，缺一不可）：`line` / `version`

`version` 与当前 transformers 不一致时，**降级为不带 `#L` 的文件链接 + 标注行号来源版本**，
而不是给出一个可能错位的锚点。这让"产物没重新生成"的后果从**错误行号**降为**少一个精确锚点**。

**产物再生的自动化**：CI 在 `pyproject.toml` 的 transformers 版本变化时重跑生成，并对 diff 分级——

| diff 类型 | 处理 |
|---|---|
| 只有 `line` 变，`class_name` + `file` 不变 | 自动合入（绝大多数情况） |
| `file` 变（模型被移动或重构） | 标记待确认 |
| `class_name` 变或消失 | **必须人工确认**——上游重构了该模型，适配产物可能也要跟改 |

第三类是免费的上游变更告警：transformers 拆了某个模型的类，在版本升级时立刻暴露，
而不是等用户报"结构不对"。

### 5.4 定位失败时留空，禁止编造

- 类不在已知包根下（remote code / 自定义算子）→ 只给 `file:line`，**不给 url**
- `inspect` 抛错、合成节点、聚合节点 → `source_ref` 留 `null`

**判据**：拼出一个未经校验的 GitHub 链接即违反。链接错了比没有链接更糟。

---

## 6. 前后端边界

### 6.1 后端只做"必须有框架才能拿到的信息"

按这条规则筛，后端只剩三样：

1. **无参子模块**（不在 safetensors 里，trie 拿不到）
2. **真实类名 + 源码位置**（`inspect` 需要类对象，见 §5.2）
3. **config / remote code 可加载性**

加上前端做不到的 **本地目录读取**（local / builtin config）。

**判据**：后端新增能力时若前端能做，就放前端。

### 6.2 前端是唯一的结构主路径

`normalizeConfig → resolveArchitecture → buildNetwork → truth merge → IR`。

**判据**：**禁止后端再产出第二份 `ModelStructure` IR**。校验只需要扁平列表
`[(module_path, class_name, source_ref, has_params)]`，不需要 IR。
"后端也建 IR"是前后端全部重复的来源。

### 6.3 校验的定义是**对比**，不是"能不能建起来"

`/api/verify` 必须返回**差异报告**：仅 transformers 有 / 仅 msv 有 / 类名不符。
只返回 pass/fail 不构成"结构校验"。

**判据**：校验能力必须有 UI 入口。没有入口的支柱等于没做。

### 6.4 来源解析策略只有一份

endpoint fallback、revision 默认值、auto 降级顺序统一由前端
`model/loadModelArtifacts.js` 持有；后端 resolve 只服务 local / builtin。

**判据**：同一策略在前后端各写一份即违反（会不一致，且已经不一致过）。

---

## 7. 芯片数据合规

1. **公开规格入库**（`cost/chips/public.js`）：每条**必带 `source` URL + `confidence`**
   （official / vendor-marketing / community），**支持字段级覆盖**。无 source 不合入。
2. **非公开规格走用户配置**（`chips.local.json`，gitignore）：仓库只提供 schema +
   加载器 + 示例；示例**必须**用 `example-chip` + 明显虚构的占位数字，
   **禁止**写"某卡示例：显存约 XX GB"（会被当真实规格传播）。
3. **缺项按字段降级**，不按卡：缺 `memory_bandwidth` → 仍可 fit/max_context，
   仅禁用 roofline；缺 `memory_bytes` → 仅出相对占比。

**判据**：**绝不用估计值填空**。写第一行芯片数据前先建/改 `chips/coverage.js`。

---

## 8. 扩展性硬约束

### 8.1 新增模型只允许"选配方 + 填数字"

真正的目标是 §4.3 的组件配方：新增模型 = 选一个已有配方 + 填数字；只有发明新组件方案
才写新代码。"家族名硬编码的非测试文件数"是它的**可测量代理指标**，不是目标本身——
配方表落地后，家族名只应出现在配方解析与映射表两处。

**判据**：家族名文件数**不得增加**。当前基线（2026-09-07，W0 以完整 pattern 实测）为
**16** 个文件，清单见 `scripts/check_principles.sh`；更早 review 用窄 pattern 得出的 11
只是漏计变体（独立 `kimi`、`deepseek_v4`、`qwen3_5` 等），这是待偿债务，不是可以照抄的先例。

**检查**：CI 统计家族名（`kimi_k3` / `qwen4_exp` / `glm5_next` / `minimax_m3_vl` /
`deepseek_v32` / … ）出现的**非测试文件数**，只允许下降。

### 8.2 禁止嵌套三元链

`layers/attention.js` 现存两条平行的多分支嵌套三元（children 与 declaredEdges
条件集不一致，必然漂移），是反例。新增分派一律用查表。

---

## 9. 合入检查清单

- [ ] PR 描述写明服务哪一支柱（§1）
- [ ] 含 children 的新模块声明了 `dataflow_edges`（§2.1）
- [ ] 新增边带 `evidence`（§2.2）
- [ ] 新算子只在 `formulas/index.js` 注册，含 `macs`（§3.1）
- [ ] `cost/` 未引入显示名参与计算（§3.2）
- [ ] 未实现成本返回 `null` 而非 `0`（§3.3）
- [ ] 数值公式带 `// ref:`（§3.5）
- [ ] 涉及 KV/TP/MLA/DP 的改动有对应单测（§3.6）
- [ ] 新数值字段带 `value_source`（§4.2）
- [ ] 新增 arch 只在角色表 / 映射表 / builder 三处声明（§4.3）
- [ ] checkpoint 对应关系是显式声明，未引入路径归一化猜测（§4.3）
- [ ] 真值缺失/冲突在 UI 可见（§4.4）
- [ ] 未适配模型未被伪造成完整结构（§4.5）
- [ ] 映射表有可逆校验测试（§4.6）
- [ ] `source_ref` 无法确定时留空而非编造（§5.4）
- [ ] 后端未新增第二份 IR 产出（§6.2）
- [ ] 芯片数据带 `source` + `confidence`（§7）
- [ ] 家族名硬编码文件数未增加（§8.1）
- [ ] 未引入嵌套三元分派（§8.2）

---

## 10. 例外登记

确需偏离本文档时，在 `implementation_plan.md` 中登记：**违反哪条、为什么、
何时收口、收口的判据**。未登记的偏离视为缺陷。

已登记的存量偏离（2026-09-07 复核）：

- §2.1/§2.2：`decoderLayer.js`、`decoderStack.js`、网络层未声明 `dataflow_edges`，
  依赖 `module-order` 兜底；UI 尚未区分 evidence。
- §2.5：残差以并列节点表达，尚未有跨层级边。
- §3.1/§3.2/§3.3：`cost/compute.js` 自带分派链，含两处显示名正则，未实现与 0 不可区分。
- §3.4：ERT 与 action counts 尚未分离。
- §4.3：无 canonical 角色表；checkpoint 对应靠 `mergeSemantics.js:56-75` 的路径归一化**相等匹配**
  推测，冲突时 `continue` 静默放弃绑定；数值仍有部分来自 config 推导。
- §4.7：`normalize.js` 混合字段归一与方案解读，含全部家族分支；方案类字段
  （attentionSchedule / layerSchedule / linearAttentionMode / normMode 等）被组网与 cost
  跨层消费（W3b 收口）。
- §3.1（counts 覆盖）：embedding gather 与残差加法无算子节点，其流量不可见
  （结构级缺口，随 W6/W7 决策；详见 `details/cost_counts.md`）。
- §4.5：未适配模型走 generic 兜底会画出看起来完整的结构图，未标注"未适配"。
- §4.6：无映射表，因此无可逆校验。
- §5.x：`source_ref` 尚未实现（`source_fields` 语义不同，当前存的是 attribute keys）。
- §6.2/§6.3：后端仍产出完整 IR；`/api/verify` 只返回 pass/fail 且无 UI 入口。
- §6.4：前后端各有一套来源解析策略。
- §8.1/§8.2：11 个文件硬编码家族名；`attention.js` 存在嵌套三元链。
