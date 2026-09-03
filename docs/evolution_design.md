# msv 演进设计方案：图粒度 × 成本规划 × 轻量动画

> 状态：**定稿，待开发**（v6，2026-09-04）
> 日期：2026-09-03 初稿 / 2026-09-04 四轮修订
> 读者：msv 维护者 + 下一个开发会话（**开工请直接看 §0.1 定位判据 → §11 开发交接**）
> 相关代码基线：`main`（含 `frontend/src/structure`、`src/model_structure_viewer/structure`）

**v2 修订摘要**（评审后的三处方向变更，取代 v1 相应内容）：

1. **节点级 `params` / `weight_shapes` / `dtype` 不再由模板公式推导，改为读 checkpoint 真值**
   （HF Hub API + safetensors header range read）。v1 §4.2 的 params 公式表作废，降级为离线 fallback。
2. **成本的核心目标从"放得下吗"改为"per-module 计算量 + 内存归因 + 多芯片（含国产芯片）瓶颈分析"**。
   模型级总量只是图的汇总，不是主功能——因为那件事十几个免费计算器已经做了。
3. **固定 15% overhead 作废**，改为「激活峰值近似常数 + 运行时常数」两项分解（依据见 §5.2）。

**v3 修订摘要**（第二轮）：

4. **负载类型明确只做推理**（prefill / decode 分开算），不做训练——不含优化器状态、梯度、激活重算。
5. **并行策略进入范围**：做 given-plan（TP / PP / EP / DP）的每卡资源投影与每层通信量，**不做 plan 搜索与吞吐预测**。
   连带影响：`interconnect` 从"可选字段"升为**必需字段**；bound 分类从「算力 vs 带宽」两类扩为**三类（+通信）**。

**v4 修订摘要**（二次系统审视后，四处变更）：

6. **拓扑骨架也改为读真值**：safetensors key 即 `nn.Module` 路径，建 trie 可免费得到**任何模型**的精确含参模块树。
   模板的职责收缩为「无参算子 + 执行序 + 语义/公式」。覆盖度从"3 个模板家族"变为"**数值 100% + 骨架 100% + 语义仅 3 家族**"。见 §4.2(b2)。
7. **三类 bound 引入显式效率因子**：峰值互比会把 comm 项系统性低估 1.2–1.6 倍，可能把该判 comm-bound 的模块判成 compute-bound。见 §5.3(4)。
8. **公式移植自 llm-analysis 的形态与效率因子设计**（v6 改为：不建对比测试，代之以 §5.7 清单）；plan 搜索明确指向 Vidur。见 §7.4。
9. **PD 分离进范围（P2e）**：plan 扩为 `{prefill_plan, decode_plan}` 一对 + KV 传输量。见 §5.3(7)。

**另有一处措辞修正（非决策）**：v3 说"per-module 激活内存不可归因"过严。**per-module 激活张量大小可算**（`dims.js` 从 config 即可推出）；**不可算的是 allocator 峰值占用**（取决于 kernel fusion / 就地操作 / 内存池复用）。lens 展示前者，不声称后者。

**v5 修订摘要**（定位收口，2026-09-04）：

10. **msv 的成本模块是「理论分析能力」，不是仿真、不是预测、不承诺精度。** 见 §5.1。
    - 效率因子**保留**（它解决的是结构性偏差，不是精度），只作为**带文献默认值的可见假设 + 用户可调旋钮**。
    - 验收标准改为"能得出正确的定性结论"，不设与实测的偏差百分比。

**v6 修订摘要**（定位重申 + 交接定稿，2026-09-04）：

11. **五支柱定位重申，并给出"越界"的可操作判据**（§0.1）。任何新能力必须能映射到五支柱之一，否则即越界。
    经此检验：per-module 计算量与 roofline bound 归入**算子级可解释**；显存分解 / 并行投影 / 多芯片 / PD 归入**"放得下吗"**。
12. **不做任何实测校准**：取消与 vLLM 日志 / XProfiler / `xccl_perf` 的一切校准动作（连"可选抽查"也不保留）。
    成本输出定性为**理论计算，结果供参考**。
13. **精度不校准，但公式必须正确**：新增 §5.7「公式正确性清单」——逐条列出必须正确的公式、
    对应的 llm-analysis 函数、以及是否需要单测。这是本方案**唯一的质量闸门**。
14. **编码约定**：每个公式的实现处必须写注释指明它对应 llm-analysis 的哪个函数（§5.7）。
15. 新增 **§11「开发交接」**，供下一个会话直接开工。

---

## 0. 文档目的

本方案把 msv 从"结构查看 + 校验"演进为兼具以下能力的工具：

1. **零权重下载**（保持）——只读 config / 元数据 / safetensors header（KB 级 range read），不下权重。
2. **完整可动画图**（升级）——节点骨架与形状/参数**取 checkpoint 真值**（safetensors key 建 trie），模板只负责无参算子、执行序、语义，不引入运行时 trace。
3. **"放得下吗"**（新增）——显存分解 + 逐卡 fit + max_context，含并行策略（TP/PP/EP/DP）与多芯片（含国产芯片）、PD 分离。
4. **transformers 校验**（保持，降权）——后端 introspection 从"数值校准器"退为"无参算子与执行序的校验器"，因为数值已有真值来源。
5. **算子级可解释**（升级）——`formulas/` + `ops/` 系统，升级为"公式 ↔ 图"双向联动；并扩展为**解释每个算子的代价与瓶颈**（per-module 计算量 + roofline bound）。
6. **轻量**（保持）——前端无 torch；静态部署可行。

### 0.1 定位判据：什么算越界（v6，开工前必读）

**msv 的定位就是上面五件事**（零权重下载拿完整可动画图 / "放得下吗" / transformers 校验 / 算子级可解释 / 轻量）。

**判据：任何新能力必须能明确映射到其中一支，否则就是越界，不做。**

本方案所有新增能力经此检验后的归属：

| 新增能力 | 归入 | 说明 |
|---|---|---|
| safetensors trie 建骨架 + header 取 shape/dtype | ①零权重下载 + ②完整可动画图 | KB 级 range read，不下权重；换来任何模型的精确树 |
| 权重/KV/常数项分解、逐卡 fit、max_context | ③"放得下吗" | 这就是"放得下吗"本身 |
| 并行投影（TP/PP/EP/DP）、多芯片、PD 分离 | ③"放得下吗" | 大模型单卡必然放不下，不谈并行与卡型答不了这个问题 |
| per-module 计算量（MACs）、roofline 三类 bound | ⑤算子级可解释 | 算子解释不只是"它做什么"，也包括"它的代价多大、瓶颈在算力还是带宽还是通信" |
| 效率因子 η（三个） | ⑤算子级可解释 | 不加会让 bound 分类结论错，属于让解释成立的必要条件 |
| 公式 ↔ 节点双向联动 | ⑤算子级可解释 | msv 唯一无替代品的交互 |
| 前端直连 HF、按字段降级、缺项协作入口 | ⑥轻量 | 让静态部署真正可用，不引后端 |

**明确判为越界、不做的**（每条都注明它越了哪一支）：

| 越界项 | 越了什么 | 替代 |
|---|---|---|
| plan 搜索 / 推荐最优并行配置 | ③④⑤ 都不是，是部署规划 | 指向 Vidur（§7.4） |
| 吞吐 / 延迟 / TTFT 预测 | 同上，且需要 msv 没有的输入 | 指向 Vidur |
| 与实测对齐的校准闭环 | 违反⑥轻量，且把③⑤推向仿真 | 不做，见 §5.1 |
| 训练内存规划（优化器/梯度/重算） | 五支柱均不涉及训练 | 不做 |
| 运行时 trace / 装 torch 到前端 | 违反①零权重下载与⑥轻量 | 用 trie + 模板（§4.2） |
| 调度 / 批处理动态 / KV 传输流水 | 是服务系统问题，非结构查看 | 不做 |
| modelmap 式脉冲回放动画 | 重投入且不服务任何一支 | hover 联动即可（§6.2） |

> 后续任何"这个也加上会更有用"的提议，先过这张表。**过不去就是越界。**

已拍板的方向决策见 §2，成熟实现选型与联网调研依据见 §7，开发交接见 §11。

---

## 1. 现状分析（代码事实，可核实）

### 1.1 双路径架构

msv 有两条独立的结构构建路径，产出同一份 `ModelStructure`（pydantic `StructureNode`）：

```
                      ┌─ 前端路径（主要，静态部署）：纯 config 驱动，无 torch
输入(config) ──→       ┤
                      └─ 后端路径（重，本地验证）：meta introspection + repair 阶梯
                                  ↓
                    同一份 ModelStructure（schemas.py: StructureNode）
```

- **前端**：`frontend/src/structure/buildStructure.js` →
  `normalizeConfig` → `resolveArchitecture(registry)` → `model_executor` builders → `createStructureIr` → `materializeModelStructure`。
- **后端**：`src/model_structure_viewer/structure/introspect.py` → `AutoModel.from_config + accelerate.init_empty_weights`（等价 meta device）→ walk nn.Module → `fold.collapse`；失败走 `recovery.py` 的 repair→retry 阶梯。

### 1.2 成本规划可利用的现有数据（关键事实）

`frontend/src/structure/config/normalize.js:61-93`（`normalizeConfig`）已归一化出**KV / 计算量解析式所需的全部模型级数字**（实测核对，字段名与产出一致）：

| 字段 | 对应 config 键（多别名归一） |
|---|---|
| `hiddenSize` | `hidden_size` / `dim` / `d_model` |
| `attentionHeads` | `num_attention_heads` / `n_heads` / `attention_heads` |
| `kvHeads` | `num_key_value_heads` / `n_kv_heads` / `kv_heads` |
| `headDim` / `valueHeadDim` | `head_dim`（含 MLA 的 `qk_nope+qk_rope` 推导）/ `v_head_dim` |
| `intermediateSize` / `moeIntermediateSize` | `intermediate_size` / `moe_intermediate_size` |
| `experts` / `expertsPerToken` | `num_local_experts` / `n_routed_experts` / `num_experts` / `moe_top_k`… |
| `vocabSize` | `vocab_size` |
| `layers` / `contextLength` | `num_hidden_layers` / `max_position_embeddings`… |
| `layerSchedule` | `mlp_layer_types` / `moe_layer_freq` / `first_k_dense_replace` |
| `attentionSchedule` | `sparse_attention_config.sparse_attention_freq` |

**它不产出的字段**（v1 漏写）：`torch_dtype`、`quantization_config`、`tie_word_embeddings`、`rope_theta`、`kv_lora_rank`、`qk_rope_head_dim`。

**结论（v2 修订）**：
- **KV 字节与计算量**的解析式输入已齐备，只需补 `kv_lora_rank` / `qk_rope_head_dim`（MLA）。
- **权重字节不走这条路**——前三个缺失字段（`torch_dtype` / `quantization_config` / `tie_word_embeddings`）恰好是 v1 打算用来推 dtype 的输入，而它们全部可被 HF API 的逐 dtype 真值取代（§4.2(a)）。因此**不需要为成本去扩 `normalizeConfig` 的 dtype 相关字段**。

### 1.3 缺口（成本规划与图粒度的前置条件）

| # | 缺口 | 现状（代码依据） | 影响 |
|---|---|---|---|
| G1 | 节点级无参数计数 | `schemas.py:12-20` `StructureNode` 无 `params`；`materializers/toStructureNode.js` 只落 `id/name/type/repeat/attributes/source_fields/confidence/children` | 无法做 per-module 成本分解 |
| G2 | **不存在数值形状（不是"被丢弃"）** | `model_executor/shapes.js`：`dimension()` 直接把数值格式化进字符串（`"hidden size=4096"`），`tensorShapes` 每个值本来就是字符串 → **数值形态从未存在过** | 需要把 `shapes.js` 拆成数值层 + 展示层两套，工作量大于 v1 估计 |
| G3 | fold 签名不含 shape 维度 | `structure/fold.py:145-152` `_signature` 只返回 `(node.type, class_label, tuple(children_sig))` | 中间维度不同的层会被误折叠，成本会错 |
| G4 | 前端不直连 HF | `frontend/src/api/client.js` 所有 HF 调用走后端 `/api/hf/config`、`/api/hf/search` | **当前静态部署形态只能看内置 catalog**；读 safetensors header 必须先开前端直连 HF 的路径 |
| G5 | 零成本存量代码 | 全仓 grep `params\|vram\|kv_cache\|flops\|macs\|numel` 无实现命中 | 成本模块从零开始 |
| G6 | 无静态部署配置 | 无 `.github/workflows/` | "静态部署可行"是可行性，非既成事实 |

> G3 对照：modelmap `collapse.py` 的签名含 `weight_shapes`（`kind + cls + weight_shapes` 递归 hash），所以不会误折叠。msv 在 P0 拿到真实 `weight_shapes` 后同步补齐即可。

### 1.4 规模基线

- 前端 `frontend/src` ≈ 3063 行 JS/JSX；后端 `src` ≈ 3712 行 Python。
- 架构模板只有 3 个真实家族：`model_executor/models/{qwen,deepseek,minimax}.js`，外加 `generic.js` 兜底与 `common.js` 共用件。
- 前端**零图库**：`frontend/package.json` 依赖仅 react / react-dom / vite / @vitejs/plugin-react。`frontend/src/diagram` 全手写 → P2 lens 与 P3 联动的渲染与命中测试都要自己实现。

---

## 2. 已拍板决策记录

| 决策点 | 结论 | 备注 |
|---|---|---|
| 节点级数值来源 | **读 checkpoint 真值**（HF API + safetensors header），模板公式推导降级为离线 fallback | v2 变更；依据 §4.2 |
| **拓扑骨架来源** | **也读真值**：safetensors key 建 trie → 精确含参模块树；模板只补无参算子 + 执行序 + 语义 | **v4 变更**；依据 §4.2(b2) |
| 覆盖策略 | **数值 100% + 骨架 100%（任何有 safetensors 的模型）+ 语义仅 3 个模板家族** | v4 变更：骨架不再受模板数量限制 |
| 成本核心目标 | **per-module 计算量 + 内存归因 + 多芯片瓶颈分析** | v2 变更；模型级总量降为图的汇总 |
| 芯片覆盖 | **含国产芯片**（昆仑芯 / 昇腾 / 寒武纪 / …），需要显存 + 显存带宽 + 各 dtype 算力 + 互联带宽四类数据 | 这是相对通用计算器的唯一结构性差异 |
| 芯片规格数据 | **公开的入库（必带 source + confidence，支持字段级覆盖），非公开的走用户配置**（`chips.local.json` + 手动录入），**缺项按字段降级 + 显式补充入口** | msv 仓库不持有非公开数据；详见 §5.4(c) |
| 成本 UI | **图上 lens 为主（P2）+ 汇总 Tab（P1，轻量）** | v2 变更：原为 Tab 先行 |
| 动画范围 | **轻量版：hover 高亮 + 执行序 + 边流线** | 不做 modelmap 式脉冲回放 |
| overhead 建模 | **激活峰值近似常数 + 运行时常数两项**，不用固定百分比 | v2 变更；依据 §5.3(5) |
| roofline | **做 per-module 瓶颈判定（算力 vs 显存带宽 vs 通信）**，**带显式效率因子** | v2 引入 / v4 补效率因子，见 §5.3(4) |
| 负载类型 | **只做推理**（prefill / decode 分开算），不做训练 | 不含优化器状态 / 梯度 / 激活重算 |
| 并行策略 | **做 given-plan 的资源投影 + 通信量**（TP / PP / EP / DP），**不做 plan 搜索与吞吐预测** | v3 新增；plan 搜索指向 Vidur（§7.4） |
| **PD 分离** | **进范围（P2e）**：plan 扩为 `{prefill_plan, decode_plan}` 一对 + KV 传输量 | **v4 新增**；依据 §5.3(7) |
| **公式来源** | **移植 llm-analysis 的公式与效率因子方法论**；正确性由 §5.7 清单 + 单测把守，**不建对比测试套件** | v4 新增 / v6 收口 |

**为什么 roofline 从"不做"改为"做"**：既然差异化押在"支持国产芯片"，那么只给一张显存容量表是**近似零差异化**的——任何计算器加一个下拉框就等价。国产芯片真正值得可视化的是它们与 NVIDIA **算力:带宽:互联比例不同**，因此同一个模型的瓶颈模块不同。要表达这一点，per-module 的 `算术强度 vs 芯片 ridge point` 是必需的。

**为什么并行策略必须进范围（v3）**：两条硬理由。

1. **不带并行的"放得下吗"对大模型无意义**。DeepSeek-V3 权重 641 GiB，任何单卡都放不下——不谈 TP/EP，结论恒为"放不下"，整个成本功能对最值得分析的模型直接失效。
2. **不谈互联，"多芯片瓶颈分析"是不完整的**。TP 每层要做 all-reduce，这个开销由互联带宽决定，而国产芯片与 NVIDIA 的 **互联:算力比例**差异往往比 HBM 带宽差异更大。也就是说：在国产卡上真正的瓶颈经常既不是算力也不是显存带宽，而是 **TP 通信**。漏掉这一项，等于漏掉了差异化的主要来源。

因此 §5.4(a) 里原本标注"可选，暂不参与计算"的 `interconnect` 字段升为**必需字段**（缺失时禁用通信相关判定，按 §5.4(c) 的按字段降级处理）。

**为什么拓扑骨架也改读真值（v4）**：v2/v3 的分工是「模板出骨架 + header 出数值」，其隐含代价是**骨架的覆盖度永远等于模板数量（当前 3 个家族）**。但 safetensors 的 key 就是 `state_dict` 的 key，也就是 `nn.Module` 的路径——对 key 建一棵 trie，直接得到**精确的含参模块树**，无需任何架构知识。模板真正独有的信息只剩三项：无参算子（SiLU / rope / 残差 / softmax）、执行顺序、语义与公式。把骨架也交给真值后，msv 对**任何**有 safetensors 的模型都能给出正确的树与正确的成本，模板缺失只表现为"图上少了无参算子、没有公式讲解"，而不是"整个模型看不了"。

**新的分界线**：做「**给定** parallel plan → 每卡资源投影 + 每层通信量 + bound 分类」；不做「**搜索** 最优 plan」与「端到端吞吐/延迟预测」。前者是已有 per-module 数据的一次投影，几乎零额外数据需求；后者需要 kernel 效率、调度、overlap 策略，msv 没有这些输入，给数字就是不诚实——**plan 搜索这件事 Vidur 已经做得很好，msv 不往那个方向长**（§7.4）。

**"不学 modelmap"的分界（v4 修订）**：

| 功能 | modelmap 做法 | msv 的选择 | 理由 |
|---|---|---|---|
| 节点数值来源 | 运行时假前向 trace（必须 torch） | **safetensors header 真值** | 比 trace 更轻**且**权重侧更准；这是 msv 相对 modelmap 的结构性优势 |
| 拓扑骨架 | trace 得到（含无参算子，但需 torch） | **header trie**（含参部分精确）+ 模板补无参算子 | 无 torch 前提下拿到精确骨架；代价是无参算子依赖模板 |
| 并行策略 | TP×PP stage 表 + planner（含搜索） | **given-plan 投影，含 EP/DP/PD 分离** | 取其资源分布计算，弃其 plan 搜索；补 modelmap 没有的 EP / DP-attention / PD 分离 |
| 吞吐预测 | roofline 端到端吞吐 | **只做 bound 分类，不给吞吐数字** | 缺 kernel 效率与 overlap 输入，给数字不诚实 |
| 权重精度 | 假设 dtype + what-if 旋钮 | **真实 dtype 分布**（API 直接给） + what-if 旋钮 | 无需假设 |
| 动画 | 脉冲回放 + 相机跟随 + beats | hover 联动 + 公式联动 | 重投入，非核心价值 |
| treemap | 自研 squarify | `d3-hierarchy`（需要时） | 可复用生态 |

**明确的能力边界（必须写死，否则 P2 会做到一半才发现）**：
per-module 可精确归因**权重内存**、**计算量**、**通信量**；**per-module 激活张量大小可算**（从 config 经 `dims.js` 推导），但**allocator 峰值占用不可算**（取决于 kernel fusion、就地操作、内存池复用）。因此 lens 展示"该模块的激活张量有多大"，**不声称**"该模块占了多少显存峰值"；模型级的 `activation_peak` 是一个带默认值、UI 可调的近似常数（不做标定）。

---

## 3. 设计总览：一条数据流串起所有能力

```
【真值侧 —— 骨架与数值都来自 checkpoint，不依赖模板】
HF Hub API  ?expand[]=safetensors ──→ 模型级精确 params + dtype 分布
safetensors index + header(range)  ──→ {张量名: {dtype, shape}}
                                            │
                                            ├─→ buildSkeleton：key 建 trie → 精确含参模块树
                                            └─→ 每节点真实 params / weight_shapes / dtype
                                                            │
【语义侧 —— 模板补 config 里才有、checkpoint 里没有的信息】                │
config.json ─→ normalizeConfig ─→ resolveArchitecture ─→ 层原语            │
                    （无参算子 SiLU/rope/残差、执行序、语义标签、公式绑定）  │
                                            │                            │
                                            └────────→ mergeSemantics ←───┘
                                                            ↓
                    IR v2：骨架(真值) + 数值(真值) + 无参算子/执行序/语义(模板)
                                                            ↓
   ┌──────────────────┬────────────────────────────┬──────────────────────────┐
materializer        cost/（新增）                   diagram lens + 公式联动
（现有视图，兼容）  ├ memory：权重(真值) + KV(解析) + 常数项(可调默认值)
                    ├ compute：per-module MACs（权重形状 × tokens）
                    ├ parallel：TP/PP/EP/DP 投影 → 逐卡分布（含 PD 分离一对 plan）
                    ├ comm：all-reduce / all-to-all / P2P / PD 的 KV 传输
                    └ roofline：三类 bound（算力 / 显存带宽 / 通信）× 效率因子

 ⇅ fallback（离线 / GGUF / gated / 无 safetensors）：模板公式推导，`value_source="derived"`
 ⇅ 后端 introspection（已有）：校验无参算子与执行序（骨架已由 trie 保证）
```

核心设计判断经过两次反转，最终形态是**"骨架与数值都取真值，模板只提供 checkpoint 里不存在的信息"**：

- v1 的判断：modelmap 靠 trace，所以 msv 必须自己从 config 推导一切。
- v2 的修正：数值不用推，`safetensors` header 是真值。
- **v4 的修正：骨架也不用推**。`safetensors` 的 key 就是 `state_dict` 的 key，也就是 `nn.Module` 路径——建 trie 即得精确含参模块树。模板真正独有的只剩三项：**无参算子、执行顺序、语义/公式**。

这个分工的实际收益：**模板数量不再决定"哪些模型能用"，只决定"哪些模型的图更好看、有公式讲解"**。

---

## 4. P0 图粒度升级（IR v2，数据基础）

### 4.1 Schema 变更

`src/model_structure_viewer/schemas.py` `StructureNode` 增加**可选**字段（向后兼容）：

```python
class StructureNode(BaseModel):
    id: str
    name: str
    type: str
    repeat: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_fields: list[str] = Field(default_factory=list)
    confidence: str = "high"
    children: list["StructureNode"] = Field(default_factory=list)
    # 新增（可选，None 表示未知）：
    params: int | None = None                 # 本节点自有参数（不含子树）
    weight_shapes: dict[str, list[int]] | None = None  # 数值形状，如 {"weight": [4096, 4096]}
    dtype: str | None = None                  # 实际 dtype：BF16/F8_E4M3/I32…
    input_shape: list[int] | None = None      # 数值 I/O；batch/seq 用 -1 占位
    output_shape: list[int] | None = None
    # v2 新增：数值来源标注，成本 UI 据此决定是否标"估算"
    value_source: str | None = None            # "checkpoint" | "derived" | "introspect"
    tensor_names: list[str] | None = None      # 绑定到本节点的 header 张量名
```

`value_source` 是 v2 的关键字段：同一棵树里可以混合真值节点与推导节点（例如 header 只覆盖了部分张量），UI 必须能区分，不能把推导值当真值展示。

前端 `materializers/toStructureNode.js` 同步产出这些字段（手写双份契约，见 §9 风险）。

### 4.2 数值与骨架来源：读 checkpoint 真值（v2 引入 / v4 扩到骨架）

**v1 的 params 公式表已作废。** 原方案要在每个层原语里按 config 数字算 `params` 与 `weight_shapes`，等价于在前端复刻 transformers 每个架构的 `__init__`——而这件事没有收敛终点：bias 有无、`tie_word_embeddings`、MLA 的 `q_a_proj/kv_a_proj_with_mqa/kv_b_proj` 三段式、`first_k_dense_replace`、共享专家、q/k norm、MTP 头、量化后 `qweight/qzeros/scales` 三张量替代一张 `weight`……每个都是一条 if，且随架构数线性增长。

**改用两个已验证的真值来源（均为零权重下载、浏览器可直接调用）：**

#### (a) 模型级：一次 HTTP 调用拿精确参数量与 dtype 分布

```
GET https://huggingface.co/api/models/{model_id}?expand[]=safetensors
```

实测结果（2026-09-04）：

```json
Qwen/Qwen3-8B
  {"parameters":{"BF16":8190735360},"total":8190735360}

deepseek-ai/DeepSeek-V3
  {"parameters":{"BF16":3918786560,"F8_E4M3":680571043840,"F32":41555600},
   "total":684531386000}
```

CORS 实测：响应头 `access-control-allow-origin` 回显请求 Origin → **静态站点可直接调用**。

这一条同时取消了 v1 §5.3(1) 的整条"实际 dtype 优先级链 + 默认 bf16 并明示假设"。精度差距的量级：DeepSeek-V3 按「总参数 × 2 bytes(bf16)」得 1275 GiB，按真实 dtype 分解得 **641 GiB**，相差 2 倍。

#### (b) 节点级：range read safetensors header 拿逐张量 dtype 与 shape

safetensors 格式 = 8 字节小端 u64 header 长度 + 该长度的纯 JSON（`{张量名: {dtype, shape, data_offsets}}`），无代码执行。

实测 `Qwen/Qwen3-8B`（2026-09-04）：

```
model.safetensors.index.json  → metadata.total_size=16381470720, weight_map 含 399 个张量
GET shard, Range: bytes=0-7   → HTTP 206, content-range: bytes 0-7/3996250744
                                 header JSON 长度 = 9328 bytes
GET shard, Range: bytes=8-9335 → 81 个张量：
    model.embed_tokens.weight               BF16 [151936, 4096]
    model.layers.0.self_attn.k_proj.weight  BF16 [1024, 4096]
    model.layers.0.mlp.gate_proj.weight     BF16 [12288, 4096]
    ...
```

三个关键性质：

1. **张量名就是 `nn.Module` 路径**：`model.layers.0.self_attn.k_proj.weight` = 模块路径 + 参数名。这不只是"能绑定"，而是"能直接建树"——见 (b2)。
2. **成本极低**：单分片 header 约 9 KB。因为层是重复的，**只读第一个分片就已覆盖全部不同的张量形状**（配合 fold 的代表层语义完全一致）。
3. **CORS 官方支持**：HF 在 `access-control-expose-headers` 中显式暴露 `Accept-Ranges, Content-Range` —— 范围请求是为浏览器场景设计的。

#### (b2) 骨架也来自 header：key 建 trie（v4 核心变更）

`safetensors` 的 key 集合 = `state_dict` 的 key 集合 = `模块路径 + "." + 参数名`。因此把每个 key 去掉最后一段、按 `.` 切分建 trie，**直接得到精确的含参模块树**：

```
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.mlp.gate_proj.weight
        ↓ 去参数名 + 切分 + 建 trie
model
└── layers
    └── 0
        ├── self_attn
        │   ├── k_proj   {weight: BF16[1024, 4096]}
        │   └── q_proj   {weight: BF16[4096, 4096]}
        └── mlp
            └── gate_proj {weight: BF16[12288, 4096]}
```

**这颗树是精确的，且零架构知识**——实现量级约 30 行。它带来四个连锁收益：

1. **覆盖度与模板解耦**：任何有 safetensors 的模型都能得到正确的树与正确的成本，不再受"只有 3 个模板家族"限制。模板缺失只表现为"少了无参算子、没有公式讲解"。
2. **量化与 MoE 变体自动正确**（已实测，2026-09-04）：

```
Qwen/Qwen3-32B-AWQ            → 量化的三张量是同一模块路径下的兄弟，trie 自然聚成一个节点
  model.layers.0.self_attn.k_proj.qweight
  model.layers.0.self_attn.k_proj.qzeros
  model.layers.0.self_attn.k_proj.scales
  model.layers.0.self_attn.k_norm.weight        ← 未量化的 norm 与之并存，同样自然

Qwen/Qwen3-30B-A3B（MoE，18867 个张量）→ 专家是编号子节点，天然是 ModuleList
  model.layers.0.mlp.experts.0.down_proj.weight
  model.layers.0.mlp.experts.1.gate_proj.weight
  model.layers.0.mlp.experts.10.up_proj.weight
  ...
```

   对照：**模板要正确处理这两种情况，必须为"量化 vs 非量化"、"专家融合（`w13_weight` 3D）vs 非融合"各写一套分支**，而且要知道每个后端用哪种。trie 不需要知道任何这些。

3. **`fold` 从数据直接解决**：折叠签名可以直接用「子树结构 + 真实 weight_shapes」的 hash，不再依赖模板的 `type + class`。G3 的修复变成数据驱动而非规则驱动。
4. **`layers.0/1/2/...` 与 `experts.0/1/...` 的重复结构一眼可见**：trie 的数字子节点天然就是 ModuleList，`repeat` 的识别不需要额外规则。

**实现注意（实测发现）**：大规模 MoE 的 `model.safetensors.index.json` 可能达到 MB 量级（Qwen3-30B-A3B 有 18867 个张量，index 约 1 MB）。仍远小于权重，但**不能当成"几 KB 的小请求"**——需要 loading 态，且建议在得到 trie 后立刻折叠再交给渲染层，不要把 18867 个节点直接送进 diagram。

**trie 拿不到的三样东西**（这才是模板的真正职责）：

| 缺失项 | 例子 | 由谁提供 |
|---|---|---|
| 无参算子 | SiLU / GELU、rope、残差相加、softmax、reshape | 模板（层原语声明） |
| 执行顺序 | norm → attn → norm → mlp 的先后 | 模板（trie 只有层级，无序） |
| 语义与公式 | "这个节点是 attention，它的公式是 …" | 模板 + `formulas/` |

**合并策略 `mergeSemantics`**：以 trie 为骨架，把模板声明的无参算子按执行序**插入**到 trie 节点之间，并给 trie 节点打语义标签。三种冲突处理：

- 模板声明了 trie 里没有的含参模块 → 模板过时（如模型换了实现），**以 trie 为准**，标注模板不匹配。
- trie 有模板未声明的模块 → 插入为 `generic` 节点，数值仍精确，**并计入"模板不完整"信号**（原 §4.4 的探测器，现在更强：不只报警，还能自动补全）。
- 模板不存在（非 3 个家族）→ 纯 trie 树 + 无语义，图仍可用、成本仍精确。

#### (c) 不要手写这一层

用 `@huggingface/hub` 的 `parseSafetensorsMetadata`（浏览器可用，支持 `computeParametersCount`）。格式本身简单，但**正确地把 header 换算成参数量不简单**，该库已处理的边界包括：

- 子字节量化的**打包容器宽度**：GPTQ/AWQ 的 4-bit 按 8 个打包进 `I32`，MXFP4 类方案按 2 个打包进 `U8`/`I8`。统一假设 32 位容器会把后者**少算 4 倍**。
- `bitsandbytes__` 量化状态前缀张量的排除。
- exponent-only dtype（`F8_E8M0` / `E8M0` / `UE8`）只承载 MX 类 block scale，从不是参数，必须排除。
- 分片 index 解析、`library_name` 提示下的主模块定位（diffusers 的 `transformer/`/`unet/`）。

> trie 建树本身自己写（30 行，无边界问题）；**参数量换算必须用库**（边界全在换算里）。两件事不要混。

#### (d) 代价与 fallback（必须诚实标注）

真值路径要求：**联网 + 目标是 HF 上有 safetensors 的 repo**。覆盖不到的场景与降级策略：

| 场景 | 降级 |
|---|---|
| 纯本地 `config.json` / 离线 | 模板公式推导（v1 §4.2 的表在这里保留使用），节点 `value_source="derived"` |
| GGUF-only repo | 同上；GGUF header 解析不在本期范围 |
| gated / 需 token 的 repo | 提示用户提供 token，否则降级 |
| 无模板的架构 | **骨架与数值仍精确**（trie 不依赖模板），仅缺无参算子、执行序与公式讲解 |

注意最后一行是 v4 的关键收益：**即使架构没有模板，只要能读到 header，树与成本数字都是精确的**——这是模板推导路径永远做不到的，也是 v2 版本（模板出骨架）做不到的。

#### (e) 前置依赖：前端直连 HF（G4）

`api/client.js` 目前所有 HF 调用都走后端。要吃到上述能力，需要新增前端直连路径（`frontend/src/api/hf.js`），走 `huggingface.co` 公开 API。这同时解决了 G6/G4 的连带问题：**静态部署形态从"只能看内置 catalog"变为"能看任意公开 HF 模型"**。

### 4.3 层原语与后端的职责调整

- **层原语（`model_executor/layers/*`）**：不再计算 `params`，**也不再负责含参模块的层级结构**。职责收缩为三项：声明**无参算子**（SiLU / rope / 残差 / softmax）、声明**执行顺序**、绑定**语义标签与公式**。此外声明本原语期望对应的含参模块名（供 `mergeSemantics` 对齐 trie 节点）。
- **`shapes.js`（G2）**：拆成 `dims.js`（返回数值数组，`batch`/`sequence` 用 `-1` 占位）与 `shapeText`（展示串，读 `dims.js`）。它服务于 KV 与**激活张量大小**的解析式计算；权重形状不走这条路（走 header）。
- **`introspect.py` `_walk`**：仍从 nn.Module 取真实 `params` / `weight_shapes`。用途缩为**离线场景的高质量 fallback** 与**无参算子/执行序的校验**（含参骨架已由 trie 保证正确）。
- **`fold.py:_signature`（G3）**：改为对「子树结构 + 真实 `weight_shapes`」做 hash，**不再依赖模板的 `type + class`**。trie 到位后这是数据驱动的修复，比规则驱动更可靠。

### 4.4 校验闭环（v4：骨架已是真值，校验对象改变）

含参骨架与数值都是真值，没有可校准对象。校验的对象变成"模板提供的那三项"与"常数项/效率因子"：

1. **无参算子与执行序校验**：`verification/transformers_verify.py` 对内置模型比对模板声明的执行序与 `nn.Module` 的实际 forward 顺序。
2. **模板完整性 + 自动补全**：`mergeSemantics` 时统计 trie 里模板未声明的模块，既作为"模板不完整"信号，又自动插入 `generic` 节点保证图不缺块。对任意模型免费生效。
> **不做实测校准（v6 确认）**：不建立任何与 vLLM 日志 / XProfiler / `xccl_perf` 的校准闭环。msv 的成本输出是**理论计算，结果供参考**（§5.1）。精度不校准；**但公式必须正确**——正确性由 §5.7 的清单与单测保证，不靠对比实测。

---

## 5. 成本规划（v6：推理 × per-module 归因 × 多芯片 × 并行策略 × PD 分离）

### 5.1 定位声明：理论分析能力，不是仿真

**这是本章最重要的一条，先于所有公式。**

msv 的成本模块提供的是**理论分析能力**（first-order analytical model）：从 checkpoint 真值与 config 出发，用解析公式给出量级正确、结构清晰、假设可见的成本分解与瓶颈归因。

- **不是仿真器**：不建模调度、批处理动态、kernel 选择、overlap、内存分配器行为。要仿真请用 Vidur（§7.4）。
- **不是预测器**：不输出吞吐 / 延迟 / TTFT 数字。三个 `*_time` 只用于互相比大小得出 bound 分类。
- **不承诺精度**：所有假设（效率因子、常数项、ring 上界、无 overlap）在 UI 上可见且可调，但**不追求逼近实测**。用户看到的是"按这套假设，账是这么算的"，而不是"实际会是这个数"。

**"准确"与"正确"要分开看** —— 这条决定了哪些测试值得做：

| 类别 | 例子 | 态度 |
|---|---|---|
| **精度**（可以不准） | 效率因子取 0.7 还是 0.65；`activation_peak` 取 1.2 GB 还是 1.5 GB；ring vs tree all-reduce | **不投入**。给文献默认值 + 可调旋钮 + 明示假设即可 |
| **正确性**（不能错） | `kv / TP` 写成 `kv / min(TP, kv_heads)`；MLA 当成可切分；漏掉 `2×`；权重 dtype 用错 | **必须守住**。倍数级错误会**改变 bound 分类结论本身**，那不是精度问题而是 bug |

因此测试只针对第二类，且范围就是 **§5.7 的公式正确性清单**（17 条，标"是"的才写单测）。不写精度测试、不建对比套件。

### 5.2 范围

**负载类型：只做推理。** prefill 与 decode 分开算（两者瓶颈性质相反）。**不做训练**——不含优化器状态、梯度、激活重算。

**核心问题不是"放得下吗"**——那件事十几个免费网页计算器已经做完了，用户填 5 个数字比打开 msv 更快，msv 在那条赛道上没有结构性优势。

**msv 要回答的是**：*这个模型在**这批卡**上，按**这个并行策略**部署，**哪个模块**是瓶颈，瓶颈在**算力、显存带宽还是互联通信**，换一张卡（尤其是国产芯片）或换一个并行策略，瓶颈会**移到哪里**。*

这些都必须以图为载体、以 per-module 数据为基础，因此：
- 唯一有结构性优势的能力是 §4 打通的「节点 ↔ checkpoint 真值」+「多芯片规格」+「并行投影」三者的组合。
- 模型级总量（fit / max_context）**降为图的自然汇总**，用最小实现，不作为主功能宣传。

**做**：给定 parallel plan（TP / PP / EP / DP 的组合）→ 每卡资源投影、每层通信量、per-module 三类 bound 分类。

**不做**：**搜索**最优 parallel plan、端到端吞吐/延迟预测、训练内存规划。前者需要一个 solver，后者需要 kernel 效率与 overlap 策略——msv 都没有这些输入，给数字就是不诚实。

**能力边界（已在 §2 写死，此处重申）**：per-module 可精确归因**权重内存**、**计算量**、**通信量**；**per-module 激活张量大小可算**（`dims.js` 从 config 推导），但**allocator 峰值占用不可算**（取决于 kernel fusion、就地操作、内存池复用）。lens 展示"该模块激活张量多大"，不声称"该模块占了多少显存峰值"。

### 5.3 公式

#### (1) 权重字节 —— 用真值，不用假设

```
模型级：weight_bytes = Σ_dtype  params[dtype] × bytes_per_element(dtype)
                       ← HF API ?expand[]=safetensors 直接给出 params[dtype]

节点级：node_weight_bytes = Σ_tensor  prod(shape) × bytes_per_element(dtype) / packing_factor
                       ← safetensors header 直接给出每个张量的 shape 与 dtype
```

- `bytes_per_element`：`F32=4, BF16/F16=2, F8_E4M3/F8_E5M2=1, I8=1`；子字节量化必须除 `packing_factor = container_bits / num_bits`（见 §4.2(c)），**不能一律按 int4=0.5**。
- 交给 `@huggingface/hub` 的 `parseSafetensorsMetadata(computeParametersCount: true)` 处理，msv 不自己实现这段换算。
- what-if 精度旋钮仍保留（"如果换成 int4 会怎样"），但**默认展示真值**，旋钮是叠加而非替代。

#### (2) KV cache —— 解析式（此项无真值来源，必须算）

```
kv_bytes_per_token = 2 × num_kv_heads × head_dim × num_layers × kv_bytes_per_element
kv_bytes_at(T, B)  = kv_bytes_per_token × T × B
```

- MLA（DeepSeek 系）走压缩路径：`kv_lora_rank + qk_rope_head_dim`，KV 显著小于 dense MHA/GQA。`normalize.js` 已能推导 `headDim`，**需补 `kv_lora_rank` / `qk_rope_head_dim` 归一化**。
- `kv_bytes_per_element` 独立于权重 dtype（fp8 KV cache 是常见配置），单独一个旋钮。
- 这条公式全行业收敛，无争议。

#### (3) per-module 计算量 —— MACs（差异化的基础）

从节点的真实 `weight_shapes` + config 数字推导，**不需要运行时 trace**：

```
Linear / matmul（weight rank 2 或 3）：
    macs = tokens × prod(weight_shape) × expert_frac
    tokens = B × T（prefill）或 B × 1（decode，单步）

Attention core（非权重项）：
    macs = B × num_heads × T² × (d_qk + d_v)        # prefill，O(T²)
    macs = B × num_heads × T   × (d_qk + d_v)       # decode 单步，O(T)

Elementwise（norm / 激活 / rope / 残差）：
    other = prod(output_shape)                     # 用 §4.3 的 dims.js

MoE：
    expert_frac = experts_per_token / experts       # 逐层用 layerSchedule，非全局一刀切
```

**prefill 与 decode 必须分开算**：两者的瓶颈性质完全相反（prefill 算力密集、decode 带宽密集），而这正是不同芯片表现分化的地方。v1 没有区分这一点，是个实质缺口。

#### (4) 瓶颈判定 —— 算术强度 vs 芯片 ridge point（三类 bound，带效率因子）

```
per-module 时间估（仅用于互相比大小，不作绝对值展示）：
    compute_time = 2 × macs      / (peak_flops(dtype) × η_flops)
    memory_time  = bytes_moved   / (memory_bandwidth  × η_hbm)
    comm_time    = comm_bytes    / (link_bandwidth    × η_comm)

    bytes_moved = node_weight_bytes + act_in_bytes + act_out_bytes

bound = argmax(compute_time, memory_time, comm_time)

等价的 roofline 表述（单卡维度，便于图上表达）：
    AI    = 2 × macs / bytes_moved                              # 算术强度 (FLOP/Byte)
    ridge = (peak_flops × η_flops) / (memory_bandwidth × η_hbm)  # 有效脊点
    AI < ridge → memory-bound;  AI > ridge → compute-bound
```

**为什么必须有效率因子（v4 引入，v5 保留）**：这**不是为了准**，而是为了消除**系统性偏差**——峰值互比不是"均匀地乐观"，而是各项乐观程度不同。llm-analysis 明确说明其输出是"下界估计，因为假设了永远达不到的峰值性能"，并给出文献区间：**FLOPS 效率推理约 0.7**（训练约 0.5），**显存带宽效率 0.9 是激进目标**。互联侧达成率更低（节点内约 0.7–0.8，跨节点 RoCE 常在 0.5–0.6）。

代入一算就知道后果：`η_hbm/η_comm ≈ 0.9/0.6 = 1.5`，意味着**用峰值互比会把 comm_time 相对低估约 1.2–1.6 倍**——刚好足以把一个真正 comm-bound 的模块误判成 compute/memory-bound。而 comm-bound 恰恰是国产芯片场景最重要的结论。所以这属于 §5.1 表格里的**"正确性"一类而非"精度"一类**：不加因子会改变分类结论。

**默认值（文献取值，不做实测校准）**：

| 因子 | 默认值 | 依据 |
|---|---|---|
| `η_flops` | 0.7 | llm-analysis 文献值（推理） |
| `η_hbm` | 0.9 | llm-analysis "激进目标" |
| `η_comm` | 0.6（跨节点）/ 0.8（节点内） | RoCE / NVLink 常见达成率 |

三个因子在 UI 上作为**可见且可调的假设**呈现（不是隐藏常量），并可写进 `chips.local.json` 按卡覆盖。**但 msv 不组织"实测校准"活动**——按 §5.1 的定位，把这三个数从 0.7 调到 0.68 不产生任何价值。用户想调就调，这是旋钮不是待办。

注意范围克制：**只做 bound 分类，不输出吞吐/延迟数字**。三个 `*_time` 仅用于互相比大小以定性分类，绝对值不展示——它们不含调度与 overlap，绝对值没有意义。

#### (5) 总内存与 fit —— 两个近似常数项，不是百分比

```
per-card total = weight_bytes_per_card              # 真值 ÷ 并行切分，见 (6)
      + kv_bytes_per_card(T, B)                     # 解析式 ÷ 并行切分，见 (6)
      + activation_peak(B, chunk_size)              # 近似常数，与 T 弱相关
      + runtime_const(backend, chip)                # CUDA/XPU context + graph pool，1.0–2.5 GB 量级
      + comm_buffer(world_size, backend)            # NCCL/XCCL 缓冲，随 world_size 增长

max_context(B=1) = (单卡显存 − 上述常数项) / kv_bytes_per_token_per_card
fit              = 每张卡都放得下（不是总量放得下）
```

`fit` 的定义在引入并行后变了：**必须逐卡判定**，PP 各 stage 与 EP 各 rank 的负载不同，"总显存够"不等于"能跑"。

三个常数项按 §5.1 的定位处理：**给一组文献/经验默认值 + UI 可调 + 明示假设，不做实测标定**。默认量级建议 `activation_peak ≈ 1–2 GB`、`runtime_const ≈ 1–2.5 GB`、`comm_buffer` 随 `world_size` 线性小项。**它们的作用是让"权重+KV 之外还有一块"这件事在账上可见，不是给出准确数值。**

**v1 的"固定 15% overhead"作废，依据如下（这是评审中证据最硬的一处）**：

注意这条**不是精度问题而是结构问题**——15% 是按比例，而这几项本质是常数，两者随模型规模的走向完全相反：

vLLM 自身的启动日志分解为四项，且激活峰值对上下文长度近似不敏感：
```
model weights 6.02GiB; non_torch_memory 0.05GiB;
PyTorch activation peak 1.19GiB   (max_model_len=4096)
PyTorch activation peak 1.26GiB   (max_model_len=16384)   ← 上下文 ×4，激活几乎不变
```
原因是 chunked prefill 把激活峰值压在 chunk 尺度上；CUDA Graph 缓冲池是另一个独立的 flat 1.5–2 GB 项。两者都**不与「权重 + KV」成比例**。

后果的量级：DeepSeek-V3 权重约 641 GiB，按 15% 会凭空加出约 96 GiB；反过来一个 4B 小模型按 15% 又远不足以覆盖 runtime context。另有公开校准数据直接记录："15% + 600MB 的初版假设对实测偏差 55.2%，改成 3% + 75MB 才对得上"——**这恰好证明该系数不是常量，而是被拟合出来的**，不能当公式用。msv 不需要那个拟合值，只需要一个不会随模型规模发散的结构。

同时必须承认 vLLM 官方 RFC #27951 指出的失准来源：MoE 专家负载不均衡（热门专家可直接 OOM）与 CUDA Graph 预留，使得**任何解析式估算在 MoE + CUDA Graph 场景下都只能给量级**。这与 §5.1 的定位一致：msv 给的是理论账，不是保证值，UI 上必须这么说。

#### (6) 并行策略投影（v3 新增，推理场景）

输入一个 plan：`{ TP, PP, EP, DP, attn_mode: "tp" | "dp" }`，约束 `TP × PP × DP = world_size`（EP 通常复用 TP×DP 的卡集合，具体由后端决定，msv 按用户给定值算并明示假设）。

**(6.1) 权重切分**

| 模块 | TP 切分 | 说明 |
|---|---|---|
| q/k/v_proj | ÷ TP（列并行） | — |
| o_proj | ÷ TP（行并行） | 触发 all-reduce |
| MLP gate/up | ÷ TP（列并行） | — |
| MLP down | ÷ TP（行并行） | 触发 all-reduce |
| MoE 专家 | ÷ EP（不按 TP） | 每卡 `ceil(experts / EP)` 个专家；**负载不均衡 → 必须给区间** |
| 共享专家 | ÷ TP | 与 dense MLP 同 |
| embedding / lm_head | ÷ TP（vocab 并行）**或**复制 | 取决于后端配置，需一个开关；`tie_word_embeddings` 时不重复计数 |
| norm | 复制（不切） | 参数量小，但别忘了计 |

PP 不改变单模块大小，只改变**归属**：每 stage 承担 `layers / PP` 层（首尾 stage 因 embedding / lm_head 而**不对称**，必须分别算）。

**(6.2) KV cache 切分 —— 三个高频错误点（实现时最容易算错的地方）**

```
GQA / MHA：
    kv_bytes_per_card = kv_bytes / min(TP, num_kv_heads)      ← 不是 / TP
```

- **错误 1：GQA 下 KV 不总是除以 TP。** `num_kv_heads = 8` 而 `TP = 16` 时，KV 头不够分，后端会复制（或补齐）→ 实际因子是 8，不是 16。写成 `/ TP` 会**低估显存 2 倍**。
- **错误 2：MLA 的 KV 无法按 TP 切分。** DeepSeek 系的 MLA 每 token 只有一个压缩 latent（`kv_lora_rank + qk_rope_head_dim`），没有头维度可切 → **TP 下全量复制**。这正是 DeepSeek 采用 DP-attention + EP-MLP 的原因。按 `/ TP` 算会把 DeepSeek 的容量**高估 TP 倍**。
- **错误 3：DP-attention 下 KV 不共享。** `attn_mode: "dp"` 时每个 DP rank 持有**自己那批序列的完整 KV**，总 KV = `DP × per-rank KV`，单卡 KV 不随 DP 下降。

**(6.3) 通信量（解析式，精确）**

每个 transformer 层的 TP all-reduce 出现在 o_proj 与 MLP-down 之后，共 2 次：

```
ring all-reduce 单次传输量 = 2 × (TP−1)/TP × B × T_eff × hidden × act_bytes
每层 comm_bytes           = 2 × 上式
每 token decode comm      = 取 T_eff = 1

EP 的 all-to-all（MoE dispatch + combine）：
    comm_bytes ≈ 2 × B × T_eff × experts_per_token × hidden × act_bytes

PP 的 stage 间 P2P：
    comm_bytes = B × T_eff × hidden × act_bytes × (PP − 1)   # 远小于 TP，通常不是瓶颈
```

对比 `interconnect.bandwidth` 得 `comm_time_est`，与 (4) 的 compute/memory 时间估比大小 → 得出 comm-bound 判定。

**这里是国产芯片差异化的核心落点**：TP all-reduce 量只与 `B × T × hidden` 有关，**与模型是否 MoE、参数量多大无关**。因此在算力强但互联弱的卡上，decode 阶段（`T_eff = 1`，计算量极小但通信量不变）极易变成 comm-bound。这个结论在通用 VRAM 计算器里完全看不到。

**(6.4) 明示假设（不能装作精确）**

- 不建模 **overlap**（通信与计算重叠）——因此 comm-bound 判定偏保守，UI 上必须写明"未考虑 overlap"。
- 不建模 all-reduce 的实际算法选择（ring / tree / NVLS 等），统一按 ring 上界。
- 跨节点与节点内分别用 `interconnect.inter_node` / `intra_node`；只有一个值时按该值统一计算并标注"偏乐观"。
- MoE 的 EP 负载不均衡按 `[平均, 最坏]` 给区间，最坏情况取"热门专家集中在单卡"。
- 效率因子 `η_flops` / `η_hbm` / `η_comm` 是可见假设，不是隐藏常量（§5.3(4)）。

#### (7) PD 分离（v4 新增，P2e）

PD 分离（Prefill/Decode disaggregation）下 prefill 与 decode 跑在**不同卡组、不同并行策略**上，因此 plan 从单个扩为一对：

```
plan = {
  prefill: { chip, world_size, TP, PP, EP, DP, attn_mode },
  decode:  { chip, world_size, TP, PP, EP, DP, attn_mode },   // 可用不同型号的卡
  transfer: { link: "inter_node", kv_dtype_bytes }
}
```

**新增的唯一一项计算：KV 传输量**

```
每请求 KV 传输字节 = kv_bytes_per_token(decode 侧布局) × prompt_len
传输时间估          = 上式 / (inter_node.bandwidth × η_comm)
与 prefill 计算时间比较 → 判断该配置是"传输受限"还是"prefill 算力受限"
```

三个必须注意的点：

1. **KV 布局要按 decode 侧的切分算**，不是 prefill 侧——KV 是给 decode 用的，`min(TP_decode, num_kv_heads)` 才是分母。若两侧 TP 不同，还存在**跨布局重排**的额外代价（msv 只标注存在该代价，不估算其大小）。
2. **MLA 在这里反而是巨大优势**：MLA 每 token 只传一个压缩 latent，KV 传输量比 GQA 小一个数量级。**"MLA vs GQA 在 PD 分离下的传输代价对比"是本项最有说服力的输出**，建议作为 P2e 验收标准。
3. **prefill 与 decode 可以是不同芯片**，这直接支持"prefill 上大算力卡、decode 上大显存/高带宽卡"这类实际配比问题——通用计算器完全不覆盖。

**不做**：调度策略、KV 传输的 overlap/流水（Mooncake / NIXL 那一层）、多副本负载均衡。msv 只回答"这个配比下，传输量与 prefill 算力量的比例是多少，瓶颈在哪一侧"。

### 5.4 芯片规格表（国产芯片支持 —— 差异化载体）

#### (a) 需要的字段

只有显存容量是不够的（那等于给计算器加个下拉框）。瓶颈分析需要四类数据：

```js
{
  id: "kunlun-p800", vendor: "昆仑芯", name: "...",
  memory_bytes: ...,            // 显存容量 —— fit 判定
  memory_bandwidth: ...,        // B/s —— ridge point 的分母
  peak_flops: {                 // 逐 dtype 峰值算力 —— ridge point 的分子
    bf16: ..., fp16: ..., fp8: ..., int8: ...
  },
  interconnect: {               // v3：必需字段（原为"可选，暂不参与计算"）
    intra_node: { kind: "C2C"|"NVLink"|"PCIe", bandwidth: ... },
    inter_node: { kind: "RoCE"|"IB", bandwidth: ... }            // 可缺，缺则按 intra_node 统一计
  },
  source: "https://...",        // 数据出处（必填，见 (c)）
  confidence: "official" | "vendor-marketing" | "community"
}
```

`peak_flops` 必须逐 dtype：国产芯片在不同精度上的算力比例与 NVIDIA 差异很大，这正是"换卡后瓶颈会移动"的成因。

`interconnect` 在 v3 从可选升为必需：TP all-reduce 是否成为瓶颈完全由它决定，而**国产芯片与 NVIDIA 的「互联:算力」比例差异往往比 HBM 带宽差异更大**——漏掉这一项等于漏掉差异化的主要来源。缺失时按 §5.4(c) 的按字段降级处理（禁用 comm-bound 判定，保留 compute/memory 判定与 fit）。

#### (b) 覆盖范围

- NVIDIA 侧：H20 / H100 / A800 / A100 / L40S / RTX 40-50 系（作为对照基线，不可省——没有基线就看不出国产芯片的相对特征）。
- 国产侧：昆仑芯 P800 系、昇腾 910 系、寒武纪思元系、海光 DCU、摩尔线程等，按可获得的公开数据逐步加。

**现实预期**：国产芯片的公开渠道经常缺 `memory_bandwidth` 或逐 dtype `peak_flops`。这不是暂时状态，而是本期必须正面设计的常态——见 (c) 的「缺项即协作入口」。

#### (c) 数据来源与合规（已拍板 2026-09-04）

**策略：公开规格入库，非公开规格走用户配置，缺项做成显式的协作入口。** msv 定位是可静态公开部署（GitHub Pages），因此分两层 + 一个缺项机制：

**第一层：`chips/public.js`（入库）**

只放公开可查的规格，每条 entry 必须带溯源字段：

```js
{ id, vendor, name,
  memory_bytes, memory_bandwidth, peak_flops: {bf16, fp16, fp8, int8},
  interconnect: { intra_node: {kind, bandwidth}, inter_node: {kind, bandwidth} },
  source: "https://...",                                  // 必填，可核实的公开链接
  confidence: "official" | "vendor-marketing" | "community" }
```

- `official` = 厂商官方白皮书/规格页；`vendor-marketing` = 发布会/宣传材料（峰值常为理论上限）；`community` = 第三方实测汇总。三者在 UI 上必须可区分，不混为一谈。
- 没有 `source` 的 entry 不允许合入——这条既是合规要求，也是数据质量闸门。
- **`source` 允许 per-field**：一张卡的显存来自官网、带宽来自发布会，应当分别标注，而不是给整张卡打一个笼统等级。建议 `source` / `confidence` 支持字段级覆盖（顶层为默认值，字段级可单独指定）。

**第二层：`chips.local.json`（配置，不入库）**

- 加入 `.gitignore`；仓库只提供 **schema + 加载器 + 一份 `chips.local.example.json`**。
- **示例文件必须用明显虚构的数字**（`"id": "example-chip"`，显存/带宽/算力用 `100e9` / `1e12` 这类整数占位）。**不要写"P800 示例：显存约 XX GB"**——哪怕数字是编的，也会被当成真实规格传播出去。
- 加载时机：运行时可选加载，与 `public.js` 合并，同 `id` 时本地覆盖公开值。
- UI 上本地来源的芯片打独立标记（如 `local`），避免用户误以为是仓库自带的公开数据。
- 另提供**手动录入入口**：填四个数字（显存 / 显存带宽 / 算力 / 互联带宽）即可临时新增一张卡，不必落文件。这样任何未公开的芯片都能用，而 msv 自己不持有任何非公开数据。

**缺项机制：缺项即协作入口（已拍板，方案 2）**

字段缺失是常态而非异常，因此按字段（不是按卡）降级，并把缺口显式暴露成可补充的邀请：

| 缺失字段 | 仍可用的能力 | 禁用的能力 | UI 表现 |
|---|---|---|---|
| 无 | 全部 | — | 正常 |
| `interconnect.bandwidth` | fit、max_context、compute/memory bound、通信量绝对值 | **comm-bound 判定**、并行策略对比 | 该卡标"缺 互联带宽"，并行视图给出补充入口 |
| `interconnect.inter_node` | 全部（按 `intra_node` 统一计） | — | 标注"跨节点按节点内带宽估算，偏乐观" |
| `memory_bandwidth` | 显存 fit、max_context、per-module 权重占比 | memory-bound 判定 | 该卡标"缺 带宽"，roofline 视图灰显并给出补充入口 |
| `peak_flops[dtype]` | 同上 + 其他已有 dtype 的 roofline | 该 dtype 的 bound 分类 | 该 dtype 选项灰显 |
| `memory_bytes` | 仅 per-module 相对占比 | fit / max_context | 该卡不进 fit 对比 |

- **绝不用估计值填空**。宁可少一个功能，不要给出看似精确的错数字。
- 每个缺失字段旁给一个明确的补充路径（指向仓库的芯片数据贡献说明 + 要求附 `source`）。国产芯片的公开数据在逐年变多（厂商发布会、白皮书、第三方评测），把这个位置做成可 PR 的表，比自己闷头攒更可持续。
- `source` + `confidence` 这套字段本来就是为可核查设计的，正好直接支撑社区补充，不需要额外机制。

> 分层的收益：msv 仓库永远不需要为"某张卡的规格准不准"负责——公开数据有 `source` 可核，非公开数据由使用者自己提供并自己承担，缺失数据公开标注为缺失。

#### (d) 可选的真值校准（用户特有能力，非本期范围）

昆仑侧有 XProfiler 可拿到**实测算子耗时**。理论上可以用它校准 msv 的 per-module 估算（对比 estimated MACs 与 measured kernel time，反推有效算力）。这会让 msv 的昆仑侧数据从"解析式估算"升级为"经验校准"，是一个别人无法复制的能力。

注意它天然属于**第二层**：校准出的有效算力是实测数据，应该落在 `chips.local.json` 而非入库。**不列入本期**，仅记录为后续方向。

### 5.5 模块结构（新目录 `frontend/src/cost/`）

```
weights.js      — 调 @huggingface/hub 取 params[dtype] 与 header
skeleton.js     — safetensors key 建 trie → 精确含参模块树（v4 核心，约 30 行）
mergeSemantics.js — trie 骨架 + 模板的无参算子/执行序/语义 合并；产出模板完整性信号
memory.js       — node_weight_bytes / kv_per_token / activation_tensor_bytes / 三个常数项
compute.js      — macs_of（linear / attention / elementwise）+ prefill|decode 两种模式
parallel.js     — plan 校验 + 权重/KV 按 TP/PP/EP/DP 投影 + 逐卡分布 + PD 一对 plan
comm.js         — TP all-reduce / EP all-to-all / PP P2P / PD 的 KV 传输量
efficiency.js   — η_flops / η_hbm / η_comm 默认值与按芯片覆盖
roofline.js     — 有效脊点、三类 bound 分类（compute / memory / comm）
aggregate.js    — 遍历 IR v2 自底向上汇总，复用 fold 的 repeat 乘数 → 图即账本
chips/
  public.js                  — 公开规格表（每条必带 source / confidence，支持字段级覆盖；无 source 不合入）
  loadLocal.js               — 可选 chips.local.json 加载器 + 同 id 覆盖合并（不入库）
  chips.local.example.json   — schema 示例，明显虚构的占位数字
  coverage.js                — 按字段计算某张卡可用/禁用的能力 + 缺项补充提示
assumptions.js  — what-if 状态（T / B / phase / kv dtype / weights dtype / chip / plan / η）
__tests__/
  kv-sharding.test.js         — §5.3(6.2) 三个错误点的回归（唯一必须维护的测试）
```

设计要点：
1. **成本从图节点算，不从 config 公式算**：遍历 IR v2 自底向上汇总，复用 fold 的 `repeat` 乘数 → 图与账本永远一致，不会出现"图显示的和算的不是一回事"。
2. **MoE 用现成 `layerSchedule`** 逐层算 active params/token，比 modelmap 的全局一刀切更准。
3. **真值与推导严格分色**：读 `value_source` 字段，推导节点在 UI 上必须可辨识。
4. **`parallel.js` 是纯投影层**：输入「未切分的 per-module 成本」+ plan，输出「per-card per-module 成本」。它不产生新数据，只做除法与归属划分——因此可以完全独立单测，且不会污染 §4 的真值链。**KV 切分的三个错误点（§5.3(6.2)）必须有单测覆盖**：GQA `TP > num_kv_heads`、MLA 不可切、DP-attention 不共享。这是**唯一值得投入测试的地方**（§5.1 的"正确性"一类）。
5. **`skeleton.js` 与 `weights.js` 严格分开**：建树自己写（无边界问题），参数量换算必须用库（边界全在换算里）。
6. **`efficiency.js` 只是一张默认值表 + 覆盖逻辑**，不是校准框架。别把它做大。

### 5.6 UI

- **P2（主）成本 lens**：diagram 节点叠加编码——颜色编码 bound 类型（compute / memory / comm / 未知），尺寸或饱和度编码成本占比。切换 chip / phase(prefill|decode) / plan 时**颜色格局整体变化**，这就是"换卡或换并行策略，瓶颈会移动"的直观表达。
- **P2 芯片对比**：并排两张卡，只高亮 bound 类型发生翻转的模块。
- **P2b+ 并行策略对比**：同一张卡下并排两个 plan（如 `TP=8` vs `TP=4,DP=2`），高亮翻转模块。**最有说服力的单一视图建议是**：国产卡 + decode 阶段下，TP 增大时模块从 memory-bound 翻成 comm-bound——这个结论通用计算器完全给不出来，建议作为 P2b 的验收标准。
- **每卡视图**：PP 各 stage / EP 各 rank 的显存柱状图（含首尾 stage 不对称、EP 负载区间），`fit` 逐卡判定而非看总量。
- **P1（轻量）汇总条**：权重 / KV / 激活 / runtime / comm-buffer 五段分解 + 逐卡 fit + max_context + what-if（T / B / phase / chip / dtype / plan）。刻意做小，因为它不是差异化所在。
- **全局必须有一处定位声明**（不是脚注，是显眼位置）：**"理论估算，非仿真、非预测；不承诺与实测一致"**，外加当前生效的假设摘要（η 三值、ring 上界、无 overlap、常数项取值）。这是 §5.1 定位的 UI 落地——它比任何精度改进都重要，因为它决定用户会不会误用这些数字。
- 需要独立 treemap 时引 `d3-hierarchy` 的 squarify，不自研。

### 5.7 公式正确性清单（v6 —— 本方案唯一的质量闸门）

精度不校准（§5.1），**但公式必须正确**。这里逐条列出必须正确的公式、常见错法、对应的 llm-analysis 参照、以及是否需要单测。

**编码约定**：每个公式的实现处必须写一行注释指明对应的 llm-analysis 函数，格式如：

```js
// ref: llm-analysis LLMAnalysis.get_memory_kv_cache_per_layer
// 注意 min(TP, num_kv_heads)：TP 超过 KV 头数时无法继续切分
```

llm-analysis 的对应位置在 `llm_analysis/analysis.py` 的 `LLMAnalysis` 类（下表函数名以其公开 API 为准，若版本有差异按语义对齐）。

| # | 公式 | 常见错法（会改变结论） | llm-analysis 参照 | 单测 |
|---|---|---|---|---|
| F1 | `weight_bytes = Σ_dtype params[dtype] × bytes(dtype)` | 用总参数 × 单一 dtype（DeepSeek-V3 会差 2 倍） | `get_num_params_total` / `get_weight_memory_per_layer` | 否（用 HF API 真值） |
| F2 | 子字节量化 `packing_factor = container_bits / num_bits` | 一律按 int4=0.5（MXFP4 少算 4 倍） | — | 否（用 `@huggingface/hub`） |
| F3 | `kv_per_token = 2 × kv_heads × head_dim × layers × kv_bytes` | 漏 `2×`（K 和 V）；用 `attention_heads` 代替 `kv_heads` | `get_memory_kv_cache_per_layer` | **是** |
| F4 | MLA 的 KV 走 `kv_lora_rank + qk_rope_head_dim` | 按 dense 公式算（DeepSeek 会差一个数量级） | —（llm-analysis 未覆盖 MLA，需自行推导） | **是** |
| F5 | `kv_per_card = kv / min(TP, kv_heads)` | 写成 `/ TP`（GQA + 大 TP 低估 2 倍+） | `get_memory_kv_cache_per_gpu` | **是** |
| F6 | MLA 在 TP 下**不可切分，全量复制** | 按 `/TP` 算（高估容量 TP 倍） | — | **是** |
| F7 | DP-attention 下每 rank 持有自己序列的完整 KV | 按 `/DP` 算 | — | **是** |
| F8 | Linear MACs `= tokens × prod(weight_shape) × expert_frac` | 漏 `expert_frac`（MoE 高估 E/K 倍）；tokens 用错（prefill `B×T` vs decode `B×1`） | `get_num_flops_fwd_per_layer_*` | **是** |
| F9 | Attention core MACs：prefill `B×heads×T²×(d_qk+d_v)`，decode `B×heads×T×(...)` | prefill/decode 用同一式（T 大时差 T 倍） | `get_num_flops_fwd_per_layer_attn` | **是** |
| F10 | `flops = 2 × macs` | 漏 `2×`（乘加各算一次） | 同上 | 否（显然） |
| F11 | ring all-reduce `= 2 × (TP−1)/TP × B × T_eff × hidden × bytes`，每层 **2 次** | 漏 `2×(N-1)/N` 系数；漏"每层两次"（o_proj 与 mlp-down） | `get_latency_fwd_per_layer_tp_comm` | **是** |
| F12 | EP all-to-all `≈ 2 × B × T_eff × experts_per_token × hidden × bytes` | 用 `experts` 而非 `experts_per_token` | —（自行推导） | **是** |
| F13 | `tie_word_embeddings` 时 lm_head 不重复计参数 | 重复计（小模型显著偏大） | `get_num_params_embedding` | 否（trie 天然不重复） |
| F14 | PP 首尾 stage 不对称（含 embedding / lm_head） | 按 `layers/PP` 平均分（首尾 stage 会 OOM） | `get_memory_weight_per_stage` | **是** |
| F15 | PD 的 KV 传输量按 **decode 侧** 布局算 | 按 prefill 侧算 | — | **是** |
| F16 | `expert_frac` 逐层取（用 `layerSchedule`），非全局一刀切 | 全局一个值（dense-MoE 混合架构会错） | — | **是** |
| F17 | 三类 bound 各自除以对应 η | 用峰值互比（comm 相对低估 1.2–1.6 倍，见 §5.3(4)） | `flops_efficiency` / `hbm_memory_efficiency` / `*_node_memory_efficiency` | **是** |

**测试组织**：全部集中在 `frontend/src/cost/__tests__/`，用 `node --test`（仓库已有该约定，见 `frontend/package.json` 的 `"test": "node --test"`）。**只测上表标"是"的项，不测精度。**

> 上表就是"公式一定要正确"的可执行形式。**F3–F7 是最高危区**（KV 与其切分），四条各自的错法都会让"放得下吗"给出反向结论。

---

## 6. 轻量动画（P3）

### 6.1 交互设计（采 Netron / TensorBoard / GNN-101 模式）

- **hover 节点 → 高亮执行序上的前驱/后继 + 边强调**（粗细 / 虚线 / 透明度编码关系强度）。依据：Netron、TensorBoard Graph Dashboard 均以此为标准交互（hover 显示 shape 与操作类型）。
- **公式 ↔ 节点双向联动（msv 差异化）**：hover 算子的公式说明 → 高亮图中对应节点，反之亦然。依据：GNN 101 的"formula ↔ visualization"双向联动设计——这是 msv 有 `formulas/` 系统而其他工具没有的独特点。
- **Layers 列表 ↔ diagram 联动**：点列表 → 图中高亮对应层（GNN 101 的 overview ↔ detail 模式）。

### 6.2 明确不做

modelmap 式脉冲回放（rAF 引擎 + 相机跟随 + HUD 逐步解说 + beats 脚本）——重投入，偏离"结构查看"定位。msv 的手写 diagram（`StructureDiagram/layout/viewport`）直接加事件即可，无需引擎。

---

## 7. 联网调研与选型依据

### 7.0 二次审视的结论：哪些该借，哪些不该长

系统审视后的定位判断（v4）：

| 能力 | 成熟方案 | msv 的取舍 |
|---|---|---|
| 逐 dtype 参数量 / 逐张量 shape | HF Hub API + `@huggingface/hub` | **直接依赖**，不自研 |
| 含参模块树 | safetensors key 本身 | **自己建 trie**（30 行，无边界） |
| 内存/延迟解析公式（含 TP/PP/SP/DP + 效率因子） | **llm-analysis** | **借公式形态与效率因子设计**；实现处注释指向其对应函数（§5.7） |
| plan 搜索 / 高保真吞吐预测 | **Vidur**（MLSys'24, MIT, <9% 误差, Vidur-Search） | **不做**，文档中明确指向它 |
| 分层图渲染 + per-node data overlay | Google Model Explorer | **退路**，不作主路径（§7.3） |
| **per-module 成本叠加在结构图上 + 国产芯片 + PD 分离** | **无成熟方案** | **msv 的全部差异化** |

结论：msv 唯一不可替代的是最后一行。上面每一行凡有成熟方案的，都应"借"而不是"造"——尤其是解析公式，自己推容易错、且错了没人发现。

### 7.1 数值真值来源（v2 主路径，均已实测验证）

| 来源 | 关键结论 | 在 msv 中的角色 |
|---|---|---|
| HF Hub API `?expand[]=safetensors` | 返回逐 dtype 精确参数量；CORS 回显 Origin，静态站点可直调（2026-09-04 实测，见 §4.2(a)） | **模型级权重字节的真值** |
| [safetensors 官方格式](https://github.com/safetensors/safetensors) | header = 8 字节 u64 长度 + 纯 JSON（dtype/shape/data_offsets），无代码执行；**key 即 `state_dict` key 即 `nn.Module` 路径** | 节点级 shape/dtype 的真值 + **骨架来源**（§4.2(b2)） |
| [`@huggingface/hub` `parseSafetensorsMetadata`](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts) | 浏览器可用；已处理分片 index、smart range request、子字节量化打包容器宽度（GPTQ/AWQ 4-bit→`I32` 8 个/容器 vs MXFP4→`U8` 2 个/容器）、`bitsandbytes__` 前缀排除、exponent-only dtype（`F8_E8M0`/`E8M0`/`UE8`）排除 | **直接依赖，不自己实现换算** |
| [GGUF 格式解析](https://mbrenndoerfer.com/writing/gguf-format-quantized-llm-storage-inference) | GGUF 自描述 header 含架构与张量表 | 后续方向，本期不做 |

**关键判断**：v1 把 header range read 列为"可选增强 P0+"，理由是"零依赖"。实测后修正为 **P0 主路径 + 必须用现成库**——格式简单不等于换算简单，上表第三行列出的边界每一条都是"手写一定会算错"的。

### 7.2 成本公式与 overhead

| 来源 | 关键结论 | 对 msv 的启示 |
|---|---|---|
| [vLLM RFC #27951 内存 profiling 失准](https://github.com/vllm-project/vllm/issues/27951) | 两年前公式对 dense + 无 CUDA Graph 尚准；MoE 专家不均衡与 CUDA Graph 预留使其失准，`gpu_memory_utilization` 语义已被扭曲 | **解析式估算在 MoE + CUDA Graph 下只能给区间** |
| vLLM 启动日志实测（社区帖） | `weights + non_torch(0.05GiB) + activation peak(1.19→1.26GiB, T 从 4096→16384)`；激活峰值对 T 近似不敏感 | **推翻固定 15% overhead**；改两常数项 |
| [Modular LLM Inference Handbook — KV cache](https://handbook.modular.com/inference-optimization/kv-cache-offloading) | `KV = 2·B·S·L·H·D·(Q/8)`，与 msv 采用式一致 | KV 公式一手依据 |
| 公开校准数据（AgenticWire LLM VRAM Calculator） | "15%+600MB 假设对实测偏差 55.2%，改 3%+75MB 才吻合" | 证明该系数是拟合值而非常量 |
| [selfhostllm](https://github.com/erans/selfhostllm)、localllm.in 等计算器 | 均为"填数字 → 一个总量"形态，**无一支持 per-module 分解** | msv 在成本上唯一有意义的差异点 |

> v1 引用的 AIMultiple / Spheron / HardwareHQ / inferenceengineering.tech 均为 SEO 内容站，非一手来源，且其 overhead 结论已被上表实测数据推翻，故从依据中移除。它们的 KV 公式部分正确但可被 Modular handbook 替代。

**与 modelmap 的对照**：`analytics.py` 的 `plan_serving` 提供 TP×PP stage 表 + roofline 吞吐，服务其"精确部署规划"定位。msv 取其 roofline 的 **bound 分类**部分（因为多芯片对比需要），弃其 stage 表与吞吐预测（超出结构查看定位，且 msv 无相应输入）。

### 7.3 图交互 / 动画 / lens 编码

| 来源 | 关键结论 | 对 msv 的启示 |
|---|---|---|
| [Google AI Edge Model Explorer](https://github.com/google-ai-edge/model-explorer)（Apache-2.0，`ai-edge-model-explorer-visualizer` npm） | 分层 graph + 逐层展开折叠、**show identical layers**（≈ msv fold）、**per-node custom data overlay 自动配色 + 侧栏聚合统计**（≈ msv P2 lens）、WebGL + web worker 惰性布局、搜索 / 子图跳转 | **P2 的成熟备选**，见下方决策 |
| [Model Explorer 研究博客](https://research.google/blog/model-explorer) | 层级布局按需计算，用 instanced rendering + MSDF 支撑万级节点 60FPS | 手写 diagram 的性能上限参照 |
| [GNN 101](https://arxiv.org/html/2411.17849v3) | 公式 ↔ 可视化双向联动；层概览 ↔ 明细联动 | msv `formulas/` 系统的最佳用法（P3 差异化） |
| [Netron / TensorBoard 图可视化](https://blog.tomsawyer.com/tensorflow-graph-visualization) | hover 显示 shape/dtype，点击高亮并联动 info card | hover 高亮为标准模式 |
| [Tom Sawyer 节点图可视化指南](https://blog.tomsawyer.com/node-graph-visualization) | 颜色编码类别、尺寸编码数值、边样式编码关系 | lens 的编码方式 |

**Model Explorer 的取舍决策（不采用，但记录理由）**：它现成提供 P2 lens 与 P3 大部分渲染能力，能省掉可观的手写工作量。不采用的原因有三条实质冲突：

1. **抽象层不匹配**：它是 op 级调试器（TFLite / MLIR / PyTorch ExportedProgram），msv 的价值在"架构语义 + 公式解释"这个更高抽象层。
2. **custom data 只支持 op 节点，不支持 layer 节点**——而 msv 恰恰要在**折叠后的层节点**上叠成本，正好错位。
3. **公式 ↔ 节点双向联动**（msv 唯一真正的差异化交互）在第三方组件里做双向绑定会非常别扭。

结论：**列为 P2 的退路**。如果手写 lens 的成本超出预期（尤其是命中测试与大图性能），转投 Model Explorer 是一条经过验证的路径，届时代价是放弃第 3 点。

### 7.4 解析式成本模型的成熟实现（v4 新增 / v6 收口）

| 来源 | 关键结论 | 在 msv 中的角色 |
|---|---|---|
| [**llm-analysis**（cli99）](https://github.com/cli99/llm-analysis) | 覆盖 TP / PP / SP / DP 的内存 + 延迟解析模型，训练与推理均支持；**显式暴露 `flops_efficiency` / `hbm_memory_efficiency` / `intra_node_memory_efficiency` / `inter_node_memory_efficiency`**；文档明说输出是"下界估计，因为假设峰值性能（永远达不到）"；文献参考值：FLOPS 效率训练 ~0.5、**推理 ~0.7**，显存效率 0.9 为激进目标 | **公式与效率因子方法论的来源**；§5.7 清单逐条注明对应函数 |
| [**Vidur**（MSR, MLSys'24, MIT）](https://github.com/microsoft/vidur) | 离散事件仿真 + 算子级 profiling，端到端延迟误差 <9%；**Vidur-Search 自动搜索最优并行/批/调度配置**（LLaMA2-70B 一小时 CPU 完成，替代 42K GPU 小时） | **msv 明确不往这个方向长**。用户需要仿真/搜索时，文档直接指向它 |
| [RAPID-LLM（arXiv:2512.19606）](https://arxiv.org/html/2512.19606v2) | tile 级建模覆盖六种并行轴；指出 llm-analysis 的默认设定在某些 DP 场景会引入 all-gather 下界导致偏差 | 提醒：**任何解析模型的默认参数都需针对场景核对**，不可盲信 |
| [harleyszhang/llm_counts](https://github.com/harleyszhang/llm_counts) | 同类实现，params/flops/memory/latency，同样带 efficiency 因子 | 交叉参照 |

**"借公式"的边界（v5 收口）**：

- msv 的公式必须自己实现（前端 JS，无 torch，静态部署 —— 无法调用 Python 库）。
- **借的是公式形态与效率因子的三/四因子设计**，不是借它的精度。llm-analysis 自己就说输出是下界估计，msv 与它对齐的是"下界 + 定性"这个定位，不是数值。
- **不做对比测试（v6）**：不建 oracle 测试套件、不存 fixture、不进 CI。理由：按 §5.1 的"准确 vs 正确"划分，对比套件守的是精度，而 msv 不承诺精度。
- **代之以 §5.7 的公式正确性清单**：逐条列出必须正确的公式、常见错法、对应的 llm-analysis 函数、是否需要单测。**这是本方案唯一的质量闸门。**
- **编码约定**：每个公式的实现处写一行 `// ref: llm-analysis <函数名>` 注释。未来怀疑某个公式时有据可查，成本近乎零。

三条来源共同确认的一件事：**解析式模型的定位是"下界 + 定性"，不是"预测"**。这与 msv "只做 bound 分类，不给吞吐数字"的边界一致 —— 但需要在 UI 上说清楚，否则用户会当预测用。

---

## 8. 阶段路线图与依赖（v4 重排 / v6 微调）

| 阶段 | 内容 | 涉及文件 | 依赖 | 可独立交付 |
|---|---|---|---|---|
| **P0a** | 前端直连 HF（G4）+ HF API 拿逐 dtype 参数量 | 新 `api/hf.js`、`client.js` | — | ✅ |
| **P0b** | safetensors header 接入（`@huggingface/hub`）+ **`skeleton.js` trie 建树** + `mergeSemantics` + IR v2 schema | 新 `cost/{weights,skeleton,mergeSemantics}.js`、`schemas.py`、`materializers/`、层原语改为只声明无参算子/执行序/语义 | P0a | ✅ |
| **P0c** | `shapes.js` 拆 `dims.js` + `fold.py:_signature` 改为子树结构 + 真实 shape 的 hash | `model_executor/shapes.js`、`structure/fold.py` | P0b | ✅ |
| **P1** | 单卡内存/计算量核心 + 轻量汇总条（prefill / decode 分开）+ **定位声明与假设摘要 UI** | 新 `cost/{memory,compute,aggregate}.js`、`normalize.js` 补 `kv_lora_rank`/`qk_rope_head_dim` | P0b | ✅ |
| **P2a** | 芯片规格表（公开数据 + `coverage.js` 按字段降级 + 本地覆盖加载器 + 贡献说明） | 新 `cost/chips/*`、`CONTRIBUTING` 芯片数据一节 | — （可与 P0/P1 并行） | ✅ |
| **P2b** | 效率因子 + roofline 三类 bound 分类 + 图上 lens + **双卡对比视图** | 新 `cost/efficiency.js`、`cost/roofline.js`、`diagram` 渲染 | P0c、P1、P2a | ✅ |
| **P2c** | **并行投影**（`parallel.js`：TP/PP/EP/DP 权重与 KV 切分 + 逐卡 fit + 每卡视图） | 新 `cost/parallel.js` | P1 | ✅ |
| **P2d** | **通信量 + comm-bound**（`comm.js`）+ 并行策略对比视图 | 新 `cost/comm.js`、`roofline.js` 扩三类、`chips` 补 `interconnect` | P2b、P2c | ✅ |
| **P2e** | **PD 分离**：plan 扩为一对 + KV 传输量 + 两侧不同芯片 | `cost/parallel.js`、`cost/comm.js` 扩展 | P2c、P2d | ✅ |
| **P3** | 轻量动画（hover 联动 + 公式↔节点双向联动） | `diagram`、`formulas` 联动 | P0c | — |

**实施顺序**：**P0a→P0b 必须先行**——成本数字的价值全部来自真值，先做一个基于推导的 P1 等于先造一个要被替换掉的东西。P0a 单独可交付且立刻有可见收益（静态部署从"只能看内置模型"变为"能看任意公开 HF 模型"）。

**P0b 的内部顺序（v4）**：先 `skeleton.js`（trie 建树，纯函数，可对任意模型立刻验证），再 `mergeSemantics`。trie 单独就能让 msv 支持任意 HF 模型出树 —— 这是整个方案里**投入产出比最高的一步**（约 30 行换来覆盖度从 3 个家族到全部）。

**P2c 可与 P2b 并行**：`parallel.js` 是对 P1 结果的纯投影（除法 + 归属划分），不依赖 roofline 与渲染，可以先写纯函数 + 单测，UI 后接。

**P2a 的内部顺序**：**先写 `coverage.js`，再填 `public.js` 的数据**。反过来做的话，很容易为了让功能跑起来而给缺失字段填一个"差不多的数"，降级逻辑就形同虚设了。

**验收标准**（按 §5.1 的定位，都是"能得出正确的定性结论"，不是"数值贴近实测"）：
- **P0b**：任选一个**没有模板**的架构（如 Llama / Gemma / GLM），能出正确的模块树与精确的参数量。这条直接证明"覆盖度与模板解耦"。
- **P1**：五段分解 + 逐卡 fit 能跑通；**UI 上有显眼的"理论估算，非仿真非预测"声明与假设摘要**。不设"与实测偏差 < X%"这类指标。
- **P2b**：并排两张卡（建议 H20 vs 昆仑芯 P800），能看出 bound 类型发生翻转的模块。
- **P2c**：DeepSeek-V3（权重 641 GiB，单卡必然放不下）能在给定 plan 下算出逐卡 fit 结论。**这条同时是"为什么并行必须进范围"的证明**。
- **P2d**：国产卡 + decode 阶段，TP 增大时能看出模块从 memory-bound 翻成 comm-bound。
- **P2e**：**MLA（DeepSeek）vs GQA（Qwen）在 PD 分离下的 KV 传输量对比**，能看出 MLA 小一个数量级。这是最有说服力的单一输出，且无同类工具覆盖。

---

## 9. 风险与权衡

| 风险 | 说明 | 缓解 |
|---|---|---|
| **真值路径的可用性边界** | 精确数值与精确骨架都要求联网 + HF 上有 safetensors 的 repo；离线 / GGUF-only / gated 覆盖不到 | 模板推导作为 fallback，`value_source` 字段区分，UI 明示"估算" |
| **倍数级公式错误（不是精度问题）** | `/TP` 写错、`2×` 漏掉，会**改变 bound 分类结论本身**；输出仍是个"看起来合理"的数字 | **§5.7 公式正确性清单**（17 条，逐条注明常见错法 + llm-analysis 参照 + 是否需单测）；F3–F7 是最高危区 |
| **KV 切分算错（最高风险）** | 三个高频错误：GQA 下 `TP > num_kv_heads` 时不能除以 TP（低估 2 倍+）；MLA 的 KV 无法按 TP 切分、必须复制（高估 TP 倍）；DP-attention 下 KV 不共享 | §5.3(6.2) 已写明公式；**三种情况必须各有单测**（§5.5 要点 4） |
| **峰值互比导致 bound 误判** | 用峰值算三类时间会把 comm 相对低估 1.2–1.6 倍，可能把 comm-bound 判成 compute-bound——正好毁掉国产芯片场景最重要的结论 | 引入 `η_flops`/`η_hbm`/`η_comm` 三个可见效率因子（§5.3(4)）。**这是消除结构性偏差，不是追求精度**——因子本身取文献默认值即可 |
| **过度追求精度反而偏离定位** | "再校准一下就更准了"是无底洞，且 msv 不承诺精度 | §5.1 写死定位声明；效率因子与三个常数项只给默认值 + 旋钮，**不组织实测校准**；验收标准不设偏差百分比 |
| **通信估算过于乐观/悲观** | 不建模 overlap → 偏保守；按 ring 上界 → 忽略 tree/NVLS 优化；跨节点带宽常缺 | 所有并行输出带一行假设声明；`inter_node` 缺失时标注"按节点内带宽估算，偏乐观" |
| **EP 负载不均衡** | 热门专家集中在单卡可直接 OOM（vLLM RFC #27951） | 按 `[平均, 最坏]` 给区间，不给单点值 |
| **plan 合法性** | 用户可能输入 `TP×PP×DP ≠ world_size`、`EP > experts`，或 PD 两侧 TP 不同导致 KV 需跨布局重排 | `parallel.js` 前置校验并给出可读错误，不静默算错；跨布局重排只标注存在、不估算 |
| **国产芯片规格数据缺失** | 多数国产芯片不公开带宽、逐 dtype 峰值算力**或互联带宽**，且这是常态非异常 | **按字段（非按卡）降级** + 缺项做成显式补充入口（§5.4(c)）；**不用估计值填空** |
| **公开版与本地版体验分叉** | 你的 local 数据不入库 → 别人打开时国产卡可能大面积"未知"，核心差异化看不到 | 已选方案 2：公开版放官方已公开数字，缺项显式标注并邀请补充（附 `source`）；公开数据逐年变多，可 PR 的表比闷头攒更可持续 |
| **示例文件被误当真实规格** | `chips.local.example.json` 若写近似真值会被传播 | 示例必须用 `example-chip` + 明显虚构整数占位 |
| **合规：非公开规格不得入库** | msv 可静态公开部署 | 双层策略：公开数据带 `source` 入库；非公开走 gitignore 的 `chips.local.json` + 手动录入入口。msv 仓库不持有任何非公开数据 |
| **激活峰值不可 per-module 归因** | allocator 峰值取决于 kernel fusion / 就地操作 / 内存池复用 | lens 只展示"激活张量大小"（可算），不声称"占了多少峰值"；模型级 `activation_peak` 是一个可调的默认常数，不做标定 |
| **解析模型被当预测用（最需防的误用）** | 用户看到数字就会当实际值用 | **UI 显眼处必须有"理论估算，非仿真非预测"声明 + 生效假设摘要**（§5.6）；不展示时间绝对值 |
| **范围蔓延到 planner / 仿真** | 一旦能算 given-plan，很容易被要求"推荐最优 plan"或"预测吞吐" | §2 已写死分界：做投影不做搜索、做分类不做预测。**这两件事指向 Vidur**（§7.4），msv 不长那个方向 |
| **模板覆盖度**（影响大幅缩小） | v4 后模板只影响无参算子、执行序、公式讲解 | **骨架与数值都不受影响**——trie 不依赖模板（§4.2(b2)），这是 v4 的核心收益 |
| **模板不完整的探测** | 模板漏声明过去无法发现 | `mergeSemantics` 统计 trie 里模板未声明的模块，既报警又自动补 `generic` 节点（§4.4） |
| **手写 diagram 的性能与工作量** | 前端零图库，lens 命中测试与大图性能全自研 | 退路：转 Model Explorer（§7.3），代价是放弃公式双向联动 |
| **契约双份手写** | `schemas.py` 与前端 materializer 手工同构，无版本号 | 延续现状；可选演进为 schema-first 代码生成（另立议题） |
| **fold 误折叠** | 当前签名不含 shape | P0c 必修，且 v4 后改为数据驱动（子树结构 + 真实 shape hash） |
| **无静态部署配置**（G6） | 无 `.github/workflows/` | P0a 交付时一并补 |

---

## 10. 已决策 / 仍待确认

**已决策（2026-09-04）**：
1. 成本目标 = per-module 计算量 + 内存 + **多芯片（含国产芯片）**，不是通用 VRAM 计算器。
2. 数值来源 = safetensors header 真值。
3. v1 §4.2 的 params 公式表作废，改读 checkpoint 真值（降级为离线 fallback）。
4. **芯片规格三段策略**：公开规格入库（必带 `source` + `confidence`，支持字段级覆盖）；非公开规格走用户配置（`chips.local.json`，gitignore；另提供手动录入入口）；**缺项按字段降级并做成显式的社区补充入口**（方案 2）。详见 §5.4(c)。
5. **负载类型只做推理**（prefill / decode 分开），不做训练。
6. **并行策略进入范围**：做 given-plan（TP / PP / EP / DP）的每卡资源投影与每层通信量，**不做 plan 搜索与吞吐预测**。连带：`interconnect` 升为必需字段；bound 分类扩为三类（compute / memory / comm）。
7. **拓扑骨架也读真值**：safetensors key 建 trie → 精确含参模块树；模板收缩为无参算子 + 执行序 + 语义/公式。覆盖度与模板数量解耦（§4.2(b2)）。
8. **三类 bound 引入显式效率因子** `η_flops`/`η_hbm`/`η_comm`，取文献默认值（§5.3(4)）。**目的是消除结构性偏差，不是追求精度。**
9. **借 llm-analysis 的公式形态与效率因子设计**；**不建对比测试套件**，代之以 §5.7 公式清单 + 实现处 `// ref:` 注释；**plan 搜索与仿真指向 Vidur，msv 不做**（§7.4）。
10. **PD 分离进范围（P2e）**：plan 扩为 `{prefill, decode}` 一对 + KV 传输量，两侧可用不同芯片（§5.3(7)）。
11. **定位收口（v5）**：msv 成本模块是**理论分析能力**，不是仿真、不是预测、**不承诺精度**（§5.1）。
    - 效率因子与三个常数项只给默认值 + UI 旋钮 + 明示假设，**不组织实测校准**。
    - **取消** vLLM 日志 / XProfiler / `xccl_perf` 的一切校准动作（v6 连"可选抽查"也不保留）。
    - 验收标准全部改为"能得出正确的定性结论"，**不设与实测的偏差百分比指标**。
    - 守住的底线是**倍数级正确**（`/TP`、`2×`、MLA 不可切这类），由 KV 切分单测覆盖 —— 这属于正确性而非精度（§5.1 的"准确 vs 正确"表）。

**仍待确认**（下一个会话开工时顺手定，都不阻塞 P0）：
1. **§5.3(6) 的 plan 输入形态**：手填 `{TP, PP, EP, DP}` 是否够？还是支持粘贴 vLLM / SGLang 启动参数自动解析？建议做，但放在 P2c 之后作为独立小增量。
2. **§5.3(6.1) embedding / lm_head 的切分开关**：建议默认按 vLLM 行为（vocab-并行）并暴露开关。
3. **§7.3**：Model Explorer 作为退路的判断（先手写，性能撑不住再换）。
4. **§6.1** 公式↔节点双向联动的投入优先级。v4 后它的价值更高——公式讲解是模板**唯一**不可替代的产出，也是⑤算子级可解释的核心。

---

## 11. 开发交接（下一个会话从这里开工）

### 11.1 先读什么

1. **§0.1 定位判据** —— 任何实现决策先过那张表，过不去就是越界。
2. 本节 11.2 的第一步任务。
3. 需要背景时再回看：§4.2(b2)（trie 建骨架，最核心的技术判断）、§5.1（成本定位）、§5.7（公式清单）。

### 11.2 第一步：`skeleton.js`（强烈建议单独交付）

**为什么先做这个**：约 30 行，换来覆盖度从 3 个模板家族到"任何有 safetensors 的模型"。它是全案投入产出比最高的一步，且与成本侧完全解耦。

```
输入：safetensors 的 key 列表（+ 每个 key 的 {dtype, shape}）
处理：key 去掉最后一段（参数名）→ 按 "." 切分 → 建 trie
输出：模块树，每个叶节点带 {参数名: {dtype, shape}}
```

关键实现点（均已在 §4.2(b2) 实测验证）：
- 数字路径段（`layers.0`、`experts.3`）即 ModuleList，直接对应 `repeat`。
- 同一模块下的多个参数（`weight` / `qweight` / `qzeros` / `scales` / `bias`）聚在同一节点，不拆。
- 大 MoE 的 `index.json` 可达 MB 量级（Qwen3-30B-A3B 18867 个张量约 1 MB）→ 需要 loading 态，且**建完 trie 立刻折叠再交给渲染层**，不要把 18867 个节点送进 diagram。

**验收**：拿一个**没有模板**的架构（Llama / Gemma / GLM 任选）出正确的模块树与精确参数量。

**前置**：P0a（前端直连 HF，`frontend/src/api/hf.js`）。当前 `api/client.js` 所有 HF 调用走后端，静态部署只能看内置 catalog。

### 11.3 阶段顺序（详见 §8）

```
P0a 前端直连 HF ──→ P0b skeleton.js + mergeSemantics + IR v2 ──→ P0c dims.js 拆分 + fold 签名
                                    ↓
                          P1 内存/计算量 + 汇总条
                                    ↓
        ┌───────────────────────────┴──────────────┐
   P2b 效率因子 + roofline + lens            P2c 并行投影
        └───────────────┬──────────────────────────┘
                   P2d 通信量 + comm-bound  ──→  P2e PD 分离
P2a 芯片表（可全程并行，内部先写 coverage.js 再填数据）
P3 hover 联动 + 公式↔节点双向联动（依赖 P0c）
```

### 11.4 五条不可违反的约定

1. **不做实测校准**。效率因子与三个常数项给默认值 + UI 旋钮，就停手（§5.1）。
2. **公式实现处必须写 `// ref: llm-analysis <函数名>` 注释**（§5.7）。
3. **只测 §5.7 标"是"的项**，不写精度测试。F3–F7（KV 与其切分）是最高危区。
4. **UI 显眼处必须有"理论估算，非仿真非预测"声明 + 生效假设摘要**（§5.6）。这是放弃精度的必要配套。
5. **`chips/public.js` 每条必带 `source`**，无 source 不合入；非公开规格只走 gitignore 的 `chips.local.json`（§5.4(c)）。

### 11.5 已验证的外部事实（无需重新调研）

- `GET https://huggingface.co/api/models/{id}?expand[]=safetensors` → 逐 dtype 精确参数量；CORS 回显 Origin，静态站点可直调。
- safetensors header = 8 字节小端 u64 长度 + 纯 JSON；单分片 header 约 9 KB；HTTP 206 + `Accept-Ranges`/`Content-Range` 已在 CORS 白名单。
- key 形态（实测）：`model.layers.0.self_attn.k_proj.weight`；AWQ 为 `k_proj.{qweight,qzeros,scales}`；MoE 为 `mlp.experts.{i}.{gate,up,down}_proj.weight`。
- 用 `@huggingface/hub` 的 `parseSafetensorsMetadata`（浏览器可用）做参数量换算，**不要手写**（子字节打包容器宽度、`bitsandbytes__` 前缀、exponent-only dtype 等边界都在里面）。

---

## 附：参考代码位置

- 前端管线：`frontend/src/structure/buildStructure.js`、`config/normalize.js`、`model_executor/**`、`ir/createStructureIr.js`、`materializers/toStructureNode.js`
- 后端：`src/model_structure_viewer/structure/{introspect,recovery,fold,semantics}.py`、`schemas.py`、`service.py`、`verification/transformers_verify.py`
- 现有 schema：`src/model_structure_viewer/schemas.py:12-20` `StructureNode`
- 形状推导（需拆两层）：`frontend/src/structure/model_executor/shapes.js`
- 折叠签名（需加 shape）：`src/model_structure_viewer/structure/fold.py:145-152` `_signature`
- HF 调用现状（需加直连路径）：`frontend/src/api/client.js`
- 架构模板（3 个真实家族）：`frontend/src/structure/model_executor/models/{qwen,deepseek,minimax}.js`
