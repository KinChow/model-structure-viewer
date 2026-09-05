# Model Structure Viewer 前端入口设计

> 状态：已实现，持续优化
> 日期：2026-09-04
> 范围：首页模型入口、模型详情页信息层级和成本配置入口；成本公式与数据协议仍需单独确认。

## 1. 产品定位

Model Structure Viewer 的首页是模型结构浏览入口。用户选择或加载模型后，进入一个以结构图为主的模型详情工作区。

入口页应参考 modelmap 的精简信息层级和 vLLM Recipes 的 Provider / Model 分类方式，但不复制其模型目录产品，也不增加动态热门模型服务。

## 2. 已确认的入口

首页只保留三条核心路径：

1. 输入 / 选择模型
2. 打开本地模型目录
3. 按 Provider 浏览内置模型

### 2.1 输入 / 选择模型

这是默认入口，统一支持：

- 直接输入 Hugging Face 模型 ID 或地址。
- 直接输入 ModelScope 模型 ID 或地址。
- 从内置模型中选择。
- 输入框下方可提供内置模型的快速选择，但不单独建立“内置模型”入口。

用户不需要理解 `source`、`endpoint`、`cache policy` 等实现参数。它们属于高级设置或内部配置，不应占据首页主入口。

### 2.2 打开本地模型目录

入口名称使用“打开本地模型目录”，而不是“导入 config”或“打开本地文件”。

用户选择目录后，由程序自动识别可用文件，例如：

- `config.json`
- 权重索引
- `safetensors` / `bin` 权重文件
- tokenizer 文件
- 自定义 `modeling_*.py` 文件

根据实际文件显示解析能力和数据可信度。`config.json` 是目录识别结果，不再作为独立的首页入口。

### 2.3 按 Provider 浏览

Provider / Model 浏览位于统一模型入口下方，是额外的模型选择方式，不是唯一入口。

默认先展示 Provider 名称、图标和模型数量，例如：

- MiniMax
- Qwen
- DeepSeek
- 其他 Provider

点击 Provider 后，背景进入高斯模糊遮罩，在前景弹层中选择具体模型。弹层标题、模型列表和模型元信息必须与用户点击的 Provider 对应。

Provider 目录使用仓库内的静态模型清单。第一阶段不请求 Hugging Face 热门模型接口，不新增热门模型后端代理或同步任务。

## 3. 明确删除的入口和内容

以下内容不出现在首页入口区：

- 独立的 `config.json` 导入入口。
- 独立的“内置模型”入口。
- “最近使用”模型。
- “示例模型 / Examples”。
- Hugging Face 动态热门模型。

当前不保存最近使用模型，因此不引入 Cookie，也不实现 `localStorage` 最近记录。

## 4. 视觉与语言

### 4.1 视觉

- 整体采用黑底、橙色强调、高对比的品牌化视觉。
- 可切换深色和浅色主题。
- 入口页保持精简，避免额外的结构示意、说明卡片和无关内容。
- 主行动是打开模型，Provider 和本地目录是次级选择路径。

### 4.2 语言

- 支持中文和英文。
- 顶部提供明确的 `中 / EN` 切换。
- 品牌名始终使用完整名称 `Model Structure Viewer`，不使用 `MSV` 作为用户可见主标题。
- 帮助入口使用明确的“帮助 / Help”，不使用含义不清的 `?` 作为唯一标识。

## 5. 后端边界

后端继续保持轻量，只保留模型验证相关能力，包括：

- 本地模型验证。
- Transformers 加载验证。
- 结构解析相关验证。

不增加以下服务：

- Hugging Face 热门模型抓取代理。
- 热门模型定时同步和缓存服务。
- 最近模型账号或 Cookie 服务。
- Provider 目录的重型动态服务。

## 6. 入口验收标准

- 首屏第一主动作是输入或选择模型。
- 输入模型与内置模型使用同一个入口，不重复展示两套入口。
- 本地路径入口明确打开模型目录。
- 首页没有独立 `config.json`、最近模型或示例模型区域。
- Provider 区域位于统一模型入口下方。
- Provider 卡片只展示 Provider 名称、图标和数量。
- 点击 Provider 后显示模糊遮罩和对应的模型列表。
- Qwen、DeepSeek、MiniMax 等 Provider 的模型列表不能串用。
- 中英文切换后主要入口文案可用，中文和英文不互相覆盖成难以理解的混合文本。
- 深色和浅色主题下文字、边框和主按钮均清晰可读。
- 移动端入口不发生横向溢出，模型输入和路径按钮保持可用。

## 7. 开发顺序

1. 用现有预览确认入口布局和交互。
2. 将入口结构接入现有 React 前端。
3. 接入内置模型目录和 Provider 静态清单。
4. 接入本地模型目录选择及现有模型加载流程。
5. 加入主题和中英文切换。
6. 通过现有页面验证脚本和前端构建检查。

## 8. 模型详情页设计

模型加载成功后，使用单一详情工作区，不把主要内容拆成多个独立页面：

```text
顶部模型身份与摘要
中央 Architecture 结构图
右侧 Node Inspector
底部 Cost Lens 与当前方案摘要
```

详情页应参考 modelmap 的信息融合方式：结构图是主内容，节点参数、Shape、权重和成本通过 Inspector 按需查看。Layers、Raw Config、Export 等功能可以作为结构图工具栏或 Inspector 的附加操作，不应把用户强制带到多个页面之间切换。

顶部摘要展示模型名称、Provider、模型类型、解析状态、总参数、激活参数、层数、Hidden Size、Experts 和 Context 等关键事实。

## 9. 成本 Lens 与配置

### 9.1 成本 Lens 多选

Cost Lens 支持多个指标同时开启，区别于 modelmap 的单选方式：

- VRAM：权重、KV Cache 和激活的显存汇总。
- Compute：MACs / FLOPs 等计算量。
- Memory：激活内存。
- KV Cache：每 token 和目标上下文下的 KV 占用。

`None` 用于清空所有 Lens。多选用于同时展示多个指标，不把多个指标合并成一个颜色。模块类型和最终瓶颈仍使用独立的视觉表达。

### 9.2 机器配置唯一入口

底部只展示当前方案摘要，不再提供第二个芯片选择器：

```text
H20 × 8 · TP8 · fits 612 / 768 GB · 配置
```

所有机器和并行参数统一从“配置”打开，避免底部 Machine 选择和配置弹层重复修改同一状态。

集中式配置包括：

- GPU / 芯片
- 单卡显存
- GPU 数量
- 节点数
- TP / PP / EP / DP
- Attention 并行方式

### 9.3 PD 分离

成本配置支持运行模式切换：

```text
[集中式] [PD 分离]
```

集中式表示 Prefill 和 Decode 在同一套机器节点资源上统一部署，使用一套部署规模和并行策略。PD 分离表示 Prefill 和 Decode 共享机器规格定义，但使用相互独立、可单独扩展的节点数量、GPU 数量和并行策略；DP 用于表达数据并行规模，不再额外设置“副本数”。

集中式不显示 KV Transfer。PD 分离配置面板通过 `Prefill / Decode` 切换，一次只显示一个阶段的部署规模和并行策略；KV Transfer 仅在 PD 分离模式中显示。

PD 分离模式分别配置 Prefill 和 Decode：

- Prefill：节点数、GPU 数、TP、PP、EP、DP。
- Decode：节点数、GPU 数、TP、PP、EP、DP。
- KV Transfer：节点间带宽、Transfer backend。

成本结果需要区分 P / D，并将 KV Transfer 计入通信结果和方案摘要。机器规格由共享的机器节点配置提供，P/D 只独立配置各自的部署规模和并行策略。

PD 结果摘要示例：

```text
PD · P: H20 × 8 · TP8
   · D: H20 × 16 · TP8 × 2
   · KV transfer 12.4 GB
```

## 10. 仍需讨论的事项

以下内容尚未进入开发确认状态：

1. PD 图谱中当前阶段成本和 KV Transfer 边界的显示方式，避免颜色和标签过载。
2. 中英文切换的默认语言、持久化方式和未翻译字段的降级规则。

这些事项应在详情页进入正式开发前确认；不影响当前入口和详情页的总体信息层级。

## 10.1 当前实现边界

当前前端已经落地：统一模型输入与内置模型选择、Hugging Face / ModelScope 配置读取、本地目录 config 与 safetensors header 读取、Provider 弹层、双语与深浅主题、单一结构详情工作区、公式/Shape/权重 Inspector、Cost Lens 多选、给定 TP / PP / EP / DP 投影、集中式与 PD 分离配置、Fit 摘要和结构导出。

仍明确不做：Hugging Face 热门模型服务、最近模型持久化、调度与吞吐仿真、KV transfer overlap、最优 parallel plan 搜索，以及 modelmap 式脉冲回放动画。

## 11. 已确认的详情交互

### 11.1 Inspector

- 桌面端使用右侧固定 Inspector。
- 移动端使用底部 Inspector Sheet。
- 未选中具体节点时，Inspector 展示模型级摘要。
- 选中节点后，Inspector 展示该节点的结构、参数、Shape、权重和成本信息。
- 移动端 Inspector Sheet 默认打开，可通过 handle 收起，让结构图获得更多空间。

### 11.2 结构图默认状态

默认只展示模型主干：

```text
Embedding → Decoder Layers × N → Final Norm → LM Head
```

Decoder Layer 内部的 Attention、MoE、Norm 和 Residual 默认作为折叠结构信息，不单独铺开。用户点击后再展开代表性层或具体子模块；重复层始终使用 `× N` 表示，不真实渲染全部重复节点。`展开全部` 是高级操作。

## 12. 下一轮确认顺序

下一步按以下顺序确认成本数据和 PD 交互：

1. 自定义机器字段和预置机器字段。
2. PD 模式下 Prefill / Decode 各自的负载参数，以及 DP 与节点规模的约束。
3. P / D / KV Transfer 在结构图、Inspector 和底部摘要中的展示方式。
4. 成本数据的展示细节和用户输入标识。

## 13. 已确认的机器与并行约束

### 13.1 不同精度算力分开建模

机器规格不能用单一算力或按字节合并表示。不同数据类型的算力必须独立字段保存和展示，至少包括：

- FP32
- FP16 / BF16
- FP8
- INT8

缺失某种精度的算力时，只对该精度显示不可用或无法评估，不用其他精度的数值代替。HBM 容量、HBM 带宽和互联带宽也属于独立硬件字段，不与算力字段合并。

### 13.2 机器节点与 P/D 部署配置分离

机器节点配置只维护一套，表示当前目标硬件的单卡和节点规格：

```text
Machine node: H20
96 GB / card · BF16 ... · HBM ...
```

Prefill 和 Decode 不分别配置机器类型。P/D 的节点数量、GPU 数量和并行策略属于各自独立的部署配置，可以分别调整和扩展；DP 表示数据并行规模，不再额外设置副本数。机器选择器展示单卡数据，最终摘要再展示每个阶段的规模。

### 13.3 P/D 全并行策略

Prefill 和 Decode 都完整支持以下并行策略：

- TP
- PP
- EP
- DP

不因为 P/D 的常见默认用法而在 UI 上删除某个维度。P 和 D 分别保存完整的 given-plan，切换阶段时展示对应阶段的 TP / PP / EP / DP。现有并行分析能力继续复用，不新增 plan 搜索或最优方案推荐。

## 14. 下一轮确认重点

接下来确认 P/D 方案的展示层级：

1. 配置面板如何展示共享的机器节点与独立的 P/D 部署规模。
2. 底部摘要如何显示当前阶段的一组结果和对应的部署规模。
3. 结构图如何表达当前阶段成本和 KV 传输边界。

## 15. 框架命名对照

MSV 的用户界面使用统一的框架中立命名，详情中的命令映射通过 tooltip 或高级信息展示，不直接把 CLI flag 当作主标签。

| MSV 名称 | vLLM / SGLang 对照 | 说明 |
|---|---|---|
| 节点数 | `--nnodes` | 分布式运行节点数量 |
| GPU / Node | 节点硬件拓扑 | 用于推导总 GPU 数，不是统一的主并行 flag |
| TP | vLLM `--tensor-parallel-size` / `--tp`；SGLang `--tp` | Tensor Parallelism |
| PP | vLLM `--pipeline-parallel-size` / `--pp`；SGLang `--pp` | Pipeline Parallelism |
| EP | Expert Parallelism 相关配置 | Expert Parallelism |
| DP | vLLM `--data-parallel-size`；SGLang `--dp` | Data Parallelism；不再额外设置“副本数” |
| Attention 并行方式 | SGLang `--enable-dp-attention` 等 | 只表示 Attention 的 TP / DP 方式，不等同于 DCP / CP |
| 集中式（非 PD） | 未启用 disaggregation | P/D 在同一套节点资源上运行 |
| PD 分离 | SGLang `--disaggregation-mode`；vLLM `--kv-transfer-config` | Prefill / Decode 分离 |
| KV Transfer backend | SGLang `--disaggregation-transfer-backend` | Mooncake、NIXL 等传输后端，不称为传输协议 |

“服务实例数”如果未来需要表达，必须作为独立的服务层概念增加，不能与 DP 同时使用相同的“副本数”标签。

## 16. 数据来源与可信度

MSV 必须区分模型配置、checkpoint 元数据、结构语义、验证状态和成本结果，不能用一个综合置信度覆盖所有信息。

### 16.1 实际数据链路

```text
config.json
  └── 模型类型、层数、Hidden、Attention、Experts、Context
        ↓
safetensors header
  └── tensor name、dtype、shape、parameter count
        ↓
skeleton trie
  └── 根据权重路径构建参数模块树
        ↓
template + truth merge
  └── 将参数真值绑定到结构语义节点
        ↓
cost calculation
  └── 基于模型、机器、负载和并行策略做理论计算
```

这里的 checkpoint 真值特指 `safetensors header truth`，不代表下载或读取完整权重。safetensors header 提供张量元数据，不提供完整模型架构语义；架构语义仍来自 config 和 MSV 模板。

### 16.2 字段来源

| 字段 | 来源标签 |
|---|---|
| 模型类型、层数、Hidden、Attention、Experts、Context | `config.json` |
| 模块权重路径 | `safetensors header` tensor name |
| 参数 Shape | `safetensors header` |
| dtype | `safetensors header` |
| checkpoint 参数量 | `safetensors header` |
| Attention / MoE / 算子语义 | MSV template |
| Transformers 结构验证状态 | `Transformers verified` |
| MACs / VRAM / KV / 通信 | `theoretical estimate` |
| 预置机器规格 | `preset` |
| 自定义机器和并行策略 | `user input` |

### 16.3 不同入口的当前边界

- 远程 HF / ModelScope 入口：读取配置，并尝试通过 safetensors header 获取 checkpoint 真值。
- `builtin`：当前主要读取仓库内置 `config.json`，不保证读取 safetensors header。
- `local`：当前后端主要返回本地配置，前端不保证继续读取本地 safetensors header。
- `auto`：命中 builtin 或 local 时可能直接结束，不一定进入远程 checkpoint truth 路径。
- 后端 `/api/structure`：当前主要使用 config + Transformers 结构构建，不复用前端 safetensors header 读取链路。

### 16.4 UI 展示规则

详情页字段应显示具体来源：

```text
参数量       checkpoint header
dtype        checkpoint header
weight shape safetensors header
层数         config.json
结构语义     MSV template
验证状态     Transformers verified
成本结果     theoretical estimate
```

如果 checkpoint header 不可用，必须明确显示：

```text
checkpoint metadata unavailable
parameter count estimated from config
weight shapes unavailable
```

不能把 config 推导值标记为 checkpoint 真值，也不能把理论成本标记为实测性能。

## 17. Chunked Prefill 计算边界

Prefill 负载参数区分完整输入和分块计算：

```text
输入 tokens / request
Prefill batch size
Chunked Prefill：关闭 / 开启
Prefill chunk size
```

Chunked Prefill 只计算以下两类结果：

1. **总量**：按完整输入长度计算 Prefill 总 MACs、最终 KV Cache 和总成本。
2. **单次峰值**：按 `Prefill chunk size` 计算单次迭代的激活、计算和显存峰值。

不模拟以下运行时行为：

- chunk 的实际调度顺序。
- PP pipeline bubble。
- chunk 之间的调度等待和吞吐变化。
- KV 的逐块传输时序和通信重叠。

PD 分离模式下，Chunked Prefill 只影响 Prefill 侧的峰值和计算节奏假设；最终 KV Transfer 总量仍按完整输入对应的 KV 计算。

UI 使用框架中立名称 `Prefill chunk size`。高级信息可以标注 SGLang 的 `--chunked-prefill-size`，以及 vLLM 的 `--max-num-batched-tokens` / `--max-num-scheduled-tokens`，但不把这些 CLI 参数直接作为主标签。

## 18. 已确认的负载、节点和展示策略

### 18.1 本地模型入口

本地模型采用双模式：

- 本地后端模式：支持输入或选择后端可访问的模型目录路径。
- 浏览器模式：使用文件夹选择器读取用户选择的模型目录。

两种模式都以模型目录为入口，由程序自动识别配置、权重索引、权重文件、tokenizer 和自定义代码。目录内的 `config.json` 不恢复为独立首页入口。

### 18.2 P/D 负载独立保存

Prefill 和 Decode 的负载参数分别保存，切换阶段时恢复各自的配置：

```text
Prefill: 输入 tokens / request、batch size、Chunked Prefill、Prefill chunk size
Decode: 当前上下文长度、batch size
```

成本视图一次只展示当前选中的阶段，不同时铺开 P/D 两组成本卡片。

### 18.3 PD 展示

PD 配置不提供 P/D 总览卡片。配置面板通过 `Prefill / Decode` 切换，一次只显示当前阶段的节点规模和完整 TP / PP / EP / DP。底部只显示当前阶段摘要；KV Transfer 仅在 PD 分离模式显示。

### 18.4 自定义机器字段

自定义机器字段与预置机器使用同一 schema，算力按精度独立保存：

```json
{
  "name": "custom",
  "memory_gib_per_gpu": 96,
  "gpus_per_node": 8,
  "peak_flops": {
    "fp32": 0,
    "fp16": 0,
    "bf16": 0,
    "fp8": 0,
    "int8": 0
  },
  "hbm_bandwidth_gbps": 0,
  "intra_node_bandwidth_gbps": 0,
  "inter_node_bandwidth_gbps": 0
}
```

缺失某种精度算力时，该精度显示不可用，不使用其他精度代替。GPU 数量、节点数和并行策略属于部署配置，不重复写入机器规格。

## 19. 已确认的 PD 与语言展示策略

### 19.1 PD 成本展示

PD 采用“当前阶段 + 独立 KV Transfer 信息条”的展示方式：

```text
Prefill · H20 × 8 · TP8
KV Transfer · 12.4 GB · Mooncake · 400 GB/s
```

结构图只显示当前选中阶段的成本指标，不把 KV Transfer 强行绘制成结构图边或额外的 P/D 图谱。KV Transfer 信息条只在 PD 分离模式出现，集中式模式不显示。

### 19.2 中英文行为

- 默认语言跟随浏览器语言，中文浏览器默认中文。
- 用户手动切换后使用 `localStorage` 记忆语言偏好。
- 当前语言缺失翻译时回退英文，不强行把技术名词翻译成可能更难理解的中文。
- 模型名、算子名、并行策略、dtype、backend 等技术标识默认保留英文或官方命名。

## 20. 产品决策确认状态

入口、详情页布局、机器节点、集中式 / PD 分离、全并行策略、Cost Lens 多选、Chunked Prefill、数据来源标签和语言行为均已确认。当前可以进入前端实现；后续新增需求应先判断是否改变本文件中的已确认边界。
