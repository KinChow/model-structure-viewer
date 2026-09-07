# 重构计划：自底向上模块替换

本文是 [`principles.md`](principles.md) 的收口路线。原则定义"应该是什么"，本文定义"按什么顺序变成那样"。

**方法**：自底向上，一次替换一层。每层替换时，其**消费方保持不变**（接口不动），
用差分测试证明行为等价后再删旧实现。

**为什么自底向上**：msv 的依赖是一条拓扑链
`formulas ← ops ← layers ← models ← IR ← graph ← cost ← UI`。
从底部替换，每一步的爆炸半径都被上层接口挡住；反向做则每改一处都要同时改所有下游。

**两个例外**（不排队等重构，因为它们不是重构）：
- **护栏**（CI 检查）必须最先，它是整个迁移过程的保险。
- **纠偏**（已知 bug）必须最先，缺陷修复不应排在结构调整之后。

---

## 依赖顺序

```
M0 绿色基线（不改代码）
     |
     v
W0 护栏 + 纠偏（非分层）
     |
     v
W0.5 normalize 提取层收敛 -> W1 formulas -> W2 ops -> W3a 分派表+角色表+配方表
     |                                                        |
     |                                                        v
     |                              W3b 方案归组网 -> W4 graph -> W4.5 映射表 -> W5 cost -> W6 UI
     |
     +--(无依赖，可并行)--> 旁路 B 后端降级 + source_ref
     +--(无依赖，可并行)--> 旁路 C 芯片自定义参数补账
     +--(无依赖，可并行)--> 旁路 D 后端内部清理
     +--(依赖 W6)--------> 旁路 E UI 状态收敛
```

W4 仅依赖 W3a，可与 W3b 并行；为叙述线性排在 W3b 之后。

全部波次都可用差分测试验收（行为不变），没有需要重做快照的波次。
**原 W7"trie 成为骨架"已取消** —— 骨架来自适配产物（§4.3），trie 只提供数值与未适配兜底视图
（§4.5）。相应地，原先围绕它的五个未决问题（折叠判据、锚定语法、双骨架、节点 id、边覆盖度）
全部消失，因为它们都是"运行时自动推断结构"这一错误设想的产物。

## 里程碑

| 里程碑 | 波次 | 完成判据 | 收口原则 |
|---|---|---|---|
| **M0** 绿色基线 | — | 四条验证命令的红绿状态已知并记录 | — |
| **M1** 护栏与止损 | W0 | 故意加一处家族名 → CI 失败；两类边样式不同 | §2.2 部分、§3.2/§8.1 检查手段 |
| **M2** 底层收敛 | W0.5 + W1 + W2 | 59 模型 `normalizeConfig` 输出深度相等；spec 树 JSON 深度相等；每条目三选一（counts/纯traffic/分解声明）且通过整模型恒等式；matrix 维度差分相等 | §3.1、§3.5、§3.7（counts 接口）、§8.2（ops 侧） |
| **M3** 结构声明化 | W3a + W4 | 边集合（含 evidence）差分一致；所有边 evidence 非空；含参节点均有 `role` | §2.1、§2.2、§4.3 层 1、§8.2 |
| **M4** 方案归组网 | W3b | 最终 spec 树差分一致；`normalizeConfig` 不再输出方案类字段 | §4.3（配方）、§4.7 |
| **M5** 真值绑定显式化 | W4.5 | 绑定差分一致 **且 `ambiguous` = 0**；可逆校验通过 | §4.3 层 2、§4.4、§4.5、§4.6 |
| **M6** 成本分层 | W5 | 现有 cost 测试数值全不变；未实现算子返回 `null`；换卡只走表乘法 | §3.2、§3.3、§3.4 |
| **M7** 诚实性上界面 | W6 | 三类边样式互不相同；未覆盖提示与 gaps 面板可见 | §2.2 UI 侧、§4.2、§4.4 |
| **旁路 B** 后端定位归位 | B | `msv verify` 出三类差异列表；跳源码且显示 transformers 版本 | §5.1–§5.4、§6.1–§6.3 |
| **旁路 C** 芯片参数补账 | C | 手工卡有 `field_sources`；单位异常有警告；文案与实现一致 | §7 |
| **旁路 D** 后端内部清理 | D | 后端无平行修复机制；无 `assert` 做入参校验 | §6.1 SRP 侧 |
| **旁路 E** UI 状态收敛 | E | `ArchitectureTab` props 数显著下降；无同值双 prop | §6 分层 |

规模为量级估计，非精确值：净变化约 **−3200 / +800** 行，最大两块是旁路 B（后端 IR 相关 −2000）
与 M3（死码 −420 + 重复消除）。

---

## M0 绿色基线（不改代码，必须最先）

差分替换的全部前提是"当前测试是绿的"。否则第一波差分失败时无法区分是新改动引入的、
还是本来就红。

```bash
npm --prefix frontend run test           # node --test
npm --prefix frontend run test:e2e       # playwright
pytest                                   # 后端
npm --prefix frontend run verify:models  # 内置模型校验
```

- **判据**：四条命令的红绿状态已知并记录；本来就红的项写明原因
- **已知风险**：开发环境 HF 不可达，`verify:models` 与部分 e2e 可能本来跑不通。
  这类项必须写成"已知不可用及原因"，否则会在后续每一波成为噪音
- **禁止**：在 M0 未完成前改任何重构代码

---

## 通用替换手法：差分替换（parallel-run）

每一波的核心动作不是"改完跑测试"，而是：

1. 新实现与旧实现**并存**（旧的改名为 `*Legacy`）；
2. 写一条差分测试：遍历 `models/catalog.json` 里的**全部内置模型**，对每个节点断言
   `new(node) === legacy(node)`；
3. 差分全绿后删除 `*Legacy` 与差分测试。

这把"数值不变 / 结构不变"从一句期望变成一个机械检查。
`models/catalog.json` 已含全部内置模型与 `config_path`，是现成的差分驱动数据。

**硬约束**：差分发现不一致时，**先查清哪个是对的**，不得直接修改测试期望值。
"重构顺手修了个 bug"通常是引入了一个 bug。

---

## W0 护栏与纠偏（非重构，最先做）

- **范围**
  1. `scripts/check_principles.sh`：① 家族名硬编码的非测试文件数 ≤ 基线 16
     （完整 pattern 实测；早年窄 pattern 的 11 系漏计变体，§8.1）
     ② `frontend/src/cost/**` 不出现 `node?.name` 参与计算（§3.2）
  2. 修 evidence 接线倒挂（详见下方"已知缺陷"）
- **入口**：`scripts/check_principles.sh`、`frontend/src/diagram/edgeStyle.js`、
  `frontend/src/diagram/ReactFlowStructureDiagram.jsx`
- **依赖**：无
- **验收**：`bash scripts/check_principles.sh` 退出码 0；故意新增一处家族名后应失败；
  `npm --prefix frontend run test:e2e` 断言 `declared` 边与 `module-order` 边样式不同
- **不包含**：新增 evidence 取值（W4）、完整三级视觉（W6）
- **回退**：纯新增脚本 + 5 行样式判断，直接 revert

### 已知缺陷：推断边被当成主流，声明边被降级

活代码只产出三种 evidence：`declared`（`graph/declaredEdges.js:29`）、
`module-order`（`graph/materializeStructureGraph.js:468`、`truth/graphTruth.js`）、
以及 `undefined`（`materializeStructureGraph.js:491`，shape-match 边）。
所有 `semantic-flow` 取值都在**已死的** `legacySemanticEdges` 里。

```js
// diagram/edgeStyle.js:9 —— 推断边固定 2.4，通常比声明边更粗
if (edge?.evidence === "module-order") return 2.4;

// diagram/ReactFlowStructureDiagram.jsx:136
const mainFlow = data?.evidence === "module-order" || data?.evidence === "semantic-flow";
//               ^ 推断边算主流                        ^ 死值，永远 false
//               => `declared` 边既不是 mainFlow，也不加 CSS class
```

`diagram/elkLayout.js:70` 是唯一正确识别 `declared` 的地方。W0 只做最小修正：
死值改 `declared`，视觉方向反转（声明实线、推断虚线）。

---

## W0.5 normalize：提取层机械收敛

范围收缩（2026-09-07 对齐）：只做无争议的机械收敛。**方案决定权归还组网（Job B 迁移）
不在本波**——若终态是消融进配方表，中间站（如拆到 derive.js）没有存在价值（§4.7）。

- **范围**
  1. `pick(keys)` 闭包 → 收敛 **63 处** `firstNumber(textConfig, keys) ?? firstNumber(config, keys)`；
  2. `modelTypeProbe` 常量 → 收敛 **14 处** model_type 探测，并显式登记现存**三种变体**
     （A：`config || textConfig` 全量探测，约 14 处；B：仅 `config`，`normalize.js:291,295`；
     C：A 再兜一层 textConfig，`normalize.js:335`）。**行为逐字保留，只集中、不统一**——
     B/C 是潜在功能分歧（A/B 在"顶层 model_type 与 text_config 不同"时结果不同），
     统一语义属功能决策，不在重构范围；
  3. ~~家族特征表 families.js~~ **已取消**：家族是错轴（§4.3 组件配方为一级概念），
     配方表在 W3a 建立。
- **入口**：`structure/config/normalize.js`（365 行）
- **依赖**：W0
- **验收**：差分测试断言全部 **59 个内置模型**的 `normalizeConfig` 输出**深度相等**；
  护栏 §8.1 计数不升（≤16）
- **不包含**：Job B 迁移（W3b）、统一 B/C probe 语义、任何字段取值变化
- **回退**：纯内部实现替换，消费方接口不变

---

## W1 formulas：注册条目产出动作向量（含芯片 schema，一次做全）

- **范围**
  1. 48 个 `FORMULAS` 条目**全部终止于 counts**——自有公式或分解声明（分解假设显式
     标注），**无白名单**（2026-09-07 对齐：不允许存在"未实现"条目，null 唯一来源是
     芯片缺字段，§3.1/§3.3）；
  2. 每条目 `counts(ctx) → { matrix, vector, sfu, bytes: {weights, actIn, actOut} }`
     （§3.1 形态；`matrix` 存 MACs，aten 公式含 2× 需显式换算注明；`bytes` 为每次前向
     compulsory traffic，无 phase 分支）；`counts` 入参只含结构化 shape 参数；
  3. 有 aten 对应的条目加 `aten` 锚点（公式照抄 `flop_registry` 并换算）；语义复合算子
     （rope / KDA / DSA / moe dispatch）用分解声明；
  4. **芯片 schema 并入本波**（依赖提前，2026-09-07 对齐）：`fp32` 语义正名
     `vector_flops`（旧名兼容）、新增 `sfu_ops`（NVIDIA 条目按 CUDA guide 官方比值
     16:64 推导并标注 source 与推导）；`chips/coverage.js` 能力门控扩展两列；
  5. **三层 golden 用例**（①②永久保留，③过渡）：
     - ① per-op 手算 golden：照 PyTorch `test_flop_counter.py` 风格，每算子手算 shape
       + 期望值；
     - ② 整模型恒等式（外部 oracle）：59 个内置模型逐个断言 matrix 聚合
       ≈ `2 × 参数量 × tokens`（MoE 按 expertFraction 缩放）；decode 场景
       traffic ≈ 权重字节；
     - ③ 旧链差分：`nodeMacs` 旧实现 vs 新表，matrix 维度逐节点相等（W5 删）。
- **入口**：`frontend/src/structure/formulas/index.js`、`frontend/src/cost/chips/{coverage,public}.js`
- **依赖**：W0
- **验收**
  - CI 新增第 ③ 项：每条目三选一（counts / 纯 traffic / 分解声明）；
  - ① 全部通过；② 59 模型恒等式通过（允许的偏差只来自 §2.3 声明的建模边界，如
    MoE 路由、量化 packed shape）；
  - ③ matrix 维度 59 模型逐节点相等；
  - `npm --prefix frontend run test`；护栏 §3.2 通过
- **不包含**：改动 `cost/compute.js` 消费方（W5）；roofline 五路 max（W5）；
  新增昇腾芯片条目（数据已查实，时点待定，见未排期）
- **回退**：纯新增字段 + schema 兼容旧名，旧路径未动

参考 PyTorch `torch/utils/flop_counter.py` 的 `flop_registry`：一个 op 一个注册点，
公式入参是 shape，未注册的 op 先尝试分解、再记 0，且该语义写在 docstring 里。

> **状态（2026-09-08 收口）**：✅ 完成。42 条目（6 死条目删除）全部终止于 counts；
> 三层 golden 齐备。恒等式终态：21 个目录 MoE 行 |ratio-1|≤1.7%（两模型精确闭合），
> 4 个合成 dense/MoE 变体 ratio=1.0000 精确闭合；统一 2% 容差无特例。
>
> **W1 问题实录**（校准期间咬出的真 bug，按发现顺序）：
> 1. routed swiglu 按 k/E 计数 → 应按 k（被选中专家完整执行）；
> 2. shared expert 中间维列表式回退两次漏模型（deepseek_v3、glm_moe_dsa）→ 通用化；
> 3. 通用化又漏 kimi_k3 fused 语义（模块宽 = moeI×n_shared，非 fused 才是单专家宽）；
> 4. `mla_query_compress` 复合含 qb + 独立 q_b 叶子重复计费；
> 5. derived 在无 attentionSchedule 时 MLA 误按 GQA 计 attention；
> 6. 期望侧 score 项 2× 双计（`2·2·heads·T²·D` → `2·heads·T²·D`）；
> 7. identity 的 normsTerm 只减一层 norm 权重（→ `2·L+1` 层）；
> 8. `generic-decoder` 结构缺 embed/final norm/lm_head——llama 不在 59 目录内，
>    目录校验从未覆盖；由合成 llama 变体暴露；
> 9. identity 测试自身两处（walk 的 repeat 乘子误读、score 项 T 双计）。
>
> **教训**：① 目录没有纯 dense 模型，dense 字段组合只能靠合成变体覆盖——合成配置
> 是恒等式的第二覆盖面，不是可选项；② 期望侧与 counts 侧是两本手工账，新架构常
> 只破坏其中一项（k/E、shared、fused 各坏一边），两侧对账 + 外部真值（官方参数量）
> 是唯一能同时抓住两类的手段，与业界实践一致（PyTorch/fvcore/Megatron 同构）。

---

## W2 ops：消除注意力尾链复制

- **范围**：抽出 `scaledDotProductTail(prefix, shapes, dims, { attentionKind, extras })`，
  替换 `attentionOperatorSpecs`、`qwen35Full`、`mla`、`minimaxAttentionCommon`、
  `minimaxM2` 五处重复的 `rope → scores → softmax → context → o_proj`；
  抽出各 builder 头部重复约 12 次的 `tensorShapes + tensorDims` 样板；
  `linearAttentionOperatorSpecs:76-90` 的 5 连纯转发 if 改为集合判断。
- **入口**：`frontend/src/structure/model_executor/ops/index.js`（875 行）
- **依赖**：W1
- **验收**：差分测试断言全部内置模型的 spec 树 JSON **深度相等**
- **不包含**：改变任何算子 id、名称或 attributes 取值（改了会影响 W4 的边解析）
- **回退**：差分不通过即整波 revert

---

## W3a layers：分派改查表 + 角色表 + 配方表

- **范围**
  1. `layers/attention.js:10-28` 与 `:29-53` 两条平行嵌套三元合并为一张表，
     **ops 与 edges 必须来自同一个表项** —— 使"children 改了 edges 没跟着改"结构上不可能；
  2. 新建 **canonical 节点角色表**（§4.3 层 1），对齐 llama.cpp 的 `MODEL_TENSOR` 枚举 +
     `MODEL_TENSORS[arch]` 列表。含参节点声明 `role`（`attn_q` / `attn_qkv` / `attn_norm` /
     `ffn_gate` …）而不是自己编 id；`role` 与 `operator_id` 是两个维度，都保留；
  3. 新建**配方表**（如 `structure/components.js`）：以组件配方为键
     （attention / norm / FFN / 位置编码 / 附加结构），家族名只出现在
     "模型 → 配方"的薄解析层（§4.3）。注意与 W3b 的分工：本波只建表并让分派消费它，
     **不改 normalizeConfig**；
  4. `decoderLayer.js` / `decoderStack.js` / 网络层显式声明顺序执行，
     让 `evidence` 落到 `module-order` 且**被明确标注**，而非兜底静默产生（§2.1）。
- **入口**：`layers/attention.js`、`layers/decoderLayer.js`、`layers/decoderStack.js`、
  `model_executor/models/common.js`、新增 `structure/roles.js`、新增 `structure/components.js`
- **依赖**：W2
- **验收**：差分测试断言 spec 树深度相等（`role` 是新增字段，不改变既有字段）；
  新增用例断言含 children 的模块均有边声明或顺序标记；断言每个含参节点都有 `role`
- **不包含**：checkpoint 映射表（W4.5）、Job B 迁移（W3b）、残差跨层级边（§2.5，暂缓）、
  attentionKind 家族品牌 id 改名（输出可见，见未排期）
- **回退**：表驱动与三元链可共存一个提交，差分后再删旧链

> **状态（2026-09-08）**：W3 已重组为单波四阶段（A 分派合表 → B 角色表+映射绑定 →
> C normalize 瘦身 → D evidence+死代码），原 W3b/W4/W4.5 全部并入（其内容一行不少，
> 消失的是等待成本）。A/B 已完成（commit f4eb1a3、6953d66、344336f），实际落点与
> 原设计的差异：组件表=**layers/ 原地升级**（非新建 components/）；每架构映射=
> **structure/archs/**（非 adapters/，命名对标 llama.cpp llama-arch）。
>
> **B2 遗留登记（勿忘）**：
> 1. **vision 域 checkpoint 绑定仍走路径兜底**——SUFFIX_ROLES 未覆盖 vision 后缀
>    （`fc1`/`fc2`/`norm1`/`patch_merge` 等）；待真实 vision checkpoint 在可逆校验
>    暴露第一处失配时，新建对应 `archs/<family>.js` 覆盖规则（机制已就位）；
> 2. **mergeSemantics.js 是死代码**（src 零引用，活路径为 graphTruth.bindTruthToGraph，
>    B2 时才发现）——D 阶段删除，连同其 5 个测试的语义评估（有价值断言已由
>    roleBinding.test.js 覆盖）；
> 3. **FAMILY_OVERRIDES 注册表当前空置**——22 个文本家族全部符合 canonical 命名；
>    首个偏离出现时新建家族文件，不预造空壳；
> 4. **ambiguous=0 目前仅在 fixture 级证明**（roleBinding.test.js 4 例）——真实
>    checkpoint 数据的 `graph_ambiguous_truth_matches` 需线上 diagnostics 观察，
>    UI 呈现（§4.5 未适配标注等）统一推 W6；
> 5. **"norm" 顶层特判**是深度规则（剥 wrapper 后 depth=1）；未来若出现非顶层
>    final-norm 例外，先归因再改规则。
>
> **W3-C 后的结构债（带债进 W5，触发点：接入新模型家族）**：C 把方案逻辑从
> normalize 搬到了 plan.js，但家族知识仍住 5 处（models/*.js、attention.js 组件表
> 匹配器、plan.js 探测、archs/、normalize 数字派生探测）。C 的对齐文档原定
> "models/*.js 迁移后删除"未随 C 执行——收口 = 家族声明化（FAMILY_OVERRIDES
> 扩展为完整配方声明，plan.js 收缩为纯执行器、护栏豁免取消）+ models/ 迁入
> archs/ + 组件表改按配方 id 键控。接新家族前必须先做，否则新家族要改 3 处。

---

## W3b 组网：方案决定权归还组网，normalizeConfig 瘦身

- **背景**：`normalizeConfig` 现在替 builder 预先做完全部方案决定
  （`attentionSchedule` / `layerSchedule` / `linearAttentionMode` / `normMode` / …），
  是 god-object 耦合枢纽——每个字段单看意义不明，合起来是某几个 builder 的私人菜单，
  且泄漏到 cost（`compute.js:308` 读 `layerSchedule` 算专家比例）。
  transformers 的分工是 `Config` 类管字段、`modeling_*.py` 管决定（§4.7）。
- **范围**
  1. 配方表（W3a 建立的）接管 Job B：逐层调度与方案选择由配方声明；
     调度读取逻辑可为共享 util，但**调用权在配方/组装点**；
  2. builder 在组装点消费 config 视图 + 配方，自行决定造什么；
  3. `normalizeConfig` 瘦身为纯字段归一（别名 + 默认值），删除全部方案类字段与家族分支；
  4. `cost/compute.js:308` 对 `layerSchedule` 的消费改为结构节点属性或显式 plan 对象。
- **入口**：`structure/config/normalize.js`、`structure/model_executor/**`、`cost/compute.js`
- **依赖**：W3a
- **验收**：**最终 spec 树**差分一致（59 模型）；`normalizeConfig` 输出不再含方案类字段
  ——后一项是输出变化，需重建差分参照并人工核对一次；
  护栏 §8.1 计数应显著下降（normalize.js 退出计数）
- **不包含**：attentionKind 家族品牌 id 改名（见未排期）
- **回退**：新旧两条组装路径可共存，差分后再删旧

---

## W4 graph：删死代码，补全 evidence

- **范围**
  1. 删 `legacySemanticEdges`（`materializeStructureGraph.js:3-422`，420 行，占该文件 78%，
     全仓零引用）；
  2. `:491` 的 `evidence: undefined` → `"shape-match"`，使 evidence 三值齐全且非空；
  3. 顺带删 `cost/comm.js:115` 未使用导出、`cost/compute.js:24` 恒假守卫、
     `App.jsx:287` 死分支与重复 prop（`expandedGroups`/`layersExpandedPaths` 同值双传）。
- **入口**：`frontend/src/structure/graph/materializeStructureGraph.js`
- **依赖**：W3
- **验收**：差分测试断言全部内置模型的边集合（含 evidence）完全一致；
  新增断言"所有输出边 evidence 非空"
- **不包含**：改变边的生成规则
- **回退**：删除类改动，revert 即可

---

## W4.5 映射表：checkpoint 对应关系显式化

替代原 W7。**这一波是本计划的核心正确性改动**：把"运行时猜"变成"适配时声明"。

现状是猜：`mergeSemantics.js:56-75` 用 `canonicalModulePath` 归一后做**相等匹配**，
同一路径匹配到多个候选时 `:58-61` 直接 `continue` **静默放弃绑定**，只记进
`ambiguous_truth_matches` 而 UI 不展示 —— 用户看到的是一个看起来完整的、缺真值的结构。

- **范围**
  1. 新建 per-arch checkpoint 映射表（§4.3 层 2），形态照抄两个先例：
     - llama.cpp `gguf-py/gguf/tensor_mapping.py` 的 `block_mappings_cfg`：
       **候选列表解决别名**（`o_proj` / `out_proj` 并列写），`{bid}` 占位解决重复层；
     - transformers `WeightRenaming` / `WeightConverter`：可组合可逆的 `ConversionOps`
       （`Chunk`/`Concatenate`、`MergeModulelist`/`SplitModulelist`、`Transpose`、`PermuteForRope`），
       其中 `MergeModulelist` 正好对应 MoE fused `w13_weight`；
  2. `mergeSemantics` 的绑定改为按映射表执行；**删除 `canonicalModulePath`**
     （其唯一实质用途就是这个绑定）；
  3. 映射缺失或对不上时**报错并在 UI 显示**，不再 `continue` 静默放弃；
  4. 未适配模型走 `skeleton-truth` 并**显式标注"未适配、无语义"**（§4.5）。
- **入口**：`structure/truth/mergeSemantics.js`、`structure/truth/graphTruth.js`、
  新增 `structure/adapters/<arch>.map.js`
- **依赖**：W3（需要角色表）、W4
- **验收**
  - 差分测试：全部内置模型的绑定结果与现有 `template+truth` 路径一致，
    且 `ambiguous_truth_matches` 数量降为 **0**；
  - **可逆校验测试**（§4.6）：用映射把 trie 反推成角色集合，再正推回 checkpoint 名，
    与原始 header 逐项对比，对不上即失败；
  - `npm --prefix frontend run test`
- **不包含**：把图构建改成纯数据（§4.3 明确禁止）；trie 充当骨架（已取消）
- **回退**：新绑定与旧绑定可共存一个提交，差分 + 可逆校验双绿后再删旧路径

**不改变节点 id、不改变图形状**，因此可用"行为不变"验收——这是它相对原 W7 最大的差别。
节点 id 仍来自适配产物；checkpoint key 只出现在映射表的右侧。

---

## W5 cost：查表 + ERT/counts 分离

- **范围**
  1. `nodeMacs` 改为查 `FORMULAS[operatorId]?.counts`（§3.1 动作向量）；
     删 `macsSource` 第二套 switch（从查表结果推导）；删两处显示名正则（`compute.js:243-244`）；
  2. 删 `qwen4ExpLinearAttentionMacs`（与 `qwen35LinearAttentionMacs` 逐行相同，只差 fallback）
     与 `nodeMacs:245-255` 内联重写的 qsa/minimax 公式；
  3. layer-index 正则 3 种变体、routed-expert 正则 3 处各统一到一处；
  4. 拆 ERT × action counts：`computeNodeCosts` 产出与芯片无关的动作向量
     `{ matrix, vector, sfu, bytes: {weights, actIn, actOut}, commBytes: { intra, inter } }`；
     新增 `chips/rates.js` 把芯片规格 + 效率因子转成 rate 表
     ——**rate 含四列**：`peak_flops[dtype]·η_flops` / `vector_flops·η=1.0` /
     `sfu_ops·η=1.0` / `memory_bandwidth·η_hbm`；`roofline.js` 退化为纯 join，
     时间 = **五路 max**（矩阵/向量/SFU/访存/通信），瓶颈分类细化为五类（§3.7）；
     其 `missing` 判定复用 `chips/coverage.js:missingFields`；
  5. ~~芯片 schema 扩展~~ **已并入 W1**（依赖提前，2026-09-07 对齐）；
     本波只做 per-chip 单元映射的取舍：昇腾缺 `sfu_ops` 时是否声明 sfu→vector rate
     （语义映射非估算，§3.7）。
- **入口**：`frontend/src/cost/{compute,roofline,aggregate}.js`、`cost/chips/rates.js`
- **依赖**：W1、W4
- **验收**：`cost/__tests__/compute.test.js` **现有断言数值全部不变**（matrix 维度）；
  新增"未实现算子返回 null 且计入 `unknownComputePaths`"用例；
  新增"同一 actionCounts 换两张卡只走表乘法"用例；
  新增"softmax 在 seq=1 时落 memory/sfu-bound、prefill 落 memory-bound"的定性用例；
  `bash scripts/check_principles.sh` 检查 ② 通过
- **不包含**：改动公式数学形式；新增国产芯片条目
- **回退**：差分不通过即整波 revert

`aggregate.js:20-27` 的 null 传播、`computeComplete`、`unknownComputePaths`、`macsSources`
**已经实现**，缺口只在叶子把"未实现"当成 0。因此第 1 项一旦落地，整条链自动生效。

> **ERT 分离的正当性来自分层，不来自芯片数量。** 公开国产芯片规格短期拿不到，
> 但用户可通过 `chips.local.json` 与手工录入自定义芯片，卡数在运行时可增长。
> 更重要的是：芯片参数出现在 action counts 计算路径里本身就违反 §3.4，
> 与当前有几张卡无关。因此第 4 项照做，不因芯片数少而推迟。

---

## W6 UI：诚实性上界面

- **范围**
  1. evidence 三级视觉区分：`declared` 实线、`shape-match` 细线、`module-order` 虚线 + hover
     提示"此边由兄弟顺序推断"（§2.2）；
  2. 汇总条展示"N 个算子成本未覆盖"（数据来自 `aggregate.js` 的 `unknownComputePaths`，已存在），
     并按单元缺失分别计数（§3.3：matrix=0 是精确陈述，不是零成本）；
  3. 瓶颈分类显示五类（矩阵 / 向量 / SFU / 访存 / 通信，§3.7）；
  4. `template_gaps` / `ambiguous_truth_matches` 上界面（现在只进 diagnostics，
     等于真值静默缺失，§4.4）；
  5. `value_source` 扩展到 Cost Lens 与汇总条（现仅 `NodeDetailPanel.jsx:24` 一处）。
- **入口**：`frontend/src/diagram/edgeStyle.js`、`components/{CostSummary,SummaryChips,NodeDetailPanel}.jsx`
- **依赖**：W4、W5
- **验收**：`npm --prefix frontend run test:e2e` 断言三类边样式互不相同、未覆盖提示可见、
  gaps 面板可见
- **不包含**：Model Explorer 式按 layer 惰性布局（可选优化，不在关键路径）
- **回退**：样式与展示层改动，revert 即可

---

## 旁路 B 后端降级 + source_ref（与 W1–W6 无依赖，可并行）

- **范围**
  1. 后端职责收缩为产出 `[(module_path, class_name, source_ref, has_params)]`；
  2. `source_ref` 采集照抄 modelmap `src/modelmap/annotate.py:116-134`
     （`inspect.getsourcefile` + `getsourcelines`[1] + 包根前缀匹配 + 版本锚定 blob 链接）；
  3. 产物写成随 catalog 发布的静态 JSON，静态部署可用（§5.3）；
  4. `/api/verify` 改为差异报告（仅 transformers 有 / 仅 msv 有 / 类名不符）并**加 UI 入口**；
  5. 下线 `/api/structure` 与后端 IR 产出；删 `graph.py`/`fold.py`/`semantics.py`/`keys.py`/
     `summary.py` 中仅为产出 IR 而存在的部分。
- **入口**：`src/model_structure_viewer/{verification,structure,api}.py`、新增产物生成脚本
- **依赖**：无（可与 W1 并行）
- **验收**：`pytest`；`msv verify --model <id>` 输出三类差异列表；
  静态部署下点节点能跳 GitHub **且界面显示 transformers 版本**
- **不包含**：remote code 沙箱、后端生产化（保留在 `implementation_plan.md` P2）
- **必须同时认下的成本**：产物需随 transformers 版本重新生成，是一个周期性 CI 任务，
  不是一次性产出。若不接受该成本，§5.3 需改为"仅在有后端时提供"并接受它不进静态部署。
  **已确认接受**（2026-09-07），自动化机制与 diff 分级见 `principles.md` §5.3。

---

## 旁路 C 芯片自定义参数补账（独立，随时可做）

`cost/chips/` 是全仓最健康的一层（字段级深合并、`validateChipEntry` 强制 source、
`loadLocal.js:41` 的 HTML content-type 防护、示例用 `example-chip` + 虚构占位）。
只需补 §7 的欠账：

1. 手工卡缺 `field_sources` —— `public.js` 每张卡逐字段标来源，`createManualChip` 完全不写。
   表单加"来源"输入（可留空则记 `user-provided`），写入所有已填字段。
2. 缺量级健全性**警告**（不是拒绝）：`memory_bandwidth` < 100 GB/s 或 > 20 TB/s、
   `memory_bytes` < 1 GB 或 > 1 TB、`bf16` < 1 TFLOPS 或 > 10 PFLOPS → 提示"请确认单位"。
   §3.6：倍数级错误会改变结论。
3. `chips.local.example.json` 在 `src/cost/chips/`，而加载器读 `/chips.local.json`
   （即 `frontend/public/`）→ 在示例 `notes` 里写明目标路径，或移位。
4. 手工卡仅会话级，刷新即失 → 加"复制为 chips.local.json 片段"按钮，打通到持久化。
5. 删 `interconnectGb` 影子字段（`manual.js:3,15,29`；表单 onChange 同写两个同值键；
   初始 state 无此键；仅 `manualChip.test.js:6` 单独用它）。
6. 修文案不一致：`coverage.js:47-49` 说"缺 `inter_node.bandwidth` 时按节点内带宽估算，
   结果偏乐观"，但 `roofline.js:35` **没有回退**（`commTime` 为 null → `bound` 变 `unknown`）。
   建议改文案为"跨节点通信判定不可用"——真回退到节点内带宽会严重低估跨节点 comm，
   而 comm-bound 恰是最重要的结论。

- **依赖**：无
- **验收**：`npm --prefix frontend run test`；手工录入一张单位填错的卡应出现警告而非静默接受
- **规模**：+80 / −15

---

## 旁路 D 后端内部清理（独立，与旁路 B 无依赖）

代码审查中发现、但不属于旁路 B 范围（B 只处理"职责收缩"）的后端债：

1. `service.py:213` 与 `:253` 是同一段"direct 模式 try/except + subprocess 回退"结构的
   **两份拷贝**，两个 entrypoint（`:340`、`:361`）同理 → 合成一个泛型 worker 包装。
2. `recovery.py:255-261` 的 `_runtime_compat_adapters()` 返回 `(name, predicate, kind)` 元组表，
   却在 `:209-220` 对 `name` 做 if/elif 决定实际动作——表面数据驱动实际不是。
   而 `repair/registry.py` 已有正规的 `RepairStrategy` Protocol + registry。
   **两套机制解决同一问题**；加一个适配器要改 4 处（`compat.py` / 元组表 / if-elif / `RecoveryKind` Literal）
   → 收进 `repair/` 的 Protocol registry。
3. `api.py:106,114` 用 `assert model_id is not None` 做入参校验 → 客户端拿 500 而非 400，
   且 `-O` 下失效 → 改 `HTTPException(400)`。
4. `resolver.py` 三层 re-export（`resolver.py` → `resolve/__init__.py` → `resolve/resolver.py`），
   而 api/cli/service/tests 都从最外层导入 → 收敛为一层。
5. 删仅测试引用的 `repair/runtime.py:21 NoopRuntimePatch`、`introspect.py:163 _walk`。
6. `schemas.py:13-50` 的 `StructureNode` / `StructureGraphNode` 逐字重复 13 字段 → 抽公共基模型
   （此项也登记在 `implementation_plan.md` 的"统一结构协议"下）。

- **依赖**：无（但若旁路 B 先做，第 2、5 项的范围会缩小）
- **验收**：`pytest` 全绿；新增"加一个 repair 策略只改一处"的用例
- **规模**：−200 / +60

---

## 旁路 E UI 状态收敛（依赖 W6）

- **范围**
  1. `App.jsx` 13 个 `useState` 按域拆分（模型入口 / 画布视图 / 搜索 / 主题语言 / 芯片）；
  2. `DetailWorkspace.jsx` 14 个 cost 面板 `useState`（phase / mode / plans / nodes / gpusPerNode /
     loads / comparison / efficiency）收进 `useCostPanel()` hook；
  3. `ArchitectureTab` 现接 **37 个 props** 并再 `{...props}` 展开给画布 → 改为传一个
     controller 对象；
  4. 删同值双 prop（`expandedGroups` 与 `layersExpandedPaths`、`selectedNode` 与 `selectedNodePath`）。
- **依赖**：W6（避免与"诚实性上界面"的展示改动交叉，使两者各自可差分验收）
- **验收**：`npm --prefix frontend run test:e2e` 行为不变；props 数量下降
- **规模**：+120 / −150
- **不包含**：引入状态管理库（React 内置 hook 足够，引库属范围扩张）

---

### 已取消：原 W7"trie 成为骨架"

原计划让 trie 建树、模板降为纯数据表。**该方案已取消**，因为它建立在"运行时自动推断结构"
这一错误设想上。适配是人工 / agent 的离线动作（§4.3），骨架来自适配产物，
所以下列五个曾被列为未决问题的项**全部不再存在**：

- **折叠判据冲突**（trie 的 `sameSubtree` 只比 shape/dtype，看不出 `compress_ratio` 这类语义变体）
  → 折叠仍由适配产物决定，trie 折叠只用于未适配视图。
- **无参算子锚定语法**（trie 不给顺序，需要表达"插在哪"）
  → 不需要。适配产物本身就是有序声明，形态与现有 `ops/index.js` 一致。
- **双骨架**（无 header 时谁当骨架）→ 不是冲突。骨架恒为适配产物，无 header 只影响数值来源。
- **节点 id 归属权** → 节点 id 恒来自适配产物；业界（ONNX `op_type`/`name`、MLIR op 名/位置、
  torch.fx `target`/`name`、modelmap `cls`/`id`）一致地把类型标识与实例标识分开，
  且**无人归一化实例路径**。跨模型可比性由角色表（§4.3 层 1）承担。
- **收益边界** → trie 不产生任何边，§2.1/§2.2/§2.5 与它无关。

---

## 原则收口对照

| 波次 | 收口的原则 |
|---|---|
| W0 | §2.2（部分）、§3.2 与 §8.1 的检查手段 |
| W0.5 | —（机械收敛；维护 §8.1 计数不升） |
| W1 | §3.1、§3.5 |
| W2 | §8.2（ops 侧） |
| W3a | §2.1、§4.3（层 1 角色表 + 配方表）、§8.2（layers 侧） |
| W3b | §4.7、§4.3（配方接管 Job B） |
| W4 | §2.2（evidence 三值齐全） |
| W4.5 | §4.3（层 2 映射表 + 禁止运行时猜测）、§4.4、§4.5、§4.6 |
| W5 | §3.2、§3.3、§3.4 |
| W6 | §2.2（UI 区分）、§4.2、§4.4（UI 展示侧） |
| B | §5.1–§5.4、§6.1、§6.2、§6.3 |
| C | §7（自定义参数侧） |
| D | §6.1（SRP 侧）、错误处理一致性 |
| E | §6（UI 分层） |
| 未排期 | §2.5 残差边、§6.4 来源解析归并、§7 国产芯片条目、§8.1 基线下调（待 W3a+W3b+W4.5 完成后重新测量）、attentionKind 家族品牌 id 改名（`qwen35_full` → 组件名；会改变输出 attributes，需单独拍板） |

每波完成后，从 [`principles.md`](principles.md) §10 例外登记中删除对应条目。
§10 清空之日即本文作废之日。
