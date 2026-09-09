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
| **M6** 成本分层 | W5 | ✅（2026-09-08）：主链查表 + ERT 分离 + 五路 max；cost 数值变化仅 routed swiglu k 语义修正（校准结论）；未实现算子返回 `null`；换卡只走表乘法（用例证明） | §3.2、§3.3、§3.4 |
| **M7** 诚实性上界面 | W6 | ✅（2026-09-08）：诊断面板（gaps/ambiguous/未适配 banner）+ evidence 数据契约与三轴样式（e2e 断言）+ 五类瓶颈时间展开 + value_source 徽标 | §2.2 UI 侧、§4.2、§4.4 |
| **M8** vision 完成 | V1 ✅（词表+绑定+kv_b 修复）/ V2（恒等式域拆分，进行中）/ V3（visualTokens 用户输入+成本链验证） | V2：38 vision 模型 ratio 收敛；V3：loads.visionTokens 输入 |
| **M9** 维护基线 | ✅ | §10 三态快照 + MAINTENANCE.md（docs/MAINTENANCE.md） |
| **M11** 诚实性收口（2026-09-08 四路审计后重定义，原名"冗余清扫 + 并行/通信层对齐"） | P0 正确性与地基 7 条 + P1 诚实性信号补齐 7 条 + P2 清洁与文档 8 条，详见下方 M11 专节。改名理由：四路审计（结构/成本/UI+后端/文档）证明主要欠账不是冗余代码，而是"算不出来就说算不出来"的承诺在最后一公里被吞 | M11 专节验收标准五条 |
| **M11.5** 结构边界调整（M11 后单独一波，2026-09-08 裁决移出） | plan.js 迁 `config/`（cost/{derivedWeights,memory,parallel}.js、formulas/extractor.js、ops/index.js 三层 5 文件消费实锤，它只从 config 派生却住在 model_executor/）；formulas↔model_executor 目录环解耦（`ops/index.js:1` → formulas，`extractor.js:29-32` → model_executor，目录级双向） | 动 import 拓扑牵连基线哈希，必须整体可回退，不与 P0 混做 |
| **M12** 并行策略功能扩展（后期） | C 档：KV keep-ratio 压缩档位、overlap 参数化、per-stage 通信五路 roofline——越过"纯理论估算"边界的行为变化，开工前单独对齐；**2026-09-08 移入三项**（原 M11 条目）：AllToAll 补 dp>1 条件（vLLM 口径，唯一有行为变化的条目）；interNode/PD 跨机通信时间（三张公开卡全无 `inter_node` 规格，需补芯片数据 + 接 roofline，`pdKvTransferBytes` 现只给字节量不进 roofline）；后端对账链路（`verification/compare_structure.py` 27 行仅自测调用，无任何真实前后端对账测试；后端 oracle 定位表述与对账方案届时一并单独对齐） | 单独对齐后定 |
| **算子本体两级化** | 2026-09-09 收官（见状态总览专行） | 四条恒等式容差 0 + 生成器 + 台账清零 | 差分测试验收 + 基线先行 |

---

## 2026-09-09 起的排期（算子本体六波收官后重排）

现状基线：node 312/312 · pytest 158 · 护栏 exit 0 · verify:models 59 ·
docs:check 一致 · e2e 9+1。四条恒等式容差 0；DECOMPOSE_PENDING 清零；
MAINTENANCE.md 棘轮已回写。

### N1 = M11.5 结构边界调整（✅ 2026-09-09 三阶段收官，见状态总览；原范围如下存档）

- **范围**（2026-09-08 裁决定稿，不变）：
  1. `model_executor/plan.js` 迁 `structure/config/plan.js`（消费方：
     cost/{derivedWeights,memory,parallel}.js、formulas/extractor.js、
     ops/index.js 等三层——「从 config 派生却住在 model_executor」）；
  2. formulas↔model_executor **目录环解耦**：环现为
     `ops/index.js:1 → formulas/index.js` 与
     `formulas/extractor.js:33-34 → model_executor/{dims,layers/vision}.js`；
  3. 共享 bytes 助手抽取（counts.js/matmul/sparse 的字节公式单处化）。
- **验收**：import 拓扑图（madge 或等价）无环；基线哈希与恒等式逐位不变；
  §8.1 家族名棘轮 ≤14（预期下降：配方表接管后）。
- **风险与回退**：动 import 拓扑牵连基线哈希——基线先行（N0 = 现状全绿即基线）、
  单独一波、不与其他项混做；每步重生成 + 人工审 diff。
- **不做**：不改任何公式数值；不动 operator id。

### N2 = 登记项小波（N1 后，可碎片化推进）

1. **逐 op compute-dtype 声明**：actions 增加可选 computeDtype，roofline 按
   op 选费率行（mhc_pre/mhc_fused_post_pre 声明 tf32——TF32 费率行已就位
   07f6aab）；否则 mHC 类 TF32 GEMM 永远按 bf16 判 bound。
2. **safetensors 头部 per-tensor dtype 接入证据库**：fetch-evidence.mjs 增读
   头部（8B 长度 + JSON），dtype/shape 落 L3 index；在场时压过
   paramDtypes 推断层（终态方案，hf-mem 同法）。
3. **量化 scale 的动态表**：dynamic 表的路径模式全部走 canonicalModulePath
   桥接（b4ff157 已立规则），新模型只加数据不加代码。

### N3 = M10 扫尾（已销案，无剩余工作）

旁路 C/D/E、昇腾条目、qwen35_full（值非分派，保留）、§2.5（等 IR 扩展，
保持登记）——状态总览已回写，无遗留动作。

### N4 = M12 并行策略扩展（**开工前需用户对齐**）

前置条件（缺一不开工）：
1. 芯片表 `inter_node` 数据补齐（三张公开卡均无该规格）；
2. 后端 oracle 定位表述裁决（`verification/compare_structure.py` 现仅自测）；
3. C 档三项（KV keep-ratio / overlap 参数化 / per-stage 通信五路）越过
   「纯理论估算」边界，逐项确认行为口径。

| **M10** 小项收尾 | ✅（2026-09-09 销案，见状态总览） | 旁路 C/D/E、昇腾条目均已落；qwen35_full 保留现名（值非分派）；§2.5 保持登记 |

---

## 状态总览（2026-09-08 复核）

| 波次 | 状态 | 实际落点与原设计的偏差 |
|---|---|---|
| W0 / W0.5 | ✅ | 按计划；护栏基线 16→15（D1 后首次下降） |
| W1 | ✅ | 42 条目动作向量 + 芯片 schema（vector_flops/sfu_ops 已入 chips.public.js） |
| W2 | ✅ | scaledDotProductTail + 59 模型 spec 树哈希基线（新增 oracle） |
| W3a | ✅（→W3-A/B） | 组件表=layers/ 原地升级；archs/ 取代 adapters 命名； ATTENTION_COMPONENTS 表 |
| W3b | ✅（→W3-C） | plan.js 承接方案决定；normalize 纯字段归一；结构债登记（家族知识 5 住址） |
| W4 | ✅（→W3-D1） | legacySemanticEdges/mergeSemantics/杂项死代码删除 |
| W4.5 | ✅（→W3-B2） | role 连接键绑定；archs/ canonical + FAMILY_OVERRIDES；可逆校验测试 |
| **M2** | ✅ | "normalizeConfig 输出深度相等"语义已由 normalize.golden 取代（C 有意删方案字段） |
| **M3** | ✅ | D2：边 evidence 全非空 + 声明覆盖执法测试 |
| **M4** | ✅ | C：normalizeConfig 不再输出方案类字段 |
| **M5** | ✅ | B2：ambiguous=0（fixture 级）+ 可逆校验 |
| **M6** | ✅（W5-1/2） | 主链查表 + ERT 分离 + 五路 max；routed swiglu k 语义修正（校准结论） |
| **M7** | ✅（W6-1/2） | 诊断面板（gaps/ambiguous/未适配 banner）+ 边三轴 evidence 契约（e2e 断言）+ 五类瓶颈 + value_source 徽标 |
| **M8** | ✅ V1 ✅ / V2 ✅（全模型恒等式断言覆盖，REGISTERED 登记结构缺口容差）/ V3 ✅（visualTokens 用户输入） | vision 词表与绑定、恒等式域拆分、qkv_hidden_size 修复、kimi_k3 KDA 去重计数、MLA g_proj、GLM hc/indexer 登记；详见 details/identity_calibration.md 案例 |
| **M9** | ✅ | §10 三态快照 + MAINTENANCE.md（五重 oracle + 变更纪律） |
| **M11** | ✅（2026-09-08 收官） | 诚实性收口：P0 七条 ✅、P1 七条 ✅、P2 八条（7 ✅ + 1 项探针推翻取消）+ 附加：算子层三缺陷修复（qsa kind 三分支/compressor ctx/GLM-Flash 证据改判 qsa→dsa）、bytes 全量补齐（§3.1c 棘轮 PENDING 清空）、§3.1b/§3.1d 护栏、内存侧基线、第六 oracle、GLM-Flash 恒等式 1.0964→1.0909、V4 0.9828→0.9933；单测 261、e2e 语义断言；详见 M11 专节落地核销 |
| **算子本体重构六波** | ✅（2026-09-09 收官，05fccf6 + 后续 13 个提交） | 两级本体（19 原子 atoms.js + 模块 modules.js，命名对标 vLLM nn.Module）；四条恒等式**全部容差 0**（融合分解 382 组 0 不闭合、权重字节 32/32 逐字节、KV 读分桶夹逼、激活流形状连续性 33434 边）；稀疏部件按算法出处拆 id、分类判据走 config 字段；MTP/residual_add/逐头 norm 权重宽/linear bias/量化 per-matrix 容量（quantBytes.js）/tid2eid buffer 分类/TF32 芯片行；DECOMPOSE_PENDING 清零（Sinkhorn 经 kernel 取证为运行时计算）；operators_reference 机器段生成器 + docs:check；逐层归因工具 diff-weight-identity.mjs。计划文件：~/.comate/plans/算子本体重构与分相位对账_1551cca5.plan.md |
| **M11.5** | ✅（2026-09-09 三阶段收官，0ed4665/d44a292/2095049） | ① plan.js 迁 config/（21 文件路径更新，输出逐位不变）；② 目录环解耦：dims.js + visionDimensions 迁 config/，formulas/ 对 model_executor 的 import 清零（layering.test.js 棘轮固化）；coverage↔public 文件环修复（validateChipEntry 迁 chipValidation.js，madge 0 环）；③ bytes 助手单处化：linear/rmsnorm 的 decompose 片段迁 counts.js（linearAtomSteps/rmsnormAtomSteps），addCounts/softmaxCounts/hashRouteCounts 委托原子（298 组输入 scratch 证明逐位相等），新增 countsAtomsConsistency 漂移守卫 |
| **M10** | ✅（2026-09-09 复核销案） | 旁路 C ✅（5128c24，field_sources + 单位异常警告 + 昇腾 910B4）/ 旁路 D ✅（17f1d65）/ 旁路 E ✅（69e2d18）；昇腾 sfu_rate_source:"vector" ✅（public.js:134）；qwen35_full 改名撤销——销案：W5 已定性为 attention kind 的**值**而非分派（护栏 §8.1 注释在案），保留现名；§2.5 残差边——保持登记（等 IR 扩展，设计决定不变） |
| **M12** | ⬜（后期） | 并行策略功能扩展（C 档：KV keep-ratio、overlap、per-stage 通信）——开工前单独对齐 |

### W5 范围校准（2026-09-08 对齐用）

- **新增**：extractor.js 内 ~20 处 legacy 镜像函数随旧链同批删除（原计划漏列）；
  切换期以 W1 差分测试为双跑断言，删旧链后差分测试退役、identity 恒等式接守；
- **收缩**：第 3 项"正则统一"已部分完成（extractor 侧 W1 已统一 LAYER_INDEX_RE，
  剩 compute.js 单处）；
- **不变**：第 1/2/4/5 项照做；ERT×counts 分离与芯片数量无关的论证仍成立；
- **新增验收**：D2 遗产——cost 改动不得触碰边基线（ops-edge.golden.json）；
  identity 恒等式 2% 容差不回退。

### 旁路里程碑（2026-09-08 修复散行渲染，内容未改）

| 里程碑 | 档位 | 完成判据 | 收口原则 |
|---|---|---|---|
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
>
> **review 补记（2026-09-08，本轮仅登记不改）**：
> 1. cost 五文件（compute/derivedWeights/memory/parallel/extractor）对 plan.js 由
>    数据耦合升级为代码耦合——收口时 plan.js 应移至 config/ 与 normalize 共享
>    解析原语（textConfigOf/modelTypeProbe 两份解析合一）；
> 2. normalize 数字派生内残留的家族探测（kimi fused / glm5_next / deepseek_v4）
>    是否也归组网，待配方表落地时与 sharedExpertIntermediateSize 一并裁决；
> 3. plan.js 的"显式声明优先"层语义边界已写入其头注释：raw config 若真出现
>    方案字段键会被静默采纳，HF 配置不含这些键，风险可控但需知晓。

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

## 旁路 B 后端保留 + source_ref（2026-09-08 裁决：不降级）

> **裁决**：后端保留，定位 = "验证前端是否正确"的 transformers oracle
> （与 principles.md §6.1/§6.3 原始设计一致）。"降级"方向作废；本旁路的
> 有效剩余范围收窄为 source_ref 采集与 /api/verify 的对齐强化（W6 前再对齐）。

## 旁路 B（原"后端降级"）+ source_ref（与 W1–W6 无依赖，可并行）

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

## M11 诚实性收口（2026-09-08 四路审计后修订，用户裁决定稿）

审计方式：四路并行只读审计（结构层 / 成本层 / UI+后端 / 文档对齐），全程零代码改动；
关键断言均经主循环复核（generic-config 崩溃用 node 探针复现，键名不匹配与
compactControls 死分支 grep 实证）。五重 oracle 实跑全绿：护栏 exit 0、单测 248/248、
verify:models 59/59、pytest 148、e2e 9 passed + 1 skipped。

**审计主结论（比单个 bug 更重要）**：三处独立发现同构——护栏 §3.1 验证的是"注册表条目
挂了 counts 函数"而非运行时路径（31/42 条空转）；`roleBinding.test.js` 与 `ui.test.js`
各自测键名两侧而不测接缝（歧义面板生产死亡）；`roofline.test.js` 直接构造入参测函数而
不走 UI 入口（五路退化三路无人发现）。**每一处都是"两端都测了，中间没测"。**
 sixth oracle 与护栏判据修订因此进 P0 而非 P2——它们是防止同类问题再生的两条。

### P0 正确性与地基（7 条，顺序有依赖，主循环做）

1. **e2e 反向断言修正 + 全链路测试（第六 oracle）**
   `frontend/e2e/viewer.spec.js:79,108` 的 `not.toMatch(/unknown/i)` 是文本断言，
   字面含义"界面不许出现 unknown 字样"——它奖励隐藏不确定性、惩罚诚实展示。
   改为语义断言（如"内置模型 bound 不得为 unknown"）。并新增第六 oracle：
   全链路测试——从 UI 入口（CostSummary 模型级 / lens 节点级）到 roofline，
   对每个内置模型断言：bound 分类时五路时间全部非 null，或 missing 显式列出；
   actions 必须来自 counts 通道而非 legacy 分支。
   （用户裁决：P0-1 暴露出需要完善测试，全链路测试。）
   **2026-09-08 联网调研定稿**：断言方式照抄 Playwright 官方惯例（data-testid +
   状态断言，全文文本匹配只用于"测的就是文案"）——扩展本仓库 W6 已有的
   `data-evidence` 契约先例，新增 `data-bound` 属性；第六 oracle 参数化照抄
   transformers 测试套件迭代 MODEL_MAPPING 逐条目一测的模式（node --test
   用 59 模型循环，失败信息带模型名）。
2. **generic-config 必崩路径修复（先联网调研）**
   `models/generic.js:1` 只 import `textDecoderNetwork`，`:10` 调用未导入的
   `networkSpec`；`buildStructureFromConfig({model_type:'mystery',...})` 实测抛
   `ReferenceError`。这条"config 字段不足无法组网"的兜底出口设计好、有诊断
   （collectDiagnostics.js:23-28）、有文档（modules.md:109）、却必然崩溃——
   59 个内置模型都有 `num_hidden_layers`，248 个测试测不到。
   开工前联网调研 transformers / llama.cpp / vLLM 对不支持架构的处理模式
   （报错 / 降级 / 空结构+告警），按调研结论定兜底行为，补该路径测试。
   （用户裁决：P0-2 联网补充信息。）
   **2026-09-08 联网调研定稿**：vLLM `_raise_for_unsupported` 与 transformers
   同一模式——报错即枚举全部支持项（"Model architectures ['X'] are not
   supported for now. Supported architectures: [...]"），且区分"不支持"与
   "解析失败"两类，**均不做兜底组网**。落点改为：删除 generic.js 的假兜底
   `buildGenericConfigNetwork`，`MODEL_BUILDERS` 查找 miss 时抛枚举式结构化
   错误 → collectDiagnostics 转 unsupported 诊断 → UI banner 展示（P0-3）。
   "遇到没见过的模型"链路 = 注册表查找 → 枚举报错 → 诊断 → banner，四段全通。
3. **unsupported 前端告警**
   `diagnostics.unsupported`（generic-config）与 `diagnostics.warnings`
   （architecture-inferred / missing-layer-count，collectDiagnostics.js:12-28）
   目前零 UI 消费者：不支持的模型静默画出一张完整假图，只显示含义完全不同的
   "未加载 checkpoint 真值"。接入 DiagnosticsPanel：不支持 = 显式告警。
   （用户裁决：unsupported 是有用的，不支持需要前端告警。）
4. **actions 断链接通**
   `CostSummary.jsx:117-122` 与 `diagram/lens.js:35-40` 均未传 actions，
   `roofline.js:12-25` legacy 分支把 vector/sfu 伪造为精确零 → 五路退化三路，
   bound 只在 matrix/memory/comm 间产生。修法：两个调用点接 counts 通道，
   删 legacy `?? 0` 分支。原 M11 条目"nodeCostPerCard 补 vector/sfu 投影"
   由本条吸收（nodeCostPerCard 经 lens 透传，同一断链同一修）。
5. **counts.bytes 接入访存侧（用户裁决：接进去）**
   事实链：extractor.js 的 `matmul` 手搓分支（:353-358）bytes 恒 0，
   注册表上 `attentionCounts`（F2，含 KV-cache 写回流量模型）从未运行；
   counts.bytes 经 aggregate.js:29 汇成 `cost.actions` 后全链路无消费者
   （ui.js 从不读它）——访存侧实际由 memory.js 的 activationTensorBytes 供数。
   修法：访存侧统一走 counts 通道，修复 matmul bytes=0（启用 F2），
   `memory.js:20-42` tensorElements 降级为对 extractor 的薄适配
   （其 4D/5D 特例如属必要则搬入 extractor，不得丢失）。
   访存数值变化需全模型基线复核并记录行为变化。
   **2026-09-08 落地**：公式侧修复 matmul scores/context（一阶 Q/K/S/V 流量）、
   KDA/linear state（forward 级读+写递归状态，形状照 vLLM kda_state_shape）、
   causal_conv1d/short_conv（窗口宽读+写）、embedding gather（每 token 读一行
   写一行）、routed swiglu 激活段（复用 F5，44 个 MoE 模型 vector/sfu 补齐）；
   split 家族显式声明 view 语义零流量（非漏算）。消费侧：CostSummary 与 lens
   的访存侧切到 counts.bytes（weights 仍以 memory 侧为权威保 what-if）。
   基线 diff 审阅：仅 actIn/actOut（59）/vector/sfu（44=MoE 数）/bytesMoved/
   times.memory 变化，零 bound 翻转，memory.js 侧零变化。
   **登记剩余**：qsa/minimax_sparse/dsv4 稀疏注意力族 bytes、KDA conv 历史
   小项、counts.weights 与 what-if 权重的统一（M11.5）；memory.js
   tensorElements 保留为 residency 展示供数（VRAM 指标），不再进 roofline。
   （后续进展见下方"算子层缺陷与 bytes 补齐"——登记项大部分已被消化。）

### 算子层缺陷与 bytes 补齐（2026-09-08 用户质疑驱动，方案 A 裁决）

用户质疑"这些 attention 是不同的，怀疑算子开始就有问题"——三处实证成立，
这是继四路审计后第二次由用户直觉定位到 oracle 盲区（五重 oracle 守护
"树和总量"，不守护"每个算子的小项"；小项错误在 2% 容差里隐形，bytes
在 P0-4 前无消费者）。

**缺陷 1（结构性）：`qsa_attention` 一个 operator_id 装三种算法。**
探针实测 16 模型 × 三种 `attention_kind`，extractor 的 case 公式不按
kind 分支，一份公式伺候访存量纲不同的算法：

| attention_kind | 模型 | 算法 | 读取的 KV |
|---|---|---|---|
| `qsa` | 4（Qwen3.8-Flash-Next×2、GLM-5.3-Flash×2） | QSA 稀疏选择 | 完整 K/V 选 budget=2048 |
| `dsa_sparse_mla` | 7（DSV3.2、GLM-5/5.1/5.2/5.2-FP8/5.3） | DSA indexer | 选中 token 的 KV |
| `dsv4_sparse_mla` | 5（V4 全系） | C4 压缩稀疏 | 压缩 4× 的 KV（量纲差 4 倍） |

`ops/index.js:41` 注释自认"真语义不同的变体（dsa、dsv4、qsa）不并入本
helper"，但 id 与公式层未贯彻。**方案 A（用户裁决）**：case 内按
attention_kind 三分支各配 bytes 公式，矩阵公式不动（恒等式已校准），
不动 operator id（方案 B 拆 id 留 M11.5 评估）。

**缺陷 2（证据缺口）：Qwen3.8-Flash-Next、GLM-5.3-Flash 的稀疏模板无
源码证据。** 两模型 config 无任何 indexer 字段（探针 `{}`），模板的
selected_tokens=2048 来自 normalize 推导；本地无 modeling。GLM-5.3-Flash
走 QSA 而 GLM-5.3 走 DSA，是产品事实还是建模臆断待源码裁决。待办：
fetch-evidence 取证（ModelScope 可达）；取证失败则该两家族 bytes 公式
保持 PENDING 登记并降级声明。

**缺陷 3（矩阵侧，已修）：`mla_kv_compress` 对 V4 的 macs 错 16-32×。**
ctxBuilder 用 `kvLoraRank+qkRopeHeadDim` 拼 out 维，V4 无 kvLoraRank →
out=64，实际应为 1024/2048（模板 output_shape 本来就对，counts 侧没用它）。
探针：compressor 实际 1.07G vs 应为 17.2G。**修复**：out 维以节点自身
`staticWidth(output_shape)` 为权威、config 组合仅作回退；非 V4 模型数值
零变化（R1/V3.1/Qwen 全系 ratio 不动）。收益：V4-Flash 恒等式
0.9828→0.9933、V4-Pro→0.9951。

**已落地（2026-09-08）**：
- minimax_sparse_attention bytes（F2 口径：Q 读 + 选中 KV 读 + scores/probs
  中间量 + O 写 + KV cache 写回——sparse 为融合算子故 cache 写回在本叶，
  dense 侧由 k/v_proj linear actOut 计费，两侧账目自洽）。证据：HF 仓库
  无 modeling（API tree 核实），采用 transformers 库随附实现作二等来源，
  已入库 models/MiniMaxAI/MiniMax-M3/ + evidence-manifest.json。
- KDA/linear state 补 conv 历史（stateUpdateCounts 与 memory.js
  linearStateElementsPerLayer 同源同式）。
- mla_kv_compress ctx 修复（缺陷 3）。
- embedding gather（bytes 完整性棘轮实测抓出：embedding 是无 operatorId
  的结构节点，被"非算子零向量"规则计为零流量；extractor 加 type 分支，
  gather 无 MACs，matrix 恒 0）。
- §3.1c bytes 完整性棘轮（bytesCompleteness.test.js）：59 模型全 leaf
  扫描，全零访存分量必须显式登记（VIEW_OPS view 语义豁免 +
  PENDING_UNMODELED 未建模清单，落地后清空）。

**待落地（方案 A 执行清单）**：
- ~~qsa_attention case 按 kind 三分支 bytes~~ **✅（2026-09-08）**：qsa 报告
  （/tmp/m11-formulas/qsa.md）落地——前提修正：Kimi-K2.5/K2.6/K2.7 不产生
  qsa_attention（纯 MLA），实际 16 模型 = Qwen3.8-Flash-Next×2(qsa) +
  DSV3.2 + GLM-5 系×8(dsa_sparse_mla) + V4×5(dsv4_sparse_mla)；公式按
  kind 分派读宽/共享度（MLA latent 族 kvHeads=1、读宽 kv_lora+rope；DSV4
  MQA kvHeads=1；逐头 QSA 按 GQA kvHeads），kvWrite 仅逐头变体（latent
  变体的 cache 写已由 kv_a_proj/compressor 计费）。matrix 零漂移
  （恒等式 5/5，ratio 与上轮一致）。
- ~~dsv4_swa/compressed bytes~~ **✅（2026-09-08）**：dsv4 报告
  （/tmp/m11-formulas/dsv4.md）落地——swa 缓存每 token 一份 headDim 宽
  KV latent（K/V 共享，权重表无 V 扩展投影实证）、compressed 读压缩缓存
  2·headDim 且无 kvWrite（归 compressor 叶）；swa/compressed 拆成独立
  case。基线 diff 审阅：actIn/actOut/bytesMoved/times.memory 恰好 16
  模型，零 bound 翻转。**PENDING_UNMODELED 已清空**，§3.1c 棘轮全绿。
- ~~dsv4_hash_route 的 tid2eid 路由表~~ **✅（2026-09-08 C 路落地）**：
  tableRows = vocabSize×expertsPerToken 实算（129280×6≈775,680 条目/层），
  weights 补齐；基线 diff 恰好 5 个 V4 模型 weights 变化。
- cost_counts.md 42 条目补 bytes 公式说明（并 P2 文档批）。
- 压缩层 hybrid 滑窗读（dsv4 报告线索）待查。
- 证据缺口仍在（不阻塞公式，阻塞口径裁决）：Qwen3.8-Flash-Next /
  GLM-5.3-Flash 稀疏模板无 modeling 源码；GLM-5.3-Flash(QSA) vs
  GLM-5.3(DSA) 待源码裁决——取证到后若算法有出入，按缺陷 1 同模式
  修公式即可（棘轮 + 基线已就位）。
- 共享 bytes 助手抽取：attention bytes 公式已是第三次手抄
  （counts.js/matmul/sparse），防抄写漂移，M11.5 评估。
- A2 口径声明：scores/probs 中间量（4×）按理论上限计，flash kernel 下
  不存在——保持理论口径（工具定位即理论估算），如需 kernel 级口径
  另行对齐。
6. **真值歧义键名对齐**
   `graphTruth.js:238` 出口发 `ambiguous_truth_matches`，`cost/ui.js:57` 读
   `graph_ambiguous_truth_matches`（内部键 `:112`）→ DiagnosticsPanel 的
   "真值绑定歧义——绑定已放弃"面板生产永不触发，歧义节点（graphTruth.js:88
   `return node` 丢弃真值）无任何提示。对齐键名 + 补接缝测试（两侧键名的
   现有测试均各自通过，接缝零覆盖——教训记入 MAINTENANCE）。
7. **护栏 §3.1 改运行时判据**
   `scripts/check_principles.sh:59-61` 只查"FORMULAS 每条挂了 counts 函数"，
   运行时 extractor.js 双分派：31 case 手搓 switch（提前 return）+ 11 条
   ctxBuilder 走注册表（实测 31+11=42、零重叠）。31 条的注册表 counts 引用
   是被护栏认证过的死代码。改为"运行时终止于 counts.js"，或显式登记
   31 条手搓例外清单（短期）；长期方向 = 手搓分支逐条搬入 counts.js。

### 成熟方案对照（2026-09-08 联网调研，零自研，用户裁决"先查方案再给计划"）

| 条目 | 成熟方案出处 | 照抄哪部分 | 本仓库落点 |
|---|---|---|---|
| P0-1 e2e + 第六 oracle | Playwright 官方惯例（data-testid + 状态断言；文本匹配只用于"测的就是文案"）+ transformers 测试套件逐条目参数化 | `data-bound` 属性契约（扩展自家 `data-evidence` 先例）；59 模型循环逐条断言 | ReactFlowStructureDiagram.jsx / NodeDetailPanel.jsx / CostSummary.jsx 挂 `data-bound`；viewer.spec.js 语义化；新测试文件全链路 |
| P0-2 不支持架构 | vLLM `registry.py _raise_for_unsupported` + transformers AutoModel 报错 | **报错即枚举支持项**，区分"不支持"与"解析失败"，不做兜底组网 | 删 `buildGenericConfigNetwork` 假兜底；MODEL_BUILDERS miss 抛枚举式错误 |
| P0-4/P0-5 counts 通道 | MIT Accelergy（MICRO'52 论文 + ISPASS'20 教程） | **action counts 是估算器唯一接口**——访存动作（GLB access/buffer read）与计算动作（MAC compute）同处一个 counts 命名空间，无第二本账；ERT 只做 action→成本映射 | 两入口改传 counts 通道删 legacy 分支；matmul bytes 启用既有 F2；memory.js 降薄适配 |
| P0-6 键名对齐 | 契约测试分类学（provider/consumer-driven/bi-directional） | **consumer-driven**：消费者声明所需字段，用生产者真实输出验证；否决 ajv/Pact（单仓库固定生产消费对，过重） | 接缝测试 import graphTruth 真实出口喂 diagnosticsModel；禁手捏 fixture（ui.test.js:30 手捏形状正是断缝根因） |
| P0-7 护栏判据 | registry completeness 参数化测试（transformers/vLLM 惯例） | 迭代注册表逐条执行断言，失败点名条目；豁免必须显式登记（vLLM "登记即存在"精神） | 护栏迭代 42 条 FORMULAS 逐条执行 counts()；手搓条目显式登记 `runtime:"extractor-switch"` |

否决记录：LLM 语义断言（用不确定物守护诚实性，方向错误）；ajv/JSON Schema 接缝校验（场景不匹配）。

### P1 诚实性信号补齐（7 条，可交 agent 并行，改动集中在 components/ + cost/ui.js）

1. chip coverage 可见性：`DetailWorkspace.jsx:158` 恒传 `compactControls`，
   `ArchitectureTab.jsx:344` 的覆盖率行永不渲染——芯片缺项与
   "缺少 inter_node.bandwidth，跨节点偏乐观"警告全部丢失。
2. `roofline.missing` 展示（roofline.js:45-56,90 产出，零读取）：
   unknown 已显示但"缺哪一项"不可见——最可惜的一块。
3. `eta.vector ?? 1` / `eta.sfu ?? 1`（rates.js:31,34）默认 100% 效率未披露；
   DEFAULT_EFFICIENCY（efficiency.js:4-9）无这两键。
4. `macsSource` + `value_source` 扩到 Cost Lens 与汇总条
   （= M7 范围项 5 欠账，refactor_plan.md 原 :465 W6 第 5 项）。
5. `checkpoint_truth_error`（toStructureNode.js:117 写入真实异常文本）与
   config/checkpoint endpoint 展示——HF 失败静默切 ModelScope
   （loadModelArtifacts.js:112-125）用户不知情。
6. `projectPlan.ok=false` errors 展示（现在只是 stage 行不渲染）。
7. 节点级 `bound=unknown` 与"未开 Lens"视觉可区分
   （ReactFlowStructureDiagram.jsx:98 / NodeDetailPanel.jsx:56 现以不渲染表达未知）。

### P2 清洁与文档（8 条，可交 agent 并行）

1. 真死代码删除（实测清单，替代原"导出引用矩阵"）：
   memory.js `product`(:14) / `BYTES_PER_DTYPE`(:138) / `activationPeakBytes`(:123)；
   chips/index.js barrel（零引用）；compute.js `computeMacsForNode`(:33-37，零调用)；
   compute.js `macs`(:64，与 compute_macs 同值重复)。
2. 多余 export 收窄（只删关键字，函数活跃）：comm.js 3 个、parallel.js 4 个。
3. **新纪律（用户裁决）**：看起来冗余的代码可能是未完成系统接入
   （本案：counts.bytes 算完无消费者、31 条手搓绕开注册表）——
   **补全接入优先于删除**；删除仅适用于确认无意图、无消费者的代码，
   且区分"真死函数"与"多余 export"。写入 MAINTENANCE.md 变更纪律。
4. 42 条 `// ref:` 来源标注 + 护栏第四项：当前 ref: 计数 0
   （grep 实证 formulas/ 与 cost/ 全部为 0），principles.md:178-184 已写成
   生效硬门槛——补护栏使门槛成真，或改口径（倾向前者，与 §8.1 棘轮同构）。
5. ~~comm.js 删路径正则兜底~~ **✗ 2026-09-08 探针推翻**：兜底是活路径——
   50 个节点（Kimi/GLM 的 routed_expert_down_proj，路径段不匹配 experts
   排除正则）正走 o_proj/down_proj 兜底分支计通信量，删除将改变其字节。
   审计"4 个 role 对齐"属实，但"兜底为死路径"推论不成立。兜底保留。
6. weightBytesPerCard 切分规则表化。
7. layerSpanForNode 收敛至共享正则。
8. 文档现状化（四路审计的文档修订清单）：MAINTENANCE 数字修正
   （246→248、`../.venv`→`.venv`、全跑 1 分钟→3.2 分钟、已修 kv_b 仍列残差、
   Qwen3.8 残差符号已反转）；modules.md 死路径（cost/skeleton.js、
   cost/mergeSemantics.js 不存在）与已删 6 条公式清理、§7 缺 3 条 vision 公式；
   graph_sources.md legacy semantic matcher 表述删除、私人绝对路径删除；
   恒等式数字刷新（K3 1.29→1.0434、K2.5 0.9989→0.9949、M3/DSV4-Vision/
   DSV4-Pro 三条补记录）；cost_counts.md 两条 generic-only 算子标注
   （linear_attention / linear_attention_gate 目录 59 模型永不触发）；
   M8 的 REGISTERED 9 项容差在 MAINTENANCE 注明（现"2% 容差"表述误导）；
   W4.5 ✅ 与"canonicalModulePath 未删"的自相矛盾裁决（:307 遗留登记为准）；
   M7 范围项 5 待 P1-4 落地后改为完整 ✅。

### 验收标准（修订，替代原"零未引用导出；PP/DP/EP 切分矩阵测试；每公式有 source"）

1. 真死函数零残留；多余导出收窄为零（两句分开判，禁止混同）。
2. 每条公式运行时可追溯：终止于 counts.js，或显式登记手搓例外（护栏 §3.1 新判据）。
3. 42 条有 `// ref:` 且护栏第四项守着（grep 计数 ≥42）。
4. 第六 oracle 全绿：59 模型全链路（UI 入口→roofline）bound 分类，
   五路时间非 null 或 missing 显式；e2e 反向断言已语义化。
5. PP/DP/EP 切分矩阵测试（保留原条）。

### 并行分工

A 路（agent）P2 来源标注 + 文档现状化 ｜ B 路（agent）P1 七条信号补齐 ｜
C 路（主循环）P0 七条。A 只动 formulas/ 注释与 docs/，B 只动 components/ 与
cost/ui.js，C 动 roofline.js / extractor.js / generic.js / graphTruth.js /
check_principles.sh / e2e/——三者文件不交集。

### 落地核销（2026-09-08 收官）

**验收五条实况**：
1. ✅ 真死函数零残留（computeMacsForNode/stageForLayer/chips barrel/重复
   macs 字段删除；memory.js 三符号取消 export）；多余导出收窄按实测收缩
   ——测试直接引用的细粒度导出保留（重写测试收益为负），零消费者 3 个收窄。
2. ✅ 运行时可追溯：§3.1b 42 条可达（手搓 31/ctxBuilder 11/豁免 0）。
3. ✅ // ref: 42/42（§3.1d 护栏第四项生效，A 路任务 1）。
4. ✅ 第六 oracle 全绿（59 模型全链路五路时间可得）；e2e 反向断言语义化
   （data-bound 契约）。
5. ✅ PP/DP/EP 切分矩阵测试（原有保留）。

**P1 七条实况**：全部 ✅（P1-1 coverage 可见性/P1-2 roofline.missing/
P1-3 η 披露/P1-4 macsSource+value_source/P1-5 truth_error+endpoint/
P1-6 projectPlan errors+无效语义修正/P1-7 unknown 徽标）。其中 P1-6 顺带
修正语义 bug：无效 plan 此前谎报"显存不足"。

**P2 八条实况**：1/3/4/6/7/8 ✅；2 收缩（理由如上）；5 **✗ 探针推翻**
（兜底为活路径，50 节点实证，删除取消——审计结论纠错已记录）。
附加完成：hash_route tableRows、bytes 完整性棘轮（§3.1c）、算子层三缺陷
（kind 三分支/compressor ctx/GLM-Flash 改判）。

**遗留（不阻塞收官，全部登记）**：
- cost_counts.md 42 条逐条 bytes 公式明细（口径总述已加，逐条待 P2 续）
- 压缩层 hybrid 滑窗读线索、共享 bytes 助手抽取（M11.5 评估）
- Kimi-K3 k3-index.json 违反 3b 入库（清理待办）
- 模块级 attention_kind 改判波及的 e2e 断言口径（如有）随下次 e2e 观察



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
| M11 | §3.1（counts 运行时接线 + 护栏判据）、§3.3（诚实性上界面补全）、§7（// ref: 来源标注） |
| B | §5.1–§5.4、§6.1、§6.2、§6.3 |
| C | §7（自定义参数侧） |
| D | §6.1（SRP 侧）、错误处理一致性 |
| E | §6（UI 分层） |
| M11.5 / M12 | 未排期——M11.5 结构边界调整（plan.js 迁移 + formulas 目录环）、M12 移入项（AllToAll dp>1 / interNode 与 PD 跨机 / 后端对账链路）、后端 oracle 定位表述 |
| 未排期 | §2.5 残差边、§6.4 来源解析归并、§7 国产芯片条目、§8.1 基线下调（待 W3a+W3b+W4.5 完成后重新测量）、attentionKind 家族品牌 id 改名（`qwen35_full` → 组件名；会改变输出 attributes，需单独拍板） |

每波完成后，从 [`principles.md`](principles.md) §10 例外登记中删除对应条目。
§10 清空之日即本文作废之日。
