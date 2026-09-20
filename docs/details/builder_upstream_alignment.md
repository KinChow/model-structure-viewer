# 重构方案 · builder 层对齐 vLLM/SGLang（不另辟蹊径）

> **状态（2026-09-20 更新）**：本方案 P0–P4 已全部落地。此后的 review 清理（D2）进一步把 P2 新增的
> 3 个薄委托入口文件**内联进注册表直连**、把 P4-② 的裸委托守卫**重写为注册表级** `SGLANG_REUSED_ARCHITECTURES`；
> §五两处"待选做小缺口"（topk_method 归一化、e_score_correction_bias 字节叶）也已在 deepseek-gate-bias-attn-sink
> 落地。下文相应小节已按当前代码更新（历史执行叙述保留，回归计数为当时快照）。

目标：以 vLLM/SGLang 为 source of truth，把 MSV 的 model builder 层从"折叠 + 别名 + 三通道注入"改造成
"per-架构入口 + 中性命名 + 共享组件组合 + 上游字段全覆盖 + 一致性守卫"，消除本轮暴露的混淆类问题
（K2.5≠V3 的 MoE 分组、命名误导、"假设同口径"）。

## 一、现状盘点（代码级）

**注册表** `models/index.js`：16~17 个 `architectures[0]` 键 → 14 个 `assembleXxx`。3 处跨模型别名：
- `KimiK25ForConditionalGeneration → assembleDeepseekV3`
- `GlmMoeDsaForCausalLM → assembleDeepseekV32`
- `Qwen3_5MoeForConditionalGeneration / Qwen3_5MoeForCausalLM → assembleQwen3_5`

**builder 厚度**：多数是 3 行壳（设 `attentionKind/defaultLayerKind` 后委托 `textDecoderNetwork/
multimodalDecoderNetwork`）；少数有真实自有代码（`deepseek_v4` ~45%、`qwen4_exp` ~40%、`qwen3_5` ~35%、
`minimax_m3` ~20%）；`deepseek_mtp.js`/`common.js` 是共享装配库。

**每模型差异经三通道注入**（这是复杂度与出错的真正来源）：
1. **normalized 字段**（config 驱动：experts、moe_intermediate、compress_ratios…）
2. **recipe 表** `archs/index.js`（per-arch class 名 + 标志：linearAttentionMode、moeClass、layerMix、pleClass…）
3. **builder opts**（attentionKind、defaultLayerKind、draft）

**根因诊断**：折叠本身不错（builder 编码结构、维度来自 config），错的是——
(a) **命名承载单一模型语义**（`assembleDeepseekV3` 却服务 Kimi-K2.5）；
(b) **上游建模的区分字段漏建**（本轮的 `n_group`/`topk_group` grouped_topk，已修）；
(c) **三通道注入无单一事实源**，谁决定某差异走哪条通道靠约定，易"假设等价"。

## 二、对齐原则（关键边界）

- **对齐"算法/结构/registry 键/字段语义"层** —— vLLM 与 SGLang 在这层一致，即 MSV 的 source of truth。
- **不抄"内核融合布局"** —— SGLang 的 `fused_qkvbfg_proj`、`qkv_conv1d` 融合是运行时性能优化；MSV 是静态
  结构分析器，应展示 **HF/vLLM 参考实现的逻辑模块**（分离 q/k/v/g）。本轮 kimi_k3 前向卡在 SGLang 融合布局
  正是反例。
- **cache/并行/通信口径** 用 SGLang 真机交叉校验（已完成的 框架/并行 profile、算子成本、kernel 口径 路线）。

## 三、重构步骤（按优先级 / 可独立落地）

### P0 · 已完成：grouped_topk 建模
`normalize.js` 加 `numExpertGroup/topkGroup`；`moe/deepseekV4/kimiK3` 三处 topk 算子按 `n_group>1` 标
`group_limited_topk`。效果：DeepSeek-V3.1/V3.2 = grouped(8,4)，Kimi-K2.5 / GLM-5 / GLM-4.7 = 普通 topk，
**结构上可区分**（含"同 builder 内 GLM-5 vs V3.2 不同"）。回归 410/410、verify 60/60、docs 全绿。

### P1 · 上游字段覆盖审计（防"下一个 n_group"）
逐个对 vLLM/SGLang 的 config 类 + MoE/attention 模块，列出其读取的路由/结构字段（如 `topk_method`、
`norm_topk_prob`、`routed_scaling_factor`、`e_score_correction_bias`、`num_expert_group`、注意力的
`use_qk_norm`/`attn_output_gate`/`partial_rotary_factor`…），核对 MSV `normalize.js` 是否覆盖、算子是否消费。
产出"字段覆盖矩阵"，缺口补建。**这是把"个案修复"升级为"系统对齐"的关键一步。**

### P2 · per-arch 入口 + 命名对齐成熟方案（vLLM/SGLang）
**命名准则（对齐成熟方案，不自创）**：每个 `architectures[0]` 一个 **按 HF 架构命名** 的入口（与 vLLM/SGLang
的模型类命名一一对应，如 `assembleKimiK25`↔`KimiK25ForConditionalGeneration`、`assembleQwen3_5Moe`↔
`Qwen3_5MoeForCausalLM`），**不引入 `assembleMlaMoeDecoder` 这类自造的结构化中性名**。代码复用走"入口委托/
继承共享基装配"——共享基保留其**规范模型名**（如 `assembleDeepseekV3`，正如 vLLM 的 `deepseek_v2.py` 是被
Kimi/GLM 复用的规范基，而非改名成结构名）。即：**入口名 = 架构名；复用 = 委托规范基**，两者都按上游习惯命名。

**已落地（P2，结构零变更）**：
- 新增 `models/kimi_k25.js`(`assembleKimiK25`)、`models/glm_moe_dsa.js`(`assembleGlmMoeDsa`)、
  `models/qwen3_5_moe.js`(`assembleQwen3_5Moe`)——分别委托 `assembleDeepseekV3`/`assembleDeepseekV32`/
  `assembleQwen3_5`（镜像 SGLang：kimi_k25 复用 deepseek 组件、`Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM)` 子类）。
- 注册表：`KimiK25→assembleKimiK25`、`GlmMoeDsa→assembleGlmMoeDsa`、`Qwen3_5Moe(×2)→assembleQwen3_5Moe`——
  **每个架构指向与上游类同名的入口**，不再跨模型别名。
- 回归 **411/411**、verify 60/60、golden **无变化**（builder 函数名不进结构输出）。
- **catalog/config 一致性核查**：Kimi-K2.5/2.6/2.7、GLM-5.x 顶层 architectures 与 catalog **一致无 mismatch**
  （此前"不一致"是误读——Kimi 顶层=`KimiK25ForConditionalGeneration`，`text_config.architectures`=DeepseekV3
  只是文本骨干声明；MSV 按顶层路由，正确）。
- **共享基不改结构化名**：`assembleDeepseekV3`/`assembleDeepseekV32`/`assembleQwen3_5` 作为规范基保留，
  被 arch 入口委托——符合成熟方案（vLLM `deepseek_v2` 基 / SGLang 子类）习惯。

> **D2 更新（当前代码）**：上述 3 个薄委托入口文件（`kimi_k25.js`/`glm_moe_dsa.js`/`qwen3_5_moe.js`）已删除，
> 注册表改为**直接映射**到共享基（`KimiK25→assembleDeepseekV3`、`GlmMoeDsa→assembleDeepseekV32`、
> `Qwen3_5Moe(×2)→assembleQwen3_5`）——与 vLLM `_MODELS` 多键指向同一 modeling 更一致，零逻辑透传壳被视为冗余移除。
> 命名误导由注册表行内注释 + `SGLANG_REUSED_ARCHITECTURES` 登记消解（见 P4-②）。

### P3 · 收敛三通道注入（降低约定负担）
- 明确边界：**维度→normalized 字段**；**class 名/命名映射→recipe 表**；**结构拓扑选择→builder/opts**。
- 把散落在 opts 的结构开关（attentionKind/defaultLayerKind）尽量下沉为 config 驱动（已有 layer_types/
  sparse_attention_config 等先例），减少 builder 侧硬编码。

**P3 结论（读 decoderStack.js/decoderLayer.js/各 builder opts 后修正）**：三通道**各自 load-bearing、不宜强行合并**：
- `decoderStackNetwork` 里 opts 只作**兜底**：`defaultLayerKind = opts || (experts?"moe":"dense")`、
  `defaultAttentionKind = opts.attentionKind || "gqa"`；每层实际 kind 由 config 的 `layerKinds`
  （first_k_dense_replace/mlp_only_layers）、`attentionScheduleOf`（layer_types/full_attention_interval）
  逐层决定。
- **反例证明不能合并**：`assembleGlm5Next` 传 `defaultLayerKind:"dense"`，但 GLM-5.3-Flash 有 288 experts
  （MoE）——若按"experts→moe"统一推导会改变其 dense/moe 兜底、动结构。故 opts 的 family 级默认是**必要的**，
  非冗余。
- **三通道的正当分工**：**config(normalized)** = 维度 + 逐层 schedule；**recipe** = class 名/模式标志
  （linearAttentionMode/moeClass/pleClass/sigmoidRouter…）；**opts** = config 无法表达的 family 级默认
  （MLA 家族无 layer_types 时的 attentionKind、draft/MTP 挂载）。三者映射清晰，**强行三合一会破坏 config
  调度型模型**，属反向"另辟蹊径"。
- **唯一可选清理（已记录，未做，避免动 golden）**：主路径 `attentionKind:"mla"` 硬编码 opt 与 MTP 路径
  `ehProjKind` 已有的 `kvLoraRank?"mla":"gqa"` config 推导不一致；可将 deepseek_v3/v32 的该 opt 换成同一
  config 推导以对齐上游"从 config 读"的习惯。属低收益 cosmetic，且 deepseek_v4 的 dsv4 注意力不可这样推导，
  收益有限、暂不做。
- **P3 处置**：不合并（保持三通道各司其职）；以本节 + P1 字段矩阵 + P4 守卫作为"约定显式化"的交付，
  取代原计划的"结构性合并"。

### P3-A · defaultLayerKind 收敛为单一 config 源 + 修 glm5_next/kimi_k3 隐患（已落地）
方案 A 的高价值核心：**把 builder 里硬编码的 `defaultLayerKind` 全部移除**，统一由 `decoderStack` 的
config 推导 `options.defaultLayerKind || (experts?"moe":"dense")` 决定（单一权威源 = config）。
- **glm5_next/kimi_k3 隐患确认并修复**：二者原硬编码 `defaultLayerKind:"dense"`，但都是 MoE
  （GLM-5.3-Flash 288 experts、Kimi-K3 有 routed experts）。经查该 opt 是**死代码**——`layerScheduleOf`
  由 `first_k_dense_replace`（各 3/1）已生成正确的 dense/moe 逐层表覆盖了兜底，故**当前渲染正确**
  （GLM-5.3-Flash = 3 dense + 42 moe，实测）。但硬编码 "dense" 是**误导 + 潜在 bug**：若未来某 glm5_next/
  kimi_k3 模型无 dense-schedule，会被误渲染成全 dense。移除后由 config 推导出正确的 "moe" 默认。
- **改动**：9 个 builder 去掉 `defaultLayerKind` opt（deepseek_v3/v32/v4、glm4_moe、glm5_next、kimi_k3、
  minimax_m2/m3、qwen3/qwen3_moe/qwen3_5/qwen4_exp）。**结构零变更**（golden 不动，411→412 仅加守卫），
  证明所有现网模型要么有 config schedule、要么 experts 推导与原字面一致。
- **守卫（P4 追加）**：`builtinModels.test.js` 新增 "MoE models render MoE layers"——断言凡 config 有 experts
  的模型都渲染出 `fused_moe_mlp` 层，锁死"MoE 被兜底成全 dense"这一 bug 类。412/412。
- **残留 opt**：`attentionKind`（deepseek 家族 "mla"、minimax_m3 "sparse"）+ `draft` 保留为 sanctioned
  family 默认——非 bug，且移入 recipe 需在 deepseek 无 recipe 条目处新建、config 推导对 V4 有边界，收益低，
  暂不动。即 opts 已从 {attentionKind, defaultLayerKind, draft} 收敛到 {attentionKind, draft}，
  defaultLayerKind 归并进 config 单一源。

### P3-A② · attentionKind 也收敛为单一 config 源（已落地，builder 变纯壳）
延续方案 A，把最后的硬编码 opt `attentionKind` 也下沉为 config 推导：
- 新增 `schedule.js:attentionKindOf(normalized)`——与 MTP 路径 `ehProjKind` **同一 config 判据**：
  `sparseTopkBlocks>0→"sparse"`、`kvLoraRank→"mla"`、否则 `"gqa"`；`decoderStack` 兜底改为
  `options.attentionKind || attentionKindOf(normalized)`。
- 移除 deepseek_v3/v32/v4、minimax_m3 主路径的硬编码 `attentionKind`。逐层 `attentionScheduleOf`
  （dsv4/dsa/sparse per-layer）仍优先覆盖，兜底值由 config 推导复现原字面。
- **结构零变更**（golden 不动，412/412）。**主路径全部 builder 现在只带 `{draft}`**（draft 本身由
  `deepSeekMtpChild`/`mtpModuleCount` 从 config 推导）——builder 退化为纯 arch→装配壳。
- 残留合法 attentionKind 硬编码仅在**非主路径**：`deepseek_mtp`（MTP 每块）、`deepseek_v4` 的 dsv4 自定义
  stage（DSpark）、`qwen3_5` MTP 的 `qwen35_full`——都是结构特化点，非 family 默认，保留正确。

**方案 A 最终态**：三通道 → 二通道（**config + recipe**）。builder opts 事实上消解为 config 推导：
defaultLayerKind = `experts?"moe":"dense"`、attentionKind = `attentionKindOf`、draft = MTP config 推导。
每层实际 kind 仍由 config 逐层 schedule 决定；recipe 承载 class 名/模式标志。回归 412/412、verify 60/60、
docs 一致，全程 golden 无变化。

### P3-A③ · 冗余清理（重构暴露的死代码，已删）
系统验证全绿后审计重构暴露的冗余：
- **删死透传**：opts 收敛后，`common.js` 的 `textDecoderNetwork/multimodalDecoderNetwork` 仍透传
  `attentionKind/defaultLayerKind`、`decoderStack` 仍留 `options.attentionKind/defaultLayerKind` 兜底分支——
  但**已无任何 caller 传这两个键**（全 builder 只传 `{draft}`）。删除：common.js 两处形参+透传、decoderStack
  的 `options` 形参与 `options.X ||` 前缀（改为直接 config 推导）、minimax_m3 的 `{}` 实参。**结构零变更
  （golden 不动，413/413）**，证明确为死代码。
- **当时不删（有意，非冗余）**：`kimi_k25.js`/`glm_moe_dsa.js`/`qwen3_5_moe.js`/`deepseek_v41.js` 薄委托入口——
  是 P2 的 per-arch 命名对齐载体；`assembleDeepseekV3/V32/Qwen3_5` 共享基仍被直接注册 + 被委托，无孤儿。
  **（D2 后修正）**：前 3 个纯透传入口已删除、注册表直连共享基（见 P2 D2 更新）——零逻辑透传壳属冗余，命名
  误导改由注册表注释 + `SGLANG_REUSED_ARCHITECTURES` 守卫消解；`deepseek_v41.js`（独立 body，非透传）保留。
- **扫描确认**：17 个改动文件**无未用 import**；§8.1 家族名硬编码 6/基线 6 无回归；legacy root 0。
- 终态回归：`node --test` **413/413**、`verify:models` **60/60**、`docs:check` 一致、`check_principles.sh` 无回归。

### P4-② · 裸委托全审计 + 守卫（回应"全绿 ≠ 能守护"）
**背景**：deepseek_v41 曾是 `return assembleDeepseekV4()` 裸委托 + 陈旧 TODO，却全程 golden 全绿——因为它
**渲染输出正确**（890 等由 config 驱动），而 golden/verify 只守**稳定性（vs 自生成基线）**，守不住"输出对但
代码路径错/语义错"，也守不住"基线本身就是错的"。这是测试哲学的固有边界：**golden = 防回归，非防正确性**；
正确性靠 ground-truth 对账（SGLang registry / header truth / 运行时）。

**全审计（对 SGLang 源码逐个查裸委托是否有复用/继承依据）**：
| MSV 裸委托 | SGLang 依据 | 判定 |
|---|---|---|
| kimi_k25 → assembleDeepseekV3 | `KimiK25ForConditionalGeneration.__init__: self.language_model = DeepseekV3ForCausalLM(...)` | ✅ 上游复用 |
| glm_moe_dsa → assembleDeepseekV32 | `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` 子类；V32 亦 DeepseekV2 基 | ✅ 上游子类 |
| qwen3_5_moe → assembleQwen3_5 | `class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM)` 子类 | ✅ 上游子类 |
| ~~deepseek_v41 → assembleDeepseekV4~~ | SGLang 无 DeepseekV41 类、V4 是 `nn.Module` 独立基、无子类/复用 | ❌ 已修为独立 body |

结论：**除已修的 deepseek_v41，无其它 v41 式错误裸委托**——其余三个都有 SGLang 复用/继承实据。

**新增真守卫** `sglangArchAlignment.test.js`（首版名 "bare builder delegations are justified by SGLang
reuse/inheritance"，扫 `models/*.js` 的 `return assembleX()` 裸委托比对 `SGLANG_JUSTIFIED_DELEGATION` 白名单）：
**任何未登记的委托即红**——若 deepseek_v41 式错误重现（委托到 SGLang 未复用的目标），或新增无据委托，测试立刻失败，
强制作者查上游。这条把"防回归"升级为"绑定 SGLang 复用图的正确性守卫"。回归 **414/414**、verify 60/60、docs 一致。

> **D2 更新（当前代码）**：3 个透传入口删除后，守卫改为**注册表级** `SGLANG_REUSED_ARCHITECTURES`（测试名
> "shared-builder reuse is justified by SGLang inheritance/reuse"）：扫 `MODELS` 注册表，凡**多个 arch 键共享同一
> 装配器**者，除 1 个属主键外其余必须在 `SGLANG_REUSED_ARCHITECTURES` 登记 SGLang 依据（附 file:line）；不变式与
> 原裸委托守卫等价，且更稳（基于注册表结构而非文件文本正则）。

### P4 · 一致性守卫测试（CI 锁死，防漂移）
1. **路由对齐**：从 vLLM `registry.py` 生成期望的 `architectures[0] → 模块归属`，断言 MSV 注册表一致。
2. **折叠等价断言**：对共享同一装配的架构，若其上游被建模字段（n_group、moeClass…）出现差异，测试红——
   强制要么各自入口、要么显式声明等价。
3. **字段覆盖回归**：断言 P1 矩阵里"上游建模的字段"MSV 都覆盖 normalize + 有算子消费。

**已落地（P4 首件）**：`builtinModels.test.js` 新增 "grouped top-k routing conforms to upstream
n_group/topk_group" 守卫——遍历全 60 模型，断言 `topk` 算子的 `group_limited_topk` 标注与 config 的
n_group/topk_group 严格一致（n_group>1 必标且 num_expert_group/topk_group 相符；否则不标）。锁住 P0 修复、
覆盖折叠等价（同 builder 下 GLM-5 vs V3.2 分组不同会被区分校验）。回归 **411/411**。
2/3（vLLM registry 对齐）待补：本地无 vLLM `registry.py`，留后续拉取上游后生成期望表。

**已落地（P4 全三件）**：
1. `builtinModels.test.js` "grouped top-k routing conforms to upstream n_group/topk_group"（P0 锁死）。
2. `builtinModels.test.js` "MoE models render MoE layers"（P3-A defaultLayerKind 锁死）。
3. **路由对齐改用本地 SGLang**（vLLM 不在本地，SGLang 在 `$SGLANG_SRC`）：
   `scripts/gen-sglang-arch-registry.mjs` 扫 SGLang `EntryClass` → committed 快照
   `frontend/src/structure/__tests__/sglang_arch_registry.json`（276 个真实架构）；
   `sglangArchAlignment.test.js` 断言 **MSV 的 17 个架构键 ⊆ SGLang EntryClass 集合**（例外表
   `KNOWN_MSV_AHEAD` 仅 `DeepseekV41ForCausalLM`——SGLang 未注册 deepseek_v41、MSV 领先，附原因；且例外表
   自检"若 SGLang 已注册则须移除"防腐烂）。命名/拼写漂移或自造架构名会立即红。**16/17 命中，1 例外登记**。
   回归 **413/413**、verify 60/60、docs 一致。

## 四、风险与收益

- **收益**：结构保真度对齐上游；命名不再误导；"假设等价"被测试挡住；未来新模型接入有明确 checklist。
- **风险/成本**：P2 拆入口会改动 golden（结构树不变、仅文件/命名组织变，可用 hash 不变来保证零语义漂移）；
  P1 审计工作量集中在阅读上游。P0/P1/P4 低风险高收益，建议先行；P2/P3 属组织性重构，可增量推进。
- **不做的事**：不逐架构复制实现（制造反向漂移）；不抄 SGLang 内核融合（破坏静态逻辑结构）。

## 复现 / 证据
P0 修复见 `frontend/src/structure/config/normalize.js`（numExpertGroup/topkGroup）+
`frontend/src/structure/operators/ops/index.js`（`groupedTopkAttrs`）；回归 `node --test` 410/410、
`verify:models` 60/60、`docs:check` 一致；重生成 `normalize.golden.json` / `ops-spec-tree.golden.json`。
区分性验证：DeepSeek-V3.1 topk={group_limited_topk, ng=8, tg=4}、Kimi-K2.5/GLM-5/GLM-4.7 topk={}。

## 五、P1 字段覆盖审计结果（对 SGLang 路由/注意力模块逐字段核对）

对 SGLang `deepseek_v2/v3`、`qwen3_moe`、`qwen3_5`、`glm4_moe`、`minimax_m2/m3` 的 MoE gate/topk +
attention __init__ 提取其 READ 的结构/路由字段，核对 MSV 覆盖：

**路由字段（全部已覆盖）**：

| 上游字段 | MSV 归一化/消费 | 通道 |
|---|---|---|
| num_experts_per_tok | expertsPerToken | normalize |
| n_routed_experts / num_experts | experts | normalize |
| n_shared_experts | sharedExperts | normalize |
| moe_intermediate_size | moeIntermediateSize | normalize |
| shared_expert_intermediate_size | sharedExpertIntermediateSize | normalize |
| **n_group / topk_group** | **numExpertGroup / topkGroup（P0 新增）** | normalize + topk 算子 |
| norm_topk_prob | normTopkProb | normalize + topk 算子 |
| scoring_func | scoringFunc | normalize + router 算子 |
| routed_scaling_factor | routedScalingFactor | normalize + deepseekV4 router |
| first_k_dense_replace | schedule.js 直读 config（layerKinds） | schedule 原始读 |

**注意力字段（全部已覆盖）**：num_attention_heads/num_key_value_heads/head_dim/q_lora_rank/kv_lora_rank/
qk_nope_head_dim/qk_rope_head_dim/v_head_dim → normalize；use_qk_norm→useQkNorm、qk_norm_type→qkNormType、
partial_rotary_factor→partialRotaryFactor（normalize+ops）；attn_output_gate→recipe `recipeAttentionOutputGate`。

**两处曾记录的小缺口 —— 均已在 deepseek-gate-bias-attn-sink 落地**：
1. **`topk_method`（noaux_tc）归一化 ✅ 已做**：`normalize.js` 读 `topk_method`，`routerCorrectionBias =
   topk_method==noaux_tc || sigmoid 路由`，作为 grouped/biased 路由的权威判据（不再仅靠 n_group>1 代理）。
2. **`e_score_correction_bias` 字节叶 ✅ 已做**：`ops/index.js` `routerWeightMatrices` 给 noaux_tc/sigmoid
   router 声明 `[n_routed_experts]` fp32 修正偏置叶（`router_correction_bias`），并补了视觉路由 `router_bias_vl`；
   进参数量与 checkpoint 张量对账，不进量化字节/KV（fp32 单源 paramDtypes）。

**审计结论**：**n_group 修复后，上游建模的路由+注意力结构字段 MSV 已全覆盖**（经 normalize / recipe /
schedule 三通道），无其它重大缺口；上述两处小项亦已落地（deepseek-gate-bias-attn-sink）。这把"个案修 n_group"升级成了
"系统确认无同类漏建"。
