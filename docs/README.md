# 文档地图

本目录按职责拆分文档。顶层文档描述当前规范；`details/` 描述实现细节。

## 当前规范

0. [开发原则](principles.md)：**强制约束，优先级高于本目录其他文档**。定位判据、结构与成本原则、真值优先、源码可追溯、前后端边界、合入检查清单。
1. [系统架构](architecture.md)：前后端边界、数据流、共享协议和部署形态。
2. [待实现计划](implementation_plan.md)：当前规范含 **后续终态（2026-09-14）**（折叠图、四本账、SGLang group 元数据、resolver 契约、明确不做）。正确性账本已闭合；剩余是触发池，不是产品 backlog。
3. [重构计划](refactor_plan.md)：原则收口路线。自底向上的分波替换顺序、差分替换手法、每波验收命令。
4. [UI/交互规范](ui_interaction.md)：入口、详情工作区、Cost Lens、PD 和交互行为。
5. [测试和发布验证](testing_release.md)：单测、模型验证、构建、API 和浏览器验证。
6. [竞品分析与 MSV 产品策略](competitive_analysis.md)：LLM Inference Analyzer、LLM Architecture Gallery、modelmap 的定位、能力对比和 MSV 路线建议。

## 实现细节

5. [模型实现细节](details/models.md)：模型清单、来源、发布时间、缓存和适配入口。
6. [模块实现细节](details/modules.md)：模型 builder、层、算子、公式、IR 和诊断。
7. [算子动作向量注册表](details/cost_counts.md)：48 个公式条目的分类、counts 公式、共享实现与假设（W1 实现规格）。

补充的实现细节：

- [图实现选型](details/graph_sources.md)
- [模型来源解析](details/models/source_resolution.md)
- [模型发布时间来源](details/models/release_metadata.md)
- [成熟框架并行策略调研](details/parallel_strategies.md)：vLLM/SGLang 的 attention、MoE、TP/DP/EP/ETP 和 shared expert 现状
- [并行与权重分片协议](details/parallel_protocol.md)：逻辑轴定义、约束等式、九项裁决和三类成本口径（协议唯一住址）
- [模型台账参考](models_reference.md)：59 内置模型机器台账（family / canonical architecture / 参数量级 / 证据库 / release_time），`gen-model-reference.mjs` 生成 + `docs:check` 守护
- [架构台账参考](architectures_reference.md)：别名表 / canonical 目录 / ARCH_RECIPES 配方机器台账，同一生成器 + `docs:check` 守护
- [算子对照参考](details/operators_reference.md)：逐算子触发面 / 恒等式机器台账（`gen-operators-reference.mjs` 生成段），同族台账此前漏登，此处补齐

## 维护规则

- **`principles.md` 是强制约束**，优先级高于其他文档。与代码冲突时视为代码缺陷；确需偏离在 `implementation_plan.md` 登记（违反哪条、为什么、何时收口）。
- 当前代码行为与历史提交冲突时，以代码、测试和顶层当前规范为准，并在实现计划中记录差异。
- 新增模型、层、算子或公式时，更新对应的 `details/` 文档和测试。
- 方案发生变化时，更新当前规范；历史版本通过 Git 提交记录追溯。
