# 文档地图

本目录按职责拆分文档。顶层文档描述当前规范；`details/` 描述实现细节。

## 当前规范

1. [系统架构](architecture.md)：前后端边界、数据流、共享协议和部署形态。
2. [待实现计划](implementation_plan.md)：当前未完成事项、优先级、验收条件和明确不做的内容。
3. [UI/交互规范](ui_interaction.md)：入口、详情工作区、Cost Lens、PD 和交互行为。
4. [测试和发布验证](testing_release.md)：单测、模型验证、构建、API 和浏览器验证。

## 实现细节

5. [模型实现细节](details/models.md)：模型清单、来源、发布时间、缓存和适配入口。
6. [模块实现细节](details/modules.md)：模型 builder、层、算子、公式、IR 和诊断。

补充的实现细节：

- [图实现选型](details/graph_sources.md)
- [模型来源解析](details/models/source_resolution.md)
- [模型发布时间来源](details/models/release_metadata.md)

## 维护规则

- 当前代码行为与历史提交冲突时，以代码、测试和顶层当前规范为准，并在实现计划中记录差异。
- 新增模型、层、算子或公式时，更新对应的 `details/` 文档和测试。
- 方案发生变化时，更新当前规范；历史版本通过 Git 提交记录追溯。
