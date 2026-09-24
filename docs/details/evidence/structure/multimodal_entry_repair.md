# 多模态入口：独立分支、融合语义与未知流量

## 范围

2026-09-24，本地 `main`，基线 `55df6d4`。修复计划 C2 的公共入口、
MiniMax-M3 独立入口、V4.1 CED 独立入口。Graph IR 保持 v2，无端口协议升级。
覆盖60条目中39个多模态条目、9种 model_type；21个纯文本条目不改拓扑。

## 证据，不由形状反推机制

本次在线重读 [架构图库](https://sebastianraschka.com/llm-architecture-gallery/)、
39个精确型号的官方模型卡，并获取前向实现。完整读取时间、revision、
URL、SHA256及卡片行号在：

- `multimodal-entry-variants.json`：39条独立卡片记录。卡片证明视觉能力，
  **不替代逐变体 checkpoint 实装审计**，后者全部明确 `not_completed`。
- `multimodal-entry-sources.json`：官方仓库固定revision的前向、Transformers
  固定版本源码与内容摘要。MiniMax库源码为main读取快照，不能冒称固定tag。
- 本地原文快照：`artifacts/architecture-repair/C2/`。

| 家族 | 外部主证据和机制锚点 | 融合语义 |
|---|---|---|
| Qwen3.5 / MoE；Qwen3.6、3.8兼容变体 | 各型号卡片；Transformers v5.16.1 `Qwen3_5Model.forward` 1527–1549 / MoE 1643–1654 | embedding 后按 image/video placeholder mask `masked_scatter` |
| Qwen3.8-Flash-Next 两变体 | 两个模型卡；前批已读取的官方报告；入库 `modeling_qwen4_exp.py` 2343–2357 | placeholder scatter，不是将图像送入 embedding lookup |
| GLM-5.3-Flash 两变体 | 各自官方卡；Transformers v5.16.1 `modeling_glm5_next.py` 1963–1971 | placeholder scatter |
| MiniMax-M3 两变体 | 各自官方卡；Transformers `MiniMaxM3VLModel.forward` 1405–1425 | embedding 与视觉特征分别产生，再按 mask scatter |
| Kimi K2.5 / K2.6 / K2.7-Code | 分别固定revision下载的 `modeling_kimi_k25.py`，`_merge_input_ids_with_image_features` | 按实际每图特征长度展开占位符，重排文本、padding、position和mask |
| Kimi K3 | 官方卡、固定revision `modeling_kimi_k3.py` 958–1090 | 同为占位符展开，但不据此推断其语言主干与K2.5相同 |
| DeepSeek V4 Vision | 官方卡及固定revision `inference/model.py` 975–996 | 视觉行按 `img.perm` 重排，构造带delimiter/pad/newline的图像块，覆写span |
| DeepSeek V4.1 | 官方卡 Multimodal architecture；固定revision `inference/model.py` 1228–1256 | IMAGE位置写入特征，START/END/NEWLINE位置写入真实delimiter参数 |

补充：图库概览并不完整描绘所有视觉和草稿模块；因此未将图库图中缺失
的分支当作应删除的证据。Kimi K2.5与K2.6下载源码摘要相同，
K2.7-Code不同，分别核对其融合函数，未假设所有版本只有训练差别。

## 修改

```text
image/video input → vision tower → optional projector ─┐
token IDs → token embedding ──────────────────────────┼→ fusion → language stack
token IDs ─────── placeholder positions（控制边）──────┘
```

- `multimodalEntry` 只负责入口组件和显式边；已有语言栈、MTP/DSpark、
  CED KV投影边保留。没有将视觉输出连向 embedding 的边。
- Qwen/GLM已有内部merger保留，Kimi/MiniMax/DeepSeek已有外部projector保留，
  不在本批重命名真实参数路径或改写塔内部结构。
- 家族语义放入现有架构配方 `visionFusion`，未识别配方返回unknown，
  不从显示名、最后维度或模型ID子串猜测。
- 新输入/融合节点是语义节点，明确 `checkpoint_module=false`，
  不允许偶然同名checkpoint张量冒充这些节点；未匹配的真实张量保留为gap。
- `dataflow_edges` 仍为二元组，控制边及特征边提示通过既有独立关系字段表达。
- 使用现有ELK分支布局，无替换布局器；纯串行主干压平规则已经在前批
  限定，真实分支保留独立纵向位置。

## 序列与费用

**工作负载sequence定义为进入语言模型的融合后长度。** Kimi运行时会从
较短占位符序列扩展，而本工具不掌握图像数量/各图长度，不能把视觉token再加一次。
数字shape保留动态维，语义属性声明 `workload_is_post_fusion`。

融合新增参数0、MAC0；没有伪造 GEMM。前向纯替换/写入不增加乘加，
但实际搬运依赖mask、图片占位量和是否原地物化，因此：

- prefill `actIn/actOut = null`，不是零；
- 普通逐token decode沿用已prefill图像状态，融合不再执行，流量为0；
- 带新图片的后续chunk应作为prefill处理，不冒称普通decode涵盖该场景。
- 总量、按功能域汇总、Lens保留unknown，模型MAC仍可单独已知；
  缺少访存导致roofline bound/time未知，而不是强行通过“所有费用有限”测试。
- 功能域阶段时间不能将未知的访存分量过滤后，取其他已知零的max并显示
  `0 ns`；`stageTimingSummary` 按已有unknown结论保留未知，附独立回归。
- 输入节点是外部来源引用，不计额外拷贝。图像输入/融合的物化容量为unknown。

## 独立对账

`scripts/evidence/structure/multimodal-entry-reconcile.mjs` 用基线git archive
只读导入，比较60条目×config/artifacts共120条路径，B=1、S=16，
prefill/decode均验证：

- 声明参数容量不变；
- MAC、权重驻留、KV驻留、递归state驻留不变；
- 逐模块费用变化仅 `image_input / text_input / multimodal_fusion`；
- 39个条目各新增3节点、4条净数据流边（控制边在内）；
- 21个纯文本条目不变。

完整对账在 `artifacts/architecture-repair/C2/reconcile.json`，日志
`reconcile.log`。只审阅更新39条目的tree/edge golden，禁止用全模型总量
的小误差替代逐节点对账。

## 测试和边界

- `multimodalEntry.test.js`：39条目两路径的独立机制断言、禁止视觉→embedding、
  双分支+控制输入、融合→语言栈、CED关系、零权重、decode不重复融合、
  unknown透传、语义节点与checkpoint同名冲突。
- `modelIdentities.test.js` 删除旧 `visual/projector → embed_tokens` 形状例外；
  只登记token IDs→fusion为索引控制边。
- 老实现新增机制用例2/2失败；独立定向机制/契约测试通过后才更新golden。
- 浏览器脚本 `e2e/multimodal-entry.spec.js` 走生产加载，报告写入artifacts，
  不覆盖受版本管理的旧全量巡检报告。

本批不证明以下事项：全部视觉塔内部参数已正确（例如DeepSeek delimiter
和aligner细节仍待专项对账）、所有变体量化checkpoint一致、K3 AttnRes、
Qwen GR、MiniMax MSA或DeepSeek compressor均已完成。总目标仍未完成。

## 验收记录

本轮日志均在 `artifacts/architecture-repair/C2/`，使用本次启动的服务；
不终止用户已有服务，不下载模型完整权重，不声称任何实测性能。

| 门禁 | 结果 | 证据 |
|---|---|---|
| 前端完整测试 | 514/514 | `unit-validated.log` |
| 后端完整pytest | 184/184 | `pytest-serial.log` |
| 内置基础检查 | 60/60 | `models-validated.log` |
| 两路径逐模块对账 | 120/120 | `reconcile.json` / `reconcile.log` |
| docs / build / 原则护栏 | 通过，build保留既有chunk体积警告 | `docs-validated.log` / `build-validated.log` / `principles.log` |
| 桌面入口巡检 | 39/39，9家族深展开及SVG导出 | `browser-desktop.log`及其report JSON |
| 桌面最终UI专项 | 1/1，unknown时间与融合搜索 | `browser-final-ui.log` |
| 移动端 | 9/9家族，包含搜索、unknown功能域时间、深展开和SVG导出 | `browser-mobile.log` |

首次并发pytest为181通过/3失败；错误信息包含 `worker timed out after 90s`。
前端/浏览器重任务结束后完整重跑184通过，未延长产品超时或跳过测试。
原失败日志保留 `pytest.log`，不将第一次写成通过。
人工查看MiniMax和V4.1桌面图、移动端英文图；V4.1保留两段CED与草稿旁挂。
这只是C2范围，不是最终要求的60条目全量浏览器架构正确性验收。
桌面39条目巡检在最后一处阶段时间显示修正之前完成；该修正未改图，
随后由桌面专项和移动端9家族重新验证，未将前一轮截图冒称最终UI截图。
