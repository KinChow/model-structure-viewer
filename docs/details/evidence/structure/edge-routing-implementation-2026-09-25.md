# 展开连线遮挡修复与浏览器复验（2026-09-25）

## 选择的实现

不是把 60 个模型统一压成单向的 ELK 全层级图，也不是自研通用寻路器。

1. 继续由 **ELK Layered** 布局各个容器：模型顶层 `RIGHT`，模块内部 `DOWN`。
2. 对每条当前可见的 Graph IR 边，在共同祖先及沿途容器的边界建立**布局专用端口**；各段由 ELK 在所属容器内正交路由，再沿边界端口拼接。原始 Graph IR 端点不变。
3. 当前顶层主干和 DSpark/MTP 旁挂位置在 ELK 后仍需调整，以保持 `Decoder → final norm → lm head` 基线。仅这些**位置已固定的顶层桥接段**使用 libavoid 避障，替换过去的手写 dogleg 和自由贝塞尔回退。
4. React Flow 按最终路线坐标绘制；路由失败会显示错误提示，不再无声地把失败当作布局完成。

该方案让成熟 ELK 处理大量内部和跨容器段，libavoid 仅处理少量 ELK 后重定位的顶层边。`graph.version = 2` 和 `schema_version = 2` 不变；桥接端口、分段 ID、路线点都是视图层数据。

## 为什么不是单次 INCLUDE_CHILDREN

使用当前 `elkjs` 0.12.0 的隔离原型验证：

- `hierarchyHandling=INCLUDE_CHILDREN` 可以路由跨层级边，但顶层 `RIGHT` 会把原本 `DOWN` 的模块内部也改成横向；
- 让内部保持 `SEPARATE_CHILDREN` 时，ELK 对跨越两个独立布局层级的原始边报 `UnsupportedGraphException`；
- 给展开容器添加边界端口，并将真实边拆成“内部段—顶层段—内部段”，则内部纵向布局和跨容器正交路由能同时保留。

因此不为追求“只用一个算法”牺牲可读性、主干位置或展开状态。

## 验收证据

基线审计（同一浏览器 SVG 采样方法）：60 个内置模型中 48 个模型有潜在遮挡，共 451 条命中非端点 tile 的边；之前 Kimi-K3 的空图是等待布局完成不足，不能算通过。

本次在构建后的本地 preview 上逐个打开 60 个 URL，等待 `data-layout-ready=true`，再点击 **Expand all** 并等待新布局完成，得到：

| 指标 | 结果 |
|---|---:|
| 模型 | 60/60 |
| 页面/审计异常 | 0 |
| 页面 JavaScript 错误 | 0 |
| 空图 | 0 |
| 连线命中非端点 tile | 0 |

实际浏览器重点检查了 DeepSeek-V4.1-Flash 的 DSpark 展开；构建产物的 libavoid WASM 能成功加载。Qwen3.8-Flash-Next、GLM-5.2、DeepSeek-V4-Flash-0731 和 Kimi-K3 的全展开图也有实际节点和 SVG 边，不再以空图冒充通过。

前端全量单测 614/614；架构修复桌面 Chrome E2E 3/3；`verify:models`、`docs:check` 和前端生产构建通过。

原始机器报告：`generated/expanded-edge-occlusion-after-routing-2026-09-25.json`。复验脚本：`scripts/evidence/structure/audit-expanded-edge-occlusion.mjs`，默认写入被忽略的 `artifacts/architecture-repair/`，不会覆盖受版本管理的证据。

## 覆盖边界

“Expand all”证明全部可展开模块在**同一展开状态下**参与实际浏览器布局和连线检测；它不等同于逐一切换每个模块、遍历所有局部展开组合。当前几何门禁检测“边穿过非端点 tile”，尚未把边与边交叉数量、标签遮挡或所有缩放比例纳入零缺陷结论。后续如发现特定局部展开组合问题，应加入对应的浏览器回归用例，不修改 Graph IR 真值图。
