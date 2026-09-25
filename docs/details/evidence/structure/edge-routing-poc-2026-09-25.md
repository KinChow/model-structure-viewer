# 连线路由双方案 POC（2026-09-25）

> 历史 POC，最终实现与完整复验见
> `edge-routing-implementation-2026-09-25.md`。这里对“ELK 一次布局”
> 的实验只是移除后处理的基线，不是跨容器边界端口方案的验证；
> 不应将它当作 ELK 不适用的结论。

## 结论摘要

本次 POC 比较两个业界方案：

1. **方案 A：ELK Layered 一次完成节点布局和正交连线**；
2. **方案 B：保留节点位置，使用 libavoid 进行固定节点障碍物路由**。

初步建议：

- 不采用“提高 edge z-index”或继续增加单模型 dogleg 特判；
- 不把方案 A 的“删除现有后处理”直接作为生产修复；
- 将 **libavoid 作为最终路线候选**，但必须先解决大图全量路由的性能和浏览器 WASM 集成问题；
- 生产实现前需要做“可见图裁剪 + 端点投影 + 仅对需要避障的边路由”的第二轮 POC。

## 为什么选择这两个方案

React Flow 官方将 ELK 列为同时支持节点布局、子流布局和边路由的成熟方案，并将 libavoid 作为自动避开节点交叉的边路由方案。ELK 官方也明确区分了 Layered 布局和 Libavoid 固定节点路由：Layered 负责布局约束，Libavoid 适合节点位置固定后只重新计算边。

因此本 POC 不比较自研 A*，也不继续扩展当前 `msvNativeEdge` / `msvEdge` 的特例逻辑。

## 测试对象

四个问题代表：

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.8-Flash-Next`
- `deepseek-ai/DeepSeek-V4-Flash-0731`
- `zai-org/GLM-5.2`

每个模型均使用全部展开后的 Graph IR 视图和当前 ELK 坐标，检查：

- 边是否生成；
- 是否有 bend points；
- 路由是否经过非端点障碍物；
- 复杂图的路由耗时是否可接受。

## 方案 A：ELK 一次布局与路由

### POC 做法

- 在隔离副本中移除当前 root-level 主干/旁挂后处理；
- 保留 ELK Layered、compound graph、ports、orthogonal routing；
- 直接使用 ELK 最终节点位置和 sections/bendPoints；
- 不改变生产代码和 Graph IR。

### 结果

| 模型 | dataflow 边 | 带 route 的边 | 障碍物命中采样 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 320 | 320 | 921 |
| Qwen3.8-Flash-Next | 1605 | 1441 | 4298 |
| DeepSeek-V4-Flash-0731 | 1766 | 1620 | 3714 |
| GLM-5.2 | 1240 | 1221 | 1036 |

### 判断

这个结果不是说 ELK Layered 不支持路由，而是说明**当前 Graph IR 的 compound projection、端点 frame、内部可见节点和 ELK route 坐标之间还没有形成同一套障碍物语义**。

如果直接删除现有后处理切换到“ELK 全包”，会重新引入或放大：

- `final norm` 主干位置变化；
- DSpark/MTP 旁挂位置不稳定；
- 多模态输入分支压平；
- 折叠/展开后的 frame 端点不一致；
- 大图中内部节点被 root edge 视为可穿越区域。

因此方案 A 暂不直接进入生产。

## 方案 B：固定节点 + libavoid

### POC 做法

- 使用当前最终节点和 frame 坐标；
- 将当前可见节点转换为障碍物；
- 将 edge 的可见 source/target 投影为端点；
- 使用 `@mr_mint/elkjs-libavoid` 0.5.0 的 orthogonal routing；
- 设置 shape buffer、segment penalty 和 crossing penalty；
- 检查输出 bendPoints 的障碍物穿越情况。

该包提供浏览器 WASM 路由器，支持固定节点、正交/折线、ports 和 hierarchical graph；本 POC 使用临时目录安装，不修改仓库依赖。

### 小图全量路由结果

`Qwen/Qwen3.5-0.8B`：

| 项目 | 结果 |
|---|---:|
| 路由边数 | 320 |
| 生成 bend 的边数 | 156 |
| 障碍物命中采样 | 0 |
| 单模型 POC 路由耗时 | 约 3.9 秒 |

### 复杂图全量路由结果

对以下模型尝试路由全部可见边时，35 秒内未完成：

- `Qwen/Qwen3.8-Flash-Next`
- `DeepSeek-V4-Flash-0731`
- `GLM-5.2`

这说明直接把 1,000+ 节点、1,000+ 边全部送入 libavoid 不适合作为当前生产路径。

### 复杂图问题边子集结果

只将浏览器审计已经发现的潜在遮挡边交给 libavoid：

| 模型 | 送入路由的边 | 生成 bend 的边 | 障碍物命中采样 |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next | 55 | 55 | 0 |
| DeepSeek-V4-Flash-0731 | 104 | 104 | 0 |
| GLM-5.2 | 21 | 21 | 0 |

这证明 libavoid 在当前坐标和障碍物表达下能够解决已发现的几何穿越，但还不能据此宣称可以直接全量替换现有路由。

## 方案比较

| 维度 | 方案 A：ELK 一次布局路由 | 方案 B：固定节点 + libavoid |
|---|---|---|
| 业界成熟度 | 高 | 高 |
| 是否新增依赖 | 否 | 是，WASM |
| 能否保留当前节点布局 | 否，需重构后处理 | 是 |
| 处理最终坐标后的避障 | 间接 | 直接 |
| 对 DSpark/MTP 位置影响 | 高风险 | 低风险 |
| 对多模态分支影响 | 高风险 | 低风险 |
| 当前 POC 遮挡结果 | 未通过 | 问题边子集为 0 |
| 当前复杂图性能 | 可完成，但有大量穿越 | 全量路由超过 35 秒 |
| 当前推荐状态 | 暂不直接采用 | 候选，但需要性能优化 |

## 重要限制

本 POC 不是最终生产实现，原因包括：

1. 方案 A 的隔离实现是“移除现有手动后处理”的基线，不是完整的 ELK constraint redesign；
2. 方案 B 使用临时安装的 npm 包，没有接入 Vite/WASM 静态资源；
3. 当前 POC 的 obstacle graph 使用了扁平化的可见节点表示，尚未完成完整 compound hierarchy 和 frame boundary 处理；
4. 复杂模型全量 libavoid 路由的性能不满足直接上线；
5. `Kimi-K3` 在上一轮浏览器审计中返回 0 个有效 tile，仍需单独修复审计等待/渲染问题；
6. 方案 B 对“问题边子集”的 0 命中不能替代 60 模型全量浏览器验收。

## 最终建议

下一轮采用 **方案 B 的优化版**，而不是直接把 libavoid 接到全部边：

### B1. 可见图预处理

- 保持 Graph IR v2 不变；
- 先完成 visible-ancestor projection；
- 只将当前真正可见的节点和 frame送入路由器；
- 对叶节点重复、折叠祖先和不参与绘制的内部节点做裁剪。

### B2. 两级路由

1. **普通局部边**：保留 ELK 已有 route；
2. **检测到可能穿越障碍物的跨容器/长边**：交给 libavoid；
3. 如果路由器返回无障碍短路径，覆盖该边的 bendPoints；
4. 不再让 `msvNativeEdge` 静默退回长贝塞尔。

### B3. 性能门槛

第二轮 POC 必须达到：

- 60 个模型全量展开可完成；
- 代表复杂模型单模型路由不超过 2 秒；
- 无非端点障碍物命中；
- 无新增节点重叠、frame 越界和容器裁切；
- 浏览器无控制台错误；
- WASM 构建产物可重复加载。

### B4. 失败策略

路由失败不能静默生成自由贝塞尔。应记录：

```text
routeStatus = routed | retained-elk | fallback | failed
routeReason
routeEngine
```

这些属于布局层诊断，不改变 Graph IR。

## 当前决策

本轮 POC 后的决策不是“直接选择 libavoid 上线”，而是：

> **生产方向选固定节点 + 成熟障碍物路由；具体实现优先继续验证 libavoid，并通过可见图裁剪、问题边筛选和缓存解决性能问题。**

方案 A 只有在完成完整 ELK constraint redesign、并能同时保持主干位置和无障碍路由后，才重新考虑。
