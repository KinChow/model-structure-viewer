# ELK-only 布局重构方案（2026-09-25）

## 决策

当前 `ELK + Libavoid` 实现只作为已验证的临时稳定基线，不作为最终架构。

最终目标是：

```text
Graph IR v2
    ↓
Layout IR（只描述当前展开状态的布局投影）
    ↓
一次 ELK compound layout
    ↓
一次 ELK edge routing
    ↓
React Flow 只消费最终坐标和 sections
```

禁止在 ELK 完成后继续修改节点 `x/y`，禁止为 DSpark、MTP、多模态或
IndexShare 增加路线特判，禁止使用第二个运行时路由器修补 ELK 结果。

## 之前纯 ELK 尝试暴露的真实问题

直接删除 post-layout 重排后，ELK 的跨层边可以生成 sections，但以下展示语义
没有被布局输入表达：

1. 顶层流水线需要横向排列；
2. 模块内部需要纵向排列；
3. `Decoder → final norm → lm head` 必须是同一主干；
4. DSpark/MTP 必须位于主干的辅助 lane，而不是与主干节点竞争同一层；
5. 多模态视觉、文本和融合入口需要保留独立 lane；
6. 真实边必须穿过 compound boundary port，而不是只在共同祖先添加约束边。

因此“删除手动重排”不是完整 ELK-only 重构，只能作为失败基线。问题不在于
继续增加 dogleg，而在于当前布局模型没有表达 lane、主干和边界端口。

## 新的布局建模

### 1. Graph IR 与 Layout IR 分离

Graph IR 继续保存事实：

- 真实节点；
- 真实父子关系；
- 真实数据流边；
- evidence、relation、canonical_id。

Layout IR 只在 `layoutGraph` 之后生成，包含：

- 当前可见节点；
- 当前可见边；
- 虚拟 layout compound；
- boundary ports；
- lane 与对齐约束；
- 真实节点到 layout 节点的映射。

虚拟 layout 节点不能进入 Graph IR、成本统计、checkpoint 绑定或导出真值。

### 2. 使用虚拟 compound 表达视觉 lane

对有明确拓扑语义的模型，布局投影使用以下概念，不修改模型事实树：

```text
model-layout（DOWN）
├── input-lane（RIGHT，可选）
├── main-lane（RIGHT）
└── auxiliary-lane（RIGHT，可选）
```

普通纯文本模型可以只有 `main-lane`。

多模态模型：

```text
input-lane：
  vision / text / embedding / merge

main-lane：
  encoder / decoder / final norm / lm head

auxiliary-lane：
  DSpark / MTP / speculative branch
```

虚拟 compound 负责把“输入分支、主干、辅助分支”的相对位置交给 ELK；
不能再由布局完成后直接写 `child.y = ...`。

### 3. 每条真实可见边经过 boundary ports

跨 compound 边拆成布局段：

```text
source leaf
  → source compound out port
  → common ancestor route
  → target compound in port
  → target leaf
```

段 ID 只存在于 Layout IR。最终绘制边由对应 segments 拼接，但源端点和目标端点
仍来自 Graph IR。

### 4. 方向由 compound 层级决定

不再依赖同一 ELK graph 同时把 root 和所有子模块强行设置为不同方向后再修坐标。

- `model-layout` 用 `DOWN`，把输入、主干、辅助 lane 排成稳定行；
- 每个 lane 用 `RIGHT`，形成执行流水线；
- 模块内部用 `DOWN`；
- 需要转向的结构通过虚拟 compound 和 ports 表达。

这样 ELK 的一次布局可以同时保留 lane 语义、内部方向和跨层路由。

## 实施阶段

### Phase 0：Layout IR 骨架，不接生产

- 新增 layout-only 类型和映射；
- 从 Graph IR 生成普通模型、主干、辅助分支和多模态 lane；
- 对 4 个代表模型输出 Layout IR 快照；
- 断言虚拟节点不出现在 Graph IR、参数、成本和 checkpoint 路径。

代表模型：

- DeepSeek-V4.1-Flash；
- DeepSeek-V4-Flash-0731；
- Qwen3.8-Flash-Next；
- GLM-5.2。

### Phase 1：ELK-only 隔离布局

- 只把 Layout IR 交给 ELK；
- 一次布局、一次路由；
- 不接 Libavoid；
- 不改变现有生产 `layoutGraphWithElk`；
- 记录每个虚拟 lane 的最终位置、端口和 sections。

门禁：

- 所有真实边都有可拼接 sections；
- 所有可见节点位置有限且不重叠；
- `Decoder → final norm → lm head` 对齐；
- DSpark/MTP 不进入主干 lane；
- 多模态输入 lane 不被压平；
- 无非端点 tile 遮挡。

### Phase 2：60 模型与局部展开矩阵

不能只测 `Expand all`。每个内置模型需要：

- 折叠基线；
- 顶层单模块展开；
- 每个顶层可展开模块单独展开；
- 代表模型的深层展开；
- 全展开。

全部状态执行浏览器 SVG 几何审计。

### Phase 3：生产切换

只有 Phase 1/2 通过后才：

1. 让生产布局使用 Layout IR；
2. 删除 Libavoid 依赖；
3. 删除 root-level dogleg 和 post-layout `x/y` 修改；
4. 将 `routePoints`、`routeStatus` 收敛为 ELK sections；
5. 更新 golden 和文档；
6. 重新执行全量测试、构建和 Chrome 验收。

## 完成标准

- 只使用 ELK 产生最终节点和边路线；
- Graph IR v2 不升级；
- 无生产级手动坐标修正；
- 无模型特判路由；
- 60/60 模型及局部展开矩阵通过；
- 无非端点边遮挡；
- 无关键节点重叠、容器裁切或空图；
- 参数、成本、checkpoint truth 与重构前保持恒等；
- 布局失败显式报错，不静默回退贝塞尔。

## 当前工作区状态

本轮纯 ELK 删除 Libavoid 的实验已回滚，没有把不完整版本留在工作区。
生产基线仍是 `061a03a`，后续从 Phase 0 的 Layout IR 隔离实现开始。
