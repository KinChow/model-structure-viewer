# UI/交互规范

本文描述当前页面行为和交互约束，不描述模块实现、公式推导或未来开发计划。实现入口见 `frontend/src/components/`，结构和公式见 [`details/modules.md`](details/modules.md)。

## 页面流程

```text
模型入口页
  -> 选择内置/远程模型，或打开本地目录
  -> 读取配置和可用的 checkpoint 元数据
  -> 生成结构
  -> 模型详情工作区
```

加载期间显示阶段状态；失败时保留当前页面并显示可读错误，不将 config fallback 包装成 transformers 验证成功。

## 模型入口页

入口页由 `ModelEntry.jsx` 实现，只提供两种工作模式。

### 输入或选择模型

- 输入 Hugging Face/ModelScope 模型 ID 或 URL。
- 来源选择器决定远程端点；仓库 catalog 中存在同 ID 时直接使用 `builtin`。
- 最近发布的 5 个内置模型显示为快捷入口，快捷入口不作为测试或业务逻辑依赖。
- Provider 卡片只负责浏览内置模型，列表支持按发布时间或名称排序。

### 打开本地模型目录

- 浏览器目录选择读取 `config.json`，并在文件可用时读取 safetensors header。
- 本地路径输入调用后端，不在纯静态部署中承诺可用。
- 没有找到 `config.json` 或配置无法解析时显示明确错误。

### 全局行为

- 支持中文和英文，默认跟随浏览器语言，用户选择保存到 `localStorage`。
- 支持深色和浅色主题，用户选择保存到 `localStorage`。
- Help 只解释入口所需信息，不展示内部 source/cache 等实现参数。
- 页面显示前端包版本，但不维护独立的手写 UI 版本号。

## 模型详情工作区

详情页由 `DetailWorkspace.jsx` 组织，保持单页工作区，不拆成多级路由。

```text
顶部：返回、模型 ID、语言、主题、设置
摘要：架构、参数、层数、hidden size、专家、上下文、来源、状态
主体左侧：搜索、结构图、成本与部署、导出/原始配置
主体右侧：模型摘要或节点 Inspector
```

### Architecture

- React Flow 是当前结构画布；ELK 负责 compound graph 布局，Smart Edge 负责普通节点间避障连线。
- 默认展示结构主干；可展开/折叠组、平移、缩放、fit、查看 minimap。
- 点击节点更新 Inspector；搜索、公式索引、Layers 和图节点共享同一节点路径。
- 公式 hover/选中只改变关联状态，不重新计算结构。
- 图导出生成包含当前 React Flow viewport 的 SVG。

### Inspector

- 未选中节点时展示模型摘要和顶层模块。
- 选中节点时展示 breadcrumb、类型、属性、shape、参数、权重来源、公式和当前 Cost Lens 数据。
- 从 breadcrumb、顶层模块、搜索结果或图节点进入同一路径时，Inspector 结果必须一致。
- 小屏幕允许收起 Inspector，不能遮挡结构图的主要交互。

### 搜索

- 搜索模型结构节点并展示命中数和候选项。
- 选择结果后展开其祖先组、选中节点并清空搜索输入。
- 搜索不能改变原始结构、成本参数或当前模型来源。

### 成本与部署

成本区域默认折叠，摘要始终展示当前部署模式、芯片、节点/GPU 数和 fit 状态。

- Cost Lens 支持 `None`、`VRAM`、`Compute`、`Memory` 和 `KV Cache` 多选。
- 模式支持集中式和 PD；PD 分别保存 Prefill/Decode 的负载、节点数和并行方案。
- 并行输入包括 TP、PP、EP、DP 和 Attention 模式。
- 对比模式互斥：关闭、芯片对比、方案对比。
- 芯片规格缺失时显示 unknown 或警告，不填入估算值伪装真值。
- 所有成本结果必须标注为理论计算，不表示调度、流水线气泡、传输重叠或实际吞吐预测。

### 导出和原始配置

- Export 支持 Mermaid、DOT 和 JSON。
- Raw config 展示生成当前结构所用的原始配置。
- 两者是详情工作区中的辅助面板，不切断结构图与 Inspector 的当前状态。

## 状态与错误

- 区分 frontend template、checkpoint truth、meta introspection 和 repaired meta introspection。
- 权重真值不可用时显示对应来源和降级原因。
- 后端不可用时，静态来源仍可继续工作；需要后端的操作应明确提示启动 API。
- loading、error、empty 和 unknown 状态不能使用相同文案或颜色表达。

## 可访问性和响应式约束

- 交互控件使用 button、input、select 和正确的 `aria-*` 状态。
- 键盘可完成模型提交、搜索选择、节点选择和面板开关。
- 文字、按钮和输入框在窄屏不能横向溢出或相互覆盖。
- 自动化验证优先使用 role、aria label 和稳定 data 属性，不依赖显示顺序或具体快捷模型。

## 明确不做

- 最近模型、收藏、账号和云端同步。
- 自动推荐最优并行方案或预测吞吐、TTFT、TPOT。
- modelmap 式运行时脉冲回放。
- 在首页暴露内部 cache policy、repair strategy 或后端实现参数。
