# 全量展开连线遮挡审计（2026-09-25）

## 目的

在本地服务 `http://127.0.0.1:4173` 上逐个打开 60 个内置模型，点击 **Expand all**，通过浏览器实际 SVG 几何采样检查数据流线是否穿过非端点模块 tile。本报告只记录问题，不在审计脚本中修改布局或 Graph IR。

## 审计结果

- 模型：60/60 个 URL 完成访问；脚本级异常：0。
- 发现潜在遮挡：48/60 个模型，共 451 条边命中非端点 tile。
- 无命中：12 个模型。
- `DeepSeek-V4.1-Flash`：0 条命中；这只说明本次全展开采样没有发现遮挡，不代表所有模型都没有问题。
- `Kimi-K3`：渲染 tile 数为 0，属于加载/渲染证据不足，不能标记为通过，需单独复跑。

> “潜在遮挡”定义为 SVG edge path 的采样点落入非端点 `.rf-model-node` 的屏幕矩形。端点本身不计入；因此这是几何问题发现器，不是最终视觉判定器。

## 问题分布

| 边渲染类型 | 命中边数 | 初步判断 |
|---|---:|---|
| `msvRoutedEdge` | 57 | 最终坐标正交路线仍未把全部可见 tile 当作障碍物 |
| `msvNativeEdge` | 210 | frame 连接仍回退自由贝塞尔，容易穿过展开子节点 |
| `msvEdge` | 184 | smart-edge 只在局部可见节点集合下寻路，存在跨容器/折叠投影遗漏 |

## 重点问题族

1. **Qwen3.5 / GLM-5.3-Flash 的主干跨容器边**：反复出现同一条 root 边穿过已展开 decoder 子节点，说明公共 root route 的障碍物模型不完整。
2. **Qwen3.8-Flash-Next**：同时出现 `msvNativeEdge` 和 `msvEdge` 命中，说明 QSA/GDN 混合展开后的跨容器边与内部 smart-edge 需要统一可见图障碍物语义，不能只改一种 edge renderer。
3. **DeepSeek V4-Flash-0731 / Vision**：大量 `msvNativeEdge` 命中，属于当前最明显的 frame-to-frame 自由曲线路由问题。
4. **GLM-5.2/5.3 IndexShare**：复用边和展开层之间有重复命中，说明跨层/共享来源边需要以可见端点投影后再做障碍物路由。
5. **Kimi-K3**：本轮 tile 数为 0，不能把它计为通过；需要修复审计等待条件或单独记录加载失败。

## 全局修复方向（暂不实施）

### 1. 不直接提高 edge z-index

当前 tile `zIndex=1`、edge `zIndex=0` 的原则本身是合理的：连线不应覆盖节点标题、公式和按钮。直接提高边层级只能把线压到文字上，属于掩盖问题。应修正路线，而不是层级。

### 2. 统一“可见图”与“障碍物图”

对每条 IR edge 先做 visible-ancestor projection，得到当前展开状态的实际 source/target；同时从同一份可见节点集合生成障碍物矩形。路由器必须排除非端点叶节点、非端点展开容器中的可见子节点，以及跨容器边经过的共同祖先内部节点。

### 3. 取消按 renderer 分裂的兜底

`msvNativeEdge`、`msvEdge`、`msvRoutedEdge` 应共享同一套 route contract：输入是最终可见节点盒子和最终端点，输出是可验证的折点。只有明确证明无障碍时才允许 native/smoothstep 回退；否则应回到正交/A* 路由，而不是自由贝塞尔。

### 4. root 级重定位后重新寻路

现有 root route 已解决 DSpark 的旧坐标问题，但 dogleg 只考虑 source/target 两个盒子，没有考虑途中展开的 decoder、attention、merger 等模块。全局修复应在所有手动 reposition 完成后重新运行 obstacle-aware routing。

### 5. 建立浏览器几何门禁

- 60/60 模型必须成功渲染；
- 每个模型点击 Expand all；
- Kimi-K3 等 tile=0 或页面错误必须单独失败；
- 非端点遮挡数必须为 0，或显式登记为已知基线；
- 同时检查边是否越出容器、节点重叠、容器裁切和控制台错误。

## 当前结论

这不是 DSpark 独有问题。全量浏览器审计已经发现公共路由层仍有系统性缺陷，尤其集中在跨容器 frame 边、手动重定位后的 root 边、IndexShare/Flash-Next 的跨层边。下一步应先完成“可见图 + 障碍物图 + 统一路由契约”的设计和失败用例，再分批修复；不应继续对单一模型增加特例。

原始机器结果：`docs/details/evidence/structure/generated/expanded-edge-occlusion-audit.json`。
可重复脚本：`scripts/evidence/structure/audit-expanded-edge-occlusion.mjs`。
