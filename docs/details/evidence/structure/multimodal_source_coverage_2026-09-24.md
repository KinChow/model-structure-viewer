# 多模态 source-ref 到 Graph IR 覆盖审计

日期：2026-09-24。该审计只检查已有的 source-ref 模块是否在当前 Graph IR
中有对应节点；它不把 header 的总参数量当作视觉权重逐模块 truth。

## 结果

- 有视觉 source-ref 模块的多模态条目：**36**
- source-ref 中的视觉/视觉塔模块记录：**752**
- 能由 Graph IR canonical ID 精确或折叠祖先表示：**752/752**
- 当前明确的 source-ref 证据缺口：
  - `deepseek-ai/DeepSeek-V4.1-Flash`
  - `moonshotai/Kimi-K3`
  - `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` 的 sidecar 存在，但没有
    `visual/vision` 模块记录。

匹配允许两种发布到图的合法表示：

1. source-ref 模块对应一个同名 canonical 节点；
2. 同构视觉层被折叠时，由包含该路径的折叠节点表示。

不允许只因为模型是多模态就把视觉模块视为已覆盖；缺少 source-ref 的两个条目
被显式列为 gap，并由测试固定为已知缺口。

## 独立测试

`frontend/src/structure/models/multimodalSourceCoverage.test.js` 从 catalog
逐条加载 config、header-truth 和 source-ref，构造生产 artifacts 路径后检查
视觉模块覆盖。它不从被测 builder 生成期望模块名，也不检查总参数误差来替代
路径匹配。

这项审计证明了**source-ref 到图节点的表示覆盖**，不证明：

- 所有视觉 checkpoint 张量均已读取；
- merger/projector 的每个参数矩阵都与 safetensors tensor name 逐一绑定；
- 融合搬运量可以静态确定；
- decode 时视觉 tower 一定不会执行；
- 获得了 GPU 或推理框架实测性能结论。

后续仍需为两个缺少 source-ref 的条目补齐固定版本来源，补齐
`DeepSeek-V4-Flash-Vision-Exp` 的视觉模块 source-ref，并对 39 个多模态
条目的 projector/merger checkpoint tensor map 做逐模块审计。
