# 多模态 source-ref 到 Graph IR 覆盖审计

日期：2026-09-24。该审计只检查已有的 source-ref 模块是否在当前 Graph IR
中有对应节点；它不把 header 的总参数量当作视觉权重逐模块 truth。

## 结果

- 有视觉 source-ref 模块的多模态条目：**39**
- source-ref 中的视觉塔/projector 模块记录：**788**
- 能由 Graph IR canonical ID 精确或折叠祖先表示：**788/788**
- 当前 source-ref sidecar 缺口：**0**

DeepSeek 两个条目的视觉 forward 已由固定 revision 的 `inference/vision.py`
补充取证，具体 SHA、类和行号见 `deepseek_v4_vision_sources.json`；Kimi-K3
则使用固定 revision 的发布 `modeling_kimi_k3.py`。这些 source-ref 行只补齐
模块定义来源，没有把它们冒充成 checkpoint 逐张量绑定。

匹配允许两种发布到图的合法表示：

1. source-ref 模块对应一个同名 canonical 节点；
2. 同构视觉层被折叠时，由包含该路径的折叠节点表示。

不允许只因为模型是多模态就把视觉模块视为已覆盖；每个多模态条目现在都有
至少一个固定 revision 的视觉 tower/projector source-ref 行，并由测试检查其
canonical ID 表示。

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

后续仍需对 39 个多模态条目的 projector/merger checkpoint tensor map 做逐模块
审计；source-ref 覆盖本身不等于 checkpoint 绑定完成。
