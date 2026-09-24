# Roofline 访存流量重复计数修复

日期：2026-09-24。Graph IR 保持 v2。

## 问题

注意力和稀疏索引公式同时输出：

- `bytes.actIn`：该动作的输入激活读流量；
- `bytes.kvRead`：输入激活中属于 KV cache 的诊断子集；
- `bytes.indexRead`：输入激活中属于 index cache 的诊断子集。

`kvRead` 和 `indexRead` 用于 cache/index 容量对账、动作明细和 UI
诊断，不是额外于 `actIn` 的两段流量。修复前
`cost/roofline.js` 计算：

```text
weights + actIn + actOut + kvRead + indexRead
```

会把同一批输入读取重复计入 roofline memory 时间、带宽移动量和算术强度。
这会影响使用 attention、DSA、QSA、MiniMax MSA 等动作的模型，但不会改变
MAC、vector/SFU 动作计数或 KV/index 诊断子项本身。

## 修复

Roofline 的互斥流量桶改为：

```text
bytesMoved = weights + actIn + actOut
```

`kvRead` 和 `indexRead` 继续保留在 action ledger、成本分组和 UI 中。
因此：

- cache/index 诊断信息不丢失；
- memory 时间不再重复计算；
- arithmetic intensity 使用去重后的总流量；
- 缺少 `weights`、`actIn` 或 `actOut` 时仍保持 `unknown`；
- Graph IR、参数量和执行动作计数不变。

## 独立断言

`frontend/src/cost/__tests__/roofline.test.js` 新增固定输入：

```text
weights=100, actIn=50, actOut=25,
kvRead=40, indexRead=20
```

期望 `bytesMoved=175`，而不是 `235`。该用例直接验证 roofline 消费者
没有把诊断子集当成额外流量。

已有的：

- `frontend/src/cost/__tests__/compute.test.js`；
- `frontend/src/cost/__tests__/ui.test.js`；
- QSA、DSA、MiniMax MSA 模型公式测试；

继续验证 `kvRead/indexRead` 在动作向量和成本 UI 中仍被保留。

## 证据边界

这是成本语义修复，不是硬件性能实测。它修正的是模型动作账本到 roofline
下界之间的流量归属，不声称获得 GPU benchmark 结果，也不改变未知融合搬运
继续保持 `unknown` 的策略。
