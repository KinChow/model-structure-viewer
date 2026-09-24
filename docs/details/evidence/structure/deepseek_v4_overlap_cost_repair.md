# DeepSeek V4 C4 overlap-window cost repair

状态：**已修复并通过前端全量门禁**。日期：2026-09-24。Graph IR 保持 v2。

## 问题

DeepSeek V4 的 C4 compressor/indexer 使用两段压缩流：

- 当前窗口的 `Cb`；
- 上一窗口的 `Ca`，作为当前窗口的 overlap half。

官方 Transformers pinned forward (`v5.16.1`,
`modeling_deepseek_v4.py`) 在首次无 cache 的 forward 中将 overlap half
初始化为 `-inf` gate，因此第一条压缩输出实际上只有当前窗口的 `ratio`
个有效 slot。后续窗口才有 `2 * ratio` 个有效 slot。之前成本公式对所有
C4 输出都直接使用 `2 * ratio`，把第一窗口不存在的历史 overlap 当作真实
计算、SFU 和激活流量。

这不是普通的 shape mismatch，也不是把父节点费用调小就能解决的问题，而是
压缩状态生命周期与窗口边界的语义错误。

## 修复

`dsv4WindowReduceCounts` 现在区分：

- C4 prefill 的首个压缩窗口：`ratio` 个有效 slot；
- C4 prefill 的后续窗口：每个窗口 `2 * ratio` 个有效 slot；
- C4 decode 在首次边界：`ratio` 个有效 slot；
- 后续 decode 边界：`2 * ratio` 个有效 slot；
- C128/HCA：保持原有非 overlap 逻辑；
- V4.1 flat compressor：不继承 V4 nested overlap 规则。

修复只影响 window reduction 的 vector/SFU/activation traffic，不修改：

- checkpoint 权重驻留；
- Graph IR 节点或协议；
- compressor/indexer 的结构；
- attention 的 compressed key 数学；
- V4.1 的 CSA2 规则。

## 验证

新增独立公式用例：

- 首窗口与后续窗口；
- prefill 与 decode；
- overlap slot 数和 softmax/SFU/流量；
- V4 五个发布变体的 nested compressor；
- V4.1 防回归。

结果：

- DeepSeek V4 compressor 专项测试：**16/16**
- Frontend 全量单测：**603/603**
- 原有 T4/W5、header truth、Graph IR 和成本测试均通过。

## 边界

本修复仍不声称获得 GPU fused-kernel 实测性能。APE 访存、实际 kernel
融合程度、量化 scale 交通和跨请求 cache 追加 prefill 仍按现有 unknown
边界处理；本次只修复有 pinned forward 证据支持的首窗口 overlap 计数。
