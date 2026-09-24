# 23 个非 V4.1 变体的独立发布资料复核

日期：2026-09-24。目标是补齐此前审计中“同族线索、未独立直读”的
23 个内置条目，不把量化/Base 变体默认当成“只改 dtype”。

## 方法

- 对每个条目读取该条目固定 revision 的 Hugging Face 官方 `README.md`；
- 记录 revision、卡片 SHA256、读取字节数和卡片 URL；
- 与本地同族基准 config 做字段差异比较；
- 对有 config 差异的条目，再结合本地 header truth 判断发布权重是否真的
  实装了对应模块；
- 卡片属于发布说明，不能替代 forward 实现或 checkpoint tensor map；
  因此本记录的状态是“资料已读”，不是“结构已验证”。

完整机器记录在
`variant_card_review_2026-09-24.json`，共 **23/23** 个条目成功读取。

## 结果

### 量化/精度变体

Qwen 的 FP8、GPTQ-Int4，GLM 的 FP8/BF16，以及 MiniMax 的 MXFP8
均成功读取了各自发布卡片。与同族基准 config 的结构字段对比中，
除量化/dtype/版本元数据外，Qwen、GLM 变体没有发现额外结构字段差异。
这支持“目前按同一拓扑建模”的决定，但不证明其 packed weight layout、
scale tensor 或 activation quantization 细节已经完全对账；这些仍由
checkpoint header/forward implementation 负责。

### Base / Instruct 变体

Qwen3.5 Base 条目、Kimi-K2-Instruct 均成功读取独立卡片。配置比较没有
发现结构字段差异。Qwen Base 卡片的宣传文字是家族级描述，不能反推
某个具体 Base 条目包含 MoE；实际拓扑仍以该条目 config 和 checkpoint
为准。

### MiniMax-M3-MXFP8 的 MTP 差异

这是本批唯一出现非 dtype/量化 config 差异的变体：

```text
MiniMax-M3:
  num_nextn_predict_layers = 1
  num_mtp_modules = 7

MiniMax-M3-MXFP8:
  num_nextn_predict_layers 缺失
  num_mtp_modules = 1
```

但 MXFP8 的本地生产 header truth 为 `mtp_tensor_count = 0`。因此不能
仅依据它的 config 生成 MTP；现有生产装配器按 checkpoint truth 抑制
MTP，符合“配置是意图、权重是实装真相”的规则。该变体不需要结构代码
修改，本次增加的是证据和回归记录。

### GLM-5.2-FP8 / GLM-5.3-BF16 / GLM-5.3-Flash-BF16

官方卡片均已独立读取；卡片内容没有披露新的宏观拓扑。GLM DSA
IndexShare 的物理路径与共享层修复见
`glm_dsa_indexshare_binding.md`，不因 BF16/FP8 变体重复实例化
indexer。

## 当前边界

- **23/23 资料读取完成**，但不能把这 23 个条目写成“架构验证通过”；
  仍需在有 forward/checkpoint 证据时逐模块确认。
- 现有 60 条目桌面扫描已完成；39 个视觉条目的 roofline 显示
  `unknown` 是有意保留的多模态 fusion traffic unknown，不是把未知
  偷换为 0，也不应伪造成 GPU 性能结论。
- 未下载完整权重；只读取官方文本、config 和已存在的 header truth。
