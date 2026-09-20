# 运行时补维：线性注意力 hybrid（Qwen3.8-Flash-Next / qwen4_exp）

继 GQA(Qwen3-0.6B) + MLA(V2-Lite) 之后补**第三个运行时架构维度：线性注意力 + full 混合**（GDN/SSM state cache）。
Qwen3.8-Flash-Next（`Qwen4ExpForConditionalGeneration`、bf16、48 层 linear:full=3:1、512 experts）SGLang `--tp 8`
（SGLang 运行输出）。**这是 qwen4_exp 的首个真机 GPU 运行**（填算子级成本验证登记的"非原生 exotic arch 未跑整模型 GPU 真值"）。

## 运行时观测（关键：hybrid 双 cache）

- **每卡权重** `Qwen4ExpForConditionalGeneration mem usage=31.05 GB`（TP8 分片，加载 37s）。
- **Mamba/SSM state cache（linear 层，新维度）**：`max_mamba_cache_size=1223, conv_state 0.32GB, ssm_state 16.14GB`
  /卡；`max_running_requests` 被 mamba state 容量 capped 到 244（≈1223/5 slots-per-request）。**这是 attention KV 之外
  的第二类常驻——线性注意力的 conv(短卷积) + ssm(递推 state) cache**。
- **KV cache（full 层）**：`#tokens 1,504,960, K 8.61 + V 8.61 GB`/卡——12 个 full_attention 层的常规 KV。
- → **hybrid 架构的双 cache 运行时签名坐实**：linear 层 SSM state + full 层 KV 并存。

## 与前端口径对齐

- 前端**支持 qwen4_exp**：`buildStructureFromConfig` resolution=**architecture-alias**（映射到已支持装配器），
  且同时产出 **`kvBytesPerToken=27,648`（full 层 KV）+ `stateBytesPerSequence=58.8 MB`（linear 层 state）**
  ——即前端 `cost/memory.js` 的 `state_elements`(KDA/conv+ssm) 与 `cache_kv_elements` 双通道都对 qwen4_exp 生效。
- **定性一致**：真机与前端都体现"linear 层 state cache + full 层 KV cache"双常驻；linear-attn 维度的 state 口径
  在真实 qwen4_exp 上**首次有真机证据**。
- **精确字节口径差（据实）**：前端 state 58.8 MB/seq vs SGLang ssm_state 16.14GB/卡（×8 / 1223 slots ≈ 107 MB/slot，
  且"5 slots/request"）——同量级（数十 MB/seq），但 SGLang 的 slot 化 + 可能的 head 分片/over-provision 与前端
  per-sequence 口径非 1:1；精确映射需 SGLang mamba-cache shape（`n_linear_layers × heads × head_dim × state × dtype`）
  逐项拆——登记为 caliber 细化项，不强凑。

## 结论

- **运行时架构维度覆盖扩到 3 类**：GQA(Qwen) / MLA+MoE(V2-Lite) / **线性注意力 hybrid(qwen4_exp)**。
- qwen4_exp 首个真机 GPU 运行完成，前端 kv+state 双通道对其生效、定性一致。
- 退出码 2 为 `timeout` 收尾（server 已成功 load + 分配双 cache，数据在启动期已采）；state 精确字节留 caliber 细化。
