# PD 分离（Prefill-Decode disaggregation）运行时验证（A100 单机双卡真机）

用户指出 PD 分离没验证。此前 MSV 只有建模（`comm.js pdKvTransferBytes`、`parallel.js
projectPdFit/validatePdPlan`、`pdSummary.test.js` 单测），`validation_status` 无 PD 项，零真机。本项补真机。

## 真机拓扑（SGLang disaggregation，单机 2×A100）

- **Prefill** 实例：GPU0，`--disaggregation-mode prefill --disaggregation-transfer-backend mooncake_tcp
  --disaggregation-bootstrap-port 9001`
- **Decode** 实例：GPU1，`--disaggregation-mode decode --disaggregation-transfer-backend mooncake_tcp`
- **Router**：`sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30031 9001
  --decode http://127.0.0.1:30032 --mini-lb`，端口 30030
- 模型 `Qwen/Qwen3-0.6B`（28 层 GQA，kv_heads=8, head_dim=128, bf16）
- 关键坑：`get_local_ip_auto()` 自动选到 IPv6 link-local `fe80::…` → ZMQ 绑定失败；
  设 `SGLANG_HOST_IP=127.0.0.1` 解决。mooncake_tcp 自动 `MC_FORCE_TCP=1`（无 RDMA，走 TCP）。

## 端到端结果

```
# prefill/decode 两侧均 "Disaggregation warmup requests completed" + "fired up and ready"
# decode 侧：Attached hybrid pool stack to UnifiedRadixCache: pools=KV, transfer_layer_num=28
# 请求经 router：
POST /generate {"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}
-> "Paris. The capital of France is also the capital of the Republic of France."
# prefill 日志：Prefill batch #new-seq:1 #new-token:5 #inflight-req:1（5 prompt token 在 P 侧算 KV）
# 输出 16 token 由 D 侧生成 → P 算 KV、TCP 传 D、D 解码，PD 分工真机成立
```

## KV 传输字节口径对账（真机 vs MSV）

- **每 token 每层每侧 K** = kv_heads·head_dim·dtype = 8·128·2B = **2,048 B**；K+V = 4,096 B/token/层。
- 全 28 层：K+V = **114,688 B/token**。反查 prefill KV 池：`K size 15.15 GiB / #tokens 283,681 =
  57,344 B/token`（K，28 层）= 8·128·28·2 ✓；K+V = 114,688 B/token ✓。
- decode 侧 `transfer_layer_num=28` → **全 28 层 KV 逐层传输**；5-token prompt 传输量 = 5·114,688 = 573,440 B。
- MSV `pdKvTransferBytes`（prefill_tp=decode_tp=1）：`perDecodeRankBytes = totalKvBytes`（不分片），
  `layoutRepackRequired = (prefill_tp≠decode_tp) = false` —— 与真机一致：同 TP 无布局重排，逐层全量传。

| 量 | 真机 SGLang | MSV | 差 |
|---|---|---|---|
| 传输层数 | 28（transfer_layer_num） | num_hidden_layers=28 | **0.0%** |
| KV /token/层/侧 | 2,048 B（kv_heads·head_dim·2B） | 2,048 B | **0.0%** |
| KV /token（K+V，全层） | 114,688 B | 114,688 B | **0.0%** |
| layoutRepack（tp1→tp1） | 无（同 TP 直传） | layoutRepackRequired=false | ✅ 一致 |

## 结论 + 边界

- **PD 分离机制 + KV 传输字节口径已 A100 真机验证**：prefill/decode 跨卡分工、KV 经 mooncake TCP 逐层传输、
  输出正确；传输层数与每 token KV 字节与 MSV `pdKvTransferBytes` 逐点一致。
- **未覆盖**：跨 TP 布局重排（prefill_tp≠decode_tp，MSV 有 `layoutRepackRequired` 建模，本次同 TP 未触发）；
  inter-node RDMA 实际带宽（本次 TCP localhost，MSV `transferSeconds=bytes/linkBandwidth` 的带宽项须多机实测）。
  这两项属多机/跨拓扑范畴，A100 单机已验到机制 + 字节口径。
- 对 `validation_status` 补充：PD 分离从"仅建模+单测"更新为"**机制 + KV 传输字节口径真机已验**"。

复现：上文三进程启动命令（设 `SGLANG_HOST_IP=127.0.0.1`）+ router `/generate`。
