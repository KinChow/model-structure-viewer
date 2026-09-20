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

## 跨-TP 布局重排补验（2026-09-20，prefill_tp=2 → decode_tp=1，A100 单机 3 卡）

补上文"未覆盖"的**跨 TP 布局重排**（MSV `comm.js: layoutRepackRequired = prefill_tp !== decode_tp`）。同 Qwen3-0.6B，
prefill server `--tp 2`（GPU0,1）+ decode server `--tp 1`（GPU2）+ mini-lb router，`mooncake_tcp`。

- **端到端跑通**（非 error）：两 server `fired up`、router `/generate` 200 OK、输出正确（"…Paris…"），错误扫描空 → **SGLang 支持 hetero-TP PD**。
- **确实触发布局重排（非 1:1 拷贝）**：decode 日志 `Performance is NOT guaranteed when using different TP sizes for non-MLA models`；
  源码 `disaggregation/common/conn.py::_resolve_rank_mapping`——等 TP `required_dst_info_num=1`（1:1、无重排），decode_tp<prefill_tp 时
  "one decode rank needs to retrieve KVCache from multiple prefill ranks"（`target_tp_ranks` 跨两 prefill rank）；`mooncake/conn.py`
  比较 `dst_attn_tp_size != attn_tp_size` 后按 head-slice 分块聚合（`group_concurrent_contiguous` / `utils.py` 的 aggregation-vs-scatter）。`transfer_layer_num=28` 逐层传。
- **字节口径一致**：decode(tp1) `KV 15.15+15.15 GB / 283,681 tok → K+V=114,688 B/token`（28·8·128·2·2）== MSV/前值；
  每 prefill(tp2) rank `K 28,672 B/tok = 28·4·128·2`（各持 4/8 kv_heads）→ decode(8) 由 prefill(4+4) 聚合 = 重排本身。
- **结论**：**MSV `layoutRepackRequired=(prefill_tp≠decode_tp)` 与 SGLang 真机一致**——等 TP 无重排（0.0% 已验）、hetero-TP 触发跨-TP KV head 布局重排（真机跑通、非硬失败）。
  至此 PD 分离的机制 + KV 字节 + 跨-TP 重排判据单机全验；仅 inter-node RDMA 实测带宽（`transferSeconds` 的带宽项）留 ≥2 节点。
