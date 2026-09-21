# Framework runtime validation artifacts

本目录按 2026-09-21 UTC 归档 H20/A100 runtime validation。
容器服务日志为 2026-09-21 19:xx；宿主带时区时间戳为
2026-09-22 03:xx +08:00。两者为同一时段，不改写原始日志和远端目录名。

## Artifacts

| Host | Archive | SHA-256 |
|---|---|---|
| H20 `10.98.95.16` | `h20-evidence-final.tar.gz` | `f40417616cd2255b79ada64969fa9bd1cca9b657de226b375ff560083c8c83dd` |
| A100 `10.55.87.81` | `a100-evidence-final2.tar.gz` | `bb66e0fc5bf24bf69cb2e036a6d478a50ea50e1f3b1282dbae58b131d139c06b` |

归档只包含：

- 容器版本、主机/GPU 快照；
- 本轮启动命令 JSON；
- server log、HTTP smoke 结果；
- runtime capture 的张量 shape/dtype/storage metadata；
- 不包含模型权重、AK/SK、token、cookie 或私密凭据。

## Scope

- H20：vLLM/SGLang Qwen3.5 GDN、GLM5-Next DSA/KDA、
  `dsv41_zzj_deploy` 的 DeepSeek-V4.1-Flash FP8 代理 build + DSpark。
- A100：vLLM DeepSeek-V2-Lite TP4 / TP4+EP4，SGLang/vLLM Qwen3.5，
  SGLang/vLLM Qwen3.5 MTP。

这些归档证明功能链路、cache spec、dtype、expert placement 和 draft pool
归属，不证明吞吐/延迟或所有 backend 的逐字节总显存等式。完整结论见：

`docs/details/evidence/memory/framework_runtime_validation_20260921.md`
