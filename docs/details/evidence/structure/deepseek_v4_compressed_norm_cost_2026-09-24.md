# DeepSeek-V4 压缩器归一化执行频率

状态：本批仅修正 **嵌套 Compressor 的 norm 执行 token 域**；不把整个
C4/C128 fused compression 的成本称作已验证。

## 可复核的前向证据

- SGLang pinned revision `d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce`：
  [Compressor.forward_native](https://github.com/sgl-project/sglang/blob/d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce/python/sglang/srt/layers/attention/dsv4/compressor.py)
  先计算 `wkv_gate`，再将压缩结果传入 `compress_fused_norm_rope_inplace`。
- 同 revision
  [prefill plan](https://github.com/sgl-project/sglang/blob/d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce/python/sglang/kernels/jit/csrc/deepseek_v4/common.cuh)
  `plan_prefill_host` 与 CUDA 对应实现均仅在 `(position+1) % compress_ratio == 0`
  时填入 `compress_plan`，其他新 token 只可能进入 `write_plan`。下载文本
  SHA-256：`793ff2ce21920bb1a62531ae5355b58a69fafd6e476d9e1dac203cbc5e261ce6`。
- [C4](https://github.com/sgl-project/sglang/blob/d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce/python/sglang/kernels/jit/csrc/deepseek_v4/c4.cuh)
  与 [C128](https://github.com/sgl-project/sglang/blob/d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce/python/sglang/kernels/jit/csrc/deepseek_v4/c128.cuh)
  decode 写原始投影到窗口；只有 `seq_len % ratio == 0` 才输出压缩状态。
  [fused norm/RoPE](https://github.com/sgl-project/sglang/blob/d4dcce12d4fcdb3eb694d54d6afd31628c2eb7ce/python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope.cuh)
  对未对齐的 decode 请求直接返回。

这说明主 C4、HCA C128 和 indexer 内嵌 C4 的 norm 权重虽常驻，每次前向
只有压缩输出才执行归一化。旧 extractor 对三处都使用原始 sequence token 数，
会高估 vector、SFU 和激活流量，并在非边界 decode 虚构一次 norm 权重读取。

## 工作负载和实现边界

- 当前 prefill 接口将 `sequence` 视为**从零开始的完整新序列**，每请求
  `⌊sequence / ratio⌋` 个压缩输出；batch 相乘。
- decode 接口的 `sequence` 是包含当前 token 的长度：到达 ratio 边界时每请求
  一个输出，否则为零。没有输入每个请求的 prefix offset，故**带任意
  已存在前缀的追加 prefill**不能照搬上述完整序列公式；须先扩展工作负载输入，
  不能猜一个统一的边界数。
- `rmsnorm` 的现有向量/SFU 单位仍是理论动作，不能等同 GPU 实测 kernel 时间。
  norm 参数精度需要逐张量 checkpoint 或 loader 进一步核对。投影之后的
  APE 加权、fused 窗口 softmax/归约、实际缓存状态交通仍未完整计入
  `mla_kv_compress`；这些部分继续保留为待办，不声称本批已封闭整条压缩链。

实现以节点 `compression_output_ratio` 显式标注输出频率，统一
`compressedOutputTokens` 计算；没有改 Graph IR v2，也没有让 V4.1 flat
compressor 继承 V4 的嵌套 norm 语义。

## 发布配置对账（batch=1）

逐节点汇总**仅上述 norm 叶**，不是压缩器或模型总成本：

| 变体 | norm 节点（主 C4 / index C4 / 主 C128） | prefill S=4096 vector 旧→新 | SFU 旧→新 | actIn/actOut 各旧→新 | decode S=4095 norm 执行 |
|---|---:|---:|---:|---:|---|
| Flash、Flash-0731、Flash-Vision-Exp | 21 / 21 / 20 | 387,719,168 → 56,317,312 | 253,952 → 43,648 | 193,986,560 → 28,180,480 B | 0 |
| Pro、Pro-0813 | 30 / 30 / 30 | 565,862,400 → 80,546,880 | 368,640 → 62,400 | 283,115,520 → 40,304,640 B | 0 |
| V4.1-Flash（防回归） | 无此嵌套 norm | 0 → 0 | 0 → 0 | 0 → 0 | 不适用 |

对齐的 decode S=4096 与原来 norm 计数相同；所有五个 V4 条目的
norm 权重驻留字节数不变（Flash 组 47,360 B、Pro 组 69,120 B，
按当前模型的未量化声明口径）。上表不能用于推断压缩核其他部分已计全。
