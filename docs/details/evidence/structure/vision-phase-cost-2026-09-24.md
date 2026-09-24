# 视觉编码在 prefill/decode 的执行归属

日期：2026-09-24。适用范围：现有 39 个多模态内置条目，默认工作负载为**请求中一次图像输入，随后自回归生成文本**。

## 一手实现与运行时对照

- 发布模型的前向实现将编码作为图像输入路径：DeepSeek-V4-Flash-Vision-Exp `Transformer.encode_image` 调用 `aligner(vision(patches))`，再用 `merge_image_embeddings` 把结果放入文本隐状态。锚点：`inference/model.py:960-975`，revision `6821d6ad3681a4b137b066b76094fa82ebd0a380`；V4.1 相同路径在 `inference/model.py:1215-1228`，revision `dba1be0a40aa45a94ad051997016db3960a90277`。发布文件分别见 `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/model.py` 与 `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py`。
- vLLM 官方 `MultiModalConfig.allow_missing_mm_embeddings` 文档描述 E/P/D 实例间的交接：encoder 产出 embedding，prefill 使用 embedding，decode 消费 prompt KV。参考 `https://docs.vllm.ai/en/latest/api/vllm/config/multimodal/`。官方 E/P/D 示例进一步明确 encoder / prefill / decode 三个独立角色：`https://docs.vllm.ai/en/latest/examples/disaggregated/disaggregated_encoder/`。

## 本仓修正与边界

此前 `tokensFor({vision:true, phase:"decode"})` 仍返回完整 `visionTokens`，导致每个**文本 decode token**都再次计入视觉塔和外部 projector 的 MAC、激活流量与执行权重读取；融合节点已有 decode 跳过策略，内部费用因此自相矛盾。

现按 `attributes.modality="vision"` 在算子费用入口对 decode 返回零执行动作：tower、internal merger、external projector 共用同一边界；text decoder、KV/state 继续计算。Graph IR 与参数权重容量不变，视觉权重仍可驻留显存。不是把融合未知流量改成零：prefill 的 scatter/image-span 物化缺少占位细节时继续报告 unknown。

视频流、每个生成步注入新帧、多轮追加图片、重新执行带图片的 prefill、不同实例的 encoder 驻留分配**不属于**“每 decode token 重跑一次”的默认假设。没有每步图像输入调度时不伪造这些场景的数值。

独立机制测试 `frontend/src/structure/models/visionPhaseCost.test.js` 对 39 个条目的 config-only 与 artifacts 两条路径逐一验证：prefill 有实际视觉 GEMM/执行权重读；decode 所有视觉算子 matrix/vector/SFU/流量为零，语言 decoder MAC 非零，驻留权重前后相等。
