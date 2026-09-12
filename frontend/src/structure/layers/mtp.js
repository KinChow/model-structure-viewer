// mtp.js —— Multi-Token Prediction 模块（W4）。
//
// 对标出处：vLLM 把 MTP 作为**独立注册项**而非 decoder 的子模块 ——
// `vllm/model_executor/models/registry.py` 里有 `DeepseekV32MTPModel` /
// `Qwen3_5MTP` / `MiniMaxM3MTP` / `Glm5NextMTPModel` / `KimiK3MTPModel`，
// 实现在各自的 `models/*/mtp.py`。本文件照此把 MTP 建成与 decoder 平级的模块。
//
// 组成（vLLM `deepseek_mtp.py` 的 DeepSeekMultiTokenPredictorLayer）：
//   enorm(RMSNorm) + hnorm(RMSNorm) + eh_proj(2H→H) + 一个完整 decoder 层
//   + shared_head.norm；lm_head 与主干共享，不重复计。
//
// **计费口径（显式声明）**：`repeat: 0`。投机解码默认不启用，MTP 不参与每次前向
// 的计算与访存，所以聚合时乘子为 0（cost/traverse.js childRepeatMultiplier）；
// 但它的**参数占显存**，由 derivedWeightParameters 计入 —— 对应五支柱的
// ②「放得下吗」。启用投机解码时把 repeat 改成实际的 speculative 步数即可。
import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { rmsNormModule } from "./norm.js";
import { operatorSpec } from "../operators/ops/index.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { hfNamedClass } from "../archs/index.js";

/** MTP 模块数：config 三种键名（DeepSeek/GLM 系、Qwen 系、MiniMax 系）。 */
export function mtpModuleCount(normalized) {
  return normalized.mtpModules || 0;
}

export function mtpModule(id, normalized, { attentionKind, layerKind }) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const hidden = normalized.hiddenSize || 0;
  const count = mtpModuleCount(normalized);
  const concatShape = `[batch, sequence, 2 x hidden size=${2 * hidden}]`;
  return withShapeDims(moduleSpec(
    id,
    "MTP",
    "mtp",
    {
      class: hfNamedClass(normalized, "mtpClass", "MTP"),
      modules: count,
      speculative_decoding: "disabled",
      compute_multiplier: 0,
      note: "投机解码未启用：参数计入显存，不计入每次前向的算力与访存",
      implementation: ["vLLM.models.*.mtp（registry 独立注册项）", "SGLang.srt/models/*_mtp.py"],
      // 数据流：enorm 与 hnorm 并行（分别归一化 embedding 与上一层 hidden），
      // 拼接后过 eh_proj，再走一个完整 decoder 层，最后 shared_head_norm。
      dataflow_edges: [
        ["enorm", "eh_proj"],
        ["hnorm", "eh_proj"],
        ["eh_proj", "layer"],
        ["layer", "shared_head_norm"],
      ],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [
      rmsNormModule(`${id}.enorm`, "embedding norm", normalized),
      rmsNormModule(`${id}.hnorm`, "hidden norm", normalized),
      operatorSpec(`${id}.eh_proj`, "embedding/hidden concat projection", "linear", {
        ...shapeFlow(concatShape, shapes.hidden),
        semantic_role: "mtp_concat_projection",
        implementation: ["vLLM.DeepSeekMultiTokenPredictorLayer.eh_proj"],
      }, { input: [-1, -1, 2 * hidden], output: dims.hidden }),
      decoderLayerModule(`${id}.layer`, normalized, { layerKind, attentionKind, layerIndex: 0 }),
      rmsNormModule(`${id}.shared_head_norm`, "shared head norm", normalized),
    ],
    // repeat=0：投机解码未启用 → 聚合乘子 0（见文件头口径声明）
    0,
  ), dims.hidden, dims.hidden);
}
