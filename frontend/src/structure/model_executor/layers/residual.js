import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { tensorDims } from "../dims.js";

export function attentionResidualModule(id, normalized) {
  const dims = tensorDims(normalized);
  const hidden = dims.hidden;
  const projection = [-1, -1, 1];
  return withShapeDims(moduleSpec(
    id,
    "Attention Residual",
    "residual",
    {
      class: "AttentionResidual",
      block_size: normalized.attnResBlockSize,
      ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[batch, sequence, hidden size]" },
    },
    [
      operatorSpec(`${id}.self_attention_res_norm`, "attention residual norm", "rmsnorm", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, hidden size]" },
      }),
      operatorSpec(`${id}.self_attention_res_proj`, "attention residual projection", "linear", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, 1]" },
      }, { input: hidden, output: projection }),
      operatorSpec(`${id}.mlp_res_norm`, "MLP residual norm", "rmsnorm", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, hidden size]" },
      }),
      operatorSpec(`${id}.mlp_res_proj`, "MLP residual projection", "linear", {
        ...{ input_shape: "[residual states, batch, sequence, hidden size]", output_shape: "[residual states, batch, sequence, 1]" },
      }, { input: hidden, output: projection }),
    ],
  ), dims.hidden, dims.hidden);
}

export function outputAttentionResidualModule(id, normalized) {
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "Output Attention Residual",
    "residual",
    { class: "OutputAttentionResidual", block_size: normalized.attnResBlockSize },
    [
      operatorSpec(`${id}.norm`, "output residual norm", "rmsnorm"),
      operatorSpec(`${id}.proj`, "output residual projection", "linear", {}, { input: dims.hidden, output: [-1, -1, 1] }),
    ],
  ), dims.hidden, dims.hidden);
}
