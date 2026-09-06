import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function hyperConnectionModule(id, normalized, phase = "branch") {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const names = {
    attn_mix: "HyperConnection attention mix",
    mlp_combine_mix: "HyperConnection MLP combine + mix",
    final: "HyperConnection final mixer",
    branch: id.split(".").at(-1) === "mixer" ? "Hyper Connection Mixer" : id.split(".").at(-1).replaceAll("_", " "),
  };
  return withShapeDims(moduleSpec(
    id,
    names[phase] || names.branch,
    "hyper-connection",
    {
      class: "GatedResidual",
      hc_phase: phase,
      hc_count: normalized.hyperConnectionCount,
      hc_lowrank: normalized.hyperConnectionLowrank,
      state_handoff: phase === "mlp_combine_mix" ? "from_previous_layer" : phase === "attn_mix" ? "to_next_layer" : undefined,
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [operatorSpec(`${id}.${phase}`, names[phase] || "hyper-connection mix", "hyper_connection", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}, injection`, shapes.hidden),
      hc_phase: phase,
      hc_count: normalized.hyperConnectionCount,
      hc_lowrank: normalized.hyperConnectionLowrank,
    }, { input: dims.hidden, output: dims.hidden })],
  ), dims.hidden, dims.hidden);
}

export function pleModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "PLE",
    "ple",
    {
      class: "Qwen4ExpPLELayer",
      embed_dim: normalized.pleEmbedDim,
      ngram_size: normalized.pleNgramSize,
      heads_per_ngram: normalized.pleHeadsPerNgram,
      conv_kernel_size: normalized.pleConvKernelSize,
      conv_dilation: normalized.pleNgramSize,
      key_projection_size: normalized.hiddenSize * (normalized.hyperConnectionCount || 1),
      value_projection_size: normalized.hiddenSize,
      implementation: ["ngram_embedding", "kv_proj", "grouped_norm", "gated_output", "dilated_short_conv"],
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [operatorSpec(`${id}.inject`, "PLE injection", "ple", {
      ...shapeFlow(`${shapes.hidden}, input_ids, ngram_context`, shapes.hidden),
      embed_dim: normalized.pleEmbedDim,
    }, { input: dims.hidden, output: dims.hidden })],
  ), dims.hidden, dims.hidden);
}

export function sharedExpertGateModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    "Shared Expert Gate",
    "shared-expert-gate",
    { class: "SharedExpertGate", ...shapeFlow(shapes.hidden, shapes.hidden) },
    [operatorSpec(`${id}.gate`, "shared expert gate", "shared_expert_gate", shapeFlow(shapes.hidden, shapes.hidden), { input: dims.hidden, output: dims.hidden })],
  ), dims.hidden, dims.hidden);
}

export function multiHyperConnectionModule(id, normalized, phase = "pre") {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const streams = normalized.mhcNumResidualStreams;
  const formulaByPhase = {
    pre: "mhc_pre",
    fused_post_pre: "mhc_fused_post_pre",
    post: "mhc_post",
    contract: "mhc_contract",
  };
  const names = {
    pre: "mHC attention pre",
    fused_post_pre: "mHC fused post + FFN pre",
    post: "mHC final post",
    contract: "mHC contract",
  };
  const formulaId = formulaByPhase[phase] || formulaByPhase.pre;
  const inputShape = phase === "pre"
    ? `[residual streams=${streams}, ${shapes.hidden}]`
    : phase === "contract"
      ? `[residual streams=${streams}, ${shapes.hidden}]`
      : `${shapes.hidden}, [residual streams=${streams}, ${shapes.hidden}]`;
  const outputShape = phase === "post" ? `[residual streams=${streams}, ${shapes.hidden}]` : shapes.hidden;
  const numericInput = phase === "pre" || phase === "contract" ? [-1, streams, ...dims.hidden.slice(1)] : dims.hidden;
  const numericOutput = phase === "post" ? [-1, streams, ...dims.hidden.slice(1)] : dims.hidden;
  return withShapeDims(moduleSpec(
    id,
    names[phase] || id.split(".").at(-1).replaceAll("_", " "),
    "multi-hyper-connection",
    {
      class: phase === "fused_post_pre" ? "MHCFusedPostPreOp" : phase === "post" ? "MHCPostOp" : phase === "contract" ? "HCContract" : "MHCPreOp",
      mhc_phase: phase,
      streams,
      sinkhorn_iterations: normalized.mhcSinkhornIterations,
      tau: normalized.mhcTau,
      hc_eps: normalized.mhcEps,
      post_mult_value: normalized.mhcPostMultValue,
      state_handoff: phase === "fused_post_pre" ? "from_previous_layer" : phase === "pre" ? "to_next_layer" : undefined,
      ...shapeFlow(inputShape, outputShape),
    },
    [operatorSpec(`${id}.${phase}`, names[phase] || "mHC operation", formulaId, {
      ...shapeFlow(inputShape, outputShape),
      streams,
      sinkhorn_iterations: normalized.mhcSinkhornIterations,
      tau: normalized.mhcTau,
      hc_eps: normalized.mhcEps,
      post_mult_value: normalized.mhcPostMultValue,
    }, { input: numericInput, output: numericOutput })],
  ), dims.hidden, dims.hidden);
}
