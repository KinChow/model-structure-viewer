import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function hyperConnectionModule(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    id,
    id.split(".").at(-1) === "mixer" ? "Hyper Connection Mixer" : id.split(".").at(-1).replaceAll("_", " "),
    "hyper-connection",
    {
      class: "GatedResidual",
      hc_count: normalized.hyperConnectionCount,
      hc_lowrank: normalized.hyperConnectionLowrank,
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    [operatorSpec(`${id}.mix`, "hyper-connection mix", "hyper_connection", {
      ...shapeFlow(`${shapes.hidden}, ${shapes.hidden}, injection`, shapes.hidden),
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
    { class: "PositionLearningEnhancement", embed_dim: normalized.pleEmbedDim, ...shapeFlow(shapes.hidden, shapes.hidden) },
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
