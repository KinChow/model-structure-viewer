import { moduleSpec, withShapeDims } from "./base.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function visionTowerModule(normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  return withShapeDims(moduleSpec(
    "vision_tower",
    "Vision Tower",
    "vision-encoder",
    {
      class: "VisionTower",
      hidden_size: normalized.visionHiddenSize,
      output_hidden_size: normalized.visionOutputSize,
      num_hidden_layers: normalized.visionLayers,
      ...shapeFlow(shapes.visionInput, shapes.visionOutput),
    },
    [],
    normalized.visionLayers,
  ), dims.visionInput, dims.visionOutput);
}
