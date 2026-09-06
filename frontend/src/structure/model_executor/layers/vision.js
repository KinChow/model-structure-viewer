import { moduleSpec, withShapeDims } from "./base.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function visionTowerModule(normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layers = normalized.visionLayers || 0;
  const layer = layers > 0
    ? moduleSpec(
      "vision_tower.0",
      "0 (VisionLayer)",
      "layer-group",
      { class: "VisionLayer", range: `0..${layers - 1}` },
      [],
      layers,
    )
    : null;
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
    layer ? [layer] : [],
    layers || undefined,
  ), dims.visionInput, dims.visionOutput);
}
