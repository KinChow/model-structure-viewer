import { moduleSpec } from "./base.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function visionTowerModule(normalized) {
  const shapes = tensorShapes(normalized);
  return moduleSpec(
    "vision_tower",
    "Vision Tower",
    "vision-encoder",
    {
      class: "VisionTower",
      hidden_size: normalized.visionConfig?.hidden_size,
      num_hidden_layers: normalized.visionLayers,
      ...shapeFlow(shapes.visionInput, shapes.visionOutput),
    },
    [],
    normalized.visionLayers,
  );
}
