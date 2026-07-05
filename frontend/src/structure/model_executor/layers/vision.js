import { moduleSpec } from "./base.js";

export function visionTowerModule(normalized) {
  return moduleSpec(
    "vision_tower",
    "Vision Tower",
    "vision-encoder",
    {
      class: "VisionTower",
      hidden_size: normalized.visionConfig?.hidden_size,
      num_hidden_layers: normalized.visionLayers,
    },
    [],
    normalized.visionLayers,
  );
}
