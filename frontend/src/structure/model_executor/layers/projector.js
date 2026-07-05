import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function projectorModule(normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.visionOutput, shapes.hidden) : {};
  return moduleSpec("projector", "Multi-modal Projector", "projector", { class: "Projector", ...flow }, [
    operatorSpec("projector.linear", "vision-text projection", "linear", flow),
  ]);
}
