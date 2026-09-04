import { moduleSpec, withShapeDims } from "./base.js";
import { operatorSpec } from "../ops/index.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function projectorModule(normalized = null) {
  const shapes = normalized ? tensorShapes(normalized) : null;
  const flow = shapes ? shapeFlow(shapes.visionOutput, shapes.hidden) : {};
  const dims = normalized ? tensorDims(normalized) : null;
  return withShapeDims(moduleSpec("projector", "Multi-modal Projector", "projector", { class: "Projector", ...flow }, [
    operatorSpec("projector.linear", "vision-text projection", "linear", flow, { input: dims?.visionOutput, output: dims?.hidden }),
  ]), dims?.visionOutput, dims?.hidden);
}
