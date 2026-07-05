import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";

export function projectorModule() {
  return moduleSpec("projector", "Multi-modal Projector", "projector", { class: "Projector" }, [
    operatorSpec("projector.linear", "vision-text projection", "linear"),
  ]);
}
