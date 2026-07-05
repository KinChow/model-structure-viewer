import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";

export function lmHeadModule(id = "lm_head") {
  return moduleSpec(id, "lm head", "output", { class: "Linear" }, [
    operatorSpec(`${id}.linear`, "output projection", "linear"),
  ]);
}
