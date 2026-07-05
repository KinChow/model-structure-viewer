import { moduleSpec } from "./base.js";
import { operatorSpec } from "../ops/index.js";

export function rmsNormModule(id, name = "RMSNorm") {
  return moduleSpec(id, name, "normalization", { class: "RMSNorm" }, [
    operatorSpec(`${id}.rmsnorm`, "RMSNorm", "rmsnorm"),
  ]);
}
