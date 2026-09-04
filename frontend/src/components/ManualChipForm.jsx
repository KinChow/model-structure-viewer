import { useState } from "react";
import { createManualChip, validateManualChipInput } from "../cost/chips/manual.js";

export default function ManualChipForm({ onAdd, language = "zh" }) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState({ name: "", memoryGb: "", memoryBandwidthTb: "", fp32Tflops: "", fp16Tflops: "", bf16Tflops: "", fp8Tflops: "", int8Tops: "", intraNodeGb: "", interNodeGb: "" });
  const [error, setError] = useState("");
  function update(key, value) { setValues((current) => ({ ...current, [key]: value })); }
  function add() {
    const errors = validateManualChipInput(values);
    if (errors.length) { setError(errors.join("；")); return; }
    onAdd?.(createManualChip(values));
    setError("");
    setOpen(false);
  }
  const labels = language === "en"
    ? { add: "Add GPU", name: "Name", memory: "Memory GB", intra: "Intra-node GB/s", inter: "Inter-node GB/s", submit: "Add" }
    : { add: "添加芯片", name: "名称", memory: "显存 GB", intra: "节点内 GB/s", inter: "节点间 GB/s", submit: "加入" };
  return <div className="manual-chip"><button type="button" onClick={() => setOpen((value) => !value)}>{labels.add}</button>{open && <div className="manual-chip-form"><label>{labels.name}<input value={values.name} onChange={(event) => update("name", event.target.value)} /></label><label>{labels.memory}<input type="number" min="0" value={values.memoryGb} onChange={(event) => update("memoryGb", event.target.value)} /></label><label>HBM TB/s<input type="number" min="0" step="0.01" value={values.memoryBandwidthTb} onChange={(event) => update("memoryBandwidthTb", event.target.value)} /></label><label>FP32 TFLOPS<input type="number" min="0" value={values.fp32Tflops} onChange={(event) => update("fp32Tflops", event.target.value)} /></label><label>FP16 TFLOPS<input type="number" min="0" value={values.fp16Tflops} onChange={(event) => update("fp16Tflops", event.target.value)} /></label><label>BF16 TFLOPS<input type="number" min="0" value={values.bf16Tflops} onChange={(event) => update("bf16Tflops", event.target.value)} /></label><label>FP8 TFLOPS<input type="number" min="0" value={values.fp8Tflops} onChange={(event) => update("fp8Tflops", event.target.value)} /></label><label>INT8 TOPS<input type="number" min="0" value={values.int8Tops} onChange={(event) => update("int8Tops", event.target.value)} /></label><label>{labels.intra}<input type="number" min="0" value={values.intraNodeGb} onChange={(event) => { update("intraNodeGb", event.target.value); update("interconnectGb", event.target.value); }} /></label><label>{labels.inter}<input type="number" min="0" value={values.interNodeGb} onChange={(event) => update("interNodeGb", event.target.value)} /></label><button type="button" onClick={add}>{labels.submit}</button>{error && <span className="cost-plan-error">{error}</span>}</div>}</div>;
}
