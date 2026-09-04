import { useState } from "react";
import { createManualChip, validateManualChipInput } from "../cost/chips/manual.js";

export default function ManualChipForm({ onAdd }) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState({ name: "", memoryGb: "", memoryBandwidthTb: "", bf16Tflops: "", interconnectGb: "" });
  const [error, setError] = useState("");
  function update(key, value) { setValues((current) => ({ ...current, [key]: value })); }
  function add() {
    const errors = validateManualChipInput(values);
    if (errors.length) { setError(errors.join("；")); return; }
    onAdd?.(createManualChip(values));
    setError("");
    setOpen(false);
  }
  return <div className="manual-chip"><button type="button" onClick={() => setOpen((value) => !value)}>添加芯片</button>{open && <div className="manual-chip-form"><label>名称<input value={values.name} onChange={(event) => update("name", event.target.value)} /></label><label>显存 GB<input type="number" min="0" value={values.memoryGb} onChange={(event) => update("memoryGb", event.target.value)} /></label><label>HBM TB/s<input type="number" min="0" step="0.01" value={values.memoryBandwidthTb} onChange={(event) => update("memoryBandwidthTb", event.target.value)} /></label><label>BF16 TFLOPS<input type="number" min="0" value={values.bf16Tflops} onChange={(event) => update("bf16Tflops", event.target.value)} /></label><label>互联 GB/s<input type="number" min="0" value={values.interconnectGb} onChange={(event) => update("interconnectGb", event.target.value)} /></label><button type="button" onClick={add}>加入</button>{error && <span className="cost-plan-error">{error}</span>}</div>}</div>;
}
