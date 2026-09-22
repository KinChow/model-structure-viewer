import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { createManualChip, validateManualChipInput } from "../cost/chips/manual.js";
import { formatIssues } from "../i18n/format.js";
import useDialog from "../hooks/useDialog.js";

export default function ManualChipForm({ onAdd, language = "zh" }) {
  const [open, setOpen] = useState(false);
  const [theme, setTheme] = useState("dark");
  const panelRef = useRef(null);
  const triggerRef = useRef(null);
  const titleId = useId();
  const close = useCallback(() => {
    setOpen(false);
    requestAnimationFrame(() => triggerRef.current?.focus());
  }, []);
  useDialog({ open, onClose: close, panelRef });
  const [values, setValues] = useState({ name: "", memoryGb: "", memoryBandwidthTb: "", fp32Tflops: "", tf32Tflops: "", fp16Tflops: "", bf16Tflops: "", fp8Tflops: "", int8Tops: "", smTflops: "", sfuTops: "", intraNodeGb: "", interNodeGb: "", sourceUrl: "" });
  const [error, setError] = useState("");
  function update(key, value) { setValues((current) => ({ ...current, [key]: value })); }
  function add() {
    const errors = validateManualChipInput(values);
    if (errors.length) { setError(formatIssues(language, errors)); return; }
    onAdd?.(createManualChip(values));
    setError("");
    setOpen(false);
  }
  const labels = language === "en"
    ? { add: "Custom GPU", title: "Add custom GPU", close: "Close", name: "Name", memory: "Memory GB", intra: "Intra-node GB/s", inter: "Inter-node GB/s", source: "Source URL (optional)", submit: "Add GPU", cancel: "Cancel" }
    : { add: "自定义 GPU", title: "添加自定义芯片", close: "关闭", name: "名称", memory: "显存 GB", intra: "节点内 GB/s", inter: "节点间 GB/s", source: "来源 URL（可选）", submit: "加入芯片", cancel: "取消" };
  useEffect(() => {
    if (!open) return undefined;
    // 顶层遮罩阻止主列滚轮穿透；锁住文档滚动并在关闭后恢复原样式。
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [open]);
  // Portal 脱离成本面板的 z-index:1，不能靠提高子层 z-index 跨越吸顶栏。
  // 拦截 React 冒泡，避免弹窗点击触发 GPU 选择器外层 label 的默认行为。
  const dialog = open && <div className={`manual-chip-backdrop theme-${theme}`} role="presentation" onClick={(event) => event.stopPropagation()} onMouseDown={(event) => {
    if (event.target === event.currentTarget) {
      event.preventDefault();
      close();
    }
  }}>
      <div className="manual-chip-form" ref={panelRef} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="manual-chip-form-header"><div><strong id={titleId}>{labels.title}</strong><span>{language === "en" ? "Use only values you can verify. SM / Vector → vector_flops; SFU → sfu_ops." : "只填写能够确认来源的规格。SM / Vector → vector_flops；SFU → sfu_ops。"}</span></div><button type="button" className="manual-chip-close" aria-label={labels.close} onClick={close}>×</button></div>
        <label>{labels.name}<input value={values.name} onChange={(event) => update("name", event.target.value)} /></label>
        <label>{labels.memory}<input type="number" min="0" value={values.memoryGb} onChange={(event) => update("memoryGb", event.target.value)} /></label>
        <label>HBM TB/s<input type="number" min="0" step="0.01" value={values.memoryBandwidthTb} onChange={(event) => update("memoryBandwidthTb", event.target.value)} /></label>
        <label>FP32 TFLOPS<input type="number" min="0" value={values.fp32Tflops} onChange={(event) => update("fp32Tflops", event.target.value)} /></label>
        <label>TF32 TFLOPS<input type="number" min="0" value={values.tf32Tflops} onChange={(event) => update("tf32Tflops", event.target.value)} /></label>
        <label>FP16 TFLOPS<input type="number" min="0" value={values.fp16Tflops} onChange={(event) => update("fp16Tflops", event.target.value)} /></label>
        <label>BF16 TFLOPS<input type="number" min="0" value={values.bf16Tflops} onChange={(event) => update("bf16Tflops", event.target.value)} /></label>
        <label>FP8 TFLOPS<input type="number" min="0" value={values.fp8Tflops} onChange={(event) => update("fp8Tflops", event.target.value)} /></label>
        <label>INT8 TOPS<input type="number" min="0" value={values.int8Tops} onChange={(event) => update("int8Tops", event.target.value)} /></label>
        <label>SM / Vector TFLOPS<input type="number" min="0" value={values.smTflops} onChange={(event) => update("smTflops", event.target.value)} /></label>
        <label>SFU TOPS<input type="number" min="0" value={values.sfuTops} onChange={(event) => update("sfuTops", event.target.value)} /></label>
        <label>{labels.intra}<input type="number" min="0" value={values.intraNodeGb} onChange={(event) => { update("intraNodeGb", event.target.value); update("interconnectGb", event.target.value); }} /></label>
        <label>{labels.inter}<input type="number" min="0" value={values.interNodeGb} onChange={(event) => update("interNodeGb", event.target.value)} /></label>
        <label className="manual-chip-source">{labels.source}<input type="url" placeholder="https://…" value={values.sourceUrl} onChange={(event) => update("sourceUrl", event.target.value)} /></label>
        {error && <span className="cost-plan-error manual-chip-error">{error}</span>}
        <div className="manual-chip-actions"><button type="button" className="manual-chip-cancel" onClick={() => setOpen(false)}>{labels.cancel}</button><button type="button" onClick={add}>{labels.submit}</button></div>
      </div>
    </div>;
  return <div className="manual-chip">
    <button ref={triggerRef} type="button" className="manual-chip-trigger" aria-haspopup="dialog" onClick={(event) => {
      setTheme(event.currentTarget.closest(".theme-light") ? "light" : "dark");
      setOpen(true);
    }}>{labels.add}</button>
    {dialog && typeof document !== "undefined" ? createPortal(dialog, document.body) : null}
  </div>;
}
