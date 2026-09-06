import { useEffect, useMemo, useState } from "react";
import { normalizeConfig } from "../structure/config/normalize.js";
import { aggregateCost } from "../cost/aggregate.js";
import { maxContextForStages, projectPdFit, projectPlan } from "../cost/parallel.js";
import { pdKvTransferBytes, planCommunicationBytes } from "../cost/comm.js";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import ManualChipForm from "./ManualChipForm.jsx";
import { DEFAULT_COMPARE_PLAN, DEFAULT_LOADS, DEFAULT_NODES, DEFAULT_PLAN } from "../cost/defaults.js";
import { DEFAULT_EFFICIENCY } from "../cost/efficiency.js";
import { classifyRoofline } from "../cost/roofline.js";
import { formatBytes, formatMacs, formatRate, formatSeconds } from "../formatters.js";

const GIB = 1024 ** 3;
const FLOPS_ORDER = ["fp32", "fp16", "bf16", "fp8", "int8"];

function formatPeakFlops(peakFlops, unknownLabel) {
  const values = FLOPS_ORDER
    .filter((dtype) => Number.isFinite(peakFlops?.[dtype]) && peakFlops[dtype] > 0)
    .map((dtype) => `${dtype.toUpperCase()} ${formatMacs(peakFlops[dtype])}`);
  return values.join(" · ") || unknownLabel;
}

function fitText(value, english = false) {
  return value == null ? (english ? "unknown" : "未知") : value ? (english ? "yes" : "是") : (english ? "no" : "否");
}

function PlanFields({ plan, onChange, english }) {
  const update = (key, value) => onChange({ ...plan, [key]: Math.max(1, Number(value) || 1) });
  return <div className="cost-plan-fields">
    <label>TP<input type="number" min="1" value={plan.tp} onChange={(event) => update("tp", event.target.value)} /></label>
    <label>PP<input type="number" min="1" value={plan.pp} onChange={(event) => update("pp", event.target.value)} /></label>
    <label>EP<input type="number" min="1" value={plan.ep} onChange={(event) => update("ep", event.target.value)} /></label>
    <label>DP<input type="number" min="1" value={plan.dp} onChange={(event) => update("dp", event.target.value)} /></label>
    <label>{english ? "Attention parallelism" : "Attention 并行方式"}<select value={plan.attnMode} onChange={(event) => onChange({ ...plan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label>
  </div>;
}

export default function CostSummary({ structure, chips = PUBLIC_CHIPS, onAddChip, language = "zh", onFitStatusChange, lenses: controlledLenses, onLensesChange, phase: controlledPhase, onPhaseChange, mode: controlledMode, onModeChange, plans: controlledPlans, onPlansChange, nodes: controlledNodes, onNodesChange, gpusPerNode: controlledGpusPerNode, onGpusPerNodeChange, machineId: controlledMachineId, onMachineIdChange, loads: controlledLoads, onLoadsChange, comparisonMode = "off", onComparisonModeChange, compareChipId = "", onCompareChipIdChange, comparePlan = DEFAULT_COMPARE_PLAN, onComparePlanChange, efficiency = DEFAULT_EFFICIENCY, onEfficiencyChange }) {
  const english = language === "en";
  const text = {
    estimate: english ? "Theoretical cost estimate" : "理论成本估算",
    disclaimer: english ? "Theoretical calculation; not simulation or prediction" : "理论计算，非仿真、非预测",
    expand: english ? "Expand config" : "展开配置",
    collapse: english ? "Collapse config" : "收起配置",
    machine: english ? "Machine node" : "机器节点",
    gpu: english ? "GPU" : "GPU / 芯片",
    gpuNode: english ? "GPU / node" : "GPU / 节点",
    unknownFlops: english ? "compute unknown" : "算力未知",
    mode: english ? "Serving mode" : "运行模式",
    centralized: english ? "Centralized" : "集中式（非 PD）",
    pd: english ? "PD disaggregation" : "PD 分离",
    node: english ? "node" : "节点",
    nodes: english ? "nodes" : "节点",
    fit: english ? "fit" : "适配",
    fitCard: english ? "Fit / card" : "单卡适配",
    needsGpus: english ? (count) => `needs ${count} GPUs` : (count) => `需要 ${count} 张 GPU`,
    memoryNoFit: english ? "memory no-fit" : "显存不足",
    planValid: english ? "plan valid" : "方案有效",
    input: english ? "Input tokens / request" : "输入 tokens / request",
    context: english ? "Current context length" : "当前上下文长度",
    independent: english ? "P/D workloads, node counts, and parallel plans are saved independently; only the current phase is shown." : "P/D 负载、节点规模和并行策略独立保存；当前只显示当前阶段结果。",
    theoretical: english ? "Cost is a theoretical estimate; no scheduling, pipeline bubble, or transfer overlap simulation." : "理论计算，不模拟调度、流水线气泡或传输重叠。",
    chunkedSummary: (total, peak) => english ? `Total uses ${total} input tokens; peak uses ${peak} tokens per chunk.` : `总量使用 ${total} 个输入 tokens；峰值按每个 chunk ${peak} 个 tokens 计算。`,
    analysis: english ? "Graph analysis" : "图分析",
    compare: english ? "Compare" : "对比",
    off: english ? "Off" : "关闭",
    chipCompare: english ? "GPU" : "芯片",
    planCompare: english ? "Plan" : "方案",
    compareGpu: english ? "Compare GPU" : "对比芯片",
    compareTp: english ? "Compare TP" : "对比 TP",
    compareEp: english ? "Compare EP" : "对比 EP",
    compareAttention: english ? "Compare Attention" : "对比 Attention",
    etaFlops: "ηF",
    etaHbm: "ηHBM",
    etaComm: "ηComm",
  };
  const phase = controlledPhase ?? "prefill";
  const mode = controlledMode ?? "centralized";
  const changePhase = (next) => onPhaseChange?.(next);
  const changeMode = (next) => onModeChange?.(next);
  const lenses = controlledLenses || new Set(["vram"]);
  const machineId = controlledMachineId ?? chips[0]?.id ?? "";
  const changeMachine = (next) => onMachineIdChange?.(next);
  const nodes = controlledNodes || DEFAULT_NODES;
  const updateNodes = (next) => onNodesChange?.(next);
  const gpusPerNode = controlledGpusPerNode ?? 8;
  const updateGpusPerNode = (next) => onGpusPerNodeChange?.(next);
  const loads = controlledLoads || DEFAULT_LOADS;
  const updateLoads = (next) => onLoadsChange?.(next);
  const plans = controlledPlans || { prefill: DEFAULT_PLAN, decode: DEFAULT_PLAN };
  const updatePlan = (next) => onPlansChange?.(next);
  const [kvElementBytes, setKvElementBytes] = useState(2);
  const [activationGiB, setActivationGiB] = useState(1.5);
  const [runtimeGiB, setRuntimeGiB] = useState(1.5);
  const [commBufferGiB, setCommBufferGiB] = useState(0);
  const [weightMode, setWeightMode] = useState("actual");
  const [expanded, setExpanded] = useState(false);
  const config = useMemo(() => structure?.extra_config ? normalizeConfig(structure.extra_config) : null, [structure]);
  const machine = chips.find((chip) => chip.id === machineId) || chips[0];
  const load = loads[phase];
  const plan = plans[phase];
  const costFor = (targetPhase) => {
    const targetLoad = loads[targetPhase] || DEFAULT_LOADS[targetPhase];
    if (!structure?.graph || !config) return null;
    return aggregateCost({ graph: structure.graph, config, parameterCount: structure.summary?.parameters_by_dtype, phase: targetPhase, batch: targetLoad.batch, sequence: targetLoad.sequence, kvBytes: kvElementBytes, activationPeak: activationGiB * GIB, runtimeConst: runtimeGiB * GIB, commBuffer: commBufferGiB * GIB, weightBytesPerParameter: weightMode === "actual" ? undefined : Number(weightMode) });
  };
  const phaseCosts = useMemo(() => ({ prefill: costFor("prefill"), decode: costFor("decode") }), [structure, config, loads, plans, kvElementBytes, activationGiB, runtimeGiB, commBufferGiB, weightMode]);
  const cost = phaseCosts[phase];
  const peakCost = useMemo(() => {
    if (!cost || !load.chunked || phase !== "prefill") return cost;
    return aggregateCost({ graph: structure.graph, config, parameterCount: structure.summary?.parameters_by_dtype, phase, batch: load.batch, sequence: Math.min(load.sequence, load.chunkSize), kvBytes: kvElementBytes, activationPeak: activationGiB * GIB, runtimeConst: runtimeGiB * GIB, commBuffer: commBufferGiB * GIB, weightBytesPerParameter: weightMode === "actual" ? undefined : Number(weightMode) });
  }, [cost, load, phase, structure, config, kvElementBytes, activationGiB, runtimeGiB, commBufferGiB, weightMode]);
  const available = machine?.memory_bytes || 0;
  const projected = useMemo(() => cost && machine ? projectPlan({ graph: structure.graph, weightBytes: cost.memory.weightBytes, kvBytes: cost.memory.kvBytes, stateBytes: cost.memory.stateBytes, config, plan }) : null, [cost, machine, structure, config, plan]);
  const communication = useMemo(() => cost ? planCommunicationBytes({ graph: structure.graph, config, plan, batch: load.batch, tokens: phase === "decode" ? 1 : (load.chunked ? Math.min(load.sequence, load.chunkSize) : load.sequence) }) : null, [cost, structure, config, plan, load, phase]);
  const roofline = useMemo(() => cost && machine ? classifyRoofline({
    macs: cost.totalMacs,
    weightBytes: cost.memory.weightBytes,
    actInBytes: peakCost?.memory.activationBytes || 0,
    commBytes: communication?.totalBytes || 0,
  }, machine, { dtype: "bf16", efficiency }) : null, [cost, machine, peakCost, communication, efficiency]);
  const pd = useMemo(() => mode === "pd" && phaseCosts.prefill && machine ? pdKvTransferBytes({ totalKvBytes: phaseCosts.prefill.memory.kvBytes, totalStateBytes: phaseCosts.prefill.memory.stateBytes, config, pdPlan: { prefill_plan: plans.prefill, decode_plan: plans.decode }, prefillChip: machine, decodeChip: machine }) : null, [mode, phaseCosts, machine, config, plans]);
  const pdFit = useMemo(() => mode === "pd" && phaseCosts.prefill && phaseCosts.decode && machine ? projectPdFit({ graph: structure.graph, weightBytes: phaseCosts.prefill.memory.weightBytes, prefillKvBytes: phaseCosts.prefill.memory.kvBytes, decodeKvBytes: phaseCosts.decode.memory.kvBytes, prefillStateBytes: phaseCosts.prefill.memory.stateBytes, decodeStateBytes: phaseCosts.decode.memory.stateBytes, config, pdPlan: { prefill_plan: plans.prefill, decode_plan: plans.decode }, prefillChip: machine, decodeChip: machine, activationBytes: peakCost?.memory.activationBytes || 0, runtimeBytes: cost.memory.runtimeBytes, commBufferBytes: cost.memory.commBufferBytes }) : null, [mode, phaseCosts, cost, machine, structure, config, plans, peakCost]);
  const currentNodes = mode === "pd" ? nodes[phase] : nodes.centralized;
  const totalGpus = currentNodes * gpusPerNode;
  const requiredGpus = plan.tp * plan.pp * plan.dp;
  const planFitsTopology = requiredGpus <= totalGpus;
  const planFitsMemory = mode === "pd"
    ? pdFit?.[phase]?.fit
    : projected?.ok && projected.stages.every((stage) => stage.weightBytes + stage.kvBytes + (stage.stateBytes || 0) + peakCost.memory.activationBytes + cost.memory.runtimeBytes + cost.memory.commBufferBytes <= available);
  const planStatus = !planFitsTopology ? text.needsGpus(requiredGpus) : planFitsMemory === false ? text.memoryNoFit : text.planValid;
  useEffect(() => {
    if (!cost || !machine) {
      onFitStatusChange?.(null);
      return;
    }
    onFitStatusChange?.({ fit: planFitsTopology && planFitsMemory === true, known: planFitsMemory != null, status: planStatus, phaseFits: mode === "pd" ? { prefill: pdFit?.prefill?.fit, decode: pdFit?.decode?.fit } : null });
  }, [cost, machine, mode, onFitStatusChange, pdFit, planFitsTopology, planFitsMemory, planStatus]);
  if (!cost || !machine) return null;
  const planMaxContext = projected?.ok ? maxContextForStages(projected.stages, { capacityBytes: available, activationBytes: peakCost.memory.activationBytes, runtimeBytes: cost.memory.runtimeBytes + cost.memory.commBufferBytes, sequence: load.sequence }) : null;
  const updateLoad = (key, value) => updateLoads({ ...loads, [phase]: { ...loads[phase], [key]: value } });
  const toggleLens = (name) => {
    const next = new Set(lenses);
    if (name === "none") next.clear();
    else next.has(name) ? next.delete(name) : next.add(name);
    onLensesChange?.(next);
  };
  return <section className="cost-summary cost-summary-modern" aria-label={text.estimate}>
    <div className="cost-summary-header"><div><b>{text.estimate}</b><span className="cost-disclaimer">{text.disclaimer}</span></div><button className="cost-expand-button" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? text.collapse : text.expand}</button></div>
    <div className="cost-lens-row"><span>Cost Lens</span>{[["none", "None"], ["vram", "VRAM"], ["compute", "Compute"], ["memory", "Memory"], ["kv", "KV Cache"]].map(([id, label]) => <button type="button" key={id} className={(id === "none" ? lenses.size === 0 : lenses.has(id)) ? "active" : ""} aria-pressed={id === "none" ? lenses.size === 0 : lenses.has(id)} onClick={() => toggleLens(id)}>{label}</button>)}</div>
    <div className="cost-machine-summary"><span><b>{machine.name}</b> · {currentNodes} {currentNodes === 1 ? text.node : text.nodes} · {totalGpus} GPU</span><span>{formatBytes(machine.memory_bytes)} / card</span><span className={planFitsTopology && planFitsMemory !== false ? "fit" : "no-fit"}>{planStatus}</span></div>
    {mode === "pd" && <div className="pd-deployment-summary"><span><b>Prefill</b> · {nodes.prefill} {nodes.prefill === 1 ? text.node : text.nodes} × {gpusPerNode} GPU · TP{plans.prefill.tp} / PP{plans.prefill.pp} / EP{plans.prefill.ep} / DP{plans.prefill.dp} · {text.fit} {fitText(pdFit?.prefill?.fit, english)}</span><span><b>Decode</b> · {nodes.decode} {nodes.decode === 1 ? text.node : text.nodes} × {gpusPerNode} GPU · TP{plans.decode.tp} / PP{plans.decode.pp} / EP{plans.decode.ep} / DP{plans.decode.dp} · {text.fit} {fitText(pdFit?.decode?.fit, english)}</span></div>}
    {expanded && <div className="cost-config-modern">
      <div className="cost-config-section"><h4>{text.machine}</h4><div className="cost-config-grid"><label>{text.gpu}<select value={machine.id} onChange={(event) => changeMachine(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label>{mode === "centralized" && <label>Nodes<input type="number" min="1" value={nodes.centralized} onChange={(event) => updateNodes({ ...nodes, centralized: Math.max(1, Number(event.target.value) || 1) })} /></label>}<label>{text.gpuNode}<input type="number" min="1" value={gpusPerNode} onChange={(event) => updateGpusPerNode(Math.max(1, Number(event.target.value) || 1))} /></label><div className="cost-machine-spec">{formatBytes(machine.memory_bytes)} / card · {formatBytes(machine.memory_bandwidth)} HBM · {formatPeakFlops(machine.peak_flops, text.unknownFlops)}</div></div><ManualChipForm language={language} onAdd={(chip) => { onAddChip?.(chip); changeMachine(chip.id); }} /></div>
      <div className="cost-config-section"><div className="cost-section-heading"><h4>{text.mode}</h4><div className="cost-segmented"><button type="button" className={mode === "centralized" ? "active" : ""} aria-pressed={mode === "centralized"} onClick={() => changeMode("centralized")}>{text.centralized}</button><button type="button" className={mode === "pd" ? "active" : ""} aria-pressed={mode === "pd"} onClick={() => changeMode("pd")}>{text.pd}</button></div></div>{mode === "pd" && <div className="cost-phase-switch"><button type="button" className={phase === "prefill" ? "active" : ""} aria-pressed={phase === "prefill"} onClick={() => changePhase("prefill")}>Prefill</button><button type="button" className={phase === "decode" ? "active" : ""} aria-pressed={phase === "decode"} onClick={() => changePhase("decode")}>Decode</button></div>}<div className="cost-config-grid"><label>{phase === "prefill" ? text.input : text.context}<input type="number" min="1" value={load.sequence} onChange={(event) => updateLoad("sequence", Math.max(1, Number(event.target.value) || 1))} /></label><label>Batch size<input type="number" min="1" value={load.batch} onChange={(event) => updateLoad("batch", Math.max(1, Number(event.target.value) || 1))} /></label>{mode === "pd" && <label>Nodes / {phase}<input type="number" min="1" value={nodes[phase]} onChange={(event) => updateNodes({ ...nodes, [phase]: Math.max(1, Number(event.target.value) || 1) })} /></label>}{mode === "pd" && phase === "prefill" && <label className="cost-check"><input type="checkbox" checked={load.chunked} onChange={(event) => updateLoad("chunked", event.target.checked)} /> Chunked Prefill</label>}{mode === "pd" && phase === "prefill" && load.chunked && <label>Prefill chunk size<input type="number" min="1" value={load.chunkSize} onChange={(event) => updateLoad("chunkSize", Math.max(1, Number(event.target.value) || 1))} /></label>}</div>{mode === "pd" && <p className="cost-config-note">{text.independent}</p>}</div>
      <div className="cost-config-section"><h4>{english ? "Parallelism" : "并行策略"}{mode === "pd" ? ` · ${phase}` : ""}</h4><PlanFields plan={plan} english={english} onChange={(next) => updatePlan({ ...plans, [phase]: next })} /></div>
      <div className="cost-config-section"><h4>{text.analysis}</h4><div className="cost-section-heading"><span className="cost-config-label">{text.compare}</span><div className="cost-segmented"><button type="button" className={comparisonMode === "off" ? "active" : ""} aria-pressed={comparisonMode === "off"} onClick={() => onComparisonModeChange?.("off")}>{text.off}</button><button type="button" className={comparisonMode === "chip" ? "active" : ""} aria-pressed={comparisonMode === "chip"} onClick={() => onComparisonModeChange?.("chip")}>{text.chipCompare}</button><button type="button" className={comparisonMode === "plan" ? "active" : ""} aria-pressed={comparisonMode === "plan"} onClick={() => onComparisonModeChange?.("plan")}>{text.planCompare}</button></div></div>{comparisonMode === "chip" && <label className="cost-config-control">{text.compareGpu}<select value={compareChipId} onChange={(event) => onCompareChipIdChange?.(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label>}{comparisonMode === "plan" && <div className="cost-plan-fields"><label>{text.compareTp}<input type="number" min="1" value={comparePlan.tp} onChange={(event) => onComparePlanChange?.({ ...comparePlan, tp: Math.max(1, Number(event.target.value) || 1) })} /></label><label>{text.compareEp}<input type="number" min="1" value={comparePlan.ep} onChange={(event) => onComparePlanChange?.({ ...comparePlan, ep: Math.max(1, Number(event.target.value) || 1) })} /></label><label>{text.compareAttention}<select value={comparePlan.attnMode} onChange={(event) => onComparePlanChange?.({ ...comparePlan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label></div>}<div className="cost-plan-fields"><label>{text.etaFlops}<input type="number" min="0.1" max="1" step="0.05" value={efficiency.flops} onChange={(event) => onEfficiencyChange?.({ ...efficiency, flops: Math.min(1, Math.max(0.1, Number(event.target.value) || 0.7)) })} /></label><label>{text.etaHbm}<input type="number" min="0.1" max="1" step="0.05" value={efficiency.hbm} onChange={(event) => onEfficiencyChange?.({ ...efficiency, hbm: Math.min(1, Math.max(0.1, Number(event.target.value) || 0.9)) })} /></label><label>{text.etaComm}<input type="number" min="0.1" max="1" step="0.05" value={efficiency.intra_node_comm} onChange={(event) => onEfficiencyChange?.({ ...efficiency, intra_node_comm: Math.min(1, Math.max(0.1, Number(event.target.value) || 0.8)) })} /></label></div></div>
      <div className="cost-config-section"><h4>Cost assumptions</h4><div className="cost-config-grid"><label>KV bytes / element<select value={kvElementBytes} onChange={(event) => setKvElementBytes(Number(event.target.value))}><option value="2">2</option><option value="1">1</option><option value="0.5">0.5</option></select></label><label>Activation peak / GiB<input type="number" min="0" step="0.1" value={activationGiB} onChange={(event) => setActivationGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Runtime / GiB<input type="number" min="0" step="0.1" value={runtimeGiB} onChange={(event) => setRuntimeGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Comm buffer / GiB<input type="number" min="0" step="0.1" value={commBufferGiB} onChange={(event) => setCommBufferGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Weight what-if<select value={weightMode} onChange={(event) => setWeightMode(event.target.value)}><option value="actual">actual / derived</option><option value="2">BF16 / FP16</option><option value="1">FP8 / INT8</option><option value="0.5">INT4</option></select></label></div></div>
    </div>}
    <div className="cost-breakdown">{[["Weights", cost.memory.weightBytes], ["KV", cost.memory.kvBytes], ["KDA state", cost.memory.stateBytes], ["Activation peak", peakCost.memory.activationBytes], ["Runtime", cost.memory.runtimeBytes], ["Communication buffer", cost.memory.commBufferBytes]].map(([label, value]) => <span key={label}><b>{label}</b>{formatBytes(value)}</span>)}</div>
    <div className="cost-metrics"><span>Total VRAM <b>{formatBytes(cost.memory.totalBytes)}</b></span><span>{text.fitCard} <b className={cost.memory.totalBytes <= available ? "fit" : "no-fit"}>{fitText(cost.memory.totalBytes <= available, english)}</b></span><span>Max context <b>{planMaxContext == null ? "-" : planMaxContext.toLocaleString()}</b></span><span>MACs / token <b>{formatMacs(cost.macsPerToken)}</b></span><span>MACs / forward <b>{formatMacs(cost.totalMacs)}</b></span><span>FLOPs / forward <b>{formatMacs(cost.totalFlops)}</b></span><span>Roofline <b>{roofline?.bound || "unknown"}</b></span><span>Communication <b>{formatBytes(communication?.totalBytes)}</b></span></div>
    {projected?.ok && <div className="cost-stages">{projected.stages.map((stage) => <span key={stage.stage}><b>Stage {stage.stage}</b> {formatBytes(stage.weightBytes)} weights · {formatBytes(stage.kvBytes)} KV · {formatBytes(stage.stateBytes || 0)} KDA state</span>)}</div>}
    {mode === "pd" && pd?.ok && <div className="pd-summary-modern"><b>KV + KDA State Transfer</b><span>{formatBytes(pd.aggregateBytes)} total · {formatBytes(pd.perDecodeRankBytes + (pd.perDecodeRankStateBytes || 0))} / Decode rank</span><span>{pd.linkSource}{pd.linkBandwidth ? ` · ${formatRate(pd.linkBandwidth)}` : ""}</span><span>Prefill {text.fit} {fitText(pdFit?.prefill?.fit, english)} · Decode {text.fit} {fitText(pdFit?.decode?.fit, english)}</span></div>}
    {mode === "pd" && pd && !pd.ok && <div className="cost-plan-error">PD plan invalid: {pd.errors.join("; ")}</div>}
    <div className="cost-assumptions">{phase === "prefill" && load.chunked ? `${text.chunkedSummary(load.sequence, Math.min(load.sequence, load.chunkSize))} ` : ""}{cost.weightSource === "derived-quantized" ? `Weights use a ${cost.assumptions.weightBytesPerParameter} B/parameter ${cost.assumptions.quantization} estimate; checkpoint metadata can refine module exceptions. ` : ""}{text.theoretical} {roofline && `Roofline compute ${formatSeconds(roofline.times.compute)} · memory ${formatSeconds(roofline.times.memory)} · comm ${formatSeconds(roofline.times.comm)}.`}</div>
  </section>;
}
