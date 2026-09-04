import { useMemo, useState } from "react";
import { normalizeConfig } from "../structure/config/normalize.js";
import { aggregateCost } from "../cost/aggregate.js";
import { maxContextForStages, projectPdFit, projectPlan } from "../cost/parallel.js";
import { pdKvTransferBytes, planCommunicationBytes } from "../cost/comm.js";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import ManualChipForm from "./ManualChipForm.jsx";

const GIB = 1024 ** 3;
const PLAN_DEFAULT = { tp: 1, pp: 1, ep: 1, dp: 1, attnMode: "tp" };

function formatBytes(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= GIB) return `${(value / GIB).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${Math.round(value)} B`;
}

function formatMacs(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} G`;
  return `${(value / 1e6).toFixed(1)} M`;
}

function formatRate(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)} TB/s`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)} GB/s`;
  return `${(value / 1e6).toFixed(1)} MB/s`;
}

function fitText(value) {
  return value == null ? "未知" : value ? "是" : "否";
}

function PlanFields({ plan, onChange }) {
  const update = (key, value) => onChange({ ...plan, [key]: Math.max(1, Number(value) || 1) });
  return <div className="cost-plan-fields">
    <label>TP<input type="number" min="1" value={plan.tp} onChange={(event) => update("tp", event.target.value)} /></label>
    <label>PP<input type="number" min="1" value={plan.pp} onChange={(event) => update("pp", event.target.value)} /></label>
    <label>EP<input type="number" min="1" value={plan.ep} onChange={(event) => update("ep", event.target.value)} /></label>
    <label>DP<input type="number" min="1" value={plan.dp} onChange={(event) => update("dp", event.target.value)} /></label>
    <label>Attention 并行方式<select value={plan.attnMode} onChange={(event) => onChange({ ...plan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label>
  </div>;
}

export default function CostSummary({ structure, chips = PUBLIC_CHIPS, onAddChip, language = "zh" }) {
  const english = language === "en";
  const text = {
    estimate: english ? "Theoretical cost estimate" : "理论成本估算",
    disclaimer: english ? "Theoretical calculation; not simulation or prediction" : "理论计算，非仿真、非预测",
    expand: english ? "Expand config" : "展开配置",
    collapse: english ? "Collapse config" : "收起配置",
    machine: english ? "Machine node" : "Machine node",
    mode: english ? "Serving mode" : "运行模式",
    centralized: english ? "Centralized" : "集中式（非 PD）",
    pd: english ? "PD disaggregation" : "PD 分离",
    input: english ? "Input tokens / request" : "输入 tokens / request",
    context: english ? "Current context length" : "当前上下文长度",
    independent: english ? "P/D workloads, node counts, and parallel plans are saved independently; only the current phase is shown." : "P/D 负载、节点规模和并行策略独立保存；当前只显示当前阶段结果。",
    theoretical: english ? "Cost is a theoretical estimate; no scheduling, pipeline bubble, or transfer overlap simulation." : "理论计算，不模拟调度、流水线气泡或传输重叠。",
  };
  const [phase, setPhase] = useState("prefill");
  const [mode, setMode] = useState("centralized");
  const [lenses, setLenses] = useState(() => new Set(["vram"]));
  const [machineId, setMachineId] = useState(chips[0]?.id || "");
  const [nodes, setNodes] = useState({ centralized: 1, prefill: 1, decode: 2 });
  const [gpusPerNode, setGpusPerNode] = useState(8);
  const [loads, setLoads] = useState({ prefill: { batch: 1, sequence: 2048, chunked: false, chunkSize: 8192 }, decode: { batch: 1, sequence: 2048 } });
  const [plans, setPlans] = useState({ prefill: PLAN_DEFAULT, decode: PLAN_DEFAULT });
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
  const cost = useMemo(() => {
    if (!structure?.root || !config) return null;
    return aggregateCost({ root: structure.root, config, parameterCount: structure.summary?.parameters_by_dtype, phase, batch: load.batch, sequence: load.sequence, kvBytes: kvElementBytes, activationPeak: activationGiB * GIB, runtimeConst: runtimeGiB * GIB, commBuffer: commBufferGiB * GIB, weightBytesPerParameter: weightMode === "actual" ? undefined : Number(weightMode) });
  }, [structure, config, phase, load, kvElementBytes, activationGiB, runtimeGiB, commBufferGiB, weightMode]);
  const peakCost = useMemo(() => {
    if (!cost || !load.chunked || phase !== "prefill") return cost;
    return aggregateCost({ root: structure.root, config, parameterCount: structure.summary?.parameters_by_dtype, phase, batch: load.batch, sequence: Math.min(load.sequence, load.chunkSize), kvBytes: kvElementBytes, activationPeak: activationGiB * GIB, runtimeConst: runtimeGiB * GIB, commBuffer: commBufferGiB * GIB, weightBytesPerParameter: weightMode === "actual" ? undefined : Number(weightMode) });
  }, [cost, load, phase, structure, config, kvElementBytes, activationGiB, runtimeGiB, commBufferGiB, weightMode]);
  const available = machine?.memory_bytes || 0;
  const projected = useMemo(() => cost && machine ? projectPlan({ root: structure.root, weightBytes: cost.memory.weightBytes, kvBytes: cost.memory.kvBytes, config, plan }) : null, [cost, machine, structure, config, plan]);
  const communication = useMemo(() => cost ? planCommunicationBytes({ root: structure.root, config, plan, batch: load.batch, tokens: phase === "decode" ? 1 : (load.chunked ? Math.min(load.sequence, load.chunkSize) : load.sequence) }) : null, [cost, structure, config, plan, load, phase]);
  const pd = mode === "pd" && cost && machine ? pdKvTransferBytes({ totalKvBytes: cost.memory.kvBytes, config, pdPlan: { prefill_plan: plans.prefill, decode_plan: plans.decode }, prefillChip: machine, decodeChip: machine }) : null;
  const pdFit = mode === "pd" && cost && machine ? projectPdFit({ root: structure.root, weightBytes: cost.memory.weightBytes, kvBytes: cost.memory.kvBytes, config, pdPlan: { prefill_plan: plans.prefill, decode_plan: plans.decode }, prefillChip: machine, decodeChip: machine, activationBytes: peakCost?.memory.activationBytes || 0, runtimeBytes: cost.memory.runtimeBytes, commBufferBytes: cost.memory.commBufferBytes }) : null;
  if (!cost || !machine) return null;
  const currentNodes = mode === "pd" ? nodes[phase] : nodes.centralized;
  const totalGpus = currentNodes * gpusPerNode;
  const requiredGpus = plan.tp * plan.pp * plan.dp;
  const planFitsTopology = requiredGpus <= totalGpus;
  const planMaxContext = projected?.ok ? maxContextForStages(projected.stages, { capacityBytes: available, activationBytes: peakCost.memory.activationBytes, runtimeBytes: cost.memory.runtimeBytes + cost.memory.commBufferBytes, sequence: load.sequence }) : null;
  const updateLoad = (key, value) => setLoads((current) => ({ ...current, [phase]: { ...current[phase], [key]: value } }));
  const toggleLens = (name) => setLenses((current) => { const next = new Set(current); if (name === "none") next.clear(); else next.has(name) ? next.delete(name) : next.add(name); return next; });
  return <section className="cost-summary cost-summary-modern" aria-label={text.estimate}>
    <div className="cost-summary-header"><div><b>{text.estimate}</b><span className="cost-disclaimer">{text.disclaimer}</span></div><button className="cost-expand-button" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? text.collapse : text.expand}</button></div>
    <div className="cost-lens-row"><span>Cost Lens</span>{[["none", "None"], ["vram", "VRAM"], ["compute", "Compute"], ["memory", "Memory"], ["kv", "KV Cache"]].map(([id, label]) => <button type="button" key={id} className={(id === "none" ? lenses.size === 0 : lenses.has(id)) ? "active" : ""} aria-pressed={id === "none" ? lenses.size === 0 : lenses.has(id)} onClick={() => toggleLens(id)}>{label}</button>)}</div>
    <div className="cost-machine-summary"><span><b>{machine.name}</b> · {currentNodes} node{currentNodes === 1 ? "" : "s"} · {totalGpus} GPU</span><span>{formatBytes(machine.memory_bytes)} / card</span><span className={planFitsTopology ? "fit" : "no-fit"}>{planFitsTopology ? "plan valid" : `needs ${requiredGpus} GPUs`}</span></div>
    {expanded && <div className="cost-config-modern">
      <div className="cost-config-section"><h4>Machine node</h4><div className="cost-config-grid"><label>GPU / 芯片<select value={machine.id} onChange={(event) => setMachineId(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label>{mode === "centralized" && <label>Nodes<input type="number" min="1" value={nodes.centralized} onChange={(event) => setNodes((current) => ({ ...current, centralized: Math.max(1, Number(event.target.value) || 1) }))} /></label>}<label>GPU / Node<input type="number" min="1" value={gpusPerNode} onChange={(event) => setGpusPerNode(Math.max(1, Number(event.target.value) || 1))} /></label><div className="cost-machine-spec">{formatBytes(machine.memory_bytes)} / card · {formatBytes(machine.memory_bandwidth)} HBM · {Object.entries(machine.peak_flops || {}).map(([dtype, value]) => `${dtype.toUpperCase()} ${formatMacs(value)}`).join(" · ") || "算力未知"}</div></div><ManualChipForm onAdd={(chip) => { onAddChip?.(chip); setMachineId(chip.id); }} /></div>
      <div className="cost-config-section"><div className="cost-section-heading"><h4>{text.mode}</h4><div className="cost-segmented"><button type="button" className={mode === "centralized" ? "active" : ""} onClick={() => setMode("centralized")}>{text.centralized}</button><button type="button" className={mode === "pd" ? "active" : ""} onClick={() => setMode("pd")}>{text.pd}</button></div></div><div className="cost-phase-switch"><button type="button" className={phase === "prefill" ? "active" : ""} onClick={() => setPhase("prefill")}>Prefill</button><button type="button" className={phase === "decode" ? "active" : ""} onClick={() => setPhase("decode")}>Decode</button></div><div className="cost-config-grid"><label>{phase === "prefill" ? text.input : text.context}<input type="number" min="1" value={load.sequence} onChange={(event) => updateLoad("sequence", Math.max(1, Number(event.target.value) || 1))} /></label><label>Batch size<input type="number" min="1" value={load.batch} onChange={(event) => updateLoad("batch", Math.max(1, Number(event.target.value) || 1))} /></label>{mode === "pd" && <label>Nodes / {phase}<input type="number" min="1" value={nodes[phase]} onChange={(event) => setNodes((current) => ({ ...current, [phase]: Math.max(1, Number(event.target.value) || 1) }))} /></label>}{phase === "prefill" && <label className="cost-check"><input type="checkbox" checked={load.chunked} onChange={(event) => updateLoad("chunked", event.target.checked)} /> Chunked Prefill</label>}{phase === "prefill" && load.chunked && <label>Prefill chunk size<input type="number" min="1" value={load.chunkSize} onChange={(event) => updateLoad("chunkSize", Math.max(1, Number(event.target.value) || 1))} /></label>}</div>{mode === "pd" && <p className="cost-config-note">{text.independent}</p>}</div>
      <div className="cost-config-section"><h4>Parallelism · {phase}</h4><PlanFields plan={plan} onChange={(next) => setPlans((current) => ({ ...current, [phase]: next }))} /></div>
      <div className="cost-config-section"><h4>Cost assumptions</h4><div className="cost-config-grid"><label>KV bytes / element<select value={kvElementBytes} onChange={(event) => setKvElementBytes(Number(event.target.value))}><option value="2">2</option><option value="1">1</option><option value="0.5">0.5</option></select></label><label>Activation peak / GiB<input type="number" min="0" step="0.1" value={activationGiB} onChange={(event) => setActivationGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Runtime / GiB<input type="number" min="0" step="0.1" value={runtimeGiB} onChange={(event) => setRuntimeGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Comm buffer / GiB<input type="number" min="0" step="0.1" value={commBufferGiB} onChange={(event) => setCommBufferGiB(Math.max(0, Number(event.target.value) || 0))} /></label><label>Weight what-if<select value={weightMode} onChange={(event) => setWeightMode(event.target.value)}><option value="actual">actual / derived</option><option value="2">BF16 / FP16</option><option value="1">FP8 / INT8</option><option value="0.5">INT4</option></select></label></div></div>
    </div>}
    <div className="cost-breakdown">{[["Weights", cost.memory.weightBytes], ["KV", cost.memory.kvBytes], ["Activation peak", peakCost.memory.activationBytes], ["Runtime", cost.memory.runtimeBytes], ["Communication buffer", cost.memory.commBufferBytes]].map(([label, value]) => <span key={label}><b>{label}</b>{formatBytes(value)}</span>)}</div>
    <div className="cost-metrics"><span>Total VRAM <b>{formatBytes(cost.memory.totalBytes)}</b></span><span>Fit / card <b className={cost.memory.totalBytes <= available ? "fit" : "no-fit"}>{cost.memory.totalBytes <= available ? "yes" : "no"}</b></span><span>Max context <b>{planMaxContext == null ? "-" : planMaxContext.toLocaleString()}</b></span><span>MACs <b>{formatMacs(cost.totalMacs)}</b></span><span>Communication <b>{formatBytes(communication?.totalBytes)}</b></span></div>
    {projected?.ok && <div className="cost-stages">{projected.stages.map((stage) => <span key={stage.stage}><b>Stage {stage.stage}</b> {formatBytes(stage.weightBytes)} weights · {formatBytes(stage.kvBytes)} KV</span>)}</div>}
    {mode === "pd" && pd?.ok && <div className="pd-summary-modern"><b>KV Transfer</b><span>{formatBytes(pd.aggregateBytes)} total · {formatBytes(pd.perDecodeRankBytes)} / Decode rank</span><span>{pd.linkSource}{pd.linkBandwidth ? ` · ${formatRate(pd.linkBandwidth)}` : ""}</span><span>Prefill fit {fitText(pdFit?.prefill?.fit)} · Decode fit {fitText(pdFit?.decode?.fit)}</span></div>}
    {mode === "pd" && pd && !pd.ok && <div className="cost-plan-error">PD plan invalid: {pd.errors.join("; ")}</div>}
    <div className="cost-assumptions">{phase === "prefill" && load.chunked ? `Total uses ${load.sequence} input tokens; peak uses ${Math.min(load.sequence, load.chunkSize)} tokens per chunk. ` : ""}{text.theoretical}</div>
  </section>;
}
