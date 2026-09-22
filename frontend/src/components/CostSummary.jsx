import { useEffect, useMemo, useState } from "react";
import { normalizeConfig } from "../structure/config/normalize.js";
import { aggregateCost } from "../cost/aggregate.js";
import { walkStructure } from "../cost/traverse.js";
import { bytesPerDtype } from "../cost/memory.js";
import { resolveFrameworkPlan } from "../cost/sharding.js";
import { maxContextForStages, planFitsCard, projectPdFit, projectPlan } from "../cost/parallel.js";
import { pdKvTransferBytes, planCommunicationBytes } from "../cost/comm.js";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";
import ManualChipForm from "./ManualChipForm.jsx";
import NumberInput from "./NumberInput.jsx";
import { DEFAULT_COMPARE_PLAN, DEFAULT_LOADS, DEFAULT_NODES, DEFAULT_PLAN } from "../cost/defaults.js";
import { DEFAULT_EFFICIENCY } from "../cost/efficiency.js";
import { classifyRoofline } from "../cost/roofline.js";
import { chipRates } from "../cost/chips/rates.js";
import { formatBytes, formatMacs, formatRate, formatSeconds } from "../formatters.js";
import { costByFormulaGroup, costSummaryModel, etaDisclosureModel } from "../cost/ui.js";
import { formatIssues, t } from "../i18n/format.js";

const FLOPS_ORDER = ["fp32", "fp16", "bf16", "fp8", "int8"];

function formatPeakFlops(peakFlops, unknownLabel) {
  const values = FLOPS_ORDER
    .filter((dtype) => Number.isFinite(peakFlops?.[dtype]) && peakFlops[dtype] > 0)
    .map((dtype) => `${dtype.toUpperCase()} ${formatMacs(peakFlops[dtype])}`);
  return values.join(" · ") || unknownLabel;
}

function formatHardware(hardware, sfuOps) {
  if (!hardware) return "";
  const values = [
    hardware.architecture,
    hardware.memory_type,
    Number.isFinite(hardware.sm_count) ? `${hardware.sm_count} SM` : null,
    Number.isFinite(hardware.cuda_cores) ? `${hardware.cuda_cores.toLocaleString("en-US")} CUDA` : null,
    Number.isFinite(hardware.tensor_cores) ? `${hardware.tensor_cores.toLocaleString("en-US")} Tensor` : null,
    Number.isFinite(sfuOps) ? `SFU ${formatMacs(sfuOps)}` : null,
    Number.isFinite(hardware.tdp_watts) ? `TDP ${hardware.tdp_watts}W` : null,
  ].filter(Boolean);
  return values.join(" · ");
}

function fitText(value, language = "zh") {
  return value == null ? t(language, "cost.unknown") : value ? t(language, "cost.yes") : t(language, "cost.no");
}

function ConfigSection({ id, title, open, onToggle, children }) {
  return (
    <section className={`cost-config-section${open ? " is-open" : ""}`} data-section={id}>
      <button type="button" className="cost-config-section-toggle" aria-expanded={open} onClick={onToggle}>
        <span className="cost-config-section-title">{title}</span>
        <span aria-hidden="true">{open ? "−" : "+"}</span>
      </button>
      <div className="cost-config-section-body" hidden={!open}>{children}</div>
    </section>
  );
}

const UNKNOWN_FIELD_LABELS = {
  runtimeWorkspace: ["runtime workspace", "运行时 workspace"],
  allocatorPadding: ["allocator alignment", "分配器对齐"],
  backendCacheLayout: ["backend cache layout", "backend cache layout"],
  dsparkBackendPackingAndPageHeadroom: ["DSpark page packing/reserve", "DSpark 页打包/保留槽"],
  compressedStateAndBackendWindowAllocation: ["compressed-state/window allocation", "压缩状态/窗口分配"],
  speculativeStateScratch: ["speculative state scratch", "投机状态 scratch"],
};

function unknownFieldLabel(field, english) {
  const labels = UNKNOWN_FIELD_LABELS[field];
  if (labels) return labels[english ? 0 : 1];
  if (field.startsWith("pool:")) return english ? `${field} allocation` : `${field} 分配`;
  return field;
}

function PlanFields({ plan, onChange, english }) {
  const setField = (key, value) => onChange({ ...plan, [key]: value });
  // P6（协议 Q9）：moe_tp/moe_ep/vocab_parallel 进 UI —— 协议层早已支持并有
  // 校验，但没有用户入口（sharding_matrix §四"一处不落即死功能"的第四处）。
  // world_size 是派生量（tp×pp×dp），只读显示。
  // 数字输入采用 draft + commit-on-blur：编辑期可清空/中间态，失焦或回车再夹到最小值 1。
  const worldSize = (plan.tp ?? 1) * (plan.pp ?? 1) * (plan.dp ?? 1);
  return <div className="cost-plan-fields">
    <label>TP<NumberInput min={1} fallback={1} value={plan.tp} onCommit={(value) => setField("tp", value)} /></label>
    <label>PP<NumberInput min={1} fallback={1} value={plan.pp} onCommit={(value) => setField("pp", value)} /></label>
    <label>EP<NumberInput min={1} fallback={1} value={plan.ep} onCommit={(value) => setField("ep", value)} /></label>
    <label>DP<NumberInput min={1} fallback={1} value={plan.dp} onCommit={(value) => setField("dp", value)} /></label>
    <label>MoE TP<NumberInput min={1} allowEmpty value={plan.moe_tp} placeholder={String(plan.tp ?? 1)} onCommit={(value) => setField("moe_tp", value)} /></label>
    <label>MoE EP<NumberInput min={1} allowEmpty value={plan.moe_ep} placeholder={String(plan.ep ?? 1)} onCommit={(value) => setField("moe_ep", value)} /></label>
    <label>{english ? "Attention parallelism" : "Attention 并行方式"}<select value={plan.attnMode} onChange={(event) => onChange({ ...plan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label>
    <label className="cost-check cost-vocab-field">{english ? "Vocab parallel" : "词表并行"}<input type="checkbox" checked={plan.vocab_parallel !== false} onChange={(event) => onChange({ ...plan, vocab_parallel: event.target.checked })} /></label>
    <label className="world-size-field" title={english ? "Total distributed processes (ranks): TP × PP × DP. EP is not multiplied again." : "分布式进程总数（ranks）：TP × PP × DP；EP 不重复相乘。"}><span>World size</span><output>{worldSize}</output><small>TP × PP × DP</small></label>
  </div>;
}

export default function CostSummary({ structure, chips = PUBLIC_CHIPS, onAddChip, language = "zh", onFitStatusChange, lenses: controlledLenses, onLensesChange, phase: controlledPhase, onPhaseChange, mode: controlledMode, onModeChange, plans: controlledPlans, onPlansChange, nodes: controlledNodes, onNodesChange, gpusPerNode: controlledGpusPerNode, onGpusPerNodeChange, machineId: controlledMachineId, onMachineIdChange, loads: controlledLoads, onLoadsChange, comparisonMode = "off", onComparisonModeChange, compareChipId = "", onCompareChipIdChange, comparePlan = DEFAULT_COMPARE_PLAN, onComparePlanChange, efficiency = DEFAULT_EFFICIENCY, onEfficiencyChange, frameworkProfile = "neutral", deploymentRecommendation, deploymentManual = false, onResetDeployment }) {
  const english = language === "en";
  const text = {
    estimate: t(language, "cost.estimate"),
    disclaimer: t(language, "cost.disclaimer"),
    expand: t(language, "cost.expand"),
    collapse: t(language, "cost.collapse"),
    machine: t(language, "cost.machine"),
    gpu: t(language, "cost.gpu"),
    gpuNode: t(language, "cost.gpuNode"),
    unknownFlops: t(language, "cost.unknownFlops"),
    mode: t(language, "cost.mode"),
    centralized: t(language, "cost.centralized"),
    pd: t(language, "cost.pd"),
    node: t(language, "cost.node"),
    nodes: t(language, "cost.nodes"),
    fit: t(language, "cost.fit"),
    fitCard: t(language, "cost.fitCard"),
    needsGpus: (count) => t(language, "cost.needsGpus", { count }),
    memoryNoFit: t(language, "cost.memoryNoFit"),
    planValid: t(language, "cost.planValid"),
    planInvalid: t(language, "cost.planInvalid"),
    input: t(language, "cost.input"),
    context: t(language, "cost.context"),
    independent: t(language, "cost.independent"),
    theoretical: t(language, "cost.theoretical"),
    chunkedSummary: (total, peak) => t(language, "cost.chunkedSummary", { total, peak }),
    analysis: t(language, "cost.analysis"),
    compare: t(language, "cost.compare"),
    off: t(language, "cost.off"),
    chipCompare: t(language, "cost.chipCompare"),
    planCompare: t(language, "cost.planCompare"),
    compareGpu: t(language, "cost.compareGpu"),
    compareTp: t(language, "cost.compareTp"),
    compareEp: t(language, "cost.compareEp"),
    compareAttention: t(language, "cost.compareAttention"),
    etaFlops: "ηF",
    etaHbm: "ηHBM",
    etaComm: "ηComm",
  };
  // 集中式只有一个部署工作点；即使用户刚从 PD 的 Decode 切回来，
  // 这里也回到 Prefill，避免 Chunk Prefill 控件被错误隐藏。
  const phase = (controlledMode ?? "centralized") === "centralized" ? "prefill" : (controlledPhase ?? "prefill");
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
  // P10（协议 Q7①）：interNode 显式开关，默认 false（保守取 intra-node 费率）。
  // 开启后 roofline 的通信时间按 chips/rates.js 的 inter_node 行计。
  const [interNode, setInterNode] = useState(false);
  const [weightMode, setWeightMode] = useState("actual");
  const [speculativeDraftTokens, setSpeculativeDraftTokens] = useState(0);
  const [speculativeMaxRequests, setSpeculativeMaxRequests] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [openConfigSections, setOpenConfigSections] = useState({
    machine: true,
    mode: true,
    parallelism: true,
    analysis: true,
    assumptions: false,
  });
  const toggleConfigSection = (section) => setOpenConfigSections((current) => ({ ...current, [section]: !current[section] }));
  const config = useMemo(() => structure?.extra_config ? normalizeConfig(structure.extra_config) : null, [structure]);
  // 数据驱动检测：图里存在亚字节(FP4)KV dtype 的叶 → 该模型 KV 走设计 fp4 口径。
  // 不硬编码家族名（§8.1）：只看 cache_kv_dtype 的字节宽度。
  const hasSubByteKv = useMemo(() => {
    if (!structure?.graph) return false;
    let found = false;
    walkStructure(structure.graph, ({ node }) => {
      const d = node?.attributes?.cache_kv_dtype;
      if (d && bytesPerDtype(d, 2) < 1) found = true;
    });
    return found;
  }, [structure]);
  const machine = chips.find((chip) => chip.id === machineId) || chips[0];
  const load = loads[phase];
  // C3b：框架 profile 只选择已实现的公式语义，不注入实测常数。
  const plan = resolveFrameworkPlan(plans[phase], frameworkProfile, config || {});
  const costFor = (targetPhase) => {
    const targetLoad = loads[targetPhase] || DEFAULT_LOADS[targetPhase];
    if (!structure?.graph || !config) return null;
    const speculative = frameworkProfile === "sglang"
      ? {
        enabled: speculativeDraftTokens > 0 || speculativeMaxRequests > 0,
        draftTokens: speculativeDraftTokens,
        stateSlots: speculativeMaxRequests,
        disaggregationMode: mode === "pd" ? targetPhase : "null",
      }
      : {};
    return aggregateCost({ graph: structure.graph, config, parameterCount: structure.summary?.parameters_by_dtype, phase: targetPhase, batch: targetLoad.batch, sequence: targetLoad.sequence, kvBytes: kvElementBytes, visionTokens: targetLoad.visionTokens ?? 1024, weightBytesPerParameter: weightMode === "actual" ? undefined : Number(weightMode), frameworkProfile, speculative });
  };
  const phaseCosts = useMemo(() => ({ prefill: costFor("prefill"), decode: costFor("decode") }), [structure, config, loads, mode, kvElementBytes, weightMode, frameworkProfile, speculativeDraftTokens, speculativeMaxRequests]);
  const cost = phaseCosts[phase];
  // 草稿 KV 已由 framework profile 选择 cache pool ownership 后进入同一份
  // accounting；这里不再把 UI 展示值额外加回总显存。
  const draftKvBytes = cost?.memory?.draftKvBytes || 0;
  const accounting = cost?.memory.accounting;
  const draftWeight = accounting?.draft.weightBytes || 0;
  const available = machine?.memory_bytes || 0;
  const projected = useMemo(() => cost && machine ? projectPlan({ graph: structure.graph, accounting, config, plan }) : null, [cost, machine, structure, accounting, config, plan]);
  const communication = useMemo(() => cost ? planCommunicationBytes({ graph: structure.graph, config, plan, batch: load.batch, tokens: phase === "decode" ? 1 : (load.chunked ? Math.min(load.sequence, load.chunkSize) : load.sequence), frameworkProfile }) : null, [cost, structure, config, plan, load, phase, frameworkProfile]);
  const roofline = useMemo(() => cost && machine ? classifyRoofline({
    // M11-P0-5：访存侧切到 counts.bytes 通道（逐算子一阶流量：actIn/actOut
    // 为聚合 actions 的扁平字段）。weights 仍以 memory 侧为权威（what-if
    // 覆盖生效）；counts.bytes.weights 与 natural weights 的统一留 M11.5。
    actions: cost.actions && {
      ...cost.actions,
      matrixTf32: cost.actions.matrixTf32 ?? cost.actions.computeDtypes?.tf32 ?? 0,
      bytes: {
        weights: cost.memory.weightBytes,
        actIn: cost.actions.actIn,
        actOut: cost.actions.actOut,
        kvRead: cost.actions.kvRead,
        indexRead: cost.actions.indexRead,
      },
      commBytes: communication?.totalBytes || 0,
    },
  }, machine, { dtype: "bf16", efficiency, interNode }) : null, [cost, machine, communication, efficiency, interNode]);
  // P10：per-stage roofline（访存路）。stage 只有字节侧分解（weights/KV/state），
  // 无逐 stage 动作向量 → 只输出 HBM 时间并保持口径标注；计算路 per-stage 待
  // stage 级 actions 落地后再扩展（登记于 protocol §二点五）。
  const stageRates = useMemo(() => machine ? chipRates(machine, { dtype: "bf16", efficiency }) : null, [machine, efficiency]);
  const pd = useMemo(() => mode === "pd" && phaseCosts.prefill && machine ? pdKvTransferBytes({ totalKvBytes: phaseCosts.prefill.memory.kvBytes, totalStateBytes: phaseCosts.prefill.memory.stateBytes, config, pdPlan: { prefill_plan: resolveFrameworkPlan(plans.prefill, frameworkProfile, config), decode_plan: resolveFrameworkPlan(plans.decode, frameworkProfile, config) }, prefillChip: machine, decodeChip: machine }) : null, [mode, phaseCosts, machine, config, plans, frameworkProfile]);
  const pdFit = useMemo(() => mode === "pd" && phaseCosts.prefill && phaseCosts.decode && machine ? projectPdFit({ graph: structure.graph, prefillAccounting: phaseCosts.prefill.memory.accounting, decodeAccounting: phaseCosts.decode.memory.accounting, config, pdPlan: { prefill_plan: plans.prefill, decode_plan: plans.decode }, prefillChip: machine, decodeChip: machine }) : null, [mode, phaseCosts, machine, structure, config, plans]);
  const summary = useMemo(() => costSummaryModel(cost, roofline, { english }), [cost, roofline, english]);
  // Cost Lens 按 FORMULAS.group 分栏：成本花在哪类算子（gemm/attention/moe/…）。
  const domainBreakdown = useMemo(() => costByFormulaGroup(cost), [cost]);
  // M11-P1-3：η 披露——vector/SFU 路固定 1.0，滑块不作用于它（见 ui.js 注释）
  const etaNote = etaDisclosureModel({ english });
  const currentNodes = mode === "pd" ? nodes[phase] : nodes.centralized;
  const totalGpus = currentNodes * gpusPerNode;
  const requiredGpus = plan.tp * plan.pp * plan.dp;
  const planFitsTopology = requiredGpus <= totalGpus;
  // M11-P1-6：plan 无效时 fit 语义是"未知"而非"显存不足"——此前无效 plan
  // 会把 planFitsMemory 压成 false 并谎报"显存不足"，且错误摘要从不上界面。
  const planInvalid = projected != null && projected.ok === false;
  const planFitsMemory = mode === "pd"
    ? pdFit?.[phase]?.fit
    : planFitsCard(projected, available);
  const planStatus = !planFitsTopology ? text.needsGpus(requiredGpus) : planInvalid ? text.planInvalid : planFitsMemory === false ? text.memoryNoFit : text.planValid;
  useEffect(() => {
    if (!cost || !machine) {
      onFitStatusChange?.(null);
      return;
    }
    onFitStatusChange?.({ fit: planFitsTopology && planFitsMemory === true, known: planFitsMemory != null, status: planStatus, phaseFits: mode === "pd" ? { prefill: pdFit?.prefill?.fit, decode: pdFit?.decode?.fit } : null });
  }, [cost, machine, mode, onFitStatusChange, pdFit, planFitsTopology, planFitsMemory, planStatus]);
  if (!cost || !machine) return null;
  const planMaxContext = projected?.ok ? maxContextForStages(projected.stages, { capacityBytes: available, sequence: load.sequence }) : null;
  const updateLoad = (key, value) => updateLoads({ ...loads, [phase]: { ...loads[phase], [key]: value } });
  const toggleLens = (name) => {
    const next = new Set(lenses);
    if (name === "none") next.clear();
    else next.has(name) ? next.delete(name) : next.add(name);
    onLensesChange?.(next);
  };
  const chunkedNote = phase === 'prefill' && load.chunked
    ? `${text.chunkedSummary(load.sequence, Math.min(load.sequence, load.chunkSize))} `
    : '';
  const weightNote = cost.weightSource === 'derived-quantized'
    ? t(language, "cost.weightsEstimate", {
      bytes: cost.assumptions.weightBytesPerParameter,
      quantization: cost.assumptions.quantization,
    })
    : '';
  const unknownLabel = t(language, "cost.na");
  const rooflineTimes = summary.times.map(
    (entry) => `${entry.label} ${entry.known ? formatSeconds(entry.seconds) : unknownLabel}`,
  ).join(" · ");
  const boundLabel = t(language, "cost.lowerBound");
  const missingSep = english ? ", " : "、";
  const missingNote = t(language, "cost.missingNote", { fields: summary.missingLabels.join(missingSep) });
  const unknownComputeNote = ` ${t(language, "cost.unknownCompute", { count: summary.unknownComputeCount })}`;
  const unknownFields = accounting.evidence.unknownFields
    .map((field) => unknownFieldLabel(field, english));
  const unknownFieldsText = unknownFields.join(missingSep);
  return <section className="cost-summary cost-summary-modern" aria-label={text.estimate} data-framework={accounting.framework}>
    <div className="cost-summary-header"><div><b>{text.estimate}</b><span className="cost-disclaimer">{text.disclaimer}</span></div><button className="cost-expand-button" type="button" aria-expanded={expanded} onClick={() => setExpanded((value) => !value)}>{expanded ? text.collapse : text.expand}</button></div>
    <div className="cost-lens-row"><span>Cost Lens</span>{[["none", "None"], ["vram", "VRAM"], ["compute", "Compute"], ["memory", "Memory"], ["kv", "KV Cache"]].map(([id, label]) => <button type="button" key={id} className={(id === "none" ? lenses.size === 0 : lenses.has(id)) ? "active" : ""} aria-pressed={id === "none" ? lenses.size === 0 : lenses.has(id)} onClick={() => toggleLens(id)}>{label}</button>)}</div>
    <div className="cost-machine-summary"><span><b>{machine.name}</b> · {currentNodes} {currentNodes === 1 ? text.node : text.nodes} · {totalGpus} GPU</span><span>{t(language, "cost.usedGpus", { count: requiredGpus })}</span><span>{formatBytes(machine.memory_bytes)} / card</span><span className={planFitsTopology && !planInvalid && planFitsMemory !== false ? "fit" : "no-fit"}>{planStatus}</span></div>
    {mode === "pd" && <div className="pd-deployment-summary"><span><b>Prefill</b> · {nodes.prefill} {nodes.prefill === 1 ? text.node : text.nodes} × {gpusPerNode} GPU · TP{plans.prefill.tp} / PP{plans.prefill.pp} / EP{plans.prefill.ep} / DP{plans.prefill.dp} · {text.fit} {fitText(pdFit?.prefill?.fit, language)}</span><span><b>Decode</b> · {nodes.decode} {nodes.decode === 1 ? text.node : text.nodes} × {gpusPerNode} GPU · TP{plans.decode.tp} / PP{plans.decode.pp} / EP{plans.decode.ep} / DP{plans.decode.dp} · {text.fit} {fitText(pdFit?.decode?.fit, language)}</span></div>}
    {expanded && <div className="cost-config-modern">
      <ConfigSection id="machine" open={openConfigSections.machine} onToggle={() => toggleConfigSection("machine")} title={text.machine}><div className="cost-config-grid"><label className="cost-gpu-field">{text.gpu}<span className="cost-gpu-select-row"><select value={machine.id} onChange={(event) => changeMachine(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select><ManualChipForm language={language} onAdd={(chip) => { onAddChip?.(chip); changeMachine(chip.id); }} /></span></label>{mode === "centralized" && <label>{t(language, "cost.nodesLabel")}<NumberInput min={1} fallback={1} value={nodes.centralized} onCommit={(value) => updateNodes({ ...nodes, centralized: value })} /></label>}<label>{text.gpuNode}<NumberInput min={1} fallback={1} value={gpusPerNode} onCommit={(value) => updateGpusPerNode(value)} /></label><div className="cost-machine-spec">{formatBytes(machine.memory_bytes)} / card · {formatBytes(machine.memory_bandwidth)} HBM · {formatPeakFlops(machine.peak_flops, text.unknownFlops)}{formatHardware(machine.hardware, machine.sfu_ops) && <><br />{formatHardware(machine.hardware, machine.sfu_ops)}</>}</div></div></ConfigSection>
      <ConfigSection id="mode" open={openConfigSections.mode} onToggle={() => toggleConfigSection("mode")} title={text.mode}><div className="cost-section-heading"><div className="cost-segmented"><button type="button" className={mode === "centralized" ? "active" : ""} aria-pressed={mode === "centralized"} onClick={() => changeMode("centralized")}>{text.centralized}</button><button type="button" className={mode === "pd" ? "active" : ""} aria-pressed={mode === "pd"} onClick={() => changeMode("pd")}>{text.pd}</button></div></div>{mode === "pd" && <div className="cost-phase-switch"><button type="button" className={phase === "prefill" ? "active" : ""} aria-pressed={phase === "prefill"} onClick={() => changePhase("prefill")}>Prefill</button><button type="button" className={phase === "decode" ? "active" : ""} aria-pressed={phase === "decode"} onClick={() => changePhase("decode")}>Decode</button></div>}<div className="cost-config-grid"><label>{phase === "prefill" ? text.input : text.context}<NumberInput min={1} fallback={1} value={load.sequence} onCommit={(value) => updateLoad("sequence", value)} /></label><label>{t(language, "cost.batchSize")}<NumberInput min={1} fallback={1} value={load.batch} onCommit={(value) => updateLoad("batch", value)} /></label>{structure?.summary?.vision_layers != null && <label>{t(language, "cost.visualTokens")}<NumberInput min={1} fallback={1024} value={load.visionTokens ?? 1024} onCommit={(value) => updateLoad("visionTokens", value)} /></label>}{mode === "pd" && <label>{t(language, "cost.nodesPerPhase", { phase })}<NumberInput min={1} fallback={1} value={nodes[phase]} onCommit={(value) => updateNodes({ ...nodes, [phase]: value })} /></label>}{phase === "prefill" && <label className="cost-check"><input type="checkbox" checked={load.chunked} onChange={(event) => updateLoad("chunked", event.target.checked)} /> Chunked Prefill</label>}{phase === "prefill" && load.chunked && <label>Prefill chunk size<NumberInput min={1} fallback={8192} value={load.chunkSize} onCommit={(value) => updateLoad("chunkSize", value)} /></label>}</div>{mode === "pd" && <p className="cost-config-note">{text.independent}</p>}</ConfigSection>
      <ConfigSection id="parallelism" open={openConfigSections.parallelism} onToggle={() => toggleConfigSection("parallelism")} title={`${t(language, "cost.parallelism")}${mode === "pd" ? ` · ${phase}` : ""}`}>{deploymentRecommendation && <div className="cost-default-deployment" data-mode={deploymentManual ? "manual" : "auto"} data-cards={deploymentRecommendation.cards} data-fit={String(deploymentRecommendation.fit)}>
        <p className="cost-config-note" data-testid="parallel-default-note">{t(language, deploymentRecommendation.fit == null ? "cost.parallelDefaultUnknown" : deploymentRecommendation.fit === false ? "cost.parallelDefaultNoFit" : "cost.parallelDefault", { cards: deploymentRecommendation.cards })}</p>
        <p className="cost-config-note">{t(language, "cost.parallelDefaultBasis")}</p>
        <p className="cost-config-note">{t(language, deploymentManual ? "cost.parallelManual" : "cost.parallelAuto")}</p>
        {deploymentManual && onResetDeployment && <button type="button" onClick={onResetDeployment}>{t(language, "cost.restoreDefault")}</button>}
      </div>}<PlanFields plan={plans[phase]} english={english} onChange={(next) => updatePlan({ ...plans, [phase]: next })} />{frameworkProfile === "sglang" && config.sharedExperts > 0 && <label className="cost-check"><input type="checkbox" checked={(plans[phase].enforceSharedExpertsFusion ?? plans[phase].enforce_shared_experts_fusion ?? false) === true} onChange={(event) => updatePlan({ ...plans, [phase]: { ...plans[phase], enforceSharedExpertsFusion: event.target.checked } })} />{english ? "Shared-expert fusion (opt-in)" : "共享专家融合（显式启用）"}</label>}</ConfigSection>
      <ConfigSection id="analysis" open={openConfigSections.analysis} onToggle={() => toggleConfigSection("analysis")} title={text.analysis}><div className="cost-section-heading"><span className="cost-config-label">{text.compare}</span><div className="cost-segmented"><button type="button" className={comparisonMode === "off" ? "active" : ""} aria-pressed={comparisonMode === "off"} onClick={() => onComparisonModeChange?.("off")}>{text.off}</button><button type="button" className={comparisonMode === "chip" ? "active" : ""} aria-pressed={comparisonMode === "chip"} onClick={() => onComparisonModeChange?.("chip")}>{text.chipCompare}</button><button type="button" className={comparisonMode === "plan" ? "active" : ""} aria-pressed={comparisonMode === "plan"} onClick={() => onComparisonModeChange?.("plan")}>{text.planCompare}</button></div></div>{comparisonMode === "chip" && <label className="cost-config-control">{text.compareGpu}<select value={compareChipId} onChange={(event) => onCompareChipIdChange?.(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label>}{comparisonMode === "plan" && <div className="cost-plan-fields"><label>{text.compareTp}<NumberInput min={1} fallback={1} value={comparePlan.tp} onCommit={(value) => onComparePlanChange?.({ ...comparePlan, tp: value })} /></label><label>{text.compareEp}<NumberInput min={1} fallback={1} value={comparePlan.ep} onCommit={(value) => onComparePlanChange?.({ ...comparePlan, ep: value })} /></label><label>{text.compareAttention}<select value={comparePlan.attnMode} onChange={(event) => onComparePlanChange?.({ ...comparePlan, attnMode: event.target.value })}><option value="tp">TP</option><option value="dp">DP</option></select></label></div>}<div className="cost-plan-fields"><label>{text.etaFlops}<NumberInput min={0.1} max={1} step="0.05" fallback={0.7} value={efficiency.flops} onCommit={(value) => onEfficiencyChange?.({ ...efficiency, flops: value })} /></label><label>{text.etaHbm}<NumberInput min={0.1} max={1} step="0.05" fallback={0.9} value={efficiency.hbm} onCommit={(value) => onEfficiencyChange?.({ ...efficiency, hbm: value })} /></label><label>{text.etaComm}<NumberInput min={0.1} max={1} step="0.05" fallback={0.8} value={efficiency.intra_node_comm} onCommit={(value) => onEfficiencyChange?.({ ...efficiency, intra_node_comm: value })} /></label><span className="cost-eta-note" title={etaNote.detail}>{etaNote.short}</span></div></ConfigSection>
      <ConfigSection id="assumptions" open={openConfigSections.assumptions} onToggle={() => toggleConfigSection("assumptions")} title={t(language, "cost.assumptions")}><div className="cost-config-grid"><label title={t(language, "cost.kvBytesPerElementHelp")}>{t(language, "cost.defaultKvBytesPerElement")}<select value={kvElementBytes} onChange={(event) => setKvElementBytes(Number(event.target.value))}><option value="2">2</option><option value="1">1</option><option value="0.5">0.5</option></select></label><label className="cost-check"><input type="checkbox" checked={interNode} onChange={(event) => setInterNode(event.target.checked)} /> {t(language, "cost.interNodeLink")}</label><label>{t(language, "cost.weightWhatIf")}<select value={weightMode} onChange={(event) => setWeightMode(event.target.value)}><option value="actual">actual / derived</option><option value="2">BF16 / FP16</option><option value="1">FP8 / INT8</option><option value="0.5">INT4</option></select></label>{frameworkProfile === "sglang" && <><label title={t(language, "cost.speculativeHelp")}>{t(language, "cost.speculativeDraftTokens")}<NumberInput min={0} fallback={0} value={speculativeDraftTokens} onCommit={setSpeculativeDraftTokens} /></label><label title={t(language, "cost.speculativeHelp")}>{t(language, "cost.speculativeMaxRequests")}<NumberInput min={0} fallback={0} value={speculativeMaxRequests} onCommit={setSpeculativeMaxRequests} /></label></>}</div><p className="cost-config-note">{t(language, "cost.kvBytesPerElementHelp")}</p>{frameworkProfile === "sglang" && <p className="cost-config-note">{t(language, "cost.speculativeHelp")}</p>}</ConfigSection>
    </div>}
    {projected && !projected.ok && <div className="cost-plan-error">{t(language, "cost.planInvalidProjection", { errors: formatIssues(language, projected.errors) })}</div>}
    {mode === "pd" && pd && !pd.ok && <div className="cost-plan-error">{t(language, "cost.pdPlanInvalid", { errors: formatIssues(language, pd.errors) })}</div>}
    {hasSubByteKv && (
      <div className="cost-note" role="note">{language === "en"
        ? "Compressed KV follows explicit FP4 graph declarations. Backend packing, page padding and runtime pool headroom are not inferred from GPU measurements."
        : "压缩 KV 遵循图中显式 FP4 声明；backend packing、页对齐和运行时池余量不使用 GPU 实测值推算。"}</div>
    )}
    {(draftWeight > 0 || draftKvBytes > 0) && <div className="cost-breakdown cost-rollup">{[[t(language, "cost.mainModel"), accounting.main.vramBytes], [t(language, "cost.draftModel"), accounting.draft.vramBytes], [english ? "Shared pools" : "共享池", accounting.shared.vramBytes], [t(language, "cost.grandTotal"), accounting.total.vramBytes]].map(([label, value]) => <span key={label} data-bytes={value}><b>{label}</b>{formatBytes(value)}</span>)}</div>}
    <div className="cost-breakdown">{[[t(language, "cost.weights"), cost.memory.weightBytes], [t(language, "cost.buffers"), cost.memory.bufferBytes || 0], [t(language, "cost.kv"), cost.memory.kvBytes], [t(language, "cost.kdaState"), cost.memory.stateBytes], ...(cost.memory.speculativeStateBytes > 0 ? [[t(language, "cost.speculativeScratch"), cost.memory.speculativeStateBytes]] : [])].map(([label, value]) => <span key={label} data-bytes={value}><b>{label}</b>{formatBytes(value)}</span>)}</div>
    <div className="cost-breakdown cost-kv-ownership">{[["main", english ? "Main KV" : "主模型 KV"], ["draft", english ? "Draft KV" : "草稿 KV"], ["shared", english ? "Shared KV" : "共享 KV"], ["total", english ? "Total KV" : "总 KV"]].map(([owner, label]) => <span key={owner} data-owner={owner} data-bytes={accounting[owner].kvBytes}><b>{label}</b>{formatBytes(accounting[owner].kvBytes)}</span>)}</div>
    <div className="cost-metrics"><span>{t(language, "cost.totalVram")} <b>{formatBytes(cost.memory.totalBytes)}</b></span><span>{text.fitCard} <b className={planFitsMemory === true ? "fit" : "no-fit"}>{fitText(planFitsMemory, language)}</b></span><span>{t(language, "cost.maxContext")} <b>{planMaxContext == null ? "-" : planMaxContext.toLocaleString()}</b></span><span>{t(language, "cost.macsPerToken")} <b>{formatMacs(cost.macsPerToken)}</b></span><span>{t(language, "cost.macsPerForward")} <b>{formatMacs(cost.totalMacs)}</b></span>{summary.macsSources.length > 0 && <span className="cost-macs-sources" title={t(language, "cost.macsOriginTitle")}>{t(language, "cost.macsOrigin")} <b>{summary.macsSources.map((entry) => `${entry.label} ${entry.count}`).join(" · ")}</b></span>}{summary.valueSourceCounts && <span className="cost-value-source" title={summary.valueSourceCounts.title}>{t(language, "cost.weightOrigin")} <b>{summary.valueSourceCounts.text}</b></span>}<span>{t(language, "cost.flopsPerForward")} <b>{formatMacs(cost.totalFlops)}</b></span><span data-bound={roofline?.bound || "unknown"}>Roofline <b>{summary.boundLabel}</b></span><span>{t(language, "cost.communication")} <b>{formatBytes(communication?.totalBytes)}</b></span>{summary.unknownComputeCount > 0 && <span className="cost-coverage-warn">{t(language, "cost.costNotCovered")} <b>{summary.unknownComputeCount}</b></span>}<span className="cost-weight-source"><b className={cost.weightSource === "checkpoint" ? "fit" : ""}>{summary.weightSourceLabel}</b></span></div>
    {projected?.ok && <div className="cost-stages">{stageRates && projected.stages.map((stage) => <span key={stage.stage}>{t(language, "cost.stageLine", { stage: stage.stage, weights: formatBytes(stage.weightBytes), kv: formatBytes(stage.kvBytes), state: formatBytes(stage.stateBytes || 0), scratch: stage.speculativeStateBytes > 0 ? ` · ${t(language, "cost.speculativeScratch")} ${formatBytes(stage.speculativeStateBytes)}` : "", hbm: formatSeconds(stage.totalBytes / stageRates.bytesPerSecond) })}</span>)}</div>}
    {domainBreakdown.length > 0 && <div className="cost-domain-breakdown" aria-label={t(language, "cost.byDomain")}><span className="cost-domain-label">{t(language, "cost.byDomain")}</span>{domainBreakdown.map((entry) => <span key={entry.group} className="cost-domain-item" data-group={entry.group}>{t(language, `cost.domain.${entry.group}`)} <b>{(entry.pct * 100).toFixed(entry.pct >= 0.1 ? 0 : 1)}%</b></span>)}</div>}
    {mode === "pd" && pd?.ok && <div className="pd-summary-modern"><b>{t(language, "cost.pdTransfer")}</b><span>{formatBytes(pd.aggregateBytes)} total · {formatBytes(pd.perDecodeRankBytes + (pd.perDecodeRankStateBytes || 0))} / Decode rank</span><span>{t(language, pd.linkSourceCode)}{pd.linkBandwidth ? ` · ${formatRate(pd.linkBandwidth)}` : ""}{pd.transferSeconds != null ? ` · ≈${pd.transferSeconds >= 1 ? pd.transferSeconds.toFixed(2) + " s" : (pd.transferSeconds * 1000).toFixed(1) + " ms"}` : ""}</span><span>Prefill {text.fit} {fitText(pdFit?.prefill?.fit, language)} · Decode {text.fit} {fitText(pdFit?.decode?.fit, language)}</span></div>}
    <div className="cost-assumptions">
        <span>{accounting.framework === "neutral" ? "neutral · config-faithful" : `${accounting.framework} · runtime profile`}. </span>
        <span title={accounting.evidence.unknownFields.join(", ")}>{unknownFieldsText
          ? `${english ? "Unknown: " : "未知项："}${unknownFieldsText}。 `
          : ""}</span>
        {chunkedNote}
        {weightNote}
        {text.theoretical}
        {' '}
        {roofline && t(language, "cost.rooflineTimes", { bound: boundLabel, times: rooflineTimes })}
        {roofline && roofline.bound === 'unknown' && summary.missingCount > 0 && missingNote}
        {summary.unknownComputeCount > 0 && unknownComputeNote}
    </div>
  </section>;
}
