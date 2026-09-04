import { useMemo, useState } from "react";
import { normalizeConfig } from "../structure/config/normalize.js";
import { aggregateCost } from "../cost/aggregate.js";
import { projectPdFit, projectPlan } from "../cost/parallel.js";
import { pdKvTransferBytes, planCommunicationBytes } from "../cost/comm.js";
import { PUBLIC_CHIPS } from "../cost/chips/public.js";

const GIB = 1024 ** 3;

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

export default function CostSummary({ structure, chips = PUBLIC_CHIPS }) {
  const [phase, setPhase] = useState("prefill");
  const [batch, setBatch] = useState(1);
  const [sequence, setSequence] = useState(2048);
  const [capacity, setCapacity] = useState(80);
  const [kvElementBytes, setKvElementBytes] = useState(2);
  const [activationGiB, setActivationGiB] = useState(1.5);
  const [runtimeGiB, setRuntimeGiB] = useState(1.5);
  const [tp, setTp] = useState(1);
  const [pp, setPp] = useState(1);
  const [ep, setEp] = useState(1);
  const [dp, setDp] = useState(1);
  const [attnMode, setAttnMode] = useState("tp");
  const [pdEnabled, setPdEnabled] = useState(false);
  const [prefillTp, setPrefillTp] = useState(1);
  const [decodeTp, setDecodeTp] = useState(1);
  const [prefillPp, setPrefillPp] = useState(1);
  const [prefillEp, setPrefillEp] = useState(1);
  const [prefillDp, setPrefillDp] = useState(1);
  const [decodePp, setDecodePp] = useState(1);
  const [decodeEp, setDecodeEp] = useState(1);
  const [decodeDp, setDecodeDp] = useState(1);
  const [prefillAttnMode, setPrefillAttnMode] = useState("tp");
  const [decodeAttnMode, setDecodeAttnMode] = useState("tp");
  const [prefillChipId, setPrefillChipId] = useState(chips[0]?.id || "");
  const [decodeChipId, setDecodeChipId] = useState(chips[1]?.id || chips[0]?.id || "");
  const cost = useMemo(() => {
    if (!structure?.root || !structure.extra_config) return null;
    return aggregateCost({ root: structure.root, config: normalizeConfig(structure.extra_config),
      parameterCount: structure.summary?.parameters_by_dtype, phase, batch, sequence,
      kvBytes: kvElementBytes, activationPeak: activationGiB * GIB, runtimeConst: runtimeGiB * GIB });
  }, [structure, phase, batch, sequence, kvElementBytes, activationGiB, runtimeGiB]);
  if (!cost) return null;
  const parallel = projectPlan({
    root: structure.root,
    weightBytes: cost.memory.weightBytes,
    kvBytes: cost.memory.kvBytes,
    config: normalizeConfig(structure.extra_config),
    plan: { tp, pp, ep, dp, attnMode },
  });
  const communication = planCommunicationBytes({ root: structure.root, config: normalizeConfig(structure.extra_config), plan: { tp, pp, ep, dp, attnMode }, batch, tokens: phase === "decode" ? 1 : sequence });
  const pd = pdEnabled ? pdKvTransferBytes({
    totalKvBytes: cost.memory.kvBytes,
    config: normalizeConfig(structure.extra_config),
    pdPlan: { prefill_plan: { tp: prefillTp, pp: prefillPp, ep: prefillEp, dp: prefillDp, attnMode: prefillAttnMode }, decode_plan: { tp: decodeTp, pp: decodePp, ep: decodeEp, dp: decodeDp, attnMode: decodeAttnMode } },
    prefillChip: chips.find((chip) => chip.id === prefillChipId),
    decodeChip: chips.find((chip) => chip.id === decodeChipId),
  }) : null;
  const pdFit = pdEnabled ? projectPdFit({
    root: structure.root,
    weightBytes: cost.memory.weightBytes,
    kvBytes: cost.memory.kvBytes,
    config: normalizeConfig(structure.extra_config),
    pdPlan: { prefill_plan: { tp: prefillTp, pp: prefillPp, ep: prefillEp, dp: prefillDp, attnMode: prefillAttnMode }, decode_plan: { tp: decodeTp, pp: decodePp, ep: decodeEp, dp: decodeDp, attnMode: decodeAttnMode } },
    prefillChip: chips.find((chip) => chip.id === prefillChipId),
    decodeChip: chips.find((chip) => chip.id === decodeChipId),
    activationBytes: cost.memory.activationBytes,
    runtimeBytes: cost.memory.runtimeBytes,
  }) : null;
  const available = capacity * GIB;
  const maxContext = cost.memory.kvBytesPerToken
    ? Math.max(0, Math.floor((available - cost.memory.weightBytes - cost.memory.activationBytes - cost.memory.runtimeBytes) / cost.memory.kvBytesPerToken))
    : null;
  const parts = [
    [`权重${cost.weightSource === "derived" ? "（估算）" : ""}`, cost.memory.weightBytes], ["KV", cost.memory.kvBytes],
    ["激活峰值", cost.memory.activationBytes], ["运行时", cost.memory.runtimeBytes],
    ["通信缓冲", cost.memory.commBufferBytes],
  ];
  return (
    <section className="cost-summary" aria-label="理论成本估算">
      <div className="cost-summary-header">
        <div><b>理论成本估算</b><span className="cost-disclaimer">非仿真、非预测；结果供参考</span></div>
        <div className="cost-controls">
          <label>阶段<select value={phase} onChange={(event) => setPhase(event.target.value)}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
          <label>B<input type="number" min="1" value={batch} onChange={(event) => setBatch(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label>T<input type="number" min="1" value={sequence} onChange={(event) => setSequence(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label>卡显存 GiB<input type="number" min="1" value={capacity} onChange={(event) => setCapacity(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label>KV bytes<select value={kvElementBytes} onChange={(event) => setKvElementBytes(Number(event.target.value))}><option value="2">2</option><option value="1">1</option><option value="0.5">0.5</option></select></label>
          <label>激活 GiB<input type="number" min="0" step="0.1" value={activationGiB} onChange={(event) => setActivationGiB(Math.max(0, Number(event.target.value) || 0))} /></label>
          <label>运行时 GiB<input type="number" min="0" step="0.1" value={runtimeGiB} onChange={(event) => setRuntimeGiB(Math.max(0, Number(event.target.value) || 0))} /></label>
        </div>
      </div>
      <div className="cost-breakdown">{parts.map(([label, value]) => <span key={label}><b>{label}</b>{formatBytes(value)}</span>)}</div>
      <div className="cost-metrics"><span>合计显存 <b>{formatBytes(cost.memory.totalBytes)}</b></span><span>单卡 fit <b className={cost.memory.totalBytes <= available ? "fit" : "no-fit"}>{cost.memory.totalBytes <= available ? "是" : "否"}</b></span><span>最大上下文约 <b>{maxContext == null ? "-" : maxContext.toLocaleString()}</b></span><span>MACs <b>{formatMacs(cost.totalMacs)}</b></span><span>计划通信 <b>{formatBytes(communication.totalBytes)}</b></span></div>
      <div className="cost-plan-controls"><b>并行计划</b><label>TP<input type="number" min="1" value={tp} onChange={(event) => setTp(Math.max(1, Number(event.target.value) || 1))} /></label><label>PP<input type="number" min="1" value={pp} onChange={(event) => setPp(Math.max(1, Number(event.target.value) || 1))} /></label><label>EP<input type="number" min="1" value={ep} onChange={(event) => setEp(Math.max(1, Number(event.target.value) || 1))} /></label><label>DP<input type="number" min="1" value={dp} onChange={(event) => setDp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Attention<select value={attnMode} onChange={(event) => setAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label></div>
      {!parallel.ok && <div className="cost-plan-error">计划无效：{parallel.errors.join("；")}</div>}
      {parallel.ok && <div className="cost-stages">{parallel.stages.map((stage) => { const stageTotal = stage.weightBytes + stage.kvBytes + cost.memory.activationBytes + cost.memory.runtimeBytes; return <span key={stage.stage}><b>Stage {stage.stage}</b>权重 {formatBytes(stage.weightBytes)} · KV {formatBytes(stage.kvBytes)} · fit <strong className={stageTotal <= available ? "fit" : "no-fit"}>{stageTotal <= available ? "是" : "否"}</strong></span>; })}</div>}
      <label className="pd-toggle"><input type="checkbox" checked={pdEnabled} onChange={(event) => setPdEnabled(event.target.checked)} />启用 PD 分离</label>
      {pdEnabled && <div className="pd-summary"><label>Prefill 芯片<select value={prefillChipId} onChange={(event) => setPrefillChipId(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label><label>Decode 芯片<select value={decodeChipId} onChange={(event) => setDecodeChipId(event.target.value)}>{chips.map((chip) => <option key={chip.id} value={chip.id}>{chip.name}</option>)}</select></label><label>Prefill TP<input type="number" min="1" value={prefillTp} onChange={(event) => setPrefillTp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Prefill PP<input type="number" min="1" value={prefillPp} onChange={(event) => setPrefillPp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Prefill EP<input type="number" min="1" value={prefillEp} onChange={(event) => setPrefillEp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Prefill DP<input type="number" min="1" value={prefillDp} onChange={(event) => setPrefillDp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Prefill Attention<select value={prefillAttnMode} onChange={(event) => setPrefillAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label><label>Decode TP<input type="number" min="1" value={decodeTp} onChange={(event) => setDecodeTp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Decode PP<input type="number" min="1" value={decodePp} onChange={(event) => setDecodePp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Decode EP<input type="number" min="1" value={decodeEp} onChange={(event) => setDecodeEp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Decode DP<input type="number" min="1" value={decodeDp} onChange={(event) => setDecodeDp(Math.max(1, Number(event.target.value) || 1))} /></label><label>Decode Attention<select value={decodeAttnMode} onChange={(event) => setDecodeAttnMode(event.target.value)}><option value="tp">TP</option><option value="dp">DP</option></select></label>{pd?.ok ? <span>按 Decode 布局：每 rank KV {formatBytes(pd.perDecodeRankBytes)} · 聚合传输 {formatBytes(pd.aggregateBytes)} · 链路 {pd.linkSource}{pd.linkBandwidth ? `（${formatRate(pd.linkBandwidth)}）` : ""} · Prefill fit {pdFit?.prefill?.fit ? "是" : "否"} · Decode fit {pdFit?.decode?.fit ? "是" : "否"}</span> : <span className="cost-plan-error">PD 计划无效：{pd?.errors?.join("；")}</span>}</div>}
      <div className="cost-assumptions">假设：KV 每元素 {kvElementBytes} bytes；激活峰值 {activationGiB} GiB；运行时常数 {runtimeGiB} GiB。通信为理论上界，不含 overlap。</div>
    </section>
  );
}
