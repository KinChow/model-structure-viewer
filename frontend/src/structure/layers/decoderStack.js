import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { compactRanges, layerKinds } from "./ranges.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { attentionScheduleOf, indexerScheduleOf, attentionKindOf, csa2ModeForLayer, indexShareSourceLayerOf, isIndexShareConfig } from "./schedule.js";
import { hfAttentionAttr, hfNamedClass } from "../archs/index.js";
import { foldedLayerName } from "./foldedLayerName.js";
import { attentionResidualStage } from "./residual.js";

const CSA2_MODE_LABEL = { full: "Full", reindex: "Reindex", reuse: "Reuse", swa: "SWA" };

// 折叠组显示名：CSA2 层显示 `CSA2(ratio, Mode)` / `SWA`（对齐 DeepSeek 官方图），
// 非 CSA2 模型（mode===null）保持既有 "DecoderLayer"。
function csa2GroupLabel(normalized, index, mode) {
  if (!mode) return "DecoderLayer";
  if (mode === "swa") return "SWA";
  const ratio = normalized?.compressRatios?.[index] ?? 0;
  return `CSA2(${ratio}, ${CSA2_MODE_LABEL[mode] || mode})`;
}

/**
 * opts.range=[start,end]（全局层号，含端点）时只装配该窗口，其余层号仍用于 schedule/
 * ratio/source 查找（全局一致）；opts.name/opts.type 覆盖模块名与类型（CED 两段拆分用）。
 * 缺省（无 range）行为与此前完全一致（单栈、type="decoder"）。
 */
export function decoderStackNetwork(id, normalized, opts = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layers = normalized.layers || 0;
  const start = Array.isArray(opts.range) ? opts.range[0] : 0;
  const end = Array.isArray(opts.range) ? opts.range[1] : layers - 1;
  const candidateSource = normalized.candidateSourceLayerId;
  // family 默认全部从 config 推导（单一源）：dense/moe 由 experts + 逐层 layerKinds 决定；
  // 注意力种类由 attentionKindOf 兜底、attentionScheduleOf 逐层覆盖。builder 不再传 opts。
  const defaultLayerKind = normalized.experts ? "moe" : "dense";
  const defaultAttentionKind = attentionKindOf(normalized);
  const kinds = layerKinds(normalized, defaultLayerKind);
  const attentionSchedule = attentionScheduleOf(normalized);
  const attentionKinds = attentionSchedule?.length
    ? attentionSchedule
    : Array.from({ length: layers }, () => defaultAttentionKind);
  const indexerSchedule = indexerScheduleOf(normalized);
  const indexShare = isIndexShareConfig(normalized);
  const indexerId = (index) => `${id}.${index}.${hfAttentionAttr(normalized, "qsa")}.indexer`;
  const combinedKinds = kinds.map((kind, index) => {
    const attentionKind = attentionKinds[index] || defaultAttentionKind;
    const compressionVariant = attentionKind === "dsv4"
      ? `:c${normalized.compressRatios?.[index] ?? 0}`
      : "";
    // V4.1 CSA2 跨层共享：只有 kv_source 层自带压缩 KV（resident）、index_source 层自带 index k_cache，
    // Reuse 层复用 source（resident=0，实测见 docs/details/evidence/memory/deepseek_v41_csa2_kv_bytes.md）。折叠签名必须
    // 区分 source/reuse，否则 compactRanges 会把 source 层与其后 Reuse 层折成一段、以 source 值 ×range_size
    // 计（抹平逐层门控）。缺省（V4-Flash/Pro 无 source_layer_ids）为空串 → V4 折叠不变。
    const csaShareVariant = attentionKind === "dsv4" && Array.isArray(normalized.kvSourceLayerIds)
      ? `${normalized.kvSourceLayerIds.includes(index) ? ":kvsrc" : ":kvreuse"}${
          Array.isArray(normalized.indexSourceLayerIds)
            ? (normalized.indexSourceLayerIds.includes(index) ? ":idxsrc" : ":idxreuse")
            : ""
        }`
      : "";
    const indexerVariant = attentionKind === "qsa" && indexerSchedule?.length
      ? `:i${indexerSchedule[index] || "compute"}`
      : "";
    // IndexShare 的 source 层必须可独立定位；否则 source 与前两个 full 层
    // 折成同一代表节点，跨层边会错误地指向 0..2 的代表。
    const indexShareSourceBoundary = indexShare
      && indexerSchedule?.[index] === "compute"
      && indexerSchedule?.[index + 1] === "reuse"
      ? ":idxsource"
      : "";
    const hasPle = normalized.pleLayerIds?.includes(index + 1) ? "ple" : "no-ple";
    // Engram 命中层（0-indexed）必须独立成段，否则与相邻非 engram 层折叠后丢失。
    const hasEngram = normalized.engramLayerIds?.includes(index) ? "engram" : "no-engram";
    const mhcBoundary = normalized.multiHyperConnection
      ? (index === (layers || 0) - 1 ? "mhc-last" : "mhc-middle")
      : "no-mhc";
    const residualStage = normalized.attnResBlockSize ? attentionResidualStage(normalized, index) : null;
    const residualSignature = residualStage
      // Each prefix_out is a different immutable state used by the next layer.
      // A repeat representative cannot stand in for that exact endpoint.
      ? `:attnres-${residualStage.block}-${residualStage.before}-${residualStage.after}-${residualStage.write}-layer${index}`
      : "";
    return `${kind}:${attentionKind}${compressionVariant}${csaShareVariant}${indexerVariant}${indexShareSourceBoundary}:${hasPle}:${hasEngram}:${mhcBoundary}${residualSignature}`;
  });
  // 段内折叠：先取窗口签名再 compactRanges，最后把区间下标偏移回全局层号，
  // 保证段边界不跨折叠、且 ratio/source/mhc 查找始终用全局层号。
  const windowKinds = combinedKinds.slice(start, end + 1);
  const children = compactRanges(windowKinds).map((range) => {
    const gStart = range.start + start;
    const gEnd = range.end + start;
    const repeat = gEnd - gStart + 1;
    const [layerKind, attentionKind] = String(range.kind).split(":");
    const layer = decoderLayerModule(`${id}.${gStart}`, normalized, {
      layerKind,
      attentionKind,
      layerIndex: gStart,
    });
    const mode = csa2ModeForLayer(normalized, gStart);
    const indexSource = indexShare ? indexShareSourceLayerOf(normalized, gStart) : null;
    layer.name = foldedLayerName(gStart, gEnd, csa2GroupLabel(normalized, gStart, mode));
    layer.type = "layer-group";
    layer.repeat = repeat;
    const csaAttrs = mode ? { csa2_mode: mode, compress_ratio: normalized.compressRatios?.[gStart] ?? 0 } : {};
    if (mode && candidateSource != null) {
      // 解码器首个 Full 层构建候选池；其后 Reindex 层受候选池约束（层级稀疏索引器）。
      if (gStart === candidateSource) csaAttrs.candidate_pool_source = true;
      else if (mode === "reindex" && gStart > candidateSource) csaAttrs.candidate_constrained = true;
    }
    layer.attributes = {
      ...layer.attributes,
      range: `${gStart}..${gEnd}`,
      ...csaAttrs,
      ...(indexSource != null
        ? {
            index_source_layer: indexSource,
            index_source_layer_id: indexerId(indexSource),
          }
        : {}),
    };
    return layer;
  });

  const segmentLayers = end - start + 1;
  const rootName = opts.name || (id.startsWith("language_model") ? "Text Decoder Layers" : "Decoder Layers");
  const rootType = opts.type || "decoder";
  const sequenceEdges = children.slice(0, -1).map((child, index) => [
    child.id,
    children[index + 1].id,
  ]);
  const indexShareEdges = [];
  if (indexShare) {
    for (const child of children) {
      const source = child.attributes.index_source_layer;
      if (source == null) continue;
      indexShareEdges.push([
        indexerId(source),
        `${child.id}.${hfAttentionAttr(normalized, "qsa")}.index_reuse`,
      ]);
    }
  }
  const indexShareRelations = indexShareEdges.map(([from, to]) => ({
    from,
    to,
    relation: "index-reuse",
    label: `reuse top-k from layer ${from.match(/layers\.(\d+)\./)?.[1] ?? "source"}`,
  }));
  const attnResEdges = normalized.attnResBlockSize ? children.slice(0, -1).flatMap((child, index) => [
    [`${child.id}.prefix_out`, `${children[index + 1].id}.layer_in`],
    [`${child.id}.bank_out`, `${children[index + 1].id}.bank_in`],
  ]) : [];
  const attnResRelations = attnResEdges.map(([from, to]) => ({
    from, to, relation: "depth-state", label: from.endsWith("bank_out") ? "depth snapshots" : "current block prefix",
  }));
  return withShapeDims(moduleSpec(
    id,
    rootName,
    rootType,
    {
      class: hfNamedClass(normalized, "modelClass", "Model"),
      num_hidden_layers: segmentLayers,
      sequence: true,
      ...(normalized.attnResBlockSize ? {
        attnres_final_prefix: `${children.at(-1).id}.prefix_out`,
        attnres_final_bank: `${children.at(-1).id}.bank_out`,
      } : {}),
      ...(indexShareEdges.length || normalized.attnResBlockSize
        ? {
            dataflow_edges: [...(normalized.attnResBlockSize ? attnResEdges : sequenceEdges), ...indexShareEdges],
            dataflow_edge_relations: [...indexShareRelations, ...attnResRelations],
          }
        : {}),
      ...shapeFlow(shapes.hidden, shapes.hidden),
    },
    children,
    segmentLayers || undefined,
  ), dims.hidden, dims.hidden);
}
