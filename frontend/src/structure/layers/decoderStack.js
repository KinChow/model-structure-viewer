import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { compactRanges, layerKinds } from "./ranges.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { attentionScheduleOf, indexerScheduleOf, attentionKindOf } from "./schedule.js";
import { hfNamedClass } from "../archs/index.js";
import { foldedLayerName } from "./foldedLayerName.js";

export function decoderStackNetwork(id, normalized) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layers = normalized.layers || 0;
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
  const combinedKinds = kinds.map((kind, index) => {
    const attentionKind = attentionKinds[index] || defaultAttentionKind;
    const compressionVariant = attentionKind === "dsv4"
      ? `:c${normalized.compressRatios?.[index] ?? 0}`
      : "";
    // V4.1 CSA2 跨层共享：只有 kv_source 层自带压缩 KV（resident）、index_source 层自带 index k_cache，
    // Reuse 层复用 source（resident=0，实测见 nv_evidence/nv5/v41_kv_shapes_reduced.json）。折叠签名必须
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
    const hasPle = normalized.pleLayerIds?.includes(index + 1) ? "ple" : "no-ple";
    // Engram 命中层（0-indexed）必须独立成段，否则与相邻非 engram 层折叠后丢失。
    const hasEngram = normalized.engramLayerIds?.includes(index) ? "engram" : "no-engram";
    const mhcBoundary = normalized.multiHyperConnection
      ? (index === (layers || 0) - 1 ? "mhc-last" : "mhc-middle")
      : "no-mhc";
    return `${kind}:${attentionKind}${compressionVariant}${csaShareVariant}${indexerVariant}:${hasPle}:${hasEngram}:${mhcBoundary}`;
  });
  const children = compactRanges(combinedKinds).map((range) => {
    const repeat = range.end - range.start + 1;
    const [layerKind, attentionKind] = String(range.kind).split(":");
    const layer = decoderLayerModule(`${id}.${range.start}`, normalized, {
      layerKind,
      attentionKind,
      layerIndex: range.start,
    });
    layer.name = foldedLayerName(range.start, range.end, "DecoderLayer");
    layer.type = "layer-group";
    layer.repeat = repeat;
    layer.attributes = {
      ...layer.attributes,
      range: `${range.start}..${range.end}`,
    };
    return layer;
  });

  return withShapeDims(moduleSpec(
    id,
    id.startsWith("language_model") ? "Text Decoder Layers" : "Decoder Layers",
    "decoder",
    { class: hfNamedClass(normalized, "modelClass", "Model"), num_hidden_layers: layers, sequence: true, ...shapeFlow(shapes.hidden, shapes.hidden) },
    children,
    layers || undefined,
  ), dims.hidden, dims.hidden);
}
