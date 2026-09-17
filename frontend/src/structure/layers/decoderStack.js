import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { compactRanges, layerKinds } from "./ranges.js";
import { shapeFlow, tensorShapes } from "../operators/shapes.js";
import { tensorDims } from "../config/dims.js";
import { attentionScheduleOf, indexerScheduleOf } from "./schedule.js";
import { hfNamedClass } from "../archs/index.js";
import { foldedLayerName } from "./foldedLayerName.js";

export function decoderStackNetwork(id, normalized, options = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layers = normalized.layers || 0;
  const defaultLayerKind = options.defaultLayerKind || (normalized.experts ? "moe" : "dense");
  const defaultAttentionKind = options.attentionKind || "gqa";
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
    const indexerVariant = attentionKind === "qsa" && indexerSchedule?.length
      ? `:i${indexerSchedule[index] || "compute"}`
      : "";
    const hasPle = normalized.pleLayerIds?.includes(index + 1) ? "ple" : "no-ple";
    // Engram 命中层（0-indexed）必须独立成段，否则与相邻非 engram 层折叠后丢失。
    const hasEngram = normalized.engramLayerIds?.includes(index) ? "engram" : "no-engram";
    const mhcBoundary = normalized.multiHyperConnection
      ? (index === (layers || 0) - 1 ? "mhc-last" : "mhc-middle")
      : "no-mhc";
    return `${kind}:${attentionKind}${compressionVariant}${indexerVariant}:${hasPle}:${hasEngram}:${mhcBoundary}`;
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
