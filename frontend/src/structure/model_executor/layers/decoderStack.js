import { moduleSpec, withShapeDims } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { compactRanges, layerKinds } from "./ranges.js";
import { shapeFlow, tensorShapes } from "../shapes.js";
import { tensorDims } from "../dims.js";

export function decoderStackNetwork(id, normalized, options = {}) {
  const shapes = tensorShapes(normalized);
  const dims = tensorDims(normalized);
  const layers = normalized.layers || 0;
  const defaultLayerKind = options.defaultLayerKind || (normalized.experts ? "moe" : "dense");
  const defaultAttentionKind = options.attentionKind || "gqa";
  const kinds = layerKinds(normalized, defaultLayerKind);
  const attentionKinds = normalized.attentionSchedule?.length
    ? normalized.attentionSchedule
    : Array.from({ length: layers }, () => defaultAttentionKind);
  const combinedKinds = kinds.map((kind, index) => {
    const attentionKind = attentionKinds[index] || defaultAttentionKind;
    const hasPle = normalized.pleLayerIds?.includes(index + 1) ? "ple" : "no-ple";
    const mhcBoundary = normalized.multiHyperConnection
      ? (index === (layers || 0) - 1 ? "mhc-last" : "mhc-middle")
      : "no-mhc";
    return `${kind}:${attentionKind}:${hasPle}:${mhcBoundary}`;
  });
  const children = compactRanges(combinedKinds).map((range) => {
    const repeat = range.end - range.start + 1;
    const [layerKind, attentionKind] = String(range.kind).split(":");
    const layer = decoderLayerModule(`${id}.${range.start}`, normalized, {
      layerKind,
      attentionKind,
      layerIndex: range.start,
    });
    layer.name = `${range.start} (DecoderLayer)`;
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
    id === "text_decoder" ? "Text Decoder Layers" : "Decoder Layers",
    "decoder",
    { class: "DecoderStack", num_hidden_layers: layers, ...shapeFlow(shapes.hidden, shapes.hidden) },
    children,
    layers || undefined,
  ), dims.hidden, dims.hidden);
}
