import { moduleSpec } from "./base.js";
import { decoderLayerModule } from "./decoderLayer.js";
import { compactRanges, layerKinds } from "./ranges.js";
import { shapeFlow, tensorShapes } from "../shapes.js";

export function decoderStackNetwork(id, normalized, options = {}) {
  const shapes = tensorShapes(normalized);
  const layers = normalized.layers || 0;
  const defaultLayerKind = options.defaultLayerKind || (normalized.experts ? "moe" : "dense");
  const defaultAttentionKind = options.attentionKind || "gqa";
  const kinds = layerKinds(normalized, defaultLayerKind);
  const attentionKinds = normalized.attentionSchedule?.length
    ? normalized.attentionSchedule
    : Array.from({ length: layers }, () => defaultAttentionKind);
  const children = compactRanges(kinds).map((range) => {
    const repeat = range.end - range.start + 1;
    const attentionKind = attentionKinds[range.start] || defaultAttentionKind;
    const layer = decoderLayerModule(`${id}.${range.start}`, normalized, {
      layerKind: range.kind,
      attentionKind,
    });
    layer.name = `${range.start} (DecoderLayer)${repeat > 1 ? ` x${repeat}` : ""}`;
    layer.type = "layer-group";
    layer.repeat = repeat;
    layer.attributes = {
      ...layer.attributes,
      range: `${range.start}..${range.end}`,
    };
    return layer;
  });

  return moduleSpec(
    id,
    id === "text_decoder" ? "Text Decoder Layers" : "Decoder Layers",
    "decoder",
    { class: "DecoderStack", num_hidden_layers: layers, ...shapeFlow(shapes.hidden, shapes.hidden) },
    children,
    layers || undefined,
  );
}
