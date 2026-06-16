export function metaForNode(node) {
  const attrs = node.attributes || {};
  const lines = [];
  if (attrs.shape) lines.push(String(attrs.shape));
  if (attrs.hidden_size) lines.push(`hidden ${attrs.hidden_size}`);
  const heads = attrs.num_attention_heads || attrs.attention_heads;
  if (heads) lines.push(`heads ${heads}`);
  const experts = attrs.routed_experts || attrs.num_local_experts;
  if (experts) lines.push(`experts ${experts}`);
  if (attrs.range) lines.push(String(attrs.range));
  if (lines.length === 0 && node.type) lines.push(node.type);
  return lines.slice(0, 2);
}

export function typeClass(type) {
  if (type.includes("embedding")) return "embedding";
  if (type.includes("attention")) return "attention";
  if (type.includes("moe")) return "moe";
  if (type.includes("mlp")) return "mlp";
  if (type.includes("projector")) return "projector";
  if (type.includes("output") || type.includes("head") || type.includes("mtp")) return "output";
  if (type.includes("vision")) return "vision";
  if (type.includes("layer") || type.includes("module")) return "layer";
  return "model";
}
