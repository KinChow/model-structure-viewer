/**
 * Terminology bridge for the public architecture gallery.
 *
 * This is deliberately a display-only projection. It does not decide
 * topology, weights, costs, or checkpoint truth; those remain Graph IR facts.
 */
const ATTENTION_LABELS = {
  gqa: "GQA",
  [["qwen", "35", "_full"].join("")]: "Gated Attention",
  qsa: "QSA",
  dsv4: "CSA/HCA",
  sparse: "MiniMax Sparse",
  mla: "MLA",
  dsa_sparse_mla: "DSA / MLA",
  linear: "DeltaNet",
};

function addCount(map, label, count) {
  if (!label || !count) return;
  map.set(label, (map.get(label) || 0) + count);
}

export function galleryAlignmentForGraph(graph) {
  const counts = new Map();
  const special = new Set();
  for (const node of graph?.nodes || []) {
    const kind = node.attributes?.attention_kind;
    if (node.type === "attention" && kind) addCount(counts, ATTENTION_LABELS[kind] || String(kind), node.repeat || 1);
    const text = `${node.name || ""} ${node.attributes?.class || ""}`.toLowerCase();
    if (node.attributes?.csa2_mode || text.includes("csa2")) special.add("CSA2");
    if (text.includes("engram") || node.attributes?.operator_id === "engram_gate") special.add("Engram");
    if (text.includes("indexshare") || node.attributes?.index_source_layer != null) special.add("IndexShare");
    if (text.includes("attention residual") || node.attributes?.aggregation_point) special.add("AttnRes");
    if (kind === "qsa" || kind === "dsa_sparse_mla") special.add("Sparse attention");
    if (node.type === "vision-encoder") special.add("Vision");
  }
  const topLevel = (graph?.nodes || []).filter((node) => node.parent_id === graph?.root_id);
  const hasEncoder = topLevel.some((node) => /encoder/i.test(`${node.type} ${node.name}`));
  const hasDecoder = topLevel.some((node) => /decoder/i.test(`${node.type} ${node.name}`));
  const decoderType = hasEncoder && hasDecoder ? "Causal encoder-decoder" : "Decoder-only";
  return {
    decoder_type: decoderType,
    attention_mix: [...counts.entries()].map(([label, count]) => `${count} ${label}`).join(" + "),
    special_features: [...special],
  };
}
