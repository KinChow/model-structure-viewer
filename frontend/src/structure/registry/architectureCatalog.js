/** Canonical architecture capabilities shared by the registry and materializer. */
export const ARCHITECTURE_CATALOG = Object.freeze({
  "gqa-decoder": { hasBuilder: true, multimodalVariant: "multimodal-gqa-decoder" },
  "gqa-moe-decoder": { hasBuilder: true, multimodalVariant: "multimodal-gqa-moe-decoder" },
  "mla-moe-decoder": { hasBuilder: true, multimodalVariant: "multimodal-mla-moe-decoder" },
  "multimodal-gqa-decoder": { hasBuilder: true },
  "multimodal-sparse-moe-decoder": { hasBuilder: true },
  "multimodal-gqa-moe-decoder": { hasBuilder: true },
  "multimodal-mla-moe-decoder": { hasBuilder: true },
  "hybrid-multimodal-moe-decoder": { hasBuilder: true },
  "unsupported": { hasBuilder: false },
});

export const BUILDER_ARCHITECTURES = new Set(
  Object.keys(ARCHITECTURE_CATALOG).filter((name) => ARCHITECTURE_CATALOG[name].hasBuilder),
);

export function hasBuilderArchitecture(canonicalArchitecture) {
  return Boolean(ARCHITECTURE_CATALOG[canonicalArchitecture]?.hasBuilder);
}

export function multimodalVariant(canonicalArchitecture) {
  return ARCHITECTURE_CATALOG[canonicalArchitecture]?.multimodalVariant || null;
}
