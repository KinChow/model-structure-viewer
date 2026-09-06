/** Canonical architecture capabilities shared by the registry and materializer. */
export const ARCHITECTURE_CATALOG = Object.freeze({
  "gqa-decoder": { hasTemplate: true, multimodalVariant: "multimodal-gqa-decoder" },
  "gqa-moe-decoder": { hasTemplate: true, multimodalVariant: "multimodal-gqa-moe-decoder" },
  "mla-moe-decoder": { hasTemplate: true, multimodalVariant: "multimodal-mla-moe-decoder" },
  "multimodal-gqa-decoder": { hasTemplate: true },
  "multimodal-sparse-moe-decoder": { hasTemplate: true },
  "multimodal-gqa-moe-decoder": { hasTemplate: true },
  "multimodal-mla-moe-decoder": { hasTemplate: true },
  "hybrid-multimodal-moe-decoder": { hasTemplate: true },
  "generic-decoder": { hasTemplate: false },
  "generic-config": { hasTemplate: false },
});

export const TEMPLATE_FAMILIES = new Set(
  Object.keys(ARCHITECTURE_CATALOG).filter((name) => ARCHITECTURE_CATALOG[name].hasTemplate),
);

export function hasTemplateArchitecture(canonicalArchitecture) {
  return Boolean(ARCHITECTURE_CATALOG[canonicalArchitecture]?.hasTemplate);
}

export function multimodalVariant(canonicalArchitecture) {
  return ARCHITECTURE_CATALOG[canonicalArchitecture]?.multimodalVariant || null;
}
