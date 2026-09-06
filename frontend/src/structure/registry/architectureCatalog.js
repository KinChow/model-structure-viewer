/** Canonical architecture capabilities shared by the registry and materializer. */
export const ARCHITECTURE_CATALOG = Object.freeze({
  "gqa-decoder": { hasTemplate: true },
  "gqa-moe-decoder": { hasTemplate: true },
  "mla-moe-decoder": { hasTemplate: true },
  "multimodal-sparse-moe-decoder": { hasTemplate: true },
  "multimodal-gqa-moe-decoder": { hasTemplate: true },
  "generic-decoder": { hasTemplate: false },
  "generic-config": { hasTemplate: false },
});

export function hasTemplateArchitecture(canonicalArchitecture) {
  return Boolean(ARCHITECTURE_CATALOG[canonicalArchitecture]?.hasTemplate);
}
