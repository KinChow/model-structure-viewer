function requireString(entry, key) {
  const value = entry[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Catalog entry missing ${key}`);
  }
  return value;
}

export function normalizeCatalog(rawCatalog) {
  const models = Array.isArray(rawCatalog?.models) ? rawCatalog.models : [];
  return {
    models: models.map((entry) => ({
      modelId: requireString(entry, "model_id"),
      revision: entry.revision || null,
      canonicalArchitecture: entry.canonical_architecture || null,
      configPath: requireString(entry, "config_path"),
      modelType: entry.model_type || null,
      architectures: Array.isArray(entry.architectures) ? entry.architectures : [],
      metadataFiles: Array.isArray(entry.metadata_files) ? entry.metadata_files : [],
      verified: Boolean(entry.verified),
      verifiedAt: entry.verified_at || null,
      validationReport: entry.validation_report || null,
    })),
  };
}

export function staticAssetPath(relativePath, baseUrl = import.meta.env?.BASE_URL || "/") {
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return `${normalizedBase}${relativePath.replace(/^\/+/, "")}`;
}

export function catalogPath(baseUrl) {
  return staticAssetPath("models/catalog.json", baseUrl);
}

export function modelConfigPath(entry, baseUrl) {
  return staticAssetPath(`models/${entry.configPath}`, baseUrl);
}
