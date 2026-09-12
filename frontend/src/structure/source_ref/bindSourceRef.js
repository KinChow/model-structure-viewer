// Bind catalog-side source_ref.json onto Graph IR nodes.
// 路径键与 verification/compare_structure.canonical_reconciliation_path 对样
// （§6.4：前后端不共享代码，契约靠折叠规则一致）。绑定主语是类定义位置，
// 实例序号与 fold 伪段不参与键。class 只在路径未命中时作为唯一类名回退。
// 版本不一致时去掉 GitHub #L 锚点（§5.3）；inspect 失败保持 null（§5.4）。

const WRAPPERS = new Set(["model", "language_model"]);
const AGGREGATE_TYPES = new Set(["layer-group", "layer-pattern-group", "vision-block-group"]);

function isFoldedSegment(segment) {
  if (/^\d+$/.test(segment)) return true;
  return /^(group|pattern)\d+$/.test(segment);
}

export function canonicalSourceRefPath(value) {
  const segments = String(value || "").split(".").filter(Boolean);
  if (segments[0] === "root") segments.shift();
  while (segments[0] && WRAPPERS.has(segments[0])) segments.shift();
  return segments.filter((segment) => !isFoldedSegment(segment)).join(".");
}

export function displaySourceRef(sourceRef, runtimeVersion) {
  if (!sourceRef || typeof sourceRef !== "object") return null;
  const file = sourceRef.file || null;
  const line = Number.isInteger(sourceRef.line) ? sourceRef.line : null;
  const version = sourceRef.version || null;
  const url = typeof sourceRef.url === "string" && sourceRef.url ? sourceRef.url : null;
  const versionMismatch = Boolean(
    runtimeVersion && version && String(runtimeVersion) !== String(version),
  );
  let href = url;
  if (href && versionMismatch) href = href.replace(/#L\d+$/, "");
  return {
    framework: sourceRef.framework || null,
    modulePath: sourceRef.module_path || null,
    className: sourceRef.class_name || null,
    file,
    line,
    version,
    url: href,
    versionMismatch,
    label: file ? (line != null ? `${file}:${line}` : file) : (sourceRef.class_name || null),
  };
}

function moduleIndex(catalog) {
  const byPath = new Map();
  const byClass = new Map();
  for (const row of catalog?.modules || []) {
    const path = canonicalSourceRefPath(row.module_path);
    if (path) {
      const pathHits = byPath.get(path) || [];
      pathHits.push(row);
      byPath.set(path, pathHits);
    }
    if (row.class_name) {
      const classHits = byClass.get(row.class_name) || [];
      classHits.push(row);
      byClass.set(row.class_name, classHits);
    }
  }
  return { byPath, byClass };
}

function sameSourceDefinition(rows) {
  const first = rows[0];
  const file = first.source_ref?.file || null;
  const line = first.source_ref?.line ?? null;
  return rows.every((row) =>
    row.class_name === first.class_name
    && (row.source_ref?.file || null) === file
    && (row.source_ref?.line ?? null) === line);
}

function pickUnique(rows, className) {
  if (!rows?.length) return null;
  const matches = className ? rows.filter((row) => row.class_name === className) : rows;
  if (!matches.length) return null;
  return sameSourceDefinition(matches) ? matches[0] : null;
}

function pickRow(node, index) {
  const path = canonicalSourceRefPath(node.canonical_id || node.module_id || node.id);
  const className = node.attributes?.class;
  const pathHit = pickUnique(path ? index.byPath.get(path) : null, className);
  if (pathHit) return pathHit;
  return pickUnique(className ? index.byClass.get(className) : null, className);
}

export function bindSourceRefToGraph(graph, catalog, { runtimeVersion } = {}) {
  if (!graph?.nodes || !catalog?.modules) {
    return { graph, diagnostics: { bound: 0, unmatched: 0, transformers_version: catalog?.transformers_version || null } };
  }
  const index = moduleIndex(catalog);
  let bound = 0;
  let unmatched = 0;
  const nodes = graph.nodes.map((node) => {
    if (AGGREGATE_TYPES.has(node.type)) {
      return { ...node, source_ref: null };
    }
    const row = pickRow(node, index);
    if (!row?.source_ref) {
      unmatched += 1;
      return { ...node, source_ref: null };
    }
    bound += 1;
    return {
      ...node,
      source_ref: displaySourceRef(row.source_ref, runtimeVersion || catalog.transformers_version),
    };
  });
  return {
    graph: { ...graph, nodes },
    diagnostics: {
      bound,
      unmatched,
      transformers_version: catalog.transformers_version || null,
    },
  };
}
