import { materializeDeclaredEdges } from "./declaredEdges.js";

function semanticEdges(item) {
  return materializeDeclaredEdges(item);
}


function isOutputNode(item) {
  const type = String(item?.node?.type || "").toLowerCase();
  const role = String(item?.node?.role || item?.node?.attributes?.role || "").toLowerCase();
  return type === "output" || type === "head" || role === "output";
}

function flattenTree(root) {
  const items = [];
  function visit(node, path, parentId = null) {
    const item = { node, path, parentId, childItems: [] };
    items.push(item);
    item.childItems = (node?.children || []).map((child, index) =>
      visit(child, `${path}.${index}`, path));
    return item;
  }
  if (root) visit(root, "root");
  return items;
}

export function materializeStructureGraph(root) {
  const items = flattenTree(root);
  const semanticParents = new Set(items.filter((item) => semanticEdges(item)).map((item) => item.path));
  const orderedPairs = new Set(items.flatMap((item) =>
    item.childItems.slice(0, -1).map((source, index) =>
      `${source.path}=>${item.childItems[index + 1].path}`)));

  const moduleOrderEdges = items.flatMap((item) => {
    if (semanticParents.has(item.path)) return [];
    return item.childItems.slice(0, -1).map((source, index) => {
      const target = item.childItems[index + 1];
      const edgeSource = isOutputNode(target) ? item.path : source.path;
      return {
        id: `${edgeSource}~${target.path}`,
        source: edgeSource,
        target: target.path,
        kind: "dataflow",
        evidence: "module-order",
      };
    });
  });

  const dataflowEdges = items.flatMap((item) => {
    const semantic = semanticEdges(item);
    if (semantic) return semantic;
    const operators = item.childItems.filter((child) => child.node?.type === "operator");
    const edges = [];
    for (const source of operators) {
      if (!source.node?.output_shape) continue;
      const target = operators.find((candidate) => {
        if (source.path === candidate.path || !candidate.node?.input_shape) return false;
        if (candidate.path <= source.path) return false;
        return JSON.stringify(source.node.output_shape) === JSON.stringify(candidate.node.input_shape);
      });
      if (target) {
        edges.push({
          id: `${source.path}=>${target.path}`,
          source: source.path,
          target: target.path,
          kind: "dataflow",
          evidence: orderedPairs.has(`${source.path}=>${target.path}`) ? "module-order" : "shape-match",
        });
      }
    }
    return edges;
  });

  const dataflowPairs = new Set(dataflowEdges.map((edge) => `${edge.source}=>${edge.target}`));
  return {
    version: 2,
    schema_version: 2,
    root_id: "root",
    nodes: items.map((item) => ({
      id: item.path,
      canonical_id: item.node?.id || item.path,
      module_id: item.node?.id || null,
      parent_id: item.parentId,
      order: item.parentId == null ? 0 : Number(item.path.split(".").at(-1)),
      name: item.node?.name || "",
      type: item.node?.type || "module",
      role: item.node?.role ?? null,
      repeat: item.node?.repeat ?? null,
      attributes: item.node?.attributes || {},
      source_fields: item.node?.source_fields || [],
      confidence: item.node?.confidence || "high",
      params: item.node?.params ?? null,
      weight_shapes: item.node?.weight_shapes || null,
      dtype: item.node?.dtype || null,
      input_shape: item.node?.input_shape || null,
      output_shape: item.node?.output_shape || null,
      value_source: item.node?.value_source || null,
      tensor_names: item.node?.tensor_names || null,
    })),
    edges: [
      ...dataflowEdges,
      ...moduleOrderEdges.filter((edge) => !dataflowPairs.has(`${edge.source}=>${edge.target}`)),
    ].map((edge) => {
      const sourceNode = items.find((item) => item.path === edge.source)?.node;
      const targetNode = items.find((item) => item.path === edge.target)?.node;
      return {
        ...edge,
        source_canonical_id: sourceNode?.id || edge.source,
        target_canonical_id: targetNode?.id || edge.target,
      };
    }),
  };
}
