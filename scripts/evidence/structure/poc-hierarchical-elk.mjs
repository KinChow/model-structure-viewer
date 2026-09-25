#!/usr/bin/env node
/**
 * Compare a hierarchical ELK layout POC against the current production layout.
 *
 * This is an evidence script only. It does not change the production layout
 * or Graph IR. Each visible hierarchy owner gets an independent ELK graph;
 * only edges whose two endpoints are direct children of that owner are
 * submitted to that layout run. Cross-hierarchy edges are recorded as
 * boundary relations and are not silently treated as same-level edges.
 *
 * Usage:
 *   node scripts/evidence/structure/poc-hierarchical-elk.mjs
 *   node scripts/evidence/structure/poc-hierarchical-elk.mjs /tmp/result.json
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildStructureFromArtifacts } from "../../../frontend/src/structure/buildStructure.js";
import { graphRoot } from "../../../frontend/src/structure/graph/selectors.js";
import { layoutGraph } from "../../../frontend/src/diagram/layout.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const { default: Elk } = await import(path.join(repoRoot, "frontend/node_modules/elkjs/lib/elk.bundled.js"));
const elk = new Elk();
const outputPath = process.argv[2] || path.join(
  repoRoot,
  "artifacts/architecture-repair/hierarchical-elk-poc-2026-09-25.json",
);

const models = [
  "deepseek-ai/DeepSeek-V4.1-Flash",
  "Qwen/Qwen3.8-Flash-Next",
  "zai-org/GLM-5.2",
  "moonshotai/Kimi-K3",
];

const parentPath = (value) => {
  const index = value.lastIndexOf(".");
  return index < 0 ? null : value.slice(0, index);
};

const depth = (value) => value.split(".").length - 1;

const stageFor = (node) => {
  const type = String(node?.node?.type || "").toLowerCase();
  if (type.includes("mtp") || type.includes("dspark") || type.includes("draft")) return 2;
  if (["embedding", "vision", "projector", "merger", "encoder", "fusion"]
    .some((token) => type.includes(token))) return 0;
  return 1;
};

function fullyExpanded(root) {
  const paths = new Set();
  function visit(node, pathName) {
    paths.add(pathName);
    (node.children || []).forEach((child, index) => visit(child, `${pathName}.${index}`));
  }
  visit(root, "root");
  return paths;
}

function commonAncestor(left, right, parents) {
  const ancestors = new Set();
  let current = left;
  while (current) {
    ancestors.add(current);
    current = parents.get(current);
  }
  current = right;
  while (current) {
    if (ancestors.has(current)) return current;
    current = parents.get(current);
  }
  return "root";
}

function directChild(owner, pathName, parents) {
  let current = pathName;
  while (parents.get(current) !== owner) {
    current = parents.get(current);
    if (!current) return owner;
  }
  return current;
}

function addOwnerEdge(edgesByOwner, owner, edge) {
  const edges = edgesByOwner.get(owner) || [];
  if (!edges.some((item) => item.source === edge.source && item.target === edge.target && item.kind === edge.kind)) {
    edges.push(edge);
  }
  edgesByOwner.set(owner, edges);
}

async function runHierarchicalPoc(graph, mode) {
  const nodeByPath = new Map(graph.nodes.map((node) => [node.path, node]));
  const parents = new Map(graph.nodes.map((node) => [node.path, parentPath(node.path)]));
  const childrenByOwner = new Map();

  for (const node of graph.nodes) {
    if (node.path === "root") continue;
    const owner = parents.get(node.path);
    const children = childrenByOwner.get(owner) || [];
    children.push(node.path);
    childrenByOwner.set(owner, children);
  }
  for (const children of childrenByOwner.values()) {
    children.sort((left, right) => Number(left.split(".").at(-1)) - Number(right.split(".").at(-1)));
  }

  const edgesByOwner = new Map();
  let crossHierarchyEdgeCount = 0;
  for (const edge of graph.edges || []) {
    if (edge.kind !== "dataflow" && edge.kind !== "module-order") continue;
    if (!nodeByPath.has(edge.source) || !nodeByPath.has(edge.target) || edge.source === edge.target) continue;
    const owner = commonAncestor(edge.source, edge.target, parents);
    const source = directChild(owner, edge.source, parents);
    const target = directChild(owner, edge.target, parents);
    if (source === target) continue;
    if (source !== edge.source || target !== edge.target) crossHierarchyEdgeCount += 1;
    addOwnerEdge(edgesByOwner, owner, {
      id: `${edge.id}::${owner}`,
      source,
      target,
      kind: edge.kind,
      original: edge.id,
    });
  }

  const layouts = new Map();

  async function solve(owner) {
    if (layouts.has(owner)) return layouts.get(owner);
    const children = childrenByOwner.get(owner) || [];
    const childResults = [];
    for (const child of children) childResults.push(await solve(child));

    const shapes = children.map((child, index) => {
      const node = nodeByPath.get(child);
      const result = childResults[index];
      const width = mode === "proxy" || !childrenByOwner.has(child) ? node.width : result.width;
      const height = mode === "proxy" || !childrenByOwner.has(child) ? node.height : result.height;
      return {
        id: child,
        width,
        height,
        layoutOptions: {
          "elk.partitioning.partition": String(owner === "root" ? stageFor(node) : 0),
        },
      };
    });

    const localEdges = (edgesByOwner.get(owner) || [])
      .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] }));
    const hasDataflow = (edgesByOwner.get(owner) || []).some((edge) => edge.kind === "dataflow");
    const orderEdges = hasDataflow ? [] : children.slice(0, -1).map((source, index) => ({
      id: `__poc_order__${owner}__${index}`,
      sources: [source],
      targets: [children[index + 1]],
    }));

    const result = await elk.layout({
      id: `poc:${owner}`,
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": owner === "root" ? "RIGHT" : "DOWN",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.partitioning.activate": owner === "root" ? "true" : "false",
        "elk.spacing.nodeNode": "28",
        "elk.spacing.edgeNode": "18",
        "elk.spacing.edgeEdge": "10",
        "elk.layered.spacing.nodeNodeBetweenLayers": owner === "root" ? "72" : "28",
        "elk.layered.spacing.edgeNodeBetweenLayers": "18",
        "elk.layered.layering.strategy": "LONGEST_PATH",
      },
      children: shapes,
      edges: [...localEdges, ...orderEdges],
    });

    const layout = {
      owner,
      width: children.length ? (result.width || 0) + 56 : nodeByPath.get(owner).width,
      height: children.length ? (result.height || 0) + 56 : nodeByPath.get(owner).height,
      children: new Map((result.children || []).map((child) => [
        child.id,
        {
          x: (child.x || 0) + 28,
          y: (child.y || 0) + 28,
          width: child.width || 0,
          height: child.height || 0,
        },
      ])),
    };
    layouts.set(owner, layout);
    return layout;
  }

  await solve("root");
  const positions = new Map();
  function place(owner, offsetX = 0, offsetY = 0) {
    const layout = layouts.get(owner);
    for (const [child, position] of layout.children) {
      const absolute = {
        x: offsetX + position.x,
        y: offsetY + position.y,
        width: position.width,
        height: position.height,
      };
      positions.set(child, absolute);
      if (childrenByOwner.has(child)) place(child, absolute.x, absolute.y);
    }
  }
  place("root");

  const topLevel = graph.nodes
    .filter((node) => depth(node.path) === 1)
    .map((node) => ({
      id: node.path,
      stage: stageFor(node),
      ...positions.get(node.path),
    }));
  const rootLayout = layouts.get("root");
  return {
    mode,
    model_id: graph.model_id || null,
    graph_nodes: graph.nodes.length,
    graph_edges: graph.edges.length,
    hierarchy_layouts: layouts.size,
    cross_hierarchy_edges: crossHierarchyEdgeCount,
    root_same_level_edges: (edgesByOwner.get("root") || []).length,
    root_extent: { width: rootLayout.width, height: rootLayout.height },
    top_level: topLevel.map((node) => ({
      id: node.id,
      stage: node.stage,
      x: Math.round(node.x),
      y: Math.round(node.y),
      width: Math.round(node.width),
      height: Math.round(node.height),
    })),
  };
}

const findings = [];
for (const modelId of models) {
  const configPath = path.join(repoRoot, "models", modelId, "config.json");
  const config = JSON.parse(await fs.readFile(configPath, "utf8"));
  const structure = buildStructureFromArtifacts({ modelId, config });
  const root = graphRoot(structure.graph);
  const expanded = fullyExpanded(root);
  const graph = layoutGraph(structure, expanded);
  graph.model_id = modelId;
  findings.push({
    model_id: modelId,
    current_visible_nodes: graph.nodes.length,
    current_visible_edges: graph.edges.length,
    intrinsic: await runHierarchicalPoc(graph, "intrinsic"),
    proxy: await runHierarchicalPoc(graph, "proxy"),
  });
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify({
  generated_at: new Date().toISOString(),
  note: "POC only; no production layout or Graph IR mutation",
  findings,
}, null, 2));

console.log(JSON.stringify({
  models: findings.length,
  output: outputPath,
  results: findings.map((finding) => ({
    model_id: finding.model_id,
    intrinsic_height: Math.round(finding.intrinsic.root_extent.height),
    proxy_height: Math.round(finding.proxy.root_extent.height),
    intrinsic_layouts: finding.intrinsic.hierarchy_layouts,
    proxy_layouts: finding.proxy.hierarchy_layouts,
    same_level_edges: finding.proxy.root_same_level_edges,
    cross_hierarchy_edges: finding.proxy.cross_hierarchy_edges,
  })),
}, null, 2));
