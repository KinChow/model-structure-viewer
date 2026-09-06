// mergeSemantics.js —— 把 checkpoint 真值（trie 骨架）与模板语义树合并。
//
// 依据 evolution_design.md §4.2(b2)/§4.4：
// - 有模板（家族命中）：以模板树为骨架，按模块路径后缀匹配绑定真值
//   （params / weight_shapes / dtype / tensor_names / value_source="checkpoint"），
//   trie 里模板未声明的含参模块计入 template_gaps（模板完整性信号）。
// - 无模板（generic-*）：直接用 trie 树作为结构树，数值仍精确。
//
// 注意：真实参数量换算已由 @huggingface/hub 完成（cost/weights.js），
// 这里只做结构合并与展示字段落位，不做任何换算。

import { buildSkeleton } from "./skeleton.js";
import { ARCHITECTURE_CATALOG, hasTemplateArchitecture } from "../registry/architectureCatalog.js";

/** 模板家族（由统一架构目录派生，避免 registry/materializer 漂移）。 */
export const TEMPLATE_FAMILIES = new Set(
  Object.keys(ARCHITECTURE_CATALOG).filter(hasTemplateArchitecture),
);

const PATH_WRAPPERS = new Set(["model", "language_model"]);

function canonicalModulePath(value) {
  const parts = String(value || "").split(".").filter(Boolean);
  while (parts.length > 0 && PATH_WRAPPERS.has(parts[0])) parts.shift();
  if (parts[0] === "layers" || parts[0] === "text_decoder") parts[0] = "decoder";
  if (parts[0] === "visual" || parts[0] === "vision") parts[0] = "vision_tower";
  return parts.join(".");
}

function walkSpec(node, visit) {
  visit(node);
  for (const child of node.children || []) walkSpec(child, visit);
}

function flattenSkeleton(node, out = []) {
  out.push(node);
  for (const child of node.children || []) flattenSkeleton(child, out);
  return out;
}

function bindTruthToTemplate(network, skeleton) {
  const trieNodes = flattenSkeleton(skeleton).filter((n) => n.params > 0);
  const truthByPath = new Map();
  for (const trieNode of trieNodes) {
    const key = canonicalModulePath(trieNode.id);
    if (!truthByPath.has(key)) truthByPath.set(key, []);
    truthByPath.get(key).push(trieNode);
  }
  const templateNodes = [];
  walkSpec(network, (node) => {
    templateNodes.push({ node, path: canonicalModulePath(node.id) });
  });

  const used = new Set();
  const boundIds = [];
  const ambiguous = [];
  for (const { node, path } of templateNodes) {
    const candidates = (truthByPath.get(path) || []).filter((candidate) => !used.has(candidate));
    if (candidates.length > 1) {
      ambiguous.push({ template: node.id, candidates: candidates.map((candidate) => candidate.id) });
      continue;
    }
    const [best] = candidates;
    if (best) {
      used.add(best);
      node.params = best.params;
      node.weight_shapes = best.weight_shapes;
      node.dtype = best.dtype;
      node.tensor_names = best.tensor_names;
      node.value_source = "checkpoint";
      if (best.weight_dtypes && Object.keys(best.weight_dtypes).length) {
        node.attributes = { ...(node.attributes || {}), weight_dtypes: best.weight_dtypes };
      }
      boundIds.push(best.id);
    }
  }

  const gaps = trieNodes.filter((n) => !used.has(n)).map((n) => n.id);
  return { boundIds, gaps, used, ambiguous };
}

/** trie 骨架节点 → 模板 spec 形态（materializer 可直接消费）。 */
function skeletonToSpec(node) {
  return {
    kind: "module",
    id: node.id,
    name: node.name,
    type: node.type,
    repeat: node.repeat,
    attributes:
      node.weight_dtypes && Object.keys(node.weight_dtypes).length
        ? { weight_dtypes: node.weight_dtypes }
        : {},
    children: (node.children || []).map(skeletonToSpec),
    params: node.params,
    weight_shapes: node.weight_shapes,
    dtype: node.dtype,
    value_source: "checkpoint",
    tensor_names: node.tensor_names,
  };
}

/** 保留未绑定真值的层级与 repeat，已被模板消费的叶节点不重复插入。 */
function skeletonGapsToSpec(node, used) {
  const children = (node.children || []).map((child) => skeletonGapsToSpec(child, used)).filter(Boolean);
  const ownGap = node.params > 0 && !used.has(node);
  if (!ownGap && children.length === 0) return null;
  const spec = skeletonToSpec(node);
  spec.children = children;
  if (!ownGap) {
    spec.params = 0;
    spec.weight_shapes = {};
    spec.tensor_names = [];
    spec.dtype = null;
    spec.attributes = {};
  }
  return spec;
}

/**
 * 合并真值与模板语义。
 * @param {object} network 模板网络（spec 树，可能被就地注入真值）
 * @param {object|null} truth fetchCheckpointTruth 的返回值
 * @param {{hasTemplate: boolean, modelName: string, canonicalArchitecture: string|null}} ctx
 * @returns {{network: object, diagnostics: object}}
 */
export function enrichNetworkWithTruth(network, truth, { hasTemplate, modelName, canonicalArchitecture }) {
  if (!truth || !Array.isArray(truth.tensors) || truth.tensors.length === 0) {
    return { network, diagnostics: { strategy: "no-truth" } };
  }
  const skeleton = buildSkeleton(truth.tensors);

  if (!hasTemplate) {
    const spec = skeletonToSpec(skeleton);
    // 空路径根（多顶层段，如 model.* + lm_head.*）仅作容器时展开，避免多余包装层
    const children = spec.params === 0 && spec.children.length > 0 ? spec.children : [spec];
    const skeletonNetwork = {
      kind: "network",
      id: "skeleton",
      name: modelName || canonicalArchitecture || "Model",
      canonicalArchitecture,
      children,
    };
    return {
      network: skeletonNetwork,
      diagnostics: {
        strategy: "skeleton-truth",
        total_tensors: truth.tensors.length,
        parameter_total: truth.parameterTotal ?? null,
      },
    };
  }

  const { boundIds, gaps, used, ambiguous } = bindTruthToTemplate(network, skeleton);
  const gapTree = skeletonGapsToSpec(skeleton, used);
  if (gapTree) {
    network.children.push({
      kind: "module",
      id: "checkpoint_gaps",
      name: "Checkpoint extra modules",
      type: "checkpoint-gaps",
      attributes: { class: "CheckpointExtraModules" },
      children: gapTree.id ? [gapTree] : gapTree.children,
    });
  }
  return {
    network,
    diagnostics: {
      strategy: "template+truth",
      bound_tensors: boundIds.length,
      total_tensors: truth.tensors.length,
      template_gaps: gaps,
      ambiguous_truth_matches: ambiguous,
    },
  };
}
