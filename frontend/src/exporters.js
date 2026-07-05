function label(node) {
  const parts = [node.name];
  if (node.repeat) parts.push(`x${node.repeat}`);
  if (node.type) parts.push(`(${node.type})`);
  return parts.join(" ");
}

function slug(value) {
  const clean = String(value).replace(/[^A-Za-z0-9_]/g, "_").replace(/^_+|_+$/g, "");
  if (!clean) return "";
  return /^\d/.test(clean) ? `n_${clean}` : clean;
}

function fallbackId(path) {
  let hash = 0;
  for (let index = 0; index < path.length; index += 1) {
    hash = (hash * 31 + path.charCodeAt(index)) >>> 0;
  }
  return hash.toString(16).slice(0, 6);
}

function walkTree(root, handlers) {
  const used = new Set();
  function assignId(path) {
    const base = slug(path) || `n_${fallbackId(path)}`;
    if (!used.has(base)) {
      used.add(base);
      return base;
    }
    const disambiguated = `${base}_${fallbackId(path)}`;
    used.add(disambiguated);
    return disambiguated;
  }

  function visit(node, parentId, path) {
    const nodeId = assignId(path || node.id);
    handlers.onNode(nodeId, node);
    if (parentId) handlers.onEdge(parentId, nodeId);
    for (const [index, child] of (node.children || []).entries()) {
      const childPath = path ? `${path}.${index}.${child.id}` : `${node.id}.${index}.${child.id}`;
      visit(child, nodeId, childPath);
    }
  }

  visit(root, null, "");
}

function escapeMermaid(value) {
  return String(value)
    .replaceAll("\\", "\\\\")
    .replaceAll('"', '\\"')
    .replaceAll("|", "\\|")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeDot(value) {
  return String(value).replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

function exportMermaid(structure) {
  const lines = ["flowchart TD"];
  walkTree(structure.root, {
    onNode: (id, node) => lines.push(`  ${id}["${escapeMermaid(label(node))}"]`),
    onEdge: (source, target) => lines.push(`  ${source} --> ${target}`),
  });
  return `${lines.join("\n")}\n`;
}

function exportDot(structure) {
  const lines = [
    "digraph ModelStructure {",
    "  rankdir=LR;",
    '  node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#64748b", fontname="Helvetica"];',
    '  edge [color="#64748b"];',
  ];
  walkTree(structure.root, {
    onNode: (id, node) => lines.push(`  ${id} [label="${escapeDot(label(node))}"];`),
    onEdge: (source, target) => lines.push(`  ${source} -> ${target};`),
  });
  lines.push("}");
  return `${lines.join("\n")}\n`;
}

export function exportStructure(structure, format) {
  if (format === "json") return `${JSON.stringify(structure, null, 2)}\n`;
  if (format === "mermaid") return exportMermaid(structure);
  if (format === "dot") return exportDot(structure);
  throw new Error(`Unsupported export format: ${format}`);
}
