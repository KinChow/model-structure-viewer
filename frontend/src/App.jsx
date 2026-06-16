import { useEffect, useMemo, useState } from "react";

const DEFAULT_SETTINGS = {
  model_root: "/Users/zhouzijian01/Desktop/workspace/models",
  hf_endpoint: "https://huggingface.co",
  cache_policy: "prefer-local",
  offline: false,
};

const TAB_ITEMS = ["Architecture", "Layers", "Export", "Raw Config"];
const SOURCE_ITEMS = ["auto", "local", "hf", "config"];

async function requestJson(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(payload?.detail || `HTTP ${response.status}`);
  }
  return payload;
}

function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [models, setModels] = useState([]);
  const [source, setSource] = useState("auto");
  const [modelId, setModelId] = useState("deepseek-ai/DeepSeek-V3.1");
  const [revision, setRevision] = useState("main");
  const [configText, setConfigText] = useState("");
  const [structure, setStructure] = useState(null);
  const [searchQuery, setSearchQuery] = useState("DeepSeek-V3.1");
  const [searchResults, setSearchResults] = useState([]);
  const [exportFormat, setExportFormat] = useState("mermaid");
  const [exportText, setExportText] = useState("");
  const [activeTab, setActiveTab] = useState("Architecture");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    refreshSettings();
    refreshModels();
  }, []);

  async function refreshSettings() {
    try {
      const data = await requestJson("/api/settings");
      setSettings({
        model_root: data.model_root,
        hf_endpoint: data.hf_endpoint,
        cache_policy: data.cache_policy,
        offline: data.offline,
      });
    } catch {
      setSettings(DEFAULT_SETTINGS);
    }
  }

  async function refreshModels() {
    try {
      setModels(await requestJson("/api/models"));
    } catch {
      setModels([]);
    }
  }

  async function saveSettings() {
    setError("");
    try {
      const data = await requestJson("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      setSettings({
        model_root: data.model_root,
        hf_endpoint: data.hf_endpoint,
        cache_policy: data.cache_policy,
        offline: data.offline,
      });
      await refreshModels();
    } catch (err) {
      setError(err.message);
    }
  }

  async function searchHf() {
    if (!searchQuery.trim()) return;
    setError("");
    setLoading(true);
    try {
      const params = new URLSearchParams({ q: searchQuery.trim(), limit: "10" });
      setSearchResults(await requestJson(`/api/hf/search?${params.toString()}`));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function buildStructure() {
    setError("");
    setExportText("");
    setLoading(true);
    try {
      let config_json = null;
      if (source === "config") {
        config_json = JSON.parse(configText);
      }
      const payload = {
        source,
        model_id: source === "config" ? null : modelId.trim(),
        config_json,
        revision,
        cache_policy: settings.offline ? "offline" : settings.cache_policy,
        model_root: settings.model_root,
        hf_endpoint: settings.hf_endpoint,
        offline: settings.offline,
        detail_level: "compressed",
      };
      const data = await requestJson("/api/structure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setStructure(data);
      setActiveTab("Architecture");
      setZoom(1);
    } catch (err) {
      setStructure(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function exportCurrent(format = exportFormat) {
    if (!structure) return;
    setError("");
    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ structure, format }),
      });
      const text = await response.text();
      if (!response.ok) throw new Error(text);
      setExportText(text);
    } catch (err) {
      setError(err.message);
    }
  }

  function downloadSvg() {
    const svg = document.querySelector(".diagram-svg");
    if (!svg) return;
    const blob = new Blob([svg.outerHTML], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${structure?.summary?.model_family || "model"}-structure.svg`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const sourceLabel = structure?.source?.kind || "not loaded";
  const rawJson = useMemo(() => (structure ? JSON.stringify(structure, null, 2) : ""), [structure]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <h1>Model Structure Viewer</h1>
          <span>{sourceLabel}</span>
        </div>
        <div className="input-bar">
          <select value={source} onChange={(event) => setSource(event.target.value)} aria-label="source">
            {SOURCE_ITEMS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          {source === "config" ? (
            <input
              value="Pasted config JSON"
              readOnly
              aria-label="config source"
              onClick={() => setDrawerOpen(true)}
            />
          ) : (
            <input
              value={modelId}
              onChange={(event) => setModelId(event.target.value)}
              placeholder="deepseek-ai/DeepSeek-V3.1"
              aria-label="model id"
            />
          )}
          <select
            value={settings.cache_policy}
            onChange={(event) => setSettings({ ...settings, cache_policy: event.target.value })}
            aria-label="cache policy"
          >
            <option value="prefer-local">prefer-local</option>
            <option value="refresh">refresh</option>
            <option value="offline">offline</option>
          </select>
          <button className="primary" onClick={buildStructure} disabled={loading}>
            {loading ? "Generating" : "Generate"}
          </button>
          <button className="icon-button" onClick={() => setDrawerOpen(!drawerOpen)} title="Settings and search">
            ⚙
          </button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="content">
        <section className="hero-panel">
          <SummaryChips structure={structure} sourceLabel={sourceLabel} />
          <div className="tabs" role="tablist">
            {TAB_ITEMS.map((item) => (
              <button
                key={item}
                className={activeTab === item ? "active" : ""}
                onClick={() => setActiveTab(item)}
                role="tab"
              >
                {item}
              </button>
            ))}
          </div>
          {activeTab === "Architecture" && (
            <section className="diagram-panel">
              <div className="panel-toolbar">
                <h2>Architecture</h2>
                <div className="toolbar-actions">
                  <button onClick={() => setZoom((value) => Math.max(0.7, value - 0.1))}>−</button>
                  <button onClick={() => setZoom(1)}>Fit</button>
                  <button onClick={() => setZoom((value) => Math.min(1.4, value + 0.1))}>+</button>
                  <button onClick={downloadSvg} disabled={!structure}>
                    SVG
                  </button>
                </div>
              </div>
              {structure ? (
                <StructureDiagram structure={structure} zoom={zoom} />
              ) : (
                <EmptyState />
              )}
            </section>
          )}
          {activeTab === "Layers" && (
            <section className="layers-panel">
              {structure ? <ModuleCards node={structure.root} depth={0} /> : <EmptyState />}
            </section>
          )}
          {activeTab === "Export" && (
            <section className="export-panel">
              <div className="panel-toolbar">
                <h2>Export</h2>
                <div className="toolbar-actions">
                  <select value={exportFormat} onChange={(event) => setExportFormat(event.target.value)}>
                    <option value="mermaid">Mermaid</option>
                    <option value="dot">DOT</option>
                    <option value="json">JSON</option>
                  </select>
                  <button onClick={() => exportCurrent()}>Export</button>
                </div>
              </div>
              <textarea readOnly value={exportText} placeholder="Export output appears here." />
            </section>
          )}
          {activeTab === "Raw Config" && (
            <section className="export-panel">
              <textarea readOnly value={rawJson} placeholder="Generate a structure to inspect raw API output." />
            </section>
          )}
        </section>

        <aside className={`drawer ${drawerOpen ? "open" : ""}`}>
          <section>
            <h2>Inputs</h2>
            <label>
              Revision
              <input value={revision} onChange={(event) => setRevision(event.target.value)} />
            </label>
            <label>
              Config JSON
              <textarea
                value={configText}
                onChange={(event) => setConfigText(event.target.value)}
                placeholder='{"model_type": "..."}'
              />
            </label>
          </section>
          <section>
            <h2>Local Models</h2>
            <button onClick={refreshModels}>Refresh</button>
            <div className="compact-list">
              {models.map((entry) => (
                <button
                  key={entry.model_id}
                  onClick={() => {
                    setModelId(entry.model_id);
                    setSource("auto");
                    setDrawerOpen(false);
                  }}
                >
                  {entry.model_id}
                </button>
              ))}
            </div>
          </section>
          <section>
            <h2>Hugging Face Search</h2>
            <div className="inline">
              <input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} />
              <button onClick={searchHf} disabled={settings.offline}>
                Search
              </button>
            </div>
            <div className="compact-list">
              {searchResults.map((item) => (
                <button
                  key={item.model_id}
                  onClick={() => {
                    setModelId(item.model_id);
                    setSource("hf");
                    setDrawerOpen(false);
                  }}
                >
                  <strong>{item.model_id}</strong>
                  <span>{item.pipeline_tag || "unknown"}</span>
                </button>
              ))}
            </div>
          </section>
          <section>
            <h2>Settings</h2>
            <label>
              Model root
              <input
                value={settings.model_root}
                onChange={(event) => setSettings({ ...settings, model_root: event.target.value })}
              />
            </label>
            <label>
              HF endpoint
              <input
                value={settings.hf_endpoint}
                onChange={(event) => setSettings({ ...settings, hf_endpoint: event.target.value })}
              />
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={settings.offline}
                onChange={(event) => setSettings({ ...settings, offline: event.target.checked })}
              />
              Offline
            </label>
            <button onClick={saveSettings}>Save settings</button>
          </section>
        </aside>
      </section>
    </main>
  );
}

function SummaryChips({ structure, sourceLabel }) {
  const summary = structure?.summary || {};
  const chips = [
    ["Model", summary.model_family || summary.model_type],
    ["Architecture", summary.architecture],
    ["Layers", summary.text_layers],
    ["Hidden", summary.hidden_size],
    ["Heads", summary.num_attention_heads],
    ["Experts", summary.num_local_experts],
    ["Context", summary.max_position_embeddings],
    ["Source", sourceLabel],
  ];
  return (
    <div className="summary-chips">
      {chips.map(([label, value]) => (
        <span className="chip" key={label}>
          <b>{label}</b>
          {value ?? "-"}
        </span>
      ))}
    </div>
  );
}

function StructureDiagram({ structure, zoom }) {
  const nodes = useMemo(() => layoutDiagram(structure.root), [structure]);
  const width = Math.max(960, ...nodes.map((node) => node.x + node.width + 40));
  const height = Math.max(430, ...nodes.map((node) => node.y + node.height + 40));
  return (
    <div className="diagram-scroll">
      <svg
        className="diagram-svg"
        viewBox={`0 0 ${width} ${height}`}
        width={width * zoom}
        height={height * zoom}
        role="img"
        aria-label="Model architecture diagram"
      >
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#7b8794" />
          </marker>
        </defs>
        {nodes.flatMap((node) =>
          node.children.map((child) => {
            const target = nodes.find((item) => item.path === child);
            if (!target) return null;
            return (
              <path
                key={`${node.path}-${child}`}
                d={`M ${node.x + node.width} ${node.y + node.height / 2} C ${node.x + node.width + 36} ${
                  node.y + node.height / 2
                }, ${target.x - 36} ${target.y + target.height / 2}, ${target.x} ${target.y + target.height / 2}`}
                fill="none"
                stroke="#8a96a5"
                strokeWidth="1.5"
                markerEnd="url(#arrow)"
              />
            );
          })
        )}
        {nodes.map((node) => (
          <g key={node.path} transform={`translate(${node.x}, ${node.y})`}>
            <rect
              width={node.width}
              height={node.height}
              rx="8"
              className={`diagram-node ${node.typeClass}`}
            />
            <text x="14" y="24" className="diagram-title">
              {truncate(node.name, 30)}
            </text>
            <text x="14" y="43" className="diagram-meta">
              {node.meta}
            </text>
            {node.repeat && (
              <text x={node.width - 16} y="24" textAnchor="end" className="diagram-repeat">
                x{node.repeat}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

function ModuleCards({ node, depth }) {
  return (
    <article className={`module-card depth-${Math.min(depth, 4)} type-${typeClass(node.type)}`}>
      <header>
        <div>
          <h3>{node.name}</h3>
          <span>{node.type}</span>
        </div>
        {node.repeat && <b>x {node.repeat}</b>}
      </header>
      <AttributeGrid attributes={node.attributes} sourceFields={node.source_fields} />
      {node.children?.length > 0 && (
        <div className="module-children">
          {node.children.map((child) => (
            <ModuleCards key={`${node.id}-${child.id}-${child.name}`} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </article>
  );
}

function AttributeGrid({ attributes, sourceFields }) {
  const entries = Object.entries(attributes || {}).slice(0, 12);
  return (
    <div className="attribute-grid">
      {entries.map(([key, value]) => (
        <span key={key}>
          <em>{key}</em>
          <strong>{formatValue(value)}</strong>
        </span>
      ))}
      {sourceFields?.length > 0 && (
        <span className="wide">
          <em>fields</em>
          <strong>{sourceFields.join(", ")}</strong>
        </span>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <strong>Generate a model structure to show the diagram.</strong>
      <span>Try deepseek-ai/DeepSeek-V3.1 or MiniMaxAI/MiniMax-M3.</span>
    </div>
  );
}

function layoutDiagram(root) {
  const levels = [];
  const paths = new Map();
  function visit(node, depth, path) {
    if (!levels[depth]) levels[depth] = [];
    const item = {
      node,
      path,
      depth,
      children: [],
      width: 210,
      height: 70,
      typeClass: typeClass(node.type),
      repeat: node.repeat,
      name: node.name,
      meta: metaForNode(node),
    };
    levels[depth].push(item);
    paths.set(path, item);
    node.children?.forEach((child, index) => {
      const childPath = `${path}.${index}`;
      item.children.push(childPath);
      visit(child, depth + 1, childPath);
    });
  }
  visit(root, 0, "root");
  const result = [];
  levels.forEach((level, depth) => {
    const totalHeight = level.length * 92;
    const startY = Math.max(26, (430 - totalHeight) / 2 + 26);
    level.forEach((item, index) => {
      item.x = 28 + depth * 260;
      item.y = startY + index * 92;
      result.push(item);
    });
  });
  return result;
}

function metaForNode(node) {
  const attrs = node.attributes || {};
  const pieces = [];
  if (attrs.shape) pieces.push(attrs.shape);
  if (attrs.hidden_size) pieces.push(`hidden ${attrs.hidden_size}`);
  if (attrs.num_attention_heads || attrs.attention_heads) pieces.push(`heads ${attrs.num_attention_heads || attrs.attention_heads}`);
  if (attrs.routed_experts || attrs.num_local_experts) pieces.push(`experts ${attrs.routed_experts || attrs.num_local_experts}`);
  if (attrs.range) pieces.push(attrs.range);
  return pieces.slice(0, 2).join(" · ") || node.type;
}

function typeClass(type) {
  if (type.includes("embedding")) return "embedding";
  if (type.includes("attention")) return "attention";
  if (type.includes("moe")) return "moe";
  if (type.includes("mlp")) return "mlp";
  if (type.includes("projector")) return "projector";
  if (type.includes("output") || type.includes("head") || type.includes("mtp")) return "output";
  if (type.includes("vision")) return "vision";
  if (type.includes("layer") || type.includes("module")) return "layer";
  return "model";
}

function formatValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function truncate(value, limit) {
  if (!value) return "";
  return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
}

export default App;
