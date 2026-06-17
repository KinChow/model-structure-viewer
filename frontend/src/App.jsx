import { useEffect, useMemo, useState } from "react";
import Header from "./components/Header";
import Tabs from "./components/Tabs";
import SummaryChips from "./components/SummaryChips";
import ArchitectureTab from "./components/ArchitectureTab";
import LayersTab from "./components/LayersTab";
import ExportTab from "./components/ExportTab";
import RawConfigTab from "./components/RawConfigTab";
import Drawer from "./components/Drawer";
import StructureSearchBox from "./components/StructureSearchBox";
import NodeDetailPanel from "./components/NodeDetailPanel";
import { useSettings } from "./hooks/useSettings";
import { useLocalModels } from "./hooks/useLocalModels";
import { useHfSearch } from "./hooks/useHfSearch";
import { useStructure } from "./hooks/useStructure";
import { useExport } from "./hooks/useExport";
import { computeMatches } from "./diagram/match";

const TAB_ITEMS = ["Architecture", "Layers", "Export", "Raw Config"];

function findNodeByPath(root, path) {
  if (!root || !path) return null;
  if (path === "root") return root;
  const parts = path.split(".").slice(1);
  let current = root;
  for (const part of parts) {
    const idx = Number(part);
    if (!current.children || Number.isNaN(idx) || idx >= current.children.length) return null;
    current = current.children[idx];
  }
  return current;
}

function hasChildren(node) {
  return node.children?.length > 0;
}

function collectCollapsiblePaths(root) {
  const paths = new Set();
  function visit(node, path) {
    if (hasChildren(node)) paths.add(path);
    node.children?.forEach((child, index) => {
      visit(child, `${path}.${index}`);
    });
  }
  if (root) visit(root, "root");
  return paths;
}

function parentPath(path) {
  const idx = path.lastIndexOf(".");
  return idx === -1 ? null : path.slice(0, idx);
}

function ancestorCollapsiblePaths(root, path) {
  const paths = [];
  let current = parentPath(path);
  while (current) {
    const node = findNodeByPath(root, current);
    if (node && hasChildren(node)) paths.push(current);
    current = parentPath(current);
  }
  return paths;
}

function App() {
  const { settings, setSettings, save: saveSettings, error: settingsError } = useSettings();
  const { models, refresh: refreshModels } = useLocalModels();
  const hf = useHfSearch();
  const { structure, build, loading, error: structureError } = useStructure();
  const exporter = useExport();

  const [source, setSource] = useState("auto");
  const [modelId, setModelId] = useState("deepseek-ai/DeepSeek-V3.1");
  const [selectedConfigPath, setSelectedConfigPath] = useState("");
  const [revision, setRevision] = useState("main");
  const [configText, setConfigText] = useState("");
  const [activeTab, setActiveTab] = useState("Architecture");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [parseError, setParseError] = useState("");
  const [selectedNodePath, setSelectedNodePath] = useState(null);
  const [layersExpandedPaths, setLayersExpandedPaths] = useState(() => new Set());
  const [searchTerm, setSearchTerm] = useState("");

  const error = parseError || structureError || hf.error || settingsError || exporter.error;
  const sourceLabel = structure?.source?.kind || "not loaded";
  const rawJson = useMemo(() => (structure ? JSON.stringify(structure, null, 2) : ""), [structure]);
  const allCollapsiblePaths = useMemo(
    () => collectCollapsiblePaths(structure?.root),
    [structure]
  );

  const searchActive = Boolean(searchTerm.trim());
  const matchedPaths = useMemo(
    () => computeMatches(structure?.root, searchTerm),
    [structure, searchTerm]
  );

  useEffect(() => {
    if (!searchActive || matchedPaths.size === 0 || !structure) return;
    const toAdd = [];
    matchedPaths.forEach((path) => {
      ancestorCollapsiblePaths(structure.root, path).forEach((collapsiblePath) => toAdd.push(collapsiblePath));
    });
    if (toAdd.length === 0) return;
    setLayersExpandedPaths((prev) => {
      const next = new Set(prev);
      let changed = false;
      toAdd.forEach((collapsiblePath) => {
        if (!next.has(collapsiblePath)) {
          next.add(collapsiblePath);
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [matchedPaths, searchActive, structure]);

  const selectedNode = useMemo(
    () => (selectedNodePath && structure ? findNodeByPath(structure.root, selectedNodePath) : null),
    [structure, selectedNodePath]
  );

  function handleSelectNode(path) {
    setSelectedNodePath(path);
    setDrawerOpen(false);
  }

  function handleSourceChange(value) {
    setSource(value);
    setSelectedConfigPath("");
  }

  function handleModelIdChange(value) {
    setModelId(value);
    setSelectedConfigPath("");
  }

  function handleToggleLayerPath(path) {
    setLayersExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function handleExpandAllLayers() {
    setLayersExpandedPaths(new Set(allCollapsiblePaths));
  }

  function handleCollapseAllLayers() {
    setLayersExpandedPaths(new Set());
  }

  function handleOpenDrawer() {
    setDrawerOpen((prev) => {
      const opening = !prev;
      if (opening) setSelectedNodePath(null);
      return opening;
    });
  }

  async function handleGenerate() {
    setParseError("");
    let configJson = null;
    if (source === "config") {
      try {
        configJson = JSON.parse(configText);
      } catch (err) {
        setParseError(err.message);
        return;
      }
    }
    const payload = {
      source,
      model_id: source === "config" || selectedConfigPath ? null : modelId.trim(),
      config_path: source === "config" ? null : selectedConfigPath || null,
      config_json: configJson,
      revision,
      cache_policy: settings.offline ? "offline" : settings.cache_policy,
      model_root: settings.model_root,
      hf_endpoint: settings.hf_endpoint,
      offline: settings.offline,
      detail_level: "compressed",
    };
    const data = await build(payload);
    if (data) {
      exporter.reset();
      setActiveTab("Architecture");
      setZoom(1);
      setSelectedNodePath(null);
      setLayersExpandedPaths(new Set());
      setSearchTerm("");
    }
  }

  async function handleSaveSettings() {
    if (await saveSettings()) {
      await refreshModels();
    }
  }

  return (
    <main className="app-shell">
      <Header
        sourceLabel={sourceLabel}
        source={source}
        onSourceChange={handleSourceChange}
        modelId={modelId}
        onModelIdChange={handleModelIdChange}
        cachePolicy={settings.cache_policy}
        onCachePolicyChange={(value) => setSettings({ ...settings, cache_policy: value })}
        onOpenDrawer={handleOpenDrawer}
        onGenerate={handleGenerate}
        loading={loading}
      />

      {error && <div className="error">{error}</div>}

      <section className="content">
        <section className="hero-panel">
          <SummaryChips structure={structure} sourceLabel={sourceLabel} />
          {structure && (
            <StructureSearchBox
              value={searchTerm}
              onChange={setSearchTerm}
              hitCount={matchedPaths.size}
            />
          )}
          <Tabs items={TAB_ITEMS} active={activeTab} onChange={setActiveTab} />
          {activeTab === "Architecture" && (
            <ArchitectureTab
              structure={structure}
              zoom={zoom}
              onZoomChange={setZoom}
              selectedPath={selectedNodePath}
              matchedPaths={matchedPaths}
              expandedGroups={allCollapsiblePaths}
              searchActive={searchActive}
              hitCount={matchedPaths.size}
              onSelectNode={handleSelectNode}
            />
          )}
          {activeTab === "Layers" && (
            <LayersTab
              structure={structure}
              selectedPath={selectedNodePath}
              matchedPaths={matchedPaths}
              expandedGroups={layersExpandedPaths}
              searchActive={searchActive}
              onSelectNode={handleSelectNode}
              onToggleGroup={handleToggleLayerPath}
              onExpandAllGroups={handleExpandAllLayers}
              onCollapseAllGroups={handleCollapseAllLayers}
            />
          )}
          {activeTab === "Export" && (
            <ExportTab
              format={exporter.format}
              onFormatChange={exporter.setFormat}
              text={exporter.text}
              onRun={() => exporter.run(structure)}
            />
          )}
          {activeTab === "Raw Config" && <RawConfigTab rawJson={rawJson} />}
        </section>

        {selectedNode ? (
          <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNodePath(null)} />
        ) : (
          <Drawer
            open={drawerOpen}
            revision={revision}
            onRevisionChange={setRevision}
            configText={configText}
            onConfigTextChange={setConfigText}
            models={models}
            onRefreshModels={refreshModels}
            onPickLocalModel={(entry) => {
              setModelId(entry.model_id);
              setSelectedConfigPath(entry.load_by === "config_path" ? entry.config_path : "");
              setSource("local");
              setDrawerOpen(false);
            }}
            searchQuery={hf.query}
            onSearchQueryChange={hf.setQuery}
            searchResults={hf.results}
            onSearch={hf.search}
            searchDisabled={settings.offline}
            onPickHfModel={(id) => {
              setModelId(id);
              setSelectedConfigPath("");
              setSource("hf");
              setDrawerOpen(false);
            }}
            settings={settings}
            onSettingsChange={setSettings}
            onSaveSettings={handleSaveSettings}
          />
        )}
      </section>
    </main>
  );
}

export default App;
