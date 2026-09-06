import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import Drawer from "./components/Drawer";
import ModelEntry from "./components/ModelEntry";
import { useSettings } from "./hooks/useSettings";
import { useBuiltinModels } from "./hooks/useBuiltinModels";
import { useLocalModels } from "./hooks/useLocalModels";
import { useHfSearch } from "./hooks/useHfSearch";
import { useStructure } from "./hooks/useStructure";
import { useExport } from "./hooks/useExport";
import { computeMatches } from "./diagram/match";
import { PUBLIC_CHIPS } from "./cost/chips/public.js";
import { loadLocalChipOverrides, mergeChipCatalog } from "./cost/chips/loadLocal.js";
import { readLocalSafetensorsHeaders } from "./cost/safetensorsReader.js";
import { graphChildren, graphViewNode } from "./structure/graph/selectors.js";

const DetailWorkspace = lazy(() => import("./components/DetailWorkspace"));

function DetailWorkspaceFallback({ theme, language }) {
  return <div className={`detail-page theme-${theme}`}><div className="detail-loading-overlay" role="status" aria-live="polite"><span className="detail-loading-dot" aria-hidden="true" /><span>{language === "en" ? "Preparing model view" : "正在准备模型视图"}<i aria-hidden="true">...</i></span></div></div>;
}

function findNodeByPath(source, path) {
  if (!source || !path) return null;
  if (Array.isArray(source.nodes)) return graphViewNode(source, path);
  if (path === "root") return source;
  const parts = path.split(".").slice(1);
  let current = source;
  for (const part of parts) {
    const idx = Number(part);
    if (!current.children || Number.isNaN(idx) || idx >= current.children.length) return null;
    current = current.children[idx];
  }
  return current;
}

function hasChildren(node, graph) {
  if (graph?.nodes) return graphChildren(graph, node.path || node.id).length > 0;
  return node.children?.length > 0;
}

function collectCollapsiblePaths(root, graph = null) {
  const paths = new Set();
  function visit(node, path) {
    if (hasChildren(node, graph)) paths.add(path);
    if (graph?.nodes) {
      graphChildren(graph, path).forEach((child) => visit(child, child.id));
      return;
    }
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

function ancestorCollapsiblePaths(root, path, graph = null) {
  const paths = [];
  let current = parentPath(path);
  while (current) {
    const node = findNodeByPath(graph || root, current);
    if (node && hasChildren(node, graph)) paths.push(current);
    current = parentPath(current);
  }
  return paths;
}

function App() {
  const { settings, setSettings, save: saveSettings, error: settingsError } = useSettings();
  const { models: builtinModels, refresh: refreshBuiltinModels } = useBuiltinModels();
  const { models, refresh: refreshModels } = useLocalModels();
  const hf = useHfSearch();
  const { structure, build, loading, loadingPhase, error: structureError } = useStructure();
  const exporter = useExport();

  const [modelId, setModelId] = useState("deepseek-ai/DeepSeek-V3.1");
  const [revision, setRevision] = useState("main");
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [fitNonce, setFitNonce] = useState(0);
  const [parseError, setParseError] = useState("");
  const [selectedNodePath, setSelectedNodePath] = useState(null);
  const [layersExpandedPaths, setLayersExpandedPaths] = useState(() => new Set(["root"]));
  const [searchTerm, setSearchTerm] = useState("");
  const [chips, setChips] = useState(PUBLIC_CHIPS);
  const [chipError, setChipError] = useState("");
  const [language, setLanguage] = useState(() => localStorage.getItem("msv-language") || (navigator.language?.toLowerCase().startsWith("zh") ? "zh" : "en"));
  const [theme, setTheme] = useState(() => localStorage.getItem("msv-theme") || "dark");
  function handleAddChip(chip) {
    setChips((current) => [...current.filter((entry) => entry.id !== chip.id), chip]);
  }

  useEffect(() => {
    let active = true;
    loadLocalChipOverrides()
      .then((localChips) => {
        if (active && localChips.length > 0) setChips(mergeChipCatalog(PUBLIC_CHIPS, localChips));
      })
      .catch(() => {
        setChipError("chips.local.json 加载失败，请检查本地芯片配置格式");
      });
    return () => { active = false; };
  }, []);

  const error = parseError || structureError || hf.error || settingsError || exporter.error || chipError;
  const sourceLabel = structure?.source?.kind || "not loaded";
  const allCollapsiblePaths = useMemo(
    () => collectCollapsiblePaths(
      structure?.graph ? graphViewNode(structure.graph, structure.graph.root_id || "root") : structure?.root,
      structure?.graph,
    ),
    [structure]
  );

  const searchActive = Boolean(searchTerm.trim());
  const matchedPaths = useMemo(
    () => computeMatches(structure?.graph || structure?.root, searchTerm),
    [structure, searchTerm]
  );
  const matchResults = useMemo(
    () => [...matchedPaths].slice(0, 12).map((path) => {
      const node = findNodeByPath(structure?.graph || structure?.root, path);
      return { path, name: node?.name || path, type: node?.type || "node" };
    }),
    [matchedPaths, structure]
  );

  useEffect(() => {
    if (!searchActive || matchedPaths.size === 0 || !structure) return;
    const toAdd = [];
    matchedPaths.forEach((path) => {
      ancestorCollapsiblePaths(structure.graph || structure.root, path, structure.graph).forEach((collapsiblePath) => toAdd.push(collapsiblePath));
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
    () => (selectedNodePath && structure ? { node: findNodeByPath(structure.graph || structure.root, selectedNodePath), path: selectedNodePath } : null),
    [structure, selectedNodePath]
  );

  function handleSelectNode(path) {
    setSelectedNodePath(path);
    setDrawerOpen(false);
    if (!path || !structure) return;
    const ancestors = ancestorCollapsiblePaths(structure.graph || structure.root, path, structure.graph);
    if (ancestors.length > 0) {
      setLayersExpandedPaths((previous) => {
        const next = new Set(previous);
        ancestors.forEach((ancestor) => next.add(ancestor));
        return next;
      });
      setFitNonce((value) => value + 1);
    }
  }

  function handleToggleLayerPath(path) {
    setSelectedNodePath(path);
    setLayersExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
    setFitNonce((value) => value + 1);
  }

  function handleExpandAllLayers() {
    setLayersExpandedPaths(new Set(allCollapsiblePaths));
    setFitNonce((value) => value + 1);
  }

  function handleCollapseAllLayers() {
    setLayersExpandedPaths(new Set());
    setFitNonce((value) => value + 1);
  }

  function handleOpenDrawer() {
    setDrawerOpen((prev) => {
      const opening = !prev;
      if (opening) setSelectedNodePath(null);
      return opening;
    });
  }

  async function handleGenerate(overrides = {}) {
    setParseError("");
    let configJson = null;
    const activeSource = overrides.source ?? "hf";
    const activeModelId = overrides.modelId ?? modelId;
    const activeConfigPath = overrides.configPath ?? "";
    const activeEndpoint = overrides.endpoint ?? "huggingface";
    const checkpointTruth = overrides.checkpointTruth ?? null;
    const sourceLabelOverride = overrides.sourceLabel ?? null;
    if (activeSource === "config") {
      configJson = overrides.configJson ?? null;
      if (!configJson || typeof configJson !== "object") {
        setParseError("config source requires a JSON object");
        return;
      }
    }
    const payload = {
      source: activeSource,
      endpoint: activeEndpoint,
      model_id: activeSource === "config" || activeConfigPath ? null : activeModelId.trim(),
      config_path: activeSource === "config" ? null : activeConfigPath || null,
      config_json: configJson,
      checkpoint_truth: checkpointTruth,
      source_label: sourceLabelOverride,
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
      setZoom(1);
      setFitNonce((value) => value + 1);
      setSelectedNodePath(null);
      setLayersExpandedPaths(new Set(["root"]));
      setSearchTerm("");
    }
  }

  async function handleSaveSettings() {
    if (await saveSettings()) {
      await refreshModels();
    }
  }

  async function handleOpenLocalFiles(files) {
    const configFile = files.find((file) => file.name === "config.json") || files.find((file) => file.name.endsWith("config.json"));
    if (!configFile) {
      setParseError(language === "en" ? "No config.json found in the selected model directory" : "所选模型目录中没有找到 config.json");
      return;
    }
    try {
      const config = JSON.parse(await configFile.text());
      let checkpointTruth = null;
      try {
        checkpointTruth = await readLocalSafetensorsHeaders(files);
      } catch {
        checkpointTruth = null;
      }
      await handleGenerate({ source: "config", configJson: config, checkpointTruth, sourceLabel: "local directory" });
    } catch (err) {
      setParseError(err.message);
    }
  }

  function handleLanguageChange(next) {
    setLanguage(next);
    localStorage.setItem("msv-language", next);
  }

  function handleThemeChange() {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("msv-theme", next);
  }

  if (!structure) {
    return (
      <main className="app-shell">
        <ModelEntry
          builtinModels={builtinModels}
          modelId={modelId}
          onModelIdChange={setModelId}
          onOpenModel={(id, selectedSource = "hf", selectedEndpoint = "huggingface") => {
            void handleGenerate({ source: selectedSource, modelId: id, endpoint: selectedEndpoint });
          }}
          onOpenLocalFiles={handleOpenLocalFiles}
          onOpenLocalPath={(path) => {
            void handleGenerate({ source: "local", configPath: path });
          }}
          language={language}
          onLanguageChange={handleLanguageChange}
          theme={theme}
          onThemeChange={handleThemeChange}
          loading={loading}
          loadingPhase={loadingPhase}
        />
        {error && <div className="error entry-error" role="alert">{error}</div>}
      </main>
    );
  }

  if (structure) {
    return (
      <>
        <Suspense fallback={<DetailWorkspaceFallback theme={theme} language={language} />}>
          <DetailWorkspace
            structure={structure}
            sourceLabel={sourceLabel}
            language={language}
            theme={theme}
            onLanguageChange={handleLanguageChange}
            onThemeChange={handleThemeChange}
            onBack={() => window.location.reload()}
            onSettings={handleOpenDrawer}
            selectedNode={selectedNode}
            selectedNodePath={selectedNodePath}
            onSelectNode={handleSelectNode}
            onCloseNode={() => setSelectedNodePath(null)}
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
            matchResults={matchResults}
            matchedPaths={matchedPaths}
            expandedGroups={layersExpandedPaths}
            zoom={zoom}
            onZoomChange={setZoom}
            fitNonce={fitNonce}
            onFit={() => { setZoom(1); setFitNonce((value) => value + 1); }}
            chips={chips}
            onAddChip={handleAddChip}
            allCollapsiblePaths={allCollapsiblePaths}
            layersExpandedPaths={layersExpandedPaths}
            onToggleLayerPath={handleToggleLayerPath}
            onExpandAllLayers={handleExpandAllLayers}
            onCollapseAllLayers={handleCollapseAllLayers}
            exporter={exporter}
            loading={loading}
            loadingPhase={loadingPhase}
          />
        </Suspense>
        {error && <div className="error detail-error">{error}</div>}
        <Drawer
          open={drawerOpen}
          revision={revision}
          onRevisionChange={setRevision}
          language={language}
          builtinModels={builtinModels}
          onRefreshBuiltinModels={refreshBuiltinModels}
          onPickBuiltinModel={(entry) => { void handleGenerate({ source: "builtin", modelId: entry.modelId }); setDrawerOpen(false); }}
          models={models}
          onRefreshModels={refreshModels}
          onPickLocalModel={(entry) => { void handleGenerate({ source: "local", modelId: entry.model_id, configPath: entry.load_by === "config_path" ? entry.config_path : "" }); setDrawerOpen(false); }}
          searchQuery={hf.query}
          onSearchQueryChange={hf.setQuery}
          searchResults={hf.results}
          onSearch={hf.search}
          searchDisabled={settings.offline}
          onPickHfModel={(id) => { void handleGenerate({ source: "hf", modelId: id }); setDrawerOpen(false); }}
          settings={settings}
          onSettingsChange={setSettings}
          onSaveSettings={handleSaveSettings}
        />
      </>
    );
  }

}

export default App;
