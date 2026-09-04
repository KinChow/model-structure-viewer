import { useEffect, useMemo, useState } from "react";
import DetailWorkspace from "./components/DetailWorkspace";
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
  const { models: builtinModels, refresh: refreshBuiltinModels } = useBuiltinModels();
  const { models, refresh: refreshModels } = useLocalModels();
  const hf = useHfSearch();
  const { structure, build, loading, error: structureError } = useStructure();
  const exporter = useExport();

  const [source, setSource] = useState("auto");
  const [endpoint, setEndpoint] = useState("huggingface");
  const [modelId, setModelId] = useState("deepseek-ai/DeepSeek-V3.1");
  const [selectedConfigPath, setSelectedConfigPath] = useState("");
  const [revision, setRevision] = useState("main");
  const [configText, setConfigText] = useState("");
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
    () => collectCollapsiblePaths(structure?.root),
    [structure]
  );

  const searchActive = Boolean(searchTerm.trim());
  const matchedPaths = useMemo(
    () => computeMatches(structure?.root, searchTerm),
    [structure, searchTerm]
  );
  const matchResults = useMemo(
    () => [...matchedPaths].slice(0, 12).map((path) => {
      const node = findNodeByPath(structure?.root, path);
      return { path, name: node?.name || path, type: node?.type || "node" };
    }),
    [matchedPaths, structure]
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
    () => (selectedNodePath && structure ? { node: findNodeByPath(structure.root, selectedNodePath), path: selectedNodePath } : null),
    [structure, selectedNodePath]
  );

  function handleSelectNode(path) {
    setSelectedNodePath(path);
    setDrawerOpen(false);
    if (!path || !structure) return;
    const ancestors = ancestorCollapsiblePaths(structure.root, path);
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
    const activeSource = overrides.source ?? source;
    const activeModelId = overrides.modelId ?? modelId;
    const activeConfigPath = overrides.configPath ?? selectedConfigPath;
    const activeEndpoint = overrides.endpoint ?? endpoint;
    const checkpointTruth = overrides.checkpointTruth ?? null;
    const sourceLabelOverride = overrides.sourceLabel ?? null;
    if (activeSource === "config") {
      try {
        configJson = overrides.configJson ?? JSON.parse(configText);
      } catch (err) {
        setParseError(err.message);
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
          onOpenModel={(id, selectedSource = "auto", selectedEndpoint = endpoint) => {
            void handleGenerate({ source: selectedSource, modelId: id, endpoint: selectedEndpoint });
          }}
          onOpenLocalFiles={handleOpenLocalFiles}
          onOpenLocalPath={(path) => {
            setSource("local");
            setSelectedConfigPath(path);
            void handleGenerate({ source: "local", configPath: path });
          }}
          language={language}
          onLanguageChange={handleLanguageChange}
          theme={theme}
          onThemeChange={handleThemeChange}
          loading={loading}
        />
        {error && <div className="error entry-error" role="alert">{error}</div>}
      </main>
    );
  }

  if (structure) {
    return (
      <>
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
        />
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
