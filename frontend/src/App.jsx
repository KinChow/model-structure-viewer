import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import Drawer from "./components/Drawer";
import ModelEntry from "./components/ModelEntry";
import { useSettings } from "./hooks/useSettings";
import { useBuiltinModels } from "./hooks/useBuiltinModels";
import { useLocalModels } from "./hooks/useLocalModels";
import { useHfSearch } from "./hooks/useHfSearch";
import { useStructure } from "./hooks/useStructure";
import { useVerify } from "./hooks/useVerify";
import { useExport } from "./hooks/useExport";
import { computeMatches } from "./diagram/match";
import { PUBLIC_CHIPS } from "./cost/chips/public.js";
import { loadLocalChipOverrides, mergeChipCatalog } from "./cost/chips/loadLocal.js";
import { readLocalSafetensorsHeaders } from "./cost/safetensorsReader.js";
import { graphChildren, graphViewNode } from "./structure/graph/selectors.js";
import { formatSourceLabel } from './formatters.js';
import { formatIssue } from "./i18n/format.js";

const DetailWorkspace = lazy(() => import("./components/DetailWorkspace"));

function DetailWorkspaceFallback({ theme, language }) {
  return <div className={`detail-page theme-${theme}`}><div className="detail-loading-overlay" role="status" aria-live="polite"><span className="detail-loading-dot" aria-hidden="true" /><span>{language === "en" ? "Preparing model view" : "正在准备模型视图"}<i aria-hidden="true">...</i></span></div></div>;
}

function findNodeByPath(source, path) {
  return source?.nodes && path ? graphViewNode(source, path) : null;
}

function hasChildren(node, graph) {
  return graph?.nodes ? graphChildren(graph, node.path || node.id).length > 0 : false;
}

function collectCollapsiblePaths(root, graph) {
  const paths = new Set();
  function visit(node, path) {
    if (hasChildren(node, graph)) paths.add(path);
    graphChildren(graph, path).forEach((child) => visit(child, child.id));
  }
  if (root) visit(root, "root");
  return paths;
}

function parentPath(path) {
  const idx = path.lastIndexOf(".");
  return idx === -1 ? null : path.slice(0, idx);
}

function ancestorCollapsiblePaths(graph, path) {
  const paths = [];
  let current = parentPath(path);
  while (current) {
    const node = findNodeByPath(graph, current);
    if (node && hasChildren(node, graph)) paths.push(current);
    current = parentPath(current);
  }
  return paths;
}

function App() {
  const { settings, setSettings, save: saveSettings, error: settingsError, ready: backendReady } = useSettings();
  const { models: builtinModels, refresh: refreshBuiltinModels } = useBuiltinModels();
  const { models, refresh: refreshModels } = useLocalModels();
  const hf = useHfSearch();
  const { structure, build, loading, loadingPhase, error: structureError } = useStructure();
  const { result: verifyResult, loading: verifyLoading, error: verifyError, verify, reset: resetVerify } = useVerify();
  const exporter = useExport();
  const lastVerifyPayload = useRef(null);

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
  const [backendNoticeDismissed, setBackendNoticeDismissed] = useState(false);
  function handleAddChip(chip) {
    setChips((current) => [...current.filter((entry) => entry.id !== chip.id), chip]);
  }

  useEffect(() => {
    let active = true;
    loadLocalChipOverrides()
      .then((localChips) => {
        if (active && localChips.length > 0) setChips(mergeChipCatalog(PUBLIC_CHIPS, localChips));
      })
      .catch((err) => {
        if (active) setChipError(err.issues || err.issue || { code: "chip.loadFailed" });
      });
    return () => { active = false; };
  }, []);

  const error = formatIssue(language, parseError || structureError || hf.error || settingsError || exporter.error || chipError);
  const sourceLabel = formatSourceLabel(structure?.source, language);
  const backendNotice = backendReady === false && !backendNoticeDismissed ? (
    <div className="backend-notice" role="status" aria-live="polite">
      <span>{language === "en"
        ? "Backend unavailable — only built-in models work; local directory / HF search / verify are disabled."
        : "后端不可用 —— 仅内置模型可用；本地目录 / HF 搜索 / 校验等能力暂不可用。"}</span>
      <button type="button" aria-label={language === "en" ? "Dismiss" : "关闭"} onClick={() => setBackendNoticeDismissed(true)}>×</button>
    </div>
  ) : null;
  const allCollapsiblePaths = useMemo(
    () => structure?.graph ? collectCollapsiblePaths(graphViewNode(structure.graph, structure.graph.root_id || "root"), structure.graph) : new Set(),
    [structure]
  );

  const matchedPaths = useMemo(
    () => computeMatches(structure?.graph, searchTerm),
    [structure, searchTerm]
  );
  const matchResults = useMemo(
    () => [...matchedPaths].slice(0, 12).map((path) => {
      const node = findNodeByPath(structure?.graph, path);
      return { path, name: node?.name || path, type: node?.type || "node" };
    }),
    [matchedPaths, structure]
  );

  function handleSelectNode(path) {
    setSelectedNodePath(path);
    setDrawerOpen(false);
    if (!path || !structure) return;
    const ancestors = ancestorCollapsiblePaths(structure.graph, path);
    if (ancestors.length > 0) {
      setLayersExpandedPaths((previous) => {
        const next = new Set(previous);
        ancestors.forEach((ancestor) => next.add(ancestor));
        return next;
      });
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
  }

  function handleExpandAllLayers() {
    setLayersExpandedPaths(new Set(allCollapsiblePaths));
  }

  function handleCollapseAllLayers() {
    setLayersExpandedPaths(new Set(["root"]));
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
      lastVerifyPayload.current = payload;
      resetVerify();
      exporter.reset();
      setZoom(1);
      setFitNonce((value) => value + 1);
      setSelectedNodePath(null);
      setLayersExpandedPaths(new Set(["root"]));
      setSearchTerm("");
    }
  }

  function handleVerify() {
    const base = lastVerifyPayload.current;
    if (!base || !structure?.graph) return;
    void verify({ ...base, msv_graph: structure.graph });
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
        {backendNotice}
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
        {backendNotice}
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
            selectedNodePath={selectedNodePath}
            onSelectNode={handleSelectNode}
            onCloseNode={() => setSelectedNodePath(null)}
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
            matchResults={matchResults}
            matchedPaths={matchedPaths}
            expandedGroups={layersExpandedPaths}
            zoom={zoom}
            fitNonce={fitNonce}
            onFit={() => { setZoom(1); setFitNonce((value) => value + 1); }}
            chips={chips}
            onAddChip={handleAddChip}
            onToggleLayerPath={handleToggleLayerPath}
            onExpandAllLayers={handleExpandAllLayers}
            onCollapseAllLayers={handleCollapseAllLayers}
            exporter={exporter}
            loading={loading}
            loadingPhase={loadingPhase}
            onVerify={handleVerify}
            verifyResult={verifyResult}
            verifyLoading={verifyLoading}
            verifyError={verifyError}
          />
        </Suspense>
        {error && <div className="error detail-error">{error}</div>}
        <Drawer
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
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
