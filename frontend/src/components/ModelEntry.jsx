import { useEffect, useMemo, useRef, useState } from "react";
import { staticAssetPath } from "../structure/catalog/manifest.js";

const PROVIDER_MARKS = { MiniMax: "M", Qwen: "Q", DeepSeek: "D", "zai-org": "Z" };
// Synced from the vLLM Recipes provider assets into /public/providers.
const PROVIDER_LOGOS = {
  MiniMaxAI: staticAssetPath("providers/MiniMaxAI.jpeg"),
  Qwen: staticAssetPath("providers/Qwen.png"),
  "deepseek-ai": staticAssetPath("providers/deepseek-ai.png"),
  moonshotai: staticAssetPath("providers/moonshotai.jpeg"),
  "zai-org": staticAssetPath("providers/zai-org.png"),
};

function providerName(modelId) {
  return String(modelId || "").split("/")[0] || "Other";
}

function modelName(modelId) {
  return String(modelId || "").split("/").pop() || modelId;
}

function ProviderIcon({ name }) {
  const [failed, setFailed] = useState(false);
  const logo = PROVIDER_LOGOS[name];
  if (!logo || failed) return <span className="provider-mark">{PROVIDER_MARKS[name] || name[0]?.toUpperCase() || "+"}</span>;
  return <img className="provider-logo" src={logo} alt="" onError={() => setFailed(true)} />;
}

function EntryButton({ active, children, onClick, disabled = false }) {
  return <button type="button" className={active ? "entry-mode active" : "entry-mode"} aria-pressed={active} disabled={disabled} onClick={onClick}>{children}</button>;
}

export default function ModelEntry({
  builtinModels = [],
  modelId,
  onModelIdChange,
  onOpenModel,
  onOpenLocalFiles,
  onOpenLocalPath,
  language = "zh",
  onLanguageChange,
  theme = "dark",
  onThemeChange,
  loading = false,
  loadingPhase = "reading",
}) {
  const [mode, setMode] = useState("model");
  const [endpoint, setEndpoint] = useState("huggingface");
  const [provider, setProvider] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [localPath, setLocalPath] = useState("");
  const fileRef = useRef(null);
  const providers = useMemo(() => {
    const grouped = new Map();
    builtinModels.forEach((entry) => {
      const name = providerName(entry.modelId);
      if (!grouped.has(name)) grouped.set(name, []);
      grouped.get(name).push(entry);
    });
    return [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [builtinModels]);
  const providerModels = providers.find(([name]) => name === provider)?.[1] || [];
  useEffect(() => {
    if (!provider && !helpOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setProvider(null);
        setHelpOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [provider, helpOpen]);
  const t = language === "en" ? {
    title: "Understand the model.",
    subtitle: "Explore architecture, inspect modules, and estimate the cost on your hardware.",
    model: "Enter / choose model",
    local: "Open local model directory",
    placeholder: "Hugging Face / ModelScope model ID or URL",
    open: "Open model",
    opening: "Opening...",
    loadingSteps: { reading: "Reading model files", building: "Building model structure", metadata: "Checking weight metadata" },
    localPlaceholder: "Path on the backend machine, or choose a folder",
    browse: "Browse by Provider",
    browseHint: "Choose a provider to view mapped models",
    choose: "Choose a model",
    empty: "No mapped models for this provider",
    help: "Help",
    helpTitle: "Quick guide",
    helpItems: ["Enter a Hugging Face or ModelScope model ID, or choose a mapped model.", "Open a local model directory to read its config and optional weight metadata.", "Browse by Provider to find models already included in this viewer."],
    close: "Close",
    chooseFolder: "Choose folder",
    openPath: "Open path",
  } : {
    title: "理解模型。",
    subtitle: "浏览模型架构、检查模块，并估算模型在目标硬件上的成本。",
    model: "输入 / 选择模型",
    local: "打开本地模型目录",
    placeholder: "Hugging Face / ModelScope 模型 ID 或地址",
    open: "打开模型",
    opening: "打开中...",
    loadingSteps: { reading: "读取模型配置", building: "构建模型结构", metadata: "检查权重元数据" },
    localPlaceholder: "后端机器上的路径，或选择本地目录",
    browse: "按 Provider 浏览",
    browseHint: "选择厂商查看已映射模型",
    choose: "选择模型",
    empty: "该 Provider 暂无已映射模型",
    help: "帮助",
    helpTitle: "快速说明",
    helpItems: ["输入 Hugging Face 或 ModelScope 模型 ID，也可以直接选择已映射模型。", "打开本地模型目录，读取 config 和可选的权重元数据。", "按 Provider 浏览仓库内已收录的模型。"],
    close: "关闭",
    chooseFolder: "打开文件夹",
    openPath: "打开路径",
  };
  const themeAction = theme === "dark"
    ? (language === "en" ? "Switch to light theme" : "切换到浅色主题")
    : (language === "en" ? "Switch to dark theme" : "切换到深色主题");

  function handleFiles(event) {
    const files = [...(event.target.files || [])];
    if (files.length > 0) onOpenLocalFiles?.(files);
    event.target.value = "";
  }

  return (
    <main className={`model-entry-page theme-${theme}`}>
      <section className="entry-hero">
        <div className="entry-topline"><div className="entry-brand">Model Structure Viewer<span>.</span></div><div className="entry-top-actions"><button type="button" onClick={() => { const next = language === "en" ? "zh" : "en"; onLanguageChange?.(next); }}>{language === "en" ? "EN / 中" : "中 / EN"}</button><button type="button" title={themeAction} aria-label={themeAction} onClick={onThemeChange}>{theme === "dark" ? (language === "en" ? "Dark" : "深色") : (language === "en" ? "Light" : "浅色")}</button><button type="button" title={t.help} onClick={() => setHelpOpen(true)}>{language === "en" ? "Help" : "帮助"}</button></div></div>
        <h1>{t.title}</h1>
        <p>{t.subtitle}</p>
      </section>
      {loading && <div className="entry-loading" role="status" aria-live="polite">
        <div className="entry-loading-card">
          <div className="entry-loading-mark" aria-hidden="true"><span /><span /><span /></div>
          <div className="entry-loading-copy"><strong>{t.loadingSteps[loadingPhase] || t.opening}</strong><span>{language === "en" ? "Preparing your model view" : "正在准备模型视图"}<i aria-hidden="true">...</i></span></div>
          <div className="entry-loading-track" aria-hidden="true"><span className={`phase-${loadingPhase}`} /></div>
        </div>
      </div>}
      <section className="entry-box" aria-label="Model entry">
        <div className="entry-modes">
          <EntryButton active={mode === "model"} disabled={loading} onClick={() => { setMode("model"); setProvider(null); }}>{t.model}</EntryButton>
          <EntryButton active={mode === "local"} disabled={loading} onClick={() => { setMode("local"); setProvider(null); }}>{t.local}</EntryButton>
        </div>
        {mode === "model" ? (
          <form className="entry-input-row" onSubmit={(event) => { event.preventDefault(); const id = modelId.trim(); const builtin = builtinModels.some((entry) => entry.modelId === id); onOpenModel?.(id, builtin ? "builtin" : "hf", endpoint); }}>
            <select className="entry-source-select" value={endpoint} onChange={(event) => setEndpoint(event.target.value)} aria-label="model source"><option value="huggingface">Hugging Face</option><option value="modelscope">ModelScope</option></select><input list="builtin-models" value={modelId} onChange={(event) => onModelIdChange?.(event.target.value)} placeholder={t.placeholder} aria-label="model id" />
            <datalist id="builtin-models">{builtinModels.map((entry) => <option key={entry.modelId} value={entry.modelId} />)}</datalist>
            <button className="entry-primary" type="submit" disabled={loading || !modelId.trim()}>{loading ? t.opening : t.open}</button>
          </form>
        ) : (
          <form className="entry-input-row" onSubmit={(event) => { event.preventDefault(); if (localPath.trim()) onOpenLocalPath?.(localPath.trim()); }}>
            <input value={localPath} onChange={(event) => setLocalPath(event.target.value)} placeholder={t.localPlaceholder} aria-label="local model path" />
            <button className="entry-secondary" type="button" disabled={loading} onClick={() => fileRef.current?.click()}>{t.chooseFolder}</button>
            <button className="entry-primary" type="submit" disabled={loading || !localPath.trim()}>{t.openPath}</button>
            <input ref={fileRef} className="visually-hidden" type="file" webkitdirectory="true" multiple tabIndex="-1" aria-hidden="true" onChange={handleFiles} />
          </form>
        )}
        {mode === "model" && <div className="entry-quick">{builtinModels.slice(0, 5).map((entry) => <button type="button" key={entry.modelId} onClick={() => { onModelIdChange?.(entry.modelId); onOpenModel?.(entry.modelId, "builtin"); }}>{modelName(entry.modelId)}</button>)}</div>}
      </section>
      {mode === "model" && <section className="provider-section">
        <div className="entry-section-heading"><h2>{t.browse}</h2><span>{t.browseHint}</span></div>
        <div className="provider-grid">{providers.map(([name, entries]) => <button type="button" className="provider-card" key={name} onClick={() => setProvider(name)}><ProviderIcon name={name} /><strong>{name}</strong><small>{entries.length} {language === "en" ? "models" : "个模型"}</small></button>)}</div>
      </section>}
      {provider && <div className="provider-overlay" role="dialog" aria-modal="true" aria-label={provider} onMouseDown={(event) => { if (event.target === event.currentTarget) setProvider(null); }}><div className="provider-picker"><header><div className="provider-picker-title"><ProviderIcon name={provider} /><div><h2>{provider}</h2><p>{t.choose}</p></div></div><button type="button" aria-label={t.close} onClick={() => setProvider(null)}>×</button></header><div className="provider-model-list">{providerModels.length ? providerModels.map((entry) => <button type="button" key={entry.modelId} onClick={() => { setProvider(null); onModelIdChange?.(entry.modelId); onOpenModel?.(entry.modelId, "builtin"); }}><strong>{modelName(entry.modelId)}</strong><span>{entry.modelType || entry.canonicalArchitecture || "mapped structure"}</span><b>→</b></button>) : <p>{t.empty}</p>}</div></div></div>}
      {helpOpen && <div className="entry-help-overlay" role="dialog" aria-modal="true" aria-label={t.helpTitle} onMouseDown={(event) => { if (event.target === event.currentTarget) setHelpOpen(false); }}><div className="entry-help-panel"><header><h2>{t.helpTitle}</h2><button type="button" aria-label={t.close} onClick={() => setHelpOpen(false)}>×</button></header><ul>{t.helpItems.map((item) => <li key={item}>{item}</li>)}</ul></div></div>}
    </main>
  );
}
