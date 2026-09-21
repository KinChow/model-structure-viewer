import { useRef } from "react";
import useDialog from "../hooks/useDialog.js";

function Drawer({
  open,
  theme = "dark",
  language = "zh",
  revision,
  onRevisionChange,
  frameworkProfile = "neutral",
  onFrameworkProfileChange,
  builtinModels,
  onRefreshBuiltinModels,
  onPickBuiltinModel,
  searchQuery,
  onSearchQueryChange,
  searchResults,
  onSearch,
  searchDisabled,
  onPickHfModel,
  onClose,
}) {
  const english = language === "en";
  const panelRef = useRef(null);
  useDialog({ open, onClose, panelRef });
  const t = english
    ? { inputs: "Model options", revision: "Revision", framework: "Framework profile", frameworkHint: "Switches vLLM/SGLang runtime divergences (e.g. recurrent state dtype). Neutral = config-faithful.", builtin: "Built-in models", refresh: "Refresh", search: "Hugging Face search", searchAction: "Search", unknown: "unknown", close: "Close" }
    : { inputs: "模型选项", revision: "Revision", framework: "框架预设", frameworkHint: "切换 vLLM/SGLang 运行时分叉（如线性 state dtype）。Neutral=按 config 如实建模。", builtin: "内置模型", refresh: "刷新", search: "Hugging Face 搜索", searchAction: "搜索", unknown: "未知", close: "关闭" };
  const frameworkOptions = [
    { value: "neutral", label: "Neutral" },
    { value: "vllm", label: "vLLM" },
    { value: "sglang", label: "SGLang" },
  ];
  return (
    <>
      {open && <div className="drawer-backdrop" aria-hidden="true" onClick={onClose} />}
      <aside className={`drawer theme-${theme} ${open ? "open" : ""}`} role="dialog" aria-modal="true" aria-label={t.inputs} ref={panelRef}>
        <button type="button" className="drawer-close" aria-label={t.close} onClick={onClose}>×</button>
      <section>
        <h2>{t.inputs}</h2>
        <label>
          {t.revision}
          <input value={revision} onChange={(event) => onRevisionChange(event.target.value)} />
        </label>
        {onFrameworkProfileChange && (
          <label>
            {t.framework}
            <select value={frameworkProfile} onChange={(event) => onFrameworkProfileChange(event.target.value)}>
              {frameworkOptions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <small className="field-hint">{t.frameworkHint}</small>
          </label>
        )}
      </section>
      <section>
        <h2>{t.builtin}</h2>
        <button onClick={onRefreshBuiltinModels}>{t.refresh}</button>
        <div className="compact-list">
          {builtinModels.map((entry) => (
            <button key={entry.configPath} onClick={() => onPickBuiltinModel(entry)}>
              <strong>{entry.modelId}</strong>
              <span>{entry.modelType || entry.architecture || "built-in config"}</span>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h2>{t.search}</h2>
        <div className="inline">
          <input aria-label={t.search} value={searchQuery} onChange={(event) => onSearchQueryChange(event.target.value)} />
          <button onClick={onSearch} disabled={searchDisabled}>
            {t.searchAction}
          </button>
        </div>
        <div className="compact-list">
          {searchResults.map((item) => (
            <button key={item.model_id} onClick={() => onPickHfModel(item.model_id)}>
              <strong>{item.model_id}</strong>
              <span>{item.pipeline_tag || t.unknown}</span>
            </button>
          ))}
        </div>
      </section>
    </aside>
    </>
  );
}

export default Drawer;
