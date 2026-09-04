function Drawer({
  open,
  language = "zh",
  revision,
  onRevisionChange,
  builtinModels,
  onRefreshBuiltinModels,
  onPickBuiltinModel,
  models,
  onRefreshModels,
  onPickLocalModel,
  searchQuery,
  onSearchQueryChange,
  searchResults,
  onSearch,
  searchDisabled,
  onPickHfModel,
  settings,
  onSettingsChange,
  onSaveSettings,
}) {
  const english = language === "en";
  const t = english ? { inputs: "Inputs", revision: "Revision", builtin: "Built-in Models", backend: "Backend Local Models", refresh: "Refresh", search: "Hugging Face Search", searchAction: "Search", settings: "Settings", modelRoot: "Model root", endpoint: "HF endpoint", offline: "Offline", save: "Save settings", unknown: "unknown", cache: "model cache", configFile: "config file" } : { inputs: "输入", revision: "Revision", builtin: "内置模型", backend: "后端本地模型", refresh: "刷新", search: "Hugging Face 搜索", searchAction: "搜索", settings: "设置", modelRoot: "模型根目录", endpoint: "HF endpoint", offline: "离线", save: "保存设置", unknown: "未知", cache: "模型缓存", configFile: "config 文件" };
  return (
    <aside className={`drawer ${open ? "open" : ""}`}>
      <section>
        <h2>{t.inputs}</h2>
        <label>
          {t.revision}
          <input value={revision} onChange={(event) => onRevisionChange(event.target.value)} />
        </label>
      </section>
      <section>
        <h2>{t.builtin}</h2>
        <button onClick={onRefreshBuiltinModels}>{t.refresh}</button>
        <div className="compact-list">
          {builtinModels.map((entry) => (
            <button key={entry.configPath} onClick={() => onPickBuiltinModel(entry)}>
              <strong>{entry.modelId}</strong>
              <span>{entry.modelType || entry.canonicalArchitecture || "built-in config"}</span>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h2>{t.backend}</h2>
        <button onClick={onRefreshModels}>{t.refresh}</button>
        <div className="compact-list">
          {models.map((entry) => (
            <button key={entry.config_path} onClick={() => onPickLocalModel(entry)}>
              <strong>{entry.model_id}</strong>
              <span>{entry.load_by === "config_path" ? t.configFile : t.cache}</span>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h2>{t.search}</h2>
        <div className="inline">
          <input value={searchQuery} onChange={(event) => onSearchQueryChange(event.target.value)} />
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
      <section>
        <h2>{t.settings}</h2>
        <label>
          {t.modelRoot}
          <input
            value={settings.model_root}
            onChange={(event) => onSettingsChange({ ...settings, model_root: event.target.value })}
          />
        </label>
        <label>
          HF endpoint
          <input
            value={settings.hf_endpoint}
            onChange={(event) => onSettingsChange({ ...settings, hf_endpoint: event.target.value })}
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.offline}
            onChange={(event) => onSettingsChange({ ...settings, offline: event.target.checked })}
          />
          {t.offline}
        </label>
        <button onClick={onSaveSettings}>{t.save}</button>
      </section>
    </aside>
  );
}

export default Drawer;
