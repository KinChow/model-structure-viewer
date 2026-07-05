function Drawer({
  open,
  revision,
  onRevisionChange,
  configText,
  onConfigTextChange,
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
  return (
    <aside className={`drawer ${open ? "open" : ""}`}>
      <section>
        <h2>Inputs</h2>
        <label>
          Revision
          <input value={revision} onChange={(event) => onRevisionChange(event.target.value)} />
        </label>
        <label>
          Config JSON
          <textarea
            value={configText}
            onChange={(event) => onConfigTextChange(event.target.value)}
            placeholder='{"model_type": "..."}'
          />
        </label>
      </section>
      <section>
        <h2>Built-in Models</h2>
        <button onClick={onRefreshBuiltinModels}>Refresh</button>
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
        <h2>Backend Local Models</h2>
        <button onClick={onRefreshModels}>Refresh</button>
        <div className="compact-list">
          {models.map((entry) => (
            <button key={entry.config_path} onClick={() => onPickLocalModel(entry)}>
              <strong>{entry.model_id}</strong>
              <span>{entry.load_by === "config_path" ? "config file" : "model cache"}</span>
            </button>
          ))}
        </div>
      </section>
      <section>
        <h2>Hugging Face Search</h2>
        <div className="inline">
          <input value={searchQuery} onChange={(event) => onSearchQueryChange(event.target.value)} />
          <button onClick={onSearch} disabled={searchDisabled}>
            Search
          </button>
        </div>
        <div className="compact-list">
          {searchResults.map((item) => (
            <button key={item.model_id} onClick={() => onPickHfModel(item.model_id)}>
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
          Offline
        </label>
        <button onClick={onSaveSettings}>Save settings</button>
      </section>
    </aside>
  );
}

export default Drawer;
