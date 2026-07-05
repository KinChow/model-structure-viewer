const SOURCE_ITEMS = ["auto", "builtin", "local", "hf", "config"];

function Header({
  sourceLabel,
  source,
  onSourceChange,
  modelId,
  onModelIdChange,
  cachePolicy,
  onCachePolicyChange,
  onOpenDrawer,
  onGenerate,
  loading,
}) {
  return (
    <header className="app-header">
      <div className="brand">
        <h1>Model Structure Viewer</h1>
        <span>{sourceLabel}</span>
      </div>
      <div className="input-bar">
        <select value={source} onChange={(event) => onSourceChange(event.target.value)} aria-label="source">
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
            onClick={onOpenDrawer}
          />
        ) : (
          <input
            value={modelId}
            onChange={(event) => onModelIdChange(event.target.value)}
            placeholder="deepseek-ai/DeepSeek-V3.1"
            aria-label="model id"
          />
        )}
        <select
          value={cachePolicy}
          onChange={(event) => onCachePolicyChange(event.target.value)}
          aria-label="cache policy"
        >
          <option value="prefer-local">prefer-local</option>
          <option value="refresh">refresh</option>
          <option value="offline">offline</option>
        </select>
        <button className="primary" onClick={onGenerate} disabled={loading} aria-busy={loading}>
          {loading ? "Generating..." : "Generate"}
        </button>
        <button className="icon-button" onClick={onOpenDrawer} title="Settings and search">
          ⚙
        </button>
      </div>
    </header>
  );
}

export default Header;
