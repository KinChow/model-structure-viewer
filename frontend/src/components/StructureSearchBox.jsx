function StructureSearchBox({ value, onChange, hitCount, disabled, results = [], onSelect }) {
  return (
    <div className="search-box">
      <input
        type="search"
        placeholder="Search nodes by name / type / class..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && results[0]) {
            event.preventDefault();
            onSelect?.(results[0].path);
          } else if (event.key === "Escape" && value) {
            event.preventDefault();
            onChange("");
          }
        }}
        disabled={disabled}
      />
      {value && (
        <>
          <span className="hit-count">{hitCount} match{hitCount === 1 ? "" : "es"}</span>
          <button className="clear" onClick={() => onChange("")}>
            Clear
          </button>
        </>
      )}
      {value && results.length > 0 && <div className="search-results" role="listbox" aria-label="Matching structure nodes">
        {results.map((result) => <button key={result.path} type="button" role="option" onClick={() => onSelect?.(result.path)}>
          <strong>{result.name}</strong><span>{result.path} · {result.type}</span>
        </button>)}
      </div>}
    </div>
  );
}

export default StructureSearchBox;
