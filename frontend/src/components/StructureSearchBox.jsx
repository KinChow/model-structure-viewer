function StructureSearchBox({ value, onChange, hitCount, disabled, results = [], onSelect }) {
  return (
    <div className="search-box">
      <input
        type="search"
        placeholder="Search nodes by name / type / class..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
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
