import { useEffect, useState } from "react";

function StructureSearchBox({ value, onChange, hitCount, disabled, results = [], onSelect }) {
  const [highlighted, setHighlighted] = useState(0);
  useEffect(() => setHighlighted(0), [value, results.length]);
  return (
    <div className="search-box">
      <input
        type="search"
        placeholder="Search nodes by name / type / class..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" && results.length > 0) {
            event.preventDefault();
            setHighlighted((index) => Math.min(index + 1, results.length - 1));
          } else if (event.key === "ArrowUp" && results.length > 0) {
            event.preventDefault();
            setHighlighted((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter" && results[highlighted]) {
            event.preventDefault();
            onSelect?.(results[highlighted].path);
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
        {results.map((result, index) => <button key={result.path} type="button" role="option" aria-selected={index === highlighted} className={index === highlighted ? "highlighted" : ""} onMouseEnter={() => setHighlighted(index)} onClick={() => onSelect?.(result.path)}>
          <strong>{result.name}</strong><span>{result.path} · {result.type}</span>
        </button>)}
      </div>}
    </div>
  );
}

export default StructureSearchBox;
