import { useEffect, useRef, useState } from "react";

function StructureSearchBox({ value, onChange, hitCount, disabled, results = [], onSelect, language = "zh" }) {
  const [highlighted, setHighlighted] = useState(0);
  const [open, setOpen] = useState(false);
  const boxRef = useRef(null);
  const english = language === "en";
  useEffect(() => setHighlighted(0), [value, results.length]);
  useEffect(() => {
    if (!open) return undefined;
    const onPointerDown = (event) => {
      if (boxRef.current && !boxRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [open]);
  const listOpen = open && Boolean(value) && results.length > 0;
  const handleSelect = (path) => {
    onSelect?.(path);
    setOpen(false);
  };
  return (
    <div className="search-box" ref={boxRef}>
      <input
        type="search"
        role="combobox"
        aria-expanded={listOpen}
        aria-controls="msv-search-listbox"
        aria-autocomplete="list"
        aria-activedescendant={listOpen ? `msv-search-opt-${highlighted}` : undefined}
        placeholder={english ? "Search nodes by name / type / class..." : "搜索节点名称 / 类型 / class..."}
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); }}
        onFocus={() => { if (value) setOpen(true); }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" && results.length > 0) {
            event.preventDefault();
            setOpen(true);
            setHighlighted((index) => Math.min(index + 1, results.length - 1));
          } else if (event.key === "ArrowUp" && results.length > 0) {
            event.preventDefault();
            setHighlighted((index) => Math.max(index - 1, 0));
          } else if (event.key === "Enter" && results[highlighted]) {
            event.preventDefault();
            handleSelect(results[highlighted].path);
          } else if (event.key === "Escape" && value) {
            event.preventDefault();
            onChange("");
            setOpen(false);
          }
        }}
        disabled={disabled}
      />
      {value && (
        <>
          <span className="hit-count">{english ? `${hitCount} match${hitCount === 1 ? "" : "es"}` : `${hitCount} 个匹配`}</span>
          <button className="clear" type="button" onClick={() => { onChange(""); setOpen(false); }}>
            {english ? "Clear" : "清空"}
          </button>
        </>
      )}
      {listOpen && <div className="search-results" id="msv-search-listbox" role="listbox" aria-label="Matching structure nodes">
        {results.map((result, index) => <button key={result.path} id={`msv-search-opt-${index}`} type="button" role="option" aria-selected={index === highlighted} className={index === highlighted ? "highlighted" : ""} onMouseEnter={() => setHighlighted(index)} onClick={() => handleSelect(result.path)}>
          <strong>{result.name}</strong><span>{result.path} · {result.type}</span>
        </button>)}
      </div>}
    </div>
  );
}

export default StructureSearchBox;
