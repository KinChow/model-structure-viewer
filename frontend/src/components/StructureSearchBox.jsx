function StructureSearchBox({ value, onChange, hitCount, disabled }) {
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
    </div>
  );
}

export default StructureSearchBox;
