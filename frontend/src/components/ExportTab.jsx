function ExportTab({ format, onFormatChange, text, onRun }) {
  return (
    <section className="export-panel">
      <div className="panel-toolbar">
        <h2>Export</h2>
        <div className="toolbar-actions">
          <select value={format} onChange={(event) => onFormatChange(event.target.value)}>
            <option value="mermaid">Mermaid</option>
            <option value="dot">DOT</option>
            <option value="json">JSON</option>
          </select>
          <button onClick={onRun}>Export</button>
        </div>
      </div>
      <textarea readOnly value={text} placeholder="Export output appears here." />
    </section>
  );
}

export default ExportTab;
