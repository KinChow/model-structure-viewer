function RawConfigTab({ rawJson }) {
  return (
    <section className="export-panel">
      <textarea
        readOnly
        value={rawJson}
        placeholder="Generate a structure to inspect raw API output."
      />
    </section>
  );
}

export default RawConfigTab;
