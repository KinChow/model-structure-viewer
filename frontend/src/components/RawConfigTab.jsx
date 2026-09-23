function RawConfigTab({ rawJson, language = "zh" }) {
  const en = language === "en";
  return (
    <section className="export-panel">
      <textarea
        readOnly
        value={rawJson}
        placeholder={en ? "Generate a structure to inspect raw API output." : "打开模型后在此查看原始配置。"}
      />
    </section>
  );
}

export default RawConfigTab;
