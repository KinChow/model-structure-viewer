import { useState } from "react";

// 导出面板：内容随格式/结构自动生成（DetailWorkspace 的 effect 负责 run），
// 这里只做格式选择 + 复制。按钮从「Export（手动生成）」改为「复制」——人因上
// 打开即见内容，无需先点一次才出结果。
function ExportTab({ format, onFormatChange, text, language = "zh" }) {
  const en = language === "en";
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* 剪贴板不可用（非安全上下文等）时静默；文本仍可手动选中复制。 */
    }
  };
  return (
    <section className="export-panel">
      <div className="panel-toolbar">
        <h2>{en ? "Export" : "导出"}</h2>
        <div className="toolbar-actions">
          <select value={format} onChange={(event) => onFormatChange(event.target.value)} aria-label={en ? "Export format" : "导出格式"}>
            <option value="mermaid">Mermaid</option>
            <option value="dot">DOT</option>
            <option value="json">JSON</option>
          </select>
          <button type="button" onClick={copy} disabled={!text}>{copied ? (en ? "Copied" : "已复制") : (en ? "Copy" : "复制")}</button>
        </div>
      </div>
      <textarea readOnly value={text} placeholder={en ? "Select a format to generate export output." : "选择格式即自动生成导出内容。"} />
    </section>
  );
}

export default ExportTab;
