import { useCallback, useState } from "react";
import { exportStructure } from "../exporters.js";

export function useExport() {
  const [format, setFormat] = useState("mermaid");
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  const run = useCallback(
    async (structure, fmt = format) => {
      if (!structure) return;
      setError("");
      try {
        setText(exportStructure(structure, fmt));
      } catch (err) {
        setError(err.message);
      }
    },
    [format]
  );

  const reset = useCallback(() => setText(""), []);

  return { format, setFormat, text, run, reset, error };
}
