import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const modelsRoot = path.join(repoRoot, "models");

function contentType(filePath) {
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  if (filePath.endsWith(".md")) return "text/markdown; charset=utf-8";
  if (filePath.endsWith(".py")) return "text/x-python; charset=utf-8";
  return "application/octet-stream";
}

function modelsStaticPlugin() {
  return {
    name: "models-static-assets",
    configureServer(server) {
      server.middlewares.use("/models", (req, res, next) => {
        const requestPath = decodeURIComponent((req.url || "").split("?")[0]).replace(/^\/+/, "");
        const filePath = path.normalize(path.join(modelsRoot, requestPath));
        if (!filePath.startsWith(modelsRoot)) {
          res.statusCode = 403;
          res.end("Forbidden");
          return;
        }
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
          next();
          return;
        }
        res.setHeader("Content-Type", contentType(filePath));
        fs.createReadStream(filePath).pipe(res);
      });
    },
    closeBundle() {
      const target = path.join(__dirname, "dist", "models");
      fs.rmSync(target, { recursive: true, force: true });
      fs.cpSync(modelsRoot, target, { recursive: true });
    },
  };
}

export default defineConfig({
  plugins: [react(), modelsStaticPlugin()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
