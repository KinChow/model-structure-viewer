from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import HfSearchResult, ModelEntry
from .settings import AppSettings

METADATA_ALLOW_RE = re.compile(
    r"^(README\.md|config\.json|(configuration|modeling|tokenization)_.*\.py)$"
)
WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
)


class SourceResolutionError(RuntimeError):
    pass


@dataclass
class ResolvedConfig:
    config: dict[str, Any]
    source: dict[str, Any]
    local_dir: Path | None = None


class ModelSourceResolver:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings.from_env()

    def list_local_models(self) -> list[ModelEntry]:
        root = self.settings.model_root
        if not root.exists():
            return []
        entries: list[ModelEntry] = []
        for config_path in sorted(root.rglob("config.json")):
            try:
                rel_parent = config_path.parent.relative_to(root)
            except ValueError:
                continue
            if not rel_parent.parts:
                continue
            model_id = "/".join(rel_parent.parts)
            entries.append(
                ModelEntry(
                    model_id=model_id,
                    config_path=str(config_path),
                    has_readme=(config_path.parent / "README.md").exists(),
                    has_remote_config_code=any(config_path.parent.glob("configuration_*.py")),
                )
            )
        return entries

    def resolve(
        self,
        *,
        source: str = "auto",
        model_id: str | None = None,
        config_path: str | None = None,
        config_json: dict[str, Any] | None = None,
        revision: str = "main",
        cache_policy: str = "prefer-local",
        detail_level: str = "compressed",
    ) -> ResolvedConfig:
        if source == "config":
            if config_json is None:
                raise SourceResolutionError("source=config requires config_json.")
            return ResolvedConfig(
                config=config_json,
                source={"kind": "uploaded config", "detail_level": detail_level},
            )

        if config_path:
            resolved = self._resolve_config_path(config_path, detail_level)
            self._maybe_fetch_remote_code(resolved, model_id=model_id, revision=revision)
            return resolved

        if not model_id:
            raise SourceResolutionError("model_id, config_path, or config_json is required.")

        effective_policy = "offline" if self.settings.offline else cache_policy
        resolved: ResolvedConfig
        if source == "local":
            resolved = self._resolve_local_model(model_id, detail_level)
        elif source == "hf":
            if effective_policy == "offline":
                raise SourceResolutionError("source=hf cannot run with offline cache policy.")
            resolved = self._resolve_hf_model(model_id, revision, cache_policy, detail_level)
        elif source == "auto":
            local = None
            if effective_policy != "refresh":
                local = self._try_local_model(model_id, detail_level)
            if local is not None:
                resolved = local
            elif effective_policy == "offline":
                expected = self.local_config_path(model_id)
                raise SourceResolutionError(f"Local config not found and offline is enabled: {expected}")
            else:
                resolved = self._resolve_hf_model(model_id, revision, cache_policy, detail_level)
        else:
            raise SourceResolutionError(f"Unsupported source: {source}")

        self._maybe_fetch_remote_code(resolved, model_id=model_id, revision=revision)
        return resolved

    def local_config_path(self, model_id: str) -> Path:
        parts = [part for part in model_id.split("/") if part]
        return self.settings.model_root.joinpath(*parts, "config.json")

    def _resolve_config_path(self, config_path: str, detail_level: str) -> ResolvedConfig:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise SourceResolutionError(f"Config file not found: {path}")
        return ResolvedConfig(
            config=self._load_json(path),
            source={"kind": "local file", "config_path": str(path), "detail_level": detail_level},
            local_dir=path.parent,
        )

    def _try_local_model(self, model_id: str, detail_level: str) -> ResolvedConfig | None:
        path = self.local_config_path(model_id)
        if not path.exists():
            return None
        return ResolvedConfig(
            config=self._load_json(path),
            source={
                "kind": "local cache",
                "model_id": model_id,
                "config_path": str(path),
                "detail_level": detail_level,
            },
            local_dir=path.parent,
        )

    def _resolve_local_model(self, model_id: str, detail_level: str) -> ResolvedConfig:
        resolved = self._try_local_model(model_id, detail_level)
        if resolved is None:
            expected = self.local_config_path(model_id)
            raise SourceResolutionError(f"Local model config not found: {expected}")
        return resolved

    def _resolve_hf_model(
        self,
        model_id: str,
        revision: str,
        cache_policy: str,
        detail_level: str,
    ) -> ResolvedConfig:
        cache_dir = self.local_config_path(model_id).parent
        config_path = cache_dir / "config.json"
        if cache_policy == "prefer-local" and config_path.exists():
            return self._resolve_local_model(model_id, detail_level)

        cache_dir.mkdir(parents=True, exist_ok=True)
        config = self._download_json(model_id, "config.json", revision)
        self._write_json(config_path, config)
        self._cache_metadata_files(model_id, revision, cache_dir)
        return ResolvedConfig(
            config=config,
            source={
                "kind": "hf remote",
                "model_id": model_id,
                "revision": revision,
                "hf_endpoint": self.settings.hf_endpoint,
                "cache_path": str(config_path),
                "detail_level": detail_level,
            },
            local_dir=cache_dir,
        )

    def search_hf_models(self, query: str, limit: int = 10) -> list[HfSearchResult]:
        if self.settings.offline:
            raise SourceResolutionError("HF search is disabled in offline mode.")
        params = urllib.parse.urlencode({"search": query, "limit": str(limit)})
        url = f"{self.settings.hf_endpoint}/api/models?{params}"
        payload = self._http_json(url)
        if not isinstance(payload, list):
            raise SourceResolutionError("Unexpected HF search response.")
        results: list[HfSearchResult] = []
        for item in payload:
            model_id = item.get("modelId") or item.get("id")
            if not model_id:
                continue
            results.append(
                HfSearchResult(
                    model_id=model_id,
                    pipeline_tag=item.get("pipeline_tag"),
                    tags=item.get("tags") or [],
                    downloads=item.get("downloads"),
                    likes=item.get("likes"),
                )
            )
        return results

    def get_remote_config(self, model_id: str, revision: str = "main") -> dict[str, Any]:
        if self.settings.offline:
            raise SourceResolutionError("HF config lookup is disabled in offline mode.")
        return self._download_json(model_id, "config.json", revision)

    def _cache_metadata_files(self, model_id: str, revision: str, cache_dir: Path) -> None:
        for item in self._tree(model_id, revision):
            file_path = item.get("path", "")
            if not file_path or "/" in file_path:
                continue
            if not METADATA_ALLOW_RE.match(file_path):
                continue
            if file_path.endswith(WEIGHT_SUFFIXES):
                continue
            target = cache_dir / file_path
            if file_path == "config.json":
                continue
            try:
                text = self._download_text(model_id, file_path, revision)
            except SourceResolutionError:
                continue
            target.write_text(text, encoding="utf-8")

    def _maybe_fetch_remote_code(
        self,
        resolved: ResolvedConfig,
        *,
        model_id: str | None,
        revision: str,
    ) -> None:
        if resolved.local_dir is None or model_id is None:
            return
        if self.settings.offline or not self.settings.auto_fetch_remote_code:
            return
        info = self.ensure_remote_code(
            model_id=model_id,
            revision=revision,
            local_dir=resolved.local_dir,
            config=resolved.config,
        )
        if info.get("fetched") or info.get("errors"):
            resolved.source["remote_code_fetch"] = info

    def ensure_remote_code(
        self,
        *,
        model_id: str,
        revision: str,
        local_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Best-effort: download modeling_*.py / configuration_*.py / tokenization_*.py.

        Only fetches files referenced in ``auto_map`` (and same-prefix metadata files
        on the repository root) when they are missing locally. Honors offline mode and
        the auto_fetch_remote_code setting via the caller.
        """
        modules = self._auto_map_modules(config)
        if not modules:
            return {"fetched": [], "errors": [], "skipped": "no_auto_map"}

        tree_paths = [item.get("path", "") for item in self._tree(model_id, revision)]
        root_files = {p for p in tree_paths if p and "/" not in p}
        if not root_files:
            return {"fetched": [], "errors": [], "skipped": "empty_tree"}

        wanted: list[str] = []
        for module in modules:
            filename = f"{module}.py"
            if filename in root_files:
                wanted.append(filename)
        # Pull same-prefix configuration_/tokenization_/modeling_ helpers so that
        # imports inside modeling_*.py succeed.
        for filename in sorted(root_files):
            if not METADATA_ALLOW_RE.match(filename):
                continue
            if filename == "config.json" or filename == "README.md":
                continue
            if filename.endswith(WEIGHT_SUFFIXES):
                continue
            if filename not in wanted:
                wanted.append(filename)

        fetched: list[str] = []
        errors: list[dict[str, str]] = []
        for filename in wanted:
            target = local_dir / filename
            if target.exists():
                continue
            try:
                text = self._download_text(model_id, filename, revision)
            except SourceResolutionError as exc:
                errors.append({"file": filename, "reason": str(exc)})
                continue
            target.write_text(text, encoding="utf-8")
            fetched.append(filename)
        return {"fetched": fetched, "errors": errors}

    @staticmethod
    def _auto_map_modules(config: dict[str, Any]) -> list[str]:
        auto_map = config.get("auto_map") or {}
        if not isinstance(auto_map, dict):
            return []
        modules: set[str] = set()
        for value in auto_map.values():
            candidates = value if isinstance(value, list) else [value]
            for item in candidates:
                if isinstance(item, str) and "." in item:
                    module = item.split(".", 1)[0]
                    if module and "/" not in module and "\\" not in module:
                        modules.add(module)
        return sorted(modules)

    def _tree(self, model_id: str, revision: str) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote(model_id, safe="/")
        rev = urllib.parse.quote(revision, safe="")
        url = f"{self.settings.hf_endpoint}/api/models/{encoded}/tree/{rev}?recursive=true"
        try:
            payload = self._http_json(url)
        except SourceResolutionError:
            return []
        return payload if isinstance(payload, list) else []

    def _download_json(self, model_id: str, filename: str, revision: str) -> dict[str, Any]:
        text = self._download_text(model_id, filename, revision)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SourceResolutionError(f"Remote file is not valid JSON: {filename}") from exc
        if not isinstance(payload, dict):
            raise SourceResolutionError(f"Remote JSON is not an object: {filename}")
        return payload

    def _download_text(self, model_id: str, filename: str, revision: str) -> str:
        encoded = urllib.parse.quote(model_id, safe="/")
        rev = urllib.parse.quote(revision, safe="")
        file_name = urllib.parse.quote(filename, safe="/")
        url = f"{self.settings.hf_endpoint}/{encoded}/resolve/{rev}/{file_name}"
        try:
            with urllib.request.urlopen(self._request(url), timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise SourceResolutionError(f"HF file not found: {model_id}/{filename} ({exc.code})") from exc
        except urllib.error.URLError as exc:
            raise SourceResolutionError(f"HF request failed: {exc.reason}") from exc

    def _http_json(self, url: str) -> Any:
        try:
            with urllib.request.urlopen(self._request(url), timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise SourceResolutionError(f"HF API request failed with HTTP {exc.code}: {url}") from exc
        except urllib.error.URLError as exc:
            raise SourceResolutionError(f"HF API request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise SourceResolutionError(f"HF API returned invalid JSON: {url}") from exc

    @staticmethod
    def _request(url: str) -> urllib.request.Request:
        return urllib.request.Request(
            url,
            headers={
                "User-Agent": "model-structure-viewer/0.1",
                "Accept": "application/json,text/plain,*/*",
            },
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise SourceResolutionError(f"Config JSON must be an object: {path}")
        return payload

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
