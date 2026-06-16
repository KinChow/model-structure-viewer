"""Best-effort fetcher for ``trust_remote_code`` modeling files.

Composes :class:`HuggingFaceClient` (network) and a target ``local_dir`` (a
:class:`Path` already created by the caller). It only fetches files referenced
by ``auto_map`` and same-prefix metadata helpers.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..errors import RemoteError
from .hf_client import HuggingFaceClient

_LOG = logging.getLogger(__name__)

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


class RemoteCodeFetcher:
    def __init__(self, hf: HuggingFaceClient):
        self.hf = hf

    def cache_metadata_files(self, model_id: str, revision: str, cache_dir: Path) -> None:
        """Mirror small, allow-listed root files (README, tokenizer json, etc.)."""
        for item in self.hf.list_tree(model_id, revision):
            file_path = item.get("path", "")
            if not file_path or "/" in file_path:
                continue
            if not METADATA_ALLOW_RE.match(file_path):
                continue
            if file_path.endswith(WEIGHT_SUFFIXES):
                continue
            if file_path == "config.json":
                continue
            target = cache_dir / file_path
            try:
                text = self.hf.download_text(model_id, file_path, revision)
            except RemoteError:
                continue
            target.write_text(text, encoding="utf-8")

    def ensure_remote_code(
        self,
        *,
        model_id: str,
        revision: str,
        local_dir: Path,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Download ``modeling_*.py`` / ``configuration_*.py`` / ``tokenization_*.py``.

        Only files referenced in ``auto_map`` (and same-prefix metadata helpers) are
        fetched, and only when missing locally. Caller is responsible for honoring
        offline / auto_fetch_remote_code settings.
        """
        modules = self._auto_map_modules(config)
        if not modules:
            _LOG.info("Remote-code fetch skipped for %s: no_auto_map", model_id)
            return {"fetched": [], "errors": [], "skipped": "no_auto_map"}

        tree_paths = [item.get("path", "") for item in self.hf.list_tree(model_id, revision)]
        root_files = {p for p in tree_paths if p and "/" not in p}
        if not root_files:
            _LOG.info("Remote-code fetch skipped for %s: empty_tree", model_id)
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
            if filename in {"config.json", "README.md"}:
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
                text = self.hf.download_text(model_id, filename, revision)
            except RemoteError as exc:
                errors.append({"file": filename, "reason": str(exc)})
                continue
            target.write_text(text, encoding="utf-8")
            fetched.append(filename)
        if fetched:
            _LOG.info("Fetched remote code for %s: %s", model_id, fetched)
        if errors:
            _LOG.warning("Remote code fetch errors for %s: %s", model_id, errors)
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
