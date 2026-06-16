"""Stateless HTTP client for the Hugging Face hub.

Knows about ``hf_endpoint`` only. Returns parsed JSON / raw text and raises
``RemoteError`` on any network or upstream failure. Has no notion of
``model_root`` or config semantics.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..errors import RemoteError

_LOG = logging.getLogger(__name__)

_USER_AGENT = "model-structure-viewer/0.1"
_TIMEOUT_SECONDS = 30


class HuggingFaceClient:
    """Thin wrapper around urllib for the HF Hub REST + resolve endpoints."""

    def __init__(self, hf_endpoint: str):
        self.hf_endpoint = hf_endpoint.rstrip("/")

    # ---- public, hub-aware helpers -------------------------------------------------
    def download_text(self, model_id: str, filename: str, revision: str) -> str:
        url = self._resolve_url(model_id, filename, revision)
        return self._request_text(url, context=f"{model_id}/{filename}")

    def download_json(self, model_id: str, filename: str, revision: str) -> dict[str, Any]:
        text = self.download_text(model_id, filename, revision)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RemoteError(f"Remote file is not valid JSON: {filename}") from exc
        if not isinstance(payload, dict):
            raise RemoteError(f"Remote JSON is not an object: {filename}")
        return payload

    def search_models(self, query: str, limit: int) -> list[dict[str, Any]]:
        params = urllib.parse.urlencode({"search": query, "limit": str(limit)})
        url = f"{self.hf_endpoint}/api/models?{params}"
        payload = self._http_json(url)
        if not isinstance(payload, list):
            raise RemoteError("Unexpected HF search response.")
        return payload

    def list_tree(self, model_id: str, revision: str) -> list[dict[str, Any]]:
        encoded = urllib.parse.quote(model_id, safe="/")
        rev = urllib.parse.quote(revision, safe="")
        url = f"{self.hf_endpoint}/api/models/{encoded}/tree/{rev}?recursive=true"
        try:
            payload = self._http_json(url)
        except RemoteError:
            return []
        return payload if isinstance(payload, list) else []

    # ---- low-level -----------------------------------------------------------------
    def _resolve_url(self, model_id: str, filename: str, revision: str) -> str:
        encoded = urllib.parse.quote(model_id, safe="/")
        rev = urllib.parse.quote(revision, safe="")
        file_name = urllib.parse.quote(filename, safe="/")
        return f"{self.hf_endpoint}/{encoded}/resolve/{rev}/{file_name}"

    def _http_json(self, url: str) -> Any:
        text = self._request_text(url, context=url)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RemoteError(f"HF API returned invalid JSON: {url}") from exc

    def _request_text(self, url: str, *, context: str) -> str:
        try:
            with urllib.request.urlopen(self._build_request(url), timeout=_TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            _LOG.warning("HF HTTP %s for %s (%s)", exc.code, context, url)
            raise RemoteError(f"HF request failed for {context} (HTTP {exc.code})") from exc
        except urllib.error.URLError as exc:
            _LOG.warning("HF URL error for %s (%s): %s", context, url, exc.reason)
            raise RemoteError(f"HF request failed for {context}: {exc.reason}") from exc

    @staticmethod
    def _build_request(url: str) -> urllib.request.Request:
        return urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/json,text/plain,*/*",
            },
        )
