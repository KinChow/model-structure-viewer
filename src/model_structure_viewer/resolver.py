"""Compatibility shim: re-export the resolution API from :mod:`.resolve`.

The implementation now lives in the ``resolve`` sub-package, split into
``hf_client`` (HTTP), ``local_cache`` (filesystem), ``remote_code`` (auto-map
fetch), and ``resolver`` (orchestrator). External callers that imported
``ModelSourceResolver`` / ``ResolvedConfig`` / ``SourceResolutionError`` from
this module continue to work.
"""
from __future__ import annotations

from .resolve import ModelSourceResolver, ResolvedConfig, SourceResolutionError
from .resolve.remote_code import METADATA_ALLOW_RE, WEIGHT_SUFFIXES

__all__ = [
    "ModelSourceResolver",
    "ResolvedConfig",
    "SourceResolutionError",
    "METADATA_ALLOW_RE",
    "WEIGHT_SUFFIXES",
]
