"""Resolution layer: split into HTTP client / local cache / remote-code fetcher / orchestrator.

Public surface preserved for backwards compatibility with the old single-file
``model_structure_viewer.resolver`` module: ``ModelSourceResolver``,
``ResolvedConfig``, and ``SourceResolutionError``.
"""
from __future__ import annotations

from ..errors import ViewerError
from .resolver import ModelSourceResolver, ResolvedConfig

# Backward-compatible alias: external callers that catch SourceResolutionError
# now catch the broader ViewerError hierarchy.
SourceResolutionError = ViewerError

__all__ = ["ModelSourceResolver", "ResolvedConfig", "SourceResolutionError"]
