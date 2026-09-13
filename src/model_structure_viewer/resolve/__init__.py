"""Resolution layer: HTTP client / local cache / remote-code fetcher / orchestrator.

Public surface: ``ModelSourceResolver``, ``ResolvedConfig``, and
``SourceResolutionError`` (alias of ``ViewerError``).
"""
from __future__ import annotations

from ..errors import ViewerError
from .resolver import ModelSourceResolver, ResolvedConfig

# Callers that catch SourceResolutionError catch the ViewerError hierarchy.
SourceResolutionError = ViewerError

__all__ = ["ModelSourceResolver", "ResolvedConfig", "SourceResolutionError"]
