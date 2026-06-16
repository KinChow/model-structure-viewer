"""Domain error hierarchy for model-structure-viewer.

Each error carries an ``http_status`` so the FastAPI layer can map all
domain failures through a single exception handler instead of sprinkling
``raise HTTPException`` calls across routes.
"""
from __future__ import annotations


class ViewerError(RuntimeError):
    """Base class for all module-structure-viewer domain errors."""

    http_status: int = 500


class ConfigError(ViewerError):
    """Caller-side problem: missing/invalid arguments or malformed config JSON."""

    http_status = 400


class NotFoundError(ViewerError):
    """A requested local resource (config / model directory) does not exist."""

    http_status = 404


class RemoteError(ViewerError):
    """Network or upstream HTTP error talking to Hugging Face."""

    http_status = 502


class IntrospectionError(ViewerError):
    """Meta-device introspection (Plan A) could not produce a structure."""

    http_status = 500
