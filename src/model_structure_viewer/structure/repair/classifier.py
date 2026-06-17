from __future__ import annotations

from .errors import IntrospectionFailureKind


def classify_introspection_error(error: BaseException) -> IntrospectionFailureKind:
    message = str(error)
    lowered = message.lower()

    if "auto_map" in message and "no local model directory" in lowered:
        return IntrospectionFailureKind.REMOTE_CODE_UNAVAILABLE
    if "is_torch_fx_available" in message and "cannot import name" in lowered:
        return IntrospectionFailureKind.REMOTE_IMPORT_COMPAT
    if "has no attribute" in lowered:
        return IntrospectionFailureKind.CONFIG_FIELD_MISSING
    if "autoconfig" in lowered:
        return IntrospectionFailureKind.CONFIG_LOAD_FAILED
    if "automodel.from_config failed" in lowered:
        return IntrospectionFailureKind.MODEL_INIT_FAILED
    return IntrospectionFailureKind.UNKNOWN
