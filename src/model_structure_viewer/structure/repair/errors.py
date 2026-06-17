from __future__ import annotations

from enum import StrEnum


class IntrospectionFailureKind(StrEnum):
    REMOTE_CODE_UNAVAILABLE = "remote_code_unavailable"
    REMOTE_IMPORT_COMPAT = "remote_import_compat"
    CONFIG_FIELD_MISSING = "config_field_missing"
    CONFIG_LOAD_FAILED = "config_load_failed"
    MODEL_INIT_FAILED = "model_init_failed"
    UNSAFE_REMOTE_CODE = "unsafe_remote_code"
    UNKNOWN = "unknown"
