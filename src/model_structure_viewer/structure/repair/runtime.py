from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol


class RuntimePatch(Protocol):
    name: str

    def activate(self) -> AbstractContextManager[None]:
        ...


class ConfigNormalizer(Protocol):
    name: str

    def normalize(self, hf_config: Any) -> dict[str, Any]:
        ...


class NoopRuntimePatch:
    name = "noop"

    def activate(self) -> AbstractContextManager[None]:
        return nullcontext()
