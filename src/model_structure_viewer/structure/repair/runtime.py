from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol


class RuntimePatch(Protocol):
    name: str

    def activate(self) -> AbstractContextManager[None]:
        ...


class NoopRuntimePatch:
    name = "noop"

    def activate(self) -> AbstractContextManager[None]:
        return nullcontext()
