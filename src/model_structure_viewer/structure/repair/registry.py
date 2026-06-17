from __future__ import annotations

from typing import Iterable, Protocol

from .context import RepairContext, RepairResult


class RepairStrategy(Protocol):
    name: str

    def matches(self, context: RepairContext) -> bool:
        ...

    def apply(self, context: RepairContext) -> RepairResult:
        ...


def default_strategies() -> list[RepairStrategy]:
    from .strategies import STRATEGIES

    return list(STRATEGIES)


def find_strategy(
    context: RepairContext,
    *,
    strategies: Iterable[RepairStrategy] | None = None,
) -> RepairStrategy | None:
    for strategy in strategies if strategies is not None else default_strategies():
        if strategy.matches(context):
            return strategy
    return None
