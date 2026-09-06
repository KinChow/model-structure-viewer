from __future__ import annotations

import logging
from typing import Iterable

from .context import RepairContext, RepairResult
from .registry import RepairStrategy, find_strategy

_LOG = logging.getLogger(__name__)


def try_repair(
    context: RepairContext,
    *,
    strategies: Iterable[RepairStrategy] | None = None,
) -> RepairResult | None:
    strategy = find_strategy(context, strategies=strategies)
    if strategy is None:
        return None
    try:
        return strategy.apply(context)
    except Exception as exc:  # noqa: BLE001 - repair must not hide the original error
        _LOG.warning("Repair strategy %s failed; retaining original introspection error", strategy.name, exc_info=True)
        return None
