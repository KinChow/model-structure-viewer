from __future__ import annotations

from .deepseek_import_compat import DeepSeekImportCompatStrategy
from .minimax_config_adapter import MiniMaxConfigAdapterStrategy

STRATEGIES = (
    DeepSeekImportCompatStrategy(),
    MiniMaxConfigAdapterStrategy(),
)

__all__ = ["STRATEGIES", "DeepSeekImportCompatStrategy", "MiniMaxConfigAdapterStrategy"]
