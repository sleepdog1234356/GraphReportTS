"""Stable public surface for the general forecasting paper model."""

from .anchored_gtr import ANCHORED_GTR_MODEL_NAME, AnchoredGTR, canonical_general_model_name
from .strategy_registry import AnchoredGTRStrategy, resolve_strategy

__all__ = [
    "ANCHORED_GTR_MODEL_NAME",
    "AnchoredGTR",
    "AnchoredGTRStrategy",
    "canonical_general_model_name",
    "resolve_strategy",
]
