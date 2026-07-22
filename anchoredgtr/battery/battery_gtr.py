"""Stable paper-facing identity for the anchor-free BatteryGTR model."""

from __future__ import annotations

from anchoredgtr.core.contracts import GTRConfig
from anchoredgtr.core.model import BatteryGTRCore


BATTERY_GTR_MODEL_NAME = "BatteryGTR"


class BatteryGTR(BatteryGTRCore):
    """Battery graph-text model with patch graphs and a direct SOH head."""

    def __init__(self, config: GTRConfig) -> None:
        if config.domain != "battery":
            raise ValueError("BatteryGTR requires domain='battery'")
        if config.graph_embedding_variant != "patch":
            raise ValueError("BatteryGTR requires graph_embedding_variant='patch'")
        super().__init__(config)


__all__ = [
    "BATTERY_GTR_MODEL_NAME",
    "BatteryGTR",
]
