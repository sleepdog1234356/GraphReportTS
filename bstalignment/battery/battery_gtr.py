"""Stable paper-facing identity for the anchor-free BatteryGTR model."""

from __future__ import annotations

from bstalignment.v2.contracts import GraphReportTSv2Config
from bstalignment.v2.model import BatteryGraphReportTSv2


BATTERY_GTR_MODEL_NAME = "BatteryGTR"
LEGACY_BATTERY_MODEL_NAME = "GraphReportTS-v2"


def canonical_battery_model_name(name: str) -> str:
    """Map only the exact former battery-main identity."""

    value = str(name)
    return BATTERY_GTR_MODEL_NAME if value == LEGACY_BATTERY_MODEL_NAME else value


class BatteryGTR(BatteryGraphReportTSv2):
    """Battery graph-text model with patch graphs and a direct SOH head."""

    def __init__(self, config: GraphReportTSv2Config) -> None:
        if config.domain != "battery":
            raise ValueError("BatteryGTR requires domain='battery'")
        if config.graph_embedding_variant != "patch":
            raise ValueError("BatteryGTR requires graph_embedding_variant='patch'")
        super().__init__(config)


__all__ = [
    "BATTERY_GTR_MODEL_NAME",
    "LEGACY_BATTERY_MODEL_NAME",
    "BatteryGTR",
    "canonical_battery_model_name",
]
