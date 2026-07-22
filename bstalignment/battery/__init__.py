"""Stable public surfaces for BatteryGTR and legacy checkpoint compatibility."""

from .battery_gtr import (
    BATTERY_GTR_MODEL_NAME,
    LEGACY_BATTERY_MODEL_NAME,
    BatteryGTR,
    canonical_battery_model_name,
)
from .graph_report_ts_v2 import BATTERY_MODEL_NAME, BatteryGraphReportTSv2

__all__ = [
    "BATTERY_GTR_MODEL_NAME",
    "LEGACY_BATTERY_MODEL_NAME",
    "BATTERY_MODEL_NAME",
    "BatteryGTR",
    "BatteryGraphReportTSv2",
    "canonical_battery_model_name",
]
