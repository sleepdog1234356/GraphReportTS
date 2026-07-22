"""Stable public surface for BatteryGTR."""

from .battery_gtr import (
    BATTERY_GTR_MODEL_NAME,
    BatteryGTR,
)
from anchoredgtr.core.model import BatteryGTRCore

__all__ = [
    "BATTERY_GTR_MODEL_NAME",
    "BatteryGTR",
    "BatteryGTRCore",
]
