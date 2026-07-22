from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


def aging_stage_from_soh(soh: float) -> int:
    """0 early, 1 middle, 2 late. Thresholds are deliberately coarse for supervision."""
    if soh >= 0.95:
        return 0
    if soh >= 0.85:
        return 1
    return 2


def aging_stage_name(stage: int) -> str:
    return ["early aging", "middle aging", "late aging"][int(stage)]


def _safe_float(x, default=0.0):
    try:
        if x is None or not np.isfinite(float(x)):
            return default
        return float(x)
    except Exception:
        return default


def parse_fast_charge_policy(policy: str) -> str:
    """Human-readable fast-charge policy description.

    MIT policies are often represented as strings like C1(Q1)-C2, but mirrors vary.
    We keep this robust and avoid assuming exact formatting.
    """
    if policy is None or str(policy).lower() in {"nan", "none", "unknown"}:
        return "fast charging policy is unknown"
    p = str(policy)
    nums = re.findall(r"[-+]?\d*\.?\d+", p)
    if len(nums) >= 2:
        return f"fast charging policy descriptor is {p}, containing current-rate or switch-point values {', '.join(nums[:4])}"
    return f"fast charging policy descriptor is {p}"


def infer_degradation_trend(hist: pd.DataFrame) -> str:
    q = hist["QD"].to_numpy(dtype=float)
    if len(q) < 3:
        return "capacity trend is not stable enough to estimate"
    x = np.arange(len(q), dtype=float)
    try:
        slope = np.polyfit(x, q, 1)[0]
    except Exception:
        slope = 0.0
    if slope < -2e-3:
        return "recent discharge capacity is decreasing quickly"
    if slope < -3e-4:
        return "recent discharge capacity is slowly decreasing"
    if slope > 3e-4:
        return "recent discharge capacity shows slight regeneration or measurement rebound"
    return "recent discharge capacity is almost stable"


def build_battery_prompt(
    cell_id: str,
    charge_policy: str,
    cycle_life: float,
    hist: pd.DataFrame,
    target_cycle_start: int,
    target_cycle_end: int,
    forecast_horizon: int,
    chemistry: str = "LFP/graphite",
    nominal_capacity_ah: float = 1.1,
    ambient_temperature_c: float = 30.0,
) -> str:
    """Dynamic cell/cycle-level prompt for multi-horizon forecasting.

    Important: this prompt uses only history available up to the input window.
    It must not use target SOH or future SOH, otherwise inference would leak labels.
    """
    last = hist.iloc[-1]
    first = hist.iloc[0]
    q_start = _safe_float(first.get("QD"))
    q_end = _safe_float(last.get("QD"))
    soh_obs = _safe_float(last.get("SOH"), q_end / max(q_start, 1e-6))
    ir_end = _safe_float(last.get("IR"))
    tmax = _safe_float(hist["Tmax"].max()) if "Tmax" in hist else 0.0
    tavg = _safe_float(hist["Tavg"].mean()) if "Tavg" in hist else 0.0
    charge_time = _safe_float(last.get("chargetime"))
    observed_stage = aging_stage_name(aging_stage_from_soh(soh_obs))
    trend = infer_degradation_trend(hist)
    policy_txt = parse_fast_charge_policy(charge_policy)

    return (
        f"Battery metadata: cell {cell_id} is a lithium-ion {chemistry} cell with nominal capacity "
        f"about {nominal_capacity_ah:.2f} Ah. The test is conducted near {ambient_temperature_c:.0f} degrees C. "
        f"The {policy_txt}. Discharge protocol is identical across cells in the MIT early-cycle dataset. "
        f"Observed cycle context: the latest observed cycle is {int(last.get('cycle', 0))}. "
        f"The current observed health stage is {observed_stage}. {trend}. "
        f"Recent summary statistics over the input window: discharge capacity changed from {q_start:.4f} Ah to {q_end:.4f} Ah; "
        f"observed SOH at the latest input cycle is {soh_obs:.4f}; "
        f"internal resistance proxy at the latest cycle is {ir_end:.5f}; maximum temperature in the window is {tmax:.2f} degrees C; "
        f"average temperature is {tavg:.2f} degrees C; latest charge time is {charge_time:.2f}. "
        f"Forecast instruction: predict the SOH trajectory for the next {forecast_horizon} cycles, "
        f"from cycle {target_cycle_start} to cycle {target_cycle_end}. "
        f"Align this battery operating-context description with the numerical time-series window and output multi-step SOH values."
    )


def build_cycle_prompt(
    *,
    cycle: int,
    forecast_horizon: int,
    charge_policy: str,
    qd: float,
    qc: float,
    ir: float,
    tmax: float,
    tavg: float,
    chargetime: float,
    qd_roll5: float,
    dqd_cycle: float,
    qd_slope5: float,
    ir_slope5: float,
    trend_label: str,
    observed_stage: str,
    chemistry: str = "LFP_graphite",
    nominal_capacity_ah: float = 1.1,
) -> str:
    """Compact, leakage-free prompt for one-cycle SOH estimation and forecasting."""
    policy_txt = parse_fast_charge_policy(charge_policy)
    return (
        "Task: estimate current SOH and forecast future SOH. "
        f"Battery: chemistry={chemistry}; nominal_capacity_Ah={nominal_capacity_ah:.2f}; {policy_txt}. "
        f"Observation: cycle={int(cycle)}; qd={qd:.4f}; qc={qc:.4f}; ir={ir:.5f}; "
        f"tmax={tmax:.2f}; tavg={tavg:.2f}; charge_time={chargetime:.2f}. "
        f"Derived: qd_roll5={qd_roll5:.4f}; delta_qd={dqd_cycle:.5f}; "
        f"qd_slope5={qd_slope5:.6f}; ir_slope5={ir_slope5:.7f}; "
        f"trend={trend_label}; observed_stage={observed_stage}. "
        f"Forecast: horizon_cycles={int(forecast_horizon)}; "
        f"output=current_soh plus next_{int(forecast_horizon)}_cycle_soh."
    )
