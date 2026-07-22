"""Sensor-only, physics-preserving per-cycle battery features for GTR.

All numeric predictors in this module are computed from a *single* cycle's
time, voltage, current and temperature arrays.  Cycle number, capacity labels,
SOH, EFC and cumulative throughput are deliberately absent from the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter


FEATURE_SCHEMA_VERSION = "battery_sensor_features.1.0"
CURVE_POINTS = 128
CURVE_AXIS_SCHEMA_VERSION = "battery_curve_axes.1.0"
IC_VOLTAGE_RANGE_V = (2.0, 5.0)
DV_NORMALIZED_Q_RANGE = (0.0, 1.0)
IC_VOLTAGE_AXIS = np.linspace(*IC_VOLTAGE_RANGE_V, CURVE_POINTS, dtype=np.float32)
DV_NORMALIZED_Q_AXIS = np.linspace(*DV_NORMALIZED_Q_RANGE, CURVE_POINTS, dtype=np.float32)


def _curve_axis_sha256() -> str:
    digest = sha256(CURVE_AXIS_SCHEMA_VERSION.encode("ascii"))
    digest.update(b"ic:voltage_v\0")
    digest.update(IC_VOLTAGE_AXIS.astype("<f4", copy=False).tobytes())
    digest.update(b"dv:normalized_q\0")
    digest.update(DV_NORMALIZED_Q_AXIS.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


CURVE_AXIS_SHA256 = _curve_axis_sha256()

GLOBAL_FEATURE_NAMES = (
    "v_mean",
    "v_std",
    "v_q90_q10",
    "v_slope",
    "i_mean",
    "i_std",
    "i_rms",
    "i_abs_q90_q10",
    "t_mean",
    "t_max",
    "t_rise",
    "t_slope",
)

_TAIL_STAT_NAMES = (
    "mean",
    "std",
    "kurtosis",
    "skewness",
    "duration_s",
    "charge_ah",
    "slope_per_s",
    "entropy",
)
CHARGE_TAIL_FEATURE_NAMES = tuple(
    f"charge_tail_{source}_{stat}"
    for source in ("voltage", "current")
    for stat in _TAIL_STAT_NAMES
)
PROCESS_FEATURE_NAMES = (
    "charge_duration_s",
    "discharge_duration_s",
    "cc_cv_ratio",
    "charge_integral_ah",
    "discharge_integral_ah",
    "current_squared_integral_a2s",
)
NON_IC_FEATURE_NAMES = GLOBAL_FEATURE_NAMES + CHARGE_TAIL_FEATURE_NAMES + PROCESS_FEATURE_NAMES

_DESCRIPTOR_SUFFIXES = (
    "primary_position",
    "primary_amplitude",
    "primary_fwhm",
    "primary_local_area",
    "secondary_position",
    "secondary_amplitude",
    "peak_spacing",
    "centroid",
)
IC_FEATURE_NAMES = tuple(f"ic_{name}" for name in _DESCRIPTOR_SUFFIXES)
DV_FEATURE_NAMES = tuple(f"dv_{name}" for name in _DESCRIPTOR_SUFFIXES)
BASE_FEATURE_NAMES = NON_IC_FEATURE_NAMES + IC_FEATURE_NAMES + DV_FEATURE_NAMES

assert len(NON_IC_FEATURE_NAMES) == 34
assert len(BASE_FEATURE_NAMES) == 50


@dataclass(frozen=True)
class CycleSignals:
    """Cleaned physical signals for one cycle."""

    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    temperature_mask: np.ndarray

    @property
    def size(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.size > 1 else 0.0


@dataclass(frozen=True)
class ChargePhases:
    charge_mask: np.ndarray
    discharge_mask: np.ndarray
    cc_mask: np.ndarray
    cv_mask: np.ndarray
    charge_sign: float
    confidence: float
    coverage: float
    cv_fraction_bounds: tuple[float, float]


@dataclass(frozen=True)
class FeatureBlock:
    values: np.ndarray
    observed_mask: np.ndarray
    reliability: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class DerivativeCurve:
    """A derivative curve with its physical (not rank-normalized) x-axis."""

    x: np.ndarray
    y: np.ndarray
    observed_mask: np.ndarray
    quality: float
    kind: str


@dataclass(frozen=True)
class DescriptorResult(FeatureBlock):
    """Peak descriptors; supports the mapping-style access used in the design."""

    def __getitem__(self, key: str) -> float:
        suffixes = [name.split("_", 1)[1] for name in self.feature_names]
        if key not in suffixes:
            raise KeyError(key)
        return float(self.values[suffixes.index(key)])


@dataclass(frozen=True)
class CycleFeatureResult:
    values: np.ndarray
    observed_mask: np.ndarray
    reliability: np.ndarray
    feature_names: tuple[str, ...]
    ic_curve: DerivativeCurve
    dv_curve: DerivativeCurve
    time_coverage: float

    @property
    def base_values(self) -> np.ndarray:
        return self.values


def _as_1d(values: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def canonicalize_cycle(
    time: Sequence[float] | np.ndarray,
    voltage: Sequence[float] | np.ndarray,
    current: Sequence[float] | np.ndarray,
    temperature: Sequence[float] | np.ndarray | None,
) -> CycleSignals:
    """Sort, deduplicate and validate one cycle without inventing measurements."""

    t, v, i, temp = map(_as_1d, (time, voltage, current, temperature))
    if temp.size == 0:
        temp = np.full(min(t.size, v.size, i.size), np.nan, dtype=np.float64)
    length = min(t.size, v.size, i.size, temp.size)
    if length < 4:
        raise ValueError("battery cycle requires at least four time/V/I observations")
    stacked = np.column_stack((t[:length], v[:length], i[:length], temp[:length]))
    stacked = stacked[np.isfinite(stacked[:, :3]).all(axis=1)]
    if len(stacked) < 4:
        raise ValueError("battery cycle requires at least four finite time/V/I observations")
    order = np.argsort(stacked[:, 0], kind="stable")
    stacked = stacked[order]
    _, unique_index = np.unique(stacked[:, 0], return_index=True)
    stacked = stacked[np.sort(unique_index)]
    if len(stacked) < 4 or stacked[-1, 0] <= stacked[0, 0]:
        raise ValueError("battery cycle has insufficient unique timestamps")
    temperature_mask = np.isfinite(stacked[:, 3])
    return CycleSignals(
        time=stacked[:, 0].astype(np.float32),
        voltage=stacked[:, 1].astype(np.float32),
        current=stacked[:, 2].astype(np.float32),
        temperature=np.where(temperature_mask, stacked[:, 3], 0.0).astype(np.float32),
        temperature_mask=temperature_mask.astype(bool),
    )


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.ptp(x)) <= 1e-12:
        return 0.0
    x0 = x.astype(np.float64) - float(np.mean(x))
    y0 = y.astype(np.float64) - float(np.mean(y))
    return float(np.dot(x0, y0) / max(float(np.dot(x0, x0)), 1e-12))


def _time_coverage(time: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    total = max(float(time[-1] - time[0]), 1e-12)
    selected = time[mask]
    return float(np.clip((selected[-1] - selected[0]) / total, 0.0, 1.0))


def extract_global_features(signals: CycleSignals) -> FeatureBlock:
    t = signals.time.astype(np.float64)
    v = signals.voltage.astype(np.float64)
    i = signals.current.astype(np.float64)
    tm = signals.temperature_mask
    temp = signals.temperature.astype(np.float64)
    values = np.zeros(12, dtype=np.float32)
    mask = np.ones(12, dtype=bool)
    reliability = np.ones(12, dtype=np.float32)

    values[:8] = (
        np.mean(v),
        np.std(v),
        np.quantile(v, 0.9) - np.quantile(v, 0.1),
        _linear_slope(t, v),
        np.mean(i),
        np.std(i),
        np.sqrt(np.mean(i * i)),
        np.quantile(np.abs(i), 0.9) - np.quantile(np.abs(i), 0.1),
    )
    if tm.sum() >= 2:
        valid_t = t[tm]
        valid_temp = temp[tm]
        values[8:] = (
            np.mean(valid_temp),
            np.max(valid_temp),
            valid_temp[-1] - valid_temp[0],
            _linear_slope(valid_t, valid_temp),
        )
        reliability[8:] = min(float(tm.mean()), _time_coverage(t, tm))
    else:
        mask[8:] = False
        reliability[8:] = 0.0
    return FeatureBlock(values, mask, reliability, GLOBAL_FEATURE_NAMES)


def _duration_for_mask(time: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    indices = np.flatnonzero(mask)
    dt = np.diff(time, prepend=time[0])
    return float(np.sum(np.maximum(dt[indices], 0.0)))


def _integral_for_mask(time: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    indices = np.flatnonzero(mask)
    total = 0.0
    for runs in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
        if len(runs) >= 2:
            total += float(trapezoid(values[runs], time[runs]))
    return total


def detect_charge_phases(signals: CycleSignals) -> ChargePhases:
    """Infer charge polarity and CC/CV phases from current and voltage only."""

    t = signals.time.astype(np.float64)
    v = signals.voltage.astype(np.float64)
    i = signals.current.astype(np.float64)
    active_threshold = max(float(np.quantile(np.abs(i), 0.9)) * 0.02, 1e-8)
    pos = i > active_threshold
    neg = i < -active_threshold

    def voltage_gain(mask: np.ndarray) -> tuple[float, int]:
        if mask.sum() < 4:
            return -np.inf, int(mask.sum())
        return _linear_slope(t[mask], v[mask]), int(mask.sum())

    pos_slope, pos_count = voltage_gain(pos)
    neg_slope, neg_count = voltage_gain(neg)
    if np.isfinite(pos_slope) or np.isfinite(neg_slope):
        charge_sign = 1.0 if pos_slope >= neg_slope else -1.0
    else:
        charge_sign = 1.0 if pos_count >= neg_count else -1.0
    signed_current = charge_sign * i
    charge = signed_current > active_threshold
    discharge = signed_current < -active_threshold

    cv = np.zeros(signals.size, dtype=bool)
    cc = charge.copy()
    charge_indices = np.flatnonzero(charge)
    confidence = 0.0
    if charge_indices.size >= 8:
        charge_v = v[charge_indices]
        upper = float(np.quantile(charge_v, 0.95))
        plateau_width = max(0.02, min(0.2, 0.08 * max(float(np.ptp(charge_v)), 0.25)))
        plateau = charge & (v >= upper - plateau_width)
        candidates = np.flatnonzero(plateau)
        if candidates.size >= 4:
            onset = int(candidates[0])
            after_onset = charge & (np.arange(signals.size) >= onset)
            start_abs_i = float(np.median(np.abs(i[np.flatnonzero(after_onset)[: min(8, after_onset.sum())]])))
            relative_i = np.abs(i) / max(start_abs_i, 1e-8)
            cv = after_onset & (relative_i >= 0.2) & (relative_i <= 1.0)
            if cv.sum() >= 4:
                cc = charge & ~cv
                current_drop = 1.0 - float(np.median(relative_i[np.flatnonzero(cv)[-min(8, cv.sum()) :]]))
                confidence = float(np.clip(0.5 * (cv.sum() / charge.sum()) + 0.5 * max(current_drop, 0.0), 0.0, 1.0))
            else:
                cv[:] = False
    coverage = float(charge.mean())
    return ChargePhases(
        charge_mask=charge,
        discharge_mask=discharge,
        cc_mask=cc,
        cv_mask=cv,
        charge_sign=charge_sign,
        confidence=confidence,
        coverage=coverage,
        cv_fraction_bounds=(0.2, 1.0),
    )


def _shape_statistics(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return mean, std, 0.0, 0.0
    z = (values - mean) / std
    return mean, std, float(np.mean(z**4) - 3.0), float(np.mean(z**3))


def _histogram_entropy(values: np.ndarray) -> float:
    if len(values) < 2 or float(np.ptp(values)) <= 1e-12:
        return 0.0
    bins = int(np.clip(np.sqrt(len(values)), 4, 16))
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(bins))


def _tail_block(signals: CycleSignals, selected: np.ndarray, source: str) -> FeatureBlock:
    names = tuple(f"charge_tail_{source}_{stat}" for stat in _TAIL_STAT_NAMES)
    values = np.zeros(8, dtype=np.float32)
    mask = np.zeros(8, dtype=bool)
    reliability = np.zeros(8, dtype=np.float32)
    if selected.sum() < 8:
        return FeatureBlock(values, mask, reliability, names)
    t = signals.time[selected].astype(np.float64)
    y = (signals.voltage if source == "voltage" else np.abs(signals.current))[selected].astype(np.float64)
    full_current = np.abs(signals.current[selected]).astype(np.float64)
    duration = float(t[-1] - t[0])
    charge_ah = float(trapezoid(full_current, t) / 3600.0)
    values[:] = (*_shape_statistics(y), duration, charge_ah, _linear_slope(t, y), _histogram_entropy(y))
    mask[:] = np.isfinite(values)
    values[~mask] = 0.0
    quality = float(np.clip(selected.mean() * 4.0, 0.0, 1.0))
    reliability[mask] = quality
    return FeatureBlock(values, mask, reliability, names)


def extract_charge_tail_features(signals: CycleSignals, phases: ChargePhases | None = None) -> FeatureBlock:
    phases = phases or detect_charge_phases(signals)
    charge = phases.charge_mask
    values = np.zeros(16, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    reliability = np.zeros(16, dtype=np.float32)
    if charge.sum() >= 8:
        charge_v = signals.voltage[charge]
        v_end = float(np.median(np.sort(charge_v)[-min(8, len(charge_v)) :]))
        voltage_window = charge & (signals.voltage >= v_end - 0.2) & (signals.voltage <= v_end + 1e-6)
        cv_indices = np.flatnonzero(phases.cv_mask)
        if cv_indices.size:
            first = cv_indices[: min(8, cv_indices.size)]
            i_cv_start = float(np.median(np.abs(signals.current[first])))
            relative = np.abs(signals.current) / max(i_cv_start, 1e-8)
            current_window = phases.cv_mask & (relative >= 0.2) & (relative <= 1.0)
        else:
            current_window = np.zeros(signals.size, dtype=bool)
        blocks = (
            _tail_block(signals, voltage_window, "voltage"),
            _tail_block(signals, current_window, "current"),
        )
        for index, block in enumerate(blocks):
            slc = slice(index * 8, (index + 1) * 8)
            values[slc] = block.values
            mask[slc] = block.observed_mask
            reliability[slc] = block.reliability * max(phases.confidence, 0.25)
    return FeatureBlock(values, mask, reliability, CHARGE_TAIL_FEATURE_NAMES)


def extract_process_features(signals: CycleSignals, phases: ChargePhases | None = None) -> FeatureBlock:
    phases = phases or detect_charge_phases(signals)
    t = signals.time.astype(np.float64)
    signed_i = phases.charge_sign * signals.current.astype(np.float64)
    charge_integral = _integral_for_mask(t, np.maximum(signed_i, 0.0), phases.charge_mask) / 3600.0
    discharge_integral = _integral_for_mask(t, np.maximum(-signed_i, 0.0), phases.discharge_mask) / 3600.0
    cc_duration = _duration_for_mask(t, phases.cc_mask)
    cv_duration = _duration_for_mask(t, phases.cv_mask)
    values = np.asarray(
        (
            _duration_for_mask(t, phases.charge_mask),
            _duration_for_mask(t, phases.discharge_mask),
            cc_duration / max(cv_duration, 1e-8),
            charge_integral,
            discharge_integral,
            float(trapezoid(signed_i * signed_i, t)),
        ),
        dtype=np.float32,
    )
    mask = np.asarray(
        (
            phases.charge_mask.sum() >= 2,
            phases.discharge_mask.sum() >= 2,
            phases.cc_mask.sum() >= 2 and phases.cv_mask.sum() >= 2,
            phases.charge_mask.sum() >= 2,
            phases.discharge_mask.sum() >= 2,
            signals.size >= 2,
        ),
        dtype=bool,
    )
    values[~mask] = 0.0
    reliability = np.zeros(6, dtype=np.float32)
    reliability[mask] = np.asarray(
        (
            phases.coverage,
            float(phases.discharge_mask.mean()),
            phases.confidence,
            phases.coverage,
            float(phases.discharge_mask.mean()),
            1.0,
        ),
        dtype=np.float32,
    )[mask]
    return FeatureBlock(values, mask, reliability, PROCESS_FEATURE_NAMES)


def extract_non_ic_features(signals: CycleSignals) -> FeatureBlock:
    phases = detect_charge_phases(signals)
    blocks = (
        extract_global_features(signals),
        extract_charge_tail_features(signals, phases),
        extract_process_features(signals, phases),
    )
    return FeatureBlock(
        values=np.concatenate([block.values for block in blocks]).astype(np.float32),
        observed_mask=np.concatenate([block.observed_mask for block in blocks]).astype(bool),
        reliability=np.concatenate([block.reliability for block in blocks]).astype(np.float32),
        feature_names=NON_IC_FEATURE_NAMES,
    )


def _empty_curve(kind: str, points: int) -> DerivativeCurve:
    if points != CURVE_POINTS:
        raise ValueError(f"BatteryGTR derivative curves require exactly {CURVE_POINTS} points")
    if kind == "ic":
        axis = IC_VOLTAGE_AXIS
    elif kind == "dv":
        axis = DV_NORMALIZED_Q_AXIS
    else:
        raise ValueError(f"unsupported battery derivative curve kind: {kind}")
    return DerivativeCurve(
        x=axis.copy(),
        y=np.zeros(points, dtype=np.float32),
        observed_mask=np.zeros(points, dtype=bool),
        quality=0.0,
        kind=kind,
    )


def _collapse_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, starts, counts = np.unique(x, return_index=True, return_counts=True)
    collapsed = np.asarray([np.median(y[start : start + count]) for start, count in zip(starts, counts)])
    return unique.astype(np.float64), collapsed.astype(np.float64)


def _smooth(values: np.ndarray) -> np.ndarray:
    window = min(11, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values
    return savgol_filter(values, window_length=window, polyorder=min(3, window - 2), mode="interp")


def _curve_quality(x: np.ndarray, y: np.ndarray, source_count: int, expected_count: int) -> float:
    finite_ratio = float(np.isfinite(y).mean())
    coverage = float(np.clip(source_count / max(expected_count, 1), 0.0, 1.0))
    noise = np.diff(y)
    noise_mad = float(np.median(np.abs(noise - np.median(noise)))) if noise.size else np.inf
    signal_scale = float(np.quantile(np.abs(y), 0.9)) + 1e-8
    smoothness = float(np.clip(1.0 - noise_mad / signal_scale, 0.0, 1.0))
    peaks, props = find_peaks(np.abs(y), prominence=max(signal_scale * 0.03, 1e-10))
    prominence = float(np.max(props.get("prominences", np.asarray([0.0])))) if peaks.size else 0.0
    peak_score = float(np.clip(prominence / signal_scale, 0.0, 1.0))
    span_score = float(np.clip(np.ptp(x) / max(abs(float(np.mean(x))), 1e-6), 0.0, 1.0))
    return float(np.clip(0.25 * coverage + 0.25 * finite_ratio + 0.25 * smoothness + 0.15 * peak_score + 0.10 * span_score, 0.0, 1.0))


def _charge_capacity(signals: CycleSignals) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    phases = detect_charge_phases(signals)
    indices = np.flatnonzero(phases.charge_mask)
    if indices.size < 12:
        return None
    t = signals.time[indices].astype(np.float64)
    v = signals.voltage[indices].astype(np.float64)
    current = phases.charge_sign * signals.current[indices].astype(np.float64)
    q = cumulative_trapezoid(np.maximum(current, 0.0), t, initial=0.0) / 3600.0
    valid = np.isfinite(t) & np.isfinite(v) & np.isfinite(q)
    if valid.sum() < 12 or np.ptp(v[valid]) < 0.02 or np.ptp(q[valid]) <= 1e-9:
        return None
    return v[valid], q[valid], indices[valid]


def compute_ic_curve(signals: CycleSignals, points: int = CURVE_POINTS) -> DerivativeCurve:
    """Compute dQ/dV on the versioned 2--5 V physical voltage grid."""

    if points != CURVE_POINTS:
        raise ValueError(f"BatteryGTR IC curves require exactly {CURVE_POINTS} points")
    source = _charge_capacity(signals)
    if source is None:
        return _empty_curve("ic", points)
    voltage, q, _ = source
    voltage, q = _collapse_coordinates(voltage, q)
    if len(voltage) < 8:
        return _empty_curve("ic", points)
    grid = IC_VOLTAGE_AXIS.astype(np.float64)
    q_grid = PchipInterpolator(voltage, q, extrapolate=False)(grid)
    finite = np.isfinite(q_grid)
    if finite.sum() < 8:
        return _empty_curve("ic", points)
    curve = np.zeros(points, dtype=np.float64)
    derivative = np.gradient(_smooth(q_grid[finite]), grid[finite])
    derivative_finite = np.isfinite(derivative)
    valid_indices = np.flatnonzero(finite)
    mask = np.zeros(points, dtype=bool)
    mask[valid_indices[derivative_finite]] = True
    curve[mask] = derivative[derivative_finite]
    quality = _curve_quality(grid[mask], curve[mask], len(voltage), signals.size)
    return DerivativeCurve(grid.astype(np.float32), curve.astype(np.float32), mask, quality, "ic")


def compute_dv_curve(signals: CycleSignals, points: int = CURVE_POINTS) -> DerivativeCurve:
    """Compute dV/dQ on the versioned normalized-Q axis in [0, 1]."""

    if points != CURVE_POINTS:
        raise ValueError(f"BatteryGTR DV curves require exactly {CURVE_POINTS} points")
    source = _charge_capacity(signals)
    if source is None:
        return _empty_curve("dv", points)
    voltage, q, _ = source
    q_span = float(np.ptp(q))
    if q_span <= 1e-9:
        return _empty_curve("dv", points)
    normalized_q = (q - float(np.min(q))) / q_span
    normalized_q, voltage = _collapse_coordinates(normalized_q, voltage)
    if len(normalized_q) < 8:
        return _empty_curve("dv", points)
    grid = DV_NORMALIZED_Q_AXIS.astype(np.float64)
    v_grid = PchipInterpolator(normalized_q, voltage, extrapolate=False)(grid)
    finite = np.isfinite(v_grid)
    if finite.sum() < 8:
        return _empty_curve("dv", points)
    curve = np.zeros(points, dtype=np.float64)
    derivative = np.gradient(_smooth(v_grid[finite]), grid[finite])
    derivative_finite = np.isfinite(derivative)
    valid_indices = np.flatnonzero(finite)
    mask = np.zeros(points, dtype=bool)
    mask[valid_indices[derivative_finite]] = True
    curve[mask] = derivative[derivative_finite]
    quality = _curve_quality(grid[mask], curve[mask], len(normalized_q), signals.size)
    return DerivativeCurve(grid.astype(np.float32), curve.astype(np.float32), mask, quality, "dv")


def extract_peak_descriptors(
    x: np.ndarray,
    y: np.ndarray,
    observed_mask: np.ndarray | None = None,
    quality: float = 1.0,
    prefix: str = "ic",
) -> DescriptorResult:
    """Extract two extrema and physical-coordinate geometry from an IC/DV curve."""

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if observed_mask is not None:
        valid &= np.asarray(observed_mask, dtype=bool).reshape(-1)
    names = tuple(f"{prefix}_{name}" for name in _DESCRIPTOR_SUFFIXES)
    values = np.zeros(8, dtype=np.float32)
    mask = np.zeros(8, dtype=bool)
    reliability = np.zeros(8, dtype=np.float32)
    if valid.sum() < 8 or np.ptp(x[valid]) <= 1e-12:
        return DescriptorResult(values, mask, reliability, names)
    xv, yv = x[valid], y[valid]
    magnitude = np.abs(yv)
    scale = max(float(np.quantile(magnitude, 0.9)), 1e-10)
    peaks, properties = find_peaks(magnitude, prominence=scale * 0.03, distance=max(len(magnitude) // 16, 1))
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(magnitude))])
        prominences = np.asarray([float(magnitude[peaks[0]])])
    else:
        prominences = properties["prominences"]
    order = np.argsort(prominences)[::-1]
    primary = int(peaks[order[0]])
    half_height = 0.5 * float(magnitude[primary])
    left = primary
    while left > 0 and magnitude[left] >= half_height:
        left -= 1
    right = primary
    while right < len(magnitude) - 1 and magnitude[right] >= half_height:
        right += 1
    left_x = float(xv[left])
    right_x = float(xv[right])
    local = (xv >= left_x) & (xv <= right_x)
    values[:4] = (
        xv[primary],
        yv[primary],
        max(right_x - left_x, 0.0),
        float(trapezoid(np.abs(yv[local]), xv[local])) if local.sum() >= 2 else 0.0,
    )
    mask[:4] = True
    if len(order) >= 2:
        secondary = int(peaks[order[1]])
        values[4:7] = (xv[secondary], yv[secondary], abs(xv[secondary] - xv[primary]))
        mask[4:7] = True
    weights = magnitude + 1e-12
    values[7] = float(np.sum(xv * weights) / np.sum(weights))
    mask[7] = True
    reliability[mask] = float(np.clip(quality, 0.0, 1.0))
    return DescriptorResult(values, mask, reliability, names)


def extract_cycle_features(signals: CycleSignals) -> CycleFeatureResult:
    non_ic = extract_non_ic_features(signals)
    ic_curve = compute_ic_curve(signals)
    dv_curve = compute_dv_curve(signals)
    ic = extract_peak_descriptors(
        ic_curve.x, ic_curve.y, ic_curve.observed_mask, ic_curve.quality, prefix="ic"
    )
    dv = extract_peak_descriptors(
        dv_curve.x, dv_curve.y, dv_curve.observed_mask, dv_curve.quality, prefix="dv"
    )
    values = np.concatenate((non_ic.values, ic.values, dv.values)).astype(np.float32)
    mask = np.concatenate((non_ic.observed_mask, ic.observed_mask, dv.observed_mask)).astype(bool)
    reliability = np.concatenate((non_ic.reliability, ic.reliability, dv.reliability)).astype(np.float32)
    values[~mask] = 0.0
    reliability = np.clip(reliability, 0.0, 1.0)
    return CycleFeatureResult(
        values=values,
        observed_mask=mask,
        reliability=reliability,
        feature_names=BASE_FEATURE_NAMES,
        ic_curve=ic_curve,
        dv_curve=dv_curve,
        time_coverage=signals.duration,
    )


def extract_cycle_from_mapping(cycle: Mapping[str, object]) -> CycleFeatureResult:
    """Generic V/I/T/time boundary shared by MIT and processed XJTU inputs."""

    def first(keys: tuple[str, ...]):
        for key in keys:
            if key in cycle:
                return cycle[key]
        return None

    signals = canonicalize_cycle(
        first(("time", "t", "Time")),
        first(("voltage", "V", "Voltage")),
        first(("current", "I", "Current")),
        first(("temperature", "T", "Temperature", "Temp")),
    )
    return extract_cycle_features(signals)
