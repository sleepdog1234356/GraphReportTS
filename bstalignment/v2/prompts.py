from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .battery_features import BASE_FEATURE_NAMES
from .battery_cache import BatteryOperatingContext


@dataclass(frozen=True)
class PromptResultV2:
    text: str
    metadata: dict[str, object]


GENERAL_PROMPT_METRICS = ("periodicity", "volatility_change", "dependence", "drift")
LEVELS = ("very low", "low", "medium", "high", "very high")
BATTERY_PROMPT_METRICS = ("charge_duration_variability", "discharge_current_variability")
GENERAL_PROMPT_CONTEXT_LENGTHS = (36, 96)


def _trend(values: np.ndarray, tolerance: float = 0.05) -> str:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return "unknown"
    y = values[finite]
    x = np.linspace(-1.0, 1.0, len(y))
    slope = float(np.polyfit(x, y, 1)[0])
    scale = float(np.nanstd(y)) + 1e-6
    if slope > tolerance * scale:
        return "rising"
    if slope < -tolerance * scale:
        return "falling"
    return "stable"


def _volatility(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return "unknown"
    ratio = float(np.std(finite) / (np.mean(np.abs(finite)) + 1e-6))
    return "low" if ratio < 0.1 else "medium" if ratio < 0.3 else "high"


def _frequency_label(frequency: object) -> str:
    if hasattr(frequency, "total_seconds"):
        seconds = float(frequency.total_seconds())
        if seconds % 3600 == 0:
            hours = int(seconds // 3600)
            return "hourly" if hours == 1 else f"{hours}-hour"
        if seconds % 60 == 0:
            return f"{int(seconds // 60)}-minute"
    return str(frequency).replace(" ", "-")


def _general_statistics(context: np.ndarray) -> dict[str, float | tuple[float, float, float]]:
    values = np.asarray(context, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] not in GENERAL_PROMPT_CONTEXT_LENGTHS:
        raise ValueError(
            f"general prompt context length must be one of {GENERAL_PROMPT_CONTEXT_LENGTHS}"
        )
    third = values.shape[0] // 3
    half = values.shape[0] // 2
    finite_count = np.isfinite(values).sum(axis=0)
    column_mean = np.divide(
        np.nansum(values, axis=0),
        finite_count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=finite_count > 0,
    )
    filled = np.where(np.isfinite(values), values, column_mean[None, :])
    robust_center = np.nanmedian(values, axis=0)
    robust_scale = 1.4826 * np.nanmedian(np.abs(values - robust_center[None, :]), axis=0)
    fallback_scale = np.nanstd(values, axis=0)
    scale = np.where(
        np.isfinite(robust_scale) & (robust_scale > 1e-6),
        robust_scale,
        np.where(np.isfinite(fallback_scale) & (fallback_scale > 1e-6), fallback_scale, 1.0),
    )
    normalized = (filled - robust_center[None, :]) / scale[None, :]

    def normalized_slopes(window: np.ndarray) -> np.ndarray:
        axis = np.linspace(-1.0, 1.0, window.shape[0])
        centered = window - window.mean(axis=0, keepdims=True)
        return (centered * axis[:, None]).sum(axis=0) / np.square(axis).sum()

    x = np.linspace(-1.0, 1.0, values.shape[0])
    slopes = ((filled - filled.mean(axis=0)) * x[:, None]).sum(axis=0) / np.square(x).sum()
    normalized_slope = slopes / scale
    early_slope = normalized_slopes(normalized[:third])
    recent_slope = normalized_slopes(normalized[-third:])
    falling = float(np.mean(normalized_slope < -0.1))
    rising = float(np.mean(normalized_slope > 0.1))
    stable = float(max(0.0, 1.0 - falling - rising))

    global_series = np.median(normalized, axis=1)
    global_axis = np.linspace(-1.0, 1.0, values.shape[0])
    global_trend = np.polyval(np.polyfit(global_axis, global_series, 1), global_axis)
    spectrum = np.abs(np.fft.rfft(global_series - global_trend))[1:] ** 2
    periodicity = float(spectrum.max() / spectrum.sum()) if spectrum.size and spectrum.sum() > 1e-12 else 0.0
    dominant_period = (
        float(values.shape[0] / (int(np.argmax(spectrum)) + 1))
        if spectrum.size and spectrum.sum() > 1e-12
        else 0.0
    )
    early_center = np.median(normalized[:third], axis=0)
    recent_center = np.median(normalized[-third:], axis=0)
    early_volatility = np.median(np.abs(normalized[:third] - early_center[None, :]), axis=0)
    recent_volatility = np.median(np.abs(normalized[-third:] - recent_center[None, :]), axis=0)
    volatility_change = float(
        np.clip(np.median(np.log((recent_volatility + 1e-4) / (early_volatility + 1e-4))), -4.0, 4.0)
    )

    selected_count = min(values.shape[1], 32)
    selected = np.linspace(0, values.shape[1] - 1, selected_count, dtype=np.int64)
    differences = np.diff(filled[:, selected], axis=0)
    if selected_count > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(differences, rowvar=False)
        off_diagonal = np.abs(correlation[~np.eye(selected_count, dtype=bool)])
        dependence = float(np.nanmedian(off_diagonal)) if np.isfinite(off_diagonal).any() else 0.0
    else:
        dependence = 0.0
    missing = float(1.0 - np.isfinite(values).mean())
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median[None, :]), axis=0)
    robust_z = np.abs(values - median[None, :]) / np.maximum(1.4826 * mad[None, :], 1e-6)
    anomaly = float(np.nanmean(robust_z > 4.0))
    first, second = filled[:half].mean(axis=0), filled[half:].mean(axis=0)
    drift = float(np.median(np.abs(second - first) / scale))
    return {
        "trend": (falling, stable, rising),
        "slope": float(np.median(normalized_slope)),
        "recent_slope": float(np.median(recent_slope)),
        "momentum": float(np.median(recent_slope - early_slope)),
        "end_position": float(np.median(normalized[-1])),
        "periodicity": periodicity,
        "dominant_period": dominant_period,
        "volatility_change": volatility_change,
        "dependence": dependence,
        "missing": missing,
        "anomaly": anomaly,
        "drift": drift,
        "level_shift": float(np.median((second - first) / scale)),
    }


def fit_general_prompt_thresholds(
    values: np.ndarray,
    numeric_starts: Sequence[int],
    prompt_len: int = 36,
    max_windows: int = 512,
) -> dict[str, tuple[float, float, float, float]]:
    """Fit prompt category boundaries from training-only earlier windows."""

    starts = np.asarray(numeric_starts, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("cannot fit prompt thresholds without training windows")
    if starts.size > max_windows:
        starts = starts[np.linspace(0, starts.size - 1, max_windows, dtype=np.int64)]
    collected: dict[str, list[float]] = {name: [] for name in GENERAL_PROMPT_METRICS}
    if int(prompt_len) not in GENERAL_PROMPT_CONTEXT_LENGTHS:
        raise ValueError(
            f"general prompt context length must be one of {GENERAL_PROMPT_CONTEXT_LENGTHS}"
        )
    for start in starts.tolist():
        if start < prompt_len:
            continue
        stats = _general_statistics(values[start - prompt_len : start])
        for name in GENERAL_PROMPT_METRICS:
            collected[name].append(float(stats[name]))
    if not collected[GENERAL_PROMPT_METRICS[0]]:
        raise ValueError(f"training windows do not contain a complete {prompt_len}-step prompt context")
    return {
        name: tuple(float(item) for item in np.quantile(series, (0.2, 0.4, 0.6, 0.8)))
        for name, series in collected.items()
    }


def _level(value: float, boundaries: Sequence[float] | None) -> str:
    if boundaries is None or len(boundaries) != 4:
        return "unavailable"
    # `left` is important for zero-inflated statistics.  With `right`, an all-zero
    # training distribution incorrectly labels zero missingness as "very high".
    return LEVELS[int(np.searchsorted(np.asarray(boundaries, dtype=np.float64), value, side="left"))]


def _signed_direction(value: float, tolerance: float = 0.1) -> str:
    if value > tolerance:
        return "up"
    if value < -tolerance:
        return "down"
    return "flat"


def _battery_feature(values: np.ndarray, name: str) -> np.ndarray:
    index = BASE_FEATURE_NAMES.index(name)
    return values[:, index].astype(np.float64, copy=False)


def _finite_variability(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return float("nan")
    return float(np.std(finite) / max(float(np.mean(np.abs(finite))), 1e-8))


def _battery_prompt_metrics(values: np.ndarray) -> dict[str, float]:
    duration = _battery_feature(values, "discharge_duration_s")
    discharged_ah = _battery_feature(values, "discharge_integral_ah")
    valid = np.isfinite(duration) & np.isfinite(discharged_ah) & (duration > 0.0) & (discharged_ah >= 0.0)
    mean_current = np.full(32, np.nan, dtype=np.float64)
    mean_current[valid] = discharged_ah[valid] * 3600.0 / duration[valid]
    return {
        "charge_duration_variability": _finite_variability(_battery_feature(values, "charge_duration_s")),
        "discharge_current_variability": _finite_variability(mean_current),
    }


def fit_battery_prompt_thresholds(
    prompt_windows: Iterable[np.ndarray],
    max_windows: int = 4096,
) -> dict[str, tuple[float, float]]:
    """Fit compact sensor-summary categories from training-cell windows only."""

    collected: dict[str, list[float]] = {name: [] for name in BATTERY_PROMPT_METRICS}
    seen = 0
    for raw in prompt_windows:
        values = np.asarray(raw, dtype=np.float32)
        if values.shape != (32, len(BASE_FEATURE_NAMES)):
            raise ValueError(f"battery prompt context must be [32,{len(BASE_FEATURE_NAMES)}]")
        if seen >= max_windows:
            break
        seen += 1
        metrics = _battery_prompt_metrics(values)
        for name, value in metrics.items():
            if np.isfinite(value):
                collected[name].append(float(value))
    if seen == 0:
        raise ValueError("cannot fit battery prompt thresholds without training windows")
    defaults = {
        "charge_duration_variability": (0.10, 0.25),
        "discharge_current_variability": (0.10, 0.25),
    }
    return {
        name: (
            tuple(float(item) for item in np.quantile(values, (1.0 / 3.0, 2.0 / 3.0)))
            if values
            else defaults[name]
        )
        for name, values in collected.items()
    }


def _variability_level(
    value: float,
    metric: str,
    thresholds: Mapping[str, Sequence[float]] | None,
) -> str:
    if not np.isfinite(value):
        return "unavailable"
    boundaries = tuple((thresholds or {}).get(metric, (0.10, 0.25)))
    if len(boundaries) != 2:
        raise ValueError(f"battery prompt threshold {metric} must contain two boundaries")
    labels = ("stable", "moderately variable", "variable")
    return labels[int(np.searchsorted(np.asarray(boundaries, dtype=np.float64), value, side="right"))]


def _compact_trend_label(label: str) -> str:
    return {"rising": "up", "falling": "down", "stable": "flat"}.get(label, label)


def _compact_variability_label(label: str) -> str:
    return {"stable": "steady", "moderately variable": "moderate", "variable": "variable"}.get(label, label)


def _compact_number(value: float) -> str:
    return f"{float(value):g}"


def _cell_specification(context: BatteryOperatingContext | None) -> str:
    if context is None:
        return "unavailable"
    form_factor = context.form_factor
    if form_factor and context.model and form_factor.lower() in context.model.lower():
        form_factor = None
    identity = " ".join(
        value
        for value in (context.manufacturer, context.model, form_factor, context.chemistry)
        if value
    )
    ratings: list[str] = []
    if context.nominal_capacity_ah is not None:
        ratings.append(f"{_compact_number(context.nominal_capacity_ah)}Ah")
    if context.nominal_voltage_v is not None:
        ratings.append(f"{_compact_number(context.nominal_voltage_v)}V")
    if context.voltage_window_v is not None:
        low, high = context.voltage_window_v
        ratings.append(f"{_compact_number(low)}-{_compact_number(high)}V")
    components = [value for value in (identity, " ".join(ratings)) if value]
    return " ".join(components) if components else "unavailable"


def _compact_protocol(protocol: str | None) -> str:
    if not protocol:
        return "unavailable"
    compact = protocol
    replacements = (
        ("constant-current", "CC"),
        (" cyclic discharge", " cyclic"),
        ("random-walk", "RW"),
        (" for ", " "),
        (" safety cutoff", " cutoff"),
        (" with scheduled variable duration and DOD below 80%", ", variable schedule, DOD<80%"),
        (" with ", " "),
        (" to ", " "),
        ("3.0V", "3V"),
        ("2.0V", "2V"),
        ("3V cutoff", "cutoff 3V"),
        ("; cutoff", " cutoff"),
        ("-newstructure", "-NS"),
    )
    for source, target in replacements:
        compact = compact.replace(source, target)
    return compact


def build_general_prompt(
    context: np.ndarray,
    columns: Sequence[str],
    frequency: object,
    pred_len: int,
    thresholds: Mapping[str, Sequence[float]] | None = None,
) -> PromptResultV2:
    """Compress only the earlier causal window into a short fixed-schema report."""

    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 2 or context.shape[0] not in GENERAL_PROMPT_CONTEXT_LENGTHS:
        raise ValueError(
            f"general prompt context length must be one of {GENERAL_PROMPT_CONTEXT_LENGTHS}"
        )
    context_length = int(context.shape[0])
    statistics = _general_statistics(context)
    falling, stable, rising = statistics["trend"]
    thresholds = thresholds or {}
    slope = float(statistics["slope"])
    recent_slope = float(statistics["recent_slope"])
    momentum = float(statistics["momentum"])
    periodicity = float(statistics["periodicity"])
    dominant_period = float(statistics["dominant_period"])
    volatility_change = float(statistics["volatility_change"])
    dependence = float(statistics["dependence"])
    level_shift = float(statistics["level_shift"])
    periodicity_boundaries = thresholds.get("periodicity")
    periodicity_floor = (
        float(periodicity_boundaries[0]) if periodicity_boundaries is not None and len(periodicity_boundaries) else 0.0
    )
    cycle = "none" if dominant_period <= 0.0 or periodicity <= periodicity_floor else f"{dominant_period:.1f} steps"
    variability_label = {
        "up": "increased",
        "down": "decreased",
        "flat": "steady",
    }[_signed_direction(volatility_change)]
    fields = [
        (
            f"Forecast the next {int(pred_len)} {_frequency_label(frequency)} steps for {len(columns)} variables. "
            f"The earlier {context_length}-step context before the numeric input is summarized below."
        ),
        (
            f"Trend: {_signed_direction(slope)} overall ({slope:+.3f}), "
            f"{_signed_direction(recent_slope)} recently ({recent_slope:+.3f}), "
            f"momentum {momentum:+.3f}; shares down {falling:.2f}, flat {stable:.2f}, up {rising:.2f}."
        ),
        (
            f"Cycle: {cycle}, {_level(periodicity, thresholds.get('periodicity'))} strength "
            f"({periodicity:.3f})."
        ),
        (
            f"Variability: {variability_label} ({np.exp(volatility_change):.2f}x recent/early). "
            f"Differenced dependence: {_level(dependence, thresholds.get('dependence'))} ({dependence:.3f}). "
            f"Level shift: {level_shift:+.3f} standard deviations."
        ),
    ]
    missing = float(statistics["missing"])
    anomaly = float(statistics["anomaly"])
    quality = []
    if missing >= 5e-4:
        quality.append(f"missing {missing:.3f}")
    if anomaly >= 5e-4:
        quality.append(f"outliers {anomaly:.3f}")
    if quality:
        fields.append(f"Data quality: {', '.join(quality)}.")
    return PromptResultV2(
        text=" ".join(fields),
        metadata={
            "context_length": context_length,
            "variables": len(columns),
            "frequency": _frequency_label(frequency),
            "pred_len": pred_len,
            "prediction_length": pred_len,
            "statistics": statistics,
        },
    )


def build_battery_prompt(
    prior_features: np.ndarray,
    prior_soh: np.ndarray | None,
    mode: str,
    *,
    operating_context: BatteryOperatingContext | Mapping[str, object] | None = None,
    thresholds: Mapping[str, Sequence[float]] | None = None,
) -> PromptResultV2:
    """Build a compact report from the earlier 32 cycles only."""

    if mode not in {"sensor_only", "soh_assisted"}:
        raise ValueError("battery prompt mode must be sensor_only or soh_assisted")
    values = np.asarray(prior_features, dtype=np.float32)
    if values.ndim != 2 or values.shape != (32, len(BASE_FEATURE_NAMES)):
        raise ValueError(f"battery prompt context must be [32,{len(BASE_FEATURE_NAMES)}]")

    context = (
        operating_context
        if isinstance(operating_context, BatteryOperatingContext)
        else BatteryOperatingContext.from_json(operating_context)
    )

    feature_index = {name: index for index, name in enumerate(BASE_FEATURE_NAMES)}

    def feature(name: str) -> np.ndarray:
        return values[:, feature_index[name]].astype(np.float64, copy=False)

    metrics = _battery_prompt_metrics(values)

    def charge_strategy() -> str:
        ratio = feature("cc_cv_ratio")
        finite = ratio[np.isfinite(ratio) & (ratio >= 0.0)]
        if finite.size < 3:
            return "unavailable"
        median = float(np.median(finite))
        if median > 3.0:
            phase = "CC-heavy short-CV"
        elif median < 0.75:
            phase = "CV-heavy"
        else:
            phase = "balanced CC-CV"
        duration = feature("charge_duration_s")
        variability = _variability_level(metrics["charge_duration_variability"], "charge_duration_variability", thresholds)
        return f"{phase}; {_compact_trend_label(_trend(duration))}/{_compact_variability_label(variability)}"

    def discharge_strategy() -> str:
        duration = feature("discharge_duration_s")
        discharged_ah = feature("discharge_integral_ah")
        valid = (
            np.isfinite(duration)
            & np.isfinite(discharged_ah)
            & (duration > 0.0)
            & (discharged_ah >= 0.0)
        )
        if valid.sum() < 3:
            return "unavailable"
        variability = _variability_level(
            metrics["discharge_current_variability"],
            "discharge_current_variability",
            thresholds,
        )
        return f"{_compact_variability_label(variability)}; {_compact_trend_label(_trend(duration))}"

    def coordinate_drift(name: str) -> str:
        position = feature(name)
        trend = _trend(position)
        if trend == "rising":
            return "higher"
        if trend == "falling":
            return "lower"
        return trend

    temperature_mean = feature("t_mean")
    temperature_peak = feature("t_max")
    duration = feature("charge_duration_s") + feature("discharge_duration_s")
    current_squared = feature("current_squared_integral_a2s")
    temperature_excursion = temperature_peak - temperature_mean
    stress = np.full(32, np.nan, dtype=np.float64)
    stress_valid = (
        np.isfinite(duration)
        & np.isfinite(current_squared)
        & np.isfinite(temperature_excursion)
        & (duration > 0.0)
    )
    stress[stress_valid] = (
        current_squared[stress_valid]
        / duration[stress_valid]
        * np.maximum(temperature_excursion[stress_valid], 0.0)
    )
    tail_columns = np.column_stack(
        (feature("charge_tail_voltage_duration_s"), feature("charge_tail_current_duration_s"))
    )
    tail_count = np.isfinite(tail_columns).sum(axis=1)
    tail_duration = np.divide(
        np.nansum(tail_columns, axis=1),
        tail_count,
        out=np.full(32, np.nan, dtype=np.float64),
        where=tail_count > 0,
    )

    soh_field = "Historical SOH: unavailable."
    soh_available = False
    if mode == "soh_assisted" and prior_soh is not None:
        soh = np.asarray(prior_soh, dtype=np.float32).reshape(-1)
        finite = soh[np.isfinite(soh)]
        if finite.size:
            soh_available = True
            soh_field = f"Historical SOH: available, {_trend(soh)}."
    declared_charge = _compact_protocol(context.charge_protocol if context else None)
    declared_discharge = _compact_protocol(context.discharge_protocol if context else None)
    fields = (
        "Task: SOH.",
        "Earlier: 32 cycles.",
        soh_field,
        f"Cell: {_cell_specification(context)}.",
        f"Charge: {declared_charge} ({charge_strategy()}).",
        f"Discharge: {declared_discharge} ({discharge_strategy()}).",
        f"Temp {_compact_trend_label(_trend(temperature_mean))}/{_compact_trend_label(_trend(temperature_peak))}; "
        f"stress/tail {_compact_trend_label(_trend(stress))}/{_compact_trend_label(_trend(tail_duration))}; "
        f"IC/DV {coordinate_drift('ic_primary_position')}/{coordinate_drift('dv_primary_position')}.",
        "Forecast: 20 cycles.",
    )
    return PromptResultV2(
        text=" ".join(fields),
        metadata={
            "context_length": 32,
            "mode": mode,
            "historical_soh_available": soh_available,
            "source": "declared_context_and_earlier_sensor_window" if context else "earlier_sensor_window_only",
            "operating_context": context.to_json() if context else None,
            "thresholds": {name: list(values) for name, values in (thresholds or {}).items()},
        },
    )
