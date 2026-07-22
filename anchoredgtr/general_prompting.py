"""Deterministic, observed-history prompts for general forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .general_protocol import FORMAL_HISTORY, FORMAL_HORIZONS


MAX_VARIABLE_SUMMARIES = 12
PRETOKEN_WORD_BUDGET = 192
_TREND_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GeneralPromptResult:
    prompt: str
    metadata: Mapping[str, object]


def _frequency_label(frequency: object) -> str:
    if isinstance(frequency, str):
        return frequency
    total_seconds = getattr(frequency, "total_seconds", None)
    if not callable(total_seconds):
        return str(frequency)
    seconds = float(total_seconds())
    if seconds > 0 and seconds.is_integer():
        whole_seconds = int(seconds)
        if whole_seconds % 3600 == 0:
            hours = whole_seconds // 3600
            return f"{hours} hour" if hours == 1 else f"{hours} hours"
        if whole_seconds % 60 == 0:
            minutes = whole_seconds // 60
            return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"
    return str(frequency)


def _format_float(value: float) -> str:
    return f"{0.0 if abs(value) < 0.00005 else value:.4f}"


def _selected_variable_indices(trends: np.ndarray) -> tuple[int, ...]:
    count = len(trends)
    if count <= MAX_VARIABLE_SUMMARIES:
        return tuple(range(count))
    indices = range(count)
    smallest = sorted(indices, key=lambda index: (abs(float(trends[index])), index))[:6]
    smallest_set = set(smallest)
    largest = sorted(
        (index for index in indices if index not in smallest_set),
        key=lambda index: (-abs(float(trends[index])), index),
    )[:6]
    return tuple(sorted((*smallest, *largest)))


def _token_count(prompt: str) -> int:
    return len(prompt.split())


def build_general_prompt_result(
    history: np.ndarray,
    columns: Sequence[str],
    frequency: object,
    pred_len: int,
) -> GeneralPromptResult:
    """Build one bounded prompt from a standardized observed history window."""

    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != FORMAL_HISTORY:
        raise ValueError(f"general prompt history must have shape ({FORMAL_HISTORY}, C)")
    if pred_len not in FORMAL_HORIZONS:
        raise ValueError(f"general prompt requires a formal horizon from {FORMAL_HORIZONS}")
    names = tuple(columns)
    if not names or len(names) != values.shape[1] or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("general prompt columns must be non-empty canonical names matching history")
    if not np.isfinite(values).all():
        raise ValueError("general prompt history must contain only finite standardized values")

    trends = values[-1] - values[0]
    volatility = np.std(values, axis=0)
    absolute_change = np.abs(np.diff(values, axis=0)).mean()
    increasing = int(np.count_nonzero(trends > _TREND_TOLERANCE))
    decreasing = int(np.count_nonzero(trends < -_TREND_TOLERANCE))
    flat = int(values.shape[1] - increasing - decreasing)
    frequency_text = _frequency_label(frequency)
    variable_count = values.shape[1]
    prefix = (
        "Task: multivariate time-series forecasting.\n"
        f"Observation: {FORMAL_HISTORY} past steps sampled every {frequency_text}; {variable_count} variables are observed.\n"
        "Window summary: "
        f"aggregate mean={_format_float(float(values.mean()))}, "
        f"standard deviation={_format_float(float(values.std()))}, "
        f"mean absolute change={_format_float(float(absolute_change))}, "
        f"trend balance={increasing} increasing/{decreasing} decreasing/{flat} approximately flat.\n"
    )
    suffix = (
        f"\nInstruction: predict all {variable_count} variables for the next {int(pred_len)} steps.\n"
        "Use only the observed window."
    )

    selected = _selected_variable_indices(trends)
    summaries: list[str] = []
    truncated = False
    for index in selected:
        summary = (
            f"{names[index]}(last={_format_float(float(values[-1, index]))}, "
            f"trend={_format_float(float(trends[index]))}, "
            f"volatility={_format_float(float(volatility[index]))})"
        )
        candidate_summaries = (*summaries, summary)
        candidate = prefix + "Variable summaries: " + "; ".join(candidate_summaries) + "." + suffix
        if _token_count(candidate) > PRETOKEN_WORD_BUDGET:
            truncated = True
            break
        summaries.append(summary)
    if len(summaries) != len(selected):
        truncated = True
    summary_text = "; ".join(summaries) if summaries else "none"
    prompt = prefix + f"Variable summaries: {summary_text}." + suffix
    metadata = {
        "pretoken_word_count": _token_count(prompt),
        "pretoken_word_budget": PRETOKEN_WORD_BUDGET,
        "pretoken_word_truncated": truncated,
        "frequency": frequency_text,
        "variable_count": variable_count,
        "summary_count": len(summaries),
    }
    return GeneralPromptResult(prompt=prompt, metadata=metadata)


def build_general_prompt(
    history: np.ndarray,
    columns: Sequence[str],
    frequency: object,
    pred_len: int,
) -> str:
    """Return the stable text portion of :func:`build_general_prompt_result`."""

    return build_general_prompt_result(history, columns, frequency, pred_len).prompt
