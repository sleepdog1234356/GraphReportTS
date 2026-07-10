from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


BATTERY_INPUT_CYCLES = 32
BATTERY_PREDICTION_CYCLES = 20
BATTERY_TARGET_PROTOCOL = "32-observed-20-future-only-full-horizon"
BATTERY_CYCLE_SCALE_PROTOCOL = "train-split-max-cycle-id-no-clip"
FORMAL_RUN_PROTOCOL_FIELDS: Dict[str, Dict[str, int]] = {
    "main": {"history_len": BATTERY_INPUT_CYCLES, "pred_len": BATTERY_PREDICTION_CYCLES},
    "baseline": {"input_len": BATTERY_INPUT_CYCLES, "pred_len": BATTERY_PREDICTION_CYCLES},
    "ablation": {"history_len": BATTERY_INPUT_CYCLES, "pred_len": BATTERY_PREDICTION_CYCLES},
}


def require_formal_battery_protocol(
    *,
    observed_cycles: int,
    prediction_cycles: int,
    context: str,
) -> None:
    if observed_cycles == BATTERY_INPUT_CYCLES and prediction_cycles == BATTERY_PREDICTION_CYCLES:
        return
    raise ValueError(
        f"{context} requires exactly {BATTERY_INPUT_CYCLES} observed cycles and "
        f"{BATTERY_PREDICTION_CYCLES} future-only targets; got "
        f"observed_cycles={observed_cycles}, prediction_cycles={prediction_cycles}"
    )


def _has_exact_protocol_fields(args: Any, expected: Mapping[str, int]) -> bool:
    if not isinstance(args, dict):
        return False
    return all(type(args.get(name)) is int and args[name] == value for name, value in expected.items())


def run_config_matches(
    config_path: str | Path,
    *,
    training_strategy_version: str,
    stage: str | None,
) -> bool:
    if stage is not None and stage not in FORMAL_RUN_PROTOCOL_FIELDS:
        raise ValueError(f"Unknown formal battery stage: {stage}")
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    root_matches = (
        isinstance(config, dict)
        and type(config.get("training_strategy_version")) is str
        and config["training_strategy_version"] == training_strategy_version
    )
    return root_matches and (
        stage is None or _has_exact_protocol_fields(config.get("args"), FORMAL_RUN_PROTOCOL_FIELDS[stage])
    )


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal battery protocol metadata")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-formal-protocol")
    validate.add_argument("--observed-cycles", type=int, required=True)
    validate.add_argument("--prediction-cycles", type=int, required=True)
    validate.add_argument("--context", required=True)

    match = subparsers.add_parser("run-config-matches")
    match.add_argument("--config", required=True)
    match.add_argument("--training-strategy-version", required=True)
    match.add_argument("--stage", choices=sorted(FORMAL_RUN_PROTOCOL_FIELDS), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    if args.command == "run-config-matches":
        return 0 if run_config_matches(
            args.config,
            training_strategy_version=args.training_strategy_version,
            stage=args.stage,
        ) else 1
    try:
        require_formal_battery_protocol(
            observed_cycles=args.observed_cycles,
            prediction_cycles=args.prediction_cycles,
            context=args.context,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


def split_mit_items(
    items: Sequence[Any],
    *,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, list[Any]]:
    values = list(items)
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    rng.shuffle(order)
    n_train = int(len(order) * train_ratio)
    n_val = int(len(order) * val_ratio)
    train = [values[int(index)] for index in order[:n_train]]
    val = [values[int(index)] for index in order[n_train : n_train + n_val]]
    test = [values[int(index)] for index in order[n_train + n_val :]]
    return {"train": train, "val": val, "test": test, "all": values}


def split_processed_items(items: Sequence[Any], *, seed: int) -> Dict[str, list[Any]]:
    values = list(items)
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    rng.shuffle(order)
    if len(order) >= 3:
        n_train = max(1, int(len(order) * 0.7))
        n_val = max(1, int(len(order) * 0.15))
        if n_train + n_val >= len(order):
            n_train = max(1, len(order) - 2)
            n_val = 1
    elif len(order) == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = len(order), 0
    return {
        "train": [values[int(index)] for index in order[:n_train]],
        "val": [values[int(index)] for index in order[n_train : n_train + n_val]],
        "test": [values[int(index)] for index in order[n_train + n_val :]],
        "all": values,
    }


def fit_cycle_scale(cycle_id_arrays: Iterable[np.ndarray], max_cycles: int | None) -> float:
    maximum = 1.0
    for values in cycle_id_arrays:
        cycle_ids = np.asarray(values, dtype=np.float64).reshape(-1)
        if max_cycles is not None:
            cycle_ids = cycle_ids[: int(max_cycles)]
        finite = cycle_ids[np.isfinite(cycle_ids)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
    return maximum


def fit_processed_cycle_scale(train_paths: Iterable[str | Path], max_cycles: int | None) -> float:
    def cycle_arrays():
        for path in train_paths:
            with np.load(Path(path), allow_pickle=True) as data:
                if "cycle_id" not in data:
                    raise ValueError(f"{path} is missing required array: cycle_id")
                yield np.array(data["cycle_id"], copy=True)

    return fit_cycle_scale(cycle_arrays(), max_cycles)


if __name__ == "__main__":
    raise SystemExit(main())
