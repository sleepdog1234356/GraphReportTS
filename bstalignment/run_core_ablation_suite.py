from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .battery_protocol import (
    FORMAL_CACHE_TASK_BATCH_SIZE,
    require_formal_battery_protocol,
    require_formal_cache_task_batch_size,
)
from .data_battery_raw import (
    battery_graph_cache_config,
    battery_graph_cache_path,
    battery_sequence_cache_config,
    battery_sequence_cache_path,
)
from .training_strategy import TRAINING_STRATEGY_VERSION


CORE_ABLATION_SUITE_VERSION = "core-v1"
CORE_BATTERY_ABLATIONS: Mapping[str, tuple[str, ...]] = {
    "no_hankel_graph": ("--battery_input_mode", "raw_sequence"),
    "no_report_prompt": ("--no_report_prompt",),
    "no_ic_dv": ("--no_ic_dv",),
    "no_text_gate": ("--no_text_gate",),
}

_DATASETS = ("mit", "calce", "xjtu")
_SPLITS = ("train", "val", "test")
_CONTROLLED_NO_FLAGS = (
    "no_ic_dv",
    "no_hankel_map",
    "no_derivative_map",
    "no_report_prompt",
    "no_cross_modal",
    "no_text_gate",
    "no_semantic_alignment",
    "no_align_loss",
)
_CORE_ONLY_SWITCHES: Mapping[str, bool] = {
    "no_dynamic_graph": False,
    "no_domain_edges": False,
    "separate_heads": False,
    "no_numeric_history": False,
    "no_multi_cycle_raw": False,
    "single_cycle_raw": False,
    "absolute_step_decoder": False,
}
_FULL_ARGUMENTS: Mapping[str, Any] = {
    "variant": "battery",
    "history_len": 32,
    "pred_len": 20,
    "batch_size": 64,
    "seed": 42,
    **{name: False for name in _CONTROLLED_NO_FLAGS},
}
_FULL_MODEL_CONFIG: Mapping[str, Any] = {
    "variant": "battery",
    "freeze_text": True,
    "use_hf_text_encoder": True,
    "use_report_prompt": True,
    "use_cross_modal_fusion": True,
    "use_dynamic_graph": True,
    "use_domain_edges": True,
    "unified_decoder": True,
    "battery_history_len": 32,
    "history_feature_dim": 8,
    "use_multi_cycle_raw": True,
    "single_cycle_raw": False,
    "use_numeric_history": True,
    "use_text_gate": True,
    "use_semantic_alignment": True,
    "use_relative_steps": True,
}
_SUMMARY_FIELDS = (
    "ablation",
    "dataset",
    "mse",
    "mae",
    "rmse",
    "best_epoch",
    "stopped_epoch",
    "mean_epoch_seconds",
    "total_train_seconds",
    "trainable_parameter_count",
    "training_strategy_version",
    "ablation_suite_version",
    "result_source",
    "source_git_commit",
    "full_reference_git_commit",
)
_TIMING_FIELDS = (
    "best_epoch",
    "stopped_epoch",
    "mean_epoch_seconds",
    "total_train_seconds",
    "trainable_parameter_count",
)


def _mismatch(dataset: str, expected: Any, observed: Any, path: Path) -> RuntimeError:
    return RuntimeError(
        f"Core ablation reference mismatch dataset={dataset} expected={expected!r} "
        f"observed={observed!r} path={path}"
    )


def _read_json_object(path: Path, dataset: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _mismatch(dataset, "valid JSON object", f"invalid JSON: {exc}", path) from exc
    if not isinstance(value, dict):
        raise _mismatch(dataset, "JSON object", type(value).__name__, path)
    return value


def _require_artifact(path: Path, dataset: str) -> None:
    if not path.is_file():
        raise _mismatch(dataset, "existing artifact", "missing", path)


def _require_exact_field(
    container: Any,
    name: str,
    expected: Any,
    *,
    dataset: str,
    path: Path,
) -> None:
    if not isinstance(container, dict):
        raise _mismatch(dataset, f"object containing {name}", type(container).__name__, path)
    observed = container.get(name, "<missing>")
    if type(observed) is not type(expected) or observed != expected:
        raise _mismatch(dataset, {name: expected}, {name: observed}, path)


def _metrics_row(metrics_path: Path, dataset: str) -> dict[str, Any]:
    metrics = _read_json_object(metrics_path, dataset)
    for name in ("mse", "mae", "rmse"):
        value = metrics.get(name, "<missing>")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise _mismatch(dataset, f"finite numeric metric {name}", value, metrics_path)
    return metrics


def _optional_full_timing(result_dir: Path, dataset: str) -> dict[str, Any]:
    path = result_dir / "run_summary.json"
    if not path.exists():
        return {name: "" for name in _TIMING_FIELDS}
    summary = _read_json_object(path, dataset)
    return {name: summary.get(name, "") for name in _TIMING_FIELDS}


def require_reusable_full_reference(
    result_dir: Path,
    dataset: str,
    training_strategy_version: str,
) -> dict[str, Any]:
    result_dir = Path(result_dir)
    checkpoint_path = result_dir / "best.pt"
    metrics_path = result_dir / "test_metrics.json"
    config_path = result_dir / "run_config.json"
    for path in (checkpoint_path, metrics_path, config_path):
        _require_artifact(path, dataset)

    config = _read_json_object(config_path, dataset)
    _require_exact_field(
        config,
        "training_strategy_version",
        training_strategy_version,
        dataset=dataset,
        path=config_path,
    )
    args = config.get("args")
    _require_exact_field(args, "dataset", dataset, dataset=dataset, path=config_path)
    for name, expected in _FULL_ARGUMENTS.items():
        _require_exact_field(args, name, expected, dataset=dataset, path=config_path)
    input_mode = args.get("battery_input_mode", "hankel_graph") if isinstance(args, dict) else "<missing>"
    if input_mode != "hankel_graph":
        raise _mismatch(dataset, {"battery_input_mode": "hankel_graph"}, {"battery_input_mode": input_mode}, config_path)

    model_cfg = config.get("model_cfg")
    for name, expected in _FULL_MODEL_CONFIG.items():
        _require_exact_field(model_cfg, name, expected, dataset=dataset, path=config_path)
    model_input_mode = (
        model_cfg.get("battery_input_mode", "hankel_graph")
        if isinstance(model_cfg, dict)
        else "<missing>"
    )
    if model_input_mode != "hankel_graph":
        raise _mismatch(
            dataset,
            {"model_cfg.battery_input_mode": "hankel_graph"},
            {"model_cfg.battery_input_mode": model_input_mode},
            config_path,
        )

    return {
        **_metrics_row(metrics_path, dataset),
        **_optional_full_timing(result_dir, dataset),
        "ablation": "full",
        "dataset": dataset,
        "training_strategy_version": training_strategy_version,
        "ablation_suite_version": CORE_ABLATION_SUITE_VERSION,
        "result_source": "reused_main",
    }


def _expected_core_arguments(dataset: str, ablation: str) -> dict[str, Any]:
    expected = dict(_FULL_ARGUMENTS)
    expected.update(_CORE_ONLY_SWITCHES)
    expected.update(
        {
            "dataset": dataset,
            "protocol_stage": "ablation",
            "ablation_suite_version": CORE_ABLATION_SUITE_VERSION,
            "battery_input_mode": "raw_sequence" if ablation == "no_hankel_graph" else "hankel_graph",
        }
    )
    if ablation in {"no_report_prompt", "no_ic_dv", "no_text_gate"}:
        expected[ablation] = True
    return expected


def _expected_core_model_config(ablation: str) -> dict[str, Any]:
    expected = dict(_FULL_MODEL_CONFIG)
    expected["battery_input_mode"] = "raw_sequence" if ablation == "no_hankel_graph" else "hankel_graph"
    if ablation == "no_report_prompt":
        expected["use_report_prompt"] = False
    elif ablation == "no_text_gate":
        expected["use_text_gate"] = False
    return expected


def core_run_config_matches(
    result_dir: Path,
    dataset: str,
    ablation: str,
    training_strategy_version: str,
) -> bool:
    if ablation not in CORE_BATTERY_ABLATIONS:
        raise ValueError(f"Unknown core battery ablation: {ablation}")
    try:
        config = json.loads((Path(result_dir) / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    roots = {
        "training_strategy_version": training_strategy_version,
        "protocol_stage": "ablation",
        "ablation_suite_version": CORE_ABLATION_SUITE_VERSION,
    }
    if any(type(config.get(name)) is not type(expected) or config.get(name) != expected for name, expected in roots.items()):
        return False
    args = config.get("args")
    model_cfg = config.get("model_cfg")
    if not isinstance(args, dict) or not isinstance(model_cfg, dict):
        return False
    return all(
        type(args.get(name)) is type(expected) and args.get(name) == expected
        for name, expected in _expected_core_arguments(dataset, ablation).items()
    ) and all(
        type(model_cfg.get(name)) is type(expected) and model_cfg.get(name) == expected
        for name, expected in _expected_core_model_config(ablation).items()
    )


def verify_prompt_cache_identity(reference_cache: Path, candidate_cache: Path) -> None:
    reference = [
        json.loads(line)
        for line in (Path(reference_cache) / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidate = [
        json.loads(line)
        for line in (Path(candidate_cache) / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"Prompt sample count mismatch: expected={len(reference)} observed={len(candidate)} "
            f"path={candidate_cache}"
        )
    for index, (expected, observed) in enumerate(zip(reference, candidate)):
        identity = ("cell_id", "cycle", "prompt")
        if any(expected[key] != observed[key] for key in identity):
            raise RuntimeError(
                f"Prompt mismatch dataset sample={index} expected={expected} observed={observed} "
                f"path={candidate_cache}"
            )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal core-v1 battery ablation matrix")
    parser.add_argument("--datasets", nargs="+", choices=_DATASETS, default=list(_DATASETS))
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(CORE_BATTERY_ABLATIONS),
        default=list(CORE_BATTERY_ABLATIONS),
    )
    parser.add_argument("--data_root", default="bstalignment/data")
    parser.add_argument("--full_result_root", required=True)
    parser.add_argument("--graph_cache_dir", required=True)
    parser.add_argument("--sequence_cache_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--text_model", required=True)
    parser.add_argument("--full_reference_commit", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_task_batch_size", type=int, default=FORMAL_CACHE_TASK_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> None:
    require_formal_battery_protocol(
        observed_cycles=32,
        prediction_cycles=20,
        batch_size=args.batch_size,
        stage="ablation",
        context="core-v1 battery ablation suite",
    )
    require_formal_cache_task_batch_size(
        batch_size=args.cache_task_batch_size,
        context="core-v1 battery ablation suite",
    )
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("core-v1 datasets must not contain duplicates")
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError("core-v1 ablations must not contain duplicates")
    if not args.full_reference_commit.strip():
        raise ValueError("full_reference_commit must be non-empty")


def _source_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty source commit")
    return commit


def _print_command(command: Sequence[str]) -> None:
    print(" ".join(str(token) for token in command))


def _cache_command(args: argparse.Namespace, dataset: str, kind: str) -> list[str]:
    module = (
        "bstalignment.precompute_battery_sequence_cache"
        if kind == "sequence"
        else "bstalignment.precompute_battery_graph_cache"
    )
    cache_root = args.sequence_cache_dir if kind == "sequence" else args.graph_cache_dir
    command = [
        sys.executable,
        "-m",
        module,
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--cache_dir",
        cache_root,
        "--splits",
        *_SPLITS,
        "--pred_len",
        "20",
        "--history_len",
        "32",
        "--batch_size",
        str(args.cache_task_batch_size),
        "--num_workers",
        str(args.num_workers),
    ]
    if kind == "no_ic_dv":
        command.append("--no_ic_dv")
    return command


def _train_command(
    args: argparse.Namespace,
    dataset: str,
    ablation: str,
    result_dir: Path,
    *,
    start_fresh: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bstalignment.train_graph_report",
        "--variant",
        "battery",
        "--dataset",
        dataset,
        "--data_root",
        args.data_root,
        "--run_dir",
        str(result_dir),
        "--batch_size",
        str(args.batch_size),
        "--history_len",
        "32",
        "--pred_len",
        "20",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--text_model",
        args.text_model,
        "--protocol_stage",
        "ablation",
        "--ablation_suite_version",
        CORE_ABLATION_SUITE_VERSION,
    ]
    if ablation == "no_hankel_graph":
        command.extend(
            [
                "--precomputed_sequence_cache_dir",
                args.sequence_cache_dir,
                "--require_precomputed_sequence_cache",
            ]
        )
    else:
        command.extend(
            [
                "--precomputed_cache_dir",
                args.graph_cache_dir,
                "--require_precomputed_cache",
            ]
        )
    if start_fresh:
        command.append("--no_resume")
    command.extend(CORE_BATTERY_ABLATIONS[ablation])
    return command


def _graph_cache_path(args: argparse.Namespace, dataset: str, split: str, include_ic_dv: bool) -> Path:
    config = battery_graph_cache_config(
        dataset_name=dataset,
        split=split,
        max_horizon=20,
        resample_len=128,
        delay_dim=8,
        delay_lag=1,
        include_derivatives=True,
        include_hankel=True,
        include_ic_dv=include_ic_dv,
        allow_summary_fallback=False,
        seed=42,
        max_cycles=None,
        history_len=32,
    )
    return battery_graph_cache_path(args.graph_cache_dir, config)


def _sequence_cache_path(args: argparse.Namespace, dataset: str, split: str) -> Path:
    config = battery_sequence_cache_config(
        dataset_name=dataset,
        split=split,
        max_horizon=20,
        resample_len=128,
        allow_summary_fallback=False,
        seed=42,
        max_cycles=None,
        history_len=32,
    )
    return battery_sequence_cache_path(args.sequence_cache_dir, config)


def _precompute_and_validate_caches(args: argparse.Namespace, dataset: str) -> None:
    kinds = ["graph"]
    if "no_hankel_graph" in args.ablations:
        kinds.append("sequence")
    if "no_ic_dv" in args.ablations:
        kinds.append("no_ic_dv")
    for kind in kinds:
        command = _cache_command(args, dataset, kind)
        _print_command(command)
        subprocess.run(command, check=True)
    for split in _SPLITS:
        reference = _graph_cache_path(args, dataset, split, include_ic_dv=True)
        if "no_hankel_graph" in args.ablations:
            verify_prompt_cache_identity(reference, _sequence_cache_path(args, dataset, split))
        if "no_ic_dv" in args.ablations:
            verify_prompt_cache_identity(
                reference,
                _graph_cache_path(args, dataset, split, include_ic_dv=False),
            )


def _trained_row(result_dir: Path, dataset: str, ablation: str) -> dict[str, Any]:
    metrics_path = result_dir / "test_metrics.json"
    summary_path = result_dir / "run_summary.json"
    _require_artifact(metrics_path, dataset)
    _require_artifact(summary_path, dataset)
    summary = _read_json_object(summary_path, dataset)
    for field in _TIMING_FIELDS:
        if field not in summary:
            raise _mismatch(dataset, f"run_summary field {field}", "<missing>", summary_path)
    return {
        **_metrics_row(metrics_path, dataset),
        **{field: summary[field] for field in _TIMING_FIELDS},
        "ablation": ablation,
        "dataset": dataset,
        "training_strategy_version": TRAINING_STRATEGY_VERSION,
        "ablation_suite_version": CORE_ABLATION_SUITE_VERSION,
        "result_source": "trained_ablation",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({name for row in rows for name in row}.difference(_SUMMARY_FIELDS))
    fields = [*_SUMMARY_FIELDS, *extra_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_dataset(
    args: argparse.Namespace,
    dataset: str,
    source_git_commit: str,
) -> list[dict[str, Any]]:
    full_dir = Path(args.full_result_root) / dataset
    full_row = require_reusable_full_reference(full_dir, dataset, TRAINING_STRATEGY_VERSION)
    full_row.update(
        {
            "source_git_commit": args.full_reference_commit,
            "full_reference_git_commit": args.full_reference_commit,
        }
    )
    _precompute_and_validate_caches(args, dataset)
    rows = [full_row]
    for ablation in args.ablations:
        result_dir = Path(args.out_root) / "battery" / dataset / ablation
        exists = result_dir.exists()
        matches = core_run_config_matches(
            result_dir,
            dataset,
            ablation,
            TRAINING_STRATEGY_VERSION,
        )
        if exists and not matches:
            raise _mismatch(
                dataset,
                f"matching {CORE_ABLATION_SUITE_VERSION} {ablation} run metadata",
                "missing, malformed, or mismatched metadata",
                result_dir / "run_config.json",
            )
        if args.force_retrain and exists:
            shutil.rmtree(result_dir)
            exists = False
            matches = False
        complete = matches and (result_dir / "test_metrics.json").is_file()
        if complete:
            print(f"skip completed core ablation {dataset} {ablation}")
        else:
            command = _train_command(
                args,
                dataset,
                ablation,
                result_dir,
                start_fresh=not exists,
            )
            _print_command(command)
            subprocess.run(command, check=True)
            if not core_run_config_matches(
                result_dir,
                dataset,
                ablation,
                TRAINING_STRATEGY_VERSION,
            ):
                raise _mismatch(
                    dataset,
                    f"matching {CORE_ABLATION_SUITE_VERSION} {ablation} run metadata after training",
                    "missing, malformed, or mismatched metadata",
                    result_dir / "run_config.json",
                )
        row = _trained_row(result_dir, dataset, ablation)
        row.update(
            {
                "source_git_commit": source_git_commit,
                "full_reference_git_commit": args.full_reference_commit,
            }
        )
        rows.append(row)
    summary_path = Path(args.out_root) / "battery" / dataset / "core_ablation_summary.csv"
    _write_csv(summary_path, rows)
    print(f"saved core ablation summary to {summary_path}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_cli(args)
    source_git_commit = _source_git_commit()
    if args.dry_run:
        for dataset in args.datasets:
            for ablation in args.ablations:
                result_dir = Path(args.out_root) / "battery" / dataset / ablation
                _print_command(
                    _train_command(args, dataset, ablation, result_dir, start_fresh=True)
                )
        return

    combined_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        combined_rows.extend(_run_dataset(args, dataset, source_git_commit))
    combined_path = Path(args.out_root) / "battery" / "core_ablation_summary.csv"
    _write_csv(combined_path, combined_rows)
    print(f"saved combined core ablation summary to {combined_path}")


if __name__ == "__main__":
    main()
