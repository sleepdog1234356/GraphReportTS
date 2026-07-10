from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .battery_protocol import (
    FORMAL_CACHE_TASK_BATCH_SIZE,
    require_formal_battery_protocol,
    require_formal_cache_task_batch_size,
    run_config_matches,
)
from .training_strategy import TRAINING_STRATEGY_VERSION


BATTERY_ABLATIONS = {
    "full": [],
    "no_numeric_history": ["--no_numeric_history"],
    "no_multi_cycle_raw": ["--no_multi_cycle_raw"],
    "single_cycle_raw": ["--single_cycle_raw"],
    "no_text_gate": ["--no_text_gate"],
    "no_semantic_alignment": ["--no_semantic_alignment"],
    "no_align_loss": ["--no_align_loss"],
    "absolute_step_decoder": ["--absolute_step_decoder"],
    "no_ic_dv": ["--no_ic_dv"],
    "no_hankel_map": ["--no_hankel_map"],
    "no_derivative_map": ["--no_derivative_map"],
    "static_graph": ["--no_dynamic_graph"],
    "no_domain_edges": ["--no_domain_edges"],
    "no_report_prompt": ["--no_report_prompt"],
    "no_cross_modal": ["--no_cross_modal"],
    "separate_heads": ["--separate_heads"],
}

GENERAL_ABLATIONS = {
    "full": [],
    "no_hankel_map": ["--no_hankel_map"],
    "no_derivative_map": ["--no_derivative_map"],
    "static_graph": ["--no_dynamic_graph"],
    "no_report_prompt": ["--no_report_prompt"],
    "no_cross_modal": ["--no_cross_modal"],
    "separate_heads": ["--separate_heads"],
}


def has_matching_strategy_version(
    result_dir: Path,
    training_strategy_version: str,
    protocol_stage: str | None = "ablation",
) -> bool:
    return run_config_matches(
        result_dir / "run_config.json",
        training_strategy_version=training_strategy_version,
        stage=protocol_stage,
    )


def should_skip_ablation(
    result_dir: Path,
    training_strategy_version: str,
    force_retrain: bool,
    protocol_stage: str | None = "ablation",
) -> bool:
    return (
        not force_retrain
        and (result_dir / "test_metrics.json").is_file()
        and has_matching_strategy_version(result_dir, training_strategy_version, protocol_stage)
    )


def remove_ablation_output_if_forced(output_dir: Path, force_retrain: bool) -> None:
    remove_ablation_output_if_fresh(output_dir, force_retrain)


def remove_ablation_output_if_fresh(output_dir: Path, start_fresh: bool) -> None:
    if start_fresh and output_dir.exists():
        shutil.rmtree(output_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Run GraphReportTS ablation suite")
    p.add_argument("--variant", choices=["battery", "general"], default="battery")
    p.add_argument("--dataset", type=str, default="mit")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_root", type=str, default="runs/graph_report_ablation")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--cache_task_batch_size", type=int, default=FORMAL_CACHE_TASK_BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--history_len", type=int, default=32)
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--temporal_layers", type=int, default=1)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--no_hf_text", action="store_true")
    p.add_argument("--allow_summary_fallback", action="store_true")
    p.add_argument("--precomputed_cache_dir", type=str, default=None)
    p.add_argument("--require_precomputed_cache", action="store_true")
    p.add_argument("--force_precompute_cache", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--training_strategy_version", type=str, default=TRAINING_STRATEGY_VERSION)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.variant == "battery":
        require_formal_battery_protocol(
            observed_cycles=args.history_len,
            prediction_cycles=args.pred_len,
            batch_size=args.batch_size,
            stage="ablation",
            context="GraphReportTS battery ablation suite",
        )
        require_formal_cache_task_batch_size(
            batch_size=args.cache_task_batch_size,
            context="GraphReportTS battery ablation suite",
        )
    suite = BATTERY_ABLATIONS if args.variant == "battery" else GENERAL_ABLATIONS
    protocol_stage = "ablation" if args.variant == "battery" else None
    rows = []
    for name, flags in suite.items():
        out_dir = Path(args.out_root) / args.variant / args.dataset / name
        result_dir = out_dir / args.variant / args.dataset
        metrics_path = result_dir / "test_metrics.json"
        if should_skip_ablation(
            result_dir,
            args.training_strategy_version,
            args.force_retrain,
            protocol_stage,
        ):
            print(f"skip completed ablation {args.dataset} {name}")
            rows.append(pd.read_json(metrics_path, typ="series").to_dict() | {"ablation": name})
            continue
        start_fresh = args.force_retrain or not has_matching_strategy_version(
            result_dir,
            args.training_strategy_version,
            protocol_stage,
        )
        if not args.dry_run:
            remove_ablation_output_if_fresh(out_dir, start_fresh)
        if args.variant == "battery" and args.precomputed_cache_dir:
            precompute_cmd = [
                sys.executable,
                "-m",
                "bstalignment.precompute_battery_graph_cache",
                "--dataset",
                args.dataset,
                "--data_root",
                args.data_root,
                "--cache_dir",
                args.precomputed_cache_dir,
                "--pred_len",
                str(args.pred_len),
                "--history_len",
                str(args.history_len),
                "--batch_size",
                str(args.cache_task_batch_size),
                "--num_workers",
                str(args.num_workers),
                "--splits",
                "train",
                "val",
                "test",
            ]
            if args.allow_summary_fallback:
                precompute_cmd.append("--allow_summary_fallback")
            for flag in flags:
                if flag in {"--no_ic_dv", "--no_hankel_map", "--no_derivative_map"}:
                    precompute_cmd.append(flag)
            if args.force_precompute_cache:
                precompute_cmd.append("--force")
            print(" ".join(precompute_cmd))
            if not args.dry_run:
                subprocess.run(precompute_cmd, check=True)
        cmd = [
            sys.executable,
            "-m",
            "bstalignment.train_graph_report",
            "--variant",
            args.variant,
            "--dataset",
            args.dataset,
            "--data_root",
            args.data_root,
            "--out_dir",
            str(out_dir),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--pred_len",
            str(args.pred_len),
            "--history_len",
            str(args.history_len),
            "--input_len",
            str(args.input_len),
            "--temporal_layers",
            str(args.temporal_layers),
            "--temporal_heads",
            str(args.temporal_heads),
            "--device",
            args.device,
            "--text_model",
            args.text_model,
        ]
        if args.no_hf_text:
            cmd.append("--no_hf_text")
        if args.allow_summary_fallback:
            cmd.append("--allow_summary_fallback")
        if args.precomputed_cache_dir:
            cmd.extend(["--precomputed_cache_dir", args.precomputed_cache_dir])
        if args.require_precomputed_cache:
            cmd.append("--require_precomputed_cache")
        if start_fresh:
            cmd.append("--no_resume")
        cmd.extend(flags)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        if metrics_path.exists():
            rows.append(pd.read_json(metrics_path, typ="series").to_dict() | {"ablation": name})
    if rows:
        summary = pd.DataFrame(rows)
        summary_path = Path(args.out_root) / args.variant / args.dataset / "ablation_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
