from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


BATTERY_ABLATIONS = {
    "full": [],
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


def parse_args():
    p = argparse.ArgumentParser(description="Run GraphReportTS ablation suite")
    p.add_argument("--variant", choices=["battery", "general"], default="battery")
    p.add_argument("--dataset", type=str, default="mit")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_root", type=str, default="runs/graph_report_ablation")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--no_hf_text", action="store_true")
    p.add_argument("--allow_summary_fallback", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    suite = BATTERY_ABLATIONS if args.variant == "battery" else GENERAL_ABLATIONS
    rows = []
    for name, flags in suite.items():
        out_dir = Path(args.out_root) / args.variant / args.dataset / name
        cmd = [
            "python",
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
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--pred_len",
            str(args.pred_len),
            "--input_len",
            str(args.input_len),
            "--device",
            args.device,
        ]
        if args.no_hf_text:
            cmd.append("--no_hf_text")
        if args.allow_summary_fallback:
            cmd.append("--allow_summary_fallback")
        cmd.extend(flags)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        metrics_path = out_dir / args.variant / args.dataset / "test_metrics.json"
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
