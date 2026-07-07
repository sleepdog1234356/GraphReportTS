from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style
except ImportError:
    from paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style


def parse_args():
    p = argparse.ArgumentParser(description="Plot experiment/ablation metric tables")
    p.add_argument("--table", type=str, required=True, help="CSV table with columns such as ablation/model, mse, mae, rmse")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--name_col", type=str, default="ablation")
    p.add_argument("--metrics", type=str, default="mse,mae,rmse")
    return p.parse_args()


def main():
    args = parse_args()
    table = Path(args.table)
    out_dir = Path(args.out_dir) if args.out_dir else table.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(table)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip() in df.columns]
    if args.name_col not in df.columns:
        raise ValueError(f"{args.name_col} not found in {table}")
    set_aaai_style()
    for metric in metrics:
        g = df[[args.name_col, metric]].dropna().sort_values(metric)
        plt.figure(figsize=(max(3.5, 0.45 * len(g)), 2.5))
        plt.bar(g[args.name_col].astype(str), g[metric], color=AAAI_COLORS["blue"], width=0.72)
        plt.ylabel(metric.upper())
        plt.xticks(rotation=35, ha="right")
        plt.title(f"{metric.upper()} Comparison")
        plt.tight_layout()
        save_paper_figure(out_dir / f"{table.stem}_{metric}")
        plt.close()
    print(f"saved figures to {out_dir}")


if __name__ == "__main__":
    main()
