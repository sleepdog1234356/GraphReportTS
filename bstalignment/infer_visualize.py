from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import torch

try:
    from .data_mit import (
        MITBatteryCycleDataset,
        StandardScalerTorch,
        add_cycle_features,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from .models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from .paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style
    from .utils import ensure_dir, seed_everything, to_device
except ImportError:
    from data_mit import (
        MITBatteryCycleDataset,
        StandardScalerTorch,
        add_cycle_features,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style
    from utils import ensure_dir, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Infer one-cycle current/future SOH with BatteryCycleLLMAssist")
    p.add_argument("--checkpoint", type=str, default="runs/mit_bstalign/best.pt")
    p.add_argument("--data_dir", type=str, default="bstalignment/data/mit")
    p.add_argument("--out_dir", type=str, default="runs/mit_bstalign/figures")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--cell_id", type=str, default=None)
    p.add_argument("--cycle", type=int, default=None)
    p.add_argument("--forecast_horizon", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def collect_split_predictions(model, ds, device, batch_size: int):
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cycle_batch)
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["x"], batch["prompt"], batch["horizon"])
        pred_now = out["soh_now"].detach().cpu()
        pred_future = out["soh_future"].detach().cpu()
        for i in range(pred_future.size(0)):
            rows.append(
                {
                    "cell_id": batch["cell_id"][i],
                    "cycle": int(batch["cycle"][i].cpu()),
                    "output_type": "now",
                    "horizon_step": 0,
                    "target_cycle": int(batch["cycle"][i].cpu()),
                    "soh_true": float(batch["y_now"][i].detach().cpu()),
                    "soh_pred": float(pred_now[i]),
                    "abs_err": float(abs(pred_now[i] - batch["y_now"][i].detach().cpu())),
                }
            )
            h = int(batch["horizon"][i].cpu())
            for step in range(h):
                rows.append(
                    {
                        "cell_id": batch["cell_id"][i],
                        "cycle": int(batch["cycle"][i].cpu()),
                        "output_type": "future",
                        "horizon_step": step + 1,
                        "target_cycle": int(batch["future_cycles"][i, step].cpu()),
                        "soh_true": float(batch["y_future"][i, step].detach().cpu()),
                        "soh_pred": float(pred_future[i, step]),
                        "abs_err": float(abs(pred_future[i, step] - batch["y_future"][i, step].detach().cpu())),
                    }
                )
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for output_type, prefix in [("now", "soh_now"), ("future", "soh_future")]:
        g = df[df["output_type"] == output_type]
        if len(g) == 0:
            metrics[f"{prefix}_mae"] = float("nan")
            metrics[f"{prefix}_rmse"] = float("nan")
            continue
        err = g["soh_pred"].to_numpy(dtype=float) - g["soh_true"].to_numpy(dtype=float)
        metrics[f"{prefix}_mae"] = float(np.mean(np.abs(err)))
        metrics[f"{prefix}_rmse"] = float(np.sqrt(np.mean(err**2)))
    return metrics


def print_metrics(metrics: dict[str, float]):
    print(
        "SOH current estimate: "
        f"MAE={metrics['soh_now_mae']:.6f}, RMSE={metrics['soh_now_rmse']:.6f}"
    )
    print(
        "SOH future forecast: "
        f"MAE={metrics['soh_future_mae']:.6f}, RMSE={metrics['soh_future_rmse']:.6f}"
    )


def plot_prediction_curve(df: pd.DataFrame, out_path: Path, title: str | None = None):
    set_aaai_style()
    g = df.sort_values("horizon_step")
    plt.figure(figsize=(3.6, 2.4))
    plt.plot(g["target_cycle"], g["soh_pred"], marker="o", label="Predicted", color=AAAI_COLORS["orange"])
    if g["soh_true"].notna().any():
        plt.plot(g["target_cycle"], g["soh_true"], marker="x", label="True", color=AAAI_COLORS["gray"])
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title(title or f"Current and future SOH: {g.iloc[0]['cell_id']} cycle {int(g.iloc[0]['cycle'])}")
    plt.legend()
    plt.tight_layout()
    save_paper_figure(out_path)
    plt.close()


def plot_scatter(df: pd.DataFrame, output_type: str, out_path: Path):
    set_aaai_style()
    g = df[df["output_type"] == output_type]
    if len(g) == 0:
        return
    plt.figure(figsize=(3.25, 3.0))
    if output_type == "future":
        max_step = max(int(g["horizon_step"].max()), 1)
        scatter = plt.scatter(
            g["soh_true"],
            g["soh_pred"],
            c=g["horizon_step"],
            cmap="RdYlGn_r",
            norm=Normalize(vmin=1, vmax=max_step),
            s=7,
            alpha=0.65,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Forecast step")
    else:
        plt.scatter(g["soh_true"], g["soh_pred"], s=7, alpha=0.55, color=AAAI_COLORS["blue"], edgecolors="none")
    lo = min(float(g["soh_true"].min()), float(g["soh_pred"].min()))
    hi = max(float(g["soh_true"].max()), float(g["soh_pred"].max()))
    pad = max((hi - lo) * 0.05, 1e-3)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1, color=AAAI_COLORS["gray"])
    plt.xlabel("True SOH")
    plt.ylabel("Predicted SOH")
    title = "Current SOH estimate" if output_type == "now" else "Future SOH forecast"
    plt.title(title)
    plt.tight_layout()
    save_paper_figure(out_path)
    plt.close()


def plot_sample_curves(df: pd.DataFrame, out_dir: Path, split: str, max_samples: int = 8):
    keys = df[["cell_id", "cycle"]].drop_duplicates().head(max_samples)
    for _, key in keys.iterrows():
        g = df[(df["cell_id"] == key["cell_id"]) & (df["cycle"] == key["cycle"])]
        safe_cell = str(key["cell_id"]).replace("/", "_")
        out_path = out_dir / f"curve_{split}_{safe_cell}_cycle{int(key['cycle'])}.png"
        plot_prediction_curve(
            g,
            out_path,
            title=f"SOH current and future prediction: {key['cell_id']} cycle {int(key['cycle'])}",
        )


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = BatteryCycleLLMAssistConfig(**ckpt["cfg"])
    model = BatteryCycleLLMAssist(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    records = load_mit_battery_pkls(args.data_dir)
    train_recs, val_recs, test_recs = split_cells(records, seed=ckpt.get("args", {}).get("seed", args.seed))
    recs = {"train": train_recs, "val": val_recs, "test": test_recs}[args.split]
    scaler = StandardScalerTorch.from_state_dict(ckpt["scaler"])

    ds = MITBatteryCycleDataset(
        recs,
        max_horizon=cfg.max_horizon if args.forecast_horizon is None else min(args.forecast_horizon, cfg.max_horizon),
        min_history=ckpt.get("args", {}).get("min_history", 5),
        features=ckpt["features"],
        scaler=scaler,
        fit_scaler=False,
        random_horizon=False,
        max_cycles=ckpt.get("args", {}).get("max_cycles", None),
    )

    if args.cell_id is not None or args.cycle is not None:
        filtered = []
        for s in ds.samples:
            rec, df = ds.cells[s["cell_id"]]
            row = add_cycle_features(rec.summary).iloc[int(s["idx"])]
            same_cell = args.cell_id is None or s["cell_id"] == args.cell_id
            same_cycle = args.cycle is None or int(row["cycle"]) == args.cycle
            if same_cell and same_cycle:
                filtered.append(s)
        ds.samples = filtered[:1]
        if len(ds) == 0:
            raise RuntimeError("No matching cell/cycle sample found.")

    df = collect_split_predictions(model, ds, device, args.batch_size)
    metrics = compute_metrics(df)
    print_metrics(metrics)

    name = "single_prediction" if len(ds) == 1 else f"predictions_{args.split}"
    df.to_csv(out_dir / f"{name}.csv", index=False)
    if len(ds) == 1:
        plot_prediction_curve(df, out_dir / f"{name}.png")
    else:
        plot_scatter(df, "now", out_dir / f"soh_now_scatter_{args.split}.png")
        plot_scatter(df, "future", out_dir / f"soh_future_scatter_{args.split}.png")
        plot_sample_curves(df, out_dir, args.split)
    print(f"saved predictions and figures to {out_dir}")


if __name__ == "__main__":
    main()
