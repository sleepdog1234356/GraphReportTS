from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from .data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from .graph_report_losses import regression_metrics
    from .graph_report_model import GraphReportTS, GraphReportTSConfig
    from .paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style
    from .utils import ensure_dir, seed_everything, to_device
except ImportError:
    from data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from graph_report_losses import regression_metrics
    from graph_report_model import GraphReportTS, GraphReportTSConfig
    from paper_style import AAAI_COLORS, save_paper_figure, set_aaai_style
    from utils import ensure_dir, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Infer and visualize GraphReportTS")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--variant", choices=["battery", "general"], default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loader(args, ckpt):
    train_args = ckpt["args"]
    variant = args.variant or train_args["variant"]
    dataset = args.dataset or train_args["dataset"]
    data_root = args.data_root or train_args["data_root"]
    if variant == "battery":
        ds = BatteryRawGraphDataset(
            dataset_name=dataset,
            data_root=data_root,
            split=args.split,
            max_horizon=train_args["pred_len"],
            resample_len=train_args["resample_len"],
            delay_dim=train_args["delay_dim"],
            delay_lag=train_args["delay_lag"],
            include_derivatives=not train_args.get("no_derivative_map", False),
            include_hankel=not train_args.get("no_hankel_map", False),
            include_ic_dv=not train_args.get("no_ic_dv", False),
            allow_summary_fallback=train_args.get("allow_summary_fallback", False),
            seed=train_args.get("seed", args.seed),
            max_cycles=train_args.get("max_cycles", None),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_report_batch), variant, dataset
    scaler = StandardScalerNP()
    train_ds = GeneralForecastGraphDataset(
        dataset_name=dataset,
        data_root=data_root,
        split="train",
        input_len=train_args["input_len"],
        pred_len=train_args["pred_len"],
        resample_len=train_args["resample_len"],
        delay_dim=train_args["delay_dim"],
        delay_lag=train_args["delay_lag"],
        include_derivatives=not train_args.get("no_derivative_map", False),
        include_hankel=not train_args.get("no_hankel_map", False),
        scaler=scaler,
        fit_scaler=True,
    )
    _ = train_ds
    ds = GeneralForecastGraphDataset(
        dataset_name=dataset,
        data_root=data_root,
        split=args.split,
        input_len=train_args["input_len"],
        pred_len=train_args["pred_len"],
        resample_len=train_args["resample_len"],
        delay_dim=train_args["delay_dim"],
        delay_lag=train_args["delay_lag"],
        include_derivatives=not train_args.get("no_derivative_map", False),
        scaler=scaler,
        fit_scaler=False,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_general_graph_batch), variant, dataset


@torch.no_grad()
def collect_predictions(model, loader, device, variant: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    mse_sum = mae_sum = count = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["maps"], batch["prompt"], batch["horizon"])
        metrics = regression_metrics(out["pred"], batch["y"], batch["mask"])
        elems = float(batch["mask"].sum().detach().cpu())
        mse_sum += metrics["mse"] * elems
        mae_sum += metrics["mae"] * elems
        count += elems
        pred = out["pred"].detach().cpu()
        y = batch["y"].detach().cpu()
        mask = batch["mask"].detach().cpu()
        for i in range(pred.size(0)):
            steps = int(mask[i].reshape(mask[i].shape[0], -1).any(dim=-1).sum())
            for step in range(steps):
                if variant == "battery":
                    true_val = float(y[i, step])
                    pred_val = float(pred[i, step, 0])
                    rows.append(
                        {
                            "series_id": batch["cell_id"][i],
                            "input_index": int(batch["cycle"][i].detach().cpu()),
                            "forecast_step": step,
                            "target_index": int(batch["target_steps"][i, step].detach().cpu()),
                            "target_dim": "SOH",
                            "y_true": true_val,
                            "y_pred": pred_val,
                            "abs_err": abs(pred_val - true_val),
                        }
                    )
                else:
                    for dim in range(pred.size(-1)):
                        true_val = float(y[i, step, dim])
                        pred_val = float(pred[i, step, dim])
                        rows.append(
                            {
                                "series_id": batch["series_id"][i],
                                "input_index": int(batch["start_index"][i].detach().cpu()),
                                "forecast_step": step + 1,
                                "target_index": int(batch["target_steps"][i, step].detach().cpu()),
                                "target_dim": dim,
                                "y_true": true_val,
                                "y_pred": pred_val,
                                "abs_err": abs(pred_val - true_val),
                            }
                        )
    mse = mse_sum / max(count, 1.0)
    return pd.DataFrame(rows), {"mse": mse, "mae": mae_sum / max(count, 1.0), "rmse": mse**0.5}


def save_figures(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if df.empty:
        return
    set_aaai_style()
    plt.figure(figsize=(3.25, 3.0))
    plt.scatter(df["y_true"], df["y_pred"], s=7, alpha=0.55, color=AAAI_COLORS["blue"], edgecolors="none")
    lo = min(float(df["y_true"].min()), float(df["y_pred"].min()))
    hi = max(float(df["y_true"].max()), float(df["y_pred"].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color=AAAI_COLORS["gray"])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Prediction Scatter")
    plt.tight_layout()
    save_paper_figure(out_dir / f"{prefix}_scatter")
    plt.close()

    step_mae = df.groupby("forecast_step")["abs_err"].mean().reset_index()
    plt.figure(figsize=(3.5, 2.3))
    plt.plot(step_mae["forecast_step"], step_mae["abs_err"], marker="o", color=AAAI_COLORS["red"])
    plt.xlabel("Forecast step")
    plt.ylabel("MAE")
    plt.title("Error by Forecast Step")
    plt.tight_layout()
    save_paper_figure(out_dir / f"{prefix}_step_mae")
    plt.close()

    for (sid, idx), g in list(df.groupby(["series_id", "input_index"]))[:6]:
        gd = g[g["target_dim"].astype(str).isin(["SOH", "0"])].sort_values("forecast_step")
        if gd.empty:
            continue
        plt.figure(figsize=(3.6, 2.4))
        plt.plot(gd["target_index"], gd["y_true"], marker="x", label="True", color=AAAI_COLORS["gray"])
        plt.plot(gd["target_index"], gd["y_pred"], marker="o", label="Pred", color=AAAI_COLORS["orange"])
        plt.xlabel("Target index")
        plt.ylabel("Value")
        plt.title(f"{sid}, input {idx}")
        plt.legend()
        plt.tight_layout()
        safe = str(sid).replace("/", "_")
        save_paper_figure(out_dir / f"{prefix}_curve_{safe}_{idx}")
        plt.close()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = GraphReportTSConfig(**ckpt["model_cfg"])
    model = GraphReportTS(cfg).to(device)
    with torch.no_grad():
        loader, variant, dataset = build_loader(args, ckpt)
        init_batch = next(iter(loader))
        init_batch = to_device(init_batch, device)
        _ = model(init_batch["maps"], init_batch["prompt"], init_batch["horizon"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    out_dir = ensure_dir(args.out_dir or Path(args.checkpoint).with_suffix("").parent / "figures")
    df, metrics = collect_predictions(model, loader, device, variant)
    prefix = f"{variant}_{dataset}_{args.split}"
    df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    save_figures(df, out_dir, prefix)
    print(f"{prefix} metrics: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")
    print(f"saved predictions and figures to {out_dir}")


if __name__ == "__main__":
    main()
