from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .data_mit import (
        CYCLE_FEATURES,
        MITBatteryCycleDataset,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from .losses import battery_cycle_forecast_loss
    from .models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from .utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device
except ImportError:
    from data_mit import (
        CYCLE_FEATURES,
        MITBatteryCycleDataset,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from losses import battery_cycle_forecast_loss
    from models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Train one-cycle LLM-assisted battery SOH forecaster")
    p.add_argument("--data_dir", type=str, default="bstalignment/data/mit")
    p.add_argument("--out_dir", type=str, default="runs/mit_bstalign")
    p.add_argument("--max_horizon", type=int, default=20)
    p.add_argument("--min_history", type=int, default=5)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--no_hf_text", action="store_true", help="Use hash embedding fallback instead of HuggingFace text encoder")
    p.add_argument("--unfreeze_text", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--w_now", type=float, default=0.5)
    p.add_argument("--w_future", type=float, default=1.0)
    p.add_argument("--w_align", type=float, default=0.01)
    p.add_argument("--w_phys", type=float, default=0.01)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, weights: Dict[str, float]):
    model.eval()
    meters = {k: AverageMeter() for k in ["total", "now", "future", "align", "phys"]}
    now_y, now_pred = [], []
    future_sqerr_sum = 0.0
    future_abserr_sum = 0.0
    future_count = 0
    step_abs = {}
    step_count = {}
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["x"], batch["prompt"], batch["horizon"])
        loss = battery_cycle_forecast_loss(out, batch["y_now"], batch["y_future"], batch["future_mask"], weights)
        n = batch["x"].size(0)
        for k, v in loss.items():
            meters[k].update(float(v.detach() if hasattr(v, "detach") else v), n)

        width = min(out["soh_future"].size(1), batch["y_future"].size(1))
        now_y.append(batch["y_now"].detach().cpu().numpy())
        now_pred.append(out["soh_now"].detach().cpu().numpy())
        pf = out["soh_future"][:, :width].detach()
        yf = batch["y_future"][:, :width].detach()
        mf = batch["future_mask"][:, :width].bool()
        diff = pf[mf] - yf[mf]
        if diff.numel():
            future_sqerr_sum += float((diff**2).sum().cpu())
            future_abserr_sum += float(diff.abs().sum().cpu())
            future_count += int(diff.numel())
        for step in range(width):
            sm = mf[:, step]
            if sm.any():
                err = (pf[sm, step] - yf[sm, step]).abs()
                step_abs[step + 1] = step_abs.get(step + 1, 0.0) + float(err.sum().cpu())
                step_count[step + 1] = step_count.get(step + 1, 0) + int(err.numel())

    y_now = np.concatenate(now_y, axis=0)
    p_now = np.concatenate(now_pred, axis=0)
    metrics = {k: m.avg for k, m in meters.items()}
    metrics.update(
        {
            "now_rmse": float(np.sqrt(np.mean((p_now - y_now) ** 2))),
            "now_mae": float(np.mean(np.abs(p_now - y_now))),
        }
    )
    metrics.update(
        {
            "future_rmse": float(np.sqrt(future_sqerr_sum / max(future_count, 1))),
            "future_mae": float(future_abserr_sum / max(future_count, 1)),
        }
    )
    if step_abs:
        last_step = max(step_abs)
        metrics["future_last_step_mae"] = float(step_abs[last_step] / max(step_count[last_step], 1))
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    records = load_mit_battery_pkls(args.data_dir)
    train_recs, val_recs, test_recs = split_cells(records, seed=args.seed)

    train_ds = MITBatteryCycleDataset(
        train_recs,
        max_horizon=args.max_horizon,
        min_history=args.min_history,
        features=CYCLE_FEATURES,
        scaler=None,
        fit_scaler=True,
        random_horizon=True,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    scaler = train_ds.scaler
    val_ds = MITBatteryCycleDataset(
        val_recs,
        max_horizon=args.max_horizon,
        min_history=args.min_history,
        features=CYCLE_FEATURES,
        scaler=scaler,
        fit_scaler=False,
        random_horizon=False,
        max_cycles=args.max_cycles,
        seed=args.seed + 1,
    )
    test_ds = MITBatteryCycleDataset(
        test_recs,
        max_horizon=args.max_horizon,
        min_history=args.min_history,
        features=CYCLE_FEATURES,
        scaler=scaler,
        fit_scaler=False,
        random_horizon=False,
        max_cycles=args.max_cycles,
        seed=args.seed + 2,
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Dataset split produced an empty set. Reduce max_horizon/min_history or check data files.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_cycle_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cycle_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_cycle_batch)

    cfg = BatteryCycleLLMAssistConfig(
        num_features=len(CYCLE_FEATURES),
        max_horizon=args.max_horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        text_model=args.text_model,
        use_hf_text_encoder=not args.no_hf_text,
        freeze_text=not args.unfreeze_text,
    )
    model = BatteryCycleLLMAssist(cfg).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "now": args.w_now,
        "future": args.w_future,
        "align": args.w_align,
        "phys": args.w_phys,
    }

    save_json(
        {
            "args": vars(args),
            "features": CYCLE_FEATURES,
            "num_cells": len(records),
            "train_cells": [r.cell_id for r in train_recs],
            "val_cells": [r.cell_id for r in val_recs],
            "test_cells": [r.cell_id for r in test_recs],
            "scaler": {k: v.tolist() for k, v in scaler.state_dict().items()},
        },
        out_dir / "run_config.json",
    )

    best_val = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        meters = {k: AverageMeter() for k in ["total", "now", "future", "align", "phys"]}
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, device)
            out = model(batch["x"], batch["prompt"], batch["horizon"])
            loss = battery_cycle_forecast_loss(out, batch["y_now"], batch["y_future"], batch["future_mask"], weights)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = batch["x"].size(0)
            for k, v in loss.items():
                meters[k].update(float(v.detach() if hasattr(v, "detach") else v), n)
            pbar.set_postfix({"loss": meters["total"].avg, "future": meters["future"].avg})

        val_metrics = evaluate(model, val_loader, device, weights)
        print(
            f"epoch={epoch} val_future_rmse={val_metrics['future_rmse']:.5f} "
            f"val_future_mae={val_metrics['future_mae']:.5f} val_now_mae={val_metrics['now_mae']:.5f}"
        )
        improved = (val_metrics["future_rmse"] + val_metrics['now_mae']) < (best_val - args.early_stop_min_delta)
        if improved:
            best_val = (val_metrics["future_rmse"] + val_metrics['now_mae'])
            stale_epochs = 0
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "features": CYCLE_FEATURES,
                "scaler": scaler.state_dict(),
                "args": vars(args),
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"saved best checkpoint to {out_dir / 'best.pt'}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}")
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, weights)
    save_json(test_metrics, out_dir / "test_metrics.json")
    print("test metrics:", test_metrics)


if __name__ == "__main__":
    main()
