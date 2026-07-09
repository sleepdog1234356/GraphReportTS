from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from .data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from .experiment_config import ExperimentConfig, ensure_research_dirs
    from .graph_report_losses import graph_report_loss, regression_metrics
    from .graph_report_model import GraphReportTS, GraphReportTSConfig
    from .utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device
except ImportError:
    from data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from experiment_config import ExperimentConfig, ensure_research_dirs
    from graph_report_losses import graph_report_loss, regression_metrics
    from graph_report_model import GraphReportTS, GraphReportTSConfig
    from utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Train Battery/General GraphReportTS")
    p.add_argument("--variant", choices=["battery", "general"], default="battery")
    p.add_argument("--dataset", type=str, default="mit")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_dir", type=str, default="runs/graph_report_ts")
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--resample_len", type=int, default=128)
    p.add_argument("--delay_dim", type=int, default=8)
    p.add_argument("--delay_lag", type=int, default=1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--patch_stride", type=int, default=4)
    p.add_argument("--topk_edges", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--no_hf_text", action="store_true")
    p.add_argument("--unfreeze_text", action="store_true")
    p.add_argument("--no_report_prompt", action="store_true")
    p.add_argument("--no_cross_modal", action="store_true")
    p.add_argument("--no_ic_dv", action="store_true")
    p.add_argument("--no_hankel_map", action="store_true")
    p.add_argument("--no_derivative_map", action="store_true")
    p.add_argument("--no_dynamic_graph", action="store_true")
    p.add_argument("--no_domain_edges", action="store_true")
    p.add_argument("--separate_heads", action="store_true")
    p.add_argument("--allow_summary_fallback", action="store_true", help="Smoke-test only: synthesize raw curves from summary if MIT raw arrays are missing")
    p.add_argument("--cache_items", action="store_true", help="Enable per-worker sample caching for raw map construction. This can grow RAM quickly with many workers.")
    p.add_argument("--no_cache_items", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--precomputed_cache_dir", type=str, default=None, help="Read deterministic battery graph samples from this cache root when available")
    p.add_argument("--require_precomputed_cache", action="store_true", help="Fail instead of falling back to online battery graph map construction")
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["smooth_l1", "mse", "mae"], default="smooth_l1")
    p.add_argument("--w_align", type=float, default=0.01)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--no_resume", action="store_true", help="Disable automatic resume from last.pt/best.pt in the output directory")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if args.variant == "battery":
        ds_kwargs = dict(
            dataset_name=args.dataset,
            data_root=args.data_root,
            max_horizon=args.pred_len,
            resample_len=args.resample_len,
            delay_dim=args.delay_dim,
            delay_lag=args.delay_lag,
            include_derivatives=not args.no_derivative_map,
            include_hankel=not args.no_hankel_map,
            include_ic_dv=not args.no_ic_dv,
            allow_summary_fallback=args.allow_summary_fallback,
            cache_items=args.cache_items and not args.no_cache_items,
            precomputed_cache_dir=args.precomputed_cache_dir,
            require_precomputed_cache=args.require_precomputed_cache,
            seed=args.seed,
            max_cycles=args.max_cycles,
        )
        train_ds = BatteryRawGraphDataset(split="train", **ds_kwargs)
        val_ds = BatteryRawGraphDataset(split="val", **ds_kwargs)
        test_ds = BatteryRawGraphDataset(split="test", **ds_kwargs)
        collate = collate_graph_report_batch
        output_dim = 1
    else:
        scaler = StandardScalerNP()
        ds_kwargs = dict(
            dataset_name=args.dataset,
            data_root=args.data_root,
            input_len=args.input_len,
            pred_len=args.pred_len,
            resample_len=args.resample_len,
            delay_dim=args.delay_dim,
            delay_lag=args.delay_lag,
            include_derivatives=not args.no_derivative_map,
            include_hankel=not args.no_hankel_map,
        )
        train_ds = GeneralForecastGraphDataset(split="train", scaler=scaler, fit_scaler=True, **ds_kwargs)
        val_ds = GeneralForecastGraphDataset(split="val", scaler=scaler, fit_scaler=False, **ds_kwargs)
        test_ds = GeneralForecastGraphDataset(split="test", scaler=scaler, fit_scaler=False, **ds_kwargs)
        collate = collate_general_graph_batch
        output_dim = len(train_ds.columns)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"prefetch_factor": 2})

    def make_loader(ds, name: str) -> DataLoader:
        kwargs = dict(loader_kwargs)
        if args.num_workers > 0:
            kwargs["persistent_workers"] = name == "train"
        return DataLoader(ds, shuffle=(name == "train"), **kwargs)

    loaders = [make_loader(ds, name) for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]]
    return loaders[0], loaders[1], loaders[2], output_dim


@torch.no_grad()
def evaluate(model, loader, device, weights: Dict[str, float], loss_type: str) -> Dict[str, float]:
    model.eval()
    meters = {k: AverageMeter() for k in ["total", "reg", "align"]}
    mse_sum = mae_sum = count = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["maps"], batch["prompt"], batch["horizon"])
        loss = graph_report_loss(out, batch["y"], batch["mask"], weights, loss_type=loss_type)
        n = batch["maps"].size(0)
        for k, v in loss.items():
            meters[k].update(float(v.detach()), n)
        metrics = regression_metrics(out["pred"], batch["y"], batch["mask"])
        elems = float(batch["mask"].sum().detach().cpu())
        mse_sum += metrics["mse"] * elems
        mae_sum += metrics["mae"] * elems
        count += elems
    mse = mse_sum / max(count, 1.0)
    mae = mae_sum / max(count, 1.0)
    out = {k: m.avg for k, m in meters.items()}
    out.update({"mse": mse, "mae": mae, "rmse": mse**0.5})
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = ExperimentConfig()
    ensure_research_dirs(cfg)
    out_dir = ensure_dir(Path(args.out_dir) / args.variant / args.dataset)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, output_dim = build_loaders(args)
    model_cfg = GraphReportTSConfig(
        variant=args.variant,
        output_dim=output_dim,
        d_model=args.d_model,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        graph_layers=args.graph_layers,
        topk_edges=args.topk_edges,
        dropout=args.dropout,
        text_model=args.text_model,
        use_hf_text_encoder=not args.no_hf_text,
        freeze_text=not args.unfreeze_text,
        use_report_prompt=not args.no_report_prompt,
        use_cross_modal_fusion=not args.no_cross_modal,
        use_domain_edges=not args.no_domain_edges,
        use_dynamic_graph=not args.no_dynamic_graph,
        unified_decoder=not args.separate_heads,
    )
    model = GraphReportTS(model_cfg).to(device)
    with torch.no_grad():
        init_batch = next(iter(train_loader))
        init_batch = to_device(init_batch, device)
        _ = model(init_batch["maps"], init_batch["prompt"], init_batch["horizon"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    weights = {"align": 0.0 if (args.no_report_prompt or args.no_cross_modal) else args.w_align}

    save_json({"args": vars(args), "model_cfg": model_cfg.__dict__, "output_dim": output_dim}, out_dir / "run_config.json")
    best = float("inf")
    stale = 0
    start_epoch = 1
    if not args.no_resume:
        last_path = out_dir / "last.pt"
        best_path = out_dir / "best.pt"
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            val_metrics = ckpt.get("val_metrics", {})
            best = float(ckpt.get("best", val_metrics.get("mse", best)))
            stale = int(ckpt.get("stale", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1 if "epoch" in ckpt else 1
            print(f"resumed from {resume_path} at epoch {start_epoch}; best val_mse={best:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        meters = {k: AverageMeter() for k in ["total", "reg", "align"]}
        pbar = tqdm(train_loader, desc=f"{args.variant}/{args.dataset} epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = to_device(batch, device)
            out = model(batch["maps"], batch["prompt"], batch["horizon"])
            loss = graph_report_loss(out, batch["y"], batch["mask"], weights, loss_type=args.loss)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = batch["maps"].size(0)
            for k, v in loss.items():
                meters[k].update(float(v.detach()), n)
            pbar.set_postfix({"loss": meters["total"].avg})
        val = evaluate(model, val_loader, device, weights, args.loss)
        print(f"epoch={epoch} val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} val_loss={val['total']:.6f}")
        score = val["mse"]
        if score < best - args.early_stop_min_delta:
            best = score
            stale = 0
            torch.save({"model": model.state_dict(), "model_cfg": model_cfg.__dict__, "args": vars(args), "val_metrics": val}, out_dir / "best.pt")
            save_json(val, out_dir / "val_metrics.json")
        else:
            stale += 1
        torch.save(
            {
                "model": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "args": vars(args),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "val_metrics": val,
            },
            out_dir / "last.pt",
        )
        if stale >= args.early_stop_patience:
            print(f"early stopping at epoch {epoch}; best val_mse={best:.6f}")
            break
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, weights, args.loss)
    save_json(test, out_dir / "test_metrics.json")
    print("test metrics:", test)


if __name__ == "__main__":
    main()
